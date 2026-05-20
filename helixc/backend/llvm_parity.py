"""
helixc/backend/llvm_parity.py — x86_64-vs-LLVM parity harness
(v3.0 Phase D, Stage 207).

Stages 200-206 built `helixc/backend/llvm_ir.py`, an ADDITIVE textual-
LLVM-IR backend that consumes the same host IR — a `tir.Module` — as
the incumbent `helixc/backend/x86_64.py`. The v3.0 migration strategy
(docs/V3_PLAN.md) is parity-gated: the LLVM backend may retire
`x86_64.py` (the Stage 221 cutover) only once a harness proves it
produces output observably identical to the incumbent across the
test-program corpus.

This module is that harness. It is built in two chunks, mirroring how
`gpu_ci.py` rolled out its validation (Stage 129 shipped mock
validation; v2.4 item 13 added real-HW dispatch):

  - Chunk A (this chunk) — the MOCK STRUCTURAL parity path. Given a
    `tir.Module`, compile it through BOTH backends and classify the
    outcome (see `ParityVerdict`). It needs no LLVM toolchain and
    always runs. It does NOT claim observable-behaviour parity — nor
    even a full LLVM verify: `MATCH` means the LLVM backend's IR
    passed the toolchain-free `mock_validate_ll` SHAPE check (target
    triple, >=1 `define`, balanced braces, terminated blocks), not
    that the IR is semantically valid. What chunk A proves is the
    weaker but still load-bearing invariant that the LLVM backend, on
    every program the corpus contains, either emits structurally
    shaped IR OR fails closed LOUDLY (`LLVMEmitError`) on an op
    outside its covered subset (a Stage 206-R residual op) — it NEVER
    silently miscompiles.

  - Chunk B — the REAL-EXECUTION parity path. Behind toolchain + WSL
    detection, compile both backends to runnable executables, run
    them, and compare observable behaviour (exit code, stdout,
    stderr); real `llvm-as` (via `llvm_toolchain.dispatch_validate_ll`)
    supersedes the chunk-A shape check. DEFERRED — never FAILED — when
    no toolchain is present, so CI on a tool-less runner stays green
    (the `gpu_ci` discipline). The `ParityResult` type already carries
    the real-execution fields chunk B fills in, so chunk B needs no
    type change.

License: Apache 2.0
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..ir import tir
from .llvm_ir import LLVMEmitError, emit_module, mock_validate_ll
from .x86_64 import compile_module_to_elf


# The program entry point the harness compares on. The x86_64 backend
# makes this function the ELF entry; the parity gate compares runnable
# PROGRAMS, and a program needs an entry point. A module without it is
# a degenerate corpus entry — classified ERROR up front by
# `check_parity`, before either backend is invoked.
_ENTRY_FN = "main"


class ParityVerdict(Enum):
    """The classification of one program's structural parity check.

    Chunk A is a STRUCTURAL check — it compares whether the two
    backends accept a module, not (yet) what the compiled programs do
    when run; observable-behaviour parity is chunk B's real-execution
    path. The four outcomes:

    - MATCH — both backends accept the module and the LLVM backend's
      emitted IR passes the toolchain-free `mock_validate_ll` SHAPE
      check (a structural sanity check — target triple, >=1 `define`,
      balanced braces, terminated blocks — NOT a full LLVM verify;
      real `llvm-as` validation is chunk B). The program is inside the
      LLVM backend's covered op subset and structural parity holds at
      the shape level.
    - UNCOVERED — the x86_64 backend accepts the module but the LLVM
      backend fails closed (`LLVMEmitError`): the program uses an op
      outside the LLVM backend's covered subset (a Stage 206-R
      residual op). This is NOT a failure — it is the fail-closed
      guarantee working exactly as designed (docs/V3_PLAN.md Stage 206
      closure: a 206-R op is rejected loudly, never miscompiled, so a
      program using one is simply outside the gate's covered subset).
    - MISMATCH — a genuine parity defect: the x86_64 backend accepts
      the module but the LLVM backend emits IR that FAILS the
      `mock_validate_ll` shape check. The LLVM backend produced
      structurally malformed IR for a covered program — a real bug.
    - ERROR — the harness could not reach a parity verdict: the input
      is degenerate (no `main`), or the x86_64 backend (the full-
      coverage incumbent) failed to compile the module, or the LLVM
      backend / the shape-checker raised a non-`LLVMEmitError`
      exception (a crash, not a clean fail-closed). Each is a defect
      the gate must surface, never swallow.
    """
    MATCH = "match"
    UNCOVERED = "uncovered"
    MISMATCH = "mismatch"
    ERROR = "error"


# The verdicts the parity gate must FAIL on — a real defect the harness
# surfaced. MATCH and UNCOVERED are both acceptable outcomes (UNCOVERED
# is the LLVM backend's fail-closed guarantee working as designed).
# `_check_parity_verdict_coverage` (below) asserts these two sets
# PARTITION ParityVerdict — a new verdict added without updating the
# partition fails loudly at module load instead of silently defaulting
# to "not a defect" (a silent-failure surface). Mirrors
# gpu_ci._check_gpu_ci_drift / llvm_toolchain._check_llvm_toolchain_drift.
_PARITY_DEFECT_VERDICTS: frozenset[ParityVerdict] = frozenset({
    ParityVerdict.MISMATCH, ParityVerdict.ERROR,
})
_PARITY_OK_VERDICTS: frozenset[ParityVerdict] = frozenset({
    ParityVerdict.MATCH, ParityVerdict.UNCOVERED,
})


def _check_parity_verdict_coverage() -> None:
    """Module-load guard: every `ParityVerdict` is classified as either
    a parity defect or an OK outcome, and the two sets are disjoint. A
    new verdict added without updating the partition fails loudly here
    rather than silently defaulting to 'not a defect' in
    `is_parity_defect`.

    This guards CLASSIFICATION, not REACHABILITY — that every verdict
    `ParityResult.verdict()` can return is exercised is pinned
    separately by the `test_verdict_*` tests in test_llvm_parity.py."""
    classified = _PARITY_DEFECT_VERDICTS | _PARITY_OK_VERDICTS
    if classified != set(ParityVerdict):
        raise AssertionError(
            f"helixc.backend.llvm_parity: the defect / OK verdict sets "
            f"classify {classified} but ParityVerdict has "
            f"{set(ParityVerdict)} — every verdict must be classified, "
            f"else a new one silently defaults to 'not a defect'")
    overlap = _PARITY_DEFECT_VERDICTS & _PARITY_OK_VERDICTS
    if overlap:
        raise AssertionError(
            f"helixc.backend.llvm_parity: verdict(s) {overlap} are "
            f"classified as BOTH a defect and an OK outcome — the two "
            f"sets must be disjoint")


_check_parity_verdict_coverage()


@dataclass(frozen=True)
class ParityResult:
    """Outcome of a structural parity check for one program.

    Stores the raw per-backend facts; `verdict()` DERIVES the
    classification from them (the `gpu_ci.ValidationResult` /
    `llvm_toolchain.LLVMDispatchResult` pattern — a stored verdict
    could drift from the facts). Frozen + tuple-backed +
    `__post_init__`-guarded so the silent-failure / illegal field
    shapes are unrepresentable.

    Facts:
    - `program` — the program's name / id (for diagnostics); never
      blank.
    - `x86_compiled` — `x86_64.compile_module_to_elf` produced an ELF.
    - `llvm_emitted` — `llvm_ir.emit_module` produced IR AND
      `mock_validate_ll` then ran to completion on it. False when emit
      raised, or fail-closed, or the shape-checker itself crashed (see
      `check_parity`).
    - `llvm_failed_closed` — `emit_module` raised `LLVMEmitError` (the
      module uses an op outside the LLVM backend's covered subset).
      Mutually exclusive with `llvm_emitted`.
    - `llvm_mock_clean` — the emitted IR passed `mock_validate_ll`.
      True only when `llvm_emitted` is True.
    - `detail` — human-readable diagnostics; every entry is non-blank.
      Empty iff the verdict is MATCH; non-empty for every other
      verdict (it carries the reason). `__post_init__` enforces the
      presence of a reason, not its wording — the verdict is derived
      from the facts, not from the detail text.

    Real-execution fields — filled by chunk B's toolchain-gated path;
    chunk A always leaves them at their defaults (not attempted):
    - `real_attempted` — a real run-and-compare was dispatched.
    - `real_passed` — its verdict (None when not attempted).
    - `real_findings` — its diagnostics (non-empty on a real failure).
    """
    program: str
    x86_compiled: bool
    llvm_emitted: bool
    llvm_failed_closed: bool
    llvm_mock_clean: bool
    detail: tuple[str, ...]
    real_attempted: bool = False
    real_passed: Optional[bool] = None
    real_findings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # --- program identity ---
        # A result must name the program it classifies, else a corpus
        # walk produces unattributable diagnostics.
        if not self.program.strip():
            raise ValueError(
                "ParityResult: program is empty/blank — every result "
                "must name the program it classifies")
        # --- fact-field consistency ---
        # emit_module either returns IR or raises — never both.
        if self.llvm_emitted and self.llvm_failed_closed:
            raise ValueError(
                "ParityResult: llvm_emitted and llvm_failed_closed are "
                "both True — emit_module either returns IR or raises "
                "LLVMEmitError, never both")
        # A fail-closed run emitted no IR, so there is none to validate.
        if self.llvm_failed_closed and self.llvm_mock_clean:
            raise ValueError(
                "ParityResult: llvm_failed_closed and llvm_mock_clean "
                "are both True — a fail-closed run emitted no IR to "
                "shape-check")
        # mock-validation is only meaningful on emitted IR — a result
        # that never emitted IR cannot be 'mock clean'.
        if self.llvm_mock_clean and not self.llvm_emitted:
            raise ValueError(
                "ParityResult: llvm_mock_clean is True but llvm_emitted "
                "is False — there is no emitted IR to have validated")
        # --- real-execution field consistency (the gpu_ci invariant) ---
        if not self.real_attempted:
            if self.real_passed is not None:
                raise ValueError(
                    f"ParityResult: real_attempted=False but "
                    f"real_passed={self.real_passed!r} — illegal")
            if self.real_findings:
                raise ValueError(
                    "ParityResult: real_attempted=False but "
                    "real_findings has entries — illegal")
        else:
            if self.real_passed is None:
                raise ValueError(
                    "ParityResult: real_attempted=True but real_passed "
                    "is None — a real run must reach a concrete verdict")
            if self.real_passed is False and not self.real_findings:
                raise ValueError(
                    "ParityResult: real_passed=False but real_findings "
                    "is empty — a real-execution failure must carry a "
                    "diagnostic")
        # --- diagnostic entries must be real text ---
        # A blank ('' / whitespace) or non-str entry is a reason-shaped
        # object carrying no reason — it would satisfy the "non-MATCH
        # must carry a detail" check below while saying nothing.
        for label, entries in (("detail", self.detail),
                               ("real_findings", self.real_findings)):
            for entry in entries:
                if not isinstance(entry, str) or not entry.strip():
                    raise ValueError(
                        f"ParityResult: {label} contains a blank or "
                        f"non-str entry ({entry!r}) — every diagnostic "
                        f"must carry actual text")
        # --- detail must explain every non-MATCH verdict ---
        # verdict() derives only from the fact fields; the
        # fact-consistency checks above run FIRST, so it never sees a
        # contradictory llvm-fact pair. (x86_compiled needs no check —
        # either bool is a legal x86_64 outcome.)
        is_match = self.verdict() is ParityVerdict.MATCH
        if is_match and self.detail:
            raise ValueError(
                f"ParityResult: verdict is MATCH but detail is "
                f"non-empty ({self.detail!r}) — a clean match carries "
                f"no diagnostic")
        if not is_match and not self.detail:
            raise ValueError(
                f"ParityResult: verdict is {self.verdict().name} but "
                f"detail is empty — a non-MATCH verdict must carry a "
                f"reason")

    def verdict(self) -> ParityVerdict:
        """Derive the parity classification from the stored facts.

        The x86_64 backend is the full-coverage incumbent: if it could
        not compile the module the harness has no baseline, so the
        result is ERROR regardless of what the LLVM backend did. Given
        a working x86_64 baseline, a clean `LLVMEmitError` is the
        designed UNCOVERED outcome; a result with no emitted IR and no
        fail-closed (an emit crash, or the shape-checker itself
        crashing) is an ERROR; emitted-but-shape-malformed IR is a
        MISMATCH; emitted-and-well-shaped IR is a MATCH."""
        if not self.x86_compiled:
            return ParityVerdict.ERROR
        if self.llvm_failed_closed:
            return ParityVerdict.UNCOVERED
        if not self.llvm_emitted:
            return ParityVerdict.ERROR
        if not self.llvm_mock_clean:
            return ParityVerdict.MISMATCH
        return ParityVerdict.MATCH

    def is_parity_defect(self) -> bool:
        """True iff the verdict is one the parity gate must fail on
        (MISMATCH or ERROR). MATCH and UNCOVERED are both acceptable —
        UNCOVERED is the LLVM backend's fail-closed guarantee working
        as designed."""
        return self.verdict() in _PARITY_DEFECT_VERDICTS


def check_parity(module: tir.Module, program: str) -> ParityResult:
    """Structurally compare the x86_64 and LLVM backends on one module.

    The chunk-A MOCK path of the Stage 207 parity gate: it needs no
    LLVM toolchain and always runs. It compiles `module` through BOTH
    backends and classifies the outcome (see `ParityVerdict`); it does
    NOT run the compiled programs — observable-behaviour parity is
    chunk B's real-execution path, and the returned `ParityResult`
    leaves its real-execution fields not-attempted.

    `module` must define `main` (the program entry the x86_64 backend
    makes the ELF entry point). A module without one — including an
    empty module — is not a runnable program and is classified ERROR
    UP FRONT, before either backend is invoked: a degenerate corpus
    entry is neither backend's fault, so it is not inferred from a
    backend's exception.

    The caller's `module` is never mutated — each backend is given its
    OWN deep copy, so the harness is side-effect-free and the two
    backends are guaranteed identical input regardless of whether a
    backend mutates the module it is handed. (The two `copy.deepcopy`
    calls below are deliberately SEPARATE — one shared copy would let a
    mutating first backend corrupt the second backend's input. Do not
    hoist them.)

    Neither backend's failure escapes as an exception: a backend that
    raises is captured into a structured `ParityResult` (an ERROR, or
    — for the LLVM backend's deliberate `LLVMEmitError` — an
    UNCOVERED). That is the harness's whole contract: turn what each
    backend does, including crashing, into a verdict — never let a
    backend crash abort the gate's corpus walk."""
    # --- degenerate input: no program entry point ---
    # The parity gate compares runnable programs; a module with no
    # `main` (an empty module is the limiting case) is not one. Classify
    # it ERROR here, with an honest diagnostic — do not let it reach the
    # backends, where x86_64's missing-entry ValueError would be
    # string-matched into a misleading "x86_64 backend failed" detail
    # and the LLVM backend would emit `define`-less IR that
    # mock_validate_ll flags as "no define", a false MISMATCH shape.
    if _ENTRY_FN not in module.functions:
        return ParityResult(
            program=program, x86_compiled=False, llvm_emitted=False,
            llvm_failed_closed=False, llvm_mock_clean=False,
            detail=(f"module {program!r} defines no {_ENTRY_FN!r} entry "
                    f"function — not a runnable program, nothing to "
                    f"compare",))

    detail: list[str] = []

    # --- x86_64 backend: the full-coverage incumbent / baseline. ---
    x86_compiled = True
    try:
        compile_module_to_elf(copy.deepcopy(module))
    except Exception as exc:
        # A broad `except Exception` is correct here (cf. the
        # gpu_ci._run_tool real-HW dispatch discipline): the harness
        # must convert ANY backend failure into a verdict, never let an
        # untyped traceback abort the corpus walk. The failure is
        # surfaced LOUDLY as an ERROR with the diagnostic below — it is
        # captured, never swallowed. (KeyboardInterrupt / SystemExit,
        # being BaseException, still propagate.)
        x86_compiled = False
        detail.append(f"x86_64 backend failed to compile {program!r}: "
                      f"{type(exc).__name__}: {exc}")

    # --- LLVM backend: the additive Phase-D backend, partial coverage. ---
    # `llvm_emitted` becomes True only when emit_module produced IR AND
    # mock_validate_ll then ran to completion — an emit crash, a
    # fail-closed, or a shape-checker crash each leave it False (see the
    # branches below), so verdict() routes them correctly.
    llvm_emitted = False
    llvm_failed_closed = False
    llvm_mock_clean = False
    try:
        ll_text = emit_module(copy.deepcopy(module))
    except LLVMEmitError as exc:
        # The DESIGNED fail-closed path: the module uses an op outside
        # the LLVM backend's covered subset (a Stage 206-R residual
        # op). Loud, not silent — classified UNCOVERED, not a defect.
        llvm_failed_closed = True
        detail.append(
            f"LLVM backend does not yet cover an op in {program!r} "
            f"(fail-closed, outside the covered subset): {exc}")
    except Exception as exc:
        # A non-LLVMEmitError exception is a CRASH, not a clean
        # fail-closed — a real bug in the LLVM backend. Captured as an
        # ERROR (see the x86_64 catch above for why the broad catch is
        # correct).
        detail.append(
            f"LLVM backend crashed emitting {program!r}: "
            f"{type(exc).__name__}: {exc}")
    else:
        # emit_module produced IR — now shape-check it. mock_validate_ll
        # returns a list of non-blank problem strings (empty ==
        # well-shaped), so any `detail` entry built from it below
        # satisfies the ParityResult non-blank-entry invariant.
        try:
            mock_findings = mock_validate_ll(ll_text)
        except Exception as exc:
            # The shape-checker itself crashed. The emitted IR is NOT
            # thereby proven malformed — the fault is the validator,
            # not the LLVM emitter — so this is an ERROR, never a
            # MISMATCH. `llvm_emitted` is left False so verdict() takes
            # the `not llvm_emitted -> ERROR` arm, not the
            # `not llvm_mock_clean -> MISMATCH` arm.
            detail.append(
                f"mock_validate_ll crashed on {program!r}'s emitted "
                f"IR: {type(exc).__name__}: {exc}")
        else:
            llvm_emitted = True
            llvm_mock_clean = not mock_findings
            if mock_findings:
                detail.append(
                    f"LLVM backend emitted IR for {program!r} that "
                    f"fails the mock_validate_ll shape check: "
                    + "; ".join(mock_findings))

    # Every branch above builds a fact-set the ParityResult invariants
    # accept — emitted / failed_closed are mutually exclusive by
    # control flow, mock_clean is set only alongside emitted, and every
    # non-MATCH outcome appended a non-blank detail — so this
    # constructor cannot raise. A future branch that breaks that must
    # be re-checked against __post_init__.
    return ParityResult(
        program=program,
        x86_compiled=x86_compiled,
        llvm_emitted=llvm_emitted,
        llvm_failed_closed=llvm_failed_closed,
        llvm_mock_clean=llvm_mock_clean,
        detail=tuple(detail),
    )
