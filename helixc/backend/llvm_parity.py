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

This module is that harness. It is built in three chunks, mirroring
how `gpu_ci.py` rolled out its validation (Stage 129 shipped mock
validation; v2.4 item 13 added real-HW dispatch):

  - Chunk A — the MOCK STRUCTURAL parity path. Given a `tir.Module`,
    compile it through BOTH backends and classify the outcome (see
    `ParityVerdict`). It needs no LLVM toolchain and always runs. It
    does NOT claim observable-behaviour parity — nor even a full LLVM
    verify: `MATCH` means the LLVM backend's IR passed the
    toolchain-free `mock_validate_ll` SHAPE check (target triple,
    >=1 `define`, balanced braces, terminated blocks), not that the IR
    is semantically valid. What chunk A proves is the weaker but still
    load-bearing invariant that the LLVM backend, on every program the
    corpus contains, either emits structurally shaped IR OR fails
    closed LOUDLY (`LLVMEmitError`) on an op outside its covered
    subset (a Stage 206-R residual op) — it NEVER silently
    miscompiles.

  - Chunk B — the curated source-program CORPUS + the mock-path
    parity GATE. `check_parity_source` runs a Helix source string
    through the frontend pipeline (parse -> lower) to a `tir.Module`
    and hands it to `check_parity`; `PARITY_CORPUS` is a curated set
    of small deterministic Helix programs exercising the LLVM
    backend's covered op surface; `run_parity_corpus` walks them. The
    Stage 207 mock-path gate (in test_llvm_parity.py) asserts every
    corpus program is MATCH — real Helix programs structurally agree
    across both backends.

  - Chunk C — the real-execution RESULT MODEL + toolchain detection.
    `RealParityStatus` (NOT_RUN / DEFERRED / PASS / FAIL) and
    `ParityResult.real_status()` model the observable-behaviour
    outcome; `detect_real_exec_support` reports whether this machine
    can run the comparison (WSL present, and `clang` inside WSL to
    compile the LLVM backend's IR to a runnable executable). It also
    relaxes a chunk-A `ParityResult` invariant that forbade the
    DEFERRED state, restoring fidelity with the
    gpu_ci.ValidationResult real-outcome model.

  - Chunk D — the program-run SUBSTRATE. `_run_x86_program` and
    `_run_llvm_program` build a backend's output to a runnable Linux
    executable and run it under WSL, capturing observable behaviour
    (exit code, stdout, stderr) into a `_ProgramRun`. Every build /
    run failure is captured into the result, never raised.

  - Chunk E — the real-execution COMPARISON + the `attempt_real`
    wiring. Compares the two `_ProgramRun`s and fills `ParityResult`'s
    real-* fields; behind the chunk-C detection it is DEFERRED — never
    FAILED — when the toolchain is absent, so CI on a tool-less runner
    stays green (the `gpu_ci` real-HW dispatch discipline). Chunk E
    closes Stage 207.

License: Apache 2.0
"""

from __future__ import annotations

import copy
import dataclasses
import os
import shutil
import subprocess
import tempfile
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
    when run; observable-behaviour parity is chunk C's real-execution
    path. The four outcomes:

    - MATCH — both backends accept the module and the LLVM backend's
      emitted IR passes the toolchain-free `mock_validate_ll` SHAPE
      check (a structural sanity check — target triple, >=1 `define`,
      balanced braces, terminated blocks — NOT a full LLVM verify;
      real `llvm-as` validation is chunk C). The program is inside the
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


class RealParityStatus(Enum):
    """The real-execution (observable-behaviour) parity outcome — the
    chunk C/D counterpart to the structural `ParityVerdict`. Derived
    from `ParityResult`'s real-* fields by `real_status()`.

    - NOT_RUN — no real run was requested; the result is mock-only
      (the chunk-A/B default — `check_parity` without a real run).
    - DEFERRED — a real run WAS requested but could not be carried out
      because the toolchain / runtime to run it is absent (no WSL, or
      no `clang` inside WSL to compile the LLVM backend's IR). NOT a
      failure — CI on a tool-less runner stays green, mirroring the
      `gpu_ci` real-HW dispatch discipline.
    - PASS — both backends were compiled, run, and produced identical
      observable behaviour (exit code, stdout, stderr).
    - FAIL — both backends were run and their observable behaviour
      differed, or a backend's executable could not be built or ran
      abnormally.
    """
    NOT_RUN = "not_run"
    DEFERRED = "deferred"
    PASS = "pass"
    FAIL = "fail"


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

    Real-execution fields — filled by chunk D's dispatch; the mock
    path (chunks A/B) always leaves them at their defaults. They
    encode the 4-state real outcome `real_status()` derives (see
    `RealParityStatus`):
    - `real_attempted` — a real run-and-compare was requested.
    - `real_passed` — True (PASS) / False (FAIL) / None. None with
      `real_attempted=False` is NOT_RUN; None with
      `real_attempted=True` is DEFERRED (requested, no toolchain).
    - `real_findings` — its diagnostics; non-empty for a DEFERRED
      (why it was deferred) or a FAIL (how the behaviour differed).
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
        # --- real-execution field consistency ---
        # The real outcome is a 4-state model (see `real_status` /
        # `RealParityStatus`): NOT_RUN, DEFERRED, PASS, FAIL, encoded
        # by (real_attempted, real_passed). chunk C relaxed an earlier
        # invariant that forbade `real_attempted=True, real_passed=None`
        # — that pair IS legal: it is the DEFERRED state (a real run
        # was requested but no toolchain could run it), mirroring the
        # gpu_ci.ValidationResult real-outcome model.
        if not self.real_attempted:
            # NOT_RUN — no real run was requested or reached.
            if self.real_passed is not None:
                raise ValueError(
                    f"ParityResult: real_attempted=False but "
                    f"real_passed={self.real_passed!r} — illegal")
            if self.real_findings:
                raise ValueError(
                    "ParityResult: real_attempted=False but "
                    "real_findings has entries — illegal")
        else:
            # real_passed: None -> DEFERRED, True -> PASS, False ->
            # FAIL. DEFERRED and FAIL must each carry a diagnostic (why
            # it was deferred / how the behaviour differed); a PASS may
            # carry advisory notes but need not.
            if self.real_passed is None and not self.real_findings:
                raise ValueError(
                    "ParityResult: real_attempted=True with "
                    "real_passed=None (DEFERRED) but real_findings is "
                    "empty — a deferred real run must record why")
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

    def real_status(self) -> RealParityStatus:
        """The real-execution (observable-behaviour) outcome, derived
        from the real-* fields — the chunk C/D counterpart to the
        structural `verdict()`. NOT_RUN when no real run was requested
        (the chunk-A/B mock-only default); DEFERRED when one was
        requested but no toolchain could run it; PASS / FAIL when both
        backends were run and their observable behaviour was identical
        / differed. Independent of `verdict()` — a structural MATCH may
        still be a real FAIL, which is exactly what chunk D exists to
        catch."""
        if not self.real_attempted:
            return RealParityStatus.NOT_RUN
        if self.real_passed is None:
            return RealParityStatus.DEFERRED
        if self.real_passed:
            return RealParityStatus.PASS
        return RealParityStatus.FAIL


def check_parity(module: tir.Module, program: str, *,
                 attempt_real: bool = False) -> ParityResult:
    """Compare the x86_64 and LLVM backends on one module.

    Compiles `module` through BOTH backends and classifies the
    STRUCTURAL outcome (see `ParityVerdict`); this mock classification
    needs no LLVM toolchain and always runs.

    With `attempt_real=True` AND a structural MATCH AND the toolchain
    present (`detect_real_exec_support().can_run_real()`), it ALSO
    builds both backends to runnable executables, runs them, and
    compares observable behaviour (exit code, stdout, stderr) — filling
    the result's real-* fields (`real_status()` PASS / FAIL). When the
    toolchain is absent the real comparison is DEFERRED, never FAILED,
    so CI on a tool-less runner stays green. With `attempt_real=False`
    (the default) the real-* fields stay NOT_RUN — the chunk-A/B
    mock-only behaviour, unchanged.

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
    result = ParityResult(
        program=program,
        x86_compiled=x86_compiled,
        llvm_emitted=llvm_emitted,
        llvm_failed_closed=llvm_failed_closed,
        llvm_mock_clean=llvm_mock_clean,
        detail=tuple(detail),
    )
    # Real-execution parity (chunk E) — only when requested AND the
    # structural verdict is MATCH: a non-MATCH program has no pair of
    # comparable runnable executables (UNCOVERED / MISMATCH / ERROR
    # means one backend produced nothing runnable). `_attempt_real_
    # parity` fills the real-* fields; without `attempt_real` they stay
    # NOT_RUN.
    if attempt_real and result.verdict() is ParityVerdict.MATCH:
        result = _attempt_real_parity(module, program, result)
    return result


# ==========================================================================
# Chunk B — the source-program corpus + the mock-path parity gate
# ==========================================================================

# A curated corpus of small, deterministic Helix programs exercising the
# LLVM backend's covered op surface (Phase D Stages 200-206): integer
# arithmetic, the bitwise ops, comparisons / select, control flow
# (if / nested if / while), local + mutable variables, stack arrays,
# direct calls (incl. recursion), the unsigned dtypes (incl. the
# signedness-sensitive udiv / unsigned-icmp paths), and bool. Every
# entry is checked with `include_stdlib=False` and is curated to be
# MATCH — the Stage 207 mock-path gate (test_llvm_parity.py) asserts
# exactly that, so a covered op regressing to UNCOVERED / MISMATCH
# breaks the gate. (The stdlib pulls in float-typed transcendentals the
# LLVM backend does not yet cover, so an `include_stdlib=True` build of
# any program is UNCOVERED — outside this gate's covered subset.)
PARITY_CORPUS: tuple[tuple[str, str], ...] = (
    ("const_return", "fn main() -> i32 { 42 }"),
    ("add", "fn main() -> i32 { 17 + 25 }"),
    ("sub", "fn main() -> i32 { 100 - 58 }"),
    ("mul", "fn main() -> i32 { 6 * 7 }"),
    ("div", "fn main() -> i32 { 84 / 2 }"),
    ("mod", "fn main() -> i32 { 142 % 100 }"),
    ("neg", "fn main() -> i32 { let x: i32 = 42; 0 - x }"),
    ("arith_mixed", "fn main() -> i32 { 2 * 10 + 22 }"),
    ("bitwise_and_or", "fn main() -> i32 { (240 | 10) & 255 }"),
    ("bitwise_xor", "fn main() -> i32 { 255 ^ 213 }"),
    ("shift_left", "fn main() -> i32 { 21 << 1 }"),
    ("shift_right", "fn main() -> i32 { 168 >> 2 }"),
    ("compare_eq", "fn main() -> i32 { if 3 == 3 { 42 } else { 0 } }"),
    ("compare_ne", "fn main() -> i32 { if 3 != 4 { 42 } else { 0 } }"),
    ("compare_le", "fn main() -> i32 { if 3 <= 3 { 42 } else { 0 } }"),
    ("if_else", "fn main() -> i32 { if 5 > 3 { 42 } else { 0 } }"),
    ("nested_if",
     "fn main() -> i32 { let x: i32 = 7; "
     "if x > 10 { 1 } else { if x > 5 { 42 } else { 0 } } }"),
    ("local_bindings",
     "fn main() -> i32 { let x: i32 = 10; let y: i32 = 32; x + y }"),
    ("mutable_local",
     "fn main() -> i32 { let mut a: i32 = 0; a = 42; a }"),
    ("while_loop",
     "fn main() -> i32 { let mut s: i32 = 0; let mut i: i32 = 0; "
     "while i < 7 { s = s + 6; i = i + 1; } s }"),
    ("direct_call",
     "fn helper() -> i32 { 42 } fn main() -> i32 { helper() }"),
    ("call_with_args",
     "fn add2(a: i32, b: i32) -> i32 { a + b } "
     "fn main() -> i32 { add2(20, 22) }"),
    ("recursion",
     "fn fact(n: i32) -> i32 { if n <= 1 { 1 } "
     "else { n * fact(n - 1) } } fn main() -> i32 { fact(5) }"),
    ("array_index",
     "fn main() -> i32 { let a: [i32; 3] = [10, 20, 12]; "
     "a[0] + a[1] + a[2] }"),
    ("unsigned_type", "fn main() -> u32 { let x: u32 = 42; x }"),
    ("unsigned_div",
     "fn main() -> u32 { let a: u32 = 200; let b: u32 = 4; a / b }"),
    ("unsigned_compare",
     "fn main() -> i32 { let a: u32 = 9; let b: u32 = 3; "
     "if a > b { 42 } else { 0 } }"),
    ("bool_function",
     "fn is_pos(n: i32) -> bool { n > 0 } "
     "fn main() -> i32 { if is_pos(5) { 42 } else { 0 } }"),
)


def _check_parity_corpus() -> None:
    """Module-load guard: every PARITY_CORPUS entry has a unique,
    non-blank name and a non-blank source. A duplicate name would make
    a corpus-walk diagnostic ambiguous about which program it
    describes. Mirrors gpu_ci._check_gpu_ci_drift."""
    names = [name for name, _ in PARITY_CORPUS]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise AssertionError(
            f"helixc.backend.llvm_parity: PARITY_CORPUS has duplicate "
            f"program name(s): {duplicates}")
    for name, source in PARITY_CORPUS:
        if not name.strip() or not source.strip():
            raise AssertionError(
                f"helixc.backend.llvm_parity: PARITY_CORPUS entry "
                f"{name!r} has a blank name or source")


_check_parity_corpus()


def check_parity_source(source: str, program: str, *,
                        include_stdlib: bool = False,
                        attempt_real: bool = False) -> ParityResult:
    """Parity-check one Helix SOURCE program.

    Runs `source` through the frontend pipeline — parse, flatten
    modules / impls, monomorphize, grad-pass, lower — to a `tir.Module`,
    then hands it to `check_parity` (see there for the verdict model
    and the `attempt_real` real-execution path, which is passed
    through). The pipeline mirrors the one `helixc/tests/test_codegen.
    py`'s `compile_and_run` uses for the x86_64 path, minus the
    optional optimizer passes — the parity check is on the unoptimized
    module.

    `include_stdlib` defaults to False: a minimal program needs no
    stdlib, and an `include_stdlib=True` build pulls in float-typed
    stdlib functions the LLVM backend does not yet cover, which would
    make the whole module UNCOVERED.

    A frontend failure (a parse / type / lower error) means the source
    never became a `tir.Module` — a degenerate corpus entry, not a
    parity question. It is captured as an ERROR `ParityResult`, never
    re-raised, consistent with `check_parity`'s contract that no
    failure escapes the harness as a traceback."""
    # Lazy frontend import: keeps the module-load import graph of this
    # backend module free of a backend->frontend dependency — the
    # frontend is pulled in only when a source program is actually
    # checked (check_parity itself, the chunk-A core, needs none of it).
    try:
        from ..frontend.parser import parse
        from ..frontend.flatten_modules import flatten_modules
        from ..frontend.flatten_impls import flatten_impls
        from ..frontend.monomorphize import monomorphize
        from ..frontend.grad_pass import grad_pass
        from ..ir.lower_ast import lower

        prog = parse(source, include_stdlib=include_stdlib)
        flatten_modules(prog)
        flatten_impls(prog)
        monomorphize(prog)
        grad_pass(prog)
        module = lower(prog)
    except Exception as exc:
        # The frontend could not produce a tir.Module. A broad catch is
        # correct (cf. check_parity): the harness converts the failure
        # into a verdict — a loud ERROR with the diagnostic — never an
        # escaping traceback. (KeyboardInterrupt / SystemExit, being
        # BaseException, still propagate.)
        return ParityResult(
            program=program, x86_compiled=False, llvm_emitted=False,
            llvm_failed_closed=False, llvm_mock_clean=False,
            detail=(f"frontend pipeline failed for {program!r}: "
                    f"{type(exc).__name__}: {exc}",))
    return check_parity(module, program, attempt_real=attempt_real)


def run_parity_corpus(*, attempt_real: bool = False) -> list[ParityResult]:
    """Run every program in `PARITY_CORPUS` through `check_parity_source`
    and return the results in corpus order — one `ParityResult` per
    entry. The Stage 207 mock-path parity gate (test_llvm_parity.py)
    asserts every result is MATCH: the corpus is curated to the LLVM
    backend's covered op surface, so a covered op regressing to
    UNCOVERED or MISMATCH breaks the gate.

    `attempt_real` (passed through to `check_parity`) additionally runs
    the real-execution comparison on each program — PASS / FAIL where
    the toolchain is present, DEFERRED where it is absent."""
    return [check_parity_source(source, name, attempt_real=attempt_real)
            for name, source in PARITY_CORPUS]


# ==========================================================================
# Chunk C — real-execution toolchain / runtime detection
# ==========================================================================

# Wall-clock cap on the WSL `clang`-detection probe. `command -v` is
# near-instant; the cap only guards against a hung `wsl` invocation.
_REAL_EXEC_PROBE_TIMEOUT_S = 20


@dataclass(frozen=True)
class RealExecSupport:
    """Whether this machine can run the Stage 207 real-execution parity
    comparison (chunk D's dispatch).

    The real path runs Linux executables: the x86_64 backend's
    freestanding ELF, and the LLVM backend's IR compiled + linked to an
    executable. Both run under WSL on a Windows host; the LLVM side
    additionally needs `clang` INSIDE WSL to turn the emitted `.ll`
    into a runnable executable — a Windows-PATH `clang` (which
    `shutil.which` would find) cannot produce a Linux executable, so
    the relevant compiler is probed inside WSL, not on the host PATH.

    Frozen + `__post_init__`-guarded (the `gpu_ci` result-type
    discipline). `can_run_real()` is the single predicate chunk D's
    dispatch gates on; when it is False the real comparison is
    DEFERRED, never FAILED."""
    # `wsl_available` is True when the `wsl` launcher is on the host
    # PATH — NOT a guarantee a distro is installed or WSL is otherwise
    # functional. A launcher-present-but-distro-less machine still
    # fails safe: `_probe_wsl_clang`'s WSL call then exits non-zero ->
    # `wsl_clang` is None -> `can_run_real()` is False -> DEFERRED.
    wsl_available: bool
    wsl_clang: Optional[str]
    detail: tuple[str, ...]

    def __post_init__(self) -> None:
        # `clang` inside WSL is unreachable if WSL itself is not.
        if self.wsl_clang is not None and not self.wsl_available:
            raise ValueError(
                "RealExecSupport: wsl_clang is set but wsl_available "
                "is False — clang inside WSL implies WSL is available")
        # A support result must always explain what it found — a
        # tool-less machine especially must say WHY the real path is
        # unavailable, so a DEFERRED is never silent.
        if not self.detail:
            raise ValueError(
                "RealExecSupport: detail is empty — the result must "
                "explain what is or is not available")
        for entry in self.detail:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    f"RealExecSupport: detail has a blank or non-str "
                    f"entry ({entry!r}) — every line must carry text")

    def can_run_real(self) -> bool:
        """True iff a real-execution parity run is possible here — WSL
        is available AND `clang` was found inside it. When False, chunk
        D's dispatch records the comparison as DEFERRED (never a hard
        failure), mirroring `gpu_ci`'s tool-absent path."""
        return self.wsl_available and self.wsl_clang is not None


def _probe_wsl_clang(wsl: str) -> Optional[str]:
    """Probe for `clang` inside WSL — return its resolved path, or None
    when it is absent or WSL is unusable.

    Applies the `gpu_ci` subprocess discipline: a `TimeoutExpired` or
    `OSError` (a `wsl` that hangs, vanished, or refuses to spawn)
    becomes a None result, never an uncaught traceback — a probe that
    cannot answer is treated exactly as 'clang not found', so detection
    fails safe to DEFERRED."""
    try:
        proc = subprocess.run(
            [wsl, "--", "bash", "-lc", "command -v clang"],
            capture_output=True, text=True, errors="replace",
            timeout=_REAL_EXEC_PROBE_TIMEOUT_S)
    except (subprocess.TimeoutExpired, OSError):
        return None
    if proc.returncode != 0:
        return None
    # `command -v clang` prints the resolved path as its FINAL stdout
    # line; a login shell (`bash -lc`) may emit profile banner lines
    # before it, so take the last non-blank line — and accept it only
    # if it is an absolute path (clang is always an executable, never a
    # shell builtin / alias), so a non-path token fails safe to None.
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        return None
    path = lines[-1]
    return path if path.startswith("/") else None


def detect_real_exec_support() -> RealExecSupport:
    """Detect whether the real-execution parity comparison can run on
    this machine.

    `wsl` is located on the host PATH via `shutil.which`; `clang` is
    then probed INSIDE WSL (a Windows-PATH clang cannot build a Linux
    executable, so it is the wrong tool to detect). A machine with no
    WSL, or with WSL but no `clang`, yields `can_run_real() == False` —
    chunk D's dispatch then records the comparison as DEFERRED, never a
    hard failure, so CI on a tool-less runner stays green. Mirrors
    `gpu_ci.detect_tools` / `llvm_toolchain.detect_llvm_tools`."""
    wsl = shutil.which("wsl")
    if wsl is None:
        return RealExecSupport(
            wsl_available=False, wsl_clang=None,
            detail=("`wsl` is not on PATH — the real-execution parity "
                    "path runs Linux executables under WSL; without it "
                    "the real comparison is DEFERRED",))
    wsl_clang = _probe_wsl_clang(wsl)
    if wsl_clang is None:
        return RealExecSupport(
            wsl_available=True, wsl_clang=None,
            detail=("WSL is available but no `clang` was found inside "
                    "it — the LLVM backend's IR needs clang to compile "
                    "+ link to a runnable executable; the real "
                    "comparison is DEFERRED",))
    return RealExecSupport(
        wsl_available=True, wsl_clang=wsl_clang,
        detail=(f"WSL is available and `clang` was found inside it at "
                f"{wsl_clang!r} — the real-execution parity comparison "
                f"can run",))


# ==========================================================================
# Chunk D — the program-run substrate
# ==========================================================================

# Wall-clock cap on a single build (clang) or run invocation. A tiny
# corpus program compiles + runs sub-second; the cap only guards a hung
# tool. Matches gpu_ci.REAL_HW_TIMEOUT_S.
_REAL_EXEC_TIMEOUT_S = 30


@dataclass(frozen=True)
class _ProgramRun:
    """The observable result of building and running one backend's
    output for a program — the chunk-D program-run substrate.

    `ran` is True only when the executable was built AND run to
    completion: then `exit_code` is concrete and `stdout` / `stderr`
    hold its captured output. `ran` is False for ANY build or run
    failure (a backend compile error, a `clang` link error, a timeout,
    a WSL error): then `exit_code` is None, the output strings are
    empty, and `findings` carries the diagnostic — a failure is never
    silent.

    Frozen + `__post_init__`-guarded (the `gpu_ci` result-type
    discipline) so the silent-failure field shapes are
    unrepresentable."""
    label: str
    ran: bool
    exit_code: Optional[int]
    stdout: str
    stderr: str
    findings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError(
                "_ProgramRun: label is empty/blank — every run must "
                "name the backend it ran")
        if not isinstance(self.stdout, str) or not isinstance(
                self.stderr, str):
            raise ValueError(
                "_ProgramRun: stdout and stderr must be str")
        if self.ran:
            if self.exit_code is None:
                raise ValueError(
                    "_ProgramRun: ran=True but exit_code is None — a "
                    "completed run has a concrete exit code")
            if self.findings:
                raise ValueError(
                    "_ProgramRun: ran=True but findings is non-empty — "
                    "a completed run carries no failure diagnostic")
        else:
            if self.exit_code is not None:
                raise ValueError(
                    f"_ProgramRun: ran=False but exit_code="
                    f"{self.exit_code!r} — a failed run has no exit "
                    f"code")
            if not self.findings:
                raise ValueError(
                    "_ProgramRun: ran=False but findings is empty — a "
                    "build/run failure must carry a diagnostic")
            if self.stdout or self.stderr:
                raise ValueError(
                    "_ProgramRun: ran=False but stdout/stderr is "
                    "non-empty — a run that did not complete captured "
                    "no output")
        for entry in self.findings:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    f"_ProgramRun: findings has a blank or non-str "
                    f"entry ({entry!r}) — every diagnostic carries text")

    @classmethod
    def failed(cls, label: str, finding: str) -> "_ProgramRun":
        """A `_ProgramRun` for a build / run that did not complete —
        `ran=False` with the single diagnostic `finding`."""
        return cls(label=label, ran=False, exit_code=None, stdout="",
                   stderr="", findings=(finding,))

    @classmethod
    def completed(cls, label: str, exit_code: int, stdout: str,
                  stderr: str) -> "_ProgramRun":
        """A `_ProgramRun` for an executable that built and ran to
        completion."""
        return cls(label=label, ran=True, exit_code=exit_code,
                   stdout=stdout, stderr=stderr, findings=())


def _win_to_wsl(win_path: str) -> str:
    """Translate a Windows absolute path (`C:\\foo\\bar`) to its WSL
    form (`/mnt/c/foo/bar`), so a file on a `/mnt`-mounted drive is
    reachable from inside WSL. Handles any drive letter; a path that is
    already POSIX-style is returned normalised. Mirrors the helper in
    helixc/tests/test_codegen.py."""
    p = os.path.abspath(win_path).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:]
        if not rest.startswith("/"):
            rest = "/" + rest
        return f"/mnt/{drive}{rest}"
    return p


def _run_under_wsl(label: str, program: str,
                   exe_win_path: str) -> _ProgramRun:
    """Run an already-built Linux executable under WSL, capturing its
    exit code, stdout and stderr into a `_ProgramRun`.

    `chmod +x` is applied first — a file on a `/mnt/c` DrvFs mount is
    not executable by default — as a SEPARATE, return-code-checked
    step. Folding it into the run command with `&&` would let a chmod
    failure (the program then never running) masquerade as the
    program's own exit code — a `ran=True` result for a program that
    never ran. Neither WSL call uses a shell: the executable path is a
    single argv element, so spaces / quotes in it need no escaping.

    Applies the gpu_ci subprocess discipline: a `TimeoutExpired` or
    `OSError` (a hung / vanished `wsl`) becomes a `_ProgramRun.failed`,
    never an uncaught traceback. Output is decoded with
    `errors='replace'` so non-UTF-8 program output cannot raise — both
    backends are decoded identically, so the comparison stays valid."""
    wsl = shutil.which("wsl")
    if wsl is None:
        return _ProgramRun.failed(
            label, f"{label}: `wsl` is not on PATH — cannot run the "
                   f"compiled program for {program!r}")
    wsl_path = _win_to_wsl(exe_win_path)
    # Step 1 — make the executable runnable; check chmod's own exit so
    # a chmod failure is a loud failure, never folded into the run.
    try:
        chmod = subprocess.run(
            [wsl, "--", "chmod", "+x", wsl_path],
            capture_output=True, text=True, errors="replace",
            timeout=_REAL_EXEC_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return _ProgramRun.failed(
            label, f"{label}: chmod of {program!r}'s executable timed "
                   f"out after {_REAL_EXEC_TIMEOUT_S}s")
    except OSError as exc:
        return _ProgramRun.failed(
            label, f"{label}: `wsl` unusable for the chmod of "
                   f"{program!r} ({type(exc).__name__}: {exc})")
    if chmod.returncode != 0:
        diag = (chmod.stderr or chmod.stdout or "").strip()
        return _ProgramRun.failed(
            label, f"{label}: chmod +x failed for {program!r}'s "
                   f"executable (exit {chmod.returncode}): "
                   f"{diag[:300]}")
    # Step 2 — run the executable and capture its observable behaviour.
    try:
        proc = subprocess.run(
            [wsl, "--", wsl_path],
            capture_output=True, text=True, errors="replace",
            timeout=_REAL_EXEC_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return _ProgramRun.failed(
            label, f"{label}: running {program!r} timed out after "
                   f"{_REAL_EXEC_TIMEOUT_S}s")
    except OSError as exc:
        return _ProgramRun.failed(
            label, f"{label}: `wsl` unusable running {program!r} "
                   f"({type(exc).__name__}: {exc})")
    # A `wsl` exit OUTSIDE 0-255 is the launcher's own error code, not
    # the program's — e.g. 0xFFFFFFFF when the WSL service is down or
    # no distro is installed. A real Linux process exit status is
    # always 0-255 (incl. signal deaths, reported as 128+signo). So an
    # out-of-range code means the program NEVER RAN — capture it as a
    # failure, never fold a launcher artifact into exit_code as a
    # `ran=True` (the chmod-masquerade class, here on the run leg).
    if not (0 <= proc.returncode <= 255):
        diag = (proc.stdout or proc.stderr or "").strip()
        return _ProgramRun.failed(
            label, f"{label}: {program!r} did not run under WSL — the "
                   f"`wsl` launcher exited {proc.returncode} (a real "
                   f"Linux run exits 0-255)"
                   + (f": {diag[:300]}" if diag else ""))
    return _ProgramRun.completed(label, proc.returncode,
                                 proc.stdout, proc.stderr)


def _run_x86_program(module: tir.Module, program: str) -> _ProgramRun:
    """Build the x86_64 backend's freestanding ELF for `module` and run
    it under WSL, capturing observable behaviour. The caller's module
    is not mutated — a deep copy is compiled."""
    try:
        elf = compile_module_to_elf(copy.deepcopy(module))
    except Exception as exc:
        # The x86_64 backend has full op coverage, so for a structural-
        # MATCH module this should not happen — captured loudly anyway.
        return _ProgramRun.failed(
            "x86_64", f"x86_64 backend failed to compile {program!r}: "
                      f"{type(exc).__name__}: {exc}")
    if not elf:
        return _ProgramRun.failed(
            "x86_64", f"x86_64 backend produced an empty ELF for "
                      f"{program!r} — there is nothing to run")
    try:
        tmpdir = tempfile.mkdtemp(prefix="helix_parity_x86_")
    except OSError as exc:
        return _ProgramRun.failed(
            "x86_64", f"could not create a temp directory to run "
                      f"{program!r} ({type(exc).__name__}: {exc})")
    try:
        exe = os.path.join(tmpdir, "x86.bin")
        try:
            with open(exe, "wb") as f:
                f.write(elf)
        except OSError as exc:
            return _ProgramRun.failed(
                "x86_64", f"could not write the x86_64 ELF for "
                          f"{program!r} ({type(exc).__name__}: {exc})")
        return _run_under_wsl("x86_64", program, exe)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _run_llvm_program(module: tir.Module, program: str,
                      clang: str) -> _ProgramRun:
    """Emit the LLVM backend's IR for `module`, compile + link it to a
    runnable Linux executable with `clang` inside WSL, and run it,
    capturing observable behaviour. `clang` is the path of a clang
    INSIDE WSL (from `detect_real_exec_support().wsl_clang`). The
    caller's module is not mutated — a deep copy is emitted."""
    try:
        ll_text = emit_module(copy.deepcopy(module))
    except Exception as exc:
        # An LLVMEmitError here means the module is not actually in the
        # covered subset — the caller (chunk E) only runs MATCH modules,
        # but a failure is captured loudly rather than raised.
        return _ProgramRun.failed(
            "llvm", f"LLVM backend failed to emit IR for {program!r}: "
                    f"{type(exc).__name__}: {exc}")
    wsl = shutil.which("wsl")
    if wsl is None:
        return _ProgramRun.failed(
            "llvm", f"`wsl` is not on PATH — cannot compile/run the "
                    f"LLVM output for {program!r}")
    try:
        tmpdir = tempfile.mkdtemp(prefix="helix_parity_llvm_")
    except OSError as exc:
        return _ProgramRun.failed(
            "llvm", f"could not create a temp directory to run "
                    f"{program!r} ({type(exc).__name__}: {exc})")
    try:
        ll_path = os.path.join(tmpdir, "module.ll")
        exe_path = os.path.join(tmpdir, "llvm.bin")
        try:
            with open(ll_path, "w", encoding="utf-8") as f:
                f.write(ll_text)
        except OSError as exc:
            return _ProgramRun.failed(
                "llvm", f"could not write the LLVM IR for {program!r} "
                        f"({type(exc).__name__}: {exc})")
        # clang takes the textual `.ll` directly — it assembles,
        # optimizes and links it (with the C runtime, which supplies
        # `_start` -> `main`) to a native executable in one step.
        compile_cmd = [wsl, "--", clang, _win_to_wsl(ll_path),
                       "-o", _win_to_wsl(exe_path)]
        try:
            proc = subprocess.run(
                compile_cmd, capture_output=True, text=True,
                errors="replace", timeout=_REAL_EXEC_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            return _ProgramRun.failed(
                "llvm", f"clang compiling {program!r}'s LLVM IR timed "
                        f"out after {_REAL_EXEC_TIMEOUT_S}s")
        except OSError as exc:
            return _ProgramRun.failed(
                "llvm", f"clang unusable compiling {program!r} "
                        f"({type(exc).__name__}: {exc})")
        if proc.returncode != 0:
            diag = (proc.stderr or proc.stdout or "").strip()
            return _ProgramRun.failed(
                "llvm", f"clang failed to compile {program!r}'s LLVM "
                        f"IR (exit {proc.returncode}): {diag[:500]}")
        # A 0 exit is necessary but not sufficient — confirm clang
        # actually wrote a non-empty executable (the gpu_ci discipline:
        # a 0 exit with no artifact is not a successful build).
        try:
            size = os.path.getsize(exe_path)
        except OSError:
            size = -1
        if size <= 0:
            return _ProgramRun.failed(
                "llvm", f"clang exited 0 but produced no executable "
                        f"for {program!r} — a 0 exit with no artifact "
                        f"is not a successful build")
        return _run_under_wsl("llvm", program, exe_path)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ==========================================================================
# Chunk E — the real-execution comparison + the attempt_real wiring
# ==========================================================================

def _compare_runs(x86_run: _ProgramRun,
                  llvm_run: _ProgramRun) -> tuple[bool, tuple[str, ...]]:
    """Compare two `_ProgramRun`s for observable-behaviour parity.

    Returns `(passed, findings)`. `passed` is True only when BOTH
    backends' executables ran to completion AND their exit code,
    stdout and stderr are byte-identical. Any build / run failure on
    either side, or any observable difference, gives `passed=False`
    with `findings` describing it.

    The comparison is EXACT (byte-identical), which is sound for the
    LLVM backend's CURRENT covered op set: its PRINT / TRAP lowering
    emits program output through a direct `write` syscall (libc
    `write` is a thin wrapper), exactly as the x86_64 freestanding ELF
    does — neither uses buffered stdio, so there is no flush-ordering
    divergence to tolerate. (A future 206-R op that lowered output via
    buffered stdio would need a flush-tolerant comparison here.) Exit
    codes are comparable for the same reason: `_run_under_wsl` has
    already pinned each captured code to the kernel-masked 0-255
    range, so both sides carry `main`'s return in the same 8-bit
    form."""
    findings: list[str] = []
    if not x86_run.ran:
        findings.append(
            "x86_64 baseline did not build/run: "
            + "; ".join(x86_run.findings))
    if not llvm_run.ran:
        findings.append(
            "LLVM backend's output did not build/run: "
            + "; ".join(llvm_run.findings))
    if findings:
        # One or both never produced an observable result — there is
        # nothing to compare, so this is not parity.
        return False, tuple(findings)
    # Both ran — compare observable behaviour exactly.
    if x86_run.exit_code != llvm_run.exit_code:
        findings.append(
            f"exit code differs: x86_64 returned {x86_run.exit_code}, "
            f"LLVM returned {llvm_run.exit_code}")
    if x86_run.stdout != llvm_run.stdout:
        # Include the byte lengths — two outputs that diverge only
        # past the 160-char snippet would otherwise show identical
        # snippets under a "stdout differs" finding.
        findings.append(
            f"stdout differs: x86_64 {len(x86_run.stdout)} bytes "
            f"{x86_run.stdout[:160]!r}, LLVM {len(llvm_run.stdout)} "
            f"bytes {llvm_run.stdout[:160]!r}")
    if x86_run.stderr != llvm_run.stderr:
        findings.append(
            f"stderr differs: x86_64 {len(x86_run.stderr)} bytes "
            f"{x86_run.stderr[:160]!r}, LLVM {len(llvm_run.stderr)} "
            f"bytes {llvm_run.stderr[:160]!r}")
    if findings:
        return False, tuple(findings)
    return True, ()


def _attempt_real_parity(module: tir.Module, program: str,
                         mock_result: ParityResult) -> ParityResult:
    """Run both backends' compiled output and compare observable
    behaviour, returning `mock_result` with its real-* fields filled.

    Called by `check_parity` only for a structural-MATCH module when
    `attempt_real=True`. When the toolchain to run the comparison is
    absent (`RealExecSupport.can_run_real()` is False) the real result
    is DEFERRED — `real_attempted=True, real_passed=None` carrying the
    detection's own reason — never a hard failure, so CI on a tool-less
    runner stays green (the `gpu_ci` real-HW dispatch discipline).
    Otherwise both backends are built + run and `_compare_runs` decides
    PASS / FAIL."""
    support = detect_real_exec_support()
    if not support.can_run_real():
        # DEFERRED — no toolchain to run the real comparison. The
        # `RealExecSupport.detail` records exactly what is missing
        # (and is guaranteed non-empty), so the DEFERRED is never
        # silent.
        return dataclasses.replace(
            mock_result, real_attempted=True, real_passed=None,
            real_findings=support.detail)
    # can_run_real() is True, so `support.wsl_clang` is a concrete
    # clang path inside WSL (its `__post_init__` ties the two).
    x86_run = _run_x86_program(module, program)
    llvm_run = _run_llvm_program(module, program, support.wsl_clang)
    passed, findings = _compare_runs(x86_run, llvm_run)
    return dataclasses.replace(
        mock_result, real_attempted=True, real_passed=passed,
        real_findings=findings)
