"""
helixc/ir/mlir/validate.py — toolchain-free MLIR-text validation
(v3.0 Phase E, Stage 211 chunk E).

`mock_validate_mlir` is the mock-path MLIR validator: a toolchain-free
STRUCTURAL shape check on MLIR textual IR, the MLIR analogue of
`helixc.backend.llvm_ir.mock_validate_ll`. It runs in CI on a machine
with no MLIR toolchain — it never `import mlir`, never shells out to
`mlir-opt`.

It is NOT a real verifier. Real MLIR verification — verifier traits,
type-correctness, SSA dominance — needs `mlir-opt` (or the in-process
bindings) and is built in Stage 212. So `mock_validate_mlir` returns a
frozen tri-state `MLIRValidation`:

- FAILED — a definite STRUCTURAL defect (non-str / empty input, no
  top-level structure, an unterminated string literal, unbalanced
  braces / parentheses). A malformed shape is malformed regardless of
  any toolchain, so the mock check FAILS with confidence.
- DEFERRED — the shape check found no defect, but that is NOT a
  certification of real MLIR validity; a real check is needed and was
  not run. This is the honest outcome for well-formed text from a
  toolchain-free checker — never a false PASSED.
- PASSED — reserved for the Stage-212 REAL validator (a successful
  `mlir-opt` verification). `mock_validate_mlir`, being toolchain-free,
  NEVER returns PASSED — it can only confidently FAIL or honestly
  DEFER. PASSED is in the tri-state so the one `MLIRValidation` type
  serves both the mock and the future real validator.

Stage 213 chunk B adds `validate_mlir_with_toolchain`, the first real
validation dispatch seam: it still runs the mock shape check first,
then invokes `mlir-opt` when that tool is available. A tool-less
machine continues to return DEFERRED — never a false PASS — so CI on a
binding-less runner stays green and the home-grown tile-IR path stays
the reversible fallback.

License: Apache 2.0
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .toolchain import MLIRSupport, detect_mlir_support


class MLIRValidationVerdict(Enum):
    """The tri-state outcome of an MLIR-text validation — the Stage 210
    decision's mock-path tri-state (section 3).

    `mock_validate_mlir` produces only FAILED (a definite structural
    defect) and DEFERRED (no defect found, real validity unverified);
    PASSED is reserved for the Stage-212 real `mlir-opt` validator."""
    PASSED = "passed"
    FAILED = "failed"
    DEFERRED = "deferred"


def _check_validation_verdicts() -> None:
    """Module-load guard: `MLIRValidationVerdict` is exactly the
    tri-state {PASSED, FAILED, DEFERRED} the Stage 210 decision's
    mock-path discipline (section 3) defines — no more, no less. A
    fourth verdict added without updating `MLIRValidation.__post_init__`
    (which branches on each verdict) and `mock_validate_mlir`
    would silently widen the contract. Mirrors the module-load drift
    guards of `toolchain.py` / `mapping.py`."""
    names = {v.name for v in MLIRValidationVerdict}
    if names != {"PASSED", "FAILED", "DEFERRED"}:
        raise AssertionError(
            f"helixc.ir.mlir.validate: MLIRValidationVerdict must be "
            f"exactly PASSED / FAILED / DEFERRED — got {sorted(names)}")


_check_validation_verdicts()


@dataclass(frozen=True)
class MLIRValidation:
    """The result of validating a piece of MLIR textual IR — a frozen
    tri-state verdict plus the findings that explain it.

    Frozen + `__post_init__`-guarded, the house discipline of
    `toolchain.MLIRSupport`:
    - a FAILED or DEFERRED result MUST carry at least one finding — it
      is never silent about why (the mock-path rule);
    - a PASSED result MUST carry NO findings — a clean pass has
      nothing to report; `findings` describes a defect or a deferral
      reason, and a PASSED has neither, so a PASSED with findings is
      an incoherent result and is rejected;
    - every finding carries text.
    """
    verdict: MLIRValidationVerdict
    findings: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.verdict, MLIRValidationVerdict):
            raise ValueError(
                f"MLIRValidation: verdict must be a "
                f"MLIRValidationVerdict — got {self.verdict!r}")
        if not isinstance(self.findings, tuple):
            raise ValueError(
                "MLIRValidation: findings must be a tuple, got "
                f"{type(self.findings).__name__}")
        for entry in self.findings:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    f"MLIRValidation: findings has a blank or non-str "
                    f"entry ({entry!r}) — every finding carries text")
        if (self.verdict in (MLIRValidationVerdict.FAILED,
                             MLIRValidationVerdict.DEFERRED)
                and not self.findings):
            raise ValueError(
                f"MLIRValidation: a {self.verdict.name} result must "
                f"carry at least one finding explaining why — it must "
                f"never be silent about a defect or a deferral")
        if self.verdict is MLIRValidationVerdict.PASSED and self.findings:
            raise ValueError(
                f"MLIRValidation: a PASSED result must carry NO "
                f"findings ({len(self.findings)} given) — a clean pass "
                f"has nothing to report; `findings` describes a defect "
                f"or a deferral reason, and a PASSED has neither, so a "
                f"PASSED carrying a finding is an incoherent result")

    def passed(self) -> bool:
        """True iff a real validator confirmed the IR is valid."""
        return self.verdict is MLIRValidationVerdict.PASSED

    def failed(self) -> bool:
        """True iff a definite structural defect was found."""
        return self.verdict is MLIRValidationVerdict.FAILED

    def deferred(self) -> bool:
        """True iff no defect was found but real validity is unverified
        — the honest mock-path outcome for well-formed text."""
        return self.verdict is MLIRValidationVerdict.DEFERRED


# A double-quoted span (MLIR string literals / quoted symbol names),
# with backslash escapes honoured — masked out before brace / paren
# counting so a `{` legally inside a string attribute is not
# miscounted as structural punctuation.
_QUOTED_SPAN = re.compile(r'"(?:[^"\\]|\\.)*"')
# An MLIR `//` line comment, to end of line.
_LINE_COMMENT = re.compile(r"//[^\n]*")

# Wall-clock cap on the real `mlir-opt` verifier dispatch. Small MLIR
# text should validate quickly; the cap exists only to avoid hanging on
# a broken tool.
_MLIR_VALIDATE_TIMEOUT_S = 30


def _structural_text(mlir_text: str) -> str:
    """`mlir_text` with string literals and `//` line comments removed,
    so brace / parenthesis counting sees only structural punctuation.

    Quoted spans are masked FIRST — so a `//` inside a string literal
    is not mistaken for the start of a comment."""
    return _LINE_COMMENT.sub("", _QUOTED_SPAN.sub("", mlir_text))


def mock_validate_mlir(mlir_text: str) -> MLIRValidation:
    """Toolchain-free STRUCTURAL shape check on MLIR textual IR.

    Returns a frozen `MLIRValidation`: FAILED on a definite structural
    defect, otherwise DEFERRED — never PASSED, because a toolchain-free
    check cannot certify real MLIR validity (see the module docstring).

    Checks — deliberately conservative, so a clean shape never yields a
    false FAILED (the real `mlir-opt` verifier at Stage 212 catches
    what this cannot):
    - the argument is a `str` — a non-str input is itself a FAILED, not
      an exception;
    - the text is non-empty;
    - it has a top-level structure — a `module` or a `func.func`;
    - string literals are terminated;
    - braces `{}` and parentheses `()` balance.
    The structure / brace / parenthesis checks run on the text with
    string literals and `//` comments removed, so punctuation inside
    them is never miscounted. When a string literal is unterminated the
    balance checks are SKIPPED — the dangling run makes the counts
    meaningless — and the unterminated literal is reported instead, so
    the finding names the real defect, not a spurious imbalance.

    Never raises — a defect (including a non-str argument) is reported
    as a FAILED finding, the discipline of `mock_validate_ll`."""
    if not isinstance(mlir_text, str):
        return MLIRValidation(
            MLIRValidationVerdict.FAILED,
            (f"not MLIR text — expected a str, got "
             f"{type(mlir_text).__name__}",))
    problems: list[str] = []
    if not mlir_text.strip():
        problems.append("empty — no MLIR text to validate")
    else:
        structural = _structural_text(mlir_text)
        if '"' in structural:
            # `_QUOTED_SPAN` masks only COMPLETE `"..."` spans, so a `"`
            # left in `structural` is a dangling quote — an unterminated
            # string literal. Its run could hold any punctuation, so the
            # brace / paren counts would be unreliable: report the real
            # defect and skip the balance checks (a spurious imbalance
            # would misdirect debugging).
            problems.append(
                "unterminated string literal — a dangling double-quote; "
                "brace / parenthesis balance not checked")
        else:
            if ("func.func" not in structural
                    and "module" not in structural):
                problems.append(
                    "no top-level structure — neither a `module` nor a "
                    "`func.func` is present")
            for opener, closer, name in (("{", "}", "brace"),
                                         ("(", ")", "parenthesis")):
                opens = structural.count(opener)
                closes = structural.count(closer)
                if opens != closes:
                    problems.append(
                        f"unbalanced {name}s: {opens} {opener!r} vs "
                        f"{closes} {closer!r}")

    if problems:
        return MLIRValidation(MLIRValidationVerdict.FAILED,
                              tuple(problems))
    return MLIRValidation(
        MLIRValidationVerdict.DEFERRED,
        ("the toolchain-free shape check found no structural defect, "
         "but real MLIR validity — verifier traits, type-correctness, "
         "SSA dominance — needs `mlir-opt`; validation is DEFERRED to "
         "`validate_mlir_with_toolchain`",))


def _run_mlir_opt_validate(
        mlir_text: str,
        mlir_opt: str,
        *,
        timeout_s: int = _MLIR_VALIDATE_TIMEOUT_S) -> MLIRValidation:
    """Run `mlir-opt` as the real MLIR verifier.

    A zero exit is necessary but not sufficient: the output artifact
    must exist and be non-empty, mirroring the LLVM dispatch hygiene.
    Tool errors are captured as FAILED findings, never uncaught
    tracebacks.
    """
    if not isinstance(mlir_opt, str) or not mlir_opt.strip():
        return MLIRValidation(
            MLIRValidationVerdict.FAILED,
            ("mlir-opt validation requested with a blank or non-str "
             f"tool path ({mlir_opt!r})",))

    with tempfile.TemporaryDirectory(prefix="helix_mlir_validate_") as tmpdir:
        mlir_path = os.path.join(tmpdir, "module.mlir")
        out_path = os.path.join(tmpdir, "verified.mlir")
        try:
            with open(mlir_path, "w", encoding="utf-8") as f:
                f.write(mlir_text)
        except OSError as exc:
            return MLIRValidation(
                MLIRValidationVerdict.FAILED,
                (f"could not write temp MLIR input {mlir_path!r} "
                 f"({type(exc).__name__}: {exc})",))

        try:
            proc = subprocess.run(
                [mlir_opt, mlir_path, "-o", out_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return MLIRValidation(
                MLIRValidationVerdict.FAILED,
                (f"mlir-opt validation timed out after {timeout_s}s",))
        except OSError as exc:
            return MLIRValidation(
                MLIRValidationVerdict.FAILED,
                (f"mlir-opt validation: tool unusable at invocation "
                 f"({type(exc).__name__}: {exc})",))

        if proc.returncode != 0:
            diag = (proc.stderr or proc.stdout or "").strip()
            if not diag:
                diag = "no diagnostic emitted"
            return MLIRValidation(
                MLIRValidationVerdict.FAILED,
                (f"mlir-opt exit {proc.returncode}: {diag[:500]}",))

        try:
            size = os.path.getsize(out_path)
        except OSError:
            size = -1
        if size <= 0:
            return MLIRValidation(
                MLIRValidationVerdict.FAILED,
                (f"mlir-opt exited 0 but produced no output artifact at "
                 f"{out_path!r} — a 0 exit with no artifact is not a "
                 "validation pass",))

    return MLIRValidation(MLIRValidationVerdict.PASSED, ())


def validate_mlir_with_toolchain(
        mlir_text: str,
        *,
        support: Optional[MLIRSupport] = None) -> MLIRValidation:
    """Validate MLIR text with the strongest available verifier.

    Always run `mock_validate_mlir` first. If it finds a structural
    defect, return that FAILED result and do not probe or invoke tools.
    If the mock shape is clean and `mlir-opt` is available, dispatch to
    it for real verification. If `mlir-opt` is absent, return an honest
    DEFERRED with the support details; the in-process bindings are only
    a capability surface here, not a verifier runner yet.
    """
    mock = mock_validate_mlir(mlir_text)
    if mock.failed():
        return mock

    if support is None:
        support = detect_mlir_support()
    if not isinstance(support, MLIRSupport):
        raise ValueError(
            "validate_mlir_with_toolchain: support must be an "
            f"MLIRSupport or None, got {support!r}")

    if support.mlir_opt is None:
        details = tuple(f"MLIR support probe: {line}"
                        for line in support.detail)
        return MLIRValidation(
            MLIRValidationVerdict.DEFERRED,
            (mock.findings
             + details
             + ("real MLIR validation is DEFERRED because `mlir-opt` "
                "is not available; in-process binding validation is "
                "not wired in Stage 213 chunk B",)))

    return _run_mlir_opt_validate(mlir_text, support.mlir_opt)
