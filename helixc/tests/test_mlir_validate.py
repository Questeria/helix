"""Tests for helixc.ir.mlir.validate — v3.0 Phase E, Stage 211 chunk
E: the toolchain-free MLIR-text validator.

`mock_validate_mlir` is the mock-path MLIR validator — a toolchain-free
STRUCTURAL shape check on MLIR textual IR (the MLIR analogue of
`llvm_ir.mock_validate_ll`). It returns a frozen tri-state
`MLIRValidation`: FAILED on a definite structural defect, DEFERRED when
the shape is clean but real validity is unverified, and — never from
the mock checker — PASSED (reserved for the Stage-212 real validator).

These tests pin: the `MLIRValidationVerdict` tri-state and its
module-load guard; the `MLIRValidation` frozen result's `__post_init__`
rejections (a FAILED / DEFERRED is never silent); the predicates;
`mock_validate_mlir`'s defect detection (empty, no structure,
unbalanced braces / parens) and its honest DEFERRED on clean text; that
it NEVER returns a false PASSED; that string-literal / comment
punctuation is masked from the brace count; and — the mock-path rule —
that the module never `import mlir`.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from helixc.ir.mlir import validate
from helixc.ir.mlir.validate import (
    MLIRValidation, MLIRValidationVerdict, mock_validate_mlir,
)

# A structurally well-formed MLIR module — the DEFERRED baseline.
_WELL_FORMED = """\
module {
  func.func @main() -> i32 {
    %0 = arith.constant 1 : i32
    return %0 : i32
  }
}
"""


# --------------------------------------------------------------------------
# MLIRValidationVerdict — the tri-state + its guard
# --------------------------------------------------------------------------
def test_validation_verdict_members():
    """`MLIRValidationVerdict` is exactly the tri-state PASSED / FAILED
    / DEFERRED, each with a distinct string value."""
    assert {v.name for v in MLIRValidationVerdict} == {
        "PASSED", "FAILED", "DEFERRED"}
    values = [v.value for v in MLIRValidationVerdict]
    assert len(values) == len(set(values)), "values must be unique"


def test_check_validation_verdicts_guard():
    """The module-load guard `_check_validation_verdicts` is callable
    and passes for the current tri-state enum."""
    validate._check_validation_verdicts()  # must not raise


# --------------------------------------------------------------------------
# MLIRValidation — __post_init__ rejects illegal / silent results
# --------------------------------------------------------------------------
def test_mlir_validation_rejects_silent_failed():
    """A FAILED result with no findings would be silent about the
    defect it found — rejected."""
    with pytest.raises(ValueError, match="must carry at least one"):
        MLIRValidation(MLIRValidationVerdict.FAILED, ())


def test_mlir_validation_rejects_silent_deferred():
    """A DEFERRED result with no findings would be silent about why it
    deferred — rejected (the mock-path rule: never a silent DEFERRED)."""
    with pytest.raises(ValueError, match="must carry at least one"):
        MLIRValidation(MLIRValidationVerdict.DEFERRED, ())


def test_mlir_validation_rejects_blank_finding():
    """A blank / non-str finding is a reason-shaped object with no
    reason — rejected."""
    with pytest.raises(ValueError, match="blank or non-str"):
        MLIRValidation(MLIRValidationVerdict.FAILED, ("   ",))


def test_mlir_validation_passed_may_be_empty():
    """A PASSED result MAY carry no findings — a clean pass needs no
    explanation (unlike FAILED / DEFERRED)."""
    ok = MLIRValidation(MLIRValidationVerdict.PASSED, ())
    assert ok.passed() and ok.findings == ()


def test_mlir_validation_predicates():
    """`passed` / `failed` / `deferred` derive from the verdict, and
    exactly one holds for any result."""
    for verdict in MLIRValidationVerdict:
        findings = () if verdict is MLIRValidationVerdict.PASSED \
            else ("a reason",)
        r = MLIRValidation(verdict, findings)
        flags = [r.passed(), r.failed(), r.deferred()]
        assert sum(flags) == 1, verdict
        assert r.passed() == (verdict is MLIRValidationVerdict.PASSED)
        assert r.failed() == (verdict is MLIRValidationVerdict.FAILED)
        assert r.deferred() == (
            verdict is MLIRValidationVerdict.DEFERRED)


# --------------------------------------------------------------------------
# mock_validate_mlir — defect detection
# --------------------------------------------------------------------------
def test_mock_validate_mlir_empty_is_failed():
    """Empty or whitespace-only text is a FAILED — there is no MLIR to
    validate."""
    for text in ("", "   ", "\n\n  \t\n"):
        r = mock_validate_mlir(text)
        assert r.failed(), text
        assert any("empty" in f for f in r.findings), r.findings


def test_mock_validate_mlir_no_structure_is_failed():
    """Text with neither a `module` nor a `func.func` has no top-level
    MLIR structure — FAILED."""
    r = mock_validate_mlir("%0 = arith.constant 1 : i32")
    assert r.failed()
    assert any("no top-level structure" in f for f in r.findings)


def test_mock_validate_mlir_unbalanced_braces_is_failed():
    """An unbalanced brace is a definite structural defect — FAILED,
    with the open/close counts named."""
    r = mock_validate_mlir("module {\n  func.func @f() {\n")
    assert r.failed()
    assert any("unbalanced brace" in f for f in r.findings), r.findings


def test_mock_validate_mlir_unbalanced_parens_is_failed():
    """An unbalanced parenthesis is a definite structural defect —
    FAILED. The test input is brace-BALANCED, so the finding isolates
    the parenthesis defect (no spurious brace finding)."""
    r = mock_validate_mlir(
        "module { func.func @f(%a: i32 { return } }")
    assert r.failed()
    assert any("unbalanced parenthes" in f for f in r.findings), \
        r.findings
    assert not any("brace" in f for f in r.findings), r.findings


def test_mock_validate_mlir_rejects_non_str_without_raising():
    """A non-str argument is itself a FAILED — `mock_validate_mlir`
    NEVER raises. A caller that passes `None` (an upstream lowering
    that produced nothing) gets a named defect, not an opaque
    `AttributeError`."""
    for bad in (None, 123, b"module {}", ["module"]):
        r = mock_validate_mlir(bad)        # must not raise
        assert r.failed(), bad
        assert any("not MLIR text" in f for f in r.findings), r.findings


def test_mock_validate_mlir_unterminated_string_is_failed():
    """An unterminated string literal is FAILED — and reported AS an
    unterminated string, NOT misattributed to a brace imbalance the
    dangling quote would otherwise fake (the balance checks are
    skipped when a quote dangles)."""
    r = mock_validate_mlir(
        'module { func.func @f() { %0 = x.y "oops }')
    assert r.failed()
    assert any("unterminated string" in f for f in r.findings), \
        r.findings
    assert not any("unbalanced brace" in f for f in r.findings), \
        r.findings


# --------------------------------------------------------------------------
# mock_validate_mlir — well-formed text DEFERS, never falsely PASSES
# --------------------------------------------------------------------------
def test_mock_validate_mlir_well_formed_defers():
    """A structurally well-formed module DEFERS — the toolchain-free
    check found no defect, but cannot certify real validity, so it is
    honestly DEFERRED, not a false PASSED. The finding explains."""
    r = mock_validate_mlir(_WELL_FORMED)
    assert r.deferred()
    assert any("mlir-opt" in f for f in r.findings), r.findings


def test_mock_validate_mlir_never_returns_passed():
    """`mock_validate_mlir` — being toolchain-free — NEVER returns
    PASSED for any input: it can only confidently FAIL or honestly
    DEFER. PASSED is reserved for the Stage-212 real validator."""
    samples = (
        _WELL_FORMED, "", "   ", "module {}",
        "func.func @f() { return }",
        "module {\n  func.func @f() {\n",      # unbalanced
        "%0 = arith.constant 1 : i32",          # no structure
        'module { func.func @g() { "s}(" } }',  # quoted punctuation
    )
    for text in samples:
        assert mock_validate_mlir(text).verdict is not \
            MLIRValidationVerdict.PASSED, text


def test_mock_validate_mlir_bare_func_defers():
    """A top-level `func.func` with no enclosing `module` is still
    recognised structure — MLIR allows it — so a balanced one DEFERS."""
    r = mock_validate_mlir("func.func @f() {\n  return\n}\n")
    assert r.deferred()


# --------------------------------------------------------------------------
# mock_validate_mlir — string-literal / comment punctuation is masked
# --------------------------------------------------------------------------
def test_mock_validate_mlir_masks_quoted_punctuation():
    """A brace or parenthesis inside a string literal must NOT be
    miscounted as structural — the well-formed module stays DEFERRED
    even with `{` / `(` characters inside a quoted attribute."""
    r = mock_validate_mlir(
        'module { func.func @f() { %0 = foo.bar {tag = "a}b)c{"} '
        ': i32 } }')
    assert r.deferred(), r.findings


def test_mock_validate_mlir_masks_comment_punctuation():
    """A brace inside a `//` line comment must NOT be miscounted — the
    comment's stray `}` does not unbalance a well-formed module."""
    r = mock_validate_mlir(
        "module {\n  // a dangling } in a comment\n"
        "  func.func @f() { return }\n}\n")
    assert r.deferred(), r.findings


# --------------------------------------------------------------------------
# the mock-path rule — validate is toolchain-free, never `import mlir`
# --------------------------------------------------------------------------
def test_validate_module_imports_without_mlir_bindings():
    """THE MOCK-PATH RULE (Stage 210 decision, section 3): `validate` is
    toolchain-free — it NEVER `import mlir`, at module top level or
    anywhere, and never shells out to `mlir-opt`. Parse the module's
    AST and confirm not one `import mlir` / `from mlir ...` statement —
    a host-independent structural pin."""
    tree = ast.parse(
        Path(validate.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith("mlir"), a.name
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("mlir"), node.module
