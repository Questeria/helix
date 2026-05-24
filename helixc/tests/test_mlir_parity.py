"""Tests for helixc.ir.mlir.parity — Stage 215 MLIR-vs-tile-IR
parity gate (chunk A: harness skeleton).

The dev machine has no MLIR / mlir-translate / chained-tool
toolchain, so the parity harness returns PARITY_DEFERRED on every
real-toolchain path. These tests pin the harness's behaviour around
the result type, the entry-point input validation, and the
verdict-from-MLIR-status mapping (PASSED -> HOLDS, FAILED -> FAILED,
DEFERRED -> DEFERRED).
"""
from __future__ import annotations

import pytest

from helixc.ir import tile_ir, tir
from helixc.ir.mlir import backends, parity
from helixc.ir.mlir.backends import (
    MLIRBackendResult,
    MLIRBackendStatus,
    MLIRBackendTarget,
)
from helixc.ir.mlir.parity import (
    ParityResult,
    ParityStatus,
    mlir_vs_tile_ir_parity_check,
)
from helixc.ir.mlir.toolchain import MLIRSupport
from helixc.ir.mlir.validate import MLIRValidation, MLIRValidationVerdict


_TK = tile_ir.TileOpKind


def _ret(*operands):
    return tile_ir.TileOp(_TK.RETURN, operands=list(operands))


def _fn(name, params, return_ty, *ops):
    return tile_ir.TileFn(
        name, list(params), return_ty,
        [tile_ir.TileBlock(0, list(params), list(ops))])


def _trivial_module() -> tile_ir.TileModule:
    """A minimal Tile-IR module with one void function — small enough
    to round-trip through the Stage 212 translator cleanly."""
    fn = _fn("main", [], tir.TIRUnit(), _ret())
    return tile_ir.TileModule(functions={"main": fn})


# --------------------------------------------------------------------------
# ParityStatus / ParityResult — type discipline
# --------------------------------------------------------------------------
def test_parity_status_tri_state():
    assert {s.name for s in ParityStatus} == {
        "PARITY_HOLDS", "PARITY_FAILED", "PARITY_DEFERRED",
    }


def test_parity_result_rejects_non_target():
    with pytest.raises(ValueError, match="target must be"):
        ParityResult(
            target="ptx",  # type: ignore[arg-type]
            status=ParityStatus.PARITY_HOLDS,
            findings=(),
        )


def test_parity_result_rejects_non_status():
    with pytest.raises(ValueError, match="status must be"):
        ParityResult(
            target=MLIRBackendTarget.PTX,
            status="holds",  # type: ignore[arg-type]
            findings=(),
        )


def test_parity_result_rejects_non_tuple_findings():
    with pytest.raises(ValueError, match="findings must be a tuple"):
        ParityResult(
            target=MLIRBackendTarget.PTX,
            status=ParityStatus.PARITY_FAILED,
            findings=["bad"],  # type: ignore[arg-type]
        )


def test_parity_result_rejects_blank_finding_entry():
    with pytest.raises(ValueError, match="blank or non-str"):
        ParityResult(
            target=MLIRBackendTarget.PTX,
            status=ParityStatus.PARITY_FAILED,
            findings=("   ",),
        )


def test_parity_result_rejects_silent_failed():
    """A PARITY_FAILED result must explain why."""
    with pytest.raises(ValueError, match="carry at least one finding"):
        ParityResult(
            target=MLIRBackendTarget.PTX,
            status=ParityStatus.PARITY_FAILED,
            findings=(),
        )


def test_parity_result_rejects_silent_deferred():
    """A PARITY_DEFERRED result must explain why."""
    with pytest.raises(ValueError, match="carry at least one finding"):
        ParityResult(
            target=MLIRBackendTarget.PTX,
            status=ParityStatus.PARITY_DEFERRED,
            findings=(),
        )


def test_parity_result_rejects_holds_with_findings():
    """A PARITY_HOLDS result must carry NO findings — a clean parity
    verdict has nothing to report."""
    with pytest.raises(ValueError, match="must carry NO"):
        ParityResult(
            target=MLIRBackendTarget.PTX,
            status=ParityStatus.PARITY_HOLDS,
            findings=("not clean",),
        )


def test_parity_result_predicates():
    holds = ParityResult(
        target=MLIRBackendTarget.PTX,
        status=ParityStatus.PARITY_HOLDS,
        findings=(),
    )
    failed = ParityResult(
        target=MLIRBackendTarget.PTX,
        status=ParityStatus.PARITY_FAILED,
        findings=("explained",),
    )
    deferred = ParityResult(
        target=MLIRBackendTarget.PTX,
        status=ParityStatus.PARITY_DEFERRED,
        findings=("toolchain absent",),
    )
    assert holds.holds() and not holds.failed() and not holds.deferred()
    assert failed.failed() and not failed.holds() and not failed.deferred()
    assert deferred.deferred() and not deferred.holds() \
        and not deferred.failed()


def test_parity_result_is_final():
    with pytest.raises(TypeError, match="final"):
        class _Subclass(ParityResult):  # type: ignore[misc]
            pass


# --------------------------------------------------------------------------
# mlir_vs_tile_ir_parity_check — input validation
# --------------------------------------------------------------------------
def test_parity_check_rejects_non_module():
    with pytest.raises(ValueError, match="must be a tile_ir.TileModule"):
        mlir_vs_tile_ir_parity_check(
            "not a module",  # type: ignore[arg-type]
            MLIRBackendTarget.PTX,
        )


def test_parity_check_rejects_non_target():
    with pytest.raises(ValueError, match="must be MLIRBackendTarget"):
        mlir_vs_tile_ir_parity_check(
            _trivial_module(),
            "ptx",  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------
# mlir_vs_tile_ir_parity_check — verdict mapping from MLIR backend
# --------------------------------------------------------------------------
def test_parity_check_defers_when_mlir_backend_defers(monkeypatch):
    """On a binding-less / toolchain-less machine the MLIR backend
    returns DEFERRED; the parity verdict is PARITY_DEFERRED with the
    deferral reason.

    The trivial module doesn't satisfy the home-grown PTX emitter
    (no @kernel function), so we mock the home-grown path to return
    a stub artifact for chunk-A's MLIR-side-only tests."""
    module = _trivial_module()
    monkeypatch.setattr(
        parity, "_run_tile_ir_path",
        lambda mod, tgt: ("stub home-grown PTX text\n", ()))
    result = mlir_vs_tile_ir_parity_check(module, MLIRBackendTarget.PTX)
    assert result.deferred()
    assert any("deferred" in f.lower() or "not on PATH" in f
               for f in result.findings), result.findings
    assert result.mlir_result is not None
    assert result.mlir_result.status() is MLIRBackendStatus.DEFERRED
    assert result.tile_ir_output == "stub home-grown PTX text\n"


def test_parity_check_fails_when_mlir_backend_fails(monkeypatch):
    """When `lower_mlir_to_backend` returns a result whose `status()`
    is FAILED, the parity gate surfaces the finding as PARITY_FAILED.

    Constructing a real FAILED MLIRBackendResult requires a PASSED
    validation (per `MLIRBackendResult.__post_init__`), which in turn
    requires the runner-registry brand. To keep chunk-A focused on
    the parity-side branch logic, this test monkeypatches the result's
    `status()` to return FAILED on a deferred-shape result; the
    underlying lowering_findings tuple is what the parity gate
    propagates regardless of how status was computed."""
    module = _trivial_module()
    monkeypatch.setattr(
        parity, "_run_tile_ir_path",
        lambda mod, tgt: ("stub home-grown PTX text\n", ()))
    deferred_shaped = MLIRBackendResult(
        target=MLIRBackendTarget.PTX,
        validation=_mock_passed_validation(),
        lowering_attempted=False,
        lowering_passed=None,
        lowering_tool=None,
        lowering_findings=("backend lowering blew up on synthetic test",),
        output_text=None,
    )
    original_status = MLIRBackendResult.status

    def _patched_status(self):
        if self is deferred_shaped:
            return MLIRBackendStatus.FAILED
        return original_status(self)

    monkeypatch.setattr(
        MLIRBackendResult, "status", _patched_status, raising=True)

    def _fake_lower(mlir_text, target, *, support=None):
        return deferred_shaped

    monkeypatch.setattr(parity, "lower_mlir_to_backend", _fake_lower)
    result = mlir_vs_tile_ir_parity_check(module, MLIRBackendTarget.PTX)
    assert result.failed(), result.findings
    assert any("backend lowering blew up" in f
               for f in result.findings), result.findings


def test_parity_check_holds_when_mlir_backend_passes(monkeypatch):
    """When `lower_mlir_to_backend` returns a result whose `status()`
    is PASSED, the parity verdict is PARITY_HOLDS with no findings.
    Chunk-B+ will replace this with a real cross-path artifact
    comparison once the home-grown side is wired in.

    Same caveat as the FAILED test: the dev machine never reaches a
    real PASSED through the runner, so we monkeypatch `status()`.

    Chunk C: when both paths PASSED, the parity gate runs the
    cross-path comparison; for PARITY_HOLDS the two artifacts must
    match after normalization. The mock here passes the same text to
    both sides so the comparison succeeds."""
    module = _trivial_module()
    homegrown_text = ".version 7.5\n.target sm_80\n.entry main() { ret; }\n"
    mlir_text = ".version 8.0\n.target sm_90\n.entry main() { ret; }\n"
    monkeypatch.setattr(
        parity, "_run_tile_ir_path",
        lambda mod, tgt: (homegrown_text, ()))
    # Construct a deferred-shaped result that still carries output_text
    # so the chunk-C comparison has something to inspect. The
    # MLIRBackendResult __post_init__ only enforces output_text on
    # PASSED results (which require the real branding); a
    # deferred-shaped result with output_text is structurally fine.
    deferred_shaped = MLIRBackendResult(
        target=MLIRBackendTarget.PTX,
        validation=_mock_passed_validation(),
        lowering_attempted=False,
        lowering_passed=None,
        lowering_tool=None,
        lowering_findings=("placeholder — status() is monkeypatched",),
        output_text=None,
    )

    # We need output_text to be the MLIR-side artifact for the chunk-C
    # comparison. Since the dataclass is frozen, monkeypatch returns a
    # property-replaced view via object.__setattr__.
    object.__setattr__(deferred_shaped, "output_text", mlir_text)

    original_status = MLIRBackendResult.status

    def _patched_status(self):
        if self is deferred_shaped:
            return MLIRBackendStatus.PASSED
        return original_status(self)

    monkeypatch.setattr(
        MLIRBackendResult, "status", _patched_status, raising=True)

    def _fake_lower(mlir_text_, target, *, support=None):
        return deferred_shaped

    monkeypatch.setattr(parity, "lower_mlir_to_backend", _fake_lower)
    result = mlir_vs_tile_ir_parity_check(module, MLIRBackendTarget.PTX)
    assert result.holds(), result.findings
    assert result.findings == ()
    assert result.mlir_result is deferred_shaped


def test_parity_check_fails_on_translator_error(monkeypatch):
    """If `emit_mlir_module` raises `MLIRTranslationError`, the parity
    gate surfaces it as a PARITY_FAILED with the translator finding —
    never silently skips."""
    module = _trivial_module()
    # Mock the home-grown path to succeed so the translator-error
    # branch is the one we're testing.
    monkeypatch.setattr(
        parity, "_run_tile_ir_path",
        lambda mod, tgt: ("stub home-grown PTX text\n", ()))

    def _fake_emit(_module):
        raise parity.MLIRTranslationError(
            "scalar.pretend_op is not yet wired")

    monkeypatch.setattr(parity, "emit_mlir_module", _fake_emit)
    result = mlir_vs_tile_ir_parity_check(module, MLIRBackendTarget.PTX)
    assert result.failed()
    assert any("MLIR translator failed" in f
               for f in result.findings), result.findings
    assert any("pretend_op" in f for f in result.findings), result.findings
    assert result.mlir_result is None
    # The home-grown output is preserved on translator failures so the
    # caller can compare what the canonical compiler produced against
    # the missing MLIR-path artifact.
    assert result.tile_ir_output == "stub home-grown PTX text\n"


# --------------------------------------------------------------------------
# Stage 215 chunk B: home-grown path runner integrated
# --------------------------------------------------------------------------
def test_parity_check_records_tile_ir_output_when_homegrown_succeeds(
        monkeypatch):
    """Chunk B: when the home-grown path produces an artifact AND the
    MLIR path returns DEFERRED, the parity verdict captures both
    pieces (the home-grown text is preserved for inspection)."""
    module = _trivial_module()
    monkeypatch.setattr(
        parity, "_run_tile_ir_path",
        lambda mod, tgt: ("HOMEGROWN-PTX-ARTIFACT\n", ()))
    result = mlir_vs_tile_ir_parity_check(
        module, MLIRBackendTarget.PTX)
    assert result.deferred()
    assert result.tile_ir_output == "HOMEGROWN-PTX-ARTIFACT\n"
    assert result.mlir_result is not None


def test_parity_check_fails_when_homegrown_fails(monkeypatch):
    """Chunk B: home-grown emitter failure surfaces as PARITY_FAILED
    BEFORE the MLIR path runs (home-grown is the canonical compiler,
    so an exception there is the most severe outcome)."""
    module = _trivial_module()
    mlir_call_count = {"n": 0}

    def _fake_emit(_mod):
        mlir_call_count["n"] += 1
        return "module { }\n"

    monkeypatch.setattr(
        parity, "_run_tile_ir_path",
        lambda mod, tgt: (None, ("PTX emitter exploded: missing kernel",)))
    monkeypatch.setattr(parity, "emit_mlir_module", _fake_emit)
    result = mlir_vs_tile_ir_parity_check(
        module, MLIRBackendTarget.PTX)
    assert result.failed()
    assert any("home-grown ptx path failed" in f
               for f in result.findings), result.findings
    assert any("PTX emitter exploded" in f
               for f in result.findings), result.findings
    # The home-grown failure short-circuits the MLIR path — it never
    # gets invoked.
    assert mlir_call_count["n"] == 0
    assert result.mlir_result is None
    assert result.tile_ir_output is None


def test_run_tile_ir_path_llvm_ir_returns_structural_deferral():
    """Chunk B contract: LLVM_IR's home-grown path takes tir.Module,
    not tile_ir.TileModule, so the parity helper returns a finding
    pointing at the Phase-D parity gate (Stage 207)."""
    module = _trivial_module()
    text, findings = parity._run_tile_ir_path(
        module, MLIRBackendTarget.LLVM_IR)
    assert text is None
    assert any("Stage 207" in f for f in findings), findings


def test_parity_check_llvm_ir_defers_with_both_sides_noted():
    """Chunk B: LLVM_IR parity returns PARITY_DEFERRED with TWO
    findings — the structural-deferral note for the home-grown side
    AND the MLIR side's status. The Stage 207 Phase-D gate is the
    canonical LLVM_IR parity check."""
    module = _trivial_module()
    result = mlir_vs_tile_ir_parity_check(
        module, MLIRBackendTarget.LLVM_IR)
    assert result.deferred()
    assert len(result.findings) == 2
    assert any("Stage 207" in f for f in result.findings), result.findings
    assert any("MLIR side for LLVM_IR returned" in f
               for f in result.findings), result.findings


# --------------------------------------------------------------------------
# Stage 215 chunk C: per-target normalization + artifact comparison
# --------------------------------------------------------------------------
def test_normalize_artifact_drops_blank_lines():
    text = "line one\n\n\n  \nline two\n"
    out = parity._normalize_artifact_text(MLIRBackendTarget.PTX, text)
    assert out == "line one\nline two"


def test_normalize_artifact_strips_line_comments_ptx():
    text = "// header\nentry main // inline\nret;\n"
    out = parity._normalize_artifact_text(MLIRBackendTarget.PTX, text)
    assert "header" not in out
    assert "inline" not in out
    assert "entry main" in out
    assert "ret;" in out


def test_normalize_artifact_strips_line_comments_llvm():
    text = "; pre\ndefine void @k() {\n  ret void ; trailing\n}\n"
    out = parity._normalize_artifact_text(
        MLIRBackendTarget.LLVM_IR, text)
    assert "pre" not in out
    assert "trailing" not in out
    assert "define void @k()" in out
    assert "ret void" in out


def test_normalize_artifact_drops_cosmetic_prefixes_ptx():
    text = (
        ".version 7.5\n"
        ".target sm_80\n"
        ".address_size 64\n"
        ".visible .entry main() { ret; }\n"
    )
    out = parity._normalize_artifact_text(MLIRBackendTarget.PTX, text)
    assert ".version" not in out
    assert ".target" not in out
    assert ".address_size" not in out
    assert ".entry main" in out


def test_normalize_artifact_collapses_whitespace_runs():
    text = "kernel    void    main(    int    *    p   )    {  }\n"
    out = parity._normalize_artifact_text(
        MLIRBackendTarget.METAL_MSL, text)
    assert out == "kernel void main( int * p ) { }"


def test_normalize_artifact_rejects_non_str():
    with pytest.raises(ValueError, match="must be str"):
        parity._normalize_artifact_text(
            MLIRBackendTarget.PTX,
            b"bytes",  # type: ignore[arg-type]
        )


def test_compare_artifacts_match_after_normalization():
    a = ".version 7.5\n.target sm_80\n.entry main() { ret; }\n"
    b = ".version 9.0\n.target sm_90\n.entry main() { ret; }\n"
    match, summary = parity._compare_artifacts(
        MLIRBackendTarget.PTX, a, b)
    assert match
    assert summary is None


def test_compare_artifacts_mismatch_emits_diff_summary():
    a = ".entry main() { ret; }\n"
    b = ".entry kernel_main() { ret; }\n"
    match, summary = parity._compare_artifacts(
        MLIRBackendTarget.PTX, a, b)
    assert not match
    assert summary is not None
    assert "first divergence at line" in summary
    assert "tile-IR=" in summary
    assert "MLIR=" in summary


def test_compare_artifacts_different_line_counts():
    a = ".entry main() { ret; }\n"
    b = ".entry main() {\n  ret;\n}\n"
    match, summary = parity._compare_artifacts(
        MLIRBackendTarget.PTX, a, b)
    # After normalization these likely still differ in line count.
    assert not match
    assert summary is not None
    assert "lines" in summary


def test_compare_artifacts_rejects_bad_inputs():
    with pytest.raises(ValueError, match="must be"):
        parity._compare_artifacts(
            "ptx", "a", "b")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="both texts must be str"):
        parity._compare_artifacts(
            MLIRBackendTarget.PTX,
            "a",
            b"b",  # type: ignore[arg-type]
        )


def test_parity_check_fails_on_artifact_mismatch(monkeypatch):
    """Chunk C: when both paths PASSED but the normalized artifacts
    differ, PARITY_FAILED with a clear diff summary."""
    module = _trivial_module()
    homegrown_text = ".visible .entry main() { ret; }\n"
    mlir_text = ".visible .entry not_main() { ret; }\n"
    monkeypatch.setattr(
        parity, "_run_tile_ir_path",
        lambda mod, tgt: (homegrown_text, ()))
    deferred_shaped = MLIRBackendResult(
        target=MLIRBackendTarget.PTX,
        validation=_mock_passed_validation(),
        lowering_attempted=False,
        lowering_passed=None,
        lowering_tool=None,
        lowering_findings=("placeholder",),
        output_text=None,
    )
    object.__setattr__(deferred_shaped, "output_text", mlir_text)
    original_status = MLIRBackendResult.status

    def _patched_status(self):
        if self is deferred_shaped:
            return MLIRBackendStatus.PASSED
        return original_status(self)

    monkeypatch.setattr(
        MLIRBackendResult, "status", _patched_status, raising=True)
    monkeypatch.setattr(
        parity, "lower_mlir_to_backend",
        lambda mlir, t, *, support=None: deferred_shaped)
    result = mlir_vs_tile_ir_parity_check(
        module, MLIRBackendTarget.PTX)
    assert result.failed(), result.findings
    assert any("normalized artifact mismatch" in f
               for f in result.findings), result.findings
    assert any("first divergence" in f
               for f in result.findings), result.findings


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _mock_passed_validation() -> MLIRValidation:
    return MLIRValidation(
        MLIRValidationVerdict.DEFERRED,
        ("mock deferred — used only by parity tests for FAILED path",),
    )


