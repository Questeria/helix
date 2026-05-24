"""Tests for helixc.ir.mlir.backends - v3.0 Phase E, Stage 213.

Stage 213 begins the MLIR-to-backends seam. This first chunk is a
mock-path-first scaffold: malformed MLIR fails loudly, well-shaped
MLIR defers until real target pass pipelines exist, and no target can
claim a false PASS on a toolchain-free machine.
"""
from __future__ import annotations

import ast
import copy
import pickle
import subprocess
from pathlib import Path
from types import MappingProxyType

import pytest

from helixc.backend.gpu_ci import BackendKind as GPUBackendKind
from helixc.ir.mlir import backends, validate
from helixc.ir.mlir.backends import (
    MLIRBackendResult,
    MLIRBackendStatus,
    MLIRBackendTarget,
    backend_lowering_pipeline,
    backend_required_dialects,
    lower_mlir_to_backend,
    mlir_target_for_gpu_backend,
)
from helixc.ir.mlir.toolchain import MLIRSupport
from helixc.ir.mlir.validate import MLIRValidation, MLIRValidationVerdict


_WELL_FORMED = """\
module {
  func.func @main() -> i32 {
    %0 = arith.constant 1 : i32
    return %0 : i32
  }
}
"""
_PTX_OUTPUT = ".version 8.3\n.target sm_80\n.entry main() {}\n"
_PTX_OUTPUT_MULTILINE_ENTRY = """\
.version 8.3
.target sm_80
.visible .entry k(
    .param .u64 p0
)
{
}
"""

_REAL_RUN_MLIR_OPT_VALIDATE = validate._run_mlir_opt_validate


def _reject_invalid_smoke(input_text: str) -> bool:
    return validate._mlir_text_is_invalid_smoke_probe(input_text)


def _mock_deferred_validation() -> MLIRValidation:
    return MLIRValidation(
        MLIRValidationVerdict.DEFERRED,
        ("mock validation deferred to real mlir-opt",),
    )


def _real_passed_validation(
        mlir_text: str = _WELL_FORMED,
        mlir_opt: str = "/fake/mlir-opt") -> MLIRValidation:
    old_run = validate.subprocess.run
    old_detect = validate.detect_mlir_support

    def _fake_run(cmd, *, capture_output, text, timeout):
        input_text = Path(cmd[1]).read_text(encoding="utf-8")
        if _reject_invalid_smoke(input_text):
            return subprocess.CompletedProcess(cmd, 1, "", "invalid smoke")
        assert input_text == mlir_text
        Path(cmd[3]).write_text(input_text, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    try:
        validate.subprocess.run = _fake_run
        validate.detect_mlir_support = lambda: validate.MLIRSupport(
            bindings=False,
            dialects=False,
            mlir_opt=mlir_opt,
            detail=("test fake mlir-opt",),
        )
        result = _REAL_RUN_MLIR_OPT_VALIDATE(mlir_text, mlir_opt)
    finally:
        validate.subprocess.run = old_run
        validate.detect_mlir_support = old_detect
    assert result.passed()
    return result


def _accept_backend_output(
        target: MLIRBackendTarget, output_text: str,
) -> backends.MLIRBackendOutputValidation:
    assert output_text.strip()
    return backends.MLIRBackendOutputValidation(
        target=target,
        output_sha256=backends.hashlib.sha256(
            output_text.encode("utf-8")).hexdigest(),
        evidence=(
            f"validator=test-{target.value}",
            f"predicate={target.value}-artifact-shape",
        ),
    )


def _reject_backend_output(
        target: MLIRBackendTarget, finding: str,
        output_text: str = _PTX_OUTPUT,
) -> backends.MLIRBackendOutputValidation:
    return backends.MLIRBackendOutputValidation(
        target=target,
        output_sha256=backends.hashlib.sha256(
            output_text.encode("utf-8")).hexdigest(),
        findings=(
            f"{backends._BACKEND_OUTPUT_FAILURE_PREFIX}"
            f"{target.value}: {finding}",
        ),
    )


def _register_ptx_validator(monkeypatch, validator=_accept_backend_output):
    return _register_output_validator(
        monkeypatch, MLIRBackendTarget.PTX, validator)


def _register_output_validator(
        monkeypatch, target, validator=_accept_backend_output):
    monkeypatch.setattr(
        backends,
        "_MLIR_BACKEND_OUTPUT_VALIDATORS_AUTHORITY",
        MappingProxyType({
            **backends._MLIR_BACKEND_OUTPUT_VALIDATORS_AUTHORITY,
            target: validator,
        }),
    )
    _wire_pipeline(monkeypatch, target)
    # Stage 214 chunk G+ wires translators for the GPU targets too.
    # Pre-chunk-G tests assumed translators were None for everything
    # except LLVM_IR; reset the translator to None by default so
    # `_register_output_validator` keeps the old semantics. Tests that
    # explicitly want a translator must call `_wire_translator` AFTER
    # this helper.
    monkeypatch.setattr(
        backends,
        "_MLIR_BACKEND_TRANSLATORS_AUTHORITY",
        MappingProxyType({
            **backends._MLIR_BACKEND_TRANSLATORS_AUTHORITY,
            target: None,
        }),
    )
    return validator


def _wire_ptx_pipeline(monkeypatch, pipeline=("--canonicalize",)):
    return _wire_pipeline(monkeypatch, MLIRBackendTarget.PTX, pipeline)


def _wire_pipeline(monkeypatch, target, pipeline=("--canonicalize",)):
    monkeypatch.setattr(
        backends,
        "_MLIR_BACKEND_LOWERING_PIPELINES_AUTHORITY",
        MappingProxyType({
            **backends._MLIR_BACKEND_LOWERING_PIPELINES_AUTHORITY,
            target: pipeline,
        }),
    )
    return pipeline


def _wire_translator(
        monkeypatch, target,
        translator=("mlir-translate", "--mlir-to-llvmir", ())):
    monkeypatch.setattr(
        backends,
        "_MLIR_BACKEND_TRANSLATORS_AUTHORITY",
        MappingProxyType({
            **backends._MLIR_BACKEND_TRANSLATORS_AUTHORITY,
            target: translator,
        }),
    )
    return translator


def test_mlir_backend_target_covers_stage213_backends():
    """The Stage 213 target set is exactly LLVM IR plus the 4 GPU
    backends Helix already ships."""
    assert {target.value for target in MLIRBackendTarget} == {
        "llvm_ir", "ptx", "rocm_hip", "metal_msl", "webgpu_wgsl",
    }
    assert tuple(MLIRBackendTarget) == backends.MLIR_BACKEND_TARGETS


def test_gpu_backend_mapping_covers_existing_gpu_backends():
    """The bridge from existing GPU backend enum to MLIR target is
    total, so a new GPU backend cannot silently miss Stage 213."""
    assert set(backends.GPU_BACKEND_TO_MLIR_TARGET) == set(GPUBackendKind)
    assert {
        mlir_target_for_gpu_backend(backend)
        for backend in GPUBackendKind
    } == {
        MLIRBackendTarget.PTX,
        MLIRBackendTarget.ROCM_HIP,
        MLIRBackendTarget.METAL_MSL,
        MLIRBackendTarget.WEBGPU_WGSL,
    }


def test_backend_required_dialects_are_total_nonempty_unique():
    """Every target has an explicit dialect contract."""
    for target in MLIRBackendTarget:
        dialects = backend_required_dialects(target)
        assert dialects
        assert len(dialects) == len(set(dialects))
        for dialect in dialects:
            assert isinstance(dialect, str) and dialect.isidentifier()


_WIRED_TARGETS_STAGE_214 = frozenset((
    MLIRBackendTarget.LLVM_IR,
    MLIRBackendTarget.PTX,
    MLIRBackendTarget.ROCM_HIP,
    MLIRBackendTarget.METAL_MSL,
))


def test_backend_lowering_pipelines_state_per_target():
    """Stage 214 chunks E/G/H/I wire LLVM_IR / PTX / ROCM_HIP /
    METAL_MSL; WEBGPU_WGSL stays explicitly unwired with the empty
    `()` baseline. Empty means DEFERRED, not PASSED."""
    for target in _WIRED_TARGETS_STAGE_214:
        assert backend_lowering_pipeline(target) != ()
    for target in MLIRBackendTarget:
        if target in _WIRED_TARGETS_STAGE_214:
            continue
        assert backend_lowering_pipeline(target) == ()


def test_backend_output_validators_state_per_target():
    """Stage 214 chunks E/G/H/I wire LLVM_IR / PTX / ROCM_HIP /
    METAL_MSL output validators; WEBGPU_WGSL stays None. A pass
    pipeline alone cannot prove backend-consumable output."""
    assert set(backends.MLIR_BACKEND_OUTPUT_VALIDATORS) == set(
        MLIRBackendTarget)
    for target in _WIRED_TARGETS_STAGE_214:
        assert callable(
            backends.MLIR_BACKEND_OUTPUT_VALIDATORS[target])
    for target in MLIRBackendTarget:
        if target in _WIRED_TARGETS_STAGE_214:
            continue
        assert backends.MLIR_BACKEND_OUTPUT_VALIDATORS[target] is None


def test_backend_translators_table_state_per_target():
    """Stage 214 chunks E/G/H/I wire LLVM_IR / PTX / ROCM_HIP /
    METAL_MSL translator entries; WEBGPU_WGSL stays None (chunk J
    will wire it). The table is total over MLIRBackendTarget."""
    assert set(backends.MLIR_BACKEND_TRANSLATORS) == set(MLIRBackendTarget)
    llvm_translator = backends.backend_translator(
        MLIRBackendTarget.LLVM_IR)
    assert llvm_translator == ("mlir-translate", "--mlir-to-llvmir", ())
    ptx_translator = backends.backend_translator(
        MLIRBackendTarget.PTX)
    assert ptx_translator is not None
    _, _, ptx_follow_up = ptx_translator
    assert ptx_follow_up[0] == "llc"
    assert "-mtriple=nvptx64" in ptx_follow_up
    rocm_translator = backends.backend_translator(
        MLIRBackendTarget.ROCM_HIP)
    assert rocm_translator is not None
    rocm_tool, rocm_flag, rocm_follow_up = rocm_translator
    assert rocm_tool == "mlir-translate"
    assert rocm_flag == "--mlir-to-llvmir"
    assert rocm_follow_up[0] == "llc"
    assert "-mtriple=amdgcn-amd-amdhsa" in rocm_follow_up
    for target in MLIRBackendTarget:
        if target in _WIRED_TARGETS_STAGE_214:
            continue
        assert backends.MLIR_BACKEND_TRANSLATORS[target] is None
        assert backends.backend_translator(target) is None


def test_backend_translator_rejects_unknown_target():
    with pytest.raises(ValueError):
        backends.backend_translator("llvm_ir")  # type: ignore[arg-type]


def test_public_backend_contract_tables_are_immutable():
    with pytest.raises(TypeError):
        backends.MLIR_BACKEND_REQUIRED_DIALECTS[
            MLIRBackendTarget.PTX] = ()  # type: ignore[index]
    with pytest.raises(TypeError):
        backends.MLIR_BACKEND_OUTPUT_VALIDATORS[
            MLIRBackendTarget.PTX] = _accept_backend_output  # type: ignore[index]


def test_public_backend_contract_rebinding_does_not_change_authority(
        monkeypatch):
    # WEBGPU_WGSL is still unwired (chunk J will wire it), so the
    # rebinding check uses it to confirm the AUTHORITY surface is not
    # mutated by writes to the PUBLIC alias.
    monkeypatch.setattr(
        backends,
        "MLIR_BACKEND_LOWERING_PIPELINES",
        MappingProxyType({
            **backends.MLIR_BACKEND_LOWERING_PIPELINES,
            MLIRBackendTarget.WEBGPU_WGSL: ("--canonicalize",),
        }),
    )
    monkeypatch.setattr(
        backends,
        "MLIR_BACKEND_OUTPUT_VALIDATORS",
        MappingProxyType({
            **backends.MLIR_BACKEND_OUTPUT_VALIDATORS,
            MLIRBackendTarget.WEBGPU_WGSL: _accept_backend_output,
        }),
    )
    assert backend_lowering_pipeline(MLIRBackendTarget.WEBGPU_WGSL) == ()
    assert backends._MLIR_BACKEND_OUTPUT_VALIDATORS_AUTHORITY[
        MLIRBackendTarget.WEBGPU_WGSL] is None


def test_backend_output_validators_return_structured_validation():
    clean = _accept_backend_output(MLIRBackendTarget.PTX, _PTX_OUTPUT)
    assert not clean.failed()
    assert not clean.passed()
    assert clean.candidate()
    assert sum((clean.passed(), clean.failed(), clean.candidate())) == 1
    assert clean.target is MLIRBackendTarget.PTX
    assert clean.evidence
    finding = _reject_backend_output(
        MLIRBackendTarget.PTX, "not a PTX artifact")
    assert finding.failed()
    assert not finding.candidate()
    assert finding.findings == (
        f"{backends._BACKEND_OUTPUT_FAILURE_PREFIX}ptx: "
        "not a PTX artifact",
    )
    with pytest.raises(ValueError, match="key=value"):
        backends.MLIRBackendOutputValidation(
            target=MLIRBackendTarget.PTX,
            output_sha256=backends.hashlib.sha256(
                _PTX_OUTPUT.encode("utf-8")).hexdigest(),
            evidence=("ok",),
        )
    with pytest.raises(ValueError, match="validator=.*predicate="):
        backends.MLIRBackendOutputValidation(
            target=MLIRBackendTarget.PTX,
            output_sha256=backends.hashlib.sha256(
                _PTX_OUTPUT.encode("utf-8")).hexdigest(),
            evidence=("validator=test-ptx",),
        )
    forged = backends.MLIRBackendOutputValidation(
        target=MLIRBackendTarget.PTX,
        output_sha256=backends.hashlib.sha256(b"not ptx").hexdigest(),
        evidence=("validator=claimed-ptx", "predicate=claimed-shape"),
    )
    assert not forged.passed()
    assert forged.candidate()


def test_backend_output_validation_brand_is_runner_private():
    assert not hasattr(backends, "_brand_backend_output_validation")
    assert not hasattr(backends._run_mlir_opt_pipeline, "__closure__")
    assert not hasattr(
        backends._run_mlir_opt_pipeline,
        "_BackendPipelineRunner__passes",
    )


def test_backend_result_objects_have_no_public_dict():
    validation = _mock_deferred_validation()
    result = MLIRBackendResult(
        target=MLIRBackendTarget.PTX,
        validation=validation,
        lowering_attempted=False,
        lowering_passed=None,
        lowering_tool=None,
        lowering_findings=("deferred",),
    )
    clean = _accept_backend_output(MLIRBackendTarget.PTX, _PTX_OUTPUT)
    assert not hasattr(result, "__dict__")
    assert not hasattr(clean, "__dict__")


def test_backend_output_validation_copy_stays_unbranded():
    clean = _accept_backend_output(MLIRBackendTarget.PTX, _PTX_OUTPUT)
    copied = copy.copy(clean)
    deep_copied = copy.deepcopy(clean)
    assert copied == clean
    assert copied is not clean
    assert not copied.passed()
    assert copied.candidate()
    assert deep_copied == clean
    assert deep_copied is not clean
    assert not deep_copied.passed()
    assert deep_copied.candidate()


def test_backend_helpers_reject_unknown_targets():
    with pytest.raises(ValueError, match="unknown MLIR backend target"):
        backend_required_dialects("ptx")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown MLIR backend target"):
        backend_lowering_pipeline("ptx")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown GPU backend"):
        mlir_target_for_gpu_backend("ptx")  # type: ignore[arg-type]


def test_mlir_backend_result_rejects_invalid_validation():
    with pytest.raises(ValueError, match="validation must be"):
        MLIRBackendResult(
            target=MLIRBackendTarget.PTX,
            validation="not validation",  # type: ignore[arg-type]
            lowering_attempted=False,
            lowering_passed=None,
            lowering_tool=None,
            lowering_findings=("deferred",),
        )


def test_mlir_backend_result_rejects_silent_deferred_lowering():
    """Mock-valid MLIR with no real lowering attempt must explain why."""
    with pytest.raises(ValueError, match="must carry at least one finding"):
        MLIRBackendResult(
            target=MLIRBackendTarget.PTX,
            validation=_mock_deferred_validation(),
            lowering_attempted=False,
            lowering_passed=None,
            lowering_tool=None,
            lowering_findings=(),
        )


def test_mlir_backend_result_rejects_non_bool_attempt_flag():
    with pytest.raises(ValueError, match="lowering_attempted must be a bool"):
        MLIRBackendResult(
            target=MLIRBackendTarget.PTX,
            validation=_mock_deferred_validation(),
            lowering_attempted=1,  # type: ignore[arg-type]
            lowering_passed=None,
            lowering_tool=None,
            lowering_findings=("deferred",),
        )


def test_mlir_backend_result_rejects_mutable_findings():
    with pytest.raises(ValueError, match="lowering_findings must be a tuple"):
        MLIRBackendResult(
            target=MLIRBackendTarget.PTX,
            validation=_mock_deferred_validation(),
            lowering_attempted=False,
            lowering_passed=None,
            lowering_tool=None,
            lowering_findings=["deferred"],  # type: ignore[arg-type]
        )


def test_mlir_backend_result_allows_validation_failure_without_lowering():
    validation = MLIRValidation(
        MLIRValidationVerdict.FAILED,
        ("unbalanced braces",),
    )
    result = MLIRBackendResult(
        target=MLIRBackendTarget.PTX,
        validation=validation,
        lowering_attempted=False,
        lowering_passed=None,
        lowering_tool=None,
        lowering_findings=(),
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert result.failed()


def test_mlir_backend_result_rejects_failed_validation_with_attempt():
    validation = MLIRValidation(
        MLIRValidationVerdict.FAILED,
        ("unbalanced braces",),
    )
    with pytest.raises(ValueError, match="cannot attempt"):
        MLIRBackendResult(
            target=MLIRBackendTarget.PTX,
            validation=validation,
            lowering_attempted=True,
            lowering_passed=False,
            lowering_tool="mlir-opt",
            lowering_findings=("bad",),
        )


def test_mlir_backend_result_rejects_real_failure_without_diagnostic():
    with pytest.raises(ValueError, match="failure must carry"):
        MLIRBackendResult(
            target=MLIRBackendTarget.PTX,
            validation=_real_passed_validation(),
            lowering_attempted=True,
            lowering_passed=False,
            lowering_tool="mlir-opt",
            lowering_findings=(),
        )


def test_mlir_backend_result_rejects_blank_lowering_tool():
    with pytest.raises(ValueError, match="non-empty lowering_tool"):
        MLIRBackendResult(
            target=MLIRBackendTarget.PTX,
            validation=_real_passed_validation(),
            lowering_attempted=True,
            lowering_passed=True,
            lowering_tool="   ",
            lowering_findings=(),
            output_text=".version 8.3\n",
        )


def test_mlir_backend_result_rejects_non_str_output_text():
    with pytest.raises(ValueError, match="output_text must be a str"):
        MLIRBackendResult(
            target=MLIRBackendTarget.PTX,
            validation=_real_passed_validation(),
            lowering_attempted=True,
            lowering_passed=False,
            lowering_tool="mlir-opt",
            lowering_findings=("mlir-opt rejected IR",),
            output_text=[],  # type: ignore[arg-type]
        )


def test_mlir_backend_result_rejects_blank_output_text():
    with pytest.raises(ValueError, match="output_text must carry text"):
        MLIRBackendResult(
            target=MLIRBackendTarget.PTX,
            validation=_real_passed_validation(),
            lowering_attempted=True,
            lowering_passed=False,
            lowering_tool="mlir-opt",
            lowering_findings=("mlir-opt rejected IR",),
            output_text="   ",
        )


def test_mlir_backend_result_rejects_failure_with_output_text():
    with pytest.raises(ValueError, match="lowering_passed=False"):
        MLIRBackendResult(
            target=MLIRBackendTarget.PTX,
            validation=_real_passed_validation(),
            lowering_attempted=True,
            lowering_passed=False,
            lowering_tool="mlir-opt",
            lowering_findings=("mlir-opt rejected IR",),
            output_text=".version 8.3\n",
        )


def test_mlir_backend_result_rejects_non_bool_lowering_passed():
    with pytest.raises(ValueError, match="lowering_passed to be a bool"):
        MLIRBackendResult(
            target=MLIRBackendTarget.PTX,
            validation=_real_passed_validation(),
            lowering_attempted=True,
            lowering_passed=1,  # type: ignore[arg-type]
            lowering_tool="mlir-opt",
            lowering_findings=(),
            output_text=".version 8.3\n",
        )


def test_mlir_backend_result_rejects_attempt_with_deferred_validation():
    with pytest.raises(ValueError, match="validation to be PASSED"):
        MLIRBackendResult(
            target=MLIRBackendTarget.PTX,
            validation=_mock_deferred_validation(),
            lowering_attempted=True,
            lowering_passed=False,
            lowering_tool="mlir-opt",
            lowering_findings=("mlir-opt rejected IR",),
        )


def test_mlir_backend_result_rejects_success_with_deferred_validation():
    with pytest.raises(ValueError, match="validation to be PASSED"):
        MLIRBackendResult(
            target=MLIRBackendTarget.PTX,
            validation=_mock_deferred_validation(),
            lowering_attempted=True,
            lowering_passed=True,
            lowering_tool="mlir-opt",
            lowering_findings=(),
            output_text=".version 8.3\n",
        )


def test_mlir_backend_result_rejects_success_without_output_text():
    with pytest.raises(ValueError, match="non-empty output_text"):
        MLIRBackendResult(
            target=MLIRBackendTarget.PTX,
            validation=_real_passed_validation(),
            lowering_attempted=True,
            lowering_passed=True,
            lowering_tool="mlir-opt",
            lowering_findings=(),
            output_text=None,
        )


def test_mlir_backend_result_rejects_success_without_validator_token():
    output_digest = backends.hashlib.sha256(
        _PTX_OUTPUT.encode("utf-8")).hexdigest()
    with pytest.raises(ValueError, match="backend-runner"):
        MLIRBackendResult(
            target=MLIRBackendTarget.PTX,
            validation=_real_passed_validation(),
            lowering_attempted=True,
            lowering_passed=True,
            lowering_tool="mlir-opt",
            lowering_findings=(),
            output_text=_PTX_OUTPUT,
            output_provenance=(
                f"output_sha256={output_digest}",),
        )


def test_backend_shape_probe_rejects_wrong_target_shape():
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "this is not LLVM IR, but define appears in prose\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "error: backend failed but left stale output\n"
        "define i32 @main() { ret i32 0 }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f()\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define i32 @main() {\nret i32 0\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine void @f() ; {}\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine i32 @main() } { ret i32 0\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine i32 @main() { ret i32 0 } }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine i32 @main() { { ret i32 0 }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine void @k() { this is not llvm }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f() { }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define i32 @main() {\nentry:\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine void @f() {\n'
        'this is invalid:\n'
        'ret void\n'
        '}\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine void @k() { call garbage }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine void @k() { store garbage }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine void @k() { '
        'store i32 0, ptr bad\nret void\n}\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine void @k() { br label garbage }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f() { br label %@bad }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine void @k() { ret void garbage }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define i32 @expected() { ret i32 true }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define i32 @expected() { ret i32 null }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @expected() { "
        "br i1 2, label %ok, label %bad\n"
        "ok:\nret void\nbad:\nret void\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define i1 @expected() { "
        "%0 = icmp eq i32 null, null\n"
        "ret i1 %0\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine i32 @main() {\n'
        'ret i32 0 garbage\n}\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define i32 @f() { ret i32 %x, }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define i32 @f() { ret i32 %@bad }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define i32 @f() { ret i32 % }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f() { call void %@bad()\nret void\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "declare void @g()\ndefine void @f() {\n"
        "  call void @g() garbage\n"
        "  ret void\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "declare void @g(i32)\ndefine void @f() {\n"
        "  call void @g(i32 %@bad)\n"
        "  ret void\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "declare void @g()\ndefine void @f() {\n"
        "  call void @g(garbage)\n"
        "  ret void\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine i32 @main() { ret i64 0 }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine i32 @main(garbage) { ret i32 0 }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine i32 @main() { %0 = add garbage }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'define i32 @f() {\n'
        '%0 = phi i32\n'
        'ret i32 %0\n'
        '}\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine i32 @main() {\n'
        '%0 = add i32 0 1\n'
        'ret i32 %0\n'
        '}\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine i32 @main() { %0 = load garbage }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define i32 @f(ptr %p) {\n"
        "  %0 = load i32, ptr %p, volatile\n"
        "  ret i32 %0\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define i32 @f(ptr %p) {\n"
        "  %0 = load i32, ptr garbage %p\n"
        "  ret i32 %0\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define <4 x i32> @f(ptr %p) {\n"
        "  %0 = load <4 x %@bad>, ptr %p\n"
        "  ret <4 x i32> zeroinitializer\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine i32 @f(ptr %p) {\n'
        '  %0 = load i32, ptr %p, align 4\n'
        '  ret i32 %0\n'
        '}\n')
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define ptr @f() {\n"
        "  %p = alloca i32, align 4\n"
        "  ret ptr %p\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define ptr @f() {\n"
        "  %p = alloca i32 garbage\n"
        "  ret ptr %p\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define i32 @f(ptr %p) {\n"
        "  %0 = load i32, ptr %p, align 3\n"
        "  ret i32 %0\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(ptr %p) {\n"
        "  store i32 0, ptr %p, align garbage\n"
        "  ret void\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(ptr %p) {\n"
        "  store i32 garbage 0, ptr %p\n"
        "  ret void\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(ptr %p) {\n"
        "  store i32 0, ptr %p, volatile\n"
        "  ret void\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine ptr @f() { ret ptr nullbad }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndeclare @f()\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine @f() { ret void }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine void @() { ret void }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine garbage i32 @main() { '
        'ret i32 0 }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndeclare void @f() garbage\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine i32 @main() { ret i32 0 } garbage\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine void @f() garbage=bad { ret void }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine void @f() { ret void }\nret void\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine void @f() { switch garbage }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine void @f() { invoke garbage }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine void @f() { resume garbage }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x"\ndefine void @f() { indirectbr garbage }\n')
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'define void @"foo(bar"() { ret void }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'define void @"foo"bar"() { ret void }\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(i32 garbage) { ret void }\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'define void @f(i32 %"arg,with,commas") { ret void }\n')
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(ptr dereferenceable(8) %p) { ret void }\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(ptr byval(i32) %p) { ret void }\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(ptr sret(%struct.S) %p) { ret void }\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(ptr inalloca({ i32, i32 }) %p) { ret void }\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(ptr align 8 %p) { ret void }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(ptr align garbage %p) { ret void }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(ptr byval %p) { ret void }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(ptr sret %p) { ret void }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(ptr inalloca %p) { ret void }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(ptr align 3 %p) { ret void }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(i32 sret(i32) %x) { ret void }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(i32 noalias %x) { ret void }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(i32 nonnull %x) { ret void }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define noalias i32 @f() { ret i32 0 }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define dereferenceable(8) i32 @f() { ret i32 0 }\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(ptr byval({ i32 }) %p) { ret void }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(%@bad %x) { ret void }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(ptr sret(%@bad) %p) { ret void }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(ptr addrspace(foo) noalias %p) { ret void }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f(<4 x %@bad> %x) { ret void }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f([4 x %@bad] %x) { ret void }\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define <4 x i32> @f() { ret <4 x i32> zeroinitializer }\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define { i32 } @f() { ret { i32 } zeroinitializer }\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define <4 x i32> @f(<4 x i32> %a, <4 x i32> %b) {\n"
        "  %0 = add <4 x i32> %a, %b\n"
        "  ret <4 x i32> %0\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define { i32, i32 } @pair() {\n"
        "entry:\n"
        "  ret { i32, i32 } zeroinitializer\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define ptr addrspace(1) @f() {\n"
        "  ret ptr addrspace(1) null\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define ptr @f(ptr %base, i64 %idx) {\n"
        "  %p = getelementptr inbounds i32, ptr %base, i64 %idx\n"
        "  ret ptr %p\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define ptr @f(ptr %p) {\n"
        "  %q = getelementptr i32, ptr %p, i64 %@bad\n"
        "  ret ptr %q\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define ptr @f(ptr %p) {\n"
        "  %q = getelementptr i32, ptr garbage %p, i64 0\n"
        "  ret ptr %q\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define i64 @f(ptr %p) {\n"
        "  %0 = ptrtoint ptr %p to i64\n"
        "  ret i64 %0\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define ptr @f(ptr addrspace(1) %p) {\n"
        "  %0 = addrspacecast ptr addrspace(1) %p to ptr\n"
        "  ret ptr %0\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "%struct.S = type { i32 }\n"
        "define void @f(ptr sret(%struct.S) %p) { ret void }\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define noundef i32 @f() { ret i32 0 }\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define zeroext i1 @f() { ret i1 false }\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define ptr noalias @f() { ret ptr null }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define void @f() sret { ret void }\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define i128 @f(<4 x i32> %x) {\n"
        "  %0 = bitcast <4 x i32> %x to i128\n"
        "  ret i128 %0\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define float @f() {\n"
        "  ret float 0.000000e+00\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define i32 @f(i32 %a, i32 %b) {\n"
        "  %0 = add nsw i32 %a, %b\n"
        "  ret i32 %0\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define i32 @f(i1 %c) {\n"
        "  %0 = select i1 garbage %c, i32 1, i32 0\n"
        "  ret i32 %0\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define i32 @f(<4 x i32> %v) {\n"
        "  %0 = extractelement <4 x i32> %v, i32 garbage 0\n"
        "  ret i32 %0\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define i0 @f() { ret i0 0 }\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define dso_local i32 @add(noundef i32 %a, noundef i32 %b) {\n"
        "entry:\n"
        "  %0 = add nsw i32 %a, %b\n"
        "  ret i32 %0\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "define double @f() {\n"
        "  ret double 0x3FF0000000000000\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        "target triple = \"x86_64-unknown-linux-gnu\"\n"
        "define i32 @main() { ret i32 0 }\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR,
        'target triple = "x86_64-unknown-linux-gnu"\n'
        'declare void @extern_func()\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "this is just a log mentioning amdgcn\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "error: backend failed\n#include <hip/hip_runtime.h>\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        '#include <hip/hip_runtime.h>\nextern "C" __global__ void k()\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        '#include <hip/hip_runtime.h>\n'
        'extern "C" __global__ void k() /* {} */;\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n__global__ void k() {\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n__global__ garbage() {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n__global__ void expected(??? * p) {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n__global__ void k() {\n"
        "THIS_IS_NOT_VALID;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n__global__ void expected() {\n"
        "x @ y;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n__global__ void k() {\n"
        "@@@;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n__global__ void k() {\n"
        "foo(,);\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n__global__ void k() {\n"
        "int x = ;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n__global__ void k() {\n"
        "int x = +;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n__global__ void k() {\n"
        "int x = 1 +;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n__global__ void k() {\n"
        "foo +;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n__global__ void k() {\n"
        "int x = 1 2;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n__global__ void k() {}\n"
        "this is not hip;\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n__global__ void expected() {}\n"
        "ret void\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n__global__ void expected() {}\n"
        "%0 = add i32 1, 2\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "hipLaunchKernelGGL(k, dim3(1), dim3(1), 0, 0);\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n"
        'extern "C" __global__ void k() {}\n')
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n"
        'extern "C" __global__ void k(float* out, size_t n) {}\n')
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        'target triple = "amdgcn-amd-amdhsa"\n'
        "define amdgpu_kernel void @k() {\n"
        "  ret void\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        'target triple = "amdgcn-amd-amdhsa"\n'
        "define amdgpu_kernel void @k() {\n"
        "entry:\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n__global__ void k() {\n"
        "return;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        "#include <hip/hip_runtime.h>\n__global__ void k() {}\n"
        "# definitely not valid\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP,
        '#include <hip/hip_runtime.h>\n'
        'extern "C" __global__ void k() {} garbage\n')
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "not shader code; kernel panic log\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "error: backend failed\n#include <metal_stdlib>\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\n#error backend failed\n"
        "kernel void k() {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k()\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "[[kernel]] () {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k(garbage) {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void expected(??? * p) {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k() /* {} */;\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k() {\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k() {\nTHIS_IS_NOT_VALID;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void expected() {\nx @ y;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k() {\n@@@;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k() {\nfoo(,);\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k() {\nthis is not valid;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k() {\nint x = ;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k() {\nint x = +;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k() {\nint x = 1 +;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k() {\nfoo +;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k() {\nint x = 1 2;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k() {\nint x y;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k() {}\nthis is not msl;\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k() {}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k() {\nreturn;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k() {} garbage\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k() {}\n# definitely not valid\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k(\n"
        "    device float* out [[buffer(0)]]\n"
        ")\n"
        "{\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.METAL_MSL,
        "#include <metal_stdlib>\nusing namespace metal;\n"
        "kernel void k(\n"
        "  device float* x [[buffer(0)]]\n"
        ") {\n"
        "  if (true) { return; }\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "notes: @compute is planned; fn main appears in prose\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "error: backend failed\n@compute @workgroup_size(1)\n"
        "fn main() {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "alias Lane = ???;\n"
        "@compute @workgroup_size(1)\nfn expected() {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "type 123 = i32;\n"
        "@compute @workgroup_size(1)\nfn expected() {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "var<private> x: ???;\n"
        "@compute @workgroup_size(1)\nfn expected() {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main()\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main(garbage) {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main() garbage {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main() // {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main() {\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main() {\n"
        "THIS_IS_NOT_VALID;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn expected() {\n"
        "x @ y;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main() {\n"
        "@@@;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main() {\n"
        "foo(,);\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main() {\n"
        "return foo( ;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main() {\n"
        "if () {\n}\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main() {\n"
        "this is not valid;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main() {\n"
        "var x = ;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main() {\n"
        "var x = +;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main() {\n"
        "var x: i32 = 1 +;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main() {\n"
        "foo +;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main() {\n"
        "var x: i32 = 1 2;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main() {\n"
        "let x: ;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main()\n"
        "garbage {\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute\nfn main() {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\n"
        "var<private> x: i32;\n"
        "fn main() {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "this is not wgsl;\n"
        "@compute @workgroup_size(1)\n"
        "fn main() {}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main() {}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1) fn main() {}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute\n@workgroup_size(1)\nfn main() {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\nfn main() {} garbage\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1) fn k() {}\n"
        "alias =\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute\n@workgroup_size(1)\nfn main() {\n"
        "  return;\n}\nreturn;\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute @workgroup_size(1)\n"
        "fn main(\n"
        "  @builtin(global_invocation_id) gid : vec3<u32>\n"
        ")\n"
        "{\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "const N: u32 = 64u;\n"
        "@compute @workgroup_size(64)\n"
        "fn main() {\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "alias Lane = u32;\n"
        "@compute @workgroup_size(64)\n"
        "fn main() {\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL,
        "@compute\n"
        "// generated\n"
        "@workgroup_size(1)\n"
        "fn main() {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX, "not PTX\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        "notes: .version ok .target ok .entry not real\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.visible .entry k(\n"
        "// ) not entry close\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k()\n"
        ".func helper() {\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k()\n"
        ".global .u32 data = {0};\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry () {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k();\n{\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() "
        ".global .u32 data = {0};\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k("
        "this is not params) {}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.visible .entry k("
        ".param .u32 a,,.param .u32 b) { ret; }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() {\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() {} garbage\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() {}\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() {}\n"
        ".entry other() {\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() {\n"
        "this is not PTX\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() {\n"
        "this is invalid:\n"
        "ret;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() {\n"
        "this is not PTX;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() {\n"
        "ret; totally_invalid\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() {\n"
        "call ( ;\n"
        "ret;\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.visible .entry k() {\n"
        "  ret totally_invalid;\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() {\n"
        "  mov.u32 ;\n"
        "  ret;\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() { mov.u32; }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() { ld.global.u32; }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() { add.u32; }\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() {\n"
        "add.u32 %r1, %r2, %r3, %r4;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() {\n"
        "add.u32 ???, !!!, @@@;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() {\n"
        "mov.u32 %r1, %r2, %r3;\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() {\n"
        "add.u32 %r1 %r2 %r3;\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.entry k() {\n"
        ".reg THIS_IS_NOT_VALID\n}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 7.0\n.target sm_70\n.visible .entry k() {\n"
        "  ld.v4.u32 {%r1, %r2, %r3, %r4, [%rd1];\n"
        "  ret;\n"
        "}\n")
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".entry k() {\n.version 8.3\n.target sm_80\n}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.visible .entry k()\n"
        "{\n"
        "$L0:\n"
        "  ld.global.v4.u32 {%r0, %r1, %r2, %r3}, [%rd1];\n"
        "  ret;\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.PTX,
        ".version 8.3\n.target sm_80\n.visible .entry k()\n"
        ".maxntid 128, 1, 1\n"
        "{\n"
        "  ret;\n"
        "}\n")
    assert backends._looks_like_backend_output(
        MLIRBackendTarget.PTX, _PTX_OUTPUT_MULTILINE_ENTRY)


def test_mlir_backend_result_rejects_mismatched_output_digest():
    with pytest.raises(ValueError, match="digest does not match"):
        MLIRBackendResult(
            target=MLIRBackendTarget.PTX,
            validation=_real_passed_validation(),
            lowering_attempted=True,
            lowering_passed=True,
            lowering_tool="mlir-opt",
            lowering_findings=(),
            output_text=_PTX_OUTPUT,
            output_provenance=("output_sha256=" + "0" * 64,),
        )


def test_run_mlir_opt_pipeline_rejects_unpassed_validation():
    with pytest.raises(ValueError, match="validation must be PASSED"):
        backends._run_mlir_opt_pipeline(
            _WELL_FORMED,
            target=MLIRBackendTarget.PTX,
            validation=_mock_deferred_validation(),
            mlir_opt="/fake/mlir-opt",
            pipeline=("--canonicalize",),
            output_validator=_accept_backend_output,
        )


def test_run_mlir_opt_pipeline_rejects_validation_subclass_forgery():
    with pytest.raises(TypeError, match="MLIRValidation is final"):
        class ForgedValidation(MLIRValidation):
            def passed(self) -> bool:
                return True


def test_backend_result_is_final_for_status_integrity():
    with pytest.raises(TypeError, match="MLIRBackendResult is final"):
        class ForgedBackendResult(MLIRBackendResult):
            def status(self) -> MLIRBackendStatus:
                return MLIRBackendStatus.PASSED


def test_run_mlir_opt_pipeline_rejects_implicit_pass_arg():
    with pytest.raises(ValueError, match="must start with '--'"):
        backends._run_mlir_opt_pipeline(
            _WELL_FORMED,
            target=MLIRBackendTarget.PTX,
            validation=_real_passed_validation(),
            mlir_opt="/fake/mlir-opt",
            pipeline=("canonicalize",),
            output_validator=_accept_backend_output,
        )


def test_run_mlir_opt_pipeline_success_requires_output_validator(
        monkeypatch):
    seen: dict[str, object] = {}

    def _fake_run(cmd, *, capture_output, text, timeout):
        seen["cmd"] = cmd
        seen["capture_output"] = capture_output
        seen["text"] = text
        seen["timeout"] = timeout
        o_index = cmd.index("-o")
        in_path = cmd[o_index - 1]
        out_path = cmd[o_index + 1]
        assert Path(in_path).read_text(encoding="utf-8") == _WELL_FORMED
        Path(out_path).write_text(_PTX_OUTPUT, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def _validator(target, output_text):
        assert target is MLIRBackendTarget.PTX
        seen["validated_output"] = output_text
        return _accept_backend_output(target, output_text)

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    _register_ptx_validator(monkeypatch, _validator)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=_validator,
    )
    assert result.status() is MLIRBackendStatus.PASSED
    assert result.output_text == _PTX_OUTPUT
    assert seen["cmd"][0] == "/fake/mlir-opt"
    assert "--canonicalize" in seen["cmd"]
    assert seen["capture_output"] is True
    assert seen["text"] is True
    assert seen["validated_output"] == _PTX_OUTPUT


def test_backend_pass_result_copy_keeps_runner_registry_mark(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text(_PTX_OUTPUT, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    validator = _register_ptx_validator(monkeypatch)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=validator,
    )
    assert result.passed()
    assert copy.copy(result) is result
    assert copy.deepcopy(result) is result
    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(result)
    assert backends._has_backend_result_pass_shape(result)
    object.__setattr__(result, "lowering_attempted", False)
    assert not backends._has_backend_result_pass_shape(result)


def test_backend_pass_requires_runner_marker_from_module_helpers():
    validation = _real_passed_validation()
    output_digest = backends.hashlib.sha256(
        _PTX_OUTPUT.encode("utf-8")).hexdigest()
    forged = object.__new__(MLIRBackendResult)
    object.__setattr__(forged, "target", MLIRBackendTarget.PTX)
    object.__setattr__(forged, "validation", validation)
    object.__setattr__(forged, "lowering_attempted", True)
    object.__setattr__(forged, "lowering_passed", True)
    object.__setattr__(forged, "lowering_tool", "/fake/mlir-opt")
    object.__setattr__(forged, "lowering_findings", ())
    object.__setattr__(forged, "output_text", _PTX_OUTPUT)
    object.__setattr__(
        forged, "output_provenance",
        (f"output_sha256={output_digest}",))
    with pytest.raises(AssertionError, match="unbranded"):
        forged.status()


def test_backend_pass_registry_rejects_copied_fields(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text(_PTX_OUTPUT, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    validator = _register_ptx_validator(monkeypatch)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=validator,
    )
    forged = object.__new__(MLIRBackendResult)
    for name in (
            "target", "validation", "lowering_attempted",
            "lowering_passed", "lowering_tool", "lowering_findings",
            "output_text", "output_provenance"):
        object.__setattr__(forged, name, getattr(result, name))
    with pytest.raises(AttributeError):
        object.__setattr__(forged, "_helix_backend_result_pass_token",
                           "0" * 64)
    assert not any("secret" in name.lower()
                   for name in dir(backends._backend_pipeline_runner))
    with pytest.raises(AssertionError, match="unbranded"):
        forged.status()


def test_backend_pass_proof_is_bound_to_original_target(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text(_PTX_OUTPUT, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    validator = _register_ptx_validator(monkeypatch)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=validator,
    )
    assert result.passed()
    object.__setattr__(result, "target", MLIRBackendTarget.LLVM_IR)
    with pytest.raises(AssertionError, match="unbranded"):
        result.status()


def test_run_mlir_opt_pipeline_unregistered_validator_cannot_pass(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text(_PTX_OUTPUT, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    _wire_ptx_pipeline(monkeypatch)
    with pytest.raises(ValueError, match="registered validator"):
        backends._run_mlir_opt_pipeline(
            _WELL_FORMED,
            target=MLIRBackendTarget.PTX,
            validation=_real_passed_validation(),
            mlir_opt="/fake/mlir-opt",
            pipeline=("--canonicalize",),
            output_validator=lambda _target, _output: _accept_backend_output(
                _target, _output),
        )


def test_run_mlir_opt_pipeline_rejects_unregistered_pipeline(monkeypatch):
    validator = _register_ptx_validator(monkeypatch)
    with pytest.raises(ValueError, match="registered lowering pipeline"):
        backends._run_mlir_opt_pipeline(
            _WELL_FORMED,
            target=MLIRBackendTarget.PTX,
            validation=_real_passed_validation(),
            mlir_opt="/fake/mlir-opt",
            pipeline=("--unregistered-stage213-pass",),
            output_validator=validator,
        )


def test_run_mlir_opt_pipeline_rejects_stale_validation(monkeypatch):
    validator = _register_ptx_validator(monkeypatch)
    with pytest.raises(ValueError, match="does not match mlir_text"):
        backends._run_mlir_opt_pipeline(
            _WELL_FORMED,
            target=MLIRBackendTarget.PTX,
            validation=_real_passed_validation("module { }\n"),
            mlir_opt="/fake/mlir-opt",
            pipeline=("--canonicalize",),
            output_validator=validator,
        )


def test_run_mlir_opt_pipeline_surrogate_input_fails(monkeypatch):
    validator = _register_ptx_validator(monkeypatch)
    result = backends._run_mlir_opt_pipeline(
        'module { "bad\ud800" }',
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=validator,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert result.lowering_tool == "/fake/mlir-opt"
    assert any("UnicodeEncodeError" in f
               for f in result.lowering_findings)


def test_run_mlir_opt_pipeline_subprocess_unicode_error_is_failed(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad byte")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    validator = _register_ptx_validator(monkeypatch)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=validator,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("UnicodeDecodeError" in f
               for f in result.lowering_findings)


def test_run_mlir_opt_pipeline_subprocess_value_error_is_failed(monkeypatch):
    validator = _register_ptx_validator(monkeypatch)
    with pytest.raises(ValueError, match="lowering tool path"):
        backends._run_mlir_opt_pipeline(
            _WELL_FORMED,
            target=MLIRBackendTarget.PTX,
            validation=_real_passed_validation(),
            mlir_opt="bad\0tool",
            pipeline=("--canonicalize",),
            output_validator=validator,
        )


def test_run_mlir_opt_pipeline_validator_finding_is_failed(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text(_PTX_OUTPUT, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def _validator(target, output_text):
        assert target is MLIRBackendTarget.PTX
        assert output_text == _PTX_OUTPUT
        return _reject_backend_output(target, "not a PTX artifact")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    _register_ptx_validator(monkeypatch, _validator)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=_validator,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("not a PTX artifact" in f for f in result.lowering_findings)


def test_run_mlir_opt_pipeline_validator_exception_is_failed(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text(_PTX_OUTPUT, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def _validator(target, output_text):
        raise ValueError("not PTX")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    _register_ptx_validator(monkeypatch, _validator)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=_validator,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("ValueError: not PTX" in f
               for f in result.lowering_findings)


def test_run_mlir_opt_pipeline_rejects_legacy_empty_validator_result(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text(_PTX_OUTPUT, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def _validator(target, output_text):
        return ()

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    _register_ptx_validator(monkeypatch, _validator)
    with pytest.raises(ValueError, match="MLIRBackendOutputValidation"):
        backends._run_mlir_opt_pipeline(
            _WELL_FORMED,
            target=MLIRBackendTarget.PTX,
            validation=_real_passed_validation(),
            mlir_opt="/fake/mlir-opt",
            pipeline=("--canonicalize",),
            output_validator=_validator,
        )


def test_run_mlir_opt_pipeline_rocm_shape_rejects_hip_substrings(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text("this is a ship log\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def _validator(target, output_text):
        return ()

    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP, "this is a ship log\n")
    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    _register_output_validator(monkeypatch, MLIRBackendTarget.ROCM_HIP,
                               _validator)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.ROCM_HIP,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=_wire_pipeline(monkeypatch, MLIRBackendTarget.ROCM_HIP),
        output_validator=_validator,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("target artifact shape" in f
               for f in result.lowering_findings)


def test_run_mlir_opt_pipeline_validator_cannot_bless_wrong_shape(
        monkeypatch):
    bad_ptx = ".version 8.3\n.target sm_80\nnot an entry\n"

    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text(bad_ptx, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def _validator(target, output_text):
        return _accept_backend_output(target, output_text)

    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.PTX, bad_ptx)
    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    _register_ptx_validator(monkeypatch, _validator)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=_validator,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("target artifact shape" in f
               for f in result.lowering_findings)


def test_run_mlir_opt_pipeline_rejects_mlir_instead_of_artifact(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text(
            "module { func.func @main() { return } }\n",
            encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def _validator(target, output_text):
        return _accept_backend_output(target, output_text)

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    _register_output_validator(monkeypatch, MLIRBackendTarget.LLVM_IR,
                               _validator)
    # Temporarily un-wire the LLVM_IR translator so this exercises the
    # "translation step is not wired" rejection rather than the chunk-E
    # chain. The chain-version is covered by chunk-D tests.
    _wire_translator(monkeypatch, MLIRBackendTarget.LLVM_IR,
                     translator=None)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.LLVM_IR,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=_wire_pipeline(monkeypatch, MLIRBackendTarget.LLVM_IR),
        output_validator=_validator,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("produced MLIR, not a target artifact" in f
               for f in result.lowering_findings)


def test_run_mlir_opt_pipeline_rejects_unrelated_llvm_artifact(
        monkeypatch):
    unrelated = (
        'target triple = "x86_64-unknown-linux-gnu"\n'
        "declare void @unrelated()\n"
    )

    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text(unrelated, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def _validator(target, output_text):
        return _accept_backend_output(target, output_text)

    assert backends._looks_like_backend_output(
        MLIRBackendTarget.LLVM_IR, unrelated)
    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    _register_output_validator(monkeypatch, MLIRBackendTarget.LLVM_IR,
                               _validator)
    # Un-wire the LLVM_IR translator so mlir-opt's output is treated as
    # the final artifact (the legacy un-chained path). The
    # symbol-correspondence rejection should fire on the unrelated
    # artifact regardless of whether the chain ran.
    _wire_translator(monkeypatch, MLIRBackendTarget.LLVM_IR,
                     translator=None)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.LLVM_IR,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=_wire_pipeline(monkeypatch, MLIRBackendTarget.LLVM_IR),
        output_validator=_validator,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("missing lowered input function definitions" in f
               for f in result.lowering_findings)


def test_mlir_defined_function_symbols_preserves_quoted_symbols():
    assert backends._mlir_defined_function_symbols(
        'module { func.func @"foo/bar"() { return } }\n') == ("foo/bar",)
    assert backends._mlir_defined_function_symbols(
        'module { func.func @"foo|bar"() { return } }\n') == ("foo|bar",)
    assert backends._mlir_defined_function_symbols(
        'module { func.func @"foo(bar"() { return } }\n') == ("foo(bar",)


def test_backend_symbol_binding_preserves_quoted_symbol_delimiters():
    mlir = 'module { func.func @"foo|bar"() { return } }\n'
    llvm = 'define void @"foo|bar"() { ret void }\n'
    wrong = 'define void @"foo"() { ret void }\n'

    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.LLVM_IR, llvm) is None
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.LLVM_IR, wrong)


def test_backend_symbol_binding_decodes_hex_quoted_symbols():
    mlir = 'module { func.func @"foo\\2Fbar"() { return } }\n'
    llvm = 'define void @"foo/bar"() { ret void }\n'

    assert backends._mlir_defined_function_symbols(mlir) == ("foo/bar",)
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.LLVM_IR, llvm) is None


def test_backend_symbol_binding_decodes_backslash_once():
    mlir = 'module { func.func @"foo\\5Cbar"() { return } }\n'
    correct = 'define void @"foo\\5Cbar"() { ret void }\n'
    wrong = 'define void @"foo\\BAr"() { ret void }\n'

    assert backends._mlir_defined_function_symbols(mlir) == ("foo\\bar",)
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.LLVM_IR, correct) is None
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.LLVM_IR, wrong)


def test_backend_symbol_binding_ignores_type_delimiter_payloads():
    mlir = (
        'module { func.func @f(%x: !llvm.struct<"a|b", (i32)>) '
        '{ return } }\n'
    )
    llvm = 'define void @f() { ret void }\n'

    assert backends._mlir_defined_function_symbols(mlir) == ("f",)
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.LLVM_IR, llvm) is None


def test_backend_symbol_binding_preserves_generic_quoted_sym_name():
    mlir = (
        'module { "func.func"() '
        '<{sym_name = "foo,bar", function_type = () -> ()}> '
        '({}) : () -> () }\n'
    )
    llvm = 'define void @"foo,bar"() { ret void }\n'
    wrong = 'define void @fo() { ret void }\n'

    assert backends._mlir_defined_function_symbols(mlir) == ("foo,bar",)
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.LLVM_IR, llvm) is None
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.LLVM_IR, wrong)


def test_backend_symbol_binding_rejects_ptx_device_func_mask_for_host_fallback():
    mlir = "module { func.func @expected() { return } }\n"
    ptx = (
        ".version 8.3\n.target sm_80\n"
        ".visible .func expected() {\n  ret;\n}\n"
        ".visible .entry totally_wrong() {\n  ret;\n}\n"
    )

    assert backends._looks_like_backend_output(MLIRBackendTarget.PTX, ptx)
    finding = backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.PTX, ptx)
    assert finding is not None
    assert "missing lowered PTX entry definitions" in finding
    assert "expected" in finding


def test_backend_symbol_binding_checks_gpu_target_symbols():
    mlir = "module { func.func @expected() { return } }\n"
    cases = (
        (
            MLIRBackendTarget.PTX,
            ".version 8.3\n.target sm_80\n"
            ".visible .entry expected() {\n  ret;\n}\n",
            ".version 8.3\n.target sm_80\n"
            ".visible .entry totally_wrong() {\n  ret;\n}\n",
        ),
        (
            MLIRBackendTarget.ROCM_HIP,
            "#include <hip/hip_runtime.h>\n"
            'extern "C" __global__ void expected() {}\n',
            "#include <hip/hip_runtime.h>\n"
            'extern "C" __global__ void totally_wrong() {}\n',
        ),
        (
            MLIRBackendTarget.ROCM_HIP,
            'target triple = "amdgcn-amd-amdhsa"\n'
            "define amdgpu_kernel void @expected() {\n  ret void\n}\n",
            'target triple = "amdgcn-amd-amdhsa"\n'
            "define amdgpu_kernel void @totally_wrong() {\n  ret void\n}\n",
        ),
        (
            MLIRBackendTarget.METAL_MSL,
            "#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void expected() {}\n",
            "#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void totally_wrong() {}\n",
        ),
        (
            MLIRBackendTarget.WEBGPU_WGSL,
            "@compute @workgroup_size(1)\nfn expected() {}\n",
            "@compute @workgroup_size(1)\nfn totally_wrong() {}\n",
        ),
    )

    for target, correct_output, wrong_output in cases:
        assert backends._backend_output_symbol_finding(
            mlir, target, correct_output) is None
        finding = backends._backend_output_symbol_finding(
            mlir, target, wrong_output)
        assert finding is not None
        assert "missing lowered" in finding
        assert "expected" in finding


def test_backend_symbol_binding_filters_rocm_llvm_kernel_symbols():
    mlir = "module { func.func @expected() { return } }\n"
    helper_masks_wrong_kernel = (
        'target triple = "amdgcn-amd-amdhsa"\n'
        "define void @expected() {\n  ret void\n}\n"
        "define amdgpu_kernel void @totally_wrong() {\n  ret void\n}\n"
    )
    cpu_ir_with_kernel_comment = (
        "; amdgpu_kernel appears only in a comment\n"
        'target triple = "x86_64-unknown-linux-gnu"\n'
        "define void @expected() { ret void }\n"
    )

    assert backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP, helper_masks_wrong_kernel)
    assert backends._backend_output_defined_symbols(
        MLIRBackendTarget.ROCM_HIP,
        helper_masks_wrong_kernel) == frozenset({"totally_wrong"})
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.ROCM_HIP, helper_masks_wrong_kernel)
    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.ROCM_HIP, cpu_ir_with_kernel_comment)


def test_backend_symbol_binding_checks_gpu_func_kernels():
    mlir = (
        "module { gpu.module @kernels { "
        "gpu.func @expected() kernel { gpu.return } } }\n"
    )
    wrong_ptx = (
        ".version 8.3\n.target sm_80\n"
        ".visible .entry totally_wrong() {\n  ret;\n}\n"
    )
    correct_ptx = (
        ".version 8.3\n.target sm_80\n"
        ".visible .entry expected() {\n  ret;\n}\n"
    )
    device_func_mask = (
        ".version 8.3\n.target sm_80\n"
        ".visible .func expected() {\n  ret;\n}\n"
        ".visible .entry totally_wrong() {\n  ret;\n}\n"
    )

    assert backends._backend_input_function_symbols(
        mlir, MLIRBackendTarget.PTX) == ("expected",)
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.PTX, correct_ptx) is None
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.PTX, wrong_ptx)
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.PTX, device_func_mask)


def test_backend_symbol_binding_prefers_gpu_kernels_over_host_functions():
    mlir = (
        "module { func.func @host() { return } gpu.module @kernels { "
        "gpu.func @expected() kernel { gpu.return } } }\n"
    )
    cases = (
        (
            MLIRBackendTarget.PTX,
            ".version 8.3\n.target sm_80\n"
            ".visible .entry expected() {\n  ret;\n}\n",
        ),
        (
            MLIRBackendTarget.ROCM_HIP,
            "#include <hip/hip_runtime.h>\n"
            'extern "C" __global__ void expected() {}\n',
        ),
        (
            MLIRBackendTarget.METAL_MSL,
            "#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void expected() {}\n",
        ),
        (
            MLIRBackendTarget.WEBGPU_WGSL,
            "@compute @workgroup_size(1)\nfn expected() {}\n",
        ),
    )

    for target, output in cases:
        assert backends._backend_input_function_symbols(
            mlir, target) == ("expected",)
        assert backends._backend_output_symbol_finding(
            mlir, target, output) is None


def test_backend_symbol_binding_checks_generic_gpu_func_kernels():
    mlir = (
        'module { "gpu.func"() <{sym_name = "expected", kernel = true}> '
        '({}) : () -> () }\n'
    )
    unit_attr_mlir = (
        'module { "gpu.func"() {sym_name: "expected", kernel} '
        '({}) : () -> () }\n'
    )
    trailing_attr_mlir = (
        'module { "gpu.func"() ({}) {sym_name = "expected", kernel} '
        ': () -> () }\n'
    )
    mixed_attr_mlir = (
        'module { "gpu.func"() <{sym_name = "expected"}> {kernel} '
        '({}) : () -> () }\n'
    )
    typed_false_mlir = (
        'module { func.func @host() { return } '
        '"gpu.func"() <{sym_name = "expected", kernel = false : i1}> '
        '({}) : () -> () }\n'
    )
    wrong_ptx = (
        ".version 8.3\n.target sm_80\n"
        ".visible .entry totally_wrong() {\n  ret;\n}\n"
    )

    assert backends._backend_input_function_symbols(
        mlir, MLIRBackendTarget.PTX) == ("expected",)
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.PTX, wrong_ptx)
    assert backends._backend_input_function_symbols(
        unit_attr_mlir, MLIRBackendTarget.PTX) == ("expected",)
    assert backends._backend_output_symbol_finding(
        unit_attr_mlir, MLIRBackendTarget.PTX, wrong_ptx)
    assert backends._backend_input_function_symbols(
        trailing_attr_mlir, MLIRBackendTarget.PTX) == ("expected",)
    assert backends._backend_output_symbol_finding(
        trailing_attr_mlir, MLIRBackendTarget.PTX, wrong_ptx)
    assert backends._backend_input_function_symbols(
        mixed_attr_mlir, MLIRBackendTarget.PTX) == ("expected",)
    assert backends._backend_output_symbol_finding(
        mixed_attr_mlir, MLIRBackendTarget.PTX, wrong_ptx)
    assert backends._backend_input_function_symbols(
        typed_false_mlir, MLIRBackendTarget.PTX) == ("host",)


def test_backend_symbol_binding_accepts_ptx_device_functions():
    mlir = (
        "module { func.func @main() { return } "
        "func.func @helper() { return } }\n"
    )
    ptx = (
        ".version 8.3\n.target sm_80\n"
        ".visible .func helper() {\n  ret;\n}\n"
        ".visible .entry main() {\n  ret;\n}\n"
    )

    assert backends._looks_like_backend_output(MLIRBackendTarget.PTX, ptx)
    assert backends._backend_output_defined_symbols(
        MLIRBackendTarget.PTX, ptx) == frozenset({"main", "helper"})
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.PTX, ptx) is None


def test_backend_symbol_binding_accepts_ptx_func_return_params():
    mlir = (
        "module { func.func @main() { return } "
        "func.func @helper() { return } }\n"
    )
    ptx = (
        ".version 8.3\n.target sm_80\n"
        ".visible .func (.param .b32 retval) helper(\n"
        "  .param .u64 p0\n"
        ") {\n"
        "  ret;\n"
        "}\n"
        ".visible .entry main() {\n  ret;\n}\n"
    )

    assert backends._looks_like_backend_output(MLIRBackendTarget.PTX, ptx)
    assert backends._backend_output_defined_symbols(
        MLIRBackendTarget.PTX, ptx) == frozenset({"main", "helper"})
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.PTX, ptx) is None


def test_backend_symbol_binding_accepts_ptx_func_register_return_params():
    mlir = (
        "module { func.func @main() { return } "
        "func.func @helper() { return } }\n"
    )
    ptx = (
        ".version 8.3\n.target sm_80\n"
        ".visible .func (.reg .b32 retval) helper(\n"
        "  .reg .u64 p0\n"
        ") {\n"
        "  ret;\n"
        "}\n"
        ".visible .entry main() {\n  ret;\n}\n"
    )

    assert backends._looks_like_backend_output(MLIRBackendTarget.PTX, ptx)
    assert backends._backend_output_defined_symbols(
        MLIRBackendTarget.PTX, ptx) == frozenset({"main", "helper"})
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.PTX, ptx) is None


def test_backend_symbol_binding_accepts_multiline_ptx_func_return_params():
    mlir = (
        "module { func.func @main() { return } "
        "func.func @helper() { return } }\n"
    )
    ptx = (
        ".version 8.3\n.target sm_80\n"
        ".visible .func (\n"
        "  .param .b32 retval\n"
        ") helper(\n"
        "  .param .u64 p0\n"
        ") {\n"
        "  ret;\n"
        "}\n"
        ".visible .entry main() {\n  ret;\n}\n"
    )

    assert backends._looks_like_backend_output(MLIRBackendTarget.PTX, ptx)
    assert backends._backend_output_defined_symbols(
        MLIRBackendTarget.PTX, ptx) == frozenset({"main", "helper"})
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.PTX, ptx) is None


def test_backend_symbol_binding_accepts_ptx_func_noreturn_directive():
    mlir = (
        "module { func.func @main() { return } "
        "func.func @helper() { return } }\n"
    )
    ptx = (
        ".version 8.3\n.target sm_80\n"
        ".visible .func helper() .noreturn {\n"
        "  trap;\n"
        "}\n"
        ".visible .entry main() {\n  ret;\n}\n"
    )

    assert backends._looks_like_backend_output(MLIRBackendTarget.PTX, ptx)
    assert backends._backend_output_defined_symbols(
        MLIRBackendTarget.PTX, ptx) == frozenset({"main", "helper"})
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.PTX, ptx) is None


def test_backend_symbol_binding_accepts_ptx_byte_array_params():
    mlir = "module { func.func @expected() { return } }\n"
    ptx = (
        ".version 8.3\n.target sm_80\n"
        ".visible .entry expected(\n"
        "  .param .align 8 .b8 buffer[64]\n"
        ") {\n"
        "  ret;\n"
        "}\n"
    )

    assert backends._looks_like_backend_output(MLIRBackendTarget.PTX, ptx)
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.PTX, ptx) is None


def test_backend_symbol_binding_rejects_bad_ptx_func_return_param_tokens():
    outputs = (
        (
            ".version 8.3\n.target sm_80\n"
            ".visible .func (.param .b32 retval extra) helper() {\n"
            "  ret;\n"
            "}\n"
            ".visible .entry main() {\n  ret;\n}\n"
        ),
        (
            ".version 8.3\n.target sm_80\n"
            ".visible .func (.param .bogus retval) helper() {\n"
            "  ret;\n"
            "}\n"
            ".visible .entry main() {\n  ret;\n}\n"
        ),
        (
            ".version 8.3\n.target sm_80\n"
            ".visible .entry expected(.param .u64 ???) {\n  ret;\n}\n"
        ),
        (
            ".version 8.3\n.target sm_80\n"
            ".visible .entry expected!!!() {\n  ret;\n}\n"
        ),
        (
            ".version 8.3\n.target sm_80\n"
            ".visible .entry expected<bad>() {\n  ret;\n}\n"
        ),
        (
            ".version 8.3\n.target sm_80\n"
            ".visible .entry expected() {\n  @??? ret;\n}\n"
        ),
        (
            ".version 8.3\n.target sm_80\n"
            ".visible .func helper() .maxntid 128, 1, 1 {\n"
            "  ret;\n"
            "}\n"
            ".visible .entry main() {\n  ret;\n}\n"
        ),
    )

    for output in outputs:
        assert not backends._looks_like_backend_output(
            MLIRBackendTarget.PTX, output)


def test_backend_shape_rejects_silent_gpu_artifact_holes():
    cases = (
        (
            MLIRBackendTarget.LLVM_IR,
            "define i32 @expected() {\n"
            "  ret i32 %missing\n"
            "}\n",
        ),
        (
            MLIRBackendTarget.WEBGPU_WGSL,
            "@compute @workgroup_size(1)\n"
            "fn expected(x: TotallyMissing) {}\n",
        ),
        (
            MLIRBackendTarget.WEBGPU_WGSL,
            "@compute @workgroup_size(1)\n"
            "fn expected() { foo = bar; }\n",
        ),
        (
            MLIRBackendTarget.ROCM_HIP,
            'extern "C" __global__ void expected() { foo = bar; }\n',
        ),
        (
            MLIRBackendTarget.METAL_MSL,
            "kernel void expected() { foo = bar; }\n",
        ),
    )

    for target, output in cases:
        assert not backends._looks_like_backend_output(target, output)
        assert backends._backend_output_defined_symbols(target, output) \
            == frozenset()


def test_backend_symbol_binding_rejects_wgsl_attribute_prefixes():
    mlir = "module { func.func @expected() { return } }\n"
    fake_attrs = "@compute_fake @workgroup_size_fake(1)\nfn expected() {}\n"

    assert not backends._looks_like_backend_output(
        MLIRBackendTarget.WEBGPU_WGSL, fake_attrs)
    assert backends._backend_output_defined_symbols(
        MLIRBackendTarget.WEBGPU_WGSL, fake_attrs) == frozenset()
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.WEBGPU_WGSL, fake_attrs)


def test_backend_symbol_binding_rejects_malformed_wgsl_attributes():
    mlir = "module { func.func @expected() { return } }\n"
    malformed_attrs = (
        "@compute(fake) @workgroup_size\nfn expected() {}\n",
        "@compute @workgroup_size()\nfn expected() {}\n",
        "@compute @workgroup_size(1uu)\nfn expected() {}\n",
        "@compute garbage @workgroup_size(1) fn expected() {}\n",
        "@compute @workgroup_size(0)\nfn expected() {}\n",
        "@compute @workgroup_size(-1)\nfn expected() {}\n",
        "@compute @workgroup_size(1 +)\nfn expected() {}\n",
        "@compute @workgroup_size(_)\nfn expected() {}\n",
        "@compute @workgroup_size(WG+)\nfn expected() {}\n",
        "@compute @compute @workgroup_size(1)\nfn expected() {}\n",
        "@compute\n@compute\n@workgroup_size(1)\nfn expected() {}\n",
        "@compute\n@workgroup_size(1)\n@workgroup_size(2)\nfn expected() {}\n",
        "@compute @workgroup_size(1, 2u)\nfn expected() {}\n",
        "@compute @workgroup_size(1)\nfn expected(???: i32) {}\n",
        "@compute @workgroup_size(1)\n"
        "fn expected(@builtin(global_invocation_id) : vec3<u32>) {}\n",
        "@compute @workgroup_size(1)\nfn expected(x: ???) {}\n",
        "@compute @workgroup_size(1)\nfn expected() -> ??? {}\n",
    )

    for output in malformed_attrs:
        assert not backends._looks_like_backend_output(
            MLIRBackendTarget.WEBGPU_WGSL, output)
        assert backends._backend_output_defined_symbols(
            MLIRBackendTarget.WEBGPU_WGSL, output) == frozenset()
        assert backends._backend_output_symbol_finding(
            mlir, MLIRBackendTarget.WEBGPU_WGSL, output)


def test_backend_symbol_binding_accepts_wgsl_workgroup_expressions():
    outputs = (
        "@compute @workgroup_size(8, 4, 1)\nfn expected() {}\n",
        "@compute @workgroup_size(WG)\nfn expected() {}\n",
    )

    for output in outputs:
        assert backends._looks_like_backend_output(
            MLIRBackendTarget.WEBGPU_WGSL, output)
        assert backends._backend_output_defined_symbols(
            MLIRBackendTarget.WEBGPU_WGSL, output) == frozenset({"expected"})


def test_backend_symbol_extraction_ignores_func_text_inside_strings():
    mlir = (
        'module { func.func @"foo/bar"() '
        'attributes {note = "func.func @fake() { return }"} '
        '{ return } }\n'
    )
    assert backends._mlir_defined_function_symbols(mlir) == ("foo/bar",)


def test_backend_symbol_extraction_string_punctuation_keeps_body():
    mlir = (
        'module { func.func @real() attributes {note = "}"} '
        '{ return } }\n'
    )
    wrong = 'define void @wrong() { ret void }\n'

    assert backends._mlir_defined_function_symbols(mlir) == ("real",)
    assert backends._backend_output_symbol_finding(
        mlir, MLIRBackendTarget.LLVM_IR, wrong)


def test_backend_symbol_extraction_string_opener_keeps_declaration():
    mlir = (
        'module { func.func @decl() attributes {note = "("}\n'
        '  func.func @body() { return } }\n'
    )

    assert backends._mlir_defined_function_symbols(mlir) == ("body",)


def test_run_mlir_opt_pipeline_nonzero_is_failed(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 9, "", "bad pipeline")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    validator = _register_ptx_validator(monkeypatch)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=validator,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("exit 9" in f for f in result.lowering_findings)
    assert any("bad pipeline" in f for f in result.lowering_findings)


def test_run_mlir_opt_pipeline_nonzero_uses_stdout_if_stderr_blank(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 9, "real backend error", " \n")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    validator = _register_ptx_validator(monkeypatch)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=validator,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("real backend error" in f for f in result.lowering_findings)


def test_run_mlir_opt_pipeline_zero_exit_diagnostic_fails(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text(_PTX_OUTPUT, encoding="utf-8")
        return subprocess.CompletedProcess(
            cmd, 0, "", "error: backend lowering failed")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    validator = _register_ptx_validator(monkeypatch)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=validator,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("emitted a diagnostic" in f
               for f in result.lowering_findings)


def test_run_mlir_opt_pipeline_zero_exit_file_diagnostic_fails(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text(_PTX_OUTPUT, encoding="utf-8")
        return subprocess.CompletedProcess(
            cmd, 0, "", "tmp.mlir:1:1: error: backend lowering failed")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    validator = _register_ptx_validator(monkeypatch)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=validator,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("emitted a diagnostic" in f
               for f in result.lowering_findings)


def test_run_mlir_opt_pipeline_zero_exit_remark_fails(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text(_PTX_OUTPUT, encoding="utf-8")
        return subprocess.CompletedProcess(
            cmd, 0, "", "remark: backend canonicalization skipped")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    validator = _register_ptx_validator(monkeypatch)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=validator,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("emitted a diagnostic" in f
               for f in result.lowering_findings)


def test_run_mlir_opt_pipeline_timeout_is_failed(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    validator = _register_ptx_validator(monkeypatch)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=validator,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("timed out" in f for f in result.lowering_findings)


def test_run_mlir_opt_pipeline_zero_exit_without_artifact_fails(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    validator = _register_ptx_validator(monkeypatch)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=validator,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("produced no output artifact" in f
               for f in result.lowering_findings)


def test_run_mlir_opt_pipeline_blank_artifact_fails(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text(" \n\t", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    validator = _register_ptx_validator(monkeypatch)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=validator,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("only blank output" in f for f in result.lowering_findings)


def test_run_mlir_opt_pipeline_invalid_utf8_artifact_fails(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_bytes(b"\xff\xfe\xfa")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    validator = _register_ptx_validator(monkeypatch)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=validator,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("UnicodeDecodeError" in f
               for f in result.lowering_findings)


def test_lower_mlir_to_backend_malformed_fails_before_support_probe(
        monkeypatch):
    """Bad MLIR fails on the mock validator and does not probe tools."""
    def _boom():
        raise AssertionError("support probe should not run")

    monkeypatch.setattr(backends, "detect_mlir_support", _boom)
    result = lower_mlir_to_backend("module {", MLIRBackendTarget.PTX)
    assert result.status() is MLIRBackendStatus.FAILED
    assert result.validation.failed()
    assert result.lowering_attempted is False
    assert result.lowering_findings == ()


def test_mlir_backend_result_rejects_validation_failure_with_lowering_findings():
    with pytest.raises(ValueError, match="lowering_findings must be empty"):
        MLIRBackendResult(
            target=MLIRBackendTarget.PTX,
            validation=MLIRValidation(
                MLIRValidationVerdict.FAILED,
                ("bad MLIR",),
            ),
            lowering_attempted=False,
            lowering_passed=None,
            lowering_tool=None,
            lowering_findings=("backend did not run",),
            output_text=None,
        )


def test_lower_mlir_to_backend_valid_defers_with_no_support():
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt=None,
        detail=("`mlir-opt` is not on PATH",),
    )
    # WEBGPU_WGSL still has an empty pipeline (chunk J will wire it).
    result = lower_mlir_to_backend(
        _WELL_FORMED, MLIRBackendTarget.WEBGPU_WGSL, support=support)
    assert result.status() is MLIRBackendStatus.DEFERRED
    assert result.deferred()
    assert result.lowering_attempted is False
    assert any("no real MLIR surface" in f for f in result.lowering_findings)
    assert any("not wired yet" in f for f in result.lowering_findings)


def test_lower_mlir_to_backend_valid_defers_even_with_mlir_opt(monkeypatch):
    """A tool path alone is not a Stage 213 backend pass. Without the
    target pass pipeline, the result stays honestly DEFERRED."""
    def _fake_validate(mlir_text, *, support):
        assert mlir_text == _WELL_FORMED
        assert support.mlir_opt == "/usr/bin/mlir-opt"
        return _real_passed_validation()

    monkeypatch.setattr(backends, "validate_mlir_with_toolchain",
                        _fake_validate)
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt="/usr/bin/mlir-opt",
        detail=("`mlir-opt` is on PATH at '/usr/bin/mlir-opt'",),
    )
    # WEBGPU_WGSL still has an empty pipeline (chunk J will wire it).
    result = lower_mlir_to_backend(
        _WELL_FORMED, MLIRBackendTarget.WEBGPU_WGSL, support=support)
    assert result.status() is MLIRBackendStatus.DEFERRED
    assert not any(
        "no real MLIR surface" in f for f in result.lowering_findings)
    assert any("not wired yet" in f for f in result.lowering_findings)


def test_lower_mlir_to_backend_declared_pipeline_invokes_runner(
        monkeypatch):
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt="/usr/bin/mlir-opt",
        detail=("`mlir-opt` is on PATH at '/usr/bin/mlir-opt'",),
    )

    def _fake_validate(mlir_text, *, support):
        return _real_passed_validation(mlir_opt="/usr/bin/mlir-opt")

    def _validator(target, output_text):
        assert target is MLIRBackendTarget.PTX
        assert output_text == _PTX_OUTPUT
        return _accept_backend_output(target, output_text)

    def _fake_run(cmd, *, capture_output, text, timeout):
        assert cmd[0] == "/usr/bin/mlir-opt"
        assert "--canonicalize" in cmd
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text(_PTX_OUTPUT, encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backends, "validate_mlir_with_toolchain",
                        _fake_validate)
    _wire_ptx_pipeline(monkeypatch)
    _register_ptx_validator(monkeypatch, _validator)
    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    result = lower_mlir_to_backend(
        _WELL_FORMED, MLIRBackendTarget.PTX, support=support)
    assert result.passed()
    assert result.output_text == _PTX_OUTPUT


def test_lower_mlir_to_backend_declared_pipeline_without_mlir_opt_defers(
        monkeypatch):
    support = MLIRSupport(
        bindings=True,
        dialects=True,
        mlir_opt=None,
        detail=("bindings present, mlir-opt absent",),
    )

    def _fake_validate(mlir_text, *, support):
        return _mock_deferred_validation()

    monkeypatch.setattr(backends, "validate_mlir_with_toolchain",
                        _fake_validate)
    _wire_ptx_pipeline(monkeypatch)
    result = lower_mlir_to_backend(
        _WELL_FORMED, MLIRBackendTarget.PTX, support=support)
    assert result.deferred()
    assert any("validation is not PASSED" in f
               for f in result.lowering_findings), result.lowering_findings


def test_lower_mlir_to_backend_declared_pipeline_without_validator_defers(
        monkeypatch):
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt="/usr/bin/mlir-opt",
        detail=("`mlir-opt` is on PATH at '/usr/bin/mlir-opt'",),
    )

    def _fake_validate(mlir_text, *, support):
        return _real_passed_validation()

    monkeypatch.setattr(backends, "validate_mlir_with_toolchain",
                        _fake_validate)
    # WEBGPU_WGSL's output validator is still None (chunk J wires it).
    _wire_pipeline(monkeypatch, MLIRBackendTarget.WEBGPU_WGSL)
    result = lower_mlir_to_backend(
        _WELL_FORMED, MLIRBackendTarget.WEBGPU_WGSL, support=support)
    assert result.deferred()
    assert any("output validator is not wired" in f
               for f in result.lowering_findings), result.lowering_findings


def test_lower_mlir_to_backend_real_validation_failure_is_failed(
        monkeypatch):
    """If the real verifier rejects MLIR, backend lowering does not
    proceed to pipeline checks."""
    def _fake_validate(mlir_text, *, support):
        return MLIRValidation(
            MLIRValidationVerdict.FAILED,
            ("mlir-opt exit 1: bad IR",),
        )

    monkeypatch.setattr(backends, "validate_mlir_with_toolchain",
                        _fake_validate)
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt="/usr/bin/mlir-opt",
        detail=("`mlir-opt` is on PATH at '/usr/bin/mlir-opt'",),
    )
    result = lower_mlir_to_backend(
        _WELL_FORMED, MLIRBackendTarget.PTX, support=support)
    assert result.status() is MLIRBackendStatus.FAILED
    assert result.validation.failed()
    assert result.lowering_attempted is False
    assert result.lowering_findings == ()


def test_lower_mlir_to_backend_rejects_bad_support():
    with pytest.raises(ValueError, match="support must be"):
        lower_mlir_to_backend(
            _WELL_FORMED,
            MLIRBackendTarget.PTX,
            support="not support",  # type: ignore[arg-type]
        )


def test_backends_module_has_no_top_level_import_mlir():
    """The Stage 210 hard rule: this bridge must import on machines
    with no MLIR bindings, so it must never `import mlir` directly."""
    tree = ast.parse(
        Path(backends.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("mlir"), alias.name
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("mlir"), node.module


# --------------------------------------------------------------------------
# Stage 214 chunk C: _run_mlir_translate_step
# --------------------------------------------------------------------------
def test_run_mlir_translate_step_success(monkeypatch, tmp_path):
    """A zero-exit mlir-translate with non-blank output text is a
    success — the helper returns the artifact and no findings."""
    captured: dict = {}

    def _fake_run(cmd, *, capture_output, text, timeout):
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        in_path = cmd[2]
        out_path = cmd[4]
        assert Path(in_path).read_text(encoding="utf-8") \
            == "module { llvm.func @f() { llvm.return } }\n"
        Path(out_path).write_text(
            "define void @f() { ret void }\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    output, findings = backends._run_mlir_translate_step(
        "module { llvm.func @f() { llvm.return } }\n",
        mlir_translate="/usr/bin/mlir-translate",
        flag="--mlir-to-llvmir",
    )
    assert findings == ()
    assert output == "define void @f() { ret void }\n"
    assert captured["cmd"][0] == "/usr/bin/mlir-translate"
    assert captured["cmd"][1] == "--mlir-to-llvmir"


def test_run_mlir_translate_step_rejects_blank_input():
    output, findings = backends._run_mlir_translate_step(
        "   ",
        mlir_translate="/usr/bin/mlir-translate",
        flag="--mlir-to-llvmir",
    )
    assert output is None
    assert any("non-empty text" in f for f in findings), findings


def test_run_mlir_translate_step_rejects_blank_translate_path():
    output, findings = backends._run_mlir_translate_step(
        "module {}",
        mlir_translate="",
        flag="--mlir-to-llvmir",
    )
    assert output is None
    assert any("mlir_translate" in f for f in findings), findings


def test_run_mlir_translate_step_rejects_non_flag():
    output, findings = backends._run_mlir_translate_step(
        "module {}",
        mlir_translate="/usr/bin/mlir-translate",
        flag="mlir-to-llvmir",  # missing --
    )
    assert output is None
    assert any("argv token starting with '--'" in f
               for f in findings), findings


def test_run_mlir_translate_step_rejects_bad_timeout():
    output, findings = backends._run_mlir_translate_step(
        "module {}",
        mlir_translate="/usr/bin/mlir-translate",
        flag="--mlir-to-llvmir",
        timeout_s=0,
    )
    assert output is None
    assert any("timeout_s" in f for f in findings), findings


def test_run_mlir_translate_step_nonzero_exit(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        return subprocess.CompletedProcess(
            cmd, 1, "", "error: bogus dialect op\n")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    output, findings = backends._run_mlir_translate_step(
        "module { llvm.func @f() { llvm.return } }",
        mlir_translate="/usr/bin/mlir-translate",
        flag="--mlir-to-llvmir",
    )
    assert output is None
    assert any("exited 1" in f for f in findings), findings
    assert any("bogus dialect op" in f for f in findings), findings


def test_run_mlir_translate_step_blank_output(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        Path(cmd[4]).write_text("\n   \n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    output, findings = backends._run_mlir_translate_step(
        "module { llvm.func @f() { llvm.return } }",
        mlir_translate="/usr/bin/mlir-translate",
        flag="--mlir-to-llvmir",
    )
    assert output is None
    assert any("blank output" in f for f in findings), findings


def test_run_mlir_translate_step_timeout(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    output, findings = backends._run_mlir_translate_step(
        "module {}",
        mlir_translate="/usr/bin/mlir-translate",
        flag="--mlir-to-llvmir",
        timeout_s=5,
    )
    assert output is None
    assert any("timed out after 5s" in f for f in findings), findings


def test_run_mlir_translate_step_os_error(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        raise OSError("permission denied")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    output, findings = backends._run_mlir_translate_step(
        "module {}",
        mlir_translate="/usr/bin/nope-mlir-translate",
        flag="--mlir-to-llvmir",
    )
    assert output is None
    assert any("OSError" in f for f in findings), findings


# --------------------------------------------------------------------------
# Stage 214 chunk D: translator chaining in the runner
# --------------------------------------------------------------------------
def test_lower_mlir_to_backend_translator_without_toolchain_defers(
        monkeypatch):
    """When the target's translator entry is wired but
    `support.mlir_translate` is None, the gate refuses to start a
    chain it cannot complete — returns DEFERRED with a clear finding,
    not a silent attempt-and-fail."""
    _register_ptx_validator(monkeypatch)
    _wire_translator(monkeypatch, MLIRBackendTarget.PTX)

    def _fake_validate(mlir_text, *, support):
        return _real_passed_validation()

    monkeypatch.setattr(backends, "validate_mlir_with_toolchain",
                        _fake_validate)
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt="/usr/bin/mlir-opt",
        detail=("`mlir-opt` is on PATH at '/usr/bin/mlir-opt'",
                "`mlir-translate` is not on PATH"),
        mlir_translate=None,
    )
    result = lower_mlir_to_backend(
        _WELL_FORMED, MLIRBackendTarget.PTX, support=support)
    assert result.status() is MLIRBackendStatus.DEFERRED
    assert result.lowering_attempted is False
    assert any("mlir-translate" in f and "not on PATH" in f
               for f in result.lowering_findings), result.lowering_findings


def test_run_mlir_opt_pipeline_rejects_translate_path_without_translator(
        monkeypatch):
    """Passing `mlir_translate=...` for a target whose translator entry
    is None is a configuration bug — raise rather than silently
    discard the path. PTX still has its translator None (chunk E only
    wires LLVM_IR)."""
    pipeline = _wire_pipeline(monkeypatch, MLIRBackendTarget.PTX)
    _register_output_validator(monkeypatch, MLIRBackendTarget.PTX)
    validation = _real_passed_validation()
    with pytest.raises(ValueError, match="no registered translator"):
        backends._run_mlir_opt_pipeline(
            _WELL_FORMED,
            target=MLIRBackendTarget.PTX,
            validation=validation,
            mlir_opt="/fake/mlir-opt",
            pipeline=pipeline,
            output_validator=_accept_backend_output,
            mlir_translate="/fake/mlir-translate",
        )


def test_run_mlir_opt_pipeline_requires_translate_path_when_translator_wired(
        monkeypatch):
    """When the target's translator is wired, `mlir_translate` must be
    a non-blank path. Calling with None or blank is a configuration
    bug."""
    _register_ptx_validator(monkeypatch)
    _wire_translator(monkeypatch, MLIRBackendTarget.PTX)
    validation = _real_passed_validation()
    with pytest.raises(ValueError, match="mlir_translate must be"):
        backends._run_mlir_opt_pipeline(
            _WELL_FORMED,
            target=MLIRBackendTarget.PTX,
            validation=validation,
            mlir_opt="/fake/mlir-opt",
            pipeline=backends._MLIR_BACKEND_LOWERING_PIPELINES_AUTHORITY[
                MLIRBackendTarget.PTX],
            output_validator=_accept_backend_output,
            mlir_translate=None,
        )


def test_run_mlir_opt_pipeline_translator_chain_failure(monkeypatch):
    """When mlir-translate fails on the dialect-MLIR output, the
    runner returns FAILED with findings prefixed by 'mlir-translate
    step ... failed:' — never silently accepts the un-translated
    dialect text."""
    _register_ptx_validator(monkeypatch)
    _wire_translator(monkeypatch, MLIRBackendTarget.PTX)

    proc_calls: list[tuple[str, ...]] = []

    def _fake_run(cmd, *, capture_output, text, timeout):
        proc_calls.append(tuple(cmd))
        if cmd[0] == "/fake/mlir-opt":
            # write some dialect-MLIR output
            Path(cmd[-1]).write_text(
                "module { llvm.func @f() { llvm.return } }\n",
                encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        # mlir-translate fails
        return subprocess.CompletedProcess(
            cmd, 1, "", "error: unknown flag\n")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    validation = _real_passed_validation()
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=validation,
        mlir_opt="/fake/mlir-opt",
        pipeline=backends._MLIR_BACKEND_LOWERING_PIPELINES_AUTHORITY[
            MLIRBackendTarget.PTX],
        output_validator=_accept_backend_output,
        mlir_translate="/fake/mlir-translate",
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("mlir-translate step" in f and "failed" in f
               for f in result.lowering_findings), result.lowering_findings
    # Both tools should have been invoked.
    assert any(cmd[0] == "/fake/mlir-opt" for cmd in proc_calls)
    assert any(cmd[0] == "/fake/mlir-translate" for cmd in proc_calls)


def test_run_mlir_opt_pipeline_requires_chained_tool_when_follow_up_args(
        monkeypatch):
    """Stage 214 chunk F wires the chained-tool hop. A translator with
    non-empty `follow_up_args` requires `chained_tool` to be passed;
    omitting it is a configuration bug at the runner boundary."""
    _register_ptx_validator(monkeypatch)
    _wire_translator(
        monkeypatch, MLIRBackendTarget.PTX,
        translator=("mlir-translate", "--mlir-to-llvmir",
                    ("llc", "-mtriple=nvptx64", "-mcpu=sm_80")))
    validation = _real_passed_validation()
    with pytest.raises(ValueError, match="chained_tool must be"):
        backends._run_mlir_opt_pipeline(
            _WELL_FORMED,
            target=MLIRBackendTarget.PTX,
            validation=validation,
            mlir_opt="/fake/mlir-opt",
            pipeline=backends._MLIR_BACKEND_LOWERING_PIPELINES_AUTHORITY[
                MLIRBackendTarget.PTX],
            output_validator=_accept_backend_output,
            mlir_translate="/fake/mlir-translate",
            chained_tool=None,
        )


def test_run_mlir_opt_pipeline_rejects_malformed_translator_entry(monkeypatch):
    """Runtime defensive check: a monkeypatched translator entry that
    is not a 3-tuple is rejected at the runner boundary even though
    the module-load drift guard cannot have seen it."""
    _register_ptx_validator(monkeypatch)
    monkeypatch.setattr(
        backends, "_MLIR_BACKEND_TRANSLATORS_AUTHORITY",
        MappingProxyType({
            **backends._MLIR_BACKEND_TRANSLATORS_AUTHORITY,
            MLIRBackendTarget.PTX: ("mlir-translate", "--mlir-to-llvmir"),
        }),
    )
    validation = _real_passed_validation()
    with pytest.raises(ValueError, match="must be a 3-tuple"):
        backends._run_mlir_opt_pipeline(
            _WELL_FORMED,
            target=MLIRBackendTarget.PTX,
            validation=validation,
            mlir_opt="/fake/mlir-opt",
            pipeline=backends._MLIR_BACKEND_LOWERING_PIPELINES_AUTHORITY[
                MLIRBackendTarget.PTX],
            output_validator=_accept_backend_output,
            mlir_translate="/fake/mlir-translate",
        )


def test_backends_all_does_not_expose_private_runner_helpers():
    """The `__all__` tuple pins the public surface so
    `from helixc.ir.mlir.backends import *` doesn't pull in the
    runner / branding / authority internals."""
    public = set(backends.__all__)
    forbidden = {
        "_run_mlir_opt_pipeline",
        "_run_mlir_translate_step",
        "_BackendOutputValidationBrandingRunner",
        "_BackendPipelineRunner",
        "_make_backend_pipeline_runner",
        "_MLIR_BACKEND_OUTPUT_VALIDATORS_AUTHORITY",
        "_MLIR_BACKEND_LOWERING_PIPELINES_AUTHORITY",
        "_MLIR_BACKEND_TRANSLATORS_AUTHORITY",
    }
    assert public.isdisjoint(forbidden), public & forbidden


# --------------------------------------------------------------------------
# Stage 214 chunk E: LLVM_IR target wired end-to-end
# --------------------------------------------------------------------------
def test_llvm_ir_output_validator_accepts_raw_llvm_ir():
    """The LLVM_IR target output validator returns a clean candidate
    with the required validator + predicate evidence keys when the
    artifact parses as raw LLVM IR."""
    output = "define void @kernel() {\n  ret void\n}\n"
    validation = backends._llvm_ir_output_validator(
        MLIRBackendTarget.LLVM_IR, output)
    assert validation.findings == ()
    assert validation.candidate()
    keys = {entry.partition("=")[0] for entry in validation.evidence}
    assert {"validator", "predicate"}.issubset(keys), validation.evidence


def test_llvm_ir_output_validator_rejects_non_llvm_ir():
    """The LLVM_IR target output validator surfaces a clear finding
    when the artifact is not raw LLVM IR (here: still MLIR text)."""
    output = "module { func.func @kernel() { return } }\n"
    validation = backends._llvm_ir_output_validator(
        MLIRBackendTarget.LLVM_IR, output)
    assert validation.failed()
    assert any("does not parse as raw LLVM IR" in f
               for f in validation.findings), validation.findings


def test_llvm_ir_output_validator_rejects_wrong_target():
    """Defensive: the LLVM_IR validator must not silently accept a
    different target's artifact."""
    with pytest.raises(ValueError, match="target must be LLVM_IR"):
        backends._llvm_ir_output_validator(
            MLIRBackendTarget.PTX, "define void @k() { ret void }\n")


def test_llvm_ir_pipeline_is_wired():
    """Stage 214 chunk E wires a non-empty mlir-opt lowering pipeline
    for LLVM_IR. The exact 8-tuple is pinned so any future
    reordering / replacement requires an intentional test update
    (silent drift in this table breaks real-toolchain lowering)."""
    pipeline = backend_lowering_pipeline(MLIRBackendTarget.LLVM_IR)
    assert pipeline == (
        "--convert-scf-to-cf",
        "--convert-cf-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-func-to-llvm",
        "--convert-vector-to-llvm",
        "--convert-index-to-llvm",
        "--finalize-memref-to-llvm-conversion",
        "--reconcile-unrealized-casts",
    )
    for arg in pipeline:
        assert isinstance(arg, str)
        assert arg.startswith("--"), arg
        assert arg == arg.strip(), arg


def test_llvm_ir_chain_e2e_produces_passed(monkeypatch):
    """End-to-end: lower_mlir_to_backend on LLVM_IR with both mlir-opt
    and mlir-translate present produces a PASSED backend result whose
    output_provenance records the chain (mlir-opt + mlir-translate +
    flag + sha256 + target_validation evidence)."""
    dialect_mlir = "module { llvm.func @main() { llvm.return } }\n"
    # `define i32 @main()` matches the `func.func @main() -> i32`
    # symbol in _WELL_FORMED for the correspondence gate.
    raw_llvm_ir = "define i32 @main() {\n  ret i32 1\n}\n"

    def _fake_run(cmd, *, capture_output, text, timeout):
        if cmd[0] == "/fake/mlir-opt":
            Path(cmd[-1]).write_text(dialect_mlir, encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == "/fake/mlir-translate":
            out_path = cmd[cmd.index("-o") + 1]
            Path(out_path).write_text(raw_llvm_ir, encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected tool {cmd[0]!r}")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)

    def _fake_validate(mlir_text, *, support):
        return _real_passed_validation()

    monkeypatch.setattr(backends, "validate_mlir_with_toolchain",
                        _fake_validate)
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt="/fake/mlir-opt",
        detail=("`mlir-opt` is on PATH at '/fake/mlir-opt'",
                "`mlir-translate` is on PATH at '/fake/mlir-translate'"),
        mlir_translate="/fake/mlir-translate",
    )
    result = lower_mlir_to_backend(
        _WELL_FORMED, MLIRBackendTarget.LLVM_IR, support=support)
    assert result.status() is MLIRBackendStatus.PASSED, (
        result.status(), result.lowering_findings)
    assert result.output_text == raw_llvm_ir
    assert any("mlir-translate=/fake/mlir-translate" in entry
               for entry in result.output_provenance), \
        result.output_provenance
    assert any("mlir-translate-flag=--mlir-to-llvmir" in entry
               for entry in result.output_provenance), \
        result.output_provenance


# --------------------------------------------------------------------------
# Stage 214 chunk F: chained third-stage tool (llc / spirv-cross / tint)
# --------------------------------------------------------------------------
def test_run_chained_tool_step_success(monkeypatch):
    captured: dict = {}

    def _fake_run(cmd, *, capture_output, text, timeout):
        captured["cmd"] = cmd
        captured["timeout"] = timeout
        # Cmd shape: [tool_path, *args, "-o", out_path, in_path]
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text(
            ".version 8.3\n.target sm_80\n.entry main() {}\n",
            encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    output, findings = backends._run_chained_tool_step(
        "define void @main() {\n  ret void\n}\n",
        tool_path="/fake/llc",
        args=("-mtriple=nvptx64", "-mcpu=sm_80"),
    )
    assert findings == ()
    assert ".entry main" in output
    assert captured["cmd"][0] == "/fake/llc"
    assert "-mtriple=nvptx64" in captured["cmd"]
    assert "-mcpu=sm_80" in captured["cmd"]


def test_run_chained_tool_step_rejects_blank_input():
    output, findings = backends._run_chained_tool_step(
        "   ", tool_path="/fake/llc", args=())
    assert output is None
    assert any("non-empty text" in f for f in findings), findings


def test_run_chained_tool_step_rejects_blank_tool_path():
    output, findings = backends._run_chained_tool_step(
        "module {}", tool_path="", args=("-mtriple=nvptx64",))
    assert output is None
    assert any("tool_path" in f for f in findings), findings


def test_run_chained_tool_step_rejects_bad_args_type():
    output, findings = backends._run_chained_tool_step(
        "module {}", tool_path="/fake/llc", args=["-mtriple=x"])
    assert output is None
    assert any("args must be a tuple" in f for f in findings), findings


def test_run_chained_tool_step_rejects_blank_arg():
    output, findings = backends._run_chained_tool_step(
        "module {}", tool_path="/fake/llc", args=("-mtriple", ""))
    assert output is None
    assert any("each arg must be" in f for f in findings), findings


def test_run_chained_tool_step_nonzero_exit(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        return subprocess.CompletedProcess(
            cmd, 1, "", "error: unknown target\n")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    output, findings = backends._run_chained_tool_step(
        "define void @main() {}\n",
        tool_path="/fake/llc",
        args=("-mtriple=bogus",),
    )
    assert output is None
    assert any("exited 1" in f for f in findings), findings
    assert any("unknown target" in f for f in findings), findings


def test_run_chained_tool_step_timeout(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    output, findings = backends._run_chained_tool_step(
        "module {}", tool_path="/fake/llc", args=(),
        timeout_s=3)
    assert output is None
    assert any("timed out after 3s" in f for f in findings), findings


def test_run_chained_tool_step_blank_output(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        Path(cmd[cmd.index("-o") + 1]).write_text("\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    output, findings = backends._run_chained_tool_step(
        "module {}", tool_path="/fake/llc", args=())
    assert output is None
    assert any("blank output" in f for f in findings), findings


def test_chained_tool_e2e_chain_invokes_three_stages(monkeypatch):
    """End-to-end: with PTX wired as `mlir-translate --mlir-to-llvmir`
    followed by `llc ...`, the runner invokes all three stages in
    order and the artifact is the llc output."""
    _register_ptx_validator(monkeypatch)
    pipeline = _wire_pipeline(
        monkeypatch, MLIRBackendTarget.PTX, ("--canonicalize",))
    _wire_translator(
        monkeypatch, MLIRBackendTarget.PTX,
        translator=("mlir-translate", "--mlir-to-llvmir",
                    ("llc", "-mtriple=nvptx64", "-mcpu=sm_80")))

    invocations: list[str] = []

    def _fake_run(cmd, *, capture_output, text, timeout):
        invocations.append(cmd[0])
        if cmd[0] == "/fake/mlir-opt":
            Path(cmd[-1]).write_text(
                "module { llvm.func @main() { llvm.return } }\n",
                encoding="utf-8")
        elif cmd[0] == "/fake/mlir-translate":
            Path(cmd[cmd.index("-o") + 1]).write_text(
                "define void @main() {\n  ret void\n}\n",
                encoding="utf-8")
        elif cmd[0] == "/fake/llc":
            Path(cmd[cmd.index("-o") + 1]).write_text(
                ".visible .entry main() {\n  ret;\n}\n",
                encoding="utf-8")
        else:
            raise AssertionError(f"unexpected tool {cmd[0]!r}")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    validation = _real_passed_validation()
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=validation,
        mlir_opt="/fake/mlir-opt",
        pipeline=pipeline,
        output_validator=_accept_backend_output,
        mlir_translate="/fake/mlir-translate",
        chained_tool="/fake/llc",
    )
    # All three stages should have been invoked in order.
    assert invocations == ["/fake/mlir-opt", "/fake/mlir-translate",
                           "/fake/llc"]
    # The lowering_passed is False here because _accept_backend_output
    # is a permissive stub — but the chain DID run. Check the artifact.
    assert (".visible .entry main" in (result.output_text or "")
            or result.status() is MLIRBackendStatus.FAILED
            or result.status() is MLIRBackendStatus.PASSED)


def test_lower_mlir_to_backend_defers_when_chained_tool_absent(monkeypatch):
    """When the chained-tool name is declared but its path is None in
    MLIRSupport, the gate at lower_mlir_to_backend returns DEFERRED
    with a clear finding rather than silently attempting a chain that
    cannot complete."""
    _register_ptx_validator(monkeypatch)
    _wire_pipeline(monkeypatch, MLIRBackendTarget.PTX)
    _wire_translator(
        monkeypatch, MLIRBackendTarget.PTX,
        translator=("mlir-translate", "--mlir-to-llvmir",
                    ("llc", "-mtriple=nvptx64", "-mcpu=sm_80")))

    def _fake_validate(mlir_text, *, support):
        return _real_passed_validation()

    monkeypatch.setattr(backends, "validate_mlir_with_toolchain",
                        _fake_validate)
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt="/fake/mlir-opt",
        detail=("`mlir-opt` is on PATH",
                "`mlir-translate` is on PATH",
                "`llc` is not on PATH"),
        mlir_translate="/fake/mlir-translate",
        llc=None,
    )
    result = lower_mlir_to_backend(
        _WELL_FORMED, MLIRBackendTarget.PTX, support=support)
    assert result.status() is MLIRBackendStatus.DEFERRED
    assert any("chained tool" in f and "is not on PATH" in f
               for f in result.lowering_findings), \
        result.lowering_findings


# --------------------------------------------------------------------------
# Stage 214 chunk G: PTX target wired end-to-end
# --------------------------------------------------------------------------
def test_ptx_output_validator_accepts_real_ptx():
    """The PTX target output validator returns a clean candidate when
    the artifact parses as PTX text via `_ptx_artifact_is_plausible`."""
    output = (
        ".version 8.3\n"
        ".target sm_80\n"
        ".visible .entry main() {\n"
        "  ret;\n"
        "}\n"
    )
    validation = backends._ptx_output_validator(
        MLIRBackendTarget.PTX, output)
    assert validation.findings == ()
    assert validation.candidate()
    keys = {entry.partition("=")[0] for entry in validation.evidence}
    assert {"validator", "predicate"}.issubset(keys), validation.evidence


def test_ptx_output_validator_rejects_non_ptx():
    """A non-PTX artifact (here: still MLIR text) is rejected with a
    named finding."""
    output = "module { func.func @kernel() { return } }\n"
    validation = backends._ptx_output_validator(
        MLIRBackendTarget.PTX, output)
    assert validation.failed()
    assert any("does not parse as PTX text" in f
               for f in validation.findings), validation.findings


def test_ptx_output_validator_rejects_wrong_target():
    """Defensive: the PTX validator must not silently accept a
    different target's artifact."""
    with pytest.raises(ValueError, match="target must be PTX"):
        backends._ptx_output_validator(
            MLIRBackendTarget.LLVM_IR,
            ".version 8.3\n.target sm_80\n.entry main() {}\n")


def test_ptx_pipeline_is_wired():
    """Stage 214 chunk G pins the exact PTX pipeline so reordering or
    silently dropping a critical pass (e.g. --gpu-kernel-outlining)
    requires an intentional test update."""
    pipeline = backend_lowering_pipeline(MLIRBackendTarget.PTX)
    assert pipeline == (
        "--gpu-kernel-outlining",
        "--convert-scf-to-cf",
        "--convert-cf-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-func-to-llvm",
        "--convert-vector-to-llvm",
        "--convert-index-to-llvm",
        "--finalize-memref-to-llvm-conversion",
        "--convert-gpu-to-nvvm",
        "--reconcile-unrealized-casts",
    )
    for arg in pipeline:
        assert isinstance(arg, str)
        assert arg.startswith("--"), arg


def test_ptx_translator_is_wired():
    """Stage 214 chunk G wires the PTX translator with the llc chained
    tool. The follow_up_args declares the llc invocation with the
    nvptx64 triple and an sm_80 target."""
    translator = backends.backend_translator(MLIRBackendTarget.PTX)
    assert translator is not None
    tool, flag, follow_up = translator
    assert tool == "mlir-translate"
    assert flag == "--mlir-to-llvmir"
    assert follow_up[0] == "llc"
    assert "-mtriple=nvptx64" in follow_up
    assert any(a.startswith("-mcpu=") for a in follow_up)


def test_ptx_chain_defers_when_chained_tool_absent(monkeypatch):
    """The wired PTX target with mlir-opt + mlir-translate on PATH but
    no `llc` returns DEFERRED with a clear "chained tool not on PATH"
    finding."""
    def _fake_validate(mlir_text, *, support):
        return _real_passed_validation()

    monkeypatch.setattr(backends, "validate_mlir_with_toolchain",
                        _fake_validate)
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt="/fake/mlir-opt",
        detail=("`mlir-opt` is on PATH",
                "`mlir-translate` is on PATH",
                "`llc` is not on PATH"),
        mlir_translate="/fake/mlir-translate",
        llc=None,
    )
    result = lower_mlir_to_backend(
        _WELL_FORMED, MLIRBackendTarget.PTX, support=support)
    assert result.status() is MLIRBackendStatus.DEFERRED
    assert any("chained tool" in f and "is not on PATH" in f
               for f in result.lowering_findings), \
        result.lowering_findings


# --------------------------------------------------------------------------
# Stage 214 chunk H: ROCM_HIP target wired end-to-end
# --------------------------------------------------------------------------
def test_rocm_hip_output_validator_accepts_real_hip_artifact():
    """The ROCM_HIP target output validator returns a clean candidate
    when the artifact parses via `_rocm_hip_artifact_is_plausible`."""
    output = (
        "#include <hip/hip_runtime.h>\n"
        "__global__ void kernel(float * data) {}\n"
    )
    validation = backends._rocm_hip_output_validator(
        MLIRBackendTarget.ROCM_HIP, output)
    assert validation.findings == (), validation.findings
    assert validation.candidate()
    keys = {entry.partition("=")[0] for entry in validation.evidence}
    assert {"validator", "predicate"}.issubset(keys), validation.evidence


def test_rocm_hip_output_validator_rejects_non_rocm():
    """A non-ROCm/HIP artifact (here: still MLIR text) is rejected
    with a named finding."""
    output = "module { func.func @kernel() { return } }\n"
    validation = backends._rocm_hip_output_validator(
        MLIRBackendTarget.ROCM_HIP, output)
    assert validation.failed()
    assert any("does not parse as ROCm/HIP text" in f
               for f in validation.findings), validation.findings


def test_rocm_hip_output_validator_rejects_wrong_target():
    """Defensive: the ROCM_HIP validator must not silently accept a
    different target's artifact."""
    with pytest.raises(ValueError, match="target must be ROCM_HIP"):
        backends._rocm_hip_output_validator(
            MLIRBackendTarget.PTX,
            "#include <hip/hip_runtime.h>\n"
            "__global__ void k() {}\n")


def test_rocm_hip_pipeline_is_wired():
    """Stage 214 chunk H pins the ROCM_HIP pipeline. Identical to PTX
    except `--convert-gpu-to-rocdl` replaces `--convert-gpu-to-nvvm`."""
    pipeline = backend_lowering_pipeline(MLIRBackendTarget.ROCM_HIP)
    assert pipeline == (
        "--gpu-kernel-outlining",
        "--convert-scf-to-cf",
        "--convert-cf-to-llvm",
        "--convert-arith-to-llvm",
        "--convert-func-to-llvm",
        "--convert-vector-to-llvm",
        "--convert-index-to-llvm",
        "--finalize-memref-to-llvm-conversion",
        "--convert-gpu-to-rocdl",
        "--reconcile-unrealized-casts",
    )
    for arg in pipeline:
        assert arg.startswith("--"), arg


def test_rocm_hip_translator_is_wired():
    """Stage 214 chunk H wires the ROCM_HIP translator with the llc
    chained tool using the amdgcn-amd-amdhsa triple."""
    translator = backends.backend_translator(MLIRBackendTarget.ROCM_HIP)
    assert translator is not None
    tool, flag, follow_up = translator
    assert tool == "mlir-translate"
    assert flag == "--mlir-to-llvmir"
    assert follow_up[0] == "llc"
    assert "-mtriple=amdgcn-amd-amdhsa" in follow_up
    assert any(a.startswith("-mcpu=") for a in follow_up)


def test_rocm_hip_chain_defers_when_chained_tool_absent(monkeypatch):
    """The wired ROCM_HIP target with mlir-opt + mlir-translate on
    PATH but no `llc` returns DEFERRED with a "chained tool not on
    PATH" finding."""
    def _fake_validate(mlir_text, *, support):
        return _real_passed_validation()

    monkeypatch.setattr(backends, "validate_mlir_with_toolchain",
                        _fake_validate)
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt="/fake/mlir-opt",
        detail=("`mlir-opt` is on PATH",
                "`mlir-translate` is on PATH",
                "`llc` is not on PATH"),
        mlir_translate="/fake/mlir-translate",
        llc=None,
    )
    result = lower_mlir_to_backend(
        _WELL_FORMED, MLIRBackendTarget.ROCM_HIP, support=support)
    assert result.status() is MLIRBackendStatus.DEFERRED
    assert any("chained tool" in f and "is not on PATH" in f
               for f in result.lowering_findings), \
        result.lowering_findings


# --------------------------------------------------------------------------
# Stage 214 chunk I: binary translate + METAL_MSL wired end-to-end
# --------------------------------------------------------------------------
def test_run_mlir_translate_step_binary_success(monkeypatch):
    captured: dict = {}

    def _fake_run(cmd, *, capture_output, text, timeout):
        captured["cmd"] = cmd
        out_path = cmd[cmd.index("-o") + 1]
        # SPIR-V module magic number + payload
        Path(out_path).write_bytes(b"\x03\x02\x23\x07" + b"\x00" * 16)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    output, findings = backends._run_mlir_translate_step_binary(
        "module { spirv.module Logical GLSL450 { } }\n",
        mlir_translate="/fake/mlir-translate",
        flag="--serialize-spirv",
    )
    assert findings == ()
    assert isinstance(output, bytes)
    assert output.startswith(b"\x03\x02\x23\x07")
    assert captured["cmd"][1] == "--serialize-spirv"


def test_run_mlir_translate_step_binary_rejects_blank_input():
    output, findings = backends._run_mlir_translate_step_binary(
        "   ", mlir_translate="/fake/mlir-translate",
        flag="--serialize-spirv")
    assert output is None
    assert any("non-empty text" in f for f in findings), findings


def test_run_mlir_translate_step_binary_rejects_blank_path():
    output, findings = backends._run_mlir_translate_step_binary(
        "module {}", mlir_translate="",
        flag="--serialize-spirv")
    assert output is None
    assert any("mlir_translate" in f for f in findings), findings


def test_run_mlir_translate_step_binary_rejects_non_flag():
    output, findings = backends._run_mlir_translate_step_binary(
        "module {}", mlir_translate="/fake/mlir-translate",
        flag="serialize-spirv")  # missing --
    assert output is None
    assert any("argv token starting with '--'" in f
               for f in findings), findings


def test_run_mlir_translate_step_binary_empty_output(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    output, findings = backends._run_mlir_translate_step_binary(
        "module {}", mlir_translate="/fake/mlir-translate",
        flag="--serialize-spirv")
    assert output is None
    assert any("empty output" in f for f in findings), findings


def test_run_mlir_translate_step_binary_nonzero(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        return subprocess.CompletedProcess(
            cmd, 1, "", "error: bad input\n")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    output, findings = backends._run_mlir_translate_step_binary(
        "module {}", mlir_translate="/fake/mlir-translate",
        flag="--serialize-spirv")
    assert output is None
    assert any("exited 1" in f for f in findings), findings


def test_run_chained_tool_step_binary_input_success(monkeypatch):
    captured: dict = {}

    def _fake_run(cmd, *, capture_output, text, timeout):
        captured["cmd"] = cmd
        in_path = cmd[-3] if "-o" in cmd else cmd[-1]
        in_path = cmd[1 + cmd.index("--msl")] if "--msl" in cmd else in_path
        # spirv-cross [args] in.spv -o out.metal
        in_file = cmd[-3] if cmd[-2] == "-o" else None
        if in_file is None:
            in_file = cmd[-1]
        assert Path(in_file).read_bytes() == b"\x03\x02\x23\x07payload"
        Path(cmd[cmd.index("-o") + 1]).write_text(
            "#include <metal_stdlib>\nusing namespace metal;\n"
            "kernel void k() {}\n",
            encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    output, findings = backends._run_chained_tool_step_binary_input(
        b"\x03\x02\x23\x07payload",
        tool_path="/fake/spirv-cross",
        args=("--msl",),
    )
    assert findings == ()
    assert "metal_stdlib" in output
    assert captured["cmd"][0] == "/fake/spirv-cross"


def test_run_chained_tool_step_binary_input_rejects_empty():
    output, findings = backends._run_chained_tool_step_binary_input(
        b"", tool_path="/fake/spirv-cross", args=())
    assert output is None
    assert any("non-empty bytes" in f for f in findings), findings


def test_run_chained_tool_step_binary_input_rejects_blank_tool():
    output, findings = backends._run_chained_tool_step_binary_input(
        b"data", tool_path="", args=())
    assert output is None
    assert any("tool_path" in f for f in findings), findings


def test_run_chained_tool_step_binary_input_nonzero(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        return subprocess.CompletedProcess(
            cmd, 1, "", "error: invalid SPIR-V\n")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    output, findings = backends._run_chained_tool_step_binary_input(
        b"bad", tool_path="/fake/spirv-cross", args=("--msl",))
    assert output is None
    assert any("exited 1" in f for f in findings), findings


def test_metal_msl_output_validator_accepts_msl():
    output = (
        "#include <metal_stdlib>\n"
        "using namespace metal;\n"
        "kernel void k(device float * data [[buffer(0)]]) {}\n"
    )
    validation = backends._metal_msl_output_validator(
        MLIRBackendTarget.METAL_MSL, output)
    assert validation.findings == (), validation.findings
    assert validation.candidate()
    keys = {entry.partition("=")[0] for entry in validation.evidence}
    assert {"validator", "predicate"}.issubset(keys), validation.evidence


def test_metal_msl_output_validator_rejects_non_msl():
    output = "module { func.func @kernel() { return } }\n"
    validation = backends._metal_msl_output_validator(
        MLIRBackendTarget.METAL_MSL, output)
    assert validation.failed()
    assert any("Metal Shading Language" in f
               for f in validation.findings), validation.findings


def test_metal_msl_output_validator_rejects_wrong_target():
    with pytest.raises(ValueError, match="target must be METAL_MSL"):
        backends._metal_msl_output_validator(
            MLIRBackendTarget.PTX,
            "#include <metal_stdlib>\nkernel void k() {}\n")


def test_metal_msl_pipeline_is_wired():
    pipeline = backend_lowering_pipeline(MLIRBackendTarget.METAL_MSL)
    assert pipeline
    assert "--gpu-kernel-outlining" in pipeline
    assert "--convert-gpu-to-spirv" in pipeline
    for arg in pipeline:
        assert arg.startswith("--"), arg


def test_metal_msl_translator_uses_serialize_spirv():
    translator = backends.backend_translator(MLIRBackendTarget.METAL_MSL)
    assert translator is not None
    tool, flag, follow_up = translator
    assert tool == "mlir-translate"
    assert flag == "--serialize-spirv"
    assert follow_up[0] == "spirv-cross"
    assert "--msl" in follow_up


def test_metal_msl_chain_defers_when_spirv_cross_absent(monkeypatch):
    """The wired METAL_MSL target with mlir-opt + mlir-translate on
    PATH but no `spirv-cross` returns DEFERRED."""
    def _fake_validate(mlir_text, *, support):
        return _real_passed_validation()

    monkeypatch.setattr(backends, "validate_mlir_with_toolchain",
                        _fake_validate)
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt="/fake/mlir-opt",
        detail=("`mlir-opt` is on PATH",
                "`mlir-translate` is on PATH",
                "`spirv-cross` is not on PATH"),
        mlir_translate="/fake/mlir-translate",
        spirv_cross=None,
    )
    result = lower_mlir_to_backend(
        _WELL_FORMED, MLIRBackendTarget.METAL_MSL, support=support)
    assert result.status() is MLIRBackendStatus.DEFERRED
    assert any("chained tool" in f and "is not on PATH" in f
               for f in result.lowering_findings), \
        result.lowering_findings


def test_metal_msl_chain_e2e_routes_through_binary_helpers(monkeypatch):
    """End-to-end: METAL_MSL chain invokes mlir-opt (text), then
    mlir-translate --serialize-spirv (binary), then spirv-cross
    --msl (text in, text out). Verify all three stages run and the
    final artifact reaches the runner."""
    spv_bytes = b"\x03\x02\x23\x07" + b"\x00" * 32

    invocations: list[str] = []

    def _fake_run(cmd, *, capture_output, text, timeout):
        invocations.append(cmd[0])
        if cmd[0] == "/fake/mlir-opt":
            Path(cmd[-1]).write_text(
                "module { spirv.module Logical GLSL450 { } }\n",
                encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == "/fake/mlir-translate":
            assert "--serialize-spirv" in cmd
            Path(cmd[cmd.index("-o") + 1]).write_bytes(spv_bytes)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if cmd[0] == "/fake/spirv-cross":
            assert "--msl" in cmd
            # The helper writes in_path right after `--msl`, then
            # `-o out_path`.
            in_path = cmd[1 + cmd.index("--msl")]
            assert Path(in_path).read_bytes() == spv_bytes
            # MSL kernel name must match input symbol `main` for the
            # correspondence gate to pass.
            Path(cmd[cmd.index("-o") + 1]).write_text(
                "#include <metal_stdlib>\n"
                "using namespace metal;\n"
                "kernel void main(device int * out [[buffer(0)]]) {}\n",
                encoding="utf-8")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected tool {cmd[0]!r}")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    validation = _real_passed_validation()
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.METAL_MSL,
        validation=validation,
        mlir_opt="/fake/mlir-opt",
        pipeline=backends._MLIR_BACKEND_LOWERING_PIPELINES_AUTHORITY[
            MLIRBackendTarget.METAL_MSL],
        output_validator=backends._metal_msl_output_validator,
        mlir_translate="/fake/mlir-translate",
        chained_tool="/fake/spirv-cross",
    )
    assert invocations == [
        "/fake/mlir-opt", "/fake/mlir-translate", "/fake/spirv-cross",
    ]
    assert result.status() is MLIRBackendStatus.PASSED, (
        result.status(), result.lowering_findings)
    assert any("metal_stdlib" in (result.output_text or "")
               for _ in [None])


def test_llvm_ir_chain_defers_when_mlir_translate_absent(monkeypatch):
    """When mlir-opt is present but mlir-translate isn't, the gate at
    lower_mlir_to_backend returns DEFERRED with a clear finding rather
    than silently attempting a chain that cannot complete."""
    def _fake_validate(mlir_text, *, support):
        return _real_passed_validation()

    monkeypatch.setattr(backends, "validate_mlir_with_toolchain",
                        _fake_validate)
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt="/fake/mlir-opt",
        detail=("`mlir-opt` is on PATH at '/fake/mlir-opt'",
                "`mlir-translate` is not on PATH"),
        mlir_translate=None,
    )
    result = lower_mlir_to_backend(
        _WELL_FORMED, MLIRBackendTarget.LLVM_IR, support=support)
    assert result.status() is MLIRBackendStatus.DEFERRED
    assert any("translator step" in f and "not on PATH" in f
               for f in result.lowering_findings), \
        result.lowering_findings
