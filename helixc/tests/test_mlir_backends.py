"""Tests for helixc.ir.mlir.backends - v3.0 Phase E, Stage 213.

Stage 213 begins the MLIR-to-backends seam. This first chunk is a
mock-path-first scaffold: malformed MLIR fails loudly, well-shaped
MLIR defers until real target pass pipelines exist, and no target can
claim a false PASS on a toolchain-free machine.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path

import pytest

from helixc.backend.gpu_ci import BackendKind as GPUBackendKind
from helixc.ir.mlir import backends
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


def _mock_deferred_validation() -> MLIRValidation:
    return MLIRValidation(
        MLIRValidationVerdict.DEFERRED,
        ("mock validation deferred to real mlir-opt",),
    )


def _real_passed_validation() -> MLIRValidation:
    return MLIRValidation(MLIRValidationVerdict.PASSED, ())


def _accept_backend_output(output_text: str) -> tuple[str, ...]:
    assert output_text.strip()
    return ()


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


def test_backend_lowering_pipelines_are_explicitly_unwired():
    """Chunk A records the pass-pipeline table but keeps every target
    unwired; empty means DEFERRED, not PASSED."""
    for target in MLIRBackendTarget:
        assert backend_lowering_pipeline(target) == ()


def test_backend_output_validators_are_explicitly_unwired():
    """A pass pipeline alone cannot prove backend-consumable output."""
    assert set(backends.MLIR_BACKEND_OUTPUT_VALIDATORS) == set(
        MLIRBackendTarget)
    for target in MLIRBackendTarget:
        assert backends.MLIR_BACKEND_OUTPUT_VALIDATORS[target] is None


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


def test_mlir_backend_result_accepts_real_success_with_output_text():
    result = MLIRBackendResult(
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        lowering_attempted=True,
        lowering_passed=True,
        lowering_tool="mlir-opt",
        lowering_findings=(),
        output_text=".version 8.3\n",
    )
    assert result.status() is MLIRBackendStatus.PASSED
    assert result.passed()


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
        Path(out_path).write_text("module { }\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def _validator(output_text):
        seen["validated_output"] = output_text
        return ()

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=_validator,
    )
    assert result.status() is MLIRBackendStatus.PASSED
    assert result.output_text == "module { }\n"
    assert seen["cmd"][0] == "/fake/mlir-opt"
    assert "--canonicalize" in seen["cmd"]
    assert seen["capture_output"] is True
    assert seen["text"] is True
    assert seen["validated_output"] == "module { }\n"


def test_run_mlir_opt_pipeline_validator_finding_is_failed(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        out_path = cmd[cmd.index("-o") + 1]
        Path(out_path).write_text("module { }\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, "", "")

    def _validator(output_text):
        assert output_text == "module { }\n"
        return ("not a PTX artifact",)

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
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


def test_run_mlir_opt_pipeline_nonzero_is_failed(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 9, "", "bad pipeline")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=_accept_backend_output,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("exit 9" in f for f in result.lowering_findings)
    assert any("bad pipeline" in f for f in result.lowering_findings)


def test_run_mlir_opt_pipeline_timeout_is_failed(monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(cmd, timeout)

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=_accept_backend_output,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("timed out" in f for f in result.lowering_findings)


def test_run_mlir_opt_pipeline_zero_exit_without_artifact_fails(
        monkeypatch):
    def _fake_run(cmd, *, capture_output, text, timeout):
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(backends.subprocess, "run", _fake_run)
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=_accept_backend_output,
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
    result = backends._run_mlir_opt_pipeline(
        _WELL_FORMED,
        target=MLIRBackendTarget.PTX,
        validation=_real_passed_validation(),
        mlir_opt="/fake/mlir-opt",
        pipeline=("--canonicalize",),
        output_validator=_accept_backend_output,
    )
    assert result.status() is MLIRBackendStatus.FAILED
    assert any("only blank output" in f for f in result.lowering_findings)


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


def test_lower_mlir_to_backend_valid_defers_with_no_support():
    support = MLIRSupport(
        bindings=False,
        dialects=False,
        mlir_opt=None,
        detail=("`mlir-opt` is not on PATH",),
    )
    result = lower_mlir_to_backend(
        _WELL_FORMED, MLIRBackendTarget.LLVM_IR, support=support)
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
    result = lower_mlir_to_backend(
        _WELL_FORMED, MLIRBackendTarget.PTX, support=support)
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
        return _real_passed_validation()

    def _validator(output_text):
        assert output_text == "module { }\n"
        return ()

    def _fake_run(mlir_text, *, target, validation, mlir_opt, pipeline,
                  output_validator):
        assert target is MLIRBackendTarget.PTX
        assert validation.passed()
        assert mlir_opt == "/usr/bin/mlir-opt"
        assert pipeline == ("--canonicalize",)
        assert output_validator is _validator
        return MLIRBackendResult(
            target=target,
            validation=validation,
            lowering_attempted=True,
            lowering_passed=True,
            lowering_tool="mlir-opt",
            lowering_findings=(),
            output_text="module { }\n",
        )

    monkeypatch.setattr(backends, "validate_mlir_with_toolchain",
                        _fake_validate)
    monkeypatch.setitem(
        backends.MLIR_BACKEND_LOWERING_PIPELINES,
        MLIRBackendTarget.PTX,
        ("--canonicalize",),
    )
    monkeypatch.setitem(
        backends.MLIR_BACKEND_OUTPUT_VALIDATORS,
        MLIRBackendTarget.PTX,
        _validator,
    )
    monkeypatch.setattr(backends, "_run_mlir_opt_pipeline", _fake_run)
    result = lower_mlir_to_backend(
        _WELL_FORMED, MLIRBackendTarget.PTX, support=support)
    assert result.passed()
    assert result.output_text == "module { }\n"


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
    monkeypatch.setitem(
        backends.MLIR_BACKEND_LOWERING_PIPELINES,
        MLIRBackendTarget.PTX,
        ("--canonicalize",),
    )
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
    monkeypatch.setitem(
        backends.MLIR_BACKEND_LOWERING_PIPELINES,
        MLIRBackendTarget.PTX,
        ("--canonicalize",),
    )
    result = lower_mlir_to_backend(
        _WELL_FORMED, MLIRBackendTarget.PTX, support=support)
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
