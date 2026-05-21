"""
helixc/ir/mlir/backends.py - MLIR backend-lowering scaffold
(v3.0 Phase E, Stage 213 chunk A).

Stage 213 starts the "MLIR -> backends" seam. This module deliberately
does NOT claim that any backend consumes MLIR yet: it defines the five
targets, records the dialect contract each target will need, validates
MLIR text through the existing mock path, and returns a frozen
PASSED/FAILED/DEFERRED result.

Until Stage 214 supplies the target pass pipelines, valid MLIR returns
DEFERRED with an explicit reason. That is the Stage 210 mock-path rule:
no MLIR toolchain on this machine must never become a false pass, and
the legacy Tile-IR backend path remains the fallback.

License: Apache 2.0
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional

from helixc.backend.gpu_ci import BackendKind as GPUBackendKind

from .toolchain import MLIRSupport, detect_mlir_support
from .validate import (
    MLIRValidation, mock_validate_mlir, validate_mlir_with_toolchain,
)


class MLIRBackendTarget(Enum):
    """The backend targets Stage 213 must eventually lower MLIR into."""
    LLVM_IR = "llvm_ir"
    PTX = "ptx"
    ROCM_HIP = "rocm_hip"
    METAL_MSL = "metal_msl"
    WEBGPU_WGSL = "webgpu_wgsl"


class MLIRBackendStatus(Enum):
    """Tri-state result for MLIR backend lowering."""
    PASSED = "passed"
    FAILED = "failed"
    DEFERRED = "deferred"


MLIR_BACKEND_TARGETS: tuple[MLIRBackendTarget, ...] = (
    MLIRBackendTarget.LLVM_IR,
    MLIRBackendTarget.PTX,
    MLIRBackendTarget.ROCM_HIP,
    MLIRBackendTarget.METAL_MSL,
    MLIRBackendTarget.WEBGPU_WGSL,
)


GPU_BACKEND_TO_MLIR_TARGET: dict[GPUBackendKind, MLIRBackendTarget] = {
    GPUBackendKind.PTX: MLIRBackendTarget.PTX,
    GPUBackendKind.ROCM_HIP: MLIRBackendTarget.ROCM_HIP,
    GPUBackendKind.METAL_MSL: MLIRBackendTarget.METAL_MSL,
    GPUBackendKind.WEBGPU_WGSL: MLIRBackendTarget.WEBGPU_WGSL,
}


MLIR_BACKEND_REQUIRED_DIALECTS: dict[MLIRBackendTarget, tuple[str, ...]] = {
    MLIRBackendTarget.LLVM_IR: (
        "func", "arith", "cf", "scf", "memref", "linalg", "vector",
        "llvm",
    ),
    MLIRBackendTarget.PTX: (
        "func", "arith", "memref", "gpu", "nvgpu", "nvvm",
    ),
    MLIRBackendTarget.ROCM_HIP: (
        "func", "arith", "memref", "gpu", "rocdl",
    ),
    MLIRBackendTarget.METAL_MSL: (
        "func", "arith", "memref", "gpu", "spirv",
    ),
    MLIRBackendTarget.WEBGPU_WGSL: (
        "func", "arith", "memref", "gpu", "spirv",
    ),
}


# Stage 213 chunk A records the table and leaves every target empty on
# purpose. A future chunk must fill this table and teach
# `lower_mlir_to_backend` how to execute those passes before any target
# can return PASSED. Stage 213 chunk C defines the runner contract:
# each entry is a complete `mlir-opt` pass argument (e.g.
# "--canonicalize" or "--pass-pipeline=..."), not a shell fragment.
MLIR_BACKEND_LOWERING_PIPELINES: dict[MLIRBackendTarget, tuple[str, ...]] = {
    MLIRBackendTarget.LLVM_IR: (),
    MLIRBackendTarget.PTX: (),
    MLIRBackendTarget.ROCM_HIP: (),
    MLIRBackendTarget.METAL_MSL: (),
    MLIRBackendTarget.WEBGPU_WGSL: (),
}

# A target pipeline alone is not enough to claim a backend pass: after
# `mlir-opt` runs, a target-specific validator must prove the output is
# the backend-consumable artifact Stage 214+ promised. Stage 213 leaves
# every validator unwired, so production lowering remains DEFERRED even
# if a test or future branch experiments with a non-empty pipeline.
MLIRBackendOutputValidator = Callable[[str], tuple[str, ...]]

MLIR_BACKEND_OUTPUT_VALIDATORS: dict[
    MLIRBackendTarget, Optional[MLIRBackendOutputValidator]] = {
        MLIRBackendTarget.LLVM_IR: None,
        MLIRBackendTarget.PTX: None,
        MLIRBackendTarget.ROCM_HIP: None,
        MLIRBackendTarget.METAL_MSL: None,
        MLIRBackendTarget.WEBGPU_WGSL: None,
}

# Wall-clock cap on a target pass-pipeline dispatch. Real production
# pipelines should be short for the small modules exercised here; the
# cap is only a dead-tool guard.
_MLIR_BACKEND_PIPELINE_TIMEOUT_S = 60


def _check_mlir_backend_tables() -> None:
    """Module-load drift guard for the Stage 213 backend tables."""
    expected = set(MLIRBackendTarget)
    if set(MLIR_BACKEND_TARGETS) != expected:
        raise AssertionError(
            "helixc.ir.mlir.backends: MLIR_BACKEND_TARGETS must cover "
            f"exactly {expected}, got {set(MLIR_BACKEND_TARGETS)}")

    expected_values = {
        "llvm_ir", "ptx", "rocm_hip", "metal_msl", "webgpu_wgsl",
    }
    if {target.value for target in MLIRBackendTarget} != expected_values:
        raise AssertionError(
            "helixc.ir.mlir.backends: MLIRBackendTarget values drifted "
            f"from {expected_values}")

    gpu_expected = set(GPUBackendKind)
    if set(GPU_BACKEND_TO_MLIR_TARGET) != gpu_expected:
        raise AssertionError(
            "helixc.ir.mlir.backends: GPU_BACKEND_TO_MLIR_TARGET keys "
            f"{set(GPU_BACKEND_TO_MLIR_TARGET)} != {gpu_expected}")
    gpu_targets = expected - {MLIRBackendTarget.LLVM_IR}
    if set(GPU_BACKEND_TO_MLIR_TARGET.values()) != gpu_targets:
        raise AssertionError(
            "helixc.ir.mlir.backends: GPU_BACKEND_TO_MLIR_TARGET values "
            f"{set(GPU_BACKEND_TO_MLIR_TARGET.values())} != "
            f"{gpu_targets}")

    for table_name, table in (
        ("MLIR_BACKEND_REQUIRED_DIALECTS", MLIR_BACKEND_REQUIRED_DIALECTS),
        ("MLIR_BACKEND_LOWERING_PIPELINES",
         MLIR_BACKEND_LOWERING_PIPELINES),
        ("MLIR_BACKEND_OUTPUT_VALIDATORS",
         MLIR_BACKEND_OUTPUT_VALIDATORS),
    ):
        if set(table) != expected:
            raise AssertionError(
                f"helixc.ir.mlir.backends: {table_name} keys "
                f"{set(table)} != MLIRBackendTarget members {expected}")

    for target, dialects in MLIR_BACKEND_REQUIRED_DIALECTS.items():
        if not dialects:
            raise AssertionError(
                f"helixc.ir.mlir.backends: {target.name} has no "
                "required dialects")
        if len(dialects) != len(set(dialects)):
            raise AssertionError(
                f"helixc.ir.mlir.backends: {target.name} has duplicate "
                f"dialect entries {dialects}")
        for dialect in dialects:
            if not isinstance(dialect, str) or not dialect.isidentifier():
                raise AssertionError(
                    f"helixc.ir.mlir.backends: {target.name} has a "
                    f"blank / non-identifier dialect {dialect!r}")

    for target, pipeline in MLIR_BACKEND_LOWERING_PIPELINES.items():
        if not isinstance(pipeline, tuple):
            raise AssertionError(
                f"helixc.ir.mlir.backends: {target.name} pipeline must "
                f"be a tuple, got {type(pipeline).__name__}")
        for pass_arg in pipeline:
            if not isinstance(pass_arg, str) or not pass_arg.strip():
                raise AssertionError(
                    f"helixc.ir.mlir.backends: {target.name} has a "
                    f"blank / non-str pass argument {pass_arg!r}")
            if pass_arg != pass_arg.strip():
                raise AssertionError(
                    f"helixc.ir.mlir.backends: {target.name} pass "
                    f"argument {pass_arg!r} has leading/trailing "
                    "whitespace")
            if not pass_arg.startswith("--"):
                raise AssertionError(
                    f"helixc.ir.mlir.backends: {target.name} pass "
                    f"argument {pass_arg!r} must start with '--' so it "
                    "is a complete argv token, not an implicit shell "
                    "fragment")

    for target, validator in MLIR_BACKEND_OUTPUT_VALIDATORS.items():
        if validator is not None and not callable(validator):
            raise AssertionError(
                f"helixc.ir.mlir.backends: {target.name} output "
                f"validator must be callable or None, got "
                f"{type(validator).__name__}")


_check_mlir_backend_tables()


def _require_backend_target(target: MLIRBackendTarget) -> MLIRBackendTarget:
    if not isinstance(target, MLIRBackendTarget):
        raise ValueError(
            f"unknown MLIR backend target {target!r}; expected one of "
            f"{list(MLIRBackendTarget)}")
    return target


def backend_required_dialects(
        target: MLIRBackendTarget) -> tuple[str, ...]:
    """Return the MLIR dialect contract for a backend target."""
    target = _require_backend_target(target)
    return MLIR_BACKEND_REQUIRED_DIALECTS[target]


def backend_lowering_pipeline(
        target: MLIRBackendTarget) -> tuple[str, ...]:
    """Return the Stage 214 pass pipeline for a backend target.

    Stage 213 chunk A intentionally returns an empty tuple for every
    target. Empty means "not wired", not "no passes needed".
    """
    target = _require_backend_target(target)
    return MLIR_BACKEND_LOWERING_PIPELINES[target]


def mlir_target_for_gpu_backend(
        backend: GPUBackendKind) -> MLIRBackendTarget:
    """Map an existing GPU backend enum to the Stage 213 MLIR target."""
    if not isinstance(backend, GPUBackendKind):
        raise ValueError(
            f"unknown GPU backend {backend!r}; expected one of "
            f"{list(GPUBackendKind)}")
    return GPU_BACKEND_TO_MLIR_TARGET[backend]


def _run_mlir_opt_pipeline(
        mlir_text: str,
        *,
        target: MLIRBackendTarget,
        validation: MLIRValidation,
        mlir_opt: str,
        pipeline: tuple[str, ...],
        output_validator: MLIRBackendOutputValidator,
        timeout_s: int = _MLIR_BACKEND_PIPELINE_TIMEOUT_S,
) -> "MLIRBackendResult":
    """Run a declared `mlir-opt` target lowering pipeline.

    This is the Stage 213 chunk-C runner contract. It only runs after
    real validation has PASSED. A 0 exit is not enough: the output
    artifact must exist, be non-empty, and read back as text before the
    backend result can be PASSED.
    """
    target = _require_backend_target(target)
    if not isinstance(mlir_text, str) or not mlir_text.strip():
        raise ValueError(
            "_run_mlir_opt_pipeline: mlir_text must be non-empty text")
    if not isinstance(validation, MLIRValidation):
        raise ValueError(
            "_run_mlir_opt_pipeline: validation must be an "
            f"MLIRValidation result, got {validation!r}")
    if not validation.passed():
        raise ValueError(
            "_run_mlir_opt_pipeline: validation must be PASSED before "
            "a backend lowering pipeline can run")
    if not isinstance(mlir_opt, str) or not mlir_opt.strip():
        raise ValueError(
            "_run_mlir_opt_pipeline: mlir_opt must be a non-empty "
            f"string, got {mlir_opt!r}")
    mlir_opt = mlir_opt.strip()
    if not isinstance(pipeline, tuple) or not pipeline:
        raise ValueError(
            "_run_mlir_opt_pipeline: pipeline must be a non-empty tuple")
    for pass_arg in pipeline:
        if not isinstance(pass_arg, str) or not pass_arg.strip():
            raise ValueError(
                "_run_mlir_opt_pipeline: pipeline has a blank or "
                f"non-str pass argument {pass_arg!r}")
        if pass_arg != pass_arg.strip():
            raise ValueError(
                "_run_mlir_opt_pipeline: pipeline pass argument "
                f"{pass_arg!r} must not have leading/trailing whitespace")
        if not pass_arg.startswith("--"):
            raise ValueError(
                "_run_mlir_opt_pipeline: pipeline pass argument "
                f"{pass_arg!r} must start with '--'")
    if not callable(output_validator):
        raise ValueError(
            "_run_mlir_opt_pipeline: output_validator must be callable")
    if ((not isinstance(timeout_s, (int, float)))
            or isinstance(timeout_s, bool)
            or timeout_s <= 0):
        raise ValueError(
            "_run_mlir_opt_pipeline: timeout_s must be a positive number")

    with tempfile.TemporaryDirectory(prefix="helix_mlir_backend_") as tmpdir:
        mlir_path = os.path.join(tmpdir, "module.mlir")
        out_path = os.path.join(tmpdir, "lowered.mlir")
        try:
            with open(mlir_path, "w", encoding="utf-8") as f:
                f.write(mlir_text)
        except OSError as exc:
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool="mlir-opt",
                lowering_findings=(
                    f"could not write temp MLIR input {mlir_path!r} "
                    f"({type(exc).__name__}: {exc})",),
                output_text=None,
            )

        try:
            proc = subprocess.run(
                [mlir_opt, *pipeline, mlir_path, "-o", out_path],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool="mlir-opt",
                lowering_findings=(
                    f"mlir-opt backend pipeline for {target.value} "
                    f"timed out after {timeout_s}s",),
                output_text=None,
            )
        except OSError as exc:
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool="mlir-opt",
                lowering_findings=(
                    f"mlir-opt backend pipeline for {target.value}: "
                    f"tool unusable at invocation ({type(exc).__name__}: "
                    f"{exc})",),
                output_text=None,
            )

        if proc.returncode != 0:
            diag = (proc.stderr or proc.stdout or "").strip()
            if not diag:
                diag = "no diagnostic emitted"
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool="mlir-opt",
                lowering_findings=(
                    f"mlir-opt backend pipeline for {target.value} exit "
                    f"{proc.returncode}: {diag[:500]}",),
                output_text=None,
            )

        try:
            size = os.path.getsize(out_path)
        except OSError:
            size = -1
        if size <= 0:
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool="mlir-opt",
                lowering_findings=(
                    f"mlir-opt backend pipeline for {target.value} "
                    f"exited 0 but produced no output artifact at "
                    f"{out_path!r} - a 0 exit with no artifact is not "
                    "a backend pass",),
                output_text=None,
            )

        try:
            with open(out_path, "r", encoding="utf-8") as f:
                output_text = f.read()
        except OSError as exc:
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool="mlir-opt",
                lowering_findings=(
                    f"could not read backend output artifact {out_path!r} "
                    f"({type(exc).__name__}: {exc})",),
                output_text=None,
            )

        if not output_text.strip():
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool="mlir-opt",
                lowering_findings=(
                    f"mlir-opt backend pipeline for {target.value} "
                    "exited 0 but produced only blank output",),
                output_text=None,
            )
        output_findings = output_validator(output_text)
        if not isinstance(output_findings, tuple):
            raise ValueError(
                "_run_mlir_opt_pipeline: output_validator must return a "
                f"tuple, got {type(output_findings).__name__}")
        for finding in output_findings:
            if not isinstance(finding, str) or not finding.strip():
                raise ValueError(
                    "_run_mlir_opt_pipeline: output_validator returned "
                    f"a blank or non-str finding {finding!r}")
        if output_findings:
            return MLIRBackendResult(
                target=target,
                validation=validation,
                lowering_attempted=True,
                lowering_passed=False,
                lowering_tool="mlir-opt",
                lowering_findings=tuple(
                    f"target output contract for {target.value}: {finding}"
                    for finding in output_findings),
                output_text=None,
            )

    return MLIRBackendResult(
        target=target,
        validation=validation,
        lowering_attempted=True,
        lowering_passed=True,
        lowering_tool="mlir-opt",
        lowering_findings=(),
        output_text=output_text,
    )


@dataclass(frozen=True)
class MLIRBackendResult:
    """Outcome of trying to lower MLIR text to one backend target.

    The mock structural validator always runs first. Real target
    lowering is represented separately and must never be silent:
    - malformed MLIR fails before any lowering attempt;
    - a valid mock shape plus no target pipeline is DEFERRED with a
      finding explaining why;
    - a real lowering failure must carry at least one diagnostic;
    - a real lowering pass must carry concrete backend output text.
    """
    target: MLIRBackendTarget
    validation: MLIRValidation
    lowering_attempted: bool
    lowering_passed: Optional[bool]
    lowering_tool: Optional[str]
    lowering_findings: tuple[str, ...]
    output_text: Optional[str] = None

    def __post_init__(self) -> None:
        _require_backend_target(self.target)
        if not isinstance(self.validation, MLIRValidation):
            raise ValueError(
                "MLIRBackendResult: validation must be an MLIRValidation "
                f"result, got {self.validation!r}")
        if not isinstance(self.lowering_attempted, bool):
            raise ValueError(
                "MLIRBackendResult: lowering_attempted must be a bool, "
                f"got {self.lowering_attempted!r}")
        if not isinstance(self.lowering_findings, tuple):
            raise ValueError(
                "MLIRBackendResult: lowering_findings must be a tuple, "
                f"got {type(self.lowering_findings).__name__}")
        if self.output_text is not None:
            if not isinstance(self.output_text, str):
                raise ValueError(
                    "MLIRBackendResult: output_text must be a str or "
                    f"None, got {type(self.output_text).__name__}")
            if not self.output_text.strip():
                raise ValueError(
                    "MLIRBackendResult: output_text must carry text "
                    "when present")

        for entry in self.lowering_findings:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    "MLIRBackendResult: lowering_findings has a blank "
                    f"or non-str entry ({entry!r})")

        if not self.lowering_attempted:
            if self.lowering_passed is not None:
                raise ValueError(
                    "MLIRBackendResult: lowering_attempted=False but "
                    f"lowering_passed={self.lowering_passed!r}")
            if self.lowering_tool is not None:
                raise ValueError(
                    "MLIRBackendResult: lowering_attempted=False but "
                    f"lowering_tool={self.lowering_tool!r}")
            if self.output_text is not None:
                raise ValueError(
                    "MLIRBackendResult: lowering_attempted=False but "
                    "output_text is present")
            if not self.validation.failed() and not self.lowering_findings:
                raise ValueError(
                    "MLIRBackendResult: mock-valid MLIR with no lowering "
                    "attempt must carry at least one finding explaining "
                    "why lowering is DEFERRED")
            return

        if self.validation.failed():
            raise ValueError(
                "MLIRBackendResult: cannot attempt backend lowering after "
                "MLIR validation FAILED")
        if not self.validation.passed():
            raise ValueError(
                "MLIRBackendResult: attempted backend lowering requires "
                "validation to be PASSED; mock-deferred validation cannot "
                "be used for a real backend attempt")
        if self.lowering_passed is None:
            raise ValueError(
                "MLIRBackendResult: lowering_attempted=True but "
                "lowering_passed is None")
        if not isinstance(self.lowering_passed, bool):
            raise ValueError(
                "MLIRBackendResult: lowering_attempted=True requires "
                "lowering_passed to be a bool, got "
                f"{self.lowering_passed!r}")
        if (not isinstance(self.lowering_tool, str)
                or not self.lowering_tool.strip()):
            raise ValueError(
                "MLIRBackendResult: lowering_attempted=True requires a "
                "non-empty lowering_tool")
        if self.lowering_passed is False and not self.lowering_findings:
            raise ValueError(
                "MLIRBackendResult: lowering_passed=False but "
                "lowering_findings is empty - a real lowering failure "
                "must carry a diagnostic")
        if self.lowering_passed is True:
            if not self.validation.passed():
                raise ValueError(
                    "MLIRBackendResult: lowering_passed=True requires "
                    "validation to be PASSED; mock-deferred validation "
                    "cannot be promoted to a backend pass")
            if self.lowering_findings:
                raise ValueError(
                    "MLIRBackendResult: lowering_passed=True but "
                    "lowering_findings is non-empty")
            if self.output_text is None:
                raise ValueError(
                    "MLIRBackendResult: lowering_passed=True requires "
                    "non-empty output_text")

    def status(self) -> MLIRBackendStatus:
        if self.validation.failed():
            return MLIRBackendStatus.FAILED
        if not self.lowering_attempted:
            return MLIRBackendStatus.DEFERRED
        if self.lowering_passed is False:
            return MLIRBackendStatus.FAILED
        if self.lowering_passed is True:
            return MLIRBackendStatus.PASSED
        raise AssertionError(
            "MLIRBackendResult.status reached an illegal state: "
            f"lowering_passed={self.lowering_passed!r}")

    def passed(self) -> bool:
        return self.status() is MLIRBackendStatus.PASSED

    def failed(self) -> bool:
        return self.status() is MLIRBackendStatus.FAILED

    def deferred(self) -> bool:
        return self.status() is MLIRBackendStatus.DEFERRED


def lower_mlir_to_backend(
        mlir_text: str,
        target: MLIRBackendTarget,
        *,
        support: Optional[MLIRSupport] = None) -> MLIRBackendResult:
    """Validate MLIR text and try to lower it to one backend target.

    Stage 213 chunk A only establishes the seam. If the text is
    structurally malformed, the result is FAILED and no target lowering
    is attempted. If the text has a clean mock shape, the result is
    DEFERRED until a real MLIR surface and a target lowering pipeline
    are both wired.
    """
    target = _require_backend_target(target)
    mock_validation = mock_validate_mlir(mlir_text)
    if mock_validation.failed():
        return MLIRBackendResult(
            target=target,
            validation=mock_validation,
            lowering_attempted=False,
            lowering_passed=None,
            lowering_tool=None,
            lowering_findings=(),
            output_text=None,
        )

    if support is None:
        support = detect_mlir_support()
    if not isinstance(support, MLIRSupport):
        raise ValueError(
            "lower_mlir_to_backend: support must be an MLIRSupport "
            f"or None, got {support!r}")

    validation = validate_mlir_with_toolchain(
        mlir_text, support=support)
    if validation.failed():
        return MLIRBackendResult(
            target=target,
            validation=validation,
            lowering_attempted=False,
            lowering_passed=None,
            lowering_tool=None,
            lowering_findings=(),
            output_text=None,
        )

    findings: list[str] = []
    if not support.is_available():
        findings.extend(
            f"no real MLIR surface available: {line}"
            for line in support.detail)

    pipeline = backend_lowering_pipeline(target)
    output_validator = MLIR_BACKEND_OUTPUT_VALIDATORS[target]
    if not pipeline:
        findings.append(
            f"Stage 213 MLIR lowering pipeline for {target.value} is "
            "not wired yet; Stage 214 must supply target passes before "
            "this backend can consume MLIR")
    elif not validation.passed():
        findings.append(
            f"Stage 213 MLIR lowering pipeline for {target.value} is "
            "declared, but real MLIR validation is not PASSED; "
            "refusing to attempt backend lowering")
    elif support.mlir_opt is None:
        findings.append(
            f"Stage 213 MLIR lowering pipeline for {target.value} is "
            "declared, but `mlir-opt` is not available to run it")
    elif output_validator is None:
        findings.append(
            f"Stage 213 MLIR lowering pipeline for {target.value} is "
            "declared, but the target output validator is not wired; "
            "refusing to claim backend output from a pass pipeline alone")
    else:
        return _run_mlir_opt_pipeline(
            mlir_text,
            target=target,
            validation=validation,
            mlir_opt=support.mlir_opt,
            pipeline=pipeline,
            output_validator=output_validator,
        )

    return MLIRBackendResult(
        target=target,
        validation=validation,
        lowering_attempted=False,
        lowering_passed=None,
        lowering_tool=None,
        lowering_findings=tuple(findings),
        output_text=None,
    )
