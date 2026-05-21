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

from dataclasses import dataclass
from enum import Enum
from typing import Optional

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
# can return PASSED.
MLIR_BACKEND_LOWERING_PIPELINES: dict[MLIRBackendTarget, tuple[str, ...]] = {
    MLIRBackendTarget.LLVM_IR: (),
    MLIRBackendTarget.PTX: (),
    MLIRBackendTarget.ROCM_HIP: (),
    MLIRBackendTarget.METAL_MSL: (),
    MLIRBackendTarget.WEBGPU_WGSL: (),
}


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
        for pass_name in pipeline:
            if not isinstance(pass_name, str) or not pass_name.strip():
                raise AssertionError(
                    f"helixc.ir.mlir.backends: {target.name} has a "
                    f"blank / non-str pass name {pass_name!r}")


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
    if not pipeline:
        findings.append(
            f"Stage 213 MLIR lowering pipeline for {target.value} is "
            "not wired yet; Stage 214 must supply target passes before "
            "this backend can consume MLIR")
    else:
        findings.append(
            f"Stage 213 has a declared MLIR pipeline for {target.value} "
            "but no pipeline runner is wired yet; refusing to claim a "
            "backend pass")

    return MLIRBackendResult(
        target=target,
        validation=validation,
        lowering_attempted=False,
        lowering_passed=None,
        lowering_tool=None,
        lowering_findings=tuple(findings),
        output_text=None,
    )
