"""
helixc/ir/mlir/parity.py — Stage 215 MLIR-vs-tile-IR parity gate.

Stage 215 verifies the MLIR path matches the home-grown tile-IR path
across the five backend targets (LLVM_IR, PTX, ROCM_HIP, METAL_MSL,
WEBGPU_WGSL). Both paths start from the same `tile_ir.TileModule` —
the home-grown path lowers it through `helixc/backend/*.py` directly;
the MLIR path runs `emit_mlir_module` to produce MLIR text, then
`lower_mlir_to_backend` to produce the target artifact (via the
Stage 213 / 214 chain).

PARITY is the cross-path consistency check: for a given module and
target, the two paths must produce equivalent results (or both fail,
or both defer for the same reason).

Stage 215 chunk A — the harness skeleton. Defines the result type
`ParityResult` and the entry point `mlir_vs_tile_ir_parity_check`,
which:

- runs `emit_mlir_module` (the home-grown→MLIR translator) and
  catches `MLIRTranslationError` for unsupported constructs;
- runs `lower_mlir_to_backend` on the resulting MLIR text;
- returns a `ParityResult` describing the verdict.

Real cross-path equivalence checking (byte-equality of the target
artifact, semantic equivalence via re-execution, etc.) is the
chunk-B+ concern: a clean harness skeleton with documented hooks
lets the subsequent chunks fill in target-specific equivalence
checks without restructuring the API.

MOCK-PATH-FIRST: on a machine without MLIR / mlir-translate / etc.,
the MLIR side returns DEFERRED; the parity verdict is then
`PARITY_DEFERRED` ("the MLIR path is unverifiable on this machine —
cannot compare"). Fail-closed: any unexpected error in the MLIR
path bubbles into a `PARITY_FAILED` verdict with the named cause.

License: Apache 2.0
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .backends import (
    MLIRBackendResult,
    MLIRBackendStatus,
    MLIRBackendTarget,
    lower_mlir_to_backend,
)
from .emit import MLIRTranslationError, emit_mlir_module
from .toolchain import MLIRSupport
from .. import tile_ir


class ParityStatus(Enum):
    """Tri-state verdict for an MLIR-vs-tile-IR parity check.

    - PARITY_HOLDS: both paths produce equivalent results.
    - PARITY_FAILED: the two paths diverge (or the MLIR path raises
      an error the home-grown path would not).
    - PARITY_DEFERRED: parity cannot be verified on this machine —
      typically the MLIR path returned DEFERRED because a real
      toolchain is absent.
    """
    PARITY_HOLDS = "parity_holds"
    PARITY_FAILED = "parity_failed"
    PARITY_DEFERRED = "parity_deferred"


@dataclass(frozen=True, slots=True)
class ParityResult:
    """The result of an MLIR-vs-tile-IR parity check for one module
    and one backend target.

    Frozen + `__post_init__`-guarded — the house discipline of the
    Stage-213 result types (see `MLIRBackendResult`):
    - PARITY_FAILED or PARITY_DEFERRED MUST carry at least one
      finding explaining why;
    - PARITY_HOLDS MUST carry no findings;
    - the `mlir_result` attribute (when present) is the
      `MLIRBackendResult` from the MLIR path, so a caller can
      inspect the backend chain's provenance / output_text without
      re-running it.
    """
    target: MLIRBackendTarget
    status: ParityStatus
    findings: tuple[str, ...]
    mlir_result: Optional[MLIRBackendResult] = None

    def __init_subclass__(cls, **kwargs) -> None:
        raise TypeError(
            "ParityResult is final; subclassing could bypass parity-"
            "result invariants")

    def __post_init__(self) -> None:
        if not isinstance(self.target, MLIRBackendTarget):
            raise ValueError(
                "ParityResult: target must be MLIRBackendTarget, got "
                f"{self.target!r}")
        if not isinstance(self.status, ParityStatus):
            raise ValueError(
                "ParityResult: status must be ParityStatus, got "
                f"{self.status!r}")
        if not isinstance(self.findings, tuple):
            raise ValueError(
                "ParityResult: findings must be a tuple, got "
                f"{type(self.findings).__name__}")
        for entry in self.findings:
            if not isinstance(entry, str) or not entry.strip():
                raise ValueError(
                    f"ParityResult: findings has a blank or non-str "
                    f"entry ({entry!r}) — every finding carries text")
        if self.status in (ParityStatus.PARITY_FAILED,
                           ParityStatus.PARITY_DEFERRED) \
                and not self.findings:
            raise ValueError(
                f"ParityResult: a {self.status.name} result must "
                "carry at least one finding explaining why")
        if self.status is ParityStatus.PARITY_HOLDS and self.findings:
            raise ValueError(
                f"ParityResult: a PARITY_HOLDS result must carry NO "
                f"findings ({len(self.findings)} given) — a clean "
                "parity verdict has nothing to report")
        if self.mlir_result is not None \
                and type(self.mlir_result) is not MLIRBackendResult:
            raise ValueError(
                "ParityResult: mlir_result must be an MLIRBackendResult "
                "or None")

    def holds(self) -> bool:
        return self.status is ParityStatus.PARITY_HOLDS

    def failed(self) -> bool:
        return self.status is ParityStatus.PARITY_FAILED

    def deferred(self) -> bool:
        return self.status is ParityStatus.PARITY_DEFERRED


def mlir_vs_tile_ir_parity_check(
        module: tile_ir.TileModule,
        target: MLIRBackendTarget,
        *,
        support: Optional[MLIRSupport] = None) -> ParityResult:
    """Run the MLIR side of the parity gate for one Tile-IR module
    and one backend target.

    Stage 215 chunk A: this entry point does not yet compare the
    MLIR backend artifact byte-for-byte against the home-grown
    backend artifact. It runs the MLIR path end-to-end and reports a
    `ParityResult` reflecting whether parity COULD be verified:

    - MLIR translator raises `MLIRTranslationError` -> PARITY_FAILED
      with the translator finding;
    - MLIR backend chain returns FAILED -> PARITY_FAILED with the
      backend findings;
    - MLIR backend chain returns DEFERRED (no toolchain, or
      pipeline-level deferral) -> PARITY_DEFERRED with the deferral
      reason;
    - MLIR backend chain returns PASSED -> PARITY_HOLDS (chunk-B+
      replaces this with a real cross-path artifact comparison once
      the home-grown path's outputs are accessible from the same
      entry point).

    The home-grown path's run is the responsibility of chunk-B+:
    chunk A confirms the MLIR path is reachable end-to-end and the
    parity-result type is safe to construct from each MLIR verdict.
    """
    if not isinstance(module, tile_ir.TileModule):
        raise ValueError(
            "mlir_vs_tile_ir_parity_check: module must be a "
            f"tile_ir.TileModule, got {type(module).__name__}")
    if not isinstance(target, MLIRBackendTarget):
        raise ValueError(
            "mlir_vs_tile_ir_parity_check: target must be "
            f"MLIRBackendTarget, got {target!r}")

    try:
        mlir_text = emit_mlir_module(module)
    except MLIRTranslationError as exc:
        return ParityResult(
            target=target,
            status=ParityStatus.PARITY_FAILED,
            findings=(
                f"MLIR translator failed for {target.value}: "
                f"{type(exc).__name__}: {exc}",),
            mlir_result=None,
        )

    mlir_result = lower_mlir_to_backend(
        mlir_text, target, support=support)
    backend_status = mlir_result.status()
    if backend_status is MLIRBackendStatus.PASSED:
        return ParityResult(
            target=target,
            status=ParityStatus.PARITY_HOLDS,
            findings=(),
            mlir_result=mlir_result,
        )
    if backend_status is MLIRBackendStatus.FAILED:
        return ParityResult(
            target=target,
            status=ParityStatus.PARITY_FAILED,
            findings=(
                f"MLIR backend chain for {target.value} failed: "
                + (mlir_result.lowering_findings[0]
                   if mlir_result.lowering_findings
                   else "no lowering finding emitted"),
            ),
            mlir_result=mlir_result,
        )
    # DEFERRED — usually no real toolchain on this machine.
    return ParityResult(
        target=target,
        status=ParityStatus.PARITY_DEFERRED,
        findings=(
            f"MLIR backend chain for {target.value} deferred — parity "
            "cannot be verified without a real toolchain: "
            + (mlir_result.lowering_findings[0]
               if mlir_result.lowering_findings
               else "no deferral reason emitted"),
        ),
        mlir_result=mlir_result,
    )
