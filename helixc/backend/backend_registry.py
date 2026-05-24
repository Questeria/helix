"""
helixc/backend/backend_registry.py — Stage 220 chunk A: shared
`Backend` interface across the four GPU targets (PTX / ROCm HIP /
Metal MSL / WebGPU WGSL).

Phase F's first stage unifies the backend surface. Each Helix
backend exports the same three artifacts:

  - an emitter class with `emit_module(TileModule) -> str`;
  - a module-level `lowering_status(kind: TileOpKind) -> str` query;
  - per-target lowering tables and constants (target string,
    required dialects, etc.).

Before Stage 220, downstream code (the CLI driver, the multi-target
test runner, autotuner) had to know each backend's module name and
import path. The new `Backend` dataclass + `BACKEND_REGISTRY`
collapses that into a single source of truth:

  >>> from helixc.backend.backend_registry import get_backend
  >>> backend = get_backend(MLIRBackendTarget.PTX)
  >>> text = backend.emit_module(tile_module)
  >>> backend.lowering_status(tile_ir.TileOpKind.SCALAR_ADD)
  'supported'

LLVM_IR is intentionally absent from this Stage 220 chunk A
registry: the home-grown LLVM path operates on `tir.Module`, not
`tile_ir.TileModule`, so it cannot share the `emit_module` signature
the GPU targets use. Stage 221's cutover will redesign the
boundary; Stage 220 chunks B+ can add a bridge adapter if needed
before then.

License: Apache 2.0
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from ..ir import tile_ir as ti
from ..ir.mlir.backends import (
    MLIRBackendTarget,
    MLIR_BACKEND_REQUIRED_DIALECTS,
)
from ._lowering_schema import BackendEmitter, LoweringStatus


@dataclass(frozen=True, slots=True)
class Backend:
    """The unified backend interface: a target identifier, an emitter
    factory, the lowering-status query, and the MLIR dialect contract.

    Frozen + `__post_init__`-guarded — the Phase-E house discipline
    for cross-cutting result types. Three runtime invariants:

    - `target` must be one of the canonical `MLIRBackendTarget`
      members (so a typo in a registry registration fails at module
      load, not at first call).
    - `emit_factory` must build a `BackendEmitter` (the per-backend
      class with `emit_module(TileModule) -> str`).
    - `lowering_status` must accept a `TileOpKind` and return one of
      the closed `LoweringStatus` literals.
    - `required_dialects` must be a non-empty tuple of unique
      identifier-shaped strings (mirroring the
      `MLIR_BACKEND_REQUIRED_DIALECTS` contract).

    The dataclass is final (no subclassing) so a downstream
    extension cannot bypass the invariants via override.
    """
    target: MLIRBackendTarget
    emit_factory: Callable[[], BackendEmitter]
    lowering_status: Callable[[ti.TileOpKind], LoweringStatus]
    required_dialects: tuple[str, ...]

    def __init_subclass__(cls, **kwargs) -> None:
        raise TypeError(
            "Backend is final; subclassing could bypass the unified-"
            "interface invariants")

    def __post_init__(self) -> None:
        if not isinstance(self.target, MLIRBackendTarget):
            raise ValueError(
                "Backend: target must be MLIRBackendTarget, got "
                f"{self.target!r}")
        if not callable(self.emit_factory):
            raise ValueError(
                "Backend: emit_factory must be callable, got "
                f"{type(self.emit_factory).__name__}")
        if not callable(self.lowering_status):
            raise ValueError(
                "Backend: lowering_status must be callable, got "
                f"{type(self.lowering_status).__name__}")
        if not isinstance(self.required_dialects, tuple):
            raise ValueError(
                "Backend: required_dialects must be a tuple, got "
                f"{type(self.required_dialects).__name__}")
        if not self.required_dialects:
            raise ValueError(
                "Backend: required_dialects must be non-empty")
        if len(self.required_dialects) != len(set(self.required_dialects)):
            raise ValueError(
                "Backend: required_dialects has duplicates: "
                f"{self.required_dialects}")
        for dialect in self.required_dialects:
            if not isinstance(dialect, str) or not dialect.isidentifier():
                raise ValueError(
                    f"Backend: required_dialects has a non-identifier "
                    f"entry {dialect!r}")

    def emit_module(self, module: ti.TileModule) -> str:
        """Convenience: build a fresh emitter and emit one module."""
        emitter = self.emit_factory()
        if not isinstance(emitter, BackendEmitter):
            raise TypeError(
                f"Backend.emit_module: factory for {self.target.value} "
                f"returned non-BackendEmitter "
                f"({type(emitter).__name__})")
        return emitter.emit_module(module)


_BACKEND_REGISTRY: dict[MLIRBackendTarget, Backend] = {}


def _build_registry() -> None:
    """Lazy-build the global `_BACKEND_REGISTRY`. Idempotent. Imports
    the per-backend emitter classes only when first called so a
    binding-less / partial-install machine still imports this module
    cleanly."""
    global _BACKEND_REGISTRY
    if _BACKEND_REGISTRY:
        return
    from . import ptx, rocm, metal, webgpu
    _BACKEND_REGISTRY = {
        MLIRBackendTarget.PTX: Backend(
            target=MLIRBackendTarget.PTX,
            emit_factory=ptx.PtxEmitter,
            lowering_status=ptx.lowering_status,
            required_dialects=MLIR_BACKEND_REQUIRED_DIALECTS[
                MLIRBackendTarget.PTX],
        ),
        MLIRBackendTarget.ROCM_HIP: Backend(
            target=MLIRBackendTarget.ROCM_HIP,
            emit_factory=rocm.HipEmitter,
            lowering_status=rocm.lowering_status,
            required_dialects=MLIR_BACKEND_REQUIRED_DIALECTS[
                MLIRBackendTarget.ROCM_HIP],
        ),
        MLIRBackendTarget.METAL_MSL: Backend(
            target=MLIRBackendTarget.METAL_MSL,
            emit_factory=metal.MslEmitter,
            lowering_status=metal.lowering_status,
            required_dialects=MLIR_BACKEND_REQUIRED_DIALECTS[
                MLIRBackendTarget.METAL_MSL],
        ),
        MLIRBackendTarget.WEBGPU_WGSL: Backend(
            target=MLIRBackendTarget.WEBGPU_WGSL,
            emit_factory=webgpu.WgslEmitter,
            lowering_status=webgpu.lowering_status,
            required_dialects=MLIR_BACKEND_REQUIRED_DIALECTS[
                MLIRBackendTarget.WEBGPU_WGSL],
        ),
    }


def get_backend(target: MLIRBackendTarget) -> Backend:
    """Return the unified `Backend` for the named target.

    Raises ValueError for LLVM_IR (the home-grown LLVM path
    consumes `tir.Module`, not `TileModule`, and is not in the
    Stage 220 chunk-A registry — Stage 221 cutover will revisit).
    """
    if not isinstance(target, MLIRBackendTarget):
        raise ValueError(
            "get_backend: target must be MLIRBackendTarget, got "
            f"{target!r}")
    if target is MLIRBackendTarget.LLVM_IR:
        raise ValueError(
            "get_backend: LLVM_IR is not in the unified backend "
            "registry (its home-grown emitter takes tir.Module, not "
            "TileModule); use helixc.backend.llvm_ir.emit_module "
            "directly, or wait for Stage 221's cutover bridge")
    _build_registry()
    backend = _BACKEND_REGISTRY.get(target)
    if backend is None:
        raise KeyError(
            f"get_backend: {target.value} is not registered "
            "(unexpected — every non-LLVM target should be wired)")
    return backend


def registered_targets() -> tuple[MLIRBackendTarget, ...]:
    """Return the targets currently in the unified registry."""
    _build_registry()
    return tuple(_BACKEND_REGISTRY)


def _check_registry_coverage() -> None:
    """Drift guard: every non-LLVM `MLIRBackendTarget` must have a
    `Backend` registered after `_build_registry()` runs. Called by
    tests to pin the contract."""
    _build_registry()
    expected = set(MLIRBackendTarget) - {MLIRBackendTarget.LLVM_IR}
    actual = set(_BACKEND_REGISTRY)
    if actual != expected:
        raise AssertionError(
            f"helixc.backend.backend_registry: registry covers "
            f"{actual} but should cover {expected}")
