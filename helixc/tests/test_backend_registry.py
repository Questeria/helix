"""Tests for helixc.backend.backend_registry — Stage 220 chunk A.

The unified `Backend` interface across the four GPU targets. LLVM_IR
is intentionally absent (its home-grown emitter consumes tir.Module,
not TileModule).
"""
from __future__ import annotations

import pytest

from helixc.backend import backend_registry
from helixc.backend._lowering_schema import BackendEmitter
from helixc.backend.backend_registry import (
    Backend,
    _check_registry_coverage,
    get_backend,
    registered_targets,
)
from helixc.ir import tile_ir
from helixc.ir.mlir.backends import (
    MLIRBackendTarget,
    MLIR_BACKEND_REQUIRED_DIALECTS,
)


# --------------------------------------------------------------------------
# Backend dataclass — __post_init__ invariants
# --------------------------------------------------------------------------
def test_backend_is_final():
    with pytest.raises(TypeError, match="final"):
        class _Subclass(Backend):  # type: ignore[misc]
            pass


def test_backend_rejects_non_target():
    with pytest.raises(ValueError, match="target must be"):
        Backend(
            target="ptx",  # type: ignore[arg-type]
            emit_factory=lambda: None,  # type: ignore[arg-type]
            lowering_status=lambda k: "supported",  # type: ignore[arg-type]
            required_dialects=("func",),
        )


def test_backend_rejects_non_callable_factory():
    with pytest.raises(ValueError, match="emit_factory must be callable"):
        Backend(
            target=MLIRBackendTarget.PTX,
            emit_factory="not callable",  # type: ignore[arg-type]
            lowering_status=lambda k: "supported",  # type: ignore[arg-type]
            required_dialects=("func",),
        )


def test_backend_rejects_non_callable_status():
    with pytest.raises(ValueError, match="lowering_status must be callable"):
        Backend(
            target=MLIRBackendTarget.PTX,
            emit_factory=lambda: None,  # type: ignore[arg-type]
            lowering_status="not callable",  # type: ignore[arg-type]
            required_dialects=("func",),
        )


def test_backend_rejects_non_tuple_dialects():
    with pytest.raises(ValueError, match="required_dialects must be a tuple"):
        Backend(
            target=MLIRBackendTarget.PTX,
            emit_factory=lambda: None,  # type: ignore[arg-type]
            lowering_status=lambda k: "supported",  # type: ignore[arg-type]
            required_dialects=["func"],  # type: ignore[arg-type]
        )


def test_backend_rejects_empty_dialects():
    with pytest.raises(ValueError, match="required_dialects must be non-empty"):
        Backend(
            target=MLIRBackendTarget.PTX,
            emit_factory=lambda: None,  # type: ignore[arg-type]
            lowering_status=lambda k: "supported",  # type: ignore[arg-type]
            required_dialects=(),
        )


def test_backend_rejects_duplicate_dialects():
    with pytest.raises(ValueError, match="duplicates"):
        Backend(
            target=MLIRBackendTarget.PTX,
            emit_factory=lambda: None,  # type: ignore[arg-type]
            lowering_status=lambda k: "supported",  # type: ignore[arg-type]
            required_dialects=("func", "func"),
        )


def test_backend_rejects_non_identifier_dialect():
    with pytest.raises(ValueError, match="non-identifier"):
        Backend(
            target=MLIRBackendTarget.PTX,
            emit_factory=lambda: None,  # type: ignore[arg-type]
            lowering_status=lambda k: "supported",  # type: ignore[arg-type]
            required_dialects=("func", "with-hyphen"),
        )


# --------------------------------------------------------------------------
# Registry coverage + lookup
# --------------------------------------------------------------------------
def test_registry_covers_all_gpu_targets():
    """Stage 220 chunk A: every non-LLVM backend target must have a
    Backend registered. The drift guard runs at import / first
    lookup; this test pins it explicitly."""
    _check_registry_coverage()
    expected = set(MLIRBackendTarget) - {MLIRBackendTarget.LLVM_IR}
    assert set(registered_targets()) == expected


def test_get_backend_returns_correct_type():
    for target in (MLIRBackendTarget.PTX, MLIRBackendTarget.ROCM_HIP,
                   MLIRBackendTarget.METAL_MSL,
                   MLIRBackendTarget.WEBGPU_WGSL):
        backend = get_backend(target)
        assert isinstance(backend, Backend)
        assert backend.target is target


def test_get_backend_dialects_match_authority_table():
    """The unified Backend's `required_dialects` must come from
    `MLIR_BACKEND_REQUIRED_DIALECTS` — single source of truth."""
    for target in (MLIRBackendTarget.PTX, MLIRBackendTarget.ROCM_HIP,
                   MLIRBackendTarget.METAL_MSL,
                   MLIRBackendTarget.WEBGPU_WGSL):
        backend = get_backend(target)
        assert backend.required_dialects == \
            MLIR_BACKEND_REQUIRED_DIALECTS[target]


def test_get_backend_emit_factory_yields_real_emitter():
    """The factory must build an instance satisfying the
    `BackendEmitter` Protocol (runtime_checkable)."""
    for target in (MLIRBackendTarget.PTX, MLIRBackendTarget.ROCM_HIP,
                   MLIRBackendTarget.METAL_MSL,
                   MLIRBackendTarget.WEBGPU_WGSL):
        backend = get_backend(target)
        emitter = backend.emit_factory()
        assert isinstance(emitter, BackendEmitter), (target, type(emitter))


def test_get_backend_lowering_status_callable_per_target():
    """Each backend's `lowering_status` accepts a `TileOpKind` and
    returns a closed-set status string."""
    valid = {"supported", "stub", "deferred", "skipped"}
    for target in (MLIRBackendTarget.PTX, MLIRBackendTarget.ROCM_HIP,
                   MLIRBackendTarget.METAL_MSL,
                   MLIRBackendTarget.WEBGPU_WGSL):
        backend = get_backend(target)
        # SCALAR_ADD is supported across the four GPU targets.
        status = backend.lowering_status(tile_ir.TileOpKind.SCALAR_ADD)
        assert status in valid, (target, status)


def test_get_backend_rejects_llvm_ir():
    """LLVM_IR is intentionally absent — its home-grown emitter
    consumes tir.Module, not TileModule."""
    with pytest.raises(ValueError, match="LLVM_IR is not in"):
        get_backend(MLIRBackendTarget.LLVM_IR)


def test_get_backend_rejects_non_target():
    with pytest.raises(ValueError, match="must be MLIRBackendTarget"):
        get_backend("ptx")  # type: ignore[arg-type]


def test_backend_emit_module_convenience_method():
    """`Backend.emit_module(mod)` builds a fresh emitter and emits
    one module — the convenience method for one-shot callers."""
    # Use a richer module that satisfies the GPU emitter's
    # @kernel-required precondition.
    from helixc.ir import tir
    fn = tile_ir.TileFn(
        "kernel_fn",
        [],
        tir.TIRUnit(),
        [tile_ir.TileBlock(
            0, [], [tile_ir.TileOp(
                tile_ir.TileOpKind.RETURN, operands=[])])],
        attrs={"kernel": True},
    )
    module = tile_ir.TileModule(functions={"kernel_fn": fn})
    backend = get_backend(MLIRBackendTarget.PTX)
    text = backend.emit_module(module)
    assert isinstance(text, str)
    assert text.strip()


def test_backend_emit_module_rejects_bad_factory():
    """If a Backend is constructed with a factory that returns
    non-BackendEmitter, the convenience method raises TypeError
    rather than silently returning whatever-it-was's `emit_module`.

    Audit-fix H1: construct a fresh Backend rather than mutating the
    registered PTX singleton (which would pollute the global
    registry and break test isolation under reorder)."""
    bad = Backend(
        target=MLIRBackendTarget.PTX,
        emit_factory=lambda: "not an emitter",
        lowering_status=lambda k: "supported",
        required_dialects=("func",),
    )
    with pytest.raises(TypeError, match="non-BackendEmitter"):
        bad.emit_module(tile_ir.TileModule())
