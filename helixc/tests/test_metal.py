"""Tests for helixc.backend.metal — Stage 125 (v2.0 Phase C) substrate.

Apple Metal Shading Language (MSL) text-emit substrate covering
tile-IR → MSL op-mapping coverage + a kernel-emit smoke test.
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from helixc.backend.metal import (
    DEFAULT_METAL_VERSION,
    DEFAULT_TARGET_FAMILY,
    SIMD_WIDTH,
    METAL_OP_LOWERING,
    MslEmitter,
    lowering_status,
)
from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir.tile_ir import lower_to_tile, TileOpKind


def test_stage125_module_constants():
    """Stage 125 — module-level constants documented (Metal 3.2 +
    Apple7 family + SIMD width 32 lanes)."""
    assert DEFAULT_METAL_VERSION == "metal3.2"
    assert DEFAULT_TARGET_FAMILY == "apple7"
    assert SIMD_WIDTH == 32


def test_stage125_lowering_coverage_complete():
    """Stage 125 — every TileOpKind has a documented lowering entry."""
    for k in TileOpKind:
        assert k in METAL_OP_LOWERING, (
            f"TileOpKind {k.name} missing from METAL_OP_LOWERING — "
            f"add a lowering or mark status='skipped'"
        )


def test_stage125_lowering_status_categories():
    """Stage 125 — every entry's status is one of the documented
    values."""
    valid = {"supported", "stub", "deferred", "skipped"}
    for kind, entry in METAL_OP_LOWERING.items():
        assert entry["status"] in valid, (
            f"TileOpKind {kind.name}: status {entry['status']!r} not in {valid}"
        )


def test_stage125_tma_marked_skipped():
    """Stage 125 — TMA (NVIDIA-only) has no Apple analog."""
    assert lowering_status(TileOpKind.TMA_LOAD) == "skipped"
    assert lowering_status(TileOpKind.TMA_STORE) == "skipped"


def test_stage125_matmul_status_stub():
    """Stage 125 — TILE_MATMUL is stub'd; Stage 126 wires SIMD path
    (pre-M5) and NA mma intrinsics (M5+)."""
    assert lowering_status(TileOpKind.TILE_MATMUL) == "stub"


def test_stage125_lowering_status_rejects_non_tileopkind():
    """Stage 125 — lowering_status raises TypeError on non-TileOpKind.
    Mirrors rocm.lowering_status + has_adjoint discipline."""
    with pytest.raises(TypeError):
        lowering_status("TILE_MATMUL")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        lowering_status(42)  # type: ignore[arg-type]


def test_stage125_emit_module_header():
    """Stage 125 — module header emits MSL boilerplate (metal_stdlib
    include + metal namespace using)."""
    emitter = MslEmitter()
    emitter.emit_module_header()
    out = emitter.buf.getvalue()
    assert "#include <metal_stdlib>" in out
    assert "using namespace metal" in out
    assert "apple7" in out
    assert "metal3.2" in out


def test_stage125_emit_kernel_stub_smoke():
    """Stage 125 — full emit_module path for a minimal @kernel.

    Substrate produces `kernel void NAME(...)` signature + empty body.
    """
    src = "@kernel fn empty_kernel() {}"
    prog = parse(src)
    tir_mod = lower(prog)
    tile_mod = lower_to_tile(tir_mod)
    emitter = MslEmitter()
    text = emitter.emit_module(tile_mod)
    assert "kernel void empty_kernel" in text
    assert "thread_position_in_threadgroup" in text


def test_stage125_emit_module_requires_kernel():
    """Stage 125 — emitting a module with no @kernel fn raises.
    MSL kernels are the only thing this backend emits."""
    src = "fn host_only() -> i32 { 0 }"
    prog = parse(src)
    tir_mod = lower(prog)
    tile_mod = lower_to_tile(tir_mod)
    emitter = MslEmitter()
    with pytest.raises(RuntimeError, match="kernel"):
        emitter.emit_module(tile_mod)


def test_stage125_lowering_status_returns_str_for_every_kind():
    """Stage 125 — lowering_status always returns a non-empty str
    for any known TileOpKind."""
    for k in TileOpKind:
        status = lowering_status(k)
        assert isinstance(status, str)
        assert len(status) > 0


def test_stage125_kernel_attribute_is_void_return():
    """Stage 125 — MSL kernels return void by spec. The emitter must
    NOT try to put a return type other than void in the kernel
    signature."""
    src = "@kernel fn k() {}"
    prog = parse(src)
    tir_mod = lower(prog)
    tile_mod = lower_to_tile(tir_mod)
    emitter = MslEmitter()
    text = emitter.emit_module(tile_mod)
    # The kernel-declaration line must be `kernel void NAME(`.
    assert "kernel void k(" in text
