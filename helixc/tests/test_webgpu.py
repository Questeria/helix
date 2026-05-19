"""Tests for helixc.backend.webgpu — Stage 127 (v2.0 Phase C) substrate.

WebGPU/WGSL text-emit substrate covering tile-IR → WGSL op-mapping
coverage + a kernel-emit smoke test.
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from helixc.backend.webgpu import (
    DEFAULT_WGSL_VERSION,
    DEFAULT_WORKGROUP_SIZE,
    WEBGPU_OP_LOWERING,
    WgslEmitter,
    lowering_status,
)
from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir.tile_ir import lower_to_tile, TileOpKind


def test_stage127_module_constants():
    """Stage 127 — WGSL spec + default workgroup_size 64."""
    assert DEFAULT_WGSL_VERSION == "wgsl-2024"
    assert DEFAULT_WORKGROUP_SIZE == 64


def test_stage127_lowering_coverage_complete():
    """Stage 127 — drift detector regression: every TileOpKind has an entry."""
    for k in TileOpKind:
        assert k in WEBGPU_OP_LOWERING, (
            f"TileOpKind {k.name} missing from WEBGPU_OP_LOWERING — "
            f"add a lowering or mark status='skipped'"
        )


def test_stage127_lowering_status_categories():
    """Stage 127 — every entry status is in the documented set."""
    valid = {"supported", "stub", "deferred", "skipped"}
    for kind, entry in WEBGPU_OP_LOWERING.items():
        assert entry["status"] in valid, (
            f"TileOpKind {kind.name}: status {entry['status']!r} not in {valid}"
        )


def test_stage127_tma_marked_skipped():
    """Stage 127 — TMA / TMEM have no WebGPU analog (matches Report 5)."""
    assert lowering_status(TileOpKind.TMA_LOAD) == "skipped"
    assert lowering_status(TileOpKind.TMA_STORE) == "skipped"


def test_stage127_matmul_status_supported():
    """Stage 128 R6 audit-fix — TILE_MATMUL is supported on WebGPU:
    Stage 128 wired the hand-rolled tile-loop emit (no Tensor Cores;
    ~1 TFLOPS ceiling per v2.0 research Report 5). The prior test
    name was `_status_stub` while the assertion checked `==
    "supported"` — a docstring-vs-assertion lie the v2.1 TEST
    5-gate caught."""
    assert lowering_status(TileOpKind.TILE_MATMUL) == "supported"


def test_stage127_lowering_status_rejects_non_tileopkind():
    """Stage 127 — lowering_status raises TypeError on bad input."""
    with pytest.raises(TypeError):
        lowering_status("TILE_MATMUL")  # type: ignore[arg-type]


def test_stage127_emit_module_header():
    """Stage 127 — header documents WGSL spec + workgroup_size."""
    emitter = WgslEmitter()
    emitter.emit_module_header()
    out = emitter.buf.getvalue()
    assert "wgsl-2024" in out


def test_stage127_emit_kernel_stub_smoke():
    """Stage 127 — full emit_module path for a minimal @kernel.

    WGSL compute kernels use @compute @workgroup_size attribute +
    `fn KERNEL()` block.
    """
    src = "@kernel fn empty_kernel() {}"
    prog = parse(src)
    tir_mod = lower(prog)
    tile_mod = lower_to_tile(tir_mod)
    emitter = WgslEmitter()
    text = emitter.emit_module(tile_mod)
    assert "@compute" in text
    assert "@workgroup_size(64)" in text
    assert "fn empty_kernel(" in text
    assert "local_invocation_id" in text


def test_stage127_emit_module_requires_kernel():
    """Stage 127 — empty-module raises (parity with rocm + metal + ptx)."""
    src = "fn host_only() -> i32 { 0 }"
    prog = parse(src)
    tir_mod = lower(prog)
    tile_mod = lower_to_tile(tir_mod)
    emitter = WgslEmitter()
    with pytest.raises(RuntimeError, match="kernel"):
        emitter.emit_module(tile_mod)


def test_stage127_custom_workgroup_size():
    """Stage 127 — emitter honors a custom workgroup_size."""
    emitter = WgslEmitter(workgroup_size=128)
    src = "@kernel fn k() {}"
    prog = parse(src)
    tir_mod = lower(prog)
    tile_mod = lower_to_tile(tir_mod)
    text = emitter.emit_module(tile_mod)
    assert "@workgroup_size(128)" in text


# ============================================================================
# Stage 128 (v2.1 Phase C WebGPU tile-loop matmul) — per-op WGSL emit
# ============================================================================
def test_stage128_barrier_wait_emits_workgroup_barrier():
    """Stage 128 — BARRIER_WAIT lowers to workgroupBarrier() (WGSL's
    parity with __syncthreads)."""
    from helixc.ir.tile_ir import TileOp, TileBlock, TileFn, TileModule
    fn = TileFn(
        name="k", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[
            TileOp(kind=TileOpKind.BARRIER_WAIT),
        ])],
        attrs={"kernel": True},
    )
    tile_mod = TileModule()
    tile_mod.functions["k"] = fn
    text = WgslEmitter().emit_module(tile_mod)
    assert "workgroupBarrier();" in text


def test_stage128_tile_matmul_hand_rolled_loop():
    """Stage 128 — TILE_MATMUL on WebGPU emits a hand-rolled tile loop
    (no Tensor Cores). Each invocation accumulates one row*col entry
    via a for-loop over the inner dimension (16-element chunk)."""
    from helixc.ir.tile_ir import TileOp, TileBlock, TileFn, TileModule
    fn = TileFn(
        name="k", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[
            TileOp(kind=TileOpKind.TILE_MATMUL),
        ])],
        attrs={"kernel": True},
    )
    tile_mod = TileModule()
    tile_mod.functions["k"] = fn
    text = WgslEmitter().emit_module(tile_mod)
    # The matmul body must contain the inner loop variable + accumulator.
    assert "var acc: f32" in text
    assert "for (var k: u32" in text
    assert "a_tile[k] * b_tile[k]" in text
    assert "no Tensor Cores" in text


def test_stage128_global_memory_ops_emit():
    """Stage 128 — TILE_LOAD_GLOBAL / TILE_STORE_GLOBAL emit storage-
    buffer reads/writes (WGSL `var<storage, ...>` storage class)."""
    from helixc.ir.tile_ir import TileOp, TileBlock, TileFn, TileModule
    fn = TileFn(
        name="k", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[
            TileOp(kind=TileOpKind.TILE_LOAD_GLOBAL),
            TileOp(kind=TileOpKind.TILE_STORE_GLOBAL),
        ])],
        attrs={"kernel": True},
    )
    tile_mod = TileModule()
    tile_mod.functions["k"] = fn
    text = WgslEmitter().emit_module(tile_mod)
    assert "buf_in[local_id.x]" in text
    assert "buf_out[local_id.x]" in text


def test_stage128_workgroup_memory_ops_emit():
    """Stage 128 — TILE_LOAD_SHARED / TILE_STORE_SHARED emit
    workgroup-memory references (WGSL `var<workgroup>` storage class)."""
    from helixc.ir.tile_ir import TileOp, TileBlock, TileFn, TileModule
    fn = TileFn(
        name="k", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[
            TileOp(kind=TileOpKind.TILE_LOAD_SHARED),
            TileOp(kind=TileOpKind.TILE_STORE_SHARED),
        ])],
        attrs={"kernel": True},
    )
    tile_mod = TileModule()
    tile_mod.functions["k"] = fn
    text = WgslEmitter().emit_module(tile_mod)
    assert "shared_mem[local_id.x]" in text


def test_stage128_stub_status_emits_helix_stub_directive():
    """Stage 128 R5 audit-fix — ops with status='stub'/'deferred' in
    WEBGPU_OP_LOWERING emit the `@@HELIX-STUB...` token at the TOP
    of `_emit_op`, which is parse-breaking in WGSL (naga rejects
    `@@`). This is the substrate's loud-stub guard; it replaces the
    silent `// (stub)` comment fallthrough that R5 found could ship
    empty kernels."""
    from helixc.ir.tile_ir import TileOp, TileBlock, TileFn, TileModule
    fn = TileFn(
        name="k", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[
            TileOp(kind=TileOpKind.TILE_REDUCE),
        ])],
        attrs={"kernel": True},
    )
    tile_mod = TileModule()
    tile_mod.functions["k"] = fn
    text = WgslEmitter().emit_module(tile_mod)
    assert "HELIX-STUB" in text and "TILE_REDUCE" in text
    assert "@@" in text  # parse-breaking marker


def test_stage128_r5_phantom_supported_raises_assertion():
    """Stage 128 R5 audit-fix — exhaustiveness guard at the bottom of
    `_emit_op` fires AssertionError if a TileOpKind has status
    'supported' in WEBGPU_OP_LOWERING but no concrete branch.
    Parity with rocm.py / metal.py R5 guards."""
    from helixc.ir.tile_ir import TileOp, TileBlock, TileFn, TileModule
    from helixc.backend import webgpu as wg_mod
    fn = TileFn(
        name="k", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[
            TileOp(kind=TileOpKind.TILE_REDUCE),  # currently "stub"
        ])],
        attrs={"kernel": True},
    )
    tile_mod = TileModule()
    tile_mod.functions["k"] = fn
    original = wg_mod.WEBGPU_OP_LOWERING[TileOpKind.TILE_REDUCE]["status"]
    wg_mod.WEBGPU_OP_LOWERING[TileOpKind.TILE_REDUCE]["status"] = "supported"
    try:
        with pytest.raises(AssertionError, match="TILE_REDUCE"):
            wg_mod.WgslEmitter().emit_module(tile_mod)
    finally:
        wg_mod.WEBGPU_OP_LOWERING[TileOpKind.TILE_REDUCE]["status"] = original
