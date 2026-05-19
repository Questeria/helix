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


def test_stage127_matmul_status_stub():
    """Stage 127 — TILE_MATMUL is stub'd; Stage 128 wires the hand-rolled
    tile-loop (no Tensor Cores; ~1 TFLOPS ceiling per Report 5)."""
    assert lowering_status(TileOpKind.TILE_MATMUL) == "stub"


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
