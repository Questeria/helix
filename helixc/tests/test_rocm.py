"""Tests for helixc.backend.rocm — Stage 123 (v2.0 Phase C) substrate.

ROCm / HIP text-emit substrate covering CUDA → AMDGPU op-mapping
coverage + a kernel-emit smoke test.
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from helixc.backend.rocm import (
    DEFAULT_TARGET,
    ROCM_OBJECT_FORMAT,
    DEFAULT_WAVE_SIZE,
    ROCM_OP_LOWERING,
    HipEmitter,
    lowering_status,
)
from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir.tile_ir import lower_to_tile, TileOpKind


def test_stage123_module_constants():
    """Stage 123 — MI300 baseline target + wave64 wave size + amdgcn
    object format documented as module constants."""
    assert DEFAULT_TARGET == "gfx942"
    assert ROCM_OBJECT_FORMAT == "amdgcn"
    assert DEFAULT_WAVE_SIZE == 64


def test_stage123_lowering_coverage_complete():
    """Stage 123 — every TileOpKind has a documented lowering entry.

    This is the drift detector that fires at module-load. If a new
    TileOpKind is added to tile_ir.py, this test reminds the dev to
    update the ROCm port table OR mark it skipped with rationale.
    """
    for k in TileOpKind:
        assert k in ROCM_OP_LOWERING, (
            f"TileOpKind {k.name} missing from ROCM_OP_LOWERING — "
            f"add a lowering or mark status='skipped'"
        )


def test_stage123_lowering_status_categories():
    """Stage 123 — every entry's status is one of the documented
    values. Catches typos like 'STUB' vs 'stub'."""
    valid = {"supported", "stub", "deferred", "skipped"}
    for kind, entry in ROCM_OP_LOWERING.items():
        assert entry["status"] in valid, (
            f"TileOpKind {kind.name}: status {entry['status']!r} not in {valid}"
        )


def test_stage123_tma_marked_skipped():
    """Stage 123 — TMA (Hopper-only memory transfer) has no AMD analog;
    must be documented as skipped, not silently routed elsewhere."""
    assert lowering_status(TileOpKind.TMA_LOAD) == "skipped"
    assert lowering_status(TileOpKind.TMA_STORE) == "skipped"


def test_stage123_matmul_status_stub():
    """Stage 123 — TILE_MATMUL is stub'd; Stage 124 will wire the
    actual MFMA instruction emit."""
    assert lowering_status(TileOpKind.TILE_MATMUL) == "stub"


def test_stage123_lowering_status_rejects_non_tileopkind():
    """Stage 123 — lowering_status raises TypeError on non-TileOpKind.
    Mirrors the has_adjoint / adjoint_outputs discipline from
    Stage 117-119 audit-fix."""
    with pytest.raises(TypeError):
        lowering_status("TILE_MATMUL")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        lowering_status(None)  # type: ignore[arg-type]


def test_stage123_emit_module_header():
    """Stage 123 — module header emit produces the .amdgcn_target
    directive with the correct triple format."""
    emitter = HipEmitter()
    emitter.emit_module_header()
    out = emitter.buf.getvalue()
    assert ".amdgcn_target" in out
    assert "amdgcn-amd-amdhsa--gfx942" in out


def test_stage123_emit_kernel_stub_smoke():
    """Stage 123 — full emit_module path for a minimal @kernel.

    Per Phase-0 Helix tile-IR lowering: bare `@kernel fn k() {}` produces
    a TileFn with the kernel attr. Substrate emit produces the function
    header + s_endpgm terminator.
    """
    src = "@kernel fn empty_kernel() {}"
    prog = parse(src)
    tir_mod = lower(prog)
    tile_mod = lower_to_tile(tir_mod)
    emitter = HipEmitter()
    text = emitter.emit_module(tile_mod)
    assert ".amdgcn_target" in text
    assert ".globl empty_kernel" in text
    assert "s_endpgm" in text


def test_stage123_emit_module_requires_kernel():
    """Stage 123 — emitting a module with no @kernel fn raises (parity
    with PtxEmitter; non-kernel modules don't make sense for AMDGPU)."""
    src = "fn host_only() -> i32 { 0 }"
    prog = parse(src)
    tir_mod = lower(prog)
    tile_mod = lower_to_tile(tir_mod)
    emitter = HipEmitter()
    with pytest.raises(RuntimeError, match="kernel"):
        emitter.emit_module(tile_mod)


def test_stage123_emit_module_skips_extern_kernels():
    """Stage 123 — extern kernels (is_extern attr) are NOT emitted; they
    are import declarations from the host side. Mirrors PtxEmitter."""
    src = """
    @kernel fn real_kernel() {}
    """
    prog = parse(src)
    tir_mod = lower(prog)
    tile_mod = lower_to_tile(tir_mod)
    # Add a synthetic extern kernel to ensure it's skipped.
    fn = tile_mod.functions["real_kernel"]
    # Synthesize a second kernel with is_extern (parity with the original
    # ptx.py behavior).
    fn2 = type(fn)(
        name="extern_kernel",
        params=[],
        return_ty=fn.return_ty,
        blocks=fn.blocks,
        attrs={"kernel": True, "is_extern": True},
    )
    tile_mod.functions["extern_kernel"] = fn2
    emitter = HipEmitter()
    text = emitter.emit_module(tile_mod)
    assert "real_kernel" in text
    assert "extern_kernel" not in text


def test_stage123_lowering_status_returns_str():
    """Stage 123 — lowering_status always returns a str (never None or
    raises for known kinds)."""
    for k in TileOpKind:
        status = lowering_status(k)
        assert isinstance(status, str)
        assert len(status) > 0
