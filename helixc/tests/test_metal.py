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


def test_stage125_matmul_status_supported():
    """Stage 126 R6 audit-fix — TILE_MATMUL is supported on Metal:
    Stage 126 wired the simdgroup_multiply_accumulate path (NA hw is
    the accelerator behind the same simdgroup intrinsic, NOT a
    separate mma_* MSL surface — the R5 audit invalidated that
    claim). The prior test name was `_status_stub` while the
    assertion checked `== "supported"` — a docstring-vs-assertion
    lie the v2.1 TEST 5-gate caught."""
    assert lowering_status(TileOpKind.TILE_MATMUL) == "supported"


def test_stage125_lowering_status_rejects_non_tileopkind():
    """Stage 125 — lowering_status raises TypeError on non-TileOpKind.
    Mirrors rocm.lowering_status + has_adjoint discipline.

    v2.3 TEST MED audit-fix: `match=` clause added so the test
    anchors to the intended error message rather than catching ANY
    TypeError. Without it, a bare `pytest.raises(TypeError)` is
    insensitive to which TypeError surfaced — masking message-
    regression bugs.
    """
    for bad in ("TILE_MATMUL", 42, None, object(), 3.14, ["TILE_MATMUL"]):
        with pytest.raises(
            TypeError, match=r"lowering_status expects TileOpKind"
        ):
            lowering_status(bad)  # type: ignore[arg-type]


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


# ============================================================================
# Stage 126 (v2.1 Phase C Metal NA matmul) — per-op MSL emit
# ============================================================================
def test_stage126_barrier_wait_emits_threadgroup_barrier():
    """Stage 126 — BARRIER_WAIT lowers to threadgroup_barrier with
    threadgroup memory-flag (parity with __syncthreads)."""
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
    text = MslEmitter().emit_module(tile_mod)
    assert "threadgroup_barrier(mem_flags::mem_threadgroup)" in text


def test_stage126_tile_matmul_pre_m5_simdgroup():
    """Stage 126 — TILE_MATMUL on pre-M5 family (apple7) emits the
    simdgroup_multiply_accumulate path with simdgroup_matrix args."""
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
    text = MslEmitter(target_family="apple7").emit_module(tile_mod)
    assert "simdgroup_multiply_accumulate" in text
    assert "simdgroup_matrix<float, 8, 8>" in text
    assert "pre-M5" in text


def test_stage126_tile_matmul_m5_plus_na():
    """Stage 126 R5 audit-fix — TILE_MATMUL on apple10+ (M5+ Neural
    Accelerators) emits the same `simdgroup_*` MSL intrinsics as
    pre-M5; the M5+ NA is a HARDWARE accelerator behind those same
    intrinsics, not a separate MSL surface. The previous test asserted
    a fictional `mma_f32_16x16x16_f16` intrinsic (PTX/CUDA syntax that
    leaked into the Metal backend) — the audit caught this as
    NEEDS-CHANGE."""
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
    # apple10 is the first M5+ family. apple9 is M3/M4 (no NA hw).
    text = MslEmitter(target_family="apple10").emit_module(tile_mod)
    assert "simdgroup_multiply_accumulate" in text
    assert "simdgroup_matrix<float, 8, 8>" in text
    assert "M5+ NA" in text
    # Regression-pin: the fictional intrinsic must NOT appear anywhere.
    assert "mma_f32_16x16x16_f16" not in text


def test_stage126_apple9_is_pre_m5_not_na():
    """Stage 126 R5 audit-fix — `apple9` is the family for M3 and M4
    (which lack Neural Accelerator hw); the previous gate incorrectly
    routed apple9 into the M5+ NA path. The R5 fix corrects the
    family-number list to `apple10+`. apple9 must take the pre-M5
    SIMD path."""
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
    text = MslEmitter(target_family="apple9").emit_module(tile_mod)
    assert "pre-M5" in text
    assert "M5+ NA" not in text


def test_stage126_global_memory_ops_emit():
    """Stage 126 — TILE_LOAD_GLOBAL / TILE_STORE_GLOBAL emit device-
    pointer reads/writes (MSL `device float*` storage class)."""
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
    text = MslEmitter().emit_module(tile_mod)
    assert "buf_in[tid]" in text
    assert "buf_out[tid]" in text


def test_stage126_threadgroup_memory_ops_emit():
    """Stage 126 — TILE_LOAD_SHARED / TILE_STORE_SHARED emit
    threadgroup-memory references (MSL `threadgroup` storage class)."""
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
    text = MslEmitter().emit_module(tile_mod)
    assert "shared_mem[tid]" in text


def test_stage126_stub_status_emits_helix_stub_directive():
    """Stage 126 R5 audit-fix — ops with status='stub'/'deferred' in
    METAL_OP_LOWERING emit the `#error "HELIX-STUB..."` directive at
    the TOP of `_emit_op`, which aborts xcrun-metal compilation. This
    is the substrate's loud-stub guard; it replaces the silent
    `// (stub)` comment fallthrough that R5 found could ship empty
    kernels for supported-but-unimplemented ops."""
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
    text = MslEmitter().emit_module(tile_mod)
    assert "HELIX-STUB" in text and "TILE_REDUCE" in text
    assert "#error" in text


def test_stage126_r5_phantom_supported_raises_assertion():
    """Stage 126 R5 audit-fix — exhaustiveness guard at the bottom of
    `_emit_op` fires AssertionError if a TileOpKind has status
    'supported' in METAL_OP_LOWERING but no concrete branch in the
    if/elif ladder. Parity with rocm.py's R1 exhaustiveness guard.
    Without this, a future bulk-promotion of a status flag without
    a matching emit-branch addition would silently ship an empty
    kernel — exactly the failure mode the R5 audit found."""
    from helixc.ir.tile_ir import TileOp, TileBlock, TileFn, TileModule
    from helixc.backend import metal as metal_mod
    # Synthesize the bug: promote a previously-stub op to "supported"
    # in the module-level table, then run _emit_op on it. The guard
    # must fire.
    fn = TileFn(
        name="k", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[
            TileOp(kind=TileOpKind.TILE_REDUCE),  # currently "stub"
        ])],
        attrs={"kernel": True},
    )
    tile_mod = TileModule()
    tile_mod.functions["k"] = fn
    # Mutate the dict in place to simulate the phantom-supported drift.
    original = metal_mod.METAL_OP_LOWERING[TileOpKind.TILE_REDUCE]["status"]
    metal_mod.METAL_OP_LOWERING[TileOpKind.TILE_REDUCE]["status"] = "supported"
    try:
        with pytest.raises(AssertionError, match="TILE_REDUCE"):
            metal_mod.MslEmitter().emit_module(tile_mod)
    finally:
        metal_mod.METAL_OP_LOWERING[TileOpKind.TILE_REDUCE]["status"] = original


# ============================================================================
# v2.2 polish items 9 + 10 — Metal target_family parsing + validation.
# Item 9: replace hardcoded apple10/11/12 list with numeric comparison
#         so apple13+ are M5-plus without code changes.
# Item 10: reject malformed target_family strings (typos like
#          "appel10", "Apple10", "apple_10") at construction time.
# ============================================================================
def test_v22_parse_apple_family_accepts_valid_names():
    """v2.2 polish item 9 — _parse_apple_family extracts the family
    number from canonical `appleN` strings."""
    from helixc.backend.metal import _parse_apple_family
    assert _parse_apple_family("apple7") == 7
    assert _parse_apple_family("apple9") == 9
    assert _parse_apple_family("apple10") == 10
    assert _parse_apple_family("apple13") == 13       # future family
    assert _parse_apple_family("apple100") == 100     # arbitrary digit count


def test_v22_parse_apple_family_rejects_typos():
    """v2.2 polish item 10 — _parse_apple_family returns None for any
    string that doesn't match `appleN` exactly. The caller uses this
    to hard-fail malformed target_family at construction."""
    from helixc.backend.metal import _parse_apple_family
    for bad in (
        "appel10",      # typo
        "Apple10",      # wrong case
        "apple_10",     # underscore
        "apple-10",     # hyphen
        "apple",        # no number
        "apple10x",     # trailing garbage
        "xapple10",     # leading garbage
        "",             # empty
        " apple10",     # leading whitespace
    ):
        assert _parse_apple_family(bad) is None, (
            f"_parse_apple_family({bad!r}) should be None, "
            f"got {_parse_apple_family(bad)!r}"
        )


def test_v22_parse_apple_family_handles_non_string_input():
    """v2.2 polish — _parse_apple_family returns None on non-string
    input (None, int, etc.) rather than raising. The caller's
    _validate_target_family raises the typed error."""
    from helixc.backend.metal import _parse_apple_family
    for bad in (None, 10, 10.0, ["apple10"], {"apple": 10}):
        assert _parse_apple_family(bad) is None  # type: ignore[arg-type]


def test_v22_msl_emitter_rejects_invalid_target_family():
    """v2.2 polish item 10 — MslEmitter.__init__ validates
    target_family. Typos like "appel10" would have previously routed
    silently to the pre-M5 path (since the typo isn't in the M5+
    list); they now hard-fail at construction time."""
    from helixc.backend import metal as metal_mod
    for bad in ("appel10", "Apple10", "apple_10", "apple"):
        with pytest.raises(ValueError, match="not a recognized Apple-Silicon family"):
            metal_mod.MslEmitter(target_family=bad)


def test_v22_msl_emitter_accepts_future_apple_families():
    """v2.2 polish item 9 — MslEmitter accepts apple13/apple14/... as
    valid families and routes them to the M5+ NA path. Before v2.2 the
    hardcoded list ("apple10", "apple11", "apple12") would have
    silently routed apple13 to the pre-M5 path."""
    from helixc.backend import metal as metal_mod
    # Construction must succeed.
    e13 = metal_mod.MslEmitter(target_family="apple13")
    e20 = metal_mod.MslEmitter(target_family="apple20")
    # The M5+ gate must consider these family >= 10.
    assert metal_mod._parse_apple_family(e13.target_family) >= 10
    assert metal_mod._parse_apple_family(e20.target_family) >= 10


def test_v22_msl_emitter_accepts_pre_m5_families():
    """v2.2 polish — verify the pre-M5 path still works after v2.2
    item 9's numeric-extraction refactor. apple7 (M2) and apple9
    (M3/M4) must construct cleanly and route to the SIMD-group path."""
    from helixc.backend import metal as metal_mod
    for pre_m5 in ("apple7", "apple8", "apple9"):
        e = metal_mod.MslEmitter(target_family=pre_m5)
        assert metal_mod._parse_apple_family(e.target_family) < 10


def test_v23_msl_emitter_skipped_status_emits_helix_skipped_error():
    """v2.3 BE MED audit-fix — status="skipped" ops (TMA_LOAD,
    TMA_STORE on the Metal path) emit `#error "HELIX-SKIPPED: ..."`
    instead of falling through to the exhaustiveness AssertionError
    with a misleading "table/dispatcher drift" message. Parity with
    rocm.py's v2.1 R1 H2 fix.
    """
    from helixc.backend import metal as metal_mod
    from helixc.ir.tile_ir import TileOp
    for skipped_kind in (TileOpKind.TMA_LOAD, TileOpKind.TMA_STORE):
        e = metal_mod.MslEmitter()
        e._emit_op(TileOp(kind=skipped_kind))
        out = e.buf.getvalue()
        assert "#error" in out, (
            f"{skipped_kind.name}: expected #error directive, got {out!r}"
        )
        assert "HELIX-SKIPPED" in out, (
            f"{skipped_kind.name}: expected HELIX-SKIPPED tag, got {out!r}"
        )
        assert "no Apple analog" in out, (
            f"{skipped_kind.name}: expected Apple-analog rationale, "
            f"got {out!r}"
        )
        assert skipped_kind.name in out, (
            f"{skipped_kind.name}: expected kind name in message, "
            f"got {out!r}"
        )
