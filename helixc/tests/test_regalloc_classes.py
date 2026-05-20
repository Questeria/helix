"""Tests for helixc.backend.regalloc_classes — v2.4 item 15 (slice 4).

Per-backend register-class models. Slice 4 ships the PTX model.
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from helixc.backend.regalloc import allocate_by_class
from typing import get_args

from helixc.backend.regalloc_classes import (
    PTX_REGISTER_POOLS,
    PtxRegClass,
    ROCM_REGISTER_POOLS,
    RocmRegClass,
    plan_ptx_registers,
    ptx_register_class,
    rocm_register_class,
)
from helixc.ir import tir
from helixc.ir.tile_ir import TileBlock, TileFn, TileOp, TileOpKind, TileValue


def test_v25_register_class_literals_pin_pool_keys():
    """v2.5 polish (item-15 type-design Finding 5) — the PtxRegClass /
    RocmRegClass Literals are the closed set of register-class keys.
    Each pool dict's keys must equal its Literal's members (the
    module-load drift checks enforce this; this test documents +
    re-pins it). Typing the classifier return as the Literal also
    makes a typo'd `return` a static mypy error."""
    assert set(get_args(PtxRegClass)) == set(PTX_REGISTER_POOLS)
    assert set(get_args(RocmRegClass)) == set(ROCM_REGISTER_POOLS)
    # The classifier outputs are members of their backend's Literal.
    assert ptx_register_class(
        TileValue(id=0, ty=tir.TIRScalar("i32"))) in get_args(PtxRegClass)
    assert rocm_register_class(
        TileValue(id=0, ty=tir.TIRScalar("bool"))) in get_args(RocmRegClass)


def _val(vid: int, dtype: str) -> TileValue:
    """A tile-IR SSA value of the given scalar dtype."""
    return TileValue(id=vid, ty=tir.TIRScalar(dtype))


def test_v24_ptx_pools_five_classes_256_deep():
    """v2.4 item 15 slice 4 — PTX declares 5 register files, each 256
    deep (PtxEmitter._REG_POOL_CAP)."""
    assert set(PTX_REGISTER_POOLS) == {"%p", "%r", "%rd", "%f", "%h"}
    assert all(n == 256 for n in PTX_REGISTER_POOLS.values())


def test_v24_ptx_classify_bool_to_pred():
    """v2.4 item 15 slice 4 — bool -> %p (predicate file)."""
    assert ptx_register_class(_val(0, "bool")) == "%p"


def test_v24_ptx_classify_32bit_int_to_r():
    """v2.4 item 15 slice 4 — i32/u32 -> %r (32-bit integer file)."""
    assert ptx_register_class(_val(0, "i32")) == "%r"
    assert ptx_register_class(_val(1, "u32")) == "%r"


def test_v24_ptx_classify_64bit_to_rd():
    """v2.4 item 15 slice 4 — 64-bit ints + pointer-width aliases
    -> %rd (64-bit file)."""
    for dt in ("i64", "u64", "isize", "usize"):
        assert ptx_register_class(_val(0, dt)) == "%rd"


def test_v24_ptx_classify_f32_to_f():
    """v2.4 item 15 slice 4 — f32 -> %f (32-bit float file)."""
    assert ptx_register_class(_val(0, "f32")) == "%f"


def test_v24_ptx_classify_narrow_and_half_to_h():
    """v2.4 item 15 slice 4 — 8/16-bit ints + f16/bf16 all -> %h
    (16-bit file). PTX has no file narrower than 16 bits, so 8-bit
    dtypes are register-allocated in %h (standard PTX practice)."""
    for dt in ("i8", "u8", "char", "i16", "u16", "f16", "bf16"):
        assert ptx_register_class(_val(0, dt)) == "%h", dt


def test_v24_ptx_classify_f64_raises_not_implemented():
    """v2.4 item 15 slice 4 — f64 has no PTX register file; classify
    raises NotImplementedError (honest gap, not a silent mis-file)."""
    with pytest.raises(NotImplementedError, match="f64 has no PTX"):
        ptx_register_class(_val(0, "f64"))


def test_v24_ptx_classify_rejects_non_scalar():
    """v2.4 item 15 slice 4 — a tile/tensor-typed value is memory-
    resident, not single-register; classify raises ValueError."""
    tile_ty = tir.TIRTileTy(
        dtype=tir.TIRScalar("f32"),
        shape=(tir.DimConst(16),),
        memspace="reg",
    )
    v = TileValue(id=0, ty=tile_ty)
    with pytest.raises(ValueError, match="scalar values only"):
        ptx_register_class(v)


def test_v24_ptx_classify_unknown_dtype_raises():
    """v2.4 item 15 slice 4 — an unrecognised scalar dtype raises
    RuntimeError (parity with PtxEmitter._ptx_type_str's KeyError
    -> raise discipline) — never a silent mis-file."""
    with pytest.raises(RuntimeError, match="unrecognised TIRScalar"):
        ptx_register_class(_val(0, "ternary"))


def test_v24_ptx_register_class_drives_allocate_by_class():
    """v2.4 item 15 slice 4 — end-to-end: ptx_register_class +
    PTX_REGISTER_POOLS passed to allocate_by_class. An i32 value and
    an f32 value, both live the whole kernel, land in DIFFERENT PTX
    register files (%r vs %f) and so do not contend."""
    vi = _val(0, "i32")   # -> %r
    vf = _val(1, "f32")   # -> %f
    blk = TileBlock(id=0, ops=[
        TileOp(kind=TileOpKind.SCALAR_CONST_INT, results=[vi]),
        TileOp(kind=TileOpKind.SCALAR_CONST_FLOAT, results=[vf]),
        TileOp(kind=TileOpKind.CALL, operands=[vi, vf]),  # keeps both live
    ])
    fn = TileFn(name="k", params=[], return_ty=tir.TIRUnit(),
                blocks=[blk], attrs={"kernel": True})
    r = allocate_by_class(fn, ptx_register_class, PTX_REGISTER_POOLS)
    assert r.spilled == set()
    assert r.assignment[0][0] == "%r"   # i32 -> %r file
    assert r.assignment[1][0] == "%f"   # f32 -> %f file
    assert set(r.per_class) == {"%r", "%f"}


# ============================================================================
# plan_ptx_registers — v2.5 item 1 (emitter-wiring prep)
# ============================================================================
def test_v25_plan_ptx_registers_runs_allocate_by_class_with_skip():
    """v2.5 item 1 (emitter-wiring prep) — plan_ptx_registers composes
    allocate_by_class with the PTX classifier + pool table + the
    scalar-skip predicate, so it runs over a REAL kernel that mixes
    register-allocated scalars with a memory-resident tile param. The
    scalars get RegAssignments in their files; the tile param lands in
    `skipped` (ptx_register_class is never handed it); no spills."""
    vi = _val(0, "i32")   # scalar -> %r
    vf = _val(1, "f32")   # scalar -> %f
    # A memory-resident tile param — not single-register; must be
    # skipped, never passed to ptx_register_class.
    tile_ty = tir.TIRTileTy(
        dtype=tir.TIRScalar("f32"),
        shape=(tir.DimConst(16),),
        memspace="reg",
    )
    vt = TileValue(id=2, ty=tile_ty)
    blk = TileBlock(id=0, ops=[
        TileOp(kind=TileOpKind.SCALAR_CONST_INT, results=[vi]),
        TileOp(kind=TileOpKind.SCALAR_CONST_FLOAT, results=[vf]),
        TileOp(kind=TileOpKind.CALL, operands=[vi, vf, vt]),
    ])
    fn = TileFn(name="k", params=[vt], return_ty=tir.TIRUnit(),
                blocks=[blk], attrs={"kernel": True})
    r = plan_ptx_registers(fn)
    assert r.spill_count == 0
    assert r.spilled == set()
    assert r.skipped == {2}            # the tile param, memory-resident
    assert r.assignment[0].reg_class == "%r"   # i32 scalar
    assert r.assignment[1].reg_class == "%f"   # f32 scalar
    assert 2 not in r.assignment       # skipped — not register-allocated


def test_v25_plan_ptx_registers_raises_loudly_on_spill(monkeypatch):
    """v2.5 item 1 — plan_ptx_registers enforces the IR LOW-2 no-spill
    contract: the emitter-wiring slice trusts every scalar vreg has a
    RegAssignment, so a spill (a vreg in neither `assignment` nor
    `skipped`) must fail loudly, never be silently trusted. Forced
    here by shrinking the %r pool to a single register via
    monkeypatch, then keeping two i32 values simultaneously live."""
    import helixc.backend.regalloc_classes as rc
    # Shrink %r to one register; keep the other 4 files at 256 so
    # allocate_by_class's classify-key-in-pools check still passes.
    monkeypatch.setattr(rc, "PTX_REGISTER_POOLS", {
        "%p": 256, "%r": 1, "%rd": 256, "%f": 256, "%h": 256,
    })
    a = _val(0, "i32")
    b = _val(1, "i32")
    blk = TileBlock(id=0, ops=[
        TileOp(kind=TileOpKind.SCALAR_CONST_INT, results=[a]),
        TileOp(kind=TileOpKind.SCALAR_CONST_INT, results=[b]),
        TileOp(kind=TileOpKind.CALL, operands=[a, b]),   # both live at once
    ])
    fn = TileFn(name="k", params=[], return_ty=tir.TIRUnit(),
                blocks=[blk], attrs={"kernel": True})
    with pytest.raises(RuntimeError, match="spilled"):
        plan_ptx_registers(fn)


def test_v25_plan_ptx_registers_propagates_f64_not_implemented():
    """v2.5 item 1 — plan_ptx_registers' documented raise contract,
    pinned end-to-end: an f64 scalar value has no PTX register file,
    so `ptx_register_class` raises NotImplementedError. That must
    propagate cleanly through `allocate_by_class` (which calls
    classify() bare — nothing swallows it) out of plan_ptx_registers.
    The existing classifier tests pin the raise on ptx_register_class
    alone; this pins that the new public function honours it too —
    a guard against a future defensive try/except quietly swallowing
    the gap."""
    vf64 = _val(0, "f64")
    blk = TileBlock(id=0, ops=[
        TileOp(kind=TileOpKind.SCALAR_CONST_FLOAT, results=[vf64]),
    ])
    fn = TileFn(name="k", params=[], return_ty=tir.TIRUnit(),
                blocks=[blk], attrs={"kernel": True})
    with pytest.raises(NotImplementedError, match="f64 has no PTX"):
        plan_ptx_registers(fn)


def test_v25_plan_ptx_registers_propagates_unknown_dtype():
    """v2.5 item 1 — plan_ptx_registers' documented raise contract:
    an unrecognised scalar dtype makes `ptx_register_class` raise
    RuntimeError; it must propagate out of plan_ptx_registers rather
    than be mis-filed. Parity with ptx_register_class's own
    unknown-dtype discipline, pinned end-to-end through the new
    function."""
    vbad = _val(0, "ternary")   # parser/typecheck-only quantized dtype
    blk = TileBlock(id=0, ops=[
        TileOp(kind=TileOpKind.SCALAR_CONST_INT, results=[vbad]),
    ])
    fn = TileFn(name="k", params=[], return_ty=tir.TIRUnit(),
                blocks=[blk], attrs={"kernel": True})
    with pytest.raises(RuntimeError, match="unrecognised TIRScalar"):
        plan_ptx_registers(fn)


# ============================================================================
# ROCm / AMDGCN register-class model (v2.4 item 15 slice 5)
# ============================================================================
def test_v24_rocm_pools_two_files():
    """v2.4 item 15 slice 5 — AMDGCN has two register files: vgpr
    (256 deep on gfx942) and sgpr (~104 usable)."""
    assert set(ROCM_REGISTER_POOLS) == {"vgpr", "sgpr"}
    assert ROCM_REGISTER_POOLS["vgpr"] == 256
    assert ROCM_REGISTER_POOLS["sgpr"] == 104


def test_v24_rocm_classify_bool_to_sgpr():
    """v2.4 item 15 slice 5 — a boolean is a wavefront condition; it
    lives in the scalar file (sgpr)."""
    assert rocm_register_class(_val(0, "bool")) == "sgpr"


def test_v24_rocm_classify_scalars_to_vgpr():
    """v2.4 item 15 slice 5 — every non-bool scalar dtype is a
    per-thread value in the vector file (vgpr), including 64-bit
    dtypes (vgpr register pairs — pairing is a later slice) and
    f64 (which PTX rejects but AMDGCN handles in a vgpr pair)."""
    for dt in ("i8", "u8", "char", "i16", "u16", "f16", "bf16",
               "i32", "u32", "f32", "i64", "u64", "isize", "usize",
               "f64"):
        assert rocm_register_class(_val(0, dt)) == "vgpr", dt


def test_v24_rocm_classify_rejects_non_scalar():
    """v2.4 item 15 slice 5 — tile/tensor values are memory-resident
    (LDS/HBM), not single-register; classify raises ValueError."""
    tile_ty = tir.TIRTileTy(
        dtype=tir.TIRScalar("f32"),
        shape=(tir.DimConst(16),),
        memspace="reg",
    )
    with pytest.raises(ValueError, match="scalar values only"):
        rocm_register_class(TileValue(id=0, ty=tile_ty))


def test_v24_rocm_classify_unknown_dtype_raises():
    """v2.4 item 15 slice 5 — an unrecognised scalar dtype raises
    RuntimeError, never a silent mis-file."""
    with pytest.raises(RuntimeError, match="unrecognised TIRScalar"):
        rocm_register_class(_val(0, "ternary"))


def test_v24_rocm_register_class_drives_allocate_by_class():
    """v2.4 item 15 slice 5 — end-to-end: rocm_register_class +
    ROCM_REGISTER_POOLS drive allocate_by_class. A bool predicate
    (sgpr) and an f32 value (vgpr), both live, land in different
    AMDGCN files and do not contend."""
    vb = _val(0, "bool")  # -> sgpr
    vf = _val(1, "f32")   # -> vgpr
    blk = TileBlock(id=0, ops=[
        TileOp(kind=TileOpKind.SCALAR_CONST_INT, results=[vb]),
        TileOp(kind=TileOpKind.SCALAR_CONST_FLOAT, results=[vf]),
        TileOp(kind=TileOpKind.CALL, operands=[vb, vf]),
    ])
    fn = TileFn(name="k", params=[], return_ty=tir.TIRUnit(),
                blocks=[blk], attrs={"kernel": True})
    r = allocate_by_class(fn, rocm_register_class, ROCM_REGISTER_POOLS)
    assert r.spilled == set()
    assert r.assignment[0][0] == "sgpr"
    assert r.assignment[1][0] == "vgpr"
    assert set(r.per_class) == {"sgpr", "vgpr"}
