"""Tests for helixc.backend.regalloc_classes — v2.4 item 15 (slice 4).

Per-backend register-class models. Slice 4 ships the PTX model.
"""
from __future__ import annotations
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from helixc.backend.regalloc import allocate_by_class
from helixc.backend.regalloc_classes import (
    PTX_REGISTER_POOLS,
    ptx_register_class,
)
from helixc.ir import tir
from helixc.ir.tile_ir import TileBlock, TileFn, TileOp, TileOpKind, TileValue


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
