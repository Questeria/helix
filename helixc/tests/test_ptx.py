"""Tests for helixc.backend.ptx (PTX emission)."""

from __future__ import annotations
import os, sys
import runpy
import subprocess
import tempfile
import pytest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend import parser as parser_mod
from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir import tir, tile_ir as ti
from helixc.ir.tile_ir import lower_to_tile
from helixc.backend.ptx import emit_ptx


def emit(src: str) -> str:
    return emit_ptx(lower_to_tile(lower(parse(src))))


def run_ptx_cli(src: str, *extra_args: str) -> subprocess.CompletedProcess[str]:
    fd, path = tempfile.mkstemp(suffix=".hx", text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(src)
        proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        return subprocess.run(
            [sys.executable, "-m", "helixc.backend.ptx", path, *extra_args],
            cwd=proj_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_stage64_inc2_tile_zeros_emits_f32_register_fills():
    """Stage 64 Inc 2 — TILE_ZEROS for a length-4 f32 tile emits 4
    `mov.f32 %fX, 0f00000000;` register-fills and maps the result
    TileValue to the base %f register."""
    from helixc.backend.ptx import PtxEmitter
    out = ti.TileValue(0, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(4),), "REG"
    ))
    op = ti.TileOp(ti.TileOpKind.TILE_ZEROS, [], [out],
                   attrs={"dtype": "f32", "length": 4})
    em = PtxEmitter()
    em.emit_op(op)
    # Expect 4 lines, each a mov.f32 to zero.
    text = em.buf.getvalue()
    fill_lines = [ln for ln in text.splitlines()
                  if "mov.f32" in ln and "0f00000000" in ln]
    assert len(fill_lines) == 4, text
    # Result mapped to base %f register.
    assert out.id in em.reg_map
    assert em.reg_map[out.id].startswith("%f"), em.reg_map[out.id]


def test_stage64_inc2_tile_zeros_emits_i32_register_fills():
    """Stage 64 Inc 2 — TILE_ZEROS for length-3 i32 tile emits
    3 `mov.b32 %rX, 0;` register-fills."""
    from helixc.backend.ptx import PtxEmitter
    out = ti.TileValue(0, tir.TIRTileTy(
        tir.TIRScalar("i32"), (tir.DimConst(3),), "REG"
    ))
    op = ti.TileOp(ti.TileOpKind.TILE_ZEROS, [], [out],
                   attrs={"dtype": "i32", "length": 3})
    em = PtxEmitter()
    em.emit_op(op)
    text = em.buf.getvalue()
    fill_lines = [ln for ln in text.splitlines()
                  if "mov.b32" in ln and " 0;" in ln]
    assert len(fill_lines) == 3, text
    assert em.reg_map[out.id].startswith("%r"), em.reg_map[out.id]


def test_stage64_inc2_tile_zeros_rejects_invalid_length():
    """Stage 64 Inc 2 — TILE_ZEROS with missing or non-positive
    length attr fails closed."""
    from helixc.backend.ptx import PtxEmitter
    out = ti.TileValue(0, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(0),), "REG"
    ))
    op = ti.TileOp(ti.TileOpKind.TILE_ZEROS, [], [out],
                   attrs={"dtype": "f32", "length": 0})
    em = PtxEmitter()
    import pytest as _pt
    with _pt.raises(RuntimeError, match="positive int 'length'"):
        em.emit_op(op)


def test_stage64_inc2_tile_zeros_rejects_unsupported_dtype():
    """Stage 64 Inc 2 — TILE_ZEROS only supports f32 / i32 in
    Phase-0; bf16/f16 etc. fail closed with a clear message."""
    from helixc.backend.ptx import PtxEmitter
    out = ti.TileValue(0, tir.TIRTileTy(
        tir.TIRScalar("bf16"), (tir.DimConst(2),), "REG"
    ))
    op = ti.TileOp(ti.TileOpKind.TILE_ZEROS, [], [out],
                   attrs={"dtype": "bf16", "length": 2})
    em = PtxEmitter()
    import pytest as _pt
    with _pt.raises(RuntimeError, match="only f32 / i32 supported"):
        em.emit_op(op)


# ============================================================================
# Stage 64 Inc 3 — TILE_ADD / TILE_SUB / TILE_MUL elementwise on
# register-tiles (speculative parallel work; Inc 3 is BACKEND ONLY,
# no frontend trigger path yet).
# ============================================================================
def _build_two_zero_tiles(em, dtype: str, length: int):
    """Helper: build two TILE_ZEROS tiles of the same dtype + length,
    emit them via `em`, and return (lhs_val, rhs_val) ready for an
    elementwise op."""
    lhs = ti.TileValue(0, tir.TIRTileTy(
        tir.TIRScalar(dtype), (tir.DimConst(length),), "REG"
    ))
    rhs = ti.TileValue(1, tir.TIRTileTy(
        tir.TIRScalar(dtype), (tir.DimConst(length),), "REG"
    ))
    em.emit_op(ti.TileOp(ti.TileOpKind.TILE_ZEROS, [], [lhs],
                         attrs={"dtype": dtype, "length": length}))
    em.emit_op(ti.TileOp(ti.TileOpKind.TILE_ZEROS, [], [rhs],
                         attrs={"dtype": dtype, "length": length}))
    return lhs, rhs


def test_stage64_inc3_tile_add_emits_f32_register_adds():
    """Stage 64 Inc 3 — TILE_ADD on two length-4 f32 register-tiles
    emits 4 `add.f32` lines elementwise, with the result mapped to
    a fresh contiguous %f base register."""
    from helixc.backend.ptx import PtxEmitter
    em = PtxEmitter()
    lhs, rhs = _build_two_zero_tiles(em, "f32", 4)
    out = ti.TileValue(2, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(4),), "REG"
    ))
    pre_text_len = len(em.buf.getvalue().splitlines())
    em.emit_op(ti.TileOp(ti.TileOpKind.TILE_ADD, [lhs, rhs], [out]))
    new_lines = em.buf.getvalue().splitlines()[pre_text_len:]
    add_lines = [ln for ln in new_lines if "add.f32" in ln]
    assert len(add_lines) == 4, em.buf.getvalue()
    # Each line should reference three %f registers.
    for ln in add_lines:
        assert ln.count("%f") == 3, ln
    # Result mapped to a %f base register.
    assert out.id in em.reg_map
    assert em.reg_map[out.id].startswith("%f"), em.reg_map[out.id]


def test_stage64_inc3_tile_sub_emits_f32_register_subs():
    """Stage 64 Inc 3 — TILE_SUB on two length-3 f32 tiles emits
    3 `sub.f32` lines."""
    from helixc.backend.ptx import PtxEmitter
    em = PtxEmitter()
    lhs, rhs = _build_two_zero_tiles(em, "f32", 3)
    out = ti.TileValue(2, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(3),), "REG"
    ))
    pre_text_len = len(em.buf.getvalue().splitlines())
    em.emit_op(ti.TileOp(ti.TileOpKind.TILE_SUB, [lhs, rhs], [out]))
    new_lines = em.buf.getvalue().splitlines()[pre_text_len:]
    sub_lines = [ln for ln in new_lines if "sub.f32" in ln]
    assert len(sub_lines) == 3, em.buf.getvalue()
    for ln in sub_lines:
        assert ln.count("%f") == 3, ln


def test_stage64_inc3_tile_mul_emits_f32_register_muls():
    """Stage 64 Inc 3 — TILE_MUL on two length-2 f32 tiles emits
    2 `mul.f32` lines."""
    from helixc.backend.ptx import PtxEmitter
    em = PtxEmitter()
    lhs, rhs = _build_two_zero_tiles(em, "f32", 2)
    out = ti.TileValue(2, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(2),), "REG"
    ))
    pre_text_len = len(em.buf.getvalue().splitlines())
    em.emit_op(ti.TileOp(ti.TileOpKind.TILE_MUL, [lhs, rhs], [out]))
    new_lines = em.buf.getvalue().splitlines()[pre_text_len:]
    mul_lines = [ln for ln in new_lines if "mul.f32" in ln]
    assert len(mul_lines) == 2, em.buf.getvalue()
    for ln in mul_lines:
        assert ln.count("%f") == 3, ln


def test_stage64_inc3_tile_add_emits_i32_register_adds():
    """Stage 64 Inc 3 — TILE_ADD on two length-3 i32 tiles emits
    3 `add.s32` lines on %r registers."""
    from helixc.backend.ptx import PtxEmitter
    em = PtxEmitter()
    lhs, rhs = _build_two_zero_tiles(em, "i32", 3)
    out = ti.TileValue(2, tir.TIRTileTy(
        tir.TIRScalar("i32"), (tir.DimConst(3),), "REG"
    ))
    pre_text_len = len(em.buf.getvalue().splitlines())
    em.emit_op(ti.TileOp(ti.TileOpKind.TILE_ADD, [lhs, rhs], [out]))
    new_lines = em.buf.getvalue().splitlines()[pre_text_len:]
    add_lines = [ln for ln in new_lines if "add.s32" in ln]
    assert len(add_lines) == 3, em.buf.getvalue()
    for ln in add_lines:
        assert ln.count("%r") == 3, ln
    assert em.reg_map[out.id].startswith("%r"), em.reg_map[out.id]


def test_stage64_inc3_tile_mul_emits_i32_register_muls_lo_s32():
    """Stage 64 Inc 3 — TILE_MUL on i32 tiles uses `mul.lo.s32`
    (low-32-bit signed multiply), matching SCALAR_MUL i32 idiom."""
    from helixc.backend.ptx import PtxEmitter
    em = PtxEmitter()
    lhs, rhs = _build_two_zero_tiles(em, "i32", 2)
    out = ti.TileValue(2, tir.TIRTileTy(
        tir.TIRScalar("i32"), (tir.DimConst(2),), "REG"
    ))
    pre_text_len = len(em.buf.getvalue().splitlines())
    em.emit_op(ti.TileOp(ti.TileOpKind.TILE_MUL, [lhs, rhs], [out]))
    new_lines = em.buf.getvalue().splitlines()[pre_text_len:]
    mul_lines = [ln for ln in new_lines if "mul.lo.s32" in ln]
    assert len(mul_lines) == 2, em.buf.getvalue()


def test_stage64_inc3_tile_add_rejects_mismatched_dtypes():
    """Stage 64 Inc 3 — TILE_ADD with f32 lhs + i32 rhs fails
    closed with a clear dtype-mismatch message."""
    from helixc.backend.ptx import PtxEmitter
    em = PtxEmitter()
    lhs = ti.TileValue(0, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(2),), "REG"
    ))
    rhs = ti.TileValue(1, tir.TIRTileTy(
        tir.TIRScalar("i32"), (tir.DimConst(2),), "REG"
    ))
    em.emit_op(ti.TileOp(ti.TileOpKind.TILE_ZEROS, [], [lhs],
                         attrs={"dtype": "f32", "length": 2}))
    em.emit_op(ti.TileOp(ti.TileOpKind.TILE_ZEROS, [], [rhs],
                         attrs={"dtype": "i32", "length": 2}))
    out = ti.TileValue(2, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(2),), "REG"
    ))
    import pytest as _pt
    with _pt.raises(RuntimeError, match="requires matching dtypes"):
        em.emit_op(ti.TileOp(ti.TileOpKind.TILE_ADD, [lhs, rhs], [out]))


def test_stage64_inc3_tile_add_rejects_missing_register():
    """Stage 64 Inc 3 — TILE_ADD where lhs or rhs has never been
    lowered (no entry in reg_map) fails closed with a clear
    'no PTX register' message."""
    from helixc.backend.ptx import PtxEmitter
    em = PtxEmitter()
    lhs = ti.TileValue(0, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(2),), "REG"
    ))
    rhs = ti.TileValue(1, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(2),), "REG"
    ))
    # Only lhs is lowered; rhs has no reg_map entry.
    em.emit_op(ti.TileOp(ti.TileOpKind.TILE_ZEROS, [], [lhs],
                         attrs={"dtype": "f32", "length": 2}))
    out = ti.TileValue(2, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(2),), "REG"
    ))
    import pytest as _pt
    with _pt.raises(RuntimeError, match="rhs has no PTX register"):
        em.emit_op(ti.TileOp(ti.TileOpKind.TILE_ADD, [lhs, rhs], [out]))


def test_stage64_inc3_tile_add_rejects_unsupported_dtype():
    """Stage 64 Inc 3 — TILE_ADD on bf16 tiles fails closed with
    a clear 'Inc 4+ will extend' message (matches Inc 2 dtype
    scope policy)."""
    from helixc.backend.ptx import PtxEmitter
    em = PtxEmitter()
    # Manually construct bf16 tiles + manually populate reg_map so
    # the dtype check is what trips, not the missing-register check.
    lhs = ti.TileValue(0, tir.TIRTileTy(
        tir.TIRScalar("bf16"), (tir.DimConst(2),), "REG"
    ))
    rhs = ti.TileValue(1, tir.TIRTileTy(
        tir.TIRScalar("bf16"), (tir.DimConst(2),), "REG"
    ))
    em.reg_map[lhs.id] = "%h0"
    em.reg_map[rhs.id] = "%h2"
    out = ti.TileValue(2, tir.TIRTileTy(
        tir.TIRScalar("bf16"), (tir.DimConst(2),), "REG"
    ))
    import pytest as _pt
    with _pt.raises(RuntimeError, match="Inc 4\\+ will extend"):
        em.emit_op(ti.TileOp(ti.TileOpKind.TILE_ADD, [lhs, rhs], [out]))


def test_stage64_inc3_tile_add_rejects_mismatched_lengths():
    """Stage 64 Inc 3 — TILE_ADD with operands of different
    lengths fails closed with a clear length-mismatch message."""
    from helixc.backend.ptx import PtxEmitter
    em = PtxEmitter()
    lhs = ti.TileValue(0, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(2),), "REG"
    ))
    rhs = ti.TileValue(1, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(3),), "REG"
    ))
    em.emit_op(ti.TileOp(ti.TileOpKind.TILE_ZEROS, [], [lhs],
                         attrs={"dtype": "f32", "length": 2}))
    em.emit_op(ti.TileOp(ti.TileOpKind.TILE_ZEROS, [], [rhs],
                         attrs={"dtype": "f32", "length": 3}))
    out = ti.TileValue(2, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(2),), "REG"
    ))
    import pytest as _pt
    with _pt.raises(RuntimeError, match="requires matching lengths"):
        em.emit_op(ti.TileOp(ti.TileOpKind.TILE_ADD, [lhs, rhs], [out]))


# ============================================================================
# Stage 64 Inc 4 / Stage 106 — TILE_MATMUL via NVIDIA wmma Tensor Core
# fragments (canonical m16n16k16 shape). Backend-only — no frontend
# trigger path yet; tests synthesize fragment-shaped operand tiles
# directly and pre-populate reg_map to avoid needing a TILE_HBM_TO_REG
# wmma-layout op (that's Inc 5 / SMEM-staging work).
# ============================================================================
def _build_wmma_operand_tiles(em, ab_dtype: str):
    """Helper for Stage 106 — synthesize a (A, B, C) operand triple
    of the canonical wmma m16n16k16 fragment shape:
        A: 4 packed .b32 regs (holding f16/bf16 pairs)
        B: 4 packed .b32 regs
        C: 8 .f32 regs (accumulator)
    Pre-populates em.reg_map with contiguous %r0..%r3, %r4..%r7,
    %f0..%f7 so the wmma dispatch sees lowered operands. Returns
    (A, B, C) TileValues."""
    a = ti.TileValue(100, tir.TIRTileTy(
        tir.TIRScalar(ab_dtype), (tir.DimConst(4),), "REG"
    ))
    b = ti.TileValue(101, tir.TIRTileTy(
        tir.TIRScalar(ab_dtype), (tir.DimConst(4),), "REG"
    ))
    c = ti.TileValue(102, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(8),), "REG"
    ))
    # Pre-allocate via _new_reg so the counters track correctly.
    a_base = None
    for _ in range(4):
        r = em._new_reg("r")
        if a_base is None:
            a_base = r
    em.reg_map[a.id] = a_base
    b_base = None
    for _ in range(4):
        r = em._new_reg("r")
        if b_base is None:
            b_base = r
    em.reg_map[b.id] = b_base
    c_base = None
    for _ in range(8):
        r = em._new_reg("f")
        if c_base is None:
            c_base = r
    em.reg_map[c.id] = c_base
    return a, b, c


def test_stage106_tile_matmul_emits_wmma_mma_sync_f16():
    """Stage 106 (Stage 64 Inc 4) — TILE_MATMUL with f16 A/B and f32
    C/D emits a single `wmma.mma.sync.aligned.m16n16k16.row.col.
    f32.f16.f16.f32` line with the expected 4+4+8+8 register-list
    layout. D is mapped to a fresh %f base register."""
    from helixc.backend.ptx import PtxEmitter
    em = PtxEmitter()
    a, b, c = _build_wmma_operand_tiles(em, "f16")
    d = ti.TileValue(103, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(8),), "REG"
    ))
    pre_text_len = len(em.buf.getvalue().splitlines())
    em.emit_op(ti.TileOp(ti.TileOpKind.TILE_MATMUL, [a, b, c], [d]))
    new_lines = em.buf.getvalue().splitlines()[pre_text_len:]
    wmma_lines = [ln for ln in new_lines if "wmma.mma.sync" in ln]
    assert len(wmma_lines) == 1, em.buf.getvalue()
    line = wmma_lines[0]
    assert "m16n16k16.row.col.f32.f16.f16.f32" in line, line
    # 4 A regs + 4 B regs + 8 C regs + 8 D regs = 24 %-prefixed refs.
    assert line.count("%r") == 8, line   # 4 A + 4 B in %r pool
    assert line.count("%f") == 16, line  # 8 C + 8 D in %f pool
    # D should be mapped to a fresh %f base register.
    assert d.id in em.reg_map
    assert em.reg_map[d.id].startswith("%f"), em.reg_map[d.id]


def test_stage106_tile_matmul_emits_wmma_mma_sync_bf16():
    """Stage 106 — TILE_MATMUL with bf16 A/B emits the bf16 wmma
    variant (`f32.bf16.bf16.f32`). Parity check with the f16
    variant above."""
    from helixc.backend.ptx import PtxEmitter
    em = PtxEmitter()
    a, b, c = _build_wmma_operand_tiles(em, "bf16")
    d = ti.TileValue(103, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(8),), "REG"
    ))
    pre_text_len = len(em.buf.getvalue().splitlines())
    em.emit_op(ti.TileOp(ti.TileOpKind.TILE_MATMUL, [a, b, c], [d]))
    new_lines = em.buf.getvalue().splitlines()[pre_text_len:]
    wmma_lines = [ln for ln in new_lines if "wmma.mma.sync" in ln]
    assert len(wmma_lines) == 1, em.buf.getvalue()
    assert ("m16n16k16.row.col.f32.bf16.bf16.f32"
            in wmma_lines[0]), wmma_lines[0]


def test_stage106_tile_matmul_rejects_mismatched_ab_dtype():
    """Stage 106 — A and B fragments must have matching dtype. Mixed
    f16/bf16 wmma exists in hardware but isn't supported in Phase-0."""
    from helixc.backend.ptx import PtxEmitter
    em = PtxEmitter()
    a, _b_unused, c = _build_wmma_operand_tiles(em, "f16")
    # Build a fresh B as bf16.
    b = ti.TileValue(200, tir.TIRTileTy(
        tir.TIRScalar("bf16"), (tir.DimConst(4),), "REG"
    ))
    b_base = None
    for _ in range(4):
        r = em._new_reg("r")
        if b_base is None:
            b_base = r
    em.reg_map[b.id] = b_base
    d = ti.TileValue(103, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(8),), "REG"
    ))
    import pytest as _pt
    with _pt.raises(RuntimeError, match="A/B dtypes must match"):
        em.emit_op(ti.TileOp(
            ti.TileOpKind.TILE_MATMUL, [a, b, c], [d]))


def test_stage106_tile_matmul_rejects_non_f32_c():
    """Stage 106 — C accumulator dtype must be f32 in Phase-0
    (Inc 5+ adds f16-accumulator wmma variants)."""
    from helixc.backend.ptx import PtxEmitter
    em = PtxEmitter()
    a, b, _c_unused = _build_wmma_operand_tiles(em, "f16")
    # Build a fresh C as f16 (illegal accumulator in Phase-0).
    c = ti.TileValue(300, tir.TIRTileTy(
        tir.TIRScalar("f16"), (tir.DimConst(8),), "REG"
    ))
    c_base = None
    for _ in range(8):
        r = em._new_reg("h")
        if c_base is None:
            c_base = r
    em.reg_map[c.id] = c_base
    d = ti.TileValue(103, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(8),), "REG"
    ))
    import pytest as _pt
    with _pt.raises(RuntimeError,
                    match="C accumulator dtype must be f32"):
        em.emit_op(ti.TileOp(
            ti.TileOpKind.TILE_MATMUL, [a, b, c], [d]))


def test_stage106_tile_matmul_rejects_unsupported_dtype():
    """Stage 106 — Phase-0 only supports f16 / bf16 for A/B. f32×f32
    wmma (via Tensor Core f32 variants), tf32, int8, fp8 all reject
    cleanly with a clear Inc 5+ message."""
    from helixc.backend.ptx import PtxEmitter
    em = PtxEmitter()
    a, _b_unused, c = _build_wmma_operand_tiles(em, "f16")
    # Build A as f32 (unsupported).
    a32 = ti.TileValue(400, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(4),), "REG"
    ))
    a_base = None
    for _ in range(4):
        r = em._new_reg("r")
        if a_base is None:
            a_base = r
    em.reg_map[a32.id] = a_base
    b32 = ti.TileValue(401, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(4),), "REG"
    ))
    b_base = None
    for _ in range(4):
        r = em._new_reg("r")
        if b_base is None:
            b_base = r
    em.reg_map[b32.id] = b_base
    d = ti.TileValue(103, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(8),), "REG"
    ))
    import pytest as _pt
    with _pt.raises(RuntimeError,
                    match="dtype must be f16 or bf16"):
        em.emit_op(ti.TileOp(
            ti.TileOpKind.TILE_MATMUL, [a32, b32, c], [d]))


def test_stage106_tile_matmul_rejects_wrong_fragment_length():
    """Stage 106 — A must be length 4 (canonical packed-pair
    fragment). Other lengths reject before reaching the wmma emit."""
    from helixc.backend.ptx import PtxEmitter
    em = PtxEmitter()
    # Build A of length 3 (illegal).
    a = ti.TileValue(500, tir.TIRTileTy(
        tir.TIRScalar("f16"), (tir.DimConst(3),), "REG"
    ))
    a_base = None
    for _ in range(3):
        r = em._new_reg("r")
        if a_base is None:
            a_base = r
    em.reg_map[a.id] = a_base
    # Build B, C as well-formed.
    b = ti.TileValue(501, tir.TIRTileTy(
        tir.TIRScalar("f16"), (tir.DimConst(4),), "REG"
    ))
    b_base = None
    for _ in range(4):
        r = em._new_reg("r")
        if b_base is None:
            b_base = r
    em.reg_map[b.id] = b_base
    c = ti.TileValue(502, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(8),), "REG"
    ))
    c_base = None
    for _ in range(8):
        r = em._new_reg("f")
        if c_base is None:
            c_base = r
    em.reg_map[c.id] = c_base
    d = ti.TileValue(503, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(8),), "REG"
    ))
    import pytest as _pt
    with _pt.raises(RuntimeError, match="A must have length 4"):
        em.emit_op(ti.TileOp(
            ti.TileOpKind.TILE_MATMUL, [a, b, c], [d]))


def test_c118_direct_ptx_cli_aborts_on_type_errors():
    proc = run_ptx_cli("@kernel fn k() { let mut b: bool = true; b += false; }\n")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "operator '+' does not support operand type bool" in proc.stderr
    assert "add.s32" not in proc.stdout


def test_c118_hbm_tile_index_missing_ptx_register_fails_closed():
    a = ti.TileValue(0, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(256),), "HBM"
    ), name_hint="a")
    missing_index = ti.TileValue(1, tir.TIRScalar("i32"))
    out = ti.TileValue(2, tir.TIRScalar("f32"))
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.TILE_INDEX_LOAD_HBM, [missing_index], [out],
                  attrs={"name": "a", "dtype": "f32"})
    ])
    fn = ti.TileFn("k", [a], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="missing PTX register for HBM tile index"):
        emit_ptx(mod)


def test_c119_hbm_tile_missing_param_map_entry_fails_closed():
    idx = ti.TileValue(0, tir.TIRScalar("i32"))
    out = ti.TileValue(1, tir.TIRScalar("f32"))
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [idx], attrs={"value": 0}),
        ti.TileOp(ti.TileOpKind.TILE_INDEX_LOAD_HBM, [idx], [out],
                  attrs={"name": "missing", "dtype": "f32"}),
    ])
    fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="not in PTX param map"):
        emit_ptx(mod)


def test_c119_hbm_tile_index_rejects_address_register_class():
    from helixc.backend.ptx import PtxEmitter
    idx = ti.TileValue(0, tir.TIRScalar("i32"))
    out = ti.TileValue(1, tir.TIRScalar("f32"))
    op = ti.TileOp(ti.TileOpKind.TILE_INDEX_LOAD_HBM, [idx], [out],
                   attrs={"name": "a", "dtype": "f32"})
    em = PtxEmitter()
    em.hbm_param_map = {"a": (0, "f32")}
    em.reg_map = {idx.id: "%rd0"}
    with pytest.raises(RuntimeError, match="expected %r register class"):
        em.emit_op(op)


def test_c119_hbm_store_value_must_match_tile_dtype():
    idx = ti.TileValue(0, tir.TIRScalar("i32"))
    bad_value = ti.TileValue(1, tir.TIRScalar("bool"))
    param = ti.TileValue(10, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(4),), "HBM"
    ), name_hint="a")
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [idx], attrs={"value": 0}),
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [bad_value], attrs={"value": 1}),
        ti.TileOp(ti.TileOpKind.TILE_INDEX_STORE_HBM, [idx, bad_value], [],
                  attrs={"name": "a", "dtype": "f32"}),
    ])
    fn = ti.TileFn("k", [param], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="unsupported PTX HBM tile store value type bool"):
        emit_ptx(mod)


def test_c119_hbm_op_dtype_must_match_param_map_dtype():
    idx = ti.TileValue(0, tir.TIRScalar("i32"))
    out = ti.TileValue(1, tir.TIRScalar("f32"))
    param = ti.TileValue(10, tir.TIRTileTy(
        tir.TIRScalar("i32"), (tir.DimConst(4),), "HBM"
    ), name_hint="a")
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [idx], attrs={"value": 0}),
        ti.TileOp(ti.TileOpKind.TILE_INDEX_LOAD_HBM, [idx], [out],
                  attrs={"name": "a", "dtype": "f32"}),
    ])
    fn = ti.TileFn("k", [param], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="dtype mismatch"):
        emit_ptx(mod)


def test_c119_hbm_param_shape_validated_in_direct_ptx():
    param = ti.TileValue(10, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(4), tir.DimConst(4)), "HBM"
    ), name_hint="a")
    fn = ti.TileFn("k", [param], tir.TIRUnit(), [ti.TileBlock(0)],
                   attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="HBM tile parameters must be 1D"):
        emit_ptx(mod)


def test_c119_hbm_ops_require_dtype_attr():
    idx = ti.TileValue(0, tir.TIRScalar("i32"))
    out = ti.TileValue(1, tir.TIRScalar("f32"))
    param = ti.TileValue(10, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(4),), "HBM"
    ), name_hint="a")
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [idx], attrs={"value": 0}),
        ti.TileOp(ti.TileOpKind.TILE_INDEX_LOAD_HBM, [idx], [out],
                  attrs={"name": "a"}),
    ])
    fn = ti.TileFn("k", [param], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="missing PTX HBM tile dtype attr"):
        emit_ptx(mod)


def test_c119_thread_idx_requires_valid_attrs_in_direct_ptx():
    out = ti.TileValue(0, tir.TIRScalar("i32"))

    def mod_for(attrs):
        block = ti.TileBlock(0, ops=[
            ti.TileOp(ti.TileOpKind.THREAD_IDX, [], [out], attrs=attrs)
        ])
        fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
        return ti.TileModule(functions={"k": fn})

    with pytest.raises(RuntimeError, match="requires explicit dim and sreg"):
        emit_ptx(mod_for({}))
    with pytest.raises(RuntimeError, match="unsupported PTX THREAD_IDX dim"):
        emit_ptx(mod_for({"dim": "w", "sreg": "tid"}))
    with pytest.raises(RuntimeError, match="unsupported PTX THREAD_IDX sreg"):
        emit_ptx(mod_for({"dim": "x", "sreg": "foo"}))


def test_c119_thread_idx_requires_valid_op_shape_in_direct_ptx():
    good = ti.TileValue(0, tir.TIRScalar("i32"))
    bad = ti.TileValue(1, tir.TIRScalar("f32"))
    attrs = {"dim": "x", "sreg": "tid"}

    def mod_for(op):
        block = ti.TileBlock(0, ops=[op])
        fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
        return ti.TileModule(functions={"k": fn})

    with pytest.raises(RuntimeError, match="expects exactly 0 operand"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.THREAD_IDX, [good], [good],
                                   attrs=attrs)))
    with pytest.raises(RuntimeError, match="expects exactly 1 result"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.THREAD_IDX, [], [],
                                   attrs=attrs)))
    with pytest.raises(RuntimeError, match="unsupported PTX THREAD_IDX result type f32"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.THREAD_IDX, [], [bad],
                                   attrs=attrs)))


def test_c119_scalar_constants_require_value_attr_in_direct_ptx():
    int_out = ti.TileValue(0, tir.TIRScalar("i32"))
    float_out = ti.TileValue(1, tir.TIRScalar("f32"))

    def mod_for(op):
        block = ti.TileBlock(0, ops=[op])
        fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
        return ti.TileModule(functions={"k": fn})

    with pytest.raises(RuntimeError, match="SCALAR_CONST_INT requires value attr"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [int_out])))
    with pytest.raises(RuntimeError, match="SCALAR_CONST_FLOAT requires value attr"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.SCALAR_CONST_FLOAT, [], [float_out])))


def test_c119_scalar_constant_values_are_not_coerced_in_direct_ptx():
    int_out = ti.TileValue(0, tir.TIRScalar("i32"))
    bool_out = ti.TileValue(1, tir.TIRScalar("bool"))
    float_out = ti.TileValue(2, tir.TIRScalar("f32"))

    def mod_for(op):
        block = ti.TileBlock(0, ops=[op])
        fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
        return ti.TileModule(functions={"k": fn})

    with pytest.raises(RuntimeError, match="i32 value must be an int"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [int_out],
                                   attrs={"value": 1.9})))
    with pytest.raises(RuntimeError, match="i32 value must be an int"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [int_out],
                                   attrs={"value": "7"})))
    with pytest.raises(RuntimeError, match="bool value must be true/false or 0/1"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [bool_out],
                                   attrs={"value": 2})))
    with pytest.raises(RuntimeError, match="value must be a float"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.SCALAR_CONST_FLOAT, [], [float_out],
                                   attrs={"value": "1.25"})))


def test_c119_i32_scalar_constants_require_i32_range():
    int_out = ti.TileValue(0, tir.TIRScalar("i32"))
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [int_out],
                  attrs={"value": 2 ** 40})
    ])
    fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="i32 value out of range"):
        emit_ptx(mod)


def test_c119_direct_ptx_rejects_non_unit_kernel_returns():
    fn = ti.TileFn("k", [], tir.TIRScalar("i32"), [ti.TileBlock(0)],
                   attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="non-unit returns"):
        emit_ptx(mod)


def test_c119_direct_ptx_rejects_return_value_ops():
    value = ti.TileValue(0, tir.TIRScalar("i32"))
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [value], attrs={"value": 1}),
        ti.TileOp(ti.TileOpKind.RETURN, [value], []),
    ])
    fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="cannot return values"):
        emit_ptx(mod)


def test_c119_scalar_compare_requires_valid_cmp_attr_in_direct_ptx():
    a = ti.TileValue(0, tir.TIRScalar("i32"))
    b = ti.TileValue(1, tir.TIRScalar("i32"))
    out = ti.TileValue(2, tir.TIRScalar("bool"))

    def mod_for(attrs):
        block = ti.TileBlock(0, ops=[
            ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [a], attrs={"value": 1}),
            ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [b], attrs={"value": 2}),
            ti.TileOp(ti.TileOpKind.SCALAR_CMP, [a, b], [out], attrs=attrs),
        ])
        fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
        return ti.TileModule(functions={"k": fn})

    with pytest.raises(RuntimeError, match="SCALAR_CMP requires cmp attr"):
        emit_ptx(mod_for({}))
    with pytest.raises(RuntimeError, match="unsupported PTX scalar compare op"):
        emit_ptx(mod_for({"cmp": "cmp.nope"}))


def test_c119_ptx_ops_require_exact_operand_counts():
    a = ti.TileValue(0, tir.TIRScalar("i32"))
    b = ti.TileValue(1, tir.TIRScalar("i32"))
    c = ti.TileValue(2, tir.TIRScalar("i32"))
    out = ti.TileValue(3, tir.TIRScalar("i32"))
    param = ti.TileValue(10, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(4),), "HBM"
    ), name_hint="a")
    fval = ti.TileValue(11, tir.TIRScalar("f32"))

    def mod_for(op, params=None):
        block = ti.TileBlock(0, ops=[op])
        fn = ti.TileFn("k", params or [], tir.TIRUnit(), [block],
                       attrs={"kernel": True})
        return ti.TileModule(functions={"k": fn})

    with pytest.raises(RuntimeError, match="SCALAR_ADD expects exactly 2 operand"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.SCALAR_ADD, [a, b, c], [out])))
    with pytest.raises(RuntimeError, match="SCALAR_CMP expects exactly 2 operand"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.SCALAR_CMP, [a, b, c], [out],
                                   attrs={"cmp": "cmp.lt"})))
    with pytest.raises(RuntimeError, match="TILE_INDEX_LOAD_HBM expects exactly 1 operand"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.TILE_INDEX_LOAD_HBM, [a, b], [fval],
                                   attrs={"name": "a", "dtype": "f32"}), [param]))
    with pytest.raises(RuntimeError, match="TILE_INDEX_STORE_HBM expects exactly 2 operand"):
        emit_ptx(mod_for(ti.TileOp(ti.TileOpKind.TILE_INDEX_STORE_HBM, [a, fval, b], [],
                                   attrs={"name": "a", "dtype": "f32"}), [param]))


def test_c119_scalar_arithmetic_result_type_must_match_operands():
    a = ti.TileValue(0, tir.TIRScalar("i32"))
    b = ti.TileValue(1, tir.TIRScalar("i32"))
    bad_out = ti.TileValue(2, tir.TIRScalar("f32"))
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [a], attrs={"value": 1}),
        ti.TileOp(ti.TileOpKind.SCALAR_CONST_INT, [], [b], attrs={"value": 2}),
        ti.TileOp(ti.TileOpKind.SCALAR_ADD, [a, b], [bad_out]),
    ])
    fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="unsupported PTX scalar add result type f32"):
        emit_ptx(mod)


def test_c119_direct_ptx_cli_rejects_kernel_helper_calls():
    src = """
    fn helper(x: i32) -> i32 { x + 1 }
    @kernel fn k() { let y = helper(41); }
    """
    proc = run_ptx_cli(src)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "unsupported PTX op call" in proc.stderr
    assert "// TODO:" not in proc.stdout


def test_c119_direct_ptx_cli_rejects_scalar_kernel_params():
    proc = run_ptx_cli("@kernel fn k(x: i32) { let z = x + 2; }\n")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "PTX kernel parameter is not supported yet" in proc.stderr
    assert "add.s32" not in proc.stdout


def test_c119_direct_ptx_cli_rejects_modules_without_kernels():
    proc = run_ptx_cli("fn helper(x: i32) -> i32 { x + 1 }\n")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "PTX emission requires at least one @kernel function" in proc.stderr
    assert ".func" not in proc.stdout


def test_stage35_direct_ptx_cli_rejects_oversized_autotune():
    src = """
    @kernel
    @autotune(A: [1, 2, 3, 4, 5], B: [10, 20, 30, 40, 50])
    fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "trap 27001" in proc.stderr
    assert ".visible .entry" not in proc.stdout


def test_stage35_direct_ptx_cli_ignores_host_helper_with_unsupported_tile_op():
    src = """
    fn host_helper(x: i32) -> i32 { x / 2 }
    @kernel fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "elem.div" not in proc.stderr


def test_stage35_direct_ptx_cli_ignores_host_ad_function():
    src = """
    fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }
    @kernel fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "ad:" in proc.stderr
    assert "unresolved generic type D" not in proc.stderr


def test_stage35_direct_ptx_cli_strict_ignores_host_ad_function():
    src = """
    fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }
    @kernel fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src, "--strict")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "ad:" in proc.stderr
    assert "unresolved generic type D" not in proc.stderr
    assert "compiler bug" not in proc.stderr


def test_stage35_direct_ptx_cli_strict_rejects_host_effect_with_dead_ad_helper():
    src = """
    fn loss(x: D<f64>) -> D<f64> { x }
    @pure fn host() -> i32 { print_int(1); 0 }
    @kernel fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src, "--strict", "--no-stdlib")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert proc.stdout == ""
    assert "--strict aborts" in proc.stderr
    assert "19001" in proc.stderr
    assert "unresolved generic type D" not in proc.stderr


def test_stage35_direct_ptx_cli_non_strict_reports_host_effect_warning():
    src = """
    @pure fn host() -> i32 { print_int(1); 0 }
    @kernel fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src, "--no-stdlib")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "warning: effect-check:" in proc.stderr
    assert "19001" in proc.stderr


def test_stage35_direct_ptx_cli_wad_error_keeps_stdout_empty():
    src = """
    fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }
    @kernel fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src, "-Wad=error")
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert proc.stdout == ""
    assert "ad:" in proc.stderr
    assert "ERROR" in proc.stderr
    assert "AD002" in proc.stderr or "24200" in proc.stderr


def test_stage35_direct_ptx_cli_accepts_wad_warn_policy():
    src = """
    fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }
    @kernel fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src, "-Wad=warn")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "ad:" in proc.stderr


def test_stage35_direct_ptx_cli_warning_policy_uses_last_flag():
    src = """
    fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }
    @kernel fn k() { let i = thread_idx(); }
    """
    error_proc = run_ptx_cli(src, "-Wad=warn", "-Wad=error")
    assert error_proc.returncode == 1, error_proc.stdout + error_proc.stderr
    assert error_proc.stdout == ""
    assert "ad:" in error_proc.stderr
    assert "ERROR" in error_proc.stderr

    warn_proc = run_ptx_cli(src, "-Wad=error", "-Wad=warn")
    assert warn_proc.returncode == 0, warn_proc.stdout + warn_proc.stderr
    assert ".visible .entry k" in warn_proc.stdout
    assert "ad:" in warn_proc.stderr
    assert "ERROR" not in warn_proc.stderr


def test_stage35_direct_ptx_cli_rejects_conflicting_stdlib_flags():
    proc = run_ptx_cli(
        "@kernel fn k() { let i = thread_idx(); }\n",
        "--stdlib", "--no-stdlib",
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert proc.stdout == ""
    assert "conflicting stdlib flags" in proc.stderr


def test_stage35_direct_ptx_cli_deprecated_warning_policy():
    src = """
    @deprecated fn old() -> i32 { 0 }
    fn host() -> i32 { old() }
    @kernel fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "deprecated:" in proc.stderr
    err_proc = run_ptx_cli(src, "-Wdeprecated=error")
    assert err_proc.returncode != 0, err_proc.stdout + err_proc.stderr
    assert err_proc.stdout == ""
    assert "deprecated:" in err_proc.stderr
    assert "ERROR" in err_proc.stderr


def test_stage35_direct_ptx_cli_drains_ad_warnings_on_error():
    src = """
    fn loss(x: D<f64>, y: D<i32>) -> D<f64> { x + y }
    @kernel fn k() { let bad: i32 = true; }
    """
    proc = run_ptx_cli(src)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert proc.stdout == ""
    assert "error:" in proc.stderr
    assert "ad:" in proc.stderr
    assert "AD002" in proc.stderr or "24200" in proc.stderr


def test_stage35_direct_ptx_cli_rejects_unwind_attr():
    proc = run_ptx_cli("@unwind @kernel fn k() { let i = thread_idx(); }\n")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "unwind" in proc.stderr
    assert ".visible .entry" not in proc.stdout


def test_stage35_direct_ptx_cli_folds_kernel_before_tile_lowering():
    proc = run_ptx_cli("@kernel fn k() { let z = 4 / 2; }\n")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "elem.div" not in proc.stderr


def test_stage35_direct_ptx_cli_flattens_module_kernel():
    src = """
    mod m {
        @kernel fn k() { let i = thread_idx(); }
    }
    """
    proc = run_ptx_cli(src)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry m__k" in proc.stdout


def test_stage35_direct_ptx_cli_rejects_duplicate_autotune_key():
    src = """
    @kernel
    @autotune(A: [1], A: [2])
    fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "duplicate parameter" in proc.stderr
    assert ".visible .entry" not in proc.stdout


def test_stage35_direct_ptx_cli_strict_rejects_effect_violation():
    src = """
    @pure fn host() -> i32 {
        print_int(1);
        0
    }
    @kernel fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src, "--strict", "--no-stdlib")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "--strict aborts" in proc.stderr
    assert ".visible .entry" not in proc.stdout


def test_stage35_direct_ptx_cli_strict_rejects_totality_failure():
    src = """
    fn spin(n: i32) -> i32 { spin(n) }
    @kernel fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src, "--strict")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "totality" in proc.stderr
    assert "--strict aborts" in proc.stderr
    assert ".visible .entry" not in proc.stdout


def test_stage35_direct_ptx_cli_includes_stdlib_by_default():
    src = """
    fn host(x: f32) -> f32 { __relu(x) }
    @kernel fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "unbound name '__relu'" not in proc.stderr


def test_stage35_direct_ptx_cli_strict_allows_clean_default_stdlib_kernel():
    proc = run_ptx_cli("@kernel fn k() { let i = thread_idx(); }\n", "--strict")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "effect-check warning" not in proc.stderr
    assert "vec_push" not in proc.stderr


def test_stage35_direct_ptx_cli_accepts_stdlib_compat_flag():
    src = """
    fn host(x: f32) -> f32 { __relu(x) }
    @kernel fn k() { let i = thread_idx(); }
    """
    proc = run_ptx_cli(src, "--stdlib")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert ".visible .entry k" in proc.stdout
    assert "unknown flag --stdlib" not in proc.stderr


def test_stage35_direct_ptx_cli_reports_missing_file_without_traceback():
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    missing = os.path.join(proj_root, "__definitely_missing_stage35__.hx")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.ptx", missing],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "cannot read" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_stage35_direct_ptx_cli_reports_encoding_error_without_traceback(tmp_path):
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bad = tmp_path / "bad_utf8.hx"
    bad.write_bytes(b"\xff\xfe\xfd")
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.ptx", str(bad)],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "encoding error reading source" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_stage35_direct_ptx_cli_bad_invocation_returns_two():
    # Restart 49 B3 added a usage banner to the bare-invocation path
    # (previously printed only "error: ptx: missing input path"). Either
    # diagnostic is acceptable as long as rc=2 and the user sees a hint.
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.ptx"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert ("missing input path" in proc.stderr
            or "usage:" in proc.stderr.lower()), (
        f"bare ptx invocation should print 'missing input path' or "
        f"'usage:'; got stderr={proc.stderr!r}"
    )

    for args in (["--strict"], ["--stdlib"], ["--strict", "--stdlib"]):
        proc = subprocess.run(
            [sys.executable, "-m", "helixc.backend.ptx", *args],
            cwd=proj_root,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert proc.returncode == 2, proc.stdout + proc.stderr
        assert ("missing input path" in proc.stderr
                or "usage:" in proc.stderr.lower()), (
            f"ptx with flag-only args {args!r} should print "
            f"'missing input path' or 'usage:'; got stderr={proc.stderr!r}"
        )

    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.ptx", "--bogus"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "unknown flag --bogus" in proc.stderr

    proc = subprocess.run(
        [sys.executable, "-m", "helixc.backend.ptx", "a.hx", "b.hx"],
        cwd=proj_root,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 2, proc.stdout + proc.stderr
    assert "expected at most one input path" in proc.stderr


def test_stage35_direct_ptx_cli_missing_strict_stdlib_returns_two(monkeypatch, capsys, tmp_path):
    src_path = tmp_path / "k.hx"
    src_path.write_text("@kernel fn k() { let i = thread_idx(); }\n", encoding="utf-8")
    monkeypatch.setenv(parser_mod.STDLIB_STRICT_ENV, "1")
    monkeypatch.setattr(parser_mod, "STDLIB_FILES", ["__definitely_missing_stage35_stdlib__.hx"])
    monkeypatch.setattr(sys, "argv", ["helixc.backend.ptx", str(src_path), "--stdlib"])
    monkeypatch.delitem(sys.modules, "helixc.backend.ptx", raising=False)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("helixc.backend.ptx", run_name="__main__")
    captured = capsys.readouterr()
    assert exc.value.code == 2, captured.out + captured.err
    assert "stdlib file missing" in captured.err
    assert "Traceback" not in captured.err


def test_stage35_direct_ptx_cli_reports_parse_error_without_traceback():
    proc = run_ptx_cli("@kernel fn k( { let i = thread_idx(); }\n")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "PARSE ERROR" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_c119_direct_ptx_cli_rejects_unsupported_hbm_float_dtype():
    src = """
    @kernel fn k(a: tile<f16, [256], HBM>) {
        let x = a[0];
        let y = x < 1.0_f16;
    }
    """
    proc = run_ptx_cli(src)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "@kernel HBM tile parameter dtype f16 is not supported" in proc.stderr
    assert "ld.global.f16" not in proc.stdout


def test_c119_direct_ptx_cli_rejects_unused_unsupported_hbm_dtype():
    proc = run_ptx_cli("@kernel fn k(a: tile<f16, [16], HBM>) { }\n")
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "@kernel HBM tile parameter dtype f16 is not supported" in proc.stderr
    assert ".visible .entry" not in proc.stdout


def test_c119_direct_ptx_cli_accepts_kernel_index_builtin():
    proc = run_ptx_cli("@kernel fn k() { let i = thread_idx(); }\n")
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "%tid.x" in proc.stdout


def test_c119_direct_ptx_cli_rejects_extern_only_kernels():
    proc = run_ptx_cli('@kernel extern "C" fn k(a: tile<f32, [16], HBM>);\n')
    assert proc.returncode != 0, proc.stdout + proc.stderr
    assert "PTX emission requires at least one @kernel function" in proc.stderr
    assert ".visible .entry" not in proc.stdout


def test_c119_emit_ptx_rejects_unsupported_kernel_ops():
    with pytest.raises(NotImplementedError, match="elem.div"):
        emit("@kernel fn k() { let z = 4 / 2; }")
    with pytest.raises(NotImplementedError, match="bit.not"):
        emit("@kernel fn k() { let z = ~1; }")
    with pytest.raises(RuntimeError, match="unsupported PTX float constant type f64"):
        emit("@kernel fn k() { let z = 1.0_f64; }")


def test_c119_ptx_scalar_ops_require_mapped_operands():
    a = ti.TileValue(100, tir.TIRScalar("i32"))
    b = ti.TileValue(101, tir.TIRScalar("i32"))
    out = ti.TileValue(102, tir.TIRScalar("i32"))
    block = ti.TileBlock(0, ops=[
        ti.TileOp(ti.TileOpKind.SCALAR_ADD, [a, b], [out])
    ])
    fn = ti.TileFn("k", [], tir.TIRUnit(), [block], attrs={"kernel": True})
    mod = ti.TileModule(functions={"k": fn})
    with pytest.raises(RuntimeError, match="missing PTX register for scalar add lhs"):
        emit_ptx(mod)


def test_c119_ptx_scalar_ops_reject_address_register_operands():
    from helixc.backend.ptx import PtxEmitter
    a = ti.TileValue(100, tir.TIRScalar("i32"))
    b = ti.TileValue(101, tir.TIRScalar("i32"))
    out = ti.TileValue(102, tir.TIRScalar("i32"))
    op = ti.TileOp(ti.TileOpKind.SCALAR_ADD, [a, b], [out])
    em = PtxEmitter()
    em.reg_map = {a.id: "%rd0", b.id: "%rd1"}
    with pytest.raises(RuntimeError, match="expected %r register class"):
        em.emit_op(op)


def test_c119_ptx_float_compare_uses_f32_setp():
    src = """
    @kernel fn k(a: tile<f32, [256], HBM>) {
        let x = a[0];
        let y = x < 1.0_f32;
    }
    """
    out = emit(src)
    assert "setp.lt.f32" in out
    assert "setp.lt.s32" not in out


def test_module_header():
    out = emit("@kernel fn k() {}")
    assert ".version" in out
    assert ".target sm_75" in out
    assert ".address_size 64" in out


def test_kernel_directive():
    out = emit("@kernel fn my_kernel() {}")
    assert ".visible .entry my_kernel" in out
    assert "{" in out and "}" in out


def test_kernel_has_register_declarations():
    out = emit("@kernel fn k() {}")
    assert ".reg .pred" in out
    assert ".reg .b32" in out
    assert ".reg .f32" in out


def test_kernel_ret():
    out = emit("@kernel fn k() {}")
    # Every kernel must end with ret;
    assert "ret;" in out


def test_scalar_const_int():
    src = "@kernel fn k() { let x = 42; }"
    out = emit(src)
    assert "mov.b32" in out
    assert "42" in out


def test_scalar_add():
    src = "@kernel fn k() { let x = 1; let y = 2; let z = x + y; }"
    out = emit(src)
    assert "add.s32" in out


def test_scalar_mul():
    src = "@kernel fn k() { let z = 3 * 4; }"
    out = emit(src)
    assert "mul.lo.s32" in out


def test_non_kernel_functions_are_not_stubbed():
    src = """
    fn helper() -> i32 { 42 }
    @kernel fn k() {}
    """
    out = emit(src)
    assert ".func" not in out
    assert ".visible .entry k" in out


# ============================================================================
# Stage 16 — GPU primitives end-to-end
# ============================================================================
def test_thread_idx_emits_tid_x():
    src = "@kernel fn k() { let i = thread_idx(); }"
    out = emit(src)
    assert "mov.u32" in out
    assert "%tid.x" in out


def test_thread_idx_outside_kernel_traps():
    # Trap-id 96001: thread_idx() outside @kernel.
    src = "fn main() -> i32 { let i = thread_idx(); 0 }"
    try:
        emit(src)
    except (SyntaxError, NotImplementedError) as e:
        assert "96001" in str(e) or "thread_idx" in str(e)
        return
    raise AssertionError("expected trap 96001 for thread_idx() outside kernel")


def test_hbm_tile_param_indexed_load_emits_ld_global_f32():
    src = """
    @kernel fn k(a: tile<f32, [256], HBM>) {
        let x = a[0];
    }
    """
    out = emit(src)
    assert "ld.param.u64" in out
    assert "cvta.to.global.u64" in out
    assert "ld.global.f32" in out


def test_hbm_tile_param_indexed_store_emits_st_global_f32():
    src = """
    @kernel fn k(a: tile<f32, [256], HBM>, b: tile<f32, [256], HBM>) {
        b[0] = a[0];
    }
    """
    out = emit(src)
    assert "ld.global.f32" in out
    assert "st.global.f32" in out


def test_vec_add_kernel_full_ptx():
    # The Stage 16 capstone: vec_add must produce a PTX kernel that:
    # - declares 3 .param .b64 entries
    # - reads %tid.x
    # - emits three ld.global.f32 sequences (a[i] + b[i] + result load for store)
    # - emits one add.f32
    # - emits one st.global.f32 to c
    src = """
    @kernel
    fn vec_add(a: tile<f32, [256], HBM>, b: tile<f32, [256], HBM>, c: tile<f32, [256], HBM>) {
        let i = thread_idx();
        c[i] = a[i] + b[i];
    }
    """
    out = emit(src)
    assert ".visible .entry vec_add" in out
    assert ".param .b64 param_0" in out
    assert ".param .b64 param_1" in out
    assert ".param .b64 param_2" in out
    assert "%tid.x" in out
    # Two HBM loads (a[i], b[i]) plus one HBM store (c[i] = ...).
    assert out.count("ld.global.f32") == 2
    assert out.count("st.global.f32") == 1
    assert "add.f32" in out
    # And the trapping `// TODO:` strings must not appear: every op was handled.
    assert "// TODO:" not in out


def test_per_prefix_register_counters():
    # %r and %f pools must be independent. Earlier shared `next_reg` would
    # produce stale labels like %r3 == %f3.
    src = """
    @kernel fn k(a: tile<f32, [16], HBM>) {
        let i = thread_idx();
        let x = a[i];
    }
    """
    out = emit(src)
    # %r0 reads tid; %f0 receives the ld.global.f32 result.
    assert "%r0" in out
    assert "%f0" in out


def test_thread_idx_y_and_z():
    src = """
    @kernel fn k() {
        let x = thread_idx();
        let y = thread_idx_y();
        let z = thread_idx_z();
    }
    """
    out = emit(src)
    assert "%tid.x" in out
    assert "%tid.y" in out
    assert "%tid.z" in out


def test_block_idx_and_block_dim():
    src = """
    @kernel fn k() {
        let bx = block_idx();
        let by = block_idx_y();
        let bdz = block_dim_z();
    }
    """
    out = emit(src)
    assert "%ctaid.x" in out
    assert "%ctaid.y" in out
    assert "%ntid.z" in out


def test_scalar_sub():
    out = emit("@kernel fn k() { let z = 10 - 3; }")
    assert "sub.s32" in out


def test_scalar_neg():
    out = emit("@kernel fn k() { let x = 5; let y = -x; }")
    assert "neg.s32" in out


def test_scalar_const_float():
    out = emit("@kernel fn k() { let x = 3.14; }")
    # Hex bit pattern of 3.14f rounded.
    assert "mov.f32" in out
    assert "0f" in out  # PTX hex-float prefix


def test_ptx_register_pool_overflow_raises():
    # Audit A3-MEDIUM-1 regression: per-prefix register pool overflow
    # used to silently emit references to undeclared registers (e.g.
    # %r33 when only %r<32> was declared). Now _new_reg raises
    # RuntimeError when the per-prefix counter exceeds _REG_POOL_CAP.
    from helixc.backend.ptx import PtxEmitter
    em = PtxEmitter()
    em.next_reg_by_prefix["r"] = PtxEmitter._REG_POOL_CAP
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match="register pool overflow"):
        em._new_reg("r")


def test_ptx_register_pool_cap_in_kernel_decl():
    # Audit A3-MEDIUM-1: bumped pool from 32 to 256 in declarations.
    out = emit("@kernel fn k() {}")
    assert ".reg .b32   %r<256>;" in out
    assert ".reg .f32   %f<256>;" in out
    assert ".reg .pred  %p<256>;" in out
    assert ".reg .b64   %rd<256>;" in out


def test_hbm_subtract_uses_sub_f32():
    src = """
    @kernel fn k(a: tile<f32, [16], HBM>, b: tile<f32, [16], HBM>) {
        let i = thread_idx();
        b[i] = a[i] - a[i];
    }
    """
    out = emit(src)
    assert "sub.f32" in out


def test_c20_1_isize_usize_treated_as_64_bit_in_ptx():
    """Audit 28.8 cycle 21 C20-1 (HIGH): PTX backend width-keyed tables
    must treat isize/usize as 64-bit, matching typecheck.py canon.

    Pre-fix `_DTYPE_SIZE.get("isize", 4)` returned 4, `_ptx_type_str`
    returned `.b32`, and `_ld_reg_prefix("isize")` returned `"r"` (32-bit
    pool) — silently 32-bit-narrowing isize values in PTX output."""
    from helixc.backend.ptx import PtxEmitter
    from helixc.ir import tir
    # Probe class-level tables directly.
    assert PtxEmitter._DTYPE_SIZE["isize"] == 8
    assert PtxEmitter._DTYPE_SIZE["usize"] == 8
    assert PtxEmitter._DTYPE_SIZE["i64"] == 8
    assert PtxEmitter._DTYPE_PTX_LOAD["isize"] == "s64"
    assert PtxEmitter._DTYPE_PTX_LOAD["usize"] == "u64"
    # _ptx_type_str via instance.
    em = PtxEmitter.__new__(PtxEmitter)  # bare instance (no __init__ side-effects)
    isize_ty = tir.TIRScalar(name="isize")
    usize_ty = tir.TIRScalar(name="usize")
    assert em._ptx_type_str(isize_ty) == ".b64"
    assert em._ptx_type_str(usize_ty) == ".b64"
    # _ld_reg_prefix — isize/usize should pick the 64-bit `rd` pool.
    assert em._ld_reg_prefix("isize") == "rd"
    assert em._ld_reg_prefix("usize") == "rd"
    assert em._ld_reg_prefix("i64") == "rd"
    assert em._ld_reg_prefix("i32") == "r"


def test_stage56_autotune_expand_produces_n_kernel_fns():
    """Stage 56 / Tier 2 #8: @autotune @kernel emits N specialized
    kernel FnDecls via autotune_expand. Verify by inspecting the
    expanded program for N kernel fns with mangled names.

    BLOCK_SIZE: [16, 32] → 2 variants → 2 @kernel FnDecls in the
    expanded program. (End-to-end PTX emission is exercised via
    test_stage56_autotune_expand_* in test_autotune.py.)
    """
    src = """
@autotune(BLOCK_SIZE: [16, 32])
@kernel fn vec_kernel(a: tile<f32, [16], HBM>) { }
"""
    from helixc.frontend.parser import parse as _parse
    from helixc.frontend.autotune_expand import expand_autotune_kernels
    from helixc.frontend import ast_nodes as A

    prog = _parse(src)
    prog = expand_autotune_kernels(prog)
    kernel_fns = [it for it in prog.items
                  if isinstance(it, A.FnDecl) and "kernel" in it.attrs]
    names = sorted(f.name for f in kernel_fns)
    assert names == [
        "vec_kernel__autotune_BLOCK_SIZE_16",
        "vec_kernel__autotune_BLOCK_SIZE_32",
    ], f"unexpected kernel names after expansion: {names}"
    # Each variant retains @kernel, drops @autotune, gains config attr.
    for f in kernel_fns:
        assert "kernel" in f.attrs
        assert "autotune" not in f.attrs
        assert any(a.startswith("autotune_config:BLOCK_SIZE=")
                   for a in f.attrs)


def test_stage64_inc1_bf16_hbm_tile_no_longer_rejected():
    """Stage 64 Inc 1 — Tier 2 #6: bf16 HBM tile elements lift
    from PTX rejection. Pre-Stage-64: emit() on a bf16 HBM tile
    raised RuntimeError 'unsupported PTX HBM tile dtype bf16'.
    Post-Stage-64: emit succeeds; .reg .b16 pool declared;
    bf16 ld/st suffixes appear in the output."""
    src = """
    @kernel fn k(a: tile<bf16, [16], HBM>) {}
    """
    out = emit(src)
    # Kernel preamble declares the .b16 register pool for bf16/f16.
    assert ".reg .b16   %h<" in out
    # Kernel directive emitted (basic sanity).
    assert ".visible .entry k" in out


def test_stage64_inc1_f16_hbm_tile_no_longer_rejected():
    """Stage 64 Inc 1: f16 also lifted (same pool as bf16)."""
    src = """
    @kernel fn k(a: tile<f16, [16], HBM>) {}
    """
    out = emit(src)
    assert ".reg .b16   %h<" in out
    assert ".visible .entry k" in out


def test_stage64_inc1_existing_f32_kernels_still_get_b16_pool():
    """Stage 64 Inc 1 regression guard: f32-only kernels still get
    the new .b16 register pool declaration (always-on, harmless
    overhead). Verifies the new pool declaration didn't break
    existing f32 kernels."""
    out = emit("@kernel fn k(a: tile<f32, [16], HBM>) {}")
    assert ".reg .b16   %h<" in out
    assert ".reg .f32   %f<" in out  # f32 pool still declared
    assert ".visible .entry k" in out


def test_stage64_inc1_unsupported_dtype_still_rejected():
    """Stage 64 Inc 1 negative: dtypes outside the now-expanded
    set (f32/i32/bf16/f16) still get rejected."""
    src = """
    @kernel fn k(a: tile<i64, [16], HBM>) {}
    """
    with pytest.raises((RuntimeError, ValueError)) as exc_info:
        emit(src)
    msg = str(exc_info.value)
    # Error message mentions the unsupported dtype OR an upstream
    # rejection at a different layer.
    assert "i64" in msg or "unsupported" in msg.lower()


def main():
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
