"""Tests for helixc.ir.tile_adjoint — Stage 120 (v2.1 Phase B.3.d).

End-to-end forward→backward kernel generation using the
Stage 117-119 TILE_OP_ADJOINTS table.
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from helixc.ir.tile_ir import (
    TileOp, TileBlock, TileFn, TileModule, TileOpKind,
)
from helixc.ir.tile_adjoint import (
    AdjointKernel,
    emit_adjoint_kernel,
    emit_adjoint_module,
)


def _make_kernel(name: str, ops: list[TileOp]) -> TileFn:
    return TileFn(
        name=name, params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=ops)],
        attrs={"kernel": True},
    )


def test_stage120_emit_adjoint_basic_add():
    """Stage 120 — TILE_ADD adjoint is identity (dz flows through to
    both dx and dy). emit_adjoint_kernel produces a backward kernel
    with one TILE_ADD placeholder op marked dispatch='identity'."""
    fwd = _make_kernel("add_k", [
        TileOp(kind=TileOpKind.TILE_ADD),
        TileOp(kind=TileOpKind.RETURN),
    ])
    result = emit_adjoint_kernel(fwd)
    assert isinstance(result, AdjointKernel)
    assert result.bwd_fn.name == "add_k__bwd"
    assert result.op_count_fwd == 2
    # 1 TILE_ADD placeholder (identity dispatch); RETURN is non-diff,
    # skipped.
    assert result.op_count_bwd == 1
    assert result.bwd_fn.blocks[0].ops[0].kind is TileOpKind.TILE_ADD
    assert result.bwd_fn.blocks[0].ops[0].attrs["dispatch"] == "identity"
    assert result.fallthrough_kinds == []


def test_stage120_emit_adjoint_matmul_3wmma_pattern():
    """Stage 120 — TILE_MATMUL adjoint emits the 3-wmma reverse pattern
    documented in Stage 117 (2 TILE_TRANSPOSE + 2 TILE_MATMUL)."""
    fwd = _make_kernel("matmul_k", [
        TileOp(kind=TileOpKind.TILE_MATMUL),
    ])
    result = emit_adjoint_kernel(fwd)
    op_kinds = [op.kind for op in result.bwd_fn.blocks[0].ops]
    assert op_kinds.count(TileOpKind.TILE_TRANSPOSE) == 2
    assert op_kinds.count(TileOpKind.TILE_MATMUL) == 2
    # Order: Bt → dA = dD @ Bt → At → dB = At @ dD
    assert op_kinds == [
        TileOpKind.TILE_TRANSPOSE,
        TileOpKind.TILE_MATMUL,
        TileOpKind.TILE_TRANSPOSE,
        TileOpKind.TILE_MATMUL,
    ]


def test_stage120_emit_adjoint_reverse_order():
    """Stage 120 — forward ops are differentiated in REVERSE order.

    Forward: TILE_TRANSPOSE → TILE_ADD → TILE_MUL
    Backward (reversed): TILE_MUL adjoint → TILE_ADD adjoint → TILE_TRANSPOSE adjoint
    """
    fwd = _make_kernel("k", [
        TileOp(kind=TileOpKind.TILE_TRANSPOSE),
        TileOp(kind=TileOpKind.TILE_ADD),
        TileOp(kind=TileOpKind.TILE_MUL),
    ])
    result = emit_adjoint_kernel(fwd)
    op_kinds = [op.kind for op in result.bwd_fn.blocks[0].ops]
    # TILE_MUL adjoint: 2 TILE_MULs.
    # TILE_ADD adjoint: 1 TILE_ADD identity placeholder.
    # TILE_TRANSPOSE adjoint: 1 TILE_TRANSPOSE.
    # Total: 4 ops; order respects reverse forward-walk.
    assert op_kinds == [
        TileOpKind.TILE_MUL,
        TileOpKind.TILE_MUL,
        TileOpKind.TILE_ADD,
        TileOpKind.TILE_TRANSPOSE,
    ]


def test_stage120_emit_adjoint_skips_non_differentiable():
    """Stage 120 — non-differentiable forward ops (RETURN, THREAD_IDX,
    TILE_LOAD_GLOBAL, etc.) are silently skipped during the reverse
    walk. They have no adjoint contribution."""
    fwd = _make_kernel("k", [
        TileOp(kind=TileOpKind.THREAD_IDX),
        TileOp(kind=TileOpKind.TILE_LOAD_GLOBAL),
        TileOp(kind=TileOpKind.TILE_ADD),
        TileOp(kind=TileOpKind.TILE_STORE_GLOBAL),
        TileOp(kind=TileOpKind.RETURN),
    ])
    result = emit_adjoint_kernel(fwd)
    # Only TILE_ADD has an adjoint; the rest are TILE_OP_NON_DIFFERENTIABLE.
    assert result.op_count_bwd == 1
    assert result.bwd_fn.blocks[0].ops[0].kind is TileOpKind.TILE_ADD


def test_stage120_emit_adjoint_rejects_non_kernel():
    """Stage 120 — host-only fns (no @kernel attr) are rejected.
    Reverse-mode AD only makes sense for kernels."""
    fwd = TileFn(
        name="host_fn", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[TileOp(kind=TileOpKind.TILE_ADD)])],
        attrs={},  # no kernel attr
    )
    with pytest.raises(ValueError, match="not @kernel"):
        emit_adjoint_kernel(fwd)


def test_stage120_emit_adjoint_rejects_multi_block_control_flow():
    """Stage 120 — kernels with >1 block (CFG with branches) need
    tape-style intermediate storage; substrate rejects."""
    fwd = TileFn(
        name="k", params=[], return_ty=None,
        blocks=[
            TileBlock(id=0, ops=[TileOp(kind=TileOpKind.TILE_ADD)]),
            TileBlock(id=1, ops=[TileOp(kind=TileOpKind.RETURN)]),
        ],
        attrs={"kernel": True},
    )
    with pytest.raises(NotImplementedError, match="control flow"):
        emit_adjoint_kernel(fwd)


def test_stage120_emit_adjoint_rejects_empty_kernel():
    """Stage 120 — empty kernel has nothing to differentiate."""
    fwd = TileFn(
        name="k", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[])],
        attrs={"kernel": True},
    )
    with pytest.raises(ValueError, match="no ops"):
        emit_adjoint_kernel(fwd)


def test_stage120_adjoint_inherits_kernel_attrs():
    """Stage 120 — backward kernel inherits forward's attrs + adds
    is_adjoint_of=<fwd_name> for round-trip provenance."""
    fwd = _make_kernel("loss_k", [TileOp(kind=TileOpKind.TILE_ADD)])
    result = emit_adjoint_kernel(fwd)
    assert result.bwd_fn.attrs["kernel"] is True
    assert result.bwd_fn.attrs["is_adjoint_of"] == "loss_k"


def test_stage120_emit_adjoint_module_skips_non_kernel():
    """Stage 120 — emit_adjoint_module produces a dict mapping
    forward-fn-name → AdjointKernel for every @kernel fn (skipping
    host-only fns + extern + already-adjoint fns)."""
    mod = TileModule()
    mod.functions["k"] = _make_kernel("k", [TileOp(kind=TileOpKind.TILE_ADD)])
    # Host-only fn — should be skipped.
    mod.functions["host"] = TileFn(
        name="host", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[TileOp(kind=TileOpKind.TILE_ADD)])],
        attrs={},
    )
    # Already an adjoint — should be skipped (don't double-differentiate).
    mod.functions["k__bwd"] = TileFn(
        name="k__bwd", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[TileOp(kind=TileOpKind.TILE_ADD)])],
        attrs={"kernel": True, "is_adjoint_of": "k"},
    )
    out = emit_adjoint_module(mod)
    assert "k" in out
    assert "host" not in out
    assert "k__bwd" not in out


def test_stage120_emit_adjoint_module_skips_uncovered():
    """Stage 120 — emit_adjoint_module silently skips kernels that
    raise (e.g., multi-block control flow or empty body) rather than
    propagating the exception. Caller can detect by checking which
    forward fns produced no entry."""
    mod = TileModule()
    mod.functions["good"] = _make_kernel("good", [TileOp(kind=TileOpKind.TILE_ADD)])
    mod.functions["bad_empty"] = TileFn(
        name="bad_empty", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[])],
        attrs={"kernel": True},
    )
    out = emit_adjoint_module(mod)
    assert "good" in out
    assert "bad_empty" not in out


def test_stage120_fallthrough_tracking():
    """Stage 120 — kinds not in TILE_OP_ADJOINTS AND not in
    TILE_OP_NON_DIFFERENTIABLE land in result.fallthrough_kinds.

    With the v2.1 audit-fix table this list should be empty for all
    documented kinds. We test by injecting a synthetic non-enum-member
    (None) — actually we can't easily without breaking the dataclass,
    so this is a structural test: verify the field exists and starts
    empty for a covered kernel.
    """
    fwd = _make_kernel("k", [TileOp(kind=TileOpKind.TILE_ADD)])
    result = emit_adjoint_kernel(fwd)
    assert result.fallthrough_kinds == []
