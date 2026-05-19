"""Tests for helixc.ir.tile_adjoint — Stage 120 (v2.1 Phase B.3.d) + R1 audit-fix.

End-to-end forward→backward kernel-shell generation using the
Stage 117-119 TILE_OP_ADJOINTS table.
"""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from helixc.ir import tile_ir as ti
from helixc.ir.tile_ir import (
    TileOp, TileBlock, TileFn, TileModule, TileOpKind, AdjointRecord,
)
from helixc.ir.tile_adjoint import (
    AdjointKernel,
    AdjointModule,
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
    """Stage 120 R1 — TILE_ADD adjoint is identity (gradient flows
    through). dispatch='identity' is encoded as ops=() in the
    canonical table, so the dispatcher emits ZERO backward ops for
    a kernel whose only differentiable forward op is a TILE_ADD."""
    fwd = _make_kernel("add_k", [
        TileOp(kind=TileOpKind.TILE_ADD),
        TileOp(kind=TileOpKind.RETURN),
    ])
    result = emit_adjoint_kernel(fwd)
    assert isinstance(result, AdjointKernel)
    assert result.bwd_fn.name == "add_k__bwd"
    assert result.op_count_fwd == 2
    # TILE_ADD identity dispatch emits zero ops; RETURN is non-diff,
    # also zero ops. Total backward: 0.
    assert result.op_count_bwd == 0
    assert result.bwd_fn.blocks[0].ops == []
    assert result.fallthrough_kinds == []
    assert result.complete is True


def test_stage120_emit_adjoint_matmul_reverse_pattern():
    """Stage 120 — TILE_MATMUL adjoint emits the reverse pattern from
    the canonical table: 2 TILE_TRANSPOSE + 2 TILE_MATMUL for the
    D = A @ B + C / dA = dD @ B^T / dB = A^T @ dD lowering."""
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
    """Stage 120 R1 — forward ops are differentiated in REVERSE order.

    Forward: TILE_TRANSPOSE → TILE_ADD → TILE_MUL
    Backward (reversed): TILE_MUL adjoint (2 ops) → TILE_ADD adjoint
    (0 ops, identity) → TILE_TRANSPOSE adjoint (1 op).
    """
    fwd = _make_kernel("k", [
        TileOp(kind=TileOpKind.TILE_TRANSPOSE),
        TileOp(kind=TileOpKind.TILE_ADD),
        TileOp(kind=TileOpKind.TILE_MUL),
    ])
    result = emit_adjoint_kernel(fwd)
    op_kinds = [op.kind for op in result.bwd_fn.blocks[0].ops]
    # TILE_MUL adjoint: 2 TILE_MULs (dx = dz*y, dy = dz*x).
    # TILE_ADD adjoint: identity → 0 ops.
    # TILE_TRANSPOSE adjoint: 1 TILE_TRANSPOSE.
    assert op_kinds == [
        TileOpKind.TILE_MUL,
        TileOpKind.TILE_MUL,
        TileOpKind.TILE_TRANSPOSE,
    ]


def test_stage120_emit_adjoint_reduce_kind_propagates_attr():
    """Stage 120 R1 audit-fix C3 — TILE_REDUCE adjoint dispatch must
    copy the forward op's `reduce_kind` attr into the backward shell
    so downstream lowering can pick broadcast (sum) vs scatter (max).
    Previously this attr was dropped, making the backward op
    indistinguishable across reduce kinds."""
    for kind in ("sum", "max", "min"):
        fwd = _make_kernel("reduce_k", [
            TileOp(kind=TileOpKind.TILE_REDUCE, attrs={"reduce_kind": kind}),
        ])
        result = emit_adjoint_kernel(fwd)
        assert result.op_count_bwd == 1
        bwd_op = result.bwd_fn.blocks[0].ops[0]
        assert bwd_op.kind is TileOpKind.TILE_REDUCE
        assert bwd_op.attrs["dispatch"] == "reduce_kind"
        assert bwd_op.attrs["adjoint_of"] == "TILE_REDUCE"
        # The audit-fix: forward's reduce_kind survives to the backward.
        assert bwd_op.attrs["reduce_kind"] == kind


def test_stage120_emit_adjoint_skips_non_differentiable():
    """Stage 120 — non-differentiable forward ops (RETURN, THREAD_IDX,
    TILE_LOAD_GLOBAL, etc.) are silently skipped during the reverse
    walk. They have no adjoint contribution."""
    fwd = _make_kernel("k", [
        TileOp(kind=TileOpKind.THREAD_IDX),
        TileOp(kind=TileOpKind.TILE_LOAD_GLOBAL),
        TileOp(kind=TileOpKind.TILE_TRANSPOSE),  # one diff op for signal
        TileOp(kind=TileOpKind.TILE_STORE_GLOBAL),
        TileOp(kind=TileOpKind.RETURN),
    ])
    result = emit_adjoint_kernel(fwd)
    # Only TILE_TRANSPOSE has an adjoint (1 op); the rest are non-diff.
    assert result.op_count_bwd == 1
    assert result.bwd_fn.blocks[0].ops[0].kind is TileOpKind.TILE_TRANSPOSE
    assert result.complete is True


def test_stage120_emit_adjoint_rejects_non_kernel():
    """Stage 120 — host-only fns (no @kernel attr) are rejected."""
    fwd = TileFn(
        name="host_fn", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[TileOp(kind=TileOpKind.TILE_ADD)])],
        attrs={},  # no kernel attr
    )
    with pytest.raises(ValueError, match="not @kernel"):
        emit_adjoint_kernel(fwd)


def test_stage120_emit_adjoint_rejects_multi_block_control_flow():
    """Stage 120 — kernels with >1 block need tape-style intermediate
    storage; substrate rejects."""
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
    fwd = _make_kernel("loss_k", [TileOp(kind=TileOpKind.TILE_TRANSPOSE)])
    result = emit_adjoint_kernel(fwd)
    assert result.bwd_fn.attrs["kernel"] is True
    assert result.bwd_fn.attrs["is_adjoint_of"] == "loss_k"


def test_stage120_emit_adjoint_module_returns_adjointmodule():
    """Stage 120 R1 audit-fix — emit_adjoint_module returns
    AdjointModule with kernels + skipped dicts. Non-kernel, extern,
    and already-adjoint fns appear in skipped with explicit reasons."""
    mod = TileModule()
    mod.functions["k"] = _make_kernel("k", [TileOp(kind=TileOpKind.TILE_TRANSPOSE)])
    mod.functions["host"] = TileFn(
        name="host", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[TileOp(kind=TileOpKind.TILE_ADD)])],
        attrs={},
    )
    mod.functions["ext"] = TileFn(
        name="ext", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[])],
        attrs={"kernel": True, "is_extern": True},
    )
    mod.functions["k__bwd"] = TileFn(
        name="k__bwd", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[TileOp(kind=TileOpKind.TILE_ADD)])],
        attrs={"kernel": True, "is_adjoint_of": "k"},
    )
    result = emit_adjoint_module(mod)
    assert isinstance(result, AdjointModule)
    assert "k" in result.kernels
    assert "host" in result.skipped
    assert result.skipped["host"] == "non-kernel"
    assert result.skipped["ext"] == "extern"
    assert result.skipped["k__bwd"] == "already-adjoint"
    assert result.total_seen == 4


def test_stage120_emit_adjoint_module_skipped_records_exception_reasons():
    """Stage 120 R1 audit-fix — kernels that raise during diff
    (NotImplementedError for control flow, ValueError for empty)
    land in skipped with the exception class + message, not
    silently dropped."""
    mod = TileModule()
    mod.functions["good"] = _make_kernel("good", [TileOp(kind=TileOpKind.TILE_TRANSPOSE)])
    mod.functions["bad_empty"] = TileFn(
        name="bad_empty", params=[], return_ty=None,
        blocks=[TileBlock(id=0, ops=[])],
        attrs={"kernel": True},
    )
    mod.functions["bad_cfg"] = TileFn(
        name="bad_cfg", params=[], return_ty=None,
        blocks=[
            TileBlock(id=0, ops=[TileOp(kind=TileOpKind.TILE_ADD)]),
            TileBlock(id=1, ops=[TileOp(kind=TileOpKind.RETURN)]),
        ],
        attrs={"kernel": True},
    )
    result = emit_adjoint_module(mod)
    assert "good" in result.kernels
    assert result.skipped["bad_empty"].startswith("ValueError")
    assert "no ops" in result.skipped["bad_empty"]
    assert result.skipped["bad_cfg"].startswith("NotImplementedError")
    assert "control flow" in result.skipped["bad_cfg"]


def test_stage120_fallthrough_marks_kernel_incomplete():
    """Stage 120 R1 audit-fix — when a forward op-kind is missing
    from BOTH TILE_OP_ADJOINTS and TILE_OP_NON_DIFFERENTIABLE, it
    lands in fallthrough_kinds and the kernel reports `complete=False`.

    Exercises the fallthrough branch by temporarily removing TILE_ADD
    from both the adjoint table and the non-diff set.
    """
    # Snapshot.
    saved_adj = ti.TILE_OP_ADJOINTS.get(TileOpKind.TILE_ADD)
    saved_inner = dict(ti._TILE_OP_ADJOINTS_INNER)
    try:
        # Remove TILE_ADD entirely so it's neither diff nor non-diff.
        # (TILE_ADD is not in TILE_OP_NON_DIFFERENTIABLE — verified at
        # module load via the partitioning test in test_tile_ir.)
        del ti._TILE_OP_ADJOINTS_INNER[TileOpKind.TILE_ADD]
        fwd = _make_kernel("k", [TileOp(kind=TileOpKind.TILE_ADD)])
        result = emit_adjoint_kernel(fwd)
        assert TileOpKind.TILE_ADD in result.fallthrough_kinds
        assert result.complete is False
        # No backward op was synthesized for the uncovered kind.
        assert result.op_count_bwd == 0
    finally:
        ti._TILE_OP_ADJOINTS_INNER.clear()
        ti._TILE_OP_ADJOINTS_INNER.update(saved_inner)
        # Sanity: TILE_ADD restored.
        assert ti.TILE_OP_ADJOINTS.get(TileOpKind.TILE_ADD) is saved_adj


def test_stage120_complete_property_on_clean_kernel():
    """Stage 120 R1 audit-fix — happy path: fully-covered kernel has
    complete=True and fallthrough_kinds=[]."""
    fwd = _make_kernel("k", [TileOp(kind=TileOpKind.TILE_TRANSPOSE)])
    result = emit_adjoint_kernel(fwd)
    assert result.fallthrough_kinds == []
    assert result.complete is True


def test_stage120_adjointrecord_rejects_explicit_with_empty_ops():
    """Stage 120 R1 audit-fix SF2 — AdjointRecord.__post_init__
    rejects dispatch='explicit' with ops=() at construction time.
    Previously such a record would silently emit zero backward ops
    and look like a successful adjoint."""
    with pytest.raises(ValueError, match="dispatch='explicit' requires"):
        AdjointRecord(
            inputs=("x",), outputs=("dx",),
            ops=(), dispatch="explicit",
        )


def test_stage120_adjointrecord_rejects_identity_with_nonempty_ops():
    """Stage 120 R1 audit-fix SF2 — AdjointRecord.__post_init__
    rejects dispatch='identity' with non-empty ops; the recorded ops
    would never be emitted and the inconsistency would mislead
    readers about the actual gradient computation."""
    with pytest.raises(ValueError, match="dispatch='identity' must have"):
        AdjointRecord(
            inputs=("x",), outputs=("dx",),
            ops=((TileOpKind.TILE_ADD, "stray"),),
            dispatch="identity",
        )


def test_stage120_r2_emit_rejects_reduce_without_kind_attr():
    """Stage 120 R2 audit-fix Finding 1 — a forward TILE_REDUCE whose
    `reduce_kind` discriminator attr is missing would silently emit
    `attrs['reduce_kind'] = None` into the backward shell, defeating
    R1 C3's whole point. The emitter must raise instead."""
    fwd = _make_kernel("reduce_k", [
        TileOp(kind=TileOpKind.TILE_REDUCE),  # no attrs at all
    ])
    with pytest.raises(ValueError, match="lacks the 'reduce_kind' attr"):
        emit_adjoint_kernel(fwd)


def test_stage120_r2_emit_rejects_reduce_with_misspelled_kind_attr():
    """Stage 120 R2 audit-fix Finding 1 — same protection covers the
    typo case: forward op carries a misspelled discriminator key, so
    `attrs.get('reduce_kind')` still returns None."""
    fwd = _make_kernel("reduce_k", [
        TileOp(kind=TileOpKind.TILE_REDUCE, attrs={"kind": "sum"}),
    ])
    with pytest.raises(ValueError, match="lacks the 'reduce_kind' attr"):
        emit_adjoint_kernel(fwd)


def test_stage120_r2_adjointrecord_rejects_runtime_keyed_with_ops():
    """Stage 120 R2 audit-fix Finding 2 — AdjointRecord.__post_init__
    must reject runtime-keyed dispatch (anything not 'explicit' or
    'identity') paired with non-empty `ops`. The consumer reads the
    forward op's attr and ignores `ops` entirely, so the recorded
    ops would be silently dropped."""
    with pytest.raises(ValueError, match="runtime-keyed"):
        AdjointRecord(
            inputs=("x",), outputs=("dx",),
            ops=((TileOpKind.TILE_ADD, "stray"),),
            dispatch="reduce_kind",
        )
    # Also catches future named-attr dispatches a maintainer might add.
    with pytest.raises(ValueError, match="runtime-keyed"):
        AdjointRecord(
            inputs=("x",), outputs=("dx",),
            ops=((TileOpKind.TILE_MUL, "stray"),),
            dispatch="future_keyed_dispatch",
        )


def test_stage120_r2_adjointmodule_rejects_overlap():
    """Stage 120 R2 audit-fix — AdjointModule.__post_init__ enforces
    the partition the docstring promises: a fn name cannot appear in
    both `kernels` and `skipped`. Today's producer maintains this by
    construction; the type-level check defends against a future
    refactor that introduces a partial-success path."""
    fn = _make_kernel("k", [TileOp(kind=TileOpKind.TILE_TRANSPOSE)])
    good = emit_adjoint_kernel(fn)
    with pytest.raises(ValueError, match="appear in both kernels and skipped"):
        AdjointModule(
            kernels={"k": good},
            skipped={"k": "non-kernel"},
        )
