"""Stage 107 / Stage 64 Inc 5 — tile-IR optimization passes tests.

v1.0 minimum-viable scaffold coverage:
  - dead_tile_elim drops unused non-side-effecting ops
  - dead_tile_elim retains side-effecting ops (stores, barriers)
  - dead_tile_elim is idempotent (fixpoint check)
  - redundant_zero_coalesce dedups same-shape TILE_ZEROS
  - redundant_zero_coalesce rewires operand uses
  - redundant_zero_coalesce preserves different-shape TILE_ZEROS
  - register_reuse_hints reports last-use positions
  - run_all_passes composition order works (DCE + coalesce both fire)

Tests use the same TileFn / TileBlock / TileOp / TileValue / TIRTileTy
shape that helixc/backend/ptx.py consumes — so passes that succeed
here produce IR the emitter accepts.
"""

from __future__ import annotations

from dataclasses import replace
from helixc.ir import tile_ir as ti
from helixc.ir import tir
from helixc.ir.passes import tile_opt


def _make_zeros(tid: int, dtype: str, length: int,
                memspace: str = "REG") -> tuple[ti.TileValue, ti.TileOp]:
    """Helper: build a (TileValue, TILE_ZEROS op) pair with the
    given id, dtype, length, memspace."""
    val = ti.TileValue(tid, tir.TIRTileTy(
        tir.TIRScalar(dtype), (tir.DimConst(length),), memspace
    ))
    op = ti.TileOp(ti.TileOpKind.TILE_ZEROS, [], [val],
                   attrs={"dtype": dtype, "length": length})
    return val, op


def _make_fn(ops: list[ti.TileOp]) -> ti.TileFn:
    """Helper: wrap ops in a single-block TileFn."""
    blk = ti.TileBlock(0, params=[], ops=ops)
    return ti.TileFn(
        name="test_fn",
        params=[],
        return_ty=tir.TIRScalar("i32"),
        blocks=[blk],
    )


def test_stage107_dead_tile_elim_drops_unused_zeros():
    """Stage 107 — TILE_ZEROS with no user gets dropped by DCE."""
    _val, op = _make_zeros(0, "f32", 4)
    fn = _make_fn([op])
    out = tile_opt.dead_tile_elim(fn)
    assert len(out.blocks[0].ops) == 0, (
        f"unused TILE_ZEROS should drop; got "
        f"{[o.kind for o in out.blocks[0].ops]}")


def test_stage107_dead_tile_elim_keeps_used_zeros():
    """Stage 107 — TILE_ZEROS that feeds a downstream TILE_ADD is
    NOT dropped."""
    lhs, lhs_op = _make_zeros(0, "f32", 4)
    rhs, rhs_op = _make_zeros(1, "f32", 4)
    sum_val = ti.TileValue(2, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(4),), "REG"
    ))
    add_op = ti.TileOp(
        ti.TileOpKind.TILE_ADD, [lhs, rhs], [sum_val])
    # sum_val is unused, so add_op itself would be dead — wrap in a
    # side-effecting store that uses sum_val to keep it alive.
    store_op = ti.TileOp(
        ti.TileOpKind.TILE_STORE_GLOBAL, [sum_val], [])
    fn = _make_fn([lhs_op, rhs_op, add_op, store_op])
    out = tile_opt.dead_tile_elim(fn)
    kinds = [o.kind for o in out.blocks[0].ops]
    assert ti.TileOpKind.TILE_ZEROS in kinds, kinds
    assert ti.TileOpKind.TILE_ADD in kinds, kinds
    assert ti.TileOpKind.TILE_STORE_GLOBAL in kinds, kinds


def test_stage107_dead_tile_elim_keeps_side_effecting_ops():
    """Stage 107 — TILE_STORE_GLOBAL has no results but has side
    effects; DCE must NOT drop it."""
    val, zeros_op = _make_zeros(0, "f32", 4)
    store_op = ti.TileOp(
        ti.TileOpKind.TILE_STORE_GLOBAL, [val], [])
    fn = _make_fn([zeros_op, store_op])
    out = tile_opt.dead_tile_elim(fn)
    kinds = [o.kind for o in out.blocks[0].ops]
    assert ti.TileOpKind.TILE_STORE_GLOBAL in kinds, kinds
    # ZEROS is kept because STORE uses it.
    assert ti.TileOpKind.TILE_ZEROS in kinds, kinds


def test_stage107_dead_tile_elim_cascades_transitively():
    """Stage 107 — DCE iterates to fixpoint. Two stacked TILE_ZEROS
    + TILE_ADD chain whose final result is unused: all 3 ops drop
    in one pass (transitive cascade)."""
    a, a_op = _make_zeros(0, "f32", 4)
    b, b_op = _make_zeros(1, "f32", 4)
    s = ti.TileValue(2, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(4),), "REG"
    ))
    add_op = ti.TileOp(ti.TileOpKind.TILE_ADD, [a, b], [s])
    fn = _make_fn([a_op, b_op, add_op])
    out = tile_opt.dead_tile_elim(fn)
    assert len(out.blocks[0].ops) == 0, (
        f"transitively-dead chain should fully drop; got "
        f"{[o.kind for o in out.blocks[0].ops]}")


def test_stage107_dead_tile_elim_idempotent():
    """Stage 107 — DCE applied twice produces the same result as
    once (fixpoint reached on first call)."""
    a, a_op = _make_zeros(0, "f32", 4)
    fn = _make_fn([a_op])
    once = tile_opt.dead_tile_elim(fn)
    twice = tile_opt.dead_tile_elim(once)
    assert len(once.blocks[0].ops) == len(twice.blocks[0].ops)


def test_stage107_zero_coalesce_dedups_same_shape():
    """Stage 107 — two TILE_ZEROS with same dtype/shape/memspace
    coalesce to one; downstream uses of the dropped tile rewire
    to the kept tile."""
    a, a_op = _make_zeros(0, "f32", 4)
    b, b_op = _make_zeros(1, "f32", 4)
    # Use b in a store so it doesn't get DCE'd.
    store_op = ti.TileOp(ti.TileOpKind.TILE_STORE_GLOBAL, [b], [])
    fn = _make_fn([a_op, b_op, store_op])
    out = tile_opt.redundant_zero_coalesce(fn)
    zeros_ops = [o for o in out.blocks[0].ops
                 if o.kind == ti.TileOpKind.TILE_ZEROS]
    assert len(zeros_ops) == 1, (
        f"redundant TILE_ZEROS should coalesce to 1; got "
        f"{len(zeros_ops)}")
    # Store should now reference `a` (the kept tile), not `b`.
    store = [o for o in out.blocks[0].ops
             if o.kind == ti.TileOpKind.TILE_STORE_GLOBAL][0]
    assert store.operands[0].id == a.id, (
        f"operand rewire failed; store still references "
        f"id={store.operands[0].id} (expected a.id={a.id})")


def test_stage107_zero_coalesce_preserves_different_shapes():
    """Stage 107 — TILE_ZEROS with different lengths / dtypes /
    memspaces are NOT coalesced (they're semantically distinct)."""
    a, a_op = _make_zeros(0, "f32", 4)  # length 4
    b, b_op = _make_zeros(1, "f32", 8)  # length 8
    c, c_op = _make_zeros(2, "i32", 4)  # different dtype
    sa = ti.TileOp(ti.TileOpKind.TILE_STORE_GLOBAL, [a], [])
    sb = ti.TileOp(ti.TileOpKind.TILE_STORE_GLOBAL, [b], [])
    sc = ti.TileOp(ti.TileOpKind.TILE_STORE_GLOBAL, [c], [])
    fn = _make_fn([a_op, b_op, c_op, sa, sb, sc])
    out = tile_opt.redundant_zero_coalesce(fn)
    zeros_ops = [o for o in out.blocks[0].ops
                 if o.kind == ti.TileOpKind.TILE_ZEROS]
    assert len(zeros_ops) == 3, (
        f"distinct-shape TILE_ZEROS must NOT coalesce; got "
        f"{len(zeros_ops)} (expected 3)")


def test_stage107_register_reuse_hints_reports_last_use():
    """Stage 107 — analysis returns {tile_id: (block_idx, op_idx)}
    for the LAST use of each TileValue."""
    a, a_op = _make_zeros(0, "f32", 4)
    b, b_op = _make_zeros(1, "f32", 4)
    s = ti.TileValue(2, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(4),), "REG"
    ))
    add_op = ti.TileOp(ti.TileOpKind.TILE_ADD, [a, b], [s])
    store_op = ti.TileOp(ti.TileOpKind.TILE_STORE_GLOBAL, [s], [])
    fn = _make_fn([a_op, b_op, add_op, store_op])
    hints = tile_opt.register_reuse_hints(fn)
    # a is last used at add_op (idx 2).
    assert hints[a.id] == (0, 2), (
        f"a.id={a.id} last-use expected (0, 2); got {hints[a.id]}")
    # b same.
    assert hints[b.id] == (0, 2), (
        f"b.id={b.id} last-use expected (0, 2); got {hints[b.id]}")
    # s last used at store_op (idx 3).
    assert hints[s.id] == (0, 3), (
        f"s.id={s.id} last-use expected (0, 3); got {hints[s.id]}")


def test_stage107_run_all_passes_composes_dce_and_coalesce():
    """Stage 107 — `run_all_passes` composes dead-tile-elim with
    redundant-zero coalescing. The input has BOTH a genuinely dead op
    (DCE must drop it) and two redundant *live* TILE_ZEROS (coalesce
    must dedup them and rewire the second tile's store onto the
    first). Both passes are independently load-bearing here: drop DCE
    from the pipeline and the dead ADD survives; drop coalesce and a
    second TILE_ZEROS survives — either way the assertions fail."""
    a, a_op = _make_zeros(0, "f32", 4)
    b, b_op = _make_zeros(1, "f32", 4)  # redundant with a (same shape)
    # Stores keep BOTH zeros alive past the first DCE, so coalesce has
    # real work to do.
    store_a = ti.TileOp(ti.TileOpKind.TILE_STORE_GLOBAL, [a], [])
    store_b = ti.TileOp(ti.TileOpKind.TILE_STORE_GLOBAL, [b], [])
    # A genuinely dead op: its ADD result has no consumer.
    dead = ti.TileValue(2, tir.TIRTileTy(
        tir.TIRScalar("f32"), (tir.DimConst(4),), "REG"
    ))
    dead_add = ti.TileOp(ti.TileOpKind.TILE_ADD, [a, a], [dead])
    fn = _make_fn([a_op, b_op, dead_add, store_a, store_b])
    out = tile_opt.run_all_passes(fn)
    ops = out.blocks[0].ops
    # DCE drops dead_add; coalesce drops b_op -> one ZEROS + two stores.
    kinds = [o.kind for o in ops]
    assert kinds == [
        ti.TileOpKind.TILE_ZEROS,
        ti.TileOpKind.TILE_STORE_GLOBAL,
        ti.TileOpKind.TILE_STORE_GLOBAL,
    ], f"expected DCE+coalesce composition; got {kinds}"
    # Coalesce must rewire BOTH stores onto the surviving canonical `a`.
    stores = [o for o in ops if o.kind == ti.TileOpKind.TILE_STORE_GLOBAL]
    assert all(s.operands[0].id == a.id for s in stores), (
        f"coalesce should rewire both stores to a.id={a.id}; got "
        f"{[s.operands[0].id for s in stores]}")
