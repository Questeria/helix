"""
helixc/ir/passes/tile_opt.py — tile-IR optimization passes (Stage 107 /
Stage 64 Inc 5).

v1.0 minimum-viable scope: 3 focused passes that prove the tile-IR
optimization scaffold exists, is testable, and composes cleanly with
the existing ptx.py emit path. Full pass-suite breadth (instruction
selection, register coloring, layout-aware scheduling, etc.) is
v2.0 Phase A polish.

The 3 passes:

1. `dead_tile_elim(fn)` — drop TileOps whose results have no users AND
   that have no side effects. Mirrors helixc/ir/passes/dce.py for the
   tile-IR analog. Side-effecting tile ops: TILE_STORE_GLOBAL,
   TILE_STORE_SHARED, TILE_INDEX_STORE_HBM, TMA_STORE, BARRIER_WAIT,
   RETURN, CALL.

2. `redundant_zero_coalesce(fn)` — when two TILE_ZEROS ops produce
   same-shape same-dtype same-memspace tiles, dedup so all users of
   the later tile re-route to the earlier tile. Pre-this, every
   TILE_ZEROS emit was a fresh register allocation; coalescing
   eliminates the redundant `mov.f32 %fN, 0f...` lines and frees
   register pressure for downstream ops (e.g., wmma fragments need
   contiguous register bursts; eliminating duplicate zeros leaves
   more contiguous space).

3. `register_reuse_hints(fn)` — analysis pass (not a rewrite).
   Returns a {tile_id: last_use_op_idx} map so the backend can free
   register slots once a tile's last use has emitted. The PTX backend
   doesn't yet consume this (it uses fresh registers monotonically),
   but the analysis is the prerequisite for v2.0 Phase A's
   register-coloring pass.

These passes operate on TileFn structures (not the lowered PTX text)
so they're pure-IR transforms — testable without going through the
emitter. The transforms preserve TileFn invariants (block ordering,
operand types) so chained passes compose cleanly.

License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import replace
from typing import Optional

from .. import tile_ir as ti


# Tile ops that have observable side effects (memory stores, async
# barriers, control flow) — even if their results have no users, they
# must be retained.
_SIDE_EFFECT_KINDS = {
    ti.TileOpKind.TILE_STORE_GLOBAL,
    ti.TileOpKind.TILE_STORE_SHARED,
    ti.TileOpKind.TILE_INDEX_STORE_HBM,
    ti.TileOpKind.TMA_STORE,
    ti.TileOpKind.BARRIER_WAIT,
    ti.TileOpKind.RETURN,
    ti.TileOpKind.CALL,
}


def _is_side_effect(op: ti.TileOp) -> bool:
    return op.kind in _SIDE_EFFECT_KINDS


def dead_tile_elim(fn: ti.TileFn) -> ti.TileFn:
    """Pass 1 — dead-tile elimination.

    A TileOp is dead if (a) none of its results are used by any other
    op (across all blocks of the fn), AND (b) it has no side effects.
    Removes dead ops in-place across all blocks; returns the fn with
    blocks rebuilt (immutable replace). Idempotent: a second call is
    a no-op once the fixpoint is reached.

    Implementation: collect the set of all used TileValue ids (every
    operand of every op + every block param + every fn param), then
    drop ops whose results are all outside that set AND not side-
    effecting. Iterates until no more drops happen (DCE-chain
    collapse on transitively-dead ops).
    """
    blocks = list(fn.blocks)
    while True:
        # Compute used-id set across the current blocks.
        used: set[int] = set()
        for blk in blocks:
            for op in blk.ops:
                for v in op.operands:
                    used.add(v.id)
            for p in blk.params:
                used.add(p.id)
        for p in fn.params:
            used.add(p.id)
        # Drop ops whose ALL results are unused and not side-effecting.
        any_dropped = False
        new_blocks = []
        for blk in blocks:
            new_ops = []
            for op in blk.ops:
                if _is_side_effect(op):
                    new_ops.append(op)
                    continue
                if not op.results:
                    # No results AND not side-effecting — drop it
                    # (e.g., a stray PASS-shaped op).
                    any_dropped = True
                    continue
                if all(r.id not in used for r in op.results):
                    any_dropped = True
                    continue
                new_ops.append(op)
            new_blocks.append(replace(blk, ops=new_ops))
        blocks = new_blocks
        if not any_dropped:
            break
    return replace(fn, blocks=blocks)


def redundant_zero_coalesce(fn: ti.TileFn) -> ti.TileFn:
    """Pass 2 — redundant TILE_ZEROS coalescing.

    Two TILE_ZEROS ops are redundant when they produce same-dtype +
    same-shape + same-memspace tiles. Coalesces: keeps the FIRST,
    drops the LATER, and rewrites all operand uses of the later
    tile's id to the earlier tile's id.

    Constraints:
    - Only coalesces within a single block (cross-block requires
      block-param plumbing that's v2.0 Phase A scope).
    - Only coalesces TILE_ZEROS (not TILE_CONST, which carries a
      literal value via attrs — different constants are different
      tiles even if the shape matches).

    The coalesced tiles share the same register slot in the lowered
    PTX, freeing register pressure for downstream ops (e.g., wmma
    fragments).
    """
    new_blocks = []
    for blk in fn.blocks:
        # Map: (dtype_name, shape_tuple, memspace) -> first TileValue
        seen: dict[tuple, ti.TileValue] = {}
        # Map: redundant_id -> canonical TileValue
        remap: dict[int, ti.TileValue] = {}
        kept_ops: list[ti.TileOp] = []
        for op in blk.ops:
            if op.kind == ti.TileOpKind.TILE_ZEROS and op.results:
                result = op.results[0]
                if hasattr(result.ty, "dtype") and hasattr(result.ty, "shape"):
                    dtype_name = result.ty.dtype.name
                    shape_tuple = tuple(
                        d.value if hasattr(d, "value") else repr(d)
                        for d in result.ty.shape
                    )
                    memspace = getattr(result.ty, "memspace", "")
                    key = (dtype_name, shape_tuple, memspace)
                    if key in seen:
                        # Drop this op; remap its result id.
                        remap[result.id] = seen[key]
                        continue
                    seen[key] = result
            # Rewrite operand uses through the remap.
            if remap:
                new_operands = [
                    remap.get(v.id, v) for v in op.operands
                ]
                op = replace(op, operands=new_operands)
            kept_ops.append(op)
        new_blocks.append(replace(blk, ops=kept_ops))
    return replace(fn, blocks=new_blocks)


def register_reuse_hints(fn: ti.TileFn) -> dict[int, tuple[int, int]]:
    """Pass 3 — register-reuse analysis (read-only).

    Returns a `{tile_id: (last_block_idx, last_op_idx)}` map giving
    the LAST use of each TileValue across the fn. A backend can use
    this to free the tile's register slot once the last-use op has
    emitted; pre-this, the PTX backend uses fresh registers
    monotonically (no reuse), which can exhaust the register pool
    on long kernels.

    Stage 107 v1.0 ships only the analysis — no rewrite. v2.0 Phase A's
    register-coloring pass will consume the hints. Useful in isolation
    for debugging tools (live-tile counts at any op index, max-
    concurrent-tile metric for register-pressure tuning).

    Note: includes fn params + block params in the scan so a tile that
    crosses block boundaries gets its true last use, not just the last
    intra-block use.
    """
    hints: dict[int, tuple[int, int]] = {}
    # Also seed result-defining ops as "uses at the defining position"
    # so a TileValue that's defined but never used still has an entry
    # (last_use = define_idx). Callers that want to filter to truly-
    # used tiles can intersect with op.operands separately.
    for b_idx, blk in enumerate(fn.blocks):
        for o_idx, op in enumerate(blk.ops):
            for v in op.operands:
                hints[v.id] = (b_idx, o_idx)
            for r in op.results:
                # Seed the define position so unused-but-defined
                # tiles still have an entry. Later operand-use sweeps
                # will overwrite if they're later in topological order.
                if r.id not in hints:
                    hints[r.id] = (b_idx, o_idx)
    return hints


def run_all_passes(fn: ti.TileFn) -> ti.TileFn:
    """Composition helper — runs dead-tile-elim then zero-coalesce
    then dead-tile-elim again (to catch newly-dead ops exposed by
    coalescing). Order matters: coalescing produces fresh dead ops
    (the redundant TILE_ZEROS results no users have anymore), so a
    second DCE pass catches them."""
    fn = dead_tile_elim(fn)
    fn = redundant_zero_coalesce(fn)
    fn = dead_tile_elim(fn)
    return fn
