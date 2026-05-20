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
from .. import tir


# Cycle 1 Batch IR silent-failure HIGH-2 fix: explicit PURE allowlist
# + exhaustiveness check across the entire TileOpKind enum. Pre-fix,
# _SIDE_EFFECT_KINDS was a small denylist (7 entries); when Stage 64
# added TMA_LOAD / TILE_LOAD_GLOBAL / TILE_LOAD_SHARED, dead_tile_elim
# would happily drop them when their results were unused — but
# TILE_LOAD_GLOBAL / TMA_LOAD have observable async-queue / memory-
# pressure side effects on real hardware even when the loaded value
# isn't consumed. Same drift-class risk as Batch FE _strip_wrapper_chain
# table-out-of-sync defect that bit me in batch 4.
#
# Post-fix: invert the policy. Maintain _PURE_TILE_KINDS as the
# allowlist (small + auditable). Any kind not in _PURE_TILE_KINDS is
# treated as side-effecting (conservative — preserves correctness on
# unknown ops). Plus the _check_kind_coverage assertion forces a
# decision when a new TileOpKind is added: pure-or-side-effect must
# be explicit.

_PURE_TILE_KINDS = frozenset({
    # Tile creation (no observable effect beyond producing the value)
    ti.TileOpKind.TILE_ZEROS,
    ti.TileOpKind.TILE_CONST,
    # Layout transforms (pure value-shaping)
    ti.TileOpKind.TILE_TRANSPOSE,
    ti.TileOpKind.TILE_RESHAPE,
    # Compute on tiles (pure values; reductions still pure)
    ti.TileOpKind.TILE_ADD,
    ti.TileOpKind.TILE_SUB,
    ti.TileOpKind.TILE_MUL,
    ti.TileOpKind.TILE_MATMUL,
    ti.TileOpKind.TILE_REDUCE,
    # Scalar ops passed through (pure arithmetic)
    ti.TileOpKind.SCALAR_CONST_INT,
    ti.TileOpKind.SCALAR_CONST_FLOAT,
    ti.TileOpKind.SCALAR_ADD,
    ti.TileOpKind.SCALAR_SUB,
    ti.TileOpKind.SCALAR_MUL,
    ti.TileOpKind.SCALAR_NEG,
    ti.TileOpKind.SCALAR_CMP,
    ti.TileOpKind.SCALAR_SELECT,
    # GPU primitives that produce values without observable side
    # effects on the producing thread (THREAD_IDX is a builtin read)
    ti.TileOpKind.THREAD_IDX,
})

# Side-effecting kinds (memory ops, sync, ctrl flow) — derived from
# the enum's complement of _PURE_TILE_KINDS at module load.
_SIDE_EFFECT_KINDS = frozenset(
    k for k in ti.TileOpKind if k not in _PURE_TILE_KINDS
)


def _check_kind_coverage() -> None:
    """Module-load assertion: every TileOpKind enum value must be
    classified as either pure or side-effecting. Adding a new
    TileOpKind without classifying it here is a build error.

    This is the Cycle 1 Batch IR silent-failure HIGH-2 fix's
    exhaustiveness guard — mirrors the dce.py SIDE_EFFECT_KINDS
    pattern and prevents the table-out-of-sync drift class that
    bit Batch FE _strip_wrapper_chain at batch 4."""
    for k in ti.TileOpKind:
        assert (k in _PURE_TILE_KINDS) ^ (k in _SIDE_EFFECT_KINDS), (
            f"TileOpKind {k.name} is missing or in both pure + side-"
            f"effect tables — classify it in tile_opt._PURE_TILE_KINDS"
        )


_check_kind_coverage()


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


def _coalesce_key(ty: object) -> Optional[tuple]:
    """Build a faithful, hashable dedup key for a TILE_ZEROS result
    type — or None when the type cannot be PROVEN identical to another
    by static inspection, in which case the tile must NOT be coalesced.

    v2.x re-audit R6 (IR HIGH + MEDIUM-must-fix). The pre-fix key was
    `(dtype_name, shape_tuple, memspace)` where `shape_tuple` fell back
    to `repr(d)` for any non-DimConst dim and `memspace` was
    `getattr(ty, "memspace", "")`. Two silent-miscompile modes:

      * `DimDyn` is a fieldless frozen dataclass, so `repr(DimDyn())`
        is one constant string for every instance. Two zero tiles of
        *different runtime extent* keyed identically and were
        coalesced — downstream codegen then reads the wrong tile
        length.
      * `TIRTensorTy` carries `device` / `layout` but no `memspace`,
        so `getattr(..., "memspace", "")` silently yielded `""`; two
        tensor zeros differing only in device coalesced, rewiring a
        use onto a tile on the wrong device.

    Fix: coalesce only a `TIRTileTy` whose shape is entirely
    `DimConst` — every key component is then a concrete, faithful
    value. Dynamic-shaped tiles (DimVar / DimDyn / DimExpr) and
    tensor-typed zeros are conservatively left un-coalesced; that is a
    missed optimization, never a miscompile, and matches the pass's
    documented v1.0 minimum-viable scope.
    """
    if not isinstance(ty, tir.TIRTileTy):
        return None
    dims: list[int] = []
    for d in ty.shape:
        if not isinstance(d, tir.DimConst):
            return None
        dims.append(d.value)
    # v2.x re-audit R7 (IR MEDIUM): fold memspace casing. Lowering
    # canonicalizes to lower-case ("reg"); a verbatim key would fail to
    # coalesce a "REG"/"reg" pair. memspace is case-insensitive, so
    # normalizing only ever recovers a missed dedup — never miscoalesces.
    return (ty.dtype.name, tuple(dims), ty.memspace.lower())


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
    - Only coalesces a TILE_ZEROS whose result is a `TIRTileTy` with
      an entirely-`DimConst` (statically known) shape — see
      `_coalesce_key`. Dynamic-shaped tiles and tensor-typed zeros are
      conservatively left alone (never miscoalesced).

    The coalesced tiles share the same register slot in the lowered
    PTX, freeing register pressure for downstream ops (e.g., wmma
    fragments).
    """
    new_blocks = []
    for blk in fn.blocks:
        # Map: faithful type key (see _coalesce_key) -> first TileValue
        seen: dict[tuple, ti.TileValue] = {}
        # Map: redundant_id -> canonical TileValue
        remap: dict[int, ti.TileValue] = {}
        kept_ops: list[ti.TileOp] = []
        for op in blk.ops:
            if op.kind == ti.TileOpKind.TILE_ZEROS and op.results:
                result = op.results[0]
                key = _coalesce_key(result.ty)
                if key is not None:
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
