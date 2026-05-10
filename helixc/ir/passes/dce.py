"""
helixc/ir/passes/dce.py — dead code elimination on Tensor IR.

After constant folding, intermediate const ops whose results no longer
have any users are dead. DCE removes them.

A value is *live* if it is:
- Used as an operand to a non-removable op (RETURN, BR, COND_BR, CALL,
  STORE_VAR, STORE_ELEM, side-effecting calls), OR
- Used as an operand to another live op.

We compute liveness via a fixpoint reverse-walk, then drop ops whose
results are all dead AND that have no side effects.

Side-effecting op kinds (kept regardless of result use):
- RETURN, BR, COND_BR
- CALL (might have effects)
- STORE_VAR, STORE_ELEM
- ALLOC_VAR, ALLOC_ARRAY (we keep these even though their results aren't
  directly used — backend uses them for layout)
- MODIFY, SPLICE
- io.print, etc.

License: Apache 2.0
"""

from __future__ import annotations

from .. import tir


SIDE_EFFECT_KINDS = {
    tir.OpKind.RETURN,
    tir.OpKind.BR,
    tir.OpKind.COND_BR,
    tir.OpKind.CALL,
    tir.OpKind.STORE_VAR,
    tir.OpKind.STORE_ELEM,
    tir.OpKind.ALLOC_VAR,
    tir.OpKind.ALLOC_ARRAY,
    tir.OpKind.MODIFY,
    tir.OpKind.SPLICE,
    tir.OpKind.PRINT,
    # QUOTE has the side effect of reserving a reflection cell handle
    # for the binary's runtime; even if the i32 result isn't directly
    # used, downstream MODIFY/SPLICE may target the cell by index.
    tir.OpKind.QUOTE,
    # REFLECT_HASH is similar: it provides a stable testing handle that
    # downstream code may reach via cell indexing.
    tir.OpKind.REFLECT_HASH,
    # Arena ops mutate a global region — even if the result (slot index)
    # is unused, the push/set must still execute for downstream reads.
    tir.OpKind.ARENA_PUSH,
    tir.OpKind.ARENA_SET,
    # Stage 16 — HBM tile stores are observable side effects (the write
    # is what kernel launchers care about). Loads + thread_idx are pure
    # functions of their inputs (operands + the implicit %tid.x), so
    # standard liveness is correct for those.
    tir.OpKind.TILE_INDEX_STORE,
    # Stage 16.5 follow-up audit CRITICAL-1: FFI calls have side effects
    # by definition (puts, free, mutex_lock, etc.). DCE was silently
    # dropping void-return extern calls because their results were
    # never live. Adding here so liveness preserves them unconditionally.
    tir.OpKind.FFI_CALL,
    # Stage 28.5 — TRAP terminates the process. Even if the result slot
    # is unused, the trap MUST execute (it's the entire point of panic).
    tir.OpKind.TRAP,
}


def dce_module(module: tir.Module) -> int:
    """Run DCE on every function. Returns total ops removed."""
    total = 0
    for fn in module.functions.values():
        total += dce_function(fn)
    return total


def dce_function(fn: tir.FnIR) -> int:
    """Compute liveness, drop dead ops. Iterates to fixpoint."""
    removed_total = 0
    changed = True
    while changed:
        changed = False
        # Compute live value-ids
        live: set[int] = set()
        # Seed: operands of side-effecting ops are live
        for blk in fn.blocks:
            for op in blk.ops:
                if op.kind in SIDE_EFFECT_KINDS:
                    for o in op.operands:
                        live.add(o.id)
        # Function params are always live
        for p in fn.params:
            live.add(p.id)
        # Block params are always live
        for blk in fn.blocks:
            for p in blk.params:
                live.add(p.id)
        # Fixpoint: any op whose result is live -> its operands are live
        spread = True
        while spread:
            spread = False
            for blk in fn.blocks:
                for op in blk.ops:
                    if op.kind in SIDE_EFFECT_KINDS:
                        continue
                    if any(r.id in live for r in op.results):
                        for o in op.operands:
                            if o.id not in live:
                                live.add(o.id)
                                spread = True

        # Drop ops whose results are all dead AND op has no side effect
        for blk in fn.blocks:
            new_ops = []
            for op in blk.ops:
                if op.kind in SIDE_EFFECT_KINDS:
                    new_ops.append(op)
                    continue
                if not op.results:
                    new_ops.append(op)
                    continue
                if any(r.id in live for r in op.results):
                    new_ops.append(op)
                else:
                    removed_total += 1
                    changed = True
            blk.ops = new_ops
    return removed_total
