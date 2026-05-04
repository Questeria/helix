"""
helixc/ir/passes/cse.py — common subexpression elimination.

Identifies pure operations whose hash (kind + operand ids + attrs) match
an earlier op in the same function. Replaces the later op's results with
the earlier op's results.

Pure op kinds (eligible for CSE):
- CONST_INT, CONST_FLOAT, CONST_BOOL
- ADD, SUB, MUL, DIV, MOD
- NEG
- CMP_*
- CAST

Side-effecting / impure ops are skipped:
- CALL (might have side effects)
- LOAD_VAR / STORE_VAR (memory aliasing — too risky for v0.1)
- LOAD_ELEM / STORE_ELEM
- BR / COND_BR / RETURN
- ALLOC_*
- io.*

After CSE, run DCE to clean up the now-dead duplicates.

License: Apache 2.0
"""

from __future__ import annotations

from .. import tir


PURE_KINDS = {
    tir.OpKind.CONST_INT,
    tir.OpKind.CONST_FLOAT,
    tir.OpKind.CONST_BOOL,
    tir.OpKind.ADD,
    tir.OpKind.SUB,
    tir.OpKind.MUL,
    tir.OpKind.DIV,
    tir.OpKind.MOD,
    tir.OpKind.NEG,
    tir.OpKind.CMP_EQ,
    tir.OpKind.CMP_NE,
    tir.OpKind.CMP_LT,
    tir.OpKind.CMP_LE,
    tir.OpKind.CMP_GT,
    tir.OpKind.CMP_GE,
    tir.OpKind.CAST,
}


def _op_hash(op: tir.Op) -> tuple:
    """Stable hash key for an op based on its semantic equivalence."""
    operand_ids = tuple(o.id for o in op.operands)
    attrs_items = tuple(sorted((k, v) for k, v in op.attrs.items()
                               if isinstance(v, (int, float, str, bool))))
    return (op.kind, operand_ids, attrs_items)


def cse_module(module: tir.Module) -> int:
    total = 0
    for fn in module.functions.values():
        total += cse_function(fn)
    return total


def cse_function(fn: tir.FnIR) -> int:
    """Within a single block, find duplicate pure ops and rewrite later
    references to use the earlier op's results.

    For v0.1 we don't try to do CSE across blocks (would need dominance
    analysis). Per-block is straightforward and catches most cases.
    """
    # value_id -> replacement_value_id (transitive)
    rewrites: dict[int, int] = {}
    found = 0

    for blk in fn.blocks:
        seen: dict[tuple, list[tir.Value]] = {}  # hash -> earlier op's results
        for op in blk.ops:
            if op.kind not in PURE_KINDS:
                continue
            # Apply known rewrites to operands first
            new_operand_ids = []
            for o in op.operands:
                new_id = rewrites.get(o.id, o.id)
                new_operand_ids.append(new_id)
            # Synthesize a fresh hash with rewritten operand ids
            attrs_items = tuple(sorted((k, v) for k, v in op.attrs.items()
                                       if isinstance(v, (int, float, str, bool))))
            key = (op.kind, tuple(new_operand_ids), attrs_items)
            if key in seen:
                # CSE: rewrite each result of THIS op to point at the earlier op's
                # corresponding result.
                earlier_results = seen[key]
                for new_r, old_r in zip(op.results, earlier_results):
                    rewrites[new_r.id] = old_r.id
                found += 1
            else:
                seen[key] = op.results

    if not rewrites:
        return 0

    # Apply rewrites to ALL operand references in the function (across blocks)
    def resolve(idval: int) -> int:
        # Walk transitive rewrites
        seen_walk = set()
        cur = idval
        while cur in rewrites and cur not in seen_walk:
            seen_walk.add(cur)
            cur = rewrites[cur]
        return cur

    for blk in fn.blocks:
        for op in blk.ops:
            for i, o in enumerate(op.operands):
                target = resolve(o.id)
                if target != o.id:
                    # Find a Value with this id... we don't have a registry.
                    # Easier: scan all blocks for the matching value object.
                    # (Inefficient but simple.)
                    found_val = _find_value_by_id(fn, target)
                    if found_val is not None:
                        op.operands[i] = found_val

    return found


def _find_value_by_id(fn: tir.FnIR, value_id: int) -> tir.Value | None:
    for p in fn.params:
        if p.id == value_id:
            return p
    for blk in fn.blocks:
        for p in blk.params:
            if p.id == value_id:
                return p
        for op in blk.ops:
            for r in op.results:
                if r.id == value_id:
                    return r
    return None
