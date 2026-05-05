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
    """Stable hash key for an op based on its semantic equivalence.
    Includes the result type so `&&` MUL (bool) doesn't merge with i32 MUL,
    and CAST ops with different target types stay distinct (audit-10)."""
    operand_ids = tuple(o.id for o in op.operands)
    # Include attrs that are primitive types (skip complex ones like
    # TIRType objects — those go through repr in the key tail below).
    attrs_items = tuple(sorted((k, v) for k, v in op.attrs.items()
                               if isinstance(v, (int, float, str, bool))))
    # Include repr of any non-primitive attrs (TIRType, etc.) so CAST's
    # from_ty/to_ty don't collide on result-type alone.
    attrs_complex = tuple(sorted(
        (k, repr(v)) for k, v in op.attrs.items()
        if not isinstance(v, (int, float, str, bool))))
    result_ty_key = repr(op.results[0].ty) if op.results else None
    return (op.kind, operand_ids, attrs_items, attrs_complex, result_ty_key)


def cse_module(module: tir.Module) -> int:
    total = 0
    for fn in module.functions.values():
        total += cse_function(fn)
    return total


def cse_function(fn: tir.FnIR) -> int:
    """Per-block CSE. Within each block, find duplicate pure ops and rewrite
    later operand references in that SAME block to use the earlier op's
    results.

    We deliberately do not propagate CSE rewrites across block boundaries:
    that would require dominance analysis (a value defined in block A is
    only safely usable from block B if A dominates B in the CFG). For v0.1
    per-block is sound and catches most cases.
    """
    found = 0
    for blk in fn.blocks:
        # block-scoped: hash -> earlier op's results
        seen: dict[tuple, list[tir.Value]] = {}
        # block-scoped: value_id -> replacement Value object
        rewrites: dict[int, tir.Value] = {}

        for op in blk.ops:
            # First, apply known rewrites to operand references inside this block
            for i, o in enumerate(op.operands):
                if o.id in rewrites:
                    op.operands[i] = rewrites[o.id]

            if op.kind not in PURE_KINDS:
                continue

            # Hash the (potentially-rewritten) op. Use the same shape as
            # `_op_hash` (defined above) so the audit-10 fix (include
            # result_ty and complex attrs) actually applies. Without
            # this, bool MUL and i32 MUL with the same operands hashed
            # to the same key and got merged — producing a bool result
            # where i32 was expected, or vice versa.
            key = _op_hash(op)
            if key in seen:
                earlier_results = seen[key]
                for new_r, old_r in zip(op.results, earlier_results):
                    rewrites[new_r.id] = old_r
                found += 1
            else:
                seen[key] = op.results

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
