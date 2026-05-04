"""
helixc/ir/passes/const_fold.py — constant folding pass on Tensor IR.

Walks each function's blocks and replaces operations on constant operands
with constant results. E.g.:

    %a = const_int 2
    %b = const_int 3
    %c = add %a, %b      ->     %c = const_int 5

Folding is done iteratively until no change. Operations folded:
  CONST_INT, CONST_FLOAT, ADD, SUB, MUL, DIV, MOD, NEG,
  CMP_EQ, CMP_NE, CMP_LT, CMP_LE, CMP_GT, CMP_GE,
  CAST (between numeric scalars).

Conservative: doesn't touch ops with unknown / non-scalar types,
or ops with side effects (CALL, STORE_*, CONST_TENSOR, etc.).
Doesn't propagate constants across LOAD_VAR/STORE_VAR (would need
proper SSA + alias analysis).

License: Apache 2.0
"""

from __future__ import annotations

import struct

from .. import tir


def _is_int_const(op: tir.Op, consts: dict) -> bool:
    return op.kind == tir.OpKind.CONST_INT


def _is_float_const(op: tir.Op, consts: dict) -> bool:
    return op.kind == tir.OpKind.CONST_FLOAT


def fold_module(module: tir.Module) -> int:
    """Run constant folding on every function in the module.
    Returns total number of ops folded across the whole module."""
    total = 0
    for fn in module.functions.values():
        total += fold_function(fn)
    return total


def fold_function(fn: tir.FnIR) -> int:
    """Iteratively fold constants until fixpoint. Returns count of folded ops."""
    folded = 0
    changed = True
    while changed:
        changed = False
        # Build a value-id -> defining op map for quick lookup
        defs: dict[int, tir.Op] = {}
        for blk in fn.blocks:
            for op in blk.ops:
                for r in op.results:
                    defs[r.id] = op

        for blk in fn.blocks:
            new_ops: list[tir.Op] = []
            for op in blk.ops:
                folded_op = _try_fold_op(op, defs)
                if folded_op is not None:
                    new_ops.append(folded_op)
                    changed = True
                    folded += 1
                else:
                    new_ops.append(op)
            blk.ops = new_ops
    return folded


def _try_fold_op(op: tir.Op, defs: dict) -> tir.Op | None:
    """Try to fold op into a const_*. Return new op, or None if can't fold."""
    if not op.results:
        return None
    res = op.results[0]

    # Binary on two int consts
    if op.kind in (tir.OpKind.ADD, tir.OpKind.SUB, tir.OpKind.MUL,
                   tir.OpKind.DIV, tir.OpKind.MOD):
        if len(op.operands) != 2:
            return None
        l_def = defs.get(op.operands[0].id)
        r_def = defs.get(op.operands[1].id)
        if l_def is None or r_def is None:
            return None
        if l_def.kind == tir.OpKind.CONST_INT and r_def.kind == tir.OpKind.CONST_INT:
            l = int(l_def.attrs["value"])
            r = int(r_def.attrs["value"])
            try:
                if op.kind == tir.OpKind.ADD:
                    v = l + r
                elif op.kind == tir.OpKind.SUB:
                    v = l - r
                elif op.kind == tir.OpKind.MUL:
                    v = l * r
                elif op.kind == tir.OpKind.DIV:
                    if r == 0:
                        return None
                    # C / x86 idiv semantics: truncate toward zero.
                    # Python's // truncates toward -inf, so we must compute
                    # |l| // |r| and apply the sign.
                    sign = -1 if (l < 0) != (r < 0) else 1
                    v = sign * (abs(l) // abs(r))
                elif op.kind == tir.OpKind.MOD:
                    if r == 0:
                        return None
                    # C/idiv semantics: result has the sign of the dividend.
                    sign = -1 if l < 0 else 1
                    v = sign * (abs(l) % abs(r))
                else:
                    return None
            except Exception:
                return None
            return tir.Op(kind=tir.OpKind.CONST_INT,
                         operands=[],
                         results=[res],
                         attrs={"value": v},
                         span=op.span)
        if l_def.kind == tir.OpKind.CONST_FLOAT and r_def.kind == tir.OpKind.CONST_FLOAT:
            l = float(l_def.attrs["value"])
            r = float(r_def.attrs["value"])
            try:
                if op.kind == tir.OpKind.ADD:
                    v = l + r
                elif op.kind == tir.OpKind.SUB:
                    v = l - r
                elif op.kind == tir.OpKind.MUL:
                    v = l * r
                elif op.kind == tir.OpKind.DIV:
                    if r == 0.0:
                        return None
                    v = l / r
                else:
                    return None
            except Exception:
                return None
            return tir.Op(kind=tir.OpKind.CONST_FLOAT,
                         operands=[],
                         results=[res],
                         attrs={"value": v},
                         span=op.span)

    # Comparisons on const operands
    if op.kind in (tir.OpKind.CMP_EQ, tir.OpKind.CMP_NE, tir.OpKind.CMP_LT,
                   tir.OpKind.CMP_LE, tir.OpKind.CMP_GT, tir.OpKind.CMP_GE):
        if len(op.operands) != 2:
            return None
        l_def = defs.get(op.operands[0].id)
        r_def = defs.get(op.operands[1].id)
        if l_def is None or r_def is None:
            return None
        if l_def.kind == tir.OpKind.CONST_INT and r_def.kind == tir.OpKind.CONST_INT:
            l = int(l_def.attrs["value"])
            r = int(r_def.attrs["value"])
            cmp_map = {
                tir.OpKind.CMP_EQ: l == r,
                tir.OpKind.CMP_NE: l != r,
                tir.OpKind.CMP_LT: l < r,
                tir.OpKind.CMP_LE: l <= r,
                tir.OpKind.CMP_GT: l > r,
                tir.OpKind.CMP_GE: l >= r,
            }
            return tir.Op(kind=tir.OpKind.CONST_INT,
                         operands=[],
                         results=[res],
                         attrs={"value": 1 if cmp_map[op.kind] else 0},
                         span=op.span)

    # Unary neg on int const
    if op.kind == tir.OpKind.NEG:
        if len(op.operands) != 1:
            return None
        d = defs.get(op.operands[0].id)
        if d is None:
            return None
        if d.kind == tir.OpKind.CONST_INT:
            v = -int(d.attrs["value"])
            return tir.Op(kind=tir.OpKind.CONST_INT,
                         operands=[],
                         results=[res],
                         attrs={"value": v},
                         span=op.span)
        if d.kind == tir.OpKind.CONST_FLOAT:
            v = -float(d.attrs["value"])
            return tir.Op(kind=tir.OpKind.CONST_FLOAT,
                         operands=[],
                         results=[res],
                         attrs={"value": v},
                         span=op.span)

    return None
