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

import math
import struct

from .. import tir


# Stage 17 trap-id: NaN-result encountered during compile-time float folding.
# Phase-0 policy is to refuse to bake a NaN literal into the binary because
# the backend's struct.pack('<f', ...) lossily rejects it (the runtime would
# still be free to produce a NaN via division-by-zero etc., but folding
# arithmetic on user-written constants down to a NaN payload is the
# compiler's choice — and Phase-0 chooses to surface it).
class FoldError(Exception):
    """Raised by fold_module when a float fold produces NaN (trap 17001).

    Cycle 19 audit-A C19-1: subclass ShiftFoldError carries trap 17002
    for out-of-range shifts on compile-time constants. Same base
    class so a single `except FoldError` catches both."""
    trap_id = 17001


class ShiftFoldError(FoldError):
    """Cycle 19 C19-1 (conf 78): out-of-range shift amount in const fold.
    Symmetric with FoldError for NaN — both are "fold would produce
    undefined behavior, refuse silently-wrong codegen"."""
    trap_id = 17002


_INT_BITS = {
    "i8": 8, "u8": 8,
    "i16": 16, "u16": 16,
    "i32": 32, "u32": 32,
    # Audit 28.8 cycle 20 C19-1 (HIGH): pointer-width aliases must be
    # 64-bit, matching typecheck.py:225-228's `_widen_canon_name`
    # aliasing (isize->i64, usize->u64) and the cycle-19 backend
    # classifier fix at x86_64.py:1005-1017. Pre-fix the 32-bit
    # entry made `_wrap_int_to_type(6_000_000_000, isize) =
    # 1_705_032_704` — silent miscompile reachable at default -O1.
    "isize": 64, "usize": 64,
    "i64": 64, "u64": 64,
    "bool": 32,  # bool comparisons reified to i32 in IR
}


def _wrap_int_to_type(value: int, ty: "tir.TIRType") -> int:
    """Wrap a Python int to the signed range of the given TIR scalar type
    (two's-complement, like x86 hardware). Phase 0 backend uses this same
    wraparound at runtime, so const-folding must match — otherwise a
    folded `INT_MAX + 1` evaluates to 2147483648 (Python int, no wrap)
    and gets stored as 8 bytes, breaking comparisons that work fine when
    folding is disabled."""
    bits = 32  # default for unknown / generic scalar types
    if isinstance(ty, tir.TIRScalar):
        bits = _INT_BITS.get(ty.name, 32)
    mask = (1 << bits) - 1
    half = 1 << (bits - 1)
    v = value & mask
    if v >= half:
        v -= (1 << bits)
    return v


def _try_algebraic_identity(op: tir.Op, defs: dict,
                             res: "tir.Value") -> "tir.Op | None":
    """Apply algebraic identities that simplify ops with one literal operand
    or two equal SSA values. Caught here in addition to the AST simplifier
    in autodiff.py because IR lowering can introduce duplicates the AST
    pass didn't see (e.g. let-inlining producing repeated subexpressions).

    Folded:
        x * 0 = 0,  0 * x = 0  (int + float)
        x - x = 0   (int + float)
        x * 1, 1 * x → still requires identity-forwarding (skipped — needs
                      SSA value remap that this pass can't safely do)
        x + 0 / 0 + x → same (skipped)
        const_int 0 / 0 (DIV by zero) → leave alone, no fold
        const_int N / 1 → const_int N
        const_int N * 1 → const_int N
        x % 1 = 0
    """
    # x * 0 = 0 (int + float)
    if op.kind == tir.OpKind.MUL and len(op.operands) == 2:
        l_def = defs.get(op.operands[0].id)
        r_def = defs.get(op.operands[1].id)
        for d in (l_def, r_def):
            if d is None:
                continue
            if d.kind == tir.OpKind.CONST_INT and int(d.attrs["value"]) == 0:
                return tir.Op(kind=tir.OpKind.CONST_INT, operands=[],
                              results=[res], attrs={"value": 0}, span=op.span)
            # NOTE: 0.0 * NaN = NaN, not 0.0 — float case intentionally
            # left alone to preserve IEEE-754 semantics. Only safe when
            # we can statically rule out NaN, which we can't here.

    # x - x = 0 (same SSA value on both sides). Only safe for INTEGERS:
    # for floats, NaN - NaN = NaN (not 0.0), and we can't statically
    # prove the operand isn't NaN.
    if op.kind == tir.OpKind.SUB and len(op.operands) == 2 \
            and op.operands[0].id == op.operands[1].id:
        ty = op.operands[0].ty
        is_float = isinstance(ty, tir.TIRScalar) and ty.name in (
            "f32", "f64", "f16", "bf16"
        )
        if not is_float:
            return tir.Op(kind=tir.OpKind.CONST_INT, operands=[],
                          results=[res], attrs={"value": 0}, span=op.span)

    # x % 1 = 0 (integer modulo by 1 always 0)
    if op.kind == tir.OpKind.MOD and len(op.operands) == 2:
        r_def = defs.get(op.operands[1].id)
        if r_def is not None and r_def.kind == tir.OpKind.CONST_INT \
                and int(r_def.attrs["value"]) == 1:
            return tir.Op(kind=tir.OpKind.CONST_INT, operands=[],
                          results=[res], attrs={"value": 0}, span=op.span)

    # Comparison of value with itself: == is true (1), != is false (0),
    # <= and >= are true, < and > are false. For IEEE floats, NaN != NaN
    # so the equality case is unsafe for floats — restrict to int.
    if op.kind in (tir.OpKind.CMP_EQ, tir.OpKind.CMP_NE,
                   tir.OpKind.CMP_LT, tir.OpKind.CMP_LE,
                   tir.OpKind.CMP_GT, tir.OpKind.CMP_GE) \
            and len(op.operands) == 2 \
            and op.operands[0].id == op.operands[1].id:
        ty = op.operands[0].ty
        is_float = isinstance(ty, tir.TIRScalar) and ty.name in (
            "f32", "f64", "f16", "bf16"
        )
        # Skip the fold for floats — NaN!=NaN edge case.
        if not is_float:
            true_kinds = {tir.OpKind.CMP_EQ, tir.OpKind.CMP_LE, tir.OpKind.CMP_GE}
            value = 1 if op.kind in true_kinds else 0
            return tir.Op(kind=tir.OpKind.CONST_INT, operands=[],
                          results=[res], attrs={"value": value}, span=op.span)

    return None


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

        # Identity forwarding: x*1, 1*x, x+0, 0+x, x/1, x-0 → forward x.
        # Builds a substitution {result_id: forwarded_value}, applies it
        # to all subsequent operands, and drops the identity ops as dead.
        #
        # Cycle 16 audit-C C2 fix (conf 80): accumulate the actual
        # substitution count returned by _propagate_identities, not a
        # constant 1. Pre-fix a single pass that forwarded N identities
        # only incremented `folded` by 1, understating the work done
        # and risking incorrect short-circuit decisions by callers
        # that compare fold-count to 0.
        prop_count = _propagate_identities(fn, defs)
        if prop_count:
            changed = True
            folded += prop_count
    return folded


def _propagate_identities(fn: tir.FnIR, defs: dict) -> int:
    """One pass of SSA-style value forwarding for algebraic identities.

    Cycle 16 audit-C C2 fix: return the count of substitutions made
    (was bool). 0 means no fold; positive value is the number of ops
    forwarded + dropped this pass."""
    subst: dict[int, "tir.Value"] = {}
    for blk in fn.blocks:
        for op in blk.ops:
            ident = _algebraic_forward(op, defs)
            if ident is not None:
                subst[op.results[0].id] = ident
    if not subst:
        return 0
    # Apply transitively — if a -> b and b -> c, then a -> c.
    changed = True
    while changed:
        changed = False
        for k, v in list(subst.items()):
            if v.id in subst:
                subst[k] = subst[v.id]
                changed = True
    # Rewrite operand lists; drop ops whose result is now in subst.
    for blk in fn.blocks:
        new_ops: list[tir.Op] = []
        for op in blk.ops:
            if op.results and op.results[0].id in subst:
                continue  # dead (forwarded)
            op.operands = [subst.get(o.id, o) for o in op.operands]
            new_ops.append(op)
        blk.ops = new_ops
    return len(subst)


def _algebraic_forward(op: tir.Op, defs: dict) -> "tir.Value | None":
    """If `op` is `x op K` (or `K op x`) where K is an algebraic identity,
    return the value that should replace `op.results[0]`. Else None.

    Identities forwarded:
        x*1 → x,  1*x → x   (int + float)
        x+0 → x,  0+x → x   (int + float)
        x-0 → x             (int + float)
        x/1 → x             (int + float; div-by-1 is exact)
    """
    if not op.results or len(op.operands) != 2:
        return None
    l_def = defs.get(op.operands[0].id)
    r_def = defs.get(op.operands[1].id)

    def is_const(d, value: int | float) -> bool:
        if d is None:
            return False
        if d.kind == tir.OpKind.CONST_INT:
            return int(d.attrs.get("value", -1)) == value
        if d.kind == tir.OpKind.CONST_FLOAT:
            try:
                return float(d.attrs.get("value", 0.0)) == float(value)
            except Exception:
                return False
        return False

    # x * 1, 1 * x  → x
    if op.kind == tir.OpKind.MUL:
        if is_const(r_def, 1):
            return op.operands[0]
        if is_const(l_def, 1):
            return op.operands[1]
    # x + 0, 0 + x  → x
    if op.kind == tir.OpKind.ADD:
        if is_const(r_def, 0):
            return op.operands[0]
        if is_const(l_def, 0):
            return op.operands[1]
    # x - 0  → x  (note: 0 - x is NEG; we don't forward that here)
    if op.kind == tir.OpKind.SUB and is_const(r_def, 0):
        return op.operands[0]
    # x / 1  → x  (int + float)
    if op.kind == tir.OpKind.DIV and is_const(r_def, 1):
        return op.operands[0]
    return None


def _try_fold_op(op: tir.Op, defs: dict) -> tir.Op | None:
    """Try to fold op into a const_*. Return new op, or None if can't fold."""
    if not op.results:
        return None
    res = op.results[0]

    # Algebraic identities — applied before constant folding to catch
    # half-constant cases (one operand is a literal, the other isn't).
    # Each rule rewrites the op into a no-op or a simpler form whose
    # result is bit-identical to the original.
    identity = _try_algebraic_identity(op, defs, res)
    if identity is not None:
        return identity

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
            except FoldError:
                # Stage 28.9 cycle 21 audit-R C20-R1 fix (conf 97):
                # FoldError / ShiftFoldError must propagate as compile
                # errors; the generic `except Exception` below would
                # silently swallow them and the trap contract (17001/
                # 17002) would never surface to the user. Re-raise
                # before the catch-all sees them.
                raise
            except Exception:
                return None
            # Wrap to target type's bit width to match runtime semantics.
            v = _wrap_int_to_type(v, res.ty)
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
            except FoldError:
                # Stage 28.9 cycle 21 audit-R C20-R1 (conf 97): defense in
                # depth — currently no FoldError raises inside this try
                # block (the NaN check at line ~385 is below `except`),
                # but a future edit might move it inside. Re-raise so
                # the trap contract is never silently swallowed.
                raise
            except Exception:
                return None
            # Stage 17 trap-id 17001: NaN-result encountered. Phase-0 refuses
            # to embed a NaN payload in a CONST_FLOAT. Note: +inf/-inf are
            # currently allowed through (the backend's f32 packing will
            # produce an OverflowError on its own); only the explicit NaN
            # case is caught here so the diagnostic is unambiguous.
            if isinstance(v, float) and math.isnan(v):
                raise FoldError(
                    f"[trap 17001] const-fold produced NaN folding "
                    f"{op.kind.name}({l!r}, {r!r}) at "
                    f"{op.span!r}; Phase-0 refuses to bake NaN into a literal"
                )
            return tir.Op(kind=tir.OpKind.CONST_FLOAT,
                         operands=[],
                         results=[res],
                         attrs={"value": v},
                         span=op.span)

    # Bitwise + shift binary on two int consts. Mirrors the arith case
    # above. Python's integer ops have the right semantics for signed
    # i32 (in particular: `>>` is arithmetic, `~` flips all bits and
    # gives the two's-complement negative). _wrap_int_to_type at the end
    # handles overflow back to i32 / i64 width.
    if op.kind in (tir.OpKind.BIT_AND, tir.OpKind.BIT_OR, tir.OpKind.BIT_XOR,
                   tir.OpKind.SHL, tir.OpKind.SHR):
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
                if op.kind == tir.OpKind.BIT_AND:
                    v = l & r
                elif op.kind == tir.OpKind.BIT_OR:
                    v = l | r
                elif op.kind == tir.OpKind.BIT_XOR:
                    v = l ^ r
                elif op.kind == tir.OpKind.SHL:
                    # Stage 28.9 cycle 19 audit-A C19-1 fix (conf 78):
                    # symmetric with FoldError NaN diagnostic at
                    # lines 374-379 — out-of-range shift on
                    # compile-time constants is undefined behavior in
                    # C semantics and a silent surprise to users.
                    # Raise FoldError (trap 17002) for loud failure.
                    # Stage 28.9 cycle 26 audit-T C24-2 fix (conf 92):
                    # canonicalize message format to `[trap NNNNN] body`
                    # (matching the NaN FoldError prefix) so downstream
                    # grep for `^helixc: const-fold error: \[trap ` works
                    # uniformly across all FoldError subclasses.
                    if r < 0 or r >= 64:
                        raise ShiftFoldError(
                            f"[trap 17002] shift amount {r} out of range "
                            f"[0, 63] in const SHL fold"
                        )
                    v = l << r
                elif op.kind == tir.OpKind.SHR:
                    if r < 0 or r >= 64:
                        raise ShiftFoldError(
                            f"[trap 17002] shift amount {r} out of range "
                            f"[0, 63] in const SHR fold"
                        )
                    v = l >> r   # arithmetic in Python for signed ints
                else:
                    return None
            except FoldError:
                # Stage 28.9 cycle 21 audit-R C20-R1 fix (conf 97): the
                # ShiftFoldError raises at the SHL/SHR range checks (lines
                # 428, 435) are INSIDE this try block. Without this
                # re-raise, the generic `except Exception` below would
                # silently swallow them — defeating the trap-17002
                # contract documented in cycle 19 audit-A C19-1.
                raise
            except Exception:
                return None
            v = _wrap_int_to_type(v, res.ty)
            return tir.Op(kind=tir.OpKind.CONST_INT,
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
            v = _wrap_int_to_type(-int(d.attrs["value"]), res.ty)
            return tir.Op(kind=tir.OpKind.CONST_INT,
                         operands=[],
                         results=[res],
                         attrs={"value": v},
                         span=op.span)
        if d.kind == tir.OpKind.CONST_FLOAT:
            v = -float(d.attrs["value"])
            # Stage 17 trap-id 17001: NaN-result from unary NEG. -NaN is
            # still NaN; refuse for the same reason as binary float folds.
            if isinstance(v, float) and math.isnan(v):
                raise FoldError(
                    f"[trap 17001] const-fold produced NaN folding "
                    f"NEG({float(d.attrs['value'])!r}) at {op.span!r}"
                )
            return tir.Op(kind=tir.OpKind.CONST_FLOAT,
                         operands=[],
                         results=[res],
                         attrs={"value": v},
                         span=op.span)

    # Unary bitwise NOT on int const. Python's ~ flips all bits and gives
    # a (potentially negative) Python int; _wrap_int_to_type then constrains
    # to i32/i64 width — bit-identical to the runtime `not eax` instruction.
    if op.kind == tir.OpKind.BIT_NOT:
        if len(op.operands) != 1:
            return None
        d = defs.get(op.operands[0].id)
        if d is None:
            return None
        if d.kind == tir.OpKind.CONST_INT:
            v = _wrap_int_to_type(~int(d.attrs["value"]), res.ty)
            return tir.Op(kind=tir.OpKind.CONST_INT,
                         operands=[],
                         results=[res],
                         attrs={"value": v},
                         span=op.span)

    return None
