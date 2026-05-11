"""
helixc/frontend/monomorphize.py

Monomorphization pass: instantiates generic functions with concrete type args.

Input: A.Program with FnDecls that have generic params (e.g., fn id[T](x: T) -> T)
       and call sites with turbofish (e.g., id::<i32>(x))
Output: A.Program with concrete instantiations replacing generic calls.

Algorithm:
  1. Collect all FnDecls with non-empty generics into a generics-table.
  2. Walk the program, find Calls whose callee is Name with non-empty generics.
  3. For each unique (fn_name, type_args) pair:
     - Generate mangled name (id__i32, pair__i32_f64).
     - Clone the FnDecl, substituting type variables in params/return_ty/body.
     - Add to prog.items (only once per unique pair).
     - Replace the callee with Name(mangled_name, generics=[]).
  4. Repeat until no more generic calls remain (cloned fns may call other generics).
  5. Drop original generic FnDecls from prog.items at the end.

License: Apache 2.0
"""

from __future__ import annotations
from copy import deepcopy
from typing import Optional

from . import ast_nodes as A


def mangle(name: str, ty_args: list[A.TyNode]) -> str:
    """Generate a unique mangled name for a (fn_name, type_args) pair.

    Conventions:
      id::<i32>           -> id__i32
      pair::<i32, f64>    -> pair__i32_f64
      box::<(i32, i64)>   -> box__tup_i32_i64
    """
    parts = [_mangle_ty(t) for t in ty_args]
    return f"{name}__" + "_".join(parts)


def _mangle_ty(t: A.TyNode) -> str:
    if isinstance(t, A.TyName):
        return t.name
    if isinstance(t, A.TyTuple):
        return "tup_" + "_".join(_mangle_ty(e) for e in t.elems)
    if isinstance(t, A.TyArray):
        return "arr_" + _mangle_ty(t.elem)
    if isinstance(t, A.TyRef):
        return ("refmut_" if t.is_mut else "ref_") + _mangle_ty(t.inner)
    if isinstance(t, A.TyFn):
        return "fn_" + "_".join(_mangle_ty(p) for p in t.params) + "_to_" + _mangle_ty(t.ret)
    if isinstance(t, A.TyTensor):
        return "tensor_" + _mangle_ty(t.dtype)
    if isinstance(t, A.TyTile):
        return "tile_" + _mangle_ty(t.dtype)
    if isinstance(t, A.TyGeneric):
        return t.base + "_" + "_".join(_mangle_ty(a) for a in t.args)
    return "X"


class ShapeFoldError(ValueError):
    """Audit 28.8 cycle 3 C3-6: raised when shape-time constant fold
    hits a hard error (currently division/modulo by zero). Trap 28801.
    Caught by the typechecker / driver and surfaced as a user-facing
    diagnostic — silent-fallthrough → length 0 is no longer allowed."""

    def __init__(self, msg: str, span: A.Span):
        super().__init__(msg)
        self.span = span
        self.msg = msg


def _fold_intlit_arith(expr: A.Expr) -> A.Expr:
    """Audit 28.8 cycle 2 B:C11: one-level constant-fold for shape
    expressions of the form Binary(IntLit, op, IntLit). Pre-fix,
    `_subst_shape_expr` substituted `Name("N")` → `IntLit(64)` but
    left the surrounding Binary unfolded, so `[N*2; 16]` mono'd
    against N=64 stayed as `Binary(*, IntLit(64), IntLit(2))` instead
    of `IntLit(128)`. Downstream lower-ast.py defaulted non-IntLit
    shapes to 0 — silent miscount.

    Applies only to commutative arithmetic with two IntLit operands.
    Non-foldable shapes (e.g. one side still has a Name leaf) flow
    through unchanged.

    Audit 28.8 cycle 3 C3-6: `/ 0` and `% 0` now raise ShapeFoldError
    (trap 28801) instead of silently leaving the Binary unfolded and
    letting lower_ast default the length to 0."""
    if not isinstance(expr, A.Binary):
        return expr
    if not (isinstance(expr.left, A.IntLit) and isinstance(expr.right, A.IntLit)):
        return expr
    lv = expr.left.value
    rv = expr.right.value
    op = expr.op
    if op == "+":
        return A.IntLit(span=expr.span, value=lv + rv, type_suffix=None)
    if op == "-":
        return A.IntLit(span=expr.span, value=lv - rv, type_suffix=None)
    if op == "*":
        return A.IntLit(span=expr.span, value=lv * rv, type_suffix=None)
    if op == "/":
        if rv == 0:
            raise ShapeFoldError(
                f"{expr.span.line}:{expr.span.col}: division by zero "
                f"in shape expression (trap 28801)",
                expr.span,
            )
        return A.IntLit(span=expr.span, value=lv // rv, type_suffix=None)
    if op == "%":
        if rv == 0:
            raise ShapeFoldError(
                f"{expr.span.line}:{expr.span.col}: modulo by zero "
                f"in shape expression (trap 28801)",
                expr.span,
            )
        return A.IntLit(span=expr.span, value=lv % rv, type_suffix=None)
    return expr


def _fold_intlit_unary(expr: A.Expr) -> A.Expr:
    """Audit 28.8 cycle 3 D5: symmetric one-level fold for
    `Unary(-, IntLit)` and `Unary(+, IntLit)`. Pre-fix, `[T; -N]` after
    substitution stayed `Unary(-, IntLit(N))` and `_resolve_size_expr`
    fell through to TyUnknown (`size expr Unary`) — silent
    miscount."""
    if not isinstance(expr, A.Unary):
        return expr
    if not isinstance(expr.operand, A.IntLit):
        return expr
    if expr.op == "-":
        return A.IntLit(span=expr.span,
                        value=-expr.operand.value,
                        type_suffix=None)
    if expr.op == "+":
        return expr.operand
    return expr


def _subst_shape_expr(expr: A.Expr, subst: dict[str, A.TyNode]) -> A.Expr:
    """Audit 28.8 B8: substitute size-kind generic params inside a tile
    or tensor shape expression.

    `subst` maps generic-param names to either TyNode (for type-kind)
    or — by convention used at the call site — a sentinel pseudo-type
    that wraps an IntLit (for size-kind, treated below). The shape AST
    uses Expr (typically Name or IntLit), so we walk and rewrite
    `Name(generic)` → `IntLit(value)` when the substituted value is a
    size literal.

    Pre-fix, substitute_ty for TyTile only handled `dtype` and shared
    `shape` + `memspace` directly — so `Tile<f32, [N], HBM>` mono'd
    against `N=128` produced a clone with `shape=[Name("N")]` (NOT
    `[IntLit(128)]`), and the lower-ast.py path defaulted shape[0] to
    0 when the leading element wasn't an IntLit. Trap 16003 is now
    reserved for that path.

    Audit 28.8 cycle 2 B:C11: after substitution, fold Binary(IntLit,
    op, IntLit) → IntLit so `[N*2; 16]` with N=64 becomes `[128; 16]`."""
    if expr is None:
        return expr
    if isinstance(expr, A.Name):
        # If the generic-param substitution provides a size literal,
        # replace with IntLit. Otherwise leave as-is.
        repl = subst.get(expr.name)
        if isinstance(repl, A.TyName) and repl.name.startswith("size_"):
            try:
                return A.IntLit(span=expr.span,
                                value=int(repl.name[len("size_"):]),
                                type_suffix=None)
            except ValueError:
                return expr
        # Audit 28.8 B8: callers may pass a raw IntLit through subst
        # by wrapping it as a TyArray-of-sorts, but the canonical
        # path is via _SizeLit (private sentinel below).
        if isinstance(repl, _SizeLitMarker):
            return A.IntLit(span=expr.span, value=repl.value,
                            type_suffix=None)
        return expr
    if isinstance(expr, A.Binary):
        folded = A.Binary(span=expr.span, op=expr.op,
                          left=_subst_shape_expr(expr.left, subst),
                          right=_subst_shape_expr(expr.right, subst))
        # Audit 28.8 cycle 2 B:C11: fold if both children are IntLits.
        return _fold_intlit_arith(folded)
    if isinstance(expr, A.Unary):
        folded = A.Unary(span=expr.span, op=expr.op,
                         operand=_subst_shape_expr(expr.operand, subst))
        # Audit 28.8 cycle 3 D5: post-fold the Unary so `-N` with N=5
        # becomes `IntLit(-5)` rather than `Unary(-, IntLit(5))`.
        return _fold_intlit_unary(folded)
    return expr


class _SizeLitMarker:
    """Internal sentinel for size-kind generic substitutions. Callers
    of substitute_ty pass `subst[N] = _SizeLitMarker(128)` to indicate
    a size-kind binding rather than a type binding. Code that walks
    `subst` and only handles TyNode treats this as opaque (it skips
    over it). Audit 28.8 B8."""
    __slots__ = ("value",)
    def __init__(self, value: int):
        self.value = value


def substitute_ty(t: A.TyNode, subst: dict[str, A.TyNode]) -> A.TyNode:
    """Substitute generic type variables with concrete types throughout t.

    Audit 28.8 B6: added TyPtr arm. Pre-fix, `fn deref<T>(p: *const T)
    -> T { unsafe { *p } }` mono'd with T=f64 silently left the param
    as `*const T` (TyName('T') inside), and the lowering path defaulted
    the inner element width to i32 — silent type-pun.

    Audit 28.8 B8: TyTile.shape and TyTensor.shape now have their
    Name(N) entries substituted when N is a size-kind generic. The
    `subst` dict carries `_SizeLitMarker(int)` sentinels for size
    bindings."""
    if isinstance(t, A.TyName):
        if t.name in subst:
            repl = subst[t.name]
            if isinstance(repl, _SizeLitMarker):
                # Size-kind generic used as TyName — convert to a
                # placeholder TyName encoding the value. Callers that
                # then walk types interpret `size_N` as a literal.
                return A.TyName(span=t.span, name=f"size_{repl.value}")
            return deepcopy(repl)
        return t
    if isinstance(t, A.TyTuple):
        return A.TyTuple(span=t.span, elems=[substitute_ty(e, subst) for e in t.elems])
    if isinstance(t, A.TyArray):
        # Audit 28.8 cycle 2 B:C8: substitute the size expression too,
        # mirroring TyTile/TyTensor shape sub. Pre-fix `[T; N]` mono'd
        # against {T=f64, N=8} produced `[f64; N]` (Name unchanged),
        # and downstream codegen took 0 as the silent default length.
        return A.TyArray(span=t.span,
                         elem=substitute_ty(t.elem, subst),
                         size=_subst_shape_expr(t.size, subst))
    if isinstance(t, A.TyRef):
        return A.TyRef(span=t.span, inner=substitute_ty(t.inner, subst), is_mut=t.is_mut)
    if isinstance(t, A.TyPtr):
        # Audit 28.8 B6 (trap reservation: handled at lower-ast level).
        return A.TyPtr(span=t.span,
                       inner=substitute_ty(t.inner, subst),
                       is_mut=t.is_mut)
    if isinstance(t, A.TyFn):
        return A.TyFn(span=t.span,
                      params=[substitute_ty(p, subst) for p in t.params],
                      ret=substitute_ty(t.ret, subst))
    if isinstance(t, A.TyTensor):
        # Audit 28.8 B8: substitute size-kind generics in shape exprs.
        new_shape = [_subst_shape_expr(s, subst) for s in t.shape]
        return A.TyTensor(span=t.span, dtype=substitute_ty(t.dtype, subst),
                          shape=new_shape, device=t.device, layout=t.layout)
    if isinstance(t, A.TyTile):
        # Audit 28.8 B8: same for TyTile.
        new_shape = [_subst_shape_expr(s, subst) for s in t.shape]
        return A.TyTile(span=t.span, dtype=substitute_ty(t.dtype, subst),
                        shape=new_shape, memspace=t.memspace)
    if isinstance(t, A.TyGeneric):
        return A.TyGeneric(span=t.span, base=t.base,
                           args=[substitute_ty(a, subst) for a in t.args])
    return t


def _walk_subst_expr(e: A.Expr, subst: dict[str, A.TyNode]) -> A.Expr:
    """Walk an expression tree, substituting types where they appear (Cast.target_ty,
    nested Block/If/Match/etc).
    """
    if isinstance(e, A.Cast):
        return A.Cast(span=e.span,
                      value=_walk_subst_expr(e.value, subst),
                      target_ty=substitute_ty(e.target_ty, subst))
    if isinstance(e, A.Block):
        return A.Block(span=e.span,
                       stmts=[_walk_subst_stmt(s, subst) for s in e.stmts],
                       final_expr=_walk_subst_expr(e.final_expr, subst) if e.final_expr is not None else None)
    if isinstance(e, A.If):
        else_ = e.else_
        if isinstance(else_, A.Block):
            else_ = _walk_subst_expr(else_, subst)
        elif isinstance(else_, A.If):
            else_ = _walk_subst_expr(else_, subst)
        return A.If(span=e.span,
                    cond=_walk_subst_expr(e.cond, subst),
                    then=_walk_subst_expr(e.then, subst),
                    else_=else_)
    if isinstance(e, A.Match):
        return A.Match(span=e.span,
                       scrutinee=_walk_subst_expr(e.scrutinee, subst),
                       arms=[A.MatchArm(span=a.span, pattern=a.pattern,
                                        guard=_walk_subst_expr(a.guard, subst) if a.guard else None,
                                        body=_walk_subst_expr(a.body, subst)) for a in e.arms])
    if isinstance(e, A.For):
        return A.For(span=e.span, var_name=e.var_name,
                     iter_expr=_walk_subst_expr(e.iter_expr, subst),
                     body=_walk_subst_expr(e.body, subst))
    if isinstance(e, A.While):
        return A.While(span=e.span,
                       cond=_walk_subst_expr(e.cond, subst),
                       body=_walk_subst_expr(e.body, subst))
    if isinstance(e, A.Loop):
        return A.Loop(span=e.span, body=_walk_subst_expr(e.body, subst))
    if isinstance(e, A.Binary):
        return A.Binary(span=e.span, op=e.op,
                        left=_walk_subst_expr(e.left, subst),
                        right=_walk_subst_expr(e.right, subst))
    if isinstance(e, A.Unary):
        return A.Unary(span=e.span, op=e.op, operand=_walk_subst_expr(e.operand, subst))
    if isinstance(e, A.Call):
        # Audit 28.8 cycle 3 D9: also substitute the callee's
        # turbofish generics list. Pre-fix, `caller[T]` calling
        # `id::<T>(x)` cloned to `caller__i32` with the body still
        # holding `Call(Name('id', generics=[T]))` — leaving an
        # unresolved type-var that downstream codegen interpreted as
        # the literal type name 'T'. Now T is substituted to i32 so
        # the mono iteration discovers `id::<i32>` and produces
        # `id__i32`.
        new_callee = e.callee
        if isinstance(new_callee, A.Name) and new_callee.generics:
            new_callee = A.Name(
                span=new_callee.span,
                name=new_callee.name,
                generics=[substitute_ty(g, subst)
                          for g in new_callee.generics],
            )
        return A.Call(span=e.span,
                      callee=_walk_subst_expr(new_callee, subst),
                      args=[_walk_subst_expr(a, subst) for a in e.args])
    if isinstance(e, A.Index):
        return A.Index(span=e.span,
                       callee=_walk_subst_expr(e.callee, subst),
                       indices=[_walk_subst_expr(a, subst) for a in e.indices])
    if isinstance(e, A.Field):
        return A.Field(span=e.span, obj=_walk_subst_expr(e.obj, subst), name=e.name)
    if isinstance(e, A.TupleLit):
        return A.TupleLit(span=e.span, elems=[_walk_subst_expr(x, subst) for x in e.elems])
    if isinstance(e, A.ArrayLit):
        return A.ArrayLit(span=e.span, elems=[_walk_subst_expr(x, subst) for x in e.elems])
    if isinstance(e, A.StructLit):
        return A.StructLit(span=e.span, name=e.name,
                           fields=[(n, _walk_subst_expr(v, subst)) for (n, v) in e.fields])
    if isinstance(e, A.Assign):
        return A.Assign(span=e.span, target=_walk_subst_expr(e.target, subst),
                        op=e.op, value=_walk_subst_expr(e.value, subst))
    if isinstance(e, A.Return):
        return A.Return(span=e.span,
                        value=_walk_subst_expr(e.value, subst) if e.value is not None else None)
    if isinstance(e, A.Break):
        return A.Break(span=e.span,
                       value=_walk_subst_expr(e.value, subst) if e.value is not None else None)
    if isinstance(e, A.Range):
        return A.Range(span=e.span,
                       start=_walk_subst_expr(e.start, subst) if e.start is not None else None,
                       end=_walk_subst_expr(e.end, subst) if e.end is not None else None)
    if isinstance(e, A.Quote):
        return A.Quote(span=e.span, inner=_walk_subst_expr(e.inner, subst))
    if isinstance(e, A.Splice):
        return A.Splice(span=e.span, inner=_walk_subst_expr(e.inner, subst))
    if isinstance(e, A.Modify):
        return A.Modify(span=e.span,
                        target=_walk_subst_expr(e.target, subst),
                        transformation=_walk_subst_expr(e.transformation, subst),
                        verifier=_walk_subst_expr(e.verifier, subst))
    # Audit 28.8 B12: UnsafeBlock — generics through unsafe regions
    # need substitution. Pre-fix, `fn read<T>(p: *const T) -> T {
    # unsafe { let x: T = *p; x } }` mono'd against T=f64 left the
    # inner `let x: T` unsubstituted; downstream lower-ast resolved
    # T as TyUnknown and silently defaulted the read to i32.
    if isinstance(e, A.UnsafeBlock):
        return A.UnsafeBlock(span=e.span,
                             body=_walk_subst_expr(e.body, subst))
    return e


def _walk_subst_stmt(s: A.Stmt, subst: dict[str, A.TyNode]) -> A.Stmt:
    if isinstance(s, A.Let):
        return A.Let(span=s.span, name=s.name, is_mut=s.is_mut,
                     ty=substitute_ty(s.ty, subst) if s.ty is not None else None,
                     value=_walk_subst_expr(s.value, subst) if s.value is not None else None)
    if isinstance(s, A.ExprStmt):
        return A.ExprStmt(span=s.span, expr=_walk_subst_expr(s.expr, subst))
    if isinstance(s, A.ConstStmt):
        return A.ConstStmt(span=s.span, name=s.name,
                           ty=substitute_ty(s.ty, subst),
                           value=_walk_subst_expr(s.value, subst))
    return s


# ============================================================================
# Call-site rewriting + instantiation
# ============================================================================
class Monomorphizer:
    def __init__(self, prog: A.Program):
        self.prog = prog
        self.generic_fns: dict[str, A.FnDecl] = {}
        for item in prog.items:
            if isinstance(item, A.FnDecl) and item.generics:
                self.generic_fns[item.name] = item
        self.instantiated: dict[tuple[str, str], A.FnDecl] = {}

    def run(self) -> int:
        """Run monomorphization. Returns count of new fns added.

        Generic fns with at least one turbofish call site get cloned per
        unique type-arg-tuple. Generic fns without any turbofish call sites
        (only un-annotated calls like `identity(42)`) are left in place so
        the legacy "T silently i32" lower path still works for backward
        compatibility.
        """
        if not self.generic_fns:
            return 0
        # Iteratively rewrite calls. Each pass may add new fns whose bodies
        # contain further generic calls. Fixed point when no new instantiations.
        changed = True
        while changed:
            changed = False
            for item in list(self.prog.items):
                if isinstance(item, A.FnDecl):
                    new_body = self._rewrite_calls_in_block(item.body, item)
                    if new_body is not item.body:
                        item.body = new_body
                        changed = True
        # Append instantiated clones; keep original generic fns intact so
        # legacy un-turbofished call sites keep resolving.
        added = len(self.instantiated)
        self.prog.items = list(self.prog.items) + list(self.instantiated.values())
        return added

    def _rewrite_calls_in_block(self, blk: A.Block, owner: A.FnDecl) -> A.Block:
        new_stmts = []
        any_change = False
        for s in blk.stmts:
            ns = self._rewrite_calls_in_stmt(s, owner)
            if ns is not s:
                any_change = True
            new_stmts.append(ns)
        new_final = blk.final_expr
        if blk.final_expr is not None:
            new_final = self._rewrite_calls_in_expr(blk.final_expr, owner)
            if new_final is not blk.final_expr:
                any_change = True
        if any_change:
            return A.Block(span=blk.span, stmts=new_stmts, final_expr=new_final)
        return blk

    def _rewrite_calls_in_stmt(self, s: A.Stmt, owner: A.FnDecl) -> A.Stmt:
        if isinstance(s, A.Let):
            new_value = s.value
            if s.value is not None:
                new_value = self._rewrite_calls_in_expr(s.value, owner)
            if new_value is s.value:
                return s
            return A.Let(span=s.span, name=s.name, is_mut=s.is_mut, ty=s.ty, value=new_value)
        if isinstance(s, A.ExprStmt):
            new_e = self._rewrite_calls_in_expr(s.expr, owner)
            if new_e is s.expr:
                return s
            return A.ExprStmt(span=s.span, expr=new_e)
        if isinstance(s, A.ConstStmt):
            new_e = self._rewrite_calls_in_expr(s.value, owner)
            if new_e is s.value:
                return s
            return A.ConstStmt(span=s.span, name=s.name, ty=s.ty, value=new_e)
        return s

    def _rewrite_calls_in_expr(self, e: A.Expr, owner: A.FnDecl) -> A.Expr:
        if isinstance(e, A.Call):
            new_callee = self._rewrite_calls_in_expr(e.callee, owner)
            new_args = [self._rewrite_calls_in_expr(a, owner) for a in e.args]
            # Detect generic call site
            if isinstance(new_callee, A.Name) and new_callee.generics and new_callee.name in self.generic_fns:
                fn = self.generic_fns[new_callee.name]
                if len(new_callee.generics) != len(fn.generics):
                    # Mismatch — leave alone, typecheck will flag
                    return A.Call(span=e.span, callee=new_callee, args=new_args)
                # Build substitution map
                subst: dict[str, A.TyNode] = {}
                for gp, ty_arg in zip(fn.generics, new_callee.generics):
                    subst[gp.name] = ty_arg
                mangled = mangle(fn.name, new_callee.generics)
                key = (fn.name, mangled)
                if key not in self.instantiated:
                    self.instantiated[key] = self._instantiate(fn, mangled, subst)
                # Replace callee with non-generic Name
                new_callee = A.Name(span=new_callee.span, name=mangled, generics=[])
                return A.Call(span=e.span, callee=new_callee, args=new_args)
            if new_callee is e.callee and all(a is b for a, b in zip(new_args, e.args)):
                return e
            return A.Call(span=e.span, callee=new_callee, args=new_args)
        if isinstance(e, A.Block):
            return self._rewrite_calls_in_block(e, owner)
        if isinstance(e, A.If):
            new_cond = self._rewrite_calls_in_expr(e.cond, owner)
            new_then = self._rewrite_calls_in_expr(e.then, owner)
            new_else = e.else_
            if new_else is not None:
                new_else = self._rewrite_calls_in_expr(new_else, owner)
            if new_cond is e.cond and new_then is e.then and new_else is e.else_:
                return e
            return A.If(span=e.span, cond=new_cond, then=new_then, else_=new_else)
        if isinstance(e, A.Match):
            new_scrut = self._rewrite_calls_in_expr(e.scrutinee, owner)
            new_arms = []
            any_change = new_scrut is not e.scrutinee
            for arm in e.arms:
                new_body = self._rewrite_calls_in_expr(arm.body, owner)
                new_guard = arm.guard
                if arm.guard is not None:
                    new_guard = self._rewrite_calls_in_expr(arm.guard, owner)
                if new_body is arm.body and new_guard is arm.guard:
                    new_arms.append(arm)
                else:
                    any_change = True
                    new_arms.append(A.MatchArm(span=arm.span, pattern=arm.pattern,
                                               guard=new_guard, body=new_body))
            if any_change:
                return A.Match(span=e.span, scrutinee=new_scrut, arms=new_arms)
            return e
        if isinstance(e, A.For):
            new_it = self._rewrite_calls_in_expr(e.iter_expr, owner)
            new_body = self._rewrite_calls_in_expr(e.body, owner)
            if new_it is e.iter_expr and new_body is e.body:
                return e
            return A.For(span=e.span, var_name=e.var_name, iter_expr=new_it, body=new_body)
        if isinstance(e, A.While):
            new_cond = self._rewrite_calls_in_expr(e.cond, owner)
            new_body = self._rewrite_calls_in_expr(e.body, owner)
            if new_cond is e.cond and new_body is e.body:
                return e
            return A.While(span=e.span, cond=new_cond, body=new_body)
        if isinstance(e, A.Loop):
            new_body = self._rewrite_calls_in_expr(e.body, owner)
            if new_body is e.body:
                return e
            return A.Loop(span=e.span, body=new_body)
        if isinstance(e, A.Binary):
            new_l = self._rewrite_calls_in_expr(e.left, owner)
            new_r = self._rewrite_calls_in_expr(e.right, owner)
            if new_l is e.left and new_r is e.right:
                return e
            return A.Binary(span=e.span, op=e.op, left=new_l, right=new_r)
        if isinstance(e, A.Unary):
            new_op = self._rewrite_calls_in_expr(e.operand, owner)
            if new_op is e.operand:
                return e
            return A.Unary(span=e.span, op=e.op, operand=new_op)
        if isinstance(e, A.Cast):
            new_v = self._rewrite_calls_in_expr(e.value, owner)
            if new_v is e.value:
                return e
            return A.Cast(span=e.span, value=new_v, target_ty=e.target_ty)
        if isinstance(e, A.Index):
            new_callee = self._rewrite_calls_in_expr(e.callee, owner)
            new_idx = [self._rewrite_calls_in_expr(i, owner) for i in e.indices]
            if new_callee is e.callee and all(a is b for a, b in zip(new_idx, e.indices)):
                return e
            return A.Index(span=e.span, callee=new_callee, indices=new_idx)
        if isinstance(e, A.Field):
            new_obj = self._rewrite_calls_in_expr(e.obj, owner)
            if new_obj is e.obj:
                return e
            return A.Field(span=e.span, obj=new_obj, name=e.name)
        if isinstance(e, A.TupleLit):
            new_elems = [self._rewrite_calls_in_expr(x, owner) for x in e.elems]
            if all(a is b for a, b in zip(new_elems, e.elems)):
                return e
            return A.TupleLit(span=e.span, elems=new_elems)
        if isinstance(e, A.ArrayLit):
            new_elems = [self._rewrite_calls_in_expr(x, owner) for x in e.elems]
            if all(a is b for a, b in zip(new_elems, e.elems)):
                return e
            return A.ArrayLit(span=e.span, elems=new_elems)
        if isinstance(e, A.StructLit):
            new_fields = [(n, self._rewrite_calls_in_expr(v, owner)) for (n, v) in e.fields]
            if all(nv is ov for (_, nv), (_, ov) in zip(new_fields, e.fields)):
                return e
            return A.StructLit(span=e.span, name=e.name, fields=new_fields)
        if isinstance(e, A.Assign):
            new_t = self._rewrite_calls_in_expr(e.target, owner)
            new_v = self._rewrite_calls_in_expr(e.value, owner)
            if new_t is e.target and new_v is e.value:
                return e
            return A.Assign(span=e.span, target=new_t, op=e.op, value=new_v)
        if isinstance(e, A.Return):
            new_v = e.value
            if e.value is not None:
                new_v = self._rewrite_calls_in_expr(e.value, owner)
            if new_v is e.value:
                return e
            return A.Return(span=e.span, value=new_v)
        if isinstance(e, A.Break):
            new_v = e.value
            if e.value is not None:
                new_v = self._rewrite_calls_in_expr(e.value, owner)
            if new_v is e.value:
                return e
            return A.Break(span=e.span, value=new_v)
        if isinstance(e, A.Range):
            new_s = e.start
            new_end = e.end
            if e.start is not None:
                new_s = self._rewrite_calls_in_expr(e.start, owner)
            if e.end is not None:
                new_end = self._rewrite_calls_in_expr(e.end, owner)
            if new_s is e.start and new_end is e.end:
                return e
            return A.Range(span=e.span, start=new_s, end=new_end)
        if isinstance(e, A.Quote):
            new_inner = self._rewrite_calls_in_expr(e.inner, owner)
            if new_inner is e.inner:
                return e
            return A.Quote(span=e.span, inner=new_inner)
        if isinstance(e, A.Splice):
            new_inner = self._rewrite_calls_in_expr(e.inner, owner)
            if new_inner is e.inner:
                return e
            return A.Splice(span=e.span, inner=new_inner)
        if isinstance(e, A.Modify):
            new_t = self._rewrite_calls_in_expr(e.target, owner)
            new_tr = self._rewrite_calls_in_expr(e.transformation, owner)
            new_v = self._rewrite_calls_in_expr(e.verifier, owner)
            if new_t is e.target and new_tr is e.transformation and new_v is e.verifier:
                return e
            return A.Modify(span=e.span, target=new_t, transformation=new_tr, verifier=new_v)
        return e

    def _instantiate(self, fn: A.FnDecl, mangled: str, subst: dict[str, A.TyNode]) -> A.FnDecl:
        """Clone fn, substitute T-vars with concrete types, drop generics.

        Audit 28.8 B7: deep-copies where_clauses with substitution
        applied per clause's constraint expression. Pre-fix the clone
        shared the template's `where_clauses` list directly — so
        downstream passes that consumed the clone saw the template's
        unsubstituted clauses (e.g. `where T: Eq` with TyName('T')
        still present).

        Also propagates `is_extern` and `extern_abi`. Pre-fix the
        clone was implicitly non-extern, which meant
        `extern "C" fn malloc<T>(n: usize) -> *mut T` mono'd to
        `malloc__i32` would have been emitted as a normal user-fn
        with empty body → ud2 trap at runtime. Phase-0 still treats
        generic-over-extern as a parse-time refusal upstream, but
        the propagation is correct here for completeness."""
        new_params = [
            A.FnParam(span=p.span, name=p.name,
                      ty=substitute_ty(p.ty, subst), is_mut=p.is_mut)
            for p in fn.params
        ]
        new_ret = substitute_ty(fn.return_ty, subst) if fn.return_ty is not None else None
        new_body_block = _walk_subst_expr(fn.body, subst)
        if not isinstance(new_body_block, A.Block):
            new_body_block = fn.body
        new_where = [
            A.WhereClause(
                span=w.span,
                constraint=_walk_subst_expr(w.constraint, subst),
            )
            for w in fn.where_clauses
        ]
        return A.FnDecl(
            span=fn.span,
            name=mangled,
            generics=[],
            params=new_params,
            return_ty=new_ret,
            where_clauses=new_where,
            body=new_body_block,
            attrs=list(fn.attrs),
            is_pub=fn.is_pub,
            is_extern=fn.is_extern,
            extern_abi=fn.extern_abi,
        )


def monomorphize(prog: A.Program) -> int:
    """Run monomorphization on a program. Returns count of fns added."""
    return Monomorphizer(prog).run()
