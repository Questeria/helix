"""
helixc/frontend/flatten_impls.py

Method-call dispatch via inherent impl blocks (Phase 1.8).

Algorithm:
  1. Collect every `impl Type { fn method(self, ...) ... }` and lift its
     methods to top-level fns named `Type__method`. Keep a global table
     method_name -> list of types-with-that-method (for diagnostics).
  2. Walk every fn body. For Calls of shape Call(callee=Field(obj, name), args)
     where `name` matches some impl-block method, rewrite to:
       Call(callee=Name("Type__name"), [obj] + args)
     Disambiguation: if multiple types have a method with the same name,
     resolution picks the FIRST registered type (registration order). For
     v0.1 we just emit and let the unresolved-symbol error trigger if
     the user-side type doesn't match — Phase 1.8 doesn't yet do real
     type-based dispatch.

Trait impls (`impl Trait for Type`) flow through the same path: the
methods are flattened to `Type__method` (the trait name is dropped at
this pass). Trait dispatch with multi-impl resolution is a Phase 2 item.

License: Apache 2.0
"""

from __future__ import annotations
from typing import Optional

from . import ast_nodes as A


def flatten_impls(prog: A.Program) -> int:
    """Lift impl-block methods to top level. Rewrite x.method(args) calls.
    Returns count of methods lifted."""
    methods_lifted = 0
    new_items: list[A.Item] = []
    method_to_target: dict[str, str] = {}  # method_name -> target type
    for item in prog.items:
        if isinstance(item, A.ImplBlock):
            for m in item.methods:
                new_name = item.target + "__" + m.name
                lifted = A.FnDecl(
                    span=m.span, name=new_name, generics=m.generics,
                    params=m.params, return_ty=m.return_ty,
                    where_clauses=m.where_clauses, body=m.body,
                    attrs=m.attrs, is_pub=m.is_pub,
                )
                new_items.append(lifted)
                methods_lifted += 1
                method_to_target.setdefault(m.name, item.target)
        else:
            new_items.append(item)
    prog.items = new_items
    if methods_lifted:
        _rewrite_method_calls(prog, method_to_target)
    return methods_lifted


def _rewrite_method_calls(prog: A.Program, m2t: dict[str, str]) -> None:
    for item in prog.items:
        if isinstance(item, A.FnDecl):
            item.body = _rewrite_expr(item.body, m2t)


def _rewrite_expr(e: A.Expr, m2t: dict[str, str]) -> A.Expr:
    if isinstance(e, A.Call):
        new_args = [_rewrite_expr(a, m2t) for a in e.args]
        # Method-call: Call(callee=Field(obj, name), args)
        if isinstance(e.callee, A.Field) and e.callee.name in m2t:
            target = m2t[e.callee.name]
            new_callee = A.Name(span=e.callee.span,
                                name=target + "__" + e.callee.name,
                                generics=[])
            new_self = _rewrite_expr(e.callee.obj, m2t)
            return A.Call(span=e.span, callee=new_callee,
                          args=[new_self] + new_args)
        new_callee = _rewrite_expr(e.callee, m2t)
        return A.Call(span=e.span, callee=new_callee, args=new_args)
    if isinstance(e, A.Block):
        return A.Block(span=e.span,
                       stmts=[_rewrite_stmt(s, m2t) for s in e.stmts],
                       final_expr=_rewrite_expr(e.final_expr, m2t) if e.final_expr is not None else None)
    if isinstance(e, A.If):
        else_ = e.else_
        if else_ is not None:
            else_ = _rewrite_expr(else_, m2t)
        return A.If(span=e.span,
                    cond=_rewrite_expr(e.cond, m2t),
                    then=_rewrite_expr(e.then, m2t),
                    else_=else_)
    if isinstance(e, A.Match):
        return A.Match(span=e.span,
                       scrutinee=_rewrite_expr(e.scrutinee, m2t),
                       arms=[A.MatchArm(span=arm.span, pattern=arm.pattern,
                                        guard=_rewrite_expr(arm.guard, m2t) if arm.guard else None,
                                        body=_rewrite_expr(arm.body, m2t)) for arm in e.arms])
    if isinstance(e, A.For):
        return A.For(span=e.span, var_name=e.var_name,
                     iter_expr=_rewrite_expr(e.iter_expr, m2t),
                     body=_rewrite_expr(e.body, m2t))
    if isinstance(e, A.While):
        return A.While(span=e.span,
                       cond=_rewrite_expr(e.cond, m2t),
                       body=_rewrite_expr(e.body, m2t))
    if isinstance(e, A.Loop):
        return A.Loop(span=e.span, body=_rewrite_expr(e.body, m2t))
    if isinstance(e, A.Binary):
        return A.Binary(span=e.span, op=e.op,
                        left=_rewrite_expr(e.left, m2t),
                        right=_rewrite_expr(e.right, m2t))
    if isinstance(e, A.Unary):
        return A.Unary(span=e.span, op=e.op, operand=_rewrite_expr(e.operand, m2t))
    if isinstance(e, A.Cast):
        return A.Cast(span=e.span, value=_rewrite_expr(e.value, m2t), target_ty=e.target_ty)
    if isinstance(e, A.Index):
        return A.Index(span=e.span, callee=_rewrite_expr(e.callee, m2t),
                       indices=[_rewrite_expr(i, m2t) for i in e.indices])
    if isinstance(e, A.Field):
        return A.Field(span=e.span, obj=_rewrite_expr(e.obj, m2t), name=e.name)
    if isinstance(e, A.TupleLit):
        return A.TupleLit(span=e.span, elems=[_rewrite_expr(x, m2t) for x in e.elems])
    if isinstance(e, A.ArrayLit):
        return A.ArrayLit(span=e.span, elems=[_rewrite_expr(x, m2t) for x in e.elems])
    if isinstance(e, A.StructLit):
        return A.StructLit(span=e.span, name=e.name,
                           fields=[(n, _rewrite_expr(v, m2t)) for (n, v) in e.fields])
    if isinstance(e, A.Assign):
        return A.Assign(span=e.span, target=_rewrite_expr(e.target, m2t),
                        op=e.op, value=_rewrite_expr(e.value, m2t))
    if isinstance(e, A.Return):
        return A.Return(span=e.span,
                        value=_rewrite_expr(e.value, m2t) if e.value is not None else None)
    if isinstance(e, A.Break):
        return A.Break(span=e.span,
                       value=_rewrite_expr(e.value, m2t) if e.value is not None else None)
    if isinstance(e, A.Range):
        return A.Range(span=e.span,
                       start=_rewrite_expr(e.start, m2t) if e.start is not None else None,
                       end=_rewrite_expr(e.end, m2t) if e.end is not None else None)
    if isinstance(e, A.Quote):
        return A.Quote(span=e.span, inner=_rewrite_expr(e.inner, m2t))
    if isinstance(e, A.Splice):
        return A.Splice(span=e.span, inner=_rewrite_expr(e.inner, m2t))
    if isinstance(e, A.Modify):
        return A.Modify(span=e.span, target=_rewrite_expr(e.target, m2t),
                        transformation=_rewrite_expr(e.transformation, m2t),
                        verifier=_rewrite_expr(e.verifier, m2t))
    return e


def _rewrite_stmt(s: A.Stmt, m2t: dict[str, str]) -> A.Stmt:
    if isinstance(s, A.Let):
        return A.Let(span=s.span, name=s.name, is_mut=s.is_mut, ty=s.ty,
                     value=_rewrite_expr(s.value, m2t) if s.value is not None else None)
    if isinstance(s, A.ExprStmt):
        return A.ExprStmt(span=s.span, expr=_rewrite_expr(s.expr, m2t))
    if isinstance(s, A.ConstStmt):
        return A.ConstStmt(span=s.span, name=s.name, ty=s.ty,
                           value=_rewrite_expr(s.value, m2t))
    return s
