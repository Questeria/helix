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


TRAP_DUPLICATE_METHOD_NAME = 74002


class DuplicateMethodError(Exception):
    """Audit 28.8 B11 (trap 74002): two distinct structs declare a
    method with the same name. Phase-0 fallback (until static dispatch
    by self-type ships in v0.2) is to reject — the first registration
    wins and the second silently aliased to the same call site,
    producing cross-struct type confusion (`Pt.area` and `Line.area`
    both rewriting `_.area()` to `Pt__area`)."""
    trap_id = TRAP_DUPLICATE_METHOD_NAME

    def __init__(self, method: str, first_target: str, second_target: str,
                 span: A.Span):
        super().__init__(
            f"{span.line}:{span.col}: duplicate method name {method!r} — "
            f"declared on both {first_target!r} and {second_target!r} "
            f"(trap 74002)"
        )
        self.method = method
        self.first_target = first_target
        self.second_target = second_target
        self.span = span


def flatten_impls(prog: A.Program) -> int:
    """Lift impl-block methods to top level. Rewrite x.method(args) calls.
    Returns count of methods lifted.

    Audit 28.8 B11 (trap 74002): Phase-0 same-name-method dispatch is
    ambiguous because the rewrite uses a flat name->target map. Until
    static dispatch by self-type ships in v0.2, we REJECT the second
    registration with a DuplicateMethodError. Pre-fix the dispatch
    picked the first-registered target silently — calling `line_var
    .area()` (where Line and Pt both have an `area` method) would
    rewrite to `Pt__area(line_var)`, passing a Line value into a
    Pt-typed receiver. SEGV or silent wrong data.

    Stage 65 Inc 1 — Tier 4 #17 multiple dispatch scaffolding:
    The registration table is now `dict[str, list[str]]` (method_name
    → list of impl targets in declaration order). Single-candidate
    dispatch behaviour is preserved; multi-candidate dispatch still
    raises DuplicateMethodError at rewrite time (fail-closed). Inc 2
    will add type-driven dispatch by inspecting the receiver
    expression's static type and selecting the matching target.
    """
    methods_lifted = 0
    new_items: list[A.Item] = []
    # Stage 65 Inc 1 — multi-target registration.
    method_to_targets: dict[str, list[str]] = {}
    # Track first-registration span for diagnostic continuity.
    method_first_span: dict[str, A.Span] = {}
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
                targets = method_to_targets.setdefault(m.name, [])
                if item.target not in targets:
                    if len(targets) == 0:
                        method_first_span[m.name] = m.span
                        targets.append(item.target)
                    else:
                        # Stage 65 Inc 1 — multi-target registration
                        # tracking is in place, but call-site dispatch
                        # by receiver type is deferred to Inc 2 (needs
                        # typecheck integration). For now, preserve
                        # the Audit 28.8 B11 fail-closed at registration:
                        # second target raises DuplicateMethodError.
                        # Inc 2 will opt into multi-dispatch via an
                        # @overload-like attribute on the impl block.
                        raise DuplicateMethodError(
                            method=m.name,
                            first_target=targets[0],
                            second_target=item.target,
                            span=m.span,
                        )
                # Same-target re-declaration (impl X { fn f() } twice)
                # is a separate concern handled by typecheck's duplicate
                # fn-name check; here we just don't double-register.
        else:
            new_items.append(item)
    prog.items = new_items
    if methods_lifted:
        _rewrite_method_calls(prog, method_to_targets,
                                method_first_span)
    return methods_lifted


# Stage 65 Inc 1 — module-level state for first-registration span
# (used by DuplicateMethodError diagnostic). Set by flatten_impls
# before walking and cleared after.
_FIRST_SPAN: "dict[str, A.Span]" = {}


def _rewrite_method_calls(prog: A.Program,
                            m2t: "dict[str, list[str]]",
                            first_span: "dict[str, A.Span] | None" = None
                            ) -> None:
    # Stage 65 Inc 1 — multi-target dispatch scaffolding.
    global _FIRST_SPAN
    _FIRST_SPAN = first_span or {}
    try:
        for item in prog.items:
            if isinstance(item, A.FnDecl):
                item.body = _rewrite_expr(item.body, m2t)
    finally:
        _FIRST_SPAN = {}


def _resolve_method_target(method_name: str,
                            m2t: "dict[str, list[str]]",
                            call_span: A.Span) -> str:
    """Stage 65 Inc 1 — pick the single target for a method-call
    rewrite. Raises DuplicateMethodError when ambiguous (multiple
    targets registered, no type-driven dispatch yet). Inc 2 will
    add type-driven dispatch by inspecting the receiver's static
    type."""
    targets = m2t.get(method_name) or []
    if len(targets) == 1:
        return targets[0]
    if len(targets) >= 2:
        sp = _FIRST_SPAN.get(method_name, call_span)
        raise DuplicateMethodError(
            method=method_name,
            first_target=targets[0],
            second_target=targets[1],
            span=sp,
        )
    return ""


def _rewrite_expr(e: A.Expr, m2t: "dict[str, list[str]]") -> A.Expr:
    if isinstance(e, A.Call):
        new_args = [_rewrite_expr(a, m2t) for a in e.args]
        # Method-call: Call(callee=Field(obj, name), args)
        if isinstance(e.callee, A.Field) and e.callee.name in m2t:
            target = _resolve_method_target(
                e.callee.name, m2t, e.span)
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
    # Stage 28.9 cycle 57 C57-1 (HIGH, conf 88): pre-fix this walker
    # was missing UnsafeBlock + TileLit arms. A method call inside
    # `unsafe { p.area() }` left p.area unrewritten and typecheck
    # failed with a misleading "struct has no field 'area'" error.
    # Same defect class as match_lower cycle-23 C22-C / cycle-7 C7-1.
    if isinstance(e, A.UnsafeBlock):
        # UnsafeBlock.body is typed Block; the Block arm above returns
        # A.Block, so a single recursive call preserves the field type.
        new_body = _rewrite_expr(e.body, m2t)
        assert isinstance(new_body, A.Block), \
            "flatten_impls: UnsafeBlock.body rewrite must return Block"
        return A.UnsafeBlock(span=e.span, body=new_body)
    if isinstance(e, A.TileLit):
        return A.TileLit(
            span=e.span, dtype=e.dtype,
            shape=[_rewrite_expr(s, m2t) for s in e.shape],
            memspace=_rewrite_expr(e.memspace, m2t),
            init=e.init,
        )
    # Cycle 57 C57-1 catchall — leaf expression types pass through
    # explicitly (lit, ref); anything else is an unmapped Expr
    # subclass and must be wired in.
    if isinstance(e, (A.IntLit, A.FloatLit, A.StrLit, A.CharLit,
                       A.BoolLit, A.Name, A.Path, A.Continue)):
        return e
    raise NotImplementedError(
        f"flatten_impls._rewrite_expr: unhandled expression kind "
        f"{type(e).__name__} at {getattr(e, 'span', '?')!r}. "
        f"Mirror cycle-57 C57-1 discipline: add an explicit arm."
    )


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
