"""
helixc/frontend/flatten_modules.py

Module-flattening pass: lifts items inside `mod foo { ... }` block modules
to the top level with mangled names (foo__name), and rewrites every call
of the form `foo::name(args)` to `foo__name(args)`.

Runs BEFORE monomorphize, so monomorphized clones use the flattened
mangled names.

Mangling scheme:
    mod foo { fn bar() { ... } }       ->  fn foo__bar() { ... }
    mod foo { mod inner { fn baz() { ... } } }
                                        ->  fn foo__inner__baz() { ... }
    foo::bar(x)                         ->  foo__bar(x)
    foo::inner::baz(x)                  ->  foo__inner__baz(x)

The pass also handles:
- ConstDecls and StructDecls inside modules (renamed similarly).
- `use foo::bar;` decls that bring `foo__bar` into scope as `bar` for
  unqualified calls within the importing scope.

License: Apache 2.0
"""

from __future__ import annotations
from typing import Optional

from . import ast_nodes as A


class FlattenError(Exception):
    """Raised when module flattening detects an unresolvable condition
    (mangled-name collision per audit-stage9-16-codegen CRITICAL-3, or
    use-decl pointing at unknown target per MEDIUM-3)."""
    pass


def flatten_modules(prog: A.Program) -> int:
    """Lift block modules to top level, mangling names. Returns count of
    items lifted out of modules."""
    flattened = 0
    new_items: list[A.Item] = []
    aliases: dict[str, str] = {}  # alias from `use foo::bar` -> "foo__bar"
    for item in prog.items:
        if isinstance(item, A.ModBlock):
            n = _flatten_one(item, prefix="", new_items=new_items)
            flattened += n
        elif isinstance(item, A.UseDecl):
            # `use foo::bar` registers `bar` -> `foo__bar` alias.
            if len(item.path) >= 2:
                last = item.path[-1]
                full = "__".join(item.path)
                aliases[last] = full
            new_items.append(item)
        else:
            new_items.append(item)
    # Audit follow-up CRITICAL-3 (Stages 9-16 audit): detect mangled-name
    # collisions BEFORE handing the AST to the lowerer (which would
    # silently overwrite the first definition with the second). Trap 78001.
    seen_names: dict[str, A.Item] = {}
    for it in new_items:
        nm = getattr(it, "name", None)
        if nm is None:
            continue
        if nm in seen_names:
            raise FlattenError(
                f"name collision after module flattening: '{nm}' is defined "
                f"more than once (trap 78001). One source is at "
                f"{getattr(seen_names[nm], 'span', None)}; the other at "
                f"{getattr(it, 'span', None)}.")
        seen_names[nm] = it
    # Audit follow-up MEDIUM-3: verify each `use` alias resolves to an
    # actual top-level item. Trap 79001.
    for alias_src, alias_tgt in aliases.items():
        if alias_tgt not in seen_names:
            raise FlattenError(
                f"use-decl '{alias_src}' resolves to unknown name '{alias_tgt}' "
                f"(path '{alias_tgt.replace('__', '::')}'; trap 79001)")
    prog.items = new_items
    if flattened or aliases:
        _rewrite_calls(prog, aliases)
    return flattened


def _flatten_one(mb: A.ModBlock, prefix: str, new_items: list[A.Item]) -> int:
    base = (prefix + "__" if prefix else "") + mb.name
    n = 0
    for sub in mb.items:
        if isinstance(sub, A.ModBlock):
            n += _flatten_one(sub, prefix=base, new_items=new_items)
        elif isinstance(sub, A.FnDecl):
            new_name = base + "__" + sub.name
            new_items.append(A.FnDecl(
                span=sub.span, name=new_name, generics=sub.generics,
                params=sub.params, return_ty=sub.return_ty,
                where_clauses=sub.where_clauses, body=sub.body,
                attrs=sub.attrs, is_pub=sub.is_pub))
            n += 1
        elif isinstance(sub, A.StructDecl):
            new_name = base + "__" + sub.name
            new_items.append(A.StructDecl(
                span=sub.span, name=new_name, generics=sub.generics,
                fields=sub.fields, is_pub=sub.is_pub))
            n += 1
        elif isinstance(sub, A.EnumDecl):
            new_name = base + "__" + sub.name
            new_items.append(A.EnumDecl(
                span=sub.span, name=new_name, generics=sub.generics,
                variants=sub.variants, is_pub=sub.is_pub))
            n += 1
        elif isinstance(sub, A.ConstDecl):
            new_name = base + "__" + sub.name
            new_items.append(A.ConstDecl(
                span=sub.span, name=new_name, ty=sub.ty,
                value=sub.value, is_pub=sub.is_pub))
            n += 1
        elif isinstance(sub, A.TypeAlias):
            new_name = base + "__" + sub.name
            new_items.append(A.TypeAlias(
                span=sub.span, name=new_name, generics=sub.generics,
                target=sub.target, is_pub=sub.is_pub))
            n += 1
        elif isinstance(sub, A.UseDecl):
            new_items.append(sub)
        else:
            new_items.append(sub)
    return n


def _rewrite_calls(prog: A.Program, aliases: dict[str, str]) -> None:
    """Walk every FnDecl body, rewriting:
       - `foo::bar(args)` Path-call -> `foo__bar(args)` Name-call
       - `bar(args)` where `bar` is in aliases -> `<alias>(args)` Name-call
    """
    for item in prog.items:
        if isinstance(item, A.FnDecl):
            item.body = _rewrite_expr(item.body, aliases)


def _rewrite_expr(e: A.Expr, aliases: dict[str, str]) -> A.Expr:
    if isinstance(e, A.Call):
        new_callee = _rewrite_callee(e.callee, aliases)
        new_args = [_rewrite_expr(a, aliases) for a in e.args]
        return A.Call(span=e.span, callee=new_callee, args=new_args)
    if isinstance(e, A.Block):
        return A.Block(span=e.span,
                       stmts=[_rewrite_stmt(s, aliases) for s in e.stmts],
                       final_expr=_rewrite_expr(e.final_expr, aliases) if e.final_expr is not None else None)
    if isinstance(e, A.If):
        else_ = e.else_
        if else_ is not None:
            else_ = _rewrite_expr(else_, aliases)
        return A.If(span=e.span,
                    cond=_rewrite_expr(e.cond, aliases),
                    then=_rewrite_expr(e.then, aliases),
                    else_=else_)
    if isinstance(e, A.Match):
        return A.Match(span=e.span,
                       scrutinee=_rewrite_expr(e.scrutinee, aliases),
                       arms=[A.MatchArm(span=arm.span, pattern=arm.pattern,
                                        guard=_rewrite_expr(arm.guard, aliases) if arm.guard else None,
                                        body=_rewrite_expr(arm.body, aliases)) for arm in e.arms])
    if isinstance(e, A.For):
        return A.For(span=e.span, var_name=e.var_name,
                     iter_expr=_rewrite_expr(e.iter_expr, aliases),
                     body=_rewrite_expr(e.body, aliases))
    if isinstance(e, A.While):
        return A.While(span=e.span,
                       cond=_rewrite_expr(e.cond, aliases),
                       body=_rewrite_expr(e.body, aliases))
    if isinstance(e, A.Loop):
        return A.Loop(span=e.span, body=_rewrite_expr(e.body, aliases))
    if isinstance(e, A.Binary):
        return A.Binary(span=e.span, op=e.op,
                        left=_rewrite_expr(e.left, aliases),
                        right=_rewrite_expr(e.right, aliases))
    if isinstance(e, A.Unary):
        return A.Unary(span=e.span, op=e.op, operand=_rewrite_expr(e.operand, aliases))
    if isinstance(e, A.Cast):
        return A.Cast(span=e.span, value=_rewrite_expr(e.value, aliases), target_ty=e.target_ty)
    if isinstance(e, A.Index):
        return A.Index(span=e.span, callee=_rewrite_expr(e.callee, aliases),
                       indices=[_rewrite_expr(i, aliases) for i in e.indices])
    if isinstance(e, A.Field):
        return A.Field(span=e.span, obj=_rewrite_expr(e.obj, aliases), name=e.name)
    if isinstance(e, A.TupleLit):
        return A.TupleLit(span=e.span, elems=[_rewrite_expr(x, aliases) for x in e.elems])
    if isinstance(e, A.ArrayLit):
        return A.ArrayLit(span=e.span, elems=[_rewrite_expr(x, aliases) for x in e.elems])
    if isinstance(e, A.StructLit):
        return A.StructLit(span=e.span, name=e.name,
                           fields=[(n, _rewrite_expr(v, aliases)) for (n, v) in e.fields])
    if isinstance(e, A.Assign):
        return A.Assign(span=e.span, target=_rewrite_expr(e.target, aliases),
                        op=e.op, value=_rewrite_expr(e.value, aliases))
    if isinstance(e, A.Return):
        return A.Return(span=e.span,
                        value=_rewrite_expr(e.value, aliases) if e.value is not None else None)
    if isinstance(e, A.Break):
        return A.Break(span=e.span,
                       value=_rewrite_expr(e.value, aliases) if e.value is not None else None)
    if isinstance(e, A.Range):
        return A.Range(span=e.span,
                       start=_rewrite_expr(e.start, aliases) if e.start is not None else None,
                       end=_rewrite_expr(e.end, aliases) if e.end is not None else None)
    if isinstance(e, A.Quote):
        return A.Quote(span=e.span, inner=_rewrite_expr(e.inner, aliases))
    if isinstance(e, A.Splice):
        return A.Splice(span=e.span, inner=_rewrite_expr(e.inner, aliases))
    if isinstance(e, A.Modify):
        return A.Modify(span=e.span, target=_rewrite_expr(e.target, aliases),
                        transformation=_rewrite_expr(e.transformation, aliases),
                        verifier=_rewrite_expr(e.verifier, aliases))
    # Stage 28.9 cycle 57 C57-2 (HIGH, conf 86): pre-fix this walker
    # was missing UnsafeBlock + TileLit arms. `use foo::bar; fn main()
    # { unsafe { bar() } }` left the `bar` callee un-aliased; the
    # subsequent name-resolution pass failed with a misleading
    # "unknown function 'bar'" instead of routing to foo__bar. Same
    # defect class as match_lower C22-C and flatten_impls C57-1.
    if isinstance(e, A.UnsafeBlock):
        new_body = _rewrite_expr(e.body, aliases)
        assert isinstance(new_body, A.Block), \
            "flatten_modules: UnsafeBlock.body rewrite must return Block"
        return A.UnsafeBlock(span=e.span, body=new_body)
    if isinstance(e, A.TileLit):
        return A.TileLit(
            span=e.span, dtype=e.dtype,
            shape=[_rewrite_expr(s, aliases) for s in e.shape],
            memspace=_rewrite_expr(e.memspace, aliases),
            init=e.init,
        )
    if isinstance(e, A.Name):
        # `use foo::bar` brings foo__bar into scope as bar; only rewrite bare
        # callee positions, not arbitrary names — but to be safe, handle it
        # only at callee positions in _rewrite_callee. Bare-name expressions
        # (e.g., a global constant reference) are not rewritten here because
        # we cannot distinguish constants from un-aliased names safely.
        return e
    # Cycle 57 C57-2 catchall — leaf expression types pass through
    # explicitly; anything else is an unmapped Expr subclass.
    if isinstance(e, (A.IntLit, A.FloatLit, A.StrLit, A.CharLit,
                       A.BoolLit, A.Path, A.Continue)):
        return e
    raise NotImplementedError(
        f"flatten_modules._rewrite_expr: unhandled expression kind "
        f"{type(e).__name__} at {getattr(e, 'span', '?')!r}. "
        f"Mirror cycle-57 C57-2 discipline: add an explicit arm."
    )


def _rewrite_callee(c: A.Expr, aliases: dict[str, str]) -> A.Expr:
    """Rewrite a Call.callee from Path/Name(generics) to a flattened Name.

    Patterns:
        Path([foo, bar])             -> Name("foo__bar")
        Path([foo, inner, baz])      -> Name("foo__inner__baz")
        Name("foo::bar") (turbofish) -> Name("foo__bar", generics=...)
        Name("bar") in aliases       -> Name(aliases["bar"])
    Generic args (turbofish) are preserved on the Name so monomorphize still sees them.
    """
    if isinstance(c, A.Path):
        if len(c.segments) >= 2:
            return A.Name(span=c.span, name="__".join(c.segments), generics=[])
        return c
    if isinstance(c, A.Name):
        # The parser collapses turbofish `foo::bar::<T>` into a single Name
        # with the literal `::` inside its name string. Re-flatten that here.
        if "::" in c.name:
            flat = c.name.replace("::", "__")
            return A.Name(span=c.span, name=flat, generics=c.generics)
        if c.name in aliases:
            return A.Name(span=c.span, name=aliases[c.name], generics=c.generics)
        return c
    return _rewrite_expr(c, aliases)


def _rewrite_stmt(s: A.Stmt, aliases: dict[str, str]) -> A.Stmt:
    if isinstance(s, A.Let):
        return A.Let(span=s.span, name=s.name, is_mut=s.is_mut, ty=s.ty,
                     value=_rewrite_expr(s.value, aliases) if s.value is not None else None)
    if isinstance(s, A.ExprStmt):
        return A.ExprStmt(span=s.span, expr=_rewrite_expr(s.expr, aliases))
    if isinstance(s, A.ConstStmt):
        return A.ConstStmt(span=s.span, name=s.name, ty=s.ty,
                           value=_rewrite_expr(s.value, aliases))
    return s
