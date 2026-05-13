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
    use_aliases_to_validate: list[tuple[str, str]] = []
    module_prefixes: set[str] = set()
    for item in prog.items:
        if isinstance(item, A.ModBlock):
            n = _flatten_one(
                item, prefix="", new_items=new_items,
                use_aliases_to_validate=use_aliases_to_validate,
                module_prefixes=module_prefixes,
            )
            flattened += n
        elif isinstance(item, A.UseDecl):
            # `use foo::bar` registers `bar` -> `foo__bar` alias.
            if len(item.path) >= 2:
                last = item.path[-1]
                full = "__".join(item.path)
                aliases[last] = full
                use_aliases_to_validate.append((last, full))
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
    for alias_src, alias_tgt in use_aliases_to_validate:
        if alias_tgt not in seen_names and alias_tgt not in module_prefixes:
            raise FlattenError(
                f"use-decl '{alias_src}' resolves to unknown name '{alias_tgt}' "
                f"(path '{alias_tgt.replace('__', '::')}'; trap 79001)")
    path_aliases: dict[str, str] = {}
    for nm in seen_names:
        if "__" not in nm:
            continue
        parts = nm.split("__")
        if any("__".join(parts[:i]) in module_prefixes
               for i in range(1, len(parts))):
            path_aliases[nm.replace("__", "::")] = nm
    rewrite_aliases = {**aliases, **path_aliases}
    prog.items = new_items
    if flattened or aliases:
        _rewrite_calls(prog, rewrite_aliases)
    return flattened


def _flatten_one(
    mb: A.ModBlock, prefix: str, new_items: list[A.Item],
    inherited_aliases: Optional[dict[str, str]] = None,
    use_aliases_to_validate: Optional[list[tuple[str, str]]] = None,
    module_prefixes: Optional[set[str]] = None,
) -> int:
    inherited_aliases = inherited_aliases or {}
    if use_aliases_to_validate is None:
        use_aliases_to_validate = []
    if module_prefixes is None:
        module_prefixes = set()
    base = (prefix + "__" if prefix else "") + mb.name
    module_prefixes.add(base)
    # Stage 28.9 cycle 66 silent-failure CN-1 fix (HIGH conf 95):
    # build intra-mod sibling aliases so that unqualified call sites
    # WITHIN this mod's body get rewritten to the mangled top-level
    # name. Without this, `mod m { fn foo() {} fn bar() { foo() } }`
    # post-flatten produces top-level `m__foo` and `m__bar` but the
    # `foo()` call inside m__bar's body still says `Name("foo")` —
    # name-based passes (totality self-call detection, deprecated
    # call-site walker, etc.) fail to match and silently miss the
    # call.
    # Stage 28.9 cycle 68 CN-1 fix (HIGH conf 92): only register
    # aliases for item kinds that are actually mangled+lifted with
    # their original name slot. Pre-fix iterating via `getattr(sub,
    # "name", None)` also pulled in ModBlock names — but a nested
    # `mod inner { ... }` is NOT itself lifted as `outer__inner`;
    # only its CHILDREN are (`outer__inner__sub`). The bogus alias
    # rewrote sibling-fn bodies that referenced `inner` (e.g. via
    # `let m = inner;`) to a non-existent `outer__inner`. AgentDecl
    # falls into the else branch (line 155) un-mangled, so it must
    # also be excluded. ImplBlock has no `name` field (it has
    # `target`), so it's naturally excluded by the type filter.
    sibling_aliases: dict[str, str] = {}
    local_use_aliases: dict[str, str] = {}
    for sub in mb.items:
        if isinstance(sub, (A.FnDecl, A.StructDecl, A.EnumDecl,
                             A.ConstDecl, A.TypeAlias)):
            sibling_aliases[sub.name] = base + "__" + sub.name
        elif isinstance(sub, A.UseDecl) and len(sub.path) >= 2:
            alias_tgt = "__".join(sub.path)
            local_use_aliases[sub.path[-1]] = alias_tgt
            use_aliases_to_validate.append((sub.path[-1], alias_tgt))
    inherited_for_children = {**inherited_aliases, **local_use_aliases}
    intra_mod_aliases = {**inherited_for_children, **sibling_aliases}
    # Stage 28.9 cycle 68 silent-failure CN-1 fix (HIGH conf 92):
    # track the indices of items lifted DIRECTLY by this _flatten_one
    # call (NOT via recursion into a nested ModBlock). Pre-fix used
    # a single `direct_lifts_start = len(new_items)` index and a
    # `range(direct_lifts_start, len(new_items))` walk after the loop,
    # which included items appended by recursive nested-mod calls.
    # That applied OUTER's intra_mod_aliases to INNER-scope bodies —
    # an unqualified `Name(sibling)` callee inside the inner mod
    # silently rebound to `outer__sibling` if the outer mod had a
    # fn `sibling`. Cross-mod name capture.
    # Fix: per-direct-lift index list. Nested-mod items are added by
    # the recursive call's own walk (with its own aliases) so we
    # don't touch them here.
    local_lift_indices: list[int] = []
    n = 0
    for sub in mb.items:
        if isinstance(sub, A.ModBlock):
            # Nested ModBlock: recurse with its own intra-mod scope.
            # We deliberately DO NOT record this in local_lift_indices
            # since the recursive call applies its own aliases to its
            # own lifted items.
            n += _flatten_one(
                sub, prefix=base, new_items=new_items,
                inherited_aliases=inherited_for_children,
                use_aliases_to_validate=use_aliases_to_validate,
                module_prefixes=module_prefixes,
            )
            continue
        # All other branches: lift to top level + record the slot
        # index so the post-loop walk only touches THIS call's
        # direct lifts.
        local_lift_indices.append(len(new_items))
        if isinstance(sub, A.FnDecl):
            new_name = base + "__" + sub.name
            type_aliases = _aliases_without_generics(
                intra_mod_aliases, sub.generics)
            # Stage 28.9 cycle 61 type-design O60-F (HIGH conf 85,
            # surfaced in cycle-60 follow-up audit): the lifted FnDecl
            # constructor pre-fix dropped `is_extern` and `extern_abi`.
            # `mod m { extern "C" fn foo() -> i32; }` was silently
            # demoted to a regular fn-decl with an empty placeholder
            # body, which then became the actual lowering target — the
            # GOT/PLT relocation that should have been emitted was
            # skipped and a call to the lifted `m__foo` produced a
            # zero-byte stub. Preserve the extern shape end-to-end.
            new_items.append(A.FnDecl(
                span=sub.span, name=new_name, generics=sub.generics,
                params=_rewrite_params(sub.params, type_aliases),
                return_ty=_rewrite_type_opt(sub.return_ty, type_aliases),
                where_clauses=_rewrite_where_clauses(
                    sub.where_clauses, type_aliases),
                body=sub.body,
                attrs=sub.attrs, is_pub=sub.is_pub,
                is_extern=sub.is_extern, extern_abi=sub.extern_abi))
            n += 1
        elif isinstance(sub, A.StructDecl):
            new_name = base + "__" + sub.name
            type_aliases = _aliases_without_generics(
                intra_mod_aliases, sub.generics)
            new_items.append(A.StructDecl(
                span=sub.span, name=new_name, generics=sub.generics,
                fields=_rewrite_params(sub.fields, type_aliases),
                is_pub=sub.is_pub))
            n += 1
        elif isinstance(sub, A.EnumDecl):
            new_name = base + "__" + sub.name
            type_aliases = _aliases_without_generics(
                intra_mod_aliases, sub.generics)
            new_items.append(A.EnumDecl(
                span=sub.span, name=new_name, generics=sub.generics,
                variants=_rewrite_enum_variants(
                    sub.variants, type_aliases),
                is_pub=sub.is_pub))
            n += 1
        elif isinstance(sub, A.ConstDecl):
            new_name = base + "__" + sub.name
            new_items.append(A.ConstDecl(
                span=sub.span, name=new_name,
                ty=_rewrite_type(sub.ty, intra_mod_aliases),
                value=_rewrite_const_expr(sub.value, intra_mod_aliases),
                is_pub=sub.is_pub))
            n += 1
        elif isinstance(sub, A.TypeAlias):
            new_name = base + "__" + sub.name
            type_aliases = _aliases_without_generics(
                intra_mod_aliases, sub.generics)
            new_items.append(A.TypeAlias(
                span=sub.span, name=new_name, generics=sub.generics,
                target=_rewrite_type(sub.target, type_aliases),
                is_pub=sub.is_pub,
                where_clauses=_rewrite_where_clauses(
                    sub.where_clauses, type_aliases)))
            n += 1
        elif isinstance(sub, A.UseDecl):
            new_items.append(sub)
        elif isinstance(sub, A.ImplBlock):
            # Stage 28.9 cycle 68 CN-2 fix (HIGH conf 85): ImplBlocks
            # nested inside a ModBlock get lifted to top level with
            # `target` field mangled to match the sibling StructDecl's
            # post-flatten name. Pre-fix the ImplBlock fell into the
            # generic `else: append(sub)` branch and was lifted
            # verbatim — its `target` stayed as the unqualified struct
            # name (e.g. "Foo") while the sibling StructDecl was lifted
            # to "m__Foo". flatten_impls then lifted methods as
            # `Foo__method` (a name that doesn't reference the real
            # struct), and method-body sibling-fn calls were never
            # rewritten because this loop only walks FnDecl/ConstDecl.
            # Now: if the target matches a sibling StructDecl (i.e. is
            # in intra_mod_aliases), rewrite it; and walk every method
            # body with the alias dict so intra-mod sibling calls in
            # methods are rewritten consistently.
            new_target = intra_mod_aliases.get(sub.target, sub.target)
            new_methods: list[A.FnDecl] = []
            for m in sub.methods:
                type_aliases = _aliases_without_generics(
                    intra_mod_aliases, m.generics)
                new_methods.append(A.FnDecl(
                    span=m.span, name=m.name, generics=m.generics,
                    params=_rewrite_params(m.params, type_aliases),
                    return_ty=_rewrite_type_opt(
                        m.return_ty, type_aliases),
                    where_clauses=_rewrite_where_clauses(
                        m.where_clauses, type_aliases),
                    body=_rewrite_expr(
                        m.body, type_aliases,
                        rewrite_bare_names=True,
                        bound=_fn_value_bindings(m),
                    ),
                    attrs=m.attrs, is_pub=m.is_pub,
                    is_extern=m.is_extern, extern_abi=m.extern_abi,
                ))
            new_items.append(A.ImplBlock(
                span=sub.span, target=new_target,
                methods=new_methods, trait_name=sub.trait_name,
                is_pub=sub.is_pub,
            ))
        else:
            new_items.append(sub)
    # Stage 28.9 cycle 66 silent-failure CN-1 fix: rewrite intra-mod
    # call sites in the items lifted by THIS call. Sibling FnDecls
    # call each other by their original unqualified name (e.g.
    # `foo()` inside `m::bar`'s body); after lifting both to
    # `m__foo` and `m__bar`, the call site must also rewrite to
    # `m__foo()` so name-based downstream passes (totality self-call
    # detection, deprecated call-site walker, etc.) can match.
    # _rewrite_expr already handles `Name(N) -> Name(aliases[N])` via
    # _rewrite_callee, so we reuse it with `intra_mod_aliases` here.
    if intra_mod_aliases:
        for i in local_lift_indices:
            it = new_items[i]
            if isinstance(it, A.FnDecl) and not it.is_extern:
                it.body = _rewrite_expr(
                    it.body,
                    _aliases_without_generics(intra_mod_aliases, it.generics),
                    rewrite_bare_names=True,
                    bound=_fn_value_bindings(it),
                )
            elif isinstance(it, A.ConstDecl):
                it.value = _rewrite_const_expr(it.value, intra_mod_aliases)
    return n


def _rewrite_calls(prog: A.Program, aliases: dict[str, str]) -> None:
    """Walk every FnDecl body, rewriting:
       - `foo::bar(args)` Path-call -> `foo__bar(args)` Name-call
       - `bar(args)` where `bar` is in aliases -> `<alias>(args)` Name-call
    """
    prog.items = [_rewrite_item_decls(item, aliases) for item in prog.items]


def _rewrite_item_decls(item: A.Item, aliases: dict[str, str]) -> A.Item:
    if isinstance(item, A.FnDecl):
        type_aliases = _aliases_without_generics(aliases, item.generics)
        return A.FnDecl(
            span=item.span, name=item.name, generics=item.generics,
            params=_rewrite_params(item.params, type_aliases),
            return_ty=_rewrite_type_opt(item.return_ty, type_aliases),
            where_clauses=_rewrite_where_clauses(
                item.where_clauses, type_aliases),
            body=item.body if item.is_extern else _rewrite_expr(
                item.body, type_aliases,
                rewrite_bare_names=True,
                bound=_fn_value_bindings(item),
            ),
            attrs=item.attrs, is_pub=item.is_pub,
            is_extern=item.is_extern, extern_abi=item.extern_abi,
        )
    if isinstance(item, A.StructDecl):
        type_aliases = _aliases_without_generics(aliases, item.generics)
        return A.StructDecl(
            span=item.span, name=item.name, generics=item.generics,
            fields=_rewrite_params(item.fields, type_aliases),
            is_pub=item.is_pub,
        )
    if isinstance(item, A.EnumDecl):
        type_aliases = _aliases_without_generics(aliases, item.generics)
        return A.EnumDecl(
            span=item.span, name=item.name, generics=item.generics,
            variants=_rewrite_enum_variants(item.variants, type_aliases),
            is_pub=item.is_pub,
        )
    if isinstance(item, A.ConstDecl):
        return A.ConstDecl(
            span=item.span, name=item.name,
            ty=_rewrite_type(item.ty, aliases),
            value=_rewrite_const_expr(item.value, aliases),
            is_pub=item.is_pub,
        )
    if isinstance(item, A.TypeAlias):
        type_aliases = _aliases_without_generics(aliases, item.generics)
        return A.TypeAlias(
            span=item.span, name=item.name, generics=item.generics,
            target=_rewrite_type(item.target, type_aliases),
            is_pub=item.is_pub,
            where_clauses=_rewrite_where_clauses(
                item.where_clauses, type_aliases),
        )
    if isinstance(item, A.ImplBlock):
        return A.ImplBlock(
            span=item.span, target=aliases.get(item.target, item.target),
            methods=[
                _rewrite_item_decls(m, aliases)
                for m in item.methods
            ],
            trait_name=item.trait_name,
            is_pub=item.is_pub,
        )
    return item


def _aliases_without_generics(
    aliases: dict[str, str], generics: list[A.GenericParam],
) -> dict[str, str]:
    shadowed = {g.name for g in generics}
    if not shadowed:
        return aliases
    return {k: v for k, v in aliases.items() if k not in shadowed}


def _fn_value_bindings(fn: A.FnDecl) -> set[str]:
    return {p.name for p in fn.params}


def _pattern_bindings(pat: A.Pattern) -> set[str]:
    if isinstance(pat, A.PatBind):
        return {pat.name}
    if isinstance(pat, A.PatTuple):
        names: set[str] = set()
        for elem in pat.elems:
            names.update(_pattern_bindings(elem))
        return names
    if isinstance(pat, A.PatOr):
        names: set[str] = set()
        for alt in pat.alts:
            names.update(_pattern_bindings(alt))
        return names
    if isinstance(pat, A.PatVariant):
        names: set[str] = set()
        for sub in pat.sub_patterns:
            names.update(_pattern_bindings(sub))
        return names
    return set()


def _rewrite_params(
    params: list[A.FnParam], aliases: dict[str, str],
) -> list[A.FnParam]:
    return [
        A.FnParam(
            span=p.span, name=p.name, ty=_rewrite_type(p.ty, aliases),
            is_mut=p.is_mut,
        )
        for p in params
    ]


def _rewrite_enum_variants(
    variants: list[A.EnumVariant], aliases: dict[str, str],
) -> list[A.EnumVariant]:
    return [
        A.EnumVariant(
            span=v.span, name=v.name,
            payload_tys=[_rewrite_type(t, aliases) for t in v.payload_tys],
        )
        for v in variants
    ]


def _rewrite_where_clauses(
    clauses: list[A.WhereClause], aliases: dict[str, str],
) -> list[A.WhereClause]:
    return [
        A.WhereClause(
            span=w.span,
            constraint=_rewrite_type_expr(
                w.constraint, aliases, bound={"self"}),
        )
        for w in clauses
    ]


def _rewrite_type_opt(
    ty: Optional[A.TyNode], aliases: dict[str, str],
    bound: Optional[set[str]] = None,
) -> Optional[A.TyNode]:
    return _rewrite_type(ty, aliases, bound=bound) if ty is not None else None


def _rewrite_type(
    ty: A.TyNode, aliases: dict[str, str],
    bound: Optional[set[str]] = None,
) -> A.TyNode:
    bound_names = set(bound or ())
    if isinstance(ty, A.TyName):
        # Type names live in the type namespace, so value bindings do not
        # shadow them. Only type-expression children such as array sizes use
        # bound_names.
        return A.TyName(span=ty.span, name=aliases.get(ty.name, ty.name))
    if isinstance(ty, A.TyTuple):
        return A.TyTuple(
            span=ty.span,
            elems=[_rewrite_type(e, aliases, bound=bound_names)
                   for e in ty.elems],
        )
    if isinstance(ty, A.TyArray):
        return A.TyArray(
            span=ty.span,
            elem=_rewrite_type(ty.elem, aliases, bound=bound_names),
            size=_rewrite_type_expr(ty.size, aliases, bound=bound_names),
        )
    if isinstance(ty, A.TyRef):
        return A.TyRef(
            span=ty.span,
            inner=_rewrite_type(ty.inner, aliases, bound=bound_names),
            is_mut=ty.is_mut,
        )
    if isinstance(ty, A.TyPtr):
        return A.TyPtr(
            span=ty.span,
            inner=_rewrite_type(ty.inner, aliases, bound=bound_names),
            is_mut=ty.is_mut,
        )
    if isinstance(ty, A.TyFn):
        return A.TyFn(
            span=ty.span,
            params=[_rewrite_type(p, aliases, bound=bound_names)
                    for p in ty.params],
            ret=_rewrite_type(ty.ret, aliases, bound=bound_names),
        )
    if isinstance(ty, A.TyTensor):
        return A.TyTensor(
            span=ty.span,
            dtype=_rewrite_type(ty.dtype, aliases, bound=bound_names),
            shape=[_rewrite_type_expr(s, aliases, bound=bound_names)
                   for s in ty.shape],
            device=_rewrite_type_expr(ty.device, aliases, bound=bound_names)
            if ty.device is not None else None,
            layout=_rewrite_type_expr(ty.layout, aliases, bound=bound_names)
            if ty.layout is not None else None,
        )
    if isinstance(ty, A.TyTile):
        return A.TyTile(
            span=ty.span,
            dtype=_rewrite_type(ty.dtype, aliases, bound=bound_names),
            shape=[_rewrite_type_expr(s, aliases, bound=bound_names)
                   for s in ty.shape],
            memspace=_rewrite_type_expr(
                ty.memspace, aliases, bound=bound_names),
        )
    if isinstance(ty, A.TyGeneric):
        return A.TyGeneric(
            span=ty.span,
            base=aliases.get(ty.base, ty.base),
            args=[_rewrite_type(a, aliases, bound=bound_names)
                  for a in ty.args],
        )
    raise NotImplementedError(
        f"flatten_modules._rewrite_type: unhandled type kind "
        f"{type(ty).__name__} at {getattr(ty, 'span', '?')!r}"
    )


def _rewrite_type_expr(
    e: A.Expr,
    aliases: dict[str, str],
    bound: Optional[set[str]] = None,
) -> A.Expr:
    """Rewrite expression children that live inside type syntax.

    Unlike value-position rewrites, bare names in type expressions can name
    compile-time constants, so module flattening must remap them too.
    """
    bound_names = set(bound or ())
    if isinstance(e, A.Name):
        return A.Name(
            span=e.span,
            name=e.name if e.name in bound_names else aliases.get(
                e.name, e.name),
            generics=[_rewrite_type(g, aliases, bound=bound_names)
                      for g in e.generics],
        )
    if isinstance(e, A.Binary):
        return A.Binary(
            span=e.span, op=e.op,
            left=_rewrite_type_expr(e.left, aliases, bound=bound_names),
            right=_rewrite_type_expr(e.right, aliases, bound=bound_names),
        )
    if isinstance(e, A.Unary):
        return A.Unary(
            span=e.span, op=e.op,
            operand=_rewrite_type_expr(
                e.operand, aliases, bound=bound_names),
        )
    if isinstance(e, A.Call):
        return A.Call(
            span=e.span,
            callee=_rewrite_callee(e.callee, aliases, bound=bound_names),
            args=[_rewrite_type_expr(a, aliases, bound=bound_names)
                  for a in e.args],
        )
    if isinstance(e, A.Path):
        return _rewrite_type_expr_path(e, aliases, bound=bound_names)
    if isinstance(e, (A.IntLit, A.FloatLit, A.StrLit, A.CharLit,
                      A.BoolLit)):
        return e
    return _rewrite_expr(
        e, aliases, rewrite_bare_names=True, bound=bound_names)


def _rewrite_type_expr_path(
    path: A.Path,
    aliases: dict[str, str],
    bound: Optional[set[str]] = None,
) -> A.Expr:
    """Rewrite a module path that appears in type syntax to a flat name."""
    bound_names = set(bound or ())
    segs = list(path.segments)
    if not segs:
        return path
    if segs[0] in bound_names:
        return path
    if segs[0] in aliases:
        segs = aliases[segs[0]].split("__") + segs[1:]
    if len(segs) >= 2:
        return A.Name(span=path.span, name="__".join(segs), generics=[])
    if segs[0] in aliases:
        return A.Name(span=path.span, name=aliases[segs[0]], generics=[])
    return path


def _rewrite_const_expr(e: A.Expr, aliases: dict[str, str]) -> A.Expr:
    """Rewrite expression children that live inside const initializers."""
    return _rewrite_type_expr(e, aliases)


def _rewrite_expr(
    e: A.Expr,
    aliases: dict[str, str],
    *,
    rewrite_bare_names: bool = False,
    bound: Optional[set[str]] = None,
) -> A.Expr:
    bound_names = set(bound or ())
    if isinstance(e, A.Call):
        new_callee = _rewrite_callee(e.callee, aliases, bound=bound_names)
        new_args = [
            _rewrite_expr(
                a, aliases,
                rewrite_bare_names=rewrite_bare_names,
                bound=bound_names,
            )
            for a in e.args
        ]
        return A.Call(span=e.span, callee=new_callee, args=new_args)
    if isinstance(e, A.Block):
        scoped = set(bound_names)
        new_stmts: list[A.Stmt] = []
        for stmt in e.stmts:
            new_stmts.append(_rewrite_stmt(
                stmt, aliases,
                rewrite_bare_names=rewrite_bare_names,
                bound=scoped,
            ))
            if isinstance(stmt, (A.Let, A.ConstStmt)):
                scoped.add(stmt.name)
        return A.Block(span=e.span,
                       stmts=new_stmts,
                       final_expr=_rewrite_expr(
                           e.final_expr, aliases,
                           rewrite_bare_names=rewrite_bare_names,
                           bound=scoped,
                       ) if e.final_expr is not None else None)
    if isinstance(e, A.If):
        else_ = e.else_
        if else_ is not None:
            else_ = _rewrite_expr(
                else_, aliases,
                rewrite_bare_names=rewrite_bare_names,
                bound=bound_names,
            )
        return A.If(span=e.span,
                    cond=_rewrite_expr(
                        e.cond, aliases,
                        rewrite_bare_names=rewrite_bare_names,
                        bound=bound_names,
                    ),
                    then=_rewrite_expr(
                        e.then, aliases,
                        rewrite_bare_names=rewrite_bare_names,
                        bound=bound_names,
                    ),
                    else_=else_)
    if isinstance(e, A.Match):
        arms: list[A.MatchArm] = []
        for arm in e.arms:
            new_pattern = _rewrite_pattern(arm.pattern, aliases)
            arm_bound = set(bound_names)
            arm_bound.update(_pattern_bindings(arm.pattern))
            arms.append(A.MatchArm(
                span=arm.span,
                pattern=new_pattern,
                guard=_rewrite_expr(
                    arm.guard, aliases,
                    rewrite_bare_names=rewrite_bare_names,
                    bound=arm_bound,
                ) if arm.guard else None,
                body=_rewrite_expr(
                    arm.body, aliases,
                    rewrite_bare_names=rewrite_bare_names,
                    bound=arm_bound,
                ),
            ))
        return A.Match(span=e.span,
                       scrutinee=_rewrite_expr(
                           e.scrutinee, aliases,
                           rewrite_bare_names=rewrite_bare_names,
                           bound=bound_names,
                       ),
                       arms=arms)
    if isinstance(e, A.For):
        body_bound = set(bound_names)
        body_bound.add(e.var_name)
        return A.For(span=e.span, var_name=e.var_name,
                     iter_expr=_rewrite_expr(
                         e.iter_expr, aliases,
                         rewrite_bare_names=rewrite_bare_names,
                         bound=bound_names,
                     ),
                     body=_rewrite_expr(
                         e.body, aliases,
                         rewrite_bare_names=rewrite_bare_names,
                         bound=body_bound,
                     ))
    if isinstance(e, A.While):
        return A.While(span=e.span,
                       cond=_rewrite_expr(
                           e.cond, aliases,
                           rewrite_bare_names=rewrite_bare_names,
                           bound=bound_names,
                       ),
                       body=_rewrite_expr(
                           e.body, aliases,
                           rewrite_bare_names=rewrite_bare_names,
                           bound=bound_names,
                       ))
    if isinstance(e, A.Loop):
        return A.Loop(span=e.span, body=_rewrite_expr(
            e.body, aliases,
            rewrite_bare_names=rewrite_bare_names,
            bound=bound_names,
        ))
    if isinstance(e, A.Binary):
        return A.Binary(span=e.span, op=e.op,
                        left=_rewrite_expr(
                            e.left, aliases,
                            rewrite_bare_names=rewrite_bare_names,
                            bound=bound_names,
                        ),
                        right=_rewrite_expr(
                            e.right, aliases,
                            rewrite_bare_names=rewrite_bare_names,
                            bound=bound_names,
                        ))
    if isinstance(e, A.Unary):
        return A.Unary(span=e.span, op=e.op, operand=_rewrite_expr(
            e.operand, aliases,
            rewrite_bare_names=rewrite_bare_names,
            bound=bound_names,
        ))
    if isinstance(e, A.Cast):
        return A.Cast(
            span=e.span,
            value=_rewrite_expr(
                e.value, aliases,
                rewrite_bare_names=rewrite_bare_names,
                bound=bound_names,
            ),
            target_ty=_rewrite_type(e.target_ty, aliases, bound=bound_names),
        )
    if isinstance(e, A.Index):
        return A.Index(
            span=e.span,
            callee=_rewrite_expr(
                e.callee, aliases,
                rewrite_bare_names=rewrite_bare_names,
                bound=bound_names,
            ),
            indices=[
                _rewrite_expr(
                    i, aliases,
                    rewrite_bare_names=rewrite_bare_names,
                    bound=bound_names,
                )
                for i in e.indices
            ],
        )
    if isinstance(e, A.Field):
        return A.Field(span=e.span, obj=_rewrite_expr(
            e.obj, aliases,
            rewrite_bare_names=rewrite_bare_names,
            bound=bound_names,
        ), name=e.name)
    if isinstance(e, A.TupleLit):
        return A.TupleLit(span=e.span, elems=[
            _rewrite_expr(
                x, aliases,
                rewrite_bare_names=rewrite_bare_names,
                bound=bound_names,
            )
            for x in e.elems
        ])
    if isinstance(e, A.ArrayLit):
        return A.ArrayLit(span=e.span, elems=[
            _rewrite_expr(
                x, aliases,
                rewrite_bare_names=rewrite_bare_names,
                bound=bound_names,
            )
            for x in e.elems
        ])
    if isinstance(e, A.StructLit):
        # Stage 28.9 cycle 71 code-review CN-3 fix (HIGH conf 88):
        # `StructLit.name` is a struct-type reference that mod-flattening
        # must remap. Inside `mod m { struct Foo; fn make() { Foo {x:1} } }`,
        # post-flatten the FnDecl becomes `m__make` and the StructDecl
        # becomes `m__Foo`, but the inner `StructLit(name="Foo")` was not
        # rewritten — typecheck then warns "unknown struct 'Foo'" and
        # the backend (which runs flatten before typecheck) compiles a
        # stale name. Apply the same alias mapping the call-site walker
        # uses: if `e.name` matches an alias, remap.
        new_name = aliases.get(e.name, e.name)
        return A.StructLit(span=e.span, name=new_name,
                           fields=[
                               (n, _rewrite_expr(
                                   v, aliases,
                                   rewrite_bare_names=rewrite_bare_names,
                                   bound=bound_names,
                               ))
                               for (n, v) in e.fields
                           ])
    if isinstance(e, A.Assign):
        return A.Assign(
            span=e.span,
            target=_rewrite_expr(
                e.target, aliases,
                rewrite_bare_names=rewrite_bare_names,
                bound=bound_names,
            ),
            op=e.op,
            value=_rewrite_expr(
                e.value, aliases,
                rewrite_bare_names=rewrite_bare_names,
                bound=bound_names,
            ))
    if isinstance(e, A.Return):
        return A.Return(span=e.span,
                        value=_rewrite_expr(
                            e.value, aliases,
                            rewrite_bare_names=rewrite_bare_names,
                            bound=bound_names,
                        ) if e.value is not None else None)
    if isinstance(e, A.Break):
        return A.Break(span=e.span,
                       value=_rewrite_expr(
                           e.value, aliases,
                           rewrite_bare_names=rewrite_bare_names,
                           bound=bound_names,
                       ) if e.value is not None else None)
    if isinstance(e, A.Range):
        return A.Range(
            span=e.span,
            start=_rewrite_expr(
                e.start, aliases,
                rewrite_bare_names=rewrite_bare_names,
                bound=bound_names,
            ) if e.start is not None else None,
            end=_rewrite_expr(
                e.end, aliases,
                rewrite_bare_names=rewrite_bare_names,
                bound=bound_names,
            ) if e.end is not None else None)
    if isinstance(e, A.Quote):
        return A.Quote(span=e.span, inner=_rewrite_expr(
            e.inner, aliases,
            rewrite_bare_names=rewrite_bare_names,
            bound=bound_names,
        ))
    if isinstance(e, A.Splice):
        return A.Splice(span=e.span, inner=_rewrite_expr(
            e.inner, aliases,
            rewrite_bare_names=rewrite_bare_names,
            bound=bound_names,
        ))
    if isinstance(e, A.Modify):
        return A.Modify(
            span=e.span,
            target=_rewrite_expr(
                e.target, aliases,
                rewrite_bare_names=rewrite_bare_names,
                bound=bound_names,
            ),
            transformation=_rewrite_expr(
                e.transformation, aliases,
                rewrite_bare_names=rewrite_bare_names,
                bound=bound_names,
            ),
            verifier=_rewrite_expr(
                e.verifier, aliases,
                rewrite_bare_names=rewrite_bare_names,
                bound=bound_names,
            ))
    # Stage 28.9 cycle 57 C57-2 (HIGH, conf 86): pre-fix this walker
    # was missing UnsafeBlock + TileLit arms. `use foo::bar; fn main()
    # { unsafe { bar() } }` left the `bar` callee un-aliased; the
    # subsequent name-resolution pass failed with a misleading
    # "unknown function 'bar'" instead of routing to foo__bar. Same
    # defect class as match_lower C22-C and flatten_impls C57-1.
    if isinstance(e, A.UnsafeBlock):
        new_body = _rewrite_expr(
            e.body, aliases,
            rewrite_bare_names=rewrite_bare_names,
            bound=bound_names,
        )
        assert isinstance(new_body, A.Block), \
            "flatten_modules: UnsafeBlock.body rewrite must return Block"
        return A.UnsafeBlock(span=e.span, body=new_body)
    if isinstance(e, A.TileLit):
        return A.TileLit(
            span=e.span, dtype=_rewrite_type(e.dtype, aliases),
            shape=[
                _rewrite_expr(
                    s, aliases,
                    rewrite_bare_names=rewrite_bare_names,
                    bound=bound_names,
                )
                for s in e.shape
            ],
            memspace=_rewrite_expr(
                e.memspace, aliases,
                rewrite_bare_names=rewrite_bare_names,
                bound=bound_names,
            ),
            init=e.init,
        )
    if isinstance(e, A.Name):
        # `use foo::bar` brings foo__bar into scope as bar; only rewrite bare
        # callee positions, not arbitrary names — but to be safe, handle it
        # only at callee positions in _rewrite_callee. Bare-name expressions
        # (e.g., a global constant reference) are not rewritten here because
        # we cannot distinguish constants from un-aliased names safely.
        if rewrite_bare_names and e.name in aliases and e.name not in bound_names:
            return A.Name(
                span=e.span,
                name=aliases[e.name],
                generics=[_rewrite_type(g, aliases) for g in e.generics],
            )
        return e
    if isinstance(e, A.Path):
        return _rewrite_value_path(e, aliases)
    # Cycle 57 C57-2 catchall — leaf expression types pass through
    # explicitly; anything else is an unmapped Expr subclass.
    if isinstance(e, (A.IntLit, A.FloatLit, A.StrLit, A.CharLit,
                       A.BoolLit, A.Continue)):
        return e
    raise NotImplementedError(
        f"flatten_modules._rewrite_expr: unhandled expression kind "
        f"{type(e).__name__} at {getattr(e, 'span', '?')!r}. "
        f"Mirror cycle-57 C57-2 discipline: add an explicit arm."
    )


def _rewrite_pattern_path(path: A.Path, aliases: dict[str, str]) -> A.Path:
    segs = list(path.segments)
    if segs and segs[0] in aliases:
        segs = aliases[segs[0]].split("__") + segs[1:]
    if len(segs) == 2 and segs[0] in aliases:
        return A.Path(span=path.span, segments=[aliases[segs[0]], segs[1]])
    if len(segs) > 2:
        return A.Path(
            span=path.span,
            segments=["__".join(segs[:-1]), segs[-1]],
        )
    return path


def _rewrite_value_path(path: A.Path, aliases: dict[str, str]) -> A.Expr:
    segs = list(path.segments)
    full = "::".join(segs)
    if full in aliases:
        return A.Name(span=path.span, name=aliases[full], generics=[])
    if segs and segs[0] in aliases:
        expanded = aliases[segs[0]].split("__") + segs[1:]
        expanded_full = "::".join(expanded)
        if expanded_full in aliases:
            return A.Name(
                span=path.span, name=aliases[expanded_full], generics=[])
    return _rewrite_pattern_path(path, aliases)


def _rewrite_pattern(pat: A.Pattern, aliases: dict[str, str]) -> A.Pattern:
    if isinstance(pat, A.PatLit):
        value = pat.value
        if isinstance(value, A.Path):
            value = _rewrite_pattern_path(value, aliases)
        else:
            value = _rewrite_expr(value, aliases)
        return A.PatLit(span=pat.span, value=value)
    if isinstance(pat, A.PatVariant):
        return A.PatVariant(
            span=pat.span,
            path=_rewrite_pattern_path(pat.path, aliases),
            sub_patterns=[_rewrite_pattern(p, aliases)
                          for p in pat.sub_patterns],
        )
    if isinstance(pat, A.PatTuple):
        return A.PatTuple(
            span=pat.span,
            elems=[_rewrite_pattern(p, aliases) for p in pat.elems],
        )
    if isinstance(pat, A.PatOr):
        return A.PatOr(
            span=pat.span,
            alts=[_rewrite_pattern(p, aliases) for p in pat.alts],
        )
    if isinstance(pat, A.PatRange):
        return A.PatRange(
            span=pat.span,
            lo=_rewrite_expr(pat.lo, aliases),
            hi=_rewrite_expr(pat.hi, aliases),
            inclusive=pat.inclusive,
        )
    if isinstance(pat, (A.PatBind, A.PatWildcard)):
        return pat
    raise NotImplementedError(
        f"flatten_modules._rewrite_pattern: unhandled pattern kind "
        f"{type(pat).__name__} at {getattr(pat, 'span', '?')!r}."
    )


def _rewrite_callee(
    c: A.Expr, aliases: dict[str, str],
    *, bound: Optional[set[str]] = None,
) -> A.Expr:
    """Rewrite a Call.callee from Path/Name(generics) to a flattened Name.

    Patterns:
        Path([foo, bar])             -> Name("foo__bar")
        Path([foo, inner, baz])      -> Name("foo__inner__baz")
        Name("foo::bar") (turbofish) -> Name("foo__bar", generics=...)
        Name("bar") in aliases       -> Name(aliases["bar"])
    Generic args (turbofish) are preserved on the Name so monomorphize still sees them.
    """
    bound_names = set(bound or ())
    if isinstance(c, A.Path):
        if len(c.segments) >= 2:
            segs = list(c.segments)
            if segs[0] in aliases:
                segs = aliases[segs[0]].split("__") + segs[1:]
            return A.Name(span=c.span, name="__".join(segs), generics=[])
        return c
    if isinstance(c, A.Name):
        generics = [_rewrite_type(g, aliases, bound=bound_names)
                    for g in c.generics]
        # The parser collapses turbofish `foo::bar::<T>` into a single Name
        # with the literal `::` inside its name string. Re-flatten that here.
        if "::" in c.name:
            segs = c.name.split("::")
            if segs[0] in aliases:
                segs = aliases[segs[0]].split("__") + segs[1:]
            flat = "__".join(segs)
            return A.Name(span=c.span, name=flat, generics=generics)
        if c.name in aliases and c.name not in bound_names:
            return A.Name(span=c.span, name=aliases[c.name], generics=generics)
        if generics != c.generics:
            return A.Name(span=c.span, name=c.name, generics=generics)
        return c
    return _rewrite_expr(c, aliases, bound=bound_names)


def _rewrite_stmt(
    s: A.Stmt,
    aliases: dict[str, str],
    *,
    rewrite_bare_names: bool = False,
    bound: Optional[set[str]] = None,
) -> A.Stmt:
    bound_names = set(bound or ())
    if isinstance(s, A.Let):
        return A.Let(
            span=s.span, name=s.name, is_mut=s.is_mut,
            ty=_rewrite_type_opt(s.ty, aliases, bound=bound_names),
            value=_rewrite_expr(
                s.value, aliases,
                rewrite_bare_names=rewrite_bare_names,
                bound=bound_names,
            ) if s.value is not None else None)
    if isinstance(s, A.ExprStmt):
        return A.ExprStmt(span=s.span, expr=_rewrite_expr(
            s.expr, aliases,
            rewrite_bare_names=rewrite_bare_names,
            bound=bound_names,
        ))
    if isinstance(s, A.ConstStmt):
        return A.ConstStmt(span=s.span, name=s.name,
                           ty=_rewrite_type(s.ty, aliases,
                                            bound=bound_names),
                           value=_rewrite_expr(
                               s.value, aliases,
                               rewrite_bare_names=rewrite_bare_names,
                               bound=bound_names,
                           ))
    return s
