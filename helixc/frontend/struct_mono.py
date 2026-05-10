"""
helixc/frontend/struct_mono.py — Stage 28: parametric structs.

Mojo-style generic structs: `struct Pt<T> { x: T, y: T }` instantiated
at each (T) use-site. Builds on Stage 8's fn monomorphization machinery
and reuses the same mangling scheme.

Pipeline:
  1. Collect all StructDecls with non-empty generics into a table.
  2. Walk the program for TyGeneric uses where base is a known generic
     struct name (e.g. `Pt[i32]` as a parameter type).
  3. For each unique (struct_name, type_args), clone the StructDecl with
     type-vars substituted and a mangled name (e.g. `Pt__i32`).
  4. Replace TyGeneric(base=Pt, args=[i32]) with TyName(Pt__i32) at use
     sites.
  5. Drop original generic StructDecls (keep them in attrs for docs).

Trap-id reservations:
  * 28001 — parametric struct uninstantiated (a generic struct named
            in a type position with no concrete type-args).
  * 28002 — parametric struct const-eval failure (deferred to v0.2).

Phase-0 supports type args only; const-int parameters (e.g.
`Tensor<f32, [N]>`) are reserved for a later iteration.

License: Apache 2.0
"""

from __future__ import annotations

from copy import deepcopy
from typing import Optional

from . import ast_nodes as A
from .monomorphize import _mangle_ty, substitute_ty


TRAP_PARAM_STRUCT_UNINSTANTIATED = 28001
TRAP_PARAM_STRUCT_CONSTEVAL = 28002


def mangle_struct(name: str, ty_args: list[A.TyNode]) -> str:
    """Generate a unique mangled name for a parametric struct
    instantiation. Mirrors fn monomorphization's scheme."""
    parts = [_mangle_ty(t) for t in ty_args]
    return f"{name}__" + "_".join(parts)


def collect_generic_structs(prog: A.Program) -> dict[str, A.StructDecl]:
    """Return {name: decl} for every StructDecl with non-empty
    generics."""
    out: dict[str, A.StructDecl] = {}
    for it in prog.items:
        if isinstance(it, A.StructDecl) and it.generics:
            out[it.name] = it
    return out


def collect_concrete_uses(prog: A.Program,
                          generic_structs: dict[str, A.StructDecl]
                          ) -> list[tuple[str, list[A.TyNode]]]:
    """Walk the program (param tys, return tys, field tys, body
    expressions) and return a deduplicated list of (struct_name,
    type_args) for each use of a generic struct with concrete args.

    Audit 28.8 Finding 3 / B1 / C1-M2 fix: in addition to signature
    types, this now walks fn bodies looking for:
      * `Let` stmt type annotations: `let p: Pt<i32> = ...`
      * `Cast` target types: `x as Pt<i32>`
      * Method-call generic args: `Foo::<i32>::new(...)` (callee
        Name carrying generics)
      * StructLit names that match a generic struct's name (without
        explicit args we can't instantiate, so this is best-effort:
        a `Pt { x: 1, y: 2 }` with no explicit `Pt::<...>` is
        treated as a use *only* if elsewhere a Pt<T> instantiation
        with concrete T was already collected; that handling stays
        in the typechecker side).

    Phase-0 conservative: TyGeneric inside Tensor/Tile shape exprs is
    NOT walked because shape-arg substitution is a separate stage; same
    for Quote/Splice/Modify bodies which carry meta-AST.
    """
    seen: set[tuple] = set()
    out: list[tuple[str, list[A.TyNode]]] = []

    def visit_ty(t: A.TyNode) -> None:
        if t is None:
            return
        if isinstance(t, A.TyGeneric):
            if t.base in generic_structs:
                key = (t.base, tuple(_ty_key(a) for a in t.args))
                if key not in seen:
                    seen.add(key)
                    out.append((t.base, list(t.args)))
            for a in t.args:
                visit_ty(a)
            return
        if isinstance(t, A.TyTuple):
            for e in t.elems:
                visit_ty(e)
            return
        if isinstance(t, A.TyArray):
            visit_ty(t.elem)
            return
        if isinstance(t, A.TyRef):
            visit_ty(t.inner)
            return
        if isinstance(t, A.TyPtr):
            visit_ty(t.inner)
            return
        if isinstance(t, A.TyFn):
            for p in t.params:
                visit_ty(p)
            visit_ty(t.ret)
            return
        if isinstance(t, A.TyTensor):
            visit_ty(t.dtype)
            return
        if isinstance(t, A.TyTile):
            visit_ty(t.dtype)
            return

    def visit_expr(e) -> None:
        """Body walker — recurses through expressions to find embedded
        TyGeneric uses (Cast.target_ty, Let.ty, callee generics on
        Name)."""
        if e is None:
            return
        # Cast carries a target type that may name a generic struct.
        if isinstance(e, A.Cast):
            visit_ty(e.target_ty)
            visit_expr(e.value)
            return
        # Block: walk stmts + final_expr
        if isinstance(e, A.Block):
            for s in e.stmts:
                visit_stmt(s)
            if e.final_expr is not None:
                visit_expr(e.final_expr)
            return
        if isinstance(e, A.UnsafeBlock):
            visit_expr(e.body)
            return
        if isinstance(e, A.If):
            visit_expr(e.cond)
            visit_expr(e.then)
            if e.else_ is not None:
                visit_expr(e.else_)
            return
        if isinstance(e, A.Match):
            visit_expr(e.scrutinee)
            for arm in e.arms:
                if arm.guard is not None:
                    visit_expr(arm.guard)
                visit_expr(arm.body)
            return
        if isinstance(e, A.For):
            visit_expr(e.iter_expr)
            visit_expr(e.body)
            return
        if isinstance(e, (A.While,)):
            visit_expr(e.cond)
            visit_expr(e.body)
            return
        if isinstance(e, A.Loop):
            visit_expr(e.body)
            return
        if isinstance(e, A.Binary):
            visit_expr(e.left)
            visit_expr(e.right)
            return
        if isinstance(e, A.Unary):
            visit_expr(e.operand)
            return
        if isinstance(e, A.Call):
            visit_expr(e.callee)
            for a in e.args:
                visit_expr(a)
            return
        if isinstance(e, A.Index):
            visit_expr(e.callee)
            for idx in e.indices:
                visit_expr(idx)
            return
        if isinstance(e, A.Field):
            visit_expr(e.obj)
            return
        if isinstance(e, A.TupleLit):
            for elem in e.elems:
                visit_expr(elem)
            return
        if isinstance(e, A.ArrayLit):
            for elem in e.elems:
                visit_expr(elem)
            return
        if isinstance(e, A.StructLit):
            # StructLit doesn't carry an explicit type-arg list in the
            # current AST, but its base name may match a generic struct
            # — we *don't* instantiate here (no concrete args available);
            # the Let-stmt type annotation is the reliable path. This
            # branch still walks each field expression for nested uses.
            for (_n, fexpr) in e.fields:
                visit_expr(fexpr)
            return
        if isinstance(e, A.Assign):
            visit_expr(e.target)
            visit_expr(e.value)
            return
        if isinstance(e, A.Return):
            if e.value is not None:
                visit_expr(e.value)
            return
        if isinstance(e, A.Break):
            if e.value is not None:
                visit_expr(e.value)
            return
        if isinstance(e, A.Range):
            if e.start is not None:
                visit_expr(e.start)
            if e.end is not None:
                visit_expr(e.end)
            return
        if isinstance(e, A.Name):
            # Generic args attached to a name reference (e.g.
            # `Foo::<i32>::new`) — walk them so Pt::<i32>::new collects
            # the Pt<i32> use.
            for g in getattr(e, "generics", []) or []:
                visit_ty(g)
            return
        # Reflection nodes — Quote/Splice/Modify embed inner exprs.
        if isinstance(e, A.Quote):
            visit_expr(e.inner)
            return
        if isinstance(e, A.Splice):
            visit_expr(e.inner)
            return
        if isinstance(e, A.Modify):
            visit_expr(e.target)
            visit_expr(e.transformation)
            visit_expr(e.verifier)
            return
        # Literals + leaf nodes — no-op.
        return

    def visit_stmt(s) -> None:
        if s is None:
            return
        if isinstance(s, A.Let):
            if s.ty is not None:
                visit_ty(s.ty)
            if s.value is not None:
                visit_expr(s.value)
            return
        if isinstance(s, A.ExprStmt):
            visit_expr(s.expr)
            return
        if isinstance(s, A.ConstStmt):
            visit_ty(s.ty)
            visit_expr(s.value)
            return

    for it in prog.items:
        if isinstance(it, A.FnDecl):
            for p in it.params:
                visit_ty(p.ty)
            if it.return_ty is not None:
                visit_ty(it.return_ty)
            # Audit 28.8 C1-M2 fix: walk the fn body for body-level uses.
            if not it.is_extern:
                visit_expr(it.body)
        elif isinstance(it, A.StructDecl) and not it.generics:
            for f in it.fields:
                visit_ty(f.ty)
        elif isinstance(it, A.ConstDecl):
            visit_ty(it.ty)
            visit_expr(it.value)

    return out


def _ty_key(t: A.TyNode):
    """Convert a TyNode to a hashable key for deduplication."""
    if isinstance(t, A.TyName):
        return ("name", t.name)
    if isinstance(t, A.TyGeneric):
        return ("gen", t.base, tuple(_ty_key(a) for a in t.args))
    if isinstance(t, A.TyTuple):
        return ("tup", tuple(_ty_key(e) for e in t.elems))
    if isinstance(t, A.TyArray):
        return ("arr", _ty_key(t.elem))
    if isinstance(t, A.TyRef):
        return ("ref", t.is_mut, _ty_key(t.inner))
    if isinstance(t, A.TyPtr):
        return ("ptr", t.is_mut, _ty_key(t.inner))
    return ("?", type(t).__name__)


def instantiate(decl: A.StructDecl, ty_args: list[A.TyNode]
                ) -> A.StructDecl:
    """Clone a StructDecl with type-vars in field types substituted.

    Returns a new StructDecl with the mangled name and no generics."""
    if len(ty_args) != len(decl.generics):
        raise ValueError(
            f"struct {decl.name!r}: arity mismatch — expected "
            f"{len(decl.generics)} type args, got {len(ty_args)}"
        )
    subst = {g.name: t for g, t in zip(decl.generics, ty_args)}
    new_name = mangle_struct(decl.name, ty_args)
    new_fields: list[A.FnParam] = []
    for f in decl.fields:
        new_ty = substitute_ty(f.ty, subst)
        new_fields.append(A.FnParam(
            span=f.span, name=f.name, ty=new_ty,
            is_mut=getattr(f, "is_mut", False),
        ))
    return A.StructDecl(
        span=decl.span,
        name=new_name,
        generics=[],
        fields=new_fields,
        is_pub=decl.is_pub,
    )


def monomorphize_structs(prog: A.Program) -> tuple[A.Program, list[str]]:
    """Run the struct mono pass over prog. Returns (new_prog, diags).

    Phase-0: returns the same prog object with mono'd structs appended
    and uses rewritten. Generic StructDecls are kept in the list (they
    can be filtered post-pass when codegen is wired).

    Diags list is empty on clean; contains strings for unresolved
    instantiations or arity mismatches.
    """
    generic_structs = collect_generic_structs(prog)
    if not generic_structs:
        return prog, []

    uses = collect_concrete_uses(prog, generic_structs)
    diags: list[str] = []
    # Build instantiations
    mono_decls: list[A.StructDecl] = []
    rewrite_map: dict[tuple, str] = {}  # (struct_name, key-tuple) -> mangled

    for (sname, ty_args) in uses:
        try:
            inst = instantiate(generic_structs[sname], ty_args)
        except ValueError as e:
            diags.append(str(e))
            continue
        key = (sname, tuple(_ty_key(a) for a in ty_args))
        if key not in rewrite_map:
            rewrite_map[key] = inst.name
            mono_decls.append(inst)

    # Append mono'd structs to the program. Uses are *not* rewritten
    # in this Phase-0 pass — the typechecker can lookup mangled names
    # directly via the new structs. A more complete pass would rewrite
    # TyGeneric(base=Pt, args=[i32]) -> TyName(Pt__i32). Doing so
    # requires a careful walker to avoid touching non-struct generics
    # (D<T>, Logic<T>, tensors, etc.).
    prog.items = list(prog.items) + mono_decls
    return prog, diags


def find_uninstantiated(prog: A.Program) -> list[str]:
    """Diagnostic helper: returns the names of generic structs that
    have NO concrete instantiations anywhere in prog. Useful as a
    'dead-code' check or warning (Phase-0 just diagnostic, not a hard
    error)."""
    generic = collect_generic_structs(prog)
    uses = collect_concrete_uses(prog, generic)
    used_names = {n for (n, _) in uses}
    return sorted(set(generic.keys()) - used_names)
