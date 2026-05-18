"""
helixc/frontend/autotune_expand.py — Stage 56 / Tier 2 #8.

Expand `@autotune @kernel` functions into N specialized variants — one
per cross-product configuration. Each variant is a deep-copy of the
original FnDecl with:
  - Renamed to mangled_variant_name(fn_name, cfg)
  - `autotune` attr stripped
  - Body walked to replace `Name(KEY)` references with `IntLit(VAL)`
    for each KEY in cfg (compile-time constant specialization)

This is the missing piece between Stage 27's autotune parsing/
validation (helixc/frontend/autotune.py) and Stage 16's PTX kernel
emission (helixc/backend/ptx.py). Pre-Stage-56, only ONE variant per
autotune fn was emitted (the un-specialized original); now each config
in the Cartesian product gets its own PTX `.entry` block.

License: Apache 2.0
"""

from __future__ import annotations

import copy

from . import ast_nodes as A
from .autotune import (
    autotune_variants,
    has_autotune,
    has_kernel,
    mangled_variant_name,
    parse_autotune_attrs,
)


def _substitute_autotune_consts(expr: A.Expr, cfg: dict[str, int]) -> A.Expr:
    """Recursively replace `Name(KEY)` with `IntLit(cfg[KEY])` for each
    KEY in cfg. Returns a NEW AST tree (deepcopy-safe). Used to
    specialize an autotune variant's body to its compile-time config
    values."""
    if expr is None:
        return expr
    if isinstance(expr, A.Name):
        if expr.name in cfg:
            return A.IntLit(span=expr.span, value=cfg[expr.name])
        return expr
    if isinstance(expr, (A.IntLit, A.FloatLit, A.BoolLit,
                          A.StrLit, A.CharLit, A.Path,
                          A.Continue)):
        return expr
    if isinstance(expr, A.Unary):
        return A.Unary(
            span=expr.span, op=expr.op,
            operand=_substitute_autotune_consts(expr.operand, cfg))
    if isinstance(expr, A.Binary):
        return A.Binary(
            span=expr.span, op=expr.op,
            left=_substitute_autotune_consts(expr.left, cfg),
            right=_substitute_autotune_consts(expr.right, cfg))
    if isinstance(expr, A.Call):
        return A.Call(
            span=expr.span,
            callee=_substitute_autotune_consts(expr.callee, cfg),
            args=[_substitute_autotune_consts(a, cfg)
                  for a in expr.args])
    if isinstance(expr, A.Cast):
        return A.Cast(
            span=expr.span,
            value=_substitute_autotune_consts(expr.value, cfg),
            target_ty=expr.target_ty)
    if isinstance(expr, A.Index):
        return A.Index(
            span=expr.span,
            callee=_substitute_autotune_consts(expr.callee, cfg),
            indices=[_substitute_autotune_consts(i, cfg)
                     for i in expr.indices])
    if isinstance(expr, A.Field):
        return A.Field(
            span=expr.span,
            obj=_substitute_autotune_consts(expr.obj, cfg),
            name=expr.name)
    if isinstance(expr, A.TupleLit):
        return A.TupleLit(
            span=expr.span,
            elems=[_substitute_autotune_consts(e, cfg)
                   for e in expr.elems])
    if isinstance(expr, A.ArrayLit):
        return A.ArrayLit(
            span=expr.span,
            elems=[_substitute_autotune_consts(e, cfg)
                   for e in expr.elems])
    if isinstance(expr, A.StructLit):
        return A.StructLit(
            span=expr.span, name=expr.name,
            fields=[(n, _substitute_autotune_consts(v, cfg))
                    for (n, v) in expr.fields])
    if isinstance(expr, A.If):
        return A.If(
            span=expr.span,
            cond=_substitute_autotune_consts(expr.cond, cfg),
            then=_substitute_autotune_consts(expr.then, cfg),
            else_=(_substitute_autotune_consts(expr.else_, cfg)
                   if expr.else_ is not None else None))
    if isinstance(expr, A.Block):
        new_stmts = []
        for s in expr.stmts:
            if isinstance(s, A.Let) and s.value is not None:
                new_stmts.append(A.Let(
                    span=s.span, name=s.name, is_mut=s.is_mut,
                    ty=s.ty,
                    value=_substitute_autotune_consts(s.value, cfg)))
            elif isinstance(s, A.ConstStmt):
                new_stmts.append(A.ConstStmt(
                    span=s.span, name=s.name, ty=s.ty,
                    value=_substitute_autotune_consts(s.value, cfg)))
            elif isinstance(s, A.ExprStmt):
                new_stmts.append(A.ExprStmt(
                    span=s.span,
                    expr=_substitute_autotune_consts(s.expr, cfg)))
            else:
                new_stmts.append(s)
        new_final = (_substitute_autotune_consts(expr.final_expr, cfg)
                     if expr.final_expr is not None else None)
        return A.Block(span=expr.span, stmts=new_stmts,
                       final_expr=new_final)
    if isinstance(expr, A.For):
        return A.For(
            span=expr.span, var_name=expr.var_name,
            iter_expr=_substitute_autotune_consts(expr.iter_expr, cfg),
            body=_substitute_autotune_consts(expr.body, cfg))
    if isinstance(expr, A.While):
        return A.While(
            span=expr.span,
            cond=_substitute_autotune_consts(expr.cond, cfg),
            body=_substitute_autotune_consts(expr.body, cfg))
    if isinstance(expr, A.Loop):
        return A.Loop(
            span=expr.span,
            body=_substitute_autotune_consts(expr.body, cfg))
    if isinstance(expr, A.UnsafeBlock):
        return A.UnsafeBlock(
            span=expr.span,
            body=_substitute_autotune_consts(expr.body, cfg))
    if isinstance(expr, A.Assign):
        return A.Assign(
            span=expr.span, op=expr.op,
            target=_substitute_autotune_consts(expr.target, cfg),
            value=_substitute_autotune_consts(expr.value, cfg))
    if isinstance(expr, A.Return):
        return A.Return(
            span=expr.span,
            value=(_substitute_autotune_consts(expr.value, cfg)
                   if expr.value is not None else None))
    if isinstance(expr, A.Break):
        return A.Break(
            span=expr.span,
            value=(_substitute_autotune_consts(expr.value, cfg)
                   if expr.value is not None else None))
    if isinstance(expr, A.Range):
        return A.Range(
            span=expr.span,
            start=(_substitute_autotune_consts(expr.start, cfg)
                   if expr.start is not None else None),
            end=(_substitute_autotune_consts(expr.end, cfg)
                 if expr.end is not None else None))
    if isinstance(expr, A.Match):
        new_arms = []
        for arm in expr.arms:
            new_arms.append(A.MatchArm(
                span=arm.span, pattern=arm.pattern,
                guard=(_substitute_autotune_consts(arm.guard, cfg)
                       if arm.guard is not None else None),
                body=_substitute_autotune_consts(arm.body, cfg)))
        return A.Match(
            span=expr.span,
            scrutinee=_substitute_autotune_consts(expr.scrutinee, cfg),
            arms=new_arms)
    return expr


def _expand_one_fn(fn: A.FnDecl) -> list[A.FnDecl]:
    """Expand a single @autotune @kernel fn into N specialized variants.

    Returns a list of N new FnDecls (NOT including the original).
    Caller is responsible for removing the original from the program.
    """
    params, _diags = parse_autotune_attrs(fn)
    variants = autotune_variants(params)
    out: list[A.FnDecl] = []
    for cfg in variants:
        clone = copy.deepcopy(fn)
        # Rename to the mangled variant name.
        clone.name = mangled_variant_name(fn.name, cfg)
        # Strip the autotune attr from the clone (its config is
        # baked in as constants). Keep @kernel + other attrs.
        clone.attrs = [a for a in clone.attrs
                       if a != "autotune"
                       and not a.startswith("autotune:")
                       and not a.startswith("autotune_product")
                       and not a.startswith("autotune_parse_error_kind")]
        # Add a marker attr recording the config (informational;
        # downstream passes can introspect it for debug output).
        for k, v in cfg.items():
            clone.attrs.append(f"autotune_config:{k}={v}")
        # Substitute Name(KEY) -> IntLit(VAL) in the body.
        if clone.body is not None:
            clone.body = _substitute_autotune_consts(clone.body, cfg)
        out.append(clone)
    return out


def expand_autotune_kernels(prog: A.Program) -> A.Program:
    """For each @autotune @kernel fn in `prog`, emit N specialized
    variants in place of the original. Non-autotune fns pass through
    unchanged.

    Returns a NEW Program (no mutation of the input). Cascade-safe
    because non-autotune programs are byte-identical to their input.
    """
    new_items: list = []
    for it in prog.items:
        if (isinstance(it, A.FnDecl)
                and has_autotune(it)
                and has_kernel(it)):
            expanded = _expand_one_fn(it)
            if expanded:
                new_items.extend(expanded)
            else:
                # Empty autotune (no variants — empty params): keep
                # original as the single variant. Defensive.
                new_items.append(it)
        else:
            new_items.append(it)
    new_prog = copy.copy(prog)
    new_prog.items = new_items
    return new_prog


def autotune_variant_names_for(fn: A.FnDecl) -> list[str]:
    """Stage 59 follow-on / Tier 2 #8 polish — return the list of
    variant names that would be emitted by `expand_autotune_kernels`
    for this fn, WITHOUT performing the full expansion.

    For a non-@autotune or non-@kernel fn, returns the singleton
    `[fn.name]` (the fn would pass through unchanged).

    Use cases:
    - Pre-compute how many variants a fn will generate before
      committing to the expanded compilation cost.
    - Inventory the variant namespace for symbol-table reservation.
    - Drive a downstream pass that needs to know the variant names
      (e.g., a benchmark harness that iterates them).
    """
    if not (has_autotune(fn) and has_kernel(fn)):
        return [fn.name]
    params, _diags = parse_autotune_attrs(fn)
    variants = autotune_variants(params)
    if not variants:
        # Defensive: empty params → single original name.
        return [fn.name]
    return [mangled_variant_name(fn.name, cfg) for cfg in variants]


def autotune_variant_count_for(fn: A.FnDecl) -> int:
    """Stage 59 follow-on / Tier 2 #8 polish — Cartesian-product
    cardinality for an @autotune @kernel fn. Convenience shortcut
    for `len(autotune_variant_names_for(fn))`."""
    return len(autotune_variant_names_for(fn))


def autotune_expansion_summary(prog: A.Program) -> dict:
    """Stage 59 follow-on / Tier 2 #8 polish — summary of the variant
    expansion that `expand_autotune_kernels(prog)` would produce.

    Returns a dict {fn_name: variant_count} for every @autotune
    @kernel fn in the program. Non-autotune fns are omitted (they
    don't get expanded).

    Use case: budget the compilation cost ahead of time. If the sum
    of variant counts is very large, the caller can prune the
    autotune-params before invoking expand_autotune_kernels.
    """
    summary: dict = {}
    for it in prog.items:
        if (isinstance(it, A.FnDecl)
                and has_autotune(it)
                and has_kernel(it)):
            summary[it.name] = autotune_variant_count_for(it)
    return summary
