"""
helixc/frontend/grad_pass.py — compile-time `grad(f)` rewriting pass.

When the program contains an expression `grad(loss)`, this pass:
1. Looks up `loss` in the program's function table
2. Symbolically differentiates loss's body via autodiff.differentiate()
3. Generates a new FnDecl `loss__grad` with the derivative as body
4. Adds the new FnDecl to the program
5. Rewrites `grad(loss)` -> `loss__grad` (a Name expression)

After this pass, `grad(loss)(x)` becomes `loss__grad(x)` — a normal
function call. The compiler doesn't need any new runtime support.

This is the missing wire-up between the autodiff engine and the language.

License: Apache 2.0
"""

from __future__ import annotations

from . import ast_nodes as A
from .autodiff import differentiate, _inline_lets


def grad_pass(prog: A.Program) -> int:
    """Walk the program; rewrite all grad(f) calls into references to
    generated f__grad functions. Returns count of grad calls rewritten."""
    # First: index existing functions by name
    fn_by_name: dict[str, A.FnDecl] = {}
    for item in prog.items:
        if isinstance(item, A.FnDecl):
            fn_by_name[item.name] = item

    new_fns: list[A.FnDecl] = []
    rewrite_count = 0

    # Walk all function bodies
    for item in prog.items:
        if isinstance(item, A.FnDecl):
            rewrite_count += _rewrite_in_block(item.body, fn_by_name, new_fns)

    # Add generated grad functions to the program
    for new_fn in new_fns:
        if new_fn.name not in fn_by_name:
            prog.items.append(new_fn)
            fn_by_name[new_fn.name] = new_fn

    return rewrite_count


def _rewrite_in_block(block: A.Block, fn_by_name: dict[str, A.FnDecl],
                      new_fns: list[A.FnDecl]) -> int:
    count = 0
    for stmt in block.stmts:
        if isinstance(stmt, A.Let) and stmt.value is not None:
            new_val, c = _rewrite_in_expr(stmt.value, fn_by_name, new_fns)
            stmt.value = new_val
            count += c
        elif isinstance(stmt, A.ExprStmt):
            new_e, c = _rewrite_in_expr(stmt.expr, fn_by_name, new_fns)
            stmt.expr = new_e
            count += c
        elif isinstance(stmt, A.ConstStmt):
            new_v, c = _rewrite_in_expr(stmt.value, fn_by_name, new_fns)
            stmt.value = new_v
            count += c
    if block.final_expr is not None:
        new_e, c = _rewrite_in_expr(block.final_expr, fn_by_name, new_fns)
        block.final_expr = new_e
        count += c
    return count


def _rewrite_in_expr(expr: A.Expr, fn_by_name: dict[str, A.FnDecl],
                     new_fns: list[A.FnDecl]) -> tuple[A.Expr, int]:
    count = 0
    if isinstance(expr, A.Call):
        # Check if this is grad(f)
        if (isinstance(expr.callee, A.Name) and expr.callee.name == "grad"
                and len(expr.args) == 1
                and isinstance(expr.args[0], A.Name)
                and expr.args[0].name in fn_by_name):
            target = fn_by_name[expr.args[0].name]
            grad_fn = _generate_grad_fn(target)
            if grad_fn is not None:
                new_fns.append(grad_fn)
                # Replace grad(f) with a Name pointing at f__grad
                return (A.Name(span=expr.span, name=grad_fn.name), 1)
        # Recurse into callee + args
        new_callee, c1 = _rewrite_in_expr(expr.callee, fn_by_name, new_fns)
        new_args = []
        c2 = 0
        for a in expr.args:
            na, ca = _rewrite_in_expr(a, fn_by_name, new_fns)
            new_args.append(na)
            c2 += ca
        return (A.Call(span=expr.span, callee=new_callee, args=new_args),
                c1 + c2)
    if isinstance(expr, A.Binary):
        l, c1 = _rewrite_in_expr(expr.left, fn_by_name, new_fns)
        r, c2 = _rewrite_in_expr(expr.right, fn_by_name, new_fns)
        return (A.Binary(span=expr.span, op=expr.op, left=l, right=r),
                c1 + c2)
    if isinstance(expr, A.Unary):
        sub, c = _rewrite_in_expr(expr.operand, fn_by_name, new_fns)
        return (A.Unary(span=expr.span, op=expr.op, operand=sub), c)
    if isinstance(expr, A.Block):
        c = _rewrite_in_block(expr, fn_by_name, new_fns)
        return (expr, c)
    if isinstance(expr, A.If):
        c, c1 = _rewrite_in_expr(expr.cond, fn_by_name, new_fns)
        expr.cond = c
        c2 = _rewrite_in_block(expr.then, fn_by_name, new_fns)
        c3 = 0
        if expr.else_ is not None:
            if isinstance(expr.else_, A.Block):
                c3 = _rewrite_in_block(expr.else_, fn_by_name, new_fns)
        return (expr, c1 + c2 + c3)
    return (expr, count)


def _generate_grad_fn(fn: A.FnDecl) -> A.FnDecl | None:
    """Build a `<fn.name>__grad` FnDecl whose body is the derivative of
    `fn`'s body w.r.t. its first parameter."""
    if not fn.params:
        return None
    var = fn.params[0].name
    deriv = differentiate(fn.body, var)
    # Wrap the derivative expression in a block (the FnDecl expects a Block body)
    new_body = A.Block(span=fn.body.span, stmts=[], final_expr=deriv)

    # Build new params (same names, all f32 — gradient takes plain floats)
    new_params = [
        A.FnParam(span=p.span, name=p.name,
                  ty=A.TyName(span=p.ty.span, name="f32"))
        for p in fn.params
    ]

    return A.FnDecl(
        span=fn.span,
        name=f"{fn.name}__grad",
        generics=[],
        params=new_params,
        return_ty=A.TyName(span=fn.span, name="f32"),
        where_clauses=[],
        body=new_body,
        attrs=["pure"],
        is_pub=fn.is_pub,
    )
