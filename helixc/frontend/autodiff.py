"""
helixc/frontend/autodiff.py — source-level forward-mode automatic differentiation.

When the user writes `grad(loss)`, the compiler walks loss's AST body and
generates a derivative AST. This module provides `differentiate(expr, var)`
that returns the symbolic derivative of `expr` with respect to `var`.

Supported expressions:
- IntLit / FloatLit  (derivative is 0)
- Name == var        (derivative is 1)
- Name != var        (derivative is 0)
- Binary +, -        (linearity)
- Binary *           (product rule)
- Binary /           (quotient rule)
- Unary -            (negation)
- Calls              (NOT YET — would need chain rule + known derivatives
                      of builtin functions)
- Block / If         (NOT YET — needs control-flow handling)

This is forward-mode AD. For ML loss functions you'd typically want
reverse-mode; that's a future enhancement.

License: Apache 2.0
"""

from __future__ import annotations

from typing import Optional

from . import ast_nodes as A


def differentiate(expr: A.Expr, var: str) -> A.Expr:
    """Return the AST of d(expr)/d(var), simplified.

    If `expr` is a Block, the block's let-bindings are inlined first so
    that subsequent uses of the bound names refer to their definitions.
    Then the block's final expression is differentiated.
    """
    inlined = _inline_lets(expr, {})
    deriv = _diff(inlined, var)
    return _simplify(deriv)


def _inline_lets(expr: A.Expr | None, env: dict[str, A.Expr]) -> A.Expr | None:
    """Walk expr, replacing references to let-bound names with the bound
    expression. Used to flatten blocks before differentiation."""
    if expr is None:
        return None
    if isinstance(expr, (A.IntLit, A.FloatLit, A.BoolLit, A.StrLit, A.CharLit)):
        return expr
    if isinstance(expr, A.Name):
        if expr.name in env:
            return env[expr.name]
        return expr
    if isinstance(expr, A.Unary):
        return A.Unary(span=expr.span, op=expr.op,
                       operand=_inline_lets(expr.operand, env))
    if isinstance(expr, A.Binary):
        return A.Binary(span=expr.span, op=expr.op,
                        left=_inline_lets(expr.left, env),
                        right=_inline_lets(expr.right, env))
    if isinstance(expr, A.Block):
        local_env = dict(env)
        for stmt in expr.stmts:
            if isinstance(stmt, A.Let) and stmt.value is not None:
                local_env[stmt.name] = _inline_lets(stmt.value, local_env)
            # ExprStmt: ignore (no derivative meaning)
            # ConstStmt: similar to Let
            elif isinstance(stmt, A.ConstStmt):
                local_env[stmt.name] = _inline_lets(stmt.value, local_env)
        if expr.final_expr is not None:
            return _inline_lets(expr.final_expr, local_env)
        return A.FloatLit(span=expr.span, value=0.0)
    if isinstance(expr, A.If):
        # Inline both branches and re-wrap in an If — the inliner only flattens
        # let-bindings, branch selection stays a runtime decision. Differentiate
        # both branches; the derivative is then the same conditional.
        new_then = _inline_lets(expr.then, env) if isinstance(expr.then, A.Block) else expr.then
        new_else = None
        if expr.else_ is not None:
            if isinstance(expr.else_, A.Block):
                new_else = _inline_lets(expr.else_, env)
            else:
                new_else = _inline_lets(expr.else_, env)
        # Wrap any non-block result in a Block(final_expr=) so If's children
        # are valid. The inliner returns expressions, not blocks.
        def _wrap(e: A.Expr | None) -> A.Block | None:
            if e is None:
                return None
            if isinstance(e, A.Block):
                return e
            return A.Block(span=e.span, stmts=[], final_expr=e)
        wrapped_then = _wrap(new_then)
        wrapped_else = _wrap(new_else)
        return A.If(span=expr.span, cond=expr.cond,
                    then=wrapped_then, else_=wrapped_else)
    return expr


# ============================================================================
# Differentiation rules
# ============================================================================
def _diff(expr: A.Expr, var: str) -> A.Expr:
    """Recursively compute the derivative AST."""
    span = expr.span
    if isinstance(expr, A.IntLit):
        return A.IntLit(span=span, value=0)
    if isinstance(expr, A.FloatLit):
        return A.FloatLit(span=span, value=0.0)
    if isinstance(expr, A.BoolLit):
        return A.IntLit(span=span, value=0)
    if isinstance(expr, A.Name):
        if expr.name == var:
            return A.FloatLit(span=span, value=1.0)
        return A.FloatLit(span=span, value=0.0)
    if isinstance(expr, A.Unary) and expr.op == "-":
        # d(-a)/dx = -da/dx
        return A.Unary(span=span, op="-", operand=_diff(expr.operand, var))
    if isinstance(expr, A.Binary):
        l = expr.left
        r = expr.right
        dl = _diff(l, var)
        dr = _diff(r, var)
        if expr.op == "+":
            # d(a+b)/dx = da/dx + db/dx
            return A.Binary(span=span, op="+", left=dl, right=dr)
        if expr.op == "-":
            return A.Binary(span=span, op="-", left=dl, right=dr)
        if expr.op == "*":
            # Product rule: d(a*b)/dx = (da/dx)*b + a*(db/dx)
            term1 = A.Binary(span=span, op="*", left=dl, right=r)
            term2 = A.Binary(span=span, op="*", left=l, right=dr)
            return A.Binary(span=span, op="+", left=term1, right=term2)
        if expr.op == "/":
            # Quotient rule: d(a/b)/dx = (da*b - a*db) / (b*b)
            num1 = A.Binary(span=span, op="*", left=dl, right=r)
            num2 = A.Binary(span=span, op="*", left=l, right=dr)
            num = A.Binary(span=span, op="-", left=num1, right=num2)
            denom = A.Binary(span=span, op="*", left=r, right=r)
            return A.Binary(span=span, op="/", left=num, right=denom)
    if isinstance(expr, A.If):
        # d/dx (if c then a else b) = if c then da/dx else db/dx.
        # Cond contributes nothing — it's a discrete choice, not differentiable.
        d_then = _diff_block_or_expr(expr.then, var, span)
        d_else = (_diff_block_or_expr(expr.else_, var, span)
                  if expr.else_ is not None
                  else A.Block(span=span, stmts=[], final_expr=A.FloatLit(span=span, value=0.0)))
        return A.If(span=span, cond=expr.cond, then=d_then, else_=d_else)
    if isinstance(expr, A.Block):
        return _diff_block_or_expr(expr, var, span)
    # Unsupported: emit zero (placeholder)
    return A.FloatLit(span=expr.span, value=0.0)


def _diff_block_or_expr(node: A.Expr | A.Block, var: str, span: A.Span) -> A.Block:
    """Differentiate a Block by differentiating its final_expr; or wrap a bare
    Expr in a single-final-expr block. The result is always a Block, suitable
    for use as a then/else child of an If."""
    if isinstance(node, A.Block):
        if node.final_expr is None:
            return A.Block(span=span, stmts=[], final_expr=A.FloatLit(span=span, value=0.0))
        d = _diff(node.final_expr, var)
        return A.Block(span=node.span, stmts=[], final_expr=d)
    d = _diff(node, var)
    return A.Block(span=span, stmts=[], final_expr=d)


# ============================================================================
# Simplification — fold trivial terms (0+x, x+0, 0*x, 1*x, etc.)
# ============================================================================
def _simplify(expr: A.Expr) -> A.Expr:
    if isinstance(expr, A.Binary):
        l = _simplify(expr.left)
        r = _simplify(expr.right)
        # Fold constant arithmetic
        l_val = _const_value(l)
        r_val = _const_value(r)
        if l_val is not None and r_val is not None:
            try:
                if expr.op == "+":
                    return _make_const(l_val + r_val, expr.span)
                if expr.op == "-":
                    return _make_const(l_val - r_val, expr.span)
                if expr.op == "*":
                    return _make_const(l_val * r_val, expr.span)
                if expr.op == "/" and r_val != 0:
                    return _make_const(l_val / r_val, expr.span)
            except Exception:
                pass
        # 0 + x = x
        if expr.op == "+":
            if _is_zero(l):
                return r
            if _is_zero(r):
                return l
        # x - 0 = x;  0 - x = -x
        if expr.op == "-":
            if _is_zero(r):
                return l
            if _is_zero(l):
                return A.Unary(span=expr.span, op="-", operand=r)
        # 0 * x = 0;  x * 0 = 0;  1 * x = x;  x * 1 = x
        if expr.op == "*":
            if _is_zero(l) or _is_zero(r):
                return A.FloatLit(span=expr.span, value=0.0)
            if _is_one(l):
                return r
            if _is_one(r):
                return l
        return A.Binary(span=expr.span, op=expr.op, left=l, right=r)
    if isinstance(expr, A.Unary):
        sub = _simplify(expr.operand)
        # -(-x) = x
        if expr.op == "-" and isinstance(sub, A.Unary) and sub.op == "-":
            return sub.operand
        # -0 = 0
        if expr.op == "-" and _is_zero(sub):
            return A.FloatLit(span=expr.span, value=0.0)
        return A.Unary(span=expr.span, op=expr.op, operand=sub)
    if isinstance(expr, A.If):
        # Recursively simplify branches.
        new_then = _simplify_block(expr.then) if expr.then is not None else None
        new_else = _simplify_block(expr.else_) if expr.else_ is not None else None
        return A.If(span=expr.span, cond=expr.cond, then=new_then, else_=new_else)
    if isinstance(expr, A.Block):
        return _simplify_block(expr)
    return expr


def _simplify_block(blk: A.Block) -> A.Block:
    if blk.final_expr is None:
        return blk
    return A.Block(span=blk.span, stmts=blk.stmts,
                   final_expr=_simplify(blk.final_expr))


def _is_zero(e: A.Expr) -> bool:
    return ((isinstance(e, A.IntLit) and e.value == 0)
            or (isinstance(e, A.FloatLit) and e.value == 0.0))


def _is_one(e: A.Expr) -> bool:
    return ((isinstance(e, A.IntLit) and e.value == 1)
            or (isinstance(e, A.FloatLit) and e.value == 1.0))


def _const_value(e: A.Expr):
    if isinstance(e, A.IntLit):
        return e.value
    if isinstance(e, A.FloatLit):
        return e.value
    if isinstance(e, A.Unary) and e.op == "-":
        v = _const_value(e.operand)
        if v is not None:
            return -v
    return None


def _make_const(value, span: A.Span) -> A.Expr:
    if isinstance(value, int):
        return A.IntLit(span=span, value=value)
    return A.FloatLit(span=span, value=float(value))


# ============================================================================
# Pretty print (for testing / showing derivatives)
# ============================================================================
def fmt(expr: A.Expr) -> str:
    if isinstance(expr, A.IntLit):
        return str(expr.value)
    if isinstance(expr, A.FloatLit):
        return f"{expr.value:g}"
    if isinstance(expr, A.Name):
        return expr.name
    if isinstance(expr, A.Binary):
        return f"({fmt(expr.left)} {expr.op} {fmt(expr.right)})"
    if isinstance(expr, A.Unary):
        return f"({expr.op}{fmt(expr.operand)})"
    return f"<{type(expr).__name__}>"
