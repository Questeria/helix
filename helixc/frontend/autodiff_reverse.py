"""
helixc/frontend/autodiff_reverse.py — symbolic reverse-mode automatic differentiation.

Where forward-mode (autodiff.py) computes ∂(expr)/∂(var) by propagating tangents
forward through the expression tree, reverse-mode propagates an adjoint backward
from the output. For a scalar-output function f(x_1, …, x_n), reverse-mode
computes ALL n gradients in a single backward sweep, whereas forward-mode needs
n separate sweeps.

Algorithmically:
  forward:   tangent at output = product of local tangents along path
  reverse:   adjoint at input  = sum of (output_adjoint * local_jacobian)

For symbolic AD over an inlined expression tree the surface result is the same
as forward-mode for any single ∂f/∂x_i, but the engine produces the WHOLE
gradient {x_1: ∂f/∂x_1, …, x_n: ∂f/∂x_n} from one traversal — which is the
shape needed for multi-output IR generation later.

Algorithm:
  1. _inline_lets to flatten let-bindings.
  2. Walk the tree top-down with a current adjoint expression. For binary ops,
     split the adjoint to each operand by its local Jacobian:
        +  : adj_l = adj,        adj_r = adj
        -  : adj_l = adj,        adj_r = -adj
        *  : adj_l = adj * r,    adj_r = adj * l
        /  : adj_l = adj / r,    adj_r = -adj * l / (r * r)
        neg: adj_op = -adj
  3. At each Name node referencing a parameter, accumulate adj into that
     parameter's bucket.
  4. After the walk, sum each parameter's bucket into the gradient.

Because the same parameter may appear multiple times in the inlined tree, the
final gradient is the sum of contributions across all occurrences — handled by
the simplifier in _sum_exprs (a chain of binary +).

Currently supported AST nodes: IntLit, FloatLit, BoolLit, Name, Unary("-"),
Binary("+", "-", "*", "/"). Calls and ifs are not yet supported.

License: Apache 2.0
"""

from __future__ import annotations

from typing import Optional

from . import ast_nodes as A
from .autodiff import _inline_lets, _simplify


def differentiate_reverse(expr: A.Expr, param_names: list[str]) -> dict[str, A.Expr]:
    """Return a dict {param_name: ∂(expr)/∂(param_name), …} for each name in
    `param_names`. The expression is first inlined (let-bindings substituted)
    and the resulting derivatives are simplified."""
    flat = _inline_lets(expr, {})
    if flat is None:
        return {p: A.FloatLit(span=expr.span, value=0.0) for p in param_names}

    # Buckets: param_name -> list of adjoint expressions to be summed
    acc: dict[str, list[A.Expr]] = {p: [] for p in param_names}
    seed = A.FloatLit(span=flat.span, value=1.0)
    _propagate(flat, seed, acc)
    return {
        p: _simplify(_sum_exprs(acc[p], flat.span))
        for p in param_names
    }


def _propagate(node: A.Expr, adj: A.Expr, acc: dict[str, list[A.Expr]]) -> None:
    """Send the adjoint `adj` through `node`, depositing contributions into
    `acc[name]` for each parameter Name encountered."""
    if isinstance(node, (A.IntLit, A.FloatLit, A.BoolLit, A.StrLit, A.CharLit)):
        return
    if isinstance(node, A.Name):
        if node.name in acc:
            acc[node.name].append(adj)
        return
    if isinstance(node, A.Unary):
        if node.op == "-":
            neg = A.Unary(span=node.span, op="-", operand=adj)
            _propagate(node.operand, neg, acc)
        return
    if isinstance(node, A.Binary):
        l, r, op = node.left, node.right, node.op
        if op == "+":
            _propagate(l, adj, acc)
            _propagate(r, adj, acc)
        elif op == "-":
            _propagate(l, adj, acc)
            neg = A.Unary(span=node.span, op="-", operand=adj)
            _propagate(r, neg, acc)
        elif op == "*":
            adj_l = A.Binary(span=node.span, op="*", left=adj, right=r)
            adj_r = A.Binary(span=node.span, op="*", left=adj, right=l)
            _propagate(l, adj_l, acc)
            _propagate(r, adj_r, acc)
        elif op == "/":
            # adj_l = adj / r
            adj_l = A.Binary(span=node.span, op="/", left=adj, right=r)
            # adj_r = -adj * l / (r * r)
            r_sq = A.Binary(span=node.span, op="*", left=r, right=r)
            l_over_r2 = A.Binary(span=node.span, op="/", left=l, right=r_sq)
            mag = A.Binary(span=node.span, op="*", left=adj, right=l_over_r2)
            adj_r = A.Unary(span=node.span, op="-", operand=mag)
            _propagate(l, adj_l, acc)
            _propagate(r, adj_r, acc)
        # Other ops (comparisons, etc) have zero local derivative for our cases.
        return
    if isinstance(node, A.Block):
        if node.final_expr is not None:
            _propagate(node.final_expr, adj, acc)
        return
    if isinstance(node, A.If):
        # Differentiating across both branches: deposit adj into both then & else.
        # (Cond's derivative is zero — it's a discrete choice, not differentiable.)
        # If else is missing, treat as zero contribution from the missing branch.
        if isinstance(node.then, A.Block) and node.then.final_expr is not None:
            _propagate(node.then.final_expr, adj, acc)
        if node.else_ is not None:
            if isinstance(node.else_, A.Block) and node.else_.final_expr is not None:
                _propagate(node.else_.final_expr, adj, acc)
            elif not isinstance(node.else_, A.Block):
                _propagate(node.else_, adj, acc)
        return
    # Unsupported nodes: silently produce no contribution.


def _sum_exprs(exprs: list[A.Expr], span: A.Span) -> A.Expr:
    if not exprs:
        return A.FloatLit(span=span, value=0.0)
    if len(exprs) == 1:
        return exprs[0]
    out = exprs[0]
    for e in exprs[1:]:
        out = A.Binary(span=span, op="+", left=out, right=e)
    return out
