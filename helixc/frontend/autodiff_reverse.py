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

import copy
from typing import Optional

from . import ast_nodes as A
from .autodiff import _inline_lets, _simplify, _inline_user_calls


def differentiate_reverse(expr: A.Expr, param_names: list[str],
                          fn_table: dict[str, "A.FnDecl"] | None = None
                          ) -> dict[str, A.Expr]:
    """Return a dict {param_name: ∂(expr)/∂(param_name), …} for each name in
    `param_names`. The expression is first inlined (user calls + let
    bindings) and the resulting derivatives are simplified.

    Pass `fn_table` to enable inlining of @pure user-defined function calls
    so the gradient propagates through them."""
    if fn_table:
        expr = _inline_user_calls(expr, fn_table)
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
        # Deepcopy `adj` whenever it appears in more than one place so
        # downstream in-place mutation passes (grad_pass alias resolution)
        # can't corrupt one branch by mutating the other.
        l, r, op = node.left, node.right, node.op
        if op == "+":
            _propagate(l, adj, acc)
            _propagate(r, copy.deepcopy(adj), acc)
        elif op == "-":
            _propagate(l, adj, acc)
            neg = A.Unary(span=node.span, op="-",
                          operand=copy.deepcopy(adj))
            _propagate(r, neg, acc)
        elif op == "*":
            adj_l = A.Binary(span=node.span, op="*",
                             left=adj, right=copy.deepcopy(r))
            adj_r = A.Binary(span=node.span, op="*",
                             left=copy.deepcopy(adj), right=copy.deepcopy(l))
            _propagate(l, adj_l, acc)
            _propagate(r, adj_r, acc)
        elif op == "/":
            # adj_l = adj / r
            adj_l = A.Binary(span=node.span, op="/",
                             left=adj, right=copy.deepcopy(r))
            # adj_r = -adj * l / (r * r)
            r_sq = A.Binary(span=node.span, op="*",
                            left=copy.deepcopy(r), right=copy.deepcopy(r))
            l_over_r2 = A.Binary(span=node.span, op="/",
                                 left=copy.deepcopy(l), right=r_sq)
            mag = A.Binary(span=node.span, op="*",
                           left=copy.deepcopy(adj), right=l_over_r2)
            adj_r = A.Unary(span=node.span, op="-", operand=mag)
            _propagate(l, adj_l, acc)
            _propagate(r, adj_r, acc)
        # Other ops (comparisons, etc) have zero local derivative for our cases.
        return
    if isinstance(node, A.Block):
        if node.final_expr is not None:
            _propagate(node.final_expr, adj, acc)
        return
    if isinstance(node, A.Call):
        # Chain rule for known transcendentals: propagate adj * f'(u) into u.
        if (isinstance(node.callee, A.Name) and len(node.args) == 1):
            name = node.callee.name
            u = node.args[0]

            def call1(fn: str, arg: A.Expr) -> A.Expr:
                return A.Call(span=node.span,
                              callee=A.Name(span=node.span, name=fn),
                              args=[arg])

            if name == "__exp":
                # adj_u = adj * exp(u)
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=call1("__exp", u))
                _propagate(u, new_adj, acc)
                return
            if name == "__log":
                # adj_u = adj * (1/u)
                recip = A.Binary(span=node.span, op="/",
                                 left=A.FloatLit(span=node.span, value=1.0),
                                 right=u)
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=recip)
                _propagate(u, new_adj, acc)
                return
            if name == "__sin":
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=call1("__cos", u))
                _propagate(u, new_adj, acc)
                return
            if name == "__cos":
                neg_sin = A.Unary(span=node.span, op="-",
                                  operand=call1("__sin", u))
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=neg_sin)
                _propagate(u, new_adj, acc)
                return
            if name == "__sqrt":
                sqrt_u = call1("__sqrt", u)
                denom = A.Binary(span=node.span, op="*",
                                 left=A.FloatLit(span=node.span, value=2.0),
                                 right=sqrt_u)
                recip = A.Binary(span=node.span, op="/",
                                 left=A.FloatLit(span=node.span, value=1.0),
                                 right=denom)
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=recip)
                _propagate(u, new_adj, acc)
                return
            if name == "__relu":
                # cond and else_ get distinct FloatLit(0.0) nodes — sharing
                # one would let in-place mutation passes corrupt both
                # places at once. (See C-1 audit fix in autodiff.py.)
                cond = A.Binary(span=node.span, op=">", left=u,
                                right=A.FloatLit(span=node.span, value=0.0))
                gated = A.If(span=node.span, cond=cond,
                             then=A.Block(span=node.span, stmts=[],
                                          final_expr=A.FloatLit(span=node.span, value=1.0)),
                             else_=A.Block(span=node.span, stmts=[],
                                           final_expr=A.FloatLit(span=node.span, value=0.0)))
                new_adj = A.Binary(span=node.span, op="*", left=adj, right=gated)
                _propagate(u, new_adj, acc)
                return
            if name == "__sigmoid":
                # Two distinct sigmoid(u) call nodes (deepcopy u for each)
                # so they don't share argument trees with each other.
                s1 = call1("__sigmoid", copy.deepcopy(u))
                s2 = call1("__sigmoid", copy.deepcopy(u))
                one_minus = A.Binary(span=node.span, op="-",
                                     left=A.FloatLit(span=node.span, value=1.0),
                                     right=s2)
                deriv = A.Binary(span=node.span, op="*", left=s1, right=one_minus)
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=deriv)
                _propagate(u, new_adj, acc)
                return
            if name == "__tanh":
                # d(tanh(u))/dx = (1 - tanh(u)^2) * du
                t = call1("__tanh", copy.deepcopy(u))
                t_sq = A.Binary(span=node.span, op="*",
                                left=t, right=copy.deepcopy(t))
                one_minus = A.Binary(span=node.span, op="-",
                                     left=A.FloatLit(span=node.span, value=1.0),
                                     right=t_sq)
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=one_minus)
                _propagate(u, new_adj, acc)
                return
            if name == "__softplus":
                # d(softplus(u))/dx = sigmoid(u) * du
                deriv = call1("__sigmoid", copy.deepcopy(u))
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=deriv)
                _propagate(u, new_adj, acc)
                return
            if name == "__silu":
                # d(silu)/du = sigmoid(u) * (1 + u*(1 - sigmoid(u)))
                s1 = call1("__sigmoid", copy.deepcopy(u))
                s2 = call1("__sigmoid", copy.deepcopy(u))
                one_minus_s = A.Binary(span=node.span, op="-",
                                       left=A.FloatLit(span=node.span, value=1.0),
                                       right=s2)
                u_times_oms = A.Binary(span=node.span, op="*",
                                       left=copy.deepcopy(u),
                                       right=one_minus_s)
                inner = A.Binary(span=node.span, op="+",
                                 left=A.FloatLit(span=node.span, value=1.0),
                                 right=u_times_oms)
                deriv = A.Binary(span=node.span, op="*", left=s1, right=inner)
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=deriv)
                _propagate(u, new_adj, acc)
                return
            if name == "__abs":
                # d(abs(u))/dx = sign(u) * du; at u=0 use 0.
                u_c = copy.deepcopy(u)
                zero = A.FloatLit(span=node.span, value=0.0)
                cond_pos = A.Binary(span=node.span, op=">", left=u_c,
                                    right=A.FloatLit(span=node.span, value=0.0))
                cond_neg = A.Binary(span=node.span, op="<", left=copy.deepcopy(u),
                                    right=A.FloatLit(span=node.span, value=0.0))
                inner_else = A.If(span=node.span, cond=cond_neg,
                                  then=A.Block(span=node.span, stmts=[],
                                               final_expr=A.FloatLit(span=node.span, value=-1.0)),
                                  else_=A.Block(span=node.span, stmts=[], final_expr=zero))
                gated = A.If(span=node.span, cond=cond_pos,
                             then=A.Block(span=node.span, stmts=[],
                                          final_expr=A.FloatLit(span=node.span, value=1.0)),
                             else_=A.Block(span=node.span, stmts=[],
                                           final_expr=inner_else))
                new_adj = A.Binary(span=node.span, op="*", left=adj, right=gated)
                _propagate(u, new_adj, acc)
                return
        # Other calls: opaque, contributes 0 to gradient.
        return
    if isinstance(node, A.If):
        # The runtime `if` picks one branch, so the gradient through it is also
        # an `if` — NOT a sum of both branches. Compute each branch's adjoint
        # contributions into separate buckets, then wrap them as
        # If(cond, sum_then, sum_else) per-parameter and append to the main
        # accumulator. (Cond's own derivative is zero — discrete choice.)
        then_acc: dict[str, list[A.Expr]] = {p: [] for p in acc}
        else_acc: dict[str, list[A.Expr]] = {p: [] for p in acc}

        def _into(branch: A.Expr | None, bucket: dict[str, list[A.Expr]],
                   adj_for_branch: A.Expr) -> None:
            if branch is None:
                return
            if isinstance(branch, A.Block):
                if branch.final_expr is not None:
                    _propagate(branch.final_expr, adj_for_branch, bucket)
            else:
                _propagate(branch, adj_for_branch, bucket)

        # Deepcopy adj for the else-branch so the two branches don't
        # share an adjoint AST node — same hazard the Binary rules
        # already deepcopy around.
        _into(node.then, then_acc, adj)
        _into(node.else_, else_acc, copy.deepcopy(adj))

        for p in acc:
            had_then = bool(then_acc[p])
            had_else = bool(else_acc[p])
            if not had_then and not had_else:
                continue
            zero = A.FloatLit(span=node.span, value=0.0)
            sum_then = _sum_exprs(then_acc[p], node.span) if had_then else zero
            sum_else = _sum_exprs(else_acc[p], node.span) if had_else else zero
            # Deep-copy the cond so the gradient AST doesn't share a
            # reference with the original program. Subsequent passes
            # (e.g. grad_pass._resolve_let_aliases) mutate Call/Name
            # nodes in-place; without this clone, mutation of the
            # original cond would silently propagate to the gradient.
            cond_copy = copy.deepcopy(node.cond)
            wrapped = A.If(
                span=node.span,
                cond=cond_copy,
                then=A.Block(span=node.span, stmts=[], final_expr=sum_then),
                else_=A.Block(span=node.span, stmts=[], final_expr=sum_else),
            )
            acc[p].append(wrapped)
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
