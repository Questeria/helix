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
  3. At each Name or Field path node referencing a parameter, accumulate adj
     into that parameter's bucket.
  4. After the walk, sum each parameter's bucket into the gradient.

Because the same parameter may appear multiple times in the inlined tree, the
final gradient is the sum of contributions across all occurrences — handled by
the simplifier in _sum_exprs (a chain of binary +).

Currently supported AST nodes: IntLit, FloatLit, BoolLit, Name, Field,
Unary("-"), Binary("+", "-", "*", "/"), Call, If, Match, Cast, UnsafeBlock.

License: Apache 2.0
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Optional

from . import ast_nodes as A
from .autodiff import (
    _inline_lets, _simplify, _inline_user_calls, _ad_warn,
    NUMERIC_FOR_AD,
    AD_INTEGER_VALUED_LOGIC, _raise_integer_logic_in_ad,
    _IDENTITY_AD_CHAIN_RULE_NAMES,
    _name_appears_in,
)


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
    flat = _inline_lets(expr, {}, mode="reverse")
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


def _field_path(node: A.Expr) -> Optional[str]:
    if isinstance(node, A.Name):
        return node.name
    if isinstance(node, A.Field):
        base = _field_path(node.obj)
        if base is None:
            return None
        return f"{base}.{node.name}"
    return None


def _propagate(node: A.Expr, adj: A.Expr, acc: dict[str, list[A.Expr]]) -> None:
    """Send the adjoint `adj` through `node`, depositing contributions into
    `acc[name]` for each parameter Name or Field path encountered."""
    if isinstance(node, (A.IntLit, A.FloatLit, A.BoolLit, A.StrLit, A.CharLit)):
        return
    if isinstance(node, A.Name):
        if node.name in acc:
            acc[node.name].append(adj)
        elif _has_related_target(node.name, acc):
            _ad_warn(
                node,
                f"name {node.name!r} is related to differentiable field "
                "leaves but no exact accumulator exists (reverse-mode)",
            )
        return
    if isinstance(node, A.Field):
        path = _field_path(node)
        if path is None:
            _ad_warn(
                node,
                "field expression has no static differentiable path "
                "(reverse-mode)",
            )
            return
        if path in acc:
            acc[path].append(adj)
        elif _has_related_target(path, acc):
            _ad_warn(
                node,
                f"field path {path!r} is related to a differentiable "
                "target but no exact leaf accumulator exists (reverse-mode)",
            )
        return
    if isinstance(node, A.Unary):
        if node.op == "-":
            neg = A.Unary(span=node.span, op="-", operand=adj)
            _propagate(node.operand, neg, acc)
        else:
            # Audit 28.8 cycle 2 C2-3: pre-fix, ANY Unary op other than
            # `-` (i.e. `!`, `~`, `&`, `*`-deref) silently returned with
            # no contribution to the gradient. Forward-mode (autodiff._diff)
            # already warns via its catch-all branch; reverse-mode was
            # asymmetric — silent. Now both modes diagnose loudly.
            _ad_warn(
                node,
                f"unary op {node.op!r} has no defined local derivative "
                f"(reverse-mode)",
            )
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
        else:
            # Audit 28.8 cycle 2 C2-3: pre-fix, Binary ops outside
            # `{+, -, *, /}` (e.g. `%`, comparisons, bitwise) silently
            # returned a zero contribution. Symmetrize with forward-mode
            # which DOES warn via its catch-all branch (autodiff.py:591).
            _ad_warn(
                node,
                f"binary op {op!r} has no defined local derivative "
                f"(reverse-mode)",
            )
        return
    if isinstance(node, A.Block):
        if node.final_expr is not None:
            _propagate(node.final_expr, adj, acc)
        return
    if isinstance(node, A.Call):
        # Stage 36 Inc 12 — close Inc 11 type-design B2 MEDIUM deferral.
        # Integer-valued boolean Logic ops are AD-pure (so let-inlining
        # doesn't trap) but have no chain rule; pre-fix they silently
        # produced a zero adjoint contribution. Mirror the forward-mode
        # guard in autodiff._diff_call_chain_rule: fail loud before any
        # chain-rule arm runs.
        if (isinstance(node.callee, A.Name)
                and node.callee.name in AD_INTEGER_VALUED_LOGIC):
            _raise_integer_logic_in_ad(node.callee.name, "reverse")
        # __powi(x, n) where n is a literal int: adj_x = adj * n * x^(n-1).
        if (isinstance(node.callee, A.Name) and node.callee.name == "__powi"
                and len(node.args) == 2 and isinstance(node.args[1], A.IntLit)):
            x = node.args[0]
            n_val = node.args[1].value
            if n_val <= 0 or n_val > 16:
                # Both edges return 1.0 from stdlib (constant), so the
                # derivative is 0. Previously we capped n_val to 16,
                # producing a wrong gradient for n > 16 — the function
                # returns the constant 1, but AD reported `n * x^15`.
                return
            n_lit = A.FloatLit(span=node.span, value=float(n_val))
            n_minus_one = A.IntLit(span=node.span, value=n_val - 1)
            x_pow = A.Call(span=node.span,
                           callee=A.Name(span=node.span, name="__powi"),
                           args=[x, n_minus_one])
            new_adj = A.Binary(
                span=node.span, op="*",
                left=A.Binary(span=node.span, op="*", left=adj, right=n_lit),
                right=x_pow,
            )
            _propagate(x, new_adj, acc)
            return
        if (isinstance(node.callee, A.Name) and node.callee.name == "__bce"
                and len(node.args) == 2):
            p = node.args[0]
            y = node.args[1]

            def f(v: float) -> A.FloatLit:
                return A.FloatLit(span=node.span, value=v)

            def binary(op: str, a: A.Expr, b: A.Expr) -> A.Binary:
                return A.Binary(span=node.span, op=op, left=a, right=b)

            def calln(fn: str, args: list[A.Expr]) -> A.Call:
                return A.Call(span=node.span,
                              callee=A.Name(span=node.span, name=fn),
                              args=args)

            p_safe = calln("__clamp", [copy.deepcopy(p), f(0.000001), f(0.999999)])
            denom = binary("*", copy.deepcopy(p_safe),
                           binary("-", f(1.0), copy.deepcopy(p_safe)))
            raw_dp = binary("/", binary("-", copy.deepcopy(p_safe),
                                        copy.deepcopy(y)), denom)
            cond_lo = binary("<", copy.deepcopy(p), f(0.000001))
            cond_hi = binary(">", copy.deepcopy(p), f(0.999999))
            gated_hi = A.If(
                span=node.span,
                cond=cond_hi,
                then=A.Block(span=node.span, stmts=[], final_expr=f(0.0)),
                else_=A.Block(span=node.span, stmts=[], final_expr=raw_dp),
            )
            deriv_p = A.If(
                span=node.span,
                cond=cond_lo,
                then=A.Block(span=node.span, stmts=[], final_expr=f(0.0)),
                else_=A.Block(span=node.span, stmts=[], final_expr=gated_hi),
            )
            log_one_minus = calln(
                "__log_stable",
                [binary("-", f(1.0), copy.deepcopy(p_safe))],
            )
            log_p = calln("__log_stable", [copy.deepcopy(p_safe)])
            deriv_y = binary("-", log_one_minus, log_p)
            _propagate(p, binary("*", adj, deriv_p), acc)
            _propagate(y, binary("*", copy.deepcopy(adj), deriv_y), acc)
            return
        # Chain rule for known transcendentals: propagate adj * f'(u) into u.
        if (isinstance(node.callee, A.Name) and len(node.args) == 1):
            name = node.callee.name
            u = node.args[0]

            def call1(fn: str, arg: A.Expr) -> A.Expr:
                return A.Call(span=node.span,
                              callee=A.Name(span=node.span, name=fn),
                              args=[arg])

            def flit(v: float, suffix: str | None = None) -> A.FloatLit:
                return A.FloatLit(span=node.span, value=v, type_suffix=suffix)

            if name == "__log_stable":
                cond = A.Binary(span=node.span, op="<=",
                                left=copy.deepcopy(u), right=flit(0.0))
                recip = A.Binary(span=node.span, op="/",
                                 left=flit(1.0), right=copy.deepcopy(u))
                deriv = A.If(
                    span=node.span,
                    cond=cond,
                    then=A.Block(span=node.span, stmts=[], final_expr=flit(0.0)),
                    else_=A.Block(span=node.span, stmts=[], final_expr=recip),
                )
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=deriv)
                _propagate(u, new_adj, acc)
                return
            if name == "__exp_f64":
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=call1("__exp_f64", u))
                _propagate(u, new_adj, acc)
                return
            if name == "__log_f64":
                recip = A.Binary(span=node.span, op="/",
                                 left=flit(1.0, "f64"), right=u)
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=recip)
                _propagate(u, new_adj, acc)
                return
            if name == "__sin_f64":
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=call1("__cos_f64", u))
                _propagate(u, new_adj, acc)
                return
            if name == "__cos_f64":
                neg_sin = A.Unary(span=node.span, op="-",
                                  operand=call1("__sin_f64", u))
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=neg_sin)
                _propagate(u, new_adj, acc)
                return
            if name == "__sqrt_f64":
                sqrt_u = call1("__sqrt_f64", u)
                denom = A.Binary(span=node.span, op="*",
                                 left=flit(2.0, "f64"), right=sqrt_u)
                recip = A.Binary(span=node.span, op="/",
                                 left=flit(1.0, "f64"), right=denom)
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=recip)
                _propagate(u, new_adj, acc)
                return
            if name == "__relu_f64":
                cond = A.Binary(span=node.span, op=">", left=u,
                                right=flit(0.0, "f64"))
                gated = A.If(span=node.span, cond=cond,
                             then=A.Block(span=node.span, stmts=[],
                                          final_expr=flit(1.0, "f64")),
                             else_=A.Block(span=node.span, stmts=[],
                                           final_expr=flit(0.0, "f64")))
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=gated)
                _propagate(u, new_adj, acc)
                return
            if name == "__sigmoid_f64":
                s1 = call1("__sigmoid_f64", copy.deepcopy(u))
                s2 = call1("__sigmoid_f64", copy.deepcopy(u))
                one_minus = A.Binary(span=node.span, op="-",
                                     left=flit(1.0, "f64"), right=s2)
                deriv = A.Binary(span=node.span, op="*", left=s1,
                                 right=one_minus)
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=deriv)
                _propagate(u, new_adj, acc)
                return
            if name == "__abs_f64":
                cond_pos = A.Binary(span=node.span, op=">",
                                    left=copy.deepcopy(u),
                                    right=flit(0.0, "f64"))
                cond_neg = A.Binary(span=node.span, op="<",
                                    left=copy.deepcopy(u),
                                    right=flit(0.0, "f64"))
                inner_else = A.If(span=node.span, cond=cond_neg,
                                  then=A.Block(span=node.span, stmts=[],
                                               final_expr=flit(-1.0, "f64")),
                                  else_=A.Block(span=node.span, stmts=[],
                                                final_expr=flit(0.0, "f64")))
                deriv = A.If(span=node.span, cond=cond_pos,
                             then=A.Block(span=node.span, stmts=[],
                                          final_expr=flit(1.0, "f64")),
                             else_=A.Block(span=node.span, stmts=[],
                                           final_expr=inner_else))
                new_adj = A.Binary(span=node.span, op="*", left=adj,
                                   right=deriv)
                _propagate(u, new_adj, acc)
                return

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
                # d(tanh(u))/dx = (1 - tanh(u)^2) * du. Two independent
                # tanh(u) calls — each with its own deepcopy of u — so
                # the square doesn't share AST structure between halves.
                t1 = call1("__tanh", copy.deepcopy(u))
                t2 = call1("__tanh", copy.deepcopy(u))
                t_sq = A.Binary(span=node.span, op="*", left=t1, right=t2)
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
            if name == "__gelu":
                # Tanh-approx GELU derivative:
                # 0.5*(1+tanh(inner)) + 0.5*u*(1-tanh(inner)^2)*inner'
                x2 = A.Binary(span=node.span, op="*",
                              left=copy.deepcopy(u), right=copy.deepcopy(u))
                x3 = A.Binary(span=node.span, op="*", left=copy.deepcopy(x2),
                              right=copy.deepcopy(u))
                inner_arg = A.Binary(
                    span=node.span,
                    op="+",
                    left=copy.deepcopy(u),
                    right=A.Binary(span=node.span, op="*",
                                   left=A.FloatLit(span=node.span, value=0.044715),
                                   right=x3),
                )
                inner = A.Binary(
                    span=node.span,
                    op="*",
                    left=A.FloatLit(span=node.span, value=0.7978846),
                    right=inner_arg,
                )
                t1 = call1("__tanh", copy.deepcopy(inner))
                t2 = call1("__tanh", copy.deepcopy(inner))
                first = A.Binary(
                    span=node.span,
                    op="*",
                    left=A.FloatLit(span=node.span, value=0.5),
                    right=A.Binary(span=node.span, op="+",
                                   left=A.FloatLit(span=node.span, value=1.0),
                                   right=t1),
                )
                one_minus_t2 = A.Binary(
                    span=node.span,
                    op="-",
                    left=A.FloatLit(span=node.span, value=1.0),
                    right=A.Binary(span=node.span, op="*", left=t2,
                                   right=call1("__tanh", copy.deepcopy(inner))),
                )
                inner_prime = A.Binary(
                    span=node.span,
                    op="*",
                    left=A.FloatLit(span=node.span, value=0.7978846),
                    right=A.Binary(
                        span=node.span,
                        op="+",
                        left=A.FloatLit(span=node.span, value=1.0),
                        right=A.Binary(
                            span=node.span,
                            op="*",
                            left=A.FloatLit(span=node.span, value=0.134145),
                            right=x2,
                        ),
                    ),
                )
                second = A.Binary(
                    span=node.span,
                    op="*",
                    left=A.Binary(span=node.span, op="*",
                                  left=A.FloatLit(span=node.span, value=0.5),
                                  right=copy.deepcopy(u)),
                    right=A.Binary(span=node.span, op="*", left=one_minus_t2,
                                   right=inner_prime),
                )
                deriv = A.Binary(span=node.span, op="+", left=first, right=second)
                new_adj = A.Binary(span=node.span, op="*", left=adj, right=deriv)
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
        # Stage 36 Increment 6: provenance + fuzzy-logic chain rules.
        # prove/unwrap_logic/attach/detach are identity functions at
        # the IR level (Logic<T> and D<T> wrappers have zero runtime
        # representation in Phase-0). For AD purposes, the chain rule
        # is therefore identity: adj of the result equals adj of the
        # first (value) argument; the provenance tag (second arg of
        # prove) is non-differentiable and doesn't get an update.
        #
        # Stage 36 Inc 9 catch-up — type-design B3 fix (reverse-mode
        # twin of the forward-mode guard in autodiff.py): require the
        # source tag of `prove` to be an integer literal in
        # differentiated code so AD can statically see the second arg
        # is non-aliased with the diff target.
        if isinstance(node.callee, A.Name) and node.callee.name in (
                "prove", "unwrap_logic", "attach", "detach"):
            if (node.callee.name == "prove"
                    and len(node.args) == 2
                    and not isinstance(node.args[1], A.IntLit)):
                raise NotImplementedError(
                    "autodiff (reverse): prove(value, source): source "
                    "must be an integer literal in differentiated code "
                    f"(got {type(node.args[1]).__name__}); use "
                    "register_derivation for dynamic source tags so AD "
                    "can statically see the tag is non-differentiable"
                )
            # v2.x re-audit R2b (FE 5-clean-gate MEDIUM): these four
            # builtins are identity w.r.t. their value argument
            # (args[0]). A call with NO arguments is malformed and
            # must fail loudly — the prior `if node.args:` silently
            # propagated no adjoint at all for a 0-arg call. Arity is
            # normally enforced by typecheck before AD; this is the
            # reverse-mode defense-in-depth guard, parity with the
            # forward-mode arity checks in autodiff.py.
            if not node.args:
                raise NotImplementedError(
                    f"autodiff (reverse): {node.callee.name}() called "
                    f"with no arguments — expected at least the value "
                    f"argument to propagate the adjoint into")
            _propagate(node.args[0], adj, acc)
            return
        # fuzzy_and(a, b) = a * b: ∂/∂a = b, ∂/∂b = a.
        if (isinstance(node.callee, A.Name)
                and node.callee.name == "fuzzy_and"
                and len(node.args) == 2):
            a_arg, b_arg = node.args
            adj_a = A.Binary(span=node.span, op="*",
                             left=adj, right=copy.deepcopy(b_arg))
            adj_b = A.Binary(span=node.span, op="*",
                             left=copy.deepcopy(adj),
                             right=copy.deepcopy(a_arg))
            _propagate(a_arg, adj_a, acc)
            _propagate(b_arg, adj_b, acc)
            return
        # fuzzy_or(a, b) = a + b - a*b:
        #   ∂/∂a = 1 - b, ∂/∂b = 1 - a.
        if (isinstance(node.callee, A.Name)
                and node.callee.name == "fuzzy_or"
                and len(node.args) == 2):
            a_arg, b_arg = node.args
            one_minus_b = A.Binary(
                span=node.span, op="-",
                left=A.FloatLit(span=node.span, value=1.0),
                right=copy.deepcopy(b_arg))
            one_minus_a = A.Binary(
                span=node.span, op="-",
                left=A.FloatLit(span=node.span, value=1.0),
                right=copy.deepcopy(a_arg))
            adj_a = A.Binary(span=node.span, op="*",
                             left=adj, right=one_minus_b)
            adj_b = A.Binary(span=node.span, op="*",
                             left=copy.deepcopy(adj), right=one_minus_a)
            _propagate(a_arg, adj_a, acc)
            _propagate(b_arg, adj_b, acc)
            return
        # Stage 36 Increment 8 — fuzzy_xor + fuzzy_implies reverse-mode.
        # fuzzy_xor(a, b) = a + b - 2*a*b:
        #   ∂/∂a = 1 - 2*b, ∂/∂b = 1 - 2*a.
        if (isinstance(node.callee, A.Name)
                and node.callee.name == "fuzzy_xor"
                and len(node.args) == 2):
            a_arg, b_arg = node.args
            two_b = A.Binary(span=node.span, op="*",
                             left=A.FloatLit(span=node.span, value=2.0),
                             right=copy.deepcopy(b_arg))
            two_a = A.Binary(span=node.span, op="*",
                             left=A.FloatLit(span=node.span, value=2.0),
                             right=copy.deepcopy(a_arg))
            coeff_a = A.Binary(span=node.span, op="-",
                               left=A.FloatLit(span=node.span, value=1.0),
                               right=two_b)
            coeff_b = A.Binary(span=node.span, op="-",
                               left=A.FloatLit(span=node.span, value=1.0),
                               right=two_a)
            adj_a = A.Binary(span=node.span, op="*",
                             left=adj, right=coeff_a)
            adj_b = A.Binary(span=node.span, op="*",
                             left=copy.deepcopy(adj), right=coeff_b)
            _propagate(a_arg, adj_a, acc)
            _propagate(b_arg, adj_b, acc)
            return
        # fuzzy_implies(a, b) = 1 - a + a*b:
        #   ∂/∂a = -1 + b, ∂/∂b = a.
        if (isinstance(node.callee, A.Name)
                and node.callee.name == "fuzzy_implies"
                and len(node.args) == 2):
            a_arg, b_arg = node.args
            coeff_a = A.Binary(span=node.span, op="-",
                               left=copy.deepcopy(b_arg),
                               right=A.FloatLit(span=node.span, value=1.0))
            adj_a = A.Binary(span=node.span, op="*",
                             left=adj, right=coeff_a)
            adj_b = A.Binary(span=node.span, op="*",
                             left=copy.deepcopy(adj),
                             right=copy.deepcopy(a_arg))
            _propagate(a_arg, adj_a, acc)
            _propagate(b_arg, adj_b, acc)
            return
        # fuzzy_not(a) = 1 - a: ∂/∂a = -1.
        if (isinstance(node.callee, A.Name)
                and node.callee.name == "fuzzy_not"
                and len(node.args) == 1):
            a_arg = node.args[0]
            adj_a = A.Unary(span=node.span, op="-", operand=adj)
            _propagate(a_arg, adj_a, acc)
            return
        # Stage 38 post-Inc-3 silent-failure F2 fix (MEDIUM): frame
        # identity wrappers — adjoint flows through unchanged on the
        # single arg. Mirrors the forward arm in autodiff.py.
        if (isinstance(node.callee, A.Name)
                and node.callee.name in _IDENTITY_AD_CHAIN_RULE_NAMES
                and len(node.args) == 1):
            _propagate(node.args[0], adj, acc)
            return
        # Stage 54 Inc 1: __min/__max chain rule.
        # Stage 54 gate-3 code-review HIGH-1 fix: mirror the
        # corrected forward-mode docstring (autodiff.py
        # _stage54_min_max_chain_rule). Subgradient at equality
        # is asymmetric AND POINTS IN OPPOSITE DIRECTIONS for
        # min vs max: __min attributes equality-case gradient
        # to the FIRST arg (`a <= b`); __max attributes it to
        # the SECOND arg (`b >= a`). Both are valid 1-sided
        # subgradient picks; the operator-pair asymmetry keeps
        # each indicator's LHS bound to that arg for forward/
        # reverse mirroring. Prior comment "picks 0 per standard
        # convention" was the pre-gate-1 misleading text.
        # __min: adj_a if a<=b else 0;  adj_b if b<a else 0
        # __max: adj_a if a>b else 0;   adj_b if b>=a else 0
        if (isinstance(node.callee, A.Name)
                and node.callee.name
                    in ("__min", "__min_f64", "__max", "__max_f64")
                and len(node.args) == 2):
            name = node.callee.name
            a_arg, b_arg = node.args
            suffix = "f64" if name.endswith("_f64") else None
            zero_lit = A.FloatLit(
                span=node.span, value=0.0, type_suffix=suffix)
            # Stage 54 gate-5 HIGH-1 forward/reverse parity:
            # warn when BOTH args depend on any tracked param
            # (user is differentiating through a kink at a==b).
            # Mirror of the forward fix in
            # `_stage54_min_max_chain_rule` (gate-4 MEDIUM-3).
            for tracked in acc.keys():
                if (_name_appears_in(a_arg, tracked)
                        and _name_appears_in(b_arg, tracked)):
                    _ad_warn(
                        node,
                        f"{name} with both args depending on "
                        f"'{tracked}' — subgradient is defined "
                        f"via lexically-first convention but "
                        f"gradients are discontinuous at a == b "
                        f"(reverse mode). Confirm this is the "
                        f"intended behavior.",
                    )
            if name in ("__min", "__min_f64"):
                op_a, op_b = "<=", "<"
            else:
                op_a, op_b = ">", ">="
            cond_a = A.Binary(span=node.span, op=op_a,
                              left=copy.deepcopy(a_arg),
                              right=copy.deepcopy(b_arg))
            cond_b = A.Binary(span=node.span, op=op_b,
                              left=copy.deepcopy(b_arg),
                              right=copy.deepcopy(a_arg))
            adj_a = A.If(
                span=node.span, cond=cond_a,
                then=A.Block(span=node.span, stmts=[],
                             final_expr=copy.deepcopy(adj)),
                else_=A.Block(span=node.span, stmts=[],
                              final_expr=copy.deepcopy(zero_lit)))
            adj_b = A.If(
                span=node.span, cond=cond_b,
                then=A.Block(span=node.span, stmts=[],
                             final_expr=copy.deepcopy(adj)),
                else_=A.Block(span=node.span, stmts=[],
                              final_expr=copy.deepcopy(zero_lit)))
            _propagate(a_arg, adj_a, acc)
            _propagate(b_arg, adj_b, acc)
            return
        # Stage 54 Inc 1: __clamp chain rule.
        # adj_x * indicator(lo<=x AND x<=hi); lo/hi are non-diff.
        if (isinstance(node.callee, A.Name)
                and node.callee.name in ("__clamp", "__clamp_f64")
                and len(node.args) == 3):
            name = node.callee.name
            x_arg, lo_arg, hi_arg = node.args
            suffix = "f64" if name.endswith("_f64") else None
            zero_lit = A.FloatLit(
                span=node.span, value=0.0, type_suffix=suffix)
            # Stage 54 gate-3 MEDIUM-3 forward/reverse parity:
            # warn when any tracked param appears in lo/hi
            # (its dlo/dhi contributions are silently dropped).
            # Mirror of the forward fix in
            # `_stage54_clamp_chain_rule`. acc is the dict of
            # params-with-adjoint-accumulators, so its keys
            # are exactly the tracked params.
            for tracked in acc.keys():
                if (_name_appears_in(lo_arg, tracked)
                        or _name_appears_in(hi_arg, tracked)):
                    _ad_warn(
                        node,
                        f"__clamp dlo/dhi w.r.t. "
                        f"'{tracked}' silently dropped — "
                        f"gradient is incomplete (reverse "
                        f"mode). Treat lo/hi as constants "
                        f"or detach them from the "
                        f"differentiation graph.",
                    )
            lo_ok = A.Binary(span=node.span, op="<=",
                             left=copy.deepcopy(lo_arg),
                             right=copy.deepcopy(x_arg))
            hi_ok = A.Binary(span=node.span, op="<=",
                             left=copy.deepcopy(x_arg),
                             right=copy.deepcopy(hi_arg))
            both = A.Binary(span=node.span, op="&&",
                            left=lo_ok, right=hi_ok)
            adj_x = A.If(
                span=node.span, cond=both,
                then=A.Block(span=node.span, stmts=[],
                             final_expr=copy.deepcopy(adj)),
                else_=A.Block(span=node.span, stmts=[],
                              final_expr=copy.deepcopy(zero_lit)))
            _propagate(x_arg, adj_x, acc)
            # lo and hi are non-differentiable constants — no
            # contribution.
            return
        # Stage 54 Inc 1: __sign + _i32 variants of min/max/clamp
        # return 0 derivative (distributional / integer-valued).
        if (isinstance(node.callee, A.Name)
                and node.callee.name in (
                    "__sign", "__sign_f64",
                    "__min_i32", "__max_i32", "__clamp_i32",
                )):
            # Derivative is 0 everywhere — emit no contribution
            # to any operand.
            return
        # Audit 28.8 B5: opaque user call — was silently a zero
        # contribution. Reverse-mode now fails closed instead of compiling
        # a zero-gradient surrogate.
        callee = getattr(node.callee, "name", "<?>")
        raise NotImplementedError(
            f"reverse-mode AD does not support opaque call {callee!r}; "
            "add a chain rule or inline a differentiable helper"
        )
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
    if isinstance(node, A.Match):
        # Like If: the runtime selects exactly one arm, so the gradient is a
        # Match over the same scrutinee/patterns whose bodies are the
        # per-arm gradient contributions. Discrete choice → scrutinee has
        # zero local derivative. Pattern-bound names (PatBind) are local
        # to the arm; if they shadow a param, that param's contribution
        # inside the arm is dropped (the shadow shadows). A PatBind alias of a
        # differentiable scrutinee is not yet rewritten into the arm body, so
        # fail closed instead of silently returning a zero gradient.
        if _expr_depends_on_param(node.scrutinee, set(acc.keys())):
            for arm in node.arms:
                if _pattern_binds_any(arm.pattern):
                    raise NotImplementedError(
                        "reverse-mode AD does not support match pattern "
                        "bindings that alias a differentiable scrutinee; "
                        "rewrite the arm to use the original value or add "
                        "alias propagation"
                    )
        arm_accs: list[dict[str, list[A.Expr]]] = [
            {p: [] for p in acc} for _ in node.arms
        ]
        for i, arm in enumerate(node.arms):
            body = arm.body
            adj_for_arm = adj if i == 0 else copy.deepcopy(adj)
            shadowed = _pattern_shadowed_names(arm.pattern, set(acc.keys()))
            visible_acc = {p: arm_accs[i][p] for p in acc if p not in shadowed}
            if isinstance(body, A.Block):
                if body.final_expr is not None:
                    _propagate(body.final_expr, adj_for_arm, visible_acc)
            else:
                _propagate(body, adj_for_arm, visible_acc)
        for p in acc:
            any_contrib = any(arm_accs[i][p] for i in range(len(node.arms)))
            if not any_contrib:
                continue
            zero = A.FloatLit(span=node.span, value=0.0)
            new_arms: list[A.MatchArm] = []
            for i, arm in enumerate(node.arms):
                body_grad = (_sum_exprs(arm_accs[i][p], node.span)
                             if arm_accs[i][p] else zero)
                new_arms.append(A.MatchArm(
                    span=arm.span,
                    pattern=copy.deepcopy(arm.pattern),
                    guard=copy.deepcopy(arm.guard) if arm.guard is not None else None,
                    body=body_grad,
                ))
            scrut_copy = copy.deepcopy(node.scrutinee)
            wrapped = A.Match(span=node.span, scrutinee=scrut_copy,
                              arms=new_arms)
            acc[p].append(wrapped)
        return
    # Audit 28.8 B5: Cast — propagate through numeric casts.
    # Audit 28.8 cycle 2 B:C9: shared NUMERIC_FOR_AD set covers
    # bool/char/fp8/mxfp4/nvfp4 too (typecheck accepts them as
    # numeric scalars; AD-pass shouldn't false-warn).
    if isinstance(node, A.Cast):
        tgt = node.target_ty
        if isinstance(tgt, A.TyName) and tgt.name in NUMERIC_FOR_AD:
            _propagate(node.value, adj, acc)
            return
        _ad_warn(node, f"cast to non-numeric target "
                       f"{type(tgt).__name__}")
        return
    # Audit 28.8 B5: UnsafeBlock — propagate adjoint through body.
    if isinstance(node, A.UnsafeBlock):
        body = node.body
        if isinstance(body, A.Block):
            if body.final_expr is not None:
                _propagate(body.final_expr, adj, acc)
        else:
            _propagate(body, adj, acc)
        return
    # Audit 28.8 B5: Quote/Splice/Modify — non-differentiable, warn.
    if isinstance(node, (A.Quote, A.Splice, A.Modify)):
        _ad_warn(node, f"{type(node).__name__} is not differentiable")
        return
    # Stage 54 gate-3 silent-failure CRITICAL-1: explicit fail-loud
    # arms for AST kinds that Inc 3a's loop-body descent now feeds
    # into _propagate. Pre-Inc-3a these were unreachable — the
    # helper-call inside a loop body stayed opaque and hit the
    # Call arm's loud NotImplementedError at line 768. Post-Inc-3a
    # the helper is inlined and the For/While/Loop node flows
    # straight into _propagate. Without explicit arms it falls to
    # the warn-and-zero catchall below — and `_ad_warn` is by
    # default a soft trap 85001 that gets suppressed unless
    # `-Wad=error` is set, leaving the user with a silent-zero
    # gradient on any loop-containing reverse-mode AD. Raise loudly
    # instead, mirroring the Stage 35 opaque-call discipline and
    # the forward-mode `_diff` arms.
    if isinstance(node, (A.For, A.While, A.Loop)):
        kind = type(node).__name__
        raise NotImplementedError(
            f"reverse-mode AD does not differentiate through {kind} "
            f"bodies; unroll the loop or move the gradient-bearing "
            f"computation outside"
        )
    if isinstance(node, (A.Assign, A.Return, A.Break, A.Continue)):
        kind = type(node).__name__
        raise NotImplementedError(
            f"reverse-mode AD does not differentiate through {kind} "
            f"statements; these are control flow, not differentiable "
            f"expressions"
        )
    if isinstance(node, A.Range):
        raise NotImplementedError(
            "reverse-mode AD does not differentiate through Range "
            "expressions; ranges are iterators, not numeric values"
        )
    # Audit 28.8 B5: Any other unhandled node — warn loudly. Pre-fix
    # this returned silently and the gradient was 0 with no diagnostic.
    _ad_warn(node, "unhandled expression kind in reverse-mode AD")


def _pattern_shadowed_names(pat: A.Pattern, candidates: set[str]) -> set[str]:
    """Return the subset of `candidates` shadowed by names bound in `pat`."""
    if isinstance(pat, A.PatBind):
        prefix = f"{pat.name}."
        return {c for c in candidates if c == pat.name or c.startswith(prefix)}
    if isinstance(pat, A.PatOr):
        out: set[str] = set()
        for alt in pat.alts:
            out |= _pattern_shadowed_names(alt, candidates)
        return out
    if isinstance(pat, A.PatTuple):
        out = set()
        for sub in pat.elems:
            out |= _pattern_shadowed_names(sub, candidates)
        return out
    if isinstance(pat, A.PatVariant):
        out = set()
        for sub in (pat.sub_patterns or []):
            out |= _pattern_shadowed_names(sub, candidates)
        return out
    return set()


def _expr_depends_on_param(expr: A.Expr, candidates: set[str]) -> bool:
    seen: set[int] = set()

    def related(path: str) -> bool:
        return any(
            path == c or path.startswith(c + ".") or c.startswith(path + ".")
            for c in candidates
        )

    def visit(value: object) -> bool:
        if value is None or isinstance(value, (str, int, float, bool)):
            return False
        if isinstance(value, (list, tuple)):
            return any(visit(v) for v in value)

        oid = id(value)
        if oid in seen:
            return False
        seen.add(oid)

        if isinstance(value, A.Expr):
            path = _field_path(value)
            if path is not None and related(path):
                return True

        if dataclasses.is_dataclass(value):
            for field in dataclasses.fields(value):
                if visit(getattr(value, field.name)):
                    return True
        return False

    return visit(expr)


def _pattern_binds_any(pat: A.Pattern) -> bool:
    if isinstance(pat, A.PatBind):
        return True
    if isinstance(pat, A.PatOr):
        return any(_pattern_binds_any(alt) for alt in pat.alts)
    if isinstance(pat, A.PatTuple):
        return any(_pattern_binds_any(sub) for sub in pat.elems)
    if isinstance(pat, A.PatVariant):
        return any(_pattern_binds_any(sub) for sub in (pat.sub_patterns or []))
    return False


def _has_related_target(path: str, acc: dict[str, list[A.Expr]]) -> bool:
    """True when `path` and any AD target sit in the same field tree.

    This keeps struct-parameter gradients loud before full pytree codegen is
    wired. `m.w` is a harmless coefficient when differentiating w.r.t. `x`,
    but it must not silently become zero when the requested target is `m`.
    """
    prefix = f"{path}."
    for target in acc:
        target_prefix = f"{target}."
        if target.startswith(prefix) or path.startswith(target_prefix):
            return True
    return False


def _sum_exprs(exprs: list[A.Expr], span: A.Span) -> A.Expr:
    if not exprs:
        return A.FloatLit(span=span, value=0.0)
    if len(exprs) == 1:
        return exprs[0]
    out = exprs[0]
    for e in exprs[1:]:
        out = A.Binary(span=span, op="+", left=out, right=e)
    return out
