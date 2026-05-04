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

import copy as _copy
from typing import Optional

from . import ast_nodes as A
from .ast_hash import structural_hash


# Module-level memoization for `differentiate()`. Keyed on:
#   (structural_hash(expr), var, fn_table_signature)
# A returned value is a deepcopy so callers can mutate freely without
# corrupting the cache. Cleared by `clear_diff_cache()` if needed.
_DIFF_CACHE: dict[tuple[str, str, str], A.Expr] = {}
_DIFF_CACHE_HITS = [0]
_DIFF_CACHE_MISSES = [0]


def clear_diff_cache() -> None:
    """Reset the differentiate-memo cache (for tests)."""
    _DIFF_CACHE.clear()
    _DIFF_CACHE_HITS[0] = 0
    _DIFF_CACHE_MISSES[0] = 0


def diff_cache_stats() -> tuple[int, int]:
    """Return (hits, misses) since last `clear_diff_cache()`."""
    return (_DIFF_CACHE_HITS[0], _DIFF_CACHE_MISSES[0])


def _fn_table_sig(fn_table: dict[str, "A.FnDecl"] | None) -> str:
    if not fn_table:
        return ""
    # The set of *names* available is the relevant key — body changes
    # invalidate the expr hash already, but adding a new fn could change
    # inlining behavior.
    return "|".join(sorted(fn_table.keys()))


def differentiate(expr: A.Expr, var: str,
                  fn_table: dict[str, "A.FnDecl"] | None = None) -> A.Expr:
    """Return the AST of d(expr)/d(var), simplified.

    Memoized by structural hash of `expr` + var + fn_table signature.
    Returns a deepcopy of the cached deriv so callers can mutate.

    Optionally accepts a `fn_table` mapping function names to FnDecls. When
    provided, calls to user-defined @pure functions in the expression are
    inlined (their bodies substituted for the call) before differentiation.
    This makes grad work across function boundaries — `grad(f)` where f's
    body calls a helper `g(x)` propagates the gradient through g.

    If `expr` is a Block, the block's let-bindings are inlined first so
    that subsequent uses of the bound names refer to their definitions.
    """
    try:
        key = (structural_hash(expr), var, _fn_table_sig(fn_table))
    except Exception:
        key = None  # hash failure → bypass cache
    if key is not None and key in _DIFF_CACHE:
        _DIFF_CACHE_HITS[0] += 1
        return _copy.deepcopy(_DIFF_CACHE[key])

    if fn_table:
        expr = _inline_user_calls(expr, fn_table)
    inlined = _inline_lets(expr, {})
    deriv = _diff(inlined, var)
    out = _simplify(deriv)

    if key is not None:
        _DIFF_CACHE_MISSES[0] += 1
        _DIFF_CACHE[key] = _copy.deepcopy(out)
    return out


def _inline_user_calls(expr: A.Expr, fn_table: dict[str, "A.FnDecl"],
                        depth: int = 0, max_depth: int = 4,
                        visiting: frozenset[str] | None = None) -> A.Expr:
    """Walk `expr` and replace each Call(Name(f), args) where f is a known
    @pure function in `fn_table` with a deepcopy of f's body, with each
    parameter substituted by the corresponding argument expression.

    Skips:
      - Transcendental builtins (`__exp`, `__log`, etc.) — they have
        analytic AD chain rules already wired into _diff.
      - Functions currently in `visiting` (mutual / direct recursion
        guard — prevents exponential AST expansion when inlining cycles
        like a→b→a).
      - depth >= max_depth (safety net).
      - Functions not in fn_table (treated as opaque external).
    """
    import copy as _copy

    # Functions with analytic AD chain rules in _diff_call_chain_rule /
    # autodiff_reverse._propagate. Inlining these would force the AD
    # engine to differentiate through their (potentially conditional)
    # bodies instead of using the closed-form derivative — producing
    # silently-wrong gradients when the body uses if/while.
    TRANSCENDENTALS = {"__exp", "__log", "__sin", "__cos", "__sqrt",
                       "__relu", "__sigmoid", "__tanh", "__softplus",
                       "__silu", "__abs", "__gelu", "__powi"}
    visiting = visiting or frozenset()

    def go(e: A.Expr) -> A.Expr:
        if isinstance(e, A.Call):
            new_callee = go(e.callee)
            new_args = [go(a) for a in e.args]
            if (isinstance(new_callee, A.Name)
                    and new_callee.name in fn_table
                    and new_callee.name not in TRANSCENDENTALS
                    and new_callee.name not in visiting
                    and depth < max_depth):
                fn = fn_table[new_callee.name]
                # Only inline @pure functions — others may have effects
                # whose differentiation is unsound.
                if "pure" not in fn.attrs:
                    return A.Call(span=e.span, callee=new_callee, args=new_args)
                if len(fn.params) != len(new_args):
                    return A.Call(span=e.span, callee=new_callee, args=new_args)
                # Build substitution map: param name -> arg expression
                substitutions = {p.name: a for p, a in zip(fn.params, new_args)}
                # Deepcopy the body so we don't share references with the
                # original function (downstream passes mutate in-place).
                body_copy = _copy.deepcopy(fn.body)
                substituted = _substitute_names(body_copy, substitutions)
                # Recursively inline within the substituted body. Add this
                # function to the visiting set so any recursive (direct or
                # mutual) call back to it is treated as opaque.
                return _inline_user_calls(substituted, fn_table, depth + 1,
                                           max_depth,
                                           visiting | {new_callee.name})
            return A.Call(span=e.span, callee=new_callee, args=new_args)
        if isinstance(e, A.Binary):
            return A.Binary(span=e.span, op=e.op, left=go(e.left), right=go(e.right))
        if isinstance(e, A.Unary):
            return A.Unary(span=e.span, op=e.op, operand=go(e.operand))
        if isinstance(e, A.Block):
            new_stmts = []
            for s in e.stmts:
                if isinstance(s, A.Let) and s.value is not None:
                    new_stmts.append(A.Let(span=s.span, name=s.name,
                                            ty=s.ty, value=go(s.value),
                                            is_mut=s.is_mut))
                elif isinstance(s, A.ConstStmt):
                    new_stmts.append(A.ConstStmt(span=s.span, name=s.name,
                                                  ty=s.ty, value=go(s.value)))
                else:
                    new_stmts.append(s)
            new_final = go(e.final_expr) if e.final_expr is not None else None
            return A.Block(span=e.span, stmts=new_stmts, final_expr=new_final)
        if isinstance(e, A.If):
            # Recurse into then/else regardless of whether they're Blocks
            # — defensively handle hand-built ASTs with bare-expr branches.
            new_then = go(e.then) if e.then is not None else None
            new_else = go(e.else_) if e.else_ is not None else None
            return A.If(span=e.span, cond=go(e.cond),
                        then=new_then, else_=new_else)
        return e

    return go(expr)


def _substitute_names(expr: A.Expr, subs: dict[str, A.Expr]) -> A.Expr:
    """Replace each occurrence of A.Name(n) where n in subs with subs[n].
    Block-scoped: a `let` shadowing a substituted name removes it from the
    scope of the rest of the block."""
    import copy as _copy

    def go(e: A.Expr, env: dict[str, A.Expr]) -> A.Expr:
        if isinstance(e, A.Name):
            if e.name in env:
                # Each substitution site gets its own copy so downstream
                # in-place mutation doesn't cross-contaminate.
                return _copy.deepcopy(env[e.name])
            return e
        if isinstance(e, A.Binary):
            return A.Binary(span=e.span, op=e.op,
                            left=go(e.left, env), right=go(e.right, env))
        if isinstance(e, A.Unary):
            return A.Unary(span=e.span, op=e.op, operand=go(e.operand, env))
        if isinstance(e, A.Call):
            return A.Call(span=e.span, callee=go(e.callee, env),
                          args=[go(a, env) for a in e.args])
        if isinstance(e, A.If):
            new_then = (_go_block(e.then, env) if isinstance(e.then, A.Block)
                        else go(e.then, env))
            new_else = (_go_block(e.else_, env)
                        if e.else_ is not None and isinstance(e.else_, A.Block)
                        else (go(e.else_, env) if e.else_ is not None else None))
            return A.If(span=e.span, cond=go(e.cond, env),
                        then=new_then, else_=new_else)
        if isinstance(e, A.Block):
            return _go_block(e, env)
        return e

    def _go_block(blk: A.Block, env: dict[str, A.Expr]) -> A.Block:
        local_env = dict(env)
        new_stmts = []
        for s in blk.stmts:
            if isinstance(s, A.Let) and s.value is not None:
                new_val = go(s.value, local_env)
                # The let shadows any incoming substitution for the same name
                local_env.pop(s.name, None)
                new_stmts.append(A.Let(span=s.span, name=s.name, ty=s.ty,
                                        value=new_val, is_mut=s.is_mut))
            elif isinstance(s, A.ConstStmt):
                new_val = go(s.value, local_env)
                local_env.pop(s.name, None)
                new_stmts.append(A.ConstStmt(span=s.span, name=s.name,
                                              ty=s.ty, value=new_val))
            else:
                new_stmts.append(s)
        new_final = go(blk.final_expr, local_env) if blk.final_expr is not None else None
        return A.Block(span=blk.span, stmts=new_stmts, final_expr=new_final)

    return go(expr, subs)


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
    if isinstance(expr, A.Call):
        # Chain rule for known transcendentals: d(f(u))/dx = f'(u) * du/dx.
        # The call is rewritten so the derivative goes through the same
        # named function whose derivative is hardcoded here.
        deriv = _diff_call_chain_rule(expr, var, span)
        if deriv is not None:
            return deriv
    # Unsupported: emit zero (placeholder)
    return A.FloatLit(span=expr.span, value=0.0)


def _diff_call_chain_rule(call: A.Call, var: str,
                          span: A.Span) -> Optional[A.Expr]:
    """Apply the analytic derivative for known transcendental builtins.
    Returns None if the callee isn't a recognised transcendental."""
    if not isinstance(call.callee, A.Name):
        return None
    # Handle __powi(x, n) separately: 2-arg with n literal int.
    # d(x^n)/dx = n * x^(n-1) * dx/dvar.
    if call.callee.name == "__powi" and len(call.args) == 2:
        x = call.args[0]
        n_arg = call.args[1]
        if isinstance(n_arg, A.IntLit):
            n_val = n_arg.value
            dx = _diff(x, var)
            if n_val <= 0:
                # x^0 = 1 → derivative 0; x^negative not supported in __powi
                return A.FloatLit(span=span, value=0.0)
            # __powi caps at 16 in stdlib (transcendentals.hx). Match the cap
            # in the chain rule so the gradient stays consistent with the
            # value the runtime actually produces. Derivative for n>16 is
            # therefore 16 * x^15 (the saturated tail).
            if n_val > 16:
                n_val = 16
            # n * __powi(x, n-1) * dx
            n_lit = A.FloatLit(span=span, value=float(n_val))
            n_minus_one = A.IntLit(span=span, value=n_val - 1)
            x_pow = A.Call(span=span,
                           callee=A.Name(span=span, name="__powi"),
                           args=[x, n_minus_one])
            return A.Binary(span=span, op="*",
                            left=A.Binary(span=span, op="*",
                                          left=n_lit, right=x_pow),
                            right=dx)
        # Non-literal n: fall through to zero derivative.
    if len(call.args) != 1:
        return None
    name = call.callee.name
    u = call.args[0]
    du = _diff(u, var)

    def mul(a: A.Expr, b: A.Expr) -> A.Expr:
        return A.Binary(span=span, op="*", left=a, right=b)

    def call1(fn: str, arg: A.Expr) -> A.Expr:
        return A.Call(span=span, callee=A.Name(span=span, name=fn), args=[arg])

    if name == "__exp":
        # d(exp(u))/dx = exp(u) * du/dx
        return mul(call1("__exp", u), du)
    if name == "__log":
        # d(log(u))/dx = (1/u) * du/dx
        recip = A.Binary(span=span, op="/",
                         left=A.FloatLit(span=span, value=1.0), right=u)
        return mul(recip, du)
    if name == "__sin":
        # d(sin(u))/dx = cos(u) * du/dx
        return mul(call1("__cos", u), du)
    if name == "__cos":
        # d(cos(u))/dx = -sin(u) * du/dx
        neg_sin = A.Unary(span=span, op="-", operand=call1("__sin", u))
        return mul(neg_sin, du)
    if name == "__sqrt":
        # d(sqrt(u))/dx = (1 / (2*sqrt(u))) * du/dx
        sqrt_u = call1("__sqrt", u)
        denom = A.Binary(span=span, op="*",
                         left=A.FloatLit(span=span, value=2.0), right=sqrt_u)
        recip = A.Binary(span=span, op="/",
                         left=A.FloatLit(span=span, value=1.0), right=denom)
        return mul(recip, du)
    if name == "__relu":
        # d(relu(u))/dx = (1 if u > 0 else 0) * du/dx
        # IMPORTANT: cond and else_ each get their OWN FloatLit(0.0) — they
        # must not share a node, otherwise downstream in-place AST mutation
        # passes (grad_pass alias resolution) corrupt both branches at once.
        cond = A.Binary(span=span, op=">", left=u,
                        right=A.FloatLit(span=span, value=0.0))
        gated = A.If(span=span, cond=cond,
                     then=A.Block(span=span, stmts=[],
                                  final_expr=A.FloatLit(span=span, value=1.0)),
                     else_=A.Block(span=span, stmts=[],
                                   final_expr=A.FloatLit(span=span, value=0.0)))
        return mul(gated, du)
    if name == "__sigmoid":
        # d(sigmoid(u))/dx = sigmoid(u) * (1 - sigmoid(u)) * du/dx
        # The two __sigmoid(u) call nodes get DEEPCOPIES of u so the second
        # call doesn't share its argument tree with the first — protects
        # against in-place mutation by later passes.
        import copy as _copy
        s1 = call1("__sigmoid", _copy.deepcopy(u))
        s2 = call1("__sigmoid", _copy.deepcopy(u))
        one_minus = A.Binary(span=span, op="-",
                             left=A.FloatLit(span=span, value=1.0), right=s1)
        return mul(mul(s2, one_minus), du)
    if name == "__tanh":
        # d(tanh(u))/dx = (1 - tanh(u)^2) * du/dx. Two distinct __tanh(u)
        # call nodes (each with deep-copied u) so neither side of the
        # square shares structure with the other — same protection used
        # by __sigmoid below to survive in-place AST mutation by
        # downstream passes.
        import copy as _copy
        t1 = call1("__tanh", _copy.deepcopy(u))
        t2 = call1("__tanh", _copy.deepcopy(u))
        t_sq = A.Binary(span=span, op="*", left=t1, right=t2)
        one_minus = A.Binary(span=span, op="-",
                             left=A.FloatLit(span=span, value=1.0), right=t_sq)
        return mul(one_minus, du)
    if name == "__softplus":
        # d(softplus(u))/dx = sigmoid(u) * du/dx
        return mul(call1("__sigmoid", u), du)
    if name == "__silu":
        # d(silu(u))/dx = sigmoid(u) + u * sigmoid(u) * (1 - sigmoid(u)) * du/dx
        # = sigmoid(u) * (1 + u * (1 - sigmoid(u))) * du/dx
        import copy as _copy
        s1 = call1("__sigmoid", _copy.deepcopy(u))
        s2 = call1("__sigmoid", _copy.deepcopy(u))
        one_minus_s = A.Binary(span=span, op="-",
                               left=A.FloatLit(span=span, value=1.0), right=s2)
        u_times_oms = A.Binary(span=span, op="*", left=_copy.deepcopy(u),
                               right=one_minus_s)
        inner = A.Binary(span=span, op="+",
                         left=A.FloatLit(span=span, value=1.0),
                         right=u_times_oms)
        return mul(mul(s1, inner), du)
    if name == "__abs":
        # d(abs(u))/dx = sign(u) * du/dx; at u=0 use 0.
        # Implement as if u>0 then 1 else (if u<0 then -1 else 0) * du.
        import copy as _copy
        u_copy = _copy.deepcopy(u)
        zero = A.FloatLit(span=span, value=0.0)
        cond_pos = A.Binary(span=span, op=">", left=u_copy,
                            right=A.FloatLit(span=span, value=0.0))
        cond_neg = A.Binary(span=span, op="<", left=_copy.deepcopy(u),
                            right=A.FloatLit(span=span, value=0.0))
        inner_else = A.If(span=span, cond=cond_neg,
                          then=A.Block(span=span, stmts=[],
                                       final_expr=A.FloatLit(span=span, value=-1.0)),
                          else_=A.Block(span=span, stmts=[], final_expr=zero))
        gated = A.If(span=span, cond=cond_pos,
                     then=A.Block(span=span, stmts=[],
                                  final_expr=A.FloatLit(span=span, value=1.0)),
                     else_=A.Block(span=span, stmts=[], final_expr=inner_else))
        return mul(gated, du)
    return None


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
