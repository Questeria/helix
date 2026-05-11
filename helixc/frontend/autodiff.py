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


# Audit 28.8 B5: trap 85001 — AD assumed 0 derivative for an unhandled
# expression kind. Both forward (_diff) and reverse (_propagate) used
# to fall through to "return 0" / "no contribution" for any unmatched
# node — Quote, Splice, Modify, UnsafeBlock, Cast on a non-arithmetic
# target. The user got `grad(f)(x) = 0` with no diagnostic.
#
# Fix: each unhandled-node site now appends a diagnostic to this
# module-level list. The CLI (helixc/check.py) flushes the list at
# the end of compilation and prints warnings to stderr; `-Wad=error`
# promotes them to errors. Tests can drain via `take_diff_warnings`.
_DIFF_WARNINGS: list[str] = []

# Trap-id reservation for "AD assumed 0 derivative" diagnostic.
TRAP_AD_ASSUMED_ZERO = 85001


# Audit 28.8 cycle 2 B:C9: shared numeric-type set for AD Cast arms.
# Pre-fix the Cast arms in both autodiff._diff and
# autodiff_reverse._propagate hardcoded a 14-element list that
# omitted `bool`, `char`, `fp8`, `mxfp4`, `nvfp4` — all five of
# which are accepted by typecheck's `_is_numeric_scalar`. So `x as
# bool` (valid per the matrix) inside a grad-rewritten fn emitted
# a spurious 85001 warning. Unify via a frozenset that mirrors
# typecheck's numeric domain.
#
# Note: bool/char casts are technically discontinuous, so a future
# enhancement could flag them with a separate diagnostic ("AD
# through discontinuous cast"). Phase-0: accept as numeric, do not
# spuriously warn.
NUMERIC_FOR_AD: frozenset[str] = frozenset({
    "i8", "i16", "i32", "i64", "isize",
    "u8", "u16", "u32", "u64", "usize",
    "f16", "bf16", "f32", "f64",
    "fp8", "mxfp4", "nvfp4",
    "bool", "char",
})


def take_diff_warnings() -> list[str]:
    """Atomically read-and-clear the module-level AD warning list.

    Callers (helixc/check.py and tests) should drain this list at a
    well-defined point so warnings from a previous compilation unit
    don't leak into the next. Multiple drains across one compile
    aggregate (the list is reset only on take())."""
    global _DIFF_WARNINGS
    out = _DIFF_WARNINGS
    _DIFF_WARNINGS = []
    return out


def _ad_warn(node, reason: str) -> None:
    """Record an AD diagnostic for an unhandled expression. Includes
    source span when available + trap-id 85001 for grep-ability."""
    span = getattr(node, "span", None)
    kind = type(node).__name__
    line_col = (f"{span.line}:{span.col}: " if span is not None
                else "")
    _DIFF_WARNINGS.append(
        f"{line_col}AD: assumed 0 derivative for {kind} ({reason}) "
        f"(trap {TRAP_AD_ASSUMED_ZERO})"
    )


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
    # Audit 28.8 cycle 2 (deferred observation #20): pre-fix this
    # `except Exception: key = None` silently disabled the cache on
    # any hash failure with NO diagnostic. Future AST extensions
    # could quietly skip caching forever (perf regression, no signal).
    # Narrowed to the actually-expected hashing exceptions; on a
    # genuine hash failure we still bypass the cache but emit a
    # warning via the AD channel so the user can spot the recurring
    # miss.
    try:
        key = (structural_hash(expr), var, _fn_table_sig(fn_table))
    except (TypeError, ValueError, AttributeError) as e:
        key = None  # hash failure → bypass cache
        _ad_warn(
            expr,
            f"differentiate cache bypassed: hashing failed "
            f"({type(e).__name__}: {e}) — perf regression but "
            f"correctness preserved",
        )
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


def _is_inferably_pure(fn: "A.FnDecl",
                        fn_table: dict[str, "A.FnDecl"],
                        visiting: frozenset[str] | None = None) -> bool:
    """Stage 13: infer whether a user fn is safe to inline for AD without an
    explicit `@pure` attribute. A fn is inferably pure iff its body uses only
    expressions whose gradient is well-defined and whose evaluation has no
    observable side-effects:

      - literals (int, float, bool, char, str)
      - parameter names / let-bound names
      - arithmetic Binary/Unary
      - If with pure cond/then/else
      - Block with pure let-stmts and pure final_expr (no Assign)
      - Calls to inferably-pure user fns or known transcendental builtins

    Anything else (Assign, For, While, Loop, Match, Index, calls to
    non-pure fns or unknown builtins) -> not inferred pure. The caller
    falls back to "leave as opaque call, derivative is 0" — same behaviour
    as before Stage 13.

    Used by `_inline_user_calls` so the plan test in Stage 13
    (`fn g(x) = x*x; fn f(x) = g(x)+x; grad(f)(3)=7`) works without forcing
    the user to mark every arithmetic helper `@pure`.
    """
    visiting = visiting or frozenset()
    # Cycles: if we're already inferring fn, treat as pure (the caller's
    # visiting-set in _inline_user_calls will block re-inlining anyway).
    if fn.name in visiting:
        return True
    if "pure" in fn.attrs:
        return True
    new_visiting = visiting | {fn.name}

    def is_pure_expr(e) -> bool:
        if e is None:
            return True
        if isinstance(e, (A.IntLit, A.FloatLit, A.BoolLit, A.CharLit, A.StrLit)):
            return True
        if isinstance(e, A.Name):
            return True
        if isinstance(e, A.Binary):
            return is_pure_expr(e.left) and is_pure_expr(e.right)
        if isinstance(e, A.Unary):
            return is_pure_expr(e.operand)
        if isinstance(e, A.Cast):
            return is_pure_expr(e.value)
        if isinstance(e, A.Block):
            for s in e.stmts:
                if isinstance(s, A.Let) and s.value is not None:
                    if not is_pure_expr(s.value):
                        return False
                elif isinstance(s, A.ConstStmt):
                    if not is_pure_expr(s.value):
                        return False
                elif isinstance(s, A.ExprStmt):
                    if not is_pure_expr(s.expr):
                        return False
                else:
                    # Assign/For/While/Loop/Return/etc. -> impure.
                    return False
            return is_pure_expr(e.final_expr)
        if isinstance(e, A.If):
            return (is_pure_expr(e.cond)
                    and is_pure_expr(e.then)
                    and is_pure_expr(e.else_))
        if isinstance(e, A.Call):
            if not isinstance(e.callee, A.Name):
                return False
            cname = e.callee.name
            # Recurse into args first.
            for a in e.args:
                if not is_pure_expr(a):
                    return False
            # Known-pure transcendental builtins (their analytic chain rules
            # are wired into _diff_call_chain_rule).
            TRANSCENDENTALS = {"__exp", "__log", "__sin", "__cos", "__sqrt",
                               "__relu", "__sigmoid", "__tanh", "__softplus",
                               "__silu", "__abs", "__gelu", "__powi",
                               "__min", "__max", "__clamp",
                               "__min_i32", "__max_i32", "__clamp_i32"}
            if cname in TRANSCENDENTALS:
                return True
            # User fn — recursively check.
            if cname in fn_table:
                return _is_inferably_pure(fn_table[cname], fn_table,
                                           new_visiting)
            # Unknown callee — conservative reject.
            return False
        # Anything else (Match, For, While, Loop, Assign, Return, Index,
        # Tuple-related, struct-related): not inferable as pure for AD.
        return False

    return is_pure_expr(fn.body)


def _inline_user_calls(expr: A.Expr, fn_table: dict[str, "A.FnDecl"],
                        depth: int = 0, max_depth: int = 4,
                        visiting: frozenset[str] | None = None) -> A.Expr:
    """Walk `expr` and replace each Call(Name(f), args) where f is a known
    inlinable function in `fn_table` with a deepcopy of f's body, with each
    parameter substituted by the corresponding argument expression.

    A function is inlinable if either it has the `@pure` attribute OR its
    body is inferably pure (Stage 13: only arithmetic / pure-call chain).
    Stage 13 added the inferred-purity path so plain helper fns work in
    `grad(f)` without forcing the user to mark every arithmetic helper.

    Skips:
      - Transcendental builtins (`__exp`, `__log`, etc.) — they have
        analytic AD chain rules already wired into _diff.
      - Functions currently in `visiting` (mutual / direct recursion
        guard — prevents exponential AST expansion when inlining cycles
        like a→b→a). Stage 13: traps via the trap-id 87001 documented in
        the plan; runtime impact is "leave call as opaque, gradient is 0".
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
                       "__silu", "__abs", "__gelu", "__powi",
                       # min/max/clamp aren't differentiable at switch
                       # points; treat as opaque (zero gradient) rather
                       # than try to inline (audit-10).
                       "__min", "__max", "__clamp",
                       "__min_i32", "__max_i32", "__clamp_i32"}
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
                # Stage 13: inline if @pure OR inferably pure (arithmetic/
                # pure-call chain). Other fns may have effects whose
                # differentiation is unsound — leave as opaque call.
                if ("pure" not in fn.attrs
                        and not _is_inferably_pure(fn, fn_table)):
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


def _is_reassigned_after(stmts: list, name: str, start_idx: int) -> bool:
    """Return True if any statement after `start_idx` reassigns `name` via
    A.Assign anywhere in its expression tree. Used by `_inline_lets` to
    decide whether a `let mut` is effectively single-assignment (safe to
    inline) or genuinely mutable (must be left alone)."""
    def _has_assign(node) -> bool:
        if node is None:
            return False
        if isinstance(node, A.Assign):
            if isinstance(node.target, A.Name) and node.target.name == name:
                return True
            return _has_assign(node.value)
        # Recurse into common containers
        for attr in ("operand", "left", "right", "value", "expr",
                     "scrutinee", "cond", "iter_expr", "callee", "obj"):
            if hasattr(node, attr) and _has_assign(getattr(node, attr)):
                return True
        for attr in ("args", "elems", "stmts", "indices", "alts",
                     "sub_patterns"):
            if hasattr(node, attr):
                for child in getattr(node, attr) or []:
                    if _has_assign(child):
                        return True
        if isinstance(node, A.Block):
            for s in node.stmts:
                if _has_assign(s):
                    return True
            if node.final_expr is not None and _has_assign(node.final_expr):
                return True
        if isinstance(node, A.If):
            if _has_assign(node.then) or _has_assign(node.else_):
                return True
        if isinstance(node, A.Match):
            for arm in node.arms:
                if _has_assign(arm.body):
                    return True
        return False
    for i in range(start_idx + 1, len(stmts)):
        if _has_assign(stmts[i]):
            return True
    return False


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
            # Inline `let` and `let mut` bindings whose name is never
            # reassigned within the rest of the block. The conservative
            # mut-skip from audit-10 was over-restrictive: it produced
            # gradient 0 for `let mut acc = x; acc` (no reassignment),
            # because `acc` had no inlining and the differentiator
            # treated the bare Name as a non-var. Cycle-4 fix: only
            # skip mutable lets that ARE actually reassigned later in
            # the same block. Single-assignment mut bindings are
            # functionally pure for AD purposes.
            if (isinstance(stmt, A.Let) and stmt.value is not None
                    and (not stmt.is_mut
                         or not _is_reassigned_after(expr.stmts, stmt.name, expr.stmts.index(stmt)))):
                local_env[stmt.name] = _inline_lets(stmt.value, local_env)
            # ExprStmt: ignore (no derivative meaning)
            # ConstStmt: similar to Let (immutable by construction)
            elif isinstance(stmt, A.ConstStmt):
                local_env[stmt.name] = _inline_lets(stmt.value, local_env)
        if expr.final_expr is not None:
            return _inline_lets(expr.final_expr, local_env)
        # Audit 28.8 cycle 2 (deferred observation #18): pre-fix this
        # returned FloatLit(0.0) silently when a Block had stmts but no
        # final expression. The let-stmts were inlined into env (so any
        # later use of bound names would resolve) but the Block's value
        # defaulted to 0 with no diagnostic. Now we WARN so the user
        # can spot the missing tail expression in an AD context.
        _ad_warn(
            expr,
            "empty block in AD context: no final expression — "
            "assumed 0",
        )
        return A.FloatLit(span=expr.span, value=0.0)
    if isinstance(expr, A.If):
        # Inline both branches and re-wrap in an If — the inliner only flattens
        # let-bindings, branch selection stays a runtime decision. Differentiate
        # both branches; the derivative is then the same conditional.
        # Audit 28.8 cycle 4 C4-3: also inline `expr.cond`. Pre-fix the
        # cond was passed through unmodified, so `let g = grad(loss);
        # if g(x) > 0.0 { ... }` left `g` unsubstituted in the cond.
        # Symmetric with While/For/Match coverage in cycle 3.
        new_cond = _inline_lets(expr.cond, env)
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
        return A.If(span=expr.span, cond=new_cond,
                    then=wrapped_then, else_=wrapped_else)
    # Audit 28.8 cycle 3 C3-5: extend _inline_lets to recurse through every
    # Expr subtype that can contain Name leaves. Pre-fix the function fell
    # through to `return expr` for Cast / Call / Field / Index / Match /
    # ArrayLit / TupleLit / StructLit / Range / Return / Break / Assign /
    # UnsafeBlock / Loop / For / While / Quote / Splice / Modify — so
    # any let-bound name appearing under those positions was never
    # substituted, defeating the reverse-mode `_ad_warn` reach (C2-3).
    if isinstance(expr, A.Cast):
        return A.Cast(span=expr.span,
                      value=_inline_lets(expr.value, env),
                      target_ty=expr.target_ty)
    if isinstance(expr, A.Call):
        new_args = [_inline_lets(a, env) for a in expr.args]
        # Callee is generally a Name or Path — Name lookups in env would
        # turn it into a different expression entirely, which breaks
        # ordinary calls. Only substitute when the resolved value is
        # itself a Name/Path (alias-of-callee).
        new_callee = expr.callee
        if isinstance(expr.callee, A.Name) and expr.callee.name in env:
            cand = env[expr.callee.name]
            # Audit 28.8 cycle 4 E6: preserve the original callee's
            # generics list (turbofish) when aliasing. Pre-fix, `let g
            # = mk_grad; g::<f64>(x)` aliased to `mk_grad(x)` and
            # dropped the `::<f64>` annotation, defeating monomorphization.
            if isinstance(cand, A.Name):
                new_callee = A.Name(
                    span=cand.span, name=cand.name,
                    generics=(list(expr.callee.generics)
                              if expr.callee.generics
                              else list(cand.generics)),
                )
            elif isinstance(cand, A.Path):
                new_callee = cand
        # Audit 28.8 cycle 4 E8: walk Field-typed callees so
        # `obj.method()` with `obj` let-bound substitutes properly.
        elif isinstance(expr.callee, A.Field):
            new_callee = _inline_lets(expr.callee, env)
        return A.Call(span=expr.span, callee=new_callee, args=new_args)
    if isinstance(expr, A.Field):
        return A.Field(span=expr.span,
                       obj=_inline_lets(expr.obj, env),
                       name=expr.name)
    if isinstance(expr, A.Index):
        return A.Index(
            span=expr.span,
            callee=_inline_lets(expr.callee, env),
            indices=[_inline_lets(i, env) for i in expr.indices],
        )
    if isinstance(expr, A.ArrayLit):
        return A.ArrayLit(
            span=expr.span,
            elems=[_inline_lets(e, env) for e in expr.elems],
        )
    if isinstance(expr, A.TupleLit):
        return A.TupleLit(
            span=expr.span,
            elems=[_inline_lets(e, env) for e in expr.elems],
        )
    if isinstance(expr, A.StructLit):
        return A.StructLit(
            span=expr.span,
            name=expr.name,
            fields=[(n, _inline_lets(v, env)) for (n, v) in expr.fields],
        )
    if isinstance(expr, A.Range):
        return A.Range(
            span=expr.span,
            start=_inline_lets(expr.start, env),
            end=_inline_lets(expr.end, env),
        )
    if isinstance(expr, A.Return):
        return A.Return(
            span=expr.span,
            value=_inline_lets(expr.value, env),
        )
    if isinstance(expr, A.Break):
        return A.Break(
            span=expr.span,
            value=_inline_lets(expr.value, env),
        )
    if isinstance(expr, A.Assign):
        return A.Assign(
            span=expr.span,
            target=_inline_lets(expr.target, env),
            op=expr.op,
            value=_inline_lets(expr.value, env),
        )
    if isinstance(expr, A.UnsafeBlock):
        body = expr.body
        new_body = _inline_lets(body, env) if isinstance(body, A.Block) else body
        if not isinstance(new_body, A.Block):
            new_body = A.Block(span=body.span, stmts=[], final_expr=new_body)
        return A.UnsafeBlock(span=expr.span, body=new_body)
    if isinstance(expr, A.Match):
        new_arms = []
        for arm in expr.arms:
            new_arms.append(A.MatchArm(
                span=arm.span,
                pattern=arm.pattern,
                guard=(_inline_lets(arm.guard, env)
                       if arm.guard is not None else None),
                body=_inline_lets(arm.body, env),
            ))
        return A.Match(
            span=expr.span,
            scrutinee=_inline_lets(expr.scrutinee, env),
            arms=new_arms,
        )
    if isinstance(expr, A.Loop):
        body = expr.body
        new_body = _inline_lets(body, env) if isinstance(body, A.Block) else body
        if not isinstance(new_body, A.Block):
            new_body = A.Block(span=body.span, stmts=[], final_expr=new_body)
        return A.Loop(span=expr.span, body=new_body)
    if isinstance(expr, A.For):
        body = expr.body
        new_body = _inline_lets(body, env) if isinstance(body, A.Block) else body
        if not isinstance(new_body, A.Block):
            new_body = A.Block(span=body.span, stmts=[], final_expr=new_body)
        return A.For(
            span=expr.span,
            var_name=expr.var_name,
            iter_expr=_inline_lets(expr.iter_expr, env),
            body=new_body,
        )
    if isinstance(expr, A.While):
        body = expr.body
        new_body = _inline_lets(body, env) if isinstance(body, A.Block) else body
        if not isinstance(new_body, A.Block):
            new_body = A.Block(span=body.span, stmts=[], final_expr=new_body)
        return A.While(
            span=expr.span,
            cond=_inline_lets(expr.cond, env),
            body=new_body,
        )
    if isinstance(expr, A.Quote):
        return A.Quote(span=expr.span, inner=_inline_lets(expr.inner, env))
    if isinstance(expr, A.Splice):
        return A.Splice(span=expr.span, inner=_inline_lets(expr.inner, env))
    if isinstance(expr, A.Modify):
        return A.Modify(
            span=expr.span,
            target=_inline_lets(expr.target, env),
            transformation=_inline_lets(expr.transformation, env),
            verifier=_inline_lets(expr.verifier, env),
        )
    # Audit 28.8 cycle 4 C4-1: leaf-like exprs that hold no let-bindable
    # children — Path (qualified name like `Maybe::None`), Continue
    # (statement-expr with no children), TileLit (compile-time shape +
    # init marker). Pre-fix the catch-all fired spurious 85001 warnings
    # for any enum-variant reference in a differentiated fn body —
    # `-Wad=error` then failed to compile legitimate AD code.
    if isinstance(expr, A.Path):
        return expr
    if isinstance(expr, A.Continue):
        return expr
    if isinstance(expr, A.TileLit):
        # Audit 28.8 cycle 6 C5-3 / F4: TileLit has Expr children
        # (shape: list[Expr], memspace: Expr). The cycle-4 identity arm
        # dropped let-bound names appearing in those positions. Walk
        # children so `let N = 4; tile<f32, [N], REG>::zeros()` (the
        # legitimate user idiom) substitutes correctly.
        return A.TileLit(
            span=expr.span,
            dtype=expr.dtype,
            shape=[_inline_lets(s, env) for s in expr.shape],
            memspace=_inline_lets(expr.memspace, env),
            init=expr.init,
        )
    # Catch-all fallthrough: warn loud so future AST extensions surface
    # immediately rather than silently dropping let-bindings. Only fires
    # when an Expr subtype is genuinely unhandled (not just a no-op
    # leaf — those are explicit arms above).
    #
    # Cycle 4 C4-3: do NOT pre-embed the trap id in the reason — _ad_warn
    # appends `(trap {TRAP_AD_ASSUMED_ZERO})` to every message. Pre-fix,
    # the rendered warning contained `(trap 85001)` twice.
    _ad_warn(
        expr,
        f"_inline_lets fell through on Expr subtype "
        f"'{type(expr).__name__}' — let-bindings beyond this point may "
        f"not be substituted",
    )
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
        # Audit 28.8 B5: unknown call site — emit warning + 0.
        _ad_warn(expr, f"unrecognized call to "
                       f"{getattr(expr.callee, 'name', '<?>')!r}")
        return A.FloatLit(span=expr.span, value=0.0)
    # Audit 28.8 B5: Cast arm. Numeric `x as f64` propagates the
    # derivative through (chain-rule factor is 1 for numeric widening).
    # Non-numeric Cast (e.g., `x as *T`) returns 0 with a warning.
    if isinstance(expr, A.Cast):
        tgt = expr.target_ty
        # Audit 28.8 cycle 2 B:C9: shared NUMERIC_FOR_AD set covers
        # bool/char/fp8/mxfp4/nvfp4 too.
        if isinstance(tgt, A.TyName) and tgt.name in NUMERIC_FOR_AD:
            # Inner derivative carries through.
            return _diff(expr.value, var)
        _ad_warn(expr, f"cast to non-numeric target "
                       f"{type(tgt).__name__}")
        return A.FloatLit(span=span, value=0.0)
    # Audit 28.8 B5: Quote/Splice/Modify/UnsafeBlock fall here and were
    # previously silently zeroed. Now we WARN. UnsafeBlock specifically
    # should propagate AD through its body — handle that case.
    if isinstance(expr, A.UnsafeBlock):
        body = expr.body
        if isinstance(body, A.Block):
            return _diff_block_or_expr(body, var, span)
        return _diff(body, var)
    if isinstance(expr, (A.Quote, A.Splice, A.Modify)):
        _ad_warn(expr, f"{type(expr).__name__} is not differentiable")
        return A.FloatLit(span=span, value=0.0)
    # Genuinely-unknown — warn loudly.
    _ad_warn(expr, "unhandled expression kind")
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
            if n_val <= 0 or n_val > 16:
                # __powi(x, n) returns constant 1.0 for n <= 0 or n > 16
                # (stdlib transcendentals.hx) — derivative is 0. Previously
                # we capped n_val to 16 here, producing a wrong gradient
                # `16 * x^15` for n > 16 even though the function itself
                # is constant at those inputs.
                return A.FloatLit(span=span, value=0.0)
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
            # Audit 28.8 cycle 2 (deferred observation #19): pre-fix
            # this `except Exception: pass` swallowed every error in
            # constant folding, falling through to the unsimplified
            # expression with no diagnostic. Narrowed to the actually
            # expected arithmetic exceptions (overflow, zero-divide,
            # value, type). Anything else surfaces as a real bug.
            try:
                if expr.op == "+":
                    return _make_const(l_val + r_val, expr.span)
                if expr.op == "-":
                    return _make_const(l_val - r_val, expr.span)
                if expr.op == "*":
                    return _make_const(l_val * r_val, expr.span)
                if expr.op == "/" and r_val != 0:
                    return _make_const(l_val / r_val, expr.span)
            except (OverflowError, ZeroDivisionError, ValueError, TypeError):
                # Genuine arithmetic limits — fall through unsimplified.
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
    if isinstance(expr, A.BoolLit):
        return "true" if expr.value else "false"
    if isinstance(expr, A.Name):
        return expr.name
    if isinstance(expr, A.Binary):
        return f"({fmt(expr.left)} {expr.op} {fmt(expr.right)})"
    if isinstance(expr, A.Unary):
        return f"({expr.op}{fmt(expr.operand)})"
    if isinstance(expr, A.Call):
        callee = fmt(expr.callee) if not isinstance(expr.callee, A.Name) else expr.callee.name
        return f"{callee}({', '.join(fmt(a) for a in expr.args)})"
    if isinstance(expr, A.Block):
        if expr.final_expr is not None and not expr.stmts:
            return fmt(expr.final_expr)
        return f"<Block>"
    if isinstance(expr, A.If):
        then_s = fmt(expr.then) if expr.then is not None else "()"
        else_s = fmt(expr.else_) if expr.else_ is not None else "()"
        return f"if {fmt(expr.cond)} {{ {then_s} }} else {{ {else_s} }}"
    if isinstance(expr, A.Match):
        arms = []
        for arm in expr.arms:
            arms.append(f"{_fmt_pattern(arm.pattern)} => {fmt(arm.body)}")
        return f"match {fmt(expr.scrutinee)} {{ {', '.join(arms)} }}"
    return f"<{type(expr).__name__}>"


def _fmt_pattern(pat: A.Pattern) -> str:
    if isinstance(pat, A.PatWildcard):
        return "_"
    if isinstance(pat, A.PatLit):
        return fmt(pat.value)
    if isinstance(pat, A.PatBind):
        prefix = "mut " if pat.is_mut else ""
        return f"{prefix}{pat.name}"
    if isinstance(pat, A.PatRange):
        return f"{fmt(pat.lo)}..{fmt(pat.hi)}"
    if isinstance(pat, A.PatOr):
        return " | ".join(_fmt_pattern(a) for a in pat.alts)
    if isinstance(pat, A.PatTuple):
        return f"({', '.join(_fmt_pattern(e) for e in pat.elems)})"
    if isinstance(pat, A.PatVariant):
        segs = pat.path.segments if hasattr(pat.path, "segments") else (
            pat.path if isinstance(pat.path, list) else [str(pat.path)]
        )
        path = "::".join(segs)
        if pat.sub_patterns:
            return f"{path}({', '.join(_fmt_pattern(s) for s in pat.sub_patterns)})"
        return path
    return f"<{type(pat).__name__}>"
