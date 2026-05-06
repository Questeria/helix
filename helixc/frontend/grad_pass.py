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
from .autodiff_reverse import differentiate_reverse


_GRAD_CALL_NAMES = frozenset({"grad", "grad_rev", "grad_rev_all"})


def _has_grad_call(prog: A.Program) -> bool:
    """Quick scan: does any function body contain a Call whose callee is
    Name("grad"|"grad_rev"|"grad_rev_all")? Used to short-circuit the
    expensive match_lower pre-pass when no AD work is needed."""
    for item in prog.items:
        if isinstance(item, A.FnDecl):
            if _expr_has_grad(item.body):
                return True
    return False


def _expr_has_grad(e) -> bool:
    if e is None:
        return False
    if isinstance(e, A.Call):
        if isinstance(e.callee, A.Name) and e.callee.name in _GRAD_CALL_NAMES:
            return True
        if _expr_has_grad(e.callee):
            return True
        return any(_expr_has_grad(a) for a in e.args)
    if isinstance(e, A.Block):
        for s in e.stmts:
            if isinstance(s, A.Let) and _expr_has_grad(s.value):
                return True
            if isinstance(s, A.ExprStmt) and _expr_has_grad(s.expr):
                return True
            if isinstance(s, A.ConstStmt) and _expr_has_grad(s.value):
                return True
        if _expr_has_grad(e.final_expr):
            return True
        return False
    if isinstance(e, A.If):
        return (_expr_has_grad(e.cond) or _expr_has_grad(e.then)
                or _expr_has_grad(e.else_))
    if isinstance(e, A.Match):
        if _expr_has_grad(e.scrutinee):
            return True
        return any(_expr_has_grad(a.body) or _expr_has_grad(a.guard)
                   for a in e.arms)
    if isinstance(e, A.Binary):
        return _expr_has_grad(e.left) or _expr_has_grad(e.right)
    if isinstance(e, A.Unary):
        return _expr_has_grad(e.operand)
    if isinstance(e, (A.For, A.While)):
        return (_expr_has_grad(getattr(e, 'cond', None)) or
                _expr_has_grad(getattr(e, 'iter_expr', None)) or
                _expr_has_grad(e.body))
    if isinstance(e, A.Loop):
        return _expr_has_grad(e.body)
    if isinstance(e, A.Cast):
        return _expr_has_grad(e.value)
    if isinstance(e, A.Return):
        return _expr_has_grad(e.value)
    if isinstance(e, A.Assign):
        return _expr_has_grad(e.target) or _expr_has_grad(e.value)
    return False


def grad_pass(prog: A.Program) -> int:
    """Walk the program; rewrite all grad(f) calls into references to
    generated f__grad functions. Returns count of grad calls rewritten.

    Also resolves let-aliases: 'let f = grad(loss); f(x)' is rewritten so
    the call to f becomes a direct call to loss__grad."""
    # Pre-pass: rewrite `match` to nested `if/let` so autodiff (which only
    # knows the smaller AST surface) never sees a Match node. Skip this
    # entirely when the program has no grad/grad_rev/grad_rev_all calls —
    # otherwise we'd desugar match in stdlib option/result helpers and
    # surface fake "if/else branches differ" warnings during typecheck.
    if _has_grad_call(prog):
        from .match_lower import lower_matches
        lower_matches(prog)
    # First: index existing functions by name
    fn_by_name: dict[str, A.FnDecl] = {}
    for item in prog.items:
        if isinstance(item, A.FnDecl):
            fn_by_name[item.name] = item

    new_fns: list[A.FnDecl] = []
    rewrite_count = 0

    # Walk all function bodies; rewrite grad(f) calls
    for item in prog.items:
        if isinstance(item, A.FnDecl):
            rewrite_count += _rewrite_in_block(item.body, fn_by_name, new_fns)

    # Add generated grad functions to the program. fn_by_name was already
    # updated inline as each grad function was generated (so nested grads
    # could resolve), so we only need to splice into prog.items here.
    existing_names = {item.name for item in prog.items if isinstance(item, A.FnDecl)}
    for new_fn in new_fns:
        if new_fn.name not in existing_names:
            prog.items.append(new_fn)
            existing_names.add(new_fn.name)

    # Second pass: resolve let-aliases.
    # 'let f = some_name;' creates an alias. When we see f(args), we
    # rewrite the call's callee to some_name (if some_name is a known fn).
    for item in prog.items:
        if isinstance(item, A.FnDecl):
            _resolve_let_aliases(item.body, fn_by_name, {})

    return rewrite_count


def _resolve_let_aliases(block: A.Block, fn_by_name: dict[str, A.FnDecl],
                         alias_env: dict[str, str]) -> None:
    """Walk a block. Track let-bindings that alias a function name. Rewrite
    Call expressions whose callee is an alias to point at the underlying
    function name."""
    local_env = dict(alias_env)
    for stmt in block.stmts:
        if isinstance(stmt, A.Let) and stmt.value is not None:
            # If the value is a Name pointing at a known function, track the
            # alias.
            _resolve_in_expr(stmt.value, fn_by_name, local_env)
            if isinstance(stmt.value, A.Name) and stmt.value.name in fn_by_name:
                local_env[stmt.name] = stmt.value.name
        elif isinstance(stmt, A.ExprStmt):
            _resolve_in_expr(stmt.expr, fn_by_name, local_env)
    if block.final_expr is not None:
        _resolve_in_expr(block.final_expr, fn_by_name, local_env)


def _resolve_in_expr(expr: A.Expr, fn_by_name: dict[str, A.FnDecl],
                     alias_env: dict[str, str]) -> None:
    if isinstance(expr, A.Call):
        # Resolve callee aliases
        if isinstance(expr.callee, A.Name) and expr.callee.name in alias_env:
            expr.callee = A.Name(span=expr.callee.span,
                                 name=alias_env[expr.callee.name])
        _resolve_in_expr(expr.callee, fn_by_name, alias_env)
        for a in expr.args:
            _resolve_in_expr(a, fn_by_name, alias_env)
    elif isinstance(expr, A.Binary):
        _resolve_in_expr(expr.left, fn_by_name, alias_env)
        _resolve_in_expr(expr.right, fn_by_name, alias_env)
    elif isinstance(expr, A.Unary):
        _resolve_in_expr(expr.operand, fn_by_name, alias_env)
    elif isinstance(expr, A.Cast):
        _resolve_in_expr(expr.value, fn_by_name, alias_env)
    elif isinstance(expr, A.Block):
        _resolve_let_aliases(expr, fn_by_name, alias_env)
    elif isinstance(expr, A.If):
        _resolve_in_expr(expr.cond, fn_by_name, alias_env)
        _resolve_let_aliases(expr.then, fn_by_name, alias_env)
        if expr.else_ is not None and isinstance(expr.else_, A.Block):
            _resolve_let_aliases(expr.else_, fn_by_name, alias_env)
    elif isinstance(expr, A.Index):
        _resolve_in_expr(expr.callee, fn_by_name, alias_env)
        for i in expr.indices:
            _resolve_in_expr(i, fn_by_name, alias_env)
    elif isinstance(expr, A.While):
        _resolve_in_expr(expr.cond, fn_by_name, alias_env)
        _resolve_let_aliases(expr.body, fn_by_name, alias_env)
    elif isinstance(expr, A.For):
        _resolve_in_expr(expr.iter_expr, fn_by_name, alias_env)
        _resolve_let_aliases(expr.body, fn_by_name, alias_env)


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
        # Post-order: recurse into args FIRST so inner grad(f) -> Name("f__grad")
        # is visible when we then check the outer grad pattern. This makes
        # grad(grad(f)) work: inner is rewritten + f__grad is registered in
        # fn_by_name, then the outer call is detected as grad(f__grad).
        new_callee, c1 = _rewrite_in_expr(expr.callee, fn_by_name, new_fns)
        new_args = []
        c2 = 0
        for a in expr.args:
            na, ca = _rewrite_in_expr(a, fn_by_name, new_fns)
            new_args.append(na)
            c2 += ca

        # grad_rev_all(f): emit a function that computes all gradients in
        # one source-level analysis and writes each ∂f/∂param[i] into
        # cells[base + i].
        if (isinstance(new_callee, A.Name)
                and new_callee.name == "grad_rev_all"
                and len(new_args) == 1
                and isinstance(new_args[0], A.Name)
                and new_args[0].name in fn_by_name):
            target = fn_by_name[new_args[0].name]
            grad_fn = _generate_grad_rev_all_fn(target, fn_by_name)
            if grad_fn is not None:
                if grad_fn.name not in fn_by_name:
                    new_fns.append(grad_fn)
                    fn_by_name[grad_fn.name] = grad_fn
                return (A.Name(span=expr.span, name=grad_fn.name), c1 + c2 + 1)

        # Now check if the (possibly-rewritten) call is grad(f) / grad(f, n)
        # or grad_rev(f) / grad_rev(f, n).
        if (isinstance(new_callee, A.Name)
                and new_callee.name in ("grad", "grad_rev")
                and len(new_args) in (1, 2)
                and isinstance(new_args[0], A.Name)
                and new_args[0].name in fn_by_name):
            target = fn_by_name[new_args[0].name]
            param_idx = _extract_param_idx_from_args(new_args, target,
                                                      kind=new_callee.name)
            mode = "reverse" if new_callee.name == "grad_rev" else "forward"
            grad_fn = _generate_grad_fn(target, param_idx, mode=mode,
                                         fn_table=fn_by_name)
            if grad_fn is not None:
                # Don't add duplicates if grad(f, n) is called multiple times
                if grad_fn.name not in fn_by_name:
                    new_fns.append(grad_fn)
                    fn_by_name[grad_fn.name] = grad_fn
                return (A.Name(span=expr.span, name=grad_fn.name), c1 + c2 + 1)
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
        new_cond, c_cond = _rewrite_in_expr(expr.cond, fn_by_name, new_fns)
        expr.cond = new_cond
        c_then = _rewrite_in_block(expr.then, fn_by_name, new_fns)
        c_else = 0
        if expr.else_ is not None and isinstance(expr.else_, A.Block):
            c_else = _rewrite_in_block(expr.else_, fn_by_name, new_fns)
        return (expr, c_cond + c_then + c_else)
    if isinstance(expr, A.Cast):
        new_inner, c = _rewrite_in_expr(expr.value, fn_by_name, new_fns)
        expr.value = new_inner
        return (expr, c)
    if isinstance(expr, A.Assign):
        new_val, c = _rewrite_in_expr(expr.value, fn_by_name, new_fns)
        expr.value = new_val
        return (expr, c)
    if isinstance(expr, A.Index):
        new_callee, c1 = _rewrite_in_expr(expr.callee, fn_by_name, new_fns)
        expr.callee = new_callee
        c2 = 0
        for i, idx in enumerate(expr.indices):
            new_idx, ci = _rewrite_in_expr(idx, fn_by_name, new_fns)
            expr.indices[i] = new_idx
            c2 += ci
        return (expr, c1 + c2)
    if isinstance(expr, A.While):
        new_cond, c1 = _rewrite_in_expr(expr.cond, fn_by_name, new_fns)
        expr.cond = new_cond
        c2 = _rewrite_in_block(expr.body, fn_by_name, new_fns)
        return (expr, c1 + c2)
    if isinstance(expr, A.For):
        new_iter, c1 = _rewrite_in_expr(expr.iter_expr, fn_by_name, new_fns)
        expr.iter_expr = new_iter
        c2 = _rewrite_in_block(expr.body, fn_by_name, new_fns)
        return (expr, c1 + c2)
    return (expr, count)


def _extract_param_idx_from_args(args: list[A.Expr], target: A.FnDecl,
                                  kind: str = "grad") -> int:
    """Pull the param index from `grad(f, n)` / `grad_rev(f, n)` args, or
    default to 0 for single-param functions. Multi-param functions REQUIRE
    an explicit index — silently differentiating only param 0 of a
    multi-param function is a correctness footgun.

    Raises ValueError on bad input so the user sees the problem, instead of
    getting a silently-wrong gradient.
    """
    if len(args) == 1:
        if len(target.params) > 1:
            raise ValueError(
                f"{kind}({target.name}) is ambiguous: {target.name} has "
                f"{len(target.params)} parameters. Use {kind}({target.name}, n) "
                f"to choose which parameter to differentiate w.r.t. "
                f"(0-indexed)."
            )
        return 0
    # Two args: kind(f, n) — n must be a non-negative IntLit in range
    idx_arg = args[1]
    if not isinstance(idx_arg, A.IntLit):
        raise ValueError(
            f"{kind}({target.name}, n): the index n must be a literal integer, "
            f"got {type(idx_arg).__name__}."
        )
    idx = idx_arg.value
    if idx < 0 or idx >= len(target.params):
        raise ValueError(
            f"{kind}({target.name}, {idx}): index out of range "
            f"(function has {len(target.params)} parameter(s))."
        )
    return idx


def _generate_grad_rev_all_fn(fn: A.FnDecl,
                                fn_table: dict[str, A.FnDecl]) -> A.FnDecl | None:
    """Build `<fn.name>__rgrad_all` — a single function that computes all
    parameter gradients via reverse-mode AD in one source-level pass and
    writes each into a reflection cell.

    Generated signature: same params as `fn` plus a trailing `base: i32`
    cell-base index. Body computes each ∂f/∂param_i, then for each i emits
    `modify_f(base + i, g_i, __always_accept)` to store it.

    Returns 0 on success. The caller reads the gradients back via
    splice_f(base + i).
    """
    if not fn.params:
        return None
    span = fn.span
    var_names = [p.name for p in fn.params]

    # Compute all gradients in one analysis pass.
    all_grads = differentiate_reverse(fn.body, var_names, fn_table=fn_table)

    # Build the body: for each param, let g_i = <gradient_expr>, then
    # modify_f(base + i, g_i, __always_accept).
    body_stmts: list[A.Stmt] = []
    base_name = "__base"
    for i, p_name in enumerate(var_names):
        g_var = f"__g_{i}"
        body_stmts.append(A.Let(
            span=span, name=g_var, ty=None,
            value=all_grads[p_name], is_mut=False,
        ))
        # base + i
        idx_expr = (A.Name(span=span, name=base_name) if i == 0
                    else A.Binary(span=span, op="+",
                                   left=A.Name(span=span, name=base_name),
                                   right=A.IntLit(span=span, value=i)))
        # modify_f(idx, g_i, __always_accept)
        call = A.Call(
            span=span,
            callee=A.Name(span=span, name="modify_f"),
            args=[idx_expr,
                  A.Name(span=span, name=g_var),
                  A.Name(span=span, name="__always_accept")],
        )
        body_stmts.append(A.ExprStmt(span=span, expr=call))

    # Final expression: 0 (success).
    new_body = A.Block(
        span=span, stmts=body_stmts,
        final_expr=A.IntLit(span=span, value=0),
    )

    # Params: original f's params (cast to f32) + trailing base: i32.
    new_params = [
        A.FnParam(span=p.span, name=p.name,
                  ty=A.TyName(span=p.ty.span, name="f32"))
        for p in fn.params
    ]
    new_params.append(A.FnParam(
        span=span, name=base_name,
        ty=A.TyName(span=span, name="i32"),
    ))

    return A.FnDecl(
        span=span,
        name=f"{fn.name}__rgrad_all",
        generics=[],
        params=new_params,
        return_ty=A.TyName(span=span, name="i32"),
        where_clauses=[],
        body=new_body,
        attrs=[],   # not @pure: it has modify_self effect
        is_pub=fn.is_pub,
    )


def _generate_grad_fn(fn: A.FnDecl, param_idx: int = 0,
                       mode: str = "forward",
                       fn_table: dict[str, A.FnDecl] | None = None
                       ) -> A.FnDecl | None:
    """Build a `<fn.name>__grad_<n>` (or `__rgrad_<n>`) FnDecl whose body is
    the derivative of `fn`'s body w.r.t. parameter `param_idx`.

    `mode` selects the AD engine. The result for ALL parameters is computed
    on first request and cached on `fn._helix_grad_cache`, so repeated
    grad(f, n) calls (for different n on the same f) share work — typical
    speedup 3-10× when a function has many parameters and the user calls
    grad_rev(f, n) for several values of n.
    """
    import copy as _copy
    if not fn.params:
        return None
    if param_idx < 0 or param_idx >= len(fn.params):
        return None
    var = fn.params[param_idx].name

    # Lazy per-FnDecl cache. Keyed by (mode, all-param-names tuple) so we
    # invalidate if the function's signature changes (defensive — the AST
    # is mostly immutable but grad_pass mutates some fields).
    cache_key = (mode, tuple(p.name for p in fn.params))
    cache = getattr(fn, "_helix_grad_cache", None)
    if cache is None or cache.get("_key") != cache_key:
        # Compute gradients for ALL parameters at once.
        all_vars = [p.name for p in fn.params]
        if mode == "reverse":
            grads_dict = differentiate_reverse(fn.body, all_vars,
                                                 fn_table=fn_table)
        else:
            # Forward mode is one-pass-per-var by construction; do them
            # all up front so subsequent param_idx's hit the cache.
            grads_dict = {v: differentiate(fn.body, v, fn_table=fn_table)
                          for v in all_vars}
        cache = {"_key": cache_key, "grads": grads_dict}
        try:
            fn._helix_grad_cache = cache
        except (AttributeError, TypeError):
            # FnDecl might be a frozen dataclass — fall back to no-cache.
            pass

    # Each generated grad fn gets its own deepcopy of the cached gradient
    # so subsequent passes (alias-resolution, lowering) can mutate the AST
    # without corrupting the cache.
    deriv = _copy.deepcopy(cache["grads"][var])
    # Wrap the derivative expression in a block (the FnDecl expects a Block body)
    new_body = A.Block(span=fn.body.span, stmts=[], final_expr=deriv)

    # Build new params (same names, all f32 — gradient takes plain floats)
    new_params = [
        A.FnParam(span=p.span, name=p.name,
                  ty=A.TyName(span=p.ty.span, name="f32"))
        for p in fn.params
    ]

    suffix_base = "__grad" if mode == "forward" else "__rgrad"
    suffix = suffix_base if len(fn.params) == 1 else f"{suffix_base}_{param_idx}"
    return A.FnDecl(
        span=fn.span,
        name=f"{fn.name}{suffix}",
        generics=[],
        params=new_params,
        return_ty=A.TyName(span=fn.span, name="f32"),
        where_clauses=[],
        body=new_body,
        attrs=["pure"],
        is_pub=fn.is_pub,
    )
