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

import copy
from dataclasses import fields, is_dataclass

from . import ast_nodes as A
from .ast_walker import ASTVisitor
from .autodiff import differentiate, _inline_lets
from .autodiff_reverse import differentiate_reverse


_GRAD_CALL_NAMES = frozenset({"grad", "grad_rev", "grad_rev_all"})
_SCALAR_GRAD_TYPES = frozenset({
    "f32", "f64",
})

# Stage 57 Inc 1 — Tier 2 #7 pytrees: module-level cache of struct
# decls in the current program. Set by `grad_pass()` at entry; read
# by `_generate_grad_rev_all_fn` to flatten struct-typed params.
# Threaded as global state to avoid signature changes across the
# recursive walker chain. Cleared at next grad_pass() entry.
_CURRENT_STRUCT_DECLS: dict = {}


def _ty_name(ty: A.TyNode | None) -> str | None:
    if isinstance(ty, A.TyName):
        return ty.name
    return None


def _with_float_literal_suffix(expr: A.Expr, suffix: str) -> A.Expr:
    """Return a copy where unsuffixed float literals carry `suffix`.

    Reverse AD can simplify a f64 derivative to an unsuffixed literal like
    `2.0`. Lowering defaults unsuffixed floats to f32, which is unsafe when the
    generated reflection writer is `modify_f64`.
    """
    cloned = copy.deepcopy(expr)

    def visit(node: object) -> None:
        if isinstance(node, A.FloatLit) and node.type_suffix is None:
            node.type_suffix = suffix
        if not is_dataclass(node):
            return
        for f in fields(node):
            value = getattr(node, f.name)
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, tuple):
                        for part in item:
                            visit(part)
                    else:
                        visit(item)
            elif isinstance(value, tuple):
                for item in value:
                    visit(item)
            else:
                visit(value)

    visit(cloned)
    return cloned


# Stage 28.8.2: grad_pass's `_expr_has_grad` predicate migrated to
# ASTVisitor. The pre-fix walker hand-rolled a 50-LoC isinstance
# cascade that audit cycle 2 C2-4 caught missing Field / Index /
# StructLit / TupleLit / ArrayLit / UnsafeBlock / Range / Break /
# Quote / Splice / Modify dispatch arms. The shared library
# introspects dataclass fields — adding a new Expr subtype no
# longer silently drops walker coverage.
#
# NOTE: only the read-only PREDICATE migrates here. The rewriter
# (`_rewrite_in_expr`) and the let-alias resolver (`_resolve_in_expr`)
# both return new nodes / mutate trees, which is a different shape
# from the ASTVisitor read-only walk contract. They keep their
# bespoke per-node dispatch because each node's rewrite semantics
# differ (some return new nodes, some mutate in place); a generic
# walker can't express that without giving up the dataclass-field
# introspection that makes ASTVisitor drift-proof. See
# docs/helix-pre-phase-A-finalization-research.md note in commit body.


class _GradCallFinder(ASTVisitor):
    """Stage 28.8.2: short-circuit visitor that flips `found` to True
    on the first encounter of a grad/grad_rev/grad_rev_all Call.

    Uses the skip-marker pattern: once `found` is set, every
    subsequent visit returns False to suppress further descent —
    saving work on large fn bodies.
    """

    def __init__(self) -> None:
        self.found = False

    def visit(self, node):
        if self.found:
            return False  # short-circuit; no more descent
        return super().visit(node)

    def visit_Call(self, node: A.Call):
        if isinstance(node.callee, A.Name) and node.callee.name in _GRAD_CALL_NAMES:
            self.found = True
            return False  # no need to descend further
        # Otherwise fall through to generic_visit (which the base
        # class's visit() will call automatically).


def _has_grad_call(prog: A.Program) -> bool:
    """Quick scan: does any function body contain a Call whose callee is
    Name("grad"|"grad_rev"|"grad_rev_all")? Used to short-circuit the
    expensive match_lower pre-pass when no AD work is needed.

    Stage 28.8.2: uses ``_GradCallFinder(ASTVisitor)``.
    """
    for item in prog.items:
        if isinstance(item, A.FnDecl):
            if _expr_has_grad(item.body):
                return True
    return False


def _expr_has_grad(e) -> bool:
    """Stage 28.8.2: predicate ``does this expression contain a
    grad/grad_rev/grad_rev_all call anywhere?``. Drop-in replacement
    for the pre-fix 100-LoC isinstance cascade (audit cycle 2 C2-4).
    """
    if e is None:
        return False
    finder = _GradCallFinder()
    finder.visit(e)
    return finder.found


def grad_pass(prog: A.Program) -> int:
    """Walk the program; rewrite all grad(f) calls into references to
    generated f__grad functions. Returns count of grad calls rewritten.

    Also resolves let-aliases: 'let f = grad(loss); f(x)' is rewritten so
    the call to f becomes a direct call to loss__grad."""
    # NOTE: match -> if/let lowering happens in lower() (the IR builder),
    # AFTER typecheck. _rewrite_in_expr now handles Match nodes natively
    # (recurse into arm bodies). This keeps the AST as Match-form during
    # typecheck so users get correct pattern-style diagnostics.
    # Stage 57 Inc 1 — Tier 2 #7 pytrees: build struct_decls dict at
    # pass entry. Used by `_generate_grad_rev_all_fn` to flatten struct-
    # typed params into their constituent leaves via
    # `pytree.flatten_pytree_param`. Threaded via module-level state to
    # avoid signature changes in the recursive walker chain.
    global _CURRENT_STRUCT_DECLS
    _CURRENT_STRUCT_DECLS = {
        it.name: it for it in prog.items
        if isinstance(it, A.StructDecl)
    }
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
    """Audit 28.8 cycle 2 C2-4: dispatch covers every Expr subtype that
    can contain Expr sub-trees. Pre-fix the dispatch missed Match / Loop /
    Field / Return / Break / Assign / Range / StructLit / TupleLit /
    ArrayLit / UnsafeBlock / Quote / Splice / Modify — so let-aliases
    nested inside any of those positions never got resolved (and the
    call to grad inside an aliased-fn slot through any of those
    positions silently never got rewritten)."""
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
    elif isinstance(expr, A.UnsafeBlock):
        _resolve_let_aliases(expr.body, fn_by_name, alias_env)
    elif isinstance(expr, A.If):
        _resolve_in_expr(expr.cond, fn_by_name, alias_env)
        _resolve_let_aliases(expr.then, fn_by_name, alias_env)
        if expr.else_ is not None and isinstance(expr.else_, A.Block):
            _resolve_let_aliases(expr.else_, fn_by_name, alias_env)
        elif expr.else_ is not None and isinstance(expr.else_, A.If):
            _resolve_in_expr(expr.else_, fn_by_name, alias_env)
    elif isinstance(expr, A.Match):
        _resolve_in_expr(expr.scrutinee, fn_by_name, alias_env)
        for arm in expr.arms:
            if arm.guard is not None:
                _resolve_in_expr(arm.guard, fn_by_name, alias_env)
            _resolve_in_expr(arm.body, fn_by_name, alias_env)
    elif isinstance(expr, A.Loop):
        _resolve_let_aliases(expr.body, fn_by_name, alias_env)
    elif isinstance(expr, A.Index):
        _resolve_in_expr(expr.callee, fn_by_name, alias_env)
        for i in expr.indices:
            _resolve_in_expr(i, fn_by_name, alias_env)
    elif isinstance(expr, A.Field):
        _resolve_in_expr(expr.obj, fn_by_name, alias_env)
    elif isinstance(expr, A.While):
        _resolve_in_expr(expr.cond, fn_by_name, alias_env)
        _resolve_let_aliases(expr.body, fn_by_name, alias_env)
    elif isinstance(expr, A.For):
        _resolve_in_expr(expr.iter_expr, fn_by_name, alias_env)
        _resolve_let_aliases(expr.body, fn_by_name, alias_env)
    elif isinstance(expr, A.Return):
        if expr.value is not None:
            _resolve_in_expr(expr.value, fn_by_name, alias_env)
    elif isinstance(expr, A.Break):
        if expr.value is not None:
            _resolve_in_expr(expr.value, fn_by_name, alias_env)
    elif isinstance(expr, A.Assign):
        _resolve_in_expr(expr.target, fn_by_name, alias_env)
        _resolve_in_expr(expr.value, fn_by_name, alias_env)
    elif isinstance(expr, A.Range):
        if getattr(expr, "start", None) is not None:
            _resolve_in_expr(expr.start, fn_by_name, alias_env)
        if getattr(expr, "end", None) is not None:
            _resolve_in_expr(expr.end, fn_by_name, alias_env)
    elif isinstance(expr, A.TupleLit):
        for x in expr.elems:
            _resolve_in_expr(x, fn_by_name, alias_env)
    elif isinstance(expr, A.ArrayLit):
        for x in expr.elems:
            _resolve_in_expr(x, fn_by_name, alias_env)
    elif isinstance(expr, A.StructLit):
        for _, v in expr.fields:
            _resolve_in_expr(v, fn_by_name, alias_env)
    elif isinstance(expr, (A.Quote, A.Splice)):
        _resolve_in_expr(expr.inner, fn_by_name, alias_env)
    elif isinstance(expr, A.Modify):
        for attr in ("target", "transformation", "verifier"):
            if hasattr(expr, attr):
                _resolve_in_expr(getattr(expr, attr), fn_by_name, alias_env)


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
            grad_fn = _generate_grad_rev_all_fn(
                target, fn_by_name,
                struct_decls=_CURRENT_STRUCT_DECLS)
            if grad_fn is not None:
                if grad_fn.name not in fn_by_name:
                    new_fns.append(grad_fn)
                    fn_by_name[grad_fn.name] = grad_fn
                    # Stage 62 Inc 1 — also register the per-leaf
                    # accessor fns generated alongside.
                    accessors = getattr(grad_fn, "_helix_accessor_fns", [])
                    for acc in accessors:
                        if acc.name not in fn_by_name:
                            new_fns.append(acc)
                            fn_by_name[acc.name] = acc
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
        # Audit 28.8 cycle 3 C3-1: handle chained `else if` (else_ is A.If),
        # not just `else { ... }` (else_ is A.Block). Mirror _resolve_in_expr.
        if expr.else_ is not None:
            if isinstance(expr.else_, A.Block):
                c_else = _rewrite_in_block(expr.else_, fn_by_name, new_fns)
            elif isinstance(expr.else_, A.If):
                new_else, c_else = _rewrite_in_expr(
                    expr.else_, fn_by_name, new_fns)
                expr.else_ = new_else
        return (expr, c_cond + c_then + c_else)
    if isinstance(expr, A.Match):
        # Recurse into the scrutinee + each arm body. This lets grad calls
        # inside match arms be rewritten without first having to desugar
        # the match. Pattern + guard expressions are not differentiable
        # so we don't recurse there.
        new_scrut, c_scrut = _rewrite_in_expr(expr.scrutinee, fn_by_name, new_fns)
        expr.scrutinee = new_scrut
        c_arms = 0
        for arm in expr.arms:
            new_body, ca = _rewrite_in_expr(arm.body, fn_by_name, new_fns)
            arm.body = new_body
            c_arms += ca
        return (expr, c_scrut + c_arms)
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
    # Audit 28.8 cycle 2 C2-4: cover the remaining Expr subtypes so
    # `grad(loss)` nested in any of these positions actually gets
    # rewritten. Pre-fix, the walker silently fell through to the
    # final `return (expr, count)` for Field / Index / StructLit /
    # TupleLit / ArrayLit / UnsafeBlock / Loop / Range / Return /
    # Break / Quote / Splice / Modify — so a user writing `[grad(f),
    # grad(g)]` (an ArrayLit) saw the `grad` symbol surface as an
    # unbound name at lowering time.
    if isinstance(expr, A.Loop):
        c = _rewrite_in_block(expr.body, fn_by_name, new_fns)
        return (expr, c)
    if isinstance(expr, A.UnsafeBlock):
        c = _rewrite_in_block(expr.body, fn_by_name, new_fns)
        return (expr, c)
    if isinstance(expr, A.Field):
        new_obj, c = _rewrite_in_expr(expr.obj, fn_by_name, new_fns)
        expr.obj = new_obj
        return (expr, c)
    if isinstance(expr, A.Return):
        if expr.value is not None:
            new_v, c = _rewrite_in_expr(expr.value, fn_by_name, new_fns)
            expr.value = new_v
            return (expr, c)
        return (expr, 0)
    if isinstance(expr, A.Break):
        if expr.value is not None:
            new_v, c = _rewrite_in_expr(expr.value, fn_by_name, new_fns)
            expr.value = new_v
            return (expr, c)
        return (expr, 0)
    if isinstance(expr, A.Range):
        c_total = 0
        if getattr(expr, "start", None) is not None:
            new_s, cs = _rewrite_in_expr(expr.start, fn_by_name, new_fns)
            expr.start = new_s
            c_total += cs
        if getattr(expr, "end", None) is not None:
            new_e, ce = _rewrite_in_expr(expr.end, fn_by_name, new_fns)
            expr.end = new_e
            c_total += ce
        return (expr, c_total)
    if isinstance(expr, A.TupleLit):
        c_total = 0
        for i, x in enumerate(expr.elems):
            new_x, cx = _rewrite_in_expr(x, fn_by_name, new_fns)
            expr.elems[i] = new_x
            c_total += cx
        return (expr, c_total)
    if isinstance(expr, A.ArrayLit):
        c_total = 0
        for i, x in enumerate(expr.elems):
            new_x, cx = _rewrite_in_expr(x, fn_by_name, new_fns)
            expr.elems[i] = new_x
            c_total += cx
        return (expr, c_total)
    if isinstance(expr, A.StructLit):
        c_total = 0
        for i, (name, v) in enumerate(expr.fields):
            new_v, cv = _rewrite_in_expr(v, fn_by_name, new_fns)
            expr.fields[i] = (name, new_v)
            c_total += cv
        return (expr, c_total)
    if isinstance(expr, (A.Quote, A.Splice)):
        new_inner, c = _rewrite_in_expr(expr.inner, fn_by_name, new_fns)
        expr.inner = new_inner
        return (expr, c)
    if isinstance(expr, A.Modify):
        c_total = 0
        for attr in ("target", "transformation", "verifier"):
            if hasattr(expr, attr):
                new_x, cx = _rewrite_in_expr(getattr(expr, attr),
                                              fn_by_name, new_fns)
                setattr(expr, attr, new_x)
                c_total += cx
        return (expr, c_total)
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


def _generate_grad_rev_all_fn(
    fn: A.FnDecl,
    fn_table: dict[str, A.FnDecl],
    struct_decls: dict | None = None,
) -> A.FnDecl | None:
    """Build `<fn.name>__rgrad_all` — a single function that computes all
    parameter gradients via reverse-mode AD in one source-level pass and
    writes each into a reflection cell.

    Generated signature: same params as `fn` plus a trailing `base: i32`
    cell-base index. Body computes each ∂f/∂leaf, then for each leaf emits
    `modify_f(base + i, g_i, __always_accept)` to store it.

    Stage 57 Inc 1 — Tier 2 #7 pytrees: when `struct_decls` is provided,
    each param is flattened to its constituent leaves via
    `flatten_pytree_param`. Scalar params produce 1 leaf with path =
    param name; struct params produce N leaves with paths like
    `"model.w1"`, `"model.w2"`. The `differentiate_reverse` engine
    already handles dotted field paths (per autodiff_reverse._field_path).

    Returns 0 on success. The caller reads the gradients back via
    splice_f(base + i), where i is the FLAT leaf index (not the
    original param index).
    """
    if not fn.params:
        return None
    _reject_unsupported_grad_signature(fn, "grad_rev_all", struct_decls)
    span = fn.span

    # Stage 57 Inc 1 — expand params into pytree leaves.
    # Each leaf is (path, ty_name). Scalar params produce one
    # leaf with path = param name; struct params produce N leaves
    # with paths like "model.w1".
    leaves: list[tuple[str, str]] = []
    if struct_decls is not None:
        # Cycle 1 Auditor 4 HIGH-4 fix: pre-fix, this used a blanket
        # `except Exception` that silently fell back to scalar-only
        # leaves when pytree flattening raised — for ANY exception
        # type, including KeyboardInterrupt-shaped errors. For a
        # struct param `my_struct` whose body references
        # `my_struct.field`, the scalar fallback would have set
        # `var_names = ["my_struct"]` which differentiate_reverse
        # then doesn't find in the body — every gradient comes back
        # as 0.0, silently. The audit-claimed "defensive — rejection
        # check should have caught this" is exactly wrong: the
        # rejection at line 600 runs FIRST and CAN miss the same
        # failure shape that generation hits here.
        #
        # Post-fix: narrow except to the specific exceptions
        # flatten_pytree_param documents (KeyError on missing struct
        # decl, ValueError on malformed pytree). Anything else
        # propagates so a real bug surfaces instead of producing
        # silently-zero gradients. AD-warn on the fallback so the
        # user knows the codegen path degraded.
        try:
            from .pytree import flatten_pytree_param
            for p in fn.params:
                p_leaves = flatten_pytree_param(p, struct_decls)
                for leaf in p_leaves:
                    leaves.append((leaf.path, leaf.ty_name))
        except (KeyError, ValueError, TypeError) as exc:
            # Documented-failure shapes from flatten_pytree_param.
            # Fall back to scalar-only and EMIT a warning so the
            # silently-zero gradient defect Auditor 4 flagged is
            # at least visible to the user.
            try:
                from .autodiff import _ad_warn
                _ad_warn(
                    fn,
                    f"pytree flattening failed ({type(exc).__name__}: "
                    f"{exc}); falling back to scalar-only param "
                    f"path — struct field gradients will be zero. "
                    f"Cycle 1 Auditor 4 HIGH-4 fix.",
                )
            except ImportError:
                pass
            leaves = [(p.name, _ty_name(p.ty) or "f32")
                      for p in fn.params]
    else:
        leaves = [(p.name, _ty_name(p.ty) or "f32")
                  for p in fn.params]
    var_names = [path for (path, _) in leaves]

    # Compute all gradients in one analysis pass.
    all_grads = differentiate_reverse(fn.body, var_names, fn_table=fn_table)

    # Build the body: for each param, let g_i = <gradient_expr>, then
    # modify_f(base + i, g_i, __always_accept).
    body_stmts: list[A.Stmt] = [
        A.Let(
            span=span, name="__status", is_mut=True,
            ty=A.TyName(span=span, name="i32"),
            value=A.IntLit(span=span, value=0),
        )
    ]
    base_name = "__base"
    # Stage 57 Inc 1: iterate over LEAVES (not fn.params). For struct
    # params, each field becomes a separate (path, ty_name) leaf and
    # writes to its own modify_f cell. Caller reads back via splice_f
    # with the flat leaf index. Original param index is no longer the
    # cell offset for struct params — caller must compute offsets via
    # the flatten_pytree_param order (alphabetical-by-field per the
    # pytree.py contract).
    for i, (p_name, ty_name) in enumerate(leaves):
        g_var = f"__g_{i}"
        ok_var = f"__ok_{i}"
        grad_expr = all_grads[p_name]
        if ty_name in ("f32", "f64"):
            grad_expr = _with_float_literal_suffix(grad_expr, ty_name)
        body_stmts.append(A.Let(
            span=span, name=g_var, ty=None,
            value=grad_expr, is_mut=False,
        ))
        # base + i
        idx_expr = (A.Name(span=span, name=base_name) if i == 0
                    else A.Binary(span=span, op="+",
                                   left=A.Name(span=span, name=base_name),
                                   right=A.IntLit(span=span, value=i)))
        modify_name = "modify_f64" if ty_name == "f64" else "modify_f"
        verifier_name = "__always_accept_f64" if ty_name == "f64" else "__always_accept"
        call = A.Call(
            span=span,
            callee=A.Name(span=span, name=modify_name),
            args=[idx_expr,
                  A.Name(span=span, name=g_var),
                  A.Name(span=span, name=verifier_name)],
        )
        body_stmts.append(A.Let(
            span=span, name=ok_var, ty=None, value=call, is_mut=False,
        ))
        body_stmts.append(A.ExprStmt(
            span=span,
            expr=A.If(
                span=span,
                cond=A.Binary(
                    span=span, op="==",
                    left=A.Name(span=span, name=ok_var),
                    right=A.IntLit(span=span, value=0),
                ),
                then=A.Block(
                    span=span,
                    stmts=[
                        A.ExprStmt(
                            span=span,
                            expr=A.Assign(
                                span=span,
                                target=A.Name(span=span, name="__status"),
                                op="=",
                                value=A.Binary(
                                    span=span, op="-",
                                    left=A.IntLit(span=span, value=0),
                                    right=A.IntLit(span=span, value=1),
                                ),
                            ),
                        )
                    ],
                    final_expr=None,
                ),
                else_=None,
            ),
        ))

    new_body = A.Block(
        span=span, stmts=body_stmts,
        final_expr=A.Name(span=span, name="__status"),
    )

    # Params: preserve original scalar types; do not silently narrow f64 to f32.
    # Stage 35 keeps f64 as f64 here instead of narrowing generated signatures.
    new_params = [
        A.FnParam(span=p.span, name=p.name,
                  ty=copy.deepcopy(p.ty))
        for p in fn.params
    ]
    new_params.append(A.FnParam(
        span=span, name=base_name,
        ty=A.TyName(span=span, name="i32"),
    ))

    rgrad_fn = A.FnDecl(
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
    # Stage 62 Inc 1 — named per-leaf gradient accessor fns
    # alongside the rgrad_all writer. Each accessor reads splice_f
    # (or splice_f64) for one leaf path; together they give users the
    # "pytree-shaped" gradient access pattern without requiring the
    # Phase-0 struct-return ABI rewrite (that's deferred to a future
    # stage).
    #
    # Generated names follow `{orig_fn}__grad_{path_sanitized}` where
    # path_sanitized replaces '.' with '_' (so "model.w1" -> "model_w1").
    # Signature: `(base: i32) -> <leaf_ty>`.
    accessor_fns = _generate_grad_leaf_accessors(
        fn, leaves, span,
    )
    # Return rgrad_fn first; callers that append in order will pick
    # up the accessors too. We return a list so caller can decide.
    rgrad_fn._helix_accessor_fns = accessor_fns  # attach for caller
    return rgrad_fn


def _generate_grad_leaf_accessors(
    fn: A.FnDecl,
    leaves: list[tuple[str, str]],
    span: A.Span,
) -> list[A.FnDecl]:
    """Stage 62 Inc 1 — generate named per-leaf gradient accessor fns.

    For each (path, ty_name) leaf, emit a fn:
        @pure
        fn {orig_fn}__grad_{sanitized_path}(base: i32) -> {ty_name} {
            splice_f(base + i)   // or splice_f64 for f64 leaves
        }

    The leaf index `i` matches the order used by grad_rev_all when
    writing modify_f(base + i, g_i, ...), so the accessors return the
    correct gradient for the named leaf.

    Sanitization rule: '.' -> '_' (so "model.inner.w1" -> "model_inner_w1").
    """
    out: list[A.FnDecl] = []
    for i, (path, ty_name) in enumerate(leaves):
        sanitized = path.replace(".", "_")
        accessor_name = f"{fn.name}__grad_{sanitized}"
        splice_name = "splice_f64" if ty_name == "f64" else "splice_f"
        # base + i
        idx_expr = (A.Name(span=span, name="base") if i == 0
                    else A.Binary(span=span, op="+",
                                   left=A.Name(span=span, name="base"),
                                   right=A.IntLit(span=span, value=i)))
        body = A.Block(
            span=span, stmts=[],
            final_expr=A.Call(
                span=span,
                callee=A.Name(span=span, name=splice_name),
                args=[idx_expr],
            ),
        )
        accessor = A.FnDecl(
            span=span,
            name=accessor_name,
            generics=[],
            params=[A.FnParam(
                span=span, name="base",
                ty=A.TyName(span=span, name="i32"),
            )],
            return_ty=A.TyName(
                span=span,
                name=ty_name if ty_name in ("f32", "f64") else "f32",
            ),
            where_clauses=[],
            body=body,
            attrs=["pure"],
            is_pub=fn.is_pub,
        )
        out.append(accessor)
    return out


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
    _reject_unsupported_grad_signature(
        fn, "grad_rev" if mode == "reverse" else "grad")
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

    # Build new params with original scalar types; f64 stays f64.
    new_params = [
        A.FnParam(span=p.span, name=p.name,
                  ty=copy.deepcopy(p.ty))
        for p in fn.params
    ]

    suffix_base = "__grad" if mode == "forward" else "__rgrad"
    suffix = suffix_base if len(fn.params) == 1 else f"{suffix_base}_{param_idx}"
    return A.FnDecl(
        span=fn.span,
        name=f"{fn.name}{suffix}",
        generics=[],
        params=new_params,
        return_ty=copy.deepcopy(fn.return_ty),
        where_clauses=[],
        body=new_body,
        attrs=["pure"],
        is_pub=fn.is_pub,
    )


def _reject_unsupported_grad_signature(
    fn: A.FnDecl, surface: str,
    struct_decls: dict | None = None,
) -> None:
    """Fail closed on aggregate parameters that can't be pytree-flattened.

    Stage 57 Inc 1 — Tier 2 #7 pytrees: relaxed from "all params must
    be scalar f32/f64" to "all params must be either scalar OR
    pytree-flattenable into all-scalar leaves". When `struct_decls`
    is provided, struct-typed params are flatten-attempted via
    pytree.flatten_pytree_param. If flattening succeeds and all
    leaves are f32/f64, the param is allowed.
    """
    unsupported: list[str] = []
    for p in fn.params:
        ty = p.ty
        if isinstance(ty, A.TyName) and ty.name in _SCALAR_GRAD_TYPES:
            continue
        # Stage 57 Inc 1: try pytree flattening for struct params.
        # Cycle 1 Auditor 4 HIGH-5 fix: pre-fix used `except
        # (ValueError, Exception)` which is identical to `except
        # Exception` and catches everything including BaseException
        # subclasses we shouldn't swallow. Mirrors the HIGH-4 fix in
        # _generate_grad_rev_all_fn: narrow to documented exception
        # shapes so a real bug surfaces. The two functions must
        # accept/reject the SAME shapes or the rejection check can
        # accept a param that generation then silently zeros — the
        # asymmetric-except bug Auditor 4 specifically named.
        if struct_decls is not None:
            try:
                from .pytree import flatten_pytree_param
                leaves = flatten_pytree_param(p, struct_decls)
                # Accept if all leaves are diff-eligible scalars.
                if all(leaf.ty_name in _SCALAR_GRAD_TYPES
                       for leaf in leaves):
                    continue
            except (KeyError, ValueError, TypeError):
                # Same narrowed except as the matched generation
                # path; falls through to rejection below.
                pass
        unsupported.append(p.name)
    if unsupported:
        joined = ", ".join(unsupported)
        raise NotImplementedError(
            f"{surface} aggregate/non-floating parameter(s) unsupported; "
            f"supports only f32/f64 scalar or pytree-flattenable struct "
            f"parameter(s) today; "
            f"unsupported parameter(s): {joined}; flatten pytree leaves or "
            "cast non-floating inputs before grad_pass"
        )
    ret_name = _ty_name(fn.return_ty)
    if ret_name not in _SCALAR_GRAD_TYPES:
        raise NotImplementedError(
            f"{surface} non-floating return type unsupported; "
            "supports only f32/f64 scalar loss outputs today; "
            f"got {ret_name or 'unknown'}"
        )


def _reject_unsupported_grad_params(fn: A.FnDecl, surface: str) -> None:
    _reject_unsupported_grad_signature(fn, surface)
