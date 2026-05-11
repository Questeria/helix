"""
helixc/frontend/match_lower.py — desugar `match` into nested `if` + `let`.

Runs as an AST → AST rewrite pass before lowering to IR.  After this pass,
no `Match`/`MatchArm`/`PatRange`/`PatOr` nodes remain in the tree, so the
rest of the pipeline (typecheck second pass, autodiff, IR lowering) can
remain match-agnostic.

Lowering scheme for `match scrut { p1 if g1 => e1, ..., pn => en }`:

    {
        let __scrut_N = scrut;
        if test(__scrut_N, p1) && (g1)? { let bindings; e1 }
        else if test(__scrut_N, p2) && (g2)? { let bindings; e2 }
        ...
        else { en }
    }

Pattern tests:
  PatWildcard          → true
  PatBind(name)        → true (and binds `name = __scrut`)
  PatLit(v)            → __scrut == v
  PatRange(lo, hi, =)  → __scrut >= lo && __scrut <= hi
  PatRange(lo, hi)     → __scrut >= lo && __scrut <  hi
  PatOr(a|b|c)         → test(a) || test(b) || test(c)
  PatTuple(...)        → conjunction of element tests + binds

License: Apache 2.0
"""

from __future__ import annotations
from typing import Optional

from . import ast_nodes as A


def lower_matches(prog: A.Program) -> A.Program:
    """Rewrite all `Match` nodes in the program in place; return the same
    Program (with mutated bodies).

    Stage 28.8.1: resets ``_FRESH_COUNTER`` at entry so successive calls
    on the same source produce identical synthetic names (``__scrut_1``,
    ``__scrut_2``, ...). Without the reset, generated names propagated
    test-suite-order pollution into IR + register hints, which the
    cycle-11 silent-failures audit traced as a contributing source of
    codegen non-determinism (see
    docs/helix-pre-phase-A-finalization-research.md § A3 / C1 source 2).
    """
    # Stage 28.8.1: per-call reset of module-level mutable state. Per-program
    # determinism: two calls to lower_matches() on the same prog must
    # generate identical fresh names.
    _FRESH_COUNTER[0] = 0
    for item in prog.items:
        if isinstance(item, A.FnDecl):
            _rewrite_block(item.body)
    return prog


# Walker: replaces Match expressions with desugared If chains.
#
# Stage 28.8.1: ``_FRESH_COUNTER`` is module-level mutable state. It is
# reset at the top of ``lower_matches(prog)`` so name generation is
# deterministic for a given program. We keep it module-level (rather
# than threading through every helper) because every fresh-name site
# is reachable only via ``lower_matches``; the reset entry point is
# the single externally-visible function.
_FRESH_COUNTER = [0]


def _fresh_name(prefix: str = "__scrut") -> str:
    _FRESH_COUNTER[0] += 1
    return f"{prefix}_{_FRESH_COUNTER[0]}"


def _rewrite_block(block: A.Block) -> None:
    if block is None:
        return
    new_stmts: list[A.Stmt] = []
    for stmt in block.stmts:
        new_stmts.append(_rewrite_stmt(stmt))
    block.stmts = new_stmts
    if block.final_expr is not None:
        block.final_expr = _rewrite_expr(block.final_expr)


def _rewrite_stmt(stmt: A.Stmt) -> A.Stmt:
    if isinstance(stmt, A.Let) and stmt.value is not None:
        stmt.value = _rewrite_expr(stmt.value)
    elif isinstance(stmt, A.ConstStmt) and stmt.value is not None:
        stmt.value = _rewrite_expr(stmt.value)
    elif isinstance(stmt, A.ExprStmt):
        stmt.expr = _rewrite_expr(stmt.expr)
    return stmt


def _rewrite_expr(expr: A.Expr) -> A.Expr:
    """Recursively rewrite Match nodes inside `expr`.  Returns a (possibly
    new) expression."""
    if expr is None:
        return expr
    if isinstance(expr, A.Match):
        # First recursively rewrite any nested matches in arms / scrutinee.
        expr.scrutinee = _rewrite_expr(expr.scrutinee)
        for arm in expr.arms:
            if arm.guard is not None:
                arm.guard = _rewrite_expr(arm.guard)
            arm.body = _rewrite_expr(arm.body)
        return _desugar_match(expr)
    if isinstance(expr, A.Block):
        _rewrite_block(expr)
        return expr
    if isinstance(expr, A.If):
        expr.cond = _rewrite_expr(expr.cond)
        _rewrite_block(expr.then)
        if expr.else_ is not None:
            if isinstance(expr.else_, A.Block):
                _rewrite_block(expr.else_)
            else:
                expr.else_ = _rewrite_expr(expr.else_)
        return expr
    if isinstance(expr, A.Binary):
        expr.left = _rewrite_expr(expr.left)
        expr.right = _rewrite_expr(expr.right)
        return expr
    if isinstance(expr, A.Unary):
        expr.operand = _rewrite_expr(expr.operand)
        return expr
    if isinstance(expr, A.Call):
        expr.callee = _rewrite_expr(expr.callee)
        expr.args = [_rewrite_expr(a) for a in expr.args]
        return expr
    if isinstance(expr, A.For):
        expr.iter_expr = _rewrite_expr(expr.iter_expr)
        _rewrite_block(expr.body)
        return expr
    if isinstance(expr, A.While):
        expr.cond = _rewrite_expr(expr.cond)
        _rewrite_block(expr.body)
        return expr
    if isinstance(expr, A.Loop):
        _rewrite_block(expr.body)
        return expr
    if isinstance(expr, A.Cast):
        expr.value = _rewrite_expr(expr.value)
        return expr
    if isinstance(expr, A.Assign):
        expr.value = _rewrite_expr(expr.value)
        return expr
    if isinstance(expr, A.TupleLit):
        expr.elems = [_rewrite_expr(e) for e in expr.elems]
        return expr
    if isinstance(expr, A.ArrayLit):
        expr.elems = [_rewrite_expr(e) for e in expr.elems]
        return expr
    if isinstance(expr, A.Index):
        expr.callee = _rewrite_expr(expr.callee)
        expr.indices = [_rewrite_expr(i) for i in expr.indices]
        return expr
    if isinstance(expr, A.Return) and expr.value is not None:
        expr.value = _rewrite_expr(expr.value)
        return expr
    if isinstance(expr, A.StructLit):
        expr.fields = [(name, _rewrite_expr(value))
                       for name, value in expr.fields]
        return expr
    if isinstance(expr, A.Field):
        expr.obj = _rewrite_expr(expr.obj)
        return expr
    return expr


def _desugar_match(m: A.Match) -> A.Expr:
    """Build:  { let __scrut = scrutinee; if-chain }."""
    span = m.span
    scrut_name = _fresh_name()
    scrut_let = A.Let(
        span=span, name=scrut_name, is_mut=False, ty=None, value=m.scrutinee,
    )
    chain = _build_chain(scrut_name, m.arms, span)
    return A.Block(span=span, stmts=[scrut_let], final_expr=chain)


def _build_chain(scrut: str, arms: list[A.MatchArm], span: A.Span) -> A.Expr:
    """Recursively build the if/else chain from a list of arms.  The last
    arm becomes the bare else (whether it tests true unconditionally or
    not — typecheck.exhaustiveness already rejected non-exhaustive matches
    on bool/unit; for other types, a final wildcard arm is required, so it
    is safe to use as a bare else)."""
    if not arms:
        # Empty match: produce a unit literal `()`.  Should not happen in
        # practice — typecheck rejects empty matches on non-unit scrutinees.
        return A.TupleLit(span=span, elems=[])
    arm = arms[0]
    rest = arms[1:]
    # Build the arm body with binders prepended as let-statements.
    # IMPORTANT: deepcopy the binds for guard vs body so the two `Let`
    # nodes don't share state — downstream passes mutate AST nodes in
    # place, and a shared `Let` whose `.value` is rewritten in the body
    # would also corrupt the guard.
    import copy as _copy
    body_binds = _collect_binds(arm.pattern, scrut, span)
    body_block = _wrap_body_with_binds(arm.body, body_binds, span)

    # Build the test: pattern-test && guard?
    pat_test = _pattern_test(arm.pattern, scrut, span)
    if arm.guard is not None:
        # Fresh binds (deep-copied) for the guard scope.
        guard_binds = [_copy.deepcopy(b) for b in body_binds]
        guard_block = _wrap_body_with_binds(arm.guard, guard_binds, span)
        cond = _and(pat_test, guard_block, span)
    else:
        cond = pat_test

    # If this is the last arm AND the test is trivially true (wildcard /
    # bare bind without guard) we collapse to just the body — eliminates
    # a useless `if true { x }` wrapper.
    if not rest and arm.guard is None and _is_total_pattern(arm.pattern):
        return body_block

    # Otherwise: if cond { body } else { rest }
    else_branch = _build_chain(scrut, rest, span)
    if not isinstance(else_branch, (A.Block, A.If)):
        else_branch = A.Block(span=span, stmts=[], final_expr=else_branch)
    if not isinstance(body_block, A.Block):
        body_block = A.Block(span=span, stmts=[], final_expr=body_block)
    return A.If(span=span, cond=cond, then=body_block, else_=else_branch)


def _is_total_pattern(pat: A.Pattern) -> bool:
    if isinstance(pat, (A.PatWildcard, A.PatBind)):
        return True
    if isinstance(pat, A.PatTuple):
        return all(_is_total_pattern(p) for p in pat.elems)
    return False


def _pattern_test(pat: A.Pattern, scrut: str, span: A.Span) -> A.Expr:
    """Build a boolean expression that's true iff `pat` matches scrut."""
    if isinstance(pat, (A.PatWildcard, A.PatBind)):
        return A.BoolLit(span=span, value=True)
    # For both PatLit and PatRange, the test reads "the tag" of __scrut.
    # We always go through Index(__scrut, 0) so the same code works for
    # both array-backed (enum/struct) and scalar scrutinees — the
    # lowerer's Index handler falls back to returning the scalar value
    # directly when the binding is scalar.
    def _scrut_tag() -> A.Expr:
        return A.Index(
            span=span,
            callee=A.Name(span=span, name=scrut),
            indices=[A.IntLit(span=span, value=0)],
        )
    if isinstance(pat, A.PatLit):
        return A.Binary(span=span, op="==",
                        left=_scrut_tag(), right=pat.value)
    if isinstance(pat, A.PatRange):
        op_hi = "<=" if pat.inclusive else "<"
        return A.Binary(
            span=span, op="&&",
            left=A.Binary(span=span, op=">=",
                          left=_scrut_tag(), right=pat.lo),
            right=A.Binary(span=span, op=op_hi,
                           left=_scrut_tag(), right=pat.hi),
        )
    if isinstance(pat, A.PatOr):
        # alts || alts || ...
        tests = [_pattern_test(a, scrut, span) for a in pat.alts]
        return _or_chain(tests, span)
    if isinstance(pat, A.PatTuple):
        # Tuple-element tests using the scrut's `.N` field access.
        # The scrut is a tuple binding (array-backed), so scrut[i] reads
        # the i-th slot. Each sub-pattern is recursively tested against
        # its slot; PatBind / PatWildcard are trivially true (binders
        # handled in _collect_binds).
        sub_tests: list[A.Expr] = []
        for i, sub in enumerate(pat.elems):
            slot_load = A.Index(
                span=span,
                callee=A.Name(span=span, name=scrut),
                indices=[A.IntLit(span=span, value=i)],
            )
            if isinstance(sub, A.PatLit):
                sub_tests.append(A.Binary(span=span, op="==",
                                          left=slot_load, right=sub.value))
            elif isinstance(sub, A.PatRange):
                op_hi = "<=" if sub.inclusive else "<"
                sub_tests.append(A.Binary(
                    span=span, op="&&",
                    left=A.Binary(span=span, op=">=",
                                  left=slot_load, right=sub.lo),
                    right=A.Binary(span=span, op=op_hi,
                                   left=slot_load, right=sub.hi),
                ))
            # PatBind / PatWildcard / nested PatTuple → trivially true
            # (binders handled in _collect_binds; nested tuples currently
            # restricted to wildcard-class subpats).
        if not sub_tests:
            return A.BoolLit(span=span, value=True)
        full = sub_tests[0]
        for t in sub_tests[1:]:
            full = A.Binary(span=span, op="&&", left=full, right=t)
        return full
    if isinstance(pat, A.PatVariant):
        # Test: __scrut[0] == tag_value AND each sub-pattern test against
        # __scrut[1], __scrut[2], ...  The scrut is expected to be a
        # tagged-value array: [tag, payload0, payload1, ...].
        scrut_name = A.Name(span=span, name=scrut)
        tag_load = A.Index(span=span, callee=scrut_name,
                           indices=[A.IntLit(span=span, value=0)])
        tag_test = A.Binary(span=span, op="==",
                            left=tag_load, right=pat.path)
        sub_tests: list[A.Expr] = []
        for i, sub in enumerate(pat.sub_patterns):
            slot_idx = i + 1
            slot_load = A.Index(
                span=span,
                callee=A.Name(span=span, name=scrut),
                indices=[A.IntLit(span=span, value=slot_idx)],
            )
            if isinstance(sub, A.PatLit):
                sub_tests.append(A.Binary(span=span, op="==",
                                          left=slot_load, right=sub.value))
            elif isinstance(sub, A.PatRange):
                op_hi = "<=" if sub.inclusive else "<"
                sub_tests.append(A.Binary(
                    span=span, op="&&",
                    left=A.Binary(span=span, op=">=",
                                  left=slot_load, right=sub.lo),
                    right=A.Binary(span=span, op=op_hi,
                                   left=slot_load, right=sub.hi),
                ))
            elif isinstance(sub, A.PatVariant):
                # Nested variant: the sub-slot is itself a tagged value
                # (typically arena index of a recursive enum). Compare
                # the sub-slot against its variant's tag — an
                # approximation that covers the common AST-walking case
                # but not full recursive sub-pattern dispatch. Deeper
                # binders inside the nested variant are still extracted
                # by _collect_binds.
                sub_tests.append(A.Binary(span=span, op="==",
                                          left=slot_load, right=sub.path))
            # PatBind / PatWildcard / nested PatTuple: trivially true;
            # binder handled in _collect_binds.
        if sub_tests:
            full = tag_test
            for t in sub_tests:
                full = A.Binary(span=span, op="&&", left=full, right=t)
            return full
        return tag_test
    return A.BoolLit(span=span, value=True)


def _collect_binds(pat: A.Pattern, scrut: str, span: A.Span) -> list[A.Let]:
    """Return a list of Let-stmts that bind any names introduced by `pat`."""
    binds: list[A.Let] = []
    if isinstance(pat, A.PatBind):
        binds.append(A.Let(
            span=span, name=pat.name, is_mut=False, ty=None,
            value=A.Name(span=span, name=scrut),
        ))
    elif isinstance(pat, A.PatVariant):
        # For each sub-pattern, emit a let-binding for the slot value.
        # PatBind: `let name = scrut[i+1]`.
        # Nested PatVariant/PatTuple: recurse via a synthetic temp so
        # inner PatBinds (e.g. `Cons(Some(x), tail)`) still get bound.
        for i, sub in enumerate(pat.sub_patterns):
            slot_idx = i + 1
            slot_load = A.Index(
                span=span,
                callee=A.Name(span=span, name=scrut),
                indices=[A.IntLit(span=span, value=slot_idx)],
            )
            if isinstance(sub, A.PatBind):
                binds.append(A.Let(
                    span=span, name=sub.name, is_mut=False, ty=None,
                    value=slot_load,
                ))
            elif isinstance(sub, (A.PatVariant, A.PatTuple)):
                tmp = _fresh_name(prefix="__sub")
                binds.append(A.Let(
                    span=span, name=tmp, is_mut=False, ty=None,
                    value=slot_load,
                ))
                binds.extend(_collect_binds(sub, tmp, span))
    elif isinstance(pat, A.PatTuple):
        # Symmetric to PatVariant but slot indices start at 0 (no tag).
        for i, sub in enumerate(pat.elems):
            slot_load = A.Index(
                span=span,
                callee=A.Name(span=span, name=scrut),
                indices=[A.IntLit(span=span, value=i)],
            )
            if isinstance(sub, A.PatBind):
                binds.append(A.Let(
                    span=span, name=sub.name, is_mut=False, ty=None,
                    value=slot_load,
                ))
            elif isinstance(sub, (A.PatVariant, A.PatTuple)):
                tmp = _fresh_name(prefix="__sub")
                binds.append(A.Let(
                    span=span, name=tmp, is_mut=False, ty=None,
                    value=slot_load,
                ))
                binds.extend(_collect_binds(sub, tmp, span))
    # PatTuple element-binds are deferred (see _pattern_test note).
    return binds


def _wrap_body_with_binds(body: A.Expr, binds: list[A.Let],
                          span: A.Span) -> A.Expr:
    if not binds:
        return body
    if isinstance(body, A.Block):
        body.stmts = list(binds) + body.stmts
        return body
    return A.Block(span=span, stmts=list(binds), final_expr=body)


def _and(left: A.Expr, right: A.Expr, span: A.Span) -> A.Expr:
    return A.Binary(span=span, op="&&", left=left, right=right)


def _or_chain(tests: list[A.Expr], span: A.Span) -> A.Expr:
    if not tests:
        return A.BoolLit(span=span, value=False)
    out = tests[0]
    for t in tests[1:]:
        out = A.Binary(span=span, op="||", left=out, right=t)
    return out
