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
        # Stage 28.9 cycle 4 C4-1 fix (conf 88): Assign.target is also
        # an Expr (e.g. `arr[match x { ... }] = v`). The prior arm
        # traversed only `value`, so a Match nested in the lvalue
        # escaped desugaring and tripped lower_ast's "Match should not
        # reach _lower_expr" assertion. Same defect class as C22-C
        # (UnsafeBlock/Range/Modify gap).
        expr.target = _rewrite_expr(expr.target)
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
    # Audit 28.8 cycle 23 C22-C (HIGH): same defect class as cycle-2
    # C2-4 — hand-rolled walker missing dispatch arms for AST subtypes
    # that hold Expr children. Pre-fix, `unsafe { match x { ... } }`,
    # `for i in 0..match n { ... }` (Range.end), and
    # `modify(x, match x { ... }, ok)` all let the inner Match persist
    # past lower_matches and crashed lower_ast's "Match should not
    # reach _lower_expr" assertion.
    if isinstance(expr, A.UnsafeBlock):
        _rewrite_block(expr.body)
        return expr
    if isinstance(expr, A.Range):
        if expr.start is not None:
            expr.start = _rewrite_expr(expr.start)
        if expr.end is not None:
            expr.end = _rewrite_expr(expr.end)
        return expr
    if isinstance(expr, A.Modify):
        expr.target = _rewrite_expr(expr.target)
        expr.transformation = _rewrite_expr(expr.transformation)
        expr.verifier = _rewrite_expr(expr.verifier)
        return expr
    # Defense-in-depth: Break.value, Quote.inner, Splice.inner are
    # latent — flagged by cycle 23 audit C as same-class. Cover here
    # to prevent regression.
    if isinstance(expr, A.Break):
        if expr.value is not None:
            expr.value = _rewrite_expr(expr.value)
        return expr
    if isinstance(expr, A.Quote):
        expr.inner = _rewrite_expr(expr.inner)
        return expr
    if isinstance(expr, A.Splice):
        expr.inner = _rewrite_expr(expr.inner)
        return expr
    if isinstance(expr, A.TileLit):
        # Stage 28.9 cycle 7 C7-1 (conf 82): TileLit holds shape (list
        # of Expr) and memspace (Expr). A Match nested in tile shape or
        # memspace position would survive past lower_matches and trip
        # lower_ast's Match-assertion. Same defect class as C22-C
        # (UnsafeBlock/Range/Modify) and Stage-28.8 cycle-6 F4
        # (autodiff._inline_lets). Phase-0 lower_ast._tile_shape_dims
        # gates shape elements to IntLit only, but defensively descend
        # here so the loud diagnostic comes from the gate, not the
        # assertion deeper in the pipeline.
        expr.shape = [_rewrite_expr(s) for s in expr.shape]
        expr.memspace = _rewrite_expr(expr.memspace)
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


def _fresh_slot_load(callee_expr: A.Expr, idx: int, span: A.Span) -> A.Expr:
    """Stage 28.9 cycle 13 audit-A C13-2 fix (conf 80): build a FRESH
    `Index(callee, [IntLit(idx)])` per call site. Pre-fix multiple
    sub-tests shared the same `slot_load` instance, violating tree
    linearity that pytree.py and validation-pass walkers assume."""
    return A.Index(
        span=span,
        callee=_dup_expr(callee_expr),
        indices=[A.IntLit(span=span, value=idx)],
    )


def _dup_expr(expr: A.Expr) -> A.Expr:
    """Stage 28.9 cycle 13 C13-2 helper: build a fresh AST node for
    common Expr shapes. Used to avoid sharing Index/Name children
    across multiple parent nodes (which would violate tree linearity).

    Cycle 14 C14-1 fix (audit-A conf 84): preserve the SOURCE span
    from `expr.span` rather than caller context. Pre-fix the
    explicit Name/IntLit/Index branches overwrote spans with caller
    context while the deepcopy fallback preserved them — silent
    inconsistency that pointed diagnostics at the wrong source line.

    Cycle 15 C15-2 fix (conf 78): the legacy `span` parameter is
    removed entirely (was silently ignored after C14-1). Callers
    updated to no longer pass it.

    Cycle 14 C14-A (codereview C14-1, audit-A C14-2): A.Path branch
    added so PatVariant.path tag tests do not silently alias across
    arms' Binary subtrees."""
    if isinstance(expr, A.Name):
        return A.Name(span=expr.span, name=expr.name,
                      generics=list(expr.generics))
    if isinstance(expr, A.IntLit):
        return A.IntLit(span=expr.span, value=expr.value,
                        type_suffix=expr.type_suffix)
    if isinstance(expr, A.Index):
        return A.Index(
            span=expr.span,
            callee=_dup_expr(expr.callee),
            indices=[_dup_expr(i) for i in expr.indices],
        )
    if isinstance(expr, A.Path):
        return A.Path(span=expr.span, segments=list(expr.segments))
    # For Field / and anything else, fall back to deepcopy (preserves spans).
    import copy
    return copy.deepcopy(expr)


def _pattern_test_expr(pat: A.Pattern, scrut_expr: A.Expr,
                        span: A.Span) -> A.Expr:
    """Stage 28.9 cycle 13 audit-A C13-1 + C13-2 structural fix
    (closes the cycle-10..cycle-12 family of nested-pattern bugs at the
    root). Canonical pattern-test implementation that takes an Expr
    scrut (not a str name). Used by:

      - `_pattern_test(pat, scrut: str, span)`: top-level wrapper that
        calls this with `Name(scrut)`.
      - sub-position dispatch in PatTuple/PatVariant arms: calls this
        directly with the freshly-built `slot_load` expr.

    Closes the cycle-12 C12-1 asymmetric-fix gap because nested
    PatTuple, PatVariant, PatOr (etc.) at any depth now route through
    this unified dispatch — no inline approximations.
    """
    if isinstance(pat, (A.PatWildcard, A.PatBind)):
        return A.BoolLit(span=span, value=True)
    if isinstance(pat, A.PatLit):
        # PatLit against a scalar/tag: scrut == lit.value.
        # Cycle 14 C14-2 (conf 82): dup BOTH operands — pat.value is
        # an AST node that would otherwise be shared if the same lit
        # appears in multiple arms (e.g. two literal-equality tests of
        # the same numeric constant).
        return A.Binary(span=span, op="==",
                        left=_dup_expr(scrut_expr),
                        right=_dup_expr(pat.value))
    if isinstance(pat, A.PatRange):
        # Cycle 14 C14-2: dup pat.lo and pat.hi as well as scrut_expr.
        op_hi = "<=" if pat.inclusive else "<"
        return A.Binary(
            span=span, op="&&",
            left=A.Binary(span=span, op=">=",
                          left=_dup_expr(scrut_expr),
                          right=_dup_expr(pat.lo)),
            right=A.Binary(span=span, op=op_hi,
                           left=_dup_expr(scrut_expr),
                           right=_dup_expr(pat.hi)),
        )
    if isinstance(pat, A.PatOr):
        tests = [_pattern_test_expr(a, scrut_expr, span) for a in pat.alts]
        return _or_chain(tests, span)
    if isinstance(pat, A.PatTuple):
        sub_tests: list[A.Expr] = []
        for i, sub in enumerate(pat.elems):
            slot_load = _fresh_slot_load(scrut_expr, i, span)
            t = _pattern_test_expr(sub, slot_load, span)
            # Drop trivially-true sub-tests so they don't bloat the AST.
            if isinstance(t, A.BoolLit) and t.value is True:
                continue
            sub_tests.append(t)
        if not sub_tests:
            return A.BoolLit(span=span, value=True)
        full = sub_tests[0]
        for t in sub_tests[1:]:
            full = A.Binary(span=span, op="&&", left=full, right=t)
        return full
    if isinstance(pat, A.PatVariant):
        # Variant: scrut[0] == path AND each sub-pattern test against
        # scrut[i+1] (sub_patterns indexed from 1; slot 0 is the tag).
        # Cycle 14 C14-2 (audit-A) + C14-1 (audit-C, both conf 82+82):
        # dup pat.path so two arms with identical PatVariant don't
        # alias the same Path node into both arms' Binary subtrees.
        tag_load = _fresh_slot_load(scrut_expr, 0, span)
        tag_test = A.Binary(span=span, op="==",
                            left=tag_load,
                            right=_dup_expr(pat.path))
        sub_tests: list[A.Expr] = []
        for i, sub in enumerate(pat.sub_patterns):
            slot_load = _fresh_slot_load(scrut_expr, i + 1, span)
            t = _pattern_test_expr(sub, slot_load, span)
            if isinstance(t, A.BoolLit) and t.value is True:
                continue
            sub_tests.append(t)
        if not sub_tests:
            return tag_test
        full = tag_test
        for t in sub_tests:
            full = A.Binary(span=span, op="&&", left=full, right=t)
        return full
    # Cycle 14 C14-3 (conf 78): the prior `return A.BoolLit(True)`
    # catchall silently accepted any future Pattern subclass — exactly
    # the silent-accept anti-pattern the cycle-13 refactor was supposed
    # to eliminate. Now raise loudly so a new Pattern type forces an
    # explicit dispatch decision.
    # Cycle 15 C15-3 (conf 76) follow-on: include the offending span
    # in the error message so a downstream catch (e.g. check.py's
    # internal-error handler) can render a clean diagnostic with
    # location instead of a bare Python traceback.
    span_str = f"{pat.span.line}:{pat.span.col}" if getattr(pat, "span", None) else "?"
    raise NotImplementedError(
        f"_pattern_test_expr at {span_str}: unhandled Pattern subclass "
        f"{type(pat).__name__}; add an explicit arm in match_lower.py "
        f"to declare its match semantics. (helixc internal bug — please "
        f"file an issue.)"
    )


def _pattern_test(pat: A.Pattern, scrut: str, span: A.Span) -> A.Expr:
    """Build a boolean expression that's true iff `pat` matches scrut.
    Stage 28.9 cycle 13: thin wrapper over `_pattern_test_expr`."""
    scrut_expr = A.Name(span=span, name=scrut, generics=[])
    return _pattern_test_expr(pat, scrut_expr, span)


# Stage 28.9 cycle 13: legacy _pattern_test body removed. The canonical
# implementation is now `_pattern_test_expr(pat, scrut_expr, span)` above,
# with `_pattern_test(pat, scrut: str, span)` as a thin wrapper. This
# structural change closes the cycle-10..cycle-12 family of nested-pattern
# bugs by routing all sub-patterns through one unified recursive dispatch.


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
            elif isinstance(sub, (A.PatVariant, A.PatTuple, A.PatOr)):
                # Stage 28.9 cycle 11 audit-C C11-1 (conf 87): PatOr
                # added to the recurse tuple. A nested PatOr inside a
                # PatVariant/PatTuple sub-position (e.g.
                # `Cons((A | B(x)), tail)`) was previously silently
                # dropped — the cycle-10 PatOr top-level fix only
                # covered direct PatOr, not nested. Routing through the
                # same temp-bind + recurse path closes the gap.
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
            elif isinstance(sub, (A.PatVariant, A.PatTuple, A.PatOr)):
                # Stage 28.9 cycle 11 audit-C C11-1 (conf 87): PatOr
                # added to the recurse tuple. A nested PatOr inside a
                # PatVariant/PatTuple sub-position (e.g.
                # `Cons((A | B(x)), tail)`) was previously silently
                # dropped — the cycle-10 PatOr top-level fix only
                # covered direct PatOr, not nested. Routing through the
                # same temp-bind + recurse path closes the gap.
                tmp = _fresh_name(prefix="__sub")
                binds.append(A.Let(
                    span=span, name=tmp, is_mut=False, ty=None,
                    value=slot_load,
                ))
                binds.extend(_collect_binds(sub, tmp, span))
    elif isinstance(pat, A.PatOr):
        # Stage 28.9 cycle 10 audit-C C10-1 fix (conf 82): typecheck
        # legitimately permits or-patterns whose alternatives bind the
        # SAME name (intersection of alt binder sets — see
        # typecheck.py:1877-1896). _collect_binds previously returned
        # [] for any PatOr regardless, so the body's Name lookup
        # found nothing in scope.
        #
        # Emit binders for names present in EVERY alternative — same
        # intersection typecheck computed. Use the first alt's
        # binding source (every alt has the same scrut at this depth,
        # so any alt's slot-load is correct).
        if pat.alts:
            first_binds = _collect_binds(pat.alts[0], scrut, span)
            first_names = {b.name for b in first_binds}
            for alt in pat.alts[1:]:
                alt_names = {b.name for b in _collect_binds(alt, scrut, span)}
                first_names &= alt_names
            binds.extend(b for b in first_binds if b.name in first_names)
    elif isinstance(pat, (A.PatWildcard, A.PatLit, A.PatRange)):
        # Leaf patterns that introduce no binders. Explicit branches
        # for clarity + to make the dispatch exhaustive — cycle 15
        # C15-1 (conf 80) flagged the silent `[]` fall-through as a
        # symmetric defect to C14-3's _pattern_test_expr catchall.
        pass
    else:
        # Cycle 15 C15-1 (conf 80): loud failure for unknown Pattern
        # subclass. Symmetric to _pattern_test_expr's NotImplementedError
        # so future Pattern types must declare their binder semantics
        # explicitly. Pre-fix this branch silently returned [] — same
        # silent-accept class the cycle-10..14 family targets.
        span_str = f"{pat.span.line}:{pat.span.col}" if getattr(pat, "span", None) else "?"
        raise NotImplementedError(
            f"_collect_binds at {span_str}: unhandled Pattern subclass "
            f"{type(pat).__name__}; add an explicit arm in match_lower.py "
            f"to declare its binder semantics. (helixc internal bug — "
            f"please file an issue.)"
        )
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
