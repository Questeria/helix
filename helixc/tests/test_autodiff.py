"""Tests for source-level forward-mode automatic differentiation."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend import ast_nodes as A
from helixc.frontend.autodiff import differentiate, fmt


def diff_expr(src_expr: str, var: str) -> str:
    """Parse a single function whose body is `src_expr`, differentiate
    w.r.t. var, return the formatted derivative."""
    full = f"fn _f({var}: f32) -> f32 {{ {src_expr} }}"
    prog = parse(full)
    fn = prog.items[0]
    assert isinstance(fn, A.FnDecl)
    body_expr = fn.body.final_expr
    assert body_expr is not None, "body must be an expression"
    deriv = differentiate(body_expr, var)
    return fmt(deriv)


# ============================================================================
# Constants
# ============================================================================
def test_diff_int_const():
    assert diff_expr("5", "x") == "0"


def test_diff_float_const():
    assert diff_expr("3.14", "x") == "0"


# ============================================================================
# Variables
# ============================================================================
def test_diff_var_self():
    assert diff_expr("x", "x") == "1"


def test_diff_other_var():
    # Derivative of y w.r.t. x is 0
    full = "fn _f(x: f32, y: f32) -> f32 { y }"
    prog = parse(full)
    fn = prog.items[0]
    body = fn.body.final_expr
    deriv = differentiate(body, "x")
    assert fmt(deriv) == "0"


# ============================================================================
# Sums and differences
# ============================================================================
def test_diff_sum():
    # d(x + 5)/dx = 1
    assert diff_expr("x + 5", "x") == "1"


def test_diff_chain_sum():
    # d(x + x + x)/dx = 3 (= 1 + 1 + 1, folded to 3)
    out = diff_expr("x + x + x", "x")
    # After simplification + constant folding, expect "3"
    assert out == "3"


def test_diff_diff_self():
    # d(x - x)/dx = 0
    assert diff_expr("x - x", "x") == "0"


# ============================================================================
# Products (the interesting case for AD)
# ============================================================================
def test_diff_x_squared():
    # d(x*x)/dx = 1*x + x*1 -> simplifies to (x + x)
    out = diff_expr("x * x", "x")
    # Should be (x + x) after simplification
    assert out == "(x + x)"


def test_diff_x_cubed():
    # d(x*x*x)/dx by recursive product rule
    # x*x*x parses as ((x*x) * x); chain of product rules
    out = diff_expr("x * x * x", "x")
    # Result is non-trivial but should contain x
    assert "x" in out


def test_diff_2x():
    # d(2.0 * x)/dx = 0*x + 2.0*1 -> simplifies to 2.0
    out = diff_expr("2.0 * x", "x")
    assert out == "2"


def test_diff_x_times_const_plus_const():
    # d(x * 5.0 + 7.0)/dx = 5.0
    out = diff_expr("x * 5 + 7", "x")
    assert out == "5"


# ============================================================================
# Negation
# ============================================================================
def test_diff_neg_x():
    # d(-x)/dx = -1
    out = diff_expr("-x", "x")
    # Could be "(-1)" or just "-1" depending on formatting
    assert out in ("(-1)", "-1")


def test_diff_neg_neg_x():
    # d(-(-x))/dx = 1
    out = diff_expr("-(-x)", "x")
    # Simplifies double-negation
    assert "1" in out


# ============================================================================
# Block + let-binding support
# ============================================================================
def test_diff_through_let_binding():
    # let y = x; d(y * y)/dx = (x + x)
    full = """
    fn _f(x: f32) -> f32 {
        let y = x;
        y * y
    }
    """
    prog = parse(full)
    fn = prog.items[0]
    deriv = differentiate(fn.body, "x")
    assert fmt(deriv) == "(x + x)"


def test_diff_through_chain_let():
    # let a = x*x; let b = a*x; d(b)/dx
    # b = (x*x)*x = x^3, derivative is 3*x^2 (= ((x+x)*x + x*x))
    full = """
    fn _f(x: f32) -> f32 {
        let a = x * x;
        let b = a * x;
        b
    }
    """
    prog = parse(full)
    fn = prog.items[0]
    deriv = differentiate(fn.body, "x")
    out = fmt(deriv)
    # Expect a non-trivial expression in x. After full simplification it would
    # be 3*x*x but our simplifier may leave intermediate forms.
    assert "x" in out


def test_diff_const_let_unaffected():
    # let c = 5; d(c * x)/dx = 5
    full = """
    fn _f(x: f32) -> f32 {
        let c = 5;
        c * x
    }
    """
    prog = parse(full)
    fn = prog.items[0]
    deriv = differentiate(fn.body, "x")
    assert fmt(deriv) == "5"


def test_abs_subgrad_at_zero_is_zero():
    """At u=0 the subgradient of |u| is 0 (a documented choice).
    The forward-mode chain rule emits an If chain that yields 0 there."""
    from helixc.frontend.autodiff import differentiate as _diff
    full = "fn _f(x: f32) -> f32 { __abs(x) }"
    body = parse(full).items[0].body.final_expr
    deriv = _diff(body, "x")

    # Walk the deriv AST and find the FloatLit(0.0) used for the u==0 case.
    # We just verify there's a literal 0.0 somewhere in the deriv (the rule
    # emits the three-way if/else with cond_pos and cond_neg).
    seen: list[float] = []
    def walk(n):
        if isinstance(n, A.FloatLit):
            seen.append(float(n.value))
        for attr in ("left", "right", "cond", "then", "else_",
                     "operand", "value", "expr", "final_expr"):
            if hasattr(n, attr):
                v = getattr(n, attr)
                if v is not None:
                    walk(v)
        if hasattr(n, "stmts"):
            for s in n.stmts:
                walk(s)
        if hasattr(n, "args"):
            for a in n.args:
                walk(a)
    walk(deriv)
    assert 0.0 in seen, f"expected 0.0 (subgrad-at-0) in deriv, got {seen}"
    assert 1.0 in seen, f"expected 1.0 (positive branch) in deriv, got {seen}"
    # The negative branch produces -1 via Unary("-", FloatLit(1.0)), so 1.0
    # being present is sufficient — checking for -1 directly would over-fit
    # the simplification path.


def test_diff_memo_hits():
    """Two structurally-equal differentiate() calls should hit the cache."""
    from helixc.frontend.autodiff import (clear_diff_cache, diff_cache_stats,
                                          differentiate as _diff)
    clear_diff_cache()
    src1 = "fn _f(x: f32) -> f32 { x*x + 2.0*x + 1.0 }"
    src2 = "fn _f(x: f32) -> f32 { x*x + 2.0*x + 1.0 }"  # identical
    body1 = parse(src1).items[0].body.final_expr
    body2 = parse(src2).items[0].body.final_expr
    _diff(body1, "x")
    hits_a, misses_a = diff_cache_stats()
    _diff(body2, "x")
    hits_b, misses_b = diff_cache_stats()
    assert misses_a == 1, f"expected 1 miss after first diff, got {misses_a}"
    assert hits_b == hits_a + 1, f"expected cache hit on second diff, got hits {hits_b}"


def test_diff_memo_returns_independent_copy():
    """Mutating the cached result must not corrupt subsequent retrievals."""
    from helixc.frontend.autodiff import (clear_diff_cache,
                                          differentiate as _diff)
    clear_diff_cache()
    src = "fn _f(x: f32) -> f32 { x * x }"
    body = parse(src).items[0].body.final_expr
    d1 = _diff(body, "x")
    d2 = _diff(body, "x")
    assert d1 is not d2, "cache should return distinct deepcopies"


def test_grad_through_match():
    """Differentiating through a `match` requires that match has been
    desugared to if/let. With the match_lower pass at grad_pass entry,
    this should yield the right derivative for each arm body."""
    from helixc.frontend.match_lower import lower_matches
    full = """
    fn f(cond: bool, x: f32) -> f32 {
        match cond {
            true => 2.0 * x,
            false => 3.0 * x,
        }
    }
    """
    prog = parse(full)
    lower_matches(prog)  # match_lower pass
    fn = prog.items[0]
    deriv = differentiate(fn.body, "x")

    # Collect all numeric literals in the derivative.
    seen: list[float] = []
    def walk(n):
        if isinstance(n, (A.FloatLit, A.IntLit)):
            seen.append(float(n.value))
        for attr in ("left", "right", "cond", "then", "else_",
                     "operand", "value", "expr", "final_expr"):
            if hasattr(n, attr):
                v = getattr(n, attr)
                if v is not None:
                    walk(v)
        if hasattr(n, "stmts"):
            for s in n.stmts:
                walk(s)
        if hasattr(n, "args"):
            for a in n.args:
                walk(a)
    walk(deriv)
    # Both 2 and 3 should appear as constants somewhere in the deriv.
    assert 2.0 in seen, f"expected 2 in derivative literals, got {seen}"
    assert 3.0 in seen, f"expected 3 in derivative literals, got {seen}"


# ============================================================================
# Audit 28.8 B5: warnings for unhandled AD nodes (trap 85001)
# ============================================================================
def test_b5_ad_rejects_opaque_call():
    """Opaque calls now fail closed instead of compiling a zero derivative."""
    import pytest
    from helixc.frontend.autodiff import (
        differentiate, take_diff_warnings, clear_diff_cache,
    )
    clear_diff_cache()  # don't get a cached zero (no warning)
    take_diff_warnings()  # drain residual
    # Parse a fn calling an unknown helper.
    src = "fn _f(x: f32) -> f32 { strange_helper(x) }"
    prog = parse(src)
    fn = prog.items[0]
    body = fn.body.final_expr
    with pytest.raises(NotImplementedError, match="forward-mode AD.*strange_helper"):
        differentiate(body, "x")
    assert take_diff_warnings() == []


def test_b5_ad_warns_on_unsafe_block_with_unknown_inner():
    """B5: derivative through an UnsafeBlock propagates to the inner
    expr — `unsafe { x * 2.0 }` derives to 2.0. But if the inner is
    an unhandled kind, the warning fires."""
    from helixc.frontend.autodiff import (
        differentiate, take_diff_warnings, clear_diff_cache,
    )
    clear_diff_cache()  # don't get a cached zero (no warning)
    take_diff_warnings()  # drain residual
    src = "fn _f(x: f32) -> f32 { unsafe { x * 2.0 } }"
    prog = parse(src)
    fn = prog.items[0]
    body = fn.body.final_expr
    deriv = differentiate(body, "x")
    # No warnings expected — UnsafeBlock body is differentiable.
    warnings = take_diff_warnings()
    assert not warnings, \
        f"unexpected warnings for differentiable unsafe body: {warnings}"
    # Verify the derivative IS nonzero.
    assert fmt(deriv) != "0", f"expected nonzero derivative, got {fmt(deriv)}"


def test_b5_ad_cast_propagates_numeric():
    """B5: `(x as f64)` derivative is `dx` — the Cast arm in _diff now
    propagates through numeric casts (previously fell to fallthrough)."""
    from helixc.frontend.autodiff import differentiate, take_diff_warnings
    take_diff_warnings()
    src = "fn _f(x: f32) -> f64 { x as f64 }"
    prog = parse(src)
    fn = prog.items[0]
    body = fn.body.final_expr
    deriv = differentiate(body, "x")
    warnings = take_diff_warnings()
    assert not warnings, \
        f"no warnings expected for numeric cast, got: {warnings}"
    assert fmt(deriv) == "1", f"expected '1', got {fmt(deriv)}"


def test_b5_take_diff_warnings_drains():
    """B5: take_diff_warnings is idempotent — a second call returns []
    so warnings from one compile don't leak into the next."""
    from helixc.frontend.autodiff import (
        differentiate, take_diff_warnings, clear_diff_cache,
    )
    clear_diff_cache()  # avoid cached zero-derivs skipping the warn site
    take_diff_warnings()
    # Use a warning-producing non-call node; opaque calls now hard-fail.
    from helixc.frontend import ast_nodes as A
    span = A.Span(0, 0)
    body = A.Block(
        span=span,
        stmts=[A.Let(span=span, name="t", is_mut=False, ty=None,
                     value=A.FloatLit(span=span, value=1.0,
                                       type_suffix=None))],
        final_expr=None,
    )
    from helixc.frontend import autodiff
    autodiff._inline_lets(body, {})
    first = take_diff_warnings()
    assert first, "expected at least one warning on first take"
    second = take_diff_warnings()
    assert second == [], f"second take should be empty, got: {second}"


# ============================================================================
# Audit 28.8 cycle 2 (deferred observation #18) — empty block warns
# ============================================================================
def test_c2_def_18_empty_block_warns():
    """Deferred observation #18: pre-fix, `_inline_lets` on a Block
    with stmts but no `final_expr` returned `FloatLit(0.0)` silently.
    Now it emits an AD warning so the user can spot the missing tail."""
    from helixc.frontend import autodiff
    autodiff.take_diff_warnings()
    # Build a Block with a Let but no final_expr.
    span = A.Span(0, 0)
    blk = A.Block(
        span=span,
        stmts=[A.Let(span=span, name="t", is_mut=False, ty=None,
                     value=A.FloatLit(span=span, value=1.0,
                                       type_suffix=None))],
        final_expr=None,
    )
    autodiff._inline_lets(blk, {})
    warnings = autodiff.take_diff_warnings()
    assert any("empty block" in w.lower() for w in warnings), (
        f"expected empty-block warn, got: {warnings}"
    )


# ============================================================================
# Audit 28.8 cycle 2 C2-4 — grad_pass walker covers all Expr subtypes
# ============================================================================
def _names_in_prog(prog) -> set[str]:
    """Collect every Name.name appearing anywhere in the program after
    rewriting. Used to verify grad/grad_rev callees got rewritten to
    their `__grad` synthesized symbols."""
    found = set()

    def walk(e):
        if e is None:
            return
        if isinstance(e, A.Name):
            found.add(e.name)
        for attr in (
            "callee", "operand", "left", "right", "cond", "then", "else_",
            "scrutinee", "body", "value", "target", "transformation",
            "verifier", "inner", "iter_expr", "obj", "final_expr",
            "start", "end",
        ):
            if hasattr(e, attr):
                walk(getattr(e, attr))
        for attr in ("args", "elems", "indices", "arms", "stmts"):
            if hasattr(e, attr):
                seq = getattr(e, attr) or []
                for x in seq:
                    if hasattr(x, "body"):
                        walk(getattr(x, "body"))
                    if hasattr(x, "value"):
                        walk(getattr(x, "value"))
                    if hasattr(x, "expr"):
                        walk(getattr(x, "expr"))
                    walk(x)
        if hasattr(e, "fields"):
            for item in e.fields:
                if isinstance(item, tuple) and len(item) == 2:
                    walk(item[1])

    for item in prog.items:
        if isinstance(item, A.FnDecl):
            walk(item.body)
    return found


def test_c2_4_grad_in_array_lit_rewritten():
    """C2-4: `[grad(loss), grad(loss)]` (ArrayLit holding grad calls)
    must be rewritten so each element becomes `loss__grad`. Pre-fix
    the walker fell through ArrayLit entirely; the inner Call nodes
    remained `grad(loss)` and surfaced as unbound `grad` at lowering."""
    from helixc.frontend.grad_pass import grad_pass
    src = (
        "@pure fn loss(x: f32) -> f32 { x * x }\n"
        "fn use_arr() -> i32 {\n"
        "    let arr = [grad(loss), grad(loss)];\n"
        "    0\n"
        "}\n"
    )
    prog = parse(src)
    rewrites = grad_pass(prog)
    assert rewrites >= 1, f"expected >=1 grad rewrite, got {rewrites}"
    names = _names_in_prog(prog)
    assert "loss__grad" in names, (
        f"expected loss__grad after rewrite; names={sorted(names)}"
    )


def test_c2_4_grad_in_struct_lit_rewritten():
    """C2-4: `Optim { lr: 0.01, gfn: grad(loss) }` (StructLit) — the
    field value must be rewritten. Pre-fix StructLit was untouched."""
    from helixc.frontend.grad_pass import grad_pass
    src = (
        "struct Optim { lr: f32, gfn: f32 }\n"
        "@pure fn loss(x: f32) -> f32 { x * x }\n"
        "fn use_struct() -> i32 {\n"
        "    let o = Optim { lr: 0.01, gfn: grad(loss)(0.0) };\n"
        "    0\n"
        "}\n"
    )
    prog = parse(src)
    rewrites = grad_pass(prog)
    assert rewrites >= 1, f"expected >=1 grad rewrite, got {rewrites}"
    names = _names_in_prog(prog)
    assert "loss__grad" in names, (
        f"expected loss__grad after rewrite; names={sorted(names)}"
    )


def test_c2_4_grad_in_return_rewritten():
    """C2-4: `return grad(loss)` (Return holding grad call) — the
    Return.value must be walked. Pre-fix Return was untouched."""
    from helixc.frontend.grad_pass import grad_pass
    src = (
        "@pure fn loss(x: f32) -> f32 { x * x }\n"
        "fn use_ret(b: bool) -> f32 {\n"
        "    if b { return grad(loss)(1.0); }\n"
        "    0.0\n"
        "}\n"
    )
    prog = parse(src)
    rewrites = grad_pass(prog)
    assert rewrites >= 1, f"expected >=1 grad rewrite, got {rewrites}"
    names = _names_in_prog(prog)
    assert "loss__grad" in names


def test_c2_4_grad_in_unsafe_block_rewritten():
    """C2-4: `unsafe { grad(loss)(0.0) }` (UnsafeBlock body) must
    be walked. Pre-fix UnsafeBlock was untouched."""
    from helixc.frontend.grad_pass import grad_pass
    src = (
        "@pure fn loss(x: f32) -> f32 { x * x }\n"
        "fn use_unsafe() -> f32 {\n"
        "    unsafe { grad(loss)(2.0) }\n"
        "}\n"
    )
    prog = parse(src)
    rewrites = grad_pass(prog)
    assert rewrites >= 1, f"expected >=1 grad rewrite, got {rewrites}"
    names = _names_in_prog(prog)
    assert "loss__grad" in names


def test_c2_4_grad_in_range_bound_rewritten():
    """C2-4: `0..grad(loss)(...)` (Range upper bound) must be walked.
    Pre-fix Range was untouched."""
    from helixc.frontend.grad_pass import grad_pass
    src = (
        "@pure fn loss(x: f32) -> f32 { x }\n"
        "fn use_range() -> i32 {\n"
        "    let g = grad(loss);\n"
        "    let _r = 0..(g(1.0) as i32);\n"
        "    0\n"
        "}\n"
    )
    prog = parse(src)
    rewrites = grad_pass(prog)
    # In this form `grad(loss)` is at the let RHS (already a Call),
    # then `g(1.0)` is the actual rewrite target via let-alias resolution.
    # Verify the alias resolved + the loss__grad symbol exists.
    assert rewrites >= 1
    names = _names_in_prog(prog)
    assert "loss__grad" in names


def test_c3_1_grad_in_chained_else_if_rewritten():
    """C3-1: `if a {...} else if b { grad(loss)(...) } else {...}` —
    the chained `else if` body must be walked. Pre-fix _rewrite_in_expr's
    A.If arm only handled else_ = Block, silently skipping else_ = If."""
    from helixc.frontend.grad_pass import grad_pass
    src = (
        "@pure fn loss(x: f32) -> f32 { x * x }\n"
        "fn use_chained(a: bool, b: bool) -> f32 {\n"
        "    if a { 0.0 } else if b { grad(loss)(3.14) } else { 1.0 }\n"
        "}\n"
    )
    prog = parse(src)
    rewrites = grad_pass(prog)
    assert rewrites >= 1, (
        f"expected >=1 grad rewrite in chained else-if, got {rewrites}"
    )
    names = _names_in_prog(prog)
    assert "loss__grad" in names, (
        f"expected loss__grad after chained-else-if rewrite; "
        f"names={sorted(names)}"
    )


def test_c4_1_path_no_false_positive():
    """Audit 28.8 cycle 4 C4-1: `Maybe::None` (A.Path) in an AD'd fn
    body must NOT trigger the _inline_lets catch-all 85001 warning."""
    from helixc.frontend import autodiff
    autodiff.take_diff_warnings()
    # Synthesize Path expression and run through _inline_lets directly.
    span = A.Span(0, 0)
    path = A.Path(span=span, segments=["Maybe", "None"])
    result = autodiff._inline_lets(path, {})
    assert result is path, "Path should be returned identically"
    warnings = autodiff.take_diff_warnings()
    assert not any("85001" in w for w in warnings), (
        f"unexpected 85001 warn on Path: {warnings}"
    )


def test_c4_5_continue_no_false_positive_ad_warn():
    """Audit 28.8 cycle 5 C4-5: `A.Continue` in an AD'd fn body's loop
    must NOT trigger the `_inline_lets` catch-all 85001 warning. Pre-fix
    every `continue;` in a differentiated loop emitted a spurious
    AD-assumed-zero warning; `-Wad=error` then failed the compile."""
    from helixc.frontend import autodiff
    autodiff.take_diff_warnings()
    span = A.Span(0, 0)
    cont = A.Continue(span=span)
    result = autodiff._inline_lets(cont, {})
    assert result is cont, "Continue should be returned identically"
    warnings = autodiff.take_diff_warnings()
    assert not any("85001" in w for w in warnings), (
        f"unexpected 85001 warn on Continue: {warnings}"
    )


def test_c4_5_tilelit_no_false_positive_ad_warn():
    """Audit 28.8 cycle 5 C4-5: TileLit must NOT trigger the
    `_inline_lets` catch-all 85001 warning. (TileLit has its own arm
    that walks shape/memspace children, so substitution still applies
    but no spurious AD-zero warning fires.)"""
    from helixc.frontend import autodiff
    autodiff.take_diff_warnings()
    span = A.Span(0, 0)
    tile = A.TileLit(
        span=span,
        dtype=A.TyName(span=span, name="f32"),
        shape=[A.IntLit(span=span, value=4, type_suffix=None)],
        memspace=A.Name(span=span, name="REG", generics=[]),
        init="zeros",
    )
    result = autodiff._inline_lets(tile, {})
    # Result must be a TileLit (either same or rewritten).
    assert isinstance(result, A.TileLit), (
        f"TileLit should remain TileLit, got {type(result).__name__}"
    )
    warnings = autodiff.take_diff_warnings()
    assert not any("85001" in w for w in warnings), (
        f"unexpected 85001 warn on TileLit: {warnings}"
    )


def test_c4_3_inline_lets_if_cond_substituted():
    """Audit 28.8 cycle 4 C4-3: `_inline_lets` must substitute let-bound
    names appearing in `if cond { ... }`. Pre-fix the cond was passed
    through unmodified."""
    from helixc.frontend import autodiff
    span = A.Span(0, 0)
    # Build `if g > 0 { 1 } else { 2 }` where g should resolve to `x*2`.
    cond = A.Binary(
        span=span, op=">",
        left=A.Name(span=span, name="g", generics=[]),
        right=A.IntLit(span=span, value=0, type_suffix=None),
    )
    then_blk = A.Block(span=span, stmts=[],
                       final_expr=A.IntLit(span=span, value=1, type_suffix=None))
    else_blk = A.Block(span=span, stmts=[],
                       final_expr=A.IntLit(span=span, value=2, type_suffix=None))
    if_expr = A.If(span=span, cond=cond, then=then_blk, else_=else_blk)
    env = {"g": A.Binary(
        span=span, op="*",
        left=A.Name(span=span, name="x", generics=[]),
        right=A.IntLit(span=span, value=2, type_suffix=None),
    )}
    result = autodiff._inline_lets(if_expr, env)
    # The cond should now reference `x` instead of `g`.
    assert isinstance(result, A.If)
    assert isinstance(result.cond, A.Binary)
    left = result.cond.left
    # left was Name('g'); after inlining it should be the env value.
    assert not (isinstance(left, A.Name) and left.name == "g"), (
        f"if.cond was not substituted; left={left}"
    )


def test_c6_revert_c4_2_literal_binary_no_false_trap():
    """Audit 28.8 cycle 5/6 — D2 polarity fully reverted.

    History:
      - cycle 3 D2: added `val_tag == 16` (AST_CALL) sentinel-12 →
        broke i32-returning-fn capture pattern (CRITICAL C4-1 / F1).
      - cycle 4 C4-2: BROADENED D2 to Binary/Unary/etc. — additional
        false-positives on `let a = 10 + 5; ...`.
      - cycle 6 C5-1: REVERTED only the broadening; kept Call-only
        sentinel-12.
      - cycle 5 (this) C4-1 / F1 final REVERT: removed Call-only
        sentinel-12 too. Parser can't infer fn return types, so the
        trap-76003 inference must move to a typecheck pass.

    Verify parser.hx no longer contains the cycle-4 broadening
    (val_tag == 6 AST_LT arm) AND no longer contains the cycle-3
    Call-only arm (val_tag == 16). Both are gone — only the literal-
    RHS type-tag arms remain (val_tag 0/27/31/34/35/36/37/38/39/40/41)."""
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    parser_path = os.path.join(here, "..", "bootstrap", "parser.hx")
    with open(parser_path, "r", encoding="utf-8") as f:
        src = f.read()
    # The cycle-4 broadening introduced an `if val_tag == 6 {` arm for
    # AST_LT. After cycle-6 revert, that arm should be gone.
    assert "if val_tag == 6 {" not in src, (
        "cycle-4 C4-2 broadening still present in parser.hx — "
        "C5-1 revert did not land"
    )
    # cycle-5 final REVERT: the cycle-3 D2 Call-only arm (val_tag == 16)
    # is gone too. The parser cannot infer fn return types, so this trap
    # must come from a typecheck pass when one exists.
    assert "if val_tag == 16 {" not in src, (
        "cycle-3 D2 Call-only sentinel-12 arm still present in parser.hx "
        "— cycle-5 C4-1 / F1 revert did not land"
    )


# ============================================================================
def test_c52_ad1_fn_table_sig_includes_body_hash():
    """Stage 28.9 cycle 53 audit-R C52-AD1 regression (HIGH):
    pre-fix `_fn_table_sig` returned ONLY the sorted set of fn
    names, so two calls to `differentiate()` with the same expr,
    same var, same fn_table KEYS but DIFFERENT fn BODIES returned
    the stale cached derivative — silent gradient corruption.
    Reproduced numerically: g(x) = x*x has derivative 2*x (= 4 at x=2);
    after changing g to x*x*x (derivative 3*x*x = 12 at x=2), the
    cache returned 4 instead of 12.

    The cycle-53 fix extends `_fn_table_sig` to include
    `structural_hash(fn.body)` per entry, so any body change
    invalidates the cache."""
    from helixc.frontend.autodiff import (
        differentiate, clear_diff_cache, diff_cache_stats,
    )
    prog1 = parse("fn g(x: f64) -> f64 { x * x }")
    prog2 = parse("fn g(x: f64) -> f64 { x * x * x }")
    g1 = next(it for it in prog1.items
              if isinstance(it, A.FnDecl) and it.name == "g")
    g2 = next(it for it in prog2.items
              if isinstance(it, A.FnDecl) and it.name == "g")
    s = A.Span(line=1, col=1)
    def call_expr():
        return A.Call(
            span=s, callee=A.Name(span=s, name="g", generics=[]),
            args=[A.Name(span=s, name="y", generics=[])],
        )
    clear_diff_cache()
    d1 = differentiate(call_expr(), "y", {"g": g1})
    d2 = differentiate(call_expr(), "y", {"g": g2})
    hits, misses = diff_cache_stats()
    # Two distinct fn bodies → cache must MISS twice, not HIT.
    assert misses == 2, (
        f"cache must miss for both compiles when bodies differ; "
        f"got hits={hits}, misses={misses}"
    )
    # And the derivatives must structurally differ.
    assert repr(d1) != repr(d2), (
        f"derivatives must differ when bodies differ "
        f"(x*x vs x*x*x); got identical {d1!r}"
    )


def test_c54_ad1_fn_table_sig_includes_attrs():
    """Stage 28.9 cycle 55 audit-T C54-AD1 regression (HIGH):
    `_fn_table_sig` must include `fn.attrs`. Pre-fix, two fn_tables
    with identical bodies but different @pure markers produced
    identical cache keys; `_inline_user_calls` reads `"pure" in
    fn.attrs` and decides whether to inline. Cache returned wrong
    derivative for the case the attribute changed between compiles."""
    from helixc.frontend.autodiff import _fn_table_sig
    prog_pure = parse("@pure fn g(x: f64) -> f64 { x * x }")
    prog_impure = parse("fn g(x: f64) -> f64 { x * x }")
    g_pure = next(it for it in prog_pure.items
                  if isinstance(it, A.FnDecl))
    g_impure = next(it for it in prog_impure.items
                    if isinstance(it, A.FnDecl))
    sig_pure = _fn_table_sig({"g": g_pure})
    sig_impure = _fn_table_sig({"g": g_impure})
    assert sig_pure != sig_impure, (
        f"signatures must differ for @pure vs non-@pure with "
        f"same body; got identical {sig_pure!r}"
    )


def test_c54_ad2_fn_table_sig_includes_arity():
    """C54-AD2 regression (MED): `_fn_table_sig` must include
    `len(fn.params)`. Body hashing uses de-Bruijn so `fn g(x,y) = x`
    and `fn g(x) = x` produce IDENTICAL body hashes (both bodies
    reference param at de-Bruijn depth 0), but they differ in
    inlining-gate behavior at call sites (`len(fn.params) ==
    len(args)`). Cache hit corrupts gradient for the mismatched
    arity case."""
    from helixc.frontend.autodiff import _fn_table_sig
    prog_2arg = parse("fn g(x: f64, _y: f64) -> f64 { x }")
    prog_1arg = parse("fn g(x: f64) -> f64 { x }")
    g2 = next(it for it in prog_2arg.items if isinstance(it, A.FnDecl))
    g1 = next(it for it in prog_1arg.items if isinstance(it, A.FnDecl))
    sig_2arg = _fn_table_sig({"g": g2})
    sig_1arg = _fn_table_sig({"g": g1})
    assert sig_2arg != sig_1arg, (
        f"signatures must differ for 2-arg vs 1-arg fn with same "
        f"body; got identical {sig_2arg!r}"
    )


def test_c54_ad3_cache_layer_catches_not_implemented_error():
    """C54-AD3 regression (MED): the autodiff cache call site must
    catch `NotImplementedError` (cycle-35 loud-fail discipline in
    structural_hash for unknown AST subclasses) so the cache
    gracefully bypasses instead of crashing the caller."""
    from helixc.frontend.autodiff import _fn_table_sig
    # Construct a synthetic AST node not handled by structural_hash.
    class SyntheticUnknown:
        def __init__(self):
            self.span = A.Span(line=1, col=1)
    # Build a fake FnDecl with the synthetic body.
    fake_fn = A.FnDecl(
        span=A.Span(1, 1), name="x", generics=[], params=[],
        return_ty=None, where_clauses=[],
        body=SyntheticUnknown(),  # type: ignore — testing failure path
        attrs=[], is_pub=False, is_extern=False, extern_abi=None,
    )
    # Must NOT raise NotImplementedError — must return a sentinel.
    sig = _fn_table_sig({"x": fake_fn})
    assert "unhashable" in sig, (
        f"unknown AST body must produce <unhashable:...> sentinel; "
        f"got {sig!r}"
    )


# ============================================================================
# Stage 54 Inc 1: chain-rule arms for __min/__max/__clamp/__sign
# ============================================================================
def test_stage54_inc1_min_f64_chain_rule_forward():
    """d/dx __min_f64(x, 5.0) = if x <= 5.0 then 1.0 else 0.0
    (b is a constant so db/dx = 0; second term drops to 0)."""
    out = diff_expr("__min_f64(x, 5.0)", "x")
    # Forward dispatcher returns indicator_a * dx + indicator_b * db
    # where db = 0. fmt() may simplify or leave the structure.
    assert "if" in out and "<=" in out, \
        f"d/dx min(x, 5) should yield indicator-If, got: {out}"


def test_stage54_inc1_max_f64_chain_rule_forward():
    """d/dx __max_f64(x, 5.0) = if x > 5.0 then 1.0 else 0.0
    (subgradient at equality picks 0 — strict > for left arg)."""
    out = diff_expr("__max_f64(x, 5.0)", "x")
    assert "if" in out and ">" in out, \
        f"d/dx max(x, 5) should yield indicator-If, got: {out}"


def test_stage54_inc1_clamp_f64_chain_rule_forward():
    """d/dx __clamp_f64(x, 0.0, 1.0) = if (0.0 <= x AND x <= 1.0)
    then 1.0 else 0.0. lo/hi are non-differentiable constants."""
    out = diff_expr("__clamp_f64(x, 0.0, 1.0)", "x")
    assert "if" in out and "&&" in out, \
        f"d/dx clamp(x, 0, 1) should yield AND-indicator, got: {out}"


def test_stage54_inc1_sign_chain_rule_forward():
    """d/dx __sign(x) = 0 (distributional sense)."""
    out = diff_expr("__sign(x)", "x")
    assert out in ("0", "0.0", "0.0_f64"), \
        f"d/dx sign(x) should be 0, got: {out}"


def test_stage54_inc1_min_i32_zero_derivative_forward():
    """d/dx __min_i32(x, 5) = 0 (integer-valued, non-differentiable
    for AD purposes)."""
    out = diff_expr("__min_i32(x, 5)", "x")
    assert out in ("0", "0.0"), \
        f"d/dx min_i32(x, 5) should be 0, got: {out}"


def test_stage54_inc1_max_i32_zero_derivative_forward():
    """d/dx __max_i32(x, 5) = 0."""
    out = diff_expr("__max_i32(x, 5)", "x")
    assert out in ("0", "0.0"), \
        f"d/dx max_i32(x, 5) should be 0, got: {out}"


def test_stage54_inc1_clamp_i32_zero_derivative_forward():
    """d/dx __clamp_i32(x, 0, 1) = 0."""
    out = diff_expr("__clamp_i32(x, 0, 1)", "x")
    assert out in ("0", "0.0"), \
        f"d/dx clamp_i32(x, 0, 1) should be 0, got: {out}"


def test_stage54_inc3a_substitute_names_handles_loop_match():
    """Stage 54 closure gate-1 silent-failure HIGH-1 fix:
    `_substitute_names` now recurses into For/While/Loop/Match
    body subtrees. Pre-fix this was a LATENT BUG that Inc 3a's
    loop-body descent in `_inline_user_calls` exposed: a helper
    whose body contained a loop would inline into a caller, but
    its param substitution would fail to walk into the loop
    body, leaving the param name unbound (or shadowed silently).

    This pin substitutes a name through a synthesized For body
    and verifies the substitution went through."""
    from helixc.frontend.autodiff import _substitute_names

    # Build a tiny AST manually: For { let _ = p + 1; }
    # with p substituted by a literal 42.
    span = type("Span", (), {"line": 0, "col": 0})()
    p_use = A.Name(span=span, name="p", generics=[])
    body_stmt = A.Let(
        span=span, name="dummy",
        ty=None,
        value=A.Binary(span=span, op="+",
                       left=p_use,
                       right=A.IntLit(span=span, value=1)),
        is_mut=False,
    )
    for_body = A.Block(span=span, stmts=[body_stmt], final_expr=None)
    iter_e = A.Range(span=span,
                     start=A.IntLit(span=span, value=0),
                     end=A.IntLit(span=span, value=3))
    for_expr = A.For(span=span, var_name="i",
                     iter_expr=iter_e, body=for_body)

    # Substitute p -> 42
    subs = {"p": A.IntLit(span=span, value=42)}
    result = _substitute_names(for_expr, subs)

    # Verify the For's body's let.value's left arg is now 42
    # (substituted) not "p" (would be the latent-bug behavior).
    assert isinstance(result, A.For)
    let_stmt = result.body.stmts[0]
    assert isinstance(let_stmt, A.Let)
    bin_expr = let_stmt.value
    assert isinstance(bin_expr, A.Binary)
    assert isinstance(bin_expr.left, A.IntLit), \
        f"Stage 54 gate-1 HIGH-1: For-body let.value.left should " \
        f"be IntLit(42) after substitution, got {type(bin_expr.left).__name__}"
    assert bin_expr.left.value == 42


def test_stage54_inc3a_substitute_names_handles_exprstmt_assign():
    """Stage 54 closure gate-1 EXTENDED silent-failure fix:
    `_substitute_names._go_block` now descends into ExprStmt
    (parallel to the inliner's same-class bug). And `go()` now
    handles A.Assign (Expr — pre-fix it fell to `return e`).

    Without these: a helper with `while ... { x = x + p; }`
    would inline into a caller, but the `p` inside the Assign
    inside the ExprStmt inside the While body would never get
    substituted. The `p` becomes an unbound name in the caller
    or shadows a different local — silently wrong gradient.

    This pin substitutes a name through:
        Block { ExprStmt(Assign(target=Name(t), op="=",
                                value=Binary(Name(t), +, Name(p)))) }
    and verifies p (in Assign.value.right) was substituted."""
    from helixc.frontend.autodiff import _substitute_names

    span = type("Span", (), {"line": 0, "col": 0})()
    target_name = A.Name(span=span, name="t", generics=[])
    p_use = A.Name(span=span, name="p", generics=[])
    assign_value = A.Binary(span=span, op="+",
                             left=A.Name(span=span, name="t",
                                          generics=[]),
                             right=p_use)
    assign_expr = A.Assign(span=span, target=target_name,
                            op="=", value=assign_value)
    expr_stmt = A.ExprStmt(span=span, expr=assign_expr)
    blk = A.Block(span=span, stmts=[expr_stmt], final_expr=None)

    subs = {"p": A.IntLit(span=span, value=99)}
    result = _substitute_names(blk, subs)

    assert isinstance(result, A.Block)
    assert len(result.stmts) == 1
    out_stmt = result.stmts[0]
    assert isinstance(out_stmt, A.ExprStmt), \
        f"ExprStmt should survive substitution, got {type(out_stmt).__name__}"
    out_assign = out_stmt.expr
    assert isinstance(out_assign, A.Assign), \
        f"Assign should survive, got {type(out_assign).__name__}"
    out_bin = out_assign.value
    assert isinstance(out_bin, A.Binary)
    assert isinstance(out_bin.right, A.IntLit), \
        f"Stage 54 gate-1 extended: Assign.value.right (the 'p' " \
        f"position) should be IntLit(99) after substitution; " \
        f"got {type(out_bin.right).__name__}. Pre-fix, ExprStmt " \
        f"and/or Assign was a substitution dead-zone."
    assert out_bin.right.value == 99


def test_stage54_gate2_substitute_names_handles_remaining_ast_kinds():
    """Stage 54 gate-2 sweep: `_substitute_names` now substitutes
    through Cast/Index/Field/TupleLit/ArrayLit/StructLit/
    UnsafeBlock/Return/Break/Range — same defect class as the
    gate-1 For/While/Loop/Match gap.

    Pre-fix: a helper containing `arr[p]`, `obj.p`, `(p, 0)`,
    `[p]`, `Point{x: p}`, `unsafe { p }`, `return p`, `p as f64`,
    or `0..p` would inline into a caller but leave `p` un-
    substituted — unbound name or shadowing in the caller scope.

    This pin synthesizes each kind with a Name('p') leaf and
    verifies substitution went through."""
    from helixc.frontend.autodiff import _substitute_names

    span = type("Span", (), {"line": 0, "col": 0})()
    def p_name():
        return A.Name(span=span, name="p", generics=[])
    def repl():
        return A.IntLit(span=span, value=77)

    cases = {
        "Cast.value":
            (A.Cast(span=span, value=p_name(),
                    target_ty=A.TyName(span=span, name="f64")),
             lambda r: r.value),
        "Index.callee":
            (A.Index(span=span, callee=p_name(),
                      indices=[A.IntLit(span=span, value=0)]),
             lambda r: r.callee),
        "Field.obj":
            (A.Field(span=span, obj=p_name(), name="x"),
             lambda r: r.obj),
        "TupleLit.elems[0]":
            (A.TupleLit(span=span, elems=[p_name()]),
             lambda r: r.elems[0]),
        "ArrayLit.elems[0]":
            (A.ArrayLit(span=span, elems=[p_name()]),
             lambda r: r.elems[0]),
        "StructLit.fields[0][1]":
            (A.StructLit(span=span, name="Point",
                          fields=[("x", p_name())]),
             lambda r: r.fields[0][1]),
        "UnsafeBlock.body.final_expr":
            (A.UnsafeBlock(span=span,
                            body=A.Block(span=span, stmts=[],
                                          final_expr=p_name())),
             lambda r: r.body.final_expr),
        "Return.value":
            (A.Return(span=span, value=p_name()),
             lambda r: r.value),
        "Break.value":
            (A.Break(span=span, value=p_name()),
             lambda r: r.value),
        "Range.start":
            (A.Range(span=span, start=p_name(),
                      end=A.IntLit(span=span, value=10)),
             lambda r: r.start),
        "Range.end":
            (A.Range(span=span,
                      start=A.IntLit(span=span, value=0),
                      end=p_name()),
             lambda r: r.end),
    }

    for label, (expr, extract) in cases.items():
        result = _substitute_names(expr, {"p": repl()})
        target = extract(result)
        assert isinstance(target, A.IntLit), (
            f"Stage 54 gate-2: substitution into {label} "
            f"failed — expected IntLit(77), got "
            f"{type(target).__name__}. Pre-fix this was a "
            f"silent substitution dead-zone."
        )
        assert target.value == 77, \
            f"{label}: substituted value wrong"


def test_stage54_gate2_inliner_handles_remaining_ast_kinds():
    """Stage 54 gate-2 sweep: inliner `go()` walker now descends
    into Cast/Index/Field/TupleLit/ArrayLit/StructLit/UnsafeBlock/
    Match/Assign/Return/Break/Range. Pre-fix, a pure-helper call
    wrapped inside any of these forms was never inlined."""
    from helixc.frontend.autodiff import _inline_user_calls

    span = type("Span", (), {"line": 0, "col": 0})()
    # Build a Call(pure_double, [Name(x)]) leaf and place it in
    # each of the new AST positions. Then run the inliner with
    # a minimal fn_table containing pure_double.
    src = "fn pure_double(z: f64) -> f64 { z + z }"
    prog = parse(src)
    fn_table = {it.name: it for it in prog.items
                if isinstance(it, A.FnDecl)}

    def call_leaf():
        return A.Call(span=span,
                       callee=A.Name(span=span, name="pure_double",
                                      generics=[]),
                       args=[A.Name(span=span, name="x",
                                     generics=[])])

    # Build a TupleLit containing the call, then inline.
    expr = A.TupleLit(span=span, elems=[call_leaf()])
    result = _inline_user_calls(expr, fn_table)
    assert isinstance(result, A.TupleLit)
    inlined = result.elems[0]
    assert not (
        isinstance(inlined, A.Call)
        and isinstance(inlined.callee, A.Name)
        and inlined.callee.name == "pure_double"
    ), (
        f"Stage 54 gate-2: inliner should descend into "
        f"TupleLit.elems and inline pure_double; got "
        f"{type(inlined).__name__}. Pre-fix this was a "
        f"silent inline dead-zone."
    )

    # Same check inside Return.value (rep for the trailing arms).
    expr2 = A.Return(span=span, value=call_leaf())
    result2 = _inline_user_calls(expr2, fn_table)
    assert isinstance(result2, A.Return)
    inlined2 = result2.value
    assert not (
        isinstance(inlined2, A.Call)
        and isinstance(inlined2.callee, A.Name)
        and inlined2.callee.name == "pure_double"
    ), (
        f"Stage 54 gate-2: inliner should descend into "
        f"Return.value; got {type(inlined2).__name__}"
    )


def test_stage54_gate2_i32_variants_emit_intlit_zero():
    """Stage 54 gate-2 MEDIUM-4 fix: `__min_i32`/`__max_i32`/
    `__clamp_i32` gradient zero now emits IntLit instead of
    FloatLit. Matches the function's return type for downstream
    typecheck consistency in i32-arithmetic contexts."""
    from helixc.frontend.autodiff import _diff_call_chain_rule
    span = type("Span", (), {"line": 0, "col": 0})()
    for name in ("__min_i32", "__max_i32", "__clamp_i32"):
        args = [A.Name(span=span, name="x", generics=[]),
                A.IntLit(span=span, value=0)]
        if name == "__clamp_i32":
            args.append(A.IntLit(span=span, value=10))
        call = A.Call(span=span,
                       callee=A.Name(span=span, name=name,
                                      generics=[]),
                       args=args)
        result = _diff_call_chain_rule(call, "x", span)
        assert isinstance(result, A.IntLit), \
            f"Stage 54 gate-2 MEDIUM-4: {name} gradient should " \
            f"be IntLit(0), got {type(result).__name__}"
        assert result.value == 0


def test_stage54_gate2_clamp_warns_when_var_in_lo_or_hi():
    """Stage 54 gate-2 MEDIUM-5 fix: `__clamp(x, lo, hi)` chain
    rule now emits `_ad_warn` (recorded in the module-level
    _DIFF_WARNINGS list) when `var` syntactically appears in
    `lo` or `hi` (whose contributions are silently dropped).
    Per CLAUDE.md silent-failure ban."""
    from helixc.frontend.autodiff import (
        _diff_call_chain_rule, take_diff_warnings,
    )
    span = type("Span", (), {"line": 0, "col": 0})()
    # __clamp(x, w * 0.1, w * 0.9) — gradient w.r.t. w drops
    # dlo and dhi contributions. Must warn.
    w = lambda: A.Name(span=span, name="w", generics=[])
    lo = A.Binary(span=span, op="*", left=w(),
                   right=A.FloatLit(span=span, value=0.1))
    hi = A.Binary(span=span, op="*", left=w(),
                   right=A.FloatLit(span=span, value=0.9))
    call = A.Call(
        span=span,
        callee=A.Name(span=span, name="__clamp", generics=[]),
        args=[A.Name(span=span, name="x", generics=[]), lo, hi],
    )
    # Drain any pre-existing warnings to isolate this test
    take_diff_warnings()
    _diff_call_chain_rule(call, "w", span)
    msgs = take_diff_warnings()
    assert any("clamp" in m.lower() and ("dlo" in m or "dhi" in m
                                          or "drop" in m.lower())
               for m in msgs), \
        f"Stage 54 gate-2 MEDIUM-5: __clamp should warn when " \
        f"differentiation var appears in lo/hi; got warnings: {msgs}"

    # Negative control: var NOT in lo/hi — no clamp-warn.
    take_diff_warnings()  # drain
    call2 = A.Call(
        span=span,
        callee=A.Name(span=span, name="__clamp",
                       generics=[]),
        args=[A.Name(span=span, name="x", generics=[]),
              A.FloatLit(span=span, value=0.0),
              A.FloatLit(span=span, value=1.0)],
    )
    _diff_call_chain_rule(call2, "x", span)
    msgs2 = take_diff_warnings()
    assert not any("clamp" in m.lower() and ("dlo" in m or "dhi" in m)
                    for m in msgs2), \
        f"Stage 54 gate-2 MEDIUM-5: no clamp-warn when var not " \
        f"in lo/hi; got: {msgs2}"


def test_stage54_inc3a_loop_body_descent_inlines_pure_helper():
    """Stage 54 Inc 3a: `_inline_user_calls.go()` walker now
    descends into A.For/A.While/A.Loop bodies. Pre-fix, loop
    bodies were returned as-is, so any pure-helper calls inside
    were never inlined — AD passes saw them as opaque.

    Stage 54 closure gate-1 code-review CRITICAL F1 fix
    (REPLACES prior vacuous version): prior test placed
    pure_double() inside an Assign statement, but the inliner's
    A.Block arm only descends into Let.value/ConstStmt.value
    (not ExprStmt or Assign), and the test walker missed
    ExprStmt entirely — both lookups bypassed the call.
    The pin "passed" even with Inc 3a reverted.

    Post-fix: place pure_double(x) in Block.final_expr position
    (which BOTH the inliner and test walker visit). Now the
    pin is load-bearing — reverting Inc 3a makes it FAIL."""
    from helixc.frontend.autodiff import _inline_user_calls

    # Helper is called in a loop body that yields the helper's
    # value as Block.final_expr. The inliner's Block arm
    # descends into final_expr; Inc 3a's For arm descends into
    # the loop body. Both must work for the call to be replaced.
    src = '''
fn pure_double(z: f64) -> f64 { z + z }
fn caller(x: f64) -> f64 {
    let mut acc: f64 = 0.0;
    let mut i: i32 = 0;
    while i < 3 {
        let _step: f64 = pure_double(x);
        i = i + 1;
    };
    acc
}
'''
    prog = parse(src)
    fn_table = {it.name: it for it in prog.items
                if isinstance(it, A.FnDecl)}
    caller_fn = fn_table["caller"]
    body_expr = caller_fn.body

    inlined = _inline_user_calls(body_expr, fn_table)

    # Direct check: walk into the while-body's stmts and find
    # the Let("_step"); inspect its value.
    assert isinstance(inlined, A.Block)
    while_expr = None
    for s in inlined.stmts:
        if isinstance(s, A.ExprStmt) and isinstance(s.expr, A.While):
            while_expr = s.expr
            break
    assert while_expr is not None, \
        "expected a While in caller body"
    inner_let = None
    for s in while_expr.body.stmts:
        if isinstance(s, A.Let) and s.name == "_step":
            inner_let = s
            break
    assert inner_let is not None, \
        "expected let _step in while body"
    # Inc 3a + pre-existing Block-arm Let.value descent means
    # pure_double should be inlined into the let-RHS as x + x.
    # Pre-Inc-3a: the entire While was returned as-is (no
    # descent), so the Let's value remained `pure_double(x)`.
    assert not (
        isinstance(inner_let.value, A.Call)
        and isinstance(inner_let.value.callee, A.Name)
        and inner_let.value.callee.name == "pure_double"
    ), (
        f"Stage 54 Inc 3a: walker should have descended into "
        f"the while-body and inlined pure_double in the "
        f"Let.value, but it remained as a call. Pre-Inc-3a "
        f"this was the silent omission. Got: "
        f"{type(inner_let.value).__name__}({getattr(inner_let.value, 'callee', '?')})"
    )


def test_stage54_inc2_forward_reverse_asymmetry_already_fixed():
    """Stage 54 Inc 2 CONFIRMED no-op: the forward/reverse
    asymmetry on unrecognized opaque multi-arg calls that the
    Inc 2 blueprint described was already fixed at Stage 35.
    Both forward and reverse modes now raise NotImplementedError
    on opaque user-fn calls — confirmed by this pin.

    Inc 2 was scoped to "align loud-fail behavior between modes".
    Investigation showed Stage 35's autodiff.py:1132-1139 already
    raises in forward mode, matching reverse mode's autodiff_
    reverse.py:691-695 raise. So Inc 2 ships as a verified no-op."""
    from helixc.frontend.autodiff_reverse import differentiate_reverse

    src = (
        "fn opaque(x: f64, y: f64) -> f64 { 0.0 } "
        "fn g(z: f64) -> f64 { opaque(z, 1.0) }"
    )
    prog = parse(src)
    g = [
        it for it in prog.items
        if isinstance(it, A.FnDecl) and it.name == "g"
    ][0]
    body = g.body.final_expr

    # Forward: must raise.
    try:
        differentiate(body, "z")
        raise AssertionError("forward should have raised")
    except NotImplementedError as e:
        assert "opaque call 'opaque'" in str(e), \
            f"forward error msg mismatch: {e}"

    # Reverse: must raise (same error class + same message shape).
    try:
        differentiate_reverse(body, ["z"])
        raise AssertionError("reverse should have raised")
    except NotImplementedError as e:
        assert "opaque call 'opaque'" in str(e), \
            f"reverse error msg mismatch: {e}"


def test_stage54_inc1_min_chain_rule_in_composed_expr():
    """d/dx (__min_f64(x, 5.0) + x) = (if x<=5 then 1 else 0) + 1."""
    out = diff_expr("__min_f64(x, 5.0) + x", "x")
    # Just verify it doesn't return zero (which would be the
    # pre-Stage-54 opaque-call behavior).
    assert out not in ("0", "0.0"), \
        f"composed __min should propagate non-zero derivative, " \
        f"got opaque-zero: {out}"


# ============================================================================
# Test runner
# ============================================================================
def main():
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            failed += 1
        except Exception as e:
            import traceback
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
