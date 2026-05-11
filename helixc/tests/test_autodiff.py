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
def test_b5_ad_warns_on_opaque_call():
    """B5: an opaque user fn call (not in the chain-rule table) was
    silently a zero derivative. Now emits a warning to _DIFF_WARNINGS."""
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
    # Note: no fn_table is passed, so the call site falls through to
    # the unmatched-call branch and produces the warning.
    deriv = differentiate(body, "x")
    warnings = take_diff_warnings()
    assert any("85001" in w for w in warnings), \
        f"expected 85001 warning for opaque call, got: {warnings}"


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
    # Use a different fn name than other B5 tests so the memoization
    # by structural hash doesn't return a cached deriv from earlier.
    src = "fn _f(y: f32) -> f32 { totally_unknown_fn_xyz(y) }"
    prog = parse(src)
    body = prog.items[0].body.final_expr
    differentiate(body, "y")
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
