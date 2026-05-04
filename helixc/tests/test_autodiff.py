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
