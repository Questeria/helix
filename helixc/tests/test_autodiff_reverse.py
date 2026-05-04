"""Unit tests for the reverse-mode AD engine.

These tests check the symbolic shape of the gradient — not numerics. Numerics
are exercised end-to-end via test_codegen tests for grad_rev(...)."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend import ast_nodes as A
from helixc.frontend.autodiff import fmt
from helixc.frontend.autodiff_reverse import differentiate_reverse


def _body_of(src: str) -> A.Expr:
    """Parse a single-fn program and return the function body."""
    prog = parse(src)
    fn = next(item for item in prog.items if isinstance(item, A.FnDecl))
    return fn.body


def test_const_zero_gradient():
    body = _body_of("fn f(x: f32) -> f32 { 5.0 }")
    grads = differentiate_reverse(body, ["x"])
    assert fmt(grads["x"]) == "0", f"got {fmt(grads['x'])}"


def test_x_squared_grad_is_2x():
    body = _body_of("fn f(x: f32) -> f32 { x * x }")
    grads = differentiate_reverse(body, ["x"])
    # Reverse mode: adjoint left = 1*x = x; adjoint right = 1*x = x. Sum = x+x = 2x.
    # The simplifier may fold (1*x)+(1*x) -> x+x or (x+x).
    out = fmt(grads["x"])
    assert "x" in out and "+" in out, f"expected sum-of-x form, got {out}"


def test_linear_gradient_x():
    body = _body_of("fn f(x: f32, y: f32) -> f32 { 3.0 * x + 5.0 * y }")
    grads = differentiate_reverse(body, ["x", "y"])
    # ∂f/∂x = 3, ∂f/∂y = 5. Reverse mode produces 1*3 = 3 (after simplify).
    assert fmt(grads["x"]) == "3", f"got {fmt(grads['x'])}"
    assert fmt(grads["y"]) == "5", f"got {fmt(grads['y'])}"


def test_quadratic_two_vars():
    body = _body_of("fn f(x: f32, y: f32) -> f32 { x * x + y * y }")
    grads = differentiate_reverse(body, ["x", "y"])
    # ∂f/∂x = 2x; reverse mode emits x+x.
    out_x = fmt(grads["x"])
    out_y = fmt(grads["y"])
    assert "x" in out_x and "+" in out_x
    assert "y" in out_y and "+" in out_y


def test_subtraction():
    body = _body_of("fn f(x: f32, y: f32) -> f32 { x - y }")
    grads = differentiate_reverse(body, ["x", "y"])
    # ∂f/∂x = 1, ∂f/∂y = -1.
    assert fmt(grads["x"]) == "1", f"got {fmt(grads['x'])}"
    # -1 may render as "-1" or "(-1)"
    out_y = fmt(grads["y"])
    assert "1" in out_y and "-" in out_y, f"got {out_y}"


def test_division_quotient_rule():
    body = _body_of("fn f(x: f32, y: f32) -> f32 { x / y }")
    grads = differentiate_reverse(body, ["x", "y"])
    # ∂f/∂x = 1/y. After simplify: (1/y).
    out_x = fmt(grads["x"])
    assert "/" in out_x and "y" in out_x
    # ∂f/∂y = -x/(y*y).
    out_y = fmt(grads["y"])
    assert "-" in out_y, f"expected negative, got {out_y}"


def test_chain_via_letbinding():
    # f = (x+1)*(x+2); ∂f/∂x = (x+2) + (x+1) = 2x+3
    body = _body_of("""
    fn f(x: f32) -> f32 {
        let a = x + 1.0;
        let b = x + 2.0;
        a * b
    }
    """)
    grads = differentiate_reverse(body, ["x"])
    # After inlining: (x+1)*(x+2). Reverse: adj_l = 1*(x+2), adj_r = 1*(x+1).
    # x appears twice, so summed contributions = (x+2) + (x+1) symbolically.
    out = fmt(grads["x"])
    # The expression should reference x (positive count).
    assert out.count("x") >= 2, f"expected x referenced multiple times, got {out}"


def test_unary_negation():
    body = _body_of("fn f(x: f32) -> f32 { -x }")
    grads = differentiate_reverse(body, ["x"])
    # ∂(-x)/∂x = -1.
    out = fmt(grads["x"])
    assert "-1" in out or out == "(-1)", f"got {out}"


def test_param_not_in_expr():
    # If a parameter is not used in the expression, its gradient is 0.
    body = _body_of("fn f(x: f32, y: f32) -> f32 { x * x }")
    grads = differentiate_reverse(body, ["x", "y"])
    assert fmt(grads["y"]) == "0", f"got {fmt(grads['y'])}"


def test_multi_use_of_param_sums():
    # f = x + x + x; ∂f/∂x = 3. Reverse mode: each + propagates 1 to its
    # operands, so x's bucket is [1, 1, 1]; sum = 3.
    body = _body_of("fn f(x: f32) -> f32 { x + x + x }")
    grads = differentiate_reverse(body, ["x"])
    out = fmt(grads["x"])
    # After simplification 1+1+1 should fold to 3.
    assert out == "3", f"got {out}"


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
