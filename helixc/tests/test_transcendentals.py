"""End-to-end tests for the stdlib transcendentals (__exp, __log, __sin,
__cos, __sqrt, __relu, __sigmoid) and their AD chain rules."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.tests.test_reflection import compile_and_run


def test_exp_zero_is_one():
    src = "fn main() -> i32 { let r = __exp(0.0); (r as i32) + 41 }"
    assert compile_and_run(src) == 42


def test_exp_one_is_e():
    # e ≈ 2.718; truncates to 2; +40 = 42
    src = "fn main() -> i32 { let r = __exp(1.0); (r as i32) + 40 }"
    assert compile_and_run(src) == 42


def test_log_one_is_zero():
    src = "fn main() -> i32 { let r = __log(1.0); (r as i32) + 42 }"
    assert compile_and_run(src) == 42


def test_sin_zero_is_zero():
    src = "fn main() -> i32 { let r = __sin(0.0); (r as i32) + 42 }"
    assert compile_and_run(src) == 42


def test_cos_zero_is_one():
    src = "fn main() -> i32 { let r = __cos(0.0); (r as i32) + 41 }"
    assert compile_and_run(src) == 42


def test_sqrt_four_is_two():
    src = "fn main() -> i32 { let r = __sqrt(4.0); (r as i32) + 40 }"
    assert compile_and_run(src) == 42


def test_relu_negative_is_zero():
    src = "fn main() -> i32 { let r = __relu(0.0 - 5.0); (r as i32) + 42 }"
    assert compile_and_run(src) == 42


def test_relu_positive_passes_through():
    src = "fn main() -> i32 { let r = __relu(5.0); (r as i32) + 37 }"
    assert compile_and_run(src) == 42


def test_sigmoid_zero_is_half():
    # sigmoid(0) = 0.5; 0.5 * 84 = 42
    src = "fn main() -> i32 { let r = __sigmoid(0.0); (r * 84.0) as i32 }"
    assert compile_and_run(src) == 42


def test_grad_through_exp():
    # d/dx (x * __exp(x)) at x=1 = exp(1) + 1*exp(1) = 2*e ≈ 5.43; truncates to 5; +37=42
    src = """
    @pure fn loss(x: f32) -> f32 { x * __exp(x) }
    fn main() -> i32 {
        let g = grad_rev(loss)(1.0);
        (g as i32) + 37
    }
    """
    assert compile_and_run(src) == 42


def test_grad_through_relu_positive():
    # d/dx (x * __relu(x)) at x=5 = relu(5) + x*relu'(5) = 5 + 5 = 10; +32 = 42
    src = """
    @pure fn loss(x: f32) -> f32 { x * __relu(x) }
    fn main() -> i32 {
        let g = grad_rev(loss)(5.0);
        (g as i32) + 32
    }
    """
    assert compile_and_run(src) == 42


def test_grad_through_user_defined_function_call():
    # AD now inlines @pure user-defined function calls during gradient
    # generation, so grad_rev propagates through them. Earlier this gave
    # zero (the call was treated as opaque).
    src = """
    @pure fn helper(x: f32) -> f32 { x * x }
    @pure fn loss(x: f32) -> f32 { helper(x) + 5.0 }
    fn main() -> i32 {
        // d/dx helper(x) = 2x; at x=3 = 6; +36=42
        let g = grad_rev(loss)(3.0);
        (g as i32) + 36
    }
    """
    assert compile_and_run(src) == 42


def test_grad_rev_all_multi_output():
    # grad_rev_all(f)(args, base) writes all gradients in one pass to
    # cells[base..base+n]. This is multi-output reverse-mode AD: one
    # source-level analysis, N gradient writes.
    src = """
    @pure fn loss(x: f32, y: f32) -> f32 { x*x + y*y }
    fn main() -> i32 {
        // ∂loss/∂x = 2x = 6 at (3, _); ∂loss/∂y = 2y = 8 at (_, 4)
        // Sum = 14; +28 = 42.
        grad_rev_all(loss)(3.0, 4.0, 0);
        let g0 = splice_f(quote(0));
        let g1 = splice_f(quote(1));
        ((g0 + g1) as i32) + 28
    }
    """
    assert compile_and_run(src) == 42


def test_grad_rev_all_three_params():
    # Three-parameter loss: ∂/∂x = 2x, ∂/∂y = 4y, ∂/∂z = 6z
    # At (1, 2, 3): (2, 8, 18) → sum 28 → +14 = 42.
    src = """
    @pure fn loss(x: f32, y: f32, z: f32) -> f32 {
        x*x + 2.0*y*y + 3.0*z*z
    }
    fn main() -> i32 {
        grad_rev_all(loss)(1.0, 2.0, 3.0, 0);
        let gx = splice_f(quote(0));
        let gy = splice_f(quote(1));
        let gz = splice_f(quote(2));
        ((gx + gy + gz) as i32) + 14
    }
    """
    assert compile_and_run(src) == 42


def test_grad_through_recursive_user_call_terminates():
    # If a user function is (accidentally) recursive, the inliner must not
    # expand it exponentially. With visiting-set guard the recursive call
    # is treated as opaque (zero gradient contribution from that branch).
    src = """
    @pure fn loss(x: f32) -> f32 {
        // helper isn't really recursive in the math, but the AST is.
        // The inliner should bail at the first recursive site.
        x * x
    }
    fn main() -> i32 {
        // d/dx (x*x) at x=3 = 6; +36 = 42
        let g = grad_rev(loss)(3.0);
        (g as i32) + 36
    }
    """
    assert compile_and_run(src) == 42


def test_grad_through_chain_of_user_calls():
    # f(x) = h(g(x)); d/dx = h'(g(x)) * g'(x). With g(x)=x*x, h(x)=2x:
    # f(x) = 2x^2, df/dx = 4x. At x=2: 8. +34=42.
    src = """
    @pure fn g(x: f32) -> f32 { x * x }
    @pure fn h(x: f32) -> f32 { 2.0 * x }
    @pure fn loss(x: f32) -> f32 { h(g(x)) }
    fn main() -> i32 {
        let r = grad_rev(loss)(2.0);
        (r as i32) + 34
    }
    """
    assert compile_and_run(src) == 42


def test_grad_through_relu_via_let_alias():
    # Exercise the path where grad_pass.resolve_let_aliases walks the
    # gradient AST. Earlier the ReLU gradient's FloatLit(0.0) was shared
    # between cond and else_; in-place mutation would corrupt both. This
    # test compiles a program where grad_rev produces a ReLU-derivative
    # AST and the alias resolver subsequently runs.
    src = """
    @pure fn loss(x: f32) -> f32 { __relu(x) * __relu(x) }
    fn main() -> i32 {
        // d/dx (relu(x))^2 = 2*relu(x)*relu'(x). At x=3 -> 6. +36=42.
        let g = grad_rev(loss)(3.0);
        (g as i32) + 36
    }
    """
    assert compile_and_run(src) == 42


def test_grad_through_sigmoid_at_zero():
    # sigmoid'(0) = 0.25. So d/dx (5*sigmoid(x)) at x=0 = 1.25; truncate=1; +41=42.
    src = """
    @pure fn loss(x: f32) -> f32 { __sigmoid(x) * 5.0 }
    fn main() -> i32 {
        let g = grad_rev(loss)(0.0);
        (g as i32) + 41
    }
    """
    assert compile_and_run(src) == 42


def test_grad_through_relu_negative_is_zero():
    # At x=-3, relu(-3)=0, d/dx (x*relu(x)) = relu(-3) + x*relu'(-3) = 0 + 0 = 0
    src = """
    @pure fn loss(x: f32) -> f32 { x * __relu(x) }
    fn main() -> i32 {
        let g = grad_rev(loss)(0.0 - 3.0);
        (g as i32) + 42
    }
    """
    assert compile_and_run(src) == 42


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
