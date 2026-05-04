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
