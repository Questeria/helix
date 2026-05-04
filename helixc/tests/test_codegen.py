"""End-to-end codegen tests: parse Helix source, produce ELF, run, check exit code."""

from __future__ import annotations
import os, sys, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend.grad_pass import grad_pass
from helixc.ir.lower_ast import lower
from helixc.ir.passes.const_fold import fold_module
from helixc.ir.passes.dce import dce_module
from helixc.ir.passes.cse import cse_module
from helixc.ir.passes.fdce import fdce_module
from helixc.backend.x86_64 import compile_module_to_elf


def compile_and_run(src: str, optimize: bool = True) -> int:
    """Compile Helix source to ELF, run via WSL, return exit code.

    Pipeline: parse -> grad_pass -> lower -> [opt] -> codegen -> ELF.
    optimize=True (default): runs const-fold + CSE + DCE + fdce before codegen.
    """
    prog = parse(src, include_stdlib=True)
    grad_pass(prog)
    mod = lower(prog)
    if optimize:
        fold_module(mod)
        cse_module(mod)
        dce_module(mod)
        fdce_module(mod)
    elf = compile_module_to_elf(mod)
    # Write to a temp file in the project tree (since WSL accesses /mnt/c)
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(proj_root, "helixc", "tests", "_tmp")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test.bin")
    with open(out_path, "wb") as f:
        f.write(elf)
    os.chmod(out_path, 0o755)
    # Run via WSL
    rel = os.path.relpath(out_path, proj_root).replace("\\", "/")
    wsl_path = f"/mnt/c/Projects/Kovostov-Native/{rel}"
    result = subprocess.run(
        ["wsl", "--", "bash", "-c", f"chmod +x {wsl_path} && {wsl_path}"],
        capture_output=True,
    )
    return result.returncode


def test_exit_zero():
    assert compile_and_run("fn main() -> i32 { 0 }") == 0


def test_exit_42():
    assert compile_and_run("fn main() -> i32 { 42 }") == 42


def test_exit_addition():
    # 17 + 25 = 42
    assert compile_and_run("fn main() -> i32 { 17 + 25 }") == 42


def test_exit_subtraction():
    # 100 - 58 = 42
    assert compile_and_run("fn main() -> i32 { 100 - 58 }") == 42


def test_exit_multiplication():
    # 6 * 7 = 42
    assert compile_and_run("fn main() -> i32 { 6 * 7 }") == 42


def test_let_binding_then_use():
    src = "fn main() -> i32 { let x = 40; let y = 2; x + y }"
    assert compile_and_run(src) == 42


def test_function_call():
    src = """
    fn add(a: i32, b: i32) -> i32 { a + b }
    fn main() -> i32 { add(20, 22) }
    """
    assert compile_and_run(src) == 42


def test_nested_calls():
    src = """
    fn double(x: i32) -> i32 { x + x }
    fn add(a: i32, b: i32) -> i32 { a + b }
    fn main() -> i32 { add(double(15), 12) }
    """
    # double(15) = 30; add(30, 12) = 42
    assert compile_and_run(src) == 42


def test_three_arg_call():
    src = """
    fn sum3(a: i32, b: i32, c: i32) -> i32 { a + b + c }
    fn main() -> i32 { sum3(10, 15, 17) }
    """
    assert compile_and_run(src) == 42


def test_compare_lt_true():
    # (3 < 5) -> 1; +41 -> 42
    src = "fn main() -> i32 { let b = 3 < 5; b + 41 }"
    assert compile_and_run(src) == 42


def test_compare_eq_true():
    src = "fn main() -> i32 { let b = 7 == 7; b + 41 }"
    assert compile_and_run(src) == 42


def test_compare_eq_false():
    # (7 == 8) -> 0; +42 -> 42
    src = "fn main() -> i32 { let b = 7 == 8; b + 42 }"
    assert compile_and_run(src) == 42


def test_compare_gt_false():
    src = "fn main() -> i32 { let b = 3 > 5; b + 42 }"
    assert compile_and_run(src) == 42


def test_if_select_then_branch():
    # condition true -> 42
    src = "fn main() -> i32 { if 1 < 2 { 42 } else { 99 } }"
    assert compile_and_run(src) == 42


def test_if_select_else_branch():
    # condition false -> else
    src = "fn main() -> i32 { if 1 > 2 { 99 } else { 42 } }"
    assert compile_and_run(src) == 42


def test_unary_neg_returns_neg_value():
    # neg(-42) for an i32 in main becomes the exit code (Linux truncates to low 8 bits)
    # Choose a positive result instead: -(-42) = 42
    src = "fn main() -> i32 { -(-42) }"
    assert compile_and_run(src) == 42


def test_six_arg_call():
    # System V ABI: 6 integer args via rdi/rsi/rdx/rcx/r8/r9
    src = """
    fn sum6(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32) -> i32 {
        a + b + c + d + e + f
    }
    fn main() -> i32 { sum6(1, 2, 3, 4, 5, 27) }
    """
    # 1+2+3+4+5+27 = 42
    assert compile_and_run(src) == 42


def test_matmul_2x2_trace():
    # The full 2x2 matmul trace example expressed with let-bindings inline
    src = """
    fn main() -> i32 {
        let a00 = 1; let a01 = 2; let a10 = 3; let a11 = 4;
        let b00 = 5; let b01 = 6; let b10 = 7; let b11 = 8;
        let c00 = a00 * b00 + a01 * b10;
        let c11 = a10 * b01 + a11 * b11;
        c00 + c11
    }
    """
    assert compile_and_run(src) == 69   # 19 + 50


def test_recursion_fib():
    # Fibonacci — the canonical proof that real CFG-based branching works
    # (SELECT-based if would infinite-recurse since it evaluates both arms)
    src = """
    fn fib(n: i32) -> i32 {
        if n < 2 {
            n
        } else {
            fib(n - 1) + fib(n - 2)
        }
    }
    fn main() -> i32 { fib(9) }
    """
    # fib(9) = 34
    assert compile_and_run(src) == 34


def test_recursion_factorial():
    src = """
    fn fact(n: i32) -> i32 {
        if n < 2 { 1 } else { n * fact(n - 1) }
    }
    fn main() -> i32 { fact(5) }
    """
    # 5! = 120
    assert compile_and_run(src) == 120


def test_recursion_count_down():
    # tail-recursive style: counts down to 0 then returns base value
    src = """
    fn count(n: i32, acc: i32) -> i32 {
        if n == 0 { acc } else { count(n - 1, acc + n) }
    }
    fn main() -> i32 { count(7, 0) }
    """
    # 7+6+5+4+3+2+1 = 28
    assert compile_and_run(src) == 28


def test_division():
    # 100 / 7 = 14 (integer division), then * 3 = 42
    src = "fn main() -> i32 { (100 / 7) * 3 }"
    assert compile_and_run(src) == 42


def test_modulo():
    # 100 % 58 = 42
    src = "fn main() -> i32 { 100 % 58 }"
    assert compile_and_run(src) == 42


def test_division_recursive_gcd():
    # Euclidean GCD via recursion
    src = """
    fn gcd(a: i32, b: i32) -> i32 {
        if b == 0 { a } else { gcd(b, a % b) }
    }
    fn main() -> i32 { gcd(126, 84) }
    """
    # gcd(126, 84) = 42
    assert compile_and_run(src) == 42


def test_mutable_let():
    # Mutable variable assignment
    src = """
    fn main() -> i32 {
        let mut x = 10;
        x = 42;
        x
    }
    """
    assert compile_and_run(src) == 42


def test_compound_assign():
    src = """
    fn main() -> i32 {
        let mut x = 10;
        x += 32;
        x
    }
    """
    assert compile_and_run(src) == 42


def test_while_loop_sum():
    # Sum 1..=8 via while loop -> 36 (no, 1+2+3+4+5+6+7+8 = 36, want 42)
    # Sum 1..=9 = 45
    # Sum 0..=8: 36
    # Use a target of 42: e.g., sum from 6 to 14 inclusive = (6+14)*9/2 = 90 (no)
    # Easier: count up to 42
    src = """
    fn main() -> i32 {
        let mut x = 0;
        while x < 42 {
            x += 1;
        }
        x
    }
    """
    assert compile_and_run(src) == 42


def test_while_loop_factorial():
    # Iterative factorial: 5! = 120
    src = """
    fn main() -> i32 {
        let mut n = 5;
        let mut result = 1;
        while n > 1 {
            result *= n;
            n -= 1;
        }
        result
    }
    """
    assert compile_and_run(src) == 120


def test_for_loop_sum():
    src = """
    fn main() -> i32 {
        let mut total = 0;
        for i in 0 .. 10 {
            total += i;
        }
        total
    }
    """
    # 0+1+2+3+4+5+6+7+8+9 = 45
    assert compile_and_run(src) == 45


def test_for_loop_count_to_42():
    src = """
    fn main() -> i32 {
        let mut x = 0;
        for _i in 0 .. 42 {
            x += 1;
        }
        x
    }
    """
    assert compile_and_run(src) == 42


def test_for_loop_nested():
    src = """
    fn main() -> i32 {
        let mut count = 0;
        for i in 0 .. 6 {
            for j in 0 .. 7 {
                count += 1;
            }
        }
        count
    }
    """
    # 6 * 7 = 42
    assert compile_and_run(src) == 42


def test_array_literal_and_index():
    src = """
    fn main() -> i32 {
        let xs = [10, 20, 12];
        xs[0] + xs[1] + xs[2]
    }
    """
    # 10 + 20 + 12 = 42
    assert compile_and_run(src) == 42


def test_array_assign():
    src = """
    fn main() -> i32 {
        let xs = [0, 0, 0];
        xs[0] = 10;
        xs[1] = 32;
        xs[0] + xs[1] + xs[2]
    }
    """
    # 10 + 32 + 0 = 42
    assert compile_and_run(src) == 42


def test_array_loop_sum():
    src = """
    fn main() -> i32 {
        let xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0];
        let mut total = 0;
        for i in 0 .. 10 {
            total += xs[i];
        }
        total
    }
    """
    # 1+2+3+4+5+6+7+8+9+0 = 45 — wait, want 42. Let me change.
    # Use 0+1+2+3+4+5+6+7+8+9 = 45 isn't 42. Use first 9: 0..=8 sum = 36
    # Actually just check sum = 45
    assert compile_and_run(src) == 45


def test_array_compound_assign():
    src = """
    fn main() -> i32 {
        let xs = [10, 0, 0];
        xs[0] += 32;
        xs[0]
    }
    """
    assert compile_and_run(src) == 42


def test_large_array_sum():
    # 32-element array sum stress test
    src = """
    fn main() -> i32 {
        let a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                 21, 22, 23, 24, 25, 26, 27, 28, 29, 30,
                 31, 32];
        let mut total = 0;
        for i in 0 .. 32 {
            total += a[i];
        }
        total
    }
    """
    # 1+2+...+32 = 32*33/2 = 528. Mod 256 = 16 (since exit codes are 8-bit on Linux)
    # Actually Linux exit codes are %256, so we need a value <= 255.
    # Sum of 1..=32 = 528 % 256 = 16.
    assert compile_and_run(src) == 16   # 528 mod 256


def test_float_const_to_int():
    # Simplest float test: cast a constant
    src = "fn main() -> i32 { 42.5 as i32 }"
    # truncating: 42.5 -> 42
    assert compile_and_run(src) == 42


def test_float_addition():
    src = "fn main() -> i32 { (40.0 + 2.0) as i32 }"
    assert compile_and_run(src) == 42


def test_float_multiplication():
    src = "fn main() -> i32 { (6.0 * 7.0) as i32 }"
    assert compile_and_run(src) == 42


def test_float_division():
    # 168.0 / 4.0 = 42.0
    src = "fn main() -> i32 { (168.0 / 4.0) as i32 }"
    assert compile_and_run(src) == 42


def test_float_subtraction():
    src = "fn main() -> i32 { (50.0 - 8.0) as i32 }"
    assert compile_and_run(src) == 42


def test_int_to_float_round_trip():
    # int -> float -> int (no change for representable values)
    src = """
    fn main() -> i32 {
        let x = 42 as f32;
        x as i32
    }
    """
    assert compile_and_run(src) == 42


def test_float_complex_expression():
    # (2.5 * 8.0) + (24.0 / 2.0) + 10.0 = 20.0 + 12.0 + 10.0 = 42.0
    src = "fn main() -> i32 { ((2.5 * 8.0) + (24.0 / 2.0) + 10.0) as i32 }"
    assert compile_and_run(src) == 42


def test_float_with_let_binding():
    src = """
    fn main() -> i32 {
        let pi = 3.14;
        let r = 4.0;
        let area = pi * r * r;
        area as i32
    }
    """
    # 3.14 * 16 = 50.24 -> 50
    assert compile_and_run(src) == 50


def test_float_array_sum():
    src = """
    fn main() -> i32 {
        let xs = [10.5, 11.5, 12.0, 8.0];
        let mut total = 0.0;
        for i in 0 .. 4 {
            total += xs[i];
        }
        total as i32
    }
    """
    # 10.5 + 11.5 + 12.0 + 8.0 = 42.0
    assert compile_and_run(src) == 42


def test_float_dot_product():
    # Real ML kernel: dot product of two vectors
    src = """
    fn main() -> i32 {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 5.0, 6.0, 2.0];
        let mut acc = 0.0;
        for i in 0 .. 4 {
            acc += a[i] * b[i];
        }
        acc as i32
    }
    """
    # 1*4 + 2*5 + 3*6 + 4*2 = 4 + 10 + 18 + 8 = 40 (close, want 42)
    # Adjust to: 1*4 + 2*5 + 3*6 + 5*2 = 4+10+18+10 = 42
    src2 = """
    fn main() -> i32 {
        let a = [1.0, 2.0, 3.0, 5.0];
        let b = [4.0, 5.0, 6.0, 2.0];
        let mut acc = 0.0;
        for i in 0 .. 4 {
            acc += a[i] * b[i];
        }
        acc as i32
    }
    """
    assert compile_and_run(src2) == 42


def test_real_float_matmul_3x3():
    # 3x3 matmul with floats — the actual ML matmul kernel
    src = """
    fn main() -> i32 {
        let a = [1.0, 0.0, 0.0,
                 0.0, 1.0, 0.0,
                 0.0, 0.0, 1.0];
        let b = [14.0, 0.0, 0.0,
                 0.0, 14.0, 0.0,
                 0.0, 0.0, 14.0];
        let c = [0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0];
        for i in 0 .. 3 {
            for j in 0 .. 3 {
                let mut acc = 0.0;
                for k in 0 .. 3 {
                    acc += a[i * 3 + k] * b[k * 3 + j];
                }
                c[i * 3 + j] = acc;
            }
        }
        let mut total = 0.0;
        for i in 0 .. 9 {
            total += c[i];
        }
        total as i32
    }
    """
    # trace = 14+14+14 = 42; total = 42
    assert compile_and_run(src) == 42


def test_quote_basic():
    # quote { ... } produces a stable handle (i64); two identical quotes
    # yield equal handles, different ones differ.
    src = """
    fn main() -> i32 {
        let h1 = quote { 1 + 2 };
        let h2 = quote { 1 + 2 };
        if h1 == h2 { 42 } else { 0 }
    }
    """
    # Wait — h1 and h2 are i64; comparison gives bool; if/else picks branch.
    # But i64 storage in 8-byte slot reads only low 32 bits via mov_eax_mem_rbp.
    # Hashes may differ in upper bits. v0.1: check low 32 bits agree.
    assert compile_and_run(src) == 42


def test_quote_different_asts_differ():
    src = """
    fn main() -> i32 {
        let h1 = quote { 1 + 2 };
        let h2 = quote { 3 * 4 };
        if h1 != h2 { 42 } else { 0 }
    }
    """
    assert compile_and_run(src) == 42


def test_modify_verifier_accepts():
    # modify(target, transformation, verifier) returns 1 if verifier non-zero, else 0
    src = """
    fn main() -> i32 {
        let target = 0;
        let xform = 0;
        let verifier_pass = 1;
        let result = modify(target, xform, verifier_pass);
        if result == 1 { 42 } else { 0 }
    }
    """
    assert compile_and_run(src) == 42


def test_modify_verifier_rejects():
    src = """
    fn main() -> i32 {
        let target = 0;
        let xform = 0;
        let verifier_fail = 0;
        let result = modify(target, xform, verifier_fail);
        if result == 0 { 42 } else { 0 }
    }
    """
    assert compile_and_run(src) == 42


# ============================================================================
# grad(f) as a real language builtin
# ============================================================================
def test_grad_simple_quadratic():
    # d(x*x)/dx at x=21 = 42
    src = """
    fn loss(x: f32) -> f32 { x * x }
    fn main() -> i32 {
        grad(loss)(21.0) as i32
    }
    """
    assert compile_and_run(src) == 42


def test_grad_with_let_bindings():
    # d/dx (2x-4)^2 at x=5 = 8*5 - 16 = 24; +18 = 42
    src = """
    fn loss(x: f32) -> f32 {
        let pred = x * 2.0 + 3.0;
        let target = 7.0;
        let diff = pred - target;
        diff * diff
    }
    fn main() -> i32 {
        let g = grad(loss)(5.0);
        (g + 18.0) as i32
    }
    """
    assert compile_and_run(src) == 42


def test_grad_linear():
    # d(3x + 5y)/dx = 3 (constant); call at x=0 -> 3; +39 = 42
    src = """
    fn linear(x: f32, y: f32) -> f32 { 3.0 * x + 5.0 * y }
    fn main() -> i32 {
        // explicit index 0 -> differentiate w.r.t. x
        let g = grad(linear, 0)(0.0, 0.0);
        (g + 39.0) as i32
    }
    """
    assert compile_and_run(src) == 42


def test_grad_linear_second_param():
    # d(3x + 5y)/dy = 5; call at any x,y -> 5; +37 = 42
    src = """
    fn linear(x: f32, y: f32) -> f32 { 3.0 * x + 5.0 * y }
    fn main() -> i32 {
        // explicit index 1 -> differentiate w.r.t. y
        let g = grad(linear, 1)(2.0, 9.0);
        (g + 37.0) as i32
    }
    """
    assert compile_and_run(src) == 42


def test_grad_multi_param_without_index_errors():
    # grad(f) on a multi-param function must raise, not silently use param 0
    src = """
    fn linear(x: f32, y: f32) -> f32 { 3.0 * x + 5.0 * y }
    fn main() -> i32 {
        let g = grad(linear)(0.0, 0.0);
        g as i32
    }
    """
    try:
        compile_and_run(src)
    except ValueError as e:
        assert "ambiguous" in str(e)
        return
    raise AssertionError("expected grad(multi_param) to error, but it succeeded")


def test_grad_let_aliased():
    # 'let f = grad(loss); f(x)' should work: f is aliased to loss__grad
    src = """
    fn loss(x: f32) -> f32 { x * x }
    fn main() -> i32 {
        let f = grad(loss);
        f(21.0) as i32
    }
    """
    assert compile_and_run(src) == 42


def test_grad_grad_quadratic():
    # d^2(x*x)/dx^2 = 2 (constant); call at any x -> 2; +40 = 42
    src = """
    fn loss(x: f32) -> f32 { x * x }
    fn main() -> i32 {
        let g2 = grad(grad(loss))(7.0);
        (g2 + 40.0) as i32
    }
    """
    assert compile_and_run(src) == 42


# ============================================================================
# Reverse-mode AD via grad_rev(f) / grad_rev(f, n)
# ============================================================================
def test_grad_rev_simple_quadratic():
    # d(x*x)/dx at x=21 = 42, computed via reverse-mode
    src = """
    fn loss(x: f32) -> f32 { x * x }
    fn main() -> i32 {
        grad_rev(loss)(21.0) as i32
    }
    """
    assert compile_and_run(src) == 42


def test_grad_rev_linear_two_params():
    # d(3x + 5y)/dx = 3 (constant); call at x=0 -> 3; +39 = 42
    src = """
    fn linear(x: f32, y: f32) -> f32 { 3.0 * x + 5.0 * y }
    fn main() -> i32 {
        let gx = grad_rev(linear, 0)(0.0, 0.0);
        let gy = grad_rev(linear, 1)(0.0, 0.0);
        (gx + gy + 34.0) as i32
    }
    """
    # gx=3, gy=5, +34 = 42
    assert compile_and_run(src) == 42


def test_grad_rev_sharing_via_let():
    # f(x) = let a = x+1; let b = x+2; a*b
    # f(3) = 4*5 = 20; ∂f/∂x = (x+2) + (x+1) = 2x+3; at x=3 -> 9
    # 9 + 33 = 42
    src = """
    fn f(x: f32) -> f32 {
        let a = x + 1.0;
        let b = x + 2.0;
        a * b
    }
    fn main() -> i32 {
        let g = grad_rev(f)(3.0);
        (g + 33.0) as i32
    }
    """
    assert compile_and_run(src) == 42


def test_grad_rev_equivalent_to_grad_for_simple_cases():
    # For pure expressions both engines should produce the same numeric result.
    src_fwd = """
    fn f(x: f32, y: f32) -> f32 { x * x + 2.0 * x * y + y * y }
    fn main() -> i32 {
        // ∂f/∂x at (3, 4) = 2x + 2y = 14
        let g = grad(f, 0)(3.0, 4.0);
        (g + 28.0) as i32
    }
    """
    src_rev = """
    fn f(x: f32, y: f32) -> f32 { x * x + 2.0 * x * y + y * y }
    fn main() -> i32 {
        let g = grad_rev(f, 0)(3.0, 4.0);
        (g + 28.0) as i32
    }
    """
    fwd = compile_and_run(src_fwd)
    rev = compile_and_run(src_rev)
    assert fwd == rev == 42, f"forward={fwd}, reverse={rev}"


def test_grad_rev_multi_param_without_index_errors():
    # grad_rev(f) on multi-param must raise, just like grad
    src = """
    fn f(x: f32, y: f32) -> f32 { x + y }
    fn main() -> i32 {
        let g = grad_rev(f)(0.0, 0.0);
        g as i32
    }
    """
    try:
        compile_and_run(src)
    except ValueError as e:
        assert "ambiguous" in str(e)
        return
    raise AssertionError("expected grad_rev(multi_param) to error")


def test_grad_through_if_takes_correct_branch():
    # f(x) = if x > 0 then x*x else x*3
    # ∂f/∂x at x=5 = 2x = 10. (Was previously broken: the autodiff would
    # always take the then branch's derivative even at x=-1, ignoring else.)
    src = """
    fn f(x: f32) -> f32 {
        if x > 0.0 { x * x } else { x * 3.0 }
    }
    fn main() -> i32 {
        let g = grad(f)(5.0);
        (g + 32.0) as i32
    }
    """
    # 10 + 32 = 42
    assert compile_and_run(src) == 42


def test_grad_through_if_else_branch():
    # Same f as above. At x=-1 we should hit the else branch: ∂(x*3)/∂x = 3.
    # 3 + 39 = 42
    src = """
    fn f(x: f32) -> f32 {
        if x > 0.0 { x * x } else { x * 3.0 }
    }
    fn main() -> i32 {
        let g = grad(f)(-1.0);
        (g + 39.0) as i32
    }
    """
    assert compile_and_run(src) == 42


def test_grad_rev_through_if_then_branch():
    # Reverse-mode through an if: must produce conditional gradient,
    # NOT a sum of both branches' adjoints.
    # f(x) = if x > 0 then x*x else x*3
    # At x=5 (then branch): ∂f/∂x = 2x = 10. 10 + 32 = 42.
    src = """
    fn f(x: f32) -> f32 {
        if x > 0.0 { x * x } else { x * 3.0 }
    }
    fn main() -> i32 {
        let g = grad_rev(f)(5.0);
        (g + 32.0) as i32
    }
    """
    assert compile_and_run(src) == 42


def test_grad_rev_through_if_else_branch():
    # At x=-1 (else branch): ∂f/∂x = 3. 3 + 39 = 42.
    src = """
    fn f(x: f32) -> f32 {
        if x > 0.0 { x * x } else { x * 3.0 }
    }
    fn main() -> i32 {
        let g = grad_rev(f)(-1.0);
        (g + 39.0) as i32
    }
    """
    assert compile_and_run(src) == 42


def test_idiv_int_min_div_neg_one_does_not_trap():
    # x86 `idiv ecx` raises #DE for INT_MIN / -1. We emit a runtime guard
    # that defines INT_MIN / -1 = INT_MIN and INT_MIN % -1 = 0 instead of
    # crashing. To exercise the runtime path (not const_fold) we hide the
    # operands behind function args.
    src = """
    fn divide(a: i32, b: i32) -> i32 { a / b }
    fn modulo(a: i32, b: i32) -> i32 { a % b }
    fn main() -> i32 {
        let int_min = -2147483648;
        let q = divide(int_min, -1);
        let r = modulo(int_min, -1);
        // q should be INT_MIN; r should be 0. INT_MIN as exit code wraps
        // around; verify by reducing to a small known value.
        let q_ok = if q == int_min { 1 } else { 0 };
        let r_ok = if r == 0 { 1 } else { 0 };
        // 1 + 1 + 40 = 42 if both pass, else < 42
        q_ok + r_ok + 40
    }
    """
    assert compile_and_run(src) == 42


def test_idiv_normal_division_still_works():
    # Make sure the new guard didn't break normal division.
    src = """
    fn divide(a: i32, b: i32) -> i32 { a / b }
    fn main() -> i32 { divide(84, 2) }
    """
    assert compile_and_run(src) == 42


def test_interleaved_int_float_params_calling_convention():
    # SysV ABI splits args by class: int -> (edi, esi, edx, ecx, r8d, r9d),
    # float -> (xmm0..xmm7). Each class has its own counter; they don't
    # share registers. Test interleaved signatures to ensure the prologue
    # spill and the call-site arg load route each arg to the right register.
    src = """
    fn mix(a: i32, b: f32, c: i32, d: f32) -> i32 {
        // SysV: a -> edi, b -> xmm0, c -> esi, d -> xmm1.
        // Compute: a + (b as i32) + c + (d as i32)  i.e. 1 + 10 + 2 + 20 = 33
        let bf = b as i32;
        let df = d as i32;
        a + bf + c + df
    }
    fn main() -> i32 {
        let r = mix(1, 10.5, 2, 20.5);
        // r should be 1 + 10 + 2 + 20 = 33; +9 = 42
        r + 9
    }
    """
    assert compile_and_run(src) == 42


def test_interleaved_float_int_returns_float():
    # Another permutation, with float return.
    src = """
    fn mix2(a: f32, b: i32, c: f32, d: i32) -> f32 {
        // SysV: a -> xmm0, b -> edi, c -> xmm1, d -> esi.
        // Compute a + (b as f32) + c + (d as f32) = 1.5 + 10 + 2.5 + 20 = 34.0
        a + (b as f32) + c + (d as f32)
    }
    fn main() -> i32 {
        let r = mix2(1.5, 10, 2.5, 20);
        (r as i32) + 8
    }
    """
    # 34 + 8 = 42
    assert compile_and_run(src) == 42


def test_float_compare_with_negative_values():
    # Earlier the compiler used signed integer cmp on float bit patterns,
    # silently miscompiling negative-value comparisons. Now uses ucomiss.
    cases = [
        ("fn main() -> i32 { let a = -2.0; if a < -0.001 { 42 } else { 0 } }", 42),
        ("fn main() -> i32 { let a = -2.0; if a > -0.001 { 0 } else { 42 } }", 42),
        ("fn main() -> i32 { let a = -5.0; let b = -10.0; if a > b { 42 } else { 0 } }", 42),
        ("fn main() -> i32 { let a = -1.0; let b = -1.0; if a == b { 42 } else { 0 } }", 42),
        ("fn main() -> i32 { let a = -1.5; let b = 2.5; if a < b { 42 } else { 0 } }", 42),
    ]
    for src, expected in cases:
        got = compile_and_run(src)
        assert got == expected, f"src={src!r}: got {got}, expected {expected}"


def test_real_matmul_3x3_via_arrays():
    # 3x3 matmul: c[i][j] = sum_k a[i][k] * b[k][j]
    # We use flat 9-element arrays with row-major indexing: a[i*3+j]
    # A = identity, B = identity -> A*B = identity, sum = 3
    # Better: A = [[1,0,0],[0,1,0],[0,0,1]] B = [[14,0,0],[0,14,0],[0,0,14]]
    #         A*B = 14*identity, sum = 42
    src = """
    fn main() -> i32 {
        let a = [1, 0, 0, 0, 1, 0, 0, 0, 1];
        let b = [14, 0, 0, 0, 14, 0, 0, 0, 14];
        let c = [0, 0, 0, 0, 0, 0, 0, 0, 0];
        for i in 0 .. 3 {
            for j in 0 .. 3 {
                let mut acc = 0;
                for k in 0 .. 3 {
                    acc += a[i * 3 + k] * b[k * 3 + j];
                }
                c[i * 3 + j] = acc;
            }
        }
        let mut total = 0;
        for i in 0 .. 9 {
            total += c[i];
        }
        total
    }
    """
    # trace: 14 + 14 + 14 = 42. Whole sum: 42 (only diagonal nonzero).
    assert compile_and_run(src) == 42


def test_or_chain_normalized_to_bool():
    """Without `||` result normalization, `1 || 1` lowered as ADD = 2,
    and `(a || b) == 1` would silently fail. With ADD-then-CMP_NE, the
    result is strictly 0 or 1 again."""
    src = """
    fn main() -> i32 {
        let a = 1 == 1;
        let b = 1 == 1;
        let c = a || b;
        if c == true { 42 } else { 0 }
    }
    """
    code = compile_and_run(src)
    assert code == 42, f"expected 42 (c should be a strict bool 1), got {code}"


def test_chained_ors_normalized():
    """Multiple ||'s in a row — without normalization, repeated ADD
    accumulates to ≥3 quickly, breaking any downstream equality check."""
    src = """
    fn main() -> i32 {
        let a = 1 == 1;
        let b = 1 == 1;
        let c = 1 == 1;
        let d = 1 == 1;
        let r = a || b || c || d;
        if r == true { 42 } else { 0 }
    }
    """
    code = compile_and_run(src)
    assert code == 42, f"expected 42 (r should be a strict bool 1), got {code}"


def test_dump_ast_hashes_flag():
    """`autodiff_cli --dump-ast-hashes <file>` prints `<fn> : <hex12>`
    deterministically across two runs on the same input."""
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src_dir = os.path.join(proj_root, "helixc", "tests", "_tmp")
    os.makedirs(src_dir, exist_ok=True)
    src_path = os.path.join(src_dir, "_dump_hashes.hx")
    with open(src_path, "w", encoding="utf-8") as f:
        f.write("""
fn f(x: f32) -> f32 { x * x }
fn g(y: f32) -> f32 { y + 1.0 }
""")
    cmd = [sys.executable, "-m", "helixc.frontend.autodiff_cli",
           "--dump-ast-hashes", src_path]
    r1 = subprocess.run(cmd, capture_output=True, cwd=proj_root)
    assert r1.returncode == 0, r1.stderr
    r2 = subprocess.run(cmd, capture_output=True, cwd=proj_root)
    assert r2.returncode == 0, r2.stderr
    out1 = r1.stdout.decode("utf-8").strip().splitlines()
    out2 = r2.stdout.decode("utf-8").strip().splitlines()
    assert out1 == out2, f"hashes not stable: {out1!r} vs {out2!r}"
    assert len(out1) == 2
    for line in out1:
        name, _, h = line.partition(" : ")
        assert name in ("f", "g"), f"unexpected fn {name!r}"
        assert len(h) == 12 and all(c in "0123456789abcdef" for c in h), \
            f"bad hash: {h!r}"


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
