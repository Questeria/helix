"""End-to-end codegen tests: parse Helix source, produce ELF, run, check exit code."""

from __future__ import annotations
import os, sys, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir.passes.const_fold import fold_module
from helixc.ir.passes.dce import dce_module
from helixc.backend.x86_64 import compile_module_to_elf


def compile_and_run(src: str, optimize: bool = True) -> int:
    """Compile Helix source to ELF, run via WSL, return exit code.

    optimize=True (default): runs const-fold + DCE before codegen.
    """
    prog = parse(src)
    mod = lower(prog)
    if optimize:
        fold_module(mod)
        dce_module(mod)
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
