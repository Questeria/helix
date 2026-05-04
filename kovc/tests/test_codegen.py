"""End-to-end codegen tests: parse Kov source, produce ELF, run, check exit code."""

from __future__ import annotations
import os, sys, subprocess, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kovc.frontend.parser import parse
from kovc.ir.lower_ast import lower
from kovc.backend.x86_64 import compile_module_to_elf


def compile_and_run(src: str) -> int:
    """Compile Kov source to ELF, run via WSL, return exit code."""
    prog = parse(src)
    mod = lower(prog)
    elf = compile_module_to_elf(mod)
    # Write to a temp file in the project tree (since WSL accesses /mnt/c)
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(proj_root, "kovc", "tests", "_tmp")
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
