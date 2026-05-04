"""Tests for string literals + print_str + write_file builtins."""

from __future__ import annotations
import os, sys, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend.grad_pass import grad_pass
from helixc.ir.lower_ast import lower
from helixc.ir.passes.fdce import fdce_module
from helixc.backend.x86_64 import compile_module_to_elf


def _build_and_run(src: str) -> tuple[int, str, str]:
    prog = parse(src, include_stdlib=True)
    grad_pass(prog)
    mod = lower(prog)
    fdce_module(mod)
    elf = compile_module_to_elf(mod)
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(proj_root, "helixc", "tests", "_tmp")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "io.bin")
    with open(out, "wb") as f:
        f.write(elf)
    rel = os.path.relpath(out, proj_root).replace("\\", "/")
    wsl_path = f"/mnt/c/Projects/Kovostov-Native/{rel}"
    p = subprocess.run(
        ["wsl", "--", "bash", "-c", f"chmod +x {wsl_path} && {wsl_path}"],
        capture_output=True, text=True, timeout=10
    )
    return p.returncode, p.stdout, p.stderr


def test_print_str_writes_to_stdout():
    src = """
    fn main() -> i32 {
        print_str("hello\\n");
        42
    }
    """
    rc, out, _ = _build_and_run(src)
    assert rc == 42
    assert out == "hello\n", f"got {out!r}"


def test_print_str_multiple_calls():
    src = """
    fn main() -> i32 {
        print_str("foo");
        print_str("bar");
        print_str("\\n");
        42
    }
    """
    rc, out, _ = _build_and_run(src)
    assert rc == 42
    assert out == "foobar\n"


def test_write_file_creates_file():
    # Write a known string to /tmp, then verify by exit code.
    # The write_file call returns 0 on success.
    src = """
    fn main() -> i32 {
        let r = write_file("/tmp/helix_unittest.txt", "test_content");
        if r == 0 { 42 } else { 1 }
    }
    """
    rc, _, _ = _build_and_run(src)
    assert rc == 42

    # Verify the file content via WSL cat
    p = subprocess.run(
        ["wsl", "--", "cat", "/tmp/helix_unittest.txt"],
        capture_output=True, text=True
    )
    assert p.returncode == 0
    assert p.stdout == "test_content"


def test_hello_world_example_runs():
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    p = os.path.join(proj_root, "helixc", "examples", "hello_world.hx")
    with open(p) as f:
        src = f.read()
    rc, out, _ = _build_and_run(src)
    assert rc == 42
    assert "Hello from Helix!" in out


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
