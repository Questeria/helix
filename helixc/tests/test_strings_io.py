"""Tests for string literals + print_str + write_file builtins."""

from __future__ import annotations
import os, sys, subprocess, tempfile, shlex
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend.grad_pass import grad_pass
from helixc.ir.lower_ast import lower
from helixc.ir.passes.fdce import fdce_module
from helixc.backend.x86_64 import compile_module_to_elf


def _win_to_wsl(win_path: str) -> str:
    p = os.path.abspath(win_path).replace("\\", "/")
    if len(p) >= 2 and p[1] == ":":
        drive = p[0].lower()
        rest = p[2:]
        if not rest.startswith("/"):
            rest = "/" + rest
        return f"/mnt/{drive}{rest}"
    return p


def _build_and_run(src: str) -> tuple[int, str, str]:
    prog = parse(src, include_stdlib=True)
    grad_pass(prog)
    mod = lower(prog)
    fdce_module(mod)
    elf = compile_module_to_elf(mod)
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out_dir = os.path.join(proj_root, "helixc", "tests", "_tmp")
    os.makedirs(out_dir, exist_ok=True)
    fd, out = tempfile.mkstemp(prefix="io_", suffix=".bin", dir=out_dir)
    with os.fdopen(fd, "wb") as f:
        f.write(elf)
    wsl_path = shlex.quote(_win_to_wsl(out))
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


def test_read_file_int_round_trips():
    # Setup: write a known 4-byte integer to a temp file via WSL python.
    setup = subprocess.run(
        ["wsl", "--", "bash", "-c",
         'python3 -c "open(\\"/tmp/helix_rt.bin\\", \\"wb\\").write((42).to_bytes(4, \\"little\\"))"'],
        capture_output=True, text=True
    )
    assert setup.returncode == 0, setup.stderr

    src = '''
    fn main() -> i32 {
        read_file_int("/tmp/helix_rt.bin")
    }
    '''
    rc, _, _ = _build_and_run(src)
    assert rc == 42


def test_read_file_int_endianness_is_little_endian():
    # Disambiguates LE from BE: byte sequence 01 00 00 00 reads as 1 LE
    # but 16777216 BE. We want LE (which is what x86-64 native i32 uses).
    setup = subprocess.run(
        ["wsl", "--", "bash", "-c",
         'python3 -c "open(\\"/tmp/helix_le.bin\\", \\"wb\\").write(bytes([1, 0, 0, 0]))"'],
        capture_output=True, text=True
    )
    assert setup.returncode == 0
    src = '''
    fn main() -> i32 {
        let v = read_file_int("/tmp/helix_le.bin");
        v + 41  // v=1 (LE), result 42; v=16777216 (BE) would overflow exit code
    }
    '''
    rc, _, _ = _build_and_run(src)
    assert rc == 42, f"got {rc}; if 0 or large, endianness is wrong"


def test_read_file_int_missing_file_returns_zero():
    src = '''
    fn main() -> i32 {
        let r = read_file_int("/tmp/this_file_does_not_exist_xyz.bin");
        r + 42  // r should be 0; result 42
    }
    '''
    rc, _, _ = _build_and_run(src)
    assert rc == 42


def test_print_int_decimal_output():
    """`print_int(2026)` writes the digits "2026" to stdout."""
    src = """
    fn main() -> i32 {
        print_int(2026);
        0
    }
    """
    rc, out, _ = _build_and_run(src)
    assert rc == 0
    assert "2026" in out, f"got {out!r}"


def test_print_int_zero():
    """`print_int(0)` writes "0" — the loop must run once even when value=0."""
    src = """
    fn main() -> i32 {
        print_int(0);
        0
    }
    """
    rc, out, _ = _build_and_run(src)
    assert rc == 0
    assert "0" in out, f"got {out!r}"


def test_print_int_negative():
    """`print_int(-42)` writes "-42"."""
    src = """
    fn main() -> i32 {
        print_int(0 - 42);
        0
    }
    """
    rc, out, _ = _build_and_run(src)
    assert rc == 0
    assert "-42" in out, f"got {out!r}"


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
