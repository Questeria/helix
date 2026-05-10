"""Stage 16.5 — FFI / extern "C" tests.

End-to-end: a Helix program that imports `puts` from libc.so.6, builds
a *const u8 from a string literal, and calls puts. The dynamic linker
resolves the GOT slot at exec time (BIND_NOW). The binary is produced
by helixc-Python and run via WSL.
"""

from __future__ import annotations
import os
import subprocess
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.backend.x86_64 import compile_module_to_elf


_PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _wsl_root() -> str:
    """Return the /mnt/c/... mount of the project root that the test was
    invoked from. Works in both the main checkout and any worktree under
    C:\\Projects\\Kovostov-Native*\\."""
    rel = _PROJ_ROOT.replace("C:\\", "/mnt/c/").replace("\\", "/")
    return rel


def _build_and_run(src: str, name: str = "ffi.bin") -> tuple[int, str, str]:
    prog = parse(src, include_stdlib=False)
    mod = lower(prog)
    elf = compile_module_to_elf(mod)
    out_dir = os.path.join(_PROJ_ROOT, "helixc", "tests", "_tmp")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, name)
    with open(out, "wb") as f:
        f.write(elf)
    rel = os.path.relpath(out, _PROJ_ROOT).replace("\\", "/")
    wsl_path = f"{_wsl_root()}/{rel}"
    p = subprocess.run(
        ["wsl", "--", "bash", "-c", f"chmod +x {wsl_path} && {wsl_path}"],
        capture_output=True, text=True, timeout=10,
    )
    return p.returncode, p.stdout, p.stderr


def test_extern_c_puts_hello():
    """Stage 16.5 hero test from the detailed plan: puts("hello\\0") returns 6
    and prints "hello\\n" (puts adds the newline). Exercises:
      - extern "C" fn parsing
      - *const u8 type
      - "literal".as_ptr() — STR_PTR op
      - FFI_CALL through .got.plt slot resolved by dynamic linker (BIND_NOW)
      - ELF dyn-link sections produced by helixc/backend/elf_dyn.py
      - libc `exit()` invoked from the entry stub so stdout flushes
        before the process exits.
    """
    src = """
    extern "C" fn puts(s: *const u8) -> i32;
    fn main() -> i32 {
        let msg: *const u8 = "hello\\0".as_ptr();
        puts(msg)
    }
    """
    rc, out, err = _build_and_run(src, name="ffi_puts.bin")
    assert rc == 6, f"expected 6 (puts returns char count), got {rc}; stderr={err!r}"
    assert out == "hello\n", f"expected 'hello\\n', got {out!r}; stderr={err!r}"


def test_extern_c_no_op_no_dynlink():
    """Sanity: a program WITHOUT any extern fn calls still produces a
    libc-free statically-linked ELF (so the existing libc-free path is
    not regressed by Stage 16.5)."""
    src = "fn main() -> i32 { 7 }"
    prog = parse(src, include_stdlib=False)
    mod = lower(prog)
    elf = compile_module_to_elf(mod)
    # Single PT_LOAD, no PT_INTERP / PT_DYNAMIC — only e_phnum == 1.
    e_phnum = int.from_bytes(elf[0x38:0x3A], "little")
    assert e_phnum == 1, f"libc-free path should produce 1 phdr, got {e_phnum}"


def test_extern_c_uses_dynlink():
    """Sanity: a program WITH an extern fn call produces a dynamically-
    linked ELF (PT_INTERP + PT_DYNAMIC + 4 phdrs total)."""
    src = """
    extern "C" fn puts(s: *const u8) -> i32;
    fn main() -> i32 {
        let m: *const u8 = "x\\0".as_ptr();
        puts(m)
    }
    """
    prog = parse(src, include_stdlib=False)
    mod = lower(prog)
    elf = compile_module_to_elf(mod)
    e_phnum = int.from_bytes(elf[0x38:0x3A], "little")
    assert e_phnum == 4, f"FFI binary should have 4 phdrs, got {e_phnum}"


if __name__ == "__main__":
    tests = [
        ("test_extern_c_puts_hello", test_extern_c_puts_hello),
        ("test_extern_c_no_op_no_dynlink", test_extern_c_no_op_no_dynlink),
        ("test_extern_c_uses_dynlink", test_extern_c_uses_dynlink),
    ]
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
