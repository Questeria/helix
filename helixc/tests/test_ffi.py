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
    """Return the /mnt/<drive>/... mount of the project root that the test
    was invoked from. Works in both the main checkout and any worktree
    regardless of which drive the repo lives on. Audit 28.8 C1-M3 fix:
    previously hard-coded `C:\\` → `/mnt/c/`; now we derive the drive
    letter from the path so D-drive checkouts (or any other) also work."""
    import pathlib
    p = pathlib.Path(_PROJ_ROOT)
    drive = p.drive  # e.g. "C:" — empty on POSIX
    if drive and len(drive) >= 2 and drive[1] == ":":
        # Windows-style absolute path: build /mnt/<letter>/... form.
        rest = _PROJ_ROOT[len(drive):].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive[0].lower()}/{rest}"
    # POSIX path (e.g. native Linux): use as-is.
    return _PROJ_ROOT.replace("\\", "/")


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


def test_c76_1_ffi_call_routes_f32_args_to_xmm0():
    """Stage 28.9 cycle 77 C76-1 + cycle 79 C78-1 regression (HIGH conf
    80+85): extern "C" fn with a f32 arg must route through xmm0
    (SysV float-class), not edi. Pre-cycle-77 FFI_CALL routed every
    operand through INT_REGS, so the f32 was silently re-bit-cast as
    i32 in edi — callee read garbage. Cycle-77 fixed the arg side;
    cycle-79 fixed the return side (movss xmm0 -> slot instead of
    mov eax -> slot). This test inspects emitted bytes for the
    expected `movss` mnemonics — a future regression to the all-INT
    path would lose them."""
    src = """
    extern "C" fn sinf(x: f32) -> f32;
    fn entry(x: f32) -> f32 {
        sinf(x)
    }
    fn main() -> i32 { 0 }
    """
    prog = parse(src, include_stdlib=False)
    mod = lower(prog)
    elf = compile_module_to_elf(mod)
    # movss xmm0, [rbp-N] — opcode `F3 0F 10` per SysV f32 load
    # convention; the cycle-77 arg-side emits exactly this for the
    # first float arg before `call`. Bare assertion: the binary
    # actually contains the f32-load opcode somewhere.
    assert b"\xf3\x0f\x10" in elf, (
        "FFI f32 arg load to xmm0 (movss xmm0, [rbp-N]) missing — "
        "C76-1 regression: f32 args still routed through INT_REGS"
    )
    # And the cycle-79 return-side stores via movss xmm0 -> slot:
    # `F3 0F 11` opcode for `movss [rbp-N], xmm0`.
    assert b"\xf3\x0f\x11" in elf, (
        "FFI f32 return store from xmm0 (movss [rbp-N], xmm0) missing — "
        "C78-1 regression: f32 return still read from eax"
    )


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
