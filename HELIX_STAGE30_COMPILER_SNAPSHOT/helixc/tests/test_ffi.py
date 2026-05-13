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
    """Stage 28.9 cycle 77 C76-1 + cycle 79 C78-1 + cycle 81 C80-1
    regression (HIGH): extern "C" fn with a f32 arg/return must route
    through xmm0 (SysV float-class), not edi/eax. Pre-cycle-77
    FFI_CALL routed every operand through INT_REGS so the f32 was
    silently re-bit-cast as i32 in edi/eax. Cycle-77 fixed the arg
    side, cycle-79 fixed the return side.

    Cycle-81 C80-1 fix: the original test signature `fn entry(x:
    f32) -> f32` produced movss opcodes in `entry`'s own prologue/
    epilogue regardless of FFI_CALL routing, so the byte-pattern
    assertion was non-discriminating. Two changes here:

      1. Build a CONTROL version of the same program with the float
         FFI replaced by an int FFI (puts), and count movss opcodes
         in BOTH. The float-FFI version must have STRICTLY MORE
         movss bytes than the int-FFI control. A regression to all-
         INT FFI_CALL routing would produce equal counts.

      2. Use a caller fn signature `fn caller() -> i32` with NO
         float params/returns, so the only movss opcodes inside the
         caller fn's frame come from the f32-literal load (CONST_
         FLOAT rip-relative) + FFI arg routing (rbp-relative) + FFI
         return store (rbp-relative). The CONTROL has no float
         literal and no float FFI, so 0 movss opcodes."""
    float_src = """
    extern "C" fn cosf(x: f32) -> f32;
    fn caller() -> i32 {
        let _r: f32 = cosf(1.5_f32);
        0
    }
    fn main() -> i32 { caller() }
    """
    int_src = """
    extern "C" fn puts(s: *const u8) -> i32;
    fn caller() -> i32 {
        let m: *const u8 = "x\\0".as_ptr();
        puts(m)
    }
    fn main() -> i32 { caller() }
    """
    float_elf = compile_module_to_elf(lower(parse(float_src, include_stdlib=False)))
    int_elf = compile_module_to_elf(lower(parse(int_src, include_stdlib=False)))
    # Count movss-load (F3 0F 10) and movss-store (F3 0F 11) byte sequences.
    # Both opcodes are SysV-mandated for f32 xmm0 transfers.
    def count(b: bytes, pat: bytes) -> int:
        return b.count(pat)
    float_load = count(float_elf, b"\xf3\x0f\x10")
    float_store = count(float_elf, b"\xf3\x0f\x11")
    int_load = count(int_elf, b"\xf3\x0f\x10")
    int_store = count(int_elf, b"\xf3\x0f\x11")
    # The float-FFI program MUST emit strictly more movss bytes than the
    # int-FFI program. Pre-fix (FFI all-INT routing), the counts would
    # be equal because the f32 arg would route through edi (mov edi,
    # [rbp-N]) and the f32 return would route through eax (mov eax,
    # [rbp-N]), with no movss involvement.
    assert float_load > int_load, (
        f"FFI f32 arg load to xmm0 not emitted — C76-1 regression: "
        f"f32 args still routed through INT_REGS. "
        f"float-program movss-load count={float_load}, "
        f"int-program movss-load count={int_load}"
    )
    assert float_store > int_store, (
        f"FFI f32 return store from xmm0 not emitted — C78-1 regression: "
        f"f32 return still read from eax. "
        f"float-program movss-store count={float_store}, "
        f"int-program movss-store count={int_store}"
    )


if __name__ == "__main__":
    # Stage 28.9 cycle 84 audit-CR CR-1 fix (HIGH conf 90): pre-fix
    # this list was hard-coded with the 3 Stage-16.5 tests and the
    # cycle-77/79/81 regression test
    # `test_c76_1_ffi_call_routes_f32_args_to_xmm0` was silently
    # omitted — so `scripts/run_all_tests.sh` invoking `python
    # helixc/tests/test_ffi.py` never executed it, leaving the FFI
    # float-class routing fix undefended in the heavy gate. Switch
    # to globals() discovery (matches test_ir.py / test_totality.py
    # pattern) so any future `def test_*` is auto-picked-up.
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
