"""Stage 29 K2 SIGILL — capture K2 binary, run under strace/dmesg-fallback,
locate the failing instruction without gdb.

Approach:
  1. Rebuild K1 (Python-compiled bootstrap).
  2. Drive K1 to produce K2 from (lexer+parser+kovc+k2_main).
  3. Save K2 binary to Win-accessible path.
  4. Parse K2's ELF, dump the .text bytes.
  5. Run K2 in a way that surfaces RIP — try clearing dmesg then exec, then read dmesg.
  6. Custom x86-64 length-decoder walk to flag obviously bad opcodes.

This is the bytewise-localizer fallback when gdb is unavailable.
"""
import os
import sys
import subprocess
import struct

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, PROJ)


def python_compile(src: str, optimize: bool = True) -> bytes:
    from helixc.frontend.parser import parse
    from helixc.frontend.flatten_modules import flatten_modules
    from helixc.frontend.flatten_impls import flatten_impls
    from helixc.frontend.monomorphize import monomorphize
    from helixc.frontend.grad_pass import grad_pass
    from helixc.ir.lower_ast import lower
    from helixc.ir.passes.const_fold import fold_module
    from helixc.ir.passes.cse import cse_module
    from helixc.ir.passes.dce import dce_module
    from helixc.ir.passes.fdce import fdce_module
    from helixc.backend.x86_64 import compile_module_to_elf
    prog = parse(src, include_stdlib=True)
    flatten_modules(prog)
    flatten_impls(prog)
    monomorphize(prog)
    grad_pass(prog)
    mod = lower(prog)
    if optimize:
        fold_module(mod)
        cse_module(mod)
        dce_module(mod)
        fdce_module(mod)
    return compile_module_to_elf(mod)


def parse_elf64(blob: bytes):
    """Return (entry, text_offset, text_size, text_vaddr)."""
    assert blob[:4] == b"\x7fELF", "not ELF"
    assert blob[4] == 2, "not 64-bit"
    e_phoff = struct.unpack("<Q", blob[32:40])[0]
    e_entry = struct.unpack("<Q", blob[24:32])[0]
    e_phentsize = struct.unpack("<H", blob[54:56])[0]
    e_phnum = struct.unpack("<H", blob[56:58])[0]
    # find first LOAD with X flag
    for i in range(e_phnum):
        off = e_phoff + i * e_phentsize
        p_type = struct.unpack("<I", blob[off:off+4])[0]
        p_flags = struct.unpack("<I", blob[off+4:off+8])[0]
        p_offset = struct.unpack("<Q", blob[off+8:off+16])[0]
        p_vaddr = struct.unpack("<Q", blob[off+16:off+24])[0]
        p_filesz = struct.unpack("<Q", blob[off+32:off+40])[0]
        if p_type == 1 and (p_flags & 1):
            return e_entry, p_offset, p_filesz, p_vaddr
    raise RuntimeError("no PT_LOAD with PF_X found")


def main():
    print("[1] Loading bootstrap sources...", flush=True)
    lexer = open(os.path.join(PROJ, "helixc/bootstrap/lexer.hx")).read()
    lexer_no_main = lexer.rsplit(
        "// --------------------------------------------------------------\n// Demo:",
        1,
    )[0]
    parser_body = open(os.path.join(PROJ, "helixc/bootstrap/parser.hx")).read()
    kovc = open(os.path.join(PROJ, "helixc/bootstrap/kovc.hx")).read()
    kovc_lib = kovc.rsplit(
        "// --------------------------------------------------------------\n// Demo:",
        1,
    )[0]

    k1_main = """
fn main() -> i32 {
    let src_start = __arena_len();
    let src_len = read_file_to_arena("/tmp/sh_k1_in.hx");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    let total = emit_elf_for_ast_to_path(ast_root);
    let elf_start = __arena_len() - total;
    write_file_to_arena("/tmp/sh_k1_out.bin", elf_start, total)
}
"""
    k2_main = """
fn main() -> i32 {
    let src_start = __arena_len();
    let src_len = read_file_to_arena("/tmp/sh_k2_in.hx");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    let total = emit_elf_for_ast_to_path(ast_root);
    let elf_start = __arena_len() - total;
    write_file_to_arena("/tmp/sh_k2_out.bin", elf_start, total)
}
"""

    k1_driver_src = lexer_no_main + parser_body + kovc_lib + k1_main
    k1_input = lexer_no_main + parser_body + kovc_lib + k2_main  # K1 will compile this

    print("[2] Staging K1 input (bootstrap source + k2_main)...", flush=True)
    subprocess.run(
        ["wsl", "-e", "bash", "-c", "cat > /tmp/sh_k1_in.hx"],
        input=k1_input.encode("utf-8"),
        check=True, timeout=30,
    )

    print("[3] Compiling K1 via Python...", flush=True)
    k1_elf = python_compile(k1_driver_src, optimize=True)
    out_dir = os.path.join(PROJ, "helixc/tests/_tmp")
    os.makedirs(out_dir, exist_ok=True)
    k1_win = os.path.join(out_dir, "stage29_k1.bin")
    with open(k1_win, "wb") as f:
        f.write(k1_elf)
    os.chmod(k1_win, 0o755)
    print(f"  K1 size: {len(k1_elf)} bytes", flush=True)

    print("[4] Running K1 — it reads /tmp/sh_k1_in.hx and writes /tmp/sh_k1_out.bin = K2...", flush=True)
    k1_wsl = "/mnt/c/Projects/Kovostov-Native/helixc/tests/_tmp/stage29_k1.bin"
    run_k1 = subprocess.run(
        ["wsl", "-e", "bash", "-c", f"chmod +x {k1_wsl} && {k1_wsl}; echo exit=$?"],
        capture_output=True, timeout=60,
    )
    print(f"  K1 stdout: {run_k1.stdout.decode()[:200]!r}", flush=True)
    print(f"  K1 stderr: {run_k1.stderr.decode()[:200]!r}", flush=True)

    print("[5] Copying K2 from /tmp/sh_k1_out.bin to win-accessible path...", flush=True)
    k2_win = os.path.join(out_dir, "stage29_k2.bin")
    subprocess.run(
        ["wsl", "-e", "bash", "-c",
         f"cp /tmp/sh_k1_out.bin /mnt/c/Projects/Kovostov-Native/helixc/tests/_tmp/stage29_k2.bin && "
         f"chmod +x /mnt/c/Projects/Kovostov-Native/helixc/tests/_tmp/stage29_k2.bin"],
        check=True, timeout=15,
    )
    with open(k2_win, "rb") as f:
        k2_elf = f.read()
    print(f"  K2 size: {len(k2_elf)} bytes", flush=True)

    print("[6] Staging tiny K2 input...", flush=True)
    subprocess.run(
        ["wsl", "-e", "bash", "-c", "printf %s 'fn main() -> i32 { 6 * 7 }' > /tmp/sh_k2_in.hx"],
        check=True, timeout=10,
    )

    print("[7] Running K2 — capturing exit code...", flush=True)
    k2_wsl = "/mnt/c/Projects/Kovostov-Native/helixc/tests/_tmp/stage29_k2.bin"
    run_k2 = subprocess.run(
        ["wsl", "-e", "bash", "-c", f"{k2_wsl}; echo exit=$?"],
        capture_output=True, timeout=30,
    )
    print(f"  K2 stdout: {run_k2.stdout.decode()[:200]!r}", flush=True)
    print(f"  K2 stderr: {run_k2.stderr.decode()[:300]!r}", flush=True)
    print(f"  K2 exit (return): {run_k2.returncode}", flush=True)

    print("[8] Parsing K2 ELF for text section info...", flush=True)
    entry, text_off, text_size, text_vaddr = parse_elf64(k2_elf)
    print(f"  K2 entry: 0x{entry:x}", flush=True)
    print(f"  K2 text: offset=0x{text_off:x} size={text_size} vaddr=0x{text_vaddr:x}", flush=True)
    print(f"  K2 text end vaddr: 0x{text_vaddr + text_size:x}", flush=True)

    print("[9] Dumping K2 text section first 256 bytes (entry context)...", flush=True)
    entry_file_off = text_off + (entry - text_vaddr)
    print(f"  entry file offset: 0x{entry_file_off:x}")
    print(f"  K2 bytes at entry: {k2_elf[entry_file_off:entry_file_off+128].hex()}")

    print("[10] Comparing K1 and K2 sizes/checksums...", flush=True)
    print(f"  K1 size: {len(k1_elf)}; K2 size: {len(k2_elf)}; diff: {len(k2_elf)-len(k1_elf)}")
    # Also save K1's text section for diff
    k1_entry, k1_text_off, k1_text_size, k1_text_vaddr = parse_elf64(k1_elf)
    print(f"  K1 entry: 0x{k1_entry:x}, text off=0x{k1_text_off:x} size={k1_text_size}")

    # If both entries are the same vaddr, we can compare entries directly.
    k1_entry_file_off = k1_text_off + (k1_entry - k1_text_vaddr)
    print(f"  K1 bytes at entry: {k1_elf[k1_entry_file_off:k1_entry_file_off+128].hex()}")

    # Save bytes for offline inspection
    with open(os.path.join(out_dir, "stage29_k1_text.bin"), "wb") as f:
        f.write(k1_elf[k1_text_off:k1_text_off+k1_text_size])
    with open(os.path.join(out_dir, "stage29_k2_text.bin"), "wb") as f:
        f.write(k2_elf[text_off:text_off+text_size])
    print(f"  K1 .text -> stage29_k1_text.bin ({k1_text_size}B)")
    print(f"  K2 .text -> stage29_k2_text.bin ({text_size}B)")


if __name__ == "__main__":
    main()
