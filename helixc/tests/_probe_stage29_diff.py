"""Stage 29 K2 SIGILL probe — bytewise divergence localizer.

Goal: find the first byte where bootstrap's codegen diverges from Python's,
on the SAME input source. Bytes-equal => bootstrap matches Python; any
divergence localizes the bug.

Strategy:
  (1) Pick a minimal input source S.
  (2) Compile S with Python pipeline -> P_S binary.
  (3) Compile S with K1 (Python-compiled bootstrap) -> K1_S binary.
  (4) Compare byte-by-byte. First divergence = bug locus.

We use a kovc.hx-style driver whose main reads a path and emits ELF
to a path. K1 is just (bootstrap + that driver) compiled by Python.

Run: python helixc/tests/_probe_stage29_diff.py
"""
import os
import sys
import subprocess

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJ = os.path.dirname(os.path.dirname(THIS_DIR))
sys.path.insert(0, PROJ)

from helixc.frontend.parser import parse  # noqa: E402
from helixc.frontend.flatten_modules import flatten_modules  # noqa: E402
from helixc.frontend.flatten_impls import flatten_impls  # noqa: E402
from helixc.frontend.monomorphize import monomorphize  # noqa: E402
from helixc.frontend.grad_pass import grad_pass  # noqa: E402
from helixc.ir.lower_ast import lower  # noqa: E402
from helixc.ir.passes.const_fold import fold_module  # noqa: E402
from helixc.ir.passes.cse import cse_module  # noqa: E402
from helixc.ir.passes.dce import dce_module  # noqa: E402
from helixc.ir.passes.fdce import fdce_module  # noqa: E402
from helixc.backend.x86_64 import compile_module_to_elf  # noqa: E402


def python_compile(src: str, optimize: bool = True) -> bytes:
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


def main():
    # Step 1: build K1 = Python-compiled bootstrap.
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
    let src_len = read_file_to_arena("/tmp/sh_probe_in.hx");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    let total = emit_elf_for_ast_to_path(ast_root);
    let elf_start = __arena_len() - total;
    write_file_to_arena("/tmp/sh_probe_k1_out.bin", elf_start, total)
}
"""

    k1_driver_src = lexer_no_main + parser_body + kovc_lib + k1_main

    # Tiny input the bootstrap will compile
    TEST_SRC = "fn main() -> i32 { 42 }"

    # Stage A: write tiny input to /tmp
    print("[1/5] Staging tiny input...", flush=True)
    subprocess.run(
        ["wsl", "-e", "bash", "-c", "cat > /tmp/sh_probe_in.hx"],
        input=TEST_SRC.encode("utf-8"),
        check=True, timeout=10,
    )

    # Stage B: compile k1_driver via Python -> binary, then run it.
    # The binary's main reads /tmp/sh_probe_in.hx and writes K1's
    # compilation of TEST_SRC to /tmp/sh_probe_k1_out.bin.
    print("[2/5] Compiling K1 (Python-compiled bootstrap)...", flush=True)
    k1_elf = python_compile(k1_driver_src, optimize=True)
    k1_path = "/tmp/sh_probe_k1_driver.bin"
    win_path = "C:/Projects/Kovostov-Native/helixc/tests/_tmp/k1_driver.bin"
    os.makedirs(os.path.dirname(win_path), exist_ok=True)
    with open(win_path, "wb") as f:
        f.write(k1_elf)
    os.chmod(win_path, 0o755)
    # Copy to /tmp via WSL
    subprocess.run(
        ["wsl", "-e", "bash", "-c",
         f"cp '/mnt/c/Projects/Kovostov-Native/helixc/tests/_tmp/k1_driver.bin' {k1_path} && chmod +x {k1_path}"],
        check=True, timeout=15,
    )

    print(f"[2/5] K1 driver written: {len(k1_elf)} bytes", flush=True)

    # Stage C: run K1 — it compiles TEST_SRC and writes K1's output.
    print("[3/5] Running K1 to compile TEST_SRC...", flush=True)
    run_k1 = subprocess.run(
        ["wsl", "-e", "bash", "-c", f"{k1_path}"],
        capture_output=True, timeout=30,
    )
    print(f"  K1 exit={run_k1.returncode}", flush=True)
    if run_k1.returncode >= 128:
        print(f"  K1 CRASHED: {run_k1.stderr!r}", flush=True)
        sys.exit(2)

    # Stage D: read K1's output
    k1_out = subprocess.run(
        ["wsl", "-e", "bash", "-c",
         "wc -c < /tmp/sh_probe_k1_out.bin && md5sum /tmp/sh_probe_k1_out.bin"],
        capture_output=True, timeout=10,
    )
    print(f"  K1's output: {k1_out.stdout.decode()}", flush=True)

    # Copy K1's output to Win-accessible path
    subprocess.run(
        ["wsl", "-e", "bash", "-c",
         "cp /tmp/sh_probe_k1_out.bin /mnt/c/Projects/Kovostov-Native/helixc/tests/_tmp/k1_emitted.bin"],
        check=True, timeout=10,
    )
    k1_emitted = open("C:/Projects/Kovostov-Native/helixc/tests/_tmp/k1_emitted.bin", "rb").read()

    # Stage E: compile TEST_SRC via Python (with optimize=False to match bootstrap)
    print("[4/5] Compiling TEST_SRC via Python (no optimize)...", flush=True)
    py_elf = python_compile(TEST_SRC, optimize=False)
    print(f"  Python output: {len(py_elf)} bytes", flush=True)
    with open("C:/Projects/Kovostov-Native/helixc/tests/_tmp/python_emitted.bin", "wb") as f:
        f.write(py_elf)

    # Stage F: Run K1's emitted binary to verify it works.
    subprocess.run(
        ["wsl", "-e", "bash", "-c",
         "chmod +x /tmp/sh_probe_k1_out.bin && /tmp/sh_probe_k1_out.bin; echo exit=$?"],
        capture_output=True, timeout=10,
    )
    run_k1emit = subprocess.run(
        ["wsl", "-e", "bash", "-c",
         "/tmp/sh_probe_k1_out.bin; echo exit=$?"],
        capture_output=True, timeout=10,
    )
    print(f"  K1's binary runs: {run_k1emit.stdout!r} stderr={run_k1emit.stderr!r}", flush=True)

    # Stage G: Diff K1's output vs Python's output.
    print(f"[5/5] Diff: K1's emit ({len(k1_emitted)} bytes) vs Python's emit ({len(py_elf)} bytes)", flush=True)

    min_len = min(len(k1_emitted), len(py_elf))
    first_diff = -1
    for i in range(min_len):
        if k1_emitted[i] != py_elf[i]:
            first_diff = i
            break

    if first_diff == -1 and len(k1_emitted) == len(py_elf):
        print("  *** IDENTICAL — bootstrap matches Python byte-for-byte! ***")
        return
    if first_diff == -1:
        print(f"  Identical up to {min_len}, then K1={len(k1_emitted)} != Py={len(py_elf)}")
        first_diff = min_len

    print(f"\n  First divergence at byte 0x{first_diff:x} (decimal {first_diff})")
    print(f"  K1[{first_diff}:{first_diff+32}]: {k1_emitted[first_diff:first_diff+32].hex()}")
    print(f"  Py[{first_diff}:{first_diff+32}]: {py_elf[first_diff:first_diff+32].hex()}")
    print(f"  K1 context [-16..+32]: {k1_emitted[max(0,first_diff-16):first_diff+32].hex()}")
    print(f"  Py context [-16..+32]: {py_elf[max(0,first_diff-16):first_diff+32].hex()}")


if __name__ == "__main__":
    main()
