"""Stage 29 probe — verify K2's "main" name bytes vs patch target.

K2's entry has UD2+NOPs where CALL main should be. That means the patch
resolver couldn't find "main" in fn_table. Either:
  (a) main wasn't added to fn_table (table cap? generic-skip mis-trigger?)
  (b) main was added but with mismatching name bytes vs the patch's target name
  (c) the patch target name bytes are corrupted

Strategy: feed K1 an intermediate-size input that progressively grows
toward the full bootstrap source. Find the boundary where main lookup
starts failing.
"""
import os
import sys
import subprocess

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


def main():
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

    k1_main_template = """
fn main() -> i32 {
    let src_start = __arena_len();
    let src_len = read_file_to_arena("/tmp/sh_probe2_in.hx");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    let total = emit_elf_for_ast_to_path(ast_root);
    let elf_start = __arena_len() - total;
    write_file_to_arena("/tmp/sh_probe2_out.bin", elf_start, total)
}
"""

    # Build K1 (Python-compiled bootstrap) ONCE
    k1_driver_src = lexer_no_main + parser_body + kovc_lib + k1_main_template
    print("[1] Compiling K1 via Python...", flush=True)
    k1_elf = python_compile(k1_driver_src, optimize=True)
    out_dir = os.path.join(PROJ, "helixc/tests/_tmp")
    os.makedirs(out_dir, exist_ok=True)
    k1_path = os.path.join(out_dir, "stage29_k1_probe2.bin")
    with open(k1_path, "wb") as f:
        f.write(k1_elf)
    os.chmod(k1_path, 0o755)
    k1_wsl = "/mnt/c/Projects/Kovostov-Native/helixc/tests/_tmp/stage29_k1_probe2.bin"
    print(f"  K1 built: {len(k1_elf)} bytes", flush=True)

    def k1_compile_and_check(name: str, src: str):
        """Feed src to K1, return (k1_exit, output_bytes_or_none, k2_exit_or_none)."""
        subprocess.run(
            ["wsl", "-e", "bash", "-c", "cat > /tmp/sh_probe2_in.hx"],
            input=src.encode("utf-8"),
            check=True, timeout=10,
        )
        run_k1 = subprocess.run(
            ["wsl", "-e", "bash", "-c", f"{k1_wsl}; echo exit=$?"],
            capture_output=True, timeout=30,
        )
        k1_exit_line = run_k1.stdout.decode().strip().splitlines()[-1] if run_k1.stdout else ""
        # If K2 was produced
        run_check = subprocess.run(
            ["wsl", "-e", "bash", "-c",
             "test -f /tmp/sh_probe2_out.bin && wc -c < /tmp/sh_probe2_out.bin || echo NONE"],
            capture_output=True, timeout=10,
        )
        size_line = run_check.stdout.decode().strip()
        if size_line == "NONE" or not size_line:
            return (k1_exit_line, None, None)
        size = int(size_line)
        # Get K2's first 32 bytes at entry (file offset 0x1000)
        run_bytes = subprocess.run(
            ["wsl", "-e", "bash", "-c",
             "xxd -s 0x1000 -l 32 -ps /tmp/sh_probe2_out.bin"],
            capture_output=True, timeout=10,
        )
        entry_bytes = run_bytes.stdout.decode().strip()
        run_k2 = subprocess.run(
            ["wsl", "-e", "bash", "-c",
             "chmod +x /tmp/sh_probe2_out.bin && /tmp/sh_probe2_out.bin; echo exit=$?"],
            capture_output=True, timeout=20,
        )
        k2_exit = run_k2.stdout.decode().strip().splitlines()[-1] if run_k2.stdout else ""
        return (k1_exit_line, size, entry_bytes, k2_exit)

    # Test 1: tiny multi-fn (should work)
    print("\n[2] TEST 1: tiny multi-fn", flush=True)
    res = k1_compile_and_check("tiny multi-fn", """
fn add(a: i32, b: i32) -> i32 { a + b }
fn main() -> i32 { add(20, 22) }
""")
    print(f"  result: {res}")

    # Test 2: medium — many small fns
    print("\n[3] TEST 2: many small fns (450 of them)", flush=True)
    fns = []
    for i in range(450):
        fns.append(f"fn f{i}(x: i32) -> i32 {{ x + {i} }}")
    fns.append("fn main() -> i32 { f0(42) }")
    res = k1_compile_and_check("450 fns", "\n".join(fns))
    print(f"  result: {res}")

    # Test 3: 500 fns (just under cap)
    print("\n[4] TEST 3: 500 fns (just under fn_table cap of 512)", flush=True)
    fns = []
    for i in range(500):
        fns.append(f"fn f{i}(x: i32) -> i32 {{ x + {i} }}")
    fns.append("fn main() -> i32 { f0(42) }")
    res = k1_compile_and_check("500 fns", "\n".join(fns))
    print(f"  result: {res}")

    # Test 4: 470 fns + main, simulating bootstrap size
    print("\n[5] TEST 4: bootstrap-size full multi-fn", flush=True)
    print(f"     (bootstrap = lexer_no_main + parser_body + kovc_lib + k2_main)", flush=True)
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
    full_src = lexer_no_main + parser_body + kovc_lib + k2_main
    res = k1_compile_and_check("full bootstrap", full_src)
    print(f"  result: {res}")


if __name__ == "__main__":
    main()
