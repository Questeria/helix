"""Simplest possible test: does the bootstrap parser handle `return`?"""
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
        "// --------------------------------------------------------------\n// Demo:", 1)[0]
    parser_body = open(os.path.join(PROJ, "helixc/bootstrap/parser.hx")).read()
    kovc = open(os.path.join(PROJ, "helixc/bootstrap/kovc.hx")).read()
    kovc_lib = kovc.rsplit(
        "// --------------------------------------------------------------\n// Demo:", 1)[0]

    k1_main = """
fn main() -> i32 {
    let src_start = __arena_len();
    let src_len = read_file_to_arena("/tmp/sh_ret2_in.hx");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    let total = emit_elf_for_ast_to_path(ast_root);
    let elf_start = __arena_len() - total;
    write_file_to_arena("/tmp/sh_ret2_out.bin", elf_start, total)
}
"""

    print("[1] Building K1...")
    k1_driver_src = lexer_no_main + parser_body + kovc_lib + k1_main
    k1_elf = python_compile(k1_driver_src, optimize=True)
    out_dir = os.path.join(PROJ, "helixc/tests/_tmp")
    os.makedirs(out_dir, exist_ok=True)
    k1_path = os.path.join(out_dir, "stage29_ret2_k1.bin")
    with open(k1_path, "wb") as f:
        f.write(k1_elf)
    os.chmod(k1_path, 0o755)
    k1_wsl = "/mnt/c/Projects/Kovostov-Native/helixc/tests/_tmp/stage29_ret2_k1.bin"

    # Test cases
    tests = [
        ("NO return — baseline (2 fns)", """
fn helper(x: i32) -> i32 { x + 10 }
fn main() -> i32 { helper(32) }
"""),
        ("WITH return inside if", """
fn helper(x: i32) -> i32 {
    if x > 0 { return 42; }
    x + 10
}
fn main() -> i32 { helper(5) }
"""),
        ("plain return at end", """
fn helper(x: i32) -> i32 { return x + 10; }
fn main() -> i32 { helper(32) }
"""),
    ]

    for name, src in tests:
        print(f"\n[TEST: {name}]")
        subprocess.run(
            ["wsl", "-e", "bash", "-c", "cat > /tmp/sh_ret2_in.hx"],
            input=src.encode("utf-8"),
            check=True, timeout=10,
        )
        run = subprocess.run(
            ["wsl", "-e", "bash", "-c", f"{k1_wsl}; echo exit=$?"],
            capture_output=True, timeout=30,
        )
        print(f"  K1: {run.stdout.decode()[:100]!r}")
        chk = subprocess.run(
            ["wsl", "-e", "bash", "-c",
             "test -f /tmp/sh_ret2_out.bin && (wc -c < /tmp/sh_ret2_out.bin; xxd -s 0x1000 -l 16 -ps /tmp/sh_ret2_out.bin) || echo NONE"],
            capture_output=True, timeout=10,
        )
        out = chk.stdout.decode().strip()
        print(f"  K2: {out}")
        # If K2 produced, try to run it
        if out and out != "NONE":
            run_k2 = subprocess.run(
                ["wsl", "-e", "bash", "-c", "chmod +x /tmp/sh_ret2_out.bin && /tmp/sh_ret2_out.bin; echo exit=$?"],
                capture_output=True, timeout=10,
            )
            print(f"  K2 run: {run_k2.stdout.decode()[:100]!r}, stderr: {run_k2.stderr.decode()[:80]!r}")


if __name__ == "__main__":
    main()
