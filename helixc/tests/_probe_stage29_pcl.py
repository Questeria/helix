"""Stage 29 — verify parse_closure_lit's body is the trigger."""
import os
import sys
import re
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


def temp_debug_kovc(kovc_lib: str) -> str:
    OLD = """            } else {
                // ud2 = 0F 0B; pad remaining 3 bytes with NOP (90).
                // disp_slot points at byte 1 of the original 5-byte
                // E8+disp instruction; opcode E8 is at disp_slot-1.
                __arena_set(disp_slot - 1, 0x0F);
                __arena_set(disp_slot, 0x0B);
                __arena_set(disp_slot + 1, 0x90);
                __arena_set(disp_slot + 2, 0x90);
                __arena_set(disp_slot + 3, 0x90);
            };"""
    NEW = """            } else {
                let ft_top = __arena_get(fn_state);
                let top_lo = ft_top % 256;
                let top_hi = (ft_top / 256) % 256;
                let tl = target_name_l % 256;
                __arena_set(disp_slot - 1, 0x0F);
                __arena_set(disp_slot, 0x0B);
                __arena_set(disp_slot + 1, top_lo);
                __arena_set(disp_slot + 2, top_hi);
                __arena_set(disp_slot + 3, tl);
            };"""
    return kovc_lib.replace(OLD, NEW)


def stub_fn(src: str, fn_name: str) -> str:
    pat = re.compile(rf'fn {re.escape(fn_name)}\([^)]*\)\s*->\s*i32\s*\{{')
    m = pat.search(src)
    if not m:
        raise RuntimeError(f"{fn_name} not found")
    body_start = m.end()
    depth = 1
    pos = body_start
    while pos < len(src) and depth > 0:
        c = src[pos]
        if c == '{': depth += 1
        elif c == '}': depth -= 1
        pos += 1
    body_end = pos
    new_src = src[:body_start] + " 0 " + src[body_end - 1:]
    return new_src


def main():
    lexer = open(os.path.join(PROJ, "helixc/bootstrap/lexer.hx")).read()
    lexer_no_main = lexer.rsplit(
        "// --------------------------------------------------------------\n// Demo:", 1)[0]
    parser_body = open(os.path.join(PROJ, "helixc/bootstrap/parser.hx")).read()
    kovc = open(os.path.join(PROJ, "helixc/bootstrap/kovc.hx")).read()
    kovc_lib = kovc.rsplit(
        "// --------------------------------------------------------------\n// Demo:", 1)[0]
    kovc_lib_debug = temp_debug_kovc(kovc_lib)

    k1_main = """
fn main() -> i32 {
    let src_start = __arena_len();
    let src_len = read_file_to_arena("/tmp/sh_pcl_in.hx");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    let total = emit_elf_for_ast_to_path(ast_root);
    let elf_start = __arena_len() - total;
    write_file_to_arena("/tmp/sh_pcl_out.bin", elf_start, total)
}
"""
    k1_driver_src = lexer_no_main + parser_body + kovc_lib_debug + k1_main

    print("[1] Building K1...")
    k1_elf = python_compile(k1_driver_src, optimize=True)
    out_dir = os.path.join(PROJ, "helixc/tests/_tmp")
    os.makedirs(out_dir, exist_ok=True)
    k1_path = os.path.join(out_dir, "stage29_pcl_k1.bin")
    with open(k1_path, "wb") as f:
        f.write(k1_elf)
    os.chmod(k1_path, 0o755)
    k1_wsl = "/mnt/c/Projects/Kovostov-Native/helixc/tests/_tmp/stage29_pcl_k1.bin"

    # Stub parse_closure_lit
    parser_stubbed = stub_fn(parser_body, "parse_closure_lit")
    src_stubbed = lexer_no_main + parser_stubbed + """
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
    # Run with stubbed parse_closure_lit (full bootstrap, just stubbed)
    full_kovc = lexer_no_main + parser_stubbed + kovc_lib + """
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
    print(f"\n[TEST: full bootstrap with parse_closure_lit STUBBED] (size: {len(full_kovc)})")
    subprocess.run(
        ["wsl", "-e", "bash", "-c", "cat > /tmp/sh_pcl_in.hx"],
        input=full_kovc.encode("utf-8"),
        check=True, timeout=15,
    )
    run = subprocess.run(
        ["wsl", "-e", "bash", "-c", f"{k1_wsl}; echo exit=$?"],
        capture_output=True, timeout=60,
    )
    chk = subprocess.run(
        ["wsl", "-e", "bash", "-c",
         "test -f /tmp/sh_pcl_out.bin && (wc -c < /tmp/sh_pcl_out.bin; xxd -s 0x1000 -l 8 -ps /tmp/sh_pcl_out.bin) || echo NONE"],
        capture_output=True, timeout=10,
    )
    out = chk.stdout.decode().strip()
    print(f"  Result: {out}")


if __name__ == "__main__":
    main()
