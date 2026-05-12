"""Stage 29 probe — does removing kovc_lib from K1's input affect parsing?
If parsing still stops at fn 161, bug is in lexer+parser. If parsing
extends, bug is in kovc_lib content.
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
    assert OLD in kovc_lib, "OLD not found"
    return kovc_lib.replace(OLD, NEW)


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
    let src_len = read_file_to_arena("/tmp/sh_nokovc_in.hx");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    let total = emit_elf_for_ast_to_path(ast_root);
    let elf_start = __arena_len() - total;
    write_file_to_arena("/tmp/sh_nokovc_out.bin", elf_start, total)
}
"""
    # K1 driver: needs all 3 (lexer+parser+kovc) since k1_main calls all
    k1_driver_src = lexer_no_main + parser_body + kovc_lib_debug + k1_main

    print("[1] Building K1 (Python-compiled)...")
    k1_elf = python_compile(k1_driver_src, optimize=True)
    out_dir = os.path.join(PROJ, "helixc/tests/_tmp")
    os.makedirs(out_dir, exist_ok=True)
    k1_path = os.path.join(out_dir, "stage29_nokovc_k1.bin")
    with open(k1_path, "wb") as f:
        f.write(k1_elf)
    os.chmod(k1_path, 0o755)
    k1_wsl = "/mnt/c/Projects/Kovostov-Native/helixc/tests/_tmp/stage29_nokovc_k1.bin"

    # K1 input variants
    variants = [
        ("lexer_only", lexer_no_main + "\nfn main() -> i32 { 42 }\n"),
        ("lexer+parser_no_kovc", lexer_no_main + parser_body + "\nfn main() -> i32 { 42 }\n"),
        ("full_bootstrap", lexer_no_main + parser_body + kovc_lib + """
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
"""),
    ]

    for name, src in variants:
        print(f"\n[TEST: {name}] (input size: {len(src)} chars)")
        subprocess.run(
            ["wsl", "-e", "bash", "-c", "cat > /tmp/sh_nokovc_in.hx"],
            input=src.encode("utf-8"),
            check=True, timeout=15,
        )
        run = subprocess.run(
            ["wsl", "-e", "bash", "-c", f"{k1_wsl}; echo exit=$?"],
            capture_output=True, timeout=60,
        )
        print(f"  K1 stdout: {run.stdout.decode()[:200]!r}")

        chk = subprocess.run(
            ["wsl", "-e", "bash", "-c",
             "test -f /tmp/sh_nokovc_out.bin && (wc -c < /tmp/sh_nokovc_out.bin; xxd -s 0x1000 -l 8 -ps /tmp/sh_nokovc_out.bin) || echo NONE"],
            capture_output=True, timeout=10,
        )
        out = chk.stdout.decode().strip()
        lines = out.split('\n')
        if len(lines) >= 2:
            size = lines[0]
            entry_hex = lines[1]
            # Decode entry: is it CALL (E8) or UD2 (0F 0B)?
            if entry_hex.startswith("e8"):
                print(f"  K2: size={size}, entry={entry_hex} = CALL OK")
            elif entry_hex.startswith("0f0b"):
                # Decode debug
                top = int(entry_hex[4:6], 16) | (int(entry_hex[6:8], 16) << 8)
                tl = int(entry_hex[8:10], 16)
                print(f"  K2: size={size}, entry={entry_hex} = UD2! fn_table top={top}, target_len={tl}")
            else:
                print(f"  K2: size={size}, entry={entry_hex} = UNKNOWN")
        else:
            print(f"  K2: {out}")


if __name__ == "__main__":
    main()
