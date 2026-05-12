"""Stage 29 bisect — find the EXACT trigger in parser.hx.

We know:
- lexer+parser (no kovc) breaks parsing at fn 161 (enum_tab_init).
- Bug is cumulative state.

Strategy: take parser.hx truncated at progressively later lines.
Use the LARGEST truncation that still leaves syntactically-valid Helix
(i.e., truncate at end-of-fn-decl). Find the SMALLEST truncation where
the bug triggers.
"""
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


def truncate_at_fn_boundary(parser_src: str, max_fn_idx: int) -> str:
    """Truncate parser.hx to keep only the first max_fn_idx top-level fns.
    Returns truncated source ending after the last kept fn's closing `}`.
    """
    brace_depth = 0
    fn_count = 0
    lines = parser_src.split('\n')
    truncated_lines = []
    for line in lines:
        line_nc = re.sub(r'//.*$', '', line)
        if re.match(r'^\s*fn\s+\w+', line_nc) and brace_depth == 0:
            if fn_count >= max_fn_idx:
                # Hit the (max_fn_idx)th fn — stop BEFORE it
                break
            fn_count += 1
        truncated_lines.append(line)
        for c in line_nc:
            if c == '{': brace_depth += 1
            elif c == '}': brace_depth -= 1
    return '\n'.join(truncated_lines)


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
    let src_len = read_file_to_arena("/tmp/sh_bisect_in.hx");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    let total = emit_elf_for_ast_to_path(ast_root);
    let elf_start = __arena_len() - total;
    write_file_to_arena("/tmp/sh_bisect_out.bin", elf_start, total)
}
"""
    k1_driver_src = lexer_no_main + parser_body + kovc_lib_debug + k1_main

    print("[1] Building K1...")
    k1_elf = python_compile(k1_driver_src, optimize=True)
    out_dir = os.path.join(PROJ, "helixc/tests/_tmp")
    os.makedirs(out_dir, exist_ok=True)
    k1_path = os.path.join(out_dir, "stage29_bisect_k1.bin")
    with open(k1_path, "wb") as f:
        f.write(k1_elf)
    os.chmod(k1_path, 0o755)
    k1_wsl = "/mnt/c/Projects/Kovostov-Native/helixc/tests/_tmp/stage29_bisect_k1.bin"

    # New bisect: vary lexer inclusion. If fn_table top = 162 always,
    # the limit is fn-count-based (or cumulative state). If it shifts
    # with lexer inclusion, it's source-content-based.
    tests = [
        ("no_lexer parser_trunc=144", "", truncate_at_fn_boundary(parser_body, 144)),
        ("with_lexer parser_trunc=144", lexer_no_main, truncate_at_fn_boundary(parser_body, 144)),
        ("no_lexer parser_trunc=160", "", truncate_at_fn_boundary(parser_body, 160)),
        ("no_lexer parser_trunc=200", "", truncate_at_fn_boundary(parser_body, 200)),
    ]
    for label, lex_part, parser_part in tests:
        src = lex_part + parser_part + "\nfn main() -> i32 { 42 }\n"
        print(f"\n[TEST: {label}] (size: {len(src)})")
        subprocess.run(
            ["wsl", "-e", "bash", "-c", "cat > /tmp/sh_bisect_in.hx"],
            input=src.encode("utf-8"),
            check=True, timeout=15,
        )
        run = subprocess.run(
            ["wsl", "-e", "bash", "-c", f"{k1_wsl}; echo exit=$?"],
            capture_output=True, timeout=60,
        )
        chk = subprocess.run(
            ["wsl", "-e", "bash", "-c",
             "test -f /tmp/sh_bisect_out.bin && (wc -c < /tmp/sh_bisect_out.bin; xxd -s 0x1000 -l 8 -ps /tmp/sh_bisect_out.bin) || echo NONE"],
            capture_output=True, timeout=10,
        )
        out = chk.stdout.decode().strip()
        lines = out.split('\n')
        if len(lines) >= 2:
            size = lines[0]
            entry_hex = lines[1]
            if entry_hex.startswith("e8"):
                # Compute expected fn_table top: 13 lexer + fn_cap parser + 1 main = fn_cap + 14, plus 1 data = +15
                exp = fn_cap + 13 + 1 + 1
                print(f"  K2: size={size}, entry=CALL OK (fully parsed, expected top~{exp})")
            elif entry_hex.startswith("0f0b"):
                top = int(entry_hex[4:6], 16) | (int(entry_hex[6:8], 16) << 8)
                tl = int(entry_hex[8:10], 16)
                print(f"  K2: size={size}, entry=UD2! fn_table top={top}, target_len={tl}")
            else:
                print(f"  K2: size={size}, entry={entry_hex}")
        else:
            print(f"  K2: {out}")


if __name__ == "__main__":
    main()
