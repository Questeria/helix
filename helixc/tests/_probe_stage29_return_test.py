"""Stage 29 hypothesis test: is `return` keyword the trigger?

Bootstrap parser.hx uses `return` in 18 active code lines, first at line 2028
inside parse_closure_lit's body. If bootstrap parser doesn't support `return`
as a keyword, it lexes "return" as TK_IDENT, then the parser tries to parse
`return mk_node(...)` as a sequence of two expressions without `;` separator,
which fails.

Test: replace the early `return` with rephrased equivalent (use if/else
expression) and see if K1 parses more of bootstrap source.
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


def main():
    lexer = open(os.path.join(PROJ, "helixc/bootstrap/lexer.hx")).read()
    lexer_no_main = lexer.rsplit(
        "// --------------------------------------------------------------\n// Demo:", 1)[0]
    parser_body = open(os.path.join(PROJ, "helixc/bootstrap/parser.hx")).read()
    kovc = open(os.path.join(PROJ, "helixc/bootstrap/kovc.hx")).read()
    kovc_lib = kovc.rsplit(
        "// --------------------------------------------------------------\n// Demo:", 1)[0]
    kovc_lib_debug = temp_debug_kovc(kovc_lib)

    # Replace ALL `return EXPR;` statements with their value (Helix has
    # expression-based fn bodies; `return` is implicit on tail expr).
    # The parse_closure_lit `return mk_node(99, 76003, 0, 0);` becomes
    # an early-exit if-else.

    # Specifically, replace the SPECIFIC line 2028 area:
    OLD = """        if nonint_capture == 1 {
            // Loud failure: AST_ERR(76003) propagates to codegen which
            // emits a hard trap when the closure is invoked. The full
            // type-preserving capture is a follow-on cycle.
            return mk_node(99, 76003, 0, 0);
        }
        let mut pi: i32 = 0;
        while pi < p_count {"""
    NEW = """        if nonint_capture == 1 {
            // Loud failure: AST_ERR(76003) propagates to codegen which
            // emits a hard trap when the closure is invoked. The full
            // type-preserving capture is a follow-on cycle.
            mk_node(99, 76003, 0, 0)
        } else {
        let mut pi: i32 = 0;
        while pi < p_count {"""
    # Need closing brace too. Let me find the end of parse_closure_lit body and
    # add `}` before its closing `}`.
    if OLD not in parser_body:
        print("OLD pattern not found in parser_body. Aborting.")
        return
    parser_modified = parser_body.replace(OLD, NEW)
    # Now find parse_closure_lit's closing `}` and add an extra `}` before it.
    # The body of parse_closure_lit ends around line 2080.
    # Match `    }\n}\n\nfn parse_primary` and add another `}` before the outer `}`.
    PCL_END_OLD = """        // Return a placeholder AST_INT(0). The closure's runtime value is
        // unused — only the binding name matters at the call site.
        mk_node(0, 0, 0, 0)
    }
}

fn parse_primary"""
    PCL_END_NEW = """        // Return a placeholder AST_INT(0). The closure's runtime value is
        // unused — only the binding name matters at the call site.
        mk_node(0, 0, 0, 0)
        }
    }
}

fn parse_primary"""
    if PCL_END_OLD not in parser_modified:
        print("PCL_END_OLD pattern not found. Aborting.")
        return
    parser_modified = parser_modified.replace(PCL_END_OLD, PCL_END_NEW)

    # Verify the modified source parses with Python (which DOES support return)
    # First check it parses
    test_src_python = parser_modified
    print(f"Modified parser size: {len(parser_modified)} (was {len(parser_body)})")

    k1_main = """
fn main() -> i32 {
    let src_start = __arena_len();
    let src_len = read_file_to_arena("/tmp/sh_ret_in.hx");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    let total = emit_elf_for_ast_to_path(ast_root);
    let elf_start = __arena_len() - total;
    write_file_to_arena("/tmp/sh_ret_out.bin", elf_start, total)
}
"""

    # Build K1 using the MODIFIED parser_body (replacing return with if/else).
    # If the modified parser_body still parses cleanly with Python, K1 will
    # have all the same logic but tokens won't include `return` keyword.
    k1_driver_src = lexer_no_main + parser_modified + kovc_lib_debug + k1_main
    print("[1] Compiling K1 with modified parser (return removed)...")
    try:
        k1_elf = python_compile(k1_driver_src, optimize=True)
    except Exception as e:
        print(f"  Python compile FAILED: {e}")
        return

    out_dir = os.path.join(PROJ, "helixc/tests/_tmp")
    os.makedirs(out_dir, exist_ok=True)
    k1_path = os.path.join(out_dir, "stage29_ret_k1.bin")
    with open(k1_path, "wb") as f:
        f.write(k1_elf)
    os.chmod(k1_path, 0o755)
    k1_wsl = "/mnt/c/Projects/Kovostov-Native/helixc/tests/_tmp/stage29_ret_k1.bin"
    print(f"  K1 size: {len(k1_elf)}")

    # K1's input: the MODIFIED bootstrap source (without return)
    k1_input = lexer_no_main + parser_modified + kovc_lib + """
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
    print(f"  K1 input size: {len(k1_input)}")
    subprocess.run(
        ["wsl", "-e", "bash", "-c", "cat > /tmp/sh_ret_in.hx"],
        input=k1_input.encode("utf-8"),
        check=True, timeout=15,
    )
    print("[2] Running K1 on modified bootstrap source...")
    run = subprocess.run(
        ["wsl", "-e", "bash", "-c", f"{k1_wsl}; echo exit=$?"],
        capture_output=True, timeout=60,
    )
    print(f"  K1 stdout: {run.stdout.decode()[:200]!r}")

    chk = subprocess.run(
        ["wsl", "-e", "bash", "-c",
         "test -f /tmp/sh_ret_out.bin && (wc -c < /tmp/sh_ret_out.bin; xxd -s 0x1000 -l 16 -ps /tmp/sh_ret_out.bin) || echo NONE"],
        capture_output=True, timeout=10,
    )
    print(f"  K2: {chk.stdout.decode().strip()}")


if __name__ == "__main__":
    main()
