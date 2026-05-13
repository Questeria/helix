"""Stage 29 — CORRECTED stub test (comment-aware brace counting)."""
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


def stub_fn_correct(src: str, fn_name: str) -> str:
    """Strip comments first, then count braces. Then map back to original."""
    pat = re.compile(rf'^fn {re.escape(fn_name)}\(', re.M)
    m = pat.search(src)
    if not m:
        raise RuntimeError(f"{fn_name} not found")
    # Walk forward line by line. Track open/close braces ignoring those
    # inside `//` comments.
    start_pos = m.start()
    pos = start_pos
    saw_open = False
    depth = 0
    while pos < len(src):
        nl = src.find('\n', pos)
        if nl < 0: nl = len(src)
        line = src[pos:nl]
        # Strip comments
        line_nc = re.sub(r'//.*$', '', line)
        # Also strip char literals like '|' that contain braces (none in
        # Helix, but skip strings just in case)
        for c in line_nc:
            if c == '{':
                depth += 1
                saw_open = True
            elif c == '}':
                depth -= 1
                if saw_open and depth == 0:
                    # Found matching close. Find its position in original line.
                    # All `{}` in the stripped line correspond to chars in line
                    # before the comment marker. So just look in `line_nc`.
                    # Find the LAST `}` in line_nc (this iteration ended here).
                    # Map: position in line of this `}`.
                    # Actually easier: we know depth dropped to 0 on this iteration,
                    # so the closing `}` is somewhere in line_nc. Find the position
                    # of the closing brace.
                    # Walk through line_nc tracking depth more carefully to find exact pos.
                    rebuilt_depth = depth + 1  # before this `}` was processed
                    # Recompute: walk line_nc; first encounter where running_depth would hit 0.
                    # Reconstruct from start of line (with current 'depth' before this line):
                    pre_depth = depth - sum(1 if c == '{' else -1 if c == '}' else 0 for c in line_nc)
                    pre_depth -= 0  # actually our running depth = pre_depth at line start
                    # Hmm too complex. Let me do simpler:
                    running = depth - sum(1 if c == '{' else -1 if c == '}' else 0 for c in line_nc)
                    found_pos = None
                    for ci, c in enumerate(line_nc):
                        if c == '{': running += 1
                        elif c == '}':
                            running -= 1
                            if running == 0:
                                found_pos = ci
                                break
                    body_end_line_offset = found_pos + 1 if found_pos is not None else len(line_nc)
                    body_end_global = pos + body_end_line_offset
                    # body_start: position of FIRST `{` after start_pos
                    body_start_global = src.find('{', start_pos) + 1
                    # Replace src[body_start_global : body_end_global - 1] with " 0 "
                    new_src = src[:body_start_global] + " 0 " + src[body_end_global - 1:]
                    return new_src
        pos = nl + 1
    raise RuntimeError(f"{fn_name} body end not found")


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
    let src_len = read_file_to_arena("/tmp/sh_pclc_in.hx");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    let total = emit_elf_for_ast_to_path(ast_root);
    let elf_start = __arena_len() - total;
    write_file_to_arena("/tmp/sh_pclc_out.bin", elf_start, total)
}
"""
    k1_driver_src = lexer_no_main + parser_body + kovc_lib_debug + k1_main

    print("[1] Building K1...")
    k1_elf = python_compile(k1_driver_src, optimize=True)
    out_dir = os.path.join(PROJ, "helixc/tests/_tmp")
    os.makedirs(out_dir, exist_ok=True)
    k1_path = os.path.join(out_dir, "stage29_pclc_k1.bin")
    with open(k1_path, "wb") as f:
        f.write(k1_elf)
    os.chmod(k1_path, 0o755)
    k1_wsl = "/mnt/c/Projects/Kovostov-Native/helixc/tests/_tmp/stage29_pclc_k1.bin"

    parser_stubbed = stub_fn_correct(parser_body, "parse_closure_lit")
    # Verify the stub is correct
    m = re.search(r'fn parse_closure_lit\([^)]*\)[^{]*\{[^{}]*\}', parser_stubbed)
    if m:
        print(f"  Stub looks correct: {m.group(0)[:100]}")
    else:
        print(f"  WARNING: stub may not be clean")
    print(f"  Original size: {len(parser_body)}, stubbed size: {len(parser_stubbed)}")

    # Test with stubbed parse_closure_lit
    src = lexer_no_main + parser_stubbed + kovc_lib + """
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
    print(f"\n[TEST: full bootstrap with CORRECT parse_closure_lit body stub] (size: {len(src)})")
    subprocess.run(
        ["wsl", "-e", "bash", "-c", "cat > /tmp/sh_pclc_in.hx"],
        input=src.encode("utf-8"),
        check=True, timeout=15,
    )
    run = subprocess.run(
        ["wsl", "-e", "bash", "-c", f"{k1_wsl}; echo exit=$?"],
        capture_output=True, timeout=60,
    )
    chk = subprocess.run(
        ["wsl", "-e", "bash", "-c",
         "test -f /tmp/sh_pclc_out.bin && (wc -c < /tmp/sh_pclc_out.bin; xxd -s 0x1000 -l 8 -ps /tmp/sh_pclc_out.bin) || echo NONE"],
        capture_output=True, timeout=10,
    )
    print(f"  K1 stdout: {run.stdout.decode()[:100]!r}")
    print(f"  K2: {chk.stdout.decode().strip()}")


if __name__ == "__main__":
    main()
