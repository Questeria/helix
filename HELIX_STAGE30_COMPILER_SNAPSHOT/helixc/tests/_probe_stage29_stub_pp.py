"""Stage 29 — verify hypothesis: parse_primary's body trips the
bootstrap parser. Strategy: replace parse_primary's body with a stub
`{ 0 }`, rebuild K1, check fn_table top in K2."""
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


def stub_parse_primary(parser_src: str) -> str:
    """Replace parse_primary's body with `{ 0 }`."""
    # Find `fn parse_primary(...)... { ` opening, then count braces to find matching close.
    pat = re.compile(r'fn parse_primary\([^)]*\)\s*->\s*i32\s*\{')
    m = pat.search(parser_src)
    if not m:
        raise RuntimeError("parse_primary not found")
    body_start = m.end()  # position right after the opening `{`
    # Find matching close brace
    depth = 1
    pos = body_start
    while pos < len(parser_src) and depth > 0:
        c = parser_src[pos]
        if c == '{': depth += 1
        elif c == '}': depth -= 1
        pos += 1
    body_end = pos  # position right after closing `}`
    # Replace body with " 0 "
    new_src = parser_src[:body_start] + " 0 " + parser_src[body_end - 1:]
    print(f"  parse_primary body was {body_end - body_start} chars; stubbed to '{{ 0 }}'")
    return new_src


def temp_debug_kovc(kovc_lib: str) -> str:
    """Inject debug encoding into the patch-fallback so we can read
    fn_table top from K2."""
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
    assert OLD in kovc_lib, "could not find patch-fallback OLD pattern in kovc.hx"
    return kovc_lib.replace(OLD, NEW)


def main():
    lexer = open(os.path.join(PROJ, "helixc/bootstrap/lexer.hx")).read()
    lexer_no_main = lexer.rsplit(
        "// --------------------------------------------------------------\n// Demo:", 1)[0]
    parser_body = open(os.path.join(PROJ, "helixc/bootstrap/parser.hx")).read()
    kovc = open(os.path.join(PROJ, "helixc/bootstrap/kovc.hx")).read()
    kovc_lib = kovc.rsplit(
        "// --------------------------------------------------------------\n// Demo:", 1)[0]

    # Stub out parse_primary's body in the BOOTSTRAP SOURCE that K1 reads
    print("[1] Stubbing parse_primary's body...")
    parser_body_stubbed = stub_parse_primary(parser_body)

    # Wrap kovc_lib (used to BUILD K1) with debug instrumentation
    print("[2] Injecting debug encoding into kovc.hx for K1 build...")
    kovc_lib_debug = temp_debug_kovc(kovc_lib)

    k1_main = """
fn main() -> i32 {
    let src_start = __arena_len();
    let src_len = read_file_to_arena("/tmp/sh_stub_in.hx");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    let total = emit_elf_for_ast_to_path(ast_root);
    let elf_start = __arena_len() - total;
    write_file_to_arena("/tmp/sh_stub_out.bin", elf_start, total)
}
"""
    k2_main = """
fn main() -> i32 {
    42
}
"""

    # K1 driver: use ORIGINAL bootstrap source (with debug kovc_lib).
    # But the INPUT to K1 will be the stubbed parser.
    k1_driver_src = lexer_no_main + parser_body + kovc_lib_debug + k1_main
    print("[3] Compiling K1 driver via Python...")
    k1_elf = python_compile(k1_driver_src, optimize=True)
    out_dir = os.path.join(PROJ, "helixc/tests/_tmp")
    os.makedirs(out_dir, exist_ok=True)
    k1_path = os.path.join(out_dir, "stage29_stub_k1.bin")
    with open(k1_path, "wb") as f:
        f.write(k1_elf)
    os.chmod(k1_path, 0o755)
    print(f"  K1 size: {len(k1_elf)}")

    # K1 input: STUBBED bootstrap source
    k1_input = lexer_no_main + parser_body_stubbed + kovc_lib + k2_main
    print(f"  K1 input size: {len(k1_input)}")
    subprocess.run(
        ["wsl", "-e", "bash", "-c", "cat > /tmp/sh_stub_in.hx"],
        input=k1_input.encode("utf-8"),
        check=True, timeout=30,
    )

    # Run K1
    k1_wsl = "/mnt/c/Projects/Kovostov-Native/helixc/tests/_tmp/stage29_stub_k1.bin"
    print("[4] Running K1...")
    run = subprocess.run(
        ["wsl", "-e", "bash", "-c", f"chmod +x {k1_wsl} && {k1_wsl}; echo exit=$?"],
        capture_output=True, timeout=60,
    )
    print(f"  K1 stdout: {run.stdout.decode()[:200]!r}")

    # Copy K2 + check entry bytes
    k2_path = os.path.join(out_dir, "stage29_stub_k2.bin")
    subprocess.run(
        ["wsl", "-e", "bash", "-c",
         f"cp /tmp/sh_stub_out.bin {os.path.join('/mnt/c/Projects/Kovostov-Native/helixc/tests/_tmp', 'stage29_stub_k2.bin').replace(os.sep, '/').replace('C:', '/mnt/c')} && chmod +x /tmp/sh_stub_out.bin"],
        check=True, timeout=15,
    )

    with open(k2_path, "rb") as f:
        k2 = f.read()
    print(f"  K2 size: {len(k2)}")
    print(f"  K2 bytes at entry (file offset 0x1000): {k2[0x1000:0x1000+16].hex()}")

    # Decode debug: 0F 0B top_lo top_hi target_len at 0x1000
    if k2[0x1000:0x1002] == b"\x0f\x0b":
        top = k2[0x1002] | (k2[0x1003] << 8)
        tl = k2[0x1004]
        print(f"  FAILED: main lookup failed. fn_table top = {top}, target_len = {tl}")
        if top > 162:
            print(f"  HYPOTHESIS VERIFIED: stubbing parse_primary increased fn_table top to {top}!")
        elif top == 162:
            print(f"  Hypothesis NOT verified: fn_table top stayed at 162.")
        else:
            print(f"  fn_table top = {top}, lower than baseline 162. Different bug.")
    else:
        print(f"  SUCCESS: K2 entry has CALL instruction (not UD2)!")
        # Run K2
        run_k2 = subprocess.run(
            ["wsl", "-e", "bash", "-c", "/tmp/sh_stub_out.bin; echo exit=$?"],
            capture_output=True, timeout=20,
        )
        print(f"  K2 exit: {run_k2.stdout.decode()[:100]!r}")


if __name__ == "__main__":
    main()
