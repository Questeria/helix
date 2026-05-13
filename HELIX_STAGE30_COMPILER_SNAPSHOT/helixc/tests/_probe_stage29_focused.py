"""Stage 29 K2 SIGILL — focused bisect.

Established facts:
- K1 (Python-compiled bootstrap) only adds 161 fns to fn_table when
  compiling the full bootstrap source. Expected ~470.
- The last fn added (source order) is `enum_tab_init` (parser.hx:3822).
- Stubbing parse_primary's body to `{ 0 }` did NOT increase fn count,
  so the bug is NOT inside parse_primary's body.

Hypothesis: when feeding K1 a small program that uses the SAME features
as fns 162+, K1 succeeds. So the issue is cumulative state — likely
something earlier in the source that breaks parser state once we get
deep enough.

This probe tests: feed K1 only enum_tab_init + var_enum_tab_init + main.
Does K1 parse both fns?
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
    let src_len = read_file_to_arena("/tmp/sh_focus_in.hx");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    let total = emit_elf_for_ast_to_path(ast_root);
    let elf_start = __arena_len() - total;
    write_file_to_arena("/tmp/sh_focus_out.bin", elf_start, total)
}
"""
    # Build K1
    k1_driver_src = lexer_no_main + parser_body + kovc_lib_debug + k1_main
    print("[1] Building K1...")
    k1_elf = python_compile(k1_driver_src, optimize=True)
    out_dir = os.path.join(PROJ, "helixc/tests/_tmp")
    os.makedirs(out_dir, exist_ok=True)
    k1_path = os.path.join(out_dir, "stage29_focus_k1.bin")
    with open(k1_path, "wb") as f:
        f.write(k1_elf)
    os.chmod(k1_path, 0o755)
    k1_wsl = "/mnt/c/Projects/Kovostov-Native/helixc/tests/_tmp/stage29_focus_k1.bin"

    # Feed K1 a TINY input that uses enum_tab_init's pattern
    test_inputs = [
        ("tiny_match", """
fn enum_tab_init(sb: i32) -> i32 {
    let et_base = __arena_push(0);
    let mut i: i32 = 1;
    while i < 40 {
        __arena_push(0);
        i = i + 1;
    }
    __arena_set(sb + 20, et_base);
    __arena_set(sb + 21, 0);
    0
}

fn var_enum_tab_init(sb: i32) -> i32 {
    let ve_base = __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_set(sb + 22, ve_base);
    __arena_set(sb + 23, 0);
    0
}

fn main() -> i32 {
    enum_tab_init(0); var_enum_tab_init(0); 42
}
"""),
    ]

    for name, src in test_inputs:
        print(f"\n[TEST: {name}]")
        subprocess.run(
            ["wsl", "-e", "bash", "-c", "cat > /tmp/sh_focus_in.hx"],
            input=src.encode("utf-8"),
            check=True, timeout=10,
        )
        run = subprocess.run(
            ["wsl", "-e", "bash", "-c", f"{k1_wsl}; echo exit=$?"],
            capture_output=True, timeout=30,
        )
        print(f"  K1 stdout: {run.stdout.decode()[:150]!r}")
        # Check K2 exists
        chk = subprocess.run(
            ["wsl", "-e", "bash", "-c",
             "test -f /tmp/sh_focus_out.bin && wc -c < /tmp/sh_focus_out.bin && xxd -s 0x1000 -l 16 -ps /tmp/sh_focus_out.bin || echo NONE"],
            capture_output=True, timeout=10,
        )
        out = chk.stdout.decode().strip()
        print(f"  K2 info: {out!r}")
        if out and out != "NONE":
            run_k2 = subprocess.run(
                ["wsl", "-e", "bash", "-c", "/tmp/sh_focus_out.bin; echo exit=$?"],
                capture_output=True, timeout=10,
            )
            print(f"  K2 result: {run_k2.stdout.decode()[:100]!r}")


if __name__ == "__main__":
    main()
