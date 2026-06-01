#!/usr/bin/env python3
# Assemble the self-compile source for the seed-built helixc (K1'), exactly as
# helixc/tests/test_self_host_fixpoint.py does -- a dev/build helper (NOT part of
# the trusted runtime; it only concatenates the FROZEN helixc/bootstrap sources).
#
#   k1_driver = lexer_no_main + parser_body + kovc_lib + driver_main
#   k1_input  = lexer_no_main + parser_body + kovc_lib + input_main  (k2 paths)
#
# The seed compiles k1_driver -> a runnable helixc; that helixc compiles k1_input.
import os
PROJ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEP = "// --------------------------------------------------------------\n// Demo:"


def read_lib(name):
    p = os.path.join(PROJ, "helixc", "bootstrap", name)
    return open(p, encoding="utf-8").read().rsplit(SEP, 1)[0]


lexer_no_main = read_lib("lexer.hx")
parser_body = open(os.path.join(PROJ, "helixc", "bootstrap", "parser.hx"), encoding="utf-8").read()
kovc_lib = read_lib("kovc.hx")

driver_main = """
fn main() -> i32 {
    let src_start = __arena_len();
    let src_len = read_file_to_arena("/tmp/k1_in.hx");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    let total = emit_elf_for_ast_to_path(ast_root);
    let elf_start = __arena_len() - total;
    write_file_to_arena("/tmp/k1_out.bin", elf_start, total)
}
"""
input_main = driver_main.replace("/tmp/k1_in.hx", "/tmp/k2_in.hx").replace("/tmp/k1_out.bin", "/tmp/k2_out.bin")

# PTX-driver main (P1.2, GPU first-light / DoD criterion #3): read an @kernel .hx,
# auto-route to PTX via the EXISTING emit_auto_for_ast_to_path dispatcher, write the
# .ptx. Differs from driver_main ONLY in the emitter call + the fixed in/out paths,
# so it touches NO frozen helixc/bootstrap source -> no self-host-fixpoint impact.
ptx_driver_main = (driver_main
    .replace("/tmp/k1_in.hx", "/tmp/kernel_in.hx")
    .replace("emit_elf_for_ast_to_path", "emit_auto_for_ast_to_path")
    .replace("/tmp/k1_out.bin", "/tmp/out.ptx")
    .replace("elf_start", "ptx_start"))

k1_driver = lexer_no_main + parser_body + kovc_lib + driver_main
k1_input = lexer_no_main + parser_body + kovc_lib + input_main
k1_ptx_driver = lexer_no_main + parser_body + kovc_lib + ptx_driver_main

here = os.path.dirname(os.path.abspath(__file__))
open(os.path.join(here, "k1src.hx"), "w", encoding="utf-8", newline="\n").write(k1_driver)
open(os.path.join(here, "k1input.hx"), "w", encoding="utf-8", newline="\n").write(k1_input)
open(os.path.join(here, "k1ptxdrv.hx"), "w", encoding="utf-8", newline="\n").write(k1_ptx_driver)
print("k1_driver:", k1_driver.count(chr(10)) + 1, "lines,", len(k1_driver.encode("utf-8")), "bytes -> k1src.hx")
print("k1_input :", k1_input.count(chr(10)) + 1, "lines,", len(k1_input.encode("utf-8")), "bytes -> k1input.hx")
print("k1_ptxdrv:", k1_ptx_driver.count(chr(10)) + 1, "lines,", len(k1_ptx_driver.encode("utf-8")), "bytes -> k1ptxdrv.hx")
