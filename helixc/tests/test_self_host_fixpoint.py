"""Self-host fixpoint milestone (2026-05-28, commit 0ee8824).

Proves the Helix bootstrap SELF-HOSTS: K1 (built by the Python reference compiler
from the full bootstrap source) compiles its OWN full ~1.43 MB source into K2 (a
working ~566 KB compiler), and K2 compiles a trivial `6*7` program into K3, which
exits 42.

Runs K1/K2 under a big stack (`ulimit -s unlimited`) because the bootstrap parser
is deeply recursive (parse_primary has ~1241 lets), which overflows the default
8 MB stack on the full self-compile -- the separate "bug #1" stack issue, fixed
later by an entry-stub that mmaps its own big stack (after which the external
ulimit is unnecessary and the canonical self_host_loop test can be unskipped).

This test locks in the milestone so the read-buffer fix (BUF_SIZE 1MB->4MB) and
the self-host capability cannot silently regress.
"""
import os
import subprocess

_SEP = "// --------------------------------------------------------------\n// Demo:"
_UL = "ulimit -s unlimited 2>/dev/null || ulimit -s 1048576; "


def _read_lib(proj, name):
    return open(os.path.join(proj, "helixc", "bootstrap", name)).read().rsplit(_SEP, 1)[0]


def _sz(path):
    r = subprocess.run(["wsl", "-e", "bash", "-c", f"stat -c %s {path} 2>/dev/null || echo 0"],
                       capture_output=True, timeout=10)
    return int(r.stdout.decode().strip() or "0")


def test_self_host_fixpoint_bigstack():
    from helixc.tests.test_codegen import _compile_src_to_elf
    proj = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    lexer_no_main = _read_lib(proj, "lexer.hx")
    parser_body = open(os.path.join(proj, "helixc", "bootstrap", "parser.hx")).read()
    kovc_lib = _read_lib(proj, "kovc.hx")
    k1_main = """
fn main() -> i32 {
    let src_start = __arena_len();
    let src_len = read_file_to_arena("/tmp/sh_fp_k1_in.hx");
    let tok_base = __arena_len();
    lex(src_start, src_len);
    let ast_root = parse_top(tok_base);
    let total = emit_elf_for_ast_to_path(ast_root);
    let elf_start = __arena_len() - total;
    write_file_to_arena("/tmp/sh_fp_k1_out.bin", elf_start, total)
}
"""
    k2_main = k1_main.replace("/tmp/sh_fp_k1_in.hx", "/tmp/sh_fp_k2_in.hx").replace("/tmp/sh_fp_k1_out.bin", "/tmp/sh_fp_k2_out.bin")
    k1_driver = lexer_no_main + parser_body + kovc_lib + k1_main
    k1_input = lexer_no_main + parser_body + kovc_lib + k2_main

    # P0 (Python) builds K1.
    k1_elf = _compile_src_to_elf(k1_driver)
    subprocess.run(["wsl", "-e", "bash", "-c", "cat > /tmp/sh_fp_k1.bin"], input=k1_elf, check=True, timeout=30)
    subprocess.run(["wsl", "-e", "bash", "-c", "cat > /tmp/sh_fp_k1_in.hx"], input=k1_input.encode("utf-8"), check=True, timeout=30)

    # K1 compiles the FULL bootstrap source -> K2.
    subprocess.run(["wsl", "-e", "bash", "-c", f"rm -f /tmp/sh_fp_k1_out.bin; {_UL}chmod +x /tmp/sh_fp_k1.bin && /tmp/sh_fp_k1.bin"], timeout=180)
    k2_size = _sz("/tmp/sh_fp_k1_out.bin")
    assert k2_size > 0, f"K1 failed to compile the full bootstrap source into K2 (K2 size={k2_size})"
    subprocess.run(["wsl", "-e", "bash", "-c", "cp /tmp/sh_fp_k1_out.bin /tmp/sh_fp_k2.bin"], check=True, timeout=10)

    # K2 (the self-built compiler) compiles 6*7 -> K3.
    subprocess.run(["wsl", "-e", "bash", "-c", 'printf "%s" "fn main() -> i32 { 6 * 7 }" > /tmp/sh_fp_k2_in.hx'], check=True, timeout=10)
    subprocess.run(["wsl", "-e", "bash", "-c", f"rm -f /tmp/sh_fp_k2_out.bin; {_UL}chmod +x /tmp/sh_fp_k2.bin && /tmp/sh_fp_k2.bin"], timeout=120)
    k3_size = _sz("/tmp/sh_fp_k2_out.bin")
    assert k3_size > 0, f"K2 (self-built compiler) failed to compile 6*7 into K3 (K3 size={k3_size})"

    # K3 must exit 42 (6*7).
    r3 = subprocess.run(["wsl", "-e", "bash", "-c", "cp /tmp/sh_fp_k2_out.bin /tmp/sh_fp_k3.bin && chmod +x /tmp/sh_fp_k3.bin && /tmp/sh_fp_k3.bin; echo exit=$?"],
                        capture_output=True, timeout=10)
    assert b"exit=42" in r3.stdout, f"K3 (compiled by the self-built K2) did not exit 42: stdout={r3.stdout!r} stderr={r3.stderr!r}"
