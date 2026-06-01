#!/usr/bin/env bash
# assemble_k1.sh -- Python-free port of assemble_k1.py (Helix v1.0 DoD #6).
# Concatenates the FROZEN bootstrap sources (lexer.hx + parser.hx + kovc.hx; lexer
# and kovc stripped at the "// Demo:" SEP, parser used whole) + 3 driver mains,
# writing k1src.hx / k1input.hx / k1ptxdrv.hx -- BYTE-IDENTICAL to the Python version
# (verified by stage0/helixc-bootstrap, see scripts that diff py vs sh output).
# Pure file concatenation, zero semantic transformation. (shell is still "another
# language"; the final v1.0 target is a Helix concatenator -- this removes the .py.)
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BOOT="$HERE/../../helixc/bootstrap"

# strip_demo <file>: print everything BEFORE the dashes line that precedes "// Demo:"
# -- mirrors Python rsplit(SEP,1)[0] where SEP = "// -----...-----\n// Demo:".
# The SEP appears exactly once in lexer.hx and kovc.hx (verified), so first==last.
# Lags the print by one line and exits on the Demo line whose previous line was the
# dashes, so the dashes line itself and everything after are dropped.
strip_demo() {
  awk '
    /^\/\/ Demo:/ && prevdash { exit }
    NR>1 { print prev }
    { prev=$0; prevdash = ($0 ~ /^\/\/ ----/) }
  ' "$1"
}

# gen_driver <in_path> <out_path> <emit_fn> <start_prefix>: the driver main, matching
# the Python triple-quoted string exactly (leading blank line + 4-space indent +
# trailing newline). printf interprets the \n escapes; %s args are literal.
gen_driver() {
  printf '\nfn main() -> i32 {\n    let src_start = __arena_len();\n    let src_len = read_file_to_arena("%s");\n    let tok_base = __arena_len();\n    lex(src_start, src_len);\n    let ast_root = parse_top(tok_base);\n    let total = %s(ast_root);\n    let %s_start = __arena_len() - total;\n    write_file_to_arena("%s", %s_start, total)\n}\n' \
    "$1" "$3" "$4" "$2" "$4"
}

# build <outfile> <in> <out> <emit> <prefix>: concat directly (NO command substitution,
# which would strip trailing newlines and break byte-identity at the boundaries). The
# frozen sources are CRLF on disk; Python's universal-newlines normalizes to LF on
# read+write, so we strip \r from the FILE parts to match (the driver printf is already
# LF). gen_driver is appended after (it must NOT be \r-stripped, though it has none).
build() {
  { strip_demo "$BOOT/lexer.hx"; cat "$BOOT/parser.hx"; strip_demo "$BOOT/kovc.hx"; } | tr -d '\r' > "$HERE/$1"
  gen_driver "$2" "$3" "$4" "$5" >> "$HERE/$1"
}

build k1src.hx    /tmp/k1_in.hx     /tmp/k1_out.bin  emit_elf_for_ast_to_path   elf
build k1input.hx  /tmp/k2_in.hx     /tmp/k2_out.bin  emit_elf_for_ast_to_path   elf
build k1ptxdrv.hx /tmp/kernel_in.hx /tmp/out.ptx     emit_auto_for_ast_to_path  ptx

echo "assembled: k1src.hx k1input.hx k1ptxdrv.hx ($(wc -c < "$HERE/k1src.hx") bytes each-ish)"
