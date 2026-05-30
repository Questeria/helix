#!/usr/bin/env bash
# build.sh -- build + verify M2 (M2-Planet, the full self-hosting C compiler)
# from vendored sources using ONLY prior rungs (cc_amd64 + catm + M0 + hex2).
# Run under WSL (Linux ELF). No assembler, no pre-built binary.
#
# Trust chain: hex0 (hand-authored, frozen) -> hex1 -> hex2 -> catm -> M0 ->
# cc_amd64 -> M2-Planet. Sources GPL-3.0, vendored from oriansj/M2-Planet +
# oriansj/M2libc at pinned commits -- see PROVENANCE.md.
set -euo pipefail
cd "$(dirname "$0")"

CATM=../catm/catm.bin
CC=../cc_amd64/cc_amd64.bin
M0=../M0/M0.bin
HEX2=../hex2/hex2.bin
# M2 itself is built by cc_amd64, so its M1 is assembled with cc_amd64's paired
# defs (these live in ../cc_amd64/, byte-identical to stage0-posix upstream).
DEFS=../cc_amd64/amd64_defs.M1
LIBC=../cc_amd64/libc-core.M1
ELF=../cc_amd64/ELF-amd64.hex2
OUT=M2.bin

for t in "$CATM" "$CC" "$M0" "$HEX2"; do [ -x "$t" ] || chmod +x "$t"; done

# 1. Build (mescc-tools-mini-kaem.kaem Phase-5, exact source order):
#    catm concatenates the bootstrap libc + all M2-Planet sources into one .c,
#    cc_amd64 compiles it to M1, catm prepends cc_amd64's defs, M0 assembles to
#    hex2, catm prepends the ELF header, hex2 links the binary.
T=$(mktemp -d)
"$CATM" "$T/M2-0.c" \
    M2libc/amd64/linux/bootstrap.c \
    M2libc/bootstrap.c \
    M2-Planet/cc.h \
    M2libc/bootstrappable.c \
    M2-Planet/cc_globals.c \
    M2-Planet/cc_reader.c \
    M2-Planet/cc_strings.c \
    M2-Planet/cc_types.c \
    M2-Planet/cc_emit.c \
    M2-Planet/cc_core.c \
    M2-Planet/cc_macro.c \
    M2-Planet/cc.c
"$CC"   "$T/M2-0.c"      "$T/M2-0.M1"
"$CATM" "$T/M2-0-0.M1"   "$DEFS" "$LIBC" "$T/M2-0.M1"
"$M0"   "$T/M2-0-0.M1"   "$T/M2-0.hex2"
"$CATM" "$T/M2-0-0.hex2" "$ELF" "$T/M2-0.hex2"
"$HEX2" "$T/M2-0-0.hex2" "$OUT"
rm -rf "$T"
chmod +x "$OUT"
echo "Built $OUT: $(stat -c%s "$OUT") bytes (cc_amd64 -> catm -> M0 -> catm -> hex2)"

# 2. ELF sanity
if ! file "$OUT" | grep -q "ELF 64-bit LSB executable"; then
    echo "ERROR: $OUT is not a valid x86-64 ELF" >&2; file "$OUT"; exit 1
fi
echo "ELF: $(file -b "$OUT")"

# 3. Disassembly (audit-only, gitignored)
if command -v objdump >/dev/null; then
    objdump -D -b binary -m i386:x86-64 "$OUT" > disasm.txt 2>/dev/null || true
    echo "Wrote disasm.txt ($(wc -l < disasm.txt) lines)"
fi

# 4. SHA-256 reproducibility check
NEW=$(sha256sum "$OUT")
if [ -f M2.sha256 ]; then
    if [ "$(cat M2.sha256)" = "$NEW" ]; then
        echo "SHA-256: ${NEW%% *}  (matches M2.sha256)"
    else
        echo "ERROR: SHA-256 mismatch (non-reproducible build)" >&2
        echo "  recorded: $(cat M2.sha256)" >&2
        echo "  rebuilt:  $NEW" >&2
        exit 1
    fi
else
    echo "$NEW" > M2.sha256
    echo "SHA-256 recorded: ${NEW%% *}"
fi

# 5. Tests
bash run_tests.sh
