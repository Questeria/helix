#!/usr/bin/env bash
# build.sh -- build + verify catm.bin from the vendored catm_AMD64.hex2 using
# ONLY the prior rung (hex2). Run under WSL (Linux ELF). No assembler.
#
# Trust chain: hex0 (hand-authored, frozen) -> hex1 -> hex2 -> catm here.
# catm source is GPL-3.0, vendored from oriansj/stage0-posix-amd64 -- see
# PROVENANCE.md. No pre-built binary is trusted.
set -euo pipefail
cd "$(dirname "$0")"

HEX2=../hex2/hex2.bin
SRC=catm_AMD64.hex2
OUT=catm.bin

[ -x "$HEX2" ] || chmod +x "$HEX2"

# 1. Build: hex2 assembles the catm source -> catm.bin
"$HEX2" "$SRC" "$OUT"
chmod +x "$OUT"
echo "Built $OUT: $(stat -c%s "$OUT") bytes (via ../hex2/hex2.bin)"

# 2. ELF sanity
if ! file "$OUT" | grep -q "ELF 64-bit LSB executable"; then
    echo "ERROR: $OUT is not a valid x86-64 ELF" >&2; file "$OUT"; exit 1
fi
echo "ELF: $(file -b "$OUT")"

# 3. Disassembly (audit-only)
if command -v objdump >/dev/null; then
    objdump -D -b binary -m i386:x86-64 "$OUT" > disasm.txt 2>/dev/null || true
    echo "Wrote disasm.txt ($(wc -l < disasm.txt) lines)"
fi

# 4. SHA-256 reproducibility check
NEW=$(sha256sum "$OUT")
if [ -f catm.sha256 ]; then
    if [ "$(cat catm.sha256)" = "$NEW" ]; then
        echo "SHA-256: ${NEW%% *}  (matches catm.sha256)"
    else
        echo "ERROR: SHA-256 mismatch (non-reproducible build)" >&2
        echo "  recorded: $(cat catm.sha256)" >&2
        echo "  rebuilt:  $NEW" >&2
        exit 1
    fi
else
    echo "$NEW" > catm.sha256
    echo "SHA-256 recorded: ${NEW%% *}"
fi

# 5. Tests
bash run_tests.sh
