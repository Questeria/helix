#!/usr/bin/env bash
# build.sh -- build + verify hex2.bin from the vendored hex2_AMD64.hex1 using
# ONLY the prior rung (hex1). Run under WSL (Linux ELF). No assembler.
#
# Trust chain: hex0 (hand-authored, frozen) built hex1; hex1 builds hex2 here.
# hex2 source is GPL-3.0, vendored from oriansj/stage0-posix-amd64 -- see
# PROVENANCE.md. No pre-built binary is trusted.
set -euo pipefail
cd "$(dirname "$0")"

HEX1=../hex1/hex1.bin
SRC=hex2_AMD64.hex1
OUT=hex2.bin

[ -x "$HEX1" ] || chmod +x "$HEX1"

# 1. Build: hex1 assembles the hex2 source (hex + single-char labels) -> hex2.bin
"$HEX1" "$SRC" "$OUT"
chmod +x "$OUT"
echo "Built $OUT: $(stat -c%s "$OUT") bytes (via ../hex1/hex1.bin)"

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
if [ -f hex2.sha256 ]; then
    if [ "$(cat hex2.sha256)" = "$NEW" ]; then
        echo "SHA-256: ${NEW%% *}  (matches hex2.sha256)"
    else
        echo "ERROR: SHA-256 mismatch (non-reproducible build)" >&2
        echo "  recorded: $(cat hex2.sha256)" >&2
        echo "  rebuilt:  $NEW" >&2
        exit 1
    fi
else
    echo "$NEW" > hex2.sha256
    echo "SHA-256 recorded: ${NEW%% *}"
fi

# 5. Tests
bash run_tests.sh
