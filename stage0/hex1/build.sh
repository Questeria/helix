#!/usr/bin/env bash
# build.sh -- build + verify hex1.bin from the vendored hex1_AMD64.hex0 using
# ONLY the prior rung (hex0). Run under WSL (Linux ELF). No assembler is used.
#
# Trust: hex1.bin is produced by feeding the auditable hex1 SOURCE
# (hex1_AMD64.hex0, GPL-3.0, vendored from oriansj/stage0-posix-amd64 -- see
# PROVENANCE.md) through OUR hand-authored, verified hex0. No pre-built binary
# is trusted; the byte-for-byte root is stage0/hex0/.
set -euo pipefail
cd "$(dirname "$0")"

HEX0=../hex0/hex0.bin
SRC=hex1_AMD64.hex0
OUT=hex1.bin

[ -x "$HEX0" ] || chmod +x "$HEX0"

# 1. Build: hex0 decodes the hex1 source (hex pairs + '#'/';' comments) -> hex1.bin
"$HEX0" < "$SRC" > "$OUT"
chmod +x "$OUT"
echo "Built $OUT: $(stat -c%s "$OUT") bytes (via ../hex0/hex0.bin)"

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
if [ -f hex1.sha256 ]; then
    if [ "$(cat hex1.sha256)" = "$NEW" ]; then
        echo "SHA-256: ${NEW%% *}  (matches hex1.sha256)"
    else
        echo "ERROR: SHA-256 mismatch (non-reproducible build)" >&2
        echo "  recorded: $(cat hex1.sha256)" >&2
        echo "  rebuilt:  $NEW" >&2
        exit 1
    fi
else
    echo "$NEW" > hex1.sha256
    echo "SHA-256 recorded: ${NEW%% *}"
fi

# 5. Tests
bash run_tests.sh
