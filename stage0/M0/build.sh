#!/usr/bin/env bash
# build.sh -- build + verify M0.bin from the vendored M0_AMD64.hex2 using ONLY
# prior rungs (catm + hex2). M0 is the macro assembler (turns M1 assembly into
# hex2). Run under WSL (Linux ELF). No assembler.
#
# Trust chain: hex0 (hand-authored, frozen) -> hex1 -> hex2 -> catm -> M0.
# Sources GPL-3.0, vendored from oriansj/stage0-posix-amd64 -- see PROVENANCE.md.
set -euo pipefail
cd "$(dirname "$0")"

CATM=../catm/catm.bin
HEX2=../hex2/hex2.bin
OUT=M0.bin

[ -x "$CATM" ] || chmod +x "$CATM"
[ -x "$HEX2" ] || chmod +x "$HEX2"

# 1. Build (canonical mescc-tools phase-3): catm prepends the ELF header to the
#    M0 program, then hex2 assembles the result.
T=$(mktemp -d)
"$CATM" "$T/M0.hex2" ELF-amd64.hex2 M0_AMD64.hex2
"$HEX2" "$T/M0.hex2" "$OUT"
rm -rf "$T"
chmod +x "$OUT"
echo "Built $OUT: $(stat -c%s "$OUT") bytes (via ../catm + ../hex2)"

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
if [ -f M0.sha256 ]; then
    if [ "$(cat M0.sha256)" = "$NEW" ]; then
        echo "SHA-256: ${NEW%% *}  (matches M0.sha256)"
    else
        echo "ERROR: SHA-256 mismatch (non-reproducible build)" >&2
        echo "  recorded: $(cat M0.sha256)" >&2
        echo "  rebuilt:  $NEW" >&2
        exit 1
    fi
else
    echo "$NEW" > M0.sha256
    echo "SHA-256 recorded: ${NEW%% *}"
fi

# 5. Tests
bash run_tests.sh
