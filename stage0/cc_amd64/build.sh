#!/usr/bin/env bash
# build.sh -- build + verify cc_amd64.bin from the vendored cc_amd64.M1 using
# ONLY prior rungs (M0 + catm + hex2). cc_amd64 is the minimal C compiler
# (compiles a C subset -> M1 assembly); it bootstraps M2-Planet. Run under WSL.
#
# Trust chain: hex0 (hand-authored, frozen) -> hex1 -> hex2 -> catm -> M0 ->
# cc_amd64. Sources GPL-3.0, vendored from oriansj/stage0-posix-amd64 -- see
# PROVENANCE.md. No pre-built binary is trusted.
set -euo pipefail
cd "$(dirname "$0")"

M0=../M0/M0.bin
CATM=../catm/catm.bin
HEX2=../hex2/hex2.bin
OUT=cc_amd64.bin

for t in "$M0" "$CATM" "$HEX2"; do [ -x "$t" ] || chmod +x "$t"; done

# 1. Build (mescc-tools phase-4): M0 assembles cc_amd64.M1 -> hex2, catm
#    prepends the ELF header, hex2 assembles the binary.
T=$(mktemp -d)
"$M0"   cc_amd64.M1 "$T/cc.hex2"
"$CATM" "$T/cc_full.hex2" ELF-amd64.hex2 "$T/cc.hex2"
"$HEX2" "$T/cc_full.hex2" "$OUT"
rm -rf "$T"
chmod +x "$OUT"
echo "Built $OUT: $(stat -c%s "$OUT") bytes (via ../M0 + ../catm + ../hex2)"

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
if [ -f cc_amd64.sha256 ]; then
    if [ "$(cat cc_amd64.sha256)" = "$NEW" ]; then
        echo "SHA-256: ${NEW%% *}  (matches cc_amd64.sha256)"
    else
        echo "ERROR: SHA-256 mismatch (non-reproducible build)" >&2
        echo "  recorded: $(cat cc_amd64.sha256)" >&2
        echo "  rebuilt:  $NEW" >&2
        exit 1
    fi
else
    echo "$NEW" > cc_amd64.sha256
    echo "SHA-256 recorded: ${NEW%% *}"
fi

# 5. Tests
bash run_tests.sh
