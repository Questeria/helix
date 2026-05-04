#!/usr/bin/env bash
# build.sh — build and verify hex0.bin from hex0.hex (the annotated source)
#
# Pipeline:
#   1. Strip comments + whitespace from hex0.hex
#   2. Convert hex pairs to binary via `xxd -r -p` (audit-only tool — no
#      assembler involved)
#   3. Verify the file is a valid x86-64 ELF
#   4. Disassemble with objdump and dump to disasm.txt for inspection
#   5. Verify SHA-256 against hex0.sha256 (reproducibility check)
#   6. Run all tests under test/
#
# Tools used (all permitted by project's hard constraint):
#   xxd      — audit-only, hex<->bin only
#   file     — audit-only, header sniffing
#   objdump  — audit-only, disassembly
#   sha256sum — integrity
#
# NOT used: nasm, as, gcc, ld, clang. The bytes in hex0.hex are the source of truth.

set -euo pipefail
cd "$(dirname "$0")"

OUT=hex0.bin
SRC=hex0.hex

# 1. Hex source -> binary
grep -v '^;' "$SRC" | sed 's/;.*//' | tr -d '[:space:]' | xxd -r -p > "$OUT"
chmod +x "$OUT"

SIZE=$(stat -c%s "$OUT")
echo "Built $OUT: $SIZE bytes"

# 2. ELF sanity check
if ! file "$OUT" | grep -q "ELF 64-bit LSB executable"; then
    echo "ERROR: $OUT is not a valid x86-64 ELF" >&2
    file "$OUT"
    exit 1
fi
echo "ELF: $(file -b "$OUT")"

# 3. Disassemble (audit dump)
objdump -D -b binary -m i386:x86-64 -M intel \
    --adjust-vma=0x600000 --start-address=0x600078 \
    "$OUT" > disasm.txt
echo "Wrote disasm.txt ($(wc -l < disasm.txt) lines)"

# 4. SHA-256 reproducibility
ACTUAL_SHA=$(sha256sum "$OUT" | cut -d' ' -f1)
if [[ -f hex0.sha256 ]]; then
    EXPECTED_SHA=$(cut -d' ' -f1 hex0.sha256)
    if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
        echo "ERROR: $OUT SHA-256 mismatch" >&2
        echo "  expected: $EXPECTED_SHA"
        echo "  actual:   $ACTUAL_SHA"
        exit 1
    fi
    echo "SHA-256: $ACTUAL_SHA  (matches hex0.sha256)"
else
    echo "$ACTUAL_SHA  $OUT" > hex0.sha256
    echo "SHA-256: $ACTUAL_SHA  (recorded in hex0.sha256)"
fi

# 5. Behavioral tests
echo
bash run_tests.sh
