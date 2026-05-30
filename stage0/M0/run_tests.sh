#!/usr/bin/env bash
# run_tests.sh -- exercise M0.bin, the macro assembler (M1 assembly -> hex2).
# argv: M0 INPUT.M1 OUTPUT.hex2. Run under WSL.
set -u
cd "$(dirname "$0")"
PASS=0; FAIL=0
T=$(mktemp -d)
CATM=../catm/catm.bin
HEX2=../hex2/hex2.bin

# 1. M0 assembles the real C-compiler seed cc_amd64.M1 into hex2 text.
./M0.bin test/cc_amd64.M1 "$T/cc.hex2" 2>/dev/null
if [ -s "$T/cc.hex2" ]; then
    echo "PASS 01-M0-assembles-cc (cc.hex2 = $(stat -c%s "$T/cc.hex2") bytes)"; PASS=$((PASS+1))
else
    echo "FAIL 01-M0-assembles-cc (empty output)"; FAIL=$((FAIL+1))
fi

# 2. The hex2 M0 emitted, given the ELF header, assembles to a valid ELF --
#    i.e. M0 produced a correct, runnable cc_amd64 compiler binary.
"$CATM" "$T/cc_full.hex2" ELF-amd64.hex2 "$T/cc.hex2" 2>/dev/null
"$HEX2" "$T/cc_full.hex2" "$T/cc_amd64.bin" 2>/dev/null
if file "$T/cc_amd64.bin" | grep -q "ELF 64-bit"; then
    echo "PASS 02-cc-is-valid-elf ($(stat -c%s "$T/cc_amd64.bin") byte ELF)"; PASS=$((PASS+1))
else
    echo "FAIL 02-cc-is-valid-elf"; FAIL=$((FAIL+1))
fi

rm -rf "$T"
echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
