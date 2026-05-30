#!/usr/bin/env bash
# run_tests.sh -- exercise hex1.bin. hex1 uses argv I/O: `hex1 INPUT OUTPUT`
# (unlike our stdin/stdout hex0). Run under WSL.
set -u
cd "$(dirname "$0")"
PASS=0; FAIL=0
T=$(mktemp -d)

# 1. hex1 is a superset of hex0: plain hex pairs decode to exact bytes.
printf '48 65 6C 6C 6F 0A\n' > "$T/plain.hex1"
./hex1.bin "$T/plain.hex1" "$T/plain.out" 2>/dev/null
got=$(od -An -tx1 "$T/plain.out" | tr -d ' \n')
if [ "$got" = "48656c6c6f0a" ]; then
    echo "PASS 01-plain-hex"; PASS=$((PASS+1))
else
    echo "FAIL 01-plain-hex (got '$got', want 48656c6c6f0a)"; FAIL=$((FAIL+1))
fi

# 2. hex1's LABEL feature: assemble the real upstream hex2 source (which uses
#    single-char labels) and confirm it yields a valid x86-64 ELF.
./hex1.bin test/02-hex2-source.hex1 "$T/hex2.out" 2>/dev/null
if file "$T/hex2.out" | grep -q "ELF 64-bit"; then
    echo "PASS 02-labels-hex2 ($(stat -c%s "$T/hex2.out") byte ELF)"; PASS=$((PASS+1))
else
    echo "FAIL 02-labels-hex2"; FAIL=$((FAIL+1))
fi

rm -rf "$T"
echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
