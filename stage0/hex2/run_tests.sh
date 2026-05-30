#!/usr/bin/env bash
# run_tests.sh -- exercise hex2.bin (argv: hex2 INPUT OUTPUT). Run under WSL.
set -u
cd "$(dirname "$0")"
PASS=0; FAIL=0
T=$(mktemp -d)

# 1. hex2 is a superset of hex0/hex1: plain hex pairs decode to exact bytes.
printf '48 65 6C 6C 6F 0A\n' > "$T/plain.hex2"
./hex2.bin "$T/plain.hex2" "$T/plain.out" 2>/dev/null
got=$(od -An -tx1 "$T/plain.out" | tr -d ' \n')
if [ "$got" = "48656c6c6f0a" ]; then
    echo "PASS 01-plain-hex"; PASS=$((PASS+1))
else
    echo "FAIL 01-plain-hex (got '$got', want 48656c6c6f0a)"; FAIL=$((FAIL+1))
fi

# 2. hex2's feature (long labels + absolute addresses): assemble the REAL M0
#    source -- the canonical phase-3 recipe is `catm M0.hex2 ELF-amd64.hex2
#    M0_AMD64.hex2` then `hex2 M0.hex2 M0`. We use cat for the concatenation
#    (catm is the next-but-one rung). Result must be a valid x86-64 ELF.
cat test/ELF-amd64.hex2 test/M0_AMD64.hex2 > "$T/M0.hex2"
./hex2.bin "$T/M0.hex2" "$T/M0.bin" 2>/dev/null
if file "$T/M0.bin" | grep -q "ELF 64-bit"; then
    echo "PASS 02-builds-M0 ($(stat -c%s "$T/M0.bin") byte ELF)"; PASS=$((PASS+1))
else
    echo "FAIL 02-builds-M0"; FAIL=$((FAIL+1))
fi

rm -rf "$T"
echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
