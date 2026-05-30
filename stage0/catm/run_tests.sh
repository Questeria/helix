#!/usr/bin/env bash
# run_tests.sh -- exercise catm.bin (argv: catm OUTPUT in1 in2 ... inN).
# catm replaces cat/shell redirection in the bootstrap. Run under WSL.
set -u
cd "$(dirname "$0")"
PASS=0; FAIL=0
T=$(mktemp -d)

# 1. Concatenate two files, in order.
printf 'AAAA' > "$T/a"; printf 'BBBBBB' > "$T/b"
./catm.bin "$T/out2" "$T/a" "$T/b" 2>/dev/null
if [ "$(cat "$T/out2" 2>/dev/null)" = "AAAABBBBBB" ]; then
    echo "PASS 01-concat-2"; PASS=$((PASS+1))
else
    echo "FAIL 01-concat-2 (got '$(cat "$T/out2" 2>/dev/null)')"; FAIL=$((FAIL+1))
fi

# 2. Concatenate three files AND be binary-safe (middle input contains a NUL).
printf 'X'      > "$T/x"
printf 'Y\x00Z' > "$T/y"
printf 'W'      > "$T/w"
./catm.bin "$T/out3" "$T/x" "$T/y" "$T/w" 2>/dev/null
exp=$(printf 'XY\x00ZW' | od -An -tx1 | tr -d ' \n')
got=$(od -An -tx1 "$T/out3" 2>/dev/null | tr -d ' \n')
if [ "$got" = "$exp" ]; then
    echo "PASS 02-concat-3-binary"; PASS=$((PASS+1))
else
    echo "FAIL 02-concat-3-binary (got $got, want $exp)"; FAIL=$((FAIL+1))
fi

rm -rf "$T"
echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
