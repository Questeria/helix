#!/usr/bin/env bash
# run_tests.sh — execute hex0.bin against all test fixtures
set -u
cd "$(dirname "$0")"

PASS=0
FAIL=0
for hex_file in test/*.hex0; do
    name=$(basename "$hex_file" .hex0)
    expected_file="test/$name.expected"
    [ -f "$expected_file" ] || continue
    actual=$(./hex0.bin < "$hex_file" 2>/dev/null || true)
    expected=$(cat "$expected_file")
    if [ "$actual" = "$expected" ]; then
        echo "PASS $name"
        PASS=$((PASS+1))
    else
        echo "FAIL $name"
        echo "  expected: $(printf %q "$expected")"
        echo "  actual:   $(printf %q "$actual")"
        FAIL=$((FAIL+1))
    fi
done
echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
