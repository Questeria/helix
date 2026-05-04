#!/usr/bin/env bash
# build.sh — assemble hex0.s with nasm (CROSS-CHECK ONLY) and run tests.
#
# IMPORTANT: nasm output is the *cross-check reference*, not the shipped artifact.
# The shipped artifact is hex0.bin which is hand-encoded from hex0.bytes.md.
# This script:
#   1. Assembles hex0.s with nasm -> hex0.nasm.bin (the reference)
#   2. If hex0.bin exists (hand-encoded), `cmp` with hex0.nasm.bin (must be identical)
#   3. Runs the test suite on hex0.bin
#
# Phase 0a: hex0.bin doesn't exist yet; we use hex0.nasm.bin as a working binary
# while we hand-encode hex0.bytes.md. Once hex0.bin is hand-typed, this script
# enforces the byte-identical cross-check.

set -euo pipefail
cd "$(dirname "$0")"

REFERENCE=hex0.nasm.bin
SHIPPED=hex0.bin

# 1. Assemble reference with nasm (Linux ELF64)
if ! command -v nasm > /dev/null; then
    echo "ERROR: nasm not installed. apt-get install nasm" >&2
    exit 1
fi
nasm -f bin -o "$REFERENCE" hex0.s
chmod +x "$REFERENCE"
echo "Assembled $REFERENCE: $(stat -c%s "$REFERENCE") bytes"

# 2. Cross-check if hex0.bin exists
if [[ -f "$SHIPPED" ]]; then
    if ! cmp "$REFERENCE" "$SHIPPED"; then
        echo "ERROR: $SHIPPED differs from nasm reference $REFERENCE" >&2
        echo "Hand-encoded bytes do not match — fix hex0.bytes.md" >&2
        exit 2
    fi
    echo "OK: $SHIPPED matches nasm reference (byte-identical)"
    BINARY="$SHIPPED"
else
    echo "NOTE: $SHIPPED not yet hand-encoded; using $REFERENCE for tests"
    BINARY="$REFERENCE"
fi

# 3. Run tests
PASS=0
FAIL=0
for hex_file in test/*.hex0; do
    name="$(basename "$hex_file" .hex0)"
    expected_file="test/$name.expected"
    [[ -f "$expected_file" ]] || { echo "SKIP $name (no .expected)"; continue; }

    actual=$(./"$BINARY" < "$hex_file" 2>/dev/null || true)
    expected=$(cat "$expected_file")

    if [[ "$actual" == "$expected" ]]; then
        echo "PASS $name"
        PASS=$((PASS + 1))
    else
        echo "FAIL $name"
        echo "  expected: $(printf '%q' "$expected")"
        echo "  actual:   $(printf '%q' "$actual")"
        FAIL=$((FAIL + 1))
    fi
done

echo
echo "Results: $PASS passed, $FAIL failed"
[[ "$FAIL" -eq 0 ]]
