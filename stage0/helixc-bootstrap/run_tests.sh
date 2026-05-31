#!/usr/bin/env bash
# run_tests.sh -- exercise the helixc-bootstrap seed. Run under WSL.
#   no args  -> seed runs its lexer + parser self-tests (exit 42).
#   in out   -> seed compiles a .hx file to a self-contained ELF.
set -u
cd "$(dirname "$0")"
PASS=0; FAIL=0
T=$(mktemp -d)

# 1. self-test mode: lexer + expr parser + full parser asserts -> exit 42.
./seed.bin; rc=$?
if [ "$rc" = "42" ]; then
    echo "PASS 03-frontend-selftest (exit $rc)"; PASS=$((PASS+1))
else
    echo "FAIL 03-frontend-selftest (exit $rc -- diagnostic index; want 42)"; FAIL=$((FAIL+1))
fi

# 2. codegen (inc 3a): compile `fn main() -> i32 { 42 }` to an ELF, run it,
#    assert exit 42 -- proves the seed emits a correct, runnable x86-64 binary.
./seed.bin test/t1.hx "$T/t1.bin" 2>/dev/null; crc=$?
chmod +x "$T/t1.bin" 2>/dev/null
"$T/t1.bin"; rc=$?
if [ "$crc" = "0" ] && [ "$rc" = "42" ]; then
    echo "PASS 3a-compile-return42 (compile rc=$crc, run exit=$rc)"; PASS=$((PASS+1))
else
    echo "FAIL 3a-compile-return42 (compile rc=$crc, run exit=$rc, want 0/42)"; FAIL=$((FAIL+1))
fi

rm -rf "$T"
echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
