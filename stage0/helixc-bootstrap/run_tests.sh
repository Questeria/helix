#!/usr/bin/env bash
# run_tests.sh -- exercise the helixc-bootstrap seed. Run under WSL.
# Grows one test per increment; increment 0 = the arena self-test.
set -u
cd "$(dirname "$0")"
PASS=0; FAIL=0

# The seed's built-in self-test returns 42 when all internal asserts pass; a
# small diagnostic code otherwise. Grows with each increment.
#   inc 0: arena push/get/set + while-sum -> 42.
#   inc 1: lex `fn main() -> i32 { let x = 41; x + 1 }` and assert the 17-token
#          stream (tags + the int values 41,1).
./seed.bin; rc=$?
if [ "$rc" = "42" ]; then
    echo "PASS 01-lexer-selftest (exit $rc)"; PASS=$((PASS+1))
else
    echo "FAIL 01-lexer-selftest (exit $rc -- diagnostic index; want 42)"; FAIL=$((FAIL+1))
fi

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
