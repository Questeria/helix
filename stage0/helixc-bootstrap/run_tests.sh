#!/usr/bin/env bash
# run_tests.sh -- exercise the helixc-bootstrap seed. Run under WSL.
# Grows one test per increment; increment 0 = the arena self-test.
set -u
cd "$(dirname "$0")"
PASS=0; FAIL=0

# The seed's built-in self-test returns 42 when all internal asserts pass; a
# small diagnostic code otherwise. Grows with each increment.
#   inc 0: arena push/get/set + while-sum -> 42.
#   inc 1: lexer -- assert the 17-token stream of the sample program.
#   inc 2a: expression parser -- assert precedence (2+3*4), parens override
#           ((2+3)*4), and a call f(7,x) build the correct ASTs.
./seed.bin; rc=$?
if [ "$rc" = "42" ]; then
    echo "PASS 02-lexer+exprparser-selftest (exit $rc)"; PASS=$((PASS+1))
else
    echo "FAIL 02-lexer+exprparser-selftest (exit $rc -- diagnostic index; want 42)"; FAIL=$((FAIL+1))
fi

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
