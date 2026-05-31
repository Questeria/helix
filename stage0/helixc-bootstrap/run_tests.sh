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
#   inc 2a: expression parser -- precedence (2+3*4), parens, call f(7,x).
#   inc 2b: full parser -- parse whole fns; assert let-mut/assign/tail-expr and
#           while + if-expression AST shapes.
./seed.bin; rc=$?
if [ "$rc" = "42" ]; then
    echo "PASS 03-full-parser-selftest (exit $rc)"; PASS=$((PASS+1))
else
    echo "FAIL 03-full-parser-selftest (exit $rc -- diagnostic index; want 42)"; FAIL=$((FAIL+1))
fi

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
