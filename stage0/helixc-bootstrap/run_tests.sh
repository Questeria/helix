#!/usr/bin/env bash
# run_tests.sh -- exercise the helixc-bootstrap seed. Run under WSL.
# Grows one test per increment; increment 0 = the arena self-test.
set -u
cd "$(dirname "$0")"
PASS=0; FAIL=0

# Increment 0: the seed's own self-test returns 42 via the arena
# (push 6,7,28 -> set slot2=29 -> while-sum -> 42). Proves the build pipeline
# and the global-arena primitive under M2-Planet.
./seed.bin; rc=$?
if [ "$rc" = "42" ]; then
    echo "PASS 00-arena-selftest (exit $rc)"; PASS=$((PASS+1))
else
    echo "FAIL 00-arena-selftest (exit $rc, want 42)"; FAIL=$((FAIL+1))
fi

echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
