#!/usr/bin/env bash
# Run every LIVE gate, from project root, under WSL/Linux. Python-free (post-K4).
#
# The pre-K4 Python pytest gate (scripts/stage31_validate.py orchestrating
# helixc/tests/test_codegen.py et al.) was deleted at K4 along with the entire
# Python reference compiler and its test suite. Those orchestrators referenced
# now-deleted files and are themselves retired (preserved at tag
# v0-pre-k4-full-with-python). The live gates below are Helix/raw-binary only:
#
#   1. stage0/hex0           -- the raw-binary trust floor (hand-typed hex0 ladder)
#   2. stage0/helixc-bootstrap/build.sh -- builds the C seed from the ladder and
#      runs 17 self-host regression tests, which compile AND run real .hx programs
#      through the seed-built toolchain (arithmetic, precedence, locals, while,
#      recursion, arena, file IO, the real lexer) -- i.e. live mint coverage.
#
# A heavier full self-host fixpoint check (seed -> K1' -> K2 -> K3 byte-identical,
# a slow ~10 min mint of the 1.5 MB compiler source) is a separate gate
# (validate_selfhost, P0.3 of docs/HELIX_FINISH_PLAN.md) and is intentionally not
# part of this quick gate.
set -u
cd "$(dirname "$0")/.."

echo "============================="
echo "stage0/hex0 (raw-binary trust floor):"
if [[ -f stage0/hex0/run_tests.sh ]]; then
    if (cd stage0/hex0 && bash run_tests.sh); then HEX0_RC=0; else HEX0_RC=$?; fi
else
    echo "stage0/hex0/run_tests.sh not found"
    HEX0_RC=1
fi

echo
echo "stage0/helixc-bootstrap (seed build from ladder + 17 self-host tests):"
if [[ -f stage0/helixc-bootstrap/build.sh ]]; then
    if (cd stage0/helixc-bootstrap && bash build.sh); then SEED_RC=0; else SEED_RC=$?; fi
else
    echo "stage0/helixc-bootstrap/build.sh not found"
    SEED_RC=1
fi

echo
echo "============================="
echo "stage0/hex0 rc:                    $HEX0_RC"
echo "seed build + 17 self-host tests rc: $SEED_RC"

if [[ "$HEX0_RC" -eq 0 && "$SEED_RC" -eq 0 ]]; then
    echo "TOTAL: all live gates passed (Python-free)"
    exit 0
fi

echo "TOTAL: one or more gates failed"
exit 1
