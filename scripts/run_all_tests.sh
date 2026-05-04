#!/usr/bin/env bash
# Run every test suite. From project root.
set -u
cd "$(dirname "$0")/.."

TOTAL_PASS=0
TOTAL_FAIL=0

for test in kovc/tests/test_*.py; do
    name=$(basename "$test" .py)
    output=$(python "$test" 2>&1)
    last_line=$(echo "$output" | tail -1)
    if [[ "$last_line" =~ ^([0-9]+)\ passed,\ ([0-9]+)\ failed$ ]]; then
        pass="${BASH_REMATCH[1]}"
        fail="${BASH_REMATCH[2]}"
        TOTAL_PASS=$((TOTAL_PASS + pass))
        TOTAL_FAIL=$((TOTAL_FAIL + fail))
        if [[ "$fail" -gt 0 ]]; then
            echo "  FAIL  $name: $pass passed, $fail failed"
        else
            echo "  ok    $name: $pass passed"
        fi
    else
        echo "  ?     $name: unrecognized output: $last_line"
        TOTAL_FAIL=$((TOTAL_FAIL + 1))
    fi
done

# Also run hex0 tests
echo
echo "stage0/hex0:"
if wsl -- bash -c "cd /mnt/c/Projects/Kovostov-Native/stage0/hex0 && bash run_tests.sh 2>&1 | tail -3"; then
    :
fi

echo
echo "============================="
echo "TOTAL: $TOTAL_PASS passed, $TOTAL_FAIL failed"
exit $([[ "$TOTAL_FAIL" -eq 0 ]] && echo 0 || echo 1)
