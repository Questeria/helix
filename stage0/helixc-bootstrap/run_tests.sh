#!/usr/bin/env bash
# run_tests.sh -- exercise the helixc-bootstrap seed. Run under WSL.
#   no args  -> seed runs its lexer + parser self-tests (exit 42).
#   in out   -> seed compiles a .hx file to a self-contained ELF.
set -u
cd "$(dirname "$0")"
PASS=0; FAIL=0
T=$(mktemp -d)

# compile_run <name> <src.hx> <expected-exit>: seed compiles src -> ELF, run it.
compile_run() {
    local name="$1" src="$2" want="$3" crc rc
    ./seed.bin "$src" "$T/$name.bin" 2>/dev/null; crc=$?
    chmod +x "$T/$name.bin" 2>/dev/null
    "$T/$name.bin"; rc=$?
    if [ "$crc" = "0" ] && [ "$rc" = "$want" ]; then
        echo "PASS $name (compiled $src, run exit=$rc)"; PASS=$((PASS+1))
    else
        echo "FAIL $name (compile rc=$crc, run exit=$rc, want $want)"; FAIL=$((FAIL+1))
    fi
}

# 1. self-test mode: lexer + expr parser + full parser asserts -> exit 42.
./seed.bin; rc=$?
if [ "$rc" = "42" ]; then
    echo "PASS 03-frontend-selftest (exit $rc)"; PASS=$((PASS+1))
else
    echo "FAIL 03-frontend-selftest (exit $rc -- diagnostic index; want 42)"; FAIL=$((FAIL+1))
fi

# 2. codegen: compile + run Helix programs, assert their exit codes.
compile_run 3a-return42  test/t1.hx 42   # { 42 }
compile_run 3b-mul       test/t2.hx 42   # { 6 * 7 }
compile_run 3b-precedence test/t3.hx 14  # { 2 + 3 * 4 }  (mul binds tighter)
compile_run 3b-compare   test/t4.hx 1    # { 5 > 3 }  -> 1
compile_run 3c-local-mut test/t5.hx 42   # { let mut x = 41; x = x + 1; x }
compile_run 3c-two-locals test/t6.hx 42  # { let a = 6; let b = 7; a * b }
compile_run 3d-while-sum test/t7.hx 36   # sum 0..8 = 36 (while loop)
compile_run 3d-if-else   test/t8.hx 42   # if x > 5 then 42 (if as value)
compile_run 3d-factorial test/t9.hx 176  # 7! = 5040, exit code = 5040 & 255 = 176
compile_run 3e-call      test/t10.hx 42  # add(40, 2) -> 42 (call + params)
compile_run 3e-call-expr test/t11.hx 42  # sq(6) + 6 -> 42
compile_run 3e-recursion test/t12.hx 55  # fib(10) -> 55 (recursion)

rm -rf "$T"
echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
