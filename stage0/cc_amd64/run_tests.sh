#!/usr/bin/env bash
# run_tests.sh -- exercise cc_amd64.bin, the minimal C compiler (C subset -> M1).
# argv: cc_amd64 INPUT.c OUTPUT.M1. Run under WSL.
#
# The definitive test: compile a C program end-to-end through the WHOLE ladder
# (cc_amd64 -> catm(defs,libc) -> M0 -> catm(ELF) -> hex2) and RUN the result.
# A correct exit code proves cc_amd64 emits correct machine code, not just bytes.
set -u
cd "$(dirname "$0")"
PASS=0; FAIL=0
T=$(mktemp -d)
M0=../M0/M0.bin
CATM=../catm/catm.bin
HEX2=../hex2/hex2.bin

# build_and_run <name> <c-source> <expected-exit>
build_and_run() {
    local name="$1" src="$2" want="$3"
    printf '%s' "$src" > "$T/$name.c"
    ./cc_amd64.bin "$T/$name.c" "$T/$name.M1" 2>/dev/null
    "$CATM" "$T/$name.full.M1" amd64_defs.M1 libc-core.M1 "$T/$name.M1" 2>/dev/null
    "$M0"   "$T/$name.full.M1" "$T/$name.hex2" 2>/dev/null
    "$CATM" "$T/$name.full.hex2" ELF-amd64.hex2 "$T/$name.hex2" 2>/dev/null
    "$HEX2" "$T/$name.full.hex2" "$T/$name.bin" 2>/dev/null
    chmod +x "$T/$name.bin"
    "$T/$name.bin"; local rc=$?
    if [ "$rc" = "$want" ]; then
        echo "PASS $name (exit $rc)"; PASS=$((PASS+1))
    else
        echo "FAIL $name (exit $rc, want $want)"; FAIL=$((FAIL+1))
    fi
}

# 1. The classic: a function returning a constant, run through the toolchain.
build_and_run 01-return-42 'int main() {
    return 42;
}
' 42

# 2. Arithmetic + a local variable -- exercises codegen beyond a bare constant.
build_and_run 02-arith 'int main() {
    int x;
    x = 6;
    return x * 7;
}
' 42

rm -rf "$T"
echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
