#!/usr/bin/env bash
# run_tests.sh -- exercise M2 (M2-Planet), the full self-hosting C compiler.
# Usage: M2 --architecture amd64 -f <src.c>... [--bootstrap-mode] -o <out.M1>.
# Run under WSL.
#
# Capability test: M2 compiles a C program end-to-end and we RUN the result.
# NOTE: M2-Planet emits M1 that pairs with M2libc's OWN amd64 defs (a different
# calling convention than cc_amd64's), so its output is assembled with the
# vendored M2libc/amd64/*.M1 -- NOT cc_amd64's defs (which would segfault).
set -u
cd "$(dirname "$0")"
PASS=0; FAIL=0
T=$(mktemp -d)
CATM=../catm/catm.bin
M0=../M0/M0.bin
HEX2=../hex2/hex2.bin
MDEFS=M2libc/amd64/amd64_defs.M1
MLIBC=M2libc/amd64/libc-core.M1
MELF=M2libc/amd64/ELF-amd64.hex2

# m2_build_run <name> <c-source> <expected-exit>
m2_build_run() {
    local name="$1" src="$2" want="$3"
    printf '%s' "$src" > "$T/$name.c"
    ./M2.bin --architecture amd64 \
        -f M2libc/amd64/linux/bootstrap.c \
        -f M2libc/bootstrap.c \
        -f M2libc/bootstrappable.c \
        -f "$T/$name.c" --bootstrap-mode -o "$T/$name.M1" 2>/dev/null
    "$CATM" "$T/$name-0.M1" "$MDEFS" "$MLIBC" "$T/$name.M1" 2>/dev/null
    "$M0"   "$T/$name-0.M1" "$T/$name.hex2" 2>/dev/null
    "$CATM" "$T/$name-0.hex2" "$MELF" "$T/$name.hex2" 2>/dev/null
    "$HEX2" "$T/$name-0.hex2" "$T/$name.bin" 2>/dev/null
    chmod +x "$T/$name.bin"
    "$T/$name.bin"; local rc=$?
    if [ "$rc" = "$want" ]; then
        echo "PASS $name (exit $rc)"; PASS=$((PASS+1))
    else
        echo "FAIL $name (exit $rc, want $want)"; FAIL=$((FAIL+1))
    fi
}

# 1. A function returning a constant, built through the whole ladder + run.
m2_build_run 01-return-42 'int main() { return 42; }
' 42

# 2. Arithmetic + a local -- exercises codegen beyond a bare constant.
m2_build_run 02-arith 'int main() {
    int x;
    x = 6;
    return x * 7;
}
' 42

rm -rf "$T"
echo
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
