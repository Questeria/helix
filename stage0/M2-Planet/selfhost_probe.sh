#!/usr/bin/env bash
# selfhost_probe.sh -- H6 bounded attempt: does M2 self-host to a byte-stable
# fixpoint when assembled with libc-full.M1 (the PROVENANCE hypothesis)?
#   gen1 = M2.bin (built by cc_amd64, existing).
#   gen2.M1 = gen1 compiling the 12 M2-Planet sources; assemble -> gen2-M2.bin.
#   CAPABILITY: gen2-M2.bin compiles `return 42` and runs -> 42 (else SIGILL => deeper than libc).
#   gen3.M1 = gen2-M2.bin compiling the same 12 sources.
#   FIXPOINT: gen2.M1 == gen3.M1 byte-identical.
# Run under WSL. Bounded: if CAPABILITY fails, STOP (do not deep-debug vendored codegen).
set -u
cd "$(dirname "$0")"
PIN=b8bb2a0159a7376716a396ec6f6bc29dd27857b5
CATM=../catm/catm.bin
M0=../M0/M0.bin
HEX2=../hex2/hex2.bin
MDEFS=M2libc/amd64/amd64_defs.M1
MFULL=M2libc/amd64/libc-full.M1
MELF=M2libc/amd64/ELF-amd64.hex2
T=$(mktemp -d)

# M2-Planet source list, build.sh order (the exact 12 -f units M2 must self-compile).
SRCS="-f M2libc/amd64/linux/bootstrap.c -f M2libc/bootstrap.c -f M2-Planet/cc.h -f M2libc/bootstrappable.c -f M2-Planet/cc_globals.c -f M2-Planet/cc_reader.c -f M2-Planet/cc_strings.c -f M2-Planet/cc_types.c -f M2-Planet/cc_emit.c -f M2-Planet/cc_core.c -f M2-Planet/cc_macro.c -f M2-Planet/cc.c"

echo "=== [0] vendor libc-full.M1 @ pin $PIN ==="
if [ ! -s "$MFULL" ]; then
    curl -fsSL --max-time 30 "https://raw.githubusercontent.com/oriansj/M2libc/$PIN/amd64/libc-full.M1" -o "$MFULL" || { echo "FETCH FAILED"; exit 3; }
fi
echo "  libc-full.M1: $(stat -c%s "$MFULL") bytes  sha256=$(sha256sum "$MFULL" | cut -d' ' -f1)"
[ -s "$MFULL" ] || { echo "FATAL: libc-full.M1 empty"; exit 3; }

# assemble <in.M1> <out.bin> <libc.M1>
assemble() {
    "$CATM" "$T/a-0.M1" "$MDEFS" "$3" "$1" 2>/dev/null
    "$M0"   "$T/a-0.M1" "$T/a.hex2" 2>/dev/null
    "$CATM" "$T/a-0.hex2" "$MELF" "$T/a.hex2" 2>/dev/null
    "$HEX2" "$T/a-0.hex2" "$2" 2>/dev/null
    chmod +x "$2"
}

echo "=== [1] gen2.M1 = M2.bin self-compiles the 12 sources ==="
./M2.bin --architecture amd64 $SRCS --bootstrap-mode -o "$T/gen2.M1" 2>"$T/g2err"; rc=$?
echo "  M2.bin compile rc=$rc; gen2.M1 = $([ -s "$T/gen2.M1" ] && stat -c%s "$T/gen2.M1" || echo 0) bytes"
[ -s "$T/gen2.M1" ] || { echo "  gen2.M1 EMPTY -- compile failed:"; head -3 "$T/g2err"; echo "VERDICT: SELFHOST_FAIL (gen2 compile)"; rm -rf "$T"; exit 1; }
assemble "$T/gen2.M1" "$T/gen2-M2.bin" "$MFULL"
echo "  gen2-M2.bin = $([ -s "$T/gen2-M2.bin" ] && stat -c%s "$T/gen2-M2.bin" || echo 0) bytes; ELF: $(file -b "$T/gen2-M2.bin" 2>/dev/null | cut -c1-30)"

echo "=== [2] CAPABILITY: gen2-M2.bin compiles 'return 42' and runs ==="
printf 'int main() { return 42; }\n' > "$T/r42.c"
"$T/gen2-M2.bin" --architecture amd64 -f M2libc/amd64/linux/bootstrap.c -f M2libc/bootstrap.c -f M2libc/bootstrappable.c -f "$T/r42.c" --bootstrap-mode -o "$T/r42.M1" 2>"$T/caperr"; crc=$?
echo "  gen2 compile-of-r42 rc=$crc; r42.M1 = $([ -s "$T/r42.M1" ] && stat -c%s "$T/r42.M1" || echo 0) bytes"
if [ -s "$T/r42.M1" ]; then
    assemble "$T/r42.M1" "$T/r42.bin" "M2libc/amd64/libc-core.M1"
    "$T/r42.bin"; r42rc=$?
    echo "  r42.bin exit=$r42rc (want 42)"
    if [ "$r42rc" != "42" ]; then echo "VERDICT: CAPABILITY_FAIL (gen2 runs but wrong/illegal: $r42rc) -- deeper than libc-pairing; STOP + document"; rm -rf "$T"; exit 1; fi
else
    echo "  gen2 could not compile r42 (rc=$crc):"; head -3 "$T/caperr"
    echo "VERDICT: CAPABILITY_FAIL (gen2 non-functional) -- STOP + document"; rm -rf "$T"; exit 1
fi

echo "=== [3] gen3.M1 = gen2-M2.bin self-compiles the 12 sources ==="
"$T/gen2-M2.bin" --architecture amd64 $SRCS --bootstrap-mode -o "$T/gen3.M1" 2>"$T/g3err"; rc3=$?
echo "  gen2 compile rc=$rc3; gen3.M1 = $([ -s "$T/gen3.M1" ] && stat -c%s "$T/gen3.M1" || echo 0) bytes"
[ -s "$T/gen3.M1" ] || { echo "  gen3.M1 EMPTY:"; head -3 "$T/g3err"; echo "VERDICT: CAPABILITY_PARTIAL (gen2 runs simple C but cannot self-compile) -- STOP + document"; rm -rf "$T"; exit 1; }

echo "=== [4] FIXPOINT: gen2.M1 vs gen3.M1 ==="
s2=$(sha256sum "$T/gen2.M1" | cut -d' ' -f1); s3=$(sha256sum "$T/gen3.M1" | cut -d' ' -f1)
echo "  gen2.M1 sha=$s2"; echo "  gen3.M1 sha=$s3"
if [ "$s2" = "$s3" ]; then echo "VERDICT: SELFHOST_FIXPOINT_OK (gen2.M1 == gen3.M1 byte-identical)"; else echo "VERDICT: FIXPOINT_DIFF (gen2 runs + self-compiles, but gen2.M1 != gen3.M1) -- closer, document the residual diff"; fi
rm -rf "$T"
