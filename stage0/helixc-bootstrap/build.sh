#!/usr/bin/env bash
# build.sh -- compile the helixc-bootstrap seed (seed.c, Apache-2.0) using ONLY
# our stage0 ladder: M2-Planet (rung 7) compiles it, then catm + M0 + hex2 link
# it into a self-contained ELF. No external toolchain. Run under WSL.
#
# Trust chain: hex0 (hand-authored) -> hex1 -> hex2 -> catm -> M0 -> cc_amd64 ->
# M2-Planet -> (this seed). The seed source is OURS (Apache-2.0); we only BUILD
# with the GPL-3.0 vendored M2-Planet/M2libc, copying none of their source here.
set -euo pipefail
cd "$(dirname "$0")"

M2=../M2-Planet/M2.bin
CATM=../catm/catm.bin
M0=../M0/M0.bin
HEX2=../hex2/hex2.bin
LIB=../M2-Planet/M2libc
MDEFS=$LIB/amd64/amd64_defs.M1
MLIBC=$LIB/amd64/libc-core.M1
MELF=$LIB/amd64/ELF-amd64.hex2
OUT=seed.bin

for t in "$M2" "$CATM" "$M0" "$HEX2"; do [ -x "$t" ] || chmod +x "$t"; done

# 1. M2-Planet compiles seed.c (+ the bootstrap libc) to M1 assembly.
T=$(mktemp -d)
"$M2" --architecture amd64 \
    -f "$LIB/amd64/linux/bootstrap.c" \
    -f "$LIB/bootstrap.c" \
    -f "$LIB/bootstrappable.c" \
    -f seed.c \
    --bootstrap-mode -o "$T/seed.M1"

# 2. Assemble M2's output with M2libc's amd64 defs (lesson 30: M2 output pairs
#    with M2libc/amd64 defs, NOT cc_amd64's), then link to a self-contained ELF.
"$CATM" "$T/seed-0.M1"   "$MDEFS" "$MLIBC" "$T/seed.M1"
"$M0"   "$T/seed-0.M1"   "$T/seed.hex2"
"$CATM" "$T/seed-0.hex2" "$MELF" "$T/seed.hex2"
"$HEX2" "$T/seed-0.hex2" "$OUT"
rm -rf "$T"
chmod +x "$OUT"
echo "Built $OUT: $(stat -c%s "$OUT") bytes (M2-Planet -> catm -> M0 -> catm -> hex2)"

# 3. ELF sanity
if ! file "$OUT" | grep -q "ELF 64-bit LSB executable"; then
    echo "ERROR: $OUT is not a valid x86-64 ELF" >&2; file "$OUT"; exit 1
fi
echo "ELF: $(file -b "$OUT")"

# 4. Tests
bash run_tests.sh
