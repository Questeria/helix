#!/usr/bin/env bash
# build.sh -- build stage0/M1/M1.bin: the mescc-tools 1.7.0 M1 macro assembler.
#
# Built by M2.bin (the self-hosting C compiler from ../M2-Planet), NOT by
# cc_amd64 -- M1 uses FILE*/fopen/fgetc/fputc/stderr/EOF which cc_amd64's
# libc-core lacks. We mirror exactly how M2-Planet builds its OWN cc.c
# (../M2-Planet/run_tests.sh : m2_build_run): M2 compiles the tool source
# TOGETHER WITH the 3 M2libc bootstrap C units, then the M1 output is assembled
# with M2libc's amd64 defs+libc-core via catm -> M0 -> catm(ELF) -> hex2.
#
# M2 ignores #include by default (--expand-includes off), so every source unit
# the tool needs must be passed explicitly as a -f file:
#   * M2libc/amd64/linux/bootstrap.c  -- read/write/open/close/exit syscalls
#   * M2libc/bootstrap.c              -- FILE*/fopen/fgetc/fputc/EOF/NULL/malloc
#   * M2libc/bootstrappable.c         -- require/match/in_set/strtoint/int2str
#   * stringify.c                     -- stringify()/LittleEndian() (no header)
#   * M1.c                            -- the assembler itself (=upstream M1-macro.c)
# (M1's own #include "M2libc/bootstrappable.h" is a no-op under M2.)
#
# M0 is FINE on these small tools -- its lea-corruption bug is large-input-only
# (it only mangles M2's 2.2MB self-output). M1.bin is ~tens of KB.
#
# Run under WSL. Sources GPL-3.0, vendored from oriansj/mescc-tools 1.7.0
# (commit fa19e34) + oriansj/M2libc -- see ../M2-Planet/PROVENANCE.md.
set -euo pipefail
cd "$(dirname "$0")"

M2=../M2-Planet/M2.bin
CATM=../catm/catm.bin
M0=../M0/M0.bin
HEX2=../hex2/hex2.bin
# M2-emitted M1 pairs with M2libc's OWN amd64 calling convention (these live
# under ../M2-Planet/M2libc/amd64/, byte-identical to upstream M2libc).
MLIB=../M2-Planet/M2libc
MDEFS=$MLIB/amd64/amd64_defs.M1
MLIBC=$MLIB/amd64/libc-core.M1
MELF=$MLIB/amd64/ELF-amd64.hex2
OUT=M1.bin

for t in "$M2" "$CATM" "$M0" "$HEX2"; do [ -x "$t" ] || chmod +x "$t"; done

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

# 1. M2 compiles M1.c + the bootstrap C units -> M1 (intermediate assembly).
"$M2" --architecture amd64 \
    -f "$MLIB/amd64/linux/bootstrap.c" \
    -f "$MLIB/bootstrap.c" \
    -f "$MLIB/bootstrappable.c" \
    -f stringify.c \
    -f M1.c \
    --bootstrap-mode -o "$T/M1.M1"
echo "M2 -> M1.M1: $(stat -c%s "$T/M1.M1") bytes"

# 2. Prepend M2libc amd64 defs + libc-core, assemble with M0.
"$CATM" "$T/M1-0.M1" "$MDEFS" "$MLIBC" "$T/M1.M1"
"$M0"   "$T/M1-0.M1" "$T/M1.hex2"
echo "M0 -> M1.hex2: $(stat -c%s "$T/M1.hex2") bytes"

# 3. Prepend the ELF header hex2, link the final binary.
"$CATM" "$T/M1-0.hex2" "$MELF" "$T/M1.hex2"
"$HEX2" "$T/M1-0.hex2" "$OUT"
chmod +x "$OUT"
echo "Built $OUT: $(stat -c%s "$OUT") bytes"

# 4. ELF sanity.
if ! file "$OUT" | grep -q "ELF 64-bit LSB executable"; then
    echo "ERROR: $OUT is not a valid x86-64 ELF" >&2; file "$OUT"; exit 1
fi
echo "ELF: $(file -b "$OUT")"

# 5. Capability: M1.bin must RUN and assemble REAL M1 input through its new
#    flag-driven CLI (the OLD positional M0 took no flags). We assemble the
#    actual amd64 defs + libc-core (known-good upstream M1 syntax) and require
#    nonempty hex2 output.
"./$OUT" -f "$MDEFS" -f "$MLIBC" --little-endian --architecture amd64 -o "$T/cap.hex2"; rc=$?
echo "M1.bin assemble rc=$rc; cap.hex2 = $([ -s "$T/cap.hex2" ] && stat -c%s "$T/cap.hex2" || echo 0) bytes"
if [ "$rc" != "0" ] || [ ! -s "$T/cap.hex2" ]; then
    echo "ERROR: M1.bin did not assemble real M1 input (rc=$rc)" >&2; exit 1
fi
echo "OK: M1.bin runs and assembles defs+libc-core (flag-driven CLI)."
