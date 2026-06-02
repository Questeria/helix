#!/usr/bin/env bash
# build.sh -- build stage0/blood-elf/blood-elf.bin: the mescc-tools 1.7.0
# blood-elf debug-symbol footer generator.
#
# Built by M2.bin (../M2-Planet), NOT cc_amd64 -- blood-elf uses
# FILE*/fopen/fgetc/fputc/stderr/EOF. Identical recipe to ../M1/build.sh:
# M2 compiles the tool source TOGETHER WITH the 3 M2libc bootstrap C units +
# stringify.c, then M1-output -> M0 -> hex2.
#
# M2 ignores #include (--expand-includes off), so each unit is an explicit -f:
#   * M2libc/amd64/linux/bootstrap.c  -- read/write/open/close/exit syscalls
#   * M2libc/bootstrap.c              -- FILE*/fopen/fgetc/fputc/EOF/NULL/malloc
#   * M2libc/bootstrappable.c         -- require/match/in_set/strtoint/int2str
#   * stringify.c                     -- stringify()/LittleEndian()
#   * blood-elf.c                     -- the footer generator itself
#
# Run under WSL. Sources GPL-3.0 (oriansj/mescc-tools 1.7.0 + M2libc).
set -euo pipefail
cd "$(dirname "$0")"

M2=../M2-Planet/M2.bin
CATM=../catm/catm.bin
M0=../M0/M0.bin
HEX2=../hex2/hex2.bin
MLIB=../M2-Planet/M2libc
MDEFS=$MLIB/amd64/amd64_defs.M1
MLIBC=$MLIB/amd64/libc-core.M1
MELF=$MLIB/amd64/ELF-amd64.hex2
OUT=blood-elf.bin

for t in "$M2" "$CATM" "$M0" "$HEX2"; do [ -x "$t" ] || chmod +x "$t"; done

T=$(mktemp -d)
trap 'rm -rf "$T"' EXIT

# 1. M2 compiles blood-elf.c + bootstrap C units -> M1.
"$M2" --architecture amd64 \
    -f "$MLIB/amd64/linux/bootstrap.c" \
    -f "$MLIB/bootstrap.c" \
    -f "$MLIB/bootstrappable.c" \
    -f stringify.c \
    -f blood-elf.c \
    --bootstrap-mode -o "$T/be.M1"
echo "M2 -> be.M1: $(stat -c%s "$T/be.M1") bytes"

# 2. Assemble with M2libc amd64 defs + libc-core.
"$CATM" "$T/be-0.M1" "$MDEFS" "$MLIBC" "$T/be.M1"
"$M0"   "$T/be-0.M1" "$T/be.hex2"
echo "M0 -> be.hex2: $(stat -c%s "$T/be.hex2") bytes"

# 3. Link.
"$CATM" "$T/be-0.hex2" "$MELF" "$T/be.hex2"
"$HEX2" "$T/be-0.hex2" "$OUT"
chmod +x "$OUT"
echo "Built $OUT: $(stat -c%s "$OUT") bytes"

# 4. ELF sanity.
if ! file "$OUT" | grep -q "ELF 64-bit LSB executable"; then
    echo "ERROR: $OUT is not a valid x86-64 ELF" >&2; file "$OUT"; exit 1
fi
echo "ELF: $(file -b "$OUT")"

# 5. Capability: blood-elf.bin must RUN and recognize its flags
#    (--64 -f --little-endian --entry -o). We feed it a tiny .M1 carrying one
#    labeled symbol and require it emits a nonempty debug-footer .M1.
printf ':_start\n:foo\n' > "$T/in.M1"
"./$OUT" --64 -f "$T/in.M1" --little-endian --entry _start -o "$T/foot.M1"; rc=$?
echo "blood-elf.bin rc=$rc; foot.M1 = $([ -s "$T/foot.M1" ] && stat -c%s "$T/foot.M1" || echo 0) bytes"
if [ "$rc" != "0" ] || [ ! -s "$T/foot.M1" ]; then
    echo "ERROR: blood-elf.bin did not emit a debug footer (rc=$rc)" >&2; exit 1
fi
echo "OK: blood-elf.bin runs and recognizes --64/--little-endian/--entry/-f/-o."
