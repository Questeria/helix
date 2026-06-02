#!/usr/bin/env bash
# DC3 -- run the FULL Python-free self-host fixpoint via the GCC-built seed (the
# independent route). Reuses the PROVEN scripts/selfhost_fixpoint_rawbinary.sh
# verbatim, swapping only the seed binary (./seed.bin -> /tmp/seed_gcc). Because
# K1_gcc == K1_m2 byte-identical (DC2), this MUST reach the same K2==K3==K4; the
# explicit run is the belt-and-suspenders evidence for the audit. seed.c FROZEN.
set -u
cd "$(dirname "$0")"   # stage0/helixc-bootstrap
INC="-include stdio.h -include stdlib.h -include unistd.h -include string.h"

echo "=== build the gcc-seed from FROZEN seed.c (no edits) ==="
gcc -std=gnu89 -w $INC -o /tmp/seed_gcc seed.c || { echo "gcc build FAIL"; exit 1; }
chmod +x /tmp/seed_gcc
echo "  seed_gcc $(stat -c%s /tmp/seed_gcc) bytes"

echo "=== run the proven fixpoint with seed.bin -> /tmp/seed_gcc (full gcc route) ==="
sed 's|\./seed\.bin |/tmp/seed_gcc |g' ../../scripts/selfhost_fixpoint_rawbinary.sh > /tmp/fixpoint_gcc.sh
bash /tmp/fixpoint_gcc.sh
