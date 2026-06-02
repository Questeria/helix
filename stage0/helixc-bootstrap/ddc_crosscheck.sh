#!/usr/bin/env bash
# ddc_crosscheck.sh -- INDEPENDENT diverse-double-compile of the C-compiler rung.
# Build the seed from the FROZEN seed.c two independent ways:
#   route M2 : the existing seed.bin (hex0..M2-Planet -> seed.c -> seed.bin)
#   route GCC: gcc (a totally separate lineage, no M2-Planet ancestry) + frozen seed.c
# Then assert BOTH seeds compile the SAME k1src.hx into a BYTE-IDENTICAL K1.
# K1 identical from two independent compilers = Wheeler DDC: M2-Planet injected
# nothing into the seed (a trojan would have to live in seed.c's visible source,
# or in BOTH gcc and M2-Planet identically). seed.c is NOT edited (headers via -include).
set -u
cd "$(dirname "$0")"   # stage0/helixc-bootstrap
INC="-include stdio.h -include stdlib.h -include unistd.h -include string.h"

echo "=== [1] gcc builds the seed from FROZEN seed.c (no edits) ==="
gcc -std=gnu89 -w $INC -o /tmp/seed_gcc seed.c 2>/tmp/gccerr || { echo "  gcc build FAIL:"; head -8 /tmp/gccerr; exit 1; }
chmod +x /tmp/seed_gcc
echo "  seed_gcc = $(stat -c%s /tmp/seed_gcc) bytes"
/tmp/seed_gcc; echo "  seed_gcc no-arg self-test exit=$? (want 42)"

echo "=== [2] both seeds compile the SAME k1src.hx (1.5 MB) -> K1 ==="
chmod +x seed.bin 2>/dev/null
t0=$SECONDS; ./seed.bin    k1src.hx /tmp/K1_m2.bin;  echo "  M2-seed  -> K1_m2  exit=$? $((SECONDS-t0))s ($(stat -c%s /tmp/K1_m2.bin 2>/dev/null) bytes)"
t0=$SECONDS; /tmp/seed_gcc k1src.hx /tmp/K1_gcc.bin; echo "  gcc-seed -> K1_gcc exit=$? $((SECONDS-t0))s ($(stat -c%s /tmp/K1_gcc.bin 2>/dev/null) bytes)"

echo "=== [3] DDC ANCHOR: K1_gcc == K1_m2 byte-identical? ==="
if [ ! -s /tmp/K1_gcc.bin ] || [ ! -s /tmp/K1_m2.bin ]; then echo "  DDC_FAIL (a K1 is empty -- build error)"; exit 2; fi
sm=$(sha256sum /tmp/K1_m2.bin  | cut -d' ' -f1)
sg=$(sha256sum /tmp/K1_gcc.bin | cut -d' ' -f1)
echo "  K1_m2  sha256=$sm"
echo "  K1_gcc sha256=$sg"
if [ "$sm" = "$sg" ]; then
  echo "  DDC_ANCHOR_OK -- gcc (independent lineage) reproduces the M2-Planet seed's K1 byte-for-byte."
  echo "  => The seed's behavior is independently double-compiled; identical K1 implies identical K2==K3==K4."
else
  echo "  DDC_ANCHOR_DIFF -- the two K1 differ. A REAL finding to investigate (seed.c non-determinism/portability, or a compiler-semantics gap)."
  cmp /tmp/K1_m2.bin /tmp/K1_gcc.bin 2>&1 | head -2
fi
