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
# rm-before (v1.3 audit-remediation 4b): no stale gcc-seed on a failed build. This is a
# C-COMPILED binary leg, so rc==0 IS a valid success assertion (kept).
rm -f /tmp/seed_gcc
gcc -std=gnu89 -w $INC -o /tmp/seed_gcc seed.c 2>/tmp/gccerr || { echo "  gcc build FAIL:"; head -8 /tmp/gccerr; exit 1; }
if [ ! -s /tmp/seed_gcc ]; then echo "  gcc build FAIL (no /tmp/seed_gcc)"; exit 1; fi
chmod +x /tmp/seed_gcc
echo "  seed_gcc = $(stat -c%s /tmp/seed_gcc) bytes"
/tmp/seed_gcc; stc=$?; echo "  seed_gcc no-arg self-test exit=$stc (want 42)"
if [ "$stc" -ne 42 ]; then echo "  DDC_FAIL (gcc-seed self-test exit=$stc != 42 -- gcc-built seed misbehaves)"; exit 2; fi

echo "=== [2] both seeds compile the SAME k1src.hx (1.5 MB) -> K1 ==="
# input sanity: a missing/empty k1src.hx would make BOTH K1 empty -> a vacuous "match".
# ALWAYS regenerate from committed source -- never trust a possibly-stale ignored k1src.hx (v1.3
# final pass): a stale nonempty k1src.hx would weaken standalone DDC evidence; the pinned K1 hash
# below also catches drift, but regenerating removes the dependency entirely.
echo "  [2.0] regenerating k1src.hx/k1input.hx/k1ptxdrv.hx from committed source via assemble_k1.sh"
rm -f k1src.hx k1input.hx k1ptxdrv.hx
bash assemble_k1.sh >/dev/null 2>&1
if [ ! -s k1src.hx ]; then echo "  DDC_FAIL (k1src.hx missing/empty after assemble_k1.sh)"; exit 2; fi
chmod +x seed.bin 2>/dev/null
# rm-before each generation (v1.3 4b). Both K1 outputs are produced by Helix-built seed
# compilers (M2-seed and gcc-seed BOTH run the Helix seed program), which exit NONZERO on
# success (output byte-count); success is the NON-EMPTY assert below, NOT rc==0.
rm -f /tmp/K1_m2.bin /tmp/K1_gcc.bin
t0=$SECONDS; ./seed.bin    k1src.hx /tmp/K1_m2.bin;  echo "  M2-seed  -> K1_m2  exit=$? $((SECONDS-t0))s ($(stat -c%s /tmp/K1_m2.bin 2>/dev/null) bytes)"
t0=$SECONDS; /tmp/seed_gcc k1src.hx /tmp/K1_gcc.bin; echo "  gcc-seed -> K1_gcc exit=$? $((SECONDS-t0))s ($(stat -c%s /tmp/K1_gcc.bin 2>/dev/null) bytes)"
# NON-EMPTY guard immediately after generation, BEFORE the SHA compare (a failed generation
# must not leave a stale file that produces a false byte-identical match).
if [ ! -s /tmp/K1_m2.bin ];  then echo "  DDC_FAIL (M2-seed produced empty K1_m2 -- build error)";  exit 2; fi
if [ ! -s /tmp/K1_gcc.bin ]; then echo "  DDC_FAIL (gcc-seed produced empty K1_gcc -- build error)"; exit 2; fi

echo "=== [3] DDC ANCHOR: K1_gcc == K1_m2 byte-identical? ==="
if [ ! -s /tmp/K1_gcc.bin ] || [ ! -s /tmp/K1_m2.bin ]; then echo "  DDC_FAIL (a K1 is empty -- build error)"; exit 2; fi
sm=$(sha256sum /tmp/K1_m2.bin  | cut -d' ' -f1)
sg=$(sha256sum /tmp/K1_gcc.bin | cut -d' ' -f1)
echo "  K1_m2  sha256=$sm"
echo "  K1_gcc sha256=$sg"
# v1.5 S0 re-mint (2026-06-13): K1 84363adb... -> 029e6822... -- K1 = seed.bin compiling k1src.hx,
# which assemble_k1.sh concatenates from the committed .hx INCL the edited parser.hx (additive ternary
# type t2, tag 12). gcc and M2-Planet seed lineages BOTH reproduce this K1 byte-identically (DDC self-
# consistency HOLDS at 029e6822); only the pinned value advances. v1.4-shipped K1 84363adb kept at the v1.4 tag.
EXPECT_K1=6ee5ec2bdd5ecbea249e46e42243dae695656f3857cfde3674e753609989658c   # pinned known-good K1; v1.5 S1 re-mint 029e6822 -> 6ee5ec2b (k1src.hx now includes the fp16-emission kovc.hx; gcc==M2-seed byte-identical at the new value)
if [ "$sm" = "$sg" ] && [ "$sm" = "$EXPECT_K1" ]; then
  echo "  DDC_ANCHOR_OK -- gcc (independent lineage) reproduces the M2-Planet seed's K1 byte-for-byte AND == pinned known-good."
  echo "  => The seed's behavior is independently double-compiled; identical K1 implies identical K2==K3==K4."
elif [ "$sm" = "$sg" ]; then
  echo "  DDC_FAIL (K1 self-consistent = $sm but != pinned known-good $EXPECT_K1 -- toolchain drifted from the release anchor)"; exit 2
else
  echo "  DDC_ANCHOR_DIFF -- the two K1 differ. A REAL finding to investigate (seed.c non-determinism/portability, or a compiler-semantics gap)."
  cmp /tmp/K1_m2.bin /tmp/K1_gcc.bin 2>&1 | head -2
  # v1.3 audit-remediation A2: a DDC anchor MISMATCH is a real finding -- FAIL CLOSED
  # (previously this branch printed the finding but fell off the end -> exit 0, masking it).
  exit 3
fi
