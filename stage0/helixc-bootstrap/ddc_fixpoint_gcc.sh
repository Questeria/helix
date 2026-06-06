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
# rm-before (v1.3 audit-remediation 4b): no stale gcc-seed on a failed build. C-COMPILED
# binary leg -> rc==0 IS a valid success assertion (kept), plus a non-empty assert.
rm -f /tmp/seed_gcc
gcc -std=gnu89 -w $INC -o /tmp/seed_gcc seed.c || { echo "gcc build FAIL"; exit 1; }
if [ ! -s /tmp/seed_gcc ]; then echo "gcc build FAIL (no /tmp/seed_gcc)"; exit 1; fi
chmod +x /tmp/seed_gcc
echo "  seed_gcc $(stat -c%s /tmp/seed_gcc) bytes"

echo "=== run the proven fixpoint with seed.bin -> /tmp/seed_gcc (full gcc route) ==="
# rm-before + non-empty guard on the generated runner (v1.3 4b): a failed sed must not leave a
# stale /tmp/fixpoint_gcc.sh. The inner fixpoint does its own kovc/seed-leg non-empty guards.
rm -f /tmp/fixpoint_gcc.sh
sed 's|\./seed\.bin |/tmp/seed_gcc |g' ../../scripts/selfhost_fixpoint_rawbinary.sh > /tmp/fixpoint_gcc.sh
if [ ! -s /tmp/fixpoint_gcc.sh ]; then echo "DDC_FAIL (sed produced empty /tmp/fixpoint_gcc.sh)"; exit 1; fi
# PROPAGATE the inner fixpoint's verdict (fail-closed): previously this fell off the end ->
# exit 0 even if the gcc-route fixpoint FAILED, masking a real mismatch.
bash /tmp/fixpoint_gcc.sh; fp_rc=$?
if [ "$fp_rc" -ne 0 ]; then echo "DDC_FIXPOINT_GCC FAIL (inner fixpoint rc=$fp_rc)"; fi
exit $fp_rc
