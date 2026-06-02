#!/usr/bin/env bash
# ddc_smoke.sh -- feasibility probe for the independent DDC cross-check of the
# C-compiler rung: can an external, independent compiler (gcc/clang/tcc) compile
# the FROZEN seed.c (no edits) into a seed that passes its own no-arg self-test
# (exit 42)? seed.c omits #includes (M2-Planet supplies FILE/fopen/calloc/etc.
# implicitly), so we supply the libc decls via -include WITHOUT touching seed.c.
set -u
cd "$(dirname "$0")"

echo "=== independent compilers available ==="
gcc --version 2>/dev/null | head -1 || echo "gcc absent"
if command -v clang >/dev/null 2>&1; then clang --version | head -1; else echo "clang absent"; fi
if command -v tcc   >/dev/null 2>&1; then tcc -v 2>&1 | head -1;   else echo "tcc absent";   fi
echo ""

INC="-include stdio.h -include stdlib.h -include unistd.h -include string.h"

echo "=== gcc -std=gnu89 -w (+prepended libc headers) compiles frozen seed.c? ==="
gcc -std=gnu89 -w $INC -o /tmp/seed_gcc seed.c 2>/tmp/gccerr
rc=$?
if [ "$rc" -eq 0 ]; then
  echo "  gcc BUILD OK ($(stat -c%s /tmp/seed_gcc) bytes)"
  /tmp/seed_gcc; echo "  seed_gcc no-arg self-test exit=$? (want 42)"
else
  echo "  gcc BUILD FAIL rc=$rc:"; head -12 /tmp/gccerr
fi
echo ""

if command -v tcc >/dev/null 2>&1; then
  echo "=== tcc compiles frozen seed.c? (tcc is more lenient, bootstrap-friendly) ==="
  tcc $INC -o /tmp/seed_tcc seed.c 2>/tmp/tccerr
  rc=$?
  if [ "$rc" -eq 0 ]; then
    echo "  tcc BUILD OK ($(stat -c%s /tmp/seed_tcc) bytes)"
    /tmp/seed_tcc; echo "  seed_tcc no-arg self-test exit=$? (want 42)"
  else
    echo "  tcc BUILD FAIL rc=$rc:"; head -12 /tmp/tccerr
  fi
fi
