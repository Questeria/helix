#!/usr/bin/env bash
# FAST INNER LOOP (T3 §1.6 H-2 rich String): build K2 from the CURRENT tree
# (NO kovc.hx change -- pure stdlib/string.hx + corpus addition) and run the
# String round-trip proof through it. seed -> K1 -> K2 (one fixpoint step),
# then K2 compiles+runs the program; assert the exit. Reports the K2 sha
# (should match the H-3 mint bdff0049... since no kovc.hx changed since H-3).
set -u
T0=$(date +%s)
ROOT=/mnt/c/Projects/Kovostov-Native
HB="$ROOT/stage0/helixc-bootstrap"
GENC="$HB/corpus_gen"
cd "$HB" || { echo "FATAL no bootstrap dir"; exit 9; }

echo "=== [A] assemble sources from current tree ==="
./seed.bin assemble_k1.hx /tmp/asm_k1.bin && chmod +x /tmp/asm_k1.bin && /tmp/asm_k1.bin \
  || { echo "FATAL assemble_k1"; exit 7; }
echo "  assembled k1src.hx ($(stat -c%s k1src.hx) bytes)"

echo "=== [B] seed -> K1 (the slow step) ==="
TB=$(date +%s)
( ulimit -s unlimited && timeout 600 ./seed.bin k1src.hx /tmp/K1.bin ) \
  || { echo "FATAL seed->K1"; exit 6; }
chmod +x /tmp/K1.bin
echo "  K1 built in $(( $(date +%s) - TB ))s ($(stat -c%s /tmp/K1.bin) bytes)"

echo "=== [C] K1 -> K2 ==="
cp k1input.hx /tmp/k1_in.hx
rm -f /tmp/k1_out.bin
timeout 60 /tmp/K1.bin
[ -s /tmp/k1_out.bin ] || { echo "FATAL K2 build failed"; exit 5; }
cp /tmp/k1_out.bin /tmp/K2.bin; chmod +x /tmp/K2.bin
echo "  K2 sha = $(sha256sum /tmp/K2.bin | awk '{print $1}')"

echo "=== [D] run the H-2 String proof through K2 ==="
run_one() {
  local f="$GENC/$1" exp="$2" b="$1"
  [ -f "$f" ] || { echo "  MISSING $b"; return 1; }
  cp "$f" /tmp/k2_in.hx; rm -f /tmp/k2_out.bin
  timeout 30 /tmp/K2.bin >/dev/null 2>&1
  [ -s /tmp/k2_out.bin ] || { echo "  COMPILE-FAIL $b"; return 1; }
  chmod +x /tmp/k2_out.bin
  timeout 10 /tmp/k2_out.bin; local rc=$?
  if [ "$rc" = "$exp" ]; then echo "  PASS $b (exit $rc)"; return 0
  else echo "  FAIL $b (exit $rc != $exp)"; return 1; fi
}
N=0; F=0
run_one H2_string.hx 42 || F=$((F+1)); N=$((N+1))
# sanity: existing string-literal arm + H-1 collections still good through K2
run_one arm_str_lit.hx 42 || F=$((F+1)); N=$((N+1))
run_one H1_vec.hx 42      || F=$((F+1)); N=$((N+1))
run_one H1_hashmap.hx 42  || F=$((F+1)); N=$((N+1))

echo "=== INNER-LOOP RESULT: $((N-F))/$N pass ; wall $(( $(date +%s) - T0 ))s ==="
[ "$F" = 0 ] && echo "INNER_GREEN" || echo "INNER_RED"
exit $F
