#!/usr/bin/env bash
# FAST INNER LOOP (T3 §1.6 H-4 trait DEFAULT methods): build K2 from the
# CURRENT tree (parser.hx CHANGED -- so the K2 sha MOVES off the H-3/H-1/H-2
# mint bdff0049...) and run the trait-default proofs through it. seed -> K1 ->
# K2 (one fixpoint step), then K2 compiles+runs each program; assert the exit.
# Also re-runs the existing trait/closure corpus through K2 to prove no
# regression (the self-host source uses traits, so this edit must not break
# trait codegen). Reports the K2 sha.
set -u
T0=$(date +%s)
ROOT=/mnt/c/Projects/Kovostov-Native
HB="$ROOT/stage0/helixc-bootstrap"
GENC="$HB/corpus_gen"
EX="$ROOT/helixc/examples"
cd "$HB" || { echo "FATAL no bootstrap dir"; exit 9; }

echo "=== [A] assemble sources from current tree (picks up edited parser.hx) ==="
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
timeout 90 /tmp/K1.bin
[ -s /tmp/k1_out.bin ] || { echo "FATAL K2 build failed (parser.hx may not self-compile)"; exit 5; }
cp /tmp/k1_out.bin /tmp/K2.bin; chmod +x /tmp/K2.bin
echo "  K2 sha = $(sha256sum /tmp/K2.bin | awk '{print $1}')"

echo "=== [D] run the H-4 trait-default proofs + trait-corpus regression through K2 ==="
run_one() {
  local f="$1" exp="$2" b; b=$(basename "$1")
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
# H-4 NEW: t1 = default-USED (impl Greet for P {} -> p.hello() uses { 42 } default)
run_one "$GENC/t1_trait_default.hx" 42 || F=$((F+1)); N=$((N+1))
# H-4 NEW: t5 = default + OVERRIDE mix (A uses default 10 ; B overrides -> 32 ; sum 42)
run_one "$GENC/t5_trait_default_mix.hx" 42 || F=$((F+1)); N=$((N+1))
# REGRESSION: existing trait corpus (signature-only methods + impl override) must still pass
run_one "$GENC/t2_trait_impl.hx" 42 || F=$((F+1)); N=$((N+1))
run_one "$GENC/t7_trait_poly.hx" 42 || F=$((F+1)); N=$((N+1))
run_one "$GENC/t7b_trait_2types.hx" 42 || F=$((F+1)); N=$((N+1))
run_one "$GENC/t7c_difffields.hx" 42 || F=$((F+1)); N=$((N+1))
# REGRESSION: a couple of unrelated programs to confirm general codegen intact
run_one "$EX/exit42.hx" 42 || F=$((F+1)); N=$((N+1))
run_one "$GENC/H2_string.hx" 42 || F=$((F+1)); N=$((N+1))

echo "=== INNER-LOOP RESULT: $((N-F))/$N pass ; wall $(( $(date +%s) - T0 ))s ==="
[ "$F" = 0 ] && echo "INNER_GREEN" || echo "INNER_RED"
exit $F
