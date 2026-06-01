#!/usr/bin/env bash
# Helix v1.0 DoD criterion #2 -- feature compile+run corpus.
#
# Builds K2 (the self-hosted FULL-LANGUAGE kovc compiler) from the raw-binary seed,
# then compiles AND runs each feature program through K2 and asserts its exit code.
# K2 is a general compiler via fixed paths: stage a .hx at /tmp/k2_in.hx, run K2
# (no args), read the emitted ELF at /tmp/k2_out.bin. Run as a FILE (never inline).
set -u
BS=/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap
EX=/mnt/c/Projects/Kovostov-Native/helixc/examples
cd "$BS" || { echo "FATAL: no bootstrap dir"; exit 9; }

echo "=== build K2 (general full-language compiler) from the raw seed ==="
python3 assemble_k1.py >/dev/null 2>&1
t0=$SECONDS
timeout 400 ./seed.bin k1src.hx /tmp/K1.bin; echo "  seed->K1 rc=$? ($((SECONDS-t0))s)"
chmod +x /tmp/K1.bin
cp k1input.hx /tmp/k1_in.hx
timeout 60 /tmp/K1.bin                       # K1 reads /tmp/k1_in.hx -> /tmp/k1_out.bin = K2
[ -s /tmp/k1_out.bin ] || { echo "FATAL: K2 not built"; exit 8; }
cp /tmp/k1_out.bin /tmp/K2.bin; chmod +x /tmp/K2.bin
echo "  K2 built: $(stat -c%s /tmp/K2.bin) bytes"
echo

echo "=== feature corpus: compile + run each via K2 ==="
pass=0; fail=0
check() { # <file> <expected-exit> <feature-label>
  local f="$1" exp="$2" feat="$3"
  if [ ! -f "$EX/$f" ]; then echo "  MISSING       $f  [$feat]"; fail=$((fail+1)); return; fi
  cp "$EX/$f" /tmp/k2_in.hx
  rm -f /tmp/k2_out.bin
  timeout 30 /tmp/K2.bin >/dev/null 2>&1
  if [ ! -s /tmp/k2_out.bin ]; then echo "  COMPILE-FAIL  $f  [$feat]"; fail=$((fail+1)); return; fi
  chmod +x /tmp/k2_out.bin
  timeout 10 /tmp/k2_out.bin; local rc=$?
  if [ "$rc" = "$exp" ]; then echo "  PASS          $f  exit=$rc  [$feat]"; pass=$((pass+1));
  else echo "  RUN-FAIL      $f  exit=$rc expect=$exp  [$feat]"; fail=$((fail+1)); fi
}

check exit42.hx                       42  baseline-literal
check matmul_2x2.hx                   69  scalar-arith
check hbs_sample_enum_struct.hx      129  struct+enum+match
check hbs_sample_option.hx            42  payload-enum+match
check hbs_sample_recursion.hx        120  enum+recursion
check dogfood_18_pat_struct_showcase.hx 42 struct-destructure
check dogfood_16_result_basic.hx      42  Result-enum
check gradient_descent.hx             42  grad+float

echo
echo "RESULT: $pass passed, $fail failed (of $((pass+fail)))"
[ "$fail" = "0" ] && echo "FEATURE_CORPUS_ALL_OK" || echo "FEATURE_CORPUS_HAS_FAILURES"
