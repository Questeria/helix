#!/usr/bin/env bash
# Helix v1.0 DoD criterion #2 -- feature compile+run corpus.
#
# Builds K2 (the self-hosted FULL-LANGUAGE kovc compiler) from the raw-binary seed,
# then compiles AND runs each feature program through K2 and asserts its exit code.
# K2 = general compiler via fixed paths: stage .hx at /tmp/k2_in.hx, run K2, read the
# emitted ELF at /tmp/k2_out.bin. Run as a FILE (never inline). Authored int-width
# probes mirror the pre-K4 snapshot test_codegen.py exactly (syntax + expected exits).
set -u
BS=/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap
EX=/mnt/c/Projects/Kovostov-Native/helixc/examples
CD=/tmp/corpus
cd "$BS" || { echo "FATAL: no bootstrap dir"; exit 9; }
mkdir -p "$CD"

gen() { cat > "$CD/$1"; }   # gen <name>  (program body on stdin)
gen i64_basic.hx <<'EOF'
fn main() -> i32 { let x: i64 = 42_i64; x as i32 }
EOF
gen i64_add.hx <<'EOF'
fn main() -> i32 { let big: i64 = 5_000_000_000_i64; let one_b: i64 = 1_000_000_000_i64; let q: i64 = big / one_b; q as i32 }
EOF
gen i64_mul.hx <<'EOF'
fn main() -> i32 { let a: i64 = 3_000_000_000_i64; let b: i64 = 2_i64; let c: i64 = a * b; let one_b: i64 = 1_000_000_000_i64; (c / one_b) as i32 }
EOF
gen i64_cmp.hx <<'EOF'
fn main() -> i32 { let a: i64 = 5_000_000_000_i64; let b: i64 = 4_000_000_000_i64; if a > b { 1 } else { 0 } }
EOF
gen i64_neg.hx <<'EOF'
fn main() -> i32 { let a: i64 = 100_i64; let b: i64 = -a + 105_i64; b as i32 }
EOF
gen u64_shr.hx <<'EOF'
fn shr_u64(x: u64) -> u64 { x >> 63_u64 }
fn main() -> i32 { let x: u64 = 1_u64 << 63_u64; shr_u64(x) as i32 }
EOF
gen u8_wrap.hx <<'EOF'
fn main() -> i32 { let x: u8 = 0_u8 - 1_u8; let y: i32 = x as i32; if y == 255 { 42 } else { 7 } }
EOF
gen u16_wrap.hx <<'EOF'
fn main() -> i32 { let x: u16 = 0_u16 - 1_u16; let y: i32 = x as i32; if y == 65535 { 42 } else { 7 } }
EOF
gen i16_ovf.hx <<'EOF'
fn main() -> i32 { let x: i16 = 32767_i16 + 1_i16; let y: i32 = x as i32; if y < 0 { 42 } else { 7 } }
EOF
gen result_inline.hx <<'EOF'
enum Result { Ok(i32), Err(i32) }
fn main() -> i32 { let r = Result::Ok(42); match r { Result::Ok(x) => x, Result::Err(e) => e } }
EOF

echo "=== build K2 (general full-language compiler) from the raw seed ==="
python3 assemble_k1.py >/dev/null 2>&1
t0=$SECONDS
timeout 400 ./seed.bin k1src.hx /tmp/K1.bin; echo "  seed->K1 rc=$? ($((SECONDS-t0))s)"
chmod +x /tmp/K1.bin
cp k1input.hx /tmp/k1_in.hx
timeout 60 /tmp/K1.bin
[ -s /tmp/k1_out.bin ] || { echo "FATAL: K2 not built"; exit 8; }
cp /tmp/k1_out.bin /tmp/K2.bin; chmod +x /tmp/K2.bin
echo "  K2 built: $(stat -c%s /tmp/K2.bin) bytes"
echo

pass=0; fail=0
check() { # <abs-file> <expected-exit> <feature-label>
  local f="$1" exp="$2" feat="$3" b; b=$(basename "$1")
  if [ ! -f "$f" ]; then echo "  MISSING       $b  [$feat]"; fail=$((fail+1)); return; fi
  cp "$f" /tmp/k2_in.hx
  rm -f /tmp/k2_out.bin
  timeout 30 /tmp/K2.bin >/dev/null 2>&1
  if [ ! -s /tmp/k2_out.bin ]; then echo "  COMPILE-FAIL  $b  [$feat]"; fail=$((fail+1)); return; fi
  chmod +x /tmp/k2_out.bin
  timeout 10 /tmp/k2_out.bin; local rc=$?
  if [ "$rc" = "$exp" ]; then echo "  PASS          $b  exit=$rc  [$feat]"; pass=$((pass+1));
  else echo "  RUN-FAIL      $b  exit=$rc expect=$exp  [$feat]"; fail=$((fail+1)); fi
}

echo "=== examples ==="
check "$EX/exit42.hx"                        42  baseline-literal
check "$EX/matmul_2x2.hx"                    69  scalar-arith
check "$EX/hbs_sample_enum_struct.hx"       129  struct+enum+match
check "$EX/hbs_sample_option.hx"             42  payload-enum+match
check "$EX/hbs_sample_recursion.hx"         120  enum+recursion
check "$EX/dogfood_18_pat_struct_showcase.hx" 42 struct-destructure
check "$CD/result_inline.hx"                 42  result-enum-userdefined
check "$EX/gradient_descent.hx"              42  grad+float
echo "=== authored int-width ==="
check "$CD/i64_basic.hx"  42  i64-cast
check "$CD/i64_add.hx"     5  i64-div
check "$CD/i64_mul.hx"     6  i64-mul
check "$CD/i64_cmp.hx"     1  i64-cmp
check "$CD/i64_neg.hx"     5  i64-neg
check "$CD/u64_shr.hx"     1  u64-logical-shift
check "$CD/u8_wrap.hx"    42  u8-wrap-cast
check "$CD/u16_wrap.hx"   42  u16-wrap-cast
check "$CD/i16_ovf.hx"    42  i16-overflow

echo
echo "RESULT: $pass passed, $fail failed (of $((pass+fail)))"
[ "$fail" = "0" ] && echo "FEATURE_CORPUS_ALL_OK" || echo "FEATURE_CORPUS_HAS_FAILURES"
