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
# i64 arithmetic BEYOND the i32 range, using sub-2^31 SOURCE LITERALS (so the lexer's
# i32 literal accumulator does not truncate) but producing/dividing 64-bit runtime
# values > i32. This tests the 64-bit imul/idiv codegen (correct). The separate
# known limitation -- source literals >= 2^31 truncate -- is documented in DoD #2.
gen i64_mul_beyond.hx <<'EOF'
fn main() -> i32 { let a: i64 = 2000000000_i64; let b: i64 = 3_i64; let c: i64 = a * b; let g: i64 = 1000000000_i64; (c / g) as i32 }
EOF
gen i64_div_beyond.hx <<'EOF'
fn main() -> i32 { let a: i64 = 2000000000_i64; let b: i64 = 2_i64; let big: i64 = a * b; let g: i64 = 80000000_i64; (big / g) as i32 }
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
# expr/operator probes -- verify [impl] features (promote to [proven]) + the
# associativity concern (left-assoc => assoc_sub=5/assoc_div=10; right => 9/40).
gen assoc_sub.hx <<'EOF'
fn main() -> i32 { 10 - 3 - 2 }
EOF
gen assoc_div.hx <<'EOF'
fn main() -> i32 { 100 / 5 / 2 }
EOF
gen cmp_ne.hx <<'EOF'
fn main() -> i32 { if 5 != 3 { 1 } else { 0 } }
EOF
gen cmp_ge.hx <<'EOF'
fn main() -> i32 { if 5 >= 5 { 1 } else { 0 } }
EOF
gen cmp_le.hx <<'EOF'
fn main() -> i32 { if 3 <= 5 { 1 } else { 0 } }
EOF
gen bit_andor.hx <<'EOF'
fn main() -> i32 { (12 & 10) | 1 }
EOF
gen bit_xor.hx <<'EOF'
fn main() -> i32 { 255 ^ 15 }
EOF
gen bit_shl.hx <<'EOF'
fn main() -> i32 { 1 << 4 }
EOF
gen arr_idx.hx <<'EOF'
fn main() -> i32 { let a = [10, 20, 30]; a[1] }
EOF
gen while_sum.hx <<'EOF'
fn main() -> i32 { let mut s = 0; let mut i = 0; while i < 5 { s = s + i; i = i + 1; } s }
EOF
gen while_break.hx <<'EOF'
fn main() -> i32 { let mut i = 0; while i < 100 { i = i + 1; if i >= 7 { break; } } i }
EOF
# types/items probes -- verify [impl] features (f64, tuples, impl-methods, or/range patterns)
gen f64_add.hx <<'EOF'
fn main() -> i32 { let a: f64 = 1.5_f64; let b: f64 = 2.5_f64; (a + b) as i32 }
EOF
gen f64_mul.hx <<'EOF'
fn main() -> i32 { let a: f64 = 3.0_f64; let b: f64 = 4.0_f64; (a * b) as i32 }
EOF
gen tuple2.hx <<'EOF'
fn main() -> i32 { let t = (3, 4); t.0 + t.1 }
EOF
gen impl_method.hx <<'EOF'
struct P { x: i32 }
impl P { fn get(self) -> i32 { self.x } }
fn main() -> i32 { let p = P { x: 42 }; p.get() }
EOF
gen match_or.hx <<'EOF'
fn main() -> i32 { let x = 2; match x { 1 | 2 | 3 => 10, _ => 0 } }
EOF
gen match_range.hx <<'EOF'
fn main() -> i32 { let x = 5; match x { 1..10 => 1, _ => 0 } }
EOF
# collections POC: a growable int Vec built on the arena (free fns) -- demonstrates
# general-purpose collections are user-implementable in Helix (DoD #7). Header slot
# = length; elements follow contiguously. 10+20+12 + len(3) = 45.
gen vec_arena.hx <<'EOF'
fn vec_new() -> i32 { __arena_push(0) }
fn vec_push(v: i32, x: i32) -> i32 { let n = __arena_get(v); __arena_set(v, n + 1); __arena_push(x) }
fn vec_len(v: i32) -> i32 { __arena_get(v) }
fn vec_get(v: i32, i: i32) -> i32 { __arena_get(v + 1 + i) }
fn main() -> i32 {
    let v = vec_new();
    vec_push(v, 10);
    vec_push(v, 20);
    vec_push(v, 12);
    vec_get(v, 0) + vec_get(v, 1) + vec_get(v, 2) + vec_len(v)
}
EOF

echo "=== build K2 (general full-language compiler) from the raw seed ==="
bash assemble_k1.sh >/dev/null 2>&1
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
check "$CD/i64_mul_beyond.hx"  6  i64-mul-beyond-i32
check "$CD/i64_div_beyond.hx" 50  i64-div-beyond-i32
check "$CD/i64_cmp.hx"     1  i64-cmp
check "$CD/i64_neg.hx"     5  i64-neg
check "$CD/u64_shr.hx"     1  u64-logical-shift
check "$CD/u8_wrap.hx"    42  u8-wrap-cast
check "$CD/u16_wrap.hx"   42  u16-wrap-cast
check "$CD/i16_ovf.hx"    42  i16-overflow
echo "=== exprs / operators ==="
check "$CD/assoc_sub.hx"   5  left-assoc-sub
check "$CD/assoc_div.hx"  10  left-assoc-div
check "$CD/cmp_ne.hx"      1  compare-ne
check "$CD/cmp_ge.hx"      1  compare-ge
check "$CD/cmp_le.hx"      1  compare-le
check "$CD/bit_andor.hx"   9  bitwise-and-or
check "$CD/bit_xor.hx"   240  bitwise-xor
check "$CD/bit_shl.hx"    16  shift-left
check "$CD/arr_idx.hx"    20  array-literal-index
check "$CD/while_sum.hx"  10  while-loop
check "$CD/while_break.hx" 7  while-break
echo "=== types / items (verify [impl]) ==="
check "$CD/f64_add.hx"     4  f64-add
check "$CD/f64_mul.hx"    12  f64-mul
check "$CD/tuple2.hx"      7  tuple-literal-field
check "$CD/impl_method.hx" 42  impl-method-self
check "$CD/match_or.hx"   10  match-or-pattern
check "$CD/match_range.hx" 1  match-range-pattern
check "$CD/vec_arena.hx"  45  collections-vec-on-arena

echo
echo "RESULT: $pass passed, $fail failed (of $((pass+fail)))"
[ "$fail" = "0" ] && echo "FEATURE_CORPUS_ALL_OK" || echo "FEATURE_CORPUS_HAS_FAILURES"
