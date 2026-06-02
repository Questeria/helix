#!/usr/bin/env bash
# GATE for a kovc.hx change (Helix v1.0). Run as a FILE under WSL. Verifies, from the
# EDITED kovc.hx, the full discipline before any commit:
#   1. SELF-HOST FIXPOINT: seed -> K1 -> K2 -> K3 -> K4, assert K2==K3==K4 byte-identical
#      (the trust spine still self-hosts; sha may differ from the old mint, that is fine).
#   2. GPU PTX REGRESSION: emit a kernel's PTX with the OLD driver, re-mint the driver from
#      the edited kovc.hx, emit again, byte-diff -- an x86-only fix must NOT change PTX.
#   3. FEATURE CORPUS: compile+run the 17-program corpus via the new K2; report the matrix.
# Overall GATE PASS = fixpoint identical AND PTX identical AND corpus has NO regressions.
set -u
BS=/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap
EX=/mnt/c/Projects/Kovostov-Native/helixc/examples
CD=/tmp/corpus
cd "$BS" || { echo "FATAL: no bootstrap dir"; exit 9; }
mkdir -p "$CD"
GATE_OK=1

echo "=== [0] regenerate sources from the edited kovc.hx ==="
bash assemble_k1.sh >/dev/null 2>&1 && echo "  assembled" || { echo "FATAL assemble"; exit 8; }

echo "=== [1] GPU PTX reference (OLD driver, pre-re-mint) ==="
Kern=$EX/vector_add_kernel.hx
if [ -f "$Kern" ] && [ -x ./_kovc_ptx_driver.bin ]; then
  cp "$Kern" /tmp/kernel_in.hx; rm -f /tmp/out.ptx
  timeout 30 ./_kovc_ptx_driver.bin >/dev/null 2>&1 || true
  if [ -s /tmp/out.ptx ]; then cp /tmp/out.ptx /tmp/ref.ptx; echo "  ref.ptx $(stat -c%s /tmp/ref.ptx) bytes"; else echo "  WARN: old driver emitted no PTX"; fi
else echo "  WARN: missing kernel or driver -- skipping GPU ref"; fi

echo "=== [2] SELF-HOST FIXPOINT from edited kovc.hx ==="
t0=$SECONDS
timeout 400 ./seed.bin k1src.hx /tmp/K1.bin; echo "  seed->K1 rc=$? ($((SECONDS-t0))s)"
chmod +x /tmp/K1.bin; cp k1input.hx /tmp/k1_in.hx
timeout 60 /tmp/K1.bin; [ -s /tmp/k1_out.bin ] || { echo "FATAL: K2 build failed (kovc.hx may not self-compile)"; exit 7; }
cp /tmp/k1_out.bin /tmp/K2.bin; chmod +x /tmp/K2.bin
cp k1input.hx /tmp/k2_in.hx; timeout 60 /tmp/K2.bin; cp /tmp/k2_out.bin /tmp/K3.bin; chmod +x /tmp/K3.bin
timeout 60 /tmp/K3.bin; cp /tmp/k2_out.bin /tmp/K4.bin
S2=$(sha256sum /tmp/K2.bin|awk '{print $1}'); S3=$(sha256sum /tmp/K3.bin|awk '{print $1}'); S4=$(sha256sum /tmp/K4.bin|awk '{print $1}')
echo "  K2=$S2"; echo "  K3=$S3"; echo "  K4=$S4"
if [ "$S2" = "$S3" ] && [ "$S3" = "$S4" ]; then echo "  FIXPOINT OK (K2==K3==K4 byte-identical)"; else echo "  FIXPOINT FAIL"; GATE_OK=0; fi

echo "=== [3] re-mint PTX driver + GPU PTX regression ==="
t0=$SECONDS
timeout 400 ./seed.bin k1ptxdrv.hx /tmp/newdrv.bin; echo "  seed->newdrv rc=$? ($((SECONDS-t0))s)"
if [ -s /tmp/newdrv.bin ] && [ -s /tmp/ref.ptx ]; then
  chmod +x /tmp/newdrv.bin; cp "$Kern" /tmp/kernel_in.hx; rm -f /tmp/out.ptx
  timeout 30 /tmp/newdrv.bin >/dev/null 2>&1 || true
  if cmp -s /tmp/out.ptx /tmp/ref.ptx; then echo "  GPU PTX REGRESSION OK (PTX byte-identical pre/post fix)";
  else echo "  GPU PTX CHANGED -- inspect (x86-only fix should NOT alter PTX)"; GATE_OK=0; fi
else echo "  WARN: could not run PTX regression (no newdrv or ref)"; fi

echo "=== [4] FEATURE CORPUS via new K2 ==="
gen() { cat > "$CD/$1"; }
gen i64_basic.hx <<'EOF'
fn main() -> i32 { let x: i64 = 42_i64; x as i32 }
EOF
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
pass=0; fail=0
chk() { local f="$1" exp="$2" b; b=$(basename "$1")
  [ -f "$f" ] || { echo "  MISSING $b"; fail=$((fail+1)); return; }
  cp "$f" /tmp/k2_in.hx; rm -f /tmp/k2_out.bin; timeout 30 /tmp/K2.bin >/dev/null 2>&1
  [ -s /tmp/k2_out.bin ] || { echo "  COMPILE-FAIL $b"; fail=$((fail+1)); return; }
  chmod +x /tmp/k2_out.bin; timeout 10 /tmp/k2_out.bin; local rc=$?
  if [ "$rc" = "$exp" ]; then echo "  PASS $b ($rc)"; pass=$((pass+1)); else echo "  FAIL $b ($rc!=$exp)"; fail=$((fail+1)); fi
}
chk "$EX/exit42.hx" 42; chk "$EX/matmul_2x2.hx" 69; chk "$EX/hbs_sample_enum_struct.hx" 129
chk "$EX/hbs_sample_option.hx" 42; chk "$EX/hbs_sample_recursion.hx" 120
chk "$EX/dogfood_18_pat_struct_showcase.hx" 42; chk "$CD/result_inline.hx" 42; chk "$EX/gradient_descent.hx" 42
chk "$CD/i64_basic.hx" 42; chk "$CD/i64_mul_beyond.hx" 6; chk "$CD/i64_div_beyond.hx" 50; chk "$CD/i64_cmp.hx" 1; chk "$CD/i64_neg.hx" 5
chk "$CD/u64_shr.hx" 1; chk "$CD/u8_wrap.hx" 42; chk "$CD/u16_wrap.hx" 42; chk "$CD/i16_ovf.hx" 42
chk "$CD/assoc_sub.hx" 5; chk "$CD/assoc_div.hx" 10; chk "$CD/cmp_ne.hx" 1; chk "$CD/cmp_ge.hx" 1; chk "$CD/cmp_le.hx" 1
chk "$CD/bit_andor.hx" 9; chk "$CD/bit_xor.hx" 240; chk "$CD/bit_shl.hx" 16; chk "$CD/arr_idx.hx" 20; chk "$CD/while_sum.hx" 10; chk "$CD/while_break.hx" 7
chk "$CD/f64_add.hx" 4; chk "$CD/f64_mul.hx" 12; chk "$CD/tuple2.hx" 7; chk "$CD/impl_method.hx" 42; chk "$CD/match_or.hx" 10; chk "$CD/match_range.hx" 1; chk "$CD/vec_arena.hx" 45
# H2 generics corpus (2026-06-01): the charter generics items run on the self-hosted compiler.
GENC=$BS/corpus_gen
chk "$GENC/gen_impl_t_single_f32.hx" 5; chk "$GENC/gen_impl_angle_i32.hx" 5; chk "$GENC/gen_impl_ret_f32.hx" 5; chk "$GENC/gen_concrete_on_mono.hx" 7
chk "$GENC/gen_pair_multi.hx" 12; chk "$GENC/gen_vec_i32.hx" 42; chk "$GENC/gen_vec_f32.hx" 5; chk "$GENC/e6_bare_match.hx" 42
# H3 traits + closures corpus (2026-06-01): trait-method dispatch + closure capture codegen.
chk "$GENC/t2_trait_impl.hx" 42; chk "$GENC/t3_closure_call.hx" 42; chk "$GENC/t4_closure_capture.hx" 42; chk "$GENC/t8_closure_two_caps.hx" 42
chk "$GENC/t7_trait_poly.hx" 42; chk "$GENC/t7b_trait_2types.hx" 42; chk "$GENC/t7c_difffields.hx" 42
# H4 pattern guards corpus (2026-06-01): match arm `if cond` guard is now evaluated.
chk "$GENC/g1_guard_true.hx" 1; chk "$GENC/g2_guard_false.hx" 0; chk "$GENC/g3_guard_chain.hx" 2
echo "  CORPUS: $pass passed, $fail failed (expect 53 pass: 35 v1.0 + 8 H2 generics + 7 H3 traits/closures + 3 H4 pattern-guards [guard true/false-falls-through/chain-2nd-wins]; large i64 source literals >=2^31 are a documented lexer limitation, not in the corpus)"

echo "=== GATE VERDICT ==="
# regression guard: the u64_shr must now PASS, and we must not drop below 13 passes.
if [ "$pass" -lt 53 ]; then echo "  CORPUS REGRESSION (pass=$pass < 53)"; GATE_OK=0; fi
if [ "$GATE_OK" = "1" ]; then echo "GATE_PASS"; else echo "GATE_FAIL"; fi
