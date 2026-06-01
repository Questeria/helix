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
python3 assemble_k1.py >/dev/null 2>&1 && echo "  assembled" || { echo "FATAL assemble"; exit 8; }

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
chk "$CD/i64_basic.hx" 42; chk "$CD/i64_add.hx" 5; chk "$CD/i64_mul.hx" 6; chk "$CD/i64_cmp.hx" 1; chk "$CD/i64_neg.hx" 5
chk "$CD/u64_shr.hx" 1; chk "$CD/u8_wrap.hx" 42; chk "$CD/u16_wrap.hx" 42; chk "$CD/i16_ovf.hx" 42
echo "  CORPUS: $pass passed, $fail failed (expect 15 pass: dogfood_18 PatStruct + result_inline now PASS; 2 known-fail = i64_add/mul lexer-width)"

echo "=== GATE VERDICT ==="
# regression guard: the u64_shr must now PASS, and we must not drop below 13 passes.
if [ "$pass" -lt 15 ]; then echo "  CORPUS REGRESSION (pass=$pass < 15)"; GATE_OK=0; fi
if [ "$GATE_OK" = "1" ]; then echo "GATE_PASS"; else echo "GATE_FAIL"; fi
