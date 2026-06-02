#!/usr/bin/env bash
# PROBE: confirm/deny the >6-arg SysV passing bug in kovc.hx.
# Builds K2 from the CURRENT on-disk kovc.hx (whatever state it is in) and
# compiles+runs three >6-arg probes via K2. Reports each exit code vs expected.
#   f8:  a..h = 1..8  -> 36
#   f9:  a..i = 1..9  -> 45
#   f11: a..k = 1..11 -> 66
set -u
BS=/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap
cd "$BS" || { echo "FATAL: no bootstrap dir"; exit 9; }

echo "=== [0] assemble sources from current kovc.hx ==="
bash assemble_k1.sh >/dev/null 2>&1 && echo "  assembled" || { echo "FATAL assemble"; exit 8; }

echo "=== [1] seed -> K1 -> K2 ==="
t0=$SECONDS
timeout 400 ./seed.bin k1src.hx /tmp/K1.bin; echo "  seed->K1 rc=$? ($((SECONDS-t0))s)"
chmod +x /tmp/K1.bin; cp k1input.hx /tmp/k1_in.hx
timeout 60 /tmp/K1.bin; [ -s /tmp/k1_out.bin ] || { echo "FATAL: K2 build failed"; exit 7; }
cp /tmp/k1_out.bin /tmp/K2.bin; chmod +x /tmp/K2.bin
echo "  K2 built ($(stat -c%s /tmp/K2.bin) bytes)"

run1() { # name source expected
  local name="$1" src="$2" exp="$3"
  printf '%s' "$src" > /tmp/k2_in.hx
  rm -f /tmp/k2_out.bin
  timeout 30 /tmp/K2.bin >/dev/null 2>&1
  if [ ! -s /tmp/k2_out.bin ]; then echo "  $name: COMPILE-FAIL (no output)"; return; fi
  chmod +x /tmp/k2_out.bin
  timeout 10 /tmp/k2_out.bin; local rc=$?
  if [ "$rc" = "$exp" ]; then echo "  $name: rc=$rc == expected $exp  (CORRECT)";
  else echo "  $name: rc=$rc != expected $exp  (WRONG)"; fi
}

echo "=== [2] >6-arg probes via K2 ==="
run1 f8 'fn f8(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32) -> i32 { a+b+c+d+e+f+g+h }
fn main() -> i32 { f8(1,2,3,4,5,6,7,8) }' 36

run1 f9 'fn f9(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32, i: i32) -> i32 { a+b+c+d+e+f+g+h+i }
fn main() -> i32 { f9(1,2,3,4,5,6,7,8,9) }' 45

run1 f11 'fn f11(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32, h: i32, i: i32, j: i32, k: i32) -> i32 { a+b+c+d+e+f+g+h+i+j+k }
fn main() -> i32 { f11(1,2,3,4,5,6,7,8,9,10,11) }' 66

echo "=== PROBE DONE ==="
