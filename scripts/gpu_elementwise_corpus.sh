#!/usr/bin/env bash
# GPU ELEMENTWISE GELU + ADAM corpus (T2/M4, 2026-06-02). Run as a FILE under WSL:
#   wsl.exe bash -c "bash /mnt/c/Projects/Kovostov-Native/scripts/gpu_elementwise_corpus.sh"
#
# Correctness + HONEST throughput harness for the two kovc elementwise transformer ops:
#   gpu_gelu = tanh-approximation GELU (Hendrycks & Gimpel):
#              y = 0.5*x*(1 + tanh(0.7978846*(x + 0.044715*x^3)))
#              tanh inlined via __gpu_exp (ex2.approx.f32). One thread per element.
#   gpu_adam = one in-place Adam optimiser step (b1=0.9, b2=0.999, lr=1e-3, eps=1e-8 baked;
#              step-dependent bias-correction bc1=1/(1-b1^t), bc2=1/(1-b2^t) passed as
#              1-elem f32 arrays). nm=b1*m+(1-b1)*g ; nv=b2*v+(1-b2)*g^2 ;
#              w -= lr*(nm*bc1)/sqrt((nv*bc2)+eps)  [1/sqrt via __gpu_rsqrt = rsqrt.approx.f32].
#              One thread per element.
#
# Both ops need NO kovc.hx change -- they compile through the ALREADY-GATED emitter
# features (f32 literals + __gpu_exp + __gpu_rsqrt), so the self-host fixpoint + the
# committed vector_add/tiled reference PTX stay byte-identical (the universal gate is
# undisturbed). This corpus is the on-HW correctness proof + the honest perf record.
#
# Each op is:
#  (a) CORRECT cell-by-cell vs an INDEPENDENT CPU reference (GELU: expf-based tanh, tol
#      1e-3 covering the ex2.approx; ADAM: nm/nv exact-arith tol 1e-5 + new_w tol 1e-4
#      covering rsqrt.approx) at small AND large N -- maxrel reported HONESTLY.
#  (b) THROUGHPUT reported as GB/s (kernel-only cuEvent median). These ops are
#      MEMORY-BOUND and have NO naive/tiled pair to beat (they ARE the elementwise form),
#      so per the charter we GATE ON CORRECTNESS and report throughput honestly -- we do
#      NOT manufacture a fake speedup.
#  (c) TWO negative controls that MUST trip:
#      A. comparator teeth: perturb one output cell pre-compare -> FAIL.
#      B. transcendental-strip (load-bearing): delete the ex2.approx.f32 lines (GELU) or
#         the rsqrt.approx.f32 lines (ADAM) from the EMITTED PTX, ptxas-accept, re-run ->
#         MUST mis-compute (FAIL), proving the exp/rsqrt are load-bearing (not dead code).
#
# Pipeline mirrors gpu_reduction_corpus.sh:
#   [A] regenerate k1ptxdrv.hx from CURRENT kovc.hx (assemble_k1) -- picks up the emitter
#   [B] seed -> K1 PTX driver (the slow ~3min step; ONE build) [skippable via REMINT=0]
#   [1] emit ONE combined.ptx with both kernels
#   [2] PROVENANCE (grep the OUTPUT, never source): ex2.approx.f32 + rsqrt.approx.f32 +
#       the GELU/Adam float-literal hex constants + both .entry + sm_86
#   [3] ptxas-accept the combined PTX for sm_86
#   [4] build the host launcher (cuda_launch.c)
#   [5] CORRECTNESS + THROUGHPUT corpus: gelu + adam at small AND large N (0-bad gate)
#   [6] NEG-CONTROL A (comparator teeth): mutate -> MUST FAIL (both ops)
#   [7] NEG-CONTROL B (transcendental-strip, load-bearing): MUST mis-compute (both ops)
#
# Reference box: RTX 3070 Laptop GPU (sm_86). Tees the emitted PTX + logs to .m1probe/.
# All GPU runs are wrapped in `timeout 90` so a hang fails fast (never wedges the loop).
set -u
T0=$(date +%s)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EX="$ROOT/helixc/examples"
OUT="$ROOT/.m1probe"
mkdir -p "$OUT"
HB=stage0/helixc-bootstrap
DRV="$HB/_kovc_ptx_driver.bin"
PTXAS="${PTXAS:-/usr/local/cuda/bin/ptxas}"
# N sizes. SMALL = correctness sanity; LARGE = correctness + a meaningful throughput
# number (elementwise wants many elements to saturate bandwidth). Pipe '|' separates.
GE_N="${GE_N:-256 | 1048576}"
AD_N="${AD_N:-256 | 1048576}"
RC=0
REMINT="${REMINT:-1}"

if [[ "$REMINT" = "1" ]]; then
  echo "=== [A] regenerate k1ptxdrv.hx from current kovc.hx (assemble_k1) ==="
  ( cd "$HB" && ./seed.bin assemble_k1.hx /tmp/asm_k1.bin && chmod +x /tmp/asm_k1.bin && /tmp/asm_k1.bin ) \
    || { echo "FATAL assemble_k1"; exit 7; }
  echo "  k1ptxdrv.hx regenerated ($(stat -c%s $HB/k1ptxdrv.hx) bytes)"
  echo "=== [B] seed -> K1 PTX driver build (the slow step) ==="
  TB=$(date +%s)
  ( cd "$HB" && ulimit -s unlimited && timeout 600 ./seed.bin k1ptxdrv.hx _kovc_ptx_driver.bin ) \
    || { echo "FATAL: seed could not compile k1ptxdrv.hx"; exit 6; }
  chmod +x "$DRV"
  echo "  K1 driver built in $(( $(date +%s) - TB ))s ($(stat -c%s $DRV) bytes)"
fi
[[ -x "$DRV" ]] || { echo "FATAL: no PTX driver"; exit 7; }

echo "=== [1] emit ONE combined.ptx (gpu_gelu + gpu_adam) ==="
: > /tmp/combined_e.hx
for k in gpu_gelu_kernel gpu_adam_kernel; do
  tr -d '\r' < "$EX/${k}.hx" >> /tmp/combined_e.hx; echo "" >> /tmp/combined_e.hx
done
cp /tmp/combined_e.hx /tmp/kernel_in.hx
rm -f /tmp/out.ptx
"$DRV" >/dev/null 2>&1 || true
[[ -s /tmp/out.ptx ]] || { echo "FATAL: driver emitted no PTX"; exit 6; }
cp /tmp/out.ptx "$OUT/elementwise_combined.ptx"
NENT=$(grep -c '\.entry' /tmp/out.ptx)
echo "  emitted $(stat -c%s /tmp/out.ptx) bytes, $NENT .entry kernels -> $OUT/elementwise_combined.ptx"

echo "=== [2] PTX-PROVENANCE (grep the OUTPUT, never source) ==="
grep -q 'ex2\.approx\.f32'   /tmp/out.ptx && echo "  ex2.approx.f32 PRESENT (GELU exp inside tanh)"    || { echo "  PROVENANCE FAIL: no ex2.approx";   RC=3; }
grep -q 'rsqrt\.approx\.f32' /tmp/out.ptx && echo "  rsqrt.approx.f32 PRESENT (ADAM 1/sqrt(vhat+eps))" || { echo "  PROVENANCE FAIL: no rsqrt.approx"; RC=3; }
grep -q '0[fF]3F4C422A'      /tmp/out.ptx && echo "  GELU const 0.7978846=sqrt(2/pi) PRESENT"          || { echo "  PROVENANCE FAIL: no GELU c1";     RC=3; }
grep -q '0[fF]3D372713'      /tmp/out.ptx && echo "  GELU const 0.044715 PRESENT"                      || { echo "  PROVENANCE FAIL: no GELU c2";     RC=3; }
grep -q '0[fF]3F666666'      /tmp/out.ptx && echo "  ADAM const 0.9=b1 PRESENT"                        || { echo "  PROVENANCE FAIL: no ADAM b1";     RC=3; }
grep -q '0[fF]3F7FBE76'      /tmp/out.ptx && echo "  ADAM const 0.999=b2 PRESENT"                      || { echo "  PROVENANCE FAIL: no ADAM b2";     RC=3; }
grep -q '\.entry gpu_gelu'   /tmp/out.ptx && echo "  gpu_gelu entry PRESENT"                           || { echo "  PROVENANCE FAIL: no gpu_gelu entry"; RC=3; }
grep -q '\.entry gpu_adam'   /tmp/out.ptx && echo "  gpu_adam entry PRESENT"                           || { echo "  PROVENANCE FAIL: no gpu_adam entry"; RC=3; }
grep -q '\.target sm_86'     /tmp/out.ptx && echo "  .target sm_86 PRESENT"                            || { echo "  PROVENANCE FAIL: not sm_86";      RC=3; }

echo "=== [3] ptxas acceptance (sm_86, -v for occupancy/spill) ==="
"$PTXAS" -arch=sm_86 -v /tmp/out.ptx -o "$OUT/elementwise_combined.cubin" 2>&1 | tee "$OUT/elementwise_ptxas.log" || { echo "  PTXAS_REJECT"; exit 2; }
echo "  PTXAS_ACCEPT ($PTXAS)"

echo "=== [4] build host launcher (cuda_launch.c + cuBLAS) ==="
gcc helixc/runtime/cuda_launch.c -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/cl \
  || { echo "FATAL gcc"; exit 2; }

run_ns () {  # $1=op-mode $2=kernel-name $3=N-string(pipe-separated)
  local op="$1" kn="$2" nstr="$3"
  local oldifs="$IFS"; IFS='|'; local cases=($nstr); IFS="$oldifs"
  local n
  for n in "${cases[@]}"; do
    n=$(echo $n | tr -d ' ')
    [[ -n "$n" ]] || continue
    echo "  -- $op N=$n --"
    timeout 90 /tmp/cl /tmp/out.ptx "$kn" "$n" "$op" 2>&1 | grep -E "gelu|adam|THROUGHPUT|mismatch" | tee -a "$OUT/elementwise_correctness.log"
    local rc=${PIPESTATUS[0]}; [[ "$rc" = "0" ]] || RC=1
  done
}

echo "=== [5] CORRECTNESS + HONEST THROUGHPUT: GELU ==="
run_ns gelu gpu_gelu "$GE_N"
echo "=== [5] CORRECTNESS + HONEST THROUGHPUT: ADAM ==="
run_ns adam gpu_adam "$AD_N"

echo "=== [6] NEG-CONTROL A (comparator teeth: mutate one out cell -> MUST FAIL) ==="
if timeout 90 /tmp/cl /tmp/out.ptx gpu_gelu 256 gelu mutate >/dev/null 2>&1; then
  echo "  NEG-CONTROL-A(gelu) FAIL: mutated compare returned PASS"; RC=4
else echo "  NEG-CONTROL-A(gelu) OK: mutated compare correctly FAILED"; fi
if timeout 90 /tmp/cl /tmp/out.ptx gpu_adam 256 adam mutate >/dev/null 2>&1; then
  echo "  NEG-CONTROL-A(adam) FAIL: mutated compare returned PASS"; RC=4
else echo "  NEG-CONTROL-A(adam) OK: mutated compare correctly FAILED"; fi

echo "=== [7] NEG-CONTROL B (transcendental-strip, load-bearing) ==="
# GELU: strip every ex2.approx.f32 line -> the exp (hence tanh, hence GELU) is gone ->
# mis-computes. ADAM: strip every rsqrt.approx.f32 -> the 1/sqrt is gone -> mis-computes.
NE=$(grep -c 'ex2\.approx\.f32' /tmp/out.ptx)
NR=$(grep -c 'rsqrt\.approx\.f32' /tmp/out.ptx)
grep -v 'ex2\.approx\.f32'   /tmp/out.ptx > /tmp/out_e_noexp.ptx
grep -v 'rsqrt\.approx\.f32' /tmp/out.ptx > /tmp/out_e_norsqrt.ptx
cp /tmp/out_e_noexp.ptx "$OUT/elementwise_noexp.ptx"
cp /tmp/out_e_norsqrt.ptx "$OUT/elementwise_norsqrt.ptx"
echo "  stripped $NE ex2.approx.f32 line(s) (GELU) + $NR rsqrt.approx.f32 line(s) (ADAM) from the emitted PTX"
# The stripped PTX leaves the result register undefined -> ptxas may accept (just wrong);
# either way the kernel mis-computes. Run each stripped variant against its own op.
if "$PTXAS" -arch=sm_86 /tmp/out_e_noexp.ptx -o "$OUT/e_noexp.cubin" 2>/dev/null; then
  echo "  no-exp PTX still ptxas-accepts (valid, just wrong)"
else echo "  NOTE: no-exp PTX ptxas-rejected (still proves ex2.approx is structural)"; fi
if timeout 90 /tmp/cl /tmp/out_e_noexp.ptx gpu_gelu 256 gelu >/dev/null 2>&1; then
  echo "  NEG-CONTROL-B(gelu) FAIL: no-exp kernel PASSED (ex2.approx NOT load-bearing?!)"; RC=4
else echo "  NEG-CONTROL-B(gelu) OK: no-exp kernel mis-computed -> GELU exp IS load-bearing"; fi
if "$PTXAS" -arch=sm_86 /tmp/out_e_norsqrt.ptx -o "$OUT/e_norsqrt.cubin" 2>/dev/null; then
  echo "  no-rsqrt PTX still ptxas-accepts (valid, just wrong)"
else echo "  NOTE: no-rsqrt PTX ptxas-rejected (still proves rsqrt.approx is structural)"; fi
if timeout 90 /tmp/cl /tmp/out_e_norsqrt.ptx gpu_adam 256 adam >/dev/null 2>&1; then
  echo "  NEG-CONTROL-B(adam) FAIL: no-rsqrt kernel PASSED (rsqrt.approx NOT load-bearing?!)"; RC=4
else echo "  NEG-CONTROL-B(adam) OK: no-rsqrt kernel mis-computed -> ADAM 1/sqrt IS load-bearing"; fi

echo "=== GPU ELEMENTWISE CORPUS VERDICT (wall $(( $(date +%s) - T0 ))s) ==="
if [[ "$RC" = "0" ]]; then echo "GPU_ELEMENTWISE_PASS (GELU + ADAM correct vs CPU at small+large, honest GB/s throughput, both neg-controls trip per op)";
else echo "GPU_ELEMENTWISE_FAIL (rc=$RC)"; fi
exit $RC
