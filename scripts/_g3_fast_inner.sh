#!/usr/bin/env bash
# FAST INNER LOOP (M-G3.2 correctness debug): assemble k1ptxdrv from CURRENT kovc.hx ->
# seed compiles the driver (K1, the slow ~4min step) -> emit kernel PTX -> ptxas-12.8 ->
# build launcher (WIP cuda_launch.c) -> run vs cublasGemmEx-TF32 at 16x8x8 distinct-input.
# ONE seed->K1 build (NOT the 4x fixpoint). Times the whole loop. Tees to /mnt/c.
set -u
T0=$(date +%s)
ROOT=/mnt/c/Projects/Kovostov-Native
cd "$ROOT"
OUT="$ROOT/.m1probe"; mkdir -p "$OUT"
LOG="$OUT/_g3_fast_inner.log"
exec > >(tee "$LOG") 2>&1
PTXAS=/usr/local/cuda/bin/ptxas
HB=stage0/helixc-bootstrap

echo "=== [A] regenerate k1ptxdrv.hx from current kovc.hx (assemble_k1) ==="
( cd "$HB" && ./seed.bin assemble_k1.hx /tmp/asm_k1.bin && chmod +x /tmp/asm_k1.bin && /tmp/asm_k1.bin ) \
  || { echo "FATAL assemble_k1"; exit 7; }
echo "  k1ptxdrv.hx regenerated ($(stat -c%s $HB/k1ptxdrv.hx) bytes)"

echo "=== [B] seed -> K1 driver build (the slow step) ==="
TB=$(date +%s)
( cd "$HB" && ulimit -s unlimited && timeout 600 ./seed.bin k1ptxdrv.hx _kovc_ptx_driver.bin ) \
  || { echo "FATAL: seed could not compile k1ptxdrv.hx"; exit 6; }
chmod +x "$HB/_kovc_ptx_driver.bin"
TBE=$(date +%s)
echo "  K1 driver built in $((TBE-TB))s ($(stat -c%s $HB/_kovc_ptx_driver.bin) bytes)"

echo "=== [C] emit the TF32 kernel PTX (kovc's own codegen) ==="
cp helixc/examples/tf32_matmul_kernel.hx /tmp/kernel_in.hx
rm -f /tmp/out.ptx
"$HB/_kovc_ptx_driver.bin" >/dev/null
[[ -s /tmp/out.ptx ]] || { echo "FATAL: driver emitted no PTX"; exit 6; }
cp /tmp/out.ptx "$OUT/tf32_matmul_kernel.ptx"
echo "  emitted $(stat -c%s /tmp/out.ptx) bytes"
echo "--- emitted mma/cvt lines (provenance preview) ---"
grep -nE 'mma\.sync|cvt\.rna\.tf32|\.version' /tmp/out.ptx | head -8
echo "--- fma check (must be NONE on accumulators) ---"
grep -c 'fma\.rn\.f32' /tmp/out.ptx

echo "=== [D] ptxas-12.8 accept (sm_86) ==="
"$PTXAS" -arch=sm_86 -v /tmp/out.ptx -o "$OUT/tf32_matmul_kernel.cubin" 2>&1 | tee "$OUT/_g3_ptxas.log" \
  || { echo "PTXAS_REJECT"; exit 2; }
echo "  PTXAS_ACCEPT"

echo "=== [E] build launcher (WIP cuda_launch.c + cuBLAS) ==="
gcc helixc/runtime/cuda_launch.c -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/cl \
  || { echo "FATAL gcc"; exit 2; }

echo "=== [F] CORRECTNESS @16x8x128 distinct-input (kovc-TF32 vs cuBLAS-TF32 @2e-3) ==="
# N must be a multiple of 8*NB*WP = 128 (4 warps x 4 subtiles x 8 cols).
timeout 120 /tmp/cl /tmp/out.ptx tf32_matmul 0 gemm_tf32 16 8 128 2>&1 | tee "$OUT/_g3_corr16.log"
RC=${PIPESTATUS[0]}
T1=$(date +%s)
echo "=== FAST-INNER-LOOP wall-time: $((T1-T0))s (K1 build alone: $((TBE-TB))s) ; corr rc=$RC ==="
exit $RC
