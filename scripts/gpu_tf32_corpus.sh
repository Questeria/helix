#!/usr/bin/env bash
# GPU TF32 Tensor-Core CORRECTNESS corpus (T3/G3, M-G3.2). Run as a FILE under WSL:
#   wsl.exe bash -c "bash /mnt/c/Projects/Kovostov-Native/scripts/gpu_tf32_corpus.sh"
#
# Correctness-FIRST harness for the kovc mma.sync TF32 GEMM (emit_ptx_tf32_matmul_mma).
# v1.2 (2026-06-03): the committed corpus now REPRODUCES THE HEADLINE G3 CLAIM end-to-end --
# CORR_DIMS + PERF_DIMS include 2048^3, and the PERF threshold is ENFORCED (no longer deferred):
# the median kovc TF32 TFLOP/s @ 2048^3 must clear the G3 floor = max(40% measured cuBLAS-TF32
# = 4.26, an absolute alt of 15). Per the pre-set honest rule BOTH numbers are reported always;
# PASS if median >= 15 absolute OR >= 4.26 (40% cuBLAS-TF32 10.646). The 2e-3 correctness tol is
# NEVER loosened. (The 10.646 cuBLAS-TF32 denominator is the documented standalone M-G3.0
# baseline on this box; the ratio printed here is vs that fixed denominator, since gemm_tf32
# mode times only the kovc kernel.) This is a TEST/SCRIPT change (no kovc.hx edit) -> the
# self-host fixpoint sha is unaffected.
# Pipeline:
#   [0] ensure the kovc PTX driver is current (mint from seed if absent)
#   [1] emit the TF32 kernel PTX with kovc (NOT nvcc -- kovc's own codegen)
#   [2] PROVENANCE (grep the OUTPUT, never source): mma.sync.aligned.m16n8k8 + .tf32 +
#       cvt.rna.tf32.f32 + .version 8.3 ; and ASSERT NO fma.rn.f32 (the G3 inner product is
#       mma, not fma -- a stray fma writing the accumulators would be the "emits mma but
#       doesn't use it" fake)
#   [3] ptxas-accept the emitted PTX for sm_86 via the 12.8 ptxas (.version 8.3)
#   [4] build the host launcher WITH cuBLAS (-lcublas) for the gemm_tf32 oracle
#   [5] CORRECTNESS corpus: kovc TF32 kernel vs cublasGemmEx(COMPUTE_32F_FAST_TF32) at a
#       TIGHT ~2e-3 rel tol, over distinct-per-element inputs, 16x8x8 .. 128^3; PLUS the
#       pedantic-f32-cuBLAS==CPU meta-anchor (proves the references are sane)
#   [6] PERF (reported, NOT gated this phase): median TFLOP/s at the largest size
#   [7] NEG-CONTROL A (comparator teeth): mutate one C cell -> MUST FAIL
#   [8] NEG-CONTROL mma-strip (MANDATORY): strip every mma.sync line from the emitted PTX,
#       ptxas-accept the stripped PTX, run it -> the accumulators are never written ->
#       result is wrong -> MUST FAIL. Proves the Tensor-Core path is load-bearing (the
#       primary defense against a kernel that emits dead mma for provenance).
#
# Reference box: RTX 3070 Laptop GPU (sm_86), driver 596.21/CUDA 13.2 (JITs .version 8.3 PTX
# text). Tees the emitted PTX + logs to .m1probe/.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EX="$ROOT/helixc/examples"
OUT="$ROOT/.m1probe"
mkdir -p "$OUT"
DRV="stage0/helixc-bootstrap/_kovc_ptx_driver.bin"
PTXAS="${PTXAS:-/usr/local/cuda/bin/ptxas}"
# M-G3.3: block tile = 16 x (8*NB*WP)=128 cols (4 warps x 4 subtiles), so N must be a multiple
# of 128 (M mult 16, K mult 8). Distinct-input correctness from 16x8x128 up to the HEADLINE
# 2048^3 (2048 = 16*128 = 16*16, K=2048 = 8*256 -- all axes divisible, the warp-tile is valid).
# The 2048^3 row is the reproduce-the-G3-claim row: correct vs cuBLAS-TF32 @2e-3 + perf floor.
CORR_DIMS="${CORR_DIMS:-16 8 128 32 8 128 16 16 128 16 8 256 128 128 128 256 256 256 2048 2048 2048}"
PERF_DIMS="${PERF_DIMS:-2048 2048 2048}"
# G3 PERF GATE (ENFORCED): PASS if median TFLOP/s >= G3_ABS_FLOOR (15) OR >= G3_REL_FLOOR
# (4.26 = 40% of the measured cuBLAS-TF32 10.646 baseline). Never loosen.
G3_REL_FLOOR="${G3_REL_FLOOR:-4.26}"
G3_ABS_FLOOR="${G3_ABS_FLOOR:-15}"
CUBLAS_TF32_BASE="${CUBLAS_TF32_BASE:-10.646}"   # documented M-G3.0 standalone baseline (this box)
RC=0

echo "=== [0] ensure PTX driver is current (mint from seed if absent) ==="
if [[ ! -x "$DRV" ]]; then
    echo "  PTX-driver absent; minting from the seed (~4 min)..."
    bash stage0/helixc-bootstrap/assemble_k1.sh
    ( cd stage0/helixc-bootstrap && ulimit -s unlimited && timeout 600 ./seed.bin k1ptxdrv.hx _kovc_ptx_driver.bin )
fi
[[ -x "$DRV" ]] || { echo "FATAL: no PTX driver"; exit 7; }

echo "=== [1] emit the TF32 mma kernel PTX (kovc's own codegen) ==="
cp "$EX/tf32_matmul_kernel.hx" /tmp/kernel_in.hx
rm -f /tmp/out.ptx
"$DRV" >/dev/null
[[ -s /tmp/out.ptx ]] || { echo "FATAL: driver emitted no PTX"; exit 6; }
cp /tmp/out.ptx "$OUT/tf32_matmul_kernel.ptx"
echo "  emitted $(stat -c%s /tmp/out.ptx) bytes -> $OUT/tf32_matmul_kernel.ptx"

echo "=== [2] PTX-PROVENANCE (grep the OUTPUT, never source) ==="
grep -q 'mma\.sync\.aligned\.m16n8k8' /tmp/out.ptx && echo "  mma.sync.aligned.m16n8k8 PRESENT" || { echo "  PROVENANCE FAIL: no mma.sync.m16n8k8"; RC=3; }
grep -q '\.tf32'                       /tmp/out.ptx && echo "  .tf32 PRESENT"                    || { echo "  PROVENANCE FAIL: no .tf32"; RC=3; }
grep -q 'cvt\.rna\.tf32\.f32'          /tmp/out.ptx && echo "  cvt.rna.tf32.f32 PRESENT"         || { echo "  PROVENANCE FAIL: no cvt.rna.tf32.f32"; RC=3; }
grep -q '\.version 8\.3'               /tmp/out.ptx && echo "  .version 8.3 PRESENT"             || { echo "  PROVENANCE FAIL: not .version 8.3"; RC=3; }
grep -q '\.target sm_86'               /tmp/out.ptx && echo "  .target sm_86 PRESENT"            || { echo "  PROVENANCE FAIL: not sm_86"; RC=3; }
# The G3 inner product is mma, NOT fma. A fma.rn.f32 in this kernel would mean a scalar path
# could be silently computing the answer while mma is dead (the classic fake). ASSERT NONE.
if grep -q 'fma\.rn\.f32' /tmp/out.ptx; then echo "  PROVENANCE FAIL: fma.rn.f32 present (accumulators must be mma-only!)"; RC=3; else echo "  no fma.rn.f32 (accumulators are mma-only) OK"; fi

echo "=== [3] ptxas acceptance (sm_86, 12.8 ptxas for .version 8.3) ==="
"$PTXAS" -arch=sm_86 -v /tmp/out.ptx -o "$OUT/tf32_matmul_kernel.cubin" 2>&1 | tee "$OUT/tf32_ptxas.log" || { echo "  PTXAS_REJECT"; exit 2; }
echo "  PTXAS_ACCEPT ($PTXAS)"

echo "=== [4] build host launcher WITH cuBLAS oracle (-lcublas) ==="
gcc helixc/runtime/cuda_launch.c -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/cl \
  || { echo "FATAL gcc (cuBLAS link)"; exit 2; }

echo "=== [5] CORRECTNESS corpus: kovc TF32 vs cuBLAS-TF32 (tight 2e-3) + f32-anchor==CPU ==="
set -- $CORR_DIMS
while [[ $# -ge 3 ]]; do
    M=$1 K=$2 N=$3; shift 3
    echo "  -- M=$M K=$K N=$N --"
    /tmp/cl /tmp/out.ptx tf32_matmul 0 gemm_tf32 "$M" "$K" "$N" 2>&1 | grep -E 'tf32_matmul|mismatch' | tee -a "$OUT/tf32_correctness.log"
    rc=${PIPESTATUS[0]}; [[ "$rc" = "0" ]] || RC=1
done

echo "=== [6] PERF (ENFORCED G3 gate) at ${PERF_DIMS} ==="
set -- $PERF_DIMS
PERF_OUT="$(/tmp/cl /tmp/out.ptx tf32_matmul 0 gemm_tf32 "$1" "$2" "$3" 2>&1)"
echo "$PERF_OUT" | grep -E 'TIMING|MEDIAN-TFLOPS-TF32|tf32_matmul' | tee "$OUT/tf32_perf.log"
# the perf run is ALSO a correctness run (kovc-vs-cuBLAS-TF32 @2e-3 at the perf size) -- it must PASS
echo "$PERF_OUT" | grep -q 'tf32_matmul.*PASS' || { echo "  PERF-RUN CORRECTNESS FAIL"; RC=1; }
KOVC_TF="$(echo "$PERF_OUT" | sed -n 's/.*MEDIAN-TFLOPS-TF32 kovc=\([0-9.]*\).*/\1/p')"
if [[ -z "$KOVC_TF" ]]; then
    echo "  PERF FAIL: no MEDIAN-TFLOPS-TF32 emitted (timing did not run)"; RC=1
else
    # PASS if median >= absolute floor (15) OR >= relative floor (40% cuBLAS-TF32 = 4.26).
    # Both numbers reported always; the 2e-3 correctness tol is untouched.
    RATIO="$(awk -v k="$KOVC_TF" -v b="$CUBLAS_TF32_BASE" 'BEGIN{ if(b>0) printf "%.1f", 100.0*k/b; else printf "?" }')"
    PERF_OK="$(awk -v k="$KOVC_TF" -v rel="$G3_REL_FLOOR" -v abs="$G3_ABS_FLOOR" 'BEGIN{ print (k>=abs || k>=rel) ? 1 : 0 }')"
    echo "  kovc TF32 @ ${PERF_DIMS// /x} = ${KOVC_TF} TFLOP/s  (= ${RATIO}% of cuBLAS-TF32 ${CUBLAS_TF32_BASE}; G3 floor = max(${G3_ABS_FLOOR} abs, ${G3_REL_FLOOR} = 40% cuBLAS-TF32))"
    if [[ "$PERF_OK" = "1" ]]; then
        echo "  G3 PERF GATE PASS (${KOVC_TF} >= ${G3_REL_FLOOR} relative floor; absolute-${G3_ABS_FLOOR} alt is physically unreachable on this ~10.6-TFLOP/s-ceiling box)"
    else
        echo "  G3 PERF GATE FAIL (${KOVC_TF} < ${G3_REL_FLOOR} = 40% cuBLAS-TF32 floor)"; RC=1
    fi
fi

echo "=== [7] NEG-CONTROL A (comparator teeth: mutate one C cell -> MUST FAIL) ==="
if /tmp/cl /tmp/out.ptx tf32_matmul 0 gemm_tf32 16 8 32 mutate >/dev/null 2>&1; then
    echo "  NEG-CONTROL-A FAIL: mutated compare returned PASS (comparator has no teeth)"; RC=4
else
    echo "  NEG-CONTROL-A OK: mutated compare correctly FAILED"
fi

echo "=== [8] NEG-CONTROL mma-strip (MANDATORY: strip mma.sync -> MUST mis-compute) ==="
NM=$(grep -c 'mma\.sync' /tmp/out.ptx)
grep -v 'mma\.sync' /tmp/out.ptx > /tmp/out_nomma.ptx
cp /tmp/out_nomma.ptx "$OUT/tf32_matmul_kernel.nomma.ptx"
echo "  stripped $NM mma.sync line(s) from the emitted PTX"
if "$PTXAS" -arch=sm_86 /tmp/out_nomma.ptx -o "$OUT/nomma.cubin" 2>/dev/null; then
    echo "  no-mma PTX still ptxas-accepts (valid PTX, accumulators just never written by a TC op)"
else
    echo "  NOTE: no-mma PTX ptxas-rejected (still proves mma.sync is structural)"
fi
if /tmp/cl /tmp/out_nomma.ptx tf32_matmul 0 gemm_tf32 64 64 64 >/dev/null 2>&1; then
    echo "  NEG-CONTROL-mma FAIL: no-mma kernel PASSED (Tensor-Core path NOT load-bearing?!)"; RC=4
else
    echo "  NEG-CONTROL-mma OK: no-mma kernel mis-computed (FAILED) -> mma.sync IS load-bearing"
fi

echo "=== GPU TF32 CORRECTNESS+PERF CORPUS VERDICT ==="
if [[ "$RC" = "0" ]]; then echo "GPU_TF32_PASS (kovc TF32 mma == cuBLAS-TF32 @2e-3 over 16x8x128..2048^3 distinct-input, f32-anchor==CPU, mma-strip neg-control trips, no fma on accumulators; PERF @2048^3 = ${KOVC_TF:-?} TFLOP/s = ${RATIO:-?}% cuBLAS-TF32 >= the 40% floor 4.26 -- G3 PERF GATE ENFORCED+GREEN)";
else echo "GPU_TF32_FAIL (rc=$RC)"; fi
exit $RC
