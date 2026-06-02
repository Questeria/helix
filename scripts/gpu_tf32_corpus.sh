#!/usr/bin/env bash
# GPU TF32 Tensor-Core CORRECTNESS corpus (T3/G3, M-G3.2). Run as a FILE under WSL:
#   wsl.exe bash -c "bash /mnt/c/Projects/Kovostov-Native/scripts/gpu_tf32_corpus.sh"
#
# Correctness-FIRST harness for the kovc mma.sync TF32 GEMM (emit_ptx_tf32_matmul_mma).
# The PERF threshold is DEFERRED this phase (G1_MIN_TFLOPS=0.1, perf reported not gated).
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
CORR_DIMS="${CORR_DIMS:-16 8 8 32 8 8 16 16 8 16 8 16 64 64 64 128 128 128}"
PERF_DIMS="${PERF_DIMS:-128 128 128}"
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

echo "=== [6] PERF (reported, NOT gated this phase) at ${PERF_DIMS} ==="
set -- $PERF_DIMS
PERF_OUT="$(/tmp/cl /tmp/out.ptx tf32_matmul 0 gemm_tf32 "$1" "$2" "$3" 2>&1)"
echo "$PERF_OUT" | grep -E 'TIMING|MEDIAN-TFLOPS-TF32|tf32_matmul' | tee "$OUT/tf32_perf.log"
echo "$PERF_OUT" | grep -q 'tf32_matmul.*PASS' || { echo "  PERF-RUN CORRECTNESS FAIL"; RC=1; }
KOVC_TF="$(echo "$PERF_OUT" | sed -n 's/.*MEDIAN-TFLOPS-TF32 kovc=\([0-9.]*\).*/\1/p')"
echo "  parsed kovc TF32 = ${KOVC_TF:-?} TFLOP/s (perf gate DEFERRED to next phase)"

echo "=== [7] NEG-CONTROL A (comparator teeth: mutate one C cell -> MUST FAIL) ==="
if /tmp/cl /tmp/out.ptx tf32_matmul 0 gemm_tf32 16 8 8 mutate >/dev/null 2>&1; then
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

echo "=== GPU TF32 CORRECTNESS CORPUS VERDICT ==="
if [[ "$RC" = "0" ]]; then echo "GPU_TF32_CORRECTNESS_PASS (kovc TF32 mma == cuBLAS-TF32 @2e-3 over 16x8x8..128^3 distinct-input, f32-anchor==CPU, mma-strip neg-control trips, no fma on accumulators; perf=${KOVC_TF:-?} TFLOP/s [gate deferred])";
else echo "GPU_TF32_CORRECTNESS_FAIL (rc=$RC)"; fi
exit $RC
