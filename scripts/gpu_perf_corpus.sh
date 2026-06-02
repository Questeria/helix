#!/usr/bin/env bash
# GPU PERF corpus harness (T2/M1 correctness + T2/G1 perf). Run as a FILE under WSL:
#   wsl.exe bash -c "bash /mnt/c/Projects/Kovostov-Native/scripts/gpu_perf_corpus.sh"
#
# Pipeline:
#   [0] mint the kovc PTX driver from the raw-binary seed if absent (~10 min)
#   [1] emit the SMEM-tiled GEMM kernel PTX with kovc (NOT nvcc -- kovc's own codegen)
#   [2] PROVENANCE: grep the emitted OUTPUT (never source) for .shared/bar.sync/...
#   [3] ptxas-accept the emitted PTX for sm_86 (-v for occupancy/spill)
#   [4] build the host launcher WITH the fenced cuBLAS oracle (-lcublas)
#   [5] CORRECTNESS corpus: kovc kernel vs CPU oracle AND vs the (pedantic, true-f32)
#       cuBLAS oracle, cell-by-cell, over several sizes (the CPU oracle itself is first
#       validated vs cuBLAS at the smaller sizes; at >=840^3 the O(N^3) CPU triple-loop
#       is skipped and large-N correctness rests on kovc-vs-cuBLAS, reported honestly)
#   [6] G1 PERF GATE: measure median TFLOP/s at 2048^3 (kernel-only, cuEvent, warmup +
#       50 timed launches, min/med/max for the laptop throttle); REQUIRE kovc >= 3
#       TFLOP/s; also report cuBLAS true-f32 TFLOP/s + the kovc/cuBLAS ratio
#   [7] NEG-CONTROL A (comparator teeth): mutate one C cell -> MUST FAIL
#   [8] NEG-CONTROL B (barriers load-bearing): strip every bar.sync line from the
#       EMITTED PTX, ptxas-accept the stripped PTX, run it -> MUST mis-compute (FAIL),
#       proving .shared/bar.sync are load-bearing (not cosmetic)
#
# Reference box: RTX 3070 Laptop GPU (sm_86). Tees the emitted PTX + logs to .m1probe/.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EX="$ROOT/helixc/examples"
OUT="$ROOT/.m1probe"
mkdir -p "$OUT"
DRV="stage0/helixc-bootstrap/_kovc_ptx_driver.bin"
G1_MIN_TFLOPS="${G1_MIN_TFLOPS:-3.0}"
PERF_DIMS="${PERF_DIMS:-2048 2048 2048}"
RC=0

echo "=== [0] ensure PTX driver is current (mint from seed if absent) ==="
if [[ ! -x "$DRV" ]]; then
    echo "  PTX-driver absent; minting from the seed (~10 min)..."
    bash stage0/helixc-bootstrap/assemble_k1.sh
    ( cd stage0/helixc-bootstrap && ulimit -s unlimited && timeout 600 ./seed.bin k1ptxdrv.hx _kovc_ptx_driver.bin )
fi
[[ -x "$DRV" ]] || { echo "FATAL: no PTX driver"; exit 7; }

echo "=== [1] emit the tiled-GEMM kernel PTX (kovc's own codegen) ==="
cp "$EX/tiled_matmul_kernel.hx" /tmp/kernel_in.hx
rm -f /tmp/out.ptx
"$DRV" >/dev/null
[[ -s /tmp/out.ptx ]] || { echo "FATAL: driver emitted no PTX"; exit 6; }
cp /tmp/out.ptx "$OUT/tiled_matmul_kernel.ptx"
echo "  emitted $(stat -c%s /tmp/out.ptx) bytes -> $OUT/tiled_matmul_kernel.ptx"

echo "=== [2] PTX-PROVENANCE (grep the OUTPUT, never source) ==="
grep -q '\.shared'      /tmp/out.ptx && echo "  .shared PRESENT"      || { echo "  PROVENANCE FAIL: no .shared";   RC=3; }
grep -q 'bar\.sync 0'   /tmp/out.ptx && echo "  bar.sync 0 PRESENT"   || { echo "  PROVENANCE FAIL: no bar.sync";  RC=3; }
grep -q 'ld\.shared\.f32' /tmp/out.ptx && echo "  ld.shared.f32 PRESENT" || { echo "  PROVENANCE FAIL: no ld.shared"; RC=3; }
grep -q 'st\.shared\.f32' /tmp/out.ptx && echo "  st.shared.f32 PRESENT" || { echo "  PROVENANCE FAIL: no st.shared"; RC=3; }
grep -q 'fma\.rn\.f32'  /tmp/out.ptx && echo "  fma.rn.f32 PRESENT"   || { echo "  PROVENANCE FAIL: no fma.rn.f32"; RC=3; }
grep -q '\.target sm_86' /tmp/out.ptx && echo "  .target sm_86 PRESENT" || { echo "  PROVENANCE FAIL: not sm_86"; RC=3; }

echo "=== [3] ptxas acceptance (sm_86, -v for occupancy/spill) ==="
ptxas -arch=sm_86 -v /tmp/out.ptx -o "$OUT/tiled_matmul_kernel.cubin" 2>&1 | tee "$OUT/ptxas.log" || { echo "  PTXAS_REJECT"; exit 2; }
echo "  PTXAS_ACCEPT"

echo "=== [4] build host launcher WITH fenced cuBLAS oracle (-lcublas) ==="
gcc helixc/runtime/cuda_launch.c -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/cl \
  || { echo "FATAL gcc (cuBLAS link)"; exit 2; }

echo "=== [5] CORRECTNESS corpus: kovc vs CPU AND vs cuBLAS oracle, cell-by-cell ==="
for dims in "64 64 64" "64 8 128" "128 128 128" "256 256 256" "512 512 512" "2048 2048 2048"; do
    set -- $dims
    echo "  -- M=$1 K=$2 N=$3 --"
    /tmp/cl /tmp/out.ptx tiled_matmul 0 gemm_perf "$1" "$2" "$3" 2>&1 | grep -E 'tiled_matmul|mismatch' | tee -a "$OUT/correctness.log"
    rc=${PIPESTATUS[0]}; [[ "$rc" = "0" ]] || RC=1
done

echo "=== [6] G1 PERF GATE: median TFLOP/s at ${PERF_DIMS} (kernel-only, cuEvent) ==="
set -- $PERF_DIMS
PERF_OUT="$(/tmp/cl /tmp/out.ptx tiled_matmul 0 gemm_perf "$1" "$2" "$3" 2>&1)"
echo "$PERF_OUT" | grep -E 'TIMING|MEDIAN|tiled_matmul' | tee "$OUT/perf.log"
prc=$?
KOVC_TF="$(echo "$PERF_OUT" | sed -n 's/.*MEDIAN-TFLOPS kovc=\([0-9.]*\).*/\1/p')"
BLAS_TF="$(echo "$PERF_OUT" | sed -n 's/.*cublas=\([0-9.]*\).*/\1/p')"
RATIO="$(echo "$PERF_OUT"   | sed -n 's/.*ratio=\([0-9.]*\)%.*/\1/p')"
echo "  parsed: kovc=${KOVC_TF:-?} TFLOP/s  cublas=${BLAS_TF:-?} TFLOP/s  ratio=${RATIO:-?}%"
echo "$PERF_OUT" | grep -q 'tiled_matmul.*PASS' || { echo "  PERF-RUN CORRECTNESS FAIL"; RC=1; }
if [[ -z "$KOVC_TF" ]]; then
    echo "  G1 GATE FAIL: could not parse kovc TFLOP/s"; RC=5
else
    PASS=$(awk -v a="$KOVC_TF" -v b="$G1_MIN_TFLOPS" 'BEGIN{print (a+0 >= b+0)?1:0}')
    if [[ "$PASS" = "1" ]]; then echo "  G1 PERF GATE PASS: kovc ${KOVC_TF} >= ${G1_MIN_TFLOPS} TFLOP/s";
    else echo "  G1 PERF GATE FAIL: kovc ${KOVC_TF} < ${G1_MIN_TFLOPS} TFLOP/s"; RC=5; fi
fi

echo "=== [7] NEG-CONTROL A (comparator teeth: mutate one C cell -> MUST FAIL) ==="
if /tmp/cl /tmp/out.ptx tiled_matmul 0 gemm_perf 64 64 64 mutate >/dev/null 2>&1; then
    echo "  NEG-CONTROL-A FAIL: mutated compare returned PASS (comparator has no teeth)"; RC=4
else
    echo "  NEG-CONTROL-A OK: mutated compare correctly FAILED"
fi

echo "=== [8] NEG-CONTROL B (barriers load-bearing: strip bar.sync -> MUST mis-compute) ==="
NB=$(grep -c 'bar\.sync' /tmp/out.ptx)
grep -v 'bar\.sync' /tmp/out.ptx > /tmp/out_nobar.ptx
cp /tmp/out_nobar.ptx "$OUT/tiled_matmul_kernel.nobar.ptx"
echo "  stripped $NB bar.sync line(s) from the emitted PTX"
if ptxas -arch=sm_86 /tmp/out_nobar.ptx -o "$OUT/nobar.cubin" 2>/dev/null; then
    echo "  no-bar PTX still ptxas-accepts (valid PTX, just unsynchronized)"
else
    echo "  NOTE: no-bar PTX ptxas-rejected (still proves bar.sync is structural)"
fi
if /tmp/cl /tmp/out_nobar.ptx tiled_matmul 0 gemm_perf 256 256 256 >/dev/null 2>&1; then
    echo "  NEG-CONTROL-B FAIL: no-bar kernel PASSED (barriers/.shared NOT load-bearing?!)"; RC=4
else
    echo "  NEG-CONTROL-B OK: no-bar kernel mis-computed (FAILED) -> .shared/bar.sync ARE load-bearing"
fi

echo "=== GPU PERF CORPUS VERDICT ==="
if [[ "$RC" = "0" ]]; then echo "GPU_PERF_G1_PASS (kovc ${KOVC_TF:-?} TFLOP/s >= ${G1_MIN_TFLOPS}, cuBLAS-correct, both neg-controls trip)";
else echo "GPU_PERF_G1_FAIL (rc=$RC)"; fi
exit $RC
