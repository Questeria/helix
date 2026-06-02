#!/usr/bin/env bash
# GPU TRANSPOSED-GEMM corpus (T2/M4, 2026-06-02). Run as a FILE under WSL:
#   wsl.exe bash -c "bash /mnt/c/Projects/Kovostov-Native/scripts/gpu_transpose_corpus.sh"
#
# Correctness + faster-than-naive harness for the kovc SMEM-tiled transposed GEMMs:
#   tiled_matmul_abt = A@B^T  (emit_ptx_tiled_matmul_t mode 0; intrinsic __matmul_abt_smem)
#   tiled_matmul_atb = A^T@B  (emit_ptx_tiled_matmul_t mode 1; intrinsic __matmul_atb_smem)
# Each is checked (a) CORRECT cell-by-cell vs a CPU transposed-GEMM oracle (integer-exact
# == compare) at small AND large sizes, and (b) FASTER-THAN-NAIVE vs the existing naive
# non-tiled kernel (gpu_matmul_abt / gpu_matmul_atb) -- the bar is a measured speedup>1,
# NOT a TFLOP/s target. Both negative controls trip (comparator-teeth + bar.sync-strip).
#
# Pipeline:
#   [A] regenerate k1ptxdrv.hx from the CURRENT kovc.hx (assemble_k1) -- picks up the new emitter
#   [B] seed -> K1 PTX driver (the slow ~3-4min step; ONE build, NOT the 4x fixpoint)
#   [1] emit ONE combined.ptx with all 4 kernels (tiled_abt + tiled_atb + naive abt + naive atb)
#   [2] PROVENANCE (grep the OUTPUT, never source): .shared / bar.sync / ld.shared / st.shared /
#       fma.rn.f32 / sm_86 present; 2 tiled .entry kernels present
#   [3] ptxas-accept the combined PTX for sm_86
#   [4] build the host launcher (cuda_launch.c)
#   [5] CORRECTNESS+SPEEDUP corpus: gemm_abt + gemm_atb at small (64^3) AND large (512^3) and a
#       non-square large case; each MUST be vs-CPU=0 vs-naive=0 AND faster-than-naive=YES
#   [6] NEG-CONTROL A (comparator teeth): mutate one C cell -> MUST FAIL (both variants)
#   [7] NEG-CONTROL bar.sync-strip (load-bearing): strip every bar.sync from the emitted PTX,
#       ptxas-accept, run -> MUST mis-compute (FAIL), proving .shared/bar.sync are load-bearing
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
# Sizes (all satisfy the tiled divisibility; N<=1024 for the naive baseline blockDim.x=N).
# abt: M%64,N%64,K%8 ; atb: K%64,N%64,M%8. SMALL = correctness-only (too little work to
# amortize the tiling overhead, so the naive kernel can match a tiny problem -- gating
# faster-than-naive there would be dishonest); LARGE + NON-SQUARE = correctness AND the
# faster-than-naive GATE. Space-separated triples; pipe '|' separates cases.
ABT_CORR_DIMS="${ABT_CORR_DIMS:-64 64 64}"
ATB_CORR_DIMS="${ATB_CORR_DIMS:-64 64 64}"
ABT_PERF_DIMS="${ABT_PERF_DIMS:-512 512 512 | 256 128 512}"
ATB_PERF_DIMS="${ATB_PERF_DIMS:-512 512 512 | 128 256 512}"
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

echo "=== [1] emit ONE combined.ptx (tiled_abt + tiled_atb + naive abt + naive atb) ==="
: > /tmp/combined_t.hx
for k in tiled_matmul_abt_kernel tiled_matmul_atb_kernel gpu_matmul_abt_kernel gpu_matmul_atb_kernel; do
  tr -d '\r' < "$EX/${k}.hx" >> /tmp/combined_t.hx; echo "" >> /tmp/combined_t.hx
done
cp /tmp/combined_t.hx /tmp/kernel_in.hx
rm -f /tmp/out.ptx
"$DRV" >/dev/null
[[ -s /tmp/out.ptx ]] || { echo "FATAL: driver emitted no PTX"; exit 6; }
cp /tmp/out.ptx "$OUT/transpose_combined.ptx"
NENT=$(grep -c '\.entry' /tmp/out.ptx)
echo "  emitted $(stat -c%s /tmp/out.ptx) bytes, $NENT .entry kernels -> $OUT/transpose_combined.ptx"

echo "=== [2] PTX-PROVENANCE (grep the OUTPUT, never source) ==="
grep -q '\.shared'        /tmp/out.ptx && echo "  .shared PRESENT"        || { echo "  PROVENANCE FAIL: no .shared";   RC=3; }
grep -q 'bar\.sync 0'     /tmp/out.ptx && echo "  bar.sync 0 PRESENT"     || { echo "  PROVENANCE FAIL: no bar.sync";  RC=3; }
grep -q 'ld\.shared\.f32' /tmp/out.ptx && echo "  ld.shared.f32 PRESENT"  || { echo "  PROVENANCE FAIL: no ld.shared"; RC=3; }
grep -q 'st\.shared\.f32' /tmp/out.ptx && echo "  st.shared.f32 PRESENT"  || { echo "  PROVENANCE FAIL: no st.shared (scalar coop load)"; RC=3; }
grep -q 'fma\.rn\.f32'    /tmp/out.ptx && echo "  fma.rn.f32 PRESENT"     || { echo "  PROVENANCE FAIL: no fma.rn.f32"; RC=3; }
grep -q '\.target sm_86'  /tmp/out.ptx && echo "  .target sm_86 PRESENT"  || { echo "  PROVENANCE FAIL: not sm_86"; RC=3; }
grep -q '\.entry tiled_matmul_abt' /tmp/out.ptx && echo "  tiled_matmul_abt entry PRESENT" || { echo "  PROVENANCE FAIL: no tiled_matmul_abt entry"; RC=3; }
grep -q '\.entry tiled_matmul_atb' /tmp/out.ptx && echo "  tiled_matmul_atb entry PRESENT" || { echo "  PROVENANCE FAIL: no tiled_matmul_atb entry"; RC=3; }

echo "=== [3] ptxas acceptance (sm_86, -v for occupancy/spill) ==="
"$PTXAS" -arch=sm_86 -v /tmp/out.ptx -o "$OUT/transpose_combined.cubin" 2>&1 | tee "$OUT/transpose_ptxas.log" || { echo "  PTXAS_REJECT"; exit 2; }
echo "  PTXAS_ACCEPT ($PTXAS)"

echo "=== [4] build host launcher (cuda_launch.c + cuBLAS) ==="
gcc helixc/runtime/cuda_launch.c -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/cl \
  || { echo "FATAL gcc"; exit 2; }

run_dims () {  # $1=op $2=kernel-name $3=extra(corr|"") $4=dims-string(pipe-separated triples)
  local op="$1" kn="$2" extra="$3" dimstr="$4"
  local oldifs="$IFS"
  IFS='|'; local cases=($dimstr); IFS="$oldifs"
  local tri
  for tri in "${cases[@]}"; do
    set -- $tri
    [[ $# -ge 3 ]] || continue
    echo "  -- $op M=$1 K=$2 N=$3 ${extra} --"
    timeout 90 /tmp/cl /tmp/out.ptx "$kn" 0 "$op" "$1" "$2" "$3" $extra 2>&1 | grep -E "tiled_matmul_|SPEEDUP|mismatch" | tee -a "$OUT/transpose_correctness.log"
    local rc=${PIPESTATUS[0]}; [[ "$rc" = "0" ]] || RC=1
  done
}

echo "=== [5a] CORRECTNESS (small, speedup reported not gated): A@B^T ==="
run_dims gemm_abt tiled_matmul_abt corr "$ABT_CORR_DIMS"
echo "=== [5a] CORRECTNESS (small, speedup reported not gated): A^T@B ==="
run_dims gemm_atb tiled_matmul_atb corr "$ATB_CORR_DIMS"
echo "=== [5b] CORRECTNESS + FASTER-THAN-NAIVE GATE (large/non-square): A@B^T ==="
run_dims gemm_abt tiled_matmul_abt "" "$ABT_PERF_DIMS"
echo "=== [5b] CORRECTNESS + FASTER-THAN-NAIVE GATE (large/non-square): A^T@B ==="
run_dims gemm_atb tiled_matmul_atb "" "$ATB_PERF_DIMS"

echo "=== [6] NEG-CONTROL A (comparator teeth: mutate one C cell -> MUST FAIL) ==="
if timeout 90 /tmp/cl /tmp/out.ptx tiled_matmul_abt 0 gemm_abt 64 64 64 mutate >/dev/null 2>&1; then
  echo "  NEG-CONTROL-A(abt) FAIL: mutated compare returned PASS"; RC=4
else echo "  NEG-CONTROL-A(abt) OK: mutated compare correctly FAILED"; fi
if timeout 90 /tmp/cl /tmp/out.ptx tiled_matmul_atb 0 gemm_atb 64 64 64 mutate >/dev/null 2>&1; then
  echo "  NEG-CONTROL-A(atb) FAIL: mutated compare returned PASS"; RC=4
else echo "  NEG-CONTROL-A(atb) OK: mutated compare correctly FAILED"; fi

echo "=== [7] NEG-CONTROL bar.sync-strip (load-bearing: strip bar.sync -> MUST mis-compute) ==="
NB=$(grep -c 'bar\.sync' /tmp/out.ptx)
grep -v 'bar\.sync' /tmp/out.ptx > /tmp/out_t_nobar.ptx
cp /tmp/out_t_nobar.ptx "$OUT/transpose_combined.nobar.ptx"
echo "  stripped $NB bar.sync line(s) from the emitted PTX"
if "$PTXAS" -arch=sm_86 /tmp/out_t_nobar.ptx -o "$OUT/t_nobar.cubin" 2>/dev/null; then
  echo "  no-bar PTX still ptxas-accepts (valid, just unsynchronized)"
else echo "  NOTE: no-bar PTX ptxas-rejected (still proves bar.sync is structural)"; fi
# 256^3 so multiple k-tiles race across the missing barrier -> wrong result.
if timeout 90 /tmp/cl /tmp/out_t_nobar.ptx tiled_matmul_abt 0 gemm_abt 256 256 256 >/dev/null 2>&1; then
  echo "  NEG-CONTROL-bar(abt) FAIL: no-bar kernel PASSED (barriers NOT load-bearing?!)"; RC=4
else echo "  NEG-CONTROL-bar(abt) OK: no-bar kernel mis-computed -> .shared/bar.sync ARE load-bearing"; fi
if timeout 90 /tmp/cl /tmp/out_t_nobar.ptx tiled_matmul_atb 0 gemm_atb 256 256 256 >/dev/null 2>&1; then
  echo "  NEG-CONTROL-bar(atb) FAIL: no-bar kernel PASSED (barriers NOT load-bearing?!)"; RC=4
else echo "  NEG-CONTROL-bar(atb) OK: no-bar kernel mis-computed -> .shared/bar.sync ARE load-bearing"; fi

echo "=== GPU TRANSPOSED-GEMM CORPUS VERDICT (wall $(( $(date +%s) - T0 ))s) ==="
if [[ "$RC" = "0" ]]; then echo "GPU_TRANSPOSE_PASS (tiled A@B^T + A^T@B correct vs CPU at 64^3..512^3 + faster-than-naive, both neg-controls trip)";
else echo "GPU_TRANSPOSE_FAIL (rc=$RC)"; fi
exit $RC
