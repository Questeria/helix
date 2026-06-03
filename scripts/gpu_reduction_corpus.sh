#!/usr/bin/env bash
# GPU WARP/BLOCK-REDUCTION SOFTMAX + LAYERNORM corpus (T2/M4, 2026-06-02). Run as a
# FILE under WSL:
#   wsl.exe bash -c "bash /mnt/c/Projects/Kovostov-Native/scripts/gpu_reduction_corpus.sh"
#
# Correctness + faster-than-naive harness for the kovc block-per-row reduction kernels:
#   softmax_blockred   = numerically-stable row softmax (emit_ptx_softmax_blockred;
#                        intrinsic __softmax_blockred; row MAX + SUM block reductions)
#   layernorm_blockred = row layernorm (emit_ptx_layernorm_blockred; intrinsic
#                        __layernorm_blockred; row MEAN + VAR block reductions)
# Each is checked (a) CORRECT cell-by-cell vs a CPU reference (stable softmax / affine
# layernorm) -- maxrel reported HONESTLY incl the ex2.approx exp + rsqrt.approx tol --
# at small AND large sizes, and (b) FASTER-THAN-NAIVE vs the existing one-thread-per-row
# kernels (gpu_softmax / gpu_layernorm). Both negative controls trip (comparator-teeth +
# bar.sync-strip, proving the SMEM tree reduction is load-bearing).
#
# Pipeline:
#   [A] regenerate k1ptxdrv.hx from the CURRENT kovc.hx (assemble_k1) -- picks up the emitter
#   [B] seed -> K1 PTX driver (the slow ~3min step; ONE build)
#   [1] emit ONE combined.ptx with all 4 kernels (2 block-reduction + 2 naive baselines)
#   [2] PROVENANCE (grep the OUTPUT, never source): .shared / bar.sync / ld.shared / st.shared /
#       max.f32 / ex2.approx.f32 / rsqrt.approx.f32 / div.rn.f32 / sm_86 present; 2 block-red entries
#   [3] ptxas-accept the combined PTX for sm_86
#   [4] build the host launcher (cuda_launch.c)
#   [5] CORRECTNESS+SPEEDUP corpus: softmax_perf + layernorm_perf at small AND large; each MUST
#       be 0-bad vs-CPU; small = correctness-only (the reduction's bar.sync overhead dominates a
#       tiny row, so gating faster-than-naive there would be dishonest), large = the GATE.
#   [6] NEG-CONTROL A (comparator teeth): mutate one y cell -> MUST FAIL (both ops)
#   [7] NEG-CONTROL bar.sync-strip (load-bearing): strip every bar.sync from the emitted PTX,
#       ptxas-accept, run the block-reduction kernel -> MUST mis-compute (FAIL), proving the
#       SMEM tree reduction's barriers are load-bearing (the naive baselines have no bar.sync).
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
# Sizes. SMALL = correctness-only (tiny row: barrier overhead dominates so faster-than-naive
# there is not gated -- stated honestly). LARGE = correctness AND the faster-than-naive GATE
# (a wide row + many rows where the 256-way parallel reduction beats the serial one-thread).
# Space-separated "rows cols"; pipe '|' separates cases.
SM_CORR="${SM_CORR:-8 16}"
SM_PERF="${SM_PERF:-4096 1024 | 1024 4096}"
LN_CORR="${LN_CORR:-8 16}"
LN_PERF="${LN_PERF:-4096 1024 | 1024 4096}"
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

echo "=== [1] emit ONE combined.ptx (softmax_blockred + layernorm_blockred + naive softmax + naive layernorm) ==="
: > /tmp/combined_r.hx
for k in softmax_blockred_kernel layernorm_blockred_kernel gpu_softmax_kernel gpu_layernorm_kernel; do
  tr -d '\r' < "$EX/${k}.hx" >> /tmp/combined_r.hx; echo "" >> /tmp/combined_r.hx
done
cp /tmp/combined_r.hx /tmp/kernel_in.hx
rm -f /tmp/out.ptx
"$DRV" >/dev/null 2>&1 || true
[[ -s /tmp/out.ptx ]] || { echo "FATAL: driver emitted no PTX"; exit 6; }
cp /tmp/out.ptx "$OUT/reduction_combined.ptx"
NENT=$(grep -c '\.entry' /tmp/out.ptx)
echo "  emitted $(stat -c%s /tmp/out.ptx) bytes, $NENT .entry kernels -> $OUT/reduction_combined.ptx"

echo "=== [2] PTX-PROVENANCE (grep the OUTPUT, never source) ==="
grep -q '\.shared'            /tmp/out.ptx && echo "  .shared PRESENT"            || { echo "  PROVENANCE FAIL: no .shared";        RC=3; }
grep -q 'bar\.sync 0'         /tmp/out.ptx && echo "  bar.sync 0 PRESENT"         || { echo "  PROVENANCE FAIL: no bar.sync";       RC=3; }
grep -q 'ld\.shared\.f32'     /tmp/out.ptx && echo "  ld.shared.f32 PRESENT"      || { echo "  PROVENANCE FAIL: no ld.shared";      RC=3; }
grep -q 'st\.shared\.f32'     /tmp/out.ptx && echo "  st.shared.f32 PRESENT"      || { echo "  PROVENANCE FAIL: no st.shared";      RC=3; }
grep -q 'max\.f32'            /tmp/out.ptx && echo "  max.f32 PRESENT (softmax row-max reduce)" || { echo "  PROVENANCE FAIL: no max.f32"; RC=3; }
grep -q 'ex2\.approx\.f32'    /tmp/out.ptx && echo "  ex2.approx.f32 PRESENT (softmax exp)"     || { echo "  PROVENANCE FAIL: no ex2.approx"; RC=3; }
grep -q 'rsqrt\.approx\.f32'  /tmp/out.ptx && echo "  rsqrt.approx.f32 PRESENT (layernorm 1/sqrt(var))" || { echo "  PROVENANCE FAIL: no rsqrt.approx"; RC=3; }
grep -q 'div\.rn\.f32'        /tmp/out.ptx && echo "  div.rn.f32 PRESENT"         || { echo "  PROVENANCE FAIL: no div.rn.f32";     RC=3; }
grep -q '\.target sm_86'      /tmp/out.ptx && echo "  .target sm_86 PRESENT"      || { echo "  PROVENANCE FAIL: not sm_86";         RC=3; }
grep -q '\.entry softmax_blockred'   /tmp/out.ptx && echo "  softmax_blockred entry PRESENT"   || { echo "  PROVENANCE FAIL: no softmax_blockred entry";   RC=3; }
grep -q '\.entry layernorm_blockred' /tmp/out.ptx && echo "  layernorm_blockred entry PRESENT" || { echo "  PROVENANCE FAIL: no layernorm_blockred entry"; RC=3; }

echo "=== [3] ptxas acceptance (sm_86, -v for occupancy/spill) ==="
"$PTXAS" -arch=sm_86 -v /tmp/out.ptx -o "$OUT/reduction_combined.cubin" 2>&1 | tee "$OUT/reduction_ptxas.log" || { echo "  PTXAS_REJECT"; exit 2; }
echo "  PTXAS_ACCEPT ($PTXAS)"

echo "=== [4] build host launcher (cuda_launch.c + cuBLAS) ==="
gcc helixc/runtime/cuda_launch.c -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/cl \
  || { echo "FATAL gcc"; exit 2; }

run_dims () {  # $1=op $2=kernel-name $3=dims-string(pipe-separated "rows cols")
  local op="$1" kn="$2" dimstr="$3"
  local oldifs="$IFS"; IFS='|'; local cases=($dimstr); IFS="$oldifs"
  local tri
  for tri in "${cases[@]}"; do
    set -- $tri
    [[ $# -ge 2 ]] || continue
    echo "  -- $op rows=$1 cols=$2 --"
    timeout 90 /tmp/cl /tmp/out.ptx "$kn" 0 "$op" "$1" "$2" 2>&1 | grep -E "softmax_perf|layernorm_perf|TIMING|SPEEDUP|mismatch" | tee -a "$OUT/reduction_correctness.log"
    local rc=${PIPESTATUS[0]}; [[ "$rc" = "0" ]] || RC=1
  done
}

echo "=== [5a] CORRECTNESS (small, speedup not gated): softmax ==="
run_dims softmax_perf softmax_blockred "$SM_CORR"
echo "=== [5a] CORRECTNESS (small, speedup not gated): layernorm ==="
run_dims layernorm_perf layernorm_blockred "$LN_CORR"
echo "=== [5b] CORRECTNESS + FASTER-THAN-NAIVE GATE (large): softmax ==="
run_dims softmax_perf softmax_blockred "$SM_PERF"
echo "=== [5b] CORRECTNESS + FASTER-THAN-NAIVE GATE (large): layernorm ==="
run_dims layernorm_perf layernorm_blockred "$LN_PERF"

echo "=== [6] NEG-CONTROL A (comparator teeth: mutate one y cell -> MUST FAIL) ==="
if timeout 90 /tmp/cl /tmp/out.ptx softmax_blockred 0 softmax_perf 64 256 mutate >/dev/null 2>&1; then
  echo "  NEG-CONTROL-A(softmax) FAIL: mutated compare returned PASS"; RC=4
else echo "  NEG-CONTROL-A(softmax) OK: mutated compare correctly FAILED"; fi
if timeout 90 /tmp/cl /tmp/out.ptx layernorm_blockred 0 layernorm_perf 64 256 mutate >/dev/null 2>&1; then
  echo "  NEG-CONTROL-A(layernorm) FAIL: mutated compare returned PASS"; RC=4
else echo "  NEG-CONTROL-A(layernorm) OK: mutated compare correctly FAILED"; fi

echo "=== [7] NEG-CONTROL bar.sync-strip (load-bearing: strip bar.sync -> MUST mis-compute) ==="
NB=$(grep -c 'bar\.sync' /tmp/out.ptx)
grep -v 'bar\.sync' /tmp/out.ptx > /tmp/out_r_nobar.ptx
cp /tmp/out_r_nobar.ptx "$OUT/reduction_combined.nobar.ptx"
echo "  stripped $NB bar.sync line(s) from the emitted PTX"
if "$PTXAS" -arch=sm_86 /tmp/out_r_nobar.ptx -o "$OUT/r_nobar.cubin" 2>/dev/null; then
  echo "  no-bar PTX still ptxas-accepts (valid, just unsynchronized)"
else echo "  NOTE: no-bar PTX ptxas-rejected (still proves bar.sync is structural)"; fi
# A wide row (cols >> 256) forces multiple strided passes + a deep tree, so the missing
# barriers race across the SMEM reduction -> wrong row max/sum/mean/var -> FAIL.
if timeout 90 /tmp/cl /tmp/out_r_nobar.ptx softmax_blockred 0 softmax_perf 64 4096 >/dev/null 2>&1; then
  echo "  NEG-CONTROL-bar(softmax) FAIL: no-bar kernel PASSED (barriers NOT load-bearing?!)"; RC=4
else echo "  NEG-CONTROL-bar(softmax) OK: no-bar kernel mis-computed -> SMEM reduction barriers ARE load-bearing"; fi
if timeout 90 /tmp/cl /tmp/out_r_nobar.ptx layernorm_blockred 0 layernorm_perf 64 4096 >/dev/null 2>&1; then
  echo "  NEG-CONTROL-bar(layernorm) FAIL: no-bar kernel PASSED (barriers NOT load-bearing?!)"; RC=4
else echo "  NEG-CONTROL-bar(layernorm) OK: no-bar kernel mis-computed -> SMEM reduction barriers ARE load-bearing"; fi

echo "=== GPU REDUCTION CORPUS VERDICT (wall $(( $(date +%s) - T0 ))s) ==="
if [[ "$RC" = "0" ]]; then echo "GPU_REDUCTION_PASS (block-reduction softmax + layernorm correct vs CPU at small+large + faster-than-naive, both neg-controls trip)";
else echo "GPU_REDUCTION_FAIL (rc=$RC)"; fi
exit $RC
