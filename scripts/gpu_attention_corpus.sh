#!/usr/bin/env bash
# GPU FUSED FLASH-STYLE ATTENTION corpus (T2/M4, 2026-06-03). Run as a FILE under WSL:
#   wsl.exe bash -c "bash /mnt/c/Projects/Kovostov-Native/scripts/gpu_attention_corpus.sh"
#
# Correctness + faster-than-naive harness for the kovc FUSED attention kernel
# (emit_ptx_flash_attention; intrinsic __flash_attention): out = softmax(Q@K^T/sqrt(d)) @ V,
# computed with the S x S scores resident ONLY in SHARED MEMORY (never materialized in HBM)
# + a numerically-stable block-reduction softmax (max-subtract) reusing the __softmax_blockred
# tree-reduce primitives, + a 256-thread-parallel P@V output.
#  (a) CORRECT cell-by-cell vs a CPU reference out=softmax(scale*Q@K^T)@V (scale=1/sqrt(d), the
#      SAME runtime rsqrt the kernel uses) -- integer inputs so Q@K^T+scale+@V are EXACT; the
#      only error is the kernel's ex2.approx exp (tol 1e-3, maxrel reported), at small AND large S.
#  (b) FASTER-THAN-NAIVE vs the unfused 3-kernel pipeline (gpu_qkt -> gpu_softmax -> naive_matmul,
#      which round-trips the S x S scores/attn matrices through HBM) -- GATED at the canonical
#      head dim d=16 (the dim the existing gpu_qkt/attention harness uses) at large S. At large
#      head dim d>=64 the kernel is correctness-only (the naive @V matmul is itself well
#      parallelized there; a warp-tiled @V to win at large d is v-next -- stated honestly).
#  (c) THREE negative controls trip: comparator-teeth (mutate one out cell -> FAIL);
#      bar.sync-strip (strip every bar.sync -> the SMEM scores buffer + the block-reduction tree
#      race -> mis-compute, barriers load-bearing); softmax-normalization-strip (force inv=1,
#      dropping the 1/l -> the output is the UNNORMALIZED weighted sum -> mis-compute, the
#      softmax normalization is load-bearing).
#
# Pipeline mirrors gpu_reduction_corpus.sh: [A] regen k1ptxdrv from CURRENT kovc.hx ->
# [B] seed->K1 PTX driver -> [1] emit ONE combined.ptx (flash_attention + the 3 naive baseline
# kernels) -> [2] provenance (grep OUTPUT) -> [3] ptxas-12.8 accept -> [4] host launcher ->
# [5] correctness (small=corr-only, large=corr) -> [6] faster-than-naive GATE (d=16) ->
# [7] 3 neg-controls. Reference box RTX 3070 Laptop (sm_86). GPU runs wrapped in timeout 90.
set -u
T0=$(date +%s)
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EX="$ROOT/helixc/examples"
OUT="$ROOT/.m1probe"; mkdir -p "$OUT"
HB=stage0/helixc-bootstrap
DRV="$HB/_kovc_ptx_driver.bin"
PTXAS="${PTXAS:-/usr/local/cuda/bin/ptxas}"
RC=0
REMINT="${REMINT:-1}"
# correctness sizes (small = correctness-only; large = correctness). "S d" pipe-separated.
A_CORR="${A_CORR:-8 16 | 8 64 | 16 16 | 64 16 | 512 16 | 512 64 | 512 128 | 1024 16}"
# faster-than-naive GATE sizes (d=16 canonical head dim, large S).
A_PERF="${A_PERF:-512 16 | 1024 16}"

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

echo "=== [1] emit ONE combined.ptx (flash_attention + naive gpu_qkt/gpu_softmax/naive_matmul) ==="
: > /tmp/combined_at.hx
for k in flash_attention_kernel gpu_qkt_kernel gpu_softmax_kernel naive_matmul_kernel; do
  tr -d '\r' < "$EX/${k}.hx" >> /tmp/combined_at.hx; echo "" >> /tmp/combined_at.hx
done
cp /tmp/combined_at.hx /tmp/kernel_in.hx
rm -f /tmp/out.ptx
"$DRV" >/dev/null 2>&1 || true
[[ -s /tmp/out.ptx ]] || { echo "FATAL: driver emitted no PTX"; exit 6; }
cp /tmp/out.ptx "$OUT/attention_combined.ptx"
NENT=$(grep -c '\.entry' /tmp/out.ptx)
echo "  emitted $(stat -c%s /tmp/out.ptx) bytes, $NENT .entry kernels -> $OUT/attention_combined.ptx"

echo "=== [2] PTX-PROVENANCE (grep the OUTPUT, never source) ==="
grep -q '\.shared'            /tmp/out.ptx && echo "  .shared PRESENT"           || { echo "  PROV FAIL: no .shared";       RC=3; }
grep -q 'bar\.sync 0'         /tmp/out.ptx && echo "  bar.sync 0 PRESENT"        || { echo "  PROV FAIL: no bar.sync";      RC=3; }
grep -q 'ld\.shared\.f32'     /tmp/out.ptx && echo "  ld.shared.f32 PRESENT"     || { echo "  PROV FAIL: no ld.shared";     RC=3; }
grep -q 'st\.shared\.f32'     /tmp/out.ptx && echo "  st.shared.f32 PRESENT"     || { echo "  PROV FAIL: no st.shared";     RC=3; }
grep -q 'max\.f32'            /tmp/out.ptx && echo "  max.f32 PRESENT (row-max)"  || { echo "  PROV FAIL: no max.f32";       RC=3; }
grep -q 'ex2\.approx\.f32'    /tmp/out.ptx && echo "  ex2.approx.f32 PRESENT (exp)" || { echo "  PROV FAIL: no ex2.approx";   RC=3; }
grep -q 'rsqrt\.approx\.f32'  /tmp/out.ptx && echo "  rsqrt.approx.f32 PRESENT (scale=1/sqrt(d))" || { echo "  PROV FAIL: no rsqrt.approx"; RC=3; }
grep -q 'div\.rn\.f32'        /tmp/out.ptx && echo "  div.rn.f32 PRESENT (1/l)"   || { echo "  PROV FAIL: no div.rn.f32";    RC=3; }
grep -q '\.target sm_86'      /tmp/out.ptx && echo "  .target sm_86 PRESENT"      || { echo "  PROV FAIL: not sm_86";        RC=3; }
grep -q '\.entry flash_attention' /tmp/out.ptx && echo "  flash_attention entry PRESENT" || { echo "  PROV FAIL: no flash_attention entry"; RC=3; }

echo "=== [3] ptxas acceptance (sm_86, -v for smem/spill) ==="
"$PTXAS" -arch=sm_86 -v /tmp/out.ptx -o "$OUT/attention_combined.cubin" 2>&1 | tee "$OUT/attention_ptxas.log" || { echo "  PTXAS_REJECT"; exit 2; }
echo "  PTXAS_ACCEPT ($PTXAS)"

echo "=== [4] build host launcher (cuda_launch.c + cuBLAS) ==="
gcc helixc/runtime/cuda_launch.c -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/cl \
  || { echo "FATAL gcc"; exit 2; }

run_cases () {  # $1=dims-string(pipe "S d") ; uses env CORRONLY (1 -> CORR_ONLY, speedup not gated)
  local oldifs="$IFS"; IFS='|'; local cases=($1); IFS="$oldifs"; local tri envv=""
  [[ "$CORRONLY" = "1" ]] && envv="CORR_ONLY=1"
  for tri in "${cases[@]}"; do
    set -- $tri; [[ $# -ge 2 ]] || continue
    echo "  -- S=$1 d=$2 --"
    env $envv timeout 90 /tmp/cl /tmp/out.ptx flash_attention 0 attn_flash "$1" "$2" 2>&1 | grep -E "attn_flash|TIMING" | tee -a "$OUT/attention_correctness.log"
    local rc=${PIPESTATUS[0]}; [[ "$rc" = "0" ]] || RC=1
  done
}

echo "=== [5] CORRECTNESS (vs CPU ref, tol 1e-3) -- small + large, speedup NOT gated ==="
: > "$OUT/attention_correctness.log"
CORRONLY=1 run_cases "$A_CORR"
echo "=== [6] FASTER-THAN-NAIVE GATE (d=16 canonical head dim, large S) ==="
CORRONLY=0 run_cases "$A_PERF"

echo "=== [7a] NEG-CONTROL A (comparator teeth: mutate -> MUST FAIL) ==="
if timeout 90 /tmp/cl /tmp/out.ptx flash_attention 0 attn_flash 8 16 mutate >/dev/null 2>&1; then
  echo "  NEG-A FAIL: mutated compare returned PASS"; RC=4
else echo "  NEG-A OK: mutated compare correctly FAILED"; fi

echo "=== [7b] NEG-CONTROL bar.sync-strip (barriers load-bearing) ==="
NB=$(grep -c 'bar\.sync' /tmp/out.ptx)
grep -v 'bar\.sync' /tmp/out.ptx > /tmp/out_at_nobar.ptx
cp /tmp/out_at_nobar.ptx "$OUT/attention_combined.nobar.ptx"
echo "  stripped $NB bar.sync line(s)"
"$PTXAS" -arch=sm_86 /tmp/out_at_nobar.ptx -o "$OUT/at_nobar.cubin" 2>/dev/null && echo "  no-bar PTX ptxas-accepts (valid, just unsynchronized)" || echo "  NOTE: no-bar ptxas-rejected"
if timeout 90 /tmp/cl /tmp/out_at_nobar.ptx flash_attention 0 attn_flash 512 16 >/dev/null 2>&1; then
  echo "  NEG-bar FAIL: no-bar kernel PASSED (barriers NOT load-bearing?!)"; RC=4
else echo "  NEG-bar OK: no-bar kernel mis-computed -> SMEM scores/reduction barriers ARE load-bearing"; fi

echo "=== [7c] NEG-CONTROL softmax-normalization-strip (the /l is load-bearing) ==="
awk '
  /\.entry flash_attention/ {inflash=1}
  inflash && /div\.rn\.f32/ && !done { n=split($0, a, /[ ,;]+/); print "    mov.f32 " a[2] ", " a[3] ";"; done=1; next }
  /^\}/ && inflash {inflash=0}
  {print}
' /tmp/out.ptx > /tmp/out_at_nonorm.ptx
cp /tmp/out_at_nonorm.ptx "$OUT/attention_combined.nonorm.ptx"
echo "  forced inv=1 (dropped the softmax /l normalization)"
"$PTXAS" -arch=sm_86 /tmp/out_at_nonorm.ptx -o "$OUT/at_nonorm.cubin" 2>/dev/null && echo "  no-norm PTX ptxas-accepts" || echo "  NOTE: no-norm ptxas-rejected"
if timeout 90 /tmp/cl /tmp/out_at_nonorm.ptx flash_attention 0 attn_flash 8 16 >/dev/null 2>&1; then
  echo "  NEG-norm FAIL: no-norm kernel PASSED (normalization NOT load-bearing?!)"; RC=4
else echo "  NEG-norm OK: no-norm kernel mis-computed -> the softmax /l normalization IS load-bearing"; fi

echo "=== GPU ATTENTION CORPUS VERDICT (wall $(( $(date +%s) - T0 ))s) ==="
if [[ "$RC" = "0" ]]; then echo "GPU_ATTENTION_PASS (fused flash-style attention correct vs CPU at small+large + faster-than-naive @d=16 + all 3 neg-controls trip)";
else echo "GPU_ATTENTION_FAIL (rc=$RC)"; fi
exit $RC
