#!/usr/bin/env bash
# llama_kd_conversion.sh -- fp -> ternary STE-QAT conversion of SmolLM2-135M WITH KNOWLEDGE
# DISTILLATION (KD), measured on MMLU held-out perplexity. The KD lever: a FIXED fp teacher (the
# same SmolLM2 with QAT off) supplies SOFT targets so the ternary student matches the teacher's full
# output distribution instead of the hard one-hot label. KD replaces ONLY the top-of-net loss
# gradient (student_softmax - onehot) -> (student_softmax - teacher_soft); the whole rest of the
# backward + Adam is unchanged. Behind HX_KD=1 (default OFF -> the CE path + llama_train_bwd_gate.sh
# are byte-for-byte unaffected).
#
# Pipeline (run as a FILE under WSL; CRLF-strip first):
#   wsl.exe bash -lc "tr -d '\r' < /mnt/c/Projects/Kovostov-Native/scripts/llama_kd_conversion.sh > /tmp/kd.sh && bash /tmp/kd.sh"
#
#   [0] (prereq) the from-raw kovc PTX driver exists (mint once via scripts/llama_train_bwd_gate.sh
#       with REMINT=1; this script REUSES it -- kovc.hx is UNTOUCHED, the kernel set is the concat list).
#   [1] mint the 21-kernel combined PTX = the bwd gate's 20 + gpu_kd_softmax_grad (a @kernel -> no
#       kovc.hx edit); ptxas sm_86 must ACCEPT.
#   [2] build the trainer (gcc, plain).
#   [3] KD correctness self-check: teacher==student -> KD grad ~0 (KD_SELFCHECK_PASS).
#   [4] run the conversion: HX_TERNARY_QAT=1 HX_KD=1, ppl-checkpoint every epoch, on the staged MMLU
#       corpus (regen via scripts/llama_stage_mmlu.sh professional_psychology 256 42 14).
#
# <=80% compute (taskset -c 0-5 nice -n 10); GPU serial. CUDA 12.8 ptxas, sm_86. Data: MMLU (MIT/open).
# Env: EPOCHS (default 5), KDT (KD temperature, default 1), REPO, DRV, PTXAS.
set -u
T0=$(date +%s)
REPO="${REPO:-/mnt/c/Projects/Kovostov-Native}"
cd "$REPO" || { echo "FATAL: no repo $REPO"; exit 9; }
EX="$REPO/helixc/examples"
OUT="$REPO/.m1probe"; mkdir -p "$OUT"
WTS="${LLAMA_WTS:-$REPO/helix-llm/models/smollm2-135m/smollm2-135m.weights}"
DRV="${LLAMA_DRV:-$HOME/gpt2_ext4/llama_kovc_drv.bin}"
PTXAS="${PTXAS:-/usr/local/cuda-12.8/bin/ptxas}"; [ -x "$PTXAS" ] || PTXAS="/usr/local/cuda/bin/ptxas"
EPOCHS="${EPOCHS:-5}"; KDT="${KDT:-1}"
TRAIN="$REPO/helix-llm/mmlu_train_ids.txt"; LENS="$REPO/helix-llm/mmlu_train_lens.txt"
HELD="$REPO/helix-llm/mmlu_heldout_ids.txt"
NICE="taskset -c 0-5 nice -n 10"
[ -s "$WTS" ] || { echo "FATAL: no weights $WTS"; exit 9; }
[ -x "$DRV" ] || { echo "FATAL: minted driver $DRV absent -- run scripts/llama_train_bwd_gate.sh (REMINT=1) once"; exit 7; }
for f in "$TRAIN" "$LENS" "$HELD"; do [ -s "$f" ] || { echo "FATAL: missing corpus $f -- run scripts/llama_stage_mmlu.sh professional_psychology 256 42 14"; exit 9; }; done

echo "=== [1] mint 21-kernel combined PTX (20 gate kernels + gpu_kd_softmax_grad) ==="
KERNELS="tiled_matmul tiled_matmul_abt tiled_matmul_atb gpu_softmax_causal gpu_rmsnorm_fwd_eps gpu_rope_rot gpu_silu_mul gpu_scale_rt vector_add gpu_rmsnorm_bwd_dx gpu_rope_bwd gpu_silu_mul_bwd gpu_repeat_kv_bwd gpu_ce_softmax_grad gpu_matmul_atb gpu_softmax_backward gpu_adam ternarize_dequant row_abs_mean ste_mask gpu_kd_softmax_grad"
: > /tmp/kernel_in.hx
for k in $KERNELS; do
  f="$EX/${k}_kernel.hx"; [ -f "$f" ] || { echo "FATAL: missing kernel source $f"; exit 6; }
  tr -d '\r' < "$f" >> /tmp/kernel_in.hx; echo "" >> /tmp/kernel_in.hx
done
rm -f /tmp/out.ptx
$NICE "$DRV" >/dev/null 2>&1 || true
[ -s /tmp/out.ptx ] || { echo "FATAL: kovc emitted no PTX"; exit 6; }
NENT=$(grep -c '\.entry' /tmp/out.ptx)
cp /tmp/out.ptx "$OUT/llama_train_kd.ptx"
echo "  combined PTX $(stat -c%s "$OUT/llama_train_kd.ptx") B, $NENT .entry kernels (want 21)"
[ "$NENT" = "21" ] || { echo "FATAL: kernel count $NENT != 21"; exit 6; }
if $NICE "$PTXAS" -arch=sm_86 "$OUT/llama_train_kd.ptx" -o /tmp/kd.cubin 2>"$OUT/kd_ptxas.log"; then
  echo "  PTXAS_ACCEPT (sm_86)"
else echo "  PTXAS_REJECT"; cat "$OUT/kd_ptxas.log"; exit 2; fi

echo "=== [2] build the trainer ==="
cd "$REPO/helixc/runtime" || exit 5
gcc llama_train.c -O2 -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -lm -o "$HOME/llama_train" 2>"$OUT/kd_gcc.log" \
  || { echo "FATAL trainer build:"; head -25 "$OUT/kd_gcc.log"; exit 5; }
cd "$REPO"
echo "  built $HOME/llama_train"
printf "504 3575 282 4649 314\n" > "$HOME/ids5.txt"

echo "=== [3]+[4] KD self-check + conversion (HX_TERNARY_QAT=1 HX_KD=1 T=$KDT, $EPOCHS epochs) ==="
HX_TERNARY_QAT=1 HX_KD=1 HX_KD_T="$KDT" HX_PPL_EVERY=1 $NICE \
  "$HOME/llama_train" "$OUT/llama_train_kd.ptx" "$WTS" "$HOME/ids5.txt" \
  --train-corpus "$TRAIN" "$LENS" "$EPOCHS" --ppl "$HELD" 256 2>&1 | tee "$OUT/kd_conversion.log" \
  | grep -aE "kd-check|teacher cache|ft-traj|CONVERTED|FT_PPL_DONE|KD_SELFCHECK"
echo "=== KD CONVERSION DONE (wall $(( $(date +%s) - T0 ))s) ==="
