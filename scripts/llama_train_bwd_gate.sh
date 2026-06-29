#!/usr/bin/env bash
# llama_train_bwd_gate.sh -- G-TRAIN-BWD gate for the SmolLM2-135M (Llama-arch) trainer BACKWARD.
# Wires the full backward + Adam into helixc/runtime/llama_train.c and PROVES it with a multi-eps
# central finite-difference gradient check (the acceptance gate). Run as a FILE under WSL (CRLF
# stripped to /tmp first):
#   wsl.exe bash -lc "tr -d '\r' < /mnt/c/Projects/Kovostov-Native/scripts/llama_train_bwd_gate.sh > /tmp/ltbg.sh && bash /tmp/ltbg.sh"
#
# FAIL-CLOSED legs (all must pass for LLAMA_TRAIN_BWD_GATE_PASS):
#   [1] from-raw kovc mints the 17-kernel combined PTX (8 fwd used by llama_train + tiled_matmul_atb
#       + the 4 new Llama bwd + gpu_ce_softmax_grad + gpu_matmul_atb + gpu_softmax_backward + gpu_adam);
#       ptxas sm_86 accepts. NO kovc.hx edit -- the kernel set is purely the concat list fed to the
#       already-minted driver (the same driver llama_model_gate.sh / llama_ops_bwd_parity.sh use).
#   [2] the trainer builds from the working tree (gcc, plain).
#   [3] FORWARD regression: the forward still reports LLAMA_TRAIN_FWD_OK on the pinned 5-id sequence
#       (the embed_gather D2D-from-resident-weight change must not move the forward).
#   [4] --fdcheck: LLAMA_TRAIN_FDCHECK_PASS (every probed gradient family matches finite-diff).
#   [5] --train smoke: 10 Adam steps drive the loss DOWN (LLAMA_TRAIN_LOSS_DECREASED) -- an
#       independent end-to-end confirmation that the gradients point downhill.
#
# Pinned probe sequence: ids "504 3575 282 4649 314" (Spad=64). SmolLM2 = Apache-2.0; data fully open.
# Reference box: RTX 3070 Laptop (sm_86), CUDA 12.8 ptxas. SERIAL GPU; <=80% compute (taskset + nice).
# Set REMINT=0 to reuse a previously-built from-raw PTX driver on the ext4 mirror.
set -u
T0=$(date +%s)
ROOT="${HELIX_SRC:-}"; if [ -z "$ROOT" ]; then ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; fi
[ -d "$ROOT/helixc/examples" ] || ROOT="/mnt/c/Projects/Kovostov-Native"
cd "$ROOT" || { echo "FATAL: no repo root"; exit 9; }
EX="$ROOT/helixc/examples"
OUT="$ROOT/.m1probe"; mkdir -p "$OUT"
MODELD="${LLAMA_MODEL_D:-$ROOT/helix-llm/models/smollm2-135m}"
WTS="${LLAMA_WTS:-$MODELD/$(basename "$MODELD").weights}"
WORK="${HELIX_WORK:-$HOME/gpt2_ext4/Kovostov-Native}"
BS_W="$WORK/stage0/helixc-bootstrap"
DRV="${LLAMA_DRV:-$HOME/gpt2_ext4/llama_kovc_drv.bin}"   # ext4: survives WSL /tmp resets
PTXAS="${PTXAS:-/usr/local/cuda-12.8/bin/ptxas}"; [ -x "$PTXAS" ] || PTXAS="/usr/local/cuda/bin/ptxas"
REMINT="${REMINT:-1}"
IDS="${LLAMA_IDS:-504 3575 282 4649 314}"
NICE="taskset -c 0-5 nice -n 10"
RC=0

echo "=================== HELIX LLAMA TRAINER BACKWARD GATE  $(date -u +%H:%M:%S) ==================="
echo "  root=$ROOT  weights=$WTS  ptxas=$PTXAS  ids='$IDS'"
[ -s "$WTS" ] || { echo "FATAL: no packed weights $WTS (gpt2_pack --arch llama)"; echo "LLAMA_TRAIN_BWD_GATE_FAIL"; exit 9; }

echo "=== [1] from-raw kovc -> 17-kernel combined PTX -> ptxas sm_86 ==="
if [ "$REMINT" = "1" ] || [ ! -x "$DRV" ]; then
  mkdir -p "$BS_W" "$WORK/helixc/bootstrap"
  cp -r "$ROOT/stage0/helixc-bootstrap/." "$BS_W"/
  cp "$ROOT/helixc/bootstrap/lexer.hx" "$ROOT/helixc/bootstrap/parser.hx" "$ROOT/helixc/bootstrap/kovc.hx" "$WORK/helixc/bootstrap/"
  sed -i "s#/mnt/c/Projects/Kovostov-Native/#$WORK/#g" "$BS_W/assemble_k1.hx"
  _seedsha=$(sha256sum "$BS_W/seed.bin" 2>/dev/null | cut -c1-8)
  [ "$_seedsha" = "9837db12" ] || { echo "FATAL: ext4 seed sha $_seedsha != 9837db12"; echo "LLAMA_TRAIN_BWD_GATE_FAIL"; exit 7; }
  cd "$BS_W" || exit 7
  rm -f /tmp/asm_k1_ltb.bin "$DRV"
  ( ulimit -s unlimited; timeout 600 ./seed.bin assemble_k1.hx /tmp/asm_k1_ltb.bin ) || { echo "FATAL assemble_k1 (seed emit)"; echo "LLAMA_TRAIN_BWD_GATE_FAIL"; exit 7; }
  chmod +x /tmp/asm_k1_ltb.bin
  ( ulimit -s unlimited; timeout 600 /tmp/asm_k1_ltb.bin ) || { echo "FATAL assemble_k1 (concat run)"; echo "LLAMA_TRAIN_BWD_GATE_FAIL"; exit 7; }
  ( ulimit -s unlimited; timeout 1200 ./seed.bin k1ptxdrv.hx "$DRV" ) || { echo "FATAL k1ptxdrv build"; echo "LLAMA_TRAIN_BWD_GATE_FAIL"; exit 6; }
  chmod +x "$DRV"
  cd "$ROOT"
  echo "  from-raw PTX driver minted ($(stat -c%s "$DRV") B)"
else
  echo "  reusing minted driver $DRV ($(stat -c%s "$DRV") B)"
fi
# the kernel set is JUST the concat list -> no kovc.hx edit, the fixpoint is untouched.
KERNELS="tiled_matmul tiled_matmul_abt tiled_matmul_atb gpu_softmax_causal gpu_rmsnorm_fwd_eps gpu_rope_rot gpu_silu_mul gpu_scale_rt vector_add gpu_rmsnorm_bwd_dx gpu_rope_bwd gpu_silu_mul_bwd gpu_repeat_kv_bwd gpu_ce_softmax_grad gpu_matmul_atb gpu_softmax_backward gpu_adam"
: > /tmp/kernel_in.hx
for k in $KERNELS; do
  f="$EX/${k}_kernel.hx"; [ -f "$f" ] || { echo "  FATAL: missing kernel source $f"; echo "LLAMA_TRAIN_BWD_GATE_FAIL"; exit 6; }
  tr -d '\r' < "$f" >> /tmp/kernel_in.hx; echo "" >> /tmp/kernel_in.hx
done
rm -f /tmp/out.ptx
$NICE "$DRV" >/dev/null 2>&1 || true
[ -s /tmp/out.ptx ] || { echo "  FATAL: kovc emitted no PTX"; echo "LLAMA_TRAIN_BWD_GATE_FAIL"; exit 6; }
NENT=$(grep -c '\.entry' /tmp/out.ptx)
cp /tmp/out.ptx "$OUT/llama_train_combined.ptx"
echo "  combined PTX $(stat -c%s "$OUT/llama_train_combined.ptx") B, $NENT .entry kernels (want 17)"
[ "$NENT" = "17" ] || { echo "  FATAL: kernel count $NENT != 17"; echo "LLAMA_TRAIN_BWD_GATE_FAIL"; exit 6; }
if $NICE "$PTXAS" -arch=sm_86 "$OUT/llama_train_combined.ptx" -o /tmp/llama_train.cubin 2>"$OUT/llama_train_ptxas.log"; then
  echo "  PTXAS_ACCEPT (sm_86)"
else
  echo "  PTXAS_REJECT"; cat "$OUT/llama_train_ptxas.log"; echo "LLAMA_TRAIN_BWD_GATE_FAIL"; exit 2
fi

echo "=== [2] build the trainer (working tree) ==="
cd "$ROOT/helixc/runtime" || exit 5
gcc llama_train.c -O2 -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -lm -o "$HOME/llama_train" 2>"$OUT/llama_train_gcc.log" \
  || { echo "FATAL trainer build:"; head -25 "$OUT/llama_train_gcc.log"; echo "LLAMA_TRAIN_BWD_GATE_FAIL"; exit 5; }
echo "  built $HOME/llama_train"
printf '%s\n' "$IDS" > /tmp/llama_train_ids.txt

echo "=== [3] FORWARD regression (LLAMA_TRAIN_FWD_OK) ==="
if $NICE "$HOME/llama_train" "$OUT/llama_train_combined.ptx" "$WTS" /tmp/llama_train_ids.txt 2>&1 | tee "$OUT/llama_train_fwd.log" | grep -q 'LLAMA_TRAIN_FWD_OK'; then
  FWD=PASS
else FWD=FAIL; RC=1; fi
grep -E 'shifted_CE_loss|FWD_OK|FWD_FAIL' "$OUT/llama_train_fwd.log" | sed 's/^/  /'
echo "  FORWARD: $FWD"

echo "=== [4] finite-difference gradient check (LLAMA_TRAIN_FDCHECK_PASS) ==="
if $NICE "$HOME/llama_train" "$OUT/llama_train_combined.ptx" "$WTS" /tmp/llama_train_ids.txt --fdcheck 2>/dev/null | tee "$OUT/llama_train_fdcheck.log" | grep -q 'LLAMA_TRAIN_FDCHECK_PASS'; then
  FD=PASS
else FD=FAIL; RC=1; fi
grep -E '\| PASS$|\| FAIL$|probes PASS' "$OUT/llama_train_fdcheck.log" | sed 's/^/  /'
echo "  FDCHECK: $FD"

echo "=== [5] train smoke: 10 Adam steps, loss must DECREASE ==="
if $NICE "$HOME/llama_train" "$OUT/llama_train_combined.ptx" "$WTS" /tmp/llama_train_ids.txt --train 10 2>/dev/null | tee "$OUT/llama_train_smoke.log" | grep -q 'LLAMA_TRAIN_LOSS_DECREASED'; then
  TR=PASS
else TR=FAIL; RC=1; fi
grep -E 'step|DECREAS' "$OUT/llama_train_smoke.log" | sed 's/^/  /'
echo "  TRAIN-SMOKE: $TR"

echo "=================== LLAMA TRAINER BWD GATE VERDICT (wall $(( $(date +%s) - T0 ))s) ==================="
echo "  combined PTX : 17 kernels, ptxas sm_86 ACCEPT (no kovc.hx edit)"
echo "  forward      : $FWD"
echo "  fdcheck      : $FD"
echo "  train-smoke  : $TR"
if [ "$FWD" = "PASS" ] && [ "$FD" = "PASS" ] && [ "$TR" = "PASS" ] && [ "$RC" = "0" ]; then
  echo "LLAMA_TRAIN_BWD_GATE_PASS"; exit 0
else
  echo "LLAMA_TRAIN_BWD_GATE_FAIL (rc=$RC)"; exit 1
fi
