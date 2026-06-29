#!/usr/bin/env bash
# ternary_convert_run_all.sh -- thin top-level orchestrator for the Helix verifiable ternary
# conversion (SmolLM2-135M). Runs, in order and stopping at the first failure:
#   [0] PREREQ : ensure the from-raw kovc PTX driver exists (mint via llama_train_bwd_gate.sh if not)
#   [1] STAGE  : build the MMLU train + disjoint held-out token corpus
#   [2] CONVERT: fp -> ternary STE-QAT + KD, persist best held-out checkpoint
#   [3] VERIFY : pack + run on the real kernel + emit/re-derive both receipts
#
# It adds NO new compute and edits NOTHING -- it only sequences the four COMMITTED scripts. The
# canonical entry points remain scripts/llama_kd_conversion.sh (convert) and
# scripts/llama_convert_certify.sh (verify); this wrapper is for running the whole chain at once.
# Full doc: docs/HELIX_TERNARY_CONVERT.md.  NO kovc.hx edit (fixpoint 31e6cc27 untouched).
#
# Run as a FILE under WSL (CRLF-strip first, since the working tree may carry CRLF):
#   wsl.exe bash -lc "tr -d '\r' < /mnt/c/Projects/Kovostov-Native/scripts/ternary_convert_run_all.sh > /tmp/tcra.sh && bash /tmp/tcra.sh"
#
# Env (all optional, sane WSL defaults):
#   REPO     repo root (default: the script's parent)
#   SUBJECT  MMLU subject for staging        (default professional_psychology)
#   EPOCHS   conversion epochs               (default 13 -- the schedule that reached the 130x best)
#   KDT      KD temperature                  (default 1)
#   REMINT   force re-mint the from-raw driver in the prereq step (default 0 -> reuse if present)
#   DRV      minted driver path              (default $HOME/gpt2_ext4/llama_kovc_drv.bin)
#   CONVERTED_WEIGHTS  weights handed to VERIFY (default $HOME/smollm2_ternary_converted.bin.best)
set -u
T0=$(date +%s)
REPO="${REPO:-$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)}"
[ -d "$REPO/scripts" ] || REPO="/mnt/c/Projects/Kovostov-Native"
cd "$REPO" || { echo "FATAL: no repo root $REPO"; exit 9; }

SUBJECT="${SUBJECT:-professional_psychology}"
EPOCHS="${EPOCHS:-13}"
KDT="${KDT:-1}"
REMINT="${REMINT:-0}"
DRV="${DRV:-$HOME/gpt2_ext4/llama_kovc_drv.bin}"
CONVERTED_WEIGHTS="${CONVERTED_WEIGHTS:-$HOME/smollm2_ternary_converted.bin.best}"

# run a committed script as a FILE with CRLF stripped, fail-closed on its exit code
run_step() {
  local label="$1" rel="$2"; shift 2
  local src="$REPO/$rel" tmp="/tmp/tcra_$(basename "$rel")"
  echo ""
  echo "============================================================"
  echo " [$label] $rel $*"
  echo "============================================================"
  [ -f "$src" ] || { echo "FATAL: missing $src"; exit 8; }
  tr -d '\r' < "$src" > "$tmp" || { echo "FATAL: could not stage $src"; exit 8; }
  bash "$tmp" "$@"
  local rc=$?
  if [ "$rc" != 0 ]; then echo "*** STOP: [$label] failed (rc=$rc) ***"; exit "$rc"; fi
}

# [0] PREREQ -- mint the from-raw kovc PTX driver iff absent (or REMINT=1). This step also proves
#     the trainer (forward + finite-diff gradient check + fp/QAT smokes) -> LLAMA_TRAIN_BWD_GATE_PASS.
if [ "$REMINT" = "1" ] || [ ! -x "$DRV" ]; then
  echo "[0] PREREQ: minting from-raw kovc PTX driver (REMINT=$REMINT, DRV present=$([ -x "$DRV" ] && echo yes || echo no))"
  REMINT=1 run_step "0/PREREQ-MINT" scripts/llama_train_bwd_gate.sh
else
  echo "[0] PREREQ: reusing existing from-raw driver $DRV"
fi

# [1] STAGE -- MMLU train + disjoint held-out (token ids only; the convert/verify reproduce path).
run_step "1/STAGE" scripts/llama_stage_mmlu.sh "$SUBJECT" 256 42 14

# [2] CONVERT -- fp -> ternary STE-QAT + KD; persists $HOME/smollm2_ternary_converted.bin[.best].
EPOCHS="$EPOCHS" KDT="$KDT" run_step "2/CONVERT" scripts/llama_kd_conversion.sh

# [3] VERIFY -- pack + real-kernel + both receipts on the BEST checkpoint -> LLAMA_CONVERT_CERTIFY_PASS.
[ -s "$CONVERTED_WEIGHTS" ] || { echo "FATAL: converted weights $CONVERTED_WEIGHTS absent (CONVERT did not persist a checkpoint?)"; exit 3; }
HELIX_SRC="$REPO" CONVERTED_WEIGHTS="$CONVERTED_WEIGHTS" run_step "3/VERIFY" scripts/llama_convert_certify.sh

echo ""
echo "============================================================"
echo " TERNARY_CONVERT_RUN_ALL_DONE  (wall $(( $(date +%s) - T0 ))s)"
echo "   convert -> LLAMA_TRAIN_FT_PPL_DONE ; verify -> LLAMA_CONVERT_CERTIFY_PASS"
echo "   HONEST: fp 8.75 -> ternary 1140.55 ppl = 130x gap (NOT near-fp); 14.49x footprint; verifiable, not lossless."
echo "============================================================"
