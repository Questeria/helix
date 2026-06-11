#!/usr/bin/env bash
# llama_model_gate.sh -- G-L1/G-L2 FULL-MODEL gate for the Llama-arch leg (SmolLM2-135M).
# docs/HELIX_LLAMA_PLAN.md section 6. Run as a FILE under WSL:
#   wsl.exe bash -lc "tr -d '\r' < /mnt/c/Projects/Kovostov-Native/scripts/llama_model_gate.sh > /tmp/lmg.sh && bash /tmp/lmg.sh"
#
# FAIL-CLOSED legs (all must pass for LLAMA_MODEL_GATE_PASS):
#   [0] independent oracle loads + G-L0 ops selftest green
#   [1] from-raw kovc mints the 11-kernel PTX (8 GPT-2 + rmsnorm/rope/silu_mul); ptxas sm_86
#   [2] the worker builds from the working tree (plain, no-serve config)
#   [3] oracle reference dumps for the PINNED prompt (ids/block0/logits/argmax/gen)
#   [4] G-L1: GPU post-layer-0 residual vs oracle (compare-block0, tol 2e-3)
#   [5] G-L2a: full-model last-row logits vs oracle (argmax EXACT + max-abs < 5e-2)
#   [6] G-L2b: 20-token greedy generation token-for-token == oracle
#   [7] NEGATIVE CONTROL: a corrupted-weights copy MUST FAIL the logits leg (teeth)
#
# The oracle (helix-llm/tools/llama_numpy_ref.py, uncommitted) reads the ORIGINAL HF
# safetensors -- independent of gpt2_pack and of the GPU path. Weights file produced by:
#   gpt2_pack model.safetensors config.json smollm2-135m.weights --arch llama
set -u
set -o pipefail   # VERIFIER FINDING: without this, `oracle | tee` tested tee's exit, not the oracle's (G-L1 would fail open)
T0=$(date +%s)
ROOT="${HELIX_SRC:-}"; if [ -z "$ROOT" ]; then ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; fi
[ -d "$ROOT/helixc/examples" ] || ROOT="/mnt/c/Projects/Kovostov-Native"
EX="$ROOT/helixc/examples"
TOOLS="$ROOT/helix-llm/tools"
MODELD="${LLAMA_MODEL_D:-$ROOT/helix-llm/models/smollm2-135m}"   # override: LLAMA_MODEL_D=<dir>
WTS="${LLAMA_WTS:-$MODELD/$(basename "$MODELD").weights}"
export LLAMA_MODEL_DIR="$MODELD"                  # the oracle reads the SAME model dir
export LLAMA_CHAT="${LLAMA_CHAT:-0}"              # 1 = templated-chat prompt mode (instruct models)
[ "$LLAMA_CHAT" = "1" ] && export HX_EOS="${HX_EOS:-2}"   # worker stops after <|im_end|> exactly like the oracle
OUT="$ROOT/.m1probe"; mkdir -p "$OUT"
WORK="${HELIX_WORK:-$HOME/gpt2_ext4/Kovostov-Native}"
BS_W="$WORK/stage0/helixc-bootstrap"
DRV="${LLAMA_DRV:-$HOME/gpt2_ext4/llama_kovc_drv.bin}"   # ext4: survives WSL /tmp resets
PTXAS="${PTXAS:-/usr/local/cuda-12.8/bin/ptxas}"; [ -x "$PTXAS" ] || PTXAS="/usr/local/cuda/bin/ptxas"
PROMPT="${PROMPT:-The capital of France is}"
NGEN="${NGEN:-20}"
RC=0

echo "=================== HELIX LLAMA MODEL GATE (G-L1/G-L2)  $(date -u +%H:%M:%S) ==================="
echo "  root=$ROOT  model=$MODELD  chat=$LLAMA_CHAT  prompt='$PROMPT' n_gen=$NGEN"

echo "=== [0] oracle present + ops selftest ==="
[ -f "$TOOLS/llama_numpy_ref.py" ] || { echo "FATAL: no full-model oracle"; echo "LLAMA_MODEL_GATE_FAIL"; exit 9; }
[ -s "$WTS" ] || { echo "FATAL: no packed weights $WTS (run gpt2_pack --arch llama)"; echo "LLAMA_MODEL_GATE_FAIL"; exit 9; }
if ( cd "$TOOLS" && python3 llama_ops_numpy_ref.py 2>&1 | grep -q '^LLAMA_OPS_REF_SELFTEST: PASS' ); then
  echo "  ops oracle selftest PASS"
else
  echo "  FATAL: ops oracle selftest failed"; echo "LLAMA_MODEL_GATE_FAIL"; exit 9
fi

echo "=== [1] from-raw kovc -> 11-kernel PTX -> ptxas sm_86 ==="
if [ ! -x "$DRV" ]; then
  mkdir -p "$BS_W" "$WORK/helixc/bootstrap"
  cp -r "$ROOT/stage0/helixc-bootstrap/." "$BS_W"/
  cp "$ROOT/helixc/bootstrap/lexer.hx" "$ROOT/helixc/bootstrap/parser.hx" "$ROOT/helixc/bootstrap/kovc.hx" "$WORK/helixc/bootstrap/"
  sed -i "s#/mnt/c/Projects/Kovostov-Native/#$WORK/#g" "$BS_W/assemble_k1.hx"
  _seedsha=$(sha256sum "$BS_W/seed.bin" 2>/dev/null | cut -c1-8)
  [ "$_seedsha" = "9837db12" ] || { echo "FATAL: ext4 seed sha $_seedsha != 9837db12"; echo "LLAMA_MODEL_GATE_FAIL"; exit 7; }
  cd "$BS_W" || exit 7
  rm -f /tmp/asm_k1_lm.bin
  ( ulimit -s unlimited; timeout 600 ./seed.bin assemble_k1.hx /tmp/asm_k1_lm.bin ) || { echo "FATAL assemble_k1"; echo "LLAMA_MODEL_GATE_FAIL"; exit 7; }
  chmod +x /tmp/asm_k1_lm.bin
  ( ulimit -s unlimited; timeout 600 /tmp/asm_k1_lm.bin ) || { echo "FATAL assemble_k1 concat"; echo "LLAMA_MODEL_GATE_FAIL"; exit 7; }
  ( ulimit -s unlimited; timeout 1200 ./seed.bin k1ptxdrv.hx "$DRV" ) || { echo "FATAL k1ptxdrv build"; echo "LLAMA_MODEL_GATE_FAIL"; exit 6; }
  chmod +x "$DRV"
  echo "  from-raw PTX driver minted ($(stat -c%s "$DRV") B)"
else
  echo "  reusing minted driver $DRV ($(stat -c%s "$DRV") B)"
fi
: > /tmp/kernel_in.hx
for k in tiled_matmul tiled_matmul_abt gpu_softmax_causal gpu_layernorm_fwd_eps gpu_add_bias_rowbcast gpu_gelu_stable vector_add gpu_scale_rt gpu_rmsnorm_fwd_eps gpu_rope_rot gpu_silu_mul gpu_gemv_abt gpu_gemv_ab gpu_softmax_row; do
  tr -d '\r' < "$EX/${k}_kernel.hx" >> /tmp/kernel_in.hx; echo "" >> /tmp/kernel_in.hx
done
rm -f /tmp/out.ptx
"$DRV" >/dev/null 2>&1 || true
[ -s /tmp/out.ptx ] || { echo "FATAL: kovc emitted no PTX"; echo "LLAMA_MODEL_GATE_FAIL"; exit 6; }
NENT=$(grep -c '\.entry' /tmp/out.ptx)
cp /tmp/out.ptx /tmp/llama_model.ptx
echo "  PTX $(stat -c%s /tmp/llama_model.ptx) B, $NENT .entry kernels (want 14)"
[ "$NENT" = "14" ] || { echo "FATAL: kernel count $NENT != 14"; echo "LLAMA_MODEL_GATE_FAIL"; exit 6; }
"$PTXAS" -arch=sm_86 /tmp/llama_model.ptx -o /tmp/llama_model.cubin 2>"$OUT/llama_model_ptxas.log" \
  && echo "  PTXAS_ACCEPT (sm_86)" || { echo "  PTXAS_REJECT"; cat "$OUT/llama_model_ptxas.log"; RC=2; }

echo "=== [2] build the worker (plain config, working tree) ==="
cd "$ROOT/helixc/runtime" || exit 5
gcc gpt2_infer.c -O2 -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -lm -o /tmp/llama_infer 2>"$OUT/llama_infer_gcc.log" \
  || { echo "FATAL worker build:"; head -20 "$OUT/llama_infer_gcc.log"; echo "LLAMA_MODEL_GATE_FAIL"; exit 5; }
echo "  built /tmp/llama_infer"

echo "=== [3] oracle reference dumps (pinned prompt) ==="
( cd "$TOOLS" && python3 llama_numpy_ref.py dump-block0 "$PROMPT" ) > "$OUT/llama_orc_b0.log" 2>&1 || { echo "FATAL oracle dump-block0"; tail -5 "$OUT/llama_orc_b0.log"; echo "LLAMA_MODEL_GATE_FAIL"; exit 4; }
( cd "$TOOLS" && python3 llama_numpy_ref.py dump-logits "$PROMPT" ) > "$OUT/llama_orc_lg.log" 2>&1 || { echo "FATAL oracle dump-logits"; tail -5 "$OUT/llama_orc_lg.log"; echo "LLAMA_MODEL_GATE_FAIL"; exit 4; }
( cd "$TOOLS" && python3 llama_numpy_ref.py dump-gen "$PROMPT" "$NGEN" ) > "$OUT/llama_orc_gen.log" 2>&1 || { echo "FATAL oracle dump-gen"; tail -5 "$OUT/llama_orc_gen.log"; echo "LLAMA_MODEL_GATE_FAIL"; exit 4; }
REFD="$ROOT/helix-llm/ref"
echo "  oracle refs: $(cat "$REFD/llama_ref_ids.txt") (argmax $(cat "$REFD/llama_ref_argmax.txt"))"

echo "=== [4] G-L1: post-layer-0 residual parity ==="
/tmp/llama_infer /tmp/llama_model.ptx "$WTS" --block0-dump "$REFD/llama_ref_ids.txt" /tmp/llama_block0_gpu.bin \
  > "$OUT/llama_gl1.log" 2>&1 || { echo "  G-L1 worker run FAIL"; tail -10 "$OUT/llama_gl1.log"; RC=1; }
if [ -s /tmp/llama_block0_gpu.bin ]; then
  if ( cd "$TOOLS" && python3 llama_numpy_ref.py compare-block0 /tmp/llama_block0_gpu.bin "$PROMPT" 2e-3 ) | tee -a "$OUT/llama_gl1.log"; then
    GL1=PASS
  else GL1=FAIL; RC=1; fi
else GL1=FAIL; RC=1; fi
echo "  G-L1: $GL1"

echo "=== [5] G-L2a: full-model logits parity (argmax exact) ==="
if /tmp/llama_infer /tmp/llama_model.ptx "$WTS" --logits "$REFD/llama_ref_logits_last.bin" "$REFD/llama_ref_argmax.txt" "$REFD/llama_ref_ids.txt" \
     2>&1 | tee "$OUT/llama_gl2a.log" | grep -q 'GPT2_LOGITS_PARITY_PASS'; then
  GL2A=PASS
else GL2A=FAIL; RC=1; fi
grep -E 'argmax|max_abs' "$OUT/llama_gl2a.log" | sed 's/^/  /'
echo "  G-L2a: $GL2A"

echo "=== [6] G-L2b: $NGEN-token greedy token-for-token ==="
if env -u HX_KV -u HX_RESIDENT -u HX_FAST /tmp/llama_infer /tmp/llama_model.ptx "$WTS" --generate "$NGEN" "$REFD/llama_ref_ids.txt" "$REFD/llama_ref_gen_ids.txt" \
     2>&1 | tee "$OUT/llama_gl2b.log" | grep -q 'TOKEN_FOR_TOKEN_MATCH'; then
  GL2B=PASS
else GL2B=FAIL; RC=1; fi
grep -E 'HELIX_GEN_IDS|TOKEN' "$OUT/llama_gl2b.log" | sed 's/^/  /'
echo "  G-L2b: $GL2B"

echo "=== [6b] G-KV: KV-cache decode A/B -- ids MUST equal the non-KV run AND the oracle ==="
cp /tmp/helix_gen_ids.txt /tmp/helix_gen_ids_fullfwd.txt 2>/dev/null || true
if HX_KV=1 HX_RESIDENT=1 /tmp/llama_infer /tmp/llama_model.ptx "$WTS" --generate "$NGEN" "$REFD/llama_ref_ids.txt" "$REFD/llama_ref_gen_ids.txt"      2>&1 | tee "$OUT/llama_gkv.log" | grep -q 'TOKEN_FOR_TOKEN_MATCH'; then
  GKV=PASS
else GKV=FAIL; RC=1; fi
if [ -s /tmp/helix_gen_ids_fullfwd.txt ] && cmp -s /tmp/helix_gen_ids.txt /tmp/helix_gen_ids_fullfwd.txt; then
  echo "  A/B: KV ids == full-reforward ids (byte-identical id files)"
else
  echo "  A/B FAIL: KV ids differ from the full-reforward ids"; GKV=FAIL; RC=1
fi
grep -E 'HELIX_GEN_IDS|TOKEN' "$OUT/llama_gkv.log" | tail -2 | sed 's/^/  /'
echo "  G-KV: $GKV"

echo "=== [6e] G-S: SEEDED SAMPLING token-for-token (pinned cfg, oracle RNG == worker RNG) ==="
# DISCLOSED RESIDUAL: both sides run the IDENTICAL pinned sampler, but the input logits
# differ ~1e-5 (GPU fp32 vs numpy). A draw landing inside that boundary gap flips the pick
# (the sampling analogue of an argmax near-tie). The leg therefore pins a config measured
# boundary-clean (seed 42 first diverges at draw #36 on this box) and gates 32 sampled
# tokens; ANY mismatch within the pinned window is a hard FAIL. User-facing reproducibility
# (same seed -> same reply ON THIS STACK) is gated separately below as G-R.
NGEN_S=32
SCFG='{"temperature":0.8,"top_p":0.95,"top_k":40,"rep_penalty":1.1,"seed":42}'
GS=""
( cd "$TOOLS" && LLAMA_SAMPLE="$SCFG" python3 llama_numpy_ref.py dump-gen "$PROMPT" "$NGEN_S" ) > "$OUT/llama_gs_oracle.log" 2>&1 || { echo "  G-S oracle dump FAIL"; GS=FAIL; RC=1; }
if [ -z "$GS" ]; then
  if HX_KV=1 HX_RESIDENT=1 /tmp/llama_infer /tmp/llama_model.ptx "$WTS" --sample "$SCFG" --generate "$NGEN_S" "$REFD/llama_ref_ids.txt" "$REFD/llama_ref_gen_ids.txt" 2>&1 | tee "$OUT/llama_gs.log" | grep -q "TOKEN_FOR_TOKEN_MATCH"; then
    GS=PASS
  else
    GS=FAIL; RC=1
  fi
fi
grep -E "HELIX_GEN_IDS|TOKEN" "$OUT/llama_gs.log" 2>/dev/null | tail -2 | sed "s/^/  /"
echo "  G-S: ${GS:-FAIL}"

echo "=== [6r] G-R: seeded REPRODUCIBILITY (same seed -> byte-identical ids, two live runs) ==="
HX_KV=1 HX_RESIDENT=1 /tmp/llama_infer /tmp/llama_model.ptx "$WTS" --sample "$SCFG" --generate "$NGEN_S" "$REFD/llama_ref_ids.txt" > "$OUT/llama_gr1.log" 2>&1
cp /tmp/helix_gen_ids.txt /tmp/helix_gr_run1.txt 2>/dev/null
HX_KV=1 HX_RESIDENT=1 /tmp/llama_infer /tmp/llama_model.ptx "$WTS" --sample "$SCFG" --generate "$NGEN_S" "$REFD/llama_ref_ids.txt" > "$OUT/llama_gr2.log" 2>&1
if [ -s /tmp/helix_gr_run1.txt ] && [ -s /tmp/helix_gen_ids.txt ] && cmp -s /tmp/helix_gr_run1.txt /tmp/helix_gen_ids.txt; then
  GR=PASS; echo "  two seeded runs byte-identical (the user-facing same-seed guarantee)"
else
  GR=FAIL; RC=1; echo "  G-R FAIL: two identical seeded runs DIFFER (nondeterminism!)"
fi
echo "  G-R: $GR"

echo "=== [6f] temp-0 REGRESSION: --sample temp 0 ids == the greedy ids (byte cmp) ==="
T0CFG='{"temperature":0}'
if HX_KV=1 HX_RESIDENT=1 /tmp/llama_infer /tmp/llama_model.ptx "$WTS" --sample "$T0CFG" --generate "$NGEN" "$REFD/llama_ref_ids.txt" > "$OUT/llama_t0.log" 2>&1 && [ -s /tmp/helix_gen_ids.txt ] && [ -s /tmp/helix_gen_ids_fullfwd.txt ] && cmp -s /tmp/helix_gen_ids.txt /tmp/helix_gen_ids_fullfwd.txt; then
  GT0=PASS; echo "  temp-0 ids byte-identical to greedy"
else
  GT0=FAIL; RC=1; echo "  GT0 FAIL: temp-0 ids differ from greedy (or run failed)"
fi
echo "  G-T0: $GT0"

echo "=== [7] NEGATIVE CONTROL: corrupted weights must FAIL the logits leg ==="
cp "$WTS" /tmp/llama_corrupt.weights
# zero out 1 MB of mid-file weights at the 200 MB offset (well past the 64B header; model-agnostic)
dd if=/dev/zero of=/tmp/llama_corrupt.weights bs=1M seek=200 count=1 conv=notrunc 2>/dev/null
if /tmp/llama_infer /tmp/llama_model.ptx /tmp/llama_corrupt.weights --logits "$REFD/llama_ref_logits_last.bin" "$REFD/llama_ref_argmax.txt" "$REFD/llama_ref_ids.txt" \
     2>&1 | grep -q 'GPT2_LOGITS_PARITY_PASS'; then
  echo "  NEG-CONTROL FAIL: corrupted weights still PASSED (no teeth)"; NEG=FAIL; RC=4
else
  echo "  NEG-CONTROL OK: corrupted weights correctly FAILED"; NEG=PASS
fi
rm -f /tmp/llama_corrupt.weights

echo "=================== LLAMA MODEL GATE VERDICT (wall $(( $(date +%s) - T0 ))s) ==================="
echo "  G-L1 block0   : $GL1"
echo "  G-L2a logits  : $GL2A"
echo "  G-L2b gen     : $GL2B"
echo "  G-KV  kv-decode: ${GKV:-SKIPPED}"
echo "  G-S   sampling : ${GS:-SKIPPED}"
echo "  G-R   reprod.  : ${GR:-SKIPPED}"
echo "  G-T0  temp0    : ${GT0:-SKIPPED}"
echo "  NEG control   : $NEG"
if [ "$GL1$GL2A$GL2B${GKV:-FAIL}${GS:-FAIL}${GR:-FAIL}${GT0:-FAIL}$NEG" = "PASSPASSPASSPASSPASSPASSPASSPASS" ] && [ "$RC" = "0" ]; then
  echo "LLAMA_MODEL_GATE_PASS"; exit 0
else
  echo "LLAMA_MODEL_GATE_FAIL (rc=$RC)"; exit 1
fi
