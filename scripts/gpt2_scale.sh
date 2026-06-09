#!/usr/bin/env bash
# SCALE FLEX gate: run a LARGER real GPT-2 (gpt2-large 774M by default, or gpt2-xl 1.5B) through the
# EXACT SAME kovc-emitted PTX kernels as the 124M MVP -- ZERO new ops/kernels, only dimension changes.
# This proves the "same code, bigger model" generalization claim AND validates/measures the demo's
# stated fp32 scale ceiling on the 8 GB sm_86 box.
#
#   MODEL=gpt2-large bash scripts/gpt2_scale.sh        # the floor (774M; comfortably fits 8 GB)
#   MODEL=gpt2-xl    bash scripts/gpt2_scale.sh        # the stretch (1.5B; TIGHT -- may OOM)
#
# Emits, fail-closed (a printed FAIL is never exit 0):
#   GPT2_SCALE_PARITY_PASS  -- the larger model's GPU forward matches the fenced numpy oracle:
#                              PRIMARY  : full-logits last-token argmax == oracle EXACTLY + logit-diff
#                                         under the diagnostic bar, AND greedy generation token-for-token.
#                              FALLBACK : if the oracle OOMs at full-logits/generation scale, block-0
#                                         hidden parity (one block) + a non-degenerate-generation
#                                         coherence check -- the mode used is printed explicitly.
# and propagates the combined verdict to the PROCESS EXIT STATUS.
#
# IDENTICAL discipline to scripts/gpt2_gpu_mvp.sh: the PTX driver is minted FRESH from the 299-byte
# raw-binary seed (seed.bin, pinned sha 9837db12...) every run -- never a cached artifact; the FROZEN
# lexer/parser/kovc sources are mirrored BIT-IDENTICALLY to ext4 (DrvFs byte-by-byte write dodge); the
# SAME forward-only kernel set is concatenated and emitted. The ONLY difference vs the 124M gate is the
# model dims, which are read from the model's config.json and passed to gpt2_infer.c via HX_* env (the
# committed gpt2_infer.c is already dimension-generic -- no source change). STRICTLY SERIAL GPU.
# Weights are read from the fenced helix-llm/ tree (gitignored host glue). Run as a FILE under WSL.
set -u

MODEL="${MODEL:-gpt2-large}"          # model dir under helix-llm/models/ (gpt2-large | gpt2-xl)
NGEN="${NGEN:-20}"
PROMPT="${PROMPT:-The capital of France is}"

# ---- paths (NEVER assign a /mnt path to a variable -- MSYS empties it; use literals inline) ----
WORK=/home/legoa/gpt2_ext4/Kovostov-Native           # ext4 mirror root (fast writes)
BS=$WORK/stage0/helixc-bootstrap                     # ext4 bootstrap
SEED_PIN=9837db12                                    # pinned raw-binary seed sha prefix
OK=1
echo "=================== GPT-2 SCALE FLEX ($MODEL)  $(date -u +%H:%M:%S) ==================="

# ---- [0] ambient HX_* neutralization, THEN set the model dims explicitly from config.json ----
for v in $(env | sed -n 's/^\(HX_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
MODELDIR=/mnt/c/Projects/Kovostov-Native/helix-llm/models/$MODEL
CFG=$MODELDIR/config.json
[ -s "$CFG" ] || { echo "FAIL: missing $CFG (download gpt2-large/gpt2-xl config first)"; OK=0; }
# read dims from config.json (the SAME source the fenced importer/oracle use) and bake them into HX_*.
if [ "$OK" = "1" ]; then
  eval "$(python3 - "$CFG" <<'PY'
import json,sys
c=json.load(open(sys.argv[1]))
ne,nh=int(c["n_embd"]),int(c["n_head"])
ni=c.get("n_inner") or c.get("n_ff")
dff=int(ni) if ni else 4*ne
print(f'export HX_NL={int(c["n_layer"])} HX_D={ne} HX_HEADS={nh} HX_V={int(c.get("vocab_size",50257))} '
      f'HX_CTX={int(c.get("n_ctx",c.get("n_positions",1024)))} HX_DFF={dff}')
PY
)"
fi
echo "  model dims from config: HX_NL=${HX_NL:-?} HX_D=${HX_D:-?} HX_HEADS=${HX_HEADS:-?} HX_V=${HX_V:-?} HX_CTX=${HX_CTX:-?} HX_DFF=${HX_DFF:-?}"
# validity guards for the tiled GEMM constraints (M%64==N%64==K%8==0 at d_model/d_ff/head_dim).
if [ "$OK" = "1" ]; then
  DH=$(( HX_D / HX_HEADS ))
  for chk in "$HX_D" "$HX_DFF" "$DH"; do
    if [ $(( chk % 64 )) -ne 0 ]; then echo "  FAIL: dim $chk not a multiple of 64 (tiled GEMM constraint)"; OK=0; fi
  done
  echo "  GEMM-dim check: d_model=$HX_D d_ff=$HX_DFF head_dim=$DH all %64==0 (vocab padded ->$(( (HX_V+63)/64*64 )))"
fi

# ---- [0b] inputs: the larger .weights (fenced) + config ----
WEIGHTS_MNT=$MODELDIR/$MODEL.weights
WEIGHTS=/home/legoa/gpt2_ext4/$MODEL.weights         # ext4 mirror of the larger weight file
TOOLS=/mnt/c/Projects/Kovostov-Native/helix-llm/tools
EXMNT=/mnt/c/Projects/Kovostov-Native/helixc/examples
RTMNT=/mnt/c/Projects/Kovostov-Native/helixc/runtime
REFDIR=/mnt/c/Projects/Kovostov-Native/helix-llm/ref
[ -s "$WEIGHTS_MNT" ] || { echo "FAIL: missing $WEIGHTS_MNT (run: HX_GPT2_MODEL=$MODEL python3 $TOOLS/gpt2_import.py)"; OK=0; }

# ---- [0c] mirror bootstrap + FROZEN sources + larger weights to ext4 (idempotent) ----
echo "=== [0c] sync bootstrap + frozen sources + weights to ext4 (DrvFs-write dodge) ==="
mkdir -p "$BS" "$WORK/helixc/bootstrap" "$WORK/helixc/examples" "$WORK/helixc/runtime"
cp -r /mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/. "$BS"/
cp /mnt/c/Projects/Kovostov-Native/helixc/bootstrap/lexer.hx  "$WORK/helixc/bootstrap/"
cp /mnt/c/Projects/Kovostov-Native/helixc/bootstrap/parser.hx "$WORK/helixc/bootstrap/"
cp /mnt/c/Projects/Kovostov-Native/helixc/bootstrap/kovc.hx   "$WORK/helixc/bootstrap/"
sed -i 's#/mnt/c/Projects/Kovostov-Native/#/home/legoa/gpt2_ext4/Kovostov-Native/#g' "$BS/assemble_k1.hx"
if grep -q '/mnt/c/' "$BS/assemble_k1.hx"; then echo "  FAIL: ext4 assemble_k1.hx still has /mnt refs"; OK=0; fi
if [ "$OK" = "1" ]; then
  if [ ! -f "$WEIGHTS" ] || [ "$(stat -c%s "$WEIGHTS" 2>/dev/null || echo 0)" != "$(stat -c%s "$WEIGHTS_MNT")" ]; then
    echo "  mirroring $MODEL weight file to ext4 ($(stat -c%s "$WEIGHTS_MNT") B)..."; cp "$WEIGHTS_MNT" "$WEIGHTS"
  fi
fi
_seedsha=$(sha256sum "$BS/seed.bin" 2>/dev/null | cut -c1-8)
if [ "$_seedsha" != "$SEED_PIN" ]; then echo "  FAIL: ext4 seed.bin sha $_seedsha != pinned $SEED_PIN"; OK=0;
else echo "  ext4 seed.bin sha=$_seedsha (== pinned $SEED_PIN); weights $(stat -c%s "$WEIGHTS" 2>/dev/null) B"; fi
[ -s "$WEIGHTS" ] || { echo "  FAIL: missing ext4 weight file"; OK=0; }

# ---- [1] FRESH-MINT the PTX driver from the raw seed (same as the 124M gate) ----
echo "=== [1] mint PTX driver from raw seed (ext4) ==="
cd "$BS" || { echo "FAIL: no ext4 bootstrap dir"; OK=0; }
rm -f /tmp/asm_k1_scale.bin /tmp/newdrv_scale.bin
if [ "$OK" = "1" ] && [ -x "$BS/seed.bin" ]; then
  ( ulimit -s unlimited; timeout 600 ./seed.bin assemble_k1.hx /tmp/asm_k1_scale.bin ) >/tmp/scale_asm.log 2>&1
  if [ ! -s /tmp/asm_k1_scale.bin ]; then echo "  FAIL: seed could not compile assemble_k1.hx"; tail -4 /tmp/scale_asm.log|sed 's/^/    /'; OK=0; fi
  chmod +x /tmp/asm_k1_scale.bin 2>/dev/null
  ( ulimit -s unlimited; timeout 600 /tmp/asm_k1_scale.bin ) >/tmp/scale_concat.log 2>&1
  if [ ! -s "$BS/k1ptxdrv.hx" ]; then echo "  FAIL: concatenator produced no k1ptxdrv.hx"; tail -4 /tmp/scale_concat.log|sed 's/^/    /'; OK=0; fi
  ( ulimit -s unlimited; timeout 1200 ./seed.bin k1ptxdrv.hx /tmp/newdrv_scale.bin ) >/tmp/scale_drv.log 2>&1; ndrc=$?
  if [ "$ndrc" -ne 0 ] || [ ! -s /tmp/newdrv_scale.bin ]; then echo "  FAIL: seed->newdrv (rc=$ndrc / empty)"; tail -4 /tmp/scale_drv.log|sed 's/^/    /'; OK=0;
  else chmod +x /tmp/newdrv_scale.bin; echo "  driver (seed-minted) $(stat -c%s /tmp/newdrv_scale.bin) B  sha=$(sha256sum /tmp/newdrv_scale.bin|cut -c1-16)"; fi
else echo "  FAIL: missing ext4 seed.bin"; OK=0; fi

# ---- [2] emit the SAME forward-only kernel set as one PTX module (ZERO new kernels) ----
echo "=== [2] emit /tmp/out.ptx (the SAME forward-only kernels as the 124M MVP) ==="
KS="tiled_matmul tiled_matmul_abt gpu_softmax_causal gpu_layernorm_fwd_eps gpu_add_bias_rowbcast gpu_gelu_stable vector_add gpu_scale_rt"
: > /tmp/kernel_in.hx
nk=0
for k in $KS; do
  src="$EXMNT/${k}_kernel.hx"
  if [ -f "$src" ]; then tr -d '\r' < "$src" >> /tmp/kernel_in.hx; echo "" >> /tmp/kernel_in.hx; nk=$((nk+1));
  else echo "  MISSING kernel source: $k"; OK=0; fi
done
echo "  concatenated $nk kernel sources ($(wc -c < /tmp/kernel_in.hx) B)"
rm -f /tmp/out.ptx
if [ "$OK" = "1" ] && [ -x /tmp/newdrv_scale.bin ]; then ( ulimit -s unlimited; /tmp/newdrv_scale.bin ) >/tmp/scale_emit.log 2>&1 || true; fi
if [ -s /tmp/out.ptx ]; then
  cp /tmp/out.ptx /tmp/gpt2_scale.ptx
  nent=$(grep -c '\.entry' /tmp/gpt2_scale.ptx)
  echo "  out.ptx $(stat -c%s /tmp/gpt2_scale.ptx) B, $nent .entry kernels"
  if [ "$nent" -lt "$nk" ]; then echo "  PTX MISSING ENTRIES ($nent < $nk)"; OK=0; fi
else echo "  PTX EMIT FAIL (no /tmp/out.ptx)"; tail -4 /tmp/scale_emit.log|sed 's/^/    /'; OK=0; fi

# ---- [3] build the (unchanged, dimension-generic) forward-only launcher on ext4 ----
echo "=== [3] build gpt2_infer.c (committed, dimension-generic via HX_*) ==="
cp "$RTMNT/gpt2_infer.c" "$WORK/helixc/runtime/gpt2_infer.c"
cd "$WORK/helixc/runtime" || { echo "FAIL: no ext4 runtime dir"; OK=0; }
rm -f /tmp/gpt2_infer
gcc gpt2_infer.c -O2 -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -lm -o /tmp/gpt2_infer 2>/tmp/scale_gcc.log \
  || { echo "  GCC FAIL"; sed 's/^/    /' /tmp/scale_gcc.log; OK=0; }
[ -x /tmp/gpt2_infer ] && echo "  built /tmp/gpt2_infer" || { echo "  build produced no binary"; OK=0; }

# ---- [4] oracle references for the LARGER model (fenced numpy oracle; HX_GPT2_MODEL selects dims) ----
echo "=== [4] oracle references for $MODEL (fenced gpt2_numpy_ref.py; may be slow at scale) ==="
rm -f "$REFDIR/ref_logits_last.bin" "$REFDIR/ref_argmax.txt" "$REFDIR/ref_ids.txt" "$REFDIR/ref_gen_ids.txt" "$REFDIR/ref_gen_text.txt"
rm -f "$REFDIR/ref_block0.npy"
ORACLE_LOGITS_OK=0; ORACLE_BLOCK0_OK=0
if [ "$OK" = "1" ]; then
  # block-0 + ids are cheap (one block) -- always produce them (the fallback anchor).
  ( cd "$TOOLS" && HX_GPT2_MODEL=$MODEL python3 - "$PROMPT" <<'PY'
import sys, os, numpy as np, gpt2_numpy_ref as r
W=r.load_safetensors(os.path.join(r.WD,"model.safetensors"))
tok=r.BPE(os.path.join(r.WD,"vocab.json"),os.path.join(r.WD,"merges.txt"))
ids=tok.encode(sys.argv[1]); os.makedirs(r.REF,exist_ok=True)
r.forward(W,ids,dump=os.path.join(r.REF,"ref_block0.npy"))
open(os.path.join(r.REF,"ref_ids.txt"),"w").write(" ".join(map(str,ids))+"\n")
print("block0+ids dumped for",len(ids),"tokens")
PY
  ) >/tmp/scale_orc_b0.log 2>&1 && ORACLE_BLOCK0_OK=1 || { echo "  ORACLE block0 FAIL"; tail -6 /tmp/scale_orc_b0.log|sed 's/^/    /'; }
  [ -s "$REFDIR/ref_block0.npy" ] || ORACLE_BLOCK0_OK=0
  # full-logits + generation are the PRIMARY parity refs -- attempt them; OOM => fall back.
  if ( cd "$TOOLS" && HX_GPT2_MODEL=$MODEL python3 gpt2_numpy_ref.py dump-logits "$PROMPT" ) >/tmp/scale_orc_logits.log 2>&1 \
     && ( cd "$TOOLS" && HX_GPT2_MODEL=$MODEL python3 gpt2_numpy_ref.py dump-gen "$PROMPT" "$NGEN" ) >/tmp/scale_orc_gen.log 2>&1; then
    if [ -s "$REFDIR/ref_logits_last.bin" ] && [ -s "$REFDIR/ref_gen_ids.txt" ]; then ORACLE_LOGITS_OK=1; fi
  fi
  if [ "$ORACLE_LOGITS_OK" = "1" ]; then echo "  oracle full refs OK (argmax=$(cat "$REFDIR/ref_argmax.txt" 2>/dev/null); ids=[$(cat "$REFDIR/ref_ids.txt" 2>/dev/null)])";
  else echo "  oracle full-logits/gen UNAVAILABLE (likely OOM) -> FALLBACK to block-0 + coherence"; tail -3 /tmp/scale_orc_logits.log 2>/dev/null|sed 's/^/    /'; fi
fi

# ============================ PARITY ============================
PARITY=0; MODE="none"
if [ "$ORACLE_LOGITS_OK" = "1" ]; then
  # -------- PRIMARY: full-logits argmax-exact + token-for-token generation --------
  MODE="full-logits + token-for-token generation"
  echo "=== [5] PRIMARY parity: full-logits (argmax exact + diff bar) ==="
  rm -f /tmp/helix_logits_last.bin
  G_LOG=0
  if [ "$OK" = "1" ] && [ -x /tmp/gpt2_infer ] && [ -s /tmp/gpt2_scale.ptx ] && [ -s "$WEIGHTS" ] && [ -s "$REFDIR/ref_logits_last.bin" ]; then
    /tmp/gpt2_infer /tmp/gpt2_scale.ptx "$WEIGHTS" --logits "$REFDIR/ref_logits_last.bin" "$REFDIR/ref_argmax.txt" "$REFDIR/ref_ids.txt" >/tmp/scale_g_log.log 2>&1; grc=$?
    sed 's/^/    /' /tmp/scale_g_log.log
    if [ ! -s /tmp/helix_logits_last.bin ]; then echo "  LOGITS FAIL: no/empty dump (rc=$grc)";
    elif grep -q '^GPT2_LOGITS_PARITY_PASS' /tmp/scale_g_log.log; then echo "  fresh artifact: /tmp/helix_logits_last.bin ($(stat -c%s /tmp/helix_logits_last.bin) B)"; G_LOG=1;
    else echo "  LOGITS FAIL (no PASS line; rc=$grc)"; fi
  else echo "  LOGITS FAIL: cannot run (missing launcher/ptx/weights/ref)"; fi

  echo "=== [6] PRIMARY parity: greedy generation (N=$NGEN) token-for-token ==="
  rm -f /tmp/helix_gen_ids.txt
  G_GEN=0
  if [ "$OK" = "1" ] && [ -x /tmp/gpt2_infer ] && [ -s /tmp/gpt2_scale.ptx ] && [ -s "$WEIGHTS" ] && [ -s "$REFDIR/ref_gen_ids.txt" ]; then
    /tmp/gpt2_infer /tmp/gpt2_scale.ptx "$WEIGHTS" --generate "$NGEN" "$REFDIR/ref_ids.txt" "$REFDIR/ref_gen_ids.txt" >/tmp/scale_g_gen.log 2>&1; ggrc=$?
    sed 's/^/    /' /tmp/scale_g_gen.log
    if [ ! -s /tmp/helix_gen_ids.txt ]; then echo "  GEN FAIL: no/empty dump (rc=$ggrc)";
    elif grep -q '^GPT2_GENERATE_MATCH_PASS' /tmp/scale_g_gen.log; then
      GENTXT=$( cd "$TOOLS" && HX_GPT2_MODEL=$MODEL python3 gpt2_numpy_ref.py decode $(cat /tmp/helix_gen_ids.txt) 2>/dev/null )
      echo "  GENERATED TEXT: $(printf '%q' "$GENTXT")"; G_GEN=1;
    else echo "  GEN FAIL (no PASS line; rc=$ggrc)"; fi
  else echo "  GEN FAIL: cannot run (missing launcher/ptx/weights/ref)"; fi
  if [ "$G_LOG" = "1" ] && [ "$G_GEN" = "1" ]; then PARITY=1; fi

elif [ "$ORACLE_BLOCK0_OK" = "1" ]; then
  # -------- FALLBACK: block-0 hidden parity + a non-degenerate generation coherence check --------
  MODE="block-0 hidden parity + generation coherence (oracle full-logits OOM)"
  echo "=== [5'] FALLBACK parity: block-0 hidden (one block) ==="
  rm -f /tmp/helix_block0.bin
  G_B0=0
  if [ "$OK" = "1" ] && [ -x /tmp/gpt2_infer ] && [ -s /tmp/gpt2_scale.ptx ] && [ -s "$WEIGHTS" ] && [ -s "$REFDIR/ref_block0.npy" ]; then
    # block0 mode uses the canonical 5-id prompt baked in gpt2_infer.c; align the oracle ref to it.
    ( cd "$TOOLS" && HX_GPT2_MODEL=$MODEL python3 - <<'PY'
import os, numpy as np, gpt2_numpy_ref as r
W=r.load_safetensors(os.path.join(r.WD,"model.safetensors"))
ids=[464,3139,286,4881,318]; os.makedirs(r.REF,exist_ok=True)
r.forward(W,ids,dump=os.path.join(r.REF,"ref_block0.npy"))
PY
    ) >/tmp/scale_orc_b0c.log 2>&1
    HX_SPAD=64 /tmp/gpt2_infer /tmp/gpt2_scale.ptx "$WEIGHTS" --block0 "$REFDIR/ref_block0.npy" >/tmp/scale_b0.log 2>&1; brc=$?
    sed 's/^/    /' /tmp/scale_b0.log
    if grep -q '^GPT2_BLOCK0_PARITY_PASS' /tmp/scale_b0.log && [ -s /tmp/helix_block0.bin ]; then echo "  block-0 parity PASS"; G_B0=1;
    else echo "  BLOCK0 FAIL (no PASS line; rc=$brc)"; fi
  else echo "  BLOCK0 FAIL: cannot run"; fi

  echo "=== [6'] FALLBACK coherence: greedy generation non-degenerate (no oracle token-match) ==="
  rm -f /tmp/helix_gen_ids.txt
  G_COH=0
  if [ "$G_B0" = "1" ] && [ -s "$REFDIR/ref_ids.txt" ]; then
    /tmp/gpt2_infer /tmp/gpt2_scale.ptx "$WEIGHTS" --generate "$NGEN" "$REFDIR/ref_ids.txt" >/tmp/scale_coh.log 2>&1; crc=$?
    sed 's/^/    /' /tmp/scale_coh.log
    if grep -q '^GPT2_GENERATE_MATCH_PASS' /tmp/scale_coh.log && [ -s /tmp/helix_gen_ids.txt ]; then
      # coherence = finite logits (PASS line) AND the continuation is not a single repeated token.
      NDISTINCT=$(tr ' ' '\n' < /tmp/helix_gen_ids.txt | sort -u | wc -l)
      GENTXT=$( cd "$TOOLS" && HX_GPT2_MODEL=$MODEL python3 gpt2_numpy_ref.py decode $(cat /tmp/helix_gen_ids.txt) 2>/dev/null )
      echo "  GENERATED TEXT: $(printf '%q' "$GENTXT")  (distinct ids=$NDISTINCT)"
      if [ "$NDISTINCT" -ge 3 ]; then G_COH=1; else echo "  COHERENCE FAIL: degenerate (<3 distinct ids)"; fi
    else echo "  COHERENCE FAIL (no finite-generation PASS line; rc=$crc)"; fi
  fi
  if [ "$G_B0" = "1" ] && [ "$G_COH" = "1" ]; then PARITY=1; fi
else
  echo "  FAIL: no oracle reference available (neither full-logits nor block-0)"; OK=0
fi

# ---- verdict ----
echo "=================== VERDICT ($MODEL, mode: $MODE) ==================="
if [ "$OK" = "1" ] && [ "$PARITY" = "1" ]; then echo "GPT2_SCALE_PARITY_PASS"; else echo "GPT2_SCALE_PARITY_FAIL"; fi
echo "(done $(date -u +%H:%M:%S))"
if [ "$OK" = "1" ] && [ "$PARITY" = "1" ]; then exit 0; else exit 1; fi
