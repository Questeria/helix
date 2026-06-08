#!/usr/bin/env bash
# P5 GATE 3 + GATE 4: GPT-2 124M FULL-LOGITS PARITY + GREEDY GENERATION on the GPU through
# kovc-emitted PTX. Extends the committed BLOCK-0 gate (scripts/gpt2_gpu_parity.sh) to the full
# 12-layer model + the tied LM head + autoregressive generation.
#
#   bash scripts/gpt2_gpu_mvp.sh
#
# Emits, fail-closed (printed FAIL is never exit 0):
#   GPT2_LOGITS_PARITY_PASS   -- 12 layers + ln_f + tied head; LAST real-token argmax == oracle EXACTLY
#                                AND max-abs logit diff under a documented diagnostic bar.
#   GPT2_GENERATE_MATCH_PASS  -- greedy N-token continuation matches the oracle TOKEN-FOR-TOKEN; coherent.
# and propagates the combined verdict to the PROCESS EXIT STATUS.
#
# PROVENANCE: the PTX driver is minted FRESH from the 299-byte raw-binary seed (seed.bin, pinned
# sha 9837db12...) every run -- never a cached artifact. To dodge the WSL DrvFs byte-by-byte
# write_file_to_arena pathology (the Helix concatenator does ~4.2M 1-byte write syscalls; on /mnt/c
# DrvFs that is 20+ min, on ext4 it is ~4 s), the bootstrap tree + the FROZEN lexer/parser/kovc
# sources are mirrored BIT-IDENTICALLY to ext4 and the assembly/mint run there. The frozen sources
# are sha-verified identical, so the minted driver + PTX are byte-identical to the DrvFs path -- only
# faster. STRICTLY SERIAL GPU (one build/launch at a time). Run as a FILE under WSL.
set -u

# ---- paths (NEVER assign a /mnt path to a variable -- MSYS empties it; use literals) ----
WORK=/home/legoa/gpt2_ext4/Kovostov-Native           # ext4 mirror root (fast writes)
BS=$WORK/stage0/helixc-bootstrap                     # ext4 bootstrap
EXMNT=/mnt/c/Projects/Kovostov-Native/helixc/examples         # kernel sources (read-only, DrvFs ok)
RTMNT=/mnt/c/Projects/Kovostov-Native/helixc/runtime          # gpt2_infer.c source (read-only)
TOOLS=/mnt/c/Projects/Kovostov-Native/helix-llm/tools         # fenced oracle (read-only)
WEIGHTS_MNT=/mnt/c/Projects/Kovostov-Native/helix-llm/models/gpt2/gpt2_124M.weights
WEIGHTS=/home/legoa/gpt2_ext4/gpt2_124M.weights      # ext4 mirror of the 498 MB weight file
REFDIR=/mnt/c/Projects/Kovostov-Native/helix-llm/ref          # oracle dumps land here
SEED_PIN=9837db12                                    # pinned raw-binary seed sha prefix
NGEN=20
PROMPT="The capital of France is"
OK=1
echo "=================== GPT-2 GPU FULL-LOGITS + GENERATION  $(date -u +%H:%M:%S) ==================="

# ---- [0] ambient HX_* neutralization (mirror gpt2_gpu_parity.sh) ----
for v in $(env | sed -n 's/^\(HX_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
_left=$(env | sed -n 's/^\(HX_[A-Za-z0-9_]*\)=.*/\1/p' | tr '\n' ' ')
if [ -n "$_left" ]; then echo "FAIL: residual HX_* env: $_left"; OK=0; fi
echo "  ambient HX_* neutralized (GPT-2 124M defaults baked in gpt2_infer.c); residual='${_left}'"

# ---- [0b] mirror bootstrap + FROZEN sources + weights to ext4 (idempotent) ----
echo "=== [0b] sync bootstrap + frozen sources + weights to ext4 (DrvFs-write dodge) ==="
mkdir -p "$BS" "$WORK/helixc/bootstrap" "$WORK/helixc/examples" "$WORK/helixc/runtime"
cp -r /mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap/. "$BS"/
cp /mnt/c/Projects/Kovostov-Native/helixc/bootstrap/lexer.hx  "$WORK/helixc/bootstrap/"
cp /mnt/c/Projects/Kovostov-Native/helixc/bootstrap/parser.hx "$WORK/helixc/bootstrap/"
cp /mnt/c/Projects/Kovostov-Native/helixc/bootstrap/kovc.hx   "$WORK/helixc/bootstrap/"
# rewrite the baked absolute read/write paths in the ext4 assemble_k1.hx -> ext4 mirror
sed -i 's#/mnt/c/Projects/Kovostov-Native/#/home/legoa/gpt2_ext4/Kovostov-Native/#g' "$BS/assemble_k1.hx"
if grep -q '/mnt/c/' "$BS/assemble_k1.hx"; then echo "  FAIL: ext4 assemble_k1.hx still has /mnt refs"; OK=0; fi
# weight-file ext4 mirror (copy only if missing or size-changed)
if [ ! -f "$WEIGHTS" ] || [ "$(stat -c%s "$WEIGHTS" 2>/dev/null || echo 0)" != "$(stat -c%s "$WEIGHTS_MNT")" ]; then
  echo "  mirroring weight file to ext4..."; cp "$WEIGHTS_MNT" "$WEIGHTS"
fi
# seed-sha provenance check (the trust anchor) -- on the ext4 copy
_seedsha=$(sha256sum "$BS/seed.bin" | cut -c1-8)
if [ "$_seedsha" != "$SEED_PIN" ]; then echo "  FAIL: ext4 seed.bin sha $_seedsha != pinned $SEED_PIN (mirror corrupt)"; OK=0;
else echo "  ext4 seed.bin sha=$_seedsha (== pinned $SEED_PIN); weights $(stat -c%s "$WEIGHTS") B"; fi
[ -s "$WEIGHTS" ] || { echo "  FAIL: missing weight file"; OK=0; }

# ---- [1] FRESH-MINT the PTX driver from the raw seed (assemble -> seed-compile -> run -> mint) ----
echo "=== [1] mint PTX driver from raw seed (ext4) ==="
cd "$BS" || { echo "FAIL: no ext4 bootstrap dir"; OK=0; }
rm -f /tmp/asm_k1_mvp.bin /tmp/newdrv_mvp.bin
if [ "$OK" = "1" ] && [ -x "$BS/seed.bin" ]; then
  # (a) seed compiles the concatenator
  ( ulimit -s unlimited; timeout 600 ./seed.bin assemble_k1.hx /tmp/asm_k1_mvp.bin ) >/tmp/mvp_asm.log 2>&1
  if [ ! -s /tmp/asm_k1_mvp.bin ]; then echo "  FAIL: seed could not compile assemble_k1.hx"; tail -4 /tmp/mvp_asm.log|sed 's/^/    /'; OK=0; fi
  chmod +x /tmp/asm_k1_mvp.bin 2>/dev/null
  # (b) run the concatenator -> k1ptxdrv.hx (fast on ext4)
  ( ulimit -s unlimited; timeout 600 /tmp/asm_k1_mvp.bin ) >/tmp/mvp_concat.log 2>&1
  if [ ! -s "$BS/k1ptxdrv.hx" ]; then echo "  FAIL: concatenator produced no k1ptxdrv.hx"; tail -4 /tmp/mvp_concat.log|sed 's/^/    /'; OK=0; fi
  # (c) seed mints the PTX driver from k1ptxdrv.hx
  ( ulimit -s unlimited; timeout 1200 ./seed.bin k1ptxdrv.hx /tmp/newdrv_mvp.bin ) >/tmp/mvp_drv.log 2>&1; ndrc=$?
  if [ "$ndrc" -ne 0 ] || [ ! -s /tmp/newdrv_mvp.bin ]; then echo "  FAIL: seed->newdrv (rc=$ndrc / empty)"; tail -4 /tmp/mvp_drv.log|sed 's/^/    /'; OK=0;
  else chmod +x /tmp/newdrv_mvp.bin; echo "  driver (seed-minted) $(stat -c%s /tmp/newdrv_mvp.bin) B  sha=$(sha256sum /tmp/newdrv_mvp.bin|cut -c1-16)"; fi
else echo "  FAIL: missing ext4 seed.bin"; OK=0; fi

# ---- [2] emit the FORWARD-ONLY kernel set as one PTX module ----
echo "=== [2] emit /tmp/out.ptx (forward-only kernels via the seed-minted driver) ==="
: > /tmp/kernel_in.hx
nk=0
for k in tiled_matmul tiled_matmul_abt gpu_softmax_causal gpu_layernorm_fwd_eps gpu_add_bias_rowbcast gpu_gelu_stable vector_add gpu_scale_rt; do
  src="$EXMNT/${k}_kernel.hx"
  if [ -f "$src" ]; then tr -d '\r' < "$src" >> /tmp/kernel_in.hx; echo "" >> /tmp/kernel_in.hx; nk=$((nk+1));
  else echo "  MISSING kernel source: $k"; OK=0; fi
done
echo "  concatenated $nk kernel sources ($(wc -c < /tmp/kernel_in.hx) B)"
rm -f /tmp/out.ptx
if [ "$OK" = "1" ] && [ -x /tmp/newdrv_mvp.bin ]; then ( ulimit -s unlimited; /tmp/newdrv_mvp.bin ) >/tmp/mvp_emit.log 2>&1 || true; fi
if [ -s /tmp/out.ptx ]; then
  cp /tmp/out.ptx /tmp/gpt2_mvp.ptx
  nent=$(grep -c '\.entry' /tmp/gpt2_mvp.ptx)
  echo "  out.ptx $(stat -c%s /tmp/gpt2_mvp.ptx) B, $nent .entry kernels"
  if [ "$nent" -lt "$nk" ]; then echo "  PTX MISSING ENTRIES ($nent < $nk)"; OK=0; fi
else echo "  PTX EMIT FAIL (no /tmp/out.ptx)"; tail -4 /tmp/mvp_emit.log|sed 's/^/    /'; OK=0; fi

# ---- [3] build the forward-only launcher on ext4 ----
echo "=== [3] build gpt2_infer.c ==="
cp "$RTMNT/gpt2_infer.c" "$WORK/helixc/runtime/gpt2_infer.c"
cd "$WORK/helixc/runtime" || { echo "FAIL: no ext4 runtime dir"; OK=0; }
rm -f /tmp/gpt2_infer
gcc gpt2_infer.c -O2 -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -lm -o /tmp/gpt2_infer 2>/tmp/mvp_gcc.log \
  || { echo "  GCC FAIL"; sed 's/^/    /' /tmp/mvp_gcc.log; OK=0; }
[ -x /tmp/gpt2_infer ] && echo "  built /tmp/gpt2_infer" || { echo "  build produced no binary"; OK=0; }

# ---- [4] generate the oracle references (fenced numpy oracle; dumps under helix-llm/ref) ----
echo "=== [4] oracle references (fenced gpt2_numpy_ref.py dump modes) ==="
rm -f "$REFDIR/ref_logits_last.bin" "$REFDIR/ref_argmax.txt" "$REFDIR/ref_ids.txt" "$REFDIR/ref_gen_ids.txt" "$REFDIR/ref_gen_text.txt"
if [ "$OK" = "1" ]; then
  ( cd "$TOOLS" && python3 gpt2_numpy_ref.py dump-logits "$PROMPT" ) >/tmp/mvp_orc_logits.log 2>&1 || { echo "  ORACLE dump-logits FAIL"; tail -5 /tmp/mvp_orc_logits.log|sed 's/^/    /'; OK=0; }
  ( cd "$TOOLS" && python3 gpt2_numpy_ref.py dump-gen "$PROMPT" "$NGEN" ) >/tmp/mvp_orc_gen.log 2>&1 || { echo "  ORACLE dump-gen FAIL"; tail -5 /tmp/mvp_orc_gen.log|sed 's/^/    /'; OK=0; }
  for f in ref_logits_last.bin ref_argmax.txt ref_ids.txt ref_gen_ids.txt; do
    [ -s "$REFDIR/$f" ] || { echo "  FAIL: oracle did not produce $f"; OK=0; }
  done
  [ -s "$REFDIR/ref_argmax.txt" ] && echo "  oracle argmax=$(cat "$REFDIR/ref_argmax.txt")  ids=[$(cat "$REFDIR/ref_ids.txt")]"
fi

# ---- [5] GATE 3: full-logits parity (argmax EXACT + logit-diff bar) ----
echo "=== [5] GATE 3: full-logits parity (12 layers + ln_f + tied head) ==="
rm -f /tmp/helix_logits_last.bin
G3=0
if [ "$OK" = "1" ] && [ -x /tmp/gpt2_infer ] && [ -s /tmp/gpt2_mvp.ptx ] && [ -s "$WEIGHTS" ] && [ -s "$REFDIR/ref_logits_last.bin" ]; then
  /tmp/gpt2_infer /tmp/gpt2_mvp.ptx "$WEIGHTS" --logits "$REFDIR/ref_logits_last.bin" "$REFDIR/ref_argmax.txt" "$REFDIR/ref_ids.txt" >/tmp/mvp_g3.log 2>&1; g3rc=$?
  sed 's/^/    /' /tmp/mvp_g3.log
  if [ ! -s /tmp/helix_logits_last.bin ]; then echo "  GATE3 FAIL: no/empty /tmp/helix_logits_last.bin (rc=$g3rc)";
  elif grep -q '^GPT2_LOGITS_PARITY_PASS' /tmp/mvp_g3.log; then echo "  fresh artifact: /tmp/helix_logits_last.bin ($(stat -c%s /tmp/helix_logits_last.bin) B)"; G3=1;
  else echo "  GATE3 FAIL (no PASS line; rc=$g3rc)"; fi
else echo "  GATE3 FAIL: cannot run (missing launcher/ptx/weights/ref)"; fi
[ "$G3" = "1" ] || OK=0

# ---- [6] GATE 4: greedy generation token-for-token match + coherent text ----
echo "=== [6] GATE 4: greedy generation (N=$NGEN) token-for-token vs oracle ==="
rm -f /tmp/helix_gen_ids.txt
G4=0
if [ "$OK" = "1" ] && [ -x /tmp/gpt2_infer ] && [ -s /tmp/gpt2_mvp.ptx ] && [ -s "$WEIGHTS" ] && [ -s "$REFDIR/ref_gen_ids.txt" ]; then
  /tmp/gpt2_infer /tmp/gpt2_mvp.ptx "$WEIGHTS" --generate "$NGEN" "$REFDIR/ref_ids.txt" "$REFDIR/ref_gen_ids.txt" >/tmp/mvp_g4.log 2>&1; g4rc=$?
  sed 's/^/    /' /tmp/mvp_g4.log
  if [ ! -s /tmp/helix_gen_ids.txt ]; then echo "  GATE4 FAIL: no/empty /tmp/helix_gen_ids.txt (rc=$g4rc)";
  elif grep -q '^GPT2_GENERATE_MATCH_PASS' /tmp/mvp_g4.log; then
    # decode the helix-produced ids to text via the fenced oracle BPE (host glue rendering only)
    GENTXT=$( cd "$TOOLS" && python3 gpt2_numpy_ref.py decode $(cat /tmp/helix_gen_ids.txt) 2>/dev/null )
    echo "  GENERATED TEXT: $(printf '%q' "$GENTXT")"
    G4=1;
  else echo "  GATE4 FAIL (no PASS line; rc=$g4rc)"; fi
else echo "  GATE4 FAIL: cannot run (missing launcher/ptx/weights/ref)"; fi
[ "$G4" = "1" ] || OK=0

# ---- verdict ----
echo "=================== VERDICT ==================="
if [ "$G3" = "1" ]; then echo "GPT2_LOGITS_PARITY_PASS"; else echo "GPT2_LOGITS_PARITY_FAIL"; fi
if [ "$G4" = "1" ]; then echo "GPT2_GENERATE_MATCH_PASS"; else echo "GPT2_GENERATE_MATCH_FAIL"; fi
echo "(done $(date -u +%H:%M:%S))"
if [ "$OK" = "1" ]; then exit 0; else exit 1; fi
