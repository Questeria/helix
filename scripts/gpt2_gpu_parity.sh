#!/usr/bin/env bash
# P5 gate-2 anchor: GPT-2 BLOCK-0 HIDDEN PARITY on the GPU through kovc-emitted PTX.
# Modeled on scripts/capstone_audit.sh: neutralize the ambient HX_* env, mint the PTX
# driver FRESH from the 299-byte raw-binary seed (never a cached artifact), concatenate
# the FORWARD-ONLY kernel set into /tmp/kernel_in.hx, emit /tmp/out.ptx, build the
# forked forward-only launcher helixc/runtime/gpt2_infer.c, run GPT-2 block 0 for the
# canonical prompt, and compare the post-block-0 hidden [5,768] to helix-llm/ref/
# ref_block0.npy at max-abs-rel < 1e-3. Run as a FILE under WSL. STRICTLY SERIAL GPU.
#
#   bash scripts/gpt2_gpu_parity.sh
#
# Emits GPT2_BLOCK0_PARITY_PASS or GPT2_BLOCK0_PARITY_FAIL and propagates the verdict to
# the PROCESS EXIT STATUS (fail-closed: a printed FAIL is never exit 0). Block-0 parity
# proves embedding + causal mask + eps-LN + multi-head + bias + GEMM orientation at once.
set -u
ROOT=/mnt/c/Projects/Kovostov-Native
BS=$ROOT/stage0/helixc-bootstrap
EX=$ROOT/helixc/examples
RT=$ROOT/helixc/runtime
WEIGHTS=$ROOT/helix-llm/models/gpt2/gpt2_124M.weights
REF=$ROOT/helix-llm/ref/ref_block0.npy
OK=1
echo "=================== GPT-2 GPU BLOCK-0 PARITY  $(date -u +%H:%M:%S) ==================="

# ---- [0] AMBIENT-ENV NEUTRALIZATION (mirror capstone_audit.sh) ----
# gpt2_infer.c reads HX_NL/HX_D/HX_HEADS/HX_V/HX_CTX/HX_DFF/HX_SPAD; a stray value would
# silently change the dims. Unset every HX_* var so the gate is reproducible, then assert none.
for v in $(env | sed -n 's/^\(HX_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
_leftover=$(env | sed -n 's/^\(HX_[A-Za-z0-9_]*\)=.*/\1/p' | tr '\n' ' ')
if [ -n "$_leftover" ]; then echo "PARITY FAIL: residual HX_* env after neutralization: $_leftover"; OK=0; fi
echo "  ambient HX_* neutralized (GPT-2 124M defaults baked in gpt2_infer.c); residual='${_leftover}'"

# ---- [0b] inputs present ----
[ -s "$WEIGHTS" ] || { echo "PARITY FAIL: missing weight file $WEIGHTS (run the P1 importer)"; OK=0; }
[ -s "$REF" ]     || { echo "PARITY FAIL: missing parity target $REF (run gpt2_numpy_ref.py)"; OK=0; }
echo "  weights $( [ -s "$WEIGHTS" ] && stat -c%s "$WEIGHTS" || echo MISSING ) B ; ref $( [ -s "$REF" ] && stat -c%s "$REF" || echo MISSING ) B"

# ---- [1] mint the PTX driver FRESH from the raw-binary seed ----
echo "=== [1] mint PTX driver from raw seed (seed.bin + k1ptxdrv.hx -> /tmp/newdrv.bin) ==="
cd "$BS" || { echo "PARITY FAIL: no bootstrap dir"; OK=0; }
DRV=/tmp/newdrv.bin
rm -f "$DRV"   # rm-before: never reuse a STALE driver if the mint fails
if [ -x "$BS/seed.bin" ] && [ -f "$BS/k1ptxdrv.hx" ]; then
  bash assemble_k1.sh >/dev/null 2>&1 || true
  ( ulimit -s unlimited; timeout 1200 ./seed.bin k1ptxdrv.hx "$DRV" ) >/tmp/gp_drv.log 2>&1; ndrc=$?
  if [ "$ndrc" -ne 0 ] || [ ! -s "$DRV" ]; then echo "  DRIVER FAIL (seed->newdrv rc=$ndrc / empty)"; tail -4 /tmp/gp_drv.log | sed 's/^/    /'; OK=0;
  else chmod +x "$DRV"; echo "  driver (seed-minted) $(stat -c%s "$DRV") B  sha=$(sha256sum "$DRV" | cut -c1-16)"; fi
else echo "  DRIVER FAIL: missing seed.bin or k1ptxdrv.hx"; OK=0; fi

# ---- [2] emit the FORWARD-ONLY kernel set as one PTX module ----
echo "=== [2] emit /tmp/out.ptx (forward-only kernels via the seed-minted driver) ==="
# GEMMs (tiled OPT, the only ones valid at N>1024) + the 3 GPT-2 gap kernels + gelu/residual/scale.
# gpu_gelu_stable (overflow-safe gelu_new): the committed gpu_gelu's direct e^(2z) NaNs at GPT-2's
# ~+/-12 c_fc pre-activations; the stable variant uses an exp-arg-<=0 tanh identity (unit-gated via
# cuda_launch `gelu_big`). The other 7 kernels are committed + GPU-unit-gated.
KS="tiled_matmul tiled_matmul_abt gpu_softmax_causal gpu_layernorm_fwd_eps gpu_add_bias_rowbcast gpu_gelu_stable vector_add gpu_scale_rt"
: > /tmp/kernel_in.hx
nk=0
for k in $KS; do
  f=$EX/${k}_kernel.hx
  if [ -f "$f" ]; then tr -d '\r' < "$f" >> /tmp/kernel_in.hx; echo "" >> /tmp/kernel_in.hx; nk=$((nk+1));
  else echo "  MISSING kernel source: $k"; OK=0; fi
done
echo "  concatenated $nk kernel sources"
rm -f /tmp/out.ptx
if [ -x "$DRV" ]; then "$DRV" >/tmp/gp_emit.log 2>&1 || true; fi
if [ -s /tmp/out.ptx ]; then
  cp /tmp/out.ptx /tmp/gpt2_combined.ptx
  nent=$(grep -c '\.entry' /tmp/gpt2_combined.ptx)
  echo "  out.ptx $(stat -c%s /tmp/gpt2_combined.ptx) B, $nent .entry kernels"
  if [ "$nent" -lt "$nk" ]; then echo "  PTX MISSING ENTRIES ($nent < $nk)"; OK=0; fi
else echo "  PTX EMIT FAIL (driver produced no /tmp/out.ptx)"; OK=0; fi

# ---- [3] build the forked forward-only launcher ----
echo "=== [3] build gpt2_infer.c (forward-only fork of train_transformer.c) ==="
cd "$RT" || { echo "PARITY FAIL: no runtime dir"; OK=0; }
rm -f /tmp/gpt2_infer
gcc gpt2_infer.c -O2 -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -lm -o /tmp/gpt2_infer 2>/tmp/gp_gcc.log \
  || { echo "  GCC FAIL"; sed 's/^/    /' /tmp/gp_gcc.log; OK=0; }
[ -x /tmp/gpt2_infer ] && echo "  built /tmp/gpt2_infer" || { echo "  build produced no binary"; OK=0; }

# ---- [4] run BLOCK-0 parity (STALE-ARTIFACT GUARD: rm the dump before, require fresh after) ----
echo "=== [4] run GPT-2 block-0 parity (canonical prompt, max-abs-rel < 1e-3) ==="
rm -f /tmp/helix_block0.bin
if [ -x /tmp/gpt2_infer ] && [ -s /tmp/gpt2_combined.ptx ] && [ -s "$WEIGHTS" ] && [ -s "$REF" ]; then
  /tmp/gpt2_infer /tmp/gpt2_combined.ptx "$WEIGHTS" --block0 "$REF" > /tmp/gp_run.log 2>&1; prc=$?
  sed 's/^/    /' /tmp/gp_run.log
  if [ ! -s /tmp/helix_block0.bin ]; then echo "  PARITY FAIL: run left no/empty /tmp/helix_block0.bin (rc=$prc)"; OK=0;
  else echo "  fresh artifact: /tmp/helix_block0.bin ($(stat -c%s /tmp/helix_block0.bin) B, written this run)"; fi
  if ! grep -q '^GPT2_BLOCK0_PARITY_PASS' /tmp/gp_run.log; then echo "  PARITY FAIL (no PASS line; rc=$prc)"; OK=0; fi
else echo "  PARITY FAIL: cannot run (missing launcher / ptx / weights / ref)"; OK=0; fi

echo "=================== VERDICT ==================="
if [ "$OK" = "1" ]; then echo "GPT2_BLOCK0_PARITY_PASS"; else echo "GPT2_BLOCK0_PARITY_FAIL"; fi
echo "(done $(date -u +%H:%M:%S))"
# FAIL-CLOSED: propagate the verdict to the process exit status.
if [ "$OK" = "1" ]; then exit 0; else exit 1; fi
