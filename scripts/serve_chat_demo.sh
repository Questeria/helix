#!/usr/bin/env bash
# serve_chat_demo.sh -- LAUNCH the live GPT-2-XL-on-Helix chat demo.
#
#   MSYS_NO_PATHCONV=1 bash scripts/serve_chat_demo.sh
#   then open:  http://127.0.0.1:8848/?source=sse
#
# Mints the XL PTX fresh from the 299-byte raw seed (9837db12), builds the persistent --serve
# worker (gpt2_infer.c + gpt2_tok.c) and the NO-Python C HTTP+SSE server (gpt2_serve_http.c),
# then starts the server bound to 127.0.0.1 with the XL weights resident. STRICTLY SERIAL GPU.
# Runs in the foreground; Ctrl-C to stop (the server reaps the worker child).
#
# Env knobs: PORT (default 8848), DETAIL (op|layer, default op), MAXCTX (default 320).
#   Portability (run on a machine other than the author's): override any of
#     HELIX_SRC         (default: auto-detected from script location, else /mnt/c/Projects/Kovostov-Native) -- the committed repo
#     HELIX_WORK        (default: $HOME/gpt2_ext4/Kovostov-Native)    -- the fast ext4 build mirror
#     HELIX_XL_WEIGHTS  (default: $HOME/gpt2_ext4/gpt2-xl.weights)    -- the XL flat .weights file
#   See docs/HELIX_GPT2_DEMO_RUNBOOK.md §0 for how to PRODUCE gpt2-xl.weights (HuggingFace + gpt2_pack.c).
# Run as a FILE under WSL.
set -u
SRC="${HELIX_SRC:-}"; if [ -z "$SRC" ]; then _sd="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; if [ -n "$_sd" ] && [ -f "$_sd/scripts/reproduce_trust.sh" ]; then SRC="$_sd"; else SRC="/mnt/c/Projects/Kovostov-Native"; fi; fi
WORK="${HELIX_WORK:-$HOME/gpt2_ext4/Kovostov-Native}"
XL_WEIGHTS="${HELIX_XL_WEIGHTS:-$HOME/gpt2_ext4/gpt2-xl.weights}"
BS=$WORK/stage0/helixc-bootstrap
SEED_PIN=9837db12
PORT="${PORT:-8848}"
DETAIL="${DETAIL:-op}"
MAXCTX="${MAXCTX:-320}"
export HX_NL=48 HX_D=1600 HX_HEADS=25 HX_V=50257 HX_CTX=1024 HX_DFF=6400

echo "=== [1/3] mint XL PTX from the raw seed ($SEED_PIN) ==="
mkdir -p "$BS" "$WORK/helixc/bootstrap" "$WORK/helixc/runtime"
cp -r $SRC/stage0/helixc-bootstrap/. "$BS"/
cp $SRC/helixc/bootstrap/lexer.hx $SRC/helixc/bootstrap/parser.hx $SRC/helixc/bootstrap/kovc.hx "$WORK/helixc/bootstrap/"
sed -i "s#/mnt/c/Projects/Kovostov-Native/#$WORK/#g" "$BS/assemble_k1.hx"
_seedsha=$(sha256sum "$BS/seed.bin" 2>/dev/null | cut -c1-8)
[ "$_seedsha" = "$SEED_PIN" ] || { echo "FAIL: seed sha $_seedsha != $SEED_PIN"; exit 1; }
cd "$BS" || exit 1
rm -f /tmp/scd_a.bin /tmp/scd_d.bin /tmp/out.ptx
( ulimit -s unlimited; timeout 600 ./seed.bin assemble_k1.hx /tmp/scd_a.bin ) >/tmp/scd_asm.log 2>&1; chmod +x /tmp/scd_a.bin 2>/dev/null
( ulimit -s unlimited; timeout 600 /tmp/scd_a.bin ) >/tmp/scd_concat.log 2>&1
( ulimit -s unlimited; timeout 1200 ./seed.bin k1ptxdrv.hx /tmp/scd_d.bin ) >/tmp/scd_drv.log 2>&1; chmod +x /tmp/scd_d.bin 2>/dev/null
: > /tmp/kernel_in.hx
# 8 GPT-2 kernels + the 3 G-L0-gated llama kernels (rmsnorm/rope/silu_mul): ONE 11-kernel
# module serves both architectures (each worker looks up only the entries it needs; the
# llama worker self-configures from the v2 weight header, overriding the HX_* XL env).
for k in tiled_matmul tiled_matmul_abt gpu_softmax_causal gpu_layernorm_fwd_eps gpu_add_bias_rowbcast gpu_gelu_stable vector_add gpu_scale_rt gpu_rmsnorm_fwd_eps gpu_rope_rot gpu_silu_mul; do
  tr -d '\r' < $SRC/helixc/examples/${k}_kernel.hx >> /tmp/kernel_in.hx; echo "" >> /tmp/kernel_in.hx
done
( ulimit -s unlimited; /tmp/scd_d.bin ) >/tmp/scd_emit.log 2>&1 || true
[ -s /tmp/out.ptx ] && cp /tmp/out.ptx /tmp/gpt2_chat.ptx && echo "  PTX $(stat -c%s /tmp/gpt2_chat.ptx) B, $(grep -c '\.entry' /tmp/gpt2_chat.ptx) kernels" || { echo "FAIL: no PTX"; exit 1; }

echo "=== [2/3] build the --serve worker + the C HTTP+SSE server ==="
cp $SRC/helixc/runtime/gpt2_infer.c $SRC/helixc/runtime/gpt2_tok.c $SRC/helixc/runtime/gpt2_serve_http.c $SRC/helixc/runtime/gpt2_unicode_ranges.inc "$WORK/helixc/runtime/"
cd "$WORK/helixc/runtime" || exit 1
gcc -DGPT2_SERVE -DGPT2_TOK_LIB gpt2_infer.c gpt2_tok.c -O2 -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -lm -o /tmp/gpt2_chat_worker 2>/tmp/scd_wgcc.log || { echo "FAIL worker build"; cat /tmp/scd_wgcc.log; exit 1; }
gcc gpt2_serve_http.c -O2 -lpthread -o /tmp/gpt2_chat_server 2>/tmp/scd_sgcc.log || { echo "FAIL server build"; cat /tmp/scd_sgcc.log; exit 1; }
echo "  built worker + server"

echo "=== [3/3] start the server (XL weights load on first readiness; ~5-60s) ==="
VOCAB=$SRC/helix-llm/models/gpt2-xl/vocab.json
MERGES=$SRC/helix-llm/models/gpt2-xl/merges.txt
echo
echo "  >>> Once it prints 'listening', open:  http://127.0.0.1:$PORT/?source=sse   <<<"
echo "      (the page shows a LIVE indicator + hides the PREVIEW banner when the worker is ready)"
echo "      NOTE: live XL ~= 10 s/token (~3 min for 20 tokens; measured 195.5 s / 20 tok, gated run)."
echo "            By design -- the demo sells verifiability, not speed. Use gpt2_gpu_mvp.sh (124M)"
echo "            when you need a snappy live generation."
echo
# ADDITIVE second model (the modern-model leg): when the SmolLM2 pack exists, serve it
# alongside XL -- two persistent workers, ONE shared GPU mutex (generations strictly serial).
SM_WTS="${HELIX_SM_WEIGHTS:-$SRC/helix-llm/models/smollm2-135m/smollm2-135m.weights}"
SM_DIR="$SRC/helix-llm/models/smollm2-135m"
SM_ARGS=""
if [ -s "$SM_WTS" ] && [ -s "$SM_DIR/vocab.json" ] && [ -s "$SM_DIR/merges.txt" ]; then
  SM_ARGS="--model2 smollm2-135m --ptx2 /tmp/gpt2_chat.ptx --weights2 $SM_WTS --vocab2 $SM_DIR/vocab.json --merges2 $SM_DIR/merges.txt"
  echo "  second model: smollm2-135m (llama-arch) ENABLED"
else
  echo "  second model: smollm2-135m weights not found -- serving XL only"
fi
exec /tmp/gpt2_chat_server --port $PORT --root $SRC/demo \
  --ptx /tmp/gpt2_chat.ptx --weights "$XL_WEIGHTS" \
  --worker-bin /tmp/gpt2_chat_worker --vocab "$VOCAB" --merges "$MERGES" \
  --max-ctx "$MAXCTX" --detail "$DETAIL" --oracle $SRC/helix-llm/tools --model gpt2-xl $SM_ARGS
