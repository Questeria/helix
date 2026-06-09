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
# Run as a FILE under WSL.
set -u
SRC=/mnt/c/Projects/Kovostov-Native
WORK=/home/legoa/gpt2_ext4/Kovostov-Native
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
sed -i 's#/mnt/c/Projects/Kovostov-Native/#/home/legoa/gpt2_ext4/Kovostov-Native/#g' "$BS/assemble_k1.hx"
_seedsha=$(sha256sum "$BS/seed.bin" 2>/dev/null | cut -c1-8)
[ "$_seedsha" = "$SEED_PIN" ] || { echo "FAIL: seed sha $_seedsha != $SEED_PIN"; exit 1; }
cd "$BS" || exit 1
rm -f /tmp/scd_a.bin /tmp/scd_d.bin /tmp/out.ptx
( ulimit -s unlimited; timeout 600 ./seed.bin assemble_k1.hx /tmp/scd_a.bin ) >/tmp/scd_asm.log 2>&1; chmod +x /tmp/scd_a.bin 2>/dev/null
( ulimit -s unlimited; timeout 600 /tmp/scd_a.bin ) >/tmp/scd_concat.log 2>&1
( ulimit -s unlimited; timeout 1200 ./seed.bin k1ptxdrv.hx /tmp/scd_d.bin ) >/tmp/scd_drv.log 2>&1; chmod +x /tmp/scd_d.bin 2>/dev/null
: > /tmp/kernel_in.hx
for k in tiled_matmul tiled_matmul_abt gpu_softmax_causal gpu_layernorm_fwd_eps gpu_add_bias_rowbcast gpu_gelu_stable vector_add gpu_scale_rt; do
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
echo
exec /tmp/gpt2_chat_server --port $PORT --root $SRC/demo \
  --ptx /tmp/gpt2_chat.ptx --weights /home/legoa/gpt2_ext4/gpt2-xl.weights \
  --worker-bin /tmp/gpt2_chat_worker --vocab "$VOCAB" --merges "$MERGES" \
  --max-ctx "$MAXCTX" --detail "$DETAIL" --oracle $SRC/helix-llm/tools --model gpt2-xl
