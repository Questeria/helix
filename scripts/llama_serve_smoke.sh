#!/usr/bin/env bash
# llama_serve_smoke.sh -- dual-model serve gate: the model switcher end-to-end.
# Run as a FILE under WSL (GPU required; STRICTLY SERIAL -- nothing else on the GPU):
#   wsl.exe bash -lc "tr -d '\r' < /mnt/c/Projects/Kovostov-Native/scripts/llama_serve_smoke.sh > /tmp/lss.sh && bash /tmp/lss.sh"
#
# FAIL-CLOSED legs:
#   [1] mint the 11-kernel PTX + build worker/server from the working tree
#   [2] spawn the server with BOTH models (XL + smollm2-135m); both workers reach READY
#   [3] /api/health advertises models[] with both entries ready
#   [4] /api/generate {model:smollm2-135m} streams SSE; served gen ids == the oracle's
#       greedy ids TOKEN-FOR-TOKEN (the same llama_ref_gen_ids.txt the model gate used)
#   [5] unknown model -> 404 {"error":"unknown model"}
#   [6] single-flight ACROSS models: a request DURING the smollm2 generation -> 409 busy
#   [7] INSTRUCT leg: templated ChatML prompt to smollm2-360m-instruct; served ids == chat
#       oracle refs TOKEN-FOR-TOKEN (C-side special-token tokenize parity + stop-at-im_end)
# Prints LLAMA_SERVE_SMOKE_PASS / _FAIL. Kills the server (by PID) on exit.
set -u
ROOT="${HELIX_SRC:-}"; if [ -z "$ROOT" ]; then ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; fi
[ -d "$ROOT/helixc/examples" ] || ROOT="/mnt/c/Projects/Kovostov-Native"
WORK="${HELIX_WORK:-$HOME/gpt2_ext4/Kovostov-Native}"
XL_WEIGHTS="${HELIX_XL_WEIGHTS:-$HOME/gpt2_ext4/gpt2-xl.weights}"
SM_DIR="$ROOT/helix-llm/models/smollm2-135m"
SM_WTS="$SM_DIR/smollm2-135m.weights"
REFD="$ROOT/helix-llm/ref"
SI_DIR="$ROOT/helix-llm/models/smollm2-360m-instruct"
SI_WTS="$SI_DIR/smollm2-360m-instruct.weights"
TOOLS="$ROOT/helix-llm/tools"
PORT="${PORT:-8851}"
BS="$WORK/stage0/helixc-bootstrap"
RC=0
T0=$(date +%s)

echo "=================== LLAMA DUAL-MODEL SERVE SMOKE  $(date -u +%H:%M:%S) ==================="
[ -s "$SM_WTS" ] || { echo "FATAL: no $SM_WTS"; echo "LLAMA_SERVE_SMOKE_FAIL"; exit 9; }
echo "=== [0] regenerate oracle refs PER MODEL (ref/ filenames are shared) ==="
( cd "$TOOLS" && LLAMA_MODEL_DIR="$SM_DIR" LLAMA_CHAT=0 python3 llama_numpy_ref.py dump-gen "The capital of France is" 20 ) >/tmp/lss_ref_base.log 2>&1 || { echo "FATAL base-ref dump"; tail -3 /tmp/lss_ref_base.log; echo "LLAMA_SERVE_SMOKE_FAIL"; exit 9; }
cp "$REFD/llama_ref_gen_ids.txt" /tmp/lss_ref_base_ids.txt
if [ -s "$SI_WTS" ]; then
  ( cd "$TOOLS" && LLAMA_MODEL_DIR="$SI_DIR" LLAMA_CHAT=1 python3 llama_numpy_ref.py dump-gen "What is the capital of France?" 40 ) >/tmp/lss_ref_chat.log 2>&1 || { echo "FATAL chat-ref dump"; tail -3 /tmp/lss_ref_chat.log; echo "LLAMA_SERVE_SMOKE_FAIL"; exit 9; }
  cp "$REFD/llama_ref_gen_ids.txt" /tmp/lss_ref_chat_ids.txt
fi
[ -s "$XL_WEIGHTS" ] || { echo "FATAL: no XL weights"; echo "LLAMA_SERVE_SMOKE_FAIL"; exit 9; }

echo "=== [1] PTX + binaries (working tree) ==="
if [ ! -s /tmp/llama_model.ptx ]; then
  echo "  no minted PTX in /tmp -- minting from the raw seed (self-sufficient gate)"
  BS_W="$WORK/stage0/helixc-bootstrap"
  DRV="${LLAMA_DRV:-$HOME/gpt2_ext4/llama_kovc_drv.bin}"   # ext4: survives WSL /tmp resets
  if [ ! -x "$DRV" ]; then
    mkdir -p "$BS_W" "$WORK/helixc/bootstrap"
    cp -r "$ROOT/stage0/helixc-bootstrap/." "$BS_W"/
    cp "$ROOT/helixc/bootstrap/lexer.hx" "$ROOT/helixc/bootstrap/parser.hx" "$ROOT/helixc/bootstrap/kovc.hx" "$WORK/helixc/bootstrap/"
    sed -i "s#/mnt/c/Projects/Kovostov-Native/#$WORK/#g" "$BS_W/assemble_k1.hx"
    _seedsha=$(sha256sum "$BS_W/seed.bin" 2>/dev/null | cut -c1-8)
    [ "$_seedsha" = "9837db12" ] || { echo "FATAL: ext4 seed sha $_seedsha != 9837db12"; echo "LLAMA_SERVE_SMOKE_FAIL"; exit 8; }
    cd "$BS_W" || exit 8
    rm -f /tmp/asm_k1_ls.bin
    ( ulimit -s unlimited; timeout 600 ./seed.bin assemble_k1.hx /tmp/asm_k1_ls.bin ) || { echo "FATAL assemble_k1"; echo "LLAMA_SERVE_SMOKE_FAIL"; exit 8; }
    chmod +x /tmp/asm_k1_ls.bin
    ( ulimit -s unlimited; timeout 600 /tmp/asm_k1_ls.bin ) || { echo "FATAL assemble_k1 concat"; echo "LLAMA_SERVE_SMOKE_FAIL"; exit 8; }
    ( ulimit -s unlimited; timeout 1200 ./seed.bin k1ptxdrv.hx "$DRV" ) || { echo "FATAL k1ptxdrv build"; echo "LLAMA_SERVE_SMOKE_FAIL"; exit 8; }
    chmod +x "$DRV"
  fi
  : > /tmp/kernel_in.hx
  for k in tiled_matmul tiled_matmul_abt gpu_softmax_causal gpu_layernorm_fwd_eps gpu_add_bias_rowbcast gpu_gelu_stable vector_add gpu_scale_rt gpu_rmsnorm_fwd_eps gpu_rope_rot gpu_silu_mul; do
    tr -d '\r' < "$ROOT/helixc/examples/${k}_kernel.hx" >> /tmp/kernel_in.hx; echo "" >> /tmp/kernel_in.hx
  done
  rm -f /tmp/out.ptx
  "$DRV" >/dev/null 2>&1 || true
  [ -s /tmp/out.ptx ] || { echo "FATAL: kovc emitted no PTX"; echo "LLAMA_SERVE_SMOKE_FAIL"; exit 8; }
  cp /tmp/out.ptx /tmp/llama_model.ptx
  echo "  minted /tmp/llama_model.ptx ($(stat -c%s /tmp/llama_model.ptx) B)"
fi
NENT=$(grep -c '\.entry' /tmp/llama_model.ptx)
[ "$NENT" = "11" ] || { echo "FATAL: PTX kernel count $NENT != 11"; echo "LLAMA_SERVE_SMOKE_FAIL"; exit 8; }
mkdir -p "$WORK/helixc/runtime"
cp "$ROOT/helixc/runtime/gpt2_infer.c" "$ROOT/helixc/runtime/gpt2_tok.c" "$ROOT/helixc/runtime/gpt2_serve_http.c" "$ROOT/helixc/runtime/gpt2_unicode_ranges.inc" "$WORK/helixc/runtime/"
cd "$WORK/helixc/runtime" || exit 8
gcc -DGPT2_SERVE -DGPT2_TOK_LIB gpt2_infer.c gpt2_tok.c -O2 -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -lm -o /tmp/lss_worker 2>/tmp/lss_wgcc.log || { echo "FAIL worker build"; cat /tmp/lss_wgcc.log; echo "LLAMA_SERVE_SMOKE_FAIL"; exit 8; }
gcc gpt2_serve_http.c -O2 -lpthread -o /tmp/lss_server 2>/tmp/lss_sgcc.log || { echo "FAIL server build"; cat /tmp/lss_sgcc.log; echo "LLAMA_SERVE_SMOKE_FAIL"; exit 8; }
echo "  built worker + server"

SI_SPAWN_ARGS=""
[ -s "$SI_WTS" ] && SI_SPAWN_ARGS="--model3 smollm2-360m-instruct --ptx3 /tmp/llama_model.ptx --weights3 $SI_WTS --vocab3 $SI_DIR/vocab.json --merges3 $SI_DIR/merges.txt --specials3 1 --eos3 2"
echo "=== [2] spawn multi-model server on :$PORT ==="
export HX_NL=48 HX_D=1600 HX_HEADS=25 HX_V=50257 HX_CTX=1024 HX_DFF=6400
/tmp/lss_server --port "$PORT" --root "$ROOT/demo" \
  --ptx /tmp/llama_model.ptx --weights "$XL_WEIGHTS" \
  --worker-bin /tmp/lss_worker \
  --vocab "$ROOT/helix-llm/models/gpt2-xl/vocab.json" --merges "$ROOT/helix-llm/models/gpt2-xl/merges.txt" \
  --max-ctx 320 --detail op --model gpt2-xl \
  --model2 smollm2-135m --ptx2 /tmp/llama_model.ptx --weights2 "$SM_WTS" \
  --vocab2 "$SM_DIR/vocab.json" --merges2 "$SM_DIR/merges.txt" $SI_SPAWN_ARGS \
  > /tmp/lss_server.log 2>&1 &
SRV_PID=$!
trap 'kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null' EXIT
READY=0
for i in $(seq 1 90); do
  H=$(curl -s -m 2 "http://127.0.0.1:$PORT/api/health" 2>/dev/null || true)
  WANT_READY=3; [ -s "$SI_WTS" ] && WANT_READY=4
  if echo "$H" | grep -q '"models":\[' && [ "$(echo "$H" | grep -o '"ready":true' | wc -l)" -ge "$WANT_READY" ]; then READY=1; break; fi
  sleep 2
done
if [ "$READY" != "1" ]; then
  echo "FATAL: workers not ready in 180s; health: $(curl -s -m 2 http://127.0.0.1:$PORT/api/health)"
  tail -10 /tmp/lss_server.log; echo "LLAMA_SERVE_SMOKE_FAIL"; exit 7
fi
echo "  both workers READY"

echo "=== [3] /api/health models[] ==="
H=$(curl -s -m 5 "http://127.0.0.1:$PORT/api/health")
echo "  $H"
echo "$H" | grep -q '"model":"gpt2-xl"' && echo "$H" | grep -q '"model":"smollm2-135m"' \
  && echo "  HEALTH_MODELS_OK" || { echo "  HEALTH_MODELS_FAIL"; RC=1; }

echo "=== [4] smollm2 generation: served ids token-for-token vs oracle ==="
PROMPT="The capital of France is"
( curl -s -N -m 600 -X POST "http://127.0.0.1:$PORT/api/generate?detail=op" \
    -H 'Content-Type: application/json' \
    -d "{\"prompt\":\"$PROMPT\",\"n_gen\":20,\"model\":\"smollm2-135m\"}" > /tmp/lss_sse.txt ) &
CURL_PID=$!
sleep 3
echo "=== [6] single-flight across models: concurrent request must 409 ==="
B=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/api/generate" \
     -H 'Content-Type: application/json' -d '{"prompt":"hi","n_gen":1}')
[ "$B" = "409" ] && echo "  CONCURRENT_409_OK" || { echo "  CONCURRENT got $B (want 409)"; RC=1; }
wait $CURL_PID
GOT_IDS=$(grep -o '"_ev":"token"[^}]*"id":[0-9]*' /tmp/lss_sse.txt | grep -o '"id":[0-9]*' | cut -d: -f2 | tr '\n' ' ')
PROMPT_IDS=$(grep -o '"_ev":"tokenize"[^]]*\]' /tmp/lss_sse.txt | head -1 | grep -o '\[[0-9, ]*\]' | tr ',' ' ' | tr -d '[]' )
SERVED="$(echo $PROMPT_IDS $GOT_IDS | tr -s ' ')"
REF="$(tr -s ' \n' '  ' < /tmp/lss_ref_base_ids.txt | sed 's/^ *//;s/ *$//')"
echo "  served: $SERVED"
echo "  oracle: $REF"
if [ "$(echo "$SERVED" | sed 's/^ *//;s/ *$//')" = "$REF" ]; then
  echo "  SERVE_TOKEN_FOR_TOKEN_OK"
else
  echo "  SERVE_TOKEN_FOR_TOKEN_FAIL"; RC=1
fi

echo "=== [7] INSTRUCT leg: templated ChatML chat over HTTP (specials + eos-stop) ==="
if [ -s "$SI_WTS" ]; then
  python3 - > /tmp/lss_chat_body.json <<PYB
import json
sys_p = "You are a helpful AI assistant named SmolLM, trained by Hugging Face"
user = "What is the capital of France?"
tmpl = "<|im_start|>system"+chr(10)+sys_p+"<|im_end|>"+chr(10)+"<|im_start|>user"+chr(10)+user+"<|im_end|>"+chr(10)+"<|im_start|>assistant"+chr(10)
print(json.dumps({"prompt": tmpl, "n_gen": 40, "model": "smollm2-360m-instruct"}))
PYB
  curl -s -N -m 600 -X POST "http://127.0.0.1:$PORT/api/generate?detail=op" -H "Content-Type: application/json" --data @/tmp/lss_chat_body.json > /tmp/lss_chat_sse.txt
  CGOT=$(grep -o '"_ev":"token"[^}]*"id":[0-9]*' /tmp/lss_chat_sse.txt | grep -o '"id":[0-9]*' | cut -d: -f2 | tr '\n' ' ')
  CPROMPT=$(grep -o '"_ev":"tokenize"[^]]*]' /tmp/lss_chat_sse.txt | head -1 | grep -o '[[][0-9, ]*]' | tr ',' ' ' | tr -d '[]')
  CSERVED="$(echo $CPROMPT $CGOT | tr -s ' ')"
  CREF="$(tr -s ' \n' '  ' < /tmp/lss_ref_chat_ids.txt | sed 's/^ *//;s/ *$//')"
  echo "  served: $CSERVED"
  echo "  oracle: $CREF"
  if [ "$(echo "$CSERVED" | sed 's/^ *//;s/ *$//')" = "$CREF" ]; then
    echo "  CHAT_TOKEN_FOR_TOKEN_OK (incl. C-tokenizer specials parity + eos-stop)"
  else
    echo "  CHAT_TOKEN_FOR_TOKEN_FAIL"; RC=1
  fi
else
  echo "  (instruct weights absent -- leg skipped)"
fi

echo "=== [5] unknown model -> 404 ==="
U=$(curl -s -m 10 -o /dev/null -w '%{http_code}' -X POST "http://127.0.0.1:$PORT/api/generate" \
     -H 'Content-Type: application/json' -d '{"prompt":"x","n_gen":1,"model":"not-a-model"}')
[ "$U" = "404" ] && echo "  UNKNOWN_MODEL_404_OK" || { echo "  UNKNOWN got $U (want 404)"; RC=1; }

kill $SRV_PID 2>/dev/null; wait $SRV_PID 2>/dev/null; trap - EXIT
echo "=================== SMOKE VERDICT (wall $(( $(date +%s) - T0 ))s) ==================="
if [ "$RC" = "0" ]; then echo "LLAMA_SERVE_SMOKE_PASS"; exit 0; else echo "LLAMA_SERVE_SMOKE_FAIL"; exit 1; fi
