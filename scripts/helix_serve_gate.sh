#!/usr/bin/env bash
# helix_serve_gate.sh -- FAIL-CLOSED integration gate for the GPT-2-XL-on-Helix LIVE chat backend.
#
#   MSYS_NO_PATHCONV=1 bash scripts/helix_serve_gate.sh
#
# Proves the live --serve worker + the C HTTP+SSE server are honest and additive:
#   G-BUILD : mint the XL PTX fresh from the 299-byte raw seed (9837db12), build the serve worker
#             (gpt2_infer.c + gpt2_tok.c, GPT2_SERVE+GPT2_TOK_LIB) and the HTTP server (gcc). SERIAL.
#   G1      : the SERVED XL continuation (POST /api/generate "The capital of France is", n_gen=20)
#             == the OFFLINE scripts/gpt2_scale.sh MODEL=gpt2-xl gen-ids TOKEN-FOR-TOKEN (25 ids;
#             argmax 262; the "...the city of Paris..." continuation). Proves served == verified-offline
#             and that the emit hooks + the d_ctx safety memset changed NOTHING numeric.
#   G-HEALTH: GET /api/health -> ready:true (weights loaded + PTX minted).
#   G7      : single-flight -- two concurrent /api/generate -> exactly one 200 stream + one 409.
#   G-FIX   : the self-host fixpoint protected files are byte-identical (git status clean on them).
# The 124M/scale/CPU regressions (G2/G3) are run by their existing gate scripts (invoked here if asked
# with FULL=1; default runs the serve-specific gates + a quick fixpoint check, since those existing
# gates each take many minutes on the GPU and are STRICTLY SERIAL).
#
# Prints HELIX_SERVE_GATE_PASS / HELIX_SERVE_GATE_FAIL and exits 0 only on PASS. Run as a FILE under WSL.
#   Portability (run on a machine other than the author's): override any of
#     HELIX_SRC         (default: auto-detected from script location, else /mnt/c/Projects/Kovostov-Native) -- the committed repo
#     HELIX_WORK        (default: $HOME/gpt2_ext4/Kovostov-Native)    -- the fast ext4 build mirror
#     HELIX_XL_WEIGHTS  (default: $HOME/gpt2_ext4/gpt2-xl.weights)    -- the XL flat .weights file
#   See docs/HELIX_GPT2_DEMO_RUNBOOK.md §0 for how to PRODUCE gpt2-xl.weights (HuggingFace + gpt2_pack.c).
set -u

SRC="${HELIX_SRC:-}"; if [ -z "$SRC" ]; then _sd="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; if [ -n "$_sd" ] && [ -f "$_sd/scripts/reproduce_trust.sh" ]; then SRC="$_sd"; else SRC="/mnt/c/Projects/Kovostov-Native"; fi; fi
WORK="${HELIX_WORK:-$HOME/gpt2_ext4/Kovostov-Native}"
XL_WEIGHTS="${HELIX_XL_WEIGHTS:-$HOME/gpt2_ext4/gpt2-xl.weights}"
BS=$WORK/stage0/helixc-bootstrap
SEED_PIN=9837db12
PORT="${PORT:-8848}"
NGEN="${NGEN:-20}"
PROMPT="${PROMPT:-The capital of France is}"
FULL="${FULL:-0}"                       # FULL=1 also runs gpt2_gpu_mvp.sh + gpt2_scale.sh xl (long, serial)
OK=1; G1=0; GHEALTH=0; G7=0; GFIX=0
echo "=================== HELIX SERVE GATE  $(date -u +%H:%M:%S) ==================="

# XL dims (env-driven, exactly as gpt2_scale.sh drives the committed dimension-generic launcher).
export HX_NL=48 HX_D=1600 HX_HEADS=25 HX_V=50257 HX_CTX=1024 HX_DFF=6400

# =================== [BUILD] mint PTX from raw seed + build worker + server (SERIAL) ===================
echo "=== [BUILD] mint XL PTX from raw seed (9837db12) + build serve worker + HTTP server ==="
mkdir -p "$BS" "$WORK/helixc/bootstrap" "$WORK/helixc/runtime"
cp -r $SRC/stage0/helixc-bootstrap/. "$BS"/
cp $SRC/helixc/bootstrap/lexer.hx  "$WORK/helixc/bootstrap/"
cp $SRC/helixc/bootstrap/parser.hx "$WORK/helixc/bootstrap/"
cp $SRC/helixc/bootstrap/kovc.hx   "$WORK/helixc/bootstrap/"
sed -i "s#/mnt/c/Projects/Kovostov-Native/#$WORK/#g" "$BS/assemble_k1.hx"
_seedsha=$(sha256sum "$BS/seed.bin" 2>/dev/null | cut -c1-8)
if [ "$_seedsha" != "$SEED_PIN" ]; then echo "  FAIL: ext4 seed.bin sha $_seedsha != pinned $SEED_PIN"; OK=0;
else echo "  ext4 seed.bin sha=$_seedsha (== pinned $SEED_PIN)"; fi

cd "$BS" || { echo "FAIL: no bootstrap dir"; OK=0; }
rm -f /tmp/asm_k1_sg.bin /tmp/newdrv_sg.bin /tmp/out.ptx
if [ "$OK" = "1" ] && [ -x "$BS/seed.bin" ]; then
  ( ulimit -s unlimited; timeout 600 ./seed.bin assemble_k1.hx /tmp/asm_k1_sg.bin ) >/tmp/sg_asm.log 2>&1
  [ -s /tmp/asm_k1_sg.bin ] || { echo "  FAIL: assemble_k1"; OK=0; }
  chmod +x /tmp/asm_k1_sg.bin 2>/dev/null
  ( ulimit -s unlimited; timeout 600 /tmp/asm_k1_sg.bin ) >/tmp/sg_concat.log 2>&1
  [ -s "$BS/k1ptxdrv.hx" ] || { echo "  FAIL: concatenator"; OK=0; }
  ( ulimit -s unlimited; timeout 1200 ./seed.bin k1ptxdrv.hx /tmp/newdrv_sg.bin ) >/tmp/sg_drv.log 2>&1
  [ -s /tmp/newdrv_sg.bin ] || { echo "  FAIL: seed->newdrv"; OK=0; }
  chmod +x /tmp/newdrv_sg.bin 2>/dev/null
fi
: > /tmp/kernel_in.hx
for k in tiled_matmul tiled_matmul_abt gpu_softmax_causal gpu_layernorm_fwd_eps gpu_add_bias_rowbcast gpu_gelu_stable vector_add gpu_scale_rt; do
  tr -d '\r' < $SRC/helixc/examples/${k}_kernel.hx >> /tmp/kernel_in.hx; echo "" >> /tmp/kernel_in.hx
done
if [ "$OK" = "1" ]; then ( ulimit -s unlimited; /tmp/newdrv_sg.bin ) >/tmp/sg_emit.log 2>&1 || true; fi
if [ -s /tmp/out.ptx ]; then cp /tmp/out.ptx /tmp/gpt2_serve.ptx
  echo "  PTX $(stat -c%s /tmp/gpt2_serve.ptx) B, $(grep -c '\.entry' /tmp/gpt2_serve.ptx) .entry kernels (seed-minted)"
else echo "  FAIL: no PTX emitted"; OK=0; fi

# build the serve worker + HTTP server on ext4
cp $SRC/helixc/runtime/gpt2_infer.c "$WORK/helixc/runtime/"
cp $SRC/helixc/runtime/gpt2_tok.c "$WORK/helixc/runtime/"
cp $SRC/helixc/runtime/gpt2_serve_http.c "$WORK/helixc/runtime/"
cp $SRC/helixc/runtime/gpt2_unicode_ranges.inc "$WORK/helixc/runtime/"
cd "$WORK/helixc/runtime" || OK=0
rm -f /tmp/gpt2_infer_serve /tmp/gpt2_serve_http
gcc -DGPT2_SERVE -DGPT2_TOK_LIB gpt2_infer.c gpt2_tok.c -O2 -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -lm -o /tmp/gpt2_infer_serve 2>/tmp/sg_worker_gcc.log \
  || { echo "  WORKER GCC FAIL"; sed 's/^/    /' /tmp/sg_worker_gcc.log; OK=0; }
gcc gpt2_serve_http.c -O2 -lpthread -o /tmp/gpt2_serve_http 2>/tmp/sg_server_gcc.log \
  || { echo "  SERVER GCC FAIL"; sed 's/^/    /' /tmp/sg_server_gcc.log; OK=0; }
[ -x /tmp/gpt2_infer_serve ] && [ -x /tmp/gpt2_serve_http ] && echo "  built /tmp/gpt2_infer_serve + /tmp/gpt2_serve_http" || { echo "  FAIL: a binary is missing"; OK=0; }

# =================== [OFFLINE ORACLE] run gpt2_scale.sh MODEL=gpt2-xl for the G1 reference ===================
# This mints its OWN PTX from the same raw seed and runs --generate 20 token-for-token vs the numpy
# oracle, dumping the 25-id sequence to /tmp/helix_gen_ids.txt. We capture THAT as the G1 target.
echo "=== [OFFLINE] gpt2_scale.sh MODEL=gpt2-xl (the G1 token-for-token reference) ==="
# G1 HARD-REQUIRES a GENUINE PRIMARY oracle PASS here: the offline gen-ids are only an honest G1
# anchor if gpt2_scale.sh did a full-logits token-for-token PASS vs the numpy oracle (printed
# ^GPT2_SCALE_PARITY_PASS). If the oracle OOMs/degrades to the helix-only block-0+coherence FALLBACK,
# /tmp/helix_gen_ids.txt would be helix-vs-helix -- so we FAIL-CLOSE (OK=0), never compare served==offline
# against an unverified offline reference.
OFFLINE_IDS=""
if [ "$OK" = "1" ]; then
  tr -d '\r' < $SRC/scripts/gpt2_scale.sh > /tmp/sg_scale.sh
  rm -f /tmp/helix_gen_ids.txt
  if MODEL=gpt2-xl NGEN=$NGEN PROMPT="$PROMPT" bash /tmp/sg_scale.sh >/tmp/sg_scale.log 2>&1; then
    if grep -q '^GPT2_SCALE_PARITY_PASS' /tmp/sg_scale.log; then
      echo "  gpt2_scale.sh xl: GPT2_SCALE_PARITY_PASS"
    else
      echo "  FAIL: gpt2_scale.sh did NOT print GPT2_SCALE_PARITY_PASS -- offline reference is not a"
      echo "        genuine PRIMARY oracle token-for-token PASS (refusing a helix-vs-helix G1 anchor)."
      tail -6 /tmp/sg_scale.log | sed 's/^/    /'
      OK=0
    fi
  else
    echo "  FAIL: gpt2_scale.sh exited nonzero (no genuine oracle PASS for the G1 reference); tail:"
    tail -8 /tmp/sg_scale.log | sed 's/^/    /'
    OK=0
  fi
  if [ "$OK" = "1" ] && [ -s /tmp/helix_gen_ids.txt ]; then
    OFFLINE_IDS=$(tr -s ' \n' ' ' < /tmp/helix_gen_ids.txt | sed 's/^ //; s/ $//')
    echo "  OFFLINE gen-ids (25): $OFFLINE_IDS"
  elif [ "$OK" = "1" ]; then
    echo "  FAIL: gpt2_scale.sh produced no /tmp/helix_gen_ids.txt (no G1 reference)"; OK=0
  fi
fi

# =================== [SERVER] start the HTTP server (worker resident) ===================
echo "=== [SERVER] start gpt2_serve_http (spawns the persistent --serve worker) ==="
SERVER_PID=""
if [ "$OK" = "1" ]; then
  VOCAB=$SRC/helix-llm/models/gpt2-xl/vocab.json
  MERGES=$SRC/helix-llm/models/gpt2-xl/merges.txt
  HX_NL=48 HX_D=1600 HX_HEADS=25 HX_V=50257 HX_CTX=1024 HX_DFF=6400 \
  /tmp/gpt2_serve_http --port $PORT --root $SRC/demo \
    --ptx /tmp/gpt2_serve.ptx --weights "$XL_WEIGHTS" \
    --worker-bin /tmp/gpt2_infer_serve --vocab "$VOCAB" --merges "$MERGES" \
    --max-ctx 320 --detail op --oracle $SRC/helix-llm/tools --model gpt2-xl \
    >/tmp/sg_server.log 2>&1 &
  SERVER_PID=$!
  echo "  server pid=$SERVER_PID; waiting for worker GPT2_SERVE_READY (load XL weights + PTX)..."
  # poll /api/health until ready (XL weight load + PTX module load can take ~30-60s)
  ready=0
  for i in $(seq 1 90); do
    sleep 2
    H=$(curl -s --max-time 3 http://127.0.0.1:$PORT/api/health 2>/dev/null || true)
    if echo "$H" | grep -q '"ready":true'; then ready=1; echo "  /api/health ready after ~$((i*2))s: $H"; break; fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then echo "  FAIL: server died early"; tail -8 /tmp/sg_server.log | sed 's/^/    /'; break; fi
  done
  if [ "$ready" = "1" ]; then GHEALTH=1; else echo "  FAIL: /api/health never reported ready"; OK=0; fi
fi

# =================== [G1] POST /api/generate -> capture SSE -> token-for-token vs offline ===================
SERVED_IDS=""
SSE_TRANSCRIPT=/tmp/sg_sse.txt
if [ "$OK" = "1" ] && [ "$GHEALTH" = "1" ]; then
  echo "=== [G1] POST /api/generate {prompt:\"$PROMPT\", n_gen:$NGEN} -> capture SSE ==="
  BODY=$(printf '{"prompt":"%s","n_gen":%d,"request_id":"gate-g1"}' "$PROMPT" "$NGEN")
  curl -sN --max-time 600 -H 'Content-Type: application/json' -H 'Accept: text/event-stream' \
    -X POST --data "$BODY" "http://127.0.0.1:$PORT/api/generate?detail=op" > "$SSE_TRANSCRIPT" 2>/tmp/sg_curl.log
  echo "  SSE bytes: $(stat -c%s "$SSE_TRANSCRIPT" 2>/dev/null || echo 0); event lines: $(grep -c '^event:' "$SSE_TRANSCRIPT")"
  echo "  --- SSE transcript (event histogram) ---"
  grep '^event:' "$SSE_TRANSCRIPT" | sort | uniq -c | sed 's/^/    /'
  # pull the done event's data line, extract gen_ids + text
  DONE_LINE=$(grep -A1 '^event: done' "$SSE_TRANSCRIPT" | grep '^data:' | head -1 | sed 's/^data: //')
  echo "  --- done event ---"; echo "    $DONE_LINE"
  # served gen_ids (20) -- the worker emits gen_ids as the GENERATED ids only; prepend the 5 prompt ids
  GEN20=$(echo "$DONE_LINE" | grep -o '"gen_ids":\[[0-9,]*\]' | sed 's/"gen_ids":\[//; s/\]//; s/,/ /g')
  # prompt ids from the tokenize event
  TOK_LINE=$(grep -A1 '^event: tokenize' "$SSE_TRANSCRIPT" | grep '^data:' | head -1 | sed 's/^data: //')
  PROMPT_IDS=$(echo "$TOK_LINE" | grep -o '"ids":\[[0-9,]*\]' | sed 's/"ids":\[//; s/\]//; s/,/ /g')
  SERVED_IDS=$(echo "$PROMPT_IDS $GEN20" | tr -s ' ' ' ' | sed 's/^ //; s/ $//')
  echo "  SERVED  ids (prompt+gen, 25): $SERVED_IDS"
  # token-for-token compare
  echo "  --- sample token events (id + string + logit) ---"
  grep -A1 '^event: token' "$SSE_TRANSCRIPT" | grep '^data:' | head -5 | sed 's/^data: /    /'
  if [ -n "$SERVED_IDS" ] && [ "$SERVED_IDS" = "$OFFLINE_IDS" ]; then
    echo "  G1 TOKEN_FOR_TOKEN_MATCH: served == offline gpt2_scale.sh (25/25 ids)"; G1=1
  else
    echo "  G1 MISMATCH:"; echo "    served : $SERVED_IDS"; echo "    offline: $OFFLINE_IDS"
  fi
fi

# =================== [G7] single-flight: two concurrent /api/generate -> one 200 + one 409 ===================
if [ "$OK" = "1" ] && [ "$GHEALTH" = "1" ]; then
  echo "=== [G7] single-flight: fire TWO concurrent /api/generate (expect one 200 stream + one 409) ==="
  BODY=$(printf '{"prompt":"%s","n_gen":%d}' "$PROMPT" 8)
  rm -f /tmp/sg_c1.code /tmp/sg_c2.code
  # request A: a real streaming generation (holds the worker); capture its status code
  ( curl -sN --max-time 300 -o /tmp/sg_c1.body -w '%{http_code}' -H 'Content-Type: application/json' \
      -X POST --data "$BODY" "http://127.0.0.1:$PORT/api/generate" > /tmp/sg_c1.code 2>/dev/null ) &
  CPID=$!
  sleep 3   # let A acquire the single-flight lock + start streaming
  # request B: should get 409 busy immediately
  C2=$(curl -s --max-time 20 -o /tmp/sg_c2.body -w '%{http_code}' -H 'Content-Type: application/json' \
      -X POST --data "$BODY" "http://127.0.0.1:$PORT/api/generate" 2>/dev/null)
  echo "  request B (concurrent) HTTP status: $C2  body: $(cat /tmp/sg_c2.body 2>/dev/null)"
  wait $CPID 2>/dev/null
  C1=$(cat /tmp/sg_c1.code 2>/dev/null)
  echo "  request A (first) HTTP status: $C1"
  if [ "$C2" = "409" ] && [ "$C1" = "200" ]; then echo "  G7 PASS: one 200 stream + one 409 busy (no interleave)"; G7=1;
  else echo "  G7 FAIL: expected A=200 B=409, got A=$C1 B=$C2"; fi
fi

# =================== teardown the server ===================
if [ -n "${SERVER_PID:-}" ]; then kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null; echo "  server stopped"; fi

# =================== [G-FIX] fixpoint protected files byte-identical (git status clean) ===================
echo "=== [G-FIX] self-host fixpoint protected files byte-identical (git status) ==="
PROTECTED="helixc/bootstrap/kovc.hx helixc/bootstrap/lexer.hx helixc/bootstrap/parser.hx helixc/runtime/train_transformer.c stage0/helixc-bootstrap/seed.c"
DIRTY=$(cd $SRC && git status --porcelain $PROTECTED 2>/dev/null)
if [ -z "$DIRTY" ]; then echo "  protected fixpoint files CLEAN (kovc.hx/lexer.hx/parser.hx/train_transformer.c/seed.c byte-identical)"; GFIX=1;
else echo "  FAIL: protected files modified:"; echo "$DIRTY" | sed 's/^/    /'; fi

# =================== [G2/G3 optional, FULL=1] existing regression gates (long, serial) ===================
if [ "$FULL" = "1" ] && [ "$OK" = "1" ]; then
  echo "=== [G2] gpt2_gpu_mvp.sh (124M logits + generation) ==="
  tr -d '\r' < $SRC/scripts/gpt2_gpu_mvp.sh > /tmp/sg_mvp.sh
  if bash /tmp/sg_mvp.sh >/tmp/sg_mvp.log 2>&1 && grep -q '^GPT2_GENERATE_MATCH_PASS' /tmp/sg_mvp.log && grep -q '^GPT2_LOGITS_PARITY_PASS' /tmp/sg_mvp.log; then
    echo "  G2 PASS (124M logits+generation green)"
  else echo "  G2 FAIL"; tail -6 /tmp/sg_mvp.log | sed 's/^/    /'; OK=0; fi
fi

# =================== VERDICT ===================
echo "=================== HELIX SERVE GATE VERDICT ==================="
echo "  G-BUILD : $([ "$OK" = "1" ] && echo PASS || echo FAIL)   (PTX minted + worker + server built, serial)"
echo "  G-HEALTH: $([ "$GHEALTH" = "1" ] && echo PASS || echo FAIL)   (/api/health ready: weights+PTX)"
echo "  G1      : $([ "$G1" = "1" ] && echo PASS || echo FAIL)   (served XL == offline gpt2_scale.sh token-for-token)"
echo "  G7      : $([ "$G7" = "1" ] && echo PASS || echo FAIL)   (single-flight: 200 + 409)"
echo "  G-FIX   : $([ "$GFIX" = "1" ] && echo PASS || echo FAIL)   (fixpoint protected files byte-identical)"
echo "(done $(date -u +%H:%M:%S))"
if [ "$OK" = "1" ] && [ "$G1" = "1" ] && [ "$GHEALTH" = "1" ] && [ "$G7" = "1" ] && [ "$GFIX" = "1" ]; then
  echo "HELIX_SERVE_GATE_PASS"; exit 0
else
  echo "HELIX_SERVE_GATE_FAIL"; exit 1
fi
