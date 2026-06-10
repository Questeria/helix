#!/usr/bin/env bash
# measure_decode_perf.sh -- HONEST decode tokens/sec for the 360M-Instruct chat path.
# Methodology (stated, crude, honest): run --generate twice per config (NGEN=2 and NGEN=42)
# and divide the extra 40 tokens by the wall-clock delta -- this cancels startup (PTX load,
# weights mmap, oracle-free) and the shared prefill, isolating per-token decode cost.
# Configs: A = baseline full-reforward (today's path), B = KV-cache (HX_KV=1 HX_RESIDENT=1),
# C = KV + fast (drop per-op host syncs; same kernels, same ids).
# REQUIREMENT: ids must be IDENTICAL across all configs (checked; mismatch = FAIL).
# Run as a FILE under WSL; GPU must be free. Prints MEASURED numbers only.
set -u
PERF_RC=0
ROOT="${HELIX_SRC:-}"; if [ -z "$ROOT" ]; then ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; fi
[ -d "$ROOT/helixc/examples" ] || ROOT="/mnt/c/Projects/Kovostov-Native"
MODELD="$ROOT/helix-llm/models/smollm2-360m-instruct"
WTS="$MODELD/smollm2-360m-instruct.weights"
REFD="$ROOT/helix-llm/ref"
TOOLS="$ROOT/helix-llm/tools"
echo "=================== DECODE PERF (SmolLM2-360M-Instruct, chat prompt) ==================="
[ -s /tmp/llama_model.ptx ] || { echo "FATAL: no /tmp/llama_model.ptx (run llama_model_gate.sh first)"; exit 9; }
[ -x /tmp/llama_infer ] || { echo "FATAL: no /tmp/llama_infer (run llama_model_gate.sh first)"; exit 9; }
# chat refs for a LONG generation (no eos-stop interference: unset HX_EOS so all N tokens run)
( cd "$TOOLS" && LLAMA_MODEL_DIR="$MODELD" LLAMA_CHAT=1 python3 llama_numpy_ref.py dump-logits "Tell me about the history of computers." ) >/tmp/perf_ref.log 2>&1 || { echo "FATAL ref dump"; exit 8; }

run_cfg () { # $1=name $2=envs
  local name="$1"; shift
  local t2 t42
  rm -f /tmp/helix_gen_ids.txt
  t2=$( { /usr/bin/time -f %e env "$@" /tmp/llama_infer /tmp/llama_model.ptx "$WTS" --generate 2 "$REFD/llama_ref_ids.txt" >/tmp/perf_g2.log; } 2>&1 | tail -1 )
  [ -s /tmp/helix_gen_ids.txt ] || { echo "  $name: NGEN=2 produced no ids (CRASH) -> FAIL"; PERF_RC=1; return 1; }
  cp /tmp/helix_gen_ids.txt "/tmp/perf_ids_${name}_2.txt"
  t42=$( { /usr/bin/time -f %e env "$@" /tmp/llama_infer /tmp/llama_model.ptx "$WTS" --generate 42 "$REFD/llama_ref_ids.txt" >/tmp/perf_g42.log; } 2>&1 | tail -1 )
  [ -s /tmp/helix_gen_ids.txt ] || { echo "  $name: NGEN=42 produced no ids (CRASH) -> FAIL"; PERF_RC=1; return 1; }
  cp /tmp/helix_gen_ids.txt "/tmp/perf_ids_${name}_42.txt"
  local dt; dt=$(python3 -c "print(max(0.0001, $t42 - $t2))")
  local tps; tps=$(python3 -c "print(round(40.0 / $dt, 2))")
  echo "  $name: NGEN=2 ${t2}s, NGEN=42 ${t42}s -> 40 extra tokens in ${dt}s = ${tps} tok/s (decode)"
}

echo "--- A: baseline (full re-forward each token; per-op host syncs) ---"
run_cfg baseline HX_KV=0
echo "--- B: KV-cache + resident weights ---"
run_cfg kv HX_KV=1 HX_RESIDENT=1
echo "--- C: KV-cache + resident + FAST (no per-op host syncs) ---"
run_cfg kvfast HX_KV=1 HX_RESIDENT=1 HX_FAST=1

echo "--- ids identical across configs? (REQUIRED) ---"
ok=1
cmp -s /tmp/perf_ids_baseline_2.txt /tmp/perf_ids_kv_2.txt || { echo "  MISMATCH baseline vs kv (NGEN=2)"; ok=0; }
cmp -s /tmp/perf_ids_baseline_2.txt /tmp/perf_ids_kvfast_2.txt || { echo "  MISMATCH baseline vs kvfast (NGEN=2)"; ok=0; }
cmp -s /tmp/perf_ids_baseline_42.txt /tmp/perf_ids_kv_42.txt || { echo "  MISMATCH baseline vs kv"; ok=0; }
cmp -s /tmp/perf_ids_baseline_42.txt /tmp/perf_ids_kvfast_42.txt || { echo "  MISMATCH baseline vs kvfast"; ok=0; }
[ "$ok" = "1" ] && echo "  IDS_IDENTICAL across all 3 configs (42-token runs)"
[ "$ok" = "1" ] && [ "$PERF_RC" = "0" ] && echo "PERF_MEASURE_OK" || { echo "PERF_MEASURE_FAIL"; exit 1; }
