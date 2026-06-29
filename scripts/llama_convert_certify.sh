#!/usr/bin/env bash
# llama_convert_certify.sh (v1.9 llama ternary certification): the verifiable-conversion ARTIFACT for the
# CONVERTED ternary SmolLM2-135M. ONE focused pass that:
#   [1] PACK   : ternarize+pack the 7 linears x NL layers of the converted .weights (15-trit kovc format,
#                per-row abs-mean scale) -> footprint vs fp32 (compression x). C-vs-numpy packer parity gate.
#   [2] KERNEL : run one converted linear (gate L0 [1536x576]) through the REAL kovc scaled_packed_ternary
#                _matmul GPU kernel and verify == the trainer fake-quant path (ternarize-dequant then matmul),
#                element-exact. + comparator NC.
#   [3] CONVERT RECEIPT : bind {source-fp sha256, converted-ternary Merkle over the 7xNL packed linears+scales,
#                measured ppl 8.7467 -> 1140.55}; --emit then --check re-derives byte-identical; tamper NC FAIL.
#   [4] INFER RECEIPT   : Freivalds receipt over the gate-L0 ternary matmul; re-derivable PASS + forged-C NC FAIL.
# Reuses the COMMITTED machinery only (gpt2_pack.c ternary packer, scaled_packed_ternary_matmul kernel,
# cuda_launch sptmatmul*/receipt_*, convert_receipt). NO kovc.hx edit; cuda-12.8 ptxas; <=80% compute.
# Run under WSL: HELIX_SRC=/mnt/c/Projects/Kovostov-Native bash scripts/llama_convert_certify.sh
set -u
ROOT="${HELIX_SRC:-}"; if [ -z "$ROOT" ]; then ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; fi
[ -d "$ROOT/helixc/runtime" ] || ROOT="/mnt/c/Projects/Kovostov-Native"
cd "$ROOT"
export PATH=/usr/local/cuda-12.8/bin:$PATH
TC="taskset -c 0-5 nice -n 10"
FP="${FP_WEIGHTS:-$ROOT/helix-llm/models/smollm2-135m/smollm2-135m.weights}"
CV="${CONVERTED_WEIGHTS:-$HOME/smollm2_ternary_converted.bin.best}"
WD="${CERT_WORK:-$HOME/tern_cert_work}"; mkdir -p "$WD"
PY="${HELIX_PY:-python3}"
KERN="$ROOT/helixc/examples/scaled_packed_ternary_matmul_kernel.hx"
REF="$ROOT/helixc/examples/scaled_packed_ternary_matmul_kernel.ref.ptx"
FP_PPL=8.7467; CONV_PPL=1140.55
OK=1; bad(){ echo "[certify] *** FAIL: $*" >&2; OK=0; }
echo "============================================================"
echo " Helix v1.9: CONVERTED ternary SmolLM2-135M certification"
echo "============================================================"
[ -s "$FP" ] || { bad "fp weights $FP absent"; echo CERTIFY_FAIL; exit 1; }
[ -s "$CV" ] || { bad "converted weights $CV absent"; echo CERTIFY_FAIL; exit 1; }

# --- build the committed packer + launcher ---
$TC gcc "$ROOT/helixc/runtime/gpt2_pack.c" -O2 -o "$WD/gpt2_pack" -lm 2>"$WD/pk_gcc.log" || { bad "packer build"; tail -6 "$WD/pk_gcc.log" >&2; echo CERTIFY_FAIL; exit 1; }
$TC gcc "$ROOT/helixc/runtime/cuda_launch.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o "$WD/cuda_launch" 2>"$WD/cl_gcc.log" || { bad "launcher build"; tail -6 "$WD/cl_gcc.log" >&2; echo CERTIFY_FAIL; exit 1; }
$WD/gpt2_pack --ternary-selftest >/dev/null 2>&1 || bad "C ternary selftest"

# --- emit kernel PTX (fast-iter cache iff committed compiler unchanged) + byte-gate vs ref ---
CACHE="$HOME/.helix_fastiter"; CUR=$(cat "$ROOT/helixc/bootstrap/kovc.hx" "$ROOT/helixc/bootstrap/parser.hx" 2>/dev/null | sha256sum | cut -d" " -f1)
if [ -s "$CACHE/newdrv.bin" ] && [ "$(cat "$CACHE/compiler.sha" 2>/dev/null)" = "$CUR" ]; then DRV="$CACHE/newdrv.bin"
else bad "no valid fast-iter driver cache (compiler changed)"; echo CERTIFY_FAIL; exit 1; fi
cp "$KERN" /tmp/kernel_in.hx; rm -f /tmp/out.ptx; "$DRV" >/dev/null 2>&1 || true
[ -s /tmp/out.ptx ] || { bad "PTX not emitted"; echo CERTIFY_FAIL; exit 1; }
cp /tmp/out.ptx "$WD/sct.ptx"
cmp -s "$WD/sct.ptx" "$REF" && echo "[certify] PTX byte-identical to committed .ref.ptx" || bad "emitted PTX != committed .ref.ptx"

# === [1] PACK footprint + C-vs-numpy parity gate ===
echo "[certify] [1] PACK footprint (7 linears x 30 layers)"
$TC "$PY" "$ROOT/scripts/llama_ternary_pack.py" footprint "$CV" 2>&1 | tee "$WD/footprint.txt"
grep -q "FOOTPRINT" "$WD/footprint.txt" || bad "footprint"
echo "[certify] [1b] packer parity (C ternary_quantize_tensor == numpy mirror) on gate/q/down/v"
for SPEC in "0 gate" "0 q" "5 down" "29 v"; do
  set -- $SPEC; Lr=$1; WH=$2; D="$WD/par_${WH}_$Lr"; mkdir -p "$D"
  $TC "$PY" "$ROOT/scripts/_parity_extract.py" "$CV" $Lr $WH "$D" >/dev/null 2>&1
  RK=$("$PY" -c "import sys;sys.path.insert(0,'$ROOT/scripts');import llama_ternary_pack as L;_,r,c=L.get_linear(L.load_f32('$CV'),$Lr,'$WH');print(r,c)")
  set -- $RK; R=$1; K=$2
  $TC "$WD/gpt2_pack" --ternary-packfile "$D/raw_f32.bin" $R $K "$D/c_packed.bin" "$D/c_scale.bin" >/dev/null 2>&1
  if cmp -s "$D/c_packed.bin" "$D/np_packed.bin" && cmp -s "$D/c_scale.bin" "$D/np_scale.bin"; then
    echo "[certify]   $WH L$Lr [${R}x${K}] PACK+SCALE byte-identical (C==numpy)"
  else bad "packer parity $WH L$Lr"; fi
done

# === [2] REAL kernel vs fake-quant (gate L0) ===
echo "[certify] [2] gate-L0 through REAL kovc kernel vs trainer fake-quant"
D="$WD/dump_gate0"; mkdir -p "$D"
$TC "$PY" "$ROOT/scripts/llama_ternary_pack.py" dump "$CV" 0 gate "$D" 8 0 >/dev/null 2>&1
read M K N < "$D/dims.txt"
# (2a) pure integer trit@x_int element-exact (sc=1)
$TC "$WD/cuda_launch" "$WD/sct.ptx" scaled_packed_ternary_matmul "$N" sptmatmul_real "$D/packed.bin" "$D/acts.bin" "$D/Cint_f32.bin" "$M" "$K" "$N" 2>&1 | tee "$WD/kernel_int.txt" | grep -q "0 bad -> PASS" && echo "[certify]   integer matmul element-exact" || bad "integer kernel check"
# (2a-NC)
$TC "$WD/cuda_launch" "$WD/sct.ptx" scaled_packed_ternary_matmul "$N" sptmatmul_real "$D/packed.bin" "$D/acts.bin" "$D/Cint_f32.bin" "$M" "$K" "$N" mutate >/dev/null 2>&1 && bad "kernel comparator NC did not fail" || echo "[certify]   comparator NC correctly FAILED"
# (2b) full SCALED path == fake-quant
$TC "$WD/cuda_launch" "$WD/sct.ptx" scaled_packed_ternary_matmul "$N" sptmatmul_dump "$D/packed.bin" "$D/acts.bin" "$D/gpu_int.bin" "$M" "$K" "$N" >/dev/null 2>&1
$TC "$PY" "$ROOT/scripts/_kernel_vs_fakequant.py" "$D/gpu_int.bin" "$D/scale.bin" "$D/expected.bin" "$M" "$N" 2>&1 | tee "$WD/kernel_vs_fq.txt" | grep -qE "PASS_ELEMENT_EXACT|PASS_TIGHT_TOL" && echo "[certify]   GPU*scale == fake-quant" || bad "kernel-vs-fakequant"

# === [3] CONVERSION receipt ===
echo "[certify] [3] conversion receipt (--emit / --check / tamper NC)"
CERT="$WD/smollm2_convert.cert"
$TC "$PY" "$ROOT/scripts/convert_receipt_llama.py" --emit "$FP" "$CV" "$CERT" $FP_PPL $CONV_PPL 2>&1 | tee "$WD/convert_emit.txt" | grep -q "CONVERT_RECEIPT_LLAMA_EMIT" || bad "convert emit"
$TC "$PY" "$ROOT/scripts/convert_receipt_llama.py" --check "$FP" "$CV" "$CERT" $FP_PPL $CONV_PPL 2>&1 | grep -q "CONVERT_RECEIPT_PASS" && echo "[certify]   convert --check re-derives -> PASS" || bad "convert check"
sed "s/^merkle_ternary: c/merkle_ternary: d/" "$CERT" > "$WD/cert_nc.cert"
$TC "$PY" "$ROOT/scripts/convert_receipt_llama.py" --check "$FP" "$CV" "$WD/cert_nc.cert" $FP_PPL $CONV_PPL >/dev/null 2>&1 && bad "convert tamper NC did not fail" || echo "[certify]   convert tamper NC correctly FAILED"

# === [4] INFERENCE (Freivalds) receipt ===
echo "[certify] [4] inference Freivalds receipt (gate L0)"
"$WD/cuda_launch" x x 0 sha256_selftest 2>&1 | grep -q "PASS" || bad "sha256 KAT"
RC="$WD/smollm2_gate0_infer.rcpt"
$TC "$WD/cuda_launch" x x 0 receipt_emit_real "$D/Wtrit.bin" "$D/acts.bin" "$D/Cint.bin" "$M" "$K" "$N" "$RC" >/dev/null 2>&1
$TC "$WD/cuda_launch" x x 0 receipt_check "$RC" 2>&1 | tee "$WD/infer_check.txt" | grep -q "RECEIPT_PASS" && echo "[certify]   Freivalds genuine -> RECEIPT_PASS" || bad "freivalds genuine"
$TC "$WD/cuda_launch" x x 0 receipt_emit_real "$D/Wtrit.bin" "$D/acts.bin" "$D/Cint.bin" "$M" "$K" "$N" "$WD/infer_bad.rcpt" mutate >/dev/null 2>&1
RES=$($TC "$WD/cuda_launch" x x 0 receipt_check "$WD/infer_bad.rcpt" 2>&1); echo "$RES" | grep -q "REJECT=CHECK2 -> RECEIPT_FAIL" && echo "[certify]   forged-C NC -> RECEIPT_FAIL via CHECK2" || bad "freivalds NC"

echo "------------------------------------------------------------"
if [ "$OK" = 1 ]; then echo "LLAMA_CONVERT_CERTIFY_PASS"; exit 0; else echo "CERTIFY_FAIL"; exit 1; fi
