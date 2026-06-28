#!/usr/bin/env bash
# gpu_bitnet_layer_check.sh (v1.9 P3c): GPU-HARDWARE exact verification that the kovc-emitted
# scaled_packed_ternary_matmul reproduces a REAL BitNet b1.58 BitLinear layer (layer-0 q_proj) INTEGER
# matmul, element-for-element. Dumps the real ternary weights + an int8-quantized activation into the
# kernel format, launches the kernel with sc=1 (so c=i2f(int_dot)), compares == the host integer ref.
# NC: comparator mutate -> must fail. Synthetic kernel correctness is covered by gpu_scaled_ternary_check.sh;
# THIS adds "on a real model getter weights". Needs the BitNet checkpoint at $HOME/bitnet-2b (MIT, multi-GB,
# NOT in the repo) + numpy; if absent it SKIPS (exit 0). Run: HELIX_SRC=<root> bash scripts/gpu_bitnet_layer_check.sh
set -u
ROOT="${HELIX_SRC:-}"; if [ -z "$ROOT" ]; then ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; fi
[ -d "$ROOT/helixc/examples" ] || ROOT="/mnt/c/Projects/Kovostov-Native"
cd "$ROOT"
MODEL="${BITNET_MODEL:-$HOME/bitnet-2b/model.safetensors}"
PY="${HELIX_PY:-$HOME/alt-eval-venv/bin/python}"
KERN="$ROOT/helixc/examples/scaled_packed_ternary_matmul_kernel.hx"
DUMP="$ROOT/scripts/bitnet_layer_dump.py"
OD="${BITNET_LAYER_DIR:-$HOME/bitnet_layer}"
echo "============================================================"
echo " Helix v1.9 P3c: BitNet BitLinear layer GPU exact check"
echo "============================================================"
if [ ! -s "$MODEL" ] || ! "$PY" -c "import numpy" 2>/dev/null; then
  echo "[bitnet_layer] SKIP: BitNet model ($MODEL) or numpy absent -- the real-data check needs the (multi-GB) checkpoint."
  echo "BITNET_LAYER_SKIP"; exit 0
fi
OK=1; bad(){ echo "[bitnet_layer] *** FAIL: $*" >&2; OK=0; }
"$PY" "$DUMP" "$MODEL" "model.layers.0.self_attn.q_proj.weight" "$OD" 0 8 || { echo BITNET_LAYER_FAIL; exit 1; }
read M K N < "$OD/dims.txt"; echo "[bitnet_layer] dims M=$M K=$K N=$N"
CACHE="$HOME/.helix_fastiter"; CUR=$(cat "$ROOT/helixc/bootstrap/kovc.hx" "$ROOT/helixc/bootstrap/parser.hx" 2>/dev/null | sha256sum | cut -d" " -f1)
if [ -s "$CACHE/newdrv.bin" ] && [ "$(cat "$CACHE/compiler.sha" 2>/dev/null)" = "$CUR" ]; then DRV="$CACHE/newdrv.bin"
else bad "no fast-iter driver cache (pre-seed $CACHE/newdrv.bin + compiler.sha)"; echo BITNET_LAYER_FAIL; exit 1; fi
cp "$KERN" /tmp/kernel_in.hx; rm -f /tmp/out.ptx; "$DRV" >/dev/null 2>&1 || true
[ -s /tmp/out.ptx ] || { bad "PTX not emitted"; echo BITNET_LAYER_FAIL; exit 1; }
cp /tmp/out.ptx /tmp/bnl.ptx
gcc "$ROOT/helixc/runtime/cuda_launch.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/bnl_cl 2>/tmp/bnl_gcc.log || { bad "launcher build failed"; tail -6 /tmp/bnl_gcc.log >&2; echo BITNET_LAYER_FAIL; exit 1; }
echo "[bitnet_layer] POSITIVE (real q_proj through kovc kernel == host int matmul)"
/tmp/bnl_cl /tmp/bnl.ptx scaled_packed_ternary_matmul "$N" sptmatmul_real "$OD/packed.bin" "$OD/acts.bin" "$OD/expected.bin" "$M" "$K" "$N" || bad "real-data GPU check did not pass"
echo "[bitnet_layer] NC (comparator mutate -> must FAIL)"
if /tmp/bnl_cl /tmp/bnl.ptx scaled_packed_ternary_matmul "$N" sptmatmul_real "$OD/packed.bin" "$OD/acts.bin" "$OD/expected.bin" "$M" "$K" "$N" mutate >/dev/null 2>&1; then bad "comparator NC did NOT fail"; else echo "[bitnet_layer] NC correctly FAILED -- the exact check has teeth"; fi
echo "------------------------------------------------------------"
if [ "$OK" = 1 ]; then echo "BITNET_LAYER_GPU_PASS"; exit 0; else echo "BITNET_LAYER_FAIL"; exit 1; fi
