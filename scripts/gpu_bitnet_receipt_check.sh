#!/usr/bin/env bash
# gpu_bitnet_receipt_check.sh (v1.9 P4): a re-derivable FREIVALDS receipt over a REAL BitNet b1.58
# BitLinear layer (layer-0 q_proj). Dumps the real ternary W + int8 activation X + C=W*X, emits a receipt
# (receipt_emit_real; seed sentinel 0xFFFFFFFF = real data in .wbytes/.xbytes/.cbytes side-files), and the
# INDEPENDENT checker (receipt_check) re-derives the Fiat-Shamir challenges itself + Freivalds-verifies
# C==W*X over F_p (p=2^31-1, soundness 2^-62). NC: a forged C (mutate) MUST be rejected via CHECK2.
# Host-side (no GPU launch); needs the BitNet checkpoint + numpy. Run: HELIX_SRC=<root> bash scripts/gpu_bitnet_receipt_check.sh
set -u
ROOT="${HELIX_SRC:-}"; if [ -z "$ROOT" ]; then ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; fi
[ -d "$ROOT/helixc/runtime" ] || ROOT="/mnt/c/Projects/Kovostov-Native"
cd "$ROOT"
MODEL="${BITNET_MODEL:-$HOME/bitnet-2b/model.safetensors}"; PY="${HELIX_PY:-$HOME/alt-eval-venv/bin/python}"; OD="${BITNET_RCPT_DIR:-$HOME/bn_receipt}"
echo "==== Helix v1.9 P4: BitNet BitLinear Freivalds receipt (real layer) ===="
if [ ! -s "$MODEL" ] || ! "$PY" -c "import numpy" 2>/dev/null; then echo "[receipt] SKIP: BitNet model/numpy absent"; echo "BITNET_RECEIPT_SKIP"; exit 0; fi
OK=1; bad(){ echo "[receipt] *** FAIL: $*" >&2; OK=0; }
gcc "$ROOT/helixc/runtime/cuda_launch.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/rcp_cl 2>/tmp/rcp_gcc.log || { bad "build"; tail -6 /tmp/rcp_gcc.log >&2; echo BITNET_RECEIPT_FAIL; exit 1; }
CL=/tmp/rcp_cl
[ "$("$CL" x x 0 sha256_selftest 2>&1 | sed -n "s/.*-> \(PASS\|FAIL\)$/\1/p" | tail -1)" = PASS ] || { bad "sha256 KAT"; echo BITNET_RECEIPT_FAIL; exit 1; }
mkdir -p "$OD"
"$PY" "$ROOT/scripts/bitnet_receipt_dump.py" "$MODEL" "model.layers.0.self_attn.q_proj.weight" "$OD" 2>&1 | grep RECEIPT_DUMP || { bad "dump"; echo BITNET_RECEIPT_FAIL; exit 1; }
read M K N < "$OD/rcdims.txt"; echo "[receipt] real q_proj M=$M K=$K N=$N"
vget(){ "$CL" x x 0 receipt_check "$1" 2>&1 | sed -n "s/.*-> \(RECEIPT_PASS\|RECEIPT_FAIL\)$/\1/p" | tail -1; }
cget(){ "$CL" x x 0 receipt_check "$1" 2>&1 | sed -n "s/.*REJECT=\([A-Z0-9_]*\).*/\1/p" | tail -1; }
"$CL" x x 0 receipt_emit_real "$OD/rcW.bin" "$OD/rcX.bin" "$OD/rcC.bin" "$M" "$K" "$N" "$OD/rc.txt" >/dev/null 2>&1
V=$(vget "$OD/rc.txt"); [ "$V" = RECEIPT_PASS ] && echo "[receipt] POSITIVE genuine -> RECEIPT_PASS  OK" || bad "positive=$V"
"$CL" x x 0 receipt_emit_real "$OD/rcW.bin" "$OD/rcX.bin" "$OD/rcC.bin" "$M" "$K" "$N" "$OD/rcbad.txt" mutate >/dev/null 2>&1
V=$(vget "$OD/rcbad.txt"); C2=$(cget "$OD/rcbad.txt")
if [ "$V" = RECEIPT_FAIL ] && [ "$C2" = CHECK2 ]; then echo "[receipt] NC forged-C -> RECEIPT_FAIL via CHECK2  OK"; else bad "NC v=$V c=$C2 (want FAIL/CHECK2)"; fi
echo "----"; if [ "$OK" = 1 ]; then echo "BITNET_RECEIPT_PASS"; exit 0; else echo "BITNET_RECEIPT_FAIL"; exit 1; fi
