#!/usr/bin/env bash
# bitnet_conversion_receipt_check.sh (v1.9 P4b): emit + re-derive-check the BitNet-2B ternary CONVERSION
# RECEIPT. emit -> check PASS (re-derives the Merkle root over the 210 packed ternary tensors + the kernel
# sha + the forward result, byte-identical) + NC (tamper the cert -> FAIL). numpy-only; needs the checkpoint.
set -u
ROOT="${HELIX_SRC:-}"; if [ -z "$ROOT" ]; then ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; fi
[ -d "$ROOT/helixc" ] || ROOT="/mnt/c/Projects/Kovostov-Native"; cd "$ROOT"
MD="${BITNET_DIR:-$HOME/bitnet-2b}"; PY="${HELIX_PY:-$HOME/alt-eval-venv/bin/python}"; OD="${BN_CONV_DIR:-$HOME/bn_conv}"
echo "==== Helix v1.9 P4b: BitNet ternary CONVERSION receipt ===="
if [ ! -s "$MD/model.safetensors" ] || ! "$PY" -c "import numpy" 2>/dev/null; then echo "[conv] SKIP: model/numpy absent"; echo "CONVERSION_RECEIPT_SKIP"; exit 0; fi
mkdir -p "$OD"; OK=1; bad(){ echo "[conv] FAIL: $*" >&2; OK=0; }
D="$ROOT/scripts/bitnet_conversion_receipt.py"
"$PY" "$D" --emit "$MD" "$ROOT" "$OD/cert.txt" "$HOME" 2>&1 | grep CONVERSION_RECEIPT_EMIT || { bad emit; echo CONVERSION_RECEIPT_GATE_FAIL; exit 1; }
V=$("$PY" "$D" --check "$MD" "$ROOT" "$OD/cert.txt" "$HOME" 2>&1 | sed -n "s/.*-> \(CONVERSION_RECEIPT_PASS\|CONVERSION_RECEIPT_FAIL\)$/\1/p" | tail -1)
[ "$V" = CONVERSION_RECEIPT_PASS ] && echo "[conv] POSITIVE re-derive -> PASS  OK" || bad "positive=$V"
cp "$OD/cert.txt" "$OD/cert_bad.txt"; sed -i "s/^merkle_root: ./merkle_root: 0/" "$OD/cert_bad.txt"
V=$("$PY" "$D" --check "$MD" "$ROOT" "$OD/cert_bad.txt" "$HOME" 2>&1 | sed -n "s/.*-> \(CONVERSION_RECEIPT_PASS\|CONVERSION_RECEIPT_FAIL\)$/\1/p" | tail -1)
[ "$V" = CONVERSION_RECEIPT_FAIL ] && echo "[conv] NC tampered-cert -> FAIL  OK" || bad "NC=$V (want FAIL)"
echo "----"; if [ "$OK" = 1 ]; then echo "CONVERSION_RECEIPT_GATE_PASS"; exit 0; else echo "CONVERSION_RECEIPT_GATE_FAIL"; exit 1; fi
