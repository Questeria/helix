#!/usr/bin/env bash
# talk_demo.sh -- the LIVE-CODING loop for the Anthropic talk.
#
#   bash scripts/talk_demo.sh prewarm   # BEFORE the talk: mint the from-raw driver + a green gate (~3-4 min)
#   bash scripts/talk_demo.sh gate      # the live loop: compile -> ptxas -> oracle parity (~15-25 s)
#   bash scripts/talk_demo.sh plant     # plant the REAL historical kovc-aliasing bug into gpu_silu_mul
#   bash scripts/talk_demo.sh restore   # restore the good kernel
#   bash scripts/talk_demo.sh status    # which kernel is in place + driver cache state
#
# THE BEAT: gate (green) -> plant -> gate (RED: silu_mul == -reference for g<0, the negative
# control still bites) -> restore (or live-fix) -> gate (green). The planted bug is the REAL
# bug found during G-L0 gating (commit 93d6021): `let mut amag = gi` made kovc alias amag
# onto gi and clobber it -- compile AND ptxas pass; ONLY numerical parity catches it.
# Run under WSL as a file. GPU must be free (stop the chat demo first).
set -u
ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"
[ -d "$ROOT/helixc/examples" ] || ROOT="/mnt/c/Projects/Kovostov-Native"
K="$ROOT/helixc/examples/gpu_silu_mul_kernel.hx"
GOOD="$ROOT/scripts/talk_assets/gpu_silu_mul_GOOD.hx"
BUGGY="$ROOT/scripts/talk_assets/gpu_silu_mul_BUGGY.hx"
GATE="$ROOT/scripts/llama_ops_parity.sh"

case "${1:-}" in
  prewarm)
    echo "== prewarm: full mint + a green baseline gate =="
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . \
      && { echo "GPU BUSY -- stop the chat demo/server first"; exit 1; }
    cp "$GOOD" "$K"
    tr -d '\r' < "$GATE" > /tmp/talk_gate.sh
    REMINT=1 bash /tmp/talk_gate.sh && echo "PREWARM_OK (driver cached; 'gate' is now ~15-25s)" || { echo "PREWARM_FAIL"; exit 1; }
    ;;
  gate)
    tr -d '\r' < "$GATE" > /tmp/talk_gate.sh
    REMINT=0 bash /tmp/talk_gate.sh
    ;;
  plant)
    cp "$BUGGY" "$K"
    echo "PLANTED: the real kovc-aliasing bug is now in $K"
    echo "  (let mut amag = gi; if gi<0 { amag = 0-gi } -- aliases amag onto gi; silu negates for g<0)"
    ;;
  restore)
    cp "$GOOD" "$K"
    echo "RESTORED: the verified kernel is back in $K"
    ;;
  status)
    if cmp -s "$K" "$GOOD"; then echo "kernel: GOOD (verified)";
    elif cmp -s "$K" "$BUGGY"; then echo "kernel: BUGGY (planted)";
    else echo "kernel: HAND-EDITED (live-coded state)"; fi
    DRVP="${LLAMA_DRV:-$HOME/gpt2_ext4/llama_kovc_drv.bin}"
    [ -x "$DRVP" ] && echo "driver: cached ($(stat -c%s "$DRVP") B) -- gate is fast" || echo "driver: NOT cached -- run prewarm"
    nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | grep -q . && echo "GPU: BUSY" || echo "GPU: free"
    ;;
  *)
    echo "usage: $0 prewarm|gate|plant|restore|status"; exit 2;;
esac
