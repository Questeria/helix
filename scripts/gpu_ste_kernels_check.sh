#!/usr/bin/env bash
# gpu_ste_kernels_check.sh (v1.9 P5): GPU element-exact verify the 3 STE QAT kernels (ternarize_dequant,
# row_abs_mean, ste_mask) -- each emitted from the kovc driver, BYTE-GATED vs its committed .ref.ptx, then
# run on a real device vs an INDEPENDENT host ref (EXACT ==; the kernels are pure div/compare/mul/in-order
# -sum, no FMA) + a comparator-mutate NC per kernel. NO kovc.hx edit (all @kernel-path).
set -u
ROOT="${HELIX_SRC:-}"; if [ -z "$ROOT" ]; then ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; fi
[ -d "$ROOT/helixc/examples" ] || ROOT="/mnt/c/Projects/Kovostov-Native"
cd "$ROOT"; EX="$ROOT/helixc/examples"
OK=1; say(){ echo "[ste] $*"; }; bad(){ echo "[ste] *** FAIL: $*" >&2; OK=0; }
echo "==== Helix v1.9 P5: STE QAT kernels GPU exact check ===="
CACHE="$HOME/.helix_fastiter"; CUR=$(cat "$ROOT/helixc/bootstrap/kovc.hx" "$ROOT/helixc/bootstrap/parser.hx" 2>/dev/null | sha256sum | cut -d" " -f1)
if [ -s "$CACHE/newdrv.bin" ] && [ "$(cat "$CACHE/compiler.sha" 2>/dev/null)" = "$CUR" ]; then DRV="$CACHE/newdrv.bin"; else bad "no fast-iter driver cache"; echo STE_KERNELS_GPU_FAIL; exit 1; fi
CL="$HOME/ste_cl"
gcc "$ROOT/helixc/runtime/cuda_launch.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o "$CL" 2>"$HOME/ste_gcc.log" || { bad "launcher build"; tail -6 "$HOME/ste_gcc.log" >&2; echo STE_KERNELS_GPU_FAIL; exit 1; }
emit(){ cp "$1" /tmp/kernel_in.hx; rm -f /tmp/out.ptx; "$DRV" >/dev/null 2>&1 || true; }
ROWS=8; COLS=16
for k in ternarize_dequant row_abs_mean ste_mask; do
  say "[$k]"
  emit "$EX/${k}_kernel.hx"
  if [ ! -s /tmp/out.ptx ]; then bad "$k PTX not emitted"; continue; fi
  PTX="$HOME/ste_${k}.ptx"; cp /tmp/out.ptx "$PTX"
  if cmp -s "$PTX" "$EX/${k}_kernel.ref.ptx"; then say "    PTX == committed .ref.ptx"; else bad "$k emitted PTX != committed .ref.ptx"; fi
  if "$CL" "$PTX" "$k" "$COLS" ste_check "$ROWS" "$COLS" >"$HOME/ste_${k}.log" 2>&1; then say "    $(grep -m1 ste_check "$HOME/ste_${k}.log")"; else bad "$k GPU exact did not pass"; tail -4 "$HOME/ste_${k}.log" >&2; fi
  if "$CL" "$PTX" "$k" "$COLS" ste_check "$ROWS" "$COLS" mutate >"$HOME/ste_${k}_nc.log" 2>&1; then bad "$k comparator NC did NOT fail"; else say "    NC (mutate) correctly FAILED"; fi
done
echo "----"; if [ "$OK" = 1 ]; then echo "STE_KERNELS_GPU_PASS"; exit 0; else echo "STE_KERNELS_GPU_FAIL"; exit 1; fi
