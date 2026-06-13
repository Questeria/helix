#!/usr/bin/env bash
# fast_iter.sh -- FAST iteration on a single GPU kernel: emit its PTX with a CACHED kovc PTX
# driver (kept on the WSL ext4 fs so it survives /tmp recycles), rebuilding the driver ONLY
# when the compiler sources (kovc.hx/parser.hx) change. Optionally runs cuda_launch on the
# emitted PTX (the launcher is also cached + rebuilt when cuda_launch.c changes).
#
#   bash scripts/fast_iter.sh <kernel.hx>
#   bash scripts/fast_iter.sh <kernel.hx> <kname> <op> <M> <K> <N> [mutate]
#
# THIS IS FOR ITERATION ONLY -- it is NOT a substitute for scripts/gate_kovc.sh. The FULL gate
# (self-host fixpoint K2==K3==K4 + corpus + PTX regressions + check_err) MUST run green before
# EVERY commit. fast_iter just skips the ~28-min fixpoint rebuild when you're only changing a
# KERNEL or a corpus FIXTURE (not the compiler): ~seconds per emit once the driver is cached.
set -u
REPO=/mnt/c/Projects/Kovostov-Native
CACHE="$HOME/.helix_fastiter"          # ext4 (NOT /mnt/c, NOT /tmp): fast + persistent
mkdir -p "$CACHE"
DRV="$CACHE/newdrv.bin"
STAMP="$CACHE/compiler.sha"
KERN="${1:?usage: fast_iter.sh <kernel.hx> [<kname> <op> <M> <K> <N> [mutate]]}"
[ -f "$KERN" ] || { echo "[fast_iter] kernel not found: $KERN"; exit 1; }
# Driver identity = sha of the two compiler sources (the only inputs that change the driver).
CUR=$(cat "$REPO/helixc/bootstrap/kovc.hx" "$REPO/helixc/bootstrap/parser.hx" 2>/dev/null | sha256sum | cut -d' ' -f1)
if [ ! -s "$DRV" ] || [ "$(cat "$STAMP" 2>/dev/null)" != "$CUR" ]; then
  echo "[fast_iter] driver stale/missing -> rebuilding once (~4min)"
  ( cd "$REPO/stage0/helixc-bootstrap" && bash assemble_k1.sh >/dev/null 2>&1 && chmod +x seed.bin && ./seed.bin k1ptxdrv.hx "$DRV" >/dev/null 2>&1 )
  if [ ! -s "$DRV" ]; then echo "[fast_iter] DRIVER BUILD FAILED"; exit 1; fi
  chmod +x "$DRV"; echo "$CUR" > "$STAMP"
  echo "[fast_iter] driver cached ($(stat -c%s "$DRV") B)"
else
  echo "[fast_iter] reusing cached driver (compiler unchanged)"
fi
# Emit the kernel PTX (the driver reads /tmp/kernel_in.hx -> writes /tmp/out.ptx).
cp "$KERN" /tmp/kernel_in.hx; rm -f /tmp/out.ptx; "$DRV" >/dev/null 2>&1 || true
if [ ! -s /tmp/out.ptx ]; then echo "[fast_iter] NO PTX emitted for $(basename "$KERN")"; exit 1; fi
cp /tmp/out.ptx "$CACHE/last.ptx"
echo "[fast_iter] emitted $(stat -c%s "$CACHE/last.ptx") B PTX from $(basename "$KERN") -> $CACHE/last.ptx"
# Optional GPU check via cuda_launch (cheap rebuild -- always fresh from source).
if [ "$#" -ge 2 ]; then
  shift
  CL="$CACHE/cl"
  gcc "$REPO/helixc/runtime/cuda_launch.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o "$CL" 2>/tmp/fi_cl.log || { echo "[fast_iter] launcher build failed:"; tail -5 /tmp/fi_cl.log; exit 1; }
  echo "[fast_iter] cuda_launch $* :"
  "$CL" "$CACHE/last.ptx" "$@"; echo "[fast_iter] cuda_launch rc=$?"
fi
