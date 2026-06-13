#!/usr/bin/env bash
# gpu_packed_ternary_check.sh (v1.5 S0 increment 3): GPU-HARDWARE exact-integer verification of
# the PACKED ternary matmul on a real CUDA device (RTX 3070, sm_86).
#
# Proves the 2-bit PACKED ternary representation (15 trits / i32 word, base-4 codes, ~15x storage
# reduction) is UNPACKED + computed correctly ON THE DEVICE via division (the @kernel path emits
# div.s32, not bitwise). The kovc-emitted packed_ternary_matmul kernel is launched on packed
# ternary weights {-1,0,+1} x signed int activations; its int32 result is compared ELEMENT-WISE
# EXACTLY (no tolerance) against an independent UNPACKED CPU integer reference in cuda_launch.c's
# 'ptmatmul' mode, which also reports the measured packed-vs-unpacked footprint.
#
# Two load-bearing negative controls: [D] comparator mutate -> the exact check MUST fail; [E] a
# kernel-source corruption (the output store acc -> acc+1, a data-INDEPENDENT off-by-one,
# re-emitted from a /tmp copy so the committed kernel is never edited) -> the check MUST fail.
#
# Run under WSL (CUDA 12.0+12.8): bash scripts/gpu_packed_ternary_check.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EX="$ROOT/helixc/examples"
KERN="$EX/packed_ternary_matmul_kernel.hx"
BS="$ROOT/stage0/helixc-bootstrap"
M=16; K=15; N=16   # K must be a multiple of 15 (15 trits per packed word)
OK=1
say(){ echo "[gpu_packed_ternary] $*"; }
bad(){ echo "[gpu_packed_ternary] *** FAIL: $*" >&2; OK=0; }
emit_ptx(){ cp "$1" /tmp/kernel_in.hx; rm -f /tmp/out.ptx; "$DRV" >/dev/null 2>&1 || true; }

echo "============================================================"
echo " Helix v1.5 S0 inc3: packed_ternary_matmul GPU exact-int check"
echo "============================================================"

# --- [A] obtain the PTX driver (reuse the fast_iter ext4 cache when the compiler is unchanged,
#         else mint from raw -- self-contained on a fresh clone) + emit the packed kernel PTX ---
say "[A] obtain PTX driver + emit packed_ternary_matmul PTX"
CACHE="$HOME/.helix_fastiter"
CUR=$(cat "$ROOT/helixc/bootstrap/kovc.hx" "$ROOT/helixc/bootstrap/parser.hx" 2>/dev/null | sha256sum | cut -d' ' -f1)
if [ -s "$CACHE/newdrv.bin" ] && [ "$(cat "$CACHE/compiler.sha" 2>/dev/null)" = "$CUR" ]; then
  DRV="$CACHE/newdrv.bin"; say "    reusing cached driver (compiler unchanged)"
else
  say "    minting driver from raw (compiler changed / no cache; ~4min)"
  ( cd "$BS" && bash assemble_k1.sh >/tmp/gpt_asm.log 2>&1 )
  chmod +x "$BS/seed.bin" 2>/dev/null || true
  ( cd "$BS" && ./seed.bin k1ptxdrv.hx /tmp/gpt_newdrv.bin >/tmp/gpt_drv.log 2>&1 ) || true
  if [ ! -s /tmp/gpt_newdrv.bin ]; then bad "PTX driver not built"; echo "GPU_PACKED_TERNARY_FAIL"; exit 1; fi
  chmod +x /tmp/gpt_newdrv.bin; DRV=/tmp/gpt_newdrv.bin
  mkdir -p "$CACHE"; cp "$DRV" "$CACHE/newdrv.bin" 2>/dev/null && echo "$CUR" > "$CACHE/compiler.sha"
fi
emit_ptx "$KERN"
if [ ! -s /tmp/out.ptx ]; then bad "packed_ternary_matmul PTX not emitted"; echo "GPU_PACKED_TERNARY_FAIL"; exit 1; fi
cp /tmp/out.ptx /tmp/gpt_packed.ptx
say "    emitted packed PTX ($(wc -c < /tmp/gpt_packed.ptx) B)"

# --- [B] build the launcher (committed cuda_launch.c, -lcuda -lcublas -lm) ---
say "[B] build cuda_launch.c launcher"
gcc "$ROOT/helixc/runtime/cuda_launch.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/gpt_cl >/tmp/gpt_gcc.log 2>&1
if [ ! -s /tmp/gpt_cl ]; then bad "launcher build failed:"; tail -6 /tmp/gpt_gcc.log >&2; echo "GPU_PACKED_TERNARY_FAIL"; exit 1; fi

# --- [C] POSITIVE: GPU packed ternary matmul == unpacked CPU int reference, EXACT ---
say "[C] GPU exact-int verify (${M}x${K}x${N}, 15 trits/word)"
if /tmp/gpt_cl /tmp/gpt_packed.ptx packed_ternary_matmul "$N" ptmatmul "$M" "$K" "$N" >/tmp/gpt_pos.log 2>&1; then
  say "    $(grep -m1 'packed_ternary_matmul' /tmp/gpt_pos.log || echo 'PASS (rc=0)')"
else
  bad "GPU exact-int verify did NOT pass (rc!=0):"; tail -6 /tmp/gpt_pos.log >&2
fi

# --- [D] NEGATIVE CONTROL 1 (comparator): mutate one cell -> MUST fail ---
say "[D] negative control: comparator mutate -> must FAIL"
if /tmp/gpt_cl /tmp/gpt_packed.ptx packed_ternary_matmul "$N" ptmatmul "$M" "$K" "$N" mutate >/tmp/gpt_ncd.log 2>&1; then
  bad "comparator NC did NOT fail (mutated result still 'passed' -> comparator not load-bearing)"
else
  say "    comparator NC correctly FAILED (rc!=0) -- the exact-int check has teeth"
fi

# --- [E] NEGATIVE CONTROL 2 (kernel corruption): store acc -> acc+1 -> re-emit -> MUST fail ---
say "[E] negative control: corrupt kernel store -> re-emit -> must FAIL"
sed 's/c\[row \* nn + col\] = acc/c[row * nn + col] = acc + 1/' "$KERN" > /tmp/gpt_kern_bad.hx
if cmp -s /tmp/gpt_kern_bad.hx "$KERN"; then
  bad "kernel-corruption NC is a NO-OP (sed pattern did not match the store line)"
else
  emit_ptx /tmp/gpt_kern_bad.hx
  if [ ! -s /tmp/out.ptx ]; then
    bad "corrupted kernel emitted no PTX -- cannot run kernel-corruption NC"
  else
    cp /tmp/out.ptx /tmp/gpt_packed_bad.ptx
    if /tmp/gpt_cl /tmp/gpt_packed_bad.ptx packed_ternary_matmul "$N" ptmatmul "$M" "$K" "$N" >/tmp/gpt_nce.log 2>&1; then
      bad "kernel-corruption NC did NOT fail (corrupted kernel still matched the ref -> not load-bearing)"
    else
      say "    kernel-corruption NC correctly FAILED (rc!=0) -- the kovc-emitted packed kernel is load-bearing"
    fi
  fi
fi
# NOTE: [E] never edits the committed kernel -- sed reads $KERN, writes /tmp; corrupted PTX from a /tmp copy.

echo "------------------------------------------------------------"
if [ "$OK" = "1" ]; then echo "PACKED_TERNARY_GPU_PASS"; exit 0; else echo "GPU_PACKED_TERNARY_FAIL"; exit 1; fi
