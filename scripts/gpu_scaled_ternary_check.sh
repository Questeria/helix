#!/usr/bin/env bash
# gpu_scaled_ternary_check.sh (v1.9 P1): GPU-HARDWARE exact verification of the SCALED packed ternary
# matmul -- the v1.5 S0 packed_ternary_matmul (15 trits/i32 word, on-device div-unpack, exact integer
# accumulate) with a per-OUTPUT-ROW f32 scale applied at the end: c[r,col] = __gpu_i2f(int_dot)*sc[r]
# (the BitNet b1.58 dequant shape). The kovc-emitted scaled_packed_ternary_matmul kernel is launched
# on packed ternary weights {-1,0,+1} x signed int activations x a per-row f32 scale; its f32 result
# is compared ELEMENT-WISE EXACTLY (==) against an independent host reference (float)int_ref*scale[r]
# in cuda_launch.c's 'sptmatmul' mode (exact because the int accumulate is small -> __gpu_i2f exact +
# one mul.f32, no FMA -> GPU==CPU bit-for-bit).
#
# Two load-bearing negative controls: [D] comparator mutate -> the exact check MUST fail; [E] a
# kernel-source corruption (the scaled store, +1.0) re-emitted from a /tmp copy -> the check MUST fail
# (the committed kernel is never edited). Plus a byte-gate of the emitted PTX vs the committed .ref.ptx.
#
# Run under WSL (CUDA 12.0+12.8): bash scripts/gpu_scaled_ternary_check.sh
set -u
ROOT="${HELIX_SRC:-}"; if [ -z "$ROOT" ]; then ROOT="$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)"; fi
[ -d "$ROOT/helixc/examples" ] || ROOT="/mnt/c/Projects/Kovostov-Native"   # robust when run as a CR-stripped /tmp copy
cd "$ROOT"
EX="$ROOT/helixc/examples"
KERN="$EX/scaled_packed_ternary_matmul_kernel.hx"
REF="$EX/scaled_packed_ternary_matmul_kernel.ref.ptx"
BS="$ROOT/stage0/helixc-bootstrap"
M=16; K=15; N=16   # K must be a multiple of 15 (15 trits per packed word)
OK=1
say(){ echo "[gpu_scaled_ternary] $*"; }
bad(){ echo "[gpu_scaled_ternary] *** FAIL: $*" >&2; OK=0; }
emit_ptx(){ cp "$1" /tmp/kernel_in.hx; rm -f /tmp/out.ptx; "$DRV" >/dev/null 2>&1 || true; }

echo "============================================================"
echo " Helix v1.9 P1: scaled_packed_ternary_matmul GPU exact check"
echo "============================================================"

# --- [A] PTX driver (reuse the fast_iter cache when the compiler is unchanged, else mint from raw) ---
say "[A] obtain PTX driver + emit scaled_packed_ternary_matmul PTX"
CACHE="$HOME/.helix_fastiter"
CUR=$(cat "$ROOT/helixc/bootstrap/kovc.hx" "$ROOT/helixc/bootstrap/parser.hx" 2>/dev/null | sha256sum | cut -d' ' -f1)
if [ -s "$CACHE/newdrv.bin" ] && [ "$(cat "$CACHE/compiler.sha" 2>/dev/null)" = "$CUR" ]; then
  DRV="$CACHE/newdrv.bin"; say "    reusing cached driver (compiler unchanged)"
else
  say "    minting driver from raw (compiler changed / no cache; ~4min)"
  ( cd "$BS" && bash assemble_k1.sh >/tmp/sct_asm.log 2>&1 )
  chmod +x "$BS/seed.bin" 2>/dev/null || true
  ( cd "$BS" && ./seed.bin k1ptxdrv.hx /tmp/sct_newdrv.bin >/tmp/sct_drv.log 2>&1 ) || true
  if [ ! -s /tmp/sct_newdrv.bin ]; then bad "PTX driver not built"; echo "GPU_SCALED_TERNARY_FAIL"; exit 1; fi
  chmod +x /tmp/sct_newdrv.bin; DRV=/tmp/sct_newdrv.bin
  mkdir -p "$CACHE"; cp "$DRV" "$CACHE/newdrv.bin" 2>/dev/null && echo "$CUR" > "$CACHE/compiler.sha"
fi
emit_ptx "$KERN"
if [ ! -s /tmp/out.ptx ]; then bad "scaled_packed_ternary_matmul PTX not emitted"; echo "GPU_SCALED_TERNARY_FAIL"; exit 1; fi
cp /tmp/out.ptx /tmp/sct.ptx
say "    emitted scaled PTX ($(wc -c < /tmp/sct.ptx) B)"
# byte-gate vs the committed .ref.ptx (provenance: re-emit must match the committed reference)
if [ -s "$REF" ]; then
  if cmp -s /tmp/sct.ptx "$REF"; then say "    PTX byte-identical to committed .ref.ptx"; else bad "emitted PTX != committed .ref.ptx -- re-mint+re-commit the ref with a reason"; fi
else
  say "    (no committed .ref.ptx yet -- skipping byte-gate)"
fi

# --- [B] build the launcher (committed cuda_launch.c, -lcuda -lcublas -lm) ---
say "[B] build cuda_launch.c launcher"
gcc "$ROOT/helixc/runtime/cuda_launch.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/sct_cl >/tmp/sct_gcc.log 2>&1
if [ ! -s /tmp/sct_cl ]; then bad "launcher build failed:"; tail -6 /tmp/sct_gcc.log >&2; echo "GPU_SCALED_TERNARY_FAIL"; exit 1; fi

# --- [C] POSITIVE: GPU scaled ternary == host ref, EXACT ---
say "[C] GPU exact verify (${M}x${K}x${N}, 15 trits/word + per-row f32 scale)"
if /tmp/sct_cl /tmp/sct.ptx scaled_packed_ternary_matmul "$N" sptmatmul "$M" "$K" "$N" >/tmp/sct_pos.log 2>&1; then
  say "    $(grep -m1 'scaled_packed_ternary_matmul' /tmp/sct_pos.log || echo 'PASS (rc=0)')"
else
  bad "GPU exact verify did NOT pass (rc!=0):"; tail -6 /tmp/sct_pos.log >&2
fi

# --- [D] NEGATIVE CONTROL 1 (comparator): mutate one cell -> MUST fail ---
say "[D] negative control: comparator mutate -> must FAIL"
if /tmp/sct_cl /tmp/sct.ptx scaled_packed_ternary_matmul "$N" sptmatmul "$M" "$K" "$N" mutate >/tmp/sct_ncd.log 2>&1; then
  bad "comparator NC did NOT fail (mutated result still 'passed')"
else
  say "    comparator NC correctly FAILED (rc!=0) -- the exact check has teeth"
fi

# --- [E] NEGATIVE CONTROL 2 (kernel corruption): scaled store +1.0 -> re-emit -> MUST fail ---
say "[E] negative control: corrupt scaled store -> re-emit -> must FAIL"
sed 's#c\[row \* nn + col\] = __gpu_i2f(acc) \* sc\[row\]#c[row * nn + col] = __gpu_i2f(acc) * sc[row] + 1.0#' "$KERN" > /tmp/sct_kern_bad.hx
if cmp -s /tmp/sct_kern_bad.hx "$KERN"; then
  bad "kernel-corruption NC is a NO-OP (sed pattern did not match the store line)"
else
  emit_ptx /tmp/sct_kern_bad.hx
  if [ ! -s /tmp/out.ptx ]; then
    bad "corrupted kernel emitted no PTX -- cannot run kernel-corruption NC"
  else
    cp /tmp/out.ptx /tmp/sct_bad.ptx
    if /tmp/sct_cl /tmp/sct_bad.ptx scaled_packed_ternary_matmul "$N" sptmatmul "$M" "$K" "$N" >/tmp/sct_nce.log 2>&1; then
      bad "kernel-corruption NC did NOT fail (corrupted kernel still matched the ref)"
    else
      say "    kernel-corruption NC correctly FAILED (rc!=0) -- the kovc-emitted scaled kernel is load-bearing"
    fi
  fi
fi
# NOTE: [E] never edits the committed kernel -- sed reads $KERN, writes /tmp; corrupted PTX from a /tmp copy.

echo "------------------------------------------------------------"
if [ "$OK" = "1" ]; then echo "SCALED_TERNARY_GPU_PASS"; exit 0; else echo "GPU_SCALED_TERNARY_FAIL"; exit 1; fi
