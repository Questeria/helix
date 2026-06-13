#!/usr/bin/env bash
# gpu_f16_check.sh (v1.5 S1): GPU-HARDWARE tolerance verification of the HALF-PRECISION
# (f16 storage, f32 accumulate) matmul on a real CUDA device (RTX 3070, sm_86).
#
# Proves the kovc @kernel PTX path emits a correct fp16 GEMM: ld.global.b16 -> cvt.f32.f16
# (load-narrow), mul.f32 + add.f32 (f32 accumulation), cvt.rn.f16.f32 -> st.global.b16
# (store-narrow), with a 2-byte element stride. The kovc-emitted naive_matmul_f16 kernel is
# launched on NON-DEGENERATE f16 data (a non-f16-exact x0.1/x0.3 scale -> the result genuinely
# rounds), and compared in cuda_launch.c's 'hgemm' mode against an independent f32-accum CPU
# reference that consumes the SAME rounded f16 inputs. f16 rounds (and the GPU may FMA-contract),
# so the compare is DUAL-bound tolerance: a cell passes if within EITHER abs 1e-3 OR rel 1e-2.
#
# Two load-bearing negative controls (REAL exit codes -- this runs as a FILE, so `if cmd` sees
# the true rc; inline `wsl.exe bash -c "...; echo $?"` does NOT and must never be trusted):
#   [D] comparator MAGNITUDE-SCALED mutate (got += 0.5 + 0.1*|got|, exceeds BOTH bounds at any
#       magnitude; a bare +1 would be vacuous on large cells) -> the tolerance check MUST fail.
#   [E] kernel-source corruption (store acc -> acc + 1, a data-INDEPENDENT +1.0 >> tolerance,
#       re-emitted from a /tmp copy so the committed kernel is never edited) -> MUST fail.
#
# Run under WSL (CUDA 12.0+12.8): bash scripts/gpu_f16_check.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EX="$ROOT/helixc/examples"
KERN="$EX/naive_matmul_f16_kernel.hx"
BS="$ROOT/stage0/helixc-bootstrap"
M=16; K=16; N=16
OK=1
say(){ echo "[gpu_f16] $*"; }
bad(){ echo "[gpu_f16] *** FAIL: $*" >&2; OK=0; }
emit_ptx(){ cp "$1" /tmp/kernel_in.hx; rm -f /tmp/out.ptx; "$DRV" >/dev/null 2>&1 || true; }

echo "============================================================"
echo " Helix v1.5 S1: naive_matmul_f16 GPU f16-IO/f32-acc tolerance check"
echo "============================================================"

# --- [A] obtain the PTX driver (reuse the fast_iter ext4 cache when the compiler is unchanged,
#         else mint from raw -- self-contained on a fresh clone) + emit the f16 kernel PTX ---
say "[A] obtain PTX driver + emit naive_matmul_f16 PTX"
CACHE="$HOME/.helix_fastiter"
CUR=$(cat "$ROOT/helixc/bootstrap/kovc.hx" "$ROOT/helixc/bootstrap/parser.hx" 2>/dev/null | sha256sum | cut -d' ' -f1)
if [ -s "$CACHE/newdrv.bin" ] && [ "$(cat "$CACHE/compiler.sha" 2>/dev/null)" = "$CUR" ]; then
  DRV="$CACHE/newdrv.bin"; say "    reusing cached driver (compiler unchanged)"
else
  say "    minting driver from raw (compiler changed / no cache; ~4min)"
  ( cd "$BS" && bash assemble_k1.sh >/tmp/f16_asm.log 2>&1 )
  chmod +x "$BS/seed.bin" 2>/dev/null || true
  ( cd "$BS" && ./seed.bin k1ptxdrv.hx /tmp/f16_newdrv.bin >/tmp/f16_drv.log 2>&1 ) || true
  if [ ! -s /tmp/f16_newdrv.bin ]; then bad "PTX driver not built"; echo "GPU_F16_FAIL"; exit 1; fi
  chmod +x /tmp/f16_newdrv.bin; DRV=/tmp/f16_newdrv.bin
  mkdir -p "$CACHE"; cp "$DRV" "$CACHE/newdrv.bin" 2>/dev/null && echo "$CUR" > "$CACHE/compiler.sha"
fi
emit_ptx "$KERN"
if [ ! -s /tmp/out.ptx ]; then bad "naive_matmul_f16 PTX not emitted"; echo "GPU_F16_FAIL"; exit 1; fi
cp /tmp/out.ptx /tmp/f16_good.ptx
say "    emitted f16 PTX ($(wc -c < /tmp/f16_good.ptx) B)"

# --- [B] build the launcher (committed cuda_launch.c, -lcuda -lcublas -lm) ---
say "[B] build cuda_launch.c launcher"
gcc "$ROOT/helixc/runtime/cuda_launch.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/f16_cl >/tmp/f16_gcc.log 2>&1
if [ ! -s /tmp/f16_cl ]; then bad "launcher build failed:"; tail -6 /tmp/f16_gcc.log >&2; echo "GPU_F16_FAIL"; exit 1; fi

# --- [C] POSITIVE: GPU f16 matmul within dual-bound tolerance of the f32-accum CPU reference ---
say "[C] GPU f16 tolerance verify (${M}x${K}x${N}, f16-IO/f32-acc)"
if /tmp/f16_cl /tmp/f16_good.ptx naive_matmul_f16 "$N" hgemm "$M" "$K" "$N" >/tmp/f16_pos.log 2>&1; then
  say "    $(grep -m1 'naive_matmul_f16' /tmp/f16_pos.log || echo 'PASS (rc=0)')"
else
  bad "GPU f16 verify did NOT pass (rc!=0):"; tail -6 /tmp/f16_pos.log >&2
fi

# --- [D] NEGATIVE CONTROL 1 (comparator): magnitude-scaled mutate -> MUST fail ---
say "[D] negative control: comparator magnitude-scaled mutate -> must FAIL"
if /tmp/f16_cl /tmp/f16_good.ptx naive_matmul_f16 "$N" hgemm "$M" "$K" "$N" mutate >/tmp/f16_ncd.log 2>&1; then
  bad "comparator NC did NOT fail (mutated result still within tolerance -> comparator not load-bearing)"
else
  say "    comparator NC correctly FAILED (rc!=0) -- the dual-bound tolerance has teeth"
fi

# --- [E] NEGATIVE CONTROL 2 (kernel corruption): store acc -> acc + 1 -> re-emit -> MUST fail ---
say "[E] negative control: corrupt kernel store -> re-emit -> must FAIL"
sed 's/c\[row \* nn + col\] = acc/c[row * nn + col] = acc + 1/' "$KERN" > /tmp/f16_kern_bad.hx
if cmp -s /tmp/f16_kern_bad.hx "$KERN"; then
  bad "kernel-corruption NC is a NO-OP (sed pattern did not match the store line)"
else
  emit_ptx /tmp/f16_kern_bad.hx
  if [ ! -s /tmp/out.ptx ]; then
    bad "corrupted kernel emitted no PTX -- cannot run kernel-corruption NC"
  else
    cp /tmp/out.ptx /tmp/f16_bad.ptx
    if /tmp/f16_cl /tmp/f16_bad.ptx naive_matmul_f16 "$N" hgemm "$M" "$K" "$N" >/tmp/f16_nce.log 2>&1; then
      bad "kernel-corruption NC did NOT fail (corrupted kernel still within tolerance -> not load-bearing)"
    else
      say "    kernel-corruption NC correctly FAILED (rc!=0) -- the kovc-emitted f16 kernel is load-bearing"
    fi
  fi
fi
# NOTE: [E] never edits the committed kernel -- sed reads $KERN, writes /tmp; corrupted PTX from a /tmp copy.

echo "------------------------------------------------------------"
if [ "$OK" = "1" ]; then echo "F16_GPU_PASS"; exit 0; else echo "GPU_F16_FAIL"; exit 1; fi
