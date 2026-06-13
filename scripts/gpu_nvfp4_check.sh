#!/usr/bin/env bash
# gpu_nvfp4_check.sh (v1.5 S3): GPU-HARDWARE verification of the NVFP4 (OCP/NVIDIA) two-level-scaled
# 4-bit DEQUANT on a real CUDA device (RTX 3070, sm_86).
#
# Proves the kovc @kernel PTX path can unpack + dequantize NVFP4 end-to-end ON THE DEVICE: E2M1 4-bit
# elements (reused from S2) packed 7/i32 word (div-unpack) + an f32-literal decode * a host-collapsed
# effective f32 scale (FP8 E4M3 micro per 16-block * FP32 per-tensor). DEQUANT-only (the DoD S3 test is
# "verified dequant"; native FP4 MMA needs Blackwell -> DEFERRED). Output is f32, so the dequant is
# f32-EXACT vs the oracle. NO kovc.hx edit (rides the existing path: S0 div-unpack + the f32 emitters),
# so the self-host fixpoint stays cdcf8673.
#
# cuda_launch.c's 'nvfp4' mode compares the device output element-wise against an INDEPENDENT CPU
# dequant oracle (from-scratch E2M1 + E4M3 codec, nvfp4_codec_selftest-verified) within a TIGHT
# dual-bound tolerance (abs 1e-5 OR rel 1e-6 -- the dequant is a single f32 mul of identical operands).
#
# FOUR load-bearing negative controls (REAL exit codes -- run as a FILE, `if cmd`): [D] comparator
# MAGNITUDE-SCALED mutate; [G] a packed-WEIGHT nibble flip (the ELEMENT is load-bearing); [H] a per-block
# effective-SCALE flip (the TWO-LEVEL SCALE is load-bearing -- S3's novelty); [E] kernel-corruption
# (the dequant store * 2.0, DOUBLING -- magnitude-robust). Plus [F] the from-scratch-codec self-test.
#
# Run under WSL (CUDA 12.0+12.8): bash scripts/gpu_nvfp4_check.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EX="$ROOT/helixc/examples"
KERN="$EX/nvfp4_dequant_kernel.hx"
BS="$ROOT/stage0/helixc-bootstrap"
M=8; K=112; N=1   # K must be a multiple of 112 = LCM(7,16)
OK=1
say(){ echo "[gpu_nvfp4] $*"; }
bad(){ echo "[gpu_nvfp4] *** FAIL: $*" >&2; OK=0; }
emit_ptx(){ cp "$1" /tmp/kernel_in.hx; rm -f /tmp/out.ptx; "$DRV" >/dev/null 2>&1 || true; }

echo "============================================================"
echo " Helix v1.5 S3: nvfp4_dequant GPU two-level-dequant check"
echo "============================================================"

# --- [A] obtain the PTX driver (reuse the fast_iter ext4 cache -- S3 needs NO kovc.hx edit so the
#         cdcf8673 driver is valid; else mint from raw) + emit the kernel ---
say "[A] obtain PTX driver + emit nvfp4_dequant PTX"
CACHE="$HOME/.helix_fastiter"
CUR=$(cat "$ROOT/helixc/bootstrap/kovc.hx" "$ROOT/helixc/bootstrap/parser.hx" 2>/dev/null | sha256sum | cut -d' ' -f1)
if [ -s "$CACHE/newdrv.bin" ] && [ "$(cat "$CACHE/compiler.sha" 2>/dev/null)" = "$CUR" ]; then
  DRV="$CACHE/newdrv.bin"; say "    reusing cached driver (compiler unchanged -- no S3 kovc edit)"
else
  say "    minting driver from raw (compiler changed / no cache; ~4min)"
  ( cd "$BS" && bash assemble_k1.sh >/tmp/nv_asm.log 2>&1 )
  chmod +x "$BS/seed.bin" 2>/dev/null || true
  ( cd "$BS" && ./seed.bin k1ptxdrv.hx /tmp/nv_newdrv.bin >/tmp/nv_drv.log 2>&1 ) || true
  if [ ! -s /tmp/nv_newdrv.bin ]; then bad "PTX driver not built"; echo "GPU_NVFP4_FAIL"; exit 1; fi
  chmod +x /tmp/nv_newdrv.bin; DRV=/tmp/nv_newdrv.bin
  mkdir -p "$CACHE"; cp "$DRV" "$CACHE/newdrv.bin" 2>/dev/null && echo "$CUR" > "$CACHE/compiler.sha"
fi
emit_ptx "$KERN"
if [ ! -s /tmp/out.ptx ]; then bad "nvfp4_dequant PTX not emitted"; echo "GPU_NVFP4_FAIL"; exit 1; fi
cp /tmp/out.ptx /tmp/nv_good.ptx
say "    emitted NVFP4 PTX ($(wc -c < /tmp/nv_good.ptx) B)"

# --- [B] build the launcher ---
say "[B] build cuda_launch.c launcher"
gcc "$ROOT/helixc/runtime/cuda_launch.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/nv_cl >/tmp/nv_gcc.log 2>&1
if [ ! -s /tmp/nv_cl ]; then bad "launcher build failed:"; tail -6 /tmp/nv_gcc.log >&2; echo "GPU_NVFP4_FAIL"; exit 1; fi

# --- [C] POSITIVE: GPU NVFP4 dequant == CPU oracle (f32-exact, tight tol) ---
say "[C] GPU NVFP4 dequant verify (${M}x${K})"
if /tmp/nv_cl /tmp/nv_good.ptx nvfp4_dequant "$N" nvfp4 "$M" "$K" "$N" >/tmp/nv_pos.log 2>&1; then
  say "    $(grep -m1 'nvfp4_dequant' /tmp/nv_pos.log || echo 'PASS (rc=0)')"
else
  bad "GPU NVFP4 verify did NOT pass (rc!=0):"; tail -6 /tmp/nv_pos.log >&2
fi

# --- [D] NEGATIVE CONTROL 1 (comparator): magnitude-scaled mutate -> MUST fail ---
say "[D] negative control: comparator magnitude-scaled mutate -> must FAIL"
if /tmp/nv_cl /tmp/nv_good.ptx nvfp4_dequant "$N" nvfp4 "$M" "$K" "$N" mutate >/tmp/nv_ncd.log 2>&1; then
  bad "comparator NC did NOT fail (mutated result still within tolerance -> comparator not load-bearing)"
else
  say "    comparator NC correctly FAILED (rc!=0) -- the tight tolerance has teeth"
fi

# --- [G] NEGATIVE CONTROL 2 (weight): packed-weight nibble flip -> MUST fail (element load-bearing) ---
say "[G] negative control: packed-weight nibble flip -> must FAIL"
if /tmp/nv_cl /tmp/nv_good.ptx nvfp4_dequant "$N" nvfp4 "$M" "$K" "$N" wflip >/tmp/nv_ncg.log 2>&1; then
  bad "weight-flip NC did NOT fail (changed E2M1 code still matched -> the element is not load-bearing)"
else
  say "    weight-flip NC correctly FAILED (rc!=0) -- the dequantized E2M1 element is load-bearing"
fi

# --- [H] NEGATIVE CONTROL 3 (scale): per-block effective-scale flip -> MUST fail (two-level scale) ---
say "[H] negative control: per-block scale flip -> must FAIL (the two-level scale is S3's novelty)"
if /tmp/nv_cl /tmp/nv_good.ptx nvfp4_dequant "$N" nvfp4 "$M" "$K" "$N" sflip >/tmp/nv_nch.log 2>&1; then
  bad "scale-flip NC did NOT fail (changed block scale still matched -> the two-level scale is not load-bearing)"
else
  say "    scale-flip NC correctly FAILED (rc!=0) -- the two-level (E4M3 micro * FP32 tensor) scale is load-bearing"
fi

# --- [E] NEGATIVE CONTROL 4 (kernel corruption): dequant store * 2.0 -> re-emit -> MUST fail ---
say "[E] negative control: corrupt kernel store (dequant * 2.0) -> re-emit -> must FAIL"
sed 's/) \* sc\[/) * 2.0 * sc[/' "$KERN" > /tmp/nv_kern_bad.hx
if cmp -s /tmp/nv_kern_bad.hx "$KERN"; then
  bad "kernel-corruption NC is a NO-OP (sed pattern did not match the store line)"
else
  emit_ptx /tmp/nv_kern_bad.hx
  if [ ! -s /tmp/out.ptx ]; then
    bad "corrupted kernel emitted no PTX -- cannot run kernel-corruption NC"
  else
    cp /tmp/out.ptx /tmp/nv_bad.ptx
    if /tmp/nv_cl /tmp/nv_bad.ptx nvfp4_dequant "$N" nvfp4 "$M" "$K" "$N" >/tmp/nv_nce.log 2>&1; then
      bad "kernel-corruption NC did NOT fail (doubled dequant still within tolerance -> not load-bearing)"
    else
      say "    kernel-corruption NC correctly FAILED (rc!=0) -- the kovc-emitted dequant kernel is load-bearing"
    fi
  fi
fi
# NOTE: [E] never edits the committed kernel -- sed reads $KERN, writes /tmp; corrupted PTX from a /tmp copy.

# --- [F] codec self-test: the from-scratch E2M1 + E4M3 codec (GPU-free; guards the decode) ---
say "[F] NVFP4 codec self-test (E2M1 16 + E4M3 boundaries incl 448/2^-9/NaN, GPU-free)"
if /tmp/nv_cl /tmp/nv_good.ptx nvfp4_dequant 0 nvfp4_codec_selftest >/tmp/nv_codec.log 2>&1; then
  say "    $(grep -m1 'nvfp4_codec_selftest' /tmp/nv_codec.log || echo 'PASS (rc=0)')"
else
  bad "NVFP4 codec self-test FAILED (rc!=0):"; tail -4 /tmp/nv_codec.log >&2
fi

echo "------------------------------------------------------------"
if [ "$OK" = "1" ]; then echo "NVFP4_GPU_PASS"; exit 0; else echo "GPU_NVFP4_FAIL"; exit 1; fi
