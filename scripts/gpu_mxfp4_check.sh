#!/usr/bin/env bash
# gpu_mxfp4_check.sh (v1.5 S2): GPU-HARDWARE verification of the MXFP4 (OCP) dequant -> f16 matmul on
# a real CUDA device (RTX 3070, sm_86).
#
# Proves the kovc @kernel PTX path can DEQUANT a 4-bit block-scaled format end-to-end ON THE DEVICE
# (E2M1 nibble div-unpack + an f32-literal decode + a per-32-block E8M0 scale) and feed it into the S1
# fp16 GEMM path -- the verifiable-dequant + memory win the FP4 slate is about. NO kovc.hx edit (rides
# the existing @kernel path: S0 div-unpack + S1 f16), so the self-host fixpoint stays cdcf8673.
#
# The kovc-emitted naive_mxfp4_matmul kernel runs on MXFP4 weights (7 E2M1/i32 word) x f16 activations;
# its result is compared in cuda_launch.c's 'mxfp4' mode against an INDEPENDENT CPU dequant+matmul
# oracle (from-scratch E2M1+E8M0 codec, mxfp4_codec_selftest-verified) within DUAL-bound tolerance
# (abs 1e-3 OR rel 1e-2; the dequant is f32-EXACT, the only error is the f16 b input + the f16 output).
#
# THREE load-bearing negative controls (REAL exit codes -- run as a FILE, `if cmd`): [D] comparator
# MAGNITUDE-SCALED mutate; [E] kernel-corruption store acc -> acc+acc (DOUBLING -- magnitude-robust;
# a bare +1 is vacuous on K=224's larger cells, the S0 non-vacuous-NC lesson); [G] a packed-WEIGHT
# nibble flip (the GPU reads a changed weight, the oracle keeps the original codes -> proves the
# weights are load-bearing). Plus [F] the from-scratch-codec self-test (GPU-free).
#
# Run under WSL (CUDA 12.0+12.8): bash scripts/gpu_mxfp4_check.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EX="$ROOT/helixc/examples"
KERN="$EX/naive_mxfp4_matmul_kernel.hx"
BS="$ROOT/stage0/helixc-bootstrap"
M=8; K=224; N=8   # K must be a multiple of 224 = LCM(7,32)
OK=1
say(){ echo "[gpu_mxfp4] $*"; }
bad(){ echo "[gpu_mxfp4] *** FAIL: $*" >&2; OK=0; }
emit_ptx(){ cp "$1" /tmp/kernel_in.hx; rm -f /tmp/out.ptx; "$DRV" >/dev/null 2>&1 || true; }

echo "============================================================"
echo " Helix v1.5 S2: naive_mxfp4_matmul GPU dequant->f16 check"
echo "============================================================"

# --- [A] obtain the PTX driver (reuse the fast_iter ext4 cache when the compiler is unchanged -- S2
#         needs NO kovc.hx edit so the cdcf8673 driver is valid; else mint from raw) + emit the kernel ---
say "[A] obtain PTX driver + emit naive_mxfp4_matmul PTX"
CACHE="$HOME/.helix_fastiter"
CUR=$(cat "$ROOT/helixc/bootstrap/kovc.hx" "$ROOT/helixc/bootstrap/parser.hx" 2>/dev/null | sha256sum | cut -d' ' -f1)
if [ -s "$CACHE/newdrv.bin" ] && [ "$(cat "$CACHE/compiler.sha" 2>/dev/null)" = "$CUR" ]; then
  DRV="$CACHE/newdrv.bin"; say "    reusing cached driver (compiler unchanged -- no S2 kovc edit)"
else
  say "    minting driver from raw (compiler changed / no cache; ~4min)"
  ( cd "$BS" && bash assemble_k1.sh >/tmp/mx_asm.log 2>&1 )
  chmod +x "$BS/seed.bin" 2>/dev/null || true
  ( cd "$BS" && ./seed.bin k1ptxdrv.hx /tmp/mx_newdrv.bin >/tmp/mx_drv.log 2>&1 ) || true
  if [ ! -s /tmp/mx_newdrv.bin ]; then bad "PTX driver not built"; echo "GPU_MXFP4_FAIL"; exit 1; fi
  chmod +x /tmp/mx_newdrv.bin; DRV=/tmp/mx_newdrv.bin
  mkdir -p "$CACHE"; cp "$DRV" "$CACHE/newdrv.bin" 2>/dev/null && echo "$CUR" > "$CACHE/compiler.sha"
fi
emit_ptx "$KERN"
if [ ! -s /tmp/out.ptx ]; then bad "naive_mxfp4_matmul PTX not emitted"; echo "GPU_MXFP4_FAIL"; exit 1; fi
cp /tmp/out.ptx /tmp/mx_good.ptx
say "    emitted MXFP4 PTX ($(wc -c < /tmp/mx_good.ptx) B)"

# --- [B] build the launcher (committed cuda_launch.c, -lcuda -lcublas -lm) ---
say "[B] build cuda_launch.c launcher"
gcc "$ROOT/helixc/runtime/cuda_launch.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/mx_cl >/tmp/mx_gcc.log 2>&1
if [ ! -s /tmp/mx_cl ]; then bad "launcher build failed:"; tail -6 /tmp/mx_gcc.log >&2; echo "GPU_MXFP4_FAIL"; exit 1; fi

# --- [C] POSITIVE: GPU MXFP4 dequant->matmul within dual-bound tolerance of the CPU oracle ---
say "[C] GPU MXFP4 dequant verify (${M}x${K}x${N})"
if /tmp/mx_cl /tmp/mx_good.ptx naive_mxfp4_matmul "$N" mxfp4 "$M" "$K" "$N" >/tmp/mx_pos.log 2>&1; then
  say "    $(grep -m1 'naive_mxfp4_matmul' /tmp/mx_pos.log || echo 'PASS (rc=0)')"
else
  bad "GPU MXFP4 verify did NOT pass (rc!=0):"; tail -6 /tmp/mx_pos.log >&2
fi

# --- [D] NEGATIVE CONTROL 1 (comparator): magnitude-scaled mutate -> MUST fail ---
say "[D] negative control: comparator magnitude-scaled mutate -> must FAIL"
if /tmp/mx_cl /tmp/mx_good.ptx naive_mxfp4_matmul "$N" mxfp4 "$M" "$K" "$N" mutate >/tmp/mx_ncd.log 2>&1; then
  bad "comparator NC did NOT fail (mutated result still within tolerance -> comparator not load-bearing)"
else
  say "    comparator NC correctly FAILED (rc!=0) -- the dual-bound tolerance has teeth"
fi

# --- [G] NEGATIVE CONTROL 2 (data): packed-WEIGHT nibble flip -> MUST fail (weights load-bearing) ---
say "[G] negative control: packed-weight nibble flip -> must FAIL"
if /tmp/mx_cl /tmp/mx_good.ptx naive_mxfp4_matmul "$N" mxfp4 "$M" "$K" "$N" wflip >/tmp/mx_ncg.log 2>&1; then
  bad "weight-flip NC did NOT fail (changed weight still matched -> weights not load-bearing / decode ignored)"
else
  say "    weight-flip NC correctly FAILED (rc!=0) -- the dequantized weights are load-bearing"
fi

# --- [E] NEGATIVE CONTROL 3 (kernel corruption): store acc -> acc+acc -> re-emit -> MUST fail ---
say "[E] negative control: corrupt kernel store (acc -> acc+acc) -> re-emit -> must FAIL"
sed 's/c\[row \* nn + col\] = acc/c[row * nn + col] = acc + acc/' "$KERN" > /tmp/mx_kern_bad.hx
if cmp -s /tmp/mx_kern_bad.hx "$KERN"; then
  bad "kernel-corruption NC is a NO-OP (sed pattern did not match the store line)"
else
  emit_ptx /tmp/mx_kern_bad.hx
  if [ ! -s /tmp/out.ptx ]; then
    bad "corrupted kernel emitted no PTX -- cannot run kernel-corruption NC"
  else
    cp /tmp/out.ptx /tmp/mx_bad.ptx
    if /tmp/mx_cl /tmp/mx_bad.ptx naive_mxfp4_matmul "$N" mxfp4 "$M" "$K" "$N" >/tmp/mx_nce.log 2>&1; then
      bad "kernel-corruption NC did NOT fail (doubled output still within tolerance -> not load-bearing)"
    else
      say "    kernel-corruption NC correctly FAILED (rc!=0) -- the kovc-emitted MXFP4 kernel is load-bearing"
    fi
  fi
fi
# NOTE: [E] never edits the committed kernel -- sed reads $KERN, writes /tmp; corrupted PTX from a /tmp copy.

# --- [F] codec self-test: the from-scratch E2M1+E8M0 codec (GPU-free; guards the decode) ---
say "[F] MXFP4 codec self-test (all 16 E2M1 codes + E8M0 scales, GPU-free)"
if /tmp/mx_cl /tmp/mx_good.ptx naive_mxfp4_matmul 0 mxfp4_codec_selftest >/tmp/mx_codec.log 2>&1; then
  say "    $(grep -m1 'mxfp4_codec_selftest' /tmp/mx_codec.log || echo 'PASS (rc=0)')"
else
  bad "MXFP4 codec self-test FAILED (rc!=0):"; tail -4 /tmp/mx_codec.log >&2
fi

echo "------------------------------------------------------------"
if [ "$OK" = "1" ]; then echo "MXFP4_GPU_PASS"; exit 0; else echo "GPU_MXFP4_FAIL"; exit 1; fi
