#!/usr/bin/env bash
# gpu_ternary_check.sh (v1.5 S0 increment 2b): GPU-HARDWARE exact-integer verification
# of the ternary_matmul kernel on a real CUDA device (RTX 3070, sm_86).
#
# Proves the first-class ternary type t2 (tag 12, BitNet b1.58, -1/0/+1) EXECUTES
# correctly on the GPU -- not merely emits plausible PTX. The kovc-emitted ternary_matmul
# kernel is launched on fixed ternary weights {-1,0,+1} x int activations and its int32
# result is compared ELEMENT-WISE EXACTLY (no tolerance) against an independent CPU integer
# reference computed in cuda_launch.c's 'imatmul' mode (a content-only edit to that existing
# committed harness; not on the self-host/trust path; the fixpoint is unaffected).
#
# Two negative controls prove the check is load-bearing:
#   [D] comparator mutate  -> the exact-int compare MUST fail (the comparator has teeth)
#   [E] kernel-source corruption (drop a product term) -> re-emit PTX -> MUST fail (the
#       kovc-emitted kernel is what's actually computing the result)
# The corrupted kernel is emitted from a /tmp COPY -- the committed kernel is never edited.
#
# Run under WSL (CUDA 12.0+12.8): bash scripts/gpu_ternary_check.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EX="$ROOT/helixc/examples"
KERN="$EX/ternary_matmul_kernel.hx"
BS="$ROOT/stage0/helixc-bootstrap"
M=16; K=16; N=16
OK=1
say(){ echo "[gpu_ternary] $*"; }
bad(){ echo "[gpu_ternary] *** FAIL: $*" >&2; OK=0; }
emit_ptx(){ cp "$1" /tmp/kernel_in.hx; rm -f /tmp/out.ptx; /tmp/gt_newdrv.bin >/dev/null 2>&1 || true; }

echo "============================================================"
echo " Helix v1.5 S0 inc2b: ternary_matmul GPU exact-integer check"
echo "============================================================"

# --- [A] mint the PTX driver from the committed compiler + emit the ternary kernel PTX ---
say "[A] mint PTX driver + emit ternary_matmul PTX"
( cd "$BS" && bash assemble_k1.sh >/tmp/gt_asm.log 2>&1 )
chmod +x "$BS/seed.bin" 2>/dev/null || true
( cd "$BS" && ./seed.bin k1ptxdrv.hx /tmp/gt_newdrv.bin >/tmp/gt_drv.log 2>&1 ) || true
if [ ! -s /tmp/gt_newdrv.bin ]; then bad "PTX driver not built (see /tmp/gt_drv.log)"; echo "GPU_TERNARY_FAIL"; exit 1; fi
chmod +x /tmp/gt_newdrv.bin
emit_ptx "$KERN"
if [ ! -s /tmp/out.ptx ]; then bad "ternary_matmul PTX not emitted"; echo "GPU_TERNARY_FAIL"; exit 1; fi
cp /tmp/out.ptx /tmp/gt_ternary.ptx
say "    emitted ternary_matmul PTX ($(wc -c < /tmp/gt_ternary.ptx) B)"

# --- [B] build the launcher (committed cuda_launch.c, -lcuda) ---
say "[B] build cuda_launch.c launcher"
gcc "$ROOT/helixc/runtime/cuda_launch.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/gt_cl >/tmp/gt_gcc.log 2>&1
if [ ! -s /tmp/gt_cl ]; then bad "launcher build failed:"; tail -6 /tmp/gt_gcc.log >&2; echo "GPU_TERNARY_FAIL"; exit 1; fi

# --- [C] POSITIVE: GPU ternary matmul == CPU int reference, EXACT ---
say "[C] GPU exact-int verify (${M}x${K}x${N})"
if /tmp/gt_cl /tmp/gt_ternary.ptx ternary_matmul "$N" imatmul "$M" "$K" "$N" >/tmp/gt_pos.log 2>&1; then
  say "    $(grep -m1 'ternary_matmul(int)' /tmp/gt_pos.log || echo 'PASS (rc=0)')"
else
  bad "GPU exact-int verify did NOT pass (rc!=0):"; tail -6 /tmp/gt_pos.log >&2
fi

# --- [D] NEGATIVE CONTROL 1 (comparator): mutate one cell -> MUST fail ---
say "[D] negative control: comparator mutate -> must FAIL"
if /tmp/gt_cl /tmp/gt_ternary.ptx ternary_matmul "$N" imatmul "$M" "$K" "$N" mutate >/tmp/gt_ncd.log 2>&1; then
  bad "comparator NC did NOT fail (mutated result still 'passed' -> comparator not load-bearing)"
else
  say "    comparator NC correctly FAILED (rc!=0) -- the exact-int check has teeth"
fi

# --- [E] NEGATIVE CONTROL 2 (kernel corruption): drop a product term -> re-emit -> MUST fail ---
say "[E] negative control: corrupt kernel source -> re-emit -> must FAIL"
# Corrupt the OUTPUT store (acc -> acc + 1): a data-INDEPENDENT off-by-one that makes EVERY
# output cell differ from the reference by exactly 1, so it is guaranteed-detectable regardless
# of the test data (a "drop a term" corruption can be masked by structured/degenerate data).
sed 's/c\[row \* nn + col\] = acc/c[row * nn + col] = acc + 1/' "$KERN" > /tmp/gt_kern_bad.hx
if cmp -s /tmp/gt_kern_bad.hx "$KERN"; then
  bad "kernel-corruption NC is a NO-OP (sed pattern did not match the store line) -- NC not load-bearing"
else
  emit_ptx /tmp/gt_kern_bad.hx
  if [ ! -s /tmp/out.ptx ]; then
    bad "corrupted kernel emitted no PTX -- cannot run kernel-corruption NC"
  else
    cp /tmp/out.ptx /tmp/gt_ternary_bad.ptx
    if /tmp/gt_cl /tmp/gt_ternary_bad.ptx ternary_matmul "$N" imatmul "$M" "$K" "$N" >/tmp/gt_nce.log 2>&1; then
      bad "kernel-corruption NC did NOT fail (corrupted kernel still matched the CPU ref -> the kovc kernel is not load-bearing!)"
    else
      say "    kernel-corruption NC correctly FAILED (rc!=0) -- the kovc-emitted ternary kernel is load-bearing"
    fi
  fi
fi

# NOTE: [E] never edits the committed kernel -- sed reads $KERN and writes /tmp/gt_kern_bad.hx,
# and the corrupted PTX is emitted from that /tmp copy. The committed kernel is untouched.

echo "------------------------------------------------------------"
if [ "$OK" = "1" ]; then echo "TERNARY_GPU_PASS"; exit 0; else echo "GPU_TERNARY_FAIL"; exit 1; fi
