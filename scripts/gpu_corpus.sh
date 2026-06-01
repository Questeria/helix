#!/usr/bin/env bash
# GPU corpus harness (P5): compile a Helix @kernel with kovc (the raw-binary-
# bootstrapped self-hosting compiler) and RUN it on the GPU, verifying the result
# against a CPU reference. This is how each kovc-emitted GPU kernel is proven on
# real hardware. Run under WSL.
#
#   Usage: bash scripts/gpu_corpus.sh <kernel.hx> <kernel_name> [N] [op:add|mul|sub]
#
# Pipeline: kovc PTX-driver reads the kernel -> /tmp/out.ptx -> cuda_launch.c
# (CUDA Driver API) loads + launches it on the device -> verify c[i] vs op(a,b).
# The PTX-driver kovc (stage0/helixc-bootstrap/_kovc_ptx_driver.bin) is a gitignored
# build artifact; if absent it is minted from the seed (~10 min, one time).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

KHX="${1:?usage: gpu_corpus.sh <kernel.hx> <kernel_name> [N] [op]}"
KNAME="${2:?kernel name required}"
N="${3:-256}"
OP="${4:-add}"
DRV="stage0/helixc-bootstrap/_kovc_ptx_driver.bin"

if [[ ! -x "$DRV" ]]; then
    echo "PTX-driver kovc absent; minting from the seed (~10 min)..."
    python3 stage0/helixc-bootstrap/assemble_k1.py
    ( cd stage0/helixc-bootstrap && ulimit -s unlimited && ./seed.bin k1ptxdrv.hx _kovc_ptx_driver.bin )
fi

gcc helixc/runtime/cuda_launch.c -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -o /tmp/cl || exit 2
cp "$KHX" /tmp/kernel_in.hx
"$DRV" >/dev/null
echo -n "emitted op: "; grep -Eo "add.f32|mul.f32|sub.f32" /tmp/out.ptx | head -1
exec /tmp/cl /tmp/out.ptx "$KNAME" "$N" "$OP"
