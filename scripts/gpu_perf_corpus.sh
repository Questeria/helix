#!/usr/bin/env bash
# GPU PERF corpus harness (T2/M1): emit the SMEM-tiled GEMM kernel with kovc, prove
# the emitted PTX is kovc's own codegen (.shared + bar.sync grepped on the OUTPUT,
# never source), ptxas-accept it for sm_86, then RUN it on the device and verify the
# GEMM cell-by-cell vs a CPU oracle (integer inputs -> exact equality). Run under WSL.
#
#   Usage: bash scripts/gpu_perf_corpus.sh
#
# This is the M1 CORRECTNESS gate (the G1 >=3 TFLOP/s + cuBLAS oracle + TFLOP/s timing
# is the NEXT chunk -- see the gemm_perf NEXT-CHUNK note in cuda_launch.c). It mints
# the PTX driver from the raw-binary seed if absent (~10 min), emits the kernel, runs
# the provenance greps + a barrier-removal-style negative control (mutate), and the
# multi-size correctness compare. Tees the emitted PTX to .m1probe/ for inspection.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EX="$ROOT/helixc/examples"
OUT="$ROOT/.m1probe"
mkdir -p "$OUT"
DRV="stage0/helixc-bootstrap/_kovc_ptx_driver.bin"
RC=0

echo "=== [0] ensure PTX driver is current (mint from seed if absent) ==="
if [[ ! -x "$DRV" ]]; then
    echo "  PTX-driver absent; minting from the seed (~10 min)..."
    bash stage0/helixc-bootstrap/assemble_k1.sh
    ( cd stage0/helixc-bootstrap && ulimit -s unlimited && timeout 600 ./seed.bin k1ptxdrv.hx _kovc_ptx_driver.bin )
fi
[[ -x "$DRV" ]] || { echo "FATAL: no PTX driver"; exit 7; }

echo "=== [1] emit the tiled-GEMM kernel PTX ==="
cp "$EX/tiled_matmul_kernel.hx" /tmp/kernel_in.hx
rm -f /tmp/out.ptx
"$DRV" >/dev/null
[[ -s /tmp/out.ptx ]] || { echo "FATAL: driver emitted no PTX"; exit 6; }
cp /tmp/out.ptx "$OUT/tiled_matmul_kernel.ptx"
echo "  emitted $(stat -c%s /tmp/out.ptx) bytes -> $OUT/tiled_matmul_kernel.ptx"

echo "=== [2] PTX-PROVENANCE (grep the OUTPUT, never source) ==="
grep -q '\.shared'      /tmp/out.ptx && echo "  .shared PRESENT"      || { echo "  PROVENANCE FAIL: no .shared";   RC=3; }
grep -q 'bar\.sync 0'   /tmp/out.ptx && echo "  bar.sync 0 PRESENT"   || { echo "  PROVENANCE FAIL: no bar.sync";  RC=3; }
grep -q 'ld\.shared\.f32' /tmp/out.ptx && echo "  ld.shared.f32 PRESENT" || { echo "  PROVENANCE FAIL: no ld.shared"; RC=3; }
grep -q 'st\.shared\.f32' /tmp/out.ptx && echo "  st.shared.f32 PRESENT" || { echo "  PROVENANCE FAIL: no st.shared"; RC=3; }
grep -q 'fma\.rn\.f32'  /tmp/out.ptx && echo "  fma.rn.f32 PRESENT"   || { echo "  PROVENANCE FAIL: no fma.rn.f32"; RC=3; }
grep -q '\.target sm_86' /tmp/out.ptx && echo "  .target sm_86 PRESENT" || { echo "  PROVENANCE FAIL: not sm_86"; RC=3; }

echo "=== [3] ptxas acceptance (sm_86, -v for occupancy/spill) ==="
ptxas -arch=sm_86 -v /tmp/out.ptx -o "$OUT/tiled_matmul_kernel.cubin" 2>&1 || { echo "  PTXAS_REJECT"; exit 2; }
echo "  PTXAS_ACCEPT"

echo "=== [4] build launcher ==="
gcc helixc/runtime/cuda_launch.c -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -lm -o /tmp/cl || { echo "FATAL gcc"; exit 2; }

echo "=== [5] CORRECTNESS vs CPU oracle (integer inputs -> exact equality) ==="
for dims in "64 64 64" "64 8 128" "128 128 128" "256 256 256" "2048 2048 2048"; do
    set -- $dims
    echo "  -- M=$1 K=$2 N=$3 --"
    /tmp/cl /tmp/out.ptx tiled_matmul 0 gemm_perf "$1" "$2" "$3" || RC=1
done

echo "=== [6] NEGATIVE CONTROL (mutate one C cell -> MUST FAIL) ==="
if /tmp/cl /tmp/out.ptx tiled_matmul 0 gemm_perf 64 64 64 mutate; then
    echo "  NEG-CONTROL FAIL: mutated compare returned PASS (comparator has no teeth)"; RC=4
else
    echo "  NEG-CONTROL OK: mutated compare correctly FAILED"
fi

echo "=== GPU PERF CORPUS VERDICT ==="
if [[ "$RC" = "0" ]]; then echo "M1_CORRECTNESS_PASS"; else echo "M1_CORRECTNESS_FAIL (rc=$RC)"; fi
exit $RC
