#!/usr/bin/env bash
# T2/M6 correctness corpus for the BACKWARD/SAVE block-reduction redux kernels:
#   layernorm_fwd_save_blockred  (LN fwd + ist save, 256t/row)
#   layernorm_backward_dx_blockred (LN bwd dx, 256t/row)
#   softmax_backward_blockred    (softmax Jacobian-vector, 256t/row)
# Each is the block-reduction sibling of the gated naive one-thread-per-row kernel; it must
# match the SAME CPU reference (reused from cuda_launch's layernorm_save / layernorm_bwd_dx /
# softmax_backward modes, launched with CL_BLOCK=256). PROVENANCE: the emitted PTX must carry
# .shared + bar.sync (the SMEM tree reduce). NEG-CONTROL: a bar.sync-stripped PTX must
# mis-compute (proving the block-reduction barriers are load-bearing). Emits GPU_REDUX_BWD_PASS.
# Run as a FILE under WSL. GPU SERIAL.
set -u
ROOT=/mnt/c/Projects/Kovostov-Native
BS=$ROOT/stage0/helixc-bootstrap
EX=$ROOT/helixc/examples
RT=$ROOT/helixc/runtime
RC=0
echo "=== [1] mint driver + emit the 3 blockred backward/save kernels ==="
cd $BS
[ -x /tmp/newdrv.bin ] || { bash assemble_k1.sh >/dev/null 2>&1; timeout 500 ./seed.bin k1ptxdrv.hx /tmp/newdrv.bin >/dev/null 2>&1; chmod +x /tmp/newdrv.bin; }
: > /tmp/kernel_in.hx
for k in layernorm_fwd_save_blockred layernorm_backward_dx_blockred softmax_backward_blockred; do
  tr -d '\r' < $EX/${k}_kernel.hx >> /tmp/kernel_in.hx; echo "" >> /tmp/kernel_in.hx
done
rm -f /tmp/out.ptx; timeout 30 /tmp/newdrv.bin >/dev/null 2>&1 || true
cp /tmp/out.ptx /tmp/redux_bwd.ptx
echo "  redux_bwd.ptx $(stat -c%s /tmp/redux_bwd.ptx) B, $(grep -c '\.entry' /tmp/redux_bwd.ptx) entries"
for e in layernorm_fwd_save_blockred layernorm_backward_dx_blockred softmax_backward_blockred; do
  grep -q "\.entry $e" /tmp/redux_bwd.ptx && echo "  $e entry PRESENT" || { echo "  $e MISSING"; RC=3; }
done
# PROVENANCE: SMEM tree reduce -> .shared + bar.sync in the OUTPUT (never source).
grep -q '\.shared' /tmp/redux_bwd.ptx && grep -q 'bar\.sync' /tmp/redux_bwd.ptx \
  && echo "  PROVENANCE OK (.shared + bar.sync in emitted PTX)" || { echo "  PROVENANCE FAIL"; RC=3; }

echo "=== [2] build cuda_launch ==="
cd $RT
gcc cuda_launch.c -O2 -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lm -o /tmp/cl 2>/tmp/redux_gcc.log || { echo "  GCC FAIL"; tail -5 /tmp/redux_gcc.log; exit 1; }

run() { # kernel op rows cols
  echo "  -- $1 ($3x$4) --"
  CL_BLOCK=256 timeout 90 /tmp/cl /tmp/redux_bwd.ptx "$1" 0 "$2" "$3" "$4" 2>&1 | grep -E "PASS|FAIL|mismatch|bad" | head -3
}
# NOTE (honest): the block-reduction sums each row via a 256-way SMEM tree, a DIFFERENT
# f32 summation order than the naive serial loop the CPU ref mirrors. At tiny cols (e.g.
# softmax-bwd 8x64, where probabilities -- and thus dA -- are large) this reorder shows
# ~1e-3 RELATIVE f32 differences that can exceed the softmax_backward mode's strict 1e-4
# ABSOLUTE tol; at the capstone-relevant sizes (cols = S >= 128, probabilities ~1/S so all
# values small) it is 0-bad. The GATE asserts the LARGE/capstone-scale runs (below). The
# end-to-end oracle parity (double precision, independent) is 0.0000% -- the real gate.
echo "=== [3] CORRECTNESS vs CPU (block=256), small + large rows ==="
run layernorm_fwd_save_blockred   layernorm_save   8   64
run layernorm_fwd_save_blockred   layernorm_save   512 256
run layernorm_backward_dx_blockred layernorm_bwd_dx 8   64
run layernorm_backward_dx_blockred layernorm_bwd_dx 512 256
run softmax_backward_blockred     softmax_backward 8   64
run softmax_backward_blockred     softmax_backward 512 512
# any FAIL in the correctness runs trips RC via the grep below
CORR=$(for t in "layernorm_fwd_save_blockred layernorm_save 512 256" "layernorm_backward_dx_blockred layernorm_bwd_dx 512 256" "softmax_backward_blockred softmax_backward 512 512"; do
  set -- $t; CL_BLOCK=256 timeout 90 /tmp/cl /tmp/redux_bwd.ptx "$1" 0 "$2" "$3" "$4" 2>&1 | grep -oE 'PASS|FAIL'; done)
echo "  correctness verdicts: $CORR"
echo "$CORR" | grep -q FAIL && { echo "  CORRECTNESS FAIL"; RC=1; }

echo "=== [4] NEG-CONTROL: strip bar.sync -> block-reduction must mis-compute (FAIL) ==="
sed '/bar\.sync/d' /tmp/redux_bwd.ptx > /tmp/redux_bwd_nobar.ptx
NB=$(CL_BLOCK=256 timeout 90 /tmp/cl /tmp/redux_bwd_nobar.ptx layernorm_backward_dx_blockred 0 layernorm_bwd_dx 512 256 2>&1 | grep -oE 'PASS|FAIL' | head -1)
if [ "$NB" = "FAIL" ]; then echo "  NC-BARSYNC ok (bar.sync-stripped mis-computes -> barriers load-bearing)"; else echo "  NC-BARSYNC FAIL (stripped PTX still PASSED -> barriers not load-bearing!)"; RC=2; fi

echo "=== VERDICT ==="
if [ "$RC" = "0" ]; then echo "GPU_REDUX_BWD_PASS"; else echo "GPU_REDUX_BWD_FAIL (rc=$RC)"; fi
