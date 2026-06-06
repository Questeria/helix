#!/usr/bin/env bash
# Helix v1.0 -- ONE capstone audit round (the DYNAMIC half of a 5-consecutive-clean
# audit). Rebuilds the GPU capstone FROM the raw-binary self-hosted compiler, trains
# the 2-layer transformer on the RTX 3070, runs the BUILT-IN finite-difference
# gradient check (verify mode), compares the loss curve to the INDEPENDENT numpy
# oracle within 2%, and runs negative controls that must fail-as-expected. Run as a
# FILE under WSL. GPU SERIAL: never invoke two of these at once.
#
#   bash scripts/capstone_audit.sh [round-label]
#
# CONTROL-FLOW NOTE (train_transformer.c): "train <ptx> verify" runs the finite-diff
# check and RETURNS (no training, no loss_curve). "train <ptx>" trains + writes
# loss_curve.csv but does NOT run finite-diff. So each leg needs its own invocation.
# DRIVER PROVENANCE: we use /tmp/newdrv.bin -- the PTX driver freshly minted from the
# raw-binary seed by gate_kovc.sh THIS round (not a possibly-stale prebuilt binary).
# Emits a final line: CAPSTONE_AUDIT_PASS  or  CAPSTONE_AUDIT_FAIL
set -u
ROOT=/mnt/c/Projects/Kovostov-Native
BS=$ROOT/stage0/helixc-bootstrap
EX=$ROOT/helixc/examples
RT=$ROOT/helixc/runtime
ORACLE=$ROOT/verification/oracle/oracle_train.py
ROUND="${1:-r?}"
OK=1
echo "=================== CAPSTONE AUDIT ROUND $ROUND  $(date -u +%H:%M:%S) ==================="

# ---- [0] AMBIENT-ENV NEUTRALIZATION (v1.3 audit-remediation 4a) ----
# The v1.0 capstone is the DEFAULT (unset) HX_* configuration: train_transformer.c reads
# HX_S/HX_D/HX_H/HX_V/HX_NL/HX_K/HX_OPT/HX_DGBFUSE/HX_PROF/HX_FASTSYNC/HX_TF32 from the env
# and a stray value in the caller's environment would silently change the dims/op-set/launch
# geometry -> a different (non-v1.0) run masquerading as the audit. Unset every HX_* var so
# this audit is reproducible regardless of the inherited environment, then ASSERT none remain.
for v in $(env | sed -n 's/^\(HX_[A-Za-z0-9_]*\)=.*/\1/p'); do unset "$v"; done
_leftover=$(env | sed -n 's/^\(HX_[A-Za-z0-9_]*\)=.*/\1/p' | tr '\n' ' ')
if [ -n "$_leftover" ]; then echo "AUDIT FAIL: residual HX_* env after neutralization: $_leftover"; OK=0; fi
echo "  ambient HX_* neutralized (v1.0 capstone = default unset config); residual='${_leftover}'"

# ---- [1] toolchain trust spine + FRESH seed-minted PTX driver (kills staleness) ----
echo "=== [1] gate_kovc.sh (fixpoint + GPU PTX + corpus; mints /tmp/newdrv.bin from seed) ==="
bash $ROOT/scripts/gate_kovc.sh > /tmp/ca_gate.log 2>&1
if grep -q "^GATE_PASS" /tmp/ca_gate.log; then
  echo "  GATE_PASS  ($(grep -m1 'K2=' /tmp/ca_gate.log | cut -c6-21)... ; $(grep -m1 'CORPUS:' /tmp/ca_gate.log | sed 's/^ *//'))"
else echo "  GATE_FAIL"; tail -8 /tmp/ca_gate.log | sed 's/^/    /'; OK=0; fi

# ---- [2] the GPU compiler: PTX-driver kovc, freshly minted from the raw-binary seed ----
echo "=== [2] PTX-driver kovc (fresh from raw-binary seed) ==="
cd $BS
DRV=/tmp/newdrv.bin   # minted by gate_kovc.sh step [3] this round
if [ ! -x "$DRV" ]; then
  echo "  gate did not leave /tmp/newdrv.bin; minting directly from seed..."
  bash assemble_k1.sh >/dev/null 2>&1
  ( ulimit -s unlimited; ./seed.bin k1ptxdrv.hx /tmp/newdrv.bin ) >/dev/null 2>&1
fi
if [ -x "$DRV" ]; then echo "  driver (seed-minted) $(stat -c%s $DRV) B  sha=$(sha256sum $DRV | cut -c1-16)"; else echo "  DRIVER FAIL"; OK=0; fi

# ---- [3] emit combined.ptx: 15 kovc-emitted transformer kernels, single PTX module ----
echo "=== [3] emit combined.ptx (15 kovc-emitted kernels via the seed-minted driver) ==="
KS="vector_add naive_matmul gpu_matmul_atb gpu_matmul_abt gpu_qkt gpu_softmax gpu_softmax_backward gpu_gelu gpu_gelu_backward gpu_layernorm_fwd_save gpu_layernorm_backward_dx gpu_layernorm_backward_dgb gpu_ce_softmax_grad gpu_scale_inplace gpu_adam"
: > /tmp/combined_kernels.hx
nk=0
for k in $KS; do
  f=$EX/${k}_kernel.hx
  if [ -f "$f" ]; then tr -d '\r' < "$f" >> /tmp/combined_kernels.hx; echo "" >> /tmp/combined_kernels.hx; nk=$((nk+1));
  else echo "  MISSING kernel source: $k"; OK=0; fi
done
echo "  concatenated $nk kernel sources"
cp /tmp/combined_kernels.hx /tmp/kernel_in.hx
rm -f /tmp/out.ptx
"$DRV" >/dev/null 2>&1 || true
if [ -s /tmp/out.ptx ]; then
  cp /tmp/out.ptx /tmp/combined.ptx
  nent=$(grep -c '\.entry' /tmp/combined.ptx)
  echo "  combined.ptx $(stat -c%s /tmp/combined.ptx) B, $nent .entry kernels"
  if [ "$nent" -lt 15 ]; then echo "  PTX MISSING ENTRIES ($nent < 15)"; OK=0; fi
else echo "  PTX EMIT FAIL (driver produced no /tmp/out.ptx)"; OK=0; fi

# ---- [4a] build launcher + finite-diff gradient check (VERIFY mode) ----
echo "=== [4a] train_transformer VERIFY: backward finite-diff gradient check ==="
cd $RT
gcc train_transformer.c -O2 -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -lm -o /tmp/train 2>/tmp/ca_gcc.log || { echo "  GCC FAIL"; sed 's/^/    /' /tmp/ca_gcc.log; OK=0; }
if [ -x /tmp/train ] && [ -s /tmp/combined.ptx ]; then
  /tmp/train /tmp/combined.ptx verify > /tmp/ca_verify.log 2>&1; vrc=$?
  grep -E "finite-diff|max\|grad-fd\|" /tmp/ca_verify.log | sed 's/^/    /'
  if grep -q "backward finite-diff: PASS" /tmp/ca_verify.log; then echo "  FINITE-DIFF: PASS (rc=$vrc)"; else echo "  FINITE-DIFF: FAIL/absent (rc=$vrc)"; OK=0; fi
else echo "  cannot run verify (no launcher or ptx)"; OK=0; fi

# ---- [4b] TRAIN on the RTX 3070 -> loss_curve.csv + init_weights.bin ----
echo "=== [4b] train_transformer TRAIN: 500 Adam steps on GPU ==="
# STALE-ARTIFACT GUARD (v1.3 audit-remediation 4a): rm the artifacts this leg writes BEFORE
# the run, drop a timestamp marker, then AFTER assert each output EXISTS and is FRESH this run
# (newer than the marker). A failed/crashed train must not leave a stale loss_curve.csv or
# init_weights.bin from a prior round to false-pass the convergence / oracle-compare gates.
rm -f "$RT/loss_curve.csv" "$RT/init_weights.bin"
# FRESHNESS via rm-before + non-empty-after (filesystem-agnostic). We DELETED these above, so any
# non-empty file present after the run was necessarily written THIS run. (The earlier mtime "-nt
# marker" test was unreliable here: the marker lived in /tmp [WSL ext4] while the artifacts live in
# $RT under /mnt/c [DrvFs], whose coarse mtime + clock skew made freshly-written files read as STALE.)
for a in loss_curve.csv init_weights.bin; do [ -e "$RT/$a" ] && { echo "  AUDIT FAIL: could not remove stale $a before run"; OK=0; }; done
if [ -x /tmp/train ] && [ -s /tmp/combined.ptx ]; then
  cd $RT
  /tmp/train /tmp/combined.ptx > /tmp/ca_train.log 2>&1; trc=$?
  grep -E "train K=|step0 loss" /tmp/ca_train.log | sed 's/^/    /'
  # fail-closed: outputs must exist + be non-empty (=> fresh this run, given the rm-before)
  for art in loss_curve.csv init_weights.bin; do
    if [ ! -s "$RT/$art" ]; then echo "  AUDIT FAIL: train left no/empty $art this run (rc=$trc)"; OK=0;
    else echo "  fresh artifact: $art ($(stat -c%s "$RT/$art") B, written this run)"; fi
  done
  finalloss=$(tail -1 $RT/loss_curve.csv 2>/dev/null | cut -d',' -f2)
  echo "  final loss = ${finalloss:-NA}  (NC1: must be 0 < L < 1.0)  rc=$trc"
  awk -v L="${finalloss:-99}" 'BEGIN{ if (L+0 < 1.0 && L+0 > 0.0) exit 0; else exit 1 }' || { echo "  NC1 FAIL (loss did not converge)"; OK=0; }
else echo "  cannot train"; OK=0; fi

# ---- [5] independent numpy oracle + within-2% compare ----
echo "=== [5] numpy oracle (independent) + within-2% compare ==="
cd $RT
# STALE-ARTIFACT GUARD (v1.3 audit-remediation 4a): rm the oracle output BEFORE the run and
# require it fresh after, so a crashed oracle cannot leave a prior round's oracle_curve.csv to
# false-pass the within-2% compare.
rm -f "$RT/oracle_curve.csv"
# FRESHNESS via rm-before + non-empty-after (filesystem-agnostic; see the [4b] note -- the /tmp-vs-
# /mnt/c "-nt marker" check was unreliable on DrvFs and false-flagged fresh files as STALE).
[ -e "$RT/oracle_curve.csv" ] && { echo "  AUDIT FAIL: could not remove stale oracle_curve.csv before run"; OK=0; }
python3 "$ORACLE" > /tmp/ca_oracle.log 2>&1; orc=$?
echo "  oracle rc=$orc"
if [ ! -s "$RT/oracle_curve.csv" ]; then echo "  AUDIT FAIL: oracle left no/empty oracle_curve.csv this run (rc=$orc)"; OK=0;
else echo "  fresh artifact: oracle_curve.csv ($(stat -c%s "$RT/oracle_curve.csv") B, written this run)"; fi
# v1.3 audit-remediation A6: a NONZERO oracle exit is a GATE FAILURE. With A5 the
# oracle now exits 1 on a failed backward self-check (analytic backprop vs finite-
# diff), so this branch gates the oracle self-check END-TO-END (self-check FAIL ->
# oracle rc!=0 -> ORACLE ERROR -> OK=0 -> CAPSTONE_AUDIT_FAIL). A clean run exits 0.
if [ "$orc" != "0" ]; then echo "  ORACLE ERROR (rc=$orc -- incl a FAILED backward self-check via A5)"; tail -4 /tmp/ca_oracle.log | sed 's/^/    /'; OK=0; fi
if [ -s $RT/oracle_curve.csv ] && [ -s $RT/loss_curve.csv ]; then
  paste -d',' $RT/loss_curve.csv $RT/oracle_curve.csv > /tmp/ca_cmp.csv
  worst=$(awk -F',' 'NF>=4 { h=$2; o=$4; if(o!=0){ d=(h-o)/o; if(d<0)d=-d; if(d>m)m=d; n++ } } END{ if(n>0) printf "%.8f", m; else printf "NaN" }' /tmp/ca_cmp.csv)
  nrows=$(awk -F',' 'NF>=4{n++} END{print n+0}' /tmp/ca_cmp.csv)
  echo "  worst-case relative diff = $worst over $nrows comparable rows  (bar = 0.02)"
  if [ "${nrows:-0}" -lt 10 ]; then echo "  WITHIN-2% FAIL (only $nrows comparable rows -- vacuous-pass guard)"; OK=0; fi
  awk -v W="$worst" 'BEGIN{ if (W != "NaN" && W+0 >= 0 && W+0 < 0.02) exit 0; else exit 1 }' || { echo "  WITHIN-2% FAIL"; OK=0; }
  if cmp -s $RT/loss_curve.csv $RT/oracle_curve.csv; then echo "  NC2 FAIL (curves byte-identical -> not independent)"; OK=0; else echo "  NC2: curves differ (genuinely independent) ok"; fi
else echo "  oracle/loss curve missing"; OK=0; fi

# ---- [6] NEGATIVE CONTROL: a corrupted backward kernel MUST be caught by finite-diff ----
echo "=== [6] negative control: corrupt gpu_gelu_backward -> finite-diff MUST catch it ==="
sed 's/0\.7978846/0.9978846/g; s/0\.044715/0.144715/g' $EX/gpu_gelu_backward_kernel.hx > /tmp/gelu_bw_bad.hx
if ! cmp -s /tmp/gelu_bw_bad.hx $EX/gpu_gelu_backward_kernel.hx; then
  : > /tmp/combined_bad.hx
  for k in $KS; do
    if [ "$k" = "gpu_gelu_backward" ]; then tr -d '\r' < /tmp/gelu_bw_bad.hx >> /tmp/combined_bad.hx;
    else tr -d '\r' < $EX/${k}_kernel.hx >> /tmp/combined_bad.hx; fi
    echo "" >> /tmp/combined_bad.hx
  done
  cp /tmp/combined_bad.hx /tmp/kernel_in.hx; rm -f /tmp/out.ptx
  "$DRV" >/dev/null 2>&1 || true
  if [ -s /tmp/out.ptx ]; then
    cp /tmp/out.ptx /tmp/combined_bad.ptx
    cd $RT; /tmp/train /tmp/combined_bad.ptx verify > /tmp/ca_bad.log 2>&1 || true
    if grep -q "backward finite-diff: PASS" /tmp/ca_bad.log; then echo "  NC-PERTURB FAIL (corrupted backward STILL passed finite-diff -> check not load-bearing!)"; OK=0;
    else echo "  NC-PERTURB ok (corrupted backward caught: $(grep -m1 'finite-diff' /tmp/ca_bad.log | sed 's/^ *//' || echo crash))"; fi
  else echo "  NC-PERTURB inconclusive (bad ptx not emitted)"; OK=0; fi
else echo "  NC-PERTURB FAIL (perturbation was a no-op: constants not found in kernel)"; OK=0; fi

# ---- restore the GOOD artifacts (the bad run overwrote loss/init) for downstream use ----
cd $RT; /tmp/train /tmp/combined.ptx > /tmp/ca_restore.log 2>&1 || true
python3 "$ORACLE" > /tmp/ca_oracle2.log 2>&1 || true

echo "=================== VERDICT ==================="
if [ "$OK" = "1" ]; then echo "CAPSTONE_AUDIT_PASS"; else echo "CAPSTONE_AUDIT_FAIL"; fi
echo "(round $ROUND done $(date -u +%H:%M:%S))"
# FAIL-CLOSED (v1.3 final-pass): propagate the verdict to the PROCESS EXIT STATUS so a caller
# (CI / a parent gate) can never read a printed CAPSTONE_AUDIT_FAIL as a success (exit 0).
if [ "$OK" = "1" ]; then exit 0; else exit 1; fi
