#!/usr/bin/env bash
# gpu_matmul_witness_check.sh (v1.5 #2): a HOST-SIDE algebraic TRANSLATION-VALIDATION witness for the
# kovc-emitted naive_matmul GPU kernel on a real CUDA device (RTX 3070, sm_86). This LIFTS the prior
# 1-sample empirical GPU match to an ALL-INPUTS equivalence proof (per this compiled shape, f32 envelope).
#
# It is a "certified kernel" first increment: NOT full formal PTX+IEEE verification, NOT ptxas/SASS (#3),
# NOT the nonlinear layers (gelu/softmax stay sampled). It certifies ONE named kernel (naive_matmul) via
# THREE complementary legs, each independently falsifiable by a negative control:
#
#   LEG 1 -- cflow (data-independent control flow): a fail-closed def-use TAINT SCAN of the emitted PTX
#     (cuda_launch.c cflow_witness, GPU-free). Taint every loaded DATA value (ld.global/ld.shared dest,
#     NOT ld.param), propagate def->use to a fixpoint, and reject if any setp source or predicated-branch
#     guard is tainted (or any call). PASS => the kernel is a FIXED straight-line program over the inputs
#     for this shape => a fixed polynomial. This is the precondition that lets a finite basis sweep stand
#     in for all inputs. (NC: a data-dependent int-flag branch -> cflow MUST reject.)
#   LEG 2 -- matmul_basis (exact equivalence): sweep ALL rank-1 0/1 probes A=e_{a,b}, B=e_{c,d} over
#     [0,L)^4. 0/1 inputs are f32-exact, so the device output must EQUAL the spec [i==a]*[b==c]*[j==d]
#     bit-for-bit, INCLUDING the off-diagonal b!=c (=> all-zero). With LEG 1, basis agreement => agreement
#     on every input. Non-vacuity: probes==L^4 AND nonzero==L^3. (NCs: drop-term / transpose / scale.)
#   LEG 3 -- matmul_bilin (bilinearity, f32 envelope): additivity + homogeneity in A and B within the
#     derived rounding bound tau = c_safe*L*u*S (u=2^-24). REQUIRED -- it catches a*a (nonlinearity),
#     which is INVISIBLE to LEG 2's 0/1 basis (0^2=0, 1^2=1). (NCs: add-const / nonlinearity.)
#
# Six negative controls, each caught by >=1 leg (and each leg uniquely necessary for >=1 NC), re-emitted
# from a /tmp copy so the COMMITTED kernel is NEVER edited. Verdicts are read from each mode's '-> PASS/
# FAIL' stdout TOKEN -- this runs as a FILE so `if cmd` sees the true rc, but the token is what we gate on
# (mem #42: an inline `wsl.exe bash -c "...; echo $?"` reports 0 regardless and must never be trusted).
#
# Run under WSL (CUDA 12.0+12.8): bash scripts/gpu_matmul_witness_check.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
EX="$ROOT/helixc/examples"
KERN="$EX/naive_matmul_kernel.hx"
BS="$ROOT/stage0/helixc-bootstrap"
L=8
OK=1
say(){ echo "[mm_wit] $*"; }
bad(){ echo "[mm_wit] *** FAIL: $*" >&2; OK=0; }
emit(){ cp "$1" /tmp/kernel_in.hx; rm -f /tmp/out.ptx; "$DRV" >/dev/null 2>&1 || true; }
# verdict <ptx> <mode> -> echoes PASS or FAIL (the trailing '-> X' token). Kernel-name arg is naive_matmul
# (ignored by cflow, which scans the whole PTX; used by basis/bilin to fetch the entry).
verdict(){ "$CL" "$1" naive_matmul "$L" "$2" "$L" 2>&1 | sed -n 's/.*-> \(PASS\|FAIL\).*/\1/p' | tail -1; }
# expect <label> <ptx> <mode> <PASS|FAIL>
expect(){ local v; v=$(verdict "$2" "$3"); if [ "$v" = "$4" ]; then say "    [$1] $3=$v (expected $4)  OK"; else bad "[$1] $3=$v but expected $4"; fi; }

echo "============================================================"
echo " Helix v1.5 #2: naive_matmul translation-validation witness"
echo "============================================================"

# --- [A] obtain the PTX driver (reuse the fast_iter ext4 cache -- #2 is HOST-SIDE only, NO kovc.hx edit,
#         so the cdcf8673 driver is valid; else mint from raw -- self-contained on a fresh clone) ---
say "[A] obtain PTX driver + emit naive_matmul PTX"
CACHE="$HOME/.helix_fastiter"
CUR=$(cat "$ROOT/helixc/bootstrap/kovc.hx" "$ROOT/helixc/bootstrap/parser.hx" 2>/dev/null | sha256sum | cut -d' ' -f1)
if [ -s "$CACHE/newdrv.bin" ] && [ "$(cat "$CACHE/compiler.sha" 2>/dev/null)" = "$CUR" ]; then
  DRV="$CACHE/newdrv.bin"; say "    reusing cached driver (compiler unchanged -- no #2 kovc edit)"
else
  say "    minting driver from raw (compiler changed / no cache; ~4min)"
  ( cd "$BS" && bash assemble_k1.sh >/tmp/mmw_asm.log 2>&1 )
  chmod +x "$BS/seed.bin" 2>/dev/null || true
  ( cd "$BS" && ./seed.bin k1ptxdrv.hx /tmp/mmw_newdrv.bin >/tmp/mmw_drv.log 2>&1 ) || true
  if [ ! -s /tmp/mmw_newdrv.bin ]; then bad "PTX driver not built"; echo "MATMUL_WITNESS_FAIL"; exit 1; fi
  chmod +x /tmp/mmw_newdrv.bin; DRV=/tmp/mmw_newdrv.bin
  mkdir -p "$CACHE"; cp "$DRV" "$CACHE/newdrv.bin" 2>/dev/null && echo "$CUR" > "$CACHE/compiler.sha"
fi
emit "$KERN"
if [ ! -s /tmp/out.ptx ]; then bad "naive_matmul PTX not emitted"; echo "MATMUL_WITNESS_FAIL"; exit 1; fi
cp /tmp/out.ptx /tmp/mmw_good.ptx
say "    emitted naive_matmul PTX ($(wc -c < /tmp/mmw_good.ptx) B)"

# --- [B] build the launcher (committed cuda_launch.c, -lcuda -lcublas -lm) ---
say "[B] build cuda_launch.c launcher"
gcc "$ROOT/helixc/runtime/cuda_launch.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/mmw_cl >/tmp/mmw_gcc.log 2>&1
if [ ! -s /tmp/mmw_cl ]; then bad "launcher build failed:"; tail -6 /tmp/mmw_gcc.log >&2; echo "MATMUL_WITNESS_FAIL"; exit 1; fi
CL=/tmp/mmw_cl

# --- [C] POSITIVE: all 3 legs PASS on the genuine kovc-emitted kernel ---
say "[C] positive: 3 legs on the genuine naive_matmul (L=$L)"
expect GENUINE /tmp/mmw_good.ptx cflow        PASS
expect GENUINE /tmp/mmw_good.ptx matmul_basis PASS
expect GENUINE /tmp/mmw_good.ptx matmul_bilin PASS

# --- [D] NC -- data-dependent branch (LEG 1 only): an int-flag-gated accumulation. f32-`if` does not
#         compile in the @kernel path, so the branch keys on a loaded i32 flag -> setp.gt.s32 on a tainted
#         value -> cflow MUST reject. (The kernel name differs; cflow ignores it.) ---
say "[D] NC data-branch -> cflow MUST FAIL (LEG 1 is load-bearing)"
cat > /tmp/mmw_db.hx <<'KEOF'
@kernel
fn matmul_databranch(a: f32, b: f32, c: f32, flag: i32, mm: i32, kk: i32, nn: i32) {
    let row = block_idx();
    let col = thread_idx();
    let mut acc = a[row * kk] * b[col];
    let mut t = 1;
    while t < kk {
        let g = flag[row * kk + t];
        if g > 0 {
            acc = acc + a[row * kk + t] * b[t * nn + col]
        };
        t = t + 1
    };
    c[row * nn + col] = acc
}
KEOF
emit /tmp/mmw_db.hx
if [ ! -s /tmp/out.ptx ]; then bad "[data-branch] NC emitted no PTX"; else
  cp /tmp/out.ptx /tmp/mmw_db.ptx
  vdb=$("$CL" /tmp/mmw_db.ptx matmul_databranch "$L" cflow "$L" 2>&1 | sed -n 's/.*-> \(PASS\|FAIL\).*/\1/p' | tail -1)
  if [ "$vdb" = FAIL ]; then say "    [data-branch] cflow=FAIL (expected FAIL)  OK"; else bad "[data-branch] cflow=$vdb but expected FAIL"; fi
fi

# --- [E] NCs via sed on the committed kernel (NEVER edited): 5 corruptions, each caught by >=1 leg.
#         nc <label> <ncfile> <exp_cflow> <exp_basis> <exp_bilin>. SED-NOOP guard: if the corruption did
#         not change the file, the pattern drifted -> hard FAIL (a vacuous NC must not pass silently). ---
nc(){
  if cmp -s "$2" "$KERN"; then bad "[$1] SED-NOOP -- corruption pattern did not match the committed kernel"; return; fi
  emit "$2"
  if [ ! -s /tmp/out.ptx ]; then bad "[$1] NC emitted no PTX"; return; fi
  cp /tmp/out.ptx /tmp/mmw_nc.ptx
  expect "$1" /tmp/mmw_nc.ptx cflow        "$3"
  expect "$1" /tmp/mmw_nc.ptx matmul_basis "$4"
  expect "$1" /tmp/mmw_nc.ptx matmul_bilin "$5"
}
say "[E] NCs (sed -> /tmp, committed kernel untouched)"
sed 's/let mut t = 1;/let mut t = 2;/' "$KERN" > /tmp/nc_drop.hx
sed 's/c\[row \* nn + col\] = acc$/c[row * nn + col] = acc + 1.0/' "$KERN" > /tmp/nc_addc.hx
sed -e 's/a\[row \* kk + t\] \* b\[t \* nn + col\]/a[row * kk + t] * a[row * kk + t] * b[t * nn + col]/' \
    -e 's/let mut acc = a\[row \* kk\] \* b\[col\];/let mut acc = a[row * kk] * a[row * kk] * b[col];/' "$KERN" > /tmp/nc_nl.hx
sed -e 's/b\[t \* nn + col\]/b[col * kk + t]/' \
    -e 's/let mut acc = a\[row \* kk\] \* b\[col\];/let mut acc = a[row * kk] * b[col * kk];/' "$KERN" > /tmp/nc_tr.hx
sed 's/c\[row \* nn + col\] = acc$/c[row * nn + col] = acc * 2.0/' "$KERN" > /tmp/nc_scale.hx
#       label         file              cflow basis bilin
nc "drop-term"    /tmp/nc_drop.hx   PASS FAIL PASS
nc "add-const"    /tmp/nc_addc.hx   PASS FAIL FAIL
nc "nonlinearity" /tmp/nc_nl.hx     PASS PASS FAIL
nc "transpose"    /tmp/nc_tr.hx     PASS FAIL PASS
nc "scale"        /tmp/nc_scale.hx  PASS FAIL PASS

# --- [F] AUTODIFF adjoint cert: apply LEG 2 (basis-agreement) to the bilinear BACKWARD kernels --
#         gpu_matmul_atb (A^T@B = the weight gradient dW = X^T @ dY) and gpu_matmul_abt (A@B^T = the
#         input gradient dX = dC @ B^T). This UPGRADES their prior sampled finite-difference check to an
#         EXACT all-inputs adjoint certificate for the matmul gradient (matmul only -- the nonlinear
#         layers gelu/softmax stay sampled, explicitly out of scope). Each: positive basis PASS + a
#         drop-term NC that basis MUST reject (so the adjoint check is load-bearing). ---
say "[F] autodiff adjoint cert: exact basis-agreement on the matmul backward (gradient) kernels"
adj(){ # $1=label  $2=kernelfile  $3=entry  $4=kind(atb|abt)
  emit "$2"; if [ ! -s /tmp/out.ptx ]; then bad "[$1] backward kernel emitted no PTX"; return; fi
  cp /tmp/out.ptx /tmp/mmw_adj.ptx
  local va; va=$("$CL" /tmp/mmw_adj.ptx "$3" "$L" matmul_basis "$L" "$4" 2>&1 | sed -n 's/.*-> \(PASS\|FAIL\).*/\1/p' | tail -1)
  if [ "$va" = PASS ]; then say "    [$1] basis[$4]=PASS (expected PASS)  OK"; else bad "[$1] basis[$4]=$va but expected PASS"; fi
  sed 's/let mut t = 1;/let mut t = 2;/' "$2" > /tmp/mmw_adjnc.hx
  if cmp -s /tmp/mmw_adjnc.hx "$2"; then bad "[$1] drop-term NC SED-NOOP"; return; fi
  emit /tmp/mmw_adjnc.hx; if [ ! -s /tmp/out.ptx ]; then bad "[$1] NC no PTX"; return; fi
  cp /tmp/out.ptx /tmp/mmw_adjnc.ptx
  local vn; vn=$("$CL" /tmp/mmw_adjnc.ptx "$3" "$L" matmul_basis "$L" "$4" 2>&1 | sed -n 's/.*-> \(PASS\|FAIL\).*/\1/p' | tail -1)
  if [ "$vn" = FAIL ]; then say "    [$1] drop-term NC basis[$4]=FAIL (expected FAIL)  OK"; else bad "[$1] drop-term NC basis[$4]=$vn but expected FAIL"; fi
}
adj "atb dW=X^T@dY" "$EX/gpu_matmul_atb_kernel.hx" gpu_matmul_atb atb
adj "abt dX=dC@B^T" "$EX/gpu_matmul_abt_kernel.hx" gpu_matmul_abt abt

echo "------------------------------------------------------------"
if [ "$OK" = "1" ]; then echo "MATMUL_WITNESS_PASS"; exit 0; else echo "MATMUL_WITNESS_FAIL"; exit 1; fi
