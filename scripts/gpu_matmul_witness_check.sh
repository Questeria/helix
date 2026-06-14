#!/usr/bin/env bash
# gpu_matmul_witness_check.sh (v1.5 #2): a HOST-SIDE algebraic TRANSLATION-VALIDATION WITNESS for the
# kovc-emitted naive_matmul GPU kernel on a real CUDA device (RTX 3070, sm_86). It LIFTS the prior
# 1-sample empirical GPU match to "basis-EXACT + bilinearity-SAMPLED equivalence under a data-independence
# precondition", on this compiled shape, within the f32 envelope. It is a WITNESS, not a machine-checked
# proof.
#
# It is a "certified kernel" first increment: NOT full formal PTX+IEEE verification, NOT ptxas/SASS (#3),
# NOT the nonlinear layers (gelu/softmax stay sampled). It certifies ONE named kernel (naive_matmul) via
# THREE complementary legs that are ALL co-necessary -- the full gate ANDs all three; no leg alone is an
# all-inputs claim. Each leg is independently falsifiable by a negative control:
#
#   LEG 1 -- cflow (data-independence): a fail-closed def-use TAINT SCAN of the emitted PTX (cuda_launch.c
#     cflow_witness, GPU-free). Taint every loaded DATA value (ld.global/ld.shared dest, NOT ld.param),
#     propagate def->use to a fixpoint, and REJECT if a tainted value reaches a setp source, a predicated
#     guard, a selp/slct selector, or a memory ADDRESS; FAIL-CLOSED on a call, a tainted non-polynomial op,
#     a cap saturation, or non-convergence. PASS => control flow + selection + addressing are all
#     data-INDEPENDENT => a FIXED straight-line dataflow over the inputs for this shape (under affine
#     addressing/arithmetic). This is the PRECONDITION for the basis lift. (NC: a data-dependent int-flag
#     branch -> cflow MUST reject.)
#   LEG 2 -- matmul_basis (exact, on the 0/1 basis): sweep ALL rank-1 0/1 probes A=e_{a,b}, B=e_{c,d} over
#     [0,L)^4. 0/1 inputs are f32-exact, so the device output must EQUAL the spec [i==a]*[b==c]*[j==d]
#     bit-for-bit, INCLUDING the off-diagonal b!=c (=> all-zero). This pins the kernel's BILINEAR
#     coefficient tensor to matmul's -- but NOT higher-degree terms, so LEG 3 is CO-NECESSARY (not
#     optional). Non-vacuity: probes==L^4 AND nonzero==L^3. (NCs: drop-term / transpose / scale.)
#   LEG 3 -- matmul_bilin (bilinearity, f32 envelope): additivity + homogeneity in A and B within the
#     derived rounding bound tau = c_safe*L*u*S (u=2^-24, ~2x over the worst case). REQUIRED -- it catches
#     a*a (nonlinearity), which is INVISIBLE to LEG 2's 0/1 basis (0^2=0, 1^2=1). SAMPLED at one input
#     tuple (a tolerance tripwire, not a sweep). (NCs: add-const / nonlinearity.)
#   => LEG 1 + LEG 2 + LEG 3 TOGETHER (not any pair) give f == matmul on all f32 inputs for this shape.
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
verdict(){ "$CL" "$1" naive_matmul "$L" "$2" "$L" 2>&1 | sed -n 's/.*-> \(PASS\|FAIL\)$/\1/p' | tail -1; }
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
  vdb=$("$CL" /tmp/mmw_db.ptx matmul_databranch "$L" cflow "$L" 2>&1 | sed -n 's/.*-> \(PASS\|FAIL\)$/\1/p' | tail -1)
  if [ "$vdb" = FAIL ]; then say "    [data-branch] cflow=FAIL (expected FAIL)  OK"; else bad "[data-branch] cflow=$vdb but expected FAIL"; fi
fi

# --- [D2] NC -- data-dependent ADDRESS (LEG 1, addressing arm): a gather b[j*nn+col] where j=idx[row] is
#          a LOADED value -> the b-load's address register is tainted -> cflow MUST reject. NO setp/branch
#          is involved, so this exercises the ADDRESSING arm specifically (distinct from [D]'s branch),
#          proving the data-dependent-address check is load-bearing, not vacuous. ---
say "[D2] NC data-gather -> cflow MUST FAIL (LEG 1 addressing arm is load-bearing)"
cat > /tmp/mmw_gather.hx <<'KEOF'
@kernel
fn matmul_gather(a: f32, b: f32, c: f32, idx: i32, mm: i32, kk: i32, nn: i32) {
    let row = block_idx();
    let col = thread_idx();
    let j = idx[row];
    let mut acc = a[row * kk] * b[j * nn + col];
    let mut t = 1;
    while t < kk {
        acc = acc + a[row * kk + t] * b[t * nn + col];
        t = t + 1
    };
    c[row * nn + col] = acc
}
KEOF
emit /tmp/mmw_gather.hx
if [ ! -s /tmp/out.ptx ]; then bad "[data-gather] NC emitted no PTX"; else
  cp /tmp/out.ptx /tmp/mmw_gather.ptx
  vg=$("$CL" /tmp/mmw_gather.ptx matmul_gather "$L" cflow "$L" 2>&1 | sed -n 's/.*-> \(PASS\|FAIL\)$/\1/p' | tail -1)
  if [ "$vg" = FAIL ]; then say "    [data-gather] cflow=FAIL (expected FAIL)  OK"; else bad "[data-gather] cflow=$vg but expected FAIL"; fi
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
# SED-INTEGRITY: assert EACH expected corruption substring is present (a whole-file cmp cannot tell that
# only ONE of a two-pattern sed matched after code drift -> the NC would silently test a different bug).
chk(){ grep -qF "$2" "$1" || bad "[sed-integrity] $1 missing expected corruption: $2"; }
chk /tmp/nc_drop.hx  'let mut t = 2;'
chk /tmp/nc_addc.hx  '= acc + 1.0'
chk /tmp/nc_nl.hx    'a[row * kk + t] * a[row * kk + t]'
chk /tmp/nc_nl.hx    'let mut acc = a[row * kk] * a[row * kk]'
chk /tmp/nc_tr.hx    'b[col * kk + t]'
chk /tmp/nc_tr.hx    'let mut acc = a[row * kk] * b[col * kk]'
chk /tmp/nc_scale.hx 'acc * 2.0'
#       label         file              cflow basis bilin
nc "drop-term"    /tmp/nc_drop.hx   PASS FAIL PASS
nc "add-const"    /tmp/nc_addc.hx   PASS FAIL FAIL
nc "nonlinearity" /tmp/nc_nl.hx     PASS PASS FAIL
nc "transpose"    /tmp/nc_tr.hx     PASS FAIL PASS
nc "scale"        /tmp/nc_scale.hx  PASS FAIL PASS

# --- [F] AUTODIFF adjoint cert: apply ALL THREE legs (cflow + basis + bilin) to the bilinear BACKWARD
#         kernels gpu_matmul_atb (A^T@B = the weight gradient dW = X^T @ dY) and gpu_matmul_abt (A@B^T =
#         the input gradient dX = dC @ B^T) -- the SAME translation-validation witness as the forward
#         kernel. This is an exact-on-the-0/1-basis + bilinearity-sampled adjoint cert for the matmul
#         gradient (matmul ONLY -- the nonlinear layers gelu/softmax stay sampled, explicitly out of
#         scope), upgrading the prior single-sample finite-difference. Each: cflow + basis + bilin all
#         PASS, plus a drop-term NC that basis MUST reject (so the basis leg is load-bearing here too). ---
say "[F] autodiff adjoint cert: 3-leg witness (cflow+basis+bilin) on the matmul backward (gradient) kernels"
legv(){ "$CL" "$1" "$2" "$L" "$3" "$L" "$4" 2>&1 | sed -n 's/.*-> \(PASS\|FAIL\)$/\1/p' | tail -1; }
adj(){ # $1=label  $2=kernelfile  $3=entry  $4=kind(atb|abt)
  emit "$2"; if [ ! -s /tmp/out.ptx ]; then bad "[$1] backward kernel emitted no PTX"; return; fi
  cp /tmp/out.ptx /tmp/mmw_adj.ptx
  local vc vb vl
  vc=$(legv /tmp/mmw_adj.ptx "$3" cflow "");          if [ "$vc" = PASS ]; then say "    [$1] cflow=PASS  OK";       else bad "[$1] cflow=$vc but expected PASS"; fi
  vb=$(legv /tmp/mmw_adj.ptx "$3" matmul_basis "$4"); if [ "$vb" = PASS ]; then say "    [$1] basis[$4]=PASS  OK";   else bad "[$1] basis[$4]=$vb but expected PASS"; fi
  vl=$(legv /tmp/mmw_adj.ptx "$3" matmul_bilin "");   if [ "$vl" = PASS ]; then say "    [$1] bilin=PASS  OK";       else bad "[$1] bilin=$vl but expected PASS"; fi
  sed 's/let mut t = 1;/let mut t = 2;/' "$2" > /tmp/mmw_adjnc.hx
  if cmp -s /tmp/mmw_adjnc.hx "$2"; then bad "[$1] drop-term NC SED-NOOP"; return; fi
  grep -qF 'let mut t = 2;' /tmp/mmw_adjnc.hx || { bad "[$1] drop-term NC drift (pattern missing)"; return; }
  emit /tmp/mmw_adjnc.hx; if [ ! -s /tmp/out.ptx ]; then bad "[$1] NC no PTX"; return; fi
  cp /tmp/out.ptx /tmp/mmw_adjnc.ptx
  local vn; vn=$(legv /tmp/mmw_adjnc.ptx "$3" matmul_basis "$4")
  if [ "$vn" = FAIL ]; then say "    [$1] drop-term NC basis[$4]=FAIL  OK"; else bad "[$1] drop-term NC basis[$4]=$vn but expected FAIL"; fi
}
adj "atb dW=X^T@dY" "$EX/gpu_matmul_atb_kernel.hx" gpu_matmul_atb atb
adj "abt dX=dC@B^T" "$EX/gpu_matmul_abt_kernel.hx" gpu_matmul_abt abt

echo "------------------------------------------------------------"
if [ "$OK" = "1" ]; then echo "MATMUL_WITNESS_PASS"; exit 0; else echo "MATMUL_WITNESS_FAIL"; exit 1; fi
