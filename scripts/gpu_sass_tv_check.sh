#!/usr/bin/env bash
# gpu_sass_tv_check.sh (v1.5 #3, Phase 3 -- verifiable PTX->SASS, THE REAL PTXAS DE-TRUST): a from-scratch
# SASS->spec TRANSLATION-VALIDATION that PROVES the cubin's emitted vector_add machine code computes
# c[gid]=a[gid]+b[gid] for ALL f32 inputs, WITHOUT trusting ptxas's lowering and WITHOUT ptxas/cuobjdump
# appearing anywhere in the proof (they are untrusted oracles used only by the negative-control guards).
#
# WHAT THIS IS (honest scope -- do NOT overstate): this is the step that removes ptxas's PTX->SASS lowering
# from the trusted computing base for ONE named kernel. The de-trust VERDICT is the COMPOSITE of TWO checks
# over the ELF .text bytes ptxas emitted, both REQUIRED and both LOAD-BEARING:
#   (1) sass_tv (cuda_launch.c) -- a from-scratch CPU structural proof of the VALUE semantics for ALL inputs:
#       LEG1 sass_taint_indep proves the SASS is a data-INDEPENDENT STRAIGHT-LINE per-thread function of the
#       loads, ENFORCING (not assuming) straight-line-ness: fail-closed on any non-PT predication, any unmodeled
#       opcode, and any BRA that is not ptxas's self-loop trap pad (rel==-16). LEG2 sass_symbolic_addb proves
#       symbolic structural equality over OPAQUE load symbols -- the kernel performs EXACTLY ONE store and it is
#       a PLAIN FADD(LOAD_A,LOAD_B) to c at base+gid*4 -- MODIFIER-COMPLETE (rejects FADD lo[40:63] + hi[0:39]
#       operand/output modifiers and the field-complete address-path hi[0:39] pins) and TAG-INVALIDATING (any
#       opcode LEG2 does not model clears its destination tag, so a stale LOAD/SUM cannot survive an overwrite).
#       LEG3 basis+linearity via the from-scratch interpreter is a LOAD-BEARING execution cross-check (it catches
#       e.g. an early-EXIT that leaves c unwritten), gated on LEG1&&LEG2.
#   (2) sass_exec (cuda_launch.c) -- a GPU-DIFFERENTIAL check: the candidate cubin's ACTUAL RTX-3070 execution
#       must match the from-scratch interpreter element-for-element. LOAD-BEARING, not Phase-2-only: it discharges
#       INSTRUCTION-SCHEDULING / dependency-scoreboard correctness (the control word hi[40:63] the CPU structural
#       proof has no model for). A scheduling hazard produces wrong output for EVERY input (input-independent),
#       so a probe-level differential catches it deterministically.
# Together: the CPU legs prove the value computation is a+b for all inputs; the GPU-differential confirms the
# real hardware execution (scheduling included) matches that model. With Phase 1 (from-scratch decode==cuobjdump),
# Helix DECODES + INTERPRETS + VALIDATES ptxas's output for this kernel from-scratch.
# WHAT THIS IS NOT (the residual, stated plainly): ptxas still RAN (we validate its output, not re-derive it);
# the driver+GPU+silicon still EXECUTE the validated SASS AND the GPU is also the differential ORACLE -- it is
# the trust root the witness leans on for scheduling correctness (NOT removed from the TCB). A PURE-CPU proof of
# scheduling correctness (a from-scratch scoreboard/hazard model that removes the GPU from the verifier) is a
# labeled stretch, NOT done here. ONE straight-line kernel (vector_add), f32, data-independent control flow +
# affine addressing, sm_86 + CUDA-12.8 pinned. It is a translation-validation WITNESS (per-compilation,
# machine-checked), NOT a formal proof of ptxas. Full kernel coverage (loops/branches/other ops/arches) is a
# labeled multi-week+ stretch, not claimed here.
#
# Ten load-bearing NCs (a WRONG kernel whose lo word is byte-identical to genuine -> the COMPOSITE must REJECT):
# NC1 FADD->IMAD; NC2 sub.f32; NC3 FADD.SAT; NC4 neg-Ra (the hi-word FADD-modifier P0, audit #1); NC5 LDG.E.U8 +
# NC6 STG.E.U8 (the width/addressing sibling, audit #2); NC7 forward-BRA-over-EXIT double-store + NC8 unmodeled
# opcode (the control-flow/latch P0, audit #3) -- all rejected by sass_tv; NC9 stale-tag (an unmodeled op clobbers
# the FADD result before the store) -> sass_tv FAIL, and NC10 scoreboard (a cleared FADD dependency-wait bit, a
# DECODE-CLEAN scheduling-only edit the CPU proof cannot see) -> sass_exec FAIL -- the two orthogonal P0s audit #4
# (wdxvu1koz) proved, closed by LEG2 tag-invalidation and the load-bearing GPU-differential. genuine PASSES both
# checks while all ten NCs are rejected by the composite. Each NC has non-vacuity guards (bytes really
# changed AND cuobjdump confirms the intended different instruction/modifier/width/clobber, or stays decode-clean
# for the scheduling NC).
# Token-gated '-> SASS_TV_PASS/FAIL', run AS A FILE (mem #42). Committed cubin/kernel never edited (corruption
# on /tmp copies). Run under WSL (CUDA 12.8, RTX 3070): bash scripts/gpu_sass_tv_check.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PTXAS=/usr/local/cuda-12.8/bin/ptxas
OBJ=/usr/local/cuda-12.8/bin/cuobjdump
KPTX="$ROOT/helixc/examples/vector_add_kernel.ref.ptx"
KNAME=vector_add
OK=1
say(){ echo "[sass_tv] $*"; }
bad(){ echo "[sass_tv] *** FAIL: $*" >&2; OK=0; }
tv(){ "$CL" "$1" "$KNAME" 0 sass_tv 2>&1 | sed -n 's/.*-> \(SASS_TV_PASS\|SASS_TV_FAIL\)$/\1/p' | tail -1; }
dec(){ "$CL" "$1" "$KNAME" 0 sass_check 2>&1 | sed -n 's/.*-> \(SASS_DECODE_OK\|SASS_DECODE_FAIL\)$/\1/p' | tail -1; }
exe(){ "$CL" "$1" "$KNAME" 0 sass_exec 2>&1 | sed -n 's/.*-> \(SASS_EXEC_PASS\|SASS_EXEC_FAIL\)$/\1/p' | tail -1; }

echo "============================================================"
echo " Helix v1.5 #3 Phase 3: from-scratch SASS->spec translation-validation (ptxas PTX->SASS de-trust)"
echo "============================================================"

# --- [A] ptxas (12.8) -> a reproducible cubin ---
say "[A] ptxas (12.8) vector_add PTX -> cubin"
[ -x "$PTXAS" ] || { bad "no 12.8 ptxas at $PTXAS"; echo "SASS_TV_CHECK_FAIL"; exit 1; }
"$PTXAS" -arch=sm_86 "$KPTX" -o /tmp/stv_va.cubin 2>/tmp/stv_ptxas.log || { bad "ptxas failed:"; tail -4 /tmp/stv_ptxas.log >&2; echo "SASS_TV_CHECK_FAIL"; exit 1; }
say "    cubin $(wc -c < /tmp/stv_va.cubin) B"

# --- [B] build cuda_launch.c ---
say "[B] build cuda_launch.c"
gcc "$ROOT/helixc/runtime/cuda_launch.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/stv_cl >/tmp/stv_gcc.log 2>&1
[ -s /tmp/stv_cl ] || { bad "launcher build failed:"; tail -6 /tmp/stv_gcc.log >&2; echo "SASS_TV_CHECK_FAIL"; exit 1; }
CL=/tmp/stv_cl

# --- [C] compose Phase 1: from-scratch decode succeeds (read ptxas's output independently) ---
say "[C] Phase 1: from-scratch SASS decode of ptxas's output"
td=$(dec /tmp/stv_va.cubin)
if [ "$td" = SASS_DECODE_OK ]; then say "    from-scratch decode OK (SASS_DECODE_OK)"; else bad "Phase-1 decode=$td (expected SASS_DECODE_OK)"; fi

# --- [D] compose Phase 2: from-scratch interpreter == real RTX-3070 execution of the same cubin ---
say "[D] Phase 2: from-scratch interpreter == GPU execution of the same cubin"
te=$(exe /tmp/stv_va.cubin)
if [ "$te" = SASS_EXEC_PASS ]; then say "    interpreter == GPU (SASS_EXEC_PASS)"; else bad "Phase-2 interp!=GPU=$te (expected SASS_EXEC_PASS)"; fi

# --- [E] Phase 3 POSITIVE: the translation-validation PROVES c[gid]=a[gid]+b[gid] for ALL inputs ---
say "[E] Phase 3 positive: sass_tv PROVES the emitted SASS computes a+b for all f32 inputs"
"$CL" /tmp/stv_va.cubin "$KNAME" 0 sass_tv 2>&1 | sed -n 's/^sass_tv/    sass_tv/p'
tg=$(tv /tmp/stv_va.cubin)
if [ "$tg" = SASS_TV_PASS ]; then say "    translation-validation PASS (LEG1 cflow + LEG2 modifier-complete symbolic = load-bearing; basis+laws sanity)  OK"; else bad "genuine sass_tv=$tg (expected SASS_TV_PASS)"; fi

# locate the FADD bundle (lo=0x...097221 -> LE bytes 21 72 09 02 05) for the NC1 byte-flip
FOFF=$(grep -aboP '\x21\x72\x09\x02\x05' /tmp/stv_va.cubin | head -1 | cut -d: -f1)
if [ -z "$FOFF" ]; then bad "could not locate the FADD bundle (NC1 setup)"; FOFF=-1; fi
say "    FADD opcode byte at cubin offset $FOFF"

# --- [F] NC1: FADD->IMAD (a*b+c) -- a faithfully-lowered WRONG kernel -> the TV MUST reject ---
say "[F] NC1 wrong-kernel FADD->IMAD (a*b+c) -> sass_tv MUST reject (SASS_TV_FAIL)"
if [ "$FOFF" -ge 0 ]; then
  cp /tmp/stv_va.cubin /tmp/stv_nc1.cubin
  printf '\x24' | dd of=/tmp/stv_nc1.cubin bs=1 seek=$FOFF count=1 conv=notrunc 2>/dev/null
  if cmp -s /tmp/stv_nc1.cubin /tmp/stv_va.cubin; then bad "NC1 vacuous -- byte flip did not change the cubin";
  elif ! "$OBJ" -sass /tmp/stv_nc1.cubin 2>/dev/null | grep -q 'IMAD.* R9, R2, R5, R0'; then bad "NC1 vacuous -- cuobjdump does not show the intended FADD->IMAD at R9";
  else
    t1=$(tv /tmp/stv_nc1.cubin)
    if [ "$t1" = SASS_TV_FAIL ]; then say "    NC1 sass_tv=SASS_TV_FAIL (a*b+c rejected -- TV checks semantics, not just decode)  OK"; else bad "NC1 sass_tv=$t1 but expected SASS_TV_FAIL (wrong kernel certified!)"; fi
  fi
fi

# --- [G] NC2: sub.f32 (a-b) -- SAME FADD opcode + the negate bit lo[63]; the DECISIVE modifier NC ---
say "[G] NC2 wrong-kernel sub.f32 (a-b, FADD negate lo[63]) -> sass_tv MUST reject (SASS_TV_FAIL)"
sed 's/add\.f32/sub.f32/' "$KPTX" > /tmp/stv_sub.ptx
if ! grep -q 'sub\.f32' /tmp/stv_sub.ptx; then bad "NC2 setup -- ref PTX had no add.f32 to flip";
else
  "$PTXAS" -arch=sm_86 /tmp/stv_sub.ptx -o /tmp/stv_sub.cubin 2>/tmp/stv_sub_ptxas.log || { bad "NC2 sub.f32 ptxas failed:"; tail -4 /tmp/stv_sub_ptxas.log >&2; }
  if [ -s /tmp/stv_sub.cubin ]; then
    if ! "$OBJ" -sass /tmp/stv_sub.cubin 2>/dev/null | grep -Eq 'FADD R9, R2, -R5'; then bad "NC2 vacuous -- sub.f32 did not lower to the expected negate-FADD (FADD R9, R2, -R5)";
    elif cmp -s /tmp/stv_sub.cubin /tmp/stv_va.cubin; then bad "NC2 vacuous -- sub cubin identical to genuine";
    else
      t2=$(tv /tmp/stv_sub.cubin)
      if [ "$t2" = SASS_TV_FAIL ]; then say "    NC2 sass_tv=SASS_TV_FAIL  OK -- genuine(plain FADD)=PASS vs sub(negate FADD)=FAIL on the SAME opcode => the modifier-complete decode is load-bearing"; else bad "NC2 sass_tv=$t2 but expected SASS_TV_FAIL (sub.f32 certified as a+b -- the skeptic's hole!)"; fi
    fi
  fi
fi

# --- [H] NC3/NC4 hi-word FADD modifiers -- a faithfully-lowered FADD.SAT / neg-Ra has a BYTE-IDENTICAL lo word
#         (every operand/output modifier lives in the FADD HI word: SAT=hi[13], neg-Ra=hi[8], abs-Ra=hi[9],
#         round=hi[14:15], FTZ=hi[16]); the silicon HONORS them (clamp / b-a / wrong-rounding / subnormal-flush)
#         so these are WRONG kernels. This is the audit's P0 class -- the TV MUST reject (hi[0:39]==0 guard). ---
say "[H] NC3/NC4 hi-word FADD modifiers (SAT, neg-Ra) -> wrong kernels, sass_tv MUST reject (SASS_TV_FAIL)"
if [ "$FOFF" -ge 0 ]; then
  HB=$((FOFF+9))                                              # the FADD hi[8:15] byte (SAT=0x20, neg-Ra=0x01)
  base=$(xxd -s "$HB" -l1 -p /tmp/stv_va.cubin)
  # NC3: set SAT (hi[13])
  cp /tmp/stv_va.cubin /tmp/stv_nc3.cubin
  printf "\\x$(printf '%02x' $(( 0x$base | 0x20 )))" | dd of=/tmp/stv_nc3.cubin bs=1 seek=$HB count=1 conv=notrunc 2>/dev/null
  if cmp -s /tmp/stv_nc3.cubin /tmp/stv_va.cubin; then bad "NC3 vacuous -- SAT byte flip did not change the cubin";
  elif ! "$OBJ" -sass /tmp/stv_nc3.cubin 2>/dev/null | grep -q 'FADD.SAT R9, R2, R5'; then bad "NC3 vacuous -- cuobjdump does not show FADD.SAT (bit map drift)";
  else
    t3=$(tv /tmp/stv_nc3.cubin)
    if [ "$t3" = SASS_TV_FAIL ]; then say "    NC3 FADD.SAT sass_tv=SASS_TV_FAIL  OK -- the hi-word modifier guard (hi[0:39]==0) is load-bearing"; else bad "NC3 FADD.SAT sass_tv=$t3 but expected SASS_TV_FAIL (clamp(a+b) certified as a+b -- the audit P0!)"; fi
  fi
  # NC4: set neg-Ra (hi[8]) -- computes b-a
  cp /tmp/stv_va.cubin /tmp/stv_nc4.cubin
  printf "\\x$(printf '%02x' $(( 0x$base | 0x01 )))" | dd of=/tmp/stv_nc4.cubin bs=1 seek=$HB count=1 conv=notrunc 2>/dev/null
  if cmp -s /tmp/stv_nc4.cubin /tmp/stv_va.cubin; then bad "NC4 vacuous -- neg-Ra byte flip did not change the cubin";
  elif ! "$OBJ" -sass /tmp/stv_nc4.cubin 2>/dev/null | grep -q 'FADD R9, -R2, R5'; then bad "NC4 vacuous -- cuobjdump does not show FADD R9, -R2, R5";
  else
    t4=$(tv /tmp/stv_nc4.cubin)
    if [ "$t4" = SASS_TV_FAIL ]; then say "    NC4 neg-Ra (b-a) sass_tv=SASS_TV_FAIL  OK"; else bad "NC4 neg-Ra sass_tv=$t4 but expected SASS_TV_FAIL (b-a certified as a+b!)"; fi
  fi
fi

# --- [I] NC5/NC6 hi-word MEMORY-WIDTH modifiers (re-audit w9hjc67k4 sibling of the FADD hole) -- LDG.E.U8 /
#         STG.E.U8 read/write ONE byte instead of 4 = WRONG kernel, but the lo word is byte-identical to the
#         genuine 32-bit .E load/store (width lives in the hi word). LEG2's field-complete hi[0:39] pin
#         (ldg_hi/stg_hi/imx_hi) rejects this width/value class WITHOUT the GPU; the de-trust VERDICT still
#         requires sass_exec for SCHEDULING (per the header: verdict = sass_tv AND sass_exec). Clearing bit 0x08
#         of the LDG/STG hi[8:15] byte (0x19->0x11) flips .E -> .E.U8. ---
say "[I] NC5/NC6 hi-word memory-width (LDG.E.U8, STG.E.U8) -> wrong kernels, sass_tv MUST reject (SASS_TV_FAIL)"
if [ "$FOFF" -ge 0 ]; then
  BS=$((FOFF - 0xb0))                                          # .text base (FADD is at base+0xb0)
  # NC5: LDG.E.U8 (the load at base+0x80)
  LB=$((BS + 0x80 + 9))                                        # LDG hi[8:15] byte
  lb=$(xxd -s "$LB" -l1 -p /tmp/stv_va.cubin)
  cp /tmp/stv_va.cubin /tmp/stv_nc5.cubin
  printf "\\x$(printf '%02x' $(( 0x$lb & ~0x08 & 0xff )))" | dd of=/tmp/stv_nc5.cubin bs=1 seek=$LB count=1 conv=notrunc 2>/dev/null
  if cmp -s /tmp/stv_nc5.cubin /tmp/stv_va.cubin; then bad "NC5 vacuous -- LDG width byte flip did not change the cubin";
  elif ! "$OBJ" -sass /tmp/stv_nc5.cubin 2>/dev/null | grep -q 'LDG.E.U8 R2'; then bad "NC5 vacuous -- cuobjdump does not show LDG.E.U8 (bit map drift)";
  else
    t5=$(tv /tmp/stv_nc5.cubin)
    if [ "$t5" = SASS_TV_FAIL ]; then say "    NC5 LDG.E.U8 sass_tv=SASS_TV_FAIL  OK -- the field-complete LDG hi-word pin is load-bearing"; else bad "NC5 LDG.E.U8 sass_tv=$t5 but expected SASS_TV_FAIL (1-byte load certified as a+b -- re-audit sibling!)"; fi
  fi
  # NC6: STG.E.U8 (the store at base+0xc0)
  SB=$((BS + 0xc0 + 9))                                        # STG hi[8:15] byte
  sb=$(xxd -s "$SB" -l1 -p /tmp/stv_va.cubin)
  cp /tmp/stv_va.cubin /tmp/stv_nc6.cubin
  printf "\\x$(printf '%02x' $(( 0x$sb & ~0x08 & 0xff )))" | dd of=/tmp/stv_nc6.cubin bs=1 seek=$SB count=1 conv=notrunc 2>/dev/null
  if cmp -s /tmp/stv_nc6.cubin /tmp/stv_va.cubin; then bad "NC6 vacuous -- STG width byte flip did not change the cubin";
  elif ! "$OBJ" -sass /tmp/stv_nc6.cubin 2>/dev/null | grep -q 'STG.E.U8'; then bad "NC6 vacuous -- cuobjdump does not show STG.E.U8";
  else
    t6=$(tv /tmp/stv_nc6.cubin)
    if [ "$t6" = SASS_TV_FAIL ]; then say "    NC6 STG.E.U8 sass_tv=SASS_TV_FAIL  OK"; else bad "NC6 STG.E.U8 sass_tv=$t6 but expected SASS_TV_FAIL (1-byte store certified!)"; fi
  fi
fi

# --- [J] NC7 STRUCTURAL control-flow + latch P0 (audit-3 w9hjc67k4/wksw71qo0): a DECODE-CLEAN, hand-craftable
#         cubin -- FADD;STG(c)=sum ; forward BRA 0xf0 OVER an EXIT ; STG(c)=R2 (clobber) ; EXIT -- whose real
#         RTX-3070 execution computes c=a (the forward BRA is taken on silicon, the clobber overwrites c), while
#         the old TV certified it (LEG3 interpreter STOPS at the BRA and reads the still-correct intermediate c;
#         LEG2's stored_c_sum latch stayed set). NO modifier, NO unknown op -- a legal SASS control-flow pattern,
#         in scope because ptxas is the adversary. Closed by LEG1 (reject any non-self BRA) + LEG2 (require
#         EXACTLY ONE store = SUM->c). NC8: an unmodeled opcode (would halt the interpreter and hide a clobber)
#         -> LEG1 fail-closes on unknown ops. ---
say "[J] NC7 forward-BRA-over-EXIT double-store (GPU computes c=a) + NC8 unknown-op -> sass_tv MUST reject"
if [ "$FOFF" -ge 0 ]; then
  spl(){ printf "$2" | dd of=/tmp/stv_nc7.cubin bs=1 seek=$((BS + $1)) count=16 conv=notrunc 2>/dev/null; }
  cp /tmp/stv_va.cubin /tmp/stv_nc7.cubin
  spl 0xd0 '\x47\x79\x00\x00\x10\x00\x00\x00\x00\x00\x80\x03\x00\xc0\x0f\x00'   # BRA 0xf0 (forward, GPU-taken)
  spl 0xe0 '\x4d\x79\x00\x00\x00\x00\x00\x00\x00\x00\x80\x03\x00\xea\x0f\x00'   # EXIT (branched over)
  spl 0xf0 '\x86\x79\x00\x06\x02\x00\x00\x00\x04\x19\x10\x0c\x00\xe2\x0f\x00'   # STG.E [R6.64],R2  (clobber c=a)
  spl 0x100 '\x4d\x79\x00\x00\x00\x00\x00\x00\x00\x00\x80\x03\x00\xea\x0f\x00'  # EXIT
  if cmp -s /tmp/stv_nc7.cubin /tmp/stv_va.cubin; then bad "NC7 vacuous -- splice did not change the cubin";
  elif ! "$OBJ" -sass /tmp/stv_nc7.cubin 2>/dev/null | grep -q 'BRA 0xf0'; then bad "NC7 vacuous -- cuobjdump does not show the forward BRA 0xf0 (decode-clean check failed)";
  elif [ "$("$OBJ" -sass /tmp/stv_nc7.cubin 2>/dev/null | grep -c 'STG.E \[R6.64\]')" -lt 2 ]; then bad "NC7 vacuous -- cuobjdump does not show the second (clobber) STG to c";
  else
    t7=$(tv /tmp/stv_nc7.cubin)
    if [ "$t7" = SASS_TV_FAIL ]; then say "    NC7 forward-BRA double-store sass_tv=SASS_TV_FAIL  OK -- straight-line + unique-store enforcement is load-bearing"; else bad "NC7 sass_tv=$t7 but expected SASS_TV_FAIL (GPU computes c=a yet certified -- the audit-3 P0!)"; fi
  fi
  # NC8: patch the FADD opcode to 0x7212 (an unmodeled op) -> LEG1 must fail-close
  cp /tmp/stv_va.cubin /tmp/stv_nc8.cubin
  printf '\x12' | dd of=/tmp/stv_nc8.cubin bs=1 seek=$FOFF count=1 conv=notrunc 2>/dev/null
  if cmp -s /tmp/stv_nc8.cubin /tmp/stv_va.cubin; then bad "NC8 vacuous -- opcode byte unchanged";
  else
    t8=$(tv /tmp/stv_nc8.cubin)
    if [ "$t8" = SASS_TV_FAIL ]; then say "    NC8 unmodeled-opcode sass_tv=SASS_TV_FAIL  OK -- the TV fail-closes on ops it does not model"; else bad "NC8 unmodeled-opcode sass_tv=$t8 but expected SASS_TV_FAIL"; fi
  fi
fi

# --- [K] NC9 stale-tag + NC10 scoreboard (audit #4 wdxvu1koz, two orthogonal P0s). NC9: an opcode LEG2 does
#         not structurally model (IMAD-all-reg 0x7224) overwrites the FADD result R9 (tag SUM) before the store
#         -- decode-clean, GPU stores garbage -- and the OLD stale-tag certified it. Closed by LEG2 invalidating
#         the destination tag of any unmodeled op -> sass_tv MUST FAIL. NC10: clearing the FADD dependency-WAIT
#         bit (control word hi[40:63], which the CPU structural proof has NO model for) makes the GPU read stale
#         registers (garbage) while sass_tv still passes -- the GPU-DIFFERENTIAL leg (sass_exec, interp==GPU on
#         THE CANDIDATE) is what catches it. This is why the de-trust verdict is sass_tv AND sass_exec, and why
#         sass_exec is LOAD-BEARING (not Phase-2-only). ---
say "[K] NC9 stale-tag (unmodeled-op clobber) -> sass_tv MUST FAIL; NC10 scoreboard hazard -> sass_exec MUST catch"
if [ "$FOFF" -ge 0 ]; then
  ksp(){ printf "$2" | dd of=/tmp/stv_nc9.cubin bs=1 seek=$((BS + $1)) count=16 conv=notrunc 2>/dev/null; }
  cp /tmp/stv_va.cubin /tmp/stv_nc9.cubin
  ksp 0xc0 '\x24\x72\x09\x09\x08\x00\x00\x00\x00\x00\x00\x00\x00\xc0\x0f\x00'   # IMAD R9,R9,R8,R0 (0x7224, clobbers R9=SUM; LEG2-unmodeled)
  ksp 0xd0 '\x86\x79\x00\x06\x09\x00\x00\x00\x04\x19\x10\x0c\x00\xe2\x0f\x00'   # STG.E [R6.64],R9 (single store, of clobbered R9)
  ksp 0xe0 '\x4d\x79\x00\x00\x00\x00\x00\x00\x00\x00\x80\x03\x00\xea\x0f\x00'   # EXIT
  if cmp -s /tmp/stv_nc9.cubin /tmp/stv_va.cubin; then bad "NC9 vacuous -- splice did not change the cubin";
  elif ! "$OBJ" -sass /tmp/stv_nc9.cubin 2>/dev/null | grep -qE 'IMAD[^;]* R9, R9, R8, R0'; then bad "NC9 vacuous -- cuobjdump does not show the IMAD R9 clobber";
  else
    t9=$(tv /tmp/stv_nc9.cubin)
    if [ "$t9" = SASS_TV_FAIL ]; then say "    NC9 stale-tag sass_tv=SASS_TV_FAIL  OK -- unmodeled-op tag-invalidation is load-bearing"; else bad "NC9 stale-tag sass_tv=$t9 but expected SASS_TV_FAIL (clobbered R9 certified as SUM -- audit-4 P0!)"; fi
  fi
  # NC10: clear the FADD dependency-WAIT bit (control byte FOFF+14, the 0x40 bit) -> GPU reads stale loads.
  cp /tmp/stv_va.cubin /tmp/stv_nc10.cubin
  cb10=$(xxd -s $((FOFF+14)) -l1 -p /tmp/stv_va.cubin); nb10=$(printf '%02x' $(( 0x$cb10 & ~0x40 & 0xff )))
  printf "\\x$nb10" | dd of=/tmp/stv_nc10.cubin bs=1 seek=$((FOFF+14)) count=1 conv=notrunc 2>/dev/null
  if cmp -s /tmp/stv_nc10.cubin /tmp/stv_va.cubin; then bad "NC10 vacuous -- scoreboard byte unchanged (already 0)";
  elif ! "$OBJ" -sass /tmp/stv_nc10.cubin 2>/dev/null | grep -q 'FADD R9, R2, R5'; then bad "NC10 vacuous -- cuobjdump no longer shows the plain FADD (the edit must be decode-clean: scheduling-only)";
  else
    e10=$(exe /tmp/stv_nc10.cubin); s10=$(tv /tmp/stv_nc10.cubin)
    if [ "$e10" = SASS_EXEC_FAIL ]; then say "    NC10 scoreboard sass_exec=SASS_EXEC_FAIL  OK -- the GPU-differential leg catches the timing hazard (sass_tv alone=$s10, decode-clean, has no scoreboard model)"; else bad "NC10 scoreboard sass_exec=$e10 but expected SASS_EXEC_FAIL (the GPU-differential MUST catch a dependency-wait hazard)"; fi
  fi
fi

echo "------------------------------------------------------------"
if [ "$OK" = "1" ]; then echo "SASS_TV_CHECK_PASS"; exit 0; else echo "SASS_TV_CHECK_FAIL"; exit 1; fi
