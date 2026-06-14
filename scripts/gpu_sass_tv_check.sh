#!/usr/bin/env bash
# gpu_sass_tv_check.sh (v1.5 #3, Phase 3 -- verifiable PTX->SASS, THE REAL PTXAS DE-TRUST): a from-scratch
# SASS->spec TRANSLATION-VALIDATION that PROVES the cubin's emitted vector_add machine code computes
# c[gid]=a[gid]+b[gid] for ALL f32 inputs, WITHOUT trusting ptxas's lowering and WITHOUT ptxas/cuobjdump
# appearing anywhere in the proof (they are untrusted oracles used only by the negative-control guards).
#
# WHAT THIS IS (honest scope -- do NOT overstate): this is the step that removes ptxas's PTX->SASS lowering
# from the trusted computing base for ONE named kernel. cuda_launch.c's sass_tv mode runs four co-necessary
# legs over the ELF .text bytes ptxas emitted: LEG1 sass_taint_indep (the SASS is a data-INDEPENDENT
# straight-line per-thread function of the loads -> a probe/symbolic check lifts to all inputs); LEG2
# sass_symbolic_addb (symbolic structural equality over OPAQUE load symbols: the value stored to c is
# EXACTLY a PLAIN FADD(LOAD_A,LOAD_B) at base+gid*4 -> holds for every f32; MODIFIER-COMPLETE, so a sub.f32
# = FADD-with-negate is rejected); LEG3 basis+linearity via the Phase-2 GPU-validated interpreter (exact
# f(1,0)=f(0,1)=1, f(0,0)=0, f(2a,2b)=2f(a,b); additivity within tau). LEG4 is Phase 2's interp==GPU FADD
# validation. Composed with Phase 1 (from-scratch decode==cuobjdump) and Phase 2 (interp==GPU), Helix now
# DECODES + INTERPRETS + PROVES ptxas's output for this kernel from-scratch.
# WHAT THIS IS NOT (the residual, stated plainly): ptxas still RAN (we validate its output, not re-derive
# it); the driver+GPU+silicon still EXECUTE the validated SASS (not removed from the TCB). ONE straight-line
# kernel (vector_add), f32, data-independent control flow + affine addressing, sm_86 + CUDA-12.8 pinned. It
# is a translation-validation WITNESS (per-compilation, machine-checked), NOT a formal proof of ptxas. Full
# kernel coverage (loops/branches/other ops/arches) is a labeled multi-week+ stretch, not claimed here.
#
# Two load-bearing NCs (a WRONG kernel that ptxas lowers FAITHFULLY -> the TV must REJECT, proving the TV
# checks the SEMANTICS, not just that decode succeeded): NC1 FADD->IMAD (a*b+c, opcode byte-flipped) ->
# SASS_TV_FAIL; NC2 sub.f32 (a-b, same FADD opcode + the negate bit lo[63]) -> SASS_TV_FAIL. NC2 is the
# decisive one: genuine (plain FADD) PASSES and sub (negate FADD) FAILS on the SAME opcode, so the rejection
# is SPECIFICALLY the modifier-complete decode (the skeptic's wrong-kernel-passes hole, closed). Each NC has
# non-vacuity guards (the bytes really changed AND cuobjdump confirms the intended different instruction).
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
if [ "$tg" = SASS_TV_PASS ]; then say "    translation-validation PASS (4 legs: cflow + symbolic + basis + laws)  OK"; else bad "genuine sass_tv=$tg (expected SASS_TV_PASS)"; fi

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

echo "------------------------------------------------------------"
if [ "$OK" = "1" ]; then echo "SASS_TV_CHECK_PASS"; exit 0; else echo "SASS_TV_CHECK_FAIL"; exit 1; fi
