#!/usr/bin/env bash
# gpu_sass_exec_check.sh (v1.5 #3, Phase 2 -- verifiable PTX->SASS): a from-scratch sm_86 SASS INTERPRETER
# (cuda_launch.c 'sass_exec') that EXECUTES the decoded vector_add SASS on the CPU, VALIDATED against real
# RTX-3070 execution of the SAME cubin (cuModuleLoadData loads the cubin's SASS directly -- the driver runs
# exactly what the interpreter models, no re-JIT, so this also closes the JIT-vs-standalone gap).
#
# WHAT THIS IS (honest scope): the from-scratch interpreter and the GPU agree element-for-element on probe
# inputs, so the interpreter's per-opcode SEMANTICS are CHECKED against hardware -- not assumed. Combined
# with Phase 1 (the from-scratch decode), Helix can now both READ and EXECUTE ptxas's emitted machine code
# independently.
# WHAT THIS IS NOT: this validates the interpreter on PROBE inputs; it does NOT yet PROVE the SASS computes
# the spec for ALL inputs (that is Phase 3, the translation-validation, which lifts decode+interpret to the
# real ptxas de-trust). ptxas still chose+lowered the SASS; the GPU still executes it. ONE straight-line kernel.
#
# Load-bearing NC: a DELIBERATELY-WRONG interpreter (sass_exec ... mutate -> FADD modeled as a-b) MUST
# diverge from the GPU -> SASS_EXEC_FAIL, proving the interp==GPU validation has teeth (a mis-modeled op is
# caught). Token-gated '-> SASS_EXEC_PASS/FAIL', run AS A FILE (mem #42). Run under WSL (CUDA 12.8, RTX 3070).
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PTXAS=/usr/local/cuda-12.8/bin/ptxas
KPTX="$ROOT/helixc/examples/vector_add_kernel.ref.ptx"
KNAME=vector_add
OK=1
say(){ echo "[sass_exec] $*"; }
bad(){ echo "[sass_exec] *** FAIL: $*" >&2; OK=0; }
extok(){ "$CL" "$1" "$KNAME" 0 sass_exec ${2:-} 2>&1 | sed -n 's/.*-> \(SASS_EXEC_PASS\|SASS_EXEC_FAIL\)$/\1/p' | tail -1; }

echo "============================================================"
echo " Helix v1.5 #3 Phase 2: from-scratch SASS interpreter validated vs RTX-3070 execution"
echo "============================================================"

# --- [A] ptxas (12.8) -> cubin ---
say "[A] ptxas (12.8) vector_add PTX -> cubin"
[ -x "$PTXAS" ] || { bad "no 12.8 ptxas at $PTXAS"; echo "SASS_EXEC_CHECK_FAIL"; exit 1; }
"$PTXAS" -arch=sm_86 "$KPTX" -o /tmp/sexec_va.cubin 2>/tmp/sexec_ptxas.log || { bad "ptxas failed:"; tail -4 /tmp/sexec_ptxas.log >&2; echo "SASS_EXEC_CHECK_FAIL"; exit 1; }
say "    cubin $(wc -c < /tmp/sexec_va.cubin) B"

# --- [B] build cuda_launch.c ---
say "[B] build cuda_launch.c"
gcc "$ROOT/helixc/runtime/cuda_launch.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/sexec_cl >/tmp/sexec_gcc.log 2>&1
[ -s /tmp/sexec_cl ] || { bad "launcher build failed:"; tail -6 /tmp/sexec_gcc.log >&2; echo "SASS_EXEC_CHECK_FAIL"; exit 1; }
CL=/tmp/sexec_cl

# --- [C] POSITIVE: the from-scratch interpreter == the GPU, element-for-element ---
say "[C] positive: from-scratch interpreter == real GPU execution of the same cubin"
tp=$(extok /tmp/sexec_va.cubin)
if [ "$tp" = SASS_EXEC_PASS ]; then say "    interpreter == GPU (SASS_EXEC_PASS)  OK"; else bad "interpreter != GPU: sass_exec=$tp (expected SASS_EXEC_PASS)"; "$CL" /tmp/sexec_va.cubin "$KNAME" 0 sass_exec 2>&1 | tail -3 >&2; fi

# --- [D] NC: a deliberately-wrong interpreter (FADD modeled as a-b) MUST diverge from the GPU ---
say "[D] NC wrong-interpreter (FADD a-b) -> MUST diverge from GPU (SASS_EXEC_FAIL)"
tn=$(extok /tmp/sexec_va.cubin mutate)
if [ "$tn" = SASS_EXEC_FAIL ]; then say "    wrong interpreter diverged from GPU (SASS_EXEC_FAIL)  OK -- the interp==GPU validation has teeth"; else bad "wrong interpreter did NOT diverge: sass_exec[mutate]=$tn (expected SASS_EXEC_FAIL -- validation is vacuous!)"; fi

echo "------------------------------------------------------------"
if [ "$OK" = "1" ]; then echo "SASS_EXEC_CHECK_PASS"; exit 0; else echo "SASS_EXEC_CHECK_FAIL"; exit 1; fi
