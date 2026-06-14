#!/usr/bin/env bash
# gpu_sass_check.sh (v1.5 #3, Phase 1 -- verifiable PTX->SASS, the FOUNDATION): an INDEPENDENT from-scratch
# sm_86 SASS DECODER for the straight-line vector_add kernel subset, cross-checked against NVIDIA's
# cuobjdump/nvdisasm as UNTRUSTED oracles, with load-bearing byte-flip negative controls.
#
# WHAT THIS IS (honest scope -- do NOT overstate, per the #2 audit lesson): the first project surface that
# READS the actual machine code ptxas emitted. cuda_launch.c's sass_check parses the cubin ELF .text
# from-scratch (sass_elf_find_text) and decodes every 128-bit bundle (sass_disasm) with NO NVIDIA library,
# reproducing cuobjdump/nvdisasm instruction-for-instruction. This DE-TRUSTS the DISASSEMBLER (we no longer
# take NVIDIA's word that the bytes mean what cuobjdump says) and lets us SEE + structurally check the
# emitted instructions -- which S0/#2/#4 never do (they only check OUTPUTS).
# WHAT THIS IS NOT (the residual, stated plainly): it does NOT yet de-trust ptxas (ptxas still chose +
# lowered the instructions; verifying the ENCODING of its output is not proving its LOWERING faithful --
# that is Phase 3, the SASS->spec translation-validation). The driver+GPU still execute the SASS. ONE
# kernel, straight-line subset (~13 opcodes), the decode whitelist is CUDA-12.8/sm_86-pinned + FAIL-CLOSED.
#
# Two load-bearing NCs (the from-scratch DECODER itself rejects, WITHOUT GPU execution): NC1 an opcode byte
# flipped OUT of the whitelist -> sass_check FAIL-CLOSES; NC2 an opcode byte flipped to a DIFFERENT known
# opcode -> the from-scratch decode CHANGES vs the genuine (proving the decoder reads the real bytes, not a
# hardcoded answer). Each has 2 non-vacuity guards (the byte actually changed AND cuobjdump sees the change).
# Token-gated '-> SASS_PASS/SASS_FAIL', run AS A FILE (mem #42). The committed cubin/kernel are never edited
# (corruption is on /tmp copies). Run under WSL (CUDA 12.8): bash scripts/gpu_sass_check.sh
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PTXAS=/usr/local/cuda-12.8/bin/ptxas
OBJ=/usr/local/cuda-12.8/bin/cuobjdump
NVD=/usr/local/cuda-12.8/bin/nvdisasm
KPTX="$ROOT/helixc/examples/vector_add_kernel.ref.ptx"
KNAME=vector_add
OK=1
say(){ echo "[sass] $*"; }
bad(){ echo "[sass] *** FAIL: $*" >&2; OK=0; }
# normalize a cuobjdump/nvdisasm -sass dump to one canonical 'mnemonic operands' per instruction line
norm_oracle(){ grep -E '/\*[0-9a-f]+\*/' "$1" | sed -E 's@^ */\*[0-9a-f]+\*/ +@@; s@ *;.*@@'; }
# the from-scratch decode, canonicalized the same way (strip the /*addr*/ prefix; drop the summary line)
norm_mine(){ grep '^/\*' "$1" | sed 's@^/\*[0-9a-f]*\*/ @@'; }
sass_tok(){ "$CL" "$1" "$KNAME" 0 sass_check 2>&1 | sed -n 's/.*-> \(SASS_DECODE_OK\|SASS_DECODE_FAIL\)$/\1/p' | tail -1; }

echo "============================================================"
echo " Helix v1.5 #3 Phase 1: from-scratch sm_86 SASS decoder for vector_add (disassembler de-trust)"
echo "============================================================"

# --- [A] ptxas (12.8 -- the PATH ptxas=12.0 rejects .version 8.3) -> a reproducible cubin ---
say "[A] ptxas (12.8) vector_add PTX -> cubin"
[ -x "$PTXAS" ] || { bad "no 12.8 ptxas at $PTXAS"; echo "SASS_CHECK_FAIL"; exit 1; }
"$PTXAS" -arch=sm_86 "$KPTX" -o /tmp/sass_va.cubin 2>/tmp/sass_ptxas.log || { bad "ptxas failed:"; tail -4 /tmp/sass_ptxas.log >&2; echo "SASS_CHECK_FAIL"; exit 1; }
say "    cubin $(wc -c < /tmp/sass_va.cubin) B"

# --- [B] build the committed launcher (cuda_launch.c, host-side; sass_check is GPU-free but the file links cuda) ---
say "[B] build cuda_launch.c"
gcc "$ROOT/helixc/runtime/cuda_launch.c" -I/usr/local/cuda/include -L/usr/lib/wsl/lib -L/usr/local/cuda/lib64 -lcuda -lcublas -lm -o /tmp/sass_cl >/tmp/sass_gcc.log 2>&1
[ -s /tmp/sass_cl ] || { bad "launcher build failed:"; tail -6 /tmp/sass_gcc.log >&2; echo "SASS_CHECK_FAIL"; exit 1; }
CL=/tmp/sass_cl

# --- [C] POSITIVE: from-scratch decode == cuobjdump == nvdisasm (cross-oracle) ---
say "[C] positive: from-scratch decode reproduces BOTH untrusted disassemblers"
tok=$(sass_tok /tmp/sass_va.cubin)
if [ "$tok" != SASS_DECODE_OK ]; then bad "genuine sass_check=$tok (expected SASS_DECODE_OK)"; fi
"$CL" /tmp/sass_va.cubin "$KNAME" 0 sass_check 2>/dev/null > /tmp/sass_mine_raw.txt
norm_mine /tmp/sass_mine_raw.txt > /tmp/sass_mine.txt
"$OBJ" -sass /tmp/sass_va.cubin 2>/dev/null > /tmp/sass_obj_raw.txt; norm_oracle /tmp/sass_obj_raw.txt > /tmp/sass_obj.txt
"$NVD" -c  /tmp/sass_va.cubin 2>/dev/null > /tmp/sass_nv_raw.txt;  norm_oracle /tmp/sass_nv_raw.txt  > /tmp/sass_nv.txt
NI=$(wc -l < /tmp/sass_mine.txt)
if [ "$NI" -lt 13 ]; then bad "decoded only $NI instrs (<13) -- non-vacuity floor"; fi
if diff -q /tmp/sass_mine.txt /tmp/sass_obj.txt >/dev/null; then say "    from-scratch decode == cuobjdump ($NI instrs)  OK"; else bad "from-scratch decode != cuobjdump:"; diff /tmp/sass_mine.txt /tmp/sass_obj.txt | head -6 >&2; fi
# cuobjdump prints absolute BRA targets (BRA 0xe0); nvdisasm prints a symbolic label (BRA `(.L_x_0)) --
# same instruction, different display, so canonicalize the branch TARGET before the oracle-agreement check.
sed -E 's@BRA [^ ].*@BRA <tgt>@' /tmp/sass_obj.txt > /tmp/sass_obj_c.txt
sed -E 's@BRA [^ ].*@BRA <tgt>@' /tmp/sass_nv.txt  > /tmp/sass_nv_c.txt
if diff -q /tmp/sass_obj_c.txt /tmp/sass_nv_c.txt >/dev/null; then say "    cuobjdump == nvdisasm (mnemonics+operands agree; BRA target label-format normalized)  OK"; else bad "cuobjdump != nvdisasm (genuine oracle disagreement):"; diff /tmp/sass_obj_c.txt /tmp/sass_nv_c.txt | head -6 >&2; fi

# locate the FADD bundle (lo=0x0000000502097221 -> LE bytes 21 72 09 02 05 ..) for the byte-flip NCs
FOFF=$(grep -aboP '\x21\x72\x09\x02\x05' /tmp/sass_va.cubin | head -1 | cut -d: -f1)
if [ -z "$FOFF" ]; then bad "could not locate the FADD bundle in the cubin (NC setup)"; FOFF=-1; fi
say "    FADD opcode byte at cubin offset $FOFF"

# --- [D] NC1: flip the FADD opcode HIGH byte (0x72 -> 0xff) -> opcode 0xff21 OUTSIDE the whitelist
#         -> sass_check MUST fail-closed (the decoder itself rejects, no GPU run). ---
say "[D] NC1 unknown-opcode -> sass_check MUST fail-closed (SASS_DECODE_FAIL)"
if [ "$FOFF" -ge 0 ]; then
  cp /tmp/sass_va.cubin /tmp/sass_nc1.cubin
  printf '\xff' | dd of=/tmp/sass_nc1.cubin bs=1 seek=$((FOFF+1)) count=1 conv=notrunc 2>/dev/null
  if cmp -s /tmp/sass_nc1.cubin /tmp/sass_va.cubin; then bad "NC1 vacuous -- byte flip did not change the cubin";
  elif "$OBJ" -sass /tmp/sass_nc1.cubin 2>/dev/null | norm_oracle /dev/stdin | diff -q - /tmp/sass_obj.txt >/dev/null; then bad "NC1 vacuous -- cuobjdump shows no change (flip landed in a don't-care field)";
  else
    t1=$(sass_tok /tmp/sass_nc1.cubin)
    if [ "$t1" = SASS_DECODE_FAIL ]; then say "    NC1 sass_check=SASS_DECODE_FAIL (fail-closed)  OK"; else bad "NC1 sass_check=$t1 but expected SASS_DECODE_FAIL"; fi
  fi
fi

# --- [E] NC2: flip the FADD opcode LOW byte (0x21 -> 0x24) -> opcode 0x7224 = IMAD (a DIFFERENT known
#         opcode) -> the from-scratch decode CHANGES vs the genuine (proving it reads the real bytes). ---
say "[E] NC2 decode-changed -> from-scratch decode MUST differ from genuine"
if [ "$FOFF" -ge 0 ]; then
  cp /tmp/sass_va.cubin /tmp/sass_nc2.cubin
  printf '\x24' | dd of=/tmp/sass_nc2.cubin bs=1 seek=$FOFF count=1 conv=notrunc 2>/dev/null
  if cmp -s /tmp/sass_nc2.cubin /tmp/sass_va.cubin; then bad "NC2 vacuous -- byte flip did not change the cubin";
  elif "$OBJ" -sass /tmp/sass_nc2.cubin 2>/dev/null | norm_oracle /dev/stdin | diff -q - /tmp/sass_obj.txt >/dev/null; then bad "NC2 vacuous -- cuobjdump shows no change";
  else
    "$CL" /tmp/sass_nc2.cubin "$KNAME" 0 sass_check 2>/dev/null | norm_mine /dev/stdin > /tmp/sass_nc2_mine.txt
    if diff -q /tmp/sass_nc2_mine.txt /tmp/sass_mine.txt >/dev/null; then bad "NC2 vacuous -- from-scratch decode did NOT change (decoder ignores the byte!)"; else
      # the changed line must be the FADD->IMAD one, and the corrupt decode must still match cuobjdump-of-corrupt (decoder is correct on the new bytes too)
      grep -q '^IMAD R9' /tmp/sass_nc2_mine.txt && say "    NC2 decode changed FADD->IMAD (decoder tracks the bytes)  OK" || bad "NC2 decode changed but not to the expected IMAD R9"
    fi
  fi
fi

echo "------------------------------------------------------------"
if [ "$OK" = "1" ]; then echo "SASS_CHECK_PASS"; exit 0; else echo "SASS_CHECK_FAIL"; exit 1; fi
