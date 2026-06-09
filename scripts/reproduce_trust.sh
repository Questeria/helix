#!/usr/bin/env bash
# reproduce_trust.sh -- ONE-COMMAND clean-room reproduction of the Helix from-raw trust core.
#
# What it proves (CPU-only -- runnable on any x86-64 Linux, incl. a CI runner, with NO local state):
#   [1] static fence            : exactly 1 committed .py, 29 committed .c/.h
#                                 (22 from-raw ladder (byte-identical to v1.3, incl. seed.c) + 7 Category-B
#                                  host harnesses (cuda_launch.c grew +273 LOC post-v1.3, so the v1.3
#                                  24-file trusted-C set is NOT all unchanged -- only the 22-file from-raw
#                                  ladder is byte-identical), all OUTSIDE the self-host fixpoint. The 7 =
#                                  the 2 v1.3 GPU harnesses (cuda_launch.c + train_transformer.c) + the 5
#                                  newest GPT-2 demo host tools:
#                                    helixc/runtime/gpt2_infer.c  -- post-v1.3 GPT-2 demo launcher
#                                                                    (CUDA-FFI, ptxas boundary); now
#                                                                    also carries the ADDITIVE,
#                                                                    forward-only --serve mode (a 4th
#                                                                    branch + a printf-only telemetry
#                                                                    emit module) for the live chat
#                                                                    demo -- numeric path byte-identical,
#                                                                    fixpoint untouched,
#                                    helixc/runtime/cpu_host.c    -- post-v1.3 CPU no-ptxas demo
#                                                                    launcher (CUDA-FREE byte-movement
#                                                                    harness; ZERO arithmetic on the
#                                                                    trust path; all math in the
#                                                                    kovc-compiled gpt2_cpu_ops.hx),
#                                    helixc/runtime/gpt2_tok.c    -- offline byte-level BPE tokenizer
#                                                                    (Python-free demo data path; ZERO
#                                                                    arithmetic on the trust path); also
#                                                                    links into the --serve worker
#                                                                    (GPT2_TOK_LIB) for in-process,
#                                                                    Python-free prompt tokenization,
#                                    helixc/runtime/gpt2_pack.c   -- offline safetensors->.weights
#                                                                    importer (byte-movement only; ZERO
#                                                                    arithmetic on the trust path), and
#                                    helixc/runtime/gpt2_serve_http.c -- the NEW dependency-light, NO-
#                                                                    Python C HTTP+SSE server for the
#                                                                    live chat demo (POSIX sockets,
#                                                                    static files + /api/health +
#                                                                    /api/generate SSE bridge +
#                                                                    /api/verify; byte-pump only, ZERO
#                                                                    arithmetic on the trust path).
#                                  The Unicode range tables gpt2_tok.c uses live in a generated DATA
#                                  file, helixc/runtime/gpt2_unicode_ranges.inc -- a .inc, NOT a .c/.h,
#                                  so it is outside this fence (like a .hx).)
#   [2] from-raw ladder         : DELETE every pre-built rung binary, then rebuild hex0->...->seed
#                                 using ONLY the prior rung (hex0 from hand-authored hex via xxd);
#                                 each rung self-verifies its committed .sha256; seed == pinned.
#   [3] self-host fixpoint       : scripts/gate_kovc.sh -> K2==K3==K4 == pinned, corpus 109/0, check_err 4/0
#   [4] gcc diverse-double-compile: gcc (zero M2-Planet ancestry) and the from-raw seed both produce
#                                 a BYTE-IDENTICAL, pinned K1 (Wheeler trusting-trust defense).
# The GPU capstone is verified SEPARATELY by scripts/capstone_audit.sh on a CUDA host (no GPU here).
#
# This resolves the "no committed one-command ladder rebuild" gap: the trust spine is reproducible
# by anyone, push-button, on a clean checkout. Exit 0 ONLY if every check matches the pinned anchors.
#
# NOTE: intended for a CLEAN CHECKOUT (CI runner or a throwaway clone). It MODIFIES the working tree
# (the disclosed /mnt/c path rewrite + rung-binary rebuilds); do not run on a tree you want pristine.
# Tools required: bash 4+, coreutils (sha256sum/stat), xxd, file, objdump (binutils), gcc, grep, sed.
set -uo pipefail
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT"
FAIL=0
say(){ echo "[reproduce_trust] $*"; }
bad(){ echo "[reproduce_trust] *** FAIL: $*" >&2; FAIL=1; }

# Pinned release anchors (the values an independent run must reproduce):
SEED_SHA=9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb
K1_SHA=84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba
FIX_SHA=0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f

echo "============================================================"
echo " Helix from-raw trust-core reproduction"
echo " repo root : $ROOT"
echo " anchors   : seed=$SEED_SHA"
echo "             K1  =$K1_SHA"
echo "             fix =$FIX_SHA"
echo "============================================================"

# --- [0] disclosed path rewrite -----------------------------------------------------------------
# assemble_k1.hx (the fixpoint concatenator) + a few scripts hardcode the original absolute build
# path; rewrite it to THIS checkout so the build runs at any path. This is the disclosed portability
# caveat (docs/CLEAN_REPRODUCTION.md "Where it walls"); the rewrite is a pure mechanical path swap.
say "[0] path rewrite  /mnt/c/Projects/Kovostov-Native -> $ROOT"
mapfile -t HCFILES < <(grep -rlI '/mnt/c/Projects/Kovostov-Native' . 2>/dev/null || true)
if [ "${#HCFILES[@]}" -gt 0 ]; then
  printf '%s\n' "${HCFILES[@]}" | xargs sed -i "s#/mnt/c/Projects/Kovostov-Native#$ROOT#g"
  say "    rewrote ${#HCFILES[@]} file(s)"
fi

# --- [1] static fence ---------------------------------------------------------------------------
say "[1] static fence"
NPY=$(git ls-files "*.py" | wc -l | tr -d ' ')
NCH=$(git ls-files "*.c" "*.h" | wc -l | tr -d ' ')
if [ "$NPY" = "1" ]; then say "    committed .py = 1 ($(git ls-files '*.py'))"; else bad "committed .py = $NPY (want 1)"; fi
if [ "$NCH" = "29" ]; then say "    committed .c/.h = 29 (22 from-raw ladder [byte-identical to v1.3, incl. seed.c] + 7 Category-B host harnesses [OUTSIDE the self-host fixpoint, zero arithmetic on the trust path]; the 5 newest are the GPT-2 demo tools: gpt2_infer.c GPU demo [+additive forward-only --serve mode] + cpu_host.c CPU no-ptxas demo + gpt2_tok.c byte-level BPE tokenizer [+--serve in-process tok] + gpt2_pack.c safetensors importer + gpt2_serve_http.c NO-Python C HTTP+SSE chat server)"; else bad "committed .c/.h = $NCH (want 29)"; fi

# --- [2] from-raw ladder ------------------------------------------------------------------------
say "[2] from-raw ladder (deleting pre-built rung binaries first, then rebuilding each from the prior)"
rm -f stage0/hex0/hex0.bin stage0/hex1/hex1.bin stage0/hex2/hex2.bin stage0/catm/catm.bin \
      stage0/M0/M0.bin stage0/cc_amd64/cc_amd64.bin stage0/M2-Planet/M2.bin \
      stage0/helixc-bootstrap/seed.bin
LADDER_OK=1
for rung in hex0 hex1 hex2 catm M0 cc_amd64 M2-Planet helixc-bootstrap; do
  if ( cd "stage0/$rung" && bash build.sh ) >"/tmp/rt_${rung}.log" 2>&1; then
    say "    rung $rung : build + self-verify OK"
  else
    bad "rung $rung build/verify failed (tail of /tmp/rt_${rung}.log):"; tail -8 "/tmp/rt_${rung}.log" >&2
    LADDER_OK=0; break
  fi
done
if [ "$LADDER_OK" = "1" ] && [ -s stage0/helixc-bootstrap/seed.bin ]; then
  GOT=$(sha256sum stage0/helixc-bootstrap/seed.bin | cut -d' ' -f1)
  if [ "$GOT" = "$SEED_SHA" ]; then say "    seed.bin == pinned ($SEED_SHA)"; else bad "seed.bin $GOT != pinned $SEED_SHA"; fi
else
  bad "ladder did not produce seed.bin"
fi

# --- [3] self-host fixpoint + corpus + PTX-text + diagnostics ------------------------------------
say "[3] self-host fixpoint gate (scripts/gate_kovc.sh)"
bash scripts/gate_kovc.sh >/tmp/rt_gate.log 2>&1 || true
if grep -q '^GATE_PASS' /tmp/rt_gate.log; then say "    GATE_PASS"; else bad "gate did not reach GATE_PASS (tail):"; tail -15 /tmp/rt_gate.log >&2; fi
if grep -q 'FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)' /tmp/rt_gate.log; then say "    fixpoint K2==K3==K4 == pinned ($FIX_SHA)"; else bad "fixpoint not pinned-OK"; fi
if grep -q 'CORPUS: 109 passed, 0 failed' /tmp/rt_gate.log; then say "    corpus 109/0"; else bad "corpus not 109/0"; fi
if grep -q 'CHECK_ERR: 4 passed, 0 failed' /tmp/rt_gate.log; then say "    check_err 4/0"; else bad "check_err not 4/0"; fi

# --- [4] gcc diverse-double-compile -------------------------------------------------------------
say "[4] gcc diverse-double-compile (stage0/helixc-bootstrap/ddc_crosscheck.sh)"
bash stage0/helixc-bootstrap/ddc_crosscheck.sh >/tmp/rt_ddc.log 2>&1 || true
if grep -q 'DDC_ANCHOR_OK' /tmp/rt_ddc.log; then say "    DDC_ANCHOR_OK"; else bad "DDC not OK (tail):"; tail -15 /tmp/rt_ddc.log >&2; fi
if grep -q "$K1_SHA" /tmp/rt_ddc.log; then say "    K1 byte-identical (gcc == M2-seed) == pinned ($K1_SHA)"; else bad "K1 != pinned $K1_SHA"; fi

# --- verdict ------------------------------------------------------------------------------------
echo "============================================================"
if [ "$FAIL" = "0" ]; then
  echo "REPRODUCE_TRUST: PASS"
  echo "  from-raw ladder + self-host fixpoint + gcc-DDC all reproduce the pinned anchors from a clean checkout."
  echo "  (GPU capstone is verified separately by scripts/capstone_audit.sh on a CUDA host.)"
  exit 0
else
  echo "REPRODUCE_TRUST: FAIL -- at least one check did not match the pinned anchors (see *** FAIL lines above)."
  exit 1
fi
