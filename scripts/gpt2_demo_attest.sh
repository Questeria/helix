#!/usr/bin/env bash
# gpt2_demo_attest.sh -- ONE-COMMAND, FAIL-CLOSED trust proof for the GPT-2-on-Helix investor demo (P6).
#
#   bash scripts/gpt2_demo_attest.sh
#
# COMPOSES the two already-verified, committed gates -- it does NOT re-derive their internals:
#   [A] scripts/reproduce_trust.sh   -> REPRODUCE_TRUST: PASS   (from-raw chain: 299-byte hex0 -> seed
#                                       9837db12, self-host fixpoint 0992dddd, gcc-DDC K1 84363adb)
#   [B] scripts/gpt2_gpu_mvp.sh      -> GPT2_LOGITS_PARITY_PASS + GPT2_GENERATE_MATCH_PASS
#                                       (GPT-2 124M full-logits parity + token-for-token greedy gen on
#                                        the GPU through kovc-emitted PTX minted from the raw seed)
#   [C] REPRODUCIBILITY SHOT (DoD #4): the generation is run a SECOND time and the produced
#                                       generated-ids artifact must be BYTE-IDENTICAL (two equal sha256).
#
# Then it computes the LIVE model.safetensors sha256 (gitignored; NOT a hardcoded repo anchor) and
# emits attestation/gpt2_attest.txt binding source/from-raw anchors -> GPT-2 output, WITH an explicit
# HONEST RESIDUALS section. Any failed leg -> the attestation is NOT written, DEMO_ATTEST_FAIL, exit 1.
#
# STRICTLY SERIAL: reproduce_trust (leg A) and the GPU run (legs B/C) never overlap -- one at a time.
# This wrapper never fakes/stubs a leg; every PASS line is parsed from a real run's stdout.
#
# Run as a FILE under WSL (bash). The two legs handle the DrvFs-write dodge / seed provenance / fresh
# PTX mint internally (gpt2_gpu_mvp.sh mirrors to ext4 and re-checks seed sha == pinned 9837db12);
# reproduce_trust.sh runs the full from-raw ladder + git fence + self-host fixpoint + gcc-DDC on the
# committed /mnt/c checkout (the ext4 mirror is only the bootstrap subset, so the full ladder needs the
# real checkout). It is the slow leg on DrvFs but it MUST genuinely run + print PASS, never stubbed.
set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
cd "$ROOT"

# Pinned trust anchors (the three from-raw values the demo binds to; bound here for the attestation,
# and asserted against reproduce_trust.sh's own output below).
SEED_SHA=9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb
K1_SHA=84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba
FIX_SHA=0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f

# Live paths (literals -- never assign a /mnt path to a variable inside a Git-Bash `bash -c`; here we
# run under real WSL bash from a file, so /mnt assignment is fine, but we keep them as literals for
# clarity and to match the gates).
SAFETENSORS="$ROOT/helix-llm/models/gpt2/model.safetensors"
TOOLS="$ROOT/helix-llm/tools"
ATTEST_DIR="$ROOT/attestation"
ATTEST="$ATTEST_DIR/gpt2_attest.txt"
PROMPT="The capital of France is"        # the pinned demo prompt (matches gpt2_gpu_mvp.sh)
NGEN=20                                   # greedy continuation length (matches gpt2_gpu_mvp.sh)

OK=1
say(){ echo "[demo_attest] $*"; }
bad(){ echo "[demo_attest] *** FAIL: $*" >&2; OK=0; }

echo "============================================================"
echo " GPT-2-on-Helix demo attestation  (fail-closed, strictly serial)"
echo " repo root : $ROOT"
echo " prompt    : \"$PROMPT\"   N_gen=$NGEN"
echo "============================================================"

# =================================================================================================
# LEG [A] -- from-raw trust core (reproduce_trust.sh).  Run FIRST, alone (serial).
# =================================================================================================
say "[A] from-raw trust core: reproduce_trust.sh on a FRESH ext4 clone (clean-checkout; /mnt/c DrvFs would be ~75x slower on the byte-level rung builds)"
RT_LOG=/tmp/demo_attest_reproduce.log
RT_CLONE=/home/legoa/helix_attest_clone
rm -f "$RT_LOG"; rm -rf "$RT_CLONE"
# A fresh clone of the committed repo to ext4 is exactly the "clean checkout" reproduce_trust.sh is
# designed for (it deletes every pre-built rung binary + rebuilds hex0->seed, asserting the pinned
# anchors), and it is the honest from-raw claim -- no local build state leaks in. ~66s on ext4.
if git clone -q "$ROOT" "$RT_CLONE" >>"$RT_LOG" 2>&1; then
  ( cd "$RT_CLONE" && bash scripts/reproduce_trust.sh ) >>"$RT_LOG" 2>&1
  RT_RC=$?
else
  echo "[demo_attest] clone to ext4 failed" >>"$RT_LOG"; RT_RC=99
fi
if [ "$RT_RC" -ne 0 ]; then
  bad "reproduce_trust.sh exited nonzero (rc=$RT_RC); tail:"; tail -20 "$RT_LOG" >&2
elif grep -q '^REPRODUCE_TRUST: PASS' "$RT_LOG"; then
  say "    REPRODUCE_TRUST: PASS"
else
  bad "reproduce_trust.sh did not print 'REPRODUCE_TRUST: PASS'; tail:"; tail -20 "$RT_LOG" >&2
fi
# Confirm the three pinned anchors actually appear in the reproduction output (not just a bare PASS).
if [ "$OK" = "1" ]; then
  grep -q "$SEED_SHA" "$RT_LOG" || bad "seed anchor $SEED_SHA not corroborated in reproduce_trust output"
  grep -q "$FIX_SHA"  "$RT_LOG" || bad "fixpoint anchor $FIX_SHA not corroborated in reproduce_trust output"
  grep -q "$K1_SHA"   "$RT_LOG" || bad "gcc-DDC K1 anchor $K1_SHA not corroborated in reproduce_trust output"
  [ "$OK" = "1" ] && say "    three anchors corroborated in reproduce_trust output (seed/fixpoint/K1)"
fi
[ "$OK" = "1" ] || { echo "DEMO_ATTEST_FAIL"; echo "(leg A failed -- attestation NOT written)"; exit 1; }

# =================================================================================================
# LEG [B] -- GPT-2 124M inference + parity + first generation (gpt2_gpu_mvp.sh).  Run AFTER leg A,
# alone (serial -- reproduce_trust has fully returned).  Captures both PASS gates + generated text.
# =================================================================================================
say "[B] GPT-2 124M inference + parity: bash scripts/gpt2_gpu_mvp.sh (serial, after leg A)"
MVP_LOG=/tmp/demo_attest_mvp.log
rm -f "$MVP_LOG" /tmp/helix_gen_ids.txt /tmp/helix_logits_last.bin
MSYS_NO_PATHCONV=1 bash scripts/gpt2_gpu_mvp.sh >"$MVP_LOG" 2>&1
MVP_RC=$?
if [ "$MVP_RC" -ne 0 ]; then
  bad "gpt2_gpu_mvp.sh exited nonzero (rc=$MVP_RC); tail:"; tail -25 "$MVP_LOG" >&2
fi
grep -q '^GPT2_LOGITS_PARITY_PASS'  "$MVP_LOG" || bad "GPT2_LOGITS_PARITY_PASS missing from gpt2_gpu_mvp.sh output"
grep -q '^GPT2_GENERATE_MATCH_PASS' "$MVP_LOG" || bad "GPT2_GENERATE_MATCH_PASS missing from gpt2_gpu_mvp.sh output"
[ "$OK" = "1" ] && say "    GPT2_LOGITS_PARITY_PASS + GPT2_GENERATE_MATCH_PASS"
[ "$OK" = "1" ] || { echo "DEMO_ATTEST_FAIL"; echo "(leg B failed -- attestation NOT written)"; exit 1; }

# --- harvest the parity numbers + the generation artifact produced by leg B (parsed from the gate's
#     own real output / artifacts -- nothing is invented here) ----------------------------------
# The launcher (--logits) printed e.g.:
#   helix argmax=<id> (logit=..)  oracle argmax=<id>  -> ARGMAX_MATCH
#   max_abs logit diff=<v> at id <i> (helix=.. ref=..)  [diag bar 5e-2: ok]
PARITY_LINE=$(grep -m1 'max_abs logit diff=' "$MVP_LOG" | sed 's/^ *//;s/^[[:space:]]*//')
ARGMAX_LINE=$(grep -m1 'helix argmax=' "$MVP_LOG" | sed 's/^ *//;s/^[[:space:]]*//')
[ -n "$PARITY_LINE" ] || bad "could not parse 'max_abs logit diff=' from gpt2_gpu_mvp.sh output"
[ -n "$ARGMAX_LINE" ] || bad "could not parse 'helix argmax=' from gpt2_gpu_mvp.sh output"

# The generated-ids artifact (prompt ids + N greedy ids) -- this is the byte-identical-output anchor.
GEN_IDS_FILE=/tmp/helix_gen_ids.txt
[ -s "$GEN_IDS_FILE" ] || bad "leg B produced no /tmp/helix_gen_ids.txt (generation artifact missing)"
GEN_IDS=$(tr '\n' ' ' < "$GEN_IDS_FILE" | sed 's/[[:space:]]\+/ /g;s/^ //;s/ $//')
RUN1_SHA=""
[ -s "$GEN_IDS_FILE" ] && RUN1_SHA=$(sha256sum "$GEN_IDS_FILE" | cut -d' ' -f1)

# Decode the helix-produced ids to text via the FENCED oracle BPE (host-glue rendering only; not a
# compute-trust step). This is exactly how gpt2_gpu_mvp.sh renders the demo text.
GEN_TEXT=""
if [ "$OK" = "1" ] && [ -n "$GEN_IDS" ]; then
  GEN_TEXT=$( cd "$TOOLS" && python3 gpt2_numpy_ref.py decode $GEN_IDS 2>/dev/null )
  [ -n "$GEN_TEXT" ] || bad "fenced oracle could not decode the generated ids to text"
fi
GEN_TEXT_SHA=""
[ -n "$GEN_TEXT" ] && GEN_TEXT_SHA=$(printf '%s' "$GEN_TEXT" | sha256sum | cut -d' ' -f1)
[ "$OK" = "1" ] || { echo "DEMO_ATTEST_FAIL"; echo "(leg B harvest failed -- attestation NOT written)"; exit 1; }
say "    generated ids: [$GEN_IDS]"
say "    generated text: $(printf '%q' "$GEN_TEXT")"
say "    run#1 gen-ids sha256: $RUN1_SHA"

# =================================================================================================
# LEG [C] -- REPRODUCIBILITY SHOT (DoD #4): run the generation a SECOND time and require the
# generated-ids artifact to be BYTE-IDENTICAL.  Strictly serial: leg B has fully returned; we reuse
# the just-built launcher (/tmp/gpt2_infer), the freshly-minted PTX (/tmp/gpt2_mvp.ptx), the weights,
# and the oracle's prompt/ref artifacts that leg B already produced -- a clean greedy re-run.
# =================================================================================================
say "[C] reproducibility shot: re-run greedy generation, assert byte-identical generated-ids (DoD #4)"
WEIGHTS_EXT4=/home/legoa/gpt2_ext4/gpt2_124M.weights      # ext4 weight mirror set up by leg B
REFDIR="$ROOT/helix-llm/ref"
REPRO_LOG=/tmp/demo_attest_repro.log
RUN2_SHA=""
rm -f "$REPRO_LOG"
# Sanity: the artifacts leg B built must still be present for an honest re-run (never fabricate).
if [ ! -x /tmp/gpt2_infer ];        then bad "leg C: /tmp/gpt2_infer launcher missing (leg B should have built it)"; fi
if [ ! -s /tmp/gpt2_mvp.ptx ];      then bad "leg C: /tmp/gpt2_mvp.ptx missing (leg B should have minted it from the raw seed)"; fi
if [ ! -s "$WEIGHTS_EXT4" ];        then bad "leg C: ext4 weight mirror missing ($WEIGHTS_EXT4)"; fi
if [ ! -s "$REFDIR/ref_ids.txt" ];  then bad "leg C: $REFDIR/ref_ids.txt (prompt ids) missing"; fi
if [ ! -s "$REFDIR/ref_gen_ids.txt" ]; then bad "leg C: $REFDIR/ref_gen_ids.txt (oracle gen ids) missing"; fi
if [ "$OK" = "1" ]; then
  rm -f "$GEN_IDS_FILE"   # remove leg B's artifact; require leg C to freshly re-create it (staleness guard)
  ( ulimit -s unlimited; /tmp/gpt2_infer /tmp/gpt2_mvp.ptx "$WEIGHTS_EXT4" --generate "$NGEN" \
       "$REFDIR/ref_ids.txt" "$REFDIR/ref_gen_ids.txt" ) >"$REPRO_LOG" 2>&1
  C_RC=$?
  if [ ! -s "$GEN_IDS_FILE" ]; then
    bad "leg C produced no fresh /tmp/helix_gen_ids.txt (rc=$C_RC); tail:"; tail -10 "$REPRO_LOG" >&2
  elif ! grep -q '^GPT2_GENERATE_MATCH_PASS' "$REPRO_LOG"; then
    bad "leg C second generation did not print GPT2_GENERATE_MATCH_PASS (rc=$C_RC); tail:"; tail -10 "$REPRO_LOG" >&2
  else
    RUN2_SHA=$(sha256sum "$GEN_IDS_FILE" | cut -d' ' -f1)
    if [ -n "$RUN1_SHA" ] && [ "$RUN1_SHA" = "$RUN2_SHA" ]; then
      say "    BYTE-IDENTICAL across two runs: $RUN2_SHA"
    else
      bad "reproducibility shot MISMATCH: run#1=$RUN1_SHA  run#2=$RUN2_SHA"
    fi
  fi
fi
[ "$OK" = "1" ] || { echo "DEMO_ATTEST_FAIL"; echo "(leg C reproducibility shot failed -- attestation NOT written)"; exit 1; }

# =================================================================================================
# LEG [4] -- LIVE model.safetensors sha256 (gitignored input; computed live, NEVER hardcoded).
# =================================================================================================
say "[4] live model.safetensors sha256 (computed now, not hardcoded)"
[ -s "$SAFETENSORS" ] || bad "model.safetensors missing at $SAFETENSORS"
SAFET_SHA=""
SAFET_BYTES=""
if [ "$OK" = "1" ]; then
  SAFET_SHA=$(sha256sum "$SAFETENSORS" | cut -d' ' -f1)
  SAFET_BYTES=$(stat -c%s "$SAFETENSORS")
  say "    model.safetensors sha256=$SAFET_SHA ($SAFET_BYTES B)"
fi
[ "$OK" = "1" ] || { echo "DEMO_ATTEST_FAIL"; echo "(leg 4 failed -- attestation NOT written)"; exit 1; }

# =================================================================================================
# LEG [5] -- emit the attestation (only reached on ALL-GREEN).  Bind everything + honest residuals.
# =================================================================================================
say "[5] all legs green -> writing attestation"
ATTEST_DATE=$(date '+%Y-%m-%d %H:%M:%S %Z')
GIT_HEAD=$(git rev-parse HEAD 2>/dev/null || echo "unknown")
GIT_DIRTY="clean"
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then GIT_DIRTY="dirty (working tree has uncommitted changes)"; fi
# The oracle's own decoded reference text (independent f64 numpy oracle) for the same prompt+N, if leg
# B dumped it -- corroborates the decoded string (token sequence already matched exactly in leg B).
ORACLE_TEXT_REF=""
[ -s "$REFDIR/ref_gen_text.txt" ] && ORACLE_TEXT_REF=$(cat "$REFDIR/ref_gen_text.txt")

mkdir -p "$ATTEST_DIR"
ATTEST_TMP="$ATTEST.tmp.$$"
{
cat <<EOF
================================================================================
  GPT-2 124M on Helix -- TRUST ATTESTATION
  (generated by scripts/gpt2_demo_attest.sh; fail-closed; all legs verified live)
================================================================================
date            : $ATTEST_DATE
repo HEAD        : $GIT_HEAD
working tree     : $GIT_DIRTY

--------------------------------------------------------------------------------
INPUT
--------------------------------------------------------------------------------
prompt                  : "$PROMPT"
greedy continuation N   : $NGEN
model.safetensors       : $SAFETENSORS
model.safetensors sha256: $SAFET_SHA
model.safetensors bytes : $SAFET_BYTES
  (computed LIVE this run; the safetensors is gitignored and is NOT a committed repo trust anchor.)

--------------------------------------------------------------------------------
FROM-RAW TRUST ANCHORS  (reproduce_trust.sh -> REPRODUCE_TRUST: PASS, verified live this run)
--------------------------------------------------------------------------------
seed.bin (from 299-byte hex0)   : $SEED_SHA
self-host fixpoint (K2==K3==K4) : $FIX_SHA
gcc diverse-double-compile (K1) : $K1_SHA
  The toolchain rebuilds from a 299-byte hand-authored hex0 to the pinned seed, reproduces itself
  byte-identically (self-host fixpoint), and an independent compiler (gcc, zero M2-Planet ancestry)
  corroborates the K1 binary byte-for-byte (Wheeler trusting-trust defense). All three matched their
  pinned values on this run.

--------------------------------------------------------------------------------
OUTPUT  (GPT-2 124M, unchanged public weights, forward re-expressed in Helix -> kovc-emitted PTX)
--------------------------------------------------------------------------------
generated token ids     : $GEN_IDS
generated text          : "$GEN_TEXT"
generated text sha256   : $GEN_TEXT_SHA
generated-ids artifact sha256 : $RUN1_SHA
oracle reference text   : $ORACLE_TEXT_REF

--------------------------------------------------------------------------------
PARITY  (vs the fenced pure-numpy f64 oracle; same prompt)
--------------------------------------------------------------------------------
$ARGMAX_LINE
$PARITY_LINE
GPT2_LOGITS_PARITY_PASS    -- last real-token argmax matches the oracle EXACTLY; max-abs logit diff
                              under the documented diagnostic bar (5e-2) after 12-layer fp32 drift.
GPT2_GENERATE_MATCH_PASS   -- the $NGEN-token greedy continuation matches the oracle TOKEN-FOR-TOKEN.

--------------------------------------------------------------------------------
REPRODUCIBILITY SHOT  (DoD item 4 -- byte-identical output across two independent runs)
--------------------------------------------------------------------------------
run #1 generated-ids sha256 : $RUN1_SHA
run #2 generated-ids sha256 : $RUN2_SHA
verdict                     : BYTE-IDENTICAL (run#1 == run#2)

--------------------------------------------------------------------------------
HONEST RESIDUALS  (the edges -- disclosed, not hidden)
--------------------------------------------------------------------------------
* Fenced host glue, NOT part of the from-raw compute trust. The weight importer, the byte-level BPE
  tokenizer, and the pure-numpy reference oracle live under gitignored helix-llm/. They are trusted
  host glue: no weights of their own, no role in the compute-trust chain. The decoded text above is
  rendered by that fenced tokenizer; the trust claim is the EXACT token-id sequence + the from-raw
  toolchain that produced it, not the host-side string rendering.
* GPU path is "complete to PTX, not to SASS." Source -> PTX is hand-auditable (hex0 -> kovc -> PTX);
  BELOW PTX, NVIDIA's closed ptxas + the GPU driver + the C CUDA-FFI launcher are trusted-once. This
  attestation does NOT claim a fully verified GPU, nor completeness to GPU machine code.
* fp32-only. All arithmetic is single-precision; parity is fp32-vs-f64 within a measured tolerance on
  hidden states, EXACT on argmax + the token sequence. This bounds the scale the stack generalizes to.
* Single GPU, sm_86 (one RTX 3070-class device). Not multi-GPU, not a cluster.
* This is a 124M-parameter demonstration, not frontier scale.
* The oracle shares the GPT-2 architecture SPEC (independent implementation in f64, not an independent
  specification). It catches implementation bugs, not a shared misunderstanding of GPT-2.
* NOT claimed: beating cuBLAS; "fully verified GPU"; completeness to GPU machine code; AGI. What is
  claimed: a 124M model, imported unchanged, generating text on a toolchain rebuildable from 299 bytes,
  output-matched token-for-token to an independent reference, and bit-for-bit reproducible.

--------------------------------------------------------------------------------
SELF-CONTAINED SIGNATURE
--------------------------------------------------------------------------------
The "signature" is this document's own content hash (no network signing service). Verify with:
    sha256sum attestation/gpt2_attest.txt
An owner may additionally produce an offline detached signature (gpg --detach-sign).
================================================================================
DEMO_ATTEST_PASS
EOF
} > "$ATTEST_TMP"

mv -f "$ATTEST_TMP" "$ATTEST"
ATTEST_SELF_SHA=$(sha256sum "$ATTEST" | cut -d' ' -f1)

echo "============================================================"
echo "DEMO_ATTEST_PASS"
echo "  attestation : $ATTEST"
echo "  self-sha256 : $ATTEST_SELF_SHA"
echo "============================================================"
exit 0
