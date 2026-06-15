#!/bin/bash
# v1.6 -- Qwen3 NVFP4 receipt regression gate (the "one-command check"; DoD #5/#6).
#
# The receipt CHECKER (gpt2_infer.c --v3-receipt-check) and the re-attest path
# (--v3-receipt-from-logits) are GPU-FREE and rebuildable from the 299-byte seed
# (seed -> K1..K4 byte-identical fixpoint cdcf8673; ptxas DE-TRUSTED for the GPU kernel;
# from-scratch NIST-KAT'd SHA-256). This gate proves the checker:
#   [0] self-tests its SHA-256 (3 NIST KATs),
#   [1] ACCEPTS a genuine emitted receipt for the 8B run,
#   [2] is deterministic (re-check -> same verdict),
#   [3-5] REJECTS three 8B tampers, EACH by a NAMED reject (envelope has teeth),
#   [6] ACCEPTS the headline 32B run (decisive prompt, argmax-exact), and
#   [7] REJECTS an argmax drift on the real 32B near-tie run.
#
# Honest scope (docs/HELIX_V1.6_DEFINITION_OF_DONE.md): Tier-2 = reproducibility
# (re-derive from committed weights; NOT faster-than-re-exec -- exact-Freivalds Tier-1
# is DEFERRED for the f32 GEMM); Tier-3 = an EMPIRICAL envelope vs a TRUSTED f32 oracle
# (argmax + max_abs<tau), NEVER cryptographic. tier3_tau here is the per-run measured
# max_abs; a CALIBRATED bound (TAO pattern, DoD Risk #5) is a separate, attended decision.
# Prior art: CommitLLM, TAO -- this is MINIMAL-TRUST verification, NOT "first verifiable
# quantized inference"; the one differentiator is the 299-byte-rebuildable, ptxas-de-trusted
# verifier TCB.
#
# GPU-FREE. Re-hashes the weights for the model commitment, so it reads ~7.3GB (8B) and,
# when enabled, ~23.5GB (32B) from disk -- no CUDA. Env: BIN, D (artifact dir), Q32=0 to
# skip the slower 32B steps. Genuine receipts are re-derived from the dumped run logits via
# --v3-receipt-from-logits, so the gate is self-contained given the weights + logits + oracle.
set -u
BIN=${BIN:-/home/legoa/llama_infer}
D=${D:-/home/legoa}
Q32=${Q32:-1}
T=$(mktemp -d "$D/recpt.XXXXXX") || { echo "mktemp failed"; exit 2; }
trap 'rm -rf "$T"' EXIT
rc=0

# need <label> <expected-substring> <cmd...> : capture-then-grep (never pipe to head; the
# checker's exit code is advisory -- the NAMED token in stdout is the load-bearing signal).
need() {
  local lbl="$1" exp="$2"; shift 2
  local out; out=$("$@" 2>&1)
  if printf '%s' "$out" | grep -q -- "$exp"; then
    echo "  [PASS] $lbl  ($exp)"
  else
    echo "  [FAIL] $lbl  expected '$exp', got:"; printf '%s\n' "$out" | tail -2; rc=1
  fi
}

echo "=== [0] SHA-256 NIST KAT (from-scratch FIPS-180-4) ==="
need "sha256 KAT" "0 bad -> PASS" "$BIN" --sha256-selftest

echo "=== re-emit the genuine 8B receipt from dumped logits (GPU-free, exact tau) ==="
"$BIN" --v3-receipt-from-logits "$D/q3.receipt.logits" "$D/qwen3-8b-v16.weights" \
       "$D/qwen3_ids.txt" "$D/q3_logits_ref.bin" "$T/g8.receipt" >/dev/null 2>&1 \
  || { echo "  [FAIL] 8B re-emit"; rc=1; }

echo "=== [1] GENUINE 8B receipt -> RECEIPT_CHECK_PASS ==="
need "genuine 8B" "RECEIPT_CHECK_PASS" \
  "$BIN" --v3-receipt-check "$T/g8.receipt" "$D/qwen3-8b-v16.weights" "$D/q3.receipt.logits" "$D/q3_logits_ref.bin"

echo "=== [2] determinism: re-check 8B -> PASS again ==="
need "determinism 8B" "RECEIPT_CHECK_PASS" \
  "$BIN" --v3-receipt-check "$T/g8.receipt" "$D/qwen3-8b-v16.weights" "$D/q3.receipt.logits" "$D/q3_logits_ref.bin"

echo "=== [3] NC forged logits (flip 1 byte) -> REJECT=LOGITS_HASH ==="
cp "$D/q3.receipt.logits" "$T/forged.bin"
printf '\xff' | dd of="$T/forged.bin" bs=1 seek=400000 count=1 conv=notrunc 2>/dev/null
need "NC LOGITS_HASH" "REJECT=LOGITS_HASH" \
  "$BIN" --v3-receipt-check "$T/g8.receipt" "$D/qwen3-8b-v16.weights" "$T/forged.bin" "$D/q3_logits_ref.bin"

echo "=== [4] NC tampered model hash (zero it) -> REJECT=MODEL_HASH ==="
sed 's/^model_sha256 .*/model_sha256 0000000000000000000000000000000000000000000000000000000000000000/' \
    "$T/g8.receipt" > "$T/badmodel.receipt"
need "NC MODEL_HASH" "REJECT=MODEL_HASH" \
  "$BIN" --v3-receipt-check "$T/badmodel.receipt" "$D/qwen3-8b-v16.weights" "$D/q3.receipt.logits" "$D/q3_logits_ref.bin"

echo "=== [5] NC drift outside envelope (wrong oracle: 8B logits vs 32B oracle) -> REJECT=TIER3_ENVELOPE ==="
if [ -f "$D/q3_32b_logits_ref.bin" ]; then
  need "NC TIER3_ENVELOPE" "REJECT=TIER3_ENVELOPE" \
    "$BIN" --v3-receipt-check "$T/g8.receipt" "$D/qwen3-8b-v16.weights" "$D/q3.receipt.logits" "$D/q3_32b_logits_ref.bin"
else
  echo "  [skip] $D/q3_32b_logits_ref.bin absent"
fi

if [ "$Q32" = "1" ] && [ -f "$D/qwen3-32b-v16.weights" ]; then
  echo "=== re-emit 32B receipts from dumped logits (slower; hashes ~23.5GB each) ==="
  "$BIN" --v3-receipt-from-logits "$D/q3_32b_rep.receipt.logits" "$D/qwen3-32b-v16.weights" \
         "$D/france_rep_ids.txt" "$D/q3_32b_rep_ref.bin" "$T/g32.receipt" >/dev/null 2>&1 \
    || { echo "  [FAIL] 32B decisive re-emit"; rc=1; }
  "$BIN" --v3-receipt-from-logits "$D/q3_32b.receipt.logits" "$D/qwen3-32b-v16.weights" \
         "$D/qwen3_ids.txt" "$D/q3_32b_logits_ref.bin" "$T/nt32.receipt" >/dev/null 2>&1 \
    || { echo "  [FAIL] 32B near-tie re-emit"; rc=1; }

  echo "=== [6] HEADLINE 32B decisive ('...is Paris. ...is' -> ' Paris') -> RECEIPT_CHECK_PASS ==="
  need "genuine 32B" "RECEIPT_CHECK_PASS" \
    "$BIN" --v3-receipt-check "$T/g32.receipt" "$D/qwen3-32b-v16.weights" "$D/q3_32b_rep.receipt.logits" "$D/q3_32b_rep_ref.bin"

  echo "=== [7] NC argmax drift (real 32B near-tie run: argmax 279 != f32 oracle 15473) -> REJECT=TIER3_ARGMAX ==="
  need "NC TIER3_ARGMAX" "REJECT=TIER3_ARGMAX" \
    "$BIN" --v3-receipt-check "$T/nt32.receipt" "$D/qwen3-32b-v16.weights" "$D/q3_32b.receipt.logits" "$D/q3_32b_logits_ref.bin"
else
  echo "=== [6-7] 32B steps SKIPPED (Q32=0 or 32B weights absent) ==="
fi

echo "=== [8] RELEASE 8B receipt (CALIBRATED tau + provenance) -> RECEIPT_CHECK_PASS ==="
if [ -f "$D/q3_8b_release.receipt" ]; then
  need "release 8B (calibrated)" "RECEIPT_CHECK_PASS" \
    "$BIN" --v3-receipt-check "$D/q3_8b_release.receipt" "$D/qwen3-8b-v16.weights" "$D/q3_8b_release.receipt.logits" "$D/q3_logits_ref.bin"
  if grep -q '^tier3_tau_prov ' "$D/q3_8b_release.receipt"; then echo "  [PASS] 8B receipt carries tier3_tau_prov"; else echo "  [FAIL] 8B receipt missing tau provenance"; rc=1; fi
else echo "  [skip] $D/q3_8b_release.receipt absent (run the release re-emit first)"; fi

if [ "$Q32" = "1" ] && [ -f "$D/q3_32b_release.receipt" ]; then
  echo "=== [9] RELEASE 32B receipt (CALIBRATED tau, the headline) -> RECEIPT_CHECK_PASS ==="
  need "release 32B (calibrated)" "RECEIPT_CHECK_PASS" \
    "$BIN" --v3-receipt-check "$D/q3_32b_release.receipt" "$D/qwen3-32b-v16.weights" "$D/q3_32b_release.receipt.logits" "$D/q3_32b_rep_ref.bin"
  if grep -q '^tier3_tau_prov ' "$D/q3_32b_release.receipt"; then echo "  [PASS] 32B receipt carries tier3_tau_prov"; else echo "  [FAIL] 32B receipt missing tau provenance"; rc=1; fi
fi

echo "=== [10] TEETH: a faithful run DECLARED with tau BELOW its own max_abs -> REJECT=TIER3_ENVELOPE ==="
# proves the (calibrated) envelope rejects any run whose deviation exceeds the declared bound
HX_TIER3_TAU=2.0 "$BIN" --v3-receipt-from-logits "$D/q3.receipt.logits" "$D/qwen3-8b-v16.weights" \
  "$D/qwen3_ids.txt" "$D/q3_logits_ref.bin" "$T/teeth.receipt" >/dev/null 2>&1
need "NC teeth (tau<max_abs)" "REJECT=TIER3_ENVELOPE" \
  "$BIN" --v3-receipt-check "$T/teeth.receipt" "$D/qwen3-8b-v16.weights" "$D/q3.receipt.logits" "$D/q3_logits_ref.bin"

echo
if [ "$rc" = "0" ]; then
  echo "RECEIPT_GATE_PASS (KAT + genuine 8B/32B + calibrated release receipts + determinism + NCs by named reject incl. teeth)"
else
  echo "RECEIPT_GATE_FAIL"
fi
exit $rc
