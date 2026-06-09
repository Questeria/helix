# /api/verify — the honest on-demand parity check: UX + contract (DESIGN; backend NEEDS GPU GATE)

**Status:** frontend renderer SHIPPED (branch `fable/features-and-longshots`, renders only
server-supplied fields); backend UNCHANGED — `/api/verify` still returns `UNAVAILABLE` by design.
**Nothing in this document weakens the no-fake-verdict invariant: until the gated backend below
exists, the only verdict a live page can ever show is UNAVAILABLE.**

## 1. Why UNAVAILABLE-by-design was right (and stays the default)

The hot path is Python-free; the fp32 oracle is a ~multi-GB numpy load; and a verdict computed by
the same box mid-demo is weaker than the committed offline gates. The chat page therefore never
shows a parity verdict today — the numbers live on page 2. That stays the DEFAULT deployment.

## 2. The future REAL check (when the operator opts in)

`POST /api/verify {gen_ids:[...], request_id?, model?}` MAY, on an operator-enabled deployment
(`--verify-oracle <path>` flag present AND python3 + the fenced oracle installed), re-run the real
comparison for EXACTLY the ids just produced: oracle forward at each step (or the cheaper pinned
variant: full-forward argmax at the final context) and an ids comparison.

Response (the contract the shipped renderer implements):

    { "verdict": "PASS" | "FAIL" | "UNAVAILABLE",
      "argmax_match": bool,            // only when actually computed
      "token_for_token": bool,         // only when actually computed
      "max_abs_logit_diff": float,     // only when actually computed
      "oracle": "numpy fp32",
      "gated_ref": "<log path / sha of the check transcript>",
      "note": string }

Rules (enforce in the C server; Opus gates):
- **Never PASS without running the check.** Any setup failure → UNAVAILABLE + note.
- Single-flight with generation (the GPU/CPU is busy during a check → 409 busy to generate).
- The check writes a transcript (`gated_ref`) so a PASS is auditable after the fact.
- Omitted fields = not computed; the frontend renders ONLY supplied fields (shipped behavior).

## 3. Frontend (SHIPPED this branch)

`renderVerifyOutcome()` renders UNAVAILABLE exactly as before; for PASS/FAIL it shows the verdict
pill + chips for each PRESENT field (argmax ✓/✕, token-for-token ✓/✕, max|Δlogit|, oracle,
gated_ref) and never defaults a missing field to green. Mock/replay still hard-return UNAVAILABLE
client-side.

## 4. /api/health `models[]` capability (the model switcher's contract)

Today: health = `{ok,serve,model,ready,device,busy}` — no `models` field → the switcher never
renders (shipped behavior). A future multi-model server MAY advertise:

    "models": [ {"id":"gpt2-xl","label":"GPT-2-XL · 1.5B","default":true},
                {"id":"gpt2-large","label":"GPT-2-Large · 774M"} ]

Then the frontend shows the switcher and sends an ADDITIVE `"model"` field on `/api/generate`
(never sent unless advertised — today's wire bytes are unchanged). Server-side options (both NEED
GPU BUILD + GATE): (a) one worker per model, server routes (RAM-heavy, simple); (b) worker
restart-on-switch with a `busy:"loading"` health state (slow switch, one GPU). Each served model
MUST have its own green `gpt2_scale.sh`-style parity evidence before appearing in `models[]` —
listing an ungated model is an overclaim.

## 5. Gate additions (Opus, Claude Code)

- G-V1: verify endpoint truthfulness — with the oracle absent, 100 calls → 100 UNAVAILABLE; with
  it present, a deliberately corrupted ids array → FAIL (fail-closed negative control).
- G-V2: verify/generate single-flight (no interleaved GPU use).
- G-V3: models[] honesty — every advertised id maps to committed parity evidence.
