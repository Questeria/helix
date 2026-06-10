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

## 6. Explain mode on the chat page (SHIPPED 2026-06, demo/index.html only)

A beginner layer over the live telemetry — same SSE events, no new event types, no wire changes.

- **Toggle**: Explain / Expert segmented control at the top of the internals column. Default
  Explain for first-time visitors; `localStorage.helix_explain_mode` ("1"/"0") remembers the
  choice. Expert view = the pre-existing panels (trust strip, ladder, kernel ticker, gauges,
  progress, summary), markup/behavior untouched — Explain just switches which set is visible
  via an `.explain-on` class on `#internalsCol`.
- **Structural map** (`#explainPanel`): [token embedding] → [layer K of N: attention | MLP] →
  [final norm] → [next-token head] → [next token]. Event mapping: `embed`→embedding node lit;
  `layer_begin`→K/N (N strictly from the event's `total`/`forward_begin.n_layers`, never
  hardcoded); `op.phase` (attn/mlp) lights the matching half; `layer_end`→within-stack progress
  bar; `head` label `ln_f`→final-norm node, anything else→head node (defensive); `token`→output
  node pulses with the real piece+id; `forward_begin` resets the walk per token step.
- **Plain-language meanings**: `LABEL_MEANING` (by op label, most specific) falling back to
  `KERNEL_MEANING` (covers the 8 GPT-2 kovc kernels + rmsnorm/rope/silu llama-arch names), with
  an honest generic fallback for unknown ops. Shown as the caption under the current-op line in
  Explain mode, and as `title=` tooltips on every kernel ticker line, legend chip, and the
  ladder op readout in Expert view (hover-only; no visual change).
- **Why-this-is-special thread**: rotating one-liners, source-aware. The "real GPU execution"
  claim renders ONLY when `TELEMETRY_SOURCE === "sse"`; replay says "replay of a real captured
  run"; mock says "simulated — structure real, numbers mock". The explain panel carries its own
  LIVE/REPLAY/MOCK chip mirroring the top badge, updated on every source switch (including the
  /api/health fallback path).
- **Onboarding**: first-visit 3-sentence overlay (`localStorage.helix_onboarding_seen`),
  dismissible by button/backdrop/Escape; honest in all modes (it points at the badge rather
  than claiming live).
- All consumers are defensive about missing/extra event fields; works identically in mock,
  replay, and live (verified in-browser on mock + replay + the sse→replay fallback; inline
  script passes `node --check`).

## 7. Source <-> execution link + Why-Helix drawer + speed-mode toggle (SHIPPED 2026-06, demo/index.html only)

- **Source link**: every kernel name in the telemetry (op-ticker lines, legend chips, the
  ladder op readout, the explain-mode current-op line) carries `.klink` + `data-kn` and opens
  a modal (`#ksrcOverlay`) showing the kernel's REAL Helix source, embedded VERBATIM in the
  inline script (`KERNEL_SOURCES`, transcribed from `helixc/examples/*.hx`: the 8 GPT-2 serve
  kernels, the 3 Llama-arch kernels, and the 3 fast-decode KV kernels). The framing is
  source-aware ("the kernel that just ran" only on live AND a fired op — `data-ran`); a gate
  status chip is shown per kernel, and the KV trio is labeled "parity gate IN PROGRESS"
  (amber) — nothing is claimed for them until their gate is green. Unknown kernel names
  degrade to an honest "source not bundled" message. Comment/keyword tinting is DOM-built
  (textContent only) — the source text is never parsed as HTML.
- **Why-Helix drawer** (`#whyOverlay`, header button "Why Helix"): six capability cards, each
  = one claim + a real code snippet + "what this enables" + the repo path(s) that back it.
  Every claim was verified against the repo before inclusion; two candidates were dropped as
  not backed by the live kovc (standalone @pure effect-system enforcement — only the
  reverse-AD @checkpoint purity scan in helixc/bootstrap/parser.hx is enforced today — and
  Presburger shape checking, which lived in the retired Python frontend), and the drawer's
  footer states the drops explicitly.
- **Speed mode** (`#speedToggle`, next to the model switcher): `glass box` = today's per-op
  behavior, unchanged (default). `fast` sends an ADDITIVE `detail:"token"` field on
  /api/generate (same additive pattern as `"model"`, mirrored in the query string) and
  consumes a SUBSET of the same SSE events — hello/tokenize/forward_begin/token/done; no
  event renames. The fast option is enabled ONLY when /api/health advertises the capability
  (`"fast": true`, or a `models[]` entry carrying `fast: true`); otherwise it stays visible
  but disabled with "needs the fast-decode backend". Honest labels: glass box = "every kernel
  shown and synchronized — slower, fully observable"; fast = "same verified kernels, no
  per-op instrumentation — the speed is from caching and not approximation". In fast mode the
  per-op surfaces (kernel ticker, ladder readout, explain caption) say "per-op detail is off
  in fast mode — switch to glass box to watch every kernel"; token progress, the tokenizer
  panel and the transcript stream unchanged. No tokens/sec figure is invented anywhere.
- All three features work in mock, replay and live, and from file:// (no new network calls);
  the inline JS passes `node --check`.
