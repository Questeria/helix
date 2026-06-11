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

## 7. Assistant-class chat UX (2026-06, fable/demo-assistant-w12) - integration contracts

Page 1 (demo/index.html) now carries a ChatGPT-class conversation layer: Stop (Esc) with
partial-kept bubbles, Continue-as-assistant-prefill, Regenerate (seed-aware), edit-a-past-message,
conversation BRANCHING (tree of turns + a ChatGPT-style < k/n > pager), saved chats + prompt
library (localStorage only), a hand-written escape-first markdown renderer with per-code-block
copy, a Ctrl/Cmd-K command palette, a generation-settings drawer, per-token confidence shading,
and a light theme. All of it works in mock / replay / live and from file://; no new network
calls are made by the UI layer itself.

The backend lands sampling in parallel. The EXACT capability gates the page honors:

- `/api/health` "sampling": true  -> the page starts sending the ADDITIVE request fields
  {temperature, top_p, top_k, min_p, rep_penalty, freq_penalty, pres_penalty, seed, stop:[...]}
  on POST /api/generate. Until that flag appears, the body stays exactly
  {prompt, n_gen, request_id, model?, detail?} - byte-identical key set to today (verified by
  fetch interception). temperature 0 stays the greedy verified default and the drawer says so.
- `/api/health` "max_ctx": <int> -> lifts the client n_gen clamp (default 256) and powers the
  "unlimited" max-tokens option: n_gen = max_ctx - rough prompt estimate (chars/4, labeled).
  The server's own budget must stay fail-closed; the client treats its estimate as advisory.
- `token` SSE event additive fields {p, H, alts:[[id,piece,p],...]} -> per-token confidence
  shading (5 buckets by p; entropy-only payloads use a labeled 1/(1+H) proxy) + a click/Enter
  popover listing the alternatives, labeled "real data from the live logits". When the fields
  are absent NOTHING renders - no legend, no shading. Never synthesize these client-side.
- Seed semantics the page assumes: a seeded sampled request is reproducible; "regenerate"
  bumps the seed, "regenerate identically" resends the recorded one. If the server echoes the
  effective seed in `done` (recommended, additive), nothing breaks - the page already shows
  the seed it sent.
- CONTINUE sends the ORIGINAL wire prompt + the assistant text so far as the new prompt
  (chat models: the ChatML template still ends inside the open assistant turn - the server
  must NOT re-template or append a new <|im_start|>; base models: plain concatenation).
  Greedy continues are exact-by-construction; sampled continues are labeled
  "continued (new sampling segment)" in the UI.
- Client aborts (Stop/Esc) surface as a closed request body stream - treat as cancel, free
  the single-flight slot. The page keeps and labels the partial text it already received.

Honesty notes for reviewers: the "$0 / watt-seconds" meter was SKIPPED - the done event
carries only n_gen/seconds/tok_per_s and the browser cannot measure GPU power, so any
energy figure would be invented. Saved chats/prompts/settings are localStorage-only and the
sidebar says so. Replay turns never offer continue/regenerate/edit (replaying the capture is
the only honest re-run). /api/verify semantics are untouched: no live PASS is ever rendered
client-side.

## 8. Deep-think + glass-box viewers (2026-06, fable/demo-assistant-w12) - integration contracts

Page 1 now carries HONEST deep-think modes and glass-box viewers. Everything orchestrates REAL
/api/generate calls or consumes REAL additive SSE fields; nothing is simulated anywhere.

Backend contract consumed (additive on the existing "token" event; absent = render nothing):

- every token event: {p, H, alts:[[id,piece,prob] x5]} - real host-side stats of the live logits
  (already consumed by W1 confidence shading; W12 reuses, never duplicates).
- request field "lens":1 (sent ONLY when the user enables Lens mode AND live AND glass box):
  DECODE-step token events additionally carry "lens":[[ [id,p] x5 ] per layer] (the logit lens:
  the residual decoded through the real final-norm+head after every layer) and
  "attn":[[ [pos,weight] x8 ] per head, last layer]. Prefill (first token) has no lens fields -
  the viewer says so. Lens parity has its own gate leg (G-LENS). Lens is SLOWER and labeled so.
- "stop":[...] now actually stops generation server-side (the page already sent it).

Features and their honesty contracts:

- DEEP-THINK selector (#dtMode, next to the model switcher; live-only, disabled elsewhere with
  an honest tooltip): normal / best-of-3 / best-of-5 / self-consistency x3 / x5 / critique&revise
  / tree-of-thought. Each mode runs REAL sequential calls (single-flight GPU; visible "sample k
  of N" progress). 409-busy during orchestration = the SAME backoff+jitter retry discipline as
  the single-turn busy-wait (queue of one, no invented position numbers). All candidates + seeds
  + scores are recorded on the reply node (meta.deepthink) and rendered in a collapsible panel;
  the visible think-budget line sums n_gen + seconds across ALL calls. Sampled modes need
  /api/health "sampling":true (refused honestly otherwise); a user temperature of 0 is bumped to
  0.8 for candidate sampling and DISCLOSED in the panel. Deep-think forces glass box (visible
  toggle switch) because the scorer needs the per-token p fields. Scorer: mean of ln(p) over the
  reply tokens; when p fields are absent the panel says "no scorer available" and shows the
  candidates unscored. Self-consistency votes on the last non-empty line (labeled heuristic,
  small-model framing). Critique&revise = 3 greedy calls (draft/critique/revision, templated
  prompts shown verbatim; labeled "a 360M model's critiques are shallow - this shows the
  technique"). Tree-of-thought = 3 sampled ~24-token stubs, user-or-auto pick, greedy extension;
  rendered as a tree and labeled sequential best-first search, never "parallel beams".
- LENS mode (#lensBtn in the speed-toggle group; live + glass box only): adds "lens":1. Click a
  token -> the existing W1 alternatives popover gains "layer-by-layer view" -> an overlay renders
  rows = layers with top-5 [token,prob] per layer, starring the first layer where the final pick
  becomes top-1, plus per-head attention chips (token text from the run's real context strings,
  intensity = weight). Token text for lens ids comes ONLY from ids actually decoded this session
  (tokenize/token/alts events feed an id->piece map); unknown ids render as raw #id - the browser
  has no tokenizer and never invents text.
- REPRODUCE-THIS-REPLY (per live reply with recorded gen_ids + request): re-sends the EXACT
  {model, wire prompt, params incl. seed, n_gen}, byte-compares returned gen_ids; green
  "reproduced byte-identical (same seed -> same tokens on this stack)" or red mismatch +
  "report this". Labeled: proves REPRODUCIBILITY on this stack; the independent-oracle check is
  the offline gate (link to the dashboard). Continued/stopped replies are multi-segment and
  never offer it.
- PROVENANCE RECEIPT (per live reply): downloads JSON {model, params, seed, wire prompt, gen_ids,
  reply text, pinned trust anchors (seed sha 9837db12..., fixpoint 0992dddd..., kovc self-compile
  698392 B, 299-byte seed)} - "everything needed to re-derive and re-verify this reply".
- MODEL RACE (tools drawer + palette; live with models[] > 1): the same text VERBATIM to every
  served model, strictly sequential (the visible queue IS the single-flight truth), greedy for a
  fair deterministic comparison; per-model tok/s + honest size labels; never added to the convo.
- TAMPER CARD (Why-Helix drawer): quotes scripts/_gate_sampling4.log's NEG-control lines verbatim
  and states explicitly that this is committed OFFLINE gate evidence - no in-browser tamper run
  is claimed.
- TOOLS-LITE (#toolsBtn drawer): calculator (hand-written shunting-yard, no eval()), dates, unit
  conversions, framed "the model cannot call tools - you can"; results insert as visible [tool]
  blocks into the next user message.
- MEMORY PINS: pinned facts ride the ChatML system turn (instruct models only - base models have
  no system prompt and the chip row says so); visible chip row above the composer; pins consume
  context budget like any text. Explicit-systemText calls (deep-think templates, summarizer)
  skip pins by design.
- SUMMARIZE-AND-CONTINUE: near the context budget, "compress older turns" runs a REAL model call
  to summarize them; the carry switches to a visibly-labeled [summary] branch (summary pair + a
  copy of the latest turn) while the ORIGINAL conversation stays reachable via the root branch
  pager. Fail/cancel = nothing changes, said visibly.
- RAG-LITE (tools drawer): 12 excerpts copied VERBATIM from README.md / TRUST_CHAIN_CLOSED.md /
  HELIX_V1_DEFINITION_OF_DONE.md with source paths, BM25-ish keyword scoring in-browser, top-2
  inserted as visible "[retrieved from <path>]" blocks. Labeled: tiny corpus, keyword match,
  shown verbatim - not semantic search.

Invariants kept: SSE event names unchanged (only additive fields consumed); request bodies gain
fields only behind their capability gates (sampling / lens / model / detail); fast mode stays a
subset (deep-think/lens/reproduce force glass box VISIBLY); mock/replay keep working with the
new controls disabled behind honest tooltips (deep-think, lens, reproduce, race are live-only);
replay stays pinned; the model is never said to call tools; no CDNs; node --check green.
