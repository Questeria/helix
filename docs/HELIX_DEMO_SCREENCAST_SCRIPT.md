# Screencast script — "GPT-2 on a stack you can check from the first byte" (~4.5 min)

Record at 1920×1080, dark room projector profile, system font UI only. Two windows: a terminal
(font ≥ 18 pt) and a browser on the demo pages. NOTHING in this script shows a number that isn't
from a real run; if a take goes red, KEEP IT (fail-closed is the pitch) and say so.

Page map (all under `demo/`, self-contained, file://-safe):
`landing.html` (start here) → `index.html` (chat) → `dashboard.html` (proof) →
`compare.html` (models & paths) → `onepager.html` (leave-behind).

0:00–0:20 — COLD OPEN, terminal.
  Type: `git clone https://github.com/Questeria/helix && cd helix`
  VO: "Every AI demo asks you to trust a stack you've never seen. This one doesn't. This repo's
  compiler rebuilds from 299 bytes you could type by hand."

0:20–0:55 — THE ONE COMMAND.
  Run: `bash scripts/reproduce_trust.sh` (pre-warmed clone so it finishes in ~1 min; let the rungs
  scroll). Freeze on `REPRODUCE_TRUST: PASS`.
  VO: "hex0 to seed to kovc, rebuilt from raw, byte-identical anchors, and an unrelated compiler —
  gcc — reproduces a rung byte-for-byte. That's the trusting-trust defense, live."

0:55–1:10 — THE LANDING, browser.
  Open `landing.html`. Hold on the headline + the four fact chips; hover the four destination cards.
  VO: "The model you know, on a stack you can verify from the first byte. Two honest modes — live
  when the GPU box is up, a clearly-labeled captured replay always — and the page always tells you
  which one you're in."

1:10–2:00 — THE MODEL, chat page (replay or live).
  Open `index.html`. If the GPU box is up: `?source=sse`, fire "The capital of France is",
  narrate the 48-layer ladder + kernel ticker while it generates (~10 s/token — SAY IT):
  VO: "Real GPT-2-XL, 1.5 billion parameters, every kernel emitted by that compiler. It's
  intentionally slow — about ten seconds a token — because the product is trust, not speed."
  If no GPU: let the REPLAY auto-run and SAY: "this is a captured run from the gated server —
  the page says so; nothing here pretends to be live."

2:00–2:30 — MULTI-TURN, same page (live or mock — say which).
  Send a SECOND prompt with "carry context" on. Point at: the ⟨…carried N chars⟩ marker in the
  model bubble, and the context meter under the composer ("context ≈ N/320 tok used … oldest text
  is cut first when over").
  VO: "Multi-turn, honestly: a base completion model has no memory, so the page re-sends the
  conversation as one prompt — and meters it against the server's real 320-token cap. When the
  budget overflows, the OLDEST text is cut and the page says so. No hidden system prompt, no
  pretend assistant."
  If a take shows the ✂ cut notice, KEEP IT — that's the feature.

2:30–3:00 — VERIFY UX, same page.
  On the finished turn, click "verify this continuation" → it returns UNAVAILABLE (by design).
  Then click "what can be verified?" and scroll the explainer.
  VO: "Watch what happens when I ask it to verify: UNAVAILABLE — by design. The real parity check
  re-runs a multi-gigabyte independent oracle offline, in the gated pipeline — not inside a chat
  request. What you CAN check live: these ids came from a gated server, and greedy decoding is
  deterministic — re-run the prompt, same ids, byte-identical. A demo that hands out green
  checkmarks it didn't earn isn't a trust demo."

3:00–3:35 — THE PROOF, dashboard.
  Scroll: trust chain ribbon → gates → side-by-side table.
  VO: "Token-for-token against an independent oracle — 25 of 25 — at 124M, 774M, and 1.5B. And a
  pure-CPU path with no GPU vendor boundary at all: two minutes per token, zero trusted arithmetic
  above the seed."

3:35–4:00 — MODELS & PATHS, compare page.
  Open `compare.html`. Hold on the five cards, then the table — point at the SmolLM2 row's
  "not yet measured" cells.
  VO: "Pick your model, pick your trust boundary. And look at the newest row: a modern
  Llama-architecture model, SmolLM2 — its gates are running right now, so its cells say
  'not yet measured'. On this site, that's what honesty looks like while the gates run."

  [PLACEHOLDER — SmolLM2 modern-model beat. RECORD ONLY AFTER ITS GATES ARE GREEN.]
  When G-L1/G-L2 (full-model parity) and the serve gate are green and committed:
  - re-record this beat showing the compare row flipped to real measured numbers;
  - if the chat backend advertises it in /api/health models[], show the model switcher and a
    short SmolLM2 generation (state its real measured pace — never reuse XL's number);
  - VO sketch: "Same 299 bytes, same gates — now running the architecture family behind today's
    open models. Every number you just saw appeared only after its gate went green."
  Until then, DO NOT show SmolLM2 output anywhere in the cut.

4:00–4:20 — THE HONEST EDGES.
  Show the residuals card (dashboard) or the landing's "what this is not" card. Read ONE aloud,
  verbatim (e.g. complete-to-PTX-not-SASS).
  VO: "We state the edges unprompted. Trust you can't interrogate isn't trust."

4:20–4:40 — CLOSE.
  Back to the attestation / DEMO_ATTEST_PASS line (or the captured transcript), then the landing's
  clone one-liner.
  VO: "The model you know, on a stack you can verify from the first byte. That's Helix — the
  verifiable execution layer. Clone it tonight and check us."

Retake rules:
- never speed up footage of generation (or disclose "footage time-compressed" on screen if you
  must cut); never crop a FAIL out of a take; the replay page must always be visibly labeled
  in-frame;
- the context meter is an ESTIMATE — if you mention it, call it an estimate (the server's
  tokenizer is the ground truth);
- the verify beat MUST show the UNAVAILABLE result as-is — never cut before it lands, never
  overlay anything green on it;
- the SmolLM2 beat stays a placeholder until its gates are green and committed — do not record
  or imply SmolLM2 results before that.
