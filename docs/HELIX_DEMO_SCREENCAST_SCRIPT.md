# Screencast script — "GPT-2 on a stack you can check from the first byte" (~3.5 min)

Record at 1920×1080, dark room projector profile, system font UI only. Two windows: a terminal
(font ≥ 18 pt) and the chat page. NOTHING in this script shows a number that isn't from a real
run; if a take goes red, KEEP IT (fail-closed is the pitch) and say so.

0:00–0:20 — COLD OPEN, terminal.
  Type: `git clone https://github.com/Questeria/helix && cd helix`
  VO: "Every AI demo asks you to trust a stack you've never seen. This one doesn't. This repo's
  compiler rebuilds from 299 bytes you could type by hand."

0:20–1:00 — THE ONE COMMAND.
  Run: `bash scripts/reproduce_trust.sh` (pre-warmed clone so it finishes in ~1 min; let the rungs
  scroll). Freeze on `REPRODUCE_TRUST: PASS`.
  VO: "hex0 to seed to kovc, rebuilt from raw, byte-identical anchors, and an unrelated compiler —
  gcc — reproduces a rung byte-for-byte. That's the trusting-trust defense, live."

1:00–1:40 — THE MODEL, page 1 (replay or live).
  Open the chat page. If the GPU box is up: `?source=sse`, fire "The capital of France is",
  narrate the 48-layer ladder + kernel ticker while it generates (~10 s/token — SAY IT):
  VO: "Real GPT-2-XL, 1.5 billion parameters, every kernel emitted by that compiler. It's
  intentionally slow — about ten seconds a token — because the product is trust, not speed."
  If no GPU: let the REPLAY auto-run and SAY: "this is a captured run from the gated server —
  the page says so; nothing here pretends to be live."

1:40–2:20 — THE PROOF, page 2.
  Scroll: trust chain ribbon → gates → side-by-side table.
  VO: "Token-for-token against an independent oracle — 25 of 25 — at 124M, 774M, and 1.5B. And a
  pure-CPU path with no GPU vendor boundary at all: two minutes per token, zero trusted arithmetic
  above the seed."

2:20–2:50 — THE HONEST EDGES.
  Show the residuals card. Read ONE aloud, verbatim (e.g. complete-to-PTX-not-SASS).
  VO: "We state the edges unprompted. Trust you can't interrogate isn't trust."

2:50–3:20 — CLOSE.
  Back to the attestation / DEMO_ATTEST_PASS line (or the captured transcript).
  VO: "The model you know, on a stack you can verify from the first byte. That's Helix — the
  verifiable execution layer. Clone it tonight and check us."

Retake rules: never speed up footage of generation (or disclose "footage time-compressed" on
screen if you must cut); never crop a FAIL out of a take; the replay page must always be visibly
labeled in-frame.
