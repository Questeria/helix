# Hosting the demo on a public website — architecture + deploy plan (DESIGN)

**Frontend prerequisites: ALL SHIPPED** (branches `fable/ui-finish` + `fable/features-and-longshots`):
dual-mode (live → labeled captured-run REPLAY fallback), `?source=` overrides, the embeddable
widget (`demo/embed.html`), the shareable replay permalink, the "Switch to LIVE" upgrade notice,
busy-409 / error / reconnect states. **Backend/queue items below NEED GPU BUILD + GATE in Claude
Code. No secrets appear anywhere in this plan.**

## 1. Two honest deployment tiers

**Tier 1 — static site, zero GPU (deployable TODAY).** Host `demo/{index.html, dashboard.html,
captured_run.js, embed.html, onepager.html}` on any static host (GitHub Pages / object storage —
they are plain files, no build step, no CDN deps). Every visitor gets the REPLAY of the real gated
run, labeled "captured run (replay) — not live", with page 2 carrying the proof + the
prove-it-yourself command. The upgrade watch is a no-op (no /api/health) — the page never hints at
liveness. **This tier makes NO live claim and needs no ops.**

**Tier 2 — live window.** The same static files served by `gpt2_serve_http` on the GPU box,
exposed through a reverse tunnel (e.g. a TLS-terminating proxy/tunnel to 127.0.0.1:8848 — the
server itself stays loopback-bound, its current design). When the operator is live, visitors who
open `?source=sse` (or click "Switch to LIVE" once health reports ready) get the real stream; when
the box goes away, the page falls back to REPLAY automatically (shipped). **Operational honesty
rule: the public landing always defaults to replay; LIVE is an explicit, health-gated upgrade.**

## 2. The queue / "your turn" UX (single GPU, strictly serial — DESIGN, needs server work + gate)

Today a second concurrent generation gets an honest 409 + retry UX (shipped). For a public live
window that becomes a queue:

- Contract (additive endpoints; Opus builds + gates):
  `POST /api/queue` → `{ticket, position, eta_s}` (eta = position × measured ~196 s/20-tok run —
  the real number, not marketing); `GET /api/queue/<ticket>` → `{position, state: waiting|ready|
  expired}`; ready tickets get a 90 s claim window to fire `/api/generate` (server validates the
  ticket, else 409).
- Frontend UX (build when the contract lands): composer shows "you're #N in line · ETA ~M min
  (measured, ~10 s/token)"; the internals panels keep replaying the captured run while waiting,
  labeled as such; a turn notification swaps the banner to "your turn — 90 s to run your prompt".
- Honesty: ETAs only from measured cadence; the queue page never shows another user's prompt or
  output (single-flight isolation, no shared transcript); queue depth capped (e.g. 20) with an
  honest "line's full — watch the replay" state.
- Abuse/ops knobs (FLAGGED FOR OPUS — rate limits, max n_gen public default 20, ticket signing):
  security review is explicitly Opus's lane, not specified here.

## 3. Public landing

`index.html` already is the landing (hero pitch ≤5 s, eyebrow, fact chips, auto-replay). Optional
extras: `embed.html` for third-party sites; `onepager.html` (print → PDF) for the deck leave-behind.
Suggested information architecture: landing(index, replay-by-default) → proof(dashboard) →
prove-it-yourself(one command) → repo.

## 4. Deploy checklist (Tier 1)

1. Copy the five demo files to the host (no transforms; they must remain byte-identical to the
   gated branch).
2. Verify `file://`-grade behavior over HTTP: REPLAY badge shows, banner labeled, autoplay runs,
   `?source=mock` works, no external requests in devtools network tab (only same-origin files).
3. Set cache headers freely (static, no API).
4. Do NOT enable any "live" wording anywhere — the pages do this correctly by themselves; never
   edit copy to imply liveness.

## 5. Out of scope here, reserved for Opus in Claude Code

Anything that touches `gpt2_serve_http.c` (queue endpoints, models[], verify-real mode, the open
SO_SNDTIMEO + heartbeat items), TLS/tunnel choices as security decisions, rate limiting policy,
and all gating (`helix_serve_gate.sh` extensions G-Q1 queue truth / G-Q2 single-flight under load).
