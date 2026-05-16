# Helix Website

This directory holds the public website for the Helix programming language.

## Source of truth

**`HELIX_REFERENCE.md`** is a website design/reference draft, not the source of truth for current shipped capability. Check root `README.md`, `docs/stage35-progress-2026-05-15.md`, and `stats_and_facts.md` before publishing copy. Use the reference draft for:

- All marketing copy
- Feature pages and explainers
- Code samples (30 ready-to-use snippets in `code_samples.md`)
- Roadmap and stage descriptions
- Comparison tables
- Visual identity direction

## What to build

A modern marketing + documentation website for Helix. Suggested pages (full details in HELIX_REFERENCE.md):

- `/` — Landing with hero animation
- `/why` — The pitch
- `/learn` — 10-lesson interactive tutorial
- `/playground` — Embedded Helix editor with live compilation
- `/features` — Feature grid with interactive demos
- `/bootstrap-chain` — Animated explorer of the hex0 → kovc chain
- `/grad` — Autodiff playground
- `/tiles` — Tile/matmul visualization
- `/reflection` — Quote/Splice/modify demo
- `/spec` — Language reference
- `/roadmap` — Stage tracker (39 stages)
- `/audits` — Public audit findings
- `/compare` — Helix vs Rust/Mojo/Triton/Python
- `/contribute` — Get involved
- `/blog` — Engineering posts

## Backend integration

The website should be built with a typed API client so backend integration is plug-and-play later. Recommended:

```ts
// src/api/CompileApi.ts
export interface CompileResult {
  tokens: Token[];
  ast: AstNode;
  ir: IrOp[];
  bytes: number[];
  exitCode: number;
  stdout: string;
  durationMs: number;
}

// Stub mode (use during MVP):
//   export const compile = async (req) => MOCK_RESULTS[req.example] ?? error;
// Live mode (when backend is ready):
//   export const compile = async (req) => fetch('/api/compile', {body: ...})
```

The user will wire in real backends later. Build against typed stubs first.

## Tech stack suggestions (Claude Design's choice)

- **Framework**: Next.js (App Router) or Astro
- **Styling**: Tailwind CSS or vanilla-extract
- **Editor**: Monaco for the playground
- **Animations**: Framer Motion or GSAP for the bootstrap-chain explorer
- **Math**: KaTeX for autodiff page
- **Hex display**: custom component (the website's signature visual)

## Aesthetic axes (pick one or remix)

1. **"From raw metal"** — terminal-black, hex-green accent, monospace everywhere
2. **"Scientific notebook"** — warm off-white, serif body, KaTeX, hand-drawn diagrams
3. **"Futuristic minimalism"** — pure whites or near-blacks, single accent, generous whitespace

Recommendation: try #3 for broadest appeal, keep hex-byte motif from #1 as a recurring ornament.

## Distinctive must-haves

1. **Byte counter** at the top: "Built from 299 bytes" — animated.
2. **Bootstrap chain** as an interactive explorer.
3. **Compilation animation** as a recurring motif.
4. **Hex byte viewer** for produced binaries.
5. **Math notation** rendered properly via KaTeX.
6. **Trap-id callouts** for the silent-corruption findings.

## Example assets to fetch

- Logo: simple `λ` in hex bracket `[λ]` or double-helix + hex trail
- Code samples: 30 in `code_samples.md`; use `HELIX_REFERENCE.md` as a partial/design-reference gallery
- Bootstrap chain stages: use the current 299-byte hex0 root, then mark later byte counts as target/approximate where they are not live artifacts.
- Number-stat block: use `stats_and_facts.md`; do not hardcode stale test totals or shipped self-hosting claims.
