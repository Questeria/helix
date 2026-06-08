# Helix: The Complete Guide

Welcome. This is the book-length guide to **Helix** — a programming language and compiler that is
built, from the very first byte, to be *trustworthy*: a from-raw-binary, self-hosting compiler
with GPU code generation whose trust chain has been closed and independently reproduced at the
[`v1.3-release`](../TRUST_CHAIN_CLOSED.md) tag.

Helix begins at a 299-byte, hand-authored ELF you can audit one byte at a time
([`stage0/hex0/`](../../stage0/hex0/)) and climbs — with **no trusted pre-built compiler anywhere
in the chain** — through a ladder of small rungs to `seed` (an Apache-2.0 C-subset compiler) and
then to **`kovc`**, the Helix compiler *written in Helix*. Along the way it proves it reproduces
itself byte-for-byte (the self-host fixpoint), defends against a trusting-trust attack (a `gcc`
diverse-double-compile), trains a real neural network on its own GPU kernels (the capstone), and
keeps **exactly one** committed `.py` in the repo. For the precise, fully-scoped claims and every
honest residual, the canonical records are
[`docs/TRUST_CHAIN_CLOSED.md`](../TRUST_CHAIN_CLOSED.md) and
[`docs/CLEAN_REPRODUCTION.md`](../CLEAN_REPRODUCTION.md).

## Who this book is for

This guide is written for **two readers at once**:

- **Human developers** — evaluating Helix, building it from raw, learning the language, or
  studying how a from-scratch trust chain is actually constructed.
- **AI operators (agents)** — LLM-driven agents that will *drive* Helix: run the build, run the
  gate, compile `.hx` programs, and reason about what the trust chain does and does not establish.

Where the guidance for an AI operator differs from the guidance for a human, you will see a
callout that begins **"For AI agents: …"**. **Part IX — For AI Agents** is the dedicated operator
manual, with the non-negotiables, the traps, and copy-paste recipes.

## Status: a staged book

The book is being written in stages. **Stage 1 has shipped**, covering the three parts you most
need to get started and to operate Helix safely:

- **Part I — Orientation** — what Helix is, a ten-minute tour, how to read this book, and trust at
  a glance.
- **Part II — Setup & Build** — prerequisites, building from raw, using `kovc`, reproducing and
  verifying the trust chain, and troubleshooting.
- **Part IX — For AI Agents** — driving Helix, the non-negotiables, the traps, and recipes.

Parts **III–VIII** (the language, the standard library, the `kovc` compiler internals, the
from-raw bootstrap ladder, GPU codegen, and trust & verification) and **Appendices A–H** are
**planned** and marked as such in the [table of contents](SUMMARY.md). They are outlined now so
the shape of the whole book is visible; their chapters will be filled in subsequent stages.

## How to start

- **Just want to see Helix work?** Read **[The ten-minute tour](part1-orientation/02-ten-minute-tour.md)**.
- **Driving Helix as an AI agent?** Start with the operator manual:
  **[Part IX — Driving Helix](part9-for-ai-agents/01-driving-helix.md)** and its
  **[Non-negotiables](part9-for-ai-agents/02-non-negotiables.md)**.
- **Want to build and verify it yourself?** Go to **[Part II — Setup & Build](part2-setup-build/01-prerequisites.md)**,
  then **[Reproduce & verify the trust chain](part2-setup-build/04-reproduce-verify-trust.md)**.
  The one-command reproduction is `bash scripts/reproduce_trust.sh`, run on a clean checkout (it
  also runs on a fresh GitHub runner via
  [`.github/workflows/trust-reproduce.yml`](../../.github/workflows/trust-reproduce.yml)).

> **For AI agents:** before acting, read **[Part IX — Non-negotiables](part9-for-ai-agents/02-non-negotiables.md)**.
> Key the gate result off the literal token `GATE_PASS`, and treat the pinned anchors as
> ground truth: `seed = 9837db12…`, self-host fixpoint `K2==K3==K4 = 0992dddd…`, gcc-DDC
> `K1 = 84363adb…`. If a chapter and a repo source ever disagree, the repo source wins.

## Every example is grounded and verified

This book holds itself to the same standard as the project it documents:

- **Every claim is grounded in real source.** Commands are quoted **verbatim** from the real
  scripts ([`scripts/reproduce_trust.sh`](../../scripts/reproduce_trust.sh),
  [`scripts/gate_kovc.sh`](../../scripts/gate_kovc.sh), the per-rung `stage0/<rung>/build.sh`,
  and the CI workflow). Paths link to real files relative to the repo root. Nothing is invented.
- **Every "Verified example" is compile-checked.** A Helix program labelled *Verified example* is
  a complete program (with an `fn main`) that was compiled — and run, where it has a defined exit
  code — before the chapter shipped, and it cites the source path it came from. Partial snippets
  are clearly marked **Fragment** and are not claimed to run on their own.
- **Residuals are stated, not hidden.** Helix is **complete to PTX, not to GPU machine code**; its
  GPU performance is a *fraction* of cuBLAS (~50–67.5% on the reference RTX 3070 Laptop, sm_86),
  not parity; the end-to-end capstone speedup is **7.0–8.7×** (Amdahl-bound), not ≥10×; it targets
  a single GPU (sm_86); and external third-party reproduction on independent hardware remains the
  one open increment. Where a limit matters, the book states it and cites
  [`docs/TRUST_CHAIN_CLOSED.md`](../TRUST_CHAIN_CLOSED.md). No overclaim.

Authors: see the **[Style Guide](STYLE_GUIDE.md)** for the conventions every chapter must follow.

## License

Code is Apache 2.0 (see [`LICENSE`](../../LICENSE)). The documentation, including this book, is
under CC-BY 4.0 (stated project policy).
