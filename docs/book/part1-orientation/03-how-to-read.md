# How to read this book

*What this chapter covers: the shape of the whole book (Parts I–IX plus appendices) and which parts have shipped versus which are planned; the two audiences this book serves at once and the conventions that let it serve both; and three concrete reading paths — for a human setting Helix up, for an AI agent operating it, and for a contributor extending it.*

This is a long book about a system whose entire reason for existing is **honest, reproducible trust**. You do not need to read it front to back. This chapter is a map: it tells you what is here, what is not here *yet*, how to tell a verified fact from an illustration, and the shortest route from where you are to what you want.

## The book is staged — read the table of contents as the source of truth

The book is being written in stages, and that status is visible in the [table of contents](../SUMMARY.md). **Stage 1 has shipped.** It covers the three parts you most need to get started and to operate Helix safely:

- **Part I — Orientation** — what Helix is, a ten-minute tour, how to read this book (you are here), and trust at a glance.
- **Part II — Setup & Build** — prerequisites, building from raw, using `kovc`, reproducing and verifying the trust chain, and troubleshooting.
- **Part IX — For AI Agents** — driving Helix, the non-negotiables, the traps, and copy-paste recipes.

Parts **III–VIII** and **Appendices A–H** are **planned**. They are *outlined* in the table of contents — with chapter titles each marked *(planned)* — so the shape of the whole book is visible, but their pages are not written yet and will be filled in subsequent stages. The planned parts are:

- **Part III — The Helix Language** (the language tour; types, effects & refinements; functions, control flow & pattern matching; generics, traits & closures; autodiff and the AGI type-system features).
- **Part IV — The Standard Library** (overview; math, transcendentals & activations; tensors, collections & I/O).
- **Part V — The Compiler (`kovc`)** (front end: lexer, parser, typecheck; IR & optimization passes; the x86-64 ELF back end).
- **Part VI — The From-Raw Bootstrap Ladder** (`hex0` and the raw-binary root; the MESCC-lineage rungs to `seed`; `seed` to `kovc`: the self-host fixpoint).
- **Part VII — GPU Codegen** (the PTX back end; GEMM, tiling & the capstone; honest performance & the PTX boundary).
- **Part VIII — Trust & Verification** (the trusting-trust problem & the gcc-DDC; the gate and the feature corpus; residuals & the trusted computing base).
- **Appendices A–H** — glossary, command reference, pinned hashes & anchors, file & directory map, example index, the trusted computing base, roadmap & Phase 2, and further reading.

> **Note:** A chapter title rendered with *(planned)* and no link in [`SUMMARY.md`](../SUMMARY.md) is an outline entry, not a page. When this book references a topic that lives in a planned part, it points you at the **real repo file** that already documents it (for example, [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) for the full trust record) rather than at an unwritten chapter. Nothing in the shipped chapters depends on a planned chapter being present.

> **For AI agents:** treat [`docs/book/SUMMARY.md`](../SUMMARY.md) as the canonical index of what exists. Do not assume a chapter is available because its topic is named here — resolve the link in `SUMMARY.md` first, and if the entry is marked *(planned)* (no target), fall back to the cited repo source. When a book chapter and a repo source ever disagree, the **repo source wins** and the chapter is the bug; flag it, do not silently follow stale prose.

## Two readers at once

Every chapter is written for **two audiences simultaneously**, and the book is structured so neither gets in the other's way:

1. **A human developer** — someone evaluating Helix, building it from raw, learning the language, or studying how a from-scratch trust chain is actually constructed. The human wants narrative, motivation, worked examples, and a clear "why."
2. **An AI operator (agent)** — an LLM-driven agent that will *drive* Helix: invoke the build, run the gate, compile `.hx` programs, and reason about what the trust chain does and does not establish. The agent wants exact commands, exact paths, exact invariants, and explicit do-not-do rules.

The main prose is written for the human. Where an AI operator needs **different or extra** guidance — a non-negotiable invariant, a trap that only bites when you are scripting, an exact string to match — the book breaks out a callout that begins **`For AI agents:`**, like the two you have already seen in this chapter. These callouts are imperative and checkable: an exact command, an exact token, an exact path, or an explicit prohibition. They appear **only when the agent guidance differs** from the human guidance; when the advice is the same for both, it is written once in the main text.

In Parts I–VIII, those `For AI agents:` callouts are **spot guidance**. The deep operator material — the full set of non-negotiables, the catalogue of traps, and the recipes — lives in **[Part IX — For AI Agents](../part9-for-ai-agents/01-driving-helix.md)**, which is the dedicated operator manual. The other parts cross-link to it rather than re-explaining it.

> **For AI agents:** before you take any action against Helix, read **[Part IX — Non-negotiables](../part9-for-ai-agents/02-non-negotiables.md)**. When you act on this book, prefer commands and paths quoted **verbatim** from a chapter over anything you infer.

## Reading conventions — how to trust what you read

Because this is a book about reproducible trust, it holds *itself* to the project's standard. A few conventions, defined in full in the [Style Guide](../STYLE_GUIDE.md), let you read every page with calibrated confidence.

### Verified examples vs fragments

Every code block carries a **bold status label on the line immediately above the fence**:

- **Verified example** — a *complete, compile-checked* program. A Helix "Verified example" is a complete program with an `fn main`, was compiled (and run, where it has a defined exit code) **before the chapter shipped**, and **cites the source path** it was taken from or added to. The observed result (e.g. an exit code) is stated in prose or a following `text` block. The canonical first one is [`helixc/examples/exit42.hx`](../../../helixc/examples/exit42.hx), which compiles to a Linux ELF and exits with status `42`.
- **Fragment** — a partial snippet that illustrates one piece of syntax or a single function. It is **not** expected to compile on its own, and the book never implies it is runnable.

When you see **Verified example**, you may rely on it: it ran. When you see **Fragment**, read it as illustration, not as a program to paste and run.

### Code fences are always tagged

Every fenced block declares its language so both readers — and an agent's parser — can tell code from output:

- ` ```helix ` for Helix (`.hx`) source.
- ` ```bash ` for shell commands, build invocations, and excerpts from `scripts/*.sh`.
- ` ```text ` for program output, hashes, REPL/console transcripts, and diagrams.

### Commands are quoted verbatim; paths are real and cited

Build and verify commands are copied **exactly** from the real scripts — never paraphrased. The source of truth, in order, is [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh), [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), the per-rung `stage0/<rung>/build.sh`, and the CI workflow [`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml). The one-command reproduction, for instance, is exactly:

```bash
bash scripts/reproduce_trust.sh
```

Every factual claim links to the **real repo file** that backs it, by its repo-root-relative path, so you (or an agent) can verify it directly. The book does not cite a file its authors have not opened.

### A small, fixed vocabulary

The book uses a deliberately small set of terms with fixed meanings — `kovc` (the Helix compiler, written in Helix), `seed` (the Apache-2.0 C-subset compiler that builds `kovc`), **the from-raw ladder**, **the self-host fixpoint**, **gcc-DDC**, **the gate**, and **the capstone**. They are defined where they first appear and collected in the (planned) glossary; the full table lives in the [Style Guide](../STYLE_GUIDE.md). These terms map to real files and real output tokens, so they are used consistently and never given synonyms.

> **For AI agents:** key your logic off the **exact strings** the book and scripts use, not off English descriptions. The gate prints the literal token `GATE_PASS` on success (`scripts/gate_kovc.sh`); the pinned anchors are ground truth — `seed = 9837db12…`, the self-host fixpoint `K2 == K3 == K4 = 0992dddd…`, and the gcc-DDC `K1 = 84363adb…`. Match those tokens and hashes, not paraphrases of them.

### Honesty and residuals

The book states limits where they matter and cites them. In particular, you will see plainly — and cross-referenced to [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R — that the chain is **complete to PTX, not to GPU machine code**; that GPU performance is a **fraction** of cuBLAS (~50–67.5% on the reference RTX 3070 Laptop, sm_86), not parity; that the end-to-end capstone speedup is **7.0–8.7×** (Amdahl-bound), not ≥10×; that the validated target is a **single GPU** (sm_86); and that **external third-party reproduction on independent hardware remains the one open increment**. A `> **Residual:**` callout marks an honest limitation in place. Where a number appears, it is paired with its honest fraction. The book does not overclaim.

> **Residual:** the byte-identical, hash-pinned diverse-double-compile covers the **`seed→K1`** surface; the broader language surface is cross-checked **behaviorally**, and that witness is out-of-tree, not clean-checkout reproducible. See [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R for the complete list.

## Recommended reading paths

You can read straight through Part I to orient, but most readers arrive with a goal. Pick the path that matches yours.

### Path A — A human setting Helix up

You want to evaluate Helix, build it on your own machine, and convince yourself the trust claims hold.

1. **[The ten-minute tour](02-ten-minute-tour.md)** — see Helix compile and run something end to end, fast.
2. **[Trust at a glance](04-trust-at-a-glance.md)** — the one-page version of what "trust chain closed" means and what it does not.
3. **[Part II — Prerequisites](../part2-setup-build/01-prerequisites.md)** — what your environment needs.
4. **[Part II — Build from raw](../part2-setup-build/02-build-from-raw.md)** — climb the from-raw ladder yourself.
5. **[Part II — Using `kovc`](../part2-setup-build/03-using-kovc.md)** — compile your own `.hx` programs.
6. **[Part II — Reproduce & verify the trust chain](../part2-setup-build/04-reproduce-verify-trust.md)** — run the one-command reproduction (`bash scripts/reproduce_trust.sh`) on a clean checkout and confirm every pinned hash. It also runs unattended on a fresh GitHub runner via [`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml).
7. If something breaks, **[Part II — Troubleshooting](../part2-setup-build/05-troubleshooting.md)**.

When you want to go deeper into *how* the language, compiler, ladder, GPU back end, or verification actually work, those are the **planned** Parts III–VIII; until they ship, the cited repo docs (starting with [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) and [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)) are the authoritative source.

### Path B — An AI agent operating Helix

You are an LLM-driven agent that must drive the build, run the gate, compile programs, and reason correctly about the trust chain. Read the operator manual first, then keep it open.

1. **[Part IX — Driving Helix](../part9-for-ai-agents/01-driving-helix.md)** — how to operate the toolchain.
2. **[Part IX — Non-negotiables](../part9-for-ai-agents/02-non-negotiables.md)** — the invariants you must never violate. Read this **before acting**.
3. **[Part IX — Traps](../part9-for-ai-agents/03-traps.md)** — the failure modes that bite agents specifically.
4. **[Part IX — Recipes](../part9-for-ai-agents/04-recipes.md)** — copy-paste sequences for common operator tasks.

Use Part I and Part II as reference when you need the human-readable "why" behind a command, and obey every `For AI agents:` callout you encounter elsewhere in the book.

> **For AI agents:** your contract when using this book is simple. Resolve links in [`SUMMARY.md`](../SUMMARY.md) before assuming a chapter exists; prefer verbatim commands and paths over inference; key off exact tokens (`GATE_PASS`) and pinned hashes (`9837db12…`, `0992dddd…`, `84363adb…`); and if a chapter contradicts a repo source, follow the repo source and flag the chapter.

### Path C — A contributor extending Helix (or this book)

You want to add to the compiler, the standard library, the GPU back end — or to write the next chapters of this book.

1. Start with **Path A** so you can build and reproduce the chain yourself; you cannot extend a trust chain you have not verified.
2. Read the deep, *current* engineering record in the repo: [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) (the verified state and every residual), [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) (rebuild from a clean checkout), and the canonical scripts [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) and [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh). Until Parts III–VIII ship, these repo files are the authoritative documentation of the internals those parts will cover.
3. Before any change, internalize **[Part IX — Non-negotiables](../part9-for-ai-agents/02-non-negotiables.md)**: a contribution that breaks the gate, weakens a residual disclosure, or moves a pinned hash without justification is a regression, no matter how good it looks.
4. **If you are authoring or amending a chapter,** the [Style Guide](../STYLE_GUIDE.md) is binding. In short: serve both audiences; tag every fence; label every block **Verified example** or **Fragment**; actually compile (and run) every Verified example and cite its source path before it ships; quote commands verbatim from the real scripts; cite real repo paths you have opened; and never exceed [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R when stating what Helix can do. Hallucination is the one unforgivable error in this book.

> **For AI agents:** if asked to add or update an example, do **not** assert it works until you have actually compiled and run it via the real toolchain. An unverifiable claim must be **removed**, not hedged. Treat [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R (residuals) as the ceiling on what the book may claim — never exceed it.

---

**Next:** **[Trust at a glance](04-trust-at-a-glance.md)** — the one-page summary of what Helix's closed trust chain establishes, the pinned anchors that prove it, and the residuals it honestly does not cover.
