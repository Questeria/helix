# Kovostov-Native — Plan

**Status**: Phase 0 (bootstrap chain). Created 2026-05-03.
**This document is living.** Updated as the work teaches us what's wrong with it. See `decisions/` for dated rationales.

---

## Hard constraints (non-negotiable)

1. **Raw binary start.** Bootstrap chain begins with hand-encoded hex bytes for the seed. No use of `as`, `gcc`, `clang`, `nasm`, `rustc`, `cargo`, `LLVM`, or `MLIR` to produce shipped artifacts. Audit-only use of `objdump`/`xxd`/`cmp` is permitted (the same way a paranoid mathematician uses a calculator to *check* a hand calculation).
2. **Open source, end-to-end.** Apache 2.0 for code. CC-BY 4.0 for docs. CC0 for trained model weights. All training data is public.
3. **No proprietary AI APIs in training or runtime.** No Claude / GPT / Gemini outputs in training data. Public corpora only.
4. **Consumer hardware.** RTX 3070 laptop now, RTX 5090 + Ryzen 9 9950X soon. Optional cloud bursts.
5. **End goal: AGI.** Open-source AGI is the north star. Honest expectation: years of work, no guarantee of success. Measurable progress on AGI-properties (sample efficiency, continual learning, compositional generalization, transfer, self-improvement) is the operational metric.

## Vision

Kovostov is an artificial general intelligence built layer-by-layer from raw binary, with every layer auditable. The bootstrap chain produces a custom programming language (Kov) and compiler (kovc) targeting x86-64 + NVIDIA PTX directly. The language is designed for AI to read and write; an AI written in Kov runs on the Kov runtime; that AI eventually rewrites and extends itself under verifier gates.

## Stack identity

| Component | Name | Purpose |
|---|---|---|
| System / AI | **Kovostov** | the AGI-aspirational artifact |
| Language | **Kov** (`.kov`) | typed AI-focused language |
| Compiler | **`kovc`** | Kov → x86-64 + PTX |
| Runtime | **`kovrt`** | loads compiled kernels, manages devices |
| Bootstrap chain | hex0 → hex1 → M0 → M1 → M2 → C-subset → kovc-bootstrap → kovc (self-hosted) | stage0 lineage |

## The bootstrap chain (Phase 0–1)

Modeled on stage0 / Live Bootstrap / GNU Mes. Each stage source is in plain text; each stage *binary* is produced by the previous stage. A tiny seed at the bottom is hand-encoded.

```
hex0 seed (~200 bytes hand-encoded)
   reads: hex characters; writes: bytes
   ↓ hand-encoded; verified by objdump and nasm cross-check
hex1 (~600 bytes, in hex0 input format)
   adds: labels, comments
   ↓
M0 — minimal macro assembler (~few KB, in hex1 input format)
   adds: register names, mnemonics
   ↓
M1 (~larger, in M0 input format)
   adds: macros, basic structures
   ↓
M2-Planet (port; ~few hundred KB, in M1 input format)
   accepts: a tiny C subset
   ↓
kovc-bootstrap (in M2 C-subset, ~5–10 kLOC)
   accepts: the bootstrap subset of Kov
   ↓
kovc (in Kov, self-hosted)
   accepts: full Kov
```

**Target**: linux-x86_64 ELF (we use WSL2 on Windows for the bootstrap chain because ELF is dramatically simpler than PE). The eventual `kovrt` runtime targets Windows native (so the AI can use the RTX 5090 via the CUDA Driver API directly).

## Language design (committed)

Synthesized from the deep-research phase (see `docs/research/2026-05-03-language-survey.md`):

- **Spiritual ancestor**: Dex (typed index sets) + Triton (tile programming) + JAX (composable transforms) + Enzyme (AD on optimized IR)
- **Type system**: Futhark-style size types as type parameters; opt-in refinement constraints over Presburger arithmetic; gradual fallback for dynamic shapes
- **Tensor primitive**: tile-as-first-class with memory-space tag (`Tile<HBM>`, `Tile<SMEM>`, `Tile<REG>`, `Tile<TMEM>` on Blackwell)
- **Element type**: `(format, block_size, scale_format)` triple — supports BF16, FP8 (E4M3/E5M2), MXFP4, NVFP4, INT8, INT2-packed4, **Ternary {-1, 0, +1}**
- **Autodiff**: compiler pass invoked via library API (`grad`, `jvp`, `vjp`, `jacrev`, `hessian` as compositions of `linearize` + `transpose`); runs after primary optimization (Enzyme rule)
- **Effects**: typed effects (Dex-style) for IO, mutation, randomness
- **Linearity**: affine ownership for buffers; tensor views alias freely (Rust model)
- **Device**: phantom type parameter; functions device-polymorphic by default
- **Compilation**: AOT-default with cached JIT specialization

## Compiler architecture

```
.kov source
  ↓ lex + parse (recursive descent)
Surface AST (typed, source positions)
  ↓ type inference + size constraint solving (Presburger)
Tensor IR (value-semantic, named axes, layouts as types, structured ops)
  ↓ passes: shape inference, layout selection, fusion
            (vertical+horizontal+reduction-chain), const fold,
            DCE, memory planning (liveness), recompute, autodiff
Tile IR (explicit tiles, memory spaces, async ops, layouts)
  ↓ passes: tiling/blocking, vectorization, software pipelining,
            warp specialization, bank-conflict resolution,
            register allocation per-tile, autotune
  ↓ backend split
  ├─ x86-64 backend: linear-scan regalloc, AVX-512/AMX, ABI
  └─ PTX backend: text emission, virtual registers, WMMA/WGMMA, TMA
```

**IR style**: typed SSA with block parameters (Cranelift CLIF / Swift SIL pattern). No phi nodes.

## Phase roadmap

### Phase 0 — Stage0 bootstrap chain (months 1–6)
- hex0 seed (hand-encoded)
- hex1 → M0 → M1 → M2-Planet
- kovc-bootstrap in C-subset
- "Hello, world" through full chain, byte-audited
- **Verifiable artifact**: a string `"Hello, Kovostov.\n"` written to stdout, produced by a binary that compiled itself through the entire stage0 chain from hand-typed hex.

### Phase 1 — Kov language MVP (months 6–10)
- Full Kov frontend (lexer, parser, type checker, size constraints)
- Tensor IR + Tile IR + ~40 passes
- x86-64 backend (basic, with AVX-512)
- PTX backend (with WMMA + async TMA)
- Autotune harness
- AD compiler pass (`linearize` + `transpose`)
- ~50 stdlib ops written in Kov
- **Verifiable artifact**: train a 1M-param toy MLP on MNIST end-to-end in Kov, compiled by kovc, run on the 3070. Loss curves match a PyTorch reference.

### Phase 2 — First Kovostov model (months 10–14)
- Mamba2 + sparse attention hybrid architecture, in Kov
- Byte-level tokenizer
- Data pipeline: stream + tokenize FineWeb-Edu / SlimPajama / The Stack v2
- Pretrain a 100M–350M param model (BF16 baseline) on the 5090
- Optional cloud burst for 1B-scale
- **Verifiable artifact**: 100M model generates coherent next-byte text. No PyTorch in the runtime. Weights released under CC0.

### Phase 3 — Memory + world model (months 14–18)
- HD-vector episodic memory (10k-dim binary, XOR/popcount, PTX kernel)
- Hebbian write rule + Generative-Agents importance reweighting
- World model: JEPA-style latent predictor
- Continual learning: train on streamed experience without catastrophic forgetting
- **Verifiable artifact**: model learns 100 facts one-shot, retains across sessions.

### Phase 4 — Society + active inference (months 18–24)
- Specialists: planner, world-model, critic, retriever, executor, prover, curriculum-generator
- Global workspace blackboard + attention auction
- Active inference outer loop (free-energy minimization)
- **Decision point**: complete self-hosting of kovc (kovc rewritten in Kov)
- **Verifiable artifact**: multi-step reasoning ≥ 2× base LM on held-out benchmarks.

### Phase 5 — Self-play + auto-curriculum (months 24–36)
- Verifier infrastructure: code interpreter, Lean integration, ARC-AGI-3 harness, gym envs
- Auto-curriculum (Goldilocks-difficulty)
- AbsoluteZero-style self-generated tasks
- Self-distillation: successful self-play traces become training data
- **Verifiable artifact**: measurable capability growth over 4+ weeks of self-play, no new external data.

### Phase 6 — Self-modification under verifier gates (months 36–48)
- Kovostov gets code-edit privileges to its own specialists, prompts, kernels
- Verifier-gated commits
- AlphaEvolve / Voyager / Eureka pattern, applied to its own substrate
- **Verifiable artifact**: at least one non-trivial self-proposed improvement that survives 2 weeks of regression testing.

### Phase 7+ — AGI iteration (years)
The actual AGI attempt. Open-ended. Each cycle: identify a generalization gap → hypothesize a missing piece → implement (via Phase 6 self-modification) → measure on AGI-property axes → keep wins.

## Compute budget

| Phase | Local | Cloud | Approx cost |
|---|---|---|---|
| 0 | 3070 laptop | — | $0 |
| 1 | 3070 / 5090 | — | $0 |
| 2 | 5090 | optional 1× H100 ~50–100h | $0–300 |
| 3 | 5090 | — | $0 |
| 4 | 5090 | — | $0 |
| 5 | 5090 | optional 1× H100 ~100h | $0–300 |
| 6 | 5090 | spot 8× H100 ~20h | $0–500 |
| 7+ | 5090 | irregular | tbd |

Cumulative cloud over the life of the project: $1k–3k.

## Risk register

1. **Stage0 timeline blow-up.** Hand-encoded bootstrap is notorious for unexpected effort. Mitigation: month-3 gate — if hex0 + hex1 + M0 are not working at month 3, evaluate downscoping (e.g., adopt M2-Planet's existing seed verbatim instead of writing our own).
2. **Compiler complexity.** Full Kov compiler with own backend is ~25k LOC. Mitigation: phase boundaries are commit gates with verifiable artifacts; no Phase N+1 work until Phase N's artifact passes.
3. **Open-source lift.** Eventually we publish. Mitigation: develop in the open from day one. Public git, public docs, public design log.
4. **AGI-never-arrives.** Probably true on any short horizon. Mitigation: define success in terms of measurable AGI-property progress, not arrival.
5. **Self-modification Goodhart.** Phase 6 risks the system "improving" by gaming verifiers. Mitigation: held-out verifiers, transfer measurement.

## Update protocol

- Every phase ships a verifiable artifact. If a phase ships, plan stands.
- If a phase fails, this `PLAN.md` is updated with what was learned and what's being tried instead.
- Decisions go in `docs/decisions/YYYY-MM-DD-<slug>.md`.
- Daily progress goes in `docs/research-log.md`.
- Phase boundaries spawn a Kovostov critic / skeptic subagent to challenge self-reporting.
- Plan revisions are commits. The plan's history is auditable.

## License recap

- Code: Apache 2.0
- Documentation: CC-BY 4.0
- Model weights: CC0 (public domain)
- Training data references: links to public datasets only

This makes Kovostov genuinely usable, forkable, and improvable by anyone, anywhere, forever.
