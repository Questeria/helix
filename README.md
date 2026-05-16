# Kovostov-Native

**An open-source AGI-aspirational system and Helix language stack, bootstrapped from raw binary up.**

This is the from-scratch implementation of Kovostov and Helix. Kovostov is the first flagship AGI-aspirational system built on the stack; Helix is the general-purpose language and compiler meant to serve any serious AGI project, scientific system, or high-certainty industrial system where correctness, provenance, and uncertainty reduction matter.

## Helix purpose

Helix is not meant to stay an internal language for Kovostov only. The long-term goal is for Helix to become the dominant open language for AGI development and for any field that benefits from auditable, reproducible, high-certainty computation: AI/AGI research, scientific discovery, medicine, genomics, physics, mathematics, robotics, climate, energy, infrastructure, education, and future industries that need machines to reason with evidence instead of guesswork.

The purpose of Helix is to remove uncertainty wherever software can honestly remove it: through typed effects, refinement and confidence types, proof-carrying compilation, deterministic self-hosting, explicit provenance, reproducible binaries, and verifier-gated self-improvement. Kovostov is the first major user of this language, not the boundary of its ambition.

## Hard constraints

1. **Raw binary as the starting point.** The target bootstrap chain begins with hand-encoded bytes that the user can audit one byte at a time. The production compiler is currently Python-hosted `helixc` until the self-hosted Helix compiler replaces it; existing tools (nasm, objdump) are used only for *audit/verification*, never to produce shipped artifacts.
2. **Open source, end-to-end.** All code (compiler, runtime, model architecture, training scripts), all weights, all training data references, and all documentation are public under permissive licenses. Apache 2.0 for code; CC-BY 4.0 for documentation; CC0 / public-domain for trained model weights to maximize freedom of use.
3. **Public training data only.** No proprietary datasets, no Claude/GPT/Gemini outputs as training data. Corpora drawn from FineWeb-Edu, SlimPajama, The Stack v2, Wikipedia, public math (Lean mathlib, ProofPile), public code, public ARC-AGI-3 puzzles.
4. **Eventual AGI.** The end goal is an artificial general intelligence — a system that exhibits sample-efficient learning, continual learning without catastrophic forgetting, compositional generalization, causal reasoning, and recursive self-improvement under verifier gates. This is a multi-year direction, not a release date.
5. **Consumer hardware.** Bootstrap and primary development on Windows 11 + RTX 3070 laptop / RTX 5090 desktop. Cloud bursts allowed for big training pushes.

## Stack identity

- **System / AI**: Kovostov, the first flagship AGI-aspirational system built on Helix
- **Language**: Helix (`.hx` source files), a general AGI and high-certainty computing language
- **Compiler**: `helixc`
- **Runtime**: `helixrt`
- **Target bootstrap chain**: stage0 (hand-encoded hex) → hex1 → M0 (assembler) → M1 → M2 (C-subset compiler) → helixc-bootstrap (in C-subset) → helixc (self-hosted in Helix)

## Status (2026-05-16)

**Current stage: Stage 35 audit cleanup.** Clean gates remain `0/3` as of the latest Stage 35 progress ledger, and the exact test count changes as each audit adds regressions. Continue from the newest pushed `git log -1 --oneline` and the tail of `docs/stage35-progress-2026-05-15.md`; restart 37 is the latest recorded fix sweep in this status text. Restart 37 fix verification collected 2,381 live `helixc/tests` pytest tests; run `python -m pytest helixc/tests --collect-only -q` for the current count.

The production compiler path is still the Python-hosted `helixc` implementation. A Helix self-hosted compiler remains the target of the bootstrap roadmap, not a shipped replacement for Python yet.

Major direction shift this session: Helix is now being optimized **for AI to USE and EXTEND, not for human developers**. Where ergonomics conflicts with structural regularity, structural regularity wins.

What works today:
- Hand-authored 299-byte ELF (`stage0/hex0/hex0.bin`) — the raw-binary foundation
- Working Helix compiler (`helixc`): parse → typecheck → IR → const-fold + CSE + DCE + fdce + effect-check → x86-64 → Linux ELF
- **Source-level forward + reverse-mode autodiff** as language built-ins (`grad`, `grad_rev`, `grad_rev_all`), with chain rules across user-defined function calls (via inlining) and stdlib transcendentals (analytic rules)
- **Verifier-gated reflection runtime**: 64 mutable cells in the binary's writable region. `quote`/`splice_f`/`modify_f` actually call your verifier function before committing
- **IR-level effect verification**: @pure functions transitively prohibited from effectful code
- 8 unique compile-time AGI type-system features (Presburger shapes, D<T>, memory tiers, agents, etc.)
- 30+ stdlib builtins: math, range-reduced exp/log/sin/cos, modern activations (sigmoid/tanh/silu/gelu/softplus/relu), losses (mse/mae/bce/huber), PRNG, optimizer steps. AD chain rules wired for all activations.
- I/O: `print_str`, `write_file`, `read_file_int` via raw syscalls
- 6 dogfood programs running real ML in Helix-emitted binaries:
  - 1-param gradient descent
  - 4-point linear regression (i32 cells)
  - Affine fit with f32 cells (200 iterations)
  - 2-layer ReLU net touching XOR
  - Logistic regression w/ sigmoid + BCE + multi-output AD ✨
  - Self-improving agent (flagship, composes everything)
- Did-you-mean error suggestions, algebraic identity folds (x*0, x-x, etc.)
- Property test verifying forward and reverse AD agree numerically

See [QUICKSTART.md](QUICKSTART.md) for build-and-run instructions, [`docs/HELIX_PURPOSE.md`](docs/HELIX_PURPOSE.md) for the broad Helix purpose, [`docs/HELIX_FINAL_PRODUCT_RESEARCH.md`](docs/HELIX_FINAL_PRODUCT_RESEARCH.md) for the research-backed final-product blueprint, [`docs/ROADMAP.md`](docs/ROADMAP.md) for prioritized roadmap, [`docs/research/WAVE1_FINDINGS.md`](docs/research/WAVE1_FINDINGS.md) for the synthesized research direction, `docs/lang/spec.md` for the language reference, `docs/lang/tutorial.md` for a beginner guide, `docs/lang/agi-features.md` for the AGI-specific features deep dive.

## License

Apache License 2.0. See `LICENSE`.

Trained model weights, when produced, will be released under CC0 (public domain dedication) to maximize freedom for the AI community.

## Relation to the Kovostov framework at C:/Projects/Kovostov

The original `Kovostov` directory is the Claude-Code-shell framework that has been the cognitive scaffold while this from-scratch implementation is built. Once Kovostov-Native is operational, the shell-framework can be retired or repurposed as a development tool. The two share a name and a goal; they are different substrates.
