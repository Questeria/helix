# Kovostov-Native

**An open-source AGI-aspirational system, bootstrapped from raw binary up.**

This is the from-scratch implementation of Kovostov: a local AI system where every layer above silicon is auditable, owned, and reproducible from a hand-written hex seed.

## Hard constraints

1. **Raw binary as the starting point.** No assembler, no compiler, no toolchain dependency for the language. The bootstrap chain begins with hand-encoded bytes that the user can audit one byte at a time. Existing tools (nasm, objdump) are used only for *audit/verification*, never to produce shipped artifacts.
2. **Open source, end-to-end.** All code (compiler, runtime, model architecture, training scripts), all weights, all training data references, and all documentation are public under permissive licenses. Apache 2.0 for code; CC-BY 4.0 for documentation; CC0 / public-domain for trained model weights to maximize freedom of use.
3. **Public training data only.** No proprietary datasets, no Claude/GPT/Gemini outputs as training data. Corpora drawn from FineWeb-Edu, SlimPajama, The Stack v2, Wikipedia, public math (Lean mathlib, ProofPile), public code, public ARC-AGI-3 puzzles.
4. **Eventual AGI.** The end goal is an artificial general intelligence — a system that exhibits sample-efficient learning, continual learning without catastrophic forgetting, compositional generalization, causal reasoning, and recursive self-improvement under verifier gates. This is a multi-year direction, not a release date.
5. **Consumer hardware.** Bootstrap and primary development on Windows 11 + RTX 3070 laptop / RTX 5090 desktop. Cloud bursts allowed for big training pushes.

## Stack identity

- **System / AI**: Kovostov
- **Language**: Helix (`.hx` source files)
- **Compiler**: `helixc`
- **Runtime**: `helixrt`
- **Bootstrap chain**: stage0 (hand-encoded hex) → hex1 → M0 (assembler) → M1 → M2 (C-subset compiler) → helixc-bootstrap (in C-subset) → helixc (self-hosted in Helix)

## Status (2026-05-11)

**1,490 tests passing** (1 skipped), Stage 28.9 active (bootstrap kovc.hx self-hosting effort), 34+ audit-driven cycles refining the Python frontend.

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

See [QUICKSTART.md](QUICKSTART.md) for build-and-run instructions, [`docs/ROADMAP.md`](docs/ROADMAP.md) for prioritized roadmap, [`docs/research/WAVE1_FINDINGS.md`](docs/research/WAVE1_FINDINGS.md) for the synthesized research direction, `docs/lang/spec.md` for the language reference, `docs/lang/tutorial.md` for a beginner guide, `docs/lang/agi-features.md` for the AGI-specific features deep dive.

## License

Apache License 2.0. See `LICENSE`.

Trained model weights, when produced, will be released under CC0 (public domain dedication) to maximize freedom for the AI community.

## Relation to the Kovostov framework at C:/Projects/Kovostov

The original `Kovostov` directory is the Claude-Code-shell framework that has been the cognitive scaffold while this from-scratch implementation is built. Once Kovostov-Native is operational, the shell-framework can be retired or repurposed as a development tool. The two share a name and a goal; they are different substrates.
