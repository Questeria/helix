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

## Status (2026-05-04)

47 commits, **269 tests passing** across the entire pipeline.

What works today:
- Hand-authored 299-byte ELF (`stage0/hex0/hex0.bin`) — the raw-binary foundation
- Working Helix compiler (`helixc`): parse → typecheck → IR → const-fold + CSE + DCE → x86-64 → Linux ELF
- Real running programs: arithmetic, recursion, loops (for/while), arrays, floats (SSE), 3×3 matmul, neural-network forward pass
- 8 unique compile-time AGI features the type checker enforces:
  1. Tensor shape constraints via Presburger arithmetic
  2. Effect/capability typing (`@pure`, `@io`, etc.)
  3. Differentiable types `D<T>` with gradient propagation
  4. Memory-tier types (Working / Episodic / Semantic / Procedural)
  5. Reflection primitives (`quote`/`splice`/`modify`)
  6. Agent declarations (society of mind)
  7. Type-level transitions (`detach`/`attach`/`consolidate`/`recall`)
  8. Auto-curriculum primitive (`learn_to`)
- Symbolic forward-mode autodiff CLI
- Rust-style error messages with source-line context
- 9 working `.hx` example programs

See [QUICKSTART.md](QUICKSTART.md) for build-and-run instructions, `docs/PLAN.md` for the full plan, `docs/lang/spec.md` for the language reference, `docs/lang/tutorial.md` for a 10-step beginner guide, `docs/lang/agi-features.md` for the unique-features deep dive, and `docs/research-log.md` for the daily implementation log.

## License

Apache License 2.0. See `LICENSE`.

Trained model weights, when produced, will be released under CC0 (public domain dedication) to maximize freedom for the AI community.

## Relation to the Kovostov framework at C:/Projects/Kovostov

The original `Kovostov` directory is the Claude-Code-shell framework that has been the cognitive scaffold while this from-scratch implementation is built. Once Kovostov-Native is operational, the shell-framework can be retired or repurposed as a development tool. The two share a name and a goal; they are different substrates.
