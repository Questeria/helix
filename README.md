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

## Status

Phase 0 — bootstrap chain. Project initialized 2026-05-03.

See `docs/PLAN.md` for the full plan, `docs/research-log.md` for daily progress, and `docs/decisions/` for the dated decision history.

## License

Apache License 2.0. See `LICENSE`.

Trained model weights, when produced, will be released under CC0 (public domain dedication) to maximize freedom for the AI community.

## Relation to the Kovostov framework at C:/Projects/Kovostov

The original `Kovostov` directory is the Claude-Code-shell framework that has been the cognitive scaffold while this from-scratch implementation is built. Once Kovostov-Native is operational, the shell-framework can be retired or repurposed as a development tool. The two share a name and a goal; they are different substrates.
