# Kovostov-Native

**An open-source AGI-aspirational system and Helix language stack, bootstrapped from raw binary up.**

This is the from-scratch implementation of Kovostov and Helix. Kovostov is the first flagship AGI-aspirational system built on the stack; Helix is the general-purpose language and compiler meant to serve any serious AGI project, scientific system, or high-certainty industrial system where correctness, provenance, and uncertainty reduction matter.

## Helix purpose

Helix is not meant to stay an internal language for Kovostov only. The long-term goal is for Helix to become the dominant open language for AGI development and for any field that benefits from auditable, reproducible, high-certainty computation: AI/AGI research, scientific discovery, medicine, genomics, physics, mathematics, robotics, climate, energy, infrastructure, education, and future industries that need machines to reason with evidence instead of guesswork.

The purpose of Helix is to remove uncertainty wherever software can honestly remove it: through typed effects, refinement and confidence types, proof-carrying compilation, deterministic self-hosting, explicit provenance, reproducible binaries, and verifier-gated self-improvement. Kovostov is the first major user of this language, not the boundary of its ambition.

## Hard constraints

1. **Raw binary as the starting point.** The bootstrap chain begins with hand-encoded bytes the user can audit one byte at a time, and the toolchain is built **entirely from that raw root — there is no trusted pre-built compiler**. `hex0` (299 hand-authored hex bytes) → … → `seed` (an Apache-2.0 C-subset compiler) → `kovc` (the Helix compiler), each rung built only by the prior rung. The toolchain is **Python-free**: the repo holds **exactly one** committed `.py` (`verification/oracle/oracle_train.py`), a fenced numpy verification *oracle* that is **never** part of the toolchain. Existing tools (gcc, objdump, nasm) are used only as independent *auditors* (e.g. the gcc diverse-double-compile of the seed), never to produce shipped artifacts.
2. **Open source, end-to-end.** All code (compiler, runtime, model architecture, training scripts), all weights, all training data references, and all documentation are public under permissive licenses. The source license is Apache 2.0 (see `LICENSE`); the documentation license is CC-BY 4.0 (stated policy) and trained-model weights, when produced, will be released under CC0 (stated policy).
3. **Public training data only.** No proprietary datasets, no Claude/GPT/Gemini outputs as training data. Corpora drawn from FineWeb-Edu, SlimPajama, The Stack v2, Wikipedia, public math (Lean mathlib, ProofPile), public code, public ARC-AGI-3 puzzles.
4. **Eventual AGI.** The end goal is an artificial general intelligence — a system that exhibits sample-efficient learning, continual learning without catastrophic forgetting, compositional generalization, causal reasoning, and recursive self-improvement under verifier gates. This is a multi-year direction, not a release date.
5. **Consumer hardware.** Bootstrap and primary development on Windows 11 + RTX 3070 laptop / RTX 5090 desktop. Cloud bursts allowed for big training pushes.

## Stack identity

- **System / AI**: Kovostov, the first flagship AGI-aspirational system built on Helix
- **Language**: Helix (`.hx` source files), a general AGI and high-certainty computing language
- **Compiler**: `helixc`
- **Runtime**: `helixrt`
- **From-raw bootstrap chain (as built, no trusted pre-built compiler)**: `hex0` (299 hand-authored hex bytes) → `hex1` → `hex2` → `catm` → `M0` → `cc_amd64` → `M2-Planet` → `seed` (Apache-2.0 C-subset compiler) → `kovc` (the Helix compiler, `helixc/bootstrap/{lexer,parser,kovc}.hx`, self-hosted in Helix). Each rung is built **only by the prior rung**; no Python anywhere in the chain.

## Status (v1.3, 2026-06-05)

**The from-raw-binary trust chain is COMPLETE to PTX, and Python has been deleted from the toolchain.** The Helix-native compiler — `helixc/bootstrap/{lexer,parser,kovc}.hx`, a from-scratch compiler written *in Helix* that emits x86-64 ELF directly (no assembler/linker/libc) — is built **entirely from the raw-binary root** (`hex0` → … → `seed` → `kovc`), with **no trusted pre-built compiler** anywhere in the chain. The two honest records are **[`docs/TRUST_CHAIN_CLOSED.md`](docs/TRUST_CHAIN_CLOSED.md)** (the verified state + every residual, stated plainly) and **[`docs/CLEAN_REPRODUCTION.md`](docs/CLEAN_REPRODUCTION.md)** (rebuild the core chain from a clean checkout) — read those for the full, precise claims.

What is verified:

- **Self-host fixpoint, byte-identical.** `seed → K1 → K2 → K3 → K4`, with **K2 == K3 == K4 byte-for-byte** (the same test a self-hosted C compiler uses: stage2 == stage3 == stage4). The compiler written in Helix reproduces itself exactly.
- **Python-free toolchain.** The repo holds **exactly one** committed `.py` — `verification/oracle/oracle_train.py`, a fenced numpy verification *oracle* **never** referenced by the toolchain (`git ls-files "*.py"` == 1). The compiler/runtime are Helix plus a small hand-authored C subset (the `seed`).
- **Diverse-double-compile of the seed (anti-trusting-trust).** `gcc` (an independent lineage with zero M2-Planet ancestry) and the from-raw M2-Planet build independently produce a **byte-identical** seed/`K1` — a Wheeler DDC. `gcc` is an **auditor**, never the shipped root.
- **Real capability — the capstone.** A ≥2-layer transformer trains **end-to-end on kovc-emitted GPU (PTX) kernels** and converges to within **~2% (reproduced at ~0%) loss difference** of an *independent* numpy oracle (proven genuinely independent — it reads only the shared initial weights, never Helix's trajectory).

**Honest residuals (no overclaim — see `docs/TRUST_CHAIN_CLOSED.md` for the full list):**

- **GPU performance is a fraction of cuBLAS, NOT parity.** On the reference RTX 3070 Laptop (sm_86): ~50–67% of cuBLAS (G1 56%, G2 67.5%, G3 TF32 50–54%). Helix emits correct, reasonably-performant kernels; it does **not** beat NVIDIA's hand-tuned library. End-to-end capstone speedup is **7.0–8.7×** (Amdahl-bound), not the originally-estimated ≥10×; loss parity (the hard gate) holds at ~0%.
- **Complete to PTX, NOT to GPU machine code.** The hand-auditable from-raw chain ends at **PTX text**. Below PTX it trusts NVIDIA's **closed `ptxas`** + CUDA driver + GPU hardware + the C host launcher. The **CPU** path is all-the-way-down from raw binary; the **GPU** path is from-hex0-to-PTX-then-`ptxas` — the one trusted-once boundary, stated openly.
- **DDC + verification scope.** The seed/K1 byte-identical DDC covers the seed surface; the v1.1 language surface (generics/traits/closures/turbofish/wide-field/bf16) is cross-checked **behaviorally** by a zero-lineage interpreter (not a byte-identical second compiler — impossible by construction). The 5 clean adversarial reproductions plus a cross-model (ChatGPT, read-only) review share the build's model lineage / are a doc-logic review — **external operator reproduction on independent hardware remains open.** Single hardware target (sm_86); no cross-arch/AMD validation.

> The older "Stage NN" / "K-bootstrap chunk counter" / Python-parity-matrix framing is superseded. For live state, read in order: `git log --oneline -8`, `docs/TRUST_CHAIN_CLOSED.md`, `docs/CLEAN_REPRODUCTION.md`, `scripts/gate_kovc.sh` (the universal gate: self-host fixpoint + 109-program corpus + PTX regression + diagnostics).

Major direction shift this session: Helix is now being optimized **for AI to USE and EXTEND, not for human developers**. Where ergonomics conflicts with structural regularity, structural regularity wins.

What works today:
- Hand-authored 299-byte ELF (`stage0/hex0/hex0.bin`) — the raw-binary foundation
- Working Helix compiler (`helixc`): parse → typecheck → IR → const-fold + CSE + DCE + fdce + effect-check → x86-64 → Linux ELF
- **Self-hosting Helix-native compiler** (`helixc/bootstrap/{lexer,parser,kovc}.hx`, `kovc`): a complete lexer + parser + x86-64-ELF code generator written *in Helix*, built from the raw-binary `seed` (no Python) into a native binary that compiles Helix programs — including its own source. Proven **byte-identical self-host fixpoint** (K2 == K3 == K4) and gated by a **109-program feature corpus** (`scripts/gate_kovc.sh`) spanning integer widths, floats incl. bf16/f16 arithmetic, control flow, generics, traits + default methods, closures (incl. capturing-by-value), pattern matching, wide struct fields, and structured `path:line:col` diagnostics
- **Source-level forward + reverse-mode autodiff** as language built-ins (`grad`, `grad_rev`, `grad_rev_all`), with chain rules across user-defined function calls (via inlining) and stdlib transcendentals (analytic rules)
- **Verifier-gated reflection runtime**: 64 mutable cells in the binary's writable region. `quote`/`splice_f`/`modify_f` actually call your verifier function before committing
- **IR-level effect verification**: @pure functions transitively prohibited from effectful code
- 8 unique compile-time AGI type-system features (Presburger shapes, D<T>, memory tiers, agents, etc.)
- Stdlib in `helixc/stdlib/*.hx` (16 modules, ~455 functions; see `helix_website/HELIX_REFERENCE.md` Standard Library section for live per-module counts): math + range-reduced transcendentals, modern activations (sigmoid/tanh/silu/gelu/softplus/relu), losses (mse/mae/bce/huber), PRNG, optimizer steps, reverse-AD, AGI search/match/memory/world primitives, hashmap, tensor/iterator/vec/string/result helpers. AD chain rules wired for all activations.
- I/O: `print_str`, `write_file`, `read_file_int` via raw syscalls
- 6 programs total (5 dogfood + 1 self-improving-agent flagship) running real ML in Helix-emitted binaries:
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
