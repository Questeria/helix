# Helix

**A source-available, fully auditable programming language whose entire compiler is rebuildable from 299 hand-typed bytes — and runs real neural networks on the GPU, verifiably.**

Helix is a from-scratch, self-hosting language and compiler for machine learning and high-certainty systems work. Its defining property: the *whole toolchain* is built from a raw-binary root with **no trusted pre-built compiler** anywhere in the chain — so you can audit it from the very first byte up to the GPU kernels it runs. Helix exists to remove uncertainty wherever software honestly can: deterministic self-hosting, reproducible binaries, explicit provenance, source-level autodiff, typed effects, and verifier-gated reflection.

## What makes Helix different

- **Verifiable from raw binary.** The bootstrap chain begins with hand-encoded bytes you can audit one at a time: `hex0` (299 hand-authored hex bytes) → `hex1` → `hex2` → `catm` → `M0` → `cc_amd64` → `M2-Planet` → `seed` (a C-subset bootstrap compiler) → `kovc` (the Helix compiler, written *in Helix*). Each rung is built **only by the prior rung** — there is no trusted pre-built compiler, and **no Python in the toolchain** (the repo holds exactly one committed `.py`: a fenced numpy verification *oracle*, never part of the compiler).
- **Self-hosting, proven byte-identical.** `seed → K1 → K2 → K3 → K4`, with **K2 == K3 == K4 byte-for-byte** — the compiler written in Helix reproduces itself exactly.
- **Anti-"trusting-trust."** An independent `gcc` lineage (zero M2-Planet ancestry) and the from-raw build produce a **byte-identical** seed/`K1` — a Wheeler diverse-double-compile. `gcc` is only an *auditor*, never the shipped root.
- **Runs real ML on the GPU.** `kovc` emits PTX for a covered transformer-kernel set that executes on real NVIDIA hardware: a ≥2-layer transformer trains end-to-end on kovc-emitted GPU kernels to within ~2% (reproduced ~0%) loss of an independent numpy oracle, and GPT-2 (124M and the 1.5B XL) runs **token-for-token-verified** on the same stack (see the demo).
- **Runs *billion-parameter* models on small GPUs, verifiably (v1.6).** Qwen3-8B — and even 32B — run on a single 8 GB consumer GPU via a from-scratch 4-bit (NVFP4) quantizer + per-layer streaming, each run emitting a reproducible **commitment + calibrated-envelope receipt** a minimal-trust verifier re-checks. Faithful *within a calibrated envelope* (the greedy next token matches the fp32 reference on decisive prompts) — honest scope, not "execution-proven" (see *Bigger models* below).
- **ML-native language.** Forward + reverse-mode autodiff as built-ins (`grad`, `grad_rev`, `grad_rev_all`), tile/tensor types, an effect system (`@pure`), and a verifier-gated reflection runtime — language features, not libraries.

The two authoritative trust records are **[`docs/TRUST_CHAIN_CLOSED.md`](docs/TRUST_CHAIN_CLOSED.md)** (the verified state + every residual, stated plainly) and **[`docs/CLEAN_REPRODUCTION.md`](docs/CLEAN_REPRODUCTION.md)** (rebuild the chain from a clean checkout). Read those for the full, precise claims.

## Status — trust core v1.3 (byte-stable as of 2026-06-05)

> The `seed.c` + `kovc.hx` sources are byte-stable, so the fixpoint / `K1` / seed SHAs do not move. The **GPT-2 demo layers** (the bring-your-weights verified-execution layer, below) were added later (2026-06) on top of that byte-stable core; the demo's host harnesses live *outside* the self-host fixpoint.

**The from-raw-binary trust chain is COMPLETE to PTX, and Python has been deleted from the toolchain.**

Verified:
- **Self-host fixpoint, byte-identical** — `K2 == K3 == K4` (the same test a self-hosted C compiler uses: stage2 == stage3 == stage4).
- **Python-free toolchain** — exactly one committed `.py` (the fenced numpy oracle, never in the compile/run path); `git ls-files "*.py"` == 1.
- **Diverse-double-compile of the seed** — `gcc` and the from-raw `M2-Planet` build independently produce a byte-identical seed/`K1`.
- **Real capability (the capstone)** — a transformer trains end-to-end on kovc-emitted PTX kernels, converging to within ~2% (reproduced ~0%) of an *independent* numpy oracle (it reads only the shared initial weights, never Helix's trajectory).

**Honest residuals (no overclaim — full list in [`docs/TRUST_CHAIN_CLOSED.md`](docs/TRUST_CHAIN_CLOSED.md)):**
- **GPU performance is ~50–67% of cuBLAS, NOT parity** (reference RTX 3070 Laptop, sm_86). Helix emits correct, reasonably-performant kernels; it does **not** beat NVIDIA's hand-tuned library. End-to-end capstone speedup is **7.0–8.7×** (Amdahl-bound); loss parity (the hard gate) holds at ~0%.
- **Complete to PTX, NOT to GPU machine code.** The hand-auditable chain ends at **PTX text**; below it trusts NVIDIA's closed `ptxas` + CUDA driver + GPU hardware + the C host launcher. The **CPU** path is all-the-way-down from raw binary; the **GPU** path is from-hex0-to-PTX-then-`ptxas` — the one trusted-once boundary, stated openly.
- **Verification scope.** The byte-identical DDC covers the seed surface; the broader v1.1 language surface (generics/traits/closures/turbofish/wide-field/bf16) is cross-checked **behaviorally** by a zero-lineage interpreter (a byte-identical second compiler is impossible by construction). Single hardware target (sm_86); fp32; **external operator reproduction on independent hardware remains open.**

> For live state, read in order: `git log --oneline -8`, [`docs/TRUST_CHAIN_CLOSED.md`](docs/TRUST_CHAIN_CLOSED.md), [`docs/CLEAN_REPRODUCTION.md`](docs/CLEAN_REPRODUCTION.md), and `scripts/gate_kovc.sh` (the universal gate: self-host fixpoint + 109-program corpus + PTX regression + diagnostics).

## What works today

- The hand-authored 299-byte ELF (`stage0/hex0/hex0.bin`) — the raw-binary foundation.
- **Self-hosting Helix-native compiler** (`helixc/bootstrap/{lexer,parser,kovc}.hx`): a complete lexer + parser + x86-64-ELF code generator written *in Helix*, built from the raw `seed` (no Python) into a native binary that compiles Helix programs — including its own source. Byte-identical self-host fixpoint, gated by a **109-program feature corpus** (`scripts/gate_kovc.sh`) spanning integer widths, floats (incl. bf16/f16), control flow, generics, traits + default methods, closures (incl. capture-by-value), pattern matching, wide struct fields, and `path:line:col` diagnostics.
- **Source-level forward + reverse-mode autodiff** as built-ins (`grad`, `grad_rev`, `grad_rev_all`), with chain rules across user-defined functions (via inlining) and stdlib transcendentals (analytic rules).
- **Verifier-gated reflection runtime** — `quote` / `splice_f` / `modify_f` call your verifier function before committing a mutation (64 mutable cells in the binary's writable region).
- **IR-level effect verification** — `@pure` functions are transitively prohibited from effectful code.
- **GPU codegen** — `kovc` emits PTX for a covered transformer-kernel set (matmul/attention/softmax/layernorm/RMSNorm/RoPE/SwiGLU/activations) that runs on real NVIDIA GPUs, each gated for numerical parity against an independent oracle.
- **Compile-time type-system research features** — Presburger-checked shapes, confidence types `D<T>`, memory tiers, agent types, and more.
- **Stdlib** in `helixc/stdlib/*.hx` (16 modules, ~455 functions): math + range-reduced transcendentals, activations (sigmoid/tanh/silu/gelu/softplus/relu), losses (mse/mae/bce/huber), PRNG, optimizer steps, reverse-AD, search/match/memory primitives, hashmap, tensor/iterator/vec/string/result helpers.
- Real ML programs running in Helix-emitted binaries: 1-param gradient descent, 4-point linear regression, an affine fit with f32 cells, a 2-layer ReLU XOR net, logistic regression with sigmoid + BCE + multi-output autodiff, and a flagship agent that composes everything.

## Demo — GPT-2 on Helix (verifiable execution)

A self-contained demo runs **GPT-2 — the real, unchanged public model (a 2019 base completion model, not an assistant)** on this from-raw stack, and proves it. The pitch is **trust, not speed**: a bring-your-weights *verifiable execution layer* whose every layer and kernel traces back to 299 hand-typed bytes, output matched **token-for-token** to an independent numpy reference and reproducible bit-for-bit.

- **Runbook (start here):** [`docs/HELIX_GPT2_DEMO_RUNBOOK.md`](docs/HELIX_GPT2_DEMO_RUNBOOK.md) — the operator script, honest-residuals card, and how a third party produces the weights from HuggingFace.
- **Live chat (GPT-2-XL on Helix):** `bash scripts/serve_chat_demo.sh`, then open <http://127.0.0.1:8848/?source=sse>. Bound to `127.0.0.1`; gated green by `scripts/helix_serve_gate.sh`.
- **One-command attestation:** `bash scripts/gpt2_demo_attest.sh` — fail-closed; rebuilds the compiler from raw, runs GPT-2 124M through kovc-emitted kernels token-for-token vs the oracle, re-runs byte-identical, and writes a signed attestation. The proof dashboard is `demo/dashboard.html`.

## Bigger models — v1.6 (Qwen3-8B and 32B on an 8 GB GPU)

v1.6 takes the same verify-don't-trust idea to *large* models. **Qwen3-8B — and even Qwen3-32B (≈4× the card's memory) — run on a single 8 GB consumer GPU** (reference RTX 3070), using a from-scratch **4-bit NVFP4 quantizer** + per-layer weight streaming so the full model never has to fit in VRAM at once. Each run writes a small, reproducible **receipt**.

**What the receipt does — and does not — prove (stated edge-first):**

- It is a **commitment + calibrated-envelope** receipt: SHA-256 over the weights, the logits, and the next-token argmax, plus a per-model calibrated bound τ. A **minimal-trust verifier** — rebuildable from the 299-byte root, with `ptxas` de-trusted and a NIST-KAT-checked SHA-256 — re-checks it and **rejects tampering by named reason** (wrong logits, wrong model, drift outside the envelope, and the "teeth" case of a too-tight declared bound).
- The quantized run is **faithful within that calibrated envelope** — the greedy next token matches the fp32 reference on decisive prompts — but it is **not** bit-identical token-for-token (it is 4-bit), and the receipt **commits to** a run; it does **not** yet prove the GPU executed every layer faithfully (exact per-layer verification is deferred to a later release). Prior art (CommitLLM, TAO, zkLLM) is acknowledged — the contribution is the *minimal-trust* verifier, not a first.

Reproduce (needs an NVIDIA GPU + the model weights):

```bash
git checkout v1.6-qwen3-32b-receipt
bash scripts/gpu_qwen3_receipt_check.sh   # -> RECEIPT_GATE_PASS (genuine 8B/32B + named-reject negative controls)
```

## Faster — v1.7 (speed pass)

v1.7 keeps v1.6's receipts byte-for-byte and makes the path **fast**. On the same 8 GB RTX 3070 the warm Qwen3-8B forward dropped from **181.6 s to ~12.6 s (~14×)** by moving the NVFP4→f32 dequant onto the GPU, and the KV-cache **decode** path (which was crashing and skipping Qwen3's per-head QK-norm in v1.6) is fixed — it now generates token-correct text at **~4.6 s/token**. Every step is **opt-in** (`HX_DQPTX`; `HX_HOSTDEQ=1` restores the exact v1.6 host path) and **byte-identical** to v1.6 (gated on `V3_UPLOAD_CHECK_PASS` + unchanged greedy argmax), with **no compiler edit** — the 299-byte self-host fixpoint is untouched. Two more ambitious levers (Tensor-Core GEMM, a fused-dequant decode kernel) were **measured and shelved** because they lost to the incumbent at our shapes. Full record + honest residuals: [`docs/HELIX_V1.7_SPEED.md`](docs/HELIX_V1.7_SPEED.md).

## Repository layout

- `stage0/` — the from-raw bootstrap ladder (`hex0` … `M2-Planet`, then the `helixc-bootstrap/seed`). Vendored rungs keep their own upstream licenses.
- `helixc/` — the Helix compiler: `bootstrap/{lexer,parser,kovc}.hx` (the self-hosting compiler), `stdlib/`, `runtime/`, `examples/`.
- `demo/` — the GPT-2-on-Helix verified-execution demo (live chat + proof dashboard).
- `scripts/` — the gates and runbooks (`gate_kovc.sh`, `reproduce_trust.sh`, `capstone_audit.sh`, the demo scripts).
- `docs/` — trust records, language reference, and the development history.
- `verification/` — the single fenced numpy oracle used by the independent gates.

## Build & docs

See **[QUICKSTART.md](QUICKSTART.md)** to build and run. Language reference: `docs/lang/spec.md`; beginner tutorial: `docs/lang/tutorial.md`; the full reference (features, stdlib, compiler internals): [`helix_website/HELIX_REFERENCE.md`](helix_website/HELIX_REFERENCE.md).

## License

**Helix is source-available and free for non-commercial use — but not open-source.** The Licensed Work is provided under the **Helix Non-Commercial License** — see [`LICENSE`](LICENSE). You may use, run, build with, study, modify, and share Helix freely for any **non-commercial** purpose (personal, learning, academic and other non-commercial research, evaluation, non-profit). **Commercial use is reserved** (drawn broadly): it requires a separate license and covers **any use by or for a company or other for-profit business — of any size, for any purpose, even internal or evaluation use — and any commercial or paid use of a modified or derived version.** For example: running it (or anything built or derived from it) in a **data center** or hosted/cloud service; use by an **AI/ML or other technology company** to train, serve, or build models, products, or services; offering it as a **paid or software-as-a-service** product; or any other for-profit, revenue-generating use. Contact via GitHub for a commercial license.

Vendored third-party components under `stage0/` (M2-Planet, M2libc, blood-elf, and the hex/M0/M1/catm/cc_amd64 rungs) remain under their own upstream licenses (GPLv3, etc.); using them to build Helix does not place Helix under those licenses.

> **Note on earlier documents:** some historical planning/design docs in this repo (kept for transparency and provenance) describe an *intended* Apache-2.0 / open-source / CC-BY / CC0 release. Those reflect earlier intent and are **superseded** by [`LICENSE`](LICENSE). The Licensor may choose to open-source Helix in the future but makes no such commitment here.
