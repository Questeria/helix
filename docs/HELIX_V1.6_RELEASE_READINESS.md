# Helix v1.6 — Release readiness

**Status: RELEASE-READY** (pending the standing push-hold). Built + verified 2026-06-15 on an
RTX 3070 Laptop (8 GB, CUDA 12.8). Push HELD until an explicit owner nod.

## The honest one-liner

> Helix v1.6 runs **Qwen3-8B** and **Qwen3-32B** — the 32B is ~8× the 8 GB card's fp16 capacity — on
> that 8 GB GPU, by NVFP4 4-bit quantization + per-layer host→device streaming (which llama.cpp/Ollama
> also do), and emits the one thing they don't: a **checkable receipt**, produced by a verifier you can
> rebuild from a 299-byte seed with `ptxas` removed from its trust base, that **commits** the run (model
> + logits + argmax hashes) and certifies it stayed within a **calibrated empirical envelope** of the
> fp32 reference — checkable **GPU-free**. It is **minimal-trust verification, NOT** "first verifiable
> quantized inference" (CommitLLM, TAO, zkLLM are prior art).

## Definition-of-Done — per-item status (the 8 items, honestly)

See `HELIX_V1.6_DEFINITION_OF_DONE.md` (read its **SHIPPED SCOPE CORRECTION** first — it governs over the
pre-build design text).

1. **Model runs** ✅ — Qwen3-32B end-to-end on the 8 GB card (64 layers, 23.5 GB streamed, ~13.5 min/forward), finite logits; Qwen3-8B likewise (36 layers). Argmax-exact vs the fp32 oracle on decisive prompts.
2. **Impossible-honestly** ✅ — fp16 (8B ~16 GB, 32B ~64 GB) and even 4-bit-resident exceed 8 GB → streaming is load-bearing; prior-art caveat stated (llama.cpp/CommitLLM/TAO).
3. **Envelope defined (with provenance)** ✅ — Tier-3 acceptance region `max_abs(logits−oracle) ≤ τ` + argmax, with **calibrated** τ (empirical/TAO) and provenance in the receipt + `HELIX_V1.6_TAU_CALIBRATION.md`. τ_8B=8.0438, τ_32B=11.2202 (= 2× worst measured deviation per model).
4. **Receipt emitted** ✅ (with one tier **DEFERRED, stated**) — Tier-2 commitment (SHA-256 of weights+logits+argmax, re-derivable) + Tier-3 empirical envelope. **Tier-1 exact per-layer Freivalds is DEFERRED** (f32 GEMM makes a tolerance-Freivalds unsound) → execution-faithfulness is **not** cryptographically proven; this is the disclosed minimal-trust scope.
5. **Independently checked, GPU-free** ✅ — a from-scratch-C checker (`--v3-receipt-check`), no CUDA, ptxas de-trusted, NIST-KAT'd SHA-256, rebuildable from the 299-byte seed (fixpoint `cdcf8673`). It re-derives the commitments + the envelope; it does **not** re-run the forward (so it is **not** a faster faithful-re-execution substitute — that would be the deferred Tier-1).
6. **≥4 negative controls, each by a NAMED reject** ✅ — `REJECT=LOGITS_HASH` (forged logits), `MODEL_HASH` (tampered model hash), `TIER3_ENVELOPE` (wrong oracle / out-of-envelope), `TIER3_ARGMAX` (the real 32B near-tie run, 279≠15473), plus a **teeth NC** (a run declared with τ below its own max_abs → `REJECT=TIER3_ENVELOPE`). All in `scripts/gpu_qwen3_receipt_check.sh` → `RECEIPT_GATE_PASS`.
7. **Honest perf** ✅ — slow: ~5.7 s/layer (8B), ~12.7 s/layer (32B); CPU-side NVFP4 dequant dominates; decode re-streams per token. No manufactured speedup. **Speed is v1.7's job** (wire the existing GPU dequant kernel). No benchmark-suite (MMLU/perplexity) accuracy is claimed — only logit-level fidelity (corr 0.976–0.984) on the measured prompts.
8. **Fixpoint + gate + audit** ✅ — all v1.6 logic is host-side C (gpt2_pack.c / gpt2_infer.c); **no kovc.hx edit**, self-host fixpoint stays `cdcf8673` byte-identical. The GPU-free regression gate passes; adversarial audits below.

## What the receipt proves — and does not

- **Proves:** the committed weights hash to the receipt's `model_sha256`; the committed logits hash to `logits_sha256`; those logits lie within the **calibrated** τ of the trusted fp32 oracle with matching argmax. Verifiable GPU-free, faster than re-running the forward, by a 299-byte-rebuildable + ptxas-de-trusted checker.
- **Does NOT prove:** that the logits were *produced by* the quantized forward (execution-faithfulness) — Tier-1 Freivalds is deferred; a party holding the committed weights + any in-envelope logit vector could mint a passing receipt. Nor exact next-token agreement on near-ties (4-bit reorders logits within the quant noise; argmax-exact holds on **decisive** prompts). Nor benchmark accuracy (not measured). Tier-3 is empirical, never cryptographic.

## Calibration (τ) — summary

Empirical/TAO: τ = 2.0 × max(max_abs over a diverse calibration set), per model, pre-declared, provenance-stamped.
- 8B (N=3: near-tie 2.488 / capital-of-France 3.264 / France-repetition 4.0219) → τ=8.0438.
- 32B (N=2: near-tie 2.974 / France-repetition 5.6101) → τ=11.2202.
- **Caveats (disclosed):** the 32B set is small (N=2; forwards are ~13.5 min) and one 32B point is a near-tie argmax flip; the shipped receipts are emitted on calibration-set (in-distribution) prompts. Generalization rests on the 2× margin + the independent max_abs recomputation + the teeth NC, not on held-out receipts. Expanding the set is cheap v1.7 work. Full detail: `HELIX_V1.6_TAU_CALIBRATION.md`.

## Verification artifacts

- Gate: `scripts/gpu_qwen3_receipt_check.sh` → `RECEIPT_GATE_PASS` (KAT + genuine 8B/32B + calibrated release receipts + determinism + 5 NCs by named reject incl. teeth).
- Release receipts: `q3_8b_release.receipt`, `q3_32b_release.receipt` (calibrated τ + `tier3_tau_prov`).
- Checker / emitter: `helixc/runtime/gpt2_infer.c` (`run_v3_receipt`, `run_v3_receipt_check`, `run_v3_receipt_from_logits`, `calib_tau`).

## Adversarial audit log

- **Round 1** — critic (code/evidence): **PASS** (HIGH); skeptic (claims/honesty): **FAIL** — found the authoritative DoD still carried pre-build claims (Qwen2.5-14B; "faithfully executed the committed model"; "checkable faster than re-running it") that overclaimed vs the honest shipped artifacts. Fixed in `504df37` (DoD SHIPPED SCOPE CORRECTION; τ-figure reconciliation).
- **Round 2** (post-fix, fresh independent) — skeptic: **PASS** (p=0.82, HIGH, no critical/major); critic: **PASS** (p=0.92, HIGH, no critical/major). Both confirmed: no governing surface claims execution-faithfulness; τ sound + matches receipts; gate independently reproduced; commit hygiene clean. Minor findings (dangling RELEASE_READINESS pointer; in-distribution caveat; thin 32B N) addressed in this doc + the calibration doc.

## Prior art (cited, honestly)

CommitLLM (Freivalds + Fiat-Shamir, CPU verifier), TAO (operator tolerance/acceptance regions = the "envelope"), zkLLM/ZKML (full ZK over quantized LLMs). Helix's only uncontested differentiator: the verifier's TCB is **rebuildable from 299 bytes with ptxas de-trusted** — every prior system runs its verifier on an unverified Python+CUDA+ptxas stack.
