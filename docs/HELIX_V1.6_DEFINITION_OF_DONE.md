# Helix v1.6 — Definition of Done

**Status:** ✅ **v1.6 SHIPPED 2026-06-15** — Qwen3-8B (warm-up) + Qwen3-32B (headline) run on the 8 GB RTX 3070 via NVFP4 + per-layer streaming, with a calibrated verifiable receipt. Push HELD. **⚠ The one-line goal, pitch, and Tier-1 items in the DESIGN TEXT below are the ORIGINAL pre-build plan; the SHIPPED SCOPE CORRECTION immediately below is authoritative — where they differ, the correction governs.**

## ⚠️ SHIPPED SCOPE CORRECTION (v1.6 release, 2026-06-15) — authoritative

The design text below was written pre-build and over-describes relative to what shipped. The honest shipped scope:

- **Model:** **Qwen3-8B** (warm-up) and **Qwen3-32B** (headline, ~8× the 8 GB card's fp16 capacity) — **not** Qwen2.5-14B. Both run end-to-end on the 8 GB RTX 3070 via NVFP4 4-bit + per-layer streaming.
- **What the receipt PROVES:** **Tier-2** commitment/reproducibility — SHA-256 of the committed weights + output logits + argmax, all re-derivable; and **Tier-3** an **empirical calibrated envelope** (`max_abs(logits − fp32_oracle) ≤ τ` AND argmax match), τ provenance-labeled (see `HELIX_V1.6_TAU_CALIBRATION.md`). The verifier TCB is rebuildable from the 299-byte seed (fixpoint `cdcf8673`), ptxas de-trusted, with a from-scratch NIST-KAT'd SHA-256.
- **What the receipt does NOT prove (DEFERRED):** **execution-faithfulness is NOT cryptographically proven.** Tier-1 exact per-layer Freivalds (DoD #4 / the "faithfully executed the committed model" language in the goal/pitch) is **DEFERRED** (f32 GEMM makes a tolerance-Freivalds unsound). A party holding the committed weights + *any* in-envelope logit vector could mint a passing receipt — this is the **disclosed minimal-trust scope**, not a flaw. Release-facing claims must say **commitment + empirical envelope**, never execution-faithfulness.
- **"Checkable faster than re-running it" (one-line goal / DoD #5) is RETRACTED for the shipped checker:** the GPU-free checker **re-derives the commitments + the envelope** (re-hash weights/logits, recompute max_abs/argmax vs the oracle); it does **not** re-execute the forward, so it is **not** a faster faithful-re-execution substitute. (A faster-than-re-exec faithfulness check is precisely the deferred Tier-1.)
- **Honest perf:** slow — ~5.7 s/layer (8B), ~12.7 s/layer (32B); CPU-side dequant dominates; speed is v1.7's job. No manufactured speedup.
- **Prior art:** CommitLLM, TAO, zkLLM — this is **minimal-trust** verification, **NOT** "first verifiable quantized inference."

Net: the DoD checklist below holds **as corrected here** — #4/Tier-1 = **DEFERRED** (not green), #5 = the GPU-free commitment+envelope check (not faster-faithful-re-exec). Full per-item evidence: `HELIX_V1.6_RELEASE_READINESS.md`.

**P0 reconciliation (2026-06-14):** design-investigate done — **gate PASS, GO for P1, NO `kovc.hx` edit** (verified against the code: `nvfp4_dequant_kernel.hx:12,24-25`; `tiled_matmul_abt_kernel.hx:21`; `gate_kovc.sh:44`). One mandatory correction absorbed below: the dequant path is **dense f32** (the `@kernel` emits f32; the tiled GEMM is f32-only), **not** f16; and the tied head is **4-bit-packed dequant-per-step**, not f16-resident (an f16 head needs a new kernel → deferred to v1.7 with the f16 VRAM saving). P1 starts with the **Qwen2.5-7B VRAM-resident warm-up**.
**One-line goal:** *(⚠ ORIGINAL PRE-BUILD DESIGN — superseded by the SHIPPED SCOPE CORRECTION above: shipped Qwen3-8B/32B, not 14B; the receipt proves commitment + empirical envelope, NOT execution-faithfulness; Tier-1 Freivalds + "faster than re-running" DEFERRED.)* run **Qwen2.5-14B** on the **8 GB RTX 3070 Laptop** GPU — should-be-impossible without 4-bit quant + host-RAM layer streaming — **and emit a verifiable receipt** that the quantized/streamed run faithfully executed the committed model and stayed within a declared numerical envelope of the fp32 reference, **checkable faster than re-running it**, by a verifier rebuildable from 299 bytes with ptxas de-trusted.

---

## ⚠️ The honest novelty (READ FIRST — this corrects the pitch)

The design pass found that the original framing ("nobody offers a checkable proof that a quantized run is faithful") is **FALSE**, and an adversarial audit would sink the release if we claimed it. State it correctly from day one:

- **Prior art we do NOT re-claim:** layer-streaming/offload (llama.cpp `-ngl`, HF accelerate, Helix's own v1.4 leg); 4-bit quant (GPTQ/AWQ/bitsandbytes/MXFP4/NVFP4); **and verifiable quantized/approximate inference itself** — **CommitLLM** (Freivalds + Fiat-Shamir, CPU-only verifier, ~12–14% overhead — the *same core* we'd use), **TAO** (operator-level tolerance/acceptance regions = literally the "envelope"), and **zkLLM/ZKML** (full ZK over quantized LLMs).
- **The ONE uncontested differentiator:** every prior verifiable-inference system runs its verifier on a conventional **unverified** Python + CUDA + **ptxas** stack — its trusted computing base silently *includes* the compiler and GPU toolchain it attests (the Trusting-Trust hole; CommitLLM explicitly "keeps the provider on the normal serving path"). **Helix is the only one whose checker is rebuildable from 299 bytes** (seed→K1→K2→K3→K4 byte-identical, fixpoint `cdcf8673`), with **ptxas de-trusted** for a kernel (v1.5 #3) and a from-scratch **NIST-KAT-gated SHA-256**.
- **So the claim is:** *minimal-trust* verification — "a quantized-inference faithfulness receipt with a 299-byte-reproducible, ptxas-de-trusted verifier TCB" — **not** "first verifiable quantized inference." **Cite CommitLLM + TAO explicitly** in the docs as prior art.

**Honest pitch line:** *Helix v1.6 runs Qwen2.5-14B on a should-be-8GB-impossible laptop GPU by 4-bit quantization + layer streaming — which llama.cpp also does — and emits the one thing nobody else does: a checkable receipt, produced by a verifier you can rebuild from 299 bytes with ptxas removed from its trust base, that proves (exactly for the quantized matmuls, spot-checked for the rest) the streamed run faithfully executed the committed model and stayed within a declared numerical envelope of the full-precision reference — verified faster than re-running it.*

---

## Model + the 8 GB fit math (why 14B, not 7B)

**Qwen2.5-14B** (Apache-2.0; 48 layers, hidden 5120, 40 q / 8 KV heads = GQA×5, head_dim 128, intermediate 13824, RoPE θ=1e6, RMSNorm, vocab ~152k, ~14.7B params) — ~95% the v1.4-verified Llama family; 48 layers already run (GPT-2-XL).

- **fp16** ~29 GB — ~4× the 8 GB card. Never fits VRAM.
- **Q4 (4-bit)** ~8.5–9.0 GB (web-confirmed GGUF) — **still exceeds ~7.3 GB usable VRAM** → **host-RAM layer-streaming is load-bearing** (one layer VRAM-resident at a time).
- **fp32** ~58 GB — **overruns the 31.8 GB host RAM** → **4-bit is load-bearing for the RAM tier too**.
- A 7–8B model *would* fit at 4-bit in VRAM → no streaming needed → the "impossible" hook collapses. **The 14B choice is load-bearing.**
- **Decode VRAM budget (S=2048), corrected after P0:** the dequant `@kernel` emits **dense f32** (not f16) and the tiled GEMM is f32-only, so 1 layer is ~0.8 GB f32 (not 0.55 f16) + a **4-bit-packed head, dequant-per-step** + KV f16 (~0.4) + scratch (~0.6) ≈ **~3 GB**, fits one-layer-at-a-time in 7.3 GB. The f16 VRAM saving is deferred to v1.7 (needs a tiled-f16 GEMM).
- **The verified OOM trap:** the tied LM head `d_wte_pad` (gpt2_infer.c:1385) is a full NVpad×DM resident buffer ≈ **3.1 GB at fp32** for Qwen's 152k vocab — ~42% of usable VRAM, OOMs once a layer + KV land. **P1 keeps the head 4-bit-packed and dequants per step** into the f32 head GEMM (mirrors layer-streaming); an f16-resident head doesn't compose (can't feed an f32 GEMM; needs a new kernel — deferred). (v1.4's 49k-vocab head hid this.)
- **Honest caveat baked in:** 14B-4bit on this card is run routinely by llama.cpp/Ollama — the feat is **not** first-to-run, it is first-to-run-**with-a-checkable-faithfulness-receipt** on a 299-byte-rebuildable, ptxas-de-trusted toolchain.

---

## The 3-tier receipt (the moat)

Built one gate-able increment at a time on the verified #4 spine. **Key insight:** separate the two error sources — **(A) quantization-encoding error** (fp32→nearest 4-bit) is a fixed property of the committed model, not a per-run honesty question; **(B) execution-faithfulness** (did the GPU compute what W_q specifies) is **exact and integer-checkable** because 4-bit dequant is dyadic-exact. Offloading changes *when/where* a tile computes, never *what* dtype it specifies.

- **Tier 1 — EXACT per-layer Freivalds** (= #4 generalized): proves `C_layer = dequant(W_q)·X` exactly mod p, with the `|C|<p/2` guard + checker-derived Fiat-Shamir challenge. Lift the per-block power-of-2 scale into the field exactly. Over 48 layers, **raise t to 3** (union bound ~2^-87) and state it. Verifier O(MK+KN+MN) ≪ O(MKN), CPU-only.
- **Tier 2 — hash-chained transcript:** `h_{i+1}=SHA(h_i‖tag‖H(in)‖H(out))` over every op incl. the f32 nonlinear glue (rmsnorm/rope/silu/softmax/residual); verifier **spot-checks k of L** layers by deterministic re-derivation (kovc kernels are bit-reproducible); catches a fraction-f cheat w.p. 1−(1−f)^k. Realizes the v1.5-deferred #4 second increment.
- **Tier 3 — fp32 envelope** (the honestly-probabilistic part, where Freivalds is FALSE): argmax-preserved + `max_abs(logits−oracle) < τ`, extending the gate's existing scaffold; **τ provenance stated** (prefer empirical-calibrated-and-labeled over analytic-Lipschitz). NEVER claim cryptographic soundness here.
- **Does NOT prove:** the fp32 model is correct/safe (oracle is the trusted reference); not ZK; not a SNARK; Tier-3 is analytic/empirical not cryptographic; attention/softmax stay sampled (Tier-2); 4-bit-exact-Freivalds needs the dyadic dequant lifted into integer code-space first (don't claim until green).

---

## Definition of Done (all must be green, honestly)

1. **Model runs** — Qwen2.5-14B end-to-end on the 8 GB card via 4-bit NVFP4 (MXFP4 fallback) + existing per-layer streaming (dequant → **dense f32** → the existing f32 GEMM), emitting ≥N greedy tokens; QKV-bias wired; tied head **4-bit-packed + dequant-per-step** (not the fp32 ~3.1 GB OOM trap; the f16-resident route is deferred — needs a new kernel).
2. **Impossible-honestly** — documented proof fp16 (~29 GB) and Q4 (~8.5–9 GB > 7.3 GB usable) and fp32 (~58 GB > 31.8 GB RAM) bounds, **with** the explicit prior-art caveat (llama.cpp/CommitLLM also run this).
3. **Envelope defined** — written Tier-3 acceptance region (argmax + max_abs<τ) vs the fp32 numpy oracle, τ provenance stated, bound into the receipt.
4. **Receipt emitted** — a real run emits the 3-tier receipt (Tier-1 exact Freivalds per layer, t=3; Tier-2 re-derivable transcript binding model hash / scales / stream schedule / envelope).
5. **Independently checked** — a from-scratch-C, GPU-free, ptxas-de-trusted checker verifies it **faster than re-running**, rooted in the 299-byte TCB; rehearsed on a FRESH machine before DONE.
6. **≥4 negative controls rejected, each by a NAMED check** — (a) flipped 4-bit nibble→Freivalds; (b) wrong block-scale→Freivalds; (c) dropped/stale streamed layer→transcript spot-check; (d) output drifts OUTSIDE the envelope→Tier-3 reject (proves the envelope has teeth).
7. **Honest perf** — measured s/token (expect ~15–40 s/token, slower than v1.4 XL's ~10) reported truthfully, no manufactured speedup; trust-first, speed → v1.7; demo uses a short generation.
8. **Fixpoint + gate + audit** — self-host fixpoint stays `cdcf8673` byte-identical (ALL new logic host-side in cuda_launch.c + gpt2_pack.c + gpt2_infer.c, **no kovc.hx edit**); universal gate passes (1 committed .py, PTX byte-identical, corpus, gcc-DDC); ships only after **several consecutive clean independent adversarial audits**, with the corrected novelty claim + CommitLLM/TAO cited.

---

## Phasing (each gate-able + adversarial-audited, v1.5 rhythm)

- **P0 — De-risk / design (no GPU build):** finalize Qwen2.5-14B; write the **envelope spec** (τ + provenance) and the **receipt threat-model** doc (3-way soundness ledger: exact / spot-check / bounded-empirical); confirm the v1.5 NVFP4/MXFP4 primitives **compose** into dequant→GEMM via the existing @kernel-decode + existing GEMM with **no kovc.hx edit**; cite CommitLLM + TAO. **Gate:** design-audit PASS, novelty corrected.
- **P1 — Capability (the visible feat):** **P1a (FIRST): Qwen2.5-7B VRAM-resident** to de-risk importer + QKV-bias + dequant with no streaming. Then: extend `gpt2_pack.c` additively (48-layer dims, NVFP4 packing, QKV bias + 152k vocab + θ=1e6 in an HXGW v3 header; the oracle must quantize identically); wire `add_bias()` into `forward_layer_llama`'s q/k/v (3 call-sites + decode mirror); dequant-on-upload in `upload_layer_ll` → **dense f32** (the `@kernel` emits f32, feeding the existing f32 GEMM); head **4-bit-packed dequant-per-step**; **pad/tile the dequant for Qwen K=5120/13824** (violate kk%112==0 → cuda_launch.c:1979 rejects non-conforming K). 14B runs end-to-end within the Tier-3 envelope; corrupted-weights NC. Delivers DoD 1,2,3,6d,7.
- **P2 — The novelty (the moat):** Tier 1 first (= #4 generalized, start on the ternary/exact-int path, then lift NVFP4 dyadic dequant into integer code-space — the one genuinely-new kernel + NCs; t=3, state the union bound) → Tier 2 (transcript) → Tier 3 (envelope→receipt field). Build the from-scratch ptxas-de-trusted checker, faster-than-re-exec. Delivers DoD 4,5,6a,6b,6c.
- **P3 — Demo + close:** the should-be-impossible end-to-end demo (run + receipt + INDEPENDENT check rehearsed on a fresh machine — ship the checker as a self-contained from-raw-buildable binary + a pinned reference receipt + a one-command check); universal gate; consecutive-clean-audit streak; honest TRUST_CHAIN/DoD doc citing CommitLLM/TAO; **tag `v1.6-*`** (preserve ALL existing tags).

---

## Top risks (mitigations are load-bearing)

1. **Over-claim (audit-sinking):** do NOT say "first/only verifiable quantized inference" — CommitLLM/TAO/zkLLM exist. Claim ONLY the de-trusted-verifier-TCB differentiator; cite them.
2. **Receipt soundness over f32:** Freivalds is UNSOUND with a tolerance (a forged in-envelope C passes = theater). Keep Freivalds **exact** on the integer dequantized GEMM; push f32 approximation into Tier-2 transcript + Tier-3 envelope; never claim crypto soundness for f32.
3. **Tier-1-on-4bit isn't free:** exact Freivalds proven only for ternary/int today; extending to NVFP4 needs the dyadic scale folded into the field exactly, or `|C|<p/2` breaks. Don't pre-announce 4-bit-exact-Freivalds until the FP4 kernel + data-independent NCs are green.
4. **Tied-head OOM trap:** P1 keeps the head 4-bit-packed + dequant-per-step (an f16-resident head needs a new kernel — deferred); verify VRAM headroom in P1.
5. **Host-RAM outer wall + envelope vacuity:** scope to 14B (70B-4bit ~35 GB needs a new disk→RAM tier — OUT of scope); a too-loose τ makes the receipt vacuous → REQUIRE the outside-envelope NC (DoD 6d); calibrate τ against the oracle's own quant error (TAO pattern), documented, never hand-tuned to pass.
6. **Importer correctness:** Qwen2.5 is bf16 + QKV bias + 152k vocab + θ=1e6; a quant bug passes ptxas and yields plausible-but-WRONG text. Oracle must quantize identically; NCs must be data-independent; land the 7B warm-up first.
7. **Gate weaker than v1.4 + possible fixpoint move:** quantized 14B won't match an fp32 oracle token-for-token, so v1.4's 25/25-greedy-id method doesn't transfer — the gate becomes logits-within-τ-of-a-4-bit-aware-oracle + receipt-verifies; DECIDE + DOCUMENT this bar before building. If dequant→GEMM ever needs a kovc.hx intrinsic, the fixpoint `cdcf8673` MOVES (re-mint + 3-way byte-identical + reasoned commit, the S0/S1 pattern) — first attempt the no-kovc-edit composition.

---

## Discipline (carried from v1.5)

From-raw + Python-free (exactly 1 committed `.py`); all new code host-side C/.sh, no new committed `.py`, **no kovc.hx edit** (fixpoint `cdcf8673` byte-identical); never ship red / never overstate; SERIAL builds (never concurrent with a Workflow); gate via gate_ext4.sh; commit LOCALLY, **push HELD until an explicit owner nod**; tag `v1.6-*` only, preserve all tags; RTX 3070 + CUDA 12.8; Telegram only on genuine completion/blocker.
