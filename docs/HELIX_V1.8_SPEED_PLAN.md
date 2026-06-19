# Helix v1.8 Speed Plan

_From the helix-v18-speed-plan ultracode workflow (5 code investigators + 4 increment designs + 4 adversarial verdicts + synthesis), 2026-06-19. Self-host fixpoint at planning time: cdcf8673._

---

The numbers and gate citations check out against source: `v3_upload_gpu:1637-1680` shows the per-token memcpy+HtoD (`:1649-1650`), the scale-cache reuse (`:1655-1665`), the `f_dq_tiled` dequant launch (`:1670`), and the Kpad→K compaction (`:1672-1678`); the `g_sc_res[400]` resident scale cache and `~1.74GB`/`L*11+t` keying are confirmed at `:246-251`. I have what I need to synthesize the plan honestly.

# HELIX v1.8 SPEED PLAN

**Status:** synthesis of 4 increment designs + adversarial verdicts + 6 subsystem investigations. **Authoritative numbers reconciled against source** (`helixc/runtime/gpt2_infer.c`, `helixc/bootstrap/kovc.hx`, `docs/HELIX_V1.7_SPEED.md`). Self-host fixpoint at synthesis time: **`cdcf8673`** (a kovc-binary hash, distinct from git HEAD `0248ddf`).

**Governing discipline (carried verbatim from v1.7):** every increment is (1) opt-in behind a new `HX_*` flag, default OFF, exact slow path as fallback; (2) measured against the incumbent *before* wiring — the INC4 lesson (oracle-correct yet 1.8× slower → reverted); (3) faithful — byte-identical where numerics are unchanged, faithful-within-envelope (re-calibrated τ) where not; **faithfulness is never traded for speed.**

---

## 1. PRIORITIZED INCREMENT ORDER

The split is determined by one fact, verified directly against `kovc.hx`: a cross-thread reduction (`shfl` = **0 occurrences**; SMEM/`bar.sync` reachable **only** through the `ptx_name_is_*` fused-intrinsic ladder) and an fp16 mma opcode (only `m16n8k8.tf32.tf32` is hardcoded) **cannot be expressed in a plain `@kernel` body**. Any kernel needing them requires a `kovc.hx` emitter edit, which moves the `cdcf8673` self-host fixpoint → **ATTENDED** (owner approval + DDC re-mint). Everything else is host/C plumbing → **AUTONOMOUS**.

| # | Increment | Flag | Fixpoint | Track | Gate type |
|---|-----------|------|----------|-------|-----------|
| **INC1** | Resident packed 4-bit layer words + scales (8B) | `HX_PACKEDRES` | none | **AUTONOMOUS** | byte-identical |
| **INC1.5** | Resident **packed lm_head** (replace FP32 `d_wte_pad`) | `HX_PACKEDHEAD` | none | **AUTONOMOUS** | byte-identical |
| **INC4** | Streaming overlap (double-buffered prefetch, 32B) | `HX_OVERLAP` | none | **AUTONOMOUS** | byte-identical |
| **INC2** | Parallel warp-reduction fused NVFP4 dequant-GEMV (decode) | `HX_FUSEDGEMV` | **moves** | **ATTENDED** | within-envelope (τ_v18) |
| **INC3** | fp16 Tensor-Core GEMM + fused dequant (long prefill) | `HX_TCGEMM` | **moves** | **ATTENDED** | within-envelope (τ_v18) |

### Ordering rationale (corrected by the verdicts)

**INC1 first, autonomously** — it is the only increment that is faster-or-equal *by deletion* (it strictly removes the per-token packed-words memcpy+HtoD; adds nothing to the hot path) and is the prerequisite that makes INC2/INC3 worth their fixpoint cost. **Do INC1.5 immediately after** — the verdicts are unanimous that without a packed lm_head the 8B resident budget does not fit 8 GB at usable context (the FP32 `d_wte_pad` is 2.318 GiB, the single biggest avoidable cost). I have **split the packed-head out of INC2's design into its own autonomous INC1.5** because three of the four designs smuggled it in as a co-requisite of a fixpoint-moving kernel — it is host-only and must not be gated behind an attended change.

**INC4 (overlap) is autonomous and independent** — pure host-C stream/event scheduling, no kovc edit, byte-identical. It is the *only* lever for 32B and can land in parallel with INC1/INC1.5.

**INC2 and INC3 are attended and must be gated on a measurement that INC1/INC1.5 enable.** The verdicts converge hard here: do **not** pay the irreversible fixpoint cost on an unproven bet. The mandatory sequence is:

1. Land INC1 + INC1.5 (autonomous, fixpoint-safe).
2. **Prototype the cheap half first:** wire the *existing serial* `gemv_abt_nvfp4` to read the now-resident packed words (no kovc edit), and run `HX_PROF` on the **decode** path to split `g_pf_htod` vs `g_pf_dq`.
3. **Only if** that measurement shows HtoD/materialization was a non-trivial fraction *and* the serial-on-resident result still loses to f32 (confirming the win must come from parallelism, not just residency) is the INC2 fixpoint move justified. Likewise INC3 is gated on a large-M (≥512) prefill benchmark beating an **upgraded cp.async A@Bᵀ** baseline, not the current G1 one.

### Overclaim corrections applied from the adversarial verdicts

- **INC1 headline narrowed.** The design framed the win as "HtoD + scale-rebuild." Since `HX_PACKEDRES` implies `g_scache` (scale cache already exists, `:1655`), the **marginal** new win is *only* the packed-words HtoD. Verdict: "MAGNITUDE UNPROVEN" — gate the wiring on the measured `g_pf_htod`/total fraction on the **decode** path, not the estimate.
- **INC2 decode speedup cut.** Design claimed "~1.5–2.5×". Verdicts (grounded in `SPEED.md:79-85`: decode is **re-dequant-bound, not gemv-bound** — the block=1→128 occupancy change "barely moved it"; residency "shaves the per-token HtoD, NOT the dequant") put the realistic landed figure at **~1.3–1.8× (decode ~4.6 → ~2.6–3.5 s/tok), with ~30–40% probability of a near-null result** if dequant-ALU dominates and warp-reduction + per-op `cuCtxSynchronize` overhead eats the coalescing win. Decode-GEMV has M=1 / **zero weight reuse**, so the win is bounded by the packed-read bandwidth minus inline-unpack ALU minus sync — *not* the reduction math.
- **INC3 "~2×" reframed as a ceiling, not an expected value.** GEMM is **~1.6% of the warm 12.6 s forward** (`SPEED.md`, Amdahl) — "infinitely-fast GEMM saves ~0.2 s." The real payoff is eliminating the `v3_upload` f32 round-trip on long prefill, **unmeasured on this box**, and the design under-scoped the emitter work: it needs **two** new PTX capabilities (`m16n8k16.f16.f16.f32` mma **and** `ldmatrix` — kovc has zero ldmatrix today), not "one new opcode."
- **INC4 mechanism corrected.** Design credited part of the win to hiding per-op `cuCtxSynchronize` bubbles — **wrong**: the measured 4.6 s/tok decode runs under `g_fast`, which already skips `if(!g_fast) SYNC` (`:254`). Those bubbles are already gone. The only legitimately hideable non-transfer work is the dequant-launch + Kpad→K compaction. Realistic: **32B prefill ~1.2–1.5× on the streaming portion; 32B decode ~1.05–1.2× at best, possibly ~0×** if 32B (like 8B) turns out dequant-kernel-bound rather than HtoD-bound (copy stream and dequant kernel contend for the one GPU).

---

## 2. EXPECTED CUMULATIVE SPEEDUP (honest ceilings)

**Optimistic target = same order of magnitude as llama.cpp. NOT parity. A hand-written non-tensor-core GEMV will not beat NVIDIA's tuned kernels — stated as a hard ceiling, not a hedge.**

### 8B decode (baseline: ~4.6 s/token, re-dequant-bound, RTX 3070 Laptop 8 GB)

| After | Cumulative decode | Confidence |
|-------|-------------------|------------|
| INC1 + INC1.5 (resident packed, autonomous) | ~4.0–4.4 s/tok (removes per-token HtoD only; dequant launch still runs) | measured-gated; could be ~5–15% |
| **+ INC2 (parallel fused GEMV, attended)** | **~2.6–3.5 s/tok (~1.3–1.8×)** | unvalidated upper bound; **~30–40% chance of near-null** |
| Theoretical bandwidth floor (packed read ~5.4–7.3 GB/tok @ ~448 GB/s) | sub-20 ms/tok | **NOT achievable on this box** — strided 4-bit unpack runs well below peak BW; inline E2M1 if-ladder ALU + per-op sync dominate |

**Honest decode ceiling:** low-single-digit-× improvement bringing decode toward **~2.6–3.5 s/tok**, *not* the bandwidth ideal and *not* llama.cpp parity. The gap is dominated by the 8 GB card, the intrinsically uncoalesced 4-bit unpack, and per-op sync. Same-order-of-magnitude as llama.cpp is the stretch target the parallel GEMV *aims* at; it is explicitly not promised.

### 8B prefill (baseline: warm forward ~12.6 s; GEMM step is ~1.6% of it)

| After | Effect | Confidence |
|-------|--------|------------|
| INC1 + INC1.5 | Removes per-forward packed HtoD each layer; compounds with resident scales | measured-gated |
| + cp.async A@Bᵀ upgrade (cheap, no-fixpoint — **recommended before INC3**) | f32 G1→G2 was +19% historically; zero new numeric surface | high confidence |
| **+ INC3 (fp16 TC-GEMM, attended)** | GEMM **step** up to ~2× (capped by cuBLAS-TF32 ~10.6 TFLOP/s vs current G2 5.445); real payoff = eliminating the f32 weight round-trip on long prefill | ceiling only; end-to-end small until weight stream removed |

**Honest prefill ceiling:** the GEMM-step ceiling is **~2× the current f32 throughput**, and **only** for compute-bound long prefill (M≥512) *after* resident weights remove the ~10–12 s stream. At M=64 (typical chat turn) the TC kernel is expected to **lose** (memory-bound; the measured TF32 outcome was 0.97× f32). End-to-end forward gain from the GEMM math alone is small (Amdahl ~1.6%); the larger, honest prefill win is the memory-traffic elimination, which is unproven on this hardware and must be measured.

---

## 3. THE HONEST 32B REALITY

**32B cannot be resident on this card. This is a hard physical wall, not a tuning problem.**

Computed from the same packing formulas (`Kpad=ceil(K/112)*112`; packed words = `rows·(Kpad/7)·4`; scales = `rows·(Kpad/16)·4`), 32B config NL=64, DM=5120, DFF=25600:

| 32B component | Size |
|---|---|
| Packed layer words (64 layers) | **15.61 GiB** |
| Layer effective-scales | 6.83 GiB |
| **Packed layer weights (words+scales)** | **22.44 GiB** |
| Total resident (packed head + KV + acts @ Smax=2048) | **~26.3 GiB** |

The packed layer **words alone (15.6 GiB) do not fit 8 GB**; the full packed layer set (22.4 GiB) is **~2.8× the card**; total resident is **~3.3×**. Even the scales cache alone (6.83 GiB) is too big. **32B MUST stream every layer per token from mmap** via the existing `v3_upload_gpu` path. `HX_PACKEDRES` must **no-op for 32B** via the fit-check.

**32B decode floor:** decode reads ~16 GB of packed words/token over a single PCIe link at a realistic ~12–16 GB/s effective HtoD ⇒ a hard **~1.0–1.3 s/token floor** that no amount of overlap can beat. Double-buffering **hides latency, it does not reduce bytes moved.**

**The only 32B win is streaming overlap (INC4):** it makes 32B *usable by streaming at ~the PCIe floor* and shaves **prefill** (~1.2–1.5× on the streaming portion, where there is per-layer compute to hide behind). It is **not a decode speedup of consequence** (~1.05–1.2× at best, possibly ~0×) and must not be sold as one. Report prefill and decode separately; state the ~1 s/token 32B decode floor explicitly.

---

## 4. FAITHFULNESS DISCIPLINE

Two regimes, drawn directly from the gate findings (`gpt2_infer.c` + `HELIX_V1.6_TAU_CALIBRATION.md`):

### Byte-identical gate — INC1, INC1.5, INC4 (numerics UNCHANGED)

These are pure plumbing/scheduling: the **same** i32 packed words from the **same** mmap, the **same** `g_e4m3_tab[micro]*ts` scales (byte-identical by construction, `:1658`), the **same** `f_dq_tiled` kernel into the **same** f32 scratch — only *where they live* (INC1/1.5) or *when they run across two streams* (INC4) changes. So the v1.7 contract's existing gate applies **unchanged, no new τ**:

1. `--v3-upload-check` with the flag ON must print **`V3_UPLOAD_CHECK_PASS`** (`run_v3_upload_check:1864-1901` is an **exact memcmp**, `:1893`, no tolerance). **Exercise the flag ON** so the L0 dequant is fed from the resident/overlapped path, not the fallback — otherwise the memcmp only proves the fallback is byte-identical.
2. **`--v3-smoke` 8B greedy argmax must stay `279`** (the v1.7 pin).
3. Flag **unset** must reproduce the prior path **bit-for-bit**, and `HX_HOSTDEQ=1` must still restore the exact v1.6 host path — so **every v1.6/v1.7 receipt still reproduces**.
4. **Negative-control teeth for INC4:** because `--v3-upload-check` only uploads L0 once, it **cannot** trigger INC4's multi-layer/multi-token buffer-recycle race. The real teeth must be the **full `--v3-smoke` (8B argmax==279)** and **32B `--v3-receipt-check`** run end-to-end under `HX_OVERLAP=1`; a deliberately-dropped `cuStreamWaitEvent` build must produce an argmax/logits-hash **mismatch** on that multi-layer path.

### Faithful-within-envelope gate — INC2, INC3 (numerics CHANGED)

A parallel reduction reorders FP accumulation (non-associative); fp16 multiplicands lose mantissa. `V3_UPLOAD_CHECK_PASS` (memcmp) **cannot pass and must not be faked.** The correct gate is the calibrated Tier-3 envelope vs a **trusted fp32 oracle**:

1. **`max_abs(logits − fp32_oracle) ≤ τ_v18`**, where **τ_v18 is RE-CALIBRATED** via the TAO method (`2.0 × max|logit_fast − oracle|` over the diverse calibration set, **pre-declared before seeing pass/fail**). Reusing **τ_8B=8.0438 / τ_32B=11.2202 is UNSOUND** — those cover NVFP4-quant-vs-fp32 only; the new kernels add an independent reduction-order / fp16-accumulation error source. Record `HX_TAU_PROV` stating τ_v18 now covers (quant + reduction/fp16) error.
2. **argmax-exact on DECISIVE prompts only** (e.g. the "…is Paris" fixture) — **not** near-ties, where 4-bit + fp16 legitimately reorder (`top1-margin < 2·tau`).
3. **Build (don't just recommend) a top-k / correlation sub-gate** — the current verifier checks only L∞ max_abs + argmax; a single outlier can pass max_abs while ranking degrades. Add a top-5-set-match or Pearson/Spearman floor vs the oracle.
4. **Re-prove both negative-control teeth against the NEW kernel:** drift > τ_v18 → `REJECT=TIER3_ENVELOPE`; decisive-prompt argmax flip → `REJECT=TIER3_ARGMAX`. Without the teeth re-proven on the new kernel, the envelope is self-fulfilling.
5. **Never label Tier-3 cryptographic** — it is empirical. Exact Freivalds (Tier-1) stays deferred for f32 GEMM.
6. **Fixpoint obligation (INC2/INC3 only):** after the `kovc.hx` edit, re-mint the self-host ladder and prove a **3-way DDC** (K2==K3==K4 byte-identical) closing the trusting-trust loop, record old→new hash (`cdcf8673` → new), and confirm all existing corpus kernels still compile byte-identically. Confirm the new emitter avoids the immutable-`let` footgun (a `mut` reassigned with an if-expression result mis-allocates an unbound `-1` register).

**Every increment is opt-in behind its own `HX_*` flag, default OFF, with the exact slow path as default** (the `HX_DQPTX`/`HX_HOSTDEQ` template at `:1453-1470,:1682-1683`). This is the guarantee that v1.6/v1.7 receipts reproduce: a verifier re-running `--v3-receipt-check` with the new flag unset re-derives identical logits/hashes; the fast path ships its **own** receipt under τ_v18, alongside (not replacing) the old ones.

---

## 5. FIRST CONCRETE STEP — INC1 (resident packed 4-bit), ready to implement

**Goal:** keep the packed NVFP4 i32 layer words + the effective per-16-block scales resident in VRAM for 8B Qwen3, so `v3_upload_gpu` skips the per-token packed-words memcpy+HtoD (`:1649-1650`) and the scale rebuild (`:1657-1665`), relaunching only the **same** `f_dq_tiled` dequant into the **same** f32 scratch. **Fixpoint-safe (no kovc edit). Byte-identical by construction.**

### Exact 8B VRAM math (formulas verified against the descriptor at `:1644-1647`)

NL=36, DM=4096, QD=4096, KVD=1024, DFF=12288, NV=151936:

| Component | Size |
|---|---|
| Resident packed layer words (36 layers) | **3.731 GiB** |
| Resident layer effective-scales (36 layers) | **1.632 GiB** (= 1.753 GB — matches the `~1.74GB` comment at `:249` exactly) |
| **Resident packed layer weights (subtotal)** | **5.363 GiB** |
| FP32 lm_head `d_wte_pad` (left FP32 in INC1) | 2.318 GiB |
| KV cache @ Smax=2048 | 0.562 GiB |
| Activations + logits @ Smax=2048 | ~1.85 GiB |
| **INC1-alone total @ Smax=2048** | **~10.1 GiB — OVER an 8 GiB card** |

**Therefore INC1 fits 8B on 8 GB only at bounded context (Smax≤~1024 ⇒ ~6.9 GiB) OR once INC1.5 packs the head.** A real "8 GB" card exposes only ~7.6–7.8 GiB after the desktop. **INC1.5 (packed head, −1.84 GiB) is the immediate follow-on that makes Smax=2048 comfortable.** 32B is excluded by the fit-check (does not fit at any context — §3).

### Implementation (host/C only)

1. Add globals beside `g_sc_res` (`:250`): `static CUdeviceptr g_pk_res[400] = {0};` and `static int g_packedres = 0;`. Reuse the existing idx = `L*11+t` keying.
2. Parse `HX_PACKEDRES` in `main()` alongside the other `HX_*` getenvs; **imply `g_scache=1` when `g_packedres`** so both halves go resident together (byte-identical — `g_sc_res` is INC2c-shipped byte-identical).
3. In `v3_upload_gpu` (`:1644-1670`), branch around the memcpy+HtoD: if `g_pk_res[idx]` use it; else if `g_packedres && fit_ok`, `cuMemAlloc(rows*kwords*4)` + one HtoD **directly from the mmap `w`** (one-time, no pinned bounce), store the ptr; **non-fatal on alloc failure** → fall back to the existing pinned-bounce streaming path (same discipline as the scale cache at `:1659-1663`). Pass the resident ptr into the unchanged `f_dq_tiled` launch (`:1669`).
4. **Add a `cuMemGetInfo` fit-check before any resident `cuMemAlloc`** (there is currently **no** `cuMemGetInfo` in the file — additive). Pre-sum the **full** co-resident budget, not just packed+scales: packed (3.731) + scales (1.632) **+ FP32 head `d_wte_pad` 2.318** (until INC1.5) **+ `d_logits` (Smax·NVpad·4 = 1.158 GiB @ Smax=2048)** + the **still-allocated fallback/lm_head scratch `d_packed`+`d_sc`+`d_dqscr`** (must remain — the lm_head chunked dequant uses `d_packed` at `:1757`) + KV + activations. Budget the **transition peak** (first-touch `cuMemAlloc` coexists with the live `d_packed` scratch). If it won't fit free VRAM, log `[packedres] need X MiB, free Y MiB -> streaming` and leave `g_packedres=0` (graceful degrade); cap/recommend Smax≤1024 for 8B until INC1.5.
5. **Teardown cleanup (load-bearing, not optional):** free `g_pk_res[]` (and `g_sc_res[]`) at teardown — each first-touch `cuMemAlloc`s ~3.7 GiB; repeated serve sessions leak otherwise.
6. Decode (`decode_step_llama:1214-1216`) and prefill (`forward_layer_llama`) need **no edits** — they call `upload_layer_ll → v3_upload → v3_upload_gpu`, which now transparently skips the HtoD when resident; the dequant launch still runs, so outputs are bit-for-bit identical.

### INC1 gate (byte-identical)

- Run `--v3-upload-check` **with `HX_PACKEDRES=1`** → require `V3_UPLOAD_CHECK_PASS` (exact memcmp at `:1893`), ensuring L0 is fed from `g_pk_res[idx]` not the fallback.
- `--v3-smoke` 8B argmax must stay **`279`**.
- `HX_PACKEDRES` unset reproduces the prior path bit-for-bit; `HX_HOSTDEQ=1` restores the exact v1.6 host path.
- **Measure-before-wire (the INC4 lesson, non-negotiable):** use `HX_PROF` to report `g_pf_htod / total` on the **decode** path against a **scale-cache-ON baseline** (since `g_packedres` implies `g_scache`, the marginal win is *only* the packed-words HtoD). Spending 5.363 GiB of VRAM is justified only if that fraction is non-trivial; if HtoD is ~1–3% of 4.6 s/tok, INC1 is correct-but-pointless VRAM pressure — gate the wiring on the **measured number**, not the estimate.

---

## REMAINING OVERCLAIMS FLAGGED

1. **INC2 "~1.5–2.5× decode" is an unvalidated upper bound, not an expected value.** Realistic landed ~1.3–1.8×, with ~30–40% probability of near-null. The headline must be framed as a ceiling gated on the measure-before-wire result.
2. **INC1's headline ("removes HtoD + scale-rebuild") overstates the marginal win.** Because it implies `g_scache`, the scale-rebuild is *already* removed at baseline; the new win is **only** the packed-words HtoD.
3. **INC3 "one new opcode" under-scopes the emitter work** — it needs **two** new PTX capabilities (`m16n8k16.f16.f16.f32` mma **and** `ldmatrix`, which kovc lacks entirely), plus the k16 fragment ABI differs from the existing k8 layout. And the tile-divisibility constraint is **K%16==0 ∧ N%8==0 ∧ M%16==0**, not the design's "pad M to 16" alone.
4. **INC4's "1.1–1.4× decode" overstates the mechanism** — it wrongly credited hiding per-op sync bubbles that `g_fast` already eliminates; honest 32B decode is ~1.05–1.2× at best, possibly ~0×. The 32B-is-transfer-bound premise is **asserted, not measured** (the only measured streaming number, 4.6 s/tok 8B, is *dequant*-bound) and must be proven with `HX_PROF` (`g_pf_htod` vs `g_pf_dq`) before any decode claim.
5. **"8B packed-resident fits 8 GB" is true only with INC1.5 (packed head) AND Smax≤~2k.** INC1 alone at Smax=2048 is ~10.1 GiB — over budget. At Smax=4096 even the packed-head total (~10.66 GiB) is over budget. Context length is the gating knob; state the landed config as **8B-only, Smax≤~2k, packed-head-required.**
6. **No "parity" claim anywhere.** A hand-written non-tensor-core GEMV will not beat NVIDIA's tuned kernels; same-order-of-magnitude as llama.cpp is the optimistic stretch target, explicitly not a promise.

---

## MEASUREMENT - measure-before-wire (2026-06-19, alt-8b-nvfp4 substitute)

**Canonical qwen3-8b-v16.weights is GONE** from /home/legoa (displaced by the Alt project files). Substituted **alt-8b-nvfp4.weights** - a valid HXGW-v3 file that is actually a Qwen3-class 8B (NL=36, QK-norm, theta=1e6, NV=151669; --v3-smoke argmax=279 max_abs=21.953 nonfinite=0 = faithful). VRAM at measure time: ~4.5 GiB free (desktop active).

--v3-smoke + HX_PROF (HX_SCALECACHE=1), 252 uploads/forward, two warm runs:

| component | run1 | run2 | INC1 removes? |
|---|---|---|---|
| mmap_touch+HtoD | **2.14 s** | **4.29 s** | **YES (resident packed)** |
| cpu_scale_build | 1.92 s | 3.53 s | cached in steady decode -> ~0 |
| dequant_launch+sync | 0.56 s | 0.53 s | NO (INC1 keeps; INC2 targets it) |
| compact2D | 0.03 s | 0.02 s | - |

**FINDING (reshapes the plan): the packed-words HtoD DOMINATES (~2-4 s/forward); the dequant kernel is small (~0.5 s).** In steady-state decode the scale-build is cached away, so per-token cost ~= HtoD + dequant + compute, and HtoD is ~80-90% of the upload.

- **INC1 (resident packed 4-bit -> eliminate the HtoD) is the BIG decode+prefill win, and it is AUTONOMOUS (no kovc edit, fixpoint cdcf8673 untouched).** Plausibly decode ~4.6 -> ~1-2 s/tok (2-4x, host-side, fixpoint-safe).
- **INC2 (fixpoint-moving fused dequant-GEMV) targets only the ~0.5 s dequant -> MARGINAL** - confirms the adversarial 30-40%-near-null verdict. Do NOT move the fixpoint for it.
- **INC3 (tensor cores) NOT exercised** - the smoke is a 6-token prompt so GEMM is tiny; tensor cores only matter for long-prompt prefill/throughput (UNMEASURED).

**Honest v1.8 takeaway: the major speedup is the AUTONOMOUS INC1+INC1.5; v1.8 likely does NOT need to move the self-host fixpoint at all.** Caveats: noisy (HtoD 2.14 vs 4.29 = desktop contention); resident-packed (~5.4 GiB + 2.3 GiB FP32 head) needs the desktop idle (~7.6 GiB free) + INC1.5 + bounded context to FIT and be GPU-verified (at 4.5 GiB free it fit-checks -> falls back to streaming).
