# Helix v1.6 — P0 design output (gate PASS)

P0 design-investigate (read-only, 3 lenses + synthesis, Workflow wfllx727y). **Gate: PASS — GO for P1, NO `kovc.hx` edit.** Verified against the code, not asserted. The DoD ([HELIX_V1.6_DEFINITION_OF_DONE.md](HELIX_V1.6_DEFINITION_OF_DONE.md)) carries the corrected plan; this doc is the build-ready P1 change-list.

## Gate evidence (the load-bearing claim, verified)
- **4-bit dequant is an already-gated `@kernel`, not a language intrinsic** — `helixc/examples/nvfp4_dequant_kernel.hx:35-51` does the full E2M1 unpack (div-based nibble extract, f32 if-ladder over the 8 dyadic magnitudes, multiply by host-decoded f32 scale) using only existing constructs; its header (`:24-25`) states "NO kovc.hx edit … the self-host fixpoint stays cdcf8673."
- **The fixpoint gate is structurally insulated** — `scripts/gate_kovc.sh` compiles only `kovc.hx` through seed→K1→K2→K3→K4 and PTX-anchors solely to `vector_add` (`REF=$EX/vector_add_kernel.ref.ptx`, `:44`); the example kernels are gated as **additive** corpus refs, so adding/using a dequant `@kernel` cannot move `cdcf8673`.
- **`add_bias` is a proven existing host mechanism** — wired on the GPT-2 path at `gpt2_infer.c:799-800,822-823,833-834,841-842`; needs only 3 new call-sites in `forward_layer_llama`.
- **The #4 receipt spine is host-side, outside the fixpoint** — `cuda_launch.c:35` (gcc-built, not in the corpus).
- **No intrinsic is needed.** The *only* thing that would move the fixpoint is putting a **tiled f16 GEMM** on the critical path — explicitly deferred to a perf phase (v1.7), NOT P1.

## The mandatory dtype correction
- `nvfp4_dequant_kernel.hx:12` emits **f32 (NOT f16)**; `tiled_matmul_abt_kernel.hx:21` is **f32-only** (`fn tiled_matmul_abt(a: f32, b: f32, c: f32, …)`). So the composing path is **dequant → DENSE f32 buffer → existing f32 `tiled_matmul_abt`, unchanged.** Per-layer resident ≈ 0.8 GB f32 (lost the f16 saving — the honest P1 trade; perf → v1.7).
- The **tied head** can't be a drop-in f16 buffer (can't feed an f32 GEMM). **P1: keep the head 4-bit-packed in the mmap, dequant-per-step** into the f32 head GEMM (or tile-stream the f32 head) — avoids the ~3.1 GB fp32-resident OOM with no new kernel.

## P1 change-list (host-side C only)
1. **`gpt2_pack.c` → HXGW v3 (largest P1 item).** Today f32-only (bf16→f32 widen `:403-413`) and the Llama order table packs **no bias** (`build_order_llama :246-268`). Add additively: (a) NVFP4-packed payload = i32 nibble words (7 E2M1/word, low-nibble-first, `kk%112` packing) + per-16-block E4M3 micro-scales + per-tensor fp32 scale; (b) q/k/v_proj.bias entries in the order table; (c) header fields θ=1e6, vocab=152064, 48-layer dims (DM=5120, KVD=1024, DFF=13824). **The oracle MUST quantize identically to this packer** (DoD risk #6 — silent-wrong-text hazard).
2. **`gpt2_infer.c upload_layer_ll` (~:1352-1370): dequant-on-upload.** Today pure f32 `cuMemcpyHtoD` from mmap. Change to: mmap packed-i32 nibbles + scales, H2D the packed bytes + scales, launch `nvfp4_dequant` → **dense f32** device buffer per weight (feeds the existing f32 GEMM, no GEMM change). **Pad/tile for Qwen K=5120/13824** (5120%112=64 → `cuda_launch.c:1979` nvfp4 rejects non-conforming K; `:1886` mxfp4).
3. **`gpt2_infer.c forward_layer_llama` (~:880-932) + decode (~:985-1042): QKV bias.** It's bias-free (`:879`); q/k/v are 3 separate GEMMs (`:886/888/890`). Insert `add_bias(d_q,d_qb,Spad*DM,DM)` after :886, `add_bias(d_k,d_kb,Spad*KVD,KVD)` after :888, `add_bias(d_v,…)` after :890; mirror in decode after the gemv at :994/996/998. New device buffers `d_qb/d_kb/d_vb` uploaded in `upload_layer_ll`. (Mechanism identical to GPT-2's `add_bias` at :800 — no kovc edit.)
4. **`gpt2_infer.c setup_head` (~:1375-1390): fix the tied-head OOM.** `d_wte_pad` is f32-resident `A(NVpad*DM)` (`:1385`) ≈ 3.11 GB. P1 route (no kovc edit, no new kernel): keep the head **4-bit-packed**, dequant-per-step into the f32 head GEMM (prefill `tiled_matmul_abt` :970; decode `gpu_gemv_abt` :1047), OR tile-stream the f32 head in NVpad-row tiles.
5. **New device buffers/allocs** (`A()` allocator, `:196`): biases, packed-i32 weight buffers + scale buffers per layer, a dense-f32 dequant scratch per weight (~0.8 GB/layer, one-at-a-time).

## Receipt build-order (P2, not a P1 blocker)
- **Tier 2 + Tier 3 land FIRST on the existing ternary/int path UNCHANGED** — `receipt_emit` already runs `ternary_matmul`, exact, 6-NC-proven (`scripts/gpu_receipt_check.sh`). This is the safe base for the Tier-1-on-4bit obligation.
- **THEN** lift NVFP4 4-bit into the field (the per-16-block E4M3/E8M0 dyadic power-of-2 scale folded into integer code-space exactly so `|C|<p/2` ⇔ mod-p holds) — the one genuinely-new kernel + its data-independent NCs. **Do NOT claim 4-bit-exact-Freivalds until that is green.** Raise t to 3 for the 48-layer union bound (~2⁻⁸⁷); state the bound.
- Tier 3 = promote the gate's existing acceptance scaffold (argmax + max_abs<τ) into a recorded receipt field; τ empirically calibrated against the identically-quantized oracle, never hand-tuned.

## P1 first step
**Qwen3-8B VRAM-resident warm-up** (fits at 4-bit in VRAM, no streaming) to de-risk the importer + arch deltas + NVFP4 dequant before the streaming feat.

---

## P1a recon — model decision + Qwen3-8B (2026-06-14)

**Model decision (owner: "best + most ambitious"):** the **Qwen3 family** (the on-disk staged models; Qwen2.5 is NOT on disk and Qwen3 is *closer* to the bias-free importer). **Warm-up = Qwen3-8B** (on disk, VRAM-resident at 4-bit, no streaming). **Headline feat = Qwen3-32B** — the most ambitious model that still fits the existing approach: 32B fp16 ~65 GB (**8× the 8 GB card**); 4-bit ~20 GB (**2.7× usable VRAM → streaming hard-load-bearing**) yet fits the 31.8 GB host RAM, so **no new disk tier** is needed (70B+ would need one — out of scope). Qwen3-14B is the fallback. Both feat models need a ~20–28 GB download; the 8B warm-up needs none.

**Qwen3-8B config (verified `~/ingredients/Qwen_Qwen3-8B/config.json`, complete: 5 shards + index + tokenizer):** `Qwen3ForCausalLM`; DM=4096, NL=36, NH=32, NKV=8 (GQA×4), head_dim=128 (NH·128=4096=DM, KVD=8·128=1024), DF=12288, vocab=151936, rope_theta=1e6, **rms_norm_eps=1e-6**, **attention_bias=false**, **tie_word_embeddings=false**, bf16 source.

**Arch deltas vs the v1.4 SmolLM2/Llama path (the importer + worker work):**
1. **QK-norm (NEW):** per-head RMSNorm on q and k (`q_norm.weight`/`k_norm.weight` per layer, shape [head_dim=128]) before RoPE. Importer order table omits them → **ADD**; worker must apply per-head RMSNorm on q/k. (No QK-norm in v1.4.)
2. **Untied head (NEW):** `tie_word_embeddings=false` → a separate `lm_head.weight` tensor exists. The importer's Llama path assumes **tied** (reuses embedding) → **ADD `lm_head.weight`**; worker uses it (not the embedding) for logits.
3. **No QKV bias:** `attention_bias=false` → the planned `add_bias` wiring is **moot** (the bias-free path is correct). The Qwen2.5-bias assumption from P0 does not apply to Qwen3.
4. **eps 1e-6 vs the kernel's baked 1e-5:** the RMSNorm `@kernel` bakes eps=1e-5 and the host **fail-closes on any other eps** (`gpt2_infer.c:321`). Qwen3 needs 1e-6. Numerically negligible (mean(x²)≫eps) but for a *faithful-execution* claim the kernel should use the model's eps → either make the `@kernel` eps a runtime param (an `@kernel` example edit — new PTX ref + NCs, **no kovc.hx edit**) or relax the guard + document eps as within the Tier-3 envelope. Decide in the build.

**Runtime note (VRAM):** only ~2.7 GB free of 8 GB (~5.2 GB held by the Windows desktop/apps). A VRAM-resident Qwen3-8B-4bit warm-up needs ~5 GB → **free ~3–5 GB (close GPU apps / idle desktop) before the GPU run.** The importer is CPU-only.

**P1a sub-plan (gate-able increments):** (i) **Importer** (`gpt2_pack.c`): add the Qwen3 order entries (q_norm/k_norm/lm_head) + the NVFP4 4-bit packing (matching the dequant `@kernel` + a from-scratch oracle EXACTLY — the silent-wrong-text hazard). **Gateable CPU-only:** pack Qwen3-8B → verify the packed bytes vs the oracle quantization. (ii) **Worker** (`gpt2_infer.c`): per-head QK-norm + untied head + eps + dequant-on-upload. **Gateable:** run + compare to the 4-bit-aware oracle within the Tier-3 envelope.

---

## P1a importer spec — NVFP4 forward quantization (build-ready, 2026-06-14)

**Key finding:** v1.5 has the NVFP4 **dequant** + codecs (`cuda_launch.c` e2m1_encode/decode :190, e4m3_decode/encode :208/220, e8m0 :199) but **NO forward quantization** — the v1.5 `nvfp4` mode (`cuda_launch.c:1972`) uses *synthetic* E2M1 codes + a fixed `tensor_scale=1/3`, not a real f32->NVFP4 quantizer. v1.6's importer must **build** the forward quantizer, and the run-oracle must use the IDENTICAL algorithm (silent-wrong-text hazard).

**NVFP4 device format the importer MUST produce byte-for-byte** (from `nvfp4_dequant_kernel.hx`): per weight [rows x K], **K multiple of 112** (LCM(7,16)). Packed words `w[]`: 7 E2M1 codes/i32 word, base-16 **low-nibble-first**, `rows*(K/7)` words (7 not 8: 16^8-1 spills the sign bit). Effective scales `sc[]` (f32): one per 16-block, `rows*(K/16)`, = `e4m3_decode(micro)*fp32_tensor_scale` (host pre-collapsed; device only does `mag*sc`). Dequant: `out[r*K+col] = sign(code)*magf(code&7)*sc[r*(K/16)+col/16]`.

**Forward quantizer to BUILD (define exactly; oracle matches):** per tensor [rows x K]: (1) `tensor_amax=max|w|`; `fp32_tensor_scale = tensor_amax/(6.0*448.0)` (E2M1_max*E4M3_max), guard >0. (2) per 16-block: `block_amax`; `micro=e4m3_encode(block_amax/(fp32_tensor_scale*6.0))`; `eff=e4m3_decode(micro)*fp32_tensor_scale`. (3) per element: `code=e2m1_encode(w/eff)` (brute-force nearest over the 16 codes; eff==0 -> code 0).

**K-padding (Qwen3 dims need it):** Qwen3-8B weight K = DM=4096 (4096%112=64) and DF=12288 (12288%112=80) -> NOT multiples of 112. **Pad each weight's K to the next multiple of 112** (4096->4144, 12288->12320) with ZEROS (dequant to E2M1 code 0 = 0.0, benign in the matmul). Worker GEMM uses K_pad; extra zero columns don't affect the result.

**Importer changes (gpt2_pack.c, host-side, no kovc.hx edit):** (1) `build_order_llama` += per-layer `q_norm.weight`+`k_norm.weight` [head_dim=128] + untied `lm_head.weight` [NV x DM]; add a per-OrderEntry `quant` flag (NVFP4-pack the big matmuls q/k/v/o/gate/up/down; keep norms/embed f32; decide lm_head). (2) copy the e2m1/e4m3 codecs into gpt2_pack.c (host); add `nvfp4_quantize_tensor(f32* w,int rows,int K, int32* outW,float* outSc)` (the quantizer + K-padding). (3) the streaming loop branches: `quant` tensors -> BF16->f32 -> nvfp4_quantize -> write packed words + scales; f32 tensors -> existing path. (4) **HXGW v3 header** (VERSION=3): a per-tensor quant-descriptor table (offset, orig shape, K_pad, packed-vs-f32 flag) so `upload_layer_ll` knows the dequant-on-upload layout.

**Verification (gate-able CPU-only, no GPU):** pack Qwen3-8B (or a slice) -> host-dequant each packed tensor (e2m1/e4m3) vs the original f32 (within the expected NVFP4 quant error) + round-trip the packed bytes through the device kernel (the gpu_nvfp4_check pattern) -> match. NCs: nibble-flip, scale-flip, wrong K_pad. THEN the worker (next phase) consumes HXGW v3.

---

## P1a — NVFP4 quantizer verified on REAL Qwen3 weights (2026-06-14)

Ran `nvfp4_quantize` on real Qwen3-8B layer-0 weights (q_proj/o_proj [4096x4096], down_proj [4096x12288], both K-paddings): consistent **rmse/rms_w ≈ 9.5%** (the inherent E2M1 4-bit granularity; max_rel=1 only at near-zero weights = small absolute error), **~4.8x compression** vs f32. The quantizer + K-padding work on real data, not just synthetic.

**COMPRESSION FINDING (revises the fit math):** measured footprint is **4.8x** (0.821 B/param), NOT 8x -- the v1.5 format stores the EFFECTIVE scale as **f32 per 16-block** (0.25 B/param, so the device that can't decode E4M3 reads it directly) + 7-codes-per-i32-word (0.571 B/param). **RAM impact on the 32B feat:** Qwen3-32B NVFP4 at 4.8x = ~27 GB host RAM + OS ~= the 31.8 GB ceiling -> **RAM-marginal**. Mitigation: the importer stores the COMPACT form (E4M3 micro = 1 B/16-block = 0.0625 B/param -> ~6.3x, 32B ~= 21 GB) + the worker decodes E4M3->f32 effective scale at upload-time (host, before H2D). 14B (~12 GB) is comfortable either way; the 8B warm-up (~6.6 GB) trivially fits. **DECISION:** use the compact E4M3 storage for the streamed feat (decode at upload); the device dequant input format is unchanged.

**Quant quality:** 9.5% per-weight RMS is normal for 4-bit; whether the model stays argmax-correct (the Tier-3 envelope) is the model-run test. The amax-per-tensor scale could be refined (percentile/outlier-aware) if the envelope proves too loose.
