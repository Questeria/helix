# Helix Model-Import Plan — Verified Inference of a Pretrained Open Model

**Status:** RESEARCH + PLAN (not started). Author: autonomous research pass, 2026-06-08.
**Goal:** Make Helix **run a pretrained open-source model's inference, unchanged weights, matching a trusted reference** — to prove Helix drops into existing AI systems *without retraining* and turns them into a **verifiable, reproducible, from-raw-auditable** execution. First target: **GPT-2 (124M)**.

> This is the concrete vehicle for the investor narrative *"Helix is the trust/verifiability layer for AI"*: instead of a toy we trained, we take **the model you already know**, import its public weights with zero retraining, and run it on a stack you can rebuild from 299 hand-typed bytes and reproduce bit-for-bit.

---

## 1. Honest framing of "integration"

Helix is **not** a PyTorch plugin. The honest, defensible claim is a **bring-your-weights verified execution layer**: Helix re-expresses a model's *forward pass* and consumes its *existing trained weights* with no retraining. So the pitch is *"your model + your weights, now running on a substrate you can verify from the first byte and reproduce exactly."* That is still a strong adoption story — and it is true.

---

## 2. Feasibility verdict: **GO** (GPT-2), bounded gaps

GPT-2's architecture is **nearly identical to the op set Helix already runs in the v1.0 capstone** (pre-LayerNorm + causal MHA + tanh-GELU MLP + tied embeddings). Grounded inventory (read of `helixc/examples/*_kernel.hx`, `helixc/bootstrap/kovc.hx`, `helixc/stdlib/{nn,tensor,transcendentals}.hx`, `helixc/runtime/{train_transformer,cuda_launch}.c`) shows **~95% of the forward pass already exists and runs on real hardware** (the capstone trained end-to-end on the RTX 3070 → `CAPSTONE_AUDIT_PASS`). The missing pieces are small and bounded.

### 2.1 GPT-2 forward op → Helix status → action

| GPT-2 forward op | Helix today | Action |
|---|---|---|
| Q/K/V/MLP/LM-head matmul | **EXISTS** — `naive_matmul` (capstone-proven on HW), `tiled_matmul`/`__tiled_matmul_smem`, `tf32_matmul` | reuse `naive_matmul` first (no divisibility constraint); `tiled` later for speed (pad vocab 50257→50304) |
| QKᵀ scores | **EXISTS** — `gpu_qkt` (scale baked per-d) or `matmul_abt` + `gpu_scale_rt` (runtime scale) | use unscaled `matmul_abt` + runtime `1/√64` scale (clean, no per-d literal) |
| Causal-masked softmax | **PARTIAL** — `gpu_softmax`/`__softmax_blockred` exist but are **un-masked**; `flash_attention` loops all keys with no `j≤i` guard | **ADD** causal mask: `scores[i,j]=-inf for j>i` before softmax (small kernel or a predicate in the existing path) |
| attention·V | **EXISTS** — `naive_matmul`, or fused in `flash_attention` | reuse |
| LayerNorm (γ,β) pre-LN ×N + final | **EXISTS on GPU** (`gpu_layernorm`/`_fwd_save` apply full affine) — **but NO epsilon** | **ADD** `eps=1e-5` before the `rsqrt` (one add). CPU `layer_norm_f32` has eps but no affine → add affine if doing CPU path |
| GELU (tanh-approx) | **EXISTS + matches GPT-2** — `gpu_gelu` and CPU `__gelu` both use the Hendrycks tanh form `0.5x(1+tanh(√(2/π)(x+0.044715x³)))` | reuse as-is ✅ (see GELU gotcha §5) |
| residual add | **EXISTS** — `vector_add` | reuse |
| bias add (Q/K/V/MLP/head) | **PARTIAL** — CPU fused; GPU has only elementwise `vector_add`/const `gpu_affine` | **ADD** a small GPU row-broadcast bias-add (or fold into GEMM epilogue; or add bias host-side for v1) |
| **token + positional embedding gather** | **MISSING** — no gather/index-select kernel anywhere; GPU codegen has never declared an i32-array param (corpus carries ints as f32) | **ADD** — but for v1 do the gather **host-side** (CPU gather `wte[id]+wpe[pos]` into a host buffer → `cuMemcpyHtoD`). Trivial, **no new kernel**. GPU gather kernel is a later optimization (would exercise `emit_ptx_index_load` + an i32-array param) |
| multi-head attention | **PARTIAL** — capstone is **single-head** (D = full head dim) | **ADD** a host-side head-split/merge loop over 12 heads of 64, reusing the existing matmul/softmax kernels. No new kernel |
| final logits matmul to 50257 | **EXISTS** (GEMM) | reuse `naive_matmul`; logits = `hidden @ wteᵀ` (tied head) |

### 2.2 The bounded new-work list
1. **Causal mask** (low–med) — the one real attention gap.
2. **LayerNorm epsilon on GPU** (trivial).
3. **Host-side embedding gather** for v1 (trivial); GPU gather kernel later (med).
4. **Multi-head split/merge** host loop (med, no new kernel).
5. **Bias row-broadcast** on GPU (low) or host-side for v1.
6. **Weight converter** (host-side, fenced — §4).
7. **BPE tokenizer** (host-side for v1 — §4).
8. **Autoregressive generation loop** (host orchestration around `forward_full`).

None of these is research; they are kernels-compositions + host glue. The only genuinely new *kernel* is the (optional, later) GPU embedding gather; v1 avoids it entirely with a host-side gather.

---

## 3. The capstone is the skeleton we extend

`helixc/runtime/train_transformer.c` already implements the **entire reusable inference path**:
- `forward_full()` / `forward_layer()` — the ordered per-layer kernel-launch sequence (LN→QKV→QKᵀ→softmax→·V→proj→residual→LN→MLP-GELU→residual), then final LN + LM head + logits D2H. **Keep verbatim**, generalize dims.
- The CUDA Driver-API launch infra in `cuda_launch.c` / `train_transformer.c` (`cuModuleLoadData` of the seed-minted PTX, `cuModuleGetFunction`, `cuMemAlloc`, `cuMemcpyHtoD/DtoH`, `cuLaunchKernel`+sync). **Keep verbatim.**
- The **flat little-endian fp32 weight file** (`init_weights.bin`), written/read in a fixed tensor order, mirrored by the numpy oracle. **This is the importer's output format.**
- **Drop** all backward/optimizer code (`backward_*`, `adam_step`, gradient buffers, the bwd kernels) — not needed for inference.

What we ADD (host code + a converter, per §2.2): real GPT-2 dims (`NL=12, d_model=768, n_head=12, V=50257, S=1024, H=3072`), multi-head, causal mask, LN eps, host embedding gather, generation loop.

---

## 4. The weight importer + tokenizer (fenced, host-side data-prep)

> **Honesty fence:** the converter and tokenizer are *data-prep / I/O*, exactly like the fenced numpy audit oracle — **not** part of the from-raw compute trust chain. State this plainly; it does not weaken the trust claim (the *compute* is still from-raw).

**Importer (HF `openai-community/gpt2` → Helix flat fp32):**
- Source: `model.safetensors` (~548 MB, **already F32** — no dequant). Tensor names confirmed: `wte.weight [50257,768]`, `wpe.weight [1024,768]`, per-layer `h.{i}.ln_1.{weight,bias}`, `h.{i}.attn.c_attn.{weight[768,2304],bias}`, `h.{i}.attn.c_proj.{weight[768,768],bias}`, `h.{i}.ln_2.*`, `h.{i}.mlp.c_fc.{weight[768,3072],bias}`, `h.{i}.mlp.c_proj.{weight[3072,768],bias}`, `ln_f.{weight,bias}`.
- **Transpose exactly the 4 Conv1D weights** (`attn.c_attn`, `attn.c_proj`, `mlp.c_fc`, `mlp.c_proj`) — HF GPT-2 stores Conv1D as `[in,out]`; an `x@W` engine wants the matching layout. Embeddings, biases, LayerNorm γ/β are **not** transposed. (This is exactly nanoGPT's `transposed=[...]` list.)
- **Tie** `wte` to the LM head: logits = `hidden @ wteᵀ` (no separate head tensor).
- **Split `c_attn`** `[768,2304]` into Q|K|V chunks of 768, reshaped to 12×64.
- Write tensors into Helix's flat fp32 order (capstone layout, extended with `wte`/`wpe`).

**Tokenizer:** GPT-2 byte-level BPE (`vocab.json` + `merges.txt`): UTF-8 bytes → 256 base tokens → greedy merges by rank → 50257 vocab. v1 host-side (reference: `karpathy/minbpe`); in-Helix (using `string.hx`) is a later "no host deps" milestone.

---

## 5. The parity gate (the honesty discipline, carried over from the capstone)

Every phase is gated on **matching a trusted reference within fp32 tolerance** — the same oracle-parity discipline as `capstone_audit.sh`, now against a pretrained model. Never fake parity; if logits drift, find the op.

- **Reference:** HF `transformers` `GPT2LMHeadModel` (with `activation_function="gelu_new"`). Use **nanoGPT** for the weight-name→transpose map, but **match logits against HF**, not nanoGPT's forward.
- **⚠️ GELU gotcha (the #1 silent-drift risk):** GPT-2 was trained with **tanh-approx GELU** (`gelu_new`). Helix uses tanh-approx ✅ — but **nanoGPT's reference forward uses exact-erf `nn.GELU()`**, so do not treat nanoGPT's logits as ground truth; match HF's `gelu_new` path.
- **Gate metric:** worst-case relative logit diff on a fixed prompt < tolerance; under **greedy** decoding the generated token sequence should match HF token-for-token.

---

## 6. Two-path strategy

- **CPU path = the purest trust artifact.** Helix's CPU path is all-the-way-down from raw binary with **no `ptxas` trusted boundary** and full determinism. Running GPT-2 forward here = *"this model ran on a stack with no trusted prebuilt component above 299 bytes, bit-reproducibly."* CPU gaps: LayerNorm affine, causal mask, gather (trivial array index). **Perf unknown — needs a spike to measure; likely seconds–minutes/token unoptimized for 124M.** Fine for a short trust demo.
- **GPU path = the speed/scale story (to PTX).** Reuse the capstone-proven kernels. Honest `ptxas`-trusted boundary below PTX. The capstone's forward kernels are proven on the RTX 3070; GPT-2 forward reuses them. *(Note: the 2026-05-28 GPU-status memo flagged "tile ops are stubs / no real-HW execution"; that predates and is superseded for the capstone forward kernels by `CAPSTONE_AUDIT_PASS`. Confirm the specific GEMM path used — `naive_matmul` is proven; `tiled_matmul` should be re-confirmed on the larger GPT-2 dims.)*

Recommendation: build the **GPU path first** (reuses the most, gives a snappy demo), then the **CPU path** as the airtight trust capstone.

---

## 7. Phased plan

**Phase 0 — Spike / de-risk (days).** Write the importer; reproduce **one** GPT-2 block forward with real 768-dim/12-head weights in the C+kernel harness; match HF's per-tensor intermediates (post-ln_1, post-attn, post-mlp) within tol. De-risks GELU/eps/multi-head/causal-mask/transpose before building the full stack. **Needs:** download `openai-community/gpt2` safetensors + a one-off HF reference dump.

**Phase 1 — Full forward, GPU (extend the capstone).** Generalize `forward_full` to GPT-2 dims; add multi-head, causal mask, LN eps, host embedding gather, bias adds. **Gate:** HF-logits parity on a fixed prompt.

**Phase 2 — Generation + tokenizer.** Host BPE + autoregressive greedy/sampled decode. **Gate:** token-for-token match vs HF greedy; coherent text.

**Phase 3 — CPU path.** Same forward on the from-raw CPU path (add CPU LN affine + mask + gather). **Gate:** logits parity. The trust artifact.

**Phase 4 — Trust wrapper + demo.** Wrap `reproduce_trust.sh` (rebuild from 299 bytes) + the GPT-2 inference + a signed attestation into the 2-minute investor runbook: *"GPT-2, imported unchanged, generating text on a compiler rebuilt from 299 bytes, output-matched to the reference, bit-reproducible."*

**Phase 5 (optional) — scale + modern arch.** `gpt2-xl` (1.5B) = *same code, billion params, zero new ops* (fits 8 GB fp32 ~6 GB). Then a modern Apache-2.0 Llama-arch model (Qwen2.5-0.5B or **OLMo-2-1B**, the most-open weights+data) — budget the **4 new ops**: RMSNorm, RoPE, SwiGLU/SiLU, GQA.

---

## 8. Honest caveats / risks
- **fp32 only** (no quant/dequant kernels yet) → download full-precision weights, skip GGUF; practical ceiling ~1.5B on the 8 GB RTX 3070 Laptop.
- **GELU variant** is the most likely silent-drift source — match HF `gelu_new`, not nanoGPT erf.
- **Inference perf** will be a fraction of llama.cpp — irrelevant to the trust pitch; never claim otherwise.
- **Converter + tokenizer are fenced host glue** — honest that they're data-prep, not from-raw compute-trust.
- **CPU-path perf for 124M is unmeasured** — Phase 0/3 must measure it; may need a smaller demo generation or the GPU path for a snappy live demo.
- **GPU real-HW execution** for the exact GEMM path on GPT-2 dims must be reconfirmed (capstone proves the forward kernels run; tiled GEMM on 768/3072/50304 to be checked).
- **Per-tensor safetensors shapes** should be dumped from the real header for byte-certainty (derived here from arch + nanoGPT).

---

## 9. Model options (recap)

| Model | Params | Arch | License | fp32 size | Fits 8 GB | New ops | Role |
|---|---|---|---|---|---|---|---|
| **GPT-2 124M** | 124M | GPT-2 | MIT | ~0.5 GB | ✅ (CPU too) | ~causal-mask + glue | **start / spike** |
| GPT-2-XL | 1.5B | GPT-2 | MIT | ~6 GB | ✅ | 0 | **hero "scale" flex** |
| Qwen2.5-0.5B | 0.5B | Llama | Apache-2.0 | ~2 GB | ✅ | +4 | modern v2 |
| OLMo-2-1B | ~1B | Llama | Apache-2.0 (open data) | ~4 GB | ✅ | +4 | "most-open model on the most-verifiable stack" |
| TinyLlama-1.1B | 1.1B | Llama | Apache-2.0 | ~4.4 GB | ✅ | +4 | recognizable |
| Llama-3.2-1B | 1B | Llama | Meta community (NOT fully open) | ~4 GB | ✅ | +4 | avoid for the open-stack story |

---

## 10. Immediate next steps
1. **Download** `openai-community/gpt2` (safetensors, ~548 MB) + record an HF reference (logits + per-tensor intermediates on a fixed prompt) as the parity oracle.
2. **Write the importer** (HF → Helix flat fp32; the 4 transposes + wte tying + c_attn split).
3. **Phase-0 single-block parity** in the C+kernel harness.
4. Decide hero model (GPT-2-XL same-code vs a modern Llama-arch with the 4 ops).

Reference files (real): `helixc/runtime/train_transformer.c`, `helixc/runtime/cuda_launch.c`, `scripts/capstone_audit.sh`, `verification/oracle/oracle_train.py`, `helixc/examples/{naive_matmul,gpu_qkt,gpu_softmax,gpu_gelu,gpu_layernorm_fwd_save,vector_add}_kernel.hx`, `helixc/bootstrap/kovc.hx` (PTX emitter), `helixc/stdlib/{nn,tensor,transcendentals}.hx`.
