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
**Qwen2.5-7B VRAM-resident warm-up** (fits at 4-bit in VRAM, no streaming) to de-risk the importer + QKV-bias + dequant before the 14B streaming integration.
