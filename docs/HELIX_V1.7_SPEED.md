# Helix v1.7 — SPEED

v1.6 proved an 8-billion-parameter model (and a 32B) runs on an 8 GB GPU via 4-bit NVFP4
quantization + per-layer streaming, with a reproducible receipt — **trust first, speed later**.
v1.7 is the speed pass. Every step here is **byte-identical** to the v1.6 result (same greedy
argmax, same `V3_UPLOAD_CHECK_PASS`), needs **no `kovc.hx` edit** (the 299-byte self-host fixpoint
`cdcf8673` is untouched), and is **opt-in** — `HX_HOSTDEQ=1` restores the exact v1.6 host path, so
v1.6's receipts still reproduce bit-for-bit.

## The Qwen3-8B forward, step by step (RTX 3070, 8 GB)

| Step | What changed | Result | Commit |
|------|--------------|--------|--------|
| v1.6 baseline | NVFP4→f32 dequant on the single-threaded **CPU** | **181.6 s** / forward | (v1.6) |
| INC1 | dequant the per-layer weights on the **GPU** (new `nvfp4_dequant_tiled` kernel, compiled by the cached kovc driver — no compiler edit) | 181.6 s → **43.6 s** (4.2×) | `c1851bf` |
| INC2a·1 | e4m3 decode via a 256-entry LUT (no `ldexpf` in the per-forward scale build) | 43.6 s → **39.1 s** | `9347eda` |
| INC2a·2 | dequant the untied **lm_head** (630 M elems) on the GPU too — it was still on the host (~22 s) | lm_head **22.2 s → 0.66 s** | `5c2aba1` |
| INC2b | `madvise(WILLNEED)` mmap prefetch + pinned-host DMA uploads | warm forward 16 s → **~12.6 s** | `1e212ea` |
| INC2c | cache the static effective per-16-block scales (skip the per-forward CPU rebuild) | per-token decode win | `f2e55a4` |

**Net:** the steady-state (warm) 8B forward is **~12.6 s**, down from the v1.6 host baseline of 181.6 s —
a **~14×** speedup — while staying byte-identical (next-token argmax = 279, layer dequant still `GPU==host`).

## What we did NOT do, and why (measured, not assumed)

- **INCREMENT 3 — Tensor-Core GEMM** was the "most ambitious" lever. We **measured** it before spending
  the attended fixpoint move: the GEMMs are only **~0.5 %** of the forward (~0.2 s of ~40 s), and the
  compiler's existing TF32 Tensor-Core kernel is actually **slower** than its f32 SMEM-tiled kernel at our
  shapes (M=64 is memory-bound, so SMEM reuse beats starved Tensor Cores). So a fixpoint-moving TF32
  A@Bᵀ emit would chase 0.5 % and lose to f32 unless also SMEM-tiled. **Shelved** — the fixpoint stays
  sacred. (`docs/HELIX_V1.7_TC_GEMM_DESIGN.md`.)
- **INCREMENT 4 — fused NVFP4-dequant GEMV for decode.** We wrote `gemv_abt_nvfp4` (reads packed 4-bit
  weights, dequantizes *inline* so f32 weights never materialize), verified it correct (an oracle check
  **and** a token-for-token decode match), then **measured** it: **~1.8× slower** than the f32 path
  (9.0 vs 4.9 s/token, 8B). The per-output **serial** unpack (one thread per output, `block=1`) loses to
  the f32 path's **parallel** dequant kernel + gemv — avoiding f32 materialization doesn't pay for the
  serial unpack. **Shelved the wiring**; kept the kernel as a verified building block. The lesson:
  benchmark a perf kernel against the incumbent *before* wiring, not just for correctness. (`a4b1f2e`.)

## Honest notes on the numbers

- **Warm vs cold.** ~12.6 s is the *warm* (page-cache hot) forward — the steady state for a serving
  process. A *cold* first load is dominated by reading the 7.3 GB weight file (a slow random mmap-fault
  pattern; `madvise(WILLNEED)` prefetches to soften it). The cold number is disk-bound and noisier.
- **Decode (v3 generation).** v1.6 only exercised prefill (the smoke + the receipts); the KV-cache
  *generate* path was both crashing (an out-of-bounds embedding read) and, once un-crashed, wrong (it
  skipped Qwen3's per-head QK-norm). Both **fixed** (`f2e55a4`) — the 8B now generates coherent,
  token-correct text. Decode is **~4.6 s/token** (after a `block=1`→`block=128` gemv-launch fix —
  byte-identical, just better warp occupancy — that shaved ~7 %). It is **not** gemv-bound: that 32×
  occupancy change barely moved it. It is **re-dequant-bound** — decode re-dequantizes all 36 layers of
  4-bit weights to f32 *every token* (the f32 weights are ~32 GB, far too big to keep resident). The fused
  dequant-in-gemv (INC4) avoids that materialization but is slower (serial per-output unpack). So decode
  is near its floor for this approach; the remaining lever is keeping the *packed* weights resident in VRAM
  (shaves the per-token HtoD, not the dequant) — a modest, VRAM-tight win, not yet built.
- **Method.** The breakdown above came from an env-gated profiler (`HX_PROF`) that splits each upload
  into mmap-touch+HtoD / CPU scale-build / dequant-launch — not from guessing. Two earlier assumptions
  (that the forward was GEMM-bound, and that the TF32 kernel was A@Bᵀ) were overturned by measuring.
- **Faithfulness is never traded for speed.** Every increment is gated on `V3_UPLOAD_CHECK_PASS`
  (byte-identical dequant) + unchanged argmax before it ships.
