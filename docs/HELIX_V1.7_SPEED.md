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

**Net:** the steady-state (warm) 8B forward is **~12.6 s**, down from the v1.6 host baseline of 181.6 s —
while staying byte-identical (next-token argmax = 279, layer dequant still `GPU==host`).

## What we did NOT do, and why (measured, not assumed)

- **INCREMENT 3 — Tensor-Core GEMM** was the "most ambitious" lever. We **measured** it before spending
  the attended fixpoint move: the GEMMs are only **~0.5 %** of the forward (~0.2 s of ~40 s), and the
  compiler's existing TF32 Tensor-Core kernel is actually **slower** than its f32 SMEM-tiled kernel at our
  shapes (M=64 is memory-bound, so SMEM reuse beats starved Tensor Cores). So a fixpoint-moving TF32
  A@Bᵀ emit would chase 0.5 % and lose to f32 unless also SMEM-tiled. **Shelved** — the fixpoint stays
  sacred. (`docs/HELIX_V1.7_TC_GEMM_DESIGN.md`.)

## Honest notes on the numbers

- **Warm vs cold.** ~12.6 s is the *warm* (page-cache hot) forward — the steady state for a serving
  process. A *cold* first load is dominated by reading the 7.3 GB weight file (a slow random mmap-fault
  pattern; `madvise(WILLNEED)` prefetches to soften it). The cold number is disk-bound and noisier.
- **Decode.** Within one process, decode steps reuse resident pages and run ~1 s/token; the per-token
  CPU effective-scale rebuild is the next lever (INC2c).
- **Method.** The breakdown above came from an env-gated profiler (`HX_PROF`) that splits each upload
  into mmap-touch+HtoD / CPU scale-build / dequant-launch — not from guessing. Two earlier assumptions
  (that the forward was GEMM-bound, and that the TF32 kernel was A@Bᵀ) were overturned by measuring.
- **Faithfulness is never traded for speed.** Every increment is gated on `V3_UPLOAD_CHECK_PASS`
  (byte-identical dequant) + unchanged argmax before it ships.
