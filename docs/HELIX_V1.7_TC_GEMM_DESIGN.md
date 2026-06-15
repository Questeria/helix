# Helix v1.7 — Tensor-Core GEMM (INCREMENT 3): MEASURED, SHELVED

## Verdict (2026-06-15): the TC-GEMM lever is not worth pursuing now — measured, not assumed.

INCREMENT 3 was chosen on the premise that, after INCREMENT 1 (GPU dequant), the 8B forward was
"GEMM-dominated". **That premise was an assumption, never measured. Direct measurement refutes it.**

### Measurement 1 — the existing TF32 kernel is SLOWER than the current f32 GEMM at projection dims
`gemm_tf32` (existing `tf32_matmul`, mma m16n8k8, "Path-2" = no SMEM/ldmatrix) vs `gemm_perf`
(current f32 `tiled_matmul`, SMEM 64×64/BK8/4×4), RTX 3070, M=Spad=64:

| dims (M K N)        | f32 SMEM-tiled (current) | TF32 Path-2 (existing) |
|---------------------|--------------------------|------------------------|
| 64×4096×4096  (q/o) | **0.522 ms** (4.11 TFLOPS) | 0.984 ms (2.18 TFLOPS) |
| 64×4096×12288 (g/u) | **1.248 ms** (5.16 TFLOPS) | 1.347 ms (4.78 TFLOPS) |
| 64×12288×4096 (down)| **1.519 ms** (4.24 TFLOPS) | 2.880 ms (2.24 TFLOPS) |

At M=64 the GEMM is memory-bound; the f32 kernel's SMEM staging beats the TF32 kernel's naked
global loads (Tensor Cores starved). Wiring the existing TF32 kernel in = a **slowdown**. A
fixpoint-moving A@Bᵀ TF32 emit in the SAME Path-2 style would also lose to f32. Beating f32 needs a
SMEM-tiled + ldmatrix TF32 emit — a large `kovc.hx` change — and even cuBLAS f32 only reaches
7–8 TFLOPS here (M=64 caps Tensor-Core utilisation).

### Measurement 2 — GEMMs are ~0.5% of the forward; the bottleneck is the per-layer weight re-stream
Summing the measured f32 GEMM times across one 8B forward: per layer q+k+v+o+gate+up+down
≈ 0.52+0.13+0.13+0.52+1.25+1.25+1.52 ≈ **5.3 ms/layer × 36 + lm_head ≈ ~0.2 s**. The forward is
**39.1 s** (INCREMENT 1). So **GEMM ≈ 0.5%**. Making GEMMs infinitely fast saves ~0.2 s of 39 s.

The remaining ~38.9 s is `v3_upload` running **every weight, every layer, every forward**
(gpt2_infer.c:1649-1657): mmap disk read of the ~4.5 GB packed 8B weights + HtoD + GPU dequant +
the single-threaded CPU effective-scale rebuild (`e4m3_decode(micro)*ts` over rows×kblk). That is
INCREMENT 2's exact target.

### Correction of the prior (wrong) claims
- Phase-1 doc said `tf32_matmul` computes `C=A@Bᵀ`. **Wrong** — the emitted PTX indexes B at `k*N+n`
  (B=[K,N]) ⇒ it's standard `C=A@B`, matching `mm_AB` (only llama use: attn@V, negligible), not the
  `mm_ABt` projections.
- "Remaining cost is GEMM-dominated." **Wrong** — measured ~0.5%.

The fixpoint `cdcf8673` was NOT touched. No `kovc.hx` edit was made. The TF32 Tensor-Core capability
(`emit_ptx_tf32_matmul_mma`) remains available for a future SMEM-tiled effort if GEMMs ever become the
bottleneck (i.e. AFTER the weight-stream cost is eliminated) — revisit then, not now.

## Pivot — INCREMENT 2 (the real SPEED lever, NO fixpoint move)
Keep the **packed 4-bit weights resident in VRAM** (8B packed ≈ 4.5 GB fits the 8 GB card; 32B at
23.5 GB stays on the v1.6 streaming path) + **cache the static effective per-16-block scales ONCE at
load** (they're identical every forward). Then each forward just launches the GPU dequant kernel from
resident packed weights + cached scales into the per-layer f32 scratch — **no mmap read, no HtoD, no
CPU scale rebuild** per forward. Worker-side C only; f32/v1.6 paths preserved (opt-in, HX_HOSTDEQ
fallback intact). Targets ~38 s of the 39 s. Also unblocks fast decode (kills the per-token re-stream).
