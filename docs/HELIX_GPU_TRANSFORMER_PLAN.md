# Helix GPU transformer-op build plan (P5/P6 → capstone)

**Date:** 2026-06-01 · From a code-architect pass over the post-matmul emitter. Full blueprint: architect task a113efb0. Drives the loop after the naive matmul (commit a38a1b6).

## Central decisions
- **ONE THREAD PER ROW** for every reduction (softmax, layernorm, attention scores): each thread owns a full row and reduces serially over the hidden dim (64–256). This **eliminates any need for `.shared` / `bar.sync`** on the correctness track. The capstone's "within 2% of PyTorch" bar is **correctness, not throughput** — even a 100× slower naive kernel passes if the loss matches. A tiled/shared-memory GEMM (roadmap Step C) is a **separate perf track**, only needed if the naive GPU matmul is too slow at the small capstone size.
- **Autodiff on GPU (DoD #4):** hand-written backward kernels mirroring the CPU backward fns in `nn.hx` (`dense_layer_f32_grad_*`, `softmax_ce_grad_f32`, `relu_layer_f32_backward`, …). This is what PyTorch does internally — not `grad`-over-tensors. Verify each GPU backward vs its CPU oracle.
- **1D launch only** (the emitter has `.x` thread/block indices only): flatten 2D tensors to a 1D thread space; `gid = block_idx()*block_dim()+thread_idx()`.

## New PTX emitter additions (Phase 1, all in kovc.hx, gated)
- `emit_ptx_f32_const(...)` — emit `mov.f32 %fN, 0fHHHHHHHH;` (the bootstrap literal parser is integer-only, so f32 constants need hex-bit emission).
- `__gpu_exp(x)` → `mul.f32 t, x, 0f3FB8AA3B (log2e=1.4426950); ex2.approx.f32 r, t` — gives e^x = 2^(x·log2e). SM86 hardware, error ~2^-22, well within 2%.
- `__gpu_rsqrt(x)` → `rsqrt.approx.f32 r, x` — 1/sqrt(x), for layernorm `inv_std`.
- `__gpu_log(x)` → `lg2.approx.f32 t, x; mul.f32 r, t, 0f3F317218 (ln2=0.6931472)` — for cross-entropy `-log(p)`.
- Each is a new `emit_ptx_call` arm (kovc.hx ~11396) + a name-matcher (mirror `ptx_name_is_thread_idx`) + a one-arg helper; add before the final `else { 0 - 1 }`.

## BUG to fix first (flagged by the architect)
`emit_ptx_cmp` (kovc.hx ~10565) emits `setp.<cc>.s32` **unconditionally**. A FLOAT compare (relu `x > 0.0`) then reinterprets the f32 bit pattern as a signed int → **wrong for negative floats**. No committed kernel hit this yet (the matmul's `t < kk` is i32). Fix: when both operands are f32 (`vtab+55==1`), emit the `.f32` suffix.

## Op build order (each fwd+bwd, gated, validated vs CPU oracle)
matmul ✅ → **exp-foundation + setp.f32 fix** → relu (elementwise) → **softmax (row)** [the recommended next full op; needs exp] → cross-entropy grad → layernorm (row, +affine; needs rsqrt) → GELU (tanh via exp) → dense/GEMM backward (grad_w/b/x) → embedding lookup → scaled-dot-product → multi-head attention (composed) → Adam update → **training loop + PyTorch oracle, iterate to within 2%**.

CPU oracles: `helixc/stdlib/nn.hx` (softmax_layer:789, layer_norm_f32:666, gelu_layer:594, adam_f32_step:285, softmax_ce_grad_f32:885, dense_layer_f32_grad_*:410-472), `transcendentals.hx` (__exp:34, __log_stable:123, __sqrt:156, __gelu).

## Build phases
1. **Emitter:** f32_const + exp + rsqrt + log + the setp.f32 fix → probe each (test_exp vs e, test_rsqrt, test_log, a relu float-compare) → fast-gate → full K2==K3 fixpoint → commit.
2. **Elementwise:** relu, GELU, Adam (each fwd+bwd) vs CPU oracle.
3. **Row-reductions:** softmax, layernorm, ce-grad (one-thread-per-row, serial).
4. **GEMM-bwd + embedding.**
5. **Attention (composed) + the 2-layer training loop + PyTorch oracle → within 2% = capstone.**

## Notes
- vtab cap is 12 vars/kernel — softmax ~8, layernorm ~6, adam ~6: all safe.
- Adam: use the SAME bias-correction variant in both Helix and the oracle (uncorrected is simplest, avoids needing `pow`).
- Each new op may surface emitter gaps (the matmul surfaced 3) — bisect with the persistent driver (`cat /tmp/out.ptx`), fix gated. Trust spine + ELF self-host stay untouched (PTX-path only).
