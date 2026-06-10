# Fable demo-fast run notes (2026-06-10, branch fable/demo-fast)

One lesson per entry; why it mattered.

1. **The M%64 tile constraint, not fp32, was the real decode blocker.** The gated tiled GEMMs
   need M%64==0; a decode step has M=1. Padding to 64 wastes 64x the FLOPs and still scales
   with context. Three tiny GEMV-form kernels (one thread per output, serial K-loop — the
   gpu_rmsnorm idiom) remove the constraint entirely: gpu_gemv_abt covers every decode GEMM
   INCLUDING attention scores vs the cached K; gpu_gemv_ab covers probs x cached-V;
   gpu_softmax_row is the causal kernel minus the mask (causality lives in WHAT is cached).

2. **"Mathematically identical" is a TOKEN-level claim, not a bit-level one.** The GEMV sums
   serially; the tiled kernel sums in 8-wide tiles — logits differ at ~1e-4 (observed 9.06e-5
   at K=960 in the kernel gate). The honest, gateable invariant is SAME IDS: the A/B leg runs
   the full-reforward and the KV path on the same prompt and requires byte-identical id files,
   PLUS token-for-token vs the independent oracle. If a near-tie token ever flips, the gate
   catches it and the answer is "fix or don't ship", never "close enough".

3. **Prefill IS the cache-fill.** The existing (gated) full-forward already computes every
   roped K head-slab — capturing d_Kh/d_Vh per kv-head into the cache during the normal
   per-head loop costs two DtoD copies per (layer, kv-head) and zero new math. Pad rows land
   beyond kv_len and get overwritten by later appends; decode only reads [0, kv_len).

4. **RoPE-at-position is a pointer offset, not a new kernel.** The rope kernel pairs buffer
   row r with table row r; decode's single row at position t just passes d_cos + t*half as
   the table base. The kernel stays byte-identical to the gated one.

5. **Per-op cuCtxSynchronize is pure instrumentation cost.** All launches are on the default
   stream (ordered); dropping host syncs cannot change results. Fast mode = skip syncs + op
   emits; one sync before the D2H logits read. Same kernels, same ids — the A/B + perf gates
   prove it rather than assert it.

6. **Resident weights turn decode from H2D-bound to launch-bound.** Streaming uploads
   ~1.4 GB per TOKEN for the 360M (45 MB/layer x 32). Keeping all layers on-device (1.45 GB —
   fine for small models, policy-excluded for XL with 3 workers sharing 8 GB) makes
   upload_layer a pointer swap.

7. **Heredoc patch scripts must WRITE AFTER EVERY STEP.** Three times this session a
   multi-step python patch died on a later anchor and silently discarded the EARLIER
   (passing) edits. Atomic per-edit writes + a syntax check after each batch is the only
   sane pattern at this file size. (Plus the known traps: heredoc strips one backslash
   level; several files are CRLF; anchors with backslashes or trailing whitespace lie.)

8. **Wall-clock perf varies run-to-run -- persist the evidence and report the clean run.** Three
   measurements of the same configs: 46x (first session), 14x (mid-session, three serve workers
   still shutting down), 39x (clean-GPU, persisted: scripts/_perf_decode_clean.log -- canonical:
   0.48 -> 3.92 -> 18.69 tok/s baseline -> KV -> KV+fast). The fair claim is "roughly 40x on this
   GPU, run-to-run variance real"; the INVARIANT claim is ids-identical, which never varied.

9. **The verifier was right that uncommitted /tmp evidence is no evidence.** Every gate verdict
   now has a persisted log under scripts/_: _gemv_gl0.log, _gate_360i_kv2.log (clean-env),
   _lss_final.log (incl. the fast A/B leg), _perf_decode_clean.log.
