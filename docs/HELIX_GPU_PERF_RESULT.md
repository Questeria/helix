# Helix GPU Performance Result (Track 2)

Result document for `GPU_PERF_PASS` (charter `docs/HELIX_COMPLETION.md` §1.2). This file
records the measured perf tiers as they land. **G1 is the first tier and is GREEN.**

Reference box: **NVIDIA GeForce RTX 3070 Laptop GPU (sm_86)**, max SM clock 2100 MHz.
Toolchain: ptxas/CUDA driver 12.0, cuBLAS 12.8.3.14, libcuda via WSL2.
Measured 2026-06-02.

---

## G1 — SMEM-tiled f32 GEMM (bar.sync) — **PASS**

**Gate (§1.2 / §6 G1):** the SMEM-tiled f32 GEMM is CORRECT vs the CPU oracle AND vs a
fenced cuBLAS oracle (cell-by-cell within tol), AND **>= 3 TFLOP/s** on the RTX 3070
Laptop, AND `.shared`/`bar.sync` provable in the emitted PTX OUTPUT (>= ~30% cuBLAS f32).

| metric | value |
|---|---|
| **kovc median TFLOP/s @ 2048^3** | **4.56 TFLOP/s** (min 2.87 / med 3.77 ms / max 5.32 ms over 50 timed kernel-only launches) |
| cuBLAS median TFLOP/s @ 2048^3 (true-f32, pedantic) | 8.15 TFLOP/s (med ~2.10 ms) |
| **kovc / cuBLAS ratio** | **~56%** (gate floor ~30%) |
| G1 bar (>= 3 TFLOP/s) | **PASS** (4.56 >= 3.0, ~1.5x margin) |
| correctness vs CPU oracle | 0 bad cells, 64^3..512^3 (integer-exact ==) |
| correctness vs cuBLAS oracle | 0 bad cells, 64^3..2048^3 (tol 1e-3 f32; integer-exact so really 0) |
| cuBLAS oracle trusted (cuBLAS == CPU) | 0 bad cells, 64^3..512^3 |
| PTX provenance (emitted OUTPUT) | `.shared`, `bar.sync 0`, `ld/st.shared.f32`, `fma.rn.f32`, `.target sm_86` all PRESENT |
| ptxas (sm_86) | 56 regs, 4096 B smem, **0 spills** |
| neg-control A (comparator teeth) | mutate one C cell -> FAILS (correct) |
| neg-control B (barriers load-bearing) | strip bar.sync from emitted PTX -> mis-computes/FAILS (correct) |

**Tile params (unchanged from M1 correctness):** BM=BN=64, BK=8, TM=TN=4, threadblock
16x16=256, grid=(N/BN, M/BM). The 64x64 register-blocked tile already clears the G1 bar,
so **no `kovc.hx` change was needed for G1** — the emitter is byte-identical to the M1
commit (`cef380a`), and the freshly-emitted PTX matches the committed reference
`helixc/examples/tiled_matmul_kernel.ref.ptx` byte-for-byte. G1 therefore required no
self-host-fixpoint re-mint (the per-milestone full gate is required only when `kovc.hx`
changes; only the host-side `cuda_launch.c` + the `scripts/gpu_perf_corpus.sh`
orchestrator changed, both OUTSIDE the fixpoint).

**Throughput across sizes (kovc TFLOP/s, kovc/cuBLAS ratio):** 1024^3 ~5.2 (65%),
2048^3 ~4.56 (56%), 4096^3 ~4.56 (57%). Every size clears 3 TFLOP/s with margin.

### Methodology (honest)

- **Fenced cuBLAS oracle.** The cuBLAS call is host-side in `cuda_launch.c`'s `gemm_perf`
  mode — a trusted-tool verification oracle exactly like the numpy/CPU oracles, NEVER in
  the Helix self-host path. Row-major `C=A*B` is computed via the column-major identity
  `C^T = B^T*A^T`: `cublasSgemm(N, N, N, M, K, B, ldN, A, ldK, C, ldN)` with swapped
  operands. cuBLAS is forced to `CUBLAS_PEDANTIC_MATH` so it is a **true-f32** oracle (no
  TF32 tensor-core contamination) — the apples-to-apples reference for kovc's true-f32
  `fma.rn.f32` kernel; the perf ratio is therefore f32-vs-f32 (a TF32-cuBLAS reference
  would be ~2x faster and unfairly deflate the %). Link: `-lcublas -L/usr/local/cuda/lib64`.
- **Chain of trust.** CPU oracle <- cuBLAS oracle <- kovc kernel. The cuBLAS oracle is
  validated cell-by-cell vs the CPU oracle FIRST (at 64^3..512^3) so it is trusted before
  it judges the kovc kernel. Integer inputs (a[i]=i%7, b[i]=i%5) keep every partial sum
  < 2^24, so all three are integer-EXACT in f32 (`==` for CPU comparisons). At >= ~840^3
  the O(N^3) single-thread CPU triple-loop is skipped (it is minutes at 2048^3) and
  large-N correctness rests on the GPU-fast kovc-vs-cuBLAS cell compare (O(M*N), 4.2M
  cells at 2048^2) plus the already-trusted oracle. This skip is REPORTED in the output
  (`[CPU-oracle:skipped(large-N; kovc-vs-cuBLAS only)]`), not hidden.
- **Timing.** cuEvent (Driver-API) kernel-only timing: 5 warmup + 50 timed launches,
  sorted to report min/median/max ms (the laptop throttles — the median is the robust
  figure; a single throttle spike inflates max but not median). TFLOP/s = 2*M*N*K /
  median_seconds. Both kovc and (pedantic) cuBLAS timed under the identical protocol.
- **Provenance.** The PTX is emitted by the kovc PTX driver (`_kovc_ptx_driver.bin`,
  re-minted from the raw-binary seed), NOT by `nvcc`. The instruction-class greps run on
  the emitted `.ptx` OUTPUT, never on source comments (same rule as M0's sm_86 check).
- **Negative controls (two, independent).** (A) The comparator has teeth: perturbing one
  output cell pre-compare must FAIL — proves the cell-by-cell check is not vacuous. (B)
  The barriers are load-bearing: deleting every `bar.sync` line from the emitted PTX
  (still ptxas-valid) must mis-compute — proves `.shared`/`bar.sync` carry real semantics
  (without them the cooperative SMEM staging races and the result is wrong). Both trip.

### Reproduce from scratch

```
wsl.exe bash -c "bash /mnt/c/Projects/Kovostov-Native/scripts/gpu_perf_corpus.sh"
# -> emits kovc PTX, provenance greps, ptxas, cuBLAS-oracle correctness corpus
#    (64^3..2048^3), the >=3 TFLOP/s G1 gate at 2048^3, and both negative controls.
# Verdict line: GPU_PERF_G1_PASS
```

### What G1 proves / does not prove

**Proves:** kovc's own PTX codegen emits a correct, cooperative SMEM-tiled f32 GEMM that
runs on real sm_86 hardware at >= 3 TFLOP/s (true-f32), holding ~56% of true-f32 cuBLAS;
the `.shared`/`bar.sync` it emits are load-bearing (a no-barrier variant mis-computes);
the result matches an independent cuBLAS oracle that is itself anchored to a CPU oracle.

**Does NOT prove (carried forward):** this is the f32 SMEM tier only. It is NOT cp.async
double-buffered (G2), NOT Tensor-Core/TF32 (G3), NOT bf16 wmma (G4). The cuBLAS oracle is
a same-vendor reference (both run on NVIDIA's stack); it catches codegen/scheduling errors
and confirms numerical agreement, but a defect shared by kovc's emitter and cuBLAS's f32
path (unlikely given the independent CPU anchor) is outside its reach. The 4.56 TFLOP/s is
the median of a throttling laptop — the floor (min-time) implies higher peak, but the
median is the reported, conservative figure.

### Readiness for G2

G1 green unblocks **G2 (cp.async double-buffer, sm_86, >= 5 TFLOP/s, >= ~50% cuBLAS f32)**.
The harness (cuBLAS oracle + cuEvent TFLOP/s timing + both negative controls + the
parse-and-gate orchestrator) is now in place and reusable: G2 only needs the emitter to
add `cp.async.cg.shared.global` + `cp.async.commit_group`/`wait_group` to software-pipeline
two SMEM buffers (a `kovc.hx` change -> FULL self-host gate + a re-minted/re-committed
tiled reference PTX), then re-run `gpu_perf_corpus.sh` with the bar raised to 5 TFLOP/s
(`G1_MIN_TFLOPS=5`) and `cp.async` added to the provenance greps.

---

## G2 — cp.async double-buffer — pending
## G3 — TF32 mma.sync Tensor-Core (committed parity tier) — pending
## G4 — bf16 wmma (STRETCH) — pending
