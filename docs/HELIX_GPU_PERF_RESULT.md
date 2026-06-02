# Helix GPU Performance Result (Track 2)

Result document for `GPU_PERF_PASS` (charter `docs/HELIX_COMPLETION.md` §1.2). This file
records the measured perf tiers as they land. **G1 and G2 are GREEN** (G1 = SMEM-tiled f32
bar.sync @ 4.56 TFLOP/s; G2 = cp.async double-buffer @ 5.445 TFLOP/s, 67.5% cuBLAS f32). G3
(TF32 Tensor-Core, the committed parity tier) is next.

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

## G2 — cp.async double-buffer (Ampere async-copy software pipeline) — **PASS**

**Gate (§1.2 / §6 G2):** the cp.async double-buffered SMEM-tiled f32 GEMM is CORRECT vs the
CPU oracle AND vs the fenced cuBLAS oracle (cell-by-cell within tol), AND **>= 5 TFLOP/s** on
the RTX 3070 Laptop, AND `cp.async.cg.shared.global`/`commit_group`/`wait_group` provable in the
emitted PTX OUTPUT (>= ~50% cuBLAS f32). Measured 2026-06-02.

| metric | value |
|---|---|
| **kovc median TFLOP/s @ 2048^3** | **5.445 TFLOP/s** (min 2.91 / med 3.155 ms / max 5.28 ms over 50 timed kernel-only launches) |
| cuBLAS median TFLOP/s @ 2048^3 (true-f32, pedantic) | 8.07 TFLOP/s (med ~2.13 ms) |
| **kovc / cuBLAS ratio** | **~67.5%** (gate floor ~50%) |
| G2 bar (>= 5 TFLOP/s) | **PASS** (5.445 >= 5.0) |
| improvement vs G1 (same tiles, synchronous copy) | **+19%** (4.56 -> 5.445 TFLOP/s); ratio 56% -> 67.5% |
| correctness vs CPU oracle | 0 bad cells, 64^3..512^3 (integer-exact ==) |
| correctness vs cuBLAS oracle | 0 bad cells, 64^3..2048^3 (tol 1e-3 f32; integer-exact so really 0) |
| cuBLAS oracle trusted (cuBLAS == CPU) | 0 bad cells, 64^3..512^3 |
| PTX provenance (emitted OUTPUT) | `cp.async.cg.shared.global` (x4), `cp.async.commit_group` (x2), `cp.async.wait_group` (x2), `.shared .align 16` (x4 buffers), `bar.sync 0`, `fma.rn.f32`, `.target sm_86` all PRESENT |
| ptxas (sm_86) | 56 regs, **8192 B smem** (4x2048 = the double-buffer), 0 spills |
| `.version` / ptxas | `.version 8.0` (cp.async is PTX ISA 7.0+; accepted by the default CUDA-12.0 ptxas AND 12.8) |
| neg-control A (comparator teeth) | mutate one C cell -> FAILS (correct) |
| neg-control B (barriers load-bearing) | strip every `bar.sync` from emitted PTX -> mis-computes/FAILS (correct) |
| neg-control B' (cp.async load-bearing) | strip every `cp.async.wait_group` from emitted PTX -> mis-computes/FAILS (correct) |

**Tile params (unchanged from G1):** BM=BN=64, BK=8, TM=TN=4, threadblock 16x16=256,
grid=(N/BN, M/BM). The perf win is purely from the **software pipeline**, not a tile change.

### What changed in the emitter (`kovc.hx`)

`emit_ptx_tiled_matmul_smem` was restructured from a synchronous (`ld.global` + `st.shared`)
single-buffer k-tile loop into a **two-stage cp.async software pipeline**:

- **FOUR** `.shared .align 16` tiles (smem_a0/a1 + smem_b0/b1, 2048 B each = 8192 B), so the
  NEXT k-tile prefetches into the idle buffer pair while the CURRENT pair feeds the FMA inner
  product. `.align 16` (was 4) is REQUIRED for the 16-byte cp.async destination.
- New byte-emitters (modelled on the M1 `emit_ptx_*` style): `emit_ptx_cp_async_cg16`
  (`cp.async.cg.shared.global [smem],[gmem],16` — a 16-byte / vec4-f32 async copy straight
  GMEM->SMEM, bypassing the register file), `emit_ptx_cp_async_commit` (`commit_group`),
  `emit_ptx_cp_async_wait` (`wait_group N`), plus `emit_ptx_gaddr` (global byte address for
  the cp.async source), `emit_ptx_selp_b32` / `emit_ptx_setp_lt_s32` / `emit_ptx_xor_imm`
  (branch-free ping-pong) and `emit_ptx_cp_tile_load` (the cooperative tile stage).
- **Pipeline shape:** a PROLOGUE prefetch stages tile 0 into buffer pair 0 + `commit_group`;
  each loop iteration (a) prefetches the NEXT tile into the OTHER buffer pair (selected
  branch-free by a tile-parity register via `selp.b32`; the source k-offset is CLAMPED to
  K-BK so the final iteration's prefetch is a harmless redundant re-fetch, keeping EXACTLY 2
  cp.async groups in flight every iteration), (b) `cp.async.wait_group 1` (waits for precisely
  the current tile, leaving the just-issued prefetch in flight) + `bar.sync 0`, (c) runs the
  identical 4x4 register micro-tile FMA accumulate reading the CURRENT buffer pair, (d)
  `bar.sync 0`, advances k0 += BK and flips the parity (`xor 1`). A trailing
  `cp.async.wait_group 0` drains the clamped final prefetch.
- The cooperative load is **16-byte vectorized**: the 512 A elems + 512 B elems are each 128
  vec4 groups; threads 0..127 each issue one A + one B `cp.async` (threads >=128 skip via a
  single guard branch). Alignment holds for every issued copy: the gate's tile-divisibility
  (M%64==N%64==0, K%8==0) makes every A/B element index a multiple of 4 (byte offset multiple
  of 16) and the `.align 16` buffers + g*16 smem offsets keep the destination 16-aligned.
- The inner-product (`ld.shared` + `fma.rn.f32`) is **byte-for-byte the same math as G1** — the
  smem tile contents and layout are identical; only the buffer BASE is now runtime-selected
  per iteration. (A negative-control during bring-up: an early A-tile vec4 index bug produced a
  `misaligned address (716)` at every size — the cuBLAS+CPU correctness corpus caught it before
  any commit; the fix made the kk0 column-offset `g*4 - r*8`.)

### Methodology (honest)

Identical chain-of-trust + timing protocol to G1 (see the G1 section): fenced
`CUBLAS_PEDANTIC_MATH` true-f32 cuBLAS oracle anchored to a CPU oracle FIRST; integer inputs so
all three are f32-integer-exact; cuEvent kernel-only timing (5 warmup + 50 timed, median is the
throttle-robust figure); provenance grepped on the emitted `.ptx` OUTPUT, never source. The
**third negative control (B')** is new for G2 and proves the async-completion barrier is
load-bearing: stripping every `cp.async.wait_group` (PTX still ptxas-valid) makes the FMA race
the in-flight copy -> wrong result. All three controls trip.

### Self-host fixpoint (this IS a `kovc.hx` change)

Because the emitter changed, the FULL gate ran: `scripts/gate_kovc.sh` GREEN — self-host
fixpoint **K2==K3==K4 byte-identical** + feature corpus **56/56** + `vector_add` PTX regression
(unchanged: the non-tiled path is untouched) + the **tiled PTX regression against the re-minted,
re-committed** `helixc/examples/tiled_matmul_kernel.ref.ptx` (the G2 milestone intentionally
changed the tiled PTX per charter 1.0 step 2; the new reference carries the cp.async
double-buffer signature) + tiled provenance (`.shared` + `bar.sync` + cp.async in the OUTPUT).
A latent bootstrap-compiler bug was found+worked-around (not papered over): the kovc codegen
mis-passes function arguments beyond the 6th (no existing function had >6 params); the cp.async
tile-load helper was kept to 4 args by stashing its invariant context in spare `vtab` slots
56..62. The >6-arg codegen bug is logged as a separate follow-up, NOT a G2 blocker.

### Reproduce from scratch

```
# perf + correctness + provenance + 3 negative controls (G2 threshold):
G1_MIN_TFLOPS=5 wsl.exe bash -c "bash /mnt/c/Projects/Kovostov-Native/scripts/gpu_perf_corpus.sh"
# -> Verdict line: GPU_PERF_G2_PASS
# full self-host gate (re-mints kovc, ~28 min on /mnt/c):
wsl.exe bash -c "bash /mnt/c/Projects/Kovostov-Native/scripts/gate_kovc.sh"   # -> GATE_PASS
```

### What G2 proves / does not prove

**Proves:** kovc's own PTX codegen emits a correct, cp.async-double-buffered SMEM-tiled f32 GEMM
that runs on real sm_86 hardware at >= 5 TFLOP/s (true-f32), holding ~67.5% of true-f32 cuBLAS;
the Ampere async-copy + commit/wait_group it emits are load-bearing (a no-wait variant
mis-computes); the result matches an independent cuBLAS oracle anchored to a CPU oracle; the
self-host fixpoint stays byte-identical.

**Does NOT prove (carried forward):** this is still the f32 tier. It is NOT Tensor-Core/TF32 (G3,
the committed parity tier, >= 15 TFLOP/s) and NOT bf16 wmma (G4, stretch). The cuBLAS oracle is a
same-vendor reference (catches codegen/scheduling errors + confirms numerical agreement; a defect
shared by kovc's emitter and cuBLAS's f32 path is outside its reach, mitigated by the independent
CPU anchor). 5.445 TFLOP/s is the median of a throttling laptop — the floor (min-time ~2.91 ms ->
~5.9 TFLOP/s) implies a higher untthrottled peak; the median is the reported conservative figure.

### Readiness for G3

G2 green unblocks **G3 (TF32 `mma.sync` Tensor-Core GEMM, sm_86, >= 15 TFLOP/s, >= ~40%
cuBLAS-TF32 — the committed parity tier)**. The harness (cuBLAS oracle + cuEvent timing + the
three negative controls + parse-and-gate) is reusable; the cp.async SMEM staging G2 builds is the
exact feed the TF32 path needs (cp.async GMEM->SMEM, then `ldmatrix`/SMEM->fragment, then
`mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`). G3 must route to the **CUDA-12.8 ptxas**
and bump the emitted `.version` to **8.3** (the default 12.0 ptxas REJECTS `.version 8.3+` and the
`mma.sync` TF32 shapes); add a TF32-cuBLAS reference (drop `CUBLAS_PEDANTIC_MATH`) for the ratio,
`cvt.rna.tf32.f32` for the operand round, and `mma.sync`/`ldmatrix` to the provenance greps. It is
a `kovc.hx` emitter change -> FULL self-host gate + a re-minted/re-committed tiled reference PTX.

## G3 — TF32 mma.sync Tensor-Core (committed parity tier) — IN PROGRESS

### M-G3.0 — cuBLAS-TF32 baseline measurement (the perf denominator), measured 2026-06-02

Standalone `cublasGemmEx(CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP)`,
operands `CUDA_R_32F`, at **2048^3** on the reference box (RTX 3070 Laptop GPU, sm_86, driver
596.21 / CUDA-13.2 capable), warmup 10 + median-of-50 kernel-only cuEvent timing:

| metric | value |
|---|---|
| **cuBLAS-TF32 median TFLOP/s @ 2048^3** | **10.646 TFLOP/s** (min 8.013 / med 10.646 / max 11.555) |
| cuBLAS-TF32 ms @ 2048^3 | min 1.4868 / med 1.6138 / max 2.1439 ms |
| measured by | standalone pure-cuBLAS bench (no kovc kernel), so the denominator cannot be lost to codegen |
| prior reading (same box, earlier run) | med 11.260 (min 9.969 / max 11.603) — throttle variance ~6%; both confirm the ~10-11 TFLOP/s ceiling |

**Load-bearing reconciliation (skeptic-confirmed):** measured cuBLAS-TF32 median is **10.6
TFLOP/s** (a re-measure read 11.26; the throttled mobile GA104 varies ~6% run-to-run, both
well under 15), which is BELOW the originally-estimated 15 TFLOP/s absolute floor. The 15 was an
estimate of ~40% of an assumed cuBLAS-TF32 peak; the actual throttled-mobile-GA104 cuBLAS-TF32
ceiling is ~10-11 TFLOP/s, so **the kovc kernel physically cannot reach 15 (it cannot beat
cuBLAS)**. Per the pre-set honest rule, the **governing G3 perf threshold is the RELATIVE one:
median GEMM @ 2048^3 >= 40% of measured cuBLAS-TF32 = >= 4.26 TFLOP/s** (0.40 x 10.646). Both
numbers (absolute TFLOP/s and the cuBLAS-TF32 ratio) are reported always. The absolute 15
floor is documented here as superseded by the relative measure on this specific box.

## G4 — bf16 wmma (STRETCH) — pending
