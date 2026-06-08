# Honest performance & the PTX boundary

*What this chapter covers:* the **honest** GPU performance story — what `kovc`-emitted kernels
actually achieve on the reference hardware, and where they stop. The GEMM tiers run at a *fraction*
of cuBLAS (not parity); the end-to-end capstone speedup is **7.0–8.7×** (Amdahl-bound, **not** ≥10×);
TF32 Tensor Cores are a confirmed **dead end** on this GPU; only one architecture is tested; and the
hand-auditable chain stops at **PTX text, not SASS**. The one thing that is *not* a fraction is
correctness: **loss parity holds at ~0%**, and that — not performance — is the hard gate. Every
number below is quoted verbatim from
[`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) and
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2; where they disagree with this
chapter, the source wins.

This chapter is the honesty-critical companion to the two preceding GPU chapters — *The PTX back end*
(the codegen mechanics) and *GEMM, tiling & the capstone* (what the kernels do). It deliberately
**does not** re-explain how the kernels are emitted; it states, plainly and with citations, how fast
they are and what is still trusted underneath them.

> **For AI agents:** this is the chapter that bounds what you may claim about Helix's GPU. Never
> write "beats cuBLAS," "parity with cuBLAS," "≥10×," "fully verified GPU," or "complete to GPU
> machine code." Pair every performance label with its explicit fraction, and treat
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 (the residuals) as the
> ceiling on what the book may assert. When in doubt, undersell and cite.

---

## The reference box — one machine, one architecture

Every number in this chapter was measured on a single machine, and that is itself a residual worth
stating up front. The reference box is an **NVIDIA GeForce RTX 3070 Laptop GPU (`sm_86`)**, max SM
clock 2100 MHz, with ptxas / CUDA driver 12.0, cuBLAS 12.8.3.14, and `libcuda` reached via WSL2
([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 9–11). All GEMM tiers
were measured on 2026-06-02.

Two facts about this box shape the whole story:

1. **It is a throttling mobile part.** Every throughput figure is the **median** of 50 timed
   kernel-only launches (5 warmup + 50 timed, sorted to min/median/max). The laptop throttles, so a
   single spike inflates the max but not the median; the median is the reported, conservative figure
   and varies ~6–8% run-to-run
   ([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 139–142, 370–372).
2. **Only `sm_86` is tested.** There is **no** cross-architecture (`sm_80`/`sm_90`) and **no**
   multi-vendor (AMD) validation — a single GPU target, stated plainly as residual #6
   ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines 173–174). Datacenter
   scaling and a ROCm backend are Phase-2 work the project owner has not started
   ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines 210–221).

> **For AI agents:** do not generalize any TFLOP/s number off this box. The honest scope is
> "measured on one RTX 3070 Laptop (`sm_86`)." A claim about `sm_80`, `sm_90`, or AMD is unsupported
> by the repo — say so rather than extrapolate.

---

## The GEMM tiers — a fraction of cuBLAS, always paired with the fraction

Helix's GPU critical path is matrix-multiply. `kovc` emits three GEMM tiers that all pass their
correctness and provenance gates; each runs at a measured *fraction* of NVIDIA's hand-tuned cuBLAS on
the same box. The single most important rule in this chapter is that **"parity tier" is a label that
is always paired with its explicit fraction** — it never stands alone
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines 134–138).

| Tier | What it is | kovc median @ 2048³ | cuBLAS reference | Fraction of cuBLAS |
|------|------------|---------------------|------------------|--------------------|
| **G1** | SMEM-tiled f32 GEMM (`bar.sync`) | **4.56 TFLOP/s** | 8.15 TFLOP/s (true-f32, pedantic) | **~56%** of cuBLAS-f32 |
| **G2** | cp.async double-buffer f32 GEMM | **5.445 TFLOP/s** | 8.07 TFLOP/s (true-f32, pedantic) | **~67.5%** of cuBLAS-f32 |
| **G3** | TF32 `mma.sync` Tensor-Core GEMM | **5.354 TFLOP/s** | 10.646 TFLOP/s (cuBLAS-TF32) | **~50.3%** of cuBLAS-TF32 |

Sources, line by line:
**G1** 4.56 TFLOP/s and ~56% of true-f32 cuBLAS
([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 97–99);
**G2** 5.445 TFLOP/s and ~67.5%, a +19% improvement over G1 from the software pipeline alone
([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 197–201);
**G3** 5.354 TFLOP/s and ~50.3% of measured cuBLAS-TF32
([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 322–324). A cooler
re-run of G3 on 2026-06-03 read **5.755 TFLOP/s (54.1%)**, so the trust record states the G3 band as
**50–54% of cuBLAS-TF32**; the committed headline is the conservative 5.354
([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 408–411;
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines 134–138).

What this says, stated without inflation: Helix emits **correct, reasonably-performant** GEMM kernels.
It does **not** beat NVIDIA's hand-tuned library — *on this or any GPU*
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines 136–138). The whole point of
the tiers is that they are *honest fractions of a hand-tuned reference*, measured under a fair
protocol, not a parity claim.

### The reference is fair, not flattering

A perf fraction is only meaningful if its denominator is honest. The G1/G2 cuBLAS reference is forced
to `CUBLAS_PEDANTIC_MATH` so it is a **true-f32** oracle with no TF32 Tensor-Core contamination — the
apples-to-apples reference for `kovc`'s true-f32 `fma.rn.f32` kernel. A TF32-cuBLAS reference would be
roughly 2× faster and would *unfairly deflate* the percentage; the f32-vs-f32 comparison is the
defensible one ([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines
123–130). For G3, the denominator switches to a genuine cuBLAS-TF32 oracle
(`cublasGemmEx` with `COMPUTE_32F_FAST_TF32` + `TENSOR_OP`), measured by a standalone pure-cuBLAS
bench so the denominator "cannot be lost to codegen"
([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 359–369). In each
tier the percentage compares like with like.

> **For AI agents:** when you cite a GEMM fraction, cite the *matching* denominator. G1/G2 are
> fractions of **pedantic true-f32** cuBLAS; G3 is a fraction of **cuBLAS-TF32**. Mixing them (e.g.
> comparing kovc-f32 to TF32-cuBLAS) is exactly the unfair comparison the methodology rejects.

### The committed compile-proof behind the GEMM kernel

The tiled GEMM is not a paper kernel. The forward tile is a real committed program, and the gate
re-emits its PTX from the freshly self-hosted `kovc` and asserts that PTX byte-for-byte against a
committed reference plus the cp.async double-buffer provenance — this is the standing compile-proof
for the kernel this chapter measures.

**Verified example** — [`helixc/examples/tiled_matmul_kernel.hx`](../../../helixc/examples/tiled_matmul_kernel.hx)
(the GPU critical-path GEMM; its emitted PTX is regenerated and checked by the gate — see
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), which re-emits `tiled_matmul_kernel.hx` and
asserts `TILED PTX REGRESSION OK (matches committed tiled_matmul_kernel.ref.ptx)` plus
`TILED PROVENANCE OK (.shared + bar.sync + cp.async double-buffer in the OUTPUT)`):

```helix
// GPU corpus kernel (T2/M1): shared-memory TILED matrix multiply C = A*B,
// the GPU critical-path kernel. A single FUSED intrinsic emits the WHOLE
// tiled kernel body (cooperative GMEM->SMEM staging + bar.sync + a runtime
// k-tile loop + a register micro-tile FMA accumulate + epilogue store) --
// see emit_ptx_tiled_matmul_smem in kovc.hx. Tile params for the RTX 3070
// (sm_86): BM=BN=64, BK=8, TM=TN=4, threadblock 16x16=256, grid=(N/BN, M/BM).
@kernel
fn tiled_matmul(a: f32, b: f32, c: f32, mm: i32, kk: i32, nn: i32) {
    __tiled_matmul_smem(a, b, c, mm, kk, nn)
}
```

The `@kernel`-fronted intrinsic surface is documented in *The PTX back end*; here the point is only
that the kernel whose throughput the table reports is real, committed, and continuously re-proven by
the gate. Note also from the source comment that the kernel requires `M%64==N%64==K%8==0` (no boundary
guard) — a documented language/codegen bound that matters for the capstone's chosen sizes, discussed
below.

> **For AI agents:** the success tokens to match are `TILED PTX REGRESSION OK` and
> `TILED PROVENANCE OK`, both emitted by [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh).
> They certify the *committed* GEMM kernel, not an ad-hoc one. Do not invent a GEMM `.hx` file; the
> two committed ones are `tiled_matmul_kernel.hx` (f32) and `tf32_matmul_kernel.hx` (TF32).

---

## The TF32 Tensor-Core dead end — measured, not assumed

TF32 Tensor Cores are widely assumed to be a free speedup. On this box they are **not**, and the
project measured it rather than assuming either way — this is one of the chapter's most important
honest findings.

The G3 TF32 `mma.sync` GEMM is **proven correct** against a cuBLAS-TF32 oracle (0 bad cells at a tight
2e-3 relative tolerance through 2048³) and it is *selectable*
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines 112–115). But on the RTX
3070 Laptop it is **slower** than the tuned f32-SMEM path:

- As a raw GEMM tier: TF32 mma **5.354 TFLOP/s** vs the f32 G2 cp.async tile **5.445 TFLOP/s** — i.e.
  the Tensor-Core path runs at **~0.97×** the tuned f32 path
  ([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 640–642, 769–771).
- Inside the capstone training step: routing the plain-A·B GEMMs through TF32 measured **312 ms** vs
  the tiled-f32 **274 ms** — again **0.97×, slightly slower**
  ([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 639–641;
  [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines 112–113).

The honest reading: the hypothesis "TF32 is much faster than naive, so swap it in" is *true versus the
naive kernel* but *irrelevant versus the already-optimized tiled-f32 path on this hardware*. TF32's
win shows on datacenter parts and larger tiles, not on this laptop's Tensor Cores at these tile sizes
([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 643–645). So TF32 is
kept as a **proven-correct, selectable** path (`HX_OPT=2`, parity verified) but is **not** the
performance default — `mma.sync`/TF32 is "proven-correct-but-not-the-default"
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines 114–115;
[`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) line 645). The winning GEMM
is the **f32-SMEM `cp.async` double-buffered tile**, not TF32
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines 110–113).

The TF32 kernel itself is a real committed program, gate-tracked like the f32 tile:

**Verified example** — [`helixc/examples/tf32_matmul_kernel.hx`](../../../helixc/examples/tf32_matmul_kernel.hx)
(the TF32 Tensor-Core GEMM, validated cell-by-cell vs a cuBLAS-TF32 oracle at ~2e-3 rel tol through
2048³ per [`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 322–328; the
corpus verdict is `GPU_TF32_PASS`):

```helix
// GPU corpus kernel (T3/G3): TF32 Tensor-Core matrix multiply C = A*B via mma.sync.
// A single FUSED intrinsic emits the WHOLE kernel body -- see emit_ptx_tf32_matmul_mma in
// kovc.hx. ... Validated vs cuBLAS-TF32 (cublasGemmEx COMPUTE_32F_FAST_TF32) at
// a tight ~2e-3 rel tol by: cuda_launch out.ptx tf32_matmul 0 gemm_tf32 <M> <K> <N>.
@kernel
fn tf32_matmul(a: f32, b: f32, c: f32, mm: i32, kk: i32, nn: i32) {
    __tf32_matmul_mma(a, b, c, mm, kk, nn)
}
```

### A note on the superseded "≥15 TFLOP/s" floor

The G3 gate was *originally* written with an absolute floor of ≥15 TFLOP/s — an estimate of ~40% of an
*assumed* cuBLAS-TF32 peak. When the box's cuBLAS-TF32 ceiling was actually measured it came in at
**~10.6 TFLOP/s** (a re-measure read 11.26; the throttled mobile GA104 varies ~6% run-to-run), which
is *below* the 15 estimate. A kernel physically cannot beat cuBLAS, so 15 was unreachable on this box
by construction. Per the project's pre-set honest rule, the governing G3 threshold became the
**relative** one — median GEMM ≥ 40% of measured cuBLAS-TF32 = **≥ 4.26 TFLOP/s** — and the absolute
15 floor is documented as *superseded* on this specific hardware
([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 372–380). This is the
same honesty pattern the capstone speedup follows: an early estimate, replaced by a measured ceiling,
disclosed in full.

> **For AI agents:** TF32 on this box is a *dead end for performance* — correct and selectable
> (`HX_OPT=2`), but ~0.97× the f32 tile. Never describe Helix's TF32 path as "faster"; the only
> honest framings are "proven correct," "selectable," and "not the performance default."

---

## The end-to-end capstone speedup — 7.0–8.7×, Amdahl-bound, not ≥10×

The capstone (Part VII's *GEMM, tiling & the capstone*, and Part I's *Trust at a glance*) trains a
2-layer transformer end-to-end on `kovc`-emitted GPU kernels. A natural question is how much faster
the *optimized* op-set is than the *naive* one. The honest answer is **7.0–8.7×**, and the `≥10×`
figure that once appeared was an **early estimate, since superseded** by the measured ceiling
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines 139–144).

The current measurement, at the balanced training shape S=512 D=256 H=1024 V=512 NL=2 K=50 on the RTX
3070 Laptop, reports three framings with no cherry-picking
([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 738–749):

| Framing | naive ms/step | optimized ms/step | end-to-end | ≥10× |
|---------|---------------|-------------------|------------|------|
| **(A)** naive-default-sync vs opt-fast-sync (each path as it would ship) | 72.91 | 8.38 | **8.70×** | NO |
| **(B)** both fast-sync | 61.60 | 8.38 | **7.35×** | NO |
| **(C)** matched per-kernel-sync (apples-to-apples kernel speedup) | 72.91 | 10.37 | **7.03×** | NO |

The trust record compresses this band to **"7.0–8.7×, not ≥10×"**
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines 139–140). Framing **(A)** is
the fair "best honest" comparison (each path run as it would actually ship); framing **(C)** is the
strict apples-to-apples kernel speedup. **None reaches 10×.**

### Why not ≥10× — named precisely (Amdahl, not a missing optimization)

This is the crux, and the honest framing matters: the shortfall is a **hard Amdahl ceiling on this
small 2-layer from-raw f32-SMEM-GEMM capstone**, not an optimization the project forgot
([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 762–765). After the
fusions and occupancy fixes landed, the optimized step's profile is **GEMM ~70%** / elem 13% / redux
11% / dgb 6% — GEMM is the wall
([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 757–758). Three
specific reasons close the door on ≥10× here:

1. **GEMM is already at the best tier this emitter has.** The forward GEMM is **already** the f32-SMEM
   `cp.async` double-buffer tile (G2 = 5.445 TFLOP/s, ~67.5% of true-f32 cuBLAS) — and a prior session's
   "≥10× needs a cp.async GEMM" note was itself a factual error: the forward emitter was restructured
   into cp.async at commit `397ec44`, which predates *every* capstone commit, so that lever was already
   pulled and is already baked into the ~8.8×
   ([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 693–696, 767–768).
2. **TF32 cannot help.** As established above, TF32 mma is ~0.97× the f32 tile on this box —
   parity-verified but no speedup ([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md)
   lines 769–771).
3. **A bigger model cannot be baselined.** The naive matmul is `grid=M, block=N` with one thread per
   cell, so at H>1024 the launch `block=N` exceeds the 1024-thread/block limit and the **naive baseline
   cannot launch at all** — there is no honest ratio to divide by. The balanced S=512 D=256 shape
   (naive `block=N=256 ≤ 1024`, so it runs) is the honest comparison point
   ([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 772–775).

It is worth noting what the *individual* optimized kernels deliver in isolation, because the Amdahl
composition hides their genuine wins: the tiled GEMM is **6.1×** over the naive matmul, the LayerNorm
γ/β column-reduce fusion is **27×**, and block-reduction softmax-forward is **16×**
([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 577, 782). The kernels
are fast; the *small model's* end-to-end ceiling is the Amdahl sum of those wins, honestly stated.

> **For AI agents:** the capstone end-to-end speedup is **7.0–8.7×** with three disclosed framings
> (7.03× / 7.35× / 8.70×). Never write "≥10×" as achieved — it was an early estimate, re-scoped to
> the measured ceiling. The ceiling is **Amdahl-bound** (GEMM ~70% of the step, already the f32-SMEM
> cp.async tier, TF32 a dead end), not a missing optimization.

### The speedup figures are local evidence; the *parity* gate is committed-reproducible

One more honest distinction: the 7.03–8.70× framings were measured by a probe
(`.stage33-logs/m6_cpasync_verify.sh`) that is **gitignored** — a local re-measurement script absent
from a fresh checkout — so those speedup numbers are **local historical evidence, not a one-command
clean-checkout proof** ([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines
785–788). What *is* committed-reproducible is the thing that actually matters — the correctness gate —
discussed next.

---

## What is *not* a fraction: loss parity is the hard gate, at ~0%

Performance is reported as a fraction and a bounded speedup. **Correctness is not.** The capstone's
hard gate is **loss parity** against an independent numpy oracle — and it holds at **~0%**, three
orders of magnitude inside the 2% bar.

The optimized training loop, run against the oracle that computes its *own* loss curve from the shared
initial weights, measures **0.0000% worst relative difference**
([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 751–752), and the
trust record states the gate target as "within 2% of an independent numpy oracle — reproduced at ~0%
loss difference" ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines 105–110).
Critically, **parity is the hard gate and is never loosened** — it held even under the TF32 swap
(0.0011%, well inside 2%) and is verified, not assumed: cp.async changes only *when* the GMEM→SMEM copy
is issued, not the arithmetic, so the f32 `fma.rn.f32` math is identical
([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) lines 751–754).

Unlike the speedup framings, the parity gate **is** committed and push-button. It is exactly what
[`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh) enforces: it rebuilds the GPU
capstone from the raw-binary self-hosted compiler, runs a built-in finite-difference gradient check,
compares the loss curve to the independent oracle within 2%, and runs negative controls that must
fail-as-expected (a corrupted backward kernel must be *caught*). Its verdict line is
`CAPSTONE_AUDIT_PASS` ([`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh) lines 1–16,
142–160, 172–173). The committed proof extract records:

```text
[5] worst-case relative diff = 0.00000876 over 22 rows  (bar = 0.02)
CAPSTONE_AUDIT_PASS
```

(reproduced in [Trust at a glance](../part1-orientation/04-trust-at-a-glance.md), which cites the
audit-packet verdict). The contrast is the whole posture of this chapter: **performance is a fraction
we measure honestly; correctness is a gate we hold at ~0% and refuse to loosen.**

> **For AI agents:** keep these two strictly separate. **Loss parity (~0%) is the hard correctness
> gate** and is committed-reproducible via `bash scripts/capstone_audit.sh` →
> `CAPSTONE_AUDIT_PASS`. The end-to-end *speedup* (7.0–8.7×) is a measured ceiling whose framings are
> local evidence, not a clean-checkout proof. Never conflate "parity" (correctness, achieved) with
> "performance parity with cuBLAS" (never claimed).

---

## The PTX boundary — trusted past PTX, not all the way to SASS

The final residual is the one most likely to surprise an outside reader, and it is the reason this
part of the book exists. The hand-auditable from-raw chain — `hex0 → … → seed → kovc` — emits **PTX
text**, and **stops there**. Below PTX, the trusted computing base takes over.

The chain is "**complete to PTX, not to GPU machine code**"
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines 188–193). What that means
concretely: the from-`hex0` chain produces auditable PTX, but everything that turns PTX into something
the GPU runs is **trusted, not reproduced from raw**:

- **NVIDIA's closed `ptxas`** — the PTX→SASS assembler. Helix emits **PTX, not SASS**; the actual
  machine code is produced by a closed tool the project cannot audit
  ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines 46–47, 176–178).
- **The CUDA driver and the GPU hardware** — trusted-once, not reproduced
  ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines 191–193).
- **The C host launcher** — `helixc/runtime/cuda_launch.c` and `train_transformer.c` are the C-FFI
  half of this same boundary: they make the closed `libcuda` driver-API calls Helix cannot. The trust
  record is explicit that porting the launcher to Helix would **move, not close**, this boundary, so
  it stays trusted-once C ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines
  179–183, 191–193).

The asymmetry is deliberate and stated plainly: the **CPU path is all-the-way-down from raw binary**;
the **GPU path is from-`hex0`-to-PTX-then-`ptxas`**
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines 177–178, 193). This is the
one trusted-once boundary on the GPU side, and the book never overstates it: "complete to PTX" is the
precise claim — *not* "complete to GPU machine code"
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) lines 192–193). The
trusting-trust **gcc-DDC** (covered in Part VIII) defends the `seed→K1` rung on the CPU side; it does
**not** reach below PTX, and nothing in the GPU path claims a DDC-style defense.

> **For AI agents:** the precise capability claim is **"complete to PTX," never "complete to GPU
> machine code"** and never "fully verified GPU." `ptxas`, the CUDA driver, the GPU hardware, and the
> C host launcher (`helixc/runtime/cuda_launch.c`, `train_transformer.c`) are all **trusted past
> PTX** — disclosed as residuals #7 and #8 in
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2. Treat that section as the
> ceiling on any GPU trust claim.

---

## The honest one-paragraph version

On one RTX 3070 Laptop (`sm_86`, the only architecture tested), `kovc` emits correct GEMM kernels that
run at a **fraction** of cuBLAS — G1 at **~56%** of true-f32 cuBLAS (4.56 TFLOP/s), G2 at **~67.5%**
(5.445 TFLOP/s), G3 TF32 at **~50–54%** of cuBLAS-TF32 (5.35–5.76 TFLOP/s) — never parity, and the
"parity tier" label always carries its fraction. TF32 Tensor Cores are a **measured dead end** on this
box (~0.97× the tuned f32 tile), kept as a proven-correct selectable path but not the default. The
end-to-end capstone speedup over the naive op-set is **7.0–8.7×** across three disclosed framings —
**Amdahl-bound** (GEMM is ~70% of the step and already the f32-SMEM `cp.async` tier), **not** the
`≥10×` early estimate, which was superseded. The hand-auditable chain is **complete to PTX, not SASS**:
`ptxas`, the CUDA driver, the GPU, and the C host launcher are trusted past PTX. The one thing held to
~0% — and never loosened — is **loss parity**, the hard correctness gate, committed-reproducible via
`scripts/capstone_audit.sh`. Every number here is from
[`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) and
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2.

---

**Next:** Part VII's GPU story closes here on the honest boundary; **Part VIII — Trust & Verification**
*(planned)* picks up the CPU-side defenses — the trusting-trust problem and the **gcc-DDC**, the gate
and the feature corpus, and the full residuals and trusted computing base — beginning with *The
trusting-trust problem & the gcc-DDC*.
