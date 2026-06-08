# GEMM, tiling & the capstone

*What this chapter covers:* the GEMM performance tiers `kovc` emits (naive → SMEM-tiled →
`cp.async` double-buffered → a TF32 Tensor-Core attempt), the tiling strategy behind them, the
full transformer op set (the `gpu_*_kernel.hx` family plus attention, layernorm, softmax and
Adam), and the **capstone**: a real 2-layer transformer trained end-to-end on `kovc`-emitted PTX
kernels, gated on *loss parity* against an independent numpy oracle. The chapter you are reading
is about **correctness**, not performance-parity — the capstone proves the emitted kernels compute
the right answer; the honest performance story (and the PTX-not-SASS boundary) is the next chapter.

This chapter builds on [The PTX back end](01-ptx-backend.md), which covers how the `@kernel` path
in [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lowers Helix to PTX text at all.
Here we assume you know that `@kernel fn` bodies are lowered to PTX `.entry` kernels and run on the
reference RTX 3070 Laptop GPU (sm_86) through a trusted-once C host launcher.

> **For AI agents:** every Helix kernel quoted below is a real committed file under
> [`helixc/examples/`](../../../helixc/examples/) and is compiled by the standing GPU corpus
> (see the corpora named per kernel). The capstone is proven by
> [`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh), which prints the literal token
> `CAPSTONE_AUDIT_PASS` on success — match that exact token. The capstone needs a **real CUDA GPU**
> and is **separate** from the CPU-only `scripts/reproduce_trust.sh`.

---

## Why GEMM, and what "correct" buys you

A transformer's compute is dominated by matrix multiplies: the Q/K/V projections, the attention
score and value products, the output projection, the two MLP GEMMs, and the language-model head.
If `kovc` can emit a **correct** GEMM, and the surrounding elementwise and reduction kernels are
correct too, then a full training step is just a sequence of those launches — and if the resulting
loss curve matches an independent oracle, the whole emitted op set is corroborated end to end.

That is the structure of the capstone, and it is worth stating the division of labour up front,
because it is the single most important honesty point in this part of the book:

- **The capstone proves correctness.** A 2-layer transformer trains end-to-end on `kovc`-emitted
  PTX and its loss curve matches an independent numpy oracle to within 2% (reproduced at ~0%). This
  is the hard gate.
- **The GEMM *tiers* are a performance track**, and Helix's performance is honestly **a fraction of
  cuBLAS, not parity** (~56–67.5% of cuBLAS-f32 on the reference GPU; the TF32 tier ~50–54% of
  cuBLAS-TF32). The end-to-end capstone speedup is **7.0–8.7×**, Amdahl-bound, not ≥10×. All of that
  is quantified in [Honest performance & the PTX boundary](03-honest-performance.md) and disclosed in
  [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R.

> **Residual:** "correct" and "fast" are different claims, kept strictly separate. The capstone's
> 2% loss-parity gate says nothing about throughput, and the GEMM TFLOP/s tiers say nothing about
> end-to-end training correctness. The book never collapses the two. See
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) residuals 1–2.

---

## The GEMM tiers

`kovc` emits the same mathematical GEMM at four increasingly sophisticated tiers. Each tier is a
real, gate-proven kernel; the design progression is recorded in
[`docs/HELIX_GPU_GEMM_ROADMAP.md`](../../../docs/HELIX_GPU_GEMM_ROADMAP.md), and the measured
numbers (TFLOP/s, cuBLAS fractions, negative controls) are in
[`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md).

### Tier 0 — naive: one thread per output cell

The first real compute kernel is `naive_matmul`: one GPU thread computes one output cell `C[row,col]`
by looping over the contraction dimension. There is no shared memory and no barrier — the launch
geometry alone (`gridDim.x = M`, `blockDim.x = N`) supplies the row and column, sidestepping the
divide/modulo the emitter does not have.

**Verified example** — [`helixc/examples/naive_matmul_kernel.hx`](../../../helixc/examples/naive_matmul_kernel.hx)
(GPU corpus kernel; validated cell-by-cell vs a CPU reference per its header — `cuda_launch out.ptx
naive_matmul <N> matmul M K N`):

```helix
@kernel
fn naive_matmul(a: f32, b: f32, c: f32, mm: i32, kk: i32, nn: i32) {
    let row = block_idx();
    let col = thread_idx();
    let mut acc = a[row * kk] * b[col];
    let mut t = 1;
    while t < kk {
        acc = acc + a[row * kk + t] * b[t * nn + col];
        t = t + 1
    };
    c[row * nn + col] = acc
}
```

Two design fingerprints recur across the whole GPU corpus and are visible here:

- **The accumulator is seeded with the `t = 0` product** (`a[row*kk] * b[col]`) rather than `0.0`.
  The bootstrap literal parser is integer-only, so an f32-literal zero needs a special codegen path;
  seeding with the first product sidesteps it. (`gpu_qkt`, `gpu_matmul_atb`, `gpu_matmul_abt` all do
  the same; the reduction kernels instead use the `x[base] - x[base]` idiom to make a zero.)
- **Scalar dims are `i32` params, pointers are `f32`.** `mm`, `kk`, `nn` are matrix dimensions read
  via `ld.param.u32`; `a`, `b`, `c` are device pointers. `mm` (M) is unused in the body because the
  row comes straight from `block_idx()`.

The naive kernel is correct but throughput-poor (no data reuse — every thread re-reads A and B rows
from global memory) and it has a hard structural ceiling: with `blockDim.x = N`, a width `N > 1024`
exceeds the per-block thread limit and the kernel **cannot launch at all**. That ceiling matters for
the honest speedup story (it is why a *wider* model cannot be baselined against naive; see
[`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) M6.2).

### Tier 1 — SMEM-tiled: cooperative staging + a register micro-tile

The throughput-unlocking idea is **data reuse through shared memory**. Instead of one thread per
cell, a 256-thread block cooperatively stages a 64×8 tile of A and an 8×64 tile of B into
`.shared` memory, synchronizes with `bar.sync`, and then each thread computes a 4×4 register
micro-tile of outputs by reading the staged tiles repeatedly. This is the tier the transformer
matmuls actually use.

In Helix this entire tiled body is emitted by a single **fused intrinsic**, `__tiled_matmul_smem`,
dispatched in `emit_ptx_tiled_matmul_smem` inside
[`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx). The `.hx` kernel is therefore a
one-liner that names the intrinsic — the complexity lives in the emitter, not the source:

**Verified example** — [`helixc/examples/tiled_matmul_kernel.hx`](../../../helixc/examples/tiled_matmul_kernel.hx)
(GPU corpus kernel; validated cell-by-cell vs a CPU GEMM oracle per its header — `cuda_launch
out.ptx tiled_matmul 0 gemm_perf <M> <K> <N>`):

```helix
@kernel
fn tiled_matmul(a: f32, b: f32, c: f32, mm: i32, kk: i32, nn: i32) {
    __tiled_matmul_smem(a, b, c, mm, kk, nn)
}
```

The tile parameters, from the kernel's own header and from
[`docs/HELIX_GPU_GEMM_ROADMAP.md`](../../../docs/HELIX_GPU_GEMM_ROADMAP.md) Step C, are tuned for the
RTX 3070 (sm_86): `BM = BN = 64`, `BK = 8`, `TM = TN = 4`, threadblock `16×16 = 256`,
grid `= (N/BN, M/BM)`. The PTX the emitter produces uses `.shared .b32` declarations,
`st.shared`/`ld.shared.f32`, and `bar.sync 0` — the new PTX constructs the tiled tier required over
the naive one.

> **Note:** the tiled kernels require every matmul axis to be a multiple of the tile (`M%64 == N%64
> == 0`, `K%8 == 0`) with **no boundary guard** — exactly like `naive_matmul` has no bounds guard.
> This divisibility constraint is why the capstone's optimized op-set runs at a scaled-up size
> (`S=128 D=64 H=256 V=128` and larger) where the tiles are valid. See `train_transformer.c`'s
> header comment ([`helixc/runtime/train_transformer.c`](../../../helixc/runtime/train_transformer.c)
> lines ~16–17).

This tier is measured as **G1** in [`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md):
4.56 TFLOP/s at 2048³ on the reference GPU, ~56% of true-f32 cuBLAS. Two negative controls back the
provenance: mutating one output cell makes the cell-by-cell comparator fail (the comparator has
teeth), and *stripping every `bar.sync`* from the emitted PTX (still ptxas-valid) makes the kernel
mis-compute — proving the `.shared`/`bar.sync` it emits are load-bearing, not cosmetic.

### Tier 2 — `cp.async` double-buffer: the winning f32 tier

The Ampere `cp.async` instruction copies global memory straight into shared memory, bypassing the
register file, and can be issued asynchronously. Tier 2 keeps the same tile parameters as Tier 1
but restructures the k-tile loop into a **two-stage software pipeline**: while the current tile
feeds the FMA inner product, the *next* tile is already prefetching into a second pair of shared
buffers. The emitter (`emit_ptx_tiled_matmul_smem`, restructured for G2) emits four
`.shared .align 16` tiles and the `cp.async.cg.shared.global` / `cp.async.commit_group` /
`cp.async.wait_group` instruction family.

This is the **winning GEMM tier on the reference hardware** — measured as **G2** at 5.445 TFLOP/s
at 2048³, ~67.5% of true-f32 cuBLAS, a +19% improvement over G1 from the software pipeline alone
([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) G2). A third negative
control is added here: stripping every `cp.async.wait_group` makes the FMA race the in-flight copy
and mis-compute, proving the async-completion barrier is load-bearing.

> **For AI agents:** [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1 states
> the winning GEMM plainly: *"The WINNING GEMM is the f32-SMEM `cp.async` double-buffered tile,
> not TF32."* Do not describe TF32 as the performance path on this GPU.

### Tier 3 — TF32 Tensor-Core (`mma.sync`): the attempt, proven correct but *not* the default

The natural next reach is the Tensor Cores via TF32 `mma.sync`. `kovc` *does* emit this: a single
fused intrinsic `__tf32_matmul_mma` emits a warp-collaborative kernel where 32 lanes jointly hold
the A/B/C fragments and the Tensor Core does a 16×8×8 contraction per `mma.sync`.

**Verified example** — [`helixc/examples/tf32_matmul_kernel.hx`](../../../helixc/examples/tf32_matmul_kernel.hx)
(GPU corpus kernel; validated vs a cuBLAS-TF32 oracle, `cublasGemmEx` `COMPUTE_32F_FAST_TF32`, at a
~2e-3 relative tolerance per its header — `cuda_launch out.ptx tf32_matmul 0 gemm_tf32 <M> <K> <N>`):

```helix
@kernel
fn tf32_matmul(a: f32, b: f32, c: f32, mm: i32, kk: i32, nn: i32) {
    __tf32_matmul_mma(a, b, c, mm, kk, nn)
}
```

The honest result is the important part. Measured as **G3** in
[`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md): 5.354 TFLOP/s
(a cooler re-run read 5.755), ~50–54% of measured cuBLAS-TF32. Crucially, on *this* laptop TF32
mma is **slightly slower** than the tuned f32 `cp.async` GEMM — 312 ms vs 274 ms for the capstone's
GEMM phase, i.e. **0.97×** ([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md)
M6.2). TF32's Tensor-Core advantage shows on datacenter parts and larger tiles, not on the throttled
mobile GA104. So the TF32 op-set is *emitted and selectable* (`HX_OPT=2`, with 2% loss parity
verified) but it is **not** the performance path — `mma.sync`/TF32 is proven-correct-but-not-the-default
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1). The kernel is still
Path-2 (manual `ld.global.f32` + `cvt.rna.tf32.f32`, no SMEM staging / no `ldmatrix`) per its header.

> **Residual:** the "parity tier" label for TF32 is always paired with its honest fraction
> (~50–54% of cuBLAS-TF32). It is a *correctness* parity (0 bad cells at 2e-3 through 2048³), not a
> performance parity with cuBLAS. bf16 `wmma` (G4) is named a stretch and **was not taken**
> ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R-5).

---

## Tiling strategy: where the complexity lives

A few decisions shape every kernel in this part, and understanding them explains why the `.hx`
sources look the way they do.

**Fused intrinsics for the heavy kernels.** The tiled and Tensor-Core GEMMs, and fused
flash-style attention, are each emitted by a *single* fused intrinsic (`__tiled_matmul_smem`,
`__tf32_matmul_mma`, `__matmul_abt_smem` / `__matmul_atb_smem`, `__flash_attention`) handled in
`emit_ptx_call` inside [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx). The `.hx`
kernel is a thin wrapper naming the intrinsic; the cooperative staging, barriers, register
micro-tile, and epilogue all live in the emitter. This keeps the emitted PTX byte-stable: a new
emitter fires *only* for its intrinsic name, so adding it leaves the other kernels'
reference PTX byte-identical and the self-host fixpoint untouched
([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) M4).

**One thread per row for reductions — no shared memory on the correctness track.** For softmax,
layernorm and attention scores, the simplest correct form is one thread owning a full row and
reducing serially over the hidden dimension (64–256 wide). This eliminates any need for `.shared` /
`bar.sync` on the correctness track. As [`docs/HELIX_GPU_TRANSFORMER_PLAN.md`](../../../docs/HELIX_GPU_TRANSFORMER_PLAN.md)
puts it: *"The capstone's 'within 2% of PyTorch' bar is correctness, not throughput — even a 100×
slower naive kernel passes if the loss matches."* Block-reduction variants exist as a separate perf
track (the `*_blockred` kernels), used by the optimized op-set.

**1-D launch and flattening.** The emitter has only `.x` thread/block indices, so 2-D tensors are
flattened to a 1-D thread space; the elementwise kernels compute a flat index
`block_idx()*block_dim() + thread_idx()` with no bounds guard, so `blocks*threads` must equal the
element count exactly ([`docs/HELIX_GPU_TRANSFORMER_PLAN.md`](../../../docs/HELIX_GPU_TRANSFORMER_PLAN.md)).

**Baked literals where the scalar-param ABI is narrow.** The bootstrap parser is integer-literal-only
and there is no f32 *scalar* param, so f32 constants are emitted as hex-bit `mov.f32` and step- or
dimension-dependent scalars are sometimes baked. For example, `gpu_qkt` bakes the attention scale
`0.25 = 1/sqrt(16)` for `d=16`, and `gpu_adam` bakes its hyperparameters and passes the
step-dependent bias-correction terms as 1-element f32 arrays. The optimized op-set introduces
`gpu_scale_rt` to make the scale a runtime scalar (dimension-agnostic) so the same kernels are
correct at any `d`.

---

## The transformer op set

The capstone needs a complete forward + backward op set. Every op is a real kernel in
[`helixc/examples/`](../../../helixc/examples/), each validated against a CPU oracle (the CPU
references live in [`helixc/stdlib/nn.hx`](../../../helixc/stdlib/nn.hx) and
[`helixc/stdlib/transcendentals.hx`](../../../helixc/stdlib/transcendentals.hx), per
[`docs/HELIX_GPU_TRANSFORMER_PLAN.md`](../../../docs/HELIX_GPU_TRANSFORMER_PLAN.md)). The
transcendental support the ops rely on is itself emitted as gated intrinsics:
`__gpu_exp` → `ex2.approx.f32` (with the `log2e` pre-multiply), `__gpu_rsqrt` → `rsqrt.approx.f32`,
and `__gpu_i2f` for integer→float conversion.

### The three matmul variants

Linear-layer backprop needs three matmul shapes, completing the set: `A@B` (`naive_matmul` /
`tiled_matmul`), `A@Bᵀ` (`gpu_qkt` scaled, `gpu_matmul_abt` unscaled / `tiled_matmul_abt`), and
`Aᵀ@B` (`gpu_matmul_atb` / `tiled_matmul_atb`). The transposed forms are the weight-gradient and
input-gradient products. The naive `Aᵀ@B`:

**Verified example** — [`helixc/examples/gpu_matmul_atb_kernel.hx`](../../../helixc/examples/gpu_matmul_atb_kernel.hx)
(GPU corpus kernel; validated cell-by-cell vs a CPU Aᵀ@B oracle with exact integer inputs per its
header — `cuda_launch out.ptx gpu_matmul_atb <N> matmul_atb M K N`):

```helix
@kernel
fn gpu_matmul_atb(a: f32, b: f32, c: f32, mm: i32, kk: i32, nn: i32) {
    let i = block_idx();
    let j = thread_idx();
    let mut acc = a[i] * b[j];
    let mut t = 1;
    while t < mm {
        acc = acc + a[t * kk + i] * b[t * nn + j];
        t = t + 1
    };
    c[i * nn + j] = acc
}
```

The contraction here runs over `mm` (the M rows both A and B share), and the output `C` is `[K, N]`.
The tiled siblings `tiled_matmul_abt` / `tiled_matmul_atb` reuse the forward tiled GEMM's SMEM
layout and 4×4 register micro-tile verbatim; the *only* change is how each tile element's global
index is computed under transposition (their headers spell this out). The transposed GEMMs are
measured correct (0 bad cells) and faster-than-naive at the large/non-square sizes where tiling's
data-reuse dominates ([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) M4).

### Softmax (the first reduction)

**Verified example** — [`helixc/examples/gpu_softmax_kernel.hx`](../../../helixc/examples/gpu_softmax_kernel.hx)
(GPU corpus kernel; one thread per row, validated vs the CPU softmax in
[`helixc/stdlib/nn.hx`](../../../helixc/stdlib/nn.hx), each row must sum to ~1 — `cuda_launch out.ptx
gpu_softmax <N> softmax rows cols`):

```helix
@kernel
fn gpu_softmax(x: f32, y: f32, rows: i32, cols: i32) {
    let row = block_idx() * block_dim() + thread_idx();
    let base = row * cols;
    let mut m = x[base];
    let mut j = 1;
    while j < cols {
        if x[base + j] > m { m = x[base + j] };
        j = j + 1
    };
    let mut s = x[base] - x[base];
    let mut k = 0;
    while k < cols {
        let e = __gpu_exp(x[base + k] - m);
        y[base + k] = e;
        s = s + e;
        k = k + 1
    };
    let mut t = 0;
    while t < cols {
        y[base + t] = y[base + t] / s;
        t = t + 1
    }
}
```

This is the canonical row-reduction shape: three serial passes (row max → exp-and-sum → normalize),
the `x[base] - x[base]` zero idiom, `__gpu_exp` for the exponentials, and an `if`-statement for the
running max. `gpu_ce_softmax_grad` (the backprop root) mirrors it, computing the clean fused
softmax-cross-entropy gradient `softmax(logits) - onehot(target)`; because the emitter has no i32
*array* param, the integer class target is carried as an f32 and the one-hot is computed branchlessly.

### LayerNorm forward + save

LayerNorm's backward pass needs the reciprocal standard deviation, so the forward kernel saves it:

**Verified example** — [`helixc/examples/gpu_layernorm_fwd_save_kernel.hx`](../../../helixc/examples/gpu_layernorm_fwd_save_kernel.hx)
(GPU corpus kernel; one thread per row, verified `y` matches a CPU affine layernorm AND
`ist[row] = 1/sqrt(var_row)` — `cuda_launch out.ptx gpu_layernorm_fwd_save <N> layernorm_save <rows> <cols>`):

```helix
@kernel
fn gpu_layernorm_fwd_save(x: f32, y: f32, gamma: f32, beta: f32, ist: f32, cols: i32) {
    let row = block_idx() * block_dim() + thread_idx();
    let base = row * cols;
    let colsf = __gpu_i2f(cols);
    let mut sm = x[base] - x[base];
    let mut j = 0;
    while j < cols { sm = sm + x[base + j]; j = j + 1 };
    let mean = sm / colsf;
    let mut vs = x[base] - x[base];
    let mut k = 0;
    while k < cols { vs = vs + (x[base + k] - mean) * (x[base + k] - mean); k = k + 1 };
    let var = vs / colsf;
    let inv = __gpu_rsqrt(var);
    ist[row] = inv;
    let mut t = 0;
    while t < cols { y[base + t] = gamma[t] * ((x[base + t] - mean) * inv) + beta[t]; t = t + 1 }
}
```

The backward gamma/beta gradient (`gpu_layernorm_backward_dgb`) and dx (`gpu_layernorm_backward_dx`)
complete LayerNorm. One detail matters for the perf story (next chapter): the naive `dgb` recomputes
the per-row mean *per column*, which is `O(rows·cols²)` and turns out to be the dominant cost at the
capstone's scale. The optimized op-set fixes this with two pure-Helix kernels — `gpu_row_mean` fills
`mean[rows]` once, and `gpu_layernorm_backward_dgb_pm` reads it — making the column-reduce
`O(rows·cols)` with *identical math*, so the finite-diff and 2% oracle checks still gate it
([`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) M6.2(b)).

### GELU and Adam

GELU is the tanh-approximation form, the first kernel to combine f32 literals with `__gpu_exp`:

**Verified example** — [`helixc/examples/gpu_gelu_kernel.hx`](../../../helixc/examples/gpu_gelu_kernel.hx)
(GPU corpus kernel; one thread per element, validated vs an independent CPU `expf`-based reference,
maxrel 1.14e-07 per [`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) M4
item 3 — `cuda_launch out.ptx gpu_gelu <N> gelu`):

```helix
@kernel
fn gpu_gelu(x: f32, y: f32, n: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    let xi = x[i];
    let x3 = xi * xi * xi;
    let inner = 0.7978846 * (xi + (0.044715 * x3));
    let e2 = __gpu_exp(2.0 * inner);
    let th = (e2 - 1.0) / (e2 + 1.0);
    y[i] = 0.5 * xi * (1.0 + th)
}
```

Adam is one in-place optimizer step. Its hyperparameters (`b1=0.9`, `b2=0.999`, `lr=1e-3`,
`eps=1e-8`) are baked literals; the two step-dependent bias-correction scalars are passed as
1-element f32 arrays (`bc1`, `bc2`) and read as `bc1[0]` / `bc2[0]` — the workaround for the
narrow scalar-param ABI:

**Verified example** — [`helixc/examples/gpu_adam_kernel.hx`](../../../helixc/examples/gpu_adam_kernel.hx)
(GPU corpus kernel; one thread per element, verified vs an independent CPU Adam step, maxrel(w)
3.81e-07 per [`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) M4 item 3 —
`cuda_launch out.ptx gpu_adam <N> adam`):

```helix
@kernel
fn gpu_adam(w: f32, g: f32, m: f32, v: f32, bc1: f32, bc2: f32) {
    let i = block_idx() * block_dim() + thread_idx();
    let gi = g[i];
    let nm = 0.9 * m[i] + 0.1 * gi;
    let nv = 0.999 * v[i] + 0.001 * gi * gi;
    let mh = nm * bc1[0];
    let vh = nv * bc2[0];
    let upd = 0.001 * mh * __gpu_rsqrt(vh + 0.00000001);
    w[i] = w[i] - upd;
    m[i] = nm;
    v[i] = nv
}
```

### Fused flash-style attention

The set also includes a fused flash-style attention kernel — one 256-thread block per query row that
computes `out = softmax(Q @ Kᵀ / sqrt(d)) @ V` with the S×S score matrix resident only in shared
memory, never materialized in HBM. Like the heavy GEMMs, the whole body comes from one fused
intrinsic:

**Verified example** — [`helixc/examples/flash_attention_kernel.hx`](../../../helixc/examples/flash_attention_kernel.hx)
(GPU corpus kernel; validated vs a CPU `out = softmax(scale·Q@Kᵀ)@V` reference, 0 bad cells, maxrel
≈ 1e-07, per [`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) M4 item 4 —
`cuda_launch out.ptx flash_attention 0 attn_flash <S> <d>`):

```helix
@kernel
fn flash_attention(q: f32, k: f32, v: f32, o: f32, ss: i32, dd: i32) {
    __flash_attention(q, k, v, o, ss, dd)
}
```

> **Residual:** this is honestly a *SMEM-resident-scores* fused attention with a block-reduction
> softmax — the real flash memory win (scores never touch HBM) — but **not** the register-tiled
> warp-level online-rescale form of cuDNN flash-attention. It is also *not* used in the capstone's
> *training* forward, precisely because it does not materialize the S×S attention-weights matrix the
> backward pass consumes; the capstone uses the non-fused tiled QKᵀ + block-reduction softmax +
> tiled @V path so the weights are saved. flash_attention is gated separately
> (`scripts/gpu_attention_corpus.sh` → `GPU_ATTENTION_PASS`); see
> [`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) M4 item 4.

> **For AI agents:** the kernels above are part of the standing GPU corpus and are compiled by
> `kovc` on every reproduction; the broader **109-program feature corpus** the gate runs
> (`CORPUS: 109 passed, 0 failed`, [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)) is the
> compile-proof for the wider language surface. Do not invent new standalone kernels — the op set is
> exactly the `gpu_*_kernel.hx` / `tiled_*_kernel.hx` / `tf32_matmul_kernel.hx` /
> `flash_attention_kernel.hx` files in [`helixc/examples/`](../../../helixc/examples/).

---

## The capstone: a real transformer, gated on loss parity

The capstone assembles these kernels into a real workload. A tiny **2-layer pre-norm transformer**
(default `V=32, D=16, S=16, 1 head, H=64, NL=2`) is trained end-to-end on `kovc`-emitted PTX. The
harness is [`helixc/runtime/train_transformer.c`](../../../helixc/runtime/train_transformer.c) —
**Option A: a trusted C launcher**. Its first line of comment states the discipline exactly:

> *"ALL math is kovc-emitted PTX, the C only does memory + launch sequencing"*
> ([`helixc/runtime/train_transformer.c`](../../../helixc/runtime/train_transformer.c) lines 1–2).

The C loads the combined PTX module, allocates device memory, uploads weights generated from a fixed
xorshift seed, and then drives the training step as a sequence of `cuLaunchKernel` calls — forward
(`forward_full`), backward (`backward_full`), and an Adam update per parameter (`adam_step`). The
loss is a host-side cross-entropy (`ce_loss`) computed from logits copied back from the device.

The launcher has two modes, and this distinction is load-bearing for the audit
([`helixc/runtime/train_transformer.c`](../../../helixc/runtime/train_transformer.c) line ~340 and
[`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh) lines 11–13):

- `/tmp/train <combined.ptx> verify` runs the **sampled finite-difference gradient check** and
  returns — no training, no loss curve.
- `/tmp/train <combined.ptx>` **trains** for `K` (default 500) Adam steps and writes
  `loss_curve.csv` + `init_weights.bin`, but does *not* run the finite-diff check.

So each leg of the audit needs its own invocation. The same harness also supports an optimized
op-set via `HX_OPT` environment variables (tiled/Tensor-Core GEMMs + block-reduction kernels at a
scaled-up, tile-valid size), with the *training math identical* in both paths — which is exactly why
the loss-parity check against the oracle is a real correctness gate, not a tautology
([`helixc/runtime/train_transformer.c`](../../../helixc/runtime/train_transformer.c) lines ~10–17).

### The finite-difference gradient check

Backward correctness is spot-checked by a **sampled central finite-difference of the loss**. For a
handful of gradient tensors, the harness perturbs a few weights by `h = 1e-3`, recomputes the loss
forward both ways, and compares `(L₊ − L₋)/(2h)` against the analytic gradient the backward kernels
produced ([`helixc/runtime/train_transformer.c`](../../../helixc/runtime/train_transformer.c) lines
~408–443). It is *independent* of the backward path because it uses only the verified forward, and
it prints the literal verdict:

```text
GPU [...] backward finite-diff: PASS
```

This is exactly the string [`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh) step
`[4a]` greps for (`grep -q "backward finite-diff: PASS"`), failing the audit if it is absent.

### The independent numpy oracle

The hard gate is **loss parity** against an independent oracle:
[`verification/oracle/oracle_train.py`](../../../verification/oracle/oracle_train.py). This is the
repository's **single committed `.py` file** — a fenced numpy *audit oracle* that is never on the
compile or run path ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1, the
Python-free invariant). The oracle was adversarially proven genuinely independent: it computes its
*own* loss curve from the *shared initial weights*, and reads Helix's `loss_curve.csv` only for the
comparison, never as an input to its own computation
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1, "Real capability").

The audit pastes the two curves together and computes the worst-case relative difference, gating it
**below 0.02** (2%). From [`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh) step `[5]`:

```bash
worst=$(awk -F',' 'NF>=4 { h=$2; o=$4; if(o!=0){ d=(h-o)/o; if(d<0)d=-d; if(d>m)m=d; n++ } } END{ if(n>0) printf "%.8f", m; else printf "NaN" }' /tmp/ca_cmp.csv)
...
awk -v W="$worst" 'BEGIN{ if (W != "NaN" && W+0 >= 0 && W+0 < 0.02) exit 0; else exit 1 }' || { echo "  WITHIN-2% FAIL"; OK=0; }
```

The honest worst-case observed in the committed run is **~0.0009%** (`0.00000876` worst-case
relative difference over 22 comparable rows) — three orders of magnitude inside the 2% bar
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1; the verbatim
`worst-case relative diff = 0.00000876 over 22 rows (bar = 0.02)` line appears in
[`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md), quoted in
[Trust at a glance](../part1-orientation/04-trust-at-a-glance.md)). "Reproduced at ~0%" means exactly
this: the f32 (Helix) and f64 (numpy) curves are close-but-not-bit-identical, and their largest
relative gap rounds to zero against a 2% bar.

### The negative controls

A gate that cannot fail proves nothing, so the audit includes **negative controls that must
fail-as-expected**:

- **NC1 — convergence (the loss is real).** The final training loss must satisfy `0 < L < 1.0`,
  asserted in step `[4b]`. A diverged or zero loss fails.
- **NC2 — independence (the curves are not the same file).** The Helix and oracle curves must *not*
  be byte-identical; if `cmp -s` finds them identical, that is `NC2 FAIL (curves byte-identical →
  not independent)` (step `[5]`).
- **NC-PERTURB — the finite-diff check is load-bearing.** Step `[6]` deliberately *corrupts* the
  GELU-backward kernel by mutating its constants
  (`sed 's/0\.7978846/0.9978846/g; s/0\.044715/0.144715/g'` on
  [`helixc/examples/gpu_gelu_backward_kernel.hx`](../../../helixc/examples/gpu_gelu_backward_kernel.hx)),
  re-emits the PTX, and re-runs `verify`. The finite-diff check **must catch it**; if the corrupted
  backward still passes, that is `NC-PERTURB FAIL (corrupted backward STILL passed finite-diff →
  check not load-bearing!)`.

The corrupted kernel is the real GELU-backward — the tanh-approx GELU derivative — and the audit
proves that perturbing its constants is detected:

**Verified example** — [`helixc/examples/gpu_gelu_backward_kernel.hx`](../../../helixc/examples/gpu_gelu_backward_kernel.hx)
(GPU corpus kernel; verified vs a CPU central finite-difference of the GELU *forward* AND vs the
analytic `gelu'` in C — `cuda_launch out.ptx gpu_gelu_backward <N> gelu_backward`):

```helix
@kernel
fn gpu_gelu_backward(x: f32, dy: f32, dx: f32, n: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    let xi = x[i];
    let inn = 0.7978846 * (xi + 0.044715 * xi * xi * xi);
    let e2 = __gpu_exp(2.0 * inn);
    let th = (e2 - 1.0) / (e2 + 1.0);
    let id = 0.7978846 * (1.0 + 0.134145 * xi * xi);
    let gp = 0.5 * (1.0 + th) + 0.5 * xi * (1.0 - th * th) * id;
    dx[i] = dy[i] * gp
}
```

### `capstone_audit.sh`: the standing proof and its legs

[`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh) is one capstone audit round — the
*dynamic* half of a 5-consecutive-clean audit — and it is the standing proof: on a CUDA host it
prints `CAPSTONE_AUDIT_PASS`. Its legs, in order:

1. **`[0]` ambient-env neutralization** — unset every `HX_*` variable and assert none remain, so the
   audit runs the default (v1.0) configuration regardless of the caller's environment. A stray
   `HX_*` would silently change the dims/op-set and masquerade as the audit.
2. **`[1]` toolchain trust spine** — run [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)
   (self-host fixpoint + GPU PTX regression + the corpus), and mint a **fresh** PTX driver
   `/tmp/newdrv.bin` *from the raw-binary seed this round* (killing staleness). The gate must print
   `GATE_PASS`.
3. **`[2]–[3]` emit the combined PTX** — the seed-minted driver emits **15** kovc-emitted transformer
   kernels into one PTX module (`vector_add naive_matmul gpu_matmul_atb gpu_matmul_abt gpu_qkt
   gpu_softmax gpu_softmax_backward gpu_gelu gpu_gelu_backward gpu_layernorm_fwd_save
   gpu_layernorm_backward_dx gpu_layernorm_backward_dgb gpu_ce_softmax_grad gpu_scale_inplace
   gpu_adam`); the module must contain ≥ 15 `.entry` kernels.
4. **`[4a]` finite-diff gradient check** — build the launcher with `gcc` and run `verify`; require
   `backward finite-diff: PASS`.
5. **`[4b]` train** — 500 Adam steps on the GPU, writing a *fresh* `loss_curve.csv` (with a
   stale-artifact guard that `rm`s the outputs before the run); assert NC1 (`0 < L < 1.0`).
6. **`[5]` oracle + within-2% compare** — run the numpy oracle (a nonzero exit is itself a failure,
   since the oracle exits 1 on a failed analytic-vs-finite-diff self-check), require ≥ 10 comparable
   rows (a vacuous-pass guard), assert worst-case rel diff `< 0.02`, and assert NC2.
7. **`[6]` NC-PERTURB** — the corrupted-GELU-backward control above.

The script is **fail-closed**: it propagates the verdict to the process exit status, so a caller can
never read a printed `CAPSTONE_AUDIT_FAIL` as success (lines 175–177).

> **For AI agents:** the capstone requires a **real CUDA GPU** (`-lcuda`, `cuLaunchKernel`,
> `cuModuleLoadData`) and is **separate** from `scripts/reproduce_trust.sh`, which is CPU-only and
> rebuilds the trust chain without touching the GPU. Run the capstone only on a CUDA host; match the
> literal token `CAPSTONE_AUDIT_PASS`. Do not conflate the two — a green `reproduce_trust.sh` does
> *not* exercise the GPU path.

> **Residual:** the capstone proves the emitted kernels are **correct**. It does not prove
> performance parity with cuBLAS — GPU throughput is a fraction of cuBLAS and the end-to-end speedup
> is 7.0–8.7× (Amdahl-bound), quantified in the next chapter and disclosed in
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R residuals 1–2. And the whole
> GPU path is **complete to PTX, not SASS**: below PTX it trusts NVIDIA's closed `ptxas`, the CUDA
> driver, the GPU hardware, and this very C launcher
> ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R residuals 7–8).

---

## What this chapter established

From a thin Helix wrapper over a fused emitter intrinsic, `kovc` produces GEMMs at four tiers — naive,
SMEM-tiled, `cp.async` double-buffered (the winning f32 tier), and a TF32 Tensor-Core attempt that is
proven correct but not the performance path on this hardware. Those GEMMs, plus a complete forward +
backward op set (softmax, layernorm, GELU, Adam, the cross-entropy gradient, and fused attention),
are assembled by a trusted C launcher into a 2-layer transformer that trains end-to-end on
`kovc`-emitted PTX. The run is gated on loss parity against an independent numpy oracle (within 2%,
reproduced at ~0.0009%), backed by a sampled finite-difference gradient check and a load-bearing
negative control — the standing proof being `CAPSTONE_AUDIT_PASS` from
[`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh) on a CUDA host. The capstone proves
**correctness**, deliberately and explicitly *not* performance parity.

---

**Next:** [Honest performance & the PTX boundary](03-honest-performance.md) puts numbers on the
performance story — the measured GEMM TFLOP/s and cuBLAS fractions, the 7.0–8.7× Amdahl-bound
end-to-end speedup, why TF32 is a dead end on this GPU, and the precise sense in which the GPU path
is **complete to PTX, not to GPU machine code**.
