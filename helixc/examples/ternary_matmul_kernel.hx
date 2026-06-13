// GPU corpus kernel (v1.5 S0 increment 2): TERNARY matrix multiply C = A*B on the GPU,
// proving the first-class type `t2` (tag 12, BitNet b1.58; scalar domain i32) is usable
// end-to-end ON THE GPU -- not just a CPU-side type label.
//
// TYPING (important + honest): `t2` is Helix's i32-DOMAIN GPU buffer type (its scalar
// domain is i32, established in S0 increment 1). All three buffers a/b/c are declared `t2`
// for a concrete ABI reason: the GPU @kernel param convention declares `i32` params as
// `.u32` SCALARS (that is how the dims mm/kk/nn are passed), so a data ARRAY/pointer param
// MUST be a non-i32 pointer-class type. A bare t2 load/store lowers to the INTEGER path
// (ld.global.u32 / st.global.u32) with t2*t2 -> mul.lo.s32 + add.s32, an EXACT integer
// accumulation (no rounding). (First cut declared c:i32 -> it became a .u32 scalar but was
// used as a pointer -> ld.param.u64 on a 4-byte slot -> illegal GPU memory access; c:t2
// makes it a proper .b64 pointer. The PTX-byte gate did not catch this -- the GPU run did.)
//
// The ternary {-1,0,+1} property is a CONVENTION on the WEIGHTS a (BitNet b1.58), enforced
// by the caller -- NOT a constraint the type enforces. b (activations) and c (output) hold
// general i32 values in t2 (i32-domain) buffers. So "ternary matmul" = ternary weights x int
// activations -> int output, exactly as in BitNet b1.58.
//
// Shape mirrors naive_matmul_kernel.hx (one thread per output cell: row=block_idx(),
// col=thread_idx(); launched gridDim.x=M, blockDim.x=N). NO kovc.hx change -> the self-host
// fixpoint is untouched; gated by a byte-exact .ref.ptx AND verified on real hardware.
//
// SCOPE (honest): NAIVE, one-thread-per-cell, mul.lo.s32 products. The genuinely
// ternary-DISTINCT optimizations -- branch-free add/subtract by sign (BitNet's no-multiply
// property) and 2-bit packed weight storage -- are a LATER increment, NOT claimed here.
// mm (M) is unused (row comes from block_idx), matching naive_matmul.
//
// Verified element-for-element EXACT (no tolerance) vs an independent CPU integer reference
// on the RTX 3070 by scripts/gpu_ternary_check.sh (cuda_launch.c 'imatmul' mode), with a
// comparator negative-control and a kernel-corruption negative-control:
//   cuda_launch out.ptx ternary_matmul <N> imatmul <M> <K> <N> [mutate]
@kernel
fn ternary_matmul(a: t2, b: t2, c: t2, mm: i32, kk: i32, nn: i32) {
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
