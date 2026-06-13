// GPU corpus kernel (v1.5 S0 increment 2): TERNARY matrix multiply C = A*B on the
// GPU, proving the first-class ternary type `t2` (tag 12, BitNet b1.58, values
// -1/0/+1, scalar domain i32) is usable end-to-end ON THE GPU -- not just a CPU-side
// type label. a and b are declared `t2` (ternary), c is the i32 accumulator output.
//
// Shape mirrors naive_matmul_kernel.hx exactly (one thread per output cell:
// row=block_idx(), col=thread_idx(); launched gridDim.x=M, blockDim.x=N), but the
// element type is `t2` instead of f32. Because t2's scalar domain is i32, a bare t2
// array load lowers to ld.global.u32 (the integer load path) and t2*t2 lowers to
// mul.lo.s32 + add.s32 -- a fully EXACT integer accumulation (no rounding), verified
// element-for-element against an independent numpy int oracle by capstone_audit.sh.
// The emitted PTX is therefore correct integer code with ZERO kovc.hx change (the
// self-host fixpoint is untouched); this kernel is gated by a byte-exact .ref.ptx.
//
// SCOPE (honest): this is a NAIVE one-thread-per-cell matmul that uses mul.lo.s32 for
// the products. The genuinely ternary-DISTINCT optimizations -- replacing the multiply
// with branch-free add/subtract by sign (BitNet's no-multiply property) and 2-bit
// packed weight storage -- are a LATER increment, NOT claimed here. The precondition
// (unchecked) is that a/b hold only -1/0/+1; with general i32 inputs this is a plain
// integer matmul. mm (M) is unused (row comes from block_idx), matching naive_matmul.
//
// Validated cell-by-cell vs a numpy int oracle by:
//   cuda_launch out.ptx ternary_matmul <N> ternary_matmul <M> <K> <N>
@kernel
fn ternary_matmul(a: t2, b: t2, c: i32, mm: i32, kk: i32, nn: i32) {
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
