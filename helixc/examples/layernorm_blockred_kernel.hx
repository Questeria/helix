// GPU corpus kernel (T2/M4): WARP/BLOCK-REDUCTION row LayerNorm. The block-per-row
// sibling of the naive one-thread-per-row gpu_layernorm. A single FUSED intrinsic
// __layernorm_blockred emits the WHOLE kernel body: a 256-thread block per row that
// (1) block-reduces the row MEAN (strided per-thread sum + SMEM tree reduce with
// bar.sync), (2) block-reduces the row VARIANCE (sum of (x-mean)^2) then inv=
// rsqrt(var), and (3) writes y[r,c]=gamma[c]*(x[r,c]-mean)*inv + beta[c]. See
// emit_ptx_layernorm_blockred in kovc.hx.
//
// NO eps (matches the naive kernel + CPU ref; non-degenerate rows have var>0).
// inv uses rsqrt.approx; the resulting tol vs the CPU 1/sqrtf reference is
// validated in scripts/gpu_layernorm_blockred_corpus.sh. The mean + var
// reduction primitives here are reused by the fused-attention milestone.
//
// Launch: grid=(rows,1,1), block=(256,1,1) -> row=ctaid.x. cols any >=1.
//   cuda_launch out.ptx layernorm_blockred 0 layernorm_perf <rows> <cols>
@kernel
fn layernorm_blockred(x: f32, y: f32, gamma: f32, beta: f32, cols: i32) {
    __layernorm_blockred(x, y, gamma, beta, cols)
}
