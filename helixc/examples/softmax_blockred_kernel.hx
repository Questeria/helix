// GPU corpus kernel (T2/M4): WARP/BLOCK-REDUCTION row softmax. The block-per-row
// sibling of the naive one-thread-per-row gpu_softmax. A single FUSED intrinsic
// __softmax_blockred emits the WHOLE kernel body: a 256-thread block per row that
// (1) block-reduces the row MAX (strided per-thread max + SMEM tree reduce with
// bar.sync), (2) block-reduces the row SUM of e=exp(x-max) -- writing y[r,c]=e --
// and (3) normalizes y[r,c]/=sum. See emit_ptx_softmax_blockred in kovc.hx.
//
// Numerically stable: subtracts the row max before exp. exp uses ex2.approx
// (e^x = ex2(x*log2e)); the resulting tol vs the CPU expf reference is validated
// in the corpus (scripts/gpu_softmax_blockred_corpus.sh). The row max + sum
// reduction primitives here are reused by the fused-attention milestone.
//
// Launch: grid=(rows,1,1), block=(256,1,1) -> row=ctaid.x. cols may be any >=1
// (threads stride j=tid,tid+256,...). rows is the grid dim, unused in the body.
//   cuda_launch out.ptx softmax_blockred 0 softmax_perf <rows> <cols>
@kernel
fn softmax_blockred(x: f32, y: f32, rows: i32, cols: i32) {
    __softmax_blockred(x, y, rows, cols)
}
