// GPU corpus kernel (T2/M6): block-reduction softmax BACKWARD (the Jacobian-vector
// product). The block-per-row (256-thread) sibling of the naive one-thread-per-row
// gpu_softmax_backward. A single FUSED intrinsic __softmax_backward_blockred emits the
// WHOLE kernel body: (1) block-reduce the row dot product dot=sum_k dP[r,k]*P[r,k] (a
// strided per-thread partial + SMEM tree reduce with bar.sync), then (2) write
// dA[r,j]=P[r,j]*(dP[r,j]-dot). See emit_ptx_softmax_backward_blockred in kovc.hx.
// Reuses the softmax block-reduction reduce primitive verbatim. Conservation: each dA
// row sums to ~0 (sum_j P*dP - (sum_j P)*dot = dot - dot = 0).
// Launch: grid=(rows,1,1), block=(256,1,1) -> row=ctaid.x. cols any >=1.
//   cuda_launch out.ptx softmax_backward_blockred 0 softmax_backward <rows> <cols>
@kernel
fn softmax_backward_blockred(p: f32, dp: f32, da: f32, rows: i32, cols: i32) {
    __softmax_backward_blockred(p, dp, da, rows, cols)
}
