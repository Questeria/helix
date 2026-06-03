// GPU corpus kernel (T2/M6): block-reduction LayerNorm BACKWARD input-gradient dx. The
// block-per-row (256-thread) sibling of the naive one-thread-per-row
// gpu_layernorm_backward_dx. A single FUSED intrinsic __layernorm_backward_dx_blockred
// emits the WHOLE kernel body, REUSING the SMEM tree-reduce primitive for FOUR
// block-reductions per row: mean(x); m1=mean(dxhat); m2=mean(dxhat*xhat); then write
//   dx[c] = ist*(dxhat[c] - m1 - xhat[c]*m2)
// where xhat=(x-mean)*ist, dxhat=dy*gamma, ist read from ist[row] (saved by the forward).
// See emit_ptx_layernorm_backward_dx_blockred in kovc.hx. Matches the naive kernel + the
// CPU ref (verified vs a central finite-difference of the layernorm FORWARD; conservation
// sum_c dx ~ 0). NO eps. Launch: grid=(rows,1,1), block=(256,1,1) -> row=ctaid.x.
//   cuda_launch out.ptx layernorm_backward_dx_blockred 0 layernorm_bwd_dx <rows> <cols>
@kernel
fn layernorm_backward_dx_blockred(x: f32, dy: f32, gamma: f32, ist: f32, dx: f32, cols: i32) {
    __layernorm_backward_dx_blockred(x, dy, gamma, ist, dx, cols)
}
