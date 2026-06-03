// GPU corpus kernel (T2/M6): block-reduction LayerNorm FORWARD + save inv_std. The
// block-per-row (256-thread) sibling of the naive one-thread-per-row
// gpu_layernorm_fwd_save. A single FUSED intrinsic __layernorm_fwd_save_blockred emits
// the WHOLE kernel body: it reuses emit_ptx_layernorm_blockred (block-reduce row MEAN +
// VAR with SMEM tree reductions, then write y=gamma*(x-mean)*inv+beta) and ADDS the
// ist[row]=inv store the backward pass needs. See emit_ptx_layernorm_fwd_save_blockred
// in kovc.hx. NO eps (matches the naive kernel + CPU ref). inv via rsqrt.approx.
// Launch: grid=(rows,1,1), block=(256,1,1) -> row=ctaid.x. cols any >=1.
//   cuda_launch out.ptx layernorm_fwd_save_blockred 0 layernorm_save <rows> <cols>
@kernel
fn layernorm_fwd_save_blockred(x: f32, y: f32, gamma: f32, beta: f32, ist: f32, cols: i32) {
    __layernorm_fwd_save_blockred(x, y, gamma, beta, ist, cols)
}
