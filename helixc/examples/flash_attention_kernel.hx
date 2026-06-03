// GPU corpus kernel (T2/M4): FUSED FLASH-STYLE ATTENTION. A single FUSED
// intrinsic __flash_attention emits the WHOLE kernel body: one 256-thread
// block per query row that streams over the K/V sequence with an ONLINE
// softmax (running max m + running denominator l + an output accumulator
// RESCALED on every new max), computing
//   out = softmax(Q @ K^T / sqrt(d)) @ V
// WITHOUT materializing the S x S scores matrix in HBM (the flash win vs the
// naive QK^T -> softmax -> @V pipeline that round-trips S x S through global
// memory twice). See emit_ptx_flash_attention in kovc.hx.
//
// It REUSES the softmax block-reduction machinery: each per-key score
// s_j = (1/sqrt(d)) * dot(Q[i,:], K[j,:]) is an SMEM-TREE-REDUCED block sum
// (the SAME emit_ptx_smem_tree_reduce __softmax_blockred uses), and the
// numerically-stable running-max rescale mirrors the softmax row-max subtract.
// scale = 1/sqrt(d) is computed at runtime (rsqrt) so the kernel is
// dimension-independent. exp uses ex2.approx (validated tol in the corpus).
//
// Layout: thread t = tid.x owns output column t (and Q[i,t]); query row
// i = ctaid.x. d must be <= 256 (the block size). Launch: grid=(S,1,1),
// block=(256,1,1). All [S,d] / out[S,d] row-major; Q,K,V same shape.
//   cuda_launch out.ptx flash_attention 0 attn_flash <S> <d>
@kernel
fn flash_attention(q: f32, k: f32, v: f32, o: f32, ss: i32, dd: i32) {
    __flash_attention(q, k, v, o, ss, dd)
}
