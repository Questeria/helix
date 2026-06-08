// GPU LayerNorm forward WITH affine AND epsilon (P5 GPT-2 gap kernel). Hand-written copy of the naive
// gpu_layernorm_fwd_save body (helixc/examples/gpu_layernorm_fwd_save_kernel.hx, which has affine but
// NO eps) with one change: __gpu_rsqrt(var) -> __gpu_rsqrt(var + 0.00001) (GPT-2 layer_norm_epsilon
// = 1e-5). NOT an edit of the fused __layernorm_fwd_save_blockred intrinsic in kovc.hx (so the
// self-host fixpoint 0992dddd is untouched). Drops the inference-irrelevant `ist` save (5 params).
//   y[row,c] = gamma[c]*(x[row,c]-mean)*rsqrt(var+eps) + beta[c]
// Biased/population variance (divide by cols) to match numpy x.var() and layer_norm_f32 (nn.hx:680).
// One thread per row (gridDim.x=rows, blockDim.x=1). Launch: cuda_launch out.ptx gpu_layernorm_fwd_eps <N> layernorm_eps <rows> <cols>
@kernel
fn gpu_layernorm_fwd_eps(x: f32, y: f32, gamma: f32, beta: f32, cols: i32) {
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
    let inv = __gpu_rsqrt(var + 0.00001);
    let mut t = 0;
    while t < cols { y[base + t] = gamma[t] * ((x[base + t] - mean) * inv) + beta[t]; t = t + 1 }
}
