// GPU LayerNorm forward + save inv_std (P5). Same as gpu_layernorm but ALSO writes
// ist[row] = 1/sqrt(var_row) -- the reciprocal std the backward pass needs (so the
// backward does not recompute it). One thread per row (gridDim.x=rows, blockDim.x=1).
//   y[row,c] = gamma[c]*(x[row,c]-mean)/sqrt(var) + beta[c];  ist[row] = 1/sqrt(var)
// 6 params (x,y,gamma,beta,ist,cols -- at the cap), 11 let-vars (at the cap-12, the
// per-element temporaries inlined as in gpu_layernorm). __gpu_i2f for cols->f32 mean/var
// divides, __gpu_rsqrt for 1/sqrt(var). NO kovc.hx change. Verified y matches a CPU
// affine layernorm AND ist[row] = 1/sqrt(var_row).
// Launch: cuda_launch out.ptx gpu_layernorm_fwd_save <N> layernorm_save <rows> <cols>
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
