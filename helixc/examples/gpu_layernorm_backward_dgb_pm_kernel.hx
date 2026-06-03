// GPU corpus kernel (T2/M6 fusion): LayerNorm backward gamma/beta gradients using a
// PRE-COMPUTED per-row mean (the "_pm" = pre-mean variant of gpu_layernorm_backward_dgb).
//   dgamma[c] = sum_s dy[s,c]*xhat[s,c];  dbeta[c] = sum_s dy[s,c]
//   xhat[s,c] = (x[s,c]-mean[s])*ist[s]    (mean[s] READ from the buffer, NOT recomputed)
// One thread per column (gridDim.x=cols, blockDim.x=1). Reading mean[s] removes the inner
// O(cols) mean recompute the original kernel did per column -> O(rows*cols) total instead of
// O(rows*cols^2). Outputs PACKED into dgb[2*cols] (dgb[c]=dgamma, dgb[cols+c]=dbeta) to stay
// at 6 params. Pure Helix (no kovc.hx change); identical MATH to gpu_layernorm_backward_dgb,
// so the finite-diff gradient check + the 2% oracle parity remain the correctness gate.
// 7 distinct let-vars (under the 12 cap). Pair with gpu_row_mean (fills mean[]).
// Launch: cuda_launch out.ptx gpu_layernorm_backward_dgb_pm <cols> layernorm_bwd_dgb_pm <rows> <cols>
@kernel
fn gpu_layernorm_backward_dgb_pm(x: f32, dy: f32, ist: f32, mean: f32, dgb: f32, rows: i32, cols: i32) {
    let c = block_idx();
    let mut dg = x[c] - x[c];
    let mut db = x[c] - x[c];
    let mut s = 0;
    while s < rows {
        let base = s * cols;
        let xh = (x[base + c] - mean[s]) * ist[s];
        dg = dg + dy[base + c] * xh;
        db = db + dy[base + c];
        s = s + 1
    };
    dgb[c] = dg;
    dgb[cols + c] = db
}
