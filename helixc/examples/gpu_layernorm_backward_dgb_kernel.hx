// GPU LayerNorm backward -- gamma/beta gradients (P5). Per-column reduction over rows:
//   dgamma[c] = sum_s dy[s,c]*xhat[s,c];   dbeta[c] = sum_s dy[s,c]
//   xhat[s,c] = (x[s,c]-mean_s)*ist[s]     (mean_s recomputed per row inline)
// One thread per column (gridDim.x=cols, blockDim.x=1). Outputs are PACKED into one
// buffer dgb[2*cols] to stay at 6 params: dgb[c]=dgamma[c], dgb[cols+c]=dbeta[c].
// 10 distinct let-vars (under the 12 cap; base/msum/mean/xh are loop-body lets emitted
// once). NO kovc.hx change. Verified vs a central finite-difference of the layernorm
// FORWARD wrt gamma/beta (independent of this analytic formula).
// Launch: cuda_launch out.ptx gpu_layernorm_backward_dgb <N> layernorm_bwd_dgb <rows> <cols>
@kernel
fn gpu_layernorm_backward_dgb(x: f32, dy: f32, ist: f32, dgb: f32, rows: i32, cols: i32) {
    let c = block_idx();
    let colsf = __gpu_i2f(cols);
    let mut dg = x[c] - x[c];
    let mut db = x[c] - x[c];
    let mut s = 0;
    while s < rows {
        let base = s * cols;
        let mut msum = x[base] - x[base];
        let mut j = 0;
        while j < cols { msum = msum + x[base + j]; j = j + 1 };
        let mean = msum / colsf;
        let xh = (x[base + c] - mean) * ist[s];
        dg = dg + dy[base + c] * xh;
        db = db + dy[base + c];
        s = s + 1
    };
    dgb[c] = dg;
    dgb[cols + c] = db
}
