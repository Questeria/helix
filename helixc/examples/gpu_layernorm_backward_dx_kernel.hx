// GPU LayerNorm backward -- input gradient dx (P5). Normalizes over cols=D per row.
// Given x, upstream dy, gamma, and ist=1/std (saved by gpu_layernorm_fwd_save):
//   xhat[c]  = (x[c]-mean)*ist          (mean recomputed from x)
//   dxhat[c] = dy[c]*gamma[c]
//   m1 = mean_c(dxhat);  m2 = mean_c(dxhat*xhat)
//   dx[c] = ist*(dxhat[c] - m1 - xhat[c]*m2)
// One thread per row (gridDim.x=rows, blockDim.x=1). Three passes (mean; then m1,m2;
// then dx), REUSING the loop counter j across passes (j=0 resets) and computing
// dxhat/xhat INLINE to fit 11 distinct let-vars under the 12 cap. 6 params (at cap).
// NO kovc.hx change. Verified vs a central finite-difference of the layernorm FORWARD
// (independent of this analytic formula) AND the conservation law sum_c dx ~ 0.
// Launch: cuda_launch out.ptx gpu_layernorm_backward_dx <N> layernorm_bwd_dx <rows> <cols>
@kernel
fn gpu_layernorm_backward_dx(x: f32, dy: f32, gamma: f32, ist: f32, dx: f32, cols: i32) {
    let row = block_idx() * block_dim() + thread_idx();
    let base = row * cols;
    let istv = ist[row];
    let colsf = __gpu_i2f(cols);
    let mut acc = x[base] - x[base];
    let mut j = 0;
    while j < cols { acc = acc + x[base + j]; j = j + 1 };
    let mean = acc / colsf;
    let mut a1 = x[base] - x[base];
    let mut a2 = x[base] - x[base];
    j = 0;
    while j < cols {
        a1 = a1 + dy[base + j] * gamma[j];
        a2 = a2 + (dy[base + j] * gamma[j]) * ((x[base + j] - mean) * istv);
        j = j + 1
    };
    let m1 = a1 / colsf;
    let m2 = a2 / colsf;
    j = 0;
    while j < cols {
        dx[base + j] = istv * (dy[base + j] * gamma[j] - m1 - ((x[base + j] - mean) * istv) * m2);
        j = j + 1
    }
}
