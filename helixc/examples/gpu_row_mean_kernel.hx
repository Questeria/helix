// GPU corpus kernel (T2/M6 fusion): per-row MEAN of a [rows,cols] row-major matrix.
//   mean[s] = (sum_c x[s,c]) / cols     one thread per row (gridDim.x=rows, blockDim.x=1).
// Hoists the mean recompute OUT of gpu_layernorm_backward_dgb (which recomputed it PER
// COLUMN -> O(rows*cols^2)); paired with gpu_layernorm_backward_dgb_pm it makes the dgamma/
// dbeta column-reduction O(rows*cols). Pure Helix (no kovc.hx change). 5 distinct let-vars.
// Launch: cuda_launch out.ptx gpu_row_mean <rows> rowmean <rows> <cols>
@kernel
fn gpu_row_mean(x: f32, mean: f32, rows: i32, cols: i32) {
    let s = block_idx() * block_dim() + thread_idx();
    let base = s * cols;
    let colsf = __gpu_i2f(cols);
    let mut msum = x[base] - x[base];
    let mut j = 0;
    while j < cols { msum = msum + x[base + j]; j = j + 1 };
    mean[s] = msum / colsf
}
