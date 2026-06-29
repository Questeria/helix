// v1.9 P5 (STE ternary QAT): per-row abs-mean scale. sc[s] = (sum_c |w[s,c]|) / cols. One thread per row
// (grid=rows, block=1). abs via (v<0)?-v:v -- compare/select, generic @kernel path, NO kovc.hx edit.
// Models gpu_row_mean (zero-init w[base]-w[base]; __gpu_i2f(cols)).
@kernel
fn row_abs_mean(w: f32, sc: f32, rows: i32, cols: i32) {
    let s = block_idx() * block_dim() + thread_idx();
    let base = s * cols;
    let colsf = __gpu_i2f(cols);
    let mut msum = w[base] - w[base];
    let mut j = 0;
    while j < cols {
        let v = w[base + j];
        let av = if v < 0.0 { 0.0 - v } else { v };
        msum = msum + av;
        j = j + 1
    };
    sc[s] = msum / colsf
}
