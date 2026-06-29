// v1.9 P5 (STE ternary QAT): forward ternarize-dequantize. wt[r,c] = clip3(w[r,c]/sc[r]) * sc[r], where
// clip3(q) = +1 if q>0.5, -1 if q<-0.5, else 0 -- the 3-level ternary quantize as COMPARE thresholds
// (additive form: pos-neg, two if/else selects; setp/selp, generic @kernel path; NO round, NO __gpu_f2i,
// NO kovc.hx edit). grid=rows, block=cols.
@kernel
fn ternarize_dequant(w: f32, sc: f32, wt: f32, rows: i32, cols: i32) {
    let row = block_idx();
    let col = thread_idx();
    let s = sc[row];
    let half = 0.5;
    let nhalf = 0.0 - 0.5;
    let q = w[row * cols + col] / s;
    let pos = if q > half { 1.0 } else { 0.0 };
    let neg = if q < nhalf { 1.0 } else { 0.0 };
    let t = pos - neg;
    wt[row * cols + col] = t * s
}
