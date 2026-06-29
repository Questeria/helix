// v1.9 P5 (STE ternary QAT): straight-through clip mask. dw[r,c] *= 1[|w[r,c]/sc[r]| <= 1] -- pass the
// grad where the latent is in-range, zero it where saturated. compare/select, generic @kernel path, NO
// kovc.hx edit. In-place on dw. grid=rows, block=cols.
@kernel
fn ste_mask(dw: f32, w: f32, sc: f32, rows: i32, cols: i32) {
    let row = block_idx();
    let col = thread_idx();
    let idx = row * cols + col;
    let q = w[idx] / sc[row];
    let aq = if q < 0.0 { 0.0 - q } else { q };
    let m = if aq > 1.0 { 0.0 } else { 1.0 };
    dw[idx] = dw[idx] * m
}
