// GPU cross-entropy + softmax backward (P5, the backprop ROOT). Computes the
// gradient of softmax-cross-entropy loss wrt the logits, which is the clean fused
// form  dlogits[r,c] = softmax(logits[r,:])[c] - onehot(target[r])[c].
// One thread per row (gridDim.x=rows, blockDim.x=1). Three serial passes mirror
// gpu_softmax: (1) row max, (2) exp+sum, (3) normalize and subtract the one-hot.
// TARGET is passed as an f32 array (tgtf[r] = the class index as a float) -- the
// emitter has no i32-array param, so the integer target is carried as f32 and the
// one-hot is computed BRANCHLESSLY: (__gpu_i2f(t) == tgtf[r]) yields an i32 0/1,
// then __gpu_i2f maps it to 0.0/1.0 (exact for class indices < 2^24). s init via
// x-x (=0). Mirrors stdlib softmax_ce_grad_f32 (nn.hx). 5 params, 10 let-vars.
// Verified vs a CPU softmax-minus-onehot reference AND the conservation property
// that each grad row sums to ~0 (sum_c (p_c - onehot_c) = 1 - 1 = 0).
// Launch: cuda_launch out.ptx gpu_ce_softmax_grad <N> ce_softmax_grad <rows> <cols>
@kernel
fn gpu_ce_softmax_grad(logits: f32, tgtf: f32, dlogits: f32, rows: i32, cols: i32) {
    let row = block_idx() * block_dim() + thread_idx();
    let base = row * cols;
    let mut m = logits[base];
    let mut j = 1;
    while j < cols {
        if logits[base + j] > m { m = logits[base + j] };
        j = j + 1
    };
    let mut s = logits[base] - logits[base];
    let mut k = 0;
    while k < cols {
        s = s + __gpu_exp(logits[base + k] - m);
        k = k + 1
    };
    let tg = tgtf[row];
    let mut t = 0;
    while t < cols {
        let p = __gpu_exp(logits[base + t] - m) / s;
        let eqi = __gpu_i2f(t) == tg;
        dlogits[base + t] = p - __gpu_i2f(eqi);
        t = t + 1
    }
}
