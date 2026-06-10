// GPU FULL-ROW softmax (KV-CACHE DECODE kernel 3 of 3; AUTHORED for the fast-decode
// leg -- NEEDS GPU BUILD + PARITY GATE before any claim). Hand-written copy of the
// gpu_softmax_causal body MINUS the causal mask: in incremental decoding the new
// token's score row attends EVERY cached position (all positions <= t), so the full
// row is valid -- the causal structure is enforced by what is IN the cache, not by
// masking. Numerically identical to the causal kernel's row t-1 over t valid columns.
// Same max-subtract / exp / normalize idiom; `cols` = the live context length t
// (arbitrary, no %64 constraint).
// One thread per row (gridDim.x = rows, blockDim.x = 1; decode uses rows=1).
// Launch: cuda_launch out.ptx gpu_softmax_row <rows> softmax_row <rows> <cols>
@kernel
fn gpu_softmax_row(x: f32, y: f32, cols: i32) {
    let row = block_idx() * block_dim() + thread_idx();
    let base = row * cols;
    let mut m = x[base];
    let mut j = 1;
    while j < cols {
        if x[base + j] > m { m = x[base + j] };
        j = j + 1
    };
    let mut s = x[base] - x[base];
    let mut k = 0;
    while k < cols {
        let e = __gpu_exp(x[base + k] - m);
        y[base + k] = e;
        s = s + e;
        k = k + 1
    };
    let mut t = 0;
    while t < cols {
        y[base + t] = y[base + t] / s;
        t = t + 1
    }
}
