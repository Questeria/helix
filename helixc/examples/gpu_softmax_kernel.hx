// GPU softmax (P5, row-wise) -- the first real REDUCTION op. One thread per row
// (launch gridDim.x=rows, blockDim.x=1, so row=block_idx() is always valid -- no
// bounds guard needed, which keeps the body a plain sequence). Three serial passes:
//   1. max:       m = max_j x[row,j]   (the if-STATEMENT path: if x>m { m=x })
//   2. exp+sum:   y[row,j] = e^(x[row,j]-m);  s = sum_j y[row,j]   (uses __gpu_exp)
//   3. normalize: y[row,j] = y[row,j] / s     (div.rn.f32)
// `s` is initialised to x[base]-x[base] (=0.0) to avoid an f32 literal; `rows` is
// unused in the body (it is the grid dim). Validated vs CPU softmax (nn.hx:789);
// each row must sum to ~1. Launch: cuda_launch out.ptx gpu_softmax <N> softmax rows cols
@kernel
fn gpu_softmax(x: f32, y: f32, rows: i32, cols: i32) {
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
