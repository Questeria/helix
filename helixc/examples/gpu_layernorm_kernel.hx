// GPU LayerNorm (P5, row-wise). One thread per row (gridDim.x=rows, blockDim.x=1).
// y[row,c] = gamma[c] * (x[row,c] - mean) / sqrt(var) + beta[c]. Uses __gpu_i2f
// (cols i32 -> f32 for the mean/var divides) and __gpu_rsqrt (1/sqrt(var)). NO eps
// (avoids the f32-scalar gap; non-degenerate rows have var>0). s/vs init via x-x (=0).
// 5 params (x,y,gamma,beta,cols); x/y/gamma/beta are f32 array(pointer) params, cols i32.
// NOTE: the per-element temporaries d and xn are INLINED to keep the kernel's distinct
// let-bound vars at 11 (the PTX vtab cap is 12). Validated vs a CPU layernorm; with
// gamma=1,beta=0 each row of y has mean~0, var~1.
// Launch: cuda_launch out.ptx gpu_layernorm <N> layernorm rows cols
@kernel
fn gpu_layernorm(x: f32, y: f32, gamma: f32, beta: f32, cols: i32) {
    let row = block_idx() * block_dim() + thread_idx();
    let base = row * cols;
    let colsf = __gpu_i2f(cols);
    let mut sm = x[base] - x[base];
    let mut j = 0;
    while j < cols {
        sm = sm + x[base + j];
        j = j + 1
    };
    let mean = sm / colsf;
    let mut vs = x[base] - x[base];
    let mut k = 0;
    while k < cols {
        vs = vs + (x[base + k] - mean) * (x[base + k] - mean);
        k = k + 1
    };
    let var = vs / colsf;
    let inv = __gpu_rsqrt(var);
    let mut t = 0;
    while t < cols {
        y[base + t] = gamma[t] * ((x[base + t] - mean) * inv) + beta[t];
        t = t + 1
    }
}
