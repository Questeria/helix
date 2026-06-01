// GPU softmax backward (P5) -- the Jacobian-vector product. Given the saved softmax
// output P[rows,cols] and upstream gradient dP[rows,cols], computes
//   dA[i,j] = P[i,j] * (dP[i,j] - sum_k dP[i,k]*P[i,k])
// One thread per row (gridDim.x=rows, blockDim.x=1). Two passes: (1) accumulate the
// row dot product dot = sum_k dP*P; (2) write dA[i,k] = P[i,k]*(dP[i,k] - dot). The dot
// accumulator inits via p-p (=0). 5 params (p, dp, da, rows, cols; rows unused = grid
// dim), 5 let-vars (row, base, dot, j, k). Only gated emitter features -> NO kovc.hx
// change. Verified vs a CPU reference AND the INDEPENDENT conservation property that
// each dA row sums to ~0 (sum_j dA = sum_j P*dP - (sum_j P)*dot = dot - 1*dot = 0,
// since sum_j P = 1). Launch: cuda_launch out.ptx gpu_softmax_backward <N> softmax_backward <rows> <cols>
@kernel
fn gpu_softmax_backward(p: f32, dp: f32, da: f32, rows: i32, cols: i32) {
    let row = block_idx() * block_dim() + thread_idx();
    let base = row * cols;
    let mut dot = p[base] - p[base];
    let mut j = 0;
    while j < cols {
        dot = dot + dp[base + j] * p[base + j];
        j = j + 1
    };
    let mut k = 0;
    while k < cols {
        da[base + k] = p[base + k] * (dp[base + k] - dot);
        k = k + 1
    }
}
