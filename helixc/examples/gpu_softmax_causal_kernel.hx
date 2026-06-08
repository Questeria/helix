// GPU CAUSAL softmax (P5 GPT-2 gap kernel). Hand-written copy of the naive gpu_softmax body
// (helixc/examples/gpu_softmax_kernel.hx) with a causal mask folded in -- NOT an edit of the
// fused __softmax_blockred intrinsic in kovc.hx (so the self-host fixpoint 0992dddd is untouched).
// One thread per query row (launch gridDim.x=rows=S, blockDim.x=1 so row=block_idx() is valid).
// For query row i, reduce over keys j in [0..=i] ONLY and write y[i,j]=0 for j>i -- numerically
// identical to adding -inf to masked scores, with no -inf literal. Scores must already carry the
// 1/sqrt(64)=0.125 scale (applied by gpu_scale_rt before this kernel). `rows` is the grid dim (unused
// in the body); `cols` = S (total keys). `0.0` is written as x[base]-x[base] (no f32 literal needed).
// Launch: cuda_launch out.ptx gpu_softmax_causal <N> softmax_causal <rows> <cols>
@kernel
fn gpu_softmax_causal(x: f32, y: f32, rows: i32, cols: i32) {
    let row = block_idx() * block_dim() + thread_idx();
    let base = row * cols;
    let nvalid = row + 1;
    let mut m = x[base];
    let mut j = 1;
    while j < nvalid {
        if x[base + j] > m { m = x[base + j] };
        j = j + 1
    };
    let mut s = x[base] - x[base];
    let mut k = 0;
    while k < nvalid {
        let e = __gpu_exp(x[base + k] - m);
        y[base + k] = e;
        s = s + e;
        k = k + 1
    };
    let mut z = nvalid;
    while z < cols {
        y[base + z] = x[base] - x[base];
        z = z + 1
    };
    let mut t = 0;
    while t < nvalid {
        y[base + t] = y[base + t] / s;
        t = t + 1
    }
}
