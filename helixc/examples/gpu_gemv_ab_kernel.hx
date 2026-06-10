// GPU GEMV A.B (KV-CACHE DECODE kernel 2 of 3; AUTHORED for the fast-decode leg --
// NEEDS GPU BUILD + PARITY GATE before any claim). The probs x Vcache product of one
// decode step: a [1,tlen] row of attention probabilities against the cached V slab
// [tlen, ncols] (row-major, UN-transposed):
//   y[n] = sum_t p[t] * m[t*ncols + n]      (p: [1,tlen], m: [tlen,ncols], y: [1,ncols])
// tlen (context length so far) and ncols (head_dim) are BOTH arbitrary -- no %64
// constraint. Same accumulator idiom as gpu_rmsnorm / gpu_gemv_abt; mathematically
// the same dot products as tiled_matmul on a single row.
// One thread per output column (gridDim.x = ncols, blockDim.x = 1).
// Launch: cuda_launch out.ptx gpu_gemv_ab <ncols> gemv_ab <tlen> <ncols>
@kernel
fn gpu_gemv_ab(p: f32, m: f32, y: f32, tlen: i32, ncols: i32) {
    let n = block_idx() * block_dim() + thread_idx();
    let mut acc = p[0] - p[0];
    let mut t = 0;
    while t < tlen {
        acc = acc + p[t] * m[t * ncols + n];
        t = t + 1
    };
    y[n] = acc
}
