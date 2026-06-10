// GPU GEMV A.Bt (KV-CACHE DECODE kernel 1 of 3; AUTHORED for the fast-decode leg --
// NEEDS GPU BUILD + PARITY GATE before any claim). The tiled_matmul_abt requires
// M%64==0; incremental decoding computes ONE row (M=1), so this is the M=1 form:
//   y[n] = sum_j x[j] * w[n*k + j]        (x: [1,k], w: [n_out, k] UN-TRANSPOSED, y: [1,n_out])
// Covers every decode-step GEMM (q/k/v/o, gate/up/down, the tied head) AND the
// attention scores against the cached K (w = Kcache [t, head_dim], k = head_dim,
// n_out = t -- n_out and k are BOTH arbitrary, no %64 constraint: one thread per
// output element with a serial j-loop, the gpu_rmsnorm accumulator idiom).
// Mathematically the SAME dot products as tiled_matmul_abt -- the token-for-token
// gate must prove the KV path emits IDENTICAL ids to the full re-forward.
// One thread per output (gridDim.x = n_out, blockDim.x = 1).
// Launch: cuda_launch out.ptx gpu_gemv_abt <n_out> gemv_abt <k>
@kernel
fn gpu_gemv_abt(x: f32, w: f32, y: f32, k: i32) {
    let n = block_idx() * block_dim() + thread_idx();
    let base = n * k;
    let mut acc = x[0] - x[0];
    let mut j = 0;
    while j < k {
        acc = acc + x[j] * w[base + j];
        j = j + 1
    };
    y[n] = acc
}
