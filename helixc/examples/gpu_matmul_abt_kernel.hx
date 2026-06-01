// GPU A @ B^T, UNSCALED (P5) -- the unscaled sibling of gpu_qkt (which bakes the
// 0.25 attention scale). Needed for d_attn = dOut @ V^T in attention backward.
// Given A[M,K] and B[N,K] (both K cols), computes C[M,N] with C[i,j]=sum_t A[i,t]*B[j,t]
// (B accessed transposed: b[j*kk+t]=B[j,t]). One thread per output cell: i=block_idx()
// in [0,M), j=thread_idx() in [0,N) (launch gridDim.x=M, blockDim.x=N). A[i,t]=a[i*kk+t]
// (kk=K is A row stride), B[j,t]=b[j*kk+t] (kk=K is B row stride), C[i,j]=c[i*nn+j]
// (nn=N is C row stride). Accumulator seeds with t=0 (a[i*kk]*b[j*kk]). 6 params (at
// cap), 4 let-vars. NO kovc.hx change. Validated cell-by-cell vs a CPU A@B^T (exact,
// integer inputs). Launch: cuda_launch out.ptx gpu_matmul_abt <N> matmul_abt M K N
@kernel
fn gpu_matmul_abt(a: f32, b: f32, c: f32, mm: i32, kk: i32, nn: i32) {
    let i = block_idx();
    let j = thread_idx();
    let mut acc = a[i * kk] * b[j * kk];
    let mut t = 1;
    while t < kk {
        acc = acc + a[i * kk + t] * b[j * kk + t];
        t = t + 1
    };
    c[i * nn + j] = acc
}
