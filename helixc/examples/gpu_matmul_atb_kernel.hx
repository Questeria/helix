// GPU A^T @ B (P5, backward workhorse) -- the THIRD matmul variant, completing
// the set: naive_matmul = A@B, gpu_qkt = A@B^T, this = A^T@B. It is the weight
// gradient dW = X^T @ dY in linear-layer backprop. Given A[M,K] and B[M,N] (both
// M rows), computes C[K,N] with C[i,j] = sum_t A[t,i]*B[t,j]. One thread per output
// cell: i=block_idx() in [0,K), j=thread_idx() in [0,N) (launch gridDim.x=K,
// blockDim.x=N). A[t,i]=a[t*kk + i] (kk=K is A's row stride); B[t,j]=b[t*nn + j]
// (nn=N is B's row stride); C[i,j]=c[i*nn + j]. Accumulator seeds with the t=0
// product (a[0*kk+i]=a[i], b[0*nn+j]=b[j]) to dodge an f32-literal zero, mirroring
// naive_matmul. 6 params (at the cap), 4 let-vars. Only already-gated emitter
// features -> NO kovc.hx change. Validated cell-by-cell vs a CPU A^T@B (exact,
// integer inputs). Launch: cuda_launch out.ptx gpu_matmul_atb <N> matmul_atb M K N
@kernel
fn gpu_matmul_atb(a: f32, b: f32, c: f32, mm: i32, kk: i32, nn: i32) {
    let i = block_idx();
    let j = thread_idx();
    let mut acc = a[i] * b[j];
    let mut t = 1;
    while t < mm {
        acc = acc + a[t * kk + i] * b[t * nn + j];
        t = t + 1
    };
    c[i * nn + j] = acc
}
