// GPU corpus kernel #4 (P5 Step B): naive matrix multiply C = A*B on the GPU --
// the FIRST real compute kernel. One thread per output cell: row=block_idx(),
// col=thread_idx() (launched gridDim.x=M, blockDim.x=N). Each thread runs a
// k-loop accumulating A[row,k]*B[k,col] into an f32 accumulator.
//
// The accumulator is initialised to the k=0 product (a[row*K]*b[col]) instead of
// 0.0_f32 -- this sidesteps an f32-literal codegen path and needs only the f32
// accumulator-assignment (mov.f32) edit. Exercises: scalar-param reads (K=kk loop
// bound, N=nn stride, via ld.param.u32), computed global indices, mul.f32 + add.f32,
// the f32 while-loop accumulator, and an f32 global store.
//
// mm (M) is unused in the body (row comes straight from block_idx). Validated
// cell-by-cell vs a CPU reference by: cuda_launch out.ptx naive_matmul <N> matmul M K N
@kernel
fn naive_matmul(a: f32, b: f32, c: f32, mm: i32, kk: i32, nn: i32) {
    let row = block_idx();
    let col = thread_idx();
    let mut acc = a[row * kk] * b[col];
    let mut t = 1;
    while t < kk {
        acc = acc + a[row * kk + t] * b[t * nn + col];
        t = t + 1
    };
    c[row * nn + col] = acc
}
