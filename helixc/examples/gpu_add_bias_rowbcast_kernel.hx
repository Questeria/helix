// GPU row-broadcast bias add (P5 GPT-2 gap kernel). y[i] += bias[i mod cols], one thread per element
// (launch like vector_add: gridDim.x*blockDim.x >= n, with an `i < n` bounds guard). Dimension-generic
// over cols in {2304 (qkv), 768 (proj/embd), 3072 (mlp_fc)}. Computes `i mod cols` as i-(i/cols)*cols
// (integer div+mul+sub) to avoid relying on a '%' operator. Standalone hand-written kernel; NO change
// to kovc.hx, so the self-host fixpoint 0992dddd is untouched.
//
// GUARD FORM: the `i < n` bound is expressed as a `while`-loop that runs AT MOST ONCE (a `done` flag
// flips after the single store, so the loop body executes only when i<n and never twice). This is
// semantically identical to `if i < n { y[i]+=bias[i mod cols] }` but ends the function on a `while`
// statement rather than a bare trailing `if`. The kovc PTX emitter mis-lowers a bare `if` used as the
// final statement of a @kernel body (it reaches for the then-branch's "value", and a store-statement
// has none -> it emits an invalid `mov.s32 %r12, %r-1`, which the ptxas/driver JIT rejects with a
// syntax error near '-'). The committed naive kernels (vector_add, gpu_scale_inplace) dodge this by
// omitting the guard entirely (exact launch geometry); GPT-2 over-launches the last block when n is
// not a multiple of the block size, so the guard is load-bearing here and is kept via this while-form.
// Launch: cuda_launch out.ptx gpu_add_bias_rowbcast <n> add_bias <n> <cols>
@kernel
fn gpu_add_bias_rowbcast(y: f32, bias: f32, n: i32, cols: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    let mut active = 0;
    if i < n { active = 1 };
    while active > 0 {
        let r = i - (i / cols) * cols;
        y[i] = y[i] + bias[r];
        active = 0
    }
}
