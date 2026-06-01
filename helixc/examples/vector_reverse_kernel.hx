// GPU corpus kernel #3 (P5 Step A edit-2 test): c[i] = a[n - 1 - i].
// This is the FIRST kernel that READS a scalar param: n (an i32 dim) is used in
// the integer index expression n-1-i, exercising the new ld.param.u32 scalar-load
// path (kovc emit_ptx_expr AST_VAR fallback). No f32 mixing, no accumulator -- it
// isolates the scalar-read fix before the matmul. With host a[i]=i and n=256,
// c[i]=a[255-i]=255-i, so c[7]=248.
// Verified by: cuda_launch /tmp/out.ptx vector_reverse 256 reverse
@kernel
fn vector_reverse(a: f32, b: f32, c: f32, n: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    c[i] = a[n - 1 - i]
}
