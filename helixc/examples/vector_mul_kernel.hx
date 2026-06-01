// GPU corpus kernel #2 (P5): elementwise multiply c[i] = a[i] * b[i] over f32
// global arrays. Same concrete non-generic @kernel shape as vector_add_kernel.hx
// (grid-stride index block_idx()*block_dim()+thread_idx(); param n present-but-
// unused). Exercises the kovc PTX emitter's mul.f32 arm on global loads/stores.
// With host a[i]=i, b[i]=2*i, expect c[i]=2*i*i, so c[7]=98.
// Verified by: cuda_launch /tmp/out.ptx vector_mul 256 mul
@kernel
fn vector_mul(a: f32, b: f32, c: f32, n: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    c[i] = a[i] * b[i]
}
