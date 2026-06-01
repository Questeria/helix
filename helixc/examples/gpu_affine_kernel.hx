// GPU affine probe (P5) -- the FIRST kernel to use f32 LITERALS in PTX.
// y[i] = x[i] * 0.5 + 0.25. The two literals exercise the new AST_FLOATLIT
// (tag 27) arm in emit_ptx_expr, which emits "mov.f32 %f, 0fXXXXXXXX;" with
// the IEEE-754 hex from parse_float_bits: 0.5 -> 0f3F000000, 0.25 -> 0f3E800000.
// One thread per element (gridDim.x=N, blockDim.x=1, no bounds guard since we
// launch exactly N threads). x/y are f32 array(pointer) params, n is i32.
// Validated vs a CPU reference 0.5*x+0.25.
// Launch: cuda_launch out.ptx gpu_affine <N> affine
@kernel
fn gpu_affine(x: f32, y: f32, n: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    y[i] = x[i] * 0.5 + 0.25
}
