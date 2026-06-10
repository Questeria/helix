// GPU fused SiLU-gate multiply, NUMERICALLY STABLE (LLAMA-ARCH op 3 of 4; AUTHORED for
// the modern-model leg -- NEEDS GPU BUILD + PARITY GATE IN CLAUDE CODE before any claim).
// The SwiGLU MLP's elementwise half:  y[i] = silu(g[i]) * u[i],  silu(x) = x*sigmoid(x),
// where g = x@W_gate and u = x@W_up come from the existing tiled_matmul (SwiGLU needs NO
// new GEMM -- only this gate). Sigmoid uses the SAME overflow-safe pattern as
// gpu_gelu_stable_kernel.hx: the exp argument is ALWAYS <= 0, so __gpu_exp never
// overflows f32 at Llama-scale pre-activations:
//   e = exp(-|g|);  g >= 0 -> sigmoid = 1/(1+e);   g < 0 -> sigmoid = e/(1+e)
// (the two `if`s select |g| and the numerator; no else-branch, matching the gelu_stable
// sign/abs idiom). Large +g -> e->0 -> sigmoid->1; large -g -> e->0 -> sigmoid->0:
// graceful saturation matching numpy. Uses ONLY already-gated emitter features
// (__gpu_exp + arithmetic + if) -- NO kovc.hx change; the self-host fixpoint 0992dddd is
// untouched. One thread per element (gridDim.x=N, blockDim.x=1, blocks*threads == n).
// Ends on a store. Oracle: llama_ops_numpy_ref.py silu_mul().
// Launch: cuda_launch out.ptx gpu_silu_mul <N> silu_mul
@kernel
fn gpu_silu_mul(g: f32, u: f32, y: f32, n: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    let gi = g[i];
    let mut amag = gi;
    if gi < 0.0 { amag = 0.0 - gi };
    let e = __gpu_exp(0.0 - amag);
    let mut num = 1.0;
    if gi < 0.0 { num = e };
    let sg = num / (1.0 + e);
    y[i] = gi * sg * u[i]
}
