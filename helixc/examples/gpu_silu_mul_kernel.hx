// GPU fused SiLU-gate multiply, NUMERICALLY STABLE (LLAMA-ARCH op 3 of 4; AUTHORED for the
// modern-model leg -- NEEDS GPU BUILD + PARITY GATE IN CLAUDE CODE before any claim).
// The SwiGLU MLP's elementwise half:  y[i] = silu(g[i]) * u[i],  silu(x) = x*sigmoid(x),
// where g = x@W_gate and u = x@W_up come from the existing tiled_matmul (SwiGLU needs NO new
// GEMM -- only this gate).
//
// Sigmoid uses the PROVEN gpu_gelu_stable_kernel.hx idiom: a `mut sgn` seeded from a CONSTANT
// (never from another local), the magnitude as an IMMUTABLE amag = sgn*gi, and the stable
// identity sigmoid(g) = 0.5*(1 + sgn*(1-e)/(1+e)), e = exp(-|g|).
// CRITICAL: gi (read in the final store) is never the init-source of a reassigned mutable. An
// earlier `let mut amag = gi; if g<0 { amag = 0-gi }` form made kovc ALIAS amag onto gi, so the
// reassignment clobbered gi to |gi| and silu came out +|g|*sig*u for g<0 -- G-L0 parity caught
// it as kernel == -reference on every g<0 element (max-abs ~1.9). The exp argument is ALWAYS
// <= 0 so __gpu_exp never overflows f32; large +g -> sigmoid->1, large -g -> sigmoid->0
// (graceful saturation, matching numpy). Uses ONLY already-gated emitter features (__gpu_exp +
// arithmetic + one `if`) -- NO kovc.hx change; the self-host fixpoint 0992dddd is untouched.
// One thread per element (gridDim.x=N, blockDim.x=1, blocks*threads == n). Ends on a store.
// Oracle: llama_ops_numpy_ref.py silu_mul().
// Launch: cuda_launch out.ptx gpu_silu_mul <N> silu_mul
@kernel
fn gpu_silu_mul(g: f32, u: f32, y: f32, n: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    let gi = g[i];
    let mut sgn = 1.0;
    if gi < 0.0 { sgn = 0.0 - 1.0 };
    let amag = sgn * gi;
    let e = __gpu_exp(0.0 - amag);
    let th = sgn * ((1.0 - e) / (1.0 + e));
    let sig = 0.5 * (1.0 + th);
    y[i] = gi * sig * u[i]
}
