// GPU SiLU-gate multiply backward, NUMERICALLY STABLE (LLAMA-ARCH backward op; AUTHORED for
// the QAT trainer leg -- NEEDS GPU BUILD + PARITY GATE IN CLAUDE CODE before any claim).
// SwiGLU forward (gpu_silu_mul_kernel.hx):  h = silu(g) * u,  silu(g) = g*sigmoid(g).
// Given upstream dh, the two input gradients are:
//   dg = dh * u * silu'(g),  silu'(g) = sigmoid(g) * (1 + g*(1 - sigmoid(g)))
//   du = dh * silu(g)        ( = dh * g * sigmoid(g) )
// (silu'(g) = sig + g*sig*(1-sig) = sig*(1 + g*(1-sig)); the standard SiLU derivative.)
// Sigmoid REUSES the exact overflow-safe idiom of the forward gpu_silu_mul_kernel.hx: a
// `mut sgn` seeded from a CONSTANT (never from a local), an IMMUTABLE amag = sgn*gi, and
// sigmoid(g) = 0.5*(1 + sgn*(1-e)/(1+e)), e = exp(-|g|). The exp argument is ALWAYS <= 0
// so __gpu_exp never overflows f32 (large +g -> sig->1, large -g -> sig->0, matching numpy).
// CRITICAL (documented forward bug): gi must NEVER be the init-source of a reassigned
// mutable, or kovc ALIASES the mutable onto gi and clobbers it. Here gi is read-only; only
// `sgn` is mutable and it is seeded from the constant 1.0. silu = gi*sig is reused for both
// outputs. One thread per element (gridDim.x=N, blockDim.x=1). Two outputs (dg, du) as
// separate params -> 6 params (g,u,dh,dg,du,n; n unused = grid dim). Distinct let-vars:
// i, gi, ui, dhi, sgn, amag, e, sig, silu, sp -> 10, under the 12 cap. Ends on a store.
// Uses ONLY already-gated emitter features (__gpu_exp + arithmetic + one `if`) -- NO
// kovc.hx change; the self-host fixpoint is untouched.
// Oracle: llama_ops_bwd_numpy_ref.py silu_mul_bwd().
// Launch: cuda_launch out.ptx gpu_silu_mul_bwd <N> silu_mul_bwd
@kernel
fn gpu_silu_mul_bwd(g: f32, u: f32, dh: f32, dg: f32, du: f32, n: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    let gi = g[i];
    let ui = u[i];
    let dhi = dh[i];
    let mut sgn = 1.0;
    if gi < 0.0 { sgn = 0.0 - 1.0 };
    let amag = sgn * gi;
    let e = __gpu_exp(0.0 - amag);
    let sig = 0.5 * (1.0 + sgn * ((1.0 - e) / (1.0 + e)));
    let silu = gi * sig;
    let sp = sig * (1.0 + gi * (1.0 - sig));
    dg[i] = dhi * ui * sp;
    du[i] = dhi * silu
}
