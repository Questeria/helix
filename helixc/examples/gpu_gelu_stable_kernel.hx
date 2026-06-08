// GPU GELU, NUMERICALLY STABLE for GPT-2-scale inputs (P5 GPT-2 gap kernel). Same tanh-approx
// gelu_new as gpu_gelu (Hendrycks & Gimpel, = stdlib __gelu, transcendentals.hx:523) but with an
// OVERFLOW-SAFE tanh. The committed gpu_gelu computes tanh(z)=(e^(2z)-1)/(e^(2z)+1) directly; for
// GPT-2 the c_fc pre-activation reaches ~+/-12, so z=0.7978846*(x+0.044715*x^3) reaches ~63 and
// e^(2z)=e^126 OVERFLOWS f32 (>3.4e38) -> +inf -> (inf-1)/(inf+1)=NaN. (gpu_gelu's own header notes
// it was validated only on [-3,3] where e^(2z) never saturates.) This kernel uses the standard
// stable identity tanh(z) = sign(z) * (1 - e^(-|2z|)) / (1 + e^(-|2z|)): the exp argument is ALWAYS
// <= 0, so e^(-|2z|) lies in (0,1] and never overflows; large |z| -> e->0 -> tanh-> +/-1 (graceful
// saturation, matching numpy's np.tanh). Bit-for-bit equal to gpu_gelu on small inputs; finite and
// correct on the large inputs GPT-2 actually produces. Only __gpu_exp + arithmetic + one `if`
// (sign/abs) -- no new kovc.hx intrinsic, so the self-host fixpoint 0992dddd is untouched. One thread
// per element (gridDim.x=N, blockDim.x=1, no bounds guard -> blocks*threads must == n). Ends on a
// store (not a bare trailing `if`) so the PTX emitter lowers it cleanly.
// Launch: cuda_launch out.ptx gpu_gelu_stable <N> gelu
@kernel
fn gpu_gelu_stable(x: f32, y: f32, n: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    let xi = x[i];
    let x3 = xi * xi * xi;
    let inner = 0.7978846 * (xi + (0.044715 * x3));
    let two = 2.0 * inner;
    let mut amag = two;
    let mut sgn = 1.0;
    if inner < 0.0 { amag = 0.0 - two; sgn = 0.0 - 1.0 };
    let e = __gpu_exp(0.0 - amag);
    let th = sgn * ((1.0 - e) / (1.0 + e));
    y[i] = 0.5 * xi * (1.0 + th)
}
