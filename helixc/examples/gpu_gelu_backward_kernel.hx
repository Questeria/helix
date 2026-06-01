// GPU GELU backward (P5). dx = dy * gelu'(x), the tanh-approx GELU derivative.
// gelu(x)   = 0.5*x*(1 + tanh(inn)),  inn = 0.7978846*(x + 0.044715*x^3)
// gelu'(x)  = 0.5*(1 + tanh(inn)) + 0.5*x*(1 - tanh(inn)^2)*inn'
//   inn'    = 0.7978846*(1 + 0.134145*x^2)   [0.134145 = 3*0.044715]
//   tanh(z) = (e^2z - 1)/(e^2z + 1)  via __gpu_exp
// Elementwise, one thread per element (gridDim.x=N, blockDim.x=1, no bounds guard).
// 4 params (x, dy, dx, n; n unused = grid dim), 7 let-vars (i, xi, inn, e2, th, id,
// gp) -- under the 12 cap. Only gated emitter features -> NO kovc.hx change. Verified
// vs a CPU central finite-difference of the GELU FORWARD (independent of this analytic
// derivative) AND vs the analytic gelu' in C. Inputs in [-3,3] so e^2z never saturates.
// Launch: cuda_launch out.ptx gpu_gelu_backward <N> gelu_backward
@kernel
fn gpu_gelu_backward(x: f32, dy: f32, dx: f32, n: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    let xi = x[i];
    let inn = 0.7978846 * (xi + 0.044715 * xi * xi * xi);
    let e2 = __gpu_exp(2.0 * inn);
    let th = (e2 - 1.0) / (e2 + 1.0);
    let id = 0.7978846 * (1.0 + 0.134145 * xi * xi);
    let gp = 0.5 * (1.0 + th) + 0.5 * xi * (1.0 - th * th) * id;
    dx[i] = dy[i] * gp
}
