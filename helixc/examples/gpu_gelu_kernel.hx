// GPU GELU (P5) -- the 4th and last core transformer op. tanh approximation
// (Hendrycks & Gimpel), mirroring stdlib __gelu (transcendentals.hx:523):
//   gelu(x) = 0.5*x*(1 + tanh(0.7978846*(x + 0.044715*x^3)))
//   tanh(z) = (e^(2z) - 1) / (e^(2z) + 1)   [inlined via __gpu_exp]
// First kernel to combine f32 LITERALS (0.5, 0.7978846, 0.044715, 1.0, 2.0 --
// the constants 0.7978846/0.044715 exercise the A-F hex-nibble emit path on HW)
// with the __gpu_exp intrinsic. One thread per element (gridDim.x=N, blockDim.x=1,
// no bounds guard). 6 distinct let-vars (under the vtab cap 12). No tanh
// short-circuit: validated on inputs in [-3,3] where |inner|<<20 so e^(2z) never
// saturates f32. x/y are f32 array(pointer) params, n is i32 (unused -- grid dim).
// Launch: cuda_launch out.ptx gpu_gelu <N> gelu
@kernel
fn gpu_gelu(x: f32, y: f32, n: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    let xi = x[i];
    let x3 = xi * xi * xi;
    let inner = 0.7978846 * (xi + (0.044715 * x3));
    let e2 = __gpu_exp(2.0 * inner);
    let th = (e2 - 1.0) / (e2 + 1.0);
    y[i] = 0.5 * xi * (1.0 + th)
}
