// GPU RMSNorm backward -- input gradient dx (LLAMA-ARCH backward op; AUTHORED for the QAT
// trainer leg -- NEEDS GPU BUILD + PARITY GATE IN CLAUDE CODE before any claim).
// RMSNorm forward (gpu_rmsnorm_fwd_eps_kernel.hx):  y = w * x * inv,
//   inv = rsqrt( mean_j(x_j^2) + eps ),  eps = 1e-5 baked (SmolLM2-135M / TinyLlama).
// Backward (one row, cols = d_model):
//   dx_i = inv*(w_i*dy_i)  -  (x_i * inv^3 / cols) * sum_j( w_j * dy_j * x_j )
// This is the RMSNorm specialisation of gpu_layernorm_backward_dx_kernel.hx: RMSNorm has
// NO mean-subtract, so the LayerNorm dmean (m1) term vanishes and only the variance/scale
// coupling (the m2-analogue, here sdot) survives -- strictly simpler than LayerNorm bwd.
// Derivation: y_c = w_c*x_c*inv, inv = (q+eps)^(-1/2), q = (1/N)*sum x_k^2.
//   d inv/d x_i = -1/2*(q+eps)^(-3/2) * (2 x_i / N) = -(x_i/N)*inv^3.
//   dL/dx_i = w_i*dy_i*inv  +  (sum_c w_c*dy_c*x_c) * d inv/dx_i
//           = inv*w_i*dy_i  -  (x_i*inv^3/N) * sum_c w_c*dy_c*x_c.
// eps BAKED 0.00001 to match the forward kernel exactly (same rms_norm_eps).
// One thread per row (gridDim.x=rows, blockDim.x=1). Two passes over cols, REUSING the
// loop counter j (j=0 resets), accumulators init via x[base]-x[base] (= f32 zero from the
// element type, the proven idiom). 6 params (at the layernorm-bwd-dx cap); distinct
// let-vars: row, base, colsf, ss, sdot, j, inv, inv3, coef, k -> 10, under the 12 cap.
// Uses ONLY already-gated emitter features (__gpu_i2f, __gpu_rsqrt, while, arithmetic) --
// NO kovc.hx change; the self-host fixpoint is untouched.
// Oracle: llama_ops_bwd_numpy_ref.py rmsnorm_bwd_dx().
// Launch: cuda_launch out.ptx gpu_rmsnorm_bwd_dx <rows> rmsnorm_bwd_dx <rows> <cols>
@kernel
fn gpu_rmsnorm_bwd_dx(x: f32, w: f32, dy: f32, dx: f32, cols: i32) {
    let row = block_idx() * block_dim() + thread_idx();
    let base = row * cols;
    let colsf = __gpu_i2f(cols);
    let mut ss = x[base] - x[base];
    let mut sdot = x[base] - x[base];
    let mut j = 0;
    while j < cols {
        ss = ss + x[base + j] * x[base + j];
        sdot = sdot + w[j] * dy[base + j] * x[base + j];
        j = j + 1
    };
    let inv = __gpu_rsqrt(ss / colsf + 0.00001);
    let inv3 = inv * inv * inv;
    let coef = inv3 / colsf * sdot;
    let mut k = 0;
    while k < cols {
        dx[base + k] = inv * (w[k] * dy[base + k]) - x[base + k] * coef;
        k = k + 1
    }
}
