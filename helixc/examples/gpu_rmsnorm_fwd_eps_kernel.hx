// GPU RMSNorm forward WITH weight AND epsilon (LLAMA-ARCH op 1 of 4; AUTHORED for the
// modern-model leg -- NEEDS GPU BUILD + PARITY GATE IN CLAUDE CODE before any claim).
// Written in the exact style of gpu_layernorm_fwd_eps_kernel.hx (the P5 GPT-2 gap kernel):
// same one-thread-per-row contract, same baked epsilon precedent, same accumulator-init
// idiom (x[base]-x[base] = f32 zero from the element type).
//   y[row,c] = w[c] * x[row,c] * rsqrt( mean_c(x[row,c]^2) + eps )
// RMSNorm (Zhang & Sennrich 2019) = LayerNorm minus the mean-subtract and minus beta;
// population mean over cols to match numpy (x*x).mean(-1) in the oracle
// (helix-llm/tools/llama_ops_numpy_ref.py, kept uncommitted like the GPT-2 oracle).
// eps is BAKED as 0.00001 (1e-5) -- the rms_norm_eps of both candidate models
// (SmolLM2-135M and TinyLlama-1.1B, config.json rms_norm_eps=1e-5), mirroring how
// gpu_layernorm_fwd_eps bakes GPT-2's layer_norm_epsilon=1e-5. A model with a different
// eps gets a sibling kernel, not a runtime scalar (keeps the 4-param shape simple).
// Uses ONLY already-gated emitter features (__gpu_i2f, __gpu_rsqrt, while/if, f32
// indexing) -- NO kovc.hx change, so the self-host fixpoint 0992dddd is untouched.
// One thread per row (gridDim.x=rows, blockDim.x=1).
// Launch: cuda_launch out.ptx gpu_rmsnorm_fwd_eps <rows> rmsnorm <rows> <cols>
@kernel
fn gpu_rmsnorm_fwd_eps(x: f32, y: f32, w: f32, cols: i32) {
    let row = block_idx() * block_dim() + thread_idx();
    let base = row * cols;
    let colsf = __gpu_i2f(cols);
    let mut ss = x[base] - x[base];
    let mut j = 0;
    while j < cols { ss = ss + x[base + j] * x[base + j]; j = j + 1 };
    let inv = __gpu_rsqrt(ss / colsf + 0.00001);
    let mut t = 0;
    while t < cols { y[base + t] = w[t] * (x[base + t] * inv); t = t + 1 }
}
