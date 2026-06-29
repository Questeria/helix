// GPU RoPE backward -- the TRANSPOSE rotation (LLAMA-ARCH backward op; AUTHORED for the QAT
// trainer leg -- NEEDS GPU BUILD + PARITY GATE IN CLAUDE CODE before any claim).
// RoPE forward (gpu_rope_rot_kernel.hx) rotates pairs (i, i+half) by (cos, sin):
//   y_i        = a*cos - b*sin        (a = x_i, b = x_{i+half})
//   y_{i+half} = b*cos + a*sin
// A 2x2 rotation R = [[cos,-sin],[sin,cos]] has transpose R^T = [[cos,sin],[-sin,cos]],
// so the backward (gradient) maps dy -> dx by the TRANSPOSE rotation:
//   dx_i        = dy_i*cos + dy_{i+half}*sin
//   dx_{i+half} = dy_{i+half}*cos - dy_i*sin
// (R is orthonormal => R^T = R^-1, i.e. backward is the inverse rotation -- consistent.)
// IN-PLACE on the packed head buffer q[rows, 2*half] (q holds dy on entry, dx on exit),
// exactly like the forward gpu_rope_rot: same cos/sin tables [rows, half] (HF inv_freq,
// base^(-2i/head_dim)), pairs (i, i+half) NOT interleaved. Reads BOTH halves into locals
// (da, db) BEFORE either store so the in-place write of dx_i does not corrupt the dx_{i+half}
// read (mirrors the forward's a,b capture). One thread per row (gridDim.x=rows, blockDim.x=1).
// Uses ONLY arithmetic + while (no intrinsic) -- NO kovc.hx change; fixpoint untouched.
// Oracle: llama_ops_bwd_numpy_ref.py rope_bwd().
// Launch: cuda_launch out.ptx gpu_rope_bwd <rows> rope_bwd <rows> <half>
@kernel
fn gpu_rope_bwd(q: f32, cosT: f32, sinT: f32, half: i32) {
    let row = block_idx() * block_dim() + thread_idx();
    let base = row * (half + half);
    let tbase = row * half;
    let mut i = 0;
    while i < half {
        let da = q[base + i];
        let db = q[base + half + i];
        let cv = cosT[tbase + i];
        let sv = sinT[tbase + i];
        q[base + i] = da * cv + db * sv;
        q[base + half + i] = db * cv - da * sv;
        i = i + 1
    }
}
