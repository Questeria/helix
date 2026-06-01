// GPU Adam optimizer step (P5). In-place update of one flat parameter array.
//   nm = b1*m + (1-b1)*g            [b1=0.9 baked]
//   nv = b2*v + (1-b2)*g*g          [b2=0.999 baked]
//   mhat = nm * bc1[0]              [bias-correction 1/(1-b1^t), host-supplied]
//   vhat = nv * bc2[0]              [bias-correction 1/(1-b2^t), host-supplied]
//   w   -= lr * mhat / sqrt(vhat + eps)   [lr=0.001, eps=1e-8 baked; 1/sqrt via __gpu_rsqrt]
//   m, v <- nm, nv
// The f32-scalar-param gap means hyperparameters are BAKED literals; the two
// step-dependent bias-correction scalars are passed as 1-ELEMENT f32 arrays (bc1,bc2)
// and read as bc1[0]/bc2[0]. One thread per element (gridDim.x=N, blockDim.x=1).
// 6 params (at cap), 7 let-vars. Only gated emitter features -> NO kovc.hx change.
// Verified vs an independent CPU Adam step (same literals + bc) -- nm,nv,new_w.
// Launch: cuda_launch out.ptx gpu_adam <N> adam
@kernel
fn gpu_adam(w: f32, g: f32, m: f32, v: f32, bc1: f32, bc2: f32) {
    let i = block_idx() * block_dim() + thread_idx();
    let gi = g[i];
    let nm = 0.9 * m[i] + 0.1 * gi;
    let nv = 0.999 * v[i] + 0.001 * gi * gi;
    let mh = nm * bc1[0];
    let vh = nv * bc2[0];
    let upd = 0.001 * mh * __gpu_rsqrt(vh + 0.00000001);
    w[i] = w[i] - upd;
    m[i] = nm;
    v[i] = nv
}
