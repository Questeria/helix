// GPU QK^T-scaled (P5) -- the new op for ATTENTION stage 1. Computes the scaled
// score matrix scores[S,S] = (1/sqrt(d)) * Q[S,d] @ K^T[d,S]:
//   scores[i,j] = 0.25 * sum_t Q[i,t] * K[j,t]
// One thread per output cell: i=block_idx(), j=thread_idx() (launch gridDim.x=S,
// blockDim.x=S). A k-loop over t in [0,d) accumulates Q[i,t]*K[j,t] -- note K is
// accessed TRANSPOSED (k[j*dd + t] = K[j,t]), which is the only difference from
// naive_matmul (that does A@B, this does A@B^T). The accumulator seeds with the t=0
// product (mirrors naive_matmul) and the result is multiplied by the f32 LITERAL
// 0.25 = 1/sqrt(16) (0f3E800000) -- so this kernel is for d=16. For d=64 the scale
// literal is 0.125 (0f3E000000); the scale is baked, so d is fixed per compiled file.
// 5 params (q,k,scores,ss,dd <= 6 cap); 4 distinct let-vars (i,j,acc,t <= 12 cap).
// Uses only already-gated emitter features -> NO kovc.hx change. ss=S (scores row
// stride + j range), dd=d (Q/K row stride + k-loop bound).
// Used by cuda_launch attention mode (stage 1 of 3). Validated vs a CPU QK^T*scale.
@kernel
fn gpu_qkt(q: f32, k: f32, scores: f32, ss: i32, dd: i32) {
    let i = block_idx();
    let j = thread_idx();
    let mut acc = q[i * dd] * k[j * dd];
    let mut t = 1;
    while t < dd {
        acc = acc + q[i * dd + t] * k[j * dd + t];
        t = t + 1
    };
    scores[i * ss + j] = 0.25 * acc
}
