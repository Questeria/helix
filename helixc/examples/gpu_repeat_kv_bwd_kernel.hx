// GPU GQA repeat-KV backward (LLAMA-ARCH backward op; AUTHORED for the QAT trainer leg --
// NEEDS GPU BUILD + PARITY GATE IN CLAUDE CODE before any claim).
// GQA forward repeats each KV head n_rep times so the n_q_heads query heads can each attend
// a KV head:  kv_head(q_head) = q_head // n_rep,  n_rep = n_q_heads / n_kv_heads (=3 for
// SmolLM2-135M: 9 q / 3 kv). The forward broadcast copies one [S, head_dim] KV block to its
// n_rep consumers; backward SUMS the upstream gradient over those n_rep copies back into the
// single KV head (broadcast's adjoint is sum-reduction).
// Memory layout (matches the forward repeat, grouped by kv head):
//   din  [n_kv*n_rep, blk]  -- upstream grad, the repeated/q-head-shaped slab (blk = S*head_dim)
//   dout [n_kv,       blk]  -- accumulated grad for the single KV heads
//   dout[kv, off] = sum_{r in [0,n_rep)} din[(kv*n_rep + r), off]
// One thread per OUTPUT element (gridDim.x = n_kv*blk, blockDim.x = 1): flat thread index
// idx in [0, n_kv*blk); kv = idx / blk, off = idx - kv*blk; while-loop the n_rep sum. The
// accumulator inits via din[idx]-din[idx] (= f32 zero from the element type, the proven
// idiom). 5 params (din, dout, nrep, blk, n; n unused = grid dim). Distinct let-vars:
// idx, kv, off, kbase, acc, r, src -> 7, under the 12 cap. Uses ONLY integer/float
// arithmetic + while (no intrinsic) -- NO kovc.hx change; the fixpoint is untouched.
// Oracle: llama_ops_bwd_numpy_ref.py repeat_kv_bwd().
// Launch: cuda_launch out.ptx gpu_repeat_kv_bwd <n_kv*blk> repeat_kv_bwd <nrep> <blk>
@kernel
fn gpu_repeat_kv_bwd(din: f32, dout: f32, nrep: i32, blk: i32, n: i32) {
    let idx = block_idx() * block_dim() + thread_idx();
    let kv = idx / blk;
    let off = idx - kv * blk;
    let kbase = kv * nrep * blk;
    let mut acc = din[idx] - din[idx];
    let mut r = 0;
    while r < nrep {
        let src = kbase + r * blk + off;
        acc = acc + din[src];
        r = r + 1
    };
    dout[idx] = acc
}
