// GPU RoPE rotation, HF-Llama rotate_half convention (LLAMA-ARCH op 2 of 4; AUTHORED for
// the modern-model leg -- NEEDS GPU BUILD + PARITY GATE IN CLAUDE CODE before any claim).
// Applies rotary position embedding IN-PLACE to ONE packed head buffer q of shape
// [rows, 2*half] (rows = sequence positions, 2*half = head_dim), using HOST-PRECOMPUTED
// cos/sin tables of shape [rows, half]:
//   a = q[p, i]          (first half,  i in [0, half))
//   b = q[p, half + i]   (second half)
//   q[p, i]        = a*cos[p,i] - b*sin[p,i]
//   q[p, half + i] = b*cos[p,i] + a*sin[p,i]
// This is EXACTLY HuggingFace Llama's  q*cos + rotate_half(q)*sin  with rotate_half(q) =
// concat(-q2, q1): pairs are (i, i+half), NOT interleaved (i, i+1) -- the oracle
// (llama_ops_numpy_ref.py) pins this convention, and the importer must NOT permute.
// The host precomputes cos/sin from inv_freq = base^(-2i/head_dim) (base = rope_theta
// from config.json: 1e5 for SmolLM2, 1e4 for TinyLlama) once per context length --
// kovc has no sin/cos intrinsic and none is added (NO kovc.hx change; the self-host
// fixpoint 0992dddd is untouched). Tables are DATA computed by trusted-once host glue,
// exactly like the weights themselves; the kernel does all the arithmetic that touches
// activations. In-place mutation follows the gpu_scale_rt precedent. The full forward
// recomputes positions 0..S-1 each step (gpt2_infer-style, no KV cache), so row == the
// absolute position and the table rows line up 1:1.
// One thread per row (gridDim.x=rows, blockDim.x=1).
// Launch: cuda_launch out.ptx gpu_rope_rot <rows> rope <rows> <half>
@kernel
fn gpu_rope_rot(q: f32, cosT: f32, sinT: f32, half: i32) {
    let row = block_idx() * block_dim() + thread_idx();
    let base = row * (half + half);
    let tbase = row * half;
    let mut i = 0;
    while i < half {
        let a = q[base + i];
        let b = q[base + half + i];
        let cv = cosT[tbase + i];
        let sv = sinT[tbase + i];
        q[base + i] = a * cv - b * sv;
        q[base + half + i] = b * cv + a * sv;
        i = i + 1
    }
}
