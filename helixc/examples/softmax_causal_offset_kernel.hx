// GPU CAUSAL softmax WITH AN ABSOLUTE-POSITION OFFSET (v1.8 prompt-lookup speculative-decode
// VERIFY kernel -- PATH B fix #1). Hand-written copy of gpu_softmax_causal's body with the
// valid-extent generalized from the window-RELATIVE `nvalid = row + 1` to the ABSOLUTE
// `nvalid = base + row + 1`, where `base` (== the verify window's first absolute position
// `pos`) is passed as a launch parameter. NOT an edit of any kovc.hx intrinsic (the self-host
// fixpoint is untouched); this is a standalone @kernel compiled by the cached kovc driver and
// loaded via cuModuleLoadData, exactly like nvfp4_dequant_tiled / fused_decode_attn.
//
// In the verify forward the score matrix is [rows, cols] where query row r sits at ABSOLUTE
// position base+r and the K matrix holds the RESIDENT prefix [0, base) followed by the m1 new
// window rows [base, base+m1). Causality requires row r to attend keys [0, base+r] ONLY
// (extent base+r+1) over the contiguous cache -- a draft placed at a position > base+r must
// NOT leak in. So for row r: reduce/normalize over [0, base+r+1) and write y[r,j]=0 for
// j >= base+r+1. This is NUMERICALLY IDENTICAL to gpu_softmax_row run with cols=base+r+1 (the
// M=1 decode softmax over base+r+1 cached positions): same first-element max seed, same
// sequential max-reduce from index 1, same sequential exp+sum from index 0, same normalize --
// so the verify row's softmax is bit-identical to the plain-decode row at that position.
//
// `base` is the absolute position of window row 0 (== `pos`). `cols` = the padded key count
// (>= base+rows). Rows beyond the live window (pad rows r where base+r+1 may exceed the real
// extent) clamp nvalid to cols and are never read by the caller (they map past kv_len). `0.0`
// is written as x[b]-x[b] (no f32 literal, mirroring the sibling kernels). Scores must already
// carry the 1/sqrt(head_dim) scale (gpu_scale_rt before this kernel), exactly like the causal
// kernel. One thread per query row: launch gridDim.x=rows, blockDim.x=1 so row=block_idx().
// Launch: cuda_launch out.ptx softmax_causal_offset <rows> softmax_causal_offset <rows> <cols> <base>
@kernel
fn softmax_causal_offset(x: f32, y: f32, rows: i32, cols: i32, base: i32) {
    let row = block_idx() * block_dim() + thread_idx();
    let bse = row * cols;
    let mut nvalid = base + row + 1;
    if nvalid > cols { nvalid = cols };
    let mut m = x[bse];
    let mut j = 1;
    while j < nvalid {
        if x[bse + j] > m { m = x[bse + j] };
        j = j + 1
    };
    let mut s = x[bse] - x[bse];
    let mut k = 0;
    while k < nvalid {
        let e = __gpu_exp(x[bse + k] - m);
        y[bse + k] = e;
        s = s + e;
        k = k + 1
    };
    let mut z = nvalid;
    while z < cols {
        y[bse + z] = x[bse] - x[bse];
        z = z + 1
    };
    let mut t = 0;
    while t < nvalid {
        y[bse + t] = y[bse + t] / s;
        t = t + 1
    }
}
