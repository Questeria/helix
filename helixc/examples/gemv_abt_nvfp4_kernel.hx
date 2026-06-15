// GPU FUSED NVFP4-dequant GEMV (v1.7 INCREMENT 4 -- the fast-decode leg). Computes ONE output row
//   y[n] = sum_j x[j] * W[n,j]
// where W is stored PACKED NVFP4 (never materialised to f32): for output n it reads that row's packed
// i32 words and dequantises each 4-bit code INLINE in the accumulation -- so decode no longer has to
// re-dequant all weights to f32 every token (the 3.3 s/token bottleneck). One thread per output
// (gridDim.x*blockDim.x = n_out). Launch: cuda_launch out.ptx gemv_abt_nvfp4 <n_out> gemv_nvfp4 <Kpad>.
//
// Reuses the verified NVFP4 unpack from nvfp4_dequant_kernel.hx VERBATIM (7 E2M1 codes / i32 word,
// base-16 low-nibble-first; E2M1 magnitudes {0,.5,1,1.5,2,3,4,6}; sign = high bit) and the same
// host-collapsed effective f32 scale `sc` (one f32 per 16-block = e4m3_decode(micro)*fp32_tensor).
// W rows are Kpad-padded (Kpad % 112 == 0); pad columns hold code 0 -> contribute 0, AND x is padded
// to Kpad with zeros, so the full-Kpad loop needs NO bounds guard and is byte-equivalent to summing
// the real K. FAITHFUL (FMA-level) vs dequant-then-f32-gemv; the parity gate checks decode token-match.
// NO kovc.hx edit (rides the existing @kernel path) -> the self-host fixpoint stays cdcf8673.
// NOTE: the per-iteration values are IMMUTABLE `let`s (the dequant if-ladder pattern); re-assigning a
// `mut` with an if-expression result mis-allocates an unbound -1 register in the kovc emitter.
@kernel
fn gemv_abt_nvfp4(x: f32, w: t2, sc: f32, y: f32, kpad: i32) {
    let n = block_idx() * block_dim() + thread_idx();
    let kwords = kpad / 7;
    let wbase = n * kwords;
    let scbase = n * (kpad / 16);
    let mut acc = x[0] - x[0];
    let mut word = 0;
    let mut wv = 0;
    let mut s = 0;
    while word < kwords {
        wv = w[wbase + word];
        s = 0;
        while s < 7 {
            let col = word * 7 + s;
            let code = wv - (wv / 16) * 16;
            let c8 = code - (code / 8) * 8;
            let magf = if c8 == 0 { 0.0 } else { if c8 == 1 { 0.5 } else { if c8 == 2 { 1.0 } else { if c8 == 3 { 1.5 } else { if c8 == 4 { 2.0 } else { if c8 == 5 { 3.0 } else { if c8 == 6 { 4.0 } else { 6.0 } } } } } } };
            acc = acc + x[col] * ((if code / 8 == 0 { magf } else { 0.0 - magf }) * sc[scbase + col / 16]);
            wv = wv / 16;
            s = s + 1
        };
        word = word + 1
    };
    y[n] = acc
}
