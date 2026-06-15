// GPU corpus kernel (v1.7 INCREMENT 1): the TILED sibling of nvfp4_dequant -- same NVFP4 (OCP/NVIDIA)
// two-level-scaled 4-bit DEQUANT, but launched with a FLATTENED 1-D grid so it scales to the real
// worker tensor widths (K=4096..12288 >> the 1024 threads/block ceiling that the original
// grid=(M,) block=(K,) launch hits). The original nvfp4_dequant_kernel.hx stays UNTOUCHED (its
// gpu_nvfp4_check.sh corpus test still launches block=(K,) for small K).
//
// FORMAT/SCALE/PACKING are identical to nvfp4_dequant (see that file): E2M1 4-bit element
// {0,0.5,1,1.5,2,3,4,6}; value = E2M1_mag(sign) * sc[block], where `sc` is the HOST-precomputed
// EFFECTIVE per-16-block f32 scale (= e4m3_decode(micro) * fp32_tensorscale) -- the device does
// mag * scale, one mul.f32, no new op. 7 E2M1 codes per i32 word, base-16 low-nibble-first.
// Requires kk % 112 == 0 (kk = the PADDED Kpad; LCM(7,16)=112 tiles both the 7/word packing and the
// 16/block scale). NO kovc.hx edit (rides the existing @kernel path: S0 div-unpack + the f32-literal
// E2M1 if-ladder + ld.global.f32 scale + st.global.f32 out) -> the self-host fixpoint stays cdcf8673.
//
// FLATTENED TILING (the only change vs the corpus kernel): instead of one block per row + K threads
// per block, launch grid=(M * Kpad/BD, 1, 1) block=(BD, 1, 1) with BD passed in `nn`. A linear block
// id `lb` decomposes into row = lb / ntile and column-tile, where ntile = Kpad/BD. Choose BD = 112
// (a divisor of every Kpad, since Kpad % 112 == 0) so the tiling covers [0, Kpad) EXACTLY -- every
// thread has col < kk, so NO boundary guard is needed (keeps the body to the proven let/while/
// if-expression corpus subset). Output is [rows x Kpad] (pad cols are 0); the host compacts
// Kpad -> K (drop pad) into the f32 weight buffer, exactly like v3_upload's memmove. mm is unused.
//
// ABI: w/sc/out are array params -> .b64 pointers; mm unused; kk = Kpad; nn = BD (threads/block).
@kernel
fn nvfp4_dequant_tiled(w: t2, sc: f32, out: f32, mm: i32, kk: i32, nn: i32) {
    let lb = block_idx();
    let ntile = (kk + nn - 1) / nn;
    let row = lb / ntile;
    let coltile = lb - row * ntile;
    let col = coltile * nn + thread_idx();
    let word = col / 7;
    let slot = col - word * 7;
    let mut wv = w[row * (kk / 7) + word];
    let mut j = 0;
    while j < slot {
        wv = wv / 16;
        j = j + 1
    };
    let code = wv - (wv / 16) * 16;
    let c8 = code - (code / 8) * 8;
    let magf = if c8 == 0 { 0.0 } else { if c8 == 1 { 0.5 } else { if c8 == 2 { 1.0 } else { if c8 == 3 { 1.5 } else { if c8 == 4 { 2.0 } else { if c8 == 5 { 3.0 } else { if c8 == 6 { 4.0 } else { 6.0 } } } } } } };
    out[row * kk + col] = (if code / 8 == 0 { magf } else { 0.0 - magf }) * sc[row * (kk / 16) + col / 16]
}
