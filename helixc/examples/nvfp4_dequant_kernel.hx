// GPU corpus kernel (v1.5 S3): NVFP4 (OCP/NVIDIA) two-level-scaled 4-bit DEQUANT on the GPU,
// proving Helix can unpack + dequantize the NVFP4 format end-to-end ON THE DEVICE -- the verifiable
// low-precision STORAGE win at the finest (FP8-microscaled) granularity of the v1.5 slate.
//
// FORMAT (NVFP4): each weight element is E2M1 (the SAME 4-bit element as MXFP4/S2: 1s/2e/1m -> the 16
// signed magnitudes {0,0.5,1,1.5,2,3,4,6}). Scaling is TWO-LEVEL: an FP8 E4M3 micro-scale per 16-block
// + an FP32 per-tensor scale. Element value = E2M1_mag * e4m3_microscale * fp32_tensorscale.
//
// SCOPE (honest, DoD S3): VERIFIED DEQUANT only -- the format + an on-device dequant verified vs an
// independent oracle. sm_86 has NO native FP4 Tensor-Core, so the FP4 MMA / throughput leg is
// explicitly DEFERRED + labeled (it needs Blackwell, sm_100/sm_120); only the format + verified
// dequant land in v1.5. No speed is claimed. Output is f32 (NOT f16) -> the dequant is f32-EXACT vs
// the oracle (a sharper comparator than S2's f16-driven tolerance).
//
// TWO-LEVEL SCALE -> ONE host f32 (the S2 E8M0 pattern, extended): there is no __gpu_exp2/__gpu_pow on
// the @kernel path, so BOTH scale levels are HOST-decoded + pre-multiplied into ONE effective linear
// f32 scale per 16-block (effective = e4m3_decode(micro) * fp32_tensor), passed in `sc` (one f32 per
// 16 elements). The DEVICE just does mag * scale -- one mul.f32, no new op. NOTE E4M3 has NO Inf (only
// S.1111.111 = NaN); the host codec handles that (the device never sees E4M3).
//
// PACKING (reuse S2 verbatim): 7 E2M1 codes per i32 word, base-16 low-nibble-first (7 NOT 8 -- 16^8-1 >
// 2^31-1 spills the sign bit, breaking the signed div.s32 unpack; the S0 15-trit lesson). `w` holds
// M*(kk/7) packed words. Require kk % 112 == 0 (LCM(7,16): the 7/word packing AND the 16/block scale
// both tile evenly). NO kovc.hx edit (rides the existing @kernel path: S0 div-unpack + the f32-literal
// E2M1 if-ladder + ld.global.f32 scale + st.global.f32 out) -> the self-host fixpoint stays cdcf8673.
//
// ABI: w/sc/out are array params -> .b64 pointers (w:t2 -> ld.global.u32; sc/out:f32 -> ld/st.global.f32).
// mm + nn are unused (one thread per ELEMENT: row=block_idx, col=thread_idx = the element index 0..kk).
// Launch grid=(M,1,1) block=(K,1,1).
//
// Verified element-for-element vs an independent from-scratch C E2M1+E4M3 dequant oracle on the RTX
// 3070 by scripts/gpu_nvfp4_check.sh (cuda_launch.c 'nvfp4' mode), with a magnitude-scaled comparator
// NC, a packed-weight nibble-flip NC, AND a scale-corruption NC (proving the two-level scale is
// load-bearing), + a from-scratch-codec self-test (E2M1 16 codes + E4M3 boundaries incl 448/2^-9/NaN).
@kernel
fn nvfp4_dequant(w: t2, sc: f32, out: f32, mm: i32, kk: i32, nn: i32) {
    let row = block_idx();
    let col = thread_idx();
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
