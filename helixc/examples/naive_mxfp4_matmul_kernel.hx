// GPU corpus kernel (v1.5 S2): MXFP4 (OCP microscaling FP4) dequant -> f16 matmul on the GPU,
// proving Helix can DEQUANT a 4-bit block-scaled format end-to-end ON THE DEVICE and feed it into
// the S1 fp16 GEMM path -- the verifiable-dequant + memory win the FP4 slate is about.
//
// FORMAT (OCP MXFP4): each weight element is E2M1 = 1 sign / 2 exp / 1 mantissa (4 bits). The 16
// codes decode to the signed magnitude set {0, 0.5, 1, 1.5, 2, 3, 4, 6} (exp=00 subnormal m*0.5;
// exp=01/10/11 normal (1+0.5m)*2^(exp-1)). A shared E8M0 8-bit power-of-2 scale per 32-element block
// gives the block value = E2M1_magnitude * 2^(E8M0-127). There is no native FP4 on sm_86, so this is
// a STORAGE + verifiable-DEQUANT (memory) win, NOT FP4 Tensor-Core throughput -- no speed is claimed.
//
// PACKING (the S0 15-not-16-trits lesson, re-derived for base-16): pack 7 E2M1 elements per i32 word
// (7*4=28 bits; max word = 16^7-1 = 268435455 < 2^31-1, so the on-device SIGNED div.s32 nibble-unpack
// is EXACT). 8 elements/word (32 bits) would set bit 31 -> negative word -> mis-decode. So `w` holds
// M*(kk/7) packed words. The E8M0 scale is HOST-decoded to a LINEAR f32 (no __gpu_exp2 exists on the
// @kernel path) and passed in `sc` as one f32 per 32-element block (M*(kk/32) values); the DEVICE does
// the full E2M1 nibble dequant + the mag*scale multiply + the matmul on-device. Require kk % 224 == 0
// (LCM(7,32)) so the 7/word packing and the 32/block scale both tile evenly.
//
// NO kovc.hx edit (rides the existing @kernel path like S0's packed ternary + S1's f16): nibble via
// division (code = wv - (wv/16)*16; wv = wv/16), the E2M1 decode is an f32-exact if-ladder over the
// 8 dyadic literals (NO __gpu_i2f needed -- the magnitude is a float literal, the sign is a float
// negate), accumulate in f32, store f16 (c:f16 -> cvt.rn.f16.f32). The self-host fixpoint stays
// cdcf8673 byte-identical. Body kept to 11 lets (the @kernel var table caps at 12 and SILENTLY
// no-ops past it -- a wrong overflow there yields unbound %r-1 / garbage).
//
// ABI: w/sc/b/c are array params -> .b64 pointers (sc:f32 -> ld.global.f32; b/c:f16 -> the S1
// b16<->f32 cvt path; w:t2 -> the i32 ld.global.u32 path). mm is unused (row from block_idx), matching
// naive_matmul / ternary_matmul. Launch grid=(M,1,1) block=(N,1,1), one thread per output cell.
//
// Verified vs an independent from-scratch C E2M1+E8M0 dequant+matmul oracle on the RTX 3070 by
// scripts/gpu_mxfp4_check.sh (cuda_launch.c 'mxfp4' mode), dual-bound fp16 tolerance, with a
// magnitude-scaled comparator NC + a kernel-corruption NC + a from-scratch-codec self-test.
@kernel
fn naive_mxfp4_matmul(w: t2, sc: f32, b: f16, c: f16, mm: i32, kk: i32, nn: i32) {
    let row = block_idx();
    let col = thread_idx();
    let kwords = kk / 7;
    let mut acc = 0.0;
    let mut kw = 0;
    while kw < kwords {
        let mut wv = w[row * kwords + kw];
        let mut j = 0;
        while j < 7 {
            let code = wv - (wv / 16) * 16;
            let c8 = code - (code / 8) * 8;
            let k = kw * 7 + j;
            let magf = if c8 == 0 { 0.0 } else { if c8 == 1 { 0.5 } else { if c8 == 2 { 1.0 } else { if c8 == 3 { 1.5 } else { if c8 == 4 { 2.0 } else { if c8 == 5 { 3.0 } else { if c8 == 6 { 4.0 } else { 6.0 } } } } } } };
            acc = acc + (if code / 8 == 0 { magf } else { 0.0 - magf }) * sc[row * (kk / 32) + k / 32] * b[k * nn + col];
            wv = wv / 16;
            j = j + 1
        };
        kw = kw + 1
    };
    c[row * nn + col] = acc
}
