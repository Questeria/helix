// GPU FUSED NVFP4-dequant BLOCK-REDUCTION GEMV (T2/M7, 2026-06-19). Computes ONE output row
//   y[n] = sum_j x[j] * dequant(W[n,j])
// where W is stored PACKED NVFP4 (never materialised to f32). ONE 256-thread block per output
// row n (grid=(N,1,1), block=(256,1,1) -> n = ctaid.x). The 256 threads COOPERATIVELY STRIPE the
// packed 4-bit weight row with COALESCED i32-word loads (thread t reads word t, t+256, t+512, ...),
// dequant each E2M1 code INLINE in the accumulation, accumulate a per-thread partial, then
// SMEM tree-reduce (bar.sync) to y[n]. This FUSES + COALESCES what used to be a dequant(separate,
// materialise-to-f32) + gemv(uncoalesced, one-thread-per-output) pair into a single coalesced pass:
// the per-output SERIAL block=1 unpack of gemv_abt_nvfp4 is replaced by a 256-way cooperative
// block-reduction, the real decode lever flagged in that kernel's v1.7-INC4 FINDING.
//
// Reuses the verified NVFP4 unpack (7 E2M1 codes / i32 word, base-16 low-nibble-first; E2M1
// magnitudes {0,.5,1,1.5,2,3,4,6}; sign = high bit). H4 (2026-06-20): the 16-block scale is now
// RAW e4m3 micro (1 byte / 16-block) packed 4-per-i32-word in `micro` PLUS one per-tensor f32 `ts`,
// e4m3-decoded IN-KERNEL as e4m3_decode(micro)*ts (was: the host-collapsed fp32 EFFECTIVE scale).
// The in-kernel e4m3 decode is BYTE-IDENTICAL to the host g_e4m3_tab[micro]*ts (the same 2^(e-7)
// pow LUT + (1+m/8) mantissa + sign), shrinking the resident scales ~4x so full 8B residency seats
// on a 7.1GB card and this fused path activates. W rows are Kpad-padded (Kpad % 112 == 0 = LCM(7,16));
// kwords = Kpad/7, scstride = Kpad/16; micro is [rows x ceil(scstride/4)] i32 words.
//
// The WHOLE body is emitted by the FUSED intrinsic __dequant_gemv_blockred (see
// emit_ptx_dequant_gemv_blockred in kovc.hx). FAITHFUL (FMA-level) vs dequant-then-f32-gemv; the
// e4m3 scale itself is byte-EXACT.
//
// !! This intrinsic is a NEW kovc.hx emitter -> building the driver from it ROTATES the
//    self-host fixpoint. The full gate (K2==K3==K4 + corpus + PTX regressions) re-pins the hash;
//    this file is the corpus kernel that exercises the new intrinsic.
//
// Launch / verify: cuda_launch out.ptx dequant_gemv_blockred <N> dgemv_blockred <Kpad> [mutate].
@kernel
fn dequant_gemv_blockred(x: f32, w_packed: t2, micro: t2, ts: f32, y: f32, kpad: i32) {
    __dequant_gemv_blockred(x, w_packed, micro, ts, y, kpad)
}
