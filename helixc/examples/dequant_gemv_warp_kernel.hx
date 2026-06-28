// GPU FUSED NVFP4-dequant WARP-PER-ROW GEMV (v1.8/P1, 2026-06-20). Computes ONE output row
//   y[n] = sum_j x[j] * dequant(W[n,j])
// where W is stored PACKED NVFP4 (never materialised to f32). The latency-bound limiter
// killer for the 8B decode path: vs the block-per-row __dequant_gemv_blockred baseline (one
// 256-thread block per row, 8-level bar.sync SMEM tree-reduce), this maps ONE 32-lane WARP
// per output row and packs 8 warps (8 rows) into a 256-thread (32x8) block. Two wins:
//   (1) the 8 bar.sync tree levels + 16 SMEM ld/st are replaced by a barrier-free 5-step
//       warp-shuffle reduce (shfl.sync.down.b32, deltas 16,8,4,2,1) -- zero barriers, zero
//       SMEM round-trips;
//   (2) 8 independent rows' loads are in flight per block (~8x memory-level parallelism) vs
//       the baseline's single-row-per-block.
// Grid (ceil(N/8),1,1), block (32,8,1):  warp = tid.y, lane = tid.x,
//   row = ctaid.x*8 + tid.y ; each lane stripes its row's i32 words (word=lane, stride +=32).
//
// Reuses the verified NVFP4 unpack VERBATIM from the baseline (7 E2M1 codes / i32 word,
// base-16 low-nibble-first; E2M1 magnitudes {0,.5,1,1.5,2,3,4,6}; sign = high bit), and the
// H4 in-kernel raw-e4m3 micro-scale decode (1 byte / 16-block, packed 4-per-i32-word in
// `micro`, times one per-tensor f32 `ts`). Only the row<->thread mapping and the reduce change;
// the dequant/FMA math is byte-faithful to __dequant_gemv_blockred (the warp shuffle only
// reorders the final f32 sum -> reduce-order-equivalent, FMA-faithful, NOT bit-identical to
// the SMEM tree -- gate on token-identical end-to-end). W rows are Kpad-padded
// (Kpad % 112 == 0 = LCM(7,16)); kwords = Kpad/7, scstride = Kpad/16; micro is
// [rows x ceil(scstride/4)] i32 words.
//
// The WHOLE body is emitted by the FUSED intrinsic __dequant_gemv_warp (see
// emit_ptx_dequant_gemv_warp in kovc.hx). Same 6-param ABI as __dequant_gemv_blockred, so the
// C++ launcher only changes grid/block (->(ceil(N/8),1,1)/(32,8,1)) + the kernel name.
//
// !! This intrinsic is a NEW kovc.hx emitter, but the name __dequant_gemv_warp appears in NO
//    self-host source (lexer.hx/parser.hx/kovc.hx have zero @kernel invocations of it) -> it is
//    DEAD CODE in K3 -> K2==K3==K4 byte-identical preserved. Building the driver from this file
//    re-pins the fixpoint hash (kovc.hx changed) but the 3-way fixpoint still holds.
//
// Launch / verify: cuda_launch out.ptx dequant_gemv_warp <N> dgemv_warp <Kpad> [mutate].
@kernel
fn dequant_gemv_warp(x: f32, w_packed: t2, micro: t2, ts: f32, y: f32, kpad: i32) {
    __dequant_gemv_warp(x, w_packed, micro, ts, y, kpad)
}
