// GPU corpus kernel (T2/M1): shared-memory TILED matrix multiply C = A*B,
// the GPU critical-path kernel. A single FUSED intrinsic emits the WHOLE
// tiled kernel body (cooperative GMEM->SMEM staging + bar.sync + a runtime
// k-tile loop + a register micro-tile FMA accumulate + epilogue store) --
// see emit_ptx_tiled_matmul_smem in kovc.hx. Tile params for the RTX 3070
// (sm_86): BM=BN=64, BK=8, TM=TN=4, threadblock 16x16=256, grid=(N/BN, M/BM).
//
// Unlike naive_matmul (one thread per output cell, register-only, zero
// .shared/bar.sync), this kernel cooperatively stages 64x8 A-tiles and 8x64
// B-tiles into shared memory and reuses them across a 4x4 register micro-tile
// per thread -- the throughput-unlocking data reuse. Requires M%64==N%64==
// K%8==0 (no boundary guard, like naive_matmul). Validated cell-by-cell vs a
// CPU GEMM oracle by: cuda_launch out.ptx tiled_matmul 0 gemm_perf <M> <K> <N>.
//
// mm (M) is unused in the body (the row block comes from ctaid.y); the kernel
// derives every index from the launch geometry + K/N strides.
@kernel
fn tiled_matmul(a: f32, b: f32, c: f32, mm: i32, kk: i32, nn: i32) {
    __tiled_matmul_smem(a, b, c, mm, kk, nn)
}
