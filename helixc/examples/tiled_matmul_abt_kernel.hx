// GPU corpus kernel (T2/M4): shared-memory TILED A @ B^T, the SMEM-tiled
// sibling of the naive gpu_matmul_abt (needed for d_attn = dOut @ V^T in the
// attention backward pass). A single FUSED intrinsic emits the WHOLE tiled
// kernel body (cooperative scalar GMEM->SMEM staging with TRANSPOSED B index
// + bar.sync + a runtime k-tile loop + a 4x4 register micro-tile FMA
// accumulate + epilogue store) -- see emit_ptx_tiled_matmul_t (mode 0) in
// kovc.hx.
//
// Given A[M,K] (a[i*K+t]) and B[N,K] (b[j*K+t], accessed transposed), computes
// C[M,N] with C[i,j]=sum_t A[i,t]*B[j,t]. Reuses the forward tiled GEMM's
// SMEM layout + register micro-tile; only B's GMEM->SMEM load index differs
// (b[(colbase+col)*K + k] instead of b[k*N + col]). Tile params for the RTX
// 3070 (sm_86): BM=BN=64, BK=8, TM=TN=4, threadblock 16x16=256, grid=(N/64,
// M/64). Requires M%64==0, N%64==0, K%8==0 (no boundary guard). Validated
// cell-by-cell vs a CPU A@B^T oracle AND faster than the naive non-tiled
// gpu_matmul_abt by: cuda_launch out.ptx tiled_matmul_abt 0 gemm_abt <M> <K> <N>.
//
// mm (M) is unused in the body (the row block comes from ctaid.y); the kernel
// derives every index from the launch geometry + K/N strides.
@kernel
fn tiled_matmul_abt(a: f32, b: f32, c: f32, mm: i32, kk: i32, nn: i32) {
    __matmul_abt_smem(a, b, c, mm, kk, nn)
}
