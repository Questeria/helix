// GPU corpus kernel (T2/M4): shared-memory TILED A^T @ B, the SMEM-tiled
// sibling of the naive gpu_matmul_atb (the weight gradient dW = X^T @ dY in
// linear-layer backprop). A single FUSED intrinsic emits the WHOLE tiled
// kernel body (cooperative scalar GMEM->SMEM staging with TRANSPOSED A index
// + bar.sync + a runtime k-tile loop + a 4x4 register micro-tile FMA
// accumulate + epilogue store) -- see emit_ptx_tiled_matmul_t (mode 1) in
// kovc.hx.
//
// Given A[M,K] (a[t*K+i], t=contraction) and B[M,N] (b[t*N+j]), computes
// C[K,N] with C[i,j]=sum_t A[t,i]*B[t,j] (contraction length = M). Reuses the
// forward tiled GEMM's SMEM layout + register micro-tile; only A's GMEM->SMEM
// load index is transposed (a[(k0+kt)*K + (rowbase+r)] instead of
// a[(rowbase+r)*K + (k0+kt)]) and the k-loop runs over M, not K. Tile params:
// BM=BN=64, BK=8, TM=TN=4, block 16x16=256, grid=(N/64, K/64). Requires
// K%64==0, N%64==0, M%8==0 (no boundary guard). Validated cell-by-cell vs a
// CPU A^T@B oracle AND faster than the naive non-tiled gpu_matmul_atb by:
// cuda_launch out.ptx tiled_matmul_atb 0 gemm_atb <M> <K> <N>.
//
// mm (M) is the contraction length (k-loop bound); kk (K) is the output-row
// count + A's row stride; nn (N) is the output-col count + B's row stride.
@kernel
fn tiled_matmul_atb(a: f32, b: f32, c: f32, mm: i32, kk: i32, nn: i32) {
    __matmul_atb_smem(a, b, c, mm, kk, nn)
}
