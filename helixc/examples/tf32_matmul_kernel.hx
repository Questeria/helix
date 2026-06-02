// GPU corpus kernel (T3/G3): TF32 Tensor-Core matrix multiply C = A*B via mma.sync.
// A single FUSED intrinsic emits the WHOLE kernel body -- see emit_ptx_tf32_matmul_mma in
// kovc.hx. SINGLE-WARP, correctness-first: ONE warp (32 lanes = one block) computes one
// 16x8 output tile via mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32, looping K/8 mma
// calls. Grid=(N/8, M/16), block=(32,1,1). Path-2 (manual ld.global.f32 + cvt.rna.tf32.f32,
// no SMEM staging, no ldmatrix -- those are perf optimizations for the next phase). Requires
// M%16==0, N%8==0, K%8==0 (no boundary guard).
//
// Unlike tiled_matmul (scalar per-thread 4x4 FMA micro-tile), this kernel is warp-
// COLLABORATIVE: the 32 lanes jointly hold the A/B/C fragments and the Tensor Core does the
// 16x8x8 contraction per mma. The fragment->lane register layout is the PTX-ISA canonical
// m16n8k8 map (A cols are k=tig/tig+4, NOT 2*tig), GPU-validated at 16x8x8 with distinct
// inputs before any tiling. Validated vs cuBLAS-TF32 (cublasGemmEx COMPUTE_32F_FAST_TF32) at
// a tight ~2e-3 rel tol by: cuda_launch out.ptx tf32_matmul 0 gemm_tf32 <M> <K> <N>.
//
// mm (M) is unused in the body (the row block comes from ctaid.y); the kernel derives every
// index from the launch geometry + K/N strides.
@kernel
fn tf32_matmul(a: f32, b: f32, c: f32, mm: i32, kk: i32, nn: i32) {
    __tf32_matmul_mma(a, b, c, mm, kk, nn)
}
