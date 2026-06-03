// GPU in-place scale by a RUNTIME factor (T2/M6 capstone re-train). a[i] = s[0]*a[i],
// where the scalar s is a 1-element device buffer (the SAME runtime-scalar pattern Adam
// uses for bias-correction). Dimension-agnostic: the attention 1/sqrt(d) factor is passed
// at launch, so the kernel works at ANY head dim d (unlike gpu_scale_inplace, which bakes
// the d=16 literal 0.25). Used by the optimized capstone re-train to scale the tiled-GEMM
// QK^T scores (forward) and dQ/dK (backward) by 1/sqrt(d). One thread per element
// (gridDim.x=N, blockDim.x=1). 3 params, 1 let-var. Uses only already-gated emitter
// features (no kovc.hx change). Launch: cuda_launch out.ptx gpu_scale_rt <N> scale_rt
@kernel
fn gpu_scale_rt(a: f32, s: f32, n: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    a[i] = s[0] * a[i]
}
