// GPU in-place scale by 0.25 (P5). a[i] = 0.25 * a[i]. Used by attention-backward
// to apply the 1/sqrt(d)=0.25 (d=16) factor to dQ/dK. One thread per element
// (gridDim.x=N, blockDim.x=1). 2 params, 1 let-var. The 0.25 is the same exact f32
// literal (0f3E800000) the forward gpu_qkt bakes. NO kovc.hx change. Verified
// a[i] == 0.25 * input[i]. Launch: cuda_launch out.ptx gpu_scale_inplace <N> scale
@kernel
fn gpu_scale_inplace(a: f32, n: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    a[i] = 0.25 * a[i]
}
