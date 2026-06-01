// GPU transcendental probe (P5 Phase 1): c[i] = e^(a[i]) via the new __gpu_exp
// intrinsic (lowers to mul.f32 by log2e + ex2.approx.f32). b is unused (kept so the
// 4-arg launcher arg layout {&a,&b,&c,&N} lines up). Verified vs CPU expf with a
// small relative tolerance (ex2.approx is ~2^-22). Launch: cuda_launch out.ptx gpu_test_exp 256 exp
@kernel
fn gpu_test_exp(a: f32, b: f32, c: f32, n: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    c[i] = __gpu_exp(a[i])
}
