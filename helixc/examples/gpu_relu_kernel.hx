// GPU ReLU probe (P5 Phase 1): c[i] = max(a[i], 0). Tests the setp.f32 fix (the
// float compare a[i] > z) and an if-EXPRESSION returning an f32. `z` is a[i]-a[i]
// (= 0.0) to avoid an f32 literal; b is unused. Launched with NEGATIVE inputs
// (a[i]=i-128) so the float compare is genuinely exercised -- if setp used .s32 on
// the f32 bit pattern, negative-float compares could misorder.
// Launch: cuda_launch out.ptx gpu_relu 256 relu
@kernel
fn gpu_relu(a: f32, b: f32, c: f32, n: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    let z = a[i] - a[i];
    c[i] = if a[i] > z { a[i] } else { z }
}
