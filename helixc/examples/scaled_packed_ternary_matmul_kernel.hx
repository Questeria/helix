// GPU corpus kernel (v1.9 P1): SCALED packed ternary matmul -- the v1.5 S0 packed_ternary_matmul
// (15 trits/i32 word, on-device base-4 division unpack, exact integer accumulation) with a
// per-OUTPUT-ROW f32 scale applied at the end: c[row,col] = i2f(int_dot) * sc[row]. This is THE
// missing inference primitive for a real ternary model: BitNet b1.58 stores ternary weights
// {-1,0,+1} packed + a per-row (abs-mean) f32 scale; the matmul is integer, the dequant is the final
// per-row scale. Rides the EXISTING @kernel path (NO kovc.hx edit): t2 packed loads + div-unpack (S0)
// + __gpu_i2f int->float (proven in gpu_ce_softmax_grad_kernel.hx) + one mul.f32 by the per-row scale
// (same shape as nvfp4_dequant's `* sc[block]`).
//
// ABI: a = packed ternary weights (t2, [rows x kpacked]); b = signed int activations (t2, [K x N]);
// sc = per-row f32 scale ([rows]); c = f32 output ([rows x N]); kk = K (mult of 15); nn = N. acc is
// i32 (tiny: |trit| <= 1, K small) so __gpu_i2f is exact and the single trailing mul.f32 has no FMA
// -> the GPU result is BIT-IDENTICAL to the host (float)int_dot * scale reference (verified element-
// exact in cuda_launch 'sptmatmul'). mm unused (row = block_idx). Requires K % 15 == 0.
@kernel
fn scaled_packed_ternary_matmul(a: t2, b: t2, sc: f32, c: f32, mm: i32, kk: i32, nn: i32) {
    let row = block_idx();
    let col = thread_idx();
    let kpacked = kk / 15;
    let mut acc = 0;
    let mut kw = 0;
    while kw < kpacked {
        let mut w = a[row * kpacked + kw];
        let mut j = 0;
        while j < 15 {
            let k = kw * 15 + j;
            let code = w - (w / 4) * 4;
            let trit = code - 3 * (code / 2);
            acc = acc + trit * b[k * nn + col];
            w = w / 4;
            j = j + 1
        };
        kw = kw + 1
    };
    c[row * nn + col] = __gpu_i2f(acc) * sc[row]
}
