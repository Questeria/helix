// GPU corpus kernel (v1.5 S1): half-precision (f16) matrix multiply C = A*B on
// the GPU, proving Helix's @kernel PTX path can emit real fp16 global-memory
// I/O -- the dequant target for the low-precision slate (the thing ternary/
// MXFP4/NVFP4 widen INTO before they compute).
//
// SCOPE (honest): f16 I/O, f32 ACCUMULATION. Each f16 element is loaded with
// ld.global.b16 then widened (cvt.f32.f16) into an %f register; the multiply
// and the running accumulator are full f32 (mul.f32 / add.f32); the result is
// narrowed (cvt.rn.f16.f32) and stored with st.global.b16. This is the standard
// "f16 storage, f32 math" GEMM contract (what cuBLAS Hgemm with a f32 compute
// type does) -- NOT a pure-f16-arithmetic kernel, and NOT claimed to be one.
// NAIVE, one thread per output cell (row=block_idx(), col=thread_idx();
// launched gridDim.x=M, blockDim.x=N), mirroring naive_matmul / ternary_matmul.
// mm (M) is unused (row comes from block_idx), matching naive_matmul.
//
// ABI (the S0 c-param-class lesson): all three buffers a/b/c are `f16` so the
// @kernel param convention gives each a .b64 POINTER. A data array param must
// be a non-i32 type -- a bare `i32` param becomes a .u32 SCALAR (that is how
// mm/kk/nn are passed) and using it as a pointer faults on the device. f16 is a
// 2-byte element, so the index stride is mul.wide.s32 ...,2 (NOT 4); the
// emitter keys the stride + the load/store width on the param's element type.
//
// Verified within fp16 tolerance (NOT exact -- f16 rounds) vs an independent
// f32-accum CPU reference on the RTX 3070 by scripts/gpu_f16_check.sh
// (cuda_launch.c 'hgemm' mode), with a magnitude-scaled comparator negative
// control and a kernel-corruption negative control:
//   cuda_launch out.ptx naive_matmul_f16 <N> hgemm <M> <K> <N> [mutate]
@kernel
fn naive_matmul_f16(a: f16, b: f16, c: f16, mm: i32, kk: i32, nn: i32) {
    let row = block_idx();
    let col = thread_idx();
    let mut acc = a[row * kk] * b[col];
    let mut t = 1;
    while t < kk {
        acc = acc + a[row * kk + t] * b[t * nn + col];
        t = t + 1
    };
    c[row * nn + col] = acc
}
