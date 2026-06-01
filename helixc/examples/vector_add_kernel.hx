// GPU first-light kernel (Helix v1.0 DoD criterion #3 -- "GPU executes").
//
// A concrete, NON-generic @kernel: c[i] = a[i] + b[i] over f32 global arrays,
// using the canonical grid-stride index  block_idx()*block_dim() + thread_idx()
// (the exact pattern kovc's PTX emitter is built for -- see kovc.hx:11002,
// thread_idx->%tid.x / block_idx->%ctaid.x / block_dim->%ntid.x).
//
// Deliberately NON-generic (no [N] clause) -- the PTX emitter has no
// monomorphize pass, so a generic @kernel would mis-lower. The count param n is
// present so the host arg array void*[]={&a,&b,&c,&N} (helixc/runtime/cuda_launch.c)
// lines up positionally, but n is UNUSED in the body: the emitter declares every
// param as .param .b64, so reading a 32-bit n would mis-decode; and with N=256,
// threads-per-block=256, blocks=1 every index is in-bounds, so no  if i < n  guard
// is needed.
//
// Pipeline:  kovc (k1ptxdrv main) reads this -> emits /tmp/out.ptx
//            -> cuda_launch /tmp/out.ptx vector_add 256  runs it on the GPU.
@kernel
fn vector_add(a: f32, b: f32, c: f32, n: i32) {
    let i = block_idx() * block_dim() + thread_idx();
    c[i] = a[i] + b[i]
}
