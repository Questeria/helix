// GPU corpus kernel (v1.5 S0 increment 3): PACKED ternary matmul C = W . X on the GPU --
// the ternary MEMORY win + the DoD S0 "packed representation" criterion. The weights W are
// stored 2 bits per trit, 15 trits per 32-bit word (a 15x storage reduction vs one i32 per
// trit), and the kovc-emitted kernel UNPACKS them ON THE DEVICE using DIVISION (no bitwise
// codegen needed -- the generic @kernel path emits div.s32; this mirrors the compiler's own
// field-unpack in kovc.hx). Activations X and output C stay i32-domain (t2 buffers).
//
// WHY 15 (not 16) trits/word: a word = sum_j code_j*4^j with codes {0,1,2}. With 16 fields the
// max word is 2*(4^16-1)/3 = 2,863,311,530 > 2^31-1, so the top field spills into the SIGN bit;
// because the @kernel path emits SIGNED div.s32, a negative (overflowed) word decodes its high
// fields wrong (a multi-agent design audit caught this -- a 16-field test was simulated to
// mismatch 90/256 cells). 15 fields use 30 bits, max word 2*(4^15-1)/3 = 715,827,882 < 2^31-1,
// so every packed word is non-negative and div.s32 is exact. (16 trits/word would need unsigned
// shifts, which the generic @kernel path does not emit -- that path stays div/mul/add only.)
//
// Encoding (host, cuda_launch 'ptmatmul'): trit -1/0/+1 -> base-4 code 2/0/1; word = sum_j code_j
// * 4^j (field j = base-4 digit j, j=0..14). Decode on device: extract digit j by repeated /4,
// code = w % 4 = w-(w/4)*4, then branch-free trit = code - 3*(code/2) ({0,1,2}->{0,+1,-1}).
// Requires K % 15 == 0 (kpacked = K/15 full words; no partial-word guard).
//
// One thread per output cell (row=block_idx, col=thread_idx; grid.x=M, block.x=N), like
// naive/ternary_matmul. Exact integer accumulation. Verified element-exact vs an UNPACKED CPU
// int reference on the RTX 3070 (scripts/gpu_packed_ternary_check.sh, cuda_launch 'ptmatmul'),
// with comparator + kernel-corruption negative controls. mm (M) unused (row from block_idx).
@kernel
fn packed_ternary_matmul(a: t2, b: t2, c: t2, mm: i32, kk: i32, nn: i32) {
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
    c[row * nn + col] = acc
}
