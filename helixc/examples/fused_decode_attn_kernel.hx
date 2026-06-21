// FUSED DECODE-ATTENTION (v1.8 "other"-reclaim, STANDALONE @kernel -- NO kovc.hx edit, fixpoint
// cdcf8673 UNTOUCHED). Collapses the per-head decode attention loop (gpt2_infer.c :1460-1473:
// NH x {gemv_abt scores, scale_rt, smrow, gemv_ab AV} + NH cuMemcpyDtoD ctx == 128 launches + 32
// D2D copies per layer) into ONE launch. Rides the existing standalone-PTX path EXACTLY like
// nvfp4_dequant_tiled / dequant_gemv_blockred: kovc emits .ptx text -> cuModuleLoadData ->
// cuModuleGetFunction("fused_decode_attn") -> cuLaunchKernel. Gated behind HX_FUSEDATTN (default OFF
// = the byte-identical per-head path). The committed __flash_attention intrinsic is NOT reusable
// here: it is grid=(S,) one-block-per-query-ROW, contiguous [S,d] K/V, NO GQA -- decode has 1 query
// row x NH heads attending a STRIDED GQA KV-cache (kv = h/group). Hence a fresh standalone kernel.
//
// LAUNCH GEOMETRY: grid = (NH, 1, 1), block = (DH, 1, 1).
//   block_idx() = h          -> the query head this block owns (0..NH).
//   thread_idx() = lane = d  -> the output dimension this thread writes (0..DH); also the K/V column
//                               it reads each step. DH (<=128 for 8B) <= 256 thread ceiling: OK.
// Each block recomputes the full per-head attention SERIALLY over the T cached positions, keeping the
// running stable-softmax state (max m, denom l) and the AV accumulator in registers. No shared memory,
// no block reduction, no inter-thread sync: every lane d is independent (it owns out[h*DH+d] and reads
// only column d of K/V). The per-lane "score" needs the FULL dot over all DH dims, so each lane
// re-reads the whole q-row + the whole k-row for position t from global -- DH-fold redundant vs a
// SMEM-staged variant, but at decode (one block per head, latency-bound, tiny T) the win is killing
// the 128 launches + 32 D2D copies, not arithmetic; a SMEM stage is a later increment if dprof says so.
//
// ============================ ADVERSARY-FLAGGED FIXES (2026-06-20) ============================
// FIX 1 (SCALE, primary): the scale is NO LONGER recomputed on-device as rsqrt.approx.f32(i2f(DH))
//   -- ptx rsqrt.approx is a few ULP off the host's correctly-rounded 1.0f/sqrtf(DH), which perturbs
//   every score and can flip an argmax. INSTEAD the host's EXACT ATTN_SCALE float is passed in via a
//   1-element f32 device array `sc` (the SAME convention the fused-gemv kernel uses for its `ts`
//   per-tensor scale: an f32 param lowers to a .b64 pointer in the @kernel ABI, read at index 0).
//   The score multiply is now `s = s * sc[0]` -- byte-identical to the host scale_rt(*d_scale).
//   (Read inline at the scale site, NOT bound to a `let`, to stay inside the 12-let dialect cap;
//   the T-fold reload of one float is negligible at decode.)
// FIX 2 (cache-offset width): the @kernel ABI lowers EVERY scalar param as .u32 (ld.param.u32) and
//   computes array indices in a 32-bit %r reg, then widens to a 64-bit BYTE address via
//   `mul.wide.s32 %rd, %r_index, 4` + add.s64. So the BYTE address is already 64-bit (>4GB tensors OK)
//   -- the only requirement is that the ELEMENT index `koff + t*DH + e` fit in i32 (< 2^31). For this
//   model (NL<=36, NKV=8, DH=128, max context NC=32768) the worst-case element base
//   `(NL*NKV)*kv_cap*DH = (36*8)*32768*128 ~= 1.21e9 < 2^31`, so i32 element indices are SAFE and the
//   host (gpt2_infer.c) ASSERTS `koff + NKV*kvstride < 2^31` at dispatch and FAILS CLOSED to the exact
//   per-head path otherwise. A true i64 SCALAR param is NOT expressible in this dialect (scalar params
//   are .u32-only) without a kovc.hx emitter change -- which would rotate the self-host fixpoint, the
//   one thing this standalone route exists to avoid. The host guard delivers fix-2's intent (no silent
//   truncation -> wrong tokens) within the dialect.
// FIX 3 (op-order honesty): the one-pass online stable-softmax below is FMA-FAITHFUL to the two-pass
//   smrow reference, NOT bit-identical. It is gated EMPIRICALLY by the token-for-token generation gate
//   (HELIX_GEN_IDS byte-identical, on vs off) exactly like the fused-gemv kernel -- NOT claimed bit-exact.
// =============================================================================================
//
// BIT-FAITHFUL MATH (matches :1460-1473 up to the FMA-faithful softmax reorder noted in FIX 3):
//   kv      = h / grp                                    GQA fan-out (grp = NH/NKV; 8B: 32/8 = 4)
//   base    = koff + (h/grp)*kvstride                    == kvoff(L, h/grp) in floats (host pre-adds
//                                                           the (lyr*NKV)*kv_cap*DH layer term into koff)
//   qoff    = h*DH                                        q-row base for this head in d_q1 (inlined)
//   scale   = sc[0]                                       == host ATTN_SCALE = 1/sqrtf(DH) (FIX 1)
//   scores[t] = scale * sum_{e<DH} q[qoff+e]*k[base+t*DH+e]        (gemv_abt + scale_rt fused)
//   ONLINE STABLE SOFTMAX over t in [0,T) (NO causal mask: the 1-step decode attends ALL cached
//     positions 0..pos == [0,T); smrow uses no mask either):
//       on a new score s: m_new = max(m, s); rescale acc *= exp(m - m_new); l = l*exp(m-m_new)+exp(s-m_new)
//       acc += exp(s - m_new) * v[base + t*DH + d]        (this lane's V column d)
//   out[h*DH + d] = acc / l                               (== ctx1_t[h*DH+d]; written straight in place,
//                                                            so the 32 cuMemcpyDtoD ctx copies vanish)
//
// ABI (q/kc/vc/out/sc are .b64 device pointers [f32 lowers to .b64 in the @kernel ABI]; T/DH/koff/grp/
//      kvstride are .u32 scalars):
//   q   = d_q1            [NH x DH] post-QK-norm post-RoPE queries
//   kc  = d_kcache base   [NL x NKV x kv_cap x DH] roped K   (kernel indexes base + t*DH + e)
//   vc  = d_vcache base   [NL x NKV x kv_cap x DH] roped V
//   out = ctx1_t          [NH x DH] attention context (written in place)
//   sc  = d_attn_scale    [1] f32 == host ATTN_SCALE (FIX 1: NOT recomputed on-device)
//   T   = cached positions (== pos+1 at this step)
//   DH  = head dim
//   koff= per-LAYER base for kv-head 0 of layer L, i.e. (lyr*NKV+0)*kv_cap*DH (floats); the kernel adds
//         (h/grp)*kvstride itself via the `grp` stride.
//   grp = group = NH/NKV ; kvstride = kv_cap*DH (floats between consecutive kv-heads in the cache)
@kernel
fn fused_decode_attn(q: f32, kc: f32, vc: f32, out: f32, sc: f32, T: i32, DH: i32, koff: i32, grp: i32, kvstride: i32) {
    let h = block_idx();
    let d = thread_idx();
    let base = koff + (h / grp) * kvstride;          // kvoff(L, h/grp) in floats (K and V share layout)
    let mut m = 0.0 - 100000000.0;                   // running row-max (seed -1e8; pre-scale scores)
    let mut l = 0.0;                                 // running softmax denominator
    let mut acc = 0.0;                               // AV accumulator for output col d
    let mut t = 0;
    while t < T {
        // score s = scale * dot(q[h], k[base + t*DH, :])  -- full DH dot, recomputed per lane.
        let mut s = 0.0;                             // (reuses one slot; inner e-loop accumulates)
        let mut e = 0;
        while e < DH {
            s = s + q[h * DH + e] * kc[base + t * DH + e];   // qoff inlined (h*DH) to stay <=12 lets
            e = e + 1
        };
        s = s * sc[0];                               // FIX 1: host ATTN_SCALE (1/sqrtf(DH)); NO on-device rsqrt
        // online stable-softmax rescale: mn = max(m,s); acc,l *= exp(m-mn); then add exp(s-mn).
        let mn = if s > m { s } else { m };          // new running max
        let r = __gpu_exp(m - mn);                   // rescale factor for prior acc/l
        let p = __gpu_exp(s - mn);                   // weight of this position
        acc = acc * r + p * vc[base + t * DH + d];   // rescale + add this lane's V column d
        l = l * r + p;
        m = mn;
        t = t + 1
    };
    out[h * DH + d] = acc / l                         // ctx1_t[h*DH + d] = sum_t p_t V[t,d]
}
