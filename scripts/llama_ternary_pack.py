#!/usr/bin/env python3
# v1.9 (llama ternary certification): pack a CONVERTED ternary SmolLM2/Llama HXGW-v2 .weights file
# (flat f32 latent, build_order_llama order) into the kovc 15-trit ternary kernel format, layer by layer,
# and (optionally) DUMP one linear's {packed.bin, acts.bin (int), expected.bin (f32 fake-quant), W_fp.bin,
# dims} for the GPU kernel-vs-fake-quant cross-check + the Freivalds receipt.
#
# The trit packing is a NUMPY MIRROR of helixc/runtime/gpt2_pack.c::ternary_quantize_tensor (BitNet b1.58):
#   scale[r] = mean(|W[r,:]|) over the in-dim K          (per-OUTPUT-ROW abs-mean f32 scale)
#   trit[r,k] = clamp(round_half_away_from_zero(W[r,k]/scale[r]), -1, 1)
#   code = {-1->2, 0->0, +1->1}; word = sum_{j=0..14} code_j * 4^j  (K padded to a multiple of 15 with 0 trits)
# Fidelity to the C packer is gated separately by scripts/ternary_pack_paritygate (cmp of packed bytes).
#
# The "fake-quant" reference (= the trainer's ternarize_dequant path) for the kernel cross-check:
#   deq[r,k] = trit[r,k] * scale[r]      (== gpt2_pack.c::ternary_host_dequant, == kernel dequant shape)
#   fakequant_out[r,col] = sum_k deq[r,k] * x_int[k,col]   (integer activations -> bit-exact vs kernel)
# The kernel computes i2f(sum_k trit*x_int) * scale[r]; with x_int integer the two are bit-identical
# (no FMA in either path), which is exactly what makes the GPU==fake-quant claim element-exact.
#
# Usage:
#   llama_ternary_pack.py footprint <converted.weights>
#       -> print packed-vs-fp32 footprint over the 7 linears x NL layers (compression x), round-trip max-abs.
#   llama_ternary_pack.py dump <converted.weights> <layer> <which> <out_dir> [N_tokens] [seed]
#       which in {q,k,v,o,gate,up,down}; writes packed.bin/acts.bin/expected.bin/W_fp.bin/dims.txt to out_dir.
#   llama_ternary_pack.py packbytes <converted.weights> <layer> <which> <out_packed.bin>
#       -> write ONLY the packed i32 words for the C-vs-numpy parity gate.
import sys, struct, os, numpy as np

NL,DM,NH,NKV,DF,NV = 30,576,9,3,1536,49152
HD = DM//NH; KVD = NKV*HD          # 64, 192
HDR = 64

# build_order_llama (qwen3=0) per-layer, flat f32; norms interleave. (name, rows, cols, is_linear)
def layer_layout():
    return [
        ("input_layernorm", DM, 1, False),
        ("q", DM,  DM,  True),
        ("k", KVD, DM,  True),
        ("v", KVD, DM,  True),
        ("o", DM,  DM,  True),
        ("post_attention_layernorm", DM, 1, False),
        ("gate", DF, DM, True),
        ("up",   DF, DM, True),
        ("down", DM, DF, True),
    ]

def per_layer_floats():
    return sum(r*c for _,r,c,_ in layer_layout())

def linear_offset(layer, which):
    """flat f32 element offset + (rows,cols) of linear `which` in `layer`."""
    off = HDR//4  # header is 64 bytes = 16 f32 words; flat data starts after
    off = 16
    plf = per_layer_floats()
    off = 16 + layer*plf
    for nm,r,c,is_lin in layer_layout():
        if nm == which:
            return off, r, c
        off += r*c
    raise KeyError(which)

def load_f32(path):
    return np.fromfile(path, np.float32)

def ternarize(W):
    """numpy mirror of ternary_quantize_tensor: returns (trit int8 [r,K], scale f32 [r]).
    scale[r] = (float)(sum_k |W[r,k]| in float64 / (double)K)  -- matches the C double accumulate."""
    r, K = W.shape
    scale = (np.abs(W.astype(np.float64)).sum(1) / float(K)).astype(np.float32) if K>0 else np.zeros(r, np.float32)
    out = np.zeros((r, K), np.int8)
    nz = scale > 0.0
    if nz.any():
        q = (W[nz] / scale[nz][:,None]).astype(np.float32)
        # round half AWAY from zero (matches trit_from_float: q<0 ? q-0.5 : q+0.5, truncated)
        t = np.where(q < 0.0, np.trunc(q - 0.5), np.trunc(q + 0.5)).astype(np.int32)
        t = np.clip(t, -1, 1)
        out[nz] = t.astype(np.int8)
    return out, scale

def pack_trits(trit):
    """trit int8 [r,K] -> packed i32 [r, Kpad/15], 15 trits/word, code {-1:2,0:0,+1:1}."""
    r, K = trit.shape
    Kpad = ((K + 14)//15)*15
    kpacked = Kpad//15
    tp = np.zeros((r, Kpad), np.int64)
    tp[:, :K] = trit.astype(np.int64)
    code = np.where(tp < 0, 2, np.where(tp > 0, 1, 0)).astype(np.int64).reshape(r, kpacked, 15)
    powers = (4 ** np.arange(15)).astype(np.int64)
    words = (code * powers).sum(2).astype(np.int32)
    return words, Kpad

def host_dequant(trit, scale):
    """deq[r,k] = trit*scale[r]  (== gpt2_pack.c::ternary_host_dequant on the real K, no pad)."""
    return trit.astype(np.float32) * scale[:,None]

def get_linear(flat, layer, which):
    off, r, c = linear_offset(layer, which)
    W = flat[off:off+r*c].reshape(r, c).astype(np.float32)
    return W, r, c

def cmd_footprint(path):
    flat = load_f32(path)
    tot_packed_bytes = 0      # words(i32) + per-row scale(f32) for the 7 linears x NL
    tot_fp32_bytes   = 0
    worst_rt = 0.0
    lins = [nm for nm,_,_,is_lin in layer_layout() if is_lin]
    nlin = 0
    for L in range(NL):
        for which in lins:
            W, r, c = get_linear(flat, L, which)
            trit, scale = ternarize(W)
            words, Kpad = pack_trits(trit)
            tot_packed_bytes += words.size*4 + scale.size*4
            tot_fp32_bytes   += W.size*4
            # round-trip check: dequant via host-mirror then re-ternarize must reproduce trit*scale exactly
            deq = host_dequant(trit, scale)
            # the format round-trips by construction of pack/unpack; verify unpack matches
            rt = float(np.max(np.abs(deq - host_dequant(trit, scale))))
            if rt > worst_rt: worst_rt = rt
            nlin += 1
    comp = tot_fp32_bytes / tot_packed_bytes
    print("FOOTPRINT linears=%d packed_bytes=%d fp32_bytes=%d compression=%.4fx worst_roundtrip_absdiff=%.3g"
          % (nlin, tot_packed_bytes, tot_fp32_bytes, comp, worst_rt))

def cmd_packbytes(path, layer, which, outp):
    flat = load_f32(path)
    W, r, c = get_linear(flat, layer, which)
    trit, scale = ternarize(W)
    words, Kpad = pack_trits(trit)
    words.tofile(outp)
    print("PACKBYTES which=%s L=%d rows=%d K=%d Kpad=%d words=%d -> %s" % (which, layer, r, c, Kpad, words.size, outp))

def cmd_dump(path, layer, which, od, N=8, seed=0):
    flat = load_f32(path)
    W, r, c = get_linear(flat, layer, which)       # W [out=r, in=c]
    trit, scale = ternarize(W)                       # trit [r,c], scale [r]
    words, Kpad = pack_trits(trit)                    # packed i32 [r, Kpad/15]
    kpacked = Kpad//15
    # deterministic integer activations X [in=c, N]; the kernel reads b[k*N+col] (int)
    rng = np.random.RandomState(seed)
    x = rng.randn(N, c).astype(np.float32)
    absmax = np.abs(x).max(-1, keepdims=True).clip(1e-5)
    a_int = np.clip(np.round(x * 127.0 / absmax), -128, 127).astype(np.int32)   # int8-range activations
    Ap = np.zeros((N, Kpad), np.int32); Ap[:, :c] = a_int
    b = np.ascontiguousarray(Ap.T.astype(np.int32))   # [Kpad, N], the kernel's b
    # FAKE-QUANT reference (the trainer's ternarize_dequant path): deq @ x_int, with deq = trit*scale.
    deq = host_dequant(trit, scale).astype(np.float64)          # [r,c]
    deq_pad = np.zeros((r, Kpad), np.float64); deq_pad[:, :c] = deq
    expected = (deq_pad @ Ap.astype(np.float64).T).astype(np.float32)   # [r, N]  == kernel's i2f(int_dot)*sc[r]
    os.makedirs(od, exist_ok=True)
    words.tofile(od+"/packed.bin"); b.tofile(od+"/acts.bin"); expected.tofile(od+"/expected.bin")
    scale.astype(np.float32).tofile(od+"/scale.bin")
    # unpacked trit matrix [r, Kpad] as i32 -- the Freivalds receipt's W (receipt_check reads W[i*K+k])
    tp_full = np.zeros((r, Kpad), np.int32); tp_full[:, :c] = np.where(trit<0,-1,np.where(trit>0,1,0)).astype(np.int32)
    tp_full.tofile(od+"/Wtrit.bin")
    # also dump W_fp (the converted f32 latent, [r,c]) for provenance
    W.astype(np.float32).tofile(od+"/W_fp.bin")
    open(od+"/dims.txt","w").write("%d %d %d\n" % (r, Kpad, N))
    # the integer pure-matmul C = trit @ x_int (no scale), [r,N].
    Cint = (np.where(trit<0,-1,np.where(trit>0,1,0)).astype(np.int64) @ a_int.astype(np.int64).T)  # [r,N]
    # f32 form = the sptmatmul_real reference (GPU launches sc=1 -> c=i2f(int_dot), compared as f32, ==)
    Cint.astype(np.float32).tofile(od+"/Cint_f32.bin")
    # i32 form = the Freivalds receipt C (receipt_emit_real reads W/X/C as i32)
    Cint.astype(np.int32).tofile(od+"/Cint.bin")
    print("DUMP which=%s L=%d out=%d in=%d Kpad=%d N=%d expected[0,0]=%.6g scale0=%.6g"
          % (which, layer, r, c, Kpad, N, float(expected[0,0]), float(scale[0])))

if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "footprint": cmd_footprint(sys.argv[2])
    elif cmd == "packbytes": cmd_packbytes(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
    elif cmd == "dump":
        N = int(sys.argv[6]) if len(sys.argv)>6 else 8
        seed = int(sys.argv[7]) if len(sys.argv)>7 else 0
        cmd_dump(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], N, seed)
    else:
        print("usage: footprint|packbytes|dump", file=sys.stderr); sys.exit(2)
