#!/usr/bin/env python3
# v1.9 P3c: dump a REAL BitNet b1.58 BitLinear layer into the kovc ternary-kernel format for the GPU gate.
# Unpacks a packed-ternary weight (U8 [out/4,in]) from a BitNet safetensors (unpack = the P3b-verified
# formula W[i*(out/4)+r,c]=((p[r,c]>>2i)&3)-1, byte-identical to transformers.unpack_weights), int8-
# quantizes a deterministic activation (the P3c-step1-verified BitLinear formula), packs the trits into
# 15-trit/i32 words (the kovc kernel format), and writes packed.bin (i32 [out x in/15]) + acts.bin
# (i32 [in_pad x N]) + expected.bin (f32 [out x N] = a_int @ W_ternary.T, the integer matmul) + dims.txt.
# The GPU gate then checks the kovc scaled_packed_ternary_matmul reproduces expected.bin element-exact.
# numpy-only (no torch). Usage: bitnet_layer_dump.py <model.safetensors> <tensor_name> <out_dir> [M_rows(0=all)] [N_tokens]
import json, struct, os, sys, numpy as np
def main():
    md, name, od = sys.argv[1], sys.argv[2], sys.argv[3]
    Mrows = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    N     = int(sys.argv[5]) if len(sys.argv) > 5 else 8
    with open(md, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
    v = hdr[name]; s, e = v["data_offsets"]
    with open(md, "rb") as f: f.seek(base + s); packed = np.frombuffer(f.read(e - s), np.uint8).reshape(v["shape"])
    op, infe = packed.shape; out = op * 4
    W = np.zeros((out, infe), np.int32)
    for i in range(4):
        W[i*op:(i+1)*op, :] = ((packed >> (2*i)) & 3).astype(np.int32) - 1
    if Mrows and Mrows < out: W = W[:Mrows]; out = Mrows
    x = np.random.RandomState(0).randn(N, infe).astype(np.float32)
    absmax = np.abs(x).max(-1, keepdims=True).clip(1e-5)
    a_int = np.clip(np.round(x * 127.0 / absmax), -128, 127).astype(np.int32)
    Kpad = ((infe + 14) // 15) * 15; kpacked = Kpad // 15
    Wp = np.zeros((out, Kpad), np.int32); Wp[:, :infe] = W
    Ap = np.zeros((N, Kpad), np.int32); Ap[:, :infe] = a_int
    code = np.where(Wp < 0, 2, np.where(Wp > 0, 1, 0)).astype(np.int64).reshape(out, kpacked, 15)
    words = (code * (4 ** np.arange(15)).astype(np.int64)).sum(2).astype(np.int32)
    b = np.ascontiguousarray(Ap.T.astype(np.int32))
    expected = (Wp.astype(np.int64) @ Ap.T.astype(np.int64)).astype(np.float32)
    os.makedirs(od, exist_ok=True)
    words.tofile(od + "/packed.bin"); b.tofile(od + "/acts.bin"); expected.tofile(od + "/expected.bin")
    open(od + "/dims.txt", "w").write("%d %d %d\n" % (out, Kpad, N))
    e00 = int((Wp[0].astype(np.int64) * Ap[0].astype(np.int64)).sum())
    print("DUMP %s out=%d Kpad=%d N=%d | expected[0,0]=%g indep=%d match=%s" % (name, out, Kpad, N, expected[0,0], e00, expected[0,0] == e00))
if __name__ == "__main__": main()
