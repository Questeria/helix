#!/usr/bin/env python3
# parity helper: extract one linear's raw f32 [rows x K] from the converted .weights to a .bin,
# and write the numpy-packed words + scale, so we can cmp vs the C --ternary-packfile output.
import sys, numpy as np
sys.path.insert(0, "/mnt/c/Projects/Kovostov-Native/scripts")
import llama_ternary_pack as L
path, layer, which, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
flat = L.load_f32(path)
W, r, c = L.get_linear(flat, layer, which)
W.astype(np.float32).tofile(outdir + "/raw_f32.bin")
trit, scale = L.ternarize(W)
words, Kpad = L.pack_trits(trit)
words.tofile(outdir + "/np_packed.bin")
scale.astype(np.float32).tofile(outdir + "/np_scale.bin")
print("EXTRACT which=%s L=%d rows=%d K=%d Kpad=%d words=%d scale0=%.9g" % (which, layer, r, c, Kpad, words.size, float(scale[0])))
