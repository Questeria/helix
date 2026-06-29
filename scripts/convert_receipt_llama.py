#!/usr/bin/env python3
# v1.9 (llama ternary certification): CONVERSION receipt for a Llama-arch (SmolLM2) HXGW-v2 .weights
# model converted fp -> ternary. Adapts scripts/convert_receipt.py (which assumed the square-attention
# "capstone" flat layout) to the REAL SmolLM2 build_order_llama layout (GQA: k/v are KVD x DM, plus
# gate/up/down, RoPE, tied embed). Binds:
#   source_fp_sha256       : sha256 over the ORIGINAL fp .weights file bytes (smollm2-135m.weights)
#   converted_latent_sha256: sha256 over the converted ternary-latent .weights file bytes
#   merkle_ternary         : Merkle root over the 7 ternarized linears x NL layers (q,k,v,o,gate,up,down),
#                            each leaf = sha256(name | per-row abs-mean scale f32 | packed 15-trit i32 words),
#                            i.e. the EXACT bytes the kovc kernel consumes (gpt2_pack.c ternary_quantize_tensor).
#   fp_loss / converted_loss / delta : measured held-out perplexity (fp 8.7467 -> converted 1140.55).
# --emit writes the cert; --check re-derives byte-identical (source sha + Merkle + cert sha must all match).
# A field mutation (tamper) -> --check FAIL (negative control). numpy + hashlib only.
#   convert_receipt_llama.py {--emit|--check} <fp.weights> <converted.weights> <cert> <fp_loss> <conv_loss>
import sys, hashlib, numpy as np
sys.path.insert(0, "/mnt/c/Projects/Kovostov-Native/scripts")
import llama_ternary_pack as L

mode, fpf, cvf, cert = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
fp_loss, conv_loss = float(sys.argv[5]), float(sys.argv[6])

def shx(b): return hashlib.sha256(b).hexdigest()
def merkle(hl):
    lv = [bytes.fromhex(h) for h in hl]
    if not lv: return shx(b"")
    while len(lv) > 1:
        lv = [hashlib.sha256(lv[i] + (lv[i+1] if i+1 < len(lv) else lv[i])).digest() for i in range(0, len(lv), 2)]
    return lv[0].hex()

def tern_leaf(name, W):
    """leaf over the EXACT kernel-consumed bytes: per-row scale + packed 15-trit words (gpt2_pack format)."""
    trit, scale = L.ternarize(W)
    words, Kpad = L.pack_trits(trit)
    return shx(name.encode() + b"|" + scale.astype(np.float32).tobytes() + b"|" + words.astype(np.int32).tobytes())

# whole-file shas (binds provenance of BOTH files; source = the fp original)
with open(fpf, "rb") as f: fp_sha = shx(f.read())
with open(cvf, "rb") as f: cv_sha = shx(f.read())

cv_flat = L.load_f32(cvf)
LINS = [nm for nm, _, _, is_lin in L.layer_layout() if is_lin]   # q,k,v,o,gate,up,down
leaves = []
for layer in range(L.NL):
    for which in LINS:
        W, r, c = L.get_linear(cv_flat, layer, which)
        leaves.append(tern_leaf("%s.L%d" % (which, layer), W))
root = merkle(leaves)

body = ("HELIX_TERNARY_CONVERSION_RECEIPT_V1_LLAMA\n"
        "kind: converted_from_fp\n"
        "arch: llama_smollm2_135m\n"
        "dims: NL=%d DM=%d NKV=%d DF=%d NV=%d\n"
        "source_fp_sha256: %s\n"
        "converted_latent_sha256: %s\n"
        "n_ternary: %d\n"
        "merkle_ternary: %s\n"
        "fp_loss: %.6f\n"
        "converted_loss: %.6f\n"
        "delta_loss: %.6f\n"
        % (L.NL, L.DM, L.NKV, L.DF, L.NV, fp_sha, cv_sha, len(leaves), root, fp_loss, conv_loss, conv_loss - fp_loss))
top = shx(body.encode())

if mode == "--emit":
    open(cert, "w", newline="\n").write(body + "certificate_sha256: %s\n" % top)
    print("CONVERT_RECEIPT_LLAMA_EMIT src=%.12s conv=%.12s merkle=%.12s n=%d fp=%.4f conv=%.4f delta=%.4f cert=%.12s"
          % (fp_sha, cv_sha, root, len(leaves), fp_loss, conv_loss, conv_loss - fp_loss, top))
else:
    st = {}
    for ln in open(cert):
        if ":" in ln:
            k, v = ln.split(":", 1); st[k.strip()] = v.strip()
    okm = st.get("merkle_ternary") == root
    oks = st.get("source_fp_sha256") == fp_sha
    okc = st.get("certificate_sha256") == top
    ok = okm and oks and okc
    print("CONVERT_RECEIPT_LLAMA_CHECK merkle=%s src=%s cert=%s -> %s"
          % (okm, oks, okc, "CONVERT_RECEIPT_PASS" if ok else "CONVERT_RECEIPT_FAIL"))
    sys.exit(0 if ok else 1)
