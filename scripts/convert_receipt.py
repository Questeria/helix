#!/usr/bin/env python3
# v1.9 P6: ternary CONVERSION receipt -- certifies a model was CONVERTED from fp to ternary + the quality
# delta. Binds {source fp sha256, converted-latent sha256, a Merkle root over the 6 ternary weights/layer
# (Wq/Wk/Wv/Wo/W1/W2) ternarized [clip3(W/sc)*sc, sc=row-abs-mean], the fp->ternary loss delta}. --emit
# writes the cert; --check re-derives byte-identical. numpy+hashlib. Weight layout = the capstone flat order.
#   convert_receipt.py {--emit|--check} <fp.bin> <converted.bin> <cert> <fp_loss> <conv_loss> <NL> <D> <H> <V>
import sys, hashlib, numpy as np
mode, fpf, cvf, cert = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
fp_loss, conv_loss = float(sys.argv[5]), float(sys.argv[6])
NL, D, H, V = int(sys.argv[7]), int(sys.argv[8]), int(sys.argv[9]), int(sys.argv[10])
DD, DH, HD = D*D, D*H, H*D; LB = 4*DD + 4*D + DH + HD
def shx(b): return hashlib.sha256(b).hexdigest()
def merkle(hl):
    lv=[bytes.fromhex(h) for h in hl]
    if not lv: return shx(b"")
    while len(lv)>1: lv=[hashlib.sha256(lv[i]+(lv[i+1] if i+1<len(lv) else lv[i])).digest() for i in range(0,len(lv),2)]
    return lv[0].hex()
def tern_leaf(nm,W):
    sc=np.abs(W).mean(1); sc=np.where(sc<1e-12,1.0,sc).astype(np.float32)
    q=W/sc[:,None]; t=np.where(q>0.5,1,np.where(q<-0.5,-1,0)).astype(np.int8)
    return shx(nm.encode()+b"|"+sc.tobytes()+b"|"+t.tobytes())
fp=np.fromfile(fpf,np.float32); cv=np.fromfile(cvf,np.float32)
fp_sha=shx(fp.tobytes()); cv_sha=shx(cv.tobytes()); leaves=[]
for L in range(NL):
    b=L*LB
    for nm,off,rows,cols in [("Wq",b,D,D),("Wk",b+DD,D,D),("Wv",b+2*DD,D,D),("Wo",b+3*DD,D,D),("W1",b+4*DD+4*D,D,H),("W2",b+4*DD+4*D+DH,H,D)]:
        leaves.append(tern_leaf("%s.L%d"%(nm,L), cv[off:off+rows*cols].reshape(rows,cols)))
root=merkle(leaves)
body=("HELIX_TERNARY_CONVERSION_RECEIPT_V1\nkind: converted_from_fp\nsource_fp_sha256: %s\nconverted_latent_sha256: %s\nn_ternary: %d\nmerkle_ternary: %s\nfp_loss: %.6f\nconverted_loss: %.6f\ndelta_loss: %.6f\n"
      %(fp_sha,cv_sha,len(leaves),root,fp_loss,conv_loss,conv_loss-fp_loss))
top=shx(body.encode())
if mode=="--emit":
    open(cert,"w").write(body+"certificate_sha256: %s\n"%top)
    print("CONVERT_RECEIPT_EMIT src=%.12s conv=%.12s merkle=%.12s n=%d fp=%.4f conv=%.4f delta=%.4f"%(fp_sha,cv_sha,root,len(leaves),fp_loss,conv_loss,conv_loss-fp_loss))
else:
    st={}
    for ln in open(cert):
        if ":" in ln: k,v=ln.split(":",1); st[k.strip()]=v.strip()
    okm=st.get("merkle_ternary")==root; oks=st.get("source_fp_sha256")==fp_sha; okc=st.get("certificate_sha256")==top
    print("CONVERT_RECEIPT_CHECK merkle=%s src=%s cert=%s -> %s"%(okm,oks,okc,"CONVERT_RECEIPT_PASS" if (okm and oks and okc) else "CONVERT_RECEIPT_FAIL"))
    sys.exit(0 if (okm and oks and okc) else 1)
