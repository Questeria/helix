#!/usr/bin/env python3
# v1.9 P4b: the TERNARY CONVERSION RECEIPT for BitNet-2B on Helix -- "this model is verified-ternary, here
# is the evidence". --emit writes a hashable certificate binding {model-id; the 210 ternary BitLinear
# tensors; a Merkle root over sha256(name|shape|weight_scale|packed-15-trit) for each; the scaled-kernel
# .ref.ptx sha256; the measured end-to-end forward argmax + logits-error vs the fp reference}. --check
# re-derives all of it from the model + repo + traces and asserts byte-identity. numpy + hashlib only.
# Usage: bitnet_conversion_receipt.py {--emit|--check} <model_dir> <repo_root> <out_cert> [trace_home]
import os, sys, struct, json, hashlib, numpy as np
mode, md_dir, repo, cert = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
home = sys.argv[5] if len(sys.argv)>5 else os.path.expanduser("~")
MD = md_dir+"/model.safetensors"
with open(MD,"rb") as f:
    n=struct.unpack("<Q",f.read(8))[0]; H=json.loads(f.read(n)); BASE=8+n
def raw(nm):
    v=H[nm]; s,e=v["data_offsets"]
    with open(MD,"rb") as f: f.seek(BASE+s); return v["dtype"], v["shape"], f.read(e-s)
def scale_of(nm):
    dt,sh,b=raw(nm); return float((np.frombuffer(b,np.uint16).astype(np.uint32)<<16).view(np.float32)[0])
def unpack_ternary(nm):
    dt,sh,b=raw(nm); p=np.frombuffer(b,np.uint8).reshape(sh); op,infe=sh; out=op*4
    W=np.zeros((out,infe),np.int32)
    for i in range(4): W[i*op:(i+1)*op]=((p>>(2*i))&3).astype(np.int32)-1
    return W,sh
def pack15(W):
    out,infe=W.shape; Kpad=((infe+14)//15)*15; kp=Kpad//15
    Wp=np.zeros((out,Kpad),np.int32); Wp[:,:infe]=W
    code=np.where(Wp<0,2,np.where(Wp>0,1,0)).astype(np.int64).reshape(out,kp,15)
    return (code*(4**np.arange(15)).astype(np.int64)).sum(2).astype(np.int32)
def shx(b): return hashlib.sha256(b).hexdigest()
def merkle(hexleaves):
    lv=[bytes.fromhex(h) for h in hexleaves]
    if not lv: return shx(b"")
    while len(lv)>1:
        lv=[hashlib.sha256(lv[i]+(lv[i+1] if i+1<len(lv) else lv[i])).digest() for i in range(0,len(lv),2)]
    return lv[0].hex()
names=[]
for L in range(30):
    for t in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
        names.append("model.layers.%d.%s.weight"%(L,t))
names.sort()
leaves=[]
for nm in names:
    W,sh=unpack_ternary(nm); packed=pack15(W); sc=scale_of(nm)
    leaves.append(shx(nm.encode()+b"|"+str(sh).encode()+b"|"+struct.pack("<f",sc)+b"|"+packed.tobytes()))
root=merkle(leaves)
with open(repo+"/helixc/examples/scaled_packed_ternary_matmul_kernel.ref.ptx","rb") as f: kptx=shx(f.read())
kp=home+"/bitnet_trace_kovc/logits.npy"; op=home+"/bitnet_trace/logits.npy"
if os.path.exists(kp) and os.path.exists(op):
    k=np.load(kp); o=np.load(op); am=int(k.argmax()); err=float(np.abs(k-o).max())
else: am=-1; err=-1.0
body=("HELIX_TERNARY_CONVERSION_RECEIPT_V1\nmodel_id: microsoft/bitnet-b1.58-2B-4T\n"
      "n_ternary_tensors: %d\nkernel_ptx_sha256: %s\nforward_argmax: %d\nforward_logits_err_vs_fp: %.9g\nmerkle_root: %s\n"
      %(len(names),kptx,am,err,root))
top=shx(body.encode())
if mode=="--emit":
    with open(cert,"w") as f: f.write(body+"certificate_sha256: %s\n"%top)
    print("CONVERSION_RECEIPT_EMIT n=%d merkle=%.16s kernel=%.16s argmax=%d err=%g cert=%.16s"%(len(names),root,kptx,am,err,top))
else:
    st={}
    with open(cert) as f:
        for ln in f:
            if ":" in ln: kk,vv=ln.split(":",1); st[kk.strip()]=vv.strip()
    okm=st.get("merkle_root")==root; okk=st.get("kernel_ptx_sha256")==kptx; okc=st.get("certificate_sha256")==top
    print("CONVERSION_RECEIPT_CHECK merkle=%s kernel=%s cert=%s -> %s"%(okm,okk,okc,"CONVERSION_RECEIPT_PASS" if (okm and okk and okc) else "CONVERSION_RECEIPT_FAIL"))
    sys.exit(0 if (okm and okk and okc) else 1)
