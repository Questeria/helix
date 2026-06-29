#!/usr/bin/env python3
# v1.9 P4: dump a REAL BitNet b1.58 BitLinear layer (W ternary [out,in], X int8 acts [in,N], C=W@X [out,N])
# for the Freivalds verifiable-inference receipt. numpy-only. The activation is the real layer-0 q_proj
# input (input_layernorm(embed[ids]) for "The capital of France is"), int8 per-token quantized (the
# verified BitLinear act-quant). Usage: bitnet_receipt_dump.py <model.safetensors> <tensor> <out_dir>
import os, sys, struct, json, hashlib, numpy as np
md, name, od = sys.argv[1], sys.argv[2], sys.argv[3]
with open(md,"rb") as f:
    n=struct.unpack("<Q",f.read(8))[0]; H=json.loads(f.read(n)); BASE=8+n
def raw(nm):
    v=H[nm]; s,e=v["data_offsets"]
    with open(md,"rb") as f: f.seek(BASE+s); return v["dtype"], v["shape"], f.read(e-s)
def bf16(nm):
    dt,sh,b=raw(nm); return (np.frombuffer(b,np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(sh)
dt,sh,b=raw(name); p=np.frombuffer(b,np.uint8).reshape(sh); op,infe=sh; out=op*4
W=np.zeros((out,infe),np.int32)
for i in range(4): W[i*op:(i+1)*op]=((p>>(2*i))&3).astype(np.int32)-1
ids=np.array([128000,791,6864,315,9822,374]); N=len(ids)
x=bf16("model.embed_tokens.weight")[ids].astype(np.float32)
lw=bf16("model.layers.0.input_layernorm.weight").astype(np.float32)
x=lw*(x/np.sqrt((x*x).mean(-1,keepdims=True)+1e-5))
absmax=np.abs(x).max(-1,keepdims=True).clip(1e-5)
a_int=np.clip(np.round(x*127.0/absmax),-128,127).astype(np.int32)
X=np.ascontiguousarray(a_int.T)
C=(W.astype(np.int64)@X.astype(np.int64)).astype(np.int32)
os.makedirs(od,exist_ok=True)
W.tofile(od+"/rcW.bin"); X.tofile(od+"/rcX.bin"); C.tofile(od+"/rcC.bin")
open(od+"/rcdims.txt","w").write("%d %d %d\n"%(out,infe,N))
print("RECEIPT_DUMP %s M=%d K=%d N=%d Cmax=%d sha256(W)=%s"%(name,out,infe,N,int(np.abs(C).max()),hashlib.sha256(W.tobytes()).hexdigest()[:16]))
