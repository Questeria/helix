import os, sys, struct, json, numpy as np, torch, torch.nn.functional as F, subprocess
KOVC = "--kovc" in sys.argv
NL = int(os.environ.get("BN_LAYERS","30"))
md=os.path.expanduser("~/bitnet-2b/model.safetensors")
with open(md,"rb") as f:
    n=struct.unpack("<Q",f.read(8))[0]; H=json.loads(f.read(n)); BASE=8+n
def raw(name):
    v=H[name]; s,e=v["data_offsets"]
    with open(md,"rb") as f: f.seek(BASE+s); return v["dtype"], v["shape"], f.read(e-s)
def bf16(name):
    dt,sh,b=raw(name); a=(np.frombuffer(b,np.uint16).astype(np.uint32)<<16).view(np.float32).reshape(sh); return torch.from_numpy(a.copy())
def unpack_ternary(name):
    dt,sh,b=raw(name); p=np.frombuffer(b,np.uint8).reshape(sh); op,infe=sh; out=op*4
    W=np.zeros((out,infe),np.float32)
    for i in range(4): W[i*op:(i+1)*op]=((p>>(2*i))&3).astype(np.float32)-1.0
    return torch.from_numpy(W)
def wscale(name):
    dt,sh,b=raw(name); return float((np.frombuffer(b,np.uint16).astype(np.uint32)<<16).view(np.float32)[0])
def rmsnorm(x,w,eps=1e-5):
    v=x.pow(2).mean(-1,keepdim=True); return w*(x*torch.rsqrt(v+eps))
BN=os.path.expanduser("~/bn_kovc"); os.makedirs(BN,exist_ok=True)
def pack15(W):
    out,infe=W.shape; Kpad=((infe+14)//15)*15; kp=Kpad//15
    Wp=np.zeros((out,Kpad),np.int32); Wp[:,:infe]=W.astype(np.int32)
    code=np.where(Wp<0,2,np.where(Wp>0,1,0)).astype(np.int64).reshape(out,kp,15)
    return (code*(4**np.arange(15)).astype(np.int64)).sum(2).astype(np.int32), Kpad
def bitlinear(x,wn,subnorm=None):
    if subnorm is not None: x=rmsnorm(x,subnorm)
    s=wscale(wn+"_scale")
    sc=127.0/x.abs().amax(-1,keepdim=True).clamp(min=1e-5)
    a=(x*sc).round().clamp(-128,127)
    if not KOVC:
        W=unpack_ternary(wn); return F.linear(a,W)*s/sc
    W=unpack_ternary(wn).numpy().astype(np.int32); out,infe=W.shape
    tag=wn.replace(".","_"); pf=BN+"/"+tag+".bin"; kf=pf+".kpad"
    if not os.path.exists(pf):
        words,Kpad=pack15(W); words.tofile(pf); open(kf,"w").write(str(Kpad))
    Kpad=int(open(kf).read()); sN=a.shape[0]
    Ap=np.zeros((sN,Kpad),np.int32); Ap[:,:infe]=a.numpy().astype(np.int32)
    np.ascontiguousarray(Ap.T).tofile(BN+"/acts.bin")
    subprocess.run(["/home/legoa/bn_kovc/cl_dump","/home/legoa/bn_kovc/sptr.ptx","scaled_packed_ternary_matmul",str(sN),"sptmatmul_dump",pf,BN+"/acts.bin",BN+"/out.bin",str(out),str(Kpad),str(sN)],check=True,capture_output=True)
    y=torch.from_numpy(np.fromfile(BN+"/out.bin",np.float32).reshape(out,sN).T.copy())
    return y*s/sc
ids=torch.tensor([128000,791,6864,315,9822,374])
emb=bf16("model.embed_tokens.weight"); x=emb[ids]
seq=x.shape[0]; HD=128; NH=20; NKV=5; theta=5e5
inv=1.0/(theta**(torch.arange(0,HD,2).float()/HD)); fr=torch.outer(torch.arange(seq).float(),inv)
cos=torch.cat([fr.cos(),fr.cos()],-1); sin=torch.cat([fr.sin(),fr.sin()],-1)
def rh(t): h=t.shape[-1]//2; return torch.cat([-t[...,h:],t[...,:h]],-1)
def rope(t): return t*cos[:,None,:]+rh(t)*sin[:,None,:]
od=os.path.expanduser("~/bitnet_trace"+("_kovc" if KOVC else "")); os.makedirs(od,exist_ok=True)
np.save(od+"/hs_00.npy",x.numpy())
for L in range(NL):
    p=f"model.layers.{L}."
    h=rmsnorm(x,bf16(p+"input_layernorm.weight"))
    q=bitlinear(h,p+"self_attn.q_proj.weight").view(seq,NH,HD)
    k=bitlinear(h,p+"self_attn.k_proj.weight").view(seq,NKV,HD)
    v=bitlinear(h,p+"self_attn.v_proj.weight").view(seq,NKV,HD)
    q=rope(q); k=rope(k)
    k=k.repeat_interleave(NH//NKV,1); v=v.repeat_interleave(NH//NKV,1)
    q=q.transpose(0,1); k=k.transpose(0,1); v=v.transpose(0,1)
    scr=q@k.transpose(-1,-2)/(HD**0.5); scr=scr+torch.triu(torch.full((seq,seq),float("-inf")),1)
    att=(F.softmax(scr,-1)@v).transpose(0,1).reshape(seq,NH*HD)
    x=x+bitlinear(att,p+"self_attn.o_proj.weight",subnorm=bf16(p+"self_attn.attn_sub_norm.weight"))
    h=rmsnorm(x,bf16(p+"post_attention_layernorm.weight"))
    g=bitlinear(h,p+"mlp.gate_proj.weight"); u=bitlinear(h,p+"mlp.up_proj.weight")
    x=x+bitlinear(F.relu(g)**2*u,p+"mlp.down_proj.weight",subnorm=bf16(p+"mlp.ffn_sub_norm.weight"))
    np.save(od+"/hs_%02d.npy"%(L+1),x.numpy())
if NL==30:
    x=rmsnorm(x,bf16("model.norm.weight")); last=F.linear(x,emb)[-1].numpy()
    np.save(od+"/logits.npy",last); am=int(last.argmax())
    print("REF%s argmax=%d top5=%s -> %s"%(" KOVC" if KOVC else "", am, np.argsort(last)[-5:][::-1].tolist(), "PARIS-MILESTONE" if am==12366 else "WRONG"))
else:
    print("partial NL=%d done"%NL)
