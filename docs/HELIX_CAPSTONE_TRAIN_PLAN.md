# Helix Capstone: train_transformer.c blueprint

Architect-verified plan (2026-06-01) for the v1.0 capstone: a tiny 2-layer transformer
trains end-to-end on the RTX 3070 in pure Helix-native (all math = kovc-emitted PTX; the
C harness only does cuMemAlloc/cuMemcpy/cuLaunchKernel) within 2% of a numpy oracle.

## Resolved setup
- ADD kernel: REUSE helixc/examples/vector_add_kernel.hx -- vector_add(a,b,c,n): c[i]=a[i]+b[i]
  with i=block_idx()*block_dim()+thread_idx(); launch grid=N block=1 for the residual + grad
  sums. NO new kernel needed.
- ORACLE deps: WSL python3 has NO torch; numpy 2.4.6 installed via pip3 --break-system-packages.
  Oracle = numpy (helixc/runtime/oracle_train.py), FENCED OFFLINE (D1), reads init_weights.bin,
  NEVER in the Helix path. (numpy is a numeric lib, not an AI API -- allowed.)

## Config
2-layer pre-norm transformer: V=32, d=16, S=16, 1 head (d_head=16 so attn 0.25=1/sqrt(16)),
MLP H=64, batch 1. Input: fixed one-hot identity x_in[s*d+s]=1 (S=d=16). Targets (s+1)%16,
as f32 for ce_softmax_grad. Adam b1=0.9 b2=0.999 lr=1e-3 eps=1e-8, K=500.

## FORWARD per layer L (x=[S,d]): each arrow -> kernel(args) grid/block
- xn1 = layernorm_fwd_save(x,xn1,LN1g,LN1b,ist1,d) gS/b1
- Q = naive_matmul(xn1,Wq,Q,S,d,d) gS/bd ; K,Vv likewise (Wk,Wv)
- scores = gpu_qkt(Q,K,scores,S,d) gS/bS  ;  attn = gpu_softmax(scores,attn,S,S) gS/b1
- ao = naive_matmul(attn,Vv,ao,S,S,d) gS/bd
- proj = naive_matmul(ao,Wo,proj,S,d,d) gS/bd  ;  h1 = vector_add(x,proj,h1,S*d) gN/b1
- xn2 = layernorm_fwd_save(h1,xn2,LN2g,LN2b,ist2,d) gS/b1
- a = naive_matmul(xn2,W1,a,S,d,H) gS/bH ; g = gpu_gelu(a,g,S*H) gN/b1 ; m = naive_matmul(g,W2,m,S,H,d) gS/bd
- h2 = vector_add(h1,m,h2,S*d) gN/b1
After 2 layers: xf=layernorm_fwd_save(h2,xf,LNfg,LNfb,istf,d); logits=naive_matmul(xf,W_lm,logits,S,d,V) gS/bV.
loss = mean_s -log(softmax(logits[s])[target[s]]) (CPU, D2H logits). NOTE: ce_softmax_grad gives
the UNAVERAGED grad (softmax-onehot); train without 1/S (oracle matches) OR scale dlogits by 1/S.

## BACKWARD (reverse): each arrow -> kernel
- dlogits = ce_softmax_grad(logits,targets_f,dlogits,S,V) gS/b1
- dW_lm = matmul_atb(xf,dlogits,dW_lm,S,d,V) gd/bV ; dxf = matmul_abt(dlogits,W_lm,dxf,S,V,d) gS/bd
- LNf bwd: dx=layernorm_backward_dx(h2_L1,dxf,LNfg,istf,dx_lnf,d); dgb=layernorm_backward_dgb(h2_L1,dxf,istf,dLNf_dgb,S,d). dh2[1]=dx_lnf.
Per layer L reverse (dh2[L] in):
- MLP: dm=dh2[L]; dW2=matmul_atb(g,dm,dW2,S,H,d) gH/bd; dg=matmul_abt(dm,W2,dg,S,d,H) gS/bH;
  da=gpu_gelu_backward(a,dg,da,S*H) gN/b1; dW1=matmul_atb(xn2,da,dW1,S,d,H) gd/bH; dxn2=matmul_abt(da,W1,dxn2,S,H,d) gS/bd.
- LN2 bwd: dx_ln2=layernorm_backward_dx(h1,dxn2,LN2g,ist2,dx_ln2,d); dgb. dh1 = vector_add(dx_ln2, dh2[L], dh1, S*d) [RESIDUAL SUM].
- Attn proj: dproj=dh1; dWo=matmul_atb(ao,dh1,dWo,S,d,d) gd/bd; dao=matmul_abt(dh1,Wo,dao,S,d,d) gS/bd.
- Attn core (given dao): dVv=matmul_atb(attn,dao,dVv,S,S,d) gS/bd; dattn=matmul_abt(dao,Vv,dattn,S,d,S) gS/bS;
  dscores=gpu_softmax_backward(attn,dattn,dscores,S,S) gS/b1; dQ=naive_matmul(dscores,K,dQ,S,S,d)+scale0.25;
  dK=matmul_atb(dscores,Q,dK,S,S,d)+scale0.25.
- QKV weights: dWq=matmul_atb(xn1,dQ,...); dWk=matmul_atb(xn1,dK,...); dWv=matmul_atb(xn1,dVv,...).
- dxn1 = (dQ@Wq^T)+(dK@Wk^T)+(dVv@Wv^T): matmul_abt x3 into tmp1/2/3, then 2x vector_add -> dxn1 [SUM].
- LN1 bwd: dx_ln1=layernorm_backward_dx(x,dxn1,LN1g,ist1,...); dgb. dx[L]=vector_add(dx_ln1, dh1, dx[L], S*d) [RESIDUAL SUM]. dx[L]=dh2[L-1].

KEY: matmuls OVERWRITE c, so the 12 add-sites/step (4 forward residual, 2 residual-grad, 6 dxn1
sums across 2 layers) use vector_add. Each weight-grad dW is written ONCE per step (no zeroing
needed). Adam m/v are PERSISTENT (never zeroed).

## Adam: per weight gpu_adam(W,dW,mW,vW,bc1,bc2) gN/b1; host writes bc1=1/(1-0.9^t), bc2=1/(1-0.999^t)
to 1-elem device arrays each step.

## init_weights.bin (LE f32, this order): per L in {0,1}: Wq,Wk,Wv,Wo[d*d], LN1g,LN1b,LN2g,LN2b[d],
W1[d*H], W2[H*d]; then LNfg,LNfb[d], W_lm[d*V]. xorshift32(seed=0x12345678) -> [-1,1]*sqrt(2/fan_in);
LN gamma=1 beta=0; Adam m,v=0. Oracle reads this exact layout.

## Build stages (each GPU-verified + committed): combined PTX = concat all kernel .hx -> one .ptx,
load once + cuModuleGetFunction per kernel (attn mode pattern). Load all fns into a struct before the loop.
A: alloc + init + write init_weights.bin + forward to logits + CPU ce loss; print step0 loss
   (sanity: ~log(32)=3.47; NaN=dim bug).
B: numpy oracle reads init_weights.bin -> step0 loss matches C (tol 1e-3; if not, D2H intermediates).
C: backward to dlogits + dW_lm vs finite-diff (perturb a W_lm elem in init_weights.bin, reload, central
   diff; magnitude-aware tol max(1e-3, 5%*|fd|)). dlogits rows sum ~0.
D: full backward; spot-check 3-5 elems/weight vs finite-diff (order: dW_lm, dLNf, L1 weights, L0). NC:
   zero a dxn1 add -> finite-diff catches it.
E: Adam + K=500 vs oracle; |loss_C - loss_oracle|/(|loss_oracle|+1e-8) < 0.02 = CAPSTONE. Loss should
   drop by ~step 100. NC: wrong targets -> oracle diverges >2%.
Files: helixc/runtime/train_transformer.c (~600 lines), helixc/runtime/oracle_train.py (numpy, ~200).
