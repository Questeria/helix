# Helix GPU Backward + Training Blueprint (capstone path)

Authored 2026-06-01 from a verified code-architect pass. Drives the backward/training
ticks toward the v1.0 capstone: a 2-layer transformer trains on the RTX 3070 in pure
Helix-native (zero Python in the loop) within 2% of a PyTorch oracle.

## Confirmed emitter facts (kovc.hx)
- setp: ALL of lt/gt/eq/ne/le/ge live, float + int (kovc.hx:10572-10578). setp.eq.f32 usable.
- i32 params are SCALARS only (.param .u32, kovc.hx:11707-11711). NO i32-array params.
  => pass classification TARGETS and token IDS as f32 arrays; compare via __gpu_i2f.
- f32 SCALAR params unsupported (an f32 param becomes a .b64 pointer).
  => runtime scalars (Adam lr/eps/beta, bias-correction) are BAKED literals or 1-elem f32 arrays.
- caps: <=6 params/kernel, <=12 distinct let-vars/kernel.
- intrinsics: __gpu_exp (ex2.approx), __gpu_rsqrt (rsqrt.approx), __gpu_i2f (cvt.rn.f32.s32).
  NO __gpu_f2i (float->int) yet => embedding gather/scatter done HOST-SIDE for the capstone.

## Orchestration DECISION (confirmed): Option A -- C training harness
helixc/runtime/train_transformer.c does ONLY cuMemAlloc / cuMemcpy / cuLaunchKernel
sequencing + scalar bookkeeping. ALL training math = Helix-emitted PTX. C is a trusted
launcher (like ptxas/cuda_launch.c), NOT Python -> "zero Python in the loop" satisfied.
In-Helix dynamic linking (so a Helix ELF can dlopen libcuda) is a v1.0-FREEZE requirement
(D2), NOT a capstone blocker.

## Backward op list (ALL no-kovc-change)
- B1 CE+softmax grad wrt logits = softmax(logits) - onehot(target). Row-wise kernel.
  Targets as f32; onehot via (__gpu_i2f(t)==tg). FIRST op. ref: nn.hx:885 softmax_ce_grad_f32.
- B2 matmul-backward: dX = dY@W -> reuse naive_matmul (A@B). dW = X^T@dY -> NEW kernel
  gpu_matmul_atb (A^T@B, the 3rd variant; A@B=naive_matmul, A@B^T=gpu_qkt).
- B3 softmax-backward: dA[i,j]=P[i,j]*(dP[i,j]-sum_j dP*P). Needs saved P. Row-wise.
- B4 gelu-backward: dx=dy*gelu'(x). Elementwise. literals 0.5/1.0/2.0/0.7978846/0.044715/0.134145.
- B5 layernorm-backward: B5a dx (row-wise, needs saved xnorm + inv_std), B5b dgamma/dbeta
  (per-column reduction). Plus gpu_layernorm_fwd_save (forward + write inv_std[row]).
- B6 attention-backward: composes B2(atb)+B3+gpu_qkt+naive_matmul + a 0.25 scale kernel.
- B7 embedding fwd+bwd: HOST-SIDE (no __gpu_f2i). Trivial for V=32,S=16.
- Adam: gpu_adam (w,g,m,v,bc1,bc2); b1/b2/lr/eps baked literals; bias-correction bc1/bc2
  as 1-elem f32 arrays (recomputed host-side per step). Uses __gpu_rsqrt. N from gridDim.

## Capstone shape (minimal): B=1, S=16, d_model=16, 1 head (d=16 => 0.25 scale), MLP H=64,
vocab V=32, 2 layers, copy/next-token task, Adam, K steps. Save all forward activations
for backprop (list in the architect transcript / nn.hx mirror).

## PyTorch oracle (D1): fenced OFFLINE only. C harness writes init_weights.bin (seeded
xorshift32 init); oracle reads it -> identical init + data + steps; compare loss curves;
within 2% = capstone. Oracle restored from tag v0-pre-k4-full-with-python. NEVER in the
Helix training path. A python compare_curves.py is an audit tool, not training.

## Build order (each GPU-verified vs an INDEPENDENT ref + neg control, then audited)
P1 foundation: gpu_matmul_atb, gpu_layernorm_fwd_save, gpu_scale_inplace.
P2 backward: ce_softmax_grad (FIRST), softmax_backward, gelu_backward, layernorm_backward_dx
   + _dgamma_dbeta, gpu_adam.
P3 integrate: train_transformer.c (forward-only loss==oracle step0; then add backward op by
   op, each weight-grad vs finite-difference / oracle .grad; then Adam; K=500 within 2%).
P4 audit each kernel + the end-to-end oracle comparison.
ALL P1/P2 are no-kovc-change. Only future extras (GPU embedding via __gpu_f2i + i32-array
params; f32-scalar params) would need a gated kovc.hx re-mint -- NOT required for the capstone.
