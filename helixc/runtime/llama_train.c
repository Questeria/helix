/* llama_train.c -- SmolLM2-135M (Llama-arch) TRAINER: forward + FULL BACKWARD + Adam (P7-step3).
 *
 * A forward pass in a TRAINABLE shape: fp32 weights kept resident + EVERY intermediate activation
 * allocated and SAVED (not freed/reused), so the backward can read them. This is NOT inference
 * (which streams + reuses buffers); it is the training forward + backward.
 *
 * P7-step3 (this file): the backward reverses the forward via the saved intermediates + the 4 new
 * Llama bwd kernels (gpu_rmsnorm_bwd_dx, gpu_rope_bwd, gpu_silu_mul_bwd, gpu_repeat_kv_bwd) + the
 * capstone-shared bwd kernels (gpu_ce_softmax_grad, tiled_matmul_atb, gpu_softmax_backward, gpu_adam),
 * accumulating a gradient buffer per weight, then an Adam step (Param list, capstone pattern).
 *   --fdcheck (or HX_FDCHECK=1): MULTI-EPS central finite-difference gradient check (THE acceptance
 *       gate). Probes the max-|grad| cell of every tensor family (final-norm, tied-embed, all 7
 *       projections + both layer-norms @ the top layer, attn+MLP @ middle & bottom layers) and
 *       compares the analytic grad to (L(w+eps)-L(w-eps))/(2eps). PASS = rel-err < 5e-2 at the best
 *       eps in {1e-3,4e-3,1.5e-2,5e-2}. (Multi-eps because the fp32 loss-noise floor needs a LARGE
 *       eps for the tiny projection grads, while the big high-curvature RMSNorm grads need a SMALL
 *       eps; HX_FD_SWEEP dumps the per-eps convergence that establishes this.)
 *   --train N: N Adam steps on the input sequence (smoke: the loss overfits a single seq to ~0).
 * The tied embedding is a SINGLE resident buffer (w_embed_pad): the input gather (embed_gather, now
 * D2D from w_embed_pad) AND the lm_head AND the embedding grad all use it, so a perturbation/Adam
 * update is seen by every leg -- which is what makes the embedding finite-diff well-posed.
 *
 * Trusted-C launcher, EXACTLY like train_transformer.c (the GPT-2 capstone) and gpt2_infer.c
 * (the Llama inference forward): ALL arithmetic stays in kovc-emitted PTX kernels; this C only
 * moves bytes (mmap the fp32 weight file, host embedding gather, host RoPE-table precompute,
 * head<->device pack/scatter, H<->D copies, host CE-loss reduction over the final logits) and
 * sequences kernel launches. It REUSES the SAME kernels the inference forward + capstone use:
 *   tiled_matmul, tiled_matmul_abt, gpu_softmax_causal, gpu_rmsnorm_fwd_eps, gpu_rope_rot,
 *   gpu_silu_mul, gpu_scale_rt, vector_add  (one combined PTX module, loaded by name).
 *
 * NO kovc.hx edit: the kernels are compiled by the seed-minted driver into combined PTX (the
 * scripts/capstone_audit.sh / scripts/llama_model_gate.sh pattern). The fixpoint is UNTOUCHED.
 *
 * Weight-file layout MATCHED to gpt2_infer.c's --arch llama v2 reader (see load_weights):
 *   64B header: magic 'HXGW'@0, ver=2@4, nl@8 dm@12 nh@16 nv@20 nc@24 dff@28, nfloat@32(u64),
 *               arch=1@40, n_kv_heads@44, rope_theta@48(f32), rms_eps@52(f32).
 *   payload (fp32, at offset 64), per layer in build order:
 *     in_ln[DM]  q[DM,DM]  k[KVD,DM]  v[KVD,DM]  o[DM,DM]  post_ln[DM]
 *     gate[DFF,DM]  up[DFF,DM]  down[DM,DFF]                 (all Linear [out,in], HF/un-transposed)
 *   then globals: embed[NV,DM]  norm[DM].  Tied head: logits = x @ embed^T.
 *
 * Forward per layer (HF Llama, GQA): rmsnorm(eps 1e-5) -> q/k/v GEMM (A@B^T) -> per q-head
 *   { pack q (QD cols) + k/v from kv-head h/(NH/NKV) (KVD cols), RoPE-rotate_half q&k, scores=Q@K^T,
 *     *1/sqrt(DH), causal softmax, @V, scatter } -> o_proj (A@B^T) -> +residual -> rmsnorm ->
 *   SwiGLU { gate=A@B^T, up=A@B^T, y=up*silu(gate) } -> down (A@B^T) -> +residual.
 * Then final rmsnorm -> tied lm_head (A@B^T) -> logits -> shifted-CE loss.
 *
 * Build (WSL): gcc llama_train.c -O2 -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -lm -o $HOME/llama_train
 * Run:  $HOME/llama_train <combined.ptx> <smollm2-135m.weights> <ids.txt> [oracle_logits.bin]
 *   -> runs the forward (saving all intermediates), prints the loss, dumps the last-real-row
 *      logits to $HOME/llama_train_logits.bin, and (if oracle given) prints the max-abs logit
 *      diff + argmax match vs the oracle row. Tolerance for a PASS: max_abs < 1e-2 AND argmax==.
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

/* ---- model dims (set from the v2 weight header by load_weights) ---- */
static int   NL=30, DM=576, NH=9, NKV=3, NV=49152, DFF=1536;
static int   DH=64;                 /* head dim = DM/NH */
static int   QD=576, KVD=192;       /* QD=NH*DH, KVD=NKV*DH */
static float ROPE_THETA=100000.0f;
static float RMS_EPS=1e-5f;
static float ATTN_SCALE=0.125f;     /* 1/sqrt(DH) */

#define MAGIC   0x48584757u   /* 'HXGW' little-endian */
#define HDR_BYTES 64

/* ===================== CUDA plumbing ===================== */
static int check(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) { const char* m=0; cuGetErrorString(r,&m);
        fprintf(stderr, "CUDA %s: %s (%d)\n", what, m?m:"?", (int)r); return 1; }
    return 0;
}
#define CK(c,w)  do { if (check((c),(w))) return 2; } while (0)
#define CKX(c,w) do { if (check((c),(w))) exit(2); } while (0)

static CUcontext ctx;
static CUmodule  g_mod;
static char*     g_ptx=NULL;
/* kernel handles (the 8 forward kernels) */
static CUfunction f_mm, f_abt, f_sm_causal, f_rms, f_rope, f_silu, f_scale, f_add;
/* backward kernels (bound only in --fdcheck/train mode): A^T@B (weight grads + dX), the 4 new
 * Llama bwd ops, CE-softmax grad (loss root), softmax bwd (attn), Adam. */
static CUfunction f_atb, f_rmsb, f_ropeb, f_silub, f_rkvb, f_ceg, f_smb, f_adam;
/* v1.9 P5 (STE ternary QAT, opt-in HX_TERNARY_QAT): the 7 LINEARS (w_q/w_k/w_v/w_o + w_gate/w_up/
 * w_down) train with a latent fp32 W + a forward ternarize-dequant (Wt = clip3(W/sc)*sc, sc = per-
 * out-row abs-mean) consumed by mm_ABt in place of W, + a backward STE clip-mask on each dW. Adam
 * stays on the latent fp W. Default OFF -> the fp baseline + the fixpoint gate are unaffected (the
 * QAT path is dormant). Same pattern as train_transformer.c (the GPT-2 capstone). All 3 kernels are
 * @kernel (already element-exact gated; NO kovc.hx edit -- they join the combined-PTX concat list). */
static int QAT = 0;
static CUfunction f_ternarize, f_rowabsmean, f_stemask;
/* the ternarized buffers + per-out-row scale buffers (one per ternary linear, per layer) are
 * declared with the resident weights below (after #define MAXL). */

/* sync after every launch (TRAINER FORWARD = correctness-first; no fast-mode skips). */
#define SYNC(w) do { CKX(cuCtxSynchronize(), w); } while (0)
#define LX(fn,grid,block,args)     do { CKX(cuLaunchKernel((fn),(grid),1,1,(block),1,1,0,0,(args),0), #fn); SYNC("sync " #fn); } while (0)
#define LX2(fn,gx,gy,bx,by,args)   do { CKX(cuLaunchKernel((fn),(gx),(gy),1,(bx),(by),1,0,0,(args),0), #fn); SYNC("sync " #fn); } while (0)

static CUdeviceptr A(size_t nf) { CUdeviceptr p; CKX(cuMemAlloc(&p, nf*sizeof(float)), "alloc"); return p; }

/* ---- kernel launch wrappers (EXACT geometry from gpt2_infer.c's llama path) ---- */
/* C[M,N] = A[M,K] @ B[K,N]  (SMEM-tiled). grid=(N/64,M/64) block=(16,16); M,N %64, K %8. */
static void mm_AB(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, int M, int K, int N) {
    int m=M,k=K,n=N; void* ar[]={ &a,&b,&c,&m,&k,&n };
    LX2(f_mm, (unsigned)(N/64), (unsigned)(M/64), 16, 16, ar);
}
/* C[M,N] = A[M,K] @ B[N,K]^T  (SMEM-tiled A@B^T). Same geometry. */
static void mm_ABt(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, int M, int K, int N) {
    int m=M,k=K,n=N; void* ar[]={ &a,&b,&c,&m,&k,&n };
    LX2(f_abt, (unsigned)(N/64), (unsigned)(M/64), 16, 16, ar);
}
/* C[K,N] = A[M,K]^T @ B[M,N]  (SMEM-tiled A^T@B; contraction = M). The weight gradient
 * dW = X^T @ dY and the dX-into-inputs path. tiled_matmul_atb args (mm=M contraction = k-loop
 * bound, kk=K out-rows + A row stride, nn=N out-cols + B row stride); grid=(N/64,K/64). Needs
 * K%64==0, N%64==0, M%8==0 (Spad=64, DM/QD=576, KVD=192, DFF=1536, NVpad=49152 all qualify). */
static void mm_AtB(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, int M, int K, int N) {
    int m=M,k=K,n=N; void* ar[]={ &a,&b,&c,&m,&k,&n };
    LX2(f_atb, (unsigned)(N/64), (unsigned)(K/64), 16, 16, ar);
}
/* RMSNorm backward dx: dx = inv*(w*dy) - (x*inv^3/cols)*sum(w*dy*x). grid=rows block=1. */
static void rms_bwd_dx(CUdeviceptr x, CUdeviceptr w, CUdeviceptr dy, CUdeviceptr dx, int rows, int cols) {
    int c=cols; void* ar[]={ &x,&w,&dy,&dx,&c }; LX(f_rmsb, (unsigned)rows, 1, ar);
}
/* RoPE backward (transpose rotation) IN-PLACE on packed [rows,DH] (q holds dy in, dx out). */
static void rope_bwd(CUdeviceptr q, CUdeviceptr dcos, CUdeviceptr dsin, int rows) {
    int half=DH/2; void* ar[]={ &q,&dcos,&dsin,&half }; LX(f_ropeb, (unsigned)rows, 1, ar);
}
/* SwiGLU silu-mul backward: dg = dh*u*silu'(g), du = dh*silu(g). grid-stride exact tiling. */
static void silu_mul_bwd(CUdeviceptr g, CUdeviceptr u, CUdeviceptr dh, CUdeviceptr dg, CUdeviceptr du, int n) {
    int nn=n; void* ar[]={ &g,&u,&dh,&dg,&du,&nn };
    int blk=1, cand[]={256,128,64,32,1}; for (int i=0;i<5;i++) if (n%cand[i]==0){blk=cand[i];break;}
    LX(f_silub, (unsigned)(n/blk), (unsigned)blk, ar);
}
/* GQA repeat-KV backward: dout[kv,off] = sum_{r<nrep} din[(kv*nrep+r),off]. grid=nkv*blk block=1. */
static void repeat_kv_bwd(CUdeviceptr din, CUdeviceptr dout, int nrep, int blk, int nkv) {
    int n=nkv*blk; void* ar[]={ &din,&dout,&nrep,&blk,&n }; LX(f_rkvb, (unsigned)(nkv*blk), 1, ar);
}
/* CE+softmax grad: dlogits[r,:] = softmax(logits[r,:]) - onehot(tgtf[r]). grid=rows block=1. */
static void ce_softmax_grad(CUdeviceptr logits, CUdeviceptr tgtf, CUdeviceptr dlog, int rows, int cols) {
    int r=rows,c=cols; void* ar[]={ &logits,&tgtf,&dlog,&r,&c }; LX(f_ceg, (unsigned)rows, 1, ar);
}
/* softmax backward (Jacobian-vector): da = p*(dp - sum_k dp_k p_k). grid=rows block=1. */
static void softmax_bwd(CUdeviceptr p, CUdeviceptr dp, CUdeviceptr da, int rows, int cols) {
    int r=rows,c=cols; void* ar[]={ &p,&dp,&da,&r,&c }; LX(f_smb, (unsigned)rows, 1, ar);
}
/* Adam in-place step (lr=1e-3, b1=.9, b2=.999, eps=1e-8 baked; bc1/bc2 1-elem device scalars). */
static void adam_step1(CUdeviceptr w, CUdeviceptr g, CUdeviceptr m, CUdeviceptr v, CUdeviceptr bc1, CUdeviceptr bc2, int n) {
    void* ar[]={ &w,&g,&m,&v,&bc1,&bc2 };
    int blk=1, cand[]={256,128,64,32,1}; for (int i=0;i<5;i++) if (n%cand[i]==0){blk=cand[i];break;}
    LX(f_adam, (unsigned)(n/blk), (unsigned)blk, ar);
}
/* RMSNorm y[r,:]=x[r,:]*w/rms(x[r,:]); kernel bakes eps=1e-5. grid=rows block=1. */
static void rms_norm(CUdeviceptr x, CUdeviceptr y, CUdeviceptr w, int rows, int cols) {
    int c=cols; void* ar[]={ &x,&y,&w,&c }; LX(f_rms, (unsigned)rows, 1, ar);
}
/* RoPE rotate_half IN-PLACE on a packed [rows,DH] slab; table row s == position s. */
static void rope(CUdeviceptr q, CUdeviceptr dcos, CUdeviceptr dsin, int rows) {
    int half=DH/2; void* ar[]={ &q,&dcos,&dsin,&half }; LX(f_rope, (unsigned)rows, 1, ar);
}
/* causal row-softmax over [rows,cols]. grid=rows block=1. */
static void softmax_causal(CUdeviceptr x, CUdeviceptr y, int rows, int cols) {
    int r=rows,c=cols; void* ar[]={ &x,&y,&r,&c }; LX(f_sm_causal, (unsigned)rows, 1, ar);
}
/* SwiGLU y[i]=u[i]*silu(g[i]) over n elems. grid-stride NO guard -> exact tiling. */
static void silu_mul(CUdeviceptr g, CUdeviceptr u, CUdeviceptr y, int n) {
    int nn=n; void* ar[]={ &g,&u,&y,&nn };
    int blk=1, cand[]={256,128,64,32,1}; for (int i=0;i<5;i++) if (n%cand[i]==0){blk=cand[i];break;}
    LX(f_silu, (unsigned)(n/blk), (unsigned)blk, ar);
}
/* in-place a[i]*=s[0]. grid-stride NO guard -> exact tiling. */
static void scale_rt(CUdeviceptr a, CUdeviceptr s, int n) {
    int nn=n; void* ar[]={ &a,&s,&nn };
    int blk=1, cand[]={256,128,64,32,1}; for (int i=0;i<5;i++) if (n%cand[i]==0){blk=cand[i];break;}
    LX(f_scale, (unsigned)(n/blk), (unsigned)blk, ar);
}
/* c[i]=a[i]+b[i] (residual). grid-stride NO guard -> exact tiling. */
static void vadd(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, int n) {
    int nn=n; void* ar[]={ &a,&b,&c,&nn };
    int blk=1, cand[]={256,128,64,32,1}; for (int i=0;i<5;i++) if (n%cand[i]==0){blk=cand[i];break;}
    LX(f_add, (unsigned)(n/blk), (unsigned)blk, ar);
}
/* ===================== STE ternary QAT helpers (v1.9 P5; no-op unless QAT) =====================
 * EW: ternarize the latent W[out,in] into wt (sc[out] = per-out-row abs-mean) and return the buffer
 *     the forward matmul should consume (wt when QAT, else W untouched).
 * MASK: STE clip-gate dw[out,in] in-place AFTER its weight-grad matmul (zeros the grad where the
 *     latent saturated |W/sc|>1; passes it through where in-range). Adam then steps the latent W.
 * Launches pipeline on the default stream (same-stream ordering -> the ternarize precedes the
 * consuming mm_ABt, and the mask precedes the Adam step). Mirrors train_transformer.c's EW/MASK.
 *
 * COLUMN-TILING (the one delta vs the capstone): ternarize_dequant + ste_mask are launched
 *   grid=rows, block=cols because the kernel maps row=block_idx(), col=thread_idx() (kovc maps
 *   thread_idx->%tid.x, no grid-stride). A CUDA block is capped at 1024 threads, but w_down has
 *   in=DFF=1536>1024. The kernel uses its `cols` ARG as the row STRIDE, so we tile the columns by
 *   OFFSETTING the w/wt(/dw) base pointers by tile_start floats while keeping cols=full-stride and
 *   block=tile_width (<=1024): the kernel then computes w[row*cols + (tile_start+col)] for col in
 *   [0,tile_width), and sc[block_idx()] is the TRUE per-row scale (sc is NOT offset). Iterating
 *   tile_start over the row covers every column with the correct per-out-row scale -- byte-identical
 *   to a single block=cols launch where cols<=1024. row_abs_mean handles cols natively (block=1,
 *   internal while-loop), no tiling needed. (For the 6 linears with in=576<=1024 this is one tile.) */
static int ste_coltile(int cols) {              /* largest tile<=1024 that divides cols */
    int cand[]={1024,768,576,512,384,256,192,128,64,32,1};
    for (int i=0;i<(int)(sizeof(cand)/sizeof(cand[0]));i++) if (cols%cand[i]==0 && cand[i]<=1024) return cand[i];
    return 1;
}
static CUdeviceptr EW(CUdeviceptr w, CUdeviceptr wt, CUdeviceptr sc, int rows, int cols) {
    if (!QAT) return w;
    /* row_abs_mean(w,sc,rows,cols): grid=rows, block=1 (one thread per out-row; loops cols inside). */
    { int r=rows,c=cols; void* am[]={ &w,&sc,&r,&c };
      CKX(cuLaunchKernel(f_rowabsmean,(unsigned)rows,1,1, 1,1,1, 0,0, am,0), "ste rowabsmean");
      SYNC("sync ste rowabsmean"); }
    /* ternarize_dequant(w,sc,wt,rows,cols): grid=rows, block=tile, column-tiled (see header). */
    int tile=ste_coltile(cols);
    for (int t0=0;t0<cols;t0+=tile) {
        CUdeviceptr wo  = w  + (CUdeviceptr)((size_t)t0*sizeof(float));
        CUdeviceptr wto = wt + (CUdeviceptr)((size_t)t0*sizeof(float));
        int r=rows,c=cols; void* tn[]={ &wo,&sc,&wto,&r,&c };   /* c = FULL stride, NOT the tile */
        CKX(cuLaunchKernel(f_ternarize,(unsigned)rows,1,1, (unsigned)tile,1,1, 0,0, tn,0), "ste ternarize");
        SYNC("sync ste ternarize");
    }
    return wt;
}
static void MASK(CUdeviceptr dw, CUdeviceptr w, CUdeviceptr sc, int rows, int cols) {
    if (!QAT) return;
    int tile=ste_coltile(cols);
    for (int t0=0;t0<cols;t0+=tile) {
        CUdeviceptr dwo = dw + (CUdeviceptr)((size_t)t0*sizeof(float));
        CUdeviceptr wo  = w  + (CUdeviceptr)((size_t)t0*sizeof(float));
        int r=rows,c=cols; void* a[]={ &dwo,&wo,&sc,&r,&c };    /* c = FULL stride, NOT the tile */
        CKX(cuLaunchKernel(f_stemask,(unsigned)rows,1,1, (unsigned)tile,1,1, 0,0, a,0), "ste mask");
        SYNC("sync ste mask");
    }
}

/* strided per-row DtoD: dst[s,0:DH] = src[s, hbase:hbase+DH], src has srccols columns. */
static void pack_head(CUdeviceptr dst, CUdeviceptr src, int hbase, int srccols, int rows) {
    for (int s=0;s<rows;s++) {
        CUdeviceptr ss = src + (CUdeviceptr)((size_t)(s*srccols+hbase)*sizeof(float));
        CUdeviceptr sd = dst + (CUdeviceptr)((size_t)(s*DH)*sizeof(float));
        CKX(cuMemcpyDtoD(sd, ss, (size_t)DH*sizeof(float)), "pack head");
    }
}
static void scatter_head(CUdeviceptr dst, int hbase, CUdeviceptr src, int dststride, int rows) {
    for (int s=0;s<rows;s++) {
        CUdeviceptr ss = src + (CUdeviceptr)((size_t)(s*DH)*sizeof(float));
        CUdeviceptr sd = dst + (CUdeviceptr)((size_t)(s*dststride+hbase)*sizeof(float));
        CKX(cuMemcpyDtoD(sd, ss, (size_t)DH*sizeof(float)), "scatter head");
    }
}

/* ===================== weight file (mmap, fp32 v2 llama layout) ===================== */
static const float* g_wbase=NULL;   /* mmap payload base (float*, at file offset 64) */
static void*  g_map=NULL; static size_t g_maplen=0; static int g_fd=-1;

/* per-layer + globals float offsets, EXACTLY gpt2_infer.c layer_offsets_ll / off_embed_ll. */
static long per_layer_floats(void) {
    return (long)DM + (long)DM*DM + 2L*(long)KVD*DM + (long)DM*DM
         + (long)DM + 2L*(long)DFF*DM + (long)DM*DFF;
}
typedef struct { long inln,qW,kW,vW,oW,postln,gateW,upW,downW; } LayerOff;
static LayerOff layer_off(void) {
    LayerOff o; long p=0;
    o.inln=p;   p+=DM;
    o.qW=p;     p+=(long)DM*DM;
    o.kW=p;     p+=(long)KVD*DM;
    o.vW=p;     p+=(long)KVD*DM;
    o.oW=p;     p+=(long)DM*DM;
    o.postln=p; p+=DM;
    o.gateW=p;  p+=(long)DFF*DM;
    o.upW=p;    p+=(long)DFF*DM;
    o.downW=p;  p+=(long)DM*DFF;
    return o;
}
static long off_layer(int L) { return (long)L*per_layer_floats(); }
static long off_embed(void)  { return (long)NL*per_layer_floats(); }
static long off_normf(void)  { return off_embed() + (long)NV*DM; }

static int load_weights(const char* path) {
    g_fd = open(path, O_RDONLY);
    if (g_fd < 0) { fprintf(stderr, "open weights '%s': %s\n", path, strerror(errno)); return 2; }
    struct stat st; if (fstat(g_fd,&st)!=0) { fprintf(stderr, "fstat\n"); return 2; }
    g_maplen=(size_t)st.st_size;
    g_map=mmap(NULL, g_maplen, PROT_READ, MAP_PRIVATE, g_fd, 0);
    if (g_map==MAP_FAILED) { fprintf(stderr, "mmap: %s\n", strerror(errno)); return 2; }
    madvise(g_map, g_maplen, MADV_WILLNEED);
    const unsigned char* hb=(const unsigned char*)g_map;
    uint32_t magic,ver,nl,dm,nh,nv,nc,dff,arch,nkv; uint64_t nfloat; float th,ep;
    memcpy(&magic,hb+0,4);  memcpy(&ver,hb+4,4);   memcpy(&nl,hb+8,4);  memcpy(&dm,hb+12,4);
    memcpy(&nh,hb+16,4);    memcpy(&nv,hb+20,4);   memcpy(&nc,hb+24,4); memcpy(&dff,hb+28,4);
    memcpy(&nfloat,hb+32,8);
    memcpy(&arch,hb+40,4);  memcpy(&nkv,hb+44,4);  memcpy(&th,hb+48,4); memcpy(&ep,hb+52,4);
    if (magic!=MAGIC) { fprintf(stderr, "bad magic 0x%08x\n", magic); return 2; }
    if (ver!=2u || arch!=1u) { fprintf(stderr, "not a v2 llama weight file (ver=%u arch=%u)\n", ver, arch); return 2; }
    NL=(int)nl; DM=(int)dm; NH=(int)nh; NV=(int)nv; DFF=(int)dff; NKV=(int)nkv;
    ROPE_THETA=th; RMS_EPS=ep;
    DH=DM/NH; KVD=NKV*DH; QD=NH*DH; ATTN_SCALE=1.0f/sqrtf((float)DH);
    if (NKV<=0 || NH%NKV) { fprintf(stderr, "bad GQA %d/%d\n", NH, NKV); return 2; }
    if (fabsf(RMS_EPS-1e-5f) > 1e-9f) { fprintf(stderr, "rms_eps %g != the kernel's baked 1e-5\n", (double)RMS_EPS); return 2; }
    size_t want=(size_t)HDR_BYTES + nfloat*4;
    if (g_maplen < want) { fprintf(stderr, "short weight file %zu < %zu\n", g_maplen, want); return 2; }
    size_t expect=(size_t)off_normf()+DM;
    if (expect != (size_t)nfloat) { fprintf(stderr, "layout %zu floats != header n_float %llu\n", expect, (unsigned long long)nfloat); return 2; }
    g_wbase=(const float*)(hb+HDR_BYTES);
    printf("[wt] mmap %zu B, n_float=%llu, llama layout verified (per_layer=%ld)\n",
           g_maplen, (unsigned long long)nfloat, per_layer_floats());
    printf("[cfg] NL=%d DM=%d NH=%d NKV=%d DH=%d QD=%d KVD=%d NV=%d DFF=%d theta=%g eps=%g scale=%g\n",
           NL, DM, NH, NKV, DH, QD, KVD, NV, DFF, (double)ROPE_THETA, (double)RMS_EPS, (double)ATTN_SCALE);
    return 0;
}

/* ===================== RESIDENT fp32 weights (trainable shape: kept on device) ===================== */
#define MAXL 64
static CUdeviceptr w_inln[MAXL], w_q[MAXL], w_k[MAXL], w_v[MAXL], w_o[MAXL],
                   w_postln[MAXL], w_gate[MAXL], w_up[MAXL], w_down[MAXL];
static CUdeviceptr w_normf, w_embed_pad;   /* final RMSNorm weight; tied head = embed padded [NVpad,DM] */
static int NVpad=0;

/* v1.9 P5 (STE ternary QAT): per ternary linear, a ternarized buffer wt_*[out,in] (same shape as W)
 * + a per-out-row scale sc_*[out]. Allocated only when QAT (in upload_all_weights). The 7 linears
 * ternarized: w_q[QD,DM] w_k[KVD,DM] w_v[KVD,DM] w_o[DM,QD] w_gate[DFF,DM] w_up[DFF,DM] w_down[DM,DFF].
 * The norms (w_inln/w_postln/w_normf) and the tied embedding (w_embed_pad) stay fp -- not ternarized. */
static CUdeviceptr wt_q[MAXL], wt_k[MAXL], wt_v[MAXL], wt_o[MAXL], wt_gate[MAXL], wt_up[MAXL], wt_down[MAXL];
static CUdeviceptr sc_q[MAXL], sc_k[MAXL], sc_v[MAXL], sc_o[MAXL], sc_gate[MAXL], sc_up[MAXL], sc_down[MAXL];

static CUdeviceptr up_slice(long foff, size_t nf) {
    CUdeviceptr d=A(nf); CKX(cuMemcpyHtoD(d, &g_wbase[foff], nf*sizeof(float)), "h2d wslice"); return d;
}
static int upload_all_weights(void) {
    LayerOff lo=layer_off();
    for (int L=0;L<NL;L++) {
        long b=off_layer(L);
        w_inln[L]  = up_slice(b+lo.inln,   DM);
        w_q[L]     = up_slice(b+lo.qW,     (size_t)DM*DM);
        w_k[L]     = up_slice(b+lo.kW,     (size_t)KVD*DM);
        w_v[L]     = up_slice(b+lo.vW,     (size_t)KVD*DM);
        w_o[L]     = up_slice(b+lo.oW,     (size_t)DM*DM);
        w_postln[L]= up_slice(b+lo.postln, DM);
        w_gate[L]  = up_slice(b+lo.gateW,  (size_t)DFF*DM);
        w_up[L]    = up_slice(b+lo.upW,    (size_t)DFF*DM);
        w_down[L]  = up_slice(b+lo.downW,  (size_t)DM*DFF);
    }
    w_normf = up_slice(off_normf(), DM);
    NVpad = ((NV+63)/64)*64;                       /* 49152 is already %64 -> 49152 */
    w_embed_pad = A((size_t)NVpad*DM);
    CKX(cuMemsetD8(w_embed_pad, 0, (size_t)NVpad*DM*sizeof(float)), "zero embed_pad");
    CKX(cuMemcpyHtoD(w_embed_pad, &g_wbase[off_embed()], (size_t)NV*DM*sizeof(float)), "h2d embed_pad");
    printf("[wt] resident: %d layers fp32 on-device + final-norm + tied head [%d,%d]\n", NL, NVpad, DM);
    /* v1.9 P5: QAT ternarized + scale buffers (only when HX_TERNARY_QAT). One per ternary linear. */
    if (QAT) {
        for (int L=0;L<NL;L++) {
            wt_q[L]   =A((size_t)QD*DM);  sc_q[L]   =A(QD);
            wt_k[L]   =A((size_t)KVD*DM); sc_k[L]   =A(KVD);
            wt_v[L]   =A((size_t)KVD*DM); sc_v[L]   =A(KVD);
            wt_o[L]   =A((size_t)DM*QD);  sc_o[L]   =A(DM);
            wt_gate[L]=A((size_t)DFF*DM); sc_gate[L]=A(DFF);
            wt_up[L]  =A((size_t)DFF*DM); sc_up[L]  =A(DFF);
            wt_down[L]=A((size_t)DM*DFF); sc_down[L]=A(DM);
        }
        printf("[qat] STE ternary QAT ON: ternarized+scale buffers for 7 linears x %d layers allocated\n", NL);
    }
    return 0;
}

/* ===================== SAVED intermediates (per layer; NOT reused, NOT freed) =====================
 * The backward (P7-step3) reads these. Buffers sized for Spad (T padded to %64). Per layer L:
 *   s_resid_in[L]  [S,DM]   block input (== residual stream entering the layer)
 *   s_rms1[L]      [S,DM]   rmsnorm(in)            (attn pre-projection)
 *   s_q[L]         [S,QD]   q_proj output (PRE-RoPE)        -- saved BEFORE rope for backward
 *   s_k[L]         [S,KVD]  k_proj output (PRE-RoPE)
 *   s_v[L]         [S,KVD]  v_proj output
 *   s_qr[L]        [S,QD]   q AFTER RoPE (rotate_half), per-head scattered back
 *   s_kr[L]        [S,KVD]  k AFTER RoPE
 *   s_attnw[L]     [NH,S,S] per-head softmax(scores) probabilities (backward needs the probs)
 *   s_ctx[L]       [S,QD]   merged attention context (concat of per-head @V)
 *   s_oproj[L]     [S,DM]   o_proj output
 *   s_resid_mid[L] [S,DM]   x after the attention residual add (== input to the MLP rmsnorm)
 *   s_rms2[L]      [S,DM]   rmsnorm(resid_mid)    (mlp pre-projection)
 *   s_gate[L]      [S,DFF]  gate_proj output
 *   s_up[L]        [S,DFF]  up_proj output
 *   s_silu[L]      [S,DFF]  up * silu(gate)
 *   s_down[L]      [S,DFF->DM] down_proj output ([S,DM])
 *   s_resid_out[L] [S,DM]   x after the MLP residual add (== next layer's resid_in)
 * Globals: s_rmsf [S,DM] final rmsnorm; d_logits [S,NVpad].
 * Scratch (per-head, reused within attention only -- NOT a saved activation): d_Qh/d_Kh/d_Vh/
 *   d_scores/d_attnw_h/d_aoh.  (scores per head are recomputable from saved q_r/k_r; we save the
 *   softmax PROBS s_attnw which the backward needs and cannot cheaply recompute.) */
static CUdeviceptr s_resid_in[MAXL], s_rms1[MAXL], s_q[MAXL], s_k[MAXL], s_v[MAXL],
                   s_qr[MAXL], s_kr[MAXL], s_attnw[MAXL], s_ctx[MAXL], s_oproj[MAXL],
                   s_resid_mid[MAXL], s_rms2[MAXL], s_gate[MAXL], s_up[MAXL], s_silu[MAXL],
                   s_down[MAXL], s_resid_out[MAXL];
static CUdeviceptr s_rmsf, d_logits;
/* per-head attention scratch (within-attn reuse) */
static CUdeviceptr d_Qh, d_Kh, d_Vh, d_scores, d_attnw_h, d_aoh;
/* RoPE tables [Smax, DH/2] */
static CUdeviceptr d_cos, d_sin, d_scale_buf;
static int Spad=0;

static int alloc_saved(int S) {
    Spad=S;
    for (int L=0;L<NL;L++) {
        s_resid_in[L]  = A((size_t)S*DM);
        s_rms1[L]      = A((size_t)S*DM);
        s_q[L]         = A((size_t)S*QD);
        s_k[L]         = A((size_t)S*KVD);
        s_v[L]         = A((size_t)S*KVD);
        s_qr[L]        = A((size_t)S*QD);
        s_kr[L]        = A((size_t)S*KVD);
        s_attnw[L]     = A((size_t)NH*S*S);
        s_ctx[L]       = A((size_t)S*QD);
        s_oproj[L]     = A((size_t)S*DM);
        s_resid_mid[L] = A((size_t)S*DM);
        s_rms2[L]      = A((size_t)S*DM);
        s_gate[L]      = A((size_t)S*DFF);
        s_up[L]        = A((size_t)S*DFF);
        s_silu[L]      = A((size_t)S*DFF);
        s_down[L]      = A((size_t)S*DM);
        s_resid_out[L] = A((size_t)S*DM);
    }
    s_rmsf   = A((size_t)S*DM);
    d_logits = A((size_t)S*NVpad);
    /* per-head scratch */
    d_Qh=A((size_t)S*DH); d_Kh=A((size_t)S*DH); d_Vh=A((size_t)S*DH);
    d_scores=A((size_t)S*S); d_attnw_h=A((size_t)S*S); d_aoh=A((size_t)S*DH);
    d_scale_buf=A(1); CKX(cuMemcpyHtoD(d_scale_buf,&ATTN_SCALE,sizeof(float)),"h2d scale");
    /* RoPE tables, host-built in double then cast f32 (gpt2_infer.c convention) */
    int half=DH/2;
    float* hc=(float*)malloc((size_t)S*half*sizeof(float));
    float* hs=(float*)malloc((size_t)S*half*sizeof(float));
    for (int s=0;s<S;s++) for (int j=0;j<half;j++) {
        double inv=pow((double)ROPE_THETA, -2.0*(double)j/(double)DH);
        double ang=(double)s*inv;
        hc[(size_t)s*half+j]=(float)cos(ang);
        hs[(size_t)s*half+j]=(float)sin(ang);
    }
    d_cos=A((size_t)S*half); d_sin=A((size_t)S*half);
    CKX(cuMemcpyHtoD(d_cos,hc,(size_t)S*half*sizeof(float)),"h2d cos");
    CKX(cuMemcpyHtoD(d_sin,hs,(size_t)S*half*sizeof(float)),"h2d sin");
    free(hc); free(hs);
    /* size report */
    double mb = ( (double)NL*( (size_t)S*DM*6 + (size_t)S*QD*3 + (size_t)S*KVD*3
                  + (size_t)NH*S*S + (size_t)S*DFF*3 )
                + (size_t)S*DM + (size_t)S*NVpad ) * 4.0 / (1024*1024);
    printf("[save] intermediates allocated for S=%d (T-padded): ~%.1f MB activations (all kept)\n", S, mb);
    return 0;
}

/* ===================== BACKWARD: gradient buffers + activation-grad scratch (P7-step3) =====================
 * One grad buffer PER resident weight (same shape as the weight), zeroed before each backward.
 * Per layer: g_inln[DM] g_q[QD,DM] g_k[KVD,DM] g_v[KVD,DM] g_o[DM,QD] g_postln[DM]
 *            g_gate[DFF,DM] g_up[DFF,DM] g_down[DM,DFF].  Globals: g_normf[DM], g_embed[NVpad,DM].
 * Activation-grad scratch (reused across layers): d_x  = grad flowing INTO the block (d_resid_out),
 * plus the per-stage intermediate grads. The per-head attention bwd reuses dh_*; the GQA fold
 * uses d_krep/d_vrep ([NH,S,DH], one slab per q-head) -> repeat_kv_bwd -> d_kfold/d_vfold ([NKV,S,DH]).
 * NOTE on the rmsnorm WEIGHT grad: there is no rmsnorm-dw kernel (RMSNorm has only the scale w,
 * no beta), so dw_rms[c] = sum_s dy[s,c]*x[s,c]*inv[s] is reduced ON THE HOST (recomputing inv per
 * row from the saved input x) -- exactly like the existing host CE reduction; the kernel path still
 * does all the dx arithmetic. Same for the tied-embedding input-scatter at the very end. */
static CUdeviceptr g_inln[MAXL], g_q[MAXL], g_k[MAXL], g_v[MAXL], g_o[MAXL],
                   g_postln[MAXL], g_gate[MAXL], g_up[MAXL], g_down[MAXL];
static CUdeviceptr g_normf, g_embed;
/* activation grads (reused each layer) */
static CUdeviceptr dg_logits, d_targets, dg_x, dg_rms, dg_tmpDM, dg_oproj, dg_ctx,
                   dg_gate, dg_up, dg_silu, dg_q, dg_k, dg_v, dg_kfold, dg_vfold,
                   dg_krep, dg_vrep;
/* per-head attention bwd scratch */
static CUdeviceptr dh_aoh, dh_attnw, dh_scores, dh_Qh, dh_Kh, dh_Vh;
static CUdeviceptr d_bc1, d_bc2, d_unscale;   /* Adam bias-correction scalars; 1/sqrt(DH) for unscale */

/* per-weight m/v Adam state, one slot per Param (built in build_params) */
typedef struct { CUdeviceptr w, g, m, v; int n; } Param;
static Param params[64*16]; static int nparams=0;

static int alloc_grads(int S) {
    for (int L=0;L<NL;L++) {
        g_inln[L]  = A(DM);                 g_q[L]   = A((size_t)QD*DM);
        g_k[L]     = A((size_t)KVD*DM);     g_v[L]   = A((size_t)KVD*DM);
        g_o[L]     = A((size_t)DM*QD);      g_postln[L]= A(DM);
        g_gate[L]  = A((size_t)DFF*DM);     g_up[L]  = A((size_t)DFF*DM);
        g_down[L]  = A((size_t)DM*DFF);
    }
    g_normf = A(DM);
    g_embed = A((size_t)NVpad*DM);
    dg_logits = A((size_t)S*NVpad);  d_targets = A((size_t)S);
    dg_x      = A((size_t)S*DM);     dg_rms    = A((size_t)S*DM);  dg_tmpDM = A((size_t)S*DM);
    dg_oproj  = A((size_t)S*DM);     dg_ctx    = A((size_t)S*QD);
    dg_gate   = A((size_t)S*DFF);    dg_up     = A((size_t)S*DFF); dg_silu  = A((size_t)S*DFF);
    dg_q      = A((size_t)S*QD);     dg_k      = A((size_t)S*KVD); dg_v     = A((size_t)S*KVD);
    dg_kfold  = A((size_t)NKV*S*DH); dg_vfold  = A((size_t)NKV*S*DH);
    dg_krep   = A((size_t)NH*S*DH);  dg_vrep   = A((size_t)NH*S*DH);
    dh_aoh=A((size_t)S*DH); dh_attnw=A((size_t)S*S); dh_scores=A((size_t)S*S);
    dh_Qh=A((size_t)S*DH); dh_Kh=A((size_t)S*DH); dh_Vh=A((size_t)S*DH);
    d_bc1=A(1); d_bc2=A(1);
    d_unscale=A(1); CKX(cuMemcpyHtoD(d_unscale,&ATTN_SCALE,sizeof(float)),"h2d unscale");
    printf("[bwd] gradient buffers + activation-grad scratch allocated (S=%d)\n", S);
    return 0;
}
static void zero_dev(CUdeviceptr d, size_t nf) { CKX(cuMemsetD8(d, 0, nf*sizeof(float)), "zero grad"); }
static void zero_all_grads(void) {
    for (int L=0;L<NL;L++) {
        zero_dev(g_inln[L],DM); zero_dev(g_q[L],(size_t)QD*DM); zero_dev(g_k[L],(size_t)KVD*DM);
        zero_dev(g_v[L],(size_t)KVD*DM); zero_dev(g_o[L],(size_t)DM*QD); zero_dev(g_postln[L],DM);
        zero_dev(g_gate[L],(size_t)DFF*DM); zero_dev(g_up[L],(size_t)DFF*DM); zero_dev(g_down[L],(size_t)DM*DFF);
    }
    zero_dev(g_normf,DM); zero_dev(g_embed,(size_t)NVpad*DM);
}

/* host reduction of an RMSNorm weight grad: dw[c] += sum_s dy[s,c]*x[s,c]*inv_s, inv_s recomputed
 * from the saved rmsnorm INPUT x (mean of x^2 + eps). Accumulates INTO the device grad gw[DM]. Only
 * the T real rows contribute (padded rows are zero activations -> zero contribution, but we sum all
 * Spad rows for generality; padded x rows are 0 so inv is rsqrt(eps) and dy rows are 0 -> no effect). */
static void rmsnorm_wgrad_host(CUdeviceptr d_x_in, CUdeviceptr d_dy, CUdeviceptr gw, int rows, int cols) {
    float* x =(float*)malloc((size_t)rows*cols*sizeof(float));
    float* dy=(float*)malloc((size_t)rows*cols*sizeof(float));
    float* acc=(float*)calloc((size_t)cols,sizeof(float));
    float* gwh=(float*)malloc((size_t)cols*sizeof(float));
    CKX(cuMemcpyDtoH(x ,d_x_in,(size_t)rows*cols*sizeof(float)),"d2h rms x");
    CKX(cuMemcpyDtoH(dy,d_dy ,(size_t)rows*cols*sizeof(float)),"d2h rms dy");
    for (int s=0;s<rows;s++) {
        double ss=0.0; const float* xr=&x[(size_t)s*cols]; const float* dr=&dy[(size_t)s*cols];
        for (int c=0;c<cols;c++) ss += (double)xr[c]*xr[c];
        double inv = 1.0/sqrt(ss/(double)cols + 1e-5);
        for (int c=0;c<cols;c++) acc[c] += (float)((double)dr[c]*(double)xr[c]*inv);
    }
    CKX(cuMemcpyDtoH(gwh,gw,(size_t)cols*sizeof(float)),"d2h gw");
    for (int c=0;c<cols;c++) gwh[c]+=acc[c];
    CKX(cuMemcpyHtoD(gw,gwh,(size_t)cols*sizeof(float)),"h2d gw");
    free(x); free(dy); free(acc); free(gwh);
}

/* ===================== embedding gather (from the RESIDENT tied weight) =====================
 * llama: token embedding ONLY (RoPE replaces positional). The embedding is TIED to the lm_head,
 * so it must be a SINGLE source of truth: we gather rows from the resident device buffer
 * w_embed_pad (NOT the host mmap), so an Adam update to the embedding (or an fdcheck perturbation)
 * is seen by BOTH the input gather and the lm_head. D2D per-row copy into s_resid_in[0]. */
static void embed_gather(const int* ids, int T, int S, CUdeviceptr dst) {
    CKX(cuMemsetD8(dst, 0, (size_t)S*DM*sizeof(float)), "zero embed dst");
    for (int s=0;s<T;s++)
        CKX(cuMemcpyDtoD(dst + (CUdeviceptr)((size_t)s*DM*sizeof(float)),
                         w_embed_pad + (CUdeviceptr)((size_t)ids[s]*DM*sizeof(float)),
                         (size_t)DM*sizeof(float)), "d2d embed gather");
}

/* ===================== the forward (saving every intermediate) ===================== */
static void forward_layer(int L) {
    int group=NH/NKV;                            /* GQA group (9/3=3) */
    CUdeviceptr x = s_resid_in[L];               /* block input (== prev resid_out, or embeds for L0) */
    /* --- attention --- */
    rms_norm(x, s_rms1[L], w_inln[L], Spad, DM);
    /* QAT: ternarize the latent W (sc=per-out-row abs-mean) and consume Wt in the GEMM; else W. */
    mm_ABt(s_rms1[L], EW(w_q[L],wt_q[L],sc_q[L],QD,DM),  s_q[L], Spad, DM, QD);    /* q [S,QD] */
    mm_ABt(s_rms1[L], EW(w_k[L],wt_k[L],sc_k[L],KVD,DM), s_k[L], Spad, DM, KVD);   /* k [S,KVD] */
    mm_ABt(s_rms1[L], EW(w_v[L],wt_v[L],sc_v[L],KVD,DM), s_v[L], Spad, DM, KVD);   /* v [S,KVD] */
    for (int h=0; h<NH; h++) {
        int kv=h/group;
        pack_head(d_Qh, s_q[L], h*DH,  QD,  Spad);
        pack_head(d_Kh, s_k[L], kv*DH, KVD, Spad);
        pack_head(d_Vh, s_v[L], kv*DH, KVD, Spad);
        rope(d_Qh, d_cos, d_sin, Spad);          /* rotate_half in place; position == row */
        rope(d_Kh, d_cos, d_sin, Spad);
        scatter_head(s_qr[L], h*DH, d_Qh, QD, Spad);    /* save roped Q (for backward) */
        scatter_head(s_kr[L], kv*DH, d_Kh, KVD, Spad);  /* save roped K (group head only -> overwrites identically) */
        mm_ABt(d_Qh, d_Kh, d_scores, Spad, DH, Spad);   /* scores[S,S] = Q_h @ K_h^T */
        scale_rt(d_scores, d_scale_buf, Spad*Spad);     /* *1/sqrt(DH) */
        softmax_causal(d_scores, d_attnw_h, Spad, Spad);
        /* save this head's softmax probs into s_attnw[L] at head slab h */
        CKX(cuMemcpyDtoD(s_attnw[L] + (CUdeviceptr)((size_t)h*Spad*Spad*sizeof(float)),
                         d_attnw_h, (size_t)Spad*Spad*sizeof(float)), "save attnw head");
        mm_AB(d_attnw_h, d_Vh, d_aoh, Spad, Spad, DH);  /* ao_h[S,DH] = attn @ V_h */
        scatter_head(s_ctx[L], h*DH, d_aoh, QD, Spad);  /* concat heads -> ctx [S,QD] */
    }
    mm_ABt(s_ctx[L], EW(w_o[L],wt_o[L],sc_o[L],DM,QD), s_oproj[L], Spad, QD, DM);  /* o_proj [S,DM] */
    vadd(x, s_oproj[L], s_resid_mid[L], Spad*DM);        /* x + attn (residual) */
    /* --- MLP (SwiGLU) --- */
    rms_norm(s_resid_mid[L], s_rms2[L], w_postln[L], Spad, DM);
    mm_ABt(s_rms2[L], EW(w_gate[L],wt_gate[L],sc_gate[L],DFF,DM), s_gate[L], Spad, DM, DFF);
    mm_ABt(s_rms2[L], EW(w_up[L],  wt_up[L],  sc_up[L],  DFF,DM), s_up[L],   Spad, DM, DFF);
    silu_mul(s_gate[L], s_up[L], s_silu[L], Spad*DFF);    /* up * silu(gate) */
    mm_ABt(s_silu[L], EW(w_down[L],wt_down[L],sc_down[L],DM,DFF), s_down[L], Spad, DFF, DM);
    vadd(s_resid_mid[L], s_down[L], s_resid_out[L], Spad*DM);  /* + residual */
}

/* run the full forward for ids[0..T); copy the LAST REAL-TOKEN logit row [NV] to out (host). */
static void forward(const int* ids, int T, float* out_last_logits) {
    embed_gather(ids, T, Spad, s_resid_in[0]);
    for (int L=0; L<NL; L++) {
        forward_layer(L);
        if (L+1 < NL) CKX(cuMemcpyDtoD(s_resid_in[L+1], s_resid_out[L], (size_t)Spad*DM*sizeof(float)), "chain residual");
    }
    rms_norm(s_resid_out[NL-1], s_rmsf, w_normf, Spad, DM);   /* final RMSNorm */
    mm_ABt(s_rmsf, w_embed_pad, d_logits, Spad, DM, NVpad);   /* tied lm_head: logits = x @ embed^T */
    CKX(cuMemcpyDtoH(out_last_logits, d_logits + (CUdeviceptr)((size_t)(T-1)*NVpad*sizeof(float)),
                     (size_t)NV*sizeof(float)), "d2h last-row logits");
}

/* ===================== the BACKWARD (reverse forward via saved intermediates) =====================
 * Shifted-CE loss = mean_{t in [0,T-1)} -log softmax(logits_t)[ids[t+1]]. So ONLY logit rows
 * 0..T-2 carry gradient (row t predicts ids[t+1]); row T-1 + the padded rows T..Spad-1 are zeroed.
 * The host CE reduction divides by (T-1); the analytic grad must match that 1/(T-1) scale, so after
 * the fused (softmax - onehot) we scale every live row by 1/(T-1).
 *
 * Linear-layer conventions (forward y = x @ W^T, W is HF [out,in]):
 *   dX = dY @ W           -> mm_AB (contraction = out)
 *   dW = dY^T @ X         -> mm_AtB(M=rows, K=out, N=in) -> [out,in]   (accumulate? no: tiled_atb
 *                            OVERWRITES, so for the few weights touched once per layer we write to a
 *                            fresh per-layer grad buffer; layers are distinct buffers so no clobber).
 * Residual adds route the incoming grad to BOTH branches (copy, then add the branch grad).            */

static int g_T=0;   /* real token count (set by backward()) for the 1/(T-1) loss scale + row masking */

/* backward through ONE layer. dIN = grad of this layer's OUTPUT (s_resid_out[L]); on return dg_x
 * holds the grad of this layer's INPUT (s_resid_in[L]). All weight grads written to g_*[L]. */
static void backward_layer(int L, CUdeviceptr dIN) {
    int group=NH/NKV;
    /* ---- MLP backward ----  resid_out = resid_mid + down(silu(gate,up)) ; both branches get dIN. */
    /* down: s_down = silu @ w_down^T (w_down [DM,DFF]); dIN flows into the down output [S,DM]. */
    mm_AB (dIN, w_down[L], dg_silu, Spad, DM, DFF);      /* d_silu = dDown @ w_down  [S,DFF] */
    mm_AtB(dIN, s_silu[L], g_down[L], Spad, DM, DFF);    /* dW_down = dDown^T @ silu [DM,DFF] */
    MASK(g_down[L], w_down[L], sc_down[L], DM, DFF);     /* QAT STE clip-mask (no-op unless QAT) */
    /* silu_mul: silu = up * silu(gate) ; d_silu -> dg_gate (into gate), dg_up (into up) */
    silu_mul_bwd(s_gate[L], s_up[L], dg_silu, dg_gate, dg_up, Spad*DFF);
    /* up: s_up = rms2 @ w_up^T  ; gate: s_gate = rms2 @ w_gate^T. Both feed dg_rms (sum into rms2 grad). */
    mm_AtB(dg_up,   s_rms2[L], g_up[L],   Spad, DFF, DM);   /* dW_up   = dUp^T  @ rms2 [DFF,DM] */
    MASK(g_up[L],   w_up[L],   sc_up[L],   DFF, DM);
    mm_AtB(dg_gate, s_rms2[L], g_gate[L], Spad, DFF, DM);   /* dW_gate = dGate^T@ rms2 [DFF,DM] */
    MASK(g_gate[L], w_gate[L], sc_gate[L], DFF, DM);
    mm_AB (dg_up,   w_up[L],   dg_rms,   Spad, DFF, DM);    /* d_rms2 (from up)   [S,DM] */
    mm_AB (dg_gate, w_gate[L], dg_tmpDM, Spad, DFF, DM);    /* d_rms2 (from gate) [S,DM] */
    vadd(dg_rms, dg_tmpDM, dg_rms, Spad*DM);                /* d_rms2 total = up + gate */
    /* rmsnorm2: s_rms2 = rmsnorm(resid_mid) * w_postln. dW_postln (host reduce); d_resid_mid via dx. */
    rmsnorm_wgrad_host(s_resid_mid[L], dg_rms, g_postln[L], Spad, DM);
    rms_bwd_dx(s_resid_mid[L], w_postln[L], dg_rms, dg_tmpDM, Spad, DM);  /* d_resid_mid (MLP branch) [S,DM] */
    /* resid_mid = resid_in + oproj : the grad reaching resid_mid is (dIN through residual) + dg_tmpDM. */
    vadd(dIN, dg_tmpDM, dg_x, Spad*DM);   /* dg_x = grad of resid_mid (== both branches of the 2nd add) */

    /* ---- attention backward ----  resid_mid = resid_in + oproj(ctx) ; dg_x is grad of resid_mid. */
    /* o_proj: s_oproj = ctx @ w_o^T (w_o [DM,QD]). dg_x is the grad into oproj output [S,DM]. */
    mm_AB (dg_x, w_o[L], dg_ctx, Spad, DM, QD);     /* d_ctx = dOproj @ w_o  [S,QD] */
    mm_AtB(dg_x, s_ctx[L], g_o[L], Spad, DM, QD);   /* dW_o = dOproj^T @ ctx [DM,QD] */
    MASK(g_o[L], w_o[L], sc_o[L], DM, QD);
    /* per q-head: ctx_h = attnw_h @ V_h ; scores -> softmax -> attnw ; scores = Qr_h @ Kr_h^T. */
    zero_dev(dg_krep,(size_t)NH*Spad*DH); zero_dev(dg_vrep,(size_t)NH*Spad*DH);
    for (int h=0; h<NH; h++) {
        int kv=h/group;
        CUdeviceptr attnw_h = s_attnw[L] + (CUdeviceptr)((size_t)h*Spad*Spad*sizeof(float));
        /* d_aoh = grad into this head's context (slab h of dg_ctx) */
        pack_head(dh_aoh, dg_ctx, h*DH, QD, Spad);              /* [S,DH] */
        pack_head(dh_Vh,  s_v[L], kv*DH, KVD, Spad);            /* V_h (post-proj, pre-rope; V is unroped) */
        /* ctx_h = attnw_h @ V_h : d_attnw = d_aoh @ V_h^T ; d_V_h = attnw_h^T @ d_aoh */
        mm_ABt(dh_aoh, dh_Vh, dh_attnw, Spad, DH, Spad);       /* d_attnw [S,S] */
        mm_AtB(attnw_h, dh_aoh, dh_Vh, Spad, Spad, DH);        /* d_V_h [S,DH] (reuse dh_Vh as output) */
        /* fold d_V_h into the q-head-shaped repeat buffer slab h (summed later by repeat_kv_bwd) */
        CKX(cuMemcpyDtoD(dg_vrep + (CUdeviceptr)((size_t)h*Spad*DH*sizeof(float)),
                         dh_Vh, (size_t)Spad*DH*sizeof(float)), "stash dVrep");
        /* softmax bwd: d_scores = J^T d_attnw (probs are attnw_h) */
        softmax_bwd(attnw_h, dh_attnw, dh_scores, Spad, Spad); /* d_scores [S,S] */
        /* the forward scaled scores by 1/sqrt(DH) AFTER the matmul, BEFORE softmax. scale is linear,
         * so the grad of the pre-scale scores = d_scores * (1/sqrt(DH)). */
        scale_rt(dh_scores, d_unscale, Spad*Spad);
        /* scores = Qr_h @ Kr_h^T : d_Qr = d_scores @ Kr_h ; d_Kr = d_scores^T @ Qr_h */
        pack_head(dh_Qh, s_qr[L], h*DH,  QD,  Spad);           /* roped Q_h (saved) */
        pack_head(dh_Kh, s_kr[L], kv*DH, KVD, Spad);           /* roped K_h (saved) */
        {   CUdeviceptr dQr=A((size_t)Spad*DH), dKr=A((size_t)Spad*DH);
            mm_AB (dh_scores, dh_Kh, dQr, Spad, Spad, DH);     /* d_Qr_h [S,DH] */
            mm_AtB(dh_scores, dh_Qh, dKr, Spad, Spad, DH);     /* d_Kr_h [S,DH] */
            /* rope_bwd (transpose rotation) maps roped-grad -> pre-rope grad, in place */
            rope_bwd(dQr, d_cos, d_sin, Spad);
            rope_bwd(dKr, d_cos, d_sin, Spad);
            /* d_Q_h goes straight to its q-head slab in dg_q; d_K_h folds into the repeat buffer */
            scatter_head(dg_q, h*DH, dQr, QD, Spad);
            CKX(cuMemcpyDtoD(dg_krep + (CUdeviceptr)((size_t)h*Spad*DH*sizeof(float)),
                             dKr, (size_t)Spad*DH*sizeof(float)), "stash dKrep");
            CKX(cuMemFree(dQr),"free dQr"); CKX(cuMemFree(dKr),"free dKr");
        }
    }
    /* GQA fold: sum the n_rep q-head copies back into each kv head (broadcast adjoint). The repeat
     * buffers are grouped by kv (slab h = kv*group + r), exactly the layout repeat_kv_bwd expects.
     * dg_kfold/dg_vfold are [NKV,S,DH] (kv-major); scatter to the [S,KVD] proj-grad layout. */
    repeat_kv_bwd(dg_krep, dg_kfold, group, Spad*DH, NKV);
    repeat_kv_bwd(dg_vrep, dg_vfold, group, Spad*DH, NKV);
    for (int kv=0; kv<NKV; kv++) {
        CUdeviceptr ks = dg_kfold + (CUdeviceptr)((size_t)kv*Spad*DH*sizeof(float));
        CUdeviceptr vs = dg_vfold + (CUdeviceptr)((size_t)kv*Spad*DH*sizeof(float));
        scatter_head(dg_k, kv*DH, ks, KVD, Spad);
        scatter_head(dg_v, kv*DH, vs, KVD, Spad);
    }
    /* q/k/v projections: s_q = rms1 @ w_q^T, etc. Accumulate dW and sum dX into dg_rms (rms1 grad). */
    mm_AtB(dg_q, s_rms1[L], g_q[L], Spad, QD,  DM);    /* dW_q [QD,DM] */
    MASK(g_q[L], w_q[L], sc_q[L], QD, DM);
    mm_AtB(dg_k, s_rms1[L], g_k[L], Spad, KVD, DM);    /* dW_k [KVD,DM] */
    MASK(g_k[L], w_k[L], sc_k[L], KVD, DM);
    mm_AtB(dg_v, s_rms1[L], g_v[L], Spad, KVD, DM);    /* dW_v [KVD,DM] */
    MASK(g_v[L], w_v[L], sc_v[L], KVD, DM);
    mm_AB (dg_q, w_q[L], dg_rms,   Spad, QD,  DM);     /* d_rms1 (from q) [S,DM] */
    mm_AB (dg_k, w_k[L], dg_tmpDM, Spad, KVD, DM); vadd(dg_rms, dg_tmpDM, dg_rms, Spad*DM);  /* + k */
    mm_AB (dg_v, w_v[L], dg_tmpDM, Spad, KVD, DM); vadd(dg_rms, dg_tmpDM, dg_rms, Spad*DM);  /* + v */
    /* rmsnorm1: s_rms1 = rmsnorm(resid_in) * w_inln. dW_inln (host); d_resid_in via dx; add residual. */
    rmsnorm_wgrad_host(s_resid_in[L], dg_rms, g_inln[L], Spad, DM);
    rms_bwd_dx(s_resid_in[L], w_inln[L], dg_rms, dg_tmpDM, Spad, DM);  /* d_resid_in (attn branch) */
    /* resid_in feeds BOTH the attn rmsnorm AND the residual into resid_mid: total = dg_x + dg_tmpDM. */
    vadd(dg_x, dg_tmpDM, dg_x, Spad*DM);   /* dg_x = grad of this layer's INPUT (s_resid_in[L]) */
}

/* full backward: CE-grad root -> tied lm_head -> final rmsnorm -> layers reversed. Leaves every
 * g_*[L], g_normf, g_embed populated. Tied embedding: the lm_head weight grad AND the input-embedding
 * scatter both accumulate into g_embed (the latter on the host, at the very end). */
static void backward(const int* ids, int T) {
    g_T=T;
    zero_all_grads();
    /* shifted targets as f32 (row t -> ids[t+1]); rows >= T-1 are masked to 0 grad after the kernel. */
    { float* tg=(float*)calloc((size_t)Spad,sizeof(float));
      for (int t=0;t<T-1;t++) tg[t]=(float)ids[t+1];
      CKX(cuMemcpyHtoD(d_targets,tg,(size_t)Spad*sizeof(float)),"h2d targets"); free(tg); }
    ce_softmax_grad(d_logits, d_targets, dg_logits, Spad, NVpad);  /* softmax - onehot, ALL rows */
    /* zero rows >= T-1 (only t in [0,T-1) are loss positions) + scale live rows by 1/(T-1). */
    { int live=T-1; double invn=1.0/(double)live; float s=(float)invn;
      CUdeviceptr sbuf=A(1); CKX(cuMemcpyHtoD(sbuf,&s,sizeof(float)),"h2d lossscale");
      if (live < Spad)
          CKX(cuMemsetD8(dg_logits + (CUdeviceptr)((size_t)live*NVpad*sizeof(float)), 0,
                         (size_t)(Spad-live)*NVpad*sizeof(float)), "zero pad+last rows");
      scale_rt(dg_logits, sbuf, live*NVpad);   /* scale the live rows by 1/(T-1) */
      CKX(cuMemFree(sbuf),"free lossscale");
    }
    /* tied lm_head: logits = rmsf @ embed^T. d_rmsf = dlogits @ embed ; g_embed += dlogits^T @ rmsf. */
    mm_AB (dg_logits, w_embed_pad, dg_rms, Spad, NVpad, DM);   /* d_rmsf [S,DM] */
    mm_AtB(dg_logits, s_rmsf, g_embed, Spad, NVpad, DM);       /* dW_embed (lm_head leg) [NVpad,DM] */
    /* final rmsnorm: rmsf = rmsnorm(resid_out[NL-1]) * w_normf. dW_normf (host); dx -> dg_x. */
    rmsnorm_wgrad_host(s_resid_out[NL-1], dg_rms, g_normf, Spad, DM);
    rms_bwd_dx(s_resid_out[NL-1], w_normf, dg_rms, dg_x, Spad, DM);  /* dg_x = grad of resid_out[NL-1] */
    /* layers in reverse (dg_x carries the grad of layer L's output into backward_layer) */
    int trace = (getenv("HX_BWD_TRACE") && atoi(getenv("HX_BWD_TRACE"))!=0);
    for (int L=NL-1; L>=0; L--) {
        backward_layer(L, dg_x);
        if (trace) {
            float* h=(float*)malloc((size_t)Spad*DM*sizeof(float));
            CKX(cuMemcpyDtoH(h,dg_x,(size_t)Spad*DM*sizeof(float)),"trace dgx");
            double nn=0; for (size_t i=0;i<(size_t)Spad*DM;i++) nn+=(double)h[i]*h[i];
            fprintf(stderr,"  [trace] after layer %d: |dg_x|=%.6g\n", L, sqrt(nn)); free(h);
        }
    }
    /* tied-embedding INPUT scatter: x0[s,:] = embed[ids[s],:], so g_embed[ids[s],:] += dg_x[s,:]
     * (dg_x now holds the grad of s_resid_in[0] = the input embeddings). Host scatter-add. */
    { float* dxin=(float*)malloc((size_t)Spad*DM*sizeof(float));
      float* ge  =(float*)malloc((size_t)NVpad*DM*sizeof(float));
      CKX(cuMemcpyDtoH(dxin, dg_x, (size_t)Spad*DM*sizeof(float)),"d2h dxin");
      CKX(cuMemcpyDtoH(ge,   g_embed, (size_t)NVpad*DM*sizeof(float)),"d2h g_embed");
      for (int s=0;s<T;s++) { int id=ids[s]; for (int c=0;c<DM;c++) ge[(size_t)id*DM+c]+=dxin[(size_t)s*DM+c]; }
      CKX(cuMemcpyHtoD(g_embed, ge, (size_t)NVpad*DM*sizeof(float)),"h2d g_embed");
      free(dxin); free(ge);
    }
}

/* ===================== device init (load PTX, bind the 8 kernels) ===================== */
static int device_init(const char* ptx_path) {
    FILE* pf=fopen(ptx_path,"rb"); if (!pf) { fprintf(stderr, "open ptx '%s'\n", ptx_path); return 2; }
    fseek(pf,0,SEEK_END); long psz=ftell(pf); fseek(pf,0,SEEK_SET);
    g_ptx=(char*)malloc(psz+1); if (fread(g_ptx,1,psz,pf)!=(size_t)psz) { fclose(pf); return 2; } g_ptx[psz]=0; fclose(pf);
    CK(cuInit(0), "init");
    CUdevice dev; CK(cuDeviceGet(&dev,0), "dev");
    char name[256]={0}; cuDeviceGetName(name,256,dev);
    CK(cuCtxCreate(&ctx,0,dev), "ctx");
    CK(cuModuleLoadData(&g_mod, g_ptx), "load ptx");
    CK(cuModuleGetFunction(&f_mm,        g_mod, "tiled_matmul"),        "tiled_matmul");
    CK(cuModuleGetFunction(&f_abt,       g_mod, "tiled_matmul_abt"),    "tiled_matmul_abt");
    CK(cuModuleGetFunction(&f_sm_causal, g_mod, "gpu_softmax_causal"),  "gpu_softmax_causal");
    CK(cuModuleGetFunction(&f_rms,       g_mod, "gpu_rmsnorm_fwd_eps"), "gpu_rmsnorm_fwd_eps");
    CK(cuModuleGetFunction(&f_rope,      g_mod, "gpu_rope_rot"),        "gpu_rope_rot");
    CK(cuModuleGetFunction(&f_silu,      g_mod, "gpu_silu_mul"),        "gpu_silu_mul");
    CK(cuModuleGetFunction(&f_scale,     g_mod, "gpu_scale_rt"),        "gpu_scale_rt");
    CK(cuModuleGetFunction(&f_add,       g_mod, "vector_add"),          "vector_add");
    /* backward kernels (present in the extended combined PTX; bound unconditionally so --fdcheck
     * and the train loop can use them. If an old fwd-only PTX is passed these GetFunction calls
     * fail loudly -- the extended mint is required for the backward). */
    CK(cuModuleGetFunction(&f_atb,   g_mod, "tiled_matmul_atb"),   "tiled_matmul_atb");
    CK(cuModuleGetFunction(&f_rmsb,  g_mod, "gpu_rmsnorm_bwd_dx"), "gpu_rmsnorm_bwd_dx");
    CK(cuModuleGetFunction(&f_ropeb, g_mod, "gpu_rope_bwd"),       "gpu_rope_bwd");
    CK(cuModuleGetFunction(&f_silub, g_mod, "gpu_silu_mul_bwd"),   "gpu_silu_mul_bwd");
    CK(cuModuleGetFunction(&f_rkvb,  g_mod, "gpu_repeat_kv_bwd"),  "gpu_repeat_kv_bwd");
    CK(cuModuleGetFunction(&f_ceg,   g_mod, "gpu_ce_softmax_grad"),"gpu_ce_softmax_grad");
    CK(cuModuleGetFunction(&f_smb,   g_mod, "gpu_softmax_backward"),"gpu_softmax_backward");
    CK(cuModuleGetFunction(&f_adam,  g_mod, "gpu_adam"),           "gpu_adam");
    /* v1.9 P5 STE kernels (bound only when QAT, so a QAT-off run accepts either the 17- or 20-kernel
     * PTX; with QAT the extended 20-kernel mint is required and these GetFunction calls fail loudly
     * if the 3 STE kernels are absent). @kernel sources -> no kovc.hx edit. */
    if (QAT) {
        CK(cuModuleGetFunction(&f_ternarize,  g_mod, "ternarize_dequant"), "ternarize_dequant");
        CK(cuModuleGetFunction(&f_rowabsmean, g_mod, "row_abs_mean"),      "row_abs_mean");
        CK(cuModuleGetFunction(&f_stemask,    g_mod, "ste_mask"),          "ste_mask");
    }
    printf("[gpu] %s; PTX %ld B, 8 fwd + 8 bwd%s kernels bound\n", name, psz, QAT?" + 3 STE-QAT":"");
    return 0;
}

/* ===================== Adam Param list (capstone pattern) ===================== */
static void addp(CUdeviceptr w, CUdeviceptr g, int n) {
    Param* p=&params[nparams++]; p->w=w; p->g=g; p->n=n; p->m=A(n); p->v=A(n);
    CKX(cuMemsetD8(p->m,0,(size_t)n*sizeof(float)),"zero m"); CKX(cuMemsetD8(p->v,0,(size_t)n*sizeof(float)),"zero v");
}
static void build_params(void) {
    for (int L=0;L<NL;L++) {
        addp(w_inln[L],g_inln[L],DM);   addp(w_q[L],g_q[L],QD*DM);   addp(w_k[L],g_k[L],KVD*DM);
        addp(w_v[L],g_v[L],KVD*DM);     addp(w_o[L],g_o[L],DM*QD);   addp(w_postln[L],g_postln[L],DM);
        addp(w_gate[L],g_gate[L],DFF*DM); addp(w_up[L],g_up[L],DFF*DM); addp(w_down[L],g_down[L],DM*DFF);
    }
    addp(w_normf,g_normf,DM);
    addp(w_embed_pad,g_embed,NVpad*DM);   /* tied: the single embedding param (lm_head + input share it) */
}
static void adam_step(int t) {
    float bc1=1.0f/(1.0f-powf(0.9f,(float)t)), bc2=1.0f/(1.0f-powf(0.999f,(float)t));
    CKX(cuMemcpyHtoD(d_bc1,&bc1,sizeof(float)),"bc1"); CKX(cuMemcpyHtoD(d_bc2,&bc2,sizeof(float)),"bc2");
    for (int i=0;i<nparams;i++){ Param* p=&params[i]; adam_step1(p->w,p->g,p->m,p->v,d_bc1,d_bc2,p->n); }
}

static int read_ids_file(const char* path, int* ids, int maxn) {
    FILE* f=fopen(path,"r"); if (!f) { fprintf(stderr, "open ids '%s': %s\n", path, strerror(errno)); return -1; }
    int n=0,v; while (n<maxn && fscanf(f,"%d",&v)==1) ids[n++]=v;
    fclose(f); return n;
}
static int argmax_row(const float* v, int n) { int a=0; for (int i=1;i<n;i++) if (v[i]>v[a]) a=i; return a; }

/* shifted cross-entropy loss over the prompt: for positions t in [0,T-1), target = ids[t+1];
 * CE_t = -log softmax(logits_t)[ids[t+1]]; loss = mean_t CE_t. Host reduction over the saved
 * d_logits rows (training glue; the forward arithmetic stayed in kernels). */
static double ce_loss(const int* ids, int T) {
    if (T < 2) return 0.0/0.0;
    float* row=(float*)malloc((size_t)NV*sizeof(float));
    double tot=0.0; int cnt=0;
    for (int t=0;t<T-1;t++) {
        CKX(cuMemcpyDtoH(row, d_logits + (CUdeviceptr)((size_t)t*NVpad*sizeof(float)), (size_t)NV*sizeof(float)), "d2h logit row");
        double mx=-1e30; for (int i=0;i<NV;i++) if ((double)row[i]>mx) mx=(double)row[i];
        double se=0.0; for (int i=0;i<NV;i++) se+=exp((double)row[i]-mx);
        int tgt=ids[t+1];
        double lp=(double)row[tgt]-mx-log(se);
        tot += -lp; cnt++;
    }
    free(row);
    return tot/(double)cnt;
}

/* run a full forward and return the shifted-CE loss (the trainer's own loss). Reused by --fdcheck:
 * perturb a resident weight scalar -> compute_loss -> restore. A throwaway last-row buffer is fine. */
static float* g_logits_scratch=NULL;
static double compute_loss(const int* ids, int T) {
    if (!g_logits_scratch) g_logits_scratch=(float*)malloc((size_t)NV*sizeof(float));
    forward(ids, T, g_logits_scratch);
    return ce_loss(ids, T);
}

/* ===================== PERPLEXITY (v1.9 step5a): chunked held-out corpus reduction =====================
 * The forward + the shifted-CE already exist (ce_loss = the MEAN over one sequence's T-1 positions).
 * Perplexity over a held-out CORPUS is the same shifted-CE but (a) chunked into non-overlapping
 * windows of <= Spad tokens, (b) SUMMED (not averaged) over every predicted position, then
 * ppl = exp(sum_CE / total_predicted). ce_sum_window is ce_loss without the 1/(T-1) divide: it
 * returns the SUM of -log p(target) over positions [0,T-1) of ONE window AND the count (T-1).
 * Works in BOTH fp mode (QAT off) AND ternary mode (QAT on) for free: it calls the same forward(),
 * which routes the GEMMs through EW() -> the ternarized Wt when HX_TERNARY_QAT=1, the fp W otherwise.
 * (Same host log-softmax reduction as ce_loss; the forward arithmetic stays in the PTX kernels.) */
static double ce_sum_window(const int* ids, int T, long* out_npred) {
    /* sum_{t in [0,T-1)} -log softmax(logits_t)[ids[t+1]] ; predicted-token count = T-1. */
    if (out_npred) *out_npred = 0;
    if (T < 2) return 0.0;
    float* row=(float*)malloc((size_t)NV*sizeof(float));
    double tot=0.0; long cnt=0;
    for (int t=0;t<T-1;t++) {
        CKX(cuMemcpyDtoH(row, d_logits + (CUdeviceptr)((size_t)t*NVpad*sizeof(float)), (size_t)NV*sizeof(float)), "d2h ppl logit row");
        double mx=-1e30; for (int i=0;i<NV;i++) if ((double)row[i]>mx) mx=(double)row[i];
        double se=0.0; for (int i=0;i<NV;i++) se+=exp((double)row[i]-mx);
        int tgt=ids[t+1];
        double lp=(double)row[tgt]-mx-log(se);
        tot += -lp; cnt++;
    }
    free(row);
    if (out_npred) *out_npred = cnt;
    return tot;
}

/* Held-out perplexity over a tokenized corpus stream (corpus[0..N)). Non-overlapping windows of
 * `win` tokens (the last partial window is evaluated as-is if it has >=2 tokens). Returns ppl and
 * fills the running totals. `win` MUST be <= Spad (the allocated activation depth). The first token
 * of each window after the first is NOT carried over as context (non-overlapping == each window is a
 * fresh sequence with its own causal mask from position 0); this is the standard "split into chunks"
 * estimator -- a sliding window would lower-bound it but costs N forwards instead of N/win. */
static double corpus_perplexity(const int* corpus, int N, int win, double* out_sumCE, long* out_npred) {
    if (win > Spad) win = Spad;
    if (!g_logits_scratch) g_logits_scratch=(float*)malloc((size_t)NV*sizeof(float));  /* fwd last-row sink */
    double sumCE=0.0; long npred=0; int nwin=0;
    for (int off=0; off<N; off+=win) {
        int T = (N-off < win) ? (N-off) : win;     /* last window may be short */
        if (T < 2) break;                            /* a 1-token tail predicts nothing */
        long np=0;
        forward(&corpus[off], T, g_logits_scratch);  /* fills d_logits[0..T) ; last row -> scratch */
        double s = ce_sum_window(&corpus[off], T, &np);
        sumCE += s; npred += np; nwin++;
    }
    if (out_sumCE) *out_sumCE = sumCE;
    if (out_npred) *out_npred = npred;
    double avg = (npred>0) ? sumCE/(double)npred : 0.0/0.0;
    double ppl = exp(avg);
    printf("[ppl] windows=%d win=%d total_predicted_tokens=%ld  sum_CE=%.4f  avg_CE(nats)=%.6f  PERPLEXITY=%.4f\n",
           nwin, win, npred, sumCE, avg, ppl);
    return ppl;
}

/* ===================== MULTI-SEQUENCE corpus training (v1.9 step5b QAT fine-tune) =====================
 * Reads a corpus stream + a per-sequence length list (lens[0..nseq)), and runs E epochs, each epoch
 * iterating the sequences in order: forward -> backward -> Adam, ONE optimizer step per sequence (a
 * sequence == a minibatch of 1). Correctness-first (the existing forward/backward already sync every
 * launch). Each sequence's T must be <= Spad. The Adam step counter `t` increments globally across all
 * (epoch,sequence) so the bias correction is monotonic (capstone convention). Works fp OR ternary
 * (the EW()/MASK() QAT path is exercised when HX_TERNARY_QAT=1). Reports the mean per-sequence loss
 * at the start and end of training (and per-epoch) so the smoke can confirm the loss trends DOWN. */
static double corpus_mean_loss(const int* corpus, const int* lens, int nseq) {
    double tot=0.0; int cnt=0; int off=0;
    for (int s=0;s<nseq;s++) {
        int T=lens[s];
        if (T>=2) { double l=compute_loss(&corpus[off], T); tot+=l; cnt++; }
        off+=T;
    }
    return (cnt>0) ? tot/(double)cnt : 0.0/0.0;
}
static int run_corpus_train(const int* corpus, const int* lens, int nseq, int epochs) {
    double l0 = corpus_mean_loss(corpus, lens, nseq);
    printf("[corpus-train] %d sequences, %d epochs ; start mean per-seq shifted-CE = %.6f (lr=1e-3 baked)\n",
           nseq, epochs, l0);
    int t=0;            /* global Adam step counter (monotone bias correction) */
    double lepoch=l0;
    for (int e=1;e<=epochs;e++) {
        int off=0; double esum=0.0; int ecnt=0;
        for (int s=0;s<nseq;s++) {
            int T=lens[s];
            if (T>=2) {
                double l=compute_loss(&corpus[off], T);   /* forward (fills d_logits) + host CE */
                backward(&corpus[off], T);                /* grads from the SAME forward's saved acts */
                t++; adam_step(t);
                esum+=l; ecnt++;
            }
            off+=T;
        }
        lepoch = (ecnt>0) ? esum/(double)ecnt : lepoch;
        printf("  epoch %2d mean per-seq loss %.6f\n", e, lepoch);
    }
    double lf = corpus_mean_loss(corpus, lens, nseq);
    printf("[corpus-train] %d epochs x %d seqs: start %.6f -> final %.6f  (%s)\n", epochs, nseq, l0, lf,
           lf < l0 ? "LLAMA_TRAIN_CORPUS_LOSS_DECREASED" : "LLAMA_TRAIN_CORPUS_LOSS_DID_NOT_DECREASE");
    return (lf < l0) ? 0 : 1;
}

/* ===================== COMBINED train-then-eval in ONE process (v1.9 step5c) =====================
 * THE CRITICAL FIX for an honest QAT conversion measurement: train the corpus AND measure the
 * held-out perplexity on the SAME (in-place fine-tuned) resident weights, in one process. The plain
 * --ppl mode returns before --train-corpus runs (see main), so a single invocation could only ever
 * report the PRE-fine-tune ternary ppl (~1.2e9) -- meaningless as a "converted model" number. Here:
 *   - epoch 0  : measure the held-out ppl on the starting weights (the ~1.2e9 raw-ternary number
 *                when QAT is on -> the trajectory's t=0 anchor).
 *   - every HX_PPL_EVERY epochs (default 5): forward+backward+Adam over all sequences for those
 *     epochs, THEN re-measure the held-out ppl on the now-updated resident latent weights (routed
 *     through EW() -> ternarized Wt when QAT). This is the RECOVERY TRAJECTORY.
 *   - final     : the held-out ppl AFTER the last epoch == the converted-ternary number to compare
 *                 to the fp baseline.
 * Adam step counter t is global+monotone across epochs (bias correction), exactly run_corpus_train.
 * The ppl evals share the saved-intermediate buffers (main sized S = max(ppl_win_pad, max_seq_pad)),
 * and forward() is re-run every training step, so a mid-train ppl eval leaves no stale state.        */
static int run_corpus_train_then_ppl(const int* corpus, const int* lens, int nseq, int epochs,
                                     const int* held, int heldN, int ppl_win) {
    int every = 5;
    if (getenv("HX_PPL_EVERY") && atoi(getenv("HX_PPL_EVERY"))>0) every = atoi(getenv("HX_PPL_EVERY"));
    double l0 = corpus_mean_loss(corpus, lens, nseq);
    printf("[ft] COMBINED fine-tune-then-eval: %d seqs, %d epochs, ppl-checkpoint every %d epoch(s)\n",
           nseq, epochs, every);
    printf("[ft] start mean per-seq shifted-CE = %.6f (lr=1e-3 baked, QAT=%d)\n", l0, QAT);
    double sc; long np;
    printf("[ft] === held-out ppl @ epoch 0 (pre-fine-tune) ===\n");
    double ppl0 = corpus_perplexity(held, heldN, ppl_win, &sc, &np);
    printf("[ft-traj] epoch 0  held_out_ppl %.6g  (train_mean_loss %.6f)\n", ppl0, l0);

    int t=0; double lepoch=l0; double ppl_last=ppl0;
    for (int e=1;e<=epochs;e++) {
        int off=0; double esum=0.0; int ecnt=0;
        for (int s=0;s<nseq;s++) {
            int T=lens[s];
            if (T>=2) {
                double l=compute_loss(&corpus[off], T);   /* forward (fills d_logits) + host CE */
                backward(&corpus[off], T);                /* grads from the SAME forward's saved acts */
                t++; adam_step(t);
                esum+=l; ecnt++;
            }
            off+=T;
        }
        lepoch = (ecnt>0) ? esum/(double)ecnt : lepoch;
        int checkpoint = (e % every == 0) || (e == epochs);
        if (checkpoint) {
            printf("[ft] === held-out ppl @ epoch %d (train_mean_loss %.6f) ===\n", e, lepoch);
            ppl_last = corpus_perplexity(held, heldN, ppl_win, &sc, &np);
            printf("[ft-traj] epoch %d  held_out_ppl %.6g  (train_mean_loss %.6f)\n", e, ppl_last, lepoch);
        } else {
            printf("  epoch %2d mean per-seq loss %.6f\n", e, lepoch);
        }
    }
    printf("[ft] DONE: %d epochs x %d seqs ; train_mean_loss %.6f -> %.6f ; held_out_ppl %.6g -> %.6g\n",
           epochs, nseq, l0, lepoch, ppl0, ppl_last);
    printf("[ft] CONVERTED_TERNARY_HELDOUT_PPL = %.6f  (QAT=%d, fp_baseline_ref=13.1243)\n", ppl_last, QAT);
    printf("LLAMA_TRAIN_FT_PPL_DONE\n");
    return isfinite(ppl_last) ? 0 : 1;
}

/* ===================== finite-difference gradient check (the acceptance gate) =====================
 * For a chosen resident-weight scalar, central difference (L(w+eps)-L(w-eps))/(2eps) vs the analytic
 * grad the backward produced for that scalar. eps in fp32; rel-err < ~1e-2 is a PASS (fp32-noisy).
 * Each probe: read the analytic grad cell, perturb the SAME device cell +/-eps, recompute the loss
 * (full forward), restore. The tied embedding uses w_embed_pad both as lm_head AND (post-embed_gather
 * fix) as the input-embedding source, so a single device perturbation captures BOTH paths. */
typedef struct { const char* name; CUdeviceptr wbuf; CUdeviceptr gbuf; long n; long idx; } Probe;

static double read_cell(CUdeviceptr buf, long idx) {
    float v; CKX(cuMemcpyDtoH(&v, buf+(CUdeviceptr)((size_t)idx*sizeof(float)), sizeof(float)), "d2h cell"); return (double)v;
}
static void write_cell(CUdeviceptr buf, long idx, float v) {
    CKX(cuMemcpyHtoD(buf+(CUdeviceptr)((size_t)idx*sizeof(float)), &v, sizeof(float)), "h2d cell");
}
/* pick the grad cell with the LARGEST |analytic| within a tensor (scan a strided sample). The
 * finite-difference of an fp32 loss has an absolute noise floor ~|L|*2^-23 (~4e-7 here); a probe is
 * only RESOLVABLE when |grad|*2*eps exceeds that floor, so we deliberately verify each tensor family
 * at its most-resolvable cell. This is honest -- a near-zero gradient cannot be confirmed by a
 * difference that is itself below the fp32 rounding of the loss. */
static long max_abs_grad_cell(CUdeviceptr gbuf, long n) {
    long stride = n>20000 ? n/20000 : 1;   /* sample up to ~20k cells (D2H of a few KB) */
    long best=0; double bestv=-1.0;
    for (long i=0;i<n;i+=stride) { double v=fabs(read_cell(gbuf,i)); if (v>bestv){bestv=v;best=i;} }
    return best;
}

static int run_fdcheck(const int* ids, int T) {
    /* MULTI-EPS central finite-difference (standard grad-check practice). A single eps cannot verify
     * both weight classes here, as the HX_FD_SWEEP diagnostic shows:
     *   - PROJECTION weights (value ~0.05, |grad| ~1e-4..1e-3): the central-diff signal 2*eps*grad
     *     must clear the fp32 loss-noise floor (~|L|*2^-23 ~ 4e-7 on the 3.87 loss), so they need a
     *     LARGE eps (>=1e-2); at eps=1e-3 numeric is pure noise.
     *   - RMSNORM weights (value ~1.0..1.4, |grad| up to ~3, large curvature): they need a SMALL eps
     *     (~1e-3); at eps=2e-2 the O(eps^2) truncation error is ~10-60%.
     * So we try eps in {1e-3, 4e-3, 1.5e-2, 5e-2} and accept the probe if rel-err < 5e-2 at the BEST
     * eps (the one in that weight's valid window between noise floor and truncation). 5% is the same
     * fp32-noisy tolerance the GPT-2 capstone finite-diff uses. */
    double epss[]={1e-3,4e-3,1.5e-2,5e-2}; int neps=(int)(sizeof(epss)/sizeof(epss[0]));
    printf("\n[fdcheck] MULTI-EPS central finite-difference (eps in {1e-3,4e-3,1.5e-2,5e-2}, accept best rel<5e-2, T=%d, Spad=%d)\n", T, Spad);
    double base = compute_loss(ids, T);
    backward(ids, T);
    printf("[fdcheck] base shifted-CE loss = %.6f ; probing the max-|grad| cell of each tensor family\n", base);

    int Lt=NL-1, Lm=NL/2, L0=0;   /* top, middle, bottom layers -> exercises cross-layer propagation */
    long qd_dm=(long)QD*DM, kvd_dm=(long)KVD*DM, dm_qd=(long)DM*QD, dff_dm=(long)DFF*DM, dm_dff=(long)DM*DFF;
    Probe probes[] = {
        /* output layers */
        { "w_normf",       w_normf,      g_normf,      DM,        0 },
        { "w_embed(tied)", w_embed_pad,  g_embed,      (long)NVpad*DM, 0 },  /* tied: lm_head + input-scatter */
        /* TOP layer: every projection + both norms (shortest path, isolates per-layer logic) */
        { "w_down[T]",     w_down[Lt],   g_down[Lt],   dm_dff,    0 },
        { "w_up[T]",       w_up[Lt],     g_up[Lt],     dff_dm,    0 },
        { "w_gate[T]",     w_gate[Lt],   g_gate[Lt],   dff_dm,    0 },
        { "w_postln[T]",   w_postln[Lt], g_postln[Lt], DM,        0 },
        { "w_o[T]",        w_o[Lt],      g_o[Lt],      dm_qd,     0 },
        { "w_v[T]",        w_v[Lt],      g_v[Lt],      kvd_dm,    0 },
        { "w_k[T]",        w_k[Lt],      g_k[Lt],      kvd_dm,    0 },
        { "w_q[T]",        w_q[Lt],      g_q[Lt],      qd_dm,     0 },
        { "w_inln[T]",     w_inln[Lt],   g_inln[Lt],   DM,        0 },
        /* MIDDLE layer: one attention + one MLP weight (deep cross-layer grad must reach here) */
        { "w_q[M]",        w_q[Lm],      g_q[Lm],      qd_dm,     0 },
        { "w_down[M]",     w_down[Lm],   g_down[Lm],   dm_dff,    0 },
        /* BOTTOM layer: attention + MLP (longest path: grad threads all 30 layers) */
        { "w_q[0]",        w_q[L0],      g_q[L0],      qd_dm,     0 },
        { "w_gate[0]",     w_gate[L0],   g_gate[L0],   dff_dm,    0 },
    };
    int nprobe=(int)(sizeof(probes)/sizeof(probes[0]));
    int sweep = (getenv("HX_FD_SWEEP") && atoi(getenv("HX_FD_SWEEP"))!=0);
    if (sweep) {   /* diagnostic kept: eps sweep exposing the fp32 loss-noise floor AND the
                    * large-gradient TRUNCATION regime (w_postln[T] has the biggest grad ~3). */
        long io=max_abs_grad_cell(g_o[Lt],dm_qd), ip=max_abs_grad_cell(g_postln[Lt],DM);
        struct { const char* nm; CUdeviceptr wb,gb; long ix; } sp[2]={{"w_o[T]",w_o[Lt],g_o[Lt],io},{"w_postln[T]",w_postln[Lt],g_postln[Lt],ip}};
        double epss[5]={1e-1,3e-2,1e-2,3e-3,1e-3};
        for (int k=0;k<2;k++){ double an=read_cell(sp[k].gb,sp[k].ix); float w0=(float)read_cell(sp[k].wb,sp[k].ix);
            fprintf(stderr,"  [sweep] %s analytic=%.6g (w0=%.5g)\n", sp[k].nm, an, (double)w0);
            for (int e=0;e<5;e++){ double ep=epss[e];
                write_cell(sp[k].wb,sp[k].ix,(float)(w0+ep)); double Lp=compute_loss(ids,T);
                write_cell(sp[k].wb,sp[k].ix,(float)(w0-ep)); double Lm=compute_loss(ids,T);
                write_cell(sp[k].wb,sp[k].ix,w0);
                fprintf(stderr,"     eps=%.0e  Lp-Lm=%.3e  numeric=%.6g  rel=%.3e\n", ep, Lp-Lm, (Lp-Lm)/(2*ep), fabs(an-(Lp-Lm)/(2*ep))/(fabs(an)+1e-9));
            }
        }
    }
    printf("  %-16s | %14s | %14s | %10s | %7s | %s\n", "weight", "analytic", "numeric(best)", "rel-err", "best-eps", "verdict");
    printf("  %-16s-+-%14s-+-%14s-+-%10s-+-%7s-+-%s\n", "----------------","--------------","--------------","----------","-------","-------");
    int nfail=0;
    for (int i=0;i<nprobe;i++) {
        Probe* p=&probes[i];
        p->idx = max_abs_grad_cell(p->gbuf, p->n);       /* verify where the gradient is resolvable */
        double analytic = read_cell(p->gbuf, p->idx);
        float  w0f=(float)read_cell(p->wbuf, p->idx);
        double best_rel=1e30, best_num=0.0, best_eps=0.0;
        for (int e=0;e<neps;e++) {
            double eps=epss[e];
            write_cell(p->wbuf, p->idx, (float)(w0f+eps)); double Lp=compute_loss(ids,T);
            write_cell(p->wbuf, p->idx, (float)(w0f-eps)); double Lm2=compute_loss(ids,T);
            double numeric=(Lp-Lm2)/(2.0*eps);
            double rel=fabs(analytic-numeric)/(fabs(analytic)+fabs(numeric)+1e-9);
            if (rel<best_rel){ best_rel=rel; best_num=numeric; best_eps=eps; }
        }
        write_cell(p->wbuf, p->idx, w0f);                 /* restore */
        int ok = (best_rel < 5e-2) && isfinite(analytic) && isfinite(best_num);
        if (!ok) nfail++;
        printf("  %-16s | %14.6g | %14.6g | %10.3e | %7.0e | %s\n", p->name, analytic, best_num, best_rel, best_eps, ok?"PASS":"FAIL");
    }
    printf("[fdcheck] %d/%d probes PASS (families: final-norm, tied-embed, all 7 projections + both layer-norms @ top, attn+MLP @ mid & bottom)\n", nprobe-nfail, nprobe);
    printf("%s\n", nfail==0 ? "LLAMA_TRAIN_FDCHECK_PASS" : "LLAMA_TRAIN_FDCHECK_FAIL");
    return nfail==0 ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <combined.ptx> <smollm2-135m.weights> <ids.txt> [oracle_logits.bin]\n"
                        "         [--fdcheck] [--train N] [--ppl <corpus_ids> [win]]\n"
                        "         [--train-corpus <corpus_ids> <lens.txt> [epochs]]\n", argv[0]);
        return 2;
    }
    const char* ptx=argv[1]; const char* wts=argv[2]; const char* idsf=argv[3];
    /* v1.9 P5: opt-in STE ternary QAT (default OFF). Must be set before device_init/upload_all_weights
     * so the 3 STE kernels are bound + the ternarized/scale buffers are allocated. */
    if (getenv("HX_TERNARY_QAT") && atoi(getenv("HX_TERNARY_QAT"))!=0) QAT=1;
    /* flag scan: --fdcheck (or HX_FDCHECK=1) the gradient check; --train N N Adam steps on argv[3];
     * --ppl <file> [win] held-out perplexity over a tokenized corpus stream (v1.9 step5a);
     * --train-corpus <file> <lens> [E] multi-sequence E-epoch training (v1.9 step5b).
     * the optional positional oracle_logits.bin is the first non-flag arg after ids (argv[4]). */
    int do_fd = (getenv("HX_FDCHECK") && atoi(getenv("HX_FDCHECK"))!=0);
    int do_train = 0; int train_K = 0;
    int do_ppl = 0; const char* pplf=NULL; int ppl_win=0;
    int do_corpus = 0; const char* corpusf=NULL; const char* lensf=NULL; int corpus_E=0;
    const char* oraclef = NULL;
    for (int i=4;i<argc;i++) {
        if (!strcmp(argv[i],"--fdcheck")) do_fd=1;
        else if (!strcmp(argv[i],"--train")) { do_train=1; if (i+1<argc && argv[i+1][0]!='-') train_K=atoi(argv[++i]); if (train_K<=0) train_K=10; }
        else if (!strcmp(argv[i],"--ppl")) { do_ppl=1;
            if (i+1<argc && argv[i+1][0]!='-') pplf=argv[++i];
            if (i+1<argc && argv[i+1][0]!='-') ppl_win=atoi(argv[++i]); }
        else if (!strcmp(argv[i],"--train-corpus")) { do_corpus=1;
            if (i+1<argc && argv[i+1][0]!='-') corpusf=argv[++i];
            if (i+1<argc && argv[i+1][0]!='-') lensf=argv[++i];
            if (i+1<argc && argv[i+1][0]!='-') corpus_E=atoi(argv[++i]); if (corpus_E<=0) corpus_E=3; }
        else if (!oraclef) oraclef=argv[i];
    }
    if (do_ppl && !pplf)    { fprintf(stderr, "--ppl needs a corpus ids file\n"); return 2; }
    if (do_corpus && (!corpusf || !lensf)) { fprintf(stderr, "--train-corpus needs <corpus_ids> <lens.txt>\n"); return 2; }

    if (load_weights(wts)) return 2;
    if (device_init(ptx)) return 2;

    int ids[4096];
    int T=read_ids_file(idsf, ids, 4096);
    if (T<=0) { fprintf(stderr, "no ids in %s\n", idsf); return 2; }
    int S=((T+63)/64)*64;
    printf("[run] T=%d ids:", T); for (int i=0;i<T;i++) printf(" %d", ids[i]); printf("  (Spad=%d)\n", S);

    /* ---- corpus modes read their stream NOW so alloc_saved is sized to the largest window/sequence.
     * Corpus capacity is generous (1M tokens); the ids[] array above stays the 4096 primary. ---- */
    int *corpus=NULL, *lens=NULL, corpusN=0, nseq=0;
    int *held=NULL, heldN=0;                               /* combined mode: SEPARATE held-out stream */
    int do_combined = (do_corpus && do_ppl);              /* v1.9 step5c: train-then-eval one process */
    if (do_ppl || do_corpus) {
        /* primary corpus stream = the train corpus when training, else the ppl stream. */
        const char* cf = do_corpus ? corpusf : pplf;
        int cap = 1<<20;                                  /* 1,048,576 tokens (laptop-sized) */
        corpus = (int*)malloc((size_t)cap*sizeof(int));
        corpusN = read_ids_file(cf, corpus, cap);
        if (corpusN<=0) { fprintf(stderr, "no ids in corpus %s\n", cf); return 2; }
        if (do_ppl) {
            if (ppl_win<=0) ppl_win = 256;                /* default window (>= the 30-pos NL min) */
            int wpad = ((ppl_win+63)/64)*64;              /* the window padded to %64 */
            if (wpad > S) S = wpad;                       /* size activations to hold a full window */
            if (!do_combined)
                printf("[ppl] corpus '%s': %d tokens, window=%d (Spad will be %d)\n", cf, corpusN, ppl_win, S);
        }
        if (do_corpus) {
            int lcap = 1<<16; lens=(int*)malloc((size_t)lcap*sizeof(int));
            nseq = read_ids_file(lensf, lens, lcap);
            if (nseq<=0) { fprintf(stderr, "no lengths in %s\n", lensf); return 2; }
            long tot=0; int mx=0; for (int s=0;s<nseq;s++){ tot+=lens[s]; if (lens[s]>mx) mx=lens[s]; }
            if (tot != corpusN) { fprintf(stderr, "lens sum %ld != corpus tokens %d\n", tot, corpusN); return 2; }
            int mpad = ((mx+63)/64)*64; if (mpad > S) S = mpad;
            printf("[corpus-train] '%s': %d tokens, %d sequences (max len %d -> Spad %d)\n", cf, corpusN, nseq, mx, S);
        }
        if (do_combined) {
            /* load the held-out stream into its OWN buffer (corpus holds the TRAIN stream). */
            held = (int*)malloc((size_t)cap*sizeof(int));
            heldN = read_ids_file(pplf, held, cap);
            if (heldN<=0) { fprintf(stderr, "no ids in held-out %s\n", pplf); return 2; }
            printf("[ft] train_corpus '%s' %d tok / %d seqs ; held_out '%s' %d tok ; ppl_win=%d (Spad will be %d)\n",
                   corpusf, corpusN, nseq, pplf, heldN, ppl_win, S);
        }
    }

    if (upload_all_weights()) return 2;
    if (alloc_saved(S)) return 2;

    /* ---- backward modes (allocate grad buffers + Adam state only when needed) ---- */
    if (do_fd || do_train || do_corpus) { if (alloc_grads(S)) return 2; build_params(); }
    if (do_combined) {   /* v1.9 step5c: train the corpus THEN eval held-out ppl on the SAME weights */
        int rc = run_corpus_train_then_ppl(corpus, lens, nseq, corpus_E, held, heldN, ppl_win);
        return rc;
    }
    if (do_ppl) {
        double sumCE=0.0; long npred=0;
        corpus_perplexity(corpus, corpusN, ppl_win, &sumCE, &npred);
        printf("LLAMA_TRAIN_PPL_DONE\n");
        return (npred>0) ? 0 : 1;
    }
    if (do_corpus) {
        int rc = run_corpus_train(corpus, lens, nseq, corpus_E);
        return rc;
    }
    if (do_fd) {
        int rc=run_fdcheck(ids, T);
        return rc;
    }
    if (do_train) {
        double l0=compute_loss(ids,T);
        printf("[train] step0 shifted-CE loss = %.6f ; running %d Adam steps (lr=1e-3 baked)\n", l0, train_K);
        double l=l0;
        for (int t=1;t<=train_K;t++) {
            l=compute_loss(ids,T);
            backward(ids,T);
            adam_step(t);
            if (t==1 || t%5==0 || t==train_K) printf("  step %3d loss %.6f\n", t, l);
        }
        double lf=compute_loss(ids,T);
        printf("[train] %d steps: start %.6f -> final %.6f  (%s)\n", train_K, l0, lf,
               lf < l0 ? "LLAMA_TRAIN_LOSS_DECREASED" : "LLAMA_TRAIN_LOSS_DID_NOT_DECREASE");
        return 0;
    }

    float* logits=(float*)malloc((size_t)NV*sizeof(float));
    forward(ids, T, logits);

    /* finite/argmax/loss */
    int nonfinite=0; double mx=0.0;
    for (int i=0;i<NV;i++){ double g=(double)logits[i]; if(!isfinite(g)){nonfinite++;continue;} double a=fabs(g); if(a>mx)mx=a; }
    int am=argmax_row(logits, NV);
    double loss=ce_loss(ids, T);
    printf("[fwd] last-row logits: nonfinite=%d max_abs=%.5g argmax=%d  shifted_CE_loss=%.6f\n",
           nonfinite, mx, am, loss);

    /* dump last-row logits for downstream */
    const char* out_path = "/home/legoa/llama_train_logits.bin";
    { FILE* of=fopen(out_path,"wb"); if (of){ fwrite(logits,sizeof(float),(size_t)NV,of); fclose(of);
        printf("[fwd] dumped last-row logits [%d] -> %s\n", NV, out_path); } }

    int pass = (nonfinite==0);
    /* cross-check vs the independent oracle (numpy ref dump) */
    if (oraclef) {
        FILE* of=fopen(oraclef,"rb");
        if (!of) { fprintf(stderr, "open oracle '%s': %s\n", oraclef, strerror(errno)); }
        else {
            float* orc=(float*)malloc((size_t)NV*sizeof(float));
            size_t got=fread(orc,sizeof(float),(size_t)NV,of); fclose(of);
            if (got != (size_t)NV) { fprintf(stderr, "oracle has %zu floats, want %d\n", got, NV); }
            else {
                double maxabs=0.0; int worst=-1;
                for (int i=0;i<NV;i++){ double e=fabs((double)logits[i]-(double)orc[i]); if(e>maxabs){maxabs=e;worst=i;} }
                int orc_am=argmax_row(orc, NV);
                int argmatch=(am==orc_am);
                printf("[xcheck] vs oracle %s\n", oraclef);
                printf("[xcheck]   max_abs_logit_diff = %.6g  (worst idx %d: mine=%.5f oracle=%.5f)\n",
                       maxabs, worst, (double)logits[worst], (double)orc[worst]);
                printf("[xcheck]   argmax mine=%d oracle=%d  match=%s\n", am, orc_am, argmatch?"YES":"NO");
                int ok = (maxabs < 1e-2) && argmatch;
                pass = pass && ok;
                printf("%s  (tol: max_abs<1e-2 AND argmax-match)\n",
                       ok ? "LLAMA_TRAIN_FWD_XCHECK_PASS" : "LLAMA_TRAIN_FWD_XCHECK_FAIL");
            }
            free(orc);
        }
    }
    free(logits);
    printf("%s\n", pass ? "LLAMA_TRAIN_FWD_OK" : "LLAMA_TRAIN_FWD_FAIL");
    return pass ? 0 : 1;
}
