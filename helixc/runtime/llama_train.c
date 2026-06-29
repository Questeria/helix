/* llama_train.c -- SmolLM2-135M (Llama-arch) TRAINER FORWARD scaffold (P7-step2 sub-step B).
 *
 * A forward-only pass in a TRAINABLE shape: fp32 weights kept resident + EVERY intermediate
 * activation allocated and SAVED (not freed/reused), so the backward (P7-step3) can read them.
 * This is NOT inference (which streams + reuses buffers); it is the forward leg of training.
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

/* ===================== host embedding gather ===================== */
/* llama: token embedding ONLY (RoPE replaces positional). Pure row copy into s_resid_in[0]. */
static void embed_gather(const int* ids, int T, int S, CUdeviceptr dst) {
    float* hx=(float*)calloc((size_t)S*DM, sizeof(float));
    const float* emb=&g_wbase[off_embed()];
    for (int s=0;s<T;s++) memcpy(&hx[(size_t)s*DM], &emb[(size_t)ids[s]*DM], (size_t)DM*sizeof(float));
    CKX(cuMemcpyHtoD(dst, hx, (size_t)S*DM*sizeof(float)), "h2d embed gather");
    free(hx);
}

/* ===================== the forward (saving every intermediate) ===================== */
static void forward_layer(int L) {
    int group=NH/NKV;                            /* GQA group (9/3=3) */
    CUdeviceptr x = s_resid_in[L];               /* block input (== prev resid_out, or embeds for L0) */
    /* --- attention --- */
    rms_norm(x, s_rms1[L], w_inln[L], Spad, DM);
    mm_ABt(s_rms1[L], w_q[L], s_q[L], Spad, DM, QD);    /* q [S,QD] */
    mm_ABt(s_rms1[L], w_k[L], s_k[L], Spad, DM, KVD);   /* k [S,KVD] */
    mm_ABt(s_rms1[L], w_v[L], s_v[L], Spad, DM, KVD);   /* v [S,KVD] */
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
    mm_ABt(s_ctx[L], w_o[L], s_oproj[L], Spad, QD, DM);  /* o_proj [S,DM] */
    vadd(x, s_oproj[L], s_resid_mid[L], Spad*DM);        /* x + attn (residual) */
    /* --- MLP (SwiGLU) --- */
    rms_norm(s_resid_mid[L], s_rms2[L], w_postln[L], Spad, DM);
    mm_ABt(s_rms2[L], w_gate[L], s_gate[L], Spad, DM, DFF);
    mm_ABt(s_rms2[L], w_up[L],   s_up[L],   Spad, DM, DFF);
    silu_mul(s_gate[L], s_up[L], s_silu[L], Spad*DFF);    /* up * silu(gate) */
    mm_ABt(s_silu[L], w_down[L], s_down[L], Spad, DFF, DM);
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
    printf("[gpu] %s; PTX %ld B, 8 kernels bound\n", name, psz);
    return 0;
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

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <combined.ptx> <smollm2-135m.weights> <ids.txt> [oracle_logits.bin]\n", argv[0]);
        return 2;
    }
    const char* ptx=argv[1]; const char* wts=argv[2]; const char* idsf=argv[3];
    const char* oraclef = (argc>=5) ? argv[4] : NULL;

    if (load_weights(wts)) return 2;
    if (device_init(ptx)) return 2;

    int ids[4096];
    int T=read_ids_file(idsf, ids, 4096);
    if (T<=0) { fprintf(stderr, "no ids in %s\n", idsf); return 2; }
    int S=((T+63)/64)*64;
    printf("[run] T=%d ids:", T); for (int i=0;i<T;i++) printf(" %d", ids[i]); printf("  (Spad=%d)\n", S);

    if (upload_all_weights()) return 2;
    if (alloc_saved(S)) return 2;

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
