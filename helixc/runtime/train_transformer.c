/* train_transformer.c -- Helix v1.0 CAPSTONE training harness (Option A: trusted C
 * launcher; ALL math is kovc-emitted PTX, the C only does memory + launch sequencing).
 * Tiny 2-layer pre-norm transformer (default V=32,d=16,S=16,1 head,H=64). See
 * docs/HELIX_CAPSTONE_TRAIN_PLAN.md. Forward verified vs numpy oracle to 1e-6 (542b02c).
 *
 * STAGE C/D: forward + FULL backward, each weight gradient verified vs a central
 * finite-difference of the loss (INDEPENDENT: uses only the verified forward).
 *
 * T2/M6 (capstone re-train): the dims are now ENV-PARAMETERIZED (HX_S/HX_D/HX_H/HX_V/
 * HX_NL/HX_K), DEFAULTING to the exact v1.0 capstone so the v1.0 audit is byte-for-byte
 * unchanged. HX_OPT=1 selects the OPTIMIZED op-set kernels (tiled/Tensor-Core GEMMs +
 * block-reduction softmax) in place of the naive ones, with the matching launch geometry.
 * The training MATH is identical in both paths -- only the kernels (and their launch
 * geometry) change -- so the 2% loss-parity vs the numpy oracle is a real correctness gate.
 * The tiled GEMMs require every matmul axis %64==0 (no boundary guard), so HX_OPT runs at a
 * scaled-up size (e.g. S=128 D=64 H=256 V=128) where they are valid and faster-than-naive.
 *
 * Build (WSL): gcc train_transformer.c -O2 -o /tmp/train -lcuda -lm -L/usr/lib/wsl/lib
 * Run:        /tmp/train <combined.ptx> [verify]
 *   v1.0 capstone:  /tmp/train combined.ptx
 *   optimized:      HX_OPT=1 HX_S=128 HX_D=64 HX_H=256 HX_V=128 /tmp/train combined.ptx
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

/* ---- ENV-parameterized dims (default = the exact v1.0 capstone) ---- */
static int V = 32, D = 16, S = 16, H = 64, NL = 2, K = 500, OPT = 0;
#define MAXNL 16
/* derived sizes (set in init_dims) */
static int SD, SS, SH, SV, DD, DH, HD, DV, NW;
static float ATTN_SCALE = 0.25f;   /* 1/sqrt(d) */
static int TILE = 64;              /* tiled-GEMM block tile (BM=BN); BK=8, micro 4x4 -> block 16x16 */
static void init_dims(void) {
    const char* e;
    if ((e = getenv("HX_V")))  V  = atoi(e);
    if ((e = getenv("HX_D")))  D  = atoi(e);
    if ((e = getenv("HX_S")))  S  = atoi(e);
    if ((e = getenv("HX_H")))  H  = atoi(e);
    if ((e = getenv("HX_NL"))) NL = atoi(e);
    if ((e = getenv("HX_K")))  K  = atoi(e);
    if ((e = getenv("HX_OPT"))) OPT = atoi(e);
    if (NL > MAXNL) { fprintf(stderr, "NL>%d unsupported\n", MAXNL); exit(2); }
    SD = S*D; SS = S*S; SH = S*H; SV = S*V; DD = D*D; DH = D*H; HD = H*D; DV = D*V;
    NW = NL*(4*DD + 4*D + DH + HD) + 2*D + DV;
    ATTN_SCALE = 1.0f / sqrtf((float)D);
}

static int check(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) { const char* m = 0; cuGetErrorString(r, &m); fprintf(stderr, "CUDA %s: %s (%d)\n", what, m ? m : "?", (int)r); return 1; }
    return 0;
}
#define CK(c, w) do { if (check((c), (w))) return 2; } while (0)
#define CKX(c, w) do { if (check((c), (w))) exit(2); } while (0)

static CUcontext ctx;
/* forward + elementwise kernels (naive set; opt set adds tiled GEMM + blockred softmax) */
static CUfunction f_add, f_ln, f_mm, f_qkt, f_sm, f_gelu;
static CUfunction f_ceg, f_atb, f_abt, f_geb, f_smb, f_lnbx, f_lnbg, f_scale, f_adam;
/* OPT-path kernels (only loaded when OPT) */
static CUfunction f_mm_t, f_atb_t, f_abt_t, f_sm_b, f_scale_rt;
/* OPT-path block-reduction backward/save redux kernels (T2/M6) */
static CUfunction f_ln_b, f_lnbx_b, f_smb_b;
static CUdeviceptr dbc1, dbc2, d_attn_scale;
static int Si, Di, Hi, Vi, SDi, SHi;

static uint32_t xs_state = 0x12345678u;
static uint32_t xs(void) { uint32_t x = xs_state; x ^= x << 13; x ^= x >> 17; x ^= x << 5; xs_state = x; return x; }
static float rnd(float scale) { return ((float)(int32_t)xs() / 2147483648.0f) * scale; }

/* coarse per-category profiling (every LX syncs, so a monotonic timer around a launch is
 * exact). PROF=1 accumulates time into t_gemm / t_redux / t_elem / t_other. */
static int PROF = 0;
static double t_gemm=0, t_redux=0, t_elem=0, t_other=0;
/* FASTSYNC=1 (env HX_FASTSYNC, default 0 = OFF): skip the per-kernel cuCtxSynchronize so
 * launches pipeline on the default stream (CUDA guarantees same-stream ordering, so each
 * kernel still sees the prior's writes; the blocking cuMemcpyDtoH in forward_full provides
 * the host sync the loss read needs). PROF forces per-kernel sync (required to time each
 * category). Measured to barely move the wall-clock here (the per-step loss D2H already
 * serializes steps), but available as an honest launch-sequencing option for both paths.
 * Default OFF keeps the v1.0 capstone audit byte-identical (per-kernel sync). */
static int FASTSYNC = 0;
static double pf_t0(void){ struct timespec ts; clock_gettime(CLOCK_MONOTONIC,&ts); return ts.tv_sec*1000.0+ts.tv_nsec/1.0e6; }
#define SYNC(w) do { if (!FASTSYNC || PROF) CKX(cuCtxSynchronize(), w); } while (0)
#define LX(fn, grid, block, args) do { CKX(cuLaunchKernel((fn), (grid),1,1, (block),1,1, 0,0, (args), 0), #fn); SYNC("sync " #fn); } while (0)
#define LX2(fn, gx, gy, bx, by, args) do { CKX(cuLaunchKernel((fn), (gx),(gy),1, (bx),(by),1, 0,0, (args), 0), #fn); SYNC("sync " #fn); } while (0)
/* timed launch into a category bucket */
#define LXT(bucket, fn, grid, block, args) do { double _t=PROF?pf_t0():0; CKX(cuLaunchKernel((fn),(grid),1,1,(block),1,1,0,0,(args),0), #fn); SYNC("sync " #fn); if(PROF)bucket+=pf_t0()-_t; } while(0)
#define LXT2(bucket, fn, gx, gy, bx, by, args) do { double _t=PROF?pf_t0():0; CKX(cuLaunchKernel((fn),(gx),(gy),1,(bx),(by),1,0,0,(args),0), #fn); SYNC("sync " #fn); if(PROF)bucket+=pf_t0()-_t; } while(0)
/* elementwise launch over n flat elements. The elem kernels use the grid-stride index
 * block_idx()*block_dim()+thread_idx() with NO bounds guard, so blocks*threads must == n
 * exactly. naive (v1.0 baseline): block=1, grid=n (1 thread/block -- poor occupancy). OPT:
 * the largest of {256,128,64,32} that divides n, grid=n/block -- same kernel, far higher
 * occupancy, no kernel change. */
static int elem_block(int n) { if (!OPT) return 1; int b[]={256,128,64,32}; for (int i=0;i<4;i++) if (n%b[i]==0) return b[i]; return 1; }
#define LXE(bucket, fn, n, args) do { int _b=elem_block(n); LXT(bucket, fn, (unsigned)((n)/_b), _b, args); } while(0)

/* C[M,N] = A[M,K] @ B[K,N]. naive: grid=M, block=N, 1 thread/cell. tiled: grid=(N/64,M/64)
 * block=(16,16). MATH identical; only launch geometry + which kernel differ. */
static void mm_AB(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, int M, int Kc, int N) {
    int m=M,k=Kc,n=N;
    if (OPT) { void* ar[] = { &a,&b,&c,&m,&k,&n }; LXT2(t_gemm, f_mm_t, n/TILE, m/TILE, TILE/4, TILE/4, ar); }
    else     { void* ar[] = { &a,&b,&c,&m,&k,&n }; LXT(t_gemm, f_mm, M, N, ar); }
}
/* C[M,N] = A[M,K] @ B[N,K]^T  (A@B^T). naive gpu_matmul_abt: grid=M,block=N. tiled abt:
 * grid=(N/64,M/64) block=(16,16). */
static void mm_ABt(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, int M, int Kc, int N) {
    int m=M,k=Kc,n=N;
    if (OPT) { void* ar[] = { &a,&b,&c,&m,&k,&n }; LXT2(t_gemm, f_abt_t, n/TILE, m/TILE, TILE/4, TILE/4, ar); }
    else     { void* ar[] = { &a,&b,&c,&m,&k,&n }; LXT(t_gemm, f_abt, M, N, ar); }
}
/* C[K,N] = A[M,K]^T @ B[M,N]  (A^T@B), contraction = M. naive gpu_matmul_atb: grid=K,block=N.
 * tiled atb: grid=(N/64,K/64) block=(16,16); kernel args (mm=M contraction, kk=K outrows, nn=N). */
static void mm_AtB(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, int M, int Kc, int N) {
    int m=M,k=Kc,n=N;
    if (OPT) { void* ar[] = { &a,&b,&c,&m,&k,&n }; LXT2(t_gemm, f_atb_t, n/TILE, k/TILE, TILE/4, TILE/4, ar); }
    else     { void* ar[] = { &a,&b,&c,&m,&k,&n }; LXT(t_gemm, f_atb, Kc, N, ar); }
}
/* row softmax over [rows,cols]. naive gpu_softmax: grid=rows,block=1. blockred: grid=rows,block=256. */
static void softmax(CUdeviceptr x, CUdeviceptr y, int rows, int cols) {
    int r=rows,c=cols;
    if (OPT) { void* ar[] = { &x,&y,&r,&c }; LXT(t_redux, f_sm_b, rows, 256, ar); }
    else     { void* ar[] = { &x,&y,&r,&c }; LXT(t_redux, f_sm, rows, 1, ar); }
}
/* LN forward+save over [rows=S, cols]. naive gpu_layernorm_fwd_save: grid=S,block=1.
 * opt layernorm_fwd_save_blockred: grid=S,block=256 (block-reduction). */
static void ln_fwd(CUdeviceptr x, CUdeviceptr y, CUdeviceptr g, CUdeviceptr b, CUdeviceptr ist, int cols) {
    int c=cols; void* ar[] = { &x,&y,&g,&b,&ist,&c };
    if (OPT) { LXT(t_redux, f_ln_b, S, 256, ar); } else { LXT(t_redux, f_ln, S, 1, ar); }
}
/* LN backward dx over [rows=S, cols]. naive gpu_layernorm_backward_dx: grid=S,block=1.
 * opt layernorm_backward_dx_blockred: grid=S,block=256. */
static void ln_bwd_dx(CUdeviceptr x, CUdeviceptr dy, CUdeviceptr g, CUdeviceptr ist, CUdeviceptr dx, int cols) {
    int c=cols; void* ar[] = { &x,&dy,&g,&ist,&dx,&c };
    if (OPT) { LXT(t_redux, f_lnbx_b, S, 256, ar); } else { LXT(t_redux, f_lnbx, S, 1, ar); }
}
/* softmax backward over [rows=S, cols=S]. naive gpu_softmax_backward: grid=S,block=1.
 * opt softmax_backward_blockred: grid=S,block=256. */
static void softmax_bwd(CUdeviceptr p, CUdeviceptr dp, CUdeviceptr da, int rows, int cols) {
    int r=rows,c=cols; void* ar[] = { &p,&dp,&da,&r,&c };
    if (OPT) { LXT(t_redux, f_smb_b, rows, 256, ar); } else { LXT(t_redux, f_smb, rows, 1, ar); }
}
/* in-place scale by 1/sqrt(d): naive gpu_scale_inplace bakes 0.25 (d=16 only); opt uses the
 * runtime-scalar gpu_scale_rt so it is correct at any d. */
static void scale_attn(CUdeviceptr a, int n) {
    int nn=n;
    if (OPT) { void* ar[] = { &a, &d_attn_scale, &nn }; LXE(t_elem, f_scale_rt, n, ar); }
    else     { void* ar[] = { &a, &nn }; LXE(t_elem, f_scale, n, ar); }
}

/* weights / activations / grads (sized to MAXNL; only NL used) */
static CUdeviceptr Wq[MAXNL], Wk[MAXNL], Wv[MAXNL], Wo[MAXNL], LN1g[MAXNL], LN1b[MAXNL], LN2g[MAXNL], LN2b[MAXNL], W1[MAXNL], W2[MAXNL], LNfg, LNfb, W_lm;
static CUdeviceptr xn1[MAXNL], Qb[MAXNL], Kb[MAXNL], Vb[MAXNL], scores[MAXNL], attn[MAXNL], ao[MAXNL], proj[MAXNL], h1[MAXNL], xn2[MAXNL], amlp[MAXNL], gmlp[MAXNL], mmlp[MAXNL], h2[MAXNL], ist1[MAXNL], ist2[MAXNL];
static CUdeviceptr xf, istf, logits, x_in, targets_f;
static CUdeviceptr dWq[MAXNL], dWk[MAXNL], dWv[MAXNL], dWo[MAXNL], dLN1[MAXNL], dLN2[MAXNL], dW1[MAXNL], dW2[MAXNL], dLNf, dW_lm;
static CUdeviceptr g_dlog, g_dxf, g_dh2[MAXNL], g_dh1, g_dg, g_da, g_dxn2, g_dxln2, g_dao, g_dVv, g_dQ, g_dK, g_dattn, g_dsc, g_dxn1, g_dxln1, g_t1, g_t2, g_t3, g_t12, g_dxscr;

static CUdeviceptr A(int nf) { CUdeviceptr p; CKX(cuMemAlloc(&p, (size_t)nf * sizeof(float)), "alloc"); return p; }

static void gen_weights(float* w) {
    int o = 0; float sc_d = sqrtf(2.0f / (float)D), sc_h = sqrtf(2.0f / (float)H);
    for (int L = 0; L < NL; L++) {
        for (int i = 0; i < DD; i++) w[o++] = rnd(sc_d);
        for (int i = 0; i < DD; i++) w[o++] = rnd(sc_d);
        for (int i = 0; i < DD; i++) w[o++] = rnd(sc_d);
        for (int i = 0; i < DD; i++) w[o++] = rnd(sc_d);
        for (int i = 0; i < D; i++) w[o++] = 1.0f;
        for (int i = 0; i < D; i++) w[o++] = 0.0f;
        for (int i = 0; i < D; i++) w[o++] = 1.0f;
        for (int i = 0; i < D; i++) w[o++] = 0.0f;
        for (int i = 0; i < DH; i++) w[o++] = rnd(sc_d);
        for (int i = 0; i < HD; i++) w[o++] = rnd(sc_h);
    }
    for (int i = 0; i < D; i++) w[o++] = 1.0f;
    for (int i = 0; i < D; i++) w[o++] = 0.0f;
    for (int i = 0; i < DV; i++) w[o++] = rnd(sc_d);
}

static void upload_weights(const float* w) {
    int o = 0;
#define UP(dst, nf) do { CKX(cuMemcpyHtoD((dst), &w[o], (size_t)(nf)*sizeof(float)), "h2d w"); o += (nf); } while (0)
    for (int L = 0; L < NL; L++) { UP(Wq[L], DD); UP(Wk[L], DD); UP(Wv[L], DD); UP(Wo[L], DD); UP(LN1g[L], D); UP(LN1b[L], D); UP(LN2g[L], D); UP(LN2b[L], D); UP(W1[L], DH); UP(W2[L], HD); }
    UP(LNfg, D); UP(LNfb, D); UP(W_lm, DV);
#undef UP
}

static void forward_layer(int L, CUdeviceptr x) {
    ln_fwd(x, xn1[L], LN1g[L], LN1b[L], ist1[L], D);
    mm_AB(xn1[L], Wq[L], Qb[L], S, D, D);
    mm_AB(xn1[L], Wk[L], Kb[L], S, D, D);
    mm_AB(xn1[L], Wv[L], Vb[L], S, D, D);
    /* scores = (1/sqrt d) Q@K^T. naive: fused gpu_qkt (bakes 0.25). opt: tiled A@B^T then runtime scale. */
    if (OPT) { mm_ABt(Qb[L], Kb[L], scores[L], S, D, S); scale_attn(scores[L], SS); }
    else { void* aqk[] = { &Qb[L], &Kb[L], &scores[L], &Si, &Di }; LXT(t_gemm, f_qkt, S, S, aqk); }
    softmax(scores[L], attn[L], S, S);
    mm_AB(attn[L], Vb[L], ao[L], S, S, D);
    mm_AB(ao[L], Wo[L], proj[L], S, D, D);
    void* ah1[] = { &x, &proj[L], &h1[L], &SDi }; LXE(t_elem, f_add, SD, ah1);
    ln_fwd(h1[L], xn2[L], LN2g[L], LN2b[L], ist2[L], D);
    mm_AB(xn2[L], W1[L], amlp[L], S, D, H);
    void* ag[] = { &amlp[L], &gmlp[L], &SHi }; LXE(t_elem, f_gelu, SH, ag);
    mm_AB(gmlp[L], W2[L], mmlp[L], S, H, D);
    void* ah2[] = { &h1[L], &mmlp[L], &h2[L], &SDi }; LXE(t_elem, f_add, SD, ah2);
}

static double ce_loss(const float* hlog, const int* tgt) {
    double loss = 0.0;
    for (int s = 0; s < S; s++) { const float* r = &hlog[s * V]; float mx = r[0]; for (int c = 1; c < V; c++) if (r[c] > mx) mx = r[c]; double sm = 0.0; for (int c = 0; c < V; c++) sm += exp((double)(r[c] - mx)); loss += -log(exp((double)(r[tgt[s]] - mx)) / sm); }
    return loss;
}

static int* g_tgt;
static double forward_full(void) {
    forward_layer(0, x_in);
    for (int L = 1; L < NL; L++) forward_layer(L, h2[L-1]);
    CUdeviceptr xlast = h2[NL-1];
    ln_fwd(xlast, xf, LNfg, LNfb, istf, D);
    mm_AB(xf, W_lm, logits, S, D, V);
    float* hlog = (float*)malloc((size_t)SV*sizeof(float)); CKX(cuMemcpyDtoH(hlog, logits, (size_t)SV*sizeof(float)), "d2h logits");
    double l = ce_loss(hlog, g_tgt); free(hlog); return l;
}

static void backward_layer(int L, CUdeviceptr dh2_in, CUdeviceptr dx_out) {
    /* MLP */
    mm_AtB(gmlp[L], dh2_in, dW2[L], S, H, D);
    mm_ABt(dh2_in, W2[L], g_dg, S, D, H);
    void* ada[] = { &amlp[L], &g_dg, &g_da, &SHi }; LXE(t_elem, f_geb, SH, ada);
    mm_AtB(xn2[L], g_da, dW1[L], S, D, H);
    mm_ABt(g_da, W1[L], g_dxn2, S, H, D);
    /* LN2 bwd: dx is block-reduced (opt); dgb (column reduce) stays naive. */
    ln_bwd_dx(h1[L], g_dxn2, LN2g[L], ist2[L], g_dxln2, D);
    void* al2g[]= { &h1[L], &g_dxn2, &ist2[L], &dLN2[L], &Si, &Di };      LXT(t_redux, f_lnbg, D, 1, al2g);
    void* adh1[]= { &g_dxln2, &dh2_in, &g_dh1, &SDi }; LXE(t_elem, f_add, SD, adh1);
    /* attn proj */
    mm_AtB(ao[L], g_dh1, dWo[L], S, D, D);
    mm_ABt(g_dh1, Wo[L], g_dao, S, D, D);
    /* attn core */
    mm_AtB(attn[L], g_dao, g_dVv, S, S, D);
    mm_ABt(g_dao, Vb[L], g_dattn, S, D, S);
    softmax_bwd(attn[L], g_dattn, g_dsc, S, S);
    mm_AB(g_dsc, Kb[L], g_dQ, S, S, D); scale_attn(g_dQ, SD);
    mm_AtB(g_dsc, Qb[L], g_dK, S, S, D); scale_attn(g_dK, SD);
    /* QKV weight grads */
    mm_AtB(xn1[L], g_dQ, dWq[L], S, D, D);
    mm_AtB(xn1[L], g_dK, dWk[L], S, D, D);
    mm_AtB(xn1[L], g_dVv, dWv[L], S, D, D);
    /* dxn1 = dQ@Wq^T + dK@Wk^T + dVv@Wv^T */
    mm_ABt(g_dQ, Wq[L], g_t1, S, D, D);
    mm_ABt(g_dK, Wk[L], g_t2, S, D, D);
    mm_ABt(g_dVv, Wv[L], g_t3, S, D, D);
    void* a12[] = { &g_t1, &g_t2, &g_t12, &SDi };  LXE(t_elem, f_add, SD, a12);
    void* a123[]= { &g_t12, &g_t3, &g_dxn1, &SDi }; LXE(t_elem, f_add, SD, a123);
    /* LN1 bwd */
    CUdeviceptr xL = (L == 0) ? x_in : h2[L-1];
    ln_bwd_dx(xL, g_dxn1, LN1g[L], ist1[L], g_dxln1, D);
    void* al1g[]= { &xL, &g_dxn1, &ist1[L], &dLN1[L], &Si, &Di };      LXT(t_redux, f_lnbg, D, 1, al1g);
    void* adx[] = { &g_dxln1, &g_dh1, &dx_out, &SDi }; LXE(t_elem, f_add, SD, adx);
}

static void backward_full(void) {
    void* ace[] = { &logits, &targets_f, &g_dlog, &Si, &Vi }; LXT(t_redux, f_ceg, S, 1, ace);
    mm_AtB(xf, g_dlog, dW_lm, S, D, V);
    mm_ABt(g_dlog, W_lm, g_dxf, S, V, D);
    CUdeviceptr xlast = h2[NL-1];
    ln_bwd_dx(xlast, g_dxf, LNfg, istf, g_dh2[NL-1], D);
    void* alfg[]= { &xlast, &g_dxf, &istf, &dLNf, &Si, &Di };          LXT(t_redux, f_lnbg, D, 1, alfg);
    for (int L = NL-1; L >= 1; L--) backward_layer(L, g_dh2[L], g_dh2[L-1]);
    backward_layer(0, g_dh2[0], g_dxscr);
}

typedef struct { CUdeviceptr w, g, m, v; int n; } Param;
static Param params[256]; static int nparams = 0;
static void addp(CUdeviceptr w, CUdeviceptr g, int n) {
    Param* p = &params[nparams++]; p->w = w; p->g = g; p->n = n; p->m = A(n); p->v = A(n);
    CKX(cuMemsetD8(p->m, 0, (size_t)n * sizeof(float)), "zero m");
    CKX(cuMemsetD8(p->v, 0, (size_t)n * sizeof(float)), "zero v");
}
static void build_params(void) {
    for (int L = 0; L < NL; L++) {
        addp(Wq[L], dWq[L], DD); addp(Wk[L], dWk[L], DD); addp(Wv[L], dWv[L], DD); addp(Wo[L], dWo[L], DD);
        addp(W1[L], dW1[L], DH); addp(W2[L], dW2[L], HD);
        addp(LN1g[L], dLN1[L], D); addp(LN1b[L], dLN1[L] + (CUdeviceptr)(D * sizeof(float)), D);
        addp(LN2g[L], dLN2[L], D); addp(LN2b[L], dLN2[L] + (CUdeviceptr)(D * sizeof(float)), D);
    }
    addp(LNfg, dLNf, D); addp(LNfb, dLNf + (CUdeviceptr)(D * sizeof(float)), D); addp(W_lm, dW_lm, DV);
}
static void adam_step(int t) {
    float bc1 = 1.0f / (1.0f - powf(0.9f, (float)t));
    float bc2 = 1.0f / (1.0f - powf(0.999f, (float)t));
    CKX(cuMemcpyHtoD(dbc1, &bc1, sizeof(float)), "bc1"); CKX(cuMemcpyHtoD(dbc2, &bc2, sizeof(float)), "bc2");
    for (int i = 0; i < nparams; i++) { Param* p = &params[i]; void* a[] = { &p->w, &p->g, &p->m, &p->v, &dbc1, &dbc2 }; LXT(t_elem, f_adam, p->n, 1, a); }
}

static double now_ms(void) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return ts.tv_sec*1000.0 + ts.tv_nsec/1.0e6; }

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <combined.ptx> [verify]\n", argv[0]); return 2; }
    init_dims();
    { const char* e = getenv("HX_PROF"); if (e) PROF = atoi(e); }
    { const char* e = getenv("HX_FASTSYNC"); if (e) FASTSYNC = atoi(e); }
    Si=S; Di=D; Hi=H; Vi=V; SDi=SD; SHi=SH;
    g_tgt = (int*)malloc((size_t)S*sizeof(int));
    FILE* f = fopen(argv[1], "rb"); if (!f) { fprintf(stderr, "open ptx\n"); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* ptx = (char*)malloc(sz + 1); if (fread(ptx, 1, sz, f) != (size_t)sz) return 2; ptx[sz] = 0; fclose(f);
    CK(cuInit(0), "init"); CUdevice dev; CK(cuDeviceGet(&dev, 0), "dev");
    char gpu[256]; gpu[0] = 0; cuDeviceGetName(gpu, 256, dev);
    CK(cuCtxCreate(&ctx, 0, dev), "ctx");
    CUmodule mod; CK(cuModuleLoadData(&mod, ptx), "load");
    CK(cuModuleGetFunction(&f_add, mod, "vector_add"), "add");
    CK(cuModuleGetFunction(&f_ln, mod, "gpu_layernorm_fwd_save"), "ln");
    CK(cuModuleGetFunction(&f_mm, mod, "naive_matmul"), "mm");
    CK(cuModuleGetFunction(&f_qkt, mod, "gpu_qkt"), "qkt");
    CK(cuModuleGetFunction(&f_sm, mod, "gpu_softmax"), "sm");
    CK(cuModuleGetFunction(&f_gelu, mod, "gpu_gelu"), "gelu");
    CK(cuModuleGetFunction(&f_ceg, mod, "gpu_ce_softmax_grad"), "ceg");
    CK(cuModuleGetFunction(&f_atb, mod, "gpu_matmul_atb"), "atb");
    CK(cuModuleGetFunction(&f_abt, mod, "gpu_matmul_abt"), "abt");
    CK(cuModuleGetFunction(&f_geb, mod, "gpu_gelu_backward"), "geb");
    CK(cuModuleGetFunction(&f_smb, mod, "gpu_softmax_backward"), "smb");
    CK(cuModuleGetFunction(&f_lnbx, mod, "gpu_layernorm_backward_dx"), "lnbx");
    CK(cuModuleGetFunction(&f_lnbg, mod, "gpu_layernorm_backward_dgb"), "lnbg");
    CK(cuModuleGetFunction(&f_scale, mod, "gpu_scale_inplace"), "scale");
    CK(cuModuleGetFunction(&f_adam, mod, "gpu_adam"), "adam");
    if (OPT) {
        CK(cuModuleGetFunction(&f_mm_t,  mod, "tiled_matmul"),     "mm_t");
        CK(cuModuleGetFunction(&f_atb_t, mod, "tiled_matmul_atb"), "atb_t");
        CK(cuModuleGetFunction(&f_abt_t, mod, "tiled_matmul_abt"), "abt_t");
        CK(cuModuleGetFunction(&f_sm_b,  mod, "softmax_blockred"), "sm_b");
        CK(cuModuleGetFunction(&f_scale_rt, mod, "gpu_scale_rt"),  "scale_rt");
        CK(cuModuleGetFunction(&f_ln_b,   mod, "layernorm_fwd_save_blockred"),   "ln_b");
        CK(cuModuleGetFunction(&f_lnbx_b, mod, "layernorm_backward_dx_blockred"), "lnbx_b");
        CK(cuModuleGetFunction(&f_smb_b,  mod, "softmax_backward_blockred"),      "smb_b");
    }
    for (int L = 0; L < NL; L++) {
        Wq[L]=A(DD); Wk[L]=A(DD); Wv[L]=A(DD); Wo[L]=A(DD); LN1g[L]=A(D); LN1b[L]=A(D); LN2g[L]=A(D); LN2b[L]=A(D); W1[L]=A(DH); W2[L]=A(HD);
        xn1[L]=A(SD); Qb[L]=A(SD); Kb[L]=A(SD); Vb[L]=A(SD); scores[L]=A(SS); attn[L]=A(SS); ao[L]=A(SD); proj[L]=A(SD); h1[L]=A(SD); xn2[L]=A(SD); amlp[L]=A(SH); gmlp[L]=A(SH); mmlp[L]=A(SD); h2[L]=A(SD); ist1[L]=A(S); ist2[L]=A(S);
        dWq[L]=A(DD); dWk[L]=A(DD); dWv[L]=A(DD); dWo[L]=A(DD); dLN1[L]=A(2*D); dLN2[L]=A(2*D); dW1[L]=A(DH); dW2[L]=A(HD); g_dh2[L]=A(SD);
    }
    LNfg=A(D); LNfb=A(D); W_lm=A(DV); xf=A(SD); istf=A(S); logits=A(SV); x_in=A(SD); targets_f=A(S); dLNf=A(2*D); dW_lm=A(DV);
    g_dlog=A(SV); g_dxf=A(SD); g_dh1=A(SD); g_dg=A(SH); g_da=A(SH); g_dxn2=A(SD); g_dxln2=A(SD); g_dao=A(SD); g_dVv=A(SD); g_dQ=A(SD); g_dK=A(SD); g_dattn=A(SS); g_dsc=A(SS); g_dxn1=A(SD); g_dxln1=A(SD); g_t1=A(SD); g_t2=A(SD); g_t3=A(SD); g_t12=A(SD); g_dxscr=A(SD);
    d_attn_scale = A(1); CK(cuMemcpyHtoD(d_attn_scale, &ATTN_SCALE, sizeof(float)), "h2d scale");

    float* hw = (float*)malloc(NW * sizeof(float));
    gen_weights(hw);
    FILE* wf = fopen("init_weights.bin", "wb"); if (wf) { fwrite(hw, sizeof(float), NW, wf); fclose(wf); }
    upload_weights(hw);
    float* hx = (float*)calloc((size_t)SD, sizeof(float)); for (int s = 0; s < S; s++) hx[s * D + (s % D)] = 1.0f;
    CK(cuMemcpyHtoD(x_in, hx, (size_t)SD*sizeof(float)), "h2d x");
    float* htf = (float*)malloc((size_t)S*sizeof(float)); for (int s = 0; s < S; s++) { g_tgt[s] = (s + 1) % S; htf[s] = (float)g_tgt[s]; }
    CK(cuMemcpyHtoD(targets_f, htf, (size_t)S*sizeof(float)), "h2d tgt");

    double loss0 = forward_full();
    printf("GPU [%s] OPT=%d dims S=%d D=%d H=%d V=%d NL=%d K=%d  step0 loss = %.6f (sum CE; mean=%.4f, log V=%.4f)\n", gpu, OPT, S, D, H, V, NL, K, loss0, loss0 / (double)S, logf((float)V));
    dbc1 = A(1); dbc2 = A(1);
    build_params();
    if (argc > 2 && strcmp(argv[2], "verify") == 0) {
        backward_full();
        int layer_block = 4*DD + 4*D + DH + HD;
        int off_Wlm = NL*layer_block + 2*D;
        struct { const char* name; int hoff; CUdeviceptr grad; int n; } checks[] = {
            { "dW_lm",  off_Wlm,                 dW_lm,  DV },
            { "dW2[1]", 1*layer_block + 4*DD+4*D+DH, dW2[1], HD },
            { "dWo[1]", 1*layer_block + 3*DD,    dWo[1], DD },
            { "dWq[0]", 0*layer_block + 0,       dWq[0], DD },
            { "dW1[0]", 0*layer_block + 4*DD+4*D, dW1[0], DH },
            { "dWv[1]", 1*layer_block + 2*DD,    dWv[1], DD },
        };
        int total_bad = 0; float h = 1.0e-3f;
        for (int ci = 0; ci < (int)(sizeof(checks)/sizeof(checks[0])); ci++) {
            float* hg = (float*)malloc(checks[ci].n * sizeof(float));
            CK(cuMemcpyDtoH(hg, checks[ci].grad, checks[ci].n * sizeof(float)), "d2h grad");
            int bad = 0; double maxe = 0.0;
            int idxs[5]; int ni = checks[ci].n < 5 ? checks[ci].n : 5;
            for (int k = 0; k < ni; k++) idxs[k] = (k * 37 + 1) % checks[ci].n;
            for (int k = 0; k < ni; k++) {
                int gi = idxs[k]; int o = checks[ci].hoff + gi;
                float save = hw[o];
                hw[o] = save + h; upload_weights(hw); double Lp = forward_full();
                hw[o] = save - h; upload_weights(hw); double Lm = forward_full();
                hw[o] = save; upload_weights(hw);
                double fd = (Lp - Lm) / (2.0 * h);
                double gv = (double)hg[gi];
                double e = gv - fd; if (e < 0) e = -e; if (e > maxe) maxe = e;
                double af = fd < 0 ? -fd : fd;
                if (isnan(gv) || (e > 1.0e-3 && e > 0.05 * af)) { if (bad < 3) fprintf(stderr, "  %s[%d] grad %g fd %g\n", checks[ci].name, gi, gv, fd); bad++; }
            }
            printf("  %-7s: %d/%d checked, max|grad-fd|=%g -> %s\n", checks[ci].name, ni, ni, maxe, bad ? "FAIL" : "ok");
            total_bad += bad; free(hg);
        }
        printf("GPU [%s] backward finite-diff: %s\n", gpu, total_bad ? "FAIL" : "PASS");
        return total_bad ? 1 : 0;
    }
    /* TRAIN: K Adam steps (forward -> backward -> Adam per weight). Timed wall-clock. */
    FILE* cf = fopen("loss_curve.csv", "wb");
    double loss = loss0;
    CKX(cuCtxSynchronize(), "presync");
    double t_start = now_ms();
    for (int t = 1; t <= K; t++) {
        loss = forward_full();
        backward_full();
        adam_step(t);
        if (t == 1 || t % 25 == 0 || t == K) { printf("  step %4d loss %.6f (mean %.5f)\n", t, loss, loss / (double)S); if (cf) fprintf(cf, "%d,%.8f\n", t, loss); }
    }
    double lf = forward_full();
    double t_end = now_ms();
    if (cf) { fprintf(cf, "%d,%.8f\n", K + 1, lf); fclose(cf); }
    double wall = t_end - t_start;
    printf("GPU [%s] train K=%d: start loss %.6f -> final %.6f (mean %.5f)\n", gpu, K, loss0, lf, lf / (double)S);
    printf("TRAIN_WALL_MS=%.3f  (OPT=%d S=%d D=%d H=%d V=%d NL=%d K=%d, %.4f ms/step)\n", wall, OPT, S, D, H, V, NL, K, wall/(double)K);
    if (PROF) { double acc=t_gemm+t_redux+t_elem+t_other; printf("PROF_MS gemm=%.1f redux=%.1f elem=%.1f other=%.1f acc=%.1f (gemm %.0f%% redux %.0f%% elem %.0f%%)\n", t_gemm,t_redux,t_elem,t_other,acc, 100*t_gemm/acc,100*t_redux/acc,100*t_elem/acc); }
    return 0;
}
