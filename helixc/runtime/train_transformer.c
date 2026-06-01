/* train_transformer.c -- Helix v1.0 CAPSTONE training harness (Option A: trusted C
 * launcher; ALL math is kovc-emitted PTX, the C only does memory + launch sequencing).
 * Tiny 2-layer pre-norm transformer (V=32,d=16,S=16,1 head,H=64). See
 * docs/HELIX_CAPSTONE_TRAIN_PLAN.md. Forward verified vs numpy oracle to 1e-6 (542b02c).
 *
 * STAGE C/D (this file): forward + FULL backward, each weight gradient verified vs a
 * central finite-difference of the loss (INDEPENDENT: uses only the verified forward).
 * Build (WSL): gcc train_transformer.c -O2 -o /tmp/train -lcuda -lm -L/usr/lib/wsl/lib
 * Run:        /tmp/train <combined.ptx>
 */
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#define V 32
#define D 16
#define S 16
#define H 64
#define NL 2
#define SD (S*D)
#define SS (S*S)
#define SH (S*H)
#define SV (S*V)
#define DD (D*D)
#define DH (D*H)
#define HD (H*D)
#define DV (D*V)
#define NW (NL*(4*DD + 4*D + DH + HD) + 2*D + DV)

static int check(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) { const char* m = 0; cuGetErrorString(r, &m); fprintf(stderr, "CUDA %s: %s (%d)\n", what, m ? m : "?", (int)r); return 1; }
    return 0;
}
#define CK(c, w) do { if (check((c), (w))) return 2; } while (0)
#define CKX(c, w) do { if (check((c), (w))) exit(2); } while (0)

static CUcontext ctx;
static CUfunction f_add, f_ln, f_mm, f_qkt, f_sm, f_gelu;
static CUfunction f_ceg, f_atb, f_abt, f_geb, f_smb, f_lnbx, f_lnbg, f_scale, f_adam;
static CUdeviceptr dbc1, dbc2;
static int Si = S, Di = D, Hi = H, Vi = V, SDi = SD, SHi = SH;

static uint32_t xs_state = 0x12345678u;
static uint32_t xs(void) { uint32_t x = xs_state; x ^= x << 13; x ^= x >> 17; x ^= x << 5; xs_state = x; return x; }
static float rnd(float scale) { return ((float)(int32_t)xs() / 2147483648.0f) * scale; }

#define LX(fn, grid, block, args) do { CKX(cuLaunchKernel((fn), (grid),1,1, (block),1,1, 0,0, (args), 0), #fn); CKX(cuCtxSynchronize(), "sync " #fn); } while (0)

/* weights */
static CUdeviceptr Wq[NL], Wk[NL], Wv[NL], Wo[NL], LN1g[NL], LN1b[NL], LN2g[NL], LN2b[NL], W1[NL], W2[NL], LNfg, LNfb, W_lm;
/* saved activations */
static CUdeviceptr xn1[NL], Qb[NL], Kb[NL], Vb[NL], scores[NL], attn[NL], ao[NL], proj[NL], h1[NL], xn2[NL], amlp[NL], gmlp[NL], mmlp[NL], h2[NL], ist1[NL], ist2[NL];
static CUdeviceptr xf, istf, logits, x_in, targets_f;
/* weight gradients */
static CUdeviceptr dWq[NL], dWk[NL], dWv[NL], dWo[NL], dLN1[NL], dLN2[NL], dW1[NL], dW2[NL], dLNf, dW_lm;
/* gradient temporaries */
static CUdeviceptr g_dlog, g_dxf, g_dh2[NL], g_dh1, g_dg, g_da, g_dxn2, g_dxln2, g_dao, g_dVv, g_dQ, g_dK, g_dattn, g_dsc, g_dxn1, g_dxln1, g_t1, g_t2, g_t3, g_t12, g_dxscr;

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
    void* a1[] = { &x, &xn1[L], &LN1g[L], &LN1b[L], &ist1[L], &Di }; LX(f_ln, S, 1, a1);
    void* aq[] = { &xn1[L], &Wq[L], &Qb[L], &Si, &Di, &Di }; LX(f_mm, S, D, aq);
    void* ak[] = { &xn1[L], &Wk[L], &Kb[L], &Si, &Di, &Di }; LX(f_mm, S, D, ak);
    void* av[] = { &xn1[L], &Wv[L], &Vb[L], &Si, &Di, &Di }; LX(f_mm, S, D, av);
    void* aqk[] = { &Qb[L], &Kb[L], &scores[L], &Si, &Di }; LX(f_qkt, S, S, aqk);
    void* asm_[] = { &scores[L], &attn[L], &Si, &Si }; LX(f_sm, S, 1, asm_);
    void* aao[] = { &attn[L], &Vb[L], &ao[L], &Si, &Si, &Di }; LX(f_mm, S, D, aao);
    void* apr[] = { &ao[L], &Wo[L], &proj[L], &Si, &Di, &Di }; LX(f_mm, S, D, apr);
    void* ah1[] = { &x, &proj[L], &h1[L], &SDi }; LX(f_add, SD, 1, ah1);
    void* a2[] = { &h1[L], &xn2[L], &LN2g[L], &LN2b[L], &ist2[L], &Di }; LX(f_ln, S, 1, a2);
    void* aa[] = { &xn2[L], &W1[L], &amlp[L], &Si, &Di, &Hi }; LX(f_mm, S, H, aa);
    void* ag[] = { &amlp[L], &gmlp[L], &SHi }; LX(f_gelu, SH, 1, ag);
    void* am[] = { &gmlp[L], &W2[L], &mmlp[L], &Si, &Hi, &Di }; LX(f_mm, S, D, am);
    void* ah2[] = { &h1[L], &mmlp[L], &h2[L], &SDi }; LX(f_add, SD, 1, ah2);
}

static double ce_loss(const float* hlog, const int* tgt) {
    /* SUM (not mean) cross-entropy: this makes the loss the antiderivative of the
     * gpu_ce_softmax_grad gradient (softmax-onehot, unaveraged), so the GPU backward
     * matches the finite-diff of this loss. The oracle sums too; the within-2% relative
     * comparison is identical to the mean convention (the S factor cancels). Accumulated
     * in DOUBLE so the finite-difference Lp-Lm (a ~1e-4 difference of two ~62 sums) does
     * not lose precision to f32 cancellation roundoff (~3e-3 at this loss magnitude). */
    double loss = 0.0;
    for (int s = 0; s < S; s++) { const float* r = &hlog[s * V]; float mx = r[0]; for (int c = 1; c < V; c++) if (r[c] > mx) mx = r[c]; double sm = 0.0; for (int c = 0; c < V; c++) sm += exp((double)(r[c] - mx)); loss += -log(exp((double)(r[tgt[s]] - mx)) / sm); }
    return loss;
}

static int g_tgt[S];
static double forward_full(void) {
    forward_layer(0, x_in); forward_layer(1, h2[0]);
    void* alf[] = { &h2[1], &xf, &LNfg, &LNfb, &istf, &Di }; LX(f_ln, S, 1, alf);
    void* alm[] = { &xf, &W_lm, &logits, &Si, &Di, &Vi }; LX(f_mm, S, V, alm);
    float hlog[SV]; CKX(cuMemcpyDtoH(hlog, logits, sizeof(hlog)), "d2h logits");
    return ce_loss(hlog, g_tgt);
}

/* backward one layer: dh2_in = grad into this layer's output h2[L]; writes weight grads
 * + the layer-input grad into dx_out. */
static void backward_layer(int L, CUdeviceptr dh2_in, CUdeviceptr dx_out) {
    /* MLP: dm = dh2_in (residual h2=h1+m) */
    void* aw2[] = { &gmlp[L], &dh2_in, &dW2[L], &Si, &Hi, &Di }; LX(f_atb, H, D, aw2);
    void* ag[]  = { &dh2_in, &W2[L], &g_dg, &Si, &Di, &Hi };     LX(f_abt, S, H, ag);
    void* ada[] = { &amlp[L], &g_dg, &g_da, &SHi };              LX(f_geb, SH, 1, ada);
    void* aw1[] = { &xn2[L], &g_da, &dW1[L], &Si, &Di, &Hi };    LX(f_atb, D, H, aw1);
    void* axn2[]= { &g_da, &W1[L], &g_dxn2, &Si, &Hi, &Di };     LX(f_abt, S, D, axn2);
    /* LN2 bwd */
    void* al2x[]= { &h1[L], &g_dxn2, &LN2g[L], &ist2[L], &g_dxln2, &Di }; LX(f_lnbx, S, 1, al2x);
    void* al2g[]= { &h1[L], &g_dxn2, &ist2[L], &dLN2[L], &Si, &Di };      LX(f_lnbg, D, 1, al2g);
    /* dh1 = dx_ln2 + dh2_in (residual) */
    void* adh1[]= { &g_dxln2, &dh2_in, &g_dh1, &SDi }; LX(f_add, SD, 1, adh1);
    /* attention proj: dproj = dh1 */
    void* awo[] = { &ao[L], &g_dh1, &dWo[L], &Si, &Di, &Di }; LX(f_atb, D, D, awo);
    void* adao[]= { &g_dh1, &Wo[L], &g_dao, &Si, &Di, &Di };  LX(f_abt, S, D, adao);
    /* attention core: dVv, dattn, dscores, dQ, dK */
    void* advv[]= { &attn[L], &g_dao, &g_dVv, &Si, &Si, &Di }; LX(f_atb, S, D, advv);
    void* adat[]= { &g_dao, &Vb[L], &g_dattn, &Si, &Di, &Si }; LX(f_abt, S, S, adat);
    void* adsc[]= { &attn[L], &g_dattn, &g_dsc, &Si, &Si };    LX(f_smb, S, 1, adsc);
    void* adq[] = { &g_dsc, &Kb[L], &g_dQ, &Si, &Si, &Di };    LX(f_mm, S, D, adq);
    void* asq[] = { &g_dQ, &SDi };                            LX(f_scale, SD, 1, asq);
    void* adk[] = { &g_dsc, &Qb[L], &g_dK, &Si, &Si, &Di };    LX(f_atb, S, D, adk);
    void* ask[] = { &g_dK, &SDi };                            LX(f_scale, SD, 1, ask);
    /* QKV weight grads */
    void* awq[] = { &xn1[L], &g_dQ, &dWq[L], &Si, &Di, &Di };  LX(f_atb, D, D, awq);
    void* awk[] = { &xn1[L], &g_dK, &dWk[L], &Si, &Di, &Di };  LX(f_atb, D, D, awk);
    void* awv[] = { &xn1[L], &g_dVv, &dWv[L], &Si, &Di, &Di }; LX(f_atb, D, D, awv);
    /* dxn1 = dQ@Wq^T + dK@Wk^T + dVv@Wv^T */
    void* at1[] = { &g_dQ, &Wq[L], &g_t1, &Si, &Di, &Di };  LX(f_abt, S, D, at1);
    void* at2[] = { &g_dK, &Wk[L], &g_t2, &Si, &Di, &Di };  LX(f_abt, S, D, at2);
    void* at3[] = { &g_dVv, &Wv[L], &g_t3, &Si, &Di, &Di }; LX(f_abt, S, D, at3);
    void* a12[] = { &g_t1, &g_t2, &g_t12, &SDi };  LX(f_add, SD, 1, a12);
    void* a123[]= { &g_t12, &g_t3, &g_dxn1, &SDi }; LX(f_add, SD, 1, a123);
    /* LN1 bwd */
    CUdeviceptr xL = (L == 0) ? x_in : h2[0];
    void* al1x[]= { &xL, &g_dxn1, &LN1g[L], &ist1[L], &g_dxln1, &Di }; LX(f_lnbx, S, 1, al1x);
    void* al1g[]= { &xL, &g_dxn1, &ist1[L], &dLN1[L], &Si, &Di };      LX(f_lnbg, D, 1, al1g);
    /* dx = dx_ln1 + dh1 (residual) */
    void* adx[] = { &g_dxln1, &g_dh1, &dx_out, &SDi }; LX(f_add, SD, 1, adx);
}

static void backward_full(void) {
    void* ace[] = { &logits, &targets_f, &g_dlog, &Si, &Vi }; LX(f_ceg, S, 1, ace);
    void* awlm[]= { &xf, &g_dlog, &dW_lm, &Si, &Di, &Vi };    LX(f_atb, D, V, awlm);
    void* adxf[]= { &g_dlog, &W_lm, &g_dxf, &Si, &Vi, &Di };  LX(f_abt, S, D, adxf);
    void* alfx[]= { &h2[1], &g_dxf, &LNfg, &istf, &g_dh2[1], &Di }; LX(f_lnbx, S, 1, alfx);
    void* alfg[]= { &h2[1], &g_dxf, &istf, &dLNf, &Si, &Di };       LX(f_lnbg, D, 1, alfg);
    backward_layer(1, g_dh2[1], g_dh2[0]);
    backward_layer(0, g_dh2[0], g_dxscr);
}

/* Adam parameter table: every trainable tensor + its grad + (m,v) moments. The LN
 * gamma/beta share a packed dgb grad buffer (dgamma=dgb[0:D], dbeta=dgb[D:2D]) so beta
 * uses a byte-offset into the same grad buffer. */
typedef struct { CUdeviceptr w, g, m, v; int n; } Param;
static Param params[80]; static int nparams = 0;
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
    for (int i = 0; i < nparams; i++) { Param* p = &params[i]; void* a[] = { &p->w, &p->g, &p->m, &p->v, &dbc1, &dbc2 }; LX(f_adam, p->n, 1, a); }
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <combined.ptx>\n", argv[0]); return 2; }
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
    for (int L = 0; L < NL; L++) {
        Wq[L]=A(DD); Wk[L]=A(DD); Wv[L]=A(DD); Wo[L]=A(DD); LN1g[L]=A(D); LN1b[L]=A(D); LN2g[L]=A(D); LN2b[L]=A(D); W1[L]=A(DH); W2[L]=A(HD);
        xn1[L]=A(SD); Qb[L]=A(SD); Kb[L]=A(SD); Vb[L]=A(SD); scores[L]=A(SS); attn[L]=A(SS); ao[L]=A(SD); proj[L]=A(SD); h1[L]=A(SD); xn2[L]=A(SD); amlp[L]=A(SH); gmlp[L]=A(SH); mmlp[L]=A(SD); h2[L]=A(SD); ist1[L]=A(S); ist2[L]=A(S);
        dWq[L]=A(DD); dWk[L]=A(DD); dWv[L]=A(DD); dWo[L]=A(DD); dLN1[L]=A(2*D); dLN2[L]=A(2*D); dW1[L]=A(DH); dW2[L]=A(HD); g_dh2[L]=A(SD);
    }
    LNfg=A(D); LNfb=A(D); W_lm=A(DV); xf=A(SD); istf=A(S); logits=A(SV); x_in=A(SD); targets_f=A(S); dLNf=A(2*D); dW_lm=A(DV);
    g_dlog=A(SV); g_dxf=A(SD); g_dh1=A(SD); g_dg=A(SH); g_da=A(SH); g_dxn2=A(SD); g_dxln2=A(SD); g_dao=A(SD); g_dVv=A(SD); g_dQ=A(SD); g_dK=A(SD); g_dattn=A(SS); g_dsc=A(SS); g_dxn1=A(SD); g_dxln1=A(SD); g_t1=A(SD); g_t2=A(SD); g_t3=A(SD); g_t12=A(SD); g_dxscr=A(SD);

    float* hw = (float*)malloc(NW * sizeof(float));
    gen_weights(hw);
    FILE* wf = fopen("init_weights.bin", "wb"); if (wf) { fwrite(hw, sizeof(float), NW, wf); fclose(wf); }
    upload_weights(hw);
    float hx[SD]; memset(hx, 0, sizeof(hx)); for (int s = 0; s < S; s++) hx[s * D + (s % D)] = 1.0f;
    CK(cuMemcpyHtoD(x_in, hx, sizeof(hx)), "h2d x");
    float htf[S]; for (int s = 0; s < S; s++) { g_tgt[s] = (s + 1) % S; htf[s] = (float)g_tgt[s]; }
    CK(cuMemcpyHtoD(targets_f, htf, sizeof(htf)), "h2d tgt");

    double loss0 = forward_full();
    printf("GPU [%s] step0 loss = %.6f (sum CE; mean=%.4f, log V=%.4f)\n", gpu, loss0, loss0 / (double)S, logf((float)V));
    dbc1 = A(1); dbc2 = A(1);
    build_params();
    if (argc > 2 && strcmp(argv[2], "verify") == 0) {
    backward_full();

    /* finite-diff verify: spot-check a few weights across both layers. weight host offsets: */
    int layer_block = 4*DD + 4*D + DH + HD;  /* 3136 */
    int off_Wlm = NL*layer_block + 2*D;       /* W_lm start */
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
    /* TRAIN: K=500 Adam steps (forward -> backward -> Adam per weight). */
    int K = 500;
    FILE* cf = fopen("loss_curve.csv", "wb");
    double loss = loss0;
    for (int t = 1; t <= K; t++) {
        loss = forward_full();
        backward_full();
        adam_step(t);
        if (t == 1 || t % 25 == 0 || t == K) { printf("  step %4d loss %.6f (mean %.5f)\n", t, loss, loss / (double)S); if (cf) fprintf(cf, "%d,%.8f\n", t, loss); }
    }
    double lf = forward_full();
    if (cf) { fprintf(cf, "%d,%.8f\n", K + 1, lf); fclose(cf); }
    printf("GPU [%s] train K=%d: start loss %.6f -> final %.6f (mean %.5f)\n", gpu, K, loss0, lf, lf / (double)S);
    return 0;
}
