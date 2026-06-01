/* train_transformer.c -- Helix v1.0 CAPSTONE training harness (Option A: trusted C
 * launcher; ALL math is kovc-emitted PTX, the C only does cuMemAlloc/cuMemcpy/cuLaunchKernel
 * sequencing + scalar bookkeeping). A tiny 2-layer pre-norm transformer trains end-to-end
 * on the GPU; the loss curve is compared to a numpy oracle (oracle_train.py) for the
 * within-2% capstone. See docs/HELIX_CAPSTONE_TRAIN_PLAN.md.
 *
 * STAGE A (this file currently): forward only -> step-0 cross-entropy loss (sanity ~log(V)).
 * Build (WSL): gcc train_transformer.c -O2 -o /tmp/train -lcuda -lm -L/usr/lib/wsl/lib
 * Run:        /tmp/train <combined.ptx>   (combined PTX = the needed @kernel files concatenated + emitted)
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
#define SD (S*D)   /* 256 */
#define SS (S*S)   /* 256 */
#define SH (S*H)   /* 1024 */
#define SV (S*V)   /* 512 */
#define DD (D*D)   /* 256 */
#define DH (D*H)   /* 1024 */
#define HD (H*D)   /* 1024 */
#define DV (D*V)   /* 512 */
#define NW (NL*(4*DD + 4*D + DH + HD) + 2*D + DV)  /* total weight floats = 6816 */

static int check(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) { const char* m = 0; cuGetErrorString(r, &m); fprintf(stderr, "CUDA %s: %s (%d)\n", what, m ? m : "?", (int)r); return 1; }
    return 0;
}
#define CK(c, w) do { if (check((c), (w))) return 2; } while (0)

static CUcontext ctx;
static CUfunction f_add, f_ln, f_mm, f_qkt, f_sm, f_gelu;
static int Si = S, Di = D, Hi = H, Vi = V, SDi = SD, SHi = SH;

/* xorshift32 -> f32 in [-1,1] then scaled. */
static uint32_t xs_state = 0x12345678u;
static uint32_t xs(void) { uint32_t x = xs_state; x ^= x << 13; x ^= x >> 17; x ^= x << 5; xs_state = x; return x; }
static float rnd(float scale) { return ((float)(int32_t)xs() / 2147483648.0f) * scale; }

#define LAUNCH(fn, grid, block, args) do { CK(cuLaunchKernel((fn), (grid),1,1, (block),1,1, 0,0, (args), 0), #fn); CK(cuCtxSynchronize(), "sync " #fn); } while (0)

/* device weight buffers */
static CUdeviceptr Wq[NL], Wk[NL], Wv[NL], Wo[NL], LN1g[NL], LN1b[NL], LN2g[NL], LN2b[NL], W1[NL], W2[NL];
static CUdeviceptr LNfg, LNfb, W_lm;
/* device activation buffers (saved for backward) */
static CUdeviceptr xn1[NL], Qb[NL], Kb[NL], Vb[NL], scores[NL], attn[NL], ao[NL], proj[NL], h1[NL], xn2[NL], amlp[NL], gmlp[NL], mmlp[NL], h2[NL], ist1[NL], ist2[NL];
static CUdeviceptr xf, istf, logits, x_in, targets_f;

static CUdeviceptr A(int nf) { CUdeviceptr p; if (cuMemAlloc(&p, (size_t)nf * sizeof(float)) != CUDA_SUCCESS) { fprintf(stderr, "alloc fail %d\n", nf); exit(2); } return p; }

/* generate the weight blob in the init_weights.bin order, return float count. */
static void gen_weights(float* w) {
    int o = 0;
    float sc_d = sqrtf(2.0f / (float)D), sc_h = sqrtf(2.0f / (float)H);
    for (int L = 0; L < NL; L++) {
        for (int i = 0; i < DD; i++) w[o++] = rnd(sc_d);  /* Wq */
        for (int i = 0; i < DD; i++) w[o++] = rnd(sc_d);  /* Wk */
        for (int i = 0; i < DD; i++) w[o++] = rnd(sc_d);  /* Wv */
        for (int i = 0; i < DD; i++) w[o++] = rnd(sc_d);  /* Wo */
        for (int i = 0; i < D; i++) w[o++] = 1.0f;        /* LN1 gamma */
        for (int i = 0; i < D; i++) w[o++] = 0.0f;        /* LN1 beta */
        for (int i = 0; i < D; i++) w[o++] = 1.0f;        /* LN2 gamma */
        for (int i = 0; i < D; i++) w[o++] = 0.0f;        /* LN2 beta */
        for (int i = 0; i < DH; i++) w[o++] = rnd(sc_d);  /* W1 [d,H] */
        for (int i = 0; i < HD; i++) w[o++] = rnd(sc_h);  /* W2 [H,d] */
    }
    for (int i = 0; i < D; i++) w[o++] = 1.0f;            /* LNf gamma */
    for (int i = 0; i < D; i++) w[o++] = 0.0f;            /* LNf beta */
    for (int i = 0; i < DV; i++) w[o++] = rnd(sc_d);      /* W_lm [d,V] */
}

/* H2D the weight blob into the device buffers in the same order. */
static int upload_weights(const float* w) {
    int o = 0;
#define UP(dst, nf) do { CK(cuMemcpyHtoD((dst), &w[o], (size_t)(nf)*sizeof(float)), "h2d w"); o += (nf); } while (0)
    for (int L = 0; L < NL; L++) {
        UP(Wq[L], DD); UP(Wk[L], DD); UP(Wv[L], DD); UP(Wo[L], DD);
        UP(LN1g[L], D); UP(LN1b[L], D); UP(LN2g[L], D); UP(LN2b[L], D);
        UP(W1[L], DH); UP(W2[L], HD);
    }
    UP(LNfg, D); UP(LNfb, D); UP(W_lm, DV);
#undef UP
    return 0;
}

/* forward one layer; x is the [S,d] input device buffer. Writes layer-L activations. */
static int forward_layer(int L, CUdeviceptr x) {
    void* a_ln1[] = { &x, &xn1[L], &LN1g[L], &LN1b[L], &ist1[L], &Di };
    LAUNCH(f_ln, S, 1, a_ln1);
    void* a_q[] = { &xn1[L], &Wq[L], &Qb[L], &Si, &Di, &Di }; LAUNCH(f_mm, S, D, a_q);
    void* a_k[] = { &xn1[L], &Wk[L], &Kb[L], &Si, &Di, &Di }; LAUNCH(f_mm, S, D, a_k);
    void* a_v[] = { &xn1[L], &Wv[L], &Vb[L], &Si, &Di, &Di }; LAUNCH(f_mm, S, D, a_v);
    void* a_qkt[] = { &Qb[L], &Kb[L], &scores[L], &Si, &Di }; LAUNCH(f_qkt, S, S, a_qkt);
    void* a_sm[] = { &scores[L], &attn[L], &Si, &Si }; LAUNCH(f_sm, S, 1, a_sm);
    void* a_ao[] = { &attn[L], &Vb[L], &ao[L], &Si, &Si, &Di }; LAUNCH(f_mm, S, D, a_ao);
    void* a_pr[] = { &ao[L], &Wo[L], &proj[L], &Si, &Di, &Di }; LAUNCH(f_mm, S, D, a_pr);
    void* a_h1[] = { &x, &proj[L], &h1[L], &SDi }; LAUNCH(f_add, SD, 1, a_h1);
    void* a_ln2[] = { &h1[L], &xn2[L], &LN2g[L], &LN2b[L], &ist2[L], &Di }; LAUNCH(f_ln, S, 1, a_ln2);
    void* a_a[] = { &xn2[L], &W1[L], &amlp[L], &Si, &Di, &Hi }; LAUNCH(f_mm, S, H, a_a);
    void* a_g[] = { &amlp[L], &gmlp[L], &SHi }; LAUNCH(f_gelu, SH, 1, a_g);
    void* a_m[] = { &gmlp[L], &W2[L], &mmlp[L], &Si, &Hi, &Di }; LAUNCH(f_mm, S, D, a_m);
    void* a_h2[] = { &h1[L], &mmlp[L], &h2[L], &SDi }; LAUNCH(f_add, SD, 1, a_h2);
    return 0;
}

static float ce_loss(const float* hlog, const int* tgt) {
    float loss = 0.0f;
    for (int s = 0; s < S; s++) {
        const float* r = &hlog[s * V];
        float mx = r[0]; for (int c = 1; c < V; c++) if (r[c] > mx) mx = r[c];
        float sm = 0.0f; for (int c = 0; c < V; c++) sm += expf(r[c] - mx);
        float p = expf(r[tgt[s]] - mx) / sm;
        loss += -logf(p);
    }
    return loss / (float)S;
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <combined.ptx>\n", argv[0]); return 2; }
    /* slurp ptx */
    FILE* f = fopen(argv[1], "rb"); if (!f) { fprintf(stderr, "open ptx\n"); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    char* ptx = (char*)malloc(sz + 1); if (fread(ptx, 1, sz, f) != (size_t)sz) return 2; ptx[sz] = 0; fclose(f);
    CK(cuInit(0), "init");
    CUdevice dev; CK(cuDeviceGet(&dev, 0), "dev");
    char gpu[256]; gpu[0] = 0; cuDeviceGetName(gpu, 256, dev);
    CK(cuCtxCreate(&ctx, 0, dev), "ctx");
    CUmodule mod; CK(cuModuleLoadData(&mod, ptx), "load");
    CK(cuModuleGetFunction(&f_add, mod, "vector_add"), "f_add");
    CK(cuModuleGetFunction(&f_ln, mod, "gpu_layernorm_fwd_save"), "f_ln");
    CK(cuModuleGetFunction(&f_mm, mod, "naive_matmul"), "f_mm");
    CK(cuModuleGetFunction(&f_qkt, mod, "gpu_qkt"), "f_qkt");
    CK(cuModuleGetFunction(&f_sm, mod, "gpu_softmax"), "f_sm");
    CK(cuModuleGetFunction(&f_gelu, mod, "gpu_gelu"), "f_gelu");
    /* alloc */
    for (int L = 0; L < NL; L++) {
        Wq[L]=A(DD); Wk[L]=A(DD); Wv[L]=A(DD); Wo[L]=A(DD);
        LN1g[L]=A(D); LN1b[L]=A(D); LN2g[L]=A(D); LN2b[L]=A(D); W1[L]=A(DH); W2[L]=A(HD);
        xn1[L]=A(SD); Qb[L]=A(SD); Kb[L]=A(SD); Vb[L]=A(SD); scores[L]=A(SS); attn[L]=A(SS);
        ao[L]=A(SD); proj[L]=A(SD); h1[L]=A(SD); xn2[L]=A(SD); amlp[L]=A(SH); gmlp[L]=A(SH);
        mmlp[L]=A(SD); h2[L]=A(SD); ist1[L]=A(S); ist2[L]=A(S);
    }
    LNfg=A(D); LNfb=A(D); W_lm=A(DV); xf=A(SD); istf=A(S); logits=A(SV); x_in=A(SD); targets_f=A(S);
    /* init weights + write init_weights.bin + upload */
    float* hw = (float*)malloc(NW * sizeof(float));
    gen_weights(hw);
    FILE* wf = fopen("init_weights.bin", "wb"); if (wf) { fwrite(hw, sizeof(float), NW, wf); fclose(wf); }
    CK(upload_weights(hw), "upload");
    /* one-hot identity input (S=d=16), targets (s+1)%S */
    float hx[SD]; memset(hx, 0, sizeof(hx)); for (int s = 0; s < S; s++) hx[s * D + (s % D)] = 1.0f;
    CK(cuMemcpyHtoD(x_in, hx, sizeof(hx)), "h2d x");
    int tgt[S]; float htf[S]; for (int s = 0; s < S; s++) { tgt[s] = (s + 1) % S; htf[s] = (float)tgt[s]; }
    CK(cuMemcpyHtoD(targets_f, htf, sizeof(htf)), "h2d tgt");
    /* FORWARD */
    CK(forward_layer(0, x_in), "L0");
    CK(forward_layer(1, h2[0]), "L1");
    void* a_lnf[] = { &h2[1], &xf, &LNfg, &LNfb, &istf, &Di }; LAUNCH(f_ln, S, 1, a_lnf);
    void* a_lm[] = { &xf, &W_lm, &logits, &Si, &Di, &Vi }; LAUNCH(f_mm, S, V, a_lm);
    float hlog[SV]; CK(cuMemcpyDtoH(hlog, logits, sizeof(hlog)), "d2h logits");
    float loss = ce_loss(hlog, tgt);
    printf("GPU [%s] train_transformer STAGE A forward: step0 loss = %.6f (sanity log(V)=%.4f) logits[0,0]=%.5f\n",
           gpu, loss, logf((float)V), hlog[0]);
    return 0;
}
