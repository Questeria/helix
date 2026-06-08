/* gpt2_infer.c -- GPT-2 124M FORWARD-ONLY inference launcher (P5).
 *
 * A FORK of helixc/runtime/train_transformer.c (the v1.0/v1.3 CAPSTONE) reduced to a
 * forward pass and reshaped for GPT-2 124M. Like the capstone it is a trusted-C launcher:
 * ALL arithmetic stays in kovc-emitted PTX kernels; this C only moves bytes (mmap the P1
 * weight file, host embedding gather, head-slice pack/scatter, H<->D copies) and sequences
 * kernel launches. train_transformer.c is NOT modified (its CAPSTONE_AUDIT_PASS stays
 * byte-identical); this sibling shares the kernel corpus + the cuModuleLoadData/cuLaunchKernel
 * infra.
 *
 * THIS BUILD targets the P5 gate-2 anchor: run GPT-2 BLOCK 0 on the GPU through the kovc PTX
 * kernels for the canonical prompt ids [464,3139,286,4881,318] (T=5), dump the post-block-0
 * hidden [T,768], and compare to helix-llm/ref/ref_block0.npy at max-abs-rel < 1e-3. Block-0
 * parity proves embedding + causal mask + eps-LN + multi-head + bias + GEMM orientation at once.
 *
 * Kernel selection (per-op, the fork picks its own CUfunction handles):
 *   - GEMMs (N in {768,2304,3072}, all > 1024): the TILED OPT kernels tiled_matmul /
 *     tiled_matmul_abt (grid=(N/64,M/64) block=(16,16); the ONLY GEMMs valid at N>1024;
 *     require M%64==N%64==K%8==0 -> S padded to 64, d_model 768 / d_ff 3072 / head_dim 64 fit).
 *   - causal softmax: gpu_softmax_causal (grid=rows block=1, mask folded in).
 *   - eps-LN: gpu_layernorm_fwd_eps (grid=rows block=1, eps=1e-5 affine, biased var).
 *   - bias row-broadcast: gpu_add_bias_rowbcast (1 thread/elem, i<n guard).
 *   - residual: vector_add. GELU: gpu_gelu_stable (= gelu_new, overflow-safe tanh; the committed
 *     gpu_gelu's direct e^(2z) NaNs at GPT-2's ~+/-12 c_fc activations). attn scale: gpu_scale_rt(0.125).
 *
 * Build (WSL): gcc gpt2_infer.c -O2 -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -lm -o /tmp/gpt2_infer
 * Run:         /tmp/gpt2_infer <combined.ptx> <gpt2_124M.weights> --block0 <ref_block0.npy>
 *   -> dumps /tmp/helix_block0.bin (flat <f4 [T,768]) and prints GPT2_BLOCK0_PARITY_PASS/FAIL.
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

/* ---- GPT-2 124M dims (env-overridable, defaulting to the public model) ---- */
static int NL = 12, DM = 768, NH = 12, NV = 50257, NC = 1024, DFF = 3072;
static int DH = 64;          /* head dim = DM/NH */
static int Spad = 64;        /* S padded up to a multiple of 64 for the tiled GEMMs */
static float ATTN_SCALE = 0.125f;  /* 1/sqrt(head_dim)=1/sqrt(64) */

/* ---- P1 weight-file header (must match helix-llm/tools/gpt2_import.py) ---- */
#define MAGIC   0x48584757u   /* 'HXGW' little-endian */
#define VERSION 1u
#define HDR_BYTES 64

static int check(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) { const char* m = 0; cuGetErrorString(r, &m); fprintf(stderr, "CUDA %s: %s (%d)\n", what, m ? m : "?", (int)r); return 1; }
    return 0;
}
#define CK(c, w)  do { if (check((c), (w))) return 2; } while (0)
#define CKX(c, w) do { if (check((c), (w))) exit(2); } while (0)

static CUcontext ctx;
/* the forward-only kernel handles (fork picks per-op) */
static CUfunction f_mm_t, f_abt_t, f_ln_eps, f_sm_causal, f_bias, f_gelu, f_add, f_scale_rt;

#define SYNC(w) do { CKX(cuCtxSynchronize(), w); } while (0)
#define LX(fn, grid, block, args)        do { CKX(cuLaunchKernel((fn),(grid),1,1,(block),1,1,0,0,(args),0), #fn); SYNC("sync " #fn); } while (0)
#define LX2(fn, gx, gy, bx, by, args)    do { CKX(cuLaunchKernel((fn),(gx),(gy),1,(bx),(by),1,0,0,(args),0), #fn); SYNC("sync " #fn); } while (0)

/* C[M,N] = A[M,K] @ B[K,N]  via the SMEM-tiled GEMM. grid=(N/64,M/64) block=(16,16).
 * M%64==N%64==K%8==0 enforced by the caller's padding (Spad=64; DM=768,DFF=3072,DH=64). */
static void mm_AB(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, int M, int Kc, int N) {
    int m=M,k=Kc,n=N; void* ar[] = { &a,&b,&c,&m,&k,&n };
    LX2(f_mm_t, (unsigned)(N/64), (unsigned)(M/64), 16, 16, ar);
}
/* C[M,N] = A[M,K] @ B[N,K]^T  via the SMEM-tiled A@B^T GEMM. Same geometry. */
static void mm_ABt(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, int M, int Kc, int N) {
    int m=M,k=Kc,n=N; void* ar[] = { &a,&b,&c,&m,&k,&n };
    LX2(f_abt_t, (unsigned)(N/64), (unsigned)(M/64), 16, 16, ar);
}
/* affine LayerNorm with eps=1e-5 over [rows,cols]. grid=rows block=1. */
static void ln_eps(CUdeviceptr x, CUdeviceptr y, CUdeviceptr g, CUdeviceptr b, int rows, int cols) {
    int c=cols; void* ar[] = { &x,&y,&g,&b,&c }; LX(f_ln_eps, (unsigned)rows, 1, ar);
}
/* causal row-softmax over [rows,cols]. grid=rows block=1. */
static void softmax_causal(CUdeviceptr x, CUdeviceptr y, int rows, int cols) {
    int r=rows,c=cols; void* ar[] = { &x,&y,&r,&c }; LX(f_sm_causal, (unsigned)rows, 1, ar);
}
/* y[i] += bias[i mod cols] over n elems. grid=ceil(n/256) block=256 (i<n guard in-kernel). */
static void add_bias(CUdeviceptr y, CUdeviceptr bias, int n, int cols) {
    int ni=n,ci=cols; void* ar[] = { &y,&bias,&ni,&ci };
    int tpb=256, bpg=(n+tpb-1)/tpb; LX(f_bias, (unsigned)bpg, (unsigned)tpb, ar);
}
/* in-place scale a[i]*=s[0] over n elems. The kernel is grid-stride with NO guard, so
 * blocks*threads must == n EXACTLY; pick the largest block in {256,128,64,32,1} dividing n. */
static void scale_rt(CUdeviceptr a, CUdeviceptr s, int n) {
    int nn=n; void* ar[] = { &a,&s,&nn };
    int blk=1; int cand[]={256,128,64,32,1}; for (int i=0;i<5;i++) if (n%cand[i]==0){blk=cand[i];break;}
    LX(f_scale_rt, (unsigned)(n/blk), (unsigned)blk, ar);
}
/* elementwise c[i]=a[i]+b[i] over n elems (residual). grid-stride, NO guard -> exact tiling. */
static void vadd(CUdeviceptr a, CUdeviceptr b, CUdeviceptr c, int n) {
    int nn=n; void* ar[] = { &a,&b,&c,&nn };
    int blk=1; int cand[]={256,128,64,32,1}; for (int i=0;i<5;i++) if (n%cand[i]==0){blk=cand[i];break;}
    LX(f_add, (unsigned)(n/blk), (unsigned)blk, ar);
}
/* GELU y[i]=gelu_new(x[i]) over n elems. grid-stride, NO guard -> exact tiling. */
static void gelu(CUdeviceptr x, CUdeviceptr y, int n) {
    int nn=n; void* ar[] = { &x,&y,&nn };
    int blk=1; int cand[]={256,128,64,32,1}; for (int i=0;i<5;i++) if (n%cand[i]==0){blk=cand[i];break;}
    LX(f_gelu, (unsigned)(n/blk), (unsigned)blk, ar);
}

static CUdeviceptr A(size_t nf) { CUdeviceptr p; CKX(cuMemAlloc(&p, nf * sizeof(float)), "alloc"); return p; }

/* ===================== P1 weight file (mmap + per-tensor offsets) ===================== */
/* The host owns the weight mmap and streams tensors to the device -- the authorized
 * "fenced host glue" byte-movement residual (NO arithmetic here). */
static const float* g_wbase = NULL;   /* mmap'd payload base (float*, at file offset 64) */
static size_t g_nfloat = 0;
static int    g_fd = -1;
static void*  g_map = NULL;
static size_t g_maplen = 0;

/* per-layer tensor float-counts, in the exact P1 build_order(). */
static long off_layer(int L);      /* float offset of layer L's first tensor (ln_1.g) */
static long off_globals(void);     /* float offset of the globals block (wte) */

static long per_layer_floats(void) {
    /* ln_1.g[DM] ln_1.b[DM] cattn.W[DM*3DM] cattn.b[3DM] cproj.W[DM*DM] cproj.b[DM]
       ln_2.g[DM] ln_2.b[DM] cfc.W[DM*DFF] cfc.b[DFF] cproj2.W[DFF*DM] cproj2.b[DM] */
    return (long)DM + DM + (long)DM*3*DM + 3*DM + (long)DM*DM + DM
         + DM + DM + (long)DM*DFF + DFF + (long)DFF*DM + DM;
}
static long off_layer(int L) { return (long)L * per_layer_floats(); }
static long off_globals(void) { return (long)NL * per_layer_floats(); }

/* in-layer float offsets of each tensor relative to off_layer(L). */
typedef struct { long ln1g, ln1b, attW, attb, prjW, prjb, ln2g, ln2b, fcW, fcb, pjW, pjb; } LayerOff;
static LayerOff layer_offsets(void) {
    LayerOff o; long p = 0;
    o.ln1g = p; p += DM;
    o.ln1b = p; p += DM;
    o.attW = p; p += (long)DM*3*DM;
    o.attb = p; p += 3*DM;
    o.prjW = p; p += (long)DM*DM;
    o.prjb = p; p += DM;
    o.ln2g = p; p += DM;
    o.ln2b = p; p += DM;
    o.fcW  = p; p += (long)DM*DFF;
    o.fcb  = p; p += DFF;
    o.pjW  = p; p += (long)DFF*DM;
    o.pjb  = p; p += DM;
    return o;
}
/* globals offsets relative to off_globals(): wte[NV*DM] wpe[NC*DM] ln_f.g[DM] ln_f.b[DM]. */
static long off_wte(void)  { return off_globals(); }
static long off_wpe(void)  { return off_globals() + (long)NV*DM; }
static long off_lnfg(void) { return off_globals() + (long)NV*DM + (long)NC*DM; }
static long off_lnfb(void) { return off_lnfg() + DM; }

static int load_gpt2_weights(const char* path) {
    g_fd = open(path, O_RDONLY);
    if (g_fd < 0) { fprintf(stderr, "open weights '%s': %s\n", path, strerror(errno)); return 2; }
    struct stat st; if (fstat(g_fd, &st) != 0) { fprintf(stderr, "fstat weights\n"); return 2; }
    g_maplen = (size_t)st.st_size;
    g_map = mmap(NULL, g_maplen, PROT_READ, MAP_PRIVATE, g_fd, 0);
    if (g_map == MAP_FAILED) { fprintf(stderr, "mmap weights: %s\n", strerror(errno)); return 2; }
    const unsigned char* hb = (const unsigned char*)g_map;
    uint32_t magic, ver, nl, dm, nh, nv, nc, dff; uint64_t nfloat;
    memcpy(&magic,&hb[0],4); memcpy(&ver,&hb[4],4); memcpy(&nl,&hb[8],4); memcpy(&dm,&hb[12],4);
    memcpy(&nh,&hb[16],4); memcpy(&nv,&hb[20],4); memcpy(&nc,&hb[24],4); memcpy(&dff,&hb[28],4);
    memcpy(&nfloat,&hb[32],8);
    if (magic != MAGIC) { fprintf(stderr, "bad magic 0x%08x (want 0x%08x)\n", magic, MAGIC); return 2; }
    if (ver != VERSION) { fprintf(stderr, "bad version %u\n", ver); return 2; }
    if ((int)nl!=NL || (int)dm!=DM || (int)nh!=NH || (int)nv!=NV || (int)nc!=NC || (int)dff!=DFF) {
        fprintf(stderr, "header dims (nl=%u dm=%u nh=%u nv=%u nc=%u dff=%u) != config\n", nl,dm,nh,nv,nc,dff); return 2;
    }
    size_t want = (size_t)HDR_BYTES + nfloat * 4;
    if (g_maplen < want) { fprintf(stderr, "short weight file: %zu < %zu\n", g_maplen, want); return 2; }
    g_nfloat = (size_t)nfloat;
    /* sanity: the computed float layout must total n_float exactly. */
    size_t expect = (size_t)off_globals() + (size_t)NV*DM + (size_t)NC*DM + DM + DM;
    if (expect != g_nfloat) { fprintf(stderr, "layout %zu floats != header n_float %zu\n", expect, g_nfloat); return 2; }
    g_wbase = (const float*)(hb + HDR_BYTES);
    printf("[wt] mmap %zu B, n_float=%zu, layout verified (per_layer=%ld globals_off=%ld)\n",
           g_maplen, g_nfloat, per_layer_floats(), off_globals());
    return 0;
}

/* upload a host weight slice [foff, foff+nf) into a fresh device buffer; return the ptr. */
static CUdeviceptr up_slice(long foff, size_t nf) {
    CUdeviceptr d = A(nf);
    CKX(cuMemcpyHtoD(d, &g_wbase[foff], nf * sizeof(float)), "h2d wslice");
    return d;
}

/* ===================== per-layer device buffers ===================== */
/* layer-0 weights on device */
static CUdeviceptr d_ln1g, d_ln1b, d_attW, d_attb, d_prjW, d_prjb, d_ln2g, d_ln2b, d_fcW, d_fcb, d_pjW, d_pjb;
/* activations (sized for Spad) */
static CUdeviceptr d_x;        /* residual stream [Spad,DM] */
static CUdeviceptr d_xn;       /* layernorm output [Spad,DM] */
static CUdeviceptr d_qkv;      /* fused QKV [Spad,3*DM] */
static CUdeviceptr d_Qh, d_Kh, d_Vh;   /* packed head slabs [Spad,DH] */
static CUdeviceptr d_scores;   /* [Spad,Spad] */
static CUdeviceptr d_attnw;    /* softmax(scores) [Spad,Spad] */
static CUdeviceptr d_aoh;      /* attn_h @ V_h [Spad,DH] */
static CUdeviceptr d_ctx;      /* merged context [Spad,DM] */
static CUdeviceptr d_proj;     /* c_proj output [Spad,DM] */
static CUdeviceptr d_xn2;      /* ln_2 output [Spad,DM] */
static CUdeviceptr d_mlp1;     /* c_fc output [Spad,DFF] */
static CUdeviceptr d_mlp1g;    /* gelu output [Spad,DFF] */
static CUdeviceptr d_mlp2;     /* mlp c_proj output [Spad,DM] */
static CUdeviceptr d_scale;    /* 1-elem attn scale buffer */
/* full-model (--logits / --generate) extras */
static CUdeviceptr d_lnfg, d_lnfb;   /* ln_f gamma/beta */
static CUdeviceptr d_wte_pad;        /* tied LM head: wte padded [NVpad,DM], rows>=NV zeroed */
static CUdeviceptr d_logits;         /* head output [Spad,NVpad] */
static int NVpad = 0;                /* NV padded up to a multiple of 64 (50257 -> 50304) */
static int Spad_max = 0;             /* largest Spad over a generation run (buffers sized for it) */

/* upload layer L's 12 weight tensors from the mmap into the reused layer-weight device buffers.
 * Byte-movement only (the authorized host glue): the same un-transposed slices the mm_AB engine
 * consumes directly. Streaming per-layer keeps device weight residency to ONE layer (~28 MB) plus
 * the tied head; the full 12-layer set would also fit 8 GB but per-layer streaming is simpler and
 * the H2D cost (7.09 M floats/layer) is negligible against the per-layer GEMM compute. */
static void upload_layer(int L) {
    LayerOff lo = layer_offsets();
    long b = off_layer(L);
    CKX(cuMemcpyHtoD(d_ln1g, &g_wbase[b+lo.ln1g], (size_t)DM*sizeof(float)),        "up ln1g");
    CKX(cuMemcpyHtoD(d_ln1b, &g_wbase[b+lo.ln1b], (size_t)DM*sizeof(float)),        "up ln1b");
    CKX(cuMemcpyHtoD(d_attW, &g_wbase[b+lo.attW], (size_t)DM*3*DM*sizeof(float)),   "up attW");
    CKX(cuMemcpyHtoD(d_attb, &g_wbase[b+lo.attb], (size_t)3*DM*sizeof(float)),      "up attb");
    CKX(cuMemcpyHtoD(d_prjW, &g_wbase[b+lo.prjW], (size_t)DM*DM*sizeof(float)),     "up prjW");
    CKX(cuMemcpyHtoD(d_prjb, &g_wbase[b+lo.prjb], (size_t)DM*sizeof(float)),        "up prjb");
    CKX(cuMemcpyHtoD(d_ln2g, &g_wbase[b+lo.ln2g], (size_t)DM*sizeof(float)),        "up ln2g");
    CKX(cuMemcpyHtoD(d_ln2b, &g_wbase[b+lo.ln2b], (size_t)DM*sizeof(float)),        "up ln2b");
    CKX(cuMemcpyHtoD(d_fcW,  &g_wbase[b+lo.fcW],  (size_t)DM*DFF*sizeof(float)),    "up fcW");
    CKX(cuMemcpyHtoD(d_fcb,  &g_wbase[b+lo.fcb],  (size_t)DFF*sizeof(float)),       "up fcb");
    CKX(cuMemcpyHtoD(d_pjW,  &g_wbase[b+lo.pjW],  (size_t)DFF*DM*sizeof(float)),    "up pjW");
    CKX(cuMemcpyHtoD(d_pjb,  &g_wbase[b+lo.pjb],  (size_t)DM*sizeof(float)),        "up pjb");
}

/* device "gather"/"scatter" of a head's columns:
 * pack: dst[s, 0:DH] = src[s, hbase:hbase+DH] for s in 0..Spad (src has `srccols` columns).
 * Done as a strided DtoD copy per row (byte-movement only; one cuMemcpyDtoD per row). */
static void pack_head(CUdeviceptr dst, CUdeviceptr src, int hbase, int srccols) {
    for (int s = 0; s < Spad; s++) {
        CUdeviceptr s_src = src + (CUdeviceptr)((size_t)(s*srccols + hbase) * sizeof(float));
        CUdeviceptr s_dst = dst + (CUdeviceptr)((size_t)(s*DH) * sizeof(float));
        CKX(cuMemcpyDtoD(s_dst, s_src, (size_t)DH * sizeof(float)), "pack head");
    }
}
static void scatter_head(CUdeviceptr dst, int hbase, CUdeviceptr src) {
    for (int s = 0; s < Spad; s++) {
        CUdeviceptr s_src = src + (CUdeviceptr)((size_t)(s*DH) * sizeof(float));
        CUdeviceptr s_dst = dst + (CUdeviceptr)((size_t)(s*DM + hbase) * sizeof(float));
        CKX(cuMemcpyDtoD(s_dst, s_src, (size_t)DH * sizeof(float)), "scatter head");
    }
}

/* DEBUG: dump a few cells of row `r` of a [*,cols] device buffer (HX_DBG only). */
static int g_dbg = 0;
static void dbg_row(const char* tag, CUdeviceptr d, int r, int cols) {
    if (!g_dbg) return;
    float tmp[8]; int n = cols < 8 ? cols : 8;
    cuMemcpyDtoH(tmp, d + (CUdeviceptr)((size_t)r*cols*sizeof(float)), (size_t)n*sizeof(float));
    fprintf(stderr, "  [dbg] %s row%d:", tag, r);
    for (int i=0;i<n;i++) fprintf(stderr, " %.5g", tmp[i]);
    fprintf(stderr, "\n");
}

/* forward_layer_gpt2(x): x is the [Spad,DM] residual stream (in-place updated).
 *   ln_eps(ln_1) -> mm_AB QKV + bias -> 12-head { pack Q/K/V, mm_ABt scores, scale, causal
 *   softmax, mm_AB @V, scatter } -> mm_AB c_proj + bias -> residual add
 *   -> ln_eps(ln_2) -> mm_AB c_fc + bias -> gelu -> mm_AB mlp c_proj + bias -> residual add. */
static void forward_layer_gpt2(void) {
    /* --- attention --- */
    dbg_row("x_in", d_x, 0, DM);
    ln_eps(d_x, d_xn, d_ln1g, d_ln1b, Spad, DM);
    dbg_row("ln1", d_xn, 0, DM);
    mm_AB(d_xn, d_attW, d_qkv, Spad, DM, 3*DM);
    add_bias(d_qkv, d_attb, Spad*3*DM, 3*DM);
    dbg_row("qkv", d_qkv, 0, 3*DM);
    for (int h = 0; h < NH; h++) {
        int qb = h*DH, kb = DM + h*DH, vb = 2*DM + h*DH;  /* Q@0.., K@DM.., V@2DM.. within the [Spad,3DM] QKV */
        pack_head(d_Qh, d_qkv, qb, 3*DM);
        pack_head(d_Kh, d_qkv, kb, 3*DM);
        pack_head(d_Vh, d_qkv, vb, 3*DM);
        mm_ABt(d_Qh, d_Kh, d_scores, Spad, DH, Spad);   /* scores[Spad,Spad]=Q_h@K_h^T */
        scale_rt(d_scores, d_scale, Spad*Spad);          /* *0.125 */
        softmax_causal(d_scores, d_attnw, Spad, Spad);
        mm_AB(d_attnw, d_Vh, d_aoh, Spad, Spad, DH);     /* ao_h[Spad,DH]=attn@V_h */
        if (g_dbg && h==0) { dbg_row("scores_h0", d_scores, 0, Spad); dbg_row("attnw_h0", d_attnw, 0, Spad); dbg_row("aoh_h0", d_aoh, 0, DH); }
        scatter_head(d_ctx, qb, d_aoh);
    }
    dbg_row("ctx", d_ctx, 0, DM);
    mm_AB(d_ctx, d_prjW, d_proj, Spad, DM, DM);
    add_bias(d_proj, d_prjb, Spad*DM, DM);
    dbg_row("proj", d_proj, 0, DM);
    vadd(d_x, d_proj, d_x, Spad*DM);                     /* x = x + attn_proj (residual) */
    dbg_row("x_after_attn", d_x, 0, DM);
    /* --- MLP --- */
    ln_eps(d_x, d_xn2, d_ln2g, d_ln2b, Spad, DM);
    mm_AB(d_xn2, d_fcW, d_mlp1, Spad, DM, DFF);
    add_bias(d_mlp1, d_fcb, Spad*DFF, DFF);
    gelu(d_mlp1, d_mlp1g, Spad*DFF);
    dbg_row("gelu", d_mlp1g, 0, DFF);
    mm_AB(d_mlp1g, d_pjW, d_mlp2, Spad, DFF, DM);
    dbg_row("mlp2", d_mlp2, 0, DM);
    add_bias(d_mlp2, d_pjb, Spad*DM, DM);
    vadd(d_x, d_mlp2, d_x, Spad*DM);                     /* x = x + mlp (residual) */
    dbg_row("x_final", d_x, 0, DM);
}

/* ===================== full forward (12 layers + ln_f + tied head) ===================== */
/* HOST embedding gather into d_x: x[s,:] = wte[ids[s]] + wpe[s] for s<T, zero pad rows.
 * Byte-movement only (host glue) -- no arithmetic-on-the-trust-path beyond the add the capstone
 * itself does for its input injection. */
static void embed_gather(const int* ids, int T, int S) {
    float* hx = (float*)calloc((size_t)S*DM, sizeof(float));
    const float* wte = &g_wbase[off_wte()];
    const float* wpe = &g_wbase[off_wpe()];
    for (int s = 0; s < T; s++) {
        const float* rw = &wte[(size_t)ids[s]*DM];
        const float* rp = &wpe[(size_t)s*DM];
        for (int c = 0; c < DM; c++) hx[(size_t)s*DM + c] = rw[c] + rp[c];
    }
    CKX(cuMemcpyHtoD(d_x, hx, (size_t)S*DM*sizeof(float)), "h2d x_in (gather)");
    free(hx);
}

/* run the full GPT-2 forward for ids[0..T) and copy the LAST REAL-TOKEN row of logits [NV] into
 * out (host). S = T padded up to a multiple of 64 (set as the global Spad for the layer launches).
 * For greedy decoding only the last real row matters, but we compute the full [S,NVpad] head (the
 * tiled GEMM needs M=S%64==0; reading just row T-1 is free). Pad cols [NV..NVpad) are zeroed in
 * d_wte_pad so their logits are 0 and never argmaxed. All arithmetic is in kovc kernels. */
static void forward_full(const int* ids, int T, float* out_last_logits) {
    int S = ((T + 63) / 64) * 64;
    Spad = S;                                  /* the layer kernels read the global Spad */
    embed_gather(ids, T, S);
    for (int L = 0; L < NL; L++) { upload_layer(L); forward_layer_gpt2(); }
    ln_eps(d_x, d_xn, d_lnfg, d_lnfb, S, DM);  /* final LayerNorm (reuse d_xn as ln_f output) */
    /* tied head: logits[S,NVpad] = xnorm[S,DM] @ wte_pad[NVpad,DM]^T (wte rows ARE the tied head). */
    mm_ABt(d_xn, d_wte_pad, d_logits, S, DM, NVpad);
    /* copy ONLY the last real-token row's first NV logits to the host (greedy decision row). */
    CKX(cuMemcpyDtoH(out_last_logits, d_logits + (CUdeviceptr)((size_t)(T-1)*NVpad*sizeof(float)),
                     (size_t)NV*sizeof(float)), "d2h last-row logits");
}

/* host argmax over a [NV] logit row (host argmax on the final logits is acceptable demo glue). */
static int argmax_row(const float* v, int n) {
    int a = 0; float best = v[0];
    for (int i = 1; i < n; i++) if (v[i] > best) { best = v[i]; a = i; }
    return a;
}

/* ===================== npy reader (minimal, for ref_block0.npy) ===================== */
/* Reads a v1.0 .npy of <f4, C-order, given an expected element count; returns malloc'd floats. */
static float* read_npy_f4(const char* path, size_t expect_elems) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open npy '%s': %s\n", path, strerror(errno)); return NULL; }
    unsigned char magic[8];
    if (fread(magic, 1, 8, f) != 8 || magic[0]!=0x93 || memcmp(magic+1,"NUMPY",5)!=0) { fprintf(stderr,"not a npy\n"); fclose(f); return NULL; }
    unsigned char hl[2]; if (fread(hl,1,2,f)!=2) { fclose(f); return NULL; }
    int header_len = hl[0] | (hl[1]<<8);
    char* hdr = (char*)malloc(header_len+1); if (fread(hdr,1,header_len,f)!=(size_t)header_len) { free(hdr); fclose(f); return NULL; }
    hdr[header_len]=0;
    if (!strstr(hdr,"'<f4'") || strstr(hdr,"'fortran_order': True")) { fprintf(stderr,"npy not <f4 C-order: %s\n",hdr); free(hdr); fclose(f); return NULL; }
    free(hdr);
    float* buf = (float*)malloc(expect_elems * sizeof(float));
    size_t got = fread(buf, sizeof(float), expect_elems, f);
    fclose(f);
    if (got != expect_elems) { fprintf(stderr,"npy short read: %zu/%zu floats\n", got, expect_elems); free(buf); return NULL; }
    return buf;
}

/* ===================== shared device init ===================== */
static CUmodule g_mod;
static char     g_gpu[256];
static char*    g_ptx = NULL;

/* load PTX, create the context, fetch the forward-only kernel handles, mmap the weight file. */
static int device_init(const char* ptx_path, const char* wpath) {
    FILE* pf = fopen(ptx_path, "rb"); if (!pf) { fprintf(stderr, "open ptx '%s'\n", ptx_path); return 2; }
    fseek(pf,0,SEEK_END); long psz=ftell(pf); fseek(pf,0,SEEK_SET);
    g_ptx=(char*)malloc(psz+1); if (fread(g_ptx,1,psz,pf)!=(size_t)psz) { fclose(pf); return 2; } g_ptx[psz]=0; fclose(pf);
    CK(cuInit(0), "init");
    CUdevice dev; CK(cuDeviceGet(&dev,0), "dev");
    g_gpu[0]=0; cuDeviceGetName(g_gpu,256,dev);
    CK(cuCtxCreate(&ctx,0,dev), "ctx");
    CK(cuModuleLoadData(&g_mod, g_ptx), "load ptx");
    CK(cuModuleGetFunction(&f_mm_t,     g_mod, "tiled_matmul"),         "tiled_matmul");
    CK(cuModuleGetFunction(&f_abt_t,    g_mod, "tiled_matmul_abt"),     "tiled_matmul_abt");
    CK(cuModuleGetFunction(&f_ln_eps,   g_mod, "gpu_layernorm_fwd_eps"),"gpu_layernorm_fwd_eps");
    CK(cuModuleGetFunction(&f_sm_causal,g_mod, "gpu_softmax_causal"),   "gpu_softmax_causal");
    CK(cuModuleGetFunction(&f_bias,     g_mod, "gpu_add_bias_rowbcast"),"gpu_add_bias_rowbcast");
    /* NUMERICALLY-STABLE gelu_new: GPT-2's c_fc pre-activation reaches ~+/-12, where the committed
     * gpu_gelu's direct e^(2z) overflows f32 -> NaN. gpu_gelu_stable uses the overflow-safe tanh
     * identity (exp arg always <=0); bit-equal to gelu_new on small x, finite on GPT-2-scale x. */
    CK(cuModuleGetFunction(&f_gelu,     g_mod, "gpu_gelu_stable"),      "gpu_gelu_stable");
    CK(cuModuleGetFunction(&f_add,      g_mod, "vector_add"),           "vector_add");
    CK(cuModuleGetFunction(&f_scale_rt, g_mod, "gpu_scale_rt"),         "gpu_scale_rt");
    if (load_gpt2_weights(wpath)) return 2;
    return 0;
}

/* allocate the layer-weight device buffers (reused across all 12 layers) + the per-Smax activations. */
static int alloc_buffers(int Smax) {
    Spad_max = Smax;
    /* layer-weight buffers (one layer resident at a time; upload_layer() refills them per layer). */
    d_ln1g = A(DM); d_ln1b = A(DM);
    d_attW = A((size_t)DM*3*DM); d_attb = A(3*DM);
    d_prjW = A((size_t)DM*DM);   d_prjb = A(DM);
    d_ln2g = A(DM); d_ln2b = A(DM);
    d_fcW  = A((size_t)DM*DFF);  d_fcb  = A(DFF);
    d_pjW  = A((size_t)DFF*DM);  d_pjb  = A(DM);
    /* activations sized for the largest padded length we will see. */
    d_x      = A((size_t)Smax*DM);
    d_xn     = A((size_t)Smax*DM);
    d_qkv    = A((size_t)Smax*3*DM);
    d_Qh     = A((size_t)Smax*DH);
    d_Kh     = A((size_t)Smax*DH);
    d_Vh     = A((size_t)Smax*DH);
    d_scores = A((size_t)Smax*Smax);
    d_attnw  = A((size_t)Smax*Smax);
    d_aoh    = A((size_t)Smax*DH);
    d_ctx    = A((size_t)Smax*DM);
    d_proj   = A((size_t)Smax*DM);
    d_xn2    = A((size_t)Smax*DM);
    d_mlp1   = A((size_t)Smax*DFF);
    d_mlp1g  = A((size_t)Smax*DFF);
    d_mlp2   = A((size_t)Smax*DM);
    d_scale  = A(1);
    CK(cuMemcpyHtoD(d_scale, &ATTN_SCALE, sizeof(float)), "h2d scale");
    CK(cuMemsetD8(d_ctx, 0, (size_t)Smax*DM*sizeof(float)), "zero ctx");
    return 0;
}

/* set up the tied LM head: ln_f buffers + the padded wte [NVpad,DM] on device + the logits buffer.
 * The tied head is wte itself (logits = x @ wte^T); we copy the NV real rows into a NVpad-row device
 * buffer whose pad rows [NV..NVpad) are zeroed so their logits are 0 and never argmaxed. */
static int setup_head(int Smax) {
    NVpad = ((NV + 63) / 64) * 64;     /* 50257 -> 50304 */
    d_lnfg = up_slice(off_lnfg(), DM);
    d_lnfb = up_slice(off_lnfb(), DM);
    d_wte_pad = A((size_t)NVpad*DM);
    CK(cuMemsetD8(d_wte_pad, 0, (size_t)NVpad*DM*sizeof(float)), "zero wte_pad");   /* zero pad rows */
    CK(cuMemcpyHtoD(d_wte_pad, &g_wbase[off_wte()], (size_t)NV*DM*sizeof(float)), "h2d wte_pad");
    d_logits = A((size_t)Smax*NVpad);
    return 0;
}

/* ===================== modes ===================== */

/* --block0: run ONE GPT-2 block on the canonical prompt + compare to ref_block0.npy (the committed
 * P5 gate-2 anchor; behaviour byte-identical to the prior single-block build). */
static int run_block0(const char* refpath) {
    int ids[] = {464, 3139, 286, 4881, 318};
    int T = (int)(sizeof(ids)/sizeof(ids[0]));
    if (T > Spad) { fprintf(stderr, "T=%d > Spad=%d\n", T, Spad); return 2; }
    upload_layer(0);
    embed_gather(ids, T, Spad);
    forward_layer_gpt2();
    float* hout = (float*)malloc((size_t)Spad*DM*sizeof(float));
    CK(cuMemcpyDtoH(hout, d_x, (size_t)Spad*DM*sizeof(float)), "d2h block0");
    FILE* of = fopen("/tmp/helix_block0.bin", "wb");
    if (!of) { fprintf(stderr, "open /tmp/helix_block0.bin: %s\n", strerror(errno)); free(hout); return 2; }
    fwrite(hout, sizeof(float), (size_t)T*DM, of); fclose(of);
    float* ref = read_npy_f4(refpath, (size_t)T*DM);
    if (!ref) { fprintf(stderr, "could not read ref %s\n", refpath); free(hout); return 2; }
    double max_abs = 0.0, max_rel_floor = 0.0, max_rel_raw = 0.0;
    int argr=0, argc2=0, rawr=0, rawc=0, nonfinite=0;
    for (int s = 0; s < T; s++) for (int c = 0; c < DM; c++) {
        double g = (double)hout[(size_t)s*DM + c];
        double o = (double)ref[(size_t)s*DM + c];
        if (!isfinite(g)) { nonfinite++; if (nonfinite <= 4) fprintf(stderr, "  NON-FINITE helix[%d,%d]=%g (ref=%g)\n", s, c, g, o); continue; }
        double ae = fabs(g - o), ao = fabs(o);
        double re_floor = ae / (ao > 1.0 ? ao : 1.0), re_raw = ae / (ao + 1e-8);
        if (ae > max_abs) max_abs = ae;
        if (re_floor > max_rel_floor) { max_rel_floor = re_floor; argr=s; argc2=c; }
        if (re_raw   > max_rel_raw)   { max_rel_raw   = re_raw;   rawr=s; rawc=c; }
    }
    double wg = (double)hout[(size_t)argr*DM + argc2], wo = (double)ref[(size_t)argr*DM + argc2];
    double rwg = (double)hout[(size_t)rawr*DM + rawc], rwo = (double)ref[(size_t)rawr*DM + rawc];
    int pass = (nonfinite == 0) && (max_rel_floor < 1e-3);
    printf("GPU [%s] GPT-2 block-0 parity (T=%d, %d real rows x %d, %d non-finite):\n", g_gpu, T, T, DM, nonfinite);
    printf("  max_abs=%.3e  max_rel(floor=1)=%.3e  [GATE: floored max-abs-rel < 1e-3]\n", max_abs, max_rel_floor);
    printf("  max_rel(raw,/(|o|+1e-8))=%.3e at cell [%d,%d] helix=%.6g ref=%.6g (|ref|=%.3e -- near-zero, abs-err=%.3e)\n",
           max_rel_raw, rawr, rawc, rwg, rwo, fabs(rwo), fabs(rwg-rwo));
    printf("  worst floored-rel cell [%d,%d]: helix=%.6g ref=%.6g\n", argr, argc2, wg, wo);
    printf("%s\n", pass ? "GPT2_BLOCK0_PARITY_PASS" : "GPT2_BLOCK0_PARITY_FAIL");
    free(hout); free(ref);
    return pass ? 0 : 1;
}

/* parse a space/newline-separated id list file into ids[] (used for the canonical prompt ids the
 * fenced oracle dumped). Returns the count, or -1 on error. */
static int read_ids_file(const char* path, int* ids, int maxn) {
    FILE* f = fopen(path, "r"); if (!f) { fprintf(stderr, "open ids '%s': %s\n", path, strerror(errno)); return -1; }
    int n = 0, v;
    while (n < maxn && fscanf(f, "%d", &v) == 1) ids[n++] = v;
    fclose(f);
    return n;
}

/* --logits: 12 layers + ln_f + tied head; dump the LAST real-token logits [NV] to /tmp/helix_logits_last.bin
 * and compare argmax (and max-abs logit diff) to the oracle. PASS = argmax EXACT + logit diff small. */
static int run_logits(const char* ref_logits_path, const char* ref_argmax_path, const char* ids_path) {
    int ids[1024];
    int T;
    if (ids_path) { T = read_ids_file(ids_path, ids, 1024); if (T <= 0) return 2; }
    else { int c[] = {464,3139,286,4881,318}; T = 5; memcpy(ids, c, sizeof(c)); }
    printf("[logits] T=%d ids:", T); for (int i=0;i<T;i++) printf(" %d", ids[i]); printf("\n");

    float* logits = (float*)malloc((size_t)NV*sizeof(float));
    forward_full(ids, T, logits);
    int am = argmax_row(logits, NV);

    /* dump the helix last-row logits (flat <f4 [NV]) for the comparator / attestation. */
    FILE* of = fopen("/tmp/helix_logits_last.bin", "wb");
    if (!of) { fprintf(stderr, "open /tmp/helix_logits_last.bin: %s\n", strerror(errno)); free(logits); return 2; }
    fwrite(logits, sizeof(float), (size_t)NV, of); fclose(of);

    /* read the oracle's last-row logits + argmax. */
    int ref_am = -1;
    { FILE* af = fopen(ref_argmax_path, "r"); if (af) { if (fscanf(af, "%d", &ref_am) != 1) ref_am = -1; fclose(af); } }
    FILE* rf = fopen(ref_logits_path, "rb");
    if (!rf) { fprintf(stderr, "open ref logits '%s': %s\n", ref_logits_path, strerror(errno)); free(logits); return 2; }
    float* ref = (float*)malloc((size_t)NV*sizeof(float));
    size_t got = fread(ref, sizeof(float), (size_t)NV, rf); fclose(rf);
    if (got != (size_t)NV) { fprintf(stderr, "ref logits short read %zu/%d\n", got, NV); free(logits); free(ref); return 2; }
    int ref_am_from_bin = argmax_row(ref, NV);
    if (ref_am < 0) ref_am = ref_am_from_bin;

    double max_abs = 0.0; int nonfinite = 0; int wc = 0;
    for (int i = 0; i < NV; i++) {
        double g = (double)logits[i];
        if (!isfinite(g)) { nonfinite++; continue; }
        double ae = fabs(g - (double)ref[i]);
        if (ae > max_abs) { max_abs = ae; wc = i; }
    }
    int argmax_match = (nonfinite == 0) && (am == ref_am);
    /* honest absolute-tol bar on logits of O(10) after 12-layer fp32 drift: 5e-2. The argmax match is
     * the load-bearing discrete gate; the float bar is the diagnostic ladder. */
    int diff_ok = (max_abs < 5e-2);
    int pass = argmax_match && diff_ok;
    printf("GPU [%s] GPT-2 full-logits parity (T=%d, NV=%d, NVpad=%d, %d non-finite):\n", g_gpu, T, NV, NVpad, nonfinite);
    printf("  helix argmax=%d (logit=%.5g)  oracle argmax=%d  -> %s\n",
           am, logits[am], ref_am, argmax_match ? "ARGMAX_MATCH" : "ARGMAX_MISMATCH");
    printf("  max_abs logit diff=%.5e at id %d (helix=%.6g ref=%.6g)  [diag bar 5e-2: %s]\n",
           max_abs, wc, logits[wc], ref[wc], diff_ok ? "ok" : "OVER");
    printf("%s\n", pass ? "GPT2_LOGITS_PARITY_PASS" : "GPT2_LOGITS_PARITY_FAIL");
    free(logits); free(ref);
    return pass ? 0 : 1;
}

/* --generate N: greedy autoregressive loop. Forward the context, take the last-real-token argmax,
 * append, repeat N times. Dumps the produced id sequence to /tmp/helix_gen_ids.txt and prints it.
 * Compares token-for-token to the oracle's greedy continuation if ref_gen_ids is provided. */
static int run_generate(int Ngen, const char* ids_path, const char* ref_gen_path) {
    int ids[1024];
    int T0;
    if (ids_path) { T0 = read_ids_file(ids_path, ids, 1024 - Ngen); if (T0 <= 0) return 2; }
    else { int c[] = {464,3139,286,4881,318}; T0 = 5; memcpy(ids, c, sizeof(c)); }
    int T = T0;
    printf("[generate] N=%d prompt T=%d ids:", Ngen, T0); for (int i=0;i<T0;i++) printf(" %d", ids[i]); printf("\n");
    float* logits = (float*)malloc((size_t)NV*sizeof(float));
    int nonfinite_any = 0;
    for (int step = 0; step < Ngen; step++) {
        forward_full(ids, T, logits);
        for (int i = 0; i < NV; i++) if (!isfinite(logits[i])) { nonfinite_any = 1; break; }
        int nxt = argmax_row(logits, NV);
        ids[T++] = nxt;
    }
    free(logits);
    /* dump produced ids (prompt + generated). */
    { FILE* gf = fopen("/tmp/helix_gen_ids.txt", "w");
      if (gf) { for (int i=0;i<T;i++) fprintf(gf, "%d%s", ids[i], i+1<T?" ":"\n"); fclose(gf); } }
    printf("HELIX_GEN_IDS:"); for (int i=0;i<T;i++) printf(" %d", ids[i]); printf("\n");

    int pass = (nonfinite_any == 0);
    if (ref_gen_path) {
        int ref[1024]; int rn = read_ids_file(ref_gen_path, ref, 1024);
        if (rn <= 0) { fprintf(stderr, "could not read ref gen ids %s\n", ref_gen_path); pass = 0; }
        else {
            int match = (rn == T);
            int firstdiff = -1;
            int lim = rn < T ? rn : T;
            for (int i = 0; i < lim; i++) if (ids[i] != ref[i]) { match = 0; if (firstdiff<0) firstdiff=i; }
            printf("  oracle gen ids (%d):", rn); for (int i=0;i<rn;i++) printf(" %d", ref[i]); printf("\n");
            if (match) printf("  TOKEN_FOR_TOKEN_MATCH (%d ids)\n", T);
            else printf("  TOKEN_MISMATCH (helix %d ids, oracle %d ids, first diff at %d)\n", T, rn, firstdiff);
            pass = pass && match;
        }
    }
    printf("%s\n", pass ? "GPT2_GENERATE_MATCH_PASS" : "GPT2_GENERATE_MATCH_FAIL");
    return pass ? 0 : 1;
}

int main(int argc, char** argv) {
    /* env dim overrides (default = GPT-2 124M) */
    const char* e;
    if ((e=getenv("HX_NL")))    NL=atoi(e);
    if ((e=getenv("HX_D")))     DM=atoi(e);
    if ((e=getenv("HX_HEADS"))) NH=atoi(e);
    if ((e=getenv("HX_V")))     NV=atoi(e);
    if ((e=getenv("HX_CTX")))   NC=atoi(e);
    if ((e=getenv("HX_DFF")))   DFF=atoi(e);
    if ((e=getenv("HX_SPAD")))  Spad=atoi(e);
    if ((e=getenv("HX_DBG")))   g_dbg=atoi(e);
    DH = DM / NH;
    ATTN_SCALE = 1.0f / sqrtf((float)DH);

    if (argc < 4) {
        fprintf(stderr,
          "usage:\n"
          "  %s <combined.ptx> <weights> --block0 <ref_block0.npy>\n"
          "  %s <combined.ptx> <weights> --logits <ref_logits_last.bin> <ref_argmax.txt> <ref_ids.txt>\n"
          "  %s <combined.ptx> <weights> --generate <N> <ref_ids.txt> [<ref_gen_ids.txt>]\n",
          argv[0], argv[0], argv[0]);
        return 2;
    }
    const char* ptx_path = argv[1];
    const char* wpath    = argv[2];
    const char* mode     = argv[3];
    int rc = 2;

    if (strcmp(mode, "--block0") == 0) {
        if (argc < 5) { fprintf(stderr, "--block0 needs <ref_block0.npy>\n"); return 2; }
        if (device_init(ptx_path, wpath)) return 2;
        if (alloc_buffers(Spad)) return 2;          /* block0: Spad (default 64) is enough */
        rc = run_block0(argv[4]);
    } else if (strcmp(mode, "--logits") == 0) {
        if (argc < 5) { fprintf(stderr, "--logits needs <ref_logits_last.bin> [<ref_argmax.txt>] [<ref_ids.txt>]\n"); return 2; }
        const char* ref_logits = argv[4];
        const char* ref_argmax = (argc > 5) ? argv[5] : "/dev/null";
        const char* ids_path   = (argc > 6) ? argv[6] : NULL;
        if (device_init(ptx_path, wpath)) return 2;
        /* the prompt length is small; 128 padded rows covers the canonical prompt comfortably. */
        int Smax = 128;
        if (alloc_buffers(Smax)) return 2;
        if (setup_head(Smax)) return 2;
        rc = run_logits(ref_logits, ref_argmax, ids_path);
    } else if (strcmp(mode, "--generate") == 0) {
        if (argc < 5) { fprintf(stderr, "--generate needs <N> [<ref_ids.txt>] [<ref_gen_ids.txt>]\n"); return 2; }
        int Ngen = atoi(argv[4]);
        const char* ids_path     = (argc > 5) ? argv[5] : NULL;
        const char* ref_gen_path = (argc > 6) ? argv[6] : NULL;
        if (device_init(ptx_path, wpath)) return 2;
        /* size buffers for the FINAL padded length: prompt(~5) + Ngen, rounded up to a multiple of 64. */
        int Tmax = 5 + Ngen + 4;
        int Smax = ((Tmax + 63) / 64) * 64;
        if (Smax < 64) Smax = 64;
        if (alloc_buffers(Smax)) return 2;
        if (setup_head(Smax)) return 2;
        rc = run_generate(Ngen, ids_path, ref_gen_path);
    } else {
        fprintf(stderr, "unknown mode '%s'\n", mode);
        return 2;
    }

    free(g_ptx);
    if (g_map && g_map != MAP_FAILED) munmap(g_map, g_maplen);
    if (g_fd >= 0) close(g_fd);
    cuModuleUnload(g_mod); cuCtxDestroy(ctx);
    return rc;
}
