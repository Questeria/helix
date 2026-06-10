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
 * hidden [T,768], and compare to helix-llm/ref/ref_block0.npy at FLOORED max-abs-rel < 1e-3
 * (the gate metric is re_floor = |g-o| / max(|o|, 1), so |o|<=1 cells degrade to absolute error
 * and |o|>1 cells are relative; see the GATE printout below). NOTE: this GPU block-0 bar is
 * DISTINCT from (and looser than) the CPU block-0 gate in cpu_host.c, which uses the stricter
 * LITERAL max_abs < 1e-3 AND mean_abs < 1e-4. Block-0 parity proves embedding + causal mask +
 * eps-LN + multi-head + bias + GEMM orientation at once.
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
#include <time.h>

/* ===== serve-mode tokenizer linkage (ADDITIVE; used ONLY by --serve) =====
 * When built as the serve worker, gpt2_tok.c is co-compiled with GPT2_TOK_LIB
 * defined (its main() #ifdef'd out) and these four entrypoints + three
 * decode-to-buffer helpers exposed (static dropped). The four existing modes
 * (--block0/--logits/--generate) never call these, so a plain single-file
 * build of gpt2_infer.c (the scale/MVP/CPU gates) is unaffected: these are
 * declared but only referenced under the --serve branch, and the linker only
 * needs them when --serve code is reached. To keep the 4 existing gates'
 * single-file `gcc gpt2_infer.c` build working unchanged, the serve symbols
 * are weak-referenced via GPT2_SERVE only. */
#ifdef GPT2_SERVE
void  build_byte_unicode(void);
void  load_vocab(const char* path);
void  load_merges(const char* path);
int*  encode_bytes(const unsigned char* text, size_t n, int* out_n);
/* decode helpers (pure byte concatenation of g_id2bytes[id]); zero arithmetic */
char* decode_one(int id);                         /* one id  -> malloc'd C string */
char* decode_range(const int* ids, int n);        /* n ids   -> malloc'd C string */
#endif

/* ---- GPT-2 124M dims (env-overridable, defaulting to the public model) ---- */
static int NL = 12, DM = 768, NH = 12, NV = 50257, NC = 1024, DFF = 3072;
static int DH = 64;          /* head dim = DM/NH */
static int Spad = 64;        /* S padded up to a multiple of 64 for the tiled GEMMs */
static float ATTN_SCALE = 0.125f;  /* 1/sqrt(head_dim)=1/sqrt(64) */

/* ---- ADDITIVE --arch llama (2026-06): SmolLM2/TinyLlama-class models on the SAME
 * launcher. ARCH + the llama dims come from the v2 weight header (gpt2_pack --arch llama):
 * bytes 40..55 = u32 arch(1=llama), u32 n_kv_heads, f32 rope_theta, f32 rms_eps. The gpt2
 * path (v1 header) is untouched; every llama difference branches on ARCH==1. */
static int   ARCH = 0;             /* 0=gpt2 (v1 header), 1=llama (v2 header) */
static int   NKV = 0;              /* llama: n_kv_heads (GQA); KVD = NKV*DH */
static int   KVD = 0;
static float ROPE_THETA = 0.0f;
static float RMS_EPS = 1e-5f;

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
/* llama-arch kernel handles (G-L0-verified; loaded only when ARCH==1) */
static CUfunction f_rms, f_rope, f_silu;

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

/* ---- llama kernel wrappers (EXACT G-L0-verified launch geometry: one thread per row/elem,
 * grid=rows|n, block=1 -- the same shapes scripts/llama_ops_parity.sh gated green). ---- */
static CUdeviceptr d_cos, d_sin;   /* host-precomputed RoPE tables [Smax, DH/2] (uploaded once) */
/* RMSNorm y[r,:]=x[r,:]*w/rms(x[r,:]), eps baked 1e-5 in the kernel. grid=rows block=1. */
static void rms_norm_k(CUdeviceptr x, CUdeviceptr y, CUdeviceptr w, int rows, int cols) {
    int c=cols; void* ar[] = { &x,&y,&w,&c }; LX(f_rms, (unsigned)rows, 1, ar);
}
/* RoPE rotate_half IN-PLACE on a packed [rows,DH] head slab; table row s == q row s == position s. */
static void rope_k(CUdeviceptr q, int rows) {
    int half = DH/2; void* ar[] = { &q,&d_cos,&d_sin,&half }; LX(f_rope, (unsigned)rows, 1, ar);
}
/* SwiGLU gate y[i] = u[i]*silu(g[i]) over n elems. grid=n block=1 (G-L0 geometry). */
static void silu_mul_k(CUdeviceptr g, CUdeviceptr u, CUdeviceptr y, int n) {
    int nn=n; void* ar[] = { &g,&u,&y,&nn }; LX(f_silu, (unsigned)n, 1, ar);
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

/* ---- llama (v2) layout: per layer in_ln[DM] q[DM,DM] k[KVD,DM] v[KVD,DM] o[DM,DM]
 * post_ln[DM] gate[DFF,DM] up[DFF,DM] down[DM,DFF]; globals embed[NV,DM] norm[DM].
 * All Linear weights [out,in] UN-TRANSPOSED (HF order) -- every llama GEMM is mm_ABt. */
static long per_layer_floats_ll(void) {
    return (long)DM + (long)DM*DM + 2L*(long)KVD*DM + (long)DM*DM
         + (long)DM + 2L*(long)DFF*DM + (long)DM*DFF;
}
typedef struct { long inln, qW, kW, vW, oW, postln, gateW, upW, downW; } LayerOffLL;
static LayerOffLL layer_offsets_ll(void) {
    LayerOffLL o; long p = 0;
    o.inln   = p; p += DM;
    o.qW     = p; p += (long)DM*DM;
    o.kW     = p; p += (long)KVD*DM;
    o.vW     = p; p += (long)KVD*DM;
    o.oW     = p; p += (long)DM*DM;
    o.postln = p; p += DM;
    o.gateW  = p; p += (long)DFF*DM;
    o.upW    = p; p += (long)DFF*DM;
    o.downW  = p; p += (long)DM*DFF;
    return o;
}
static long off_layer_ll(int L)  { return (long)L * per_layer_floats_ll(); }
static long off_embed_ll(void)   { return (long)NL * per_layer_floats_ll(); }
static long off_normf_ll(void)   { return off_embed_ll() + (long)NV*DM; }

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
    if (ver != VERSION && ver != 2u) { fprintf(stderr, "bad version %u\n", ver); return 2; }
    if ((ver == 2u) != (ARCH == 1)) { fprintf(stderr, "header ver %u vs ARCH %d mismatch (peek failed?)\n", ver, ARCH); return 2; }
    if ((int)nl!=NL || (int)dm!=DM || (int)nh!=NH || (int)nv!=NV || (int)nc!=NC || (int)dff!=DFF) {
        fprintf(stderr, "header dims (nl=%u dm=%u nh=%u nv=%u nc=%u dff=%u) != config\n", nl,dm,nh,nv,nc,dff); return 2;
    }
    size_t want = (size_t)HDR_BYTES + nfloat * 4;
    if (g_maplen < want) { fprintf(stderr, "short weight file: %zu < %zu\n", g_maplen, want); return 2; }
    g_nfloat = (size_t)nfloat;
    /* sanity: the computed float layout must total n_float exactly. */
    size_t expect = ARCH
        ? (size_t)off_normf_ll() + DM
        : (size_t)off_globals() + (size_t)NV*DM + (size_t)NC*DM + DM + DM;
    if (expect != g_nfloat) { fprintf(stderr, "layout %zu floats != header n_float %zu\n", expect, g_nfloat); return 2; }
    g_wbase = (const float*)(hb + HDR_BYTES);
    printf("[wt] mmap %zu B, n_float=%zu, arch=%s layout verified (per_layer=%ld)\n",
           g_maplen, g_nfloat, ARCH ? "llama" : "gpt2", ARCH ? per_layer_floats_ll() : per_layer_floats());
    return 0;
}

/* peek the weight header BEFORE device init: a v2 (llama) file is self-describing, so it
 * SETS the model dims + arch fields (the HX_* env defaults describe gpt2 family only). */
static int peek_weights_header(const char* path) {
    FILE* f = fopen(path, "rb"); if (!f) { fprintf(stderr, "open weights '%s': %s\n", path, strerror(errno)); return 2; }
    unsigned char hb[HDR_BYTES];
    if (fread(hb, 1, HDR_BYTES, f) != HDR_BYTES) { fclose(f); fprintf(stderr, "short weights header\n"); return 2; }
    fclose(f);
    uint32_t magic, ver; memcpy(&magic, hb, 4); memcpy(&ver, hb+4, 4);
    if (magic != MAGIC) { fprintf(stderr, "peek: bad magic\n"); return 2; }
    if (ver == 2u) {
        uint32_t nl,dm,nh,nv,nc,dff,arch,nkv; float th, ep;
        memcpy(&nl,hb+8,4); memcpy(&dm,hb+12,4); memcpy(&nh,hb+16,4); memcpy(&nv,hb+20,4);
        memcpy(&nc,hb+24,4); memcpy(&dff,hb+28,4);
        memcpy(&arch,hb+40,4); memcpy(&nkv,hb+44,4); memcpy(&th,hb+48,4); memcpy(&ep,hb+52,4);
        if (arch != 1u) { fprintf(stderr, "peek: v2 header with unknown arch %u\n", arch); return 2; }
        ARCH = 1; NL=(int)nl; DM=(int)dm; NH=(int)nh; NV=(int)nv; NC=(int)nc; DFF=(int)dff;
        NKV=(int)nkv; ROPE_THETA=th; RMS_EPS=ep;
        DH = DM / NH; KVD = NKV * DH;
        ATTN_SCALE = 1.0f / sqrtf((float)DH);
        if (NKV <= 0 || NH % NKV) { fprintf(stderr, "peek: bad GQA heads %d/%d\n", NH, NKV); return 2; }
        /* the rmsnorm kernel BAKES eps=1e-5 (G-L0-verified); fail closed on any other eps
         * rather than run silently-wrong numerics. */
        if (fabsf(RMS_EPS - 1e-5f) > 1e-9f) { fprintf(stderr, "peek: rms_eps %g != the kernel's baked 1e-5\n", (double)RMS_EPS); return 2; }
        printf("[peek] llama v2 header: NL=%d DM=%d NH=%d NKV=%d NV=%d NC=%d DFF=%d theta=%g\n",
               NL, DM, NH, NKV, NV, NC, DFF, (double)ROPE_THETA);
    }
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
/* llama-arch extras (ARCH==1 only) */
static void upload_layer_ll(int L);                                  /* defined after alloc_buffers */
static int  read_ids_file(const char* path, int* ids, int maxn);     /* defined with the modes */
static CUdeviceptr d_qW, d_kW, d_vW, d_oW, d_gateW, d_upW, d_downW;  /* layer weights */
static CUdeviceptr d_q, d_k, d_v;    /* q/k/v GEMM outputs (carved from d_qkv) */
static CUdeviceptr d_mlp1c;          /* silu_mul output [Spad,DFF] */
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

/* ===================== SERVE-MODE TELEMETRY EMIT MODULE (ADDITIVE) =====================
 * A tiny printf-to-an-fd side-effect layer. EVERY writer reads values ALREADY in host
 * scope (step, layer index, literal kernel-name strings, the already-D2H'd argmax logit).
 * It reads NO device memory in any new way, adds NO cuCtxSynchronize, mutates NO buffer,
 * and changes NO kernel argument. When g_serve==0 (the 4 existing modes) every emit is a
 * no-op, so --block0/--logits/--generate stay byte-identical in behavior. The numeric
 * forward path (forward_full/forward_layer_gpt2) is unchanged; G1 token-for-token proves it.
 *
 * Wire format: one compact JSON object per line, '\n'-terminated, written to g_emit_fd.
 * Each object carries "_ev" (event name) + "seq" (monotone). The HTTP server reads those
 * to set the SSE event:/id: lines and forwards data: verbatim (no double schema). */
extern char g_gpu[256];          /* device name (cuDeviceGetName); defined in the device-init section */
static int  g_serve     = 0;     /* set in the --serve branch; gates all emits */
static int  g_emit_fd   = 1;     /* fd the JSON lines go to (default stdout) */
static long g_seq       = 0;     /* monotone ordering token (server mirrors -> SSE id:) */
static int  g_emit_step = 0;     /* current generation step (read by op hooks) */
static int  g_emit_layer= 0;     /* current layer index (read by op hooks) */
static int  g_layer_ops = 0;     /* op count within the current layer (for layer_end.ops) */
static int  g_timing    = 0;     /* --timing 1: real per-layer host-clock ms; else 0 */
static long g_ptx_len   = 0;     /* minted-PTX byte length captured in device_init */
#define DETAIL_LAYER 0
#define DETAIL_OP    1
static int  g_detail    = DETAIL_OP;

/* pinned trust anchors (the REAL hashes; same values demo/dashboard.html embeds). */
static const char* SEED_SHA_PIN     = "9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb";
static const char* FIXPOINT_SHA_PIN = "0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f";
static const char* GCC_DDC_SHA_PIN  = "84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba";

static double now_seconds(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static int clampi(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

static void emit_raw(const char* json) {        /* json has a trailing '\n' */
    if (g_emit_fd < 0) return;
    size_t n = strlen(json), off = 0;
    while (off < n) { ssize_t w = write(g_emit_fd, json + off, n - off); if (w <= 0) break; off += (size_t)w; }
}

/* JSON-escape a byte string into out (caps at outcap-1). Only the GPT-2 byte pieces can
 * contain control bytes / quotes / backslashes / newlines; everything else is ASCII. */
static void jesc(const char* s, char* out, size_t outcap) {
    size_t k = 0;
    if (!s) { out[0] = 0; return; }
    for (size_t i = 0; s[i] && k + 7 < outcap; i++) {
        unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"':  out[k++]='\\'; out[k++]='"';  break;
            case '\\': out[k++]='\\'; out[k++]='\\'; break;
            case '\n': out[k++]='\\'; out[k++]='n';  break;
            case '\r': out[k++]='\\'; out[k++]='r';  break;
            case '\t': out[k++]='\\'; out[k++]='t';  break;
            default:
                if (c < 0x20) { /* other control bytes -> \u00XX */
                    static const char* H = "0123456789abcdef";
                    out[k++]='\\'; out[k++]='u'; out[k++]='0'; out[k++]='0';
                    out[k++]=H[(c>>4)&0xF]; out[k++]=H[c&0xF];
                } else out[k++] = (char)c;       /* keep raw UTF-8 bytes verbatim */
        }
    }
    out[k] = 0;
}

static void emit_init(int fd, const char* detail) {
    g_serve = 1; g_emit_fd = fd; g_seq = 0;
    g_detail = (detail && strcmp(detail, "layer") == 0) ? DETAIL_LAYER : DETAIL_OP;
}

/* hello: static model/trust/device header. ALL values are real: dims from the live globals
 * (set from the HX_ env / the verified weight header), device = cuDeviceGetName, ptx_bytes =
 * the minted-PTX byte length, kernels = the 8 literal cuModuleGetFunction names, seed/fixpoint
 * = the pinned anchors. */
static void emit_hello(void) {
    if (!g_serve) return;
    char b[1400];
    snprintf(b, sizeof b,
      "{\"_ev\":\"hello\",\"seq\":%ld,\"schema_version\":1,\"model\":\"gpt2-xl\",\"params\":\"1.5B\","
      "\"n_layer\":%d,\"n_head\":%d,\"d_model\":%d,\"d_ff\":%d,\"n_vocab\":%d,"
      "\"device\":\"%s\",\"sm\":\"sm_86\",\"precision\":\"fp32\",\"build\":\"forward-only\",\"mode\":\"serve\","
      "\"ptx_bytes\":%ld,\"seed_sha\":\"%s\",\"fixpoint_sha\":\"%s\",\"gcc_ddc_sha\":\"%s\","
      "\"kernels\":[\"tiled_matmul\",\"tiled_matmul_abt\",\"gpu_layernorm_fwd_eps\",\"gpu_softmax_causal\","
      "\"gpu_add_bias_rowbcast\",\"gpu_gelu_stable\",\"vector_add\",\"gpu_scale_rt\"]}\n",
      g_seq++, NL, NH, DM, DFF, NV, g_gpu, g_ptx_len, SEED_SHA_PIN, FIXPOINT_SHA_PIN, GCC_DDC_SHA_PIN);
    emit_raw(b);
}

/* tokenize: the real prompt ids + their decoded display pieces + n_prompt + s_pad. */
static void emit_tokenize(const int* ids, int T0, int s_pad) {
    if (!g_serve) return;
    /* ids array */
    size_t cap = (size_t)T0 * 12 + 64; char* idbuf = (char*)malloc(cap); size_t k = 0;
    for (int i = 0; i < T0; i++) k += (size_t)snprintf(idbuf + k, cap - k, "%s%d", i ? "," : "", ids[i]);
    /* strings array (decode each id; escape) */
    size_t scap = (size_t)T0 * 24 + 64; char* sbuf = (char*)malloc(scap); size_t sk = 0;
    sk += (size_t)snprintf(sbuf + sk, scap - sk, "[");
#ifdef GPT2_SERVE
    for (int i = 0; i < T0; i++) {
        char* piece = decode_one(ids[i]); char esc[256]; jesc(piece ? piece : "", esc, sizeof esc);
        /* grow if needed */
        size_t need = sk + strlen(esc) + 8;
        if (need >= scap) { scap = need * 2; sbuf = (char*)realloc(sbuf, scap); }
        sk += (size_t)snprintf(sbuf + sk, scap - sk, "%s\"%s\"", i ? "," : "", esc);
        free(piece);
    }
#endif
    sk += (size_t)snprintf(sbuf + sk, scap - sk, "]");
    size_t obcap = cap + scap + 128; char* ob = (char*)malloc(obcap);
    snprintf(ob, obcap,
      "{\"_ev\":\"tokenize\",\"seq\":%ld,\"ids\":[%s],\"strings\":%s,\"n_prompt\":%d,\"s_pad\":%d}\n",
      g_seq++, idbuf, sbuf, T0, s_pad);
    emit_raw(ob);
    free(idbuf); free(sbuf); free(ob);
}

static void emit_forward_begin(int step, int context_len, int s_pad, int n_layers) {
    if (!g_serve) return;
    char b[160];
    snprintf(b, sizeof b,
      "{\"_ev\":\"forward_begin\",\"seq\":%ld,\"step\":%d,\"context_len\":%d,\"s_pad\":%d,\"n_layers\":%d}\n",
      g_seq++, step, context_len, s_pad, n_layers);
    emit_raw(b);
}
static void emit_embed(int step, int t, int d_model) {
    if (!g_serve) return;
    char b[128];
    snprintf(b, sizeof b, "{\"_ev\":\"embed\",\"seq\":%ld,\"step\":%d,\"t\":%d,\"d_model\":%d}\n",
             g_seq++, step, t, d_model);
    emit_raw(b);
}
static void emit_layer_begin(int step, int idx, int total) {
    if (!g_serve) return;
    g_layer_ops = 0;
    char b[128];
    snprintf(b, sizeof b, "{\"_ev\":\"layer_begin\",\"seq\":%ld,\"step\":%d,\"idx\":%d,\"total\":%d}\n",
             g_seq++, step, idx, total);
    emit_raw(b);
}
/* op: wraps a REAL cuLaunchKernel. Dropped entirely at --detail layer (a real subset). */
static void emit_op(int layer, int seq_in_layer, const char* kernel, const char* phase,
                    const char* label, int agg) {
    if (!g_serve || g_detail < DETAIL_OP) return;
    char b[256];
    snprintf(b, sizeof b,
      "{\"_ev\":\"op\",\"seq\":%ld,\"step\":%d,\"layer\":%d,\"seq_in_layer\":%d,"
      "\"kernel\":\"%s\",\"phase\":\"%s\",\"label\":\"%s\",\"agg\":%d}\n",
      g_seq++, g_emit_step, layer, seq_in_layer, kernel, phase, label, agg);
    emit_raw(b); g_layer_ops++;
}
static void emit_layer_end(int step, int idx, double ms) {
    if (!g_serve) return;
    char b[160];
    snprintf(b, sizeof b, "{\"_ev\":\"layer_end\",\"seq\":%ld,\"step\":%d,\"idx\":%d,\"ms\":%.3f,\"ops\":%d}\n",
             g_seq++, step, idx, ms, g_layer_ops);
    emit_raw(b);
}
static void emit_head(int step, const char* label, const char* kernel) {
    if (!g_serve) return;
    char b[160];
    snprintf(b, sizeof b, "{\"_ev\":\"head\",\"seq\":%ld,\"step\":%d,\"label\":\"%s\",\"kernel\":\"%s\"}\n",
             g_seq++, step, label, kernel);
    emit_raw(b);
}
static void emit_token(int step, int id, const char* string, double logit, int context_len) {
    if (!g_serve) return;
    char esc[256]; jesc(string ? string : "", esc, sizeof esc);
    char b[512];
    snprintf(b, sizeof b,
      "{\"_ev\":\"token\",\"seq\":%ld,\"step\":%d,\"id\":%d,\"string\":\"%s\",\"logit\":%.5f,\"context_len\":%d}\n",
      g_seq++, step, id, esc, logit, context_len);
    emit_raw(b);
}
static void emit_done(int n_prompt, int n_gen, int n_total, double seconds, double tok_per_s,
                      const char* text, const int* gen_ids, int ngi, int nonfinite) {
    if (!g_serve) return;
    char escbuf[8192]; jesc(text ? text : "", escbuf, sizeof escbuf);
    size_t gcap = (size_t)ngi * 12 + 64; char* gbuf = (char*)malloc(gcap); size_t k = 0;
    for (int i = 0; i < ngi; i++) k += (size_t)snprintf(gbuf + k, gcap - k, "%s%d", i ? "," : "", gen_ids[i]);
    size_t obcap = sizeof(escbuf) + gcap + 256; char* ob = (char*)malloc(obcap);
    snprintf(ob, obcap,
      "{\"_ev\":\"done\",\"seq\":%ld,\"n_prompt\":%d,\"n_gen\":%d,\"n_total\":%d,"
      "\"seconds\":%.3f,\"tok_per_s\":%.3f,\"text\":\"%s\",\"gen_ids\":[%s],\"nonfinite\":%d}\n",
      g_seq++, n_prompt, n_gen, n_total, seconds, tok_per_s, escbuf, gbuf, nonfinite);
    emit_raw(ob);
    free(gbuf); free(ob);
}
static void emit_error(const char* where, const char* message, int fatal) {
    if (!g_serve) return;
    char em[512]; jesc(message ? message : "", em, sizeof em);
    char b[640];
    snprintf(b, sizeof b, "{\"_ev\":\"error\",\"seq\":%ld,\"where\":\"%s\",\"message\":\"%s\",\"fatal\":%s}\n",
             g_seq++, where ? where : "driver", em, fatal ? "true" : "false");
    emit_raw(b);
}

/* forward_layer_gpt2(x): x is the [Spad,DM] residual stream (in-place updated).
 *   ln_eps(ln_1) -> mm_AB QKV + bias -> NH-head { pack Q/K/V, mm_ABt scores, scale, causal
 *   softmax, mm_AB @V, scatter } -> mm_AB c_proj + bias -> residual add
 *   -> ln_eps(ln_2) -> mm_AB c_fc + bias -> gelu -> mm_AB mlp c_proj + bias -> residual add.
 * NOTE: the "12-head"/"per-head" figures below describe the 124M default (NH=12); at XL the
 * runtime overrides NH=25 (and NL=48) via the HX_* env (see main(): HX_HEADS/HX_NL). The head
 * loop is `for (h=0; h<NH; h++)`, so it is dimension-generic -- the comment counts are illustrative
 * of the 124M case, not a fixed 12. */
static void forward_layer_gpt2(void) {
    /* Emit hooks are printf-only on host-scope values (g_emit_layer/g_emit_step + literal
     * kernel-name strings). They wrap the SAME real launches; they read no device memory,
     * add no sync, and change no arg. The NH per-head launches stay emit-silent; the 3
     * dominating attention kernels are reported once after the head loop as aggregates
     * (agg=NH), so the frontend sees a faithful 17-op/layer map without NH events/layer.
     * (NH=12 at the 124M default; NH=25 at XL via HX_HEADS.) */
    int L = g_emit_layer;
    /* --- attention --- */
    dbg_row("x_in", d_x, 0, DM);
    emit_op(L, 0, "gpu_layernorm_fwd_eps", "attn", "ln_1", 1);
    ln_eps(d_x, d_xn, d_ln1g, d_ln1b, Spad, DM);
    dbg_row("ln1", d_xn, 0, DM);
    emit_op(L, 1, "tiled_matmul", "attn", "qkv_gemm", 1);
    mm_AB(d_xn, d_attW, d_qkv, Spad, DM, 3*DM);
    emit_op(L, 2, "gpu_add_bias_rowbcast", "attn", "qkv_bias", 1);
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
    /* the 3 aggregate attention ops (one per real per-head kernel; agg=NH launches each) */
    emit_op(L, 3, "tiled_matmul_abt", "attn", "attn_scores", NH);
    emit_op(L, 4, "gpu_scale_rt",     "attn", "attn_scale",  NH);
    emit_op(L, 5, "gpu_softmax_causal","attn","attn_softmax", NH);
    emit_op(L, 6, "tiled_matmul",     "attn", "attn_av",     NH);
    dbg_row("ctx", d_ctx, 0, DM);
    emit_op(L, 7, "tiled_matmul", "attn", "attn_proj_gemm", 1);
    mm_AB(d_ctx, d_prjW, d_proj, Spad, DM, DM);
    emit_op(L, 8, "gpu_add_bias_rowbcast", "attn", "attn_proj_bias", 1);
    add_bias(d_proj, d_prjb, Spad*DM, DM);
    dbg_row("proj", d_proj, 0, DM);
    emit_op(L, 9, "vector_add", "attn", "attn_residual", 1);
    vadd(d_x, d_proj, d_x, Spad*DM);                     /* x = x + attn_proj (residual) */
    dbg_row("x_after_attn", d_x, 0, DM);
    /* --- MLP --- */
    emit_op(L, 10, "gpu_layernorm_fwd_eps", "mlp", "ln_2", 1);
    ln_eps(d_x, d_xn2, d_ln2g, d_ln2b, Spad, DM);
    emit_op(L, 11, "tiled_matmul", "mlp", "fc_gemm", 1);
    mm_AB(d_xn2, d_fcW, d_mlp1, Spad, DM, DFF);
    emit_op(L, 12, "gpu_add_bias_rowbcast", "mlp", "fc_bias", 1);
    add_bias(d_mlp1, d_fcb, Spad*DFF, DFF);
    emit_op(L, 13, "gpu_gelu_stable", "mlp", "gelu", 1);
    gelu(d_mlp1, d_mlp1g, Spad*DFF);
    dbg_row("gelu", d_mlp1g, 0, DFF);
    emit_op(L, 14, "tiled_matmul", "mlp", "proj2_gemm", 1);
    mm_AB(d_mlp1g, d_pjW, d_mlp2, Spad, DFF, DM);
    dbg_row("mlp2", d_mlp2, 0, DM);
    emit_op(L, 15, "gpu_add_bias_rowbcast", "mlp", "proj2_bias", 1);
    add_bias(d_mlp2, d_pjb, Spad*DM, DM);
    emit_op(L, 16, "vector_add", "mlp", "mlp_residual", 1);
    vadd(d_x, d_mlp2, d_x, Spad*DM);                     /* x = x + mlp (residual) */
    dbg_row("x_final", d_x, 0, DM);
}

/* ===================== full forward (NL layers + ln_f + tied head) ===================== */
/* NL=12 at the 124M default; NL=48 at XL via HX_NL. The layer loop below is `for (L=0; L<NL; L++)`,
 * so this path is dimension-generic -- "NL layers" is not a fixed 12. */
/* HOST embedding gather into d_x: x[s,:] = wte[ids[s]] + wpe[s] for s<T, zero pad rows.
 * Byte-movement only (host glue) -- no arithmetic-on-the-trust-path beyond the add the capstone
 * itself does for its input injection. */
static void embed_gather(const int* ids, int T, int S) {
    float* hx = (float*)calloc((size_t)S*DM, sizeof(float));
    if (ARCH) {
        /* llama: token embedding ONLY (RoPE replaces positional embeddings) -- pure row copy. */
        const float* emb = &g_wbase[off_embed_ll()];
        for (int s = 0; s < T; s++)
            memcpy(&hx[(size_t)s*DM], &emb[(size_t)ids[s]*DM], (size_t)DM*sizeof(float));
    } else {
        const float* wte = &g_wbase[off_wte()];
        const float* wpe = &g_wbase[off_wpe()];
        for (int s = 0; s < T; s++) {
            const float* rw = &wte[(size_t)ids[s]*DM];
            const float* rp = &wpe[(size_t)s*DM];
            for (int c = 0; c < DM; c++) hx[(size_t)s*DM + c] = rw[c] + rp[c];
        }
    }
    CKX(cuMemcpyHtoD(d_x, hx, (size_t)S*DM*sizeof(float)), "h2d x_in (gather)");
    free(hx);
}

/* forward_layer_llama(x): the Llama-arch block on the SAME machinery (docs/HELIX_LLAMA_PLAN.md
 * section 2; 5 reused GPT-2 kernels + the 3 G-L0-verified new ones; GQA is host indexing only):
 *   rmsnorm -> q/k/v GEMMs (A.Bt, HF [out,in] weights untransposed) -> per q-head { pack q
 *   (DM cols) + k/v from the kv head kv=h/(NH/NKV) (KVD cols), RoPE q+k, scores=Q.Kt, *scale,
 *   causal softmax, @V, scatter } -> o_proj (A.Bt, no bias) -> residual -> rmsnorm ->
 *   SwiGLU { gate=A.Bt, up=A.Bt, y=u*silu(g) } -> down (A.Bt) -> residual. NO biases anywhere. */
static void forward_layer_llama(void) {
    int L = g_emit_layer;
    int group = NH / NKV;                 /* GQA group size (9/3 = 3 for SmolLM2) */
    emit_op(L, 0, "gpu_rmsnorm_fwd_eps", "attn", "rms_1", 1);
    rms_norm_k(d_x, d_xn, d_ln1g, Spad, DM);
    emit_op(L, 1, "tiled_matmul_abt", "attn", "q_gemm", 1);
    mm_ABt(d_xn, d_qW, d_q, Spad, DM, DM);
    emit_op(L, 2, "tiled_matmul_abt", "attn", "k_gemm", 1);
    mm_ABt(d_xn, d_kW, d_k, Spad, DM, KVD);
    emit_op(L, 3, "tiled_matmul_abt", "attn", "v_gemm", 1);
    mm_ABt(d_xn, d_vW, d_v, Spad, DM, KVD);
    for (int h = 0; h < NH; h++) {
        int kv = h / group;               /* the pinned GQA mapping (oracle: gqa_kv_head) */
        pack_head(d_Qh, d_q, h*DH,  DM);
        pack_head(d_Kh, d_k, kv*DH, KVD);
        pack_head(d_Vh, d_v, kv*DH, KVD);
        rope_k(d_Qh, Spad);               /* rotate_half RoPE in-place, position == row */
        rope_k(d_Kh, Spad);
        mm_ABt(d_Qh, d_Kh, d_scores, Spad, DH, Spad);
        scale_rt(d_scores, d_scale, Spad*Spad);
        softmax_causal(d_scores, d_attnw, Spad, Spad);
        mm_AB(d_attnw, d_Vh, d_aoh, Spad, Spad, DH);
        scatter_head(d_ctx, h*DH, d_aoh);
    }
    emit_op(L, 4, "gpu_rope_rot",       "attn", "rope_qk",      2*NH);
    emit_op(L, 5, "tiled_matmul_abt",   "attn", "attn_scores",  NH);
    emit_op(L, 6, "gpu_scale_rt",       "attn", "attn_scale",   NH);
    emit_op(L, 7, "gpu_softmax_causal", "attn", "attn_softmax", NH);
    emit_op(L, 8, "tiled_matmul",       "attn", "attn_av",      NH);
    emit_op(L, 9, "tiled_matmul_abt",   "attn", "attn_proj", 1);
    mm_ABt(d_ctx, d_oW, d_proj, Spad, DM, DM);
    emit_op(L, 10, "vector_add", "attn", "attn_residual", 1);
    vadd(d_x, d_proj, d_x, Spad*DM);
    emit_op(L, 11, "gpu_rmsnorm_fwd_eps", "mlp", "rms_2", 1);
    rms_norm_k(d_x, d_xn2, d_ln2g, Spad, DM);
    emit_op(L, 12, "tiled_matmul_abt", "mlp", "fc_gate", 1);
    mm_ABt(d_xn2, d_gateW, d_mlp1, Spad, DM, DFF);
    emit_op(L, 13, "tiled_matmul_abt", "mlp", "fc_up", 1);
    mm_ABt(d_xn2, d_upW, d_mlp1g, Spad, DM, DFF);
    emit_op(L, 14, "gpu_silu_mul", "mlp", "silu_mul", 1);
    silu_mul_k(d_mlp1, d_mlp1g, d_mlp1c, Spad*DFF);
    emit_op(L, 15, "tiled_matmul_abt", "mlp", "proj_down", 1);
    mm_ABt(d_mlp1c, d_downW, d_mlp2, Spad, DFF, DM);
    emit_op(L, 16, "vector_add", "mlp", "mlp_residual", 1);
    vadd(d_x, d_mlp2, d_x, Spad*DM);
}

/* run the full GPT-2 forward for ids[0..T) and copy the LAST REAL-TOKEN row of logits [NV] into
 * out (host). S = T padded up to a multiple of 64 (set as the global Spad for the layer launches).
 * For greedy decoding only the last real row matters, but we compute the full [S,NVpad] head (the
 * tiled GEMM needs M=S%64==0; reading just row T-1 is free). Pad cols [NV..NVpad) are zeroed in
 * d_wte_pad so their logits are 0 and never argmaxed. All arithmetic is in kovc kernels. */
static void forward_full(const int* ids, int T, float* out_last_logits) {
    int S = ((T + 63) / 64) * 64;
    Spad = S;                                  /* the layer kernels read the global Spad */
    /* SERVE no-leak safety: d_ctx is cuMemsetD8-zeroed only at alloc, then written by
     * scatter_head per layer. A smaller-T request after a larger-T one could otherwise see
     * a previous request's stale pad rows (rows >= T). This zeroes ONLY pad/unused rows to a
     * deterministic state; real-token rows are fully overwritten by scatter_head, so the
     * argmaxed last real row is unaffected -> served ids unchanged (G1 confirms). No-op for
     * the 4 non-serve modes (g_serve==0). */
    if (g_serve) CKX(cuMemsetD8(d_ctx, 0, (size_t)S*DM*sizeof(float)), "serve zero d_ctx");
    embed_gather(ids, T, S);
    emit_embed(g_emit_step, T, DM);            /* one per token-step (no-op when !g_serve) */
    for (int L = 0; L < NL; L++) {
        g_emit_layer = L;
        double t_layer = (g_serve && g_timing) ? now_seconds() : 0.0;
        if (ARCH) upload_layer_ll(L); else upload_layer(L);
        emit_layer_begin(g_emit_step, L, NL);  /* primary heartbeat; after upload, before compute */
        if (ARCH) forward_layer_llama(); else forward_layer_gpt2();
        double ms = (g_serve && g_timing) ? (now_seconds() - t_layer) * 1e3 : 0.0;
        emit_layer_end(g_emit_step, L, ms);
    }
    if (ARCH) {
        emit_head(g_emit_step, "norm_f", "gpu_rmsnorm_fwd_eps");
        rms_norm_k(d_x, d_xn, d_lnfg, S, DM);  /* final RMSNorm (llama has no final bias) */
    } else {
        emit_head(g_emit_step, "ln_f", "gpu_layernorm_fwd_eps");
        ln_eps(d_x, d_xn, d_lnfg, d_lnfb, S, DM);  /* final LayerNorm (reuse d_xn as ln_f output) */
    }
    /* tied head: logits[S,NVpad] = xnorm[S,DM] @ wte_pad[NVpad,DM]^T (wte rows ARE the tied head). */
    emit_head(g_emit_step, "lm_head", "tiled_matmul_abt");
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
char            g_gpu[256];       /* non-static: read by the emit module's hello (declared extern above) */
static char*    g_ptx = NULL;

/* load PTX, create the context, fetch the forward-only kernel handles, mmap the weight file. */
static int device_init(const char* ptx_path, const char* wpath) {
    FILE* pf = fopen(ptx_path, "rb"); if (!pf) { fprintf(stderr, "open ptx '%s'\n", ptx_path); return 2; }
    fseek(pf,0,SEEK_END); long psz=ftell(pf); fseek(pf,0,SEEK_SET);
    g_ptx=(char*)malloc(psz+1); if (fread(g_ptx,1,psz,pf)!=(size_t)psz) { fclose(pf); return 2; } g_ptx[psz]=0; fclose(pf);
    g_ptx_len = psz;                          /* real minted-PTX byte length for the hello event */
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
    if (ARCH) {   /* the 3 G-L0-verified llama kernels (present in the 11-kernel llama PTX mint) */
        CK(cuModuleGetFunction(&f_rms,  g_mod, "gpu_rmsnorm_fwd_eps"),  "gpu_rmsnorm_fwd_eps");
        CK(cuModuleGetFunction(&f_rope, g_mod, "gpu_rope_rot"),         "gpu_rope_rot");
        CK(cuModuleGetFunction(&f_silu, g_mod, "gpu_silu_mul"),         "gpu_silu_mul");
    }
    if (load_gpt2_weights(wpath)) return 2;
    return 0;
}

/* allocate the layer-weight device buffers (reused across all NL layers; NL=12 at the 124M default,
 * NL=48 at XL via HX_NL) + the per-Smax activations. One layer's weights resident at a time. */
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
    if (ARCH) {   /* llama extras: separate-GEMM weights, the SwiGLU 3rd slab, RoPE tables */
        d_qW    = A((size_t)DM*DM);
        d_kW    = A((size_t)KVD*DM);
        d_vW    = A((size_t)KVD*DM);
        d_oW    = A((size_t)DM*DM);
        d_gateW = A((size_t)DFF*DM);
        d_upW   = A((size_t)DFF*DM);
        d_downW = A((size_t)DM*DFF);
        d_mlp1c = A((size_t)Smax*DFF);
        /* q/k/v GEMM outputs are CARVED out of the (larger) fused d_qkv slab:
         * q[Smax,DM] @ 0, k[Smax,KVD] after it, v[Smax,KVD] after that
         * (DM + 2*KVD <= 3*DM always, since KVD <= DM). */
        d_q = d_qkv;
        d_k = d_qkv + (CUdeviceptr)((size_t)Smax*DM*sizeof(float));
        d_v = d_k   + (CUdeviceptr)((size_t)Smax*KVD*sizeof(float));
        /* RoPE tables [Smax, DH/2]: HF inv_freq convention, host-built in double, cast f32
         * (cos/sin tables are DATA like weights -- the plan's pinned convention). */
        int half = DH/2;
        float* hc = (float*)malloc((size_t)Smax*half*sizeof(float));
        float* hs = (float*)malloc((size_t)Smax*half*sizeof(float));
        for (int s = 0; s < Smax; s++) {
            for (int j = 0; j < half; j++) {
                double inv = pow((double)ROPE_THETA, -2.0*(double)j/(double)DH);
                double ang = (double)s * inv;
                hc[(size_t)s*half+j] = (float)cos(ang);
                hs[(size_t)s*half+j] = (float)sin(ang);
            }
        }
        d_cos = A((size_t)Smax*half);
        d_sin = A((size_t)Smax*half);
        CK(cuMemcpyHtoD(d_cos, hc, (size_t)Smax*half*sizeof(float)), "h2d rope cos");
        CK(cuMemcpyHtoD(d_sin, hs, (size_t)Smax*half*sizeof(float)), "h2d rope sin");
        free(hc); free(hs);
    }
    return 0;
}

/* upload llama layer L's 9 tensors (reuses d_ln1g for input_ln, d_ln2g for post_ln). */
static void upload_layer_ll(int L) {
    LayerOffLL lo = layer_offsets_ll();
    long b = off_layer_ll(L);
    CKX(cuMemcpyHtoD(d_ln1g,  &g_wbase[b+lo.inln],   (size_t)DM*sizeof(float)),      "up in_ln");
    CKX(cuMemcpyHtoD(d_qW,    &g_wbase[b+lo.qW],     (size_t)DM*DM*sizeof(float)),   "up qW");
    CKX(cuMemcpyHtoD(d_kW,    &g_wbase[b+lo.kW],     (size_t)KVD*DM*sizeof(float)),  "up kW");
    CKX(cuMemcpyHtoD(d_vW,    &g_wbase[b+lo.vW],     (size_t)KVD*DM*sizeof(float)),  "up vW");
    CKX(cuMemcpyHtoD(d_oW,    &g_wbase[b+lo.oW],     (size_t)DM*DM*sizeof(float)),   "up oW");
    CKX(cuMemcpyHtoD(d_ln2g,  &g_wbase[b+lo.postln], (size_t)DM*sizeof(float)),      "up post_ln");
    CKX(cuMemcpyHtoD(d_gateW, &g_wbase[b+lo.gateW],  (size_t)DFF*DM*sizeof(float)),  "up gateW");
    CKX(cuMemcpyHtoD(d_upW,   &g_wbase[b+lo.upW],    (size_t)DFF*DM*sizeof(float)),  "up upW");
    CKX(cuMemcpyHtoD(d_downW, &g_wbase[b+lo.downW],  (size_t)DM*DFF*sizeof(float)),  "up downW");
}

/* set up the tied LM head: ln_f buffers + the padded wte [NVpad,DM] on device + the logits buffer.
 * The tied head is wte itself (logits = x @ wte^T); we copy the NV real rows into a NVpad-row device
 * buffer whose pad rows [NV..NVpad) are zeroed so their logits are 0 and never argmaxed. */
static int setup_head(int Smax) {
    NVpad = ((NV + 63) / 64) * 64;     /* 50257 -> 50304 (llama 49152 is already %64) */
    if (ARCH) {
        d_lnfg = up_slice(off_normf_ll(), DM);   /* final RMSNorm weight (no bias in llama) */
        d_lnfb = 0;
    } else {
        d_lnfg = up_slice(off_lnfg(), DM);
        d_lnfb = up_slice(off_lnfb(), DM);
    }
    long emb = ARCH ? off_embed_ll() : off_wte();
    d_wte_pad = A((size_t)NVpad*DM);
    CK(cuMemsetD8(d_wte_pad, 0, (size_t)NVpad*DM*sizeof(float)), "zero wte_pad");   /* zero pad rows */
    CK(cuMemcpyHtoD(d_wte_pad, &g_wbase[emb], (size_t)NV*DM*sizeof(float)), "h2d wte_pad");
    d_logits = A((size_t)Smax*NVpad);
    return 0;
}

/* ===================== modes ===================== */

/* --block0-dump <ids.txt> <out.bin>: run ONE layer (either arch) and dump the post-layer-0
 * residual rows 0..T as flat <f4 to out.bin. COMPARISON lives in the readable oracle
 * (llama_numpy_ref.py compare-block0) -- this side only computes and dumps (G-L1). */
static int run_block0_dump(const char* ids_path, const char* out_path) {
    int ids[1024];
    int T = read_ids_file(ids_path, ids, 1024);
    if (T <= 0) { fprintf(stderr, "no ids in %s\n", ids_path); return 2; }
    int S = ((T + 63) / 64) * 64;
    Spad = S;
    embed_gather(ids, T, S);
    if (ARCH) { upload_layer_ll(0); forward_layer_llama(); }
    else      { upload_layer(0);    forward_layer_gpt2(); }
    float* hx = (float*)malloc((size_t)T*DM*sizeof(float));
    CKX(cuMemcpyDtoH(hx, d_x, (size_t)T*DM*sizeof(float)), "d2h block0");
    FILE* of = fopen(out_path, "wb");
    if (!of) { fprintf(stderr, "open out '%s': %s\n", out_path, strerror(errno)); free(hx); return 2; }
    fwrite(hx, sizeof(float), (size_t)T*DM, of); fclose(of); free(hx);
    printf("BLOCK0_DUMP_OK arch=%s T=%d DM=%d -> %s\n", ARCH ? "llama" : "gpt2", T, DM, out_path);
    return 0;
}

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

/* ===================== --serve: persistent forward+generate worker (ADDITIVE) =====================
 * Does the expensive setup (device_init -> mint/load PTX + 8 kernel handles + weight mmap;
 * alloc_buffers; setup_head) ONCE, then loops reading one request frame per stdin line:
 *   {"prompt":"...","n_gen":N,"request_id":"..."}   (in-process tokenize via gpt2_tok), OR
 *   {"ids":[..],"n_gen":N,"request_id":"..."}        (pre-tokenized fallback), OR
 *   {"cmd":"quit"}                                    -> teardown.
 * For each request: tokenize -> emit hello+tokenize -> for each of n_gen steps run the UNCHANGED
 * forward_full (telemetry hooks fire inside) -> host argmax -> emit token -> emit done. The
 * numeric path is byte-identical to --generate; only printf-on-host-scope hooks were added. */

/* minimal request-frame fields parsed off one JSON line (only what serve needs). */
typedef struct {
    char* prompt; size_t prompt_len;
    int   ids[1024]; int n_ids;            /* pre-tokenized fallback */
    int   n_gen;
    int   is_quit;
} ServeReq;

/* find a top-level "key" and return a pointer just past the ':' (or NULL). */
static const char* json_find_key(const char* s, const char* key) {
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char* p = strstr(s, pat);
    if (!p) return NULL;
    p += strlen(pat);
    while (*p == ' ' || *p == '\t') p++;
    if (*p != ':') return NULL;
    p++;
    while (*p == ' ' || *p == '\t') p++;
    return p;
}

/* parse one request line. Returns 0 on success, 1 on bad json. */
static int parse_req_json(const char* line, ServeReq* r) {
    memset(r, 0, sizeof *r);
    r->n_gen = 20;
    const char* p;
    if ((p = json_find_key(line, "cmd")) && *p == '"' && strncmp(p + 1, "quit", 4) == 0) {
        r->is_quit = 1; return 0;
    }
    if ((p = json_find_key(line, "n_gen")) != NULL) r->n_gen = atoi(p);
    /* prompt (a JSON string with \n \t \" \\ escapes) */
    if ((p = json_find_key(line, "prompt")) != NULL && *p == '"') {
        p++;
        size_t cap = strlen(p) + 1; char* out = (char*)malloc(cap); size_t k = 0;
        while (*p && *p != '"') {
            if (*p == '\\' && p[1]) {
                p++;
                switch (*p) {
                    case 'n': out[k++] = '\n'; break;
                    case 't': out[k++] = '\t'; break;
                    case 'r': out[k++] = '\r'; break;
                    case '"': out[k++] = '"';  break;
                    case '\\': out[k++] = '\\'; break;
                    case '/': out[k++] = '/';  break;
                    case 'u': { /* \uXXXX -> UTF-8 (BMP only; sufficient for prompts) */
                        unsigned v = 0; for (int i = 0; i < 4 && p[1]; i++) { p++;
                            char h = *p; v <<= 4;
                            if (h>='0'&&h<='9') v|=(unsigned)(h-'0');
                            else if (h>='a'&&h<='f') v|=(unsigned)(h-'a'+10);
                            else if (h>='A'&&h<='F') v|=(unsigned)(h-'A'+10); }
                        if (v < 0x80) out[k++] = (char)v;
                        else if (v < 0x800) { out[k++] = (char)(0xC0|(v>>6)); out[k++] = (char)(0x80|(v&0x3F)); }
                        else { out[k++] = (char)(0xE0|(v>>12)); out[k++] = (char)(0x80|((v>>6)&0x3F)); out[k++] = (char)(0x80|(v&0x3F)); }
                        break;
                    }
                    default: out[k++] = *p; break;
                }
                p++;
            } else out[k++] = *p++;
        }
        out[k] = 0; r->prompt = out; r->prompt_len = k;
    }
    /* pre-tokenized ids fallback: "ids":[a,b,c] */
    if (!r->prompt && (p = json_find_key(line, "ids")) != NULL && *p == '[') {
        p++; int n = 0;
        while (*p && *p != ']' && n < 1024) {
            while (*p == ' ' || *p == ',') p++;
            if (*p == ']' || !*p) break;
            r->ids[n++] = atoi(p);
            while (*p && *p != ',' && *p != ']') p++;
        }
        r->n_ids = n;
    }
    if (!r->prompt && r->n_ids == 0) return 1;     /* nothing to run */
    return 0;
}

static int run_serve(const char* ptx_path, const char* wpath,
                     int emit_fd, int max_ctx, int timing, const char* detail,
                     const char* vocab, const char* merges) {
    emit_init(emit_fd, detail);
    g_timing = timing;
    if (device_init(ptx_path, wpath)) { emit_error("load", "device_init failed", 1); return 2; }
    int Smax = ((max_ctx + 63) / 64) * 64; if (Smax < 64) Smax = 64;
    if (alloc_buffers(Smax)) { emit_error("load", "alloc_buffers failed", 1); return 2; }
    if (setup_head(Smax))    { emit_error("load", "setup_head failed", 1);    return 2; }

    int in_proc_tok = (vocab && merges);
#ifdef GPT2_SERVE
    if (in_proc_tok) { build_byte_unicode(); load_vocab(vocab); load_merges(merges); }
#else
    in_proc_tok = 0;   /* tokenizer not linked in a plain single-file build */
#endif

    /* readiness line on stderr (the HTTP server's /api/health waits for this; stdout stays
     * pure newline-JSON telemetry so the server can pump it 1:1). */
    fprintf(stderr, "GPT2_SERVE_READY\n"); fflush(stderr);

    char* line = NULL; size_t cap = 0; ssize_t got;
    while ((got = getline(&line, &cap, stdin)) > 0) {
        ServeReq req;
        if (parse_req_json(line, &req)) { emit_error("load", "bad request json", 0); continue; }
        if (req.is_quit) { free(req.prompt); break; }

        int  T0; int* ids; int free_ids = 0;
        if (in_proc_tok && req.prompt) {
#ifdef GPT2_SERVE
            ids = encode_bytes((const unsigned char*)req.prompt, req.prompt_len, &T0); free_ids = 1;
#else
            ids = NULL; T0 = 0;
#endif
        } else if (req.n_ids > 0) {
            ids = req.ids; T0 = req.n_ids;
        } else {
            emit_error("tokenize", "no in-process tokenizer and no pre-tokenized ids", 0);
            free(req.prompt); continue;
        }
        if (!ids || T0 <= 0) { emit_error("tokenize", "empty/failed tokenization", 0);
            if (free_ids) free(ids); free(req.prompt); continue; }

        int Ngen = clampi(req.n_gen, 1, 256);
        if (((T0 + Ngen + 63) / 64) * 64 > Smax) {     /* honest bound, not a hang */
            emit_error("forward", "context exceeds --max-ctx; raise --max-ctx", 0);
            if (free_ids) free(ids); free(req.prompt); continue;
        }

        emit_hello();
        emit_tokenize(ids, T0, ((T0 + 63) / 64) * 64);

        /* working id buffer: prompt + room for Ngen generated ids. */
        int* work = (int*)malloc((size_t)(T0 + Ngen) * sizeof(int));
        memcpy(work, ids, (size_t)T0 * sizeof(int));
        int T = T0;
        float* logits = (float*)malloc((size_t)NV * sizeof(float));
        int* gen_ids  = (int*)malloc((size_t)Ngen * sizeof(int));
        int nonfinite = 0;
        double t0 = now_seconds();
        for (int step = 0; step < Ngen; step++) {
            g_emit_step = step;
            emit_forward_begin(step, T, ((T + 63) / 64) * 64, NL);
            forward_full(work, T, logits);             /* UNCHANGED arithmetic; hooks fire inside */
            for (int i = 0; i < NV; i++) if (!isfinite(logits[i])) { nonfinite = 1; break; }
            int nxt = argmax_row(logits, NV);
#ifdef GPT2_SERVE
            char* piece = decode_one(nxt);
#else
            char* piece = NULL;
#endif
            emit_token(step, nxt, piece ? piece : "", (double)logits[nxt], T + 1);
            free(piece);
            gen_ids[step] = nxt;
            work[T++] = nxt;
        }
        double secs = now_seconds() - t0;
        double tps = secs > 0 ? (double)Ngen / secs : 0.0;
#ifdef GPT2_SERVE
        char* full = decode_range(gen_ids, Ngen);
#else
        char* full = NULL;
#endif
        emit_done(T0, Ngen, T, secs, tps, full ? full : "", gen_ids, Ngen, nonfinite);
        free(full);

        free(work); free(logits); free(gen_ids);
        if (free_ids) free(ids);
        free(req.prompt);
    }
    free(line);
    return 0;
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
          "  %s <combined.ptx> <weights> --generate <N> <ref_ids.txt> [<ref_gen_ids.txt>]\n"
          "  %s <combined.ptx> <weights> --serve [--emit-fd N] [--max-ctx M] [--timing 0|1]\n"
          "                                      [--detail op|layer] [--vocab v.json --merges m.txt]\n",
          argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    const char* ptx_path = argv[1];
    const char* wpath    = argv[2];
    const char* mode     = argv[3];
    int rc = 2;

    /* a v2 (llama) weight file is self-describing: peek sets ARCH + all dims (incl. GQA/RoPE)
     * BEFORE any device or buffer setup. gpt2 v1 files leave the env-configured dims as-is. */
    if (peek_weights_header(wpath)) return 2;

    if (strcmp(mode, "--block0-dump") == 0) {
        if (argc < 6) { fprintf(stderr, "--block0-dump needs <ids.txt> <out.bin>\n"); return 2; }
        if (device_init(ptx_path, wpath)) return 2;
        if (alloc_buffers(128)) return 2;          /* prompt-sized: 128 padded rows */
        rc = run_block0_dump(argv[4], argv[5]);
    } else if (strcmp(mode, "--block0") == 0) {
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
    } else if (strcmp(mode, "--serve") == 0) {
        /* additive 4th mode: persistent forward+generate worker over stdin/stdout JSON. */
        int   emit_fd = 1, max_ctx = 320, timing = 0;
        const char* detail = "op";
        const char* vocab  = NULL;
        const char* merges = NULL;
        for (int i = 4; i < argc; i++) {
            if      (!strcmp(argv[i], "--emit-fd") && i+1 < argc) emit_fd = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--max-ctx") && i+1 < argc) max_ctx = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--timing")  && i+1 < argc) timing  = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--detail")  && i+1 < argc) detail  = argv[++i];
            else if (!strcmp(argv[i], "--vocab")   && i+1 < argc) vocab   = argv[++i];
            else if (!strcmp(argv[i], "--merges")  && i+1 < argc) merges  = argv[++i];
        }
        rc = run_serve(ptx_path, wpath, emit_fd, max_ctx, timing, detail, vocab, merges);
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
