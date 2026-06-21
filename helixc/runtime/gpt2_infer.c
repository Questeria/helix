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
int*  encode_bytes_special(const unsigned char* text, size_t n, int* out_n);  /* chat control tokens */
void  tok_enable_specials(void);
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
static int   QD  = 0;              /* query/attention dim = NH*DH (== DM for 8B/SmolLM2; != DM for Qwen3-32B) */
static float ROPE_THETA = 0.0f;
static int   EOS_ID = -1;
static int   g_specials_on = 0;    /* HX_SPECIALS: ChatML control-token tokenize (serve, instruct) */
static int   KV_ON = 0;            /* HX_KV: incremental decoding with device K/V caches (llama path) */
static int   RESIDENT = 0;         /* HX_RESIDENT: keep ALL layer weights on-device (small models) */
static int   g_fast = 0;           /* fast mode: skip per-op host syncs + per-op emits (same kernels, same ids) */          /* HX_EOS: stop greedy generation after emitting this id (chat eos; oracle convention: append then stop) */
static float RMS_EPS = 1e-5f;

/* ---- P1 weight-file header (must match helix-llm/tools/gpt2_import.py) ---- */
#define MAGIC   0x48584757u   /* 'HXGW' little-endian */
#define VERSION 1u
#define HDR_BYTES 64

/* ---- v1.6 HXGW v3 NVFP4 dequant (host) -- MIRRORS gpt2_pack.c quantizer/nvfp4_dequant_kernel.hx
 * EXACTLY (same codecs, same device formula) so the worker reconstructs byte-identical f32 to the
 * importer's verifier. v3 file = 64B hdr (VERSION=3@4, head_dim@56, n_tensors@60) + an n_tensors-entry
 * HXGWv3Desc table (32B each) + data. packed tensor: data_off->i32 words[rows*(Kpad/7)];
 * scale_off->E4M3 micro bytes[rows*(Kpad/16)] then ONE f32 per-tensor scale ts. */
typedef struct { uint32_t packed, rows, K, Kpad; uint64_t data_off, scale_off; } HXGWv3Desc;
static float v3_e4m3_decode(int c) {
    int s=(c>>7)&1, e=(c>>3)&15, m=c&7;
    if (e==15 && m==7) return nanf("");
    float mag; if (e==0) mag=(m/8.0f)*ldexpf(1.0f,-6); else mag=(1.0f+m/8.0f)*ldexpf(1.0f,e-7);
    return s ? -mag : mag;
}
/* dequant packed [rows x Kpad] NVFP4 -> out f32 (compact scales: micro byte per 16-block * ts). */
static void v3_dequant_packed(const int32_t* w, const uint8_t* micro, float ts, float* out, int rows, int Kpad) {
    int kwords=Kpad/7, kblk=Kpad/16;
    for (int r=0;r<rows;r++) for (int col=0;col<Kpad;col++){
        int word=col/7, slot=col-word*7;
        int wv = w[(long)r*kwords+word];
        for (int j=0;j<slot;j++) wv = wv/16;
        int code = wv - (wv/16)*16;
        int c8 = code - (code/8)*8;
        float magf = (c8==0)?0.0f:(c8==1)?0.5f:(c8==2)?1.0f:(c8==3)?1.5f:(c8==4)?2.0f:(c8==5)?3.0f:(c8==6)?4.0f:6.0f;
        float sm = (code/8==0)?magf:-magf;
        float eff = v3_e4m3_decode(micro[(long)r*kblk + col/16]) * ts;
        out[(long)r*Kpad+col] = sm * eff;
    }
}
/* ---- v3 RUN-PATH state + build-order indexing (set by peek/load when VERSION==3) ---- */
static int QWEN3 = 0;            /* HXGW v3 / Qwen3 family: untied head + per-head QK-norm */
static int HD_CFG = 0;           /* config head_dim (Qwen3 explicit @byte56; else DM/NH) */
static const HXGWv3Desc* g_desc = NULL;   /* per-tensor descriptor table (points into the mmap) */
static int g_ntensors = 0;
/* build-order index of layer L's tensor (mirrors gpt2_pack build_order_llama qwen3=1: 11/layer). */
enum { V3_INLN=0, V3_Q=1, V3_QN=2, V3_K=3, V3_KN=4, V3_V=5, V3_O=6, V3_POSTLN=7, V3_GATE=8, V3_UP=9, V3_DOWN=10 };
static int v3_idx_layer(int L, int t) { return L*11 + t; }
static int v3_idx_embed(void)  { return NL*11 + 0; }
static int v3_idx_norm(void)   { return NL*11 + 1; }
static int v3_idx_lmhead(void) { return NL*11 + 2; }

/* ===== v1.6 receipt: from-scratch FIPS-180-4 SHA-256 (copied verbatim from cuda_launch.c, the v1.5
 * #4 receipt spine; KAT-gated by sha256_selftest before any hash is trusted) ===== */
static const uint32_t SHA256_K[64] = {
  0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
  0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
  0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
  0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
  0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
  0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
  0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
  0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2
};
#define SHA256_ROR(x,n) (((x) >> (n)) | ((x) << (32 - (n))))
static void sha256_block(uint32_t st[8], const unsigned char* p) {
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = ((uint32_t)p[i*4] << 24) | ((uint32_t)p[i*4+1] << 16) | ((uint32_t)p[i*4+2] << 8) | (uint32_t)p[i*4+3];
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = SHA256_ROR(w[i-15],7) ^ SHA256_ROR(w[i-15],18) ^ (w[i-15] >> 3);
        uint32_t s1 = SHA256_ROR(w[i-2],17) ^ SHA256_ROR(w[i-2],19) ^ (w[i-2] >> 10);
        w[i] = w[i-16] + s0 + w[i-7] + s1;
    }
    uint32_t a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
    for (int i = 0; i < 64; i++) {
        uint32_t S1 = SHA256_ROR(e,6) ^ SHA256_ROR(e,11) ^ SHA256_ROR(e,25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + S1 + ch + SHA256_K[i] + w[i];
        uint32_t S0 = SHA256_ROR(a,2) ^ SHA256_ROR(a,13) ^ SHA256_ROR(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = S0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d; st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}
static void sha256(const unsigned char* data, size_t len, unsigned char out[32]) {
    uint32_t st[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    size_t full = len / 64;
    for (size_t i = 0; i < full; i++) sha256_block(st, data + i*64);
    unsigned char buf[128]; size_t rem = len - full*64;
    memcpy(buf, data + full*64, rem);
    buf[rem] = 0x80;
    size_t padlen = (rem < 56) ? 64 : 128;
    memset(buf + rem + 1, 0, padlen - rem - 1 - 8);
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) buf[padlen - 1 - i] = (unsigned char)(bits >> (8*i));
    sha256_block(st, buf);
    if (padlen == 128) sha256_block(st, buf + 64);
    for (int i = 0; i < 8; i++) { out[i*4]=(unsigned char)(st[i]>>24); out[i*4+1]=(unsigned char)(st[i]>>16); out[i*4+2]=(unsigned char)(st[i]>>8); out[i*4+3]=(unsigned char)st[i]; }
}
static void sha256_hex(const unsigned char dig[32], char out[65]) {
    static const char HX[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) { out[i*2] = HX[dig[i] >> 4]; out[i*2+1] = HX[dig[i] & 15]; }
    out[64] = 0;
}
/* SHA-256 a whole file via mmap (streams the big .weights file). Returns 1 + hex, or 0. */
static int sha256_file(const char* path, char hexout[65]) {
    int fd = open(path, O_RDONLY); if (fd < 0) return 0;
    struct stat st; if (fstat(fd, &st) != 0) { close(fd); return 0; }
    unsigned char* m = (unsigned char*)mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (m == MAP_FAILED) { close(fd); return 0; }
    unsigned char dig[32]; sha256(m, (size_t)st.st_size, dig); sha256_hex(dig, hexout);
    munmap(m, (size_t)st.st_size); close(fd); return 1;
}
/* 3 NIST FIPS-180-4 known-answer tests; MUST pass before any receipt hash is trusted. */
static int sha256_selftest(void) {
    struct { const char* msg; const char* want; } KAT[] = {
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq", "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
    };
    int bad = 0;
    for (int i = 0; i < 3; i++) {
        unsigned char dig[32]; char hex[65];
        sha256((const unsigned char*)KAT[i].msg, strlen(KAT[i].msg), dig); sha256_hex(dig, hex);
        if (strcmp(hex, KAT[i].want) != 0) { fprintf(stderr, "sha256 KAT %d FAIL got %s want %s\n", i, hex, KAT[i].want); bad++; }
    }
    printf("sha256_selftest: 3 NIST KAT, %d bad -> %s\n", bad, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}

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
/* KV-cache decode kernels (G-KV0-gated; loaded only when ARCH && KV_ON) */
static CUfunction f_gv_abt, f_gv_ab, f_smrow;
/* v1.7 INCREMENT 1: GPU NVFP4 dequant (the tiled sibling kernel). Default ON when HX_DQPTX points at
 * the compiled nvfp4_dequant_tiled.ptx; HX_HOSTDEQ=1 forces the v1.6 host-dequant path (the fallback
 * that keeps v1.6 behaviour one env away). d_packed/d_sc/d_dqscr are device scratch; g_scbuf is the
 * host-built effective per-16-block scale (e4m3_decode(micro)*ts) staged before the launch. */
static CUmodule   g_dqmod = 0;
static CUfunction f_dq_tiled = 0;
static int        g_gpu_dq = 0, g_hostdeq = 0;
static CUdeviceptr d_packed = 0, d_sc = 0, d_dqscr = 0;
static float*     g_scbuf = NULL;   /* pinned (cuMemAllocHost) v1.7 INC2b -> DMA HtoD for the effective scales */
static void*      g_hpin  = NULL;   /* v1.7 INC2b: pinned host bounce for the packed-words HtoD (DMA vs pageable staging) */
static float      g_e4m3_tab[256];   /* v1.7: e4m3_decode LUT (256 byte values) -> the per-forward scale build avoids ldexpf */
/* v1.7 INCREMENT 2: env-gated profiling (HX_PROF=1) of the per-forward upload breakdown -- splits the
 * v3_upload_gpu cost into mmap-touch+HtoD vs the CPU effective-scale build vs the dequant launch+sync. */
static int    g_prof = 0;
static double g_pf_htod = 0, g_pf_sb = 0, g_pf_dq = 0, g_pf_cmp = 0;
static long   g_pf_n = 0;
/* HX_DPROF=1: decode-compute category profiler (measure-before-wire for the fused-GEMV fixpoint move).
 * Splits per-token decode wall time into gemv_abt (the coalescing-fix candidate) vs per-layer dequant
 * (upload_layer_ll) vs the residual (norms/attn/rope/launches). Run with HX_FAST UNSET for clean per-op
 * attribution (every op syncs). Prints cumulative breakdown after each decode_step_llama. */
static double now_seconds(void);
static int    g_dprof = 0;
static int    g_genprof = 0;          /* HX_GENPROF: HX_FAST-honest end-to-end decode s/tok (one sync at each end of the decode loop) */
static double g_dp_gemv = 0, g_dp_dq = 0, g_dp_total = 0;
static double g_dp_lmh = 0;           /* T2/M7b: lm_head (dequant+gemv, or the fused kernel) wall, isolated from "other" */
static long   g_dp_n = 0;
static long   g_dqscr_elems = 0;   /* v1.7 INC2: element capacity of d_dqscr/g_scbuf (= alloc_buffers sc); sizes the chunked lm_head GPU dequant */
/* v1.7 INC2c: resident effective per-16-block scale cache, one device buffer per packed tensor idx.
 * The scales are STATIC (weights never change), so DECODE rebuilds + re-uploads them every token for
 * nothing. Opt-in via HX_SCALECACHE; built lazily on first touch; non-fatal on VRAM-full (falls back to
 * the per-forward scratch build). 8B layer scales ~1.74GB resident -> fits the 8GB card. */
static CUdeviceptr g_sc_res[400] = {0};   /* idx = L*11+t < 36*11 for 8B; 400 covers all packed tensors.
                                           * H4: when g_fusedgemv is ON, this holds the RAW e4m3 micro packed
                                           * 4-per-i32-word (rows*ceil(kblk/4) i32, ~4x smaller than the fp32
                                           * effective) for the fused kernel's in-kernel decode; the non-fused
                                           * f_dq_tiled path then builds fp32-effective into the scratch d_sc
                                           * each forward (NOT cached here). When fusedgemv is OFF, unchanged
                                           * (fp32 effective cache). */
static CUdeviceptr g_ts_res[400] = {0};   /* H4: per-tensor ts as a 1-elem f32 device buffer (the fused kernel
                                           * reads ts[0]); paired with the g_sc_res packed-micro entry. */
static int    g_scache = 0;
/* v1.8 INC1: keep the PACKED 4-bit layer words resident in VRAM (one buffer per packed tensor idx) so the
 * per-token packed-words HtoD -- the measured ~2-4s/forward bottleneck -- is eliminated. Opt-in HX_PACKEDRES
 * (implies HX_SCALECACHE). One-time cuMemGetInfo fit-check; per-tensor first-touch alloc; non-fatal -> the
 * existing pinned-bounce streaming path. NO kovc edit; byte-identical (same dequant reads the same bytes). */
static CUdeviceptr g_pk_res[400] = {0};
static int    g_packedres = 0;       /* HX_PACKEDRES requested */
static int    g_packedres_off = 0;   /* latched: residency disabled (fit-check or an alloc failed) -> stream */
static int    g_packedres_chk = 0;   /* the one-time fit-check has run */
static size_t g_resmargin = (size_t)768*1024*1024;  /* HX_RESMARGIN (MiB): VRAM headroom kept free of resident allocs (driver overhead + any lazy alloc). Tunable to fit full residency on a less-idle card. */
/* v1.8 INC1.5: keep the PACKED lm_head words+scales resident and dequant it in VOCAB-ROW chunks at use
 * time, instead of the persistent FP32 d_wte_pad ([NVpad x DM] ~2.49 GiB). Opt-in HX_PACKEDHEAD (implied
 * by HX_PACKEDRES). When ON+fitting, setup_head skips d_wte_pad and lm_head_logits_chunked() streams the
 * chunked dequant->GEMM/GEMV. Byte-identical: the SAME g_e4m3_tab[micro]*ts scales feed the SAME f_dq_tiled
 * into the SAME-math GEMM, just chunked over vocab rows. Non-fatal: alloc failure -> the FP32 d_wte_pad. */
static int    g_packedhead = 0;       /* HX_PACKEDHEAD requested */
static int    g_packedhead_on = 0;    /* latched: resident packed head active (allocs succeeded + fit) */
static CUdeviceptr g_phk_words = 0;   /* resident packed lm_head i32 words [rows x kwords] */
static CUdeviceptr g_phk_sc    = 0;   /* resident lm_head scales [rows x kblk]. H4: when g_fusedgemv is ON
                                       * this holds RAW e4m3 micro packed 4/i32-word (rows*ceil(kblk/4) i32),
                                       * decoded in-kernel by fused_lm_head; the non-fused head_dequant_chunk
                                       * fallback rebuilds fp32 from the mmap into scratch. OFF: fp32 effective. */
static CUdeviceptr g_phk_ts    = 0;   /* H4: per-tensor lm_head ts as a 1-elem f32 device buffer (fused path). */
static CUdeviceptr d_head_chunk = 0;  /* reusable dequant target [CHUNK x DM] f32 (compacted Kpad->DM) */
static CUdeviceptr d_head_temp  = 0;  /* prefill GEMM temp [Smax x CHUNK] f32 (scattered into d_logits cols) */
static int    g_head_chunk_rows = 0;  /* CHUNK = vocab rows per dequant pass (== the setup chunk count) */
static int    g_head_rows = 0, g_head_kwords = 0, g_head_kblk = 0, g_head_Kpad = 0;
static float  g_head_ts = 0;          /* per-tensor scale ts for the packed lm_head */
/* T2/M7 MEASUREMENT (2026-06-19): OPT-IN fused dequant-GEMV decode path. For the 7 per-layer
 * projections (q/k/v/o/gate/up/down) call __dequant_gemv_blockred on the RESIDENT packed words +
 * resident effective scales (one coalesced block-reduction pass: read packed 4-bit weights coalesced,
 * dequant inline, multiply x, SMEM tree-reduce -> y[n]). This SKIPS the per-token separate dequant of
 * those 7 weights (the measured ~52% of decode) + the f32 weight round-trip + the uncoalesced block=1
 * gemv (~26%). Default (HX_FUSEDGEMV unset) = the EXACT current path (dequant-on-upload + gemv_abt),
 * byte-identical. REQUIRES HX_PACKEDRES (the fused path reads the resident g_pk_res / g_sc_res); if
 * residency is off/latched-off it falls back to the normal path. Standalone PTX module (like HX_DQPTX),
 * NO kovc.hx edit. Acceptance gate = token-for-token identical generation vs the baseline. */
static int    g_fusedgemv = 0;        /* HX_FUSEDGEMV requested */
static int    g_fusedgemv_on = 0;     /* latched: fused path active (module loaded + residency available) */
static int    g_nofusehead = 0;       /* HX_NOFUSEHEAD: MEASUREMENT escape hatch -- keep the lm_head on the chunked
                                       * path even when the projections fuse (isolates the head's added win). */
/* H1 (2026-06-20): OPT-IN post-prefill free of the 7 f32 projection buffers (d_qW..d_downW, ~736MiB) at the
 * decode fused-latch, so the resident packed scales can SEAT and the committed fused path goes ACTIVE on a
 * ~7.1GB card (full residency is ~700MiB-1GB short until those dead-under-fused buffers are freed). PREFILL-
 * SAFE: those buffers are written FULL-SIZE by the prefill forward (v3_upload+mm_ABt), so we free them ONLY
 * at the first decode_step_llama (step>0, prefill done) AND only when fused is requested. If the scales then
 * seat -> g_fusedgemv_on=1 AUTHORITATIVE (the per-projection gemv_abt fallback that reads d_qW is unreachable
 * under fg). If they DON'T seat -> re-alloc the 7 buffers full-size (safety net) + restore residency state +
 * g_fusedgemv_on=0 (exactly the non-fused fallback as today). Requires HX_FUSEDGEMV + HX_PACKEDRES; default
 * OFF = byte-identical to today (no free, no re-make). */
static int    g_freeproj = 0;         /* HX_FREEPROJ requested */
static CUmodule   g_fgmod = 0;
static CUfunction f_dgemv = 0;        /* the dequant_gemv_blockred entry */
/* v1.8 FUSED DECODE-ATTENTION (the "other"-reclaim): one STANDALONE kernel (fused_decode_attn) replaces
 * the per-head decode attention loop (:1460-1473 -- NH x {scores gemv, scale, softmax, AV gemv} + NH
 * ctx D2D == 128 launches + 32 D2D / layer) with ONE launch/layer. Default OFF (HX_FUSEDATTN unset) =
 * the EXACT per-head path, byte-unchanged. Rides cuModuleLoadData EXACTLY like HX_DQPTX / HX_FUSEDGEMV;
 * NO kovc.hx edit, fixpoint UNTOUCHED. Scale is the HOST ATTN_SCALE passed via a 1-elem f32 array (no
 * on-device rsqrt). Acceptance gate = token-for-token identical generation vs OFF. */
static int        g_fusedattn = 0;    /* HX_FUSEDATTN requested */
static CUmodule   g_famod = 0;
static CUfunction f_fattn = 0;        /* the fused_decode_attn entry (0 => fall back to per-head) */
static int    g_fg_kpad_dm = 0;       /* Kpad of a [*, DM]  packed tensor (q/k/v/o/gate/up input pad) */
static int    g_fg_kpad_dff = 0;      /* Kpad of a [*, DFF] packed tensor (down input pad) */
static CUdeviceptr d_xn1pad = 0, d_xn21pad = 0, d_ctx1pad = 0, d_m1pad = 0;  /* Kpad-padded, tail-zeroed x for the fused gemv */

#define SYNC(w) do { CKX(cuCtxSynchronize(), w); } while (0)
#define LX(fn, grid, block, args)        do { CKX(cuLaunchKernel((fn),(grid),1,1,(block),1,1,0,0,(args),0), #fn); if (!g_fast) SYNC("sync " #fn); } while (0)
#define LX2(fn, gx, gy, bx, by, args)    do { CKX(cuLaunchKernel((fn),(gx),(gy),1,(bx),(by),1,0,0,(args),0), #fn); if (!g_fast) SYNC("sync " #fn); } while (0)

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
/* RoPE at an EXPLICIT position: same kernel, table base offset so buffer row 0 pairs with
 * table row `pos` (decode: the new token's single row at position pos). */
static void rope_at(CUdeviceptr q, int rows, int pos) {
    int half = DH/2;
    CUdeviceptr c = d_cos + (CUdeviceptr)((size_t)pos * half * sizeof(float));
    CUdeviceptr sn = d_sin + (CUdeviceptr)((size_t)pos * half * sizeof(float));
    void* ar[] = { &q,&c,&sn,&half }; LX(f_rope, (unsigned)rows, 1, ar);
}
/* SwiGLU gate y[i] = u[i]*silu(g[i]) over n elems. grid=n block=1 (G-L0 geometry). */
static void silu_mul_k(CUdeviceptr g, CUdeviceptr u, CUdeviceptr y, int n) {
    int nn=n; void* ar[] = { &g,&u,&y,&nn }; LX(f_silu, (unsigned)n, 1, ar);
}

/* ---- KV-decode wrappers (G-KV0 launch geometry: one thread per output, grid=N block=1) ---- */
/* y[1,N] = x[1,K] . W[N,K]^T  (covers every decode GEMM incl. attention scores vs cached K) */
static void gemv_abt(CUdeviceptr x, CUdeviceptr w, CUdeviceptr y, int N, int K) {
    int k=K; void* ar[] = { &x,&w,&y,&k };
    /* v1.7: block=1 ran each output on its own warp (1/32 lanes used). block=128 packs 128 outputs/block
     * -> full warps, ~32x the in-flight threads. Byte-identical (block size is pure scheduling; n=bid*bdim+tid).
     * Weight gemvs + lm_head have N%128==0 (QD/KVD/DFF, NVpad=128*1187); attention scores (N=T) keep block=1. */
    double _t0 = g_dprof ? now_seconds() : 0.0;
    if (N % 128 == 0) LX(f_gv_abt, (unsigned)(N/128), 128, ar);
    else              LX(f_gv_abt, (unsigned)N, 1, ar);
    if (g_dprof) { cuCtxSynchronize(); g_dp_gemv += now_seconds() - _t0; }
}
/* y[1,N] = p[1,T] . M[T,N]  (decode probs x cached V) */
static void gemv_ab(CUdeviceptr p, CUdeviceptr m, CUdeviceptr y, int N, int T) {
    int t=T,n=N; void* ar[] = { &p,&m,&y,&t,&n }; LX(f_gv_ab, (unsigned)N, 1, ar);
}
/* full-row softmax over [rows,cols], NO causal mask (the decode row attends every cached pos) */
static void smrow(CUdeviceptr x, CUdeviceptr y, int rows, int cols) {
    int c=cols; void* ar[] = { &x,&y,&c }; LX(f_smrow, (unsigned)rows, 1, ar);
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

/* dequant HXGW v3 tensor `idx` to host f32 (the RUN-path reader). packed -> [rows x Kpad] (NVFP4
 * dequant, padding cols are 0); f32 -> [rows x K] direct. Returns elem count. Caller sizes `out`
 * to at least rows*Kpad. Byte-identical to the importer's dequant (proven cross-tool in P3a). */
static long v3_get_tensor(int idx, float* out) {
    const HXGWv3Desc* d = &g_desc[idx];
    const unsigned char* base = (const unsigned char*)g_map;
    if (d->packed) {
        int rows=(int)d->rows, Kpad=(int)d->Kpad, kblk=Kpad/16;
        const int32_t* w = (const int32_t*)(base + d->data_off);
        const uint8_t* micro = (const uint8_t*)(base + d->scale_off);
        float ts; memcpy(&ts, base + d->scale_off + (size_t)rows*kblk, 4);
        v3_dequant_packed(w, micro, ts, out, rows, Kpad);
        return (long)rows*Kpad;
    }
    long n = (long)d->rows * d->K;
    memcpy(out, base + d->data_off, (size_t)n*4);
    return n;
}

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
    /* v1.7 INCREMENT 2b: prefetch the whole payload so the per-tensor uploads hit cached pages instead
     * of the slow random mmap-fault pattern (cold mmap-fault read was ~3x a sequential read). WILLNEED
     * (not SEQUENTIAL) keeps pages cached for decode's re-reads. Best-effort -- ignore the return. */
    madvise(g_map, g_maplen, MADV_WILLNEED);
    const unsigned char* hb = (const unsigned char*)g_map;
    uint32_t magic, ver, nl, dm, nh, nv, nc, dff; uint64_t nfloat;
    memcpy(&magic,&hb[0],4); memcpy(&ver,&hb[4],4); memcpy(&nl,&hb[8],4); memcpy(&dm,&hb[12],4);
    memcpy(&nh,&hb[16],4); memcpy(&nv,&hb[20],4); memcpy(&nc,&hb[24],4); memcpy(&dff,&hb[28],4);
    memcpy(&nfloat,&hb[32],8);
    if (magic != MAGIC) { fprintf(stderr, "bad magic 0x%08x (want 0x%08x)\n", magic, MAGIC); return 2; }
    if (ver == 3u) {   /* HXGW v3: descriptor-driven NVFP4 layout (peek already set the Qwen3 config) */
        if (!QWEN3) { fprintf(stderr, "v3 file but QWEN3 unset (peek skipped?)\n"); return 2; }
        if ((int)nl!=NL || (int)dm!=DM || (int)nh!=NH || (int)nv!=NV || (int)dff!=DFF) {
            fprintf(stderr, "v3 header dims (nl=%u dm=%u nh=%u nv=%u dff=%u) != peek config\n", nl,dm,nh,nv,dff); return 2;
        }
        g_desc = (const HXGWv3Desc*)(hb + HDR_BYTES);   /* descriptor table follows the 64B header */
        /* integrity: descriptors must tile the file CONTIGUOUSLY from data_base to EOF. */
        uint64_t cur = (uint64_t)HDR_BYTES + (uint64_t)g_ntensors*sizeof(HXGWv3Desc);
        for (int i=0;i<g_ntensors;i++){
            const HXGWv3Desc* d=&g_desc[i];
            if (d->data_off != cur) { fprintf(stderr, "v3 desc %d data_off %llu != %llu\n", i,(unsigned long long)d->data_off,(unsigned long long)cur); return 2; }
            if (d->packed) {
                int Kpad=(int)d->Kpad, kwords=Kpad/7, kblk=Kpad/16;
                uint64_t wb=(uint64_t)d->rows*kwords*4;
                if (d->scale_off != cur+wb) { fprintf(stderr, "v3 desc %d scale_off mismatch\n", i); return 2; }
                cur += wb + (uint64_t)d->rows*kblk + 4;
            } else cur += (uint64_t)d->rows*d->K*4;
        }
        if (cur != g_maplen) { fprintf(stderr, "v3 layout %llu != file %zu\n", (unsigned long long)cur, g_maplen); return 2; }
        g_wbase = (const float*)(hb + HDR_BYTES);   /* non-NULL; v3 compute addresses via g_desc */
        printf("[wt] mmap %zu B, v3 NVFP4: %d tensors, descriptor table validated (tiles file exactly)\n", g_maplen, g_ntensors);
        return 0;
    }
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
        DH = DM / NH; KVD = NKV * DH; QD = NH * DH;
        ATTN_SCALE = 1.0f / sqrtf((float)DH);
        if (NKV <= 0 || NH % NKV) { fprintf(stderr, "peek: bad GQA heads %d/%d\n", NH, NKV); return 2; }
        /* the rmsnorm kernel BAKES eps=1e-5 (G-L0-verified); fail closed on any other eps
         * rather than run silently-wrong numerics. */
        if (fabsf(RMS_EPS - 1e-5f) > 1e-9f) { fprintf(stderr, "peek: rms_eps %g != the kernel's baked 1e-5\n", (double)RMS_EPS); return 2; }
        printf("[peek] llama v2 header: NL=%d DM=%d NH=%d NKV=%d NV=%d NC=%d DFF=%d theta=%g\n",
               NL, DM, NH, NKV, NV, NC, DFF, (double)ROPE_THETA);
    } else if (ver == 3u) {   /* v1.6 HXGW v3: Qwen3 (untied head + per-head QK-norm), NVFP4-packed */
        uint32_t nl,dm,nh,nv,nc,dff,arch,nkv,hd,ntens; float th, ep;
        memcpy(&nl,hb+8,4); memcpy(&dm,hb+12,4); memcpy(&nh,hb+16,4); memcpy(&nv,hb+20,4);
        memcpy(&nc,hb+24,4); memcpy(&dff,hb+28,4);
        memcpy(&arch,hb+40,4); memcpy(&nkv,hb+44,4); memcpy(&th,hb+48,4); memcpy(&ep,hb+52,4);
        memcpy(&hd,hb+56,4); memcpy(&ntens,hb+60,4);
        if (arch != 1u) { fprintf(stderr, "peek: v3 header with unknown arch %u\n", arch); return 2; }
        ARCH = 1; QWEN3 = 1;
        NL=(int)nl; DM=(int)dm; NH=(int)nh; NV=(int)nv; NC=(int)nc; DFF=(int)dff;
        NKV=(int)nkv; ROPE_THETA=th; RMS_EPS=ep; HD_CFG=(int)hd; g_ntensors=(int)ntens;
        DH = HD_CFG > 0 ? HD_CFG : DM / NH; KVD = NKV * DH; QD = NH * DH;
        ATTN_SCALE = 1.0f / sqrtf((float)DH);
        if (NKV <= 0 || NH % NKV) { fprintf(stderr, "peek: bad GQA heads %d/%d\n", NH, NKV); return 2; }
        /* the rmsnorm kernel BAKES eps=1e-5; Qwen3 uses 1e-6. The gap is ~1e-6/mean(x^2), far below
         * the f32 + ~9.5% NVFP4 envelope, so ACCEPT it (vs the v2 fail-close) with a note. */
        if (fabsf(RMS_EPS - 1e-5f) > 1e-9f)
            fprintf(stderr, "[peek] note: rms_eps=%g; the kernel bakes 1e-5 (diff negligible vs the quant envelope)\n", (double)RMS_EPS);
        printf("[peek] qwen3 v3 header: NL=%d DM=%d NH=%d NKV=%d HD=%d NV=%d NC=%d DFF=%d theta=%g eps=%g n_tensors=%d\n",
               NL, DM, NH, NKV, DH, NV, NC, DFF, (double)ROPE_THETA, (double)RMS_EPS, g_ntensors);
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
static void upload_layer_ll_fused(int L);                            /* T2/M7: fused-decode upload (norms only) */
static int  fused_gemv(int idx, CUdeviceptr d_xpad, CUdeviceptr d_y, int N);  /* T2/M7: fused dequant-GEMV projection */
static int  fused_lm_head(CUdeviceptr d_xpad, int xpad_kpad, CUdeviceptr d_out);  /* T2/M7b: fused dequant-GEMV lm_head */
static int  v3_resident_ptrs(int idx, CUdeviceptr* pk, CUdeviceptr* sc, CUdeviceptr* tsp, int* rows_out, int* kpad_out);  /* T2/M7+H4: resident packed/micro-scale/ts ptrs */
static int  read_ids_file(const char* path, int* ids, int maxn);     /* defined with the modes */
static CUdeviceptr d_qW, d_kW, d_vW, d_oW, d_gateW, d_upW, d_downW;  /* layer weights */
static CUdeviceptr d_qnorm = 0, d_knorm = 0;   /* Qwen3 per-head QK-norm weights [DH] (v3 only) */
static float* g_v3scratch = NULL;              /* reusable host dequant buffer (max layer tensor) */
static CUdeviceptr d_q, d_k, d_v;    /* q/k/v GEMM outputs (carved from d_qkv) */
static CUdeviceptr d_mlp1c;          /* silu_mul output [Spad,DFF] */
/* ---- KV-cache decode state (HX_KV=1; llama path only) ---- */
static CUdeviceptr d_kcache = 0, d_vcache = 0;  /* [NL][NKV][kv_cap][DH] roped K / V (device) */
static int kv_cap = 0;               /* cache rows per (layer, kv-head) == Smax */
static int kv_len = 0;               /* valid cached positions (== tokens processed) */
/* decode (M=1) work buffers */
static CUdeviceptr d_x1, d_xn1, d_q1, d_k1, d_v1, d_sc1, d_pr1, d_ao1, d_ctx1, d_pj1, d_xn21, d_g1, d_u1, d_m1, d_mo1, d_lg1;
/* resident weights (HX_RESIDENT=1; small models): per-layer device copies, ptr-swap on use */
#define MAX_RES_LAYERS 64
static CUdeviceptr res_inln[MAX_RES_LAYERS], res_qW[MAX_RES_LAYERS], res_kW[MAX_RES_LAYERS],
                   res_vW[MAX_RES_LAYERS], res_oW[MAX_RES_LAYERS], res_postln[MAX_RES_LAYERS],
                   res_gateW[MAX_RES_LAYERS], res_upW[MAX_RES_LAYERS], res_downW[MAX_RES_LAYERS];
static int res_loaded = 0;
/* cache slab offset (floats) for (layer, kv-head, position 0) */
static size_t kvoff(int L, int kv) { return (((size_t)L * NKV + (size_t)kv) * (size_t)kv_cap) * (size_t)DH; }
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
static void scatter_head(CUdeviceptr dst, int hbase, CUdeviceptr src, int dststride) {
    for (int s = 0; s < Spad; s++) {
        CUdeviceptr s_src = src + (CUdeviceptr)((size_t)(s*DH) * sizeof(float));
        CUdeviceptr s_dst = dst + (CUdeviceptr)((size_t)(s*dststride + hbase) * sizeof(float));
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
    if (g_fast) return;
    if (!g_serve) return;
    char b[128];
    snprintf(b, sizeof b, "{\"_ev\":\"embed\",\"seq\":%ld,\"step\":%d,\"t\":%d,\"d_model\":%d}\n",
             g_seq++, step, t, d_model);
    emit_raw(b);
}
static void emit_layer_begin(int step, int idx, int total) {
    if (g_fast) return;   /* fast mode: token-level heartbeat only (forward_begin/token/done) */
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
    if (!g_serve || g_fast || g_detail < DETAIL_OP) return;
    char b[256];
    snprintf(b, sizeof b,
      "{\"_ev\":\"op\",\"seq\":%ld,\"step\":%d,\"layer\":%d,\"seq_in_layer\":%d,"
      "\"kernel\":\"%s\",\"phase\":\"%s\",\"label\":\"%s\",\"agg\":%d}\n",
      g_seq++, g_emit_step, layer, seq_in_layer, kernel, phase, label, agg);
    emit_raw(b); g_layer_ops++;
}
static void emit_layer_end(int step, int idx, double ms) {
    if (g_fast) return;
    if (!g_serve) return;
    char b[160];
    snprintf(b, sizeof b, "{\"_ev\":\"layer_end\",\"seq\":%ld,\"step\":%d,\"idx\":%d,\"ms\":%.3f,\"ops\":%d}\n",
             g_seq++, step, idx, ms, g_layer_ops);
    emit_raw(b);
}
static void emit_head(int step, const char* label, const char* kernel) {
    if (g_fast) return;
    if (!g_serve) return;
    char b[160];
    snprintf(b, sizeof b, "{\"_ev\":\"head\",\"seq\":%ld,\"step\":%d,\"label\":\"%s\",\"kernel\":\"%s\"}\n",
             g_seq++, step, label, kernel);
    emit_raw(b);
}
/* ===================== GLASS-BOX TOKEN TELEMETRY + LOGIT LENS (W3/W4, 2026-06) =====================
 * ALL ADDITIVE: extra FIELDS on the existing "token" event (never a new event name; the
 * fast path emits none of this). Values are REAL host-side derivations of the same logits
 * the pick consumed; the lens re-uses the gated rms+gemv kernels read-only between layers.
 * The lens is opt-in per request ("lens":1), glass-box only, and SLOWER (one extra
 * NVpad-row GEMV + D2H per layer per token) -- the UI labels it so. Lens parity has its
 * own gate leg (G-LENS: per-layer top-1 ids vs the oracle's dump-lens). */
static const char* json_find_key(const char* s, const char* key);   /* fwd (serve module) */
static int    g_lens_on = 0;                     /* per-request: "lens":1 */
static int    g_lens_nl = 0;                     /* layers captured this step */
static int    g_lens_top[MAX_RES_LAYERS][5];     /* per-layer top-5 ids */
static double g_lens_p[MAX_RES_LAYERS][5];       /* per-layer top-5 softmax probs */
static int    g_attn_nh = 0;                     /* heads captured (last layer) */
static int    g_attn_pos[32][8];                 /* per-head top-8 attended positions */
static double g_attn_w[32][8];                   /* per-head top-8 attention weights */
static float* g_lens_host = NULL;                /* [NVpad] D2H scratch */
static char   g_tok_extra[16384];                /* pre-rendered JSON fragment for emit_token */

/* host top-k of a float row (k small): ids by (-v, id) pinned */
static void host_top5(const float* v, int n, int* oid, double* op) {
    for (int s = 0; s < 5; s++) { oid[s] = -1; op[s] = -1e30; }
    for (int i = 0; i < n; i++) {
        double x = (double)v[i];
        for (int s = 0; s < 5; s++) {
            if (x > op[s]) {
                for (int t = 4; t > s; t--) { op[t] = op[t-1]; oid[t] = oid[t-1]; }
                op[s] = x; oid[s] = i; break;
            }
        }
    }
}
static void host_top8(const float* v, int n, int* oid, double* op) {
    for (int s = 0; s < 8; s++) { oid[s] = -1; op[s] = 0.0; }
    for (int i = 0; i < n; i++) {
        double x = (double)v[i];
        for (int s = 0; s < 8; s++) {
            if (oid[s] < 0 || x > op[s]) {
                for (int t = 7; t > s; t--) { op[t] = op[t-1]; oid[t] = oid[t-1]; }
                op[s] = x; oid[s] = i; break;
            }
        }
    }
}
static float g_attn_host[512];   /* one attention row (T <= Smax <= 512) */
/* softmax-prob the 5 tops against the full row (double, max-subtract; stats only) */
static void soft5(const float* row, int n, const int* oid, double* op) {
    double mx = -1e30; for (int i = 0; i < n; i++) if ((double)row[i] > mx) mx = (double)row[i];
    double s = 0.0;     for (int i = 0; i < n; i++) s += exp((double)row[i] - mx);
    for (int t = 0; t < 5; t++) op[t] = (oid[t] >= 0) ? exp((double)row[oid[t]] - mx) / s : 0.0;
}
/* chosen-token prob + entropy of the full distribution (double; stats only) */
static void tok_stats(const float* row, int n, int chosen, double* p_out, double* h_out) {
    double mx = -1e30; for (int i = 0; i < n; i++) if ((double)row[i] > mx) mx = (double)row[i];
    double s = 0.0;     for (int i = 0; i < n; i++) s += exp((double)row[i] - mx);
    double H = 0.0;
    for (int i = 0; i < n; i++) { double p = exp((double)row[i] - mx) / s; if (p > 1e-12) H -= p * log(p); }
    *p_out = exp((double)row[chosen] - mx) / s; *h_out = H;
}
/* render the additive token-event fields for this step (call AFTER the pick; serve+glassbox only) */
static void tok_extra_render(const float* logits, int chosen) {
    g_tok_extra[0] = 0;
    if (g_fast || !g_serve) return;
    double p, H; tok_stats(logits, NV, chosen, &p, &H);
    int aid[5]; double ap[5];
    host_top5(logits, NV, aid, ap); soft5(logits, NV, aid, ap);
    size_t k = 0, cap = sizeof g_tok_extra;
    k += (size_t)snprintf(g_tok_extra + k, cap - k, ",\"p\":%.5f,\"H\":%.3f,\"alts\":[", p, H);
    for (int t = 0; t < 5; t++) {
        char esc[64] = "";
#ifdef GPT2_SERVE
        if (aid[t] >= 0) { char* pc = decode_one(aid[t]); if (pc) { jesc(pc, esc, sizeof esc); free(pc); } }
#endif
        k += (size_t)snprintf(g_tok_extra + k, cap - k, "%s[%d,\"%s\",%.5f]", t ? "," : "", aid[t], esc, ap[t]);
    }
    k += (size_t)snprintf(g_tok_extra + k, cap - k, "]");
    if (g_lens_on && g_lens_nl > 0) {
        k += (size_t)snprintf(g_tok_extra + k, cap - k, ",\"lens\":[");
        for (int L = 0; L < g_lens_nl; L++) {
            k += (size_t)snprintf(g_tok_extra + k, cap - k, "%s[", L ? "," : "");
            for (int t = 0; t < 5; t++)
                k += (size_t)snprintf(g_tok_extra + k, cap - k, "%s[%d,%.4f]", t ? "," : "", g_lens_top[L][t], g_lens_p[L][t]);
            k += (size_t)snprintf(g_tok_extra + k, cap - k, "]");
        }
        k += (size_t)snprintf(g_tok_extra + k, cap - k, "]");
    }
    if (g_lens_on && g_attn_nh > 0) {
        k += (size_t)snprintf(g_tok_extra + k, cap - k, ",\"attn\":[");
        for (int h = 0; h < g_attn_nh; h++) {
            k += (size_t)snprintf(g_tok_extra + k, cap - k, "%s[", h ? "," : "");
            for (int t = 0; t < 8; t++)
                k += (size_t)snprintf(g_tok_extra + k, cap - k, "%s[%d,%.4f]", t ? "," : "", g_attn_pos[h][t], g_attn_w[h][t]);
            k += (size_t)snprintf(g_tok_extra + k, cap - k, "]");
        }
        k += (size_t)snprintf(g_tok_extra + k, cap - k, "]");
    }
    g_lens_nl = 0; g_attn_nh = 0;
}
/* ---- stop sequences (per request; up to 4 x 31 bytes; matched on the decoded tail) ---- */
static char g_stops[4][32]; static int g_nstops = 0;
static char g_tail[256]; static int g_tail_len = 0;
static void stops_parse(const char* line) {
    g_nstops = 0; g_tail_len = 0; g_tail[0] = 0;
    const char* p = json_find_key(line, "stop");
    if (!p || *p != '[') return;
    p++;
    while (g_nstops < 4) {
        while (*p == ' ' || *p == ',') p++;
        if (*p != '"') break;
        p++;
        int k = 0;
        while (*p && *p != '"' && k < 31) {
            if (*p == '\\' && p[1]) { p++; g_stops[g_nstops][k++] = (*p == 'n') ? '\n' : (*p == 't') ? '\t' : *p; p++; }
            else g_stops[g_nstops][k++] = *p++;
        }
        while (*p && *p != '"') p++;
        if (*p == '"') p++;
        g_stops[g_nstops][k] = 0;
        if (k > 0) g_nstops++;
        while (*p == ' ') p++;
        if (*p != ',') break;
    }
}
/* append a decoded piece to the tail ring; return 1 if any stop sequence just completed */
static int stops_hit(const char* piece) {
    if (g_nstops == 0 || !piece) return 0;
    size_t pl = strlen(piece);
    if (pl >= sizeof g_tail) { memcpy(g_tail, piece + (pl - sizeof g_tail + 1), sizeof g_tail - 1); g_tail_len = (int)sizeof g_tail - 1; }
    else {
        if (g_tail_len + (int)pl >= (int)sizeof g_tail) {
            int keep = (int)sizeof g_tail - 1 - (int)pl;
            memmove(g_tail, g_tail + g_tail_len - keep, (size_t)keep); g_tail_len = keep;
        }
        memcpy(g_tail + g_tail_len, piece, pl); g_tail_len += (int)pl;
    }
    g_tail[g_tail_len] = 0;
    for (int i = 0; i < g_nstops; i++) {
        size_t sl = strlen(g_stops[i]);
        if ((size_t)g_tail_len >= sl && memcmp(g_tail + g_tail_len - sl, g_stops[i], sl) == 0) return 1;
    }
    return 0;
}

static void emit_token(int step, int id, const char* string, double logit, int context_len) {
    if (!g_serve) return;
    char esc[256]; jesc(string ? string : "", esc, sizeof esc);
    static char b[20480];   /* token + staged glass-box extras */
    snprintf(b, sizeof b,
      "{\"_ev\":\"token\",\"seq\":%ld,\"step\":%d,\"id\":%d,\"string\":\"%s\",\"logit\":%.5f,\"context_len\":%d%s}\n",
      g_seq++, step, id, esc, logit, context_len, g_tok_extra);
    g_tok_extra[0] = 0;
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
        scatter_head(d_ctx, qb, d_aoh, DM);   /* gpt2: d_ctx row-stride = DM */
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
        /* llama: token embedding ONLY (RoPE replaces positional embeddings) -- pure row copy.
         * v3/Qwen3: embed_tokens is an f32 v3 tensor -> address it via its descriptor (gather just
         * the T needed rows host-side; never upload the whole 2.5GB table). */
        const float* emb = QWEN3
            ? (const float*)((const unsigned char*)g_map + g_desc[v3_idx_embed()].data_off)
            : &g_wbase[off_embed_ll()];
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
    mm_ABt(d_xn, d_qW, d_q, Spad, DM, QD);   /* q_proj: [Spad,DM] @ [QD,DM]^T -> [Spad,QD] */
    emit_op(L, 2, "tiled_matmul_abt", "attn", "k_gemm", 1);
    mm_ABt(d_xn, d_kW, d_k, Spad, DM, KVD);
    emit_op(L, 3, "tiled_matmul_abt", "attn", "v_gemm", 1);
    mm_ABt(d_xn, d_vW, d_v, Spad, DM, KVD);
    for (int h = 0; h < NH; h++) {
        int kv = h / group;               /* the pinned GQA mapping (oracle: gqa_kv_head) */
        pack_head(d_Qh, d_q, h*DH,  QD);   /* d_q row-stride = QD (= NH*DH) */
        pack_head(d_Kh, d_k, kv*DH, KVD);
        pack_head(d_Vh, d_v, kv*DH, KVD);
        if (QWEN3) {   /* Qwen3 per-head QK-norm: RMSNorm over DH (one shared weight), BEFORE RoPE */
            rms_norm_k(d_Qh, d_Qh, d_qnorm, Spad, DH);
            rms_norm_k(d_Kh, d_Kh, d_knorm, Spad, DH);
        }
        rope_k(d_Qh, Spad);               /* rotate_half RoPE in-place, position == row */
        rope_k(d_Kh, Spad);
        mm_ABt(d_Qh, d_Kh, d_scores, Spad, DH, Spad);
        scale_rt(d_scores, d_scale, Spad*Spad);
        softmax_causal(d_scores, d_attnw, Spad, Spad);
        mm_AB(d_attnw, d_Vh, d_aoh, Spad, Spad, DH);
        scatter_head(d_ctx, h*DH, d_aoh, QD);   /* d_ctx row-stride = QD (= NH*DH) */
        if (KV_ON && (h % group) == 0) {
            /* PREFILL CAPTURE: d_Kh holds kv-head kv's ROPED K rows [Spad,DH]; d_Vh its V.
             * Pad rows land beyond kv_len and are overwritten by later appends -- decode
             * only ever reads rows [0, kv_len). */
            CKX(cuMemcpyDtoD(d_kcache + (CUdeviceptr)(kvoff(L, kv) * sizeof(float)), d_Kh, (size_t)Spad*DH*sizeof(float)), "kv capture K");
            CKX(cuMemcpyDtoD(d_vcache + (CUdeviceptr)(kvoff(L, kv) * sizeof(float)), d_Vh, (size_t)Spad*DH*sizeof(float)), "kv capture V");
        }
    }
    emit_op(L, 4, "gpu_rope_rot",       "attn", "rope_qk",      2*NH);
    emit_op(L, 5, "tiled_matmul_abt",   "attn", "attn_scores",  NH);
    emit_op(L, 6, "gpu_scale_rt",       "attn", "attn_scale",   NH);
    emit_op(L, 7, "gpu_softmax_causal", "attn", "attn_softmax", NH);
    emit_op(L, 8, "tiled_matmul",       "attn", "attn_av",      NH);
    emit_op(L, 9, "tiled_matmul_abt",   "attn", "attn_proj", 1);
    mm_ABt(d_ctx, d_oW, d_proj, Spad, QD, DM);   /* o_proj: [Spad,QD] @ [DM,QD]^T -> [Spad,DM] */
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

/* v1.8 INC1.5: forward decl (defined after v3_upload_gpu, which it mirrors); used by the head call sites. */
static void lm_head_logits_chunked(CUdeviceptr d_in, CUdeviceptr d_out, int M, int is_gemv);

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
    if (ARCH && KV_ON) kv_len = T;   /* prefill done: T real positions cached */
    if (ARCH) {
        emit_head(g_emit_step, "norm_f", "gpu_rmsnorm_fwd_eps");
        rms_norm_k(d_x, d_xn, d_lnfg, S, DM);  /* final RMSNorm (llama has no final bias) */
    } else {
        emit_head(g_emit_step, "ln_f", "gpu_layernorm_fwd_eps");
        ln_eps(d_x, d_xn, d_lnfg, d_lnfb, S, DM);  /* final LayerNorm (reuse d_xn as ln_f output) */
    }
    /* tied head: logits[S,NVpad] = xnorm[S,DM] @ wte_pad[NVpad,DM]^T (wte rows ARE the tied head). */
    emit_head(g_emit_step, "lm_head", "tiled_matmul_abt");
    if (g_packedhead_on) lm_head_logits_chunked(d_xn, d_logits, S, 0);   /* v1.8 INC1.5: chunked packed head */
    else                 mm_ABt(d_xn, d_wte_pad, d_logits, S, DM, NVpad);
    /* copy ONLY the last real-token row's first NV logits to the host (greedy decision row). */
    CKX(cuMemcpyDtoH(out_last_logits, d_logits + (CUdeviceptr)((size_t)(T-1)*NVpad*sizeof(float)),
                     (size_t)NV*sizeof(float)), "d2h last-row logits");
}

/* ===================== KV-CACHE incremental decode (HX_KV=1; llama only) =====================
 * One step: forward ONLY the new token at position `pos` against the cached K/V.
 * Greedy-identical to the full re-forward BY CONSTRUCTION (same weights, same math per
 * position); the A/B leg of the model gate PROVES the ids match. Decode GEMMs use the
 * G-KV0-gated gpu_gemv_* kernels (M=1, no %64 constraint); softmax is the full-row form
 * (every cached position is attendable -- causality lives in WHAT is cached). */
static void decode_step_llama(int new_id, int pos, float* out_logits) {
    double _td = g_dprof ? now_seconds() : 0.0;
    int T = pos + 1;
    int group = NH / NKV;
    /* v3 (QWEN3) lays the embedding out via the descriptor table, not the linear off_embed_ll() offset;
     * decode used the non-v3 path unconditionally -> OOB host read -> segfault. Mirror embed_gather(). */
    const float* emb = QWEN3
        ? (const float*)((const unsigned char*)g_map + g_desc[v3_idx_embed()].data_off)
        : &g_wbase[off_embed_ll()];
    CKX(cuMemcpyHtoD(d_x1, &emb[(size_t)new_id*DM], (size_t)DM*sizeof(float)), "h2d x1");
    /* T2/M7: latch the fused-gemv path once. Active iff requested + module loaded + QWEN3 + residency is
     * available (prefill or v3_resident_ptrs first-touch makes the 7 weights resident). The padded x
     * buffers d_*pad carry the K-logical x with a zeroed [K,Kpad) tail for the full-Kpad fused sum. */
    static int fg_latched = 0;
    if (g_fusedgemv && !fg_latched) {
        fg_latched = 1;
        if (getenv("HX_DBG")) {   /* H1 census: free at decode-start + total resident words/scales need */
            size_t _fr=0,_to=0; cuMemGetInfo(&_fr,&_to);
            size_t nw=0, nsc=0; int nres_w=0, nres_s=0;
            for (int i=0;i<g_ntensors;i++){ const HXGWv3Desc* d=&g_desc[i]; if(!d->packed||i==v3_idx_lmhead())continue;
                int Kp=(int)d->Kpad, kblk=Kp/16; nw+=(size_t)d->rows*(Kp/7)*4;
                nsc += g_fusedgemv ? (size_t)d->rows*((kblk+3)/4)*4   /* H4: resident scale = packed e4m3 micro (~4x smaller) */
                                   : (size_t)d->rows*kblk*4;          /* fp32 effective */
                if(i<400&&g_pk_res[i])nres_w++; if(i<400&&g_sc_res[i])nres_s++; }
            fprintf(stderr,"[h1census] decode-start free=%zuMiB  resident-need words=%zuMiB scales=%zuMiB (sum=%zuMiB, scale=%s)  already-resident words=%d/%d scales=%d/%d  resmargin=%zuMiB\n",
                _fr/1048576, nw/1048576, nsc/1048576, (nw+nsc)/1048576, g_fusedgemv?"raw-micro":"fp32", nres_w, 7*NL, nres_s, 7*NL, g_resmargin/1048576);
        }
        /* H1 (HX_FREEPROJ): PREFILL IS DONE at this first decode step (callers run decode_step_llama only at
         * step>0; step 0 ran the full prefill forward). Under FUSED decode the 7 f32 projection buffers
         * d_qW..d_downW (~736MiB) are DEAD: upload_layer_ll_fused uploads norms only, and every per-projection
         * gemv_abt(...,d_qW,...) fallback is guarded by `!fg || !fused_gemv(...)` -> unreachable once fg is on.
         * Prefill needed them full-size (v3_upload + mm_ABt), but now they're freeable. Freeing them +736MiB
         * lets the resident scales (which latched g_packedres_off=1 mid-prefill at the ~768MiB margin) SEAT.
         * Snapshot the buffers so a non-seat can re-alloc them (the safety net keeps a non-seat correctness-safe). */
        int sv_resoff = g_packedres_off;            /* restore exactly if we don't seat */
        size_t sv_margin = g_resmargin;             /* restore after the (scoped-margin) re-make */
        int freed = 0, freed_scr = 0;
        /* Whether the fused HEAD will own the lm_head (so the chunked-head fallback -- the ONLY decode user of
         * the streaming scratch d_dqscr/d_packed/d_sc -- can't fire). If so, that scratch (~240MiB) is also dead
         * under full-resident+fused and can be freed UP FRONT to help the remaining projections seat. */
        int head_will_fuse = (g_packedhead_on && !g_nofusehead);
        if (g_freeproj) {
            /* PREFILL DONE: free the 7 dead f32 projection buffers (~736MiB). They were full-size-live through
             * prefill (v3_upload + mm_ABt); under fused decode upload_layer_ll_fused skips them and every
             * gemv_abt(...,d_qW,...) fallback is `!fg`-guarded -> unreachable. d_qW..d_downW already hold layer
             * NL-1's data (per-layer scratch) -- no state to preserve, just free. */
            cuMemFree(d_qW); cuMemFree(d_kW); cuMemFree(d_vW); cuMemFree(d_oW);
            cuMemFree(d_gateW); cuMemFree(d_upW); cuMemFree(d_downW);
            d_qW=d_kW=d_vW=d_oW=d_gateW=d_upW=d_downW=0;
            freed = 1;
            /* Also free the streaming dequant scratch up front WHEN the fused head will own lm_head. v3_resident_ptrs
             * (the re-make) does NOT use d_dqscr/d_packed/d_sc (it bounces through g_hpin/g_scbuf), so freeing them
             * now is safe and recovers ~240MiB toward seating. The non-seat path re-allocs them for the fallback. */
            if (head_will_fuse) {
                if (d_dqscr) { cuMemFree(d_dqscr); d_dqscr = 0; }
                if (d_packed){ cuMemFree(d_packed); d_packed = 0; }
                if (d_sc)    { cuMemFree(d_sc); d_sc = 0; }
                freed_scr = 1;
            }
            g_packedres_off = 0;                    /* freed VRAM -> re-enable residency so the remaining tensors can seat */
            /* Scope a SMALLER per-alloc headroom for the re-make only: prefill's larger g_resmargin (its job was to
             * leave room for prefill transients) over-reserves now that prefill is done and the decode buffers/KV
             * are already allocated. A small tail margin lets the last few tensors seat. Tunable via HX_REMARGIN. */
            { const char* _r = getenv("HX_REMARGIN"); long _v = _r ? atol(_r) : 96;
              if (_v < 0) _v = 0; g_resmargin = (size_t)_v*1024*1024; }
            { size_t _fr=0,_to=0; cuMemGetInfo(&_fr,&_to);
              fprintf(stderr, "[freeproj] freed d_qW..d_downW%s; free now %zuMiB, residency re-enabled (re-make margin %zuMiB)\n",
                      freed_scr ? " + scratch(d_dqscr/d_packed/d_sc)" : "", _fr/1048576, g_resmargin/1048576); }
        }
        /* Pre-make ALL 7 projection weights of ALL layers resident up front (idempotent first-touch). Only
         * then is the per-token fused dispatch guaranteed to find them -> the per-projection gemv_abt
         * fallback is unreachable under fg (so it never reads the undequantized d_qW we skip). If any
         * tensor can't go resident, latch fg OFF -> the normal full-dequant path runs (correct, just slow). */
        int ok = (f_dgemv && QWEN3 && d_xn1pad);
        if (ok) {
            int proj[7] = { V3_Q, V3_K, V3_V, V3_O, V3_GATE, V3_UP, V3_DOWN };
            for (int Lz = 0; Lz < NL && ok; Lz++)
                for (int pj = 0; pj < 7 && ok; pj++) {
                    CUdeviceptr pk=0, sc=0, tsp=0; int rr=0, kk=0;
                    if (!v3_resident_ptrs(v3_idx_layer(Lz, proj[pj]), &pk, &sc, &tsp, &rr, &kk)) {
                        ok = 0;
                        if (getenv("HX_DBG")) { size_t _fr=0,_to=0; cuMemGetInfo(&_fr,&_to);
                            fprintf(stderr, "[h1remake] STOPPED at layer %d proj %d (v3 idx %d): free=%zuMiB off=%d\n",
                                    Lz, pj, v3_idx_layer(Lz, proj[pj]), _fr/1048576, g_packedres_off); }
                    }
                }
        }
        if (g_freeproj) g_resmargin = sv_margin;    /* restore prefill margin (decode's own resident-guards use it) */
        if (ok) {
            g_fusedgemv_on = 1;
            fprintf(stderr, "[fusedgemv] ACTIVE: 7 projections via fused dequant-GEMV (per-token weight dequant skipped)\n");
            if (freed && !freed_scr) {   /* head NOT fused but projections seated: the scratch is still live for the
                                          * chunked head -- leave it. (freed_scr already freed it when head fuses.) */
                size_t _fr=0,_to=0; cuMemGetInfo(&_fr,&_to);
                fprintf(stderr, "[freeproj] seated (scratch kept for chunked head); free now %zuMiB\n", _fr/1048576);
            }
        } else {
            fprintf(stderr, "[fusedgemv] residency unavailable -> OFF (normal dequant+gemv path)\n");
            if (freed) {
                /* SAFETY NET (crash-proof): residency didn't seat even after freeing -> restore the non-fused
                 * world. The re-make may have consumed VRAM into newly-resident g_pk_res/g_sc_res (which still
                 * BENEFIT the non-fused path -- v3_upload_gpu reuses resident words), so free can be tight. We
                 * must re-alloc d_qW..d_downW (+scratch) without an OOM abort. Compute the bytes we need, and if
                 * free is short, EVICT this session's resident packed words/scales (newest-first is overkill --
                 * evict all, the non-fused path re-streams them correctly, just slower) until it fits. fg stays 0. */
                size_t need = (size_t)QD*DM + (size_t)KVD*DM*2 + (size_t)DM*QD
                            + (size_t)DFF*DM*2 + (size_t)DM*DFF;            /* 7 projection floats */
                if (freed_scr) { long _sc=g_dqscr_elems; need += (size_t)_sc + (_sc+6)/7 + (_sc+15)/16; }
                need *= sizeof(float);
                size_t _fr=0,_to=0; cuMemGetInfo(&_fr,&_to);
                if (_fr < need + (size_t)64*1024*1024) {   /* +64MiB slack for alloc granularity */
                    int nres1 = (int)(sizeof(g_pk_res)/sizeof(g_pk_res[0]));
                    int ev = 0;
                    for (int i=0;i<nres1;i++){ if(g_pk_res[i]){cuMemFree(g_pk_res[i]); g_pk_res[i]=0; ev++;}
                                               if(g_sc_res[i]){cuMemFree(g_sc_res[i]); g_sc_res[i]=0;}
                                               if(g_ts_res[i]){cuMemFree(g_ts_res[i]); g_ts_res[i]=0;} }   /* H4: free paired ts */
                    sv_resoff = 1;                          /* everything streamed now -> residency latched off */
                    cuMemGetInfo(&_fr,&_to);
                    fprintf(stderr, "[freeproj] non-seat tight (free<need): evicted %d resident packed tensors -> free %zuMiB (non-fused will re-stream)\n", ev, _fr/1048576);
                }
                d_qW    = A((size_t)QD*DM);  d_kW = A((size_t)KVD*DM); d_vW = A((size_t)KVD*DM);
                d_oW    = A((size_t)DM*QD);  d_gateW = A((size_t)DFF*DM);
                d_upW   = A((size_t)DFF*DM); d_downW = A((size_t)DM*DFF);
                if (freed_scr) {
                    long _sc = g_dqscr_elems;                 /* the alloc_buffers size sc (== d_dqscr capacity) */
                    long _mp = (_sc+6)/7, _ms = (_sc+15)/16;
                    d_packed = A(_mp); d_sc = A(_ms); d_dqscr = A(_sc);
                }
                g_packedres_off = sv_resoff;
                cuMemGetInfo(&_fr,&_to);
                fprintf(stderr, "[freeproj] non-seat -> re-alloced d_qW..d_downW%s, restored residency state (off=%d); free now %zuMiB\n",
                        freed_scr ? " + scratch" : "", g_packedres_off, _fr/1048576);
            }
        }
    }
    int fg = g_fusedgemv_on;
    /* fused producers write the normed/ctx x straight into the zero-tailed Kpad buffer (no extra copy). */
    CUdeviceptr xn1_t  = fg ? d_xn1pad  : d_xn1;
    CUdeviceptr ctx1_t = fg ? d_ctx1pad : d_ctx1;
    CUdeviceptr xn21_t = fg ? d_xn21pad : d_xn21;
    CUdeviceptr m1_t   = fg ? d_m1pad   : d_m1;
    for (int L = 0; L < NL; L++) {
        g_emit_layer = L;
        double _tu = g_dprof ? now_seconds() : 0.0;
        if (fg) upload_layer_ll_fused(L);   /* T2/M7: norms only -- skip the 7 gemv-weight dequants */
        else    upload_layer_ll(L);
        if (g_dprof) { cuCtxSynchronize(); g_dp_dq += now_seconds() - _tu; }
        emit_layer_begin(g_emit_step, L, NL);
        emit_op(L, 0, "gpu_rmsnorm_fwd_eps", "attn", "rms_1", 1);
        rms_norm_k(d_x1, xn1_t, d_ln1g, 1, DM);
        emit_op(L, 1, "gpu_gemv_abt", "attn", "q_gemv", 1);
        if (!fg || !fused_gemv(v3_idx_layer(L, V3_Q), xn1_t, d_q1, DM))  gemv_abt(xn1_t, d_qW, d_q1, DM, DM);
        emit_op(L, 2, "gpu_gemv_abt", "attn", "k_gemv", 1);
        if (!fg || !fused_gemv(v3_idx_layer(L, V3_K), xn1_t, d_k1, KVD)) gemv_abt(xn1_t, d_kW, d_k1, KVD, DM);
        emit_op(L, 3, "gpu_gemv_abt", "attn", "v_gemv", 1);
        if (!fg || !fused_gemv(v3_idx_layer(L, V3_V), xn1_t, d_v1, KVD)) gemv_abt(xn1_t, d_vW, d_v1, KVD, DM);
        if (QWEN3) {   /* Qwen3 per-head QK-norm over DH, BEFORE RoPE -- prefill does this (forward_layer_llama
                        * ~1110); decode omitted it -> un-normalized Q/K -> wrong attention -> wrong tokens.
                        * d_q1 is [NH x DH], d_k1 is [NKV x DH]; one rms_norm_k call normalizes every head. */
            rms_norm_k(d_q1, d_q1, d_qnorm, NH,  DH);
            rms_norm_k(d_k1, d_k1, d_knorm, NKV, DH);
        }
        for (int h = 0; h < NH; h++)
            rope_at(d_q1 + (CUdeviceptr)((size_t)h*DH*sizeof(float)), 1, pos);
        for (int kv = 0; kv < NKV; kv++) {
            CUdeviceptr ks = d_k1 + (CUdeviceptr)((size_t)kv*DH*sizeof(float));
            rope_at(ks, 1, pos);
            CKX(cuMemcpyDtoD(d_kcache + (CUdeviceptr)((kvoff(L,kv) + (size_t)pos*DH)*sizeof(float)), ks, (size_t)DH*sizeof(float)), "kv app K");
            CKX(cuMemcpyDtoD(d_vcache + (CUdeviceptr)((kvoff(L,kv) + (size_t)pos*DH)*sizeof(float)),
                             d_v1 + (CUdeviceptr)((size_t)kv*DH*sizeof(float)), (size_t)DH*sizeof(float)), "kv app V");
        }
        emit_op(L, 4, "gpu_rope_rot", "attn", "rope_qk", NH + NKV);
        /* v1.8 FUSED DECODE-ATTENTION: ONE launch/layer replaces the NH-iteration per-head loop below
         * (NH x {scores gemv, scale, softmax, AV gemv} + NH ctx D2D == 128 launches + 32 D2D / layer).
         * Geometry grid=(NH) block=(DH); each block owns query head h, each lane d owns ctx1_t[h*DH+d].
         * Args: q=d_q1, kc/vc=d_kcache/d_vcache, out=ctx1_t (in place), sc=d_scale (HOST ATTN_SCALE, 1-elem
         * f32 array -- FIX 1: no on-device rsqrt), T, DH, koff=kvoff(L,0) (kernel adds (h/grp)*kvstride),
         * grp=group, kvstride=kv_cap*DH. FIX 2: scalar params are .u32 + indices widen via mul.wide.s32
         * (byte address is 64-bit), so the only requirement is the ELEMENT index koff+NKV*kvstride < 2^31.
         * The guard below ASSERTS that and FAILS CLOSED to the exact per-head path if it would overflow
         * (true at this model's NC<=32768; the guard makes a future larger cache safe-by-fallback). The
         * per-head attention logit lens reads d_pr1 which the fused kernel never fills -> bypassed under
         * fused (lens is a !g_fast debug path; HX_FUSEDATTN disables the per-head attention lens). */
        size_t _famax = kvoff(L, 0) + (size_t)NKV * (size_t)kv_cap * (size_t)DH;
        int _fattn_ok = (g_fusedattn && f_fattn && _famax < 0x7fffffffULL);
        if (_fattn_ok) {
            int Ti = T, DHi = DH, koffi = (int)kvoff(L, 0), grpi = group, kvsti = (int)((size_t)kv_cap * DH);
            void* ar[] = { &d_q1, &d_kcache, &d_vcache, &ctx1_t, &d_scale, &Ti, &DHi, &koffi, &grpi, &kvsti };
            LX(f_fattn, (unsigned)NH, (unsigned)DH, ar);   /* grid=NH, block=DH, shared=0 */
        } else
        for (int h = 0; h < NH; h++) {
            int kv = h / group;
            gemv_abt(d_q1 + (CUdeviceptr)((size_t)h*DH*sizeof(float)),
                     d_kcache + (CUdeviceptr)(kvoff(L,kv)*sizeof(float)), d_sc1, T, DH);
            scale_rt(d_sc1, d_scale, T);
            smrow(d_sc1, d_pr1, 1, T);
            if (g_lens_on && !g_fast && L == NL - 1 && h < 32 && T <= 512) {
                CKX(cuMemcpyDtoH(g_attn_host, d_pr1, (size_t)T * sizeof(float)), "d2h attn");
                host_top8(g_attn_host, T, g_attn_pos[h], g_attn_w[h]);
                g_attn_nh = (h + 1 > g_attn_nh) ? h + 1 : g_attn_nh;
            }
            gemv_ab(d_pr1, d_vcache + (CUdeviceptr)(kvoff(L,kv)*sizeof(float)), d_ao1, DH, T);
            CKX(cuMemcpyDtoD(ctx1_t + (CUdeviceptr)((size_t)h*DH*sizeof(float)), d_ao1, (size_t)DH*sizeof(float)), "ctx1");
        }
        emit_op(L, 5, "gpu_gemv_abt",    "attn", "attn_scores",  NH);
        emit_op(L, 6, "gpu_scale_rt",    "attn", "attn_scale",   NH);
        emit_op(L, 7, "gpu_softmax_row", "attn", "attn_softmax", NH);
        emit_op(L, 8, "gpu_gemv_ab",     "attn", "attn_av",      NH);
        emit_op(L, 9, "gpu_gemv_abt",    "attn", "attn_proj", 1);
        if (!fg || !fused_gemv(v3_idx_layer(L, V3_O), ctx1_t, d_pj1, DM)) gemv_abt(ctx1_t, d_oW, d_pj1, DM, DM);
        emit_op(L, 10, "vector_add", "attn", "attn_residual", 1);
        vadd(d_x1, d_pj1, d_x1, DM);
        emit_op(L, 11, "gpu_rmsnorm_fwd_eps", "mlp", "rms_2", 1);
        rms_norm_k(d_x1, xn21_t, d_ln2g, 1, DM);
        emit_op(L, 12, "gpu_gemv_abt", "mlp", "fc_gate", 1);
        if (!fg || !fused_gemv(v3_idx_layer(L, V3_GATE), xn21_t, d_g1, DFF)) gemv_abt(xn21_t, d_gateW, d_g1, DFF, DM);
        emit_op(L, 13, "gpu_gemv_abt", "mlp", "fc_up", 1);
        if (!fg || !fused_gemv(v3_idx_layer(L, V3_UP),   xn21_t, d_u1, DFF)) gemv_abt(xn21_t, d_upW, d_u1, DFF, DM);
        emit_op(L, 14, "gpu_silu_mul", "mlp", "silu_mul", 1);
        silu_mul_k(d_g1, d_u1, m1_t, DFF);
        emit_op(L, 15, "gpu_gemv_abt", "mlp", "proj_down", 1);
        if (!fg || !fused_gemv(v3_idx_layer(L, V3_DOWN), m1_t, d_mo1, DM)) gemv_abt(m1_t, d_downW, d_mo1, DM, DFF);
        emit_op(L, 16, "vector_add", "mlp", "mlp_residual", 1);
        vadd(d_x1, d_mo1, d_x1, DM);
        if (g_lens_on && !g_fast && L < MAX_RES_LAYERS) {
            /* LOGIT LENS: decode the residual stream THROUGH the real final-norm + head
             * (read-only reuse of gated kernels; d_xn1/d_lg1 are free between layers). */
            rms_norm_k(d_x1, d_xn1, d_lnfg, 1, DM);
            if (g_packedhead_on) lm_head_logits_chunked(d_xn1, d_lg1, 1, 1);   /* v1.8 INC1.5 */
            else                 gemv_abt(d_xn1, d_wte_pad, d_lg1, NVpad, DM);
            if (!g_lens_host) g_lens_host = (float*)malloc((size_t)NV * sizeof(float));
            SYNC("lens sync");
            CKX(cuMemcpyDtoH(g_lens_host, d_lg1, (size_t)NV * sizeof(float)), "d2h lens");
            host_top5(g_lens_host, NV, g_lens_top[L], g_lens_p[L]);
            soft5(g_lens_host, NV, g_lens_top[L], g_lens_p[L]);
            g_lens_nl = L + 1;
        }
        emit_layer_end(g_emit_step, L, 0.0);
    }
    emit_head(g_emit_step, "norm_f", "gpu_rmsnorm_fwd_eps");
    emit_head(g_emit_step, "lm_head", "gpu_gemv_abt");
    /* T2/M7b: fuse the lm_head too when the fused-GEMV path is active AND the packed head is resident.
     * The final RMSNorm writes straight into the zero-tailed Kpad pad buffer d_xn1pad (first DM elems; the
     * [DM,Kpad) tail stays zero across tokens), then ONE fused dequant-GEMV pass computes the logits from
     * the resident packed head -- skipping the INC1.5 chunked fp32 dequant entirely. fused_lm_head() is
     * fail-closed: it returns 0 if the head isn't resident or d_xn1pad is too small for g_head_Kpad, in
     * which case we fall through to the EXACT prior path (chunked packed head, or FP32 d_wte_pad). */
    int head_fused = 0;
    if (fg && g_packedhead_on && !g_nofusehead) {
        rms_norm_k(d_x1, d_xn1pad, d_lnfg, 1, DM);                 /* normed hidden -> zero-tailed pad buffer */
        head_fused = fused_lm_head(d_xn1pad, g_fg_kpad_dm, d_lg1);
    }
    if (!head_fused) {
        double _tl = g_dprof ? now_seconds() : 0.0;
        rms_norm_k(d_x1, d_xn1, d_lnfg, 1, DM);
        if (g_packedhead_on) lm_head_logits_chunked(d_xn1, d_lg1, 1, 1);   /* v1.8 INC1.5: chunked packed head */
        else                 gemv_abt(d_xn1, d_wte_pad, d_lg1, NVpad, DM);
        if (g_dprof) { cuCtxSynchronize(); g_dp_lmh += now_seconds() - _tl; }   /* isolate the head's wall */
    }
    if (!g_fast) SYNC("decode end");
    CKX(cuMemcpyDtoH(out_logits, d_lg1, (size_t)NV*sizeof(float)), "d2h logits1");
    kv_len = T;
    if (g_dprof) {
        g_dp_total += now_seconds() - _td; g_dp_n++;
        double other = g_dp_total - g_dp_gemv - g_dp_dq - g_dp_lmh;
        fprintf(stderr, "[dprof] tok#%ld cumulative: total=%.3fs  gemv=%.3fs(%.0f%%)  dequant=%.3fs(%.0f%%)  lm_head=%.3fs(%.0f%%)  other=%.3fs(%.0f%%)\n",
                g_dp_n, g_dp_total,
                g_dp_gemv, 100.0*g_dp_gemv/g_dp_total,
                g_dp_dq,   100.0*g_dp_dq/g_dp_total,
                g_dp_lmh,  100.0*g_dp_lmh/g_dp_total,
                other,     100.0*other/g_dp_total);
    }
}

/* ===================== SEEDED REPRODUCIBLE SAMPLING (Wave 2, 2026-06) =====================
 * Host-only Category-B code: consumes the SAME host logits the greedy argmax uses; the
 * GPU compute path is untouched (G-L1/G-L2/G-KV legs unaffected). The sampler is PINNED
 * identically in the independent oracle (llama_numpy_ref.py sample_next: pure-python
 * math.exp + identical op order + the same PCG32), so sampled generation is gated
 * token-for-token for any (seed, params) -- reproducible sampling, not vibes.
 * temp == 0 (default) bypasses all of this: greedy argmax, the verified default.
 * Pipeline (pinned order, float64): rep-penalty (HF style) -> freq/presence penalties ->
 * temperature -> top-k -> softmax (max-subtract, libm exp) -> sort by (-p, id) -> top-p
 * (include the crossing token) -> min-p (>= min_p * p_max) -> renormalize -> one PCG32
 * draw u in [0,1) -> first index with cumsum > u.
 * Platform note (disclosed): C exp() and python math.exp() are the same WSL glibc libm
 * here, so the gate is bit-deterministic on this box; across OTHER platforms a 1-ulp libm
 * difference could flip a rare near-tie -- the gate, run on any given box, decides. */
static const char* json_find_key(const char* s, const char* key);   /* defined below (serve module) */
static uint64_t g_pcg_state = 0, g_pcg_inc = 0;
static void pcg32_seed(uint64_t seed) {
    g_pcg_state = 0u; g_pcg_inc = (54u << 1) | 1u;
    g_pcg_state = g_pcg_state * 6364136223846793005ULL + g_pcg_inc;
    g_pcg_state += seed;
    g_pcg_state = g_pcg_state * 6364136223846793005ULL + g_pcg_inc;
}
static uint32_t pcg32_next(void) {
    uint64_t old = g_pcg_state;
    g_pcg_state = old * 6364136223846793005ULL + g_pcg_inc;
    uint32_t xorshifted = (uint32_t)(((old >> 18) ^ old) >> 27);
    uint32_t rot = (uint32_t)(old >> 59);
    return (xorshifted >> rot) | (xorshifted << ((32 - rot) & 31));
}
typedef struct {
    double temp, top_p, min_p, rep_pen, freq_pen, pres_pen;
    int top_k; uint64_t seed; int seeded;
} SampleCfg;
static SampleCfg g_scfg = { 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0, 0u, 0 };
static unsigned short* g_tokcount = NULL;     /* [NV] occurrence counts (penalties) */
static void sample_reset(const int* prompt_ids, int n_prompt) {
    if (!g_tokcount) g_tokcount = (unsigned short*)malloc((size_t)NV * sizeof(unsigned short));
    memset(g_tokcount, 0, (size_t)NV * sizeof(unsigned short));
    for (int i = 0; i < n_prompt; i++)
        if (prompt_ids[i] >= 0 && prompt_ids[i] < NV && g_tokcount[prompt_ids[i]] < 65535) g_tokcount[prompt_ids[i]]++;
    pcg32_seed(g_scfg.seed);
}
static void sample_count(int id) {
    if (g_tokcount && id >= 0 && id < NV && g_tokcount[id] < 65535) g_tokcount[id]++;
}
typedef struct { int id; double v; } SCand;
static int scand_cmp(const void* a, const void* b) {   /* pinned: (-v, id) */
    const SCand* x = (const SCand*)a; const SCand* y = (const SCand*)b;
    if (x->v > y->v) return -1;
    if (x->v < y->v) return 1;
    return (x->id < y->id) ? -1 : (x->id > y->id) ? 1 : 0;
}
static int sample_next(const float* logits, int nv) {
    /* 1+2: penalties on float64 copies of the raw logits */
    double* l = (double*)malloc((size_t)nv * sizeof(double));
    for (int i = 0; i < nv; i++) {
        double x = (double)logits[i];
        int c = g_tokcount ? (int)g_tokcount[i] : 0;
        if (c > 0 && g_scfg.rep_pen != 1.0) { x = (x > 0.0) ? x / g_scfg.rep_pen : x * g_scfg.rep_pen; }
        if (c > 0) x -= g_scfg.freq_pen * (double)c + g_scfg.pres_pen;
        l[i] = x;
    }
    /* 3: temperature */
    double invt = 1.0 / g_scfg.temp;
    for (int i = 0; i < nv; i++) l[i] *= invt;
    /* 4: top-k preselect (k<=0 = all). Build candidate list. */
    int k = (g_scfg.top_k > 0 && g_scfg.top_k < nv) ? g_scfg.top_k : nv;
    SCand* cand = (SCand*)malloc((size_t)nv * sizeof(SCand));
    for (int i = 0; i < nv; i++) { cand[i].id = i; cand[i].v = l[i]; }
    qsort(cand, (size_t)nv, sizeof(SCand), scand_cmp);    /* by (-logit, id), pinned */
    int n = k;
    /* 5: softmax over the n kept (max-subtract; libm exp; float64) */
    double mx = cand[0].v;
    double sum = 0.0;
    for (int i = 0; i < n; i++) { cand[i].v = exp(cand[i].v - mx); sum += cand[i].v; }
    for (int i = 0; i < n; i++) cand[i].v /= sum;          /* probs, already sorted (-p, id) */
    /* 6: top-p nucleus (include the crossing token) */
    if (g_scfg.top_p < 1.0) {
        double cum = 0.0; int keep = n;
        for (int i = 0; i < n; i++) { cum += cand[i].v; if (cum >= g_scfg.top_p) { keep = i + 1; break; } }
        n = keep;
    }
    /* 7: min-p relative floor */
    if (g_scfg.min_p > 0.0) {
        double floor_p = g_scfg.min_p * cand[0].v;
        int keep = n;
        for (int i = 0; i < n; i++) if (cand[i].v < floor_p) { keep = i; break; }
        n = (keep > 0) ? keep : 1;
    }
    /* 8: renormalize + one PCG32 draw + CDF walk */
    double rsum = 0.0;
    for (int i = 0; i < n; i++) rsum += cand[i].v;
    double u = (double)pcg32_next() / 4294967296.0;
    double cum2 = 0.0; int pick = cand[n-1].id;
    for (int i = 0; i < n; i++) { cum2 += cand[i].v / rsum; if (cum2 > u) { pick = cand[i].id; break; } }
    free(cand); free(l);
    return pick;
}
/* parse "k=v,k=v" --sample spec (CLI) or per-request JSON fields into g_scfg.
 * Returns 1 if sampling is ACTIVE (temp > 0). */
static int sample_cfg_from_json(const char* line) {
    const char* p;
    g_scfg.temp = 0.0; g_scfg.top_p = 1.0; g_scfg.min_p = 0.0; g_scfg.top_k = 0;
    g_scfg.rep_pen = 1.0; g_scfg.freq_pen = 0.0; g_scfg.pres_pen = 0.0; g_scfg.seed = 0u; g_scfg.seeded = 0;
    if ((p = json_find_key(line, "temperature")) != NULL) g_scfg.temp = atof(p);
    if ((p = json_find_key(line, "top_p"))       != NULL) g_scfg.top_p = atof(p);
    if ((p = json_find_key(line, "top_k"))       != NULL) g_scfg.top_k = atoi(p);
    if ((p = json_find_key(line, "min_p"))       != NULL) g_scfg.min_p = atof(p);
    if ((p = json_find_key(line, "rep_penalty")) != NULL) g_scfg.rep_pen = atof(p);
    if ((p = json_find_key(line, "freq_penalty"))!= NULL) g_scfg.freq_pen = atof(p);
    if ((p = json_find_key(line, "pres_penalty"))!= NULL) g_scfg.pres_pen = atof(p);
    if ((p = json_find_key(line, "seed"))        != NULL) { g_scfg.seed = (uint64_t)strtoull(p, NULL, 10); g_scfg.seeded = 1; }
    if (g_scfg.temp < 0.0) g_scfg.temp = 0.0;
    if (g_scfg.temp > 4.0) g_scfg.temp = 4.0;
    if (g_scfg.top_p <= 0.0 || g_scfg.top_p > 1.0) g_scfg.top_p = 1.0;
    if (g_scfg.rep_pen <= 0.0) g_scfg.rep_pen = 1.0;
    return g_scfg.temp > 0.0;
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

/* v1.7: optionally load the tiled NVFP4 dequant kernel (2nd module) -> sets g_gpu_dq. QWEN3/v3 only;
 * HX_HOSTDEQ forces the host path; a missing/unloadable HX_DQPTX is non-fatal (host-dequant fallback).
 * Called from device_init (forward modes) AND run_v3_upload_check (the byte-identical gate). */
static void load_dequant_module(void) {
    g_hostdeq = getenv("HX_HOSTDEQ") ? 1 : 0;
    const char* dqp = getenv("HX_DQPTX");
    if (!dqp || g_hostdeq || !QWEN3) return;
    FILE* df = fopen(dqp, "rb");
    if (!df) { fprintf(stderr, "[dq] HX_DQPTX '%s' not found; using host dequant\n", dqp); return; }
    fseek(df,0,SEEK_END); long dsz=ftell(df); fseek(df,0,SEEK_SET);
    char* dbuf=(char*)malloc(dsz+1);
    if (dbuf && fread(dbuf,1,dsz,df)==(size_t)dsz) {
        dbuf[dsz]=0;
        if (cuModuleLoadData(&g_dqmod, dbuf)==CUDA_SUCCESS &&
            cuModuleGetFunction(&f_dq_tiled, g_dqmod, "nvfp4_dequant_tiled")==CUDA_SUCCESS) {
            g_gpu_dq=1; for (int c=0;c<256;c++) g_e4m3_tab[c]=v3_e4m3_decode(c);   /* v1.7: e4m3 LUT */
            fprintf(stderr, "[dq] GPU NVFP4 dequant ON (%s)\n", dqp);
        } else fprintf(stderr, "[dq] dequant ptx load failed; using host dequant\n");
    }
    fclose(df);
}

/* T2/M7: load the STANDALONE fused dequant-GEMV PTX module (entry "dequant_gemv_blockred"). Gated behind
 * HX_FUSEDGEMV (which already required HX_PACKEDRES in device_init). Path from HX_FUSEDGEMV_PTX, default
 * /home/legoa/dequant_gemv_blockred.ptx. Non-fatal: a missing/unloadable module just leaves g_fusedgemv
 * armed-but-inactive -> the decode dispatch sees f_dgemv==0 and uses the normal gemv_abt path. */
static void load_fusedgemv_module(void) {
    if (!g_fusedgemv || !QWEN3) return;
    const char* fp = getenv("HX_FUSEDGEMV_PTX");
    if (!fp || !fp[0]) fp = "/home/legoa/dequant_gemv_blockred.ptx";
    FILE* ff = fopen(fp, "rb");
    if (!ff) { fprintf(stderr, "[fusedgemv] PTX '%s' not found -> OFF (normal gemv path)\n", fp); g_fusedgemv = 0; return; }
    fseek(ff,0,SEEK_END); long fsz=ftell(ff); fseek(ff,0,SEEK_SET);
    char* fbuf=(char*)malloc(fsz+1);
    int ok = 0;
    if (fbuf && fread(fbuf,1,fsz,ff)==(size_t)fsz) {
        fbuf[fsz]=0;
        if (cuModuleLoadData(&g_fgmod, fbuf)==CUDA_SUCCESS &&
            cuModuleGetFunction(&f_dgemv, g_fgmod, "dequant_gemv_blockred")==CUDA_SUCCESS) {
            ok = 1;
            fprintf(stderr, "[fusedgemv] fused dequant-GEMV module ON (%s)\n", fp);
        }
    }
    if (!ok) { fprintf(stderr, "[fusedgemv] fused ptx load failed -> OFF (normal gemv path)\n"); g_fusedgemv = 0; f_dgemv = 0; }
    if (fbuf) free(fbuf);
    fclose(ff);
}

/* v1.8: load the STANDALONE fused decode-attention PTX module (entry "fused_decode_attn"). Gated behind
 * HX_FUSEDATTN. Path from HX_FUSEDATTN_PTX, default /home/legoa/fused_decode_attn.ptx. Non-fatal: a
 * missing/unloadable module just leaves g_fusedattn armed-but-inactive -> the decode dispatch sees
 * f_fattn==0 and runs the EXACT per-head loop. Mirrors load_fusedgemv_module exactly. */
static void load_fusedattn_module(void) {
    if (!g_fusedattn) return;
    const char* fp = getenv("HX_FUSEDATTN_PTX");
    if (!fp || !fp[0]) fp = "/home/legoa/fused_decode_attn.ptx";
    FILE* ff = fopen(fp, "rb");
    if (!ff) { fprintf(stderr, "[fusedattn] PTX '%s' not found -> OFF (per-head path)\n", fp); g_fusedattn = 0; return; }
    fseek(ff,0,SEEK_END); long fsz=ftell(ff); fseek(ff,0,SEEK_SET);
    char* fbuf=(char*)malloc(fsz+1);
    int ok = 0;
    if (fbuf && fread(fbuf,1,fsz,ff)==(size_t)fsz) {
        fbuf[fsz]=0;
        if (cuModuleLoadData(&g_famod, fbuf)==CUDA_SUCCESS &&
            cuModuleGetFunction(&f_fattn, g_famod, "fused_decode_attn")==CUDA_SUCCESS) {
            ok = 1;
            fprintf(stderr, "[fusedattn] fused decode-attention module ON (%s)\n", fp);
        }
    }
    if (!ok) { fprintf(stderr, "[fusedattn] fused ptx load failed -> OFF (per-head path)\n"); g_fusedattn = 0; f_fattn = 0; }
    if (fbuf) free(fbuf);
    fclose(ff);
}

/* load PTX, create the context, fetch the forward-only kernel handles, mmap the weight file. */
static int device_init(const char* ptx_path, const char* wpath) {
    g_prof   = getenv("HX_PROF") ? 1 : 0;        /* v1.7 INCREMENT 2: per-upload profiling */
    g_dprof  = getenv("HX_DPROF") ? 1 : 0;       /* v1.8: decode-compute breakdown (gemv vs dequant vs rest) */
    g_genprof = getenv("HX_GENPROF") ? 1 : 0;    /* v1.8: HX_FAST-honest end-to-end decode s/tok (production wall) */
    g_scache = getenv("HX_SCALECACHE") ? 1 : 0;  /* v1.7 INC2c: resident static-scale cache */
    g_packedres = getenv("HX_PACKEDRES") ? 1 : 0;  /* v1.8 INC1: resident packed 4-bit layer weights */
    { const char* _rm = getenv("HX_RESMARGIN"); if (_rm) { long _v = atol(_rm); if (_v > 0) g_resmargin = (size_t)_v*1024*1024; } }  /* tunable resident VRAM headroom (MiB) */
    if (g_packedres) g_scache = 1;                 /* residency implies the static-scale cache */
    g_packedhead = getenv("HX_PACKEDHEAD") ? 1 : 0;  /* v1.8 INC1.5: resident packed lm_head (no FP32 d_wte_pad) */
    if (g_packedres) g_packedhead = 1;               /* resident layers imply the resident head (both shrink VRAM) */
    g_fusedgemv = getenv("HX_FUSEDGEMV") ? 1 : 0;    /* T2/M7: fused dequant-GEMV decode (needs resident packed weights) */
    g_fusedattn = getenv("HX_FUSEDATTN") ? 1 : 0;    /* v1.8: fused decode-attention (one kernel/layer; independent of the gemv path) */
    g_nofusehead = getenv("HX_NOFUSEHEAD") ? 1 : 0;  /* T2/M7b: measurement-only -- keep lm_head chunked */
    g_freeproj = getenv("HX_FREEPROJ") ? 1 : 0;      /* H1: post-prefill free d_qW..d_downW to seat full residency (needs HX_FUSEDGEMV+HX_PACKEDRES) */
    if (g_freeproj && (!g_fusedgemv || !g_packedres)) {   /* only meaningful with the fused+resident path */
        fprintf(stderr, "[freeproj] needs HX_FUSEDGEMV=1 + HX_PACKEDRES=1 -> OFF\n");
        g_freeproj = 0;
    }
    if (g_fusedgemv && !g_packedres) {               /* the fused path reads g_pk_res/g_sc_res -> requires residency */
        fprintf(stderr, "[fusedgemv] needs resident packed weights (set HX_PACKEDRES=1) -> OFF\n");
        g_fusedgemv = 0;
    }
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
        if (KV_ON) {   /* the 3 G-KV0-gated decode kernels (present in the 14-kernel mint) */
            CK(cuModuleGetFunction(&f_gv_abt, g_mod, "gpu_gemv_abt"),    "gpu_gemv_abt");
            CK(cuModuleGetFunction(&f_gv_ab,  g_mod, "gpu_gemv_ab"),     "gpu_gemv_ab");
            CK(cuModuleGetFunction(&f_smrow,  g_mod, "gpu_softmax_row"), "gpu_softmax_row");
        }
    }
    load_dequant_module();   /* v1.7: GPU dequant (HX_DQPTX) -> g_gpu_dq; HX_HOSTDEQ forces host */
    load_fusedgemv_module(); /* T2/M7: standalone fused dequant-GEMV module (HX_FUSEDGEMV) */
    load_fusedattn_module(); /* v1.8: standalone fused decode-attention module (HX_FUSEDATTN) */
    if (load_gpt2_weights(wpath)) return 2;
    return 0;
}

/* allocate the layer-weight device buffers (reused across all NL layers; NL=12 at the 124M default,
 * NL=48 at XL via HX_NL) + the per-Smax activations. One layer's weights resident at a time. */
static int alloc_buffers(int Smax) {
    Spad_max = Smax;
    /* layer-weight buffers (one layer resident at a time; upload_layer() refills them per layer).
     * The fused-GPT-2 weight slabs (attW/prjW/fcW/pjW) are UNUSED on the llama/qwen3 path (which has
     * its own d_qW..d_downW); for QWEN3 allocate them as size-1 stubs to save ~668MB VRAM (the head
     * needs 2.5GB) -- SmolLM2 (ARCH && !QWEN3) keeps the full alloc, byte-unchanged. */
    d_ln1g = A(DM); d_ln1b = A(DM);
    d_attW = A(QWEN3 ? 1 : (size_t)DM*3*DM); d_attb = A(QWEN3 ? 1 : 3*DM);
    d_prjW = A(QWEN3 ? 1 : (size_t)DM*DM);   d_prjb = A(DM);
    d_ln2g = A(DM); d_ln2b = A(DM);
    d_fcW  = A(QWEN3 ? 1 : (size_t)DM*DFF);  d_fcb  = A(QWEN3 ? 1 : DFF);
    d_pjW  = A(QWEN3 ? 1 : (size_t)DFF*DM);  d_pjb  = A(DM);
    /* activations sized for the largest padded length we will see. */
    d_x      = A((size_t)Smax*DM);
    d_xn     = A((size_t)Smax*DM);
    d_qkv    = A((size_t)Smax * (ARCH ? (size_t)(QD + 2*KVD) : (size_t)(3*DM)));   /* llama carve q[QD]+k[KVD]+v[KVD] */
    d_Qh     = A((size_t)Smax*DH);
    d_Kh     = A((size_t)Smax*DH);
    d_Vh     = A((size_t)Smax*DH);
    d_scores = A((size_t)Smax*Smax);
    d_attnw  = A((size_t)Smax*Smax);
    d_aoh    = A((size_t)Smax*DH);
    d_ctx    = A((size_t)Smax * (ARCH ? (size_t)QD : (size_t)DM));   /* merged attention ctx is [Spad,QD] for llama */
    d_proj   = A((size_t)Smax*DM);
    d_xn2    = A((size_t)Smax*DM);
    d_mlp1   = A((size_t)Smax*DFF);
    d_mlp1g  = A((size_t)Smax*DFF);
    d_mlp2   = A((size_t)Smax*DM);
    d_scale  = A(1);
    CK(cuMemcpyHtoD(d_scale, &ATTN_SCALE, sizeof(float)), "h2d scale");
    CK(cuMemsetD8(d_ctx, 0, (size_t)Smax * (ARCH ? (size_t)QD : (size_t)DM) * sizeof(float)), "zero ctx");
    if (ARCH) {   /* llama extras: separate-GEMM weights, the SwiGLU 3rd slab, RoPE tables */
        d_qW    = A((size_t)QD*DM);   /* q_proj [QD,DM] */
        d_kW    = A((size_t)KVD*DM);
        d_vW    = A((size_t)KVD*DM);
        d_oW    = A((size_t)DM*QD);   /* o_proj [DM,QD] */
        d_gateW = A((size_t)DFF*DM);
        d_upW   = A((size_t)DFF*DM);
        d_downW = A((size_t)DM*DFF);
        d_mlp1c = A((size_t)Smax*DFF);
        if (QWEN3) {   /* per-head QK-norm weights [DH] + a reusable host dequant scratch (max layer tensor) */
            d_qnorm = A(DH); d_knorm = A(DH);
            long kpad_dm = ((DM+111)/112)*112, kpad_dff = ((DFF+111)/112)*112, kpad_qd = ((QD+111)/112)*112;
            long sc = (long)DFF*kpad_dm; if ((long)DM*kpad_dff > sc) sc = (long)DM*kpad_dff;   /* gate/up + down */
            if ((long)QD*kpad_dm > sc) sc = (long)QD*kpad_dm;   /* q_proj [QD x DM] (QD!=DM on 32B) */
            if ((long)DM*kpad_qd > sc) sc = (long)DM*kpad_qd;   /* o_proj [DM x QD] */
            g_v3scratch = (float*)malloc((size_t)sc*sizeof(float));
            if (!g_v3scratch) { fprintf(stderr, "v3 scratch malloc %ld failed\n", sc); return 2; }
            g_dqscr_elems = sc;   /* v1.7 INC2: capacity for chunked lm_head GPU dequant */
            if (g_gpu_dq) {   /* v1.7: device scratch for the GPU dequant, sized off sc = max rows*Kpad floats */
                long mp = (sc+6)/7, ms = (sc+15)/16;       /* max rows*kwords, max rows*kblk */
                d_packed = A(mp);                          /* packed i32 words (4B each) */
                d_sc     = A(ms);                          /* effective per-16-block scales (f32) */
                d_dqscr  = A(sc);                          /* dequant out [rows x Kpad] before Kpad->K compaction */
                CK(cuMemAllocHost((void**)&g_scbuf, (size_t)ms*4), "pinned g_scbuf");   /* v1.7 INC2b: pinned -> DMA HtoD */
                CK(cuMemAllocHost(&g_hpin,          (size_t)mp*4), "pinned packed bounce");
            }
        }
        /* q/k/v GEMM outputs are CARVED out of the (larger) fused d_qkv slab:
         * q[Smax,DM] @ 0, k[Smax,KVD] after it, v[Smax,KVD] after that
         * (DM + 2*KVD <= 3*DM always, since KVD <= DM). */
        d_q = d_qkv;
        d_k = d_qkv + (CUdeviceptr)((size_t)Smax*QD*sizeof(float));   /* d_q occupies [Smax,QD] */
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
        if (KV_ON) {
            /* K/V caches: NL x NKV x Smax x DH floats each (360M @ Smax 384: ~31 MB x2) */
            kv_cap = Smax; kv_len = 0;
            size_t cfl = (size_t)NL * NKV * kv_cap * DH;
            d_kcache = A(cfl); d_vcache = A(cfl);
            /* M=1 decode buffers */
            d_x1 = A(DM); d_xn1 = A(DM); d_q1 = A(DM); d_k1 = A(KVD); d_v1 = A(KVD);
            d_sc1 = A((size_t)Smax); d_pr1 = A((size_t)Smax); d_ao1 = A(DH);
            d_ctx1 = A(DM); d_pj1 = A(DM); d_xn21 = A(DM);
            d_g1 = A(DFF); d_u1 = A(DFF); d_m1 = A(DFF); d_mo1 = A(DM);
            d_lg1 = A((size_t)((NV + 63) / 64) * 64);
            /* T2/M7: Kpad-padded, tail-ZEROED x scratch for the fused gemv. The fused kernel sums over the
             * FULL Kpad row (pad cols dequant to 0, contributing 0) so x MUST be defined + zero in [K, Kpad).
             * The rmsnorm/copy producers write only the first K elems; the zeroed tail persists across tokens.
             * Kpad is read from the descriptor table (q==[*,DM]; down==[*,DFF]); same Kpad for q/k/v/o/gate/up. */
            if (g_fusedgemv && QWEN3) {
                /* size each padded x to ITS projection's input Kpad (q/k/v share the DM-input pad; o uses the
                 * QD-input pad -- == DM-pad when QD==DM as on 8B, but distinct for QD!=DM models like 32B). */
                int kp_q    = (int)g_desc[v3_idx_layer(0, V3_Q)].Kpad;    /* q/k/v input = DM  */
                int kp_o    = (int)g_desc[v3_idx_layer(0, V3_O)].Kpad;    /* o     input = QD  */
                int kp_gate = (int)g_desc[v3_idx_layer(0, V3_GATE)].Kpad; /* gate/up input = DM */
                int kp_down = (int)g_desc[v3_idx_layer(0, V3_DOWN)].Kpad; /* down  input = DFF */
                g_fg_kpad_dm = kp_q; g_fg_kpad_dff = kp_down;
                d_xn1pad  = A((size_t)kp_q);    CK(cuMemsetD8(d_xn1pad,  0, (size_t)kp_q   *sizeof(float)), "fg xn1 zero");
                d_ctx1pad = A((size_t)kp_o);    CK(cuMemsetD8(d_ctx1pad, 0, (size_t)kp_o   *sizeof(float)), "fg ctx1 zero");
                d_xn21pad = A((size_t)kp_gate); CK(cuMemsetD8(d_xn21pad, 0, (size_t)kp_gate*sizeof(float)), "fg xn21 zero");
                d_m1pad   = A((size_t)kp_down); CK(cuMemsetD8(d_m1pad,   0, (size_t)kp_down*sizeof(float)), "fg m1 zero");
            }
        }
        if (RESIDENT && NL <= MAX_RES_LAYERS) {
            /* upload EVERY layer once; upload_layer_ll becomes a pointer swap (decode is
             * H2D-free). 360M fp32 = 1.45 GB on-device -- small models only by policy. */
            LayerOffLL lo2 = layer_offsets_ll();
            for (int L = 0; L < NL; L++) {
                long b2 = off_layer_ll(L);
                res_inln[L]   = A(DM);              CKX(cuMemcpyHtoD(res_inln[L],   &g_wbase[b2+lo2.inln],   (size_t)DM*sizeof(float)),      "res in_ln");
                res_qW[L]     = A((size_t)DM*DM);   CKX(cuMemcpyHtoD(res_qW[L],     &g_wbase[b2+lo2.qW],     (size_t)DM*DM*sizeof(float)),   "res qW");
                res_kW[L]     = A((size_t)KVD*DM);  CKX(cuMemcpyHtoD(res_kW[L],     &g_wbase[b2+lo2.kW],     (size_t)KVD*DM*sizeof(float)),  "res kW");
                res_vW[L]     = A((size_t)KVD*DM);  CKX(cuMemcpyHtoD(res_vW[L],     &g_wbase[b2+lo2.vW],     (size_t)KVD*DM*sizeof(float)),  "res vW");
                res_oW[L]     = A((size_t)DM*DM);   CKX(cuMemcpyHtoD(res_oW[L],     &g_wbase[b2+lo2.oW],     (size_t)DM*DM*sizeof(float)),   "res oW");
                res_postln[L] = A(DM);              CKX(cuMemcpyHtoD(res_postln[L], &g_wbase[b2+lo2.postln], (size_t)DM*sizeof(float)),      "res post_ln");
                res_gateW[L]  = A((size_t)DFF*DM);  CKX(cuMemcpyHtoD(res_gateW[L],  &g_wbase[b2+lo2.gateW],  (size_t)DFF*DM*sizeof(float)),  "res gateW");
                res_upW[L]    = A((size_t)DFF*DM);  CKX(cuMemcpyHtoD(res_upW[L],    &g_wbase[b2+lo2.upW],    (size_t)DFF*DM*sizeof(float)),  "res upW");
                res_downW[L]  = A((size_t)DM*DFF);  CKX(cuMemcpyHtoD(res_downW[L],  &g_wbase[b2+lo2.downW],  (size_t)DM*DFF*sizeof(float)),  "res downW");
            }
            res_loaded = 1;
                        printf("[kv] resident weights: %d layers on-device\n", NL);
        }
    }
    return 0;
}

/* upload llama layer L's 9 tensors (reuses d_ln1g for input_ln, d_ln2g for post_ln). */
/* v3 dequant-on-upload: dequant tensor `idx` to host f32 (rows x K, NVFP4 padding dropped) -> dst. */
/* v1.7 INCREMENT 1: dequant tensor `idx` ON THE GPU (replaces the host v3_dequant_packed for the
 * per-layer packed weights). HtoD the packed words + the host-built effective per-16-block scale,
 * launch nvfp4_dequant_tiled -> [rows x Kpad], then cuMemcpy2D-compact Kpad->K into dst. BYTE-IDENTICAL
 * to v3_upload's host path by construction (same mag * (e4m3_decode(micro)*ts), single precision);
 * --v3-upload-check gates it. Small f32 tensors stay on a plain HtoD. */
/* v1.8 INC1: one-time cuMemGetInfo budget check for resident packed words + scales across all packed
 * tensors. If the full resident set won't fit free VRAM with a transient margin, latch residency OFF so
 * every tensor streams (graceful degrade -- exactly the v1.6/v1.7 path, byte-identical). */
static void packedres_fitcheck(void) {
    if (g_packedres_chk) return;
    g_packedres_chk = 1;
    size_t need = 0;
    for (int i = 0; i < g_ntensors; i++) {
        const HXGWv3Desc* d = &g_desc[i];
        if (!d->packed) continue;
        if (i == v3_idx_lmhead()) continue;   /* v1.8: the lm_head never goes through g_pk_res -- setup_head (INC1.5 resident, or FP32 d_wte_pad) owns it; don't double-count */
        int Kpad = (int)d->Kpad, kwords = Kpad/7;
        need += (size_t)d->rows*kwords*4;   /* resident packed WORDS only (the dominant ~3.7GiB). The effective
                                             * scales (~1.6GiB) go resident via the SEPARATE g_sc_res lazy path,
                                             * which already has its own non-fatal fallback -- budgeting them
                                             * here too double-counts and wrongly blocks the fit by ~1.6GiB. */
    }
    size_t margin = g_resmargin;  /* headroom for the dequant scratch + the lazily-resident scales (HX_RESMARGIN) */
    size_t freeb = 0, totb = 0; cuMemGetInfo(&freeb, &totb);
    if (freeb < need + margin) {
        g_packedres_off = 1;
        fprintf(stderr, "[packedres] need %zu MiB +margin, free %zu MiB -> streaming (resident OFF)\n",
                need/(1024*1024), freeb/(1024*1024));
    } else {
        fprintf(stderr, "[packedres] resident ON: need %zu MiB, free %zu MiB\n",
                need/(1024*1024), freeb/(1024*1024));
    }
}

/* H4: pack [rows x kblk] raw e4m3 micro BYTES (from the mmap) into [rows x micstride] i32 WORDS, 4 micro
 * per word base-256 low-byte-first, micstride = ceil(kblk/4) words/row. Output goes into `dst32` (caller
 * provides >= rows*micstride ints). NVFP4 micro-scales are positive magnitudes (bit7 always 0) and never
 * the NaN code, so each word <= 0x7F7F7F7F is non-negative -> the in-kernel signed-safe decode is exact.
 * Byte-identical scale values to g_e4m3_tab[micro]*ts; this just re-lays-out the bytes for residency. */
static void v3_pack_micro(const uint8_t* micro, long rows, int kblk, int32_t* dst32) {
    int micstride = (kblk + 3) / 4;
    memset(dst32, 0, (size_t)rows * micstride * 4);
    for (long r = 0; r < rows; r++) {
        const uint8_t* mr = micro + (size_t)r * kblk;
        int32_t* wr = dst32 + (size_t)r * micstride;
        for (int b = 0; b < kblk; b++) {
            wr[b >> 2] |= ((int32_t)(mr[b] & 0xFF)) << (8 * (b & 3));
        }
    }
}

static void v3_upload_gpu(int idx, CUdeviceptr dst) {
    const HXGWv3Desc* d = &g_desc[idx];
    const unsigned char* base = (const unsigned char*)g_map;
    if (!d->packed) {
        CKX(cuMemcpyHtoD(dst, base + d->data_off, (size_t)((long)d->rows*d->K)*4), "dq f32 HtoD");
        return;
    }
    int rows=(int)d->rows, K=(int)d->K, Kpad=(int)d->Kpad, kwords=Kpad/7, kblk=Kpad/16, BD=112;
    const int32_t* w = (const int32_t*)(base + d->data_off);
    const uint8_t* micro = (const uint8_t*)(base + d->scale_off);
    float ts; memcpy(&ts, base + d->scale_off + (size_t)rows*kblk, 4);
    double tA = g_prof ? now_seconds() : 0.0;
    /* v1.8 INC1: use the resident packed words if present; else first-touch alloc them resident (one-time
     * HtoD straight from the mmap); else stream via the pinned bounce (the v1.7 INC2b path). The dequant
     * kernel reads identical bytes either way -> output byte-identical. */
    CUdeviceptr d_packed_use = d_packed;
    int nsc1 = (int)(sizeof(g_pk_res)/sizeof(g_pk_res[0]));
    if (g_packedres && !g_packedres_off) packedres_fitcheck();
    if (g_packedres && !g_packedres_off && idx >= 0 && idx < nsc1 && g_pk_res[idx]) {
        d_packed_use = g_pk_res[idx];                                       /* resident: NO memcpy, NO HtoD */
    } else if (g_packedres && !g_packedres_off && idx >= 0 && idx < nsc1) {
        CUdeviceptr p = 0;                                                  /* first touch: alloc resident + one-time HtoD */
        memcpy(g_hpin, w, (size_t)rows*kwords*4);                          /* v1.8 FIX: fault the mmap pages into PINNED RAM first (a direct cuMemcpyHtoD from a COLD file-backed mmap can hang the DMA; bytes identical to HtoD-from-w). */
        size_t _frp=0,_top=0; cuMemGetInfo(&_frp,&_top);                   /* v1.8 FIX: proactive VRAM-headroom guard. On driver 610.62 a cuMemAlloc at ~0 free HANGS (the old driver returned OOM, which latched the graceful fallback). The fit-check under-budgets by ~1GiB of resident scales, so it would otherwise alloc until free=0 and hang. Latch off + stream BEFORE exhaustion, leaving 768MiB for dequant scratch + activations + KV. */
        if (_frp < (size_t)rows*kwords*4 + g_resmargin) g_packedres_off = 1;
        if (!g_packedres_off && cuMemAlloc(&p, (size_t)rows*kwords*4) == CUDA_SUCCESS &&
            cuMemcpyHtoD(p, g_hpin, (size_t)rows*kwords*4) == CUDA_SUCCESS) { g_pk_res[idx] = p; d_packed_use = p; }
        else { if (p) cuMemFree(p); g_packedres_off = 1;                    /* VRAM-tight: latch off, stream this + the rest (g_hpin already holds w) */
               CKX(cuMemcpyHtoD(d_packed, g_hpin, (size_t)rows*kwords*4), "dq packed HtoD"); }
    } else {
        memcpy(g_hpin, w, (size_t)rows*kwords*4);                          /* v1.7 INC2b: mmap->pinned, then DMA HtoD */
        CKX(cuMemcpyHtoD(d_packed, g_hpin, (size_t)rows*kwords*4), "dq packed HtoD");
    }
    double tB = g_prof ? now_seconds() : 0.0; g_pf_htod += tB - tA;
    /* v1.7 INC2c: reuse the resident effective-scale cache (the scales are static) or build into scratch.
     * H4: when g_fusedgemv is ON, g_sc_res holds RAW e4m3 micro (for the fused decode), NOT the fp32
     * effective scale -- so the f_dq_tiled dequant here always builds the fp32 effective into scratch d_sc
     * (the resident fp32 cache is OFF under fused). Byte-identical output either way. */
    CUdeviceptr d_sc_use = d_sc;
    int nsc = (int)(sizeof(g_sc_res)/sizeof(g_sc_res[0]));
    int sc_cache_ok = (g_scache && !g_fusedgemv);   /* fp32 effective cache only when NOT fused */
    if (sc_cache_ok && idx >= 0 && idx < nsc && g_sc_res[idx]) {
        d_sc_use = g_sc_res[idx];                                        /* cached: skip the build + HtoD */
    } else {
        for (long i=0;i<(long)rows*kblk;i++) g_scbuf[i] = g_e4m3_tab[micro[i]] * ts;   /* byte-identical to v3_e4m3_decode */
        if (sc_cache_ok && idx >= 0 && idx < nsc && !g_packedres_off) {  /* first touch: cache it resident (non-fatal on VRAM-tight) */
            CUdeviceptr p = 0; size_t _frs=0,_tos=0; cuMemGetInfo(&_frs,&_tos);   /* v1.8 FIX: same proactive VRAM-headroom guard as the packed alloc (scales are ~1GiB total; don't exhaust VRAM). */
            if (_frs >= (size_t)rows*kblk*4 + g_resmargin &&
                cuMemAlloc(&p, (size_t)rows*kblk*4) == CUDA_SUCCESS &&
                cuMemcpyHtoD(p, g_scbuf, (size_t)rows*kblk*4) == CUDA_SUCCESS) { g_sc_res[idx] = p; d_sc_use = p; }
            else { if (p) cuMemFree(p); CKX(cuMemcpyHtoD(d_sc, g_scbuf, (size_t)rows*kblk*4), "dq scale HtoD"); }
        } else CKX(cuMemcpyHtoD(d_sc, g_scbuf, (size_t)rows*kblk*4), "dq scale HtoD");
    }
    double tD = g_prof ? now_seconds() : 0.0; g_pf_sb += tD - tB;        /* scale build+upload, or cache reuse */
    CUdeviceptr out_dev = (Kpad==K) ? dst : d_dqscr;
    int mm0=0, kk=Kpad, bd=BD;
    void* ar[] = { &d_packed_use, &d_sc_use, &out_dev, &mm0, &kk, &bd };
    if (getenv("HX_DBG")) { size_t _fr=0,_to=0; cuMemGetInfo(&_fr,&_to); fprintf(stderr, "[dbg] idx=%d pre-dequant free=%zuMiB off=%d (rows=%ld Kpad=%d)\n", idx, _fr/1048576, g_packedres_off, (long)rows, Kpad); }
    LX(f_dq_tiled, (unsigned)((long)rows*(Kpad/BD)), (unsigned)BD, ar);
    double tE = g_prof ? now_seconds() : 0.0; g_pf_dq += tE - tD; g_pf_n++;
    if (Kpad != K) {   /* compact [rows x Kpad] -> [rows x K], drop the pad cols (host did this via memmove) */
        CUDA_MEMCPY2D cp; memset(&cp, 0, sizeof(cp));
        cp.srcMemoryType=CU_MEMORYTYPE_DEVICE; cp.srcDevice=d_dqscr; cp.srcPitch=(size_t)Kpad*4;
        cp.dstMemoryType=CU_MEMORYTYPE_DEVICE; cp.dstDevice=dst;    cp.dstPitch=(size_t)K*4;
        cp.WidthInBytes=(size_t)K*4; cp.Height=(size_t)rows;
        CKX(cuMemcpy2D(&cp), "dq compact 2D");
    }
    if (g_prof) g_pf_cmp += now_seconds() - tE;
}

/* T2/M7: ensure tensor `idx`'s RESIDENT packed words + resident effective scales are present and return
 * them (+ rows, Kpad). Mirrors the residency-ensure half of v3_upload_gpu (first-touch alloc + one-time
 * HtoD straight from the mmap, byte-identical bytes) but does NO dequant -- the fused kernel consumes the
 * packed words directly. Returns 1 if both are resident; 0 if residency is unavailable (latched off /
 * VRAM-full / not packed) so the caller can fall back to the normal dequant+gemv path. */
static int v3_resident_ptrs(int idx, CUdeviceptr* pk, CUdeviceptr* sc, CUdeviceptr* tsp, int* rows_out, int* kpad_out) {
    if (!g_packedres || g_packedres_off) return 0;
    const HXGWv3Desc* d = &g_desc[idx];
    if (!d->packed) return 0;
    int rows=(int)d->rows, Kpad=(int)d->Kpad, kwords=Kpad/7, kblk=Kpad/16, micstride=(kblk+3)/4;
    const unsigned char* base = (const unsigned char*)g_map;
    const int32_t* w = (const int32_t*)(base + d->data_off);
    const uint8_t* micro = (const uint8_t*)(base + d->scale_off);
    float ts; memcpy(&ts, base + d->scale_off + (size_t)rows*kblk, 4);
    int nsc1 = (int)(sizeof(g_pk_res)/sizeof(g_pk_res[0]));
    if (idx < 0 || idx >= nsc1) return 0;
    packedres_fitcheck();
    if (g_packedres_off) return 0;
    /* resident packed words */
    if (!g_pk_res[idx]) {
        CUdeviceptr p = 0;
        memcpy(g_hpin, w, (size_t)rows*kwords*4);                          /* v1.8 FIX: fault mmap -> pinned (cold-mmap DMA can hang) */
        size_t _fr=0,_to=0; cuMemGetInfo(&_fr,&_to);                       /* v1.8 FIX: VRAM-headroom guard -- a cuMemAlloc at ~0 free HANGS on driver 610.62; bail to fallback before exhaustion */
        if (_fr < (size_t)rows*kwords*4 + g_resmargin) { g_packedres_off = 1; return 0; }
        if (cuMemAlloc(&p, (size_t)rows*kwords*4) == CUDA_SUCCESS &&
            cuMemcpyHtoD(p, g_hpin, (size_t)rows*kwords*4) == CUDA_SUCCESS) g_pk_res[idx] = p;
        else { if (p) cuMemFree(p); g_packedres_off = 1; return 0; }
    }
    /* H4: resident RAW e4m3 micro (packed 4/i32-word, ~4x smaller than the fp32 effective) + a 1-elem ts
     * buffer; the fused kernel decodes e4m3_decode(micro)*ts in-kernel (byte-identical to g_e4m3_tab*ts). */
    if (!g_sc_res[idx]) {
        v3_pack_micro(micro, rows, kblk, (int32_t*)g_scbuf);              /* pack into g_scbuf (rows*micstride*4 <= rows*kblk*4 fits) */
        CUdeviceptr p = 0, pts = 0;
        size_t _fr=0,_to=0; cuMemGetInfo(&_fr,&_to);                       /* v1.8 FIX: VRAM-headroom guard */
        if (_fr < (size_t)rows*micstride*4 + g_resmargin) { g_packedres_off = 1; return 0; }
        if (cuMemAlloc(&p, (size_t)rows*micstride*4) == CUDA_SUCCESS &&
            cuMemcpyHtoD(p, g_scbuf, (size_t)rows*micstride*4) == CUDA_SUCCESS &&
            cuMemAlloc(&pts, sizeof(float)) == CUDA_SUCCESS &&
            cuMemcpyHtoD(pts, &ts, sizeof(float)) == CUDA_SUCCESS) { g_sc_res[idx] = p; g_ts_res[idx] = pts; }
        else { if (p) cuMemFree(p); if (pts) cuMemFree(pts); return 0; }
    }
    *pk = g_pk_res[idx]; *sc = g_sc_res[idx]; *tsp = g_ts_res[idx]; *rows_out = rows; *kpad_out = Kpad;
    return 1;
}

/* T2/M7: the fused projection. y[1,N] = x[1,Kpad] . dequant(W_packed[N,Kpad]) in ONE coalesced
 * block-reduction kernel (grid=(N,1,1), block=(256,1,1)). d_xpad MUST be Kpad-padded + zero-tailed.
 * Returns 1 on launch, 0 if the resident weights aren't available (caller falls back to gemv_abt). */
static int fused_gemv(int idx, CUdeviceptr d_xpad, CUdeviceptr d_y, int N) {
    CUdeviceptr pk = 0, sc = 0, tsp = 0; int rows = 0, kpad = 0;
    if (!f_dgemv || !v3_resident_ptrs(idx, &pk, &sc, &tsp, &rows, &kpad)) return 0;
    /* N (logical output rows) must equal the tensor's rows; the kernel writes y[0..rows). H4: sc = packed
     * e4m3 micro, tsp = 1-elem ts buffer -> the kernel decodes the effective scale in-kernel. */
    int kp = kpad; void* ar[] = { &d_xpad, &pk, &sc, &tsp, &d_y, &kp };
    double _t0 = g_dprof ? now_seconds() : 0.0;
    CKX(cuLaunchKernel(f_dgemv, (unsigned)N,1,1, 256,1,1, 0,0, ar, 0), "fused dgemv");
    if (!g_fast) SYNC("sync f_dgemv");
    if (g_dprof) { cuCtxSynchronize(); g_dp_gemv += now_seconds() - _t0; }
    return 1;
}

/* T2/M7b: FUSED lm_head logits. logits[NV] = x[Kpad] . dequant(packed_head[NV,Kpad]) in ONE coalesced
 * block-reduction pass over the RESIDENT packed head words + resident effective scales (g_phk_words /
 * g_phk_sc), exactly like fused_gemv does for the 7 projections -- grid=(g_head_rows,1,1), block=(256,1,1).
 * This SKIPS the INC1.5 chunked fp32 dequant (head_dequant_chunk -> d_head_chunk -> gemv_abt), the biggest
 * remaining decode slice. The head's packed layout is byte-identical to the projections' (setup_head built
 * g_phk_words as [rows x Kpad/7] i32 + g_phk_sc as [rows x Kpad/16] effective e4m3*ts), so the SAME kernel
 * consumes it with kpad=g_head_Kpad. The pad x buffer (d_xpad, the final-normed hidden zero-tailed to its
 * own Kpad) MUST be >= g_head_Kpad so the kernel's full-Kpad sum sees zeros past DM. Rows are written to
 * y[0..g_head_rows) (== NV, the real vocab); the [NV,NVpad) logits are never touched -- but the host argmax
 * + the DtoH copy both span only NV, so untouched pad logits cannot affect the result. Returns 1 on launch,
 * 0 (fail-closed) if the resident packed head isn't available or d_xpad is too small for g_head_Kpad. */
static int fused_lm_head(CUdeviceptr d_xpad, int xpad_kpad, CUdeviceptr d_out) {
    if (!f_dgemv || !g_packedhead_on || !g_phk_words || !g_phk_sc || !g_phk_ts) return 0;  /* H4: need the micro+ts resident */
    if (g_head_rows <= 0 || g_head_Kpad <= 0) return 0;
    if (xpad_kpad < g_head_Kpad) return 0;   /* the pad buffer must cover the full Kpad the kernel sums */
    /* H4: g_phk_sc = packed e4m3 micro, g_phk_ts = 1-elem ts -> the fused kernel decodes the scale in-kernel. */
    int kp = g_head_Kpad;
    void* ar[] = { &d_xpad, &g_phk_words, &g_phk_sc, &g_phk_ts, &d_out, &kp };
    double _t0 = g_dprof ? now_seconds() : 0.0;
    CKX(cuLaunchKernel(f_dgemv, (unsigned)g_head_rows,1,1, 256,1,1, 0,0, ar, 0), "fused lm_head");
    if (!g_fast) SYNC("sync fused lm_head");
    if (g_dprof) { cuCtxSynchronize(); g_dp_lmh += now_seconds() - _t0; }
    return 1;
}

/* v1.8 INC1.5: dequant a single vocab-row chunk [r0, r0+nr) of the resident packed lm_head into
 * d_head_chunk [nr x DM] (compacted Kpad->DM). Mirrors the setup_head chunk loop EXACTLY (same packed
 * words, same g_e4m3_tab*ts scales, same f_dq_tiled, same Kpad->K cuMemcpy2D) so the dequantized rows
 * are byte-identical to what d_wte_pad held. Rows in [NV, r0+nr) (vocab pad) are zeroed -> 0 logits. */
static void head_dequant_chunk(long r0, long nr) {
    int Kpad = g_head_Kpad, kwords = g_head_kwords, kblk = g_head_kblk, BD = 112;
    long real = nr;                                   /* real (non-pad) rows in this chunk */
    if (r0 + real > (long)NV) real = (long)NV - r0;   /* clamp: rows >= NV are vocab pad */
    if (real < 0) real = 0;
    if (nr > real)   /* zero the dequant target tail so pad rows produce 0 logits (like the zeroed d_wte_pad) */
        CKX(cuMemsetD8(d_head_chunk + (CUdeviceptr)((size_t)real*DM*sizeof(float)), 0,
                       (size_t)(nr-real)*DM*sizeof(float)), "head pad zero");
    if (real <= 0) return;
    CUdeviceptr wsrc = g_phk_words + (CUdeviceptr)((size_t)r0*kwords*4);   /* resident packed words for [r0,..) */
    CUdeviceptr ssrc;
    if (g_fusedgemv) {
        /* H4: g_phk_sc holds RAW e4m3 micro (for the fused path); f_dq_tiled needs the fp32 EFFECTIVE scale,
         * so build it for this row-chunk from the mmap into the scratch d_sc (byte-identical g_e4m3_tab*ts).
         * This fallback fires only when fused_lm_head returned 0 (residency dropped); rare, so per-chunk
         * rebuild is fine. real*kblk <= CHUNK*kblk <= g_dqscr_elems/16 fits g_scbuf. */
        const unsigned char* base = (const unsigned char*)g_map;
        const HXGWv3Desc* dl = &g_desc[v3_idx_lmhead()];
        const uint8_t* mb = (const uint8_t*)(base + dl->scale_off);
        for (long i=0;i<real*(long)kblk;i++) g_scbuf[i] = g_e4m3_tab[mb[(size_t)r0*kblk + i]] * g_head_ts;
        CKX(cuMemcpyHtoD(d_sc, g_scbuf, (size_t)real*kblk*4), "head chunk fp32 scale HtoD");
        ssrc = d_sc;
    } else {
        ssrc = g_phk_sc + (CUdeviceptr)((size_t)r0*kblk*4);               /* resident effective scales for [r0,..) */
    }
    int mm0=0, kk=Kpad, bd=BD; void* ar[] = { &wsrc, &ssrc, &d_dqscr, &mm0, &kk, &bd };
    LX(f_dq_tiled, (unsigned)((long)real*(Kpad/BD)), (unsigned)BD, ar);
    CUDA_MEMCPY2D cp; memset(&cp,0,sizeof(cp));   /* compact [real x Kpad] -> d_head_chunk rows [0,real) [real x DM] */
    cp.srcMemoryType=CU_MEMORYTYPE_DEVICE; cp.srcDevice=d_dqscr;      cp.srcPitch=(size_t)Kpad*4;
    cp.dstMemoryType=CU_MEMORYTYPE_DEVICE; cp.dstDevice=d_head_chunk; cp.dstPitch=(size_t)DM*4;
    cp.WidthInBytes=(size_t)DM*4; cp.Height=(size_t)real;
    CKX(cuMemcpy2D(&cp), "head chunk compact 2D");
}

/* v1.8 INC1.5: compute lm_head logits without the persistent FP32 d_wte_pad. Loops vocab-row chunks of
 * the resident packed head, dequanting each into d_head_chunk then running the chunk's GEMM/GEMV into the
 * right slice of d_out. Math-identical to mm_ABt/gemv_abt over the full d_wte_pad.
 *   is_gemv==0 (prefill, M>1): d_out is [M x NVpad] row-major; gemm temp[M,nr]=d_in@chunk^T then scatter
 *     the nr columns into d_out[:, r0:r0+nr] (non-contiguous per row -> cuMemcpy2D scatter).
 *   is_gemv==1 (decode, M==1):  d_out is [NVpad]; gemv writes nr contiguous outputs at d_out + r0.
 * Chunks tile [0, NVpad); CHUNK and NVpad are both %64 so every nr is %64 (mm_ABt N/64 grid is valid). */
static void lm_head_logits_chunked(CUdeviceptr d_in, CUdeviceptr d_out, int M, int is_gemv) {
    long CHUNK = g_head_chunk_rows;
    for (long r0 = 0; r0 < (long)NVpad; r0 += CHUNK) {
        long nr = (r0 + CHUNK <= (long)NVpad) ? CHUNK : ((long)NVpad - r0);   /* r0,CHUNK,NVpad %64 -> nr %64 */
        head_dequant_chunk(r0, nr);
        if (is_gemv) {
            /* y[r0:r0+nr] = d_head_chunk[nr,DM] @ d_in[DM]; contiguous write at d_out + r0. */
            gemv_abt(d_in, d_head_chunk, d_out + (CUdeviceptr)((size_t)r0*sizeof(float)), (int)nr, DM);
        } else {
            /* temp[M,nr] = d_in[M,DM] @ d_head_chunk[nr,DM]^T (M%64 from prefill, nr%64). */
            mm_ABt(d_in, d_head_chunk, d_head_temp, M, DM, (int)nr);
            /* scatter temp[M,nr] (pitch nr) -> d_out[:, r0:r0+nr] (pitch NVpad, col offset r0). */
            CUDA_MEMCPY2D cp; memset(&cp,0,sizeof(cp));
            cp.srcMemoryType=CU_MEMORYTYPE_DEVICE; cp.srcDevice=d_head_temp; cp.srcPitch=(size_t)nr*4;
            cp.dstMemoryType=CU_MEMORYTYPE_DEVICE;
            cp.dstDevice=d_out + (CUdeviceptr)((size_t)r0*sizeof(float)); cp.dstPitch=(size_t)NVpad*4;
            cp.WidthInBytes=(size_t)nr*4; cp.Height=(size_t)M;
            CKX(cuMemcpy2D(&cp), "head logits scatter 2D");
        }
    }
}

static void v3_upload(int idx, CUdeviceptr dst) {
    if (g_gpu_dq && !g_hostdeq) { v3_upload_gpu(idx, dst); return; }
    const HXGWv3Desc* d = &g_desc[idx];
    int rows=(int)d->rows, K=(int)d->K, Kpad=(int)d->Kpad;
    v3_get_tensor(idx, g_v3scratch);                   /* packed -> [rows x Kpad]; f32 -> [rows x K] */
    if (d->packed && Kpad != K)                         /* compact [rows x Kpad] -> [rows x K] (drop pad);
                                                         * increasing-r memmove is overlap-safe (dst r*K
                                                         * < src r*Kpad, and row r's dst ends before row
                                                         * r+1's src starts). */
        for (int r=1;r<rows;r++) memmove(g_v3scratch + (size_t)r*K, g_v3scratch + (size_t)r*Kpad, (size_t)K*4);
    CKX(cuMemcpyHtoD(dst, g_v3scratch, (size_t)rows*K*sizeof(float)), "v3 HtoD");
}
static void upload_layer_ll(int L) {
    if (res_loaded) {   /* resident: zero-copy pointer swap */
        d_ln1g = res_inln[L]; d_qW = res_qW[L]; d_kW = res_kW[L]; d_vW = res_vW[L];
        d_oW = res_oW[L]; d_ln2g = res_postln[L]; d_gateW = res_gateW[L];
        d_upW = res_upW[L]; d_downW = res_downW[L];
        return;
    }
    if (QWEN3) {   /* v3 descriptor-driven dequant-on-upload (11 tensors/layer; +q_norm/k_norm) */
        v3_upload(v3_idx_layer(L, V3_INLN),   d_ln1g);
        v3_upload(v3_idx_layer(L, V3_Q),      d_qW);
        v3_upload(v3_idx_layer(L, V3_QN),     d_qnorm);
        v3_upload(v3_idx_layer(L, V3_K),      d_kW);
        v3_upload(v3_idx_layer(L, V3_KN),     d_knorm);
        v3_upload(v3_idx_layer(L, V3_V),      d_vW);
        v3_upload(v3_idx_layer(L, V3_O),      d_oW);
        v3_upload(v3_idx_layer(L, V3_POSTLN), d_ln2g);
        v3_upload(v3_idx_layer(L, V3_GATE),   d_gateW);
        v3_upload(v3_idx_layer(L, V3_UP),     d_upW);
        v3_upload(v3_idx_layer(L, V3_DOWN),   d_downW);
        return;
    }
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

/* T2/M7: the fused-decode upload. Uploads ONLY the 4 norm tensors (in_ln / q_norm / k_norm / post_ln);
 * the 7 gemv-projection weights (q/k/v/o/gate/up/down) are consumed PACKED+RESIDENT by the fused kernel,
 * so their per-token dequant (the ~52% the fused path removes) is SKIPPED entirely. QWEN3-only; the
 * caller guarantees g_fusedgemv_on (which implies residency). Returns nothing. */
static void upload_layer_ll_fused(int L) {
    v3_upload(v3_idx_layer(L, V3_INLN),   d_ln1g);
    v3_upload(v3_idx_layer(L, V3_QN),     d_qnorm);
    v3_upload(v3_idx_layer(L, V3_KN),     d_knorm);
    v3_upload(v3_idx_layer(L, V3_POSTLN), d_ln2g);
}

/* set up the tied LM head: ln_f buffers + the padded wte [NVpad,DM] on device + the logits buffer.
 * The tied head is wte itself (logits = x @ wte^T); we copy the NV real rows into a NVpad-row device
 * buffer whose pad rows [NV..NVpad) are zeroed so their logits are 0 and never argmaxed. */
static int setup_head(int Smax) {
    NVpad = ((NV + 63) / 64) * 64;     /* 50257 -> 50304 (llama 49152 is already %64; qwen3 151936 is %64) */
    if (QWEN3) {   /* v3: final RMSNorm (f32 v3 tensor) + the UNTIED lm_head (packed v3 tensor) */
        d_lnfg = A(DM);
        v3_get_tensor(v3_idx_norm(), g_v3scratch);
        CK(cuMemcpyHtoD(d_lnfg, g_v3scratch, (size_t)DM*sizeof(float)), "h2d v3 norm_f");
        d_lnfb = 0;
        const HXGWv3Desc* dl = &g_desc[v3_idx_lmhead()];
        int rows=(int)dl->rows, K=(int)dl->K, Kpad=(int)dl->Kpad;   /* NV x DM, Kpad>=DM */
        /* v1.8 INC1.5: resident PACKED lm_head instead of the 2.49 GiB FP32 d_wte_pad. Keep the packed
         * words+effective-scales resident (alloc once, HtoD once from the mmap) + a small reusable dequant
         * target d_head_chunk [CHUNK x DM] + a prefill GEMM temp d_head_temp [Smax x CHUNK]. lm_head_logits_
         * chunked() then dequants+GEMMs in vocab-row chunks. Byte-identical, opt-in, non-fatal fallback. */
        if (g_packedhead && g_gpu_dq && !g_hostdeq && dl->packed) {
            int kwords=Kpad/7, kblk=Kpad/16, micstride=(kblk+3)/4;
            int hfused = g_fusedgemv;                                       /* H4: resident scale = packed micro (fused) vs fp32 effective (chunked) */
            long CHUNK = g_dqscr_elems / Kpad; if (CHUNK < 1) CHUNK = 1;   /* same chunk count as the setup loop */
            if (CHUNK > NVpad) CHUNK = NVpad;
            if (CHUNK % 64) CHUNK -= (CHUNK % 64);                          /* keep nr %64 for the mm_ABt N/64 grid */
            if (CHUNK < 64) CHUNK = 64;
            size_t scbytes = hfused ? (size_t)rows*micstride*4 : (size_t)rows*kblk*4;  /* H4: ~4x smaller resident scale when fused */
            size_t need = (size_t)rows*kwords*4 + scbytes                   /* resident packed words + scales */
                        + (size_t)CHUNK*DM*4 + (size_t)Smax*CHUNK*4;        /* d_head_chunk + d_head_temp */
            size_t freeb=0, totb=0; cuMemGetInfo(&freeb, &totb);
            size_t margin = (size_t)256*1024*1024;
            CUdeviceptr pw=0, ps=0, pc=0, pt=0, pts=0;
            const unsigned char* lb = (const unsigned char*)g_map;
            const int32_t* wb = (const int32_t*)(lb + dl->data_off);
            const uint8_t*  mb = (const uint8_t*)(lb + dl->scale_off);
            float ts; memcpy(&ts, lb + dl->scale_off + (size_t)rows*kblk, 4);
            int ok = (freeb >= need + margin);
            if (ok) ok = (cuMemAlloc(&pw, (size_t)rows*kwords*4) == CUDA_SUCCESS);
            if (ok) ok = (cuMemAlloc(&ps, scbytes)               == CUDA_SUCCESS);
            if (ok) ok = (cuMemAlloc(&pc, (size_t)CHUNK*DM*4)    == CUDA_SUCCESS);
            if (ok) ok = (cuMemAlloc(&pt, (size_t)Smax*CHUNK*4)  == CUDA_SUCCESS);
            if (ok && hfused) ok = (cuMemAlloc(&pts, sizeof(float)) == CUDA_SUCCESS);
            if (ok && hfused) ok = (cuMemcpyHtoD(pts, &ts, sizeof(float)) == CUDA_SUCCESS);
            if (ok) ok = (cuMemcpyHtoD(pw, wb, (size_t)rows*kwords*4) == CUDA_SUCCESS);  /* one-time, from the mmap */
            if (ok && hfused) {   /* H4: resident RAW e4m3 micro (packed 4/i32-word), decoded in-kernel by fused_lm_head. */
                for (long r0=0; ok && r0<rows; r0+=CHUNK) {
                    long nr = (r0+CHUNK<=rows) ? CHUNK : (rows-r0);
                    v3_pack_micro(mb + (size_t)r0*kblk, nr, kblk, (int32_t*)g_scbuf);
                    ok = (cuMemcpyHtoD(ps + (CUdeviceptr)((size_t)r0*micstride*4), g_scbuf, (size_t)nr*micstride*4) == CUDA_SUCCESS);
                }
            } else if (ok) {   /* non-fused: build the fp32 effective scales (e4m3_decode(micro)*ts, byte-identical) in ROW-CHUNKS. */
                for (long r0=0; ok && r0<rows; r0+=CHUNK) {
                    long nr = (r0+CHUNK<=rows) ? CHUNK : (rows-r0);
                    for (long i=0;i<nr*(long)kblk;i++) g_scbuf[i] = g_e4m3_tab[mb[(size_t)r0*kblk + i]] * ts;
                    ok = (cuMemcpyHtoD(ps + (CUdeviceptr)((size_t)r0*kblk*4), g_scbuf, (size_t)nr*kblk*4) == CUDA_SUCCESS);
                }
            }
            if (ok) {
                g_phk_words = pw; g_phk_sc = ps; g_phk_ts = pts; d_head_chunk = pc; d_head_temp = pt;
                g_head_chunk_rows = (int)CHUNK; g_head_rows = rows; g_head_kwords = kwords;
                g_head_kblk = kblk; g_head_Kpad = Kpad; g_head_ts = ts;
                g_packedhead_on = 1;
                d_logits = A((size_t)Smax*NVpad);
                fprintf(stderr, "[packedhead] resident ON: packed lm_head [%dx%d] chunk=%ld rows, "
                                "scale=%s need %zu MiB free %zu MiB (no FP32 d_wte_pad)\n",
                        rows, K, CHUNK, hfused?"raw-e4m3-micro":"fp32-effective", need/(1024*1024), freeb/(1024*1024));
                return 0;
            }
            if (pw) cuMemFree(pw); if (ps) cuMemFree(ps); if (pts) cuMemFree(pts);  /* non-fatal: release + fall to FP32 */
            if (pc) cuMemFree(pc); if (pt) cuMemFree(pt);
            fprintf(stderr, "[packedhead] need %zu MiB +margin, free %zu MiB -> FP32 d_wte_pad fallback\n",
                    need/(1024*1024), freeb/(1024*1024));
        }
        d_wte_pad = A((size_t)NVpad*DM);
        CK(cuMemsetD8(d_wte_pad, 0, (size_t)NVpad*DM*sizeof(float)), "zero wte_pad");
        if (g_gpu_dq && !g_hostdeq && dl->packed) {
            /* v1.7 INCREMENT 2: GPU-dequant the ~630M-element lm_head in row-chunks (reusing the layer
             * scratch d_packed/d_sc/d_dqscr), replacing the ~22s single-threaded host v3_get_tensor.
             * Byte-identical (same g_e4m3_tab[micro]*ts * E2M1-mag); writes straight into d_wte_pad's
             * first NV rows. Drops the 2.5GB host scratch (less RAM pressure) as a bonus. */
            const unsigned char* lb = (const unsigned char*)g_map;
            int kwords=Kpad/7, kblk=Kpad/16, BD=112;
            const int32_t* wb = (const int32_t*)(lb + dl->data_off);
            const uint8_t*  mb = (const uint8_t*)(lb + dl->scale_off);
            float ts; memcpy(&ts, lb + dl->scale_off + (size_t)rows*kblk, 4);
            long chunk = g_dqscr_elems / Kpad; if (chunk < 1) chunk = 1; if (chunk > rows) chunk = rows;
            double t_lm = g_prof ? now_seconds() : 0.0;
            for (long r0=0; r0<rows; r0+=chunk) {
                long nr = (r0+chunk<=rows) ? chunk : (rows-r0);
                memcpy(g_hpin, wb + (size_t)r0*kwords, (size_t)nr*kwords*4);   /* v1.7 INC2b: pinned bounce -> DMA */
                CK(cuMemcpyHtoD(d_packed, g_hpin, (size_t)nr*kwords*4), "lm packed HtoD");
                for (long i=0;i<nr*(long)kblk;i++) g_scbuf[i] = g_e4m3_tab[mb[(size_t)r0*kblk + i]] * ts;
                CK(cuMemcpyHtoD(d_sc, g_scbuf, (size_t)nr*kblk*4), "lm scale HtoD");
                int mm0=0, kk=Kpad, bd=BD; void* ar[]={ &d_packed,&d_sc,&d_dqscr,&mm0,&kk,&bd };
                LX(f_dq_tiled, (unsigned)((long)nr*(Kpad/BD)), (unsigned)BD, ar);
                CUDA_MEMCPY2D cp; memset(&cp,0,sizeof(cp));   /* compact [nr x Kpad] -> d_wte_pad rows [r0,r0+nr) [nr x K] */
                cp.srcMemoryType=CU_MEMORYTYPE_DEVICE; cp.srcDevice=d_dqscr; cp.srcPitch=(size_t)Kpad*4;
                cp.dstMemoryType=CU_MEMORYTYPE_DEVICE; cp.dstDevice=d_wte_pad+(CUdeviceptr)((size_t)r0*DM*sizeof(float)); cp.dstPitch=(size_t)DM*4;
                cp.WidthInBytes=(size_t)K*4; cp.Height=(size_t)nr;
                CK(cuMemcpy2D(&cp), "lm compact 2D");
            }
            if (g_prof) fprintf(stderr, "[prof] lm_head GPU dequant (%dx%d, chunk=%ld rows): %.2fs\n", rows, Kpad, chunk, now_seconds()-t_lm);
        } else {
            float* hs = (float*)malloc((size_t)rows*Kpad*sizeof(float));   /* ~2.5GB host (lm_head dequant) */
            if (!hs) { fprintf(stderr, "lm_head host scratch [%dx%d] malloc failed\n", rows, Kpad); return 2; }
            double t_lm = g_prof ? now_seconds() : 0.0;
            v3_get_tensor(v3_idx_lmhead(), hs);                           /* [NV x Kpad] */
            if (g_prof) fprintf(stderr, "[prof] lm_head HOST dequant (%dx%d): %.2fs\n", rows, Kpad, now_seconds()-t_lm);
            if (Kpad != K) for (int r=1;r<rows;r++) memmove(hs+(size_t)r*K, hs+(size_t)r*Kpad, (size_t)K*sizeof(float));
            CK(cuMemcpyHtoD(d_wte_pad, hs, (size_t)rows*K*sizeof(float)), "h2d v3 lm_head");
            free(hs);
        }
        d_logits = A((size_t)Smax*NVpad);
        fprintf(stderr, "[head] qwen3 UNTIED lm_head [%dx%d] -> d_wte_pad, final-norm loaded\n", rows, K);
        return 0;
    }
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

/* --v3-dequant-dump <weights.v3> <tensor_idx> <out.bin>: self-contained (no GPU) reader+dequant of
 * ONE HXGW v3 tensor by its build-order index -> f32 to out.bin ([rows x Kpad] for packed, [rows x K]
 * for f32). STEP-1 gate: must be byte-identical to gpt2_pack --nvfp4-testmodel's dequant of the same
 * tensor (proves the worker reads the v3 format + dequants exactly like the importer). */
static int run_v3_dequant_dump(const char* wpath, int idx, const char* outbin) {
    int fd = open(wpath, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open %s: %s\n", wpath, strerror(errno)); return 2; }
    struct stat st; if (fstat(fd, &st) != 0) { fprintf(stderr, "fstat\n"); return 2; }
    unsigned char* base = (unsigned char*)mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { fprintf(stderr, "mmap %s\n", strerror(errno)); return 2; }
    uint32_t magic, ver, ntens;
    memcpy(&magic, base, 4); memcpy(&ver, base+4, 4); memcpy(&ntens, base+60, 4);
    if (magic != MAGIC) { fprintf(stderr, "bad magic 0x%08x\n", magic); return 2; }
    if (ver != 3u) { fprintf(stderr, "not an HXGW v3 file (ver=%u)\n", ver); return 2; }
    if (idx < 0 || idx >= (int)ntens) { fprintf(stderr, "idx %d out of range [0,%u)\n", idx, ntens); return 2; }
    HXGWv3Desc* desc = (HXGWv3Desc*)(base + HDR_BYTES);
    HXGWv3Desc d = desc[idx];
    fprintf(stderr, "[v3-dequant] idx=%d packed=%u rows=%u K=%u Kpad=%u data_off=%llu scale_off=%llu\n",
            idx, d.packed, d.rows, d.K, d.Kpad, (unsigned long long)d.data_off, (unsigned long long)d.scale_off);
    FILE* of = fopen(outbin, "wb"); if (!of) { fprintf(stderr, "open out %s\n", outbin); return 2; }
    if (d.packed) {
        int rows=(int)d.rows, Kpad=(int)d.Kpad, kblk=Kpad/16;
        const int32_t* w = (const int32_t*)(base + d.data_off);
        const uint8_t* micro = (const uint8_t*)(base + d.scale_off);
        float ts; memcpy(&ts, base + d.scale_off + (size_t)rows*kblk, 4);
        float* out = (float*)malloc((size_t)rows*Kpad*4);
        v3_dequant_packed(w, micro, ts, out, rows, Kpad);
        size_t nw = fwrite(out, 4, (size_t)rows*Kpad, of); free(out);
        fprintf(stderr, "[v3-dequant] wrote %zu f32 ([rows x Kpad]) ts=%.6g\n", nw, (double)ts);
    } else {
        long n = (long)d.rows * d.K;
        size_t nw = fwrite(base + d.data_off, 4, (size_t)n, of);
        fprintf(stderr, "[v3-dequant] wrote %zu f32 (dense)\n", nw);
    }
    fclose(of); munmap(base, (size_t)st.st_size); close(fd);
    printf("V3_DEQUANT_DUMP_OK idx=%d packed=%u rows=%u Kpad=%u\n", idx, d.packed, d.rows, d.Kpad);
    return 0;
}

/* --v3-upload-check <weights.v3>: GPU gate for the v3 RUN path. peek + minimal CUDA init (no PTX) +
 * load + alloc + upload_layer_ll(0) (dequant-on-upload) -> read each layer-0 weight back (DtoH) and
 * assert it equals the host dequant -> proves the descriptor indexing + dequant + padding-drop +
 * buffer sizing on the GPU, before the forward. */
static int run_v3_upload_check(const char* wpath) {
    if (peek_weights_header(wpath)) return 2;
    CK(cuInit(0), "init");
    CUdevice dev; CK(cuDeviceGet(&dev, 0), "dev");
    char gpu[256] = {0}; cuDeviceGetName(gpu, 256, dev);
    CK(cuCtxCreate(&ctx, 0, dev), "ctx");
    if (load_gpt2_weights(wpath)) return 2;
    if (!QWEN3) { fprintf(stderr, "upload-check: not a v3/qwen3 file\n"); return 2; }
    fprintf(stderr, "[upload-check] GPU=%s; uploading Qwen3 layer 0 (dequant-on-upload)...\n", gpu);
    load_dequant_module();   /* v1.7: so this gate exercises the GPU dequant path (g_gpu_dq) vs the host ref */
    if (alloc_buffers(64)) return 2;
    upload_layer_ll(0);
    CKX(cuCtxSynchronize(), "sync upload");
    int idxs[11] = {V3_INLN,V3_Q,V3_QN,V3_K,V3_KN,V3_V,V3_O,V3_POSTLN,V3_GATE,V3_UP,V3_DOWN};
    CUdeviceptr ptrs[11] = {d_ln1g,d_qW,d_qnorm,d_kW,d_knorm,d_vW,d_oW,d_ln2g,d_gateW,d_upW,d_downW};
    const char* nms[11] = {"input_ln","q_proj","q_norm","k_proj","k_norm","v_proj","o_proj","post_ln","gate","up","down"};
    int allok = 1;
    for (int i = 0; i < 11; i++) {
        int gi = v3_idx_layer(0, idxs[i]);
        const HXGWv3Desc* d = &g_desc[gi];
        int rows=(int)d->rows, K=(int)d->K, Kpad=(int)d->Kpad;
        long nK = (long)rows * K;
        float* ref = (float*)malloc((size_t)nK*4);
        float* tmp = (float*)malloc((size_t)rows*Kpad*4);
        v3_get_tensor(gi, tmp);
        if (d->packed) for (int r=0;r<rows;r++) memcpy(ref+(size_t)r*K, tmp+(size_t)r*Kpad, (size_t)K*4);
        else           memcpy(ref, tmp, (size_t)nK*4);
        float* got = (float*)malloc((size_t)nK*4);
        CKX(cuMemcpyDtoH(got, ptrs[i], (size_t)nK*4), "d2h check");
        int mism = memcmp(ref, got, (size_t)nK*4) != 0;
        printf("[upload-check] L0 %-9s idx=%-3d packed=%u [%d x %d] -> %s\n",
               nms[i], gi, d->packed, rows, K, mism ? "MISMATCH" : "match");
        if (mism) allok = 0;
        free(ref); free(tmp); free(got);
    }
    printf("%s\n", allok ? "V3_UPLOAD_CHECK_PASS" : "V3_UPLOAD_CHECK_FAIL");
    return allok ? 0 : 1;
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

/* --v3-smoke <ids.txt>: run the FULL Qwen3 forward (36 layers + untied head) + assert finite logits +
 * report the argmax next-token. Dumps the [NV] logits to /home/legoa/q3_logits.bin for the next-tick
 * oracle compare. SMOKE = executes + finite + plausible argmax; correctness is the oracle's job. */
static int run_v3_smoke(const char* ids_path) {
    int ids[1024];
    int T = read_ids_file(ids_path, ids, 1024);
    if (T <= 0) { fprintf(stderr, "no ids in %s\n", ids_path); return 2; }
    printf("[v3-smoke] T=%d ids:", T); for (int i=0;i<T;i++) printf(" %d", ids[i]); printf("\n");
    float* logits = (float*)malloc((size_t)NV*sizeof(float));
    forward_full(ids, T, logits);
    if (g_prof) fprintf(stderr, "[prof] %ld uploads: mmap_touch+HtoD=%.2fs cpu_scale_build=%.2fs dequant_launch+sync=%.2fs compact2D=%.2fs\n",
                        g_pf_n, g_pf_htod, g_pf_sb, g_pf_dq, g_pf_cmp);
    int nonfinite = 0; double mx = 0.0;
    for (int i=0;i<NV;i++){ double g=(double)logits[i]; if(!isfinite(g)){nonfinite++; continue;} double a=fabs(g); if(a>mx)mx=a; }
    int am = argmax_row(logits, NV);
    FILE* of = fopen("/home/legoa/q3_logits.bin", "wb");
    if (of) { fwrite(logits, sizeof(float), (size_t)NV, of); fclose(of); }
    printf("[v3-smoke] logits nonfinite=%d max_abs=%.5g argmax=%d (valid_token=%d)\n",
           nonfinite, mx, am, (am>=0 && am<NV));
    free(logits);
    printf("%s argmax=%d nonfinite=%d\n", nonfinite==0 ? "V3_SMOKE_OK" : "V3_SMOKE_FAIL", am, nonfinite);
    return nonfinite==0 ? 0 : 1;
}

/* v1.6 Tier-3 envelope tau. RELEASE path: a CALIBRATED bound supplied via HX_TIER3_TAU (the
 * empirical-TAO pattern -- tau = a documented multiple of the max logit deviation measured over a
 * calibration prompt set, a property of the (model,quant), NOT the per-run value), with HX_TAU_PROV
 * labeling its provenance in the receipt. DEV default (env unset): the per-run measured max_abs. A
 * calibrated tau is what gives Tier-3 independent teeth (a drifted run with max_abs>tau is rejected). */
static double calib_tau(double measured_max_abs) {
    const char* e = getenv("HX_TIER3_TAU");
    double t = e ? atof(e) : 0.0;
    return (t > 0.0) ? t : measured_max_abs;
}

/* --v3-receipt <ids> <oracle_logits.bin> <out.receipt>: run forward_full + emit a Tier-2(commitment)
 * + Tier-3(envelope) v1.6 receipt. SCOPE (honest, per docs/HELIX_V1.6_DEFINITION_OF_DONE.md):
 *   Tier-2 = reproducibility: SHA-256(committed packed weights) + ids + SHA-256(output logits) bind
 *            the run; a verifier RE-DERIVES (re-runs the quantized forward, deterministic) -- NOT
 *            faster-than-re-exec (that is the deferred Tier-1 exact Freivalds; f32 GEMM -> a
 *            tolerance-Freivalds would be UNSOUND, so it is NOT built/claimed here).
 *   Tier-3 = the run's logits within tau of a TRUSTED f32 oracle (max_abs + argmax) -- EMPIRICAL,
 *            never cryptographic. Prior art: CommitLLM, TAO (cited; this is minimal-TRUST not first). */
static int run_v3_receipt(const char* ids_path, const char* oracle_path, const char* out_path) {
    int ids[1024];
    int T = read_ids_file(ids_path, ids, 1024);
    if (T <= 0) { fprintf(stderr, "no ids in %s\n", ids_path); return 2; }
    float* logits = (float*)malloc((size_t)NV*sizeof(float));
    forward_full(ids, T, logits);
    int am = argmax_row(logits, NV);
    /* persist the logits next to the receipt (the checker hashes THIS file) */
    char lpath[1100]; snprintf(lpath, sizeof(lpath), "%s.logits", out_path);
    { FILE* lf = fopen(lpath, "wb"); if (lf) { fwrite(logits, sizeof(float), (size_t)NV, lf); fclose(lf); } }
    char model_hex[65] = {0}, logits_hex[65] = {0};
    if (!g_map) { fprintf(stderr, "receipt: weights not mmapped\n"); free(logits); return 2; }
    { unsigned char d[32]; sha256((const unsigned char*)g_map, g_maplen, d); sha256_hex(d, model_hex); }
    { unsigned char d[32]; sha256((const unsigned char*)logits, (size_t)NV*sizeof(float), d); sha256_hex(d, logits_hex); }
    double max_abs = -1.0; int argmatch = -1;
    { FILE* of = fopen(oracle_path, "rb");
      if (of) { float* orc = (float*)malloc((size_t)NV*sizeof(float));
        size_t got = fread(orc, sizeof(float), (size_t)NV, of); fclose(of);
        if (got == (size_t)NV) { double m=0; for (int i=0;i<NV;i++){ double e=fabs((double)logits[i]-(double)orc[i]); if(e>m)m=e; }
          max_abs = m; argmatch = (am == argmax_row(orc, NV)) ? 1 : 0; }
        free(orc); } }
    FILE* rf = fopen(out_path, "wb");
    if (!rf) { fprintf(stderr, "open receipt %s\n", out_path); free(logits); return 2; }
    fprintf(rf, "HELIX_V16_RECEIPT\n");
    fprintf(rf, "model_sha256 %s\n", model_hex);
    fprintf(rf, "ids"); for (int i=0;i<T;i++) fprintf(rf, " %d", ids[i]); fprintf(rf, "\n");
    fprintf(rf, "n_logits %d\n", NV);
    fprintf(rf, "logits_sha256 %s\n", logits_hex);
    fprintf(rf, "argmax %d\n", am);
    fprintf(rf, "tier3_tau %.17g\n", calib_tau(max_abs));   /* calibrated bound (HX_TIER3_TAU) or per-run max_abs; %.17g = exact double round-trip */
    fprintf(rf, "tier3_max_abs %.17g\n", max_abs);
    fprintf(rf, "tier3_argmax_match %d\n", argmatch);
    { const char* tp = getenv("HX_TAU_PROV"); if (tp && *tp) fprintf(rf, "tier3_tau_prov %s\n", tp); }
    fprintf(rf, "note Tier-2=reproducibility(re-derive from committed weights, NOT faster-than-re-exec); Tier-3=envelope vs a TRUSTED f32 oracle (EMPIRICAL not crypto; argmax-preservation empirical when top1-margin<2*tau); Tier-1 exact-Freivalds DEFERRED (f32 GEMM). Prior art CommitLLM, TAO.\n");
    fclose(rf);
    printf("V3_RECEIPT_EMIT_OK argmax=%d model_sha=%.12s logits_sha=%.12s tier3_max_abs=%.4f argmatch=%d -> %s\n",
           am, model_hex, logits_hex, max_abs, argmatch, out_path);
    free(logits);
    return 0;
}

/* --v3-receipt-from-logits <logits.bin> <weights> <ids.txt> <oracle.bin> <out.receipt>: GPU-FREE
 * re-attestation. Re-serialize a v1.6 receipt from an ALREADY-DUMPED logits file (produced by a real
 * --v3-receipt / --v3-smoke GPU run) + the committed weights + the f32 oracle -- no forward, no CUDA.
 * Identical fields to run_v3_receipt (same SHA-256 commitments, same envelope) except tau is written at
 * full double precision and there is no GPU pass. Use to re-emit a stored run's receipt (e.g. with the
 * corrected tau precision) or to re-attest off-GPU; the GPU --v3-receipt is the canonical emit. NV is
 * taken from the logits file size. */
static int run_v3_receipt_from_logits(const char* logits_path, const char* weights_path,
                                      const char* ids_path, const char* oracle_path, const char* out_path) {
    FILE* lf = fopen(logits_path, "rb");
    if (!lf) { fprintf(stderr, "from-logits: open logits %s\n", logits_path); return 2; }
    fseek(lf, 0, SEEK_END); long lsz = ftell(lf); fseek(lf, 0, SEEK_SET);
    int nlog = (int)(lsz / (long)sizeof(float));
    if (nlog <= 0) { fprintf(stderr, "from-logits: empty logits %s\n", logits_path); fclose(lf); return 2; }
    float* logits = (float*)malloc((size_t)nlog * sizeof(float));
    if (!logits) { fclose(lf); return 2; }
    if (fread(logits, sizeof(float), (size_t)nlog, lf) != (size_t)nlog) {
        fprintf(stderr, "from-logits: short logits read\n"); fclose(lf); free(logits); return 2; }
    fclose(lf);
    int ids[1024];
    int T = read_ids_file(ids_path, ids, 1024);
    if (T <= 0) { fprintf(stderr, "from-logits: no ids in %s\n", ids_path); free(logits); return 2; }
    int am = argmax_row(logits, nlog);
    char model_hex[65] = {0}, logits_hex[65] = {0};
    if (!sha256_file(weights_path, model_hex)) { fprintf(stderr, "from-logits: hash weights %s\n", weights_path); free(logits); return 2; }
    if (!sha256_file(logits_path, logits_hex)) { fprintf(stderr, "from-logits: hash logits %s\n", logits_path); free(logits); return 2; }
    double max_abs = -1.0; int argmatch = -1;
    { FILE* of = fopen(oracle_path, "rb");
      if (of) { float* orc = (float*)malloc((size_t)nlog*sizeof(float));
        size_t got = orc ? fread(orc, sizeof(float), (size_t)nlog, of) : 0; fclose(of);
        if (got == (size_t)nlog) { double m=0; for (int i=0;i<nlog;i++){ double e=fabs((double)logits[i]-(double)orc[i]); if(e>m)m=e; }
          max_abs = m; argmatch = (am == argmax_row(orc, nlog)) ? 1 : 0; }
        free(orc); } }
    FILE* rf = fopen(out_path, "wb");
    if (!rf) { fprintf(stderr, "from-logits: open receipt %s\n", out_path); free(logits); return 2; }
    fprintf(rf, "HELIX_V16_RECEIPT\n");
    fprintf(rf, "model_sha256 %s\n", model_hex);
    fprintf(rf, "ids"); for (int i=0;i<T;i++) fprintf(rf, " %d", ids[i]); fprintf(rf, "\n");
    fprintf(rf, "n_logits %d\n", nlog);
    fprintf(rf, "logits_sha256 %s\n", logits_hex);
    fprintf(rf, "argmax %d\n", am);
    fprintf(rf, "tier3_tau %.17g\n", calib_tau(max_abs));   /* calibrated bound (HX_TIER3_TAU) or per-run max_abs */
    fprintf(rf, "tier3_max_abs %.17g\n", max_abs);
    fprintf(rf, "tier3_argmax_match %d\n", argmatch);
    { const char* tp = getenv("HX_TAU_PROV"); if (tp && *tp) fprintf(rf, "tier3_tau_prov %s\n", tp); }
    fprintf(rf, "note re-serialized from dumped logits (GPU-free re-attest); tau at full double precision; Tier-2=reproducibility (re-derive from committed weights, NOT faster-than-re-exec); Tier-3=EMPIRICAL envelope vs a TRUSTED f32 oracle (NOT crypto); Tier-1 exact-Freivalds DEFERRED. Prior art CommitLLM, TAO.\n");
    fclose(rf);
    printf("V3_RECEIPT_FROM_LOGITS_OK n_logits=%d argmax=%d model_sha=%.12s logits_sha=%.12s tier3_max_abs=%.9g argmatch=%d -> %s\n",
           nlog, am, model_hex, logits_hex, max_abs, argmatch, out_path);
    free(logits);
    return 0;
}

/* --v3-receipt-check <receipt> <weights> <logits.bin> <oracle.bin>: INDEPENDENT fail-closed verifier
 * (no GPU). Re-hash the committed weights + logits (Tier-2), re-argmax, re-check the Tier-3 envelope
 * vs the trusted oracle. Prints RECEIPT_CHECK_PASS / RECEIPT_CHECK_FAIL REJECT=<which>. */
static int run_v3_receipt_check(const char* receipt, const char* weights, const char* logits_path, const char* oracle_path) {
    FILE* rf = fopen(receipt, "rb");
    if (!rf) { printf("RECEIPT_CHECK_FAIL REJECT=OPEN\n"); return 1; }
    char line[1024]; int hdr=0, r_argmax=-1, r_nlog=0, r_argmatch=-1; char r_model[80]={0}, r_logits[80]={0}; double r_tau=-1;
    while (fgets(line, sizeof(line), rf)) {
        if (strncmp(line,"HELIX_V16_RECEIPT",17)==0) hdr=1;
        else if (sscanf(line,"model_sha256 %79s", r_model)==1) continue;
        else if (sscanf(line,"logits_sha256 %79s", r_logits)==1) continue;
        else if (sscanf(line,"n_logits %d", &r_nlog)==1) continue;
        else if (sscanf(line,"argmax %d", &r_argmax)==1) continue;
        else if (sscanf(line,"tier3_tau %lf", &r_tau)==1) continue;
        else if (sscanf(line,"tier3_argmax_match %d", &r_argmatch)==1) continue;
    }
    fclose(rf);
    if (!hdr || !r_model[0] || !r_logits[0] || r_argmax<0 || r_tau<0 || r_nlog<=0) { printf("RECEIPT_CHECK_FAIL REJECT=VACUITY\n"); return 1; }
    NV = r_nlog;   /* the receipt declares the logit count (the checker doesn't run peek) */
    char h[65];
    if (!sha256_file(weights, h))    { printf("RECEIPT_CHECK_FAIL REJECT=MODEL_HASH (weights unreadable)\n"); return 1; }
    if (strcmp(h, r_model) != 0)     { printf("RECEIPT_CHECK_FAIL REJECT=MODEL_HASH\n"); return 1; }
    if (!sha256_file(logits_path, h)){ printf("RECEIPT_CHECK_FAIL REJECT=LOGITS_HASH (logits unreadable)\n"); return 1; }
    if (strcmp(h, r_logits) != 0)    { printf("RECEIPT_CHECK_FAIL REJECT=LOGITS_HASH\n"); return 1; }
    FILE* lf = fopen(logits_path, "rb"); if (!lf) { printf("RECEIPT_CHECK_FAIL REJECT=LOGITS_OPEN\n"); return 1; }
    float* L = (float*)malloc((size_t)NV*sizeof(float)); size_t got = fread(L, sizeof(float), (size_t)NV, lf); fclose(lf);
    if (got != (size_t)NV) { printf("RECEIPT_CHECK_FAIL REJECT=LOGITS_SIZE\n"); free(L); return 1; }
    int am = argmax_row(L, NV);
    if (am != r_argmax) { printf("RECEIPT_CHECK_FAIL REJECT=ARGMAX (logits %d != receipt %d)\n", am, r_argmax); free(L); return 1; }
    FILE* orf = fopen(oracle_path, "rb"); if (!orf) { printf("RECEIPT_CHECK_FAIL REJECT=ORACLE_OPEN\n"); free(L); return 1; }
    float* O = (float*)malloc((size_t)NV*sizeof(float)); got = fread(O, sizeof(float), (size_t)NV, orf); fclose(orf);
    if (got != (size_t)NV) { printf("RECEIPT_CHECK_FAIL REJECT=ORACLE_SIZE\n"); free(L); free(O); return 1; }
    double m=0; for (int i=0;i<NV;i++){ double e=fabs((double)L[i]-(double)O[i]); if(e>m)m=e; }
    int om = argmax_row(O, NV);
    free(L); free(O);
    if (m > r_tau)   { printf("RECEIPT_CHECK_FAIL REJECT=TIER3_ENVELOPE (max_abs=%.6f > tau=%.6f)\n", m, r_tau); return 1; }
    if (am != om)    { printf("RECEIPT_CHECK_FAIL REJECT=TIER3_ARGMAX (logits %d != oracle %d)\n", am, om); return 1; }
    printf("receipt_check: Tier-2 commitments OK (model+logits+argmax); Tier-3 envelope OK (max_abs=%.6f <= tau=%.6f, argmax-match) -> RECEIPT_CHECK_PASS\n", m, r_tau);
    return 0;
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
    if (g_scfg.temp > 0.0) sample_reset(ids, T0);
    float* logits = (float*)malloc((size_t)NV*sizeof(float));
    int nonfinite_any = 0;
    double _gp0 = 0;   /* HX_GENPROF: time the decode-only steps (1..) with one sync at each end -> HX_FAST-honest s/tok */
    for (int step = 0; step < Ngen; step++) {
        if (g_genprof && step == 1) { cuCtxSynchronize(); _gp0 = now_seconds(); }
        if (ARCH && KV_ON && step > 0) decode_step_llama(ids[T-1], T-1, logits);
        else forward_full(ids, T, logits);
        for (int i = 0; i < NV; i++) if (!isfinite(logits[i])) { nonfinite_any = 1; break; }
        int nxt = (g_scfg.temp > 0.0) ? sample_next(logits, NV) : argmax_row(logits, NV);
        ids[T++] = nxt;
        if (EOS_ID >= 0 && nxt == EOS_ID) break;   /* chat eos-stop (matches the oracle: append eos, then stop) */
    }
    if (g_genprof && _gp0 > 0) { cuCtxSynchronize(); double _el = now_seconds() - _gp0; int _ds = T - T0 - 1; if (_ds < 1) _ds = 1;
        fprintf(stderr, "[genprof] decode %d steps in %.3fs = %.4f s/tok (g_fast=%d fused=%d)\n", _ds, _el, _el / _ds, g_fast, g_fusedgemv_on); }
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
    if (in_proc_tok) { build_byte_unicode(); load_vocab(vocab); load_merges(merges);
        if (g_specials_on) tok_enable_specials(); }
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
            ids = g_specials_on
                ? encode_bytes_special((const unsigned char*)req.prompt, req.prompt_len, &T0)
                : encode_bytes((const unsigned char*)req.prompt, req.prompt_len, &T0);
            free_ids = 1;
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
        if (g_scfg.temp > 0.0) sample_reset(ids, T0);

        /* working id buffer: prompt + room for Ngen generated ids. */
        /* per-request FAST mode: "detail":"token" drops per-op syncs + op/layer events for
         * THIS generation (same kernels, same ids -- instrumentation only). KV caches are
         * per-request: a fresh prefill fills them (kv_len reset). */
        {
            const char* dp = json_find_key(line, "detail");
            g_fast = (dp && *dp == '"' && strncmp(dp + 1, "token", 5) == 0) ? 1 : 0;
        }
        /* per-request sampling config (defaults = greedy when fields absent). SEEDED
         * sampling is gated token-for-token vs the oracle's identical sampler (G-S leg). */
        sample_cfg_from_json(line);
        { const char* lp = json_find_key(line, "lens"); g_lens_on = (lp && atoi(lp) != 0) ? 1 : 0; }
        stops_parse(line);
        kv_len = 0;
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
            if (ARCH && KV_ON && step > 0) decode_step_llama(work[T-1], T-1, logits);
            else forward_full(work, T, logits);             /* UNCHANGED arithmetic; hooks fire inside */
            for (int i = 0; i < NV; i++) if (!isfinite(logits[i])) { nonfinite = 1; break; }
            int nxt = (g_scfg.temp > 0.0) ? sample_next(logits, NV) : argmax_row(logits, NV);
#ifdef GPT2_SERVE
            char* piece = decode_one(nxt);
#else
            char* piece = NULL;
#endif
            tok_extra_render(logits, nxt);
            emit_token(step, nxt, piece ? piece : "", (double)logits[nxt], T + 1);
            int stop_now = stops_hit(piece);
            free(piece);
            gen_ids[step] = nxt;
            work[T++] = nxt;
            if (g_scfg.temp > 0.0) sample_count(nxt);
            if (stop_now) { Ngen = step + 1; break; }                      /* user stop-sequence hit */
            if (EOS_ID >= 0 && nxt == EOS_ID) { Ngen = step + 1; break; }  /* chat eos-stop; done reports the REAL count */
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
    if ((e=getenv("HX_EOS")))   EOS_ID=atoi(e);
    if ((e=getenv("HX_SPECIALS"))) g_specials_on=atoi(e);
    if ((e=getenv("HX_KV")))       KV_ON=atoi(e);
    if ((e=getenv("HX_RESIDENT"))) RESIDENT=atoi(e);
    if ((e=getenv("HX_FAST")))     g_fast=atoi(e);
    DH = DM / NH; QD = NH * DH;
    ATTN_SCALE = 1.0f / sqrtf((float)DH);

    /* v1.6 STEP-1 self-contained gate (no GPU, no ptx): dequant ONE HXGW v3 tensor to f32. */
    if (argc >= 5 && strcmp(argv[1], "--v3-dequant-dump") == 0)
        return run_v3_dequant_dump(argv[2], atoi(argv[3]), argv[4]);
    /* v1.6 P3b gate (GPU, no ptx): upload Qwen3 layer 0 via the v3 path + verify weights on-device. */
    if (argc >= 3 && strcmp(argv[1], "--v3-upload-check") == 0)
        return run_v3_upload_check(argv[2]);
    /* v1.6 receipt (no GPU, no ptx): SHA-256 KAT + the INDEPENDENT receipt verifier. */
    if (argc >= 2 && strcmp(argv[1], "--sha256-selftest") == 0)
        return sha256_selftest();
    if (argc >= 6 && strcmp(argv[1], "--v3-receipt-check") == 0)
        return run_v3_receipt_check(argv[2], argv[3], argv[4], argv[5]);
    /* v1.6 (no GPU, no ptx): re-serialize a receipt from ALREADY-DUMPED logits (GPU-free re-attest / re-emit with exact tau). */
    if (argc >= 7 && strcmp(argv[1], "--v3-receipt-from-logits") == 0)
        return run_v3_receipt_from_logits(argv[2], argv[3], argv[4], argv[5], argv[6]);

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
    /* pre-pass: --sample <json-fragment> may appear ANYWHERE; consume it before the
     * positional mode dispatch (it configures g_scfg; temp 0 = greedy default). */
    for (int i = 3; i < argc - 1; i++) {
        if (strcmp(argv[i], "--sample") == 0) {
            sample_cfg_from_json(argv[i + 1]);
            for (int j = i; j + 2 < argc; j++) argv[j] = argv[j + 2];
            argc -= 2;
            break;
        }
    }
    mode = argv[3];   /* re-read: the pre-pass may have shifted the mode into slot 3 */
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
    } else if (strcmp(mode, "--v3-smoke") == 0) {
        if (argc < 5) { fprintf(stderr, "--v3-smoke needs <ids.txt>\n"); return 2; }
        if (device_init(ptx_path, wpath)) return 2;
        int Smax = 128;
        if (alloc_buffers(Smax)) return 2;
        if (setup_head(Smax)) return 2;
        rc = run_v3_smoke(argv[4]);
    } else if (strcmp(mode, "--v3-receipt") == 0) {
        if (argc < 7) { fprintf(stderr, "--v3-receipt needs <ids.txt> <oracle_logits.bin> <out.receipt>\n"); return 2; }
        if (device_init(ptx_path, wpath)) return 2;
        int Smax = 128;
        if (alloc_buffers(Smax)) return 2;
        if (setup_head(Smax)) return 2;
        rc = run_v3_receipt(argv[4], argv[5], argv[6]);
    } else if (strcmp(mode, "--generate") == 0) {
        if (argc < 5) { fprintf(stderr, "--generate needs <N> [<ref_ids.txt>] [<ref_gen_ids.txt>]\n"); return 2; }
        int Ngen = atoi(argv[4]);
        const char* ids_path     = (argc > 5) ? argv[5] : NULL;
        const char* ref_gen_path = (argc > 6) ? argv[6] : NULL;
        if (device_init(ptx_path, wpath)) return 2;
        /* size buffers for the FINAL padded length: prompt(~5) + Ngen, rounded up to a multiple of 64. */
        /* templated chat prompts are ~40 ids, not ~5: pre-read the file for the REAL count. */
        int T0_real = 5;
        if (ids_path) { int _tmp[1024]; int _n = read_ids_file(ids_path, _tmp, 1024); if (_n > 0) T0_real = _n; }
        int Tmax = T0_real + Ngen + 4;
        int Smax = ((Tmax + 63) / 64) * 64;
        if (Smax < 64) Smax = 64;
        if (alloc_buffers(Smax)) return 2;
        if (setup_head(Smax)) return 2;
        rc = run_generate(Ngen, ids_path, ref_gen_path);
    } else if (strcmp(mode, "--lens-dump") == 0) {
        /* G-LENS: per-layer logit-lens top-1 for the LAST prompt position (prefill T-1
         * tokens, lens-decode the last). Requires the llama KV path (HX_KV=1). */
        if (argc < 6) { fprintf(stderr, "--lens-dump needs <ids.txt> <out.txt>\n"); return 2; }
        if (!ARCH || !KV_ON) { fprintf(stderr, "--lens-dump needs a llama model + HX_KV=1\n"); return 2; }
        int lids[1024]; int LT = read_ids_file(argv[4], lids, 1024);
        if (LT < 2) { fprintf(stderr, "need >= 2 ids\n"); return 2; }
        if (device_init(ptx_path, wpath)) return 2;
        int LS = ((LT + 4 + 63) / 64) * 64;
        if (alloc_buffers(LS)) return 2;
        if (setup_head(LS)) return 2;
        float* llg = (float*)malloc((size_t)NV * sizeof(float));
        forward_full(lids, LT - 1, llg);
        g_lens_on = 1;
        decode_step_llama(lids[LT - 1], LT - 1, llg);
        FILE* lf = fopen(argv[5], "w");
        if (!lf) return 2;
        for (int L = 0; L < g_lens_nl; L++) fprintf(lf, "%d\n", g_lens_top[L][0]);
        fclose(lf);
        printf("LENS_DUMP_OK %d layers\n", g_lens_nl);
        free(llg);
        rc = 0;
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
        else if (!strcmp(argv[i], "--sample")  && i+1 < argc) sample_cfg_from_json(argv[++i]);
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
