/* gpt2_pack.c -- safetensors -> .weights importer (HXGW format), Category-B HOST TOOL.
 *
 * TRUST CLAIM / FENCE ROLE: an OFFLINE host tool, OUTSIDE the self-host fixpoint, with
 * ZERO arithmetic on the compute-trust path -- exactly like helixc/runtime/cpu_host.c.
 * It performs only byte-movement (parse a JSON header, copy F32 tensor bytes in a fixed
 * order) for the demo's offline weight conversion. It exists to eliminate the Python
 * interpreter dependency from the demo's PRODUCTION data path; the independent numpy
 * oracle (helix-llm/tools/gpt2_numpy_ref.py) STAYS Python on purpose.
 *
 * It is a faithful re-authoring of helix-llm/tools/gpt2_import.py: same canonical
 * build_order (per-layer x NL, then globals wte/wpe/ln_f), weights stored UN-TRANSPOSED
 * (safetensors is row-major C-order F32 == the np.ascontiguousarray(...).ravel() the
 * Python importer writes -- so each tensor is a DIRECT byte copy, no transpose, no math),
 * and the same 64-byte header: <8I MAGIC,VERSION,NL,DM,NH,NV,NC,DF> + <Q N_FLOAT> + 24
 * zero bytes. The output is byte-identical (sha256) to gpt2_import.py's .weights, which
 * the committed consumer helixc/runtime/gpt2_infer.c (and the CPU twin cpu_host.c) mmap.
 *
 * Dims are read from config.json (same as the Python importer), so the format generalizes
 * to gpt2-large / gpt2-xl with no edits.
 *
 * STREAMING: tensors are copied straight from an mmap of the safetensors blob to the
 * output FILE in fixed-size chunks -- RAM stays bounded (we never materialize the ~500 MB
 * payload; the mmap is demand-paged and we copy through a small stack buffer).
 *
 * Build (host): gcc gpt2_pack.c -O2 -o gpt2_pack
 * Usage: gpt2_pack <model.safetensors> <config.json> <out.weights>
 *
 * License: Apache 2.0.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>

#define MAGIC   0x48584757u   /* 'HXGW' (little-endian bytes: 'W''G''X''H') */
#define VERSION 1u
#define HDR_BYTES 64

/* ============================ config.json (minimal hand-rolled) ============================ */
/* ARCH note (ADDITIVE --arch llama, 2026-06): the same byte-mover also packs Llama-arch
 * checkpoints (SmolLM2/TinyLlama): different tensor names/order, BF16 source tensors
 * (widened to f32 by BIT-SHIFT u32 = u16<<16 -- bit movement, not arithmetic), no biases,
 * tied lm_head (absent from the file; the consumer reuses the embedding). Header VERSION=2
 * with the arch fields in the formerly-zero bytes 40..55: u32 arch(1=llama), u32 n_kv_heads,
 * f32 rope_theta, f32 rms_eps. GPT-2 packs still emit VERSION=1 byte-identical to before. */
/* find an integer-valued field "key": <int> in a small JSON blob. Returns 1 + sets *out. */
static int json_find_int(const char* s, size_t n, const char* key, long* out) {
    size_t klen = strlen(key);
    for (size_t i = 0; i + klen + 2 < n; i++) {
        if (s[i] == '"' && memcmp(s + i + 1, key, klen) == 0 && s[i + 1 + klen] == '"') {
            size_t j = i + 1 + klen + 1;            /* after closing quote of the key */
            while (j < n && (s[j] == ' ' || s[j] == ':' || s[j] == '\t' || s[j] == '\n' || s[j] == '\r')) j++;
            int neg = 0; if (j < n && s[j] == '-') { neg = 1; j++; }
            if (j >= n || s[j] < '0' || s[j] > '9') continue;   /* not an int value (e.g. null) */
            long v = 0; int any = 0;
            while (j < n && s[j] >= '0' && s[j] <= '9') { v = v * 10 + (s[j] - '0'); j++; any = 1; }
            if (!any) continue;
            *out = neg ? -v : v;
            return 1;
        }
    }
    return 0;
}

/* ============================ safetensors header parse ============================ */
/* We need, per tensor name: dtype (must be F32), shape product, and [data_offset_start,
 * data_offset_end) into the blob (the bytes AFTER the 8-byte length + the JSON header). */
typedef struct { char name[128]; long nbytes; long off_start; long off_end; long shape[8]; int ndim; int bf16;
                 const unsigned char* blob; } STensor;   /* blob = this tensor's shard data base (sharded models) */

/* find a float-valued field "key": <num> (handles 1e-05 scientific notation). */
static int json_find_float(const char* s, size_t n, const char* key, double* out) {
    size_t klen = strlen(key);
    for (size_t i = 0; i + klen + 2 < n; i++) {
        if (s[i] == '"' && memcmp(s + i + 1, key, klen) == 0 && s[i + 1 + klen] == '"') {
            size_t j = i + 1 + klen + 1;
            while (j < n && (s[j] == ' ' || s[j] == ':' || s[j] == '\t' || s[j] == '\n' || s[j] == '\r')) j++;
            if (j >= n || !((s[j] >= '0' && s[j] <= '9') || s[j] == '-' || s[j] == '.')) continue;
            char buf[64]; int o = 0;
            while (j < n && o < 63 && ((s[j] >= '0' && s[j] <= '9') || s[j] == '-' || s[j] == '+' ||
                                       s[j] == '.' || s[j] == 'e' || s[j] == 'E')) buf[o++] = s[j++];
            buf[o] = 0;
            *out = atof(buf);
            return 1;
        }
    }
    return 0;
}

/* parse a JSON string at s[*i]=='"' into out (ASCII tensor names only); advance *i. */
static int parse_str(const char* s, size_t n, size_t* i, char* out, int cap) {
    int o = 0; (*i)++;
    while (*i < n && s[*i] != '"') {
        char c = s[*i];
        if (c == '\\') { (*i)++; if (*i < n) { if (o < cap - 1) out[o++] = s[*i]; (*i)++; } }
        else { if (o < cap - 1) out[o++] = c; (*i)++; }
    }
    if (*i < n) (*i)++;
    out[o] = 0;
    return o;
}
static void skip_ws(const char* s, size_t n, size_t* i) {
    while (*i < n && (s[*i] == ' ' || s[*i] == '\t' || s[*i] == '\n' || s[*i] == '\r')) (*i)++;
}

/* parse the whole safetensors JSON object; fill tensors[]; return count. */
static int parse_safetensors_header(const char* j, size_t jn, STensor* ts, int maxt) {
    size_t i = 0;
    skip_ws(j, jn, &i);
    if (i >= jn || j[i] != '{') { fprintf(stderr, "safetensors: header not an object\n"); exit(2); }
    i++;
    int nt = 0;
    for (;;) {
        skip_ws(j, jn, &i);
        if (i >= jn || j[i] == '}') break;
        if (j[i] == ',') { i++; continue; }
        if (j[i] != '"') { i++; continue; }
        char name[128];
        parse_str(j, jn, &i, name, sizeof(name));
        skip_ws(j, jn, &i);
        if (i < jn && j[i] == ':') i++;
        skip_ws(j, jn, &i);
        if (i >= jn || j[i] != '{') { fprintf(stderr, "safetensors: value for %s not object\n", name); exit(2); }
        /* parse the tensor object: dtype, shape, data_offsets */
        int depth = 0;
        char dtype[40] = {0};            /* > the parse_str v[] cap so the copy is provably bounded */
        long shape[8]; int ndim = 0;
        long off0 = -1, off1 = -1;
        /* manual scan of this object */
        for (;;) {
            if (i >= jn) break;
            if (j[i] == '{') { depth++; i++; continue; }
            if (j[i] == '}') { depth--; i++; if (depth == 0) break; else continue; }
            if (j[i] == '"') {
                char field[32];
                parse_str(j, jn, &i, field, sizeof(field));
                skip_ws(j, jn, &i);
                if (i < jn && j[i] == ':') i++;
                skip_ws(j, jn, &i);
                if (strcmp(field, "dtype") == 0) {
                    char v[16]; parse_str(j, jn, &i, v, sizeof(v));
                    memcpy(dtype, v, sizeof(v));          /* v is NUL-terminated within 16; dtype[40] holds it */
                } else if (strcmp(field, "shape") == 0) {
                    if (i < jn && j[i] == '[') {
                        i++;
                        ndim = 0;
                        for (;;) {
                            skip_ws(j, jn, &i);
                            if (i < jn && j[i] == ']') { i++; break; }
                            if (i < jn && j[i] == ',') { i++; continue; }
                            long v = 0; int any = 0;
                            while (i < jn && j[i] >= '0' && j[i] <= '9') { v = v * 10 + (j[i] - '0'); i++; any = 1; }
                            if (any && ndim < 8) shape[ndim++] = v;
                            else if (!any) i++;
                        }
                    }
                } else if (strcmp(field, "data_offsets") == 0) {
                    if (i < jn && j[i] == '[') {
                        i++;
                        long vals[2]; int k = 0;
                        for (;;) {
                            skip_ws(j, jn, &i);
                            if (i < jn && j[i] == ']') { i++; break; }
                            if (i < jn && j[i] == ',') { i++; continue; }
                            long v = 0; int any = 0;
                            while (i < jn && j[i] >= '0' && j[i] <= '9') { v = v * 10 + (j[i] - '0'); i++; any = 1; }
                            if (any && k < 2) vals[k++] = v;
                            else if (!any) i++;
                        }
                        if (k == 2) { off0 = vals[0]; off1 = vals[1]; }
                    }
                } else {
                    /* unknown field: skip its value (string/number/array/object) */
                    if (i < jn && j[i] == '"') { char tmp[256]; parse_str(j, jn, &i, tmp, sizeof(tmp)); }
                    else if (i < jn && j[i] == '[') { int d2 = 0; do { if (j[i]=='[') d2++; else if (j[i]==']') d2--; i++; } while (i < jn && d2 > 0); }
                    else if (i < jn && j[i] == '{') { int d2 = 0; do { if (j[i]=='{') d2++; else if (j[i]=='}') d2--; i++; } while (i < jn && d2 > 0); }
                    else { while (i < jn && j[i] != ',' && j[i] != '}') i++; }
                }
            } else {
                i++;
            }
        }
        if (strcmp(name, "__metadata__") == 0) continue;   /* skip metadata pseudo-tensor */
        if (nt >= maxt) { fprintf(stderr, "too many tensors\n"); exit(2); }
        STensor* t = &ts[nt++];
        strncpy(t->name, name, sizeof(t->name) - 1); t->name[sizeof(t->name)-1] = 0;
        t->ndim = ndim; for (int d = 0; d < ndim; d++) t->shape[d] = shape[d];
        t->off_start = off0; t->off_end = off1; t->nbytes = off1 - off0;
        if (strcmp(dtype, "F32") == 0) t->bf16 = 0;
        else if (strcmp(dtype, "BF16") == 0) t->bf16 = 1;
        else { fprintf(stderr, "tensor %s dtype %s not F32/BF16\n", name, dtype); exit(2); }
        if (off0 < 0 || off1 < 0) { fprintf(stderr, "tensor %s missing data_offsets\n", name); exit(2); }
    }
    return nt;
}

/* find a tensor by name; NULL if absent. */
static STensor* find_tensor(STensor* ts, int nt, const char* name) {
    for (int i = 0; i < nt; i++) if (strcmp(ts[i].name, name) == 0) return &ts[i];
    return NULL;
}

/* ============================ build order (mirrors gpt2_import.py) ============================ */
/* We append (name, expected_shape...) entries into a list; the writer copies them in order. */
typedef struct { char name[128]; long shape[4]; int ndim; } OrderEntry;

static int build_order(OrderEntry* ord, int cap, int NL, int DM, int NV, int NC, int DF) {
    int n = 0;
#define ADD1(NM, S0)        do { snprintf(ord[n].name, 128, "%s", NM); ord[n].ndim=1; ord[n].shape[0]=(S0); n++; } while(0)
#define ADD2(NM, S0, S1)    do { snprintf(ord[n].name, 128, "%s", NM); ord[n].ndim=2; ord[n].shape[0]=(S0); ord[n].shape[1]=(S1); n++; } while(0)
    char p[64];
    for (int L = 0; L < NL; L++) {
        snprintf(p, sizeof(p), "h.%d.", L);
        char nm[128];
        snprintf(nm, sizeof(nm), "%sln_1.weight", p);        ADD1(nm, DM);
        snprintf(nm, sizeof(nm), "%sln_1.bias", p);          ADD1(nm, DM);
        snprintf(nm, sizeof(nm), "%sattn.c_attn.weight", p); ADD2(nm, DM, 3*DM);
        snprintf(nm, sizeof(nm), "%sattn.c_attn.bias", p);   ADD1(nm, 3*DM);
        snprintf(nm, sizeof(nm), "%sattn.c_proj.weight", p); ADD2(nm, DM, DM);
        snprintf(nm, sizeof(nm), "%sattn.c_proj.bias", p);   ADD1(nm, DM);
        snprintf(nm, sizeof(nm), "%sln_2.weight", p);        ADD1(nm, DM);
        snprintf(nm, sizeof(nm), "%sln_2.bias", p);          ADD1(nm, DM);
        snprintf(nm, sizeof(nm), "%smlp.c_fc.weight", p);    ADD2(nm, DM, DF);
        snprintf(nm, sizeof(nm), "%smlp.c_fc.bias", p);      ADD1(nm, DF);
        snprintf(nm, sizeof(nm), "%smlp.c_proj.weight", p);  ADD2(nm, DF, DM);
        snprintf(nm, sizeof(nm), "%smlp.c_proj.bias", p);    ADD1(nm, DM);
        if (n > cap - 16) { fprintf(stderr, "order overflow\n"); exit(2); }
    }
    ADD2("wte.weight", NV, DM);
    ADD2("wpe.weight", NC, DM);
    ADD1("ln_f.weight", DM);
    ADD1("ln_f.bias", DM);
#undef ADD1
#undef ADD2
    return n;
}

/* Llama-arch build order (HF LlamaForCausalLM names). Per layer: input_layernorm,
 * q/k/v/o_proj (HF Linear [out,in], stored UN-TRANSPOSED; the consumer's GEMMs are A.Bt),
 * post_attention_layernorm, gate/up/down_proj. Globals: embed_tokens, final norm.
 * lm_head is TIED (absent); KVD = n_kv_heads * head_dim.
 *
 * qwen3=1 (v1.6, additive): adds per-layer self_attn.q_norm.weight + k_norm.weight [head_dim]
 * (Qwen3 applies a per-head RMSNorm on q/k -- weight shape = head_dim) right after q_proj/k_proj,
 * and an UNTIED lm_head.weight [NV x DM] after the final norm. qwen3=0 is byte-identical to the
 * SmolLM2/TinyLlama order (head_dim is then unused). */
/* QD = num_q_heads * head_dim = the attention/query-projection dim. For SmolLM2 + Qwen3-8B QD==DM
 * (so q/o are [DM,DM]); for Qwen3-32B QD=8192 != DM=5120 (q_proj [QD,DM], o_proj [DM,QD]). */
static int build_order_llama(OrderEntry* ord, int cap, int NL, int DM, int NV, int DF, int KVD,
                             int QD, int head_dim, int qwen3) {
    int n = 0;
#define ADD1(NM, S0)        do { snprintf(ord[n].name, 128, "%s", NM); ord[n].ndim=1; ord[n].shape[0]=(S0); n++; } while(0)
#define ADD2(NM, S0, S1)    do { snprintf(ord[n].name, 128, "%s", NM); ord[n].ndim=2; ord[n].shape[0]=(S0); ord[n].shape[1]=(S1); n++; } while(0)
    char nm[128];
    for (int L = 0; L < NL; L++) {
        snprintf(nm, sizeof(nm), "model.layers.%d.input_layernorm.weight", L);          ADD1(nm, DM);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.q_proj.weight", L);         ADD2(nm, QD, DM);
        if (qwen3) { snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.q_norm.weight", L); ADD1(nm, head_dim); }
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.k_proj.weight", L);         ADD2(nm, KVD, DM);
        if (qwen3) { snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.k_norm.weight", L); ADD1(nm, head_dim); }
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.v_proj.weight", L);         ADD2(nm, KVD, DM);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.o_proj.weight", L);         ADD2(nm, DM, QD);
        snprintf(nm, sizeof(nm), "model.layers.%d.post_attention_layernorm.weight", L); ADD1(nm, DM);
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate_proj.weight", L);            ADD2(nm, DF, DM);
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.up_proj.weight", L);              ADD2(nm, DF, DM);
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.down_proj.weight", L);            ADD2(nm, DM, DF);
        if (n > cap - 16) { fprintf(stderr, "order overflow\n"); exit(2); }
    }
    ADD2("model.embed_tokens.weight", NV, DM);
    ADD1("model.norm.weight", DM);
    if (qwen3) ADD2("lm_head.weight", NV, DM);   /* untied head (tie_word_embeddings=false) */
#undef ADD1
#undef ADD2
    return n;
}

static long shape_prod(const long* s, int nd) { long p = 1; for (int i = 0; i < nd; i++) p *= s[i]; return p; }

/* ============================ v1.6 NVFP4 forward quantizer (host) ============================
 * The packed format MUST match helixc/examples/nvfp4_dequant_kernel.hx byte-for-byte. Per weight
 * [rows x K] with K padded to a multiple of 112 (LCM(7,16)): packed words w[rows*(Kp/7)] hold 7 E2M1
 * codes/i32 word base-16 LOW-NIBBLE-FIRST; effective f32 scales sc[rows*(Kp/16)] hold one per 16-block,
 * value = e4m3_decode(micro) * fp32_tensor_scale (host pre-collapsed; device does mag*sc only).
 * Codecs copied VERBATIM from cuda_launch.c (the v1.5 oracle) -- never re-derive. */
static float e2m1_decode(int code) {
    int s=(code>>3)&1, e=(code>>1)&3, m=code&1; float mag;
    if (e==0) mag = 0.5f*(float)m; else mag = (1.0f+0.5f*(float)m)*(float)(1<<(e-1));
    return s ? -mag : mag;
}
/* REFERENCE: 16-way nearest search (kept for the equivalence self-test). */
static int e2m1_encode_ref(float v) {
    int best=0; float bd=1.0e30f;
    for (int c=0;c<16;c++){ float d=e2m1_decode(c)-v; if(d<0)d=-d; if(d<bd){bd=d;best=c;} }
    return best;
}
/* FAST e2m1 encode (dev-opt, ~10x): direct nearest-magnitude threshold map over the E2M1 magnitudes
 * {0,.5,1,1.5,2,3,4,6}; midpoints {.25,.75,1.25,1.75,2.5,3.5,5} use <= so a tie picks the LOWER
 * magnitude == the 16-way's first-min tie-break; magnitude 0 -> code 0 (sign-less, the 16-way finds
 * code 0 before code 8). PROVEN bit-identical to e2m1_encode_ref by e2m1_equiv_selftest. */
static int e2m1_encode(float v) {
    float a = v < 0.0f ? -v : v;
    int mag;
    if      (a <= 0.25f) return 0;
    else if (a <= 0.75f) mag = 1;
    else if (a <= 1.25f) mag = 2;
    else if (a <= 1.75f) mag = 3;
    else if (a <= 2.5f)  mag = 4;
    else if (a <= 3.5f)  mag = 5;
    else if (a <= 5.0f)  mag = 6;
    else                 mag = 7;
    return mag | (v < 0.0f ? 8 : 0);
}
/* the fast encode MUST be bit-identical to the 16-way ref over a fine grid + the exact f32 midpoints. */
static int e2m1_equiv_selftest(void) {
    int bad = 0; long n = 0;
    for (double x = -8.0; x <= 8.0; x += 0.0005) {
        float v = (float)x;
        if (e2m1_encode(v) != e2m1_encode_ref(v)) { if (bad<8) fprintf(stderr,"e2m1 mismatch v=%.6g fast=%d ref=%d\n", v, e2m1_encode(v), e2m1_encode_ref(v)); bad++; }
        n++;
    }
    float B[] = {0.0f,0.25f,0.5f,0.75f,1.0f,1.25f,1.5f,1.75f,2.0f,2.5f,3.0f,3.5f,4.0f,5.0f,6.0f,7.0f};
    for (int i=0;i<16;i++) for (int s=-1;s<=1;s+=2) { float v=(float)s*B[i];
        if (e2m1_encode(v)!=e2m1_encode_ref(v)){ if(bad<8) fprintf(stderr,"e2m1 boundary mismatch v=%.6g fast=%d ref=%d\n", v, e2m1_encode(v), e2m1_encode_ref(v)); bad++; } n++; }
    printf("e2m1_equiv_selftest: %ld cases, %d mismatches -> %s\n", n, bad, bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
static float e4m3_decode(int c) {
    int s=(c>>7)&1, e=(c>>3)&15, m=c&7;
    if (e==15 && m==7) return nanf("");
    float mag; if (e==0) mag=(m/8.0f)*ldexpf(1.0f,-6); else mag=(1.0f+m/8.0f)*ldexpf(1.0f,e-7);
    return s ? -mag : mag;
}
static int e4m3_encode(float v) {
    int best=0; float bd=1.0e30f;
    for (int c=0;c<256;c++){ if((c&0x7F)==0x7F) continue; float d=e4m3_decode(c)-v; if(d<0)d=-d; if(d<bd){bd=d;best=c;} }
    return best;
}
/* Quantize [rows x K] f32 -> NVFP4; K padded to Kpad (mult of 112) with ZEROS. Caller sizes
 * outW[rows*(Kpad/7)] (i32). outSc[rows*(Kpad/16)] (f32 effective scale) is OPTIONAL -- pass NULL
 * to skip it. For COMPACT storage (v1.6 HXGW v3): pass outMicro[rows*(Kpad/16)] (u8, the per-block
 * E4M3 micro code) + outTs (f32, the one per-tensor scale); at upload-time the worker rebuilds the
 * effective scale eff = e4m3_decode(micro) * (*outTs). Either/both of outSc/outMicro may be NULL.
 * Returns Kpad. */
static int nvfp4_quantize_tensor(const float* w, int rows, int K, int32_t* outW, float* outSc,
                                 uint8_t* outMicro, float* outTs) {
    int Kpad = ((K + 111)/112)*112;
    int kwords = Kpad/7, kblk = Kpad/16;
    float amax = 0.0f;
    for (long i=0;i<(long)rows*K;i++){ float a = w[i]<0?-w[i]:w[i]; if (a>amax) amax=a; }
    float ts = amax/(6.0f*448.0f); if (ts <= 0.0f) ts = 1.0f;   /* E2M1_max(6)*E4M3_max(448); all-zero guard */
    if (outTs) *outTs = ts;
    int* codes = (int*)malloc((size_t)Kpad*sizeof(int));
    for (int r=0;r<rows;r++){
        for (int bk=0;bk<kblk;bk++){
            float bamax=0.0f;
            for (int j=0;j<16;j++){ int k=bk*16+j; float v=(k<K)?w[(long)r*K+k]:0.0f; float a=v<0?-v:v; if(a>bamax)bamax=a; }
            int micro = e4m3_encode(bamax/(ts*6.0f));
            float eff = e4m3_decode(micro)*ts;
            if (outSc)    outSc[(long)r*kblk+bk] = eff;
            if (outMicro) outMicro[(long)r*kblk+bk] = (uint8_t)micro;
            for (int j=0;j<16;j++){ int k=bk*16+j; float v=(k<K)?w[(long)r*K+k]:0.0f; codes[k]=(eff>0.0f)?e2m1_encode(v/eff):0; }
        }
        for (int kw=0;kw<kwords;kw++){ int word=0,pw=1; for(int j=0;j<7;j++){ word += (codes[kw*7+j]&15)*pw; pw*=16; } outW[(long)r*kwords+kw]=word; }
    }
    free(codes);
    return Kpad;
}
/* Host dequant -- a LINE-FOR-LINE mirror of nvfp4_dequant_kernel.hx (the device formula). */
static void nvfp4_host_dequant(const int32_t* w, const float* sc, float* out, int rows, int Kpad) {
    int kwords=Kpad/7, kblk=Kpad/16;
    for (int r=0;r<rows;r++) for (int col=0;col<Kpad;col++){
        int word=col/7, slot=col-word*7;
        int wv = w[(long)r*kwords+word];
        for (int j=0;j<slot;j++) wv = wv/16;
        int code = wv - (wv/16)*16;
        int c8 = code - (code/8)*8;
        float magf = (c8==0)?0.0f:(c8==1)?0.5f:(c8==2)?1.0f:(c8==3)?1.5f:(c8==4)?2.0f:(c8==5)?3.0f:(c8==6)?4.0f:6.0f;
        float sm = (code/8==0)?magf:-magf;
        out[(long)r*Kpad+col] = sm * sc[(long)r*kblk+col/16];
    }
}
/* CPU-only self-test: quantize a synthetic tensor (with a non-112-multiple K to exercise padding),
 * dequant via the device-mirror, and require the pack/unpack to round-trip the codes + scale index
 * EXACTLY (the FORMAT is the part that must be byte-exact; the quant error is informational). */
static int nvfp4_selftest(void) {
    int rows=5, K=100;                          /* 100 -> Kpad 112 (exercises zero-padding) */
    int Kpad=((K+111)/112)*112, kwords=Kpad/7, kblk=Kpad/16;
    float*   w     = (float*)malloc((size_t)rows*K*sizeof(float));
    int32_t* W     = (int32_t*)malloc((size_t)rows*kwords*sizeof(int32_t));
    float*   Sc    = (float*)malloc((size_t)rows*kblk*sizeof(float));
    float*   recon = (float*)malloc((size_t)rows*Kpad*sizeof(float));
    for (int r=0;r<rows;r++) for (int k=0;k<K;k++) w[(long)r*K+k] = ((float)((r*37+k*13)%101)-50.0f)*0.043f;
    int kp = nvfp4_quantize_tensor(w, rows, K, W, Sc, NULL, NULL);
    nvfp4_host_dequant(W, Sc, recon, rows, kp);
    int fmtbad=0, padbad=0; float qmax=0.0f;
    for (int r=0;r<rows;r++) for (int k=0;k<kp;k++){
        float eff  = Sc[(long)r*kblk + k/16];
        float orig = (k<K)? w[(long)r*K+k] : 0.0f;
        int   code = (eff>0.0f)? e2m1_encode(orig/eff) : 0;     /* the SAME choice the quantizer made */
        float expect = e2m1_decode(code) * eff;
        float g = recon[(long)r*kp+k];
        if (g != expect) { if (fmtbad<4) fprintf(stderr,"nvfp4 FORMAT mismatch r%d k%d got %g expect %g\n",r,k,g,expect); fmtbad++; }
        if (k>=K && g != 0.0f) padbad++;
        if (k<K){ float e=g-orig; if(e<0)e=-e; if(e>qmax)qmax=e; }
    }
    printf("nvfp4_selftest: rows=%d K=%d Kpad=%d | FORMAT exact-mismatches=%d | pad-nonzero=%d | quant max_abs=%.6g -> %s\n",
           rows,K,kp,fmtbad,padbad,qmax,(fmtbad==0&&padbad==0)?"PASS":"FAIL");
    free(w);free(W);free(Sc);free(recon);
    return (fmtbad==0&&padbad==0)?0:1;
}
/* CPU test on a REAL safetensors tensor: load [out,in] (bf16->f32), quantize, dequant, report the
 * quant error. rmse/rms_w (relative RMS error) is the Tier-3 envelope basis; a high value would mean
 * the amax-per-tensor scale is outlier-dominated. Usage: --nvfp4-testreal <shard.safetensors> <name>. */
static int nvfp4_testreal(const char* shard, const char* tname, const char* dumpbin) {
    int fd = open(shard, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open %s: %s\n", shard, strerror(errno)); return 2; }
    struct stat st; if (fstat(fd, &st) != 0) { fprintf(stderr, "fstat\n"); return 2; }
    size_t flen = (size_t)st.st_size;
    unsigned char* base = (unsigned char*)mmap(NULL, flen, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { fprintf(stderr, "mmap: %s\n", strerror(errno)); return 2; }
    uint64_t hlen; memcpy(&hlen, base, 8);
    const char* jhdr = (const char*)(base + 8);
    const unsigned char* blob = base + 8 + hlen;
    STensor* ts = (STensor*)malloc(sizeof(STensor) * 8192);
    int nt = parse_safetensors_header(jhdr, (size_t)hlen, ts, 8192);
    STensor* t = find_tensor(ts, nt, tname);
    if (!t) { fprintf(stderr, "tensor %s not in this shard (%d tensors)\n", tname, nt); return 2; }
    if (t->ndim != 2) { fprintf(stderr, "need a 2D weight; %s is %dD\n", tname, t->ndim); return 2; }
    int rows = (int)t->shape[0], K = (int)t->shape[1];
    long n = (long)rows * K;
    float* w = (float*)malloc((size_t)n * sizeof(float));
    const unsigned char* src = blob + t->off_start;
    if (t->bf16) { for (long i=0;i<n;i++){ uint16_t u; memcpy(&u, src+i*2, 2); uint32_t x=((uint32_t)u)<<16; memcpy(&w[i], &x, 4); } }
    else          { memcpy(w, src, (size_t)n * 4); }
    int Kpad = ((K+111)/112)*112, kwords = Kpad/7, kblk = Kpad/16;
    int32_t* W   = (int32_t*)malloc((size_t)rows*kwords*sizeof(int32_t));
    float*   Sc  = (float*)malloc((size_t)rows*kblk*sizeof(float));
    float*   rec = (float*)malloc((size_t)rows*Kpad*sizeof(float));
    nvfp4_quantize_tensor(w, rows, K, W, Sc, NULL, NULL);
    nvfp4_host_dequant(W, Sc, rec, rows, Kpad);
    double sumsq_e=0, sumsq_w=0, sum_abs_e=0; float maxabs=0, maxrel=0, wamax=0; long cnt=0;
    for (int r=0;r<rows;r++) for (int k=0;k<K;k++){
        float o = w[(long)r*K+k], g = rec[(long)r*Kpad+k];
        float e = g-o; if (e<0) e=-e;
        float ao = o<0?-o:o; if (ao>wamax) wamax=ao;
        if (e>maxabs) maxabs=e;
        float rel = ao>1e-9f ? e/ao : 0.0f; if (rel>maxrel) maxrel=rel;
        sumsq_e += (double)e*e; sumsq_w += (double)o*o; sum_abs_e += e; cnt++;
    }
    double rmse = sqrt(sumsq_e/(double)cnt), rms_w = sqrt(sumsq_w/(double)cnt);
    double packed = (double)rows*kwords*4 + (double)rows*kblk*4 + 4, f32b = (double)n*4;
    printf("nvfp4_testreal %s [%dx%d] Kpad=%d | w_amax=%.5g rms_w=%.5g | quant max_abs=%.5g mean_abs=%.5g rmse=%.5g (rmse/rms_w=%.4f) max_rel=%.3g | nvfp4=%.1fMB vs f32=%.1fMB (%.2fx)\n",
           tname, rows, K, Kpad, wamax, rms_w, maxabs, sum_abs_e/(double)cnt, rmse, rmse/rms_w, maxrel, packed/1e6, f32b/1e6, f32b/packed);
    if (dumpbin) {   /* STEP-1 cross-tool gate: dump rec [rows x Kpad] for the worker to byte-match */
        FILE* df = fopen(dumpbin, "wb");
        if (df) { fwrite(rec, 4, (size_t)rows*Kpad, df); fclose(df);
                  fprintf(stderr, "[testreal] dumped rec [%dx%d] -> %s\n", rows, Kpad, dumpbin); }
    }
    free(w); free(W); free(Sc); free(rec); free(ts); munmap(base, flen); close(fd);
    return 0;
}
/* sharded safetensors: find which shard file holds a tensor, from model.safetensors.index.json
 * (weight_map = {"tensor.name":"model-0000N-of-...safetensors", ...}). Returns 1 + the shard
 * filename in `out`, or 0. Plain substring search of the JSON text (the index is small). */
static int index_shard(const char* idx_path, const char* tname, char* out, int outsz) {
    int fd = open(idx_path, O_RDONLY); if (fd < 0) return 0;
    struct stat st; if (fstat(fd, &st) != 0) { close(fd); return 0; }
    char* j = (char*)malloc((size_t)st.st_size + 1);
    ssize_t got = read(fd, j, (size_t)st.st_size); close(fd);
    if (got < 0) { free(j); return 0; } j[got] = 0;
    char key[300]; snprintf(key, sizeof(key), "\"%s\"", tname);
    char* p = strstr(j, key); if (!p) { free(j); return 0; }
    p += strlen(key);
    while (*p && *p != ':') p++;
    if (*p != ':') { free(j); return 0; }
    p++;
    while (*p && *p != '"') p++;
    if (*p != '"') { free(j); return 0; }
    p++;
    char* q = strchr(p, '"');
    if (!q) { free(j); return 0; }
    int n = (int)(q - p); if (n >= outsz) n = outsz - 1;
    memcpy(out, p, (size_t)n); out[n] = 0;
    free(j); return 1;
}
/* like --nvfp4-testreal but on a (possibly sharded) MODEL DIR: look the tensor up in the index ->
 * the right shard -> quantize-test it. Usage: --nvfp4-testmodel <model_dir> <tensor.name>. */
static int nvfp4_testmodel(const char* dir, const char* tname, const char* dumpbin) {
    char idx[1100]; snprintf(idx, sizeof(idx), "%s/model.safetensors.index.json", dir);
    char shard[300];
    if (!index_shard(idx, tname, shard, sizeof(shard))) { fprintf(stderr, "tensor %s not in index %s\n", tname, idx); return 2; }
    char path[1500]; snprintf(path, sizeof(path), "%s/%s", dir, shard);
    fprintf(stderr, "[testmodel] %s -> shard %s\n", tname, shard);
    return nvfp4_testreal(path, tname, dumpbin);
}

/* ===================== v1.6 sharded-model loader + HXGW v3 NVFP4 pack ===================== */
/* collect the UNIQUE shard filenames from an index.json weight_map (values "model-...safetensors"). */
static int index_shards(const char* idx_json, char names[][300], int maxn) {
    int n = 0;
    const char* p = idx_json;
    const char* needle = ".safetensors\"";
    size_t nl = strlen(needle);
    while ((p = strstr(p, needle)) != NULL) {
        const char* end = p + strlen(".safetensors");   /* points AT the closing quote */
        const char* q = p;
        while (q > idx_json && *(q-1) != '"') q--;       /* walk back to first char after opening quote */
        int len = (int)(end - q);
        if (len > 0 && len < 300) {
            char nm[300]; memcpy(nm, q, (size_t)len); nm[len] = 0;
            int seen = 0; for (int k=0;k<n;k++) if (strcmp(names[k], nm)==0) { seen=1; break; }
            if (!seen && n < maxn) { memcpy(names[n], nm, (size_t)len+1); n++; }
        }
        p += nl;
    }
    return n;
}
typedef struct { int fd; size_t flen; unsigned char* base; } ShardMap;
/* load a (sharded OR single) safetensors model into ONE combined STensor list; each STensor.blob
 * points at its shard's tensor-data region. Fills sh[] with the per-shard mmaps (caller munmaps). */
static int load_model_tensors(const char* dir, STensor* ts, int maxt, ShardMap* sh, int* n_sh) {
    char idx[1100]; snprintf(idx, sizeof(idx), "%s/model.safetensors.index.json", dir);
    char shards[64][300]; int nshard = 0;   /* up to 64 shards (Qwen3-32B has 17; 8B has 5) */
    int ifd = open(idx, O_RDONLY);
    if (ifd >= 0) {
        struct stat ist; fstat(ifd, &ist);
        char* ij = (char*)malloc((size_t)ist.st_size + 1);
        ssize_t g = read(ifd, ij, (size_t)ist.st_size); close(ifd);
        if (g < 0) g = 0;
        ij[g] = 0;
        nshard = index_shards(ij, shards, 64);
        free(ij);
        if (nshard == 0) { fprintf(stderr, "[load] index.json has no shard filenames\n"); exit(2); }
    } else {
        snprintf(shards[0], 300, "model.safetensors"); nshard = 1;   /* single-file fallback (v1.4-style) */
    }
    int nt = 0; *n_sh = 0;
    for (int s = 0; s < nshard; s++) {
        char path[1500]; snprintf(path, sizeof(path), "%s/%s", dir, shards[s]);
        int fd = open(path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "[load] open shard %s: %s\n", path, strerror(errno)); exit(2); }
        struct stat st; if (fstat(fd, &st) != 0) { fprintf(stderr, "[load] fstat %s\n", path); exit(2); }
        size_t flen = (size_t)st.st_size;
        unsigned char* base = (unsigned char*)mmap(NULL, flen, PROT_READ, MAP_PRIVATE, fd, 0);
        if (base == MAP_FAILED) { fprintf(stderr, "[load] mmap %s: %s\n", path, strerror(errno)); exit(2); }
        uint64_t hlen; memcpy(&hlen, base, 8);
        const char* jhdr = (const char*)(base + 8);
        const unsigned char* blob = base + 8 + hlen;
        int got = parse_safetensors_header(jhdr, (size_t)hlen, ts + nt, maxt - nt);
        for (int k = 0; k < got; k++) ts[nt + k].blob = blob;
        nt += got;
        sh[*n_sh].fd = fd; sh[*n_sh].flen = flen; sh[*n_sh].base = base; (*n_sh)++;
        fprintf(stderr, "[load] %s: %d tensors\n", shards[s], got);
    }
    fprintf(stderr, "[load] total %d tensors across %d shard(s)\n", nt, *n_sh);
    return nt;
}
/* a tensor is NVFP4-packed iff it is a 2D matmul weight (q/k/v/o/gate/up/down/lm_head); the 1D
 * norms and the embedding (a gather, not a matmul) stay dense f32. */
static int is_packed_name(const char* nm, int ndim) {
    if (ndim != 2) return 0;
    if (strstr(nm, "embed_tokens")) return 0;
    return (strstr(nm,"q_proj")||strstr(nm,"k_proj")||strstr(nm,"v_proj")||strstr(nm,"o_proj")||
            strstr(nm,"gate_proj")||strstr(nm,"up_proj")||strstr(nm,"down_proj")||strstr(nm,"lm_head")) ? 1 : 0;
}
static void materialize_f32(const STensor* t, float* w) {
    long n = shape_prod(t->shape, t->ndim);
    const unsigned char* src = t->blob + t->off_start;
    if (t->bf16) { for (long i=0;i<n;i++){ uint16_t u; memcpy(&u, src+i*2, 2); uint32_t x=((uint32_t)u)<<16; memcpy(&w[i], &x, 4); } }
    else         { memcpy(w, src, (size_t)n*4); }
}
/* HXGW v3 per-tensor descriptor (32 bytes). packed=1: data_off->packed i32 words [rows*(Kpad/7)],
 * scale_off->compact scale = E4M3 micro bytes [rows*(Kpad/16)] then ONE f32 per-tensor scale.
 * packed=0: data_off->dense f32 [rows*K] (rows=1,K=numel,Kpad=numel), scale_off=0. */
typedef struct { uint32_t packed, rows, K, Kpad; uint64_t data_off, scale_off; } HXGWv3Desc;
_Static_assert(sizeof(HXGWv3Desc) == 32, "HXGWv3Desc must be 32 bytes");

/* --pack-qwen3 <model_dir> <out.weights>: pack a (sharded) Qwen3 checkpoint to HXGW v3 NVFP4,
 * then VERIFY by re-reading the file + host-dequant each packed tensor vs the original (~9.5% RMS).
 * ISOLATED from the v1.4 single-file pack path (main) -> zero risk to the SmolLM2 demo. */
static int pack_qwen3(const char* model_dir, const char* out_path) {
    long NL=0,DM=0,NH=0,NKV=0,DF=0,NV=0,NC=0,HD=0; double theta=0, eps=0;
    {
        char cfgp[1100]; snprintf(cfgp, sizeof(cfgp), "%s/config.json", model_dir);
        int fd = open(cfgp, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "[pack-qwen3] open %s: %s\n", cfgp, strerror(errno)); return 2; }
        struct stat st; fstat(fd, &st);
        char* cfg = (char*)malloc((size_t)st.st_size+1);
        ssize_t got = read(fd, cfg, (size_t)st.st_size); close(fd); if (got<0) got=0; cfg[got]=0;
        size_t cn = (size_t)got;
        if (!json_find_int(cfg,cn,"num_hidden_layers",&NL) || !json_find_int(cfg,cn,"hidden_size",&DM) ||
            !json_find_int(cfg,cn,"num_attention_heads",&NH) || !json_find_int(cfg,cn,"num_key_value_heads",&NKV) ||
            !json_find_int(cfg,cn,"intermediate_size",&DF) || !json_find_int(cfg,cn,"vocab_size",&NV)) {
            fprintf(stderr, "[pack-qwen3] config missing a required int field\n"); free(cfg); return 2;
        }
        if (!json_find_int(cfg,cn,"head_dim",&HD)) HD = DM/NH;
        if (!json_find_int(cfg,cn,"max_position_embeddings",&NC)) NC = 40960;
        if (!json_find_float(cfg,cn,"rope_theta",&theta)) theta = 1000000.0;
        if (!json_find_float(cfg,cn,"rms_norm_eps",&eps)) eps = 1e-6;
        free(cfg);
    }
    long KVD = NKV * HD;          /* kv-projection dim  */
    long QD  = NH * HD;           /* query/attention dim (== DM for 8B; 8192 != DM=5120 for 32B) */
    fprintf(stderr, "[pack-qwen3] NL=%ld DM=%ld NH=%ld NKV=%ld HD=%ld QD=%ld DF=%ld NV=%ld KVD=%ld theta=%g eps=%g\n",
            NL,DM,NH,NKV,HD,QD,DF,NV,KVD,theta,eps);

    STensor* ts = (STensor*)malloc(sizeof(STensor)*4096);
    ShardMap sh[64]; int n_sh = 0;
    int nt = load_model_tensors(model_dir, ts, 4096, sh, &n_sh);

    int cap = (int)(NL*12 + 24);
    OrderEntry* ord = (OrderEntry*)malloc(sizeof(OrderEntry)*cap);
    int no = build_order_llama(ord, cap, (int)NL,(int)DM,(int)NV,(int)DF,(int)KVD,(int)QD,(int)HD, 1);

    /* descriptor sizes/offsets (no data touched yet) */
    HXGWv3Desc* desc = (HXGWv3Desc*)calloc((size_t)no, sizeof(HXGWv3Desc));
    uint64_t cur = (uint64_t)HDR_BYTES + (uint64_t)no*sizeof(HXGWv3Desc);
    int n_packed = 0;
    for (int e=0;e<no;e++) {
        STensor* t = find_tensor(ts, nt, ord[e].name);
        if (!t) { fprintf(stderr, "[pack-qwen3] missing tensor %s\n", ord[e].name); return 2; }
        long want = shape_prod(ord[e].shape, ord[e].ndim), have = shape_prod(t->shape, t->ndim);
        if (want != have) { fprintf(stderr, "[pack-qwen3] %s shape prod %ld != expected %ld\n", ord[e].name, have, want); return 2; }
        int packed = is_packed_name(ord[e].name, ord[e].ndim);
        desc[e].packed = (uint32_t)packed;
        if (packed) {
            int rows=(int)t->shape[0], K=(int)t->shape[1];
            int Kpad=((K+111)/112)*112, kwords=Kpad/7, kblk=Kpad/16;
            desc[e].rows=(uint32_t)rows; desc[e].K=(uint32_t)K; desc[e].Kpad=(uint32_t)Kpad;
            desc[e].data_off = cur;
            uint64_t wbytes=(uint64_t)rows*kwords*4;
            desc[e].scale_off = cur + wbytes;
            cur += wbytes + (uint64_t)rows*kblk + 4;       /* words + micro bytes + 1 f32 ts */
            n_packed++;
        } else {
            desc[e].rows=1; desc[e].K=(uint32_t)have; desc[e].Kpad=(uint32_t)have;
            desc[e].data_off = cur; desc[e].scale_off = 0;
            cur += (uint64_t)have*4;
        }
    }
    uint64_t total_bytes = cur;
    fprintf(stderr, "[pack-qwen3] %d tensors (%d packed), out = %llu B (%.2f GB)\n",
            no, n_packed, (unsigned long long)total_bytes, (double)total_bytes/1e9);

    FILE* of = fopen(out_path, "wb");
    if (!of) { fprintf(stderr, "[pack-qwen3] open out %s: %s\n", out_path, strerror(errno)); return 2; }
    {
        unsigned char hdr[HDR_BYTES]; memset(hdr,0,sizeof(hdr));
        uint32_t h32[8] = { MAGIC, 3u, (uint32_t)NL,(uint32_t)DM,(uint32_t)NH,(uint32_t)NV,(uint32_t)NC,(uint32_t)DF };
        for (int k=0;k<8;k++) memcpy(hdr+k*4, &h32[k], 4);
        uint32_t arch=1u, nkv=(uint32_t)NKV; float th=(float)theta, ep=(float)eps;
        memcpy(hdr+40,&arch,4); memcpy(hdr+44,&nkv,4); memcpy(hdr+48,&th,4); memcpy(hdr+52,&ep,4);
        uint32_t hd=(uint32_t)HD, ntens=(uint32_t)no;
        memcpy(hdr+56,&hd,4); memcpy(hdr+60,&ntens,4);    /* @56 head_dim, @60 n_tensors (v3) */
        if (fwrite(hdr,1,HDR_BYTES,of)!=HDR_BYTES){fprintf(stderr,"[pack-qwen3] write hdr\n");return 2;}
        if (fwrite(desc,sizeof(HXGWv3Desc),(size_t)no,of)!=(size_t)no){fprintf(stderr,"[pack-qwen3] write desc\n");return 2;}
    }
    /* stream the data in order */
    for (int e=0;e<no;e++) {
        STensor* t = find_tensor(ts, nt, ord[e].name);
        long have = shape_prod(t->shape, t->ndim);
        long pos_now = ftell(of);
        if ((uint64_t)pos_now != desc[e].data_off) { fprintf(stderr,"[pack-qwen3] FATAL offset drift %s: at %ld want %llu\n", ord[e].name, pos_now, (unsigned long long)desc[e].data_off); return 2; }
        float* w = (float*)malloc((size_t)have*4); materialize_f32(t, w);
        if (desc[e].packed) {
            int rows=(int)desc[e].rows, K=(int)desc[e].K, Kpad=(int)desc[e].Kpad, kwords=Kpad/7, kblk=Kpad/16;
            int32_t* W=(int32_t*)malloc((size_t)rows*kwords*4);
            uint8_t* micro=(uint8_t*)malloc((size_t)rows*kblk);
            float tsv=1.0f;
            nvfp4_quantize_tensor(w, rows, K, W, NULL, micro, &tsv);
            int ok = (fwrite(W,4,(size_t)rows*kwords,of)==(size_t)rows*kwords)
                   & (fwrite(micro,1,(size_t)rows*kblk,of)==(size_t)rows*kblk)
                   & (fwrite(&tsv,4,1,of)==1);
            free(W); free(micro);
            if (!ok) { fprintf(stderr,"[pack-qwen3] write packed %s\n", ord[e].name); free(w); return 2; }
        } else {
            if (fwrite(w,4,(size_t)have,of)!=(size_t)have){fprintf(stderr,"[pack-qwen3] write f32 %s\n",ord[e].name);free(w);return 2;}
        }
        free(w);
    }
    long endpos = ftell(of);
    fclose(of);
    if ((uint64_t)endpos != total_bytes) { fprintf(stderr,"[pack-qwen3] FATAL size %ld != %llu\n", endpos,(unsigned long long)total_bytes); return 2; }
    fprintf(stderr, "[pack-qwen3] wrote %s (%ld B). Re-reading + verifying every tensor...\n", out_path, endpos);

    /* VERIFY: re-read the FILE, dequant each packed tensor from the written bytes, compare to original */
    int vfd = open(out_path, O_RDONLY);
    struct stat vst; fstat(vfd, &vst);
    unsigned char* vbase = (unsigned char*)mmap(NULL, (size_t)vst.st_size, PROT_READ, MAP_PRIVATE, vfd, 0);
    if (vbase == MAP_FAILED) { fprintf(stderr,"[pack-qwen3] verify mmap\n"); return 2; }
    HXGWv3Desc* vdesc = (HXGWv3Desc*)(vbase + HDR_BYTES);
    double worst_rms = 0.0; const char* worst_name = "(none)";
    int f32_exact = 1, verified = 0;
    for (int e=0;e<no;e++) {
        STensor* t = find_tensor(ts, nt, ord[e].name);
        long have = shape_prod(t->shape, t->ndim);
        float* w = (float*)malloc((size_t)have*4); materialize_f32(t, w);
        if (vdesc[e].packed) {
            int rows=(int)vdesc[e].rows, K=(int)vdesc[e].K, Kpad=(int)vdesc[e].Kpad, kblk=Kpad/16;
            const int32_t* W = (const int32_t*)(vbase + vdesc[e].data_off);
            const uint8_t* micro = (const uint8_t*)(vbase + vdesc[e].scale_off);
            float tsv; memcpy(&tsv, vbase + vdesc[e].scale_off + (size_t)rows*kblk, 4);
            float* eff = (float*)malloc((size_t)rows*kblk*4);
            for (long i=0;i<(long)rows*kblk;i++) eff[i] = e4m3_decode(micro[i])*tsv;   /* rebuild effective scale */
            float* deq = (float*)malloc((size_t)rows*Kpad*4);
            nvfp4_host_dequant(W, eff, deq, rows, Kpad);
            double se=0, sw=0;
            for (int r=0;r<rows;r++) for (int k=0;k<K;k++){ float o=w[(long)r*K+k], g=deq[(long)r*Kpad+k]; double d=(double)g-o; se+=d*d; sw+=(double)o*o; }
            double rmse=sqrt(se/((double)rows*K)), rms_w=sqrt(sw/((double)rows*K));
            double rel = rms_w>0 ? rmse/rms_w : 0;
            if (rel > worst_rms) { worst_rms = rel; worst_name = ord[e].name; }
            free(eff); free(deq);
        } else {
            const float* fv = (const float*)(vbase + vdesc[e].data_off);
            for (long i=0;i<have;i++) if (fv[i] != w[i]) { if (f32_exact) fprintf(stderr,"[pack-qwen3] f32 MISMATCH in %s at elt %ld\n", ord[e].name, i); f32_exact = 0; break; }
        }
        free(w); verified++;
    }
    munmap(vbase, (size_t)vst.st_size); close(vfd);
    for (int s=0;s<n_sh;s++){ munmap(sh[s].base, sh[s].flen); close(sh[s].fd); }
    free(ts); free(ord); free(desc);
    int pass = (worst_rms < 0.15) && f32_exact;     /* packed ~9.5% expected; f32 must be bit-exact */
    fprintf(stderr, "[pack-qwen3] VERIFY: %d tensors | worst packed rmse/rms_w=%.4f (%s) | f32 exact=%s -> %s\n",
            verified, worst_rms, worst_name, f32_exact?"yes":"NO", pass?"PASS":"FAIL");
    printf("%s worst_rms=%.4f f32_exact=%d tensors=%d\n", pass ? "PACK_QWEN3_OK" : "PACK_QWEN3_FAIL", worst_rms, f32_exact, verified);
    return pass ? 0 : 1;
}

int main(int argc, char** argv) {
    if (argc >= 2 && strcmp(argv[1], "--e2m1-equiv") == 0) return e2m1_equiv_selftest();
    if (argc >= 2 && strcmp(argv[1], "--nvfp4-selftest") == 0) return nvfp4_selftest();
    if (argc >= 4 && strcmp(argv[1], "--nvfp4-testreal") == 0) return nvfp4_testreal(argv[2], argv[3], argc>=5?argv[4]:NULL);
    if (argc >= 4 && strcmp(argv[1], "--nvfp4-testmodel") == 0) return nvfp4_testmodel(argv[2], argv[3], argc>=5?argv[4]:NULL);
    if (argc >= 4 && strcmp(argv[1], "--pack-qwen3") == 0) return pack_qwen3(argv[2], argv[3]);
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.safetensors> <config.json> <out.weights> [--arch llama]\n", argv[0]);
        return 2;
    }
    const char* safet_path = argv[1];
    const char* cfg_path   = argv[2];
    const char* out_path   = argv[3];
    int arch_llama = 0;
    for (int a = 4; a < argc; a++)
        if (strcmp(argv[a], "--arch") == 0 && a + 1 < argc && strcmp(argv[a+1], "llama") == 0) arch_llama = 1;

    /* ---- config dims (keys differ per arch; dims ALWAYS from config, never a table) ---- */
    long NL=0, DM=0, NH=0, NV=0, NC=0, DF=0, n_inner=0, NKV=0;
    double rope_theta = 0.0, rms_eps = 0.0;
    {
        int fd = open(cfg_path, O_RDONLY);
        if (fd < 0) { fprintf(stderr, "open config '%s': %s\n", cfg_path, strerror(errno)); return 2; }
        struct stat st; fstat(fd, &st);
        char* cfg = (char*)malloc((size_t)st.st_size + 1);
        ssize_t got = read(fd, cfg, (size_t)st.st_size); close(fd);
        if (got < 0) { fprintf(stderr, "read config\n"); return 2; }
        cfg[got] = 0;
        if (arch_llama) {
            if (!json_find_int(cfg, (size_t)got, "num_hidden_layers", &NL))   { fprintf(stderr, "config: no num_hidden_layers\n"); return 2; }
            if (!json_find_int(cfg, (size_t)got, "hidden_size", &DM))         { fprintf(stderr, "config: no hidden_size\n"); return 2; }
            if (!json_find_int(cfg, (size_t)got, "num_attention_heads", &NH)) { fprintf(stderr, "config: no num_attention_heads\n"); return 2; }
            if (!json_find_int(cfg, (size_t)got, "num_key_value_heads", &NKV)){ fprintf(stderr, "config: no num_key_value_heads\n"); return 2; }
            if (!json_find_int(cfg, (size_t)got, "intermediate_size", &DF))   { fprintf(stderr, "config: no intermediate_size\n"); return 2; }
            if (!json_find_int(cfg, (size_t)got, "vocab_size", &NV))          { fprintf(stderr, "config: no vocab_size\n"); return 2; }
            if (!json_find_int(cfg, (size_t)got, "max_position_embeddings", &NC)) NC = 2048;
            if (!json_find_float(cfg, (size_t)got, "rope_theta", &rope_theta)){ fprintf(stderr, "config: no rope_theta\n"); return 2; }
            if (!json_find_float(cfg, (size_t)got, "rms_norm_eps", &rms_eps)) { fprintf(stderr, "config: no rms_norm_eps\n"); return 2; }
            if (NH <= 0 || NKV <= 0 || (NH % NKV) != 0) { fprintf(stderr, "config: bad GQA heads %ld/%ld\n", NH, NKV); return 2; }
        } else {
            if (!json_find_int(cfg, (size_t)got, "n_layer", &NL)) { fprintf(stderr, "config: no n_layer\n"); return 2; }
            if (!json_find_int(cfg, (size_t)got, "n_embd",  &DM)) { fprintf(stderr, "config: no n_embd\n"); return 2; }
            if (!json_find_int(cfg, (size_t)got, "n_head",  &NH)) { fprintf(stderr, "config: no n_head\n"); return 2; }
            if (!json_find_int(cfg, (size_t)got, "vocab_size", &NV)) NV = 50257;
            if (!json_find_int(cfg, (size_t)got, "n_ctx", &NC)) { if (!json_find_int(cfg,(size_t)got,"n_positions",&NC)) NC = 1024; }
            /* d_ff = n_inner if present (and >0), else 4*n_embd */
            if (json_find_int(cfg, (size_t)got, "n_inner", &n_inner) && n_inner > 0) DF = n_inner;
            else DF = 4 * DM;
        }
        free(cfg);
    }
    long KVD = arch_llama ? (NKV * (DM / NH)) : 0;     /* n_kv_heads * head_dim */
    long per_layer, N_FLOAT;
    if (arch_llama) {
        /* in_ln[DM] q[DM*DM] k[KVD*DM] v[KVD*DM] o[DM*DM] post_ln[DM] gate[DF*DM] up[DF*DM] down[DM*DF] */
        per_layer = DM + (long)DM*DM + (long)KVD*DM + (long)KVD*DM + (long)DM*DM
                  + DM + (long)DF*DM + (long)DF*DM + (long)DM*DF;
        N_FLOAT = NL * per_layer + (long)NV*DM + DM;   /* + embed + final norm (lm_head TIED) */
    } else {
        per_layer = (DM + DM + (long)DM*3*DM + 3*DM + (long)DM*DM + DM
                      + DM + DM + (long)DM*DF + DF + (long)DF*DM + DM);
        N_FLOAT = NL * per_layer + (long)NV*DM + (long)NC*DM + DM + DM;
    }
    fprintf(stderr, "[pack] arch=%s dims NL=%ld DM=%ld NH=%ld NKV=%ld NV=%ld NC=%ld DF=%ld -> N_FLOAT=%ld (%ld B file)\n",
            arch_llama ? "llama" : "gpt2", NL, DM, NH, NKV, NV, NC, DF, N_FLOAT, (long)HDR_BYTES + N_FLOAT * 4);

    /* ---- mmap safetensors ---- */
    int fd = open(safet_path, O_RDONLY);
    if (fd < 0) { fprintf(stderr, "open safetensors '%s': %s\n", safet_path, strerror(errno)); return 2; }
    struct stat st; if (fstat(fd, &st) != 0) { fprintf(stderr, "fstat\n"); return 2; }
    size_t flen = (size_t)st.st_size;
    unsigned char* base = (unsigned char*)mmap(NULL, flen, PROT_READ, MAP_PRIVATE, fd, 0);
    if (base == MAP_FAILED) { fprintf(stderr, "mmap: %s\n", strerror(errno)); return 2; }
    uint64_t hlen; memcpy(&hlen, base, 8);
    const char* jhdr = (const char*)(base + 8);
    const unsigned char* blob = base + 8 + hlen;     /* tensor data region */

    STensor* ts = (STensor*)malloc(sizeof(STensor) * 4096);
    int nt = parse_safetensors_header(jhdr, (size_t)hlen, ts, 4096);
    fprintf(stderr, "[pack] safetensors: %d tensors, header %llu B\n", nt, (unsigned long long)hlen);

    /* ---- build order ---- */
    OrderEntry* ord = (OrderEntry*)malloc(sizeof(OrderEntry) * (NL * 12 + 8 + 16));
    int no = arch_llama
        ? build_order_llama(ord, (int)(NL * 12 + 24), (int)NL, (int)DM, (int)NV, (int)DF, (int)KVD, (int)DM, (int)(DM/NH), 0)
        : build_order(ord, (int)(NL * 12 + 24), (int)NL, (int)DM, (int)NV, (int)NC, (int)DF);

    /* ---- open output, write 64-byte header (v2 carries the llama arch fields) ---- */
    FILE* of = fopen(out_path, "wb");
    if (!of) { fprintf(stderr, "open out '%s': %s\n", out_path, strerror(errno)); return 2; }
    {
        uint32_t ver = arch_llama ? 2u : VERSION;
        uint32_t h32[8] = { MAGIC, ver, (uint32_t)NL, (uint32_t)DM, (uint32_t)NH, (uint32_t)NV, (uint32_t)NC, (uint32_t)DF };
        unsigned char hdr[HDR_BYTES];
        memset(hdr, 0, sizeof(hdr));
        for (int k = 0; k < 8; k++) { memcpy(hdr + k*4, &h32[k], 4); }   /* little-endian host assumed (x86-64) */
        uint64_t nf = (uint64_t)N_FLOAT;
        memcpy(hdr + 32, &nf, 8);
        if (arch_llama) {                                                /* v2 extension: bytes 40..55 */
            uint32_t arch = 1u, nkv = (uint32_t)NKV;
            float th = (float)rope_theta, ep = (float)rms_eps;
            memcpy(hdr + 40, &arch, 4);
            memcpy(hdr + 44, &nkv, 4);
            memcpy(hdr + 48, &th, 4);
            memcpy(hdr + 52, &ep, 4);
        }                                                                /* else bytes 40..63 stay zero (v1) */
        if (fwrite(hdr, 1, HDR_BYTES, of) != HDR_BYTES) { fprintf(stderr, "write hdr\n"); return 2; }
    }

    /* ---- stream tensors in build order (byte copy; BF16 widened by BIT-SHIFT, no transpose, no math) ---- */
    long total_floats = 0;
    long written_bytes = 0;
    unsigned char buf[1 << 20];
    uint32_t wide[1 << 18];                       /* 256K floats per chunk for the bf16 widen path */
    for (int e = 0; e < no; e++) {
        STensor* t = find_tensor(ts, nt, ord[e].name);
        if (!t) { fprintf(stderr, "missing tensor %s\n", ord[e].name); return 2; }
        /* shape check against the expected build-order shape (catches a wrong model) */
        long want = shape_prod(ord[e].shape, ord[e].ndim);
        long have = shape_prod(t->shape, t->ndim);
        if (want != have) { fprintf(stderr, "tensor %s shape product %ld != expected %ld\n", ord[e].name, have, want); return 2; }
        long nbytes = t->nbytes;
        long esz = t->bf16 ? 2 : 4;
        if (nbytes != have * esz) { fprintf(stderr, "tensor %s nbytes %ld != floats*%ld %ld\n", ord[e].name, nbytes, esz, have * esz); return 2; }
        const unsigned char* src = blob + t->off_start;
        if (!t->bf16) {
            /* F32: direct chunked byte copy (unchanged v1 path) */
            long pos = 0;
            while (pos < nbytes) {
                long chunk = nbytes - pos; if (chunk > (long)sizeof(buf)) chunk = (long)sizeof(buf);
                memcpy(buf, src + pos, (size_t)chunk);                   /* mmap demand-pages; small buffer */
                if (fwrite(buf, 1, (size_t)chunk, of) != (size_t)chunk) { fprintf(stderr, "write body %s\n", ord[e].name); return 2; }
                pos += chunk;
            }
            written_bytes += nbytes;
        } else {
            /* BF16 -> F32: u32 = (u32)u16 << 16 (bf16 IS the top half of f32 -- a bit move, no rounding,
             * no arithmetic; exactly what numpy's (u16.astype(u32)<<16).view(f32) does in the oracle). */
            long done = 0;
            while (done < have) {
                long n = have - done; if (n > (long)(sizeof(wide)/4)) n = (long)(sizeof(wide)/4);
                const unsigned char* sp = src + done * 2;
                for (long i = 0; i < n; i++) {
                    uint16_t u; memcpy(&u, sp + i*2, 2);
                    wide[i] = ((uint32_t)u) << 16;
                }
                if (fwrite(wide, 4, (size_t)n, of) != (size_t)n) { fprintf(stderr, "write body %s\n", ord[e].name); return 2; }
                done += n;
            }
            written_bytes += have * 4;
        }
        total_floats += have;
    }
    fclose(of);

    /* ---- accounting: confirm float total + the leftover-skip set ---- */
    if (total_floats != N_FLOAT) { fprintf(stderr, "FATAL: wrote %ld floats != N_FLOAT %ld\n", total_floats, N_FLOAT); return 2; }
    /* gpt2: leftovers must be exactly the NL causal-mask buffers h.<L>.attn.bias.
     * llama (tied embeddings): there must be ZERO leftovers -- every tensor consumed. */
    int consumed_ok = 1, leftover = 0, masks = 0;
    for (int i = 0; i < nt; i++) {
        int used = 0;
        for (int e = 0; e < no; e++) if (strcmp(ord[e].name, ts[i].name) == 0) { used = 1; break; }
        if (!used) {
            leftover++;
            const char* nm = ts[i].name;
            int is_mask = 0;
            if (!arch_llama && strncmp(nm, "h.", 2) == 0) { const char* dot = strstr(nm, ".attn.bias"); if (dot && dot[10] == 0) is_mask = 1; }
            if (is_mask) masks++; else { consumed_ok = 0; fprintf(stderr, "UNEXPECTED leftover tensor: %s\n", nm); }
        }
    }
    if (arch_llama ? (leftover != 0) : (!consumed_ok || masks != NL)) {
        fprintf(stderr, "FATAL: leftover accounting off (arch=%s leftover=%d masks=%d NL=%ld)\n",
                arch_llama ? "llama" : "gpt2", leftover, masks, NL);
        return 2;
    }

    long expect_file = (long)HDR_BYTES + N_FLOAT * 4;
    fprintf(stderr, "[pack] wrote %s : header 64 + %ld floats = %ld B (skipped %d attn.bias masks)\n",
            out_path, total_floats, expect_file, masks);
    /* final size check */
    {
        struct stat so; if (stat(out_path, &so) == 0) {
            if ((long)so.st_size != expect_file) { fprintf(stderr, "FATAL: out size %ld != %ld\n", (long)so.st_size, expect_file); return 2; }
        }
    }
    munmap(base, flen); close(fd);
    free(ts); free(ord);
    printf("GPT2_PACK_OK floats=%ld bytes=%ld\n", total_floats, expect_file);
    return 0;
}
