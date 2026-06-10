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
typedef struct { char name[128]; long nbytes; long off_start; long off_end; long shape[8]; int ndim; int bf16; } STensor;

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
 * lm_head is TIED (absent); KVD = n_kv_heads * head_dim. */
static int build_order_llama(OrderEntry* ord, int cap, int NL, int DM, int NV, int DF, int KVD) {
    int n = 0;
#define ADD1(NM, S0)        do { snprintf(ord[n].name, 128, "%s", NM); ord[n].ndim=1; ord[n].shape[0]=(S0); n++; } while(0)
#define ADD2(NM, S0, S1)    do { snprintf(ord[n].name, 128, "%s", NM); ord[n].ndim=2; ord[n].shape[0]=(S0); ord[n].shape[1]=(S1); n++; } while(0)
    char nm[128];
    for (int L = 0; L < NL; L++) {
        snprintf(nm, sizeof(nm), "model.layers.%d.input_layernorm.weight", L);          ADD1(nm, DM);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.q_proj.weight", L);         ADD2(nm, DM, DM);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.k_proj.weight", L);         ADD2(nm, KVD, DM);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.v_proj.weight", L);         ADD2(nm, KVD, DM);
        snprintf(nm, sizeof(nm), "model.layers.%d.self_attn.o_proj.weight", L);         ADD2(nm, DM, DM);
        snprintf(nm, sizeof(nm), "model.layers.%d.post_attention_layernorm.weight", L); ADD1(nm, DM);
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.gate_proj.weight", L);            ADD2(nm, DF, DM);
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.up_proj.weight", L);              ADD2(nm, DF, DM);
        snprintf(nm, sizeof(nm), "model.layers.%d.mlp.down_proj.weight", L);            ADD2(nm, DM, DF);
        if (n > cap - 16) { fprintf(stderr, "order overflow\n"); exit(2); }
    }
    ADD2("model.embed_tokens.weight", NV, DM);
    ADD1("model.norm.weight", DM);
#undef ADD1
#undef ADD2
    return n;
}

static long shape_prod(const long* s, int nd) { long p = 1; for (int i = 0; i < nd; i++) p *= s[i]; return p; }

int main(int argc, char** argv) {
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
        ? build_order_llama(ord, (int)(NL * 12 + 24), (int)NL, (int)DM, (int)NV, (int)DF, (int)KVD)
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
