/* cpu_host.c -- GPT-2 124M CPU FORWARD harness (the NO-ptxas / NO-GPU twin of
 * gpt2_infer.c, which is itself the CUDA-free fork of train_transformer.c).
 *
 * TRUST CLAIM: ALL ARITHMETIC stays in the kovc-compiled Helix op ELF
 * (gpt2_cpu_ops.bin, rebuildable from the raw seed). This C harness does ONLY
 * byte-movement: mmap the P1 weight file, host embedding gather, multi-head
 * pack/scatter, GEMM N-tiling, and per-op file staging (write the input tile,
 * exec the Helix op ELF, read the output tile). NO float math on the trust path
 * beyond the embedding-gather add (the same residual the capstone does for input
 * injection) and the final parity comparison.
 *
 * Staging protocol (driver_k1input.hx pattern): per op, write /tmp/gpc/in.bin =
 *   [op:i32, d0, d1, d2, d3, d4]  (6x LE i32 header)  ++  input f32 tile(s) (LE)
 * exec OP_ELF (its main() reads /tmp/gpc/in.bin, computes, writes /tmp/gpc/out.bin),
 * read /tmp/gpc/out.bin = output f32 tile (LE). Tiles are kept < the Helix 1 MB
 * read buffer (256K floats) and the working set under the 6,291,456-slot arena;
 * big GEMMs (c_attn/c_fc/mlp_proj) are tiled over the N (output-col) dimension.
 *
 * Build (WSL ext4): gcc cpu_host.c -O2 -lm -o cpu_host
 * Run:  ./cpu_host <gpt2_124M.weights> <gpt2_cpu_ops.bin> --block0 <ref_block0.npy>
 *   -> dumps /tmp/gpc/helix_block0.bin (flat <f4 [T,768]) + prints GPT2_CPU_BLOCK0_PARITY_PASS/FAIL.
 *
 * License: Apache 2.0.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <unistd.h>

/* ---- GPT-2 124M dims (env-overridable, defaulting to the public model) ---- */
static int NL = 12, DM = 768, NH = 12, NV = 50257, NC = 1024, DFF = 3072;
static int DH = 64;                 /* head dim = DM/NH */
/* N-tiling block for the big GEMMs (cols per Helix invocation). Chosen so the
 * staged input (A[T*K] + W[K*Nblk] + bias[Nblk]) stays well under the 256K-float
 * Helix read buffer: K<=3072, Nblk=64 -> ~197K floats worst case. */
static int NTILE = 64;

/* ---- P1 weight-file header (must match helix-llm/tools/gpt2_import.py) ---- */
#define MAGIC   0x48584757u         /* 'HXGW' little-endian */
#define VERSION 1u
#define HDR_BYTES 64

static const char* OP_ELF = NULL;   /* path to gpt2_cpu_ops.bin */
static const char* GPC_DIR = "/tmp/gpc";
static char IN_PATH[256], OUT_PATH[256];

/* op codes (match gpt2_cpu_ops.hx) */
#define OP_LAYERNORM 1
#define OP_MATMUL    2
#define OP_MATMUL_B  3
#define OP_GELU      4
#define OP_ADD       5
#define OP_SOFTMAX   6

/* ===================== weight file (mmap + per-tensor offsets) ===================== */
static const float* g_wbase = NULL; /* mmap'd payload base (float*, at file offset 64) */
static size_t g_nfloat = 0;
static int    g_fd = -1;
static void*  g_map = NULL;
static size_t g_maplen = 0;

static long per_layer_floats(void) {
    return (long)DM + DM + (long)DM*3*DM + 3*DM + (long)DM*DM + DM
         + DM + DM + (long)DM*DFF + DFF + (long)DFF*DM + DM;
}
static long off_layer(int L) { return (long)L * per_layer_floats(); }
static long off_globals(void) { return (long)NL * per_layer_floats(); }

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
    g_wbase = (const float*)(hb + HDR_BYTES);
    printf("[wt] mmap %zu B, n_float=%zu (per_layer=%ld globals_off=%ld)\n",
           g_maplen, g_nfloat, per_layer_floats(), off_globals());
    return 0;
}

/* ===================== the Helix-op staging primitive ===================== */
/* write an i32 LE */
static void w_i32(FILE* f, int v) { unsigned char b[4]; b[0]=v&0xff; b[1]=(v>>8)&0xff; b[2]=(v>>16)&0xff; b[3]=(v>>24)&0xff; fwrite(b,1,4,f); }

/* run the Helix op ELF on /tmp/gpc/in.bin -> /tmp/gpc/out.bin. fork+exec (no shell). */
static int exec_op(void) {
    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return -1; }
    if (pid == 0) {
        /* child: silence stdout (the op may print nothing, but be safe) */
        execl(OP_ELF, OP_ELF, (char*)NULL);
        perror("execl op"); _exit(127);
    }
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) { perror("waitpid"); return -1; }
    /* kovc returns the output byte-count as exit status (nonzero on success), and
     * a SIGILL/segfault would show via WIFSIGNALED -- treat only a signal as fatal. */
    if (WIFSIGNALED(status)) { fprintf(stderr, "op ELF killed by signal %d\n", WTERMSIG(status)); return -1; }
    return 0;
}

/* read out.bin into dst[nf] floats; assert the file has exactly nf floats. */
static int read_out(float* dst, size_t nf) {
    FILE* f = fopen(OUT_PATH, "rb");
    if (!f) { fprintf(stderr, "open out '%s': %s\n", OUT_PATH, strerror(errno)); return -1; }
    size_t got = fread(dst, sizeof(float), nf, f);
    long extra = 0; { fseek(f, 0, SEEK_END); long sz = ftell(f); extra = sz - (long)(nf*4); }
    fclose(f);
    if (got != nf) { fprintf(stderr, "out short read %zu/%zu\n", got, nf); return -1; }
    if (extra != 0) { fprintf(stderr, "out size mismatch: %ld extra bytes (wanted %zu floats)\n", extra, nf); return -1; }
    return 0;
}

/* ---- op wrappers (each stages the input, execs the op, reads the output) ---- */

/* LayerNorm affine: y[rows*cols] = LN(x) * gamma + beta. */
static int helix_layernorm(const float* x, const float* gamma, const float* beta,
                           int rows, int cols, float* y) {
    FILE* f = fopen(IN_PATH, "wb"); if (!f) { perror("open in"); return -1; }
    w_i32(f, OP_LAYERNORM); w_i32(f, rows); w_i32(f, cols); w_i32(f, 0); w_i32(f, 0); w_i32(f, 0);
    fwrite(x, sizeof(float), (size_t)rows*cols, f);
    fwrite(gamma, sizeof(float), (size_t)cols, f);
    fwrite(beta, sizeof(float), (size_t)cols, f);
    fclose(f);
    if (exec_op()) return -1;
    return read_out(y, (size_t)rows*cols);
}

/* one GEMM tile: C[M*Nblk] = A[M*K] @ Bcol[K*Nblk] (+ bias[Nblk] if has_bias).
 * Bcol must be the contiguous [K, Nblk] column-block (caller packs it). */
static int helix_matmul_tile(const float* A, int M, int K,
                             const float* Bcol, int Nblk,
                             const float* bias /*or NULL*/, float* Cblk) {
    FILE* f = fopen(IN_PATH, "wb"); if (!f) { perror("open in"); return -1; }
    int op = bias ? OP_MATMUL_B : OP_MATMUL;
    w_i32(f, op); w_i32(f, M); w_i32(f, K); w_i32(f, Nblk); w_i32(f, 0); w_i32(f, 0);
    fwrite(A, sizeof(float), (size_t)M*K, f);
    fwrite(Bcol, sizeof(float), (size_t)K*Nblk, f);
    if (bias) fwrite(bias, sizeof(float), (size_t)Nblk, f);
    fclose(f);
    if (exec_op()) return -1;
    return read_out(Cblk, (size_t)M*Nblk);
}

/* full GEMM C[M,N] = A[M,K] @ W[K,N] (+bias[N]), tiled over N. W is row-major
 * [K,N] (the un-transposed HF Conv1D layout the oracle's a@W consumes). The
 * harness packs each [K,Nblk] column-block contiguously (byte-movement only). */
static int helix_matmul(const float* A, int M, int K, const float* W, int N,
                        const float* bias /*or NULL*/, float* C) {
    float* Bcol = (float*)malloc((size_t)K*NTILE*sizeof(float));
    float* Cblk = (float*)malloc((size_t)M*NTILE*sizeof(float));
    float* bcol = bias ? (float*)malloc((size_t)NTILE*sizeof(float)) : NULL;
    if (!Bcol || !Cblk || (bias && !bcol)) { fprintf(stderr,"oom gemm\n"); return -1; }
    int rc = 0;
    for (int c0 = 0; c0 < N && rc == 0; c0 += NTILE) {
        int nb = (c0 + NTILE <= N) ? NTILE : (N - c0);
        /* pack the [K, nb] column-block contiguously */
        for (int kk = 0; kk < K; kk++)
            for (int cc = 0; cc < nb; cc++)
                Bcol[(size_t)kk*nb + cc] = W[(size_t)kk*N + (c0 + cc)];
        if (bias) for (int cc = 0; cc < nb; cc++) bcol[cc] = bias[c0 + cc];
        rc = helix_matmul_tile(A, M, K, Bcol, nb, bias ? bcol : NULL, Cblk);
        if (rc) break;
        /* scatter Cblk[M,nb] into C[M,N] at columns [c0, c0+nb) */
        for (int r = 0; r < M; r++)
            for (int cc = 0; cc < nb; cc++)
                C[(size_t)r*N + (c0 + cc)] = Cblk[(size_t)r*nb + cc];
    }
    free(Bcol); free(Cblk); if (bcol) free(bcol);
    return rc;
}

/* GELU over n elements (single invocation; n=T*DFF=5*3072=15360 floats < 256K). */
static int helix_gelu(const float* x, int n, float* y) {
    FILE* f = fopen(IN_PATH, "wb"); if (!f) { perror("open in"); return -1; }
    w_i32(f, OP_GELU); w_i32(f, n); w_i32(f, 0); w_i32(f, 0); w_i32(f, 0); w_i32(f, 0);
    fwrite(x, sizeof(float), (size_t)n, f);
    fclose(f);
    if (exec_op()) return -1;
    return read_out(y, (size_t)n);
}

/* residual add a[n] + b[n]. */
static int helix_add(const float* a, const float* b, int n, float* y) {
    FILE* f = fopen(IN_PATH, "wb"); if (!f) { perror("open in"); return -1; }
    w_i32(f, OP_ADD); w_i32(f, n); w_i32(f, 0); w_i32(f, 0); w_i32(f, 0); w_i32(f, 0);
    fwrite(a, sizeof(float), (size_t)n, f);
    fwrite(b, sizeof(float), (size_t)n, f);
    fclose(f);
    if (exec_op()) return -1;
    return read_out(y, (size_t)n);
}

/* causal softmax (with 0.125 scale folded in) over scores[rows*cols]. */
static int helix_softmax_causal(const float* scores, int rows, int cols, float* probs) {
    FILE* f = fopen(IN_PATH, "wb"); if (!f) { perror("open in"); return -1; }
    w_i32(f, OP_SOFTMAX); w_i32(f, rows); w_i32(f, cols); w_i32(f, 0); w_i32(f, 0); w_i32(f, 0);
    fwrite(scores, sizeof(float), (size_t)rows*cols, f);
    fclose(f);
    if (exec_op()) return -1;
    return read_out(probs, (size_t)rows*cols);
}

/* ===================== embedding gather (host byte-movement) ===================== */
/* x[s,:] = wte[ids[s]] + wpe[s]. The add mirrors the capstone's input injection. */
static void embed_gather(const int* ids, int T, float* x) {
    const float* wte = &g_wbase[off_wte()];
    const float* wpe = &g_wbase[off_wpe()];
    for (int s = 0; s < T; s++) {
        const float* rw = &wte[(size_t)ids[s]*DM];
        const float* rp = &wpe[(size_t)s*DM];
        for (int c = 0; c < DM; c++) x[(size_t)s*DM + c] = rw[c] + rp[c];
    }
}

/* ===================== one transformer block (all arithmetic in Helix) ===================== */
/* x[T,DM] is the residual stream (updated in place). Mirrors gpt2_infer.c forward_layer_gpt2. */
static int forward_block(int L, int T, float* x) {
    LayerOff lo = layer_offsets();
    long b = off_layer(L);
    const float* ln1g = &g_wbase[b+lo.ln1g];
    const float* ln1b = &g_wbase[b+lo.ln1b];
    const float* attW = &g_wbase[b+lo.attW];   /* [DM, 3*DM] */
    const float* attb = &g_wbase[b+lo.attb];
    const float* prjW = &g_wbase[b+lo.prjW];   /* [DM, DM] */
    const float* prjb = &g_wbase[b+lo.prjb];
    const float* ln2g = &g_wbase[b+lo.ln2g];
    const float* ln2b = &g_wbase[b+lo.ln2b];
    const float* fcW  = &g_wbase[b+lo.fcW];     /* [DM, DFF] */
    const float* fcb  = &g_wbase[b+lo.fcb];
    const float* pjW  = &g_wbase[b+lo.pjW];     /* [DFF, DM] */
    const float* pjb  = &g_wbase[b+lo.pjb];

    int rc = 0;
    float* xn   = (float*)malloc((size_t)T*DM*sizeof(float));
    float* qkv  = (float*)malloc((size_t)T*3*DM*sizeof(float));
    float* Qh   = (float*)malloc((size_t)T*DH*sizeof(float));
    float* Kh   = (float*)malloc((size_t)T*DH*sizeof(float));
    float* Vh   = (float*)malloc((size_t)T*DH*sizeof(float));
    float* sc   = (float*)malloc((size_t)T*T*sizeof(float));
    float* aw   = (float*)malloc((size_t)T*T*sizeof(float));
    float* aoh  = (float*)malloc((size_t)T*DH*sizeof(float));
    float* ctx  = (float*)malloc((size_t)T*DM*sizeof(float));
    float* proj = (float*)malloc((size_t)T*DM*sizeof(float));
    float* xn2  = (float*)malloc((size_t)T*DM*sizeof(float));
    float* mlp1 = (float*)malloc((size_t)T*DFF*sizeof(float));
    float* mlpg = (float*)malloc((size_t)T*DFF*sizeof(float));
    float* mlp2 = (float*)malloc((size_t)T*DM*sizeof(float));
    if (!xn||!qkv||!Qh||!Kh||!Vh||!sc||!aw||!aoh||!ctx||!proj||!xn2||!mlp1||!mlpg||!mlp2) { fprintf(stderr,"oom block\n"); return -1; }

    /* --- attention --- */
    rc = helix_layernorm(x, ln1g, ln1b, T, DM, xn);                    if (rc) goto done;
    rc = helix_matmul(xn, T, DM, attW, 3*DM, attb, qkv);               if (rc) goto done;  /* QKV[T,3DM] */
    /* 12-head loop: pack Q/K/V head columns (contiguous), scores, scale+softmax, @V, scatter */
    for (int h = 0; h < NH && rc == 0; h++) {
        int qb = h*DH, kb = DM + h*DH, vb = 2*DM + h*DH;
        for (int s = 0; s < T; s++) {
            for (int d = 0; d < DH; d++) {
                Qh[(size_t)s*DH + d] = qkv[(size_t)s*3*DM + qb + d];
                Kh[(size_t)s*DH + d] = qkv[(size_t)s*3*DM + kb + d];
                Vh[(size_t)s*DH + d] = qkv[(size_t)s*3*DM + vb + d];
            }
        }
        /* scores[T,T] = Q_h[T,DH] @ K_h^T[DH,T]. We need B=[DH,T] row-major = K_h^T.
         * Pack Kt[DH,T] = transpose(Kh[T,DH]) (byte-movement). */
        {
            float* Kt = (float*)malloc((size_t)DH*T*sizeof(float));
            if (!Kt) { rc=-1; break; }
            for (int s = 0; s < T; s++) for (int d = 0; d < DH; d++) Kt[(size_t)d*T + s] = Kh[(size_t)s*DH + d];
            rc = helix_matmul(Qh, T, DH, Kt, T, NULL, sc);            /* sc[T,T] */
            free(Kt);
            if (rc) break;
        }
        rc = helix_softmax_causal(sc, T, T, aw);                       if (rc) break;  /* scale+causal+softmax */
        rc = helix_matmul(aw, T, T, Vh, DH, NULL, aoh);                if (rc) break;  /* aoh[T,DH]=aw@V_h */
        for (int s = 0; s < T; s++) for (int d = 0; d < DH; d++) ctx[(size_t)s*DM + (qb + d)] = aoh[(size_t)s*DH + d];
    }
    if (rc) goto done;
    rc = helix_matmul(ctx, T, DM, prjW, DM, prjb, proj);              if (rc) goto done;  /* c_proj[T,DM] */
    rc = helix_add(x, proj, T*DM, x);                                 if (rc) goto done;  /* residual */
    /* --- MLP --- */
    rc = helix_layernorm(x, ln2g, ln2b, T, DM, xn2);                  if (rc) goto done;
    rc = helix_matmul(xn2, T, DM, fcW, DFF, fcb, mlp1);               if (rc) goto done;  /* c_fc[T,DFF] */
    rc = helix_gelu(mlp1, T*DFF, mlpg);                               if (rc) goto done;
    rc = helix_matmul(mlpg, T, DFF, pjW, DM, pjb, mlp2);              if (rc) goto done;  /* mlp c_proj[T,DM] */
    rc = helix_add(x, mlp2, T*DM, x);                                 if (rc) goto done;  /* residual */
done:
    free(xn);free(qkv);free(Qh);free(Kh);free(Vh);free(sc);free(aw);free(aoh);free(ctx);free(proj);free(xn2);free(mlp1);free(mlpg);free(mlp2);
    return rc;
}

/* ===================== npy reader (for ref_block0.npy) ===================== */
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

/* ===================== block-0 mode ===================== */
static int run_block0(const char* refpath) {
    int ids[] = {464, 3139, 286, 4881, 318};
    int T = (int)(sizeof(ids)/sizeof(ids[0]));
    float* x = (float*)malloc((size_t)T*DM*sizeof(float));
    if (!x) return 2;
    embed_gather(ids, T, x);
    printf("[cpu] block-0: T=%d ids=", T); for (int i=0;i<T;i++) printf(" %d", ids[i]); printf("\n");
    if (forward_block(0, T, x)) { fprintf(stderr, "forward_block failed\n"); free(x); return 2; }

    char dump[256]; snprintf(dump, sizeof(dump), "%s/helix_block0.bin", GPC_DIR);
    FILE* of = fopen(dump, "wb");
    if (!of) { fprintf(stderr, "open %s: %s\n", dump, strerror(errno)); free(x); return 2; }
    fwrite(x, sizeof(float), (size_t)T*DM, of); fclose(of);

    float* ref = read_npy_f4(refpath, (size_t)T*DM);
    if (!ref) { fprintf(stderr, "could not read ref %s\n", refpath); free(x); return 2; }
    double max_abs = 0.0, sum_abs = 0.0, max_rel_floor = 0.0;
    int argr=0, argc2=0, nonfinite=0; long ncell=(long)T*DM;
    for (int s = 0; s < T; s++) for (int c = 0; c < DM; c++) {
        double g = (double)x[(size_t)s*DM + c];
        double o = (double)ref[(size_t)s*DM + c];
        if (!isfinite(g)) { nonfinite++; if (nonfinite <= 4) fprintf(stderr,"  NON-FINITE helix[%d,%d]=%g (ref=%g)\n",s,c,g,o); continue; }
        double ae = fabs(g - o), ao = fabs(o);
        double re_floor = ae / (ao > 1.0 ? ao : 1.0);
        if (ae > max_abs) max_abs = ae;
        sum_abs += ae;
        if (re_floor > max_rel_floor) { max_rel_floor = re_floor; argr=s; argc2=c; }
    }
    double mean_abs = sum_abs / (double)ncell;
    double wg = (double)x[(size_t)argr*DM + argc2], wo = (double)ref[(size_t)argr*DM + argc2];
    /* ACCEPTANCE: max_abs < 1e-3 AND mean_abs < 1e-4 (the P2 gate). */
    int pass = (nonfinite == 0) && (max_abs < 1e-3) && (mean_abs < 1e-4);
    printf("CPU GPT-2 block-0 parity (T=%d x %d=%ld cells, %d non-finite):\n", T, DM, ncell, nonfinite);
    printf("  max_abs=%.3e  mean_abs=%.3e   [GATE: max_abs<1e-3 AND mean_abs<1e-4]\n", max_abs, mean_abs);
    printf("  worst-cell [%d,%d]: helix=%.6g ref=%.6g (abs=%.3e, floored-rel=%.3e)\n", argr, argc2, wg, wo, fabs(wg-wo), max_rel_floor);
    printf("%s\n", pass ? "GPT2_CPU_BLOCK0_PARITY_PASS" : "GPT2_CPU_BLOCK0_PARITY_FAIL");
    free(x); free(ref);
    return pass ? 0 : 1;
}

/* ===================== full forward (12 layers + ln_f + tied head) ===================== */
/* tied LM head, LAST position only: logits[v] = dot(x_last[DM], wte[v][DM]) for v in 0..NV.
 * Computed in Helix by tiling the vocab: for each block of VTILE vocab rows, transpose the
 * wte block [vb, DM] -> Wt[DM, vb] (byte-movement) and run op_matmul x_last[1,DM] @ Wt[DM,vb]
 * -> logits[1, vb]. ALL arithmetic stays in the Helix op ELF. */
static int VTILE = 512;
static int helix_lm_head_last(const float* x_last, float* logits) {
    const float* wte = &g_wbase[off_wte()];
    float* Wt   = (float*)malloc((size_t)DM*VTILE*sizeof(float));
    float* lblk = (float*)malloc((size_t)VTILE*sizeof(float));
    if (!Wt || !lblk) { fprintf(stderr,"oom head\n"); return -1; }
    int rc = 0;
    for (int v0 = 0; v0 < NV && rc == 0; v0 += VTILE) {
        int vb = (v0 + VTILE <= NV) ? VTILE : (NV - v0);
        /* Wt[d, j] = wte[v0+j][d]  (transpose the [vb,DM] block into [DM,vb]) */
        for (int j = 0; j < vb; j++) {
            const float* wr = &wte[(size_t)(v0+j)*DM];
            for (int d = 0; d < DM; d++) Wt[(size_t)d*vb + j] = wr[d];
        }
        rc = helix_matmul(x_last, 1, DM, Wt, vb, NULL, lblk);   /* [1,vb] = x_last @ Wt */
        if (rc) break;
        for (int j = 0; j < vb; j++) logits[v0 + j] = lblk[j];
    }
    free(Wt); free(lblk);
    return rc;
}

/* run the full GPT-2 forward for ids[0..T); fill out_last_logits[NV] (last real-token row).
 * 12 layers -> ln_f -> tied head (last position). ALL arithmetic in Helix. */
static int forward_full(const int* ids, int T, float* out_last_logits) {
    int rc = 0;
    float* x   = (float*)malloc((size_t)T*DM*sizeof(float));
    float* xnf = (float*)malloc((size_t)T*DM*sizeof(float));
    if (!x || !xnf) { fprintf(stderr,"oom full\n"); free(x); free(xnf); return -1; }
    embed_gather(ids, T, x);
    for (int L = 0; L < NL && rc == 0; L++) rc = forward_block(L, T, x);
    if (rc) { free(x); free(xnf); return rc; }
    /* final LayerNorm (ln_f) */
    rc = helix_layernorm(x, &g_wbase[off_lnfg()], &g_wbase[off_lnfb()], T, DM, xnf);
    if (rc) { free(x); free(xnf); return rc; }
    /* tied head on the LAST real-token row only (greedy decision row) */
    rc = helix_lm_head_last(&xnf[(size_t)(T-1)*DM], out_last_logits);
    free(x); free(xnf);
    return rc;
}

static int argmax_row(const float* v, int n) {
    int a = 0; float best = v[0];
    for (int i = 1; i < n; i++) if (v[i] > best) { best = v[i]; a = i; }
    return a;
}

static int read_ids_file(const char* path, int* ids, int maxn) {
    FILE* f = fopen(path, "r"); if (!f) { fprintf(stderr, "open ids '%s': %s\n", path, strerror(errno)); return -1; }
    int n = 0, v;
    while (n < maxn && fscanf(f, "%d", &v) == 1) ids[n++] = v;
    fclose(f);
    return n;
}

/* --logits: dump last-row logits[NV], compare argmax + max-abs diff to the oracle. */
static int run_logits(const char* ref_logits_path, const char* ref_argmax_path, const char* ids_path) {
    int ids[1024]; int T;
    if (ids_path) { T = read_ids_file(ids_path, ids, 1024); if (T <= 0) return 2; }
    else { int c[] = {464,3139,286,4881,318}; T = 5; memcpy(ids, c, sizeof(c)); }
    printf("[logits] T=%d ids:", T); for (int i=0;i<T;i++) printf(" %d", ids[i]); printf("\n");
    float* logits = (float*)malloc((size_t)NV*sizeof(float));
    if (!logits) return 2;
    if (forward_full(ids, T, logits)) { fprintf(stderr,"forward_full failed\n"); free(logits); return 2; }
    int am = argmax_row(logits, NV);
    { FILE* of = fopen("/tmp/gpc/helix_logits_last.bin", "wb"); if (of) { fwrite(logits, sizeof(float), (size_t)NV, of); fclose(of); } }
    int ref_am = -1;
    if (ref_argmax_path) { FILE* af = fopen(ref_argmax_path, "r"); if (af) { if (fscanf(af, "%d", &ref_am) != 1) ref_am = -1; fclose(af); } }
    double max_abs = 0.0; int nonfinite = 0, wc = 0; float* ref = NULL;
    if (ref_logits_path) {
        FILE* rf = fopen(ref_logits_path, "rb");
        if (rf) { ref = (float*)malloc((size_t)NV*sizeof(float)); size_t got = fread(ref, sizeof(float), (size_t)NV, rf); fclose(rf);
                  if (got != (size_t)NV) { free(ref); ref = NULL; } }
    }
    if (ref) {
        if (ref_am < 0) ref_am = argmax_row(ref, NV);
        for (int i = 0; i < NV; i++) { double g=(double)logits[i]; if (!isfinite(g)) { nonfinite++; continue; } double ae=fabs(g-(double)ref[i]); if (ae>max_abs){max_abs=ae;wc=i;} }
    } else { for (int i=0;i<NV;i++) if (!isfinite(logits[i])) nonfinite++; }
    int argmax_match = (nonfinite == 0) && (ref_am >= 0) && (am == ref_am);
    int diff_ok = (ref == NULL) ? 1 : (max_abs < 1e-2);   /* P3 gate: logit max-abs < 1e-2 */
    int pass = argmax_match && diff_ok;
    printf("CPU GPT-2 full-logits parity (T=%d, NV=%d, %d non-finite):\n", T, NV, nonfinite);
    printf("  helix argmax=%d (logit=%.5g)  oracle argmax=%d  -> %s\n", am, logits[am], ref_am, argmax_match?"ARGMAX_MATCH":"ARGMAX_MISMATCH");
    if (ref) printf("  max_abs logit diff=%.5e at id %d (helix=%.6g ref=%.6g)  [gate 1e-2: %s]\n", max_abs, wc, logits[wc], ref[wc], diff_ok?"ok":"OVER");
    printf("%s\n", pass ? "GPT2_CPU_LOGITS_PARITY_PASS" : "GPT2_CPU_LOGITS_PARITY_FAIL");
    free(logits); if (ref) free(ref);
    return pass ? 0 : 1;
}

/* --generate N: greedy autoregressive loop; compare token-for-token to the oracle. */
static int run_generate(int Ngen, const char* ids_path, const char* ref_gen_path) {
    int ids[1024]; int T0;
    if (ids_path) { T0 = read_ids_file(ids_path, ids, 1024 - Ngen); if (T0 <= 0) return 2; }
    else { int c[] = {464,3139,286,4881,318}; T0 = 5; memcpy(ids, c, sizeof(c)); }
    int T = T0;
    printf("[generate] N=%d prompt T=%d ids:", Ngen, T0); for (int i=0;i<T0;i++) printf(" %d", ids[i]); printf("\n");
    float* logits = (float*)malloc((size_t)NV*sizeof(float));
    if (!logits) return 2;
    int nonfinite_any = 0;
    for (int step = 0; step < Ngen; step++) {
        if (forward_full(ids, T, logits)) { fprintf(stderr,"forward_full failed at step %d\n", step); free(logits); return 2; }
        for (int i = 0; i < NV; i++) if (!isfinite(logits[i])) { nonfinite_any = 1; break; }
        ids[T++] = argmax_row(logits, NV);
    }
    free(logits);
    { FILE* gf = fopen("/tmp/gpc/helix_gen_ids.txt", "w"); if (gf) { for (int i=0;i<T;i++) fprintf(gf, "%d%s", ids[i], i+1<T?" ":"\n"); fclose(gf); } }
    printf("HELIX_GEN_IDS:"); for (int i=0;i<T;i++) printf(" %d", ids[i]); printf("\n");
    int pass = (nonfinite_any == 0);
    if (ref_gen_path) {
        int ref[1024]; int rn = read_ids_file(ref_gen_path, ref, 1024);
        if (rn <= 0) { fprintf(stderr, "could not read ref gen %s\n", ref_gen_path); pass = 0; }
        else {
            int match = (rn == T); int firstdiff = -1; int lim = rn < T ? rn : T;
            for (int i = 0; i < lim; i++) if (ids[i] != ref[i]) { match = 0; if (firstdiff<0) firstdiff=i; }
            printf("  oracle gen ids (%d):", rn); for (int i=0;i<rn;i++) printf(" %d", ref[i]); printf("\n");
            if (match) printf("  TOKEN_FOR_TOKEN_MATCH (%d ids)\n", T);
            else printf("  TOKEN_MISMATCH (helix %d, oracle %d, first diff @ %d)\n", T, rn, firstdiff);
            pass = pass && match;
        }
    }
    printf("%s\n", pass ? "GPT2_CPU_GENERATE_MATCH_PASS" : "GPT2_CPU_GENERATE_MATCH_FAIL");
    return pass ? 0 : 1;
}

int main(int argc, char** argv) {
    const char* e;
    if ((e=getenv("HX_NL")))    NL=atoi(e);
    if ((e=getenv("HX_D")))     DM=atoi(e);
    if ((e=getenv("HX_HEADS"))) NH=atoi(e);
    if ((e=getenv("HX_V")))     NV=atoi(e);
    if ((e=getenv("HX_CTX")))   NC=atoi(e);
    if ((e=getenv("HX_DFF")))   DFF=atoi(e);
    if ((e=getenv("HX_NTILE"))) NTILE=atoi(e);
    if ((e=getenv("HX_VTILE"))) VTILE=atoi(e);
    DH = DM / NH;

    if (argc < 4) {
        fprintf(stderr,
          "usage:\n"
          "  %s <weights> <ops.bin> --block0 <ref_block0.npy>\n"
          "  %s <weights> <ops.bin> --logits [<ref_logits_last.bin>] [<ref_argmax.txt>] [<ref_ids.txt>]\n"
          "  %s <weights> <ops.bin> --generate <N> [<ref_ids.txt>] [<ref_gen_ids.txt>]\n",
          argv[0], argv[0], argv[0]);
        return 2;
    }
    const char* wpath = argv[1];
    OP_ELF             = argv[2];
    const char* mode   = argv[3];

    /* staging dir + file paths */
    mkdir(GPC_DIR, 0755);
    snprintf(IN_PATH,  sizeof(IN_PATH),  "%s/in.bin",  GPC_DIR);
    snprintf(OUT_PATH, sizeof(OUT_PATH), "%s/out.bin", GPC_DIR);
    { struct stat st; if (stat(OP_ELF,&st)!=0) { fprintf(stderr,"op ELF missing: %s\n", OP_ELF); return 2; } }

    if (load_gpt2_weights(wpath)) return 2;

    int rc = 2;
    if (strcmp(mode, "--block0") == 0) {
        if (argc < 5) { fprintf(stderr, "--block0 needs <ref_block0.npy>\n"); return 2; }
        rc = run_block0(argv[4]);
    } else if (strcmp(mode, "--logits") == 0) {
        const char* ref_logits = (argc > 4) ? argv[4] : NULL;
        const char* ref_argmax = (argc > 5) ? argv[5] : NULL;
        const char* ids_path   = (argc > 6) ? argv[6] : NULL;
        rc = run_logits(ref_logits, ref_argmax, ids_path);
    } else if (strcmp(mode, "--generate") == 0) {
        if (argc < 5) { fprintf(stderr, "--generate needs <N>\n"); return 2; }
        int Ngen = atoi(argv[4]);
        const char* ids_path     = (argc > 5) ? argv[5] : NULL;
        const char* ref_gen_path = (argc > 6) ? argv[6] : NULL;
        rc = run_generate(Ngen, ids_path, ref_gen_path);
    } else {
        fprintf(stderr, "unknown mode '%s'\n", mode); return 2;
    }

    if (g_map && g_map != MAP_FAILED) munmap(g_map, g_maplen);
    if (g_fd >= 0) close(g_fd);
    return rc;
}
