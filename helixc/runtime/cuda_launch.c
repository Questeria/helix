/* helixc/runtime/cuda_launch.c -- Helix GPU first-light launcher.
 *
 * DoD criterion #3 (docs/HELIX_V1_DEFINITION_OF_DONE.md) -- "GPU executes", THE GATE.
 *
 * A tiny, generic CUDA Driver-API host launcher. It loads a PTX module as TEXT
 * (exactly the form kovc's emit_ptx_* path produces, or nvcc -ptx for testing),
 * gets a vector_add-class kernel  f(const float* a, const float* b, float* c, int n),
 * runs it on the GPU, copies the result back, and verifies c[i] == a[i] + b[i].
 *
 * WHY C, AND WHY OUTSIDE THE SELF-HOST FIXPOINT (Decision D2, HELIX_FINISH_PLAN.md):
 * libcuda.so is a DYNAMIC shared library; the Helix compiler emits a static,
 * syscall-only, single-PT_LOAD ELF with no dynamic linker, so it cannot link or
 * call libcuda today. This launcher is therefore a trusted-tool boundary, exactly
 * like ptxas or the from-raw-binary build ladder: it sits OUTSIDE the self-host
 * fixpoint. Helix emits the PTX (already real); this C shim is what talks to the
 * driver. In-Helix dynamic linking (path-b) is scheduled to retire this shim
 * before the formal v1.0 freeze, keeping the toolchain pure at the limit.
 *
 * Build (WSL): gcc cuda_launch.c -I/usr/local/cuda/include -L/usr/lib/wsl/lib -lcuda -o cuda_launch
 * Usage:       cuda_launch <module.ptx> <kernel_name> [N]
 * Exit:        0 = launched + numerically verified; 1 = result mismatch; 2 = usage/driver error.
 */

#include <cuda.h>
#include <cublas_v2.h>   /* G1: fenced cuBLAS f32 GEMM oracle (gemm_perf mode only); link -lcublas */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>      /* v1.5 #4: fixed-width words for the from-scratch SHA-256 + Freivalds mod-p */

/* v1.5 #4 (verifiable-inference receipts): a from-scratch FIPS-180-4 SHA-256 (no deps), used to COMMIT
 * to the ternary matmul's weights/input/output in the receipt and to derive the Fiat-Shamir Freivalds
 * challenges. Host-side, OUTSIDE the kovc self-host fixpoint (like the f16/e2m1/e4m3 codecs above). It is
 * KAT-gated by the sha256_selftest op BEFORE any receipt is trusted -- an endianness/padding bug would
 * silently void every binding (the #4 design's #1 risk). */
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
/* one-shot SHA-256 over [data,len) -> out[32] (big-endian digest). FIPS-180-4 padding. */
static void sha256(const unsigned char* data, size_t len, unsigned char out[32]) {
    uint32_t st[8] = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    size_t full = len / 64;
    for (size_t i = 0; i < full; i++) sha256_block(st, data + i*64);
    unsigned char buf[128]; size_t rem = len - full*64;          /* rem in [0,63] */
    memcpy(buf, data + full*64, rem);
    buf[rem] = 0x80;
    size_t padlen = (rem < 56) ? 64 : 128;                       /* room for the 0x80 + the 8-byte length */
    memset(buf + rem + 1, 0, padlen - rem - 1 - 8);
    uint64_t bits = (uint64_t)len * 8;
    for (int i = 0; i < 8; i++) buf[padlen - 1 - i] = (unsigned char)(bits >> (8*i));
    sha256_block(st, buf);
    if (padlen == 128) sha256_block(st, buf + 64);
    for (int i = 0; i < 8; i++) { out[i*4]=(unsigned char)(st[i]>>24); out[i*4+1]=(unsigned char)(st[i]>>16); out[i*4+2]=(unsigned char)(st[i]>>8); out[i*4+3]=(unsigned char)st[i]; }
}
/* hex-encode a 32-byte digest into a 65-char buffer (64 lowercase hex + NUL). */
static void sha256_hex(const unsigned char dig[32], char out[65]) {
    static const char HX[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) { out[i*2] = HX[dig[i] >> 4]; out[i*2+1] = HX[dig[i] & 15]; }
    out[64] = 0;
}

/* v1.5 #4 receipt helpers (shared by receipt_emit + receipt_check). The committed W (ternary) / X (int)
 * are regenerated DETERMINISTICALLY from the receipt's seed fields -- seed-parameterized so flipping a
 * seed in a forged receipt changes the regenerated matrix and trips the H_W/H_X commitment (the
 * tampered-weight NC). seed 0 == the imatmul/ternary_matmul test data (W[i]=(i%3)-1, X[i]=((i*13+7)%11)-5). */
static void receipt_gen_W(int* W, size_t n, unsigned seed) { for (size_t i = 0; i < n; i++) W[i] = (int)(((i + seed) % 3)) - 1; }
static void receipt_gen_X(int* X, size_t n, unsigned seed) { for (size_t i = 0; i < n; i++) X[i] = (int)((((i + seed) * 13 + 7) % 11)) - 5; }
/* Fiat-Shamir Freivalds challenge: r = SHA256(dW || dX || dC || round || j) reduced UNIFORMLY mod p, bound
 * to the RECOMPUTED commitments so a malicious runner cannot pick an r in the error null-space (a checker
 * that trusted runner-supplied projections would be unsound -- the runner-chosen-challenge hole; NC closes
 * it). DE-BIASED by rejection sampling over the digest's 8 four-byte words: a bare u32%p over-represents
 * the residues {0,1} (since 2^32 = 2p+2), which would weaken the per-round Freivalds bound to ~1.5/p;
 * rejecting any word >= 2p (the largest multiple of p <= 2^32) makes r EXACTLY uniform on [0,p), so the
 * per-round bound is a clean 1/p and the stated <= (1/p)^t holds literally (not approximately). */
static long long receipt_fs_r(const unsigned char dW[32], const unsigned char dX[32], const unsigned char dC[32], int rd, int j, long long p) {
    unsigned char fb[104]; memcpy(fb, dW, 32); memcpy(fb + 32, dX, 32); memcpy(fb + 64, dC, 32);
    fb[96]=(unsigned char)(rd>>24); fb[97]=(unsigned char)(rd>>16); fb[98]=(unsigned char)(rd>>8); fb[99]=(unsigned char)rd;
    fb[100]=(unsigned char)(j>>24); fb[101]=(unsigned char)(j>>16); fb[102]=(unsigned char)(j>>8); fb[103]=(unsigned char)j;
    unsigned char fd[32]; sha256(fb, 104, fd);
    unsigned int lim = 2u * (unsigned int)p;   /* = 4294967294; reject [2p, 2^32) -> uniform u%p on [0,p) */
    for (int w = 0; w < 8; w++) {
        unsigned int u = ((unsigned)fd[w*4]<<24)|((unsigned)fd[w*4+1]<<16)|((unsigned)fd[w*4+2]<<8)|(unsigned)fd[w*4+3];
        if (u < lim) return (long long)(u % (unsigned)p);
    }
    /* all 8 words landed in [2p,2^32) (probability ~ (2/2^32)^8 = 2^-248) -> negligible fallback */
    { unsigned int u = ((unsigned)fd[0]<<24)|((unsigned)fd[1]<<16)|((unsigned)fd[2]<<8)|(unsigned)fd[3]; return (long long)(u % (unsigned)p); }
}

/* v1.5 S1: IEEE-754 binary16 <-> binary32, in plain C (NOT cuda_fp16.h -- that is a
 * C++ header that does not compile cleanly under gcc-as-C, AND a from-scratch codec
 * keeps the hgemm oracle's f16 conversion trusted-from-source, matching the Helix
 * no-hidden-NVIDIA-math ethos). f16 storage is a uint16. f16->f32 is EXACT (every
 * binary16 maps to one binary32) -- this is the one that MUST be rigorous, since it
 * decodes the GPU's cvt.rn.f16.f32 output and up-converts the shared rounded inputs.
 * f32->f16 is round-to-nearest-even (used only to choose the input bits both sides
 * then share, so its exact rounding is non-critical, only determinism + validity).
 * Covers normal/subnormal/zero/inf/nan; the hgemm data is bounded so only the
 * normal+zero paths are exercised, but the full range is implemented for honesty. */
static unsigned short f32_to_f16(float f) {
    unsigned int x; memcpy(&x, &f, 4);
    unsigned int sign = (x >> 16) & 0x8000u;
    unsigned int e8 = (x >> 23) & 0xFFu;
    unsigned int mant = x & 0x7FFFFFu;
    if (e8 == 0xFFu) return (unsigned short)(sign | 0x7C00u | (mant ? 0x200u : 0u)); /* inf/nan */
    int exp = (int)e8 - 127 + 15;
    if (exp >= 0x1F) return (unsigned short)(sign | 0x7C00u);                        /* overflow -> inf */
    if (exp <= 0) {                                                                  /* subnormal/zero */
        if (exp < -10) return (unsigned short)sign;
        mant |= 0x800000u;
        int shift = 14 - exp;                       /* exp in [-10,0] -> shift [14,24] */
        unsigned int half = mant >> shift;
        unsigned int rem = mant & ((1u << shift) - 1u);
        unsigned int mid = 1u << (shift - 1);
        if (rem > mid || (rem == mid && (half & 1u))) half++;
        return (unsigned short)(sign | half);
    }
    unsigned int half = ((unsigned int)exp << 10) | (mant >> 13);
    unsigned int rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1u))) half++; /* RNE; a carry into exp is correct */
    return (unsigned short)(sign | half);
}
static float f16_to_f32(unsigned short h) {
    unsigned int sign = (unsigned int)(h & 0x8000u) << 16;
    unsigned int exp = (h >> 10) & 0x1Fu;
    unsigned int mant = h & 0x3FFu;
    unsigned int out;
    if (exp == 0u) {
        if (mant == 0u) { out = sign; }
        else {                                       /* subnormal: normalize into binary32 */
            int e = 0;
            while (!(mant & 0x400u)) { mant <<= 1; e++; }
            mant &= 0x3FFu;
            out = sign | ((unsigned int)(127 - 14 - e) << 23) | (mant << 13);  /* 113-e (S1-audit fix): a half subnormal mantissa m*2^-10 * 2^-14 normalizes to 1.f * 2^(-14-e); biased f32 exp = (-14-e)+127 = 113-e. Was 127-15-e, an off-by-one that halved EVERY f16 subnormal (out-of-path for S1, but S2/MXFP4 hits small magnitudes). Verified by the codec_selftest op below. */
        }
    } else if (exp == 0x1Fu) {
        out = sign | 0x7F800000u | (mant << 13);     /* inf/nan */
    } else {
        out = sign | ((exp - 15u + 127u) << 23) | (mant << 13);
    }
    float f; memcpy(&f, &out, 4); return f;
}

/* v1.5 S2 (MXFP4): from-scratch OCP MXFP4 codec (NOT a library) -- E2M1 4-bit element + E8M0 8-bit
 * block scale. This is HOST code (the oracle), so bitwise is fine here; only the @kernel DEVICE path
 * forbids bitwise (it div-unpacks). E2M1 code (s | e1e0 | m): exp==0 -> subnormal mag = 0.5*m; else
 * normal mag = (1 + 0.5*m) * 2^(exp-1). Magnitudes {0,0.5,1,1.5,2,3,4,6}; the sign bit negates.
 * MUST EXACTLY match the @kernel's f32-literal if-ladder in naive_mxfp4_matmul_kernel.hx. E8M0 is a
 * pure power-of-2 exponent (bias 127): linear scale = 2^(e8m0-127); 0xFF is the OCP NaN sentinel. */
static float e2m1_decode(int code) {
    int s = (code >> 3) & 1;
    int e = (code >> 1) & 3;
    int m = code & 1;
    float mag;
    if (e == 0) mag = 0.5f * (float)m;                          /* subnormal: 0 or 0.5 */
    else mag = (1.0f + 0.5f * (float)m) * (float)(1 << (e - 1)); /* normal: (1|1.5) * 2^(e-1) */
    return s ? -mag : mag;
}
/* f32 -> nearest E2M1 code (brute-force over the 16 codes; the format is tiny). Used by the
 * round-trip selftest + any f32-source packing. */
static int e2m1_encode(float v) {
    int best = 0; float bestd = 1.0e30f;
    for (int c = 0; c < 16; c++) {
        float d = e2m1_decode(c) - v; if (d < 0) d = -d;
        if (d < bestd) { bestd = d; best = c; }
    }
    return best;
}
/* E8M0 (8-bit exponent, bias 127) -> linear f32 scale 2^(e-127). 0xFF = OCP NaN sentinel. */
static float e8m0_scale(int e8m0) {
    if (e8m0 == 0xFF) return nanf("");   /* NaN sentinel -- a caller must not feed it as data */
    return ldexpf(1.0f, e8m0 - 127);
}
/* v1.5 S3 (NVFP4): from-scratch FP8 E4M3 micro-scale codec (HOST oracle; bitwise OK). 1 sign / 4 exp
 * (bias 7) / 3 mantissa. CRITICAL: E4M3 has NO Inf -- the ONLY special is S.1111.111 = NaN; e=15, m<=6
 * are FINITE normals (max |448|). So do NOT copy the f16 'exp==0x1F -> inf' arm (line ~77). exp==0 ->
 * subnormal (m/8)*2^-6 (min nonzero 2^-9); else normal (1+m/8)*2^(e-7). Used by the NVFP4 two-level
 * scale (host-collapses e4m3_micro * fp32_tensor -> one effective f32 / 16-block, the S2 E8M0 pattern). */
static float e4m3_decode(int c) {
    int s = (c >> 7) & 1;
    int e = (c >> 3) & 15;
    int m = c & 7;
    if (e == 15 && m == 7) return nanf("");                   /* the only special: NaN (no Inf in E4M3) */
    float mag;
    if (e == 0) mag = (m / 8.0f) * ldexpf(1.0f, -6);          /* subnormal: (m/8)*2^-6 */
    else        mag = (1.0f + m / 8.0f) * ldexpf(1.0f, e - 7); /* normal: (1+m/8)*2^(e-7) */
    return s ? -mag : mag;
}
/* f32 -> nearest E4M3 code (brute-force over the 256 codes, skipping the +/-NaN sentinel); for the
 * round-trip selftest + any f32-source micro-scale packing. */
static int e4m3_encode(float v) {
    int best = 0; float bestd = 1.0e30f;
    for (int c = 0; c < 256; c++) {
        if ((c & 0x7F) == 0x7F) continue;                     /* skip +NaN / -NaN */
        float d = e4m3_decode(c) - v; if (d < 0) d = -d;
        if (d < bestd) { bestd = d; best = c; }
    }
    return best;
}

/* v1.5 #2 LEG 1 (certified kernels -- translation validation): a data-independence witness over the
 * emitted PTX (HOST static text analysis, no GPU). Taint every loaded DATA value (the dest of
 * ld.global/ld.shared -- NOT ld.param, which loads dims/pointers, not data) and propagate def->use to a
 * fixpoint, then REJECT if a tainted value reaches: (1) a setp compare source, (2) a predicated branch
 * or predicated-op guard, (3) a selp/slct SELECTOR, or (4) a memory-ADDRESS operand (data-dependent
 * gather/scatter). FAIL-CLOSED (-1) on a call, on a tainted non-polynomial op (div/rcp/sqrt/ex2/...), on
 * any cap saturation (regs/taint/iteration/line-length), or on non-convergence. PASS => control flow,
 * selection, AND addressing are all data-INDEPENDENT, so for a fixed (M,K,N) the kernel runs a FIXED
 * straight-line dataflow over the inputs. This is the load-bearing PRECONDITION; LEG 2 (exact 0/1 basis)
 * AND LEG 3 (bilinearity within the f32 envelope) are BOTH co-necessary on top of it to conclude
 * f == matmul -- LEG 3 is NOT optional (the nonlinearity a*a NC is invisible to the 0/1 basis). Honest
 * scope: this lifts a single empirical sample to basis-exact + bilinearity-SAMPLED equivalence on this
 * compiled shape, under the affine-addressing/arithmetic assumption -- a translation-validation WITNESS,
 * not a machine-checked proof. SCOPE NOTE: the certified kernel set (naive_matmul + the atb/abt adjoints)
 * emits only @...bra predication and affine addressing; predicated MEMORY ops (@%p ld/st) and cp.async
 * (multi-bracket, tiled kernels only) are out of this increment's scope -- generalizing the witness to
 * tiled kernels would extend the @-arm to also check the underlying ld/st address. */
static int cf_is_taint(char taint[][12], int nt, const char* r) {
    for (int i = 0; i < nt; i++) if (strcmp(taint[i], r) == 0) return 1;
    return 0;
}
/* extract %-register tokens from a PTX line, in order (e.g. "%f2","%rd3","%p0"); stores up to maxr but
 * RETURNS THE TRUE COUNT so the caller can fail-closed when a line carries more regs than the buffer. */
static int cf_regs(const char* line, char out[][12], int maxr) {
    int n = 0, total = 0;
    for (const char* p = line; *p; p++) {
        if (*p == '%') {
            char buf[12]; int j = 0; buf[j++] = '%'; p++;
            while (*p && j < 11 && ((*p >= 'a' && *p <= 'z') || (*p >= 'A' && *p <= 'Z') ||
                                    (*p >= '0' && *p <= '9') || *p == '_')) buf[j++] = *p++;
            buf[j] = 0; p--;
            total++;
            if (n < maxr) { strcpy(out[n], buf); n++; }
        }
    }
    return total;
}
/* extract the %-registers inside the FIRST [...] of a PTX line -- the memory-ADDRESS operand of a
 * ld/st. A tainted register here means a data-DEPENDENT address (gather/scatter), which breaks the
 * fixed-straight-line precondition. Returns the true count (stores up to maxr). */
static int cf_addr_regs(const char* line, char out[][12], int maxr) {
    const char* lb = strchr(line, '[');
    if (!lb) return 0;
    const char* rb = strchr(lb, ']');
    if (!rb || rb < lb) return 0;
    char sub[256]; int len = (int)(rb - lb - 1); if (len < 0) len = 0; if (len > 255) len = 255;
    memcpy(sub, lb + 1, (size_t)len); sub[len] = 0;
    return cf_regs(sub, out, maxr);
}
/* returns 1 = data-INDEPENDENT (PASS), 0 = data-DEPENDENT (FAIL), -1 = fail-closed (unanalyzable / a
 * cap saturated / non-convergence). Multi-pass taint to a fixpoint over the whole .entry. The reg buffer
 * holds CFCAP=64 (realistic PTX lines carry <10); a line exceeding it sets fail-closed rather than
 * silently dropping regs. */
#define CFCAP 64
static int cflow_witness(const char* ptx) {
    static char taint[4096][12]; int nt = 0;
    char line[1024], regs[CFCAP][12];
    const char* p; const char* e; int len;
    int failclosed = 0;
    /* PASS 0: seed taint from ld.global/ld.shared dests (the loaded DATA values) */
    for (p = ptx; *p; p = (*e ? e + 1 : e)) {
        e = p; while (*e && *e != '\n') e++;
        len = (int)(e - p); if (len > 1023) { len = 1023; failclosed = 1; } memcpy(line, p, len); line[len] = 0;
        const char* o = line; while (*o == ' ' || *o == '\t') o++;
        if (strncmp(o, "ld.global", 9) == 0 || strncmp(o, "ld.shared", 9) == 0) {
            int raw = cf_regs(line, regs, CFCAP); if (raw > CFCAP) failclosed = 1;
            if (raw >= 1 && !cf_is_taint(taint, nt, regs[0])) {
                if (nt >= 4096) failclosed = 1; else { strcpy(taint[nt], regs[0]); nt++; }
            }
        }
    }
    /* PASS 1..: propagate (any non-load/store/branch op whose source is tainted taints its dest) */
    int changed = 1, guard = 0;
    while (changed && guard < 200) {
        changed = 0; guard++;
        for (p = ptx; *p; p = (*e ? e + 1 : e)) {
            e = p; while (*e && *e != '\n') e++;
            len = (int)(e - p); if (len > 1023) { len = 1023; failclosed = 1; } memcpy(line, p, len); line[len] = 0;
            const char* o = line; while (*o == ' ' || *o == '\t') o++;
            if (*o == 0 || *o == '/' || *o == '.' || *o == '{' || *o == '}' || *o == '$' || *o == '@') continue;
            if (strncmp(o, "ld.", 3) == 0 || strncmp(o, "st.", 3) == 0 || strncmp(o, "bra", 3) == 0 ||
                strncmp(o, "ret", 3) == 0 || strncmp(o, "bar", 3) == 0) continue;  /* dest handled elsewhere / none */
            int raw = cf_regs(line, regs, CFCAP); if (raw > CFCAP) { failclosed = 1; continue; }
            if (raw < 2) continue;
            int srctaint = 0;
            for (int i = 1; i < raw; i++) if (cf_is_taint(taint, nt, regs[i])) { srctaint = 1; break; }
            if (srctaint && !cf_is_taint(taint, nt, regs[0])) {
                if (nt >= 4096) failclosed = 1; else { strcpy(taint[nt], regs[0]); nt++; changed = 1; }
            }
        }
    }
    if (changed) failclosed = 1;                       /* taint did not converge within the guard -> reject */
    /* CHECK pass -- a tainted value reaching any of these is a data-dependence violation (or fail-closed) */
    int violation = 0;
    for (p = ptx; *p; p = (*e ? e + 1 : e)) {
        e = p; while (*e && *e != '\n') e++;
        len = (int)(e - p); if (len > 1023) { len = 1023; failclosed = 1; } memcpy(line, p, len); line[len] = 0;
        const char* o = line; while (*o == ' ' || *o == '\t') o++;
        if (strncmp(o, "setp", 4) == 0) {              /* (1) data-dependent COMPARE -> control flow */
            int raw = cf_regs(line, regs, CFCAP); if (raw > CFCAP) failclosed = 1; int nr = raw > CFCAP ? CFCAP : raw;
            for (int i = 1; i < nr; i++) if (cf_is_taint(taint, nt, regs[i])) { violation = 1; break; }  /* regs[0] is the %p dest */
        } else if (strncmp(o, "selp", 4) == 0 || strncmp(o, "slct", 4) == 0) {  /* (3) data-dependent SELECTION */
            int raw = cf_regs(line, regs, CFCAP); if (raw > CFCAP) failclosed = 1; int nr = raw > CFCAP ? CFCAP : raw;
            if (nr >= 1 && cf_is_taint(taint, nt, regs[nr - 1])) violation = 1;  /* selector is the LAST operand */
        } else if (*o == '@') {                        /* (2) predicated op (usually @%p bra): guard is the first %reg */
            int raw = cf_regs(line, regs, CFCAP); if (raw > CFCAP) failclosed = 1; int nr = raw > CFCAP ? CFCAP : raw;
            if (nr >= 1 && cf_is_taint(taint, nt, regs[0])) violation = 1;
        } else if (strncmp(o, "ld.", 3) == 0 || strncmp(o, "st.", 3) == 0) {    /* (4) data-dependent ADDRESS */
            int na = cf_addr_regs(line, regs, CFCAP); if (na > CFCAP) failclosed = 1; int nr = na > CFCAP ? CFCAP : na;
            for (int i = 0; i < nr; i++) if (cf_is_taint(taint, nt, regs[i])) { violation = 1; break; }
        } else if (strncmp(o, "div", 3) == 0 || strncmp(o, "rcp", 3) == 0 || strncmp(o, "sqrt", 4) == 0 ||
                   strncmp(o, "rsqrt", 5) == 0 || strncmp(o, "ex2", 3) == 0 || strncmp(o, "lg2", 3) == 0 ||
                   strncmp(o, "sin", 3) == 0 || strncmp(o, "cos", 3) == 0 || strncmp(o, "tanh", 4) == 0) {
            /* a NON-polynomial op on a loaded value -> not a fixed polynomial -> fail-closed. (matmul's PTX
             * is +/* only; LEG 3 would also reject a non-bilinear map, but reject here too for soundness.) */
            int raw = cf_regs(line, regs, CFCAP); if (raw > CFCAP) failclosed = 1; int nr = raw > CFCAP ? CFCAP : raw;
            for (int i = 1; i < nr; i++) if (cf_is_taint(taint, nt, regs[i])) { failclosed = 1; break; }
        } else if (strncmp(o, "call", 4) == 0) {
            failclosed = 1;                            /* a function call we cannot analyze -> conservatively reject */
        }
    }
    return failclosed ? -1 : (violation ? 0 : 1);
}
#undef CFCAP

static int check(CUresult r, const char* what) {
    if (r != CUDA_SUCCESS) {
        const char* msg = 0;
        cuGetErrorString(r, &msg);
        fprintf(stderr, "CUDA error in %s: %s (%d)\n", what, msg ? msg : "?", (int)r);
        return 1;
    }
    return 0;
}
#define CK(call, what) do { if (check((call), (what))) return 2; } while (0)

/* L for one layernorm row with gamma and a perturbation delta at column cc -- the
 * forward used by the dx finite-difference. y[k]=gamma[k]*(x[k]-mean)/std + beta[k];
 * beta cancels in the central difference so it is omitted. L = sum_k dy[k]*y[k]. */
static float ln_rowL(const float* xr, const float* dyr, const float* gam, int cols, int cc, float delta) {
    float mean = 0.0f;
    for (int k = 0; k < cols; k++) mean += xr[k] + (k == cc ? delta : 0.0f);
    mean /= (float)cols;
    float var = 0.0f;
    for (int k = 0; k < cols; k++) { float v = xr[k] + (k == cc ? delta : 0.0f) - mean; var += v * v; }
    var /= (float)cols;
    float istd = 1.0f / sqrtf(var);
    float L = 0.0f;
    for (int k = 0; k < cols; k++) { float xv = xr[k] + (k == cc ? delta : 0.0f); L += dyr[k] * gam[k] * (xv - mean) * istd; }
    return L;
}

/* CPU single-head attention forward: out = softmax(0.25*Q@K^T) @ V, all [S,d] row-major.
 * Used by the attn_backward finite-difference (perturb an input, recompute out, L=sum(dOut*out)).
 * S is bounded by 64 here so the SxS scratch fits the static buffers. */
static void attn_forward_cpu(const float* Q, const float* K, const float* V, float* out, int S, int d) {
    static float sc[4096]; static float at[4096];
    for (int i = 0; i < S; i++) {
        for (int j = 0; j < S; j++) { float dot = 0.0f; for (int t = 0; t < d; t++) dot += Q[i * d + t] * K[j * d + t]; sc[i * S + j] = 0.25f * dot; }
        float mx = sc[i * S]; for (int j = 1; j < S; j++) if (sc[i * S + j] > mx) mx = sc[i * S + j];
        float sm = 0.0f; for (int j = 0; j < S; j++) sm += expf(sc[i * S + j] - mx);
        for (int j = 0; j < S; j++) at[i * S + j] = expf(sc[i * S + j] - mx) / sm;
    }
    for (int i = 0; i < S; i++) for (int t = 0; t < d; t++) { float o = 0.0f; for (int j = 0; j < S; j++) o += at[i * S + j] * V[j * d + t]; out[i * d + t] = o; }
}

/* v1.5 #2 LEG 3 helper: one square-L naive_matmul launch -- hOut[L*L] = hA @ hB (both [L*L] row-major).
 * Alloc/copy/launch/copy/free each call (L is small; bilinearity needs ~7 evals). Returns 0 on success. */
static int mm_eval(CUfunction fn, const float* hA, const float* hB, float* hOut, int L) {
    CUdeviceptr dA = 0, dB = 0, dC = 0; size_t n = (size_t)L * L; int rc = 0;
    if (cuMemAlloc(&dA, n * 4) || cuMemAlloc(&dB, n * 4) || cuMemAlloc(&dC, n * 4)) { rc = 1; goto done; }
    {
        int M = L, K = L, N = L; void* args[] = { &dA, &dB, &dC, &M, &K, &N };
        rc = cuMemcpyHtoD(dA, hA, n * 4) || cuMemcpyHtoD(dB, hB, n * 4)
           || cuLaunchKernel(fn, L, 1, 1, L, 1, 1, 0, 0, args, 0) || cuCtxSynchronize()
           || cuMemcpyDtoH(hOut, dC, n * 4);
    }
done:
    if (dA) cuMemFree(dA); if (dB) cuMemFree(dB); if (dC) cuMemFree(dC);
    return rc ? 1 : 0;
}

/* v1.5 #2 LEG 3 helper: count elements where |LHS-RHS| exceeds the derived f32 matmul rounding bound
 * tau_k = c_safe * L * u * S[k] (u = 2^-24 unit roundoff; S[k] = sum of |operands| flowing into element
 * k; c_safe an envelope multiple over the standard gamma_K = K*u summation bound -- derived, not tuned).
 * When tau==0 (a zero operand row), require an exact match. Tracks the worst rel = |LHS-RHS|/tau. */
static void bilin_one(const float* LHS, const float* RHS, const float* S, size_t n,
                      int L, float u, float c_safe, const char* nm, long* bad, float* maxrel) {
    for (size_t k = 0; k < n; k++) {
        float tau = c_safe * (float)L * u * S[k];
        float diff = LHS[k] - RHS[k]; if (diff < 0) diff = -diff;
        float rel = (tau > 0.0f) ? diff / tau : (diff == 0.0f ? 0.0f : 1.0e30f);
        if (rel > *maxrel) *maxrel = rel;
        if (!(diff <= tau)) { if (*bad < 4) fprintf(stderr, "bilin %s [%zu] |LHS-RHS|=%g tau=%g\n", nm, k, diff, tau); (*bad)++; }
    }
}

/* v1.5 #3 Phase 1 (verifiable PTX->SASS, the foundation): a from-scratch sm_86 SASS DECODER for the
 * straight-line vector_add opcode subset, derived empirically (probe + diff vs ptxas, NOT from NVIDIA
 * docs) and validated to reproduce cuobjdump/nvdisasm instruction-for-instruction. It reads the cubin's
 * .text.<kernel> bytes (sass_elf_find_text -- a from-scratch ELF64 walk, no libelf) and decodes each
 * 128-bit bundle with NO NVIDIA library, so the disassembly no longer takes cuobjdump's word for what the
 * bytes mean. HONEST SCOPE: Phase 1 de-trusts the DISASSEMBLER and lets us SEE the emitted machine code;
 * it does NOT yet de-trust ptxas (that is Phase 3's SASS->spec translation-validation). Opcode dispatch
 * uses lo&0xffff (the unpredicated PT subset; predicated forms fall to .UNKNOWN -> FAIL-CLOSED, out of
 * Phase-1 scope). Empirically-confirmed fields: Rd=lo[16:23], Ra=lo[24:31], Rb=lo[32:39] (or hi[0:7] when
 * a c[][] occupies the lo word), c[][] byte offset=lo[40:53]<<2, S2R sel=hi[8:15], IMAD.WIDE writes the
 * pair Rd:Rd+1, BRA target=addr+16+(i32)lo[32:63], .reuse=hi[58:59]. */
static unsigned long long sass_bits(unsigned long long v, int lo, int hi) {
    return (v >> lo) & ((hi - lo == 63) ? ~0ULL : ((1ULL << (hi - lo + 1)) - 1));
}
static const char* sass_sr_name(unsigned sel) {
    switch (sel) { case 0x21: return "SR_TID.X"; case 0x22: return "SR_TID.Y"; case 0x23: return "SR_TID.Z";
        case 0x25: return "SR_CTAID.X"; case 0x26: return "SR_CTAID.Y"; case 0x27: return "SR_CTAID.Z";
        case 0x00: return "SR_LANEID"; default: return "SR_?"; }
}
/* decode one 128-bit bundle -> a cuobjdump-format mnemonic string. Returns 0 for a known opcode, -1 if
 * the opcode is outside the whitelist (FAIL-CLOSED -- the unknown-instruction / corruption signal). */
static int sass_disasm(unsigned long long lo, unsigned long long hi, int addr, char* out) {
    unsigned op = (unsigned)(lo & 0xffff);
    int pred = (int)sass_bits(lo, 12, 15); char pfx[16] = "";
    if (pred != 7) { int neg = (pred >> 3) & 1; int pi = pred & 7; sprintf(pfx, "@%sP%d ", neg ? "!" : "", pi); }
    int Rd = (int)sass_bits(lo, 16, 23), Ra = (int)sass_bits(lo, 24, 31);
    unsigned coff = (unsigned)sass_bits(lo, 40, 53) * 4;
    const char* su = (sass_bits(hi, 8, 15) == 0) ? ".U32" : "";  /* IMAD-family signedness: hi[8:15]=0 -> .U32, 0x02 -> signed (suffix-free), matching cuobjdump */
    switch (op) {
        case 0x7a02: sprintf(out, "%sMOV R%d, c[0x0][0x%x]", pfx, Rd, coff); break;
        case 0x7802: sprintf(out, "%sMOV R%d, 0x%x", pfx, Rd, (unsigned)sass_bits(lo, 32, 63)); break;
        case 0x7919: sprintf(out, "%sS2R R%d, %s", pfx, Rd, sass_sr_name((unsigned)sass_bits(hi, 8, 15))); break;
        case 0x7ab9: sprintf(out, "%sULDC.64 UR%d, c[0x0][0x%x]", pfx, Rd, coff); break;
        case 0x7a24: sprintf(out, "%sIMAD%s R%d, R%d, c[0x0][0x%x], R%d", pfx, su, Rd, Ra, coff, (int)sass_bits(hi, 0, 7)); break;
        case 0x7224: sprintf(out, "%sIMAD%s R%d, R%d, R%d, R%d", pfx, su, Rd, Ra, (int)sass_bits(lo, 32, 39), (int)sass_bits(hi, 0, 7)); break;
        case 0x7625: { int reuse = (int)sass_bits(hi, 58, 59); char r1[8] = "", r2[8] = ""; if (reuse & 1) strcpy(r1, ".reuse"); if (reuse & 2) strcpy(r2, ".reuse");
            sprintf(out, "%sIMAD.WIDE%s R%d, R%d%s, R%d%s, c[0x0][0x%x]", pfx, su, Rd, Ra, r1, (int)sass_bits(hi, 0, 7), r2, coff); } break;
        case 0x7981: sprintf(out, "%sLDG.E R%d, [R%d.64]", pfx, Rd, Ra); break;
        case 0x7221: { /* FADD: decode the FULL modifier set (empirically pinned, sm_86/12.8) so a modified */
            int nra = (int)sass_bits(hi, 8, 8), ara = (int)sass_bits(hi, 9, 9);   /* FADD never disassembles as plain. */
            int nrb = (int)sass_bits(lo, 63, 63), arb = (int)sass_bits(lo, 62, 62); /* Ra mods in hi[8:9]; Rb mods in lo[62:63]. */
            int sat = (int)sass_bits(hi, 13, 13), ftz = (int)sass_bits(hi, 16, 16); int rmv = (int)sass_bits(hi, 14, 15);
            const char* rs = (rmv == 1) ? ".RM" : (rmv == 2) ? ".RP" : (rmv == 3) ? ".RZ" : ""; int Rbb = (int)sass_bits(lo, 32, 39);
            char oa[24], ob[24];
            if (ara) sprintf(oa, "|R%d|", Ra); else sprintf(oa, "%sR%d", nra ? "-" : "", Ra);
            if (arb) sprintf(ob, "|R%d|", Rbb); else sprintf(ob, "%sR%d", nrb ? "-" : "", Rbb);
            sprintf(out, "%sFADD%s%s%s R%d, %s, %s", pfx, rs, ftz ? ".FTZ" : "", sat ? ".SAT" : "", Rd, oa, ob); } break;
        case 0x7986: sprintf(out, "%sSTG.E [R%d.64], R%d", pfx, Ra, (int)sass_bits(lo, 32, 39)); break;
        case 0x794d: sprintf(out, "%sEXIT", pfx); break;
        case 0x7947: { long long rel = (int)sass_bits(lo, 32, 63); sprintf(out, "%sBRA 0x%x", pfx, (unsigned)(addr + 16 + (int)rel)); } break;
        case 0x7918: sprintf(out, "%sNOP", pfx); break;
        default: sprintf(out, "%s.UNKNOWN op=0x%04x", pfx, op); return -1;
    }
    return 0;
}
/* from-scratch ELF64 section-header walk -> {file offset, size} of the section named ".text.<kname>".
 * No libelf. Returns 0 on success; fail-closed (nonzero) on bad magic / not-ELF64 / section absent. */
static int sass_elf_find_text(const unsigned char* b, size_t sz, const char* kname, size_t* off, size_t* len) {
    if (sz < 64 || b[0] != 0x7f || b[1] != 'E' || b[2] != 'L' || b[3] != 'F' || b[4] != 2 /*ELFCLASS64*/) return 1;
    unsigned long long e_shoff = 0; memcpy(&e_shoff, b + 0x28, 8);
    unsigned short e_shentsize = 0, e_shnum = 0, e_shstrndx = 0;
    memcpy(&e_shentsize, b + 0x3a, 2); memcpy(&e_shnum, b + 0x3c, 2); memcpy(&e_shstrndx, b + 0x3e, 2);
    if (e_shoff == 0 || e_shentsize < 64 || e_shnum == 0 || e_shstrndx >= e_shnum) return 2;
    if (e_shoff + (size_t)e_shnum * e_shentsize > sz) return 3;
    const unsigned char* sh_str = b + e_shoff + (size_t)e_shstrndx * e_shentsize;   /* the .shstrtab section header */
    unsigned long long str_off = 0, str_sz = 0; memcpy(&str_off, sh_str + 0x18, 8); memcpy(&str_sz, sh_str + 0x20, 8);
    if (str_off + str_sz > sz) return 4;
    char want[128]; snprintf(want, sizeof want, ".text.%s", kname);
    for (unsigned i = 0; i < e_shnum; i++) {
        const unsigned char* sh = b + e_shoff + (size_t)i * e_shentsize;
        unsigned int sh_name = 0; memcpy(&sh_name, sh + 0x00, 4);
        if (str_off + sh_name >= sz) continue;
        const char* nm = (const char*)(b + str_off + sh_name);
        if (strcmp(nm, want) == 0) {
            unsigned long long so = 0, sl = 0; memcpy(&so, sh + 0x18, 8); memcpy(&sl, sh + 0x20, 8);
            if (so + sl > sz) return 5;
            *off = (size_t)so; *len = (size_t)sl; return 0;
        }
    }
    return 6;
}

/* v1.5 #3 Phase 2: a from-scratch sm_86 SASS INTERPRETER for the vector_add subset -- it EXECUTES the
 * decoded SASS on the CPU (modeling the register file, uniform registers, the constant bank, the special
 * registers, and a flat global memory), so we can run ptxas's emitted machine code OURSELVES. The
 * per-opcode semantics are VALIDATED against real RTX-3070 execution of the SAME cubin (the sass_exec
 * mode), not assumed. HONEST: Phase 2 validates the interpreter vs hardware on PROBE inputs -- it does NOT
 * yet PROVE the SASS computes the spec for ALL inputs (that is Phase 3's translation-validation, which
 * lifts decode+interpret to the real ptxas de-trust). */
typedef struct { unsigned int R[256], UR[64]; unsigned int ctaid_x, tid_x, ntid_x; unsigned int cbank[1024]; unsigned char* gmem; unsigned int gmem_sz; } SassVM;
static unsigned long long sv_cbank64(SassVM* vm, unsigned b) { unsigned d = b / 4; return (unsigned long long)vm->cbank[d] | ((unsigned long long)vm->cbank[d + 1] << 32); }
static unsigned int sv_cbank32(SassVM* vm, unsigned b) { return vm->cbank[b / 4]; }
static unsigned long long sv_regpair(SassVM* vm, int r) { return (unsigned long long)vm->R[r] | ((unsigned long long)vm->R[r + 1] << 32); }
static void sv_set_regpair(SassVM* vm, int r, unsigned long long v) { vm->R[r] = (unsigned int)v; vm->R[r + 1] = (unsigned int)(v >> 32); }
/* execute one decoded bundle over the VM state; returns 1 to STOP (EXIT/BRA or an unmodeled op), 0 to go
 * on. nc_wrong_fadd!=0 deliberately MIS-models FADD (a-b instead of a+b) -- the load-bearing NC: a wrong
 * interpreter must DIVERGE from the GPU, proving the interp==GPU validation has teeth. */
static int sass_exec1(SassVM* vm, unsigned long long lo, unsigned long long hi, int nc_wrong_fadd) {
    unsigned op = (unsigned)(lo & 0xffff);
    int Rd = (int)sass_bits(lo, 16, 23), Ra = (int)sass_bits(lo, 24, 31);
    unsigned coff = (unsigned)sass_bits(lo, 40, 53) * 4;
    switch (op) {
        case 0x7a02: vm->R[Rd] = sv_cbank32(vm, coff); break;                                  /* MOV Rd, c[][] */
        case 0x7802: vm->R[Rd] = (unsigned)sass_bits(lo, 32, 63); break;                        /* MOV Rd, imm */
        case 0x7919: { unsigned sel = (unsigned)sass_bits(hi, 8, 15); vm->R[Rd] = (sel == 0x25) ? vm->ctaid_x : (sel == 0x21) ? vm->tid_x : 0; } break; /* S2R */
        case 0x7ab9: { unsigned long long v = sv_cbank64(vm, coff); vm->UR[Rd] = (unsigned)v; vm->UR[Rd + 1] = (unsigned)(v >> 32); } break; /* ULDC.64 */
        case 0x7a24: { unsigned a = vm->R[Ra], b = sv_cbank32(vm, coff), c = vm->R[(int)sass_bits(hi, 0, 7)]; vm->R[Rd] = a * b + c; } break; /* IMAD Rd,Ra,c[][],Rc */
        case 0x7224: { unsigned a = vm->R[Ra], b = vm->R[(int)sass_bits(lo, 32, 39)], c = vm->R[(int)sass_bits(hi, 0, 7)]; vm->R[Rd] = a * b + c; } break; /* IMAD all-reg */
        case 0x7625: { int sgn = (sass_bits(hi, 8, 15) == 0x02); int a = (int)vm->R[Ra], b = (int)vm->R[(int)sass_bits(hi, 0, 7)];
            unsigned long long base = sv_cbank64(vm, coff);
            long long prod = sgn ? (long long)a * (long long)b : (long long)((unsigned long long)(unsigned)a * (unsigned long long)(unsigned)b);
            sv_set_regpair(vm, Rd, (unsigned long long)((long long)base + prod)); } break;       /* IMAD.WIDE -> 64-bit pair (address calc) */
        case 0x7981: { unsigned long long ad = sv_regpair(vm, Ra); if (vm->gmem_sz && ad + 4 > vm->gmem_sz) return 1; /* OOB -> STOP (no wild read; a data-dep gather is already rejected by LEG1) */ unsigned v; memcpy(&v, vm->gmem + ad, 4); vm->R[Rd] = v; } break; /* LDG.E [Rptr.64] */
        case 0x7221: { float fa, fb; unsigned ra = vm->R[Ra], rb = vm->R[(int)sass_bits(lo, 32, 39)]; memcpy(&fa, &ra, 4); memcpy(&fb, &rb, 4); if (sass_bits(lo, 63, 63)) fb = -fb; /* honor negate-Rb (sub.f32) */ float r = nc_wrong_fadd ? (fa - fb) : (fa + fb); memcpy(&vm->R[Rd], &r, 4); } break; /* FADD (nc: a-b) */
        case 0x7986: { unsigned long long ad = sv_regpair(vm, Ra); if (vm->gmem_sz && ad + 4 > vm->gmem_sz) return 1; /* OOB -> STOP (no wild write) */ unsigned v = vm->R[(int)sass_bits(lo, 32, 39)]; memcpy(vm->gmem + ad, &v, 4); } break; /* STG.E */
        case 0x794d: return 1;                                                                  /* EXIT */
        case 0x7947: return 1;                                                                  /* BRA (self-loop trap) */
        case 0x7918: break;                                                                     /* NOP */
        default: return 1;   /* unmodeled op -> stop; sass_exec then sees the interpreter DIVERGE from the GPU */
    }
    return 0;
}

/* v1.5 #3 Phase 3 LEG1 (SASS-level cflow): a fail-closed data-INDEPENDENCE taint that ALSO ENFORCES the
 * straight-line assumption it relies on. Single in-order pass. Seed: every LDG.E destination is a loaded DATA
 * value; propagate def->use. Return -1 FAIL-CLOSED if: a tainted reg reaches an LDG/STG ADDRESS; ANY non-PT
 * predicated instruction; ANY unmodeled opcode (it would halt the interpreter and could hide a clobber); or
 * ANY BRA that is not ptxas's benign self-loop trap pad (rel==-16) -- so no forward/backward branch can
 * re-route execution (closes the audit-3 forward-BRA-over-EXIT P0). Return 1 => a fixed per-thread truly
 * straight-line function of the loads, which licenses lifting the symbolic check to ALL inputs. NOT
 * cflow_witness (that is over the PTX = ptxas's INPUT; this is over ptxas's OUTPUT SASS -- the non-circular part). */
/* the opcodes the TV fully models. Anything else is fail-closed: an unmodeled op halts the interpreter
 * (sass_exec1 default returns 1) just like a branch, which could hide a post-latch clobber store from LEG3. */
static int sass_known_op(unsigned op) {
    switch (op) {
        case 0x7a02: case 0x7802: case 0x7919: case 0x7ab9: case 0x7a24: case 0x7224:
        case 0x7625: case 0x7981: case 0x7221: case 0x7986: case 0x794d: case 0x7947: case 0x7918:
            return 1;
        default: return 0;
    }
}

static int sass_taint_indep(const unsigned char* cb, size_t toff, int ninst) {
    unsigned char taint[256]; memset(taint, 0, sizeof taint);
    for (int i = 0; i < ninst; i++) {
        unsigned long long lo = 0, hi = 0;
        memcpy(&lo, cb + toff + (size_t)i * 16, 8); memcpy(&hi, cb + toff + (size_t)i * 16 + 8, 8);
        unsigned op = (unsigned)(lo & 0xffff);
        int pred = (int)sass_bits(lo, 12, 15);
        int Rd = (int)sass_bits(lo, 16, 23), Ra = (int)sass_bits(lo, 24, 31), Rb = (int)sass_bits(lo, 32, 39), Rc = (int)sass_bits(hi, 0, 7);
        if (pred != 7) return -1;                                  /* any non-PT predication -> out of the data-independent straight-line scope */
        if (!sass_known_op(op)) return -1;                         /* fail-closed on unmodeled ops (they halt the interpreter and can hide a clobber) */
        /* CONTROL-FLOW SOUNDNESS: the whole analysis assumes straight-line execution, so ENFORCE it. The only
         * benign branch is ptxas's self-loop trap pad (BRA to its own address: target=addr+16+rel==addr <=>
         * rel==-16), which sits AFTER EXIT and is never reached. ANY other BRA can re-route execution -- e.g. a
         * forward BRA over an EXIT to a clobber store, or a branch skipping the FADD -- which silently breaks the
         * straight-line assumption while the from-scratch interpreter merely STOPS at the branch. Reject it.
         * (Closes the audit-3 P0: decode-clean forward-BRA-over-EXIT double-store that computed c=a on silicon.) */
        if (op == 0x7947 /*BRA*/) { if ((int)(unsigned)(lo >> 32) != -16) return -1; continue; }
        if (op == 0x794d /*EXIT*/) continue;                       /* unconditional here (pred==7 enforced above) */
        if (op == 0x7981 /*LDG*/ || op == 0x7986 /*STG*/) {
            if (taint[Ra & 255] || taint[(Ra + 1) & 255]) return -1; /* the [Rptr.64] address must be gid-derived, never a loaded value */
        }
        if (op == 0x7981) { taint[Rd & 255] = 1; }                 /* a load defines a tainted (data) value */
        else if (op == 0x7986) { /* STG: no register dest */ }
        else {
            int src_t = taint[Ra & 255] || taint[Rb & 255] || taint[Rc & 255];
            taint[Rd & 255] = src_t ? 1 : 0;                       /* over-taint is SAFE (only more fail-closures); under-taint is not */
            if (op == 0x7625 /*IMAD.WIDE writes a pair*/) taint[(Rd + 1) & 255] = src_t ? 1 : 0;
        }
    }
    return 1;
}

/* v1.5 #3 Phase 3 LEG2 (symbolic structural equality): run the decoded SASS in program order tracking a
 * symbolic TAG per register (over OPAQUE load symbols, so the result holds for ALL f32 inputs). Return 1
 * iff the kernel performs EXACTLY ONE store and that store is a PLAIN FADD(LOAD_A, LOAD_B) to addr_c, with
 * the loads at base_a/base_b + gid*4, gid = ctaid*ntid + tid -- i.e. the SASS computes c[gid]=a[gid]+b[gid]
 * and nothing clobbers it (unique-store rule, not a monotone latch -- closes the audit-3 double-store P0). The
 * value/address-path opcodes are MODIFIER-GUARDED (fail-closed on an unmodeled set bit): FADD requires
 * lo[40:63]==0 (no Rb-side negate lo[63]/abs lo[62] -> sub.f32 REJECTED) AND hi[0:39]==0 (no Ra-side neg
 * hi[8]/abs hi[9], no SAT hi[13], no round hi[14:15], no FTZ hi[16] -- the FULL FADD operand/output modifier
 * region; scheduling bits start at hi[41]). The address-path opcodes are FIELD-COMPLETE: each pins its
 * UNMODELED hi[0:39] region to the genuine plain encoding (LDG hi[0:39]==0x0C1E1900, STG ==0x0C101904,
 * IMAD/IMAD.WIDE hi[8:39]==0x00078E02; scheduling hi[40:63] excluded) plus lo[54:63]==0 (IMAD) / zero
 * immediate offset lo[40:63]==0 (LDG/STG) -- so a memory-WIDTH (LDG.E.U8), +UR/cx[] addressing, or
 * signedness modifier cannot spoof the structure either. NO operand/output/width/address modifier (lo OR
 * hi word) on any value/address-path opcode can yield a TG_SUM. (sm_86/CUDA-12.8-pinned, like the rest of
 * the TV; a legit recompile with a different encoding fails CLOSED -- sound, not a hole.) */
enum { TG_UNK = 0, TG_CTAID, TG_TID, TG_C4, TG_GID, TG_ADDRA, TG_ADDRB, TG_ADDRC, TG_LOADA, TG_LOADB, TG_SUM };
static int sass_symbolic_addb(const unsigned char* cb, size_t toff, int ninst) {
    int tg[256]; for (int i = 0; i < 256; i++) tg[i] = TG_UNK;
    int n_stg = 0, c_sum_store = 0;     /* count ALL stores (must be exactly 1) + that the one store is SUM->c */
    for (int i = 0; i < ninst; i++) {
        unsigned long long lo = 0, hi = 0;
        memcpy(&lo, cb + toff + (size_t)i * 16, 8); memcpy(&hi, cb + toff + (size_t)i * 16 + 8, 8);
        unsigned op = (unsigned)(lo & 0xffff);
        int Rd = (int)sass_bits(lo, 16, 23), Ra = (int)sass_bits(lo, 24, 31), Rb = (int)sass_bits(lo, 32, 39), Rc = (int)sass_bits(hi, 0, 7);
        unsigned coff = (unsigned)sass_bits(lo, 40, 53) * 4;
        int himod0 = ((lo >> 54) == 0);                            /* no high lo-modifier (genuine IMAD/IMAD.WIDE have lo[54:63]==0) */
        int off0   = ((lo >> 40) == 0);                            /* zero immediate offset (genuine LDG/STG/plain-FADD) */
        /* FIELD-COMPLETE hi-word pins (sm_86/CUDA-12.8): the value/address-path opcodes carry width/cache/UR-
         * descriptor/cx[]/signedness control bits in hi[0:39] (scheduling is hi[40:63], excluded). Pin the
         * UNMODELED region to the genuine plain encoding so no such modifier -- the audit's LDG/STG-width +
         * IMAD.WIDE-cx[]/+UR sibling of the FADD hi-word hole -- can change semantics undetected. The register/
         * signedness fields I DO model (hi[0:7], IMAD.WIDE hi[8:15]) are excluded from the pinned mask. */
        int ldg_hi  = ((hi & 0xFFFFFFFFFFULL) == 0x0C1E1900ULL);    /* LDG.E 32-bit global, plain (both genuine loads agree) */
        int stg_hi  = ((hi & 0xFFFFFFFFFFULL) == 0x0C101904ULL);    /* STG.E 32-bit global, plain */
        int imx_hi  = (((hi >> 8) & 0xFFFFFFFFULL) == 0x00078E02ULL); /* IMAD + IMAD.WIDE signed, no cx/+UR/sgn-flip (hi[8:39]; both genuine agree) */
        switch (op) {
            case 0x7919: { unsigned sel = (unsigned)sass_bits(hi, 8, 15); tg[Rd] = (sel == 0x25) ? TG_CTAID : (sel == 0x21) ? TG_TID : TG_UNK; } break; /* S2R */
            case 0x7802: tg[Rd] = (sass_bits(lo, 32, 63) == 4) ? TG_C4 : TG_UNK; break;   /* MOV imm 4 -> the f32 stride */
            case 0x7a24: tg[Rd] = (tg[Ra] == TG_CTAID && coff == 0x0 && tg[Rc] == TG_TID && himod0 && imx_hi) ? TG_GID : TG_UNK; break; /* IMAD gid=ctaid*ntid+tid */
            case 0x7625: tg[Rd] = (tg[Ra] == TG_GID && tg[Rc] == TG_C4 && himod0 && imx_hi) ?         /* IMAD.WIDE addr=ptr+gid*4; multiplier reg is hi[0:7] (=Rc), matching sass_exec1 */
                ((coff == 0x160) ? TG_ADDRA : (coff == 0x168) ? TG_ADDRB : (coff == 0x170) ? TG_ADDRC : TG_UNK) : TG_UNK;
                tg[(Rd + 1) & 255] = TG_UNK; break;   /* IMAD.WIDE writes the PAIR Rd:Rd+1 -- the high half carries no value tag; clear it so a stale tag cannot survive a pair-high clobber (audit-5 P2: LEG2 tag-invalidating invariant made literally true; the genuine high-halves R3/R5/R7 are untagged here) */
            case 0x7981: tg[Rd] = (off0 && ldg_hi && tg[Ra] == TG_ADDRA) ? TG_LOADA : (off0 && ldg_hi && tg[Ra] == TG_ADDRB) ? TG_LOADB : TG_UNK; break; /* LDG.E 32-bit, plain */
            case 0x7221: { int ab = (tg[Ra] == TG_LOADA && tg[Rb] == TG_LOADB) || (tg[Ra] == TG_LOADB && tg[Rb] == TG_LOADA);
                /* PLAIN = NO operand/output modifier on EITHER side. Rb-side mods (neg lo[63], abs lo[62]) are
                 * caught by off0 (lo[40:63]==0). Ra-side + output mods (neg-Ra hi[8], abs-Ra hi[9], SAT hi[13],
                 * round hi[14:15], FTZ hi[16]) ALL live in the FADD hi word; the genuine plain FADD has hi[0:39]==0
                 * (scheduling bits start at hi[41]), so require it -- FAIL-CLOSED on the ENTIRE hi modifier region,
                 * known or not (closes the audit's P0: a faithfully-lowered FADD.SAT/-Ra/.RZ/.FTZ no longer certifies). */
                int plain = off0 && ((hi & 0x000000FFFFFFFFFFULL) == 0);
                tg[Rd] = (plain && ab) ? TG_SUM : TG_UNK; } break;
            case 0x7986: n_stg++; if (off0 && stg_hi && tg[Ra] == TG_ADDRC && tg[Rb] == TG_SUM) c_sum_store++; break; /* STG.E 32-bit, plain: c[gid]=SUM */
            /* TAG-INVALIDATION (audit-4 stale-tag P0): any opcode LEG2 does not STRUCTURALLY model still WRITES a
             * register on the GPU -- so it must INVALIDATE its destination tag(s), else a stale TG_LOADA/TG_SUM/
             * TG_ADDRC survives an overwrite and a GPU-wrong kernel certifies (e.g. IMAD-all-reg 0x7224 or MOV-c
             * 0x7a02 overwriting a load reg). Clear tg[Rd] AND tg[Rd+1] (an unmodeled op may write a pair),
             * mirroring LEG1's over-taint -- fail-closed. The genuine vector_add's default-case ops (MOV-c R1,
             * ULDC R4, EXIT, BRA, NOP) write regs that are untagged at that point, so the genuine still PASSES. */
            default: tg[Rd] = TG_UNK; tg[(Rd + 1) & 255] = TG_UNK; break;
        }
    }
    /* SOUND against the audit-3 latch P0: require EXACTLY ONE store in the whole kernel, and that store is the
     * plain SUM to addr_c. A monotone "saw a sum store" latch is NOT enough -- a second (clobber) store to c
     * could overwrite it on silicon while the latch stayed set. vector_add stores c[gid] exactly once; any
     * extra store (a write-after-write clobber, or a store elsewhere) => not vector_add => reject. */
    return (n_stg == 1 && c_sum_store == 1) ? 1 : 0;
}

/* v1.5 #3 Phase 3 LEG3 helper: run the from-scratch interpreter (CPU only, NO GPU) on chosen inputs and
 * fill c_out -- reuses sass_exec1, whose per-opcode semantics are Phase-2 GPU-validated. Models global
 * memory a@0, b@256, c@512 exactly as the sass_exec mode does. Used for the basis/linearity cross-check. */
static void sass_cpu_run(const unsigned char* cb, size_t toff, int ninst, const float* ha, const float* hb, int N, int ntid, int nblk, float* c_out) {
    static unsigned char G[4096]; memset(G, 0, sizeof G);
    unsigned ab = 0, bb = 256, cbo = 512;
    memcpy(G + ab, ha, (size_t)N * 4); memcpy(G + bb, hb, (size_t)N * 4);
    for (int blk = 0; blk < nblk; blk++) for (int t = 0; t < ntid; t++) {
        SassVM vm; memset(&vm, 0, sizeof vm); vm.gmem = G; vm.gmem_sz = (unsigned)sizeof G; vm.ctaid_x = (unsigned)blk; vm.tid_x = (unsigned)t; vm.ntid_x = (unsigned)ntid;
        vm.cbank[0x0 / 4] = (unsigned)ntid;
        vm.cbank[0x160 / 4] = ab; vm.cbank[0x168 / 4] = bb; vm.cbank[0x170 / 4] = cbo;
        for (int i = 0; i < ninst; i++) {
            unsigned long long lo = 0, hi = 0;
            memcpy(&lo, cb + toff + (size_t)i * 16, 8); memcpy(&hi, cb + toff + (size_t)i * 16 + 8, 8);
            if (sass_exec1(&vm, lo, hi, 0)) break;
        }
    }
    memcpy(c_out, G + cbo, (size_t)N * 4);
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <module.ptx> <kernel_name> [N] [op:add|mul|sub|reverse]\n", argv[0]);
        return 2;
    }
    const char* ptx_path = argv[1];
    const char* kname    = argv[2];
    int N = (argc > 3) ? atoi(argv[3]) : 256;
    if (N <= 0) N = 256;
    /* op selects the CPU reference the GPU result is checked against, so this one
     * launcher verifies a growing kernel corpus (vector_add/mul/sub) over the same
     * f32 inputs a[i]=i, b[i]=2*i. Default add (back-compat with the first-light run). */
    const char* op = (argc > 4) ? argv[4] : "add";
    int is_exp = (strcmp(op, "exp") == 0);    /* c[i]=e^a[i] via __gpu_exp; small inputs, tol-checked */
    int is_relu = (strcmp(op, "relu") == 0);  /* c[i]=max(a[i],0); negative inputs exercise the float compare */

    /* v1.5 S1 audit follow-up: codec_selftest -- verify the from-scratch IEEE binary16 codec
     * (f16_to_f32 / f32_to_f16 above) across normal / zero / subnormal / boundary values. Guards
     * the subnormal-decode off-by-one fixed above (it halved every f16 subnormal; out-of-path for
     * S1's data but S2/MXFP4 dequant produces small magnitudes). No GPU/PTX needed -- returns before
     * the module load. Usage: cuda_launch <anyptx> x 0 codec_selftest. Each case: f16 bits decode to
     * the IEEE value within a tight relative epsilon AND the value round-trips f32->f16->f32 exactly. */
    if (strcmp(op, "codec_selftest") == 0) {
        struct { unsigned short h; float v; } CASES[] = {
            { 0x0000, 0.0f }, { 0x3C00, 1.0f }, { 0xC000, -2.0f }, { 0x3800, 0.5f },
            { 0x0001, 5.9604644775390625e-08f },  /* smallest subnormal = 2^-24 */
            { 0x03FF, 6.097555160522461e-05f },   /* largest subnormal = 1023*2^-24 */
            { 0x0400, 6.103515625e-05f },         /* smallest normal = 2^-14 */
            { 0x7BFF, 65504.0f }                  /* max finite normal */
        };
        int ncase = (int)(sizeof(CASES) / sizeof(CASES[0]));
        int cbad = 0;
        for (int i = 0; i < ncase; i++) {
            float got = f16_to_f32(CASES[i].h);
            float want = CASES[i].v;
            float e = got - want; if (e < 0) e = -e;
            float den = want < 0 ? -want : want;
            float rel = den > 0.0f ? e / den : e;
            int decode_ok = (e == 0.0f) || (rel <= 1.0e-6f);
            unsigned short rt = f32_to_f16(want);        /* every CASES value is f16-exact -> round-trip must be identity */
            float back = f16_to_f32(rt);
            int rt_ok = (back == want);
            if (!decode_ok || !rt_ok) {
                fprintf(stderr, "codec_selftest FAIL case %d h=0x%04X got=%g want=%g rel=%g rt=0x%04X back=%g\n",
                        i, CASES[i].h, got, want, rel, rt, back);
                cbad++;
            }
        }
        printf("codec_selftest: %d cases, %d bad -> %s\n", ncase, cbad, cbad ? "FAIL" : "PASS");
        return cbad ? 1 : 0;
    }

    /* v1.5 S2 (MXFP4): mxfp4_codec_selftest -- verify the from-scratch E2M1+E8M0 codec across ALL 16
     * E2M1 codes (incl the subnormal 0.5 + both signs + zero) vs a canonical table, a round-trip, and
     * a few E8M0 scales. NEVER hand-verify the decode -- this catches an off-by-one (the S1 subnormal
     * bug class). GPU-free. Usage: cuda_launch <anyptx> x 0 mxfp4_codec_selftest. */
    if (strcmp(op, "mxfp4_codec_selftest") == 0) {
        float WANT[16] = {  0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                           -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f };
        int mbad = 0;
        for (int c = 0; c < 16; c++) {
            float got = e2m1_decode(c);
            if (got != WANT[c]) { fprintf(stderr, "mxfp4 e2m1 decode FAIL code %d got %g want %g\n", c, got, WANT[c]); mbad++; }
            int rt = e2m1_encode(WANT[c]);                 /* round-trip: re-encode must decode to the same value */
            if (e2m1_decode(rt) != WANT[c]) { fprintf(stderr, "mxfp4 e2m1 round-trip FAIL code %d -> %d (%g vs %g)\n", c, rt, e2m1_decode(rt), WANT[c]); mbad++; }
        }
        int SE[5] = { 127, 128, 126, 130, 123 }; float SV[5] = { 1.0f, 2.0f, 0.5f, 8.0f, 0.0625f };
        for (int i = 0; i < 5; i++) {
            float got = e8m0_scale(SE[i]); float e = got - SV[i]; if (e < 0) e = -e;
            if (e > 1.0e-9f * SV[i]) { fprintf(stderr, "mxfp4 e8m0 FAIL e=%d got %g want %g\n", SE[i], got, SV[i]); mbad++; }
        }
        printf("mxfp4_codec_selftest: 16 E2M1 codes + 5 E8M0 scales, %d bad -> %s\n", mbad, mbad ? "FAIL" : "PASS");
        return mbad ? 1 : 0;
    }

    /* v1.5 S3 (NVFP4): nvfp4_codec_selftest -- verify the E2M1 (reused) + FP8 E4M3 codec. The E4M3
     * boundary set covers 0, the min subnormal (2^-9), the min normal (2^-6), 1.0 (e=7), a mid value,
     * the MAX FINITE (448 = e15/m6, NOT Inf), and confirms the NaN sentinel (0x7F) decodes to NaN.
     * NEVER hand-verify the decode -- this catches an off-by-one (the S1/S2 subnormal-bug class).
     * GPU-free. Usage: cuda_launch <anyptx> x 0 nvfp4_codec_selftest. */
    if (strcmp(op, "nvfp4_codec_selftest") == 0) {
        float E2[16] = {  0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                         -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f };
        int nbad = 0;
        for (int c = 0; c < 16; c++) if (e2m1_decode(c) != E2[c]) { fprintf(stderr, "nvfp4 e2m1 FAIL code %d got %g want %g\n", c, e2m1_decode(c), E2[c]); nbad++; }
        /* (E4M3 code, expected value): 0x00->0; 0x01(e0,m1)->2^-9; 0x08(e1,m0)->2^-6; 0x38(e7,m0)->1.0;
         * 0x7E(e15,m6)->1.75*256=448 (FINITE, not Inf); 0x40(e8,m0)->2.0. */
        int   EC[6] = { 0x00,        0x01,          0x08,        0x38, 0x7E,   0x40 };
        float EV[6] = { 0.0f, 0.001953125f, 0.015625f, 1.0f, 448.0f, 2.0f };
        for (int i = 0; i < 6; i++) {
            float got = e4m3_decode(EC[i]); float e = got - EV[i]; if (e < 0) e = -e;
            if (e > 1.0e-9f * (EV[i] > 0 ? EV[i] : 1.0f)) { fprintf(stderr, "nvfp4 e4m3 FAIL code 0x%02X got %g want %g\n", EC[i], got, EV[i]); nbad++; }
        }
        float nv = e4m3_decode(0x7F);                         /* the NaN sentinel MUST decode to NaN */
        if (!(nv != nv)) { fprintf(stderr, "nvfp4 e4m3 NaN-sentinel FAIL: 0x7F -> %g (expected NaN)\n", nv); nbad++; }
        printf("nvfp4_codec_selftest: 16 E2M1 + 6 E4M3 + NaN sentinel, %d bad -> %s\n", nbad, nbad ? "FAIL" : "PASS");
        return nbad ? 1 : 0;
    }

    /* v1.5 #4 STEP 1: sha256_selftest -- the from-scratch SHA-256 MUST match the 3 NIST FIPS-180-4 known
     * answers BEFORE any receipt binding is trusted (a hash bug silently voids every commitment). GPU-free;
     * returns before the module load. Usage: cuda_launch <anyptx> x 0 sha256_selftest. */
    if (strcmp(op, "sha256_selftest") == 0) {
        struct { const char* msg; const char* want; } KAT[3] = {
            { "", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855" },
            { "abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad" },
            { "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
              "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1" }
        };
        int sbad = 0;
        for (int i = 0; i < 3; i++) {
            unsigned char dig[32]; char hex[65];
            sha256((const unsigned char*)KAT[i].msg, strlen(KAT[i].msg), dig);
            sha256_hex(dig, hex);
            if (strcmp(hex, KAT[i].want) != 0) { fprintf(stderr, "sha256 KAT %d FAIL: got %s want %s\n", i, hex, KAT[i].want); sbad++; }
        }
        printf("sha256_selftest: 3 NIST KAT vectors, %d bad -> %s\n", sbad, sbad ? "FAIL" : "PASS");
        return sbad ? 1 : 0;
    }

    /* v1.5 #4 STEP 3: receipt_check -- the INDEPENDENT verifier (GPU-free, fail-closed, modeled on
     * cflow_witness). Given a receipt (+ its .cbytes side-file), ACCEPT iff three ANDed checks hold:
     *   CHECK1 commitment-bind: regenerate W,X from the receipt seeds, recompute SHA-256, == H_W,H_X.
     *   CHECK3 output-bind: SHA-256 of the C-bytes == H_C (so the verified C is exactly the committed one).
     *   CHECK2 FREIVALDS over F_p (exact integer): for each of t Fiat-Shamir rounds, r derived by THIS
     *     checker (never read from the receipt) and UNIFORM mod p (de-biased in receipt_fs_r), assert
     *     W*(X*r) == C*r mod p for all M rows. A RANGE GUARD (|C[i]| < p/2) makes mod-p equality <=>
     *     integer equality (genuine |C| <= K*max|W|*max|X| << p/2; any multiple-of-p forgery has an entry
     *     >= p-|C| > p/2 and is caught), so this is EXACT (no tolerance). Soundness: C != W*X => accept
     *     prob <= (1/p)^t, literally (uniform r); with p=2^31-1, t=2 that is <= 2^-62 + the SHA-256
     *     collision advantage.
     * Cost O(MK+KN+MN) << the O(MKN) matmul -> faster than re-execution; the checker NEVER runs the kernel.
     * SCOPE: NOT zero-knowledge (W,X,C are revealed to the checker), NOT a succinct SNARK (only Freivalds
     * is sublinear), matmul-only. ENDIANNESS: the commitments hash native int bytes, so emit + check must
     * run on the same-endianness host (cross-endian it fails CLOSED -- a spurious RECEIPT_FAIL, never an
     * accepted forgery). Usage: cuda_launch <anyptx-ignored> x 0 receipt_check <receiptpath>. Prints
     * '-> RECEIPT_PASS/RECEIPT_FAIL' + a REJECT=<which-check> cause. */
    if (strcmp(op, "receipt_check") == 0) {
        const char* rpath = (argc > 5) ? argv[5] : "/tmp/helix_receipt.txt";
        FILE* rf = fopen(rpath, "r");
        if (!rf) { printf("receipt_check [%s]: cannot open receipt -> RECEIPT_FAIL\n", rpath); return 1; }
        char ln[512]; int Md=0,Kd=0,Nd=0,t=0; long long p=0; unsigned uW=0,uX=0;
        char hW[96]={0},hX[96]={0},hC[96]={0}; int ok_hdr=0,haveW=0,haveX=0,haveC=0;
        while (fgets(ln, sizeof ln, rf)) {
            if (strncmp(ln,"HELIX_RECEIPT_V1",16)==0) ok_hdr=1;
            else if (strncmp(ln,"shape ",6)==0) sscanf(ln,"shape M=%d K=%d N=%d",&Md,&Kd,&Nd);
            else if (strncmp(ln,"prime ",6)==0) sscanf(ln,"prime p=%lld",&p);
            else if (strncmp(ln,"rounds ",7)==0) sscanf(ln,"rounds t=%d",&t);
            else if (strncmp(ln,"seed_W=",7)==0) sscanf(ln,"seed_W=%u",&uW);
            else if (strncmp(ln,"seed_X=",7)==0) sscanf(ln,"seed_X=%u",&uX);
            else if (strncmp(ln,"H_W=",4)==0) sscanf(ln,"H_W=%64s",hW), haveW=1;
            else if (strncmp(ln,"H_X=",4)==0) sscanf(ln,"H_X=%64s",hX), haveX=1;
            else if (strncmp(ln,"H_C=",4)==0) sscanf(ln,"H_C=%64s",hC), haveC=1;
            /* round lines are IGNORED -- the checker re-derives r and recomputes the projections itself */
        }
        fclose(rf);
        if (!ok_hdr || !haveW || !haveX || !haveC || Md<=0 || Kd<=0 || Nd<=0 || t<1 || p<2) {
            printf("receipt_check [%s]: malformed/vacuous receipt REJECT=VACUITY -> RECEIPT_FAIL\n", rpath); return 1;
        }
        size_t nW=(size_t)Md*Kd, nX=(size_t)Kd*Nd, nC=(size_t)Md*Nd;
        int* W=(int*)malloc(nW*sizeof(int)); int* X=(int*)malloc(nX*sizeof(int)); int* C=(int*)malloc(nC*sizeof(int));
        if (!W||!X||!C) { printf("receipt_check: oom -> RECEIPT_FAIL\n"); free(W);free(X);free(C); return 1; }
        receipt_gen_W(W,nW,uW); receipt_gen_X(X,nX,uX);
        int bad = 0; const char* rej = "NONE";   /* records WHICH check first rejected (per-check NC attribution) */
        unsigned char dW[32],dX[32],dC[32]; char xW[65],xX[65],xC[65];
        sha256((unsigned char*)W,nW*sizeof(int),dW); sha256_hex(dW,xW);
        sha256((unsigned char*)X,nX*sizeof(int),dX); sha256_hex(dX,xX);
        if (strcmp(xW,hW)!=0) { fprintf(stderr,"CHECK1 H_W mismatch\n"); bad=1; rej="CHECK1"; }
        if (strcmp(xX,hX)!=0) { fprintf(stderr,"CHECK1 H_X mismatch\n"); bad=1; rej="CHECK1"; }
        char cpath[600]; snprintf(cpath,sizeof cpath,"%s.cbytes",rpath);
        FILE* cf=fopen(cpath,"rb");
        if (!cf) { printf("receipt_check: cannot open C-bytes %s -> RECEIPT_FAIL\n",cpath); free(W);free(X);free(C); return 1; }
        size_t cgot=fread(C,1,nC*sizeof(int),cf); int extra=fgetc(cf)!=EOF; fclose(cf);
        if (cgot != nC*sizeof(int) || extra) { printf("receipt_check: C-bytes size mismatch -> RECEIPT_FAIL\n"); free(W);free(X);free(C); return 1; }
        sha256((unsigned char*)C,nC*sizeof(int),dC); sha256_hex(dC,xC);
        if (strcmp(xC,hC)!=0) { fprintf(stderr,"CHECK3 H_C mismatch\n"); bad=1; if (!strcmp(rej,"NONE")) rej="CHECK3"; }
        long long half = p/2;
        for (size_t i=0;i<nC && !bad;i++) if ((long long)C[i] >= half || (long long)C[i] <= -half) { fprintf(stderr,"CHECK2 range guard: |C[%zu]|=%d >= p/2\n",i,C[i]); bad=1; rej="CHECK2_RANGE"; }
        for (int rd=0; rd<t && !bad; rd++) {
            long long* r=(long long*)malloc((size_t)Nd*sizeof(long long));
            long long* Br=(long long*)malloc((size_t)Kd*sizeof(long long));
            if (!r||!Br){ printf("receipt_check: oom -> RECEIPT_FAIL\n"); free(r);free(Br);free(W);free(X);free(C); return 1; }
            for (int j=0;j<Nd;j++) r[j]=receipt_fs_r(dW,dX,dC,rd,j,p);
            for (int k=0;k<Kd;k++){ long long s=0; for(int j=0;j<Nd;j++){ long long xv=(((long long)X[(size_t)k*Nd+j])%p+p)%p; s=(s+xv*r[j])%p; } Br[k]=s; }
            for (int i=0;i<Md;i++){
                long long lhs=0; for(int k=0;k<Kd;k++){ long long wv=(((long long)W[(size_t)i*Kd+k])%p+p)%p; lhs=(lhs+wv*Br[k])%p; }
                long long rhs=0; for(int j=0;j<Nd;j++){ long long cv=(((long long)C[(size_t)i*Nd+j])%p+p)%p; rhs=(rhs+cv*r[j])%p; }
                if (lhs!=rhs){ fprintf(stderr,"CHECK2 Freivalds FAIL round %d row %d lhs=%lld rhs=%lld\n",rd,i,lhs,rhs); bad=1; rej="CHECK2"; break; }
            }
            free(r); free(Br);
        }
        printf("receipt_check [%s]: M=%d K=%d N=%d t=%d p=%lld REJECT=%s -> %s\n", rpath, Md,Kd,Nd,t,p, rej, bad?"RECEIPT_FAIL":"RECEIPT_PASS");
        free(W);free(X);free(C);
        return bad ? 1 : 0;
    }

    /* v1.5 #3 Phase 1: sass_check -- read the cubin's .text.<kname> and decode every 128-bit bundle from
     * scratch, printing cuobjdump-format lines + a final token. FAIL-CLOSED on an unknown opcode or a bad
     * ELF (the corruption signal). GPU-free; argv[1] is a CUBIN here (not PTX), so this returns before the
     * module load. Usage: cuda_launch <cubin> <kname> 0 sass_check. The gate (gpu_sass_check.sh) cross-
     * checks this output vs cuobjdump/nvdisasm (untrusted oracles) for the positive, and detects a
     * corrupted cubin by this from-scratch decode CHANGING vs the genuine, or fail-closing. */
    if (strcmp(op, "sass_check") == 0) {
        FILE* cf = fopen(ptx_path, "rb");
        if (!cf) { printf("sass_check [%s]: cannot open cubin -> SASS_DECODE_FAIL\n", ptx_path); return 1; }
        fseek(cf, 0, SEEK_END); long csz = ftell(cf); fseek(cf, 0, SEEK_SET);
        if (csz <= 0) { fclose(cf); printf("sass_check: empty cubin -> SASS_DECODE_FAIL\n"); return 1; }
        unsigned char* cb = (unsigned char*)malloc((size_t)csz);
        if (!cb) { fclose(cf); return 2; }
        if (fread(cb, 1, (size_t)csz, cf) != (size_t)csz) { fclose(cf); free(cb); printf("sass_check: short read -> SASS_DECODE_FAIL\n"); return 1; }
        fclose(cf);
        size_t toff = 0, tlen = 0;
        if (sass_elf_find_text(cb, (size_t)csz, kname, &toff, &tlen) != 0) {
            printf("sass_check [%s]: .text.%s not found / bad ELF -> SASS_DECODE_FAIL\n", ptx_path, kname); free(cb); return 1;
        }
        int nbundle = (int)(tlen / 16), unknown = 0, real = 0;
        for (int i = 0; i < nbundle; i++) {
            unsigned long long lo = 0, hi = 0;
            memcpy(&lo, cb + toff + (size_t)i * 16, 8); memcpy(&hi, cb + toff + (size_t)i * 16 + 8, 8);
            char buf[160];
            int r = sass_disasm(lo, hi, i * 16, buf);
            printf("/*%04x*/ %s\n", i * 16, buf);
            if (r != 0) unknown++;
            else if (strncmp(buf, "NOP", 3) != 0) real++;
        }
        int ok = (unknown == 0 && real >= 10);     /* vector_add = 15 real instrs; >=10 is the non-vacuity floor */
        printf("sass_check [%s]: %d bundles, %d real, %d unknown -> %s\n", ptx_path, nbundle, real, unknown,
               ok ? "SASS_DECODE_OK" : "SASS_DECODE_FAIL");
        free(cb);
        return ok ? 0 : 1;
    }

    /* v1.5 #3 Phase 2: sass_exec -- EXECUTE the cubin's decoded vector_add SASS on the CPU (from-scratch
     * interpreter) AND run the SAME cubin on the RTX 3070 (cuModuleLoadData of the cubin -> the driver runs
     * its SASS directly, no re-JIT, so the interpreter models EXACTLY what the GPU runs), then assert the
     * interpreter's c[] == the GPU's c[] element-for-element. This VALIDATES the interpreter semantics vs
     * HARDWARE on probe inputs (not assumed). Usage: cuda_launch <cubin> <kname> 0 sass_exec. argv[1] is a
     * CUBIN; returns before the normal PTX path. Prints '-> SASS_EXEC_PASS/SASS_EXEC_FAIL'. */
    if (strcmp(op, "sass_exec") == 0) {
        FILE* cf = fopen(ptx_path, "rb");
        if (!cf) { printf("sass_exec [%s]: cannot open cubin -> SASS_EXEC_FAIL\n", ptx_path); return 1; }
        fseek(cf, 0, SEEK_END); long csz = ftell(cf); fseek(cf, 0, SEEK_SET);
        if (csz <= 0) { fclose(cf); printf("sass_exec: empty cubin -> SASS_EXEC_FAIL\n"); return 1; }
        unsigned char* cb = (unsigned char*)malloc((size_t)csz);
        if (!cb) { fclose(cf); return 2; }
        if (fread(cb, 1, (size_t)csz, cf) != (size_t)csz) { fclose(cf); free(cb); return 2; }
        fclose(cf);
        size_t toff = 0, tlen = 0;
        if (sass_elf_find_text(cb, (size_t)csz, kname, &toff, &tlen) != 0) { printf("sass_exec: .text.%s not found -> SASS_EXEC_FAIL\n", kname); free(cb); return 1; }
        int ninst = (int)(tlen / 16);
        if (ninst < 1 || ninst > 4096) { printf("sass_exec: bad ninst %d -> SASS_EXEC_FAIL\n", ninst); free(cb); return 1; }
        int N = 8, ntid = 4, nblk = 2;
        int mutate = (argc > 5 && strcmp(argv[5], "mutate") == 0);   /* NC: a deliberately-wrong interpreter (FADD a-b) must diverge from the GPU */
        float ha[8], hb[8]; for (int i = 0; i < N; i++) { ha[i] = (float)(i + 1); hb[i] = 10.0f * (float)(i + 1); }
        /* --- INTERPRET the decoded SASS on the CPU (model global memory: a@0, b@256, c@512) --- */
        static unsigned char G[4096]; memset(G, 0, sizeof G);
        unsigned ab = 0, bb = 256, cbo = 512;
        memcpy(G + ab, ha, (size_t)N * 4); memcpy(G + bb, hb, (size_t)N * 4);
        for (int blk = 0; blk < nblk; blk++) for (int t = 0; t < ntid; t++) {
            SassVM vm; memset(&vm, 0, sizeof vm); vm.gmem = G; vm.gmem_sz = (unsigned)sizeof G; vm.ctaid_x = (unsigned)blk; vm.tid_x = (unsigned)t; vm.ntid_x = (unsigned)ntid;
            vm.cbank[0x0 / 4] = (unsigned)ntid;
            vm.cbank[0x160 / 4] = ab; vm.cbank[0x168 / 4] = bb; vm.cbank[0x170 / 4] = cbo;   /* a/b/c pointers (32-bit offsets; high halves stay 0) */
            for (int i = 0; i < ninst; i++) {
                unsigned long long lo = 0, hi = 0;
                memcpy(&lo, cb + toff + (size_t)i * 16, 8); memcpy(&hi, cb + toff + (size_t)i * 16 + 8, 8);
                if (sass_exec1(&vm, lo, hi, mutate)) break;
            }
        }
        float c_interp[8]; memcpy(c_interp, G + cbo, (size_t)N * 4);
        /* --- run the SAME cubin on the GPU (driver loads its SASS directly) --- */
        if (check(cuInit(0), "cuInit")) { free(cb); return 2; }
        CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
        if (check(cuDeviceGet(&dev, 0), "cuDeviceGet") || check(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate")) { free(cb); return 2; }
        if (check(cuModuleLoadData(&mod, cb), "cuModuleLoadData(cubin)") || check(cuModuleGetFunction(&fn, mod, kname), "getFunc")) { free(cb); return 2; }
        CUdeviceptr dA, dB, dC; int nn = N;
        if (check(cuMemAlloc(&dA, (size_t)N * 4), "A") || check(cuMemAlloc(&dB, (size_t)N * 4), "B") || check(cuMemAlloc(&dC, (size_t)N * 4), "C")) { free(cb); return 2; }
        cuMemcpyHtoD(dA, ha, (size_t)N * 4); cuMemcpyHtoD(dB, hb, (size_t)N * 4);
        void* args[] = { &dA, &dB, &dC, &nn };
        if (check(cuLaunchKernel(fn, (unsigned)nblk, 1, 1, (unsigned)ntid, 1, 1, 0, 0, args, 0), "launch") || check(cuCtxSynchronize(), "sync")) { free(cb); return 2; }
        float c_gpu[8]; cuMemcpyDtoH(c_gpu, dC, (size_t)N * 4);
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC); cuModuleUnload(mod); cuCtxDestroy(ctx);
        /* --- compare the from-scratch interpreter vs the GPU (the load-bearing semantics validation) --- */
        int bad = 0;
        for (int i = 0; i < N; i++) if (c_interp[i] != c_gpu[i]) { if (bad < 4) fprintf(stderr, "sass_exec mismatch [%d] interp=%g gpu=%g\n", i, c_interp[i], c_gpu[i]); bad++; }
        printf("sass_exec [%s]: %d instrs, N=%d (gpu c[0..2]=%g,%g,%g interp=%g,%g,%g) %d bad -> %s\n",
               ptx_path, ninst, N, c_gpu[0], c_gpu[1], c_gpu[2], c_interp[0], c_interp[1], c_interp[2], bad, bad ? "SASS_EXEC_FAIL" : "SASS_EXEC_PASS");
        free(cb);
        return bad ? 1 : 0;
    }

    /* v1.5 #3 Phase 3: sass_tv -- the SASS->spec TRANSLATION-VALIDATION (the REAL ptxas de-trust). PROVE the
     * cubin's emitted vector_add SASS computes c[gid]=a[gid]+b[gid] for ALL f32 inputs, from-scratch, WITHOUT
     * trusting ptxas's lowering (no ptxas/cuobjdump in the CPU proof). The DE-TRUST VERDICT is the COMPOSITE
     * sass_tv AND sass_exec -- BOTH are load-bearing (see LEG4); sass_tv ALONE is NOT sound:
     *   LEG1 sass_taint_indep  -- the SASS is a data-INDEPENDENT straight-line per-thread function of the loads;
     *                             ENFORCED, fail-closed on non-PT predication, any unmodeled opcode, and any BRA
     *                             that is not the self-loop pad (rel==-16) -- licenses lifting to ALL inputs.
     *   LEG2 sass_symbolic_addb-- symbolic structural equality over OPAQUE load symbols: the kernel does EXACTLY
     *                             ONE store and it is a PLAIN FADD(LOAD_A,LOAD_B) at base+gid*4 (holds for every
     *                             f32). MODIFIER-COMPLETE (FADD lo[40:63]==0 + hi[0:39]==0 reject all operand/
     *                             output modifiers; address-path opcodes pin their hi[0:39]) and TAG-INVALIDATING
     *                             (any opcode LEG2 does not model clears its dest tag, so a stale LOAD/SUM cannot
     *                             survive an overwrite -- audit-4 stale-tag P0).
     *   LEG3 basis + homogeneity-- a LOAD-BEARING execution cross-check via the from-scratch interpreter, gated on
     *                             LEG1&&LEG2: catches e.g. an early-EXIT that leaves c unwritten. (Does NOT catch
     *                             modifier kernels -- LEG2 does -- nor scheduling hazards -- LEG4 does.)
     *   LEG4 sass_exec (separate mode, REQUIRED) -- the GPU-DIFFERENTIAL: the CANDIDATE's actual RTX-3070 execution
     *                             must match the from-scratch interpreter. This discharges INSTRUCTION-SCHEDULING /
     *                             dependency-scoreboard correctness (control word hi[40:63], which this CPU proof
     *                             has NO model for: a cleared dependency-wait bit is decode-clean yet GPU-wrong --
     *                             audit-4 scoreboard P0). The de-trust requires sass_tv AND sass_exec; a pure-CPU
     *                             scoreboard model that would remove the GPU from the verifier is a labeled stretch.
     * Usage: cuda_launch <cubin> <kname> 0 sass_tv. argv[1] is a CUBIN; returns before the PTX path. */
    if (strcmp(op, "sass_tv") == 0) {
        FILE* cf = fopen(ptx_path, "rb");
        if (!cf) { printf("sass_tv [%s]: cannot open cubin -> SASS_TV_FAIL\n", ptx_path); return 1; }
        fseek(cf, 0, SEEK_END); long csz = ftell(cf); fseek(cf, 0, SEEK_SET);
        if (csz <= 0) { fclose(cf); printf("sass_tv: empty cubin -> SASS_TV_FAIL\n"); return 1; }
        unsigned char* cb = (unsigned char*)malloc((size_t)csz);
        if (!cb) { fclose(cf); return 2; }
        if (fread(cb, 1, (size_t)csz, cf) != (size_t)csz) { fclose(cf); free(cb); return 2; }
        fclose(cf);
        size_t toff = 0, tlen = 0;
        if (sass_elf_find_text(cb, (size_t)csz, kname, &toff, &tlen) != 0) { printf("sass_tv: .text.%s not found -> SASS_TV_FAIL\n", kname); free(cb); return 1; }
        int ninst = (int)(tlen / 16);
        if (ninst < 1 || ninst > 4096) { printf("sass_tv: bad ninst %d -> SASS_TV_FAIL\n", ninst); free(cb); return 1; }
        /* LEG1: data-independence (over ptxas's OUTPUT SASS) */
        int cflow = (sass_taint_indep(cb, toff, ninst) == 1);
        /* LEG2: symbolic structural equality c[gid] == plain FADD(LOAD_A, LOAD_B), all inputs */
        int symbolic = (sass_symbolic_addb(cb, toff, ninst) == 1);
        /* LEG3: a LOAD-BEARING execution cross-check via the from-scratch interpreter -- it is the leg that
         * catches e.g. an early-EXIT that leaves c unwritten (basis reads the final c). RUN ONLY when the
         * structural legs already passed, so the interpreter never executes a rejected / data-dependent kernel
         * (with the LDG/STG bounds-check, the OOB path is closed two ways). basis distinguishes add from
         * sub/mul/scale; homogeneity is a linearity tripwire. NOTE: LEG3 runs the from-scratch interpreter,
         * which has NO scheduling/scoreboard model -- so it does NOT catch instruction-scheduling hazards; the
         * separate sass_exec GPU-differential (a REQUIRED part of the de-trust verdict) is what discharges that. */
        int N = 8, ntid = 4, nblk = 2;
        float c[8], a[8], b[8], a2[8], b2[8], c1[8], c2[8];
        int i, basis = 0, homog = 0, addit = 0;
        if (cflow && symbolic) {
            basis = homog = addit = 1;
            for (i = 0; i < N; i++) { a[i] = 1.0f; b[i] = 0.0f; } sass_cpu_run(cb, toff, ninst, a, b, N, ntid, nblk, c); for (i = 0; i < N; i++) if (c[i] != 1.0f) basis = 0;   /* f(1,0)=1 */
            for (i = 0; i < N; i++) { a[i] = 0.0f; b[i] = 1.0f; } sass_cpu_run(cb, toff, ninst, a, b, N, ntid, nblk, c); for (i = 0; i < N; i++) if (c[i] != 1.0f) basis = 0;   /* f(0,1)=1 */
            for (i = 0; i < N; i++) { a[i] = 0.0f; b[i] = 0.0f; } sass_cpu_run(cb, toff, ninst, a, b, N, ntid, nblk, c); for (i = 0; i < N; i++) if (c[i] != 0.0f) basis = 0;   /* f(0,0)=0 */
            for (i = 0; i < N; i++) { a[i] = (float)(i + 1); b[i] = 10.0f * (float)(i + 1); } sass_cpu_run(cb, toff, ninst, a, b, N, ntid, nblk, c1);
            for (i = 0; i < N; i++) { a2[i] = 2.0f * a[i]; b2[i] = 2.0f * b[i]; } sass_cpu_run(cb, toff, ninst, a2, b2, N, ntid, nblk, c2);
            for (i = 0; i < N; i++) if (c2[i] != 2.0f * c1[i]) homog = 0;                                               /* f(2a,2b)=2 f(a,b) exact (interpreter-linearity tripwire) */
            for (i = 0; i < N; i++) { a[i] = 0.5f * (i + 1); a2[i] = 0.25f * (i + 1); b[i] = 3.0f * (i + 1); b2[i] = 7.0f * (i + 1); }
            { float A[8], B[8], cab[8], ca[8], cbv[8]; double u = ldexp(1.0, -24);
              for (i = 0; i < N; i++) { A[i] = a[i] + a2[i]; B[i] = b[i] + b2[i]; }
              sass_cpu_run(cb, toff, ninst, A, B, N, ntid, nblk, cab);
              sass_cpu_run(cb, toff, ninst, a, b, N, ntid, nblk, ca);
              sass_cpu_run(cb, toff, ninst, a2, b2, N, ntid, nblk, cbv);
              for (i = 0; i < N; i++) { double tau = 8.0 * u * ((double)fabsf(A[i]) + (double)fabsf(B[i])) + 1e-30;
                if ((double)fabsf(cab[i] - (ca[i] + cbv[i])) > tau) addit = 0; } }                                      /* f(a1+a2,b1+b2)=f(a1,b1)+f(a2,b2) within tau */
        }
        int laws = homog && addit;
        /* sass_tv (CPU) requires cflow (data-independence + straight-line) AND symbolic (modifier-complete +
         * unique-store + tag-invalidating structural equality) AND basis/laws (the load-bearing interpreter
         * cross-check, e.g. early-EXIT). sass_tv proves the VALUE semantics for all inputs but has NO scheduling
         * model -- the DE-TRUST verdict additionally requires the sass_exec GPU-differential (see the header). */
        int pass = cflow && symbolic && basis && laws;
        printf("sass_tv [%s]: %d instrs cflow=%d symbolic=%d basis=%d laws=%d(homog=%d addit=%d) -> %s\n",
               ptx_path, ninst, cflow, symbolic, basis, laws, homog, addit, pass ? "SASS_TV_PASS" : "SASS_TV_FAIL");
        free(cb);
        return pass ? 0 : 1;
    }

    /* slurp the PTX text (NUL-terminated; cuModuleLoadData wants a C string) */
    FILE* f = fopen(ptx_path, "rb");
    if (!f) { fprintf(stderr, "cannot open ptx: %s\n", ptx_path); return 2; }
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz < 0) { fclose(f); return 2; }
    char* ptx = (char*)malloc((size_t)sz + 1);
    if (!ptx) { fclose(f); return 2; }
    if (fread(ptx, 1, (size_t)sz, f) != (size_t)sz) { fclose(f); free(ptx); return 2; }
    ptx[sz] = 0; fclose(f);

    /* v1.5 #2 LEG 1: cflow mode -- data-independent-control-flow witness on the loaded PTX (GPU-free,
     * returns before cuInit). Usage: cuda_launch <ptx> <kname-ignored> 0 cflow. PASS => the kernel is a
     * fixed straight-line program (data-independent control flow + selection); FAIL => a setp/branch
     * consumes a loaded value (data-dependent). */
    if (strcmp(op, "cflow") == 0) {
        int r = cflow_witness(ptx);
        const char* v = (r == 1) ? "DATA-INDEPENDENT control flow + selection + addressing (fixed straight-line dataflow)"
                      : (r == 0) ? "DATA-DEPENDENT (a loaded value reaches a setp / predicate / selp-slct selector / address)"
                                 : "FAIL-CLOSED (call, tainted non-polynomial op, cap saturation, or non-convergence)";
        printf("cflow_witness [%s]: %s -> %s\n", ptx_path, v, (r == 1) ? "PASS" : "FAIL");
        free(ptx);
        return (r == 1) ? 0 : 1;
    }

    CK(cuInit(0), "cuInit");
    CUdevice dev; CK(cuDeviceGet(&dev, 0), "cuDeviceGet");
    char gpu[256]; gpu[0] = 0; cuDeviceGetName(gpu, (int)sizeof gpu, dev);
    CUcontext ctx; CK(cuCtxCreate(&ctx, 0, dev), "cuCtxCreate");

    CUmodule mod; CK(cuModuleLoadData(&mod, ptx), "cuModuleLoadData");
    CUfunction fn; CK(cuModuleGetFunction(&fn, mod, kname), "cuModuleGetFunction");

    /* attention mode: cuda_launch <combined.ptx> gpu_qkt <Nignored> attention <S> <d>.
     * combined.ptx carries 3 entries (gpu_qkt, gpu_softmax, naive_matmul -- concat the
     * 3 kernel .hx files, emit once). Single-head scaled dot-product attention:
     *   scores = (1/sqrt(d)) * Q @ K^T   (gpu_qkt bakes 0.25 = 1/sqrt(16), so d MUST be 16)
     *   attn   = softmax(scores) row-wise
     *   out    = attn @ V                (naive_matmul, A=attn[S,S] B=V[S,d])
     * mod is already loaded from argv[1]; fetch all 3 entries from it (argv[2]=gpu_qkt
     * satisfies the generic get above). Verify out vs a 3-stage CPU reference AND
     * (independent) that each GPU attn row sums to ~1. Integer-valued inputs so QK^T is
     * exact; 0.25*int is exact; the only error is softmax ex2.approx (tol 1e-3). */
    if (strcmp(op, "attention") == 0) {
        int S = (argc > 5) ? atoi(argv[5]) : 4;
        int d = (argc > 6) ? atoi(argv[6]) : 16;
        CUfunction f_qkt, f_sm, f_mm;
        CK(cuModuleGetFunction(&f_qkt, mod, "gpu_qkt"), "get gpu_qkt");
        CK(cuModuleGetFunction(&f_sm, mod, "gpu_softmax"), "get gpu_softmax");
        CK(cuModuleGetFunction(&f_mm, mod, "naive_matmul"), "get naive_matmul");
        size_t sd = (size_t)S * d, ssz = (size_t)S * S;
        float* hQ = (float*)malloc(sd * sizeof(float));
        float* hK = (float*)malloc(sd * sizeof(float));
        float* hV = (float*)malloc(sd * sizeof(float));
        float* hO = (float*)malloc(sd * sizeof(float));
        float* hA = (float*)malloc(ssz * sizeof(float));
        float* sc = (float*)malloc(ssz * sizeof(float));
        float* at = (float*)malloc(ssz * sizeof(float));
        if (!hQ || !hK || !hV || !hO || !hA || !sc || !at) return 2;
        for (size_t i = 0; i < sd; i++) {
            hQ[i] = (float)((int)(i % 7) - 3);
            hK[i] = (float)((int)(i % 5) - 2);
            hV[i] = (float)((int)(i % 9) - 4);
        }
        CUdeviceptr dQ, dK, dV, dS, dA, dO;
        CK(cuMemAlloc(&dQ, sd * sizeof(float)), "alloc Q");
        CK(cuMemAlloc(&dK, sd * sizeof(float)), "alloc K");
        CK(cuMemAlloc(&dV, sd * sizeof(float)), "alloc V");
        CK(cuMemAlloc(&dS, ssz * sizeof(float)), "alloc scores");
        CK(cuMemAlloc(&dA, ssz * sizeof(float)), "alloc attn");
        CK(cuMemAlloc(&dO, sd * sizeof(float)), "alloc out");
        CK(cuMemcpyHtoD(dQ, hQ, sd * sizeof(float)), "H2D Q");
        CK(cuMemcpyHtoD(dK, hK, sd * sizeof(float)), "H2D K");
        CK(cuMemcpyHtoD(dV, hV, sd * sizeof(float)), "H2D V");
        /* stage 1: scores = 0.25 * Q@K^T. gridDim=S, blockDim=S. */
        void* a1[] = { &dQ, &dK, &dS, &S, &d };
        CK(cuLaunchKernel(f_qkt, S, 1, 1, S, 1, 1, 0, 0, a1, 0), "launch qkt");
        CK(cuCtxSynchronize(), "sync qkt");
        /* stage 2: attn = softmax(scores) row-wise. gridDim=S, blockDim=1, cols=S. */
        void* a2[] = { &dS, &dA, &S, &S };
        CK(cuLaunchKernel(f_sm, S, 1, 1, 1, 1, 1, 0, 0, a2, 0), "launch softmax");
        CK(cuCtxSynchronize(), "sync softmax");
        /* stage 3: out = attn @ V. naive_matmul(a,b,c,M,K,N): M=S,K=S,N=d. gridDim=S, blockDim=d. */
        void* a3[] = { &dA, &dV, &dO, &S, &S, &d };
        CK(cuLaunchKernel(f_mm, S, 1, 1, d, 1, 1, 0, 0, a3, 0), "launch matmul");
        CK(cuCtxSynchronize(), "sync matmul");
        CK(cuMemcpyDtoH(hO, dO, sd * sizeof(float)), "D2H out");
        CK(cuMemcpyDtoH(hA, dA, ssz * sizeof(float)), "D2H attn");
        /* CPU reference: scores -> row softmax -> out. */
        float scale = 1.0f / sqrtf((float)d);
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < S; j++) {
                float dot = 0.0f;
                for (int t = 0; t < d; t++) dot += hQ[i * d + t] * hK[j * d + t];
                sc[i * S + j] = scale * dot;
            }
            float mx = sc[i * S]; for (int j = 1; j < S; j++) if (sc[i * S + j] > mx) mx = sc[i * S + j];
            float sm = 0.0f; for (int j = 0; j < S; j++) sm += expf(sc[i * S + j] - mx);
            for (int j = 0; j < S; j++) at[i * S + j] = expf(sc[i * S + j] - mx) / sm;
        }
        int abad = 0; float ref0 = 0.0f; float maxrs = 0.0f;
        for (int i = 0; i < S; i++) {
            for (int kk = 0; kk < d; kk++) {
                float ref = 0.0f; for (int j = 0; j < S; j++) ref += at[i * S + j] * hV[j * d + kk];
                if (i == 0 && kk == 0) ref0 = ref;
                float got = hO[i * d + kk];
                float e = got - ref; if (e < 0) e = -e;
                if (isnan(got) || e > 1.0e-3f) { if (abad < 4) fprintf(stderr, "attention mismatch out[%d,%d]=%g ref %g\n", i, kk, got, ref); abad++; }
            }
        }
        /* independent cross-check: each GPU attn row sums to ~1. */
        for (int i = 0; i < S; i++) {
            float rs = 0.0f; for (int j = 0; j < S; j++) rs += hA[i * S + j];
            float e = rs - 1.0f; if (e < 0) e = -e; if (e > maxrs) maxrs = e;
            if (isnan(rs) || e > 1.0e-3f) { if (abad < 4) fprintf(stderr, "attention attn row %d sum %g (want 1)\n", i, rs); abad++; }
        }
        printf("GPU [%s] attention S=%d d=%d: out[0,0]=%g ref %g, max|attn_rowsum-1|=%g, %d bad -> %s\n",
               gpu, S, d, hO[0], ref0, maxrs, abad, abad ? "FAIL" : "PASS");
        cuMemFree(dQ); cuMemFree(dK); cuMemFree(dV); cuMemFree(dS); cuMemFree(dA); cuMemFree(dO);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hQ); free(hK); free(hV); free(hO); free(hA); free(sc); free(at); free(ptx);
        return abad ? 1 : 0;
    }

    /* attn_flash mode (T2/M4): cuda_launch <combined.ptx> flash_attention <Nignored> attn_flash <S> <d> [mutate].
     * The FUSED FLASH-STYLE ATTENTION milestone. combined.ptx carries the kovc fused kernel
     * (argv[2]=flash_attention, ONE block per query row, online softmax, NO S x S materialized)
     * AND the naive 3-kernel baseline (gpu_qkt + gpu_softmax + naive_matmul -- the unfused
     * QK^T -> softmax -> @V pipeline that round-trips the S x S scores/attn matrices through HBM).
     *  (a) CORRECTNESS: the fused out[S,d] is compared cell-by-cell vs a CPU reference
     *      out = softmax(scale*Q@K^T) @ V, scale=1/sqrt(d) (the SAME runtime scale the kernel
     *      computes via rsqrt) -- integer-valued inputs so Q@K^T + scale + the @V matmul are
     *      EXACT; the only error is the kernel's ex2.approx exp, covered by tol 1e-3 (maxrel
     *      reported honestly). An INDEPENDENT cross-check that the implied attention weights sum
     *      to 1 per row is folded into the algebra (out is an exact convex combination of V rows;
     *      verified by the cell match against the normalized CPU ref).
     *  (b) FASTER-THAN-NAIVE: kernel-only cuEvent median of the single fused launch vs the naive
     *      3-launch pipeline (qkt + softmax + matmul, each materializing S x S in HBM). The naive
     *      gpu_qkt bakes 0.25=1/sqrt(16), so the timing baseline uses d=16; correctness is at the
     *      runtime scale for any d. Reported median + SPEEDUP; gated faster-than-naive at LARGE S.
     *  (c) NEG-CONTROLS: "mutate" perturbs one out cell pre-compare (comparator teeth -> must FAIL);
     *      the online-rescale-strip neg-control is driven by the corpus script (delete the exp(m-m_new)
     *      corr lines -> the running softmax mis-normalizes -> mis-computes, proving the online
     *      rescale is load-bearing). d <= 256 (kernel block size). */
    if (strcmp(op, "attn_flash") == 0) {
        int S = (argc > 5) ? atoi(argv[5]) : 8;
        int d = (argc > 6) ? atoi(argv[6]) : 16;
        int mutate = (argc > 7 && strcmp(argv[7], "mutate") == 0);
        int corr_only = (getenv("CORR_ONLY") != NULL);
        float ATTN_TOL = 1.0e-3f; { const char* e; if ((e = getenv("ATTN_TOL"))) ATTN_TOL = (float)atof(e); }
        if (d > 256) { fprintf(stderr, "attn_flash: d<=256 (block size); got %d\n", d); return 2; }
        if (256 % d != 0) { fprintf(stderr, "attn_flash: d must divide 256 (Phase-3 col split); got %d\n", d); return 2; }
        if (S > 4096) { fprintf(stderr, "attn_flash: S<=4096 (smem scores 16384B); got %d\n", S); return 2; }
        size_t sd = (size_t)S * d, ss = (size_t)S * S;
        float* hQ = (float*)malloc(sd * sizeof(float));
        float* hK = (float*)malloc(sd * sizeof(float));
        float* hV = (float*)malloc(sd * sizeof(float));
        float* hO = (float*)malloc(sd * sizeof(float));
        float* refo = (float*)malloc(sd * sizeof(float));
        float* sc = (float*)malloc(ss * sizeof(float));
        if (!hQ || !hK || !hV || !hO || !refo || !sc) return 2;
        /* integer-valued inputs (Q@K^T exact); same distribution family as the attention mode. */
        for (size_t i = 0; i < sd; i++) {
            hQ[i] = (float)((int)(i % 7) - 3);
            hK[i] = (float)((int)(i % 5) - 2);
            hV[i] = (float)((int)(i % 9) - 4);
        }
        CUdeviceptr dQ, dK, dV, dO;
        CK(cuMemAlloc(&dQ, sd * sizeof(float)), "alloc Q");
        CK(cuMemAlloc(&dK, sd * sizeof(float)), "alloc K");
        CK(cuMemAlloc(&dV, sd * sizeof(float)), "alloc V");
        CK(cuMemAlloc(&dO, sd * sizeof(float)), "alloc O");
        CK(cuMemcpyHtoD(dQ, hQ, sd * sizeof(float)), "H2D Q");
        CK(cuMemcpyHtoD(dK, hK, sd * sizeof(float)), "H2D K");
        CK(cuMemcpyHtoD(dV, hV, sd * sizeof(float)), "H2D V");
        /* FUSED launch: grid=(S,1,1), block=(256,1,1). */
        void* fa[] = { &dQ, &dK, &dV, &dO, &S, &d };
        CK(cuLaunchKernel(fn, S, 1, 1, 256, 1, 1, 0, 0, fa, 0), "launch flash_attention");
        CK(cuCtxSynchronize(), "sync flash_attention");
        CK(cuMemcpyDtoH(hO, dO, sd * sizeof(float)), "D2H O");
        if (mutate) hO[0] += 1.0f;   /* comparator negative control */
        /* CPU reference: out = softmax(scale*Q@K^T) @ V, scale = 1/sqrt(d). */
        float scale = 1.0f / sqrtf((float)d);
        int abad = 0; float maxrel = 0.0f; float ref0 = 0.0f;
        for (int i = 0; i < S; i++) {
            for (int j = 0; j < S; j++) {
                float dot = 0.0f; for (int t = 0; t < d; t++) dot += hQ[i * d + t] * hK[j * d + t];
                sc[i * S + j] = scale * dot;
            }
            float mx = sc[i * S]; for (int j = 1; j < S; j++) if (sc[i * S + j] > mx) mx = sc[i * S + j];
            float sm = 0.0f; for (int j = 0; j < S; j++) sm += expf(sc[i * S + j] - mx);
            for (int j = 0; j < S; j++) sc[i * S + j] = expf(sc[i * S + j] - mx) / sm;
            for (int t = 0; t < d; t++) {
                float o = 0.0f; for (int j = 0; j < S; j++) o += sc[i * S + j] * hV[j * d + t];
                refo[i * d + t] = o;
            }
        }
        for (int i = 0; i < S; i++) for (int t = 0; t < d; t++) {
            float ref = refo[i * d + t]; float got = hO[i * d + t];
            if (i == 0 && t == 0) ref0 = ref;
            float e = got - ref; if (e < 0) e = -e;
            float ar = ref < 0 ? -ref : ref; float rel = e / (ar > 1.0f ? ar : 1.0f);
            if (rel > maxrel) maxrel = rel;
            if (isnan(got) || (e > ATTN_TOL && rel > ATTN_TOL)) { if (abad < 6) fprintf(stderr, "attn_flash mismatch out[%d,%d]=%g ref %g (rel %.2e)\n", i, t, got, ref, rel); abad++; }
        }
        /* TIMING (kernel-only): fused single launch vs the naive 3-kernel pipeline.
         * The naive gpu_qkt bakes 0.25=1/sqrt(16) so the timing baseline is valid for d=16;
         * for d!=16 correctness still holds (runtime scale) but the naive baseline is skipped. */
        double fused_med = 0.0, naive_med = 0.0; int timed = 0;
        if (!mutate && !corr_only) {
            CUfunction f_qkt, f_sm, f_mm;
            if (cuModuleGetFunction(&f_qkt, mod, "gpu_qkt") == CUDA_SUCCESS &&
                cuModuleGetFunction(&f_sm, mod, "gpu_softmax") == CUDA_SUCCESS &&
                cuModuleGetFunction(&f_mm, mod, "naive_matmul") == CUDA_SUCCESS && S <= 1024) {
                CUdeviceptr dS2, dA2;
                CK(cuMemAlloc(&dS2, ss * sizeof(float)), "alloc scores");
                CK(cuMemAlloc(&dA2, ss * sizeof(float)), "alloc attn");
                void* a1[] = { &dQ, &dK, &dS2, &S, &d };
                void* a2[] = { &dS2, &dA2, &S, &S };
                void* a3[] = { &dA2, &dV, &dO, &S, &S, &d };
                int WARM = 5, ITERS = 50;
                float* ts = (float*)malloc((size_t)ITERS * sizeof(float));
                CUevent e0, e1; CK(cuEventCreate(&e0, 0), "evt0"); CK(cuEventCreate(&e1, 0), "evt1");
                /* warm + time the FUSED kernel */
                for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(fn, S, 1, 1, 256, 1, 1, 0, 0, fa, 0), "warm fused");
                CK(cuCtxSynchronize(), "sync warm fused");
                for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0, 0), "f0"); CK(cuLaunchKernel(fn, S, 1, 1, 256, 1, 1, 0, 0, fa, 0), "time fused"); CK(cuEventRecord(e1, 0), "f1"); CK(cuEventSynchronize(e1), "fes"); CK(cuEventElapsedTime(&ts[it], e0, e1), "fel"); }
                for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
                fused_med = (double)ts[ITERS / 2];
                /* warm + time the NAIVE 3-kernel pipeline (qkt -> softmax -> matmul). */
                for (int w = 0; w < WARM; w++) { CK(cuLaunchKernel(f_qkt, S, 1, 1, S, 1, 1, 0, 0, a1, 0), "warm qkt"); CK(cuLaunchKernel(f_sm, S, 1, 1, 1, 1, 1, 0, 0, a2, 0), "warm sm"); CK(cuLaunchKernel(f_mm, S, 1, 1, d, 1, 1, 0, 0, a3, 0), "warm mm"); }
                CK(cuCtxSynchronize(), "sync warm naive");
                for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0, 0), "n0"); CK(cuLaunchKernel(f_qkt, S, 1, 1, S, 1, 1, 0, 0, a1, 0), "t qkt"); CK(cuLaunchKernel(f_sm, S, 1, 1, 1, 1, 1, 0, 0, a2, 0), "t sm"); CK(cuLaunchKernel(f_mm, S, 1, 1, d, 1, 1, 0, 0, a3, 0), "t mm"); CK(cuEventRecord(e1, 0), "n1"); CK(cuEventSynchronize(e1), "nes"); CK(cuEventElapsedTime(&ts[it], e0, e1), "nel"); }
                for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
                naive_med = (double)ts[ITERS / 2];
                cuEventDestroy(e0); cuEventDestroy(e1); free(ts);
                cuMemFree(dS2); cuMemFree(dA2);
                timed = 1;
                printf("GPU [%s] TIMING attn S=%d d=%d: fused med=%.4f ms  naive(3-kernel) med=%.4f ms  SPEEDUP=%.2fx\n", gpu, S, d, fused_med, naive_med, naive_med / fused_med);
            }
        }
        int faster = (mutate || corr_only || !timed) ? 1 : (naive_med > fused_med);
        int totbad = abad || !faster;
        printf("GPU [%s] attn_flash (FUSED online-softmax, 256t/row, NO SxS) S=%d d=%d%s: out[0,0]=%g ref %g, maxrel=%.2e (tol=%.0e), %d bad, faster-than-naive=%s -> %s\n",
               gpu, S, d, mutate ? " [MUTATED]" : "", hO[0], ref0, maxrel, ATTN_TOL, abad,
               mutate ? "n/a" : (corr_only || !timed ? "not-gated(corr)" : (faster ? "YES" : "NO")), totbad ? "FAIL" : "PASS");
        cuMemFree(dQ); cuMemFree(dK); cuMemFree(dV); cuMemFree(dO);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hQ); free(hK); free(hV); free(hO); free(refo); free(sc); free(ptx);
        return totbad ? 1 : 0;
    }

    /* attn_backward mode: cuda_launch <combined.ptx> gpu_qkt <Nignored> attn_backward <S> <d>.
     * combined.ptx has 7 entries (gpu_qkt, gpu_softmax, naive_matmul, gpu_matmul_abt,
     * gpu_matmul_atb, gpu_softmax_backward, gpu_scale_inplace). Single-head attention forward
     * (scores=0.25*Q@K^T -> softmax -> out=attn@V) + backward given dOut: dV=attn^T@dOut,
     * d_attn=dOut@V^T, d_scores=softmax_bwd(attn,d_attn), dQ=0.25*d_scores@K, dK=0.25*d_scores^T@Q.
     * Verify dQ/dK/dV vs a central finite-difference of L=sum(dOut*out) wrt Q/K/V (INDEPENDENT;
     * uses only the forward; tol 2e-2). mod already loaded from argv[1]. */
    if (strcmp(op, "attn_backward") == 0) {
        int S = (argc > 5) ? atoi(argv[5]) : 8;
        int d = (argc > 6) ? atoi(argv[6]) : 16;
        CUfunction f_qkt, f_sm, f_mm, f_abt, f_atb, f_smb, f_sc;
        CK(cuModuleGetFunction(&f_qkt, mod, "gpu_qkt"), "get qkt");
        CK(cuModuleGetFunction(&f_sm, mod, "gpu_softmax"), "get softmax");
        CK(cuModuleGetFunction(&f_mm, mod, "naive_matmul"), "get matmul");
        CK(cuModuleGetFunction(&f_abt, mod, "gpu_matmul_abt"), "get abt");
        CK(cuModuleGetFunction(&f_atb, mod, "gpu_matmul_atb"), "get atb");
        CK(cuModuleGetFunction(&f_smb, mod, "gpu_softmax_backward"), "get sm_bwd");
        CK(cuModuleGetFunction(&f_sc, mod, "gpu_scale_inplace"), "get scale");
        size_t sd = (size_t)S * d, ss = (size_t)S * S;
        float* hQ = (float*)malloc(sd * sizeof(float));
        float* hK = (float*)malloc(sd * sizeof(float));
        float* hV = (float*)malloc(sd * sizeof(float));
        float* hdO = (float*)malloc(sd * sizeof(float));
        float* hdV = (float*)malloc(sd * sizeof(float));
        float* hdQ = (float*)malloc(sd * sizeof(float));
        float* hdK = (float*)malloc(sd * sizeof(float));
        float* tmp = (float*)malloc(sd * sizeof(float));
        float* ob = (float*)malloc(sd * sizeof(float));
        if (!hQ || !hK || !hV || !hdO || !hdV || !hdQ || !hdK || !tmp || !ob) return 2;
        for (size_t i = 0; i < sd; i++) {
            hQ[i] = (float)((int)(i % 7) - 3) * 0.1f;
            hK[i] = (float)((int)(i % 5) - 2) * 0.1f;
            hV[i] = (float)((int)(i % 9) - 4) * 0.1f;
            hdO[i] = (float)((int)(i % 6) - 2) * 0.1f;
        }
        CUdeviceptr gQ, gK, gV, gdO, gSc, gAt, gDat, gDsc, gOut, gDV, gDQ, gDK;
        CK(cuMemAlloc(&gQ, sd * sizeof(float)), "Q"); CK(cuMemAlloc(&gK, sd * sizeof(float)), "K");
        CK(cuMemAlloc(&gV, sd * sizeof(float)), "V"); CK(cuMemAlloc(&gdO, sd * sizeof(float)), "dO");
        CK(cuMemAlloc(&gSc, ss * sizeof(float)), "Sc"); CK(cuMemAlloc(&gAt, ss * sizeof(float)), "At");
        CK(cuMemAlloc(&gDat, ss * sizeof(float)), "Dat"); CK(cuMemAlloc(&gDsc, ss * sizeof(float)), "Dsc");
        CK(cuMemAlloc(&gOut, sd * sizeof(float)), "Out"); CK(cuMemAlloc(&gDV, sd * sizeof(float)), "DV");
        CK(cuMemAlloc(&gDQ, sd * sizeof(float)), "DQ"); CK(cuMemAlloc(&gDK, sd * sizeof(float)), "DK");
        CK(cuMemcpyHtoD(gQ, hQ, sd * sizeof(float)), "h2d Q"); CK(cuMemcpyHtoD(gK, hK, sd * sizeof(float)), "h2d K");
        CK(cuMemcpyHtoD(gV, hV, sd * sizeof(float)), "h2d V"); CK(cuMemcpyHtoD(gdO, hdO, sd * sizeof(float)), "h2d dO");
        int Si = S, di = d, sdi = (int)sd;
        /* FORWARD */
        void* fa[] = { &gQ, &gK, &gSc, &Si, &di }; CK(cuLaunchKernel(f_qkt, S, 1, 1, S, 1, 1, 0, 0, fa, 0), "qkt"); CK(cuCtxSynchronize(), "s");
        void* fb[] = { &gSc, &gAt, &Si, &Si }; CK(cuLaunchKernel(f_sm, S, 1, 1, 1, 1, 1, 0, 0, fb, 0), "sm"); CK(cuCtxSynchronize(), "s");
        void* fc[] = { &gAt, &gV, &gOut, &Si, &Si, &di }; CK(cuLaunchKernel(f_mm, S, 1, 1, d, 1, 1, 0, 0, fc, 0), "mm"); CK(cuCtxSynchronize(), "s");
        /* BACKWARD */
        void* b1[] = { &gAt, &gdO, &gDV, &Si, &Si, &di }; CK(cuLaunchKernel(f_atb, S, 1, 1, d, 1, 1, 0, 0, b1, 0), "dV"); CK(cuCtxSynchronize(), "s");
        void* b2[] = { &gdO, &gV, &gDat, &Si, &di, &Si }; CK(cuLaunchKernel(f_abt, S, 1, 1, S, 1, 1, 0, 0, b2, 0), "dAttn"); CK(cuCtxSynchronize(), "s");
        void* b3[] = { &gAt, &gDat, &gDsc, &Si, &Si }; CK(cuLaunchKernel(f_smb, S, 1, 1, 1, 1, 1, 0, 0, b3, 0), "dScores"); CK(cuCtxSynchronize(), "s");
        void* b4[] = { &gDsc, &gK, &gDQ, &Si, &Si, &di }; CK(cuLaunchKernel(f_mm, S, 1, 1, d, 1, 1, 0, 0, b4, 0), "dQ"); CK(cuCtxSynchronize(), "s");
        void* s1[] = { &gDQ, &sdi }; CK(cuLaunchKernel(f_sc, sdi, 1, 1, 1, 1, 1, 0, 0, s1, 0), "scaleQ"); CK(cuCtxSynchronize(), "s");
        void* b5[] = { &gDsc, &gQ, &gDK, &Si, &Si, &di }; CK(cuLaunchKernel(f_atb, S, 1, 1, d, 1, 1, 0, 0, b5, 0), "dK"); CK(cuCtxSynchronize(), "s");
        void* s2[] = { &gDK, &sdi }; CK(cuLaunchKernel(f_sc, sdi, 1, 1, 1, 1, 1, 0, 0, s2, 0), "scaleK"); CK(cuCtxSynchronize(), "s");
        CK(cuMemcpyDtoH(hdV, gDV, sd * sizeof(float)), "d2h dV");
        CK(cuMemcpyDtoH(hdQ, gDQ, sd * sizeof(float)), "d2h dQ");
        CK(cuMemcpyDtoH(hdK, gDK, sd * sizeof(float)), "d2h dK");
        /* finite-difference of L=sum(dOut*out) wrt Q (->dQ), K (->dK), V (->dV). */
        int bad = 0; float maxe = 0.0f, q0 = hdQ[0], q0r = 0.0f, h = 1.0e-3f;
        for (int pass = 0; pass < 3; pass++) {
            float* base = (pass == 0) ? hQ : (pass == 1) ? hK : hV;
            float* got = (pass == 0) ? hdQ : (pass == 1) ? hdK : hdV;
            for (size_t i = 0; i < sd; i++) {
                for (size_t k = 0; k < sd; k++) tmp[k] = base[k];
                tmp[i] = base[i] + h;
                attn_forward_cpu(pass == 0 ? tmp : hQ, pass == 1 ? tmp : hK, pass == 2 ? tmp : hV, ob, S, d);
                float Lp = 0.0f; for (size_t k = 0; k < sd; k++) Lp += hdO[k] * ob[k];
                tmp[i] = base[i] - h;
                attn_forward_cpu(pass == 0 ? tmp : hQ, pass == 1 ? tmp : hK, pass == 2 ? tmp : hV, ob, S, d);
                float Lm = 0.0f; for (size_t k = 0; k < sd; k++) Lm += hdO[k] * ob[k];
                float fd = (Lp - Lm) / (2.0f * h);
                if (pass == 0 && i == 0) q0r = fd;
                float e = got[i] - fd; if (e < 0) e = -e; if (e > maxe) maxe = e;
                float af = fd < 0 ? -fd : fd;
                /* magnitude-aware: an absolute floor 1e-3 (ignores finite-diff/f32 noise on
                 * near-zero grads) AND a 5% relative bound (catches multiplicative errors like a
                 * wrong backward scale on the small dQ/dK gradients -- an audit found an
                 * absolute-only 2e-2 tol missed a 2x scale error here). */
                if (isnan(got[i]) || (e > 1.0e-3f && e > 0.05f * af)) { if (bad < 6) fprintf(stderr, "attn_bwd pass %d [%zu] got %g fd %g\n", pass, i, got[i], fd); bad++; }
            }
        }
        printf("GPU [%s] attn_backward S=%d d=%d: dQ[0]=%g fd %g, max|grad-fd|=%g, %d bad -> %s\n",
               gpu, S, d, q0, q0r, maxe, bad, bad ? "FAIL" : "PASS");
        cuMemFree(gQ); cuMemFree(gK); cuMemFree(gV); cuMemFree(gdO); cuMemFree(gSc); cuMemFree(gAt);
        cuMemFree(gDat); cuMemFree(gDsc); cuMemFree(gOut); cuMemFree(gDV); cuMemFree(gDQ); cuMemFree(gDK);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hQ); free(hK); free(hV); free(hdO); free(hdV); free(hdQ); free(hdK); free(tmp); free(ob); free(ptx);
        return bad ? 1 : 0;
    }

    /* ce_softmax_grad mode: cuda_launch <ptx> gpu_ce_softmax_grad <Nignored> ce_softmax_grad <rows> <cols>.
     * softmax-cross-entropy backward: dlogits[r,c]=softmax(logits[r,:])[c]-onehot(tgt[r])[c]. Targets
     * passed as f32 (class index as float). Verify cell-by-cell vs a CPU softmax-minus-onehot ref
     * (tol 1e-4) AND the INDEPENDENT conservation property that each grad row sums to ~0 (a missing
     * or misplaced onehot breaks this even when every cell formula looks plausible). */
    if (strcmp(op, "ce_softmax_grad") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 8;
        int cols = (argc > 6) ? atoi(argv[6]) : 16;
        size_t ne = (size_t)rows * cols;
        float* hL = (float*)malloc(ne * sizeof(float));
        float* hG = (float*)malloc(ne * sizeof(float));
        float* htg = (float*)malloc((size_t)rows * sizeof(float));
        if (!hL || !hG || !htg) return 2;
        for (size_t i = 0; i < ne; i++) hL[i] = (float)((int)((i * 7 + 3) % 13) - 6);
        for (int r = 0; r < rows; r++) htg[r] = (float)((r * 5 + 2) % cols);
        CUdeviceptr dL, dT, dG;
        CK(cuMemAlloc(&dL, ne * sizeof(float)), "alloc L");
        CK(cuMemAlloc(&dT, (size_t)rows * sizeof(float)), "alloc T");
        CK(cuMemAlloc(&dG, ne * sizeof(float)), "alloc G");
        CK(cuMemcpyHtoD(dL, hL, ne * sizeof(float)), "H2D L");
        CK(cuMemcpyHtoD(dT, htg, (size_t)rows * sizeof(float)), "H2D T");
        void* cargs[] = { &dL, &dT, &dG, &rows, &cols };
        CK(cuLaunchKernel(fn, rows, 1, 1, 1, 1, 1, 0, 0, cargs, 0), "launch ce_grad");
        CK(cuCtxSynchronize(), "sync ce_grad");
        CK(cuMemcpyDtoH(hG, dG, ne * sizeof(float)), "D2H G");
        int cbad = 0; float g00 = 0.0f, ref00 = 0.0f, maxrs = 0.0f;
        for (int r = 0; r < rows; r++) {
            float mx = hL[r * cols]; for (int c = 1; c < cols; c++) if (hL[r * cols + c] > mx) mx = hL[r * cols + c];
            float sm = 0.0f; for (int c = 0; c < cols; c++) sm += expf(hL[r * cols + c] - mx);
            int tgt = (int)htg[r];
            float rs = 0.0f;
            for (int c = 0; c < cols; c++) {
                float p = expf(hL[r * cols + c] - mx) / sm;
                float ref = p - ((c == tgt) ? 1.0f : 0.0f);
                float got = hG[r * cols + c];
                if (r == 0 && c == 0) { g00 = got; ref00 = ref; }
                rs += got;
                float e = got - ref; if (e < 0) e = -e;
                if (isnan(got) || e > 1.0e-4f) { if (cbad < 4) fprintf(stderr, "ce_grad mismatch g[%d,%d]=%g ref %g\n", r, c, got, ref); cbad++; }
            }
            float ers = rs < 0 ? -rs : rs; if (ers > maxrs) maxrs = ers;
            if (isnan(rs) || ers > 1.0e-3f) { if (cbad < 4) fprintf(stderr, "ce_grad row %d sum %g (want 0)\n", r, rs); cbad++; }
        }
        printf("GPU [%s] ce_softmax_grad %dx%d: g[0,0]=%g ref %g, max|row_sum|=%g, %d bad -> %s\n",
               gpu, rows, cols, g00, ref00, maxrs, cbad, cbad ? "FAIL" : "PASS");
        cuMemFree(dL); cuMemFree(dT); cuMemFree(dG);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hL); free(hG); free(htg); free(ptx);
        return cbad ? 1 : 0;
    }

    /* matmul mode: cuda_launch <ptx> <kernel> <Nvec-ignored> matmul <M> <K> <N>.
     * One thread per output cell -- gridDim.x=M (row=block_idx), blockDim.x=N
     * (col=thread_idx). Verifies EVERY M*N cell of C against a CPU reference in the
     * SAME accumulation order the kernel uses (acc=a[row*K]*b[col], then += k=1..K-1),
     * with integer-valued inputs (a[i]=i%7, b[i]=i%5) so all sums are exact in f32. */
    if (strcmp(op, "matmul") == 0) {
        int Md = (argc > 5) ? atoi(argv[5]) : 16;
        int Kd = (argc > 6) ? atoi(argv[6]) : 16;
        int Nd = (argc > 7) ? atoi(argv[7]) : 16;
        size_t aN = (size_t)Md * Kd, bN = (size_t)Kd * Nd, cN = (size_t)Md * Nd;
        float* hA = (float*)malloc(aN * sizeof(float));
        float* hB = (float*)malloc(bN * sizeof(float));
        float* hC = (float*)malloc(cN * sizeof(float));
        if (!hA || !hB || !hC) return 2;
        for (size_t i = 0; i < aN; i++) hA[i] = (float)(i % 7);
        for (size_t i = 0; i < bN; i++) hB[i] = (float)(i % 5);
        CUdeviceptr dA, dB, dC;
        CK(cuMemAlloc(&dA, aN * sizeof(float)), "cuMemAlloc A");
        CK(cuMemAlloc(&dB, bN * sizeof(float)), "cuMemAlloc B");
        CK(cuMemAlloc(&dC, cN * sizeof(float)), "cuMemAlloc C");
        CK(cuMemcpyHtoD(dA, hA, aN * sizeof(float)), "cuMemcpyHtoD A");
        CK(cuMemcpyHtoD(dB, hB, bN * sizeof(float)), "cuMemcpyHtoD B");
        void* margs[] = { &dA, &dB, &dC, &Md, &Kd, &Nd };
        CK(cuLaunchKernel(fn, Md, 1, 1, Nd, 1, 1, 0, 0, margs, 0), "cuLaunchKernel matmul");
        CK(cuCtxSynchronize(), "cuCtxSynchronize");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(float)), "cuMemcpyDtoH C");
        int mbad = 0;
        for (int r = 0; r < Md; r++) {
            for (int cc = 0; cc < Nd; cc++) {
                float ref = hA[r * Kd] * hB[cc];
                for (int t = 1; t < Kd; t++) ref += hA[r * Kd + t] * hB[t * Nd + cc];
                float got = hC[r * Nd + cc];
                if (got != ref) { if (mbad < 4) fprintf(stderr, "matmul mismatch C[%d,%d]=%g ref %g\n", r, cc, got, ref); mbad++; }
            }
        }
        printf("GPU [%s] naive_matmul %dx%dx%d over %d cells: C[1,1]=%g, %d bad -> %s\n",
               gpu, Md, Kd, Nd, Md * Nd, hC[1 * Nd + 1], mbad, mbad ? "FAIL" : "PASS");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hC); free(ptx);
        return mbad ? 1 : 0;
    }

    /* v1.5 #2 LEG 2 (basis-agreement): the EXACT all-inputs equivalence teeth. For the kovc-emitted
     * naive_matmul at a fixed square shape L (default 8), sweep ALL rank-1 0/1 probes A=e_{a,b}, B=e_{c,d}
     * over (a,b,c,d) in [0,L)^4. With 0/1 inputs every product/partial sum is 0 or 1 -> EXACT in f32, so
     * the device output must EQUAL the spec bit-for-bit: matmul(e_ab, e_cd)[i][j] = [i==a]*[b==c]*[j==d]
     * (a single 1 at [a][d] when b==c, else all-zero). Given LEG 1 (the kernel is a fixed straight-line
     * data-INDEPENDENT program for this shape), basis agreement pins the kernel's BILINEAR coefficient
     * tensor to the matmul's -- but that ALONE does not exclude higher-degree terms, so LEG 3 (bilinearity)
     * is CO-NECESSARY (the a*a nonlinearity NC passes this 0/1 basis: 0^2=0, 1^2=1). LEG 1 + LEG 2 + LEG 3
     * TOGETHER give f == matmul on all f32 inputs (this compiled shape, f32 envelope); LEG 2 is not a
     * standalone all-inputs proof. We
     * sweep the FULL quad incl. the OFF-DIAGONAL b!=c (=> all-zero) so a spurious extra/transposed term
     * cannot hide, and assert executed-probes == L^4 AND expected-nonzero == L^3 (non-vacuity: a skipped
     * sweep cannot false-pass). Usage: cuda_launch <ptx> naive_matmul 0 matmul_basis [L]. */
    if (strcmp(op, "matmul_basis") == 0) {
        int L = (argc > 5) ? atoi(argv[5]) : 8;
        if (L < 2) L = 2; if (L > 16) L = 16;            /* L^4 launches: 16 -> 65536, keep bounded */
        /* kind selects the bilinear spec so the SAME basis sweep certifies the matmul AND its two adjoint
         * (autodiff) siblings: ab = A@B [naive_matmul], atb = A^T@B [gpu_matmul_atb, the weight gradient],
         * abt = A@B^T [gpu_matmul_abt, the input gradient]. For square L the launch geometry (grid=L,
         * block=L, args {.,.,.,L,L,L}) is identical; only the 0/1 reference index map differs. */
        const char* kind = (argc > 6) ? argv[6] : "ab";
        int k_ab = (strcmp(kind, "ab") == 0), k_atb = (strcmp(kind, "atb") == 0), k_abt = (strcmp(kind, "abt") == 0);
        if (!k_ab && !k_atb && !k_abt) { fprintf(stderr, "matmul_basis: unknown kind '%s' (ab|atb|abt)\n", kind); return 2; }
        size_t nn2 = (size_t)L * L;
        float* hA = (float*)calloc(nn2, sizeof(float));
        float* hB = (float*)calloc(nn2, sizeof(float));
        float* hC = (float*)malloc(nn2 * sizeof(float));
        if (!hA || !hB || !hC) return 2;
        CUdeviceptr dA, dB, dC;
        CK(cuMemAlloc(&dA, nn2 * sizeof(float)), "basis A");
        CK(cuMemAlloc(&dB, nn2 * sizeof(float)), "basis B");
        CK(cuMemAlloc(&dC, nn2 * sizeof(float)), "basis C");
        long probes = 0, nonzero_probes = 0, bad = 0;
        for (int a = 0; a < L; a++) for (int b = 0; b < L; b++)
        for (int cc = 0; cc < L; cc++) for (int d = 0; d < L; d++) {
            hA[(size_t)a * L + b] = 1.0f;                /* A = e_{a,b} */
            hB[(size_t)cc * L + d] = 1.0f;               /* B = e_{cc,d} */
            CK(cuMemcpyHtoD(dA, hA, nn2 * sizeof(float)), "basis HtoD A");
            CK(cuMemcpyHtoD(dB, hB, nn2 * sizeof(float)), "basis HtoD B");
            int Md = L, Kd = L, Nd = L;
            void* bargs[] = { &dA, &dB, &dC, &Md, &Kd, &Nd };
            CK(cuLaunchKernel(fn, L, 1, 1, L, 1, 1, 0, 0, bargs, 0), "basis launch");
            CK(cuCtxSynchronize(), "basis sync");
            CK(cuMemcpyDtoH(hC, dC, nn2 * sizeof(float)), "basis DtoH");
            int nz = k_ab ? (b == cc) : k_atb ? (a == cc) : (b == d);  /* this probe yields a single 1 */
            if (nz) nonzero_probes++;
            for (int i = 0; i < L; i++) for (int j = 0; j < L; j++) {
                /* ab: C[i,j]=[i==a][b==cc][j==d]; atb: [a==cc][i==b][j==d]; abt: [i==a][j==cc][b==d] */
                int one = k_ab  ? (i == a && j == d  && b == cc)
                        : k_atb ? (i == b && j == d  && a == cc)
                        :         (i == a && j == cc && b == d);
                float ref = one ? 1.0f : 0.0f;
                if (hC[(size_t)i * L + j] != ref) {
                    if (bad < 4) fprintf(stderr, "matmul_basis[%s] mismatch A=e[%d,%d] B=e[%d,%d] C[%d,%d]=%g ref %g\n",
                                         kind, a, b, cc, d, i, j, hC[(size_t)i * L + j], ref);
                    bad++;
                }
            }
            hA[(size_t)a * L + b] = 0.0f;                /* reset for the next probe */
            hB[(size_t)cc * L + d] = 0.0f;
            probes++;
        }
        long ep = (long)L * L * L * L, enz = (long)L * L * L;
        int vacuity = (probes != ep) || (nonzero_probes != enz);
        printf("matmul_basis[%s] L=%d: probes=%ld/%ld nonzero=%ld/%ld bad=%ld vacuity=%d -> %s\n",
               kind, L, probes, ep, nonzero_probes, enz, bad, vacuity, (bad == 0 && !vacuity) ? "PASS" : "FAIL");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hC); free(ptx);
        return (bad == 0 && !vacuity) ? 0 : 1;
    }

    /* v1.5 #2 LEG 3 (bilinearity): the f32-envelope certificate + REQUIRED redundancy (it catches the
     * nonlinearity NC a*a, which is INVISIBLE to LEG 2's 0/1 basis since 0^2=0, 1^2=1). On random f32
     * A1,A2,B1,B2 (|.|<=R) and a scalar s, verify the kernel f obeys bilinearity within the derived f32
     * matmul rounding bound: additivity f(A1+A2,B)=f(A1,B)+f(A2,B) (and in B) and homogeneity
     * f(s*A,B)=s*f(A,B) (and in B). Per element tau = c_safe*L*u*S, u=2^-24, S=sum_t |operands| into both
     * sides (L is the contraction length K at the certified SQUARE shape M=K=N=L; a non-square reuse must
     * substitute the real K, not L). The true worst case is ~4*K*u*S (each f32 dot of length K accumulates (2K-1)*u, plus the host
     * As=fl(A1+A2) rounding and the final add, across both compared sides); c_safe=8 gives 8*K*u*S, a
     * conservative ~2x rigorous margin (exactly 2x at K=L=8) -- DERIVED, not reverse-tuned. NOTE this is a
     * SAMPLED tripwire: bilinearity is checked at ONE input tuple (a single fixed LCG seed + one s, 7
     * evals), not swept -- it catches the nonlinearity NC a*a (invisible to LEG 2's 0/1 basis) but is a
     * tolerance probe, not a proof. A genuine bilinear matmul passes; +const / a*a break it. Reports
     * max rel = |LHS-RHS|/tau (PASS => <=1). Usage: cuda_launch <ptx> naive_matmul 0 matmul_bilin [L]. */
    if (strcmp(op, "matmul_bilin") == 0) {
        int L = (argc > 5) ? atoi(argv[5]) : 8;
        if (L < 2) L = 2; if (L > 64) L = 64;
        float R = 4.0f, s = 2.5f, u = 5.9604645e-8f /* 2^-24 */, c_safe = 8.0f;
        size_t n = (size_t)L * L;
        float *A1 = malloc(n*4), *A2 = malloc(n*4), *B1 = malloc(n*4), *B2 = malloc(n*4);
        float *As = malloc(n*4), *Bs = malloc(n*4), *sA = malloc(n*4), *sB = malloc(n*4);
        float *FA1B1 = malloc(n*4), *FA2B1 = malloc(n*4), *FA1B2 = malloc(n*4);
        float *FAsB1 = malloc(n*4), *FA1Bs = malloc(n*4), *FsAB1 = malloc(n*4), *FA1sB = malloc(n*4);
        float *RHS = malloc(n*4), *Sb = malloc(n*4);
        if (!A1||!A2||!B1||!B2||!As||!Bs||!sA||!sB||!FA1B1||!FA2B1||!FA1B2||!FAsB1||!FA1Bs||!FsAB1||!FA1sB||!RHS||!Sb) return 2;
        unsigned int seed = 0x1234567u;                  /* deterministic LCG -> [-R,R] (reproducible; NO rand/time) */
        for (size_t i = 0; i < n; i++) {
            seed = seed*1664525u+1013904223u; A1[i] = ((float)(seed>>8)/(float)(1u<<24))*2.0f*R-R;
            seed = seed*1664525u+1013904223u; A2[i] = ((float)(seed>>8)/(float)(1u<<24))*2.0f*R-R;
            seed = seed*1664525u+1013904223u; B1[i] = ((float)(seed>>8)/(float)(1u<<24))*2.0f*R-R;
            seed = seed*1664525u+1013904223u; B2[i] = ((float)(seed>>8)/(float)(1u<<24))*2.0f*R-R;
        }
        for (size_t i = 0; i < n; i++) { As[i]=A1[i]+A2[i]; Bs[i]=B1[i]+B2[i]; sA[i]=s*A1[i]; sB[i]=s*B1[i]; }
        int rc = 0;
        rc |= mm_eval(fn,A1,B1,FA1B1,L); rc |= mm_eval(fn,A2,B1,FA2B1,L); rc |= mm_eval(fn,A1,B2,FA1B2,L);
        rc |= mm_eval(fn,As,B1,FAsB1,L); rc |= mm_eval(fn,A1,Bs,FA1Bs,L);
        rc |= mm_eval(fn,sA,B1,FsAB1,L); rc |= mm_eval(fn,A1,sB,FA1sB,L);
        if (rc) { fprintf(stderr, "matmul_bilin: a GPU eval failed\n"); return 2; }
        long bad = 0; float maxrel = 0.0f, maxsig = 0.0f;
        for (size_t k = 0; k < n; k++) { float v = FA1B1[k]; if (v<0) v=-v; if (v>maxsig) maxsig=v; }
        /* additivity in A: f(A1+A2,B1) == f(A1,B1)+f(A2,B1); S = sum_t (|A1|+|A2|)*|B1| */
        for (int i=0;i<L;i++) for (int j=0;j<L;j++){ size_t k=(size_t)i*L+j; RHS[k]=FA1B1[k]+FA2B1[k];
            float S=0; for (int t=0;t<L;t++) S += (fabsf(A1[i*L+t])+fabsf(A2[i*L+t]))*fabsf(B1[t*L+j]); Sb[k]=S; }
        bilin_one(FAsB1, RHS, Sb, n, L, u, c_safe, "add_A", &bad, &maxrel);
        /* additivity in B: f(A1,B1+B2) == f(A1,B1)+f(A1,B2); S = sum_t |A1|*(|B1|+|B2|) */
        for (int i=0;i<L;i++) for (int j=0;j<L;j++){ size_t k=(size_t)i*L+j; RHS[k]=FA1B1[k]+FA1B2[k];
            float S=0; for (int t=0;t<L;t++) S += fabsf(A1[i*L+t])*(fabsf(B1[t*L+j])+fabsf(B2[t*L+j])); Sb[k]=S; }
        bilin_one(FA1Bs, RHS, Sb, n, L, u, c_safe, "add_B", &bad, &maxrel);
        /* homogeneity in A: f(s*A1,B1) == s*f(A1,B1); S = |s| * sum_t |A1|*|B1| */
        for (int i=0;i<L;i++) for (int j=0;j<L;j++){ size_t k=(size_t)i*L+j; RHS[k]=s*FA1B1[k];
            float S=0; for (int t=0;t<L;t++) S += fabsf(A1[i*L+t])*fabsf(B1[t*L+j]); Sb[k]=fabsf(s)*S; }
        bilin_one(FsAB1, RHS, Sb, n, L, u, c_safe, "hom_A", &bad, &maxrel);
        /* homogeneity in B: f(A1,s*B1) == s*f(A1,B1); same S as hom_A */
        for (int i=0;i<L;i++) for (int j=0;j<L;j++){ size_t k=(size_t)i*L+j; RHS[k]=s*FA1B1[k];
            float S=0; for (int t=0;t<L;t++) S += fabsf(A1[i*L+t])*fabsf(B1[t*L+j]); Sb[k]=fabsf(s)*S; }
        bilin_one(FA1sB, RHS, Sb, n, L, u, c_safe, "hom_B", &bad, &maxrel);
        int sig_ok = (maxsig > 1.0f);                    /* non-vacuity: outputs are real signal, not ~0 */
        printf("matmul_bilin L=%d R=%g s=%g c_safe=%g: 4 laws x %zu cells, bad=%ld maxrel=%g maxsig=%g sig_ok=%d -> %s\n",
               L, R, s, c_safe, n, bad, maxrel, maxsig, sig_ok, (bad == 0 && sig_ok) ? "PASS" : "FAIL");
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(A1);free(A2);free(B1);free(B2);free(As);free(Bs);free(sA);free(sB);
        free(FA1B1);free(FA2B1);free(FA1B2);free(FAsB1);free(FA1Bs);free(FsAB1);free(FA1sB);free(RHS);free(Sb);free(ptx);
        return (bad == 0 && sig_ok) ? 0 : 1;
    }

    /* imatmul mode (v1.5 S0 increment 2b): INTEGER (ternary) matmul verify.
     *   cuda_launch <ptx> ternary_matmul <Nignored> imatmul <M> <K> <N> [mutate].
     * Mirrors the f32 'matmul' mode above but with INT32 buffers, so it exercises the
     * ternary_matmul kernel (a/b declared t2 -> ld.global.u32, mul.lo.s32, add.s32). A
     * (weights) is TERNARY {-1,0,+1}; B (activations) is small int. Integer sums are EXACT,
     * so the GPU result must EQUAL the CPU int reference bit-for-bit (NO tolerance -- the
     * point of ternary on the GPU). The optional "mutate" arg perturbs one C cell pre-compare
     * (comparator negative control -> the check MUST then FAIL, proving it has teeth). Launch
     * geometry matches naive/ternary: grid=(M,1,1) block=(N,1,1), one thread per output cell. */
    if (strcmp(op, "imatmul") == 0) {
        int Md = (argc > 5) ? atoi(argv[5]) : 16;
        int Kd = (argc > 6) ? atoi(argv[6]) : 16;
        int Nd = (argc > 7) ? atoi(argv[7]) : 16;
        int mutate = (argc > 8 && strcmp(argv[8], "mutate") == 0);
        size_t aN = (size_t)Md * Kd, bN = (size_t)Kd * Nd, cN = (size_t)Md * Nd;
        int* hA = (int*)malloc(aN * sizeof(int));
        int* hB = (int*)malloc(bN * sizeof(int));
        int* hC = (int*)malloc(cN * sizeof(int));
        if (!hA || !hB || !hC) return 2;
        /* NON-DEGENERATE data: a (i%3)-1 ternary weight against a simple b made the matmul
         * rank-1-collapsible (a wrong kernel could pass), so b uses a period-11 signed fill
         * coprime with the 3/5/16 strides -> a genuine matmul that a wrong kernel must fail. */
        for (size_t i = 0; i < aN; i++) hA[i] = (int)(i % 3) - 1;            /* ternary weights: -1/0/+1 */
        for (size_t i = 0; i < bN; i++) hB[i] = (int)((i * 13 + 7) % 11) - 5; /* signed int activations [-5,5] */
        CUdeviceptr dA, dB, dC;
        CK(cuMemAlloc(&dA, aN * sizeof(int)), "cuMemAlloc A");
        CK(cuMemAlloc(&dB, bN * sizeof(int)), "cuMemAlloc B");
        CK(cuMemAlloc(&dC, cN * sizeof(int)), "cuMemAlloc C");
        CK(cuMemcpyHtoD(dA, hA, aN * sizeof(int)), "cuMemcpyHtoD A");
        CK(cuMemcpyHtoD(dB, hB, bN * sizeof(int)), "cuMemcpyHtoD B");
        void* margs[] = { &dA, &dB, &dC, &Md, &Kd, &Nd };
        CK(cuLaunchKernel(fn, Md, 1, 1, Nd, 1, 1, 0, 0, margs, 0), "cuLaunchKernel imatmul");
        CK(cuCtxSynchronize(), "cuCtxSynchronize");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(int)), "cuMemcpyDtoH C");
        if (mutate && cN > 0) hC[0] += 1;   /* comparator negative control: must trip a mismatch */
        int mbad = 0;
        for (int r = 0; r < Md; r++) {
            for (int cc = 0; cc < Nd; cc++) {
                long ref = (long)hA[r * Kd] * (long)hB[cc];
                for (int t = 1; t < Kd; t++) ref += (long)hA[r * Kd + t] * (long)hB[t * Nd + cc];
                long got = (long)hC[r * Nd + cc];
                if (got != ref) { if (mbad < 4) fprintf(stderr, "imatmul mismatch C[%d,%d]=%ld ref %ld\n", r, cc, got, ref); mbad++; }
            }
        }
        printf("GPU [%s] ternary_matmul(int) %dx%dx%d over %d cells: C[1,1]=%d, %d bad -> %s\n",
               gpu, Md, Kd, Nd, Md * Nd, hC[1 * Nd + 1], mbad, mbad ? "FAIL" : "PASS");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hC); free(ptx);
        return mbad ? 1 : 0;
    }

    /* v1.5 #4 STEP 2: receipt_emit -- run the GENUINE kovc-emitted ternary_matmul (fn, the #2-certified
     * kernel) on deterministic W (ternary) / X (int), read back C, and write a verifiable-inference RECEIPT
     * (commitments H_W,H_X,H_C + a diagnostic round echo the checker ignores) plus a raw C-bytes side-file.
     * Reuses imatmul's launch geometry (grid=M, block=N). Usage:
     *   cuda_launch <ternary.ptx> ternary_matmul <Nignored> receipt_emit <M> <K> <N> <receiptpath>. */
    if (strcmp(op, "receipt_emit") == 0) {
        int Md = (argc > 5) ? atoi(argv[5]) : 16, Kd = (argc > 6) ? atoi(argv[6]) : 16, Nd = (argc > 7) ? atoi(argv[7]) : 16;
        const char* rpath = (argc > 8) ? argv[8] : "/tmp/helix_receipt.txt";
        int mutate = (argc > 9 && strcmp(argv[9],"mutate")==0);  /* NC: emit a SELF-CONSISTENT forged receipt */
        long long p = 2147483647LL; int t = 2; unsigned uW = 0, uX = 0;
        size_t nW=(size_t)Md*Kd, nX=(size_t)Kd*Nd, nC=(size_t)Md*Nd;
        int* W=(int*)malloc(nW*sizeof(int)); int* X=(int*)malloc(nX*sizeof(int)); int* C=(int*)malloc(nC*sizeof(int));
        if (!W||!X||!C) return 2;
        receipt_gen_W(W,nW,uW); receipt_gen_X(X,nX,uX);
        CUdeviceptr dA,dB,dC;
        CK(cuMemAlloc(&dA,nW*sizeof(int)),"emit A"); CK(cuMemAlloc(&dB,nX*sizeof(int)),"emit B"); CK(cuMemAlloc(&dC,nC*sizeof(int)),"emit C");
        CK(cuMemcpyHtoD(dA,W,nW*sizeof(int)),"emit HtoD A"); CK(cuMemcpyHtoD(dB,X,nX*sizeof(int)),"emit HtoD B");
        void* eargs[]={ &dA,&dB,&dC,&Md,&Kd,&Nd };
        CK(cuLaunchKernel(fn,Md,1,1,Nd,1,1,0,0,eargs,0),"emit launch"); CK(cuCtxSynchronize(),"emit sync");
        CK(cuMemcpyDtoH(C,dC,nC*sizeof(int)),"emit DtoH");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
        if (mutate && nC > 0) C[0] += 1;   /* forge ONE output cell -> H_C/cbytes/echo all still self-consistent, but C != W*X, so the Freivalds leg MUST reject (the soundness teeth, mirrors imatmul's mutate NC) */
        unsigned char dWg[32],dXg[32],dCg[32]; char xW[65],xX[65],xC[65];
        sha256((unsigned char*)W,nW*sizeof(int),dWg); sha256_hex(dWg,xW);
        sha256((unsigned char*)X,nX*sizeof(int),dXg); sha256_hex(dXg,xX);
        sha256((unsigned char*)C,nC*sizeof(int),dCg); sha256_hex(dCg,xC);
        char cpath[600]; snprintf(cpath,sizeof cpath,"%s.cbytes",rpath);
        FILE* cfp=fopen(cpath,"wb"); if(!cfp){ fprintf(stderr,"receipt_emit: cannot write %s\n",cpath); free(W);free(X);free(C); return 2; }
        fwrite(C,1,nC*sizeof(int),cfp); fclose(cfp);
        FILE* rfp=fopen(rpath,"w"); if(!rfp){ fprintf(stderr,"receipt_emit: cannot write %s\n",rpath); free(W);free(X);free(C); return 2; }
        fprintf(rfp,"HELIX_RECEIPT_V1\nshape M=%d K=%d N=%d\nprime p=%lld\nrounds t=%d\nseed_W=%u\nseed_X=%u\nH_W=%s\nH_X=%s\nH_C=%s\n",
                Md,Kd,Nd,p,t,uW,uX,xW,xX,xC);
        for (int rd=0; rd<t; rd++) {
            long long* r=(long long*)malloc((size_t)Nd*sizeof(long long)); long long* Br=(long long*)malloc((size_t)Kd*sizeof(long long));
            long long lhsf=0,rhsf=0;
            for (int j=0;j<Nd;j++) r[j]=receipt_fs_r(dWg,dXg,dCg,rd,j,p);
            for (int k=0;k<Kd;k++){ long long s=0; for(int j=0;j<Nd;j++){ long long xv=(((long long)X[(size_t)k*Nd+j])%p+p)%p; s=(s+xv*r[j])%p; } Br[k]=s; }
            for (int i=0;i<Md;i++){ long long lh=0; for(int k=0;k<Kd;k++){ long long wv=(((long long)W[(size_t)i*Kd+k])%p+p)%p; lh=(lh+wv*Br[k])%p; } lhsf=(lhsf+lh*(i+1))%p;
                                    long long rh=0; for(int j=0;j<Nd;j++){ long long cv=(((long long)C[(size_t)i*Nd+j])%p+p)%p; rh=(rh+cv*r[j])%p; } rhsf=(rhsf+rh*(i+1))%p; }
            fprintf(rfp,"round %d lhs=%lld rhs=%lld\n",rd,lhsf,rhsf);
            free(r); free(Br);
        }
        fclose(rfp);
        printf("receipt_emit [%s]: ternary_matmul %dx%dx%d committed (H_C=%.16s...) -> RECEIPT_EMITTED\n", rpath, Md,Kd,Nd, xC);
        free(W); free(X); free(C);
        cuModuleUnload(mod); cuCtxDestroy(ctx); free(ptx);
        return 0;
    }

    /* ptmatmul mode (v1.5 S0 increment 3): PACKED TERNARY matmul verify.
     *   cuda_launch <ptx> packed_ternary_matmul <Nignored> ptmatmul <M> <K> <N> [mutate].
     * Like 'imatmul' but the weights are 2-bit PACKED: 15 trits per i32 word, base-4 code
     * (trit -1/0/+1 -> code 2/0/1; word = sum_{j=0..14} code_j*4^j). 15 (NOT 16) fields keep
     * the word < 2^31 so the device's SIGNED div.s32 unpack stays exact (16 fields would spill
     * into the sign bit -> mis-decode). K MUST be divisible by 15. Host packs W -> M*(K/15)
     * words; the kernel unpacks ON DEVICE (code=w-(w/4)*4; trit=code-3*(code/2)). EXACT integer
     * compare vs the UNPACKED CPU reference; 'mutate' is the comparator NC; the packed-vs-unpacked
     * footprint (15x) is reported. */
    if (strcmp(op, "ptmatmul") == 0) {
        int Md = (argc > 5) ? atoi(argv[5]) : 16;
        int Kd = (argc > 6) ? atoi(argv[6]) : 15;
        int Nd = (argc > 7) ? atoi(argv[7]) : 16;
        int mutate = (argc > 8 && strcmp(argv[8], "mutate") == 0);
        if (Kd % 15 != 0) { fprintf(stderr, "ptmatmul: K must be divisible by 15 (15 trits/word); got K=%d\n", Kd); return 2; }
        int kpacked = Kd / 15;
        size_t wN = (size_t)Md * Kd;       /* unpacked weights (for the pack + the CPU ref) */
        size_t aP = (size_t)Md * kpacked;  /* packed words (the device weight buffer) */
        size_t bN = (size_t)Kd * Nd, cN = (size_t)Md * Nd;
        int* hW = (int*)malloc(wN * sizeof(int));
        int* hA = (int*)malloc(aP * sizeof(int));
        int* hB = (int*)malloc(bN * sizeof(int));
        int* hC = (int*)malloc(cN * sizeof(int));
        if (!hW || !hA || !hB || !hC) return 2;
        for (size_t i = 0; i < wN; i++) hW[i] = (int)(i % 3) - 1;             /* ternary weights -1/0/+1 */
        for (int r = 0; r < Md; r++) {
            for (int kw = 0; kw < kpacked; kw++) {
                int word = 0, power = 1;
                for (int j = 0; j < 15; j++) {
                    int w = hW[r * Kd + kw * 15 + j];
                    int code = (w < 0) ? 2 : (w > 0 ? 1 : 0);   /* -1->2, 0->0, +1->1 */
                    word += code * power;
                    power *= 4;
                }
                hA[r * kpacked + kw] = word;
            }
        }
        for (size_t i = 0; i < bN; i++) hB[i] = (int)((i * 13 + 7) % 11) - 5; /* signed activations [-5,5] */
        CUdeviceptr dA, dB, dC;
        CK(cuMemAlloc(&dA, aP * sizeof(int)), "cuMemAlloc A (packed)");
        CK(cuMemAlloc(&dB, bN * sizeof(int)), "cuMemAlloc B");
        CK(cuMemAlloc(&dC, cN * sizeof(int)), "cuMemAlloc C");
        CK(cuMemcpyHtoD(dA, hA, aP * sizeof(int)), "cuMemcpyHtoD A (packed)");
        CK(cuMemcpyHtoD(dB, hB, bN * sizeof(int)), "cuMemcpyHtoD B");
        void* pargs[] = { &dA, &dB, &dC, &Md, &Kd, &Nd };
        CK(cuLaunchKernel(fn, Md, 1, 1, Nd, 1, 1, 0, 0, pargs, 0), "cuLaunchKernel ptmatmul");
        CK(cuCtxSynchronize(), "cuCtxSynchronize");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(int)), "cuMemcpyDtoH C");
        if (mutate && cN > 0) hC[0] += 1;   /* comparator negative control: must trip a mismatch */
        int pbad = 0;
        for (int r = 0; r < Md; r++) {
            for (int cc = 0; cc < Nd; cc++) {
                long ref = (long)hW[r * Kd] * (long)hB[cc];
                for (int t = 1; t < Kd; t++) ref += (long)hW[r * Kd + t] * (long)hB[t * Nd + cc];
                long got = (long)hC[r * Nd + cc];
                if (got != ref) { if (pbad < 4) fprintf(stderr, "ptmatmul mismatch C[%d,%d]=%ld ref %ld\n", r, cc, got, ref); pbad++; }
            }
        }
        printf("GPU [%s] packed_ternary_matmul(15t/word) %dx%dx%d: C[1,1]=%d, %d bad -> %s | footprint: packed_a=%zuB vs unpacked=%zuB (%.1fx)\n",
               gpu, Md, Kd, Nd, hC[1 * Nd + 1], pbad, pbad ? "FAIL" : "PASS",
               aP * sizeof(int), wN * sizeof(int), (double)wN / (double)aP);
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hW); free(hA); free(hB); free(hC); free(ptx);
        return pbad ? 1 : 0;
    }

    /* sptmatmul mode (v1.9 P1): SCALED PACKED TERNARY matmul verify.
     *   cuda_launch <ptx> scaled_packed_ternary_matmul <Nignored> sptmatmul <M> <K> <N> [mutate].
     * Like 'ptmatmul' (15 trits/word packed, on-device div-unpack, exact i32 accumulate) but the
     * kernel applies a per-OUTPUT-ROW f32 scale: c[r,col] = i2f(int_dot) * sc[r] (the BitNet dequant
     * shape). Output is f32. The host reference is (float)int_ref * scale[r] -- the SAME op order as
     * the kernel (__gpu_i2f then one mul.f32); the int accumulate is small so __gpu_i2f is exact and
     * there is no FMA, so the GPU f32 result is BIT-IDENTICAL to the host -> the compare is EXACT (==),
     * same teeth as ptmatmul. 'mutate' perturbs one output cell pre-compare (comparator NC). */
    if (strcmp(op, "sptmatmul") == 0) {
        int Md = (argc > 5) ? atoi(argv[5]) : 16;
        int Kd = (argc > 6) ? atoi(argv[6]) : 15;
        int Nd = (argc > 7) ? atoi(argv[7]) : 16;
        int mutate = (argc > 8 && strcmp(argv[8], "mutate") == 0);
        if (Kd % 15 != 0) { fprintf(stderr, "sptmatmul: K must be divisible by 15 (15 trits/word); got K=%d\n", Kd); return 2; }
        int kpacked = Kd / 15;
        size_t wN = (size_t)Md * Kd, aP = (size_t)Md * kpacked;
        size_t bN = (size_t)Kd * Nd, cN = (size_t)Md * Nd;
        int*   hW = (int*)malloc(wN * sizeof(int));
        int*   hA = (int*)malloc(aP * sizeof(int));
        int*   hB = (int*)malloc(bN * sizeof(int));
        float* hS = (float*)malloc((size_t)Md * sizeof(float));
        float* hC = (float*)malloc(cN * sizeof(float));
        if (!hW || !hA || !hB || !hS || !hC) return 2;
        for (size_t i = 0; i < wN; i++) hW[i] = (int)(i % 3) - 1;             /* ternary weights -1/0/+1 */
        for (int r = 0; r < Md; r++) {
            for (int kw = 0; kw < kpacked; kw++) {
                int word = 0, power = 1;
                for (int j = 0; j < 15; j++) {
                    int w = hW[r * Kd + kw * 15 + j];
                    int code = (w < 0) ? 2 : (w > 0 ? 1 : 0);                 /* -1->2, 0->0, +1->1 */
                    word += code * power; power *= 4;
                }
                hA[r * kpacked + kw] = word;
            }
        }
        for (size_t i = 0; i < bN; i++) hB[i] = (int)((i * 13 + 7) % 11) - 5; /* signed activations [-5,5] */
        for (int r = 0; r < Md; r++) hS[r] = 0.013125f * (float)(r + 1);      /* per-row scale, distinct + non-f32-exact */
        CUdeviceptr dA, dB, dS, dC;
        CK(cuMemAlloc(&dA, aP * sizeof(int)),          "cuMemAlloc A (packed)");
        CK(cuMemAlloc(&dB, bN * sizeof(int)),          "cuMemAlloc B");
        CK(cuMemAlloc(&dS, (size_t)Md * sizeof(float)), "cuMemAlloc S (scale)");
        CK(cuMemAlloc(&dC, cN * sizeof(float)),        "cuMemAlloc C (f32)");
        CK(cuMemcpyHtoD(dA, hA, aP * sizeof(int)),          "cuMemcpyHtoD A");
        CK(cuMemcpyHtoD(dB, hB, bN * sizeof(int)),          "cuMemcpyHtoD B");
        CK(cuMemcpyHtoD(dS, hS, (size_t)Md * sizeof(float)), "cuMemcpyHtoD S");
        void* pargs[] = { &dA, &dB, &dS, &dC, &Md, &Kd, &Nd };
        CK(cuLaunchKernel(fn, Md, 1, 1, Nd, 1, 1, 0, 0, pargs, 0), "cuLaunchKernel sptmatmul");
        CK(cuCtxSynchronize(), "cuCtxSynchronize");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(float)), "cuMemcpyDtoH C (f32)");
        if (mutate && cN > 0) hC[0] += 1.0f;   /* comparator negative control: must trip a mismatch */
        int pbad = 0;
        for (int r = 0; r < Md; r++) {
            for (int cc = 0; cc < Nd; cc++) {
                long iref = (long)hW[r * Kd] * (long)hB[cc];
                for (int t = 1; t < Kd; t++) iref += (long)hW[r * Kd + t] * (long)hB[t * Nd + cc];
                float ref = (float)iref * hS[r];        /* i2f(int) * scale[r] -- identical op to the kernel */
                float got = hC[r * Nd + cc];
                if (got != ref) { if (pbad < 4) fprintf(stderr, "sptmatmul mismatch C[%d,%d]=%.9g ref %.9g (d=%.3g)\n", r, cc, got, ref, got - ref); pbad++; }
            }
        }
        printf("GPU [%s] scaled_packed_ternary_matmul(15t/word + per-row f32 scale) %dx%dx%d: C[1,1]=%.6g, %d bad -> %s\n",
               gpu, Md, Kd, Nd, hC[1 * Nd + 1], pbad, pbad ? "FAIL" : "PASS");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dS); cuMemFree(dC);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hW); free(hA); free(hB); free(hS); free(hC); free(ptx);
        return pbad ? 1 : 0;
    }

    /* sptmatmul_real mode (v1.9 P3c): the SAME scaled_packed_ternary_matmul kernel, but on REAL BitNet
     * data loaded from .bin files instead of synthetic.
     *   cuda_launch <ptx> scaled_packed_ternary_matmul <Nignored> sptmatmul_real <packed.bin> <acts.bin> <expected.bin> <M> <K> <N> [mutate].
     * packed.bin = host-packed ternary weights (i32 [M x K/15], 15 trits/word); acts.bin = signed int
     * activations (i32 [K x N]); expected.bin = the reference INTEGER matmul (f32 [M x N], = python
     * a_int @ W_ternary.T). Launches with sc=ALL-ONES so c = i2f(int_dot) is the pure integer result,
     * and compares c == expected EXACTLY (==). Proves the kovc-emitted ternary kernel reproduces a REAL
     * BitNet BitLinear's integer matmul element-for-element (the per-tensor/per-token dequant scales are
     * applied host-side downstream, matching the verified P3c-step1 decomposition). 'mutate' = comparator NC. */
    if (strcmp(op, "sptmatmul_real") == 0) {
        const char* pf = (argc > 5) ? argv[5] : 0;
        const char* af = (argc > 6) ? argv[6] : 0;
        const char* ef = (argc > 7) ? argv[7] : 0;
        int Md = (argc > 8) ? atoi(argv[8]) : 0;
        int Kd = (argc > 9) ? atoi(argv[9]) : 0;
        int Nd = (argc > 10) ? atoi(argv[10]) : 0;
        int mutate = (argc > 11 && strcmp(argv[11], "mutate") == 0);
        if (!pf || !af || !ef || Md <= 0 || Kd <= 0 || Nd <= 0) { fprintf(stderr, "sptmatmul_real: need <packed.bin> <acts.bin> <expected.bin> M K N\n"); return 2; }
        if (Kd % 15 != 0) { fprintf(stderr, "sptmatmul_real: K must be divisible by 15 (15 trits/word); got K=%d\n", Kd); return 2; }
        int kpacked = Kd / 15;
        size_t aP = (size_t)Md * kpacked, bN = (size_t)Kd * Nd, cN = (size_t)Md * Nd;
        int*   hA = (int*)malloc(aP * sizeof(int));
        int*   hB = (int*)malloc(bN * sizeof(int));
        float* hS = (float*)malloc((size_t)Md * sizeof(float));
        float* hC = (float*)malloc(cN * sizeof(float));
        float* hE = (float*)malloc(cN * sizeof(float));
        if (!hA || !hB || !hS || !hC || !hE) return 2;
        FILE* fp;
        fp = fopen(pf, "rb"); if (!fp || fread(hA, sizeof(int),   aP, fp) != aP) { fprintf(stderr, "sptmatmul_real: bad packed.bin (want %zu i32)\n", aP); return 2; } fclose(fp);
        fp = fopen(af, "rb"); if (!fp || fread(hB, sizeof(int),   bN, fp) != bN) { fprintf(stderr, "sptmatmul_real: bad acts.bin (want %zu i32)\n", bN); return 2; } fclose(fp);
        fp = fopen(ef, "rb"); if (!fp || fread(hE, sizeof(float), cN, fp) != cN) { fprintf(stderr, "sptmatmul_real: bad expected.bin (want %zu f32)\n", cN); return 2; } fclose(fp);
        for (int r = 0; r < Md; r++) hS[r] = 1.0f;   /* sc = all-ones -> c = i2f(int_dot), the pure integer matmul */
        CUdeviceptr dA, dB, dS, dC;
        CK(cuMemAlloc(&dA, aP * sizeof(int)),           "cuMemAlloc A (packed)");
        CK(cuMemAlloc(&dB, bN * sizeof(int)),           "cuMemAlloc B");
        CK(cuMemAlloc(&dS, (size_t)Md * sizeof(float)), "cuMemAlloc S (scale)");
        CK(cuMemAlloc(&dC, cN * sizeof(float)),         "cuMemAlloc C (f32)");
        CK(cuMemcpyHtoD(dA, hA, aP * sizeof(int)),           "cuMemcpyHtoD A");
        CK(cuMemcpyHtoD(dB, hB, bN * sizeof(int)),           "cuMemcpyHtoD B");
        CK(cuMemcpyHtoD(dS, hS, (size_t)Md * sizeof(float)), "cuMemcpyHtoD S");
        void* pargs[] = { &dA, &dB, &dS, &dC, &Md, &Kd, &Nd };
        CK(cuLaunchKernel(fn, Md, 1, 1, Nd, 1, 1, 0, 0, pargs, 0), "cuLaunchKernel sptmatmul_real");
        CK(cuCtxSynchronize(), "cuCtxSynchronize");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(float)), "cuMemcpyDtoH C (f32)");
        if (mutate && cN > 0) hC[0] += 1.0f;   /* comparator negative control: must trip a mismatch */
        int pbad = 0;
        for (size_t i = 0; i < cN; i++) {
            if (hC[i] != hE[i]) { if (pbad < 4) fprintf(stderr, "sptmatmul_real mismatch C[%zu]=%.9g exp %.9g (d=%.3g)\n", i, hC[i], hE[i], hC[i] - hE[i]); pbad++; }
        }
        printf("GPU [%s] scaled_packed_ternary_matmul on REAL BitNet data %dx%dx%d: C[0]=%.6g exp %.6g, %d bad -> %s\n",
               gpu, Md, Kd, Nd, hC[0], hE[0], pbad, pbad ? "FAIL" : "PASS");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dS); cuMemFree(dC);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hS); free(hC); free(hE); free(ptx);
        return pbad ? 1 : 0;
    }

    /* hgemm mode (v1.5 S1): HALF-PRECISION (f16 storage, f32 accumulate) matmul verify.
     *   cuda_launch <ptx> naive_matmul_f16 <Nignored> hgemm <M> <K> <N> [mutate].
     * Device buffers are uint16 IEEE-binary16 -- exactly what the kernel's
     * ld.global.b16 / cvt.f32.f16 ... cvt.rn.f16.f32 / st.global.b16 path reads/writes.
     * Host generates NON-DEGENERATE data (period-coprime fills so a rank-collapsed wrong
     * kernel fails) at a NON-f16-exact scale (x0.1 / x0.3 -> neither 0.1 nor 0.3 is exactly
     * representable in binary16), rounds it to f16, and up-converts the SAME f16 bits
     * (refA/refB) for the f32-accum reference -- so oracle + kernel agree on the input
     * EXACTLY (both consume the identical rounded f16 value; input agreement is exact
     * regardless of the original scale). The GPU may contract mul+add into FMA and the
     * RESULT is f16-rounded, so genuine (nonzero) error appears and the compare is
     * DUAL-bound tolerance: a cell is OK if within EITHER abs 1e-3 OR rel 1e-2 (small refs blow up
     * rel, large refs blow up abs -- so "bad" needs BOTH exceeded). 'mutate' perturbs one
     * output cell by a MAGNITUDE-SCALED amount (0.5 + 0.1*|got|) guaranteed to exceed both
     * bounds at any magnitude -> the comparator NC then MUST FAIL (a bare +1 is vacuous on
     * large cells). Launch geometry matches naive/ternary: grid=(M,1,1) block=(N,1,1). */
    if (strcmp(op, "hgemm") == 0) {
        int Md = (argc > 5) ? atoi(argv[5]) : 16;
        int Kd = (argc > 6) ? atoi(argv[6]) : 16;
        int Nd = (argc > 7) ? atoi(argv[7]) : 16;
        int mutate = (argc > 8 && strcmp(argv[8], "mutate") == 0);
        size_t aN = (size_t)Md * Kd, bN = (size_t)Kd * Nd, cN = (size_t)Md * Nd;
        unsigned short* hA = (unsigned short*)malloc(aN * sizeof(unsigned short));
        unsigned short* hB = (unsigned short*)malloc(bN * sizeof(unsigned short));
        unsigned short* hC = (unsigned short*)malloc(cN * sizeof(unsigned short));
        float* refA = (float*)malloc(aN * sizeof(float));   /* f16-rounded inputs, up-converted */
        float* refB = (float*)malloc(bN * sizeof(float));
        if (!hA || !hB || !hC || !refA || !refB) return 2;
        for (size_t i = 0; i < aN; i++) { float v = ((float)((i * 7 + 3) % 13) - 6.0f) * 0.1f; hA[i] = f32_to_f16(v); refA[i] = f16_to_f32(hA[i]); }
        for (size_t i = 0; i < bN; i++) { float v = ((float)((i * 5 + 1) % 11) - 5.0f) * 0.3f; hB[i] = f32_to_f16(v); refB[i] = f16_to_f32(hB[i]); }
        CUdeviceptr dA, dB, dC;
        CK(cuMemAlloc(&dA, aN * sizeof(unsigned short)), "cuMemAlloc A (f16)");
        CK(cuMemAlloc(&dB, bN * sizeof(unsigned short)), "cuMemAlloc B (f16)");
        CK(cuMemAlloc(&dC, cN * sizeof(unsigned short)), "cuMemAlloc C (f16)");
        CK(cuMemcpyHtoD(dA, hA, aN * sizeof(unsigned short)), "cuMemcpyHtoD A (f16)");
        CK(cuMemcpyHtoD(dB, hB, bN * sizeof(unsigned short)), "cuMemcpyHtoD B (f16)");
        void* hargs[] = { &dA, &dB, &dC, &Md, &Kd, &Nd };
        CK(cuLaunchKernel(fn, Md, 1, 1, Nd, 1, 1, 0, 0, hargs, 0), "cuLaunchKernel hgemm");
        CK(cuCtxSynchronize(), "cuCtxSynchronize");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(unsigned short)), "cuMemcpyDtoH C (f16)");
        int hbad = 0; float maxabs = 0.0f, maxrel = 0.0f;
        for (int r = 0; r < Md; r++) {
            for (int cc = 0; cc < Nd; cc++) {
                float ref = refA[r * Kd] * refB[cc];
                for (int t = 1; t < Kd; t++) ref += refA[r * Kd + t] * refB[t * Nd + cc];
                float got = f16_to_f32(hC[r * Nd + cc]);
                if (mutate && r == 0 && cc == 0) { got += 0.5f + 0.1f * fabsf(got); } /* magnitude-scaled NC */
                float e = got - ref; if (e < 0) e = -e;
                float denom = fabsf(ref); float rel = denom > 1.0e-6f ? e / denom : e;
                if (e > maxabs) maxabs = e;
                if (rel > maxrel) maxrel = rel;
                if (isnan(got) || (e > 1.0e-3f && rel > 1.0e-2f)) {
                    if (hbad < 4) fprintf(stderr, "hgemm mismatch C[%d,%d]=%g ref %g (abs %g rel %g)\n", r, cc, got, ref, e, rel);
                    hbad++;
                }
            }
        }
        printf("GPU [%s] naive_matmul_f16(f16-IO/f32-acc) %dx%dx%d over %d cells: C[1,1]=%g, max_abs=%g max_rel=%g, %d bad -> %s\n",
               gpu, Md, Kd, Nd, Md * Nd, f16_to_f32(hC[1 * Nd + 1]), maxabs, maxrel, hbad, hbad ? "FAIL" : "PASS");
        int rc = hbad ? 1 : 0;
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hC); free(refA); free(refB); free(ptx);
        return rc;
    }

    /* mxfp4 mode (v1.5 S2): MXFP4 (OCP) dequant -> f16 matmul verify.
     *   cuda_launch <ptx> naive_mxfp4_matmul <Nignored> mxfp4 <M> <K> <N> [mutate|wflip].
     * Weights are MXFP4: E2M1 4-bit codes packed 7/i32 word (base-16 low-nibble-first; 7 NOT 8 keeps
     * the word < 2^31 so the kernel's signed div.s32 nibble-unpack is exact) + a shared E8M0 scale per
     * 32-element block, HOST-decoded to a linear f32 'sc' buffer (no __gpu_exp2 on the @kernel path).
     * The device kernel dequants ON-DEVICE (div-unpack the nibble, f32-literal E2M1 decode, * the f32
     * scale) and matmuls in f32, storing f16. K must be a multiple of 224 = LCM(7,32). Compared vs an
     * INDEPENDENT CPU dequant+matmul reference (SAME codes, SAME f32 scales, SAME f16-rounded b) within
     * DUAL-bound tolerance (abs 1e-3 OR rel 1e-2; the dequant is f32-EXACT, the only error is the f16 b
     * input + the f16 output round). NCs: "mutate" = magnitude-scaled comparator (got += 0.5+0.1*|got|,
     * MUST fail at any magnitude); "wflip" = a packed-WEIGHT nibble bump (the GPU reads a changed weight
     * while the oracle keeps the original codes -> MUST fail, proving the weights are load-bearing). The
     * kernel-corruption NC is driven externally by gpu_mxfp4_check.sh [E]. Measured footprint vs f32
     * AND f16 reported (the 7-not-8 sign-safety cap is in the measured bytes -- honest, not theoretical). */
    if (strcmp(op, "mxfp4") == 0) {
        int Md = (argc > 5) ? atoi(argv[5]) : 8;
        int Kd = (argc > 6) ? atoi(argv[6]) : 224;
        int Nd = (argc > 7) ? atoi(argv[7]) : 8;
        int mutate = (argc > 8 && strcmp(argv[8], "mutate") == 0);
        int wflip  = (argc > 8 && strcmp(argv[8], "wflip") == 0);
        if (Kd % 224 != 0) { fprintf(stderr, "mxfp4: K must be a multiple of 224 = LCM(7,32); got K=%d\n", Kd); return 2; }
        int kwords = Kd / 7;     /* packed i32 words per row */
        int kblk   = Kd / 32;    /* E8M0 scale blocks per row */
        size_t wN = (size_t)Md * kwords, scN = (size_t)Md * kblk, bN = (size_t)Kd * Nd, cN = (size_t)Md * Nd;
        int* hCode = (int*)malloc((size_t)Md * Kd * sizeof(int));   /* original E2M1 codes (for the oracle) */
        int* hW = (int*)malloc(wN * sizeof(int));                   /* packed words (device weight buffer) */
        float* hSc = (float*)malloc(scN * sizeof(float));           /* linear f32 block scales */
        unsigned short* hB = (unsigned short*)malloc(bN * sizeof(unsigned short));
        float* refB = (float*)malloc(bN * sizeof(float));
        unsigned short* hC = (unsigned short*)malloc(cN * sizeof(unsigned short));
        if (!hCode || !hW || !hSc || !hB || !refB || !hC) return 2;
        /* NON-DEGENERATE weight codes (period-coprime fill spans all 16 magnitudes/signs) */
        for (int r = 0; r < Md; r++)
            for (int k = 0; k < Kd; k++)
                hCode[r * Kd + k] = (r * 5 + k * 7 + 3) % 16;
        /* pack 7 codes/word base-16 low-nibble-first: word = sum_{j=0..6} code_{kw*7+j} * 16^j */
        for (int r = 0; r < Md; r++)
            for (int kw = 0; kw < kwords; kw++) {
                int word = 0, pw = 1;
                for (int j = 0; j < 7; j++) { word += (hCode[r * Kd + kw * 7 + j] & 15) * pw; pw *= 16; }
                hW[r * kwords + kw] = word;
            }
        /* E8M0 scales near 127 (0.5..2.0), host-decoded to linear f32 */
        for (int r = 0; r < Md; r++)
            for (int bk = 0; bk < kblk; bk++)
                hSc[r * kblk + bk] = e8m0_scale(126 + ((r + bk) % 3));   /* 126,127,128 -> 0.5,1,2 */
        /* activations b: non-degenerate f16-representable, f16-rounded (refB shares the rounded bits) */
        for (size_t i = 0; i < bN; i++) { float v = ((float)((i * 5 + 1) % 11) - 5.0f) * 0.3f; hB[i] = f32_to_f16(v); refB[i] = f16_to_f32(hB[i]); }
        /* wflip NC: bump element-0's nibble (the GPU sees a changed weight; the oracle keeps hCode).
         * The fixed data has code[0]=3 (1.5) -> 4 (2.0): a real value change, in MOST C[0,*] sums
         * (b[k=0,cc=3]=0 zeros that one cell's delta; the NC only needs >=1 of the 8 cells to fail). */
        if (wflip) { int n0 = hW[0] & 15; hW[0] = hW[0] - n0 + ((n0 + 1) & 15); }
        CUdeviceptr dW, dSc, dB, dC;
        CK(cuMemAlloc(&dW, wN * sizeof(int)), "cuMemAlloc W (mxfp4 packed)");
        CK(cuMemAlloc(&dSc, scN * sizeof(float)), "cuMemAlloc Sc (e8m0->f32)");
        CK(cuMemAlloc(&dB, bN * sizeof(unsigned short)), "cuMemAlloc B (f16)");
        CK(cuMemAlloc(&dC, cN * sizeof(unsigned short)), "cuMemAlloc C (f16)");
        CK(cuMemcpyHtoD(dW, hW, wN * sizeof(int)), "HtoD W");
        CK(cuMemcpyHtoD(dSc, hSc, scN * sizeof(float)), "HtoD Sc");
        CK(cuMemcpyHtoD(dB, hB, bN * sizeof(unsigned short)), "HtoD B");
        void* margs[] = { &dW, &dSc, &dB, &dC, &Md, &Kd, &Nd };
        CK(cuLaunchKernel(fn, Md, 1, 1, Nd, 1, 1, 0, 0, margs, 0), "cuLaunchKernel mxfp4");
        CK(cuCtxSynchronize(), "cuCtxSynchronize");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(unsigned short)), "DtoH C");
        int xbad = 0; float maxabs = 0.0f, maxrel = 0.0f;
        for (int r = 0; r < Md; r++) {
            for (int cc = 0; cc < Nd; cc++) {
                float ref = 0.0f;
                for (int k = 0; k < Kd; k++)
                    ref += e2m1_decode(hCode[r * Kd + k]) * hSc[r * kblk + k / 32] * refB[k * Nd + cc];
                float got = f16_to_f32(hC[r * Nd + cc]);
                if (mutate && r == 0 && cc == 0) { got += 0.5f + 0.1f * fabsf(got); }
                float e = got - ref; if (e < 0) e = -e;
                float denom = fabsf(ref); float rel = denom > 1.0e-6f ? e / denom : e;
                if (e > maxabs) maxabs = e;
                if (rel > maxrel) maxrel = rel;
                if (isnan(got) || (e > 1.0e-3f && rel > 1.0e-2f)) {
                    if (xbad < 4) fprintf(stderr, "mxfp4 mismatch C[%d,%d]=%g ref %g (abs %g rel %g)\n", r, cc, got, ref, e, rel);
                    xbad++;
                }
            }
        }
        double mxbytes = (double)wN * 4.0 + (double)scN * 1.0;       /* STORAGE footprint: packed words (4B each) + the E8M0 scale at its native 1B. (The device sc buffer is f32 only as a runtime dequant artifact -- no __gpu_exp2 on the @kernel path -- NOT the stored size; the storage win is what is claimed.) */
        double f32b = (double)Md * Kd * 4.0, f16b = (double)Md * Kd * 2.0;
        printf("GPU [%s] naive_mxfp4_matmul(dequant->f32-acc->f16) %dx%dx%d over %d cells: C[1,1]=%g, max_abs=%g max_rel=%g, %d bad -> %s | footprint(measured): mxfp4=%.0fB vs f32=%.0fB (%.2fx) vs f16=%.0fB (%.2fx)\n",
               gpu, Md, Kd, Nd, Md * Nd, f16_to_f32(hC[1 * Nd + 1]), maxabs, maxrel, xbad, xbad ? "FAIL" : "PASS",
               mxbytes, f32b, f32b / mxbytes, f16b, f16b / mxbytes);
        int rc = xbad ? 1 : 0;
        cuMemFree(dW); cuMemFree(dSc); cuMemFree(dB); cuMemFree(dC);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hCode); free(hW); free(hSc); free(hB); free(refB); free(hC); free(ptx);
        return rc;
    }

    /* nvfp4 mode (v1.5 S3): NVFP4 (OCP/NVIDIA) two-level-scaled 4-bit DEQUANT verify (NOT a matmul --
     * the DoD S3 test is "verified dequant"; native FP4 MMA needs Blackwell, DEFERRED+labeled).
     *   cuda_launch <ptx> nvfp4_dequant <Nignored> nvfp4 <M> <K> <N> [mutate|wflip|sflip].
     * Weights are E2M1 (reused from S2) packed 7/i32 word; the scale is TWO-level: an FP8 E4M3 micro
     * per 16-block + an FP32 per-tensor, HOST-collapsed to ONE effective f32/16-block (the device just
     * reads it -- the S2 E8M0 pattern). The device dequants ON-DEVICE (div-unpack + the f32-literal
     * E2M1 if-ladder + mag*scale) writing f32, so the result is f32-EXACT vs the CPU oracle (tight tol:
     * abs 1e-5 OR rel 1e-6 -- the dequant is a single f32 mul of identical operands both sides). K must
     * be a multiple of 112 = LCM(7,16). NCs: "mutate" = magnitude-scaled comparator; "wflip" = a
     * packed-weight nibble bump (the ELEMENT is load-bearing); "sflip" = a per-block effective-scale
     * bump (the TWO-LEVEL SCALE is load-bearing -- S3's novelty). The kernel-corruption NC is external
     * (gpu_nvfp4_check.sh). Measured footprint vs f32 AND f16 reported. */
    if (strcmp(op, "nvfp4") == 0) {
        int Md = (argc > 5) ? atoi(argv[5]) : 8;
        int Kd = (argc > 6) ? atoi(argv[6]) : 112;
        int Nd = (argc > 7) ? atoi(argv[7]) : 1;
        int mutate = (argc > 8 && strcmp(argv[8], "mutate") == 0);
        int wflip  = (argc > 8 && strcmp(argv[8], "wflip") == 0);
        int sflip  = (argc > 8 && strcmp(argv[8], "sflip") == 0);
        if (Kd % 112 != 0) { fprintf(stderr, "nvfp4: K must be a multiple of 112 = LCM(7,16); got K=%d\n", Kd); return 2; }
        int kwords = Kd / 7;     /* packed i32 words per row */
        int kblk   = Kd / 16;    /* E4M3 micro-scale blocks per row */
        float tensor_scale = 1.0f / 3.0f;   /* non-trivial per-tensor FP32 scale (load-bearing) */
        size_t wN = (size_t)Md * kwords, scN = (size_t)Md * kblk, oN = (size_t)Md * Kd;
        int* hCode = (int*)malloc(oN * sizeof(int));      /* original E2M1 codes (for the oracle) */
        int* hW = (int*)malloc(wN * sizeof(int));         /* packed words (device weight buffer) */
        int* hMicro = (int*)malloc(scN * sizeof(int));    /* E4M3 micro-scale codes (for the oracle) */
        float* hSc = (float*)malloc(scN * sizeof(float)); /* effective f32 = e4m3*tensor (device scale) */
        float* hOut = (float*)malloc(oN * sizeof(float));
        if (!hCode || !hW || !hMicro || !hSc || !hOut) return 2;
        for (int r = 0; r < Md; r++)
            for (int k = 0; k < Kd; k++)
                hCode[r * Kd + k] = (r * 5 + k * 7 + 3) % 16;        /* NON-DEGENERATE, spans all 16 E2M1 */
        for (int r = 0; r < Md; r++)
            for (int kw = 0; kw < kwords; kw++) {                    /* pack 7/word base-16 low-nibble-first */
                int word = 0, pw = 1;
                for (int j = 0; j < 7; j++) { word += (hCode[r * Kd + kw * 7 + j] & 15) * pw; pw *= 16; }
                hW[r * kwords + kw] = word;
            }
        for (int r = 0; r < Md; r++)
            for (int bk = 0; bk < kblk; bk++) {                      /* NON-DEGENERATE E4M3 micro, never NaN */
                int mc = (r * 3 + bk * 11 + 0x30) & 0x7E;
                if ((mc & 0x7F) == 0x7F) mc = 0x38;                  /* belt+braces: never the NaN code -> 1.0 */
                hMicro[r * kblk + bk] = mc;
                hSc[r * kblk + bk] = e4m3_decode(mc) * tensor_scale; /* the effective f32 the device reads */
            }
        /* corruption NCs touch ONLY the device buffers (the oracle recomputes from the originals below) */
        if (wflip) { int n0 = hW[0] & 15; hW[0] = hW[0] - n0 + ((n0 + 1) & 15); } /* bump element-0's nibble */
        if (sflip) { hSc[0] = hSc[0] * 1.5f + 0.123f; }                           /* bump block-0's scale */
        CUdeviceptr dW, dSc, dOut;
        CK(cuMemAlloc(&dW, wN * sizeof(int)), "cuMemAlloc W (nvfp4 packed)");
        CK(cuMemAlloc(&dSc, scN * sizeof(float)), "cuMemAlloc Sc (effective f32)");
        CK(cuMemAlloc(&dOut, oN * sizeof(float)), "cuMemAlloc Out (f32 dequant)");
        CK(cuMemcpyHtoD(dW, hW, wN * sizeof(int)), "HtoD W");
        CK(cuMemcpyHtoD(dSc, hSc, scN * sizeof(float)), "HtoD Sc");
        void* nargs[] = { &dW, &dSc, &dOut, &Md, &Kd, &Nd };
        CK(cuLaunchKernel(fn, Md, 1, 1, Kd, 1, 1, 0, 0, nargs, 0), "cuLaunchKernel nvfp4_dequant");
        CK(cuCtxSynchronize(), "cuCtxSynchronize");
        CK(cuMemcpyDtoH(hOut, dOut, oN * sizeof(float)), "DtoH Out");
        int nbad = 0; float maxabs = 0.0f, maxrel = 0.0f;
        for (int r = 0; r < Md; r++) {
            for (int k = 0; k < Kd; k++) {
                int mc = hMicro[r * kblk + k / 16];                  /* ORIGINAL micro (oracle is un-sflip'd) */
                float oscale = e4m3_decode(mc) * tensor_scale;
                float ref = e2m1_decode(hCode[r * Kd + k]) * oscale;
                float got = hOut[r * Kd + k];
                if (mutate && r == 0 && k == 0) { got += 0.5f + 0.1f * fabsf(got); }
                float e = got - ref; if (e < 0) e = -e;
                float denom = fabsf(ref); float rel = denom > 1.0e-9f ? e / denom : e;
                if (e > maxabs) maxabs = e;
                if (rel > maxrel) maxrel = rel;
                if (isnan(got) || (e > 1.0e-5f && rel > 1.0e-6f)) {  /* TIGHT: dequant is f32-exact */
                    if (nbad < 4) fprintf(stderr, "nvfp4 mismatch out[%d,%d]=%g ref %g (abs %g rel %g)\n", r, k, got, ref, e, rel);
                    nbad++;
                }
            }
        }
        double nvbytes = (double)wN * 4.0 + (double)scN * 1.0 + 4.0; /* packed words + 1B/E4M3-micro + one FP32 tensor */
        double f32b = (double)Md * Kd * 4.0, f16b = (double)Md * Kd * 2.0;
        int si = ((int)oN > 17) ? 17 : 0;
        printf("GPU [%s] nvfp4_dequant(E2M1 + E4M3/16-block + FP32-tensor, f32-EXACT) %dx%d (%d elems): out[%d]=%g, max_abs=%g max_rel=%g, %d bad -> %s | footprint(measured): nvfp4=%.0fB vs f32=%.0fB (%.2fx) vs f16=%.0fB (%.2fx)\n",
               gpu, Md, Kd, (int)oN, si, hOut[si], maxabs, maxrel, nbad, nbad ? "FAIL" : "PASS",
               nvbytes, f32b, f32b / nvbytes, f16b, f16b / nvbytes);
        int rc = nbad ? 1 : 0;
        cuMemFree(dW); cuMemFree(dSc); cuMemFree(dOut);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hCode); free(hW); free(hMicro); free(hSc); free(hOut); free(ptx);
        return rc;
    }

    /* gemv_nvfp4 mode (v1.7 INC4): cuda_launch <ptx> gemv_abt_nvfp4 <Nout> gemv_nvfp4 <K> [mutate].
     * Verifies the FUSED NVFP4-dequant GEMV (y[n]=sum_j x[j]*dequant(W[n,j])) vs a from-scratch CPU
     * oracle (host NVFP4 dequant + f32 gemv). Same column accumulation order -> agreement to ~FMA
     * level; [mutate] bumps y[0] and MUST trip the compare (the gemv is load-bearing). */
    if (strcmp(op, "gemv_nvfp4") == 0) {
        int Nout = (argc > 5) ? atoi(argv[5]) : 64;
        int Kd   = (argc > 6) ? atoi(argv[6]) : 112;
        int mutate = (argc > 7 && strcmp(argv[7], "mutate") == 0);
        if (Kd % 112 != 0) { fprintf(stderr, "gemv_nvfp4: K must be a multiple of 112; got %d\n", Kd); return 2; }
        int kwords = Kd / 7, kblk = Kd / 16; float ts = 1.0f / 3.0f;
        size_t wN = (size_t)Nout * kwords, scN = (size_t)Nout * kblk;
        int* hCode = (int*)malloc((size_t)Nout * Kd * sizeof(int));
        int* hW = (int*)malloc(wN * sizeof(int));
        int* hMicro = (int*)malloc(scN * sizeof(int));
        float* hSc = (float*)malloc(scN * sizeof(float));
        float* hX = (float*)malloc((size_t)Kd * sizeof(float));
        float* hY = (float*)malloc((size_t)Nout * sizeof(float));
        float* yref = (float*)malloc((size_t)Nout * sizeof(float));
        if (!hCode||!hW||!hMicro||!hSc||!hX||!hY||!yref) return 2;
        for (int r=0;r<Nout;r++) for (int k=0;k<Kd;k++) hCode[r*Kd+k] = (r*5+k*7+3)%16;
        for (int r=0;r<Nout;r++) for (int kw=0;kw<kwords;kw++){ int word=0,pw=1; for(int j=0;j<7;j++){word+=(hCode[r*Kd+kw*7+j]&15)*pw; pw*=16;} hW[r*kwords+kw]=word; }
        for (int r=0;r<Nout;r++) for (int bk=0;bk<kblk;bk++){ int mc=(r*3+bk*11+0x30)&0x7E; if((mc&0x7F)==0x7F)mc=0x38; hMicro[r*kblk+bk]=mc; hSc[r*kblk+bk]=e4m3_decode(mc)*ts; }
        for (int k=0;k<Kd;k++) hX[k] = (float)((k%11)-5) * 0.25f;
        for (int n=0;n<Nout;n++){ float acc=0.0f; for(int j=0;j<Kd;j++){ float w=e2m1_decode(hCode[n*Kd+j])*(e4m3_decode(hMicro[n*kblk+j/16])*ts); acc+=hX[j]*w; } yref[n]=acc; }
        CUdeviceptr dX,dW,dSc,dY;
        CK(cuMemAlloc(&dX, Kd*sizeof(float)), "alloc X");
        CK(cuMemAlloc(&dW, wN*sizeof(int)), "alloc W");
        CK(cuMemAlloc(&dSc, scN*sizeof(float)), "alloc Sc");
        CK(cuMemAlloc(&dY, Nout*sizeof(float)), "alloc Y");
        CK(cuMemcpyHtoD(dX, hX, Kd*sizeof(float)), "H2D X");
        CK(cuMemcpyHtoD(dW, hW, wN*sizeof(int)), "H2D W");
        CK(cuMemcpyHtoD(dSc, hSc, scN*sizeof(float)), "H2D Sc");
        void* gargs[] = { &dX, &dW, &dSc, &dY, &Kd };
        CK(cuLaunchKernel(fn, Nout, 1, 1, 1, 1, 1, 0, 0, gargs, 0), "launch gemv_abt_nvfp4");
        CK(cuCtxSynchronize(), "sync gemv_abt_nvfp4");
        CK(cuMemcpyDtoH(hY, dY, Nout*sizeof(float)), "D2H Y");
        if (mutate) hY[0] += 0.5f + 0.1f*fabsf(hY[0]);
        int nbad=0; float maxrel=0.0f;
        for (int n=0;n<Nout;n++){ float e=fabsf(hY[n]-yref[n]); float d=fabsf(yref[n]); float rel=d>1.0e-6f?e/d:e; if(rel>maxrel)maxrel=rel; if(isnan(hY[n])||rel>1.0e-3f){ if(nbad<4)fprintf(stderr,"gemv_nvfp4 mismatch y[%d]=%g ref %g (rel %g)\n",n,hY[n],yref[n],rel); nbad++; } }
        printf("GPU [%s] gemv_abt_nvfp4 (fused NVFP4-dequant GEMV) Nout=%d K=%d: y[0]=%g ref=%g max_rel=%g, %d bad -> %s\n",
               gpu, Nout, Kd, hY[0], yref[0], maxrel, nbad, nbad?"FAIL":"PASS");
        int rc=nbad?1:0;
        cuMemFree(dX);cuMemFree(dW);cuMemFree(dSc);cuMemFree(dY);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hCode);free(hW);free(hMicro);free(hSc);free(hX);free(hY);free(yref);free(ptx);
        return rc;
    }

    /* dgemv_blockred mode (T2/M7): cuda_launch <ptx> dequant_gemv_blockred <N> dgemv_blockred <Kpad> [mutate].
     * Verifies the FUSED NVFP4-dequant BLOCK-REDUCTION GEMV (__dequant_gemv_blockred: one 256-thread
     * block per output row, coalesced striped unpack + SMEM tree-reduce) vs a from-scratch CPU oracle
     * (host NVFP4 dequant + f32 gemv) -- byte-identical setup to the gemv_nvfp4 mode above, the only
     * difference being the launch geometry (block=(256,1,1) here vs block=(1,1,1) there) since the
     * block-reduction is the load-bearing change. y[n]=sum_j x[j]*e2m1_decode(code(n,j))*sc[n*(K/16)+j/16].
     * Same column accumulation order -> agreement to ~FMA level; [mutate] bumps y[0] and MUST trip the
     * compare (the block-reduction is load-bearing). K must be a multiple of 112 = LCM(7,16). */
    if (strcmp(op, "dgemv_blockred") == 0) {
        /* H4: the fused kernel now takes RAW e4m3 micro (1 byte/16-block, packed 4-per-i32-word,
         * base-256 low-byte-first, micstride = ceil(kblk/4) words/row) + a per-tensor f32 ts (a
         * 1-elem device buffer), decoding e4m3_decode(micro)*ts IN-KERNEL. New arg layout:
         * {x, w_packed, micro, ts, y, kpad}. The byte-EXACT scale self-check below proves the
         * in-kernel decode reproduces the host e4m3_decode(micro)*ts bit-for-bit; the gemv stays
         * FMA-faithful. NCs: "mutate" bumps y[0]; "sflip" bumps a micro byte (the e4m3 SCALE is
         * load-bearing). K must be a multiple of 112 = LCM(7,16). */
        int Nout = (argc > 5) ? atoi(argv[5]) : 64;
        int Kd   = (argc > 6) ? atoi(argv[6]) : 112;
        int mutate = (argc > 7 && strcmp(argv[7], "mutate") == 0);
        int sflip  = (argc > 7 && strcmp(argv[7], "sflip") == 0);
        if (Kd % 112 != 0) { fprintf(stderr, "dgemv_blockred: K must be a multiple of 112 = LCM(7,16); got %d\n", Kd); return 2; }
        int kwords = Kd / 7, kblk = Kd / 16; float ts = 1.0f / 3.0f;
        int micstride = (kblk + 3) / 4;                 /* i32 micro words per row (4 micro/word) */
        size_t wN = (size_t)Nout * kwords, scN = (size_t)Nout * kblk, micN = (size_t)Nout * micstride;
        int* hCode = (int*)malloc((size_t)Nout * Kd * sizeof(int));
        int* hW = (int*)malloc(wN * sizeof(int));
        int* hMicro = (int*)malloc(scN * sizeof(int));
        int* hMicW = (int*)calloc(micN, sizeof(int));   /* packed micro words (zero-init for the tail byte slots) */
        float* hX = (float*)malloc((size_t)Kd * sizeof(float));
        float* hY = (float*)malloc((size_t)Nout * sizeof(float));
        float* yref = (float*)malloc((size_t)Nout * sizeof(float));
        if (!hCode||!hW||!hMicro||!hMicW||!hX||!hY||!yref) return 2;
        for (int r=0;r<Nout;r++) for (int k=0;k<Kd;k++) hCode[r*Kd+k] = (r*5+k*7+3)%16;
        for (int r=0;r<Nout;r++) for (int kw=0;kw<kwords;kw++){ int word=0,pw=1; for(int j=0;j<7;j++){word+=(hCode[r*Kd+kw*7+j]&15)*pw; pw*=16;} hW[r*kwords+kw]=word; }
        /* micro codes: bit7 ALWAYS clear (scales are positive) + never the NaN code -> packed words non-negative */
        for (int r=0;r<Nout;r++) for (int bk=0;bk<kblk;bk++){ int mc=(r*3+bk*11+0x30)&0x7E; if((mc&0x7F)==0x7F)mc=0x38; hMicro[r*kblk+bk]=mc; }
        /* pack 4 micro/word base-256 low-byte-first, per row (micstride words/row) */
        for (int r=0;r<Nout;r++) for (int bk=0;bk<kblk;bk++){ int wi=bk/4, sl=bk%4; hMicW[r*micstride+wi] |= (hMicro[r*kblk+bk]&0xFF) << (8*sl); }
        for (int k=0;k<Kd;k++) hX[k] = (float)((k%11)-5) * 0.25f;
        for (int n=0;n<Nout;n++){ float acc=0.0f; for(int j=0;j<Kd;j++){ float w=e2m1_decode(hCode[n*Kd+j])*(e4m3_decode(hMicro[n*kblk+j/16])*ts); acc+=hX[j]*w; } yref[n]=acc; }
        if (sflip) { hMicW[0] = (hMicW[0] & ~0xFF) | ((hMicro[0]==0x10?0x18:0x10)); }  /* bump block-0's micro byte */
        CUdeviceptr dX,dW,dMic,dTs,dY;
        CK(cuMemAlloc(&dX, Kd*sizeof(float)), "alloc X");
        CK(cuMemAlloc(&dW, wN*sizeof(int)), "alloc W");
        CK(cuMemAlloc(&dMic, micN*sizeof(int)), "alloc Micro");
        CK(cuMemAlloc(&dTs, sizeof(float)), "alloc Ts");
        CK(cuMemAlloc(&dY, Nout*sizeof(float)), "alloc Y");
        CK(cuMemcpyHtoD(dX, hX, Kd*sizeof(float)), "H2D X");
        CK(cuMemcpyHtoD(dW, hW, wN*sizeof(int)), "H2D W");
        CK(cuMemcpyHtoD(dMic, hMicW, micN*sizeof(int)), "H2D Micro");
        CK(cuMemcpyHtoD(dTs, &ts, sizeof(float)), "H2D Ts");
        void* gargs[] = { &dX, &dW, &dMic, &dTs, &dY, &Kd };
        /* BLOCK-REDUCTION launch: ONE 256-thread block per output row. */
        CK(cuLaunchKernel(fn, Nout, 1, 1, 256, 1, 1, 0, 0, gargs, 0), "launch dequant_gemv_blockred");
        CK(cuCtxSynchronize(), "sync dequant_gemv_blockred");
        CK(cuMemcpyDtoH(hY, dY, Nout*sizeof(float)), "D2H Y");
        if (mutate) hY[0] += 0.5f + 0.1f*fabsf(hY[0]);
        /* (a) BYTE-EXACT scale self-check: the in-kernel decode path replicated on the host (the SAME
         * exact-literal mantissa/pow recipe the emitter uses) MUST equal e4m3_decode(mc)*ts bit-for-bit. */
        int sbad = 0;
        for (int r=0;r<Nout && sbad<4;r++) for (int bk=0;bk<kblk;bk++){
            int mc = hMicro[r*kblk+bk];
            float href = e4m3_decode(mc) * ts;
            int s=(mc>>7)&1, e=(mc>>3)&15, m=mc&7;
            static const float FRAC[8]={0.0f,0.125f,0.25f,0.375f,0.5f,0.625f,0.75f,0.875f};
            float pw = (e==0)? ldexpf(1.0f,-6) : ldexpf(1.0f,e-7);
            float mant = ((e==0)?0.0f:1.0f) + FRAC[m];
            float mag = mant * pw; float kref = (s==0? mag : -mag) * ts;
            unsigned ah, bh; memcpy(&ah,&href,4); memcpy(&bh,&kref,4);
            if (ah != bh) { if (sbad<4) fprintf(stderr,"dgemv_blockred SCALE not byte-exact r%d bk%d mc=%d host=0x%08x kern=0x%08x\n",r,bk,mc,ah,bh); sbad++; }
        }
        int nbad=0; float maxrel=0.0f;
        for (int n=0;n<Nout;n++){ float e=fabsf(hY[n]-yref[n]); float d=fabsf(yref[n]); float rel=d>1.0e-6f?e/d:e; if(rel>maxrel)maxrel=rel; if(isnan(hY[n])||rel>1.0e-3f){ if(nbad<4)fprintf(stderr,"dgemv_blockred mismatch y[%d]=%g ref %g (rel %g)\n",n,hY[n],yref[n],rel); nbad++; } }
        printf("GPU [%s] dequant_gemv_blockred (fused NVFP4-dequant block-reduction GEMV, block=256, IN-KERNEL e4m3) Nout=%d K=%d: y[0]=%g ref=%g max_rel=%g, %d bad, scale_byte_exact=%s -> %s\n",
               gpu, Nout, Kd, hY[0], yref[0], maxrel, nbad, sbad?"NO":"YES", (nbad||sbad)?"FAIL":"PASS");
        int rc=(nbad||sbad)?1:0;
        cuMemFree(dX);cuMemFree(dW);cuMemFree(dMic);cuMemFree(dTs);cuMemFree(dY);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hCode);free(hW);free(hMicro);free(hMicW);free(hX);free(hY);free(yref);free(ptx);
        return rc;
    }

    /* dgemv_warp mode (v1.8/P1): cuda_launch <ptx> dequant_gemv_warp <N> dgemv_warp <Kpad> [mutate|sflip].
     * Verifies the FUSED NVFP4-dequant WARP-PER-ROW GEMV (__dequant_gemv_warp: ONE 32-lane warp per
     * output row, 8 warps (8 rows) per 256-thread (32x8) block, barrier-free warp-shuffle reduce) vs
     * the SAME from-scratch CPU oracle as dgemv_blockred. BYTE-IDENTICAL setup, oracle, scale self-check
     * and arg layout {x,w_packed,micro,ts,y,kpad} to dgemv_blockred above -- the ONLY difference is the
     * launch geometry: grid (ceil(N/8),1,1), block (32,8,1) (warp-per-row + 8x MLP) vs the baseline's
     * (N,1,1)/(256,1,1). The warp-shuffle reduce reorders the final f32 sum vs the SMEM tree, so the
     * compare is reduce-order-equivalent (FMA-faithful, rel<=1e-3), NOT bit-identical -- the gate is
     * token-identical end-to-end (verified by the parent on GPU). [mutate] bumps y[0] and MUST trip the
     * compare; [sflip] bumps a micro byte (the e4m3 SCALE is load-bearing). K must be a multiple of 112. */
    if (strcmp(op, "dgemv_warp") == 0) {
        int Nout = (argc > 5) ? atoi(argv[5]) : 64;
        int Kd   = (argc > 6) ? atoi(argv[6]) : 112;
        int mutate = (argc > 7 && strcmp(argv[7], "mutate") == 0);
        int sflip  = (argc > 7 && strcmp(argv[7], "sflip") == 0);
        if (Kd % 112 != 0) { fprintf(stderr, "dgemv_warp: K must be a multiple of 112 = LCM(7,16); got %d\n", Kd); return 2; }
        int kwords = Kd / 7, kblk = Kd / 16; float ts = 1.0f / 3.0f;
        int micstride = (kblk + 3) / 4;                 /* i32 micro words per row (4 micro/word) */
        size_t wN = (size_t)Nout * kwords, scN = (size_t)Nout * kblk, micN = (size_t)Nout * micstride;
        int* hCode = (int*)malloc((size_t)Nout * Kd * sizeof(int));
        int* hW = (int*)malloc(wN * sizeof(int));
        int* hMicro = (int*)malloc(scN * sizeof(int));
        int* hMicW = (int*)calloc(micN, sizeof(int));   /* packed micro words (zero-init for the tail byte slots) */
        float* hX = (float*)malloc((size_t)Kd * sizeof(float));
        float* hY = (float*)malloc((size_t)Nout * sizeof(float));
        float* yref = (float*)malloc((size_t)Nout * sizeof(float));
        if (!hCode||!hW||!hMicro||!hMicW||!hX||!hY||!yref) return 2;
        for (int r=0;r<Nout;r++) for (int k=0;k<Kd;k++) hCode[r*Kd+k] = (r*5+k*7+3)%16;
        for (int r=0;r<Nout;r++) for (int kw=0;kw<kwords;kw++){ int word=0,pw=1; for(int j=0;j<7;j++){word+=(hCode[r*Kd+kw*7+j]&15)*pw; pw*=16;} hW[r*kwords+kw]=word; }
        /* micro codes: bit7 ALWAYS clear (scales are positive) + never the NaN code -> packed words non-negative */
        for (int r=0;r<Nout;r++) for (int bk=0;bk<kblk;bk++){ int mc=(r*3+bk*11+0x30)&0x7E; if((mc&0x7F)==0x7F)mc=0x38; hMicro[r*kblk+bk]=mc; }
        /* pack 4 micro/word base-256 low-byte-first, per row (micstride words/row) */
        for (int r=0;r<Nout;r++) for (int bk=0;bk<kblk;bk++){ int wi=bk/4, sl=bk%4; hMicW[r*micstride+wi] |= (hMicro[r*kblk+bk]&0xFF) << (8*sl); }
        for (int k=0;k<Kd;k++) hX[k] = (float)((k%11)-5) * 0.25f;
        for (int n=0;n<Nout;n++){ float acc=0.0f; for(int j=0;j<Kd;j++){ float w=e2m1_decode(hCode[n*Kd+j])*(e4m3_decode(hMicro[n*kblk+j/16])*ts); acc+=hX[j]*w; } yref[n]=acc; }
        if (sflip) { hMicW[0] = (hMicW[0] & ~0xFF) | ((hMicro[0]==0x10?0x18:0x10)); }  /* bump block-0's micro byte */
        CUdeviceptr dX,dW,dMic,dTs,dY;
        CK(cuMemAlloc(&dX, Kd*sizeof(float)), "alloc X");
        CK(cuMemAlloc(&dW, wN*sizeof(int)), "alloc W");
        CK(cuMemAlloc(&dMic, micN*sizeof(int)), "alloc Micro");
        CK(cuMemAlloc(&dTs, sizeof(float)), "alloc Ts");
        CK(cuMemAlloc(&dY, Nout*sizeof(float)), "alloc Y");
        CK(cuMemcpyHtoD(dX, hX, Kd*sizeof(float)), "H2D X");
        CK(cuMemcpyHtoD(dW, hW, wN*sizeof(int)), "H2D W");
        CK(cuMemcpyHtoD(dMic, hMicW, micN*sizeof(int)), "H2D Micro");
        CK(cuMemcpyHtoD(dTs, &ts, sizeof(float)), "H2D Ts");
        void* gargs[] = { &dX, &dW, &dMic, &dTs, &dY, &Kd };
        /* WARP-PER-ROW launch: grid (ceil(N/8),1,1), block (32,8,1) -- 8 warps (8 rows) per block. */
        int gx = (Nout + 7) / 8;
        CK(cuLaunchKernel(fn, gx, 1, 1, 32, 8, 1, 0, 0, gargs, 0), "launch dequant_gemv_warp");
        CK(cuCtxSynchronize(), "sync dequant_gemv_warp");
        CK(cuMemcpyDtoH(hY, dY, Nout*sizeof(float)), "D2H Y");
        if (mutate) hY[0] += 0.5f + 0.1f*fabsf(hY[0]);
        /* (a) BYTE-EXACT scale self-check: the in-kernel decode path replicated on the host (the SAME
         * exact-literal mantissa/pow recipe the emitter uses) MUST equal e4m3_decode(mc)*ts bit-for-bit. */
        int sbad = 0;
        for (int r=0;r<Nout && sbad<4;r++) for (int bk=0;bk<kblk;bk++){
            int mc = hMicro[r*kblk+bk];
            float href = e4m3_decode(mc) * ts;
            int s=(mc>>7)&1, e=(mc>>3)&15, m=mc&7;
            static const float FRAC[8]={0.0f,0.125f,0.25f,0.375f,0.5f,0.625f,0.75f,0.875f};
            float pw = (e==0)? ldexpf(1.0f,-6) : ldexpf(1.0f,e-7);
            float mant = ((e==0)?0.0f:1.0f) + FRAC[m];
            float mag = mant * pw; float kref = (s==0? mag : -mag) * ts;
            unsigned ah, bh; memcpy(&ah,&href,4); memcpy(&bh,&kref,4);
            if (ah != bh) { if (sbad<4) fprintf(stderr,"dgemv_warp SCALE not byte-exact r%d bk%d mc=%d host=0x%08x kern=0x%08x\n",r,bk,mc,ah,bh); sbad++; }
        }
        int nbad=0; float maxrel=0.0f;
        for (int n=0;n<Nout;n++){ float e=fabsf(hY[n]-yref[n]); float d=fabsf(yref[n]); float rel=d>1.0e-6f?e/d:e; if(rel>maxrel)maxrel=rel; if(isnan(hY[n])||rel>1.0e-3f){ if(nbad<4)fprintf(stderr,"dgemv_warp mismatch y[%d]=%g ref %g (rel %g)\n",n,hY[n],yref[n],rel); nbad++; } }
        printf("GPU [%s] dequant_gemv_warp (fused NVFP4-dequant warp-per-row GEMV, block=(32,8), 8 rows/block, shfl reduce, IN-KERNEL e4m3) Nout=%d K=%d: y[0]=%g ref=%g max_rel=%g, %d bad, scale_byte_exact=%s -> %s\n",
               gpu, Nout, Kd, hY[0], yref[0], maxrel, nbad, sbad?"NO":"YES", (nbad||sbad)?"FAIL":"PASS");
        int rc=(nbad||sbad)?1:0;
        cuMemFree(dX);cuMemFree(dW);cuMemFree(dMic);cuMemFree(dTs);cuMemFree(dY);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hCode);free(hW);free(hMicro);free(hMicW);free(hX);free(hY);free(yref);free(ptx);
        return rc;
    }

    /* gemm_perf mode (T2/M1 correctness + T2/G1 perf): cuda_launch <ptx> tiled_matmul
     *   <Nignored> gemm_perf <M> <K> <N> [mutate].
     * Launches the kovc SMEM-tiled GEMM (emit_ptx_tiled_matmul_smem) on the device.
     * The launch geometry is read from the emitted PTX-implied tile params via the
     * BM/BN/BK constants below (kept in sync with kovc.hx's emit_ptx_tiled_matmul_smem).
     *
     * G1 (this chunk):
     *  (a) FENCED cuBLAS ORACLE (cublasSgemm, column-major -> swapped operands for a
     *      row-major C=A*B), forced to CUBLAS_PEDANTIC_MATH so it is a TRUE-f32 oracle
     *      (no TF32 tensor-core contamination), and VALIDATED vs the CPU oracle FIRST so
     *      the oracle itself is trusted before it judges the kovc kernel. Link with
     *      -lcublas -L/usr/local/cuda/lib64.
     *  (b) CORRECTNESS: every M*N cell of the kovc result vs the CPU oracle (integer
     *      inputs a[i]=i%7,b[i]=i%5 -> sums < 2^24 are EXACT in f32, == compare) AND vs
     *      the cuBLAS oracle cell-by-cell within tol (1e-3 rel; integer-exact so it is
     *      really 0). A mismatch -> exit 1.
     *  (c) TIMING via cuEvent: ~5 warmup + ~50 timed KERNEL-ONLY launches, report
     *      min/median/max ms (the laptop throttles), TFLOP/s = 2*M*N*K / median_seconds,
     *      for BOTH the kovc kernel and the (pedantic, true-f32) cuBLAS reference, plus
     *      the kovc/cuBLAS ratio. The >=3 TFLOP/s G1 bar is checked by the orchestrator
     *      (scripts/gpu_perf_corpus.sh) which parses the "MEDIAN-TFLOPS" line below.
     * The optional "mutate" arg perturbs one C cell pre-compare to prove the comparator
     * has teeth (comparator negative control -> must FAIL). The barrier-removal negative
     * control (a bar.sync-stripped PTX must mis-compute, proving .shared/bar.sync are
     * load-bearing) is driven by the orchestrator, which re-runs THIS mode on a PTX with
     * the bar.sync lines deleted and asserts it FAILs.
     * Requires M%BM==N%BN==K%BK==0 (asserted) so the launch covers all of C. */
    if (strcmp(op, "gemm_perf") == 0) {
        int Md = (argc > 5) ? atoi(argv[5]) : 64;
        int Kd = (argc > 6) ? atoi(argv[6]) : 64;
        int Nd = (argc > 7) ? atoi(argv[7]) : 64;
        int mutate = (argc > 8 && strcmp(argv[8], "mutate") == 0);
        /* Tile params -- KEEP IN SYNC with kovc.hx emit_ptx_tiled_matmul_smem. The block
         * is (BN/TN)x(BM/TM) threads; grid is (N/BN, M/BM). Override via env for the G1
         * upscale experiments without recompiling the launcher. */
        int BM = 64, BN = 64, BK = 8, TM = 4, TN = 4;
        { const char* e;
          if ((e = getenv("GEMM_BM"))) BM = atoi(e);
          if ((e = getenv("GEMM_BN"))) BN = atoi(e);
          if ((e = getenv("GEMM_BK"))) BK = atoi(e);
          if ((e = getenv("GEMM_TM"))) TM = atoi(e);
          if ((e = getenv("GEMM_TN"))) TN = atoi(e); }
        int bdimx = BN / TN, bdimy = BM / TM;   /* threads per block in x/y */
        /* tile-divisibility assert: billed cells == computed cells, no boundary code. */
        if (Md % BM != 0 || Nd % BN != 0 || Kd % BK != 0) {
            fprintf(stderr, "gemm_perf: dims must satisfy M%%%d==0 (%d), N%%%d==0 (%d), K%%%d==0 (%d)\n", BM, Md, BN, Nd, BK, Kd);
            return 2;
        }
        size_t aN = (size_t)Md * Kd, bN = (size_t)Kd * Nd, cN = (size_t)Md * Nd;
        float* hA = (float*)malloc(aN * sizeof(float));
        float* hB = (float*)malloc(bN * sizeof(float));
        float* hC = (float*)malloc(cN * sizeof(float));
        float* hR = (float*)malloc(cN * sizeof(float));   /* cuBLAS result */
        if (!hA || !hB || !hC || !hR) return 2;
        for (size_t i = 0; i < aN; i++) hA[i] = (float)(i % 7);
        for (size_t i = 0; i < bN; i++) hB[i] = (float)(i % 5);
        for (size_t i = 0; i < cN; i++) hC[i] = -1.0f;
        CUdeviceptr dA, dB, dC, dR;
        CK(cuMemAlloc(&dA, aN * sizeof(float)), "alloc A");
        CK(cuMemAlloc(&dB, bN * sizeof(float)), "alloc B");
        CK(cuMemAlloc(&dC, cN * sizeof(float)), "alloc C");
        CK(cuMemAlloc(&dR, cN * sizeof(float)), "alloc R(cublas)");
        CK(cuMemcpyHtoD(dA, hA, aN * sizeof(float)), "H2D A");
        CK(cuMemcpyHtoD(dB, hB, bN * sizeof(float)), "H2D B");
        CK(cuMemcpyHtoD(dC, hC, cN * sizeof(float)), "H2D C");   /* sentinel so an unwritten cell shows */
        void* gargs[] = { &dA, &dB, &dC, &Md, &Kd, &Nd };
        unsigned gx = (unsigned)(Nd / BN), gy = (unsigned)(Md / BM);
        CK(cuLaunchKernel(fn, gx, gy, 1, (unsigned)bdimx, (unsigned)bdimy, 1, 0, 0, gargs, 0), "launch tiled_matmul");
        CK(cuCtxSynchronize(), "sync tiled_matmul");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(float)), "D2H C");
        if (mutate) hC[0] += 1.0f;   /* comparator negative control: must trip the compare */

        /* --- fenced cuBLAS oracle: true-f32 (pedantic), validated vs CPU FIRST --- */
        cublasHandle_t cbh;
        if (cublasCreate(&cbh) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasCreate failed\n"); return 2; }
        cublasSetMathMode(cbh, CUBLAS_PEDANTIC_MATH);   /* TRUE f32 -- no TF32 contamination */
        float alpha = 1.0f, beta = 0.0f;
        /* row-major C(MxN)=A*B  <=>  column-major C^T(NxM)=B^T*A^T:
         *   Sgemm(N,N, N,M,K, B(ld N), A(ld K), C(ld N)). */
        if (cublasSgemm(cbh, CUBLAS_OP_N, CUBLAS_OP_N, Nd, Md, Kd, &alpha,
                        (const float*)dB, Nd, (const float*)dA, Kd, &beta, (float*)dR, Nd) != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasSgemm failed\n"); return 2;
        }
        CK(cuCtxSynchronize(), "sync cublas");
        CK(cuMemcpyDtoH(hR, dR, cN * sizeof(float)), "D2H R");

        /* The CPU-oracle triple-loops (1)+(2) are O(M*N*K) on ONE host thread -- at
         * 2048^3 that is ~17 GFLOP twice (minutes). They establish the chain-of-trust
         * CPU<-cuBLAS<-kovc and are run in FULL at the smaller corpus sizes; above a cell
         * budget they are SKIPPED and large-N correctness rests on (3) kovc-vs-cuBLAS
         * (GPU-fast) plus the already-trusted oracle. Override the budget via GEMM_CPU_MNK
         * (set 0 to force-skip). This is an honest, reported relaxation -- not a hidden one. */
        double mnk = (double)Md * (double)Nd * (double)Kd;
        double cpu_budget = 6.0e8;   /* ~840^3; covers 64..512 cubed in full */
        { const char* e; if ((e = getenv("GEMM_CPU_MNK"))) cpu_budget = atof(e); }
        int cpu_oracle_ran = (mnk <= cpu_budget);
        /* (1) trust the cuBLAS oracle: cuBLAS == CPU oracle, cell-by-cell. */
        int obad = 0;
        if (cpu_oracle_ran) {
            for (int r = 0; r < Md; r++) {
                for (int cc = 0; cc < Nd; cc++) {
                    float ref = 0.0f;
                    for (int t = 0; t < Kd; t++) ref += hA[r * Kd + t] * hB[t * Nd + cc];
                    float got = hR[r * Nd + cc];
                    float e = got - ref; if (e < 0) e = -e;
                    float aref = ref < 0 ? -ref : ref;
                    if (isnan(got) || (e > 1.0e-3f && e > 1.0e-3f * aref)) { if (obad < 6) fprintf(stderr, "cuBLAS-vs-CPU mismatch [%d,%d]=%g ref %g\n", r, cc, got, ref); obad++; }
                }
            }
        }
        /* (2) kovc kernel == CPU oracle (integer-exact, == compare). */
        int gbad = 0;
        if (cpu_oracle_ran) {
            for (int r = 0; r < Md; r++) {
                for (int cc = 0; cc < Nd; cc++) {
                    float ref = 0.0f;
                    for (int t = 0; t < Kd; t++) ref += hA[r * Kd + t] * hB[t * Nd + cc];
                    float got = hC[r * Nd + cc];
                    if (got != ref) { if (gbad < 6) fprintf(stderr, "gemm_perf(CPU) mismatch C[%d,%d]=%g ref %g\n", r, cc, got, ref); gbad++; }
                }
            }
        }
        /* (3) kovc kernel vs cuBLAS oracle, cell-by-cell within tol (1e-3 f32) -- O(M*N), GPU-trusted. */
        int cbad = 0;
        for (int r = 0; r < Md; r++) {
            for (int cc = 0; cc < Nd; cc++) {
                float got = hC[r * Nd + cc], refb = hR[r * Nd + cc];
                float e = got - refb; if (e < 0) e = -e;
                float ar = refb < 0 ? -refb : refb;
                if (isnan(got) || (e > 1.0e-3f && e > 1.0e-3f * ar)) { if (cbad < 6) fprintf(stderr, "gemm_perf(cuBLAS) mismatch C[%d,%d]=%g cublas %g\n", r, cc, got, refb); cbad++; }
            }
        }

        /* --- TIMING (kernel-only) via cuEvent: warmup + timed launches, min/med/max --- */
        double flop = 2.0 * (double)Md * (double)Nd * (double)Kd;
        double kov_med = 0.0, blas_med = 0.0;
        if (!mutate) {
            int WARM = 5, ITERS = 50;
            float* ts = (float*)malloc((size_t)ITERS * sizeof(float));
            CUevent e0, e1; CK(cuEventCreate(&e0, 0), "evt0"); CK(cuEventCreate(&e1, 0), "evt1");
            /* kovc kernel timing */
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(fn, gx, gy, 1, (unsigned)bdimx, (unsigned)bdimy, 1, 0, 0, gargs, 0), "warm kov");
            CK(cuCtxSynchronize(), "sync warm kov");
            for (int it = 0; it < ITERS; it++) {
                CK(cuEventRecord(e0, 0), "rec0");
                CK(cuLaunchKernel(fn, gx, gy, 1, (unsigned)bdimx, (unsigned)bdimy, 1, 0, 0, gargs, 0), "time kov");
                CK(cuEventRecord(e1, 0), "rec1"); CK(cuEventSynchronize(e1), "evsync");
                CK(cuEventElapsedTime(&ts[it], e0, e1), "elapsed");
            }
            /* sort to get min/median/max */
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            float kmin = ts[0], kmax = ts[ITERS - 1], kmed = ts[ITERS / 2];
            kov_med = (double)kmed;
            /* cuBLAS (pedantic, true-f32) timing -- same protocol */
            for (int w = 0; w < WARM; w++) cublasSgemm(cbh, CUBLAS_OP_N, CUBLAS_OP_N, Nd, Md, Kd, &alpha, (const float*)dB, Nd, (const float*)dA, Kd, &beta, (float*)dR, Nd);
            CK(cuCtxSynchronize(), "sync warm blas");
            for (int it = 0; it < ITERS; it++) {
                CK(cuEventRecord(e0, 0), "brec0");
                cublasSgemm(cbh, CUBLAS_OP_N, CUBLAS_OP_N, Nd, Md, Kd, &alpha, (const float*)dB, Nd, (const float*)dA, Kd, &beta, (float*)dR, Nd);
                CK(cuEventRecord(e1, 0), "brec1"); CK(cuEventSynchronize(e1), "bevsync");
                CK(cuEventElapsedTime(&ts[it], e0, e1), "belapsed");
            }
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            float bmin = ts[0], bmax = ts[ITERS - 1], bmed = ts[ITERS / 2];
            blas_med = (double)bmed;
            double kov_tf = flop / (kov_med * 1.0e-3) / 1.0e12;
            double blas_tf = flop / (blas_med * 1.0e-3) / 1.0e12;
            printf("GPU [%s] TIMING kovc   %dx%dx%d: min=%.4f med=%.4f max=%.4f ms\n", gpu, Md, Kd, Nd, kmin, kmed, kmax);
            printf("GPU [%s] TIMING cuBLAS %dx%dx%d (pedantic f32): min=%.4f med=%.4f max=%.4f ms\n", gpu, Md, Kd, Nd, bmin, bmed, bmax);
            printf("MEDIAN-TFLOPS kovc=%.3f cublas=%.3f ratio=%.1f%%\n", kov_tf, blas_tf, 100.0 * kov_tf / blas_tf);
            cuEventDestroy(e0); cuEventDestroy(e1); free(ts);
        }
        cublasDestroy(cbh);

        int totbad = gbad + obad + cbad;
        printf("GPU [%s] tiled_matmul (SMEM %dx%d/BK%d/%dx%d) %dx%dx%d over %d cells%s: C[1,1]=%g, kovc-vs-CPU=%d oracle(cuBLAS-vs-CPU)=%d kovc-vs-cuBLAS=%d [CPU-oracle:%s] -> %s\n",
               gpu, BM, BN, BK, TM, TN, Md, Kd, Nd, Md * Nd, mutate ? " [MUTATED]" : "", hC[1 * Nd + 1], gbad, obad, cbad,
               cpu_oracle_ran ? "ran" : "skipped(large-N; kovc-vs-cuBLAS only)", totbad ? "FAIL" : "PASS");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC); cuMemFree(dR);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hC); free(hR); free(ptx);
        return totbad ? 1 : 0;
    }

    /* ===================================================================== *
     * gemm_tf32 mode (T3/G3): cuda_launch <ptx> tf32_matmul 0 gemm_tf32 <M> <K> <N> [mutate]
     *
     * Correctness + perf harness for the kovc TF32 Tensor-Core (mma.sync) GEMM. The kernel
     * is warp-collaborative: ONE warp (32 threads = one block) computes one 16x8 output tile
     * via mma.sync.aligned.m16n8k8.row.col.f32.tf32, looping K/8 times. Grid = (N/8, M/16),
     * block = 32. Requires M%16==0, N%8==0, K%8==0.
     *
     * ORACLE DESIGN (skeptic-corrected, the load-bearing part):
     *  - PRIMARY correctness: judge the kovc TF32 kernel against cublasGemmEx(
     *    CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP) at a TIGHT ~2e-3 rel
     *    tol. Two correct TF32 GEMMs differ only by accumulation order (both truncate the
     *    SAME 10-bit mantissa), so they agree to ~1e-3; a dropped/mis-indexed fragment
     *    stands out sharply. This is the test that has TEETH for a reduced-precision kernel.
     *  - META-ANCHOR (retained): pedantic-f32 cublasSgemm == CPU triple-loop, exact-ish
     *    (1e-3), proves the cuBLAS references themselves are sane (chain-of-trust CPU<-cuBLAS).
     *  - DISTINCT-per-element inputs (NOT i%7/i%5 uniform): every one of the 32 lanes'
     *    fragments is individually observable, so a fragment-permutation bug cannot hide
     *    behind periodic/low-rank input. Inputs are small bounded values to keep the
     *    accumulation noise tiny but each (r,c) element is distinct.
     *  - kovc-TF32 vs CPU-f32 deviation is reported for CONTEXT only (NOT a gate -- f32 ref
     *    vs a tf32 kernel is the self-contradictory tol the skeptic warned against).
     * ===================================================================== */
    if (strcmp(op, "gemm_tf32") == 0) {
        int Md = (argc > 5) ? atoi(argv[5]) : 16;
        int Kd = (argc > 6) ? atoi(argv[6]) : 8;
        int Nd = (argc > 7) ? atoi(argv[7]) : 32;
        int mutate = (argc > 8 && strcmp(argv[8], "mutate") == 0);
        /* TF32_NB / TF32_WP MUST equal the `nb` / `wp` constants in emit_ptx_tf32_matmul_mma
         * (kovc.hx): a block = (32, WP, 1) of WP warps, each warp a 16 x (8*NB) strip, so the
         * block covers 16 x (8*NB*WP) and grid=(N/(8*NB*WP), M/16). */
        const int TF32_NB = 4;
        const int TF32_WP = 4;
        const int NTILE = 8 * TF32_NB * TF32_WP;   /* = 128: the N-span one block covers */
        if (Md % 16 != 0 || Nd % NTILE != 0 || Kd % 8 != 0) {
            fprintf(stderr, "gemm_tf32: dims must satisfy M%%16==0 (%d), N%%%d==0 (%d), K%%8==0 (%d)\n", Md, NTILE, Nd, Kd);
            return 2;
        }
        size_t aN = (size_t)Md * Kd, bN = (size_t)Kd * Nd, cN = (size_t)Md * Nd;
        float* hA = (float*)malloc(aN * sizeof(float));
        float* hB = (float*)malloc(bN * sizeof(float));
        float* hC = (float*)malloc(cN * sizeof(float));   /* kovc TF32 result */
        float* hR = (float*)malloc(cN * sizeof(float));   /* cuBLAS-TF32 result (primary ref) */
        float* hF = (float*)malloc(cN * sizeof(float));   /* cuBLAS-f32-pedantic (meta-anchor) */
        if (!hA || !hB || !hC || !hR || !hF) return 2;
        /* DISTINCT per-element inputs: a quasi-random but deterministic bounded spread so
         * every element differs from its neighbours (exposes any lane/fragment swap), yet
         * stays small (|elem|<~7.8, |prod|<~61, K-partial sums < ~1e3) so TF32 truncation
         * noise is low. NOTE the (int) cast BEFORE the -125/-120: the modulo is computed in
         * size_t (unsigned), so subtracting in size_t would UNDERFLOW to ~1.8e19 for the
         * lower half of values (-> garbage ~1.15e18 floats); cast to int first. */
        for (size_t i = 0; i < aN; i++) hA[i] = (float)((int)((i * 131 + 7) % 251) - 125) * 0.0625f;
        for (size_t i = 0; i < bN; i++) hB[i] = (float)((int)((i * 97 + 13) % 241) - 120) * 0.0625f;
        for (size_t i = 0; i < cN; i++) hC[i] = -123456.0f;   /* sentinel: unwritten cell shows */
        CUdeviceptr dA, dB, dC, dR;
        CK(cuMemAlloc(&dA, aN * sizeof(float)), "alloc A");
        CK(cuMemAlloc(&dB, bN * sizeof(float)), "alloc B");
        CK(cuMemAlloc(&dC, cN * sizeof(float)), "alloc C");
        CK(cuMemAlloc(&dR, cN * sizeof(float)), "alloc R(cublas)");
        CK(cuMemcpyHtoD(dA, hA, aN * sizeof(float)), "H2D A");
        CK(cuMemcpyHtoD(dB, hB, bN * sizeof(float)), "H2D B");
        CK(cuMemcpyHtoD(dC, hC, cN * sizeof(float)), "H2D C");
        /* launch: WP warps/block, each a 16 x (8*NB) strip. grid=(N/(8*NB*WP), M/16),
         * block=(32, WP, 1). */
        void* gargs[] = { &dA, &dB, &dC, &Md, &Kd, &Nd };
        unsigned gx = (unsigned)(Nd / NTILE), gy = (unsigned)(Md / 16);
        CK(cuLaunchKernel(fn, gx, gy, 1, 32, TF32_WP, 1, 0, 0, gargs, 0), "launch tf32_matmul");
        CK(cuCtxSynchronize(), "sync tf32_matmul");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(float)), "D2H C");
        /* comparator negative control: perturb one cell so the tol compare MUST trip. The
         * delta is SCALED to exceed the magnitude-aware tol (skeptic §5.7): a bare +1.0f
         * could be < TF32_REL*|hC[0]| on a large-magnitude cell -> vacuous. */
        if (mutate) { float d = 0.05f * fabsf(hC[0]); hC[0] += (d > 1.0f ? d : 1.0f); }

        cublasHandle_t cbh;
        if (cublasCreate(&cbh) != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasCreate failed\n"); return 2; }
        float alpha = 1.0f, beta = 0.0f;
        /* (A) cuBLAS-TF32 reference (PRIMARY): cublasGemmEx COMPUTE_32F_FAST_TF32 + TENSOR_OP.
         * row-major C(MxN)=A*B <=> col-major C^T=B^T*A^T: GemmEx(N,N, N,M,K, B(ld N), A(ld K), R(ld N)). */
        cublasSetMathMode(cbh, CUBLAS_TF32_TENSOR_OP_MATH);
        if (cublasGemmEx(cbh, CUBLAS_OP_N, CUBLAS_OP_N, Nd, Md, Kd, &alpha,
                         (const void*)dB, CUDA_R_32F, Nd, (const void*)dA, CUDA_R_32F, Kd, &beta,
                         (void*)dR, CUDA_R_32F, Nd, CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP)
            != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cublasGemmEx TF32 failed\n"); return 2; }
        CK(cuCtxSynchronize(), "sync cublas tf32");
        CK(cuMemcpyDtoH(hR, dR, cN * sizeof(float)), "D2H R(tf32)");
        /* (B) cuBLAS-f32 pedantic (META-ANCHOR): true f32, no TF32 contamination. */
        cublasSetMathMode(cbh, CUBLAS_PEDANTIC_MATH);
        if (cublasSgemm(cbh, CUBLAS_OP_N, CUBLAS_OP_N, Nd, Md, Kd, &alpha,
                        (const float*)dB, Nd, (const float*)dA, Kd, &beta, (float*)dR, Nd) != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasSgemm failed\n"); return 2;
        }
        CK(cuCtxSynchronize(), "sync cublas f32");
        CK(cuMemcpyDtoH(hF, dR, cN * sizeof(float)), "D2H F(f32)");

        /* The two CPU triple-loops below are O(M*N*K). At small corpus sizes (<=128^3) they
         * run in full and give the harness its teeth. At a large PERF size (e.g. 2048^3) a
         * CPU triple-loop is ~17e9 scalar MACs = MINUTES at 0% GPU -- the "hang" the prior
         * agent hit was NOT a kernel illegal-access but this CPU reference. So gate them on a
         * work cap: above it, skip the CPU loops and rely on the PRIMARY GPU-side kovc-vs-
         * cuBLAS-TF32 compare (line below, only O(M*N) cells -- cheap even at 2048^2), which is
         * itself a sound oracle (cuBLAS-TF32 was validated == CPU at the small sizes). */
        double cpu_work = (double)Md * (double)Nd * (double)Kd;
        int big = (cpu_work > 4.0e8);   /* ~735^3; 128^3=2.1e6 runs, 512^3=1.34e8 runs, 2048^3 skips */
        /* META-ANCHOR check: pedantic-f32 cuBLAS == CPU triple-loop (cell-by-cell, ~1e-3).
         * Proves the references are correct; runs in full at the small G3 sizes (skipped when big). */
        int obad = 0;
        if (!big) for (int r = 0; r < Md; r++) for (int cc = 0; cc < Nd; cc++) {
            float ref = 0.0f;
            for (int t = 0; t < Kd; t++) ref += hA[r * Kd + t] * hB[t * Nd + cc];
            float got = hF[r * Nd + cc];
            float e = got - ref; if (e < 0) e = -e;
            float aref = ref < 0 ? -ref : ref;
            if (isnan(got) || (e > 1.0e-3f && e > 1.0e-3f * aref)) { if (obad < 6) fprintf(stderr, "f32-cuBLAS-vs-CPU mismatch [%d,%d]=%g ref %g\n", r, cc, got, ref); obad++; }
        }
        /* PRIMARY (ALWAYS, even when big): kovc TF32 kernel vs cuBLAS-TF32, tight ~2e-3 RELATIVE
         * (magnitude-aware). O(M*N) only -- this is the real correctness gate at every size. */
        const float TF32_REL = 2.0e-3f, TF32_ABS = 2.0e-3f;
        int tbad = 0; float maxrel = 0.0f;
        for (int r = 0; r < Md; r++) for (int cc = 0; cc < Nd; cc++) {
            float got = hC[r * Nd + cc], refb = hR[r * Nd + cc];
            float e = got - refb; if (e < 0) e = -e;
            float ar = refb < 0 ? -refb : refb;
            float rel = e / (ar > 1.0f ? ar : 1.0f); if (rel > maxrel) maxrel = rel;
            if (isnan(got) || (e > TF32_ABS && e > TF32_REL * ar)) { if (tbad < 6) fprintf(stderr, "tf32 kovc-vs-cuBLAS-TF32 mismatch C[%d,%d]=%g tf32ref %g (rel %g)\n", r, cc, got, refb, rel); tbad++; }
        }
        /* CONTEXT only: kovc TF32 vs CPU-f32 max relative deviation (NOT a gate; -1 = skipped when big). */
        float ctx_maxrel = big ? -1.0f : 0.0f;
        if (!big) for (int r = 0; r < Md; r++) for (int cc = 0; cc < Nd; cc++) {
            float ref = 0.0f; for (int t = 0; t < Kd; t++) ref += hA[r * Kd + t] * hB[t * Nd + cc];
            float got = hC[r * Nd + cc]; float e = got - ref; if (e < 0) e = -e;
            float ar = ref < 0 ? -ref : ref; float rel = e / (ar > 1.0f ? ar : 1.0f);
            if (rel > ctx_maxrel) ctx_maxrel = rel;
        }

        /* --- TIMING (kernel-only) via cuEvent: warmup + median-of-50, min/med/max. The
         * absolute perf GATE is DEFERRED this phase (correctness-first); set via G1_MIN_TFLOPS
         * by the corpus. We still report the number for the orchestrator. --- */
        double flop = 2.0 * (double)Md * (double)Nd * (double)Kd;
        if (!mutate) {
            int WARM = 5, ITERS = 50;
            float* ts = (float*)malloc((size_t)ITERS * sizeof(float));
            CUevent e0, e1; CK(cuEventCreate(&e0, 0), "evt0"); CK(cuEventCreate(&e1, 0), "evt1");
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(fn, gx, gy, 1, 32, TF32_WP, 1, 0, 0, gargs, 0), "warm kov");
            CK(cuCtxSynchronize(), "sync warm kov");
            for (int it = 0; it < ITERS; it++) {
                CK(cuEventRecord(e0, 0), "rec0");
                CK(cuLaunchKernel(fn, gx, gy, 1, 32, TF32_WP, 1, 0, 0, gargs, 0), "time kov");
                CK(cuEventRecord(e1, 0), "rec1"); CK(cuEventSynchronize(e1), "evsync");
                CK(cuEventElapsedTime(&ts[it], e0, e1), "elapsed");
            }
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            float kmin = ts[0], kmax = ts[ITERS - 1], kmed = ts[ITERS / 2];
            double kov_tf = flop / (kmed * 1.0e-3) / 1.0e12;
            printf("GPU [%s] TIMING tf32_kovc %dx%dx%d: min=%.4f med=%.4f max=%.4f ms\n", gpu, Md, Kd, Nd, kmin, kmed, kmax);
            printf("MEDIAN-TFLOPS-TF32 kovc=%.3f\n", kov_tf);
            cuEventDestroy(e0); cuEventDestroy(e1); free(ts);
        }
        cublasDestroy(cbh);

        int totbad = tbad + obad;
        printf("GPU [%s] tf32_matmul (mma m16n8k8) %dx%dx%d over %d cells%s: C[1,1]=%g tf32ref=%g, kovc-vs-cuBLAS-TF32=%d(maxrel=%.2e tol=%.0e) anchor(f32cuBLAS-vs-CPU)=%d [ctx:kovc-vs-CPUf32 maxrel=%.2e] -> %s\n",
               gpu, Md, Kd, Nd, Md * Nd, mutate ? " [MUTATED]" : "", hC[1 * Nd + 1], hR[1 * Nd + 1], tbad, maxrel, (double)TF32_REL, obad, ctx_maxrel, totbad ? "FAIL" : "PASS");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC); cuMemFree(dR);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hC); free(hR); free(hF); free(ptx);
        return totbad ? 1 : 0;
    }

    /* matmul_abt mode: cuda_launch <ptx> gpu_matmul_abt <Nignored> matmul_abt <M> <K> <N>.
     * C[M,N] = A[M,K] @ B[N,K]^T, i.e. C[i,j]=sum_t A[i,t]*B[j,t] -- the UNSCALED A@B^T used
     * for d_attn=dOut@V^T. gridDim=M, blockDim=N. Integer inputs so exact; verify every cell. */
    if (strcmp(op, "matmul_abt") == 0) {
        int M = (argc > 5) ? atoi(argv[5]) : 16;
        int K = (argc > 6) ? atoi(argv[6]) : 16;
        int Nn = (argc > 7) ? atoi(argv[7]) : 16;
        size_t aN = (size_t)M * K, bN = (size_t)Nn * K, cN = (size_t)M * Nn;
        float* hA = (float*)malloc(aN * sizeof(float));
        float* hB = (float*)malloc(bN * sizeof(float));
        float* hC = (float*)malloc(cN * sizeof(float));
        if (!hA || !hB || !hC) return 2;
        for (size_t i = 0; i < aN; i++) hA[i] = (float)(i % 7);
        for (size_t i = 0; i < bN; i++) hB[i] = (float)(i % 5);
        CUdeviceptr dA, dB, dC;
        CK(cuMemAlloc(&dA, aN * sizeof(float)), "alloc A");
        CK(cuMemAlloc(&dB, bN * sizeof(float)), "alloc B");
        CK(cuMemAlloc(&dC, cN * sizeof(float)), "alloc C");
        CK(cuMemcpyHtoD(dA, hA, aN * sizeof(float)), "H2D A");
        CK(cuMemcpyHtoD(dB, hB, bN * sizeof(float)), "H2D B");
        void* uargs[] = { &dA, &dB, &dC, &M, &K, &Nn };
        CK(cuLaunchKernel(fn, M, 1, 1, Nn, 1, 1, 0, 0, uargs, 0), "launch matmul_abt");
        CK(cuCtxSynchronize(), "sync matmul_abt");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(float)), "D2H C");
        int tbad = 0;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < Nn; j++) {
                float ref = hA[i * K] * hB[j * K];
                for (int t = 1; t < K; t++) ref += hA[i * K + t] * hB[j * K + t];
                float got = hC[i * Nn + j];
                if (got != ref) { if (tbad < 4) fprintf(stderr, "matmul_abt mismatch C[%d,%d]=%g ref %g\n", i, j, got, ref); tbad++; }
            }
        }
        printf("GPU [%s] matmul_abt (A@B^T) %dx%dx%d over %d cells: C[1,1]=%g, %d bad -> %s\n",
               gpu, M, K, Nn, M * Nn, hC[1 * Nn + 1], tbad, tbad ? "FAIL" : "PASS");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hC); free(ptx);
        return tbad ? 1 : 0;
    }

    /* matmul_atb mode: cuda_launch <ptx> gpu_matmul_atb <Nignored> matmul_atb <M> <K> <N>.
     * C[K,N] = A[M,K]^T @ B[M,N], i.e. C[i,j]=sum_t A[t,i]*B[t,j] -- the dW=X^T@dY backward
     * workhorse (the 3rd matmul variant). gridDim=K, blockDim=N. Integer inputs so the result
     * is exact in f32; verify every K*N cell vs a CPU A^T@B in the kernel's accumulation order. */
    if (strcmp(op, "matmul_atb") == 0) {
        int M = (argc > 5) ? atoi(argv[5]) : 16;
        int K = (argc > 6) ? atoi(argv[6]) : 16;
        int Nn = (argc > 7) ? atoi(argv[7]) : 16;
        size_t aN = (size_t)M * K, bN = (size_t)M * Nn, cN = (size_t)K * Nn;
        float* hA = (float*)malloc(aN * sizeof(float));
        float* hB = (float*)malloc(bN * sizeof(float));
        float* hC = (float*)malloc(cN * sizeof(float));
        if (!hA || !hB || !hC) return 2;
        for (size_t i = 0; i < aN; i++) hA[i] = (float)(i % 7);
        for (size_t i = 0; i < bN; i++) hB[i] = (float)(i % 5);
        CUdeviceptr dA, dB, dC;
        CK(cuMemAlloc(&dA, aN * sizeof(float)), "alloc A");
        CK(cuMemAlloc(&dB, bN * sizeof(float)), "alloc B");
        CK(cuMemAlloc(&dC, cN * sizeof(float)), "alloc C");
        CK(cuMemcpyHtoD(dA, hA, aN * sizeof(float)), "H2D A");
        CK(cuMemcpyHtoD(dB, hB, bN * sizeof(float)), "H2D B");
        void* targs[] = { &dA, &dB, &dC, &M, &K, &Nn };
        CK(cuLaunchKernel(fn, K, 1, 1, Nn, 1, 1, 0, 0, targs, 0), "launch matmul_atb");
        CK(cuCtxSynchronize(), "sync matmul_atb");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(float)), "D2H C");
        int tbad = 0;
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < Nn; j++) {
                float ref = hA[0 * K + i] * hB[0 * Nn + j];
                for (int t = 1; t < M; t++) ref += hA[t * K + i] * hB[t * Nn + j];
                float got = hC[i * Nn + j];
                if (got != ref) { if (tbad < 4) fprintf(stderr, "matmul_atb mismatch C[%d,%d]=%g ref %g\n", i, j, got, ref); tbad++; }
            }
        }
        printf("GPU [%s] matmul_atb (A^T@B) %dx%dx%d over %d cells: C[1,1]=%g, %d bad -> %s\n",
               gpu, M, K, Nn, K * Nn, hC[1 * Nn + 1], tbad, tbad ? "FAIL" : "PASS");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hC); free(ptx);
        return tbad ? 1 : 0;
    }

    /* gemm_abt mode (T2/M4): cuda_launch <combined.ptx> tiled_matmul_abt 0 gemm_abt <M> <K> <N> [mutate].
     * The SMEM-tiled A@B^T (emit_ptx_tiled_matmul_t mode 0) vs (a) a CPU A@B^T oracle for
     * CORRECTNESS and (b) the naive non-tiled gpu_matmul_abt for the FASTER-THAN-NAIVE bar.
     * combined.ptx must carry BOTH kernels (concat tiled_matmul_abt_kernel.hx +
     * gpu_matmul_abt_kernel.hx, emit once). A[M,K] (a[i*K+t]), B[N,K] (b[j*K+t]) transposed,
     * C[M,N] (c[i*N+j]); C[i,j]=sum_t A[i,t]*B[j,t]. Tiled: grid=(N/64,M/64) block=(16,16);
     * naive: grid.x=M block.x=N (so the speedup compare needs N<=1024). Integer inputs
     * (a[i]=i%7,b[i]=i%5) keep every cell sum < 2^24 for K up to ~7e5 -> EXACT == compare.
     * Requires M%64==0, N%64==0, K%8==0 (tiled, asserted). */
    if (strcmp(op, "gemm_abt") == 0) {
        int M = (argc > 5) ? atoi(argv[5]) : 64;
        int K = (argc > 6) ? atoi(argv[6]) : 64;
        int Nn = (argc > 7) ? atoi(argv[7]) : 64;
        int mutate = (argc > 8 && strcmp(argv[8], "mutate") == 0);
        /* "corr" (8th arg): correctness-only -- still measure+report the tiled-vs-naive speedup
         * but do NOT gate on it (used for tiny sizes like 64^3 where there is too little work to
         * amortize the tiling overhead so the naive kernel can match/beat it; the faster-than-
         * naive GATE is asserted at the large/perf sizes). */
        int corr_only = (argc > 8 && strcmp(argv[8], "corr") == 0);
        if (M % 64 != 0 || Nn % 64 != 0 || K % 8 != 0) {
            fprintf(stderr, "gemm_abt: dims must satisfy M%%64==0 (%d), N%%64==0 (%d), K%%8==0 (%d)\n", M, Nn, K);
            return 2;
        }
        if (Nn > 1024) { fprintf(stderr, "gemm_abt: N<=1024 (naive baseline blockDim.x=N); got %d\n", Nn); return 2; }
        CUfunction f_naive; CK(cuModuleGetFunction(&f_naive, mod, "gpu_matmul_abt"), "get gpu_matmul_abt");
        size_t aN = (size_t)M * K, bN = (size_t)Nn * K, cN = (size_t)M * Nn;
        float* hA = (float*)malloc(aN * sizeof(float));
        float* hB = (float*)malloc(bN * sizeof(float));
        float* hC = (float*)malloc(cN * sizeof(float));
        if (!hA || !hB || !hC) return 2;
        for (size_t i = 0; i < aN; i++) hA[i] = (float)(i % 7);
        for (size_t i = 0; i < bN; i++) hB[i] = (float)(i % 5);
        CUdeviceptr dA, dB, dC;
        CK(cuMemAlloc(&dA, aN * sizeof(float)), "alloc A");
        CK(cuMemAlloc(&dB, bN * sizeof(float)), "alloc B");
        CK(cuMemAlloc(&dC, cN * sizeof(float)), "alloc C");
        CK(cuMemcpyHtoD(dA, hA, aN * sizeof(float)), "H2D A");
        CK(cuMemcpyHtoD(dB, hB, bN * sizeof(float)), "H2D B");
        { float neg = -1.0f; for (size_t i = 0; i < cN; i++) hC[i] = neg; }
        CK(cuMemcpyHtoD(dC, hC, cN * sizeof(float)), "H2D C(sentinel)");
        void* gargs[] = { &dA, &dB, &dC, &M, &K, &Nn };
        unsigned gx = (unsigned)(Nn / 64), gy = (unsigned)(M / 64);
        CK(cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, 0, gargs, 0), "launch tiled_matmul_abt");
        CK(cuCtxSynchronize(), "sync tiled_matmul_abt");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(float)), "D2H C");
        if (mutate) hC[0] += 1.0f;
        /* CORRECTNESS: cell-by-cell vs CPU A@B^T (integer-exact). Work-capped: above ~735^3
         * the O(M*N*K) CPU loop is skipped (sizes here stay <=512 so it runs in full). */
        double cpu_work = (double)M * (double)Nn * (double)K;
        int big = (cpu_work > 4.0e8);
        int cbad = 0;
        if (!big) for (int i = 0; i < M; i++) for (int j = 0; j < Nn; j++) {
            float ref = hA[i * K] * hB[j * K];
            for (int t = 1; t < K; t++) ref += hA[i * K + t] * hB[j * K + t];
            float got = hC[i * Nn + j];
            if (got != ref) { if (cbad < 6) fprintf(stderr, "gemm_abt mismatch C[%d,%d]=%g ref %g\n", i, j, got, ref); cbad++; }
        }
        /* FASTER-THAN-NAIVE: time the tiled kernel AND the naive gpu_matmul_abt, kernel-only,
         * cuEvent median-of-30. The naive run also writes dC -> validate it agrees with the
         * tiled result (GPU-side, O(M*N)) so the baseline is a REAL same-answer comparison. */
        double speedup = 0.0; int nbad = 0;
        if (!mutate) {
            int WARM = 3, ITERS = 30;
            float* ts = (float*)malloc((size_t)ITERS * sizeof(float));
            CUevent e0, e1; CK(cuEventCreate(&e0, 0), "evt0"); CK(cuEventCreate(&e1, 0), "evt1");
            float* hCn = (float*)malloc(cN * sizeof(float));
            void* nargs[] = { &dA, &dB, &dC, &M, &K, &Nn };
            /* tiled timing */
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, 0, gargs, 0), "warm tiled");
            CK(cuCtxSynchronize(), "sync warm tiled");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0,0),"r0"); CK(cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, 0, gargs, 0),"t"); CK(cuEventRecord(e1,0),"r1"); CK(cuEventSynchronize(e1),"s"); CK(cuEventElapsedTime(&ts[it],e0,e1),"e"); }
            for (int i = 0; i < ITERS; i++) for (int j = i+1; j < ITERS; j++) if (ts[j] < ts[i]) { float t=ts[i]; ts[i]=ts[j]; ts[j]=t; }
            float t_tiled = ts[ITERS/2];
            /* naive timing (grid.x=M, block.x=N) */
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(f_naive, (unsigned)M, 1, 1, (unsigned)Nn, 1, 1, 0, 0, nargs, 0), "warm naive");
            CK(cuCtxSynchronize(), "sync warm naive");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0,0),"r0"); CK(cuLaunchKernel(f_naive,(unsigned)M,1,1,(unsigned)Nn,1,1,0,0,nargs,0),"t"); CK(cuEventRecord(e1,0),"r1"); CK(cuEventSynchronize(e1),"s"); CK(cuEventElapsedTime(&ts[it],e0,e1),"e"); }
            for (int i = 0; i < ITERS; i++) for (int j = i+1; j < ITERS; j++) if (ts[j] < ts[i]) { float t=ts[i]; ts[i]=ts[j]; ts[j]=t; }
            float t_naive = ts[ITERS/2];
            CK(cuMemcpyDtoH(hCn, dC, cN * sizeof(float)), "D2H Cn(naive)");
            for (int i = 0; i < M; i++) for (int j = 0; j < Nn; j++) if (hCn[i*Nn+j] != hC[i*Nn+j]) { if (nbad < 6) fprintf(stderr, "gemm_abt tiled-vs-naive mismatch C[%d,%d] tiled=%g naive=%g\n", i, j, hC[i*Nn+j], hCn[i*Nn+j]); nbad++; }
            speedup = (t_naive > 0.0f) ? (double)t_naive / (double)t_tiled : 0.0;
            printf("GPU [%s] SPEEDUP-ABT %dx%dx%d: tiled med=%.4f ms  naive med=%.4f ms  speedup=%.2fx\n", gpu, M, K, Nn, t_tiled, t_naive, speedup);
            cuEventDestroy(e0); cuEventDestroy(e1); free(ts); free(hCn);
        }
        int faster = (mutate || corr_only) ? 1 : (speedup > 1.0);
        int totbad = cbad + nbad + (faster ? 0 : 1);
        printf("GPU [%s] tiled_matmul_abt (A@B^T, SMEM 64x64/BK8/4x4) %dx%dx%d over %d cells%s: C[1,1]=%g, vs-CPU=%d vs-naive=%d [CPU:%s] faster-than-naive=%s -> %s\n",
               gpu, M, K, Nn, M * Nn, mutate ? " [MUTATED]" : "", hC[1 * Nn + 1], cbad, nbad,
               big ? "skipped(large)" : "ran", mutate ? "n/a" : (corr_only ? "not-gated(corr)" : (faster ? "YES" : "NO")), totbad ? "FAIL" : "PASS");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hC); free(ptx);
        return totbad ? 1 : 0;
    }

    /* gemm_atb mode (T2/M4): cuda_launch <combined.ptx> tiled_matmul_atb 0 gemm_atb <M> <K> <N> [mutate].
     * The SMEM-tiled A^T@B (emit_ptx_tiled_matmul_t mode 1) vs (a) a CPU A^T@B oracle for
     * CORRECTNESS and (b) the naive non-tiled gpu_matmul_atb for the FASTER-THAN-NAIVE bar.
     * combined.ptx carries BOTH kernels (concat tiled_matmul_atb_kernel.hx + gpu_matmul_atb_kernel.hx).
     * A[M,K] (a[t*K+i], t=contraction), B[M,N] (b[t*N+j]), C[K,N] (c[i*N+j]); C[i,j]=sum_t A[t,i]*B[t,j],
     * contraction length = M. Tiled: grid=(N/64,K/64) block=(16,16); naive: grid.x=K block.x=N (N<=1024).
     * Integer inputs keep cell sums < 2^24 for M up to ~7e5 -> EXACT == compare.
     * Requires K%64==0, N%64==0, M%8==0 (tiled, asserted). */
    if (strcmp(op, "gemm_atb") == 0) {
        int M = (argc > 5) ? atoi(argv[5]) : 64;
        int K = (argc > 6) ? atoi(argv[6]) : 64;
        int Nn = (argc > 7) ? atoi(argv[7]) : 64;
        int mutate = (argc > 8 && strcmp(argv[8], "mutate") == 0);
        int corr_only = (argc > 8 && strcmp(argv[8], "corr") == 0);   /* correctness-only; speedup reported not gated */
        if (K % 64 != 0 || Nn % 64 != 0 || M % 8 != 0) {
            fprintf(stderr, "gemm_atb: dims must satisfy K%%64==0 (%d), N%%64==0 (%d), M%%8==0 (%d)\n", K, Nn, M);
            return 2;
        }
        if (Nn > 1024) { fprintf(stderr, "gemm_atb: N<=1024 (naive baseline blockDim.x=N); got %d\n", Nn); return 2; }
        CUfunction f_naive; CK(cuModuleGetFunction(&f_naive, mod, "gpu_matmul_atb"), "get gpu_matmul_atb");
        size_t aN = (size_t)M * K, bN = (size_t)M * Nn, cN = (size_t)K * Nn;
        float* hA = (float*)malloc(aN * sizeof(float));
        float* hB = (float*)malloc(bN * sizeof(float));
        float* hC = (float*)malloc(cN * sizeof(float));
        if (!hA || !hB || !hC) return 2;
        for (size_t i = 0; i < aN; i++) hA[i] = (float)(i % 7);
        for (size_t i = 0; i < bN; i++) hB[i] = (float)(i % 5);
        CUdeviceptr dA, dB, dC;
        CK(cuMemAlloc(&dA, aN * sizeof(float)), "alloc A");
        CK(cuMemAlloc(&dB, bN * sizeof(float)), "alloc B");
        CK(cuMemAlloc(&dC, cN * sizeof(float)), "alloc C");
        CK(cuMemcpyHtoD(dA, hA, aN * sizeof(float)), "H2D A");
        CK(cuMemcpyHtoD(dB, hB, bN * sizeof(float)), "H2D B");
        { float neg = -1.0f; for (size_t i = 0; i < cN; i++) hC[i] = neg; }
        CK(cuMemcpyHtoD(dC, hC, cN * sizeof(float)), "H2D C(sentinel)");
        void* gargs[] = { &dA, &dB, &dC, &M, &K, &Nn };
        unsigned gx = (unsigned)(Nn / 64), gy = (unsigned)(K / 64);
        CK(cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, 0, gargs, 0), "launch tiled_matmul_atb");
        CK(cuCtxSynchronize(), "sync tiled_matmul_atb");
        CK(cuMemcpyDtoH(hC, dC, cN * sizeof(float)), "D2H C");
        if (mutate) hC[0] += 1.0f;
        double cpu_work = (double)K * (double)Nn * (double)M;
        int big = (cpu_work > 4.0e8);
        int cbad = 0;
        if (!big) for (int i = 0; i < K; i++) for (int j = 0; j < Nn; j++) {
            float ref = hA[0 * K + i] * hB[0 * Nn + j];
            for (int t = 1; t < M; t++) ref += hA[t * K + i] * hB[t * Nn + j];
            float got = hC[i * Nn + j];
            if (got != ref) { if (cbad < 6) fprintf(stderr, "gemm_atb mismatch C[%d,%d]=%g ref %g\n", i, j, got, ref); cbad++; }
        }
        double speedup = 0.0; int nbad = 0;
        if (!mutate) {
            int WARM = 3, ITERS = 30;
            float* ts = (float*)malloc((size_t)ITERS * sizeof(float));
            CUevent e0, e1; CK(cuEventCreate(&e0, 0), "evt0"); CK(cuEventCreate(&e1, 0), "evt1");
            float* hCn = (float*)malloc(cN * sizeof(float));
            void* nargs[] = { &dA, &dB, &dC, &M, &K, &Nn };
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, 0, gargs, 0), "warm tiled");
            CK(cuCtxSynchronize(), "sync warm tiled");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0,0),"r0"); CK(cuLaunchKernel(fn, gx, gy, 1, 16, 16, 1, 0, 0, gargs, 0),"t"); CK(cuEventRecord(e1,0),"r1"); CK(cuEventSynchronize(e1),"s"); CK(cuEventElapsedTime(&ts[it],e0,e1),"e"); }
            for (int i = 0; i < ITERS; i++) for (int j = i+1; j < ITERS; j++) if (ts[j] < ts[i]) { float t=ts[i]; ts[i]=ts[j]; ts[j]=t; }
            float t_tiled = ts[ITERS/2];
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(f_naive, (unsigned)K, 1, 1, (unsigned)Nn, 1, 1, 0, 0, nargs, 0), "warm naive");
            CK(cuCtxSynchronize(), "sync warm naive");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0,0),"r0"); CK(cuLaunchKernel(f_naive,(unsigned)K,1,1,(unsigned)Nn,1,1,0,0,nargs,0),"t"); CK(cuEventRecord(e1,0),"r1"); CK(cuEventSynchronize(e1),"s"); CK(cuEventElapsedTime(&ts[it],e0,e1),"e"); }
            for (int i = 0; i < ITERS; i++) for (int j = i+1; j < ITERS; j++) if (ts[j] < ts[i]) { float t=ts[i]; ts[i]=ts[j]; ts[j]=t; }
            float t_naive = ts[ITERS/2];
            CK(cuMemcpyDtoH(hCn, dC, cN * sizeof(float)), "D2H Cn(naive)");
            for (int i = 0; i < K; i++) for (int j = 0; j < Nn; j++) if (hCn[i*Nn+j] != hC[i*Nn+j]) { if (nbad < 6) fprintf(stderr, "gemm_atb tiled-vs-naive mismatch C[%d,%d] tiled=%g naive=%g\n", i, j, hC[i*Nn+j], hCn[i*Nn+j]); nbad++; }
            speedup = (t_naive > 0.0f) ? (double)t_naive / (double)t_tiled : 0.0;
            printf("GPU [%s] SPEEDUP-ATB %dx%dx%d: tiled med=%.4f ms  naive med=%.4f ms  speedup=%.2fx\n", gpu, M, K, Nn, t_tiled, t_naive, speedup);
            cuEventDestroy(e0); cuEventDestroy(e1); free(ts); free(hCn);
        }
        int faster = (mutate || corr_only) ? 1 : (speedup > 1.0);
        int totbad = cbad + nbad + (faster ? 0 : 1);
        printf("GPU [%s] tiled_matmul_atb (A^T@B, SMEM 64x64/BK8/4x4) %dx%dx%d over %d cells%s: C[1,1]=%g, vs-CPU=%d vs-naive=%d [CPU:%s] faster-than-naive=%s -> %s\n",
               gpu, M, K, Nn, K * Nn, mutate ? " [MUTATED]" : "", hC[1 * Nn + 1], cbad, nbad,
               big ? "skipped(large)" : "ran", mutate ? "n/a" : (corr_only ? "not-gated(corr)" : (faster ? "YES" : "NO")), totbad ? "FAIL" : "PASS");
        cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hA); free(hB); free(hC); free(ptx);
        return totbad ? 1 : 0;
    }

    /* softmax mode: cuda_launch <ptx> <kernel> <Nignored> softmax <rows> <cols>.
     * One thread per row (gridDim.x=rows, blockDim.x=1). Non-constant inputs so each
     * row differs; verify every cell vs a CPU max-subtract softmax (tol 1e-3 for
     * ex2.approx) AND that each row sums to ~1. */
    if (strcmp(op, "softmax") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 8;
        int cols = (argc > 6) ? atoi(argv[6]) : 16;
        size_t ne = (size_t)rows * cols;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hy = (float*)malloc(ne * sizeof(float));
        if (!hx || !hy) return 2;
        for (size_t i = 0; i < ne; i++) { hx[i] = (float)((int)((i * 7 + 3) % 13) - 6); hy[i] = -1.0f; }
        CUdeviceptr dx, dy;
        CK(cuMemAlloc(&dx, ne * sizeof(float)), "cuMemAlloc x");
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "cuMemAlloc y");
        CK(cuMemcpyHtoD(dx, hx, ne * sizeof(float)), "cuMemcpyHtoD x");
        void* sargs[] = { &dx, &dy, &rows, &cols };
        CK(cuLaunchKernel(fn, rows, 1, 1, 1, 1, 1, 0, 0, sargs, 0), "cuLaunchKernel softmax");
        CK(cuCtxSynchronize(), "cuCtxSynchronize");
        CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "cuMemcpyDtoH y");
        int sbad = 0;
        for (int r = 0; r < rows; r++) {
            float mx = hx[r * cols];
            for (int j = 1; j < cols; j++) if (hx[r * cols + j] > mx) mx = hx[r * cols + j];
            float sm = 0.0f;
            for (int kk = 0; kk < cols; kk++) sm += expf(hx[r * cols + kk] - mx);
            float rowsum = 0.0f;
            for (int c = 0; c < cols; c++) {
                float ref = expf(hx[r * cols + c] - mx) / sm;
                float got = hy[r * cols + c];
                rowsum += got;
                float d = got - ref; if (d < 0) d = -d;
                if (isnan(got) || d > 1.0e-3f) { if (sbad < 4) fprintf(stderr, "softmax mismatch y[%d,%d]=%g ref %g\n", r, c, got, ref); sbad++; }
            }
            float ds = rowsum - 1.0f; if (ds < 0) ds = -ds;
            if (isnan(rowsum) || ds > 1.0e-3f) { if (sbad < 4) fprintf(stderr, "softmax row %d sum %g (want 1)\n", r, rowsum); sbad++; }
        }
        float r0s = 0.0f; for (int c = 0; c < cols; c++) r0s += hy[c];
        printf("GPU [%s] softmax %dx%d: row0 sum=%g (want 1), %d bad -> %s\n",
               gpu, rows, cols, r0s, sbad, sbad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy); free(ptx);
        return sbad ? 1 : 0;
    }

    /* softmax_perf mode (T2/M4): cuda_launch <combined.ptx> softmax_blockred <Nignored>
     *   softmax_perf <rows> <cols> [mutate].
     * Correctness + faster-than-naive harness for the WARP/BLOCK-REDUCTION row softmax.
     * combined.ptx carries BOTH the kovc block-reduction kernel (argv[2], here
     * softmax_blockred) AND the naive one-thread-per-row gpu_softmax (the baseline).
     *  (a) CORRECTNESS: every cell of the block-reduction y vs a CPU stable-softmax
     *      reference (expf, max-subtracted); report maxrel + a per-row sum~1 check. The
     *      tol accounts HONESTLY for the kernel's ex2.approx exp (sm_86 ~2^-22) -- tol
     *      SM_TOL (default 1e-3, the same the naive softmax mode uses).
     *  (b) TIMING (kernel-only, cuEvent): block-reduction grid=(rows,1,1) block=(256,1,1)
     *      vs naive grid=(rows,1,1) block=(1,1,1); report medians + speedup-vs-naive.
     * The optional "mutate" perturbs one y cell pre-compare (comparator negative control
     * -> must FAIL). The bar.sync-strip neg-control is driven by the corpus script. */
    if (strcmp(op, "softmax_perf") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 64;
        int cols = (argc > 6) ? atoi(argv[6]) : 256;
        int mutate = (argc > 7 && strcmp(argv[7], "mutate") == 0);
        float SM_TOL = 1.0e-3f; { const char* e; if ((e = getenv("SM_TOL"))) SM_TOL = (float)atof(e); }
        CUfunction f_naive;
        CK(cuModuleGetFunction(&f_naive, mod, "gpu_softmax"), "get gpu_softmax (naive baseline)");
        size_t ne = (size_t)rows * cols;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hy = (float*)malloc(ne * sizeof(float));
        if (!hx || !hy) return 2;
        for (size_t i = 0; i < ne; i++) { hx[i] = (float)((int)((i * 7 + 3) % 13) - 6); hy[i] = -1.0f; }
        CUdeviceptr dx, dy;
        CK(cuMemAlloc(&dx, ne * sizeof(float)), "alloc x");
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "alloc y");
        CK(cuMemcpyHtoD(dx, hx, ne * sizeof(float)), "H2D x");
        CK(cuMemcpyHtoD(dy, hy, ne * sizeof(float)), "H2D y(sentinel)");
        void* sargs[] = { &dx, &dy, &rows, &cols };
        /* block-reduction launch: ONE block of 256 threads per row. */
        CK(cuLaunchKernel(fn, rows, 1, 1, 256, 1, 1, 0, 0, sargs, 0), "launch softmax_blockred");
        CK(cuCtxSynchronize(), "sync softmax_blockred");
        CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "D2H y");
        if (mutate) hy[0] += 1.0f;   /* comparator negative control */
        /* CPU stable-softmax reference: maxrel + per-row sum~1. */
        int sbad = 0; float maxrel = 0.0f, maxsumerr = 0.0f;
        for (int r = 0; r < rows; r++) {
            float mx = hx[r * cols];
            for (int j = 1; j < cols; j++) if (hx[r * cols + j] > mx) mx = hx[r * cols + j];
            float sm = 0.0f;
            for (int kk = 0; kk < cols; kk++) sm += expf(hx[r * cols + kk] - mx);
            float rowsum = 0.0f;
            for (int c = 0; c < cols; c++) {
                float ref = expf(hx[r * cols + c] - mx) / sm;
                float got = hy[r * cols + c];
                rowsum += got;
                float d = got - ref; if (d < 0) d = -d;
                float ar = ref < 0 ? -ref : ref; float rel = d / (ar > 1e-6f ? ar : 1e-6f);
                if (rel > maxrel) maxrel = rel;
                if (isnan(got) || (d > SM_TOL && rel > SM_TOL)) { if (sbad < 6) fprintf(stderr, "softmax_perf mismatch y[%d,%d]=%g ref %g (rel %.2e)\n", r, c, got, ref, rel); sbad++; }
            }
            float ds = rowsum - 1.0f; if (ds < 0) ds = -ds; if (ds > maxsumerr) maxsumerr = ds;
            if (isnan(rowsum) || ds > 1.0e-3f) { if (sbad < 6) fprintf(stderr, "softmax_perf row %d sum %g (want 1)\n", r, rowsum); sbad++; }
        }
        /* TIMING (kernel-only): block-reduction vs naive. */
        double br_med = 0.0, nv_med = 0.0;
        if (!mutate) {
            int WARM = 5, ITERS = 50;
            float* ts = (float*)malloc((size_t)ITERS * sizeof(float));
            CUevent e0, e1; CK(cuEventCreate(&e0, 0), "evt0"); CK(cuEventCreate(&e1, 0), "evt1");
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(fn, rows, 1, 1, 256, 1, 1, 0, 0, sargs, 0), "warm br");
            CK(cuCtxSynchronize(), "sync warm br");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0, 0), "r0"); CK(cuLaunchKernel(fn, rows, 1, 1, 256, 1, 1, 0, 0, sargs, 0), "time br"); CK(cuEventRecord(e1, 0), "r1"); CK(cuEventSynchronize(e1), "es"); CK(cuEventElapsedTime(&ts[it], e0, e1), "el"); }
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            br_med = (double)ts[ITERS / 2];
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(f_naive, rows, 1, 1, 1, 1, 1, 0, 0, sargs, 0), "warm nv");
            CK(cuCtxSynchronize(), "sync warm nv");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0, 0), "n0"); CK(cuLaunchKernel(f_naive, rows, 1, 1, 1, 1, 1, 0, 0, sargs, 0), "time nv"); CK(cuEventRecord(e1, 0), "n1"); CK(cuEventSynchronize(e1), "nes"); CK(cuEventElapsedTime(&ts[it], e0, e1), "nel"); }
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            nv_med = (double)ts[ITERS / 2];
            cuEventDestroy(e0); cuEventDestroy(e1); free(ts);
            printf("GPU [%s] TIMING softmax %dx%d: blockred med=%.4f ms  naive med=%.4f ms  SPEEDUP=%.2fx\n", gpu, rows, cols, br_med, nv_med, nv_med / br_med);
        }
        printf("GPU [%s] softmax_perf (block-reduction 256t/row) %dx%d%s: maxrel=%.2e (tol=%.0e) max|rowsum-1|=%.2e, %d bad -> %s\n",
               gpu, rows, cols, mutate ? " [MUTATED]" : "", maxrel, SM_TOL, maxsumerr, sbad, sbad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy); free(ptx);
        return sbad ? 1 : 0;
    }

    /* gelu_backward mode: cuda_launch <ptx> gpu_gelu_backward <N> gelu_backward.
     * dx = dy * gelu'(x). INDEPENDENT ref = central finite-difference of the GELU FORWARD
     * (uses tanhf, does NOT share the kernel's analytic derivative): gp_fd=(gelu(x+h)-gelu(x-h))/2h,
     * h=1e-3, dx_ref=dy*gp_fd, tol 2e-2; plus a tighter analytic gelu' check at tol 1e-3.
     * Inputs in [-3,3] so e^2z never saturates. */
    if (strcmp(op, "gelu_backward") == 0) {
        size_t ne = (size_t)N;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hdy = (float*)malloc(ne * sizeof(float));
        float* hdx = (float*)malloc(ne * sizeof(float));
        if (!hx || !hdy || !hdx) return 2;
        for (size_t i = 0; i < ne; i++) { hx[i] = (float)((int)(i % 61) - 30) * 0.1f; hdy[i] = (float)((int)(i % 7) - 3); }
        CUdeviceptr dxb, ddy, ddx;
        CK(cuMemAlloc(&dxb, ne * sizeof(float)), "alloc x");
        CK(cuMemAlloc(&ddy, ne * sizeof(float)), "alloc dy");
        CK(cuMemAlloc(&ddx, ne * sizeof(float)), "alloc dx");
        CK(cuMemcpyHtoD(dxb, hx, ne * sizeof(float)), "H2D x");
        CK(cuMemcpyHtoD(ddy, hdy, ne * sizeof(float)), "H2D dy");
        void* gargs[] = { &dxb, &ddy, &ddx, &N };
        CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, gargs, 0), "launch gelu_bwd");
        CK(cuCtxSynchronize(), "sync gelu_bwd");
        CK(cuMemcpyDtoH(hdx, ddx, ne * sizeof(float)), "D2H dx");
        int gbad = 0; float maxfd = 0.0f, g0 = 0.0f, r0 = 0.0f;
        for (size_t i = 0; i < ne; i++) {
            float xx = hx[i];
            float inn = 0.7978846f * (xx + 0.044715f * xx * xx * xx);
            float e2 = expf(2.0f * inn); float th = (e2 - 1.0f) / (e2 + 1.0f);
            float idv = 0.7978846f * (1.0f + 0.134145f * xx * xx);
            float gp_a = 0.5f * (1.0f + th) + 0.5f * xx * (1.0f - th * th) * idv;
            float ref_a = hdy[i] * gp_a;
            float h = 1.0e-3f, xp = xx + h, xm = xx - h;
            float gpx = 0.5f * xp * (1.0f + tanhf(0.7978846f * (xp + 0.044715f * xp * xp * xp)));
            float gmx = 0.5f * xm * (1.0f + tanhf(0.7978846f * (xm + 0.044715f * xm * xm * xm)));
            float ref_fd = hdy[i] * (gpx - gmx) / (2.0f * h);
            float got = hdx[i];
            if (i == 0) { g0 = got; r0 = ref_fd; }
            float ea = got - ref_a; if (ea < 0) ea = -ea;
            float ef = got - ref_fd; if (ef < 0) ef = -ef; if (ef > maxfd) maxfd = ef;
            if (isnan(got) || ea > 1.0e-3f || ef > 2.0e-2f) { if (gbad < 4) fprintf(stderr, "gelu_bwd mismatch dx[%zu]=%g ref_a %g ref_fd %g (x=%g)\n", i, got, ref_a, ref_fd, xx); gbad++; }
        }
        printf("GPU [%s] gelu_backward N=%d: dx[0]=%g ref_fd %g, max|dx-fd|=%g, %d bad -> %s\n",
               gpu, N, g0, r0, maxfd, gbad, gbad ? "FAIL" : "PASS");
        cuMemFree(dxb); cuMemFree(ddy); cuMemFree(ddx);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hdy); free(hdx); free(ptx);
        return gbad ? 1 : 0;
    }

    /* softmax_backward mode: cuda_launch <ptx> gpu_softmax_backward <N> softmax_backward <rows> <cols>.
     * dA[i,j]=P[i,j]*(dP[i,j]-sum_k dP[i,k]*P[i,k]). P is a valid CPU softmax, dP a varied upstream
     * grad. Verify per-cell vs CPU (1e-4) AND the INDEPENDENT conservation that each dA row sums
     * to ~0 (sum_j dA = dot - 1*dot = 0 since sum_j P = 1). */
    if (strcmp(op, "softmax_backward") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 8;
        int cols = (argc > 6) ? atoi(argv[6]) : 16;
        size_t ne = (size_t)rows * cols;
        float* hP = (float*)malloc(ne * sizeof(float));
        float* hdP = (float*)malloc(ne * sizeof(float));
        float* hdA = (float*)malloc(ne * sizeof(float));
        if (!hP || !hdP || !hdA) return 2;
        for (int r = 0; r < rows; r++) {
            float mx = -1.0e30f;
            for (int c = 0; c < cols; c++) { float z = (float)((int)(((r * cols + c) * 7 + 3) % 13) - 6); hP[r * cols + c] = z; if (z > mx) mx = z; }
            float sm = 0.0f; for (int c = 0; c < cols; c++) sm += expf(hP[r * cols + c] - mx);
            for (int c = 0; c < cols; c++) hP[r * cols + c] = expf(hP[r * cols + c] - mx) / sm;
            for (int c = 0; c < cols; c++) hdP[r * cols + c] = (float)((int)(((r * cols + c) * 5 + 1) % 7) - 3);
        }
        CUdeviceptr dP, ddP, ddA;
        CK(cuMemAlloc(&dP, ne * sizeof(float)), "alloc P");
        CK(cuMemAlloc(&ddP, ne * sizeof(float)), "alloc dP");
        CK(cuMemAlloc(&ddA, ne * sizeof(float)), "alloc dA");
        CK(cuMemcpyHtoD(dP, hP, ne * sizeof(float)), "H2D P");
        CK(cuMemcpyHtoD(ddP, hdP, ne * sizeof(float)), "H2D dP");
        void* sargs[] = { &dP, &ddP, &ddA, &rows, &cols };
        int tb_sb = getenv("CL_BLOCK") ? atoi(getenv("CL_BLOCK")) : 1;  /* 256 for blockred kernel */
        CK(cuLaunchKernel(fn, rows, 1, 1, tb_sb, 1, 1, 0, 0, sargs, 0), "launch softmax_bwd");
        CK(cuCtxSynchronize(), "sync softmax_bwd");
        CK(cuMemcpyDtoH(hdA, ddA, ne * sizeof(float)), "D2H dA");
        int sbad = 0; float maxrs = 0.0f, a0 = 0.0f, r0 = 0.0f;
        for (int r = 0; r < rows; r++) {
            float dot = 0.0f; for (int c = 0; c < cols; c++) dot += hdP[r * cols + c] * hP[r * cols + c];
            float rs = 0.0f;
            for (int c = 0; c < cols; c++) {
                float ref = hP[r * cols + c] * (hdP[r * cols + c] - dot);
                float got = hdA[r * cols + c];
                if (r == 0 && c == 0) { a0 = got; r0 = ref; }
                rs += got;
                float e = got - ref; if (e < 0) e = -e;
                if (isnan(got) || e > 1.0e-4f) { if (sbad < 4) fprintf(stderr, "softmax_bwd mismatch dA[%d,%d]=%g ref %g\n", r, c, got, ref); sbad++; }
            }
            float ers = rs < 0 ? -rs : rs; if (ers > maxrs) maxrs = ers;
            if (isnan(rs) || ers > 1.0e-3f) { if (sbad < 4) fprintf(stderr, "softmax_bwd row %d sum %g (want 0)\n", r, rs); sbad++; }
        }
        printf("GPU [%s] softmax_backward %dx%d: dA[0,0]=%g ref %g, max|row_sum|=%g, %d bad -> %s\n",
               gpu, rows, cols, a0, r0, maxrs, sbad, sbad ? "FAIL" : "PASS");
        cuMemFree(dP); cuMemFree(ddP); cuMemFree(ddA);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hP); free(hdP); free(hdA); free(ptx);
        return sbad ? 1 : 0;
    }

    /* layernorm mode: cuda_launch <ptx> <kernel> <Nignored> layernorm <rows> <cols>.
     * TWO passes. pass 0: gamma=1,beta=0 -> verify each cell vs a CPU layernorm AND
     * that each row of y has mean~0, var~1. pass 1: non-trivial per-column gamma/beta
     * -> verify each cell vs an affine-aware reference gamma[c]*norm+beta[c], which
     * exercises the kernel's gamma[t]/beta[t] load+store path (a kernel that ignored
     * gamma/beta would pass pass 0 but mismatch pass 1). All checks NaN-guarded; the
     * real kernel must pass BOTH passes. */
    if (strcmp(op, "layernorm") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 8;
        int cols = (argc > 6) ? atoi(argv[6]) : 16;
        size_t ne = (size_t)rows * cols;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hy = (float*)malloc(ne * sizeof(float));
        float* hg = (float*)malloc((size_t)cols * sizeof(float));
        float* hbe = (float*)malloc((size_t)cols * sizeof(float));
        if (!hx || !hy || !hg || !hbe) return 2;
        for (size_t i = 0; i < ne; i++) { hx[i] = (float)((int)((i * 7 + 3) % 13) - 6); }
        CUdeviceptr dx, dy, dg, db;
        CK(cuMemAlloc(&dx, ne * sizeof(float)), "cuMemAlloc x");
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "cuMemAlloc y");
        CK(cuMemAlloc(&dg, (size_t)cols * sizeof(float)), "cuMemAlloc g");
        CK(cuMemAlloc(&db, (size_t)cols * sizeof(float)), "cuMemAlloc b");
        CK(cuMemcpyHtoD(dx, hx, ne * sizeof(float)), "cuMemcpyHtoD x");
        void* largs[] = { &dx, &dy, &dg, &db, &cols };
        int lbad = 0;
        float r0m = 0.0f;
        for (int pass = 0; pass < 2; pass++) {
            for (int c = 0; c < cols; c++) {
                if (pass == 0) { hg[c] = 1.0f; hbe[c] = 0.0f; }
                else { hg[c] = 1.0f + 0.25f * (float)(c % 4); hbe[c] = 0.5f * (float)((c % 3) - 1); }
            }
            for (size_t i = 0; i < ne; i++) hy[i] = -7.0f;
            CK(cuMemcpyHtoD(dy, hy, ne * sizeof(float)), "cuMemcpyHtoD y");
            CK(cuMemcpyHtoD(dg, hg, (size_t)cols * sizeof(float)), "cuMemcpyHtoD g");
            CK(cuMemcpyHtoD(db, hbe, (size_t)cols * sizeof(float)), "cuMemcpyHtoD b");
            CK(cuLaunchKernel(fn, rows, 1, 1, 1, 1, 1, 0, 0, largs, 0), "cuLaunchKernel layernorm");
            CK(cuCtxSynchronize(), "cuCtxSynchronize");
            CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "cuMemcpyDtoH y");
            for (int r = 0; r < rows; r++) {
                float mean = 0.0f; for (int c = 0; c < cols; c++) mean += hx[r * cols + c]; mean /= (float)cols;
                float v = 0.0f; for (int c = 0; c < cols; c++) { float d = hx[r * cols + c] - mean; v += d * d; } v /= (float)cols;
                float inv = 1.0f / sqrtf(v);
                for (int c = 0; c < cols; c++) {
                    float norm = (hx[r * cols + c] - mean) * inv;
                    float ref = hg[c] * norm + hbe[c];
                    float got = hy[r * cols + c];
                    float d = got - ref; if (d < 0) d = -d;
                    if (isnan(got) || d > 1.0e-3f) { if (lbad < 4) fprintf(stderr, "layernorm[g/b pass %d] mismatch y[%d,%d]=%g ref %g\n", pass, r, c, got, ref); lbad++; }
                }
                if (pass == 0) {
                    float ym = 0.0f; for (int c = 0; c < cols; c++) ym += hy[r * cols + c]; ym /= (float)cols;
                    float yv = 0.0f; for (int c = 0; c < cols; c++) { float dd = hy[r * cols + c] - ym; yv += dd * dd; } yv /= (float)cols;
                    float dm = ym < 0 ? -ym : ym;
                    float dv = (yv - 1.0f) < 0 ? -(yv - 1.0f) : (yv - 1.0f);
                    if (isnan(ym) || dm > 1.0e-2f) { if (lbad < 4) fprintf(stderr, "layernorm row %d mean %g (want 0)\n", r, ym); lbad++; }
                    if (isnan(yv) || dv > 1.0e-2f) { if (lbad < 4) fprintf(stderr, "layernorm row %d var %g (want 1)\n", r, yv); lbad++; }
                    if (r == 0) r0m = ym;
                }
            }
        }
        printf("GPU [%s] layernorm %dx%d (gamma/beta affine-checked): row0 mean=%g (want 0), %d bad -> %s\n",
               gpu, rows, cols, r0m, lbad, lbad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy); cuMemFree(dg); cuMemFree(db);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy); free(hg); free(hbe);
        return lbad ? 1 : 0;
    }

    /* layernorm_perf mode (T2/M4): cuda_launch <combined.ptx> layernorm_blockred <Nignored>
     *   layernorm_perf <rows> <cols> [mutate].
     * Correctness + faster-than-naive harness for the WARP/BLOCK-REDUCTION row LayerNorm.
     * combined.ptx carries BOTH the kovc block-reduction kernel (argv[2]) AND the naive
     * one-thread-per-row gpu_layernorm (baseline). Non-trivial per-column gamma/beta (so a
     * kernel that ignored gamma/beta would mismatch). CORRECTNESS: every cell vs a CPU
     * affine layernorm reference gamma[c]*(x-mean)/sqrt(var)+beta[c]; report maxrel. The tol
     * (LN_TOL, default 1e-3) accounts HONESTLY for rsqrt.approx. TIMING: block-reduction
     * grid=(rows,1,1) block=(256,1,1) vs naive block=(1,1,1); speedup-vs-naive. The "mutate"
     * arg perturbs one y cell (comparator neg-control -> must FAIL). */
    if (strcmp(op, "layernorm_perf") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 64;
        int cols = (argc > 6) ? atoi(argv[6]) : 256;
        int mutate = (argc > 7 && strcmp(argv[7], "mutate") == 0);
        float LN_TOL = 1.0e-3f; { const char* e; if ((e = getenv("LN_TOL"))) LN_TOL = (float)atof(e); }
        CUfunction f_naive;
        CK(cuModuleGetFunction(&f_naive, mod, "gpu_layernorm"), "get gpu_layernorm (naive baseline)");
        size_t ne = (size_t)rows * cols;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hy = (float*)malloc(ne * sizeof(float));
        float* hg = (float*)malloc((size_t)cols * sizeof(float));
        float* hbe = (float*)malloc((size_t)cols * sizeof(float));
        if (!hx || !hy || !hg || !hbe) return 2;
        for (size_t i = 0; i < ne; i++) hx[i] = (float)((int)((i * 7 + 3) % 13) - 6);
        for (int c = 0; c < cols; c++) { hg[c] = 1.0f + 0.25f * (float)(c % 4); hbe[c] = 0.5f * (float)((c % 3) - 1); }
        CUdeviceptr dx, dy, dg, db;
        CK(cuMemAlloc(&dx, ne * sizeof(float)), "alloc x");
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "alloc y");
        CK(cuMemAlloc(&dg, (size_t)cols * sizeof(float)), "alloc g");
        CK(cuMemAlloc(&db, (size_t)cols * sizeof(float)), "alloc b");
        for (size_t i = 0; i < ne; i++) hy[i] = -7.0f;
        CK(cuMemcpyHtoD(dx, hx, ne * sizeof(float)), "H2D x");
        CK(cuMemcpyHtoD(dy, hy, ne * sizeof(float)), "H2D y(sentinel)");
        CK(cuMemcpyHtoD(dg, hg, (size_t)cols * sizeof(float)), "H2D g");
        CK(cuMemcpyHtoD(db, hbe, (size_t)cols * sizeof(float)), "H2D b");
        void* largs[] = { &dx, &dy, &dg, &db, &cols };
        CK(cuLaunchKernel(fn, rows, 1, 1, 256, 1, 1, 0, 0, largs, 0), "launch layernorm_blockred");
        CK(cuCtxSynchronize(), "sync layernorm_blockred");
        CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "D2H y");
        if (mutate) hy[0] += 1.0f;
        /* CPU affine layernorm reference: maxrel. */
        int lbad = 0; float maxrel = 0.0f;
        for (int r = 0; r < rows; r++) {
            float mean = 0.0f; for (int c = 0; c < cols; c++) mean += hx[r * cols + c]; mean /= (float)cols;
            float v = 0.0f; for (int c = 0; c < cols; c++) { float d = hx[r * cols + c] - mean; v += d * d; } v /= (float)cols;
            float inv = 1.0f / sqrtf(v);
            for (int c = 0; c < cols; c++) {
                float norm = (hx[r * cols + c] - mean) * inv;
                float ref = hg[c] * norm + hbe[c];
                float got = hy[r * cols + c];
                float d = got - ref; if (d < 0) d = -d;
                float ar = ref < 0 ? -ref : ref; float rel = d / (ar > 1e-6f ? ar : 1e-6f);
                if (rel > maxrel) maxrel = rel;
                if (isnan(got) || (d > LN_TOL && rel > LN_TOL)) { if (lbad < 6) fprintf(stderr, "layernorm_perf mismatch y[%d,%d]=%g ref %g (rel %.2e)\n", r, c, got, ref, rel); lbad++; }
            }
        }
        /* TIMING (kernel-only): block-reduction vs naive. */
        double br_med = 0.0, nv_med = 0.0;
        if (!mutate) {
            int WARM = 5, ITERS = 50;
            float* ts = (float*)malloc((size_t)ITERS * sizeof(float));
            CUevent e0, e1; CK(cuEventCreate(&e0, 0), "evt0"); CK(cuEventCreate(&e1, 0), "evt1");
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(fn, rows, 1, 1, 256, 1, 1, 0, 0, largs, 0), "warm br");
            CK(cuCtxSynchronize(), "sync warm br");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0, 0), "r0"); CK(cuLaunchKernel(fn, rows, 1, 1, 256, 1, 1, 0, 0, largs, 0), "time br"); CK(cuEventRecord(e1, 0), "r1"); CK(cuEventSynchronize(e1), "es"); CK(cuEventElapsedTime(&ts[it], e0, e1), "el"); }
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            br_med = (double)ts[ITERS / 2];
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(f_naive, rows, 1, 1, 1, 1, 1, 0, 0, largs, 0), "warm nv");
            CK(cuCtxSynchronize(), "sync warm nv");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0, 0), "n0"); CK(cuLaunchKernel(f_naive, rows, 1, 1, 1, 1, 1, 0, 0, largs, 0), "time nv"); CK(cuEventRecord(e1, 0), "n1"); CK(cuEventSynchronize(e1), "nes"); CK(cuEventElapsedTime(&ts[it], e0, e1), "nel"); }
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            nv_med = (double)ts[ITERS / 2];
            cuEventDestroy(e0); cuEventDestroy(e1); free(ts);
            printf("GPU [%s] TIMING layernorm %dx%d: blockred med=%.4f ms  naive med=%.4f ms  SPEEDUP=%.2fx\n", gpu, rows, cols, br_med, nv_med, nv_med / br_med);
        }
        printf("GPU [%s] layernorm_perf (block-reduction 256t/row, affine) %dx%d%s: maxrel=%.2e (tol=%.0e), %d bad -> %s\n",
               gpu, rows, cols, mutate ? " [MUTATED]" : "", maxrel, LN_TOL, lbad, lbad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy); cuMemFree(dg); cuMemFree(db);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy); free(hg); free(hbe);
        return lbad ? 1 : 0;
    }

    /* gelu mode: cuda_launch <ptx> gpu_gelu <N> gelu. y=0.5*x*(1+tanh(0.7978846*
     * (x+0.044715*x^3))), the tanh GELU. CPU ref mirrors stdlib __gelu exactly.
     * Inputs in [-3,3] so e^(2z) never saturates. Combines f32 literals + __gpu_exp. */
    if (strcmp(op, "gelu") == 0) {
        /* optional "mutate" (argv[5], after <ptx> <kname> <N> <op>) = comparator-teeth control. */
        int mutate = (argc > 5 && strcmp(argv[5], "mutate") == 0);
        size_t ne = (size_t)N;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hy = (float*)malloc(ne * sizeof(float));
        if (!hx || !hy) return 2;
        for (size_t i = 0; i < ne; i++) { hx[i] = (float)((int)(i % 61) - 30) * 0.1f; hy[i] = -7.0f; }
        CUdeviceptr dx, dy;
        CK(cuMemAlloc(&dx, ne * sizeof(float)), "cuMemAlloc x");
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "cuMemAlloc y");
        CK(cuMemcpyHtoD(dx, hx, ne * sizeof(float)), "cuMemcpyHtoD x");
        CK(cuMemcpyHtoD(dy, hy, ne * sizeof(float)), "cuMemcpyHtoD y");
        void* gargs[] = { &dx, &dy, &N };
        CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, gargs, 0), "cuLaunchKernel gelu");
        CK(cuCtxSynchronize(), "cuCtxSynchronize");
        CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "cuMemcpyDtoH y");
        if (mutate) hy[0] += 1.0f;   /* comparator negative control -> must FAIL */
        int gbad = 0; float gref0 = 0.0f, maxrel = 0.0f, maxabs = 0.0f;
        for (size_t i = 0; i < ne; i++) {
            float xx = hx[i];
            float x3 = xx * xx * xx;
            float inner = 0.7978846f * (xx + 0.044715f * x3);
            float e2 = expf(2.0f * inner);
            float th = (e2 - 1.0f) / (e2 + 1.0f);
            float ref = 0.5f * xx * (1.0f + th);
            if (i == 0) gref0 = ref;
            float got = hy[i];
            float d = got - ref; if (d < 0) d = -d; if (d > maxabs) maxabs = d;
            float ar = ref < 0 ? -ref : ref; float rel = d / (ar > 1e-6f ? ar : 1e-6f);
            if (rel > maxrel) maxrel = rel;
            if (isnan(got) || d > 1.0e-3f) { if (gbad < 4) fprintf(stderr, "gelu mismatch y[%zu]=%g ref %g (x=%g)\n", i, got, ref, xx); gbad++; }
        }
        /* THROUGHPUT (kernel-only, cuEvent): elementwise GELU is MEMORY-BOUND
         * (read N + write N f32 = 8N bytes). NO naive/tiled pair exists -- this
         * IS the elementwise form -- so we report GB/s honestly, never a fake
         * speedup. Gate rests on correctness (gbad==0). */
        if (!mutate) {
            int WARM = 5, ITERS = 50;
            float* ts = (float*)malloc((size_t)ITERS * sizeof(float));
            CUevent e0, e1; CK(cuEventCreate(&e0, 0), "evt0"); CK(cuEventCreate(&e1, 0), "evt1");
            for (int w = 0; w < WARM; w++) CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, gargs, 0), "warm gelu");
            CK(cuCtxSynchronize(), "sync warm gelu");
            for (int it = 0; it < ITERS; it++) { CK(cuEventRecord(e0, 0), "r0"); CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, gargs, 0), "time gelu"); CK(cuEventRecord(e1, 0), "r1"); CK(cuEventSynchronize(e1), "es"); CK(cuEventElapsedTime(&ts[it], e0, e1), "el"); }
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            double med = (double)ts[ITERS / 2];
            double gbps = (8.0 * (double)ne) / (med * 1.0e-3) / 1.0e9;
            cuEventDestroy(e0); cuEventDestroy(e1); free(ts);
            printf("GPU [%s] THROUGHPUT gelu N=%d: med=%.4f ms  %.1f GB/s (8N B, memory-bound elementwise)\n", gpu, N, med, gbps);
        }
        printf("GPU [%s] gelu N=%d (tanh approx, f32 literals + exp)%s: y[0]=%g ref %g, maxrel=%.2e maxabs=%.2e (tol 1e-3), %d bad -> %s\n",
               gpu, N, mutate ? " [MUTATED]" : "", hy[0], gref0, maxrel, maxabs, gbad, gbad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy);
        return gbad ? 1 : 0;
    }

    /* gelu_big mode (P5 GPT-2): cuda_launch <ptx> gpu_gelu_stable <N> gelu_big [mutate].
     * The SAME gelu_new (tanh-approx) correctness check as `gelu`, but over GPT-2-SCALE inputs in
     * [-12,12] (GPT-2's c_fc pre-activation reaches ~+/-12). At x~12 the tanh argument z reaches ~63,
     * so the DIRECT form e^(2z)=e^126 OVERFLOWS f32 -> +inf -> (inf-1)/(inf+1)=NaN. The reference is
     * the OVERFLOW-SAFE f64 tanh (tanhf would also be stable); the kovc kernel must match it within
     * tol 1e-3 AND stay finite at every cell. The committed `gpu_gelu` (direct e^(2z)) FAILS this gate
     * by producing NaN around x~11.5 (proving the gate is load-bearing); `gpu_gelu_stable` PASSES.
     * "mutate" perturbs y[0] pre-compare (comparator teeth). */
    if (strcmp(op, "gelu_big") == 0) {
        int mutate = (argc > 5 && strcmp(argv[5], "mutate") == 0);
        size_t ne = (size_t)N;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hy = (float*)malloc(ne * sizeof(float));
        if (!hx || !hy) return 2;
        /* sweep [-12,12] across N cells (deterministic), so x~11.5 (the empirical NaN point) is hit. */
        for (size_t i = 0; i < ne; i++) { hx[i] = -12.0f + 24.0f * ((float)(i % 241) / 240.0f); hy[i] = -7.0f; }
        CUdeviceptr dx, dy;
        CK(cuMemAlloc(&dx, ne * sizeof(float)), "alloc x");
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "alloc y");
        CK(cuMemcpyHtoD(dx, hx, ne * sizeof(float)), "H2D x");
        CK(cuMemcpyHtoD(dy, hy, ne * sizeof(float)), "H2D y(sentinel)");
        void* gargs[] = { &dx, &dy, &N };
        CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, gargs, 0), "launch gelu_big");
        CK(cuCtxSynchronize(), "sync gelu_big");
        CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "D2H y");
        if (mutate) hy[0] += 1.0f;
        int gbad = 0; float maxrel = 0.0f, maxabs = 0.0f; int nnan = 0; float y0 = hy[0], ref0 = 0.0f;
        for (size_t i = 0; i < ne; i++) {
            double xx = (double)hx[i];
            double x3 = xx * xx * xx;
            double inner = 0.7978846 * (xx + 0.044715 * x3);
            double th = tanh(inner);                       /* overflow-safe reference */
            double ref = 0.5 * xx * (1.0 + th);
            if (i == 0) ref0 = (float)ref;
            float got = hy[i];
            if (isnan(got) || isinf(got)) { nnan++; if (gbad < 4) fprintf(stderr, "gelu_big NON-FINITE y[%zu]=%g (x=%g)\n", i, got, hx[i]); gbad++; continue; }
            double d = (double)got - ref; if (d < 0) d = -d; if (d > maxabs) maxabs = (float)d;
            double ar = ref < 0 ? -ref : ref; double rel = d / (ar > 1e-6 ? ar : 1e-6);
            if (rel > maxrel) maxrel = (float)rel;
            if (d > 1.0e-3) { if (gbad < 4) fprintf(stderr, "gelu_big mismatch y[%zu]=%g ref %g (x=%g)\n", i, got, ref, hx[i]); gbad++; }
        }
        printf("GPU [%s] gelu_big N=%d (gelu_new over [-12,12], GPT-2 scale, overflow-safe ref)%s: y[0]=%g ref %g, maxrel=%.2e maxabs=%.2e non-finite=%d (tol 1e-3), %d bad -> %s\n",
               gpu, N, mutate ? " [MUTATED]" : "", y0, ref0, maxrel, maxabs, nnan, gbad, gbad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy); free(ptx);
        return gbad ? 1 : 0;
    }

    /* adam mode: cuda_launch <ptx> gpu_adam <N> adam. One in-place Adam step on w (with g,m,v).
     * bc1,bc2 (bias-correction) passed as 1-elem arrays; here bc1=10, bc2=1000 (step t=1). Verify
     * nm,nv (tol 1e-5, exact arithmetic) and new_w (tol 1e-4, rsqrt.approx) vs an independent CPU
     * Adam step with the same baked literals. */
    if (strcmp(op, "adam") == 0) {
        /* optional "mutate" (argv[5], after <ptx> <kname> <N> <op>) = comparator-teeth
         * control. Re-runs are non-idempotent (Adam mutates w,m,v in place), so timing
         * re-uploads the pristine host arrays before each launch. */
        int mutate = (argc > 5 && strcmp(argv[5], "mutate") == 0);
        size_t ne = (size_t)N;
        float* hw = (float*)malloc(ne * sizeof(float));
        float* hg = (float*)malloc(ne * sizeof(float));
        float* hm = (float*)malloc(ne * sizeof(float));
        float* hv = (float*)malloc(ne * sizeof(float));
        float* hw0 = (float*)malloc(ne * sizeof(float));
        float* hm0 = (float*)malloc(ne * sizeof(float));
        float* hv0 = (float*)malloc(ne * sizeof(float));
        if (!hw || !hg || !hm || !hv || !hw0 || !hm0 || !hv0) return 2;
        for (size_t i = 0; i < ne; i++) {
            hw[i] = hw0[i] = (float)((int)(i % 5) - 2);
            hg[i] = (float)((int)(i % 7) - 3) * 0.1f;
            hm[i] = hm0[i] = (float)((int)(i % 3) - 1) * 0.05f;
            hv[i] = hv0[i] = (float)(i % 4) * 0.01f;
        }
        float hbc1 = 10.0f, hbc2 = 1000.0f;
        CUdeviceptr dw, dg, dm, dv, dbc1, dbc2;
        CK(cuMemAlloc(&dw, ne * sizeof(float)), "alloc w");
        CK(cuMemAlloc(&dg, ne * sizeof(float)), "alloc g");
        CK(cuMemAlloc(&dm, ne * sizeof(float)), "alloc m");
        CK(cuMemAlloc(&dv, ne * sizeof(float)), "alloc v");
        CK(cuMemAlloc(&dbc1, sizeof(float)), "alloc bc1");
        CK(cuMemAlloc(&dbc2, sizeof(float)), "alloc bc2");
        CK(cuMemcpyHtoD(dw, hw, ne * sizeof(float)), "H2D w");
        CK(cuMemcpyHtoD(dg, hg, ne * sizeof(float)), "H2D g");
        CK(cuMemcpyHtoD(dm, hm, ne * sizeof(float)), "H2D m");
        CK(cuMemcpyHtoD(dv, hv, ne * sizeof(float)), "H2D v");
        CK(cuMemcpyHtoD(dbc1, &hbc1, sizeof(float)), "H2D bc1");
        CK(cuMemcpyHtoD(dbc2, &hbc2, sizeof(float)), "H2D bc2");
        void* aargs[] = { &dw, &dg, &dm, &dv, &dbc1, &dbc2 };
        CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, aargs, 0), "launch adam");
        CK(cuCtxSynchronize(), "sync adam");
        CK(cuMemcpyDtoH(hw, dw, ne * sizeof(float)), "D2H w");
        CK(cuMemcpyDtoH(hm, dm, ne * sizeof(float)), "D2H m");
        CK(cuMemcpyDtoH(hv, dv, ne * sizeof(float)), "D2H v");
        if (mutate) hw[0] += 1.0f;   /* comparator negative control -> must FAIL */
        int abad = 0; float w0 = 0.0f, w0ref = 0.0f, maxrel = 0.0f;
        for (size_t i = 0; i < ne; i++) {
            float nm = 0.9f * hm0[i] + 0.1f * hg[i];
            float nv = 0.999f * hv0[i] + 0.001f * hg[i] * hg[i];
            float mh = nm * hbc1, vh = nv * hbc2;
            float nw = hw0[i] - 0.001f * mh / sqrtf(vh + 1.0e-8f);
            if (i == 0) { w0 = hw[i]; w0ref = nw; }
            float em = hm[i] - nm; if (em < 0) em = -em;
            float ev = hv[i] - nv; if (ev < 0) ev = -ev;
            float ew = hw[i] - nw; if (ew < 0) ew = -ew;
            float aw = nw < 0 ? -nw : nw; float rw = ew / (aw > 1e-6f ? aw : 1e-6f);
            if (rw > maxrel) maxrel = rw;
            if (isnan(hw[i]) || em > 1.0e-5f || ev > 1.0e-5f || ew > 1.0e-4f) { if (abad < 4) fprintf(stderr, "adam mismatch i=%zu w=%g(ref %g) m=%g(ref %g) v=%g(ref %g)\n", i, hw[i], nw, hm[i], nm, hv[i], nv); abad++; }
        }
        /* THROUGHPUT (kernel-only, cuEvent): the Adam step is MEMORY-BOUND --
         * reads w,g,m,v + writes w,m,v = 7 array touches = 28N bytes. NO
         * naive/tiled pair (this IS the elementwise form), so report GB/s
         * honestly; gate rests on correctness (abad==0). Re-upload w,m,v each
         * launch (in-place mutation). */
        if (!mutate) {
            int WARM = 5, ITERS = 50;
            float* ts = (float*)malloc((size_t)ITERS * sizeof(float));
            CUevent e0, e1; CK(cuEventCreate(&e0, 0), "evt0"); CK(cuEventCreate(&e1, 0), "evt1");
            for (int w = 0; w < WARM; w++) {
                CK(cuMemcpyHtoD(dw, hw0, ne * sizeof(float)), "re-w"); CK(cuMemcpyHtoD(dm, hm0, ne * sizeof(float)), "re-m"); CK(cuMemcpyHtoD(dv, hv0, ne * sizeof(float)), "re-v");
                CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, aargs, 0), "warm adam");
            }
            CK(cuCtxSynchronize(), "sync warm adam");
            for (int it = 0; it < ITERS; it++) {
                CK(cuMemcpyHtoD(dw, hw0, ne * sizeof(float)), "re-w"); CK(cuMemcpyHtoD(dm, hm0, ne * sizeof(float)), "re-m"); CK(cuMemcpyHtoD(dv, hv0, ne * sizeof(float)), "re-v");
                CK(cuCtxSynchronize(), "pre-time sync");
                CK(cuEventRecord(e0, 0), "r0"); CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, aargs, 0), "time adam"); CK(cuEventRecord(e1, 0), "r1"); CK(cuEventSynchronize(e1), "es"); CK(cuEventElapsedTime(&ts[it], e0, e1), "el");
            }
            for (int i = 0; i < ITERS; i++) for (int j = i + 1; j < ITERS; j++) if (ts[j] < ts[i]) { float t = ts[i]; ts[i] = ts[j]; ts[j] = t; }
            double med = (double)ts[ITERS / 2];
            double gbps = (28.0 * (double)ne) / (med * 1.0e-3) / 1.0e9;
            cuEventDestroy(e0); cuEventDestroy(e1); free(ts);
            printf("GPU [%s] THROUGHPUT adam N=%d: med=%.4f ms  %.1f GB/s (28N B, memory-bound elementwise)\n", gpu, N, med, gbps);
        }
        printf("GPU [%s] adam N=%d%s: w[0]=%g ref %g, maxrel(w)=%.2e (tol m/v 1e-5, w 1e-4), %d bad -> %s\n",
               gpu, N, mutate ? " [MUTATED]" : "", w0, w0ref, maxrel, abad, abad ? "FAIL" : "PASS");
        cuMemFree(dw); cuMemFree(dg); cuMemFree(dm); cuMemFree(dv); cuMemFree(dbc1); cuMemFree(dbc2);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hw); free(hg); free(hm); free(hv); free(hw0); free(hm0); free(hv0); free(ptx);
        return abad ? 1 : 0;
    }

    /* scale mode: cuda_launch <ptx> gpu_scale_inplace <N> scale. a[i]=0.25*a[i] in place. */
    if (strcmp(op, "scale") == 0) {
        size_t ne = (size_t)N;
        float* ha = (float*)malloc(ne * sizeof(float));
        float* ha0 = (float*)malloc(ne * sizeof(float));
        if (!ha || !ha0) return 2;
        for (size_t i = 0; i < ne; i++) ha[i] = ha0[i] = (float)((int)(i % 11) - 5);
        CUdeviceptr da;
        CK(cuMemAlloc(&da, ne * sizeof(float)), "alloc a");
        CK(cuMemcpyHtoD(da, ha, ne * sizeof(float)), "H2D a");
        void* scargs[] = { &da, &N };
        CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, scargs, 0), "launch scale");
        CK(cuCtxSynchronize(), "sync scale");
        CK(cuMemcpyDtoH(ha, da, ne * sizeof(float)), "D2H a");
        int scbad = 0;
        for (size_t i = 0; i < ne; i++) {
            float ref = 0.25f * ha0[i];
            float e = ha[i] - ref; if (e < 0) e = -e;
            if (isnan(ha[i]) || e > 1.0e-6f) { if (scbad < 4) fprintf(stderr, "scale mismatch a[%zu]=%g ref %g\n", i, ha[i], ref); scbad++; }
        }
        printf("GPU [%s] scale N=%d: a[1]=%g ref %g, %d bad -> %s\n",
               gpu, N, ha[1], 0.25f * ha0[1], scbad, scbad ? "FAIL" : "PASS");
        cuMemFree(da);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(ha); free(ha0); free(ptx);
        return scbad ? 1 : 0;
    }

    /* layernorm_save mode: cuda_launch <ptx> gpu_layernorm_fwd_save <N> layernorm_save <rows> <cols>.
     * Forward layernorm with non-trivial gamma/beta, ALSO writing ist[row]=1/sqrt(var). Verify y
     * per-cell vs CPU affine layernorm (1e-3) AND ist[row]=1/sqrt(var_row) (1e-3, independent). */
    if (strcmp(op, "layernorm_save") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 8;
        int cols = (argc > 6) ? atoi(argv[6]) : 16;
        size_t ne = (size_t)rows * cols;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hy = (float*)malloc(ne * sizeof(float));
        float* hg = (float*)malloc((size_t)cols * sizeof(float));
        float* hb = (float*)malloc((size_t)cols * sizeof(float));
        float* hist = (float*)malloc((size_t)rows * sizeof(float));
        if (!hx || !hy || !hg || !hb || !hist) return 2;
        for (size_t i = 0; i < ne; i++) hx[i] = (float)((int)((i * 7 + 3) % 13) - 6);
        for (int c = 0; c < cols; c++) { hg[c] = 1.0f + 0.25f * (float)(c % 4); hb[c] = 0.5f * (float)((c % 3) - 1); }
        CUdeviceptr dx, dy, dg, db, dist;
        CK(cuMemAlloc(&dx, ne * sizeof(float)), "alloc x");
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "alloc y");
        CK(cuMemAlloc(&dg, (size_t)cols * sizeof(float)), "alloc g");
        CK(cuMemAlloc(&db, (size_t)cols * sizeof(float)), "alloc b");
        CK(cuMemAlloc(&dist, (size_t)rows * sizeof(float)), "alloc ist");
        CK(cuMemcpyHtoD(dx, hx, ne * sizeof(float)), "H2D x");
        CK(cuMemcpyHtoD(dg, hg, (size_t)cols * sizeof(float)), "H2D g");
        CK(cuMemcpyHtoD(db, hb, (size_t)cols * sizeof(float)), "H2D b");
        void* largs[] = { &dx, &dy, &dg, &db, &dist, &cols };
        int tb_ls = getenv("CL_BLOCK") ? atoi(getenv("CL_BLOCK")) : 1;  /* 256 for blockred kernel */
        CK(cuLaunchKernel(fn, rows, 1, 1, tb_ls, 1, 1, 0, 0, largs, 0), "launch ln_save");
        CK(cuCtxSynchronize(), "sync ln_save");
        CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "D2H y");
        CK(cuMemcpyDtoH(hist, dist, (size_t)rows * sizeof(float)), "D2H ist");
        int lbad = 0; float i0 = 0.0f, i0ref = 0.0f;
        for (int r = 0; r < rows; r++) {
            float mean = 0.0f; for (int c = 0; c < cols; c++) mean += hx[r * cols + c]; mean /= (float)cols;
            float var = 0.0f; for (int c = 0; c < cols; c++) { float d = hx[r * cols + c] - mean; var += d * d; } var /= (float)cols;
            float invref = 1.0f / sqrtf(var);
            if (r == 0) { i0 = hist[r]; i0ref = invref; }
            float ei = hist[r] - invref; if (ei < 0) ei = -ei;
            if (isnan(hist[r]) || ei > 1.0e-3f) { if (lbad < 4) fprintf(stderr, "ln_save ist[%d]=%g ref %g\n", r, hist[r], invref); lbad++; }
            for (int c = 0; c < cols; c++) {
                float ref = hg[c] * ((hx[r * cols + c] - mean) * invref) + hb[c];
                float got = hy[r * cols + c];
                float e = got - ref; if (e < 0) e = -e;
                if (isnan(got) || e > 1.0e-3f) { if (lbad < 4) fprintf(stderr, "ln_save y[%d,%d]=%g ref %g\n", r, c, got, ref); lbad++; }
            }
        }
        printf("GPU [%s] layernorm_save %dx%d: ist[0]=%g ref %g, %d bad -> %s\n",
               gpu, rows, cols, i0, i0ref, lbad, lbad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy); cuMemFree(dg); cuMemFree(db); cuMemFree(dist);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy); free(hg); free(hb); free(hist); free(ptx);
        return lbad ? 1 : 0;
    }

    /* layernorm_bwd_dx mode: cuda_launch <ptx> gpu_layernorm_backward_dx <N> layernorm_bwd_dx <rows> <cols>.
     * dx = layernorm input gradient. INDEPENDENT ref = central finite-difference of the layernorm
     * FORWARD: dx_fd[s,c]=(L(x+h e_sc)-L(x-h e_sc))/2h, L=sum_k dy*y, h=1e-3, tol 2e-2; plus the
     * conservation law sum_c dx[s,:] ~ 0 (tol 1e-3). gamma non-trivial; ist computed on CPU. */
    if (strcmp(op, "layernorm_bwd_dx") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 8;
        int cols = (argc > 6) ? atoi(argv[6]) : 16;
        size_t ne = (size_t)rows * cols;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hdy = (float*)malloc(ne * sizeof(float));
        float* hg = (float*)malloc((size_t)cols * sizeof(float));
        float* hist = (float*)malloc((size_t)rows * sizeof(float));
        float* hdx = (float*)malloc(ne * sizeof(float));
        if (!hx || !hdy || !hg || !hist || !hdx) return 2;
        for (size_t i = 0; i < ne; i++) { hx[i] = (float)((int)((i * 7 + 3) % 13) - 6); hdy[i] = (float)((int)((i * 5 + 1) % 7) - 3) * 0.5f; }
        for (int c = 0; c < cols; c++) hg[c] = 1.0f + 0.25f * (float)(c % 4);
        for (int r = 0; r < rows; r++) { float m = 0.0f; for (int c = 0; c < cols; c++) m += hx[r * cols + c]; m /= (float)cols; float v = 0.0f; for (int c = 0; c < cols; c++) { float d = hx[r * cols + c] - m; v += d * d; } v /= (float)cols; hist[r] = 1.0f / sqrtf(v); }
        CUdeviceptr dx_, ddy, dg, dist, ddx;
        CK(cuMemAlloc(&dx_, ne * sizeof(float)), "alloc x");
        CK(cuMemAlloc(&ddy, ne * sizeof(float)), "alloc dy");
        CK(cuMemAlloc(&dg, (size_t)cols * sizeof(float)), "alloc g");
        CK(cuMemAlloc(&dist, (size_t)rows * sizeof(float)), "alloc ist");
        CK(cuMemAlloc(&ddx, ne * sizeof(float)), "alloc dx");
        CK(cuMemcpyHtoD(dx_, hx, ne * sizeof(float)), "H2D x");
        CK(cuMemcpyHtoD(ddy, hdy, ne * sizeof(float)), "H2D dy");
        CK(cuMemcpyHtoD(dg, hg, (size_t)cols * sizeof(float)), "H2D g");
        CK(cuMemcpyHtoD(dist, hist, (size_t)rows * sizeof(float)), "H2D ist");
        void* ax[] = { &dx_, &ddy, &dg, &dist, &ddx, &cols };
        int tb_lb = getenv("CL_BLOCK") ? atoi(getenv("CL_BLOCK")) : 1;  /* 256 for blockred kernel */
        CK(cuLaunchKernel(fn, rows, 1, 1, tb_lb, 1, 1, 0, 0, ax, 0), "launch ln_bwd_dx");
        CK(cuCtxSynchronize(), "sync ln_bwd_dx");
        CK(cuMemcpyDtoH(hdx, ddx, ne * sizeof(float)), "D2H dx");
        int bad = 0; float maxfd = 0.0f, maxrs = 0.0f, d0 = 0.0f, r0 = 0.0f, h = 1.0e-3f;
        for (int s = 0; s < rows; s++) {
            float rs = 0.0f;
            for (int c = 0; c < cols; c++) {
                float fd = (ln_rowL(&hx[s * cols], &hdy[s * cols], hg, cols, c, h) - ln_rowL(&hx[s * cols], &hdy[s * cols], hg, cols, c, -h)) / (2.0f * h);
                float got = hdx[s * cols + c];
                if (s == 0 && c == 0) { d0 = got; r0 = fd; }
                rs += got;
                float e = got - fd; if (e < 0) e = -e; if (e > maxfd) maxfd = e;
                if (isnan(got) || e > 2.0e-2f) { if (bad < 4) fprintf(stderr, "ln_bwd_dx mismatch dx[%d,%d]=%g fd %g\n", s, c, got, fd); bad++; }
            }
            float ers = rs < 0 ? -rs : rs; if (ers > maxrs) maxrs = ers;
            if (isnan(rs) || ers > 1.0e-3f) { if (bad < 4) fprintf(stderr, "ln_bwd_dx row %d sum %g (want 0)\n", s, rs); bad++; }
        }
        printf("GPU [%s] layernorm_bwd_dx %dx%d: dx[0,0]=%g fd %g, max|dx-fd|=%g, max|row_sum|=%g, %d bad -> %s\n",
               gpu, rows, cols, d0, r0, maxfd, maxrs, bad, bad ? "FAIL" : "PASS");
        cuMemFree(dx_); cuMemFree(ddy); cuMemFree(dg); cuMemFree(dist); cuMemFree(ddx);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hdy); free(hg); free(hist); free(hdx); free(ptx);
        return bad ? 1 : 0;
    }

    /* layernorm_bwd_dgb mode: cuda_launch <ptx> gpu_layernorm_backward_dgb <N> layernorm_bwd_dgb <rows> <cols>.
     * dgamma=dgb[c], dbeta=dgb[cols+c]. Verify per-cell vs a CPU reduction dgamma[c]=sum_s
     * dy[s,c]*xhat[s,c], dbeta[c]=sum_s dy[s,c] (xhat=(x-mean_s)*ist[s]), tol 1e-3. */
    if (strcmp(op, "layernorm_bwd_dgb") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 8;
        int cols = (argc > 6) ? atoi(argv[6]) : 16;
        size_t ne = (size_t)rows * cols;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hdy = (float*)malloc(ne * sizeof(float));
        float* hist = (float*)malloc((size_t)rows * sizeof(float));
        float* hdgb = (float*)malloc((size_t)(2 * cols) * sizeof(float));
        if (!hx || !hdy || !hist || !hdgb) return 2;
        for (size_t i = 0; i < ne; i++) { hx[i] = (float)((int)((i * 7 + 3) % 13) - 6); hdy[i] = (float)((int)((i * 5 + 1) % 7) - 3) * 0.5f; }
        for (int r = 0; r < rows; r++) { float m = 0.0f; for (int c = 0; c < cols; c++) m += hx[r * cols + c]; m /= (float)cols; float v = 0.0f; for (int c = 0; c < cols; c++) { float d = hx[r * cols + c] - m; v += d * d; } v /= (float)cols; hist[r] = 1.0f / sqrtf(v); }
        CUdeviceptr dx_, ddy, dist, ddgb;
        CK(cuMemAlloc(&dx_, ne * sizeof(float)), "alloc x");
        CK(cuMemAlloc(&ddy, ne * sizeof(float)), "alloc dy");
        CK(cuMemAlloc(&dist, (size_t)rows * sizeof(float)), "alloc ist");
        CK(cuMemAlloc(&ddgb, (size_t)(2 * cols) * sizeof(float)), "alloc dgb");
        CK(cuMemcpyHtoD(dx_, hx, ne * sizeof(float)), "H2D x");
        CK(cuMemcpyHtoD(ddy, hdy, ne * sizeof(float)), "H2D dy");
        CK(cuMemcpyHtoD(dist, hist, (size_t)rows * sizeof(float)), "H2D ist");
        void* ag[] = { &dx_, &ddy, &dist, &ddgb, &rows, &cols };
        CK(cuLaunchKernel(fn, cols, 1, 1, 1, 1, 1, 0, 0, ag, 0), "launch ln_bwd_dgb");
        CK(cuCtxSynchronize(), "sync ln_bwd_dgb");
        CK(cuMemcpyDtoH(hdgb, ddgb, (size_t)(2 * cols) * sizeof(float)), "D2H dgb");
        int bad = 0; float maxe = 0.0f, g0 = 0.0f, g0r = 0.0f;
        for (int c = 0; c < cols; c++) {
            float dgr = 0.0f, dbr = 0.0f;
            for (int s = 0; s < rows; s++) {
                float m = 0.0f; for (int k = 0; k < cols; k++) m += hx[s * cols + k]; m /= (float)cols;
                float xh = (hx[s * cols + c] - m) * hist[s];
                dgr += hdy[s * cols + c] * xh;
                dbr += hdy[s * cols + c];
            }
            float gg = hdgb[c], gb = hdgb[cols + c];
            if (c == 0) { g0 = gg; g0r = dgr; }
            float eg = gg - dgr; if (eg < 0) eg = -eg; if (eg > maxe) maxe = eg;
            float eb = gb - dbr; if (eb < 0) eb = -eb; if (eb > maxe) maxe = eb;
            if (isnan(gg) || isnan(gb) || eg > 1.0e-3f || eb > 1.0e-3f) { if (bad < 4) fprintf(stderr, "ln_bwd_dgb c=%d dgamma=%g(ref %g) dbeta=%g(ref %g)\n", c, gg, dgr, gb, dbr); bad++; }
        }
        printf("GPU [%s] layernorm_bwd_dgb %dx%d: dgamma[0]=%g ref %g, max|err|=%g, %d bad -> %s\n",
               gpu, rows, cols, g0, g0r, maxe, bad, bad ? "FAIL" : "PASS");
        cuMemFree(dx_); cuMemFree(ddy); cuMemFree(dist); cuMemFree(ddgb);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hdy); free(hist); free(hdgb); free(ptx);
        return bad ? 1 : 0;
    }

    /* affine probe: cuda_launch <ptx> gpu_affine <N> affine. y[i]=0.5*x[i]+0.25,
     * validating the f32-LITERAL PTX emitter (0.5=0f3F000000, 0.25=0f3E800000). */
    if (strcmp(op, "affine") == 0) {
        size_t ne = (size_t)N;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hy = (float*)malloc(ne * sizeof(float));
        if (!hx || !hy) return 2;
        for (size_t i = 0; i < ne; i++) { hx[i] = (float)((int)((i * 7 + 3) % 13) - 6); hy[i] = -7.0f; }
        CUdeviceptr dx, dy;
        CK(cuMemAlloc(&dx, ne * sizeof(float)), "cuMemAlloc x");
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "cuMemAlloc y");
        CK(cuMemcpyHtoD(dx, hx, ne * sizeof(float)), "cuMemcpyHtoD x");
        CK(cuMemcpyHtoD(dy, hy, ne * sizeof(float)), "cuMemcpyHtoD y");
        void* aargs[] = { &dx, &dy, &N };
        CK(cuLaunchKernel(fn, N, 1, 1, 1, 1, 1, 0, 0, aargs, 0), "cuLaunchKernel affine");
        CK(cuCtxSynchronize(), "cuCtxSynchronize");
        CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "cuMemcpyDtoH y");
        int abad = 0;
        for (size_t i = 0; i < ne; i++) {
            float ref = 0.5f * hx[i] + 0.25f;
            float got = hy[i];
            float d = got - ref; if (d < 0) d = -d;
            if (isnan(got) || d > 1.0e-4f) { if (abad < 4) fprintf(stderr, "affine mismatch y[%zu]=%g ref %g\n", i, got, ref); abad++; }
        }
        printf("GPU [%s] affine N=%d (y=0.5x+0.25, f32 literals): y[0]=%g ref %g, %d bad -> %s\n",
               gpu, N, hy[0], 0.5f * hx[0] + 0.25f, abad, abad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy);
        return abad ? 1 : 0;
    }

    /* ===== P5 GPT-2 gap-kernel unit gates (HELIX_GPT2_DEMO_EXECUTION_PLAN.md P5 gate 1) ===== */

    /* softmax_causal mode: cuda_launch <ptx> gpu_softmax_causal <Nignored> softmax_causal <S> <S> [mutate].
     * The NEW causal row-softmax kernel gpu_softmax_causal(x,y,rows,cols): grid=rows(=S) block=1; for
     * query row i it stable-softmaxes over keys j<=i ONLY and writes y[i,j]=0 for j>i (no -inf literal).
     * Random [S,S] scores. CPU reference = the same causal softmax (max over [0..=i], expf, normalize;
     * exact 0 for j>i). Two independent checks: (1) cell-by-cell vs the CPU ref (tol 1e-3, maxrel
     * reported); (2) each row i sums to ~1 over j<=i AND every y[i,j]==0 for j>i (exact). The "mutate"
     * arg perturbs one valid output cell pre-compare -> comparator negative control (must FAIL).
     * cols==rows==S here (square scores). S<=4096 (static scratch). */
    if (strcmp(op, "softmax_causal") == 0) {
        int S = (argc > 5) ? atoi(argv[5]) : 64;
        int cols = (argc > 6) ? atoi(argv[6]) : S;
        int mutate = (argc > 7 && strcmp(argv[7], "mutate") == 0);
        if (cols != S) { fprintf(stderr, "softmax_causal: this gate uses square [S,S] scores (cols must==S); got S=%d cols=%d\n", S, cols); return 2; }
        if (S > 4096) { fprintf(stderr, "softmax_causal: S<=4096; got %d\n", S); return 2; }
        size_t ne = (size_t)S * cols;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hy = (float*)malloc(ne * sizeof(float));
        if (!hx || !hy) return 2;
        /* pseudo-random scores in ~[-4,4] (deterministic LCG so the run is reproducible). */
        unsigned int seed = 0x9e3779b9u;
        for (size_t i = 0; i < ne; i++) {
            seed = seed * 1664525u + 1013904223u;
            hx[i] = ((float)(seed >> 8) / (float)0xFFFFFFu) * 8.0f - 4.0f;
            hy[i] = -7.0f;   /* sentinel: an unwritten cell shows up */
        }
        CUdeviceptr dx, dy;
        CK(cuMemAlloc(&dx, ne * sizeof(float)), "alloc x");
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "alloc y");
        CK(cuMemcpyHtoD(dx, hx, ne * sizeof(float)), "H2D x");
        CK(cuMemcpyHtoD(dy, hy, ne * sizeof(float)), "H2D y(sentinel)");
        void* sargs[] = { &dx, &dy, &S, &cols };
        CK(cuLaunchKernel(fn, S, 1, 1, 1, 1, 1, 0, 0, sargs, 0), "launch gpu_softmax_causal");
        CK(cuCtxSynchronize(), "sync gpu_softmax_causal");
        CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "D2H y");
        if (mutate) hy[(size_t)(S - 1) * cols] += 1.0f;  /* perturb a VALID cell (row S-1, col 0): must trip the compare */
        int sbad = 0; float maxabs = 0.0f, maxrel = 0.0f, maxsumerr = 0.0f, maxzero = 0.0f;
        for (int r = 0; r < S; r++) {
            int nvalid = r + 1;
            float mx = hx[(size_t)r * cols];
            for (int j = 1; j < nvalid; j++) if (hx[(size_t)r * cols + j] > mx) mx = hx[(size_t)r * cols + j];
            float sm = 0.0f;
            for (int j = 0; j < nvalid; j++) sm += expf(hx[(size_t)r * cols + j] - mx);
            float rowsum = 0.0f;
            for (int c = 0; c < cols; c++) {
                float got = hy[(size_t)r * cols + c];
                float ref = (c < nvalid) ? expf(hx[(size_t)r * cols + c] - mx) / sm : 0.0f;
                if (c < nvalid) {
                    rowsum += got;
                    float e = got - ref; if (e < 0) e = -e; if (e > maxabs) maxabs = e;
                    float ar = ref < 0 ? -ref : ref; float rel = e / (ar > 1e-6f ? ar : 1e-6f);
                    if (rel > maxrel) maxrel = rel;
                    if (isnan(got) || (e > 1.0e-3f && rel > 1.0e-3f)) { if (sbad < 6) fprintf(stderr, "softmax_causal mismatch y[%d,%d]=%g ref %g (abs %.2e rel %.2e)\n", r, c, got, ref, e, rel); sbad++; }
                } else {
                    /* masked region: must be EXACTLY 0 (the causal write). */
                    float az = got < 0 ? -got : got; if (az > maxzero) maxzero = az;
                    if (isnan(got) || got != 0.0f) { if (sbad < 6) fprintf(stderr, "softmax_causal masked cell y[%d,%d]=%g (want exact 0)\n", r, c, got); sbad++; }
                }
            }
            float ds = rowsum - 1.0f; if (ds < 0) ds = -ds; if (ds > maxsumerr) maxsumerr = ds;
            if (isnan(rowsum) || ds > 1.0e-3f) { if (sbad < 6) fprintf(stderr, "softmax_causal row %d sum-over-j<=i %g (want 1)\n", r, rowsum); sbad++; }
        }
        printf("GPU [%s] softmax_causal S=%d (causal row-softmax, grid=S block=1)%s: maxabs=%.2e maxrel=%.2e (tol=1e-3) max|rowsum-1|=%.2e max|masked|=%.2e, %d bad -> %s\n",
               gpu, S, mutate ? " [MUTATED]" : "", maxabs, maxrel, maxsumerr, maxzero, sbad, sbad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy); free(ptx);
        return sbad ? 1 : 0;
    }

    /* layernorm_eps mode: cuda_launch <ptx> gpu_layernorm_fwd_eps <Nignored> layernorm_eps <rows> <cols> [mutate|epscheck].
     * The NEW affine LayerNorm-WITH-eps kernel gpu_layernorm_fwd_eps(x,y,gamma,beta,cols): grid=rows
     * block=1; y[r,c]=gamma[c]*(x[r,c]-mean)*rsqrt(var+1e-5)+beta[c], biased/population variance.
     * Random [rows,cols] (cols=768 for GPT-2). CPU reference = affine LN with eps=1e-5 + biased var
     * (divide by cols); cell-by-cell tol 1e-3 (maxrel reported).
     *  - "mutate": perturb one output cell pre-compare -> comparator negative control (must FAIL).
     *  - "epscheck": EPS-IS-APPLIED negative control. Feed a near-zero-variance row (all entries ~equal)
     *    and compare the GPU (eps) output against an eps-STRIPPED CPU reference (1/sqrt(var), no eps).
     *    With var~0 the two MUST DIVERGE (the eps-less ref blows up / differs hugely); a kernel that
     *    silently dropped eps would instead MATCH the stripped ref. So here a LARGE divergence == PASS
     *    (eps proven load-bearing); near-agreement == FAIL. */
    if (strcmp(op, "layernorm_eps") == 0) {
        int rows = (argc > 5) ? atoi(argv[5]) : 8;
        int cols = (argc > 6) ? atoi(argv[6]) : 768;
        int mutate  = (argc > 7 && strcmp(argv[7], "mutate") == 0);
        int epschk  = (argc > 7 && strcmp(argv[7], "epscheck") == 0);
        const float EPS = 1.0e-5f;
        size_t ne = (size_t)rows * cols;
        float* hx = (float*)malloc(ne * sizeof(float));
        float* hy = (float*)malloc(ne * sizeof(float));
        float* hg = (float*)malloc((size_t)cols * sizeof(float));
        float* hb = (float*)malloc((size_t)cols * sizeof(float));
        if (!hx || !hy || !hg || !hb) return 2;
        unsigned int seed = 0x12345677u;
        for (size_t i = 0; i < ne; i++) {
            seed = seed * 1664525u + 1013904223u;
            hx[i] = ((float)(seed >> 8) / (float)0xFFFFFFu) * 6.0f - 3.0f;
            hy[i] = -7.0f;
        }
        for (int c = 0; c < cols; c++) { hg[c] = 1.0f + 0.25f * (float)(c % 4); hb[c] = 0.5f * (float)((c % 3) - 1); }
        /* epscheck: make EVERY row near-zero-variance (a tiny nonzero spread so the eps-less CPU ref is
         * finite-but-HUGE, not a degenerate 0*inf=NaN). var ~ 1e-8 << eps=1e-5, so 1/sqrt(var) ~ 1e4
         * (eps-less) vs 1/sqrt(var+eps) ~ 316 (the GPU): the normalized outputs then differ by a large,
         * finite, measurable amount -> eps proven load-bearing. A kernel that silently dropped eps would
         * instead match the eps-less ref. Spread = +-1e-4 around 2.5 (alternating) gives var ~ 1e-8. */
        if (epschk) {
            for (size_t i = 0; i < ne; i++) hx[i] = 2.5f + ((i & 1) ? 1.0e-4f : -1.0e-4f);
        }
        CUdeviceptr dx, dy, dg, db;
        CK(cuMemAlloc(&dx, ne * sizeof(float)), "alloc x");
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "alloc y");
        CK(cuMemAlloc(&dg, (size_t)cols * sizeof(float)), "alloc g");
        CK(cuMemAlloc(&db, (size_t)cols * sizeof(float)), "alloc b");
        CK(cuMemcpyHtoD(dx, hx, ne * sizeof(float)), "H2D x");
        CK(cuMemcpyHtoD(dy, hy, ne * sizeof(float)), "H2D y(sentinel)");
        CK(cuMemcpyHtoD(dg, hg, (size_t)cols * sizeof(float)), "H2D g");
        CK(cuMemcpyHtoD(db, hb, (size_t)cols * sizeof(float)), "H2D b");
        void* largs[] = { &dx, &dy, &dg, &db, &cols };   /* 5 params: NO ist save (inference kernel) */
        CK(cuLaunchKernel(fn, rows, 1, 1, 1, 1, 1, 0, 0, largs, 0), "launch gpu_layernorm_fwd_eps");
        CK(cuCtxSynchronize(), "sync gpu_layernorm_fwd_eps");
        CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "D2H y");
        if (mutate) hy[0] += 1.0f;   /* comparator negative control */
        if (epschk) {
            /* compare GPU(eps) row 0 vs an EPS-STRIPPED CPU ref on the near-zero-variance row; they MUST diverge. */
            float mean = 0.0f; for (int c = 0; c < cols; c++) mean += hx[c]; mean /= (float)cols;
            float var = 0.0f; for (int c = 0; c < cols; c++) { float d = hx[c] - mean; var += d * d; } var /= (float)cols;
            float inv_noeps = 1.0f / sqrtf(var);          /* eps-LESS: var~0 -> huge/inf */
            float inv_eps   = 1.0f / sqrtf(var + EPS);    /* what the GPU kernel uses */
            float maxdiv = 0.0f; int finite_ok = 1;
            for (int c = 0; c < cols; c++) {
                float ref_noeps = hg[c] * ((hx[c] - mean) * inv_noeps) + hb[c];
                float got = hy[c];
                if (isnan(got) || isinf(got)) finite_ok = 0;   /* GPU(eps) must stay finite */
                float e = got - ref_noeps; if (e < 0) e = -e;
                if (!isnan(e) && !isinf(e) && e > maxdiv) maxdiv = e;   /* finite divergence only */
            }
            /* PASS == the GPU(eps) output is finite AND diverges hugely from the eps-less ref, on a
             * genuinely near-zero-variance row (var << eps) where eps is the load-bearing term. */
            int diverged = finite_ok && (maxdiv > 1.0f) && (var < 1.0e-4f);
            printf("GPU [%s] layernorm_eps EPS-CONTROL rows=%d cols=%d: near-zero var=%.3e, GPU-eps inv=%.4f vs eps-less inv=%.4f, max|GPU-eps - CPU-noeps|=%.3e, GPU finite=%d -> eps-applied=%s -> %s\n",
                   gpu, rows, cols, var, inv_eps, inv_noeps, maxdiv, finite_ok, diverged ? "YES" : "NO", diverged ? "PASS" : "FAIL");
            cuMemFree(dx); cuMemFree(dy); cuMemFree(dg); cuMemFree(db);
            cuModuleUnload(mod); cuCtxDestroy(ctx);
            free(hx); free(hy); free(hg); free(hb); free(ptx);
            return diverged ? 0 : 1;
        }
        /* normal correctness gate: CPU affine LN with eps + biased variance, cell-by-cell. */
        int lbad = 0; float maxabs = 0.0f, maxrel = 0.0f;
        for (int r = 0; r < rows; r++) {
            float mean = 0.0f; for (int c = 0; c < cols; c++) mean += hx[(size_t)r * cols + c]; mean /= (float)cols;
            float var = 0.0f; for (int c = 0; c < cols; c++) { float d = hx[(size_t)r * cols + c] - mean; var += d * d; } var /= (float)cols;
            float inv = 1.0f / sqrtf(var + EPS);
            for (int c = 0; c < cols; c++) {
                float ref = hg[c] * ((hx[(size_t)r * cols + c] - mean) * inv) + hb[c];
                float got = hy[(size_t)r * cols + c];
                float e = got - ref; if (e < 0) e = -e; if (e > maxabs) maxabs = e;
                float ar = ref < 0 ? -ref : ref; float rel = e / (ar > 1e-6f ? ar : 1e-6f);
                if (rel > maxrel) maxrel = rel;
                if (isnan(got) || (e > 1.0e-3f && rel > 1.0e-3f)) { if (lbad < 6) fprintf(stderr, "layernorm_eps mismatch y[%d,%d]=%g ref %g (abs %.2e rel %.2e)\n", r, c, got, ref, e, rel); lbad++; }
            }
        }
        printf("GPU [%s] layernorm_eps rows=%d cols=%d (affine + eps=1e-5, biased var, grid=rows block=1)%s: maxabs=%.2e maxrel=%.2e (tol=1e-3), %d bad -> %s\n",
               gpu, rows, cols, mutate ? " [MUTATED]" : "", maxabs, maxrel, lbad, lbad ? "FAIL" : "PASS");
        cuMemFree(dx); cuMemFree(dy); cuMemFree(dg); cuMemFree(db);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx); free(hy); free(hg); free(hb); free(ptx);
        return lbad ? 1 : 0;
    }

    /* add_bias mode: cuda_launch <ptx> gpu_add_bias_rowbcast <n> add_bias <n> <cols> [mutate].
     * The NEW row-broadcast bias-add kernel gpu_add_bias_rowbcast(y,bias,n,cols): one thread/element
     * (vector_add-style launch grid=ceil(n/256) block=256) with an i<n guard; y[i]+=bias[i mod cols].
     * y starts as x0 (random), bias random length cols (GPT-2: cols in {2304,768,3072}); n is typically
     * NOT a multiple of cols so the i<n guard + the wrap are both exercised. CPU reference: assert
     * y[i]==x0[i]+bias[i mod cols] CELL-EXACT (fp32 add of the same two summands -> bit-identical, ==
     * compare, NOT a tolerance). "mutate" perturbs one cell pre-compare -> comparator neg-control. */
    if (strcmp(op, "add_bias") == 0) {
        int n    = (argc > 5) ? atoi(argv[5]) : (768 * 5 + 17);
        int cols = (argc > 6) ? atoi(argv[6]) : 768;
        int mutate = (argc > 7 && strcmp(argv[7], "mutate") == 0);
        if (n <= 0 || cols <= 0) { fprintf(stderr, "add_bias: n>0 and cols>0 required\n"); return 2; }
        size_t ne = (size_t)n;
        float* hx0  = (float*)malloc(ne * sizeof(float));     /* original y (=x0) */
        float* hy   = (float*)malloc(ne * sizeof(float));     /* device result */
        float* hbias= (float*)malloc((size_t)cols * sizeof(float));
        if (!hx0 || !hy || !hbias) return 2;
        unsigned int seed = 0xcafef00du;
        for (size_t i = 0; i < ne; i++) {
            seed = seed * 1664525u + 1013904223u;
            hx0[i] = ((float)(seed >> 8) / (float)0xFFFFFFu) * 20.0f - 10.0f;
        }
        for (int c = 0; c < cols; c++) {
            seed = seed * 1664525u + 1013904223u;
            hbias[c] = ((float)(seed >> 8) / (float)0xFFFFFFu) * 4.0f - 2.0f;
        }
        CUdeviceptr dy, dbias;
        CK(cuMemAlloc(&dy, ne * sizeof(float)), "alloc y");
        CK(cuMemAlloc(&dbias, (size_t)cols * sizeof(float)), "alloc bias");
        CK(cuMemcpyHtoD(dy, hx0, ne * sizeof(float)), "H2D y(=x0)");
        CK(cuMemcpyHtoD(dbias, hbias, (size_t)cols * sizeof(float)), "H2D bias");
        int ni = n, ci = cols;
        void* aargs[] = { &dy, &dbias, &ni, &ci };
        int tpb = 256; int bpg = (n + tpb - 1) / tpb;
        CK(cuLaunchKernel(fn, bpg, 1, 1, tpb, 1, 1, 0, 0, aargs, 0), "launch gpu_add_bias_rowbcast");
        CK(cuCtxSynchronize(), "sync gpu_add_bias_rowbcast");
        CK(cuMemcpyDtoH(hy, dy, ne * sizeof(float)), "D2H y");
        if (mutate) hy[n / 2] += 1.0f;   /* perturbed-cell negative control: must trip the == compare */
        int abad = 0; float maxabs = 0.0f;
        for (int i = 0; i < n; i++) {
            int r = i - (i / cols) * cols;            /* i mod cols, the kernel's own formula */
            float ref = hx0[i] + hbias[r];            /* same two summands -> bit-identical add */
            float got = hy[i];
            float e = got - ref; if (e < 0) e = -e; if (e > maxabs) maxabs = e;
            if (isnan(got) || got != ref) { if (abad < 6) fprintf(stderr, "add_bias mismatch y[%d]=%g ref %g (i mod cols=%d)\n", i, got, ref, r); abad++; }
        }
        printf("GPU [%s] add_bias n=%d cols=%d (y[i]+=bias[i mod cols], 1 thread/elem, i<n guard)%s: maxabs=%.2e (cell-exact ==), %d bad -> %s\n",
               gpu, n, cols, mutate ? " [MUTATED]" : "", maxabs, abad, abad ? "FAIL" : "PASS");
        cuMemFree(dy); cuMemFree(dbias);
        cuModuleUnload(mod); cuCtxDestroy(ctx);
        free(hx0); free(hy); free(hbias); free(ptx);
        return abad ? 1 : 0;
    }

    size_t bytes = (size_t)N * sizeof(float);
    float* ha = (float*)malloc(bytes);
    float* hb = (float*)malloc(bytes);
    float* hc = (float*)malloc(bytes);
    if (!ha || !hb || !hc) { return 2; }
    for (int i = 0; i < N; i++) {
        ha[i] = is_exp ? (float)((i % 8) - 4) : is_relu ? (float)(i - 128) : (float)i;
        hb[i] = (float)(2 * i); hc[i] = -1.0f;
    }

    CUdeviceptr da, db, dc;
    CK(cuMemAlloc(&da, bytes), "cuMemAlloc a");
    CK(cuMemAlloc(&db, bytes), "cuMemAlloc b");
    CK(cuMemAlloc(&dc, bytes), "cuMemAlloc c");
    CK(cuMemcpyHtoD(da, ha, bytes), "cuMemcpyHtoD a");
    CK(cuMemcpyHtoD(db, hb, bytes), "cuMemcpyHtoD b");

    void* args[] = { &da, &db, &dc, &N };
    int tpb = 256;
    int bpg = (N + tpb - 1) / tpb;
    CK(cuLaunchKernel(fn, bpg, 1, 1, tpb, 1, 1, 0, 0, args, 0), "cuLaunchKernel");
    CK(cuCtxSynchronize(), "cuCtxSynchronize");
    CK(cuMemcpyDtoH(hc, dc, bytes), "cuMemcpyDtoH c");

    int is_mul = (strcmp(op, "mul") == 0);
    int is_sub = (strcmp(op, "sub") == 0);
    int is_rev = (strcmp(op, "reverse") == 0);  /* c[i]=a[N-1-i]: exercises an i32 scalar param read in the index */
    int bad = 0;
    for (int i = 0; i < N; i++) {
        float want = is_exp ? expf(ha[i]) : is_relu ? (ha[i] > 0.0f ? ha[i] : 0.0f)
                   : is_rev ? ha[N - 1 - i] : is_mul ? ha[i] * hb[i] : is_sub ? ha[i] - hb[i] : ha[i] + hb[i];
        int bad_i;
        if (is_exp) { float d = hc[i] - want; if (d < 0) d = -d; float aw = want < 0 ? -want : want; bad_i = (isnan(hc[i]) || d > 1.0e-3f * (aw + 1.0e-6f)); }
        else bad_i = (hc[i] != want);
        if (bad_i) { if (bad < 4) fprintf(stderr, "mismatch c[%d]=%g want %g\n", i, hc[i], want); bad++; }
    }
    float want7 = is_exp ? expf(ha[7]) : is_relu ? (ha[7] > 0.0f ? ha[7] : 0.0f)
                : is_rev ? ha[N - 1 - 7] : is_mul ? ha[7] * hb[7] : is_sub ? ha[7] - hb[7] : ha[7] + hb[7];
    printf("GPU [%s] kernel '%s' op=%s over %d elems: c[7]=%g (want %g) -> %s\n",
           gpu, kname, op, N, hc[7], want7, bad ? "FAIL" : "PASS");

    cuMemFree(da); cuMemFree(db); cuMemFree(dc);
    cuModuleUnload(mod); cuCtxDestroy(ctx);
    free(ha); free(hb); free(hc); free(ptx);
    return bad ? 1 : 0;
}
