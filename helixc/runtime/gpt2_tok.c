/* gpt2_tok.c -- GPT-2 byte-level BPE tokenizer (encode + decode), Category-B HOST TOOL.
 *
 * TRUST CLAIM / FENCE ROLE: this is an OFFLINE host tool, OUTSIDE the self-host
 * fixpoint, with ZERO arithmetic on the compute-trust path -- exactly like
 * helixc/runtime/cpu_host.c. It performs only string<->token-id bookkeeping for the
 * demo's offline pre/post-processing. The TRUST CLAIM of the demo is the exact
 * token-id sequence + the from-raw toolchain that executes it, NOT this host-side
 * string rendering (see docs/HELIX_GPT2_DEMO_RUNBOOK.md residual #1). It exists to
 * eliminate the Python interpreter dependency from the demo's PRODUCTION data path;
 * the independent numpy oracle (helix-llm/tools/gpt2_numpy_ref.py) STAYS Python on
 * purpose (it is the cross-check verifier and its independence is the whole point).
 *
 * It is a faithful re-authoring of the byte-level BPE inside gpt2_numpy_ref.py
 * (class BPE): the 256-entry byte<->unicode map, the GPT-2 pre-tokenization split,
 * the merges.txt rank-ordered merges, and the vocab.json id map. The pre-tokenizer
 * is hand-written (NO regex library): it reproduces the GPT-2 pattern
 *   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
 * as an ordered, first-alternative-wins matcher over a UTF-8 codepoint stream, with
 * Unicode \p{L}/\p{N}/\s classification from the baked range tables below (generated
 * once from Python's `regex` module so they are bit-exact; data, not code).
 *
 * KEY SIMPLIFICATION (provably equivalent): the byte<->unicode map is a bijection,
 * so the BPE on the b2u-mapped chars is structurally identical to BPE on the original
 * bytes. We therefore work entirely in BYTE space: a vocab token IS a byte string
 * (each \uXXXX key char decodes via u2b back to one byte), a merge rule is a
 * byte-string pair, encode emits ids for byte-string symbols, and decode just
 * concatenates the byte strings of the ids -- which is exactly the Python decode
 * `bytearray(u2b[c] for c in s)`. This avoids ever materializing the unicode strings.
 *
 * Build (host): gcc gpt2_tok.c -O2 -o gpt2_tok
 * Usage:
 *   gpt2_tok <vocab.json> <merges.txt> encode            # stdin text -> ids (space-sep) on stdout
 *   gpt2_tok <vocab.json> <merges.txt> encode-file <in>  # file text  -> ids (space-sep) on stdout
 *   gpt2_tok <vocab.json> <merges.txt> decode <id>...     # ids (args) -> text (raw bytes) on stdout
 *   gpt2_tok <vocab.json> <merges.txt> decode-file <ids>  # ids (file, space/nl-sep) -> text on stdout
 *
 * License: Apache 2.0.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>

/* ============================ Unicode range tables ============================ */
/* Generated once from Python `regex` (\p{L}, \p{N}, \s) so they are bit-exact with
 * the oracle's pretokenizer. Pure DATA (range pairs), not executable logic. */
#include "gpt2_unicode_ranges.inc"

static int in_ranges(unsigned int cp, const unsigned int (*tbl)[2], int n) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {                       /* tables are sorted, disjoint -> binary search */
        int mid = (lo + hi) >> 1;
        if (cp < tbl[mid][0]) hi = mid - 1;
        else if (cp > tbl[mid][1]) lo = mid + 1;
        else return 1;
    }
    return 0;
}
static int is_L(unsigned int cp) { return in_ranges(cp, UL, UL_N); }
static int is_N(unsigned int cp) { return in_ranges(cp, UN, UN_N); }
static int is_S(unsigned int cp) { return in_ranges(cp, US, US_N); }

/* ============================ small helpers ============================ */
static void* xmalloc(size_t n) { void* p = malloc(n ? n : 1); if (!p) { fprintf(stderr, "oom (%zu)\n", n); exit(2); } return p; }
static void* xrealloc(void* p, size_t n) { void* q = realloc(p, n ? n : 1); if (!q) { fprintf(stderr, "oom realloc (%zu)\n", n); exit(2); } return q; }

/* read an entire file into a malloc'd buffer (NUL-terminated); returns length (excl NUL). */
static char* slurp(const char* path, size_t* out_len) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "open '%s': %s\n", path, strerror(errno)); exit(2); }
    if (fseek(f, 0, SEEK_END) != 0) { fprintf(stderr, "seek '%s'\n", path); exit(2); }
    long sz = ftell(f); if (sz < 0) { fprintf(stderr, "ftell '%s'\n", path); exit(2); }
    fseek(f, 0, SEEK_SET);
    char* buf = (char*)xmalloc((size_t)sz + 1);
    size_t got = fread(buf, 1, (size_t)sz, f);
    fclose(f);
    buf[got] = 0;
    if (out_len) *out_len = got;
    return buf;
}
/* read all of stdin into a malloc'd buffer (NUL-terminated); returns length. */
static char* slurp_stdin(size_t* out_len) {
    size_t cap = 1 << 16, len = 0;
    char* buf = (char*)xmalloc(cap);
    size_t r;
    while ((r = fread(buf + len, 1, cap - len, stdin)) > 0) {
        len += r;
        if (len == cap) { cap <<= 1; buf = (char*)xrealloc(buf, cap); }
    }
    buf = (char*)xrealloc(buf, len + 1);
    buf[len] = 0;
    if (out_len) *out_len = len;
    return buf;
}

/* ============================ byte<->unicode map ============================ */
/* bytes_to_unicode(): bs = printable ASCII + Latin-1 punctuation ranges; the rest
 * map to 256+k. We need the INVERSE: codepoint -> byte (u2b). Build it once. */
static int g_b2u[256];        /* byte -> codepoint */
static int g_u2b[512];        /* codepoint -> byte (-1 if none); max codepoint is 323 */
/* LIB-EXPOSED (serve mode links gpt2_tok.c with GPT2_TOK_LIB; gpt2_infer.c calls these
 * four + the three decode helpers below). Standalone build keeps the same behavior. */
void build_byte_unicode(void) {
    int bs[256], used[256];
    memset(used, 0, sizeof(used));
    int nbs = 0;
    for (int c = '!'; c <= '~'; c++)            { bs[nbs++] = c; used[c] = 1; }
    for (int c = 0xA1; c <= 0xAC; c++)          { bs[nbs++] = c; used[c] = 1; }
    for (int c = 0xAE; c <= 0xFF; c++)          { bs[nbs++] = c; used[c] = 1; }
    int cs[256];
    for (int i = 0; i < nbs; i++) cs[i] = bs[i];   /* cs starts == bs for the direct range */
    int k = 0;
    for (int b = 0; b < 256; b++) {
        if (!used[b]) { bs[nbs] = b; cs[nbs] = 256 + k; nbs++; k++; }
    }
    for (int i = 0; i < 512; i++) g_u2b[i] = -1;
    for (int i = 0; i < nbs; i++) {            /* nbs == 256 now: full byte coverage */
        g_b2u[bs[i]] = cs[i];
        g_u2b[cs[i]] = bs[i];
    }
}

/* ============================ UTF-8 decode (for vocab keys + input text) ============================ */
/* decode one UTF-8 codepoint at s[i..]; advance *i; returns codepoint. On a malformed
 * lead byte, returns the raw byte and advances 1 (matches a permissive byte stream). */
static unsigned int utf8_next(const unsigned char* s, size_t n, size_t* i) {
    unsigned int b0 = s[*i];
    if (b0 < 0x80) { (*i)++; return b0; }
    if ((b0 & 0xE0) == 0xC0 && *i + 1 < n + 0) {
        unsigned int cp = ((b0 & 0x1F) << 6) | (s[*i + 1] & 0x3F);
        *i += 2; return cp;
    }
    if ((b0 & 0xF0) == 0xE0) {
        unsigned int cp = ((b0 & 0x0F) << 12) | ((s[*i + 1] & 0x3F) << 6) | (s[*i + 2] & 0x3F);
        *i += 3; return cp;
    }
    if ((b0 & 0xF8) == 0xF0) {
        unsigned int cp = ((b0 & 0x07) << 18) | ((s[*i + 1] & 0x3F) << 12)
                        | ((s[*i + 2] & 0x3F) << 6) | (s[*i + 3] & 0x3F);
        *i += 4; return cp;
    }
    (*i)++; return b0;
}

/* ============================ vocab.json (hand-rolled) ============================ */
/* The vocab maps a key string (the b2u-encoded token, with \uXXXX escapes for the
 * non-ASCII b2u chars) to an id. We decode each key to its BYTE string (via u2b) and
 * store byte-string -> id in an open-addressing hash table. We also keep id -> byte
 * string for decode. */
typedef struct { unsigned char* bytes; int len; int id; } VEntry;
static VEntry* g_vtab = NULL;       /* hash table (open addressing) */
static int g_vcap = 0;              /* power of two */
static int g_vmask = 0;
static unsigned char** g_id2bytes = NULL;  /* id -> bytes (into the same allocations) */
static int* g_id2len = NULL;
static int g_nvocab = 0;

static uint64_t fnv1a(const unsigned char* s, int n) {
    uint64_t h = 1469598103934665603ULL;
    for (int i = 0; i < n; i++) { h ^= s[i]; h *= 1099511628211ULL; }
    return h;
}
static void vtab_init(int approx) {
    g_vcap = 1; while (g_vcap < approx * 2) g_vcap <<= 1;
    g_vmask = g_vcap - 1;
    g_vtab = (VEntry*)xmalloc((size_t)g_vcap * sizeof(VEntry));
    for (int i = 0; i < g_vcap; i++) { g_vtab[i].bytes = NULL; g_vtab[i].len = 0; g_vtab[i].id = -1; }
}
static void vtab_put(unsigned char* bytes, int len, int id) {
    uint64_t h = fnv1a(bytes, len);
    int idx = (int)(h & g_vmask);
    while (g_vtab[idx].bytes) {
        if (g_vtab[idx].len == len && memcmp(g_vtab[idx].bytes, bytes, len) == 0) {
            g_vtab[idx].id = id; return;          /* dup key (shouldn't happen) -> overwrite */
        }
        idx = (idx + 1) & g_vmask;
    }
    g_vtab[idx].bytes = bytes; g_vtab[idx].len = len; g_vtab[idx].id = id;
}
static int vtab_get(const unsigned char* bytes, int len) {
    uint64_t h = fnv1a(bytes, len);
    int idx = (int)(h & g_vmask);
    while (g_vtab[idx].bytes) {
        if (g_vtab[idx].len == len && memcmp(g_vtab[idx].bytes, bytes, len) == 0)
            return g_vtab[idx].id;
        idx = (idx + 1) & g_vmask;
    }
    return -1;
}

/* parse a JSON string starting at s[*i]=='"'; decode escapes; map b2u-chars -> bytes;
 * write the decoded BYTE string into out (caller-owned, big enough); return byte len.
 * Advances *i past the closing quote. */
static int parse_json_string_to_bytes(const char* s, size_t n, size_t* i, unsigned char* out) {
    int olen = 0;
    (*i)++;                                       /* skip opening quote */
    while (*i < n && s[(size_t)*i] != '"') {
        unsigned char c = (unsigned char)s[(size_t)*i];
        unsigned int cp;
        if (c == '\\') {
            (*i)++;
            char e = s[(size_t)*i];
            switch (e) {
                case 'n': cp = '\n'; (*i)++; break;
                case 't': cp = '\t'; (*i)++; break;
                case 'r': cp = '\r'; (*i)++; break;
                case 'b': cp = '\b'; (*i)++; break;
                case 'f': cp = '\f'; (*i)++; break;
                case '/': cp = '/';  (*i)++; break;
                case '\\': cp = '\\'; (*i)++; break;
                case '"': cp = '"';  (*i)++; break;
                case 'u': {
                    (*i)++;
                    unsigned int v = 0;
                    for (int k = 0; k < 4; k++) {
                        char h = s[(size_t)*i]; (*i)++;
                        v <<= 4;
                        if (h >= '0' && h <= '9') v |= (unsigned)(h - '0');
                        else if (h >= 'a' && h <= 'f') v |= (unsigned)(h - 'a' + 10);
                        else if (h >= 'A' && h <= 'F') v |= (unsigned)(h - 'A' + 10);
                    }
                    cp = v;
                    break;
                }
                default: cp = (unsigned char)e; (*i)++; break;
            }
        } else if (c < 0x80) {
            cp = c; (*i)++;
        } else {
            /* raw UTF-8 (GPT-2 vocab uses \u escapes, but be permissive) */
            cp = utf8_next((const unsigned char*)s, n, i);
        }
        /* cp is a b2u codepoint -> map back to its single byte */
        int b = (cp < 512) ? g_u2b[cp] : -1;
        if (b < 0) {
            /* not a b2u char (shouldn't occur for GPT-2 vocab/merges); fall back to
             * the codepoint's own UTF-8 bytes so we never silently drop data. */
            if (cp < 0x80) out[olen++] = (unsigned char)cp;
            else if (cp < 0x800) { out[olen++] = (unsigned char)(0xC0 | (cp >> 6)); out[olen++] = (unsigned char)(0x80 | (cp & 0x3F)); }
            else { out[olen++] = (unsigned char)(0xE0 | (cp >> 12)); out[olen++] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F)); out[olen++] = (unsigned char)(0x80 | (cp & 0x3F)); }
        } else {
            out[olen++] = (unsigned char)b;
        }
    }
    if (*i < n) (*i)++;                            /* skip closing quote */
    return olen;
}

/* load vocab.json: { "<key>": <id>, ... }. */
void load_vocab(const char* path) {
    size_t n; char* s = slurp(path, &n);
    /* first pass: count entries (number of ':' at object depth 1) is overkill; just
     * size generously to vocab_size ~50257; grow id arrays as needed. */
    vtab_init(60000);
    int id2cap = 60000;
    g_id2bytes = (unsigned char**)xmalloc((size_t)id2cap * sizeof(unsigned char*));
    g_id2len   = (int*)xmalloc((size_t)id2cap * sizeof(int));
    for (int i = 0; i < id2cap; i++) { g_id2bytes[i] = NULL; g_id2len[i] = 0; }
    unsigned char keybuf[2048];
    size_t i = 0;
    /* skip to first '{' */
    while (i < n && s[i] != '{') i++;
    if (i < n) i++;
    for (;;) {
        while (i < n && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t' || s[i] == ',')) i++;
        if (i >= n || s[i] == '}') break;
        if (s[i] != '"') { i++; continue; }
        int klen = parse_json_string_to_bytes(s, n, &i, keybuf);
        while (i < n && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t' || s[i] == ':')) i++;
        /* parse integer id */
        int neg = 0; if (i < n && s[i] == '-') { neg = 1; i++; }
        long id = 0; int any = 0;
        while (i < n && s[i] >= '0' && s[i] <= '9') { id = id * 10 + (s[i] - '0'); i++; any = 1; }
        if (!any) continue;
        if (neg) id = -id;
        unsigned char* kb = (unsigned char*)xmalloc((size_t)klen ? (size_t)klen : 1);
        memcpy(kb, keybuf, klen);
        vtab_put(kb, klen, (int)id);
        if (id >= id2cap) {
            int old = id2cap; while (id >= id2cap) id2cap <<= 1;
            g_id2bytes = (unsigned char**)xrealloc(g_id2bytes, (size_t)id2cap * sizeof(unsigned char*));
            g_id2len   = (int*)xrealloc(g_id2len, (size_t)id2cap * sizeof(int));
            for (int j = old; j < id2cap; j++) { g_id2bytes[j] = NULL; g_id2len[j] = 0; }
        }
        g_id2bytes[id] = kb; g_id2len[id] = klen;
        if ((int)id + 1 > g_nvocab) g_nvocab = (int)id + 1;
    }
    free(s);
}

/* ============================ merges.txt ============================ */
/* Each merge rule: left byte-string + right byte-string -> rank. We store, for fast
 * rank lookup of an adjacent symbol pair, a hash table keyed by (left||0xFF||right).
 * 0xFF never appears in a b2u byte string boundary collision because we include the
 * exact lengths in the key separator scheme: we hash left then a separator then right
 * using their byte contents AND lengths. */
typedef struct { unsigned char* key; int klen; int rank; } MEntry;
static MEntry* g_mtab = NULL;
static int g_mcap = 0, g_mmask = 0;

static void mtab_init(int approx) {
    g_mcap = 1; while (g_mcap < approx * 2) g_mcap <<= 1;
    g_mmask = g_mcap - 1;
    g_mtab = (MEntry*)xmalloc((size_t)g_mcap * sizeof(MEntry));
    for (int i = 0; i < g_mcap; i++) { g_mtab[i].key = NULL; g_mtab[i].klen = 0; g_mtab[i].rank = 0; }
}
/* build a pair key: [llen:2][rlen:2][left bytes][right bytes] so it is unambiguous. */
static int make_pair_key(const unsigned char* L, int ll, const unsigned char* R, int rl, unsigned char* out) {
    out[0] = (unsigned char)(ll & 0xFF); out[1] = (unsigned char)((ll >> 8) & 0xFF);
    out[2] = (unsigned char)(rl & 0xFF); out[3] = (unsigned char)((rl >> 8) & 0xFF);
    memcpy(out + 4, L, ll); memcpy(out + 4 + ll, R, rl);
    return 4 + ll + rl;
}
static void mtab_put(unsigned char* key, int klen, int rank) {
    uint64_t h = fnv1a(key, klen);
    int idx = (int)(h & g_mmask);
    while (g_mtab[idx].key) idx = (idx + 1) & g_mmask;
    g_mtab[idx].key = key; g_mtab[idx].klen = klen; g_mtab[idx].rank = rank;
}
/* return rank for the pair (L,R) or -1 if not a merge. */
static int merge_rank(const unsigned char* L, int ll, const unsigned char* R, int rl) {
    unsigned char kb[4096];
    int klen = make_pair_key(L, ll, R, rl, kb);
    uint64_t h = fnv1a(kb, klen);
    int idx = (int)(h & g_mmask);
    while (g_mtab[idx].key) {
        if (g_mtab[idx].klen == klen && memcmp(g_mtab[idx].key, kb, klen) == 0)
            return g_mtab[idx].rank;
        idx = (idx + 1) & g_mmask;
    }
    return -1;
}

/* decode one whitespace-separated b2u token field at s[*i..] into a BYTE string;
 * stop at ' ', '\n', '\r', '\t', or EOF. Advances *i. Returns byte len. */
static int field_to_bytes(const char* s, size_t n, size_t* i, unsigned char* out) {
    int olen = 0;
    while (*i < n) {
        char c = s[(size_t)*i];
        if (c == ' ' || c == '\n' || c == '\r' || c == '\t') break;
        unsigned int cp;
        if ((unsigned char)c < 0x80) { cp = (unsigned char)c; (*i)++; }
        else cp = utf8_next((const unsigned char*)s, n, i);
        int b = (cp < 512) ? g_u2b[cp] : -1;
        if (b < 0) {
            if (cp < 0x80) out[olen++] = (unsigned char)cp;
            else if (cp < 0x800) { out[olen++] = (unsigned char)(0xC0 | (cp >> 6)); out[olen++] = (unsigned char)(0x80 | (cp & 0x3F)); }
            else { out[olen++] = (unsigned char)(0xE0 | (cp >> 12)); out[olen++] = (unsigned char)(0x80 | ((cp >> 6) & 0x3F)); out[olen++] = (unsigned char)(0x80 | (cp & 0x3F)); }
        } else out[olen++] = (unsigned char)b;
    }
    return olen;
}

void load_merges(const char* path) {
    size_t n; char* s = slurp(path, &n);
    mtab_init(60000);
    size_t i = 0;
    /* skip the first line if it is a "#version" comment (or any leading '#...' line). */
    if (i < n && s[i] == '#') { while (i < n && s[i] != '\n') i++; if (i < n) i++; }
    int rank = 0;
    unsigned char lb[2048], rb[2048], kb[4096];
    while (i < n) {
        /* skip blank lines */
        while (i < n && (s[i] == '\n' || s[i] == '\r')) i++;
        if (i >= n) break;
        int ll = field_to_bytes(s, n, &i, lb);
        while (i < n && (s[i] == ' ' || s[i] == '\t')) i++;
        int rl = field_to_bytes(s, n, &i, rb);
        /* advance to end of line */
        while (i < n && s[i] != '\n') i++;
        if (i < n) i++;
        if (ll == 0 || rl == 0) continue;          /* malformed/blank -> skip (keeps rank aligned with the oracle's [1:-1] slice) */
        int klen = make_pair_key(lb, ll, rb, rl, kb);
        unsigned char* key = (unsigned char*)xmalloc((size_t)klen);
        memcpy(key, kb, klen);
        mtab_put(key, klen, rank);
        rank++;
    }
    free(s);
}

/* ============================ BPE on a byte string ============================ */
/* symbols: a list of (offset,len) slices into the working byte buffer `w`. We merge
 * the lowest-rank adjacent pair repeatedly (ties -> lowest rank, == earliest merge
 * rule), exactly like the oracle's min(pairs, key=rank) + left-to-right rewrite. */
typedef struct { int off; int len; } Sym;

static void bpe_word(const unsigned char* w, int wlen, int* out_ids, int* out_n) {
    if (wlen == 0) { *out_n = 0; return; }
    /* fast path: whole word is a known token (the oracle's per-token cache hot path
     * is structurally just the merge loop; this is purely an optimization and does
     * not change the result -- the loop below would reach the same symbols). */
    Sym* syms = (Sym*)xmalloc((size_t)wlen * sizeof(Sym));
    int ns = wlen;
    for (int i = 0; i < wlen; i++) { syms[i].off = i; syms[i].len = 1; }
    for (;;) {
        if (ns < 2) break;
        int best_rank = -1, best_i = -1;
        for (int i = 0; i + 1 < ns; i++) {
            int r = merge_rank(w + syms[i].off, syms[i].len, w + syms[i + 1].off, syms[i + 1].len);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) { best_rank = r; best_i = i; }
        }
        if (best_i < 0) break;                     /* no mergeable pair */
        /* merge ALL non-overlapping occurrences of this exact pair, left-to-right,
         * matching the oracle's while-loop rewrite (which rebuilds `word` in one pass
         * merging every adjacent (first,second) it encounters). */
        int li = best_i;
        int ll = syms[li].len, rl = syms[li + 1].len;
        const unsigned char* Lp = w + syms[li].off;
        const unsigned char* Rp = w + syms[li + 1].off;
        int wi = 0;                                /* write index for the compacted list */
        Sym* ns_buf = syms;                        /* compact in place */
        int i = 0;
        while (i < ns) {
            if (i + 1 < ns
                && syms[i].len == ll && memcmp(w + syms[i].off, Lp, ll) == 0
                && syms[i + 1].len == rl && memcmp(w + syms[i + 1].off, Rp, rl) == 0) {
                ns_buf[wi].off = syms[i].off;
                ns_buf[wi].len = ll + rl;
                wi++; i += 2;
            } else {
                ns_buf[wi++] = syms[i++];
            }
        }
        ns = wi;
    }
    /* emit ids: each final symbol is a byte string -> vocab id. */
    int cnt = 0;
    for (int i = 0; i < ns; i++) {
        int id = vtab_get(w + syms[i].off, syms[i].len);
        if (id < 0) {
            /* This must not happen for GPT-2 (every byte and every reachable merge is
             * in the vocab). Fail closed loudly rather than emit a wrong id. */
            fprintf(stderr, "FATAL: BPE symbol not in vocab (len=%d): ", syms[i].len);
            for (int k = 0; k < syms[i].len; k++) fprintf(stderr, "%02x", w[syms[i].off + k]);
            fprintf(stderr, "\n");
            exit(3);
        }
        out_ids[cnt++] = id;
    }
    *out_n = cnt;
    free(syms);
}

/* ============================ pre-tokenizer (the hand-written GPT-2 split) ============================ */
/* We scan the UTF-8 text into codepoints. At each position we try, in order:
 *   1. contractions  's 't 're 've 'm 'll 'd   (apostrophe U+0027 then ASCII letters)
 *   2.  ?\p{L}+
 *   3.  ?\p{N}+
 *   4.  ?[^\s\p{L}\p{N}]+
 *   5. \s+(?!\S)   (whitespace run; if followed by a non-ws char, leave the LAST ws)
 *   6. \s+
 * First alternative that matches wins; quantifiers are greedy. Each produced chunk is
 * the substring of ORIGINAL bytes (UTF-8), which is what gets byte-level-BPE'd.
 *
 * Implementation detail: we precompute a codepoint array with each codepoint's byte
 * span [bstart,bend) in the original buffer, so a chunk is just a byte range. */
typedef struct { unsigned int cp; size_t bstart; } CP;

static void emit_chunk(const unsigned char* text, size_t b0, size_t b1,
                       int* ids, int* nids, int* cap) {
    int len = (int)(b1 - b0);
    if (len <= 0) return;
    int tmp_n = 0;
    int* tmp = (int*)xmalloc((size_t)len * sizeof(int));   /* at most `len` ids (each byte) */
    bpe_word(text + b0, len, tmp, &tmp_n);
    if (*nids + tmp_n > *cap) { while (*nids + tmp_n > *cap) *cap <<= 1; }
    /* caller's ids buffer is grown by the driver; here we assume cap already ensured */
    for (int k = 0; k < tmp_n; k++) ids[(*nids)++] = tmp[k];
    free(tmp);
}

/* ---- ADDITIVE: special-token-aware encode (instruct-model chat, 2026-06) ----
 * ChatML control tokens (<|im_start|>, <|im_end|>, ...) live IN vocab.json but are
 * never produced by BPE merges, so encode must SPLIT on the literal strings first.
 * tok_enable_specials() registers every vocab entry of the form <...> (longest-first);
 * encode_bytes_special() then emits gap-text through the UNCHANGED encode_bytes() and
 * splices the special ids. Mirrors the python oracle (llama_numpy_ref.encode_special);
 * parity is part of the gated token-for-token claim. Byte movement + table lookup only. */
int* encode_bytes(const unsigned char* text, size_t n, int* out_n);   /* defined below */
typedef struct { const unsigned char* bytes; int len; int id; } SpecialTok;
static SpecialTok g_specials[64];
static int g_nspecials = 0;
void tok_enable_specials(void) {
    g_nspecials = 0;
    for (int id = 0; id < g_nvocab && g_nspecials < 64; id++) {
        const unsigned char* b = g_id2bytes[id]; int l = g_id2len[id];
        if (b && l >= 3 && b[0] == '<' && b[l-1] == '>') {
            g_specials[g_nspecials].bytes = b; g_specials[g_nspecials].len = l;
            g_specials[g_nspecials].id = id; g_nspecials++;
        }
    }
    /* longest-first so <|im_start|> beats any shorter overlapping form */
    for (int i = 0; i < g_nspecials; i++)
        for (int j = i + 1; j < g_nspecials; j++)
            if (g_specials[j].len > g_specials[i].len) { SpecialTok t = g_specials[i]; g_specials[i] = g_specials[j]; g_specials[j] = t; }
    fprintf(stderr, "[tok] specials enabled: %d control tokens\n", g_nspecials);
}
int* encode_bytes_special(const unsigned char* text, size_t n, int* out_n) {
    int cap = 1024, nids = 0;
    int* ids = (int*)xmalloc((size_t)cap * sizeof(int));
    size_t i = 0;
    while (i < n) {
        /* earliest (then longest) special match at or after i */
        size_t best_pos = n; int best = -1;
        for (int k = 0; k < g_nspecials; k++) {
            const SpecialTok* sp = &g_specials[k];
            for (size_t j = i; j + (size_t)sp->len <= n && j <= best_pos; j++) {
                if (text[j] == sp->bytes[0] && memcmp(text + j, sp->bytes, (size_t)sp->len) == 0) {
                    if (j < best_pos) { best_pos = j; best = k; }
                    break;
                }
            }
        }
        size_t seg_end = (best >= 0) ? best_pos : n;
        if (seg_end > i) {
            int m = 0; int* seg = encode_bytes(text + i, seg_end - i, &m);
            if (nids + m + 1 > cap) { while (nids + m + 1 > cap) cap <<= 1; ids = (int*)xrealloc(ids, (size_t)cap * sizeof(int)); }
            memcpy(ids + nids, seg, (size_t)m * sizeof(int)); nids += m; free(seg);
        }
        if (best >= 0) {
            if (nids + 1 > cap) { cap <<= 1; ids = (int*)xrealloc(ids, (size_t)cap * sizeof(int)); }
            ids[nids++] = g_specials[best].id;
            i = best_pos + (size_t)g_specials[best].len;
        } else break;
    }
    *out_n = nids;
    return ids;
}

int* encode_bytes(const unsigned char* text, size_t n, int* out_n) {
    /* build codepoint index */
    CP* cps = (CP*)xmalloc((n + 1) * sizeof(CP));
    int ncp = 0;
    { size_t i = 0; while (i < n) { size_t st = i; unsigned int cp = utf8_next(text, n, &i); cps[ncp].cp = cp; cps[ncp].bstart = st; ncp++; } }
    cps[ncp].cp = 0; cps[ncp].bstart = n;          /* sentinel: bstart of "end" = n */

    int cap = 1024, nids = 0;
    int* ids = (int*)xmalloc((size_t)cap * sizeof(int));

    int p = 0;                                     /* codepoint index */
    while (p < ncp) {
        size_t b_start = cps[p].bstart;
        unsigned int c = cps[p].cp;

        /* --- alt 1: contractions --- */
        if (c == '\'') {
            /* check 'll / 're / 've then single-letter 's 't 'm 'd  (order in the
             * pattern: 's 't 're 've 'm 'll 'd -- but as fixed strings, longest unique
             * match is unambiguous since they don't prefix-collide except 'l-> 'll). */
            unsigned int c1 = (p + 1 < ncp) ? cps[p + 1].cp : 0;
            unsigned int c2 = (p + 2 < ncp) ? cps[p + 2].cp : 0;
            int adv = 0;
            if      (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') adv = 2;
            else if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') || (c1 == 'l' && c2 == 'l')) adv = 3;
            if (adv) {
                size_t b_end = cps[p + adv].bstart;
                if (nids + (int)(b_end - b_start) > cap) { while (nids + (int)(b_end - b_start) > cap) cap <<= 1; ids = (int*)xrealloc(ids, (size_t)cap * sizeof(int)); }
                emit_chunk(text, b_start, b_end, ids, &nids, &cap);
                p += adv;
                continue;
            }
        }

        /* --- alts 2/3/4 share the optional leading single space --- */
        {
            int q = p;
            int had_space = 0;
            if (cps[q].cp == ' ') { had_space = 1; q++; }     /* ' ?' : one literal space */
            unsigned int d = (q < ncp) ? cps[q].cp : 0;
            if (q < ncp && is_L(d)) {                          /* alt 2:  ?\p{L}+ */
                int r = q; while (r < ncp && is_L(cps[r].cp)) r++;
                size_t b_end = cps[r].bstart;
                if (nids + (int)(b_end - b_start) > cap) { while (nids + (int)(b_end - b_start) > cap) cap <<= 1; ids = (int*)xrealloc(ids, (size_t)cap * sizeof(int)); }
                emit_chunk(text, b_start, b_end, ids, &nids, &cap);
                p = r; continue;
            }
            if (q < ncp && is_N(d)) {                          /* alt 3:  ?\p{N}+ */
                int r = q; while (r < ncp && is_N(cps[r].cp)) r++;
                size_t b_end = cps[r].bstart;
                if (nids + (int)(b_end - b_start) > cap) { while (nids + (int)(b_end - b_start) > cap) cap <<= 1; ids = (int*)xrealloc(ids, (size_t)cap * sizeof(int)); }
                emit_chunk(text, b_start, b_end, ids, &nids, &cap);
                p = r; continue;
            }
            if (q < ncp && !is_S(d) && !is_L(d) && !is_N(d)) { /* alt 4:  ?[^\s\p{L}\p{N}]+ */
                int r = q; while (r < ncp && !is_S(cps[r].cp) && !is_L(cps[r].cp) && !is_N(cps[r].cp)) r++;
                size_t b_end = cps[r].bstart;
                if (nids + (int)(b_end - b_start) > cap) { while (nids + (int)(b_end - b_start) > cap) cap <<= 1; ids = (int*)xrealloc(ids, (size_t)cap * sizeof(int)); }
                emit_chunk(text, b_start, b_end, ids, &nids, &cap);
                p = r; continue;
            }
            /* if we consumed a lone trailing space (had_space) but the next char is not
             * L/N/other (i.e. it is whitespace or EOF), alt 2/3/4 did NOT match -> fall
             * through to the whitespace alternatives, which will re-handle from p. */
            (void)had_space;
        }

        /* --- alt 5: \s+(?!\S) --- and alt 6: \s+ --- */
        if (is_S(c)) {
            int r = p; while (r < ncp && is_S(cps[r].cp)) r++;   /* maximal whitespace run [p,r) */
            /* alt 5 first: \s+ then (?!\S). Greedy match consumes [p,r); the char after
             * is cps[r] (EOF or non-ws). (?!\S) fails iff cps[r] is a non-ws char.
             * Backtrack one ws at a time until the following char is ws or EOF. Since
             * [p,r) is the maximal run, after consuming all of it the next is non-ws or
             * EOF; if non-ws, alt 5 must leave the last ws (so the char after the match
             * is whitespace). Net: if r<ncp (non-ws follows) and run length>1, alt 5
             * matches [p, r-1); if run length==1 and non-ws follows, alt 5 matches
             * nothing (lookahead fails for the only char) and alt 6 takes the 1 ws. */
            int end5;
            if (r >= ncp) end5 = r;                 /* run at EOF: (?!\S) holds for whole run */
            else end5 = r - 1;                      /* leave the last ws so a ws follows */
            if (end5 > p) {                         /* alt 5 matched a nonempty prefix */
                size_t b_end = cps[end5].bstart;
                if (nids + (int)(b_end - b_start) > cap) { while (nids + (int)(b_end - b_start) > cap) cap <<= 1; ids = (int*)xrealloc(ids, (size_t)cap * sizeof(int)); }
                emit_chunk(text, b_start, b_end, ids, &nids, &cap);
                p = end5; continue;
            }
            /* alt 6: \s+ (greedy) -- here the run is a single ws followed by non-ws (or
             * end5==p). Match the whole maximal run [p,r). */
            {
                size_t b_end = cps[r].bstart;
                if (nids + (int)(b_end - b_start) > cap) { while (nids + (int)(b_end - b_start) > cap) cap <<= 1; ids = (int*)xrealloc(ids, (size_t)cap * sizeof(int)); }
                emit_chunk(text, b_start, b_end, ids, &nids, &cap);
                p = r; continue;
            }
        }

        /* Should be unreachable: every codepoint is L, N, whitespace, or "other".
         * Advance one to guarantee progress (fail-safe). */
        {
            size_t b_end = cps[p + 1].bstart;
            if (nids + (int)(b_end - b_start) > cap) { while (nids + (int)(b_end - b_start) > cap) cap <<= 1; ids = (int*)xrealloc(ids, (size_t)cap * sizeof(int)); }
            emit_chunk(text, b_start, b_end, ids, &nids, &cap);
            p++;
        }
    }
    free(cps);
    *out_n = nids;
    return ids;
}

/* ============================ decode ============================ */
static void decode_ids(const int* ids, int n) {
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= g_nvocab || g_id2bytes[id] == NULL) {
            fprintf(stderr, "FATAL: decode id %d out of range / no token\n", id);
            exit(3);
        }
        fwrite(g_id2bytes[id], 1, (size_t)g_id2len[id], stdout);
    }
}

/* ============================ decode-to-buffer (serve mode; LIB-EXPOSED) ============================ */
/* Pure byte concatenation of g_id2bytes[id] -- the SAME bytes decode_ids() writes to stdout,
 * returned as a NUL-terminated malloc'd C string instead. ZERO arithmetic; out-of-range ids
 * are rendered as the empty string (never abort the long-lived serve worker). The contract's
 * per-id strings[], per-token token.string, and final done.text all come from these. */
char* decode_range(const int* ids, int n) {
    size_t cap = 64; char* out = (char*)xmalloc(cap); size_t k = 0;
    for (int i = 0; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= g_nvocab || g_id2bytes[id] == NULL) continue;
        int len = g_id2len[id];
        if (k + (size_t)len + 1 >= cap) { while (k + (size_t)len + 1 >= cap) cap <<= 1; out = (char*)xrealloc(out, cap); }
        memcpy(out + k, g_id2bytes[id], (size_t)len); k += (size_t)len;
    }
    out[k] = 0;
    return out;
}
char* decode_one(int id) { return decode_range(&id, 1); }

/* ============================ main ============================ */
#ifndef GPT2_TOK_LIB
int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
          "usage:\n"
          "  %s <vocab.json> <merges.txt> encode            (stdin text -> ids)\n"
          "  %s <vocab.json> <merges.txt> encode-file <in>  (file text  -> ids)\n"
          "  %s <vocab.json> <merges.txt> decode <id>...     (ids args   -> text)\n"
          "  %s <vocab.json> <merges.txt> decode-file <ids>  (ids file   -> text)\n",
          argv[0], argv[0], argv[0], argv[0]);
        return 2;
    }
    const char* vocab_path  = argv[1];
    const char* merges_path = argv[2];
    const char* mode        = argv[3];

    build_byte_unicode();
    load_vocab(vocab_path);
    load_merges(merges_path);

    if (strcmp(mode, "encode") == 0 || strcmp(mode, "encode-file") == 0) {
        size_t n; char* text;
        if (strcmp(mode, "encode-file") == 0) {
            if (argc < 5) { fprintf(stderr, "encode-file needs <in>\n"); return 2; }
            text = slurp(argv[4], &n);
        } else {
            text = slurp_stdin(&n);
        }
        int nids;
        int* ids = encode_bytes((const unsigned char*)text, n, &nids);
        for (int i = 0; i < nids; i++) printf("%d%s", ids[i], i + 1 < nids ? " " : "\n");
        free(ids); free(text);
        return 0;
    } else if (strcmp(mode, "decode") == 0) {
        int n = argc - 4;
        int* ids = (int*)xmalloc((size_t)(n ? n : 1) * sizeof(int));
        for (int i = 0; i < n; i++) ids[i] = atoi(argv[4 + i]);
        decode_ids(ids, n);
        free(ids);
        return 0;
    } else if (strcmp(mode, "decode-file") == 0) {
        if (argc < 5) { fprintf(stderr, "decode-file needs <ids>\n"); return 2; }
        size_t n; char* s = slurp(argv[4], &n);
        int cap = 1024, cnt = 0;
        int* ids = (int*)xmalloc((size_t)cap * sizeof(int));
        size_t i = 0;
        while (i < n) {
            while (i < n && (s[i] == ' ' || s[i] == '\n' || s[i] == '\r' || s[i] == '\t')) i++;
            if (i >= n) break;
            int neg = 0; if (s[i] == '-') { neg = 1; i++; }
            long v = 0; int any = 0;
            while (i < n && s[i] >= '0' && s[i] <= '9') { v = v * 10 + (s[i] - '0'); i++; any = 1; }
            if (!any) break;
            if (neg) v = -v;
            if (cnt == cap) { cap <<= 1; ids = (int*)xrealloc(ids, (size_t)cap * sizeof(int)); }
            ids[cnt++] = (int)v;
        }
        decode_ids(ids, cnt);
        free(ids); free(s);
        return 0;
    } else {
        fprintf(stderr, "unknown mode '%s'\n", mode);
        return 2;
    }
}
#endif /* GPT2_TOK_LIB */
