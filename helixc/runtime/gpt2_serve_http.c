/* gpt2_serve_http.c -- dependency-light, NO-PYTHON local HTTP+SSE server for the
 * GPT-2-XL-on-Helix chat demo. Category-B HOST TOOL (HTTP/byte-pump), OUTSIDE the
 * self-host fixpoint, with ZERO arithmetic on the compute-trust path -- exactly like
 * helixc/runtime/cpu_host.c / gpt2_tok.c / gpt2_pack.c.
 *
 * WHAT IT DOES
 *   (a) serves demo/ static files (GET / -> index.html, /dashboard.html, assets with
 *       correct content-types; rejects path traversal '..' / absolute paths), and
 *   (b) bridges the browser to ONE persistent `gpt2_infer --serve` worker child:
 *       - spawns the worker once over two pipes (server->worker stdin = request frames;
 *         worker->server stdout = newline-JSON telemetry events; worker stderr is read
 *         until the worker prints GPT2_SERVE_READY, then logged),
 *       - POST /api/generate streams text/event-stream: it writes the request frame to the
 *         worker stdin and re-frames EACH worker stdout line as one SSE message, flushing
 *         immediately (TCP_NODELAY). The server NEVER reformats event content -- the worker's
 *         JSON line is the SSE data: verbatim; the server only adds id:/event:/blank framing
 *         by reading "_ev" and "seq" out of the line.
 *       - GET /api/health -> readiness JSON (worker spawned + GPT2_SERVE_READY seen).
 *       - POST /api/verify -> JSON; returns {"verdict":"UNAVAILABLE"} if python3 / the numpy
 *         oracle is absent (the demo path is Python-free by design) -- NEVER fakes a verdict.
 *
 * SINGLE-FLIGHT: the GPU path is strictly serial (every cuLaunchKernel is followed by
 * cuCtxSynchronize), so the server serializes /api/generate behind ONE mutex; a second
 * concurrent generation gets 409 {"error":"busy"}. This matches the real hardware constraint.
 *
 * Binds 127.0.0.1 ONLY. Pure C + POSIX sockets + libc; buildable with the same gcc the rest
 * of the demo uses; zero third-party deps.
 *
 * Build (host): gcc gpt2_serve_http.c -O2 -lpthread -o gpt2_serve_http
 * Run:
 *   gpt2_serve_http --port 8848 --root <abs demo dir> \
 *       --ptx <combined.ptx> --weights <gpt2-xl.weights> \
 *       --vocab <vocab.json> --merges <merges.txt> \
 *       [--worker-bin <gpt2_infer_serve>] [--max-ctx 320] [--detail op] \
 *       [--oracle <gpt2_numpy_ref.py dir>] [--model gpt2-xl]
 *   -- or pass the whole worker argv explicitly with --worker "<cmd...>".
 *
 * License: Apache 2.0.
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/time.h>           /* struct timeval for SO_RCVTIMEO */
#include <sys/wait.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <limits.h>
#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

/* Upper bound on a request body (the only POST payloads are tiny JSON: a prompt + n_gen).
 * A client-supplied Content-Length above this is rejected with 413 BEFORE any allocation,
 * so a malicious/oversized Content-Length cannot drive a multi-GB malloc (DoS). */
#define MAX_BODY (256 * 1024)

/* DoS hardening for the accept loop (the listener is 127.0.0.1-only, but a local slow-loris could
 * otherwise dribble partial headers and pin one thread + FD + 16KB stack per connection forever,
 * and there was no cap on concurrent connections):
 *   - CONN_RCVTIMEO_SEC: a per-accepted-connection SO_RCVTIMEO read deadline so a stalled header
 *     (or body) read returns EAGAIN/EWOULDBLOCK instead of blocking a thread indefinitely.
 *   - MAX_CONN: a hard cap on concurrent connection-handler threads. Beyond it the listener closes
 *     the new fd immediately (the kernel still has the backlog from listen(); legitimate clients
 *     retry). Cheap counter guarded by a mutex; the single-flight GPU mutex is a SEPARATE layer that
 *     only serializes /api/generate and does NOT bound connection/thread growth. */
#define CONN_RCVTIMEO_SEC 10
/* per-connection WRITE deadline -- the symmetric complement to SO_RCVTIMEO. A client that stops
 * READING the SSE stream would otherwise fill the socket send buffer and block write() forever WHILE
 * this thread holds the single-flight g_worker.lock, pinning the one GPU generation slot indefinitely
 * (every other /api/generate then 409s -- the round-5 starvation vector). With SO_SNDTIMEO a stalled
 * write returns EAGAIN after CONN_SNDTIMEO_SEC; handle_generate then stops writing to the dead client
 * but KEEPS draining the worker to its terminal event, so the persistent worker stays protocol-
 * coherent and the mutex hold is bounded to one generation instead of forever. */
#define CONN_SNDTIMEO_SEC 20
#define MAX_CONN          16

/* Worst-case worker telemetry line: the `tokenize` event (ids[]+strings[] for up to a max-ctx
 * prompt) and the `done` event (gen_ids[] up to n_gen=256 + the decoded text). At max-ctx 320 +
 * n_gen 256 these are a few KB; we size the per-line pump + SSE framing buffers WELL above that so a
 * pathological long prompt can never truncate a valid tokenize/done JSON line into invalid JSON that
 * the browser would silently drop. snprintf still truncates safely (no overflow); this just makes
 * the line buffer comfortably exceed the worst case so no real event is ever cut. */
#define LINE_BUF  (256 * 1024)                 /* worker stdout line pump */
#define SSE_BUF   (LINE_BUF + 256)             /* SSE frame = data line + small id:/event: header */

/* ============================ config ============================ */
typedef struct {
    int   port;
    char  root[1024];          /* abs path to demo/ */
    char  ptx[1024];
    char  weights[1024];
    char  vocab[1024];
    char  merges[1024];
    char  worker_bin[1024];    /* path to the gpt2_infer serve binary */
    int   max_ctx;
    char  detail[16];
    char  oracle_dir[1024];    /* dir containing gpt2_numpy_ref.py (optional) */
    char  model[64];
    /* ---- ADDITIVE second model (2026-06, the modern-model leg): all five --*2 flags must
     * be given to enable it. Same worker binary; the v2 weight header makes it self-config. */
    char  model2[64];
    char  ptx2[1024];
    char  weights2[1024];
    char  vocab2[1024];
    char  merges2[1024];
    int   specials2;          /* HX_SPECIALS for slot 1 (ChatML control tokens; instruct models) */
    int   eos2;               /* HX_EOS for slot 1 (-1 = none) */
    /* slot 2 (third model -- e.g. the instruct model alongside XL + the base model) */
    char  model3[64];
    char  ptx3[1024];
    char  weights3[1024];
    char  vocab3[1024];
    char  merges3[1024];
    int   specials3;
    int   eos3;
    int   kv2;                /* HX_KV+HX_RESIDENT for slot 1 (KV-cache decode; small llama models) */
    int   kv3;                /* same for slot 2 */
} Cfg;

/* ============================ worker child ============================ */
typedef struct {
    pid_t pid;
    int   to_worker;           /* server -> worker stdin (write) */
    int   from_worker;         /* worker stdout -> server (read) */
    int   ready;               /* GPT2_SERVE_READY seen */
    char  device[256];         /* real device name, captured from the worker's first hello SSE event */
    pthread_mutex_t lock;      /* single-flight: one generation at a time */
} Worker;

static Cfg     g_cfg;
#define MAX_MODELS 3
static Worker  g_workers[MAX_MODELS];     /* [0] = the primary model; [1] = the optional second */
static int     g_nmodels = 1;
#define g_worker (g_workers[0])           /* existing single-model references stay valid */
/* ONE GLOBAL GPU mutex: generations are single-flight ACROSS models (strictly serial GPU).
 * The per-worker .lock fields remain but the cross-model gate is this one. */
static pthread_mutex_t g_gpu_lock = PTHREAD_MUTEX_INITIALIZER;
static volatile int g_busy = 0;   /* reflected in /api/health */

/* model-i config view (keeps the v1 scalar fields untouched for model 0) */
static const char* cfg_model_name(int i) {
    if (i == 2) return g_cfg.model3;
    if (i == 1) return g_cfg.model2;
    return g_cfg.model[0] ? g_cfg.model : "gpt2-xl";
}

/* concurrent-connection cap (DoS): incremented when a handler thread starts, decremented when it
 * ends; the accept loop refuses (closes) a new fd once g_conns >= MAX_CONN. */
static int             g_conns = 0;
static pthread_mutex_t g_conns_lock = PTHREAD_MUTEX_INITIALIZER;

/* ============================ small helpers ============================ */
static void die(const char* msg) { fprintf(stderr, "gpt2_serve_http: %s: %s\n", msg, strerror(errno)); exit(2); }

static ssize_t write_all(int fd, const char* buf, size_t n) {
    size_t off = 0;
    while (off < n) {
        ssize_t w = write(fd, buf + off, n - off);
        if (w < 0) { if (errno == EINTR) continue; return -1; }
        if (w == 0) break;
        off += (size_t)w;
    }
    return (ssize_t)off;
}

/* read one '\n'-terminated line from fd into buf (incl the '\n'); returns length, 0 on EOF,
 * -1 on error. Byte-at-a-time is fine: the worker emits whole JSON lines and we want them as
 * they arrive (real-time SSE cadence). */
static ssize_t read_line_fd(int fd, char* buf, size_t cap) {
    size_t k = 0;
    while (k + 1 < cap) {
        char c; ssize_t r = read(fd, &c, 1);
        if (r < 0) { if (errno == EINTR) continue; return -1; }
        if (r == 0) { if (k == 0) return 0; break; }
        buf[k++] = c;
        if (c == '\n') break;
    }
    buf[k] = 0;
    return (ssize_t)k;
}

/* find "key": then return the VALUE token (numeric or bareword) into out; or "" if absent.
 * Only used to pull "_ev" and "seq" out of a worker line for the SSE framing (not a parser). */
static void json_str_field(const char* line, const char* key, char* out, size_t outcap) {
    out[0] = 0;
    char pat[64]; snprintf(pat, sizeof pat, "\"%s\"", key);
    const char* p = strstr(line, pat);
    if (!p) return;
    p += strlen(pat);
    while (*p == ' ' || *p == ':') p++;
    if (*p == '"') {                       /* string value */
        p++;
        size_t k = 0;
        while (*p && *p != '"' && k + 1 < outcap) { if (*p == '\\' && p[1]) p++; out[k++] = *p++; }
        out[k] = 0;
    } else {                               /* numeric/bareword */
        size_t k = 0;
        while (*p && *p != ',' && *p != '}' && *p != ' ' && k + 1 < outcap) out[k++] = *p++;
        out[k] = 0;
    }
}

/* The telemetry contract: the ONLY worker "_ev" values that may be re-framed onto the SSE wire.
 * Any worker stdout line that is not a JSON object carrying one of these is DROPPED (it never
 * reaches the browser) so the stream stays pure newline-JSON telemetry -- no decorative
 * "event: message" leak, no internal diagnostic line on a wire that may be web-exposed. */
static int is_telemetry_event(const char* ev) {
    if (!ev || !ev[0]) return 0;
    static const char* const CONTRACT[] = {
        "hello", "tokenize", "forward_begin", "embed", "layer_begin", "op",
        "layer_end", "head", "token", "done", "error", NULL
    };
    for (int i = 0; CONTRACT[i]; i++) if (!strcmp(ev, CONTRACT[i])) return 1;
    return 0;
}

/* ============================ static file serving ============================ */
static const char* mime_for(const char* path) {
    const char* dot = strrchr(path, '.');
    if (!dot) return "application/octet-stream";
    if (!strcmp(dot, ".html")) return "text/html; charset=utf-8";
    if (!strcmp(dot, ".js"))   return "text/javascript; charset=utf-8";
    if (!strcmp(dot, ".css"))  return "text/css; charset=utf-8";
    if (!strcmp(dot, ".json")) return "application/json; charset=utf-8";
    if (!strcmp(dot, ".svg"))  return "image/svg+xml";
    if (!strcmp(dot, ".png"))  return "image/png";
    if (!strcmp(dot, ".woff2"))return "font/woff2";
    if (!strcmp(dot, ".ico"))  return "image/x-icon";
    if (!strcmp(dot, ".txt"))  return "text/plain; charset=utf-8";
    return "application/octet-stream";
}

static void send_status(int fd, int code, const char* status, const char* ctype, const char* body) {
    char hdr[512];
    size_t blen = body ? strlen(body) : 0;
    int n = snprintf(hdr, sizeof hdr,
        "HTTP/1.1 %d %s\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %zu\r\n"
        "Connection: close\r\n"
        "Access-Control-Allow-Origin: http://127.0.0.1:%d\r\n"
        "\r\n",
        code, status, ctype, blen, g_cfg.port);
    write_all(fd, hdr, (size_t)n);
    if (blen) write_all(fd, body, blen);
}

/* serve demo/<urlpath>; reject traversal. urlpath starts with '/'. */
static void serve_static(int fd, const char* urlpath) {
    /* reject '..' and backslashes outright (no traversal) */
    if (strstr(urlpath, "..") || strchr(urlpath, '\\')) { send_status(fd, 403, "Forbidden", "text/plain", "forbidden\n"); return; }
    const char* rel = urlpath;
    if (rel[0] == '/') rel++;
    if (rel[0] == 0 || !strcmp(rel, "")) rel = "index.html";

    char full[3200];
    snprintf(full, sizeof full, "%s/%s", g_cfg.root, rel);

    /* canonicalize and confirm the resolved path stays under root (defense in depth).
     * realpath() writes up to PATH_MAX bytes -- the buffers MUST be PATH_MAX-sized or it
     * smashes the stack (fortify abort). */
    char rp[PATH_MAX], root_rp[PATH_MAX];
    if (!realpath(full, rp) || !realpath(g_cfg.root, root_rp)) { send_status(fd, 404, "Not Found", "text/plain", "not found\n"); return; }
    size_t rl = strlen(root_rp);
    if (strncmp(rp, root_rp, rl) != 0 || (rp[rl] != '/' && rp[rl] != 0)) { send_status(fd, 403, "Forbidden", "text/plain", "forbidden\n"); return; }

    int f = open(rp, O_RDONLY);
    if (f < 0) { send_status(fd, 404, "Not Found", "text/plain", "not found\n"); return; }
    struct stat st;
    if (fstat(f, &st) != 0 || !S_ISREG(st.st_mode)) { close(f); send_status(fd, 404, "Not Found", "text/plain", "not found\n"); return; }

    char hdr[512];
    int n = snprintf(hdr, sizeof hdr,
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: %s\r\n"
        "Content-Length: %lld\r\n"
        "Cache-Control: no-cache\r\n"
        "Connection: close\r\n"
        "\r\n",
        mime_for(rp), (long long)st.st_size);
    write_all(fd, hdr, (size_t)n);
    char buf[65536]; ssize_t r;
    while ((r = read(f, buf, sizeof buf)) > 0) { if (write_all(fd, buf, (size_t)r) < 0) break; }
    close(f);
}

/* ============================ /api/health ============================ */
static void handle_health(int fd) {
    /* top-level model/ready/device keep describing model 0 (frontend backward-compat);
     * the ADDITIVE models[] array advertises every loaded model (the switcher capability). */
    char models[512]; size_t mo = 0;
    mo += (size_t)snprintf(models + mo, sizeof models - mo, "[");
    for (int i = 0; i < g_nmodels; i++) {
        mo += (size_t)snprintf(models + mo, sizeof models - mo,
            "%s{\"model\":\"%s\",\"ready\":%s,\"device\":\"%s\"}",
            i ? "," : "", cfg_model_name(i),
            g_workers[i].ready ? "true" : "false",
            g_workers[i].device[0] ? g_workers[i].device : "");
    }
    snprintf(models + mo, sizeof models - mo, "]");
    char body[1024];
    snprintf(body, sizeof body,
        "{\"ok\":true,\"serve\":true,\"fast\":true,\"model\":\"%s\",\"ready\":%s,\"device\":\"%s\",\"busy\":%s,\"models\":%s}",
        cfg_model_name(0),
        g_worker.ready ? "true" : "false",
        g_worker.device[0] ? g_worker.device : "",
        g_busy ? "true" : "false",
        models);
    send_status(fd, 200, "OK", "application/json; charset=utf-8", body);
}

/* ============================ /api/generate (SSE bridge) ============================ */
/* read the HTTP request body given the already-read header (Content-Length). */
static char* read_body(int fd, const char* hdr_end, const char* hdrbuf, size_t hdrlen, int content_len) {
    if (content_len <= 0 || content_len > MAX_BODY) return NULL;   /* bound the allocation */
    char* body = (char*)malloc((size_t)content_len + 1);
    if (!body) return NULL;                                        /* no NULL-deref below */
    /* bytes already in the header buffer past hdr_end */
    size_t have = 0;
    size_t consumed = (size_t)(hdr_end - hdrbuf);
    if (consumed < hdrlen) {
        have = hdrlen - consumed;
        if (have > (size_t)content_len) have = (size_t)content_len;
        memcpy(body, hdr_end, have);
    }
    while (have < (size_t)content_len) {
        ssize_t r = read(fd, body + have, (size_t)content_len - have);
        if (r <= 0) { if (r < 0 && errno == EINTR) continue; break; }
        have += (size_t)r;
    }
    body[have] = 0;
    return body;
}

/* extract "prompt" / "n_gen" / "request_id" out of the request body and build the worker
 * frame line. Returns a malloc'd frame ('\n'-terminated) or NULL. */
static char* build_worker_frame(const char* body) {
    /* We forward prompt + n_gen + request_id to the worker verbatim-ish: simplest correct
     * thing is to pass the body straight through (the worker's parser reads the same keys),
     * after stripping any trailing newline and ensuring exactly one. The worker clamps n_gen. */
    size_t n = body ? strlen(body) : 0;
    char* frame = (char*)malloc(n + 2);
    if (!frame) return NULL;                  /* caller handles NULL (no deref) */
    size_t k = 0;
    for (size_t i = 0; i < n; i++) { if (body[i] == '\n' || body[i] == '\r') continue; frame[k++] = body[i]; }
    frame[k++] = '\n';
    frame[k] = 0;
    return frame;
}

static int sse_write_event(int fd, const char* seq, const char* ev, const char* data_line) {
    /* data_line includes its own trailing '\n'; strip it for the SSE data: field. */
    char buf[SSE_BUF];
    /* copy data_line minus trailing newline */
    size_t dl = strlen(data_line);
    while (dl > 0 && (data_line[dl-1] == '\n' || data_line[dl-1] == '\r')) dl--;
    /* `ev` is ALWAYS a non-empty whitelisted contract event: handle_generate only calls this after
     * is_telemetry_event(ev) passed, and the 3 error call sites pass the literal "error". So there is
     * no decorative/empty fallback -- the SSE event field is always a known telemetry name. */
    int n = snprintf(buf, sizeof buf, "id: %s\nevent: %s\ndata: %.*s\n\n",
                     (seq && seq[0]) ? seq : "0", ev, (int)dl, data_line);
    if (n > 0) {
        /* snprintf truncates safely; clamp the write to what actually fit in the buffer. */
        size_t wn = (n < (int)sizeof buf) ? (size_t)n : sizeof buf - 1;
        return (write_all(fd, buf, wn) < 0) ? -1 : 0;   /* -1 == client gone / send timed out */
    }
    return 0;
}

static void handle_generate(int fd, const char* body) {
    /* model routing (ADDITIVE): an optional "model" field in the request body picks the
     * worker; absent -> model 0 (unchanged v1 behavior). Unknown model -> honest 404. */
    int midx = 0;
    {
        char want[64] = {0};
        json_str_field(body, "model", want, sizeof want);
        if (want[0]) {
            midx = -1;
            for (int i = 0; i < g_nmodels; i++)
                if (strcmp(want, cfg_model_name(i)) == 0) { midx = i; break; }
            if (midx < 0) {
                send_status(fd, 404, "Not Found", "application/json; charset=utf-8", "{\"error\":\"unknown model\"}");
                return;
            }
        }
    }
    Worker* w = &g_workers[midx];

    /* single-flight ACROSS models: try-lock the ONE GPU mutex; if busy -> 409. */
    if (pthread_mutex_trylock(&g_gpu_lock) != 0) {
        send_status(fd, 409, "Conflict", "application/json; charset=utf-8", "{\"error\":\"busy\"}");
        return;
    }
    g_busy = 1;

    if (!w->ready) {
        g_busy = 0; pthread_mutex_unlock(&g_gpu_lock);
        send_status(fd, 503, "Service Unavailable", "application/json; charset=utf-8", "{\"error\":\"worker not ready\"}");
        return;
    }

    /* SSE headers + TCP_NODELAY for real-time flush. */
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    const char* sse_hdr =
        "HTTP/1.1 200 OK\r\n"
        "Content-Type: text/event-stream; charset=utf-8\r\n"
        "Cache-Control: no-cache, no-transform\r\n"
        "Connection: keep-alive\r\n"
        "X-Accel-Buffering: no\r\n"
        "\r\n";
    write_all(fd, sse_hdr, strlen(sse_hdr));

    /* write the request frame to the worker stdin. */
    char* frame = build_worker_frame(body);
    if (!frame) {
        sse_write_event(fd, "0", "error", "{\"_ev\":\"error\",\"where\":\"driver\",\"message\":\"out of memory\",\"fatal\":true}\n");
        g_busy = 0; pthread_mutex_unlock(&g_gpu_lock); return;
    }
    if (write_all(w->to_worker, frame, strlen(frame)) < 0) {
        sse_write_event(fd, "0", "error", "{\"_ev\":\"error\",\"where\":\"driver\",\"message\":\"worker stdin closed\",\"fatal\":true}\n");
        free(frame); g_busy = 0; pthread_mutex_unlock(&g_gpu_lock); return;
    }
    free(frame);

    /* pump worker stdout lines -> SSE until done/fatal-error (or worker EOF).
     * LINE_BUF is sized well above the worst-case tokenize/done JSON line (max-ctx 320 + n_gen 256)
     * so a valid event is never split into invalid JSON the browser would drop. */
    char line[LINE_BUF];
    int saw_terminal = 0;
    int client_gone  = 0;   /* set when an SSE write times out/fails; we then drain the worker only */
    for (;;) {
        ssize_t r = read_line_fd(w->from_worker, line, sizeof line);
        if (r <= 0) {
            /* worker EOF/exit mid-stream: emit an honest terminal error so the UI never hangs. */
            if (!saw_terminal && !client_gone)
                sse_write_event(fd, "0", "error", "{\"_ev\":\"error\",\"where\":\"driver\",\"message\":\"worker stream ended\",\"fatal\":true}\n");
            break;
        }
        char ev[64], seq[32];
        json_str_field(line, "_ev", ev, sizeof ev);
        /* TELEMETRY PURITY: only forward lines that are valid telemetry JSON objects carrying a
         * recognized "_ev". Drop anything else (e.g. a stray worker diagnostic) -- log it to our
         * stderr so it is not silently lost, but never leak it onto the SSE wire as "event: message". */
        if (!is_telemetry_event(ev)) {
            fprintf(stderr, "[serve] dropped non-telemetry worker line: %s%s",
                    line, (line[0] && line[strlen(line)-1] == '\n') ? "" : "\n");
            continue;
        }
        json_str_field(line, "seq", seq, sizeof seq);
        /* opportunistically capture the real device name from the first hello for /api/health. */
        if (!strcmp(ev, "hello") && !w->device[0])
            json_str_field(line, "device", w->device, sizeof w->device);
        if (!client_gone && sse_write_event(fd, seq, ev, line) < 0) {
            /* client stopped reading (SO_SNDTIMEO fired) or closed the socket: stop writing to the
             * dead fd, but keep reading the worker stream to its terminal so the persistent worker
             * does not block on a full stdout pipe and stays in sync for the next request. The GPU
             * mutex (released below) is then held for at most the rest of THIS generation. */
            fprintf(stderr, "[serve] client write timed out/closed mid-stream; draining worker to terminal\n");
            client_gone = 1;
        }
        if (!strcmp(ev, "done")) { saw_terminal = 1; break; }
        if (!strcmp(ev, "error")) {
            /* terminal only if fatal:true */
            if (strstr(line, "\"fatal\":true")) { saw_terminal = 1; break; }
        }
    }

    g_busy = 0;
    pthread_mutex_unlock(&g_gpu_lock);
    /* one-shot: close the connection after the stream. */
}

/* ============================ /api/verify ============================ */
/* HONEST + NEVER-HANG + NEVER-FAKE. The deep token-for-token parity re-derivation is the OFFLINE
 * gate (scripts/gpt2_scale.sh, surfaced on page 2): it loads the full fp32 numpy oracle (6.4 GB at
 * XL) and re-runs the entire forward -- emphatically NOT something to do synchronously on an HTTP
 * request (it would load gigabytes + block the single GPU). So the live /api/verify NEVER puts that
 * on the hot path. It returns:
 *   - UNAVAILABLE  (oracle ABSENT): python3+numpy or the oracle script is missing -> the demo's
 *                  Python-free design; point to page 2.
 *   - UNAVAILABLE  (oracle PRESENT but off-hot-path BY DESIGN): the oracle exists, but the deep
 *                  re-check is the offline gate; we do NOT fake a PASS here. The note says so and the
 *                  UI links to page 2's real, committed parity numbers.
 * Either way: a fast, structured, honest JSON -- no weight load, no hang, no fabricated verdict. */
static int file_exists(const char* p) { struct stat st; return stat(p, &st) == 0; }

static int oracle_present(void) {
    if (!g_cfg.oracle_dir[0]) return 0;
    char oracle_py[1200];
    snprintf(oracle_py, sizeof oracle_py, "%s/gpt2_numpy_ref.py", g_cfg.oracle_dir);
    if (!file_exists(oracle_py)) return 0;
    if (system("python3 -c 'import numpy' >/dev/null 2>&1") != 0) return 0;   /* cheap; no weights */
    return 1;
}

static void handle_verify(int fd, const char* body) {
    (void)body;
    if (!oracle_present()) {
        send_status(fd, 200, "OK", "application/json; charset=utf-8",
            "{\"verdict\":\"UNAVAILABLE\",\"argmax_match\":false,\"token_for_token\":false,"
            "\"oracle\":\"numpy fp32 (absent)\","
            "\"note\":\"python/numpy oracle absent -- the live demo path is Python-free by design. "
            "The real token-for-token XL parity is the committed offline gate; see page 2.\"}");
        return;
    }
    /* oracle present, but the deep re-derivation loads the full 6.4 GB XL fp32 oracle + re-runs the
     * forward -- the OFFLINE gate's job, never the hot path. We do NOT fake a PASS. */
    send_status(fd, 200, "OK", "application/json; charset=utf-8",
        "{\"verdict\":\"UNAVAILABLE\",\"argmax_match\":false,\"token_for_token\":false,"
        "\"oracle\":\"numpy fp32 (present, off-hot-path by design)\","
        "\"note\":\"The fp32 numpy oracle exists but re-running it would load the full XL weights "
        "and block the single GPU -- it is the OFFLINE gate (scripts/gpt2_scale.sh), not a live "
        "hot-path check. Not faking a verdict; the real, committed token-for-token XL parity is on page 2.\"}");
}

/* ============================ connection handler ============================ */
static void conn_done(int fd) {
    /* single teardown for every handle_conn exit: close the fd + release the concurrency slot. */
    close(fd);
    pthread_mutex_lock(&g_conns_lock);
    if (g_conns > 0) g_conns--;
    pthread_mutex_unlock(&g_conns_lock);
}

static void* handle_conn(void* arg) {
    int fd = (int)(intptr_t)arg;

    /* per-connection read deadline (DoS / slow-loris): a stalled header or body read returns
     * EAGAIN/EWOULDBLOCK after CONN_RCVTIMEO_SEC instead of pinning this thread forever. read()/
     * read_line_fd() then see r<0 (errno != EINTR) and bail, and conn_done() reclaims the slot. */
    struct timeval rcvto; rcvto.tv_sec = CONN_RCVTIMEO_SEC; rcvto.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &rcvto, sizeof rcvto);
    /* per-connection WRITE deadline (see CONN_SNDTIMEO_SEC): bounds a write() to a client that has
     * stopped reading so it can never pin the single-flight GPU mutex forever. */
    struct timeval sndto; sndto.tv_sec = CONN_SNDTIMEO_SEC; sndto.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &sndto, sizeof sndto);

    /* read the request header (until \r\n\r\n). */
    char hdrbuf[16384]; size_t hlen = 0;
    const char* hdr_end = NULL;
    while (hlen + 1 < sizeof hdrbuf) {
        ssize_t r = read(fd, hdrbuf + hlen, sizeof hdrbuf - 1 - hlen);
        if (r <= 0) { if (r < 0 && errno == EINTR) continue; break; }
        hlen += (size_t)r;
        hdrbuf[hlen] = 0;
        hdr_end = strstr(hdrbuf, "\r\n\r\n");
        if (hdr_end) { hdr_end += 4; break; }
    }
    if (!hdr_end) { conn_done(fd); return NULL; }

    /* method + path */
    char method[16] = {0}, path[2048] = {0};
    sscanf(hdrbuf, "%15s %2047s", method, path);
    /* strip query string for routing/static (but keep it for the SSE detail param if needed) */
    char pathq[2048]; snprintf(pathq, sizeof pathq, "%s", path);
    char* q = strchr(path, '?'); if (q) *q = 0;

    /* content-length */
    int content_len = 0;
    { const char* cl = strcasestr(hdrbuf, "Content-Length:");
      if (cl) content_len = atoi(cl + 15); }

    if (!strcmp(method, "GET")) {
        if (!strcmp(path, "/api/health")) handle_health(fd);
        else serve_static(fd, path);
    } else if (!strcmp(method, "POST")) {
        if (content_len > MAX_BODY) {   /* reject oversized bodies before allocating */
            send_status(fd, 413, "Payload Too Large", "application/json; charset=utf-8",
                        "{\"error\":\"request body too large\"}");
            conn_done(fd); return NULL;
        }
        char* body = read_body(fd, hdr_end, hdrbuf, hlen, content_len);
        if (!strcmp(path, "/api/generate")) handle_generate(fd, body ? body : "");
        else if (!strcmp(path, "/api/verify")) handle_verify(fd, body ? body : "");
        else send_status(fd, 404, "Not Found", "application/json; charset=utf-8", "{\"error\":\"not found\"}");
        free(body);
    } else if (!strcmp(method, "OPTIONS")) {
        /* CORS preflight */
        const char* h = "HTTP/1.1 204 No Content\r\nAccess-Control-Allow-Origin: *\r\n"
                        "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
                        "Access-Control-Allow-Headers: Content-Type\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        write_all(fd, h, strlen(h));
    } else {
        send_status(fd, 405, "Method Not Allowed", "text/plain", "method not allowed\n");
    }
    (void)pathq;
    conn_done(fd);
    return NULL;
}

/* ============================ worker spawn + readiness ============================ */
/* read worker i's stderr until GPT2_SERVE_READY (or EOF), logging lines to our stderr. */
typedef struct { int fd; int idx; } DrainArg;
static void* drain_worker_stderr(void* arg) {
    DrainArg* da = (DrainArg*)arg;
    int fd = da->fd, idx = da->idx;
    free(da);
    char line[4096];
    for (;;) {
        ssize_t r = read_line_fd(fd, line, sizeof line);
        if (r <= 0) break;
        fprintf(stderr, "[worker:%s] %s", cfg_model_name(idx), line);
        if (strstr(line, "GPT2_SERVE_READY")) { g_workers[idx].ready = 1; }
    }
    close(fd);
    return NULL;
}

/* spawn worker `idx` (0 = the primary model; 1 = the optional --*2 model). */
static void spawn_worker(int idx) {
    int in_pipe[2], out_pipe[2], err_pipe[2];
    if (pipe(in_pipe) || pipe(out_pipe) || pipe(err_pipe)) die("pipe");

    const char* w_ptx     = (idx == 2) ? g_cfg.ptx3     : idx ? g_cfg.ptx2     : g_cfg.ptx;
    const char* w_weights = (idx == 2) ? g_cfg.weights3 : idx ? g_cfg.weights2 : g_cfg.weights;
    const char* w_vocab   = (idx == 2) ? g_cfg.vocab3   : idx ? g_cfg.vocab2   : g_cfg.vocab;
    const char* w_merges  = (idx == 2) ? g_cfg.merges3  : idx ? g_cfg.merges2  : g_cfg.merges;
    int w_specials = (idx == 2) ? g_cfg.specials3 : (idx == 1) ? g_cfg.specials2 : 0;
    int w_eos      = (idx == 2) ? g_cfg.eos3      : (idx == 1) ? g_cfg.eos2      : -1;
    int w_kv       = (idx == 2) ? g_cfg.kv3       : (idx == 1) ? g_cfg.kv2       : 0;

    pid_t pid = fork();
    if (pid < 0) die("fork");
    if (pid == 0) {
        /* child: wire stdin<-in_pipe[0], stdout->out_pipe[1], stderr->err_pipe[1] */
        dup2(in_pipe[0], 0);
        dup2(out_pipe[1], 1);
        dup2(err_pipe[1], 2);
        close(in_pipe[0]); close(in_pipe[1]);
        close(out_pipe[0]); close(out_pipe[1]);
        close(err_pipe[0]); close(err_pipe[1]);
        /* per-model chat config travels via env (the worker reads HX_SPECIALS/HX_EOS). */
        if (w_specials) setenv("HX_SPECIALS", "1", 1); else unsetenv("HX_SPECIALS");   /* never inherited from the launch shell */
        if (w_eos >= 0) { char eb[16]; snprintf(eb, sizeof eb, "%d", w_eos); setenv("HX_EOS", eb, 1); } else unsetenv("HX_EOS");
        if (w_kv) { setenv("HX_KV", "1", 1); setenv("HX_RESIDENT", "1", 1); } else { unsetenv("HX_KV"); unsetenv("HX_RESIDENT"); }
        /* exec the worker: gpt2_infer <ptx> <weights> --serve --emit-fd 1 --max-ctx M
         *                  --detail D --vocab v --merges m */
        char maxctx[16]; snprintf(maxctx, sizeof maxctx, "%d", g_cfg.max_ctx);
        char* av[24]; int n = 0;
        av[n++] = g_cfg.worker_bin;
        av[n++] = (char*)w_ptx;
        av[n++] = (char*)w_weights;
        av[n++] = "--serve";
        av[n++] = "--emit-fd"; av[n++] = "1";
        av[n++] = "--max-ctx"; av[n++] = maxctx;
        /* --timing 1: per-layer ms is REAL host wall-clock. Every kernel launch in forward_layer_gpt2
         * goes through LX/LX2, each of which calls cuCtxSynchronize() (gpt2_infer.c:90-91), so the GPU
         * is fully synced at both the first and last launch of a layer. The now_seconds() delta around
         * forward_layer_gpt2() therefore measures genuine per-layer GPU compute time (not launch-enqueue
         * noise). NOT CUDA-event timing -- host-clock around a per-layer barrier. Without this flag the
         * worker reports ms:0.000 for every layer, so a LIVE viewer would see a flat zero chart. */
        av[n++] = "--timing"; av[n++] = "1";
        av[n++] = "--detail";  av[n++] = g_cfg.detail[0] ? g_cfg.detail : (char*)"op";
        if (w_vocab[0] && w_merges[0]) {
            av[n++] = "--vocab";  av[n++] = (char*)w_vocab;
            av[n++] = "--merges"; av[n++] = (char*)w_merges;
        }
        av[n] = NULL;
        execv(g_cfg.worker_bin, av);
        fprintf(stderr, "execv worker '%s' failed: %s\n", g_cfg.worker_bin, strerror(errno));
        _exit(127);
    }
    /* parent */
    close(in_pipe[0]); close(out_pipe[1]); close(err_pipe[1]);
    g_workers[idx].pid         = pid;
    g_workers[idx].to_worker   = in_pipe[1];
    g_workers[idx].from_worker = out_pipe[0];
    pthread_mutex_init(&g_workers[idx].lock, NULL);

    /* drain stderr in the background; it sets g_workers[idx].ready when GPT2_SERVE_READY appears. */
    DrainArg* da = (DrainArg*)malloc(sizeof *da);
    da->fd = err_pipe[0]; da->idx = idx;
    pthread_t th;
    pthread_create(&th, NULL, drain_worker_stderr, da);
    pthread_detach(th);
}

/* ============================ listen + accept ============================ */
static int listen_local(int port) {
    int s = socket(AF_INET, SOCK_STREAM, 0);
    if (s < 0) die("socket");
    int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
    struct sockaddr_in addr; memset(&addr, 0, sizeof addr);
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);   /* 127.0.0.1 ONLY */
    addr.sin_port = htons((uint16_t)port);
    if (bind(s, (struct sockaddr*)&addr, sizeof addr) < 0) die("bind");
    if (listen(s, 64) < 0) die("listen");
    return s;
}

/* ============================ cli ============================ */
static void usage(const char* a0) {
    fprintf(stderr,
      "usage: %s --port P --root <abs demo dir> --ptx <ptx> --weights <w> --worker-bin <gpt2_infer>\n"
      "          [--vocab v.json --merges m.txt] [--max-ctx 320] [--detail op|layer]\n"
      "          [--oracle <dir with gpt2_numpy_ref.py>] [--model gpt2-xl]\n", a0);
}

int main(int argc, char** argv) {
    signal(SIGPIPE, SIG_IGN);                       /* a closed browser socket must not kill us */
    memset(&g_cfg, 0, sizeof g_cfg);
    g_cfg.port = 8848; g_cfg.max_ctx = 320;
    g_cfg.eos2 = -1; g_cfg.eos3 = -1;
    snprintf(g_cfg.detail, sizeof g_cfg.detail, "op");
    snprintf(g_cfg.model, sizeof g_cfg.model, "gpt2-xl");

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--port")      && i+1 < argc) g_cfg.port = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--root")      && i+1 < argc) snprintf(g_cfg.root, sizeof g_cfg.root, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--ptx")       && i+1 < argc) snprintf(g_cfg.ptx, sizeof g_cfg.ptx, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--weights")   && i+1 < argc) snprintf(g_cfg.weights, sizeof g_cfg.weights, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--vocab")     && i+1 < argc) snprintf(g_cfg.vocab, sizeof g_cfg.vocab, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--merges")    && i+1 < argc) snprintf(g_cfg.merges, sizeof g_cfg.merges, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--worker-bin")&& i+1 < argc) snprintf(g_cfg.worker_bin, sizeof g_cfg.worker_bin, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--max-ctx")   && i+1 < argc) g_cfg.max_ctx = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--detail")    && i+1 < argc) snprintf(g_cfg.detail, sizeof g_cfg.detail, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--oracle")    && i+1 < argc) snprintf(g_cfg.oracle_dir, sizeof g_cfg.oracle_dir, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--model")     && i+1 < argc) snprintf(g_cfg.model, sizeof g_cfg.model, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--model2")    && i+1 < argc) snprintf(g_cfg.model2, sizeof g_cfg.model2, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--ptx2")      && i+1 < argc) snprintf(g_cfg.ptx2, sizeof g_cfg.ptx2, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--weights2")  && i+1 < argc) snprintf(g_cfg.weights2, sizeof g_cfg.weights2, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--vocab2")    && i+1 < argc) snprintf(g_cfg.vocab2, sizeof g_cfg.vocab2, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--merges2")   && i+1 < argc) snprintf(g_cfg.merges2, sizeof g_cfg.merges2, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--specials2") && i+1 < argc) g_cfg.specials2 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--eos2")      && i+1 < argc) g_cfg.eos2 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--model3")    && i+1 < argc) snprintf(g_cfg.model3, sizeof g_cfg.model3, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--ptx3")      && i+1 < argc) snprintf(g_cfg.ptx3, sizeof g_cfg.ptx3, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--weights3")  && i+1 < argc) snprintf(g_cfg.weights3, sizeof g_cfg.weights3, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--vocab3")    && i+1 < argc) snprintf(g_cfg.vocab3, sizeof g_cfg.vocab3, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--merges3")   && i+1 < argc) snprintf(g_cfg.merges3, sizeof g_cfg.merges3, "%s", argv[++i]);
        else if (!strcmp(argv[i], "--specials3") && i+1 < argc) g_cfg.specials3 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--eos3")      && i+1 < argc) g_cfg.eos3 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv2")       && i+1 < argc) g_cfg.kv2 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--kv3")       && i+1 < argc) g_cfg.kv3 = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-h") || !strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
    }
    if (!g_cfg.root[0] || !g_cfg.ptx[0] || !g_cfg.weights[0] || !g_cfg.worker_bin[0]) {
        usage(argv[0]); return 2;
    }
    /* the second model needs all of model2/ptx2/weights2 (vocab2/merges2 too for chat). */
    if (g_cfg.model2[0]) {
        if (!g_cfg.ptx2[0] || !g_cfg.weights2[0]) {
            fprintf(stderr, "--model2 requires --ptx2 and --weights2\n"); return 2;
        }
        g_nmodels = 2;
    }
    if (g_cfg.model3[0]) {
        if (g_nmodels < 2) { fprintf(stderr, "--model3 requires --model2\n"); return 2; }
        if (!g_cfg.ptx3[0] || !g_cfg.weights3[0]) {
            fprintf(stderr, "--model3 requires --ptx3 and --weights3\n"); return 2;
        }
        g_nmodels = 3;
    }

    for (int m = 0; m < g_nmodels; m++) spawn_worker(m);
    int s = listen_local(g_cfg.port);
    fprintf(stderr, "gpt2_serve_http: listening on http://127.0.0.1:%d/  (root=%s, worker pid=%d%s)\n",
            g_cfg.port, g_cfg.root, (int)g_worker.pid,
            g_nmodels > 1 ? " + a second-model worker" : "");

    for (;;) {
        int fd = accept(s, NULL, NULL);
        if (fd < 0) { if (errno == EINTR) continue; break; }
        /* concurrent-connection cap (DoS): refuse beyond MAX_CONN in-flight handler threads.
         * Reserve the slot BEFORE spawning so the count can't be raced past the cap; release it in
         * conn_done() (or here on a failed pthread_create). The kernel keeps the listen() backlog,
         * so a legitimate client that hits the cap simply retries. */
        pthread_mutex_lock(&g_conns_lock);
        int over = (g_conns >= MAX_CONN);
        if (!over) g_conns++;
        pthread_mutex_unlock(&g_conns_lock);
        if (over) { close(fd); continue; }      /* at capacity: drop this connection */
        pthread_t th;
        if (pthread_create(&th, NULL, handle_conn, (void*)(intptr_t)fd) != 0) {
            /* spawn failed: release the slot we reserved and drop the connection */
            pthread_mutex_lock(&g_conns_lock);
            if (g_conns > 0) g_conns--;
            pthread_mutex_unlock(&g_conns_lock);
            close(fd); continue;
        }
        pthread_detach(th);
    }
    return 0;
}
