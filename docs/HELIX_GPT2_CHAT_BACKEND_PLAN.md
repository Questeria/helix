# Helix × GPT-2-XL — Interactive Chat Demo: BACKEND BUILD PLAN

**Status: DESIGN ONLY. Build/run NOTHING from this document yet.** This is the implementable
blueprint for a *later* build pass. It commits **no `.c`/`.h` files** (skeletons are inline fenced
code only). It does **not** touch `helixc/runtime/gpt2_infer.c`, `kovc.hx`, `lexer.hx`, `parser.hx`,
`train_transformer.c`, `seed.c`, `reproduce_trust.sh`, the trust-inventory docs, or anything under
`helix-llm/`. A separate agent concurrently owns `helix-llm/tools/*`, `scripts/gpt2_pyfree.sh`, and
the trust-fence docs — those are out of scope here and must be coordinated, not duplicated (risk R10).

**Companion design contract:** the SSE/telemetry wire format and page-1 layout are frozen by the
chat-demo DESIGN CONTRACT (telemetry_events / sse_protocol / page1_layout / panels / framing_rules).
This document is the **backend** half: the `--serve` mode of `gpt2_infer.c`, the C HTTP+SSE server,
their wiring, and the fail-closed integration gates. It implements that contract exactly.

**The one-line frame (inherited from the runbook):** the product is a **bring-your-weights verified
execution layer**, not a fast inference engine or a smart assistant. The chat page is a GPT-2-XL
*completion playground*; the hero is *verified compute*, not model intelligence. Lead with trust,
never speed.

---

## 0. Ground truth this plan is built on (verified against the live tree, 2026-06)

All design below reuses **existing, verified** machinery. The load-bearing facts:

### 0.1 `helixc/runtime/gpt2_infer.c` (the file `--serve` extends) — verified structure

- **Dimension-generic, env-driven.** `main()` reads `HX_NL/HX_D/HX_HEADS/HX_V/HX_CTX/HX_DFF/HX_SPAD/HX_DBG`
  (lines 602–610), then `DH = DM/NH`, `ATTN_SCALE = 1/sqrt(DH)`. Committed defaults are GPT-2 124M
  (`NL=12, DM=768, NH=12, NV=50257, NC=1024, DFF=3072`, line 43). XL is purely env-driven exactly as
  `scripts/gpt2_scale.sh` already drives it (`HX_NL=48 HX_D=1600 HX_HEADS=25 HX_V=50257 HX_CTX=1024 HX_DFF=6400`).
  **No new dim logic is needed for serve mode.**
- **Existing setup primitives (reused verbatim):**
  - `device_init(ptx_path, wpath)` (line 383): `fopen`+read the PTX, `cuInit`, `cuDeviceGet`,
    `cuDeviceGetName(g_gpu,256,dev)` → the device-name string, `cuCtxCreate`, `cuModuleLoadData`,
    then the **8** `cuModuleGetFunction` calls (lines 392–402), then `load_gpt2_weights` (mmap, header
    validate against the env dims, `PROT_READ`).
  - `alloc_buffers(Smax)` (line 408): allocates the reused per-layer weight device buffers + all
    activation buffers sized for `Smax`; uploads `ATTN_SCALE` to `d_scale`; **`cuMemsetD8(d_ctx, 0, …)`
    once** (line 435).
  - `setup_head(Smax)` (line 442): `NVpad = ((NV+63)/64)*64` (50257→50304), uploads `ln_f` γ/β, the
    zero-padded tied head `d_wte_pad`, allocates `d_logits`.
- **Existing forward (reused verbatim, ZERO arithmetic change):**
  - `forward_full(ids, T, out_last_logits)` (line 337): sets `S = ((T+63)/64)*64`, `Spad = S`,
    `embed_gather(ids,T,S)`, the `for L in 0..NL { upload_layer(L); forward_layer_gpt2(); }` loop,
    final `ln_eps(d_lnfg/d_lnfb)` (= `ln_f`), tied-head `mm_ABt(d_xn, d_wte_pad, d_logits, S, DM, NVpad)`,
    then a D2H of **only** the last real-token row's `NV` logits.
  - `embed_gather(ids, T, S)` (line 319): host `wte[id]+wpe[s]` gather into a calloc'd host buffer,
    one `cuMemcpyHtoD` into `d_x`. Pad rows are zero (calloc).
  - `upload_layer(L)` (line 226): the 12 per-layer `cuMemcpyHtoD` streams into the **reused** layer-weight
    buffers — this is the existing **bounded-device-residency** design (~one layer + tied head resident).
  - `forward_layer_gpt2()` (line 276): the real per-op kernel sequence (see §1.4 for the exact op map),
    including the `for h in 0..NH` head loop (pack Q/K/V → `mm_ABt` scores → `scale_rt` → `softmax_causal`
    → `mm_AB @V` → scatter). Every `cuLaunchKernel` is **immediately followed by `cuCtxSynchronize`**
    (the `LX`/`LX2` macros, lines 65–66) — this is what makes per-op telemetry truthful & real-time at
    zero added synchronization (contract: ORDERING/FLUSH).
  - `argmax_row(v, n)` (line 351): host argmax over the last logit row; returns id and the winning
    value is `v[a]` (already in host memory — the contract's `token.logit`).
- **The 8 real kovc kernel CUfunction names** (the only legal `op.kernel`/`head.kernel` values), from
  `device_init`:
  `tiled_matmul`, `tiled_matmul_abt`, `gpu_layernorm_fwd_eps`, `gpu_softmax_causal`,
  `gpu_add_bias_rowbcast`, `gpu_gelu_stable`, `vector_add`, `gpu_scale_rt`.
- **Existing modes:** `--block0`, `--logits`, `--generate` (dispatched in `main()` by `argv[3]`).
  `--serve` is a **new fourth branch**; the existing three are untouched.

### 0.2 `helixc/runtime/gpt2_tok.c` (in-process tokenizer for serve mode) — verified entrypoints

The C, Python-free byte-level BPE tokenizer. In-process linkable functions (all currently `static`,
so a small refactor exposes them — see §1.6):
- `build_byte_unicode()` (one-time), `load_vocab(const char* vocab_path)`, `load_merges(const char* merges_path)`.
- `int* encode_bytes(const unsigned char* text, size_t n, int* out_n)` → malloc'd id array + count
  (the **real** `tokenize.ids` / `n_prompt`).
- `decode_ids(const int* ids, int n)` — currently writes raw bytes to **stdout**; serve mode needs a
  **string-returning** sibling `decode_to_buf(ids, n, char** out, size_t* outlen)` for the contract's
  per-id `strings[]`, the per-token `token.string`, and the final `done.text` (see §1.6). This is a
  pure-bookkeeping addition: it concatenates `g_id2bytes[id]` for each id (same bytes `decode_ids`
  already emits), **zero arithmetic**.
- Verified pinned behavior (gate anchors): `"The capital of France is"` → ids `464 3139 286 4881 318`
  (gpt2_pyfree.sh T3); hero decode round-trips exactly (T4).

### 0.3 The offline oracle for the G1 parity gate — verified

`scripts/gpt2_scale.sh MODEL=gpt2-xl` is the reference. It mints the PTX **fresh from the 299-byte
raw seed** (`seed.bin` sha `9837db12`), emits the **same 8 forward-only kernels**, builds the
committed dimension-generic `gpt2_infer.c`, generates oracle refs from the fenced numpy oracle
(`helix-llm/tools/gpt2_numpy_ref.py`), and runs `--logits` + `--generate 20` token-for-token. Its XL
green result (runbook §"Scale flex") is the pinned target for G1:
> XL argmax id **262**, max-abs logit diff **4.4e-05**, **25/25 ids**, output:
> *"The capital of France is the city of Paris. It is the capital of France and the largest city in France. It is"*

### 0.4 The pinned trust anchors (verified, from `demo/dashboard.html` EMBEDDED_REPORT + runbook)

These are the **real** values the `hello`/`trust-strip` events carry (`fixpoint_sha`, `seed_sha`):
- **seed.bin** sha `9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb` (62,467 B).
- **kovc self-host fixpoint** (K2==K3==K4) `0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f` (698,392 B).
- **gcc-DDC** `84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba`.
- Device/precision: **RTX 3070-class, sm_86, 8 GB, fp32, forward-only** (runbook §0).

### 0.5 The fence count — VERIFIED LIVE (correcting the contract's parenthetical)

The chat DESIGN CONTRACT prompt said *"each new .c/.h bumps the count from 26"*. **That figure is
stale.** Verified against the live tree at HEAD (`86abedf`):

```
$ git ls-files "*.c" "*.h" | wc -l
28
```

The **real committed `.c`/`.h` count is 28** (6 in `helixc/runtime/` + 22 in `stage0/`), matching
`docs/TRUSTED_C_INVENTORY.md` §0 ("at HEAD the committed C/H is 28 files / 18 131 LOC"). The from-raw
Category-A ladder is **24 files / 15 605 LOC UNCHANGED** (Category A's 22 from-raw + nothing else moved;
the inventory's "24" tally = the 22 stage0 files counted in the from-raw ladder plus the trust-root
accounting it uses — see that doc). The fence implication of this plan is therefore stated against the
**true base of 28** in §5/G6, not 26.

---

## 1. (1) Persistent `--serve` mode on `gpt2_infer.c`

**Goal.** Add an additive, forward-only `--serve` branch that performs the expensive setup
(`device_init` → mint/load PTX + 8 kernel handles + weight mmap; `alloc_buffers`; `setup_head`)
**ONCE**, then loops on stdin request frames: read prompt → tokenize in-process (gpt2_tok) → for each
of `n_gen` steps run `forward_full` with telemetry emit-hooks firing at the real points → host argmax →
emit `token` → repeat → emit `done`. Per-layer streaming (`upload_layer`) gives bounded device
residency for free. **Strictly additive; forward-only; must NOT alter the self-host fixpoint
(`0992dddd`) or existing 124M/scale/`--block0`/`--logits`/`--generate` behavior.**

### 1.1 Why this does not perturb the fixpoint (the hard constraint)

`gpt2_infer.c` is a **downstream CUDA-FFI launcher** outside the self-host fixpoint
(`scripts/gate_kovc.sh` never compiles it; `docs/TRUSTED_C_INVENTORY.md` §2a). It does not touch
`kovc.hx`/`lexer.hx`/`parser.hx`/`seed.c`/`train_transformer.c` or anything mirrored into the from-raw
build. The `--serve` patch is **additive C inside `gpt2_infer.c`'s `main()` + a new emit module**, so
the `0992dddd` self-host fixpoint and the `84363adb` gcc-DDC are structurally unaffected. **G4 must
byte-match those anchors before any merge** (§4). NOTE: this design *does not* edit `gpt2_infer.c`; the
owning agent applies the patch later.

### 1.2 CLI surface (additive — new 4th mode)

```
gpt2_infer <ptx> <weights> --serve [--emit-fd N] [--detail op|layer]
                                    [--vocab vocab.json --merges merges.txt]
                                    [--max-ctx M] [--timing 0|1]
```
- `--emit-fd N` (default `1` / stdout): the fd the one-line-JSON telemetry is written to.
- `--vocab/--merges`: when given, serve mode tokenizes **in-process** (preferred: true single-process,
  Python-free, fork-free). When absent, serve mode accepts **pre-tokenized id lists** on the wire
  (the HTTP server then tokenizes by shelling the standalone `gpt2_tok` binary — fallback path).
- `--max-ctx M`: the serve-session max sequence length (prompt cap + max `n_gen`); buffers are sized
  **once** for `Smax = ((M+63)/64)*64`. Default `M = 320` (covers a generous prompt + `n_gen ≤ 256`,
  but the server clamps `n_gen` to `1..256` per the contract; size to whatever cap the operator sets).
- `--timing 1`: enable real CUDA-event/host-clock per-layer timing for `layer_end.ms`; `0` (default)
  emits `ms:0` (the contract permits `0` = "untimed", **never fabricated**).

Env dims arrive exactly as today (`HX_*`), so XL = `HX_NL=48 …` in front of the binary; **no new dim
parsing**.

### 1.3 The serve loop (new code, all under `if (mode=="--serve")`)

Pseudocode for the new branch in `main()` (skeleton; not committed):

```c
/* --- NEW: --serve branch (additive; the existing 3 modes are untouched) --- */
else if (strcmp(mode, "--serve") == 0) {
    int   emit_fd  = opt_int("--emit-fd", 1);
    int   max_ctx  = opt_int("--max-ctx", 320);
    int   timing   = opt_int("--timing", 0);
    const char* vocab  = opt_str("--vocab",  NULL);
    const char* merges = opt_str("--merges", NULL);
    emit_init(emit_fd, /*detail=*/opt_str("--detail","op"));   /* §1.5 */

    if (device_init(ptx_path, wpath)) { emit_error("load","device_init failed",1); return 2; }
    int Smax = ((max_ctx + 63) / 64) * 64; if (Smax < 64) Smax = 64;
    if (alloc_buffers(Smax)) { emit_error("load","alloc_buffers failed",1); return 2; }
    if (setup_head(Smax))    { emit_error("load","setup_head failed",1);    return 2; }

    int in_proc_tok = (vocab && merges);
    if (in_proc_tok) { build_byte_unicode(); load_vocab(vocab); load_merges(merges); }

    /* buffers + PTX + weights are warm: announce readiness ONCE. */
    print_ready_line();   /* "GPT2_SERVE_READY\n" on stderr/log for /api/health */

    char* line = NULL; size_t cap = 0;
    while (getline(&line, &cap, stdin) > 0) {           /* one request frame per line */
        ServeReq req;
        if (parse_req_json(line, &req)) { emit_error("load","bad request json",0); continue; }
        if (req.is_quit) break;                          /* {"cmd":"quit"} -> teardown */

        /* tokenize (in-process gpt2_tok) OR accept req.ids[] from the wire */
        int  T0; int* ids;
        if (in_proc_tok) ids = encode_bytes((const unsigned char*)req.prompt, req.prompt_len, &T0);
        else { ids = req.ids; T0 = req.n_ids; }
        if (T0 <= 0) { emit_error("tokenize","empty/failed tokenization",0); free(ids); continue; }

        int Ngen = clampi(req.n_gen, 1, 256);
        if (((T0 + Ngen + 63)/64)*64 > Smax) {           /* honest bound, not a hang */
            emit_error("forward","context exceeds --max-ctx; raise --max-ctx",0);
            if (in_proc_tok) free(ids); continue;
        }

        /* hello carries the static trust/model header (real values) -- ONCE per request */
        emit_hello(/*ptx_bytes=*/g_ptx_len, /*device=*/g_gpu, /*dims from HX_* / header*/);

        /* tokenize event: real ids + decoded display pieces + n_prompt + s_pad */
        emit_tokenize(ids, T0, /*s_pad=*/((T0+63)/64)*64);

        int  T = T0;
        float* logits = (float*)malloc((size_t)NV * sizeof(float));
        int nonfinite = 0;
        double t0 = now_seconds();
        int* gen_ids = (int*)malloc((size_t)Ngen * sizeof(int));
        for (int step = 0; step < Ngen; step++) {
            g_emit_step = step;                          /* the emit hooks read this */
            emit_forward_begin(step, /*context_len=*/T, /*s_pad=*/((T+63)/64)*64, NL);
            forward_full(ids, T, logits);                /* UNCHANGED arithmetic; hooks fire inside */
            for (int i = 0; i < NV; i++) if (!isfinite(logits[i])) { nonfinite = 1; break; }
            int nxt = argmax_row(logits, NV);
            emit_token(step, nxt, /*string=*/decode_one(nxt), /*logit=*/logits[nxt],
                       /*context_len=*/T + 1);
            gen_ids[step] = nxt;
            ids[T++] = nxt;
        }
        double secs = now_seconds() - t0;
        emit_done(T0, Ngen, T, secs, (double)Ngen/secs,
                  /*text=*/decode_range(gen_ids, Ngen), gen_ids, Ngen, nonfinite);

        free(logits); free(gen_ids);
        if (in_proc_tok) free(ids);
        emit_flush();                                    /* ensure the worker fd is drained */
    }
    /* teardown (EOF or {"cmd":"quit"}) */
    free(line);
    free(g_ptx);
    if (g_map && g_map != MAP_FAILED) munmap(g_map, g_maplen);
    if (g_fd >= 0) close(g_fd);
    cuModuleUnload(g_mod); cuCtxDestroy(ctx);
    return 0;
}
```

**Buffer reuse across requests.** `alloc_buffers`/`setup_head` already design the layer-weight buffers
and activation buffers for **reuse across layers and steps**; serve mode extends that reuse **across
requests** with no re-alloc. `embed_gather` + the layer passes fully overwrite `d_x/d_qkv/d_xn/…`
every request, so no stale activation leaks. **One subtlety (see §1.7):** `d_ctx` is `cuMemsetD8`-zeroed
**once** at alloc and thereafter overwritten by `scatter_head` per layer — a per-request safety memset
of `d_ctx` is added under serve (no-arithmetic, see §1.7).

### 1.4 The emit hooks inside `forward_full` / `forward_layer_gpt2` (the real points)

These are **printf-to-an-fd side effects only** — they read values already in host scope (`step`, `L`,
literal kernel-name strings, the already-copied argmax logit). They do **not** read device memory in any
new way, add any sync, change any kernel arg, or mutate any buffer. The existing `LX`/`LX2` macros
already `cuCtxSynchronize` after every launch, so emitting *after* the op is truthful real-time cadence.

Mapping of each contract `telemetry_event` to its **real** emit site (compiled in only when serve mode
is active — guarded by a global `g_serve` flag set in the `--serve` branch, so the other 3 modes emit
nothing and stay byte-identical in behavior):

| Event | Real site in `gpt2_infer.c` | Notes |
|---|---|---|
| `hello` | serve loop, before tokenize | static header: dims from `HX_*`/HXGW header; `device`=`g_gpu` (`cuDeviceGetName`); `sm='sm_86'`; `ptx_bytes`=`g_ptx` length captured in `device_init`; `kernels`=the 8 literal `cuModuleGetFunction` names; `seed_sha`/`fixpoint_sha`=pinned constants (§0.4); `precision='fp32'`, `build='forward-only'`, `mode='serve'`. |
| `tokenize` | after `encode_bytes`, before first forward | `ids`/`strings` real from gpt2_tok; `n_prompt=T0`; `s_pad=((T0+63)/64)*64`. |
| `forward_begin` | top of each `forward_full` call (serve loop) | `step`, `context_len=T`, `s_pad=((T+63)/64)*64`, `n_layers=NL` (48 for XL). |
| `embed` | inside `forward_full`, right after `embed_gather` returns | `step`, `t=T`, `d_model=DM`. One per token-step. |
| `layer_begin` | top of the `for L` body in `forward_full`, **after** `upload_layer(L)`, **before** `forward_layer_gpt2()` | `step`, `idx=L`, `total=NL`. Primary heartbeat. Per-layer, never per-row. |
| `op` | inside `forward_layer_gpt2`, around each real `cuLaunchKernel`, in execution order | `step`, `layer`, `seq`, `kernel` (literal CUfunction), `phase`, `label`. The 12 per-head launches are **collapsed into 3 aggregate attention ops** (`attn_scores`/`attn_softmax`/`attn_av`) emitted **once after the head loop** with the dominating kernel; the head loop itself emits **no** per-head events. ~16 ops/layer. NO `op` that does not wrap a real launch. |
| `layer_end` | bottom of the `for L` body, after `forward_layer_gpt2()` returns + `cuCtxSynchronize` drained | `step`, `idx=L`, `ms` (real CUDA-event/host-clock duration when `--timing 1`, else `0`), `ops` emitted this layer. |
| `head` (×2) | in `forward_full` after the 48 layers: around the final `ln_eps`(`ln_f`) and around the tied-head `mm_ABt` | `label='ln_f'`/`'lm_head'`; `kernel='gpu_layernorm_fwd_eps'`/`'tiled_matmul_abt'`. |
| `token` | serve loop, after `argmax_row`, before append | `step`, `id`, `string` (gpt2_tok decode of the one id), `logit=logits[id]`, `context_len=T+1`. Exactly `Ngen` per reply. |
| `done` | serve loop, after the gen loop | real `seconds`/`tok_per_s` (clock around the loop), `text` (decode of `gen_ids`), `gen_ids`, `nonfinite`. Terminal. |
| `error` | any real failure path (CK/`check()` CUDA error, OOM, tokenizer fail, load mismatch) | `where`∈{tokenize,forward,oom,driver,load}, `message`, `fatal`. `fatal=true` ⇒ stream closes. |

**The exact per-layer `op` sequence** (the real lines of `forward_layer_gpt2`, lines 276–313), so the
frontend ticker and the mock match the real worker structure (this is the literal label→kernel map):

| seq | label | kernel | phase | real line |
|---:|---|---|---|---|
| 0 | `ln_1` | `gpu_layernorm_fwd_eps` | attn | `ln_eps(d_x,d_xn,ln1g,ln1b)` |
| 1 | `qkv_gemm` | `tiled_matmul` | attn | `mm_AB(d_xn,d_attW,d_qkv,…,3*DM)` |
| 2 | `qkv_bias` | `gpu_add_bias_rowbcast` | attn | `add_bias(d_qkv,d_attb,…)` |
| 3 | `attn_scores` | `tiled_matmul_abt` | attn | **aggregate** of the per-head `mm_ABt` (Q@Kᵀ) over the head loop |
| 4 | `attn_scale` | `gpu_scale_rt` | attn | **aggregate** of per-head `scale_rt(*0.125)` |
| 5 | `attn_softmax` | `gpu_softmax_causal` | attn | **aggregate** of per-head `softmax_causal` |
| 6 | `attn_av` | `tiled_matmul` | attn | **aggregate** of per-head `mm_AB(attnw,V)` |
| 7 | `attn_proj_gemm` | `tiled_matmul` | attn | `mm_AB(d_ctx,d_prjW,d_proj,…)` |
| 8 | `attn_proj_bias` | `gpu_add_bias_rowbcast` | attn | `add_bias(d_proj,d_prjb,…)` |
| 9 | `attn_residual` | `vector_add` | attn | `vadd(d_x,d_proj,d_x,…)` |
| 10 | `ln_2` | `gpu_layernorm_fwd_eps` | mlp | `ln_eps(d_x,d_xn2,ln2g,ln2b)` |
| 11 | `fc_gemm` | `tiled_matmul` | mlp | `mm_AB(d_xn2,d_fcW,d_mlp1,…,DFF)` |
| 12 | `fc_bias` | `gpu_add_bias_rowbcast` | mlp | `add_bias(d_mlp1,d_fcb,…)` |
| 13 | `gelu` | `gpu_gelu_stable` | mlp | `gelu(d_mlp1,d_mlp1g,…)` |
| 14 | `proj2_gemm` | `tiled_matmul` | mlp | `mm_AB(d_mlp1g,d_pjW,d_mlp2,…,DM)` |
| 15 | `proj2_bias` | `gpu_add_bias_rowbcast` | mlp | `add_bias(d_mlp2,d_pjb,…)` |
| 16 | `mlp_residual` | `vector_add` | mlp | `vadd(d_x,d_mlp2,d_x,…)` |

That is a fixed **17 `op` events/layer** (seq 0–16), so XL = `48 × 17 = 816` op events + 2 `head` +
1 `embed` + 1 `forward_begin` + (per-layer) 48 `layer_begin`/`layer_end` per token-step. The
aggregate attention ops are reported with the dominating kernel and an `agg` count in the JSON so the
frontend can show "×25 launches" honestly without 12 events/layer (contract: NON-FLOOD).

**Collapsing detail without lying.** `--detail layer` (server query `?detail=layer`) drops the 17 `op`
events, keeping `layer_begin`/`layer_end`/`token` only — a real subset, never synthetic padding.

### 1.5 The emit module (new, tiny — `emit_*()` over an fd)

A small new translation unit *inside* `gpt2_infer.c` (or a sibling `.h` the patch `#include`s — but per
the "no separate .c/.h committed now" rule, the later patch keeps it **inside `gpt2_infer.c`** so the
fence count does not change for the `--serve` work; see §5). Design:

- A single `static int g_emit_fd; static int g_serve = 0; static long g_seq = 0;` plus the JSON writers.
- Each writer builds a **compact one-line JSON object** ending in `\n` and `write()`s it to `g_emit_fd`,
  then (when `g_emit_fd` is a pipe) relies on the OS; the HTTP server does the socket flush. The worker
  has **zero HTTP knowledge** — emit = newline-JSON on a fd (contract serve_mode_design §4).
- Every object includes the `seq` field (the contract's monotonically increasing integer; the server
  copies it to the SSE `id:` line). The writers also stamp an `_ev` field = the event name so the HTTP
  server can set `event:` without inferring (contract server_design step 3 mentions `_ev`).
- JSON-escaping: a minimal escaper for the only strings that can contain control bytes —
  `tokenize.strings[]`, `token.string`, `done.text` (GPT-2 byte pieces can include spaces shown as the
  display glyph and newlines, which must be `\n`-escaped so each event stays one line). This is the
  same byte-bookkeeping spirit as gpt2_tok.c; **zero arithmetic on the trust path.**

Skeleton (illustrative; not committed):

```c
static int  g_emit_fd = 1; static int g_serve = 0; static long g_seq = 0;
static int  g_emit_step = 0; static int g_layer_ops = 0;
static void emit_init(int fd, const char* detail){ g_emit_fd=fd; g_serve=1; /* parse detail */ }
static void emit_raw(const char* json){ /* json has trailing '\n' */ (void)!write(g_emit_fd,json,strlen(json)); }
static void jesc(const char* s, int n, char* out, int* k); /* \" \\ \n \r \t + \u00XX for <0x20 */

static void emit_op(int layer,int seq,const char* kernel,const char* phase,const char* label){
    if(!g_serve || g_detail<DETAIL_OP) return;
    char b[256];
    snprintf(b,sizeof b,
      "{\"_ev\":\"op\",\"seq\":%ld,\"step\":%d,\"layer\":%d,\"seq_op\":%d,"
      "\"kernel\":\"%s\",\"phase\":\"%s\",\"label\":\"%s\"}\n",
      g_seq++,g_emit_step,layer,seq,kernel,phase,label);
    emit_raw(b); g_layer_ops++;
}
/* emit_hello / emit_tokenize / emit_forward_begin / emit_embed / emit_layer_begin /
   emit_layer_end / emit_head / emit_token / emit_done / emit_error analogous. */
```

The op-emit call site inside `forward_layer_gpt2` wraps each launch, e.g. before `ln_eps(d_x,d_xn,…)`:
`emit_op(L, 0, "gpu_layernorm_fwd_eps", "attn", "ln_1");`. The three aggregate attention ops are
emitted **after** the `for h` loop (which itself stays emit-silent), e.g.
`emit_op(L, 3, "tiled_matmul_abt", "attn", "attn_scores");` etc. `forward_layer_gpt2` needs the layer
index `L` and `step`, which it can read from globals `g_emit_layer`/`g_emit_step` set by the caller — a
pure read, no signature change to the numeric helpers (keeping the arithmetic path byte-identical).

### 1.6 In-process tokenizer linkage (the Python-free, fork-free path)

The preferred serve path tokenizes **in the same process** by calling gpt2_tok's encode/decode. Because
gpt2_tok.c's functions are `static`, the later patch links the tokenizer one of two clean ways
(decided by the owning agent; both keep the fence honest):

- **(a) Single-TU include (preferred, zero new committed file).** The `--serve` patch adds, guarded by
  a build macro, `#include "gpt2_tok_core.inc"` — but since we must not add committed `.c/.h` now and a
  `.inc` is outside the fence anyway, the cleanest *later* move is: the gpt2_tok owner exposes the four
  entrypoints (`build_byte_unicode`, `load_vocab`, `load_merges`, `encode_bytes`) + the new
  `decode_to_buf`/`decode_one`/`decode_range` by dropping `static` on them, and the serve build compiles
  **both** `gpt2_infer.c` and `gpt2_tok.c` into the worker binary (two existing committed files, no new
  file). The `main()` of gpt2_tok.c is `#ifndef GPT2_TOK_LIB`-guarded so it is excluded when linked as
  a library. **This is the recommended approach: no new committed `.c/.h`, reuses two existing files.**
- **(b) Shell-the-binary fallback.** If linking is deferred, the HTTP server tokenizes by exec'ing the
  standalone `gpt2_tok` binary and passes pre-tokenized `ids[]` on the wire (serve mode's non-`--vocab`
  branch). Still Python-free; one extra fork per request (acceptable, but not the hero path).

The new decode-to-buffer helpers are the only tokenizer additions, and they are **pure byte
concatenation** of `g_id2bytes[id]` (the exact bytes `decode_ids` already writes) — zero arithmetic,
no fence/compute-trust impact (gpt2_tok is Category-B host glue by classification).

### 1.7 Forward-only / no-leak invariants (enforced by G1/G2/G3)

- **Numeric path byte-identical.** The forward is the same `forward_full`/`forward_layer_gpt2` code as
  `--generate`; emit hooks are printf-only on host-scope values. Proof obligation = **G1** (served XL ids
  == offline `gpt2_scale.sh` XL ids token-for-token) + **G2** (124M `--logits`/`--generate` unchanged).
- **No persistent device-state leak between requests.** `d_x/d_xn/d_qkv/d_Qh/d_Kh/d_Vh/d_scores/d_attnw/
  d_aoh/d_proj/d_xn2/d_mlp1/d_mlp1g/d_mlp2/d_logits` are fully overwritten by `embed_gather` + the layer
  passes + the head each request. **`d_ctx` is the one buffer `cuMemsetD8`-zeroed only at alloc** (line
  435) and thereafter written by `scatter_head` per layer. Because `scatter_head` writes only the real
  head columns of each row and `mm_AB(d_ctx,…)` reads all `Spad` rows, **pad rows (rows ≥ T) of `d_ctx`
  could in principle carry a previous request's larger-T residue** if a later request has a smaller T.
  **Mitigation (no-arithmetic safety memset):** under `g_serve`, add one
  `cuMemsetD8(d_ctx, 0, (size_t)Spad*DM*sizeof(float))` at the top of `forward_full` (after `Spad=S`).
  This zeroes only pad/unused rows to a deterministic state; it does **not** change any real-token row
  (those are fully overwritten by scatter), so the served ids are unaffected — **G1 confirms this.**
  (If G1 passes byte-for-token *without* the memset, the memset is still kept as a cheap correctness
  guard; it is provably a no-op on the argmaxed last real row.)
- **Weights stay `PROT_READ`.** `load_gpt2_weights` mmaps `MAP_PRIVATE, PROT_READ`; serve mode never
  writes weights. **Forward-only: no training kernels, no weight mutation.**
- **Existing modes untouched.** `--block0`/`--logits`/`--generate` keep their exact `main()` branches;
  committed default dims (124M) unchanged; serve XL is purely env-driven, same as scale today.

### 1.8 Bounded device residency (the existing design, inherited)

`upload_layer(L)` streams one layer's ~weights H2D into the reused layer-weight buffers, so **only one
layer + the tied head are resident** at a time (runbook §"Scale flex": per-layer streaming keeps
device residency low; XL fits the 8 GB box). Serve mode allocates activation buffers **once** for
`Smax` and reuses them across requests — slightly raising steady-state VRAM vs a one-shot run, but the
dominant cost is the tied head (`d_wte_pad` = NVpad×DM = 50304×1600×4 ≈ 322 MB) + one resident layer
+ activations sized for a bounded `Smax`. **Honest OOM handling:** if any `cuMemAlloc`/launch returns
`CUDA_ERROR_OUT_OF_MEMORY`, the `check()`/CK path surfaces an `error{where:"oom"}` event (not a hang)
and, if fatal, closes the stream (risk R8). The documented ~residency claim must be re-verified on the
target 3070 during the build pass.

---

## 2. (2) Dependency-light, NO-PYTHON local HTTP+SSE server

**A small single-file C HTTP+SSE server** (later file: `helixc/runtime/gpt2_serve_http.c`, **NEW —
owned by the later build, NOT created now**). Pure C + POSIX sockets + libc, buildable with the same
`gcc` the rest of the demo uses, **zero third-party deps** — consistent with the Category-B C-host-tool
precedent (`cpu_host.c`, `gpt2_tok.c`, `gpt2_pack.c`). It (a) serves `demo/` static files and (b)
bridges the browser to the persistent `gpt2_infer --serve` worker.

### 2.1 Process topology (on-message push, no Python anywhere)

- The server forks/execs **ONE long-lived child**:
  `gpt2_infer <ptx> <xl.weights> --serve --emit-fd <pipe_w> --vocab … --merges …`, wired over **two
  pipes** (server→worker = `worker_stdin` = request frames; worker→server = `worker_stdout` =
  newline-JSON events). The worker holds XL weights + minted PTX resident; the server is a thin
  socket↔pipe pump.
- **Single-flight.** Because the GPU path is strictly serial (every `cuLaunchKernel` is followed by
  `cuCtxSynchronize`), the server serializes `/api/generate` behind **one mutex/queue**; a second
  concurrent generation gets **409 `{"error":"busy"}`** and the UI disables send while streaming. This
  matches the real hardware constraint, not an artificial limit (gate **G7**).

### 2.2 Endpoints (exactly the contract's sse_protocol)

All endpoints **bind 127.0.0.1 only**; content UTF-8.

| Method + path | Response | Behavior |
|---|---|---|
| `GET /` | 200 `text/html` | serves `demo/index.html` (the NEW chat MAIN page) |
| `GET /dashboard.html` | 200 `text/html` | serves `demo/dashboard.html` **byte-for-byte UNCHANGED** (page 2) |
| `GET /<static>` | 200 (or 404) | serves `demo/*.{html,css,js,svg,json,woff2,png,…}` with correct MIME; **reject `..` / absolute paths** (no traversal) |
| `GET /api/health` | 200 `application/json` | `{"ok":true,"serve":<bool>,"model":"gpt2-xl","ready":<bool>,"device":"…","busy":<bool>}`; cheap liveness+readiness (worker printed `GPT2_SERVE_READY`); the UI polls this to flip LIVE/PREVIEW + enable the composer |
| `POST /api/generate` | 200 `text/event-stream` | the streaming generation request — **the core endpoint** (§2.3) |
| `POST /api/verify` | 200 `application/json` (**non-streaming**) | on-demand parity re-check of the exact ids just produced (§2.4) |

### 2.3 `POST /api/generate` request flow

1. **Parse JSON body** with a tiny hand-rolled reader (only needs `prompt`/`n_gen`/`request_id` — same
   spirit as the minimal JSON parsing in `gpt2_pack.c`/`gpt2_tok.c`). Honor `Accept: text/event-stream`.
2. **Single-flight gate:** try-lock the worker mutex; if busy → **409** `{"error":"busy"}` (no body
   stream). Else proceed.
3. **Write the request frame** to the worker stdin pipe: a newline-delimited
   `{"prompt":"…","n_gen":N,"request_id":"…"}` (clamp `n_gen` to `1..256`).
4. **Stream:** read the worker stdout **line by line**; for each line, re-frame as one SSE message and
   **flush immediately** (§2.5). Inject `:` heartbeats on idle (§2.5).
5. **Terminate:** on the worker's `done` (or fatal `error`) line, finish the SSE stream and **close the
   TCP connection** (one-shot); release the mutex so the next queued request can run.

**Request body** (contract):
`{"prompt": string, "n_gen": int (default 20, clamp 1..256), "request_id": string (echoed)}`.

### 2.4 `POST /api/verify` (optional, the ONLY path that may touch Python)

Re-runs the **real** parity check for the exact ids just produced — the `gpt2_scale.sh`-equivalent
argmax/token-for-token compare against the fenced numpy oracle — and returns:
`{verdict:'PASS'|'FAIL'|'UNAVAILABLE', argmax_match:bool, max_abs_logit_diff:float,
token_for_token:bool, oracle:'numpy fp32', note}`.
- **UNAVAILABLE when python/the oracle isn't installed** (the demo path is Python-free **by design**):
  the server checks for `python3` + `helix-llm/tools/gpt2_numpy_ref.py`; if absent, return
  `UNAVAILABLE` and the UI links to page 2 — **never fakes a verdict** (framing_rule + gate **G5**).
- It must **never** show PASS without actually having run the check, and it must **never** put python on
  the hot path — `/api/verify` is the sole, optional, off-hot-path python touch (risk R7).

### 2.5 Response framing + flush contract (the C writer)

Headers: `Content-Type: text/event-stream; charset=utf-8`, `Cache-Control: no-cache, no-transform`,
`Connection: keep-alive`, `X-Accel-Buffering: no`. Each telemetry event is one SSE message:
```
id: <seq>\n
event: <event-name>\n
data: <single-line JSON object>\n
\n
```
- `event:` = the JSON's `_ev`; `data:` = the worker's line **verbatim** (the server does **not**
  reformat content); `id:` = the `seq` inside `data` (Last-Event-ID resume diagnostics; resume itself is
  not required for one-shot generation).
- **Flush per event:** `write()` to the socket with **`TCP_NODELAY`** set, no userspace buffering, so the
  browser renders layers advancing in real time. Because the GPU path is synchronous, emitting right
  after each `cuCtxSynchronize` gives a truthful real-time cadence with zero added synchronization.
- **Heartbeat:** a `: ping\n\n` comment every ~15 s of wall time if no event has flushed (keeps
  proxies/EventSource alive during long XL forwards). Comments are ignored by EventSource and are **not**
  telemetry.
- **Ordering (causal, fixed):**
  `hello → tokenize → for each step { forward_begin → embed → 48×(layer_begin → ~17×op → layer_end) →
  head(ln_f) → head(lm_head) → token } → done`. `error` may appear anytime and is terminal when fatal.
- **Backpressure / non-flood:** XL `Ngen=20` at op-detail ≈ `20 × (2 + 48×(2+17) + 2) ≈ 20×916 ≈ 18k`
  events over tens of seconds (hundreds/sec) — acceptable for SSE. `?detail=layer` drops the `op` events
  (~`Ngen×(2+48×2+2)` ≈ `Ngen×100`) for low-power clients. `detail=op` is the default hero experience.

### 2.6 Static file serving

URL path → `demo/<path>` with an extension→MIME table (`html, js, css, svg, json, woff2, png`). Reject
`..` and absolute paths. `demo/dashboard.html` is served **byte-for-byte** (page 2 unchanged). The whole
two-page demo runs from one `gpt2_serve_http` process at `http://127.0.0.1:<port>/` with **no Node, no
Python, no external web server**.

### 2.7 Config / lifecycle / concurrency

- **CLI:** `gpt2_serve_http --port 8848 --root <abs demo dir> --worker '<gpt2_infer cmd…>'`
  (or it builds the worker cmd from `--ptx/--weights/--vocab/--merges`). Binds 127.0.0.1.
- `/api/health` reports readiness by checking the worker emitted its `GPT2_SERVE_READY`/`hello` line;
  until then the UI shows PREVIEW/connecting.
- **Graceful shutdown:** send `{"cmd":"quit"}` to the worker stdin, reap the child.
- **Concurrency model:** one acceptor thread + a small fixed thread pool. Static files serve
  concurrently; `/api/generate` is funneled through the single worker (single-flight). Blocking,
  thread-per-connection is sufficient for a local demo and keeps the C simple; SSE flushing is plain
  `write()` + `TCP_NODELAY`.
- **Why not Python/Node:** the runbook's headline residual upgrade is a **Python-free production data
  path** (`gpt2_tok.c` + `gpt2_pack.c` already eliminate the interpreter). A C server keeps that claim
  intact end-to-end — the live chat demo runs with **ZERO Python/Node installed**, only the from-raw C
  toolchain + the worker. The numpy oracle stays Python **on purpose** and is invoked **only** by the
  optional `/api/verify` (and only if present), never on the hot path.

Skeleton (illustrative; not committed):

```c
/* gpt2_serve_http.c (LATER, NEW) -- POSIX sockets, no deps. Sketch only. */
int main(int argc,char**argv){
    Cfg c = parse_cli(argc,argv);                  /* port, root=demo dir, worker cmd */
    Worker w = spawn_worker(c.worker_argv);        /* fork/exec gpt2_infer --serve; 2 pipes */
    wait_ready(&w);                                /* read GPT2_SERVE_READY on worker stdout */
    int s = listen_local(c.port);                  /* bind 127.0.0.1 only */
    for(;;){ int fd = accept(s,0,0); pool_submit(handle_conn, fd, &c, &w); }
}
static void handle_conn(int fd, Cfg* c, Worker* w){
    Req r = read_http_request(fd);
    if (is_get(&r) && is_api_health(&r))      return write_json(fd, health_json(w));
    if (is_get(&r))                           return serve_static(fd, c->root, r.path); /* reject ".." */
    if (is_post(&r) && eq(r.path,"/api/generate")) return do_generate(fd, w, &r);
    if (is_post(&r) && eq(r.path,"/api/verify"))   return do_verify(fd, &r);            /* may be UNAVAILABLE */
    write_status(fd, 404);
}
static void do_generate(int fd, Worker* w, Req* r){
    if (!worker_trylock(w)) return write_status_json(fd, 409, "{\"error\":\"busy\"}");
    set_tcp_nodelay(fd); write_sse_headers(fd);
    worker_write_frame(w, build_frame(r->prompt, clampi(r->n_gen,1,256), r->request_id));
    char line[8192]; double last=now();
    while (worker_read_line(w, line, sizeof line)) {        /* one event per line */
        const char* ev = json_field(line,"_ev");
        sse_write(fd, json_field(line,"seq"), ev, line);    /* id:/event:/data:/blank + flush */
        last = now();
        if (is_done_or_fatal(ev, line)) break;
        if (now()-last > 15.0) sse_comment(fd, "ping");     /* heartbeat (also via idle timer) */
    }
    close(fd); worker_unlock(w);
}
```

---

## 3. (3) The wiring between (1) and (2)

The contract between worker and server is intentionally **dumb and one-way per direction**:

```
 browser (EventSource/fetch ReadableStream)
    │  POST /api/generate  {prompt,n_gen,request_id}        (HTTP, 127.0.0.1)
    ▼
 gpt2_serve_http.c  ── single-flight mutex ──┐
    │  worker_stdin pipe:                     │
    │     {"prompt":"…","n_gen":N,"request_id":"…"}\n      (newline-delimited JSON frame)
    ▼                                         │
 gpt2_infer --serve --emit-fd <pipe_w>        │  (worker holds XL weights + minted PTX resident)
    │  worker_stdout pipe (= --emit-fd):       │
    │     {"_ev":"hello",...,"seq":0}\n        │
    │     {"_ev":"tokenize",...,"seq":1}\n     │  one-line JSON per event, '\n'-terminated
    │     {"_ev":"forward_begin",...}\n  …     │
    │     {"_ev":"done",...}\n                 │
    ▲─────────────────────────────────────────┘
    │  server re-frames EACH line as ONE SSE message and flushes:
    │     id: <seq>\n  event: <_ev>\n  data: <line>\n  \n
    ▼
 browser  → app.js handlers {onHello,onTokenize,…,onDone,onError}  → panels + chat bubble
```

Key wiring invariants:
1. **The worker is a pure compute worker with ZERO HTTP knowledge.** It writes newline-JSON to a fd; it
   never speaks HTTP/SSE. Keeping emit = newline-JSON means serve mode is independently testable by
   piping a request frame on stdin and reading events on stdout (no server needed) — and it keeps the
   server a thin pump.
2. **The server never reformats event content.** The worker's JSON line becomes the SSE `data:` verbatim;
   the server only adds the `id:`/`event:`/blank framing (reading `_ev` and `seq` out of the line). This
   guarantees the wire payload the frontend parses is exactly what the worker emitted — no double schema.
3. **`seq` is the single ordering token.** The worker stamps it; the server mirrors it to `id:`. The
   frontend uses it only for diagnostics (resume not required for one-shot).
4. **Single-flight is enforced at the server.** The worker is inherently serial; the server's mutex makes
   that an honest 409 to the *second* caller rather than interleaving two requests' frames into one
   worker (which would corrupt the shared buffers — risk R-buffers, gate G7).
5. **Readiness handshake.** The server `wait_ready()`s on the worker's `GPT2_SERVE_READY`/`hello` line
   before flipping `/api/health.ready=true`; the frontend gates the composer on that (mock_plan
   switch-over: if `source=sse` but health not ready → "connecting…", optionally fall back to mock with
   the PREVIEW banner so the page is never blank).
6. **Frontend source-swap is invisible to the render layer.** Both the SSE source and the MOCK source
   expose the same `startGeneration(prompt,nGen,handlers)` interface (contract mock_plan); the SSE source
   parses `event:`/`data:` frames and dispatches to the matching handler. Flipping `TELEMETRY_SOURCE`
   from `'mock'` to `'sse'` is the only change at switch-over (build step 5).

---

## 4. (4) FAIL-CLOSED integration gates

Every gate is fail-closed: a printed FAIL is **never** exit 0, and **any divergence blocks serve mode**.
These reuse the existing committed gate scripts as-is (owned elsewhere) and add the new G1/G7 serve checks.

### G1 — SERVE == OFFLINE (the load-bearing gate)

The served XL output for the pinned prompt **"The capital of France is"** (`gpt2_infer --serve`,
in-process tokenize, greedy `Ngen`) **MUST equal the offline `scripts/gpt2_scale.sh MODEL=gpt2-xl`
gen-ids TOKEN-FOR-TOKEN**, and the decoded text must equal:
> *"The capital of France is the city of Paris. It is the capital of France and the largest city in France. It is"*

Method: capture the `done.gen_ids` from a real served run (drive the worker with a single request frame
on stdin, read the `done` line), and `diff` against `gpt2_scale.sh`'s `--generate 20` ids
(`/tmp/helix_gen_ids.txt` from the scale run / the oracle's `ref_gen_ids.txt`). **This proves the emit
hooks + the `d_ctx` safety memset changed nothing numeric.** Any mismatch ⇒ fail-closed, serve blocked.

### G2 — 124M / SCALE STAY GREEN

`scripts/gpt2_gpu_mvp.sh` (`GPT2_LOGITS_PARITY_PASS` + `GPT2_GENERATE_MATCH_PASS`; argmax **id 262**;
25/25 ids) **and** `scripts/gpt2_scale.sh` (large + xl, token-for-token vs oracle) remain green **after**
the `--serve` patch — proving `--block0`/`--logits`/`--generate` are untouched (the new branch is purely
additive and the shared `forward_full`/`forward_layer_gpt2` arithmetic is unchanged).

### G3 — CPU NO-PTXAS STAYS GREEN

`scripts/gpt2_cpu_parity.sh` (block-0 max-abs ~`1.14e-04`, argmax 262 == oracle, greedy token-for-token)
remains green — the CPU purest-trust path (`cpu_host.c` + `gpt2_cpu_ops.hx`) is untouched by the GPU
serve work.

### G4 — FIXPOINT BYTE-IDENTICAL (the hardest fail-closed line)

`scripts/reproduce_trust.sh` / `scripts/gpt2_demo_attest.sh` still produce the pinned anchors
**byte-for-byte**:
- seed **`9837db12`**, self-host fixpoint **`0992dddd`** (K2==K3==K4), gcc-DDC **`84363adb`**.

The `--serve` patch (downstream launcher, outside the fixpoint) and the new HTTP server (host tool) must
**NOT** perturb these. Because neither touches `kovc.hx`/`lexer.hx`/`parser.hx`/`seed.c`/
`train_transformer.c` or the from-raw mirror inputs, the fixpoint is structurally unaffected — but **G4
must be re-run green before merge** (risk R1: a careless edit to a shared header or the kernel corpus
could ripple; this design confines all changes to `gpt2_infer.c` + the new server file).

### G5 — PYTHON-FREE PRESERVED

`scripts/gpt2_pyfree.sh` stays green and the live chat demo runs with **ZERO Python/Node installed**
(server + worker are pure C; tokenize is in-process gpt2_tok). `/api/verify` is the **only** path that may
invoke python (the oracle) and **must degrade to `UNAVAILABLE`** — never crash the demo — when python is
absent.

### G6 — HONEST FENCE ACCOUNTING (see §5 for the full inventory)

The new committed `.c` (`helixc/runtime/gpt2_serve_http.c`) is reflected in the fence inventory with the
correct **Category-B host-tool** classification (HTTP/byte-pump, **zero compute-trust arithmetic**, like
`cpu_host.c`/`gpt2_tok.c`). The `--serve` additions stay **inside the existing `gpt2_infer.c`** (no new
file for those). The committed-`.c/.h` count and the "24 from-raw UNCHANGED" tally are updated truthfully
**by the trust-inventory owner, not in this task**.

### G7 — SINGLE-FLIGHT TRUTH

The server enforces **one** concurrent generation (matching the strictly-serial GPU); a concurrent
request gets **409** and the UI disables send — verified by a **two-request test** (fire two
`/api/generate` near-simultaneously; assert the second gets 409 and the first's stream is uncorrupted).
No silent interleaving that would corrupt the shared worker buffers.

### Gate sequencing (build steps 6–7)

1. **G1** (serve-mode parity capture) immediately after the `--serve` patch builds.
2. **Regression sweep** for **G2–G5**: re-run `gpt2_gpu_mvp.sh`, `gpt2_cpu_parity.sh`, `gpt2_scale.sh`
   (large+xl), `gpt2_pyfree.sh`, and `reproduce_trust.sh`/`gpt2_demo_attest.sh` — confirm the fixpoint
   (`0992dddd`) and gcc-DDC (`84363adb`) are byte-identical and all gates green.
3. **G7** (two-request single-flight test) against a live `gpt2_serve_http` + worker.
4. **G6** handled by the trust-inventory owner once `gpt2_serve_http.c` is committed.

All builds happen **later** under the **same from-raw ext4-mirror flow `gpt2_scale.sh` uses** (PTX minted
fresh from the `9837db12` seed; strictly serial GPU). **Nothing is built now.**

---

## 5. Honest Category-B fence accounting (G6 detail)

**Verified base (this plan's ground truth):** the live committed `.c`/`.h` count at HEAD is **28**
(`git ls-files "*.c" "*.h" | wc -l` = 28), **not 26** as the contract prompt's parenthetical stated. The
24-file / 15 605-LOC from-raw Category-A ladder + trust root is **UNCHANGED** and the self-host fixpoint
stays `0992dddd`. (The "26" figure does not match either the live tree or `docs/TRUSTED_C_INVENTORY.md`;
the truthful base is 28 — see §0.5.)

**What this chat-demo backend adds to the fence:**

| File | New? | Category | Compute-trust arithmetic? | Fence effect |
|---|---|---|---|---|
| `helixc/runtime/gpt2_infer.c` | **No** (additive `--serve` branch + inline emit module **inside** it) | B (existing) | **No** (emit hooks are printf-only; numeric path byte-identical) | **count unchanged** |
| `helixc/runtime/gpt2_tok.c` | **No** (drop `static` on 4 entrypoints + add pure-bookkeeping `decode_to_buf`/`decode_one`/`decode_range`; `main()` `#ifndef GPT2_TOK_LIB`-guarded) | B (existing) | **No** (string↔id byte concat only) | **count unchanged** |
| `helixc/runtime/gpt2_serve_http.c` | **YES (1 new committed `.c`)** | **B — host tool (HTTP/byte-pump)** | **No** (POSIX sockets, JSON re-framing, pipe pump; ZERO arithmetic on the compute-trust path, exactly like `cpu_host.c`/`gpt2_tok.c`) | **+1** |

**Expected new count after the build pass: 28 → 29 committed `.c`/`.h`** (the single new file
`gpt2_serve_http.c`). The from-raw Category-A ladder stays **24 / 15 605 LOC UNCHANGED**; Category B goes
from **6 → 7 files**. `gpt2_serve_http.c` is **outside the self-host fixpoint** (`gate_kovc.sh` never
compiles it) and performs **zero arithmetic on the compute-trust path** — it is a trusted-once host pump,
the same classification as `cpu_host.c`/`gpt2_tok.c`/`gpt2_pack.c`.

> **Recommended-path note.** If the later build instead chooses to put the emit module or any HTTP glue
> in a **new `.h`/`.c`** rather than inline, that would add **more** than +1 — so the design's explicit
> recommendation is: keep the `--serve` emit module **inline in `gpt2_infer.c`** and add **exactly one**
> new committed file (`gpt2_serve_http.c`), making the honest delta **+1 (28 → 29)**. Any `.inc`/`.hx`
> the build uses is outside the `.c/.h` fence (like `gpt2_unicode_ranges.inc`) and does not count.

The actual inventory update (count + "24 from-raw UNCHANGED" tally + the Category-B addendum row) is made
by the **trust-inventory owner**, not in this task (G6, and risk R10 coordination).

---

## 6. Build steps (LATER — nothing built now)

1. **PREREQ (owned elsewhere, must be green first):** the Python-free pipeline (`gpt2_tok.c`,
   `gpt2_pack.c`, `gpt2_pyfree.sh`) and the XL weights/config under `helix-llm/` — already exist; **do
   NOT modify**. Confirm `gpt2_scale.sh MODEL=gpt2-xl` is green as the G1 reference oracle.
2. **Design freeze (this doc + the contract):** telemetry schema + SSE framing locked so frontend (mock)
   and backend share one wire format.
3. **Frontend on mock (now, no builds):** `demo/index.html` + `demo/app.js` with the mock source default
   (owned by the frontend pass; the contract's `mock_plan`). Reviewable with zero backend.
4. **`gpt2_infer --serve` patch (LATER, by the `gpt2_infer` owner):** the additive `--serve` branch +
   `emit_*()` hooks at the real points (§1.4); in-process tokenize via the existing `gpt2_tok` logic
   (§1.6); reuse `device_init`/`alloc_buffers`/`setup_head`/`forward_full` unchanged; add the `d_ctx`
   safety memset under `g_serve` (§1.7); build via the SAME from-raw ext4-mirror flow `gpt2_scale.sh`
   uses (PTX minted fresh from `9837db12`). **No change to the numeric path.**
5. **`gpt2_serve_http.c` (LATER, NEW):** the small C HTTP+SSE server (§2) — static handler (serves
   `index.html` + `dashboard.html` unchanged), `POST /api/generate` forking the `--serve` worker, pump
   worker newline-JSON → SSE, `/api/health`, optional `/api/verify` (degrades to `UNAVAILABLE`). Plain
   POSIX sockets, gcc, no deps. Add to the fence inventory as Category-B host tool (G6).
6. **Wire frontend to real (LATER):** flip `TELEMETRY_SOURCE` default to `'sse'` (or `?source=sse`);
   verify the SSE source renders identically to mock against a live worker; add `/api/health`-gated
   connecting/fallback.
7. **Gates:** run **G1** (serve==offline), then the **G2–G5** regression sweep, then **G7** (two-request
   single-flight). **G6** by the inventory owner.
8. **Docs (owned by the docs agent):** add a short serve-mode + chat-demo section to the runbook; **do not
   edit trust-inventory docs here**. Update page-2 link wiring only if needed.

---

## 7. Risks (and mitigations) — backend-specific

| # | Risk | Mitigation |
|---|---|---|
| R1 | **Fixpoint perturbation (highest).** A careless `--serve` edit touching shared headers / the kernel corpus / from-raw mirror inputs could ripple into the self-host inputs. | Keep `--serve` additive and confined to `gpt2_infer.c`'s `main()` + the inline emit module; **G4 must byte-match `0992dddd`/`84363adb` before merge**. This task edits `gpt2_infer.c` at **zero** lines — the risk lands only when the owning agent applies the patch. |
| R2 | **Emit-hook numeric drift.** A hook that reads device memory adding a sync, or mutates a buffer, could diverge the ids. | Hooks are printf-only on host-scope values (`step`, `L`, literal kernel names, the already-copied argmax logit); **G1 token-for-token catches any drift**. |
| R3 | **`d_ctx` pad-row carryover between requests.** `d_ctx` is memset once at alloc; smaller-T requests after larger-T could see stale pad rows. | Add one no-arithmetic `cuMemsetD8(d_ctx,…)` at the top of `forward_full` under `g_serve` (§1.7); provably a no-op on real rows; **G1 confirms**. |
| R4 | **Event flooding / UI jank at XL scale (~18k events / 20-tok reply).** | Server flushes per event but the **renderer** batches via `requestAnimationFrame`; `?detail=layer` fallback; the collapsed-per-head op design (3 attn ops/layer, not 12 launches). |
| R5 | **OOM on the 8 GB 3070 (XL is "TIGHT").** Resident buffers across requests raise steady-state VRAM. | Rely on existing per-layer streaming (one layer + tied head resident); size activations for a bounded `Smax`; surface an honest `error{where:"oom"}` event (not a hang); re-verify the residency claim on the target box. |
| R6 | **SSE batched by buffering/proxies → kills real-time feel.** | 127.0.0.1 only, `TCP_NODELAY`, `no-transform`/`X-Accel-Buffering:no` headers, explicit per-event flush, `:` heartbeats. |
| R7 | **Python leaking into the hot path** (via `/api/verify`). | Hot path is 100% C (in-process `gpt2_tok` + `gpt2_infer`); python touched **only** by optional `/api/verify` and degrades to `UNAVAILABLE` (G5). |
| R8 | **Worker crash mid-stream** (driver error, OOM) leaves the SSE hanging. | `check()`/CK error paths emit `error{fatal:true}`; the server detects worker EOF/exit, emits a terminal `error` if none was sent, closes the stream, and the UI shows an honest red state (never hangs). |
| R9 | **Single-flight bypass** (two requests interleaving into one worker → buffer corruption). | One server-side mutex; second concurrent `/api/generate` → 409; **G7 two-request test**. |
| R10 | **Scope collision with the concurrent agent** editing `helix-llm/tools/*`, `gpt2_pyfree.sh`, trust-fence docs. | This design touches **none** of those and creates **no** files; the later build coordinates so the fence-inventory update (G6) and any `gpt2_pyfree.sh` changes are made by their owner, not duplicated. |

---

## 8. Summary

- **`--serve`** = a 4th, additive, forward-only branch in `gpt2_infer.c` that runs `device_init` +
  `alloc_buffers` + `setup_head` **once**, then loops on stdin request frames, calling the **unchanged**
  `forward_full`/`forward_layer_gpt2` with printf-only `emit_*()` hooks at the real op/layer/head/token
  sites, in-process tokenizing via `gpt2_tok`, with a no-arithmetic `d_ctx` safety memset and bounded
  per-layer device residency. **Numeric path byte-identical; fixpoint untouched.**
- **Server** = a single new C file `gpt2_serve_http.c` (POSIX sockets, no deps) that serves `demo/`
  static files (incl. `dashboard.html` byte-for-byte), forks **one** persistent `--serve` worker over two
  pipes, single-flights `/api/generate` (409 on busy), pumps the worker's newline-JSON → SSE with
  per-event `TCP_NODELAY` flush + heartbeats, exposes `/api/health`, and an optional `/api/verify` that
  degrades to `UNAVAILABLE` so the demo stays Python-free.
- **Wiring** = worker writes one-line JSON to `--emit-fd`; server re-frames each line as one SSE message
  (`id:`/`event:`/`data:` verbatim) and flushes; `seq` is the single ordering token; readiness handshake
  on `GPT2_SERVE_READY`; frontend swaps mock↔sse behind one uniform `startGeneration` interface.
- **Fail-closed gates:** **G1** served-XL ids == offline `gpt2_scale.sh` XL ids token-for-token (the
  load-bearing proof the hooks changed nothing); **G2** 124M/scale stay green; **G3** CPU no-ptxas green;
  **G4** fixpoint byte-identical (seed `9837db12`, fixpoint `0992dddd`, gcc-DDC `84363adb`); **G5**
  Python-free preserved; **G6** honest fence accounting; **G7** single-flight 409 verified.
- **Fence implication:** verified live base is **28** committed `.c`/`.h` (correcting the contract's stale
  "26"). The chat-demo backend adds **exactly one** new committed `.c` —
  `helixc/runtime/gpt2_serve_http.c`, a **Category-B host tool** (HTTP/byte-pump, zero compute-trust
  arithmetic) — bringing the count to **29**; the `--serve` work stays inside the existing
  `gpt2_infer.c`. The 24-file from-raw ladder stays **UNCHANGED**.
