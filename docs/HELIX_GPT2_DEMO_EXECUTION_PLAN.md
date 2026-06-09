# Helix × GPT-2 124M — Investor-Demo Execution Plan

**Status:** Phase 0 DONE. This is the step-by-step plan from here to a completed, honest investor
demonstration: GPT-2 124M (unchanged public weights) running a forward pass + greedy generation on
a Helix stack rebuildable from 299 hand-typed bytes, output-matched to a trusted reference and
bit-reproducible.

**Synthesized from six grounded design slices** (importer, CPU forward, GPU forward, tokenizer +
generation + CLI, parity/trust wrapper, the demo itself). Every load-bearing fact below was
re-verified against the real files on 2026-06-08 (oracle interface + constants, the `*` gitignore
fence + exactly-1-committed-`.py`, the pinned trust anchors `seed 9837db12…` / `K1 84363adb…` /
fixpoint `0992dddd…`, the verdict strings `REPRODUCE_TRUST: PASS` / `CAPSTONE_AUDIT_PASS`, the
`config.json` dims, and the `ref/ref_block0.npy` parity target).

> **The one principle that overrides everything: lead with trust/verifiability, never performance.**
> The product is a *bring-your-weights verified execution layer*, not a fast inference engine.
> Speed is a disclosed residual (a fraction of cuBLAS; the CPU path is slow). We never fake parity,
> never ship red, and gate before every commit.

---

## 1. Executive summary + Definition-of-Done

### 1.1 What we are building

Helix already contains ~95% of a GPT-2 forward pass: the v1.0/v1.3 GPU transformer capstone
(`helixc/runtime/train_transformer.c`) is a complete, GPU-resident, pre-norm transformer **forward**
built entirely from `kovc`-emitted PTX kernels and proven on a real RTX 3070 Laptop
(`CAPSTONE_AUDIT_PASS`). This project re-uses that engine (and adds a pure-Helix CPU twin) to run the
*public, unchanged* GPT-2 124M weights, matching a fenced pure-numpy reference oracle that already
loads the real weights and generates coherent text.

The deliverable is a ~2–3 minute live demo with a one-command trust wrapper that binds
*source → from-raw binary → GPT-2 output* into a signed attestation.

### 1.2 Definition-of-Done

**Minimum Viable Demo (MVP) — all must hold:**

1. `bash scripts/reproduce_trust.sh` prints **`REPRODUCE_TRUST: PASS`** live (the from-raw chain is
   real, not pre-recorded). *Already true today; the demo only sequences it.*
2. The GPU inference command produces **coherent** GPT-2 continuation from a fixed prompt, streamed
   token-by-token.
3. The parity gate prints **`PARITY: PASS`** with a worst-case relative logit diff strictly below
   tolerance vs the fenced oracle — fail-closed, never faked.
4. Running the hero command **twice** yields **byte-identical** output (two equal `sha256sum`) — the
   reproducibility shot.
5. The whole runbook fits in **≤ 3 minutes** in a timed dry-run; `runbook.json` dry-run passes.
6. A **pre-recorded backup** of a gated, green run exists and shows the same four success artifacts.
7. An **honest-residuals card** is in the runbook and the operator can state the residuals unprompted
   (complete-to-PTX-not-SASS; a fraction of cuBLAS; single GPU sm_86; importer/tokenizer are fenced
   host glue; fp32-only ⇒ ~1.5 B-param ceiling — **measured: GPT-2-XL 1.5 B runs on the 8 GB box;
   see `scripts/gpt2_scale.sh`**).

**Full Demo — MVP plus:**

8. The **CPU path** (`--cpu`) runs to coherent output (the no-`ptxas`-boundary closer) with a
   **measured** sec/token recorded, so the live-vs-recorded CPU decision is evidence-based.
9. CPU-path output is **bit-reproducible** across two runs (twin hashes) — making "zero trusted
   boundary above 299 bytes, bit-for-bit" a *live* claim.
10. A one-page **signed attestation** (`attestation/gpt2_attest.txt`) is generated and shown/handed
    out, binding prompt + the three pinned hashes + the parity diff + the output hash + date.

**Explicitly OUT of the DoD (forbidden overclaims):** beating cuBLAS; "fully verified GPU";
"complete to GPU machine code"; any AGI claim; `gpt2-xl`/Llama *performance* beats. (The `gpt2-xl`
*generalization* flex — same kernels, bigger model, no speed claim — was the optional Phase 8 and is
now DONE: GPT-2-Large 774 M + GPT-2-XL 1.5 B pass token-for-token via `scripts/gpt2_scale.sh`.)

---

## 2. The pitch + honest scope

### 2.1 The pitch (spoken close, ~20 s)

> *"Every AI company asks you to trust a stack you've never seen — a compiler built by a compiler
> built by a compiler, down to vendors you can't audit. Helix is the verifiable execution layer
> underneath your model. You bring your weights — here, GPT-2, completely unchanged — and we run them
> on a toolchain you can rebuild from 299 hand-typed bytes, that proves it reproduces itself
> byte-for-byte, that an independent compiler corroborates, and whose output matches a trusted
> reference and reproduces bit-for-bit. We're honest about the edges: it's complete to PTX, not
> below; it's a fraction of cuBLAS on speed; we run one GPU. What we sell isn't speed — it's the one
> thing nobody else can hand you: a model whose execution you can verify from the very first byte."*

### 2.2 Honest scope / residuals (disclosed in the demo and in the attestation)

- **Fenced host glue.** The importer, the byte-level BPE tokenizer, and the numpy oracle live under
  gitignored `helix-llm/` (the `.gitignore` is `*`). They are trusted host glue — no weights of their
  own, no compute-trust role. The committed toolchain stays **exactly 1 `.py`**
  (`verification/oracle/oracle_train.py`). Mitigation path: importer + tokenizer can be re-authored in
  C/Helix to become committable without touching the fence.
- **`ptxas` boundary (GPU path only).** Hand-auditable `hex0 → PTX`; below PTX, NVIDIA's closed
  `ptxas` + driver + the C CUDA-FFI launcher are trusted-once. **The CPU path has no such boundary**
  — that is its entire reason to exist.
- **fp32-only ⇒ ~1.5 B-param ceiling on the 8 GB sm_86 box; single GPU target.** GPT-2 124M (and even
  XL 1.5 B) fit; the ceiling bounds what "this stack runs" generalizes to. **Measured (2026-06-08):**
  GPT-2-Large (774 M) **and** GPT-2-XL (1.5 B) both run a real forward + greedy generation through the
  identical `kovc`-emitted PTX kernels (zero new ops) and pass token-for-token vs the oracle — the fp32
  1.5 B ceiling is confirmed by direct measurement, not estimated. Reproduce: `scripts/gpt2_scale.sh`
  (`MODEL=gpt2-large|gpt2-xl`). The committed `gpt2_infer.c` is already dimension-generic (dims via
  `HX_*` env, read from `config.json`); per-layer weight streaming keeps device residency to one layer
  plus the tied head, so the 36-/48-layer depth does not gate VRAM.
- **Parity is fp32-vs-fp32 within a measured tolerance on hidden states; EXACT on argmax + token
  sequence.** Never conflate "matches the reference" (within-tol floats) with "reproduces itself"
  (bit-exact, Helix-vs-Helix).
- **The oracle shares the architecture spec.** It is independent in *implementation* (separate code,
  fp32) but not in *specification* (same GPT-2 math). It catches implementation bugs, not a shared
  misunderstanding of GPT-2.

---

## 3. Current state — Phase 0 (DONE)

Verified on disk under `C:/Projects/Kovostov-Native/helix-llm/` (the whole tree is gitignored):

- `models/gpt2/model.safetensors` — F32, ~548 MB, **160 tensors all F32** (12 layers × 12 + 4 globals
  + 12 unused `h.{L}.attn.bias [1,1,1024,1024]` mask buffers; **no `lm_head` tensor** → wte/LM-head
  tie confirmed). `config.json`: `activation_function=gelu_new`, `layer_norm_epsilon=1e-5`, 12 layers,
  n_embd 768, 12 heads, vocab 50257, n_ctx 1024. Plus `vocab.json` + `merges.txt`.
- `tools/gpt2_numpy_ref.py` — **the parity oracle**: pure-numpy (no torch), correct byte-level BPE
  (uses the `regex` module for `\p{L}/\p{N}`), correct forward (`gelu_new` tanh, `eps=1e-5` affine
  LayerNorm, mask `np.triu(-1e10,1)`, score scale `1/sqrt(64)`, Conv1D as `x@W`, tied
  `logits = x @ wte.T`). Generates coherent text. **Interface:**
  `python3 gpt2_numpy_ref.py "<prompt>" <n_new>` → prints `ids`, dumps `ref/ref_block0.npy`, prints
  `OUTPUT`. (Needs `regex`: `pip install --break-system-packages regex`.)
- `ref/ref_block0.npy` — `(T,768)` post-block-0 hidden state, the first Helix parity target (15488 B
  on disk; T=5 for the default 5-token prompt).

**Param arithmetic (exact):** 124,439,808 floats = 124.44 M params = 497,759,232 bytes (474.7 MiB)
flat fp32. The 548 MB safetensors is larger because of the 12 unused `attn.bias` masks (~50 MB) + the
JSON header — the importer drops those.

---

## 4. Phases P1–P7 (dependency order)

### Ordering decision: importer first, then **GPU-first**, then CPU as the trust capstone

The two forward paths share one weight file and one oracle; building GPU first de-risks the importer
orientation, the four gap-fills (mask/eps/multi-head/bias), and the parity tolerances, so the CPU path
inherits validated weights and a known-good diff. The GPU path also reuses the capstone-proven
kernels → snappy live demo. **The CPU path is the airtight trust capstone (no `ptxas`), built second.**
A GPU-only MVP is reachable after P5; the CPU path (P3 below is *authored* early but *bring-up* lands
in P6′—see note) upgrades it to the full demo.

> **Sequencing note.** The slices were designed CPU-as-P2/P3 and GPU-as-P5. To honor the GPU-first
> build recommendation while keeping the numbered phases readable, we **author** the CPU op layer
> early (it shares the importer + oracle) but **gate CPU bring-up after GPU parity is green**. The
> milestone checklist (§8) makes the true critical path explicit: P1 → P5(GPU) → P4(gen) → MVP →
> P2/P3(CPU) → P6(trust) → P7(demo). The phase numbers below are kept in the brief's requested order
> (P1 importer, P2 CPU block, P3 CPU full, P4 tokenizer/gen, P5 GPU, P6 trust, P7 demo); read §8 for
> the executed sequence.

---

### P1 — Weight importer + Helix weight-file format

**Goal.** Convert `model.safetensors` → a flat little-endian fp32 weight binary in the exact tensor
order both forward paths consume, and prove it matches the oracle bit-for-bit / to ≤1e-6.

**The canonical contract — `gpt2_124M.weights` (header + flat fp32).** Per layer L=0..11, in this
exact order (none transposed for the capstone `mm_AB` engine — see the transpose resolution below):

| # | logical | source tensor | shape `[rows,cols]` | floats |
|---|---|---|---|---|
| 1 | ln_1.g | `h.{L}.ln_1.weight` | [768] | 768 |
| 2 | ln_1.b | `h.{L}.ln_1.bias` | [768] | 768 |
| 3 | attn.W_qkv | `h.{L}.attn.c_attn.weight` | [768,2304] | 1,769,472 |
| 4 | attn.b_qkv | `h.{L}.attn.c_attn.bias` | [2304] | 2,304 |
| 5 | attn.W_proj | `h.{L}.attn.c_proj.weight` | [768,768] | 589,824 |
| 6 | attn.b_proj | `h.{L}.attn.c_proj.bias` | [768] | 768 |
| 7 | ln_2.g | `h.{L}.ln_2.weight` | [768] | 768 |
| 8 | ln_2.b | `h.{L}.ln_2.bias` | [768] | 768 |
| 9 | mlp.W_fc | `h.{L}.mlp.c_fc.weight` | [768,3072] | 2,359,296 |
| 10 | mlp.b_fc | `h.{L}.mlp.c_fc.bias` | [3072] | 3,072 |
| 11 | mlp.W_proj | `h.{L}.mlp.c_proj.weight` | [3072,768] | 2,359,296 |
| 12 | mlp.b_proj | `h.{L}.mlp.c_proj.bias` | [768] | 768 |

Per-layer total = **7,087,872 floats**. Then globals once: `wte [50257,768]` (38,597,376),
`wpe [1024,768]` (786,432), `ln_f.g [768]`, `ln_f.b [768]`. **No separate LM-head tensor** — logits =
`hidden @ wte^T` (the tie costs zero extra storage). `attn.W_qkv` ships **fused [768,2304]** (Q|K|V is
a *column* slice the forward does at consume time, matching `np.split(qkv,3,axis=-1)`); `attn.b_qkv`
splits the same way.

**File format (64-byte header + payload):**
```
0  magic   u32 = 0x48584757 ('HXGW')   16 n_head  u32 = 12
4  version u32 = 1                      20 n_vocab u32 = 50257
8  n_layer u32 = 12                     24 n_ctx   u32 = 1024
12 d_model u32 = 768                    28 d_ff    u32 = 3072
32 n_float u64 = 124439808              40 reserved[24] = 0
64 payload float32[n_float] LE, in the order above
```
Total file = 64 + 497,759,232 = **497,759,296 bytes**. The header is self-describing so the reader
fails loudly on a wrong/short file (the capstone's headerless `init_weights.bin` has no such guard;
this is a deliberate robustness upgrade). Dims are asserted against `config.json` at read time.

**The transpose question, resolved honestly (highest-value correctness point).** HF Conv1D stores
`[in,out]`. The Helix capstone engine `mm_AB(x[S,K], W[K,N])` computes `x @ W` with weights-on-the-
right in `[in,out]` layout — *identical* to the oracle's `a @ W`. **Therefore the four Conv1D weights
ship as-is, NO physical transpose** for the capstone path. The "transpose four weights" lore is true
only for a `W@x` (weights-left) engine. The importer exposes a per-tensor `transpose: bool` flag (all
`False` for `mm_AB`) so the contract is explicit and a single flag flips if a forward GEMM ever uses
the other convention. **The trap:** `attn.c_proj.weight [768,768]` is *square*, so a wrong-orientation
copy is shape-valid and silent — only the oracle cross-check (gate 3) catches it.

**Files to create (all fenced under `helix-llm/`, no fence impact):**
- `helix-llm/tools/gpt2_import.py` — v1 importer. Reuses `gpt2_numpy_ref.load_safetensors` (identical
  byte logic), builds the order list programmatically over `L in range(12)`, asserts shape+dtype per
  tensor, **explicitly skips and accounts for** the 12 `attn.bias` masks (unexpected leftover = hard
  error), concatenates, asserts `total.size == 124439808`, writes the 64-byte header + payload, prints
  a manifest (per-tensor `[offset,count]` + checksum + payload sha256). CLI:
  `python3 gpt2_import.py [--model …] [--out …] [--validate]`. ~150 lines.
- `helix-llm/tools/import_parity.py` — the oracle cross-check (gate 3 below).
- **Output (gitignored):** `helix-llm/models/gpt2/gpt2_124M.weights` (498 MB; `--out` can write
  straight to the WSL ext4 mirror; the `C:\` copy is the single source of truth).
- **Committable route (lands before any "Python-free toolchain" public claim):**
  `helixc/runtime/gpt2_import.c` — mmaps the safetensors, hand-rolls a minimal JSON-header scanner
  (only `name`/`dtype`/`data_offsets`), writes the same format. C is not Python → commits freely.
  ~250 lines. (A pure-Helix `.hx` importer is a deferred Route B, blocked on Helix file-IO
  primitives.)

**Acceptance gate (GREEN only if ALL pass):**
1. **Shape/dtype/count (in-importer):** all 148 packed tensors match `config.json` dims; 12 masks
   accounted-for-and-skipped; payload == 124,439,808 floats; header round-trips.
2. **Round-trip identity (`--validate`):** re-read the flat file, slice back per offsets, assert
   **bit-exact** (`np.array_equal`) vs `load_safetensors`. fp32→fp32 with no math ⇒ must be exact.
3. **Oracle cross-check (the real one):** reconstruct the dict the oracle uses *from the flat file*,
   run `gpt2_numpy_ref.forward` on the **imported** weights for the canonical prompt, assert the
   post-block-0 hidden equals `ref/ref_block0.npy` to **≤1e-6 max-abs**. This is the only gate that
   catches the square-matrix `c_proj` transpose trap.
4. **Determinism:** print + log the payload sha256 so the forward slice and reviewers read the
   identical bytes the gate signed off on.

**Effort:** ~1 day (Python importer + round-trip + oracle cross-check). Route-A C: ~1 day, before the
"Python-free" claim.
**Dependencies:** none for v1 (only `model.safetensors`, `config.json`, the existing oracle +
`ref_block0.npy`). **Provides** the §P1 tensor order + format + signed sha256 to P2 and P5.
**Risks:** square `c_proj` transpose (mitigated by gate 3); fused-QKV column-split convention
(contract says "column slice"; gate 3 exercises real attention); wte-tie (stored once, no head tensor
to mishandle); DrvFs write speed (use `--out` ext4); endianness (x86-64 LE matches safetensors; assert
contiguity before `.tofile`).

---

### P2 — CPU single-block parity (the purest-trust artifact, block-0 milestone)

**Goal.** Re-express block 0 of the GPT-2 forward in pure Helix (`.hx`), compiled by the self-hosted
`kovc` (rebuildable from 299 bytes, **no `ptxas` boundary**), and match `ref/ref_block0.npy`.

**The hard architectural constraint that drives the CPU design.** `kovc` programs have a single fixed
global arena of `helix_arena_cap() = 6,291,456` i32 slots (~25 MB), one f32 per slot. GPT-2 124M
(~474 MB of weights) is ~20× too large to live in the arena at once. Plus: `read_file_to_arena(path:
STRLIT)` has a 1 MB read buffer with a hard `ud2` truncation trap and packs one byte per i32 slot (so
it cannot ingest the model and would 4×-inflate bytes→slots); the bootstrap has **no argv**
(`main() -> i32`, paths are baked literals). **Conclusion:** the CPU forward cannot mmap-and-hold the
model in-Helix. The design is a **C harness (the `train_transformer.c` pattern minus CUDA) that owns
the weight `mmap` and streams tensors into the Helix arena window-by-window**, calling per-(layer,
sub-op) Helix entry points that operate on whatever arena tile the harness filled. **Every arithmetic
op stays in Helix** (the trust claim); the C harness does only byte-movement (mmap, tile upload,
token/positional gather) — the authorized "fenced host glue" residual.

**Op inventory (exists vs must-add), grounded in the real stdlib:**
- **matmul — EXISTS:** `tf2d_matmul` (`tensor.hx:1112`), `tf2d_matvec` (`tensor.hx:672`), f32 row-major,
  NaN-skipping.
- **GELU tanh — EXISTS, exact:** `__gelu` (`transcendentals.hx:523`) is bit-for-bit `gelu_new`
  (`gpt2_numpy_ref.py:95`). Wrap via `gelu_layer` (`nn.hx:594`). No new op.
- **residual add — EXISTS:** `tf1d_add` (`tensor.hx:1026`). No new op.
- **(1) token+positional embedding gather — HOST-SIDE, no new kernel.** C harness gathers
  `wte[id]+wpe[pos]` into the arena hidden buffer.
- **(2) causal-masked softmax — NEW `gpt2_causal_softmax_row`.** `softmax_layer` (`nn.hx:789`) exists
  but is unmasked. Clean Helix form: for query row `i`, run the max-subtract softmax over keys
  `[0..i]` only and write 0 to keys `[i+1..T)` — numerically identical to adding `-inf`, no `-inf`
  literal needed. Reuse `tf1d_max` (`tensor.hx:942`) + `__exp` (`transcendentals.hx:34`); keep the
  `sum_e<=0||NaN` fail-closed guard. Scores must already carry the `1/sqrt(64)=0.125` scale (apply via
  `tf1d_mul_scalar`, `tensor.hx:1093`). ~20 lines.
- **(3) LayerNorm affine + eps — NEW `gpt2_layernorm_affine`.** Genuinely new on *both* paths: CPU
  `layer_norm_f32` (`nn.hx:666`) has eps but no affine; GPU `gpu_layernorm_fwd_save` has affine but no
  eps. Copy `layer_norm_f32`'s mean/var/`safe_eps`/`denom` + its `denom<=0||NaN` fail-closed guard,
  then in the write loop multiply by `γ[j]` and add `β[j]`. Variance is the **biased/population** form
  (divide by `n`, matching numpy's `x.var()` default and `layer_norm_f32:680`). Pass eps as the f32
  literal `0.00001_f32`. ~25 lines.
- **(4) multi-head split/merge — host loop in Helix, NO new kernel.** For each layer: after the QKV
  linear (`T×768` Q/K/V slabs), loop `h in 0..12`; head `h` is columns `[h*64:(h+1)*64]`. Copy the
  head's columns into a contiguous `T×64` scratch (`tf2d_matmul` needs contiguous header-valid
  tensors), do `Q_h@K_h^T` → scale → mask+softmax → `@V_h`, scatter back into the merged `T×768` at
  columns `[h*64:]`. Reuses only matmul/softmax.
- **(5) Q/K/V/MLP bias row-broadcast — free via `dense_layer_f32_forward`.** Call the biased dense per
  output row (`nn.hx:386` is matvec + bias). No new op.

**Files to create (fenced under `helix-llm/src/`):**
- `gpt2_cpu_ops.hx` — the two NEW ops + thin GELU/residual/biased-linear wrappers.
- `gpt2_cpu_forward.hx` — layer orchestration, head split/merge, arena offset map.
- `gpt2_cpu_main.hx` — all-Helix block-0 driver via pre-sharded `<1 MB` tensor files (the standalone
  proof for the small milestone).
- `cpu_host.c` — the CUDA-free `train_transformer.c` twin: mmap weights, host embedding gather, tile→
  arena streaming, op invocation, output write.
- *In-Helix transpose (already available, no add needed):* `tf2d_transpose` **already exists** at
  `helixc/stdlib/tensor.hx:1299` (a real f32 transpose with `t2d_shape_ok` guards) — use it directly if
  we choose in-Helix transpose; otherwise the harness/importer handles orientation. (`ti2d_transpose:1167`
  is the i32 sibling.) **No change to `helixc/bootstrap/kovc.hx`/lexer/parser** ⇒ the self-host fixpoint
  sha (`0992dddd…`) stays byte-identical (this slice is library + harness only).

**Block-0 sequence:** host gather → `gpt2_layernorm_affine`(ln_1) → c_attn biased linear → split
Q|K|V → 12-head loop (scale 0.125 → mask → softmax → @V → merge) → c_proj biased linear + residual →
`gpt2_layernorm_affine`(ln_2) → c_fc biased linear → `gpt2_gelu_inplace` → c_proj biased linear +
residual → dump `x`, diff vs `ref_block0.npy`.

**Acceptance gate:** `max_abs(helix_x − ref_block0) < 1e-3` AND `mean_abs < 1e-4` over all `T*768`
elements (fp32; tighten to 1e-4/1e-5 if the Taylor `__exp`/`__tanh` hold). **Block-0 parity is the
go/no-go for the full CPU forward.**

**Effort:** new ops ~1 day; orchestration + head bookkeeping ~2–3 days; block-0 bring-up/debug to
tolerance ~2–4 days. **Dependencies:** P1 weight file (hard); tokenizer ids can be hardcoded from the
oracle for the canonical prompt. **Risks:** Taylor transcendental accuracy compounding across layers
(f64 variants `__exp_f64`/`__sqrt_f64` exist as a fallback for accumulation-sensitive reductions);
header-valid tensor requirement for head slices (assert non-error returns); the arena ceiling means
the purest "zero C" claim is *not* met by v1 (the all-Helix sharded loader is a documented stretch
goal) — pitch honestly: "arithmetic is 100% Helix-from-raw; weight byte-movement is fenced host glue."

---

### P3 — CPU full-forward parity (12 layers + tied LM head + generation)

**Goal.** Loop the block over all 12 layers (residual stream threading through), then `ln_f`, then the
tied LM head `logits = x @ wte^T`, and match the oracle's logits + next-token argmax + greedy
sequence.

**Tasks.** Reuse the P2 op layer unchanged across 12 layers. Final `ln_f` via
`gpt2_layernorm_affine`(ln_f). Tied head: for greedy decode only the **last position's** logits matter
→ use a **matvec of the last hidden row against `wte` rows** (50257-wide), not a full `T×50257`
matmul (the single biggest free speedup; the head is the most expensive op). Argmax via `tf1d_argmax`
(`tensor.hx:1005`) or `argmax_rows_f32` (`nn.hx:1051`, NaN-robust). Generation = O(N) full forwards
(no KV-cache in v1, matching the oracle).

**Acceptance gates:**
1. **Logit parity:** `max_abs(helix_logits[-1] − oracle) < 1e-2` over the 50257 row (looser than
   block-0 because error compounds across 12 layers).
2. **Argmax match (headline):** `argmax(logits[-1])` equals the oracle's next-token id — **exact**.
3. **Generation:** N-token greedy decode produces the **exact same token-id sequence** as the oracle
   — exact, not within-tol.
4. **Reproducibility:** two runs of the compiled CPU binary produce **byte-identical** output `.bin`
   (no RNG, no threading, fixed arena).
5. **Trust:** the binary is produced by the gated `kovc` (`gate_kovc.sh` PASS unchanged); the only
   host residual is the fenced C harness (byte-movement only).

**Effort:** ~1–2 days once block-0 is green (CPU path total ~1.5–2.5 weeks incl. P2, assuming GPU has
shaken out importer/order details first). **Honest perf:** naive scalar triple-loop matmuls;
single-forward of a short prompt ≈ seconds to tens of seconds; multi-token ≈ minutes. Mitigations
(all honest): cap generated tokens for the live run (N=3–5) + pre-record a longer offline run; short
prompt; last-row-only LM head; frame as "slow but bit-reproducible" with the GPU path as the speed
story. **Dependencies:** P2 (hard), P1 weight file (hard), tokenizer ids (P4 or hardcoded).

---

### P4 — Tokenizer + generation loop + inference CLI (coherent text)

**Goal.** The end-to-end driver: `prompt string in → generated text out`, wiring a byte-level BPE
tokenizer + an autoregressive loop to **either** backend through one narrow seam. No kernels, no math.

**The forward seam (the one interface shared with P2/P3/P5)** — `helix-llm/src/forward_iface.h`:
```c
typedef struct {
    void* ctx;                                   // backend-private (weights, GPU module, buffers)
    int (*step)(void* ctx, const int* ids, int T, float* out_logits);  // last-row logits[50257]
    void (*free)(void* ctx);
} fwd_iface_t;
fwd_iface_t cpu_forward_load(const char* weights_path);
fwd_iface_t gpu_forward_load(const char* weights_path, const char* ptx_path);
```
`step` recomputes from `ids[0..T)` each call (stateless except immutable weights), matching the
oracle. **Publish this header first** so P2/P3/P5 build against it. The embedding gather is host-side
*inside the backend* (this slice hands over the id array; it does not gather).

**Tokenizer — v1 fenced host C, spec-locked to the oracle** (`gpt2_numpy_ref.py:31–87`). Reproduce
byte-identically: the 256-entry `bytes_to_unicode` permutation; the **pretokenizer regex**
`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`; the merge-by-lowest-rank
loop (`merges.txt`, drop the `#version` header + trailing blank, ref:45); the `vocab.json` map +
inverse; `decode` with `errors="replace"` (emit U+FFFD on invalid UTF-8). **Pretokenizer regex is the
#1 tokenizer risk** (`\p{L}/\p{N}` + the `\s+(?!\S)` lookahead). **Decision: vendor PCRE2 into the
fence** (Option A) — compiles the oracle pattern verbatim → identical by construction; it is host glue
under gitignored `helix-llm/`, no fence impact. A hand-rolled UTF-8 state machine (Option B, ~150–250
lines) is the dependency-free fallback and the seed of a later in-Helix tokenizer. **Rejected:**
shelling out to the Python oracle in the driver (muddies the Python-free story even inside the fence).

**Files to create (fenced under `helix-llm/src/`):**
- `forward_iface.h` (the seam, published first).
- `tokenizer.{h,c}` + `json_min.c` (byte↔unicode table, vocab hash + reverse array, merge-rank map,
  merge loop, a ~40-line `vocab.json` scanner handling `\"`/`\\`/`\uXXXX` escapes).
- `generate.{h,c}` — the loop + sampling. `gen_cfg_t {greedy, temp, topk, seed, max_new, eos_id}`.
  Greedy = argmax (the parity mode, deterministic). Temperature/top-k compute softmax in **f64**
  host-side (sampling is glue; f64 here does not affect the f32 forward-parity claim). Seeded xorshift
  RNG (reuse the harness's `xs()`) → "same seed → same text". `n_ctx=1024` guard: cap-and-stop for v1
  (left-truncation noted as the extension); EOS = 50256.
- `helix_gpt2_main.c` — the CLI `helix-gpt2`:
  `[--backend cpu|gpu] [--weights PATH] [--vocab PATH] [--merges PATH] [--n N] [--temp T] [--topk K]
  [--seed S] [--show-ids] [--selftest] [--ptx combined.ptx] "prompt"`. Default `--backend cpu`;
  `--ptx` required for gpu. Prints decoded full text (`--quiet` prints only the generated suffix for
  scripted diffing). `--show-ids` mirrors the oracle's `print("ids", …)` for parity debugging.
- `tok_selftest.c` (+ `--selftest`) and a few-line extension to `gpt2_numpy_ref.py` to dump
  `ref/tok_cases.json` (oracle ids per case).

**Acceptance gates:**
1. **Tokenizer gate (forward-independent, can be green before any forward exists):** C `tok_encode`
   is **id-for-id identical** to the oracle `BPE.encode` across a prompt battery (ASCII, leading-space
   tokens, contractions, digits, punctuation, UTF-8 like `"café — naïve"`); `decode∘encode`
   round-trips losslessly. Source of truth = `ref/tok_cases.json`. No GPU, no weights.
2. **Headline gate — greedy token-for-token:** for a fixed prompt + `--n`, `helix-gpt2 --seed 0`
   greedy produces the **exact same token-id sequence** as the oracle (whose loop is ref:133–135), for
   **both** backends.
3. **Decode fidelity:** the printed string equals the oracle's `tok.decode(ids)` byte-for-byte.
4. **Reproducibility (sampled):** with `--seed S`, two sampled runs are byte-identical; sampled mode
   is *not* token-matched to the oracle (its pre-softmax logits are what the forward gates check).

**Effort:** ~7–9 days total; T1–T6 (seam, tokenizer, loop, CLI plumbing) are **forward-independent**
and run fully in parallel with P2/P3/P5 — only the end-to-end greedy gate blocks on a working `step`.
**Dependencies:** ≥1 backend's `step` (hard, for gate 2 only); P1 weight file (runtime input the CLI
passes through). **Risks:** pretokenizer Unicode-class correctness (mitigated by PCRE2); the fence
residual (v1 tokenizer is fenced host C + vendored PCRE2 — disclosed; the committable end state is an
in-Helix `.hx` tokenizer over `helixc/stdlib/string.hx` + `read_file_to_arena_dyn`, a scoped later
milestone); no KV-cache (perf residual; CPU may be slow per token, GPU is the snappy path);
`vocab.json` escape handling (caught by gate 1).

---

### P5 — GPU-path GPT-2 forward (speed; extend the capstone)

**Goal.** A forward-only inference launcher that reuses the capstone's GPU-resident transformer +
kovc-emitted PTX kernels, adds exactly the four GPT-2 gap-fills, and matches the oracle. **This is the
snappy live demo and the first path to bring up (it de-risks the importer + gaps for the CPU path).**

**Why fork, not extend in place.** `train_transformer.c` is the frozen v1.0/v1.3 capstone artifact
(`CAPSTONE_AUDIT_PASS` asserts byte-identical default behavior). Adding GPT-2 control flow risks
perturbing the audited path. A sibling `gpt2_infer.c` keeps the audit untouched while sharing the
kernel corpus. **Do NOT touch `train_transformer.c`.**

**GEMM-validity reconfirmation (honest).** The capstone naive kernels launch `grid=M, block=N` (one
thread per output cell) — but `block=N` is capped at 1024 threads, and GPT-2 has N ∈ {768, 3072,
50257}, all > 1024. So **route every forward GEMM through the tiled `HX_OPT` kernels** (`tiled_matmul`
/ `_abt` / `_atb`, `grid=(N/64,M/64) block=(16,16)=256`, N-magnitude-independent, already proven vs a
pedantic-f32 cuBLAS oracle in `gemm_perf` mode). Constraint: **M%64==0, N%64==0, K%8==0**. d_model 768,
d_ff 3072, head_dim 64 all satisfy it. **Pad S to a multiple of 64** (causal mask makes pad rows
harmless; read only real-token logit rows). **Pad vocab 50257 → 50304** (=786·64; extra 47 columns
zeroed, never argmaxed/read). No new GEMM kernel needed. **The `gpt2_infer.c` fork selects kernels
per-op** (it loads its own `CUfunction` handles): GEMMs use the **tiled** OPT kernels (`tiled_matmul`
/`_abt`, the only ones valid at N>1024), while the masked-softmax and eps-LN below run **block-per-row
(grid=rows, block=1)** — a mixed selection the fork is free to make. This is the capstone's own
forward_layer (`train_transformer.c:227`, `mm_AB`/`mm_ABt` wrappers at :126/:134) minus the four GPT-2
deltas; verified that `mm_AB` at `HX_OPT>=1` already routes to the tiled `grid=(N/64,M/64) block=(16,16)`
kernel and that, at d=64, the capstone already does `mm_ABt + scale_attn(1/sqrt d)=0.125` (so dropping
`gpu_qkt` is exactly the existing OPT behavior).

> **Kernel-source rule (a fence-preservation constraint, verified).** The two new ops below are authored
> as standalone `@kernel` `.hx` files whose **bodies are hand-written** — modeled on the *naive*
> one-thread/block-per-row `gpu_softmax_kernel.hx` and `gpu_layernorm_fwd_save_kernel.hx` (both have full
> editable bodies). They are **NOT** edits of the `*_blockred` variants: `softmax_blockred` /
> `layernorm_fwd_save_blockred` are one-line wrappers around the **fused intrinsics**
> `__softmax_blockred` / `__layernorm_fwd_save_blockred` whose bodies live inside `helixc/bootstrap/kovc.hx`
> (the self-host compiler). Touching those would change `kovc.hx` and **break the byte-identical
> self-host fixpoint `0992dddd…`** — forbidden by the from-raw trust constraint. The per-row naive
> kernels launch grid=rows(=S≤1024), block=1, which is valid for GPT-2 (the block-size cap that forces
> tiled GEMMs does not bite a per-row kernel). The lower occupancy of block=1 softmax/LN over an S≤1024
> attention matrix is an accepted, disclosed perf residual, not a correctness issue.

**The four gaps and how each is closed:**
- **(1) embedding gather — HOST-side, no kernel.** Read `wte`/`wpe` into host buffers; gather
  `wte[tok[s]]+wpe[s]` into a host `[S,768]` (zero pad rows); one `cuMemcpyHtoD` into `x_in` — this is
  exactly how the capstone already injects its input (`train_transformer.c:400`).
- **(2) causal mask — NEW `gpu_softmax_causal`.** Fold the predicate into a **hand-written copy of the
  naive `gpu_softmax` body** (max/exp/normalize, `block_idx()`=query row, block=1): for query row `i`,
  reduce over keys `j<=i` only and write `y[i,j]=0` for `j>i` (numerically identical to a `-inf` add, no
  `-inf` literal, no extra HBM round-trip). New file `helixc/examples/gpu_softmax_causal_kernel.hx`. (A
  `softmax_causal_blockred` higher-occupancy variant would require a `kovc.hx` intrinsic and is therefore
  **deferred** — it is NOT on the v1 path.)
- **(3) LayerNorm eps — NEW `gpu_layernorm_fwd_eps`.** A **hand-written copy of the naive
  `gpu_layernorm_fwd_save` body** (which has affine but no eps, verified at `:23`) with one change:
  `__gpu_rsqrt(var)` → `__gpu_rsqrt(var + 0.00001)`. Keep the existing affine; biased/population variance.
  New file `helixc/examples/gpu_layernorm_fwd_eps_kernel.hx`. (The `ist` save the backward needs is
  irrelevant for inference; the forward-only kernel can drop it or keep it harmlessly.)
- **(4) bias row-broadcast — NEW `gpu_add_bias_rowbcast`.** `y[i] += bias[i % cols]`, one thread per
  element (mirrors `vector_add` launch), dimension-generic (cols ∈ {2304,768,3072}). Host-side bias
  add is rejected (5 extra full-tensor transfers/layer). New file
  `helixc/examples/gpu_add_bias_rowbcast_kernel.hx`.
- **Note:** `gpu_qkt` bakes 0.25 (d=16 only) and is **dropped from the forward path**; use `mm_ABt` +
  `gpu_scale_rt(0.125)` at d=64 (stated to avoid a silent wrong-scale bug).

**Multi-head split/merge — host loop, recommended on-device packing.** One fused QKV GEMM
(`xn1[S,768]@Wqkv[768,2304]→QKV[S,2304]`) + `gpu_add_bias_rowbcast`. Treat QKV as three `[S,768]` slabs
(Q@0, K@768, V@1536), each 12 contiguous `[S,64]` head blocks (head-major). Loop h=0..11: pack
contiguous `Q_h/K_h/V_h[S,64]` (a tiny copy — a head slice's stride ≠ width, so the proven GEMM
kernels need contiguous operands); `mm_ABt(Q_h,K_h)→scores_h[S,S]` → `gpu_scale_rt(0.125)` →
`gpu_softmax_causal` → `mm_AB(attn_h,V_h)→ao_h[S,64]`; scatter into `ctx_attn[:,h*64:]`. Then c_proj
GEMM + bias + residual `vector_add`. Per-head scratch allocated once, reused across heads/layers.

**Per-layer `forward_layer_gpt2(L,x)`:** ln_fwd_eps(ln_1) → mm_AB QKV + bias → 12-head loop → mm_AB
c_proj + bias → vector_add residual → ln_fwd_eps(ln_2) → mm_AB c_fc + bias → `gpu_gelu` (unchanged =
gelu_new) → mm_AB c_proj + bias → vector_add residual. `forward_full_gpt2`: 12 layers → ln_fwd_eps
(ln_f) → mm_AB tied head (`W_lm = wte^T`, padded [768,50304]) → copy real-token logit rows DtoH. No
CE/Adam.

**Files:**
- **CREATE** `helixc/runtime/gpt2_infer.c` — the forked forward-only launcher (the bulk). Replaces
  `gen_weights`/`upload_weights` with `load_gpt2_weights()` reading the P1 flat file in order;
  supports `--logits` dump (real-token `[S,50257]` to a flat `<f4` file for parity) and `--generate N`
  greedy loop. Env defaults bake GPT-2 (`HX_NL=12 HX_D=768 HX_HEADS=12 HX_V=50304 …`).
- **CREATE** the three kernel `.hx` files above (+ optional `softmax_causal_blockred`,
  `gpu_gather_head`).
- **CREATE** `scripts/gpt2_gpu_parity.sh` — the slice's gate, modeled on `capstone_audit.sh`
  (neutralize `HX_*` env, mint driver from the 299-byte seed, emit combined PTX, build, run, compare
  to oracle, negative controls). STRICTLY SERIAL GPU.
- **MODIFY (additive only)** `cuda_launch.c` — add standalone verify modes for the three new kernels
  (same pattern as existing `layernorm_save`/`softmax`/`affine` modes), so each is unit-gated vs a CPU
  reference before integration. **Do NOT touch `train_transformer.c`.**
- **MODIFY** the fenced importer to also emit the extended layout (biases + tied head padded to 50304
  + raw `wte`/`wpe` blocks for host gather).

**PTX mint flow (unchanged in spirit from the capstone audit):** `gate_kovc.sh` mints the driver from
the 299-byte raw-binary seed; concatenate the forward-only kernel set into `/tmp/kernel_in.hx`; the
seed-minted driver emits `/tmp/out.ptx`; `gpt2_infer.c` `cuModuleLoadData`s it. Kernel list drops
backward/adam, adds the 3 new ones. **Provenance rule:** the GPT-2 binary/PTX MUST be the one just
minted from the rebuilt seed, never a cached artifact.

**Acceptance gates:**
1. **Per-kernel unit gates (cuda_launch):** `gpu_softmax_causal` — row i has `y[i,j]=0` for j>i, sums
   to 1 over j≤i, matches CPU causal-softmax (tol 1e-3) + mutate negative control; `gpu_layernorm_fwd_eps`
   — vs CPU affine-LN-with-eps + `ist[r]=rsqrt(var+1e-5)` (tol 1e-3), eps-stripped PTX must diverge on
   a near-zero-variance row; `gpu_add_bias_rowbcast` — `y[i]==x0[i]+bias[i%cols]` cell-exact + perturbed-cell
   negative control.
2. **Block-0 hidden parity (the anchor):** dump post-block-0 `[S,768]`, diff vs `ref/ref_block0.npy`,
   **max abs rel error < 1e-3** on real-token rows. Proves embedding + mask + eps + multi-head + bias
   + GEMM orientation all at once.
3. **Full-logits parity (headline):** all 12 layers + ln_f + tied head, dump `[S,50257]` (ignore the
   47 pad cols), diff vs oracle. **Argmax of the last real-token row matches the oracle exactly**, AND
   max abs logit diff < a small absolute tol (~1e-2 on logits of O(10), reported honestly; an
   argmax-match + top-5-overlap fallback is acceptable *if stated*, never faked).
4. **Coherent generation:** `--generate ~20` greedy → token IDs match the oracle's continuation
   exactly.
5. **GEMM-at-scale reconfirmation:** a `gemm_perf` run at true GPT-2 magnitudes (M=64..1024,
   K∈{768,3072}, N∈{768,3072,50304}) PASSES vs the cuBLAS oracle on sm_86.
6. **Fail-closed wrapper:** `gpt2_gpu_parity.sh` emits `GPT2_GPU_PARITY_PASS`/`FAIL` → process exit.

**Effort:** ~3–5 focused sessions: 3 kernels + cuda_launch modes ~0.5; `gpt2_infer.c` fork + head loop
+ bias + eps LN + gather ~1.5–2; importer extension ~0.5; parity gate + block-0 + full-logits +
generation bring-up ~1–1.5 (most real time chasing GEMM orientation + head-slice off-by-ones).
**Dependencies:** P1 (hard, the flat weight file + orientation contract — this slice owns the loader
and the importer extension, so it measures the orientation end-to-end); P4 tokenizer at the fence for
`--generate` ids (v1 reuses the oracle's BPE). **Risks:** `ptxas` boundary (inherent, disclosed —
CPU path is the airtight one); single-head→12-head host loop is highest bug-density (block-0 anchor
flushes it); S/vocab padding correctness (enforce zeroed pad rows + slice logits to 50257);
fp32-vs-fp32 numeric drift (per-op unit gates + block-0 anchor catch it; argmax+top-k is the honest
fallback gate the demo actually needs).

---

### P6 — Trust wrapper + attestation (the verification spine)

**Goal.** A one-command, fail-closed wrapper that binds *source → from-raw binary → GPT-2 output* into
a signed attestation, plus the per-phase parity methodology and load-bearing negative controls.

**Fence rule for this phase:** anything Python lives under gitignored `helix-llm/` (audit witnesses,
never on the compile/run path); anything committed (the wrapper, the attestation emitter) is **bash +
existing committed C/Helix**, never Python. The committed `.py` count stays at 1.

**Parity methodology (mirror the capstone's reviewed discipline):**
- **Extend the oracle (`gpt2_numpy_ref.py`, fenced)** to dump the full ladder under `helix-llm/ref/`
  for a **pinned prompt**: `ref_embed.npy`, `ref_block{0..11}.npy`, `ref_lnf.npy`, `ref_logits.npy`,
  `ref_argmax.npy`, `ref_tokens.json`. Add a fenced twin emitter `dump_ref_bins.py` that re-saves each
  as flat `<f4` `.bin` (the format the capstone already uses — `train_transformer.c` writes
  `fwrite(...,sizeof(float),...)`, `oracle_train.py` reads `np.fromfile(dtype="<f4")`), so neither
  side needs a `.npy` parser and no new serialization code is written.
- **Comparator `helix-llm/tools/parity_check.py` (fenced):** worst-case **relative** diff
  `max |h−o| / (|o| + 1e-8)` (the capstone metric + a denominator floor for hidden states that pass
  through 0); for **logits**, also report an **absolute** diff (near-zero logits are common — argmax
  is what matters there). Signature `parity_check.py <helix.bin> <ref.bin> <shape> <rel_bar>
  <abs_bar>`; a count mismatch is a **hard FAIL** (the `nrows<10` vacuous-pass guard generalized);
  exits nonzero on FAIL.
- **Tolerance policy (measure-then-pin, never assert a hoped number):** hidden/final-LN states use a
  worst-case **relative bar 2e-2** (the capstone's already-accepted f32-vs-f64 bar; loose enough for
  honest 12-block drift, tight enough that a real bug blows through by orders of magnitude). **The
  discrete gates get ZERO tolerance:** `ref_argmax` exact integer equality, `ref_tokens.json` exact
  byte-for-byte on the id list — **this is the product-level headline parity claim**; float tolerance
  on hidden states is the *diagnostic ladder* that localizes a failure (first diverging gate pinpoints
  the broken op). If a verified-correct forward measures >2e-2 on a deep block but argmax + the full
  token sequence still match exactly, raise the **float** bar to the measured value *with documented
  justification* — never loosen the discrete gate.
- **Comparator self-tests (it must be proven before it can gate):** self-compare ⇒ `worst_rel==0`
  PASS; `ref_block0.bin` vs `ref_block1.bin` ⇒ FAIL (discriminates); different element counts ⇒ hard
  FAIL.

**Negative controls (prove the gates are load-bearing — the capstone's most important idea):**
- **NC-A (weight perturb):** scale one tensor by 1.001 on either side, re-dump, compare clean-vs-
  perturbed ⇒ **must FAIL** (`negctl_weight.py`, fenced).
- **NC-B (op perturb, the strong one):** drop the causal mask / drop the LN eps / swap tanh-GELU for
  erf-GELU / omit one Conv1D transpose ⇒ each **must FAIL** (`negctl_op.py`, fenced, + Helix-side
  env-flag/broken-build toggles the forward slices expose). **If any "broken" variant PASSES, the
  whole parity claim is void.**

**The wrapper `scripts/demo_attest.sh` (committed bash; `set -uo pipefail`):**
1. **[A]** `bash scripts/reproduce_trust.sh` → must print `REPRODUCE_TRUST: PASS` (rebuilds hex0→seed,
   self-host fixpoint `K2==K3==K4==0992dddd…`, gcc-DDC `K1==84363adb…`). Nonzero exit ⇒ abort. Record
   `SEED_SHA=9837db12…`, `K1_SHA=84363adb…`, `FIX_SHA=0992dddd…`.
2. **[B]** Build the GPT-2 forward **from that just-rebuilt seed** (GPU: re-mint PTX from `seed.bin`
   exactly as the capstone audit does — never a cached artifact; CPU: build from the same seed-minted
   `kovc`).
3. **[C]** Run the fenced importer → `gpt2_124M.weights`, run the forward, dump the Helix ladder
   (`helix_embed.bin … helix_logits.bin`, `helix_argmax`, `helix_tokens.json`) for the **same pinned
   prompt**. Delete every `helix_*.bin` + `/tmp/newdrv.bin` before the run and assert each is freshly
   non-empty after (staleness guard).
4. **[D]** Parity gate: every hidden/logit gate ≤ its bar AND `helix_argmax==ref_argmax` exactly AND
   `helix_tokens.json==ref_tokens.json` exactly. Any miss ⇒ `DEMO_ATTEST_FAIL`, exit 1.
5. **[E]** On all-green, write `attestation/gpt2_attest.txt` binding: **source** (`git rev-parse HEAD`
   + clean/dirty + the three anchors), **binary** (sha256 of `seed.bin`, the minted `/tmp/newdrv.bin`
   or CPU binary, and the PTX), **input** (sha256 of `model.safetensors` *computed live, not a
   hard-coded literal*, `gpt2_124M.weights`, `vocab.json`, `merges.txt`, the prompt string),
   **output** (sha256 of every `helix_*.bin`, the decoded string, the argmax, the per-gate worst-rel
   diff), **verdict** (verbatim PASS lines), and the **residual-disclosure block** (§2.2). Print
   `sha256sum attestation/gpt2_attest.txt` as the attestation digest. (No network signing service —
   the "signature" is the self-contained content hash + the owner's git-committed/optional offline
   `gpg --detach-sign`.)
6. **Final line** `DEMO_ATTEST_PASS`/`DEMO_ATTEST_FAIL` → process exit.

**Acceptance gate for the wrapper:** a clean run prints `DEMO_ATTEST_PASS`, exits 0, every hash
re-verifies; **and** a run with any single negative control active prints `DEMO_ATTEST_FAIL` and exits
nonzero (the wrapper is only trustworthy if it fails on a known-bad pipeline). Also run a **3–5 prompt
parity sweep** (argmax + token-sequence exact on each) before declaring the gate trustworthy; the
attestation records which prompt set was swept.

**Effort:** ~2–4 focused sessions (most discipline is *ported* from `capstone_audit.sh` /
`oracle_train.py`, not invented). **Dependencies:** the forward slices must expose the per-phase Helix
dumps in flat `<f4` + the per-op disable toggles (hard); P1 produces the weight file the attestation
hashes (hard). **Inbound:** *every* other phase depends on this one for its acceptance gate — nothing
ships until `demo_attest.sh` is green. **Risks:** f32 12-block accumulation may exceed 2e-2 on a
*correct* forward (measure-then-pin; lean on the exact argmax/token gate); comparator-as-blind-spot
(mitigated by self-tests + negative controls); pinned-prompt overfit (the 3–5 prompt sweep);
attestation staleness (rm-before/non-empty-after guards); no third-party reproduction of the *GPT-2*
leg yet (the CPU path is the route to closing it push-button; the trust-core leg is already CI-green).

---

### P7 — The investor demo (storyboard + runbook + DoD)

**Goal.** Wrap the artifacts into a runnable, honest, ≤3-minute demonstration with a mandatory
pre-recorded backup and a fallback decision tree.

**Pre-staged before the room (NOT shown as live work):** the from-raw ladder built once this session
(so live `reproduce_trust.sh` is a ~1 min *re-verify*, not a cold build); weights imported to the flat
file; a `PARITY: PASS` report on disk from the latest gated run; the oracle output captured; terminal
command history pre-loaded (operator presses ↑, never types); `runbook.json` dry-run passed.

**Storyboard — two windows (LEFT terminal, RIGHT one held slide), three beats:**

- **Beat A — "rebuild from 299 bytes" (~45 s):** run the committed, already-green
  `bash scripts/reproduce_trust.sh` → `REPRODUCE_TRUST: PASS`. Narrate the four printed checks; punch
  the three anchors on the slide (`seed 9837db12…`, fixpoint `0992dddd…`, gcc-DDC `K1 84363adb…`).
  *Switchable to a pre-recorded asciinema clip if the room's attention is short — it's CI-green on a
  clean runner, so "anyone can re-run this" stays true.*
- **Beat B — "a model you know, unchanged, runs on it" (~60 s, the hero shot):**
  `bash helix-llm/demo/run_demo.sh --prompt "The capital of France is" --new 12` (GPU default; `--cpu`
  for the purer path). Narrate: MIT-licensed `openai-community/gpt2`, 124 M params, downloaded
  unchanged, zero retraining; the forward is re-expressed in Helix and emitted by the compiler we just
  rebuilt from 299 bytes; on GPU the kernels are the capstone-proven ones; we import the weights, we
  don't touch them. Tokens stream; **must be coherent** (the oracle already produces coherent text).
- **Beat C — "matches a trusted reference, exactly, reproducibly" (~45 s, the punchline):**
  `bash helix-llm/demo/parity_demo.sh` → worst-case rel logit diff + `PARITY: PASS`; then run the hero
  command a second time piped to `sha256sum` → identical hash. Punchline slide: *"GPT-2, imported
  unchanged, generating text on a compiler rebuilt from 299 bytes — output-matched to an independent
  reference, and bit-for-bit reproducible."*
- **Close (~20 s):** the §2.1 spoken pitch.

**Files to create (all gitignored under `helix-llm/demo/`, fence-safe — they orchestrate the fenced
importer/oracle/weights):** `run_demo.sh` (selects GPU/`--cpu`, mirrors to ext4, calls the P4 CLI,
streams a stable hashable token stream), `parity_demo.sh` (runs oracle + Helix, diffs logits, prints
`PARITY: PASS`/`FAIL`, fail-closed), `RUNBOOK.md` (commands + narration + slide cues + timings +
fallback tree + pre-flight checklist + the honest-residuals card), `runbook.json` (machine-readable
step list for auto-play + demo-morning dry-run), `record_backup.sh` (asciinema cast + mp4 of the full
green run), `slides.md` (the 3 text-only slides with pinned hashes).

**CPU-vs-GPU variants + fallback ladder (RUNBOOK encodes it as a decision tree):** GPU = snappy live
default (residual: complete-to-PTX); CPU = purer-trust closer (no `ptxas`, bit-reproducible, slower —
lead with the hash, not wall-clock). Ladder: (1) GPU flaky → CPU live; (2) CPU too slow → single
next-token or pre-recorded CPU cast; (3) any live failure → cut to the **mandatory** pre-recorded
backup (from a gated green run, ≤3 min) without breaking narration; (4) **parity red on stage → do NOT
improvise a pass** — state "the gate caught a drift; we never ship red — that discipline is the
product," cut to pre-recorded green parity (converts failure into a trust proof); (5) projector-only →
mp4 + slides.

**Acceptance gate:** the §1.2 DoD (MVP items 1–7; full demo items 8–10). A timed dry-run ≤3 min, the
`runbook.json` dry-run green, the backup recorded from the frozen demo commit.

**Effort:** ~3–4 days. **Dependencies:** P4 (Beat B coherent text), P5 (GPU path) or P2/P3 (CPU path),
P6 (the parity/attestation it presents). **Risks (demo-specific):** live GPU failure (backup + CPU
fallback); demo-morning regression (freeze the branch the night before; `runbook.json` morning
dry-run); parity red on stage (D3 → trust proof); investor anchors on speed (pre-empt in the close —
name the cuBLAS-fraction residual *first*); overclaim slip (residuals scripted verbatim so they're
said by design); coherence underwhelms (pre-pick a prompt where 124 M is visibly coherent; narrate
that quality == GPT-2's by construction); `reproduce_trust.sh` modifies the tree (run on a throwaway
clone, pre-warm).

---

## 5. Consolidated risk register

L = likelihood, I = impact.

| # | Risk | L | I | Mitigation |
|---|---|---|---|---|
| **T1** | **GELU drift** — matching erf-GELU instead of HF `gelu_new` (tanh). #1 silent drift. | Med | High | Helix already uses the tanh form (`__gelu`/`gpu_gelu`) = `gelu_new` ✅. Block-0 gate (P2/P5) catches it before the full stack. Oracle uses `gelu_new` (verified). |
| **T2** | **Transpose / tie error** — wrong Conv1D orientation or `wte` not tied → garbage/subtle drift. Square `c_proj [768,768]` makes a wrong transpose silent. | Med | High | For the `mm_AB` engine the 4 Conv1D weights ship **un-transposed**; logits = `hidden @ wte^T`. P1 gate 3 (oracle cross-check on real attention) is the only thing that catches the square-matrix trap — mandatory. |
| **T3** | **Causal mask / LN-eps omission** → wrong logits. | Med | Med | Mask = softmax over valid prefix (CPU) / fold predicate into softmax (GPU); eps = `+1e-5` before rsqrt. Block-0 + per-kernel unit gates catch it. |
| **T4** | **Tiled-GEMM at GPT-2 dims / vocab-50304 padding** unconfirmed on real HW. | Med | Med | GEMM-at-scale reconfirmation gate (P5 gate 5) vs cuBLAS oracle; pad S to %64, vocab to 50304; capstone already proved the tiled kernels. |
| **T5** | **CPU-path perf for 124 M unmeasured** — could be minutes/token. | Med | Med | P3 measures sec/token; runbook fallback (single-token or pre-recorded CPU); CPU is the *trust* closer, not the speed beat. |
| **T6** | **fp32-only ceiling** (~1.5 B on 8 GB). | Low | Low | GPT-2 124 M (and XL 1.5 B) fit. Disclosed residual; irrelevant to 124 M. |
| **T7** | **WSL `/mnt` path mangling / DrvFs slowness** (`VAR=/mnt/...` comes out empty; DrvFs slow for byte I/O). | Med | Med | Literal paths / cd+relative in all scripts; mirror to ext4 at runtime; verify in the dry-run. |
| **T8** | **Taylor transcendental accumulation** (`__exp`/`__tanh`) over 12 layers (CPU). | Med | Med | Layer-appropriate tolerances; f64 variants (`__exp_f64`/`__sqrt_f64`) for accumulation-sensitive reductions if drift exceeds the bar. |
| **T9** | **Arena ceiling** forces a C harness for CPU weight movement (purest "zero C" not met by v1). | High | Low | Authorized fenced host-glue residual; pitch "arithmetic 100% Helix-from-raw, byte-movement is glue"; all-Helix sharded loader is a documented stretch goal. |
| **T10** | **Bias-presence mismatch** with the capstone layout (capstone has no biases; GPT-2 does). | Med | High | P1 contract defines bias slots explicitly; the CPU/GPU op layers already support biases; fix is purely file-layout. |
| **T11** | **Pretokenizer Unicode-class correctness** (`\p{L}/\p{N}` + lookahead). #1 tokenizer drift. | Med | High | Vendor PCRE2 into the fence (compiles the oracle pattern verbatim → identical by construction); tokenizer gate catches regressions. |
| **T12** | **f32 12-block accumulation may exceed the 2e-2 hidden-state bar on a correct forward.** | Med | Med | Measure-then-pin the float bar with documented justification; the **exact argmax + token-sequence** gate is the real product claim; never loosen the discrete gate. |
| **T13** | **Comparator-as-blind-spot** (a buggy comparator passes a broken forward). | Low | High | Comparator self-tests (self-compare=0, cross-tensor=FAIL, count-mismatch=FAIL) + negative controls; reject the comparator if it fails its mismatch test. |
| **D1** | **Live GPU failure on stage.** | Med | High | Mandatory pre-recorded backup; CPU live fallback; fallback ladder steps 1–3. |
| **D2** | **Demo-morning regression** breaks the green path silently. | Med | High | Freeze the demo commit the night before; `runbook.json` morning dry-run; backup from that exact frozen state. |
| **D3** | **Parity red on stage.** | Low | High | Never fake it — state the gate caught it (that's the product), cut to pre-recorded green parity. Failure → trust proof. |
| **D4** | **Investor anchors on speed.** | High | Med | Pre-empt in the close — name the cuBLAS-fraction residual first, reframe to verifiability. Never claim a speed win. |
| **D5** | **Overclaim slip** ("fully verified GPU" / "AGI" / "complete to machine code"). | Med | High | Honest-residuals card in RUNBOOK (DoD #7); residuals scripted verbatim in the close. |
| **D6** | **Coherence underwhelms** (124 M is small). | Med | Med | Pre-pick a coherent prompt; narrate quality == GPT-2's by construction (weights unchanged). |
| **D7** | **`reproduce_trust.sh` runs long / modifies tree** (rm's rung binaries, rewrites paths). | Med | Med | Run on a throwaway clone; pre-warm; if still long, switch Beat A to the asciinema clip (CI proves green). |

---

## 6. Demo runbook (the live ~2–3 minute script)

**Pre-flight (before the room):** clean throwaway clone; `reproduce_trust.sh` pre-warmed once; weights
imported (`gpt2_124M.weights` present, sha logged); latest `demo_attest.sh` run green with
`attestation/gpt2_attest.txt` on disk; `runbook.json` dry-run passed (every step hits its success token
within max-seconds); backup mp4/cast recorded from the frozen commit; terminal history loaded; two
windows arranged (LEFT terminal, RIGHT slide).

**Beat A (~45 s) — trust core.**
```bash
bash scripts/reproduce_trust.sh            # -> REPRODUCE_TRUST: PASS
```
Narrate the four checks; slide shows `seed 9837db12… · fixpoint 0992dddd… · gcc-DDC K1 84363adb…`.

**Beat B (~60 s) — the model runs (hero shot).**
```bash
bash helix-llm/demo/run_demo.sh --prompt "The capital of France is" --new 12          # GPU default
# closer / fallback:
bash helix-llm/demo/run_demo.sh --cpu --prompt "The capital of France is" --new 12     # CPU, no ptxas
```
Narrate: unchanged MIT GPT-2 124 M, zero retraining, forward re-expressed in Helix from the
just-rebuilt compiler; tokens stream coherently (e.g. "…Paris.").

**Beat C (~45 s) — parity + reproducibility (punchline).**
```bash
bash helix-llm/demo/parity_demo.sh         # -> worst-case rel logit diff + PARITY: PASS
bash helix-llm/demo/run_demo.sh --prompt "The capital of France is" --new 12 | sha256sum   # twice -> identical hash
```
Punchline slide held: *"GPT-2, imported unchanged, generating text on a compiler rebuilt from 299
bytes — output-matched to an independent reference, and bit-for-bit reproducible."*

**Close (~20 s):** the §2.1 spoken pitch (residuals named first, then reframe to verifiability).

**Optional full-demo extras:** show `bash scripts/demo_attest.sh` → `DEMO_ATTEST_PASS` and hand out
the one-page `attestation/gpt2_attest.txt`; show the CPU twin-hash for the live bit-reproducibility
claim.

**Fallback decision tree (operator memorizes):** GPU flaky → CPU live → too slow → single-token or
pre-recorded CPU → any failure → pre-recorded backup → parity red → "the gate caught it; we never ship
red" + pre-recorded green parity → projector-only → mp4 + slides.

---

## 7. Honest residuals + how each HARD CONSTRAINT is honored

| Constraint | How honored |
|---|---|
| **Claude-subscription only, no external AI APIs** | GPT-2 = static open weights we read and execute ourselves (not an API). No model call leaves the box. The numpy oracle is local. |
| **Python-free shipped toolchain (exactly 1 committed `.py`)** | Verified: `git ls-files '*.py'` = 1 (`verification/oracle/oracle_train.py`). The importer, tokenizer, parity comparator, negative controls, and demo scripts ALL live under gitignored `helix-llm/` (`.gitignore` is `*`) or are committed **bash**. The Route-A C importer + (later) in-Helix `.hx` tokenizer remove the last Python from the import/tokenize *steps* before any "Python-free toolchain" public claim. `.hx` and `.c` are not Python → commit freely. |
| **From-raw trust (rebuildable from 299 bytes)** | The forward runs on `kovc` minted from the 299-byte seed; `demo_attest.sh` step [A] re-proves `REPRODUCE_TRUST: PASS` live (hex0→seed, fixpoint `0992dddd…`, gcc-DDC `K1 84363adb…`). The CPU path adds **no** compiler-source change (no `kovc.hx`/lexer/parser edit) → the self-host fixpoint sha stays byte-identical. The GPU binary is minted from the just-rebuilt seed, never cached. |
| **`ptxas` boundary** | Disclosed: the GPU path is hand-auditable to PTX; below PTX NVIDIA's closed `ptxas`+driver+the C launcher are trusted-once. **The CPU path crosses no `ptxas` boundary** — it is the airtight artifact, and the attestation records which path produced it. |
| **fp32 ceiling (~1.5 B on 8 GB sm_86, single GPU)** | Disclosed; GPT-2 124 M fits comfortably. The weight-file `version` field reserves an fp16/bf16 evolution path. No cross-arch/multi-vendor claim. |
| **Never fake parity / never ship red / gated before commit** | Fail-closed at every gate (importer gate 3, block-0, full-logits, argmax+token-exact, per-kernel unit gates, comparator self-tests). Negative controls prove the gates bite (if a broken variant PASSES, the claim is void). `demo_attest.sh` propagates FAIL to process exit. On-stage parity red → state it, never improvise a pass. |
| **Strictly serial builds (one compiler/GPU build at a time)** | `gpt2_gpu_parity.sh` and `demo_attest.sh` run serial; no parallel `kovc`/GPU builds. |
| **WSL mirror to ext4, commit via Windows-native git** | All heavy `kovc`/inference/IO runs mirror `C:\...` (DrvFs) to ext4; the `C:\` copy stays the single source of truth. Literal/cd+relative paths inside `wsl.exe bash -c` (never `VAR=/mnt/...`). Commits (when authorized) via Windows-native git. |
| **Never read `C:/Projects/Neptune/api.env`** | Out of scope; not touched anywhere in this plan. |

**Net honest residuals carried into the attestation + pitch:** (1) fenced host glue (importer +
tokenizer + oracle, no compute-trust role); (2) the `ptxas` boundary on the GPU path only; (3) fp32-
only ⇒ ~1.5 B ceiling, single sm_86 GPU; (4) parity is fp32-vs-fp32 within a measured tolerance on
hidden states but EXACT on argmax+tokens (never conflate "matches the reference" with "reproduces
itself"); (5) the oracle is implementation-independent but spec-shared; (6) no third-party
reproduction of the GPT-2 leg yet (the CPU path is the route to closing it).

---

## 8. Critical-path / sequencing summary + milestone checklist

### Executed sequence (GPU-first; the brief's P-numbers map as noted)

```
P1 importer ─┬─> P5 GPU forward ──> P4 generation/CLI ──> ┌─ MVP demo (GPU-only) ─┐
             │        (de-risks importer + the 4 gaps)    │                       │
             └─> P2 CPU block ─> P3 CPU full ─────────────┘                       │
                                                                                  v
                                              P6 trust wrapper + attestation ──> P7 full demo
```

- **Critical path to MVP:** P1 → P5 → P4(gate 2) → P6(GPU parity + attest) → P7(MVP). The MVP is
  GPU-only and reachable without the CPU path.
- **Critical path to full demo:** + P2 → P3 (CPU bring-up, gated *after* GPU parity is green so it
  inherits validated weights/orientation) → P6(CPU twin-hash + attest) → P7(full).
- **Parallelizable now:** P4's tokenizer + loop + CLI plumbing + the `forward_iface.h` seam are
  forward-independent (publish the seam first); P1's Route-A C importer; P6's comparator + oracle
  ladder dump + negative-control harnesses (against the existing `ref_block0.npy`).
- **Shared single coupling point:** the P1 weight-file order/format + orientation contract, and the
  fenced oracle + `ref/*` dumps. The forward slices own their readers; P1 owns the contract; P6
  measures orientation end-to-end.

### Milestone checklist

- [ ] **M0 — Phase 0 (DONE):** weights + validated numpy oracle + `ref_block0.npy` present.
- [ ] **M1 — Importer GREEN:** `gpt2_124M.weights` written; round-trip bit-exact; **oracle
      cross-check ≤1e-6** (catches the square-`c_proj` trap); payload sha256 logged.
- [ ] **M2 — Forward seam published:** `forward_iface.h` committed-to-fence; P2/P3/P5 build against it.
- [ ] **M3 — Tokenizer GREEN (forward-independent):** C `tok_encode` id-for-id == oracle on the
      battery; `decode∘encode` round-trips (incl. UTF-8).
- [ ] **M4 — GPU per-kernel unit gates GREEN:** `gpu_softmax_causal` / `gpu_layernorm_fwd_eps` /
      `gpu_add_bias_rowbcast` each PASS vs CPU ref + negative control.
- [ ] **M5 — GPU block-0 anchor GREEN:** post-block-0 hidden vs `ref_block0.npy` < 1e-3 rel.
- [ ] **M6 — GPU full-logits GREEN:** argmax == oracle exactly; logit abs-diff < tol (or stated
      argmax+top-k fallback); GEMM-at-scale reconfirmed vs cuBLAS.
- [ ] **M7 — Generation GREEN:** `helix-gpt2 --backend gpu` greedy token-for-token == oracle; coherent
      text. **← MVP unblocked.**
- [ ] **M8 — CPU block-0 GREEN:** Helix CPU post-block-0 vs `ref_block0.npy` (max_abs<1e-3,
      mean_abs<1e-4).
- [ ] **M9 — CPU full + generation GREEN:** logit parity < 1e-2; argmax + token sequence exact;
      two-run byte-identical output; sec/token measured.
- [ ] **M10 — Trust wrapper GREEN:** `demo_attest.sh` → `DEMO_ATTEST_PASS`, every hash re-verifies,
      every negative control FAILS, 3–5 prompt sweep exact; `attestation/gpt2_attest.txt` emitted.
- [ ] **M11 — Demo ready:** `runbook.json` dry-run ≤3 min green; pre-recorded backup from the frozen
      commit; honest-residuals card in `RUNBOOK.md`. **← MVP DoD met.**
- [ ] **M12 — Full demo:** CPU live (or pre-recorded CPU) + CPU twin-hash bit-reproducibility +
      handed-out attestation. **← Full DoD met.**

---

### Appendix: unresolved conflicts flagged across the design sections

1. **`model.safetensors` sha256 literal.** Two slices cite `248dfc39…` as the pinned weights hash, but
   the committed capstone packet pins different anchors and no `248dfc39…` appears in the repo for the
   safetensors. **Resolution:** `demo_attest.sh` **computes the safetensors sha256 live** and records
   it in the attestation rather than hard-asserting a possibly-stale literal. Do not bake `248dfc39…`
   as a gate constant until it is re-derived from the actual file. *(The trust-core anchors
   `9837db12 / 84363adb / 0992dddd` ARE verified in-repo and are safe to assert.)*

2. **"Transpose four Conv1D weights" vs "ship un-transposed."** The ESTABLISHED FACTS and the CPU slice
   say "transpose exactly four weights"; the importer and GPU slices show the capstone `mm_AB` engine
   wants `[in,out]` = the HF Conv1D layout already, so **no physical transpose** is needed for that
   engine. **Resolution:** the importer ships the 4 weights **un-transposed for the `mm_AB` path** and
   exposes a per-tensor `transpose` flag (all `False`); the forward slices' GEMM-orientation choice is
   the single decision that sets the flags, and **P1 gate 3 measures the correct orientation
   end-to-end** (we measure, not hand-assert). The CPU slice's "transpose" note applies only if it
   chooses a `W@x` matvec; if it reuses `tf2d_matmul` as `x@W` like the GPU path, it is also
   un-transposed. Coordinate this one flag before locking the parity gate.

3. **Phase numbering vs build order.** The brief requests P1=importer, P2=CPU-block, P3=CPU-full,
   P4=tokenizer/gen, P5=GPU, but also recommends building GPU **first**. **Resolution:** kept the
   requested P-numbers for readability; §8 makes the executed critical path explicit (P1 → P5 → P4 →
   MVP → P2/P3 → P6 → P7). CPU ops are *authored* early (shared importer/oracle) but *bring-up* is
   gated after GPU parity is green. No technical conflict — only a numbering-vs-sequence note.

4. **CPU weight loading: C harness vs all-Helix.** The CPU slice's primary design uses a C harness
   (arena is ~20× too small to hold the model); a "zero C" all-Helix sharded loader is acknowledged as
   ugly/stretch. **Resolution (no conflict):** ship the C-harness streaming design as the v1 CPU path
   (authorized fenced host glue, byte-movement only); document the all-Helix loader as a stretch goal
   for a maximally-pure variant. The trust claim is precise: "arithmetic 100% Helix-from-raw; weight
   byte-movement is fenced host glue (no math)."
