# Helix × a MODERN model — Llama-arch on the verified stack (AUTHORED; NEEDS GPU BUILD + GATE)

**Status: AUTHORED in Cowork (Claude Fable 5, 2026-06). NOTHING here is claimed to run yet.**
The kernels compile-test against a from-raw-built kovc (CPU-only PTX emission — see §5 for what was
and was NOT verified); every GPU-execution and parity claim is **deferred to the Claude Code build +
gate pass (Opus)**. This plan deliberately mirrors the GPT-2 demo's structure so the same honesty
machinery (fail-closed gates, independent oracle, residuals card) carries over unchanged.

**UPDATE 2026-06-09 (Opus, Claude Code) — G-L0 PASS (all 3 kernels).** Compiled via the seed-built
from-raw kovc, ptxas-accepted at sm_86, and matched the numpy oracle to <2e-6 (rmsnorm 3.8e-6, rope
2.4e-7, silu_mul 1.9e-6); the per-kernel negative-controls all bit. One real kovc codegen bug was
found + fixed in `gpu_silu_mul`: a `let mut amag = gi; if g<0 { amag = 0-gi }` aliased `amag` onto
`gi` and clobbered it to `|gi|` for g<0, so silu emerged as -reference on every negative-g element
(G-L0 caught it at max-abs ~1.9; compile + ptxas had passed it silently). Rewritten to
`gpu_gelu_stable`'s constant-seeded `sgn` idiom (only `sgn` mutable, `gi` pristine). Gate:
`scripts/llama_ops_parity.sh` (from-raw compile -> ptxas sm_86 -> parity vs oracle + mutate
negative-controls). STILL REMAINING (Code, gated): G-L1 block-0, G-L2 logits + token-for-token (need
the host wiring + a SmolLM2-135M import), G-L3 regression sweep, G-L4 fence accounting.

**UPDATE 2026-06-09 (Fable 5 builder run, branch fable/demo-complete) — G-L1 + G-L2 PASS; the
full model runs.** `scripts/llama_model_gate.sh` (fail-closed, first run, 72 s): SmolLM2-135M
imported by the additive `gpt2_pack --arch llama` (BF16 bit-shift widening; HXGW v2 header carries
arch/NKV/rope_theta/rms_eps; GPT-2 repack regression byte-identical sha c661e224) and run by the
additive `--arch llama` path in gpt2_infer.c (v2-header self-config; separate q/k/v A.Bt GEMMs on
untransposed HF [out,in] weights; GQA host mapping; host-built RoPE tables; SwiGLU; tied head):
- **G-L1** post-layer-0 residual: max-abs **3.2e-05** (tol 2e-3) — PASS
- **G-L2a** full-model last-row logits: argmax **EXACT** (260), max-abs **4.9e-05** over 49,152 — PASS
- **G-L2b** 20-token greedy: **TOKEN_FOR_TOKEN_MATCH 25/25** vs the independent oracle — PASS
- negative control (corrupted weights) correctly FAILED — the comparator has teeth.
The full-model oracle is `helix-llm/tools/llama_numpy_ref.py` (uncommitted; reads the ORIGINAL HF
safetensors — independent of both the importer and the GPU path; its own greedy run produces
coherent text, sanity-validating every pinned convention).
**G-L4 fence accounting: ZERO new committed .c/.h/.py** — both host changes are additive
extensions of existing Category-B files (gpt2_pack.c, gpt2_infer.c); the fence stays 1 .py /
29 .c-h; the self-host fixpoint inputs are untouched. G-L3 (full GPT-2 serve-gate regression on
the changed worker) run in the same session — see scripts/_serve_regate3.log. Serve integration
(model switcher, §6.6): gpt2_serve_http.c gained an additive second-model worker + models[] in
/api/health + per-request model routing with ONE cross-model GPU mutex; gated by
scripts/llama_serve_smoke.sh. PENDING: the independent Opus re-gate before merge to main.

## 0. The pitch extension (why this is the headline long-shot)

GPT-2 (2019) proved the stack. A **current-generation, Apache-2.0, Llama-architecture model running
token-for-token-verified on a compiler rebuildable from 299 bytes** turns "we verified a classic
model" into "we verify the architecture family that powers today's open models." Same trust thesis,
zero new trust surface: the 3 new kernels below use only already-gated emitter features — **no
kovc.hx change, fixpoint `0992dddd` untouched.**

## 1. Target models (both Apache-2.0, both fit the 8 GB sm_86 box at fp32)

| | SmolLM2-135M (FIRST TARGET) | TinyLlama-1.1B (the scale flex) |
|---|---|---|
| License | Apache-2.0 (HuggingFaceTB) | Apache-2.0 |
| Params / layers | 135M / 30 | 1.1B / 22 |
| d_model / d_ff | 576 / 1536 | 2048 / 5632 |
| Heads (q / kv) | 9 / 3 (GQA ×3) | 32 / 4 (GQA ×8) |
| head_dim | 64 | 64 |
| rms_norm_eps | 1e-5 | 1e-5 |
| rope_theta | 1e5 | 1e4 |
| Vocab | 49152 | 32000 |
| fp32 weights | ~540 MB | ~4.4 GB |
| Why | smallest honest Llama-arch; fastest gate loop | "1.1B modern model" headline; same kernels |

(Verify every figure above against each model's `config.json` during the import step — the gate must
read dims from config, never trust this table.)

## 2. The 4 new ops → exactly 3 new kernels + 1 host rule

Authored, committed in this branch under `helixc/examples/` (`.hx` files are OUTSIDE the .c/.h
fence; they add nothing to the trusted-C count):

1. **`gpu_rmsnorm_fwd_eps_kernel.hx`** — RMSNorm, eps baked 1e-5 (the `gpu_layernorm_fwd_eps`
   precedent). One thread per row.
2. **`gpu_rope_rot_kernel.hx`** — RoPE, **HF rotate_half convention pinned** (pairs (i, i+half)),
   in-place on a packed per-head buffer, host-precomputed cos/sin tables (kovc has no sin/cos
   intrinsic and none is added; tables are data, like weights).
3. **`gpu_silu_mul_kernel.hx`** — SwiGLU's elementwise gate `y = u * silu(g)`, overflow-safe
   sigmoid (the `gpu_gelu_stable` idiom). The two SwiGLU GEMMs reuse `tiled_matmul`.
4. **GQA = zero kernels.** In the per-head host loop (the gpt2_infer design), grouped-query
   attention is pure host indexing: `kv_head = q_head / (n_q / n_kv)` when packing K/V. The mapping
   is pinned in the oracle (`gqa_kv_head`).

**Reused unchanged from GPT-2:** `tiled_matmul`, `tiled_matmul_abt`, `gpu_softmax_causal`,
`gpu_scale_rt` (1/sqrt(64) = 0.125, same head_dim), `vector_add` (residuals). **Dropped for Llama:**
both bias kernels (Llama has no biases), `gpu_gelu_stable` (replaced by silu_mul),
`gpu_layernorm_fwd_eps` (replaced by rmsnorm). Total kernel set: **8 again** (5 reused + 3 new).

Per-layer op sequence (the telemetry ticker's future llama map, 15 ops/layer):
`rms_1 → q_gemm → k_gemm → v_gemm → rope_q → rope_k → attn_scores → attn_scale → attn_softmax →
attn_av → attn_proj → attn_residual → rms_2 → [gate_gemm + up_gemm + silu_mul + down_gemm] →
mlp_residual` (the SwiGLU block collapses to ticker labels `fc_gate`, `fc_up`, `silu_mul`,
`proj_down`).

## 3. The independent oracle (DONE, runs green in Cowork)

`helix-llm/tools/llama_ops_numpy_ref.py` — **uncommitted under gitignored `helix-llm/`,
preserving the exactly-1-committed-.py fence** (same policy as the GPT-2 oracle). Pins all four
conventions (population-mean RMSNorm; HF rotate_half RoPE incl. the inv_freq table builder;
overflow-safe silu_mul; the GQA integer mapping) and self-tests fp32-vs-fp64 + rotation/identity/
HF-equivalence properties: `LLAMA_OPS_REF_SELFTEST: PASS` (run 2026-06-09 in the Cowork sandbox).
The full-model oracle (end-to-end forward) extends it the way `gpt2_numpy_ref.py` does — load the
safetensors in numpy, run the reference forward, dump logits + greedy ids.

## 4. Importer + host wiring (DESIGN ONLY — every .c change is fence-relevant)

- **Importer:** a `gpt2_pack.c`-style flat `.weights` converter for Llama checkpoints. Options:
  (a) extend `gpt2_pack.c` additively (no new committed file, count stays 29) — RECOMMENDED;
  (b) a new `llama_pack.c` (**+1 to the 29 fence — needs Opus's honest fence accounting**).
  Either way: safetensors → contiguous fp32 in layer-stream order, dims read from config.json,
  byte-exact gate vs a reference dump.
- **Worker:** `gpt2_infer.c` is dimension-generic but GPT-2-shaped (fused qkv, biases, gelu, LN).
  The Llama forward needs: split q/k/v GEMMs (GQA shapes), the rope step after q/k packing, the
  no-bias path, rmsnorm calls, and the SwiGLU MLP. RECOMMENDED: an additive `--arch llama` branch
  reusing `device_init`/`alloc_buffers`/`upload_layer` machinery + the same emit hooks (telemetry
  events unchanged — the SSE contract carries over verbatim, `n_layers` etc. already dynamic).
  Worked patch sketch maintained alongside the kernels; **applied + built only in Claude Code.**
- **cos/sin tables:** host computes once per `--max-ctx` at startup (double precision, cast f32),
  uploads as two device buffers, slices per current S.

## 5. What WAS verified in Cowork vs what REMAINS

**Verified here (CPU-only, no GPU) — transcript: deliverables `ptx_compile_test.log`:**
- The from-raw ladder was rebuilt in a clean /tmp clone of this branch (hex0→…→seed; the rebuilt
  `seed.bin` sha-matches the committed `seed.sha256`, 62,467 B). The k1ptxdrv driver was then built
  with that seed and used to emit PTX.
- **Calibration:** the 8 known GPT-2 kernels emit **PTX 44019 B / 8 `.entry`** — byte-size-identical
  to the GPU machine's gated serve run (`scripts/_gate_run.log`: “PTX 44019 B, 8 .entry kernels”),
  so this sandbox flow reproduces the production emission exactly.
- **The 3 new kernels:** 8 + 3 emit **PTX 50757 B / 11 `.entry`** — `gpu_rmsnorm_fwd_eps`,
  `gpu_rope_rot`, `gpu_silu_mul` all lower to real PTX bodies (66–79 lines each, correct
  `.param .b64×ptrs + .u32` signatures). sha256(base8) `a846f314…`, sha256(llama11) `665c78af…`.
- The numpy reference self-tests green (above): `LLAMA_OPS_REF_SELFTEST: PASS`.

**NOT verified (NEEDS GPU BUILD + GATE IN CLAUDE CODE — do not claim before):**
- ptxas acceptance at sm_86; numerical parity kernel-vs-oracle; full-model token-for-token vs the
  numpy forward; VRAM residency; serve integration; any speed number.

## 6. Fail-closed gate plan (mirrors the GPT-2 legs 1:1)

1. **G-L0 kernel parity** (new `scripts/llama_ops_parity.sh`): each new kernel vs
   `llama_ops_numpy_ref.py` on random tensors at model dims — max-abs tolerances of the same order
   as the GPT-2 ops (~1e-4 fp32); fail-closed.
2. **G-L1 block-0 parity**: layer-0 hidden state vs the full-model oracle (the `--block0` pattern).
3. **G-L2 logits + token-for-token**: argmax exact + N-token greedy == oracle (the
   `gpt2_scale.sh` pattern), pinned prompt, PRIMARY-mode required.
4. **G-L3 regression sweep**: ALL existing GPT-2 gates stay green (same binaries; additive only);
   `reproduce_trust.sh` anchors byte-identical (the kernels don't touch the fixpoint inputs, but
   gate anyway).
5. **G-L4 fence accounting** (Opus): .c/.h count delta declared honestly (0 if option (a) importer
   + inline `--arch` branch; otherwise each +1 named).
6. Only after G-L0..L4: wire into the chat demo (model switcher entry via the /api/health
   `models[]` capability — frontend already supports it, hidden until advertised).

## 7. Honest-residuals additions for the Llama leg

State unprompted, alongside the existing card: the oracle again shares the architecture *spec*;
rope tables are trusted-once host data (like weights); SmolLM2-135M is a small modern model, not a
frontier one (TinyLlama is the scale flex, same kernels); fp32-only still bounds scale; "modern
architecture, verifiably executed" is the claim — NOT "modern capability."

**UPDATE 2026-06-10 (Fable 5, branch fable/demo-agentic) — the INSTRUCT leg: a chat-capable
model, gated.** SmolLM2-360M-Instruct (Apache-2.0; 32L / 960 / 15:5 GQA / head_dim 64 / dff 2560 /
theta 1e5 / eps 1e-5 / tied) runs on the SAME gated kernels with ZERO kernel or arch changes —
the additions are tokenizer + protocol only: ChatML special-token encode (gpt2_tok.c
`encode_bytes_special`, opt-in per worker via HX_SPECIALS; mirrored in the oracle's
`encode_special`) and an eos-stop (HX_EOS; pinned convention: append `<|im_end|>` then stop).
Gate results (`scripts/llama_model_gate.sh`, model-parameterized via LLAMA_MODEL_D + LLAMA_CHAT=1,
TEMPLATED chat prompt, 121 s): **G-L1** layer-0 max-abs **2.4e-04** (tol 2e-3, T=37);
**G-L2a** logits argmax EXACT; **G-L2b** greedy **TOKEN_FOR_TOKEN_MATCH 45/45** — the verified
output is a genuine assistant answer ("The capital of France is Paris.<|im_end|>"), generation
stopping at eos exactly like the independent oracle; corrupted-weights negative control FAILED
correctly. Serve integration: the server now takes a third model slot (--model3/--specials3/--eos3);
`scripts/llama_serve_smoke.sh` adds leg [7] — the templated conversation over real HTTP matches
the oracle TOKEN-FOR-TOKEN including the C tokenizer's special-token parity and the eos-stop
(`CHAT_TOKEN_FOR_TOKEN_OK`, full smoke PASS 64 s, 3 models READY). Honest framing: 360M is a
SMALL instruct model — "real chat, verifiably executed", not frontier capability. Fences: the
.py/.c-h counts are unchanged (all host edits are to existing Category-B files). PENDING: the
independent Opus re-gate + honesty audit before merge.
