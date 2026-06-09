# GPT-2 124M CPU forward — the no-ptxas, pure-Helix path (P2/P3)

The purest-trust artifact: GPT-2 124M forward re-expressed in pure Helix, compiled by the
self-hosted `kovc` (rebuildable from the 299-byte seed), running on the CPU with **no ptxas / no GPU
boundary**. All arithmetic stays in kovc-compiled Helix; the C harness does only byte-movement.

> **Canonical, committed location.** `gpt2_cpu_ops.hx` and `cpu_host.c` live here under
> `helixc/runtime/` (alongside the committed GPU launcher `gpt2_infer.c`) so the CPU no-ptxas path is
> **reproducible from a clean clone**. Only the model weights (`helix-llm/models/gpt2/…`) and the numpy
> oracle reference (`helix-llm/ref/ref_block0.npy`) remain fenced under `helix-llm/` (large /
> independent-verification artifacts). `cpu_host.c` is a Category-B host harness (CUDA-FREE, zero
> arithmetic on the trust path); `gpt2_cpu_ops.hx` is a `.hx` and is not counted by the `.c`/`.h` fence.

## Files
- `gpt2_cpu_ops.hx` — the pure-Helix op-dispatch ELF source (ALL arithmetic). Two genuinely-new ops
  (`gpt2_layernorm_affine`, `gpt2_causal_softmax_row` with the 0.125 scale folded in) plus matmul /
  matmul+bias / GELU (gelu_new tanh, overflow-safe via tanh saturation) / residual-add. f32 lives as
  raw bits in i32 arena slots via `__bits_of_f32`/`__f32_from_bits` (same as stdlib `tensor.hx`).
- `cpu_host.c` — the CUDA-free byte-movement harness (the `train_transformer.c`/`gpt2_infer.c` twin
  minus CUDA). mmaps the P1 `gpt2_124M.weights`, does host embedding gather, multi-head pack/scatter,
  GEMM N-tiling, and per-op file staging. NO arithmetic on the trust path beyond the embedding add.
- `../../scripts/gpt2_cpu_parity.sh` — the fail-closed gate (mint kovc from the raw seed → compile the
  ops → build the harness → run block-0 → `GPT2_CPU_BLOCK0_PARITY_PASS`/`FAIL` + exit code).

## The staging mechanism (the chosen design)

The hard constraint: `kovc` programs have a single fixed arena (`helix_arena_cap()` = 6,291,456 i32
slots, ~25 MB). The 474 MB weights, and even one block's c_fc weight (2.36 M floats), do not fit; and
`read_file_to_arena` packs **1 byte per slot** with a hard 1 MB / 256K-float `ud2` truncation trap,
while bootstrap `main()` has no argv (paths are baked literals).

**Choice: a single pure-Helix op-dispatch ELF, driven tile-by-tile by the C harness via baked-path
file round-trips** (the `drivers/driver_k1input.hx` pattern). Per op the harness writes ONE input file
`/tmp/gpc/in.bin` = a 6×i32 LE header `[op, d0, d1, d2, d3, d4]` immediately followed by the input
f32 tile(s) as little-endian raw bytes; runs the op ELF (its `main()` reads `in.bin`, reassembles each
LE f32 from its 4 bytes — `b0 + b1*256 + b2*65536 + b3*2^24`, i32 wraparound reproduces the exact
2's-complement bit pattern, then `__f32_from_bits` — computes, and writes the output f32 tile to
`/tmp/gpc/out.bin`). Big GEMMs (c_attn N=2304, c_fc N=3072, mlp_proj/c_proj) are tiled over the N
(output-column) dimension at `NTILE=64` cols/call so each invocation's staged input
(A[T·K] + W[K·Nblk] + bias[Nblk]) stays under the 256K-float read buffer and the arena cap; the tied
LM head is tiled over the vocab at `VTILE=512`. Every tile's arithmetic runs in the kovc-emitted ELF.

Why this and not the alternatives: an all-Helix `read_file_to_arena_dyn` loader cannot ingest the model
(byte-per-slot 4× inflation + the 1 MB buffer); holding a whole weight matrix in the arena is blocked
by the same read cap. Harness-tiled file staging keeps the trust claim crisp — *arithmetic is 100%
Helix-from-raw; weight byte-movement is fenced host glue (no math)* — at the cost of a process spawn
per tile (the dominant perf cost; see below).

## Results (verified 2026-06-08, weights = P1 `gpt2_124M.weights`, prompt ids [464,3139,286,4881,318])

- **Block-0 parity: PASS** — `max_abs = 1.144e-04` (gate < 1e-3), `mean_abs = 9.562e-07`
  (gate < 1e-4), 0 non-finite. Bit-reproducible across two runs (identical sha256).
- **Full 12-layer forward + ln_f + tied head: PASS** — helix argmax = 262 = oracle argmax (exact);
  last-row logit `max_abs = 2.75e-04` (gate < 1e-2).
- **Greedy generation: token-for-token MATCH** — helix `464 3139 286 4881 318 262 3139 286 262 4141`
  == oracle (prompt5 + gen5).
- Trust: seed sha `9837db12…` (anchor), K2 self-host fixpoint sha `0992dddd…` (unchanged — no
  kovc.hx/lexer.hx/parser.hx edit). NO ptxas, NO GPU.

## Honest perf (the disclosed residual)
Naive scalar triple-loop matmuls in interpreted-from-raw Helix, one OS process spawn per op-tile:
block-0 ≈ 8 s; one full 12-layer forward ≈ 111 s; greedy generation ≈ **~130 sec/token** (650 s for
5 tokens over a growing context). Slow by design — the product is bit-reproducible verifiable
execution from the first byte, not speed (the GPU path is the speed story). Knobs: `HX_NTILE`,
`HX_VTILE` raise tile sizes to cut spawn count (bounded by the 256K-float read buffer).

## Reproduce
```
bash scripts/gpt2_cpu_parity.sh    # fresh mint from the raw seed; -> GPT2_CPU_BLOCK0_PARITY_PASS
```
Build runs on WSL ext4 (`CPU_BUILD_DIR`, default `~/gpt2cpu/bs`) to avoid DrvFs 1-byte-write slowness.
