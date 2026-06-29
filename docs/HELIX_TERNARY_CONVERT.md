# Helix verifiable ternary conversion (SmolLM2-135M)

**Branch:** `v1.9-ternary` &nbsp;|&nbsp; **Fixpoint:** `31e6cc27` (`helixc/bootstrap/kovc.hx` UNCHANGED) &nbsp;|&nbsp; **GPU:** RTX 3070 Laptop (sm_86), CUDA 12.8 ptxas

This documents the **convert → run → verify** flow that turns a full-precision (fp32) SmolLM2-135M
into a **ternary** model ({-1, 0, +1} weights + a per-row scale), runs one of its linears on the
Helix from-raw-trusted ternary GPU kernel, and binds the whole thing with two re-derivable,
tamper-evident receipts.

> ### Read this first — what this is and is NOT
>
> **The deliverable is VERIFIABILITY, not quality and not compression.** This flow proves, on a
> compiler trusted from raw bytes, *exactly which* ternary weights were produced from *exactly which*
> fp source, that they **run bit-exactly** on the Helix ternary kernel, and *what held-out quality
> they actually have* — all cryptographically bound so the claim cannot be forged or silently edited.
>
> **It does NOT make the model good, and the conversion is NOT lossless or near-lossless.** On the
> proven run the held-out perplexity goes from **fp 8.75 → ternary 1140.55 — a 130× gap.** That is a
> large, honestly-measured loss of quality, not a rounding error. See [Honest limits](#honest-limits)
> before citing any number here. The moat is that the 130× is *measured, bound, and re-derivable* —
> not that it is small.

---

## 1. What the flow does

Three stages, each with its own committed script and its own pass token.

| Stage | What it does | Driver script | Pass token |
|-------|--------------|---------------|------------|
| **CONVERT** | fp32 → ternary via straight-through-estimator QAT + knowledge distillation; persists the converted latent fp32 weights | `scripts/llama_kd_conversion.sh` | `LLAMA_TRAIN_FT_PPL_DONE` (+ saved `.best`) |
| **RUN** | packs the converted weights to the 15-trit kovc format and executes one linear on the real ternary GPU kernel | (inside the verify driver) | element-exact match |
| **VERIFY** | binds source-fp + ternary-Merkle + measured ppl gap in a conversion receipt, and the kernel output in a Freivalds inference receipt; both re-derive and have tamper negative-controls | `scripts/llama_convert_certify.sh` | `LLAMA_CONVERT_CERTIFY_PASS` |

### CONVERT — `helixc/runtime/llama_train.c` (env-gated)

The trainer carries the fp32 weights as **latent** weights and, when `HX_TERNARY_QAT=1`, inserts a
**ternarize-dequant** in the forward of the 7 linears per layer (q, k, v, o, gate, up, down):

- per-output-row scale `sc[r] = mean(|W[r,:]|)` (the BitNet b1.58 abs-mean scale),
- `Wt[r,k] = clip(round_half_away_from_zero(W[r,k] / sc[r]), -1, 1) * sc[r]`,
- consumed by the matmul **in place of** `W`; the backward uses a straight-through clip-mask; **Adam
  still updates the latent fp32 weights**. Norms and the tied embedding stay fp32.

With `HX_KD=1` a **fixed fp teacher** (the same SmolLM2 with QAT off) supplies soft targets: the
loss-root gradient changes from `(student_softmax − onehot)` to `(student_softmax − teacher_soft)`
via the `gpu_kd_softmax_grad` kernel; the entire rest of backward + Adam is unchanged. The teacher
distribution is computed once per training sequence and cached in host RAM.

`HX_SAVE=<path>` writes the resident latent fp32 weights back in the **same v2 llama file layout** the
loader reads, and also snapshots the **best held-out checkpoint** to `<path>.best` (early-stopping is
captured automatically). Every one of these is **opt-in and default OFF**, so the fp baseline and the
fixpoint gate are byte-for-byte unaffected when they are unset.

### RUN — `gpt2_pack.c` packer + `scaled_packed_ternary_matmul` kernel

- `helixc/runtime/gpt2_pack.c --ternary-packfile` ternarizes a raw fp32 weight matrix and packs it to
  the kovc kernel format: **15 trits per i32 word** (base-4 codes, K zero-padded to a multiple of 15)
  + a per-row f32 abs-mean scale. This is the committed, self-tested `ternary_quantize_tensor`.
- `helixc/examples/scaled_packed_ternary_matmul_kernel.hx` is the inference primitive: an integer
  trit·activation dot product (on-device base-4 unpack, exact i32 accumulation) followed by **one**
  per-row `* sc[row]`. Because the activations are integers and there is no FMA in either path, the
  GPU result is **bit-identical** to the host `(float)int_dot * scale` reference.

The kernel is emitted by the from-raw kovc compiler and gated **byte-identical** to the committed
`scaled_packed_ternary_matmul_kernel.ref.ptx`.

### VERIFY — two receipts (`scripts/convert_receipt_llama.py` + the Freivalds path in `cuda_launch.c`)

1. **Conversion receipt** binds, in one certificate whose own sha256 covers every field:
   - `source_fp_sha256` — sha256 of the original fp `.weights` bytes,
   - `converted_latent_sha256` — sha256 of the converted latent `.weights` bytes,
   - `merkle_ternary` — Merkle root over the **210** ternarized linears (7 × 30 layers); each leaf =
     `sha256(name | per-row scale f32 | packed 15-trit i32 words)`, i.e. the **exact bytes the kernel
     consumes**,
   - `fp_loss` / `converted_loss` / `delta_loss` — the **measured held-out perplexity** (8.7467 →
     1140.55). `--emit` writes it, `--check` re-derives it byte-identical, and three tamper
     negative-controls (mutate the Merkle, mutate the loss, mutate the source sha) each force
     `CONVERT_RECEIPT_FAIL`. **The quality number is part of what the certificate sha binds** — you
     cannot edit "1140.55" down to "9.0" and still pass.
2. **Inference receipt (Freivalds)** over the gate-L0 ternary matmul `C = W·X` over a prime field
   (`p = 2^31−1`, `t = 2`, soundness `2^-62`): a genuine receipt re-derives to `RECEIPT_PASS`; a
   forged output is caught (`REJECT=CHECK2 → RECEIPT_FAIL`).

---

## 2. Reproduce

All commands run under WSL with `<=80%` compute (`taskset -c 0-5 nice -n 10`), single serial GPU job,
CUDA 12.8 ptxas, sm_86. Repo root assumed at `/mnt/c/Projects/Kovostov-Native`. Shell scripts are
CRLF-stripped to `/tmp` before running where noted.

### 2.0 Prerequisite (once): mint the from-raw kovc PTX driver

The convert and verify steps reuse a from-raw-minted kovc PTX driver (kovc.hx is **never** edited —
the kernel set is just the concat list fed to the already-minted driver). Mint it once via the
trainer backward gate, which also proves the trainer itself:

```bash
wsl.exe bash -lc "tr -d '\r' < /mnt/c/Projects/Kovostov-Native/scripts/llama_train_bwd_gate.sh > /tmp/g.sh && REMINT=1 bash /tmp/g.sh"
# -> LLAMA_TRAIN_BWD_GATE_PASS
# (forward regression + finite-diff gradient check + fp/QAT train smokes; mints $HOME/gpt2_ext4/llama_kovc_drv.bin)
```

### 2.1 Stage the corpus (MMLU, MIT/open — no download, token ids only)

```bash
scripts/llama_stage_mmlu.sh professional_psychology 256 42 14
# -> STAGE_MMLU_DONE ; writes helix-llm/mmlu_train_ids.txt / _lens.txt / _heldout_ids.txt
#    (DISJOINT by row index between train and held-out; only token COUNTS are printed, never text)
```

### 2.2 CONVERT (fp → ternary, QAT + KD, persist best checkpoint)

```bash
scripts/llama_kd_conversion.sh      # EPOCHS / KDT overridable via env (defaults EPOCHS=5 KDT=1)
```

This mints the 21-kernel KD PTX (20 trainer kernels + `gpu_kd_softmax_grad`), builds the trainer,
runs a KD-gradient self-check, then runs the conversion. The proven 130× best used an **extended
13-epoch** run with `HX_SAVE` (early-stopped on held-out); the raw invocation:

```bash
HX_TERNARY_QAT=1 HX_KD=1 HX_KD_T=1 HX_PPL_EVERY=1 HX_SAVE=$HOME/smollm2_ternary_converted.bin \
  taskset -c 0-5 nice -n 10 stdbuf -oL $HOME/llama_train .m1probe/llama_train_kd.ptx \
  helix-llm/models/smollm2-135m/smollm2-135m.weights $HOME/ids5.txt \
  --train-corpus helix-llm/mmlu_train_ids.txt helix-llm/mmlu_train_lens.txt 13 \
  --ppl helix-llm/mmlu_heldout_ids.txt 256
# -> LLAMA_TRAIN_FT_PPL_DONE ; best held-out checkpoint saved to $HOME/smollm2_ternary_converted.bin.best
```

> **Ship the `.best` checkpoint (epoch 10, ppl 1140.55), NOT the final epoch** — the held-out ppl
> turns over and overfits past the minimum (epoch 13 = 1900.39 on this tiny corpus).

### 2.3 RUN + VERIFY (pack to 15-trit, run on the real kernel, emit + re-derive both receipts)

```bash
HELIX_SRC=/mnt/c/Projects/Kovostov-Native \
  CONVERTED_WEIGHTS=$HOME/smollm2_ternary_converted.bin.best \
  bash scripts/llama_convert_certify.sh
# -> LLAMA_CONVERT_CERTIFY_PASS
```

This single driver does all four verify sub-steps: [1] pack footprint + C-vs-numpy packer parity,
[2] gate-L0 through the real kernel == trainer fake-quant (element-exact) + comparator NC, [3]
conversion receipt `--emit` / `--check` + tamper NC, [4] Freivalds inference receipt + forged-C NC.

### Convenience: chain all stages

`scripts/ternary_convert_run_all.sh` is a thin orchestrator that runs the prerequisite check →
stage → convert → certify in order, stopping at the first failure. It adds no new compute and edits
nothing — it just sequences the four committed scripts above. The **canonical entry points remain**
`scripts/llama_kd_conversion.sh` (convert) and `scripts/llama_convert_certify.sh` (verify); use the
wrapper only if you want one command for the whole chain.

---

## 3. The proven result (SmolLM2-135M)

Real numbers, reproduced across multiple artifacts (`scripts/llama_kd_extended_result.txt`,
`scripts/llama_convert_receipt_result.txt`). **State these exactly — do not round 130× to "close".**

| Quantity | Value |
|----------|-------|
| fp32 held-out perplexity (QAT off, MMLU held-out) | **8.7467** |
| Converted-ternary held-out perplexity (best, epoch 10, KD) | **1140.55** |
| **Quality gap** | **130.4× (1140.55 / 8.7467) — NOT near-fp** |
| Hard-CE best (no KD), for comparison | 2302.24 (263.2×) — KD narrowed 263× → 130× |
| Weight compression (210 linears, 15-trit + per-row scale vs fp32) | **14.49×** (29,306,880 B vs 424,673,280 B) |
| Round-trip max-abs diff of the packer | 0 |
| Kernel vs trainer fake-quant (gate L0, full scaled path) | **element-exact**, max_abs_diff = 0, 0/12288 mismatches |
| Kernel PTX vs committed `.ref.ptx` | **byte-identical** (sha `9ddc0fcf…`) |
| Conversion receipt `--emit` then `--check` | **re-derives byte-identical** → `CONVERT_RECEIPT_PASS`; 3 tamper NCs → FAIL |
| Freivalds inference receipt | genuine → `RECEIPT_PASS`; forged-C → `RECEIPT_FAIL` |
| Fixpoint | `31e6cc27` (kovc.hx unchanged); all gates green |

**Receipt provenance (epoch-10 best, from the certify record):**

- `source_fp_sha256 = 427397322d0d58a5dcae7d04d0e79358b30fc9c380616af921e68b7bebee1b4e`
- `merkle_ternary   = ccf706658114aaa82d2dcf2b16ee938e3eefdc106922ebbc6eb71e89d2110a5a`
- conversion certificate sha `05d4472c…`; inference receipt file sha `3a89e0ce…`

The KD recovery trajectory is monotone-then-noisy: raw ternary ≈ `2.59e9` ppl → ~2679 (epoch 1) →
**1140.55 (epoch 10, the minimum)** → rises again (overfit). The minimum, not the endpoint, is the
artifact.

---

## 4. Honest limits

These are stated plainly because honesty *is* the deliverable. None of them is softened.

1. **Quality: a laptop QAT fine-tune does NOT reach fp parity, and the gap is large.** The converted
   ternary model is **130× worse** in held-out perplexity than the fp32 source (8.75 → 1140.55). Near-fp
   ternary needs **pretraining-scale data and compute** — BitNet-class ternary models are trained from
   scratch on the order of **trillions of tokens**. The 130× gap is the honest ceiling for an **8 GB
   laptop** fine-tuning on a **~4.3K-token corpus**. Knowledge distillation genuinely helped (it
   narrowed the gap from 263× under hard-label cross-entropy to 130×, ~1.8× better, and changed the
   failure mode from immediate overfit to a real recovery curve), **but the gap to fp remains large.**
   This is **not** a near-lossless or near-fp conversion, and nothing here should be read as one.

2. **Size: ~135M parameters is about the ceiling for this trainer on 8 GB.** The trainer saves **all
   activations** (no activation checkpointing is implemented), so memory scales with model size.
   Larger models need a model download **and** more VRAM, or activation checkpointing that does not yet
   exist. The corpus is also the binding limit at this size — on ~4.3K train tokens the model overfits
   by ~epoch 10–11, and the held-out estimator (≈1166 predicted tokens) is too small to resolve
   sub-percent epoch-to-epoch gains. Larger/more-diverse data (with a larger held-out set) is the next
   lever, not more epochs.

3. **The moat is verifiability, not compression.** What is novel and defensible here is that the
   conversion is **re-derivable and tamper-evident on a from-raw-trusted compiler**: the exact ternary
   weights are Merkle-bound to the exact fp source, the measured quality gap is bound by the certificate
   sha (you cannot edit the number and pass), the linear **runs bit-exactly** on the Helix ternary
   kernel, and the kernel output carries a Freivalds receipt. The 14.49× footprint reduction is a real
   byproduct of ternary packing, **but this is lossy conversion — it is NOT lossless or near-lossless
   compression**, and it must never be presented as such.

---

## 5. File map

| Role | Path |
|------|------|
| Trainer (convert: QAT + KD + HX_SAVE, env-gated) | `helixc/runtime/llama_train.c` |
| Ternary packer (`--ternary-packfile`, `--ternary-selftest`) | `helixc/runtime/gpt2_pack.c` |
| Inference kernel | `helixc/examples/scaled_packed_ternary_matmul_kernel.hx` (+ `.ref.ptx`) |
| Freivalds receipt + launcher | `helixc/runtime/cuda_launch.c` |
| Corpus staging | `scripts/llama_stage_mmlu.sh` |
| **Convert driver** | `scripts/llama_kd_conversion.sh` |
| Trainer backward gate (mints the from-raw driver) | `scripts/llama_train_bwd_gate.sh` |
| **Verify driver** | `scripts/llama_convert_certify.sh` |
| Conversion receipt | `scripts/convert_receipt_llama.py` |
| Pack / dump helper (numpy mirror of the C packer) | `scripts/llama_ternary_pack.py` |
| Chain-all wrapper (convenience only) | `scripts/ternary_convert_run_all.sh` |
| Result records | `scripts/llama_kd_extended_result.txt`, `scripts/llama_kd_result.txt`, `scripts/llama_mmlu_ternary_result.txt`, `scripts/llama_convert_receipt_result.txt` |
