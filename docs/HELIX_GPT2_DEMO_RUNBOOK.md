# Helix × GPT‑2 124M — Investor Demo Runbook (P7)

**What this demo proves, in one line:** GPT‑2 124M — the real, unchanged public model — runs and
generates text on a compiler you can rebuild from **299 hand‑typed bytes**, its output matched
**token‑for‑token** to an independent reference and **reproducible bit‑for‑bit**, with a signed
attestation. *The product is a bring‑your‑weights **verified execution layer**, not a fast inference
engine — lead with trust, never speed.*

Execution plan + phase detail: `docs/HELIX_GPT2_DEMO_EXECUTION_PLAN.md`. This runbook is the live
operator script and the MVP Definition‑of‑Done closeout.

---

## 0. Pre‑flight (once, before the room)

- **Machine:** WSL2 (Ubuntu) on Windows, an NVIDIA RTX 3070‑class GPU (`sm_86`, 8 GB), CUDA 12.x
  driver + a PTX‑8.3‑capable `ptxas` on `PATH`. `nvidia-smi` should list the GPU.
- **Repo + weights:** the repo at `C:\Projects\Kovostov-Native`; the GPT‑2 weights + the P1 flat
  weight file + the fenced oracle live (gitignored) under `helix-llm/` (re‑downloadable via
  `helix-llm/tools/gpt2_import.py` if absent). `python3 -m pip install --user regex` for the oracle.
- **Warm‑up (do this before the slot):** run the hero command once so the ext4 mirror + the seed‑minted
  PTX are warm and you've captured a green transcript as the backup (see §3). A cold full run is a few
  minutes (the from‑raw rebuild dominates); a warm GPU generation is seconds.
- **The one command:**
  ```bash
  MSYS_NO_PATHCONV=1 wsl.exe bash -c "bash /mnt/c/Projects/Kovostov-Native/scripts/gpt2_demo_attest.sh"
  ```

---

## 1. The live demo (≤ 3 minutes)

**Beat 1 — the frame (~15 s).**
> "Every AI company asks you to *trust* a stack you've never seen — a compiler built by a compiler
> built by a compiler, down to vendors you can't audit. I'm going to show you GPT‑2 — the model you
> know — running on a stack you can **check from the very first byte**, and I'll prove it live."

**Beat 2 — the hero command (~90 s).** Run `gpt2_demo_attest.sh` and narrate each line as it prints:
- `REPRODUCE_TRUST: PASS` → *"That just rebuilt the entire compiler from 299 hand‑typed bytes — no
  trusted pre‑built compiler anywhere — reproduced it byte‑identically, and an independent compiler
  (gcc) corroborated it. Seed `9837db12`, fixpoint `0992dddd`, gcc‑DDC `84363adb`."*
- `GPT2_LOGITS_PARITY_PASS` + `GPT2_GENERATE_MATCH_PASS` → *"GPT‑2's public weights, unchanged,
  running on that toolchain through compiler‑emitted GPU kernels — and its output matches an
  independent numpy reference **token‑for‑token**."*
- `BYTE-IDENTICAL across two runs` → *"Run it twice — identical to the byte. Deterministic."*
- `DEMO_ATTEST_PASS` → *"And it wrote a signed attestation binding all of it together."*

**Beat 3 — show the artifact (~30 s).**
```bash
cat attestation/gpt2_attest.txt
```
Point at: the generated sentence; the three from‑raw anchors; the live model hash; the two equal
run hashes; **and the Honest Residuals section** (read one line aloud — see §4).

**Close (~20 s).**
> "What we sell isn't speed — it's the one thing nobody else can hand you: a model whose execution
> you can **verify from the first byte**. Helix is the verifiable execution layer underneath your AI."

---

## 2. The hero output (what a green run shows)

The model generates (GPT‑2‑124M greedy is grammatical‑but‑repetitive — that's the real model, not a
bug):
> **"The capital of France is the capital of the French Republic, and the capital of the French
> Republic is the capital of the French"**

Parity vs the independent f64 numpy oracle (same prompt): **last‑token argmax matches exactly** (id
262), max‑abs logit diff `2.59e‑04` on logits of magnitude ~130; the 20‑token greedy continuation
matches the oracle **token‑for‑token** (25 ids identical).

---

## 3. Captured green run (the recorded backup — DoD item 6)

This is verbatim output from a real, fail‑closed run (the backup if a live run isn't possible). A
screen recording of the same is the operator's to capture during warm‑up.

```
============================================================
 GPT-2-on-Helix demo attestation  (fail-closed, strictly serial)
============================================================
[demo_attest] [A] from-raw trust core: reproduce_trust.sh on a FRESH ext4 clone ...
[demo_attest]     REPRODUCE_TRUST: PASS
[demo_attest]     three anchors corroborated in reproduce_trust output (seed/fixpoint/K1)
[demo_attest] [B] GPT-2 124M inference + parity: bash scripts/gpt2_gpu_mvp.sh ...
[demo_attest]     GPT2_LOGITS_PARITY_PASS + GPT2_GENERATE_MATCH_PASS
[demo_attest]     generated text: The capital of France is the capital of the French Republic, and the capital of the French Republic is the capital of the French
[demo_attest]     run#1 gen-ids sha256: 8a2595cde91b893445f8000d360d8de64854e16e1c91d52e1513505116f3e70c
[demo_attest] [C] reproducibility shot: BYTE-IDENTICAL across two runs: 8a2595cd...
[demo_attest] [4] live model.safetensors sha256=248dfc39... (548105171 B)
[demo_attest] [5] all legs green -> writing attestation
============================================================
DEMO_ATTEST_PASS
  attestation : attestation/gpt2_attest.txt
============================================================
```
The committed proof a third party can re‑run themselves: `scripts/gpt2_demo_attest.sh` (this box) and
the `trust-reproduce` GitHub Actions CI (the from‑raw core, on a clean different machine).

---

## 4. Honest‑residuals card (say these unprompted, and answer plainly when asked)

State the edges before you're asked — the honesty *is* the pitch:

1. **Fenced host glue.** The weight importer, the byte‑level BPE tokenizer, and the numpy reference
   oracle are trusted host glue under gitignored `helix-llm/` — no compute‑trust role. The trust claim
   is the **exact token‑id sequence + the from‑raw toolchain that produced it**, not the host‑side
   string rendering.
2. **Complete to PTX, not to SASS.** Source → PTX is hand‑auditable (`hex0` → `kovc` → PTX); **below
   PTX**, NVIDIA's closed `ptxas` + the GPU driver + the C CUDA‑FFI launcher are trusted‑once. (The
   CPU path — a planned upgrade — has *no* such boundary.)
3. **fp32‑only**, parity exact on argmax + the token sequence, within a measured tolerance on hidden
   states. This bounds the scale the stack generalizes to (~≤1.5 B params on this 8 GB box).
4. **Single GPU, `sm_86`.** One RTX 3070‑class device; not multi‑GPU, not a cluster.
5. **A 124 M demonstration, not frontier scale.** The point is *verifiability*, not size or speed.
6. **The oracle shares the GPT‑2 spec** (independent f64 implementation, not an independent
   specification) — it catches implementation bugs, not a shared misunderstanding of GPT‑2.
7. **Never claimed:** beating cuBLAS, "fully verified GPU," completeness to GPU machine code, or AGI.

---

## 5. Fallback tree (if something isn't available live)

- **No GPU in the room / projector laptop has no CUDA:** show the **CI** — the `trust-reproduce`
  GitHub Actions run is green on a clean Ubuntu runner (the from‑raw core, a different machine), and
  play the captured §3 transcript + show a pre‑generated `attestation/gpt2_attest.txt`.
- **Full run too slow for the slot:** pre‑run leg A (the from‑raw rebuild) during warm‑up; live‑run
  only the GPU **generation** (`scripts/gpt2_gpu_mvp.sh`, seconds when warm) + show the attestation.
- **A leg goes red live:** that's the system working — every gate is fail‑closed and never fakes. Show
  the `*** FAIL` line, state honestly what it caught, and fall back to the captured green transcript.

---

## 6. MVP Definition‑of‑Done — status

| # | DoD item | Status |
|---|---|---|
| 1 | `reproduce_trust.sh` → `REPRODUCE_TRUST: PASS` live | ✅ |
| 2 | Coherent GPT‑2 generation on the GPU path | ✅ |
| 3 | `PARITY: PASS` (argmax + token‑for‑token vs oracle), fail‑closed | ✅ |
| 4 | Byte‑identical output across two runs | ✅ (`8a2595cd…`) |
| 5 | ≤ 3‑minute runbook | ✅ (this doc) |
| 6 | Pre‑recorded/captured green‑run backup | ✅ (§3; screen‑capture during warm‑up) |
| 7 | Honest‑residuals card, operator can state unprompted | ✅ (§4) |

**The MVP demo is complete.** Optional upgrades (the *full* demo + beyond): the CPU path
(no‑`ptxas` purest‑trust closer), re‑authoring the importer/tokenizer in C/Helix for a
"Python‑free toolchain" public claim, and a scale flex (GPT‑2‑XL, or a modern Apache‑2.0
Llama‑arch model with the 4 extra ops).
