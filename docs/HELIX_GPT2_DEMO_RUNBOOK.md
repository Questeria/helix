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
  weight file + the fenced numpy oracle live **gitignored** under `helix-llm/` (large binaries +
  the independent verifier — not in a clone). **To obtain them from scratch** (a third party on a
  fresh clone): download the model from **HuggingFace `openai-community/gpt2`** (MIT license —
  `model.safetensors`, `vocab.json`, `merges.txt`, `config.json`), then convert to the demo's flat
  `.weights` with the **committed, Python‑free** importer `helixc/runtime/gpt2_pack.c`
  (`gcc -O2 helixc/runtime/gpt2_pack.c -o gpt2_pack && ./gpt2_pack model.safetensors gpt2_124M.weights`)
  and tokenize prompts with the **committed** `helixc/runtime/gpt2_tok.c` — both gated byte/bit‑exact
  vs the Python originals by `scripts/gpt2_pyfree.sh`. The from‑raw trust **core** is fully
  third‑party‑reproducible from the repo alone (`scripts/reproduce_trust.sh` + CI); only the GPT‑2
  demo legs need these public weights fetched + converted. The fenced numpy oracle
  (`helix-llm/tools/gpt2_numpy_ref.py`, an independent reference — kept Python by design) needs
  `python3 -m pip install --break-system-packages regex` (or `--user` on a non‑PEP‑668 box).
- **Live chat server (GPT‑2‑XL) — producing the XL `.weights` + paths (third party, fresh clone).**
  The live interactive chat (`scripts/serve_chat_demo.sh`) serves **GPT‑2‑XL (1.5 B)**. Its `.weights`
  file is gitignored (not in a clone); produce it with the **same Python‑free path** used for 124M:
  download **HuggingFace `openai-community/gpt2-xl`** (MIT — `model.safetensors`, `vocab.json`,
  `merges.txt`, `config.json`), then convert with the committed `helixc/runtime/gpt2_pack.c`
  (`gcc -O2 helixc/runtime/gpt2_pack.c -o gpt2_pack && ./gpt2_pack gpt2-xl/model.safetensors gpt2-xl.weights`),
  and place `gpt2-xl/{vocab.json,merges.txt,config.json}` under `helix-llm/models/gpt2-xl/`. The server
  + serve gate read three **overridable env vars** with sensible defaults (so they run on a machine
  other than the author's): `HELIX_SRC` (default = this checkout), `HELIX_WORK` (default
  `$HOME/gpt2_ext4/Kovostov-Native`, the fast ext4 build mirror), and `HELIX_XL_WEIGHTS` (default
  `$HOME/gpt2_ext4/gpt2-xl.weights`). Put the converted file at `$HELIX_XL_WEIGHTS` (or export it to your
  own path). `scripts/helix_serve_gate.sh` then certifies the served XL output == the offline numpy‑oracle
  reference token‑for‑token.
- **Warm‑up (do this before the slot):** run the hero command once so the ext4 mirror + the seed‑minted
  PTX are warm and you've captured a green transcript as the backup (see §3). A cold full run is a few
  minutes (the from‑raw rebuild dominates); a warm GPU generation is seconds.
- **The one command:**
  ```bash
  MSYS_NO_PATHCONV=1 wsl.exe bash -c "bash /mnt/c/Projects/Kovostov-Native/scripts/gpt2_demo_attest.sh"
  ```

---

## 0.1 Reproducibility tiers (read this before claiming "reproducible")

Be precise about what a *third party* can reproduce. There are two tiers, and only Tier A is
repo‑only:

- **TIER A — the trust core (FULLY third‑party‑reproducible from the committed repo alone).**
  `scripts/reproduce_trust.sh` + the `trust-reproduce.yml` CI rebuild the entire `hex0 → seed → kovc`
  ladder from raw, run the byte‑identical self‑host fixpoint (`K2 == K3 == K4`) and the gcc
  diverse‑double‑compile, and assert the pinned anchors. This needs **NO model weights and NO
  oracle** — it runs CPU‑only from a clean clone in ~1 min, and the CI proves it on a clean,
  different‑machine `ubuntu-latest` runner. **This is the load‑bearing trust claim**, and it is the
  only tier that is repo‑only reproducible.

- **TIER B — the GPT‑2 demo legs (parity / scale / serve / attestation).** These additionally require
  **external artifacts that are NOT in the committed repo** and live under the gitignored
  `helix-llm/`:
  1. **The public GPT‑2 weights + vocab/merges.** Fetch from HuggingFace
     **`openai-community/gpt2`** (and `gpt2-xl` for the live chat) — **MIT‑licensed** —
     (`model.safetensors`, `vocab.json`, `merges.txt`, `config.json`), then convert to the demo's flat
     `.weights` with the **committed, Python‑free** importer `helixc/runtime/gpt2_pack.c` and tokenize
     with the **committed** `helixc/runtime/gpt2_tok.c` (both gated byte/bit‑exact vs the Python
     originals by `scripts/gpt2_pyfree.sh`).
  2. **The independent numpy reference oracle** (`helix-llm/tools/gpt2_numpy_ref.py`) — an
     **OUT‑OF‑FIXPOINT verifier** that 6 parity gates depend on (`gpt2_gpu_mvp.sh`, `gpt2_scale.sh`,
     `gpt2_cpu_parity.sh`, `gpt2_gpu_parity.sh`, `gpt2_pyfree.sh`, `gpt2_demo_attest.sh`; transitively
     `helix_serve_gate.sh` via `gpt2_scale.sh`). It is **kept DELIBERATELY uncommitted** to preserve
     the toolchain's **exactly‑1‑committed‑`.py` fence** (`verification/oracle/oracle_train.py` is the
     only committed `.py`). A third party can either **use OUR fenced oracle** (shipped with the demo
     bundle, not the public repo) **OR supply their OWN independent numpy GPT‑2 forward** — independence
     is precisely the point of a cross‑check, so any honest independent reimplementation is acceptable.

  **Do NOT claim the demo parity legs are repo‑only‑reproducible.** Only Tier A is. The Tier‑B legs are
  reproducible *given the public weights + an (our‑bundled or own) independent oracle*. Keeping the
  oracle uncommitted is the **honest fix**, not a gap to be closed by committing it — committing it
  would break the 1‑`.py` fence.

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

Parity vs the independent fp32 numpy oracle (same prompt): **last‑token argmax matches exactly** (id
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

1. **Host glue (now Python‑free on the production path).** The demo's two offline steps — the
   byte‑level BPE tokenizer and the safetensors→`.weights` importer — are **committed C host tools**
   (`helixc/runtime/gpt2_tok.c` + `helixc/runtime/gpt2_pack.c`): Category‑B, **outside** the self‑host
   fixpoint, **zero arithmetic on the compute‑trust path** (exactly like `cpu_host.c`). So the demo now
   runs with **zero Python installed** — fail‑closed‑gated bit‑exact by `scripts/gpt2_pyfree.sh`
   (tokenizer encode/decode parity + the pinned prompt + the hero decode; importer output byte‑identical,
   sha256, to the reference). The independent **numpy reference oracle** (`helix-llm/tools/gpt2_numpy_ref.py`)
   **stays Python on purpose** — it is the cross‑check verifier and its independence is the whole point.
   The trust claim is still the **exact token‑id sequence + the from‑raw toolchain that executes it**, not
   the host‑side string rendering. (The tokenizer's Unicode tables are a generated DATA `.inc`, bit‑exact
   with Python's `regex`; it is not a `.c`/`.h`, so no fence cost.)
2. **Complete to PTX, not to SASS.** Source → PTX is hand‑auditable (`hex0` → `kovc` → PTX); **below
   PTX**, NVIDIA's closed `ptxas` + the GPU driver + the C CUDA‑FFI launcher are trusted‑once. (The
   CPU path — a planned upgrade — has *no* such boundary.)
3. **fp32‑only**, parity exact on argmax + the token sequence, within a measured tolerance on hidden
   states. This bounds the scale the stack generalizes to (**measured: up to 1.5 B params — GPT‑2‑XL —
   runs on this 8 GB sm_86 box at fp32**; the same `kovc`‑emitted kernels also pass token‑for‑token at
   GPT‑2‑Large 774 M. See `scripts/gpt2_scale.sh`.).
4. **Single GPU, `sm_86`.** One RTX 3070‑class device; not multi‑GPU, not a cluster.
5. **A 124 M demonstration, not frontier scale.** The point is *verifiability*, not size or speed.
6. **The oracle shares the GPT‑2 spec** (independent fp32 implementation, not an independent
   specification) — it catches implementation bugs, not a shared misunderstanding of GPT‑2.
7. **Never claimed:** beating cuBLAS, "fully verified GPU," completeness to GPU machine code, or AGI.

---

## 5. Fallback tree (if something isn't available live)

- **No GPU in the room / projector laptop has no CUDA:** show the **CI** — the `trust-reproduce`
  GitHub Actions run is green on a clean Ubuntu runner (the from‑raw core, a different machine), and
  play the captured §3 transcript + show the attestation you captured during warm‑up (§0). (`attestation/`
  is gitignored — a fresh attestation is regenerated per run — so there is no committed file in a clone;
  the artifact to show is the one your own §0 warm‑up wrote.)
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
(no‑`ptxas` purest‑trust closer) — **done**; re‑authoring the importer/tokenizer in C for a
"Python‑free production data path" claim — **done** (`helixc/runtime/gpt2_tok.c` +
`helixc/runtime/gpt2_pack.c`, fail‑closed‑gated by `scripts/gpt2_pyfree.sh`: the demo runs with **zero
Python installed**; the independent numpy oracle stays Python as the verifier); and (a modern
Apache‑2.0 Llama‑arch model with the 4 extra ops).

**Scale flex — DONE.** The "same code, bigger model" generalization is now demonstrated:
GPT‑2‑Large (774 M, 36 layers) **and** GPT‑2‑XL (1.5 B, 48 layers) both run a real forward + greedy
generation through the **exact same 8 `kovc`‑emitted PTX kernels** as the 124 M MVP — **zero new
ops/kernels**, only dimension changes (read from each model's `config.json`). Both pass token‑for‑token
vs the fenced numpy oracle (Large argmax id 262, max‑abs logit diff 3.8e‑05, 25/25 ids; XL argmax id
262, max‑abs logit diff 4.4e‑05, 25/25 ids; XL output: *"The capital of France is the city of Paris.
It is the capital of France and the largest city in France. It is"*). Reproducible via the fail‑closed
gate `scripts/gpt2_scale.sh` (`MODEL=gpt2-large|gpt2-xl`); the verbatim **PRIMARY‑mode** verdict
evidence for both (mode line, argmax‑exact, the `max_abs logit diff=` line, token‑for‑token, gen‑ids)
is committed in `scripts/scale_results.txt` so these figures trace to a real run. This **measures** the
fp32 ceiling residual: 1.5 B fits the 8 GB sm_86 box (the committed `gpt2_infer.c` is dimension‑generic;
per‑layer weight streaming keeps device residency low, so layer count does not gate VRAM).
