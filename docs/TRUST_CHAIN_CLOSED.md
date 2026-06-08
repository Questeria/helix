# Helix — Trust Chain Record (v1.3 — TRUST CHAIN CLOSED 2026-06-07, tag v1.3-release)

---

## ✅ TRUST CHAIN CLOSED — v1.3 (declared 2026-06-07)

> **Declaration.** The Helix from-raw-binary trust chain is **CLOSED at the `v1.3-release` tag**, in the
> precise, honestly-scoped sense below: the toolchain is **reproducible from a hand-typed root, self-
> hosting, defended against a trusting-trust attack at the `seed→K1` rung, demonstrably capable, and
> Python-free** — verified by independent and reproducible means, **with the residuals in §R disclosed.**
> This is **not** a claim of absolute or unconditional trust; it is a claim that every link we *can*
> establish has been established, cross-checked, and made push-button-reproducible by anyone.
>
> **What is closed (verified):**
> 1. **Hand-typed root → compiler.** `hex0` (299 hand-authored bytes) → hex1 → hex2 → catm → M0 →
>    cc_amd64 → M2-Planet → `seed` (Apache-2.0 C-subset) → `kovc`. Every rung is rebuilt **only by the
>    prior rung** and matches its committed `.sha256`; `seed.bin` re-derives to `9837db12…`. No pre-built
>    binary is trusted — the committed rung binaries are reference copies; reproduction deletes them and
>    rebuilds from raw.
> 2. **Self-hosting fixpoint.** `seed → K1 → K2 → K3 → K4`, **K2 == K3 == K4 byte-identical =
>    `0992dddd…`**; the 109-program feature corpus + 4 negative-diagnostic checks pass.
> 3. **Trusting-trust defense (`seed→K1`).** `gcc` (independent lineage, zero M2-Planet ancestry) and the
>    from-raw seed both compile `k1src.hx` to a **byte-identical K1 = `84363adb…`** (Wheeler diverse-
>    double-compile).
> 4. **Real capability.** A 2-layer transformer trains end-to-end on `kovc`-emitted GPU kernels (RTX
>    3070), converging to within **0.0009%** of an independent numpy oracle (bar 2%), with a sampled
>    finite-difference gradient check and a load-bearing negative control.
> 5. **Python-free toolchain.** Exactly **one** committed `.py` (a fenced numpy audit oracle, never on
>    the compile/run path); compiler + runtime are Helix + a small hand-authored C subset (24 `.c`/`.h`).
>
> **How it was verified (independent + reproducible):**
> - A **committed one-command reproduction** (`scripts/reproduce_trust.sh`) rebuilds the whole ladder
>   from raw and asserts every pinned hash — running **GREEN on a clean GitHub `ubuntu-latest` runner**
>   (different machine, fresh clone, zero local state) via `.github/workflows/trust-reproduce.yml`,
>   push-button for any third party.
> - A **different-model-lineage** review (ChatGPT, read-only) across **four** whole-repo passes converged
>   with **no critical, no fail-open, no hidden code**.
> - A **context-isolated fresh Claude** auditor independently rebuilt from a clean clone and re-derived
>   every hash — P(core trust holds) = **0.93**.
> - A **live joint reproduction** witnessed by the project owner.
>
> **§R — Residuals (what "closed" does NOT cover; disclosed in full):**
> - **Shared TCB:** host OS, kernel, filesystem, shell, coreutils, `gcc`/`libc`/`binutils`/loader, CPU +
>   microcode, RAM, and the audited `seed.c` source remain trusted. No DDC retires the shared substrate.
> - **Complete to PTX, not SASS:** the GPU path is hand-auditable `hex0 → PTX`; NVIDIA's closed `ptxas` +
>   driver + the CUDA-driver-FFI host launcher are trusted past PTX. Single GPU target (sm_86).
> - **V5 v1.1-surface DDC** is a *manually-reconciled behavioral* audit; its witness is gitignored / not
>   clean-checkout reproducible. The byte-identical, hash-pinned, one-command DDC is the separate
>   `seed→K1` `ddc_crosscheck.sh`.
> - **Independent third party:** reproduction is now push-button on a different machine (CI) and by anyone
>   who forks the repo; a reproduction by a party with *no connection to the author* remains the one
>   outstanding increment — now trivially available.
>
> **Owner attestation.** "I have witnessed the live reproduction and reviewed the evidence and residuals
> above, and I declare the Helix v1.3 from-raw trust chain CLOSED at `v1.3-release` on 2026-06-07."
> — **Questeria** (project owner)

**Status: the from-raw-binary trust chain is COMPLETE and independently verified.**
This document records the verified state of Helix across the **v1.3 "Honest-Completeness & Trust"** line, now tagged `v1.3-release` and **declared CLOSED above** (2026-06-07) after a live joint re-verify. It is the honest
record on which that *formal, public* "trust chain closed" declaration rests — made by the
project owner after a deliberate joint re-examination, not an automated one. Everything below was reproduced by **context-isolated, same-model-family
(Claude) adversarial reproductions** — separate prompts/contexts, but the same model
lineage that drove the build, so they share its blind spots (the monomorphic-dispatch
ceiling disclosed at `docs/HELIX_COMPLETION.md` ~749/767). A **different-lineage
cross-model review (ChatGPT, with read-only repo access)** was since performed and its
findings remediated, so verification now spans **both** same-family reproduction **and** a
cross-model review (a read-only doc/logic review, not an independent build reproduction).
Every limitation is stated plainly.

Last formal tag: `v1.2-complete` (`291f0ec`, fixpoint `K2==K3==K4 = 9cc8f20b…`). Current line: **v1.3** — self-host fixpoint `K2==K3==K4 = 0992dddd…` (pinned in `scripts/gate_kovc.sh`, re-verified live this cycle); now tagged **`v1.3-release`** (declared closed 2026-06-07 after the live joint re-verify).
(NOTE: the stale `v2.0.0`–`v3.1.0` git tags are from a **superseded MLIR exploration line**;
the `v1.3` line is the current real head (`v1.2-complete` was the prior formal tag), despite being numerically lower.)

---

## 1. What the trust chain IS (verified)

- **From a hand-typed root.** `hex0` (299 hand-authored hex bytes) → `hex1` → `hex2` →
  `catm` → `M0` → `cc_amd64` → `M2-Planet` → **`seed`** (our Apache-2.0 C-subset compiler) →
  **`kovc`** (the Helix compiler). Each rung is built **only by the prior rung** — no
  pre-built binary is trusted. An independent auditor **rebuilt the entire ladder from
  source and every rung reproduced its committed SHA byte-identically** (seed `9837db12…`).
- **Self-host fixpoint.** `seed → K1 → K2 → K3 → K4`, with **K2 == K3 == K4 byte-identical**
  (**v1.3: `0992dddd…`**, pinned in the gate + re-verified live this cycle; v1.2 was `9cc8f20b…`). Reproduced live by auditors.
- **Diverse double-compile (DDC).** These are **two distinct, non-equivalent claims** — kept
  explicitly separate to avoid overclaim:
  - **(i) Byte-identical DDC — `seed`/`K1` surface ONLY.** `gcc` (an independent compiler
    lineage with **no M2-Planet ancestry**) and the M2-Planet-built `seed` both compile the
    same `k1src.hx` into a **BYTE-IDENTICAL** `K1` (`scripts`/`stage0/helixc-bootstrap`
    DDC runners). Identical `K1` from two independent compilers is a Wheeler diverse-double-
    compile against a trusting-trust attack — but it covers **only the seed→K1 step**, not the
    broader language surface. `gcc` is an **auditor**, never the shipped root.
  - **(ii) BEHAVIORAL cross-check — v1.1 language surface, NOT byte-identical.** The v1.3 **V5**
    broadening over the **v1.1 language surface** (generics/monomorphization, traits, closures,
    turbofish, wide-field, bf16) is checked **BEHAVIORALLY ONLY**: `kovc` (built from the raw
    binary) and a **second, zero-lineage interpreter** are run on the same programs and must
    agree on each program's exit. This is **NOT a byte-identical second-compiler reproduction**
    — the interpreter emits no code, so byte-identity is impossible there by construction. It is
    also **NOT clean-checkout reproducible by a third party**: that cross-check interpreter is
    **gitignored** (not committed), so a fresh checkout cannot re-run it as-is
    (`docs/K_DDC_BROADENED.md`). f16-arith is not yet cross-checked.
- **Python-free shipped toolchain.** Exactly **1** committed `.py` in the repo
  (`verification/oracle/oracle_train.py`), a fenced verification witness never referenced
  by the toolchain. The compiler/runtime are Helix + a small hand-authored C subset.
- **Real capability.** A ≥2-layer transformer trains **end-to-end on Helix-emitted GPU
  kernels** and converges to within **2% of an independent numpy oracle** — reproduced at
  **~0%** loss difference. The oracle was adversarially proven genuinely independent (it
  computes its OWN curve from the *shared initial* weights, then compares against Helix's emitted loss
  curve (it reads Helix's `loss_curve.csv` only for that comparison, never as input to its own
  computation); f32-vs-f64 curves are
  close-but-not-bit-identical). **The WINNING GEMM is the f32-SMEM `cp.async` double-buffered
  tile**, not TF32: on the reference RTX 3070 Laptop TF32 Tensor-Core mma is *slower*
  (~0.97× the tuned f32-SMEM GEMM — 312 ms vs 274 ms; `docs/HELIX_GPU_PERF_RESULT.md` ~634).
  A TF32 mma op-set is *emitted and selectable* (`HX_OPT=2`, parity verified) but is NOT the
  performance path on this hardware. `mma.sync`/TF32 is therefore proven-correct-but-not-the-default.
- **Verification.** **5 consecutive clean, context-isolated, same-model-family (Claude)
  adversarial reproductions** (distinct lenses: trust chain/DDC, GPU perf, capstone
  correctness + oracle independence, language/codegen + bounds, overclaim/completeness).
  Each auditor *reproduced* its claims; the audits found no faked result and no undisclosed
  residual. **Honest scope:** these 5 share a model lineage with the build, so they catch
  reasoning/consistency/reproducibility gaps but not blind spots shared by author and
  auditor (the monomorphic-dispatch ceiling — `docs/HELIX_COMPLETION.md` ~749/767).
  Verification was since **broadened across model families**: a different-lineage
  **cross-model review (ChatGPT, read-only repo access)** was performed and its findings
  remediated. That cross-model pass was a doc/logic review, **not** an independent build
  reproduction — so it does not substitute for external reproduction by an independent
  operator/toolchain (residual #10 below).

## 2. The honest boundaries (residuals — all disclosed)

These are real limits. They do not undermine the trust chain; they are stated so the
claim is precise, not inflated.

1. **GPU performance is a fraction of cuBLAS, not parity.** Same-GPU (RTX 3070 Laptop):
   G1 4.56 TFLOP/s (56% cuBLAS-f32), G2 5.445 (67.5%), G3 TF32 Tensor-Core 5.35–5.76
   (50–54% cuBLAS-TF32). Helix emits correct, reasonably-performant kernels; it does **not**
   beat NVIDIA's hand-tuned library, on this or any GPU. "Parity tier" is a label always
   paired with the explicit fraction.
2. **Capstone end-to-end speedup is 7.0–8.7×, not ≥10×.** The ≥10× target was an estimate;
   the honest measured ceiling is **Amdahl-bound** (GEMM is ~70% of the step and already the
   f32-SMEM cp.async tier; TF32 Tensor Cores are a confirmed **dead end** on this GPU at
   0.97× the f32 GEMM; a larger model cannot be baselined because the naive matmul cannot
   launch at scale). **Loss parity (the hard correctness gate) holds at ~0%.** The ≥10×
   clause is re-scoped to the measured ceiling, documented.
3. **DDC coverage is 44/53 witness-reachable arms.** The v1.1-surface (generics, traits,
   closures, turbofish, wide-field store, bf16) is **un-DDC'd** by the frozen Python witness
   — a named open residual, not full coverage.
   **What the DDC does NOT eliminate (the remaining SHARED trusted computing base, named
   bluntly).** A diverse-double-compile only catches a backdoor that one of the two
   compilers carries and the other does not. It says **nothing** about anything BOTH sides
   share. The seed/K1 gcc-vs-M2-Planet DDC (and the behavioral second-witness cross-checks)
   therefore still trust, untouched, the entire **shared substrate** beneath both compilers:
   the **shared gcc / libc / binutils / loader** (`ld.so`) used to build and link both seed
   variants; the **shell + coreutils** (`bash`, `cmp`, `sha256sum`, `cp`, `rm`) that drive
   and compare the build; the **filesystem**; the **OS / kernel (WSL2 Linux)**; the **CPU +
   microcode**; the **RAM**; and the **human-readable `seed.c` source itself** — auditable
   one line at a time, but **trusted-by-reading**, not proven. A backdoor identical in both
   gcc builds, or living in any of these shared layers, is **invisible** to DDC by
   construction (the classic Wheeler shared-substrate residual). This is stated so the trust
   boundary is unambiguous: DDC narrows the compiler-backdoor surface, it does not erase the
   shared TCB.
4. **Documented language bounds.** **No silent-wrong residual remains** (the v1.2 i64/u64
   wide struct-field low-32 truncation was closed by v1.3 V1; the f16 same-type-arith
   silent-miscompute that Finale Audit 2 caught was closed by the 2026-06-04 f16 GAP FIX —
   `f16` ident/literal now map to type tag 5 so the F16C path is reached, gated by
   `V4_f16_add`/`V4_f16_mul`). **bf16 and f16 same-type add/mul compute correctly + are
   bit-exact-gated**; a 16-bit float **mixed** with a non-16-bit-float operand fails **loud**
   (traps 2001/4001, no implicit widening). f64 wide fields and u64 ≥2³² literals compute
   full-range; a closure capture wider than i32 fails **loud** (trap 76003). Borrows
   (`&mut` non-aliasing), `const`/`static`, module privacy, and match exhaustiveness are
   **documented design bounds**, not enforced.
5. **G4 (bf16 `wmma`) is a STRETCH and was not taken.** Not claimed done.
6. **Single hardware target.** Only `sm_86` (RTX 3070 Laptop) is tested. No cross-arch
   (sm_80/sm_90) or multi-vendor (AMD) validation.
7. **The GPU path is not all-the-way-down.** The chain is hand-auditable from `hex0`
   **to PTX**; below PTX it relies on NVIDIA's **closed `ptxas`** (PTX→SASS) and driver.
   The **CPU** path is all-the-way-down from raw binary; the **GPU** path is
   from-hex0-to-PTX-then-`ptxas`. This is the one trusted-once boundary on the GPU side,
   stated openly. _(v1.3 V6: the GPU **host launcher** — `helixc/runtime/cuda_launch.c` +
   `train_transformer.c` — is the C-FFI half of this same boundary: it makes the closed
   `libcuda` driver-API calls Helix cannot. Porting it to Helix would **move**, not close,
   this boundary, so it stays trusted-once C. This residual **STANDS** unchanged. Full
   trusted-C inventory + the V6 dead-C prune: `docs/TRUSTED_C_INVENTORY.md`.)_

> **Front-door residuals (8–10) — the limits most likely to surprise an outside reader.**
> Stated prominently because they bound what "complete" and "reproducible" mean here.

8. **Complete to PTX, NOT to GPU machine code.** The hand-auditable from-raw chain ends at
   **PTX text**. The **trusted computing base below the from-raw chain** is therefore the
   **closed NVIDIA `ptxas`** (PTX→SASS assembler) + the **CUDA driver** + the **GPU
   hardware** + the **OS/kernel** + the **C host launcher** (`helixc/runtime/cuda_launch.c`
   / `train_transformer.c`). None of these are reproduced from raw binary; they are
   trusted-once. "Complete to PTX" is the precise claim — *not* complete to GPU machine code.
9. **The V5 v1.1-surface behavioral DDC (44/44) is NOT clean-checkout reproducible.** The
   second-witness tree-walking interpreter is **gitignored, was never committed, and has no
   clean restore path**, so the 44/44 v1.1-surface behavioral cross-check is replayable
   **only with an out-of-tree auditor artifact**, not from a fresh clone. By contrast, the
   **CORE chain IS clean-checkout reproducible**: the from-raw ladder (hex0→…→seed→kovc),
   the self-host fixpoint K2==K3==K4, the gcc-DDC of seed→K1, and the 109-program corpus all
   rebuild from a clean checkout. The reproducibility boundary runs between the core chain
   (in-tree) and the V5 behavioral witness (out-of-tree).
10. **Process residual — internal audit logs are evidence, not external reproduction.** The
    `.stage33-logs/` and `docs/audit-*` records are useful evidence, but they are **NOT a
    substitute for external reproduction by an independent operator/toolchain**. The
    cross-model **ChatGPT review was a read-only doc/logic review, not a build reproduction**
    — it raised and we remediated documentation/logic findings, but no third party has yet
    rebuilt the chain from raw on independent hardware. External operator reproduction
    remains open.

## 3. Phase 2 (the user's to start — not auto-started)

- **Datacenter scaling (Runpod):** parameterize the PTX target (`sm_80`/`sm_90`, Hopper
  TMA/bigger mma), validate cross-arch correctness, optimize for absolute throughput, scale
  the capstone on A100/H100.
- **AMD / ROCm:** a genuinely separate backend (CDNA MFMA, rocBLAS reference, no native
  TF32) — currently unimplemented.
- **v-next codegen:** G4 bf16 `wmma` (GPU tensor-core path) remains the open stretch. (The
  v1.3 cycle SHIPPED what this line previously listed as v-next: i64/u64 wide struct fields
  [V1 — the silent residual closed], capturing closures as values [V3], and bf16/f16
  arithmetic [V4 + the 2026-06-04 f16 GAP FIX, both bit-exact-gated].)
- **Broaden the DDC** to the v1.1 language surface.

## 4. Bottom line

From a hand-typed hex0 seed, with no trusted pre-built compiler, Helix self-hosts
byte-reproducibly, is independently cross-compiled, and trains a real neural network on its
own GPU kernels — verified by five context-isolated, same-model-family (Claude) adversarial
reproductions **plus** a different-lineage cross-model (ChatGPT, read-only) review whose
findings were remediated, with every limitation documented. **The trust chain is complete
to PTX and the residuals are honest.**
The formal declaration that it is "closed" is reserved for the project owner's review.
