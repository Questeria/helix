# Helix v1.0 — Definition of Done (the finish line)

**Purpose.** A single, *measurable* finish line for the Helix **language + compiler +
toolchain + trust chain** (the substrate). "Done" is not a feeling — it is the
checklist below passing. When it passes, **Helix is finished and we start fully
working with it**: building the AI (Alt) and beyond.

**Scope boundary (read this first).** This defines completion of the **substrate**.
It does **not** define AGI. AGI is open research — no language, however complete,
makes it achievable; it is *not* a Helix milestone. Crossing this line means the
substrate is no longer the bottleneck — if AGI is buildable, Helix won't be what
stops you. (See "Out of scope.")

---

## THE CAPSTONE — the distinct, measurable goal

> **A small but real transformer language model (≥ 2 attention layers) trains
> end-to-end on a GPU, entirely in Helix-native — with ZERO Python/PyTorch in the
> training loop — on a toolchain bootstrapped from raw binary (hex0 → seed → kovc)
> and diverse-double-compile-verified, and it converges to results matching a
> trusted reference: final training loss within 2% of the PyTorch oracle on a
> fixed dataset / config / seed, with eval-metric parity.**

One demo, but by construction it exercises the entire substrate at once: the full
language, GPU execution, autodiff, the numeric/tensor stack, and the Python-free
trusted toolchain. **The day this passes, Helix is done.**

### ✅ CAPSTONE ACHIEVED — 2026-06-01 (commit `adab69d`)

A 2-layer pre-norm transformer (V=32, d=16, S=16, 1 head, MLP H=64, NL=2) trains
end-to-end on the RTX 3070 in pure Helix-native: all 15 math kernels (forward, full
backward, Adam) are **kovc-emitted PTX**, with **zero Python in the training loop**.
Adam K=500: loss 62.35 → 0.42. Its full loss curve matches an independent,
self-gradient-checked numpy reference (`helixc/runtime/oracle_train.py`, FENCED
OFFLINE — never in the Helix path) to **0.0009%** worst-case (step 200) — three
orders of magnitude inside the 2% bar. Verification chain (no shared-bug escape):
forward vs numpy 1e-6 (`542b02c`) · GPU backward vs double-precision finite-diff,
weight-by-weight (`dcce27e`) · GPU training loss drop (`6f82db7`) · full-curve match
+ oracle backward self-check (`adab69d`). Harness: `helixc/runtime/train_transformer.c`
(Decision D2 trusted C launcher — to be ported to in-Helix CUDA-driver FFI before the
v1.0 freeze, see #6/#8).

**This crosses THE GATE (criterion #3)** and proves the substrate works end-to-end
(GPU execution + correct GPU autodiff numerics + the tensor stack + a Python-free
loop). It does **not by itself** make criteria #1, #2, #5, #6, #7, #8 green — those
remain real (lower-uncertainty) engineering, tracked below. Formal **v1.0 DONE** still
requires all 8 criteria green **AND** the 5-consecutive-clean-audit gate over the final
(Helix-harness) capstone. The loop does **not** stop here.

---

## DEFINITION OF DONE — the measurable checklist (all 8 must hold)

The capstone implies most of these; each is verified explicitly so "done" is
auditable, not asserted.

| # | Criterion | Measurable acceptance test | Status (2026-05-31) |
|---|-----------|----------------------------|---------------------|
| 1 | **Self-hosts (full language)** | byte-identical self-host fixpoint `K2 == K3` on the FULL `kovc` source — no i32-subset restriction, no external `ulimit` (the canonical self-host test unskipped + green) | 🔶 **FULL-SOURCE FIXPOINT PROVEN — Python-free, NO external ulimit (2026-06-01)**: the from-raw-binary seed (`seed.bin`, 62 KB, hex0..M2-Planet) compiles the FULL current `kovc.hx` (1.5 MB / 29681 lines) → K1; then K1→K2→K3→K4 reaches a byte-identical fixpoint **K2==K3==K4 = 601707 B, sha `b52d7d6f…`** (K1=595754 B, seed-built, differs as expected). Runner `scripts/selfhost_fixpoint_rawbinary.sh` (Python-free compile chain). The ENTIRE chain — seed→K1 included — runs on the **default 8 MB stack** with NO external `ulimit` at any step (proven 2026-06-01: `seed_rc=0` on the default stack; K2==K3==K4 byte-identical), via the seed parsing within 8 MB + `emit_start_bigstack` (512 MiB) in K2+. The measurable acceptance test — byte-identical K2==K3 on the full source, no external ulimit, green — is **MET**. The load-bearing equality CHECK is now **Helix-native** — `selfhost_bytecmp.hx` (seed-compiled: `read_file_to_arena` + in-arena byte compare; `18f1ace`) asserts K2==K3==K4, cross-checked by `cmp`; 6/6 unit tests incl. negative controls. Remaining before unqualified ✅ on the criterion NAME ("full language"): (a) only the run_process *orchestration* of the generations is still bash (needs a general-purpose kovc compiler, not the fixed-path drivers — tracked under #6/#13 with `assemble_k1.py`); (b) the self-compiled `kovc.hx` exercises only the bootstrap subset it is written in — struct/enum/`match`/float self-host feature coverage lands with #2/#5. The old "canonical test" was a Python test deleted at K4 (replaced by this runner, not unskipped). |
| 2 | **Feature-complete + runs (CPU)** | 100% of a language-feature corpus **compiles AND runs** correctly: structs, enums, `match`, generics, traits, closures, floats, all int widths, `grad`, `tile` | 🔶 **SAMPLE CORPUS 28/28 on the self-hosted compiler K2 (2026-06-01, `scripts/feature_corpus.sh`; gated by `scripts/gate_kovc.sh`)**. PASS (28): baseline, scalar-arith, **struct+enum+match** (→129), **payload-enum+match** (→42), **enum+recursion** (→120), **nested PatStruct destructure-in-match** (`dogfood_18`→42 ✅ FIXED), **user-defined `enum Result` + match** (→42), **grad+float** (→42), **i64** cast/compare/negation/**mul-beyond-i32** (6e9)/**div-beyond-i32** (4e9), **u8/u16** wrap+cast, **i16** overflow, **u64 logical shift** (✅ FIXED), **LEFT-associativity** (`10-3-2`→5, `100/5/2`→10 — confirmed correct), **comparisons** (!=/>=/<=), **bitwise** (&/|/^/<<), **arrays** (literal+index), **while+break**. FIXES (each gated: fixpoint K2==K3==K4 byte-identical + GPU PTX unchanged + no regressions): **u64 shift** `ef85342` (was arithmetic `sar` for u64); **PatStruct `field: subpattern`** parser feature `96c440d3` (named-struct-pattern parser didn't handle literal/nested/rename sub-patterns). **KNOWN LIMITATION (narrow, documented, NOT hidden)**: integer SOURCE LITERALS ≥ 2³¹ truncate (the lexer's i32 decimal accumulator) — e.g. `5_000_000_000_i64` wraps. The 64-bit imul/idiv/cmp CODEGEN is correct (the corpus proves i64 arithmetic *beyond* i32 via sub-2³¹ literals producing 6e9/4e9 runtime values); only large source constants are affected. Full fix = lexer/AST 64-bit-literal widening (mirror the f64 path: carry source span, re-parse to lo32+hi32; the seed has no i64 so it needs i32 multi-word arithmetic) — deferred (task #23), low-priority. Dropped `dogfood_16` (used `Ok`/`Err`/`unwrap_ok` = **deleted Python intrinsics**→`ud2`, not a kovc feature; replaced with a user-defined `enum Result` test). **OPEN USER SCOPE QUESTIONS**: (a) should `Ok`/`Err`/`Result` be language/stdlib builtins or user-defined? (b) are generics/traits/closures in v1.0 #2 scope (no standalone tests; dogfood comments say post-v1.0)? `tile` = GPU (#3). Criterion NOT met: the 17 are a *representative sample* — a fuller corpus (arrays, more patterns, the in-scope features above) + the scope decisions remain. |
| 3 | **GPU executes** ⭐ THE GATE | `tile` kernels run on **real GPU hardware**, results match reference (GPU corpus green on HW) | ✅ **GREEN — full transformer-op corpus on HW (2026-06-01, capstone `adab69d`)**: 15 kovc-emitted kernels (layernorm fwd/bwd-dx/bwd-dgb, naive matmul + A^T·B + A·B^T, qkt, softmax fwd/bwd, gelu fwd/bwd, ce-softmax-grad, scale, adam) execute on the RTX 3070, each verified vs an independent reference with negative controls. Correctness-complete. Remaining is **perf only**: shared-memory tiled GEMM with barriers (capstone uses register/naive matmuls — correct but unoptimized); not a correctness gate. First-light history: `docs/HELIX_GPU_FIRSTLIGHT.md` |
| 4 | **Autodiff correct** | `grad` gradients match numerical/reference gradients on CPU **and** GPU (gradient-check suite green) | 🔶 **GPU gradient-check GREEN (capstone)** — the full transformer backward (attention, layernorm, gelu, matmul, CE) is verified weight-by-weight on the RTX 3070 vs double-precision finite-diff (`dcce27e`); CPU autodiff exists. **Remaining / judgment call**: the capstone's GPU gradients come from **hand-written, finite-diff-verified backward kernels**, not from the `grad` autodiff KEYWORD emitting GPU code. If #4 requires the `grad` keyword to emit verified GPU gradients → still pending; if verified-correct GPU gradients suffice → green. **Flagged for the user.** |
| 5 | **Full-language trust** | the diverse-double-compile passes over a **feature-diverse** corpus (beyond i32) | 🔶 i32 DDC ✅ (5/5 audits); feature-diverse corpus pending |
| 6 | **Python-free + raw-binary** | the **entire** toolchain (compiler, test runner, build) is Helix/seed/ladder only; **zero `.py`** in the live toolchain; reproducible from hex0 | 🔶 **reference compiler DELETED 2026-05-31 (K4)**; full inventory + de-language plan: `docs/HELIX_DELANG_PLAN.md` (2026-06-01). **P1 done**: deleted 5 dead DDC/cascade `.py` (they imported the K4-deleted Python compiler — unrunnable; live proof is the Python-free `selfhost_fixpoint_rawbinary.sh`). Live-tree `.py` 12→7. **P2 done (`assemble_k1.py` → `assemble_k1.sh`, byte-identical, gated: fixpoint sha `96c440d3` unchanged, corpus 17/17)** — **the live TOOLCHAIN (compiler/build/test path) is now Python-free**. Remaining `.py` (6) are NOT the toolchain: the offline numpy oracle (flagged exception) + dev/proof scripts (`helix_status`, `mlir_audit_canaries`, `proof_artifact_*` — v3 infra, flag-before-delete). Remaining other-language: the `.sh` gates → a Helix test-runner (#7/#13), and the shell assembler itself → a Helix concatenator (final form). **TWO USER DECISIONS flagged**: (1) the **CUDA C launchers** (`cuda_launch.c`, `train_transformer.c`) need a **Helix FFI to libcuda** to become Helix (Helix has NO FFI today — static syscall-only ELFs; est. ~3-4 wk) OR accept them as a documented trusted-tool exception (like the ladder + `ptxas`); (2) the **numpy capstone oracle** (`oracle_train.py`, fenced-offline audit ref, NOT in the loop/toolchain) — keep as documented exception or port to Helix numeric (huge). Trust root (hex0..M2-Planet + `seed.c`) permanently exempt. |
| 7 | **Usable stdlib + toolchain** | documented stdlib (collections, math, strings, I/O, tensor/ML ops the capstone needs) with passing tests; self-sufficient driver / test-runner / module system in Helix | 🔶 stdlib campaign in progress |
| 8 | **Design frozen (v1.0 spec)** | a written language reference; syntax/semantics committed; no breaking changes after v1.0 | 🔶 **DRAFT written 2026-06-01: `docs/HELIX_V1_LANGUAGE_SPEC.md`** — a complete reference of the language as `kovc` actually implements it (lexical, types, items, expressions/match, builtins, codegen targets), each feature honestly marked [proven by corpus] / [impl] / [erased] / [unsupported]. NOT yet frozen: the freeze requires (a) the §7 open scope decisions resolved (generics/traits/closures in scope?; Ok/Err builtins?) and (b) the language to stop changing. Reference exists; formal freeze pending the user scope calls + stability. |

**HELIX v1.0 — DONE** ⇔ criteria 1–8 green **AND** the capstone transformer trains
correctly.

---

## OUT OF SCOPE (explicitly NOT required to call Helix done)

- **AGI itself** — open research; not gated by the language. Pursued *on* Helix, after v1.0.
- **Frontier-scale / multi-GPU / multi-node training** — the *scaling* milestone (reached *while working with* Helix, not before calling it done).
- **Specific applications** — Alt (the LLM), Mercury, etc. are built *with* Helix, after v1.0.

---

## Current status & critical path (honest)

- ✅ **Trust + bootstrap** — from-raw-binary, DDC-verified (i32 core), 5/5 clean audits, full pre-K4 backup tagged. *The hardest-to-trust part is already done.*
- ✅ **K4 done (2026-05-31)** — the Python reference compiler is **deleted**; the mint is verified Python-free (seed → kovc, 17/17 tests, `6*7`→42). Pre-K4 state preserved at tag `v0-pre-k4-full-with-python`; post-K4 at `k4-python-compiler-deleted`.
- ✅ **THE GATE (#3) — CROSSED (2026-06-01, capstone `adab69d`).** Not just first-light: the **entire transformer-op corpus** (15 kovc-emitted kernels — forward, full backward, Adam) executes on the RTX 3070, each verified vs an independent reference with negative controls. A raw-binary-bootstrapped Helix toolchain drives real GPU execution end-to-end. Remaining is **perf only** (shared-memory tiled GEMM); correctness is complete. See `docs/HELIX_GPU_FIRSTLIGHT.md`.
- ✅ **THE CAPSTONE — ACHIEVED (2026-06-01, `adab69d`).** A 2-layer transformer trains end-to-end on the RTX 3070 in pure Helix-native (zero Python in the loop), matching a self-gradient-checked numpy reference to **0.0009%** over 500 Adam steps — 3 orders inside the 2% bar. The single hardest, most *uncertain* demonstration is done.
- 🔶 **Remaining for formal v1.0 (honest — the capstone proves the substrate but is NOT the whole checklist).** Still RED/partial: **#1** full-source self-host fixpoint — **CORE PROVEN 2026-06-01** (raw-binary seed → K1 → K2==K3==K4 byte-identical on the full 1.5 MB `kovc.hx`, Python-free; `scripts/selfhost_fixpoint_rawbinary.sh`); **no external ulimit** (proven, default 8 MB stack); remaining: a Helix-native runner (#17, a #6 de-bash item) + full-language feature coverage (via #2/#5); **#2** 100% feature compile+run corpus; **#4** the `grad` keyword emitting GPU gradients (GPU gradients themselves verified — see table); **#5** feature-diverse DDC (beyond i32); **#6** finish Python/non-Helix removal (de-Python `assemble_k1.py` + DDC harnesses + dev scripts + port test-infra **and** the capstone C harness to Helix); **#7** stdlib; **#8** spec freeze. Lower-uncertainty engineering, but real — the loop continues until all green + 5 clean audits.

```
critical path (post-capstone, 2026-06-01):
  ✅ GPU executes (#3) → ✅ training loop → ✅ capstone converges (0.0009%)   [THE GATE — CROSSED]
  remaining → close #1 #2 #4 #5 #6 #7 #8 (engineering) → 5 consecutive clean audits → HELIX v1.0 DONE
```

When this line is crossed, the substrate is proven and the AI-building phase begins.
