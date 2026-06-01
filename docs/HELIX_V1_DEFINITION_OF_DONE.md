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

---

## DEFINITION OF DONE — the measurable checklist (all 8 must hold)

The capstone implies most of these; each is verified explicitly so "done" is
auditable, not asserted.

| # | Criterion | Measurable acceptance test | Status (2026-05-31) |
|---|-----------|----------------------------|---------------------|
| 1 | **Self-hosts (full language)** | byte-identical self-host fixpoint `K2 == K3` on the FULL `kovc` source — no i32-subset restriction, no external `ulimit` (the canonical self-host test unskipped + green) | 🔶 partial (i32 fixpoint ✅; canonical test still skipped pending big-stack) |
| 2 | **Feature-complete + runs (CPU)** | 100% of a language-feature corpus **compiles AND runs** correctly: structs, enums, `match`, generics, traits, closures, floats, all int widths, `grad`, `tile` | 🔶 most features compile; full compile+run corpus not yet 100% |
| 3 | **GPU executes** ⭐ THE GATE | `tile` kernels run on **real GPU hardware**, results match reference (GPU corpus green on HW) | 🔶 **FIRST-LIGHT GREEN (2026-06-01)** — kovc-emitted `vector_add` PTX executes on the RTX 3070, `c[7]=21` verified over 256 elems, 2-agent adversarial audit PASS/HIGH (negative controls); remaining: tiled GEMM + transformer-op corpus on HW (P5). See `docs/HELIX_GPU_FIRSTLIGHT.md` |
| 4 | **Autodiff correct** | `grad` gradients match numerical/reference gradients on CPU **and** GPU (gradient-check suite green) | 🔶 CPU autodiff exists; GPU + full gradient-check pending |
| 5 | **Full-language trust** | the diverse-double-compile passes over a **feature-diverse** corpus (beyond i32) | 🔶 i32 DDC ✅ (5/5 audits); feature-diverse corpus pending |
| 6 | **Python-free + raw-binary** | the **entire** toolchain (compiler, test runner, build) is Helix/seed/ladder only; **zero `.py`** in the live toolchain; reproducible from hex0 | 🔶 **reference compiler DELETED 2026-05-31 (K4)**; mint verified Python-free; remaining: de-Python `assemble_k1.py` + DDC harnesses + 5 dev scripts, + port test-infra to Helix |
| 7 | **Usable stdlib + toolchain** | documented stdlib (collections, math, strings, I/O, tensor/ML ops the capstone needs) with passing tests; self-sufficient driver / test-runner / module system in Helix | 🔶 stdlib campaign in progress |
| 8 | **Design frozen (v1.0 spec)** | a written language reference; syntax/semantics committed; no breaking changes after v1.0 | 🔶 near-stable (self-host parity); not formally frozen |

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
- 🔶 **The gate (#3) — FIRST-LIGHT CROSSED (2026-06-01).** kovc-emitted PTX runs correctly on the RTX 3070 (`vector_add`, independently audited PASS/HIGH with negative controls). The GATE *mechanism* — a raw-binary-bootstrapped Helix toolchain driving real GPU execution end-to-end — is proven. Remaining for full #3 is **breadth**: general tiled GEMM (shared-memory + barriers, not register-only) + the transformer-op corpus on HW (P5). See `docs/HELIX_GPU_FIRSTLIGHT.md`.
- 🔶 **Remaining** — full-feature compile+run + feature-diverse DDC (#1,#2,#4,#5); *finish* Python removal (#6 — compiler gone; still to de-Python `assemble_k1.py` + the DDC harnesses + 5 dev scripts, and port test-infra to Helix); stdlib + spec freeze (#7,#8).

```
critical path:
  GPU executes (#3)  →  single-GPU training loop  →  capstone transformer converges  =  HELIX v1.0 DONE
  (in parallel: finish #1,#2,#4,#5,#6,#7,#8)
```

When this line is crossed, the substrate is proven and the AI-building phase begins.
