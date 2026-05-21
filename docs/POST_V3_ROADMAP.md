# Helix — Post-v3.0 Roadmap (v4 → v9)

**Last updated:** 2026-05-21 · **Status:** proposal · **Supersedes:** the
earlier v4–v8 draft.

This roadmap covers Helix *after* v3.0. It starts only once the v3
Stage-222 cutover/tag gate is genuinely green (Phase E + Phase F of
`docs/V3_PLAN.md` are still in progress). It is **direction, not
commitment**: commit hard to v4; treat v5–v9 as a sketch and re-plan
after v4 ships with real-world feedback.

It is grounded in a short research pass — compiler-correctness practice
(Alive2, CompCert/CakeML, Csmith/YARPGen, Diverse Double-Compiling),
the MLIR→GPU path (the `gpu`/`nvgpu` dialects, IREE, Triton), and
new-language adoption evidence (Rust/Go/Zig trajectories). Sources are
cited inline.

---

## 0. What changed from the v4–v8 draft, and why

The earlier draft's instincts were right — credibility-first, no
overclaiming, falsifiable exit gates. Seven changes make it
shippable and close two real gaps:

1. **Every version now ships an external proof point** a stranger can
   verify or run (rule R1) — not just inward-facing infrastructure.
2. **A minimal real-hardware proof is pulled into v4.** Research
   confirms the MLIR `gpu-lower-to-nvvm-pipeline` → cubin path is
   mature enough for a thin "one workload, measured, on a real GPU"
   proof today. "Trusted" must include "demonstrably runs on real
   silicon" — otherwise v4–v5 are two versions with no outside proof.
3. **Formal mechanized verification moves OFF the release path** into
   a parallel research track (Track B). CompCert was ~42k lines of
   Coq / ~3 person-years for *one* language and limited targets; no
   mainstream compiler (GCC, LLVM, Rust, Go) is formally verified.
   v5 instead leads with **translation validation + differential
   testing + Diverse Double-Compiling** — the pragmatic ~80%-of-the-
   credibility path real compilers actually use.
4. **An Ergonomics & Adoption lane runs through every version** (Track
   A). The draft was a textbook "trust museum" risk: years of SBOMs
   and proofs while the *adoptable* surface (LSP, errors, package
   manager, docs) stays empty and nobody uses it. Adoption evidence is
   blunt — languages compete on ecosystem and tooling, not syntax.
5. **New insight — the AI-corpus barrier.** New languages are now
   disadvantaged because coding assistants lack training data for
   them. Machine-readable, LLM-ingestible docs and examples are a
   deliberate v4 deliverable, not an afterthought.
6. **The draft's v8 is split.** Six domain packs in one version is a
   portfolio, not a version: v8 = *one* domain proven end-to-end; v9 =
   the Kovostov/AGI layer (the actual top-line vision) as its own
   version, no longer gated behind six domain packs.
7. **"Truth reset" becomes permanent** — a continuous CI gate (rule
   R2), not a one-time v4 task. The project already does this for live
   test counts (`scripts/helix_status.py`); generalize it.

---

## 1. Premise

After v3, Helix's bottleneck is **credibility** — but credibility is
not only trust infrastructure. A compiler that is verified, attested,
and self-hosted but that nobody can *use* and that doesn't run *fast*
on real hardware is a museum piece. Real credibility =

> **it is honest** · **it runs on real hardware** · **people can
> actually build real programs with it.**

Two failure modes to avoid in equal measure: **overclaiming** (the
v3-era risk) and the **trust museum** (years of provenance machinery,
zero users).

---

## 2. Standing principles (apply to every version)

Carried from v3: **fail-closed** (never emit wrong output — an
unsupported construct raises), **additive** (a new path never breaks
the old one until parity-gated), **mock-path discipline** where a real
toolchain is unavailable.

New, enforced every version:

- **R1 — External proof point.** Every version delivers something an
  outsider can independently verify or run. No version is "done" on
  internal say-so.
- **R2 — Continuous truth-reset.** README, QUICKSTART, website facts,
  benchmarks, test counts, and known-limits are *generated from
  commands and artifacts*; doc/reality drift is a CI failure, not a
  cleanup task.
- **R3 — Adoption is not deferrable.** Track A (below) ships in every
  version. Tooling and ergonomics are the entry gate, not a polish
  phase.

---

## 3. Two continuous tracks (parallel to the numbered versions)

**Track A — Ergonomics & Adoption (interleaved through v4–v9).**
The evidence: tooling, error-message quality, docs, and a killer use
case drive adoption; raw language features do not
([Meyerovich & Rabkin adoption study](https://developers.slashdot.org/story/12/03/16/0240232/why-new-programming-languages-succeed-or-fail);
[Google's Rust journey](https://opensource.googleblog.com/2023/06/rust-fact-vs-fiction-5-insights-from-googles-rust-journey-2022.html)).
Per-version items appear under each version below; the spine is: a
language server, error messages treated as a product surface, a
package manager + build tool bundled with the toolchain, LLM-ingestible
docs, and at least one compelling example.

**Track B — Formal Verification (research track, NOT a release gate).**
Freeze a small stable **Core IR** subset (in v5); pursue mechanized
semantic preservation for *that subset only*, in parallel, as capacity
allows. CompCert and CakeML are north stars, not promises. Track B
never blocks a release; it can be promoted to a gate only if a
specific high-stakes customer or domain hard-requires it.

---

## 4. The versions

### v4 — Trusted, Usable, Running Public Compiler

**Goal:** make the v3 MLIR/LLVM rewrite boring, validated, releasable,
publicly honest — *and* demonstrably usable *and* demonstrably running
on real hardware. This is the heaviest version (a real first public
release is a lot); treat its workstreams as independently shippable
increments (v4.0 release-engineering, v4.1 adoption surface, …) if
scope demands.

**Do — trust & correctness:**
- Post-cutover quarantine — keep the home-grown path as a debug /
  reference path for one release cycle, then remove it.
- Close the residual Tile-IR → MLIR op gaps as far as is tractable
  (async, `memref` movement, `matmul`/`reduce`, the HBM-index bridge);
  whatever genuinely needs more design stays **fail-closed and
  documented**, never silently half-supported.
- Real validation, not mock-only: drive MLIR's
  [dialect-conversion model](https://mlir.llvm.org/docs/DialectConversion/)
  (it fails when illegal ops remain) and adopt **translation
  validation** —
  [Alive2](https://github.com/AliveToolkit/alive2)-style refinement
  checks on the LLVM IR, plus the project's structural checks on MLIR.
- Translation-validation manifests per build: source hash, IR hashes,
  MLIR, LLVM, the pass pipeline, tool versions, verifier status.
- Truth reset (and make it permanent per R2): regenerate
  README/QUICKSTART/website/roadmap/benchmarks/test-counts/known-limits
  from commands and artifacts; wire drift detection into CI.
- Release provenance: SBOM ([SPDX](https://spdx.github.io/spdx-spec/v3.0.1/)),
  [SLSA provenance](https://slsa.dev/spec/v1.2/provenance), GitHub
  attestations, [reproducible-build](https://reproducible-builds.org/docs/definition/)
  gates.

**Do — the minimal real-hardware proof (pulled in early):**
- One real NVIDIA workload, compiled end-to-end through the
  MLIR → NVVM → cubin path
  ([MLIR `gpu` dialect](https://mlir.llvm.org/docs/Dialects/GPU/)),
  run on a GPU CI runner, numerically checked against a CPU reference,
  with a *published wall-clock number*. Not "fast" yet — just
  "provably runs on real silicon." Every piece is off-the-shelf; this
  does not need to wait for v6.

**Do — Track A (the adoption minimum bar):**
- Single-binary install.
- Package manager + build MVP: `Helix.toml`, `Helix.lock`, workspaces,
  deterministic archives.
- LSP basics — one [Language Server Protocol](https://microsoft.github.io/language-server-protocol/)
  implementation + a VS Code extension (one server serves many
  editors).
- Error messages treated as a **product surface** — clear, actionable
  diagnostics (Rust-grade is the bar; this measurably moves adoption).
- A real tutorial + one nontrivial GPU/AI example program.
- **LLM-ingestible docs** — structured, machine-readable docs and
  examples so coding assistants can learn Helix.

**Exit gate:** a stranger can install Helix, verify the release
(SBOM + provenance), build the examples, **run one real GPU workload
and see a measured number**, use an LSP in their editor, hit a genuine
error and get a good message, follow a tutorial, and inspect exactly
what is shipped vs. experimental — with **no hidden fallback paths**.

### v5 — Self-Hosting & Translation-Validated Core

**Goal:** make Helix trust itself — self-hosted, and validated by the
pragmatic, affordable correctness stack.

**Do:**
- Complete the bootstrap chain toward a self-hosted `kovc`.
- Move production frontend / driver / release logic out of Python;
  Python remains as a reference oracle and fuzzer, not as production
  authority.
- **Translation validation as the release gate** — per-compilation
  correctness checks on the IR transforms.
- **Differential testing** — Csmith / YARPGen-style random program
  generation cross-checked against a reference compiler (this class of
  tool has found 200+ real bugs in GCC/LLVM/ICC).
- **Diverse Double-Compiling** + bit-identical N-generation fixed-point
  self-host gates ([DDC](https://dwheeler.com/trusting-trust/)).
- Freeze and document a small, stable **Core IR** subset — the
  foundation Track B can later target.
- Track A: a debugger story; standard-library breadth grows; more
  worked examples.

**Exit gate:** the release build is self-hosted, reproducible, and
bit-identical across self-host generations; Python is not in the
trusted/production path; the trusted computing base is explicitly
documented; translation validation + differential testing run in CI on
every build.

### v6 — Performance & Hardware Platform

**Goal:** turn "provably runs" (v4) into "runs serious workloads
*fast*, across *honest* targets."

**Do:**
- A real **schedule layer** — Triton-style: tiles, warps/subgroups,
  shared memory, async copies, tensor-core MMA selection, autotune
  records. Performance is a separate IR layer (an algorithm/schedule
  split), not a backend flag
  ([Triton](https://triton-lang.org/main/programming-guide/chapter-1/introduction.html)).
- NVIDIA-first production path under **hardware CI** — SASS
  disassembly checks, profiler smoke tests, numerical parity vs. CPU.
  No multi-vendor claims until the CUDA/NVVM path is real under HW CI.
- A HAL-style runtime seam (sketched in v4, matured here) so target
  abstraction is not retrofitted — IREE's lesson: split scheduling IR
  from per-target executable IR
  ([IREE deployment](https://iree.dev/guides/deployment-configurations/)).
- ROCm only with explicit `gfx*` target manifests; an IREE-style
  portability lane for Vulkan/SPIR-V, Metal, WebGPU, CPU.
- Per-backend capability manifests; fail-closed unsupported features.
- Track A: the **killer demo** — one showcased GPU/AI workload that is
  genuinely compelling, not a toy.

**Exit gate:** at least one NVIDIA path is production-real (fast,
hardware-CI'd, disassembly- and profiler-checked); at least one
portability path is honest; every backend carries a capability
manifest and fails closed on unsupported features.

### v7 — AI-Native Runtime & Evidence

**Goal:** make Helix the language/runtime where AI work is
evidence-bearing by construction. Ship this as ordered increments —
do the spine first, do each piece *well*, do not start all four at
once.

**Do (in order):**
1. **The provenance spine** — a [W3C PROV](https://www.w3.org/TR/prov-dm/)-compatible
   event graph for builds, tools, model calls, data mutations,
   verifier results, and releases. Everything else hangs off this.
2. Deterministic scheduler; snapshot / rollback; resource & deadline
   metadata.
3. Agent identity and scoped authority for autonomous actions
   ([NIST AI Agent Standards](https://www.nist.gov/artificial-intelligence/ai-agent-standards-initiative)).
4. `Confidence<T>` / `Uncertain<T>`-style primitives, calibration
   reports, abstention / referral APIs.
- Track A: the runtime-evidence APIs get first-class docs + examples.

**Exit gate:** every meaningful AI/runtime action can be replayed,
audited, attributed, risk-scored, and rejected by policy.

### v8 — One Domain, Proven End-to-End

**Goal:** prove the *whole stack* in **one** high-stakes domain,
credibly — not a thin pass over six.

**Do:**
- Pick a single domain where Helix's GPU/AI + evidence story is
  strongest (e.g. scientific computing or one regulated ML domain).
  Do it fully: a domain pack, a safety case, a hazard log, an
  intended-use manifest, human-oversight rules, an evaluation suite,
  and a regulatory dry-run bundle —
  [NIST AI RMF](https://www.nist.gov/itl/ai-risk-management-framework)
  as the general frame, [FDA GMLP](https://www.fda.gov/media/153486/download)
  as the precedent if the domain is medical.
- Extract the domain pack into a **reusable template** so subsequent
  domains are productization, not new roadmap versions.

**Exit gate:** a real, audited domain demo where code, data, model
behavior, proof/test evidence, supply chain, and human authority are
inspectable end-to-end.

### v9 — Kovostov / AGI Layer

**Goal:** the top-line vision — Kovostov built *on* Helix, as the
trustworthy substrate for self-improving AI work. The most speculative
version; its contents are deliberately a sketch and will be re-planned.

**Do:**
- Build Kovostov on Helix using v7's evidence/runtime primitives:
  verifier-gated self-improvement, rollback, provenance, public-data
  constraints, scoped agent authority.

**Exit gate:** the Kovostov/AGI layer runs on Helix with every
self-improvement step verifier-gated, provenance-tracked, rollback-able,
and policy-bounded — inspectable end to end.

---

## 5. Sequencing, risk, and the re-plan checkpoint

**Dependency spine:** v4 (trusted + usable + runs) → v5 (self-trust) →
v6 (fast + portable) → v7 (evidence runtime) → v8 (one domain) → v9
(AGI layer). Track A is interleaved throughout; Track B runs in
parallel and gates nothing.

**The single biggest risk is scope.** Each version must stay
shippable. If a version balloons, *split it* (v4.0/v4.1, …) rather
than overstuffing — five right-sized versions beat three heroic ones.

**Re-plan after v4.** Once Helix has a real public release, real
users, and real benchmarks, the specifics of v5–v9 *will and should*
change. Do not over-invest in their exact contents now. This document
is a direction; v4 is the commitment.

**The one-line test for the whole roadmap:** v4 should make people
*trust* the compiler and be *able to use it*; v5 should make Helix
*trust itself*; v6 should make it *fast on real machines*; v7 should
make AI work *auditable*; v8 should *prove the stack* in a domain that
matters; v9 should *be the vision*. If a planned task does not move one
of those, cut it.

---

## 6. Research basis

This roadmap was grounded by a 2026-05-21 research pass; the load-
bearing findings:

- **Verification.** Translation validation (Alive2 + Z3 refinement
  checks on LLVM IR) is automatic, needs no compiler changes, and
  found 47 bugs across LLVM's own tests — high per-build assurance at
  low cost. Full mechanized verification (CompCert ≈ 42k lines Coq /
  ≈ 3 person-years; CakeML similar) does not scale to an evolving
  multi-target compiler and is why no mainstream compiler is verified.
  → lead with translation validation + differential testing
  (Csmith/YARPGen: 200+ real compiler bugs found) + DDC; formal proof
  is a research track.
- **MLIR → GPU.** The `gpu-lower-to-nvvm-pipeline` → cubin path is
  documented and usable today (the `gpu` dialect is still marked
  experimental; the pipeline does not auto-parallelize; `vector`→
  intrinsic lowering is the rough edge). A *minimal* measured-on-real-
  hardware proof needs only off-the-shelf pieces → it can land in v4.
  Production speed (the Triton-style schedule layer, the IREE-style
  HAL seam, mature hardware CI) is the separate, larger v6.
- **Adoption.** Languages compete on *ecosystem and tooling*, not
  syntax; error-message quality measurably drives adoption; a killer
  use case is decisive; new languages are now also handicapped by
  coding assistants lacking a training corpus. → an LSP, good
  diagnostics, a package manager/build tool, LLM-ingestible docs, and
  one compelling example are part of the *first* public release, not a
  later phase.
