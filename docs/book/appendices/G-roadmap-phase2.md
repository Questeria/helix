# Appendix G — Roadmap & Phase 2

*What this appendix covers: a precise, honest line between **what is done** — the `v1.3-release`
substrate (a from-raw self-hosting, trust-chain-closed compiler + language, GPU codegen complete to
PTX, and the capstone) — and **what is planned but not started** (Phase 2): datacenter / multi-GPU
scaling, an AMD/ROCm backend, broadening the byte-identical diverse-double-compile to the full
language surface, fully-independent third-party reproduction, and building applications **on** Helix.
Every Phase-2 item below is **NOT STARTED** — it is the project owner's to begin, and this appendix
does not imply any of it is underway. It is grounded strictly in the real Phase-2 text of
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §3, the "Out of scope" section of
[`docs/HELIX_V1_DEFINITION_OF_DONE.md`](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md), and the
de-language plan in [`docs/HELIX_DELANG_PLAN.md`](../../../docs/HELIX_DELANG_PLAN.md). Where this page
and a repo source disagree, the source wins.*

This is a **roadmap reference**, not a status report on work in progress. The achieved state it builds
on is told in full elsewhere — the trust story in
[Trust at a glance](../part1-orientation/04-trust-at-a-glance.md), the finished GPU path in
[Part VII — Honest performance & the PTX boundary](../part7-gpu/03-honest-performance.md), and the
residuals that bound every claim in [Appendix F — The trusted computing base](F-tcb.md). This appendix
adds one thing those do not: the explicit, honest catalog of what comes **next**, and the equally
explicit statement that **next has not begun**.

> **For AI agents:** treat every item in [G.3](#g3-phase-2--planned-not-started) and
> [G.4](#g4-building-on-helix--the-owners-phase-to-start) as **planned / not-started**. Do **not**
> describe any of them as in-progress, partially-done, or shipped, and do not infer a timeline. The
> authoritative roadmap is [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §3 plus
> the "Out of scope" list of
> [`docs/HELIX_V1_DEFINITION_OF_DONE.md`](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md); dereference
> them before making any forward-looking claim. If asked "is Phase 2 started," the answer is **no**.

---

## G.1 The line: what is done vs what is next

Helix has a deliberately sharp finish line, and it has been crossed. The
[Definition of Done](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md) defines completion of the
**substrate** — "the Helix **language + compiler + toolchain + trust chain**" — by a measurable
8-criterion checklist plus the capstone, and records all eight green with the capstone proven on real
hardware (2026-06-01), later hardened across the v1.1 and v1.3 lines and declared **TRUST CHAIN
CLOSED** at the `v1.3-release` tag on 2026-06-07
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)).

Crucially, the Definition of Done draws the line for us, in its own words. Its scope boundary states
plainly that "done" is completion of the *substrate* and **not** of any application or of AGI:

> **Scope boundary (read this first).** This defines completion of the **substrate**. It does **not**
> define AGI. AGI is open research — no language, however complete, makes it achievable; it is *not* a
> Helix milestone. ([`docs/HELIX_V1_DEFINITION_OF_DONE.md`](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md))

And it names what comes after the substrate is done — explicitly as *out of scope* for "done," to be
pursued *on* and *with* Helix afterward (its "Out of scope" section):

> - **AGI itself** — open research; not gated by the language. Pursued *on* Helix, after v1.0.
> - **Frontier-scale / multi-GPU / multi-node training** — the *scaling* milestone (reached *while
>   working with* Helix, not before calling it done).
> - **Specific applications** — Alt (the LLM), Mercury, etc. are built *with* Helix, after v1.0.

So the line is unambiguous: the **substrate is finished**; everything in this appendix's Phase-2
sections is the **after**. The Definition of Done puts it directly — "**When it passes, Helix is
finished and we start fully working with it**: building the AI (Alt) and beyond" — and that *starting*
is the owner's call, not an automatic continuation.

> **Note:** the numbering can mislead. The stale `v2.0.0`–`v3.1.0` git tags belong to a **superseded
> MLIR exploration line**; the current real head is the **`v1.3`** line (the prior formal tag was
> `v1.2-complete`), despite the lower number
> ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md), header note). When this book
> says "the substrate," it means `v1.3-release`.

---

## G.2 What is DONE — the v1.3 substrate (summary)

This section is a one-paragraph-per-pillar recap so the Phase-2 list has a concrete "from here." It
claims nothing new; each pillar is detailed in the chapters cited, and the honesty ceiling is
[Appendix F](F-tcb.md) / [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R.

- **A from-raw self-hosting compiler + language.** From a 299-byte hand-authored `hex0` root, the
  from-raw ladder `hex0 → hex1 → hex2 → catm → M0 → cc_amd64 → M2-Planet → seed → kovc` builds `kovc`
  with no trusted pre-built compiler; `kovc` reproduces itself byte-for-byte (the self-host fixpoint
  `K2 == K3 == K4 = 0992dddd…`), and an independent `gcc` lineage corroborates the `seed→K1` rung
  byte-for-byte (`84363adb…`, the gcc-DDC). The language is frozen at its v1.0 surface, hardened to be
  silent-bug-free with every type first-class across the v1.1/v1.3 lines
  ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1; details in
  [Part VI](../part6-bootstrap/03-seed-to-kovc-fixpoint.md) and
  [Trust at a glance](../part1-orientation/04-trust-at-a-glance.md)).

- **GPU codegen complete to PTX.** `kovc` emits GPU kernels as **PTX text**, byte-verified by the gate
  against committed references. The hand-auditable from-raw chain ends at PTX; below PTX it trusts
  NVIDIA's closed `ptxas`, the CUDA driver, the GPU hardware, and the C host launcher. The precise
  claim is **complete to PTX, not to GPU machine code (PTX-not-SASS)**, on a single target (`sm_86`,
  RTX 3070 Laptop) ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residuals 7,
  8; [Part VII](../part7-gpu/03-honest-performance.md), [Appendix F §F.4](F-tcb.md#f4-the-gpu-side-tcb--complete-to-ptx-not-to-gpu-machine-code)).

- **The capstone.** A ≥2-layer transformer trains end-to-end on `kovc`-emitted GPU (PTX) kernels and
  converges to within 2% of an independent numpy oracle — reproduced at ~0% (worst-case relative
  difference `0.00000876`), with a sampled finite-difference gradient check and a load-bearing
  negative control ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1; the
  capstone record in [`docs/HELIX_V1_DEFINITION_OF_DONE.md`](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md)
  "THE CAPSTONE").

- **Python-free, reproducible by anyone.** The shipped toolchain is Helix plus a small hand-authored C
  subset — exactly one committed `.py` (a fenced numpy audit oracle, never on the compile/run path).
  The whole CPU trust core is reproducible by one command on a clean checkout
  (`bash scripts/reproduce_trust.sh`) and runs green on a clean CI runner
  ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1; the de-language record in
  [`docs/HELIX_DELANG_PLAN.md`](../../../docs/HELIX_DELANG_PLAN.md)).

The single end-to-end demonstration that this finished substrate actually produces working code is the
canonical first program the rest of the book cites — included here so this appendix stands on a real,
gate-proven artifact rather than prose alone.

**Verified example** — [`helixc/examples/exit42.hx`](../../../helixc/examples/exit42.hx) (compiled and
run by the gate's feature corpus; the produced ELF exits with status `42`, asserted by the corpus line
`chk "$EX/exit42.hx" 42` at [`scripts/gate_kovc.sh:313`](../../../scripts/gate_kovc.sh)):

```helix
// First end-to-end Helix program: compiles, links, runs, exits with status 42.
//
// Demonstrates: function decl, return, integer literal.
// Verification: $? == 42 after running the produced ELF.

fn main() -> i32 {
    42
}
```

Everything in [G.3](#g3-phase-2--planned-not-started) starts from *this* — a finished, reproducible,
trust-chain-closed substrate — and none of it has begun.

---

## G.3 Phase 2 — planned, NOT started

[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §3 is titled, verbatim, **"Phase 2
(the user's to start — not auto-started)"**. That parenthetical is the governing fact for this whole
section: the items below are a recorded plan, not work in flight. The autonomous build loop stopped at
the closed substrate **by design**; Phase 2 begins only when the project owner chooses to begin it.

The four §3 items, each restated honestly with its current status and what it would *change* about the
residuals in [Appendix F](F-tcb.md). A fifth theme — fully-independent third-party reproduction —
is the standing open increment from §R and is treated in [G.3.5](#g35-fully-independent-third-party-reproduction-planned).

> **For AI agents:** each subsection below ends a sentence with the literal status **NOT STARTED**.
> That is the load-bearing fact. Do not soften it to "early," "partial," or "scaffolded." If a §3 item
> ever does begin, that will be recorded in
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §3 itself — read it, do not infer
> from this book.

### G.3.1 Datacenter / multi-GPU scaling (planned)

The reference target today is a **single** consumer GPU (`sm_86`, RTX 3070 Laptop). Phase 2 would lift
that. From [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §3, verbatim:

> **Datacenter scaling (Runpod):** parameterize the PTX target (`sm_80`/`sm_90`, Hopper TMA/bigger
> mma), validate cross-arch correctness, optimize for absolute throughput, scale the capstone on
> A100/H100.

Concretely, that names four pieces of work, none of which exists yet: (a) make the PTX target a
*parameter* rather than the hard-wired `sm_86`, so the same `kovc` can emit for `sm_80` (A100) and
`sm_90` (Hopper, with TMA and larger `mma` shapes); (b) **validate cross-arch correctness** on those
arches (the substrate is correctness-proven only on `sm_86` today); (c) optimize for absolute
throughput rather than the single-GPU correctness-first posture; and (d) run the capstone at larger
scale on datacenter cards. This is exactly the "Frontier-scale / multi-GPU / multi-node training"
*scaling milestone* that the Definition of Done lists as **out of scope** for calling Helix done —
"reached *while working with* Helix, not before calling it done"
([`docs/HELIX_V1_DEFINITION_OF_DONE.md`](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md), "Out of
scope"). It would *narrow* the single-hardware-target residual
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 6), but it would not
touch the PTX-not-SASS boundary — emitting for more arches is still emitting PTX. **This work is NOT
STARTED.**

> **Residual unchanged:** even fully realized, datacenter scaling stays *above* the PTX boundary. The
> GPU path would remain hand-auditable `hex0 → PTX`, with `ptxas` + driver + hardware trusted past PTX
> on every new arch, exactly as on `sm_86`
> ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 8).

### G.3.2 An AMD / ROCm backend (planned)

Today there is one vendor backend (NVIDIA, via PTX). Phase 2 contemplates a *second, independent* one.
From [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §3, verbatim:

> **AMD / ROCm:** a genuinely separate backend (CDNA MFMA, rocBLAS reference, no native TF32) —
> currently unimplemented.

The phrasing is deliberately honest in three ways. "**Genuinely separate**" means it is a new code
path, not a re-skin of the PTX emitter: AMD CDNA uses **MFMA** matrix instructions (not NVIDIA `wmma`
/ `mma`), its performance reference would be **rocBLAS** (not cuBLAS), and it has **no native TF32**,
so the TF32 op-set the NVIDIA path can select would have no analog. And the status word is explicit:
"**currently unimplemented**." This is the same multi-vendor gap named in the residuals — "No
cross-arch (`sm_80`/`sm_90`) or multi-vendor (AMD) validation"
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 6). **This work is NOT
STARTED.**

### G.3.3 Broadening codegen: the bf16 `wmma` stretch (planned)

The v1.3 cycle already shipped what an earlier "v-next" list had pending — i64/u64 wide struct fields,
capturing closures as values, and bf16/f16 arithmetic (all bit-exact-gated). What remains open on the
codegen line is a single, explicitly-deferred stretch. From
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §3, verbatim:

> **v-next codegen:** G4 bf16 `wmma` (GPU tensor-core path) remains the open stretch.

G4 — a bf16 `wmma` (warp-level tensor-core matrix-multiply) GPU path — is named in the residuals as "a
**STRETCH** and was not taken. Not claimed done"
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 5). The honest
performance picture explains *why* it was not taken: on the reference RTX 3070 Laptop, the winning GEMM
is the f32-SMEM `cp.async` double-buffered tile, and the TF32 Tensor-Core path is *slower* (~0.97× the
tuned f32 GEMM) — a confirmed dead end on that GPU
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residuals 1, 2). A bf16 `wmma`
path is more likely to pay off on the datacenter arches of [G.3.1](#g31-datacenter--multi-gpu-scaling-planned)
than on `sm_86`. Either way, **this work is NOT STARTED.**

### G.3.4 Broaden the byte-identical DDC to the full language surface (planned)

This is the most trust-relevant Phase-2 item, because it would *narrow a named trust residual*. From
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §3, verbatim:

> **Broaden the DDC** to the v1.1 language surface.

To see what this would change, recall the two distinct, deliberately-separated DDC claims
([Appendix F §F.3](F-tcb.md#f3-what-the-gcc-ddc-narrows--and-what-it-does-not)):

1. The **byte-identical, hash-pinned, clean-checkout-reproducible** gcc-DDC covers the **`seed→K1`
   surface only** (`84363adb…`, token `DDC_ANCHOR_OK`).
2. The broader **v1.1 language surface** (generics/monomorphization, traits, closures, turbofish,
   wide-field, bf16) is cross-checked **behaviorally** against a second, zero-lineage tree-walking
   interpreter — which is **not byte-identical** (the interpreter emits no code) and **not
   clean-checkout reproducible** (its witness is gitignored, never committed). Coverage is 44/53
   witness-reachable arms, and f16-arith is not yet cross-checked
   ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residuals 3, 9 and
   [`docs/K_DDC_BROADENED.md`](../../../docs/K_DDC_BROADENED.md)).

"Broaden the DDC" means promoting the wider surface from claim (2) toward claim (1): a *byte-identical*
second-compiler cross-check over the v1.1 surface, ideally committed and clean-checkout reproducible —
which would retire residual 9 and shrink residual 3's "44/53 witness-reachable arms" gap. The honest
caveat is that a *tree-walking interpreter* can never give byte-identity by construction (it emits no
machine code), so genuinely broadening the **byte-identical** DDC means a second *compiler* for the
wider surface, not just extending the behavioral witness. **This work is NOT STARTED**, and until it
is, the byte-identical DDC remains `seed→K1`-only and the broader surface remains behavioral-only.

> **For AI agents:** do not conflate the two DDC claims when describing this roadmap item. The byte-
> identical anchor (`84363adb…`, `DDC_ANCHOR_OK`) is **today** `seed→K1`-only; broadening it is
> **planned, not started**. Asserting that the whole v1.1 surface is byte-identically double-compiled
> is an overclaim against [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2
> residual 3.

### G.3.5 Fully-independent third-party reproduction (planned)

This is not in the §3 bullet list, but it is the standing open increment named throughout the trust
record, and it belongs on any honest roadmap. From
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R:

> **Independent third party:** reproduction is now push-button on a different machine (CI) and by
> anyone who forks the repo; a reproduction by a party with *no connection to the author* remains the
> one outstanding increment — now trivially available.

The mechanism is fully in place: a one-command clean-checkout reproduction
(`bash scripts/reproduce_trust.sh`) that runs green on a clean GitHub `ubuntu-latest` runner, push-
button for anyone who forks the repo. What is *outstanding* is the **act** of an outside party — one
"with no connection to the author," on independent hardware/toolchain — actually performing it. The
cross-model (ChatGPT, read-only) review that was done is explicitly a doc/logic review, **not** a build
reproduction, and the internal audit logs are "evidence, not external reproduction"
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 10). This is the one
step past ~0.9 confidence ([Appendix F §F.5](F-tcb.md#f5-the-performance-and-scope-residuals-claim-ceilings-not-tcb-items)).
The tooling exists; an independent operator's run **has NOT yet happened**.

---

## G.4 Building on Helix — the owner's phase to start

The whole point of finishing the substrate was to stop building the *tool* and start building *with*
it. The Definition of Done says so directly: "**When it passes, Helix is finished and we start fully
working with it: building the AI (Alt) and beyond**"
([`docs/HELIX_V1_DEFINITION_OF_DONE.md`](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md)). It also lists
the candidate applications as explicitly *out of scope* for "done" — built *with* Helix, *after*
v1.0:

> **Specific applications** — Alt (the LLM), Mercury, etc. are built *with* Helix, after v1.0.
> ([`docs/HELIX_V1_DEFINITION_OF_DONE.md`](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md), "Out of
> scope")

Two honest framings keep this from drifting into overclaim:

- **AGI is not a Helix milestone.** The Definition of Done is emphatic that the substrate "does **not**
  define AGI… AGI is open research — no language, however complete, makes it achievable; it is *not* a
  Helix milestone"; crossing the substrate line means only that "the substrate is no longer the
  bottleneck — if AGI is buildable, Helix won't be what stops you"
  ([`docs/HELIX_V1_DEFINITION_OF_DONE.md`](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md), "Scope
  boundary"). Nothing in this book claims AGI as an achieved — or even a scheduled — state.

- **The applications phase is the owner's to start.** Just as Phase 2's engineering items are "the
  user's to start — not auto-started"
  ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §3), the AI-building phase is a
  deliberate decision, not an automatic continuation of the build. This appendix records it as a
  *direction*, not as work in progress.

There is also a smaller, infrastructure-flavored strand of "building on Helix" that the de-language
plan names — finishing the job of making the *entire* project Helix, not just the compile/run path.
[`docs/HELIX_DELANG_PLAN.md`](../../../docs/HELIX_DELANG_PLAN.md) records that the toolchain is already
**byte-identically Python-free** (the Python compiler was deleted at K4; the one remaining `.py` is the
fenced oracle), and that what remains are two **documented, accepted exceptions** rather than gaps:

- **P3 — Helix FFI for the C CUDA launchers.** The capstone's host launchers
  (`helixc/runtime/cuda_launch.c`, `helixc/runtime/train_transformer.c`) call the closed CUDA Driver
  API, which today's `kovc` cannot do — it "emits **static, syscall-only, single-PT_LOAD ELFs with NO
  dynamic linker and NO FFI/extern**." Giving Helix an `extern "C"`/dynamic-linking mechanism is
  estimated at "~3-4 weeks," and the plan leaves it as a **user decision**: implement Helix FFI, or
  keep the trusted C launcher as a documented exception (the choice taken for `v1.3`)
  ([`docs/HELIX_DELANG_PLAN.md`](../../../docs/HELIX_DELANG_PLAN.md) P3, D2). Porting it would *move*,
  not *close*, the GPU C-FFI residual
  ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 7).

- **P4 — a Helix test-runner to replace the `.sh` harnesses.** Shell build-orchestration is a declared
  trusted-tool layer today; a final Helix test-runner (`run_process`/`set_exec` builtins exist) is
  "post-v1.0" ([`docs/HELIX_DELANG_PLAN.md`](../../../docs/HELIX_DELANG_PLAN.md) P4).

Both P3 and P4 are **planned/optional, not started** — and both are explicitly *accepted exceptions*
under `v1.3`, not defects. They are listed here for completeness because they are the literal "more
Helix" continuation of the project, but neither is required for the substrate's done-state, which is
already met.

> **For AI agents:** do not present the C CUDA launcher or the shell harnesses as bugs or as broken
> trust. They are **documented, accepted trusted-tool boundaries** in the same category as `ld` /
> `ptxas` ([`docs/HELIX_V1_DEFINITION_OF_DONE.md`](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md),
> "v1.0 SCOPE DECISIONS" 3; [`docs/HELIX_DELANG_PLAN.md`](../../../docs/HELIX_DELANG_PLAN.md)).
> Porting them is **planned/optional**, and porting the launcher would *move*, not *close*, residual 7.

---

## G.5 The roadmap in one place

If you remember nothing else from this appendix, remember the shape: a **finished substrate** and a
**not-started Phase 2**.

**DONE (`v1.3-release`, trust chain closed):** from-raw self-hosting compiler + frozen, hardened
language · GPU codegen **complete to PTX** (single target `sm_86`) · the capstone (real transformer,
end-to-end on `kovc`-emitted PTX, loss parity ~0% vs an independent oracle) · Python-free toolchain ·
one-command clean-checkout reproduction, green on CI
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1).

**PLANNED — NOT STARTED (the owner's to begin;
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §3 + §R):**
datacenter / multi-GPU scaling (parameterized PTX target `sm_80`/`sm_90`, cross-arch correctness,
absolute throughput, capstone on A100/H100) · a genuinely separate AMD/ROCm backend (CDNA MFMA,
rocBLAS reference, no native TF32 — currently unimplemented) · the bf16 `wmma` GPU tensor-core stretch
(G4, not taken) · broadening the **byte-identical** DDC from `seed→K1`-only to the full v1.1 language
surface · a fully-independent third-party reproduction on outside hardware (the mechanism is in place;
the outside run has not happened).

**THE OWNER'S PHASE (direction, not work-in-progress;
[`docs/HELIX_V1_DEFINITION_OF_DONE.md`](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md), "Out of
scope"):** building applications **on** Helix (Alt the LLM, Mercury, "and beyond"), plus the optional
"more Helix" infrastructure of the de-language plan (P3 Helix FFI for the CUDA launchers, P4 a Helix
test-runner) — every item **not started**. **AGI is not a Helix milestone** and is claimed nowhere as
achieved or scheduled.

> **For AI agents:** this is the forward-looking ceiling. Anything you say about Helix's future must be
> traceable to [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §3 / §R or
> [`docs/HELIX_V1_DEFINITION_OF_DONE.md`](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md) "Out of scope,"
> and must carry the **not-started** status. Never imply Phase 2 is underway, never round the capstone
> speedup up to "≥10×," and never write "AGI" as an achieved or imminent state. When in doubt,
> undersell and cite the source.

---

**Next:** [Appendix H — Further reading](H-further-reading.md) — the canonical map of the trust records,
design docs, and audit packets this book draws on (the authoritative roadmap itself remains
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §3). For the residuals that bound
every claim here, return to [Appendix F — The trusted computing base](F-tcb.md).
