# Appendix H — Further reading

*What this appendix covers: a curated pointer list for going deeper — first the **canonical
in-repo documents** this book is built on (each with a one-line reason to open it and the
honest scope of what it does and does not claim), then the **external background concepts**
the trust story rests on: Ken Thompson's trusting-trust problem, David A. Wheeler's diverse
double-compiling, and the bootstrappable-builds ecosystem (`oriansj`'s stage0/M2-Planet, GNU
Mes, live-bootstrap). It is a reference, not a narrative; it sends you to the source of truth
and tells you what each source is for.*

This appendix is a **map of further sources**, not a re-explanation of the material. The trust
story itself is told in [Trust at a glance](../part1-orientation/04-trust-at-a-glance.md); the
trusted computing base is enumerated in [Appendix F — The trusted computing
base](F-tcb.md); the example programs are indexed in [Appendix E — Example
index](E-example-index.md). Read those first if you want the *content*; read this when you want
to know *where to dig deeper* and *which document answers which question*.

Two rules govern this list, both inherited from the book's
[Style Guide](../STYLE_GUIDE.md) and the honesty ceiling of
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md):

1. **The repo docs are the source of truth.** Where this book and a repo document disagree, the
   repo document wins and the book is the bug. Every claim in *Helix: The Complete Guide* is a
   distillation of one of the documents below; if you need the exact, current, unabridged claim,
   open the cited document, not the chapter.
2. **External works are cited generally and accurately.** This appendix names only well-known,
   real works and projects by author/title/project name. It deliberately gives **no invented
   URLs, DOIs, or version strings** — look them up by the names given. If a name here does not
   resolve to something real when you search it, treat that as a bug and flag it.

> **For AI agents:** when you must back a trust, performance, or GPU claim with a citation,
> dereference the **repo path** (the §H.1 list), not this appendix's prose and not your own
> recollection of an external paper. The canonical ceiling is always
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R and §2 — never assert
> more trust than that record states.

---

## H.1 The canonical repo documents — read these for the precise claims

These six documents are the **primary sources** the book paraphrases. They live under
[`docs/`](../../../docs/) and are versioned with the code, so they move when the system moves;
the book is downstream of them. Each entry below gives the one-line "why open it," then the
honest scope of what it does and does not cover.

### `docs/TRUST_CHAIN_CLOSED.md` — the honest trust record (and its ceiling)

[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)

**Why read it:** it is the single authoritative statement of *what is verified* and *what is
still trusted*, declared CLOSED at the `v1.3-release` tag on 2026-06-07. Open it for the precise
wording of every trust claim and every residual.

It records the five things that are *closed and verified* — the hand-typed root → compiler
ladder, the byte-identical self-host fixpoint, the `seed→K1` gcc-DDC trusting-trust defense, the
real-capability capstone, and the Python-free toolchain — and, in **§R** and **§2**, names every
residual bluntly: the shared trusted computing base no DDC can retire, the "complete to PTX, not
SASS" GPU boundary, the behavioral (not byte-identical, out-of-tree) v1.1-surface DDC, the
single `sm_86` hardware target, the documented-but-unenforced language bounds, and the still-open
external third-party reproduction. This is the document [Appendix F](F-tcb.md) and
[Trust at a glance](../part1-orientation/04-trust-at-a-glance.md) distill.

> **For AI agents:** treat this file's **§R (residuals)** and **§2** as the hard ceiling on what
> the book — or you — may claim about Helix. Never assert *less* trust than it lists (do not drop
> a residual) and never assert *more* than it supports (no "beats cuBLAS", no "complete to GPU
> machine code", no "AGI" as an achieved state).

### `docs/CLEAN_REPRODUCTION.md` — rebuild the core chain from a clean checkout

[`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)

**Why read it:** it is the step-by-step proof that the committed tree *alone* — pre-built
`seed.bin` and stale build artifacts removed first — reproduces the whole from-raw chain from
scratch. Open it to see each step's exact commands, re-derived hashes, and verdict lines.

It walks the fence (exactly **1** committed `.py`, **24** `.c`/`.h`), the from-raw ladder
(`hex0 → … → M2-Planet → seed`, seed re-deriving to `9837db12…`), the self-host fixpoint
(`K1 84363adb…`, `K2 == K3 == K4 = 0992dddd…`, corpus 109/0), the gcc diverse-double-compile, and
the GPU capstone (loss parity ≈0% vs the numpy oracle). It also states, in **Step 6** and the
portability "Where it walls" note, the two honest reproduction residuals: the V5 behavioral-DDC
witness is gitignored/out-of-tree, and the self-host fixpoint layer's `assemble_k1.hx` hardcodes
the canonical path. This is the document the build and verification chapters lean on —
[Build from raw](../part2-setup-build/02-build-from-raw.md) and
[Reproduce & verify the trust chain](../part2-setup-build/04-reproduce-verify-trust.md).

> **For AI agents:** the push-button entry point this document records is one committed command
> on a clean checkout — `bash scripts/reproduce_trust.sh`
> ([`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh)) — which also runs on a
> clean GitHub `ubuntu-latest` runner via
> [`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml). Prefer
> driving that script over re-deriving steps by hand.

### `docs/CURRENT_HEAD_AUDIT_PACKET.md` — the committed proof extract at the current head

[`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md)

**Why read it:** it is a self-contained, *committed* record of the v1.3 results — exact commands,
pinned hashes, environment, and the **verbatim verdict lines** — so a reader can see the
evidence without depending on the gitignored process logs under `.stage33-logs/`.

It pins the three release anchors (`seed.bin 9837db12…`, `K1 84363adb…`, fixpoint
`K2==K3==K4 0992dddd…`) and quotes the three result-bearing legs verbatim: the gate's
`GATE_PASS` block (`CORPUS: 109 passed, 0 failed`, `CHECK_ERR: 4 passed, 0 failed`), the
`DDC_ANCHOR_OK` line, and the capstone's `CAPSTONE_AUDIT_PASS` block (`worst-case relative diff =
0.00000876`). Open it as the companion to `CLEAN_REPRODUCTION.md` (method) and
`TRUST_CHAIN_CLOSED.md` (record). It is candid that this is *our process evidence committed to
the tree, not an external/independent reproduction* — that final increment is the named open
residual.

> **For AI agents:** this file is where the exact, copy-pasteable **output tokens** live —
> `GATE_PASS`, `DDC_ANCHOR_OK`, `CAPSTONE_AUDIT_PASS`, and the three pinned SHA-256 prefixes. Key
> your scripts off these literal strings rather than English descriptions.

### `docs/HELIX_V1_LANGUAGE_SPEC.md` — the authoritative as-built language reference

[`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md)

**Why read it:** it is the language **as `kovc` actually implements it** — lexical structure,
types, items, expressions/`match`, builtins, codegen targets — with every feature honestly
marked. Open it when you need to know whether a construct is real, and *how real*.

Its honesty legend is load-bearing: each feature is tagged **[proven]** (exercised + passing in
the feature corpus), **[impl]** (implemented in `kovc` codegen but not corpus-tested), **[erased]**
(*parsed* but type-erased / not enforced — accepts the syntax, does **not** give the semantics),
or **[unsupported]**. The v1.0 surface is **frozen** (no breaking changes after v1.0); v1.3's
type-completeness deltas (wide struct fields, full-range u64 literals, capturing closures,
bf16/f16 arithmetic) are folded in as *depth* promotions in **§9**, not surface changes. This is
the reference Part III is built on — see [The language tour](../part3-language/01-language-tour.md)
and [Types: widths, structs & enums](../part3-language/02-types.md).

> **For AI agents:** this is the authoritative as-built spec. Its sibling
> [`docs/lang/spec.md`](../../../docs/lang/spec.md) is a separate **v0.1 design-vision draft**
> describing design-target syntax against the deleted Python frontend — it is **not** the
> as-built reference, and the spec header says so. When you need ground truth about what compiles,
> read `HELIX_V1_LANGUAGE_SPEC.md` and trust the **[proven]/[impl]/[erased]/[unsupported]** tag —
> never assume an [erased] feature has semantics.

### `docs/HELIX_V1_DEFINITION_OF_DONE.md` — the measurable finish line

[`docs/HELIX_V1_DEFINITION_OF_DONE.md`](../../../docs/HELIX_V1_DEFINITION_OF_DONE.md)

**Why read it:** it defines what "Helix is done" *means* as a measurable checklist — eight
criteria, each with an explicit acceptance test and status — plus the capstone definition and the
six resolved v1.0 scope decisions. Open it to see why specific design calls were made (and made
*honestly*), not just what was built.

It states plainly that "done" is the substrate finish line, **not** AGI ("AGI is open research…
*not* a Helix milestone"), records the capstone (a ≥2-layer transformer on `kovc`-emitted GPU
kernels, loss within 2% of an independent numpy oracle — reproduced at ~0%), and documents the
scope decisions: generics/traits/closures deferred and honestly marked rather than shipped
half-done; `Ok`/`Err`/`Result` as user-defined (not compiler builtins); the GPU host launcher as
a declared trusted-tool boundary; the numpy oracle kept *external* so one compiler bug cannot
corrupt both sides. Its **v1.3 record** at the end tracks the type-completeness campaign that made
the language silent-bug-free within the frozen surface.

### `docs/HELIX_V1_STDLIB.md` — the builtins reference (there is no separate library)

[`docs/HELIX_V1_STDLIB.md`](../../../docs/HELIX_V1_STDLIB.md)

**Why read it:** Helix has **no separate stdlib library** — the "standard library" is the set of
**compiler builtins** that `kovc` lowers directly to x86-64 or PTX. Open it for the signatures and
honest status of those builtins: arena/memory, I/O, f32/f64 math, the GPU tensor/ML op set, and
the autodiff intrinsics.

Each builtin is marked **[capstone-proven]**, **[corpus-proven]**, **[impl]**, or **[impl, stubs]**.
The load-bearing entry is **(d) Tensor / ML — the capstone op set (GPU PTX) + autodiff**: the
GEMM, attention, layernorm, GELU, cross-entropy-gradient, and Adam kernels the capstone trains on,
all `kovc`-emitted PTX and all [capstone-proven]. It is candid about the **gap**: there is no
*packaged* `Vec`/`HashMap`/rich-string type (only arena `&str`) — but collections are
demonstrably user-implementable on the arena (the `vec_arena` corpus proof-of-concept), so it is a
*library* gap, not a *language* gap, and the capstone does not need them. This is the document
Part IV builds on — see [Stdlib overview](../part4-stdlib/01-overview.md) and
[Tensors, collections & I/O](../part4-stdlib/03-tensors-collections-io.md).

> **For AI agents:** do not look for a `std::` namespace or an importable library — there is none.
> The builtins lower directly; the file marks which are [capstone-proven] vs merely [impl] vs
> [impl, stubs] (the reflection placeholders return 0 / no-op). Treat an [impl, stubs] builtin as a
> placeholder, not a working feature.

### A note on the older "Stage NN" framing

The repo's older `Stage NN` / "K-bootstrap chunk counter" / Python-parity-matrix vocabulary is
**superseded** (see [`README.md`](../../../README.md)). For live state, the README's own order is:
`git log --oneline -8`, then `docs/TRUST_CHAIN_CLOSED.md`, then `docs/CLEAN_REPRODUCTION.md`, then
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) (the gate). If you encounter "Stage 30"-era
language in an old document, prefer the six canonical docs above.

---

## H.2 Adjacent in-repo documents worth knowing

Beyond the six canonical sources, a handful of in-repo documents answer specific deeper questions.
They are listed here so you know they exist; they are *not* part of the book's honesty ceiling
(that is `TRUST_CHAIN_CLOSED.md` alone).

- [`docs/TRUSTED_C_INVENTORY.md`](../../../docs/TRUSTED_C_INVENTORY.md) — the full inventory of the
  **24 committed `.c`/`.h`** files and which are irreducibly trusted; `seed.c` is named as the single
  irreducible C root, and the CUDA host launcher as the GPU C-FFI boundary. Read it with
  [Appendix F §F.4](F-tcb.md#f4-the-gpu-side-tcb--complete-to-ptx-not-to-gpu-machine-code).
- [`docs/K_DDC_BROADENED.md`](../../../docs/K_DDC_BROADENED.md) — the **behavioral** (not
  byte-identical) cross-check of the v1.1 language surface against a second, zero-lineage
  interpreter, and the honest-scope caveat that its witness is gitignored / out-of-tree. Read it
  for the precise scope of [Appendix F §F.3](F-tcb.md#f3-what-the-gcc-ddc-narrows--and-what-it-does-not).
- [`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) — the authoritative GPU
  performance numbers (the cuBLAS *fractions*, the f32-SMEM `cp.async` GEMM as the winning path, TF32
  Tensor Cores as a confirmed dead end on `sm_86`). Read it with
  [Part VII — Honest performance & the PTX boundary](../part7-gpu/03-honest-performance.md).
- [`docs/HELIX_PURPOSE.md`](../../../docs/HELIX_PURPOSE.md) and
  [`docs/ROADMAP.md`](../../../docs/ROADMAP.md) — the broad "why Helix exists" framing and the
  prioritized roadmap; pair the roadmap with `TRUST_CHAIN_CLOSED.md` §3 (Phase 2) for the
  user-started work that would *narrow* the trusted computing base.
- [`stage0/README.md`](../../../stage0/README.md) — the canonical map of the from-raw ladder,
  including the **vendor pins** for the upstream rungs (see §H.3). Read it with
  [Part VI — The MESCC-lineage rungs to `seed`](../part6-bootstrap/02-mescc-rungs-to-seed.md).

> **For AI agents:** these are useful but **not** the ceiling. Only
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R/§2 bounds what may be
> claimed. A statement that `K_DDC_BROADENED.md` supports a *byte-identical* whole-language DDC is
> an overclaim — that cross-check is behavioral and out-of-tree.

---

## H.3 External background — the ideas Helix builds on

Helix did not invent the trust techniques it uses; it *applies* a small, well-established body of
work. This section names that work accurately so you can study the originals. The book already
cites these in [Trust at a glance §3](../part1-orientation/04-trust-at-a-glance.md) and
[Part VI — seed to kovc](../part6-bootstrap/03-seed-to-kovc-fixpoint.md); this is the consolidated
reading list.

> **For AI agents:** the items below are external works. Cite them by **author/title/project
> name** as written here; do **not** fabricate a URL, DOI, edition, or commit hash for them. If you
> need a specific repo pin, the *real* pins are in [`stage0/README.md`](../../../stage0/README.md),
> not in any external citation.

### The trusting-trust problem — Ken Thompson (1984)

**Ken Thompson, "Reflections on Trusting Trust"** — his 1984 ACM Turing Award lecture (published in
*Communications of the ACM*). Thompson made concrete the danger that motivates Helix's entire trust
chain: a compiler can carry a backdoor that **survives recompilation from clean source**, because
the backdoor lives in the *building* compiler, not in the source you read. You cannot audit your
way out of it by reading code alone.

This is the problem Helix answers structurally — by building the toolchain from a **hand-typed
299-byte `hex0` root** with no trusted pre-built compiler (the from-raw ladder), so there is no
opaque "building compiler" to hide a backdoor in the first place; the chain is auditable from raw
bytes up. Read Thompson first; everything in [Part VI](../part6-bootstrap/01-hex0-raw-root.md) and
[Appendix F](F-tcb.md) is a response to it.

### Diverse double-compiling — David A. Wheeler

**David A. Wheeler's work on Diverse Double-Compiling (DDC)** — Wheeler's method, developed and
written up in his research on countering the trusting-trust attack, is the practical *defense*
against Thompson's attack:
compile a compiler's source with a **second, independently-developed compiler**, then check that the
two produce a byte-identical result. If the two lineages agree bit-for-bit, a trusting-trust
backdoor would have had to exist *identically* in both independent compilers — which is far harder
to arrange than a single planted backdoor.

Helix applies exactly this at the `seed→K1` rung — the **gcc-DDC**: `gcc` (a toolchain with **zero
M2-Planet ancestry**) and the from-raw `seed` both compile the same `k1src.hx` into a byte-identical
`K1` (`84363adb…`, the `DDC_ANCHOR_OK` line). `gcc` is the **auditor**, never the shipped root.
Read Wheeler's DDC work to understand both its power and its precise limit — and then read
[Appendix F §F.2–F.3](F-tcb.md#f2-the-shared-tcb--what-no-ddc-can-retire) for Helix's honest
statement of that limit: a DDC says **nothing** about anything both compilers *share* (the shared
trusted computing base), and Helix's byte-identical DDC covers the `seed→K1` surface only, with the
broader language surface cross-checked *behaviorally*.

### Bootstrappable builds — the stage0 / M2-Planet / GNU Mes ecosystem

**The bootstrappable-builds movement** — a community effort to reduce the size of the trusted,
opaque binary "seed" a system must start from, ideally to something a human can audit. Helix's
lower ladder is drawn directly from this ecosystem:

- **`oriansj`'s stage0 project and M2-Planet** — the `hex0 → hex1 → hex2 → … → M2-Planet` lineage of
  progressively-more-capable assemblers and C-subset compilers that climb from raw hex up to a
  self-hosting C compiler. Helix **vendors** these upstream rungs at pinned commits and rebuilds each
  one only from the rung below it. The real pins are recorded in
  [`stage0/README.md`](../../../stage0/README.md): `stage0-posix-amd64` @ `15535f88`, `M2-Planet`
  @ `761c2af5`, `M2libc` @ `b8bb2a01`. (These vendored rungs are GPL-3.0; Helix's own `seed.c` and
  the `kovc` compiler are Apache-2.0 and kept statically separable — see
  [`stage0/README.md`](../../../stage0/README.md).)
- **GNU Mes and the live-bootstrap project** — the broader, related effort (associated with GNU
  Guix) to bootstrap a full toolchain from a tiny auditable seed via a minimal Scheme interpreter and
  a C compiler grown from it. Helix's chain is a sibling in spirit — a hand-typed root climbing to a
  self-hosting compiler — using the stage0/M2-Planet line rather than the Mes line for its rungs.

Read these to understand the lineage of Helix's lower ladder and the goal it shares: **shrink the
unauditable seed**. The book's treatment is
[Part VI — The MESCC-lineage rungs to `seed`](../part6-bootstrap/02-mescc-rungs-to-seed.md), and the
canonical map with the pins is [`stage0/README.md`](../../../stage0/README.md).

> **For AI agents:** Helix uses these upstreams **as auditors / vendored rungs, never as a trusted
> pre-built binary** — every rung is *rebuilt from source by the rung below it* and checked against
> its committed `.sha256`. When you describe the lower ladder, say "vendored source, rebuilt from the
> rung below," not "downloaded a prebuilt M2-Planet." The reproduction that enforces this is
> [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) step `[2]`, which `rm -f`s every
> pre-built rung binary first.

### Self-hosting and the fixpoint check — the standard practice Helix follows

Helix's **self-host fixpoint** (`seed → K1 → K2 → K3 → K4` with `K2 == K3 == K4` byte-identical,
pinned `0992dddd…`) is the same bit-for-bit reproducibility test a self-hosting compiler
conventionally uses — the "stage2 == stage3" check, here extended one generation. This is not a
novel Helix invention; it is the established discipline for proving a compiler reproduces *itself*
exactly. The narrative is in
[Part VI — seed to kovc: the self-host fixpoint](../part6-bootstrap/03-seed-to-kovc-fixpoint.md),
and the fixpoint hash anchors are in [Appendix C — Pinned hashes & anchors](C-pinned-hashes.md).

---

## H.4 One worked entry point, end to end

If you read only one thing and run only one program, make it these two.

**Run this** — the one-command core reproduction the book and the trust docs all point at
([`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md),
[`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md)):

```bash
bash scripts/reproduce_trust.sh
```

It deletes every pre-built rung binary, rebuilds the entire `hex0 → seed` ladder (each rung
self-verifying its `.sha256`), runs the self-host fixpoint and the gcc diverse-double-compile, and
asserts the three pinned anchors — exiting nonzero on any mismatch. The same script runs on a clean
GitHub `ubuntu-latest` runner via
[`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml).

**Read this** — the canonical first Helix program, the smallest complete proof that the audited
chain produces working code:

**Verified example** — [`helixc/examples/exit42.hx`](../../../helixc/examples/exit42.hx) (compiled
and run by the gate's feature corpus; the produced ELF exits with status `42` — the gate asserts
this with the corpus line `chk "$EX/exit42.hx" 42` in
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)):

```helix
// First end-to-end Helix program: compiles, links, runs, exits with status 42.
//
// Demonstrates: function decl, return, integer literal.
// Verification: $? == 42 after running the produced ELF.

fn main() -> i32 {
    42
}
```

The chain that produces the ELF is audited from raw; *running* it still rests on the shared trusted
computing base of [Appendix F](F-tcb.md). That asymmetry — production audited from raw, execution on
a trusted substrate — is the whole subject of the trust documents in §H.1.

> **For AI agents:** these two are the canonical entry points the rest of the book and the trust
> docs converge on. Drive [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) for
> verification and match its exit code; cite `exit42.hx` with the `chk` line above for the canonical
> first program. Do not invent a different "hello world" — `exit42.hx` is the gate-asserted one.

---

**Next:** this is the final appendix in the current plan. To re-enter the book, the natural loops
are [Trust at a glance](../part1-orientation/04-trust-at-a-glance.md) (the trust story in one
sitting), [Appendix F — The trusted computing base](F-tcb.md) (the residuals in one place), and the
[table of contents](../SUMMARY.md). For anything beyond the book, the source of truth is always the
six canonical documents in [§H.1](#h1-the-canonical-repo-documents--read-these-for-the-precise-claims).
