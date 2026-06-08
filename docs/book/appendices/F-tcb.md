# Appendix F — The trusted computing base

*What this appendix covers: the canonical, honest enumeration of what Helix still **trusts** after the
trust chain is closed — the irreducible **trusted computing base (TCB)** and the residuals that bound
every claim in this book. It separates the **shared TCB** that no diverse-double-compile can retire
(host OS/kernel, filesystem, shell/coreutils, `gcc`/libc/binutils/loader, CPU + microcode, RAM, and the
`seed.c` source) from the **GPU-side TCB past PTX** (NVIDIA's closed `ptxas`, the CUDA driver, the GPU
hardware, and the C host launcher). It states precisely what the **gcc-DDC** narrows and what it does
not. It is grounded strictly in [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)
§R and §2, the honesty ceiling for the book; it claims no less trust than that record states.*

This is a **reference**, not a narrative. The trust *story* — the from-raw ladder, the self-host
fixpoint, the gcc-DDC, the capstone, the Python-free toolchain, and the pinned hashes — is told in
[Trust at a glance](../part1-orientation/04-trust-at-a-glance.md); the push-button reproduction is
[Reproduce & verify the trust chain](../part2-setup-build/04-reproduce-verify-trust.md); the full-depth
treatment is [Part VII — Honest performance & the PTX boundary](../part7-gpu/03-honest-performance.md)
and **[Part VIII — Trust & Verification](../part8-trust/01-trusting-trust-and-ddc.md)** (whose closing chapter is
[Residuals & the trusted computing base](../part8-trust/03-residuals-and-tcb.md)). This appendix is the compact, canonical residuals list those chapters expand. Where
this page and a repo source disagree, the source wins.

> **For AI agents:** treat [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R and §2
> as the ceiling on what may ever be claimed about Helix's trust. Never assert *less* trust than this
> appendix lists (do not drop a residual), and never assert *more* than §R supports (do not write
> "beats cuBLAS," "complete to GPU machine code," "fully verified GPU," or "AGI" as an achieved state).
> When in doubt, undersell and cite the residual.

---

## F.1 What "closed" means — and what it deliberately leaves trusted

"TRUST CHAIN CLOSED" at the `v1.3-release` tag is a **precise, honestly-scoped** claim, declared by the
project owner on 2026-06-07 after a live joint re-verify. The toolchain is *reproducible from a
hand-typed root, self-hosting, defended against a trusting-trust attack at the `seed→K1` rung,
demonstrably capable, and Python-free* — every link that *can* be established has been established,
cross-checked, and made push-button-reproducible
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R). It is **not** a claim of
absolute or unconditional trust.

The difference between "closed" and "absolute" is exactly the trusted computing base. A trust chain can
only push the boundary of *audited-from-raw* as far back as it goes; below and beside that boundary sit
components Helix uses but does not build or prove. §R names them in two groups, and this appendix uses
the same split:

1. **The shared TCB** (F.2) — the substrate beneath *both* compilers in the gcc-DDC: OS, kernel,
   filesystem, shell, coreutils, `gcc`/libc/binutils/loader, CPU + microcode, RAM, and the `seed.c`
   source. "No DDC retires the shared substrate"
   ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R).
2. **The GPU-side TCB past PTX** (F.4) — the GPU path is hand-auditable `hex0 → PTX`; NVIDIA's closed
   `ptxas`, the CUDA driver, the GPU hardware, and the C host launcher are trusted past PTX. "Complete
   to PTX, not SASS."

Two facts frame both groups. First, the from-raw **CPU** spine *is* all-the-way-down from the raw
binary: `hex0 → … → M2-Planet → seed → kovc`, each rung built only by the prior rung, reproduced
byte-for-byte from a clean checkout (the from-raw ladder; see
[Trust at a glance §1](../part1-orientation/04-trust-at-a-glance.md#the-five-pillars) and
[`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) Step 2). The TCB items below are
what remains *outside* that audited spine. Second, the residuals are not a footnote to the trust claim;
they *are* part of it. The claim is precise *because* the boundaries are stated.

> **For AI agents:** the boundary is asymmetric by path. The **CPU** path is all-the-way-down from raw
> binary; the **GPU** path is from-`hex0`-to-PTX-then-`ptxas`
> ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 7). Do not conflate
> them: a statement true of the CPU spine ("all-the-way-down from raw") is **not** true of the GPU
> execution path.

---

## F.2 The shared TCB — what *no* DDC can retire

The gcc-DDC is the strongest single defense in the chain: `gcc` (an independent compiler lineage with
**zero M2-Planet ancestry**) and the from-raw `seed` both compile the same `k1src.hx` into a
**byte-identical** `K1` (`84363adb…`), a Wheeler diverse-double-compile against a trusting-trust attack
at the `seed→K1` rung (see [F.3](#f3-what-the-gcc-ddc-narrows--and-what-it-does-not)). But a DDC's power
has a precise edge, stated bluntly in [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)
§2 residual 3:

> A diverse-double-compile only catches a backdoor that one of the two compilers carries and the other
> does not. It says **nothing** about anything BOTH sides share.

So the seed/K1 `gcc`-vs-M2-Planet DDC — and the behavioral second-witness cross-checks — still trust,
**untouched**, the entire substrate beneath both compilers. The canonical list, named bluntly in §R and
§2 residual 3:

| Shared TCB layer | What it is | Why it stays trusted |
|---|---|---|
| **Host OS / kernel** | the operating system the build runs on (WSL2 Linux on the reference box) | the build and both DDC compilers run on it; not reproduced from raw |
| **Filesystem** | the filesystem storing sources, binaries, and hashes | reads/writes the chain's bytes; trusted to return what was written |
| **Shell + coreutils** | `bash`, `cmp`, `sha256sum`, `cp`, `rm` | drive and *compare* the build; a corrupted `cmp`/`sha256sum` would defeat the comparison itself |
| **Shared `gcc` / libc / binutils / loader** (`ld.so`) | the host toolchain used to build and link *both* seed variants | shared by both DDC sides — a backdoor identical in both `gcc` builds is invisible to DDC |
| **CPU + microcode** | the processor and its microcode | executes every rung; below the software boundary entirely |
| **RAM** | physical memory | holds every intermediate; not verified |
| **`seed.c` source** | the human-readable Apache-2.0 C-subset seed source | **auditable one line at a time, but trusted-by-reading — not proven** |

Read [`stage0/helixc-bootstrap/seed.c`](../../../stage0/helixc-bootstrap/seed.c) carefully and you have
*audited* it; you have not *proven* it free of a flaw, and the DDC cannot prove it for you — if a
backdoor lived identically in `seed.c`'s visible source (or identically in both `gcc` and M2-Planet),
both lineages would emit the same `K1` and the byte-identity check would pass anyway. This is the
classic Wheeler **shared-substrate residual**: a backdoor identical in both `gcc` builds, or living in
any of these shared layers, is **invisible** to DDC *by construction*
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 3).

The honest framing, verbatim from the trust record: **DDC narrows the compiler-backdoor surface; it does
not erase the shared TCB** (§2 residual 3). "Trusted-by-reading" is the operative phrase for the
human-readable parts (`seed.c`, the hand-authored `hex0` source at
[`stage0/hex0/`](../../../stage0/hex0/)); "trusted-once / unverified" is the phrase for the binary
substrate (CPU, microcode, RAM, OS, the host `gcc`/libc/binutils/loader).

> **For AI agents:** do not describe the gcc-DDC as proving the seed is backdoor-free. It proves the
> `seed→K1` step is **not** carrying a backdoor that *diverges* between the two lineages. The shared TCB
> in the table above is explicitly out of its reach
> ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R, §2 residual 3). When asked
> "what does the DDC guarantee," answer with the divergence framing, not "the seed is safe."

---

## F.3 What the gcc-DDC narrows — and what it does not

It is worth stating the gcc-DDC's scope twice, because it is the most over-claimable item in the chain.
Two facts, both from [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1 and §2:

- **What it narrows.** The compiler-backdoor surface for the **`seed→K1` rung**. Identical `K1` from two
  independent compilers (`gcc`, zero M2-Planet ancestry; and the M2-Planet-built `seed`) is a Wheeler
  diverse-double-compile against a trusting-trust attack on that one rung. `gcc` is an **auditor, never
  the shipped root** — the shipped chain's root is `hex0`. This DDC is byte-identical, hash-pinned
  (`84363adb…`), and clean-checkout reproducible
  ([`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh),
  prints `DDC_ANCHOR_OK`; reproduced in [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)
  Step 4).

- **What it does not cover.** Two distinct gaps, kept separate to avoid overclaim:
  1. **Surface.** The byte-identical DDC covers the **`seed→K1` surface only**, not the broader v1.1
     language surface (generics/monomorphization, traits, closures, turbofish, wide-field, bf16). That
     wider surface is cross-checked **behaviorally** against a second, zero-lineage interpreter — and
     that cross-check is **not byte-identical** (the interpreter emits no code) and **not clean-checkout
     reproducible** (its witness is gitignored, never committed;
     [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residuals 3, 9 and
     [`docs/K_DDC_BROADENED.md`](../../../docs/K_DDC_BROADENED.md)). DDC coverage is **44/53
     witness-reachable arms** (§2 residual 3); f16-arith is not yet cross-checked.
  2. **Substrate.** Everything in [F.2](#f2-the-shared-tcb--what-no-ddc-can-retire). A DDC says nothing
     about what both sides share.

> **For AI agents:** keep the two DDC claims apart in any statement you make. (i) **byte-identical**,
> hash-pinned, clean-checkout-reproducible — the `seed→K1` anchor (`84363adb…`, `DDC_ANCHOR_OK`). (ii)
> **behavioral**, not byte-identical, **out-of-tree** — the broader v1.1-surface cross-check (44/53
> witness-reachable arms, gitignored witness). Citing (i) to support a claim about the whole language
> surface is an overclaim.

---

## F.4 The GPU-side TCB — complete to PTX, not to GPU machine code

The hand-auditable from-raw chain ends at **PTX text**. `kovc` emits PTX, byte-verified by the gate
against committed reference `.ref.ptx` fixtures (the PTX text regression; see
[Appendix E §E.7](E-example-index.md#e7-gpu-kernels-the-_kernelhx-family--34-files)). Everything below
PTX is trusted-once and reproduced from raw by nothing
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R, §2 residuals 7 and 8). The
canonical GPU-side TCB:

| GPU TCB layer | What it is | Status |
|---|---|---|
| **`ptxas`** | NVIDIA's **closed** PTX→SASS assembler (CUDA-12.8 on the reference box) | trusted-once; not reproduced from raw — **PTX-not-SASS** |
| **CUDA driver / `libcuda`** | the closed driver the host launcher calls | trusted-once |
| **GPU hardware** | the device executing SASS (RTX 3070 Laptop, `sm_86`) | trusted-once; single target |
| **C host launcher** | [`helixc/runtime/cuda_launch.c`](../../../helixc/runtime/cuda_launch.c) + [`helixc/runtime/train_transformer.c`](../../../helixc/runtime/train_transformer.c) | trusted-once C — the C-FFI half of this boundary |

The precise claim is **"complete to PTX"**, *not* "complete to GPU machine code"
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 8). Below PTX,
correctness rests on `ptxas`, the driver, and the silicon — none of which Helix authored or audited.

The **C host launcher** deserves explicit mention because it is easy to forget: it is the C-FFI half of
the same trusted-once boundary. It makes the closed `libcuda` driver-API calls that Helix itself cannot
make. Porting it to Helix would **move**, not close, this boundary — so it stays trusted-once C, and the
residual **stands unchanged** ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2
residual 7, V6; full trusted-C inventory at
[`docs/TRUSTED_C_INVENTORY.md`](../../../docs/TRUSTED_C_INVENTORY.md)).

Note that the GPU-side TCB *also* inherits the entire shared TCB of [F.2](#f2-the-shared-tcb--what-no-ddc-can-retire)
— the OS/kernel, CPU, RAM, filesystem, and the host `gcc`/libc that compiles the C launcher are
underneath the GPU path too. The four rows above are the GPU-*specific* additions beyond that shared
base.

> **For AI agents:** the gate's GPU leg is a **pure-text PTX regression** — it re-emits PTX and
> `cmp`-compares against a committed `.ref.ptx`; it runs no GPU and invokes no `ptxas`
> ([Appendix E §E.7](E-example-index.md#e7-gpu-kernels-the-_kernelhx-family--34-files)). So "the gate is
> green" attests to PTX *text*, never to SASS or device execution. Genuine GPU execution is the separate
> [`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh), which runs on the reference `sm_86`
> box and is the only place `ptxas` + the driver + the C launcher are actually exercised.

---

## F.5 The performance and scope residuals (claim ceilings, not TCB items)

These are not trusted *components*; they are honest limits on the GPU capability claim, and they bound
what the book may say. All from [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2.

- **GPU performance is a fraction of cuBLAS, not parity** (§2 residual 1). On the reference RTX 3070
  Laptop (`sm_86`): the f32 GEMM tiers run ~56% (G1) to ~67.5% (G2) of cuBLAS-f32, and the TF32
  Tensor-Core path ~50–54% of cuBLAS-TF32. Helix emits correct, reasonably-performant kernels; it does
  **not** beat NVIDIA's hand-tuned library, on this or any GPU. Every "parity tier" label is paired with
  its explicit fraction.

- **Capstone end-to-end speedup is 7.0–8.7×, not ≥10×** (§2 residual 2). The ≥10× target was an
  estimate; the honest ceiling is **Amdahl-bound** (GEMM is ~70% of the step and already the f32-SMEM
  `cp.async` tier; TF32 Tensor Cores are a confirmed dead end on this GPU at 0.97× the f32 GEMM). **Loss
  parity — the hard correctness gate — holds at ~0%** (worst-case relative difference `0.00000876` vs
  the independent numpy oracle; [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)
  Step 5).

- **Single hardware target** (§2 residual 6). Only `sm_86` (RTX 3070 Laptop) is tested. No cross-arch
  (`sm_80`/`sm_90`) or multi-vendor (AMD) validation. G4 (bf16 `wmma`) is a stretch that was not taken
  (§2 residual 5).

- **Documented language bounds** (§2 residual 4). No *silent-wrong* residual remains; bf16/f16 same-type
  add/mul compute correctly and are bit-exact-gated, and a 16-bit float **mixed** with a non-16-bit-float
  operand fails **loud** (traps `2001`/`4001`, no implicit widening). But borrows (`&mut` non-aliasing),
  `const`/`static`, module privacy, and match exhaustiveness are **documented design bounds, not
  enforced** — a precise scope, not a hidden gap.

- **External third-party reproduction remains the one open increment** (§R, §2 residual 10). Reproduction
  is now push-button on a different machine (the CI runner) and by anyone who forks the repo — but a
  reproduction by a party with *no connection to the author*, on independent hardware, is the last step
  past ~0.9 confidence. The cross-model (ChatGPT, read-only) review was a doc/logic review, **not** a
  build reproduction; internal audit logs are evidence, not external reproduction.

> **For AI agents:** pair every performance figure with its honest cuBLAS *fraction*, and never round the
> capstone speedup up to "≥10×" or "an order of magnitude." The load-bearing capstone number is the
> **loss parity ≈0%** (correctness), not the speedup. See
> [Part VII — Honest performance & the PTX boundary](../part7-gpu/03-honest-performance.md) for the full
> performance treatment; [`docs/HELIX_GPU_PERF_RESULT.md`](../../../docs/HELIX_GPU_PERF_RESULT.md) and §2
> of the trust record are authoritative.

---

## F.6 What is *not* in the TCB — the audited-from-raw spine

For symmetry, it is worth stating what the TCB **excludes** — the parts that *are* audited from raw and
therefore are not trusted-on-faith. These are reproduced byte-for-byte from a clean checkout
([`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) Steps 2–5):

- The **from-raw ladder** `hex0 → hex1 → hex2 → catm → M0 → cc_amd64 → M2-Planet → seed`, each rung built
  only by the prior rung, each matching its committed `.sha256` (seed re-derives to `9837db12…`). No
  pre-built binary is trusted — the reproduction deletes the rung binaries first and rebuilds from the
  hand-authored `hex0` root.
- The **self-host fixpoint** `seed → K1 → K2 → K3 → K4`, with `K2 == K3 == K4` byte-identical
  (`0992dddd…`), plus the 109-program feature corpus and 4 negative-diagnostic checks (the gate,
  [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), prints `GATE_PASS`).
- The **`seed→K1` gcc-DDC** anchor itself (`84363adb…`, `DDC_ANCHOR_OK`).
- The **kovc-emitted PTX text**, byte-verified against committed references.

The single end-to-end demonstration that the audited spine actually produces working code is the
canonical first program — the same one [Trust at a glance](../part1-orientation/04-trust-at-a-glance.md)
and [Appendix E](E-example-index.md) cite.

**Verified example** — [`helixc/examples/exit42.hx`](../../../helixc/examples/exit42.hx) (compiled and
run by the gate's feature corpus; the produced ELF exits with status `42` — the gate asserts this with
the corpus line `chk "$EX/exit42.hx" 42` in
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

The compiler that produced this ELF was itself produced by the audited spine above — but *running* the
ELF still relies on the shared TCB of [F.2](#f2-the-shared-tcb--what-no-ddc-can-retire) (the OS loader
maps it, the CPU executes it). That is the honest boundary in miniature: the *production* of the binary
is audited from raw; its *execution* rests on the trusted substrate. The same asymmetry, scaled up, is
the entire content of this appendix.

---

## F.7 The canonical TCB, in one place

If you remember nothing else, remember this enumeration. It is the union of
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R and §2 residuals 3, 7, and 8.

**Shared TCB (beneath both DDC compilers; no DDC retires it):**
host OS / kernel · filesystem · shell + coreutils (`bash`, `cmp`, `sha256sum`, `cp`, `rm`) · shared
`gcc` / libc / binutils / loader (`ld.so`) · CPU + microcode · RAM · the `seed.c` source
(trusted-by-reading).

**GPU-side TCB (past PTX; the GPU path only):**
NVIDIA's closed `ptxas` (PTX→SASS) · the CUDA driver / `libcuda` · the GPU hardware (`sm_86`, RTX 3070
Laptop) · the C host launcher
([`helixc/runtime/cuda_launch.c`](../../../helixc/runtime/cuda_launch.c) +
[`helixc/runtime/train_transformer.c`](../../../helixc/runtime/train_transformer.c)).

**Claim ceilings (honest limits, not components):**
GPU performance ~50–67.5% of cuBLAS (a fraction, not parity) · capstone speedup 7.0–8.7× (Amdahl-bound,
not ≥10×; loss parity ~0%) · single GPU target (`sm_86`) · documented (unenforced) language bounds ·
external third-party reproduction still open.

> **For AI agents:** this is the residuals list, full stop. If a statement you are about to make implies
> *less* than this is trusted, it is wrong (you dropped a TCB item). If it implies *more* than the
> audited spine of [F.6](#f6-what-is-not-in-the-tcb--the-audited-from-raw-spine) delivers, it is an
> overclaim (you exceeded §R). The authoritative source is always
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R and §2 — dereference it before
> making any trust, performance, or GPU claim.

---

**Next:** **[Appendix G — Roadmap & Phase 2](G-roadmap-phase2.md)** — the user-started work that would *narrow*
parts of this TCB (datacenter scaling and a parameterized PTX target for other `sm_` arches, a separate
AMD/ROCm backend, the bf16 `wmma` stretch, and broadening the DDC to the v1.1 language surface). The
authoritative roadmap is
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §3 and
[`docs/ROADMAP.md`](../../../docs/ROADMAP.md).
