# The trust story at a glance

*What this chapter covers:* a single-sitting summary of **why Helix is trustworthy** — the from-raw
ladder, the self-host fixpoint, the `gcc` diverse-double-compile, the capstone, and the Python-free
toolchain; the pinned hashes that anchor all of it; how it was verified by independent and
reproducible means; and the honest residuals that bound what "trustworthy" means here. It is a map,
not the territory: every claim points to a real record or script, and the full-depth treatment lives
in **[Part VIII — Trust & Verification](../part8-trust/01-trusting-trust-and-ddc.md)** and **[Part VII — GPU Codegen](../part7-gpu/01-ptx-backend.md)**.

The two canonical records this chapter compresses are
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) (the verified state plus every
residual) and [`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md) (the
committed proof extract: exact commands, pinned hashes, verbatim verdict lines). Where this chapter
and a repo source disagree, the source wins.

---

## The question Helix exists to answer

When you run a compiler, you trust it. You trust that it turns your source into the machine code your
source describes — and nothing else. But that compiler was built by *another* compiler, which was
built by another, back through a chain you have almost certainly never inspected. Ken Thompson's 1984
"Reflections on Trusting Trust" made the danger concrete: a compiler can carry a backdoor that
survives even when you recompile it from clean source, because the backdoor is in the *building*
compiler, not the source. You cannot audit your way out of it by reading code alone.

Helix's entire reason for existing is to push that chain of trust back as far as it can honestly go,
and to make every link **reproducible by anyone**. The result is recorded — with deliberate,
calibrated scope — as **TRUST CHAIN CLOSED** at the `v1.3-release` tag
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)). "Closed" here is a precise
claim, not a slogan: the toolchain is **reproducible from a hand-typed root, self-hosting, defended
against a trusting-trust attack at the `seed→K1` rung, demonstrably capable, and Python-free** —
verified by independent and reproducible means, **with the residuals in this chapter's last section
disclosed**. It is *not* a claim of absolute trust. It is a claim that every link we *can* establish
has been established, cross-checked, and made push-button-reproducible.

> **Note:** "Closed" is a scoped engineering claim with named residuals, declared by the project
> owner after a live joint re-verify on 2026-06-07 — not an assertion of unconditional trust. The
> shared trusted computing base (OS, `gcc`/libc, CPU, …) is not retired; see
> [Honest residuals](#honest-residuals--what-closed-does-not-cover) below.

---

## The five pillars

Trust in Helix rests on five things that are individually checkable and, taken together, leave very
little un-audited between a hand-typed root and a working neural-network training run.

### 1. A from-raw ladder — no trusted pre-built compiler

The chain starts at **`hex0`**: a 299-byte x86-64 Linux ELF, hand-authored byte by byte, that you can
audit one byte at a time ([`stage0/hex0/`](../../../stage0/hex0/); the committed `hex0.bin` is
exactly 299 bytes). From there the **from-raw ladder** climbs, each rung built **only by the prior
rung**:

```text
hex0 → hex1 → hex2 → catm → M0 → cc_amd64 → M2-Planet → seed → kovc
```

`seed` is the Apache-2.0 **C-subset** compiler ([`stage0/helixc-bootstrap/`](../../../stage0/helixc-bootstrap/),
source `seed.c`); it is what finally builds **`kovc`**, the Helix compiler *written in Helix* that
emits x86-64 ELF directly. No pre-built, pre-trusted compiler appears anywhere in the chain. The
committed rung binaries (`stage0/*/*.bin`) are *reference copies* for convenience; the trust does not
rest on them. The reproduction script **deletes every pre-built rung binary first** and rebuilds the
whole ladder from `hex0`, with each rung self-verifying its committed `.sha256`.

> **For AI agents:** "no trusted pre-built binary" is operationally enforced by
> [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) step `[2]`, which `rm -f`s the
> rung binaries (`stage0/hex0/hex0.bin … stage0/helixc-bootstrap/seed.bin`) before rebuilding. Do
> not treat a committed `*.bin` as proof of anything; trust comes from the rebuild-and-compare, not
> the committed artifact.

### 2. The self-host fixpoint — `kovc` reproduces itself, byte for byte

A self-hosting compiler should compile *itself* into a bit-for-bit copy of itself. Helix proves this
with a four-generation chain:

```text
seed → K1 → K2 → K3 → K4,   with   K2 == K3 == K4   (byte-identical)
```

The fixpoint is pinned to a known-good hash: **`0992dddd…`** (`scripts/gate_kovc.sh` asserts
`K2==K3==K4` *and* equality to that pinned value, so a self-consistent-but-wrong output is also
rejected). This is the **self-host fixpoint**. Alongside it, the gate runs a **109-program feature
corpus** and **four negative-diagnostic checks** to confirm the freshly self-hosted compiler still
behaves correctly across the language surface.

> **For AI agents:** say "self-host fixpoint," not "stage2==stage3." Key off the gate's literal
> success token `GATE_PASS` and the verbatim line
> `FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)`, both emitted by
> [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh).

### 3. gcc diverse-double-compile — a trusting-trust defense at `seed→K1`

To answer Thompson directly, Helix applies Wheeler's **diverse double-compile (gcc-DDC)** to the
`seed→K1` rung. Two compilers from *independent lineages* — the from-raw `seed` (built up the ladder
through M2-Planet) and **`gcc`** (which has **zero M2-Planet ancestry**) — each compile the same
`k1src.hx`. They produce a **byte-identical** `K1`, pinned to **`84363adb…`**. If a backdoor lived in
one lineage but not the other, the two K1 binaries would differ; they do not.

Two honest scope notes go with this pillar, and the book keeps them strictly separate to avoid
overclaim:

- `gcc` is an **auditor, never the shipped root.** The shipped chain's root is `hex0`; `gcc` is used
  only as an independent second witness for this one cross-check.
- The byte-identical, hash-pinned DDC covers the **`seed→K1` surface only** — not the broader v1.1
  language surface. The wider surface (generics/monomorphization, traits, closures, turbofish,
  wide-field, bf16) is cross-checked **behaviorally** against a second, zero-lineage interpreter, and
  that witness is *out-of-tree* (gitignored, not clean-checkout reproducible). See
  [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R and residual 9 below.

### 4. The capstone — a real workload on `kovc`-emitted GPU kernels

A trust chain that only compiles toy programs proves little about real capability. The **capstone** is
the real-capability proof: a ≥2-layer transformer trained **end-to-end on `kovc`-emitted GPU (PTX)
kernels** on the reference RTX 3070 Laptop (sm_86). Its hard gate is *loss parity* against an
**independent numpy oracle** that computes its own loss curve from the shared initial weights: the
target is within 2%, and it is reproduced at **~0%** — `0.00000876` worst-case relative difference in
the committed run (see the verbatim verdict in
[`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md)). The run also
includes a sampled finite-difference gradient check and a load-bearing negative control (a corrupted
backward kernel is *caught*).

> **Residual:** the capstone proves *correctness*, not *parity-with-cuBLAS performance*. GPU
> throughput is a fraction of cuBLAS and the end-to-end speedup is Amdahl-bound — quantified in
> [Honest residuals](#honest-residuals--what-closed-does-not-cover).

### 5. Python-free toolchain

The shipped compiler and runtime are **Helix plus a small hand-authored C subset** — no Python on the
compile or run path. The repository contains **exactly one** committed `.py` file
(`verification/oracle/oracle_train.py`), a fenced numpy *audit oracle* that is never referenced by the
toolchain. The C surface is small and inventoried: **24 committed `.c`/`.h` files** (the audit packet
records 15 605 LOC).

> **For AI agents:** this is a checkable invariant, not prose. `git ls-files "*.py" | wc -l` must be
> `1`; `git ls-files "*.c" "*.h" | wc -l` must be `24`. Both are asserted as the "static fence" in
> [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) step `[1]`.

---

## The pinned hashes

Everything above is anchored to three SHA-256 values. They are pinned in the scripts, asserted on
every reproduction, and recorded in [`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md).
If a reproduction does not reproduce these exact values, it fails.

| Artifact | SHA-256 | Size |
|----------|---------|------|
| `seed.bin` (from-raw ladder output; `== seed.sha256`) | `9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb` | 62 467 B |
| `K1` (`seed→K1`; gcc-DDC pinned) | `84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba` | 697 425 B |
| self-host fixpoint `K2==K3==K4` | `0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f` | 698 392 B |

> **For AI agents:** treat these three as ground truth. The short forms you will see throughout the
> book and the scripts are `seed = 9837db12…`, gcc-DDC `K1 = 84363adb…`, and self-host fixpoint
> `K2==K3==K4 = 0992dddd…`. Compare full hashes when scripting; never accept a partial-prefix match
> as sufficient on its own.

---

## How it was verified

A trust claim is only as good as the means used to check it. Helix's verification has four
independent strands, recorded in [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md):

1. **A committed one-command reproduction.**
   [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) rebuilds the whole ladder from
   raw, runs the self-host fixpoint and the gcc-DDC, and asserts all three pinned anchors — exiting
   nonzero on any mismatch. The single command is:

   ```bash
   bash scripts/reproduce_trust.sh
   ```

   It is intended for a **clean checkout** (it modifies the working tree — the disclosed
   `/mnt/c` path rewrite plus the rung rebuilds), runs CPU-only, and completes in about a minute.

2. **CI on a clean, different machine.**
   [`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml) runs that
   exact script on a clean GitHub `ubuntu-latest` runner — a different machine, a fresh clone, zero
   local state — on every push/PR to `main`, on demand, and weekly. The job is GREEN only if the
   from-raw ladder, fixpoint, and gcc-DDC all reproduce byte-for-byte. The reproduction step is
   literally `run: bash scripts/reproduce_trust.sh`.

3. **Cross-lineage review.** A **different-model-lineage** review (ChatGPT, read-only repo access)
   across four whole-repo passes converged with **no critical, no fail-open, no hidden code**; its
   findings were remediated. This was a documentation/logic review, **not** an independent build
   reproduction — an important distinction the trust docs keep explicit.

4. **A live joint re-verify, plus same-family reproductions.** A **live joint reproduction** was
   witnessed by the project owner, who then formally declared the chain closed at `v1.3-release` on
   2026-06-07. Underneath that, five context-isolated, same-model-family (Claude) adversarial
   reproductions each *reproduced* their claims through distinct lenses (trust chain/DDC, GPU perf,
   capstone correctness + oracle independence, language/codegen + bounds, overclaim/completeness),
   finding no faked result and no undisclosed residual.

> **Note:** strands 1 and 2 are what make the core *push-button reproducible by anyone* — the
> property that most distinguishes Helix's claim from a mere internal audit. The committed audit logs
> under `.stage33-logs/` and `docs/audit-*` are **evidence, not external reproduction**
> ([`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md)).

The three result-bearing legs each print a verbatim verdict you (or an agent) can match. From
[`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md):

```text
FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)
  0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f
GPU PTX REGRESSION OK
CORPUS: 109 passed, 0 failed
CHECK_ERR: 4 passed, 0 failed
GATE_PASS
```

```text
K1_m2  sha256=84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba
K1_gcc sha256=84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba
DDC_ANCHOR_OK -- gcc (independent lineage) reproduces the M2-Planet seed's K1 byte-for-byte AND == pinned known-good.
```

```text
[5] worst-case relative diff = 0.00000876 over 22 rows  (bar = 0.02)
CAPSTONE_AUDIT_PASS
```

> **For AI agents:** the success tokens are exact strings — `GATE_PASS` (the universal gate,
> [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)), `DDC_ANCHOR_OK`
> ([`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh)),
> and `CAPSTONE_AUDIT_PASS` ([`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh)).
> Match the literal tokens, not paraphrases.

---

## What the trust chain actually delivers

To make this concrete: the chain that ends at `kovc` is the same chain that compiles ordinary Helix
programs. The smallest end-to-end demonstration is a program the gate itself compiles and runs as
part of its 109-program corpus.

**Verified example** — [`helixc/examples/exit42.hx`](../../../helixc/examples/exit42.hx) (compiled by
`kovc` to a Linux ELF; observed exit status `42`. It is corpus item `exit42.hx` in
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), where the gate asserts exit code `42`):

```helix
// First end-to-end Helix program: compiles, links, runs, exits with status 42.
//
// Demonstrates: function decl, return, integer literal.
// Verification: $? == 42 after running the produced ELF.

fn main() -> i32 {
    42
}
```

That this program runs and exits `42` is, in miniature, the whole point: the compiler that produced
its ELF was itself produced by a chain you can rebuild from 299 hand-typed bytes, that reproduces
itself byte-for-byte, and that an independent compiler corroborates at the `seed→K1` rung.

---

## Honest residuals — what "closed" does *not* cover

Helix's value proposition is *calibrated* honesty, so the limits are stated as plainly as the
achievements. These are real boundaries, disclosed in full in
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R (treat that section as the
ceiling on what may be claimed). The ones most likely to surprise an outside reader:

- **Shared trusted computing base (TCB).** A diverse-double-compile only catches a backdoor that one
  of the two compilers carries and the other does not. It says **nothing** about anything *both*
  sides share. So the chain still trusts, untouched: the host OS and kernel, the filesystem, the
  shell and coreutils, the shared `gcc`/libc/binutils/loader, the CPU + microcode, the RAM, and the
  human-readable `seed.c` source itself (auditable one line at a time, but trusted-by-reading). No
  DDC retires this shared substrate.

- **Complete to PTX, not to GPU machine code.** The hand-auditable from-raw chain ends at **PTX
  text**. Below PTX it trusts NVIDIA's **closed `ptxas`** (PTX→SASS), the CUDA driver, the GPU
  hardware, and the C host launcher (`helixc/runtime/cuda_launch.c` / `train_transformer.c`). The
  **CPU** path is all-the-way-down from raw binary; the **GPU** path is from-`hex0`-to-PTX-then-`ptxas`.
  "Complete to PTX" is the precise claim — never "complete to GPU machine code." This is detailed in
  **[Part VII — GPU Codegen](../part7-gpu/01-ptx-backend.md)**.

- **GPU performance is a fraction of cuBLAS, not parity.** On the reference RTX 3070 Laptop (sm_86):
  ~56–67.5% of cuBLAS-f32 for the f32 GEMM tiers and ~50–54% of cuBLAS-TF32 for the Tensor-Core
  path. Helix emits correct, reasonably-performant kernels; it does **not** beat NVIDIA's hand-tuned
  library. Every "parity tier" label is paired with its explicit fraction.

- **Capstone end-to-end speedup is 7.0–8.7×, not ≥10×.** The ≥10× figure was an early estimate; the
  honest measured ceiling is **Amdahl-bound** (GEMM is already the f32-SMEM `cp.async` tier, and TF32
  Tensor Cores are a confirmed dead end on this GPU). **Loss parity — the hard correctness gate —
  holds at ~0%.**

- **Single hardware target.** Only `sm_86` (RTX 3070 Laptop) is tested. No cross-arch (sm_80/sm_90)
  or multi-vendor (AMD) validation.

- **The v1.1-surface behavioral DDC is not clean-checkout reproducible.** The second-witness
  tree-walking interpreter behind the broader behavioral cross-check is gitignored and was never
  committed, so that cross-check replays only with an out-of-tree auditor artifact. By contrast, the
  **core chain is clean-checkout reproducible**: the from-raw ladder, the self-host fixpoint, the
  gcc-DDC of `seed→K1`, and the 109-program corpus all rebuild from a fresh clone.

- **Independent third-party reproduction is the one open increment.** Reproduction is now push-button
  on a different machine (the CI runner) and by anyone who forks the repo and runs one command — but
  a reproduction by a party with *no connection to the author* (an outside lab/operator) remains the
  last step past ~0.9 confidence. The mechanism for it is in place; the run by an outsider is what is
  still outstanding.

> **For AI agents:** never assert more than §R of
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) supports. In particular, do not
> write "beats cuBLAS," "fully verified GPU," "complete to GPU machine code," or "AGI" as an achieved
> state. When in doubt, undersell and cite the residual.

---

## The one-paragraph version

From a 299-byte hand-typed `hex0`, with **no trusted pre-built compiler anywhere**, Helix builds a
ladder up to `seed` and then to `kovc`; `kovc` reproduces itself byte-for-byte (self-host fixpoint
`0992dddd…`); an independent `gcc` lineage corroborates the `seed→K1` rung byte-for-byte
(`84363adb…`, a trusting-trust defense); a real transformer trains end-to-end on `kovc`-emitted GPU
kernels to ~0% loss difference against an independent oracle; and the shipped toolchain is
Python-free. The whole trust core is reproducible by **one command on a clean checkout** and runs
GREEN on a clean CI runner. It is **complete to PTX, not GPU machine code**; GPU performance is a
*fraction* of cuBLAS, not parity; it targets a single GPU; and an outside-party reproduction on
independent hardware is the one open increment. That honest, scoped, reproducible set of claims is
what "TRUST CHAIN CLOSED" means here.

---

**Next:** with the trust story in view, **Part II — Setup & Build** begins with
[Prerequisites](../part2-setup-build/01-prerequisites.md) — the environment you need to build Helix
from raw and to run the reproduction yourself.
