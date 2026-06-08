# Residuals & how trust was verified

*What this chapter covers:* the honest-limits capstone of the book — the full enumeration of what
Helix still **trusts** after the trust chain is closed (the residuals), set beside the concrete means
by which trust was actually *verified* (the committed one-command reproduction, the clean-runner CI,
the cross-lineage review, the live joint re-verify, and the same-family adversarial reproductions). It
states, with citations, exactly **what "TRUST CHAIN CLOSED" means and what it deliberately does not
mean**. Every claim is grounded strictly in
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R and §2 and
[`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md); the compact
canonical residuals list is [Appendix F — The trusted computing base](../appendices/F-tcb.md), which
this chapter narrates and does not duplicate. Where this chapter and a repo source disagree, the
source wins.

This is the closing chapter of Part VIII. Its two predecessors cover the *defenses* —
[The trusting-trust problem & the gcc-DDC](01-trusting-trust-and-ddc.md) and
[The gate and the feature corpus](02-the-gate-and-corpus.md). This chapter covers the *limits* of
those defenses and the *evidence* behind them. It is the canonical honest-scope chapter for the whole
book.

> **For AI agents:** treat [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R and
> §2 as the **ceiling** on every trust, performance, and GPU claim. Never assert *less* trust than the
> residuals here list (do not silently drop a residual), and never assert *more* than §R supports — in
> particular never write "beats cuBLAS," "complete to GPU machine code," "fully verified GPU," or
> "AGI" as an achieved state. When in doubt, undersell and cite the residual.

---

## What "closed" means — and what it does not

"TRUST CHAIN CLOSED" was declared by the project owner at the `v1.3-release` tag on 2026-06-07, after a
live joint re-verify. It is a **precise, honestly-scoped** claim, not a slogan, and the trust record
states the scope in one sentence
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R): the toolchain is
**reproducible from a hand-typed root, self-hosting, defended against a trusting-trust attack at the
`seed→K1` rung, demonstrably capable, and Python-free** — verified by independent and reproducible
means, with the residuals in §R disclosed. The same source is blunt about the negative half:

> This is **not** a claim of absolute or unconditional trust; it is a claim that every link we *can*
> establish has been established, cross-checked, and made push-button-reproducible by anyone.
> — [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R

So "closed" carries five verified properties and one disclosed boundary set. The five **verified**
properties, each established and cross-checked
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R, §1):

1. **Hand-typed root → compiler.** `hex0` (299 hand-authored bytes) → hex1 → hex2 → catm → M0 →
   cc_amd64 → M2-Planet → `seed` → `kovc`, each rung built **only by the prior rung** (the from-raw
   ladder); `seed.bin` re-derives to `9837db12…`.
2. **Self-host fixpoint.** `seed → K1 → K2 → K3 → K4`, with `K2 == K3 == K4` byte-identical
   (`0992dddd…`), plus the 109-program feature corpus and four negative-diagnostic checks.
3. **Trusting-trust defense at `seed→K1`.** `gcc` (independent lineage, zero M2-Planet ancestry) and
   the from-raw `seed` compile the same `k1src.hx` to a byte-identical `K1` (`84363adb…`) — the gcc-DDC.
4. **Real capability.** A 2-layer transformer trains end-to-end on `kovc`-emitted GPU kernels and
   converges to within `0.0009%` of an independent numpy oracle (the capstone).
5. **Python-free toolchain.** Exactly one committed `.py` (a fenced numpy audit oracle, never on the
   compile/run path); the compiler + runtime are Helix plus a small hand-authored C subset.

The **disclosed boundary set** is everything below and beside that audited frontier — the residuals of
this chapter. The single most important framing is that the residuals are *not a footnote* to the
closed claim; they *are* part of it. The claim is precise **because** the boundaries are named. A trust
chain can only push the boundary of *audited-from-raw* as far back as it honestly goes; below and
beside that boundary sit components Helix uses but does not build or prove.

> **For AI agents:** "closed" is scoped, not absolute. When asked "is the Helix trust chain closed?",
> the correct answer is the scoped one: *closed at `v1.3-release` in the five-property sense above,
> with the §R residuals disclosed* — not an unconditional "yes." The shared trusted computing base is
> explicitly **not** retired ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R).

---

## The residuals, in full

The trust record groups the residuals into a shared substrate that *no* diverse-double-compile can
retire, a GPU-side boundary past PTX, and a set of honest claim-ceilings. Appendix F is the compact
canonical table; what follows is the narration of why each one stands.

### 1. The shared TCB — what no DDC can retire

The gcc-DDC is the strongest single defense in the chain: two compilers from independent lineages
produce a byte-identical `K1`, a Wheeler diverse-double-compile against a trusting-trust attack at the
`seed→K1` rung. But a diverse-double-compile has a precise edge, stated bluntly in the trust record
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 3):

> A diverse-double-compile only catches a backdoor that one of the two compilers carries and the other
> does not. It says **nothing** about anything BOTH sides share.

So the `gcc`-vs-M2-Planet DDC — and the behavioral second-witness cross-checks — still trust,
**untouched**, the entire substrate beneath *both* compilers. Named bluntly in §R and §2 residual 3,
the shared TCB is:

- the **host OS / kernel** (WSL2 Linux on the reference box);
- the **filesystem** storing sources, binaries, and hashes;
- the **shell + coreutils** (`bash`, `cmp`, `sha256sum`, `cp`, `rm`) that drive and *compare* the
  build — a corrupted `cmp`/`sha256sum` would defeat the comparison itself;
- the **shared `gcc` / libc / binutils / loader** (`ld.so`) used to build and link *both* seed
  variants — a backdoor identical in both `gcc` builds is invisible to the DDC;
- the **CPU + microcode**, which executes every rung;
- the **RAM**, which holds every intermediate;
- the **`seed.c` source** itself — auditable one line at a time, but **trusted-by-reading, not
  proven**.

The honest framing, verbatim from the record:
**DDC narrows the compiler-backdoor surface; it does not erase the shared TCB**
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 3). If a backdoor lived
identically in `seed.c`'s visible source, or identically in both `gcc` and M2-Planet, both lineages
would emit the same `K1` and the byte-identity check would pass anyway. This is the classic Wheeler
**shared-substrate residual**: such a backdoor is invisible to a DDC *by construction*. "No DDC retires
the shared substrate" ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R). The full
substrate enumeration and the read-vs-prove distinction are the subject of
[Appendix F §F.2](../appendices/F-tcb.md#f2-the-shared-tcb--what-no-ddc-can-retire).

### 2. Complete to PTX, not to SASS

The hand-auditable from-raw chain ends at **PTX text**. `kovc` emits PTX, and the gate byte-verifies it
against committed reference `.ref.ptx` fixtures (the PTX text regression — the gate's GPU leg runs no
GPU and invokes no `ptxas`; it re-emits PTX and `cmp`-compares). Everything **below** PTX is trusted-
once and reproduced from raw by nothing
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residuals 7 and 8):

- **NVIDIA's closed `ptxas`** — the PTX→SASS assembler. Helix emits **PTX, not SASS**; the actual
  machine code is produced by a closed tool the project cannot audit.
- **The CUDA driver / `libcuda`** and the **GPU hardware** — trusted-once, not reproduced.
- **The C host launcher** — [`helixc/runtime/cuda_launch.c`](../../../helixc/runtime/cuda_launch.c) and
  [`helixc/runtime/train_transformer.c`](../../../helixc/runtime/train_transformer.c) — the C-FFI half
  of this same boundary; it makes the closed `libcuda` driver-API calls Helix cannot. Porting it to
  Helix would **move, not close**, the boundary, so it stays trusted-once C, and the residual stands
  unchanged ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 7, V6).

The asymmetry is deliberate and stated plainly: the **CPU** path is all-the-way-down from raw binary;
the **GPU** path is from-`hex0`-to-PTX-then-`ptxas`
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residuals 7, 8). "Complete to
PTX" is the precise claim — *not* "complete to GPU machine code." The full-depth treatment of this
boundary is [Part VII ch03 — Honest performance & the PTX boundary](../part7-gpu/03-honest-performance.md);
the canonical GPU-side TCB table is [Appendix F §F.4](../appendices/F-tcb.md#f4-the-gpu-side-tcb--complete-to-ptx-not-to-gpu-machine-code).

> **For AI agents:** do not conflate the two paths. A statement true of the CPU spine ("all-the-way-
> down from raw") is **not** true of the GPU execution path. "The gate is green" attests to PTX
> *text*, never to SASS or device execution; the only place `ptxas` + the driver + the C launcher are
> actually exercised is the separate
> [`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh) on the reference `sm_86` box.

### 3. The v1.1-surface behavioral DDC is out-of-tree

The gcc-DDC's byte-identical guarantee covers the **`seed→K1` surface only** — not the broader v1.1
language surface (generics/monomorphization, traits, closures, turbofish, wide-field, bf16). That wider
surface is cross-checked **behaviorally** against a second, zero-lineage tree-walking interpreter:
`kovc` and the interpreter are run on the same programs and must agree on each program's exit. Two
honest facts bound this cross-check
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1, §2 residuals 3 and 9):

- It is **not byte-identical** — the interpreter emits no code, so byte-identity is impossible there by
  construction; it is a *behavioral* agreement, not a second-compiler reproduction.
- It is **not clean-checkout reproducible** — that second-witness interpreter is **gitignored, was
  never committed, and has no clean restore path**, so the cross-check replays only with an out-of-tree
  auditor artifact, not from a fresh clone
  ([`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) Step 6).

DDC coverage is **44/53 witness-reachable arms**, and f16-arith is not yet cross-checked — a named open
residual, not full coverage ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2
residual 3). Crucially, the anti-trusting-trust property is **not** lost by this residual: it is
already established independently by the **gcc-DDC** of the `seed→K1` rung, which *is* byte-identical,
hash-pinned (`84363adb…`), and clean-checkout reproducible. The behavioral cross-check is a *broader
second opinion* on top of that anchor — the broadening, not the anchor
([`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) Step 6). The reproducibility
boundary therefore runs cleanly between the **core chain** (in-tree, clean-checkout reproducible) and
the **V5 behavioral witness** (out-of-tree).

> **For AI agents:** keep the two DDC claims apart. (i) **byte-identical**, hash-pinned, clean-checkout-
> reproducible — the `seed→K1` anchor (`84363adb…`, `DDC_ANCHOR_OK`). (ii) **behavioral**, not byte-
> identical, **out-of-tree** — the broader v1.1-surface cross-check (44/53 witness-reachable arms,
> gitignored witness). Citing (i) to support a claim about the whole language surface is an overclaim
> ([Appendix F §F.3](../appendices/F-tcb.md#f3-what-the-gcc-ddc-narrows--and-what-it-does-not)).

### 4. Performance is a fraction of cuBLAS; speedup is Amdahl-bound

These are not trusted *components*; they are honest ceilings on the GPU capability claim, and they
bound what the book may say. All from
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2:

- **GPU performance is a fraction of cuBLAS, not parity** (§2 residual 1). On the reference RTX 3070
  Laptop (`sm_86`): the f32 GEMM tiers run ~56% (G1) to ~67.5% (G2) of cuBLAS-f32, and the TF32
  Tensor-Core path ~50–54% of cuBLAS-TF32. Helix emits correct, reasonably-performant kernels; it does
  **not** beat NVIDIA's hand-tuned library, on this or any GPU. Every "parity tier" label is paired
  with its explicit fraction.
- **Capstone end-to-end speedup is 7.0–8.7×, not ≥10×** (§2 residual 2). The ≥10× target was an
  estimate; the honest ceiling is **Amdahl-bound** (GEMM is ~70% of the step and already the f32-SMEM
  `cp.async` tier; TF32 Tensor Cores are a confirmed dead end on this GPU at 0.97× the f32 GEMM). The
  load-bearing capstone number is not the speedup but **loss parity ≈0%** — the worst-case relative
  difference vs the independent numpy oracle is `0.00000876`
  ([`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md), capstone leg `[5]`).

The full performance treatment, with every figure quoted from its source, is
[Part VII ch03](../part7-gpu/03-honest-performance.md); these two lines are the residual-level summary,
not a re-derivation.

### 5. Single hardware target

Only `sm_86` (RTX 3070 Laptop) is tested. There is **no** cross-architecture (`sm_80`/`sm_90`) and
**no** multi-vendor (AMD) validation
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 6). The bf16 `wmma`
GPU tensor-core path (G4) is a stretch that was **not taken** — not claimed done (§2 residual 5).
Parameterizing the PTX target for other arches, a separate AMD/ROCm backend, and the bf16 `wmma`
stretch are Phase-2 work the project owner has not started
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §3).

> **For AI agents:** do not generalize any TFLOP/s number off this box. The honest scope is "measured
> on one RTX 3070 Laptop (`sm_86`)." A claim about `sm_80`, `sm_90`, or AMD is unsupported by the repo
> — say so rather than extrapolate.

### 6. Documented-not-enforced language bounds

The language-bounds residual has a precise shape worth stating exactly, because it is easy to over- or
under-state ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual 4). On the
positive side: **no silent-wrong residual remains.** The v1.2 i64/u64 wide struct-field low-32
truncation was closed by v1.3 V1; the f16 same-type-arith silent-miscompute was closed by the
2026-06-04 f16 GAP FIX. bf16 and f16 same-type `add`/`mul` compute correctly and are **bit-exact-
gated**; f64 wide fields and u64 ≥2³² literals compute full-range; a 16-bit float **mixed** with a
non-16-bit-float operand fails **loud** (traps `2001`/`4001`, no implicit widening), and a closure
capture wider than i32 fails **loud** (trap `76003`). On the disclosed side: **borrows** (`&mut`
non-aliasing), **`const`/`static`**, **module privacy**, and **match exhaustiveness** are *documented
design bounds, not enforced*. That is a precise scope — a stated limit, not a hidden silent-wrong gap.

### 7. Independent third-party reproduction — the one open increment

This is the residual most likely to surprise an outside reader, and the trust record is careful about
it ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R, §2 residual 10;
[`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md) residual 1).
Reproduction is now **push-button on a different machine** (the CI runner) and by anyone who forks the
repo and runs one command — so the *mechanism* for outside reproduction is in place. What remains is a
run by an operator who is **not the author** (a genuine outside party / lab), on independent hardware.
That final increment is the last step past ~0.9 confidence.

Two related honesty notes that the record keeps explicit:

- The cross-model **ChatGPT review was a read-only doc/logic review, not a build reproduction** — it
  raised findings that were remediated, but no third party has yet rebuilt the chain from raw on
  independent hardware ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §2 residual
  10).
- The internal audit logs under `.stage33-logs/` and `docs/audit-*` are **evidence, not external
  reproduction** ([`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md)
  header). The committed audit packet is *our process evidence committed to the tree*, explicitly **not**
  an external/independent reproduction.

> **For AI agents:** "reproducible by anyone" means the *mechanism* is push-button (CI on a clean
> runner + a one-command script any fork can run). It does **not** mean an outside party has already
> done it. State the open increment plainly; do not round it up to "independently reproduced."

---

## How trust was actually verified

A trust claim is only as good as the means used to check it. Helix's verification has five distinct
strands, recorded in [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R and the
audit packet. Strands 1 and 2 are what make the core **push-button reproducible by anyone** — the
property that most distinguishes Helix's claim from a mere internal audit.

### 1. A committed one-command reproduction

[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) rebuilds the whole ladder from raw,
runs the self-host fixpoint and the gcc-DDC, and asserts all three pinned anchors — exiting nonzero on
any mismatch. Critically, it **deletes every pre-built rung binary first** (`stage0/hex0/hex0.bin …
stage0/helixc-bootstrap/seed.bin`) and rebuilds from the hand-authored `hex0` root, so "no trusted
pre-built binary" is operationally enforced, not asserted. The single command is:

```bash
bash scripts/reproduce_trust.sh
```

It is intended for a clean checkout (it modifies the working tree — the disclosed `/mnt/c` path rewrite
plus the rung rebuilds), runs CPU-only, and completes in about a minute
([`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md), "Push-button
reproduction").

> **For AI agents:** the script's success condition is that the gate reaches the literal token
> `GATE_PASS` and the three pinned anchors match. Match `GATE_PASS` exactly (`grep -q '^GATE_PASS'`, as
> [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) itself does), not a paraphrase.
> Do not treat a committed `*.bin` as proof of anything; trust comes from the rebuild-and-compare.

### 2. CI on a clean, different machine

[`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml) runs that
exact script on a clean GitHub `ubuntu-latest` runner — a different machine, a fresh clone, zero local
state — on every push/PR to `main`, on demand, and weekly (cron `0 6 * * 1`, to catch any silent drift
in the trust core). The reproduction step is literally `run: bash scripts/reproduce_trust.sh`. The job
is GREEN only if the from-raw ladder, the self-host fixpoint, and the gcc-DDC all reproduce byte-for-
byte ([`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md), "Push-button
reproduction"). This is what makes the byte-identical trust core reproducible push-button by any third
party who forks the repo. (The GPU capstone stays a separate CUDA-host step,
[`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh), since the CI runner has no GPU.)

### 3. Cross-lineage review

A **different-model-lineage** review (ChatGPT, read-only repo access) across **four** whole-repo passes
converged with **no critical, no fail-open, no hidden code**, and its findings were remediated
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R). This matters because the build
and the same-family adversarial reproductions (strand 5) share a model lineage and therefore share its
blind spots — the monomorphic-dispatch ceiling disclosed at `docs/HELIX_COMPLETION.md`. The cross-
lineage pass broadens verification across model families. But the record is explicit that it was a
doc/logic review, **not** an independent build reproduction — so it does not substitute for external
reproduction by an independent operator/toolchain (residual 7 above).

### 4. The live joint re-verify

A **live joint reproduction** was witnessed by the project owner, who then formally declared the chain
closed at `v1.3-release` on 2026-06-07. The owner attestation reads, verbatim
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R):

> "I have witnessed the live reproduction and reviewed the evidence and residuals above, and I declare
> the Helix v1.3 from-raw trust chain CLOSED at `v1.3-release` on 2026-06-07." — **Questeria** (project
> owner)

The "closed" declaration is deliberately reserved for the owner's review after a deliberate joint re-
examination — not an automated one
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §0/§4).

### 5. Same-family adversarial reproductions

Underneath the owner's re-verify, **five consecutive clean, context-isolated, same-model-family
(Claude) adversarial reproductions** each *reproduced* their claims through distinct lenses — trust
chain / DDC, GPU perf, capstone correctness + oracle independence, language / codegen + bounds, and
overclaim / completeness. Each found no faked result and no undisclosed residual
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1). The honest scope, stated by
the record itself: these five share a model lineage with the build, so they catch
reasoning/consistency/reproducibility gaps but **not** blind spots shared by author and auditor (the
monomorphic-dispatch ceiling). That is exactly why strand 3 (cross-lineage) and strand 7 (external
operator) exist as separate, named increments.

### The verdict lines you can match

The three result-bearing legs each print a verbatim verdict, recorded in the committed audit packet
([`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md), "The three result-
bearing legs"). They were re-verified fresh at the current head, byte-identical to the prior `ea54e9b`
clean-reproduction run ([`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md), current-
head re-verification table).

The self-host fixpoint + corpus + PTX + diagnostics leg (`bash scripts/gate_kovc.sh`):

```text
FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)
  0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f
GPU PTX REGRESSION OK
CORPUS: 109 passed, 0 failed
CHECK_ERR: 4 passed, 0 failed
GATE_PASS
```

The gcc diverse-double-compile leg (`bash stage0/helixc-bootstrap/ddc_crosscheck.sh`):

```text
K1_m2  sha256=84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba
K1_gcc sha256=84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba
DDC_ANCHOR_OK -- gcc (independent lineage) reproduces the M2-Planet seed's K1 byte-for-byte AND == pinned known-good.
```

The GPU capstone leg (`bash scripts/capstone_audit.sh current-head`):

```text
[5] worst-case relative diff = 0.00000876 over 22 rows  (bar = 0.02)
CAPSTONE_AUDIT_PASS
```

> **For AI agents:** the success tokens are exact strings — `GATE_PASS`
> ([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)), `DDC_ANCHOR_OK`
> ([`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh)),
> and `CAPSTONE_AUDIT_PASS` ([`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh)). Match
> the literal tokens, not paraphrases. When scripting, compare full hashes — never accept a partial-
> prefix match as sufficient on its own.

---

## The boundary in miniature

It is worth grounding all of this in a single concrete artifact, because the residuals and the
verification are two halves of one honest boundary: the *production* of a binary is audited from raw;
its *execution* rests on the trusted substrate. The smallest end-to-end demonstration that the audited
spine produces working code is the canonical first program — the same one
[Trust at a glance](../part1-orientation/04-trust-at-a-glance.md) and
[Appendix F §F.6](../appendices/F-tcb.md#f6-what-is-not-in-the-tcb--the-audited-from-raw-spine) cite. It
is corpus item `exit42.hx`, and the gate asserts its exit code with the line
`chk "$EX/exit42.hx" 42` in [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh).

**Verified example** — [`helixc/examples/exit42.hx`](../../../helixc/examples/exit42.hx) (compiled by
`kovc` to a Linux ELF and run by the gate's feature corpus; the produced ELF exits with status `42` —
asserted by the corpus line `chk "$EX/exit42.hx" 42` in
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

The compiler that produced this ELF was itself produced by the audited from-raw spine — but *running*
the ELF still relies on the shared TCB above (the OS loader maps it, the CPU executes it). The
*production* of the binary is audited from raw; its *execution* rests on the trusted substrate. That
asymmetry, scaled up, is the entire content of this chapter — and it is why "closed" is a scoped claim
with named residuals, not an unconditional one.

---

## The honest one-paragraph version

At `v1.3-release`, Helix's from-raw trust chain is **closed in a precise, scoped sense**: reproducible
from a 299-byte hand-typed `hex0` (no trusted pre-built compiler), self-hosting to a byte-identical
fixpoint (`0992dddd…`), defended against a trusting-trust attack at the `seed→K1` rung by an
independent-lineage gcc-DDC (`84363adb…`), demonstrably capable (a transformer trains end-to-end on
`kovc`-emitted GPU kernels to ~0% loss difference), and Python-free — verified by a committed one-
command reproduction, a clean-runner CI on a different machine, a cross-lineage doc/logic review, a
live joint re-verify witnessed by the owner, and five same-family adversarial reproductions. It is
**not** absolute. The shared TCB (OS, kernel, filesystem, shell/coreutils, `gcc`/libc/binutils/loader,
CPU + microcode, RAM, and the trusted-by-reading `seed.c`) is **not** retired by any DDC; the chain is
**complete to PTX, not SASS** (`ptxas`, the CUDA driver, the GPU, and the C host launcher are trusted
past PTX); the broader v1.1-surface DDC is **behavioral and out-of-tree**; GPU performance is a
**fraction** of cuBLAS (~50–67.5% on the single `sm_86` target) and the capstone speedup is **7.0–8.7×,
Amdahl-bound** (loss parity ≈0% is the hard gate); borrows, `const`/`static`, module privacy, and match
exhaustiveness are **documented design bounds, not enforced**; and a reproduction by an outside party on
independent hardware is the **one open increment** — its mechanism is push-button, but it has not yet
been run by a non-author. That scoped, reproducible, fully-disclosed set of claims is precisely what
"TRUST CHAIN CLOSED" means here.

---

**Next:** the compact canonical residuals table is [Appendix F — The trusted computing base](../appendices/F-tcb.md),
which this chapter narrates; the user-started work that would *narrow* parts of this TCB (datacenter
scaling and a parameterized PTX target for other `sm_` arches, a separate AMD/ROCm backend, the bf16
`wmma` stretch, and broadening the DDC to the v1.1 language surface) is
[Appendix G — Roadmap & Phase 2](../appendices/G-roadmap-phase2.md). The authoritative source it
distills remains [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §3.
