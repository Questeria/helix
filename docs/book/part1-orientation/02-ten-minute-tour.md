# The ten-minute tour

*What this chapter covers:* the fastest honest path from a fresh clone to *trust* — clone the
repo, run the one committed reproduction script to rebuild the compiler from a hand-typed root
and verify the self-host trust core, read the real `REPRODUCE_TRUST: PASS` verdict (with the
pinned hashes), then compile and run a tiny Helix program through the from-raw compiler. This is
a skim, not a deep dive: every step below is grounded in a real script, and Part II takes each
one apart.

The whole point of Helix is that you don't have to take this chapter's word for any of it. The
trust core is reproducible push-button, on a clean checkout, by anyone — so the tour *is* the
proof.

---

## Before you start

You need a Linux environment (native Linux, or **WSL2** on Windows), because the build produces
and runs x86-64 Linux ELF binaries. The reproduction script lists its exact tool requirements at
the top of [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh):

```text
# Tools required: bash 4+, coreutils (sha256sum/stat), xxd, file, objdump (binutils), gcc, grep, sed.
```

That is the whole prerequisite list for the trust core. Note what is **not** there: no Python,
no pre-built compiler, no package install of a toolchain. `gcc` appears only as an independent
*auditor* (the diverse-double-compile in step `[4]`), never as the shipped root. The GPU capstone
needs a CUDA host and is verified separately (see [The honest scope](#the-honest-scope-of-this-tour)
below); you do **not** need a GPU for this tour.

> **For AI agents:** the authoritative prerequisite list is the `Tools required:` comment line in
> [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh). Do not add or assume tools
> beyond it. The script is CPU-only by design and runs on a clean GitHub `ubuntu-latest` runner
> (see [`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml));
> a missing tool is a setup error, never a reason to skip a check.

---

## Step 1 — Clone

```bash
git clone https://github.com/<owner>/Kovostov-Native.git
cd Kovostov-Native
```

The release this tour targets is the tag **`v1.3-release`** — the point at which the trust chain
was declared closed (see [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)). Check it
out so your hashes match the ones printed below:

```bash
git checkout v1.3-release
```

> **Note:** the committed tree carries the compiler *source* and the hand-authored hex root, but
> **not** the pre-built `seed` binary — `stage0/helixc-bootstrap/seed.bin` is gitignored. That is
> deliberate: a fresh clone has nothing pre-built to trust, so reproduction must rebuild the
> compiler from raw. This is verified in [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md)
> (Step 1, "FENCE + INVENTORY").

---

## Step 2 — Reproduce the trust core (one command)

This is the heart of the tour. A single committed script rebuilds the entire toolchain from a
hand-typed root and verifies it against pinned hashes:

```bash
bash scripts/reproduce_trust.sh
```

> **Warning:** this script **modifies the working tree** — it deletes the pre-built rung binaries
> so it can rebuild each from the one below it, and it applies a disclosed mechanical path rewrite.
> Run it on a **clean checkout** (a CI runner or a throwaway clone), *not* on a tree you want to
> keep pristine. The script's own header says so:
>
> ```text
> # NOTE: intended for a CLEAN CHECKOUT (CI runner or a throwaway clone). It MODIFIES the working tree
> ```

### What it actually does

The script runs four checks, in order. You don't need to memorise them to follow the tour, but it
helps to know what each `[n]` line in the output means. These are the script's own descriptions
(from the header of [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh)):

```text
#   [1] static fence            : exactly 1 committed .py, 24 committed .c/.h
#   [2] from-raw ladder         : DELETE every pre-built rung binary, then rebuild hex0->...->seed
#                                 using ONLY the prior rung (hex0 from hand-authored hex via xxd);
#                                 each rung self-verifies its committed .sha256; seed == pinned.
#   [3] self-host fixpoint       : scripts/gate_kovc.sh -> K2==K3==K4 == pinned, corpus 109/0, check_err 4/0
#   [4] gcc diverse-double-compile: gcc (zero M2-Planet ancestry) and the from-raw seed both produce
#                                 a BYTE-IDENTICAL, pinned K1 (Wheeler trusting-trust defense).
```

In plain terms:

- **`[1]` static fence** — proves the toolchain is Python-free: exactly **one** committed `.py`
  (a numpy audit oracle that never touches the compile path) and **24** committed `.c`/`.h` files
  (the small hand-authored C subset). See the [terminology](#a-map-of-the-terms-you-just-saw)
  note below for `seed` and `kovc`.
- **`[2]` the from-raw ladder** — deletes every pre-built rung, then rebuilds the chain
  `hex0 → hex1 → hex2 → catm → M0 → cc_amd64 → M2-Planet → seed`, where each rung is built **only**
  by the rung below it. `hex0` is 299 hand-authored hex bytes; nothing above it is trusted
  pre-built. The rebuilt `seed` must hash to the pinned value.
- **`[3]` the self-host fixpoint** — runs [the gate](../../../scripts/gate_kovc.sh), which mints
  the Helix compiler `kovc` and feeds it to itself: `seed → K1 → K2 → K3 → K4`, asserting
  `K2 == K3 == K4` byte-identical and equal to the pinned hash, plus a 109-program feature corpus
  and 4 negative-diagnostic checks.
- **`[4]` the gcc diverse-double-compile** — builds the seed a *second*, independent way with
  `gcc` (which has zero M2-Planet ancestry) and checks that both routes produce a **byte-identical**
  `K1`. This is a Wheeler trusting-trust defense over the `seed→K1` surface.

The build is real work: the gate dominates the wall time (the seed compiling the multi-megabyte
self-host source, then the byte-by-byte self-compile rounds). On a clean runner the whole script
takes on the order of tens of minutes; on a slow filesystem it can be longer.

> **For AI agents:** treat the four `[n]` checks as the spec. The script accumulates failures
> (`set -uo pipefail`, no global `set -e`) and prints a `*** FAIL:` line for each mismatch, then a
> single final verdict. Key off the final verdict line and the pinned hashes, not on intermediate
> timing or `say` lines.

---

## Step 3 — Read the verdict

When every check matches, the script prints exactly this (from the tail of
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh)):

```text
============================================================
REPRODUCE_TRUST: PASS
  from-raw ladder + self-host fixpoint + gcc-DDC all reproduce the pinned anchors from a clean checkout.
  (GPU capstone is verified separately by scripts/capstone_audit.sh on a CUDA host.)
```

`REPRODUCE_TRUST: PASS` is the whole verdict. The script `exit 0`s **only** if every check
matched the pinned anchors; any mismatch yields `REPRODUCE_TRUST: FAIL` and a nonzero exit.

### The pinned hashes

The three anchors the run reproduces are pinned literally in the script
([`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh), the "Pinned release anchors"
block) — and the script prints them in its banner before it starts:

```text
 anchors   : seed=9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb
             K1  =84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba
             fix =0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f
```

| Anchor | Hash (SHA-256) | What it pins |
|--------|----------------|--------------|
| `seed` | `9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb` | the from-raw-rebuilt `seed` compiler binary |
| `K1`   | `84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba` | the first `kovc` build — **byte-identical** from the from-raw seed *and* from gcc |
| `fix`  | `0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f` | the self-host fixpoint `K2 == K3 == K4` |

If your run reproduces those three values and prints `REPRODUCE_TRUST: PASS`, you have rebuilt the
Helix compiler from a hand-typed root and confirmed it self-hosts byte-for-byte — on your own
machine, with nothing pre-built trusted.

> **For AI agents:** verify success by **both** signals — `grep -q '^REPRODUCE_TRUST: PASS'` on
> the output **and** the three pinned hashes above appearing. A pinned hash is the byte-exact
> contract; reproduce it or report `FAIL`. The fixpoint anchor `0992dddd…` is also pinned inside
> the gate ([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), `EXPECT_FIX`), so a fixpoint
> drift fails both that script's `GATE_PASS` and this one's verdict.

### Want to watch it run on someone else's machine?

You don't even need to run it locally to see it pass on independent hardware. The same one command
is wired into CI at [`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml),
whose only build step is:

```bash
bash scripts/reproduce_trust.sh
```

It runs on a clean GitHub `ubuntu-latest` runner — a different machine, a fresh clone, zero local
state — on every push to `main`, every PR, and weekly. Anyone can fork the repo and watch it go
green.

---

## Step 4 — Compile and run a tiny Helix program

You just rebuilt a working compiler. Here is the smallest possible Helix program, which is also
the canonical first program in this book:

**Verified example** — [`helixc/examples/exit42.hx`](../../../helixc/examples/exit42.hx)
(compiles to a Linux ELF; the produced binary exits with status `42`):

```helix
// First end-to-end Helix program: compiles, links, runs, exits with status 42.
//
// Demonstrates: function decl, return, integer literal.
// Verification: $? == 42 after running the produced ELF.

fn main() -> i32 {
    42
}
```

A Helix program is C-family in shape: `fn main() -> i32` is the entry point, and the value of its
last expression is the function's result. Here that value, `42`, becomes the process exit status.

### The fast way: the reproduction already ran it

Here is the honest shortcut. `exit42.hx` is the **first program in the gate's 109-program feature
corpus**, so step `[3]` of the reproduction you ran in Step 2 *already compiled and ran it through
the freshly self-hosted compiler* — and checked that it exits `42`. In the gate output
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), the `=== [4] FEATURE CORPUS ===`
section), it appears as:

```text
  PASS exit42.hx (42)
```

That single line is a complete end-to-end proof for the program: the self-hosted `kovc` compiled
the `.hx` source to an ELF, the ELF ran, and its exit status was the expected `42`. The same run
checks 108 more programs the same way and reports the tally:

```text
  CORPUS: 109 passed, 0 failed ...
```

So if you only want to *see* a Helix program compile and run on the from-raw compiler, you already
have — it is one of the lines scrolling past in Step 2.

### The direct way: invoke the compiler yourself

To drive the compiler by hand, you work in the bootstrap directory where the rebuilt `seed` lives,
`stage0/helixc-bootstrap/`. The `seed` compiler takes a source file and an output path as two
positional arguments — exactly as the gate invokes it
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), step `[2]`):

```bash
timeout 1200 ./seed.bin k1src.hx /tmp/K1.bin
```

That is the real compile invocation form: `./seed.bin <source.hx> <output.bin>`. The output is a
Linux ELF you then mark executable and run, reading its exit status the usual way:

```bash
chmod +x /tmp/out.bin
./out.bin
echo $?
```

> **Residual:** there is **no committed one-liner wrapper** that compiles an arbitrary `.hx` file
> end-to-end — the repo's reproducible compile path is the gate's corpus runner, which mints the
> self-hosted `kovc` and feeds programs to it through fixed input paths. Compiling your *own* file
> by hand means mirroring what the gate does, not calling a polished `helixc build foo.hx` CLI.
> The mechanics — minting the compiler, the kovc binary's input/output conventions, and writing
> your own programs — are walked through in Part II. Don't infer a wrapper flag that isn't in a
> committed script.

> **Note:** an older, **Python-hosted** frontend (`helixc`, e.g.
> `python -m helixc.backend.x86_64 hello.hx hello.bin`) exists in the history and in
> [`QUICKSTART.md`](../../../QUICKSTART.md), but it is **not** on the shipped, from-raw compile
> path and is retained for reference only. This tour — and this book's trust claims — are about the
> Python-free `kovc` toolchain you rebuilt in Step 2.

### Two more gate-verified programs to glance at

If you want slightly meatier examples that the gate also compiles and runs (and asserts the exit
code of), two good ones are:

**Verified example** — [`helixc/examples/matmul_2x2.hx`](../../../helixc/examples/matmul_2x2.hx)
(gate-checked to exit `69`):

```helix
fn main() -> i32 {
    let a00 = 1; let a01 = 2; let a10 = 3; let a11 = 4;
    let b00 = 5; let b01 = 6; let b10 = 7; let b11 = 8;
    let c00 = a00 * b00 + a01 * b10;
    let c11 = a10 * b01 + a11 * b11;
    c00 + c11   // 19 + 50 = 69
}
```

**Verified example** —
[`helixc/examples/hbs_sample_recursion.hx`](../../../helixc/examples/hbs_sample_recursion.hx)
(gate-checked to exit `120`): computes `5! = 120` with a recursive state machine over an `enum`,
exercising `match`, enum payload extraction, and pass-by-value across a self-call. Both appear in
the gate's corpus list, so a green `REPRODUCE_TRUST: PASS` includes them passing too
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh): `chk "$EX/matmul_2x2.hx" 69` and
`chk "$EX/hbs_sample_recursion.hx" 120`).

---

## A map of the terms you just saw

A few terms recur throughout the rest of the book; the tour is the first place they appear, so
here is the one-line version of each. Each maps to a real file or output token.

| Term | One-line meaning |
|------|------------------|
| `seed` | the Apache-2.0 **C-subset** compiler (`stage0/helixc-bootstrap/`) built by the raw ladder; it builds `kovc`. Pinned `9837db12…`. |
| `kovc` | the Helix compiler **written in Helix** (`helixc/bootstrap/{lexer,parser,kovc}.hx`) that emits x86-64 ELF directly. |
| the from-raw ladder | `hex0` (299 hand-authored hex bytes) → hex1 → hex2 → catm → M0 → cc_amd64 → M2-Planet → `seed`, each rung built only by the prior one. |
| the self-host fixpoint | `seed → K1 → K2 → K3 → K4` with `K2 == K3 == K4` byte-identical. Pinned `0992dddd…`. |
| the gate | [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) — fixpoint + 109-program corpus + PTX-text regression + negative diagnostics. Prints `GATE_PASS`. |
| gcc-DDC | the gcc diverse-double-compile of `seed→K1` (byte-identical `K1 = 84363adb…`); `gcc` is an auditor, never the shipped root. |

---

## The honest scope of this tour

The tour proves a specific, bounded thing — and Helix's whole value is being precise about which
thing. From [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md):

- **What `REPRODUCE_TRUST: PASS` covers:** the **CPU trust core** — the from-raw ladder, the
  self-host fixpoint, and the gcc diverse-double-compile, all reproducing the pinned anchors from a
  clean checkout. The CPU path is auditable **all the way down** from the hand-typed `hex0` root.
- **What it does *not* cover here:** the **GPU capstone** (a transformer trained on `kovc`-emitted
  kernels) is verified *separately* by `scripts/capstone_audit.sh` on a CUDA host — the
  reproduction script says so in its own verdict and never claims otherwise.
- **Complete to PTX, not to GPU machine code.** Even on the GPU side, the from-raw chain is
  hand-auditable only **to PTX text**; below PTX it trusts NVIDIA's closed `ptxas`, the CUDA
  driver, the GPU hardware, and a C host launcher. GPU performance is a *fraction* of cuBLAS
  (~50–67.5% on the reference RTX 3070 Laptop), not parity. See
  [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R / residuals 7–8.
- **Two reproducibility caveats.** The `seed→K1→…` fixpoint leg currently expects the checkout to
  live at the canonical build path (`assemble_k1.hx` hardcodes it; the reproduction script applies
  a disclosed mechanical rewrite to your checkout's path before building), and the broader v1.1
  language-surface cross-check uses an out-of-tree witness that a clean clone does not carry. The
  **core** chain you ran here *is* clean-checkout reproducible; these caveats are spelled out in
  [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) ("Where it walls") and
  [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) (residuals 9–10). External
  reproduction by a party with no connection to the author remains the one open increment — now
  push-button via the CI workflow above.

None of these caveats weakens what you just did: you rebuilt a self-hosting compiler from a
hand-typed root and watched it reproduce its pinned hashes. The honest framing is simply *that, and
exactly that.*

---

## What you proved in ten minutes

- Cloned the repo at `v1.3-release` — source and a hand-authored hex root, no pre-built compiler.
- Ran `bash scripts/reproduce_trust.sh` to rebuild `hex0 → … → seed → kovc` from raw and verify the
  self-host fixpoint and the gcc diverse-double-compile.
- Read `REPRODUCE_TRUST: PASS` and confirmed the three pinned hashes (`seed 9837db12…`,
  `K1 84363adb…`, fixpoint `0992dddd…`).
- Saw a real Helix program (`exit42.hx`) compile and run through the from-raw compiler — exit `42`,
  as the corpus line `PASS exit42.hx (42)` shows.

That is the trust chain, end to end, on your own machine. Everything else in this book is the
detail behind those four lines.

---

**Next:** [Part II — Prerequisites](../part2-setup-build/01-prerequisites.md) — the environment you
need to build Helix from raw yourself, before [Build from raw](../part2-setup-build/02-build-from-raw.md)
opens the `hex0 → … → seed → kovc` ladder one rung at a time.
