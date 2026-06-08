# Reproduce & verify the trust chain (push-button)

*What this chapter covers:* how to reproduce Helix's from-raw trust core yourself, with one
command, on a clean checkout — what [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh)
does stage by stage and what each stage *proves*; the universal gate
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) (self-host fixpoint + 109-program corpus +
GPU-PTX text regression + negative diagnostics); the `gcc` diverse-double-compile
[`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh);
the GitHub Actions CI that reproduces all of it on a clean runner
([`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml)); the
pinned anchors to expect; and exactly how a third party verifies Helix's trust for themselves.

The previous chapter, [Build from raw](02-build-from-raw.md), walked the ladder by hand —
`hex0` (299 hand-authored hex bytes) → `hex1` → `hex2` → `catm` → `M0` → `cc_amd64` →
`M2-Planet` → `seed` → `kovc`. This chapter is about *not* doing that by hand: the same chain,
rebuilt and checked by a single script, so that anyone — including someone who has never met
the author and trusts nothing in the tree — can confirm the trust claim for themselves.

> **Note:** the whole point of Helix is honest, auditable trust (see
> [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)). The verification machinery
> below **fails closed**: it exits non-zero unless every check matches a *pinned* hash. A green
> result means the bytes you built equal the bytes the author committed — not "it looked fine."

---

## 1. The one command

From a clean checkout of the repository, the entire from-raw trust core reproduces with:

```bash
bash scripts/reproduce_trust.sh
```

That is the real, committed one-command reproduction — there is no installer and no hidden
setup step. The script deletes every pre-built rung binary, rebuilds the whole ladder from
`hex0`, runs the self-host fixpoint gate, runs the `gcc` diverse-double-compile, and asserts the
three pinned release anchors. It exits `0` **only** if every check matches; any mismatch exits
non-zero with a `*** FAIL` line naming what diverged.

It is CPU-only, so it runs on any x86-64 Linux — including a GitHub runner — with no GPU and no
local state. On a fast filesystem it completes in about a minute.

Two warnings the script itself states up front, which you must respect:

> **Warning:** `reproduce_trust.sh` **modifies the working tree**. It deletes the pre-built rung
> binaries (to force a real rebuild) and applies a disclosed `/mnt/c → checkout` path rewrite
> (see [§6](#6-the-pinned-anchors)). Run it on a **clean checkout or a throwaway clone**, never
> on a tree you want to keep pristine.

The tools it needs are standard, and the header lists them verbatim:

```text
Tools required: bash 4+, coreutils (sha256sum/stat), xxd, file, objdump (binutils), gcc, grep, sed.
```

> **For AI agents:** the success contract is the final line. On PASS the script prints
> `REPRODUCE_TRUST: PASS`; on failure it prints `REPRODUCE_TRUST: FAIL` and exits non-zero. Gate
> on the **exit status** (`bash scripts/reproduce_trust.sh && …`), and if you parse output, match
> the literal token `REPRODUCE_TRUST: PASS`, not a paraphrase. Do not run it in a worktree you
> need to keep clean — it deletes rung binaries and rewrites paths in place.

---

## 2. What each stage proves

`reproduce_trust.sh` runs four numbered checks (after a disclosed step `[0]` path rewrite). Each
one closes a specific gap in the trust argument; together they cover *root of trust* →
*self-reproduction* → *independent-compiler corroboration*.

### Stage [1] — static fence

```bash
say "[1] static fence"
NPY=$(git ls-files "*.py" | wc -l | tr -d ' ')
NCH=$(git ls-files "*.c" "*.h" | wc -l | tr -d ' ')
```

This asserts the committed tree contains **exactly 1** `.py` file and **24** `.c`/`.h` files.
*What it proves:* the compile/run path is not secretly leaning on a large pile of C or Python.
The single committed `.py` is the independent numpy oracle used by the GPU capstone
(`verification/oracle/oracle_train.py`); the 24 C/H files are the audited raw-ladder sources
(chiefly the `seed`'s `seed.c`). If either count drifts, some untracked host code crept into the
trusted base, and the fence fails closed.

### Stage [2] — the from-raw ladder

```bash
say "[2] from-raw ladder (deleting pre-built rung binaries first, then rebuilding each from the prior)"
rm -f stage0/hex0/hex0.bin stage0/hex1/hex1.bin stage0/hex2/hex2.bin stage0/catm/catm.bin \
      stage0/M0/M0.bin stage0/cc_amd64/cc_amd64.bin stage0/M2-Planet/M2.bin \
      stage0/helixc-bootstrap/seed.bin
LADDER_OK=1
for rung in hex0 hex1 hex2 catm M0 cc_amd64 M2-Planet helixc-bootstrap; do
  if ( cd "stage0/$rung" && bash build.sh ) >"/tmp/rt_${rung}.log" 2>&1; then
    say "    rung $rung : build + self-verify OK"
  ...
```

The loop deletes the pre-built binary for **every** rung, then rebuilds each rung **using only
the prior rung** — `hex0` is reconstructed from its hand-authored hex via `xxd`, then `hex0`
builds the next stage, and so on up to the `seed`. Each rung's own `build.sh` self-verifies its
committed `.sha256` (for `hex0`, that is the SHA-256 check in
[`stage0/hex0/build.sh`](../../../stage0/hex0/build.sh)). Finally the script hashes the produced
`seed.bin` and compares it to the pinned `seed` anchor.

*What it proves:* "no trusted pre-built compiler." The committed `stage0/*/*.bin` are convenience
**reference** copies; trust rests on rebuilding each rung from source and matching its committed
hash — and this stage does exactly that, push-button, starting from 299 hand-typed bytes. This is
the resolution of the old "no committed one-command ladder rebuild" gap noted in
[`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md).

### Stage [3] — the self-host fixpoint (+ corpus + PTX + diagnostics)

```bash
say "[3] self-host fixpoint gate (scripts/gate_kovc.sh)"
bash scripts/gate_kovc.sh >/tmp/rt_gate.log 2>&1 || true
if grep -q '^GATE_PASS' /tmp/rt_gate.log; then say "    GATE_PASS"; else ...
```

This delegates to the universal gate (detailed in [§3](#3-the-universal-gate-gate_kovcsh)) and
then asserts four literal lines in its output:

```text
FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)
CORPUS: 109 passed, 0 failed
CHECK_ERR: 4 passed, 0 failed
GATE_PASS
```

*What it proves:* the `seed` builds a `kovc` that **reproduces itself exactly** (the self-host
fixpoint `seed → K1 → K2 → K3 → K4` with `K2 == K3 == K4` byte-identical and equal to the pinned
fixpoint hash), that the language still compiles and runs a 109-program feature corpus with zero
failures, that the GPU PTX emitter is unchanged at the byte level, and that malformed programs
still fail closed with correct `path:line:col` diagnostics.

### Stage [4] — the `gcc` diverse-double-compile

```bash
say "[4] gcc diverse-double-compile (stage0/helixc-bootstrap/ddc_crosscheck.sh)"
bash stage0/helixc-bootstrap/ddc_crosscheck.sh >/tmp/rt_ddc.log 2>&1 || true
if grep -q 'DDC_ANCHOR_OK' /tmp/rt_ddc.log; then say "    DDC_ANCHOR_OK"; else ...
```

This delegates to the DDC cross-check (detailed in [§4](#4-the-gcc-diverse-double-compile)) and
asserts the `DDC_ANCHOR_OK` verdict plus the pinned K1 hash.

*What it proves:* a Wheeler trusting-trust defense. `gcc` — a toolchain with **zero M2-Planet
ancestry** — and the from-raw `seed` both compile the same source into a **byte-identical** K1.
Identical output from two independent compilers means `M2-Planet` did not inject anything into the
`seed`.

The GPU capstone (a real transformer trained on `kovc`-emitted PTX kernels) is **verified
separately**, because GitHub runners have no GPU — see [§7](#7-what-this-does-not-cover-honest-residuals).

---

## 3. The universal gate (`gate_kovc.sh`)

[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) is the gate every change to the compiler
must pass before it can be committed. It is also the heart of Stage [3] above. Run it directly —
under WSL or any x86-64 Linux — with:

```bash
bash scripts/gate_kovc.sh
```

Its overall verdict is one line: it prints `GATE_PASS` on success and `GATE_FAIL` on any failure,
and (since the v1.3 hardening) its **exit status reflects the verdict** — `0` on pass, `1` on
fail — so a runner can gate on the exit code. Internally it runs these sections:

**[0] Regenerate sources.** It runs `assemble_k1.sh` to regenerate the compiler sources
(`k1src.hx` / `k1input.hx` / `k1ptxdrv.hx`) from the committed `kovc.hx` / `lexer.hx` /
`parser.hx`, so the gate always tests the *current* source, never a stale artifact.

**[1]–[3] Self-host fixpoint and PTX text regression.** It builds `seed → K1 → K2 → K3 → K4` and
asserts the three `kovc` self-compiles are byte-identical to each other **and** equal to the
pinned fixpoint hash. The fixpoint check is the load-bearing one:

```bash
if [ "$S2" = "$S3" ] && [ "$S3" = "$S4" ] && cmp -s /tmp/K2.bin /tmp/K3.bin && cmp -s /tmp/K3.bin /tmp/K4.bin; then
  if [ "$S2" = "$EXPECT_FIX" ]; then
    echo "  FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)"
```

Then it re-mints the PTX driver from the edited compiler and **byte-compares** the emitted PTX
for a known kernel against a *committed reference* PTX (`vector_add_kernel.ref.ptx` and
`tiled_matmul_kernel.ref.ptx`). This leg is pure text — it needs no GPU and no `ptxas`:

> **Note:** the PTX regression is a **text** check, by design. It emits a kernel's PTX and
> `cmp`s it against the committed `.ref.ptx`. A *missing* committed reference is therefore not a
> benign "GPU absent" skip — it means the text regression cannot run at all, so the gate treats
> it as a real failure. (Genuine GPU-hardware-absent execution skips live in
> [`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh), not here.)

**[4] Feature corpus.** It compiles and runs a **109-program** corpus through the freshly built
`K2`, checking each program's exit code against an expected value, and prints:

```text
CORPUS: 109 passed, 0 failed
```

The corpus is not padding — each program exercises a specific language feature: `i64`/`u64`
arithmetic past 2³², `f64`/`bf16`/`f16` arithmetic, pattern matching with guards and ranges,
generics, traits with default methods, closures (including capturing closures passed by value),
struct/enum return-by-value, `>6`-argument calls (SysV stack-passing), arena-backed `Vec`/
`HashMap`/`String`, and more. One member is the canonical first program from the previous chapter:

**Verified example** — [`helixc/examples/exit42.hx`](../../../helixc/examples/exit42.hx)
(compiled and run by the gate corpus; the produced ELF exits with status `42`):

```helix
// First end-to-end Helix program: compiles, links, runs, exits with status 42.
//
// Demonstrates: function decl, return, integer literal.
// Verification: $? == 42 after running the produced ELF.

fn main() -> i32 {
    42
}
```

The gate's own corpus harness invokes it as `chk "$EX/exit42.hx" 42` — it compiles the program
with the just-built `K2`, runs the resulting ELF, and asserts the exit code is `42`.

**[4b] Negative diagnostics (`check_err`).** Finally it feeds malformed programs to `K2` and
asserts each produces a **compile-time non-zero exit**, writes **no** output ELF, and emits an
exact `path:line:col: parse error: unexpected token` diagnostic with the hand-computed line and
column of the offending token:

```text
CHECK_ERR: 4 passed, 0 failed
```

The gate passes only if **all** of these hold: fixpoint byte-identical *and* pinned, PTX
byte-identical to the committed references, corpus 109/0, and `check_err` 4/0.

> **For AI agents:** key off the literal tokens, not English. Success is the final `GATE_PASS`
> line **and** exit `0`. The four sub-anchors to confirm are the verbatim lines
> `FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)`,
> `CORPUS: 109 passed, 0 failed`, `CHECK_ERR: 4 passed, 0 failed`, and a `GPU PTX REGRESSION OK`
> line. A `GATE_FAIL` (or any non-zero exit) is a hard stop — never treat it as a warning.

---

## 4. The `gcc` diverse-double-compile

[`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh)
is the trusting-trust defense for the C-compiler rung. Run it directly with:

```bash
bash stage0/helixc-bootstrap/ddc_crosscheck.sh
```

The idea (Wheeler's diverse double-compiling) is to build the `seed` two independent ways and
prove they behave identically:

- **route M2** — the existing `seed.bin`, built by the from-raw ladder
  (`hex0 … M2-Planet → seed.c → seed.bin`).
- **route GCC** — `gcc`, a completely separate lineage with **no** `M2-Planet` ancestry, compiling
  the **frozen** `seed.c`:

```bash
gcc -std=gnu89 -w $INC -o /tmp/seed_gcc seed.c 2>/tmp/gccerr || { echo "  gcc build FAIL:"; head -8 /tmp/gccerr; exit 1; }
```

`seed.c` is **not edited** (its headers are supplied via `-include`, listed as
`INC="-include stdio.h -include stdlib.h -include unistd.h -include string.h"`). The script then
has **both** seeds compile the **same** regenerated `k1src.hx` (~1.5 MB) into K1, and asserts the
two K1 binaries are byte-identical *and* equal to the pinned K1 anchor:

```bash
if [ "$sm" = "$sg" ] && [ "$sm" = "$EXPECT_K1" ]; then
  echo "  DDC_ANCHOR_OK -- gcc (independent lineage) reproduces the M2-Planet seed's K1 byte-for-byte AND == pinned known-good."
```

*What it proves:* if `M2-Planet` had injected a Thompson-style trojan into the `seed`, the
`M2`-built seed and the `gcc`-built seed would produce **different** K1 — unless the trojan lived
in the *visible source* of `seed.c`, or in *both* `gcc` and `M2-Planet` identically. A
byte-identical K1 from two independent compilers rules out the silent-injection case. `gcc` here
is an **auditor**, never the shipped root of trust.

> **Warning:** the DDC fails *closed* in both directions. If the two K1 match each other but not
> the pinned anchor, the script prints `DDC_FAIL` and exits non-zero (toolchain drift). If the two
> K1 differ, it prints `DDC_ANCHOR_DIFF`, shows the first bytes of difference, and exits `3` — a
> real finding to investigate, never silently passed.

---

## 5. Continuous reproduction on a clean runner (CI)

A reproduction you run yourself on your own machine is good evidence; a reproduction that runs
automatically on a *different* machine with a *fresh* clone and *zero* local state is stronger.
That is what [`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml)
provides. Its job is short, because all the work is in the committed script:

```yaml
jobs:
  reproduce:
    runs-on: ubuntu-latest
    timeout-minutes: 60
    steps:
      - name: Checkout (clean clone, no local state)
        uses: actions/checkout@v4

      - name: Install audit tools (xxd / binutils / gcc / file)
        run: |
          sudo apt-get update
          sudo apt-get install -y xxd binutils gcc file coreutils

      - name: Reproduce from-raw trust core (ladder + self-host fixpoint + gcc-DDC)
        run: bash scripts/reproduce_trust.sh
```

It runs on every push and pull request to `main`, on manual `workflow_dispatch`, and on a weekly
schedule to catch silent drift in the trust core:

```yaml
  schedule:
    - cron: '0 6 * * 1'   # weekly (Mon 06:00 UTC) -- catch any silent drift in the trust core
```

The job is green **only** if the full ladder rebuild, the self-host fixpoint, and the `gcc`-DDC
all reproduce the pinned hashes byte-for-byte on the clean runner. It is CPU-only by design — the
hosted runner has no GPU, so the transformer capstone is not run here; it is verified on a CUDA
host via [`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh) (a self-hosted GPU
runner could add that as a separate job later).

---

## 6. The pinned anchors

Every check above terminates in a comparison against a *pinned* SHA-256. These are the release
anchors an independent run must reproduce. The values are declared verbatim near the top of
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh):

```bash
SEED_SHA=9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb
K1_SHA=84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba
FIX_SHA=0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f
```

| Artifact | What it is | SHA-256 | Size |
|----------|------------|---------|------|
| `seed.bin` | the from-raw C-subset compiler (Stage [2]) | `9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb` | 62 467 B |
| `K1` | `seed → K1`; the `gcc`-DDC anchor (Stage [4]) | `84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba` | 697 425 B |
| self-host fixpoint `K2==K3==K4` | the `kovc` self-reproduction (Stage [3]) | `0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f` | 698 392 B |

These same three values are recorded, with their verbatim verdict lines, in
[`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md) — the committed
proof extract for the `v1.3-release` tag.

> **For AI agents:** treat these three hashes as the canonical anchors. `seed = 9837db12…`,
> `K1 = 84363adb…`, fixpoint `= 0992dddd…`. Key your assertions off these exact strings; if a run
> produces a *different* but self-consistent hash, that is **drift** and a failure, not a new
> baseline — do not re-pin.

**One disclosed caveat (step `[0]`).** The fixpoint layer's `assemble_k1.hx` hardcodes an
absolute build path (`/mnt/c/Projects/Kovostov-Native/...`). So that the build runs at *any* path,
`reproduce_trust.sh` step `[0]` mechanically rewrites that string to the current checkout:

```bash
say "[0] path rewrite  /mnt/c/Projects/Kovostov-Native -> $ROOT"
```

This is a pure path swap, disclosed in
[`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md) under "Where it walls." It is
what lets the CI and any fresh clone build at their own location. (A native parameterization of
the concatenator remains a possible future cleanup; the from-raw ladder → `seed` itself, including
the `gcc`-DDC anchor, already builds from *any* directory using only relative paths.)

---

## 7. How a third party verifies Helix trust

You do not have to take the author's word for any of this. There are two independent paths, and
both are push-button.

**Path A — clone and run one command.** On any x86-64 Linux box (WSL is fine), clone the repo to
a throwaway location and run the reproduction:

```bash
bash scripts/reproduce_trust.sh
```

A `0` exit with `REPRODUCE_TRUST: PASS` means *you* rebuilt the entire `hex0 → seed` ladder from
299 hand-typed bytes, *you* reproduced the self-host fixpoint, and *you* confirmed the `gcc`-DDC —
all matching the pinned anchors. Nothing was taken on faith except the shared substrate
([§7, residuals](#honest-residuals)). Remember the warning: run this on a clone you are willing to
let the script modify.

**Path B — fork and watch CI.** Fork the repository on GitHub. The
[`trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml) workflow runs on a clean
`ubuntu-latest` runner — a machine you do not control and the author does not control — and turns
green only if the byte-identical trust core reproduces there. You can also trigger it on demand
via `workflow_dispatch`. This is the "independent operator" reproduction the trust docs name as
the step beyond same-machine evidence.

### Honest residuals

Matching the calibrated tone of the trust record, here is what this reproduction does **not**
cover (the full list is in [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)
§R and [`docs/CURRENT_HEAD_AUDIT_PACKET.md`](../../../docs/CURRENT_HEAD_AUDIT_PACKET.md)):

> **Residual — complete to PTX, not to GPU machine code.** The CPU path is all-the-way-down from
> raw binary. The GPU path is hand-auditable from `hex0` to PTX, but **below PTX it trusts
> NVIDIA's closed `ptxas` (PTX→SASS), the CUDA driver, the GPU hardware, and the C host launcher.**
> This is the one trusted-once boundary on the GPU side.

> **Residual — GPU performance is a fraction of cuBLAS, not parity.** On the reference RTX 3070
> Laptop (sm_86), the emitted GEMM kernels reach roughly **50–67.5%** of cuBLAS, and the
> end-to-end capstone speedup is **7.0–8.7×** (Amdahl-bound), not ≥10×. Loss parity — the hard
> gate — holds at ~0% (worst-case relative diff `0.00000876` against the independent numpy
> oracle). It is a **single hardware target** (sm_86); there is no cross-arch or AMD validation.

> **Residual — the byte-identical, hash-pinned DDC covers the `seed→K1` surface.** The broader
> v1.1 language surface is cross-checked *behaviorally* (a manually-reconciled audit whose witness
> is out-of-tree, not clean-checkout reproducible). And the **shared TCB** — OS, kernel,
> filesystem, shell, coreutils, `gcc`/`libc`/`binutils`/loader, CPU + microcode, RAM, and the
> audited `seed.c` source — remains trusted; no DDC retires the substrate.

The one open increment, stated plainly in the trust docs, is a reproduction by a party with **no
connection to the author**. The mechanism for it is now in place and trivial: fork the repo and
watch CI, or clone it and run the one command above. Everything required to do that is committed
to the tree.

---

**Next:** [Troubleshooting](05-troubleshooting.md) — what to do when a build, a gate run, or a
reproduction does not go green, and how to read the failure output.
