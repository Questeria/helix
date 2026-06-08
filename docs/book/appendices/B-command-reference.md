# Appendix B — Command reference

*What this appendix covers:* a terse, one-screen-per-command lookup of every command a reader runs
to drive or verify Helix — the exact invocation (quoted verbatim from the real script), one line on
what it does, the literal success token to match, and what it proves. This is the quick-lookup
companion to [Part IX, Recipes](../part9-for-ai-agents/04-recipes.md), which carries the full
narrative, the warnings, and the per-step breakdown. When this appendix and a script disagree, the
script wins and this appendix is the bug — flag it, do not follow stale prose.

> **For AI agents:** key every check off the **literal token** in the "Success token" column
> (`REPRODUCE_TRUST: PASS`, `GATE_PASS`, `DDC_ANCHOR_OK`, `CAPSTONE_AUDIT_PASS`), not off an exit
> code alone and not off a paraphrase. Several legs of these scripts **exit non-zero on success**
> (the `kovc` self-compile legs return their *output byte count* as the process status), so the
> printed token — and, for the wrapper scripts, the propagated process exit — is the reliable
> signal. See [Part IX, Traps](../part9-for-ai-agents/03-traps.md).

A note on working directory and platform, carried from
[Part IX, Recipes](../part9-for-ai-agents/04-recipes.md): the wrapper
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) computes the repo root from its
own location and `cd`s there, so it runs from anywhere. The lower-level scripts
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh),
[`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh)) hardcode the absolute build path
`/mnt/c/Projects/Kovostov-Native` and are written to **run as a file under WSL** — invoke them on
the WSL2 Linux side. The wrapper does a disclosed one-time mechanical path rewrite of those
hardcoded paths to the current checkout (its step `[0]`); the documented portability caveat is in
[`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md).

---

## Quick-lookup table

| Command | What it does | Success token | Exit | Proves |
|---------|--------------|---------------|------|--------|
| `bash scripts/reproduce_trust.sh` | One push-button clean-room reproduction of the from-raw trust **core** (CPU-only): static fence + the from-raw ladder + self-host fixpoint + gcc-DDC, all against pinned anchors. | `REPRODUCE_TRUST: PASS` | `0` | From a hand-typed `hex0`, the `seed` rebuilds to its pinned hash, `kovc` reaches the self-host fixpoint, and an independent-lineage `gcc` produces the same `K1` — all reproducible by anyone on a clean checkout. |
| `bash scripts/gate_kovc.sh` | The universal **gate** for any compiler edit: self-host fixpoint + GPU PTX text regression + 109-program feature corpus + 4-program negative-diagnostics corpus. | `GATE_PASS` | `0` | After your change, `kovc` still self-hosts to the pinned fixpoint, the emitted GPU PTX is unchanged, every supported feature still compiles+runs to its expected result, and malformed input is still rejected with a correct diagnostic. |
| `bash stage0/helixc-bootstrap/ddc_crosscheck.sh` | The gcc **diverse-double-compile** of the `seed → K1` rung: build the `seed` two independent ways (from-raw `seed.bin`; `gcc` from the frozen `seed.c`) and assert both produce a byte-identical `K1`. | `DDC_ANCHOR_OK` | `0` | `M2-Planet` injected nothing into the `seed` — a trojan would have to live in `seed.c`'s visible source or be present identically in both `gcc` and `M2-Planet`. A Wheeler trusting-trust defense over the `seed → K1` surface. |
| `bash scripts/capstone_audit.sh [round-label]` | One **capstone** audit round (the dynamic half): rebuild the GPU capstone from the self-hosted compiler, train a 2-layer transformer on `kovc`-emitted PTX kernels, finite-diff gradient check, within-2% numpy-oracle compare, negative controls. **Needs a CUDA host.** | `CAPSTONE_AUDIT_PASS` | `0` | The from-raw self-hosted compiler emits GPU kernels that train a real network to within 2% (reproduced at ~0%) of an independent oracle, and the gradient self-check catches a corrupted kernel. Honest residuals below. |
| `bash scripts/reproduce_trust.sh` (CI) | The same wrapper, run on a fresh GitHub runner by the `trust-reproduce` workflow — a different machine, a fresh clone, zero local state. | `REPRODUCE_TRUST: PASS` (job GREEN) | `0` | External-operator, push-button reproduction of the trust core; the step beyond same-machine evidence. CPU-only (no GPU on hosted runners). |
| compile + run one `.hx` (see below) | Hand a `.hx` program and an output path to a Helix compiler binary, mark it executable, run it; the program's `fn main` `i32` return is the process exit status. | `exit=<return value>` | the program's `fn main` value | The toolchain compiles a complete Helix program to a native x86-64 ELF that runs and returns the expected status. |

The three pinned anchors that the trust-core commands assert (treat as ground truth — a *different*
hash is a real finding, never a thing to "fix" by editing the anchor):

```text
seed = 9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb
K1   = 84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba
fix  = 0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f
```

(`seed` and `K1` from [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) lines
29–31; the same three are reprinted in its banner. The full pinned-hash table — with what each one
covers — is [Appendix C, Pinned hashes](C-pinned-hashes.md).)

---

## 1. `bash scripts/reproduce_trust.sh` — reproduce the full trust core

**Command** (from [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh)):

```bash
bash scripts/reproduce_trust.sh
```

**What it does.** Four checks, from the script's banner: `[1]` static fence (exactly 1 committed
`.py`, 24 committed `.c`/`.h`); `[2]` the from-raw ladder — deletes every pre-built rung binary then
rebuilds `hex0 → hex1 → hex2 → catm → M0 → cc_amd64 → M2-Planet → seed` using only the prior rung,
each rung self-verifying its committed `.sha256`, final `seed.bin` == pinned; `[3]` the self-host
fixpoint via the gate (command 2); `[4]` the gcc-DDC (command 3).

> **Warning:** this script **modifies the working tree** — it does the disclosed `/mnt/c` path
> rewrite and **deletes every pre-built rung binary** before rebuilding. Its own header: *"intended
> for a CLEAN CHECKOUT (CI runner or a throwaway clone)… do not run on a tree you want pristine."*

**Success token** — the final lines on success are exactly:

```text
REPRODUCE_TRUST: PASS
  from-raw ladder + self-host fixpoint + gcc-DDC all reproduce the pinned anchors from a clean checkout.
  (GPU capstone is verified separately by scripts/capstone_audit.sh on a CUDA host.)
```

and the process **exits `0`**. On any mismatch it prints `REPRODUCE_TRUST: FAIL` and exits `1`.

> **For AI agents:** assert `grep -q '^REPRODUCE_TRUST: PASS'` **and** `exit 0`. The script
> accumulates failures and still runs to the end, so the verdict line is authoritative — do not
> declare success on a partial log. CPU-only: this does **not** run the GPU capstone (that is
> command 4).

**What it proves.** From a hand-typed `hex0` root with no trusted pre-built compiler anywhere in the
chain, the `seed` rebuilds byte-for-byte to its pinned hash, `kovc` reproduces itself exactly (the
self-host fixpoint), and an independent-lineage `gcc` build of the `seed` produces the same `K1`.
Full narrative: [Part IX, Recipe 1](../part9-for-ai-agents/04-recipes.md#recipe-1--reproduce-the-full-trust-core-one-command).

---

## 2. `bash scripts/gate_kovc.sh` — the universal gate

**Command** (the invocation [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) uses
at its step `[3]`):

```bash
bash scripts/gate_kovc.sh
```

The script hardcodes its bootstrap directory as
`BS=/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap`
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) line 11) and runs there; invoke it on the
WSL2 side.

**What it does.** `[0]` regenerates `k1src.hx` / `k1input.hx` / `k1ptxdrv.hx` from the edited
compiler sources via `assemble_k1.sh`; `[2]` the **self-host fixpoint** from the edited `kovc.hx`
(`seed → K1 → K2 → K3 → K4`, asserting `K2 == K3 == K4` byte-identical **and** == the pinned
known-good fixpoint hash); `[1]`/`[3]` the **GPU PTX text regression** (re-mints the PTX driver from
`seed`, byte-diffs the emitted PTX of `vector_add_kernel.hx` and `tiled_matmul_kernel.hx` against
their committed `.ref.ptx` references — **no GPU, no `ptxas`**); `[4]` the **feature corpus** (109
programs through the freshly built `K2`, each exit code checked); `[4b]` **CHECK_ERR** (4 malformed
programs that must each emit a `path:line:col:` diagnostic and exit non-zero with no output ELF).

**Success token** — the final line is the literal token:

```text
GATE_PASS
```

and the process **exits `0`**. The internal milestones a clean run prints — the exact strings
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) greps for (its step `[3]`) — are:

```text
  FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)
  CORPUS: 109 passed, 0 failed
  CHECK_ERR: 4 passed, 0 failed
```

On any failure it prints `GATE_FAIL` and exits `1`. (The gate's end-of-run guards fail closed if the
corpus drops below 109 or CHECK_ERR is not 4/0:
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) lines 660 and 728.)

> **For AI agents:** match `grep -q '^GATE_PASS'`. Do **not** infer pass/fail from the `kovc`
> self-compile legs' exit codes: `kovc` returns its **output byte count** as the process status, so
> those legs are non-zero on success and are validated by non-empty output + the SHA, never by
> `rc==0`. The pinned fixpoint hash the gate asserts is `0992dddd…`.

**What it proves.** A green gate is the precondition for any commit to the compiler: after the
change, `kovc` still self-hosts to the exact pinned fixpoint, an x86-only change did not perturb the
emitted GPU PTX, every supported language feature still compiles and runs to its expected result,
and the compiler still rejects malformed input with a correct diagnostic. Full narrative:
[Part IX, Recipe 2](../part9-for-ai-agents/04-recipes.md#recipe-2--run-the-universal-gate); to *add*
a corpus test, [Part IX, Recipe 6](../part9-for-ai-agents/04-recipes.md#recipe-6--add-a-feature-corpus-test-and-re-gate).

---

## 3. `bash stage0/helixc-bootstrap/ddc_crosscheck.sh` — the gcc diverse-double-compile

**Command** (the invocation [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) uses
at its step `[4]`):

```bash
bash stage0/helixc-bootstrap/ddc_crosscheck.sh
```

**What it does.** `[1]` `gcc` builds the `seed` from the **frozen** `seed.c` (headers via `-include`,
source unedited), and the gcc-seed must pass its no-arg self-test with exit `42`; `[2.0]`
regenerates `k1src.hx` from committed source via `assemble_k1.sh` (never trusts a possibly-stale
file); `[2]` both seeds compile the **same** `k1src.hx` → `K1_m2` and `K1_gcc`; `[3]` asserts
`K1_gcc == K1_m2` byte-identical **and** == the pinned known-good `K1`. The exact `gcc` build line is:

```bash
gcc -std=gnu89 -w $INC -o /tmp/seed_gcc seed.c 2>/tmp/gccerr || { echo "  gcc build FAIL:"; head -8 /tmp/gccerr; exit 1; }
```

where `INC="-include stdio.h -include stdlib.h -include unistd.h -include string.h"`
([`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh)
lines 12, 18).

**Success token** — on success it prints:

```text
  DDC_ANCHOR_OK -- gcc (independent lineage) reproduces the M2-Planet seed's K1 byte-for-byte AND == pinned known-good.
  => The seed's behavior is independently double-compiled; identical K1 implies identical K2==K3==K4.
```

with both K1 hashes == `84363adb…`, and the process **exits `0`**. The two failure verdicts are
deliberately distinct and both **fail closed**: `DDC_FAIL` (a K1 is self-consistent but `!=` the
pinned anchor — toolchain drift) exits `2`; `DDC_ANCHOR_DIFF` (the two K1 **differ** — a real
finding to investigate) exits `3`.

> **For AI agents:** match `grep -q 'DDC_ANCHOR_OK'`. A `DDC_ANCHOR_DIFF` is **never** benign — it
> means the two independent compilers disagree; surface it, do not retry until it "passes."

**What it proves.** `M2-Planet` injected nothing into the `seed`. Honest scope: the DDC covers the
**`seed → K1`** surface only, and it does **not** erase the shared trusted computing base both sides
use (shared `gcc`/libc/binutils/loader, shell, filesystem, OS, CPU, and `seed.c`-by-reading). The
broader v1.1 language surface is cross-checked only behaviorally — see residuals §2 and §9 of
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md). Full narrative:
[Part IX, Recipe 3](../part9-for-ai-agents/04-recipes.md#recipe-3--run-the-gcc-diverse-double-compile-gcc-ddc).

---

## 4. `bash scripts/capstone_audit.sh` — the GPU capstone audit

**Command** (from [`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh); the optional
argument is just a round label):

```bash
bash scripts/capstone_audit.sh [round-label]
```

**What it does.** `[0]` unsets every `HX_*` env var so the run is the default v1.0 capstone config
regardless of the inherited environment; `[1]` runs [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)
(command 2), which also mints the PTX driver `/tmp/newdrv.bin` fresh from the raw-binary `seed` this
round; `[2]`–`[3]` emits a single PTX module of **15** `kovc`-emitted transformer kernels
(`combined.ptx`), asserting at least 15 `.entry` kernels; `[4a]` builds the C host launcher
(`train_transformer.c`) and runs the built-in **finite-diff gradient check** (verify mode), which
must print `backward finite-diff: PASS`; `[4b]` **trains** 500 Adam steps on the GPU, asserting
fresh `loss_curve.csv` / `init_weights.bin` and a converged final loss (`0 < L < 1.0`); `[5]` runs
the **independent numpy oracle** and asserts the worst-case relative difference vs the Helix loss
curve is **< 0.02** over ≥ 10 comparable rows, with the two curves genuinely non-identical; `[6]` a
**negative control** corrupts a backward kernel's constants and asserts the finite-diff check
**catches** it.

> **Warning:** **GPU serial — never invoke two of these at once**
> ([`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh) line 7). It needs a **CUDA host
> with a GPU** and is *not* run by command 1 or in CI.

**Success token** — the final verdict line is the literal token:

```text
CAPSTONE_AUDIT_PASS
```

and the process **exits `0`** (the verdict is propagated to the exit status so a parent gate cannot
read a printed fail as success:
[`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh) lines 175–177). On any failed leg
it prints `CAPSTONE_AUDIT_FAIL` and exits `1`.

> **For AI agents:** match `grep -q '^CAPSTONE_AUDIT_PASS'` **and** require `exit 0`. The reference
> is an RTX 3070 Laptop (`sm_86`); on a host with **no** GPU it will fail, and that is correct, not
> a bug to route around. Do not run it on the CI runner from command 5.

**What it proves — and the honest residuals you must not overstate.** The from-raw-binary
self-hosted compiler emits GPU kernels that train a real network whose loss curve matches an
independent oracle to within 2% (reproduced at ~0%), and the gradient self-check detects a corrupted
kernel. Residuals, all documented in §2 of
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md):

> **Residual:** the chain is **complete to PTX, not to GPU machine code** (NVIDIA's closed `ptxas` +
> CUDA driver + GPU hardware + the C host launcher are trusted past PTX — PTX-not-SASS); GPU
> performance is a **fraction of cuBLAS, not parity** (~50–67.5% on the reference `sm_86`);
> end-to-end speedup is **7.0–8.7×**, Amdahl-bound, not ≥10×; and only `sm_86` is tested (single
> hardware target).

Full narrative: [Part IX, Recipe 4](../part9-for-ai-agents/04-recipes.md#recipe-4--run-the-gpu-capstone-audit).

---

## 5. The CI workflow — `trust-reproduce`

The `trust-reproduce` GitHub Actions workflow
([`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml)) runs the
trust-core reproduction on a fresh `ubuntu-latest` runner — a different machine, a fresh clone, zero
local state — on push/PR to `main`, on `workflow_dispatch`, and weekly (Mon 06:00 UTC). After
installing audit tools, its single reproduction step is, verbatim:

```bash
bash scripts/reproduce_trust.sh
```

**Success token.** The job is **GREEN** only if that command prints `REPRODUCE_TRUST: PASS` and
exits `0` (same token and exit as command 1). CPU-only by design — GitHub-hosted runners have no
GPU, so the capstone (command 4) is **not** run here; it is verified separately on a CUDA host.

**What it proves.** The external-operator, push-button reproduction of the from-raw trust core: the
step beyond same-machine evidence. Anyone can fork the repo and watch it run, or run
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) locally on a throwaway clone.

> **Residual:** even a green CI run is the *same logic* on a hosted x86-64 Linux runner — it shares
> the runner's OS / CPU / `gcc` / loader as a trusted computing base, and does not run the GPU
> capstone. External third-party reproduction on independent hardware remains the one open
> increment; see §R of [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md).

---

## 6. Compile and run a single Helix program

Hand a `.hx` program and an output path to a Helix compiler binary, mark the output executable, run
it, and read the exit code — the program's `fn main` `i32` return becomes the process exit status.
This is exactly the pattern the real scripts use to drive `seed` and `kovc` (e.g.
`./seed.bin k1src.hx /tmp/K1.bin` in
[`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh)
line 39, and the compile-then-`chmod +x`-then-run sequence in
[`scripts/run_corpus_helix.sh`](../../../scripts/run_corpus_helix.sh)).

The program below is the repo's first end-to-end example, committed and compile-verified, and it is
the very first corpus row the gate asserts (`chk "$EX/exit42.hx" 42`,
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) line 313 — i.e. the gate asserts its produced
ELF exits `42`):

**Verified example** — [`helixc/examples/exit42.hx`](../../../helixc/examples/exit42.hx) (compiles
to a Linux x86-64 ELF; running it gives `$? == 42`):

```helix
// First end-to-end Helix program: compiles, links, runs, exits with status 42.
//
// Demonstrates: function decl, return, integer literal.
// Verification: $? == 42 after running the produced ELF.

fn main() -> i32 {
    42
}
```

Both forms below run from the bootstrap directory `stage0/helixc-bootstrap/` (the directory
`ddc_crosscheck.sh` `cd`s into and where the from-raw `seed.bin` is produced), so the example is
referenced by its path *relative to that directory*, `../../helixc/examples/exit42.hx`.

**Command — via the from-raw `seed`** (same invocation form `ddc_crosscheck.sh` uses for
`./seed.bin k1src.hx /tmp/K1.bin`):

```bash
./seed.bin ../../helixc/examples/exit42.hx /tmp/exit42.bin
chmod +x /tmp/exit42.bin
/tmp/exit42.bin; echo "exit=$?"
```

**Command — via the self-hosted `kovc` (`K2`)** (the same compile→`chmod`→run pattern
[`scripts/run_corpus_helix.sh`](../../../scripts/run_corpus_helix.sh) uses; `/tmp/K2.bin` is the
`kovc` binary built by the fixpoint and reads its input from the fixed path `/tmp/k2_in.hx`, writing
`/tmp/k2_out.bin` — those filenames are hardcoded in the compiler sources):

```bash
cp ../../helixc/examples/exit42.hx /tmp/k2_in.hx; rm -f /tmp/k2_out.bin
/tmp/K2.bin
chmod +x /tmp/k2_out.bin
/tmp/k2_out.bin; echo "exit=$?"
```

**Success token** — the produced ELF runs and its exit code is the program's return value:

```text
exit=42
```

> **For AI agents:** two non-obvious points, both real. First, the **compiler** process exits
> **non-zero on success** — it returns its *output byte count* as the exit status — so validate a
> compile by checking the output file is **non-empty**, never by `rc==0` of the compiler. Second,
> the **compiled program's** exit code *is* its `fn main` return value (here, `42`), so an "exit 42"
> is the success, not an error. The `K2` form reads `/tmp/k2_in.hx` and writes `/tmp/k2_out.bin`.

**What it proves.** The toolchain compiles a complete Helix program to a native x86-64 ELF that runs
and returns the expected status — the smallest end-to-end smoke test of the compile/run path. For
the full set of features that compile and run, see the 109-program feature corpus the gate exercises
(command 2). Full narrative:
[Part IX, Recipe 5](../part9-for-ai-agents/04-recipes.md#recipe-5--compile-and-run-a-single-helix-program).

---

**Next:** [Appendix C — Pinned hashes](C-pinned-hashes.md), the canonical table of every pinned
anchor these commands assert (`seed`, the self-host fixpoint, the gcc-DDC `K1`, and the committed
PTX references) with what each one covers.
