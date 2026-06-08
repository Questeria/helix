# For AI agents: recipes

*What this chapter covers:* copy-paste recipes — each with the **exact** command(s), the **literal
success signal** to match, and **what it proves** — for the six operations you will run most often
when driving Helix: reproduce the full trust core, run the universal gate, run the gcc
diverse-double-compile, run the GPU capstone audit, compile and run a single `.hx` program, and add
a feature-corpus test and re-gate.

Every command below is quoted verbatim from a real script in the repo. The scripts are the source
of truth; this chapter is a convenience index over them. If a command here ever disagrees with the
script it cites, the script wins and this chapter is the bug — flag it, do not silently follow stale
prose.

> **For AI agents:** key every check off the **literal token** named in each recipe (e.g.
> `GATE_PASS`, `REPRODUCE_TRUST: PASS`, `DDC_ANCHOR_OK`, `CAPSTONE_AUDIT_PASS`), not off an exit
> code alone and not off a paraphrase. Several legs of these scripts exit **non-zero on success**
> (see Recipe 5 and the [Traps](03-traps.md) chapter), so the printed token — and, for the
> wrapper scripts, the propagated process exit — is the reliable signal. Treat the pinned hashes
> `9837db12…` (seed), `0992dddd…` (self-host fixpoint), and `84363adb…` (gcc-DDC K1) as ground
> truth; if a run prints a *different* hash, that is a real finding, not a thing to "fix" by
> editing the anchor.

A note on the working directory and platform. The reproduction wrapper
([`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh)) computes the repo root from
its own location and `cd`s there, so you can invoke it from anywhere. Several of the lower-level
scripts ([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh),
[`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh)) hardcode the absolute build path
`/mnt/c/Projects/Kovostov-Native` and are written to **run as a file under WSL** — run them on the
WSL2 Linux side, not Windows. `reproduce_trust.sh` does a disclosed one-time mechanical path rewrite
of those hardcoded paths to the current checkout (its step `[0]`); that is the documented
portability caveat in [`docs/CLEAN_REPRODUCTION.md`](../../../docs/CLEAN_REPRODUCTION.md). For
verbatim invocations of the lower-level scripts (Recipes 2–4), this chapter shows the absolute
`/mnt/c/...` paths exactly as the scripts and their headers use them.

---

## Recipe 1 — Reproduce the full trust core (one command)

This is the single push-button reproduction of the from-raw trust **core**, on a clean checkout,
with no local state. It is the same command the CI workflow runs on a fresh GitHub runner
([`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml), step
"Reproduce from-raw trust core").

**Command** (from [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh)):

```bash
bash scripts/reproduce_trust.sh
```

> **Warning:** this script **modifies the working tree** — it does the disclosed `/mnt/c` path
> rewrite and it **deletes every pre-built rung binary** before rebuilding each from the prior
> rung. Its own header says: *"intended for a CLEAN CHECKOUT (CI runner or a throwaway clone)… do
> not run on a tree you want pristine."* Run it on a clone you can discard, not on a tree with
> uncommitted work.

**What it does**, in four checks (from the script's banner):

1. **Static fence** — exactly 1 committed `.py`, 24 committed `.c`/`.h`.
2. **From-raw ladder** — delete every pre-built rung binary, then rebuild
   `hex0 → hex1 → hex2 → catm → M0 → cc_amd64 → M2-Planet → seed` using **only the prior rung**
   (hex0 from hand-authored hex via `xxd`); each rung self-verifies its committed `.sha256`; the
   final `seed.bin` must equal the pinned seed hash.
3. **Self-host fixpoint** — runs [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) (Recipe 2).
4. **gcc diverse-double-compile** — runs
   [`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh)
   (Recipe 3).

**Success signal.** The final lines on success are exactly:

```text
REPRODUCE_TRUST: PASS
  from-raw ladder + self-host fixpoint + gcc-DDC all reproduce the pinned anchors from a clean checkout.
  (GPU capstone is verified separately by scripts/capstone_audit.sh on a CUDA host.)
```

and the process exits `0`. On any mismatch it prints `REPRODUCE_TRUST: FAIL` and exits `1`. Along
the way it confirms each pinned anchor; the three to match are:

```text
seed=9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb
K1  =84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba
fix =0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f
```

> **For AI agents:** assert `grep -q '^REPRODUCE_TRUST: PASS'` on the output **and** `exit 0`. Do
> not declare success on a partial log: the script accumulates failures and still runs to the end,
> so the verdict line is authoritative. This recipe is **CPU-only** and does **not** run the GPU
> capstone — that is Recipe 4, on a CUDA host.

**What it proves.** From a hand-typed `hex0` root, with **no trusted pre-built compiler anywhere in
the chain**, the `seed` compiler rebuilds byte-for-byte to its pinned hash; `kovc` reproduces
itself exactly (the self-host fixpoint); and an independent-lineage `gcc` build of the `seed`
produces the *same* `K1`, a Wheeler trusting-trust defense — all reproducible by anyone on a clean
checkout. It does **not** establish the GPU path or the broader v1.1 language surface (see Recipe 4
and the residuals in [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)).

---

## Recipe 2 — Run the universal gate

The gate is the per-change discipline for any edit to the compiler: it asserts the self-host
fixpoint, the GPU PTX **text** regression (no GPU needed), the 109-program feature corpus, and the
4-program negative-diagnostics corpus. Run it as a file under WSL.

**Command** (the invocation [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh)
uses at its step `[3]`):

```bash
bash scripts/gate_kovc.sh
```

The script itself hardcodes its bootstrap directory as
`BS=/mnt/c/Projects/Kovostov-Native/stage0/helixc-bootstrap` and runs there; invoke it on the WSL2
side.

**What it does** (from [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)):

- **[0]** Regenerates `k1src.hx` / `k1input.hx` / `k1ptxdrv.hx` from the edited compiler sources via
  `assemble_k1.sh`.
- **[2]** **Self-host fixpoint** from the edited `kovc.hx`: `seed → K1 → K2 → K3 → K4`, asserting
  `K2 == K3 == K4` byte-identical **and** equal to the pinned known-good fixpoint hash.
- **[1]/[3]** **GPU PTX regression** (pure text): re-mints the PTX driver from `seed` and byte-diffs
  the emitted PTX of `vector_add_kernel.hx` and `tiled_matmul_kernel.hx` against their committed
  `.ref.ptx` references. This needs **no GPU and no `ptxas`**; a missing reference is a real
  failure, not a skip.
- **[4]** **Feature corpus**: compiles + runs 109 programs through the freshly built `K2` and checks
  each program's exit code against its expected value.
- **[4b]** **CHECK_ERR**: 4 malformed programs that must each produce a `path:line:col:` diagnostic
  and a non-zero compile exit with no output ELF.

**Success signal.** The final line is the literal token:

```text
GATE_PASS
```

and the process exits `0`. The internal milestones the gate prints on a clean run — and the exact
strings [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) greps for — are:

```text
  FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)
  CORPUS: 109 passed, 0 failed
  CHECK_ERR: 4 passed, 0 failed
```

On any failure it prints `GATE_FAIL` and exits `1`.

> **For AI agents:** match `grep -q '^GATE_PASS'`. Do **not** infer pass/fail from the `kovc`
> self-compile legs' exit codes: `kovc` returns its **output byte count** as the process exit
> status, so those legs are **non-zero on success** and are validated by non-empty output + the
> SHA, never by `rc==0` (see the [Traps](03-traps.md) chapter). The pinned fixpoint hash the gate
> asserts is `0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f`.

**What it proves.** After your change, `kovc` still self-hosts to the exact pinned fixpoint, an
x86-only change did not perturb the emitted GPU PTX, every supported language feature still
compiles and runs to its expected result, and the compiler still rejects malformed input with a
correct diagnostic. A green gate is the precondition for any commit to the compiler.

---

## Recipe 3 — Run the gcc diverse-double-compile (gcc-DDC)

The gcc-DDC is the trusting-trust defense for the `seed → K1` rung: build the `seed` two
independent ways (the from-raw `seed.bin`, and `gcc` — which has **zero** M2-Planet ancestry — from
the *frozen* `seed.c`), then assert both produce a **byte-identical** `K1`.

**Command** (the invocation
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) uses at its step `[4]`):

```bash
bash stage0/helixc-bootstrap/ddc_crosscheck.sh
```

**What it does** (from
[`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh)):

- **[1]** `gcc` builds the `seed` from the **frozen** `seed.c` (headers supplied via `-include`,
  source unedited); the gcc-seed must pass its no-arg self-test with exit `42`.
- **[2.0]** Regenerates `k1src.hx` from committed source via `assemble_k1.sh` (never trusts a
  possibly-stale `k1src.hx`).
- **[2]** Both seeds compile the **same** `k1src.hx` → `K1_m2` and `K1_gcc`.
- **[3]** Asserts `K1_gcc == K1_m2` byte-identical **and** equal to the pinned known-good `K1`.

The exact `gcc` build line, for reference, is:

```bash
gcc -std=gnu89 -w $INC -o /tmp/seed_gcc seed.c 2>/tmp/gccerr || { echo "  gcc build FAIL:"; head -8 /tmp/gccerr; exit 1; }
```

where `INC="-include stdio.h -include stdlib.h -include unistd.h -include string.h"`.

**Success signal.** On success it prints:

```text
  DDC_ANCHOR_OK -- gcc (independent lineage) reproduces the M2-Planet seed's K1 byte-for-byte AND == pinned known-good.
  => The seed's behavior is independently double-compiled; identical K1 implies identical K2==K3==K4.
```

with both K1 hashes equal to:

```text
84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba
```

The two failure verdicts are deliberately distinct and both **fail closed**: `DDC_FAIL` (a K1 is
self-consistent but `!=` the pinned anchor, i.e. toolchain drift) exits `2`, and `DDC_ANCHOR_DIFF`
(the two K1 **differ** — a real finding to investigate) exits `3`.

> **For AI agents:** match `grep -q 'DDC_ANCHOR_OK'`. A `DDC_ANCHOR_DIFF` is **never** benign — it
> means the two independent compilers disagree; surface it, do not retry until it "passes."

**What it proves.** `M2-Planet` injected nothing into the `seed`: a trojan would have to live in
`seed.c`'s **visible** source (auditable by reading) or be present **identically** in both `gcc`
and `M2-Planet`. Identical `K1` from two independent compilers narrows the compiler-backdoor
surface. Honest scope: the DDC covers the **`seed → K1`** surface only, and it does **not** erase
the shared trusted computing base both sides use (shared `gcc`/libc/binutils/loader, shell,
filesystem, OS, CPU, and `seed.c`-by-reading). The broader v1.1 language surface is cross-checked
only behaviorally and that witness is out-of-tree — see residuals §2 and §9 of
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md).

---

## Recipe 4 — Run the GPU capstone audit

The capstone is the real-capability proof: a 2-layer transformer trained **end-to-end on
`kovc`-emitted GPU (PTX) kernels**, gated against an independent numpy oracle, with negative
controls. One run of this script is the **dynamic** half of a clean audit round. It needs a **CUDA
host with a GPU** — it is *not* run by Recipe 1 or in CI.

**Command** (from [`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh); the optional
argument is just a round label):

```bash
bash scripts/capstone_audit.sh [round-label]
```

> **Warning:** **GPU serial — never invoke two of these at once** (the script header states this
> explicitly). The first thing it does is unset every `HX_*` environment variable so the run is the
> default v1.0 capstone configuration regardless of the inherited environment.

**What it does** (from [`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh)):

- **[1]** Runs [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) (Recipe 2), which also mints
  the PTX driver `/tmp/newdrv.bin` fresh from the raw-binary `seed` this round (kills staleness).
- **[2]–[3]** Uses that seed-minted driver to emit a single PTX module of **15** `kovc`-emitted
  transformer kernels (`combined.ptx`), asserting at least 15 `.entry` kernels.
- **[4a]** Builds the C host launcher (`train_transformer.c`) and runs the built-in **finite-diff
  gradient check** (verify mode), which must print `backward finite-diff: PASS`.
- **[4b]** **Trains** 500 Adam steps on the GPU, asserting fresh `loss_curve.csv` /
  `init_weights.bin` and a converged final loss (`0 < L < 1.0`).
- **[5]** Runs the **independent numpy oracle** and asserts the worst-case relative difference
  between the Helix loss curve and the oracle curve is **< 0.02** (within 2%), over ≥ 10 comparable
  rows, with the two curves genuinely non-identical.
- **[6]** **Negative control**: corrupts a backward kernel's constants and asserts the finite-diff
  check **catches** it (proving the check is load-bearing).

**Success signal.** The final verdict line is the literal token:

```text
CAPSTONE_AUDIT_PASS
```

and the process exits `0` (it propagates the verdict to the exit status so a parent gate cannot read
a printed fail as success). On any failed leg it prints `CAPSTONE_AUDIT_FAIL` and exits `1`.

> **For AI agents:** match `grep -q '^CAPSTONE_AUDIT_PASS'` **and** require `exit 0`. This recipe
> requires real GPU hardware (the reference is an RTX 3070 Laptop, `sm_86`); on a host with no GPU
> it will fail, and that is correct, not a bug to route around. Do not run it on the CI runner from
> Recipe 1.

**What it proves.** The from-raw-binary self-hosted compiler emits GPU kernels that train a real
neural network whose loss curve matches an independent oracle to within 2% (reproduced at ~0%), and
the gradient self-check actually detects a corrupted kernel. Honest residuals you must **not**
overstate: the chain is **complete to PTX, not to GPU machine code** (NVIDIA's closed `ptxas` +
CUDA driver + GPU hardware + the C host launcher are trusted past PTX); GPU performance is a
**fraction of cuBLAS, not parity** (~50–67.5% on this GPU); end-to-end speedup is **7.0–8.7×**,
Amdahl-bound, not ≥10×; and only `sm_86` is tested. All four are documented in §2 of
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md).

---

## Recipe 5 — Compile and run a single Helix program

To compile one `.hx` program and run it, hand it and an output path to a Helix compiler binary, mark
the output executable, run it, and read the exit code — the program's `i32` return becomes the
process exit status. This is exactly the pattern the real scripts use to drive `seed` and `kovc`
(e.g. `./seed.bin k1src.hx /tmp/K1.bin` in
[`stage0/helixc-bootstrap/ddc_crosscheck.sh`](../../../stage0/helixc-bootstrap/ddc_crosscheck.sh),
and the compile-then-`chmod +x`-then-run sequence in
[`scripts/run_corpus_helix.sh`](../../../scripts/run_corpus_helix.sh)).

The program we compile here is the repo's first end-to-end example, which is committed and
compile-verified in the tree:

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

Both commands below run from the bootstrap directory `stage0/helixc-bootstrap/` — the directory
`ddc_crosscheck.sh` `cd`s into and where the from-raw `seed.bin` is produced — so the example is
referenced by its path *relative to that directory*, `../../helixc/examples/exit42.hx`.

**Command — via the from-raw `seed`** (the compiler the ladder produces; same invocation form
`ddc_crosscheck.sh` uses for `./seed.bin k1src.hx /tmp/K1.bin`, which resolves its source path
relative to the bootstrap working directory):

```bash
./seed.bin ../../helixc/examples/exit42.hx /tmp/exit42.bin
chmod +x /tmp/exit42.bin
/tmp/exit42.bin; echo "exit=$?"
```

**Command — via the self-hosted `kovc` (`K2`)** (the same compile→`chmod`→run pattern
[`scripts/run_corpus_helix.sh`](../../../scripts/run_corpus_helix.sh) uses, where `/tmp/K2.bin` is
the `kovc` binary built by the fixpoint and reads its input from the fixed path `/tmp/k2_in.hx`):

```bash
cp ../../helixc/examples/exit42.hx /tmp/k2_in.hx; rm -f /tmp/k2_out.bin
/tmp/K2.bin
chmod +x /tmp/k2_out.bin
/tmp/k2_out.bin; echo "exit=$?"
```

**Success signal.** The produced ELF runs and its exit code is the program's return value:

```text
exit=42
```

> **For AI agents:** two non-obvious points, both real. First, the **compiler** process exits
> **non-zero on success** — it returns its *output byte count* as the exit status — so validate a
> compile by checking the output file is **non-empty**, never by `rc==0` of the compiler (see the
> [Traps](03-traps.md) chapter). Second, the **compiled program's** exit code *is* its `fn main`
> return value (for `exit42.hx`, `42`), so an "exit 42" is the success here, not an error. The
> `K2` form reads `/tmp/k2_in.hx` and writes `/tmp/k2_out.bin` — those filenames are hardcoded in
> the compiler sources, so copy your program to `/tmp/k2_in.hx` and read the result from
> `/tmp/k2_out.bin`.

**What it proves.** The toolchain compiles a complete Helix program to a native x86-64 ELF that
runs and returns the expected status — the smallest end-to-end smoke test of the compile/run path.
For the full set of language features that compile and run, see the feature corpus exercised by the
gate (Recipe 2).

---

## Recipe 6 — Add a feature-corpus test and re-gate

The feature corpus lives **inside** [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), step
`[4]`. Each test is a small complete program with a known exit code, and one `chk` line that
compiles it through the freshly built `K2` and asserts the exit code. To add coverage you write the
program, add a `chk` line, **bump the corpus count guard**, and re-run the gate (Recipe 2).

**Step 1 — write the program.** Two equivalent ways the gate already does this:

- **Inline via a `gen` heredoc** (the gate writes the source into `$CD`, i.e. `/tmp/corpus`,
  itself). For example, the committed `assoc_sub.hx` corpus entry is generated exactly like this:

  ```bash
  gen assoc_sub.hx <<'EOF'
  fn main() -> i32 { 10 - 3 - 2 }
  EOF
  ```

  and later checked with `chk "$CD/assoc_sub.hx" 5`.

- **As a committed fixture** under the generics corpus directory `GENC=$BS/corpus_gen` (i.e.
  `stage0/helixc-bootstrap/corpus_gen/`), checked by path, e.g. `chk "$GENC/gen_vec_i32.hx" 42`.

The program must be a **complete** Helix program with an `fn main` whose return value is your
expected exit code, and it must keep that value `< 256` so it fits the process exit byte (this is
why the existing corpus entries are constructed to land on small numbers).

**Step 2 — add the `chk` line.** `chk` takes the program path and the expected exit code:

```bash
chk "$CD/assoc_sub.hx" 5
```

Its definition, verbatim from the gate, makes the contract explicit — compile via `K2` (output must
be non-empty), then run and compare the exit code:

```bash
chk() { local f="$1" exp="$2" b; b=$(basename "$1")
  [ -f "$f" ] || { echo "  MISSING $b"; fail=$((fail+1)); return; }
  cp "$f" /tmp/k2_in.hx; rm -f /tmp/k2_out.bin; timeout 30 /tmp/K2.bin >/dev/null 2>&1
  [ -s /tmp/k2_out.bin ] || { echo "  COMPILE-FAIL $b"; fail=$((fail+1)); return; }
  chmod +x /tmp/k2_out.bin; timeout 10 /tmp/k2_out.bin; local rc=$?
  if [ "$rc" = "$exp" ]; then echo "  PASS $b ($rc)"; pass=$((pass+1)); else echo "  FAIL $b ($rc!=$exp)"; fail=$((fail+1)); fi
}
```

**Step 3 — bump the count guard.** The gate fails closed if the corpus shrinks. At the end of
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) it asserts:

```bash
if [ "$pass" -lt 109 ]; then echo "  CORPUS REGRESSION (pass=$pass < 109)"; GATE_OK=0; fi
```

Adding a passing test makes the run report e.g. `CORPUS: 110 passed, 0 failed`; you must raise the
`109` guard to the new total so the higher count is enforced going forward (the gate's history is a
long series of exactly these bumps — `56 → 59 → 60 → …`). Leaving the guard low would let a future
regression silently drop your test.

**Step 4 — re-gate.** Run Recipe 2 again:

```bash
bash scripts/gate_kovc.sh
```

**Success signal.** Your new test appears in the corpus matrix as a `PASS` line, the corpus tally
reflects the new total, and the gate still ends with `GATE_PASS`:

```text
  PASS assoc_sub.hx (5)
  ...
  CORPUS: 110 passed, 0 failed
GATE_PASS
```

(If your program is supposed to be **rejected** at compile time, add it to the CHECK_ERR negative
corpus with `chk_err <fixture> <line> <col>` instead, and bump the `CHECK_ERR` count guard the same
way — that path asserts a non-zero compile exit, no output ELF, and an exact `path:line:col:`
diagnostic.)

> **For AI agents:** never lower a count guard to make a run pass, and never edit a committed
> `.ref.ptx` or a pinned hash to silence a diff — those are fail-closed anchors. If your new test
> *changes* the self-host fixpoint hash, that is expected only when you actually changed the
> compiler sources used in self-compilation; a pure corpus addition that does not touch
> `lexer.hx`/`parser.hx`/`kovc.hx` must leave `K2 == K3 == K4` **byte-identical** to the prior
> mint (the gate comments call out exactly which additions are expected to move the hash and which
> are not). Verify the new program in isolation first with Recipe 5 before adding the `chk` line.

**What it proves.** A new, externally specified behavior compiles and runs correctly on the
self-hosted compiler, is now permanently guarded by the gate, and did so **without** regressing the
self-host fixpoint, the GPU PTX references, or any existing corpus program.

---

**Next:** this is the final Stage-1 chapter. Return to the operator manual's foundations in
[Driving Helix](01-driving-helix.md), the [Non-negotiables](02-non-negotiables.md), and the
[Traps](03-traps.md) — and treat [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)
as the ceiling on every claim you make on Helix's behalf.
