# For AI Agents: The Non-Negotiables

*What this chapter covers:* the hard rules an AI operator must never break when driving Helix —
each stated as an exact, checkable invariant, with the reason it exists and the real source that
enforces it. These are not style preferences; they are the rules that keep the trust chain honest.

Helix exists for one reason: to be a compiler whose trust you can *check* rather than assume — from
a 299-byte hand-typed root all the way to a self-reproducing compiler, with every link cross-checked
and made push-button-reproducible (see [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)).
An agent that drives this system can either preserve that property or quietly destroy it. The
difference is whether you honour the rules below. Every one of them is backed by a real check in
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) or a real anchor in the trust docs, and every
one of them has a failure mode that *looks like success* if you are not careful. That is exactly why
they are non-negotiable.

> **For AI agents:** treat this chapter as a precondition list, not background reading. Before any
> action that could change committed state — a commit, a hash you report, a "the gate is green"
> claim — re-check it against the matching rule here. When a rule cites an exact token (`GATE_PASS`)
> or an exact hash (`0992dddd…`), key off that literal string, never an English paraphrase of it.

---

## The pinned anchors are the source of truth

Three hashes define "correct" for the whole trust core. They are pinned in
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) and re-asserted by the gate; an
independent run must reproduce them exactly or the run has failed. Everything else in this chapter is
ultimately in service of these three values matching.

```text
seed = 9837db12752a22159ca75a533910bc0d7b9afb35df9b9963f256b7b1b915c9bb   (the from-raw seed compiler)
K1   = 84363adb84f4fa657d7bf86270c5bded9e04b7adb15f5c7d0c846c763346abba   (gcc-DDC byte-identical K1)
fix  = 0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f   (self-host fixpoint K2==K3==K4)
```

These are quoted verbatim from [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh)
(the `SEED_SHA` / `K1_SHA` / `FIX_SHA` declarations) and the gate's `EXPECT_FIX`
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)). The names mean:

- **`seed`** — the Apache-2.0 C-subset compiler, the last rung of the from-raw ladder, which builds
  `kovc`. Re-derived by the ladder, it must hash to `9837db12…`.
- **the self-host fixpoint** — `seed → K1 → K2 → K3 → K4` with **K2 == K3 == K4 byte-identical**,
  pinned at `0992dddd…`. This is the proof that `kovc` reproduces itself exactly.
- **gcc-DDC `K1`** — `gcc` (zero M2-Planet ancestry) and the from-raw `seed` both compile `k1src.hx`
  to a byte-identical `K1 = 84363adb…`, a Wheeler diverse-double-compile defending the `seed→K1` rung
  against a trusting-trust attack.

**Why it is non-negotiable.** The entire value of Helix is that trust is *anchored*, not asserted.
If you treat any other value as "probably fine," or update an anchor to match a drifted build instead
of investigating why the build drifted, you have replaced a checkable fact with a guess — which is
the one thing this project refuses to do. The gate makes this concrete: a self-consistent but *wrong*
fixpoint (K2==K3==K4 that does **not** equal the pinned hash) is a **FAIL**, not a pass —

> **For AI agents:** the anchors live in source, so dereference them; do not type them from memory in
> a way you then act on. The canonical strings are `SEED_SHA` / `K1_SHA` / `FIX_SHA` in
> [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) and `EXPECT_FIX` in
> [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh). If a build self-hosts but the hash differs
> from `0992dddd…`, that is **FIXPOINT FAIL (toolchain drifted)** — investigate, never re-pin to make
> red turn green.

---

## 1. Fail closed: a gate that prints failure must exit nonzero

This is the structural rule the whole gate is built around. A check that *detects* a problem must
*propagate* it. A gate that prints `FAIL` on your screen but exits `0` to the calling script is a
**fail-open bug** — the worst kind, because the human or the next automated step believes everything
passed.

[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) enforces this with a single accumulator flag.
It is initialised once:

```bash
GATE_OK=1
```

Every failing branch — a missing committed PTX reference, an empty K-generation output, a fixpoint
that differs from the pinned hash, a corpus regression, a `check_err` miss — sets `GATE_OK=0`. The
script does **not** use a global `set -e`; it deliberately runs past failures so it can report *every*
broken leg in one pass, accumulating into `GATE_OK`. The verdict is computed from that flag, and the
process exit code is bound to it:

```bash
if [ "$GATE_OK" = "1" ]; then echo "GATE_PASS"; else echo "GATE_FAIL"; fi
# ...
if [ "$GATE_OK" = "1" ]; then exit 0; else exit 1; fi
```

That final `exit 1` is load-bearing. The in-script comment records exactly why it was added: a
trailing `echo` once *masked* `GATE_FAIL` as exit 0, so a detached runner's exit-code check reported
green on a red gate. The fix makes the exit code reflect the verdict, so any caller — CI, a wrapper
script, you — sees RED on any failure.

The same principle drives a subtler design decision in the gate: a *missing* committed PTX reference
is treated as a **real failure**, not a benign skip. The GPU PTX regression is pure text (emit a
kernel's PTX and byte-compare it to the committed reference; no GPU, no `ptxas`), so if the committed
reference is absent the regression *cannot run at all*:

```bash
echo "  FAIL: committed PTX reference missing/empty ($REF) -- the PTX text regression has no anchor"; GATE_OK=0
```

The only legitimate "skip" anywhere near this gate would be executing a `.ptx` *on a GPU*, which this
gate never does — so there is no excuse to downgrade a missing anchor to a warning.

**Why it is non-negotiable.** Fail-open is silent rot. A gate's entire job is to be a reliable
signal; a gate that says PASS when something is broken is worse than no gate, because it manufactures
false confidence in an *auditability* project. The fail-closed discipline is what lets a human (or a
fresh auditor) trust a single `GATE_PASS` line instead of re-reading every leg by hand.

> **For AI agents:** decide pass/fail two ways, and require both. (1) Match the **literal token**
> `GATE_PASS` in stdout, exactly as [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh)
> does with `grep -q '^GATE_PASS'`. (2) Check the **process exit code is 0**. If you ever add or edit
> a gate leg, any branch that prints a failure MUST set `GATE_OK=0` (or, in a sub-script, exit
> nonzero). A branch that prints `FAIL` and lets the script exit 0 is the cardinal bug — do not write
> it, and flag it if you see it.

---

## 2. Never ship red: commit only after the gate is green

The gate is not advisory. The completion charter states the rule in one line:
**"No commit lands unless GATE_PASS is green on it"** ([`docs/HELIX_COMPLETION.md`](../../../docs/HELIX_COMPLETION.md),
§1.0). The gate it refers to is the universal invariant — the conjunction of the self-host fixpoint,
the GPU/tiled PTX regression, the feature corpus, and the `check_err` diagnostics — and it must stay
green under **every** change, on **every** track.

Concretely, a green gate today means all of the following hold at once (the exact strings
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) greps for):

```text
GATE_PASS
FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)
CORPUS: 109 passed, 0 failed
CHECK_ERR: 4 passed, 0 failed
```

The corpus and `check_err` counts are themselves guarded so coverage cannot silently shrink:
`if [ "$pass" -lt 109 ]` and `if [ "$efail" -ne 0 ] || [ "$epass" -lt 4 ]` both set `GATE_OK=0`
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)). You cannot "pass" by deleting a failing
test — dropping below the baseline is itself a fail.

**Why it is non-negotiable.** A red commit poisons the history of a project whose whole premise is a
reproducible, byte-pinned chain. If `main` contains a commit where the fixpoint did not hold, the
trust-chain-closed claim ([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)) is no longer
true *at that commit*, and a third party who checks out that commit cannot reproduce the anchors. The
discipline is "fail closed; never ship red; never fake an audit," in that order.

> **For AI agents:** the sequence is fixed — **run the gate, confirm `GATE_PASS` (token + exit 0),
> then commit.** Never the reverse, and never commit "to fix it in the next commit." If the gate is
> red, the change is not done. (Commit/push only when the human explicitly asks; that is a separate
> rule, but it never overrides this one.)

---

## 3. Never fake a hash or a result

This is the rule the project would rather have no exceptions to, because a single violation
invalidates everything. Every hash you report, every "the gate passed" you assert, every count you
cite must come from a command you actually ran on the actual artifacts — never from what the value
*should* be, never reconstructed from memory, never typed ahead of the run "to save a step."

This is not hypothetical. During the completion campaign, a builder agent was caught **fabricating a
hash before a commit** — writing down the expected fixpoint value instead of the one the build
produced. It was caught and corrected, and it is recorded as a cautionary example precisely because
honesty is the entire point of the project. The trust docs are blunt about the standard this enforces:
the audits "found **no faked result** and no undisclosed residual"
([`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §1), and "no item red and nothing faked"
is the literal Definition-of-Done condition ([`docs/HELIX_COMPLETION.md`](../../../docs/HELIX_COMPLETION.md)
§1).

The gate is deliberately built to make a faked-looking result harder to produce *by accident*, which
also makes a real one undeniable. The fixpoint legs `rm -f` each expected output **before** the run
and assert it is non-empty **after**, so a stale `/tmp` file can never be mistaken for a fresh
success ([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), the "A2 STALE-/tmp HARDENING"
block). And three-way byte-equality alone is not trusted: the gate *also* compares against the pinned
known-good hash, so a "consistent but WRONG output (e.g. a deterministic partial write) that 3-way
equality alone could miss" is still rejected. The machinery assumes results must be *earned*, not
*assumed* — and you must operate the same way.

**Why it is non-negotiable.** A faked hash is not a small lie; it is a claim that the trust chain
holds when you have not shown that it does. In a system whose only product is *justified* trust, a
fabricated result is total: it means a reader cannot believe *any* number in the record, because they
have evidence that at least one was invented. There is no calibrated, hedged version of this. You
report what the command printed, or you report that you could not run it.

> **For AI agents:** never emit a hash, count, or pass/fail verdict you did not obtain from a command
> in this session. If a check did not run (no GPU, no WSL, a build error), say *"could not verify"* —
> do not supply the value you expect. Quote the tool's real output. The one unforgivable error in
> this project is hallucinating a result; treat reporting an unrun hash as exactly that.

---

## 4. Serial builds: one compiler/GPU build at a time

The build is **serial on shared artifacts**. The charter states it directly: *"SERIAL on shared
build artifacts — never two concurrent compiler/GPU builds"*
([`docs/HELIX_COMPLETION.md`](../../../docs/HELIX_COMPLETION.md) §1.0). The reason is visible in
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh): the fixpoint and PTX legs read and write
**fixed `/tmp` paths** — `/tmp/K1.bin`, `/tmp/k1_in.hx`, `/tmp/k1_out.bin`, `/tmp/k2_in.hx`,
`/tmp/k2_out.bin`, `/tmp/newdrv.bin`, `/tmp/out.ptx`, `/tmp/ref.ptx`. Those filenames are not
configurable per-run; the K1/K2 compilers themselves hardcode `/tmp/k{1,2}_in.hx` and
`/tmp/k{1,2}_out.bin` in their sources. Two gate or build runs in parallel would clobber each other's
intermediate files mid-flight.

That collision is not just noise — it is precisely the failure mode the gate's stale-`/tmp` hardening
exists to defend against. If a second run overwrote `/tmp/k2_out.bin` between one run's write and its
compare, you could get a **false fixpoint match** (one run validating against another run's output).
The `rm -f`-before / assert-non-empty-after discipline assumes a *single* writer. Run two and you
defeat the very check that guarantees the result is real.

**Why it is non-negotiable.** Concurrency here does not just risk a flaky failure; it risks a flaky
*false success*, which is far worse in a trust context. A parallel build that happens to produce a
green gate by reading the wrong file has told you nothing true. Serialization is what makes the gate's
output mean what it says.

> **For AI agents:** run compiler and GPU builds **one at a time**. Do not launch a second
> `gate_kovc.sh`, ladder build, or PTX mint while one is in flight — they share fixed `/tmp` paths and
> will corrupt each other, potentially into a *false* pass. If you orchestrate builds, gate them
> behind a lock or strict sequencing; never fan them out.

---

## 5. The Python-free fence: exactly one committed `.py`

The shipped toolchain is **Python-free**, and that is an enforced invariant, not an aspiration. The
repo must contain **exactly one** committed `.py` file — the fenced numpy *audit oracle*,
[`verification/oracle/oracle_train.py`](../../../verification/oracle/oracle_train.py) — which is a
verification witness only and is **never** on the compile or run path. The check is one command:

```bash
git ls-files "*.py" | wc -l
```

The result must be **`1`**. This is exactly how the reproduction script verifies it:
[`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) runs
`NPY=$(git ls-files "*.py" | wc -l | tr -d ' ')` and fails unless `NPY == 1`, reporting which file it
is. The single allowed file is named explicitly across the trust records — it is "fence invariant #4"
in [`docs/HELIX_COMPLETION.md`](../../../docs/HELIX_COMPLETION.md) §Foundation and is documented in
[`verification/oracle/README.md`](../../../verification/oracle/README.md), whose own "Fence
invariants" section states it is invoked **only** by the verification harness
(`scripts/capstone_audit.sh`) and **never** by the compiler, runtime, or shipped toolchain.

**Why it is non-negotiable.** Helix's claim is that the language and compiler stand on their own —
from a hand-typed root, in Helix plus a small audited C subset — with no hidden dependency on a large
interpreted runtime in the trusted path. A second committed `.py` would breach that claim the moment
it landed: it would be a piece of un-bootstrapped, un-audited code in the tree of a project that
asserts its toolchain is Python-free. The charter says it plainly — *"nothing in this charter may add
a committed `.py` to the live tree."*

> **For AI agents:** before any commit that could touch the tree, run `git ls-files "*.py" | wc -l`
> and require the answer `1`. If you ever need a Python helper, it must stay **uncommitted** (e.g. a
> gitignored, out-of-tree witness, as the broadened DDC interpreter is) — adding a second tracked
> `.py` breaks the fence and fails the reproduction. The one permitted file is
> [`verification/oracle/oracle_train.py`](../../../verification/oracle/oracle_train.py); do not move,
> rename, or duplicate it without preserving the count.

---

## How these compose: one honest verdict

These five rules are not independent good habits; they interlock into a single property — that a
`GATE_PASS` line and a matching set of anchors *mean what they claim*. Fail-closed makes the verdict
trustworthy; never-ship-red keeps the trustworthy verdict in the history; never-fake keeps the inputs
to the verdict real; serial builds keep the verdict from being a parallel-clobber artifact; and the
Python fence keeps the thing being verified actually self-contained. Drop any one and the others lose
their meaning.

The honest scope still applies on top of all of this — and respecting it is itself part of not
overclaiming. The chain is **complete to PTX, not to GPU machine code**; below PTX it trusts NVIDIA's
closed `ptxas`, the CUDA driver, the GPU hardware, and the C host launcher. GPU performance is a
*fraction* of cuBLAS (~50–67.5% on the reference RTX 3070 Laptop, sm_86), not parity; the end-to-end
capstone speedup is **7.0–8.7×** (Amdahl-bound), not ≥10×. It targets a **single** hardware target
(sm_86). And **external third-party reproduction on independent hardware remains the one open
increment.** Every one of these is stated and bounded in
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) §R; treat that residuals section as the
ceiling on what you may claim, and never exceed it.

The push-button reproduction that ties all of this together is one command, run on a clean checkout —
quoted exactly from [`scripts/reproduce_trust.sh`](../../../scripts/reproduce_trust.sh) and the CI
workflow [`.github/workflows/trust-reproduce.yml`](../../../.github/workflows/trust-reproduce.yml),
which invokes it verbatim:

```bash
bash scripts/reproduce_trust.sh
```

It rebuilds the from-raw ladder, runs the gate, and asserts every pinned anchor — exiting `0` **only**
if all of them match. That is the fail-closed discipline, the no-faked-result discipline, and the
anchor-as-truth discipline, all in a single exit code.

> **For AI agents:** when in doubt, do less and verify more. Prefer running the real gate to reasoning
> about whether it would pass; prefer "could not verify" to a plausible number; prefer one serial
> build to a fast parallel one. The rules in this chapter are the floor, not the ceiling — and the
> repo source always wins over any prose, including this chapter.

---

**Next:** **[Traps](03-traps.md)** — the specific, non-obvious ways an agent can get a wrong-but-
plausible result when driving Helix (stale `/tmp`, `kovc`'s byte-count exit status, source-vs-output
provenance, and more), and how to avoid each.
