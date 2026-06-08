# The gate and the feature corpus

*What this chapter covers:* the universal gate
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) as the per-change discipline — its four
checks (self-host fixpoint, GPU PTX **text** regression, the 109-program feature corpus, the four
`check_err` negative diagnostics), the **fail-closed** structure that makes a green gate mean
something, and how (and why) to extend the corpus — plus the GPU **capstone** audit
([`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh)) as the separate
real-capability gate. Part IX, [Recipes](../part9-for-ai-agents/04-recipes.md) §2 and §6, tells an
operator *how to run* the gate and *how to add a corpus test*; this chapter is the *why* and the
*mechanism* behind those recipes — read it before you trust, edit, or extend the gate.

The previous chapter covered the trusting-trust problem and the `gcc` diverse-double-compile —
*how we know the `seed` itself is honest*. This chapter is about everything that happens **after**
the `seed` exists: how every subsequent change to `kovc` is held to the same standard, so the
trust the ladder establishes is never silently eroded by a later edit.

---

## 1. What the gate is, and why it is one script

[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) is the single discipline a change to the
compiler must pass before it can be committed. Its own header states the contract directly:

> *"GATE for a kovc.hx change … Verifies, from the EDITED kovc.hx, the full discipline before any
> commit."*
> — [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), header

It is deliberately *one* script and not a constellation of optional checks, because a discipline
that is easy to run partially is a discipline that gets run partially. The gate bundles four
independent guarantees into one verdict so that "did this change keep the tree honest?" has exactly
one answer — the literal token `GATE_PASS` on the last line, and a `0` process exit — and no way to
get a partial credit.

The four checks, in the order the gate runs them, are:

1. **Self-host fixpoint** — `seed → K1 → K2 → K3 → K4`, asserting `K2 == K3 == K4` byte-identical
   **and** equal to the pinned known-good fixpoint hash `0992dddd…`. (Step `[2]`.)
2. **GPU PTX text regression** — re-mint the PTX driver from the edited compiler and byte-`cmp` the
   emitted PTX of two committed kernels against their committed `.ref.ptx` references. Pure text;
   no GPU, no `ptxas`. (Steps `[1]` and `[3]`.)
3. **The feature corpus** — compile **and run** 109 small programs through the freshly built `K2`
   and check each program's exit code against its expected value. (Step `[4]`.)
4. **The `check_err` negative diagnostics** — feed 4 malformed programs to `K2` and assert each
   fails closed: a non-zero compile exit, no output ELF, and an exact `path:line:col:` diagnostic.
   (Step `[4b]`.)

A green gate is therefore a conjunction: the compiler still reproduces itself to the *exact* pinned
bytes, an x86-only edit did not perturb the GPU PTX, every supported language feature still
compiles and runs to its expected result, and the compiler still rejects malformed input with a
correct, located diagnostic. The final guard is literal:

**Fragment** (the overall verdict, from
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step "GATE VERDICT"; not a standalone
program):

```bash
if [ "$GATE_OK" = "1" ]; then echo "GATE_PASS"; else echo "GATE_FAIL"; fi
# ...
if [ "$GATE_OK" = "1" ]; then exit 0; else exit 1; fi
```

> **For AI agents:** the gate's success contract is the literal token `GATE_PASS` on the final line
> **and** `exit 0`. Match `grep -q '^GATE_PASS'` and require the zero exit — do **not** infer the
> verdict from any single leg's exit code. Several legs of this script exit *non-zero on success*
> (the `kovc` self-compiles return their output byte count as the exit status; see §3 below and the
> [Traps](../part9-for-ai-agents/03-traps.md) chapter). The recipe-level "how to run it" lives in
> [Recipes §2](../part9-for-ai-agents/04-recipes.md); this chapter is the mechanism.

Before any check runs, step `[0]` regenerates the self-host sources (`k1src.hx`, `k1input.hx`,
`k1ptxdrv.hx`) from the *edited* compiler sources via `assemble_k1.sh`, so the gate always tests the
current `kovc.hx` / `lexer.hx` / `parser.hx`, never a stale artifact. If that regeneration fails the
gate aborts immediately with a `FATAL assemble` and a hard exit — there is no path where a missing
or stale source silently produces a green result.

---

## 2. The fail-closed philosophy (the heart of this chapter)

Everything else in the gate is downstream of one design decision: **the gate fails closed.** It is
worth being precise about what that means, because it is the difference between a verification tool
and a comfort blanket.

A check **fails open** if, when it *cannot run*, it reports success (or a benign "skip") anyway. A
check **fails closed** if, when it cannot run — a missing input, an empty output, a build that did
not complete, an anchor that is absent — it reports **failure**. A fail-open check is worse than no
check at all, because it manufactures false confidence: the green light no longer distinguishes
"the property holds" from "we never tested the property."

The gate is built so that *every* way a check could fail to run is routed to `GATE_OK=0`, never to a
pass and never to a silent skip. Three concrete instances make the philosophy legible.

### The missing-reference case: a missing anchor is a failure, not a skip

The GPU PTX regression is a pure **text** comparison: emit a kernel's PTX from the freshly re-minted
driver and `cmp` it byte-for-byte against a *committed* reference (`vector_add_kernel.ref.ptx` and
`tiled_matmul_kernel.ref.ptx`). It needs no GPU and no `ptxas`. So what should happen if the
committed reference is *absent*? A naive "GPU work, so skip if unavailable" instinct would skip it —
and fail open. The gate refuses, and the comment that enforces it is the clearest statement of the
whole philosophy in the codebase:

**Fragment** (the fail-closed reasoning, from
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[1]`; not a standalone program):

```bash
# FAIL-CLOSED vs FAIL-OPEN distinction (v1.3 audit-remediation A1): the GPU PTX
# REGRESSION is PURE TEXT -- emit a kernel's PTX from the (re-minted) driver and
# byte-cmp it to the COMMITTED reference. It needs NO GPU and NO ptxas. The ONLY
# legitimate "skip" in this gate is running a .ptx ON A GPU (kernel EXECUTION),
# which this gate never does. Therefore a MISSING committed reference is NOT a
# benign GPU-absent skip -- it means the text regression cannot run at all, so it
# is a REAL gate FAILURE (GATE_OK=0), never a WARN. (Genuine GPU-hardware-absent
# execution skips live in scripts/capstone_audit.sh, not here.)
```

The implementation matches the comment exactly: if the committed reference is missing or empty, the
gate prints `FAIL: committed PTX reference missing/empty` and sets `GATE_OK=0` rather than skipping.
The same logic guards the re-minted driver itself — `seed → newdrv` is a pure x86 build, so a
non-zero rc *or* an empty/stale driver means the emitter cannot run, which is a real failure, "not a
GPU skip." The only thing the gate calls a legitimate skip is *executing* a `.ptx` on actual GPU
hardware — and the gate never does that. Hardware-absent execution skips are confined to the
capstone audit (§5), where they belong.

### The stale-output case: never compare a leftover artifact

The second instance is subtler and just as important. Each leg of the fixpoint writes its output to
a fixed `/tmp` path (the filenames are hardcoded in the compiler sources). If a leg *fails* but a
previous run had left an output file at that path, a naive comparison would read the **stale** file
and could announce a false fixpoint match. The gate hardens against this on both sides: it `rm -f`s
each expected output **before** the run, and **after** the run it asserts the output is non-empty
before anything is compared or hashed.

**Fragment** (the stale-`/tmp` hardening rationale, from
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[2]`; not a standalone program):

```bash
# A2 STALE-/tmp HARDENING (v1.3 ChatGPT 3rd-pass MAJOR): every K-generation now
# (a) rm -f's its expected output file BEFORE the run and (b) after the run asserts the
# produced output is NON-EMPTY. ... On ANY failure we set GATE_OK=0, echo
# a clear "FIXPOINT FAIL: K<n> generation rc/empty" reason, mark FIX_OK=0, and SKIP
# the downstream cmp/sha so a STALE /tmp output can NEVER be copied into a later
# K<n> and yield a FALSE fixpoint match.
```

A `FIX_OK` flag short-circuits the comparison the instant any generation fails or produces empty
output, so the gate never hashes a file it did not just produce. This is fail-closed reasoning
applied to *staleness*: the absence of fresh evidence is treated as failure, not as license to reuse
old evidence. (The capstone audit applies the identical `rm`-before / non-empty-after pattern to its
own artifacts — see §5.)

### The continue-on-failure structure: accumulate, then judge

The third instance is structural. The gate does **not** use a global `set -e`. It intentionally
continues past a failed leg so it can run *every* check and report the full picture — which leg
broke, what the corpus matrix looked like, whether the diagnostics still fire — rather than aborting
at the first red. The trick that makes "continue on failure" safe is that failures are *latched*
into a flag (`GATE_OK`, plus the leg-local `FIX_OK`) that can only go from `1` to `0`, never back. No
later success can clear an earlier failure. So the gate gives you a complete diagnostic report **and**
a verdict that is the logical AND of every check. It runs with `set -u` throughout, so an unset
variable is itself an error rather than a silent empty string.

> **For AI agents:** the inverse rule is just as binding. A `GATE_FAIL`, a non-zero exit, a
> `FIXPOINT FAIL`, a `CORPUS REGRESSION`, a `CHECK_ERR REGRESSION`, or a `GPU PTX CHANGED` is a
> **hard stop** — never a warning to route around. In particular: never "fix" a red by lowering the
> corpus count guard, editing a committed `.ref.ptx`, or re-pinning a hash. Those are the
> fail-closed anchors; editing them to get green inverts the entire point of the gate. If a run
> prints a *different* but self-consistent fixpoint hash, that is **toolchain drift** and a real
> finding, not a new baseline (see [Recipes §6](../part9-for-ai-agents/04-recipes.md)).

---

## 3. Check 1 — the self-host fixpoint

The load-bearing check is the self-host fixpoint, detailed in
[Part VI, *seed to kovc*](../part6-bootstrap/03-seed-to-kovc-fixpoint.md) and in
[Part II §3](../part2-setup-build/04-reproduce-verify-trust.md). The gate's role is to re-prove it
*from the edited `kovc.hx`*: build `seed → K1 → K2 → K3 → K4` and assert the three `kovc`
self-compiles are byte-identical to one another **and** equal to the pinned anchor.

**Fragment** (the fixpoint assertion, from
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[2]`; not a standalone program):

```bash
EXPECT_FIX=0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f
if [ "$S2" = "$S3" ] && [ "$S3" = "$S4" ] && cmp -s /tmp/K2.bin /tmp/K3.bin && cmp -s /tmp/K3.bin /tmp/K4.bin; then
  if [ "$S2" = "$EXPECT_FIX" ]; then
    echo "  FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)"
  else
    echo "  FIXPOINT FAIL (K2==K3==K4 self-consistent but != pinned known-good $EXPECT_FIX -- toolchain drifted)"; GATE_OK=0
  fi
else
  echo "  FIXPOINT FAIL (K2/K3/K4 differ)"; GATE_OK=0
fi
```

Two independent conditions must both hold, and the gate's own comment explains why both are needed:
the byte-identical three-way equality is the *fundamental* check (the compiler reproduces itself),
while the pinned `0992dddd…` hash *additionally* rejects a consistent-but-wrong output — a
deterministic partial write, say — that three-way equality alone could miss. A change that is
self-consistent but lands on a different hash is **drift**, and the gate calls it a failure by name.

There is one non-obvious point that the gate documents at length and that an operator must
internalize: the `kovc` self-compile legs (`K1 → K2 → K3 → K4`) are **not** validated by `rc == 0`.
`kovc` returns its **output byte count** as the process exit status — for the 698 392-byte
self-compile, `698392 mod 256 == 24`, i.e. *non-zero on success*. So those legs are validated by
non-empty output plus the SHA, never by the exit code. Only the `seed → K1` leg (a C-compiled
binary) is checked for `rc == 0`, because the `seed` exits `0` on success in the ordinary way.

> **For AI agents:** the pinned fixpoint the gate asserts is
> `0992dddd0edba367d6ff32599c18c4316df1b56d644db36bbc6f69ff0a4bd20f`. Confirm the verbatim line
> `FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)` in the output. Do **not** key
> off the self-compile legs' exit codes — they are non-zero by design. Treat
> [`docs/TRUST_CHAIN_CLOSED.md`](../../TRUST_CHAIN_CLOSED.md) as the authority on what this hash
> attests and on its residual scope.

A pure corpus or documentation change that does **not** touch `lexer.hx` / `parser.hx` / `kovc.hx`
must leave the fixpoint hash byte-identical to the prior mint; only a change to the sources used in
self-compilation may legitimately move it. The gate's history (its inline comments) is a running
ledger of exactly which edits were expected to move the hash and which were not — a discipline that
turns the fixpoint into a tamper-evident seal on the compiler's own source.

---

## 4. Checks 2–4 — PTX regression, the corpus, and negative diagnostics

### The GPU PTX text regression

§2 already covered the *fail-closed* design of this leg; here is what it actually proves. After
re-minting the PTX driver from the edited compiler, the gate emits the PTX for `vector_add_kernel.hx`
and `tiled_matmul_kernel.hx` and byte-compares each against its committed `.ref.ptx`. The intent is
captured in the leg's own message — an x86-only fix must **not** change the emitted PTX:

**Fragment** (the vector-add PTX regression verdict, from
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[3]`; not a standalone program):

```bash
elif cmp -s /tmp/out.ptx /tmp/ref.ptx; then echo "  GPU PTX REGRESSION OK (PTX byte-identical pre/post fix)";
else echo "  GPU PTX CHANGED -- inspect (x86-only fix should NOT alter PTX)"; GATE_OK=0; fi
```

The tiled-GEMM leg adds a *provenance* check on top of the byte-compare: it greps the **emitted
output** (never the source) for the instruction signatures that prove the tiled, double-buffered
GEMM actually lowered — `.shared`, `bar.sync 0`, and the `cp.async` `commit_group` / `wait_group`
double-buffer family. If the emitter *intentionally* changes the PTX (as milestones T2/M0 and T2/G2
did, when the target moved to `sm_86` and `cp.async` double-buffering landed), the discipline is to
re-mint and re-commit the reference *with a recorded reason* — never to relax the check. This is the
PTX-not-SASS boundary in action: the gate verifies the emitted PTX **text** exactly, and trusts
nothing below it (see [Part VII §3, *Honest performance*](../part7-gpu/03-honest-performance.md) and
[`docs/TRUST_CHAIN_CLOSED.md`](../../TRUST_CHAIN_CLOSED.md)).

### The 109-program feature corpus

Step `[4]` is the breadth check: 109 small complete programs, each compiled **and run** through the
freshly built `K2`, each asserting a specific exit code. The corpus is not padding — every entry
locks down a concrete language feature. The members range over `i64`/`u64` arithmetic past 2³²,
`f64`/`bf16`/`f16` arithmetic, pattern matching with guards and ranges, generics, traits with
default methods, closures (including capturing closures passed by value), struct/enum
return-by-value, `>6`-argument calls (SysV stack-passing), and the arena-backed `Vec` / `HashMap` /
`String` from the stdlib. Some are inlined into the gate via `gen` heredocs (written into
`/tmp/corpus`); others are committed fixtures under
[`stage0/helixc-bootstrap/corpus_gen/`](../../../stage0/helixc-bootstrap/corpus_gen) checked by path.

The contract per program is the `chk` function, and reading it is the fastest way to understand what
"in the corpus" actually means:

**Fragment** (the corpus harness, from
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4]`; not a standalone program):

```bash
chk() { local f="$1" exp="$2" b; b=$(basename "$1")
  [ -f "$f" ] || { echo "  MISSING $b"; fail=$((fail+1)); return; }
  cp "$f" /tmp/k2_in.hx; rm -f /tmp/k2_out.bin; timeout 30 /tmp/K2.bin >/dev/null 2>&1
  [ -s /tmp/k2_out.bin ] || { echo "  COMPILE-FAIL $b"; fail=$((fail+1)); return; }
  chmod +x /tmp/k2_out.bin; timeout 10 /tmp/k2_out.bin; local rc=$?
  if [ "$rc" = "$exp" ]; then echo "  PASS $b ($rc)"; pass=$((pass+1)); else echo "  FAIL $b ($rc!=$exp)"; fail=$((fail+1)); fi
}
```

Every branch is fail-closed: a missing fixture is a `MISSING` failure; an empty `K2` output is a
`COMPILE-FAIL`; a wrong exit code is a `FAIL`. Only a present fixture that compiles to a non-empty
ELF *and* runs to the expected code counts as a `PASS`. Note the `timeout` on both compile and run —
a hang is a failure, not a wedge, so the gate always terminates.

Because the corpus consists of real `.hx` files compiled and run by the gate, it is the standing
compile-proof for the language surface. A typical member is a complete program whose exit code is
its own assertion. The pattern-guard fixture is representative:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/g1_guard_true.hx`](../../../stage0/helixc-bootstrap/corpus_gen/g1_guard_true.hx)
(compiled and run by the gate corpus; the gate asserts exit code `1` via `chk "$GENC/g1_guard_true.hx" 1`):

```helix
fn main() -> i32 {
    let x = 7;
    match x {
        n if n > 5 => 1,
        _ => 0
    }
}
```

Here `x == 7`, the guard `n > 5` holds, so the first arm is taken and the program exits `1` — which
is exactly the value the gate's `chk` line demands. The whole pattern-matching feature, from the
earliest example [`helixc/examples/exit42.hx`](../../../helixc/examples/exit42.hx) (`chk
"$EX/exit42.hx" 42`) through the struct-destructuring showcase
[`helixc/examples/dogfood_18_pat_struct_showcase.hx`](../../../helixc/examples/dogfood_18_pat_struct_showcase.hx)
(`chk "$EX/dogfood_18_pat_struct_showcase.hx" 42`) and the generics/closures fixtures in
`corpus_gen/`, is exercised this way on every gate run. The corpus result line is literal:

```text
  CORPUS: 109 passed, 0 failed
```

and is reproduced `PASS` (109/0) at the `v1.3-release` tag — the standing compile-proof referenced
throughout this book.

### The four `check_err` negative diagnostics

Compiling correct programs proves the compiler *accepts* what it should; step `[4b]` proves it
*rejects* what it should — and rejects it *well*. Four malformed fixtures are fed to `K2`, and each
must fail closed in three ways at once: a non-zero compile exit, **no** output ELF, and an **exact**
`path:line:col: parse error: unexpected token` diagnostic with the hand-computed line and column of
the offending token. The contract is the `chk_err` function:

**Fragment** (the negative-diagnostics harness, from
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4b]`; not a standalone program):

```bash
chk_err() { # <fixture> <expected_line> <expected_col>
  local f="$1" el="$2" ec="$3" b; b=$(basename "$1")
  [ -f "$f" ] || { echo "  EMISSING $b"; efail=$((efail+1)); return; }
  cp "$f" /tmp/k2_in.hx; rm -f /tmp/k2_out.bin
  local out rc want; out=$(timeout 20 /tmp/K2.bin 2>&1); rc=$?
  want="/tmp/k2_in.hx:${el}:${ec}: parse error: unexpected token"
  if [ "$rc" = "0" ]; then echo "  EFAIL $b (compiler exited 0 on a parse error)"; efail=$((efail+1)); return; fi
  if [ -s /tmp/k2_out.bin ]; then echo "  EFAIL $b (wrote an output ELF despite the error)"; efail=$((efail+1)); return; fi
  if [ "$out" = "$want" ]; then echo "  EPASS $b -> '$out' (exit $rc)"; epass=$((epass+1));
  else echo "  EFAIL $b: got '$out' want '$want' (exit $rc)"; efail=$((efail+1)); fi
}
```

This is fail-closed in a second sense: it is not enough for the compiler to *reject* bad input — it
must reject it without emitting an ELF (so a downstream step cannot accidentally run a half-built
binary) and it must point at the *exact* offending token. A bare byte offset or a runtime trap would
not satisfy the `want` string. The four fixtures are the smallest possible probes:

**Verified example** — [`stage0/helixc-bootstrap/corpus_gen/err_at_l1.hx`](../../../stage0/helixc-bootstrap/corpus_gen/err_at_l1.hx)
(a malformed program; the gate asserts a non-zero compile exit, no ELF, and the diagnostic
`…:1:20: parse error: unexpected token` via `chk_err "$GENC/err_at_l1.hx" 1 20`):

```helix
fn main() -> i32 { @ }
```

The stray `@` is the 19th byte of the line, so the reported column is `1:20` (1-based) — the value
the gate computes by hand and checks against the compiler's output verbatim. The other three
fixtures place the same illegal token in a let-RHS (`err_let_rhs.hx`, expect `1:28`), on the third
line of a multi-line program (`err_multiline_l3.hx`, expect `3:13`), and after a binary operator
(`err_after_op_l2.hx`, expect `2:9`) — each exercising the lexer's line/column tracking through a
different parse context. The result line is literal:

```text
  CHECK_ERR: 4 passed, 0 failed
```

A clean program produces *no* diagnostic at all (its AST carries no error node), so the negative
corpus never perturbs the fixpoint or the positive corpus.

---

## 5. The count guards — and how to extend the corpus

The breadth checks are protected by **count guards** that fail closed if the corpus *shrinks*. This
is the mechanism that turns "add a test" into "permanently guard a behavior":

**Fragment** (the corpus and `check_err` count guards, from
[`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step "GATE VERDICT"; not a standalone
program):

```bash
if [ "$efail" -ne 0 ] || [ "$epass" -lt 4 ]; then echo "  CHECK_ERR REGRESSION (epass=$epass efail=$efail; want 4/0)"; GATE_OK=0; fi
# ...
if [ "$pass" -lt 109 ]; then echo "  CORPUS REGRESSION (pass=$pass < 109)"; GATE_OK=0; fi
```

The `< 109` and `< 4` guards mean a future change that *drops* a corpus program — by breaking it, by
deleting its `chk` line — fails the gate even if everything that remains passes. Without the guard, a
silent shrink would be invisible: the matrix would still be all-green, just shorter. The guard makes
the *count itself* an anchor.

This is why extending the corpus is a four-step operation (the operator-level walkthrough is
[Recipes §6](../part9-for-ai-agents/04-recipes.md); the *reason* for each step is here):

1. **Write a complete program** with an `fn main` whose return value is the expected exit code, kept
   `< 256` so it fits the process exit byte. Either inline it via a `gen` heredoc or commit it under
   `corpus_gen/`. (Programs that must be *rejected* go to the `check_err` corpus instead.)
2. **Add the `chk` line** (or `chk_err <fixture> <line> <col>`). For a positive test the contract is
   exactly the `chk` shown in §4: compile via `K2`, require non-empty output, run, compare the exit
   code.
3. **Bump the count guard** — raise the `109` (or `4`) to the new total. This is the load-bearing
   step: it is what makes the higher count *enforced* going forward. The gate's history is a long
   series of exactly these bumps (`56 → 59 → 60 → … → 109`), each one a one-line commit recording why
   the count rose.
4. **Re-gate** and confirm the new `PASS` line, the new tally, and `GATE_PASS`.

The honesty discipline here is sharp and worth stating plainly: when you add a corpus test, you must
also know whether your change touched the self-host sources. A pure corpus addition that does **not**
edit `lexer.hx` / `parser.hx` / `kovc.hx` *must* leave the fixpoint hash byte-identical to the prior
mint — the gate's comments call out, for every historical addition, whether it was expected to move
the hash. A test that *both* adds coverage *and* (silently) moves the fixpoint is a red flag that
the change did more than advertised.

> **For AI agents:** never lower a count guard, edit a committed `.ref.ptx`, or re-pin a hash to make
> a run go green — those are fail-closed anchors, and editing them inverts the gate's purpose. Verify
> a new program in isolation first ([Recipes §5](../part9-for-ai-agents/04-recipes.md)) before adding
> its `chk` line. If your addition unexpectedly moves the fixpoint hash, stop and find out why — only
> a deliberate change to the self-compiled sources may do that.

---

## 6. The capstone audit — the real-capability gate

The gate proves the compiler is *correct and self-reproducing*. It does not, by itself, prove the
compiler can do anything *hard*. That is the job of the **capstone**: a real-capability proof
separate from, and built on top of, the gate.

[`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh) runs one round of the dynamic half
of the capstone: it rebuilds the GPU capstone **from the raw-binary self-hosted compiler**, trains a
2-layer transformer on the reference GPU (an RTX 3070 Laptop, `sm_86`), runs a built-in
finite-difference gradient check, compares the loss curve to an **independent** numpy oracle within
2%, and runs negative controls that must fail-as-expected. Its first act is to run the gate itself:

**Fragment** (the capstone audit's first leg, from
[`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh) step `[1]`; not a standalone
program):

```bash
echo "=== [1] gate_kovc.sh (fixpoint + GPU PTX + corpus; mints /tmp/newdrv.bin from seed) ==="
bash $ROOT/scripts/gate_kovc.sh > /tmp/ca_gate.log 2>&1
if grep -q "^GATE_PASS" /tmp/ca_gate.log; then
  echo "  GATE_PASS  ($(grep -m1 'K2=' /tmp/ca_gate.log | cut -c6-21)... ; $(grep -m1 'CORPUS:' /tmp/ca_gate.log | sed 's/^ *//'))"
else echo "  GATE_FAIL"; tail -8 /tmp/ca_gate.log | sed 's/^/    /'; OK=0; fi
```

So the capstone audit *includes* the full gate — and additionally uses the gate's freshly
seed-minted PTX driver (`/tmp/newdrv.bin`) to emit the transformer kernels, which kills staleness:
the kernels that train the network are emitted by the same driver the gate just minted from the
raw-binary `seed` this round, never a possibly-stale prebuilt binary.

The capstone is fail-closed end-to-end, with the same `rm`-before / non-empty-after artifact
discipline as the gate, plus several guards that specifically defend against a *vacuous* pass:

- It **unsets every `HX_*` environment variable** before running, so a stray value in the caller's
  environment cannot silently change the dimensions or op-set into a different run masquerading as
  the v1.0 capstone — and it then asserts none remain.
- The finite-diff gradient check must print the literal `backward finite-diff: PASS`.
- The trained final loss must satisfy `0 < L < 1.0` (negative control NC1 — a non-converging or
  degenerate run fails).
- The within-2% oracle compare requires the worst-case relative difference `< 0.02` over **≥ 10**
  comparable rows (a "vacuous-pass guard" — too few rows is itself a failure), and requires the two
  curves to be genuinely **non-identical** (NC2 — byte-identical curves would mean they were not
  independent).
- The oracle's own exit status must be `0`; a non-zero oracle exit (which includes a *failed* analytic-vs-finite-diff
  backward self-check) is a gate failure.
- A **perturbation negative control** corrupts a backward kernel's constants and asserts the
  finite-diff check **catches** it — if the corrupted kernel still passed, the check would not be
  load-bearing, and the audit fails.

The verdict propagates to the process exit, exactly like the gate:

**Fragment** (the capstone verdict, from
[`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh) step "VERDICT"; not a standalone
program):

```bash
if [ "$OK" = "1" ]; then echo "CAPSTONE_AUDIT_PASS"; else echo "CAPSTONE_AUDIT_FAIL"; fi
# ...
if [ "$OK" = "1" ]; then exit 0; else exit 1; fi
```

> **For AI agents:** match `grep -q '^CAPSTONE_AUDIT_PASS'` **and** require `exit 0`. This audit
> needs real GPU hardware (the reference is an RTX 3070 Laptop, `sm_86`); on a host with no GPU it
> *fails*, and that is correct — it is **not** a bug to route around, and it is **not** run by the
> CPU-only reproduction or in CI. Run it serially: never invoke two capstone audits at once.

**The honest residual.** The capstone is a genuine, falsifiable capability proof — but it is precisely
bounded, and the book must not overstate it. The trust chain is **complete to PTX, not to GPU machine
code**: below PTX it trusts NVIDIA's closed `ptxas`, the CUDA driver, the GPU hardware, and the C host
launcher. GPU performance is a **fraction of cuBLAS, not parity** — roughly **50–67.5%** of cuBLAS on
the reference `sm_86`, with an end-to-end speedup of **7.0–8.7×** (Amdahl-bound), not ≥10×. It is a
**single hardware target** (`sm_86`); there is no cross-architecture or AMD validation. The hard gate
that *does* hold tightly is loss parity against the independent oracle — reproduced at ~0%. All of
these residuals are recorded, with their measured numbers, in
[`docs/TRUST_CHAIN_CLOSED.md`](../../TRUST_CHAIN_CLOSED.md) and discussed in
[Part VII §3](../part7-gpu/03-honest-performance.md). Treat that document as the ceiling on what may be
claimed here.

---

## 7. Why two gates, not one

It is worth closing on the division of labor, because it is deliberate. The **gate** is the
*per-change* discipline: fast, CPU-only, run on every edit and in CI, and it answers "is the compiler
still correct and still itself?" The **capstone audit** is the *real-capability* discipline: slow,
GPU-bound, run on a CUDA host, and it answers "does this compiler actually emit kernels that train a
real network to oracle parity?" The gate is a precondition for the capstone (the capstone runs the
gate as its first leg), but the capstone is *not* a precondition for a commit — because most commits
do not change the GPU path, and gating every commit on a GPU run would make the discipline too
expensive to honor.

Both fail closed. Both propagate their verdict to the process exit. Both refuse to treat "could not
run" as "passed." That is the entire trust posture of the project compressed into two scripts: a
green light only ever means *the property was tested and held* — never *the test was skipped*.

> **For AI agents:** the two literal verdict tokens are `GATE_PASS` (from
> [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh), every change) and `CAPSTONE_AUDIT_PASS`
> (from [`scripts/capstone_audit.sh`](../../../scripts/capstone_audit.sh), the GPU capability proof).
> Each requires both its token **and** `exit 0`. They are not interchangeable: a green gate does not
> imply a green capstone, and the capstone is GPU-only.

---

**Next:** [Residuals & how trust was verified](03-residuals-and-tcb.md) — the honest ceiling on every
claim in this book: what the from-raw ladder, the gate, and the capstone do *not* establish (the
PTX-not-SASS boundary, the shared TCB, the single-target GPU performance fraction, and the one open
increment of third-party reproduction), drawn from
[`docs/TRUST_CHAIN_CLOSED.md`](../../TRUST_CHAIN_CLOSED.md).
