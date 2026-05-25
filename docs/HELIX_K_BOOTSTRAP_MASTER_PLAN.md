# Helix K-Bootstrap — Master Plan

**Status:** new top-line goal · **Started:** 2026-05-25 · **Supersedes:**
the v3.1 step-6 framing (`docs/V3_HANDOFF.md` §4a). The
v3.1 cleanup-track shipped a useful slice (steps 3a/3b/4/5/6a, commits
`f5bfd6d` → `56859d1`) and that work stands — but the question "should
v3.1 step 6 delete `x86_64.py`?" has been re-scoped to "build
Helix-in-Helix until *no* Python is in the trusted/production path,
then delete all of Python."

User directive (2026-05-25, verbatim):

> "Do what you feel is best, I do not want any compromises. The end
> product must be completely in Helix compiled in Helix all the way
> from binary, no Python, what you do to achieve this goal with no
> compromise the best way possible is up to you."

This document is the durable plan for that effort.

---

## 1. The goal, restated

The final Helix product:

1. **Source is Helix.** Every compiler pass, every backend, every
   tool, every test runner is `.hx`. The Python code in
   `helixc/` is deleted.
2. **Self-compilation.** A Helix binary `kovc` compiles
   `kovc.hx` (its own source) into a byte-identical `kovc` (the
   N-generation self-host fixpoint).
3. **Bootstrappable from raw binary.** A documented, small "trusted
   seed" — either a hand-assembled stage-0 ELF or a hex-blob — can
   recompile the entire chain. No Python in the trust chain.
4. **DDC-validated** ([Diverse Double-Compiling](https://dwheeler.com/trusting-trust/))
   — at least two independent paths produce bit-identical binaries.

The Python `helixc` package will exist during the transition as a
reference oracle and a fuzzer, then be deleted.

## 2. Why "delete x86_64.py" was the wrong framing

The v3.1 step-6 plan said: delete `x86_64.py` once `test_codegen` is
migrated and a Telegram confirmation is sent. That's correct as far
as it goes, but it left the rest of the Python compiler untouched.
Even with `x86_64.py` gone, the Python `helixc check` driver, the
LLVM IR emitter, all 8 frontend passes, the type checker, the AD
framework — every one of those is still Python. Deleting one file
doesn't reach the no-Python end state.

The new framing: the Helix-in-Helix path (`helixc/bootstrap/kovc.hx`
+ `parser.hx` + `lexer.hx`) becomes feature-complete with Python
`helixc`; then *all* Python in `helixc/` deletes together. The end
state is unambiguous.

## 3. Current Helix-in-Helix coverage (as of `56859d1`)

The bootstrap chain at `helixc/bootstrap/` already exists:

| File | Lines | What it does |
|------|-------|--------------|
| `lexer.hx` | 686 | Helix lexer in Helix. Tokenizes a `.hx` source into the token stream the parser consumes. |
| `parser.hx` | 8060 | Helix parser in Helix. Builds an AST. |
| `kovc.hx` | 6723 | Helix codegen in Helix. Walks the AST and emits a Linux x86-64 ELF binary directly (no LLVM IR intermediate). |
| `evaluator.hx` | 149 | Constant-folding evaluator for AD chain rules. |
| **Total** | **15,618** | |

`kovc.hx` compiles a *subset* of Helix today — it's the historical
bootstrap K0/K1/K2 chain that proved self-hosting was viable. Stages
1-65 of `docs/APPROACH_A_PLAN.md` were the original port plan; many
landed, some did not. The most recent stage progress doc is
`docs/stage66-progress-2026-05-18.md` (borrow-checker scaffolding,
still in Python).

The gap is the **delta between what `kovc.hx` accepts** and **what
Python `helixc/frontend/parser.py` + `helixc/check.py` accept**.
Closing that delta is the work.

## 4. Strategy

### 4.1 No new code paths — extend `kovc.hx`, do not start over

The temptation is to design a new LLVM-IR-to-ELF assembler in Helix
(option C from the user's table). That's a smaller surface but it
also throws away 15,618 lines of working Helix code that already
does Helix-to-ELF directly. The cleaner path: extend
`kovc.hx`/`parser.hx`/`lexer.hx` to feature-complete with the
Python compiler. LLVM IR is not the intermediate — Helix AST is.
Same architecture as `x86_64.py` today (AST → x86), just in Helix.

### 4.2 Three concurrent tracks

**Track K — feature catchup.** Port every Python compiler feature
into `kovc.hx`. This is the bulk of the work. Each port = one chunk =
one audit = one commit, following the established discipline.

**Track P — parity.** A test harness that runs the SAME `.hx`
source through both Python `helixc` and `kovc` (the binary built
from `kovc.hx`), and asserts byte-identical or
behaviorally-identical output. This is the load-bearing gate. The
existing v3.0 Stage-207 / Stage-215 parity infrastructure is the
foundation — extend it.

**Track S — seed-bootstrap.** A minimal stage-0 path: hand-document
or hand-assemble the smallest possible bytes that can compile a
tiny `kovc.hx` subset, then bootstrap up to the full `kovc`. This
is the "from raw binary" piece — the trusted seed.

### 4.3 Discipline

- **Per-chunk discipline preserved.** 3-axis audit
  (silent-failure-hunter / type-design-analyzer / code-reviewer);
  fix HIGH + must-fix MEDIUM; re-audit; commit; push; Telegram.
- **Multi-month effort acknowledged.** This is not finishable in
  one cron iteration. The cron loop's job is to advance the plan
  one coherent chunk per fire.
- **Python stays during transition.** Python `helixc` is the
  reference oracle Track P compares against. Only when every test
  passes byte-identical under both does Python become deletable.
- **Audit-first, not delete-first.** Track K closes the
  Helix-vs-Python feature gap before any Python deletion. The cron
  loop must NEVER auto-delete Python files; that's a final gate
  after Track P is 100% green.

## 5. Stages

### Stage K0 — survey

**Goal:** definitive catalog of what `kovc.hx` accepts vs what
Python `helixc` accepts. Two columns; one row per language feature.

**Deliverable:** `docs/K_BOOTSTRAP_FEATURE_MATRIX.md`.

**Mechanism:** read every Python frontend pass + every test pinning
a feature; cross-reference against `parser.hx`/`kovc.hx`; mark each
row as PARITY / KOVC-MISSING / PYTHON-MISSING. This is the gap
list.

**Estimated effort:** 1-2 cron iterations.

### Stage K1 — first ports

**Goal:** start porting the smallest, most isolated Python features
into `kovc.hx`. Order by dependency — type-system foundations
first, then optimization passes, then advanced features.

The original `APPROACH_A_PLAN.md` Stage 1-28 sequence is the
template. It got partway through (Stages 1-65 in the new
numbering). The K1 work is to identify what's NOT YET ported and
finish it, in dependency order.

**Estimated effort:** dozens of cron iterations across weeks.

### Stage K2 — parity harness

**Goal:** A test runner that, for every `.hx` file in the test
suite, compiles it via both paths and compares output.

- Same source → both compilers → both binaries → same `stdout` +
  same exit code → PASS.
- Phase 1: behavioral parity (same stdout + exit code).
- Phase 2: structural parity (byte-identical ELF, after relocations
  normalized).
- Phase 3: N-generation fixpoint (compile `kovc.hx` with `kovc`,
  get `kovc'`; compile with `kovc'`, get `kovc''`; assert
  `kovc' == kovc''`).

**Estimated effort:** 5-10 cron iterations.

### Stage K3 — seed

**Goal:** the smallest possible "stage 0" that compiles a tiny
subset of Helix sufficient to bootstrap up to `kovc`.

Options to research:
- **Hex-blob stage 0**: a hand-assembled tiny ELF that lexes /
  parses / compiles a hex-encoded `kovc-tiny.hx`.
- **CommonLisp/Scheme stage 0**: pre-existing trusted seed (the
  "live coding bootstrap" projects — bootstrappable.org, GNU
  Mes — have done this work; pull from their patterns).
- **C stage 0**: a minimal C compiler that compiles
  `kovc-tiny.hx`. Compromises the "no language but Helix"
  framing, but matches CompCert/GCC norms.

The decision goes here when the time comes; not blocking on it.

**Estimated effort:** several cron iterations, possibly weeks.

### Stage K4 — the cutover

**Goal:** Python `helixc/` is deleted.

Prerequisites — **all** required:
1. Track K complete: every Python feature has a `kovc.hx`
   counterpart. Spot-checked across the feature matrix.
2. Track P complete: every test in the test suite is byte- or
   behavior-identical between the two paths.
3. Track S complete: there's a documented bootstrap chain from
   the trusted seed.
4. **User confirmation in Telegram.** The same gate that
   protects v3.1 step 6 protects this — only stronger.

#### K4-pre — continuous-audit phase (autonomous, NO deletion)

**User directive 2026-05-25 (verbatim):**

> "I want you to not stop working until I stop you or you get to
> Python deletion at K4 in which case you should run continuous
> audits at K4 before Python deletion fixing anything that comes
> up until I stop you and give you the green light for Python
> deletion."

When the cron-tick worker observes prerequisites 1-3 are met but
prerequisite 4 (user TG confirmation) is still PENDING, it MUST
NOT delete Python. Instead it enters K4-pre — a continuous-audit
loop:

- Each cron-tick chunk = one multi-agent audit cycle across the
  full repo (silent-failure-hunter / type-design-analyzer /
  code-reviewer in parallel; optionally a 4th and 5th axis once
  K2 is wired — see §5 K2 for the parity-harness audit).
- Each cycle's HIGH and must-fix MEDIUM findings are fixed in
  the same iteration (or scoped into a follow-up chunk if too
  large), 3-axis re-audited until clean.
- The autonomous worker NEVER calls a Python-deletion command
  (no `rm helixc/`, no `git rm -r helixc/`, no `git push`
  containing a deletion) while in K4-pre.
- Cycles continue indefinitely until the user explicitly
  greenlights the deletion in Telegram. Then K4-proper executes.

This phase exists because (a) K-track is multi-month and the
user may want a different long-term direction before the
irreversible deletion, and (b) the audit cycles are still
PRODUCTIVE — they harden the eventual self-hosted code rather
than burning idle cycles.

#### K4-proper — the irreversible cutover

Once green-lit:
- `helixc/` (the Python package) removed.
- Tests ported to `.hx`.
- CI runs Helix-only.
- helix_status.py VERSIONS table flips the next-version status
  to "released" once the K4 commit lands and K5 closes.

### Stage K5 — DDC + 5-clean-audits final gate

After cutover, run [Diverse Double-Compiling](https://dwheeler.com/trusting-trust/):
two independent compiler paths produce the bit-identical `kovc`
binary. Then 5 consecutive multi-agent audit cycles with zero new
findings against the self-hosted code.

When this lands, "Helix in Helix from raw binary, zero Python" is
declared shipped.

## 6. Versioning

In the existing `scripts/helix_status.py` VERSIONS table, this is
**v5** per `docs/POST_V3_ROADMAP.md` §v5. Bringing it forward to be
the active focus (instead of v4 first) is the user's call; this
plan accommodates either order.

A reasonable bias: ship v3.1 + v3.2 (cleanup + parity gate) first
under option D — they're already 80% done — and start Track K
in parallel. Track K is multi-month; v3.1/v3.2 can land in days.
This minimizes wasted work and gets a clean release boundary
between the v3 cleanup era and the K-bootstrap era.

## 7. First chunk

**K0 chunk 1** — write
`docs/K_BOOTSTRAP_FEATURE_MATRIX.md`. Iterate every Python
compiler feature; cross-reference against `kovc.hx`; produce the
authoritative gap list.

That's the next cron-tick chunk after this plan lands. Subsequent
iterations pick the highest-priority gap row and port it.

## 8. Cron-loop integration

The standing cron prompt currently targets v3.1 → v3.2 cleanup. It
should be updated (separate from this commit) to point at the K
track when v3.1.0 ships. Until then:

- Cron iterations continue v3.1 finish (under option D) — ship
  v3.1.0 with the cleanup work done so far.
- After v3.1.0 ships, cron picks up K-track per this plan.
- v3.2 (real-execution parity gate) is absorbed into Track P (it's
  the same work, framed differently).

## 9. Honest tradeoffs

- **Time.** This is months of work. The v1.0 release (2026-05-18)
  shipped after roughly a year of dense effort; K-track replicates
  ~30-50% of that effort but in Helix.
- **Risk.** Helix-in-Helix has fewer libraries / tools than
  Python-in-Helix (Python's `helixc/` was built atop a mature
  ecosystem). Track K will surface every gap in the Helix language
  itself; some will need new language features.
- **Dogfooding payoff.** Every Helix language gap surfaced by
  K-track is a real bug or missing feature that benefits all
  Helix users. The pain is the product.

## 10. Standing principles

These carry from v3 unchanged:

- **Fail-closed always** — no silent fallbacks, no plausible-but-wrong
  output.
- **Additive** — new paths don't break old ones until parity-gated.
- **Mock-path discipline** — if a toolchain or feature isn't
  available, return DEFERRED, not FAIL.
- **Per-chunk audit on 3 axes**; fix every HIGH + must-fix MEDIUM
  before commit.
- **Telegram per commit**, beginner-friendly prose.
- **Never force-push, never skip hooks, Claude subscription only,
  never read `C:/Projects/Neptune/api.env`**.

### Autonomous-driving policy (2026-05-25)

Per the user directive recorded in §5 K4-pre, the cron-tick
worker has these stopping conditions and these only:

| Condition | Action |
|-----------|--------|
| User stops the worker explicitly (any TG / chat message saying stop). | Halt. |
| All K4 prerequisites 1-3 met; prerequisite 4 (TG confirmation for deletion) PENDING. | Enter K4-pre continuous-audit loop. Never delete Python. Wait for green-light. |
| Chunk fails audit hard AND fix requires design input only the user can give. | Pause that chunk, surface the question, continue with a different chunk if one is available. |
| Any irreversible action other than K4 (force-push, tag deletion, rm -rf-style ops). | Halt. Surface to user. |

Otherwise: the cron keeps firing, the worker keeps porting K-track
rows one chunk at a time, audited, committed, pushed, Telegram'd.

---

## Appendix A — Why not option C (LLVM IR → ELF in Helix)?

Option C from the 2026-05-25 user-decision table was "bundle a
Python LLVM-IR-to-ELF assembler" — and the user explicitly asked
whether it could be in Helix instead. The plan above rejects
option C in either Python or Helix form, in favor of extending the
existing `kovc.hx` direct-to-ELF path. Three reasons:

1. **`kovc.hx` already exists**, 15K lines of working code. Starting
   a new LLVM-IR-consuming Helix tool throws that away.
2. **LLVM IR is not the natural Helix intermediate.** It exists in
   the Python compiler as a v3.0 cutover artifact. `kovc.hx` lowers
   Helix AST directly to x86 — fewer steps, fewer drift surfaces.
3. **A new LLVM-IR-to-ELF tool would need its own parser, type
   system, code generator, ELF emitter.** Same surface as extending
   `kovc.hx`, but with a redundant intermediate language and no
   reuse of existing Helix-in-Helix work.

The end state is the same — Helix compiling Helix to ELF, no Python.
The path is just more direct.
