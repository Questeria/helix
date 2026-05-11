# Stage 28.8 Pre-29 Audit Gate — Cycle 11, Audit C: Code Review

**Date**: 2026-05-11
**Commit**: c2e36d4 (read-only)
**Scope**: Audit C (general code-review) at commit c2e36d4 —
**stability re-verification cycle**. No new commits have landed
since cycle 10 (last commit c2e36d4 "Audit 28.8 cycle 10:
regression tests for C8-1 + C8-2 (close C9-1 LOW)"). This cycle
performs a fresh-eyes pass on the cycle-9 + cycle-10 surface
area (the only state-change since cycle 8's clean baseline at
5d1ca24) and re-evaluates every prior below-threshold concern
from cycles 6 / 7 / 8 / 9 / 10 to confirm none should now be
promoted to >= 80 confidence.

Cycle-counter status: prior cycle (10) was CLEAN, advancing
the gate from 0 to 1 toward the 5-clean-cycles deprecation
gate. This cycle (11), if clean, advances 1 -> 2.

Files re-reviewed at HEAD (c2e36d4):

1. `helixc/check.py` lines 55-64 — module-level imports,
   confirming the monkeypatch target `typecheck` is bound here.
2. `helixc/check.py` lines 246-256 — the cycle-9
   `_emit_env_error` helper (strip-and-prefix invariant).
3. `helixc/check.py` lines 286-318 — the cycle-5/cycle-9 outer
   exception cascade (env-error arms route through helper;
   ImportError falls to broad-Exception compiler-bug arm).
4. `helixc/check.py` lines 319-338 — the cycle-3 finally drain
   wrap (unchanged since cycle 5; verified still byte-equal).
5. `helixc/check.py` line 403 — the `typecheck(prog)` call site
   where the monkeypatched typecheck is invoked.
6. `helixc/tests/test_typecheck.py` lines 1530-1569 — the
   cycle-5 C4-6 sibling regression tests.
7. `helixc/tests/test_typecheck.py` lines 1572-1634 — the three
   cycle-10 regression tests added to close C9-1.
8. `helixc/frontend/parser.py` lines 1582-1592 — the only known
   pre-prefixed FileNotFoundError producer (strict-stdlib).

**Method**: (a) Re-read cycles 6, 7, 8, 9, 10 codereview docs
to load the cumulative invariant set and below-threshold
concern list. (b) Confirmed via `git log` that c2e36d4 is HEAD
and no commits have landed since cycle 10. (c) Confirmed via
`git diff HEAD~1 HEAD --stat` that the cycle-10 -> cycle-11
state-change is doc additions + 65 lines of test additions
only; no production-code touch. (d) Walked the cycle-9
production code (`_emit_env_error` + cascade rework) and the
cycle-10 test triple again with fresh eyes, looking for
anything that prior cycles classified at 50-75 confidence
that should now be promoted given a second pass. (e) Ran the
three cycle-10 tests + the two cycle-5 C4-6 siblings at HEAD
to confirm all 5 pass cleanly (`5 passed in 0.64s`).

**Reporting threshold**: confidence >= 80 (per the cycle-11
audit-C prompt's strict criterion).

**Result**: **0 findings (0 CRITICAL, 0 HIGH, 0 MEDIUM, 0 LOW)
at or above the confidence-80 reporting threshold.**

---

## Summary table

| ID    | Severity | Confidence | Component | Issue |
|-------|----------|------------|-----------|-------|
| —     | —        | —          | —         | (none at or above threshold) |

**Cycle 11 Audit C: CLEAN — 0 findings at the confidence-80
threshold.**

Per user directive 2026-05-10 (strict criterion): cycle counts
CLEAN only when zero findings of ANY severity at or above the
audit threshold. **This cycle qualifies as clean.**

---

## Stability verification (no new commits since cycle 10)

`git log --oneline -5` at audit start:

```
c2e36d4 Audit 28.8 cycle 10: regression tests for C8-1 + C8-2 (close C9-1 LOW)
6968755 Audit 28.8 cycle 9 fix-sweep: close C8-1 + C8-2 (check.py exception classifier)
68bdb7f Persist cycle-8 codereview audit doc (partial state, at 5d1ca24)
e0a04a4 Audit 28.8 cycle 5 C4-6 / MEDIUM: regression tests for exception classifier
0a120b7 Audit 28.8 cycle 5 C4-7 / F6: regression test for cast ref prefix
```

HEAD is c2e36d4 — identical to the HEAD audited by cycle 10.
No production code has changed; no tests have been added or
removed; no docs have been edited; the worktree is clean
(only `docs/audit-stage28-8-cycle8-codereview-rev.md` shows as
deleted but that was an intentional rev-deletion from cycle 9
landing the canonical cycle-9 audit doc set, not a regression
risk). The cycle-11 audit therefore reduces to a fresh-eyes
re-evaluation of the cycle-9 + cycle-10 surface area, with
particular attention to whether any concern previously rated
50-75 should be promoted to >= 80 given a second independent
look.

---

## Fresh-eyes re-evaluation of cycle 9 + cycle 10 surface area

### 1. `_emit_env_error` helper (check.py:246-256) — PASS

```python
def _emit_env_error(msg: str) -> None:
    text = msg
    if text.lstrip().startswith("helixc:"):
        text = text.lstrip()[len("helixc:"):].lstrip()
    print(f"helixc: {text}", file=sys.stderr)
```

Re-walked the strip logic against the eight shapes probed in
the cycle-9 silent-failures audit:

- `"helixc: foo"` (space-prefixed, canonical parser shape) —>
  `.lstrip()` no-op, `.startswith("helixc:")` true, slice [7:] is
  `" foo"`, `.lstrip()` -> `"foo"`, emit `"helixc: foo"`. Single
  prefix.
- `"helixc:foo"` (no space) -> slice [7:] is `"foo"`, `.lstrip()`
  no-op, emit `"helixc: foo"`. Adds a canonicalising space.
  Cosmetic.
- `"  helixc: foo"` (leading whitespace) -> `.lstrip()` ->
  `"helixc: foo"`, then strip as above -> emit `"helixc: foo"`.
  Single prefix.
- `"helixc: helixc: foo"` (degenerate double-pre-prefix) -> strip
  once -> `"helixc: foo"`, emit -> `"helixc: helixc: foo"`.
  Strip is single-shot. No known callee produces this shape;
  the only pre-prefixer is parser.py:1587 which prefixes
  exactly once.
- `"HELIXC: foo"` (uppercase) -> startswith check is
  case-sensitive, no strip -> emit `"helixc: HELIXC: foo"`. No
  known callee emits uppercase; cosmetic at worst.
- `""` (empty) -> startswith false, emit `"helixc: "`. Trivial.
- `"helixc:"` (bare prefix, no content) -> slice [7:] is `""`,
  `.lstrip()` -> `""`, emit `"helixc: "`. Trivial.
- `"foo helixc: bar"` (embedded mid-string) -> startswith false
  (anchored at start), no strip -> emit
  `"helixc: foo helixc: bar"`. Correct: the embedded substring
  is preserved verbatim.

All eight shapes behave consistently and defensibly. Cycle 9
classified the degenerate / uppercase / no-space cases at
confidence 25-30 (cosmetic, unreachable in production). Fresh
re-evaluation confirms: none are reachable through any
production callee, and even if a future callee produced one,
the helper would degrade gracefully (one cosmetic
single-character difference at most). **No promotion warranted.
Confidence remains <= 30.**

### 2. Cascade rework (check.py:299-318) — PASS

Re-walked the post-cycle-9 cascade against every reachable
exception class from `_main_inner`:

- `OSError` subclasses caught explicitly: FileNotFoundError,
  PermissionError, IsADirectoryError, NotADirectoryError. All
  four route through `_emit_env_error(str(e))` with rc=2. The
  cycle-8 codereview rated below-threshold concern "OSError
  edge cases not in the explicit set" (BlockingIOError,
  BrokenPipeError, ConnectionError family, TimeoutError, raw
  OSError for ENOSPC / EIO) at confidence 55. Fresh
  re-evaluation: these classes are vanishingly rare in a
  single-file compile pipeline that does no networking, no
  subprocess, and only reads from `path`. The current
  4-class set covers the dominant user-env-error population.
  **No promotion warranted. Confidence remains 55.**

- `UnicodeDecodeError` -> `_emit_env_error(f"encoding error
  reading source: {e}")` with rc=2. The helper sees no
  `helixc:` prefix to strip (the format-string prepends a
  caller-controlled prefix that doesn't start with `helixc:`),
  so the strip branch is unreachable for this path. Single
  prefix guaranteed.

- `Exception` (broad) -> rc=1 + "this is a compiler bug — please
  file an issue." This now catches ImportError per the cycle-9
  drop. Verified `issubclass(ImportError, Exception) == True`
  and ImportError is not an OSError subclass, so the broad arm
  is the matching arm. Re-verified at HEAD by direct Python
  invocation.

The cascade ordering is `OSError-family` first, `UnicodeDecodeError`
second, `Exception` last. UnicodeDecodeError is a subclass of
ValueError (via UnicodeError), not of OSError, so cascade order
is correct. No subclass-shadowing concern.

### 3. Cycle-10 regression tests (test_typecheck.py:1572-1634) — PASS

Re-ran all three cycle-10 tests + both cycle-5 C4-6 siblings at
HEAD: 5 passed in 0.64s. Re-confirmed:

- `test_c8_1_import_error_attributed_as_compiler_bug` — asserts
  rc=1, "compiler bug" substring, "internal error" substring.
  Pins the post-cycle-9 ImportError -> broad-Exception contract.
- `test_c8_2_env_error_no_double_helixc_prefix` — asserts rc=2,
  "helixc: helixc:" absent, "stdlib file missing" present.
  Pins the C8-2 strip-prefix invariant for pre-prefixed
  callees (parser.py:1587 shape).
- `test_c8_2_env_error_no_prefix_still_prefixed` — asserts
  rc=2, `count("helixc:") == 1`. Pins the strip helper's
  no-prefix branch (single prefix-append, no double-print).

The cycle-10 codereview rated three observability gaps below
threshold (no pathological-shape coverage at conf 35; no
end-to-end strict-stdlib coverage at conf 30; no
UnicodeDecodeError single-prefix coverage at conf 25). Fresh
re-evaluation: all three remain cosmetic test-density
concerns rather than correctness defects. The cycle-5
UnicodeDecodeError test still passes post-cycle-9, confirming
the helper-routed path works correctly for that class. The
strict-stdlib path is tested indirectly via the
monkeypatched FileNotFoundError with the exact same message
shape (`parser.py:1587`'s pre-formatted prefix). **No
promotions warranted. All three remain below threshold.**

### 4. Monkeypatch target binding — PASS

The cycle-10 tests use `monkeypatch.setattr(check_mod,
"typecheck", boom)`. For the patch to genuinely intercept, the
production call site must resolve `typecheck` through the
module-level attribute, not via a local rebind.

- `check.py:58` — `from .frontend.typecheck import typecheck`
  binds `typecheck` as a module-level attribute of `check`.
- `check.py:403` — `tc_errs = typecheck(prog)` resolves
  `typecheck` via the LEGB rule: no local binding (the
  function has no `typecheck = ...` assignment), no enclosing
  scope, module-global hits at `check.typecheck`, which is
  exactly the attribute monkeypatch sets.

Verified at HEAD: no `typecheck = ` assignment exists inside
`_main_inner` or any helper between line 58 and line 403. The
monkeypatch target is correct and the interception is genuine.

### 5. Drain finally invariant (check.py:319-338) — PASS

The cycle-5 try/except inside `finally` is preserved byte-for-
byte. Verified no cycle-9 / cycle-10 touch by re-reading the
block. Drain still runs on every exit path (including the new
ImportError -> broad-Exception path); drain failures still
wrap in their own `try/except Exception as drain_e` and emit
a `warning:` notice without altering rc. No regression.

### 6. parser.py:1582-1592 (pre-prefixed FileNotFoundError) — PASS

Re-read the strict-stdlib branch:

```python
msg = f"helixc: stdlib file missing: {stdlib_path}"
if strict:
    raise FileNotFoundError(msg)
print(msg, file=_sys.stderr)
continue
```

The `raise FileNotFoundError(msg)` produces an exception whose
`str(e)` is exactly `"helixc: stdlib file missing: <path>"` —
the canonical input shape the cycle-9 helper is designed to
handle. The non-strict branch `print(msg, ...)` does NOT raise
and is not in the outer-cascade scope. **Confirmed: only one
pre-prefixed producer in the codebase, exactly the shape the
strip helper handles.**

---

## Below-threshold re-evaluation (cycles 6-10 carryover)

Walked the below-threshold concern list from each prior
codereview cycle to check whether fresh eyes would now
classify any at >= 80:

**Cycle 6 below-threshold** (test additions for cycle-5 C5-1
TRAP constants; monomorphize_safe docstring drift): docstring
drift remains a deferred housekeeping item, conf 40, not
promotable. No promotion.

**Cycle 7 below-threshold** (D-vs-Quote diagnostic text):
remains a cosmetic message-quality concern, conf 30, not
promotable. No promotion.

**Cycle 8 below-threshold** (C7-1 close test-coverage gap for
`_compatible(TyMemTier, TyVar)` / `_compatible(TyMemTier,
TySize)`): cycle 10 did not add these tests. The gap remains
at conf 55. Not promotable to >= 80 because the production
fix (cycle 8) is correct on its own; absence of tests is a
density concern, not a correctness defect. No promotion.

**Cycle 9 below-threshold** (no regression tests for C8-1 /
C8-2; pathological double-pre-prefix; OSError edge cases):
cycle 10 closed the test-coverage half explicitly (which is
how C9-1 was opened then closed). Pathological + OSError items
remain at conf 25-55, not promotable. No promotion.

**Cycle 10 below-threshold** (pathological strip-shape
coverage; end-to-end strict-stdlib coverage; UnicodeDecodeError
single-prefix coverage; test ordering; `*_args, **_kw`
signature; no-prefix-branch coverage; commit-message hygiene):
all rated conf 10-35. Fresh eyes: none rise above conf 35.
No promotion.

---

## Open prior findings (not addressed this cycle)

Per the cumulative carryover from cycle 10:

- **audit-C4-1** (CRITICAL — D2 Call-RHS i32 SIGILL): still
  open; deferred pending parse-time constant folding or a
  typecheck pass before closure capture. Not addressed in
  cycle 11. Out of scope for an audit-C stability cycle.
  **Highest-priority still-open carryover.**
- **audit-C4-4** (HIGH — D9 paper-only): still open; not
  addressed in cycle 11. Out of scope.
- **audit-C4-8 deferred** (LOW — check.py doesn't call
  fn-mono): still open; not addressed in cycle 11. Out of
  scope.
- **monomorphize_safe docstring drift** (cycle-6 deferred):
  still open; not addressed. Out of scope.
- **D-vs-Quote diagnostic text** (cycle-7 deferred): still
  open; not addressed. Out of scope.
- **C7-1 close test-coverage gap** (cycle-8 housekeeping
  candidate — `_compatible(TyMemTier, TyVar) is False`
  regression test): still open; not addressed. Out of scope.

No code-review regressions introduced by cycle 11 (no commits
since cycle 10):

- `check.py` byte-for-byte unchanged from cycle-10 HEAD.
- `parser.py` byte-for-byte unchanged.
- `test_typecheck.py` byte-for-byte unchanged.
- No new dependencies or fixtures.
- All cycle-1 through cycle-10 production-code contracts
  preserved verbatim.
- Cycle-10 + cycle-5 exception-classifier tests pass at HEAD
  (5 passed in 0.64s).

Pre-cycle-1 audit-stage5-6, audit-stage7-8, audit-stage9-16
baselines unchanged from cycle-1 status.

---

## Verdict

**Cycle 11 Audit C: CLEAN — 0 findings (0 CRITICAL, 0 HIGH,
0 MEDIUM, 0 LOW) at or above the confidence-80 reporting
threshold.**

Strict-zero rule per user directive 2026-05-10. Cycle counter
advances provided cycles A (silent-failure) and B (type-
design) at this commit are also clean.

This is a stability re-verification cycle: HEAD is unchanged
from cycle 10. Fresh-eyes pass over cycle-9 + cycle-10 surface
area confirms the post-cycle-9 production code (cascade rework
+ `_emit_env_error` helper) is correct against every reachable
exception class, the monkeypatch interception in the cycle-10
tests is genuine (module-level `typecheck` binding at
check.py:58 + lookup at line 403, no local rebind), the strip
helper degrades gracefully on all eight pathological shapes,
the drain finally invariant is preserved, and the cycle-10
tests pin all three observable post-cycle-9 contracts (C8-1
ImportError -> rc=1 + compiler-bug; C8-2 already-prefixed
-> single prefix; C8-2 unprefixed -> single prefix added).
Re-evaluation of every below-threshold concern from cycles 6
through 10 produced no promotions to >= 80.

If cycles A (silent-failures) and B (type-design) at c2e36d4
are also clean, this is the **second clean cycle in a row**
at this HEAD (counting cycle 10 as the first); cycle-counter
advances from 1 to 2 in the 5-clean-cycles deprecation gate.
