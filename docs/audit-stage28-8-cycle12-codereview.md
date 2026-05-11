# Stage 28.8 Pre-29 Audit Gate — Cycle 12, Audit C: Code Review

**Date**: 2026-05-11
**Commit**: df825ac (read-only)
**Scope**: Audit C (general code-review) at commit df825ac —
**stability re-verification cycle 2**. No new commits since
cycle 11 (df825ac itself is the cycle-11 audit-doc persist
commit; the production HEAD audited by both cycles 10 and 11
remains c2e36d4 at the source-code level — df825ac touches
only `docs/audit-stage28-8-cycle11-*.md`). This cycle performs
a third independent fresh-eyes pass over the cycle-9 +
cycle-10 surface area (the only state-change since cycle-8's
clean baseline at 5d1ca24) and re-evaluates every prior
below-threshold concern from cycles 6 / 7 / 8 / 9 / 10 / 11
to confirm none should now be promoted to >= 80 confidence.

**Cycle-counter status**: prior cycles 10 and 11 were both
CLEAN, advancing the gate from 0 -> 1 -> 2 toward the 5-clean-
cycles deprecation gate. This cycle (12), if clean, advances
2 -> 3 (subject to corresponding cycles A and B at this HEAD
also returning CLEAN).

Files re-reviewed at HEAD (df825ac):

1. `helixc/check.py` lines 54-58 — module-level imports,
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
   cycle-10 regression tests that close C9-1.
8. `helixc/frontend/parser.py` lines 1582-1592 — the only known
   pre-prefixed FileNotFoundError producer (strict-stdlib).

**Method**:

(a) Read prior codereview docs (cycles 6 / 7 / 8 / 8-rev / 9 /
10 / 11) to load the cumulative invariant set and below-
threshold concern list.

(b) Confirmed via `git log --oneline -20` that df825ac is HEAD
and the only commit since cycle 10 (c2e36d4) is the cycle-11
audit-doc persist (df825ac), which adds three documentation
files only.

(c) Confirmed via `git show --stat df825ac` that the cycle-11
persist touches exclusively:
- `docs/audit-stage28-8-cycle11-codereview.md` (+372)
- `docs/audit-stage28-8-cycle11-silent-failures.md` (+469)
- `docs/audit-stage28-8-cycle11-type-design.md` (+343)
i.e., +1184 lines across 3 documentation files, 0 production
code lines, 0 test lines.

(d) Confirmed via `git diff --stat c2e36d4..df825ac` that no
production source file (`helixc/check.py`, `helixc/frontend/`,
`helixc/ir/`, `helixc/backend/`, `helixc/lib/`) and no test
file is touched at all by df825ac.

(e) Confirmed via `git diff --stat 6968755..df825ac --
':!docs' ':!helixc/tests/'` that since cycle-9 fix HEAD
(6968755) every production file under `helixc/**.py` is byte-
for-byte unchanged. The only non-doc change in the cycle 10
-> 12 window was the +65 line test-file append in c2e36d4
(cycle-10 fix-sweep). df825ac added zero production or test
lines.

(f) Walked the cycle-9 production code (`_emit_env_error`
helper + cascade rework at check.py:246-318) and the cycle-10
test triple (test_typecheck.py:1572-1634) again with fresh
eyes, looking for anything that prior cycles classified at
50-75 confidence that should now be promoted given a third
independent pass.

(g) Ran the three cycle-10 tests + the two cycle-5 C4-6 sibling
tests at HEAD via `python -m pytest ...test_c8_1... ...test_c8_2_no_double...
...test_c8_2_no_prefix... ...test_c4_6_filenotfound...
...test_c4_6_unicode_decode... -v` and confirmed
`5 passed in 0.86s` cleanly.

**Reporting threshold**: confidence >= 80 (per the cycle-12
audit-C prompt's strict criterion).

**Result**: **0 findings (0 CRITICAL, 0 HIGH, 0 MEDIUM, 0 LOW)
at or above the confidence-80 reporting threshold.**

---

## Summary table

| ID    | Severity | Confidence | Component | Issue |
|-------|----------|------------|-----------|-------|
| —     | —        | —          | —         | (none at or above threshold) |

**Cycle 12 Audit C: CLEAN — 0 findings at the confidence-80
threshold.**

Per user directive 2026-05-10 (strict criterion): cycle counts
CLEAN only when zero findings of ANY severity at or above the
audit threshold. **This cycle qualifies as clean.**

---

## Stability verification (no production-code change since cycle 10)

`git log --oneline -5` at audit start:

```
df825ac Audit 28.8 cycle 11: persist 3 cycle-11 CLEAN audit docs
9685c3a Persist cycle-10 audit docs (silent-failures + type-design + codereview)
c2e36d4 Audit 28.8 cycle 10: regression tests for C8-1 + C8-2 (close C9-1 LOW)
6968755 Audit 28.8 cycle 9 fix-sweep: close C8-1 + C8-2 (check.py exception classifier)
68bdb7f Persist cycle-8 codereview audit doc (partial state, at 5d1ca24)
```

HEAD is df825ac. The cycle 10 -> 11 -> 12 window introduced
6 documentation files totaling +2603 lines (the three cycle-10
audit docs at 9685c3a + the three cycle-11 audit docs at
df825ac) and the +65 line test-file append at c2e36d4. Zero
production-code touches. The worktree is clean.

The cycle-12 audit therefore reduces to a fresh-eyes
re-evaluation of:

- The cycle-9 production code (cascade rework + `_emit_env_error`
  helper) — third independent pass.
- The cycle-10 regression tests (3 added at test_typecheck.py:
  1572-1634) — third independent pass.
- Every below-threshold concern from cycles 6 / 7 / 8 / 9 / 10 /
  11 — fresh check for promotability.

---

## Fresh-eyes re-evaluation of cycle 9 + cycle 10 surface area

### 1. `_emit_env_error` helper (check.py:246-256) — PASS

```python
def _emit_env_error(msg: str) -> None:
    """Audit 28.8 cycle 9 C8-2: print a user-environment error with a
    single `helixc:` prefix. ..."""
    text = msg
    if text.lstrip().startswith("helixc:"):
        text = text.lstrip()[len("helixc:"):].lstrip()
    print(f"helixc: {text}", file=sys.stderr)
```

Re-walked the strip logic against the eight shapes probed in
the cycle-9 silent-failures audit:

- `"helixc: foo"` (space-prefixed, canonical parser shape) ->
  `.lstrip()` no-op, `.startswith("helixc:")` true, slice [7:]
  is `" foo"`, `.lstrip()` -> `"foo"`, emit `"helixc: foo"`.
  Single prefix. **Canonical path; load-bearing.**
- `"helixc:foo"` (no space after colon) -> slice [7:] is
  `"foo"`, `.lstrip()` no-op, emit `"helixc: foo"`. Adds a
  canonicalising space. Cosmetic.
- `"  helixc: foo"` (leading whitespace) -> `.lstrip()` ->
  `"helixc: foo"`, then strip as above -> emit `"helixc: foo"`.
  Single prefix.
- `"helixc: helixc: foo"` (degenerate double-pre-prefix) ->
  strip once -> `"helixc: foo"`, emit -> `"helixc: helixc:
  foo"`. Strip is single-shot. No known callee produces this
  shape; the only pre-prefixer is parser.py:1587 which
  prefixes exactly once.
- `"HELIXC: foo"` (uppercase) -> startswith check is
  case-sensitive, no strip -> emit `"helixc: HELIXC: foo"`.
  No known callee emits uppercase; cosmetic at worst.
- `""` (empty) -> startswith false, emit `"helixc: "`. Trivial.
- `"helixc:"` (bare prefix, no content) -> slice [7:] is `""`,
  `.lstrip()` -> `""`, emit `"helixc: "`. Trivial.
- `"foo helixc: bar"` (embedded mid-string) -> startswith
  false (anchored at start), no strip -> emit `"helixc: foo
  helixc: bar"`. Correct: the embedded substring is preserved
  verbatim.

All eight shapes behave consistently and defensibly. Third-pass
re-evaluation confirms the cycles-9-and-11 ratings: none of the
degenerate / uppercase / no-space cases are reachable through
any production callee, and even if a future callee produced
one, the helper would degrade gracefully with at most a
single-character cosmetic difference. **No promotion warranted.
Confidence remains <= 30.**

### 2. Cascade rework (check.py:299-318) — PASS

Re-walked the post-cycle-9 cascade against every reachable
exception class from `_main_inner`:

- `(FileNotFoundError, PermissionError, IsADirectoryError,
  NotADirectoryError)` -> `_emit_env_error(str(e))` with rc=2.
  Pre-existing cycle-8 below-threshold concern: "OSError edge
  cases not in the explicit set (BlockingIOError, BrokenPipeError,
  ConnectionError family, TimeoutError, raw OSError for ENOSPC /
  EIO)" at confidence 55. Third-pass re-evaluation: these classes
  are vanishingly rare in a single-file compile pipeline that
  does no networking, no subprocess, and only reads from `path`.
  The current 4-class set covers the dominant user-env-error
  population. **No promotion warranted. Confidence remains 55.**

- `UnicodeDecodeError` -> `_emit_env_error(f"encoding error
  reading source: {e}")` with rc=2. The helper sees no
  `helixc:` prefix to strip (the format-string prepends a
  caller-controlled prefix that doesn't start with `helixc:`),
  so the strip branch is unreachable for this path. Single
  prefix guaranteed by construction.

- `Exception` (broad) -> rc=1 + "this is a compiler bug —
  please file an issue." This now catches `ImportError` per the
  cycle-9 drop. Re-verified at HEAD via direct Python invocation
  (`issubclass(ImportError, Exception) == True`, `issubclass(
  ImportError, OSError) == False`), so the broad arm is the
  matching arm for ImportError. The cycle-10 regression test
  `test_c8_1_import_error_attributed_as_compiler_bug` pins this
  contract.

Cascade ordering is `OSError-family` first, `UnicodeDecodeError`
second, `Exception` last. `UnicodeDecodeError` is a subclass of
`ValueError` (via `UnicodeError`), not of `OSError`, so cascade
order is correct. `ImportError` is a direct subclass of
`Exception` and not of any other arm. No subclass-shadowing
concern.

### 3. Cycle-10 regression tests (test_typecheck.py:1572-1634) — PASS

Re-ran all three cycle-10 tests + both cycle-5 C4-6 siblings at
HEAD df825ac:

```
helixc/tests/test_typecheck.py::test_c8_1_import_error_attributed_as_compiler_bug PASSED
helixc/tests/test_typecheck.py::test_c8_2_env_error_no_double_helixc_prefix PASSED
helixc/tests/test_typecheck.py::test_c8_2_env_error_no_prefix_still_prefixed PASSED
helixc/tests/test_typecheck.py::test_c4_6_filenotfound_not_attributed_as_compiler_bug PASSED
helixc/tests/test_typecheck.py::test_c4_6_unicode_decode_error_clean_message PASSED
============================== 5 passed in 0.86s ==============================
```

Re-confirmed:

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

The cycle-10 + cycle-11 codereviews rated three observability
gaps below threshold (no pathological-shape coverage at conf
35; no end-to-end strict-stdlib coverage at conf 30; no
UnicodeDecodeError single-prefix coverage at conf 25). Third-
pass re-evaluation: all three remain cosmetic test-density
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
byte. Verified no cycle-9 / cycle-10 / cycle-11 / cycle-12
touch by re-reading the block. Drain still runs on every exit
path (including the new ImportError -> broad-Exception path);
drain failures still wrap in their own `try/except Exception
as drain_e` and emit a `helixc: warning:` notice without
altering rc. No regression.

### 6. parser.py:1582-1592 (pre-prefixed FileNotFoundError) — PASS

Re-read the strict-stdlib branch:

```python
for fname in STDLIB_FILES:
    stdlib_path = _os.path.join(stdlib_dir, fname)
    if not _os.path.isfile(stdlib_path):
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

## Below-threshold re-evaluation (cycles 6-11 carryover)

Walked the below-threshold concern list from each prior
codereview cycle to check whether fresh eyes (third pass for
cycle-9 / cycle-10 surface area, second-or-later pass for
older items) would now classify any at >= 80:

**Cycle 6 below-threshold** (test additions for cycle-5 C5-1
TRAP constants; monomorphize_safe docstring drift): docstring
drift remains a deferred housekeeping item, conf 40, not
promotable. No promotion.

**Cycle 7 below-threshold** (D-vs-Quote diagnostic text):
remains a cosmetic message-quality concern, conf 30, not
promotable. No promotion.

**Cycle 8 below-threshold** (C7-1 close test-coverage gap for
`_compatible(TyMemTier, TyVar)` / `_compatible(TyMemTier,
TySize)`): cycles 10, 11 did not add these tests. The gap
remains at conf 55. Not promotable to >= 80 because the
production fix (cycle 8) is correct on its own; absence of
tests is a density concern, not a correctness defect. No
promotion.

**Cycle 9 below-threshold** (no regression tests for C8-1 /
C8-2; pathological double-pre-prefix; OSError edge cases):
cycle 10 closed the test-coverage half explicitly (which is
how C9-1 was opened then closed). Pathological + OSError items
remain at conf 25-55, not promotable. No promotion.

**Cycle 10 below-threshold** (pathological strip-shape
coverage; end-to-end strict-stdlib coverage; UnicodeDecodeError
single-prefix coverage; test ordering; `*_args, **_kw`
signature; no-prefix-branch coverage; commit-message hygiene):
all rated conf 10-35. Third-pass eyes: none rise above conf 35.
No promotion.

**Cycle 11 below-threshold** (same set as cycle 10, re-rated
identically): all remain conf 10-55. No promotion.

---

## Open prior findings (not addressed this cycle)

Per the cumulative carryover from cycle 11:

- **audit-C4-1** (CRITICAL — D2 Call-RHS i32 SIGILL): still
  open; deferred pending parse-time constant folding or a
  typecheck pass before closure capture. Not addressed in
  cycle 12. Out of scope for an audit-C stability cycle.
  **Highest-priority still-open carryover.**
- **audit-C4-4** (HIGH — D9 paper-only): still open; not
  addressed in cycle 12. Out of scope.
- **audit-C4-8 deferred** (LOW — check.py doesn't call
  fn-mono): still open; not addressed in cycle 12. Out of
  scope.
- **monomorphize_safe docstring drift** (cycle-6 deferred):
  still open; not addressed. Out of scope.
- **D-vs-Quote diagnostic text** (cycle-7 deferred): still
  open; not addressed. Out of scope.
- **C7-1 close test-coverage gap** (cycle-8 housekeeping
  candidate — `_compatible(TyMemTier, TyVar) is False`
  regression test): still open; not addressed. Out of scope.

No code-review regressions introduced by cycle 12 (no commits
since cycle 11's doc-only persist; no production-code change
since cycle 9 at 6968755):

- `check.py` byte-for-byte unchanged from cycle-10 HEAD.
- `parser.py` byte-for-byte unchanged.
- `test_typecheck.py` byte-for-byte unchanged.
- No new dependencies or fixtures.
- All cycle-1 through cycle-10 production-code contracts
  preserved verbatim.
- Cycle-10 + cycle-5 exception-classifier tests pass at HEAD
  (5 passed in 0.86s).

Pre-cycle-1 audit-stage5-6, audit-stage7-8, audit-stage9-16
baselines unchanged from cycle-1 status.

---

## Verdict

**Cycle 12 Audit C: CLEAN — 0 findings (0 CRITICAL, 0 HIGH,
0 MEDIUM, 0 LOW) at or above the confidence-80 reporting
threshold.**

Strict-zero rule per user directive 2026-05-10. Cycle counter
advances provided cycles A (silent-failure) and B (type-
design) at this HEAD are also clean.

This is the second consecutive stability re-verification cycle:
production source HEAD (`helixc/**.py`) is byte-for-byte
unchanged from cycle 10's HEAD c2e36d4, and df825ac's only
diff is +1184 lines across three cycle-11 audit-doc files
(documentation, out of audit-C scope). Third-pass fresh-eyes
review over cycle-9 + cycle-10 surface area confirms the
post-cycle-9 production code (cascade rework + `_emit_env_error`
helper) is correct against every reachable exception class,
the monkeypatch interception in the cycle-10 tests is genuine
(module-level `typecheck` binding at check.py:58 + lookup at
line 403, no local rebind), the strip helper degrades
gracefully on all eight pathological shapes, the drain finally
invariant is preserved, and the cycle-10 tests pin all three
observable post-cycle-9 contracts (C8-1 ImportError -> rc=1 +
compiler-bug; C8-2 already-prefixed -> single prefix; C8-2
unprefixed -> single prefix added). Re-evaluation of every
below-threshold concern from cycles 6 through 11 produced no
promotions to >= 80.

If cycles A (silent-failures) and B (type-design) at df825ac
are also clean, this is the **third clean cycle in a row**
at this production-code HEAD (counting cycle 10 as the first
and cycle 11 as the second); cycle-counter advances from 2 to
3 in the 5-clean-cycles deprecation gate.
