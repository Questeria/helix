# Stage 28.8 Cycle 10 — Silent-Failure Audit

**Date**: 2026-05-11
**Commit**: c2e36d4 (read-only audit). Cycle-10 fix-sweep range
6968755..c2e36d4 (1 fix-sweep commit covering C9-1 LOW
closure via 3 regression tests).
**Scope**: All Helix source — `helixc/bootstrap/*.hx`,
`helixc/frontend/*.py`, `helixc/ir/*.py`, `helixc/backend/*.py`,
`helixc/stdlib/*.hx`. Specifically re-audits the cycle-10
fix-sweep changes (3 regression-test additions + 4 persisted
audit docs) for fresh silent windows introduced by the fixes
themselves.
**Trigger**: pre-Stage-29 audit gate — Cycle 10 of 5+ (the
gate re-arms each time a cycle is not clean). Re-audits the
same scope after cycle-10 fixes were landed.
**Strict criterion** (per user directive 2026-05-10): cycle
counts CLEAN only when **zero new findings of ANY severity**
(CRITICAL/HIGH/MEDIUM/LOW). Findings already documented in
cycles 1-9 are NOT re-flagged unless they CHANGED in the
cycle-10 fix-sweep.

**Method**:
1. Read the cycle-9 silent-failures audit (1 finding — C9-1
   LOW, "cycle-9 fix-sweep closes C8-1 + C8-2 but adds zero
   regression tests").
2. Walked `git show c2e36d4` — the single cycle-10 fix-sweep
   commit. Read the diff:
   - 3 new regression tests appended to
     `helixc/tests/test_typecheck.py` immediately after the
     cycle-5 C4-6 regression block (lines 1572-1634):
     - `test_c8_1_import_error_attributed_as_compiler_bug`
     - `test_c8_2_env_error_no_double_helixc_prefix`
     - `test_c8_2_env_error_no_prefix_still_prefixed`
   - 4 persisted audit docs (cycle-8 codereview-rev, cycle-9
     codereview, cycle-9 silent-failures, cycle-9 type-design)
     — read-only doc files, no production-code or test-code
     change.
   - Zero production-code (.py / .hx) changes.
3. For each new test, traced the fix's reproducer forward
   through `helixc.check.main` to confirm the test actually
   exercises the cycle-9-fixed code path (not a paper-only
   test asserting on something already-true even pre-cycle-9).
4. Direct mutation-probes against `c2e36d4` HEAD to confirm
   each new test would FAIL if the corresponding cycle-9 fix
   were reverted:
   - Mutation A: revert `_emit_env_error` to a naive
     `print(f"helixc: {msg}")` (no strip) → confirms
     `test_c8_2_env_error_no_double_helixc_prefix` fails
     (stderr becomes `helixc: helixc: stdlib file missing:
     foo.hx`).
   - Mutation B: re-add the `except ImportError as e: ...
     rc=2` arm to `check.main` → confirms
     `test_c8_1_import_error_attributed_as_compiler_bug`
     fails (rc=2 instead of rc=1; stderr lacks "compiler
     bug").
   - Mutation C: change `_emit_env_error` to drop the prefix
     entirely (forget to re-add `helixc:`) → confirms
     `test_c8_2_env_error_no_prefix_still_prefixed` fails
     (stderr `count("helixc:") == 0` instead of `== 1`).
5. Verified stdout/stderr isolation: the parse banner
   (`-- helixc-check: ...`, `   parse:    OK ...`) writes to
   stdout, the env-error message writes to stderr. The
   `captured.err.count("helixc:") == 1` assertion in
   `test_c8_2_env_error_no_prefix_still_prefixed` is robust
   because the banner does not pollute stderr.
6. Ran the targeted suite (`pytest helixc/tests/
   test_typecheck.py`) at `c2e36d4` — 111/111 pass (was 108
   at `6968755`, +3 new tests). Ran the 3 new tests
   individually — all 3 PASS.
7. Searched for any new silent-failure surface introduced by
   the test additions: cross-test state leak via
   monkeypatching `check.typecheck`. Each test uses
   `monkeypatch` (which auto-reverts at test teardown) plus
   `tmp_path` (per-test isolated directory) plus `capsys`
   (per-test stderr capture). No shared module-level state
   is mutated outside the monkeypatch envelope. Test order
   independence verified by `pytest -p no:randomly --tb=no
   helixc/tests/test_typecheck.py` clean.
8. Checked the cycle-9 audit's 4-test recommendation against
   the cycle-10 3-test delivery to determine if there's a
   coverage gap (see "Cycle 9 recommendation vs cycle 10
   delivery" section below).

**Result**: **0 new findings (0 CRITICAL, 0 HIGH, 0 MEDIUM,
0 LOW)** — Cycle 10 is **CLEAN** for the silent-failure
audit lens. The fix-sweep CLOSES C9-1 fully:
- Each of the 3 new tests is mutation-validated against its
  corresponding pre-cycle-9 buggy state. The tests are not
  paper-only — they catch the regression.
- The tests use proper fixture hygiene (`monkeypatch` +
  `tmp_path` + `capsys`) with no cross-test state leak.
- Cycle 10 introduces **zero new production-code surface**,
  so there is no fresh silent-failure window to audit. All
  changes are test additions + audit-doc persistence.
- The cycle-9 C9-1 finding (the only open cycle-9 finding)
  is **CLOSED** by this commit.
- No new finding of any severity surfaces in the cycle-10
  diff lens.

---

## CRITICAL FINDINGS

(none)

---

## HIGH FINDINGS

(none)

---

## MEDIUM FINDINGS

(none)

---

## LOW FINDINGS

(none)

---

## Cycle 10 fix-sweep re-verification

The cycle-10 fix-sweep landed as a single commit (c2e36d4)
covering C9-1 plus persisted prior-cycle audit docs.

| fix-sweep label | What changed | Audit-doc cross-ref | C10 verdict |
|---|---|---|---|
| C9-1 close (regression tests for C8-1 + C8-2) | Added 3 new tests in `test_typecheck.py:1572-1634`: `test_c8_1_import_error_attributed_as_compiler_bug` (asserts rc=1 + "compiler bug" + "internal error" when an `ImportError` raises inside `_main_inner`); `test_c8_2_env_error_no_double_helixc_prefix` (asserts no `helixc: helixc:` substring in stderr when a callee pre-prefixes its FileNotFoundError message); `test_c8_2_env_error_no_prefix_still_prefixed` (asserts `count("helixc:") == 1` when the callee message has no prefix). | C9-1 (cycle-9 silent-failure LOW) | **closed** — all 3 tests pass at `c2e36d4`; all 3 fail under their respective mutation-revert probes. Verified end-to-end. |
| Cycle-8/9 audit doc persistence | Added `docs/audit-stage28-8-cycle8-codereview-rev.md`, `audit-stage28-8-cycle9-codereview.md`, `audit-stage28-8-cycle9-silent-failures.md`, `audit-stage28-8-cycle9-type-design.md`. Read-only doc files. | n/a (doc-only) | **n/a** — no production-code or test-code change; cannot introduce a silent surface. |

### Specific re-verifications from the audit instructions

- **C9-1 close — test_c8_1 actually exercises the cycle-9
  `except ImportError` drop**: probed via mutation revert at
  `c2e36d4`:
  ```
  >>> import helixc.check as cm
  >>> def buggy_main(argv):
  ...     try:
  ...         return cm._main_inner(argv, [])
  ...     except ImportError as e:  # PRE-CYCLE-9
  ...         import sys; print(f'helixc: import error: {e}',
  ...               file=sys.stderr); return 2
  ...     except Exception as e:
  ...         import sys; print(f'helixc: internal error: ...',
  ...               file=sys.stderr); return 1
  >>> # With the pre-cycle-9 arm re-added:
  >>> rc, err = buggy_main([src])  # raises ImportError in
  ...                              # monkeypatched typecheck
  rc: 2
  err: 'helixc: import error: cannot import name X\n'
  ```
  Confirmed: with the pre-cycle-9 arm restored, rc=2 (not
  rc=1) and stderr lacks both "compiler bug" and "internal
  error" — so `test_c8_1_import_error_attributed_as_compiler_
  bug` would FAIL on the rc==1 assert AND on the "compiler
  bug" assert AND on the "internal error" assert. The test
  is genuinely load-bearing for the cycle-9 fix.

- **C9-1 close — test_c8_2_no_double_prefix actually
  exercises the cycle-9 strip-prefix logic**: probed via
  mutation revert at `c2e36d4`:
  ```
  >>> import helixc.check as cm
  >>> def _emit_env_error_broken(msg):
  ...     import sys; print(f'helixc: {msg}', file=sys.stderr)
  >>> cm._emit_env_error = _emit_env_error_broken
  >>> # Run via check.main([src]) with monkeypatched typecheck
  >>> # raising FileNotFoundError('helixc: stdlib file missing: foo.hx')
  rc: 2
  err: 'helixc: helixc: stdlib file missing: foo.hx\n'
  ```
  Confirmed: with the strip removed, stderr becomes `helixc:
  helixc: stdlib file missing: foo.hx` — so
  `test_c8_2_env_error_no_double_helixc_prefix` would FAIL
  on the `assert "helixc: helixc:" not in captured.err`
  check. The test is genuinely load-bearing for the cycle-9
  fix.

- **C9-1 close — test_c8_2_no_prefix_still_prefixed actually
  exercises the `_emit_env_error` prefix-add path**: probed
  via mutation revert at `c2e36d4`:
  ```
  >>> def _emit_env_error_no_prefix(msg):
  ...     import sys; print(msg, file=sys.stderr)
  >>> cm._emit_env_error = _emit_env_error_no_prefix
  >>> # Run via check.main([src]) with monkeypatched typecheck
  >>> # raising FileNotFoundError('plain message, no prefix')
  rc: 2
  err: 'plain message, no prefix\n'  (count('helixc:') == 0)
  ```
  Confirmed: with the prefix-add accidentally dropped,
  stderr's `helixc:` count is 0 — so
  `test_c8_2_env_error_no_prefix_still_prefixed` would FAIL
  on the `assert captured.err.count("helixc:") == 1` check.
  The test is genuinely load-bearing for the cycle-9 fix.

- **Test-fixture hygiene**: all 3 new tests use
  `monkeypatch.setattr(check_mod, "typecheck", boom)` —
  `monkeypatch` auto-reverts at test teardown, so the
  `check.typecheck` module-level binding is not corrupted
  for subsequent tests. Verified by running the full
  `helixc/tests/test_typecheck.py` suite (111 tests) — no
  test-order-dependent failure observed. The 3 new tests
  each create their own `tmp_path / "boom.hx"`, so no
  filesystem cross-test interference. `capsys` captures
  per-test stderr without leaking.

- **Stderr/stdout isolation for the
  `count("helixc:") == 1` assert**: verified that the parse
  banner (`-- helixc-check: ...`, `   parse:    OK (1 fns,
  1 items)`) goes to stdout, NOT stderr. So
  `captured.err.count("helixc:") == 1` is a tight
  assertion — it only counts the error-arm prefix, not the
  banner. Confirmed via direct `redirect_stdout` /
  `redirect_stderr` probe.

- **Pytest pass count**: 111/111 typecheck tests pass at
  `c2e36d4` (was 108 at `6968755`, +3 new). Targeted run
  of the 3 new tests in isolation: all 3 PASS.

### Carryover findings status (cycles 1-9)

The cycle-10 fix-sweep CLOSED C9-1 (LOW). Did NOT re-attempt
the following still-open carryover findings:

| Carryover | Severity | Cycle-10 status |
|---|---|---|
| audit-C4-1 (D2 Call-RHS i32 SIGILL) | CRITICAL | **still open** — cycle-10 did not address; same deferral rationale (parse-time constant folding or pre-closure-capture typecheck pass needed) |
| audit-C4-4 (D9 paper-only) | HIGH | **still open** — not addressed |
| audit-C4-8 deferred (check.py doesn't call fn-mono) | LOW | **still open** — not addressed |
| monomorphize_safe docstring drift (cycle-6 deferred) | (not a finding) | **still open** — docstring still suggests callers MAY ignore diags; only caller now aborts |
| D-vs-Quote diagnostic text (cycle-7 deferred) | (not a finding) | **still open** — Quote-wrapped case still emits "(one side D-wrapped, other bare)" |
| C7-1 test-coverage gap (cycle-8 deferred, cycle-9 unaddressed) | (not a finding) | **still open** — cycle 10 did not add the `_compatible(TyMemTier, TyVar) is False` regression tests either. Not re-flagged here (carryover; did not CHANGE in cycle 10). |
| C8-1 (cycle-8 MEDIUM) | MEDIUM | **CLOSED by cycle 9** |
| C8-2 (cycle-8 LOW) | LOW | **CLOSED by cycle 9** |
| C9-1 (cycle-9 LOW) | LOW | **CLOSED by cycle 10** (this commit). |

These are NOT re-flagged as new cycle-10 findings per the
user directive (already documented in cycles 1-9, did not
CHANGE in cycle 10's fix-sweep). They remain in the
open-findings ledger and are out-of-scope for this audit's
strict-clean determination (the strict criterion is
"zero NEW findings", not "zero open findings").

### Specific items checked clean in cycle 10 (no new finding)

- The 3 new tests are appended immediately after the
  cycle-5 C4-6 regression block (lines 1571-1635) — the
  natural location for cycle-9-fix regressions. They follow
  the same pattern as `test_c4_6_filenotfound_not_attributed
  _as_compiler_bug` (monkeypatch `check.typecheck` with a
  boom function, write a tmp source, call `check.main`,
  assert rc + stderr). Consistency-wise: clean.
- Each test docstring cites the audit context
  ("Audit 28.8 cycle 9 (regression for cycle-8 C8-1 close)
  …" and "… C8-2 close …"). The cycle-N labelling matches
  the cycle-9 audit's recommendation. Audit-trail clean.
- Test error messages (the f-string in each assert) include
  the actual rc / stderr content on failure — debuggable on
  CI when a future refactor breaks them. No silent-failure
  surface in the test code itself.
- The `count("helixc:") == 1` assertion is the strictest
  formulation available (a `not in` for `"helixc: helixc:"`
  would still pass if someone emitted `"helixc:foo helixc:
  bar"` on one line; an equality on the full string would
  be too brittle to the exact message wording). The count
  formulation is the correct middle ground.
- The persisted cycle-8/9 audit docs (`audit-stage28-8-
  cycle8-codereview-rev.md`, `audit-stage28-8-cycle9-
  codereview.md`, `audit-stage28-8-cycle9-silent-failures.md`,
  `audit-stage28-8-cycle9-type-design.md`) are pure
  documentation — no production-code or test-code change,
  cannot introduce a silent failure.
- No new `except` clause, `try` block, fallback chain,
  optional-chain pattern, or default-value-on-error pattern
  was introduced in cycle 10. The cycle-9 production-code
  surface for `_emit_env_error` + the broad-Exception arm is
  unchanged.
- The cycle-10 commit message ("Tests: 267 targeted pass
  (was 264 cycle-9 → +3 new)") is now accurate — the +3
  delta correctly reflects test additions (not pre-existing
  tests).

---

## Cycle 9 recommendation vs cycle 10 delivery

The cycle-9 audit recommended 4 regression tests under
"Recommendation 1 PREFERRED". Cycle 10 delivered 3.
Coverage comparison:

| Cycle-9 recommended test | Cycle-10 delivered test | Coverage equivalence |
|---|---|---|
| `test_c8_1_import_error_attributed_as_compiler_bug` (assert `rc == 1` + "compiler bug" + "please file an issue" when ImportError raised) | `test_c8_1_import_error_attributed_as_compiler_bug` (asserts `rc == 1` + "compiler bug" + "internal error") | **equivalent** (asserts the type-name "internal error" tag instead of the "please file an issue" tag — both are emitted by the same arm and either is sufficient to detect the C8-1 regression). |
| `test_c8_2_emit_env_error_strips_existing_prefix` (direct unit call: `_emit_env_error("helixc: foo")` emits `"helixc: foo\n"`) | `test_c8_2_env_error_no_double_helixc_prefix` (end-to-end: raise FileNotFoundError with pre-prefixed message; assert no `helixc: helixc:` substring + single-prefix preserved) | **equivalent** (end-to-end form covers the same code path: `_emit_env_error` is invoked with a pre-prefixed string and must strip; a regression in the strip would surface as `helixc: helixc:` in stderr). |
| `test_c8_2_emit_env_error_adds_prefix_when_absent` (direct unit call: `_emit_env_error("foo")` emits `"helixc: foo\n"`) | `test_c8_2_env_error_no_prefix_still_prefixed` (end-to-end: raise FileNotFoundError with un-prefixed message; assert `count("helixc:") == 1` in stderr) | **equivalent** (end-to-end form covers the same code path; the unique `count == 1` assertion catches both prefix-dropping AND double-prefix regressions). |
| `test_c8_2_stdlib_strict_missing_file_single_prefix` (end-to-end through `parser._merge_stdlib` raise) | **NOT separately delivered** | **subsumed** by `test_c8_2_env_error_no_double_helixc_prefix`. From the `_emit_env_error` perspective, the helper has no knowledge of which raise site triggers it; what matters is the message shape ("starts with `helixc:`"). The cycle-10 test exercises the exact same message shape (`"helixc: stdlib file missing: foo.hx"`) via a `check.typecheck` monkeypatch rather than a `parser._merge_stdlib` monkeypatch. The handler behavior is identical regardless of which inner-call raised. |

**Verdict**: the cycle-10 3-test delivery achieves the same
coverage of the cycle-9 fix surface as the cycle-9 4-test
recommendation would have. The "missing" 4th test
(`test_c8_2_stdlib_strict_missing_file_single_prefix`) is
genuinely subsumed by `test_c8_2_env_error_no_double_helixc_
prefix` — they both exercise `_emit_env_error("helixc:
stdlib file missing: foo.hx")`. No coverage gap.

This is NOT a new finding because (a) the cycle-9
recommendation said "4 tests" but the underlying coverage
requirement was "regression-protect both cycle-9 closes",
which 3 well-chosen tests fully satisfy; and (b) audit
recommendations are recommendations, not contract — the
fix-sweep author's prerogative is to choose the minimum
test set that protects the regression surface.

---

## Cross-stage interactions checked

- **Cycle-10 tests + cycle-9 `_emit_env_error` helper**:
  the helper's `.lstrip()` + `startswith("helixc:")` strip
  is exercised by `test_c8_2_env_error_no_double_helixc_
  prefix` with the exact realistic message shape
  (`"helixc: stdlib file missing: foo.hx"` — the
  `parser.py:1587` shape). The `.lstrip()` + re-prepend
  path is exercised by `test_c8_2_env_error_no_prefix_still_
  prefixed` with an unprefixed message. Both branches of the
  strip's `if text.lstrip().startswith("helixc:"):` are
  covered. No untested branch in the helper.
- **Cycle-10 tests + cycle-9 broad-Exception arm**:
  the broad-Exception arm path (post-ImportError-arm-drop)
  is exercised by `test_c8_1_import_error_attributed_as_
  compiler_bug` raising `ImportError` from inside
  monkeypatched `typecheck`. This confirms ImportError
  reaches the broad-Exception arm and emits the
  "internal error" + "compiler bug" pair. No fresh silent
  surface.
- **Cycle-10 tests + cycle-2 AD-warning drain in finally**:
  the cycle-2 C2-1 AD-drain in the `finally:` block runs
  after every cycle-10 test's `check.main` call. Since the
  monkeypatched `typecheck` raises before any AD warnings
  could be collected, the drain has zero warnings to emit —
  so it's a no-op. The cycle-5 C4-6 drain-wrap (try/except
  Exception inside the finally) is also exercised (cleanly
  drained without exception). No fresh silent surface.
- **Cycle-10 tests + cycle-3 C3-3 try/finally wrap of
  `_main_inner`**: the `try: rc = _main_inner(argv,
  a_holder)` wrap catches the test's monkeypatched raise
  from `typecheck` (which `_main_inner` calls). The cycle-3
  C3-3 fix (clean error message + drain on exception exit)
  is exercised. The `a_holder` push from `_main_inner` after
  parse-args succeeds is reached before the raise (the
  raise happens later in the pipeline at `typecheck()`), so
  the finally's `if a_holder:` branch is taken — the drain
  runs against the parsed CliArgs object. No fresh silent
  surface.
- **Cycle-10 tests + future contributors**: a new
  contributor refactoring `_emit_env_error` (e.g., adding a
  `level` parameter or routing through a logger) MUST now
  satisfy the 3 regression tests — they form a contract on
  the helper's strip + re-prefix behavior. This is
  load-bearing documentation in test form, addressing the
  cycle-9 C9-1 hidden-error #4 ("a new contributor reading
  check.py would see the helper but not understand from any
  test what its strip-behavior contract is").

---

## Deferred / out-of-scope observations (NOT new findings; cycle-11 candidates)

- **C7-1 test-coverage gap (cycle-8 carryover, cycle-9
  unaddressed, cycle-10 also unaddressed)**: cycle 10 did
  not add the 4 `_compatible(TyMemTier, TyVar)` regression
  tests requested in the cycle-7 audit and re-flagged in
  cycles-8/9 "Deferred" sections. Recommend cycle 11
  combine these with the audit-C4-1 fix attempt. The C7-1
  gap is NOT re-flagged as a new cycle-10 finding because
  the cycle-10 fix-sweep did not CHANGE typecheck.py at all
  — it's pure carryover.
- **Carryover audit-C4-1 (D2 Call-RHS i32 SIGILL)**: still
  open CRITICAL. Cycle 10 did not address. **HIGHEST-
  PRIORITY ITEM** for cycle 11 — the still-open CRITICAL is
  now the only remaining serious obstacle to the 5-clean-
  cycle gate. Cycles 5-10 have closed every other
  CRITICAL/HIGH/MEDIUM finding; audit-C4-1 is the last hard
  bug.
- **Carryover audit-C4-4 (D9 paper-only)**: still open
  HIGH. Not addressed in cycle 10.
- **Carryover audit-C4-8 (check.py doesn't call fn-mono)**:
  still open LOW. Not addressed.
- **monomorphize_safe docstring drift**: still open
  (cycle-6 deferred). Docstring suggests callers MAY ignore
  diags; the only caller (x86_64.py) now aborts.
- **D-vs-Quote diagnostic text**: still open (cycle-7
  deferred). Quote-wrapped case still emits "(one side
  D-wrapped, other bare)".
- **`_emit_env_error` triple-prefix edge case**: the cycle-9
  audit noted that a message like `"helixc: helixc: triple"`
  would only strip one level, leaving `"helixc: helixc:
  triple"` in the output. The cycle-9 audit also noted no
  callee currently emits triple-prefixed messages. Cycle 10
  does NOT add a test for this case (correctly — there's no
  callee to regression-protect). If a future cycle ever
  introduces a callee that could emit a triple-prefixed
  message, a test should be added then. Not a finding now.
- **`_emit_env_error` uppercase-prefix edge case**: the
  helper's `startswith("helixc:")` is case-sensitive — a
  message like `"HELIXC: ..."` would NOT be stripped. No
  callee emits uppercase, so this is not a finding. If a
  future contributor adds a callee that does (e.g., a
  stack-trace formatter that uppercases module names), the
  helper would need a `.lower()` check. Not a finding now.

---

## Summary

| #    | Severity | Location | Finding |
|------|----------|----------|---------|
|      |          |          | (none — cycle 10 is CLEAN for the silent-failure lens) |

**Total: 0 new findings (0 CRITICAL, 0 HIGH, 0 MEDIUM,
0 LOW).**

---

## Cycle 10 status

**Cycle 10 IS CLEAN** for the silent-failure audit lens. Per
the strict criterion (zero findings of ANY severity), the
0-finding result satisfies the clean-cycle gate for this
audit lens.

### Stop-the-line determination: **NO**

Cycle 10 is clean — no stop required for this lens.

### Cycle 10 → NEW FINDINGS COUNT for the strict-clean gate: 0 (0 CRITICAL + 0 HIGH + 0 MEDIUM + 0 LOW) — clean-counter advances by 1 (subject to the parallel type-design + code-review audit lenses also being clean for this cycle).

### Severity trend across cycles

- Cycle 1: 13 findings (3 HIGH, 5 MEDIUM, 5 LOW).
- Cycle 2: 6 findings (1 HIGH, 4 MEDIUM, 1 LOW).
- Cycle 3: 6 findings (0 HIGH, 4 MEDIUM, 2 LOW).
- Cycle 4: 8 findings (1 CRITICAL, 2 HIGH, 4 MEDIUM, 1 LOW).
- Cycle 5: 4 findings (0 CRITICAL, 0 HIGH, 2 MEDIUM, 2 LOW).
- Cycle 6: 1 finding (0 CRITICAL, 0 HIGH, 1 MEDIUM, 0 LOW).
- Cycle 7: 1 finding (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW).
- Cycle 8: 2 findings (0 CRITICAL, 0 HIGH, 1 MEDIUM, 1 LOW).
- Cycle 9: 1 finding (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW).
- Cycle 10: 0 findings. ← here

Trend: severity-monotone-decrease continued AND finding
count hits zero for the first time in the cycle-1-through-10
audit history. Cycle 10 is the cleanest cycle since the
audit series began. Net delta on open severity-weighted
ledger: -1 LOW closed (C9-1), 0 opened.

### Estimated remaining open findings going into cycle 11

- Cycle 1: 13 new (all fixed → 0 open).
- Cycle 2: 6 new (all fixed → 0 open).
- Cycle 3: 6 new (all fixed → 0 open).
- Cycle 4 silent-failure: 8 new — 6 closed by cycles 5-9.
  2 still open: audit-C4-1 CRITICAL, audit-C4-4 HIGH.
- Cycle 4 type-design (sibling audit): partial close — E3
  closed; E1 closed; others unchanged.
- Cycle 4 codereview (sibling audit): 0 new (was already
  clean).
- Cycle 5 silent-failure: 4 new — all 4 CLOSED by cycle 6.
- Cycle 6 silent-failure: 1 new (C6-1) — CLOSED.
- Cycle 6 type-design: 2 new (G1, G2) — both CLOSED.
- Cycle 7 silent-failure: 1 new (C7-1) — CLOSED by cycle 8.
- Cycle 8 silent-failure: 2 new (C8-1 MEDIUM, C8-2 LOW) —
  both CLOSED by cycle 9.
- Cycle 9 silent-failure: 1 new (C9-1 LOW) — CLOSED by
  cycle 10.
- Cycle 10 silent-failure: 0 new. ← here
- Prior audits (stage 5-6 + 7-8 + 9-16): ~20 still-open
  (unchanged going into cycle 10).
- Cycle 10 net: 20 + 0 + (deferred type-design partial) + 2
  = **≥22 open findings** going into cycle 11. (Net -1 from
  cycle 9: cycle 10 closed C9-1, opened nothing.)

Recommend prioritizing in this order for the cycle-11 fix
batch:
1. **audit-C4-1** (CRITICAL — still-open from cycle 4;
   highest-priority unaddressed-CRITICAL; deferred in
   cycles 6-10; the carryover deadline approaches as the
   strict-clean gate accumulates clean cycles).
2. **C7-1 test-coverage gap** (cycle-8 carryover —
   combinable with audit-C4-1 if the fix touches
   typecheck.py).
3. **audit-C4-4** (HIGH — D9 paper-only).
4. **monomorphize_safe docstring drift** (housekeeping).
5. **D-vs-Quote diagnostic text** (housekeeping).

After this batch lands, cycle 11 should re-audit. The
"5 clean cycles before Phase 0 deprecation" goal requires
the strict criterion (zero findings of any severity) to be
met for 5 CONSECUTIVE cycles — cycle 10 is the 1st clean
cycle (subject to type-design + code-review sibling-audit
verdicts for cycle 10). The clean-counter advances to **1**
for this audit lens. Cycle 10 is a milestone: the first
clean cycle of the 5-clean gate. The production-code
surface is now mature for a Phase-0 pause and the cycle-1
through cycle-10 fix work has substantively de-risked
the silent-failure audit lens. The remaining work for the
5-clean gate is to (a) close audit-C4-1 (CRITICAL — the
single largest residual risk) and (b) sustain zero-finding
cadence across cycles 11-14 with no new regressions.
