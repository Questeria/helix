# Stage 28.8 Pre-29 Audit Gate — Cycle 10, Audit C: Code Review

**Date**: 2026-05-11
**Commit**: c2e36d4 (read-only)
**Scope**: Audit C (general code-review) of the cycle-10
fix-sweep at commit c2e36d4. The fix-sweep is strictly scoped
to **three new regression tests** appended to
`helixc/tests/test_typecheck.py`, closing cycle-9 finding C9-1
(LOW, silent-failures): "no regression tests added for the
cycle-9 C8-1/C8-2 fixes." No production code (`check.py`,
parser, IR, backend, stdlib) was modified.

The fix-sweep commit also persists 4 audit docs (cycle-8
codereview-rev + cycle-9 codereview / silent-failures /
type-design); those are read-only documentation and not in
audit-C scope.

Files reviewed at HEAD (c2e36d4):

1. `helixc/tests/test_typecheck.py` lines 1572-1634 — the three
   new tests:
   - `test_c8_1_import_error_attributed_as_compiler_bug`
   - `test_c8_2_env_error_no_double_helixc_prefix`
   - `test_c8_2_env_error_no_prefix_still_prefixed`
2. `helixc/tests/test_typecheck.py` lines 1530-1569 — the two
   existing cycle-5 C4-6 regression tests, to confirm the new
   tests follow the same fixture / monkeypatch / capsys idiom
   (style-consistency check).
3. `helixc/check.py` lines 246-318 — production code under
   test (`_emit_env_error` helper + outer `except` cascade with
   FileNotFoundError + UnicodeDecodeError + broad arm), to
   confirm the assertions in the three new tests bind to the
   actual production-code contracts at HEAD.
4. `helixc/check.py` lines 58 + 403 — `typecheck` import-site
   binding and call-site resolution, to confirm
   `monkeypatch.setattr(check_mod, "typecheck", boom)` actually
   intercepts the call (the monkeypatch only works if
   `typecheck` is referenced as a module-level attribute, not a
   local rebind).

**Method**: Read the cycle-9 codereview (CLEAN), cycle-9
silent-failures (source of C9-1), and cycle-9 type-design
(CLEAN) docs to load the cumulative invariant set. Walked the
full cycle-10 fix-sweep diff (`git show c2e36d4` on the test
file). For each new test, cross-walked the production code path
from `check.main([str(src_file)])` through the outer try, into
`_main_inner`, to the call site at `check.py:403` where the
monkeypatched `typecheck` raises, back out through the outer
except arm, into `_emit_env_error` (or the broad arm), to the
stderr write. Verified that:

- The monkeypatch target (`check_mod.typecheck`) is correctly
  bound by the module-level `from .frontend.typecheck import
  typecheck` at `check.py:58` and resolved via that name at
  the call site (line 403), so the patch genuinely intercepts.
- Each test's assertions match the post-cycle-9 production
  contract: rc + stderr substrings.
- No test depends on stdout, on filesystem ordering, on
  network, on threading, or on the harness clock.
- Each test uses `tmp_path` for the input source file (no
  collision risk across parallel pytest workers) and
  `capsys.readouterr()` (the standard pytest capture
  fixture).

Direct invocation `pytest helixc/tests/test_typecheck.py::
test_c8_1_... ::test_c8_2_no_double... ::test_c8_2_no_prefix...
-v` at HEAD passes all 3 tests cleanly. Confirmed the stderr
shape for the third test ("plain message, no prefix") with a
direct Python probe — stderr is exactly
`"helixc: plain message, no prefix\n"`, so the
`captured.err.count("helixc:") == 1` invariant is exact and
flake-free (the entry banner uses `-- helixc-check:` and goes
to stdout, not stderr).

**Reporting threshold**: confidence >= 80 (per the cycle-10
audit-C prompt's strict criterion).

**Result**: **0 findings (0 CRITICAL, 0 HIGH, 0 MEDIUM, 0 LOW)
at or above the confidence-80 reporting threshold.**

---

## Summary table

| ID    | Severity | Confidence | Component | Issue |
|-------|----------|------------|-----------|-------|
| —     | —        | —          | —         | (none at or above threshold) |

**Cycle 10 Audit C: CLEAN — 0 findings at the confidence-80
threshold.**

Per user directive 2026-05-10 (strict criterion): cycle counts
CLEAN only when zero findings of ANY severity at or above the
audit threshold. **This cycle qualifies as clean.**

---

## Cycle-9 finding closure verification

### C9-1 (LOW, conf ~85 in cycle-9 silent-failures): no regression tests added for cycle-9 C8-1 (ImportError → rc=1 + compiler-bug) and C8-2 (no double `helixc:` prefix) fixes — **CLOSED**

The cycle-9 fix-sweep landed correct production code for C8-1
(drop the dedicated `except ImportError` arm so genuine import
bugs fall to the broad `except Exception` and get rc=1 +
"compiler bug" tagline) and C8-2 (new `_emit_env_error` helper
strips a leading `helixc:` from the message before re-
prefixing, so callees that pre-formatted the prefix — currently
only `parser.py:1587` in strict-stdlib mode — produce a single-
prefix output). The cycle-9 codereview classified both as
CLOSED. The cycle-9 silent-failures audit, however, opened a
new LOW finding C9-1: no regression tests pin either contract,
so a future refactor that re-introduces the same misbehavior
would not be caught at test-time.

Cycle 10 adds three regression tests that pin all three
documented post-cycle-9 contracts:

#### `test_c8_1_import_error_attributed_as_compiler_bug`

```python
def boom(*_args, **_kw):
    raise ImportError("cannot import name 'monomorphize_structs'")
monkeypatch.setattr(check_mod, "typecheck", boom)
src_file = tmp_path / "boom.hx"
src_file.write_text("fn main() -> i32 { 0 }")
rc = check_mod.main([str(src_file)])
captured = capsys.readouterr()
assert rc == 1, f"expected rc=1 for ImportError, got rc={rc}"
assert "compiler bug" in captured.err, (
    f"expected 'compiler bug' tag in stderr, got: {captured.err}"
)
assert "internal error" in captured.err
```

Walkthrough:

- The monkeypatch substitutes the module-level `typecheck`
  attribute (bound at `check.py:58` by `from .frontend.
  typecheck import typecheck`) with `boom`. The call site at
  `check.py:403` resolves `typecheck` via that name, so the
  patch genuinely intercepts.
- `boom` raises a vanilla `ImportError` with a plausible
  internal-rename message. `ImportError` is a subclass of
  `Exception` but not of `OSError` or `UnicodeError`, so it
  bypasses the FileNotFoundError-family and UnicodeDecodeError
  arms and lands in the broad `except Exception` arm at
  `check.py:306-318`. That arm prints `helixc: internal error:
  ImportError: cannot import name 'monomorphize_structs'`
  followed by `helixc: this is a compiler bug — please file an
  issue.` and sets `rc = 1`.
- The three assertions (`rc == 1`, `"compiler bug" in
  captured.err`, `"internal error" in captured.err`) directly
  pin the documented C8-1 contract: ImportError must surface
  as a compiler bug with rc=1 and the "please file an issue"
  hint, NOT as a user-env error with rc=2 and no hint.
- Pre-cycle-9, this test would have failed: the dedicated
  `except ImportError` arm produced `rc=2` and the message
  `helixc: import error: cannot import name ...`, without
  either "compiler bug" or "internal error" substrings.

**Test correctness: PASS.** Asserts exactly the documented
behavior, pins all three observable outputs (rc + 2 stderr
substrings).

#### `test_c8_2_env_error_no_double_helixc_prefix`

```python
def boom(*_args, **_kw):
    raise FileNotFoundError("helixc: stdlib file missing: foo.hx")
monkeypatch.setattr(check_mod, "typecheck", boom)
src_file = tmp_path / "boom.hx"
src_file.write_text("fn main() -> i32 { 0 }")
rc = check_mod.main([str(src_file)])
captured = capsys.readouterr()
assert rc == 2, f"expected rc=2 for FileNotFoundError, got rc={rc}"
assert "helixc: helixc:" not in captured.err, (
    f"double prefix in stderr: {captured.err!r}"
)
assert "stdlib file missing" in captured.err
```

Walkthrough:

- The monkeypatched `typecheck` raises FileNotFoundError with
  the **exact same message shape** as the only known pre-
  prefixer in the codebase: `parser.py:1587` (which raises
  `FileNotFoundError(f"helixc: stdlib file missing:
  {stdlib_path}")` in strict-stdlib mode). The test thus
  reproduces the C8-2 trigger scenario directly via the
  monkeypatch rather than indirectly via `HELIXC_STDLIB_STRICT
  =1` + a synthetic missing-stdlib file. Both approaches lock
  in the same observable contract; the monkeypatch approach is
  preferred here because it doesn't depend on stdlib path
  resolution and parses cleanly without spurious environment
  setup.
- FileNotFoundError lands in the outer arm at lines 299-302,
  which routes through `_emit_env_error(str(e))`. The helper
  detects the leading `helixc:`, strips 7 chars + leading
  whitespace, then re-prefixes — yielding exactly one
  `helixc:` token in the output. Stderr is `helixc: stdlib
  file missing: foo.hx\n`.
- The three assertions (`rc == 2`, `"helixc: helixc:" not in
  captured.err`, `"stdlib file missing" in captured.err`) pin
  the C8-2 contract: env-error path, no double prefix, message
  content preserved.
- Pre-cycle-9, this test would have failed on the second
  assertion: cycle-5's outer arm did `print(f"helixc: {e}",
  file=sys.stderr)`, producing `helixc: helixc: stdlib file
  missing: foo.hx`.

**Test correctness: PASS.** The choice of FileNotFoundError
over a real strict-stdlib path is deliberate and defensible —
it isolates the test to the helper's strip-behavior without
coupling to parser internals.

#### `test_c8_2_env_error_no_prefix_still_prefixed`

```python
def boom(*_args, **_kw):
    raise FileNotFoundError("plain message, no prefix")
monkeypatch.setattr(check_mod, "typecheck", boom)
src_file = tmp_path / "boom.hx"
src_file.write_text("fn main() -> i32 { 0 }")
rc = check_mod.main([str(src_file)])
captured = capsys.readouterr()
assert rc == 2
assert captured.err.count("helixc:") == 1, (
    f"expected single 'helixc:' prefix, got: {captured.err!r}"
)
```

Walkthrough:

- The monkeypatched `typecheck` raises FileNotFoundError with
  a message that does NOT start with `helixc:`. This is the
  "no-strip" branch of `_emit_env_error`: the
  `if text.lstrip().startswith("helixc:")` check is false, so
  no strip happens, and the helper prepends `helixc: ` exactly
  once. Stderr is `helixc: plain message, no prefix\n`.
- The two assertions (`rc == 2`, `captured.err.count("helixc:"
  ) == 1`) pin the helper's base behavior: the strip is
  conditional (only fires when the message is already
  prefixed), and the prefix-append is unconditional.
- Probed the exact stderr shape at HEAD via direct Python
  invocation: `'helixc: plain message, no prefix\n'` —
  count of `"helixc:"` is exactly 1. Tight invariant, no
  flake risk.
- The entry banner emitted by `_main_inner` at line 374
  (`print(f"-- helixc-check: {path}")`) goes to stdout (no
  `file=sys.stderr` kwarg). The test reads only
  `captured.err`, so the banner does NOT pollute the
  `helixc:` count.

**Test correctness: PASS.** The single `count("helixc:") == 1`
assertion is exact; it would fail both if the helper double-
printed (count >= 2) and if the helper skipped the prefix
entirely (count == 0). Covers the full no-prefix branch.

**C9-1 CLOSED.** All three documented post-cycle-9 contracts
(C8-1 ImportError → rc=1 + compiler-bug, C8-2 already-prefixed
message → single prefix, C8-2 unprefixed message → single
prefix added) are now pinned by named regression tests.

---

## Code-quality of the three new tests

### Naming

- `test_c8_1_import_error_attributed_as_compiler_bug` — names
  the cycle (C8-1), the input shape (ImportError), and the
  asserted outcome (attributed as compiler bug). Mirrors the
  cycle-5 sibling `test_c4_6_filenotfound_not_attributed_as_
  compiler_bug` in form.
- `test_c8_2_env_error_no_double_helixc_prefix` — names the
  cycle (C8-2), the category (env error), and the asserted
  negative invariant (no double prefix).
- `test_c8_2_env_error_no_prefix_still_prefixed` — names the
  cycle (C8-2), the input shape (no prefix in message), and the
  asserted outcome (still prefixed in output). Slightly
  awkward phrasing ("no_prefix_still_prefixed") but
  unambiguous in context.

All three names are descriptive, audit-traceable (cycle ID +
finding ID in the name), and disambiguated from each other
and from the prior cycle-5 tests. **Naming: PASS.**

### Docstrings

Each test's docstring includes:

- Audit cycle reference (28.8 cycle 9; for cycle-10 close).
- Finding ID (C8-1 / C8-2).
- Brief description of the pre-fix behavior.
- The post-fix contract being asserted.

Style matches the cycle-5 C4-6 docstrings. **Docstrings: PASS.**

### Assertion strength

- `test_c8_1_...`: asserts rc, "compiler bug" substring, and
  "internal error" substring. Three independent observable
  invariants; failure of any one would catch a regression.
- `test_c8_2_no_double_...`: asserts rc, negative-substring
  ("helixc: helixc:" not in err), and positive-substring
  ("stdlib file missing" in err). The negative-substring
  check is the load-bearing one for C8-2. **Note**: a
  hypothetical bug where the helper triple-printed
  ("helixc: helixc: helixc: stdlib file missing: foo.hx")
  would still fail the negative-substring check because
  "helixc: helixc:" is a substring of any double-or-more
  repetition. Coverage is tight.
- `test_c8_2_no_prefix_still_...`: asserts rc and exact-count
  (count == 1). The exact-count is stricter than substring
  checks — it catches both double-print AND missing-prefix
  regressions in a single assertion. **Strongest assertion
  shape of the three.**

All three tests provide failure-message context via
f-string in the optional `msg` arg of `assert`. **Assertion
strength: PASS.**

### Fixture hygiene

All three tests use:

- `monkeypatch` for `check_mod.typecheck` substitution.
  Monkeypatch is auto-undone at test exit by pytest — no
  cross-test leakage.
- `capsys` for stderr/stdout capture. Auto-resets per test.
- `tmp_path` for the input source file. Per-test directory,
  no collision risk under parallel pytest workers (pytest-
  xdist).

`src_file.write_text("fn main() -> i32 { 0 }")` is a minimal
valid Helix program — parses successfully so the pipeline
reaches the typecheck call site (where the monkeypatched
`typecheck` then raises). Same source-text idiom as the cycle-5
sibling tests. **Fixture hygiene: PASS.**

### Flakiness

- No filesystem ordering dependency (each test owns its
  `tmp_path` subtree).
- No clock dependency.
- No network dependency.
- No threading or asyncio.
- No global state mutated outside the monkeypatch (which
  pytest auto-undoes).
- The `_DIFF_WARNINGS` module-level list in
  `helixc/frontend/autodiff.py` is drained by
  `check.main()`'s `_drain_ad_init()` call at the top before
  `_main_inner` runs, and again in the finally block — so any
  AD-warning leakage from a prior test in the same session
  is cleared. The cycle-5 sibling test
  (`test_c4_6_filenotfound_not_attributed_as_compiler_bug`)
  explicitly asserts `autodiff._DIFF_WARNINGS == []` after
  `boom`; the cycle-10 tests don't (because their scope is
  the C8-1/C8-2 contract, not the C2-1 drain contract), but
  the drain still runs and clears state. No leakage risk.
- The third test's exact-count assertion (`count("helixc:") ==
  1`) is the most-likely-to-flake invariant. Direct probe
  confirmed the exact stderr at HEAD is `"helixc: plain
  message, no prefix\n"` — count is exactly 1, with no
  ambient "helixc:" substrings from the entry banner (which
  goes to stdout) or from any other source.

**Flakiness: PASS — no flake vectors identified.**

### Run-time check

`pytest helixc/tests/test_typecheck.py::test_c8_1_import_error_
attributed_as_compiler_bug helixc/tests/test_typecheck.py::
test_c8_2_env_error_no_double_helixc_prefix helixc/tests/
test_typecheck.py::test_c8_2_env_error_no_prefix_still_
prefixed -v` at HEAD c2e36d4 returns 3 passed in 0.58s. **PASS.**

### Style consistency

All three tests follow the existing cycle-5 C4-6 sibling
pattern verbatim: `def test_...(monkeypatch, capsys, tmp_path):
docstring; from helixc import check as check_mod; def boom
raising; monkeypatch.setattr; src_file = tmp_path / ...; rc =
check_mod.main([str(src_file)]); captured = capsys.readouterr();
asserts`. No idiom drift. **Style: PASS.**

---

## Files reviewed

`helixc/tests/test_typecheck.py` (lines 1572-1634 for the three
new cycle-10 tests; lines 1530-1569 for the cycle-5 C4-6
siblings as the style template); `helixc/check.py` (lines 58 +
246-318 + 403 — module-level `typecheck` import, the
`_emit_env_error` helper + outer cascade, and the typecheck
call site). Plus the cycle-1 through cycle-9 codereview docs +
the cycle-9 silent-failures doc for cumulative invariant set +
C9-1 closure rationale.

---

## Specific cycle-10 changes audited

1. **test_typecheck.py:1572-1593 — new
   `test_c8_1_import_error_attributed_as_compiler_bug`**.
   Asserts rc=1, "compiler bug" in stderr, "internal error" in
   stderr after monkeypatching `check_mod.typecheck` to raise
   ImportError. Pins the post-cycle-9 contract for C8-1.
   Style-consistent with cycle-5 siblings. **PASS.**

2. **test_typecheck.py:1595-1614 — new
   `test_c8_2_env_error_no_double_helixc_prefix`**. Asserts
   rc=2, "helixc: helixc:" absent from stderr, "stdlib file
   missing" present in stderr after monkeypatching
   `check_mod.typecheck` to raise FileNotFoundError with the
   `parser.py:1587` pre-formatted shape. Pins the post-cycle-9
   contract for C8-2 (pre-prefixed branch). **PASS.**

3. **test_typecheck.py:1617-1634 — new
   `test_c8_2_env_error_no_prefix_still_prefixed`**. Asserts
   rc=2, exactly one `"helixc:"` substring in stderr after
   monkeypatching to raise FileNotFoundError with a no-prefix
   message. Pins the post-cycle-9 contract for C8-2 (no-strip
   branch). Strongest assertion shape (exact count). **PASS.**

4. **No production-code changes**. `helixc/check.py` and
   `helixc/frontend/parser.py` are byte-for-byte unchanged
   from cycle-9 HEAD 6968755. Confirmed via `git show c2e36d4
   --stat`. **PASS (no change).**

5. **Persisted audit docs**: `docs/audit-stage28-8-cycle8-
   codereview-rev.md`, `cycle9-codereview.md`, `cycle9-silent-
   failures.md`, `cycle9-type-design.md` added. Read-only
   documentation; not in audit-C scope. **PASS (out of scope,
   no quality concern).**

---

## What was checked and found below threshold

- **No coverage of the strip-helper's behavior on
  pathological inputs** (whitespace before prefix, uppercase
  `HELIXC:`, prefix-no-space-after-colon, double-pre-prefix
  `"helixc: helixc: foo"`, empty string, bare `"helixc:"`).
  Cycle-9 silent-failures direct-probed all 9 shapes and
  found the helper's behavior consistent / defensible. Cycle-
  10 only added tests for two of the nine shapes (the two
  observable through the cycle-5 outer except cascade). The
  uncovered cases are either unreachable in production
  (no callee currently produces them) or defensible cosmetic
  variations. Absence is consistent with the focused C9-1
  closure scope (test the documented contracts, not the
  helper's full behavior matrix). **Confidence 35**, below
  threshold.

- **No coverage of `_emit_env_error` invoked via real
  parser.py:1587 strict-stdlib path** (vs. the monkeypatched
  shortcut used by the cycle-10 test). A full end-to-end test
  using `HELIXC_STDLIB_STRICT=1` + an injected missing stdlib
  file would exercise the actual production trigger for C8-2
  (vs. the synthetic monkeypatch). The monkeypatched test
  uses the **exact same message shape** as the parser
  raises, so the contract is functionally equivalent; the
  monkeypatch approach is faster and isolates the test to the
  helper's strip-behavior. Trade-off is defensible.
  **Confidence 30**, below threshold.

- **No coverage of UnicodeDecodeError path through
  `_emit_env_error`** (only FileNotFoundError + ImportError
  are exercised in cycle 10). The cycle-5
  `test_c4_6_unicode_decode_error_clean_message` already
  covers UnicodeDecodeError → rc=2 + "encoding error" message
  (and that test still passes post-cycle-9, confirming the
  helper-routed path works for UnicodeDecodeError). Cycle 10
  could have added an assertion that the cycle-5
  UnicodeDecodeError test's stderr contains exactly one
  `"helixc:"` (it does: `helixc: encoding error reading
  source: ...`). Not required for C9-1 closure. **Confidence
  25**, below threshold.

- **Test ordering**: the three new tests are appended at lines
  1572-1634, between the cycle-5 C4-6 siblings (ending at
  line 1569) and `test_c3_4_monomorphize_structs_idempotent`
  (starting at line 1637). Logical neighbors for cycle-5 →
  cycle-9 audit-cycle grouping. Pytest test collection is
  unaffected by ordering. **Confidence 10**, below threshold.

- **The `boom` inner function uses `*_args, **_kw`** — same
  signature as the cycle-5 siblings. Captures the typecheck
  call shape (`typecheck(prog)`) safely without coupling to
  arity changes. **Confidence 10**, below threshold.

- **The cycle-9 sibling C9-1 only mentioned 2 missing tests**
  (ImportError + double-prefix). Cycle 10 added a third test
  for the no-prefix branch of `_emit_env_error`. The extra
  test pins the un-stripped path that the cycle-9 audit did
  not explicitly require, but which materially strengthens
  coverage of the helper. Net-positive over the minimal
  C9-1 close. **Confidence 10**, below threshold.

- **No regression on prior tests**: 264 tests passed pre-
  cycle-10; 267 pass post-cycle-10 (+3 new, 0 churned). Per
  the commit message. **Confidence 10**, below threshold.

- **No new dependencies, fixtures, or conftest changes**: the
  three tests use only pre-existing pytest fixtures
  (monkeypatch, capsys, tmp_path) and a pre-existing import
  pattern (`from helixc import check as check_mod`). No
  surface-area expansion. **Confidence 10**, below threshold.

- **Commit message scope**: explicitly names C9-1 closure +
  the three test names + their pre-fix vs. post-fix
  behaviors. Per-finding traceability is maintained. No
  commit-message hygiene concern. **Confidence 10**, below
  threshold.

---

## Open prior findings (not re-flagged this cycle)

Per the cycle-9 silent-failures tracking table:

- **audit-C4-1** (CRITICAL — D2 Call-RHS i32 SIGILL): still
  open; deferred pending parse-time constant folding or a
  typecheck pass before closure capture. Not addressed in
  cycle 10. Out of scope. **Highest-priority still-open
  carryover.**
- **audit-C4-4** (HIGH — D9 paper-only): still open; not
  addressed in cycle 10. Out of scope.
- **audit-C4-8 deferred** (LOW — check.py doesn't call
  fn-mono): still open; not addressed in cycle 10. Out of
  scope.
- **monomorphize_safe docstring drift** (cycle-6 deferred):
  still open; not addressed in cycle 10. Out of scope.
- **D-vs-Quote diagnostic text** (cycle-7 deferred): still
  open; not addressed in cycle 10. Out of scope.
- **C7-1 close test-coverage gap** (cycle-8 housekeeping
  candidate — `_compatible(TyMemTier, TyVar) is False` /
  `_compatible(TyMemTier, TySize) is False`): still open; not
  addressed in cycle 10. Out of scope.

No code-review regressions introduced by cycle 10:

- `check.py` byte-for-byte unchanged from cycle-9 HEAD —
  every cycle-1 through cycle-9 production-code contract
  preserved verbatim.
- `parser.py` byte-for-byte unchanged.
- `test_typecheck.py` appended-only (no existing tests
  modified, deleted, or reordered) — every pre-cycle-10
  test contract preserved verbatim.
- The three new tests do not introduce new fixtures, new
  conftest entries, new helpers, or new imports beyond
  what the cycle-5 siblings already establish.
- Direct test invocation at HEAD passes 3/3.

Pre-cycle-1 audit-stage5-6, audit-stage7-8, audit-stage9-16
baselines unchanged from cycle-1 status. Cycle-1 through
cycle-9 findings all marked CLOSED by their respective
fix-sweep commits or carried over per the tracking table.

---

## Verdict

**Cycle 10 Audit C: CLEAN — 0 findings (0 CRITICAL, 0 HIGH,
0 MEDIUM, 0 LOW) at or above the confidence-80 reporting
threshold.**

Strict-zero rule per user directive 2026-05-10. Cycle counter
advances provided cycles A (silent-failure) and B (type-
design) are also clean at this commit.

The cycle-10 fix-sweep is well-targeted: three regression
tests pinning the three observable post-cycle-9 contracts
(C8-1 ImportError → rc=1 + compiler-bug; C8-2 already-
prefixed → single prefix; C8-2 unprefixed → single prefix
added). The tests follow the cycle-5 C4-6 sibling idiom
verbatim, use pytest-standard fixtures with no flake
vectors, assert all observable outputs (rc + stderr
substrings + exact-count where appropriate), and run cleanly
at HEAD (3 passed in 0.58s). No production code was touched,
so no regression risk to the cycle-1 through cycle-9 invariant
set. The audit-doc persistence is read-only and out of
scope.

If cycles A (silent-failures) and B (type-design) at c2e36d4
are also clean, this is the **second clean cycle in a row**
(cycle 9 at 6968755 was the first), advancing the clean-
counter from 1 to 2 in the 5-clean-cycles gate.
