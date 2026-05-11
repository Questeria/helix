# Stage 28.8 Pre-29 Audit Gate — Cycle 10, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit**: c2e36d4 (read-only)
**Scope**: Re-audit the type-system surface after cycle 10's fix-sweep.
Cycle 10 is a **tests-only** commit (no production code changes):

- `helixc/tests/test_typecheck.py` +65 lines, 0 deletions. Three new
  regression tests appended after `test_c4_6_unicode_decode_error_
  clean_message` (line 1569):
  - `test_c8_1_import_error_attributed_as_compiler_bug` (lines
    1572-1592): asserts ImportError raised from inside `_main_inner`
    surfaces as rc=1 with `"compiler bug"` + `"internal error"`
    diagnostic strings in stderr (closes cycle-9 forward note #1).
  - `test_c8_2_env_error_no_double_helixc_prefix` (lines 1595-1614):
    asserts a callee FileNotFoundError with a pre-existing `helixc:`
    prefix produces single-prefix stderr output via `_emit_env_error`
    (closes cycle-9 forward note #2).
  - `test_c8_2_env_error_no_prefix_still_prefixed` (lines 1617-1634):
    asserts the helper still prepends `helixc:` when the callee
    message has no prefix (preserves base behavior).
- `docs/audit-stage28-8-cycle8-codereview-rev.md` +559 lines (new doc,
  persistence of the cycle-8 codereview-rev audit).
- `docs/audit-stage28-8-cycle9-codereview.md` +519 lines (new doc).
- `docs/audit-stage28-8-cycle9-silent-failures.md` +627 lines (new doc).
- `docs/audit-stage28-8-cycle9-type-design.md` +428 lines (new doc).

No `helixc/check.py`, `helixc/frontend/*.py`, or any other production-
code file is touched. `git diff c2e36d4~1 c2e36d4 -- helixc/` shows
only `helixc/tests/test_typecheck.py` modified inside the helixc tree.

**Method**: read the cumulative cycle-1 through cycle-9 type-design
audit docs to confirm no prior-cycle invariant was reopened. Ran
`git show c2e36d4` and `git diff c2e36d4~1 c2e36d4 --stat` to confirm
the tests-only scope at the file level. Walked the three new test
functions to confirm they (a) exercise only the cycle-9 fix-sweep
contract that was already audited CLEAN in cycle 9, (b) do not
introduce any new type-design surface (no new class, no new function
signature visible to production code, no monkey-patch of typecheck
internals that would imply a hidden contract), and (c) the
`monkeypatch.setattr(check_mod, "typecheck", boom)` injection-site
pattern relies on `helixc.check` re-exporting `typecheck` from the
frontend module — an existing public-import contract (lines 73-76 of
check.py at HEAD c2e36d4) that was already audited in cycles 1-9 and
is unchanged.

**Findings summary**: No type-system contract is touched. Re-verified
by inspection:

- `_compatible` TyMemTier strict-separation contract (typecheck.py
  ~2224-2252): unchanged from cycle 9.
- `_size_compatible` shape-position cascade (typecheck.py ~2208-2222):
  unchanged.
- D-binop diagnostic-text accuracy (typecheck.py ~1349-1381):
  unchanged.
- Call-boundary `_compatible` invocation pre-filter (typecheck.py
  ~746-752): unchanged.
- `_emit_env_error` helper contract (check.py 246-255): unchanged
  from cycle 9 (CLEAN). The three new tests confirm the post-cycle-9
  behavior is observable from outside `main()` — they do NOT alter
  the helper.
- `main()` outer-dispatch classifier contract (check.py 286-318):
  unchanged from cycle 9 (CLEAN). Tests confirm the rc=1/rc=2 split
  and the `helixc:` prefix invariant hold under realistic
  monkey-patched typecheck failures.
- Cycle-9 forward notes #1 and #2 are now CLOSED by the new tests
  (this is the explicit purpose stated in the cycle-10 commit
  message and verified by reading the test bodies).

The new test functions are well-formed:
- Each follows the existing `monkeypatch + capsys + tmp_path` pattern
  used by sibling tests `test_c4_6_*` and `test_c5_*` (no new test
  scaffolding introduced).
- Each replaces a single dependency (`check_mod.typecheck`) with a
  raise-only stub, which is the standard way to exercise the outer-
  dispatch arms without standing up a full pipeline.
- Each writes a minimal valid source file to `tmp_path` solely to
  pass the file-existence pre-banner check at check.py:363 (the
  pre-banner check fires before the wrapped try/except so a missing
  file would short-circuit before reaching the arm under test).
- The assertions are conservative: substring matches for the
  diagnostic tags (`"compiler bug"`, `"internal error"`, `"stdlib
  file missing"`) rather than full-string equality, which is the
  correct invariant-strength level for a regression test (does not
  freeze cosmetic wording).
- The `count("helixc:") == 1` assertion in
  `test_c8_2_env_error_no_prefix_still_prefixed` is the right
  enforcement for the helper's single-prefix invariant.

The tests do NOT over-specify. The C8-1 test asserts both
`"compiler bug"` AND `"internal error"` are present, mirroring the
two-line diagnostic produced by the broad `except Exception` arm at
check.py:306-318. The C8-2 tests assert prefix-count invariants,
which is the precise contract the helper enforces.

**Result**: **0 findings (0 HIGH, 0 MEDIUM, 0 LOW)**. Cycle 10 is a
pure tests-only commit that closes the cycle-9 forward notes #1 and
#2. No type-system surface is added, modified, or removed. The new
regression tests strengthen the audit trail by making the cycle-9
contract enforceable at CI time. The strict criterion ("zero findings
of any severity") is **MET**.

---

## Cycle 9 finding re-verification

| ID   | Severity prev | Audit (prev)              | Status     | Notes                                                                                                                                                                                                                                                                                                                                                                       |
|------|---------------|---------------------------|------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| C9-1 | LOW           | silent-failures (cycle 9) | CLOSED     | The three new tests in `helixc/tests/test_typecheck.py` (test_c8_1_import_error_attributed_as_compiler_bug, test_c8_2_env_error_no_double_helixc_prefix, test_c8_2_env_error_no_prefix_still_prefixed) provide regression coverage for the C8-1 and C8-2 fixes that were landed without tests in cycle 9 (6968755). The cycle-10 commit message documents this explicitly. |

No prior-cycle type-design findings need re-verification. Cycle 9
type-design was CLEAN and cycle 10 does not touch any type-system
surface.

---

## Per-surface review (cycle-10 touchpoints)

### Surface 1: new regression tests in test_typecheck.py

**Placement**: `helixc/tests/test_typecheck.py:1572-1634`, immediately
after `test_c4_6_unicode_decode_error_clean_message` (the last
exception-classifier regression test). Logical proximity is correct:
all four tests (`test_c4_6_*`, `test_c8_1_*`, two `test_c8_2_*`)
exercise the same outer-dispatch try/except arm in `check.py:286-318`.

**Contracts asserted**: each test asserts an externally-visible
contract on `helixc.check.main()`. No internal type or invariant is
referenced; the tests are black-box w.r.t. the `_main_inner` /
`_emit_env_error` separation. This is the correct test-granularity
for a regression-against-CLI-output contract.

**Monkey-patch surface**: `monkeypatch.setattr(check_mod, "typecheck",
boom)` overrides the re-exported `typecheck` symbol on the `helixc.
check` module. This relies on the import pattern at check.py:73-76:

```python
from .frontend.typecheck import (
    typecheck,
    ...
)
```

which exposes `typecheck` as a module-attribute on `helixc.check`.
That re-export is a pre-existing public surface (used by all cycle-3,
4, 5, 9 regression tests). Cycle 10 does not introduce or change it.

**Test independence**: each test uses `tmp_path` for its source file,
so no cross-test state. Each uses `capsys` for stderr capture, which
is pytest-native and does not require fixtures. No `@pytest.fixture`
additions. No new conftest entry. The tests integrate cleanly into
the existing test_typecheck.py module.

**Type signatures**: all three test functions are `(monkeypatch,
capsys, tmp_path) -> None` (implicit `None` return). Standard pytest
signature. No type-design surface.

**Edge cases handled**:
- `test_c8_2_env_error_no_prefix_still_prefixed`: asserts the
  helper's "prefix-when-absent" branch (the False side of the
  `text.lstrip().startswith("helixc:")` conditional). Complements the
  "strip-when-present" branch tested by
  `test_c8_2_env_error_no_double_helixc_prefix`. Both branches of the
  helper's conditional are now under test.

**Edge cases NOT covered** (forward note, not a finding):
- The empty-string edge case (`_emit_env_error("")`) — cycle 9 noted
  this prints `helixc: ` and is acceptable but unlikely. No test
  asserts the behavior. Not blocking because no production callee
  passes an empty string.
- The nested-prefix edge case (`"helixc: helixc: foo"`) — cycle 9
  noted the helper only strips one layer. No test asserts the
  remaining single layer. Not blocking because no production callee
  produces nested prefixes.
- The leading-whitespace+prefix edge case (`"   helixc: foo"`) — not
  tested. Not blocking; the production callee at parser.py:1587 does
  not produce leading whitespace.

These are forward-test-coverage suggestions, not findings against
cycle 10. The cycle-10 tests cover the primary cycle-9 contracts
(rc classification + single-prefix invariant for prefixed and
unprefixed inputs); the edge cases above are second-order
hardenings.

### Surface 2: docs/audit-stage28-8-cycle{8-codereview-rev, 9-*}.md

**Placement**: four new markdown files in `docs/`, all audit reports.
No code surface. No type-design implication. Persistence of prior-
cycle audit findings into source control. Not a finding.

---

## Other surfaces (re-verified, not touched in cycle 10)

### check.py (cycles 1-9)

Cycle 10 does not modify `helixc/check.py`. The following contracts
are unchanged from cycle 9:

- `_emit_env_error` helper contract (lines 246-255, cycle 9 CLEAN).
- `main()` outer-dispatch classifier contract (lines 286-318, cycle
  9 CLEAN).
- `_drain_ad_warnings` helper (lines unchanged from cycle 5).
- Env-error callee contract (implicit, callees throughout helixc/).

Verified by inspection of the cycle-10 diff (`git show c2e36d4 --stat`
shows no helixc/check.py modification).

### typecheck.py (cycles 1-9)

Cycle 10 does not modify `helixc/frontend/typecheck.py`. All
prior-cycle invariants preserved by inspection:

- `_compatible` TyMemTier strict-separation contract.
- `_size_compatible` shape-position cascade.
- D-binop diagnostic-text accuracy.
- Call-boundary `_compatible` invocation pre-filter.

### parser.py (cycles 1-9)

Cycle 10 does not modify `helixc/frontend/parser.py`. The
`raise FileNotFoundError(f"helixc: stdlib file missing: {p}")` at
parser.py:1587 (the production single-layer-prefix callee that
exercises the helper's strip branch) is unchanged. The cycle-10
test `test_c8_2_env_error_no_double_helixc_prefix` uses a synthetic
raise rather than triggering this real callee, which is the correct
abstraction — the test should be insulated from parser.py refactors.

---

## Cycle 10 invariant snapshot (post-fix)

No new invariants introduced. The cycle-9 invariant snapshot remains
authoritative; cycle 10 simply pins the cycle-9 contract under CI
regression coverage.

For completeness, the cycle-9 contracts now exercised by cycle-10
tests:

**`_emit_env_error` helper contract** (check.py:246-255):
- Now exercised by both branches: prefix-strip (test_c8_2_env_error_
  no_double_helixc_prefix) AND prefix-add (test_c8_2_env_error_no_
  prefix_still_prefixed).

**`main()` outer-dispatch classifier contract** (check.py:284-318):
- ImportError → rc=1 + compiler-bug now exercised by
  test_c8_1_import_error_attributed_as_compiler_bug.
- FileNotFoundError-family → rc=2 + single-prefix exercised by both
  C8-2 tests.

---

## Cycle 10 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity.**

This cycle finds **0 findings (0 HIGH, 0 MEDIUM, 0 LOW)**.

By the strict criterion, **cycle 10 counts CLEAN**.

The severity trend across cycles is now:
- Cycle 1: HIGH-tier finding(s)
- Cycle 2: HIGH + MEDIUM
- Cycle 3: HIGH + MEDIUM + LOW (multiple LOW)
- Cycle 4: MEDIUM-tier
- Cycle 5: 3 MEDIUM + 3 LOW
- Cycle 6: 1 MEDIUM + 2 LOW
- Cycle 7: 0 + 0 + 0  ←  CLEAN
- Cycle 8: 0 + 0 + 0  ←  CLEAN
- Cycle 9: 0 + 0 + 0  ←  CLEAN
- Cycle 10: 0 + 0 + 0  ←  CLEAN

This is the FOURTH consecutive cycle to meet the strict criterion
under Audit B. The cycle-10 commit is a pure tests-only addition
that strengthens the regression-test floor for the cycle-9 fix-sweep
without introducing any new type-system surface or invariant.

The 5-clean-cycles requirement (per the cycle-5 doc's projection for
Python-helixc deprecation) is now 4/5. Cycle 11 would need to clean
to satisfy that bar.

**Recommendation**: no fix-sweep needed for cycle 10. Proceed to
cycle 11 audit gate.

---

## Forward notes (not cycle-10 findings)

1. **Empty-string edge case for `_emit_env_error`**: no test exercises
   `_emit_env_error("")`. The cycle-9 doc noted this prints `helixc: `
   and is acceptable but unlikely. A future cycle could add a
   defensive test that asserts the empty-input behavior is stable.
   Not blocking.

2. **Nested-prefix edge case for `_emit_env_error`**: no test
   exercises `_emit_env_error("helixc: helixc: foo")`. The cycle-9
   doc noted the helper only strips one layer; a defensive test
   could pin this single-layer behavior. Not blocking — no
   production callee produces nested prefixes.

3. **Whitespace-handling edge case**: no test exercises `"   helixc:
   foo"` (leading whitespace before prefix). The helper handles this
   correctly per cycle 9. A defensive test could pin the behavior.
   Not blocking.

4. **Convention note carry-over from cycle 9**: the implicit
   raise-message-prefix contract (callees MAY include single prefix,
   MUST NOT nest) could be codified in a contributor guide. Carried
   over from cycle 9 forward note #3 — still unaddressed but not
   blocking.

These are forward-test-coverage / docs suggestions; none are
findings against cycle 10.
