# Stage 28.8 Cycle 9 — Silent-Failure Audit

**Date**: 2026-05-11
**Commit**: 6968755 (read-only audit). Cycle-9 fix-sweep range
5d1ca24..6968755 (1 fix-sweep commit covering C8-1 closure +
C8-2 closure).
**Scope**: All Helix source — `helixc/bootstrap/*.hx`,
`helixc/frontend/*.py`, `helixc/ir/*.py`, `helixc/backend/*.py`,
`helixc/stdlib/*.hx`. Specifically re-audits the cycle-9
fix-sweep changes for fresh silent windows introduced by the
fixes themselves.
**Trigger**: pre-Stage-29 audit gate — Cycle 9 of 5+ (the gate
re-arms each time a cycle is not clean). Re-audits the same
scope after cycle-9 fixes were landed.
**Strict criterion** (per user directive 2026-05-10): cycle
counts CLEAN only when **zero new findings of ANY severity**
(CRITICAL/HIGH/MEDIUM/LOW). Per the user directive for cycle 9,
findings already documented in cycles 1-8 are NOT re-flagged
unless they CHANGED in the cycle-9 fix-sweep.

**Method**:
1. Read the cycle-8 silent-failures audit (2 findings — C8-1
   MEDIUM, C8-2 LOW).
2. Walked `git show 6968755` — the single cycle-9 fix-sweep
   commit. Read the diff for `check.py`:
   - Added new module-level helper `_emit_env_error(msg)` that
     strips an already-present `helixc:` prefix from `msg`
     before re-prefixing with `helixc:` (C8-2 close).
   - Dropped the `except ImportError as e: ... rc=2` arm so
     genuine internal ImportError falls into the broad
     `except Exception` arm with rc=1 + "please file an issue"
     hint (C8-1 close).
   - Routed FileNotFoundError + family + UnicodeDecodeError
     prints through `_emit_env_error`.
   - Updated the in-method docstring/comment.
3. For each cycle-9 fix's diff, traced data flow forward to
   check whether the fix opened a fresh silent window, left a
   fix incomplete (paper-only), compounded a prior-cycle
   regression, or over-corrected.
4. Direct Python probes against `6968755` HEAD to confirm
   reproducer behavior for the helper's strip-behavior on
   9 message shapes (already-prefixed, unprefixed, leading
   whitespace + prefix, prefix-in-middle, triple-prefix,
   prefix-no-space-after-colon, uppercase-`HELIXC:`, empty
   string, bare `helixc:`).
5. Direct end-to-end probes against `check.py` for: (a) the
   corrupted-internal-import case (`delattr` on
   `monomorphize_structs`) — must now land in the compiler-bug
   arm with rc=1 and emit "please file an issue"; (b) the
   stdlib-strict missing-file path — must now emit a single-
   prefix message; (c) the pre-existing C4-6 regression tests
   (`test_c4_6_filenotfound_not_attributed_as_compiler_bug`,
   `test_c4_6_unicode_decode_error_clean_message`) must still
   pass.
6. Ran the cli regression suite (`pytest helixc/tests/
   test_cli.py`) at `6968755` — 38/38 pass.
7. Searched for any pre-formatted `helixc:` prefixes in raised
   exceptions across the codebase (only `parser.py:1587` in
   strict-stdlib mode currently does it).
8. Searched for any pre-existing regression tests asserting
   either the cycle-9 behaviors (`_emit_env_error` strip, or
   ImportError → compiler-bug arm). None exist.

**Result**: **1 new finding (0 CRITICAL, 0 HIGH, 0 MEDIUM,
1 LOW)** — Cycle 9 NOT clean. The fix-sweep CLOSES BOTH C8-1
and C8-2 correctly:
- `_compatible(...)` unchanged from cycle 8 (cycle 9 didn't
  touch typecheck.py). The cycle-7/8 C7-1 close is preserved.
- The new `_emit_env_error` helper strips a single leading
  `helixc:` prefix (with `.lstrip()` on both sides of the
  literal). End-to-end probe with a fake strict-stdlib
  FileNotFoundError carrying `helixc: stdlib file missing:
  /fake/path` now emits one prefix only.
- `except ImportError` arm dropped: end-to-end probe with
  `del struct_mono.monomorphize_structs` followed by
  `check.main([src_path])` correctly lands in the broad
  `except Exception` arm and prints `helixc: internal error:
  ImportError: cannot import name 'monomorphize_structs' from
  ...` followed by `helixc: this is a compiler bug — please
  file an issue.` with rc=1. Verified at `6968755` HEAD.

The single new finding is the cycle-9 fix-sweep's failure to
add regression tests for the C8-1 + C8-2 closes — repeating
the cycle-8 anti-pattern flagged in cycle-8 "Deferred
observations" as a cycle-9 candidate. This is the cycle-9
"test-coverage gap" for the cycle-9 closes (analogous to the
cycle-7 C7-1 test-coverage gap that cycle 8 inherited and
cycle 9 also did not address).

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

### Finding C9-1: cycle-9 fix-sweep closes C8-1 + C8-2 but adds zero regression tests, so a future refactor can re-introduce either silent-failure window with no test detecting it

**Location**:
- helixc/check.py:246-255 (the new `_emit_env_error` helper —
  no test exercises its strip behavior on the
  already-prefixed / unprefixed / whitespace-prefix / no-
  space-after-colon cases).
- helixc/check.py:284-318 (the new outer-except topology that
  drops the ImportError arm — no test exercises the
  corrupted-internal-import case `delattr` →
  `except Exception` rc=1 + "please file an issue").
- helixc/check.py:299-305 (the cycle-8 outer arms now routed
  through `_emit_env_error` — no test asserts no-double-prefix
  for the strict-stdlib FileNotFoundError end-to-end).
- helixc/tests/test_typecheck.py (the place where the cycle-5
  `test_c4_6_filenotfound_*` regressions live and where the
  parallel cycle-9 regressions would naturally live).
- helixc/tests/test_cli.py (alternate location for end-to-end
  exit-code regression tests).
**Severity**: LOW
**Category**: cycle-9-fix-introduced verification gap — closes
behavior without locking it down with regression tests
**Stage**: 28.8 cycle-9 commit 6968755 (the entire fix-sweep)

**Description**:
The cycle-9 fix-sweep closes both cycle-8 findings substantively
correctly (verified end-to-end via direct Python probes). But
the commit adds NO regression tests asserting either close:

- **C8-1 close (drop ImportError arm)**: no test asserts that
  an `ImportError` raised inside `_main_inner` lands in the
  broad `except Exception` arm with rc=1 + the "please file an
  issue" diagnostic. A future cycle could re-add the
  `except ImportError as e: ... rc=2` arm (e.g., to handle a
  legitimate user-env partial-install case) and silently
  re-introduce the C8-1 mis-attribution without any test
  detecting it.
- **C8-2 close (`_emit_env_error` strip-prefix helper)**: no
  test asserts that `_emit_env_error("helixc: foo")` emits a
  single-prefix `"helixc: foo\n"`. A future refactor could
  refactor the helper into a simpler `print(f"helixc: {msg}")`
  (e.g., re-inlining the helper, or moving the strip into a
  different call site) and silently re-introduce the
  `helixc: helixc: ` double-prefix without any test detecting
  it.
- **Strict-stdlib end-to-end**: no test asserts the
  end-to-end "set `HELIXC_STDLIB_STRICT=1`, delete stdlib file,
  run check.main(), assert single-prefix in stderr". This is
  the actual reproducer the cycle-8 audit captured for C8-2.

The cycle-8 audit doc's "Deferred / out-of-scope observations"
section explicitly listed cycle-9 candidate items:
- C8-1 recommendation 4: "Add a regression test that simulates
  the corrupted-import case (`monkeypatch.delattr` on a
  struct_mono symbol) and asserts the exit code + message".
- C8-2 recommendation 4: "Add a regression test that asserts
  no double-prefix in the stdlib-strict missing-file path".

Neither test was added. The cycle-9 commit message states:
"Tests: 264 targeted tests pass (autodiff + typecheck +
struct_mono + pytree + autodiff_reverse + cli)". Those are all
PRE-EXISTING tests — none assert any of the new cycle-9
behaviors. Verified by `grep -rn "_emit_env_error\|c8_1\|c8_2"
helixc/tests/` returning zero hits and `grep -rn "stdlib file
missing\|please file an issue" helixc/tests/` returning only
the cycle-5 C4-6 docstring mention (line 1535) and the
cycle-? stdlib parser test (test_parser.py:772).

This is the same anti-pattern flagged in the cycle-8 audit
("Test-coverage gap for C7-1 close: cycle 8 closed C7-1 but
added no regression test asserting `_compatible(TyMemTier,
TyVar) is False`..."). Cycle 9 inherited that gap AND
introduced its own parallel gap for C8-1 + C8-2.

The new gap is LOW severity (not MEDIUM) because:
1. The cycle-9 fix is substantively correct — both C8-1 and
   C8-2 close their respective windows in production code, so
   the user-facing silent-failure behavior is fixed.
2. The gap is a verification debt, not an active silent
   failure. A future refactor would have to actively undo the
   fix to re-introduce the silent failure.
3. The carryover C7-1 test-coverage gap is also LOW
   (cycle-8-deferred) and not re-flagged here per the user
   directive — but cycle 9 introducing its OWN parallel gap
   IS a CHANGE in the surface area and is therefore a new
   cycle-9 finding under the strict criterion.

**Hidden errors**:
- A future cycle that re-adds the `except ImportError as e: ...
  rc=2` arm (e.g., for a legitimate user-env case like
  `helixc.backend.ptx` missing on a CPU-only install) would
  silently re-introduce the C8-1 mis-attribution for the 18
  lazy in-`_main_inner` imports. No test would fail.
- A future cycle that re-inlines `_emit_env_error` back to
  `print(f"helixc: {e}")` (e.g., as part of a code-cleanup
  pass removing the helper because it has "only two call
  sites") would silently re-introduce the C8-2 double-prefix
  for the strict-stdlib path. No test would fail.
- A future cycle that adds a NEW pre-formatted-prefix raise
  somewhere (e.g., a new env-check that raises
  `PermissionError("helixc: write permission denied: ...")`)
  would benefit from the strip behavior — but the lack of a
  test asserting "strip behavior covers PermissionError too"
  means a future refactor of the helper to only-strip-for-
  FileNotFound could silently re-introduce double-prefix for
  that new path.
- A new contributor reading `check.py` would see the helper
  but not understand from any test what its strip-behavior
  contract is. The docstring is informative but a test would
  be load-bearing documentation.
- The cycle-9 commit message claims "264 targeted tests pass"
  which can mislead a reviewer scanning the commit into
  believing the C8-1/C8-2 behaviors are now test-covered.
  They are not.

**Recommendation**:
1. **PREFERRED**: add four regression tests to
   `helixc/tests/test_typecheck.py` (or
   `helixc/tests/test_cli.py` — wherever the cycle-5 C4-6
   regressions live):
   - `test_c8_1_import_error_attributed_as_compiler_bug` —
     `monkeypatch.delattr(struct_mono, "monomorphize_structs")`,
     run `check.main([src_path])`, assert `rc == 1` and
     `"compiler bug" in captured.err` and
     `"please file an issue" in captured.err`.
   - `test_c8_2_emit_env_error_strips_existing_prefix` —
     direct unit test of the helper:
     `_emit_env_error("helixc: foo")` →
     `captured.err == "helixc: foo\n"` (single prefix).
   - `test_c8_2_emit_env_error_adds_prefix_when_absent` —
     `_emit_env_error("foo")` →
     `captured.err == "helixc: foo\n"`.
   - `test_c8_2_stdlib_strict_missing_file_single_prefix` —
     end-to-end: `monkeypatch.setattr(parser_mod,
     "_merge_stdlib", lambda *a, **kw: raise FileNotFoundError(
     "helixc: stdlib file missing: /fake"))`, run
     `check.main([src_path])`, assert `rc == 2` and
     `captured.err.count("helixc:") == 1`.
2. **ALTERNATIVE**: add a single property-style test that
   asserts the helper is idempotent in the sense
   `_emit_env_error("helixc: " + x) == _emit_env_error(x)` for
   the realistic message shapes. Simpler but loses the
   end-to-end / exit-code coverage.
3. Also add the cycle-8 carryover regression tests for C7-1
   (`test_c7_1_compatible_tymem_tier_tyvar_rejects` etc.) to
   close the C7-1 test-coverage gap that cycle 8 left open.
   Not strictly part of cycle 9's fix, but combining the two
   coverage gaps into one fix batch makes the cycle-10
   re-audit cheaper.

**Trap-id**: n/a (check.py CLI dispatch + helper, no trap-id).

---

## Cycle 9 fix-sweep re-verification

Each cycle-9 fix-sweep change was inspected for paper-only
fixes, silent windows, false positives, and state-leak. The
cycle-9 fix-sweep landed as a single commit (6968755) covering
both cycle-8 findings.

| fix-sweep label | What changed | Audit-doc cross-ref | C9 verdict |
|---|---|---|---|
| C8-1 close (drop ImportError arm) | Removed cycle-8 lines 292-296 `except ImportError as e: print(f"helixc: import error: {e}") rc=2`; updated the surrounding comment block to remove the "missing modules" language and note that ImportError now falls into the compiler-bug arm | C8-1 (cycle-8 silent-failure MEDIUM) | **closed** — `del struct_mono.monomorphize_structs` followed by `check.main([src_path])` lands in `except Exception` arm with rc=1 and emits `helixc: internal error: ImportError: ...` + `helixc: this is a compiler bug — please file an issue.` Verified end-to-end. |
| C8-2 close (`_emit_env_error` helper) | Added new module-level helper at lines 246-255 that `.lstrip()`s, checks `startswith("helixc:")`, strips the prefix + a trailing `.lstrip()`, then re-emits `f"helixc: {text}"`. Routed both the FileNotFoundError-family arm AND the UnicodeDecodeError arm through it | C8-2 (cycle-8 silent-failure LOW) | **closed for the realistic case** — pre-formatted `helixc: stdlib file missing: ...` from `parser.py:1587` now emits a single-prefix message; opens C9-1 LOW for the test-coverage debt. |

### Specific re-verifications from the audit instructions

- **C8-1 close (ImportError → compiler-bug arm)**: probed
  via direct Python at `6968755`:
  ```
  $ python -c "
  import helixc.frontend.struct_mono as sm
  del sm.monomorphize_structs
  from helixc.check import main
  import sys, tempfile
  with tempfile.NamedTemporaryFile(mode='w', suffix='.hx',
                                     delete=False) as f:
      f.write('fn main() -> i32 { 0 }')
      src = f.name
  rc = main([src])
  print('rc=', rc)
  "
  helixc: internal error: ImportError: cannot import name
    'monomorphize_structs' from 'helixc.frontend.struct_mono'
    (...)
  helixc: this is a compiler bug — please file an issue.
  -- helixc-check: <tmp>
     parse:    OK  (1 fns, 1 items)
     typecheck: OK
  rc= 1
  ```
  Confirmed: rc=1 (was rc=2 pre-cycle-9); the "please file an
  issue" hint is present (was absent pre-cycle-9). Correct.

- **C8-2 close (`_emit_env_error` strip on the 9 message
  shapes)**: probed via direct unit calls on the helper:
  | Input | Output |
  |---|---|
  | `helixc: stdlib file missing: /foo` | `helixc: stdlib file missing: /foo\n` |
  | `No such file: /foo` | `helixc: No such file: /foo\n` |
  | `  helixc:  doubled-with-leading-ws` | `helixc: doubled-with-leading-ws\n` |
  | `mention of helixc: in middle` | `helixc: mention of helixc: in middle\n` (correct — `startswith` only matches at offset 0) |
  | `helixc: helixc: triple` | `helixc: helixc: triple\n` (one level stripped → output still has one redundant prefix; not a realistic shape — no callee emits this) |
  | `helixc:no-space-after-colon` | `helixc: no-space-after-colon\n` |
  | `HELIXC: uppercase-prefix` | `helixc: HELIXC: uppercase-prefix\n` (correct — strip is case-sensitive; no callee emits uppercase) |
  | `` (empty) | `helixc: \n` |
  | `helixc:` (bare) | `helixc: \n` |
  All shapes match expected behavior for the realistic-message
  population (single-pre-prefixed). The triple-prefix case
  would leave one redundant prefix, but no callee in the
  current codebase emits triple-prefixed messages — verified
  via `grep -rn "helixc:.*helixc:" helixc/` returning only the
  docstring of `_emit_env_error` itself (lines 248, 251).

- **C8-1 / C8-2 end-to-end check**: the pre-existing
  cycle-5 C4-6 regressions
  (`test_c4_6_filenotfound_not_attributed_as_compiler_bug` and
  `test_c4_6_unicode_decode_error_clean_message`) both pass at
  `6968755` (verified). These tests don't exercise the new
  strip behavior (they use the standard Python
  `FileNotFoundError(2, "No such file or directory",
  "missing.something")` shape which has no `helixc:` prefix)
  so they don't test the actual cycle-9 fix surface, but they
  confirm the cycle-9 fix didn't regress the cycle-5 fix.

- **CLI regression**: ran `pytest helixc/tests/test_cli.py`
  at `6968755` — 38/38 pass. No existing test fails due to
  the cycle-9 refactor.

- **Test-coverage gap**: cycle 9 closed C8-1 + C8-2 but added
  NO regression tests for either close — captured as C9-1
  LOW above.

### Carryover findings status (cycles 1-8)

The cycle-9 fix-sweep CLOSED C8-1 (MEDIUM) and C8-2 (LOW). Did
NOT re-attempt the following still-open carryover findings:

| Carryover | Severity | Cycle-9 status |
|---|---|---|
| audit-C4-1 (D2 Call-RHS i32 SIGILL) | CRITICAL | **still open** — cycle-9 did not address; deferred per cycle-6's commit message pending parse-time constant folding or a typecheck pass before closure capture |
| audit-C4-4 (D9 paper-only) | HIGH | **still open** — not addressed |
| audit-C4-8 deferred (check.py doesn't call fn-mono) | LOW | **still open** — not addressed |
| monomorphize_safe docstring drift (cycle-6 deferred) | (not a finding) | **still open** — docstring still suggests callers MAY ignore diags; only caller now aborts |
| D-vs-Quote diagnostic text (cycle-7 deferred) | (not a finding) | **still open** — Quote-wrapped case still emits "(one side D-wrapped, other bare)" |
| C7-1 test-coverage gap (cycle-8 deferred) | (not a finding then; now elevated by C9-1 parallel) | **still open** — cycle 9 did not add the `_compatible(TyMemTier, TyVar) is False` regression tests. Not re-flagged here (carryover, did not CHANGE in cycle 9). Captured under C9-1 recommendation 3 as a fix-batch combine target. |
| C8-1 (cycle-8 MEDIUM) | MEDIUM | **CLOSED by cycle 9** (this commit). |
| C8-2 (cycle-8 LOW) | LOW | **CLOSED by cycle 9** (this commit). |

These are NOT re-flagged as new cycle-9 findings per the user
directive (already documented in cycles 1-8, did not CHANGE
in cycle 9's fix-sweep). They remain in the open-findings
ledger.

### Specific items checked clean in cycle 9 (no new finding)

- The `_emit_env_error` helper's `.lstrip()` + `startswith(
  "helixc:")` + `[len("helixc:"):].lstrip()` sequence correctly
  handles the realistic-message shape (one-pre-prefixed,
  optionally with whitespace before the literal or between the
  colon and content). The case-sensitive match is appropriate
  (no callee emits `HELIXC:`).
- The cycle-9 commit preserves the broad `except Exception`
  arm at lines 306-318 — both the `helixc: internal error:
  <Type>: <msg>` and the `helixc: this is a compiler bug —
  please file an issue.` prints are unchanged. Genuine
  pipeline-internal exceptions still get the correct attribution.
- The cycle-9 commit preserves the finally-drain wrap at lines
  319-337 (broad `except Exception as drain_e`). The cycle-5
  C4-6 rationale (drain failures should not mask primary
  failure) is preserved.
- The inner-print sites at `_main_inner:356` (argparse error
  loop `print(f"helixc: {e}")`) and `_main_inner:363`
  (`print(f"helixc: file not found: {path}")`) do NOT route
  through `_emit_env_error`. This is fine — those prints emit
  author-controlled message bodies (argparse error strings or
  the fixed literal `file not found:`), neither of which is
  pre-prefixed with `helixc:`. Routing them through the helper
  would be defensive but not load-bearing. Not a finding.
- The AD-warning drain print at `_drain_ad_warnings:240`
  (`print(f"     helixc: {w}", file=sys.stderr)`) emits
  warnings collected via `take_diff_warnings()` from
  `helixc.frontend.autodiff`. The `w` values are author-
  controlled strings appended in-pass (e.g.,
  `B13: widening from ...`); none start with `helixc:`. Not
  a finding.
- The cycle-7 / cycle-8 `_compatible` close (G2 carve-out
  drop) is unchanged in cycle 9. The 105 typecheck tests at
  test_typecheck.py still pass (cycle 9 didn't touch
  typecheck.py at all).
- The cycle-5 C4-6 regression tests
  (`test_c4_6_filenotfound_not_attributed_as_compiler_bug`,
  `test_c4_6_unicode_decode_error_clean_message`) both still
  pass — cycle-9's refactor of the print routes through the
  new helper did NOT regress the cycle-5 fix.
- The exit-code semantics post-cycle-9 are now:
  - rc=0: success.
  - rc=1: compile error OR compiler bug (broad-Exception arm).
  - rc=2: bad invocation OR user-env error (FileNotFoundError /
    PermissionError / IsADirectoryError / NotADirectoryError /
    UnicodeDecodeError / explicit-file-not-exists / argparse
    failure).
  Genuine ImportError inside `_main_inner` now correctly maps
  to rc=1 (was rc=2 in cycle 8). The exit-code semantics are
  now consistent with the cycle-4 audit-C4-7 recommendation 1
  (compiler-bug arm includes ImportError, AttributeError,
  KeyError, IndexError, AssertionError, TypeError,
  RuntimeError, ValueError).

---

## Cross-stage interactions checked

- **C9 `_emit_env_error` helper + future pre-formatted
  callees**: only `parser.py:1587` currently raises a pre-
  prefixed FileNotFoundError. The helper is generic enough to
  handle PermissionError / IsADirectoryError / etc. with the
  same pre-prefixing pattern, but no test asserts this — see
  C9-1. A future contributor adding a new pre-prefixed raise
  in (e.g.) `lexer.py` would benefit from the helper but
  needs to verify by inspection that the raise lands in one of
  the FileNotFound-family / UnicodeDecode arms (not the broad
  Exception arm, which does NOT route through the helper).
- **C9 ImportError fallthrough + cycle-8 C8-1 attribution
  semantics**: the cycle-8 audit's C8-1 description noted that
  the cycle-5 audit-C4-7 recommendation 1 listed
  `AttributeError, KeyError, IndexError, AssertionError,
  TypeError, RuntimeError, ValueError` as compiler-bug-arm
  members but did NOT include `ImportError` either way. Cycle 9
  has now placed ImportError on the compiler-bug side
  (correctly, per the cycle-8 recommendation 1 PREFERRED).
  The semantic is now: ALL `_main_inner`-reachable
  ImportError → compiler bug, rc=1, "please file an issue". A
  future scenario where a legitimate user-env ImportError
  could reach `_main_inner` (e.g., a plugin loader) would need
  a finer-grained classification — but no such scenario
  currently exists.
- **C9 `_emit_env_error` strip + downstream log scraping**:
  CI tools scraping stderr for `^helixc: ` lines see a uniform
  single-prefix format post-cycle-9. Pre-cycle-9 some lines
  had `helixc: helixc: ` which broke prefix-anchored regex
  matches. Cycle-9 normalizes this. No silent surface.
- **C9 broad-Exception arm + ImportError sequencing**: Python
  exception-chain semantics: `except Exception` post-cycle-9
  now catches ImportError because ImportError is a subclass of
  Exception and no earlier arm matches. The compiler-bug print
  `helixc: internal error: ImportError: <msg>` correctly
  includes the type name. Verified.

---

## Deferred / out-of-scope observations (NOT new findings; cycle-10 candidates)

- **C9-1 own remedy**: add 4 regression tests
  (`test_c8_1_import_error_attributed_as_compiler_bug`,
  `test_c8_2_emit_env_error_strips_existing_prefix`,
  `test_c8_2_emit_env_error_adds_prefix_when_absent`,
  `test_c8_2_stdlib_strict_missing_file_single_prefix`) — see
  C9-1 recommendation 1. **HIGHEST-PRIORITY ITEM** for cycle 10
  given C9-1 is the only blocker to a clean cycle.
- **C7-1 test-coverage gap (cycle-8 carryover, cycle-9
  unaddressed)**: cycle 9 did not add the 4 `_compatible`
  TyMemTier/TyVar/TySize regression tests requested in the
  cycle-7 audit and re-flagged in cycle-8's "Deferred"
  section. Recommend the cycle-10 fix batch combine these
  with the C9-1 close — net 8 new regression tests for a
  one-commit cycle-10 fix-sweep.
- **Carryover audit-C4-1 (D2 Call-RHS i32 SIGILL)**: still
  open CRITICAL. Cycle 9 did not address. The cycle-6 commit's
  deferral rationale still applies. **HIGHEST-PRIORITY ITEM**
  for cycle 10 — the still-open CRITICAL is the strongest
  obstacle to the 5-clean-cycle gate.
- **Carryover audit-C4-4 (D9 paper-only)**: still open HIGH.
  Not addressed.
- **Carryover audit-C4-8 (check.py doesn't call fn-mono)**:
  still open LOW. Not addressed.
- **monomorphize_safe docstring drift**: still open (cycle-6
  deferred). Docstring suggests callers MAY ignore diags; the
  only caller (x86_64.py) now aborts. Could be addressed by
  rewriting the docstring or refactoring `Monomorphizer.run`
  per-instance.
- **D-vs-Quote diagnostic text**: still open (cycle-7
  deferred). Quote-wrapped case still emits "(one side D-
  wrapped, other bare)" — imprecise but not a CHANGED
  behavior. Could be generalized in a cycle-10 housekeeping
  batch.
- **Cycle-9 commit message coverage**: the cycle-9 commit
  message accurately describes both C8-1 and C8-2 closes, but
  the "Tests: 264 targeted tests pass" claim is misleading —
  none of those 264 tests assert the new cycle-9 behaviors.
  A reader scanning the commit might infer test coverage that
  doesn't exist. Recommend the cycle-10 commit message
  explicitly distinguish "tests asserting the new fix" vs
  "tests confirming no regression in pre-existing fixes".

---

## Summary

| #    | Severity | Location                                                    | Finding                                                                                                            |
|------|----------|-------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| C9-1 | LOW      | check.py:246-255 + check.py:284-318 + helixc/tests/         | cycle-9 fix-sweep closes C8-1 + C8-2 substantively correctly but adds zero regression tests, so a future refactor can re-introduce either silent-failure window with no test detecting it (verification debt) |

**Total: 1 new finding (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW).**

---

## Cycle 9 status

**Cycle 9 NOT clean.** Per the strict criterion (zero findings
of ANY severity), the 1 LOW new finding BLOCKS the cycle-9
clean determination.

### Stop-the-line determination: **NO**

C9-1 is LOW (verification debt — the production code is fixed;
the gap is in the regression-test surface). The cycle-9 fix-
sweep made strong substantive progress — closed both cycle-8
findings (C8-1 MEDIUM + C8-2 LOW), net -2 closed, +1 new
(LOWER severity than either closed finding). Severity trend is
monotone-decreasing for the first time since cycle 7 (cycle 8
broke the monotone-decrease; cycle 9 restores it).

C9-1 is a mechanical-fix:
- Add 4 regression tests (1 per behavior + 1 end-to-end).
  ~30-line diff across `helixc/tests/test_typecheck.py` or
  `test_cli.py`. No production-code change required.

Recommend addressing in the cycle-10 fix batch alongside:
1. The C7-1 test-coverage gap (cycle-8 carryover) — 4 more
   tests, combinable with C9-1 in one commit.
2. The still-open carryover audit-C4-1 (CRITICAL — top
   priority).

### Cycle 9 → NEW FINDINGS COUNT for the strict-clean gate: 1 (0 CRITICAL + 0 HIGH + 0 MEDIUM + 1 LOW) — clean-counter remains at 0.

### Severity trend across cycles

- Cycle 1: 13 findings (3 HIGH, 5 MEDIUM, 5 LOW).
- Cycle 2: 6 findings (1 HIGH, 4 MEDIUM, 1 LOW).
- Cycle 3: 6 findings (0 HIGH, 4 MEDIUM, 2 LOW).
- Cycle 4: 8 findings (1 CRITICAL, 2 HIGH, 4 MEDIUM, 1 LOW).
- Cycle 5: 4 findings (0 CRITICAL, 0 HIGH, 2 MEDIUM, 2 LOW).
- Cycle 6: 1 finding (0 CRITICAL, 0 HIGH, 1 MEDIUM, 0 LOW).
- Cycle 7: 1 finding (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW).
- Cycle 8: 2 findings (0 CRITICAL, 0 HIGH, 1 MEDIUM, 1 LOW).
- Cycle 9: 1 finding (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW). ← here

Trend: severity-monotone-decrease restored. The new LOW is the
secondary effect of closing 1 MEDIUM + 1 LOW from cycle 8. Net
delta on open severity-weighted ledger: -1 MEDIUM closed,
-1 LOW closed, +1 LOW opened. Cycle 9 is the cleanest cycle
since cycle 7 by severity, with the lowest finding count tied
with cycles 6 and 7.

### Estimated remaining open findings going into cycle 10

- Cycle 1: 13 new (all fixed → 0 open).
- Cycle 2: 6 new (all fixed → 0 open).
- Cycle 3: 6 new (all fixed → 0 open).
- Cycle 4 silent-failure: 8 new — 6 closed by cycles 5-9
  (audit-C4-7 was closed by cycle 8). 2 still open:
  audit-C4-1 CRITICAL, audit-C4-4 HIGH. Net: 2 still open.
- Cycle 4 type-design (sibling audit): partial close — E3
  closed via C5-4/F3; E1 closed via C5-2/F1 mechanism; others
  unchanged.
- Cycle 4 codereview (sibling audit): 0 new (was already clean).
- Cycle 5 silent-failure: 4 new — all 4 CLOSED by cycle 6.
- Cycle 6 silent-failure: 1 new (C6-1) — CLOSED for all
  positions by cycles 7-8. Net: 0 open.
- Cycle 6 type-design: 2 new (G1, G2) — both CLOSED.
- Cycle 7 silent-failure: 1 new (C7-1) — CLOSED by cycle 8.
- Cycle 8 silent-failure: 2 new (C8-1 MEDIUM, C8-2 LOW) —
  both CLOSED by cycle 9.
- Cycle 9 silent-failure: 1 new (C9-1 LOW open).
- Prior audits (stage 5-6 + 7-8 + 9-16): ~20 still-open
  (unchanged going into cycle 9 — cycle 9 didn't touch them).
- Cycle 9 net: 20 + 1 + (deferred type-design partial) + 2 =
  **≥23 open findings** going into cycle 10. (Net -1 from
  cycle 8: cycle 9 closed C8-1 + C8-2, opened C9-1; net delta
  -1.)

Recommend prioritizing in this order for the cycle-10 fix
batch:
1. **C9-1** (LOW — add 4 regression tests for C8-1 + C8-2
   closes; ~30-line diff; closes the only cycle-9 blocker
   to a clean cycle).
2. **C7-1 test-coverage gap** (carryover housekeeping — add 4
   regression tests for `_compatible` TyMemTier/TyVar/TySize
   matrix; combinable with C9-1 in one commit).
3. **audit-C4-1** (CRITICAL — still-open from cycle 4;
   highest-priority unaddressed-CRITICAL; deferred in cycles
   6-9; the carryover deadline approaches as cycles 1-9
   progress).
4. **audit-C4-4** (HIGH — D9 paper-only).
5. **monomorphize_safe docstring drift** (housekeeping).
6. **D-vs-Quote diagnostic text** (housekeeping).
7. **Cycle-10 commit message coverage** (housekeeping —
   explicit "tests for new fix" vs "regression tests confirm
   no break" distinction).

After this batch lands, cycle 10 should re-audit. The "5 clean
cycles before Phase 0 deprecation" goal requires the strict
criterion (zero findings of any severity) to be met for
5 CONSECUTIVE cycles — cycle 9 is the 9th cycle and is NOT
clean (1 LOW), so the clean-counter remains at 0. Cycle 9
has 1 mechanical-fix LOW finding + 1 still-open CRITICAL
carryover; the strongest realistic path to a clean cycle 10
is: address C9-1 (4 small test additions), address C7-1 test-
coverage gap (4 more small test additions, combinable into
one commit), and address audit-C4-1 (CRITICAL — the largest
single risk to the clean gate). If executed cleanly, cycle 10
has a credible shot at being the first clean cycle of the
5-clean gate — the production-code surface is now narrower
than it has been since cycle 4.
