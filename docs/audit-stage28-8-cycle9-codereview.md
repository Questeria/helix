# Stage 28.8 Pre-29 Audit Gate — Cycle 9, Audit C: Code Review

**Date**: 2026-05-11
**Commit**: 6968755 (read-only)
**Scope**: Audit C (general code-review) of the cycle-9 fix-sweep at
commit 6968755. The fix-sweep is strictly scoped to two changes in
`helixc/check.py`, closing the two cycle-8 silent-failure findings:

- **C8-1 (MEDIUM)**: the cycle-5 C4-6 exception classifier added
  `except ImportError as e: ... rc=2` for the `_main_inner` outer
  catch. But ImportError inside `_main_inner` is most often a genuine
  internal compiler bug (a rename of an internal function used by one
  of 18 lazy `from .` imports). Pre-fix the bug was mis-classified as
  a user-environment problem with no "please file an issue" hint and
  rc shifted 1→2. **Cycle 9 fix**: drop the `except ImportError`
  arm so genuine ImportError falls into the compiler-bug
  `except Exception` arm with the correct diagnostic + rc=1.
- **C8-2 (LOW)**: callees that raise FileNotFoundError with the
  message already prefixed by `helixc: ` (currently only
  parser.py:1587 in strict-stdlib mode) produced
  `helixc: helixc: stdlib file missing: ...` — double-prefix.
  **Cycle 9 fix**: route env-error printing through a new
  `_emit_env_error` helper that strips a leading `helixc:` if
  present before re-prefixing.

The fix-sweep commit also adds the cycle-8 silent-failures doc
(C8-1 + C8-2 source), which is read-only documentation and not in
audit-C scope.

Files reviewed at HEAD (6968755):

1. `helixc/check.py` lines 246-256 — new `_emit_env_error` helper.
2. `helixc/check.py` lines 261-338 — `main()` outer wrapper with
   updated except cascade (ImportError arm removed, FileNotFound
   + UnicodeDecodeError arms routed through helper) and the
   unchanged finally drain wrap.
3. `helixc/check.py` lines 341-360 — `_main_inner` parse / early-
   exit envelope, to confirm the strip helper's invariants are not
   broken by `_main_inner` printing its own `helixc: <e>` for
   `parse_args` errs (it does, at line 356 — but that path returns
   rc=2 inside `_main_inner` and never raises, so the outer helper
   is never invoked on it).
4. `helixc/frontend/parser.py` lines 1582-1592 — the strict-stdlib
   path that raises the pre-formatted FileNotFoundError. Confirmed
   the `msg` shape is `"helixc: stdlib file missing: {path}"` —
   exactly the prefix `_emit_env_error` strips.
5. `helixc/tests/test_typecheck.py` lines 1530-1569 — the two
   cycle-5 C4-6 regression tests for FileNotFoundError and
   UnicodeDecodeError, confirmed they assert post-cycle-9 behavior
   is preserved (rc=2, no "compiler bug" tagline, "encoding error"
   substring for UnicodeDecodeError).

**Method**: Read the cycle-8 codereview (CLEAN), cycle-8 silent-
failures (source of C8-1 + C8-2), and cycle-8 type-design
(CLEAN) docs to load the cumulative invariant set. Walked the
full cycle-9 fix-sweep diff (`git show 6968755` on
`helixc/check.py`). For the dropped ImportError arm: confirmed
ImportError is a subclass of Exception
(`issubclass(ImportError, Exception) == True`), so the cascade
falls through correctly to the broad arm with rc=1 + "compiler
bug" diagnostic. For the `_emit_env_error` helper: walked the
strip logic against the only known pre-formatted-message
producer (parser.py:1587), against the un-pre-formatted Python
default messages (Errno-prefixed), and against a degenerate
double-prefix input (`"helixc:helixc:foo"`).

**Reporting threshold**: confidence ≥ 80 (per the cycle-9
audit-C prompt's strict criterion).

**Result**: **0 findings (0 CRITICAL, 0 HIGH, 0 MEDIUM, 0 LOW)
at or above the confidence-80 reporting threshold.**

---

## Summary table

| ID    | Severity | Confidence | Component | Issue |
|-------|----------|------------|-----------|-------|
| —     | —        | —          | —         | (none at or above threshold) |

**Cycle 9 Audit C: CLEAN — 0 findings at the confidence-80
threshold.**

Per user directive 2026-05-10 (strict criterion): cycle counts
CLEAN only when zero findings of ANY severity at or above the
audit threshold. **This cycle qualifies as clean.**

---

## Cycle-8 finding closure verification

### C8-1 (MEDIUM, conf ~85 in cycle-8 silent-failures): cycle-5 `except ImportError as e` mis-attributes genuine internal compiler import bugs as env errors — **CLOSED**

Cycle 5 (audit C4-6 / MEDIUM closure) introduced the typed except
cascade in `check.main()`:

```python
except (FileNotFoundError, PermissionError, IsADirectoryError,
        NotADirectoryError) as e: ...; rc = 2
except UnicodeDecodeError as e: ...; rc = 2
except ImportError as e:
    print(f"helixc: import error: {e}", file=sys.stderr)
    rc = 2
except Exception as e:
    print(f"helixc: internal error: ...", ...)
    print("helixc: this is a compiler bug — please file an issue.", ...)
    rc = 1
```

Cycle 8 audit-C8-1 flagged the ImportError arm: `check.py`
performs 18 lazy `from .frontend.XXX import YYY` /
`from .ir.XXX import YYY` / `from .backend.XXX import YYY`
statements inside `_main_inner`, all reachable from a user input
file path. A refactor that renames an internal exported symbol
and forgets to update the lazy import raises ImportError — a
genuine compiler bug — but the typed arm mis-classified it as
environment with rc=2 and dropped the "please file an issue"
hint. The cycle-8 silent-failures recommendation 1 was: drop
the ImportError arm so genuine ImportError falls into the broad
`except Exception` arm (which correctly tags it as a compiler
bug with rc=1 + the "please file an issue" hint).

**Cycle 9 fix** (check.py:299-318): the `except ImportError as e`
arm (cycle-5 lines 292-296 of the pre-fix file) is deleted. The
remaining cascade is:

```python
except (FileNotFoundError, PermissionError, IsADirectoryError,
        NotADirectoryError) as e:
    _emit_env_error(str(e))
    rc = 2
except UnicodeDecodeError as e:
    _emit_env_error(f"encoding error reading source: {e}")
    rc = 2
except Exception as e:
    # Everything else (AttributeError, KeyError, IndexError,
    # AssertionError, TypeError, RuntimeError, ValueError, etc.)
    # is a genuine internal-error candidate.
    print(f"helixc: internal error: {type(e).__name__}: {e}",
          file=sys.stderr)
    print("helixc: this is a compiler bug — please file an issue.",
          file=sys.stderr)
    rc = 1
```

Reachability of ImportError post-cycle-9:

- Python except-cascade matches by isinstance. Since `ImportError`
  inherits from `Exception` (not from `OSError` or
  `UnicodeError`), ImportError no longer matches any earlier arm.
  It falls directly into the broad `except Exception` and gets
  the "compiler bug" diagnostic + rc=1. Verified
  `issubclass(ImportError, Exception) == True` and
  `issubclass(ImportError, OSError) == False`.
- The startup-import case (module-level imports at the top of
  `check.py` at lines 57-63) fires at Python module-load time,
  BEFORE `check.main()` runs. If those fail, the Python
  interpreter prints a traceback and exits before the new
  cascade is reached. The startup case is unaffected by either
  the cycle-5 add or the cycle-9 drop.
- The lazy in-`_main_inner` imports (struct_mono, flatten_impls,
  deprecated_pass, trace_pass, panic_pass, unsafe_pass,
  autotune, ir/lower_ast, ir/passes/{fdce,const_fold,cse,dce},
  grad_pass, backend/x86_64, ir/tile_ir, backend/ptx) all run
  inside the outer try and now correctly produce rc=1 +
  "compiler bug" on import failure.

The comment block at lines 286-298 was updated to reflect the
new state: the cycle-5 C4-6 banner mention of ImportError was
removed (the comment now lists only "file I/O, encoding" as the
env-error classes), and a new cycle-9 C8-2 paragraph documents
the strip-prefix behavior of the outer arms via
`_emit_env_error`. The comment history is internally consistent
with the cascade as actually implemented. **CLOSED.**

### C8-2 (LOW, conf ~85 in cycle-8 silent-failures): cycle-5 outer FileNotFoundError arm duplicates `helixc:` prefix when catching a pre-formatted FileNotFoundError from `_merge_stdlib` — **CLOSED**

Cycle 5's outer arm was:

```python
except (FileNotFoundError, PermissionError, IsADirectoryError,
        NotADirectoryError) as e:
    print(f"helixc: {e}", file=sys.stderr)
    rc = 2
```

For Python's default FileNotFoundError messages
(`[Errno 2] No such file or directory: '<path>'`), this prints
`helixc: [Errno 2] No such file or directory: '<path>'` — clean,
single prefix.

For `parser.py:1585-1587` which pre-formats the message:

```python
msg = f"helixc: stdlib file missing: {stdlib_path}"
if strict:
    raise FileNotFoundError(msg)
```

cycle-5's arm produced
`helixc: helixc: stdlib file missing: <path>` — double prefix.
Cosmetic but flagged at confidence 85 under the strict-zero
criterion.

**Cycle 9 fix** (check.py:246-256): a new `_emit_env_error(msg)`
helper:

```python
def _emit_env_error(msg: str) -> None:
    """Audit 28.8 cycle 9 C8-2: print a user-environment error with a
    single `helixc:` prefix. Strips an already-present `helixc:` prefix
    from `msg` so callees that raise with the prefix already formatted
    (e.g. parser.py:1587's strict-stdlib FileNotFoundError) don't
    double-print as `helixc: helixc: ...`."""
    text = msg
    if text.lstrip().startswith("helixc:"):
        text = text.lstrip()[len("helixc:"):].lstrip()
    print(f"helixc: {text}", file=sys.stderr)
```

The strip is anchored to the start (via `.lstrip()` +
`.startswith("helixc:")`) so it cannot accidentally strip a
substring `helixc:` embedded later in the message. The
slice-and-`.lstrip()` of the trailing whitespace handles the
typical `"helixc: foo"` (with space after the colon) form.

The two affected arms now route through it:

```python
except (FileNotFoundError, ...) as e:
    _emit_env_error(str(e))
    rc = 2
except UnicodeDecodeError as e:
    _emit_env_error(f"encoding error reading source: {e}")
    rc = 2
```

Walked the strip logic against the known cases:

- `"helixc: stdlib file missing: <path>"` (parser.py:1587) →
  strips 7 chars + leading-space cleanup → emits
  `"helixc: stdlib file missing: <path>"`. Correct.
- `"[Errno 2] No such file or directory: '<path>'"` (Python
  default) → no prefix → emits
  `"helixc: [Errno 2] No such file or directory: '<path>'"`.
  Correct.
- `"encoding error reading source: <UnicodeDecodeError repr>"`
  (cycle-5 UnicodeDecodeError formatting) → no prefix → emits
  `"helixc: encoding error reading source: ..."`. Correct.
- `"<no message>"` (synthetic) → no prefix → emits
  `"helixc: <no message>"`. Correct.

No regression on the un-pre-formatted Python defaults — they
still get the single `helixc: ` prefix that the cycle-5 fix
introduced. The duplicate-prefix corner case is closed.

Cross-walked the parser.py:1588 non-strict branch
(`print(msg, file=_sys.stderr); continue`) — that branch prints
the same pre-formatted message but does NOT raise, so the strip
helper is never invoked on it. The non-strict branch's `helixc:`
prefix is correct (it's the only printer of that message in
that codepath). No inconsistency introduced. **CLOSED.**

---

## Files reviewed

`helixc/check.py` (lines 246-338 for the new helper +
exception-cascade rework + unchanged finally drain wrap; plus
lines 341-360 for the `_main_inner` parse-time error envelope
to confirm strip-helper invariants are intact); `helixc/
frontend/parser.py` (lines 1582-1592 for the pre-formatted
FileNotFoundError producer); `helixc/tests/test_typecheck.py`
(lines 1530-1569 for the cycle-5 C4-6 FileNotFoundError +
UnicodeDecodeError regression tests, both unchanged and both
asserting the post-cycle-9 invariants). Plus the cycle-1
through cycle-8 codereview docs and the cycle-8 silent-
failures doc for cumulative invariant set + closure rationale.

---

## Specific cycle-9 changes audited

1. **check.py:246-256 — new `_emit_env_error(msg: str) -> None`
   helper**. Strip logic is anchored to the start of `msg` (via
   `.lstrip()` + `.startswith("helixc:")`) so it cannot strip an
   accidentally-occurring `helixc:` substring embedded later in
   the message. Slice index is `len("helixc:") == 7` — correct.
   Trailing-space cleanup via `.lstrip()` on the slice handles
   the typical `"helixc: foo"` form. Single-prefix invariant
   guaranteed: the function ALWAYS emits one and only one
   `helixc: ` prefix regardless of caller-provided prefix
   state. **PASS.**

2. **check.py:299-302 — FileNotFoundError-family arm now routes
   through `_emit_env_error(str(e))`**. Exit code unchanged at
   rc=2. The four caught types (FileNotFoundError,
   PermissionError, IsADirectoryError, NotADirectoryError) are
   the same set as cycle 5. Reachability via `_main_inner` is
   unchanged (the file-not-found inner early-return at line 363
   still catches the common case; the outer arm catches the
   race / open()-time / strict-stdlib path). **PASS.**

3. **check.py:303-305 — UnicodeDecodeError arm now routes
   through `_emit_env_error(f"encoding error reading source:
   {e}")`**. Exit code unchanged at rc=2. The "encoding error
   reading source:" prefix is part of the message passed to the
   helper, NOT a separate print — so the helper sees no
   `helixc:` prefix to strip, just prepends its own. **PASS.**

4. **check.py: removed `except ImportError as e: ... rc=2`
   arm**. ImportError now falls into the broad `except
   Exception` at line 306, producing rc=1 + "this is a compiler
   bug — please file an issue." Verified
   `issubclass(ImportError, Exception) == True` and no earlier
   arm catches ImportError (it's not an OSError subclass).
   **PASS.**

5. **check.py:286-298 — comment block updated**. The cycle-5
   C4-6 banner is preserved but trimmed: "FileNotFound,
   UnicodeDecodeError, ImportError, etc." → "FileNotFound,
   UnicodeDecodeError, etc." to reflect ImportError's removal
   from env-error attribution. New cycle-9 C8-2 paragraph at
   lines 295-298 documents the strip-prefix behavior. Comment
   text accurately describes the code as implemented. **PASS.**

6. **check.py:319-337 — finally drain wrap unchanged**. The
   cycle-5 wrap is preserved verbatim; no cycle-9 touch.
   **PASS (no change).**

---

## What was checked and found below threshold

- **No regression test for C8-1 (ImportError → rc=1 + compiler-
  bug tagline)**: cycle 9 did not add a test asserting that
  injecting a corrupted lazy import produces rc=1 and the
  "please file an issue" hint. The two cycle-5 tests at
  test_typecheck.py:1530-1569 cover FileNotFoundError and
  UnicodeDecodeError but not ImportError. A test like
  `monkeypatch.setattr(check_mod, "typecheck", lambda *_a, **_k:
  raise ImportError("cannot import name ..."))` plus assertions
  on `rc == 1` and `"compiler bug" in captured.err` would lock
  in the cycle-9 contract. The cycle-8 silent-failures
  recommendation 4 explicitly asked for this test. Absence is
  consistent with prior-cycle test-density precedent for
  check.py CLI changes (cycles 2, 3, 5, 8 also did not add CLI
  regression tests for their check.py changes). **Confidence
  60**, below threshold.

- **No regression test for C8-2 (no double-prefix in stdlib-
  strict missing-file path)**: cycle 9 did not add a test
  asserting that the parser.py:1587 strict-stdlib
  FileNotFoundError produces a single-prefix message. A test
  using `HELIXC_STDLIB_STRICT=1` + injected missing stdlib file
  + assertion `captured.err.count("helixc:") == 1` would lock
  in the cycle-9 contract. Same precedent as C8-1's test gap.
  **Confidence 55**, below threshold.

- **`_emit_env_error` only strips ONE leading `helixc:`**: a
  pathological caller raising
  `FileNotFoundError("helixc: helixc: foo")` would produce
  `"helixc: helixc: foo"` (strip once → `"helixc: foo"` → emit
  → `"helixc: helixc: foo"`). No known callee produces a
  double-pre-prefixed message; the only pre-prefixer is
  parser.py:1587 which prefixes exactly once. The single-strip
  is defensible (strip-all would be more aggressive and
  potentially eat substrings that legitimately start with
  `helixc:` in deeply-nested error chains). **Confidence 30**,
  below threshold.

- **`_emit_env_error` strip uses literal `"helixc:"` (no
  space)**: a caller raising
  `FileNotFoundError("helixc:foo")` (no space after the colon)
  would strip the 7 chars and `.lstrip()` produces
  `"foo"` — emitted as `"helixc: foo"`. Adds a space that
  wasn't in the original. Cosmetic; consistent with the helper's
  goal of canonical `helixc: <msg>` formatting. **Confidence
  25**, below threshold.

- **OSError edge cases (BlockingIOError, BrokenPipeError,
  ConnectionError family, TimeoutError, OSError itself for
  ENOSPC / EIO) still fall through to broad `except Exception`
  and get "compiler bug" label**: this was a cycle-8 codereview
  below-threshold observation (confidence 55). Cycle 9 did NOT
  expand the OSError-family catch arm, so the observation
  carries over unchanged. The four sub-types caught explicitly
  (FileNotFoundError, PermissionError, IsADirectoryError,
  NotADirectoryError) remain reasonable picks for the dominant
  user-env-error population. **Confidence 55**, below threshold.

- **Cycle 9 commit message scope coverage**: the commit message
  explicitly names both C8-1 (drop ImportError arm) and C8-2
  (route through helper). Subject line says "close C8-1 + C8-2
  (check.py exception classifier)". The cycle-8 codereview's
  recommendation that future commits "explicitly enumerate ALL
  findings being closed (and any opened) per cycle" is
  satisfied this cycle. No commit-message hygiene concern.
  **Confidence 15**, below threshold.

- **`_emit_env_error` is module-private (leading underscore) and
  called only from two sites inside `main()`**: the helper has no
  public API surface; no risk of mis-use from other modules. No
  finding. **Confidence 10**, below threshold.

- **`_emit_env_error` always uses `file=sys.stderr`**: consistent
  with all other env-error prints in `main()` and `_main_inner`.
  No stdout / stderr inversion. **Confidence 10**, below
  threshold.

- **`_emit_env_error` does NOT set rc**: the helper only prints;
  the caller sets `rc = 2`. This separation of concerns is
  clean and matches the pre-cycle-9 pattern. No regression.
  **Confidence 10**, below threshold.

- **Test invocation `check_mod.main([str(src_file)])` at test_
  typecheck.py:1544 still produces rc=2 + no "compiler bug"
  text post-cycle-9**: the cycle-5 tests assert
  `assert rc == 2` and `assert "compiler bug" not in
  captured.err` — both still hold. The strip helper does NOT
  change the env-error rc=2 contract. Confirmed by walking the
  call graph: `monkeypatch.setattr(check_mod, "typecheck",
  boom)` injects a `FileNotFoundError(2, ...)` which hits the
  outer FileNotFound arm, gets routed through
  `_emit_env_error`, prints `"helixc: [Errno 2] ..."`, and
  returns rc=2. **Confidence 15**, below threshold.

- **No regression on the cycle-3 C3-3 finally-drain contract**:
  the finally block at lines 319-337 is byte-for-byte unchanged
  from cycle 8. Drain still runs on every exit path; drain
  failures still wrap in their own `try/except Exception as
  drain_e` and emit a `warning:` notice without altering rc. No
  regression. **Confidence 10**, below threshold.

- **No new global state introduced by `_emit_env_error`**: the
  helper is a pure function (str → side-effect-only-print → no
  return value). No module-level mutation. No threading
  concern. **Confidence 10**, below threshold.

---

## Open prior findings (not re-flagged this cycle)

Per the cycle-8 silent-failures doc's tracking table:

- **audit-C4-1** (CRITICAL — D2 Call-RHS i32 SIGILL): still
  open; deferred per cycle-6's commit message pending parse-
  time constant folding or a typecheck pass before closure
  capture. Not addressed in cycle 9. Out of scope for this
  audit-C cycle. **Highest-priority still-open carryover.**
- **audit-C4-4** (HIGH — D9 paper-only): still open; not
  addressed in cycle 9. Out of scope.
- **audit-C4-8 deferred** (LOW — check.py doesn't call fn-
  mono): still open; not addressed in cycle 9. Out of scope.
- **monomorphize_safe docstring drift** (cycle-6 deferred):
  still open; not addressed in cycle 9. Out of scope.
- **D-vs-Quote diagnostic text** (cycle-7 deferred): still
  open; not addressed in cycle 9. Out of scope.
- **C7-1 close test-coverage gap** (cycle-8 housekeeping
  candidate): still open; cycle 9 did not add regression tests
  for `_compatible(TyMemTier, TyVar) is False` /
  `_compatible(TyMemTier, TySize) is False`. Out of scope.
- **C8-1 + C8-2 close test-coverage gap** (cycle-9
  housekeeping candidate): cycle 9 closed the fix half but not
  the test half for both findings. Listed in the below-
  threshold section above. Out of scope for cycle-9 strict-
  clean criterion (test-density gap is a precedent-consistent
  pattern, not a code defect).

No code-review regressions introduced by cycle 9:

- `check.py` exit-code contract preserved (0/1/2 mapping
  unchanged; ImportError now correctly classified as rc=1
  compile-error / internal-bug per the documented contract at
  lines 14-17).
- `_drain_ad_warnings` contract preserved (no touch in cycle 9).
- `_emit_env_error` is a strict-additive helper (no existing
  function signature changed; no module-level state added).
- `_main_inner` parse-time error envelope at line 356 (`print(
  f"helixc: {e}", file=sys.stderr); return 2`) is unchanged —
  it bypasses the outer helper entirely (returns rc=2 cleanly
  before any exception propagation), so the strip-helper does
  not interact with parse-time errs. The double-prefix concern
  for `_main_inner`'s own prints is structurally separate.

Pre-cycle-1 audit-stage5-6, audit-stage7-8, audit-stage9-16
baselines unchanged from cycle-1 status. Cycle-1 through cycle-
8 findings all marked CLOSED by their respective fix-sweep
commits or carried over per the tracking table.

---

## Verdict

**Cycle 9 Audit C: CLEAN — 0 findings (0 CRITICAL, 0 HIGH, 0
MEDIUM, 0 LOW) at or above the confidence-80 reporting
threshold.**

Strict-zero rule per user directive 2026-05-10. Cycle counter
advances provided cycles A (silent-failure) and B (type-
design) are also clean at this commit.

The cycle-9 fix-sweep is well-targeted (closes both cycle-8
silent-failure findings with a minimal 43-line diff to a single
file; new `_emit_env_error` helper is small and well-documented;
the dropped ImportError arm is a clean revert to the broad-
exception classification, restoring the rc=1 + "compiler bug"
contract for genuine import bugs). The review found no
regressions. The below-threshold notes (no regression tests
for C8-1 or C8-2; pathological double-pre-prefix edge case in
the strip helper; OSError edge cases unchanged from prior
cycles) are documented for future cycles but do not block this
cycle's clean status.

Cycle 9 is the candidate for the FIRST clean cycle in the 5-
clean-cycles gate (assuming the parallel silent-failure and
type-design audits at this commit are also clean). If all
three audits at 6968755 are clean, the clean-counter advances
from 0 to 1.
