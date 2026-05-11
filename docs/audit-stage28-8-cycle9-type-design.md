# Stage 28.8 Pre-29 Audit Gate — Cycle 9, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit**: 6968755 (read-only)
**Scope**: Re-audit type-system soundness focused on cycle-9's fix-sweep
(6968755 — closes cycle-8's silent-failures C8-1 MEDIUM + C8-2 LOW via
check.py exception classifier refinement). The cycle-9 fix-sweep
touches one file:

- `helixc/check.py:246-255` — new `_emit_env_error(msg: str) -> None`
  module-private helper. Strips a single leading `helixc:` from the
  passed message (after lstrip) and prints `f"helixc: {text}"` to
  stderr. Used by both env-error exception arms in `main()`.
- `helixc/check.py:286-305` — outer-arm exception classifier in
  `main()` reworked:
  - The `FileNotFoundError | PermissionError | IsADirectoryError |
    NotADirectoryError` arm now routes through `_emit_env_error`
    instead of direct `print(f"helixc: {e}", ...)`.
  - The `UnicodeDecodeError` arm now routes through `_emit_env_error`
    with the context prefix `"encoding error reading source: "`.
  - The `ImportError` arm (cycle-5 C4-6 introduction) is DELETED.
    ImportError now falls through to the broad `except Exception` arm
    at lines 306-318, classified as a compiler-internal bug with rc=1.
- `helixc/check.py:286-298` — comment block rewritten: cycle-5 prose
  trimmed to drop ImportError mention, and a cycle-9 C8-2 note added
  explaining the double-prefix strip.

The functional cycle-9 changes are: (1) new helper `_emit_env_error`,
(2) deletion of the `ImportError` arm, (3) two call-site refactors
to use the helper. No other files touched.

**Method**: read cycle-1 through cycle-8 type-design audit docs to
build the cumulative invariant set, then walked the cycle-9 diff
through each contract it touches. Specifically:

For the new `_emit_env_error` helper: characterized its contract as
"input is any `str`; output is exactly one `helixc:`-prefixed line on
stderr; an already-present leading `helixc:` is stripped once".
Verified the prefix-strip logic at line 253-254: `text.lstrip()
.startswith("helixc:")` is True only when the first non-whitespace
characters are `helixc:`; the strip removes those seven characters
then lstrip's the remainder. The function does NOT strip nested
prefixes (`helixc: helixc: helixc: foo` becomes `helixc: helixc: foo`)
— but no callee in the tree produces nested prefixes (verified by
grepping `raise FileNotFoundError|raise PermissionError|raise
IsADirectoryError|raise NotADirectoryError|raise UnicodeDecodeError`
across helixc/ — only parser.py:1587 raises with a `helixc:` prefix
in the message, and that's a single layer).

For the `ImportError` arm removal: traced the post-cycle-9 path for
an ImportError raised inside `_main_inner` (the realistic case being
a stale `from .frontend.xxx import yyy` lazy-import in `_main_inner`
where `yyy` got renamed). The exception propagates out of the try
block, no longer matches the deleted arm, and is caught by the broad
`except Exception` at lines 306-318. The diagnostic becomes `helixc:
internal error: ImportError: cannot import name 'yyy' from
'helixc.frontend.xxx'` followed by `helixc: this is a compiler bug —
please file an issue.` with rc=1. This is the correct classification
(per the commit-message claim) because ImportError inside
`_main_inner` is almost-always an internal-rename bug, not a
user-environment problem.

For the call-site refactors: enumerated both arms.
- Arm 1 (`FileNotFoundError | PermissionError | IsADirectoryError |
  NotADirectoryError`): pre-cycle-9 `print(f"helixc: {e}", ...)`;
  post-cycle-9 `_emit_env_error(str(e))`. The behavior delta is the
  prefix-strip: when the underlying exception was raised with `e =
  FileNotFoundError("helixc: stdlib file missing: /path/foo.hx")`
  (parser.py:1587 strict-stdlib case), pre-cycle-9 printed `helixc:
  helixc: stdlib file missing: /path/foo.hx` (double prefix), post-
  cycle-9 prints `helixc: stdlib file missing: /path/foo.hx` (single
  prefix). All other FileNotFoundError-family raises in the tree are
  Python-builtin raises (e.g. `open()` failures, `os.remove()`
  failures, the test-suite synthetic raise at test_cli.py:304's
  `PermissionError("synthetic permission denied")`) whose messages do
  NOT carry a `helixc:` prefix, so `_emit_env_error` behaves
  identically to the pre-cycle-9 direct print for those.
- Arm 2 (`UnicodeDecodeError`): pre-cycle-9 `print(f"helixc: encoding
  error reading source: {e}", ...)`; post-cycle-9 `_emit_env_error(
  f"encoding error reading source: {e}")`. Behavior is identical
  because no callee raises UnicodeDecodeError with a `helixc:`-
  prefixed message (the standard `str.encode()` / `str.decode()` /
  `open(..., encoding=...)` machinery produces messages like `'utf-8'
  codec can't decode byte 0xff in position 0: invalid start byte`).

For the broad `except Exception` arm at 306-318: unchanged. Re-
verified the rc=1 contract for genuine compiler-internal exceptions
(AttributeError, KeyError, IndexError, AssertionError, TypeError,
RuntimeError, ValueError, ImportError now). Pre-cycle-9 ImportError
was misclassified as rc=2 env-error; post-cycle-9 it is correctly
rc=1 compiler-bug. No regression.

For the finally-block drain (319-337): unchanged in cycle 9. The
nested try/except still wraps the drain to prevent it from masking
the primary failure. Re-verified that the drain's own ImportError
risk (the lazy `from .frontend.autodiff import take_diff_warnings`)
is caught by the drain's `except Exception` at line 332, not the
outer arm, so cycle-9's outer-arm ImportError reclassification does
not interact with the finally-block path.

For tests: cycle 9's commit message claims "264 targeted tests pass
(autodiff + typecheck + struct_mono + pytree + autodiff_reverse +
cli)". Read-only audit cannot run tests, but verified the test fixture
at test_cli.py:304 raises `PermissionError("synthetic permission
denied")` (no `helixc:` prefix) so it goes through the first arm
unchanged. test_typecheck.py:1539 raises `FileNotFoundError(2, "No
such file or directory", ...)` whose `str(e)` is `[Errno 2] No such
file or directory: ...` (no `helixc:` prefix) — first arm unchanged.
test_typecheck.py:1561 raises `UnicodeDecodeError("utf-8", b"\xff",
0, 1, "invalid start byte")` whose `str(e)` is `'utf-8' codec can't
decode byte 0xff in position 0: invalid start byte` (no `helixc:`
prefix) — second arm unchanged.

**Result**: **0 findings (0 HIGH, 0 MEDIUM, 0 LOW)**. Cycle 9
addresses cycle-8's two silent-failures findings (C8-1 MEDIUM, C8-2
LOW) cleanly at the contract level. The `_emit_env_error` helper is
a narrow, well-encapsulated addition (module-private, single-purpose,
self-documenting docstring referencing the C8-2 finding by ID). The
ImportError arm deletion is a strict tightening: a misclassified
arm is removed and the broad `except Exception` arm picks up the
slack with the correct rc=1 + "please file an issue" diagnostic.
The strict criterion ("zero findings of any severity") is **MET**.

---

## Cycle 8 finding re-verification

| ID   | Severity prev | Audit (prev)            | Status     | Notes                                                                                                                                                                                                                                                                                                                                                                                          |
|------|---------------|-------------------------|------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| C8-1 | MEDIUM        | silent-failures (cyc 8) | CLOSED     | The `except ImportError as e: print(f"helixc: import error: {e}", ...); rc = 2` arm is deleted from `main()`. Genuine ImportError now falls through to the broad `except Exception` arm at lines 306-318, producing `helixc: internal error: ImportError: ...` followed by `helixc: this is a compiler bug — please file an issue.` with rc=1. Correct classification for the lazy-import-rename case.                                                                  |
| C8-2 | LOW           | silent-failures (cyc 8) | CLOSED     | New `_emit_env_error` helper at lines 246-255 strips a single leading `helixc:` from the exception message before printing. Both env-error arms (FileNotFoundError-family, UnicodeDecodeError) route through this helper. parser.py:1587's `raise FileNotFoundError(f"helixc: stdlib file missing: {p}")` now produces `helixc: stdlib file missing: /path/foo.hx` instead of the prior `helixc: helixc: stdlib file missing: /path/foo.hx` double-prefix. |

No cycle-8 type-design findings (cycle 8 type-design was CLEAN) need
re-verification. The cycle-9 fix-sweep does not touch any
typecheck.py contract surface that was audited in cycles 1-8, so
prior-cycle invariants (`_compatible` TyMemTier strict separation,
`_size_compatible` shape-position cascade, D-binop diagnostic-text
accuracy, etc.) are preserved by inspection (no diff in those files).

---

## Per-surface review (cycle-9 touchpoints)

### Surface 1: `_emit_env_error` helper

**Placement**: `helixc/check.py:246-255`, between `_drain_ad_warnings`
and the `main()` dispatch entry. Module-private (leading underscore).
Logical proximity is correct — both `_drain_ad_warnings` and
`_emit_env_error` are stderr-printing helpers used by the outer
dispatch.

**Type signature**: `(msg: str) -> None`. Pure side-effect function
(no return value, prints to `sys.stderr`). The `str` input type is
the broadest reasonable type — callers pass `str(e)` for
FileNotFoundError-family exceptions and an f-string concatenation for
UnicodeDecodeError. No `Optional[str]` (no callsite passes None).

**Invariant (input)**: `msg` may or may not have a leading `helixc:`
prefix (after lstrip). The helper handles both cases.

**Invariant (output)**: exactly one `helixc: ` prefix on the printed
line. Output goes to `sys.stderr`. No newline-stripping (relies on
`print()`'s default newline behavior).

**Edge cases**:
- Empty string: `_emit_env_error("")` prints `helixc: `. Acceptable
  but unlikely (no callsite passes empty). Not a finding.
- Whitespace-only string: `_emit_env_error("   ")` — `lstrip()`
  returns `""`, `startswith("helixc:")` is False, prints
  `helixc:    ` (preserving the trailing whitespace). Acceptable
  cosmetic edge case. Not a finding.
- Already-prefixed: `_emit_env_error("helixc: foo")` strips to
  `foo`, prints `helixc: foo`. Correct.
- Leading whitespace + prefix: `_emit_env_error("   helixc: foo")` —
  the `text.lstrip()` test fires True, the strip uses
  `text.lstrip()[len("helixc:"):].lstrip()` which removes the leading
  whitespace AS WELL AS the prefix, prints `helixc: foo`. Correct.
- Nested prefix: `_emit_env_error("helixc: helixc: foo")` strips one
  layer, prints `helixc: helixc: foo`. The helper does NOT strip
  nested prefixes. Acceptable — no callsite in the tree produces
  nested prefixes (verified by grep over `raise FileNotFoundError`,
  `raise PermissionError`, `raise IsADirectoryError`, `raise
  NotADirectoryError`, `raise UnicodeDecodeError` in helixc/; only
  parser.py:1587 raises with a single-layer prefix). Not a finding.
- Prefix-without-space: `_emit_env_error("helixc:foo")` — `lstrip()
  .startswith("helixc:")` is True, the strip removes `helixc:`, the
  trailing `.lstrip()` is a no-op since there's no whitespace after,
  prints `helixc: foo`. Correct (the helper normalizes spacing).

**Encapsulation**: leading underscore signals module-private. No
class state, no global state mutation. Pure function modulo the
print side-effect.

**Self-documentation**: docstring at lines 247-251 explicitly cites
the C8-2 finding ID and the parser.py:1587 reproducer. Maintainers
removing the prefix-strip logic would need to acknowledge the audit
trail. Strong contract documentation.

**Risk of misuse**: a future caller could route a genuinely-empty
error message through `_emit_env_error`, getting `helixc: ` with no
context. Mitigation: the helper is private and grep-discoverable;
all current callsites format a context string. Not a finding.

### Surface 2: outer-arm exception classifier post-cycle-9

**Placement**: `helixc/check.py:286-318` (the `main()` try/except
block).

**Contract (pre-cycle-9)**:
- Arm 1 (FileNotFoundError-family): print `f"helixc: {e}"`, rc=2.
- Arm 2 (UnicodeDecodeError): print `f"helixc: encoding error
  reading source: {e}"`, rc=2.
- Arm 3 (ImportError): print `f"helixc: import error: {e}"`, rc=2.
- Arm 4 (broad Exception): print compiler-bug diagnostic, rc=1.

**Contract (post-cycle-9)**:
- Arm 1 (FileNotFoundError-family): `_emit_env_error(str(e))`, rc=2.
  (Strips a single `helixc:` prefix from the message if present.)
- Arm 2 (UnicodeDecodeError): `_emit_env_error(f"encoding error
  reading source: {e}")`, rc=2.
- Arm 3 (broad Exception, including ImportError): print compiler-bug
  diagnostic, rc=1.

**Contract integrity**: cleaner than cycle 8. The classifier now
splits exceptions into two clean categories: "user-environment"
(file I/O + encoding) vs. "compiler-internal" (everything else).
ImportError is correctly classified as the latter because
`_main_inner` uses 18 lazy `from .` imports of internal modules; an
ImportError from those is a rename bug, not a missing dependency on
the user's end. The cycle-5 C4-6 cascade attempted to be helpful by
adding the ImportError arm but the placement at the outer-dispatch
boundary means it catches BOTH genuine missing-dependency cases
(rare, since helixc has minimal external deps) AND internal-rename
cases (common), and mis-classifies them all as the former. Cycle 9's
removal restores the correct asymmetry: deps are checked at install/
boot time, internal renames are compiler bugs.

**Arm ordering**: FileNotFoundError-family → UnicodeDecodeError →
Exception. The first two are subclasses-of-OSError-and-ValueError
respectively (UnicodeDecodeError is a subclass of ValueError, not
OSError); the broad `Exception` catches everything else. Order is
correct: specific arms first, broad arm last.

**Symmetry with rc**: env-error arms rc=2 ("invocation / config
error"); compiler-bug arm rc=1 ("compiler internal error"). This
matches Unix convention (rc=2 for usage/config, rc=1 for runtime
failure).

**Interaction with finally-block drain**: the drain at 319-337 fires
on all four paths (success rc=0, env-error rc=2, compiler-bug rc=1).
Drain rc overrides the primary rc only when primary rc was 0; otherwise
drain failures print a warning but don't override. Unchanged from
cycle 5/6. Correct.

### Surface 3: callee contract for env-error messages

**Placement**: implicit contract between callees that raise
FileNotFoundError-family / UnicodeDecodeError and the cycle-9 outer-
dispatch arm.

**Contract (post-cycle-9)**: callees MAY raise with a `helixc:`-
prefixed message (the helper will strip one layer). Callees SHOULD
NOT raise with a nested `helixc: helixc: ...` prefix.

**Audit of callees**:
- `parser.py:1587`: `raise FileNotFoundError(f"helixc: stdlib file
  missing: {stdlib_path}")` — single-layer prefix, helper strips
  correctly. The prefix was added in stage-15-ish to make the
  non-strict path's `print(msg, ...)` at line 1588 carry the
  `helixc:` prefix. Both paths now produce the same prefix-count.
- `test_cli.py:304` (test fixture): `raise PermissionError(
  "synthetic permission denied")` — no prefix, helper passthrough.
- `test_typecheck.py:1539`: `raise FileNotFoundError(2, "No such
  file or directory", ...)` — no prefix, helper passthrough.
- `test_typecheck.py:1561`: `raise UnicodeDecodeError("utf-8",
  b"\xff", 0, 1, "invalid start byte")` — no prefix, helper
  passthrough.

Production callees are clean. Test fixtures are clean. The cycle-9
contract is well-defined and the callee surface is small (one
production raise-with-prefix at parser.py:1587).

**Forward-compat note**: if a future callee inside _main_inner's
pipeline (parser/typecheck/lowering/codegen) raises a builtin OSError
subclass with `helixc:` prefix, the helper handles it. If a callee
double-prefixes the message, the helper strips one layer and a
single redundant `helixc:` remains. This is graceful degradation
(not silent acceptance of malformed input), but a forward
recommendation is to keep raise-message convention consistent: callee
EITHER includes the prefix (parser.py:1587 style) OR omits it
(builtin Python convention); never doubles it.

---

## Other surfaces (re-verified, not touched in cycle 9)

### typecheck.py contracts (cycles 1-8)

Cycle 9 does not modify `helixc/frontend/typecheck.py`. The
following contracts are unchanged from cycle 8's snapshot:

- `_compatible` TyMemTier strict-separation contract
  (typecheck.py:2224-2252).
- `_size_compatible` shape-position cascade
  (typecheck.py:2208-2222).
- D-binop diagnostic-text accuracy
  (typecheck.py:1349-1381).
- Call-boundary `_compatible` invocation pre-filter
  (typecheck.py:746-752).

Verified by inspection of the cycle-9 diff (`git show 6968755` shows
only `helixc/check.py` and `docs/audit-stage28-8-cycle8-silent-
failures.md` modified).

### check.py other env-error sites (lines 240, 356, 363, 678)

These print `helixc:`-prefixed messages directly (not via
`_emit_env_error`):
- Line 240: AD warning forwarding inside `_drain_ad_warnings` —
  warnings already have their own format, no double-prefix risk.
- Line 356: parse_args error rendering — error strings come from
  argparse-style parsing, never prefixed with `helixc:`.
- Line 363: `file not found: {path}` pre-banner check (before the
  try/except wrapper) — never reaches `_emit_env_error`.
- Line 678: output-write OSError catch — message is the helixc-
  context wrapper, the underlying `e` is a builtin error string.

None of these sites are in the cycle-9 helper's stated scope
(exception-message routing out of `main()`'s try/except). Not a
finding.

---

## Cycle 9 invariant snapshot (post-fix)

The cycle-9 fix-sweep tightens the check.py outer-dispatch contract:

**`_emit_env_error` helper contract** (check.py:246-255):
- Module-private (leading underscore).
- Input: any `str`.
- Output: exactly one `helixc:`-prefixed line on stderr.
- Idempotency: strips a single leading `helixc:` (after lstrip).
- Side-effects: print to `sys.stderr` only.
- Self-documenting: docstring cites C8-2 + parser.py:1587.

**`main()` outer-dispatch classifier contract** (check.py:284-318):
- Try `_main_inner(argv, a_holder)`.
- On `FileNotFoundError | PermissionError | IsADirectoryError |
  NotADirectoryError`: `_emit_env_error(str(e))`, rc=2.
- On `UnicodeDecodeError`: `_emit_env_error(f"encoding error reading
  source: {e}")`, rc=2.
- On any other Exception (including ImportError): compiler-bug
  diagnostic, rc=1.
- Finally: wrapped AD-warning drain that doesn't mask primary
  failures.

**Env-error callee contract** (implicit, callees throughout helixc/):
- MAY raise FileNotFoundError-family / UnicodeDecodeError with a
  single-layer `helixc:` prefix (helper strips it).
- SHOULD NOT raise with a nested `helixc: helixc:` prefix (helper
  only strips one layer; a redundant prefix would survive).
- Production callees verified clean (parser.py:1587 is single-
  layer); test fixtures verified clean.

---

## Cycle 9 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity.**

This cycle finds **0 findings (0 HIGH, 0 MEDIUM, 0 LOW)**.

By the strict criterion, **cycle 9 counts CLEAN**.

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

This is the THIRD consecutive cycle to meet the strict criterion
under Audit B. The cycle-9 fix-sweep is narrow (a 12-line helper
added, a 5-line ImportError arm deleted, two call-sites refactored
to use the helper), and the helper's contract is well-encapsulated
with strong docstring documentation citing the audit-finding ID.

The 5-clean-cycles requirement (per the cycle-5 doc's projection for
Python-helixc deprecation) is now 3/5. Cycles 10-11 would need to
clean to satisfy that bar.

**Recommendation**: no fix-sweep needed for cycle 9. Proceed to
cycle 10 audit gate.

---

## Forward notes (not cycle-9 findings)

1. **Test for C8-1 ImportError-as-compiler-bug**: cycle 9 deletes
   the ImportError arm but no regression test was added that asserts
   "ImportError inside `_main_inner` produces rc=1 + 'compiler bug'
   diagnostic" (the inverse of C8-1). A future cycle's housekeeping
   could add a fixture that monkey-patches a lazy import to raise
   ImportError and asserts the post-cycle-9 contract. Not blocking.

2. **Test for C8-2 double-prefix strip**: similarly, no regression
   test asserts that `parser.py:1587`'s strict-stdlib path produces
   single-prefix output. A future cycle could add a fixture that
   triggers the strict-stdlib path with a missing file and asserts
   the output is `helixc: stdlib file missing: ...` (single prefix).
   Not blocking.

3. **Convention note for raise-message prefix**: a future style
   doc / contributor guide could codify the cycle-9 implicit
   contract: callees of `main()`'s try block MAY include a single
   `helixc:` prefix in raise messages (preserved for callees that
   need the prefix in alternative non-raised print paths, like
   parser.py:1587's non-strict branch at line 1588). Helper strips
   one layer; never include nested prefixes. This is a documentation
   improvement, not blocking.

These are forward suggestions for the test suite and contributor
guide; they are not findings against cycle 9.
