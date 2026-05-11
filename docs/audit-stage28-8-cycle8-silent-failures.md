# Stage 28.8 Cycle 8 — Silent-Failure Audit

**Date**: 2026-05-11
**Commit**: 5d1ca24 (read-only audit). Cycle-8 fix-sweep range
b8e047e..5d1ca24 (1 fix-sweep commit covering C7-1 closure +
unsignaled C4-7 closure).
**Scope**: All Helix source — `helixc/bootstrap/*.hx`,
`helixc/frontend/*.py`, `helixc/ir/*.py`, `helixc/backend/*.py`,
`helixc/stdlib/*.hx`. Specifically re-audits the cycle-8
fix-sweep changes for fresh silent windows introduced by the
fixes themselves.
**Trigger**: pre-Stage-29 audit gate — Cycle 8 of 5+ (the gate
re-arms each time a cycle is not clean). Re-audits same scope
after cycle-8 fixes were landed.
**Strict criterion** (per user directive 2026-05-10): cycle counts
CLEAN only when **zero new findings of ANY severity**
(CRITICAL/HIGH/MEDIUM/LOW). Per the user directive for cycle 8,
findings already documented in cycles 1-7 are NOT re-flagged unless
they CHANGED in the cycle-8 fix-sweep.

**Method**:
1. Read the cycle-7 silent-failures audit (1 LOW finding — C7-1).
2. Walked `git show 5d1ca24` — the single cycle-8 fix-sweep
   commit. Read the diff for each of: typecheck.py (drop of the
   cycle-7 G2 TyMemTier × (TyVar|TySize) carve-out at lines
   2208-2211) and check.py (narrowing of the cycle-3 broad
   `except Exception` into env-error arms + drain-failure
   suppression).
3. For each cycle-8 fix's diff, traced data flow forward to check
   whether the fix opened a fresh silent window, left a fix
   incomplete (paper-only), compounded a prior-cycle regression,
   or over-corrected.
4. Direct Python probes against `5d1ca24` HEAD to confirm
   reproducer behavior for `_compatible(TyMemTier, TyVar)` /
   `_compatible(TyMemTier, TySize)` (both must now return False),
   the same-name TyVar pass (`_compatible(TyVar('T'),TyVar('T'))`
   must still return True), the both-MemTier same-tier same-inner
   pass, and the `_size_compatible` shape-position cascade still
   accepting `(TyVar, TyPrim)`.
5. Direct end-to-end probes against `check.py` for: (a) the
   nonexistent-file path landing in the inner `os.path.exists`
   check rather than the new outer catch; (b) the stdlib-strict
   FileNotFoundError path triggering the new outer catch; (c) a
   genuine internal ImportError landing in the new outer catch.
6. Ran the typecheck regression suite (`pytest helixc/tests/
   test_typecheck.py`) at 5d1ca24 — 105/105 pass.
7. Cross-checked the cycle-8 fix coverage against the still-open
   carryover findings from cycles 1-7 to identify which carryovers
   were actually CLOSED by cycle 8 vs which remain open.

**Result**: **2 new findings (0 CRITICAL, 1 MEDIUM, 1 LOW)** —
Cycle 8 NOT clean. The fix-sweep CLOSES the C7-1 LOW carve-out
correctly — `_compatible(TyMemTier, TyVar)` and
`_compatible(TyMemTier, TySize)` both reject post-fix, restoring
the body-vs-return / let-declared / if-arm / match-arm diagnostics
symmetrically with the rest of the cycle-7 narrowing. The cycle-8
commit ALSO landed the cycle-5 audit-C4-7 closure (narrowing the
broad `except Exception` in check.py into FileNotFoundError /
PermissionError / IsADirectoryError / NotADirectoryError /
UnicodeDecodeError / ImportError arms with rc=2, plus wrapping
the finally drain). Both the C7-1 close and the C4-7 close are
substantively correct.

The two new findings are SECONDARY EFFECTS of the cycle-8 C4-7
narrowing:

- **C8-1 (MEDIUM)**: the new `except ImportError as e: ... rc=2`
  arm catches genuine internal compiler bugs (e.g., a refactor
  rename of `monomorphize_structs` that leaves a stale
  `from .frontend.struct_mono import monomorphize_structs` lazy
  import inside `_main_inner`) and mis-attributes them as user
  environment issues. The "please file an issue" hint is
  suppressed, exit code shifts 1→2, and the bug stays hidden.
- **C8-2 (LOW)**: the new `except (FileNotFoundError, ...) as e:
  ... print(f"helixc: {e}")` arm catches a pre-formatted
  FileNotFoundError raised by `_merge_stdlib` (parser.py:1587 when
  `HELIXC_STDLIB_STRICT=1`) whose message ALREADY starts with
  "helixc: ". The user sees `helixc: helixc: stdlib file missing:
  <path>` — duplicate prefix.

The C7-1 LOW from cycle 7 is correctly CLOSED. The cycle-8 commit
message did not mention the simultaneous C4-7 closure (which is
also landed in this commit), but the closure itself is sound. The
cycle-8 fix-sweep made strong progress: C7-1 LOW closed + the
still-open audit-C4-7 MEDIUM closed; net +2 closed, +2 new
(differing severities — C8-1 is MEDIUM equal to the C4-7 it
closes, C8-2 is LOW lower than C7-1).

---

## CRITICAL FINDINGS

(none)

---

## HIGH FINDINGS

(none)

---

## MEDIUM FINDINGS

### Finding C8-1: cycle-8 C4-7-closure `except ImportError as e` (check.py:292-296) mis-attributes genuine internal compiler import bugs as user-environment "import error", suppressing the "please file an issue" hint and shifting rc from 1 to 2

**Location**:
- helixc/check.py:292-296 (the new ImportError catch arm).
- helixc/check.py:413 (lazy `from .frontend.struct_mono import
  monomorphize_structs`).
- helixc/check.py:429-431 (lazy
  `from .frontend.flatten_impls import flatten_impls,
  DuplicateMethodError`).
- helixc/check.py:452 (lazy
  `from .frontend.deprecated_pass import emit_warnings`).
- helixc/check.py:467 (lazy
  `from .frontend.trace_pass import validate_trace_attrs`).
- helixc/check.py:481-482 (lazy `from .frontend.panic_pass
  import (...)`).
- helixc/check.py:500 (lazy `from .frontend.unsafe_pass import
  check_unsafe_ops`).
- helixc/check.py:515 (lazy
  `from .frontend.autotune import validate_autotune_prog`).
- helixc/check.py:544-548 (lazy ir/passes imports).
- helixc/check.py:554 (lazy
  `from .frontend.grad_pass import grad_pass`).
- helixc/check.py:602, 647 (lazy `from .backend.x86_64 import
  compile_module_to_elf`).
- helixc/check.py:628 (lazy `from .ir import tile_ir as ti`).
- helixc/check.py:629 (lazy `from .backend.ptx import emit_ptx`).
**Severity**: MEDIUM
**Category**: cycle-8-fix-introduced over-broad catch that
mis-attributes compiler bug as environment issue
**Stage**: 28.8 cycle-8 commit 5d1ca24 (C4-7 closure ImportError
arm)

**Description**:
The cycle-8 commit closes the still-open cycle-5 audit-C4-7
MEDIUM by narrowing the broad `except Exception` in
`check.main()` into specific env-error arms. One of those arms is:

```python
except ImportError as e:
    # An import failure is an environment problem (missing module
    # or broken install), not a user-source compile bug.
    print(f"helixc: import error: {e}", file=sys.stderr)
    rc = 2
```

The comment's rationale — "import failure is an environment
problem" — holds for the import-at-startup case (e.g., the user
installed helixc in a virtualenv missing a dependency). But
`check.py` performs **18 lazy `from .frontend.XXX import YYY`
statements INSIDE `_main_inner`**, all reachable from a user
input file path. A refactor that renames an internal function
(e.g., the cycle-3 commit dccfc7e renamed `monomorphize` →
`Monomorphizer.run`) and forgets to update the lazy import would
raise:

```
ImportError: cannot import name 'monomorphize_structs' from 'helixc.frontend.struct_mono'
```

Pre-cycle-8 this lands in the broad `except Exception` arm at
line 297-309 and prints:

```
helixc: internal error: ImportError: cannot import name 'monomorphize_structs' from ...
helixc: this is a compiler bug — please file an issue.
```

with rc=1. Post-cycle-8 the new `except ImportError` arm fires
FIRST (Python except-chain semantics: most-specific match) and
prints:

```
helixc: import error: cannot import name 'monomorphize_structs' from ...
```

with rc=2. The "please file an issue" hint is gone. The user
sees a message that looks like an environment problem (rc=2 is
documented in check.py:14-17 as "bad invocation"). They check
their install, find nothing wrong, and either give up or assume
the file they passed in is the problem. The compiler bug stays
hidden.

Verified end-to-end via direct Python probe against 5d1ca24 by
deleting `monomorphize_structs` from the module before invoking
`check.main()`:

```
$ python -c "
import helixc.frontend.struct_mono as sm
del sm.monomorphize_structs
from helixc.check import main
rc = main(['/tmp/foo.hx'])
print('rc=', rc)
"
helixc: import error: cannot import name 'monomorphize_structs' from 'helixc.frontend.struct_mono' (...)
-- helixc-check: /tmp/foo.hx
   parse:    OK  (1 fns, 1 items)
   typecheck: OK
rc= 2
```

The cycle-4 audit-C4-7 finding's recommendation 1 listed
`AttributeError, KeyError, IndexError, AssertionError, TypeError,
RuntimeError, ValueError` as the compiler-bug catch arm but did
NOT include `ImportError` in either list. The cycle-8 fix made
the unstated decision that `ImportError` belongs on the env-error
side. From the silent-failure-audit lens the decision splits the
ImportError population into two classes — startup imports (lines
57-63 at module import time) which would fail BEFORE
`check.main()` runs (so the new arm never sees them; the Python
interpreter prints a traceback at import time), and lazy
in-`_main_inner` imports which post-cycle-8 are
mis-attributed.

**Hidden errors**:
- Any refactor that renames a function symbol exported from
  `helixc.frontend.struct_mono`, `helixc.frontend.flatten_impls`,
  `helixc.frontend.deprecated_pass`,
  `helixc.frontend.trace_pass`, `helixc.frontend.panic_pass`,
  `helixc.frontend.unsafe_pass`, `helixc.frontend.autotune`,
  `helixc.ir.lower_ast`, `helixc.ir.passes.fdce`,
  `helixc.ir.passes.const_fold`, `helixc.ir.passes.cse`,
  `helixc.ir.passes.dce`, `helixc.frontend.grad_pass`,
  `helixc.backend.x86_64`, `helixc.ir.tile_ir`, or
  `helixc.backend.ptx` and forgets to update the lazy import in
  `check.py` will now silently mis-attribute as env error.
- A typo in any of the 18 lazy-import statements (e.g.,
  `from .frontend.struct_mono import monomoprhize_structs`) is now
  reported as user-env "import error" rc=2 with no
  "please file an issue" hint.
- A genuine module-not-found from a partial install (e.g.,
  `helixc.backend.ptx` missing from a CPU-only install) is
  correctly attributed (cycle-8's intended use case). But this
  is rare — once installed, the modules stay installed unless
  someone deletes them.
- Exit code semantics regression: pre-cycle-8 rc=1 ("compile
  error"); post-cycle-8 rc=2 ("bad invocation"). CI scripts that
  branch on `rc=2 → user input bad; rc=1 → compiler bug → ping
  on-call` now route compiler bugs to the "user input" lane.

**Recommendation**:
1. **PREFERRED**: re-classify `ImportError` to the compiler-bug
   side. Move the ImportError catch from lines 292-296 into the
   broad `except Exception` arm (or add `except ImportError as e:`
   that delegates to the compiler-bug print + rc=1). This matches
   the original cycle-4 audit-C4-7 recommendation 1 (which didn't
   list ImportError on either side; the safer default is the
   compiler-bug side because any ImportError reachable from
   `_main_inner` is by definition a compiler / install bug).

2. **ALTERNATIVE**: split the lazy imports into a startup batch
   at the top of `_main_inner` (or into the module-level imports
   at lines 57-63), so an ImportError can only originate from
   startup (which by the cycle-8 rationale IS an env issue). This
   makes the cycle-8 attribution correct, but at the cost of
   eager imports — every `check.main()` call would import all 18
   pipeline modules even on `--help` or `--emit-ast` exits.

3. **ALTERNATIVE**: keep the env-error attribution but add a
   `helixc: hint: if you did not modify the helixc install, this
   may be a compiler bug — please file an issue at ...` second
   print after the "import error" line. Preserves rc=2 but
   restores the file-an-issue signal.

4. Add a regression test that simulates the corrupted-import
   case (`monkeypatch.delattr` on a struct_mono symbol) and
   asserts the exit code + message — recommended regardless of
   which of 1-3 is chosen, to guard against the next cycle of
   drift.

**Trap-id**: n/a (check.py CLI dispatch, no trap-id).

---

## LOW FINDINGS

### Finding C8-2: cycle-8 C4-7-closure `except (FileNotFoundError, ...) as e: print(f"helixc: {e}")` (check.py:282-285) catches a pre-formatted FileNotFoundError from `_merge_stdlib` and emits a duplicate `helixc: helixc: ...` message prefix

**Location**:
- helixc/check.py:282-285 (new outer FileNotFoundError catch arm).
- helixc/frontend/parser.py:1582-1588 (the stdlib-missing path
  in `_merge_stdlib` that raises the pre-formatted
  FileNotFoundError when `HELIXC_STDLIB_STRICT=1`).
**Severity**: LOW
**Category**: cycle-8-fix-introduced cosmetic prefix-doubling at
catch-and-rewrap boundary
**Stage**: 28.8 cycle-8 commit 5d1ca24 (C4-7 closure
FileNotFoundError arm)

**Description**:
The cycle-8 commit narrows the broad `except Exception` in
`check.main()` into:

```python
except (FileNotFoundError, PermissionError, IsADirectoryError,
        NotADirectoryError) as e:
    print(f"helixc: {e}", file=sys.stderr)
    rc = 2
```

The new arm prepends `helixc: ` to whatever `str(e)` is. This is
correct for the un-pre-formatted Python defaults
(`[Errno 2] No such file or directory: '/path/foo.hx'` →
`helixc: [Errno 2] No such file or directory: '/path/foo.hx'`)
but produces a duplicate prefix for `_merge_stdlib`'s pre-formatted
FileNotFoundError:

```python
# parser.py:1585-1587
msg = f"helixc: stdlib file missing: {stdlib_path}"
if strict:
    raise FileNotFoundError(msg)
```

After cycle-8 wraps this with `helixc: ` again:

```
helixc: helixc: stdlib file missing: C:\Projects\Kovostov-Native\helixc\stdlib\nonexistent.hx
```

Verified end-to-end via direct Python probe against 5d1ca24
with `HELIXC_STDLIB_STRICT=1` and an injected nonexistent stdlib
file:

```
$ HELIXC_STDLIB_STRICT=1 python -c "..."
helixc: helixc: stdlib file missing: C:\...\helixc\stdlib\nonexistent_stdlib_file.hx
-- helixc-check: C:\...\tmpXXXX.hx
rc= 2
```

Note ALSO the output ordering: the duplicate-prefix message
prints to stderr BEFORE the `-- helixc-check: <tmp>` banner
prints to stdout (because the exception originated in
`_merge_stdlib` during `parse(...)` AFTER the banner-print at
line 365 was buffered to stdout but before stdout was flushed).
On a terminal with unbuffered stderr / buffered stdout, the
error message visually precedes the banner. Cosmetic but
confusing — the user sees "stdlib file missing" before they see
which file is being compiled.

**Hidden errors**:
- The duplicate `helixc: helixc: ` prefix is a minor cosmetic
  defect but signals to a careful reader that the catch-and-
  rewrap layering wasn't audited for already-formatted
  exception messages.
- Future contributors raising a pre-formatted `helixc: ...`
  FileNotFoundError / PermissionError / IsADirectoryError /
  NotADirectoryError / UnicodeDecodeError ANYWHERE inside
  `_main_inner` (or its callees) will trigger the same
  duplicate prefix. Currently only `_merge_stdlib`'s
  FileNotFoundError is pre-formatted, but the pattern leaks.
- The output-ordering inversion (stderr error before stdout
  banner) makes log scraping less reliable — a CI parser
  expecting `^-- helixc-check: ` as the first line will
  mis-attribute the FileNotFoundError to the previous compile.
- The strict-clean-criterion lens flags this as a finding even
  though it's pure cosmetic, because the strict criterion is
  "zero findings of any severity".

**Recommendation**:
1. **PREFERRED**: strip a leading `helixc: ` from `str(e)` before
   re-prefixing in the new outer arms:
   ```python
   except (FileNotFoundError, PermissionError, IsADirectoryError,
           NotADirectoryError) as e:
       msg = str(e)
       if msg.startswith("helixc: "):
           print(msg, file=sys.stderr)
       else:
           print(f"helixc: {e}", file=sys.stderr)
       rc = 2
   ```
   Same pattern for the `UnicodeDecodeError`, `ImportError`, and
   broad `Exception` arms (lines 286-296, 297-309).

2. **ALTERNATIVE**: change `_merge_stdlib` to raise an
   UN-prefixed FileNotFoundError (`raise FileNotFoundError(
   f"stdlib file missing: {stdlib_path}")`) and let the outer
   arm add the `helixc: ` prefix uniformly. Requires changing
   the non-strict-mode print at parser.py:1588 to also drop the
   leading `helixc: ` to keep both branches consistent.

3. Drop the explicit `helixc: ` prefix in the new outer arms;
   trust callees to prefix correctly. Simplest but rolls back
   the diagnostic-consistency rationale of cycle-8.

4. Add a regression test that asserts no double-prefix in the
   stdlib-strict missing-file path.

**Trap-id**: n/a.

---

## Cycle 8 fix-sweep re-verification

Each cycle-8 fix-sweep change was inspected for paper-only fixes,
silent windows, false positives, and state-leak. The cycle-8
fix-sweep landed as a single commit (5d1ca24) covering: C7-1
closure (drop G2 carve-out from `_compatible`) + audit-C4-7
closure (narrow `except Exception` in check.main() into env-error
arms + wrap finally drain).

| fix-sweep label | What changed | Audit-doc cross-ref | C8 verdict |
|---|---|---|---|
| C7-1 close | Removed cycle-7 lines 2208-2211 from `_compatible` (the TyMemTier × (TyVar/TySize) carve-out, both orders); updated the in-method comment to explain the rationale (per-callsite hard-error preferred over silent acceptance) | C7-1 (cycle-7 silent-failure LOW) | **closed** — `_compatible(TyMemTier, TyVar)` and `_compatible(TyMemTier, TySize)` both return False post-fix; the four affected callsites (body / let / if / match) now emit clean diagnostics for TyMemTier-mismatched-T pairs |
| C4-7 close (FileNotFound family) | New `except (FileNotFoundError, PermissionError, IsADirectoryError, NotADirectoryError)` arm at lines 282-285 → `print(f"helixc: {e}")` + rc=2 | audit-C4-7 (cycle-4 silent-failure MEDIUM) | **closed for the common case** — file-not-found, permission-denied, is-a-directory all get clean env-error attribution; **opens C8-2** for the pre-formatted-message case |
| C4-7 close (UnicodeDecodeError) | New `except UnicodeDecodeError` arm at lines 286-291 → `print(f"helixc: encoding error reading source: {e}")` + rc=2 | audit-C4-7 (cycle-4 silent-failure MEDIUM) | **closed** — encoding errors now attribute correctly with a clean diagnostic; no fresh silent surface |
| C4-7 close (ImportError) | New `except ImportError as e` arm at lines 292-296 → `print(f"helixc: import error: {e}")` + rc=2 | audit-C4-7 (cycle-4 silent-failure MEDIUM) | **closed at startup-import case but opens C8-1 at lazy-import case** — the 18 lazy `from .` imports inside `_main_inner` are genuine compiler bugs when they fail, now mis-attributed as env errors |
| C4-7 close (finally drain wrap) | Wrapped the AD-warning drain in `try/except Exception as drain_e` → `print` warning + leave rc unchanged | audit-C4-7 (cycle-4 silent-failure MEDIUM, recommendation 3) | **closed** — drain failures no longer mask the primary failure; the broad-`except` is acceptable per the comment rationale (narrow surface, fail-safe by design); no fresh silent finding |

### Specific re-verifications from the audit instructions

- **C7-1 carve-out drop**: verified via direct file inspection at
  lines 2241-2244 — only the both-MemTier and the strict-
  separation arms remain. Python probe at 5d1ca24:
  `_compatible(TyMemTier('WorkingMem', TyPrim('i32')), TyVar('T'))`
  returns False; `_compatible(TyMemTier('WorkingMem',
  TyPrim('i32')), TySize('N'))` returns False;
  `_compatible(TyMemTier('WorkingMem', TyPrim('i32')),
  TyMemTier('WorkingMem', TyPrim('i32')))` returns True (same-tier
  same-inner still passes); `_compatible(TyMemTier('WorkingMem',
  TyPrim('i32')), TyMemTier('LongTermMem', TyPrim('i32')))`
  returns False (strict-separation preserved); `_compatible(
  TyVar('T'), TyVar('T'))` returns True (same-name id-fn still
  passes via `a == b`).
- **`_size_compatible` unchanged**: verified at lines 2232-2246
  — the cycle-7 shape-position cascade helper is unchanged.
  `_size_compatible(TyVar('T'), TyPrim('i32'))` returns True
  (shape-position cascade-pass). The cycle-7 close of cycle-6
  C6-1 is preserved.
- **C4-7 close coverage**: probed via direct
  `python -m helixc.check /nonexistent/foo.hx` — lands in the
  inner `os.path.exists` check at _main_inner:353 (returns rc=2
  cleanly, never reaches the new outer catch). Probed via
  injected stdlib-strict + missing stdlib file — lands in the
  new outer `except (FileNotFoundError, ...)` arm and emits the
  duplicate-prefix message (C8-2). Probed via injected corrupted
  internal import (delattr `monomorphize_structs`) — lands in
  the new outer `except ImportError` arm and mis-attributes as
  env error rc=2 (C8-1).
- **Typecheck regression**: ran `pytest helixc/tests/
  test_typecheck.py` at 5d1ca24 — 105/105 pass. No existing test
  fails due to the G2 carve-out drop. (The audit-stage28-8-cycle7
  audit had flagged that `test_c6_1_compatible_tyvar_not_top_
  cascade` covers the TyVar/TyPrim narrow but not the TyMemTier
  side; cycle 8 closed the silent-acceptance hole but added no
  test asserting the new strict behavior — see "Test-coverage
  gap" below.)
- **Test-coverage gap**: cycle 8 closed the C7-1 LOW but added
  NO regression test for `_compatible(TyMemTier, TyVar) is False`
  / `_compatible(TyMemTier, TySize) is False`. A future refactor
  could re-introduce the carve-out (or a different silent-
  acceptance window for TyMemTier-vs-generic-T) without any
  test detecting it. This is the same coverage-gap pattern the
  cycle-7 audit flagged (C7-1 recommendation 4); cycle-8 closed
  the fix half but not the test half. **NOT flagged as a new
  finding** per the user directive — this carryover concern was
  already documented in C7-1 recommendation 4. Listed in the
  "Deferred observations" section below for the cycle-9
  housekeeping batch.

### Carryover findings status (cycles 1-7)

The cycle-8 fix-sweep CLOSED audit-C4-7 (cycle-4 MEDIUM). Did NOT
re-attempt the following still-open carryover findings:

| Carryover | Severity | Cycle-8 status |
|---|---|---|
| audit-C4-1 (D2 Call-RHS i32 SIGILL) | CRITICAL | **still open** — cycle-8 did not address; deferred per cycle-6's commit message pending parse-time constant folding or a typecheck pass before closure capture |
| audit-C4-4 (D9 paper-only) | HIGH | **still open** — not addressed |
| audit-C4-7 (check.py `except Exception`) | MEDIUM | **CLOSED by cycle 8** (this commit). Opens C8-1 MEDIUM + C8-2 LOW as secondary effects. |
| audit-C4-8 deferred (check.py doesn't call fn-mono) | LOW | **still open** — not addressed |
| monomorphize_safe docstring drift (cycle-6 deferred) | (not a finding) | **still open** — docstring still suggests callers MAY ignore diags; only caller now aborts |
| D-vs-Quote diagnostic text (cycle-7 deferred) | (not a finding) | **still open** — Quote-wrapped case still emits "(one side D-wrapped, other bare)" |
| C7-1 (G2 TyMemTier carve-out asymmetry) | LOW | **CLOSED by cycle 8** (this commit). Net delta: -1. |

These are NOT re-flagged as new cycle-8 findings per the user
directive (already documented in cycles 1-7, did not CHANGE in
cycle 8's fix-sweep). They remain in the open-findings ledger.

### Specific items checked clean in cycle 8 (no new finding)

- The cycle-7 `_size_compatible` helper is unchanged. Shape-
  position cascade for TyVar/TySize still works correctly; size
  positions accept generic-T as deferred without leaking to
  value positions.
- Both-MemTier `_compatible` arm at line 2241-2242 still
  requires `a.tier == b.tier` AND inner-compatible. Strict
  separation between WorkingMem / LongTermMem / EpisodicMem /
  SemanticMem etc. is preserved. The 105 typecheck tests
  including the cross-tier-confusion regression suite at
  test_typecheck.py:288-391 all pass.
- The finally drain wrap at check.py:315-328 uses a broad
  `except Exception as drain_e` but logs a clear warning naming
  the exception class + message. Does NOT modify rc — the
  rationale (drain failures should not mask the primary
  failure) is correct. The broad-`except` here is justified by
  the narrow surface (`_drain_ad_warnings` only touches the
  module-level `_DIFF_WARNINGS` list).
- The UnicodeDecodeError arm at check.py:286-291 has a single
  origin path (`f.read()` at _main_inner:357-358) and emits a
  clean, single-prefix `helixc: encoding error reading source:
  ...` message. No catch-and-rewrap layering issue here.
- The TyVar same-name pass (`fn id[T](x: T) -> T { x }`) is
  preserved post-cycle-8 — `_compatible(TyVar('T'),TyVar('T'))`
  still returns True via `a == b` (the dataclass equality on
  `name`). No regression on the canonical id-fn pattern.
- The call-boundary check at typecheck.py:746-747 still
  symmetrically skips `_compatible` when EITHER side is
  TyVar/TySize/TyUnknown. So the cycle-8 drop of the G2 carve-
  out doesn't affect call-argument checking — generic-T
  arguments passed to TyMemTier parameters still defer to mono
  at the call boundary (same behavior as pre-cycle-7). The drop
  only restores hard-rejection at the four body/let/if/match
  callsites.

---

## Cross-stage interactions checked

- **C8-1 ImportError arm + lazy import ordering**: the lazy
  imports inside `_main_inner` are positionally distributed —
  some run early (struct_mono at line 413, after typecheck), some
  run only on specific flag combinations (PTX emit at line 629
  only when `--emit-ptx` is set). A refactor that breaks the
  ptx-only path would now mis-attribute as env error ONLY when
  `--emit-ptx` is used; CPU-only flag combinations would never
  see it. The intermittent / flag-gated nature of the silent
  failure makes it harder to detect in CI than a uniform
  startup-import failure.
- **C8-2 duplicate-prefix + non-strict stdlib path**: the non-
  strict path at parser.py:1588 prints `helixc: stdlib file
  missing: <path>` to stderr and `continue`s past the missing
  file. The pre-formatted prefix here is correct for that path
  (no catch-and-rewrap). Only the strict path (raise
  FileNotFoundError with the same pre-formatted message) hits
  the duplicate-prefix. Internally consistent on the parser side,
  externally inconsistent on the check.py side.
- **C8-1 + audit-C4-1 interaction**: audit-C4-1 (CRITICAL, still
  open) is a SIGILL trap, not an ImportError. The new
  ImportError arm doesn't affect the audit-C4-1 reproducer path.
  No new cross-stage silent surface.
- **`_compatible` recursion safety post-cycle-8**: the
  inner-recursion calls at typecheck.py:2242 (`self._compatible(
  a.inner, b.inner)` for both-MemTier) now bottoms out at
  TyPrim/TyTensor/TyArray/TyTuple — no longer cascades through
  the dropped G2 arm. The strict-separation rule's correctness
  is preserved because both-MemTier was checked BEFORE the
  carve-out arms (which is why cycle 7's G2 only affected the
  asymmetric pair). Cycle 8's drop is a pure simplification.
- **Drain-failure swallow vs primary success**: at check.py
  lines 315-328, if `_main_inner` returned rc=0 (success) and
  the drain raises, the new wrap prints a `helixc: warning: AD-
  warning drain failed: ...` and `rc=0` propagates. The user
  sees success exit code but a stderr warning. CI tools that
  branch only on rc may miss the warning — but this is a
  conscious trade-off (fail-safe drain) and matches the cycle-5
  C4-6 audit recommendation 3. Not a new finding.

---

## Deferred / out-of-scope observations (NOT new findings; cycle-9 candidates)

- **Test-coverage gap for C7-1 close**: cycle 8 closed C7-1 but
  added no regression test asserting `_compatible(TyMemTier,
  TyVar) is False` / `_compatible(TyMemTier, TySize) is False` /
  the body-vs-return diagnostic for the TyMemTier-T pattern. The
  cycle-7 audit (C7-1 recommendation 4) had asked for these
  tests. Recommend the cycle-9 fix batch add 3-4 small
  regression tests:
  - `test_c7_1_compatible_tymem_tier_tyvar_rejects`
  - `test_c7_1_compatible_tymem_tier_tysize_rejects`
  - `test_c7_1_body_return_tymem_tier_vs_tyvar_error`
  - `test_c7_1_both_memtier_same_tier_inner_still_passes`
  (regression-guard for the both-MemTier arm at line 2241-2242).
- **Carryover audit-C4-1 (D2 Call-RHS i32 SIGILL)**: still open
  CRITICAL. Cycle 8 did not address. The cycle-6 commit's
  deferral rationale still applies. **HIGHEST-PRIORITY ITEM** for
  cycle 9 — the still-open CRITICAL is the strongest obstacle to
  the strict-clean criterion for the 5-cycle-clean gate.
- **Carryover audit-C4-4 (D9 paper-only)**: still open HIGH. Not
  addressed.
- **Carryover audit-C4-8 (check.py doesn't call fn-mono)**:
  still open LOW. Not addressed.
- **monomorphize_safe docstring drift**: still open (cycle-6
  deferred). Docstring suggests callers MAY ignore diags; the
  only caller (x86_64.py) now aborts. Could be addressed by
  rewriting the docstring or refactoring `Monomorphizer.run`
  per-instance.
- **D-vs-Quote diagnostic text**: still open (cycle-7 deferred).
  Quote-wrapped case still emits "(one side D-wrapped, other
  bare)" — imprecise but not a CHANGED behavior. Could be
  generalized in a cycle-9 housekeeping batch.
- **Cycle-8 commit message coverage**: the cycle-8 commit
  message describes ONLY the C7-1 close (G2 carve-out drop).
  The simultaneous audit-C4-7 closure (check.py narrowing) is
  NOT mentioned in the message body. Not a silent-failure
  finding but a documentation-drift observation — a future
  audit reading only the commit message would be surprised by
  the check.py diff. Recommend the cycle-9 fix-sweep commit
  message explicitly enumerate ALL findings being closed (and
  any opened) per cycle.

---

## Summary

| #    | Severity | Location                                                    | Finding                                                                                                            |
|------|----------|-------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| C8-1 | MEDIUM   | check.py:292-296 + 18 lazy `from .` import sites in `_main_inner` | cycle-8 `except ImportError as e` mis-attributes genuine internal compiler import bugs as env errors (rc shifts 1→2, "please file an issue" hint suppressed) |
| C8-2 | LOW      | check.py:282-285 + parser.py:1582-1588                      | cycle-8 `except (FileNotFoundError, ...) as e: print(f"helixc: {e}")` duplicates the `helixc: ` prefix when catching a pre-formatted FileNotFoundError from `_merge_stdlib` |

**Total: 2 new findings (0 CRITICAL, 0 HIGH, 1 MEDIUM, 1 LOW).**

---

## Cycle 8 status

**Cycle 8 NOT clean.** Per the strict criterion (zero findings of
ANY severity), the 1 MEDIUM + 1 LOW new findings BLOCK the
cycle-8 clean determination.

### Stop-the-line determination: **NO**

C8-1 is MEDIUM (mis-attributes compiler bug as env error; user
sees rc=2 instead of rc=1 and loses the "please file an issue"
hint). C8-2 is LOW (cosmetic duplicate-prefix). Neither is a
data-loss / miscompile / silent-acceptance class issue. The
cycle-8 fix-sweep made real progress — closed audit-C4-7 MEDIUM
(still open since cycle 4) AND C7-1 LOW (from cycle 7), net -2
closed, +2 new (one of equal severity, one lower). Severity
trend is non-monotone for the first time since cycle 4 (cycle 7
had 0 MEDIUM; cycle 8 has 1 MEDIUM), but the new MEDIUM is the
secondary effect of closing a MEDIUM — a defensible trade.

Both new findings are mechanical-fix:
- C8-1: move `ImportError` to the compiler-bug arm (preferred);
  3-line diff.
- C8-2: strip-already-prefixed in the new outer arms (preferred);
  5-line diff.

Recommend addressing in the cycle-9 fix batch alongside the
still-open carryover audit-C4-1 (CRITICAL — top priority).

### Cycle 8 → NEW FINDINGS COUNT for the strict-clean gate: 2 (0 CRITICAL + 0 HIGH + 1 MEDIUM + 1 LOW) — clean-counter remains at 0.

### Severity trend across cycles

- Cycle 1: 13 findings (3 HIGH, 5 MEDIUM, 5 LOW).
- Cycle 2: 6 findings (1 HIGH, 4 MEDIUM, 1 LOW).
- Cycle 3: 6 findings (0 HIGH, 4 MEDIUM, 2 LOW).
- Cycle 4: 8 findings (1 CRITICAL, 2 HIGH, 4 MEDIUM, 1 LOW).
- Cycle 5: 4 findings (0 CRITICAL, 0 HIGH, 2 MEDIUM, 2 LOW).
- Cycle 6: 1 finding (0 CRITICAL, 0 HIGH, 1 MEDIUM, 0 LOW).
- Cycle 7: 1 finding (0 CRITICAL, 0 HIGH, 0 MEDIUM, 1 LOW).
- Cycle 8: 2 findings (0 CRITICAL, 0 HIGH, 1 MEDIUM, 1 LOW). ← here

Trend: severity-monotone-decrease was broken by cycle 8 (cycle 7
had 0 MEDIUM; cycle 8 has 1 MEDIUM). But the new MEDIUM is the
secondary effect of CLOSING the still-open cycle-4 audit-C4-7
MEDIUM. Net: -1 MEDIUM closed, +1 MEDIUM opened in a different
location with a different attribution defect. The cycle-8 commit
delivered real value (closed 2 carryover findings: 1 LOW + 1
MEDIUM) at the cost of 2 new findings (1 MEDIUM + 1 LOW).

### Estimated remaining open findings going into cycle 9

- Cycle 1: 13 new (all fixed → 0 open).
- Cycle 2: 6 new (all fixed → 0 open).
- Cycle 3: 6 new (all fixed → 0 open).
- Cycle 4 silent-failure: 8 new — 6 closed by cycles 5-8 (now
  including audit-C4-7 closed by cycle 8). 2 still open:
  audit-C4-1 CRITICAL, audit-C4-4 HIGH. Net: 2 still open.
- Cycle 4 type-design (sibling audit): partial close — E3
  closed via C5-4/F3; E1 closed via C5-2/F1 mechanism; others
  unchanged.
- Cycle 4 codereview (sibling audit): 0 new (was already clean).
- Cycle 5 silent-failure: 4 new — all 4 CLOSED by cycle 6.
- Cycle 6 silent-failure: 1 new (C6-1) — CLOSED for all
  positions by cycles 7-8 (cycle 8's C7-1 close finished the
  TyMemTier remnant). Net: 0 open.
- Cycle 6 type-design: 2 new (G1, G2) — both CLOSED.
- Cycle 7 silent-failure: 1 new (C7-1) — CLOSED by cycle 8.
- Cycle 8 silent-failure: 2 new (C8-1, C8-2 open).
- Prior audits (stage 5-6 + 7-8 + 9-16): ~20 still-open
  (unchanged going into cycle 8 — cycle 8 didn't touch them).
- Cycle 8 net: 20 + 2 + (deferred type-design partial) + 2 =
  **≥24 open findings** going into cycle 9. (Roughly unchanged
  from end of cycle 7 — cycle 8 closed C7-1 + audit-C4-7 but
  opened C8-1 + C8-2. Net delta: 0.)

Recommend prioritizing in this order for the cycle-9 fix batch:
1. **C8-1** (MEDIUM — move ImportError catch to compiler-bug
   arm per recommendation 1; 3-line diff + 1 regression test).
2. **C8-2** (LOW — strip-already-prefixed in outer arms per
   recommendation 1; 5-line diff + 1 regression test).
3. **C7-1 close test-coverage gap** (housekeeping — add 4
   regression tests for `_compatible` TyMemTier/TyVar/TySize
   matrix per the C7-1 recommendation 4 carryover).
4. **audit-C4-1** (CRITICAL — still-open from cycle 4;
   highest-priority unaddressed-CRITICAL; deferred in cycles
   6-8; the carryover deadline approaches as cycles 1-8
   progress).
5. **audit-C4-4** (HIGH — D9 paper-only).
6. **monomorphize_safe docstring drift** (housekeeping).
7. **D-vs-Quote diagnostic text** (housekeeping).
8. **Cycle-9 commit message coverage** (housekeeping — explicit
   enumeration of all findings being closed/opened).

After this batch lands, cycle 9 should re-audit. The "5 clean
cycles before Phase 0 deprecation" goal requires the strict
criterion (zero findings of any severity) to be met for
5 CONSECUTIVE cycles — cycle 8 is the 8th cycle and is NOT
clean, so the clean-counter remains at 0. Cycle 8 has 2
mechanical-fix findings + 1 still-open CRITICAL carryover;
the strongest realistic path to a clean cycle 9 is: address
C8-1 + C8-2 (both 1-line-class fixes), address audit-C4-1
(CRITICAL — the largest single risk to the clean gate), and
land 4 regression tests covering both the C7-1 close and the
C8-1/C8-2 closes. If executed cleanly, cycle 9 has a credible
shot at being the first clean cycle of the 5-clean gate.
