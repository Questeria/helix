# Stage 28.8 Cycle 13 — Silent-Failure Audit

**Date**: 2026-05-11
**Commit**: 98834de (read-only audit). Commits since c2e36d4:
`git log --oneline c2e36d4..HEAD` →
- 98834de "Audit 28.8 cycle 12: persist 3 cycle-12 CLEAN audit docs"
- df825ac "Audit 28.8 cycle 11: persist 3 cycle-11 CLEAN audit docs"
- 9685c3a "Persist cycle-10 audit docs (silent-failures + type-design + codereview)"

All three commits are **DOC-ONLY**. `git diff --stat c2e36d4..HEAD`
shows changes exclusively under `docs/audit-stage28-8-cycle{10,11,12}-*.md`
(9 new doc files, 3960 lines of audit prose). **Zero
production-code (.py outside tests, .hx) delta.** Confirmed by
`git diff c2e36d4..HEAD -- 'helixc/*.py' 'helixc/stdlib/*.hx'
'helixc/bootstrap/*.hx' ':(exclude)helixc/tests'` returning empty.

**Scope**: All Helix source — `helixc/bootstrap/*.hx`,
`helixc/frontend/*.py`, `helixc/ir/*.py`, `helixc/backend/*.py`,
`helixc/stdlib/*.hx`. The cycle-13 lens is identical to cycles
10/11/12: any silent-failure window NOT already counted in
cycles 1-12 as a carryover. Documented carryovers (audit-C4-1
CRITICAL, audit-C4-4 HIGH, audit-C4-8 LOW, C5-10 LOW,
monomorphize_safe docstring drift, D-vs-Quote diagnostic text,
C7-1 test-coverage gap) are NOT re-flagged per the user's
strict re-flag rule (a carryover is re-flagged only if it
CHANGED since the prior cycle — and none did, because no
production code has changed since c2e36d4).

**Trigger**: pre-Stage-29 audit gate — Cycle 13 of the 5+
clean-cycle gate. Cycle 10 was the 1st clean cycle (counter
1/5), cycle 11 the 2nd (2/5), cycle 12 the 3rd (3/5). Cycle
13 is the re-stability check at 98834de.

**Strict criterion** (per user directive 2026-05-10): cycle
counts CLEAN only when **zero new findings of ANY severity**
(CRITICAL/HIGH/MEDIUM/LOW) at the confidence-greater-than-80
reporting threshold. Findings already in the carryover ledger
are explicitly excluded.

**Rotation for fresh-eyes coverage**: cycle 12 emphasized
lower_ast.py try/finally scope at :596 + :1800, backend/x86_64.py
defensive attrs.get, backend/ptx.py + elf_dyn.py zero-except,
ir/tile_ir.py + tir.py zero-raise. Cycle 13 rotates the
spot-check surface to lightly-covered modules:
- `helixc/frontend/grad_pass.py:639-643` (try/except
  AttributeError, TypeError on frozen-dataclass cache-attach
  fallback)
- `helixc/frontend/pytree.py:293-296` (try/except ValueError
  in `validate_pytree` — diagnostics-collection pattern)
- `helixc/frontend/hash_cons.py:335` (single `raise
  HashConsError`, no try/except)
- `helixc/frontend/flatten_impls.py:88` (single `raise
  DuplicateMethodError`)
- `helixc/frontend/flatten_modules.py:67, 77` (`raise
  FlattenError` only)
- `helixc/frontend/trace_pass.py:67` (single `raise
  OverflowError`)
- `helixc/ir/passes/effect_check.py:228` (single `raise
  EffectError`)
- `helixc/examples/dashboard_server.py:117-165` (multiple
  try/except — verified NON-PRODUCTION: example/, not on
  the build path).

**Method**:
1. `git log --oneline c2e36d4..HEAD` listed three commits, all
   doc-only. `git diff --stat c2e36d4..HEAD` confirmed no .py
   outside docs and no .hx changes since cycle 10. `git diff
   c2e36d4..HEAD -- 'helixc/*.py' 'helixc/stdlib/*.hx'
   'helixc/bootstrap/*.hx' ':(exclude)helixc/tests'` returned
   empty — definitive evidence that production-code surface
   is identical to cycle 10.
2. Read the cycle-10, cycle-11, and cycle-12 silent-failures
   audit verdicts (all 0 new findings, all CLEAN). Enumerated
   carryovers excluded from cycle-13 lens.
3. Walked the full production `except` ledger again,
   cross-checking against the cycle-11 and cycle-12
   enumerations:
   - `helixc/check.py:299` `except (FileNotFoundError,
     PermissionError, IsADirectoryError, NotADirectoryError)`:
     audited in cycle 5 (C4-7 close), narrowed correctly.
   - `helixc/check.py:303` `except UnicodeDecodeError`:
     audited in cycle 5 (C4-6 close).
   - `helixc/check.py:306` `except Exception` (broad arm):
     audited in cycles 5/8/9 (ImportError arm correctly
     dropped in cycle 9 -> broad arm catches internal bugs
     with rc=1 + "compiler bug" tag). Still correct.
   - `helixc/check.py:332` `except Exception as drain_e`
     (finally drain suppressor): audited in cycle 5 (C4-6
     wrap), emits a warning rather than masking.
   - `helixc/check.py:382` `except ParseError`, `:443` `except
     DuplicateMethodError`, `:676` `except OSError`:
     domain-typed narrow excepts that surface user-facing
     diagnostics with rc!=0. No silent failure.
   - `helixc/check.py:618,649,663` (backend-call wraps for
     --emit-asm / --emit-ptx / -o): Audit 28.8 A9 pattern,
     emits "internal error" + "compiler bug" + rc=1.
   - `helixc/frontend/autodiff.py:155` `except (TypeError,
     ValueError, AttributeError)`: cycle-2 narrowing (C2-1
     observation #20), emits AD warning on hash failure. No
     regression.
   - `helixc/frontend/autodiff.py:1012` `except (OverflowError,
     ZeroDivisionError, ValueError, TypeError)`: cycle-2
     narrowing (C2-1 observation #19). No regression.
   - `helixc/frontend/autotune.py:80` `except ValueError`:
     cycle-2 narrowing.
   - `helixc/frontend/deprecated_pass.py:132` `except
     TypeError`: cycle-2 narrowing.
   - `helixc/frontend/grad_pass.py:639-643` `try: fn.
     _helix_grad_cache = cache / except (AttributeError,
     TypeError): pass`: cycle-13 fresh spot-check. The
     comment "FnDecl might be a frozen dataclass — fall
     back to no-cache" explains why the assignment can
     raise on frozen dataclasses. AttributeError +
     TypeError are the exact Python exception types raised
     when assigning to a frozen dataclass attribute
     (`FrozenInstanceError` is a subclass of
     `AttributeError`). The `pass` is a documented
     "fall back to no-cache" path — subsequent grad
     calls will recompute. This is a CORRECTNESS-
     preserving fallback (correctness is not weakened by
     missing the cache, only performance). The exception
     types are tightly narrow and the contract is
     correctly documented. **Not a silent failure** —
     the next call recomputes from scratch and the
     compiler emits identical code. No user-visible
     symptom, no debug-time confusion. Stable
     non-finding. Never flagged in cycles 1-12.
     Confirmed non-finding for cycle 13.
   - `helixc/frontend/grad_pass.py:641` `except
     (AttributeError, TypeError)`: same site as above —
     type-narrow, pre-existing, fine.
   - `helixc/frontend/lexer.py:401` `except ValueError`:
     type-narrow integer literal parse, fine.
   - `helixc/frontend/monomorphize.py:203` `except ValueError`:
     pre-existing, type-narrow. Re-checked: when `int(size_<N>)`
     fails, returns original `expr` so downstream typecheck
     flags the bad size literal. Not silent.
   - `helixc/frontend/monomorphize.py:759` `except
     ShapeFoldError`: domain-typed, returns `(0, [str(e)])`
     for the caller (x86_64.py) which aborts. Docstring drift
     (cycle-6 housekeeping carryover) NOT re-flagged.
   - `helixc/frontend/panic_pass.py:97` `except TypeError`:
     cycle-2 narrowing. Fine.
   - `helixc/frontend/parser.py:375` `except ValueError`:
     re-raises as `ParseError(f"bad integer literal {t.value!r}",
     t)` — user-visible diagnostic, not silent.
   - `helixc/frontend/struct_mono.py:448,454` `except
     ShapeFoldError`, `except ValueError`: pre-existing,
     narrow.
   - `helixc/frontend/pytree.py:293-296` `try: flatten_pytree
     (decl, struct_decls) / except ValueError as e: diags.
     append(str(e))`: cycle-13 fresh spot-check. The function
     `validate_pytree` is a diagnostics-collection wrapper
     around `flatten_pytree` (which raises ValueError on the
     11+ structural-validity error sites enumerated in
     pytree.py:135, 141, 144, 147, 171, 230, 232, 247, 252,
     264, 281). The wrapper catches ValueError, converts the
     error message to a diagnostic string via `str(e)`, and
     returns a `list[str]` of diagnostics. Callers
     (typecheck.py) merge these into the unified diagnostic
     stream. The ValueError is narrowly typed to the domain
     (every raise site uses ValueError; no broader
     exception types are expected), and the diagnostic
     message preserves the full raise-site text including
     filename/decl-name context. **Not a silent failure** —
     the error is surfaced as a user-visible diagnostic
     through the standard typecheck output path. Stable
     non-finding. Never flagged in cycles 1-12. Confirmed
     non-finding for cycle 13.
   - `helixc/frontend/typecheck.py:415,423`: `except
     TypeError_` (domain type) — appends to `self.errors`,
     surfaced as diagnostics. Not silent.
   - `helixc/frontend/typecheck.py:636`: `except ValueError`
     in `_size_type_to_lin` — function signature is
     `Optional[LinExpr]`, so returning `None` on `int(size_<N>)`
     failure matches the contract (caller handles None
     explicitly). Not silent.
   - `helixc/frontend/unsafe_pass.py:93` `except TypeError`:
     cycle-2 narrowing. Fine.
   - `helixc/frontend/diagnostics.py:76` `except Exception`
     (isatty fallback to False): defensive UX fallback (no-
     color is the right fail-safe when stream.isatty() raises).
     Pre-existing, never flagged in cycles 1-12. Re-confirmed
     non-finding for cycle 13.
   - `helixc/ir/lower_ast.py:282, 2066` `except ValueError`:
     pre-existing narrow.
   - `helixc/ir/lower_ast.py:596, 1800` `try: ... finally:
     self._pop_scope()`: scope-management pattern, exception
     propagates upward unchanged. Cycle-12 fresh spot-check.
     Re-confirmed for cycle 13.
   - `helixc/ir/lower_ast.py:2115` `except Exception` (quote-
     handle structural_hash -> `_pretty` fallback): **C5-10 LOW
     carryover**, deferred and never closed. Not re-flagged
     for cycle 13 (no change since cycle 5).
   - `helixc/ir/passes/const_fold.py:250, 324, 349, 401`
     `except Exception` (arithmetic fold returns `None` /
     `False` on failure to fall back to runtime): defensive
     fail-open-to-runtime — the correct semantic for a fold
     pass is "leave the op alone if folding fails". The
     runtime semantics are unchanged. Not silent. Stable
     non-finding.
4. Cycle-13 fresh-eyes walk of raise-only modules
   (no try/except in production code):
   - `helixc/frontend/hash_cons.py:335` `raise HashConsError
     (...)`: domain-typed error from
     `helixc/frontend/hash_cons.py`'s structural-hash
     consistency check. Propagates upward through callers
     to the top-level error handler in `check.py`. Not
     silent.
   - `helixc/frontend/flatten_impls.py:88` `raise
     DuplicateMethodError(...)`: caught at `check.py:443`,
     emits user diagnostic + rc != 0. Not silent.
   - `helixc/frontend/flatten_modules.py:67, 77` `raise
     FlattenError(...)`: domain-typed flatten-failure,
     propagates to `check.py` error handler. Not silent.
   - `helixc/frontend/trace_pass.py:67` `raise OverflowError
     (...)`: tracer recursion-bound trip. Caught at the
     trace_pass call site in typecheck.py (which builds
     a user-facing diagnostic). Not silent.
   - `helixc/ir/passes/effect_check.py:228` `raise
     EffectError("\n".join(errs))`: aggregates effect
     diagnostics into a single domain-typed error, caught
     by the typecheck driver. User sees the full
     diagnostic list. Not silent.
5. Cycle-13 fresh-eyes walk of `helixc/examples/
   dashboard_server.py:117, 122, 136, 163` (4 try/except
   sites): confirmed this file is under `examples/`, not
   on the compile path. The build system (check.py,
   parser.py, monomorphize.py, lower_ast.py, x86_64.py,
   ptx.py, elf_dyn.py) does NOT import dashboard_server.
   Verified by `grep -r 'dashboard_server' helixc/` — only
   self-reference and test references. **NOT production
   code** — out of scope for the silent-failure lens (same
   as `helixc/tests/*`).
6. Verified `helixc/backend/{x86_64.py, ptx.py, elf_dyn.py}`:
   zero `try` / `except`. The three backends are exception-
   transparent (any internal Python error propagates up to
   `check.py:618/649/663`'s broad-Exception wrap which emits
   "internal error" + "compiler bug" + rc=1). Not silent.
7. Verified `helixc/ir/{tile_ir.py, tir.py}`: zero `try` /
   `except` / `raise`. Pure IR containers. Not applicable to
   the silent-failure lens.
8. Re-walked all `except: pass` patterns. **Zero matches in
   production code** (grep `except\s*:\s*pass`). Note: the
   only `pass` in an except arm in PRODUCTION is
   `grad_pass.py:643` — but this is `except (AttributeError,
   TypeError): pass` (narrow, documented fallback), NOT a
   bare `except: pass`. Re-confirmed non-finding.
9. Confirmed no production code adds try/finally suppression
   of the outer exception since cycle 12. The cycle-3 C3-3
   `try/finally` in `check.main` correctly captures the
   inner rc and propagates the outer exception type to the
   appropriate typed-except arm.
10. Verified `check.py:_emit_env_error` is invariant under
    re-reading at HEAD (98834de is doc-only since c2e36d4,
    which is identical to the cycle-12 HEAD).
11. Re-ran `pytest helixc/tests/test_typecheck.py
    helixc/tests/test_cli.py` at HEAD: **149/149 pass**
    (test_typecheck.py 111/111 + test_cli.py 38/38).
    Identical to cycle-10/11/12 verdict.

**Result**: **0 new findings (0 CRITICAL, 0 HIGH, 0 MEDIUM,
0 LOW)** — Cycle 13 is **CLEAN** for the silent-failure
audit lens. Re-audit confirms cycle-10/11/12 clean verdicts
are stable across four consecutive read-only re-walks:
- No production-code commit has landed since c2e36d4 (the
  cycle-10 commit). By construction, the production-code
  surface is identical and any clean verdict at cycle 10
  must hold at cycle 13.
- The fresh re-walk for cycle 13 rotated the spot-check
  surface (grad_pass.py:639-643 frozen-dataclass cache
  fallback, pytree.py:293-296 validate_pytree diagnostic
  collection, hash_cons.py + flatten_impls.py +
  flatten_modules.py + trace_pass.py + effect_check.py
  raise-only modules, and confirmed dashboard_server.py
  is non-production) and did not surface any silent-failure
  window that cycles 10-12 might have overlooked.
- Every `except` arm in production code is either type-
  narrow (cycle 2 narrowing), a defensive fail-safe with
  correct semantic (const_fold, diagnostics isatty,
  grad_pass frozen-dataclass cache-fallback, pytree
  diagnostic collection, lower_ast scope try/finally,
  backend attrs.get), a previously-flagged carryover
  (C5-10 lower_ast.py:2115), or a correctly-emitting
  compiler-bug arm (check.py:306/618/649/663).
- All carryovers from cycles 1-12 are excluded from the
  re-flag (per user directive).

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

## Re-audit verification on 98834de (production surface identical to c2e36d4)

The cycle-10 fix-sweep landed as a single commit (c2e36d4)
covering C9-1 plus persisted prior-cycle audit docs. The three
subsequent commits (9685c3a, df825ac, 98834de) are doc-only
persistence of cycle-10/11/12 audit deliverables. The cycle-13
re-audit therefore covers the IDENTICAL production-code surface
as cycles 10, 11, and 12.

| Re-audit pass | C10 | C11 | C12 | C13 | Stability |
|---|---|---|---|---|---|
| `_emit_env_error` strip helper (check.py:246-255) | clean | clean | clean | clean | stable |
| Outer-except topology (check.py:284-318) | clean | clean | clean | clean | stable |
| Finally drain-failure suppressor (check.py:319-337) | clean | clean | clean | clean | stable |
| Backend-call wraps (check.py:618,649,663) | clean | clean | clean | clean | stable |
| AD-warning narrowed excepts (autodiff.py:155,1012) | clean | clean | clean | clean | stable |
| const_fold defensive folds (const_fold.py:250,324,349,401) | clean | clean | clean | clean | stable |
| Quote-handle fallback (lower_ast.py:2115) | C5-10 carryover | C5-10 carryover | C5-10 carryover | C5-10 carryover | stable carryover |
| diagnostics isatty fallback (diagnostics.py:76) | non-finding | non-finding | non-finding | non-finding | stable |
| `getattr(it, "is_kernel", False)` (check.py:641) | non-finding | non-finding | non-finding | non-finding | stable |
| lower_ast.py try/finally scope at :596, :1800 | (not enumerated) | (not enumerated) | C12 fresh: clean | clean (cycle-12 finding stable) | stable |
| backend/x86_64.py attrs.get defaults | (not enumerated) | (not enumerated) | C12 fresh: clean | clean (cycle-12 finding stable) | stable |
| backend/ptx.py, elf_dyn.py zero-except | (not enumerated) | (not enumerated) | C12 fresh: clean | clean (cycle-12 finding stable) | stable |
| ir/tile_ir.py, tir.py zero-raise | (not enumerated) | (not enumerated) | C12 fresh: n/a | n/a | n/a |
| frontend/parser.py:375 ValueError -> ParseError re-raise | clean | clean | clean | clean | stable |
| frontend/typecheck.py:415,423 TypeError_ -> diag append | clean | clean | clean | clean | stable |
| frontend/typecheck.py:636 ValueError -> Optional None | clean | clean | clean | clean | stable |
| frontend/monomorphize.py:203 ValueError -> return expr | clean | clean | clean | clean | stable |
| frontend/monomorphize.py:759 ShapeFoldError -> diag list | clean | clean | clean | clean | stable |
| **frontend/grad_pass.py:639-643 frozen-dataclass cache fallback** | (not enumerated) | (not enumerated) | (not enumerated) | **C13 fresh: clean** (narrow `except (AttributeError, TypeError)`, documented fallback to no-cache, correctness preserved — next call recomputes; identical emitted code) | stable |
| **frontend/pytree.py:293-296 validate_pytree diagnostic collection** | (not enumerated) | (not enumerated) | (not enumerated) | **C13 fresh: clean** (narrow `except ValueError`, message converted to diagnostic, surfaced through typecheck output path) | stable |
| **frontend/hash_cons.py:335 raise HashConsError** | (not enumerated) | (not enumerated) | (not enumerated) | **C13 fresh: clean** (domain-typed raise, propagates to check.py error handler) | stable |
| **frontend/flatten_impls.py:88 raise DuplicateMethodError** | (not enumerated) | (not enumerated) | (not enumerated) | **C13 fresh: clean** (caught at check.py:443) | stable |
| **frontend/flatten_modules.py:67,77 raise FlattenError** | (not enumerated) | (not enumerated) | (not enumerated) | **C13 fresh: clean** (domain-typed, propagates) | stable |
| **frontend/trace_pass.py:67 raise OverflowError** | (not enumerated) | (not enumerated) | (not enumerated) | **C13 fresh: clean** (tracer recursion-bound trip; user-facing diagnostic) | stable |
| **ir/passes/effect_check.py:228 raise EffectError** | (not enumerated) | (not enumerated) | (not enumerated) | **C13 fresh: clean** (aggregates effect diagnostics; user sees full list) | stable |
| **examples/dashboard_server.py try/except sites (lines 117, 122, 136, 163)** | (not enumerated) | (not enumerated) | (not enumerated) | **C13 fresh: not applicable** (under `examples/`, not on compile path — same scope-exclusion as `helixc/tests/`) | n/a |
| No `except: pass` in production | clean | clean | clean | clean (zero matches for `except\s*:\s*pass` in production; only narrow `except (AttributeError, TypeError): pass` at grad_pass.py:643 which is documented fallback) | stable |

### Specific items re-checked clean in cycle 13

- **No new production commits -> no new code surface**: `git
  diff --stat c2e36d4..HEAD` is doc-only (3960 doc-line
  additions, 0 .py outside docs, 0 .hx). `git diff
  c2e36d4..HEAD -- 'helixc/*.py' 'helixc/stdlib/*.hx'
  'helixc/bootstrap/*.hx' ':(exclude)helixc/tests'`
  definitively returns empty. By construction the cycle-12
  clean verdict propagates to cycle 13 unless the fresh
  re-walk finds an overlooked window. None found.
- **`grad_pass.py:639-643` frozen-dataclass cache-attach
  fallback**: the `try: fn._helix_grad_cache = cache` /
  `except (AttributeError, TypeError): pass` pattern is a
  type-narrow Python-idiomatic guard. Frozen dataclasses
  raise `FrozenInstanceError` (subclass of `AttributeError`)
  on attribute assignment; some non-dataclass FnDecl-like
  objects may raise `TypeError`. Both are caught with
  surgical precision. The `pass` is a CORRECTNESS-
  preserving fallback — the cache is a performance
  optimization; the next call recomputes from scratch via
  `differentiate_reverse` or `differentiate`, producing
  identical emitted code. No user-visible symptom under
  failure. Cycle-13 fresh spot-check, not previously
  enumerated in any cycle's table. Confirmed non-finding.
- **`pytree.py:293-296` `validate_pytree` diagnostic
  collection**: the wrapper catches ValueError raised by
  `flatten_pytree` (11+ raise sites at lines 135, 141, 144,
  147, 171, 230, 232, 247, 252, 264, 281 — all narrowly
  ValueError) and converts each to a diagnostic string via
  `str(e)`. The diagnostic list is returned to the typecheck
  driver, which merges it into the unified error stream.
  The pattern is "raise-as-diagnostic" — load-bearing for
  the typecheck UX. Not silent. Cycle-13 fresh spot-check,
  not previously enumerated. Confirmed non-finding.
- **Raise-only modules `hash_cons.py:335`, `flatten_impls.
  py:88`, `flatten_modules.py:67,77`, `trace_pass.py:67`,
  `effect_check.py:228`**: all five emit domain-typed
  exceptions that propagate through the compiler driver
  (`check.py`, `typecheck.py`) to user-visible diagnostics.
  None have any `try/except` — they are pure raise sites.
  The silent-failure lens does not apply to modules that
  raise without catching. Cycle-13 fresh spot-check.
  Confirmed not-applicable to silent-failure lens.
- **`examples/dashboard_server.py`**: confirmed NON-
  PRODUCTION. The file is under `helixc/examples/` —
  parallel scope-exclusion to `helixc/tests/`. The 4
  try/except sites (lines 117, 122, 136, 163) handle
  subprocess.TimeoutExpired and KeyboardInterrupt for an
  interactive dashboard demo, not compiler code paths.
  Out of scope for the silent-failure lens.
- **`_emit_env_error` triple-prefix / uppercase-prefix
  edge cases**: still no callee triggers either. Not
  findings.

### Cross-stage interactions re-checked (cycle 13)

- **check.py outer-except + backend-wrap interaction**: if
  `helixc.backend.x86_64.compile(prog)` raises `KeyError`
  during emit (e.g., from a missing op attr that the
  defensive `.get` default doesn't cover — though all
  current `.get` calls DO have defaults), the exception
  propagates up to `check.py:663`'s broad-Exception arm
  which emits "internal error: KeyError: ..." + "compiler
  bug" + rc=1. Cycle-11 verified this; cycle-12 re-verified.
  Cycle 13 re-verifies by re-reading the backend ->
  `check.py:618-665` topology and confirming no intermediate
  `except` swallows it. Not a finding.
- **grad_pass.py:639-643 frozen-cache + cycle-2 AD-drain
  interaction**: when the cache attach fails, `cache` is
  built but not stored on `fn`. The `cache["grads"]` dict
  is still used at `grad_pass.py:648` (`_copy.deepcopy(cache
  ["grads"][var])`) for the current call. The drop-on-floor
  semantic is local to the cache attach, not the gradient
  computation. The AD-drain runs in `check.main`'s `finally`
  and emits any accumulated `_DIFF_WARNINGS` (collected by
  `differentiate`/`differentiate_reverse`); the cache-attach
  failure does not interfere with warning collection. Not
  a finding.
- **pytree.py:293-296 + typecheck driver**: the
  `validate_pytree` return value (`list[str]` of
  diagnostics) is consumed by typecheck.py callers (e.g.
  the `@derive(PyTree)` lowering pass). When non-empty, the
  driver emits each string as a diagnostic line with
  rc != 0. The cycle-13 trace through the pytree ->
  typecheck -> check.py pipeline confirms the diagnostic
  list reaches user-visible output. Not a finding.
- **monomorphize_safe -> x86_64.py abort chain**: same as
  cycle 12. Not a finding.
- **typecheck.py:_size_type_to_lin -> Presburger consumer**:
  same as cycle 12. Not a finding.

### Carryover findings status (cycles 1-12) — unchanged

The cycle-13 re-audit closed nothing (read-only by design)
and introduced no new finding. The carryover ledger is
identical to cycle 12's closing snapshot.

| Carryover | Severity | Cycle-13 status |
|---|---|---|
| audit-C4-1 (D2 Call-RHS i32 SIGILL) | CRITICAL | **still open** — not addressed. Highest-priority unaddressed-CRITICAL. |
| audit-C4-4 (D9 paper-only) | HIGH | **still open** — not addressed |
| audit-C4-8 (check.py doesn't call fn-mono) | LOW | **still open** — not addressed |
| C5-10 (lower_ast.py:2113-2117 + 2079-2092 + 2093-2101) | LOW | **still open** — not addressed; not re-flagged per the user's strict re-flag rule |
| monomorphize_safe docstring drift | (housekeeping) | **still open** — docstring still suggests callers MAY ignore diags |
| D-vs-Quote diagnostic text | (housekeeping) | **still open** — Quote-wrapped case still emits "(one side D-wrapped, other bare)" |
| C7-1 test-coverage gap | (housekeeping) | **still open** — `_compatible(TyMemTier, TyVar) is False` regression tests not added |
| C8-1 (cycle-8 MEDIUM) | MEDIUM | CLOSED by cycle 9 |
| C8-2 (cycle-8 LOW) | LOW | CLOSED by cycle 9 |
| C9-1 (cycle-9 LOW) | LOW | CLOSED by cycle 10 |
| C10 silent-failure | n/a | **0 new findings (CLEAN)** |
| C11 silent-failure | n/a | **0 new findings (CLEAN)** |
| C12 silent-failure | n/a | **0 new findings (CLEAN)** |

These are NOT re-flagged as new cycle-13 findings per the
user directive (already documented in cycles 1-12, did not
CHANGE in cycle 13 — and indeed could not have changed
because no production commit landed). They remain in the
open-findings ledger and are out-of-scope for this audit's
strict-clean determination.

---

## Deferred / out-of-scope observations (NOT new findings; cycle-14 candidates)

- **Carryover audit-C4-1 (D2 Call-RHS i32 SIGILL)**: still
  open CRITICAL. Cycle 13 did not address (read-only
  re-audit by definition). **STILL THE HIGHEST-PRIORITY
  ITEM** for any future fix-sweep — the only remaining
  CRITICAL across the audit series. As the clean-counter
  accumulates (now 4/5 if cycle 13's clean verdict holds
  across all three audit lenses), the question of whether
  the Stage-29 gate requires CRITICAL=0-open (stricter
  interpretation) or merely 5-consecutive-clean (lenient
  interpretation) becomes load-bearing. The cycle-12
  recommendation stands: prioritize audit-C4-1 in the next
  fix-sweep regardless.
- **Carryover audit-C4-4 (D9 paper-only)**: still open HIGH.
  Not addressed.
- **Carryover audit-C4-8 (check.py doesn't call fn-mono)**:
  still open LOW. Not addressed.
- **C5-10 lower_ast.py silent fallbacks**: still open LOW
  (Pattern A: quote-handle structural_hash -> _pretty
  fallback at :2115; Pattern B: Cast None inner ->
  const_int(0); Pattern C: Field no-array-match returns
  None). Not addressed; not re-flagged.
- **monomorphize_safe docstring drift**: still open
  (cycle-6 deferred).
- **D-vs-Quote diagnostic text**: still open (cycle-7
  deferred).
- **C7-1 test-coverage gap**: still open. Cycle 13 also did
  not add the 4 `_compatible(TyMemTier, TyVar)` regression
  tests.
- **`_emit_env_error` triple-prefix / uppercase-prefix
  edge cases**: still no callee triggers either. Not
  findings.

---

## Cycle 12 vs cycle 13 — clean-cycle counter check

Cycle 10 = 1st clean (counter 1/5). Cycle 11 = 2nd clean
(counter 2/5). Cycle 12 = 3rd clean (counter 3/5). The user
directive for cycle 13 explicitly instructs: re-audit the
same scope and confirm nothing has regressed; do not
re-flag prior-cycle carryovers unchanged since cycle 10.

The cycle-13 re-audit honors that directive:
- `audit-C4-1 CRITICAL`, `audit-C4-4 HIGH`, `audit-C4-8 LOW`:
  not re-flagged.
- `C5-10 LOW` (lower_ast.py:2113-2117 + 2093-2101 +
  2079-2092): not re-flagged.
- `monomorphize_safe docstring drift`, `D-vs-Quote
  diagnostic text`, `C7-1 test-coverage gap`: not
  re-flagged.

Cycle 13 produces **zero NEW findings of any severity**, so
the clean-cycle counter advances to **4/5** (cycle 10 = 1,
cycle 11 = 2, cycle 12 = 3, cycle 13 = 4) under the strict
criterion — subject to the parallel type-design +
code-review audit lenses also being clean for cycle 13.

---

## Summary

| #    | Severity | Location | Finding |
|------|----------|----------|---------|
|      |          |          | (none — cycle 13 is CLEAN for the silent-failure lens) |

**Total: 0 new findings (0 CRITICAL, 0 HIGH, 0 MEDIUM,
0 LOW).**

---

## Cycle 13 status

**Cycle 13 IS CLEAN** for the silent-failure audit lens. Per
the strict criterion (zero findings of ANY severity), the
0-finding result satisfies the clean-cycle gate for this
audit lens.

### Stop-the-line determination: **NO**

Cycle 13 is clean — no stop required for this lens.

### Cycle 13 -> NEW FINDINGS COUNT for the strict-clean gate: 0 (0 CRITICAL + 0 HIGH + 0 MEDIUM + 0 LOW) — clean-counter advances to **4/5** for this audit lens (subject to the parallel type-design + code-review audit lenses also being clean for cycle 13).

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
- Cycle 10: 0 findings.
- Cycle 11: 0 findings.
- Cycle 12: 0 findings.
- Cycle 13: 0 findings. <- here

Trend: **4 consecutive clean cycles**. Cycle 13's clean
verdict is by construction (no new production commits since
c2e36d4) plus a fresh-eyes re-walk that rotated the spot-
check surface to grad_pass.py:639-643 frozen-dataclass
cache fallback, pytree.py:293-296 validate_pytree diagnostic
collection, raise-only modules (hash_cons.py,
flatten_impls.py, flatten_modules.py, trace_pass.py,
effect_check.py), and confirmed examples/dashboard_server.
py is non-production. All confirmed non-findings (or
not-applicable). The audit series is stable at zero new
findings.

### Estimated remaining open findings going into cycle 14

- Cycle 1: 13 new (all fixed -> 0 open).
- Cycle 2: 6 new (all fixed -> 0 open).
- Cycle 3: 6 new (all fixed -> 0 open).
- Cycle 4 silent-failure: 8 new — 6 closed by cycles 5-9.
  2 still open: audit-C4-1 CRITICAL, audit-C4-4 HIGH.
- Cycle 5 silent-failure: 4 new — 3 closed by cycle 6
  (C5-5, C5-6, C5-7, C5-8 MEDIUM and C5-9 LOW), 1 still
  open (C5-10 LOW, lower_ast.py fallbacks).
- Cycle 6 silent-failure: 1 new (C6-1) — CLOSED.
- Cycle 6 type-design: 2 new (G1, G2) — both CLOSED.
- Cycle 7 silent-failure: 1 new (C7-1) — CLOSED by cycle 8.
- Cycle 8 silent-failure: 2 new (C8-1 MEDIUM, C8-2 LOW) —
  both CLOSED by cycle 9.
- Cycle 9 silent-failure: 1 new (C9-1 LOW) — CLOSED by
  cycle 10.
- Cycle 10 silent-failure: 0 new.
- Cycle 11 silent-failure: 0 new.
- Cycle 12 silent-failure: 0 new.
- Cycle 13 silent-failure: 0 new. <- here
- Prior audits (stage 5-6 + 7-8 + 9-16): ~20 still-open
  (unchanged going into cycle 14).
- Cycle 13 net: 20 + 2 (C4-1 + C4-4) + 1 (C5-10) + 0
  (cycle-13 new) + (deferred type-design partial) = **>=23
  open findings** going into cycle 14. (Net 0 delta from
  cycles 10/11/12: cycle 13 closed nothing, opened nothing.)

Recommend prioritizing in this order for the cycle-14 fix
batch (if user elects to land fixes between clean
re-audits):
1. **audit-C4-1** (CRITICAL — still-open from cycle 4;
   highest-priority unaddressed-CRITICAL; deferred in
   cycles 6-13; the carryover deadline approaches as the
   strict-clean gate accumulates clean cycles).
2. **audit-C4-4** (HIGH — D9 paper-only).
3. **C5-10** (LOW — lower_ast.py fallbacks).
4. **C7-1 test-coverage gap** (combinable with audit-C4-1
   if the fix touches typecheck.py).
5. **monomorphize_safe docstring drift** (housekeeping).
6. **D-vs-Quote diagnostic text** (housekeeping).

The "5 clean cycles before Phase 0 deprecation" goal
requires the strict criterion (zero findings of any
severity) to be met for 5 CONSECUTIVE cycles. Cycle 10 was
the 1st; cycle 11 the 2nd; cycle 12 the 3rd; cycle 13 the
4th. **One more clean cycle (14) and the gate fires** for
this audit lens. The cycle-13 re-audit confirms the
production-code surface remains stable: a re-audit on
identical production HEAD (only doc commits since c2e36d4)
with rotated fresh-eyes spot-checks finds no overlooked
silent-failure window.

**Cycle 13 status: CLEAN**
**Counter status: 4/5** (cycles 10, 11, 12, 13 all clean
for the silent-failure audit lens; subject to the parallel
type-design + code-review lenses also being clean for
cycle 13).
