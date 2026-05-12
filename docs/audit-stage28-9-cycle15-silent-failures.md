# Stage 28.9 Cycle 15 — Audit A (silent failures)

**Date:** 2026-05-11. **HEAD:** `e847fa9` (post C14-1/2/3: span preserve + right-dup + loud catchall in `_pattern_test_expr`). **Lens:** silent failures. **Criterion:** ZERO findings at conf >=75 (strict). **Going in:** 0/5.

## Scope

Read-only delta `bb3b5ec..e847fa9` (`match_lower.py`, `test_match.py`). No re-flag of prior cleared findings (C10-C14 family).

## Finding C15-1 — `_collect_binds` silently returns `[]` for unhandled Pattern subclasses

**Severity:** HIGH. **Confidence:** 80. **Location:** `helixc/frontend/match_lower.py` lines 444-530.

**Issue.** `_collect_binds` handles `PatBind`/`PatVariant`/`PatTuple`/`PatOr` via `if/elif` chain; any unmatched subclass falls through to `return binds` with `binds == []`. C14-3 fixed exactly this anti-pattern in the **sibling** dispatcher `_pattern_test_expr` (replacing the trailing `BoolLit(True)` with `raise NotImplementedError`). The binder dispatcher was left silent. A future `PatStruct`/`PatSlice` with named sub-binders silently emits zero `Let` stmts; the arm body's `Name` references those binders resolve to outer-scope shadows or unbound — exactly the C10-1 class of bug the family is trying to extinguish.

**Hidden errors.** New Pattern subclasses with binders silently lose them; tests pass for arms without binder usage; runtime `Name` lookup hits unrelated shadowed scope.

**Recommendation.** Mirror C14-3: add a trailing `raise NotImplementedError(f"_collect_binds: unhandled {type(pat).__name__}")`. Keep `PatLit`/`PatRange`/`PatWildcard` as explicit no-bind arms so the loud arm is reached only for genuinely new subclasses.

## Finding C15-2 — `_dup_expr` `span` parameter is silently ignored

**Severity:** MEDIUM. **Confidence:** 78. **Location:** lines 303, 313-314, 319-335.

**Issue.** C14-1 made `_dup_expr` preserve `expr.span` and demoted the `span` parameter to "unused legacy default kept for backwards-compat". Callers (e.g. `_fresh_slot_load` line 298: `_dup_expr(callee_expr, span)`) still pass a span argument and receive no warning that it has no effect. A future maintainer reading `_fresh_slot_load`'s call site reasonably infers the synthetic span propagates; it doesn't. Silent parameter no-op.

**Hidden errors.** Maintainer expects `_dup_expr(expr, my_span)` to override span; gets `expr.span` instead. No deprecation warning, no removal.

**Recommendation.** Either drop the parameter entirely (fixing call sites) or `warnings.warn(DeprecationWarning, ...)` when a non-None span is passed.

## Finding C15-3 — `_pattern_test_expr` raises bare `NotImplementedError`, bypasses project diagnostic pipeline

**Severity:** MEDIUM. **Confidence:** 76. **Location:** line 423-427.

**Issue.** The C14-3 catchall raises a Python `NotImplementedError`. Project convention (`typecheck.py`, ~30 sites) collects errors into `self.errors.append(Diagnostic(...))` for structured reporting with span + error code. The bare exception crashes `lower_matches` with a Python traceback — loud but unstructured: no span attribution, no diagnostic code, no integration with the diag-collection downstream.

**Hidden errors.** Users see a Python stack trace rather than `error[E####]: unsupported pattern X at file:line:col`; IDE diagnostics integration breaks; compiler exit code path differs from typecheck errors.

**Recommendation.** Route through `Diagnostic` with `pat.span` and a registered error code (or accumulate on a passed-in error collector). Bare Python raise is loud-but-non-canonical.

## Tally

| Severity | Count |
|---|---|
| HIGH | 1 |
| MEDIUM | 2 |
| **Total** | **3** |

**Cycle 15: NOT CLEAN.** C14 closed the right-operand sharing and span loss in `_pattern_test_expr` but the symmetric silent-accept in `_collect_binds` remains, the `_dup_expr` parameter is now a silent no-op, and the loud catchall bypasses the project diagnostic pipeline. Counter holds at 0/5.

## Files touched

None — read-only. Only this doc.
