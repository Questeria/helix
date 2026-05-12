# Stage 28.9 Cycle 13 — Audit A (silent failures)

**Date:** 2026-05-11
**HEAD:** `7108733` (post C12-1 `_sub_pat_or_test` + C12-2 test)
**Lens:** silent failures (Audit A)
**Criterion:** ZERO findings at confidence >=75% (strict).
**Cycle-clean counter (going in):** 0/5 (reset by cycle 12).

## Scope

Read-only delta `4ad80fa..7108733` (`match_lower.py` +80,
`test_match.py` +53). No re-flag of prior cleared findings.

## Finding C13-1 — `_sub_pat_or_test` else-arm silently passes nested PatTuple / PatOr alts

**Severity:** HIGH. **Confidence:** 86.
**Location:** `helixc/frontend/match_lower.py` lines 337-344.

**Issue.** The new helper covers `PatLit`/`PatRange`/`PatVariant`
explicitly and treats `PatBind`/`PatWildcard` as trivially-true
(correct, binders handled in `_collect_binds`). But the same `else`
arm also silently emits `BoolLit(True)` for *nested* `PatTuple` and
nested `PatOr` alts (e.g. `((1,2) | (3,4), _)`, parses successfully —
verified by re-parse). Result: such alts match ANY slot value while
binders may still be emitted. Same class of bug as C12-1, recurring
in the fix itself.

**Hidden errors.** False-positive arm selection for alt shapes the
parser accepts but the helper doesn't recognize; no diagnostic.

**Recommendation.** For unsupported alt shapes, raise an explicit
`UnsupportedPattern` diagnostic (or recurse via a temp through
`_pattern_test`). Do not silently widen the OR.

## Finding C13-2 — `slot_load` AST node aliased across all alt Binaries

**Severity:** MEDIUM. **Confidence:** 80.
**Location:** lines 316-336, 416, 472.

**Issue.** Caller passes one `A.Index` instance as `scrut_load`;
`_sub_pat_or_test` embeds the *same* node in every alt's Binary
`left`. The AST now has multiple parents pointing at one child,
violating tree linearity that walkers (`pytree.py`, validation
passes) rely on. Mutation in any single pass silently corrupts
sibling alts; structural-hash dedup masks divergent spans.

**Hidden errors.** Visitor double-walks, span-rewrite collisions,
diff-tool false equivalences — all silent.

**Recommendation.** Deep-copy `scrut_load` per alt, or accept a
*thunk* `() -> A.Expr` and rebuild the Index per call.

## Tally

| Severity | Count |
|---|---|
| HIGH | 1 |
| MEDIUM | 1 |
| **Total** | **2** |

**Cycle 13: NOT CLEAN.** C12-1 fix introduced two new silent
paths. Counter holds at 0/5.

## Files touched

None — read-only. Only this doc.
