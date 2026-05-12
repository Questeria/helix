# Stage 28.9 Cycle 12 — Audit A (silent failures)

**Date:** 2026-05-11
**HEAD:** `4ad80fa` (post C11-1 nested-PatOr binders fix)
**Lens:** silent failures (Audit A)
**Criterion:** ZERO findings at confidence >=75% (strict).
**Cycle-clean counter (going in):** 4/5.

## Scope

Read-only re-pass over delta `48a714e..4ad80fa`
(`match_lower.py` +16/-2, no test add) plus re-sweep of
cleared except-ledger. No re-flag of prior cleared findings.

## Finding C12-1 — nested PatOr in `_pattern_test` silently matches

**Severity:** HIGH. **Confidence:** 88.
**Location:** `helixc/frontend/match_lower.py` lines 322-403
(`_pattern_test`, PatVariant + PatTuple sub-dispatch).

**Issue.** C11-1 extended `_collect_binds` to recurse into
nested `PatOr` inside `PatVariant`/`PatTuple` (lines 431, 458),
emitting intersection-binders. But `_pattern_test` only
dispatches `PatLit`/`PatRange`/(nested-variant tag-only) on
sub-patterns. A nested `PatOr` slot falls through the chain at
lines 373-394 (variant) and 335-349 (tuple) to "trivially
true," so `Cons((A | B(x)), tail)` matches `Cons(C, tail)`
silently. The new C11-1 binders then emit a `Let x =
__sub_N[1]` loading from a non-matching slot — body sees
garbage.

**Hidden errors.** False-positive match arm selection;
binder load of wrong tag/slot data; no diagnostic.

**Recommendation.** Add nested-`PatOr` arm in both
sub-dispatches that emits `_or_chain([_pattern_test(alt,
slot_name, span) for alt in sub.alts])` against a fresh temp
matching the C11-1 `__sub` recursion. Mirror the symmetry
C11-1 established on the binder side.

## Finding C12-2 — C11-1 fix landed without regression test

**Severity:** MEDIUM. **Confidence:** 92.
**Location:** commit `4ad80fa`, +16/-2 in `match_lower.py`,
zero in `tests/test_match.py`.

**Issue.** Unlike C10-1 (which added a +51 regression in the
same cycle per cycle-11 audit doc § Drift inspection), the
C11-1 nested-PatOr fix shipped with no test pinning the
`Cons((A | B(x)), tail)` shape. The existing 28 match tests
"still pass" but none exercises the nested case the commit
message describes.

**User impact.** Regression rollback is undetectable by the
test suite. Pairs with C12-1: if C12-1 is fixed, no test
covers the joint binder+test contract.

**Recommendation.** Add a test that constructs the
`Cons((A | B(x)), tail)` AST, lowers it, and asserts both a
`Let` for `x` is emitted AND the test expression contains a
disjunction over the inner alts.

## Tally

| Severity | Count |
|---|---|
| CRITICAL | 0 |
| HIGH | 1 |
| MEDIUM | 1 |
| **Total** | **2** |

**Cycle 12: NOT CLEAN.** Counter resets to 0/5. The C11-1
fix is half a fix; the test side of the contract is missing
in `_pattern_test`, and no regression test pins either side.

## Files touched

None — read-only. Only this doc.
