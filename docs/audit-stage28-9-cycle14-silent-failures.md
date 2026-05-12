# Stage 28.9 Cycle 14 — Audit A (silent failures)

**Date:** 2026-05-11. **HEAD:** `bb3b5ec` (post C13 structural refactor: `_pattern_test_expr` unifies sub-pattern dispatch). **Lens:** silent failures. **Criterion:** ZERO findings at conf >=75 (strict). **Going in:** 0/5.

## Scope

Read-only delta `7108733..bb3b5ec` (`match_lower.py`, +86/-162). No re-flag of prior cleared findings.

## Finding C14-1 — `_dup_expr` silently overwrites source spans on Name/IntLit/Index

**Severity:** HIGH. **Confidence:** 84. **Location:** `helixc/frontend/match_lower.py` lines 307-317.

**Issue.** The explicit branches build new `Name`/`IntLit`/`Index` nodes with the *caller's* `span` parameter, discarding `expr.span`. The `deepcopy` fallback for `Path`/`Field`/`Call`/`Binary` preserves the original span. So `_dup_expr` is span-lossy for some shapes and span-preserving for others — **silently**. Downstream diagnostics on duplicated `Name`/`Index`/`IntLit` nodes (typecheck "use of unbound name", overflow checks on `IntLit`) now point at the *match-arm* span, not the original literal/identifier source location.

**Hidden errors.** Error messages with wrong source locations; users see "error at line of the `match` keyword" instead of "error at line where the integer literal was written"; impossible to triage. No log.

**Recommendation.** Pass `expr.span` (not the caller's `span`) into the new nodes, or take `span` only for *synthetic* nodes the caller is fabricating, not for nodes being duplicated from input.

## Finding C14-2 — `_pattern_test_expr` shares `pat.value` / `pat.path` / `pat.lo` / `pat.hi` across multiple parent Binary nodes

**Severity:** HIGH. **Confidence:** 82. **Location:** lines 345-346, 351-354, 378-379.

**Issue.** C13-2 fixed `slot_load` sharing by routing through `_fresh_slot_load` + `_dup_expr`, but the SAME class of bug remains on the *right* operand: `right=pat.value` (PatLit), `right=pat.lo`/`right=pat.hi` (PatRange), and `right=pat.path` (PatVariant) embed the pattern's own AST node directly. When the same `PatLit` literal appears in multiple match arms (or a `PatOr` expands to several Binary tests over the same literal alts), the SAME `IntLit`/`Path` instance becomes the `right` of N different parent `Binary` nodes. Identical tree-linearity violation that C13-2 was trying to close.

**Hidden errors.** Visitor double-walks; span-rewrite passes mutate one Binary's right and silently corrupt the sibling arm; pytree validators may assert-fail intermittently depending on visit order.

**Recommendation.** Wrap each `right=` with `_dup_expr(pat.value, pat.value.span)` (and similarly for `pat.lo`, `pat.hi`, `pat.path`). Symmetric with the C13-2 fix on the left operand.

## Finding C14-3 — `_pattern_test_expr` final fallback silently emits `BoolLit(True)` for unknown Pattern subtypes

**Severity:** MEDIUM. **Confidence:** 78. **Location:** line 393.

**Issue.** The terminal `return A.BoolLit(span=span, value=True)` catches any `Pattern` subclass not enumerated above (e.g. future `PatStruct`, `PatSlice`, `PatGuard`). Result: a new pattern type silently matches everything with no diagnostic. The cycle-7/10/11/12/13 family was exactly this class of bug; the canonical helper is supposed to close it.

**Hidden errors.** Future pattern additions become silent-accept holes; tests pass trivially; runtime behavior is "first arm wins" garbage.

**Recommendation.** Replace the trailing `return BoolLit(True)` with `raise UnsupportedPattern(...)` or emit a hard diagnostic with the pattern type name and span.

## Tally

| Severity | Count |
|---|---|
| HIGH | 2 |
| MEDIUM | 1 |
| **Total** | **3** |

**Cycle 14: NOT CLEAN.** C13 structural refactor closed the left-operand and slot-load sharing, but reintroduces (a) span loss in `_dup_expr` and (b) right-operand sharing on every literal/range/variant test, plus (c) the same trailing silent-accept the refactor was supposed to eliminate. Counter holds at 0/5.

## Files touched

None — read-only. Only this doc.
