# Stage 36 closure gate-3 code-review audit

**HEAD**: 97dbfbc (gate-2 commit; ranged through current working tree at 2a6aedd which is docs-only, so helixc/ diff is identical)
**Scope**: `git diff e7c3552..HEAD -- helixc/`
**Date**: 2026-05-16

## Findings

**ZERO FINDINGS — CLEAN**

Gate-3 audit lane finds no high-confidence (>= 80) code-review issues
in the cumulative Inc 15 + gate-1 fix sweep + gate-2 test additions
diff. Clean-gate counter advances 2/3 -> 3/3. Stage 36 Inc 15
closure-ready.

## Verification trail

The audit walked the full diff (539 insertions / 52 deletions across
6 files) and explicitly checked every concern enumerated in the
gate-3 task brief:

### Bugs / logic errors in the new runtime guards
- `parent_at` runtime guards in `lower_ast.py:2189-2231` correctly use
  CMP_GT/CMP_GE/CMP_LT producing 0/1 booleans (verified
  `backend/x86_64.py:2168-2197` uses SETcc family). BIT_AND of 0/1
  operands yields 0/1. SELECT semantics (`x86_64.py:2309-2367`) test
  via `test eax, eax / je`, so any nonzero condition selects arm A.
  Composition is sound: `handle > 0 AND slot >= 0 AND slot < 3` ->
  raw_read; else -> -1.
- Off-by-one check: `slot_lt_3` uses CMP_LT against constant 3, so
  slot in {0,1,2} passes and slot >= 3 fails. Matches the typecheck
  literal-bound `not (0 <= slot_literal_value <= 2)`.
- The `raw_read` (`_safe_arena_get(eff_idx, 0)`) is evaluated
  unconditionally even when guards fail; this is safe because
  `_safe_arena_get` (`lower_ast.py:2066-2120`) is itself bounds-checked
  and returns -1 sentinel on OOB - no memory-safety risk.
- The new upper guard (`slot < 3`) is NOT redundant with
  `_safe_arena_get`'s sentinel: it catches the cross-record hazard
  where `eff_idx` is in-bounds of the arena but reads sibling-record
  data (e.g. `parent_at(1, 3)` after two 2-parent derivations reads
  arena[3] = sibling's right value).

### Test coverage of new helpers
- `evidence_middle`, `evidence_third`, `trace_evidence3` each have a
  dedicated positive runtime test on a 3-parent handle
  (test_stage36_provenance.py:2516-2572).
- Stdlib visibility test (line 2009-2017) was extended to assert all
  three new helpers are parsed into the merged program.
- The absence of a negative-control test exercising
  `evidence_third`/`trace_evidence3` ON A 2-PARENT HANDLE is
  intentional: the stdlib docstring explicitly says "caller is
  responsible for knowing the handle is 3-parent" and references the
  `stage36-inc16-arity-in-handle` TODO for the deterministic-failure
  rewrite. Pinning the cross-record behavior would over-specify
  Phase-0 arena layout. Below the >=80 confidence threshold.

### Typecheck error message format
- Strict-i32 messages (typecheck.py:2960-2966) include the offending
  type via `self._fmt(arg_tys[0])` and explain WHY (truncation
  history). Tests assert the substring "must be exactly i32"
  (test:2375, 2392), pinning the format.
- Literal-slot error (typecheck.py:3025-3032) includes the actual
  slot value via interpolation. Tests assert "literal slot -1",
  "literal slot 3", "literal slot 999999" substrings (test:2410,
  2427, 2245) - format is pinned.

### Duplicated runtime guards / DRY-able scaffolding
- The three CMP+BIT_AND emissions in `parent_at` (handle>0,
  slot>=0, slot<3) could in principle be factored into a helper,
  but each composes differently with the SELECT and there's only
  one call site. Below the >=80 threshold.
- Tests do duplicate the `parse + typecheck + lower + compile + run`
  scaffold across the 11 new Inc 15 + 2 new clean-gate-1 tests, but
  this is the established Stage 36 fixture style (matches the 120+
  pre-Inc-15 tests in the same file). No regression in conventions.

### Adherence to project conventions
- Naming: `evidence_*` (alias family), `parent_*_at` (raw arena
  reads), `register_derivation*` — all match the Stage 36 family
  conventions established in Inc 5/9/13/14.
- Test naming: `test_stage36_inc15_*` and
  `test_stage36_clean_gate1_*` match the increment-tagged pattern
  established in Inc 1-14.
- Comment style: every new code block carries an "Inc 15 ..." or
  "clean-gate-1 ..." prefix with the audit class (silent-failure
  H1/M1, type-design M1, code-review L1/L2/L3) and links to the
  audit doc - consistent with the Stage 28/35/36 convention.

### Deferred-TODO marker verification
- `TODO(stage36-inc16-arity-in-handle)` is present in:
  - `helixc/ir/lower_ast.py:2178` (parent_at lowering) [VERIFIED]
  - `helixc/frontend/typecheck.py:3004` (parent_at typecheck)
    [VERIFIED]
  - `helixc/stdlib/provenance.hx:26, 81` (doc comments) [VERIFIED]
  - `helixc/tests/test_stage36_provenance.py:2541` (doc) [VERIFIED]
- `TODO(stage36-inc16-arena-cursor-set)` is present in
  `helixc/tests/test_stage36_provenance.py:2292` (in the
  `test_stage36_inc14_register_derivation3_arena_overflow_returns_zero_handle`
  docstring marking the deferred direct-overflow exercise)
  [VERIFIED]
- The task brief mentions `TODO(stage37-arity-in-handle)`; the actual
  markers use the `stage36-inc16-arity-in-handle` spelling
  (consistent within the codebase, just a different bookkeeping
  label than the brief assumed). Not a code finding.

### Dead code / commented-out experiments / unused imports
- No commented-out code in the diff. All multi-line comments are
  documentation, not disabled code.
- No new top-level imports in `lower_ast.py` or `typecheck.py`; the
  new code uses already-imported `tir`, `A.IntLit`, `A.Unary`, etc.
- `test_stage36_inc15_register_derivation3_ad_erasure_fails_closed_reverse`
  uses local-scope `import pytest` (line 2511) - pytest is not at
  module top-level in this file (only inside `if __name__ ==
  "__main__"`), so the local import is correct, not dead.
- `provenance.hx` adds three new public fns + three doc-comment
  rewrites; no orphaned helpers.
- `dogfood_09_knowledge_graph.hx` L1 fix (line 33) - the corrected
  comment now matches the existing line-22-23 and line-65 invariants
  (`h_ad == 3`). Consistent.

### Cross-checks against gate-1 + gate-2 fixes
- A1 HIGH (parent_at dynamic upper bound): runtime CMP_LT against
  literal 3 + BIT_AND into guards_pass. Tested by
  `test_stage36_clean_gate1_parent_at_dynamic_slot_three_returns_sentinel`
  and `_huge_returns_sentinel` (gate-2 additions).
- B1 LOW (fuzzy_not return-None symmetry): `if a is None: return None`
  in `lower_ast.py:2287-2292`. Functionally identical to pre-fix
  (`return a` where a is None == `return None`); pure grep-symmetry
  cleanup. No behavior change risk.
- L1 LOW (dogfood_09 stale comment): line 33 now reads `h_ad == 3`
  with the slot[1] disambiguation clause. Matches the program's
  actual invariant.

## Summary

Total findings by severity: **0 critical, 0 important, 0 low**.

Audit reviewed the 6-file cumulative diff (539+ / 52- lines across
typecheck.py, lower_ast.py, provenance.hx, parser.py,
dogfood_09_knowledge_graph.hx, and test_stage36_provenance.py) for
Stage 36 Inc 15 + closure gate-1 fix sweep + gate-2 test additions.
The new `parent_at` typecheck + runtime guards are correctly composed
(CMP/BIT_AND produce 0/1, SELECT tests nonzero), the new
three-parent provenance helpers each have positive runtime test
coverage on a 3-parent handle, deferred-work TODO markers
(`stage36-inc16-arity-in-handle`, `stage36-inc16-arena-cursor-set`)
are present at the expected sites, error-message formats are pinned
by test assertions, naming and comment conventions match the Stage 28
/ 35 / 36 family, and no dead code or unused imports were introduced.
Architectural backlog items (per-handle arity word, arena cursor-set
test helper, TyDerivationHandle wrapper) are documented deferrals,
not findings.

Clean-gate counter: **2/3 -> 3/3. Stage 36 Inc 15 CLOSURE-READY.**
