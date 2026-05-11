# Stage 28.8 Pre-29 Audit Gate — Cycle 15, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit**: 1e4c3e6 (read-only)
**Cycle-14 baseline**: 1e4c3e6 (same commit — no production-code
change has landed between cycle 14 and cycle 15).
**Scope**: Fresh-eyes re-audit of the type-system / dispatch /
soundness surface after the cycle-14 fix-sweep, confirming no new
defect since cycle 14's CLEAN type-design audit.

**Counter context** (per user directive 2026-05-10):

- Cycle 14 was FULLY CLEAN (all three audit categories: silent-failures,
  type-design, code-review). Counter advanced from 0/5 → 1/5.
- Wait — re-reading the user directive for cycle 15: "Cycle 14 FULLY
  CLEAN. Fresh counter: 1/5." Confirmed: counter is 1/5 entering
  cycle 15.
- This is cycle 15's type-design audit. If CLEAN under the strict
  criterion (zero findings of any severity at confidence ≥ 80), and
  conditional on cycle-15 silent-failures and code-review also being
  CLEAN, the counter advances to 2/5.

---

## Cycle-15 production-code delta

```
git diff 1e4c3e6..HEAD -- helixc/    (empty)
git rev-parse HEAD                    1e4c3e639a593995dc66c7c3369a0955b6ddbf83
```

HEAD is identical to the cycle-14 fix-sweep commit. No production-code
change has landed in cycle 15. The working tree carries only
unstaged changes under `docs/` (audit doc additions for cycle 14
and cycle 15, plus minor docs edits to cycle-11 audit files). No
unstaged change under `helixc/`.

Cross-check against the cycle-10 baseline c2e36d4 (last commit prior
to the cycle-14 fix-sweep that touched `helixc/`):

```
git diff c2e36d4..HEAD --stat -- helixc/
 helixc/ir/passes/dce.py  | 13 +++++++++++++
 helixc/tests/test_dce.py | 35 +++++++++++++++++++++++++++++++++++
 2 files changed, 48 insertions(+)
```

Same delta as audited in cycle 14. The full `helixc/frontend/`
subtree, `helixc/check.py`, and the remainder of the `helixc/ir/`
subtree (parser, lower_ast, const_fold, monomorphize, struct_mono,
autodiff, tir.py, etc.) are byte-identical to the cycle-10 baseline.

---

## Prior-cycle finding re-verification

| ID | Severity prev | Audit (prev) | Status now | Notes |
|---|---|---|---|---|
| — | n/a | type-design (cycle 14) | n/a (was CLEAN) | Cycle 14's type-design audit was CLEAN; no type-design finding to re-verify. C13-1 (HIGH, code-review) was closed by the cycle-14 fix-sweep that landed in 1e4c3e6 — already covered in cycle 14 doc, does not re-enter this cycle. |

No prior-cycle type-design findings need re-verification under the
type-design audit category. Per user directive, prior-cycle findings
ARE NOT re-flagged.

---

## Fresh-eyes re-audit (cycle-14 touchpoints, second pass)

The cycle-14 fix-sweep touched `helixc/ir/passes/dce.py:32-81`
(`SIDE_EFFECT_KINDS` set literal) and `helixc/tests/test_dce.py:112-145`
(two regression tests). Cycle 14's type-design audit examined both
surfaces and found zero issues. This cycle does an independent
second-pass read to look for anything cycle 14 might have missed.

### Surface T1 (re-read): `SIDE_EFFECT_KINDS` set in `dce.py`

Read against HEAD at `helixc/ir/passes/dce.py:32-81`:

```python
SIDE_EFFECT_KINDS = {
    tir.OpKind.RETURN,
    tir.OpKind.BR,
    ...
    tir.OpKind.TRAP,
    # Audit 28.8 cycle 13 C13-1 (HIGH): ...
    tir.OpKind.TRACE_ENTRY,
    tir.OpKind.TRACE_EXIT,
}
```

Independent fresh-eyes checks performed this cycle:

1. **Enum-member existence**: `tir.OpKind.TRACE_ENTRY` and
   `tir.OpKind.TRACE_EXIT` both exist in `helixc/ir/tir.py:301-302`
   (carried verification from cycle 14, re-read against HEAD —
   unchanged). Static type-check: both are valid `str`-Enum members.
2. **No-duplicate check**: walked the set top to bottom; each enum
   member appears at most once. Python sets de-duplicate at
   construction in any case, but a duplicate would be a stylistic
   smell. None present.
3. **No-typo check**: each comment-cited enum name matches a real
   `tir.OpKind` member declared at its claimed line. Spot-checked
   `TILE_INDEX_STORE`, `FFI_CALL`, `TRAP`, `QUOTE`, `REFLECT_HASH`,
   `ARENA_PUSH`, `ARENA_SET` against `tir.py:125-310`. All present.
4. **Side-effect rationale**: cross-checked each member against its
   semantic role. RETURN / BR / COND_BR are terminators (must be
   preserved for control flow). STORE_VAR / STORE_ELEM / MODIFY /
   SPLICE / ALLOC_VAR / ALLOC_ARRAY mutate state. PRINT / QUOTE /
   REFLECT_HASH / ARENA_PUSH / ARENA_SET / TILE_INDEX_STORE / FFI_CALL
   / TRAP / TRACE_ENTRY / TRACE_EXIT have observable runtime effects.
   CALL is conservative (any user fn may have effects). Coverage is
   sound — no member is in the set without justification.
5. **Soundness of asymmetric retention**: the seed phase (lines
   100-105) and drop phase (lines 127-142) both consult the same set.
   The fixpoint phase (lines 113-125) skips ops in the set (no need
   to spread liveness through them — their operands are already live
   from the seed phase). This is internally consistent. No skew
   pathway introduced by the cycle-14 additions.
6. **Type annotation**: the set has no explicit type annotation;
   Python infers `set[tir.OpKind]`. This is consistent with the rest
   of the module's conventions. Not a finding.
7. **Mutability**: declared as a `set` (not `frozenset`). External
   code *could* in principle mutate it. Grep'd the codebase for
   `SIDE_EFFECT_KINDS` writes — only the declaration site appears
   (one match). No mutation pathway. Consistent with codebase
   convention; cycle 14 documented this. Not a finding.

No new defect surfaced in the fresh-eyes pass.

### Surface T2 (re-read): 2 regression tests in `test_dce.py`

Read against HEAD at `helixc/tests/test_dce.py:112-145`. Tests use
the existing `lower_fold_dce(src)` helper, gather ops by kind, and
assert on post-pass module state. They exercise:

- `test_c13_1_dce_preserves_trace_exit_operand`: for a unit-returning
  `@trace` fn, asserts every operand of the surviving `TRACE_EXIT`
  op has its value-id in the live-id set.
- `test_c13_1_dce_preserves_trace_entry_in_kept_set`: asserts exactly
  1 `TRACE_ENTRY` op survives DCE.

Fresh-eyes check performed this cycle:

1. **Enum lookup validity**: `tir.OpKind.TRACE_EXIT` /
   `tir.OpKind.TRACE_ENTRY` enum lookups are valid (re-verified
   against `tir.py:301-302`).
2. **Failure-mode soundness**: tests would have failed pre-fix —
   confirmed by the cycle-14 fix-sweep commit message in 1e4c3e6.
3. **No new type-design surface**: the tests do not introduce any
   new type contract. They are pure black-box assertions against
   the DCE pass's behavior.

No defect introduced. No new type-design surface.

---

## Per-surface review (carried surfaces, byte-identical to cycle 14)

The five originally-scoped type-system contract surfaces remain
byte-identical to the cycle-10 baseline and to cycle 14:

### Surface 1: `_compatible` TyMemTier strict-separation
**Location**: `helixc/frontend/typecheck.py:2248-2276`.
**Status**: byte-identical to cycle-10 baseline. Cycle-8 C7-1
carve-out drop is preserved. Tier subsumption (cycle-5 F4 / MEDIUM)
remains a deferred enhancement, not a current finding.

### Surface 2: `_size_compatible` shape-position cascade
**Location**: `helixc/frontend/typecheck.py:2232-2246`.
**Status**: byte-identical. Cycle-7 C6-1 shape-position-only cascade
boundary preserved.

### Surface 3: `_check_call_basic` symmetric filter
**Location**: `helixc/frontend/typecheck.py:687-757`.
**Status**: byte-identical. Cycle-5 C4-3 symmetric `aty` filter in
place.

### Surface 4: `Monomorphizer.run` iteration order
**Location**: `helixc/frontend/monomorphize.py:433-492`.
**Status**: byte-identical. Both cycle-5 C4-4 key fixes preserved.

### Surface 5: `check.py` env-error helper + outer dispatch
**Location**: `helixc/check.py` (`_emit_env_error` + `main()`).
**Status**: byte-identical. Cycle-9 contributor-style implicit
contract intact.

---

## Cross-surface invariant snapshot (post-cycle-15)

No invariant change since cycle 14. The cycle-14 snapshot extends
unchanged. For completeness:

**`SIDE_EFFECT_KINDS`** (dce.py:32-81):
- Members: RETURN, BR, COND_BR, CALL, STORE_VAR, STORE_ELEM,
  ALLOC_VAR, ALLOC_ARRAY, MODIFY, SPLICE, PRINT, QUOTE, REFLECT_HASH,
  ARENA_PUSH, ARENA_SET, TILE_INDEX_STORE, FFI_CALL, TRAP,
  TRACE_ENTRY, TRACE_EXIT. (Same 20 members as cycle 14.)
- Membership is the seed-rooting AND drop-phase keep condition for
  the DCE pass. Adding a member is a monotone tightening.
- Each member is annotated with a Stage- or Audit-citing comment.

The other type-system invariants (4 surfaces) are preserved by the
empty diff over `helixc/frontend/` and `helixc/check.py` since
cycle 14.

---

## Cycle 15 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity at confidence ≥ 80.**

This cycle finds **0 findings (0 HIGH, 0 MEDIUM, 0 LOW)** under the
type-design audit category.

HEAD is byte-identical to the cycle-14 fix-sweep commit (1e4c3e6).
The fresh-eyes second-pass re-read of the two cycle-14 touchpoints
(SIDE_EFFECT_KINDS set, 2 regression tests) found no new defect.
The five originally-scoped type-system contract surfaces are
byte-identical to cycle 14, which in turn was byte-identical to the
cycle-10 baseline — their invariant snapshots are preserved by
construction.

By the strict criterion, **cycle 15 counts CLEAN** for the
type-design audit category.

**Counter status (5-clean-consecutive gate under the strict
criterion)**: was **1/5** after cycle 14's full clean. With cycle
15's type-design audit CLEAN — and conditional on cycle-15's
silent-failures and code-review audits also being CLEAN — the
counter advances to **2/5**. Three more clean cycles (16, 17, 18)
remain required before Stage 29 can proceed.

The severity trend across cycles, against the strict-criterion bar:
- Cycle 1: HIGH-tier finding(s) — not clean
- Cycle 2: HIGH + MEDIUM — not clean
- Cycle 3: HIGH + MEDIUM + LOW (multiple LOW) — not clean
- Cycle 4: MEDIUM-tier — not clean
- Cycle 5: 3 MEDIUM + 3 LOW — not clean
- Cycle 6: 1 MEDIUM + 2 LOW — not clean
- Cycle 7: 0 + 0 + 0 — pre-directive era; CLEAN under loose
  criterion only
- Cycle 8: 0 + 0 + 0 — same
- Cycle 9: 0 + 0 + 0 — same
- Cycle 10: 0 + 0 + 0 — first clean cycle under strict criterion
  per user directive 2026-05-10 → counter 1/5
- Cycle 11: 0 + 0 + 0 — CLEAN → counter 2/5
- Cycle 12: 0 + 0 + 0 — CLEAN → counter 3/5
- Cycle 13: 1 HIGH (C13-1 code-review) — NOT CLEAN → counter reset
  3/5 → 0/5
- Cycle 14: FULLY CLEAN (all three categories) → counter 0/5 → 1/5
- Cycle 15 (type-design only): 0 + 0 + 0 — CLEAN (this doc) →
  conditional contribution to counter 1/5 → 2/5

**Recommendation**: no fix-sweep needed for cycle 15's type-design
findings (there are none). The cycle-14 fix-sweep (1e4c3e6) remains
type-sound as verified.

---

## Forward notes (not cycle-15 findings)

Carried forward unchanged from cycle 14 (themselves carried from
prior cycles). None are blocking.

1. **Empty-string edge case for `_emit_env_error`**: no test asserts
   `_emit_env_error("")` produces `helixc: ` (and remains stable
   across refactors). No production callee passes empty. Not
   blocking.

2. **Nested-prefix edge case for `_emit_env_error`**: no test
   asserts `_emit_env_error("helixc: helixc: foo")` strips exactly
   one layer. No production callee produces nested prefixes. Not
   blocking.

3. **Whitespace-handling edge case for `_emit_env_error`**: no test
   asserts `_emit_env_error("   helixc: foo")` produces a
   single-prefix output. Not blocking.

4. **Convention note for raise-message prefix**: a contributor-style
   doc could codify the implicit cycle-9 contract (callees MAY
   include a single `helixc:` prefix; MUST NOT nest). Not blocking.

5. **`SIDE_EFFECT_KINDS` static cross-check**: there is no static
   guarantee that every `OpKind` with side-effect semantics is in
   `SIDE_EFFECT_KINDS` — membership is audit-driven. Stage-29-class
   hardening could move the side-effect bit onto the `OpKind` enum
   itself (e.g., a dataclass per kind with a `side_effect: bool`
   field, or a parallel `OpKindInfo` table) so that any new op kind
   must declare its side-effect status at definition time. Not a
   cycle-15 finding — long-standing convention. Recorded again here
   so the Stage-29 rewrite (Helix-native helixc) can decide whether
   to adopt the stronger pattern.

6. **Cycle-16 baseline confirmation**: if cycle 16 is docs-only, the
   counter advances on stability alone. If a non-trivial production
   change lands between cycles 15 and 16, the next audit should give
   the diff a full read rather than relying on the empty-diff
   shortcut. Process note for future audit runs.

7. **Stage-29 readiness**: with the counter at 1/5 after cycle 14
   (and provisionally 2/5 after cycle 15 conditional on the other
   two cycle-15 audits also being CLEAN), Stage 29 is gated by
   three more consecutive clean cycles (16, 17, 18). Cycle 15's
   type-design audit being CLEAN is one of three audit categories
   (silent-failures, type-design, code-review) that must all be
   clean for the cycle to count toward the streak.
