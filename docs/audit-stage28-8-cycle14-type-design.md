# Stage 28.8 Pre-29 Audit Gate — Cycle 14, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit**: 1e4c3e6 (read-only)
**Cycle-10 baseline**: c2e36d4 (last commit that touched any code
or test under `helixc/` prior to the cycle-14 fix-sweep).
**Scope**: Audit the type-system / dispatch / soundness surface
after the cycle-14 fix-sweep landed in 1e4c3e6, plus a fresh-eyes
re-verification of the four previously named type-system contract
surfaces (`helixc/frontend/typecheck.py`,
`helixc/frontend/monomorphize.py`, `helixc/frontend/struct_mono.py`,
`helixc/frontend/autodiff.py`, `helixc/check.py`).

**Counter context** (per user directive 2026-05-10):

- Cycle 13's code-review audit found C13-1 HIGH (DCE drops the sole
  operand of TRACE_EXIT for unit-returning `@trace` fns, causing an
  `-O2` codegen `KeyError`).
- That HIGH **reset** the 5-clean-consecutive-cycles counter from
  3/5 → 0/5 (cycle 13 was not clean overall, even though cycle 13's
  type-design audit itself was CLEAN).
- Cycle 14 fix-sweep landed in 1e4c3e6: added `TRACE_ENTRY` and
  `TRACE_EXIT` to `SIDE_EFFECT_KINDS` in `helixc/ir/passes/dce.py`
  + 2 regression tests in `helixc/tests/test_dce.py`. Total delta
  from prior HEAD (98834de): 2 files, +48 lines.
- Cycle 14 is the **first** of the new streak. This audit must be
  CLEAN to advance the counter to 1/5.

---

## Cycle-14 production-code delta (since cycle-10 baseline c2e36d4)

```
helixc/ir/passes/dce.py  | 13 +++++++++++++
helixc/tests/test_dce.py | 35 +++++++++++++++++++++++++++++++++++
2 files changed, 48 insertions(+)
```

The full `helixc/frontend/` subtree, `helixc/check.py`, and the
remainder of the `helixc/ir/` subtree (parser, lower_ast, const_fold,
monomorphize, struct_mono, autodiff, tir.py, etc.) are **byte-identical
to the cycle-10 baseline c2e36d4**. Verified by:

```
git diff c2e36d4..HEAD -- helixc/frontend/   (empty)
git diff c2e36d4..HEAD -- helixc/check.py    (empty)
git diff c2e36d4..HEAD -- helixc/ir/tir.py   (empty)
git diff c2e36d4..HEAD -- helixc/ir/lower_ast.py  (empty)
git diff c2e36d4..HEAD -- helixc/ir/passes/const_fold.py  (empty)
```

The only production-code surface touched in cycle 14 is
`helixc/ir/passes/dce.py` (`SIDE_EFFECT_KINDS` set). This was not
previously a type-design contract surface in any cycle-1-through-13
type-design audit (verified by grep across the 13 prior type-design
docs — none mention `dce` or `SIDE_EFFECT`). It enters the type-design
scope this cycle solely because the cycle-14 fix-sweep touched it.

---

## Cycle 13 finding re-verification

| ID | Severity prev | Audit (prev) | Status now | Notes |
|---|---|---|---|---|
| — | n/a | type-design (cycle 13) | n/a (was CLEAN) | Cycle 13's type-design audit was CLEAN; no type-design finding to re-verify. C13-1 was a code-review finding, not a type-design finding, so it does not re-enter under this audit category. |

No prior-cycle type-design findings need re-verification under the
type-design audit category.

---

## Per-surface review (cycle-14 touchpoints)

### Surface T1: `SIDE_EFFECT_KINDS` set in `helixc/ir/passes/dce.py`
**Location**: `helixc/ir/passes/dce.py:32-81`.

#### What changed

The set literal at module scope gained two new members at the end:

```python
SIDE_EFFECT_KINDS = {
    tir.OpKind.RETURN,
    tir.OpKind.BR,
    ...
    tir.OpKind.TRAP,
    # Audit 28.8 cycle 13 C13-1 (HIGH): TRACE_ENTRY / TRACE_EXIT are
    # @trace prologue/epilogue ops with side effects (the runtime
    # records entry/exit events). ...
    tir.OpKind.TRACE_ENTRY,
    tir.OpKind.TRACE_EXIT,
}
```

#### Type-design contract

`SIDE_EFFECT_KINDS` is a module-level `frozenset`-shaped set literal,
inferred by Python as `set[tir.OpKind]`. Its semantic contract is:

> An `OpKind` is in this set iff a `tir.Op` with that kind has an
> observable runtime side effect that must NOT be eliminated even
> when its result value-ids are all dead.

Membership has two operational consequences inside `dce_function`
(dce.py:92-143):

1. **Seed phase** (lines 101-105): for any op whose kind is in the
   set, all of its operand value-ids are marked live. This roots the
   liveness analysis on side-effecting ops.
2. **Drop phase** (lines 128-142): ops whose kind is in the set are
   unconditionally retained (`new_ops.append(op); continue`), even
   if they have no results or all results are dead.

There is also a "no-results keep" arm (lines 134-136) that retains
any op with `not op.results` regardless of side-effect status — that
arm exists for ops like `BR` / `COND_BR` / `STORE_VAR` that genuinely
have no result slot. Crucially, that arm does NOT seed operands as
live (the seed phase is gated by `SIDE_EFFECT_KINDS` membership
only). This asymmetry is precisely the bug C13-1 exploited: a
no-results op outside the set is kept by the drop phase but its
operands are not rooted, so its operand producers get DCE'd, leaving
a dangling operand. The cycle-14 fix closes the gap by moving
`TRACE_ENTRY` / `TRACE_EXIT` into the set, which both roots their
operands AND retains them, restoring the invariant.

#### Type-soundness of the added members

- `tir.OpKind` is a `str`-valued `Enum` declared at `helixc/ir/tir.py:125`.
- `tir.OpKind.TRACE_ENTRY` is declared at `helixc/ir/tir.py:301`
  (`= "trace.entry"`).
- `tir.OpKind.TRACE_EXIT` is declared at `helixc/ir/tir.py:302`
  (`= "trace.exit"`).
- Both are valid enum members, both type-compatible with the set's
  inferred element type `tir.OpKind`. No `Literal`-narrowing,
  `Final`, or `Protocol` annotation gates the set.
- The set is a module-level mutable `set` (not `frozenset`). This is
  consistent with every other entry in the set, including the
  pre-existing Stage-28.5 `TRAP` addition. No type-design regression
  here; the open-set mutability has been the codebase convention
  since the FFI / TILE_INDEX_STORE additions.

#### Invariants strengthened, not weakened

Adding members to a "kinds that must be preserved" set is a
**monotone tightening** of the DCE pass's correctness envelope: every
op kind that was previously preserved is still preserved, and two new
kinds gain explicit preservation. This cannot introduce a soundness
regression in the DCE pass itself. It could in principle introduce a
performance regression (less code eliminated), but only for the two
specific kinds, which appear at most twice per `@trace` fn (once
entry, once exit) — negligible.

#### Re-verification against the cycle-13 audit

Cycle 13's code-review audit (the audit that found C13-1) prescribed
exactly this fix: add `TRACE_ENTRY` / `TRACE_EXIT` to
`SIDE_EFFECT_KINDS`. The fix landed as prescribed. The accompanying
type-design contract is intact:

- `SIDE_EFFECT_KINDS: set[tir.OpKind]` — preserved (no annotation
  change; element-type inference unchanged).
- "Ops in the set are seeded as live roots and unconditionally kept"
  — preserved (no change to `dce_function`'s seed-or-drop logic).
- "Ops outside the set with empty `op.results` are kept but their
  operands are NOT seeded as live" — preserved (the no-results keep
  arm at line 134-136 is unchanged). This is the asymmetry that
  motivated the fix; the fix avoids the asymmetry by moving
  `TRACE_EXIT` into the seed-rooting set rather than reworking the
  no-results arm. That's the minimal, type-design-safe choice.

#### Type-design rating

| Dimension | Rating | Justification |
|---|---|---|
| Encapsulation | 9/10 | Module-level set, single source of truth, no leaky alternate paths in `dce.py`. The set is read-only at runtime (only written at module load); no caller mutates it. Minor: it is a `set`, not a `frozenset`, so external code *could* mutate it — but no callee does. Consistent with codebase convention. |
| Invariant expression | 9/10 | Each member has a comment explaining why it's in the set; the cycle-14 additions follow that convention with a multi-line comment citing C13-1. The semantic invariant ("kinds whose ops have side effects and whose operands must be seeded live") is communicated through naming + comments. Compile-time enforcement (e.g., `Annotated[set[OpKind], "no-DCE roots"]` or a `@side_effect` decorator on each enum member) would be nicer but is not the codebase convention. |
| Invariant usefulness | 10/10 | The invariant directly prevents real bugs (C13-1 is one such bug; the Stage-16.5 FFI_CALL addition was another; Stage-28.5 TRAP is another). The set is the single authoritative answer to "what must DCE never delete," and every member's presence has been justified by a real defect history. |
| Invariant enforcement | 9/10 | The set is consulted at exactly two points (seed phase + drop phase), both inside `dce_function`. Both points read the same set — no risk of skew. The two new members are exercised by 2 new regression tests in `test_dce.py` (`test_c13_1_dce_preserves_trace_exit_operand`, `test_c13_1_dce_preserves_trace_entry_in_kept_set`). The remaining 1/10 gap: there is no static cross-check that *every* `OpKind` with side-effect semantics is in the set (would require encoding the side-effect bit on `OpKind` itself). The existing convention is "audit-driven discovery" — a kind enters the set when an audit finds a bug. That's pragmatic but not bulletproof. Not a cycle-14 finding because it's a long-standing design choice, not a cycle-14 regression. |

**Finding count from Surface T1**: 0 HIGH, 0 MEDIUM, 0 LOW.

---

### Surface T2: `helixc/tests/test_dce.py` — 2 new regression tests
**Location**: `helixc/tests/test_dce.py:112-145`.

The two added tests follow the existing file convention (use
`lower_fold_dce(src)` helper, gather ops by kind, assert on
post-pass module state). They exercise the cycle-14 fix:

- `test_c13_1_dce_preserves_trace_exit_operand`: asserts that for a
  unit-returning `@trace` fn, the surviving `TRACE_EXIT` op's
  operands are all in the live-id set. Pre-fix: would have failed
  (operand dropped). Post-fix: passes.
- `test_c13_1_dce_preserves_trace_entry_in_kept_set`: asserts that
  exactly 1 `TRACE_ENTRY` op survives DCE. Belt-and-suspenders for
  the second new set member.

Both tests use `tir.OpKind.TRACE_EXIT` / `tir.OpKind.TRACE_ENTRY`
enum lookups, which are now valid (verified above against
`tir.py:301-302`). No type-design surface introduced.

**Finding count from Surface T2**: 0 HIGH, 0 MEDIUM, 0 LOW.

---

## Per-surface review (carried surfaces, not touched in cycle 14)

The five originally-scoped type-system contract surfaces are
byte-identical to cycle 13 (which was byte-identical to cycle 10).
Re-verified by direct read of each surface against HEAD.

### Surface 1: `_compatible` TyMemTier strict-separation
**Location**: `helixc/frontend/typecheck.py:2248-2276`.

**Status**: byte-identical to cycle-10 baseline. The cycle-8 C7-1
carve-out drop is preserved (no `TyMemTier × (TyVar | TySize)`
carve-out). Tier subsumption remains a deferred enhancement (cycle-5
F4 / MEDIUM), not a current finding.

### Surface 2: `_size_compatible` shape-position cascade
**Location**: `helixc/frontend/typecheck.py:2232-2246`.

**Status**: byte-identical. Shape-position-only cascade boundary
preserved (cycle-7 C6-1). No regression pathway from cycle 14 (which
touches DCE, not typecheck).

### Surface 3: `_check_call_basic` symmetric filter
**Location**: `helixc/frontend/typecheck.py:687-757` (filter at
lines 746-752).

**Status**: byte-identical. Cycle-5 C4-3 symmetric `aty` filter in
place. The generic-adapter pattern continues to compile clean.

### Surface 4: `Monomorphizer.run` iteration order
**Location**: `helixc/frontend/monomorphize.py:433-492`.

**Status**: byte-identical. Both cycle-5 C4-4 key fixes preserved
(generic fns not walked at top level; clones promoted into walk set
each pass).

### Surface 5: `check.py` env-error helper + outer dispatch
**Location**: `helixc/check.py` (`_emit_env_error` + `main()`).

**Status**: byte-identical. The cycle-9 contributor-style implicit
contract (callees MAY include a single `helixc:` prefix; MUST NOT
nest) is intact. The cycle-10 forward notes (empty-string,
nested-prefix, leading-whitespace edge cases) remain non-blocking
forward notes, not findings.

---

## Cross-surface invariant snapshot (post-cycle-14)

No invariant is weakened by cycle 14. One invariant is **explicitly
strengthened**: the DCE pass now correctly preserves both
`TRACE_ENTRY` / `TRACE_EXIT` and their operand producers, closing the
C13-1 bug. The cycle-13 invariant snapshot extends to include:

**`SIDE_EFFECT_KINDS`** (dce.py:32-81):
- Contains all op kinds whose ops have observable side effects:
  RETURN, BR, COND_BR, CALL, STORE_VAR, STORE_ELEM, ALLOC_VAR,
  ALLOC_ARRAY, MODIFY, SPLICE, PRINT, QUOTE, REFLECT_HASH,
  ARENA_PUSH, ARENA_SET, TILE_INDEX_STORE, FFI_CALL, TRAP,
  **TRACE_ENTRY** (cycle-14), **TRACE_EXIT** (cycle-14).
- Membership is the seed-rooting AND drop-phase keep condition for
  the DCE pass. Adding a member is a monotone tightening.
- Each member is annotated with a `# Stage N — ...` or `# Audit ...`
  comment explaining the side-effect rationale.

The other type-system invariants (4 surfaces, cycle-13 snapshot) are
preserved by the empty diff over `helixc/frontend/` and
`helixc/check.py`.

---

## Cycle 14 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity at confidence ≥ 80.**

This cycle finds **0 findings (0 HIGH, 0 MEDIUM, 0 LOW)** under the
type-design audit category.

The cycle-14 fix-sweep (1e4c3e6) touches a previously-untouched
type-design surface (`SIDE_EFFECT_KINDS` in `dce.py`). The addition
is type-sound: both new members (`tir.OpKind.TRACE_ENTRY`,
`tir.OpKind.TRACE_EXIT`) are valid `OpKind` enum values declared at
`tir.py:301-302`, type-compatible with the set's inferred element
type. The set's semantic invariant (kinds that root DCE liveness and
are unconditionally kept) is monotonically tightened, not weakened.
The accompanying regression tests use the same enum members and
exercise both the seed-phase root behavior (operand preservation)
and the drop-phase keep behavior (op preservation).

The five originally-scoped type-system contract surfaces are
byte-identical to the cycle-10 baseline; their cycle-13 invariant
snapshot is preserved by construction.

By the strict criterion, **cycle 14 counts CLEAN** for the
type-design audit category.

**Counter status (5-clean-consecutive gate under the strict
criterion)**: was **0/5** after the cycle-13 C13-1 reset. With cycle
14's type-design audit CLEAN — and conditional on cycle-14's
silent-failures and code-review audits also being CLEAN — the
counter advances to **1/5**. Four more clean cycles (15, 16, 17, 18)
are required before Stage 29 can proceed.

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
- Cycle 14 (type-design only): 0 + 0 + 0 — CLEAN (this doc) →
  conditional contribution to counter 1/5

**Recommendation**: no fix-sweep needed for cycle 14's type-design
findings (there are none). The cycle-14 fix-sweep that landed in
1e4c3e6 is the cycle-13-derived fix and is verified type-sound by
this audit.

---

## Forward notes (not cycle-14 findings)

Carried forward unchanged from cycle 13 (themselves carried from
cycle 12 / 11 / 10). None are blocking.

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

5. **Cycle-15 baseline confirmation**: if cycle 15 is docs-only, the
   counter advances on stability alone. If a non-trivial production
   change lands between cycles 14 and 15, the next audit should give
   the diff a full read rather than relying on the empty-diff
   shortcut. Process note for future audit runs.

6. **`SIDE_EFFECT_KINDS` static cross-check** (NEW, cycle 14):
   there is no static guarantee that every `OpKind` with side-effect
   semantics is in `SIDE_EFFECT_KINDS` — membership is audit-driven.
   Stage-29-class hardening could move the side-effect bit onto the
   `OpKind` enum itself (e.g., a dataclass per kind with a
   `side_effect: bool` field, or a parallel `dataclass`
   `OpKindInfo` table) so that any new op kind must declare its
   side-effect status at definition time. Not a cycle-14 finding —
   it's a long-standing convention. Recorded here so future audits
   can decide whether the Stage-29 rewrite (Helix-native helixc)
   should adopt the stronger pattern.

7. **Stage-29 readiness**: with the counter reset to 0/5 by the
   cycle-13 C13-1 fix-sweep, Stage 29 is gated by four more
   consecutive clean cycles (15, 16, 17, 18) after cycle 14's
   clean. Cycle 14's type-design audit being CLEAN is one of three
   audit categories (silent-failures, type-design, code-review) that
   must all be clean for the cycle to count toward the streak.
