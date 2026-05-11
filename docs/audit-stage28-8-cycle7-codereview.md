# Stage 28.8 Pre-29 Audit Gate — Cycle 7, Audit C: Code Review

**Date**: 2026-05-11
**Commit**: b8e047e (read-only)
**Scope**: Audit C (general code-review) of the cycle-7 fix-sweep at
commit b8e047e, which lands the cycle-6 audit closures (C6-1 MEDIUM
from silent-failures, G1/G2 LOW from type-design) on top of the
cycle-6 baseline. Specifically reviewed:

- `helixc/frontend/typecheck.py` — three independent edits:
  1. `_size_compatible` helper added (C6-1 close): shape-position-
     only TyVar/TySize cascade extracted from cycle-6 F1's over-broad
     top-level `_compatible` cascade.
  2. `_compatible` rewritten to drop the top-level TyVar/TySize
     cascade (C6-1), with an explicit `TyMemTier` × `TyVar/TySize`
     carve-out (G2).
  3. `_handle_binary`'s D-domain mixed-inner warn extra-text branch
     (G1): distinguishes "D-wrapped vs Logic-wrapped" from
     "D-wrapped vs bare".
- `helixc/tests/test_typecheck.py` — two regression tests:
  `test_c5_2_size_compatible_tysize_cascade` (renamed from cycle-6's
  `test_c5_2_compatible_tysize_cascade`) now exercises the helper
  directly, and `test_c6_1_compatible_tyvar_not_top_cascade` asserts
  top-level `_compatible(TyVar, TyPrim)` does NOT cascade.

**Method**: Read the full b8e047e diff in `git show`. Walked
typecheck.py at HEAD covering lines 1280–1430 (binop domain handling)
and 2160–2300 (the rewritten `_size_compatible` / `_compatible`
pair). Cross-walked every existing `_compatible(...)` call site (lines
742, 1142, 1164, 1450, 1560, 1586, 1659, plus the recursive arms in
`_compatible` itself) to verify the narrowed cascade is correct at
each. Re-verified the cycle-6 audit's C6-1 reproducer
(`fn g[T]() -> T { 42 }`) is now rejected. Ran the two new regression
tests (PASSED).

**Reporting threshold**: confidence ≥ 80 (per the cycle-7 audit-C
prompt's strict criterion).

**Result**: **0 findings (0 CRITICAL, 0 HIGH, 0 MEDIUM, 0 LOW) at or
above the confidence-80 reporting threshold.**

---

## Summary table

| ID    | Severity | Confidence | Component | Issue |
|-------|----------|------------|-----------|-------|
| —     | —        | —          | —         | (none at or above threshold) |

**Cycle 7 Audit C: CLEAN — 0 findings at the confidence-80 threshold.**

Per user directive 2026-05-10 (strict criterion): cycle counts CLEAN
only when zero findings of ANY severity at or above the audit
threshold. **This cycle qualifies as clean.**

---

## Cycle-6 finding closure verification

### C6-1 (MEDIUM, conf 88 in cycle-6 silent-failures): Over-broad TyVar/TySize cascade in top-level `_compatible` — **CLOSED**

Cycle 6 F1 introduced a top-level cascade-safe arm
`isinstance(a, (TyVar, TySize)) or isinstance(b, (TyVar, TySize)) →
True` at the very top of `_compatible`. Cycle-6 silent-failures audit
found this fired at body-vs-return, let-binding, if/else merge, and
match-arm-merge sites, silently typechecking `fn g[T]() -> T { 42 }`.

**Cycle 7 fix** (typecheck.py:2176-2190): extracted the cascade-safe
arm into a new `_size_compatible(a, b)` helper that's called only at
the three shape-position sites:

- `TyArray` size compare (line 2251).
- `TyTensor` shape-element compare (line 2283).
- `TyTile` shape-element compare (line 2293).

The top-level `_compatible` (line 2192-2298) no longer has the TyVar/
TySize cascade arm; control falls through to the existing structural
arms and finally to `a == b`. Walked each non-shape call site:

- Line 742 (call boundary): pre-gated on `not isinstance(pty,
  (TyVar, TySize, TyUnknown))`, so the narrowing is a no-op here.
- Line 1142 (body-vs-return): now rejects `body=TyPrim('i32'), ret=
  TyVar('T')` correctly. **Reproducer for C6-1 fixed.**
- Line 1164 (let-stmt declared type vs value): same — narrowing
  surfaces the error.
- Line 1450 (call-arg vs expected, downstream of the line-742 gate):
  reaches here only when both sides are non-TyVar; no behavior change.
- Lines 1560, 1586 (if/else merge, match-arm merge): now `TyVar('T')
  vs TyPrim('i32')` rejects, which matches the user intent for
  exhaustive arm typing.
- Line 1659 (let-stmt expected-vs-actual): same as 1164. **PASS.**

Test coverage: `test_c6_1_compatible_tyvar_not_top_cascade`
(test_typecheck.py:1469-1478) directly asserts `_compatible(TyVar,
TyPrim)` returns False. **CLOSED.**

### G1 (LOW, conf ~70 in cycle-6 type-design): D-domain mixed-inner warn says "bare" when other side is Logic-wrapped — **CLOSED**

Cycle-6 type-design audit observed that the F2 fix at typecheck.py:1349
made `D<f64> + f64` warn with extra-text "(one side D-wrapped, other
bare)", but the same extra-text fired when the asymmetric pair was
`D<f64> + Logic<f64>` — the right side is Logic-wrapped, not bare.

**Cycle 7 fix** (typecheck.py:1365-1378): the extra-text branch now
computes `other_is_logic = (l_is_diff and r_is_logic) or (r_is_diff
and l_is_logic)` and selects `" (one side D-wrapped, other Logic-
wrapped)"` vs `" (one side D-wrapped, other bare)"` accordingly.

Cross-checked the definitions: `l_is_diff = isinstance(l, TyDiff)`
(line 1280); `l_is_logic = isinstance(l, TyLogic) or (isinstance(l,
TyDiff) and isinstance(l.inner, TyLogic))` (lines 1293-1298). When
the asymmetric branch fires (gate at line 1349: `(l_is_diff or
r_is_diff) and not (l_is_diff and r_is_diff)` via the inner
`if not (l_is_diff and r_is_diff)` at line 1365), exactly one side
carries TyDiff. If that side is `TyDiff(TyLogic(T))`, then
`l_is_diff=True` and `l_is_logic=True`, and the other side is bare
T or Logic<T>. The `other_is_logic` test correctly identifies whether
the *non-D-wrapped* side is Logic.

Minor sub-threshold note: when `l = TyDiff(TyLogic(T))` and
`r = TyLogic(T)`, both sides are "logic-wrapped" but only one is
D-wrapped; `other_is_logic` evaluates True (because `l_is_diff=True`
and `r_is_logic=True`), yielding the correct "(D-wrapped, other
Logic-wrapped)" text. Behavior matches user-facing intent.
**CLOSED.**

### G2 (LOW, conf ~70 in cycle-6 type-design): TyMemTier × TyVar cross-tier reject — **CLOSED**

Cycle-6 type-design audit observed that with the cycle-6 F1 cascade
removed (per C6-1), `_compatible(WorkingMem<i32>, T)` would fall
through the TyMemTier-or-TyMemTier arm and return False — i.e. the
narrowing of C6-1 would have reintroduced a different false-positive
for generic functions over memory tiers.

**Cycle 7 fix** (typecheck.py:2208-2211): two new arms added:
```python
if isinstance(a, TyMemTier) and isinstance(b, (TyVar, TySize)):
    return True
if isinstance(b, TyMemTier) and isinstance(a, (TyVar, TySize)):
    return True
```
These fire BEFORE the cross-tier reject arm
(`if isinstance(a, TyMemTier) or isinstance(b, TyMemTier): return
False` at line 2212-2213). The carve-out is correctly placed
(positioned between the both-MemTier arm at 2206 and the
one-MemTier-other-rejected arm at 2212). **CLOSED.**

---

## Files reviewed

`helixc/frontend/typecheck.py` (lines 1280-1430 and 2160-2300),
`helixc/tests/test_typecheck.py` (lines 1436-1478), plus the
persisted cycle-6 audit-doc files (`audit-stage28-8-cycle6-
{codereview,silent-failures,type-design}.md`) for cross-reference.

---

## Specific cycle-7 changes audited (5 items)

1. **`_size_compatible` new helper at line 2176-2190** — shape-
   position-only cascade arm. Calls `_compatible` at the tail, so
   composite types nested inside shape positions (unusual but
   permitted, e.g. `TyArray<TyArray<i32; N>; M>` if anyone ever
   constructed one) recurse correctly. The TyVar/TySize check fires
   first so a pair like `(TySize('N'), TySize('M'))` returns True
   without entering `_compatible`. The redundant TyUnknown +
   `a == b` checks before falling through to `_compatible` are
   harmless (fast path); `_compatible` would re-test them anyway.
   **PASS.**

2. **`_compatible` top-level cascade removed at line 2192** — the
   former 4-line `isinstance(a, (TyVar, TySize)) ...` arm is gone.
   The comment block (lines 2195-2205) correctly documents the
   narrowing decision and references the cycle-5 audit's option (b).
   **PASS.**

3. **`_compatible` TyMemTier × TyVar/TySize carve-out at lines
   2208-2211** — two arms inserted between the both-MemTier arm and
   the one-MemTier reject arm. Order is correct
   (specific-before-general). The asymmetric direction (a-vs-b vs
   b-vs-a) is handled by the two separate arms. **PASS.**

4. **TyTensor / TyTile shape compares switched from `_compatible` to
   `_size_compatible`** at lines 2283 and 2293. Walked the
   comprehension: `all(self._size_compatible(x, y) for x, y in
   zip(a.shape, b.shape))`. Shape elements are `Type` (TySize for
   generic-size, TyUnknown for inferred, TyPrim("size_N") for
   concrete) — all three cases now defer correctly when a generic
   size meets a concrete size. **PASS.**

5. **D-mixed warn extra-text branch at lines 1365-1378** — the
   conditional now selects between two strings based on
   `other_is_logic`. The branch is reached only inside
   `if not (l_is_diff and r_is_diff)` (line 1365) — i.e. exactly
   one side carries TyDiff. The `other_is_logic` predicate correctly
   identifies which side is non-D and asks whether it carries Logic.
   Edge case `TyDiff(TyLogic(T)) + Logic(T)`: `l_is_diff=True,
   r_is_diff=False, r_is_logic=True` → branch picks "D-wrapped,
   Logic-wrapped" — correct (the L side carries D, the R side
   carries Logic but not D). **PASS.**

---

## What was checked and found below threshold

- **`_size_compatible` redundant `a == b` check on line 2188**:
  pre-tail it tests `a == b → True`, but the tail calls
  `_compatible(a, b)` which falls through eventually to its own
  `return a == b` at line 2298. The pre-tail check is a fast-path
  optimization, not a correctness concern. **Confidence 35**, below
  threshold.

- **`_size_compatible` falls back to `_compatible` even though
  TyArray/TyTensor/TyTile shape positions only ever hold TySize /
  TyUnknown / TyPrim** (per the grammar): the `return
  self._compatible(a, b)` tail at line 2190 covers a theoretical
  future where shape positions hold richer types (e.g. computed
  sizes). Not a bug; small over-generalization. **Confidence 40**,
  below threshold.

- **G1 extra-text branch doesn't distinguish "D<Logic<T>> +
  Logic<T>" from "D<T> + Logic<T>"**: in the first case the L side
  is *also* logic-wrapped (just under a D), and the user may want
  to know the inner-Logic provenance was preserved through the
  widen. The current "(one side D-wrapped, other Logic-wrapped)"
  text doesn't disclose that the D side also carries Logic
  underneath. This is a diagnostic-elaboration concern, not a
  silent-failure path. **Confidence 55**, below threshold.

- **Tests for G1 / G2 not landed**: the commit message names tests
  for C6-1 (size-cascade and tyvar-not-top-cascade, 2 tests) but
  no tests directly exercise the new "Logic-wrapped" extra-text
  (G1) or the new `TyMemTier × TyVar` carve-out (G2). G1 is a
  diagnostic-string assertion (would need to capture
  `_DIFF_WARNINGS`); G2 is a positive
  `_compatible(TyMemTier(...), TyVar('T'))` assertion. Both are
  small; absence is below threshold given the test-density baseline
  established in cycle 6. **Confidence 60**, below threshold.

- **`_size_compatible` is called only on `.size` and `.shape[i]`
  positions** — no recursion into nested composites needed since
  shape elements are leaf types in practice. Recursion safety
  verified: a shape-position TyArray would route to `_compatible`
  via the tail, which uses `_size_compatible` for the inner
  TyArray's size — bounded by the AST depth. **Confidence 30**,
  below threshold.

- **`_compatible(TyVar('T'), TyVar('T'))` falls through to
  `a == b`**: returns True only if the dataclass equality holds
  (same `name` field). Two `TyVar('T')` instances at different
  call sites are equal under dataclass `__eq__`. This is the
  intended behavior (a generic param `T` is compatible with
  itself), and matches the pre-cycle-6 behavior. **Confidence 30**,
  below threshold.

- **Order of `_compatible` arms**: TyMemTier-MemTier (2206) → new
  carve-out arms (2208-2211) → cross-tier reject (2212-2213).
  Adding a third path-type pair (e.g. TyMemTier × TyUnknown) would
  need a new arm OR rely on the top-of-function TyUnknown check
  at 2193 (which fires first). Currently TyUnknown is handled.
  No latent gap. **Confidence 35**, below threshold.

- **Test rename `test_c5_2_compatible_tysize_cascade →
  test_c5_2_size_compatible_tysize_cascade`**: the renamed test now
  exercises both `_size_compatible` directly AND `_compatible` via
  the TyArray composite (which routes to `_size_compatible`
  internally). The rename matches the new helper's name. Good
  hygiene. **Confidence 25**, below threshold.

---

## Open prior findings (not re-flagged this cycle)

All three cycle-6 findings (C6-1 from silent-failures, G1 + G2 from
type-design) are closed per the analysis above.

Cycle-6 audit C was already CLEAN. No regression introduced by
cycle-7 fix-sweep:

- TyArray size compare still cascades (now via `_size_compatible`).
  Test `test_c5_2_size_compatible_tysize_cascade` verifies.
- TyMemTier × TyMemTier same-tier inner-compat still works
  (line 2207 unchanged).
- TyDiff / TyLogic asymmetric warn still fires per cycle-6 F2
  (gate at line 1349 unchanged); only the extra-text wording got
  the G1 distinction.
- Call-boundary line 742 pre-gate is intact; narrowing the
  top-level cascade has no effect there.

Pre-cycle-1 audit-stage5-6, audit-stage7-8, audit-stage9-16 baselines
unchanged from cycle-1 status; cycle-1 through cycle-6 findings all
marked CLOSED by their respective fix-sweep commits.

---

## Verdict

**Cycle 7 Audit C: CLEAN — 0 findings (0 CRITICAL, 0 HIGH, 0 MEDIUM,
0 LOW) at or above the confidence-80 reporting threshold.**

Strict-zero rule per user directive 2026-05-10. Cycle counter advances
provided cycles A (silent-failure) and B (type-design) are also clean
at this commit.

No recommended fixes for cycle 7 audit-C scope. The below-threshold
notes (redundant fast-path in `_size_compatible`, possible "D<Logic<
T>> + Logic<T>" diagnostic refinement, untested G1/G2 paths) are
documented for future cycles but do not block this cycle's clean
status.
