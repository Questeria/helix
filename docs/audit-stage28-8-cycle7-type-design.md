# Stage 28.8 Pre-29 Audit Gate — Cycle 7, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit**: b8e047e (read-only)
**Scope**: Re-audit type-system soundness focused on cycle-7's fix-sweep
(b8e047e — closes cycle-6 C6-1 MEDIUM + G1, G2 LOW). The cycle-7
fix-sweep touches:

- `helixc/frontend/typecheck.py:2176-2190` — new `_size_compatible`
  helper. Cascade-safe for TyVar / TySize, then TyUnknown, then
  identity, then delegate to `_compatible`. Replaces the cycle-6 F1
  top-level cascade arm at the old line 2178-2179.
- `helixc/frontend/typecheck.py:2192-2213` — `_compatible` no longer
  has a top-level TyVar / TySize cascade. The TyMemTier strict-
  separation block now has an explicit TyMemTier × (TyVar | TySize)
  carve-out at lines 2208-2211 (both orders), placed BEFORE the
  broad `if isinstance(a, TyMemTier) or isinstance(b, TyMemTier):
  return False` at line 2212-2213.
- `helixc/frontend/typecheck.py:2245-2251, 2280-2284, 2289-2295` —
  TyArray / TyTensor / TyTile size-element compares now route
  through `_size_compatible` rather than `_compatible`.
- `helixc/frontend/typecheck.py:1349-1381` — D-domain binop gate
  extra-text now distinguishes "(one side D-wrapped, other Logic-
  wrapped)" from "(one side D-wrapped, other bare)" via a new
  `other_is_logic` predicate (cycle-6 G1 close).
- `helixc/tests/test_typecheck.py:1436-1478` — renamed
  `test_c5_2_compatible_tysize_cascade` →
  `test_c5_2_size_compatible_tysize_cascade` and added
  `test_c6_1_compatible_tyvar_not_top_cascade` for the narrow-vs-
  broad distinction.

**Method**: read cycle-1 through cycle-6 type-design audit docs to
build the cumulative invariant set, then walked the cycle-7 diff
through each of the three contracts it touches.

For `_size_compatible`: walked the four arms in order. Verified the
helper is invoked from exactly three call sites (TyArray.size,
TyTensor.shape, TyTile.shape) and nowhere else. Probed whether a
caller could accidentally re-introduce the broad cascade by mis-
routing a value-position compare through `_size_compatible`.
Verified the call-boundary check at line 742 still skips top-level
TyVar/TySize at pty BEFORE invoking `_compatible`, so the pre-
cycle-6 generic-arg-at-value-position path is preserved.

For the `_compatible` top-level cascade removal: traced
`fn g[T]() -> T { 42 }` body-vs-return check at line 1142. Pre-
cycle-7 (cycle-6 with F1) the top-level cascade silently swallowed
this. Post-cycle-7 the cascade is gone; the check rejects with
"body type i32 does not match return type T". Reproducer matches
the cycle-7 commit message claim.

For the TyMemTier × TyVar/TySize carve-out: walked the matrix of
(TyMemTier(W, X), TyVar('T')) and reverse. Verified arm ordering:
both-MemTier arm fires first (tier-match check), then the two
asymmetric TyVar/TySize arms, then the broad-or arm. Probed the
nested case `_compatible(TyMemTier(W, TyVar('T')), TyMemTier(W,
i32))` — both-MemTier arm fires, recurses on inner `_compatible(
TyVar('T'), TyPrim('i32'))` which now falls through to `a == b` =
False (because the top-level cascade is gone). This nested case
matches PRE-cycle-6 behavior; it was not flagged in any prior cycle
and is consistent with the cycle-7 stated intent ("shape positions
only").

For the G1 D-arm extra-text: walked the full eight-row matrix from
the cycle-6 G1 finding through the new `other_is_logic` predicate.
Verified each previously-inaccurate row now produces the correct
text:
- `D<T> + Logic<T>`: l_is_diff=T, r_is_logic=T → other_is_logic=T
  → "(one side D-wrapped, other Logic-wrapped)" — correct.
- `Logic<T> + D<T>`: l_is_logic=T, r_is_diff=T → other_is_logic=T
  → "(one side D-wrapped, other Logic-wrapped)" — correct.
- `D<T> + bareT`: l_is_diff=T, r_is_logic=F, l_is_logic=F →
  other_is_logic=F → "(one side D-wrapped, other bare)" — correct.
- `D<Logic<T>> + bareT`: l_is_diff=T, l_is_logic=T (TyDiff inner is
  TyLogic), r_is_logic=F → other_is_logic = (l_is_diff and
  r_is_logic) or (r_is_diff and l_is_logic) = (T and F) or (F and T)
  = F → "(one side D-wrapped, other bare)" — correct (the OTHER
  side is bare; the D-wrapped side happens to be D-over-Logic but
  the diagnostic correctly names the contrast pair).
- `D<Logic<T>> + Logic<T>`: l_is_diff=T, l_is_logic=T, r_is_logic=T
  → other_is_logic = (T and T) or (F and T) = T → "(one side
  D-wrapped, other Logic-wrapped)" — correct.
- `D<T> + D<T>` and `Logic<T> + Logic<T>`: gate doesn't enter the
  extra-text branch (the `if not (l_is_diff and r_is_diff)` guard
  is False, extra stays ""). Correct.

**Result**: **0 findings (0 HIGH, 0 MEDIUM, 0 LOW)**. Cycle 7
addresses all three cycle-6 findings cleanly at the contract level
and does not open new contract surface that weakens any prior
invariant. The strict criterion ("zero findings of any severity")
is **MET**.

---

## Cycle 6 finding re-verification

| ID  | Severity prev | Status     | Notes                                                                                                                                                                                                                                                                                                   |
|-----|---------------|------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| C6-1| MEDIUM (B)    | CLOSED     | The cycle-6 F1 top-level cascade is removed. `_size_compatible` carries the cascade only for TyArray / TyTensor / TyTile size-element positions. `fn g[T]() -> T { 42 }` now correctly emits "body type i32 does not match return type T" because `_compatible(TyVar('T'), TyPrim('i32'))` no longer auto-passes at the top. Verified by the new `test_c6_1_compatible_tyvar_not_top_cascade` assertion. |
| G1  | LOW           | CLOSED     | D-arm extra-text branch (line 1364-1378) now uses an `other_is_logic` predicate that checks `(l_is_diff and r_is_logic) or (r_is_diff and l_is_logic)`. The eight-row matrix from the cycle-6 G1 finding all map to accurate diagnostic text. The Logic-arm at line 1382-1409 still emits "(one side Logic-wrapped, other bare)" — this is accurate because the Logic-arm is only reached when neither side carries D (elif chain), so the "other" side cannot be D-wrapped at that branch. |
| G2  | LOW           | CLOSED     | TyMemTier strict-separation now has an explicit two-arm carve-out: `_compatible(TyMemTier(...), TyVar/TySize) → True` and reverse. Placed BEFORE the broad `or` arm so the carve-out takes precedence. The carve-out is documented at lines 2198-2205 (cycle-6 G2 Option A) and the test suite covers the call-boundary case. |

All three cycle-6 findings are closed at the contract level with
no new surface introduced.

---

## Per-surface review (cycle-7 touchpoints)

### Surface 1: `_size_compatible` helper

**Placement**: `helixc/frontend/typecheck.py:2176-2190`, immediately
above `_compatible`. Logical proximity is correct — both are
type-equivalence predicates and the docstring of `_size_compatible`
references `_compatible`.

**Scope**: invoked from exactly three sites — TyArray (2251),
TyTensor (2283), TyTile (2293) — all size-element comparisons.
No external callers. Encapsulation is appropriate; the leading
underscore signals private. The docstring explicitly says "Used
inside TyArray / TyTensor / TyTile size compares".

**Invariant strength**: the helper's contract is "cascade-safe for
TyVar/TySize at the call site's position; defer to full
`_compatible` otherwise". The four-arm ordering (TyVar/TySize →
TyUnknown → identity → delegate) is sound:
- TyVar/TySize-first: explicit cascade for the cycle-4 E1 use case.
- TyUnknown-second: redundant with `_compatible`'s top arm but
  harmless (short-circuit).
- Identity-third: redundant with `_compatible`'s final `return a
  == b` but harmless (short-circuit before the structural arms).
- Delegate-fourth: full structural recursion.

The identity-before-delegate short-circuit is a micro-optimization
(skips structural-arm dispatch for trivially-equal types like
`TyPrim('size_3') == TyPrim('size_3')`). It does not change
behavior because `_compatible(a, b)` ends with `return a == b` for
the same case.

**Risk of misuse**: a future caller could route a value-position
compare through `_size_compatible` and re-introduce the over-broad
cascade C6-1 was closing. Mitigations in place: the docstring
warns explicitly, the leading underscore signals private, the
method name "size" telegraphs the intended scope. No code-level
enforcement, but this is the standard Python convention. Not a
finding.

### Surface 2: `_compatible` TyMemTier × TyVar/TySize carve-out

**Placement**: `helixc/frontend/typecheck.py:2208-2211`, between
the both-MemTier arm (2206-2207) and the broad-or arm (2212-2213).
Arm ordering is correct: the more-specific TyVar/TySize arms run
before the broad rejection.

**Symmetry**: both orders covered. `_compatible(TyMemTier, TyVar)`
and `_compatible(TyVar, TyMemTier)` both return True. The carve-
out is symmetric.

**Invariant impact on TyMemTier "no cross-tier mixing" contract**:
the carve-out is explicitly noted in the docstring comment at lines
2198-2205. The contract now reads: "TyMemTier values are
incompatible across tiers AND with non-TyMemTier types — EXCEPT
when the other side is TyVar / TySize / TyUnknown, in which case
the compare defers to mono substitution". The cycle-6 G2
documentation-only fix would have noted this; cycle 7 implements
the carve-out explicitly with the comment, which is stronger than
G2's Option A (the cycle-6 doc preferred path) because the carve-
out is now in the code rather than implied by the cycle-6-F1
cascade-arm-runs-first ordering. With cycle 7's removal of the
top-level cascade, the explicit arms are required to preserve the
intent.

**Interaction with nested TyVar inside TyMemTier**: the case
`_compatible(TyMemTier(W, TyVar('T')), TyMemTier(W, TyPrim('i32')))`
recurses through the both-MemTier arm into `_compatible(TyVar('T'),
TyPrim('i32'))`, which now falls through to `return a == b` =
False. This matches PRE-cycle-6 behavior (cycle-6 F1 temporarily
made this pass via the top-level cascade; cycle 7 reverts). The
cycle-7 commit explicitly states the scope is "shape positions
only"; the nested-TyVar-at-value-position-inside-TyMemTier path
is intentionally NOT addressed. Not a regression (pre-cycle-6
baseline restored).

### Surface 3: D-arm extra-text accuracy

**Placement**: `helixc/frontend/typecheck.py:1364-1378`, inside the
D-binop-gate branch.

**Predicate `other_is_logic`**: `(l_is_diff and r_is_logic) or
(r_is_diff and l_is_logic)`. This correctly identifies the case
where exactly one side carries D AND the other side (the non-D
one) carries Logic. The eight-row matrix from cycle-6 G1 all
resolve to accurate diagnostic text per the trace above.

**Symmetry with Logic-arm**: the Logic-arm at line 1382-1409 still
emits "(one side Logic-wrapped, other bare)" without an analogous
predicate. This is correct because the Logic-arm is reached only
when NEITHER side carries D (elif chain at line 1382). At that
point, "other bare" is unambiguous: the non-Logic side cannot be
D-wrapped (else the D-arm would have fired). The cycle-6 G1
recommendation noted this asymmetry was acceptable; cycle-7
matches that recommendation.

**No-fire cases**: when `l_is_diff and r_is_diff` are BOTH True,
the guard `if not (l_is_diff and r_is_diff)` is False and `extra`
stays "". Inner-type-mismatch text in the main warning carries the
operand information. Correct.

---

## Cycle 7 invariant snapshot (post-fix)

The cycle-7 fix-sweep moved several invariants to a clean state:

**`_size_compatible` contract** (typecheck.py:2176-2190):
- New shape-position-only cascade helper.
- TyVar/TySize cascade-pass (first arm).
- TyUnknown cascade-pass (second arm, redundant short-circuit).
- Identity short-circuit (third arm, redundant short-circuit).
- Delegate to `_compatible` (fourth arm, full structural).
- Invoked from TyArray.size, TyTensor.shape, TyTile.shape only.

**`_compatible` contract** (typecheck.py:2192-2298):
- TyUnknown cascades (a or b is TyUnknown → True).
- TyMemTier strict tier match when both sides are TyMemTier.
- TyMemTier × TyVar/TySize carve-out (NEW cycle 7): defers to mono.
- TyMemTier × any-other-type rejects.
- TyQuote / TyDiff / TyLogic / TyTuple / TyArray / TyRef / TyPtr /
  TyFn / TyTensor / TyTile: kind-tagged-equal arms that recurse on
  inner types. TyArray.size / TyTensor.shape / TyTile.shape route
  through `_size_compatible` (NOT `_compatible`) for the cascade-
  at-shape-position-only behavior.
- Catch-all: `return a == b` (identity). TyVar/TySize at top-level
  value position falls through here and rejects unless names match.

**D-binop diagnostic-text contract** (typecheck.py:1349-1381):
- Gate fires on `(l_is_diff or r_is_diff) AND (inner_mismatch OR
  (l_is_diff != r_is_diff))` (cycle-6 F2).
- Extra-text branch distinguishes D-vs-Logic from D-vs-bare via
  `other_is_logic = (l_is_diff and r_is_logic) or (r_is_diff and
  l_is_logic)`.

**Logic-binop contract** (typecheck.py:1382-1409): unchanged
from cycle 6. Extra-text "(one side Logic-wrapped, other bare)"
is unambiguous because the Logic-arm only fires when no D is
present (elif chain).

---

## Cycle 7 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity.**

This cycle finds **0 findings (0 HIGH, 0 MEDIUM, 0 LOW)**.

By the strict criterion, **cycle 7 counts CLEAN**.

The severity trend across cycles is now:
- Cycle 1: HIGH-tier finding(s)
- Cycle 2: HIGH + MEDIUM
- Cycle 3: HIGH + MEDIUM + LOW (multiple LOW)
- Cycle 4: MEDIUM-tier
- Cycle 5: 3 MEDIUM + 3 LOW
- Cycle 6: 1 MEDIUM + 2 LOW
- Cycle 7: 0 + 0 + 0  ←  CLEAN

This is the first cycle to meet the strict criterion under
Audit B. The cycle-7 fix-sweep is narrow, targeted, and does not
expand contract surface — the `_size_compatible` helper is a
single-purpose extraction, the G1 fix is a 6-line predicate, and
the G2 carve-out is two arms with explicit symmetric coverage.

Forward look: the 5-clean-cycles requirement (per the cycle-5 doc's
projection for Python-helixc deprecation) now has its first
qualifying cycle. Cycles 8-12 would need to clean to satisfy that
bar. Each clean cycle should be re-audited against any subsequent
code changes; if cycles 8-12 introduce no new type-design
regressions, cycle 7 is the start of the clean streak.

**Recommendation**: no fix-sweep needed for cycle 7. Proceed to
cycle 8 audit gate.
