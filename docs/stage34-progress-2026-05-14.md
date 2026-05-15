# Stage 34 Progress - 2026-05-14

## Increment 1 - Numeric Bound Implication

Stage 34 started with a focused proof/refinement upgrade: Helix can now carry a
refinement proof from a stronger simple numeric bound to a weaker one.

Examples now accepted:

- `self >= 1.0` proves `self >= 0.0`
- `self <= 0.5` proves `self <= 1.0`
- `0.25 <= self <= 0.75` proves `0.0 <= self <= 1.0`
- Reordered equivalent bounds such as `self >= 0.0` and `0.0 <= self`

The implementation remains fail-closed:

- `self >= 0.0` does not prove `self > 0.0`
- `self <= 1.0` does not prove `self < 1.0`
- Unsupported or non-bound predicates still do not gain proof-carry behavior

## Verification

- Focused Stage 34 regression slice:
  - `python -m pytest -q helixc/tests/test_typecheck.py::test_stage34_numeric_bound_implication_carries_proofs helixc/tests/test_typecheck.py::test_stage34_numeric_bound_implication_respects_strictness helixc/tests/test_cli.py::test_stage34_emit_proof_obligations_json_for_numeric_bound_implication`
  - Result: 3 passed
- Proof/refinement quick gate:
  - `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, 56 tests
- Full typechecker suite:
  - `python -m pytest -q helixc/tests/test_typecheck.py`
  - Result: 240 passed

## Next Stage 34 Work

Good follow-up slices:

- Extend implication to simple equality-derived bounds when it is safe.
- Add proof-artifact summaries that distinguish exact proof-carry from
  implication proof-carry.
- Expand clean gates around proof-carry through arrays, tuples, and references.

## Increment 2 - Equality-Derived Bounds

The next Stage 34 slice extends proof-carry for exact numeric equality. Helix
can now treat `self == N` and `N == self` as both an inclusive lower bound and
an inclusive upper bound.

Examples now accepted:

- `self == 1.0` proves `self >= 0.0`
- `self == 1.0` proves `self <= 1.0`
- `self == 1.0` and `1.0 == self` carry equivalent exact-value proofs

The implementation remains fail-closed:

- `self == 1.0` does not prove `self < 1.0`
- `self == 1.0` does not prove `self > 1.0`
- `self != 0.0` does not prove any lower or upper bound

## Increment 3 - Compound And Container Regression Coverage

Stage 34 now pins proof-carry behavior through compound predicate forms and
simple refined containers.

Covered cases:

- `self >= A && self <= B` can carry proof into equivalent comma-separated
  bounds.
- Comma-separated bounds can carry proof into equivalent `&&` bounds.
- Stronger numeric bounds carry through refined array elements.
- Stronger numeric bounds carry through refined tuple elements.

## Increment 4 - Proof-Carry Artifact Records

Proof artifacts now include a separate `proof_carries` section. This records
cases where Helix did not need a new proof obligation because the value already
carried proof for the target refinement.

Recorded strategies:

- `same-refinement`
- `exact-predicate-subset`
- `numeric-bound-implication`

This keeps `obligations` focused on proof work that had to be checked at the
assignment/call/return site, while still making accepted carried proofs visible
in machine-readable output.

## Increment 5 - Negated Comparison Predicates

Refinement predicates now support boolean negation over supported predicates.
The first proof-carry use is negated numeric comparisons.

Examples now accepted:

- `!(self < 0.0)` behaves as `self >= 0.0`
- `!(self <= 0.0)` behaves as `self > 0.0`
- `!(self > 1.0)` behaves as `self <= 1.0`
- `!(self >= 1.0)` behaves as `self < 1.0`

Strictness is preserved: `!(self < 0.0)` does not prove `self > 0.0`.

## Increment 6 - Container Proof-Carry Artifact Records

Proof-carry records now include refined array and tuple element carries when a
container proof is accepted.

Example:

- Passing `[AtLeastOne; 2]` where `[NonNegative; 2]` is required now emits a
  `proof_carries` record for the array element refinement proof.

This keeps the proof artifact useful for higher-level audit tools that need to
see why a container-valued call or assignment was accepted.

## Increment 7 - Simple Affine Bound Implication

Proof-carry now handles simple linear arithmetic around `self` when reducing
numeric bounds.

Examples now accepted:

- `self + 1.0 >= 2.0` proves `self >= 1.0`
- `2.0 * self >= 2.0` proves `self >= 1.0`
- `1.5 - self >= 1.0` proves `self <= 0.5`

Strictness is preserved: `self + 1.0 >= 1.0` proves `self >= 0.0`, but not
`self > 0.0`.

## Increment 8 - Named Constant Bound Coverage

Stage 34 now pins proof-carry through top-level numeric constants used inside
refinement predicates.

Covered cases:

- `self >= FLOOR` can prove `self >= ZERO` when the constant values imply it.
- `self + OFFSET >= TARGET` can prove `self >= OFFSET` when the constants make
  the affine bound equivalent.

## Increment 9 - Mid-Stage Audit Fixes

The first mid-stage Stage 34 audit found one safety-relevant guard gap and a
few proof-artifact visibility gaps.

Fixes:

- Numeric-bound implication now explicitly requires the source and target
  refinements to erase to the same base type.
- Cross-base refined casts no longer reuse numeric proof just because the
  predicate values imply each other.
- Explicit returns, casts, and function-typed calls now route accepted proof
  carries through the artifact recorder.
- Quick-gate proof-artifact coverage now includes tuple carries plus affine and
  negated-bound carries.

## Increment 10 - Proof-Carry Strategy Summary

Proof artifacts now summarize carried proof strategies in
`summary.proof_carry_strategies`.

Example:

```json
{
  "numeric-bound-implication": 2
}
```

The validator checks that this summary matches the actual `proof_carries`
records, so downstream tools can trust the quick counts without re-walking the
full list.
