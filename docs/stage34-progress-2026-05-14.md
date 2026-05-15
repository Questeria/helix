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

## Initial Verification

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

Current follow-up slices:

- Restart three clean audit gates after the target-representation fixes are
  verified.
- Keep proof-carry artifact coverage aligned with every accepted proof-carry
  strategy and route.
- Consider reference, pointer, tensor, and tile proof-carry artifact records
  only when Stage 34 explicitly claims those surfaces.

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

## Clean Gate 1 - Failed; Fix Verified; Counter Reset

The first Stage 34 clean gate did not count as clean. The audit found three
issues:

- Cross-base refined casts checked literal predicates before target conversion.
  Example: `0.5_f64 as ExactlyHalfInt` could incorrectly prove
  `self == 0.5` for an `i32` refinement even though the target value becomes
  `0`.
- Boolean-to-numeric refined casts could also bypass the normal assignment
  check and produce a misleading proved artifact.
- The artifact route claims for explicit returns, refined casts, and
  function-typed calls needed direct regression tests.

The fix now checks refined cast predicates against the converted target value
and refuses unsupported source-to-target proof conversions instead of recording
a false proof. It also adds direct machine-readable proof-artifact coverage for
those three routes. Clean-gate counting restarts after the focused and quick
gates are green again.

Verification after the fix:

- Focused regression slice: `7 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py`: `252 passed`
- `python -m pytest -q helixc/tests/test_cli.py helixc/tests/test_proof_artifact_gate.py`:
  `148 passed`

## Clean Gate 1 Restart - Failed; Fix Verified; Counter Reset

The restarted clean gate also did not count as clean. It found that target
representation must be checked before a proof is recorded:

- Refined integer aliases could prove values impossible for their erased base,
  such as `type Exactly300 = u8 where self == 300`.
- Refined `f32` casts could prove predicates against Python's unrounded
  double value instead of the real stored `f32` value.
- The `same-refinement` proof-carry strategy needed a direct producer test,
  not only a validator allowlist entry.
- Older planning text still used obsolete Stage 34 numbering.

The fix now checks predicates against the represented target value for both
plain refined values and refined casts, adds direct regressions for impossible
integer aliases and rounded `f32` values, pins the `same-refinement` artifact
strategy, and marks older stage-numbering plans as superseded by the live
roadmap.

Verification after the target-representation fix:

- Focused regression slice: `5 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_gate.py`:
  `405 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass

## Clean Gate 1 Second Restart - Failed; Fix Verified; Counter Reset

The second restarted clean gate did not count as clean. The proof-artifact
audit found that `f32` overflow could still be converted to `inf` inside the
checker and then marked proved, while the backend still tried to pack the
original finite literal and failed.

The fix now treats `f32` overflow as not representable for proof purposes, so
overflowing refined `f32` literals fail before any proved obligation can be
recorded.

The same clean-gate attempt also found that a failed refined cast still
returned the refined target type internally, allowing an enclosing return to
record a misleading `same-refinement` carry. The cast checker now reports
whether its proof succeeded; failed refined casts no longer type as proven
refined values, so enclosing contexts cannot record proof carries from them.

Verification after the overflow and failed-cast artifact fixes:

- Focused regression slice: `5 passed`
- Exact failed-cast repro: `rc=1`, `proof_carries=[]`; pinned by
  `test_stage34_failed_refined_cast_does_not_emit_return_carry`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_gate.py`:
  `408 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass after built-in retry recovered no-codegen shard 1

## Clean Gate 1 Restart - Failed; Fix Verified; Counter Reset

The next clean-gate attempt found two more artifact-honesty gaps:

- Failed casts to refined composite targets, such as `[NonNegative; 1]`, still
  typed internally as the refined target and could let enclosing `let`
  contexts record bogus array or tuple proof carries.
- `proof_artifact_validate.py --require-clean --source` verified source hashes
  but did not recompute the artifact, so a forged clean JSON artifact could be
  accepted when the source itself still failed proof checking.

The fixes make rejected refined composite casts type as unknown and make the
validator recompute source-backed clean artifacts before accepting them.

The follow-up validator audit found two tighter trust-boundary requirements:

- Source-backed `--require-clean` recomputation must compare the artifact
  `path`, not only hash-bearing metadata and proof lists.
- Source-unavailable artifacts must not carry `proof_carries`, because no
  source exists to support those proof records.

Verification after the validator path-honesty fixes:

- Focused regression slice: `3 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `42 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass after built-in retry recovered no-codegen shard 4

Verification after the composite-cast and validator fixes:

- Focused regression slice: `3 passed`
- Composite failed-cast repro: type errors present, `proof_carries=[]`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `435 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass

## Clean Gate 1 Third Restart - Failed; Fix Verified; Counter Reset

The next clean-gate attempt still did not count as clean. The replacement
soundness audit found two more ways proof artifacts could overstate what the
checker had actually proved:

- A failed refined initializer still bound the declared refined type in local
  scope. Later returns or function-typed calls could then record a
  `same-refinement` carry for a variable whose initializer had already failed.
- A non-finite `f32` literal such as `1e309_f32` could enter the checker as
  `inf` and satisfy impossible predicates before backend representation checks
  had a chance to reject it.

The fix makes failed refined `let` and local `const` initializers bind the
erased base type instead of the refined declared type. That preserves the
shape of the value for follow-up diagnostics but prevents later code from
carrying a proof that was never established. The checker now also rejects
non-finite scalar values before using them for refined `f32` or `f64` proof
evaluation.

Verification after the failed-initializer and non-finite literal fixes:

- New focused regressions: `3 passed`
- Nearby proof-carry and proof-gate slice: `29 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `439 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass after built-in retry recovered no-codegen shards 1 and 3

## Clean Gate 1 Fourth Restart - Failed; Fix Verified; Counter Reset

The next clean-gate attempt found two more proof-honesty issues:

- Failed top-level refined `const` initializers still resolved as their
  declared refined type when referenced later. That could emit false
  `same-refinement` carries through call arguments and returns.
- Non-finite float values cast to refined integer aliases could raise an
  internal `OverflowError` during `int(value)` conversion instead of becoming a
  normal proof/typecheck failure.

The fix tracks invalid top-level const declarations and erases their
refinements during later `Name` lookup. It also rejects non-finite integer
target conversions before calling `int()`, so casts such as
`1e309_f64 as NonNegativeInt` fail closed without crashing the checker.

Verification after the top-level const and non-finite integer fixes:

- New focused regressions: `5 passed`
- Nearby proof-carry and proof-gate slice: `28 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `444 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass with no shard retries
