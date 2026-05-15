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

Initial Stage 34 work accepted simple linear arithmetic around `self` when
reducing numeric bounds.

Those early examples were later narrowed by clean-gate findings: affine
proof-carry extraction now fails closed for fixed-width integer and
floating-point bases unless it is a simple direct `self` versus constant bound.
The old examples below are retained as design intent only. They are not current
accepted behavior:

- `self + 1.0 >= 2.0` proves `self >= 1.0`
- `2.0 * self >= 2.0` proves `self >= 1.0`
- `1.5 - self >= 1.0` proves `self <= 0.5`

Current behavior is conservative: affine proof-carry strictness examples are
rejected as unproven until Helix can model overflow and rounding exactly.

## Increment 8 - Named Constant Bound Coverage

Stage 34 now pins proof-carry through top-level numeric constants used inside
refinement predicates.

Covered cases:

- `self >= FLOOR` can prove `self >= ZERO` when the constant values imply it.
- Direct named bounds remain supported. Affine named-constant proof carry for
  fixed-width numbers is now fail-closed until overflow and rounding semantics
  can be modeled exactly.

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
- Quick-gate proof-artifact coverage now includes tuple carries plus
  negated-bound carries, while affine cases are covered as fail-closed
  typecheck regressions.

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

## Clean Gate 1 Third Restart - Failed; Fix Verified; Counter Reset

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

## Clean Gate 1 Fourth Restart - Failed; Fix Verified; Counter Reset

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

## Clean Gate 1 Fifth Restart - Failed; Fix Verified; Counter Reset

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

## Clean Gate 1 Sixth Restart - Failed; Fix Verified; Counter Reset

The next clean-gate attempt found another proof-honesty edge case:

- If a refined type used only self-independent predicates such as `where true`,
  an unrepresentable constant value could still be treated as proven because
  there were no pending `self` predicates to force the target-representation
  error path.
- The same issue could poison calls through a function whose refined return
  body failed; later callers still saw the declared refined return type.

The fix makes any known constant value that cannot be represented by the
erased target type a proof/typecheck error, even when all predicates are
self-independent. It also tracks functions whose refined return checking
failed and erases their return refinements for later direct calls and function
references, preventing downstream proof carries from a failed producer. The
function-body pass now reaches a fixed point over failed refined-return
producers, so callers declared before a failed producer are also checked with
that fail-closed knowledge.

Two follow-up audits of the same restart found related holes before this fix
set was committed:

- The fixed-point body loop truncated diagnostics between passes but did not
  reset unbound-name suppression, so `fn bad() -> AlwaysI32 { missing }` could
  lose the `unbound name 'missing'` diagnostic and leave a false proved
  `where true` obligation.
- Refinement predicate float literals with explicit suffixes were evaluated as
  raw Python floats. For example, `16777217.0_f32` inside a predicate did not
  round to `16777216.0`, allowing a false proof of
  `16777216.0_f32 < 16777217.0_f32`.

The final fix also resets unbound-name suppression on each fixed-point pass and
evaluates explicit `_f32` / `_f64` predicate literals through their Helix
representation before comparison.

Verification after the self-independent and invalid-return fixes:

- Predicate-literal focused regressions: `2 passed`
- Nearby proof-carry and proof-gate slices: `33 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `451 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass after built-in retry recovered no-codegen shards 2 and 3

## Clean Gate 1 Seventh Restart - Failed; Fix Verified; Counter Reset

The next clean-gate attempt found another set of proof-honesty issues:

- Numeric-bound proof-carry implication still evaluated predicate constants as
  raw Python floats. This meant `self >= 16777217.0_f32` could imply
  `self > 16777216.0_f32` even though both bounds represent
  `16777216.0` as `f32`.
- Top-level const scalar caching stored raw literal values before the declared
  const type was applied, so `const LIMIT: f32 = 16777217.0_f32` could be used
  as if `LIMIT` were still `16777217.0` inside a predicate.
- The fixed-point function-body loop reset unbound-name suppression but not
  unknown-type suppression, so an `unknown type 'Missing'` diagnostic could be
  lost across fixed-point passes.

The fix makes numeric-bound and affine proof-carry extraction evaluate
predicate constants with explicit float suffix representation. It also stores
top-level scalar const values after casting them through their declared
primitive/refined base type, and resets unknown-type suppression on each
fixed-point pass.

Verification after the bound-implication, const-cache, and unknown-type fixes:

- Exact focused regressions: `8 passed`
- Validator trust regressions: `5 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `463 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass with no shard retries

Two follow-up clean-gate auditors found additional coverage and trust gaps
before this fix set was committed:

- Unsuffixed float predicate literals must also use Helix's default `f32`
  representation in predicate evaluation and bound implication.
- Predicate arithmetic can create `inf` or `nan` after literal checks; those
  results must fail closed instead of becoming proof constants.
- Plain `proof_artifact_validate.py --source` should reject a forged artifact
  `path`, not only `--require-clean`.

Those are now covered by additional typecheck, proof-gate, validator, and quick
gate tests.

## Clean Gate 1 Eighth Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate auditors on commit `1487810` found two more Stage 34
proof-honesty gaps:

- `f32` predicate arithmetic rounded literal leaves but did not round each
  arithmetic operation result back through `f32`. This let
  `self + 1.0_f32 > 16777216.0_f32` prove true for
  `self = 16777216.0_f32`, even though real `f32` arithmetic rounds the left
  side to `16777216.0`.
- Normal `proof_artifact_validate.py --source` still accepted source-backed
  artifacts with `path: null`, and it also accepted artifacts where carried
  proof metadata was stripped. `--require-clean` caught the forged artifacts by
  recomputing, but plain validation now fails closed too.

The fix threads the erased numeric base into refinement predicate evaluation,
rounds `f32` predicate arithmetic results after each operation, disables affine
real-number bound extraction for `f32` proof carries, stores local scalar consts
through their declared representation, and requires source-backed proof
artifacts to keep `path`, `proof_carries`, `summary.proof_carries`, and
`summary.proof_carry_strategies`.

Verification after this fix set:

- Focused auditor regressions: `6 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `467 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass across all 12 shards

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Ninth Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate auditors on commit `3d20693` found two proof-honesty gaps:

- Float affine proof-carry extraction still treated target predicates as exact
  real-number algebra. This let precision-boundary values satisfy predicates
  such as `self + 1.0 > 16777216.0` for `f32`, even when IEEE arithmetic rounds
  the addition back to the original value. The same risk existed for `f64`.
- Plain source-backed proof artifact validation rejected missing carried-proof
  metadata but still accepted an internally consistent artifact where
  `proof_carries` was erased and `summary.proof_carries` /
  `summary.proof_carry_strategies` were updated to match the erased list.

The fix passes the erased numeric base into target requirement extraction,
fails closed for affine proof-carry extraction over floating-point bases,
keeps direct integer and float bounds as the supported proof-carry subset, and
makes normal `proof_artifact_validate.py --source` recompute and compare
carried-proof metadata even without `--require-clean`.

Verification after this fix set:

- Focused float-affine and erased-carry regressions: `7 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `470 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass across all 12 shards

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Tenth Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate auditors on commit `4be3e3c` found one docs issue and two
proof-honesty gaps:

- The progress doc still had old affine examples worded as current accepted
  behavior. Those examples are now documented as design intent only.
- Integer refinement predicate arithmetic used Python real division, while Helix
  integer division truncates toward zero. This allowed `self / 2 > 0` to prove
  true for `self = 1`.
- Source-backed proof artifact validation recomputed carried-proof metadata only
  when the user supplied explicit `--source`; an artifact with a valid embedded
  source path could still erase proof carries and update its summary.

The fix makes direct integer predicate arithmetic use Helix-style truncating
division and modulo, fails closed when fixed-width integer arithmetic leaves the
declared range, fails closed for affine proof-carry extraction over all
fixed-width numeric bases, and recomputes carried-proof metadata from either an
explicit source or a resolvable embedded artifact path.

Verification after this fix set:

- Focused fixed-width arithmetic and embedded-carry regressions: `9 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `473 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass after retry recovered no-codegen shard 2
- `python -m pytest -q helixc/tests/test_strings_io.py::test_print_int_decimal_output`:
  pass after inspecting the recovered shard's transient failure

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Eleventh Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate auditors on commit `2ebac36` found two proof-honesty gaps:

- Refined initializer checks used generic Python scalar evaluation instead of
  the declared source machine type. This let `-1_i32 % 2_i32` pass a positive
  refinement because Python modulo produced `1`, while Helix signed machine
  modulo produces `-1`. The same path could also miss per-operation `f32`
  rounding at precision boundaries.
- Plain source-backed artifact validation recomputed some carry metadata, but
  did not compare the full proof-relevant artifact body. A forged artifact could
  promote an unproved obligation to `proved` and erase typecheck errors without
  being rejected by default validation.

The fix evaluates refined initializer and cast constants through the erased
source numeric type before target refinement checking, keeps a raw-scalar
fallback for representability diagnostics, compares the full source-recomputed
proof artifact surface, and keeps strict clean-policy reporting active even when
source recomputation already finds mismatches.

Verification after this fix set:

- Focused refined-initializer and source-recompute regressions: `6 passed`
- Follow-up strict-mode and representability regressions: `8 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `476 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass after retry recovered no-codegen shard 1
- `python -m pytest -q helixc/tests/test_transcendentals.py::test_grad_through_user_defined_function_call`:
  pass after inspecting the recovered shard's transient failure

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Twelfth Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate auditors on commit `8ddb14f` found two proof-honesty gaps and
one documentation consistency issue:

- Source-backed default validation recomputed the proof body, but did not
  compare recomputed `input` or `cache_key`. A forged artifact could advertise a
  different proof input and internally consistent cache key while still passing
  default validation.
- Constant scalar evaluation did not model nested `as` casts. This let
  `(1e309_f64 as f64) as AlwaysF64` hide the same nonfinite representability
  failure that direct `1e309_f64 as AlwaysF64` already rejected.
- The progress file listed recent restart sections out of order
  (`Seventh`, `Tenth`, `Ninth`, `Eighth`). The sections now read
  chronologically.

The fix makes source-backed default validation compare `schema`, `cache_key`,
`path`, and `input` in addition to the proof body, adds constant evaluation for
simple primitive casts, and keeps raw-scalar diagnostics from hiding known
unrepresentable cast sources.

Verification after this fix set:

- Focused latest-reset regressions: `7 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `478 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass after retry recovered no-codegen shards 1 and 2
- `python -m pytest -q helixc/tests/test_strings_io.py::test_print_str_multiple_calls helixc/tests/test_strings_io.py::test_print_int_decimal_output`:
  pass after inspecting the recovered shard's transient failures

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Thirteenth Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate auditors on commit `eaff977` found two proof-honesty gaps and
one documentation consistency issue:

- Constant scalar evaluation still collapsed nonfinite arithmetic to unknown.
  This let `(1e309_f64 + 0.0_f64) as AlwaysF64` pass a `where true`
  refinement and produce clean proof artifacts, even though the value is not a
  representable `f64`.
- Source-backed validation replayed artifact-controlled `input.flags` directly
  into `helixc.check`. A forged artifact could use flags such as `-o` and cause
  validation to write an output file before validation failed.
- The progress file still had a duplicate unnumbered `Clean Gate 1 Restart`
  after `Second Restart`, then continued with `Third Restart`. That made the
  stage record non-chronological.

The fix preserves raw nonfinite arithmetic evidence for refinement cast
diagnostics, reconstructs validator replay arguments from a proof-safe
whitelist, compares relative and absolute source paths semantically during
source-backed recomputation, and renumbers the restart headings into sequence.

Verification after this fix set:

- Focused latest-reset regressions: `5 passed`
- Focused proof artifact validator/gate files: `65 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `480 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass across all 12 shards with no retries
- `git diff --check`: pass

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Fourteenth Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate auditors on commit `c16afeb` found two more Stage 34 issues:

- Refined cast checking could fall back from failed typed `f32` constant
  evaluation to raw Python double arithmetic. This let
  `(3.4028235e38_f32 * 2.0_f32) as AlwaysF64` prove cleanly even though the
  `f32 * f32` operation overflows before the value can be refined as `f64`.
- `proof_artifact_validate.py --require-clean` ignored the embedded source path
  that default validation had already resolved. A valid clean artifact with
  `path: "input.hx"` passed normal validation but failed strict validation
  unless `--source input.hx` was supplied redundantly.

The fix distinguishes a truly unknown constant from a known source value that
failed representation under its typed source base, and lets raw fallback force a
diagnostic without rescuing a failed typed computation into a valid proof value.
Strict clean validation now reuses the same resolved source path used by default
source-backed recomputation.

Verification after this fix set:

- Focused latest-reset regressions: `5 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `482 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass across all 12 shards with no retries

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Fifteenth Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate auditors on commit `0f8b6ca` found two more Stage 34 issues:

- A primitive `as f64` cast could hide a prior `f32` overflow in refined return
  checking. `(3.4028235e38_f32 * 2.0_f32) as f64` returned as `AlwaysF64`
  passed even though the inner `f32 * f32` overflowed before the value reached
  the `f64` cast.
- Source-unavailable proof artifacts still accepted unsafe or impossible
  `input.flags`, including output flags such as `-o`, because the proof-safe
  replay flag whitelist only ran during source-backed recomputation.

The fix detects unrepresentable typed constant subexpressions inside primitive
casts and runs proof-safe flag validation during structural artifact validation,
including source-unavailable artifacts.

Verification after this fix set:

- Focused latest-reset regressions: `5 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `484 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass across all 12 shards with no retries

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Sixteenth Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate auditors on commit `e879f48` found two more Stage 34 issues:

- A top-level or local primitive `const` could hide a prior `f32` overflow from
  later refined return checking. The checker remembered only successfully
  represented scalar constants, so references to an unrepresentable constant
  source could look like an unknown value and pass a self-independent
  refinement such as `where true`.
- Proof artifact replay inputs still accepted libraries. The gate allowed
  `-l forgedlib`, and source-backed validation reconstructed `-l <lib>` from
  artifact-controlled `input.libs`.

The fix tracks named constants whose source scalar expression contains an
unrepresentable typed value, keeps local-const proof evidence visible while
checking function final-expression refinements, rejects `-l` at the clean proof
gate, and requires `input.libs` to be empty for proof replay validation.

Verification after this fix set:

- Focused latest-reset regressions: `4 passed`
- Wider targeted regressions: `7 passed`
- `python scripts\stage31_validate.py --mode quick`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `342 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass after built-in retry recovered no-codegen shard 3

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Seventeenth Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate auditors on commit `5f422b5` found three more Stage 34 issues:

- Suffixed integer literals were not checked against their own source width
  before refined proof checking. This let `2147483648_i32 as PositiveI64`
  record a proved obligation after casting to `i64`, even though the source
  literal cannot be represented as `i32`.
- Source-unavailable artifacts accepted impossible input metadata, including
  opt levels outside `0..3`, unknown warning names or policies, and fake stdlib
  manifests with self-consistent hashes.
- The progress document still had stale affine proof-carry wording that sounded
  accepted, even though current Stage 34 behavior rejects those affine cases as
  unproven.

The fix makes integer constant evaluation apply literal suffix/source-base
representation before refined proofs, hardens source-unavailable artifact
metadata validation to match proof-safe checker inputs, and updates the affine
documentation to describe current fail-closed behavior.

Verification after this fix set:

- Focused latest-reset regressions: `3 passed`
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `344 passed`
- `python scripts\stage31_validate.py --mode quick`: pass
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass across all 12 shards with no retries

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Eighteenth Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate auditors on commit `2d91f4e` found one more validator trust
issue:

- Source-unavailable artifacts could still carry forged `warning_diagnostics`,
  and their `input.flags` could be duplicated or non-canonical even though real
  artifacts emit sorted and deduplicated flags.

A proof-soundness auditor did not find a new scalar/refinement false-clean in
its temp repro matrix, but it did catch quick-gate assertions that needed to be
updated after the stricter validator behavior.

The fix requires proof replay flags to match the compiler's canonical metadata
form and rejects `pipeline_errors` plus `warning_diagnostics` when
`input.source_sha256` is null. Tests now expect these stricter structural
failures.

Verification after this fix set:

- Focused latest-reset regressions: `4 passed`
- `python -m pytest -q helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `74 passed`
- `python scripts\stage31_validate.py --mode quick`: pass
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass after built-in retry recovered no-codegen shard 1
- `python -m pytest -q helixc/tests/test_strings_io.py::test_print_str_writes_to_stdout`:
  pass after inspecting the recovered shard's transient failure

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Nineteenth Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate auditors on commit `2acc0cc` found another Stage 34
proof-soundness issue:

- Value-producing control flow could hide an unrepresentable scalar source from
  refined proof checking. Direct `1e309_f64` returns failed correctly, but
  `if b { 1e309_f64 } else { 0.0_f64 }` returned as `AlwaysF64 where true`
  could still produce a proved obligation and pass clean proof validation.
  Similar holes existed through `match`, local `let` indirection, and refined
  casts fed by such control flow.

The fix makes unrepresentable typed scalar detection walk value-producing
syntax such as `if`, `match`, blocks, tuples, arrays, structs, fields, indexes,
calls, and assignments. Local `let` bindings initialized from such sources now
carry fail-closed evidence into later name references, and simple assignments
update that local evidence.

Verification after this fix set:

- Focused latest-reset regressions: `2 passed`
- Stage 34 focused typecheck/CLI/proof-gate slice: `51 passed`
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `490 passed`
- `python scripts\stage31_validate.py --mode quick`: pass
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass across all 12 shards with no retries

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Twentieth Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate auditors on commit `02007a5` found one more Stage 34
proof-soundness issue and one coverage/documentation mismatch:

- A primitive-return function could hide an unrepresentable scalar from a later
  refined proof. For example, `raw_bad() -> f64 { 1e309_f64 }` could be called
  by `f() -> AlwaysF64 where true`, and the refined return could look clean.
- The prior Nineteenth restart progress note claimed unrepresentable scalar
  detection covered blocks, tuples, arrays, structs, fields, indexes, calls,
  and assignments, but tests only pinned a narrower subset at that time.

A separate validator trust-boundary auditor did not find a proof artifact
gate/validator bypass on `02007a5`.

The fix tracks primitive-return producers as unsafe proof sources without
turning plain primitive returns into immediate type errors. Calls to those
producers now count as unrepresentable evidence when used for refined proofs.
The regression suite now pins primitive producers, explicit returns, local call
consumers, block final expressions, tuple elements, array elements, struct
fields, field/index access, call arguments, assignment evidence, and assignment
repair/clear behavior. The proof artifact gate also covers the
primitive-producer false-pass shape, and the quick gate includes the new Stage
34 coverage tests.

Verification after this fix set:

- Focused latest-reset regressions plus const-fold compatibility check:
  `4 passed`
- Stage 34 focused typecheck/CLI/proof-gate slice: `53 passed`
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `493 passed`
- `python scripts\stage31_validate.py --mode quick`: pass
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass after built-in retry recovered no-codegen shard 1
- `python -m pytest -q helixc/tests/test_strings_io.py::test_print_int_zero`:
  pass after inspecting the recovered shard's transient failure

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Twenty First Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate auditors on commit `2cc20ba` found one more Stage 34
proof-soundness issue and one audit-trail issue:

- An unrepresentable scalar could enter a primitive numeric parameter of a
  function that returns a refined value. For example,
  `accept(1e309_f64)` could call `accept(x: f64) -> AlwaysF64` and let the
  callee prove `where true` from parameter `x`.
- The Twentieth restart progress entry was inserted before older restart
  sections, leaving the chronological audit trail misleading.

The fix rejects unrepresentable typed scalar evidence at the call boundary when
the callee can return a refined value through compatible primitive numeric
parameters. This keeps Stage 34 fail-closed until Helix has finer
interprocedural dependency tracking. The progress file now records the
Nineteenth, Twentieth, and Twenty First restarts in chronological order.

Verification after this fix set:

- Focused latest-reset regressions: `5 passed`
- Stage 34 focused typecheck/CLI/proof-gate slice: `54 passed`
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `495 passed`
- `python scripts\stage31_validate.py --mode quick`: pass
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass after built-in retry recovered no-codegen shards 1 and 2
- `python -m pytest -q helixc/tests/test_strings_io.py::test_read_file_int_round_trips helixc/tests/test_strings_io.py::test_read_file_int_missing_file_returns_zero`:
  `2 passed` after inspecting the recovered shard failures
- Direct checks for the proof-artifact auditor's archived-copy full-gate
  concerns: stage31 validation shard-guard repros `2 passed`, strings I/O
  repros `3 passed`

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Twenty Second Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate auditors on commit `8a497f4` found one more Stage 34
proof-soundness issue and two clean-gate reproducibility issues:

- Generic pass-through could hide unrepresentable scalar evidence before a
  refined-return call. `accept(1e309_f64)` failed correctly, but
  `accept(id(1e309_f64))` could pass because the generic `id[T]` left the
  argument type as `T` at the call boundary.
- `git archive` could extract shell scripts with CRLF line endings, so bash
  rejected `scripts/run_all_tests.sh` before tests ran.
- Some WSL runtime helpers used a hardcoded live checkout path, so archive-copy
  tests could execute binaries from `C:\Projects\Kovostov-Native` instead of
  the extracted archive.

The fix runs the Stage 34 representability check across deferred generic
`TyVar` and `TySize` argument/parameter boundaries when the callee can return a
refined value. It also pins shell scripts to LF through `.gitattributes`, makes
strings I/O, reflection, and select-codegen WSL helpers derive paths from the
current checkout, and gives those helpers unique temporary binary names to
avoid parallel shard collisions.

Verification after this fix set:

- Focused latest-reset and archive-helper regressions: `7 passed`
- Stage 34 focused typecheck/CLI/proof-gate slice: `55 passed`
- Direct archive-repro tests for shard guards and strings I/O: `5 passed`
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py helixc/tests/test_strings_io.py helixc/tests/test_reflection.py helixc/tests/test_select_codegen.py`:
  `526 passed`
- `python scripts\stage31_validate.py --mode quick`: pass
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass across all 12 shards with no retries
- Staged-tree archive check for shell scripts: `scripts/run_all_tests.sh`,
  `stage0/hex0/run_tests.sh`, and `stage0/hex0/build.sh` extracted with
  `CRLF=0`; `bash -n` accepted the shell scripts, and archive-copy
  `test_print_int_zero` passed

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Twenty Third Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate docs and coverage auditors on commit `343587d` found one
stale broad claim in a touched test file:

- `helixc/tests/test_reflection.py` made an overbroad uniqueness claim about
  verifier-gated reflection rather than limiting the docstring to the runtime
  behavior under test.

The fix narrows the reflection test docstring to the behavior under test:
verifier-gated reflection runtime cells, `modify`, and `splice`.

Verification after this fix set:

- Stale broad-claim grep in `helixc/tests/test_reflection.py`: no matches
- Focused regression slice: `3 passed`
- `python scripts\stage31_validate.py --mode quick`: pass
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass across all 12 shards with no retries
- Staged-tree archive check for shell scripts: `scripts/run_all_tests.sh`,
  `stage0/hex0/run_tests.sh`, and `stage0/hex0/build.sh` extracted with
  `CRLF=0`; `bash -n` accepted the shell scripts

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Twenty Fourth Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate auditors on commit `e27eb87` found one more Stage 34
proof-soundness issue and one documentation discipline issue:

- A generic wrapper returning a refined value could still hide unrepresentable
  scalar evidence. `via[T](x: T) -> AlwaysF64` accepted `via(1e309_f64)`
  because the representability check used formal type `T`, which has no
  concrete numeric base.
- `helixc/tests/test_reflection.py` still had comments whose wording exceeded
  the specific behavior under test.

The fix carries erased numeric base evidence alongside local unrepresentable
scalar markers and lets the Stage 34 representability check recover a base from
the expression itself when the formal parameter type is still generic.
Typecheck and proof-gate regressions now pin direct generic wrappers,
local-let wrappers, and nested generic pass-through into a refined return. The
remaining reflection comments were narrowed to test-scoped behavior.

Verification after this fix set:

- Focused generic-wrapper and reflection regressions: `20 passed`
- Stage 34 focused typecheck/CLI/proof-gate slice plus reflection tests:
  `56 passed`
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py helixc/tests/test_strings_io.py helixc/tests/test_reflection.py helixc/tests/test_select_codegen.py`:
  `528 passed`
- `python scripts\stage31_validate.py --mode quick`: pass
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass across all 12 shards with no retries

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Twenty Fifth Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate docs and coverage auditors on commit `c9f9606` found one
remaining inaccurate comment in `helixc/tests/test_reflection.py`:

- `test_verifier_can_bound_state` described the test as gradient-descent
  learning, but the body performs fixed verifier-gated `modify` updates and
  checks the final reflected state.

The fix narrows that comment to the behavior under test: fixed verifier-gated
updates must keep reflected state inside a safe range.

Verification after this fix set:

- Broad reflection wording grep in `helixc/tests/test_reflection.py`: no
  matches
- Focused regression slice: `3 passed`
- `python scripts\stage31_validate.py --mode quick`: pass
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass across all 12 shards with no retries

The clean-gate counter remains reset to `0/3`.

## Additional Finding From Clean Gate 1 Twenty Fifth Restart - Fix Verified

The same failed clean-gate rotation on commit `c9f9606` also found one more
Stage 34 proof-honesty gap:

- Indexed assignments could hide unrepresentable scalar evidence. Plain
  assignment evidence worked for `x = bad`, but `xs[0] = bad; xs[0]` did not
  mark the aggregate `xs`, so the later index read could prove a
  self-independent refinement as clean.

The fix set tracks simple static indexed evidence separately. A later indexed
read sees the matching unsafe element marker and refined proof checking fails
closed, while a repair assignment to the same static element can clear only
that element.

Verification after this fix set:

- Focused index-assignment and reflection checks: `3 passed`
- Stage 34 focused typecheck/CLI/proof-gate slice plus reflection tests:
  `57 passed`
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py helixc/tests/test_strings_io.py helixc/tests/test_reflection.py helixc/tests/test_select_codegen.py`:
  `530 passed`
- `python scripts\stage31_validate.py --mode quick`: pass
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass across all 12 shards with no retries

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Twenty Sixth Restart - Failed; Fix Verified; Counter Reset

Fresh clean-gate auditors on commit `b374615` found one more Stage 34
proof-soundness issue and two documentation discipline issues:

- Indexed assignment repair was too broad. After `xs[0] = bad`, a clean write
  to `xs[1]` cleared the whole aggregate marker, so a later `xs[0]` read could
  still look clean.
- The previous reflection-bound-comment and index-assignment findings were
  recorded as sequential restarts even though both came from the same failed
  `c9f9606` clean-gate rotation.
- `helixc/tests/test_reflection.py` still had broad binary-classifier comment
  wording.

The fix tracks simple static indexed evidence per element. A clean write to
`xs[1]` can clear only `xs[1]`, while evidence on `xs[0]` remains visible to
later `xs[0]` reads. The historical docs now describe the reflection and index
assignment findings as one failed `c9f9606` rotation, and the broad reflection
comments were narrowed to the behaviors under test.

Verification after this fix set:

- Focused index-repair and reflection checks: `4 passed`
- Stage 34 typecheck/proof-gate/reflection slice:
  `38 passed, 296 deselected`
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py helixc/tests/test_strings_io.py helixc/tests/test_reflection.py helixc/tests/test_select_codegen.py`:
  `531 passed`
- `python scripts\stage31_validate.py --mode quick`: pass
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass across all 12 shards

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Twenty Seventh Restart - Failed; Fix Verified; Counter Reset

Fresh archive reproducibility auditors on commit `b636256` found one Stage 34
gate issue outside the proof checker:

- A clean `git archive HEAD` extraction failed `stage0/hex0/run_tests.sh`
  under WSL because `01-hello.expected` and `02-comments-ws.expected` exported
  with CRLF bytes. The expected values contained a trailing carriage return,
  while the actual `hex0.bin` output did not.

The fix adds `.gitattributes` rules forcing LF for Stage 0 hex0 `.expected`
and `.hex0` fixtures. A new regression test archives the candidate tree,
checks those fixture bytes, and runs the Stage 0 shell gate from the extracted
archive. The quick validation list now includes that archive fixture
regression.

Verification after this fix set:

- Candidate archive fixture byte scan: `CR=0` for the checked Stage 0 fixture
  files
- Candidate archive Stage 0 shell gate: `3 passed, 0 failed`
- Stage 0 archive regression: `1 passed`
- `python scripts\stage31_validate.py --mode quick`: pass

The clean-gate counter remains reset to `0/3`.

## Clean Gate 1 Passed - Counter 1/3

Three fresh read-only clean-gate lanes passed on commit `8cc5512`:

- Proof-soundness audit: PASS with high confidence. It found no refined-return
  false-clean across the Stage 34 proof surfaces and verified the wrong-index
  repair shape fails closed while same-index repair remains allowed.
- Proof artifact and archive reproducibility audit: PASS with high confidence.
  It verified the committed archive has LF shell scripts and Stage 0 fixtures,
  the extracted Stage 0 run/build gates pass, proof-artifact negative tests
  pass, and quick/full validation pass from the archive tree.
- Documentation and gate-discipline audit: PASS with medium confidence and no
  findings. It verified chronology through Twenty Seventh and the clean-gate
  counter wording.

The clean-gate counter advances to `1/3`. Clean Gate 2 should start from the
commit containing this pass record.
