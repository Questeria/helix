# Stage 34 Clean Gate 1 Failed Audit - 2026-05-15

This audit attempt did not count as a clean gate. Three independent Stage 34
auditors found real proof-honesty or coverage issues after commit `62987c4`.

## Findings

1. Failed top-level refined `const` declarations could still emit false proof
   carries when referenced later.

   Minimal shape:

   ```hx
   type One = i32 where self == 1;
   const BAD: One = 2;
   fn use_one(x: One) -> i32 { 0 }
   fn f() -> i32 { use_one(BAD) }
   fn g() -> One { BAD }
   ```

   Observed before the fix: the artifact had the expected const proof failure,
   but also recorded `same-refinement` carries for the call argument and return.

2. Non-finite float values cast to refined integer aliases could crash the
   checker.

   Minimal shape:

   ```hx
   type NonNegativeInt = i32 where self >= 0;
   fn f() -> NonNegativeInt {
       1e309_f64 as NonNegativeInt
   }
   ```

   Observed before the fix: an internal `OverflowError` from `int(inf)`.

3. Coverage needed to pin local const initializer erasure and `f64`
   non-finite proof rejection, not only local `let` and `f32`.

## Fix Summary

- Track invalid top-level const declarations and erase their refinements during
  later `Name` lookup.
- Treat non-finite integer target conversions as unprovable instead of calling
  `int()` on them.
- Add targeted tests for failed local const, failed top-level const, non-finite
  `f64`, and non-finite refined integer casts.
- Add the new discriminative tests to the quick validation gate.

## Verification

- Focused regressions: `5 passed`
- Nearby proof-carry and proof-gate slice: `28 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `444 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass with no shard retries

Clean-gate counter remains reset to `0`.
