# Stage 34 Clean Gate 1 Failed Audit - Bound Implication

This audit attempt did not count as a clean gate. Fresh Stage 34 auditors found
three proof-honesty issues after commit `935af17`.

## Findings

1. Numeric-bound proof-carry implication ignored explicit float literal
   representation.

   Minimal shape:

   ```hx
   type Source = f32 where self >= 16777217.0_f32;
   type Target = f32 where self > 16777216.0_f32;
   fn make() -> Source { 16777216.0_f32 }
   fn bad(s: Source) -> Target { s }
   fn main() -> i32 { let t: Target = bad(make()); 0 }
   ```

   Observed before the fix: proof artifact and proof gate accepted a clean
   `numeric-bound-implication` carry. This is false because `16777217.0_f32`
   represents `16777216.0`.

2. Top-level const scalar caching stored raw literal values before applying the
   declared type.

   Minimal shape:

   ```hx
   const LIMIT: f32 = 16777217.0_f32;
   type BelowLimit = f32 where self < LIMIT;
   fn f() -> BelowLimit { 16777216.0_f32 }
   ```

   Observed before the fix: `LIMIT` behaved like raw `16777217.0`, not the
   represented `f32` value `16777216.0`.

3. The fixed-point function-body loop reset unbound-name suppression but not
   unknown-type suppression.

   Minimal shape:

   ```hx
   type AlwaysI32 = i32 where true;
   fn bad() -> AlwaysI32 {
       let x: Missing = 0;
       1e309_f64 as AlwaysI32
   }
   ```

   Observed before the fix: final diagnostics could lose
   `unknown type 'Missing'`.

## Fix Summary

- Numeric-bound and affine bound extraction now evaluate predicate constants
  with explicit float suffix representation.
- Top-level scalar const indexing stores values after applying the declared
  primitive or refined base type.
- The fixed-point function-body loop resets unknown-type suppression as well
  as unbound-name suppression.
- Added discriminative typecheck, CLI proof-artifact, proof-gate, and quick-gate
  coverage for these cases.

## Verification

- Exact focused regressions: `6 passed`
- Nearby proof-carry and proof-gate slices: `32 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `457 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass after built-in retry recovered no-codegen shard 1

Clean-gate counter remains reset to `0`.
