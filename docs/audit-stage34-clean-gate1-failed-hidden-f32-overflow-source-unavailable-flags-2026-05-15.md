# Stage 34 Clean Gate 1 Hidden F32 Overflow And Source-Unavailable Flags Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

Fresh auditors on commit `0f8b6ca` found two more Stage 34 issues.

1. A primitive `as f64` cast could hide a prior `f32` overflow in refined
   return checking. `(3.4028235e38_f32 * 2.0_f32) as f64` returned as
   `AlwaysF64` passed even though the inner `f32 * f32` overflowed before the
   value reached the `f64` cast.

2. Source-unavailable proof artifacts still accepted unsafe or impossible
   `input.flags`, including output flags such as `-o`, because the proof-safe
   replay flag whitelist only ran during source-backed recomputation.

## Fix

- Source-expression analysis now detects unrepresentable typed constant
  subexpressions inside primitive casts, so an overflowing inner `f32`
  expression cannot be hidden by a later `as f64`.
- Proof-safe flag validation now runs during structural artifact validation,
  including source-unavailable artifacts. Source-backed recomputation still uses
  the same safe replay whitelist to construct actual checker arguments.

## Verification

- Focused latest-reset regressions: `5 passed`.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: passed.
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`: `484 passed`.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`: passed across all 12 shards with no retries.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
