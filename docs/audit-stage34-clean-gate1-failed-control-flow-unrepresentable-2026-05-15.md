# Stage 34 Clean Gate 1 Control-Flow Unrepresentable Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

Fresh auditors on commit `2acc0cc` found another Stage 34 proof-soundness
issue.

Direct unrepresentable scalar values failed correctly, but value-producing
control flow could hide the same bad scalar source. For example:

```hx
type AlwaysF64 = f64 where true;
fn f(b: bool) -> AlwaysF64 {
    if b { 1e309_f64 } else { 0.0_f64 }
}
```

The checker treated the `if` as a non-constant expression, then proved the
self-independent `where true` refinement without preserving the fact that one
branch contained an unrepresentable `f64`. Similar holes existed through
`match`, local `let` indirection, and refined casts fed by such control flow.

## Fix

- Unrepresentable typed scalar source detection now walks value-producing
  syntax, including `if`, `match`, blocks, tuples, arrays, structs, fields,
  indexes, calls, and assignments.
- Local `let` bindings initialized from an unrepresentable source now carry
  that fail-closed evidence into later name references.
- Plain assignments update local unrepresentable-name evidence for the assigned
  name, so repaired values can clear stale local evidence in simple cases.

## Verification

- Focused latest-reset regressions: `2 passed`.
- Stage 34 focused typecheck/CLI/proof-gate slice: `51 passed`.
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`: `490 passed`.
- `python scripts\stage31_validate.py --mode quick`: passed.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  passed across all 12 shards with no retries.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
