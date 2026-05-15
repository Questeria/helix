# Stage 34 Clean Gate 1 Float Affine And Erased Carry Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

Fresh auditors on commit `3d20693` found two more Stage 34 proof-honesty
failures.

1. Float affine proof-carry extraction still treated target predicates as exact
   real-number algebra. This allowed values such as `16777216.0_f32` and
   `9007199254740992.0_f64` to satisfy `self + 1 > self` style target
   predicates, even though IEEE float arithmetic rounds those additions back to
   the original value.

2. Normal `proof_artifact_validate.py --source` rejected missing carried-proof
   metadata, but it still accepted an artifact where `proof_carries` was erased
   and the carry summary was updated to match the erased list. That made the
   artifact internally consistent while losing proof provenance from the real
   source.

## Fix

- Numeric-bound proof carry now passes the erased numeric base into target
  requirement extraction.
- Affine proof-carry extraction fails closed for floating-point bases. Direct
  float bounds such as `self >= 0.0` remain supported; algebraic float
  rearrangement is deferred until it can be proved with float-aware semantics.
- Integer affine proof-carry coverage remains active and was retargeted to
  `i32` regressions.
- Plain source-backed artifact validation now recomputes and compares carried
  proof metadata (`proof_carries`, `summary.proof_carries`, and
  `summary.proof_carry_strategies`) even when `--require-clean` is not used.

## Verification

- Focused float-affine and erased-carry regressions: `7 passed`.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: passed.
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`: `470 passed`.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`: passed across all 12 shards.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
