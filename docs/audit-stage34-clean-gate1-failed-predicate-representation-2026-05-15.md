# Stage 34 Clean Gate 1 Predicate Representation Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

The clean-gate follow-up found three proof-soundness gaps after the previous
bound-implication hardening.

1. Unsuffixed float predicate literals were evaluated as raw Python `float`
   values inside refinement predicates, even though Helix expression semantics
   default unsuffixed float literals to `f32`. This let a predicate such as
   `self < 16777217.0` be treated as a strict raw bound instead of the
   representable `f32` value.

2. Predicate arithmetic could create non-finite values after literal validation.
   Expressions such as `(1e308_f64 * 10.0_f64)` could become `inf`, and compound
   arithmetic could produce `nan`; these values must fail closed instead of
   becoming proof constants or bound-carry inputs.

3. Plain `proof_artifact_validate.py --source` trusted a forged artifact `path`
   unless `--require-clean` was also used. The artifact validator should reject
   a source/path mismatch whenever an explicit source is supplied.

## Fix

- Treat unsuffixed predicate float literals as represented `f32` values.
- Reject non-finite scalar results from predicate unary and binary arithmetic.
- Require proof artifact paths to match the provided `--source` path in normal
  validation mode as well as clean recomputation mode.
- Add typecheck, proof-gate, artifact-validator, and quick-gate regressions for
  the exact failure patterns.

## Verification

- Focused predicate and validator regressions: passed.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: passed.
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`: `463 passed`.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`: passed with no shard retries.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
