# Stage 34 Clean Gate 1 F32 Arithmetic And Artifact Metadata Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

Two clean-gate auditors found additional proof-honesty issues after commit
`1487810`.

1. `f32` predicate arithmetic rounded float literal leaves but did not round the
   arithmetic result back through `f32`. This allowed
   `self + 1.0_f32 > 16777216.0_f32` to prove true for
   `self = 16777216.0_f32`, even though real `f32` arithmetic rounds the left
   side back to `16777216.0`.

2. Normal proof artifact validation with `--source` still accepted source-backed
   artifacts with `path: null` and accepted artifacts where `proof_carries`,
   `summary.proof_carries`, and `summary.proof_carry_strategies` had been
   stripped. `--require-clean` caught the forgery by recomputing, but plain
   validation should fail closed too.

## Fix

- Predicate evaluation now carries the erased numeric base into scalar
  expression evaluation.
- `f32` predicate arithmetic rounds each unary/binary arithmetic result through
  the same `f32` representation used by values.
- `f32` numeric-bound proof extraction avoids affine real-number algebra and
  keeps only simple direct `self` versus constant bounds.
- Top-level and local scalar const indexing use represented values after the
  declared scalar type is applied.
- Proof artifact validation now requires source-backed artifacts to keep a
  string `path`, the top-level `proof_carries` field, and carry summary fields
  even when the carry list is empty.

## Verification

- Focused regressions for the two auditor findings: `6 passed`.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: passed.
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`: `467 passed`.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`: passed across all 12 shards.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
