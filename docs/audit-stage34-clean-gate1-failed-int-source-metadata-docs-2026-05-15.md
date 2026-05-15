# Stage 34 Clean Gate 1 Int Source Metadata Docs Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

Fresh auditors on commit `5f422b5` found three more Stage 34 issues.

1. Suffixed integer literals that were not representable in their own source
   type could still be cast to a wider refined target and recorded as proved.
   For example, `2147483648_i32 as PositiveI64` passed even though the source
   literal cannot be represented as `i32`.

2. Source-unavailable artifacts still accepted impossible input metadata:
   `input.opt_level` outside the checker's `0..3` range, unknown warning names
   or policies, and fake stdlib manifests with self-consistent hashes.

3. The Stage 34 progress document still had two stale affine proof-carry
   sentences that sounded like current accepted behavior, even though affine
   proof carries now fail closed for fixed-width numeric bases.

## Fix

- Constant integer literal evaluation now applies the literal suffix, or the
  source numeric base when there is no suffix, before refined proof checking.
  Unrepresentable source literals now force the same fail-closed path as
  overflowing typed arithmetic.
- Proof artifact structural validation now rejects out-of-range opt levels,
  unknown warning metadata, invalid warning policies, and source-unavailable
  artifacts that claim stdlib participation.
- The Stage 34 progress document now states that old affine examples are
  design intent only and that current affine coverage is fail-closed typecheck
  coverage, not accepted proof-carry artifact coverage.

## Verification

- Focused latest-reset regressions: `3 passed`.
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`: `344 passed`.
- `python scripts\stage31_validate.py --mode quick`: passed.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  passed across all 12 shards with no retries.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
