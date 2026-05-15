# Stage 34 Clean Gate 1 Arithmetic, Safe Replay, And Numbering Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

Fresh auditors on commit `eaff977` found three more issues.

1. Constant scalar evaluation still collapsed nonfinite arithmetic to unknown.
   That let `(1e309_f64 + 0.0_f64) as AlwaysF64` pass a `where true`
   refinement and produce clean proof artifacts, even though the value is not a
   representable `f64`.

2. Source-backed validation replayed artifact-controlled `input.flags` directly
   into `helixc.check`. A forged artifact could use flags such as `-o` and
   cause validation to write an output file before validation failed.

3. The Stage 34 progress file still had a duplicate unnumbered `Clean Gate 1
   Restart` after `Second Restart`, then continued with `Third Restart`. That
   made the restart sequence internally inconsistent.

A related validator correctness issue was also fixed: a valid artifact produced
from a relative embedded source path could fail source-backed default validation
because the recomputed absolute path was compared byte-for-byte against the
original relative path.

## Fix

- Raw constant scalar fallback now evaluates literal arithmetic without
  collapsing nonfinite results to unknown before refinement cast diagnostics.
- Source-backed proof validation now reconstructs replay arguments from a
  proof-safe whitelist instead of directly appending artifact-controlled flags.
- Relative and absolute source paths are compared semantically during
  source-backed recomputation.
- The progress file restart headings were renumbered so the sequence is
  chronological through the latest reset.

## Verification

- Focused latest-reset regressions: `5 passed`.
- Focused proof artifact validator/gate files: `65 passed`.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: passed.
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`: `480 passed`.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`: passed across all 12 shards with no retries.
- `git diff --check`: passed.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
