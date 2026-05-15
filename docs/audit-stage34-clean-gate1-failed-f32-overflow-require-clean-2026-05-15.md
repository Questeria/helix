# Stage 34 Clean Gate 1 F32 Overflow And Require-Clean Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

Fresh auditors on commit `c16afeb` found two more Stage 34 issues.

1. Refined cast checking could fall back from failed typed `f32` constant
   evaluation to raw Python double arithmetic. This let
   `(3.4028235e38_f32 * 2.0_f32) as AlwaysF64` prove cleanly even though the
   `f32 * f32` operation overflows before the value can be refined as `f64`.

2. `proof_artifact_validate.py --require-clean` ignored the embedded source
   path that default validation had already resolved. A valid clean artifact
   with `path: "input.hx"` passed normal validation but failed strict
   validation unless `--source input.hx` was supplied redundantly.

## Fix

- Refinement checks now distinguish a truly unknown constant from a known source
  value that failed representation under its typed source base. Raw fallback may
  force a rejection for diagnostics, but cannot rescue a failed typed `f32`
  computation into a valid `f64` proof value.
- `--require-clean` now reuses the same resolved source path used by default
  source-backed recomputation, while still failing closed when no explicit or
  embedded source path exists.

## Verification

- Focused latest-reset regressions: `5 passed`.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: passed.
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`: `482 passed`.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`: passed across all 12 shards with no retries.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
