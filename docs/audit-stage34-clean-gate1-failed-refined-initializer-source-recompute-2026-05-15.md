# Stage 34 Clean Gate 1 Refined Initializer And Source Recompute Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

Fresh auditors on commit `2ebac36` found two more Stage 34 proof-honesty
failures.

1. Refined initializer checks evaluated the initializer expression with generic
   Python scalar semantics instead of the declared source machine type. This let
   `-1_i32 % 2_i32` pass a positive refinement because Python modulo produced
   `1`, while Helix integer modulo truncates through the signed machine value
   and produces `-1`. The same path could also miss per-operation `f32`
   rounding, accepting precision-boundary initializer values that the Helix
   runtime would not produce.

2. Plain `proof_artifact_validate.py --source` recomputed carried-proof
   metadata, but it did not compare the full proof-relevant artifact content.
   A forged source-backed artifact could promote an unproved obligation to
   `proved` and erase typecheck errors while still passing non-strict
   validation.

## Fix

- Refined initializer and refined cast checks now evaluate constant source
  expressions through the erased source numeric type before casting to the
  refined target. That gives initializer checks Helix-style signed division and
  modulo semantics, fixed-width integer range checks, and per-operation `f32`
  rounding.
- Constant refinement checks retain a raw-scalar fallback for diagnostics when
  machine-typed evaluation cannot produce a representable value.
- Source-backed proof artifact validation now recomputes and compares the full
  proof-relevant artifact surface: `summary`, `obligations`, `proof_carries`,
  `pipeline_errors`, `typecheck_errors`, and `warning_diagnostics`.
- Strict clean validation now still reports clean-policy problems even when
  source recomputation has already found mismatches.
- The Stage 31 quick gate now includes the refined-initializer machine-semantics
  regression and the default source-backed forged-clean-artifact rejection.

## Verification

- Focused refined-initializer and source-recompute regressions: `6 passed`.
- Follow-up strict-mode and representability regressions: `8 passed`.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: passed.
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`: `476 passed`.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`: passed after the built-in retry recovered one no-codegen shard.
- `python -m pytest -q helixc/tests/test_transcendentals.py::test_grad_through_user_defined_function_call`: passed after inspecting the recovered shard's transient failure.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
