# Stage 34 Clean Gate 1 Cast And Input Cache Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

Fresh auditors on commit `8ddb14f` found two Stage 34 proof-honesty failures
and one progress-document issue.

1. Source-backed default validation recomputed the proof body, but did not
   compare recomputed `input` or `cache_key`. A forged artifact could advertise
   a different proof input and internally consistent cache key while still
   passing default validation.

2. Constant scalar evaluation did not model nested `as` casts. This let
   `(1e309_f64 as f64) as AlwaysF64` hide the same nonfinite representability
   failure that direct `1e309_f64 as AlwaysF64` already rejected.

3. The Stage 34 progress file listed recent restart sections out of order. The
   sections now read chronologically.

## Fix

- Source-backed default validation now compares `schema`, `cache_key`, `path`,
  and `input` in addition to the proof body.
- Constant scalar evaluation now evaluates simple primitive casts and uses the
  source expression's inferred primitive base where possible, so signed
  division/modulo and float rounding do not disappear inside nested casts.
- Raw-scalar fallback now preserves known unrepresentable cast sources for
  diagnostics instead of reducing them to unknown values.
- The quick gate includes the forged-input/cache regression through
  `test_validate_rejects_forged_input_and_cache_with_source_by_default`; the
  hidden nonfinite-cast regression is covered by the existing Stage 34
  unrepresentable-values quick target and the full proof-gate file.

## Verification

- Focused latest-reset regressions: `7 passed`.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: passed.
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`: `478 passed`.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`: passed after the built-in retry recovered no-codegen shards 1 and 2.
- `python -m pytest -q helixc/tests/test_strings_io.py::test_print_str_multiple_calls helixc/tests/test_strings_io.py::test_print_int_decimal_output`: passed after inspecting the recovered shard's transient failures.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
