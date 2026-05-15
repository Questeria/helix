# Stage 34 Clean Gate 1 Call Argument And Progress Order Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

Fresh auditors on commit `2cc20ba` found one proof-soundness issue and one
audit-trail issue.

The proof-soundness issue was an unrepresentable primitive argument flowing
through a function that returns a refined value:

```hx
type AlwaysF64 = f64 where true;
fn accept(x: f64) -> AlwaysF64 { x }
fn f() -> AlwaysF64 { accept(1e309_f64) }
```

The call boundary accepted `1e309_f64` as a primitive `f64` argument. The
callee then proved `where true` for parameter `x`, and the proof artifact gate
accepted the result as clean.

The audit-trail issue was that the Twentieth restart progress entry had been
inserted before older restart sections, so the progress document was no longer
chronological.

## Fix

- Call checking now rejects unrepresentable typed scalar evidence at compatible
  primitive numeric parameters when the callee can return a refined value.
- The new rule is intentionally conservative until Helix has interprocedural
  dependency tracking for which parameters can influence a refined return.
- Regression tests now pin the direct call-argument false pass and the same
  shape through value-producing control flow.
- The proof artifact gate now rejects the direct call-argument false pass.
- The quick validation list now includes the new call-argument regression.
- The Stage 34 progress file now records the Nineteenth, Twentieth, and Twenty
  First restarts in chronological order.

## Verification

- Focused latest-reset regressions: `5 passed`.
- Stage 34 focused typecheck/CLI/proof-gate slice: `54 passed`.
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`: `495 passed`.
- `python scripts\stage31_validate.py --mode quick`: passed.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  passed after built-in retry recovered no-codegen shards 1 and 2.
- `python -m pytest -q helixc/tests/test_strings_io.py::test_read_file_int_round_trips helixc/tests/test_strings_io.py::test_read_file_int_missing_file_returns_zero`:
  `2 passed` after inspecting the recovered shard failures.
- Direct checks for the proof-artifact auditor's archived-copy full-gate
  concerns:
  - `python -m pytest -q -p no:cacheprovider helixc/tests/test_stage31_validate.py::test_run_all_tests_rejects_excessive_manual_shards_before_gates helixc/tests/test_stage31_validate.py::test_run_all_tests_rejects_zero_padded_excessive_manual_shards_before_gates`: `2 passed`.
  - `python -m pytest -q -p no:cacheprovider helixc/tests/test_strings_io.py::test_print_str_writes_to_stdout helixc/tests/test_strings_io.py::test_print_int_negative helixc/tests/test_strings_io.py::test_read_file_int_endianness_is_little_endian`: `3 passed`.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
