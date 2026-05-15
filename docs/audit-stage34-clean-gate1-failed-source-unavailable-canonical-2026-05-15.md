# Stage 34 Clean Gate 1 Source-Unavailable Canonical Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

Fresh auditors on commit `2d91f4e` found one more validator trust issue.

Source-unavailable proof artifacts still accepted unverifiable diagnostic and
input metadata:

- `warning_diagnostics` could be forged even though there is no source hash or
  cache key to support the warning payload.
- `input.flags` could be duplicated or non-canonical, even though real proof
  artifacts are emitted with sorted and deduplicated flags.

A proof-soundness auditor did not find a new scalar/refinement false-clean in
its temp repro matrix, but did catch the quick gate failing on stale assertions
after the stricter validator behavior.

## Fix

- Proof replay flags are now required to match the compiler's canonical
  sorted-and-deduplicated metadata form.
- Source-unavailable artifacts must now have empty `pipeline_errors` and
  `warning_diagnostics`, matching the existing empty obligations, proof carries,
  and typecheck errors requirements.
- Tests now assert the stricter structural failures instead of expecting older
  clean-policy or cache-mismatch diagnostics to fire first.

## Verification

- Focused latest-reset regressions: `4 passed`.
- `python -m pytest -q helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`: `74 passed`.
- `python scripts\stage31_validate.py --mode quick`: passed.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  passed after built-in retry recovered no-codegen shard 1.
- `python -m pytest -q helixc/tests/test_strings_io.py::test_print_str_writes_to_stdout`: passed after inspecting the recovered shard's transient failure.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
