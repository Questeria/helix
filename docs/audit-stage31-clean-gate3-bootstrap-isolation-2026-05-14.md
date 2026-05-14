# Stage 31 Clean Gate 3 - Bootstrap Temp Isolation

Result: CLEAN

Scope:
- `helixc/tests/test_codegen.py::test_bootstrap_kovc_full_pipeline_arithmetic`

Checks performed:
- Reviewed the bootstrap arithmetic harness changes.
- Confirmed the cached Helix bootstrap driver now uses stable per-worktree WSL temp paths instead of global `/tmp/helix_src_in.hx` and `/tmp/helix_bin_out.bin`.
- Confirmed the stable temp paths are part of the bootstrap cache key.
- Confirmed cached bootstrap writes are atomic through a per-process temp file and `os.replace`.
- Confirmed each compiled-source execution takes a WSL lock before touching shared temp paths and reports a clear timeout sentinel if the lock cannot be acquired.

Validation evidence:
- `python -m pytest -q helixc\tests\test_codegen.py::test_bootstrap_kovc_full_pipeline_arithmetic`
  - First post-lock run: `1 passed`
  - Second post-lock run: `1 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot`
  - Result: all codegen shards returned `rc=0`.

Findings:
- No blocking findings.

Residual risk:
- If a process is killed while holding the WSL lock directory, a later run can wait up to 60 seconds and fail with `__HARNESS_FAIL_BOOTSTRAP_LOCK_TIMEOUT__`. That is preferable to a false green or missing-output race, and the failure mode is explicit.
