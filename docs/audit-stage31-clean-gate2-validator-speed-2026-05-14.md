# Stage 31 Clean Gate 2 - Validator Speed And Coverage

Result: CLEAN

Scope:
- `scripts/stage31_validate.py`
- `helixc/tests/test_stage31_validate.py`
- `.stage31-logs/pytest-no-codegen-shard-*.log`
- `.stage31-logs/pytest-codegen-shard-*.log`

Checks performed:
- Reviewed full-mode validation changes.
- Confirmed the broad non-codegen suite is still collected by pytest and filtered only by stable node-id sharding.
- Confirmed `test_codegen.py` remains excluded from non-codegen shards and included in codegen shards.
- Confirmed manual shard bounds remain capped by existing `MAX_SHARDS`.
- Confirmed the validator unit test asserts non-codegen sharding commands and codegen shard commands are both emitted.

Validation evidence:
- `python -m pytest -q helixc\tests\test_stage31_validate.py`
  - Result: included in the focused `14 passed` run.
- `python scripts\stage31_validate.py --mode full --skip-snapshot`
  - Result: all 4 non-codegen shards and all 8 codegen shards returned `rc=0`.
- `git diff --check`
  - Result: no whitespace errors.

Findings:
- No blocking findings.

Residual risk:
- The full gate is still sensitive to machine load because codegen tests execute many WSL binaries. The next speedup should add slow-test telemetry and better codegen shard balancing, not remove coverage.
