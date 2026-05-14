# Stage 32 Verification Speed - 2026-05-14

Purpose: make future Helix development faster without weakening verification.

## Slice 1 - Machine-Readable Shard Timing

The Stage 31/32 validator now writes `.stage31-logs/pytest-shard-timings.json`
after full parallel pytest runs. The JSON contains:

- schema id: `helix.stage31.pytest_shard_timings.v0`
- generation time
- shard name
- shard duration in seconds
- pytest summary line
- log path

This preserves the existing console output and pass/fail behavior. The file is
for future speed work, especially duration-weighted shard assignment.

## Evidence

- `python -m pytest -q helixc\tests\test_stage31_validate.py`
  - Result: `14 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: `rc=0`
- `python scripts\stage31_validate.py --mode full --shards 2 --skip-snapshot --no-retry-failed`
  - Result: `rc=0`
  - Produced `.stage31-logs/pytest-shard-timings.json`

## Timing Signal From The First JSON Run

With 2 codegen shards and 2 non-codegen shards:

- codegen shard 1: about 12m54s
- codegen shard 2: about 11m37s
- non-codegen shard 1: about 2m50s
- non-codegen shard 2: about 2m17s

That confirms codegen remains the largest verification bottleneck when shard
count is low. The next Stage 32 improvement should use the JSON data to balance
codegen shards by historical duration instead of only stable hashing.

## Slice 2 - Duration-Weighted Shard Assignment

The shard helper now records per-test node durations and can use a prior
duration file to assign collected tests greedily by historical runtime. If no
duration file exists, the helper falls back to stable hash sharding.

The full validator now passes:

- `--weights .stage31-logs/pytest-node-durations.json`
- `--durations-out .stage31-logs/<shard-name>-node-durations.json`

After the parallel group finishes, it merges shard duration files back into
`.stage31-logs/pytest-node-durations.json`. This gives the next run better
balancing data while preserving one-shard-per-test coverage.

Validation after duration-weighted sharding:

- `python -m pytest -q helixc\tests\test_pytest_shard.py helixc\tests\test_stage31_validate.py`
  - Result: `19 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: `rc=0`
- `python scripts\stage31_validate.py --mode full --shards 2 --skip-snapshot --no-retry-failed`
  - Result: `rc=0`
- `bash scripts/run_all_tests.sh`
  - Result: all gates passed
  - Parallel pytest group time: about 4m21s
  - Snapshot smoke: `rc=42`
  - stage0/hex0: `3 passed, 0 failed`

## Slice 3 - Focused-Test Selector

Stage 32 now includes `scripts/stage32_select_tests.py`, a conservative
changed-file to pytest-target selector for the fast edit loop. It does not
replace full gates. It answers: "which small tests should run first for these
changed files?"

Examples:

- Validation tooling changes select the Stage 31 validator, shard helper, and
  selector tests.
- Typecheck changes select `test_typecheck.py` plus the Stage 31 proof and
  refinement regression targets.
- Backend x86 changes select codegen, codegen determinism, and the CLI
  backend-pass regression.
- Docs-only changes produce no pytest targets and recommend `git diff --check`.

The selector can print newline-separated pytest targets for shell use or JSON
for future automation:

- `python scripts/stage32_select_tests.py helixc/frontend/typecheck.py`
- `python scripts/stage32_select_tests.py --json docs/ROADMAP.md`

Validation after the focused-test selector:

- `python -m pytest -q helixc\tests\test_stage32_select_tests.py`
  - Result: `10 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: `rc=0`
- `bash scripts/run_all_tests.sh`
  - Result: all gates passed
  - Parallel pytest group time: about 4m15s
  - All shards passed without retry in the final run
  - Snapshot smoke: `rc=42`
  - stage0/hex0: `3 passed, 0 failed`
