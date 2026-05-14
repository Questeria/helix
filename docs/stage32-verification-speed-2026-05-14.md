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
