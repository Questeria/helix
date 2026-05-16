# Stage 35 Clean Gate 1 - Thirty-Fourth Restart Audit

Date: 2026-05-16
Base commit: `09b692c` (`Fix Stage 35 thirty-third restart findings`)
Result: not clean; fixes applied in the same restart.

## Baseline

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m py_compile helixc\check.py helixc\backend\x86_64.py helixc\backend\ptx.py helixc\tests\test_cli.py helixc\tests\test_codegen.py helixc\tests\test_ptx.py`
  - Result: passed.
- CLI/codegen/PTX support slices passed before the fresh audit.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,362 tests collected.
- `git diff --check`
  - Result: passed.

## Findings

### Lane A - Runtime and Stdlib

Status: findings present.

- AGI world-model tables could be constructed with invalid dimensions and then
  write into adjacent arena objects.
- AGI BFS, visited-set, priority-queue, beam, A*, and attention helpers lacked
  enough handle/slice validation for forged or short buffers.
- Failed deep unification could leave stale variable bindings.
- Prediction-error helpers could overflow into negative or wrapped metrics.

### Lane B - CLI and Backend

Status: findings present.

- `helixc.check` could print AD warning summaries to stdout when a different
  warning class, such as deprecated, was promoted to error.
- Direct x86 rejected explicit `--stdlib`, unlike `helixc.check` and direct
  PTX.
- `helixc.check -o` and direct x86 wrote output artifacts non-atomically and
  could leave partial temp/final artifacts on write or chmod failure.

### Lane C - Docs and Public Claims

Status: findings present.

- Current progress docs still described restart 33 as needing commit/push after
  it had already landed as `09b692c`.
- Public count surfaces still cited older live-suite collection counts.

## Fix Sweep

- Added validated world-model table headers/footers and checked transition
  offsets.
- Added container guards for BFS, visited-set, and priority queue.
- Added slice validation for hill-climb, beam search, A*, and attention helpers.
- Added unification rollback on failed recursive matches.
- Saturated prediction-error arithmetic.
- Kept AD warning summaries off stdout during any warning-error failure.
- Added direct x86 `--stdlib` compatibility and stdlib-conflict rejection.
- Switched `helixc.check -o` and direct x86 output writes to temp-file plus
  replace with cleanup on failure.
- Updated current docs and public stats to restart 34 and 2,372 collected tests.

## Verification

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m py_compile helixc\check.py helixc\backend\x86_64.py helixc\tests\test_cli.py helixc\tests\test_codegen.py`
  - Result: passed.
- `python -m pytest helixc\tests\test_codegen.py -q -k "wmt_rejects_invalid or prediction_error_saturates or search_rejects_forged or attention_rejects_short or unify_deep_failures_rewind"`
  - Result: 5 passed, 890 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "deprecated_error_with_ad_warning or atomic_replace_failure or handles_oserror_on_write or direct_x86_accepts_stdlib or direct_x86_rejects_conflicting_stdlib or direct_x86_chmod_failure"`
  - Result: 6 passed, 192 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "agi_ or wmt or wml or wm_prediction or bfs or visited or pq or beam or astar or attention or unify"`
  - Result: 59 passed, 836 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35 or output or direct_x86 or deprecated or wad or check_only or stdlib"`
  - Result: 81 passed, 117 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35 or direct_ptx or wad or deprecated or stdlib"`
  - Result: 42 passed, 36 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35"`
  - Result: 42 passed, 853 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - Result: 53 passed, 145 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - Result: 26 passed, 52 deselected.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,372 tests collected.
- `git diff --check`
  - Result: passed.

## Next Step

After this restart-34 fix sweep lands, begin restart 35 as another fresh Stage
35 clean-gate attempt from the newest pushed HEAD.
