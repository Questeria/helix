# Stage 35 Clean Gate 1 - Thirty-Third Restart Audit

Date: 2026-05-16
Base commit: `5d8b4c4` (`Fix Stage 35 thirty-second restart findings`)
Result: not clean; fixes applied in the same restart.

## Baseline

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m py_compile helixc\check.py helixc\backend\x86_64.py helixc\tests\test_cli.py helixc\tests\test_codegen.py`
  - Result: passed.
- CLI/codegen/PTX support slices passed before the fresh audit.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,354 tests collected.
- `git diff --check`
  - Result: passed.

## Findings

### Lane A - Runtime and Stdlib

Status: findings present.

- `ti2d_matvec` and `tf2d_matvec` validated matrix shape but not x/y vector
  slices, allowing short vectors to be read or written.
- `ti2d_set` and `tf2d_set` silently accepted invalid coordinates instead of
  returning a failure status.
- `wm_ok` and `ep_ok` validated object-level ticks but not per-entry stored
  timestamps, so corrupted entry recency/tick values remained usable.

### Lane B - CLI and PTX

Status: findings present.

- `--check-only` could be combined with stdout artifact modes or `-o`, causing
  artifact/output requests to be silently ignored.
- Direct PTX parsed warning flags through an unordered set, so repeated
  `-Wad=...` flags could choose the wrong policy.
- Direct PTX accepted conflicting `--stdlib` and `--no-stdlib` flags.

### Lane C - Docs and Continuity

Status: findings present.

- The Stage 35 ledger, restart-32 audit doc, and handoff pointer still described
  restart 32 as needing commit/push after it had already landed as `5d8b4c4`.

## Fix Sweep

- Added matvec x/y slice validation for integer and f32 2D helpers.
- Made integer and f32 2D setters fail loudly with `t2d_error()` on invalid
  coordinates.
- Added per-entry timestamp validation to working and episodic memory guards.
- Rejected conflicting stdlib flags in `helixc.check`.
- Rejected `--check-only` combined with artifact modes or `-o`.
- Reworked direct PTX flag parsing to preserve argument order, honor the last
  warning policy, and reject stdlib conflicts.
- Updated continuation docs to record restart 32 as closed at `5d8b4c4` and
  restart 33 as the current failed/fixed restart.

## Verification

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m py_compile helixc\check.py helixc\backend\ptx.py helixc\tests\test_cli.py helixc\tests\test_ptx.py helixc\tests\test_codegen.py`
  - Result: passed.
- `python -m pytest helixc\tests\test_codegen.py -q -k "2d_matvec_rejects_short_vectors or 2d_setters_return_error or agi_memory_rejects_corrupt_entry_timestamps"`
  - Result: 3 passed, 887 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "conflicting_stdlib or check_only_rejects_artifact"`
  - Result: 3 passed, 190 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "warning_policy_uses_last_flag or conflicting_stdlib"`
  - Result: 2 passed, 76 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "tf2d or ti2d or t2d or matvec or agi_memory"`
  - Result: 30 passed, 860 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35 or check_only or emit_ir or emit_asm or emit_ptx or output or stdlib"`
  - Result: 89 passed, 104 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35 or direct_ptx or wad or deprecated or stdlib"`
  - Result: 42 passed, 36 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35"`
  - Result: 37 passed, 853 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - Result: 48 passed, 145 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - Result: 26 passed, 52 deselected.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,362 tests collected.
- `git diff --check`
  - Result: passed.

## Next Step

Commit and push this restart-33 fix sweep, then begin restart 34 as another
fresh Stage 35 clean-gate attempt from the newest pushed HEAD.
