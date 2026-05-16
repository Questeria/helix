# Stage 35 Clean Gate 1 - Thirty-First Restart Failed

Date: 2026-05-16
Base HEAD: `9efee28` (`Fix Stage 35 thirtieth restart findings`)
Result: NOT CLEAN. Stage 35 remains at `0/3` clean gates.

Restart 31 began from the pushed restart-30 fix sweep. Support checks were
green, then three fresh audit lanes found additional blockers in reverse-AD
tape integrity, f32 reducers, NN argmax, AGI memory object validation, direct
x86 error handling, artifact stdout discipline, and continuation docs. The
blockers were fixed in this restart, but because the gate found issues, this
restart does not count as a clean gate.

## Audit Findings

### Lane A - AD, Tensor, Runtime, NN

Status: findings present.

- Reverse-AD tape validation could still be bypassed by changing two payload
  values after adjoint allocation while preserving the old rolling digest.
- f32 reducers `tf1d_max`, `tf1d_min`, `tf1d_argmax`, and `tf1d_argmin` did
  not consistently reject inflated logical lengths.
- Public NN `argmax` read unvalidated slices while sibling helpers had already
  gained guard checks.
- Working-memory and episodic-memory validators accepted forged tensor buffers
  that happened to contain plausible counters.

### Lane B - Backend, PTX, CLI

Status: findings present.

- The direct x86 backend leaked Python tracebacks for invalid UTF-8 input.
- The direct x86 backend leaked Python tracebacks when strict stdlib merging
  raised `FileNotFoundError`.
- `helixc.check --emit-ir` and `--emit-asm` still allowed warning summaries to
  share stdout with emitted artifacts in warn mode.

### Lane C - Docs and Public Claims

Status: findings present.

- Continuation docs still pointed at restart 30 as active after restart 30 had
  been committed and restart 31 had begun.
- Public docs cited the restart-30 live collection count instead of directing
  readers to refresh the current scoped count.

## Fix Sweep

- Reverse-AD adjoint allocation now snapshots tape payload slots next to the
  adjoint metadata; `rev_adj_cap` compares the live tape against the snapshot
  before seeds, gradients, or backward propagation can use it.
- Attempts to append to a reverse-AD tape after adjoint allocation now poison
  the tape footer so later backward passes fail closed.
- f32 reducers and public NN `argmax` now use `t1d_slice_ok` before reading.
- Working-memory and episodic-memory allocations now carry magic and footer
  guards, so forged tensor buffers fail `wm_ok` / `ep_ok`.
- Direct x86 CLI now reports invalid UTF-8 and strict-missing-stdlib failures
  as clean `error:` diagnostics without tracebacks.
- `--emit-ir` and `--emit-asm` now route AD/deprecated warning summaries to
  stderr so stdout remains artifact-only.
- Public continuation docs now describe restart 30 as closed and restart 31 as
  the active fix sweep.

## Verification

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m py_compile helixc\check.py helixc\backend\x86_64.py helixc\frontend\grad_pass.py helixc\tests\test_cli.py helixc\tests\test_codegen.py`
  - Result: passed.
- `python -m pytest helixc\tests\test_cli.py -q -k "wad_warn_emit_ir or deprecated_warn_emit_asm or invalid_utf8 or missing_strict_stdlib"`
  - Result: 4 passed, 182 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "digest_collision or tf1d_reducers_reject_short_input or nn_argmax_rejects_short_input or agi_memory_rejects_forged_tensor_objects"`
  - Result: 4 passed, 880 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "tape_grown_after_adjoints or digest_collision or tape_value_mutated_after_adjoints"`
  - Result: 3 passed, 881 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "revad or grad_rev_all or autodiff_reverse or tf1d or t1d or dense_classifier or argmax_rows or accuracy_count or ce_loss_batch or mae_loss or count_correct or agi_memory"`
  - Result: 89 passed, 795 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35 or wad or deprecated or direct_x86"`
  - Result: 43 passed, 143 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35 or direct_ptx or wad or deprecated"`
  - Result: 40 passed, 36 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "autodiff or autodiff_reverse or transcendentals"`
  - Result: 14 passed, 870 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "reflection or effect"`
  - Result: 3 passed, 881 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35"`
  - Result: 31 passed, 853 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - Result: 41 passed, 145 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - Result: 24 passed, 52 deselected.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,347 tests collected.
- `git diff --check`
  - Result: passed.
- Unscoped `python -m pytest --collect-only -q`
  - Result: failed because it also collected `HELIX_STAGE30_COMPILER_SNAPSHOT`
    and hit duplicate pytest module names. This is a command-scope issue; use
    the scoped live-suite collection above for Stage 35.

## Next Step

Restart 31 was committed and pushed as `fb9400d`. Restart 32 began from that
pushed HEAD and found additional issues, so continue from the Stage 35 progress
ledger and live git state rather than this historical next-step note.
