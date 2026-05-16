# Stage 35 Clean Gate 1 - Thirtieth Restart Failed

Date: 2026-05-16
Base HEAD: `b3f7796` (`Fix Stage 35 twenty-ninth restart findings`)
Result: NOT CLEAN. Stage 35 remains at `0/3` clean gates.

Restart 30 began from the pushed restart-29 fix sweep. A support check first
caught a one-line stdlib guard type regression, then three fresh audit lanes
found additional blockers. The blockers were fixed in this restart, but because
the gate found issues, this restart does not count as a clean gate.

## Audit Findings

### Lane A - AD, Tensor, Runtime, NN

Status: findings present.

- `grad_rev_all` ignored failed reflection writes and returned success when a
  bad base handle caused `modify_f` / `modify_f64` to reject the write.
- `grad` / `grad_rev` / `grad_rev_all` accepted non-floating loss return types
  and generated nonsensical derivative signatures.
- Several tensor reducers/accessors still read beyond validated extents or
  rejected valid interior slices.
- NN classifier and metric helpers missed short-buffer checks for bias,
  inputs, targets, and output buffers.
- Episodic memory accessors accepted negative indices.
- Reverse-AD could still use mutated tape payload values after adjoints were
  allocated.

### Lane B - Backend, PTX, CLI

Status: findings present.

- The direct x86 backend leaked Python tracebacks for missing inputs,
  duplicate impl flattening, and missing output directories.
- The direct x86 backend silently ignored unknown flags, so typoed safety flags
  such as `--strcit` could still emit binaries.

### Lane C - Docs and Public Claims

Status: findings present.

- Continuation pointers still described restart 30 as the next step even after
  restart 30 had begun.
- Historical handoff docs pointed readers at live HEAD without warning that the
  active restart had local dirty fixes.
- Quickstart public evidence still cited restart 28's older collection count.

## Fix Sweep

- `grad_rev_all` now accumulates reflection-write status and returns `-1` if any
  gradient cell write is rejected.
- AD generation now rejects non-floating loss outputs before lowering.
- 1D tensor ops now use slice-aware validation consistently, and previously raw
  reducers/accessors fail closed instead of reading footers or later
  allocations.
- NN classifier, argmax/accuracy/CE, and metric helpers gained slice-aware
  checks for bias, input, output, and target buffers.
- Working/episodic memory helpers gained bounded object validation, and
  episodic indexed access rejects negative indices.
- Reverse-AD adjoint guards now include a digest of the tape payload snapshot,
  so tape mutation after adjoint allocation invalidates the backward pass.
- Direct x86 CLI now rejects unknown flags and reports input, impl, codegen,
  output, and chmod failures as clean `error:` diagnostics without tracebacks.
- Current docs now identify restart 30 as the active restart and cite the live
  2,339-test collection count.

## Verification So Far

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m py_compile helixc\frontend\grad_pass.py helixc\backend\x86_64.py helixc\tests\test_codegen.py helixc\tests\test_cli.py`
  - Result: passed.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35_grad_rev_all_reports_failed_reflection_write or stage35_grad_rejects_nonfloating_loss_return or stage35_grad_rev_all_rejects_nonfloating_loss_return or stage35_tf1d_add_accepts_valid_interior_slices or stage35_tensor_reducers_do_not_read_footers or stage35_dense_classifier_rejects_short_bias_vector or stage35_nn_classifier_helpers_reject_short_outputs_and_targets or stage35_nn_metrics_reject_short_inputs or stage35_episodic_accessors_reject_negative_indices or stage35_revad_backward_rejects_tape_value_mutated_after_adjoints"`
  - Result: 10 passed, 870 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "direct_x86_rejects_unknown_flags or direct_x86_missing_input or direct_x86_duplicate_impl or direct_x86_missing_output_dir"`
  - Result: 4 passed, 178 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "t1d or tf1d or ti1d or dense_classifier or argmax_rows or accuracy_count or ce_loss_batch or mae_loss or count_correct or revad or grad_rev_all"`
  - Result: 104 passed, 776 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35 or wad or deprecated or direct_x86"`
  - Result: 39 passed, 143 deselected.
- `python -m pytest helixc\tests\test_autodiff.py helixc\tests\test_autodiff_reverse.py helixc\tests\test_transcendentals.py -q`
  - Result: 103 passed.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35 or direct_ptx or wad or deprecated"`
  - Result: 40 passed, 36 deselected.
- `python -m pytest helixc\tests\test_reflection.py helixc\tests\test_effect_check.py -q`
  - Result: 50 passed.
- `python -m pytest helixc\tests --collect-only -q -p no:cacheprovider`
  - Result: 2,339 tests collected.
- `git diff --check`
  - Result: passed.
- `python -m pytest helixc\tests\test_codegen.py -q`
  - Result: timed out after 20 minutes with no useful partial output; stale
    pytest process was stopped.
- Four-way collected `test_codegen.py` chunk run
  - Result: timed out after 15 minutes per chunk in this environment; no
    chunk-specific failure was surfaced, and no stale pytest process remained.

## Next Step

Commit the restart-30 fix sweep, then begin restart 31 from the new HEAD as
another clean-gate attempt. For speed, prefer the targeted/wide slices above
over the full monolithic `test_codegen.py` command unless the environment can
run it with much longer timeouts.
