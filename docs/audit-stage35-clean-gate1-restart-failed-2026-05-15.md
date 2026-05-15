# Stage 35 Clean Gate 1 Restart - Failed And Fixed

Date: 2026-05-15

Result: failed clean gate restart, fixed, reset to 0/3 clean gates.

After commit `8bfe30d`, a fresh Stage 35 clean-gate restart found additional
coverage and behavior issues. This restart therefore did not count as clean.

## Findings Fixed

1. `grad` lacked the same aggregate-parameter regression coverage already added
   for `grad_rev` and `grad_rev_all`.
2. Public `--emit-ptx` could route the new Tile IR fail-closed exception as an
   internal compiler-bug-looking path unless the Tile IR error was handled in
   the PTX branch.
3. The autotune docstring still implied compile-time variant generation already
   exists, when Phase-0 currently records and validates the static sweep spec.
4. `ce_loss_batch_f32` averaged its invalid-label sentinel over `rows`.
5. Scalar `__adam_step` still had the zero-denominator edge fixed in the array
   Adam helper.
6. `dense_classifier_sgd_step_f32` briefly fixed clobbering by allocating an
   internal temporary every call, which avoided clobbering but leaked arena cells
   in repeated training.
7. `softmax_ce_grad_f32` could mutate earlier rows before discovering a later
   invalid label.

## Fix Summary

- Added the missing `grad(...)` aggregate-parameter regression.
- Filtered `--emit-ptx` Tile IR lowering to kernel functions and kept Tile IR
  lowering errors inside the PTX error path.
- Reworded the autotune docstring so current support is static spec validation;
  variant generation and runtime timing are explicitly future design.
- Made batch CE return the full sentinel for any invalid label.
- Made scalar `__adam_step` clamp zero or negative denominators.
- Changed `dense_classifier_sgd_step_f32` to reuse caller scratch while applying
  weight updates directly, avoiding the large weight-gradient scratch matrix.
- Made `softmax_ce_grad_f32` prevalidate all labels before writing gradients.

## Verification

- `python -m pytest -q helixc\tests\test_codegen.py -k "grad_rejects_aggregate_param or ce_loss_batch_f32_invalid_label_not_averaged_down or softmax_ce_grad_f32_invalid_batch_does_not_partially_mutate or dense_classifier_sgd_step_f32_reuses_scratch_without_arena_growth or dense_classifier_sgd_step_f32_does_not_clobber_small_scratch or builtin_adam_step_zero_denom_returns_zero or adam_f32_step_zero_grad_zero_eps_keeps_weight" --tb=short`
  - Result: 7 passed.
- `python -m pytest -q helixc\tests\test_cli.py -k "stage35_emit_ptx_reports_tile_lowering_error_without_bug_label or c117_emit_ptx_uses_kernel_attrs or c119_emit_ptx_rejects_no_kernel_modules or c119_emit_ptx_allows_folded_bool_constants or c119_emit_ptx_accepts_kernel_index_builtin or c119_emit_ptx_rejects_extern_only_kernels" --tb=short`
  - Result: 6 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or stage35 or softmax or ce_loss or dense_classifier_sgd_step_f32 or adam_f32_step or builtin_adam_step_zero_denom_returns_zero or grad_rejects_aggregate_param or grad_rev_rejects_aggregate_param or grad_rev_all_rejects_aggregate_param or builtin_bce_uses_stable_log_near_zero" --tb=short`
  - Result: 55 passed.
- `python -m pytest -q helixc\tests\test_autodiff_reverse.py helixc\tests\test_pytree.py helixc\tests\test_ptx.py helixc\tests\test_tile_ir.py helixc\tests\test_autotune.py helixc\tests\test_ffi.py helixc\tests\test_cli.py -k "emit_ptx or ptx or tile_ir or autotune or ffi or stage35 or grad_rejects_aggregate" --tb=short`
  - Result: 108 passed.

## Next Gate

The Stage 35 clean-gate counter remains `0/3`. The next attempt must be a fresh
clean gate on the fixed commit.
