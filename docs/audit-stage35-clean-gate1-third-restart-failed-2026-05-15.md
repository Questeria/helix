# Stage 35 Clean Gate 1 - Third Restart Failed

Date: 2026-05-15
Baseline: `4dec3f7`
Result: failed. Clean-gate count remains `0/3`.

## Findings

Three fresh audit lanes found real Stage 35 issues:

1. Reverse-mode AD could still fail open on unsupported opaque calls. An extern
   or otherwise bodyless helper could be treated as an empty inlined body, then
   compile a zero-gradient surrogate after only an AD warning.
2. Scalar cross-entropy had a checked helper for positive out-of-range labels,
   but the public `ce_loss` API still lacked the row width needed to reject
   positive OOB labels safely.
3. The standalone `python -m helixc.backend.ptx` entry point did not match the
   main `--emit-ptx` path. It bypassed autotune validation and lowered host
   helper functions to Tile IR, so host-only unsupported ops could break device
   emission.
4. Stage 35 documentation still had stale language around kernel purity,
   Phase-0 GPU lowering, FFI status, and dense-classifier scratch handling.

## Fix Sweep

- `_inline_user_calls` now leaves extern/bodyless functions opaque instead of
  inlining empty declarations.
- Reverse-mode AD now raises `NotImplementedError` on opaque calls, so the
  public `grad_rev` path fails closed until a chain rule or differentiable
  helper exists.
- `ce_loss` now requires `cols` and rejects both negative and positive invalid
  class labels with the loud sentinel.
- The standalone PTX CLI now runs `validate_autotune_prog` and filters the TIR
  module to `@kernel` functions before Tile IR lowering.
- `docs/lang/spec.md`, `docs/lang/agi-features.md`, and the Stage 35 progress
  log were updated to reflect current behavior.

## Verification

- `python -m pytest -q helixc\tests\test_codegen.py -k "grad_rev_rejects_opaque_call_in_loss or ce_loss_rejects_negative_scalar_label or ce_loss_rejects_positive_out_of_range_label or dense_classifier_sgd_step_f32_leaves_scratch_unchanged or builtin_bce_and_nn_bce_are_stable_near_one" --tb=short`
  - Result: 5 passed.
- `python -m pytest -q helixc\tests\test_ptx.py -k "stage35_direct_ptx_cli_rejects_oversized_autotune or stage35_direct_ptx_cli_ignores_host_helper_with_unsupported_tile_op or c119_direct_ptx_cli_rejects_modules_without_kernels or c119_direct_ptx_cli_rejects_kernel_helper_calls" --tb=short`
  - Result: 4 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or stage35 or softmax or ce_loss or dense_classifier_sgd_step_f32 or adam_f32_step or builtin_adam_step or grad_rejects_aggregate_param or grad_rev_rejects_aggregate_param or grad_rev_all_rejects_aggregate_param or scalar_target_when_sibling_aggregate or grad_rev_rejects_opaque_call_in_loss or embedded_ptx_ignores_host_helper or builtin_bce_uses_stable_log_near_zero or builtin_bce_and_nn_bce_are_stable_near_one" --tb=short`
  - Result: 65 passed.
- `python -m pytest -q helixc\tests\test_autodiff.py helixc\tests\test_autodiff_reverse.py helixc\tests\test_ptx.py helixc\tests\test_tile_ir.py helixc\tests\test_autotune.py helixc\tests\test_ffi.py helixc\tests\test_cli.py -k "ad_warns_on_opaque_call or reverse or emit_ptx or ptx or tile_ir or autotune or ffi or stage35 or grad_rejects_aggregate or scalar_target_when_sibling_aggregate" --tb=short`
  - Result: 130 passed.

## Gate Status

This was a failed clean-gate attempt because findings were real and required
code changes. After this fix sweep, restart Stage 35 clean gate 1 from the new
commit. Clean-gate count remains `0/3`.
