# Stage 35 Clean Gate 1 Second Restart - Failed And Fixed

Date: 2026-05-15

Result: failed clean gate restart, fixed, reset to 0/3 clean gates.

After commit `88594d3`, a fresh Stage 35 clean-gate restart found additional
NN and embedded-PTX issues. This attempt therefore did not count as clean.

## Findings Fixed

1. `dense_classifier_sgd_step_f32` still trusted caller scratch capacity for
   `3 * classes` cells. The existing guard test used exactly the required
   length, so it missed the undersized case.
2. Adam zero-denominator handling failed open for nonzero numerator: clamping the
   denominator to `0.000001` turned `m = 1, v = 0, eps = 0` into an enormous
   update.
3. Scalar `ce_loss` accepted a negative label and read the arena cell before the
   probability row.
4. Embedded PTX generation in `x86_64.py` lowered the entire TIR module, so a
   valid host-only helper with an unsupported-for-PTX op could break GPU kernel
   embedding.
5. The public language spec overclaimed current GPU/tile support for SMEM/REG
   and `bf16` kernels.
6. Aggregate AD fail-closed tests did not pin the case where the requested
   derivative target was scalar but a sibling parameter was aggregate.

## Fix Summary

- `dense_classifier_sgd_step_f32` now computes the dense scores, softmax
  denominator, and per-class deltas directly without writing caller scratch.
- `adam_f32_step` and scalar `__adam_step` now treat a zero or negative
  denominator as no step instead of a huge artificial step.
- Scalar `ce_loss` returns the loud sentinel for negative target labels.
- Embedded PTX lowering now constructs a kernel-only TIR module before Tile IR
  lowering, matching the public `--emit-ptx` path.
- The language spec now states the current Phase-0 PTX limit: 1D HBM `f32` and
  `i32` kernel parameters only.
- Added sibling-aggregate `grad(..., 1)` and `grad_rev(..., 1)` regressions.

## Verification

- `python -m pytest -q helixc\tests\test_codegen.py -k "dense_classifier_sgd_step_f32_does_not_clobber_small_scratch or adam_f32_step_nonzero_m_zero_denom_keeps_weight or builtin_adam_step_nonzero_m_zero_denom_returns_zero or ce_loss_rejects_negative_scalar_label or grad_rejects_scalar_target_when_sibling_aggregate_param_exists or grad_rev_rejects_scalar_target_when_sibling_aggregate_param_exists or embedded_ptx_ignores_host_helper_with_unsupported_tile_op" --tb=short`
  - Result: 7 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or stage35 or softmax or ce_loss or dense_classifier_sgd_step_f32 or adam_f32_step or builtin_adam_step or grad_rejects_aggregate_param or grad_rev_rejects_aggregate_param or grad_rev_all_rejects_aggregate_param or scalar_target_when_sibling_aggregate or embedded_ptx_ignores_host_helper or builtin_bce_uses_stable_log_near_zero" --tb=short`
  - Result: 61 passed.
- `python -m pytest -q helixc\tests\test_autodiff_reverse.py helixc\tests\test_pytree.py helixc\tests\test_ptx.py helixc\tests\test_tile_ir.py helixc\tests\test_autotune.py helixc\tests\test_ffi.py helixc\tests\test_cli.py -k "emit_ptx or ptx or tile_ir or autotune or ffi or stage35 or grad_rejects_aggregate or scalar_target_when_sibling_aggregate" --tb=short`
  - Result: 108 passed.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Next Gate

The Stage 35 clean-gate counter remains `0/3`. The next attempt must be a fresh
clean gate on the fixed commit.
