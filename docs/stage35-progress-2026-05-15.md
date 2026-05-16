# Stage 35 Progress - 2026-05-15

## Stage Goal

Stage 35 is the AI/ML Capability Push.

Beginner meaning: Helix needs to become better at building and checking AI
systems directly. That means gradients, structured model parameters, tensor
helpers, and later GPU/tile/autotune paths.

## Reconnaissance Baseline

Initial Stage 35 reconnaissance found:

- `grad_rev_all` and reverse-mode AD already support scalar parameters.
- Pytree flatten/unflatten support exists for nested model-like structures.
- Tensor and neural-network stdlib helpers exist and are a safer near-term path
  than GPU work.
- Tile, PTX, and autotune surfaces exist but are still higher-risk scaffolding.

The first safe path is therefore non-GPU AI/ML structure work before expanding
PTX/tile lowering.

## Increment 1 - Reverse-Mode Model Field Leaves

Reverse-mode AD now treats static field paths as differentiable leaves.

Examples now supported at the symbolic AD level:

- `m.w1`
- `m.w2`
- `m.layer.w`

This matters because real models are usually structured values with fields, not
only loose scalar parameters.

Behavior added:

- `differentiate_reverse(body, ["m.w1"])` can accumulate the gradient for
  `m.w1`.
- Nested field leaves such as `m.layer.w` work the same way.
- Static non-target fields such as `m.w` are treated as coefficients when
  differentiating with respect to another value like `x`.
- Non-static field expressions still warn instead of silently pretending they
  have a proven derivative path.

## Verification

Initial focused checks:

- `python -m pytest -q helixc\tests\test_autodiff_reverse.py -k "stage35_reverse_ad" --tb=short`
  - Result before implementation: 3 failed, proving the gap.
  - Result after implementation: 3 passed.
- `python -m pytest -q helixc\tests\test_autodiff_reverse.py helixc\tests\test_pytree.py --tb=short`
  - Result: 46 passed.
- `python -m pytest -q helixc\tests\test_autodiff_parity.py --tb=short`
  - Result: 7 passed.
- `python -m pytest -q helixc\tests\test_transcendentals.py -k "grad_rev_all" --tb=short`
  - Result: 2 passed, 22 deselected.
- `python -m pytest -q helixc\tests\test_codegen.py::test_bootstrap_kovc_full_pipeline_arithmetic --tb=short`
  - Result: 1 passed.
  - Note: this large pipeline test contains the older Stage 14
    `grad_rev_all` codegen checks.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Increment 2 - f32 Gradient-Norm Clipping

The Helix neural-network stdlib now includes:

- `clip_grad_norm_f32(g_start, max_norm, n)`

It computes the f32 gradient vector's L2 norm and scales the vector in place
when the norm is larger than `max_norm`.

This matters for AI training because clipping prevents a very large gradient
from making a model update unstable.

Behavior added:

- Large gradients are scaled down toward the requested maximum norm.
- Small gradients are left unchanged.
- Empty or zero-norm gradients return cleanly.
- Negative `max_norm` is treated as `0.0`, so the helper remains deterministic.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "clip_grad_norm_f32 or sgd_f32_step or tf1d_l2_norm_sq or tf1d_scale_inplace" --tb=short`
  - Result: 5 passed, 751 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Increment 3 - f32 Weight-Decay Gradient Helper

The Helix neural-network stdlib now includes:

- `add_weight_decay_grad_f32(g_start, w_start, decay, n)`

It updates a gradient vector in place:

- `g[i] = g[i] + decay * w[i]`

This is the standard gradient contribution for L2 weight decay. In beginner
terms, it helps keep model weights from growing too large during training.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "weight_decay_grad_f32 or clip_grad_norm_f32 or tf1d_axpby" --tb=short`
  - Result: 4 passed, 753 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Increment 4 - f32 Stable SGD Step

The Helix neural-network stdlib now includes:

- `sgd_f32_step_decay_clip(w_start, g_start, lr, decay, max_norm, n)`

It composes the training helpers into one practical update:

1. Add weight decay to the gradient.
2. Clip the gradient norm.
3. Apply the f32 SGD step.

This is still simple, but it is closer to what real model training needs than
raw SGD alone.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "sgd_f32_step_decay_clip or weight_decay_grad_f32 or clip_grad_norm_f32 or sgd_f32_step" --tb=short`
  - Result: 5 passed, 753 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Increment 5 - f32 Layer Normalization

The Helix neural-network stdlib now includes:

- `layer_norm_f32(x_start, y_start, n, eps)`

It normalizes one f32 vector by subtracting its mean and dividing by
`sqrt(variance + eps)`.

This is an important AI building block because layer normalization keeps model
activations in a stable range, especially in transformer-style networks.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "layer_norm_f32 or softmax or tanh_layer or leaky_relu" --tb=short`
  - Result: 6 passed, 753 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Increment 6 - f32 Dropout

The Helix neural-network stdlib now includes:

- `dropout_f32(x_start, y_start, n, keep_prob, seed)`

It implements deterministic inverted dropout for f32 vectors. Kept values are
scaled by `1 / keep_prob`; dropped values become zero. The helper returns the
final RNG state so callers can continue a reproducible training stream.

This matters because dropout is a standard training-time regularization tool.
In beginner terms, it helps models avoid relying too heavily on one activation
path.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "dropout_f32 or layer_norm_f32 or rand_step" --tb=short`
  - Result: 3 passed, 758 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Increment 7 - Modern f32 Activation Layers

The Helix neural-network stdlib now includes vector layers for existing scalar
modern activations:

- `softplus_layer`
- `silu_layer`
- `gelu_layer`

These matter because modern AI models often use smooth activations rather than
plain ReLU alone. GELU and SiLU in particular are common in transformer-style
networks.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "modern_activation_layers or softplus or silu or gelu or sigmoid_layer or tanh_layer" --tb=short`
  - Result: 2 passed, 760 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Increment 8 - Row-Wise f32 Argmax

The Helix neural-network stdlib now includes:

- `argmax_rows_f32(logits_start, rows, cols, out_start)`

It reads a row-major f32 logits matrix and writes one predicted class id per
row. This is a common step after model inference for classification tasks.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "argmax_rows_f32 or softmax or count_correct" --tb=short`
  - Result: 5 passed, 758 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Neural-Network Cluster Verification

After increments 2 through 6, the broader neural-network regression slice was:

- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or attention_softmax_f32" --tb=short`
  - Result: 24 passed, 737 deselected.

## Increment 9 - f32 Classification Helpers

The Helix neural-network stdlib now includes:

- `accuracy_count_from_logits_f32`
- `ce_loss_batch_f32`

These make classification workflows more complete:

- `accuracy_count_from_logits_f32` counts correct predictions directly from a
  row-major logits matrix.
- `ce_loss_batch_f32` computes average cross-entropy over probability rows.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "accuracy_count_from_logits_f32 or ce_loss_batch_f32 or argmax_rows_f32 or ce_loss or count_correct" --tb=short`
  - Result: 4 passed, 761 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Increment 10 - f32 Dense-Layer Backprop Helpers

The Helix neural-network stdlib now includes dense-layer gradient helpers for
`y = W @ x + b`:

- `dense_layer_f32_grad_w`
- `dense_layer_f32_grad_b`
- `dense_layer_f32_grad_x`

These compute gradients for weights, bias, and inputs. This is a core step
toward training multi-layer models directly in Helix.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "dense_layer_f32_grad or dense_layer_f32_forward or tf2d_matvec or sgd_f32_step" --tb=short`
  - Result: 5 passed, 763 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Increment 11 - f32 Activation Backprop Helpers

The Helix neural-network stdlib now includes backward helpers for common f32
activation layers:

- `relu_layer_f32_backward`
- `sigmoid_layer_backward`
- `tanh_layer_backward`

These helpers turn upstream gradients into input gradients for activation
layers. This is another step toward writing full training loops directly in
Helix.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "activation_backprop_layers or relu_layer_f32_backward or sigmoid_layer_backward or tanh_layer_backward or modern_activation_layers" --tb=short`
  - Result: 2 passed, 767 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Increment 12 - f32 MSE-Loss Gradient Helper

The Helix neural-network stdlib now includes:

- `mse_loss_f32_grad(y_start, t_start, dy_start, n)`

It writes the gradient of mean squared error with respect to the prediction
vector:

- `dy[i] = 2 * (y[i] - target[i]) / n`

This supplies a simple starting gradient for backpropagation through a model.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "mse_loss_f32_grad or mse_loss_f32 or dense_layer_f32_grad" --tb=short`
  - Result: 4 passed, 766 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Increment 13 - PTX Kernel Regression Expansion

Stage 35 added GPU/PTX regression coverage for additional kernel forms:

- f32 HBM vector multiply emits `mul.f32`.
- f32 HBM vector negation emits `neg.f32`.
- i32 HBM vector add emits signed global load/store and `add.s32`.

This does not yet mean Helix runs GPU kernels directly from the test harness.
It strengthens the PTX codegen path by proving the embedded PTX module covers
more than the original f32 add kernel.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "stage35_vec_mul_kernel_ptx_in_binary or stage35_vec_neg_kernel_ptx_in_binary or stage35_i32_kernel_ptx_in_binary or stage16" --tb=short`
  - Result: 6 passed, 767 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Neural-Network Cluster Verification 2

After the dense and activation backprop work, the broader neural-network slice
was:

- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or attention_softmax_f32" --tb=short`
  - Result: 32 passed, 737 deselected.

## Increment 14 - Deterministic Autotune Variant Order

Autotune variant generation now sorts parameter keys before creating the
Cartesian product.

This makes generated variant order reproducible even if attributes arrive in a
different key order. Reproducibility matters because autotune output eventually
feeds generated kernel names, cache records, and benchmark comparisons.

Focused verification:

- `python -m pytest -q helixc\tests\test_autotune.py -k "variants" --tb=short`
  - Result: 4 passed, 21 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Increment 15 - Pytree Parameter Path Bridge

The Python frontend pytree support now includes:

- `flatten_pytree_param(param, struct_decls)`

It turns function parameters into AD-ready leaf paths:

- scalar parameter `x: f32` becomes `x`
- struct parameter `model: Model` becomes leaves like `model.layer.w`

This bridges the Stage 35 field-leaf AD work with nested model parameters. It
does not yet fully expose `grad(loss)(model)` as a runtime surface, but it
adds the static naming helper needed for that next wire-up.

Focused verification:

- `python -m pytest -q helixc\tests\test_pytree.py -k "stage35_flatten_pytree_param or flatten_nested_struct" --tb=short`
  - Result: 3 passed, 21 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Increment 16 - Stage 35 Audit Response

Three read-only Stage 35 audit lanes found real issues. This increment closes
the concrete findings before adding more features.

AD and pytree safety:

- Reverse-mode AD now warns when a caller asks for a top-level struct target
  such as `m`, but the expression only has a field path such as `m.w`.
- Match-pattern shadowing now treats `m` as shadowing dotted leaves like
  `m.w`, preventing false gradients from arm-local bindings.

PTX regression strength:

- Stage 35 PTX tests now extract the embedded PTX module from the ELF and
  check dataflow-sensitive details, including parameter slots, `%tid.x`,
  global load/store counts, and exact operation register flow.

Neural-network stdlib safety:

- Cross-entropy and BCE now use `__log_stable` for ordinary probabilities
  such as `0.5` and `0.1`, not only one-hot `1.0`.
- `ce_loss_batch_f32` now rejects invalid class labels with a loud sentinel
  instead of reading outside the row.
- `softplus_layer` now uses the stable softplus formula through
  `__log_stable`, covering central values such as `0`, `-2`, and `2`.

Focused verification:

- `python -m pytest -q helixc\tests\test_autodiff_reverse.py -k "stage35 or pattern_shadow" --tb=short`
  - Result: 6 passed, 20 deselected.
- `python -m pytest -q helixc\tests\test_codegen.py -k "stage35_vec_mul_kernel_ptx_in_binary or stage35_vec_neg_kernel_ptx_in_binary or stage35_i32_kernel_ptx_in_binary" --tb=short`
  - Result: 3 passed, 773 deselected.
- `python -m pytest -q helixc\tests\test_codegen.py -k "ce_loss_batch_f32 or softplus_layer_central_range or modern_activation_layers" --tb=short`
  - Result: 5 passed, 771 deselected.
- `python -m pytest -q helixc\tests\test_autodiff_reverse.py helixc\tests\test_pytree.py helixc\tests\test_ptx.py helixc\tests\test_tile_ir.py --tb=short`
  - Result: 109 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or stage35 or stage16" --tb=short`
  - Result: 41 passed, 735 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Increment 17 - Classifier Training Helpers

The Helix neural-network stdlib now includes two classifier-training helpers:

- `softmax_rows_f32(logits_start, probs_start, rows, cols)`
- `softmax_ce_grad_f32(probs_start, target_start, grad_start, rows, cols)`

Beginner meaning:

- `softmax_rows_f32` turns each row of classifier scores into probabilities.
- `softmax_ce_grad_f32` writes the standard gradient for softmax plus
  cross-entropy, which is the usual first backward step for classification
  models.

Safety behavior:

- Empty or invalid matrix sizes return success without writing.
- Invalid class labels return sentinel status `35001`, so bad target data does
  not quietly produce believable gradients.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "softmax_rows_f32 or softmax_ce_grad_f32 or ce_loss_batch_f32 or softmax_sums_to_one" --tb=short`
  - Result: 7 passed, 772 deselected.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or softmax or ce_loss or dense_layer_f32_grad" --tb=short`
  - Result: 39 passed, 740 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Increment 18 - Adam Optimizer Helper

The Helix neural-network stdlib now includes:

- `adam_f32_step(w_start, g_start, m_start, v_start, lr, beta1, beta2, eps, n)`

Beginner meaning:

- SGD moves weights directly by the current gradient.
- Adam also tracks a moving average of the gradient and squared gradient, which
  is much closer to how modern AI models are usually trained.

Current scope:

- This is an Adam-style step with uncorrected moving moments.
- It updates weights, first moment `m`, and second moment `v` in Helix memory.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "adam_f32_step or sgd_f32_step or clip_grad_norm_f32 or add_weight_decay_grad_f32" --tb=short`
  - Result: 6 passed, 774 deselected.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or adam_f32_step or sgd_f32_step or dense_layer_f32_grad" --tb=short`
  - Result: 39 passed, 741 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

## Increment 19 - Dense Classifier SGD Step

The Helix neural-network stdlib now includes:

- `dense_classifier_sgd_step_f32(w_start, b_start, x_start, target, scratch_start, shape_start, lr)`

Beginner meaning:

This is a compact one-sample classifier training step. It composes existing
Helix-native pieces:

- dense forward pass with bias
- softmax probabilities
- softmax-cross-entropy gradient
- dense weight gradient
- SGD updates for weights and bias

The API intentionally uses `scratch_start` and `shape_start` arena handles so it
stays within the current backend's six-integer-argument limit.

Safety behavior:

- Invalid class labels return sentinel status `35001`.
- Invalid or empty dimensions return success without writing.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "dense_classifier_sgd_step_f32 or softmax_ce_grad_f32 or dense_layer_f32_grad_w or sgd_f32_step" --tb=short`
  - Result: 7 passed, 775 deselected.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or dense_classifier_sgd_step_f32 or softmax_ce_grad_f32 or dense_layer_f32" --tb=short`
  - Result: 41 passed, 741 deselected.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - First result: failed because the new helper called `dense_layer_f32_forward`
    with the old five-argument shape.
  - Fixed by using the existing bias-aware six-argument signature.
  - Final result: passed, `stage31-quick: rc=0`.

## Increment 20 - Clean Gate 1 Fix Sweep

The first Stage 35 clean-gate audit did not count as clean. Three read-only
audit lanes found real issues in the AD/pytree public rewrite surface, NN
training helpers, BCE stability, Adam edge handling, and Tile IR fail-open
behavior.

Fixes landed in this increment:

- `grad`, `grad_rev`, and `grad_rev_all` now reject aggregate/non-scalar
  parameters in `grad_pass` until pytree leaf expansion is wired into the
  public surface.
- `dense_classifier_sgd_step_f32` no longer needs a full weight-gradient scratch
  matrix. A later restart sweep tightened this further: the helper no longer
  writes the caller scratch handle at all, so repeated training steps do not
  grow the arena or mutate scratch cells.
- `adam_f32_step` avoids a zero denominator, so zero-gradient updates with
  `eps = 0` keep weights stable.
- Scalar `__adam_step` now shares the same zero-denominator guard.
- Builtin `__bce` uses `__log_stable`, matching the fixed NN BCE path.
- Batch CE invalid labels stay sentinel-grade instead of being averaged down.
- `softmax_ce_grad_f32` prevalidates targets before writing output gradients.
- Tile IR now raises on unsupported TIR ops instead of silently mapping them to
  generic opaque calls.
- Public `--emit-ptx` now reports those Tile IR failures as normal PTX errors
  without an internal-compiler-bug label.
- The autotune docstring now distinguishes current Phase-0 static spec support
  from the long-term runtime timing/dispatch design.

Regression and focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "grad_rejects_aggregate_param or ce_loss_batch_f32_invalid_label_not_averaged_down or softmax_ce_grad_f32_invalid_batch_does_not_partially_mutate or dense_classifier_sgd_step_f32_reuses_scratch_without_arena_growth or dense_classifier_sgd_step_f32_does_not_clobber_small_scratch or builtin_adam_step_zero_denom_returns_zero or adam_f32_step_zero_grad_zero_eps_keeps_weight" --tb=short`
  - Result: 7 passed.
- `python -m pytest -q helixc\tests\test_cli.py -k "stage35_emit_ptx_reports_tile_lowering_error_without_bug_label or c117_emit_ptx_uses_kernel_attrs or c119_emit_ptx_rejects_no_kernel_modules or c119_emit_ptx_allows_folded_bool_constants or c119_emit_ptx_accepts_kernel_index_builtin or c119_emit_ptx_rejects_extern_only_kernels" --tb=short`
  - Result: 6 passed.
- `python -m pytest -q helixc\tests\test_tile_ir.py -k "tile_ir_rejects_unmapped_scalar_div or if_lowered_to_cfg_in_tile_ir or arith_passes_through or call_lowered" --tb=short`
  - Result: 4 passed.
- `python -m pytest -q helixc\tests\test_autodiff_reverse.py helixc\tests\test_pytree.py helixc\tests\test_ptx.py helixc\tests\test_tile_ir.py helixc\tests\test_autotune.py helixc\tests\test_ffi.py --tb=short`
  - Result: 139 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or stage35 or softmax or ce_loss or dense_classifier_sgd_step_f32 or adam_f32_step or builtin_adam_step_zero_denom_returns_zero or grad_rejects_aggregate_param or grad_rev_rejects_aggregate_param or grad_rev_all_rejects_aggregate_param or builtin_bce_uses_stable_log_near_zero" --tb=short`
  - Result: 55 passed.
- `python -m pytest -q helixc\tests\test_autodiff_reverse.py helixc\tests\test_pytree.py helixc\tests\test_ptx.py helixc\tests\test_tile_ir.py helixc\tests\test_autotune.py helixc\tests\test_ffi.py helixc\tests\test_cli.py -k "emit_ptx or ptx or tile_ir or autotune or ffi or stage35 or grad_rejects_aggregate" --tb=short`
  - Result: 108 passed.

Clean-gate status:

- Stage 35 clean gates reset to `0/3`.
- Next step is a fresh Stage 35 clean gate on the fixed commit.

## Increment 21 - Second Clean-Gate Restart Fix Sweep

Another fresh Stage 35 clean-gate restart found more concrete issues, so the
gate still did not count as clean.

Fixes landed in this increment:

- `dense_classifier_sgd_step_f32` no longer writes caller scratch at all. It
  computes dense scores, the softmax denominator, and each class delta directly
  before applying weight and bias updates.
- `adam_f32_step` and scalar `__adam_step` now treat a zero or negative
  denominator as no step, instead of creating a huge artificial update.
- Scalar `ce_loss` returns the loud sentinel for negative target labels.
- Embedded PTX generation in `x86_64.py` now lowers only kernel functions to
  Tile IR, so host-only helpers with unsupported-for-PTX ops do not break kernel
  embedding.
- `docs/lang/spec.md` now states current Phase-0 PTX support honestly: 1D HBM
  `f32` and `i32` kernel parameters only.
- Added `grad(..., 1)` and `grad_rev(..., 1)` sibling-aggregate regressions.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "dense_classifier_sgd_step_f32_does_not_clobber_small_scratch or adam_f32_step_nonzero_m_zero_denom_keeps_weight or builtin_adam_step_nonzero_m_zero_denom_returns_zero or ce_loss_rejects_negative_scalar_label or grad_rejects_scalar_target_when_sibling_aggregate_param_exists or grad_rev_rejects_scalar_target_when_sibling_aggregate_param_exists or embedded_ptx_ignores_host_helper_with_unsupported_tile_op" --tb=short`
  - Result: 7 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or stage35 or softmax or ce_loss or dense_classifier_sgd_step_f32 or adam_f32_step or builtin_adam_step or grad_rejects_aggregate_param or grad_rev_rejects_aggregate_param or grad_rev_all_rejects_aggregate_param or scalar_target_when_sibling_aggregate or embedded_ptx_ignores_host_helper or builtin_bce_uses_stable_log_near_zero" --tb=short`
  - Result: 61 passed.
- `python -m pytest -q helixc\tests\test_autodiff_reverse.py helixc\tests\test_pytree.py helixc\tests\test_ptx.py helixc\tests\test_tile_ir.py helixc\tests\test_autotune.py helixc\tests\test_ffi.py helixc\tests\test_cli.py -k "emit_ptx or ptx or tile_ir or autotune or ffi or stage35 or grad_rejects_aggregate or scalar_target_when_sibling_aggregate" --tb=short`
  - Result: 108 passed.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

Clean-gate status:

- Stage 35 clean gates reset to `0/3`.
- Next step is another fresh Stage 35 clean gate on the fixed commit.

## Increment 22 - Third Clean-Gate Restart Fix Sweep

The next fresh Stage 35 audit restart again found real issues, so the gate did
not count as clean and remains at `0/3`.

Fixes landed in this increment:

- Reverse-mode AD now fails closed on opaque calls instead of compiling a
  zero-gradient surrogate. The shared AD inliner now leaves extern/bodyless
  declarations opaque so `grad_rev` can reject them explicitly.
- Public scalar `ce_loss` now takes the row width and rejects both negative and
  positive out-of-range class labels with the loud sentinel.
- The standalone PTX CLI now matches the main `--emit-ptx` safety path by
  running autotune validation and lowering only `@kernel` functions to Tile IR.
- Stage documentation now reflects the current Phase-0 GPU limits, current FFI
  status, and the finalized dense-classifier scratch behavior.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "grad_rev_rejects_opaque_call_in_loss or ce_loss_rejects_negative_scalar_label or ce_loss_rejects_positive_out_of_range_label or dense_classifier_sgd_step_f32_leaves_scratch_unchanged or builtin_bce_and_nn_bce_are_stable_near_one" --tb=short`
  - Result: 5 passed.
- `python -m pytest -q helixc\tests\test_ptx.py -k "stage35_direct_ptx_cli_rejects_oversized_autotune or stage35_direct_ptx_cli_ignores_host_helper_with_unsupported_tile_op or c119_direct_ptx_cli_rejects_modules_without_kernels or c119_direct_ptx_cli_rejects_kernel_helper_calls" --tb=short`
  - Result: 4 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or stage35 or softmax or ce_loss or dense_classifier_sgd_step_f32 or adam_f32_step or builtin_adam_step or grad_rejects_aggregate_param or grad_rev_rejects_aggregate_param or grad_rev_all_rejects_aggregate_param or scalar_target_when_sibling_aggregate or grad_rev_rejects_opaque_call_in_loss or embedded_ptx_ignores_host_helper or builtin_bce_uses_stable_log_near_zero or builtin_bce_and_nn_bce_are_stable_near_one" --tb=short`
  - Result: 65 passed.
- `python -m pytest -q helixc\tests\test_autodiff.py helixc\tests\test_autodiff_reverse.py helixc\tests\test_ptx.py helixc\tests\test_tile_ir.py helixc\tests\test_autotune.py helixc\tests\test_ffi.py helixc\tests\test_cli.py -k "ad_warns_on_opaque_call or reverse or emit_ptx or ptx or tile_ir or autotune or ffi or stage35 or grad_rejects_aggregate or scalar_target_when_sibling_aggregate" --tb=short`
  - Result: 130 passed.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 23 - Fourth Clean-Gate Restart Fix Sweep

The next fresh Stage 35 audit restart again found real code and documentation
issues, so the gate did not count as clean and remains at `0/3`.

Fixes landed in this increment:

- Forward-mode `grad` now fails closed on opaque calls, matching the hardened
  `grad_rev` behavior.
- `grad_pass` now accepts only `f32`/`f64` scalar gradient parameters and
  preserves f64 generated signatures instead of narrowing them to f32.
- The Helix reverse-AD tape runtime now clamps negative capacity and rejects
  full-tape pushes without writing past the allocated tape.
- `dense_classifier_sgd_step_f32` now rejects invalid model shapes with the
  same loud sentinel used for invalid labels.
- The standalone PTX CLI now runs trace/panic/unwind/unsafe/autotune
  validation, runs `grad_pass`, and applies fold/CSE before kernel-only Tile IR
  lowering.
- Public docs now align Stage 35 with the AI/ML Capability Push, document the
  current scalar AD contract, mark broader transforms as future targets, expose
  current FFI status, and remove stale next-work text.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "grad_rejects_opaque_call_in_loss or grad_rev_rejects_opaque_call_in_loss or grad_pass_preserves_f64_gradient_signature or stage13c_grad_recursion_guard_does_not_infinite_loop or stage13d_grad_mutual_recursion_does_not_infinite_loop or revad_push_rejects_full_tape_without_overwrite or revad_negative_capacity_is_clamped_to_zero or dense_classifier_sgd_step_f32_rejects_invalid_shape" --tb=short`
  - Result: 8 passed.
- `python -m pytest -q helixc\tests\test_ptx.py -k "stage35_direct_ptx_cli_rejects_unwind_attr or stage35_direct_ptx_cli_folds_kernel_before_tile_lowering or stage35_direct_ptx_cli_rejects_oversized_autotune or stage35_direct_ptx_cli_ignores_host_helper_with_unsupported_tile_op" --tb=short`
  - Result: 4 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or stage35 or softmax or ce_loss or dense_classifier_sgd_step_f32 or adam_f32_step or builtin_adam_step or revad_ or builtin_bce_uses_stable_log_near_zero or builtin_bce_and_nn_bce_are_stable_near_one" --tb=short`
  - Result: 70 passed.
- `python -m pytest -q helixc\tests\test_ptx.py helixc\tests\test_tile_ir.py helixc\tests\test_autotune.py helixc\tests\test_cli.py -k "emit_ptx or ptx or tile_ir or autotune or unwind or unsafe or trace or stage35" --tb=short`
  - Result: 105 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "stage13 or grad_rejects_opaque_call_in_loss or grad_rev_rejects_opaque_call_in_loss or grad_pass_preserves_f64_gradient_signature or grad_rev_all or grad_rev or grad_rejects_aggregate_param or scalar_target_when_sibling_aggregate" --tb=short`
  - Result: 21 passed.
- `python -m pytest -q helixc\tests\test_autodiff.py helixc\tests\test_autodiff_reverse.py helixc\tests\test_pytree.py --tb=short`
  - Result: 90 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "c115_mixed_signed_unsigned_div_mod_runtime_parity or grad_rejects_opaque_call_in_loss or grad_pass_preserves_f64_gradient_signature" --tb=short`
  - Result: 3 passed.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 24 - Fifth Clean-Gate Restart Fix Sweep

The next fresh Stage 35 audit restart found real AD/runtime, PTX/autotune, and
documentation issues, so the gate did not count as clean and remains at `0/3`.

Fixes landed in this increment:

- Added explicit f64 reflection support with `splice_f64`, `modify_f64`, and
  `__always_accept_f64`; `grad_rev_all` now writes f64 gradients through the
  f64 cell path instead of corrupting them through `modify_f`.
- Hardened the Helix reverse-AD tape runtime so operation constructors reject
  invalid operand indices and `rev_backward` fails closed on corrupt tape
  operands instead of writing before the adjoint array.
- Clamped scalar CE probabilities to the same safe open interval used by BCE,
  preventing negative loss for probability values above 1.
- Made `softmax_layer` reject negative lengths without touching output cells,
  and made `tf1d_max` return 0 for non-positive lengths.
- Brought the standalone PTX CLI closer to `helixc.check --emit-ptx` parity by
  running module flattening, impl flattening, struct monomorphization, and
  function monomorphization before typecheck/lowering.
- `@autotune` duplicate keys now fail closed instead of overwriting earlier
  values.
- Public docs now identify Stage 35 as the active stage, remove the old
  zero-gradient opaque-call claim, and state the current x86_64 f64/PTX limits
  accurately.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "grad_rev_all_writes_f64_gradient_to_f64_cell or revad_ops_reject_invalid_operand_index_without_push or revad_backward_rejects_corrupt_operand_index or nn_ce_loss_clamps_probability_above_one or nn_softmax_layer_rejects_negative_length_without_write" --tb=short`
  - Result: 5 passed.
- `python -m pytest -q helixc\tests\test_autotune.py -k "duplicate_key or validate_autotune_surfaces_parse_diags or validate_autotune_prog_collects_diags" --tb=short`
  - Result: 3 passed.
- `python -m pytest -q helixc\tests\test_ptx.py -k "stage35_direct_ptx_cli_flattens_module_kernel or stage35_direct_ptx_cli_rejects_duplicate_autotune_key or stage35_direct_ptx_cli_rejects_unwind_attr or stage35_direct_ptx_cli_folds_kernel_before_tile_lowering" --tb=short`
  - Result: 4 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or stage35 or softmax or ce_loss or dense_classifier_sgd_step_f32 or adam_f32_step or builtin_adam_step or revad_ or grad_rev_all_writes_f64_gradient_to_f64_cell or builtin_bce_uses_stable_log_near_zero or builtin_bce_and_nn_bce_are_stable_near_one" --tb=short`
  - Result: 75 passed.
- `python -m pytest -q helixc\tests\test_ptx.py helixc\tests\test_tile_ir.py helixc\tests\test_autotune.py helixc\tests\test_cli.py -k "emit_ptx or ptx or tile_ir or autotune or unwind or unsafe or trace or stage35" --tb=short`
  - Result: 108 passed.
- `python -m pytest -q helixc\tests\test_autodiff.py helixc\tests\test_autodiff_reverse.py helixc\tests\test_pytree.py --tb=short`
  - Result: 90 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "stage13 or grad_rejects_opaque_call_in_loss or grad_rev_rejects_opaque_call_in_loss or grad_pass_preserves_f64_gradient_signature or grad_rev_all or grad_rev or grad_rejects_aggregate_param or scalar_target_when_sibling_aggregate" --tb=short`
  - Result: 22 passed.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.
- `git diff --check`
  - Result: passed.
- Stage/docs stale-claim scan for historic stage labels and old gradient/PTX/f64
  claims
  - Result: no matches.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 25 - Sixth Clean-Gate Restart Fix Sweep

The next fresh Stage 35 audit restart found AD/runtime, PTX CLI parity, and
documentation/status issues, so the gate did not count as clean and remains at
`0/3`.

Fixes landed in this increment:

- `grad_rev_all` now suffixes generated f64 gradient float literals before
  `modify_f64`, closing the constant-gradient corruption path.
- Reverse-AD adjoint arrays now carry capacity metadata, and `rev_seed` /
  `rev_grad` reject invalid indices instead of reading or writing outside the
  adjoint array.
- Sibling tensor/NN helpers now treat negative lengths as empty/sentinel
  requests instead of reading from unintended arena cells.
- The standalone PTX CLI now parses `--strict` / `--no-stdlib`, includes the
  bundled stdlib by default like `helixc.check --emit-ptx`, runs IR effect
  checking, and renders lex/parse diagnostics without Python tracebacks.
- `docs/lang/agi-features.md` now marks the `bf16` SMEM/REG tile matmul
  example as a future design target and states the current Phase-0 PTX limit:
  1D HBM `tile<f32, ...>` / `tile<i32, ...>` kernels plus a small scalar-op
  subset.
- The remaining-work table now says broader tile/GPU lowering and broader
  transform surfaces remain, instead of implying all tile codegen and scalar
  `grad` are still future.
- This progress ledger now lists increments 23, 24, and 25 in chronological
  order so the final visible status belongs to the latest restart.

Focused verification:

- `python -m pytest -q helixc\tests\test_codegen.py -k "grad_rev_all_writes_f64_constant_gradient_to_f64_cell or grad_rev_all_writes_f64_gradient_to_f64_cell or revad_seed_rejects_invalid_index_without_corrupting_tape or revad_grad_invalid_index_returns_zero or negative_length_tensor_nn_helpers_return_empty_values or revad_ops_reject_invalid_operand_index_without_push or revad_backward_rejects_corrupt_operand_index" --tb=short`
  - Result: 7 passed.
- `python -m pytest -q helixc\tests\test_ptx.py -k "stage35_direct_ptx_cli_strict_rejects_effect_violation or stage35_direct_ptx_cli_includes_stdlib_by_default or stage35_direct_ptx_cli_reports_parse_error_without_traceback or stage35_direct_ptx_cli_flattens_module_kernel or stage35_direct_ptx_cli_rejects_duplicate_autotune_key" --tb=short`
  - Result: 5 passed.
- `python -m py_compile helixc\frontend\grad_pass.py helixc\backend\ptx.py`
  - Result: passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or stage35 or softmax or ce_loss or dense_classifier_sgd_step_f32 or adam_f32_step or builtin_adam_step or revad_ or grad_rev_all_writes_f64 or negative_length_tensor_nn_helpers or builtin_bce_uses_stable_log_near_zero or builtin_bce_and_nn_bce_are_stable_near_one" --tb=short`
  - Result: 79 passed.
- `python -m pytest -q helixc\tests\test_ptx.py helixc\tests\test_tile_ir.py helixc\tests\test_autotune.py helixc\tests\test_cli.py -k "emit_ptx or ptx or tile_ir or autotune or unwind or unsafe or trace or stage35" --tb=short`
  - Result: 111 passed.
- `python -m pytest -q helixc\tests\test_autodiff.py helixc\tests\test_autodiff_reverse.py helixc\tests\test_pytree.py --tb=short`
  - Result: 90 passed.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.
- `python -m pytest -q helixc\tests\test_codegen.py -k "stage13 or grad_rejects_opaque_call_in_loss or grad_rev_rejects_opaque_call_in_loss or grad_pass_preserves_f64_gradient_signature or grad_rev_all or grad_rev or grad_rejects_aggregate_param or scalar_target_when_sibling_aggregate" --tb=short`
  - Result: 23 passed.
- `git diff --check`
  - Result: passed.
- Stage/docs stale-claim scan for historic stage labels and old gradient/PTX/f64
  claims
  - Result: no matches.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 26 - Seventh Clean-Gate Restart Fix Sweep

The next fresh Stage 35 audit restart found one remaining tensor negative-length
sentinel gap, direct PTX CLI parity drift, and public-documentation overclaiming,
so the gate did not count as clean and remains at `0/3`.

Fixes landed in this increment:

- `tf1d_is_empty` now treats `n <= 0` as empty, matching the hardened integer
  mirror and the rest of the f32 empty/sentinel helpers.
- The negative-length tensor/NN regression now explicitly covers
  `tf1d_is_empty(fx, -1)`.
- The standalone PTX CLI now accepts `--stdlib` as the compatibility spelling of
  the default bundled-stdlib mode.
- Direct PTX effect checking now scopes bundled-stdlib diagnostics the same way
  `helixc.check --emit-ptx` does, so clean strict kernels are not failed by
  unused stdlib helper warnings.
- Direct PTX missing-file errors now render as clean user-facing diagnostics
  instead of Python tracebacks.
- Public docs now frame AGI feature claims as Helix differentiator targets and
  under-served combinations instead of absolute comparison claims.
- The living language spec date now reflects the Stage 35 update.

Focused verification:

- `python -m py_compile helixc\backend\ptx.py`
  - Result: passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "negative_length_tensor_nn_helpers_return_empty_values" --tb=short`
  - Result: 1 passed.
- `python -m pytest -q helixc\tests\test_ptx.py -k "stage35_direct_ptx_cli_strict_allows_clean_default_stdlib_kernel or stage35_direct_ptx_cli_accepts_stdlib_compat_flag or stage35_direct_ptx_cli_reports_missing_file_without_traceback or stage35_direct_ptx_cli_strict_rejects_effect_violation or stage35_direct_ptx_cli_includes_stdlib_by_default or stage35_direct_ptx_cli_reports_parse_error_without_traceback" --tb=short`
  - Result: 6 passed.
- Public docs absolute-comparison/date scan for broad competitor claims and old
  spec date
  - Result: no matches.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or stage35 or softmax or ce_loss or dense_classifier_sgd_step_f32 or adam_f32_step or builtin_adam_step or revad_ or grad_rev_all_writes_f64 or negative_length_tensor_nn_helpers or builtin_bce_uses_stable_log_near_zero or builtin_bce_and_nn_bce_are_stable_near_one" --tb=short`
  - Result: 79 passed.
- `python -m pytest -q helixc\tests\test_ptx.py helixc\tests\test_tile_ir.py helixc\tests\test_autotune.py helixc\tests\test_cli.py -k "emit_ptx or ptx or tile_ir or autotune or unwind or unsafe or trace or stage35" --tb=short`
  - Result: 114 passed.
- `python -m pytest -q helixc\tests\test_autodiff.py helixc\tests\test_autodiff_reverse.py helixc\tests\test_pytree.py --tb=short`
  - Result: 90 passed.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.
- `python -m pytest -q helixc\tests\test_codegen.py -k "stage13 or grad_rejects_opaque_call_in_loss or grad_rev_rejects_opaque_call_in_loss or grad_pass_preserves_f64_gradient_signature or grad_rev_all or grad_rev or grad_rejects_aggregate_param or scalar_target_when_sibling_aggregate" --tb=short`
  - Result: 23 passed.
- `git diff --check`
  - Result: passed.
- Stage/docs stale-claim scan for historic stage labels and old gradient/PTX/f64
  claims
  - Result: no matches.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 27 - Eighth Clean-Gate Restart Fix Sweep

The next fresh Stage 35 audit restart found deeper runtime, PTX strict-parity,
and documentation/status issues, so the gate did not count as clean and remains
at `0/3`.

Fixes landed in this increment:

- Reverse-AD valid indices are now bounded by both tape `count` and tape `cap`.
- `rev_backward` now rejects corrupted tape counts before reading or writing
  adjoints when `count` exceeds the tape or adjoint capacity.
- 2D tensor helpers now use a safe shape length helper so negative dimensions
  are treated as empty instead of multiplying into a positive element count.
- `ti1d_min` and `ti1d_max` now treat negative lengths as empty sentinel cases.
- The standalone PTX CLI now mirrors `helixc.check --emit-ptx --strict`
  totality enforcement before emitting PTX.
- `docs/lang/agi-features.md` and `docs/ROADMAP.md` now avoid broad absolute
  comparison phrasing and use differentiator/target wording.
- The Stage 35 progress ledger is mechanically reordered so visible increment
  chronology reads `1` through `27`.

Focused verification:

- `python -m py_compile helixc\backend\ptx.py`
  - Result: passed.
- `python -m pytest -q helixc\tests\test_ptx.py -k "stage35_direct_ptx_cli_strict_rejects_totality_failure or stage35_direct_ptx_cli_strict_allows_clean_default_stdlib_kernel or stage35_direct_ptx_cli_accepts_stdlib_compat_flag or stage35_direct_ptx_cli_reports_missing_file_without_traceback" --tb=short`
  - Result: 4 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "revad_backward_rejects_count_above_capacity_without_adj_corruption or negative_length_integer_min_max_return_empty_sentinel or negative_2d_shape_helpers_treat_shape_as_empty or negative_length_tensor_nn_helpers_return_empty_values" --tb=short`
  - Result: 4 passed.
- Public docs absolute-comparison scan
  - Result: no matches.
- Stage/docs stale-claim scan for historic stage labels and old gradient/PTX/f64
  claims
  - Result: no matches.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or stage35 or softmax or ce_loss or dense_classifier_sgd_step_f32 or adam_f32_step or builtin_adam_step or revad_ or grad_rev_all_writes_f64 or negative_length_tensor_nn_helpers or negative_length_integer_min_max or negative_2d_shape_helpers or builtin_bce_uses_stable_log_near_zero or builtin_bce_and_nn_bce_are_stable_near_one" --tb=short`
  - Result: 82 passed.
- `python -m pytest -q helixc\tests\test_ptx.py helixc\tests\test_tile_ir.py helixc\tests\test_autotune.py helixc\tests\test_cli.py -k "emit_ptx or ptx or tile_ir or autotune or unwind or unsafe or trace or stage35" --tb=short`
  - Result: 115 passed.
- `python -m pytest -q helixc\tests\test_autodiff.py helixc\tests\test_autodiff_reverse.py helixc\tests\test_pytree.py --tb=short`
  - Result: 90 passed.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 28 - Ninth Clean-Gate Restart Fix Sweep

The next fresh Stage 35 audit restart found one reverse-AD metadata integrity
gap, one direct PTX CLI invocation-parity gap, and several documentation
current-vs-future wording issues, so the gate did not count as clean and
remains at `0/3`.

Fixes landed in this increment:

- Reverse-AD adjoint arrays now store redundant metadata guards before and
  after the adjoint array.
- `rev_seed`, `rev_grad`, and `rev_backward` now validate adjoint metadata before
  trusting the capacity.
- Direct PTX now returns exit code `2` for bad invocations such as unknown
  flags, extra paths, and missing input files, matching `helixc.check`.
- AGI feature docs now say reflection, `modify`, and auto-curriculum runtime
  behavior are design targets while the current implementation is stub or
  type-level.
- The spec now separates unrestricted runtime type info from the verifier-gated
  reflection scaffold, marks the GPU/tile example as design-target code, and
  labels the 2026-05-04 implementation/test section as historical.
- The roadmap now calls reflection a scaffold and avoids broad AGI win
  comparison wording.

Focused verification:

- `python -m py_compile helixc\backend\ptx.py`
  - Result: passed.
- `python -m pytest -q helixc\tests\test_ptx.py -k "stage35_direct_ptx_cli_bad_invocation_returns_two or stage35_direct_ptx_cli_reports_missing_file_without_traceback or stage35_direct_ptx_cli_strict_rejects_totality_failure" --tb=short`
  - Result: 3 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "revad_seed_rejects_corrupt_adj_cap_metadata_without_guard_write or revad_grad_rejects_corrupt_adj_cap_metadata_without_guard_read or revad_seed_rejects_corrupt_adj_guard_metadata or revad_backward_rejects_count_above_capacity_without_adj_corruption or revad_seed_rejects_invalid_index_without_corrupting_tape or revad_grad_invalid_index_returns_zero" --tb=short`
  - Result: 6 passed.
- Current-vs-future docs scan for reflection, modify, auto-curriculum, PTX, test
  count, and broad comparison wording
  - Result: no matches.
- Stage/docs stale-claim scan for historic stage labels and old gradient/PTX/f64
  claims
  - Result: no matches.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or stage35 or softmax or ce_loss or dense_classifier_sgd_step_f32 or adam_f32_step or builtin_adam_step or revad_ or grad_rev_all_writes_f64 or negative_length_tensor_nn_helpers or negative_length_integer_min_max or negative_2d_shape_helpers or builtin_bce_uses_stable_log_near_zero or builtin_bce_and_nn_bce_are_stable_near_one" --tb=short`
  - Result: 85 passed.
- `python -m pytest -q helixc\tests\test_ptx.py helixc\tests\test_tile_ir.py helixc\tests\test_autotune.py helixc\tests\test_cli.py -k "emit_ptx or ptx or tile_ir or autotune or unwind or unsafe or trace or stage35" --tb=short`
  - Result: 116 passed.
- `python -m pytest -q helixc\tests\test_autodiff.py helixc\tests\test_autodiff_reverse.py helixc\tests\test_pytree.py --tb=short`
  - Result: 90 passed.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 29 - Tenth Clean-Gate Restart Fix Sweep

The next fresh Stage 35 audit restart found one negative-shape runtime write
bug, two direct PTX invocation-parity gaps, and current-vs-future documentation
issues, so the gate did not count as clean and remains at `0/3`.

Fixes landed in this increment:

- `ti2d_matvec` and `tf2d_matvec` now treat non-positive row or column counts as
  empty no-op shapes before writing outputs.
- Integer and f32 dense-layer forward helpers now inherit the same non-positive
  shape no-op behavior instead of adding bias over invalid matrix metadata.
- Direct PTX no-argument invocation now returns exit code `2` with a missing
  input path diagnostic.
- Direct PTX strict stdlib loading now converts missing stdlib files into clean
  exit code `2` diagnostics instead of Python tracebacks.
- AGI feature docs now describe `modify_self` as the current explicit
  capability boundary for future source-rewrite operations.
- Memory-tier docs now distinguish the current type-level wrappers and selected
  builtin checks from future first-class cross-tier runtime invariants.
- The roadmap dogfood count now matches the five current dogfood programs/tests.

Focused verification:

- Per-file stdlib parser sweep across `STDLIB_FILES`
  - Result: all stdlib files parsed.
- `python -m py_compile helixc\backend\ptx.py`
  - Result: passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "negative_2d_matvec_shapes_do_not_write_outputs or negative_dense_layer_shapes_do_not_write_outputs or revad_seed_rejects_corrupt_adj_cap_metadata_without_guard_write or revad_grad_rejects_corrupt_adj_cap_metadata_without_guard_read or revad_seed_rejects_corrupt_adj_guard_metadata" --tb=short`
  - Result: 5 passed.
- `python -m pytest -q helixc\tests\test_ptx.py -k "stage35_direct_ptx_cli_bad_invocation_returns_two or stage35_direct_ptx_cli_missing_strict_stdlib_returns_two or stage35_direct_ptx_cli_reports_missing_file_without_traceback or stage35_direct_ptx_cli_strict_rejects_totality_failure" --tb=short`
  - Result: 4 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or stage35 or softmax or ce_loss or dense_classifier_sgd_step_f32 or adam_f32_step or builtin_adam_step or revad_ or grad_rev_all_writes_f64 or negative_length_tensor_nn_helpers or negative_length_integer_min_max or negative_2d_shape_helpers or builtin_bce_uses_stable_log_near_zero or builtin_bce_and_nn_bce_are_stable_near_one" --tb=short`
  - Result: 85 passed.
- `python -m pytest -q helixc\tests\test_ptx.py helixc\tests\test_tile_ir.py helixc\tests\test_autotune.py helixc\tests\test_cli.py -k "emit_ptx or ptx or tile_ir or autotune or unwind or unsafe or trace or stage35" --tb=short`
  - Result: 117 passed.
- `python -m pytest -q helixc\tests\test_autodiff.py helixc\tests\test_autodiff_reverse.py helixc\tests\test_pytree.py --tb=short`
  - Result: 90 passed.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.
- `git diff --check`
  - Result: passed.
- Current-vs-future docs scan for dogfood count, source rewriting,
  memory-tier runtime claims, and broad comparison wording
  - Result: remaining matches are explicitly future-target phrasing, not
    current capability claims.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 30 - Eleventh Clean-Gate Restart Fix Sweep

The next fresh Stage 35 audit restart found invalid-shape f32 gradient writes,
negative-offset f32 range reads, direct PTX flags-only invocation drift, and one
roadmap contradiction, so the gate did not count as clean and remains at `0/3`.

Fixes landed in this increment:

- `dense_layer_f32_grad_x` now treats non-positive row or column counts as a
  no-op before writing `grad_x`.
- `tf1d_argmax_in_range` and `tf1d_sum_in_range` now reject negative lower
  bounds before reading from the arena.
- `tf1d_dot_with_offset` now rejects negative offsets or non-positive lengths
  before reading from the arena.
- Direct PTX now requires a source path after flag parsing, so `--strict`,
  `--stdlib`, or both without a path return exit code `2`.
- The roadmap now describes current Phase-0 PTX tile lowering honestly while
  keeping broader tensor/tile GPU lowering as future work.

Focused verification:

- Per-file stdlib parser sweep across `STDLIB_FILES`
  - Result: all stdlib files parsed.
- `python -m py_compile helixc\backend\ptx.py`
  - Result: passed.
- `python -m pytest -q helixc\tests\test_ptx.py -k "stage35_direct_ptx_cli_bad_invocation_returns_two or stage35_direct_ptx_cli_missing_strict_stdlib_returns_two" --tb=short`
  - Result: 2 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "negative_dense_layer_f32_grad_x_shape_does_not_write_outputs or negative_tf1d_dot_with_offset_does_not_read_before_start or negative_tf1d_range_helpers_do_not_read_before_start or nn_dense_layer_f32_grad_x or stdlib_tf1d_dot_with_offset or stdlib_tf1d_argmax_in_range or stdlib_tf1d_sum_in_range" --tb=short`
  - Result: 7 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or stage35 or softmax or ce_loss or dense_classifier_sgd_step_f32 or adam_f32_step or builtin_adam_step or revad_ or grad_rev_all_writes_f64 or negative_length_tensor_nn_helpers or negative_length_integer_min_max or negative_2d_shape_helpers or negative_tf1d or builtin_bce_uses_stable_log_near_zero or builtin_bce_and_nn_bce_are_stable_near_one" --tb=short`
  - Result: 87 passed.
- `python -m pytest -q helixc\tests\test_ptx.py helixc\tests\test_tile_ir.py helixc\tests\test_autotune.py helixc\tests\test_cli.py -k "emit_ptx or ptx or tile_ir or autotune or unwind or unsafe or trace or stage35" --tb=short`
  - Result: 117 passed.
- `python -m pytest -q helixc\tests\test_autodiff.py helixc\tests\test_autodiff_reverse.py helixc\tests\test_pytree.py --tb=short`
  - Result: 90 passed.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`
  - Result: passed, `stage31-quick: rc=0`.
- `git diff --check`
  - Result: passed.
- Docs scan for stale tile-lowering, dogfood, clean-gate, source-rewrite, and
  reflection/modify current-vs-future wording
  - Result: no matches.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 31 - Twelfth Clean-Gate Restart Fix Sweep

The next fresh Stage 35 audit restart found stale live-stage documentation
claims, and local runtime review found sibling invalid-shape matrix helpers in
the same family as recent matvec/dense fixes. The gate did not count as clean
and remains at `0/3`.

Fixes landed in this increment:

- `ti2d_matmul` and `tf2d_matmul` now treat non-positive matrix dimensions as
  no-op cases before writing output matrices.
- `tf2d_row_sum` and `tf2d_col_sum` now treat non-positive rows or columns as
  no-op cases before writing destination vectors.
- `docs/HELIX_V1_FINAL_FEATURES.md` and
  `docs/HELIX_FINAL_PRODUCT_RESEARCH.md` now point current stage tracking at
  Stage 35 instead of stale Stage 34 wording.
- `docs/lang/hbs.md` now labels the 2026-05-04 HBS verification and 501-test
  count as historical snapshot evidence, not current Stage 35 gate evidence.
- `docs/APPROACH_A_PLAN.md` and `docs/APPROACH_A_DETAILED_PLAN.md` now carry
  historical/superseded banners for live stage tracking.
- Timed-out audit lanes were closed instead of being waited on indefinitely;
  they will be relaunched fresh after this fix.

Focused verification:

- Per-file stdlib parser sweep across `STDLIB_FILES`
  - Result: all stdlib files parsed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "negative_ti2d_matmul_shapes_do_not_write_outputs or negative_tf2d_matmul_shapes_do_not_write_outputs or negative_tf2d_row_col_sum_shapes_do_not_write_outputs or negative_dense_layer_f32_grad_x_shape_does_not_write_outputs or negative_tf1d_dot_with_offset_does_not_read_before_start or negative_tf1d_range_helpers_do_not_read_before_start or tensor_ti2d_matmul or tensor_f32_matmul or stdlib_tf2d_row_sum or stdlib_tf2d_col_sum" --tb=short`
  - Result: 11 passed.
- Docs scan for stale current-stage, historical test-count, stale tile-lowering,
  dogfood, clean-gate, and source-rewrite wording
  - Result: only an old Stage 34 progress-file reference remained; no Stage 35
    contradiction.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 32 - Thirteenth Clean-Gate Restart Fix Sweep

The next fresh Stage 35 audit restart found one direct PTX decode-diagnostic
gap, two runtime shape-boundary API gaps, and several stale documentation
overclaims. The gate did not count as clean and remains at `0/3`.

Fixes landed in this increment:

- Direct PTX now reads source files as UTF-8 explicitly and reports decode
  failures as clean exit-code-2 diagnostics instead of Python tracebacks.
- `tf1d_argmax_in_range` and `tf1d_sum_in_range` are now length-aware and reject
  `hi > n` before reading past the vector.
- `tf2d_diag` and `tf2d_trace` now take `rows, cols` and no-op / return zero
  unless the shape is non-empty and square.
- Approach A detailed-plan wording now agrees with its historical/superseded
  banner.
- Reflection/self-improvement docs now consistently describe the current
  reflective-cell / quote scaffold and leave runtime AST handles, real splice
  execution, and source rewrite/commit semantics as future work.
- The HBS and tutorial docs now avoid live "NOW" and unsupported competitor
  exclusivity wording.

Focused verification:

- Per-file stdlib parser sweep across `STDLIB_FILES`
  - Result: all stdlib files parsed.
- `python -m py_compile helixc\backend\ptx.py`
  - Result: passed.
- `python -m pytest -q helixc\tests\test_ptx.py -k "stage35_direct_ptx_cli_reports_encoding_error_without_traceback or stage35_direct_ptx_cli_reports_missing_file_without_traceback or stage35_direct_ptx_cli_bad_invocation_returns_two" --tb=short`
  - Result: 3 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "stdlib_tf1d_argmax_in_range or stdlib_tf1d_sum_in_range or negative_tf1d_range_helpers_do_not_read_before_start or overrange_tf1d_range_helpers_do_not_read_after_end or stdlib_tf2d_diag or stdlib_tf2d_trace or rectangular_tf2d_diag_trace_do_not_read_after_matrix or stdlib_tf2d_ones" --tb=short`
  - Result: 8 passed.
- Docs scan for stale reflection, historical-plan, HBS, broad tutorial,
  tile-lowering, dogfood, clean-gate, and source-rewrite wording
  - Result: only an old Stage 34 progress-file reference remained; no Stage 35
    contradiction.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 33 - Fourteenth Clean-Gate Restart Fix Sweep

The next fresh Stage 35 audit restart found reverse-AD match-alias gradient
loss, 2D tensor length overflow aliasing, non-pure `--emit-ptx` stdout, and
several stale documentation claims. The gate did not count as clean and remains
at `0/3`.

Fixes landed in this increment:

- Reverse-mode AD now fails closed for match pattern bindings that alias a
  differentiable scrutinee instead of silently returning zero gradients.
- `t2d_len` now returns zero on positive dimension overflow, and 2D constructors
  use an allocation length helper that reserves a sentinel slot for positive
  overflow so invalid handles do not alias the next allocation.
- `helixc.check --emit-ptx` now routes progress/status text to stderr so stdout
  starts with the PTX module.
- Tutorial and research docs now avoid stale example counts, unsupported
  competitor-exclusivity wording, and unmarked pre-Stage-29 snapshot context.

Focused verification:

- Per-file stdlib parser sweep across `STDLIB_FILES`
  - Result: all stdlib files parsed.
- `python -m py_compile helixc\check.py helixc\frontend\autodiff_reverse.py`
  - Result: passed.
- `python -m pytest -q helixc\tests\test_autodiff_reverse.py -k "match_bind_alias_of_scrutinee_fails_closed or match_with_pattern_shadow_is_zero or match_bool_propagates_per_arm" --tb=short`
  - Result: 3 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "overflow_t2d_len_and_alloc_do_not_alias_next_slot or negative_ti2d_matmul_shapes_do_not_write_outputs or negative_tf2d_matmul_shapes_do_not_write_outputs" --tb=short`
  - Result: 3 passed.
- `python -m pytest -q helixc\tests\test_cli.py -k "stage35_emit_ptx_stdout_starts_with_ptx_module or c117_emit_ptx_uses_kernel_attrs" --tb=short`
  - Result: 2 passed.
- Docs scan for stale tutorial/research/current-stage/canonical-plan/tile/dogfood
  and clean-gate wording
  - Result: only an old Stage 34 progress-file reference remained; no Stage 35
    contradiction.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 34 - Fifteenth Clean-Gate Restart Fix Sweep

The next fresh Stage 35 audit restart found tensor square-overflow reads,
remaining `--emit-ptx` stdout contamination on validation failures, and stale
historical research/work-queue wording. The gate did not count as clean and
remains at `0/3`.

Fixes landed in this increment:

- `tf2d_diag` and `tf2d_trace` now reject positive square shapes whose
  `rows * cols` length overflows before deriving diagonal offsets.
- `helixc.check --emit-ptx` now routes frontend validation diagnostics through
  stderr when stdout is reserved for PTX artifacts, including autotune and
  typecheck failures.
- `docs/research-log.md` and `docs/research/WORK_QUEUE.md` now mark old claims
  as historical snapshot evidence instead of current competitor or gate status.

Focused verification:

- Per-file stdlib parser sweep across `STDLIB_FILES`
  - Result: parsed 16 stdlib files.
- `python -m py_compile helixc\check.py`
  - Result: passed.
- `python -m pytest helixc\tests\test_codegen.py helixc\tests\test_cli.py -k "overflow_tf2d_diag_trace_do_not_read_after_matrix or rectangular_tf2d_diag_trace_do_not_read_after_matrix or overflow_t2d_len_and_alloc_do_not_alias_next_slot or stage35_emit_ptx_stdout_starts_with_ptx_module or stage35_emit_ptx_autotune_failure_stdout_is_empty or stage35_emit_ptx_typecheck_failure_stdout_is_empty or c117_emit_ptx_uses_kernel_attrs or c119_emit_ptx_rejects" -q`
  - Result: 15 passed.
- Docs scan for stale Phase 3 exclusivity, work-queue test-count, and old
  projection wording
  - Result: no matches.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 35 - Sixteenth Clean-Gate Restart Fix Sweep

The next fresh Stage 35 audit restart found a shallow reverse-AD match
dependency guard, one remaining `--emit-ptx` AD-warning stdout leak, and stale
roadmap wording about IO. The gate did not count as clean and remains at `0/3`.

Fixes landed in this increment:

- Reverse-mode AD now recursively checks compound scrutinees before allowing
  match pattern bindings, covering scalar and field-path expressions such as
  `x + 1.0` and `m.w + 1.0`.
- The final AD-warning drain now routes its summary to stderr when
  `--emit-ptx` reserves stdout for PTX artifacts.
- `docs/ROADMAP.md` now says basic diagnostic stdout and narrow file builtins
  exist, while richer capability-typed dataset/checkpoint IO remains Stage 35
  work.

Focused verification:

- `python -m py_compile helixc\check.py helixc\frontend\autodiff_reverse.py`
  - Result: passed.
- `python -m pytest helixc\tests\test_autodiff_reverse.py -q`
  - Result: 29 passed.
- `python -m pytest helixc\tests\test_cli.py -k "emit_ptx" -q`
  - Result: 16 passed.
- `python -m pytest helixc\tests\test_codegen.py -k "overflow_t2d_len_and_alloc_do_not_alias_next_slot or overrange_tf1d_range_helpers_do_not_read_after_end or rectangular_tf2d_diag_trace_do_not_read_after_matrix or overflow_tf2d_diag_trace_do_not_read_after_matrix" -q`
  - Result: 4 passed.
- Docs scan for stale IO, competitor-exclusivity, clean-gate, and work-queue
  projection wording
  - Result: the initial regex found no matches, but the restart 17 docs lane
    later found additional historical IO/work-queue wording. That follow-up is
    fixed in Increment 36.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 36 - Seventeenth Clean-Gate Restart Fix Sweep

The widened restart 17 audit protocol reported whole issue families instead of
stopping at the first bug. It found 2D shape-overflow gaps in tensor/NN helpers,
PTX artifact isolation gaps, and additional historical documentation wording.
The gate did not count as clean and remains at `0/3`.

Fixes landed in this increment:

- Public 2D tensor helpers now gate positive shapes through `t2d_len` before
  row-major loops or offset math. Covered families include matvec, matmul,
  transpose, row/column sum, and `tf2d_eye`.
- NN helpers now apply the same 2D overflow guard before dense, softmax,
  classifier, argmax, accuracy, and batch CE row-major work.
- `helixc.check --emit-ptx` and direct `helixc.backend.ptx` now filter to the
  kernel AST before lowering, so unrelated host AD functions do not block a
  clean PTX artifact.
- Direct `helixc.backend.ptx` now drains AD warnings to stderr on exit.
- Historical research/work-queue docs now phrase old IO and landed-ticket
  claims as snapshot wording, and this progress note acknowledges the restart
  17 follow-up docs finding.

Focused verification:

- Per-file stdlib parser sweep across `STDLIB_FILES`
  - Result: parsed 16 stdlib files.
- `python -m py_compile helixc\check.py helixc\backend\ptx.py`
  - Result: passed.
- `python -m pytest helixc\tests\test_codegen.py -k "stage35_public_2d_helpers_have_overflow_guards or tensor_2d_matvec or tensor_ti2d_matmul or negative_ti2d_matmul_shapes_do_not_write_outputs or negative_2d_matvec_shapes_do_not_write_outputs or negative_dense_layer_shapes_do_not_write_outputs or nn_dense_classifier_sgd_step_f32 or nn_softmax_rows_f32 or nn_argmax_rows_f32 or nn_accuracy_count_from_logits_f32 or nn_ce_loss_batch_f32 or stdlib_tf2d_eye or stdlib_tf2d_row_sum or stdlib_tf2d_col_sum or negative_tf2d_row_col_sum_shapes_do_not_write_outputs or stdlib_tf2d_transpose" -q`
  - Result: 24 passed.
- `python -m pytest helixc\tests\test_cli.py -k "emit_ptx" -q`
  - Result: 17 passed.
- `python -m pytest helixc\tests\test_ptx.py -q`
  - Result: 70 passed.
- Docs scan for stale historical IO/work-queue wording in current research docs
  - Result: no matches.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 37 - Eighteenth Clean-Gate Restart Fix Sweep

Restart 18 began from commit `017c873` with green smoke checks and broader
supporting verification, but the audit lanes found one remaining
artifact-isolation edge and one direct 2D accessor overflow sibling issue. The
gate did not count as clean and remains at `0/3`.

Fixes landed in this increment:

- `helixc.check --emit-ptx` now drains AD warnings before printing PTX. Normal
  warning policy still leaves the PTX artifact on stdout and sends diagnostics
  to stderr, while `-Wad=error` exits before printing any PTX artifact.
- A subprocess regression now locks the `--emit-ptx -Wad=error` behavior so
  failed AD-warning promotions cannot leak a valid-looking PTX module.
- Direct `ti2d_set/get` and `tf2d_set/get` now route through a checked
  `t2d_offset` helper, rejecting negative, out-of-row, multiplication-overflow,
  and final-addition-overflow offsets instead of aliasing unrelated arena slots.
- Behavioral regressions now prove overflowing, negative, and out-of-row direct
  2D accessor offsets do not write before/over the target tensor and return
  empty values.

Focused verification:

- `python -m pytest helixc\tests\test_cli.py -k "emit_ptx" -q`
  - Result: 18 passed.
- `python -m pytest helixc\tests\test_ptx.py -q`
  - Result: 70 passed.
- `python -m pytest helixc\tests\test_autodiff_reverse.py -q`
  - Result: 29 passed.
- `python -m pytest helixc\tests\test_codegen.py -k "stage35_2d_accessors_reject_overflow_offsets or stage35_2d_accessors_reject_negative_offsets or stage35_2d_accessors_reject_out_of_row_offsets or stage35_public_2d_helpers_have_overflow_guards" -q`
  - Result: 4 passed.
- `python -m pytest helixc\tests\test_codegen.py -k "tf2d or ti2d or dense_layer or softmax_rows_f32 or softmax_ce_grad_f32 or argmax_rows_f32 or accuracy_count_from_logits_f32 or ce_loss_batch_f32" -q`
  - Result: 37 passed.
- `python -m pytest helixc\tests\test_cli.py helixc\tests\test_ptx.py -q`
  - Result: 222 passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 stdlib files.
- `python -m py_compile helixc\check.py`
  - Result: passed.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 38 - Nineteenth Clean-Gate Restart Fix Sweep

Restart 19 began from commit `33a6b11` with green smoke checks and a clean
runtime lane, but the PTX/CLI and documentation lanes found remaining issues.
The gate did not count as clean and remains at `0/3`.

Fixes landed in this increment:

- `helixc.check --emit-ptx` now keeps artifact stdout empty for missing source
  paths and reports the invocation diagnostic on stderr.
- Public `helixc.check --emit-ptx` now lowers only the kernel-reachable AST in
  both strict and non-strict modes, so unrelated host-only AD helpers no longer
  surface as compiler-bug diagnostics before AD warning policy is applied.
- New regressions cover `--emit-ptx` missing-path stdout isolation, strict
  host-AD warning mode, and strict `-Wad=error` stdout isolation.
- `README.md`, `QUICKSTART.md`, and website draft facts/reference files now
  identify Stage 35 audit cleanup as current, keep clean gates at `0/3`, state
  that the production compiler is still Python-hosted `helixc`, use the
  299-byte hex0 root, and remove unsupported absolute comparison / shipped
  self-hosting / `3000+` test claims.

Focused verification:

- `python -m pytest helixc\tests\test_cli.py -k "stage35_emit_ptx_missing_path_keeps_stdout_empty or stage35_emit_ptx_strict_ignores_host_ad_function or stage35_emit_ptx_strict_wad_error_keeps_stdout_empty or stage35_emit_ptx_wad_error_does_not_emit_artifact or stage35_emit_ptx_ignores_host_ad_function" -q`
  - Result: 5 passed.
- `python -m pytest helixc\tests\test_cli.py -k "emit_ptx" -q`
  - Result: 21 passed.
- `python -m pytest helixc\tests\test_cli.py -q`
  - Result: 155 passed.
- `python -m pytest helixc\tests\test_ptx.py -q`
  - Result: 70 passed.
- `python -m py_compile helixc\check.py`
  - Result: passed.
- `python -m pytest helixc\tests --collect-only -q -p no:cacheprovider`
  - Result: 2,254 tests collected.
- Public-doc stale claim scan for old status/test/self-hosting/120-byte/3000+
  phrases
  - Result: no matches.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 39 - Twentieth Clean-Gate Restart Fix Sweep

Restart 20 began from commit `b776d2a` with green smoke checks and supporting
regression slices, but all three audit lanes found remaining issues. The gate
did not count as clean and remains at `0/3`.

Fixes landed in this increment:

- 2D tensors now carry lightweight row/column metadata through `t2d_new`, so
  direct `ti2d_*` and `tf2d_*` accessors can reject row-index out-of-bounds
  accesses instead of clobbering later arena allocations.
- `helixc.check --emit-ptx --strict` now validates full-program effects before
  kernel filtering, while keeping PTX artifact lowering restricted to
  kernel-reachable code.
- Direct `helixc.backend.ptx --strict` now mirrors the public CLI path for
  host-only AD helpers, and direct PTX now supports `-Wad=error`.
- Public docs, website draft docs, API contracts, and the historical plan
  snapshot were cleaned so old 120-byte, shipped self-hosting, `3000+` test,
  and absolute comparison claims do not remain on public status surfaces.

Focused verification:

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 stdlib files.
- `python -m py_compile helixc\check.py helixc\backend\ptx.py`
  - Result: passed.
- `python -m pytest helixc/tests/test_codegen.py -k "stage35_2d_accessors or public_2d_helpers" -q`
  - Result: 5 passed.
- `python -m pytest helixc/tests/test_cli.py -k "stage35_emit_ptx" -q`
  - Result: 11 passed.
- `python -m pytest helixc/tests/test_ptx.py -k "stage35_direct_ptx_cli" -q`
  - Result: 20 passed.
- `python -m pytest helixc/tests/test_codegen.py -k "t1d or t2d or ti2d or tf2d or tensor or stage35_2d" -q`
  - Result: 45 passed.
- `python -m pytest helixc/tests/test_cli.py -q`
  - Result: 156 passed.
- `python -m pytest helixc/tests/test_ptx.py -q`
  - Result: 72 passed.
- `python -m pytest helixc/tests/test_autodiff_reverse.py -q`
  - Result: 29 passed.
- `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider`
  - Result: 2,258 tests collected.
- Public-doc stale claim scan for old status/test/self-hosting/120-byte/3000+
  phrases
  - Result: no matches.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 40 - Twenty-First Clean-Gate Restart Fix Sweep

Restart 21 began from commit `c6273a9` with green smoke checks and supporting
regression slices, but Lane A and Lane C found remaining issues. Lane B did not
produce a completed PTX finding packet before the restart-21 fix sweep began, so
restart 22 must include a fresh PTX lane. The gate did not count as clean and
remains at `0/3`.

Fixes landed in this increment:

- Higher-level 2D, matrix, and NN helpers now require matching 2D arena metadata
  before treating a flat buffer as a matrix, closing the gap left after direct
  accessor row/column checks were added.
- `rev_backward` now pre-validates the full reverse-AD tape before mutating
  adjoints, so corrupt later tape entries cannot leave partial backward-state
  changes behind.
- Forward-mode and reverse-mode AD now have analytic chain rules for `__gelu`.
- `bce_loss_scalar` now routes through AD-known `__bce`, and forward/reverse AD
  chain rules cover BCE gradients directly.
- Public docs and website draft status surfaces now reflect the restart-21 fix
  verification count and the current 299-byte hex0 value.

Focused verification:

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 stdlib files.
- `python -m py_compile helixc\frontend\autodiff.py helixc\frontend\autodiff_reverse.py`
  - Result: passed.
- `python -m pytest helixc/tests/test_codegen.py -k "stage35_2d_helpers_reject_shape_metadata_mismatch or revad_backward_prevalidates_before_adj_mutation" -q`
  - Result: 2 passed.
- `python -m pytest helixc/tests/test_transcendentals.py -k "gelu or bce" -q`
  - Result: 4 passed.
- `python -m pytest helixc/tests/test_transcendentals.py helixc/tests/test_autodiff.py helixc/tests/test_autodiff_reverse.py -q`
  - Result: 97 passed.
- `python -m pytest helixc/tests/test_codegen.py -k "softmax_rows_f32 or argmax_rows_f32 or accuracy_count_from_logits_f32 or ce_loss_batch_f32 or softmax_ce_grad_f32 or dense_classifier_sgd_step_f32 or dense_layer_f32_grad_w or dense_layer_f32_grad_x" -q`
  - Result: 19 passed.
- `python -m pytest helixc/tests/test_codegen.py -k "t1d or t2d or ti2d or tf2d or tensor or stage35_2d or nn_ or dense_layer or softmax_rows_f32 or softmax_ce_grad_f32 or argmax_rows_f32 or accuracy_count_from_logits_f32 or ce_loss_batch_f32 or bce or gelu or revad" -q`
  - Result: 122 passed.
- `python -m pytest helixc/tests/test_cli.py -q`
  - Result: 156 passed.
- `python -m pytest helixc/tests/test_ptx.py -q`
  - Result: 72 passed.
- `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider`
  - Result: 2,264 tests collected.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Next Work

Likely follow-up slices:

- Run another fresh Stage 35 clean gate from the newest fixed commit.
- Keep PTX/tile/autotune expansion behind focused tests until the CPU AI/ML substrate is stronger.

## Increment 41 - Twenty-Second Clean-Gate Restart Fix Sweep

Restart 22 began from commit `c6dfb53` with green smoke/support checks, but
all three audit lanes found remaining Stage 35 issues. The gate did not count
as clean and Stage 35 remains at `0/3` clean gates.

Fixes landed in this increment:

- Strengthened 2D tensor metadata with an allocation footer and extent checks so
  forged/truncated matrix headers fail closed.
- Changed status-returning 2D and NN helpers to return `35001` on metadata
  mismatch or overflow instead of reporting success with stale output buffers.
- Hardened `rev_backward` so its tape must match its adjoint buffer and tape
  operands must only reference earlier entries.
- Added analytic forward and reverse AD rules for `__log_stable` plus f64
  exp/log/sin/cos/sqrt/relu/sigmoid/abs helpers.
- Fixed strict PTX and embedded-binary validation so unreachable differentiable
  helpers do not hide reachable host effect errors.
- Aligned direct PTX CLI warning policy with `helixc.check`, including
  `-Wad=warn`, `-Wad=error`, `-Wdeprecated`, and deprecated-warning emission.
- Updated public docs and website API contracts so live bootstrap facts are not
  mixed with target/roadmap byte counts.

Focused verification:

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 stdlib files.
- `python -m py_compile helixc\frontend\autodiff.py helixc\frontend\autodiff_reverse.py helixc\check.py helixc\backend\ptx.py helixc\backend\x86_64.py`
  - Result: passed.
- `python -m pytest helixc/tests/test_transcendentals.py helixc/tests/test_autodiff.py helixc/tests/test_autodiff_reverse.py -q`
  - Result: 101 passed.
- `python -m pytest helixc/tests/test_codegen.py -k "t1d or t2d or ti2d or tf2d or tensor or stage35_2d or nn_ or dense_layer or softmax_rows_f32 or softmax_ce_grad_f32 or argmax_rows_f32 or accuracy_count_from_logits_f32 or ce_loss_batch_f32 or bce or gelu or revad" -q`
  - Result: 125 passed.
- `python -m pytest helixc/tests/test_cli.py -q`
  - Result: 159 passed.
- `python -m pytest helixc/tests/test_ptx.py -q`
  - Result: 75 passed.
- `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider`
  - Result: 2,277 tests collected.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 42 - Twenty-Third Clean-Gate Restart Fix Sweep

Restart 23 began from commit `01f3d46` with green smoke/support checks. The
first audit batch timed out after five minutes, so it was closed and replaced
with a tighter scoped audit batch. The replacement lanes found blockers, so the
gate did not count as clean and Stage 35 remains at `0/3` clean gates.

Fixes landed in this increment:

- Shared the AD-known builtin set between inferred purity and builtin inlining
  skips so helper functions using `__log_stable` and f64 math helpers are
  treated consistently.
- Hardened reverse-AD tape validation so leaf records must keep both operand
  slots as `-1`, closing a forged-leaf gradient-drop path.
- Pruned unreachable differentiable-signature helpers for every host-lowering
  path in `helixc.check`, including `--emit-asm` and bare `--strict`.
- Updated bootstrap/status docs so target bootstrap links are not described as
  live and the pause handoff points at restart 23.

Focused verification:

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 stdlib files.
- `python -m py_compile helixc\frontend\autodiff.py helixc\frontend\autodiff_reverse.py helixc\check.py helixc\backend\ptx.py helixc\backend\x86_64.py`
  - Result: passed.
- `python -m pytest helixc/tests/test_transcendentals.py helixc/tests/test_autodiff.py helixc/tests/test_autodiff_reverse.py -q`
  - Result: 103 passed.
- `python -m pytest helixc/tests/test_codegen.py -k "t1d or t2d or ti2d or tf2d or tensor or stage35_2d or nn_ or dense_layer or softmax_rows_f32 or softmax_ce_grad_f32 or argmax_rows_f32 or accuracy_count_from_logits_f32 or ce_loss_batch_f32 or bce or gelu or revad" -q`
  - Result: 126 passed.
- `python -m pytest helixc/tests/test_cli.py -q`
  - Result: 161 passed.
- `python -m pytest helixc/tests/test_ptx.py -q`
  - Result: 75 passed.
- `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider`
  - Result: 2,282 tests collected.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is another fresh Stage 35 clean gate on this fixed commit.

## Increment 43 - Twenty-Fourth Clean-Gate Restart Fix Sweep

Restart 24 began from commit `a3874b1` with green smoke/support checks. Lane B
found no CLI/backend blocker, but Lane A and Lane C found issues, so the gate
did not count as clean and Stage 35 remains at `0/3` clean gates.

Fixes landed in this increment:

- Reverse-AD adjoint metadata now records the logical tape count at allocation,
  so `rev_seed` and `rev_grad` reject indices between current count and
  capacity.
- `rev_backward` now rejects tapes that grew after adjoints were allocated.
- Stage 35 progress ledger ordering is restored so increments are chronological
  and the newest work is visible at the tail of the file.
- Bootstrap comparison docs now label full self-hosting and full-from-hex
  behavior as targets rather than shipped behavior.
- The restart-21 pause handoff no longer presents a stale active restart as
  current truth.

Focused verification:

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 stdlib files.
- `python -m py_compile helixc\frontend\autodiff.py helixc\frontend\autodiff_reverse.py helixc\check.py helixc\backend\ptx.py helixc\backend\x86_64.py`
  - Result: passed.
- `python -m pytest helixc/tests/test_codegen.py -k "revad_seed_rejects_index_between_count_and_capacity or revad_grad_hides_index_between_count_and_capacity or revad_backward_rejects_tape_grown_after_adjoints_allocated or revad_seed_rejects_corrupt_adj_cap_metadata or revad_grad_rejects_corrupt_adj_cap_metadata or forged_leaf or foreign_adjoint_buffer or self_referential_operand or prevalidates_before_adj_mutation" -q`
  - Result: 9 passed.
- `python -m pytest helixc/tests/test_transcendentals.py helixc/tests/test_autodiff.py helixc/tests/test_autodiff_reverse.py -q`
  - Result: 103 passed.
- `python -m pytest helixc/tests/test_codegen.py -k "t1d or t2d or ti2d or tf2d or tensor or stage35_2d or nn_ or dense_layer or softmax_rows_f32 or softmax_ce_grad_f32 or argmax_rows_f32 or accuracy_count_from_logits_f32 or ce_loss_batch_f32 or bce or gelu or revad" -q`
  - Result: 129 passed.
- `python -m pytest helixc/tests/test_cli.py -q`
  - Result: 161 passed.
- `python -m pytest helixc/tests/test_ptx.py -q`
  - Result: 75 passed.
- `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider`
  - Result: 2,285 tests collected.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is restart 25, a fresh Stage 35 clean-gate audit from commit
  `8f56b5b`.

## Increment 44 - Twenty-Fifth Clean-Gate Restart Fix Sweep

Restart 25 began from commit `8f56b5b` with green smoke/support checks. All
three audit lanes found remaining Stage 35 issues, so the gate did not count as
clean and Stage 35 remains at `0/3` clean gates.

Fixes landed in this increment:

- Reverse-AD tapes now carry magic/footer validation, and tape-mutating APIs
  reject forged or truncated arena buffers before writing.
- Tensor allocation helpers are no longer marked `@pure`.
- AD helper inlining now inspects function bodies instead of blindly trusting
  explicit `@pure`, and let-flattening refuses to erase allocation/effecting
  expressions while still allowing pure containers such as `match`.
- PTX kernel tile lowering is validated before host DCE/FDCE can erase
  unsupported kernel operations in binary-emission paths.
- Non-strict PTX modes now report full-program effect warnings while still
  emitting kernel PTX.
- Public/status docs now reflect restart 25, 2,291 collected tests, and the
  current Python-hosted compiler boundary.

Focused verification:

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 stdlib files.
- `python -m py_compile helixc\frontend\autodiff.py helixc\check.py helixc\backend\ptx.py helixc\backend\x86_64.py`
  - Result: passed.
- `python -m pytest helixc/tests/test_transcendentals.py helixc/tests/test_autodiff.py helixc/tests/test_autodiff_reverse.py -q`
  - Result: 103 passed.
- `python -m pytest helixc/tests/test_codegen.py -k "t1d or t2d or ti2d or tf2d or tensor or stage35_2d or nn_ or dense_layer or softmax_rows_f32 or softmax_ce_grad_f32 or argmax_rows_f32 or accuracy_count_from_logits_f32 or ce_loss_batch_f32 or bce or gelu or revad or grad_rejects_allocator_let" -q`
  - Result: 131 passed.
- `python -m pytest helixc/tests/test_cli.py -q`
  - Result: 164 passed.
- `python -m pytest helixc/tests/test_ptx.py -q`
  - Result: 76 passed.
- `python -m pytest helixc/tests/test_effect_check.py -q`
  - Result: 34 passed.
- `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider`
  - Result: 2,291 tests collected.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is restart 26 as another fresh Stage 35 clean gate from the newest
  committed fix sweep.

## Increment 45 - Twenty-Sixth Clean-Gate Restart Fix Sweep

Restart 26 began from commit `45bf6ff` with green smoke/support checks. All
three audit lanes eventually returned useful findings, so the gate did not
count as clean and Stage 35 remains at `0/3` clean gates.

Fixes landed in this increment:

- `tf2d_zeros` is no longer marked `@pure`, matching the allocation behavior of
  the tensor allocator family.
- `rev_backward` now rejects tapes whose logical count changed in either
  direction after adjoint allocation, closing the remaining shrunk-tape
  validation gap.
- Strict proof-obligation effect diagnostics now prune unreachable
  differentiable-signature helpers before running AD/lowering, matching the
  normal output paths for dead `D<T>` helpers.
- The pause handoff now points to restart 26 as current history, and public docs
  no longer describe live bootstrap state as fully self-hosted.
- Public/status docs now reflect restart 26 and 2,294 collected tests.

Focused verification:

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed stdlib files.
- `python -m py_compile helixc\check.py`
  - Result: passed.
- `python -m pytest helixc/tests/test_codegen.py -k "revad_backward_rejects_tape_shrunk_after_adjoints_allocated or stage35_tensor_allocators_are_not_marked_pure or revad_backward_rejects_tape_grown_after_adjoints_allocated" -q`
  - Result: 3 passed, 849 deselected.
- `python -m pytest helixc/tests/test_cli.py -k "stage35_emit_proof_obligations_strict_ignores_dead_ad_helper or stage31_emit_proof_obligations_classifies_strict_effect_error or stage31_emit_proof_obligations_strict_effect_pass_failure_stays_json" -q`
  - Result: 3 passed, 162 deselected.
- `python -m pytest helixc/tests/test_transcendentals.py helixc/tests/test_autodiff.py helixc/tests/test_autodiff_reverse.py -q`
  - Result: 103 passed.
- `python -m pytest helixc/tests/test_codegen.py -k "t1d or t2d or ti2d or tf2d or tensor or stage35_2d or nn_ or dense_layer or softmax_rows_f32 or softmax_ce_grad_f32 or argmax_rows_f32 or accuracy_count_from_logits_f32 or ce_loss_batch_f32 or bce or gelu or revad or grad_rejects_allocator_let or stage35_tensor_allocators_are_not_marked_pure" -q`
  - Result: 133 passed, 719 deselected.
- `python -m pytest helixc/tests/test_cli.py -q`
  - Result: 165 passed.
- `python -m pytest helixc/tests/test_ptx.py -q`
  - Result: 76 passed.
- `python -m pytest helixc/tests/test_effect_check.py -q`
  - Result: 34 passed.
- `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider`
  - Result: 2,294 tests collected.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is restart 27 as another fresh Stage 35 clean gate from the newest
  committed fix sweep.

## Increment 46 - Twenty-Seventh Clean-Gate Restart Fix Sweep

Restart 27 began from commit `44c6b6a` with green support checks. The docs lane
returned clean, but the AD/tensor and backend/PTX lanes found multiple
remaining blockers, so the gate did not count as clean and Stage 35 remains at
`0/3` clean gates.

Fixes landed in this increment:

- Forward and reverse AD now reject side-effecting block final expressions
  instead of silently compiling final assignments into zero gradients.
- Reverse-AD adjoint buffers now record their owner tape, and `rev_backward`
  rejects spoofed foreign adjoint buffers even if a tape header is mutated to
  point at them.
- `t1d_new` now reserves an empty-allocation sentinel for zero/negative
  1D allocations; 1D setters and f32 dense-gradient helpers use capacity guards
  to avoid writing through empty output buffers.
- PTX validation paths run AD rewriting before full-program lowering so valid
  host `grad(...)` code no longer blocks `--emit-ptx` kernel emission.
- `-Wad=error` is checked before x86 artifact emission for `-o`, `--emit-asm`,
  and the direct `helixc.backend.x86_64` entry point.
- Kernel PTX embedding now requires pre-DCE kernel tile validation, preventing
  direct backend API callers from validating only after optimizer cleanup has
  erased unsupported dead kernel operations.
- Public/status docs now reflect restart 27 and 2,304 collected tests.

Focused verification:

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed stdlib files.
- `python -m py_compile helixc\frontend\autodiff.py helixc\backend\ptx.py helixc\backend\x86_64.py helixc\check.py`
  - Result: passed.
- `python -m pytest helixc/tests/test_codegen.py -k "side_effecting_final_assignment or spoofed_foreign_adjoint_buffer or negative_t1d_new_does_not_alias_next_allocation or dense_layer_f32_grad_x_rejects_empty_output_buffer or compile_module_to_elf_requires_pre_dce_kernel_validation" -q`
  - Result: 6 passed, 852 deselected.
- `python -m pytest helixc/tests/test_cli.py -k "emit_ptx_allows_valid_host_grad_call or wad_error_output_binary_does_not_write_artifact or wad_error_emit_asm_does_not_print_artifact or direct_x86_honors_wad_error_before_writing" -q`
  - Result: 4 passed, 165 deselected.
- `python -m pytest helixc/tests/test_transcendentals.py helixc/tests/test_autodiff.py helixc/tests/test_autodiff_reverse.py -q`
  - Result: 103 passed.
- `python -m pytest helixc/tests/test_codegen.py -k "t1d or t2d or ti2d or tf2d or tensor or stage35_2d or nn_ or dense_layer or softmax_rows_f32 or softmax_ce_grad_f32 or argmax_rows_f32 or accuracy_count_from_logits_f32 or ce_loss_batch_f32 or bce or gelu or revad or grad_rejects_allocator_let or side_effecting_final_assignment or negative_t1d_new or compile_module_to_elf_requires_pre_dce_kernel_validation" -q`
  - Result: 139 passed, 719 deselected.
- `python -m pytest helixc/tests/test_cli.py -q`
  - Result: 169 passed.
- `python -m pytest helixc/tests/test_ptx.py -q`
  - Result: 76 passed.
- `python -m pytest helixc/tests/test_effect_check.py -q`
  - Result: 34 passed.
- `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider`
  - Result: 2,304 tests collected.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is restart 28 as another fresh Stage 35 clean gate from the newest
  committed fix sweep.

## Increment 47 - Twenty-Eighth Clean-Gate Restart Fix Sweep

Restart 28 began from commit `3830869` with green support checks. The first
audit launch hit the agent thread limit, so stale completed agents were closed
and the audit was relaunched. All three relaunched lanes found remaining
blockers, so the gate did not count as clean and Stage 35 remains at `0/3`
clean gates.

Fixes landed in this increment:

- Public and handoff docs now avoid overclaiming the current bootstrap state:
  self-hosting and fully reproducible bootstrap parity remain roadmap targets,
  while the current production compiler is still Python-hosted `helixc`.
- `-Wad=error` now drains before `--emit-ir`, default clean stdout, and
  `--check-only` clean stdout so no artifact-like output appears before an AD
  warning is promoted to an error.
- The direct x86 backend now honors both `ad` and `deprecated` warning policies,
  drains AD warnings on typecheck failures, and promotes deprecated warnings
  before artifact writes.
- PTX tile validation now fails closed if DCE/FDCE has already touched an
  unvalidated kernel module, preserving the pre-DCE validation requirement for
  embedded kernel PTX.
- 1D tensors now have checked header/footer metadata while keeping the existing
  data-start handle API. Negative indices, positive OOB writes after later
  allocations, short f32 gradient buffers, and short f32 vector outputs fail
  closed.
- Reverse-AD adjoints now require immediate post-tape allocation and validate
  that layout invariant, closing the consistently forged adjoint metadata gap.
- Public/status docs now reflect restart 28 and 2,316 collected tests.

Focused verification:

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m py_compile helixc\frontend\autodiff.py helixc\frontend\autodiff_reverse.py helixc\check.py helixc\backend\ptx.py helixc\backend\x86_64.py helixc\ir\passes\dce.py helixc\ir\passes\fdce.py`
  - Result: passed.
- `python -m pytest helixc/tests/test_codegen.py -q -k "t1d or tf1d or ti1d or dense_layer_f32_grad or mse_loss_f32_grad"`
  - Result: 56 passed, 809 deselected.
- `python -m pytest helixc/tests/test_codegen.py -q -k "revad"`
  - Result: 30 passed, 835 deselected.
- `python -m pytest helixc/tests/test_cli.py -q -k "stage35 or wad or deprecated"`
  - Result: 31 passed, 143 deselected.
- `python -m pytest helixc/tests/test_codegen.py -q -k "stage35_compile_module_to_elf_requires_pre_dce_kernel_validation or ptx_in_binary or kernel_ptx"`
  - Result: 5 passed, 860 deselected.
- `python -m pytest helixc/tests/test_codegen.py -q -k "t1d or t2d or ti2d or tf2d or tensor or stage35_2d or nn_ or dense_layer or softmax_rows_f32 or softmax_ce_grad_f32 or argmax_rows_f32 or accuracy_count_from_logits_f32 or ce_loss_batch_f32 or bce or gelu or revad or grad_rejects_allocator_let or side_effecting_final_assignment or negative_t1d_new or compile_module_to_elf_requires_pre_dce_kernel_validation or mse_loss_f32_grad or wad"`
  - Result: 145 passed, 720 deselected.
- `python -m pytest helixc/tests/test_transcendentals.py helixc/tests/test_autodiff.py helixc/tests/test_autodiff_reverse.py helixc/tests/test_cli.py helixc/tests/test_ptx.py helixc/tests/test_effect_check.py -q`
  - Result: 387 passed.
- `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider`
  - Result: 2,316 tests collected.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is restart 29 as another fresh Stage 35 clean gate from the newest
  committed fix sweep.

## Increment 48 - Twenty-Ninth Clean-Gate Restart Fix Sweep

Restart 29 began from commit `585ae84` with green support checks. All three
fresh audit lanes found remaining blockers, so the gate did not count as clean
and Stage 35 remains at `0/3` clean gates.

Fixes landed in this increment:

- Tensor range helpers now reject positive out-of-bounds slices and inflated
  logical lengths.
- `tf1d_lerp` and NN f32 vector writers now reject short output buffers while
  still allowing valid interior sub-slices through the new `t1d_slice_ok`
  helper.
- Reverse-AD adjoint metadata now binds the tape footer and adjoint guards to
  owner, cap, count, and actual adjoint start, closing the immediate post-tape
  forged-slice path.
- Warning-as-error CLI paths now keep progress and diagnostic summaries off
  stdout and on stderr.
- Direct x86 early validation exits now drain pending AD warnings before exit.
- PTX validation now runs PTX emission before setting the kernel tile validation
  marker.
- Historical handoff/status docs now identify themselves as historical, point
  restart 29 at `585ae84`, and narrow current PTX/bootstrap claims.

Focused verification:

- Per-file stdlib parser sweep for `tensor.hx`, `nn.hx`, and
  `autodiff_reverse.hx`
  - Result: passed.
- `python -m py_compile helixc\check.py helixc\backend\ptx.py helixc\backend\x86_64.py helixc\tests\test_cli.py helixc\tests\test_codegen.py`
  - Result: passed.
- `python -m pytest helixc\tests\test_codegen.py -q -k "tf1d_dot_with_offset or tf1d_range_helpers or tf1d_lerp_rejects_short_output or gelu_layer_rejects_short_output or forged_immediate_adjoint_slice"`
  - Result: 9 passed, 861 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "kernel_helper_call or wad_error_output_binary or wad_error_emit_asm or wad_error_emit_ir or wad_error_default or wad_error_check_only or deprecated_error_default or direct_x86_drains_ad_warnings_on_deprecated_error"`
  - Result: 9 passed, 169 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "tf1d_dot_with_offset or tf1d_range_helpers or tf1d_lerp or gelu_layer or revad"`
  - Result: 40 passed, 830 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35 or wad or deprecated"`
  - Result: 35 passed, 143 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35 or direct_ptx or wad or deprecated"`
  - Result: 40 passed, 36 deselected.
- `python -m pytest helixc\tests\test_transcendentals.py helixc\tests\test_autodiff.py helixc\tests\test_autodiff_reverse.py helixc\tests\test_cli.py helixc\tests\test_ptx.py helixc\tests\test_effect_check.py -q`
  - Result: 391 passed.
- `python -m pytest helixc\tests\test_codegen.py -q -k "t1d or t2d or ti2d or tf1d or tf2d or tensor or stage35_2d or nn_ or dense_layer or softmax_rows_f32 or softmax_ce_grad_f32 or argmax_rows_f32 or accuracy_count_from_logits_f32 or ce_loss_batch_f32 or bce or gelu or mse_loss_f32_grad"`
  - Initial result: two over-strict sub-slice guard regressions.
  - Final result after `t1d_slice_ok`: 142 passed, 728 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "revad or grad_rejects_allocator_let or side_effecting_final_assignment or negative_t1d_new or compile_module_to_elf_requires_pre_dce_kernel_validation or ptx_in_binary or kernel_ptx or wad"`
  - Result: 40 passed, 830 deselected.
- `python -m pytest helixc\tests --collect-only -q -p no:cacheprovider`
  - Result: 2,325 tests collected.
- `git diff --check`
  - Result: passed.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Next step is restart 30 as another fresh Stage 35 clean gate from the newest
  committed fix sweep.

## Increment 49 - Thirtieth Clean-Gate Restart Fix Sweep

Restart 30 began from commit `b3f7796` with restart-29 pushed to `origin/main`.
A support check first exposed a one-line stdlib guard type regression, which was
fixed before the fresh audit lanes were launched. All three audit lanes then
found remaining blockers, so the gate did not count as clean and Stage 35
remains at `0/3` clean gates.

Fixes landed in this increment:

- `grad_rev_all` now reports failed reflection writes instead of returning
  success when `modify_f` / `modify_f64` rejects an invalid base handle.
- `grad`, `grad_rev`, and `grad_rev_all` now reject non-floating loss return
  types before generating derivative functions.
- Tensor 1D helpers now consistently use slice-aware validation, allowing valid
  interior slices while preventing reducers/accessors from reading footers or
  adjacent arena allocations.
- NN classifier, argmax/accuracy/CE, and metric helpers now check bias, input,
  output, and target buffers before reads or writes.
- Working/episodic memory helpers gained bounded object validation, and
  episodic accessors now reject negative indices.
- Reverse-AD adjoint guards now include a digest of the tape payload snapshot,
  so post-allocation tape mutation invalidates the backward pass.
- The direct x86 backend now rejects unknown flags and converts missing input,
  duplicate impl, output, chmod, and codegen failures into clean diagnostics.
- Continuation docs now reflect that restart 29 is closed, restart 30 found and
  fixed issues, and the live collection count is 2,339 tests.

Focused verification:

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
- Full `python -m pytest helixc\tests\test_codegen.py -q`
  - Result: timed out after 20 minutes with no useful partial output; stale
    pytest process was stopped.
- Four-way collected `test_codegen.py` chunk run
  - Result: timed out after 15 minutes per chunk in this environment; no
    chunk-specific failure was surfaced, and no stale pytest process remained.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Restart 30 was committed and pushed as `9efee28`.
- Restart 31 later began from `9efee28` and was committed as `fb9400d`; this
  entry is historical, so use the newest increment and live git state.

## Increment 50 - Thirty-First Clean-Gate Restart Fix Sweep

Restart 31 began from commit `9efee28` with restart-30 pushed to `origin/main`.
Support checks were green, then three fresh audit lanes found remaining
blockers. The gate did not count as clean and Stage 35 remains at `0/3` clean
gates.

Fixes landed in this increment:

- Reverse-AD adjoint allocation now stores a payload snapshot next to adjoint
  metadata; validation compares the live tape to that snapshot before seeds,
  gradients, or backward propagation use it.
- Attempts to append to a reverse-AD tape after adjoint allocation now poison
  the tape footer so later backward passes fail closed.
- f32 reducers and public NN `argmax` now validate logical slice lengths before
  reads.
- Working-memory and episodic-memory objects now carry magic and footer guards,
  preventing forged tensor buffers from passing as memory objects.
- Direct x86 CLI now converts invalid UTF-8 and strict-missing-stdlib failures
  into clean `error:` diagnostics without tracebacks.
- `--emit-ir` and `--emit-asm` warning summaries now stay on stderr so stdout
  remains artifact-only.
- Continuation docs now describe restart 30 as closed and restart 31 as the
  current fix sweep.

Focused verification:

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
    scoped live-suite collection for Stage 35.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Restart 31 was committed and pushed as `fb9400d`.
- Restart 32 began from `fb9400d` and found additional issues, so continue from
  the next increment and live git state.

## Increment 51 - Thirty-Second Clean-Gate Restart Fix Sweep

Restart 32 began from commit `fb9400d` with restart-31 pushed to `origin/main`.
Support checks were green, then three fresh audit lanes found remaining
runtime, CLI, and docs blockers. The gate did not count as clean and Stage 35
remains at `0/3` clean gates.

Fixes landed in this increment:

- 2D f32 output helpers now validate destination slices before writing.
- Reverse-AD metadata accessors now fail closed on forged tape handles.
- Working-memory validation now rejects negative ticks, and working/episodic
  mutators reject max-int ticks before incrementing.
- `helixc.check -o` and direct x86 now reject flag-shaped output paths instead
  of writing files named like flags.
- stdout emit modes now reject `-o` instead of silently ignoring it.
- `--emit-ast` now keeps diagnostics on stderr so stdout is artifact-only.
- Continuation docs now record restart 31 as committed/pushed and restart 32 as
  the current failed fix sweep.

Focused verification:

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m py_compile helixc\check.py helixc\backend\x86_64.py helixc\tests\test_cli.py helixc\tests\test_codegen.py`
  - Result: passed.
- `python -m pytest helixc\tests\test_codegen.py -q -k "tf2d_output_helpers_reject_short_destinations or revad_metadata_accessors_reject_fake_tape or agi_memory_rejects_corrupt_and_overflow_ticks"`
  - Result: 3 passed, 884 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "parse_args_output_rejects_flag_value or emit_ir_with_output_is_error or output_flag_value_rejected_without_writing or direct_x86_rejects_flag_shaped_output or main_emit_ast"`
  - Result: 5 passed, 185 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "tf2d or t2d or tensor or revad or agi_memory"`
  - Result: 78 passed, 809 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35 or emit_ast or emit_ir or emit_asm or emit_ptx or output or direct_x86"`
  - Result: 65 passed, 125 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35 or direct_ptx or wad or deprecated"`
  - Result: 40 passed, 36 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35"`
  - Result: 34 passed, 853 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - Result: 45 passed, 145 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - Result: 24 passed, 52 deselected.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,354 tests collected.
- `git diff --check`
  - Result: passed.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Restart 32 was committed and pushed as `5d8b4c4`.
- Restart 33 began from `5d8b4c4`; support checks were green, then three fresh
  audit lanes found remaining runtime, CLI/PTX, and docs blockers.

## Increment 52 - Thirty-Third Clean-Gate Restart Fix Sweep

Restart 33 began from commit `5d8b4c4` with restart-32 pushed to
`origin/main`. Baseline support checks were green:

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m py_compile helixc\check.py helixc\backend\x86_64.py helixc\tests\test_cli.py helixc\tests\test_codegen.py`
  - Result: passed.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35 or emit_ast or emit_ir or emit_asm or emit_ptx or output or direct_x86"`
  - Result: 65 passed, 125 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "tf2d or t2d or tensor or revad or agi_memory"`
  - Result: 104 passed, 783 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35 or direct_ptx or wad or deprecated"`
  - Result: 40 passed, 36 deselected.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,354 tests collected.
- `git diff --check`
  - Result: passed.

The fresh restart-33 audit was not clean. Findings:

- 2D integer/f32 matvec helpers validated the matrix but not the x/y vector
  slices.
- 2D integer/f32 setters did not fail loudly on invalid coordinates.
- Working-memory and episodic-memory validators accepted corrupted per-entry
  timestamps.
- `--check-only` could be combined with stdout artifact modes or `-o`, causing
  artifact requests to be silently ignored.
- Direct PTX warning policy parsing did not preserve argument order, so repeated
  `-Wad=...` flags could select the wrong policy.
- Direct PTX accepted conflicting `--stdlib` and `--no-stdlib` flags.
- Continuation docs still described restart 32 as needing commit/push after it
  had already landed as `5d8b4c4`.

Fixes landed in this increment:

- `ti2d_matvec` and `tf2d_matvec` now validate both input and output vector
  slices before reading or writing.
- `ti2d_set` and `tf2d_set` now return `t2d_error()` on invalid coordinates.
- `wm_ok` and `ep_ok` now reject negative or future per-entry timestamps.
- `helixc.check` now rejects conflicting stdlib flags and rejects
  `--check-only` combined with artifact modes or `-o`.
- Direct PTX now parses flags in CLI order, honors the last warning policy, and
  rejects conflicting stdlib flags.
- The handoff and restart docs now point at restart 33 from pushed
  `5d8b4c4`.

Verification:

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

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Restart 33 was committed and pushed as `09b692c`.
- Restart 34 began from `09b692c`; support checks were green, then three fresh
  audit lanes found AGI runtime, CLI/backend, and docs blockers.

## Increment 53 - Thirty-Fourth Clean-Gate Restart Fix Sweep

Restart 34 began from commit `09b692c` with restart-33 pushed to
`origin/main`. Baseline support checks were green:

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m py_compile helixc\check.py helixc\backend\x86_64.py helixc\backend\ptx.py helixc\tests\test_cli.py helixc\tests\test_codegen.py helixc\tests\test_ptx.py`
  - Result: passed.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35 or emit_ast or emit_ir or emit_asm or emit_ptx or output or direct_x86"`
  - Result: 68 passed, 125 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "tf2d or t2d or tensor or revad or agi_memory"`
  - Result: 79 passed, 811 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35 or direct_ptx or wad or deprecated"`
  - Result: 42 passed, 36 deselected.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,362 tests collected.
- `git diff --check`
  - Result: passed.

The fresh restart-34 audit was not clean. Findings:

- AGI world-model tables lacked magic/footer and bounds validation.
- AGI search containers and raw-buffer helpers accepted forged or short
  buffers.
- Failed deep unification could leave stale variable bindings.
- Prediction-error helpers could overflow into invalid metric values.
- `helixc.check` could print AD warning summaries to stdout during a different
  promoted warning failure.
- Direct x86 rejected explicit `--stdlib` unlike `helixc.check` and direct PTX.
- `helixc.check -o` and direct x86 wrote output artifacts non-atomically.
- Public/current docs still pointed to restart 33 as needing commit/push and
  older test counts.

Fixes landed in this increment:

- Added magic/footer validation and bounds-checked offsets for world-model
  tables.
- Added guard validation for BFS, visited-set, and priority-queue containers.
- Added slice validation for hill-climb, beam search, A*, and attention helpers.
- Added rollback on failed deep and table-driven unification.
- Saturated absolute and squared prediction errors at `2147483647`.
- Routed AD warning summaries to stderr whenever any warning policy is
  promoted to error.
- Added direct x86 `--stdlib` compatibility and stdlib conflict rejection.
- Reworked `helixc.check -o` and direct x86 output writes to temp-file plus
  replace, with cleanup on failure.
- Updated current docs and website facts to restart 34 and 2,372 collected
  tests.

Verification:

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

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Restart 34 is closed at pushed commit `fcfc20e`.
- Restart 35 began from `fcfc20e`; the fresh audit found more fixable issues,
  so the gate still does not count as clean.

## Increment 54 - Thirty-Fifth Clean-Gate Restart Fix Sweep

Restart 35 began from pushed commit `fcfc20e` with restart-34 closed. Baseline
support checks were clean before the audit lanes:

- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,372 tests collected at the `fcfc20e` baseline.

The fresh restart-35 audit was not clean. Findings:

- `hashmap_*` helpers accepted forged arena handles and mismatched capacities,
  allowing writes through non-hashmap slices.
- `wmt_set` accepted impossible next states outside the declared state range.
- `wml_predict` treated any three-slot arena slice as a valid linear world
  model.
- `helixc.check --emit-asm` and `helixc.check -o` reported a missing `main`
  function as an internal compiler bug instead of a user-facing codegen error.
- Current docs and website facts still pointed at restart 34 as the newest
  action.

Fixes in this increment:

- Added hashmap magic/header/footer validation while preserving the public
  `(start, cap)` carry-pair API.
- Guarded hashmap reads, writes, aggregations, and capacity helpers against
  forged handles and mismatched capacities.
- Rejected invalid table next states in `wmt_set`.
- Added magic/footer validation for `wml_new` / `wml_predict`.
- Routed missing-main x86 artifact failures to `helixc: codegen error` without
  the compiler-bug tagline.
- Updated public docs, handoff text, and website facts to restart 35 and the
  live 2,376-test collection.

Verification:

- `python -m py_compile helixc\check.py helixc\tests\test_cli.py helixc\tests\test_codegen.py`
  - Result: passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35_hashmap_rejects_forged or stage35_wmt_rejects_invalid or stage35_wml_rejects_forged or agi_wml_predict or agi_wmt_predict or agi_wmt_predict_or or stdlib_hashmap_put_get_round_trip or stdlib_hashmap_collision_probing"`
  - Result: 8 passed, 889 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "missing_main_is_user_codegen_error or main_emit_asm_traps_backend_error or main_o_traps_backend_error"`
  - Result: 4 passed, 196 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "hashmap or agi_ or wmt or wml or wm_prediction or tensor or revad or agi_memory or bfs or visited or pq or astar or attention or unify"`
  - Result: 152 passed, 745 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35 or emit_ast or emit_ir or emit_asm or emit_ptx or output or direct_x86 or missing_main"`
  - Result: 75 passed, 125 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35 or direct_ptx or wad or deprecated"`
  - Result: 42 passed, 36 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35"`
  - Result: 44 passed, 853 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - Result: 55 passed, 145 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - Result: 26 passed, 52 deselected.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,376 tests collected.
- `git diff --check`
  - Result: passed.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Restart 35 is a fix sweep, not a clean gate.
- Restart 35 was committed and pushed as `465a9b4`.
- Restart 36 began from `465a9b4`; the fresh audit found more fixable issues,
  so the gate still does not count as clean.

## Increment 55 - Thirty-Sixth Clean-Gate Restart Fix Sweep

Restart 36 began from pushed commit `465a9b4` with restart-35 closed. Baseline
support checks were green:

- `python -m py_compile helixc\check.py helixc\backend\x86_64.py helixc\backend\ptx.py helixc\tests\test_cli.py helixc\tests\test_codegen.py helixc\tests\test_ptx.py`
  - Result: passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35 or emit_ast or emit_ir or emit_asm or emit_ptx or output or direct_x86 or missing_main"`
  - Result: 75 passed, 125 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "hashmap or agi_ or wmt or wml or wm_prediction or tensor or revad or agi_memory or bfs or visited or pq or astar or attention or unify"`
  - Result: 152 passed, 745 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35 or direct_ptx or wad or deprecated"`
  - Result: 42 passed, 36 deselected.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,376 tests collected at the `465a9b4` baseline.

The fresh restart-36 audit was not clean. Findings:

- `hashmap_hash` could produce a negative bucket for an `INT_MIN` remainder,
  allowing a valid hashmap to write before its bucket region.
- `wmt_rollout` silently returned impossible start states from a bounded model.
- `helixc.check -o` wrote ELF artifacts without executable permission on POSIX.
- The recent x86 exception helper left unreachable legacy diagnostic blocks.
- Current docs still pointed at restart 35 / `fcfc20e` instead of restart 36.

Fixes in this increment:

- Normalized hashmap remainders after modulo so `hashmap_hash` never returns a
  negative bucket.
- Added rollout start-state validation for table-backed world models.
- Made `helixc.check -o` chmod temporary ELF artifacts to `0o755` before the
  atomic replace and clean up temp files if chmod fails.
- Removed unreachable legacy exception-print blocks after the x86 diagnostic
  helper return.
- Updated current docs, handoff text, and website facts to restart 36 and the
  live 2,379-test collection.

Verification:

- `python -m py_compile helixc\check.py helixc\tests\test_cli.py helixc\tests\test_codegen.py`
  - Result: passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m pytest helixc\tests\test_codegen.py -q -k "hashmap_hash_int_min or wmt_rollout_rejects_invalid or stage35_hashmap_rejects_forged or agi_wmt_rollout"`
  - Result: 4 passed, 895 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "main_o_writes_file or check_output_chmod_failure or check_output_atomic_replace_failure or missing_main_is_user_codegen_error"`
  - Result: 5 passed, 196 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "hashmap or agi_ or wmt or wml or wm_prediction or tensor or revad or agi_memory or bfs or visited or pq or astar or attention or unify"`
  - Result: 154 passed, 745 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35 or emit_ast or emit_ir or emit_asm or emit_ptx or output or direct_x86 or missing_main"`
  - Result: 76 passed, 125 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35 or direct_ptx or wad or deprecated"`
  - Result: 42 passed, 36 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35"`
  - Result: 46 passed, 853 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - Result: 56 passed, 145 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - Result: 26 passed, 52 deselected.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,379 tests collected.
- `git diff --check`
  - Result: passed.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Restart 36 is a fix sweep, not a clean gate.
- Restart 36 was committed and pushed as `0734ebf`.
- Restart 37 began from `0734ebf`; the fresh audit found more fixable issues,
  so the gate still does not count as clean.

## Increment 56 - Thirty-Seventh Clean-Gate Restart Fix Sweep

Restart 37 began from pushed commit `0734ebf` with restart-36 closed. Baseline
support checks were green:

- `python -m py_compile helixc\check.py helixc\backend\x86_64.py helixc\backend\ptx.py helixc\tests\test_cli.py helixc\tests\test_codegen.py helixc\tests\test_ptx.py`
  - Result: passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35 or emit_ast or emit_ir or emit_asm or emit_ptx or output or direct_x86 or missing_main"`
  - Result: 76 passed, 125 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "hashmap or agi_ or wmt or wml or wm_prediction or tensor or revad or agi_memory or bfs or visited or pq or astar or attention or unify"`
  - Result: 154 passed, 745 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35 or direct_ptx or wad or deprecated"`
  - Result: 42 passed, 36 deselected.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,379 tests collected at the `0734ebf` baseline.

The fresh restart-37 audit was not clean. Findings:

- `wmt_rollout` failed open for forged world-model handles and short action
  buffers by returning the starting state.
- `hashmap_avg_value_x100` could overflow in `sum * 100` before dividing even
  when the mathematically correct scaled average still fit in `i32`.
- Current docs still used exact active-restart wording that became stale after
  each commit.
- The CLI/backend/PTX lane was clean.

Fixes in this increment:

- Made `wmt_rollout` return `-1` for negative steps, invalid model handles,
  invalid action slices, and invalid start states.
- Reworked `hashmap_avg_value_x100` to compute the scaled average through
  `i64`, with saturation back to the `i32` return range.
- Added regressions for forged/short rollout inputs and large-but-valid scaled
  hashmap averages.
- Reworded current-facing status and handoff surfaces to point continuations at
  live `git log -1 --oneline` plus the ledger tail, reducing exact-hash status
  churn between restarts.

Verification:

- `python -m py_compile helixc\tests\test_codegen.py`
  - Result: passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m pytest helixc\tests\test_codegen.py -q -k "wmt_rollout_rejects_forged or wmt_rollout_rejects_invalid or hashmap_avg_value_x100_avoids or stdlib_hashmap_avg_value_x100 or agi_wmt_rollout"`
  - Result: 5 passed, 896 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "hashmap or agi_ or wmt or wml or wm_prediction or tensor or revad or agi_memory or bfs or visited or pq or astar or attention or unify"`
  - Result: 156 passed, 745 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35 or emit_ast or emit_ir or emit_asm or emit_ptx or output or direct_x86 or missing_main"`
  - Result: 76 passed, 125 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35 or direct_ptx or wad or deprecated"`
  - Result: 42 passed, 36 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35"`
  - Result: 48 passed, 853 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - Result: 56 passed, 145 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - Result: 26 passed, 52 deselected.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,381 tests collected.
- `git diff --check`
  - Result: passed.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Restart 37 is a fix sweep, not a clean gate.

## Increment 57 - Thirty-Eighth Clean-Gate Restart Fix Sweep

Restart 38 began from pushed commit `40e64ca` with restart-37 closed. Baseline
support checks were green:

- `python -m py_compile helixc\check.py helixc\backend\x86_64.py helixc\backend\ptx.py helixc\tests\test_cli.py helixc\tests\test_codegen.py helixc\tests\test_ptx.py`
  - Result: passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35 or emit_ast or emit_ir or emit_asm or emit_ptx or output or direct_x86 or missing_main"`
  - Result: 76 passed, 125 deselected at the restart-38 baseline support check.
- `python -m pytest helixc\tests\test_codegen.py -q -k "hashmap or agi_ or wmt or wml or wm_prediction or tensor or revad or agi_memory or bfs or visited or pq or astar or attention or unify"`
  - Result: 156 passed, 745 deselected at the restart-38 baseline support check.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35 or direct_ptx or wad or deprecated"`
  - Result: 42 passed, 36 deselected at the restart-38 baseline support check.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,381 tests collected at the `40e64ca` baseline.

The fresh restart-38 audit was not clean. Findings:

- `wmt_rollout` still failed open for invalid action values and unset
  transitions by keeping the previous state instead of returning `-1`.
- `hashmap_avg_value_x100` widened after calling `hashmap_sum_values`, but
  `hashmap_sum_values` itself accumulated in `i32`, so multi-entry overflow
  could still corrupt the scaled average before saturation.
- `helixc.check -o` and direct `python -m helixc.backend.x86_64` allowed the
  source file to also be the output path, so a successful compile could replace
  `.hx` source text with ELF bytes.
- The docs/status lane was clean.

Fixes in this increment:

- Made `wmt_rollout` fail closed with `-1` when a rollout step has an invalid
  action or an unset transition.
- Reworked `hashmap_avg_value_x100` to sum occupied bucket values in `i64`
  before scaling and saturating back into the `i32` return range.
- Added output path guards to both `helixc.check -o` and the direct x86 backend
  CLI so the compiler rejects source-as-output before writing any artifact.
- Added regressions for invalid/unset rollout steps, positive and negative
  pre-scale hashmap average overflow, and both source-as-output CLI paths.
- Updated current-facing status docs and website facts to the restart-38 test
  count.

Verification:

- `python -m py_compile helixc\check.py helixc\backend\x86_64.py helixc\tests\test_cli.py helixc\tests\test_codegen.py`
  - Result: passed.
- `python -m pytest helixc\tests\test_cli.py -q -k "source_as_output or chmod_failure or flag_shaped_output"`
  - Result: 5 passed, 198 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "wmt_rollout_rejects_invalid_action_and_unset_transition or hashmap_avg_value_x100_sums_in_i64_before_scaling or hashmap_avg_value_x100_negative_sum_saturates or hashmap_avg_value_x100_avoids_predivide_overflow"`
  - Result: 4 passed, 900 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35 or agi or hashmap or tensor"`
  - Result: 158 passed, 746 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - Result: 58 passed, 145 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - Result: 26 passed, 52 deselected.
- `python -m pytest helixc\tests\test_cli.py -q`
  - Result: 203 passed.
- `python -m pytest helixc\tests\test_ptx.py -q`
  - Result: 78 passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,386 tests collected.
- `python -m pytest --collect-only -q`
  - Result: not used as gate evidence because the unscoped command collected
    the read-only Stage 30 snapshot and hit duplicate-module import mismatch
    errors. Scoped live-suite collection above is the meaningful count.
- `git diff --check`
  - Result: passed.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Restart 38 is a fix sweep, not a clean gate.
- Next step is restart 39 as another fresh Stage 35 clean gate from the newest
  pushed HEAD.

## Increment 58 - Thirty-Ninth Clean-Gate Restart Fix Sweep

Restart 39 began from pushed commit `258b7a6` with restart-38 closed. Baseline
support checks were green:

- `python -m py_compile helixc\check.py helixc\backend\x86_64.py helixc\backend\ptx.py helixc\tests\test_cli.py helixc\tests\test_codegen.py helixc\tests\test_ptx.py`
  - Result: passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35 or agi or hashmap or tensor"`
  - Result: 158 passed, 746 deselected at the restart-39 baseline support check.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - Result: 58 passed, 145 deselected at the restart-39 baseline support check.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - Result: 26 passed, 52 deselected at the restart-39 baseline support check.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,386 tests collected at the `258b7a6` baseline.

The fresh restart-39 audit was not clean. Findings:

- Direct x86 still accepted a flag-shaped input position. If a real file named
  `--no-stdlib` existed, `python -m helixc.backend.x86_64 --no-stdlib victim.hx`
  could treat the flag-file as source and overwrite `victim.hx`.
- Direct x86 returned exit code `1` for bad invocation / input-environment
  failures where `helixc.check` and direct PTX use bad-invocation code `2`.
- `wmt_predict_or` and `wmt_is_self_loop` failed open on invalid lookups, and
  table-backed world-model reads could accept corrupted next states outside
  the declared state range.
- `t1d_slice_ok` could spend a huge amount of time scanning backward from a
  forged positive start instead of first checking the arena length.
- AGI match raw-buffer helpers accepted forged out-of-bounds buffers as
  zero-filled data.
- Public `hashmap_sum_values` still accumulated in `i32`, so large maps could
  wrap even though `hashmap_avg_value_x100` had been hardened.
- Website code samples claimed 20 samples while the file contains 30.

Fixes in this increment:

- Direct x86 now rejects flag-shaped input paths before reading or writing,
  and bad input/environment exits use code `2`.
- Table-backed world-model prediction helpers now fail closed on invalid
  lookups and corrupted next states; rollout also rejects corrupted next-state
  cells.
- `t1d_slice_ok` rejects starts beyond the current arena length before entering
  the backward scan.
- `bag_similarity`, `bag_difference`, `bag_count_unique`, and
  `sequence_match` require valid `t1d` slices before reading.
- `hashmap_sum_values` now accumulates in `i64` and saturates into the `i32`
  return range.
- `sequence_match` was normalized to the same shape as `ti1d_eq_count` after
  verification exposed a direct-x86 stdlib hash-cons collision between
  alpha-equivalent blocks.
- The website code sample count now says 30.
- Added regressions for the direct-x86 flag-input overwrite class, WMT invalid
  / corrupted states, huge forged tensor starts, forged AGI match buffers, and
  hashmap sum saturation.

Verification:

- `python -m py_compile helixc\backend\x86_64.py helixc\tests\test_cli.py helixc\tests\test_codegen.py`
  - Result: passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m pytest helixc\tests\test_cli.py -q -k "flag_shaped_input or missing_input_reports_clean_error or invalid_utf8_reports_clean_error or missing_strict_stdlib_reports_clean_error or source_as_output"`
  - Result: 6 passed, 198 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "wmt_predictors_reject_invalid_and_corrupt_states or t1d_slice_ok_rejects_huge_forged_start_fast or agi_match_helpers_reject_forged_slices or hashmap_sum_values_saturates_on_overflow"`
  - Result: 4 passed, 904 deselected.
- `python -m pytest helixc\tests\test_cli.py::test_stage35_direct_x86_accepts_stdlib_compat_flag -q`
  - Result: 1 passed.
- `python -m pytest helixc\tests\test_codegen.py -q -k "sequence_match or agi_match_helpers_reject_forged_slices"`
  - Result: 2 passed, 906 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35 or agi or hashmap or tensor"`
  - Result: 162 passed, 746 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - Result: 59 passed, 145 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - Result: 26 passed, 52 deselected.
- `python -m pytest helixc\tests\test_cli.py -q`
  - Result: 204 passed.
- `python -m pytest helixc\tests\test_ptx.py -q`
  - Result: 78 passed.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,391 tests collected.
- `git diff --check`
  - Result: passed.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Restart 39 is a fix sweep, not a clean gate.
- Next step is restart 40 as another fresh Stage 35 clean gate from the newest
  pushed HEAD.

## Increment 59 - Fortieth Clean-Gate Restart Fix Sweep

Restart 40 began from a clean baseline after restart 39:

- `python -m py_compile helixc\check.py helixc\tests\test_cli.py helixc\tests\test_codegen.py`
  - Result: passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35 or agi or hashmap or tensor"`
  - Result: 162 passed, 746 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - Result: 59 passed, 145 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - Result: 26 passed, 52 deselected.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,391 tests collected.

Restart 40 audit findings:

- `hier_count_achieved` read outside the achieved table when subgoal ids
  referenced cells beyond the table slice.
- `ensemble_mean`, `ensemble_uncertainty`, and `ensemble_argmax` accepted
  forged prediction slices; mean and uncertainty also had `i32` overflow
  edges.
- `wmt_count_set` counted corrupt next-state cells as if they were valid set
  transitions.
- `hashmap_increment` and `hashmap_sum_keys` still had overflow sibling
  surfaces.
- `helixc.check -l` could swallow a following flag as a library name.
- Website README status still described the code sample surface as 20 samples
  after the reference had grown to 30.

Fixes in this increment:

- `hier_count_achieved` now validates the subgoal-id slice and every achieved
  table access before reading.
- Ensemble helpers now reject forged slices; mean and uncertainty accumulate in
  `i64` and saturate back to `i32` where needed.
- `wmt_count_set` now counts only next states in the declared state range.
- `hashmap_increment` and `hashmap_sum_keys` now use `i64` intermediates and
  saturate into the `i32` return range.
- `helixc.check -l` now rejects flag-shaped separate and attached library
  values.
- Website README status now points at the 30-sample code sample file.
- Added regressions for all fixed surfaces.

Verification:

- `python -m py_compile helixc\check.py helixc\tests\test_cli.py helixc\tests\test_codegen.py`
  - Result: passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m pytest helixc\tests\test_cli.py -q -k "lib_rejects_flag_value or parse_args_lib or output_rejects_flag_value"`
  - Result: 4 passed, 201 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "wmt_count_set_ignores_corrupt or hier_count_achieved_rejects_forged or ensemble_rejects_forged or hashmap_sum_keys_saturates or hashmap_increment_saturates"`
  - Result: 5 passed, 908 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35 or agi or hashmap or tensor"`
  - Result: 167 passed, 746 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - Result: 60 passed, 145 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - Result: 26 passed, 52 deselected.
- `python -m pytest helixc\tests\test_cli.py -q`
  - Result: 205 passed.
- `python -m pytest helixc\tests\test_ptx.py -q`
  - Result: 78 passed.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,397 tests collected.
- `git diff --check`
  - Result: passed.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Restart 40 is a fix sweep, not a clean gate.
- Next step is restart 41 as another fresh Stage 35 clean gate from the newest
  pushed HEAD.

## Increment 60 - Forty-First Clean-Gate Restart Fix Sweep

Restart 41 began from pushed commit `1b7064e` after restart 40. Baseline support
checks:

- `git status --short --branch`
  - Result: clean at `1b7064e`.
- `python -m py_compile helixc\check.py helixc\backend\x86_64.py helixc\tests\test_cli.py helixc\tests\test_codegen.py helixc\tests\test_ptx.py`
  - Result: passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35 or agi or hashmap or tensor"`
  - Result: 167 passed, 746 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - Result: 60 passed, 145 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - Result: 26 passed, 52 deselected.

Restart 41 audit findings:

- Forged AGI tree offsets could match as valid nodes because shallow equality
  and unification read raw arena slots without first proving the node offset
  covered four live arena cells.
- Forged binding tables could advertise counts beyond the documented 32-entry
  capacity and make `bindings_get` read outside the binding table.
- `wml_predict` used plain `i32` arithmetic, so large linear predictions could
  wrap instead of saturating.
- `t1d_slice_ok(ptr, 0)` accepted forged starts without proving the start was
  inside a real tensor slice.
- Source-required non-artifact CLI modes such as `--check-only` and `-o out`
  with no source printed help on stdout instead of `source path required` on
  stderr.
- Website reference stats still had a stale exact audit-pass count and
  2026-05-15 review marker.

Fixes in this increment:

- Added `tree_node_ok` and guarded shallow tree equality, tree hashing,
  variable checks, and unification entry points before arena reads.
- Added binding storage/count checks; `bindings_get` now fails closed for
  counts outside `0..32`, and `bindings_set` rejects invalid or negative
  counts before writing.
- `wml_predict` now computes in `i64` and saturates into the `i32` return
  range.
- Zero-length `t1d` slice/range validation now requires a real backing tensor
  handle instead of accepting arbitrary forged starts.
- Missing source for non-artifact modes now reports `source path required` on
  stderr with empty stdout.
- Website stats wording now avoids stale exact audit-pass counts and marks the
  current-status review as 2026-05-16.
- Added regressions for every fixed surface.

Verification:

- `python -m py_compile helixc\check.py helixc\tests\test_cli.py helixc\tests\test_codegen.py`
  - Result: passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m pytest helixc\tests\test_cli.py -q -k "source_required_modes or emit_ptx_missing_path"`
  - Result: 2 passed, 204 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "wml_predict_saturates or tree_and_unify_reject_forged or t1d_zero_length_validators or bindings_get_rejects_forged"`
  - Result: 4 passed, 913 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35 or agi or hashmap or tensor"`
  - Result: 171 passed, 746 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - Result: 61 passed, 145 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - Result: 26 passed, 52 deselected.
- `python -m pytest helixc\tests\test_cli.py -q`
  - Result: 206 passed.
- `python -m pytest helixc\tests\test_ptx.py -q`
  - Result: 78 passed.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,402 tests collected.
- `git diff --check`
  - Result: passed.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Restart 41 is a fix sweep, not a clean gate.
- Next step is restart 42 as another fresh Stage 35 clean gate from the newest
  pushed HEAD.

## Increment 61 - Forty-Second Clean-Gate Restart Fix Sweep

Restart 42 began from pushed commit `e512418` after restart 41. Baseline support
checks:

- `git status --short --branch`
  - Result: clean at `e512418`.
- `python -m py_compile helixc\check.py helixc\backend\x86_64.py helixc\tests\test_cli.py helixc\tests\test_codegen.py helixc\tests\test_ptx.py`
  - Result: passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - Result: 61 passed, 145 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - Result: 26 passed, 52 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35 or agi or hashmap or tensor"`
  - Result: 171 passed, 746 deselected.

Restart 42 audit findings:

- Forged in-bounds binding handles could still pass `bindings_storage_ok`
  because the check proved only cell coverage, not object identity.
- `bindings_rewind` could mutate a forged object because it skipped the
  storage check.
- Forged in-bounds tensor slices could impersonate tree nodes because
  `tree_node_ok` proved only four readable cells.
- Raw tree accessors still returned arena values for invalid offsets, which
  failed open to zero through `__arena_get`.
- CLI/backend lane and docs/status lane were clean.

Fixes in this increment:

- Tree nodes now carry a magic header and footer, and `tree_node_ok` validates
  both before tree equality, hashing, variable checks, and unification can read
  payload cells.
- Raw tree accessors now return a sentinel invalid value for invalid handles.
- Binding tables now carry a magic header and footer, and
  `bindings_storage_ok` validates both before reads or writes.
- `bindings_rewind` now rejects forged binding handles before mutation.
- Added regressions for forged in-bounds tree handles, raw invalid tree
  accessors, forged in-bounds binding tables, and forged binding rewinds.

Verification:

- `python -m py_compile helixc\tests\test_codegen.py`
  - Result: passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m pytest helixc\tests\test_codegen.py -q -k "tree_helpers_reject_forged or bindings_reject_forged or tree_and_unify_reject_forged or bindings_get_rejects_forged"`
  - Result: 4 passed, 915 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "stage35 or agi or hashmap or tensor"`
  - Result: 173 passed, 746 deselected.
- `python -m pytest helixc\tests\test_cli.py -q -k "stage35"`
  - Result: 61 passed, 145 deselected.
- `python -m pytest helixc\tests\test_ptx.py -q -k "stage35"`
  - Result: 26 passed, 52 deselected.
- `python -m pytest helixc\tests\test_cli.py -q`
  - Result: 206 passed.
- `python -m pytest helixc\tests\test_ptx.py -q`
  - Result: 78 passed.
- `python -m pytest helixc\tests --collect-only -q`
  - Result: 2,404 tests collected.

Clean-gate status:

- Stage 35 clean gates remain `0/3`.
- Restart 42 is a fix sweep, not a clean gate.
- Next step is restart 43 as another fresh Stage 35 clean gate from the newest
  pushed HEAD.
