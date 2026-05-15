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

## Next Work

Likely follow-up slices:

- Wire field-path leaves into the higher-level `grad_rev_all`/pytree model
  surface.
- Keep PTX/tile/autotune expansion behind focused tests until the CPU AI/ML
  substrate is stronger.

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
