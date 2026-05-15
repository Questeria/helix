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
- Add a small training-stability helper such as gradient-norm clipping in the
  Helix stdlib.
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
