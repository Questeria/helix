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
