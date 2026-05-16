# Stage 35 Restart 62 — Lane A (Runtime/Safety) Audit Report

**HEAD**: `c697f3d` (Fix Stage 35 sixty-first restart findings)
**Date**: 2026-05-16
**Lane**: A (Runtime/Safety)
**Discipline**: combined audit-and-fix. Findings applied in the same
commit as this doc.

## Findings (2 MEDIUM = 2 findings, Family 2 — optimizer NaN-fail-closed)

### A1 MEDIUM — `sgd_f32_step` NaN-fail-closed (Family 2 — optimizer)

`helixc/stdlib/nn.hx sgd_f32_step` overwrote `w[i]` unconditionally
with `w[i] - lr * g[i]`. A NaN gradient slot (or NaN lr) propagated
NaN into the weight buffer, poisoning every subsequent forward pass.
Sibling of `adam_f32_step` (restart 50 A2) and
`dense_classifier_sgd_step_f32` (restart 48 A2) — both already
NaN-fail-closed.

**Fix**: compute `new_w = w[i] - lr * g[i]`; only write back if
`new_w == new_w` (NaN-skip). Mirrors the adam discipline.

**Canary**: `test_stage35_restart62_sgd_f32_step_nan_fails_closed`.

### A2 MEDIUM — `momentum_step` NaN-fail-closed (Family 2 — optimizer)

`helixc/stdlib/nn.hx momentum_step` overwrote both `v[i]` and `w[i]`
unconditionally. A NaN gradient was permanently latched into the
velocity buffer (which carries forward across steps), so a single
NaN gradient corrupted momentum SGD irrecoverably. Sibling of
`adam_f32_step` (restart 50 A2) and the just-fixed `sgd_f32_step`
(restart 62 A1) — momentum is the carrier-of-state mirror of SGD.

**Fix**: compute `new_v` and `new_w`; only write back both if neither
is NaN. Preserves both the velocity buffer and the weight on a NaN
gradient.

**Canary**: `test_stage35_restart62_momentum_step_nan_fails_closed`.

## CLEAN spot-checks (no findings)

- `adam_f32_step` (restart 50 A2) — already fail-closed.
- `dense_classifier_sgd_step_f32` (restart 48 A2) — already fail-closed.
- `clip_grad_norm_f32` (restart 51 A2) — already NaN-and-zero guarded.
- `softmax_layer` (restart 48 A2) — uniform-distribution-on-fail.
- `layer_norm_f32` (restart 47 A2) — zeros-on-fail.
- `dropout_f32` — per-element copy; NaN propagation is documented
  garbage-in/garbage-out for stochastic regularization.
- `mse_loss_f32` / `mae_loss_f32` (restart 58 A6) — NaN-skip.
- `accuracy_count_from_logits_f32` (restart 61 A2) — seen=0 sentinel.

## Carry-forward to restart 63

None — restart 62 closed both findings in the same commit.
