# Stage 35 Restart 61 — Lane A (Runtime/Safety) Audit Report

**HEAD**: `8f774a4` (Linter test additions before restart 61)
**Date**: 2026-05-16
**Lane**: A (Runtime/Safety)
**Discipline**: applied audit. Findings landed in commit `c697f3d`
("Fix Stage 35 sixty-first restart findings"). Lane doc landed
retroactively in restart 62.

## Findings (2 HIGH + 1 MEDIUM)

### A1 HIGH — `tf1d_running_sum` NaN-skip (Family 2 — f32 reduction NaN-skip)

`helixc/stdlib/tensor.hx tf1d_running_sum` accumulated NaN into the
running prefix sum without skipping. One NaN slot poisoned every
subsequent slot (NaN + anything = NaN). Sibling of `tf1d_sum`
(restart 57 A1), `tf1d_sum_in_range` (restart 58 A1), `tf2d_row_sum`
(restart 58 A5).

**Fix**: per-element `if v == v` guard around the accumulator update.

**Canary**: `test_stage35_restart61_tf1d_running_sum_nan_skip_fails_closed`.

### A2 HIGH — `accuracy_count_from_logits_f32` NaN-at-col-0 (Family 2 — argmax NaN-init)

`helixc/stdlib/nn.hx accuracy_count_from_logits_f32` bare-initialized
`best_val = arena_get(row_start)`. If col 0 was NaN, `v > best_val` is
always false (IEEE-754), so the row's argmax stayed at index 0
regardless of any later numeric maxima. Sibling of `tf1d_argmax` /
`argmax_rows_f32` (restart 58 A7).

**Fix**: `seen = 0` sentinel pattern — adopt the first non-NaN slot.

**Canary**: `test_stage35_restart61_accuracy_count_from_logits_f32_nan_at_col_0`.

### A3 MEDIUM — `__abs_i32` INT32_MIN saturate (Family 3 — INT32_MIN abs)

`helixc/stdlib/transcendentals.hx __abs_i32` had documented UB for
INT32_MIN (-INT32_MIN is not representable as i32 → wraps back to
INT32_MIN). Sibling of `vec_negate_inplace` / `vec_map_neg` (restart
51 A5), `ti1d_max_abs` / `vec_max_abs` (restart 56 A2/A3),
`vec_map_abs` (restart 58 A2). `__abs_i32` is the canonical helper —
should be total.

**Fix**: saturate INT32_MIN to INT32_MAX.

**Canary**: `test_stage35_restart61_abs_i32_saturates_int32_min`.

## CLEAN spot-checks (no findings)

- `__softplus` / `__sigmoid` / `__tanh` / `__gelu` / `__silu` / `__relu` —
  all transcendental wrappers verified safe.
- `__powi` — bounded loop already, returns 1.0 for n>16.
- Autodiff reverse mode (`rev_add` / `rev_mul` / `rev_neg` / `rev_sub`)
  — restart 54 A1 closed.
- All NaN-skip in tf1d / tf2d reductions verified through restart 58.

## Carry-forward to restart 62

None — restart 62's Lane A found 2 NaN-fail-closed optimizer findings
(`sgd_f32_step` + `momentum_step`) that this audit missed.
