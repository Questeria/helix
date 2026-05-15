# Stage 35 Clean Gate 1 - Failed And Fixed

Date: 2026-05-15

Result: failed clean gate, fixed, reset to 0/3 clean gates.

Three read-only Stage 35 audit lanes inspected current HEAD `0153941` before
this fix sweep. The gate did not count as clean because the auditors found real
issues in all three lanes.

## Findings Fixed

1. `dense_classifier_sgd_step_f32` trusted caller scratch space and could
   overwrite adjacent arena cells when scratch was undersized.
2. `__bce` in `transcendentals.hx` still used the short `__log` path instead of
   the stable log path for probabilities near 0 or 1.
3. `adam_f32_step` could divide by zero for zero gradient and `eps = 0`.
4. Public `grad_pass` still generated bogus scalar gradient wrappers for
   aggregate/struct parameters instead of failing closed until the pytree bridge
   is wired.
5. Tile IR lowered unsupported TIR ops into generic opaque `call` placeholders,
   which made the tile layer report success for behavior it did not represent.
6. The autotune docstring described future runtime measurement as if it already
   existed.

## Fix Summary

- `helixc/stdlib/nn.hx`
  - `adam_f32_step` clamps a zero or negative denominator to a small positive
    epsilon.
  - `dense_classifier_sgd_step_f32` now allocates its own internal temporary
    workspace sized from the model shape instead of writing past caller scratch.
- `helixc/stdlib/transcendentals.hx`
  - `__bce` now uses `__log_stable`.
- `helixc/frontend/grad_pass.py`
  - `grad`, `grad_rev`, and `grad_rev_all` now reject aggregate/non-scalar
    parameters with `NotImplementedError` until pytree leaf expansion is wired.
- `helixc/ir/tile_ir.py`
  - Unsupported TIR ops now fail closed with `NotImplementedError`.
- `helixc/frontend/autotune.py`
  - The docstring now states that runtime timing/dispatch is long-term design,
    while Phase-0 records and validates the static sweep spec.

## Regression Tests Added

- Dense classifier step does not clobber a guard after undersized scratch.
- Adam zero-gradient, zero-epsilon step keeps the weight unchanged.
- Builtin BCE uses stable log near zero.
- `grad_rev` rejects aggregate parameters at the public rewrite surface.
- `grad_rev_all` rejects aggregate parameters at the public rewrite surface.
- Tile IR rejects unmapped scalar division and branch/control-flow ops instead
  of lowering them as opaque calls.
- PTX unsupported-op tests now expect the earlier tile-layer failure point.

## Verification

- `python -m pytest -q helixc\tests\test_codegen.py -k "dense_classifier_sgd_step_f32_does_not_clobber_small_scratch or adam_f32_step_zero_grad_zero_eps_keeps_weight or builtin_bce_uses_stable_log_near_zero or grad_rev_rejects_aggregate_param or grad_rev_all_rejects_aggregate_param" --tb=short`
  - Result: 5 passed.
- `python -m pytest -q helixc\tests\test_tile_ir.py -k "tile_ir_rejects_unmapped_scalar_div or if_lowered_to_cfg_in_tile_ir or arith_passes_through or call_lowered" --tb=short`
  - Result: 4 passed.
- `python -m pytest -q helixc\tests\test_autodiff_reverse.py helixc\tests\test_pytree.py helixc\tests\test_ptx.py helixc\tests\test_tile_ir.py helixc\tests\test_autotune.py helixc\tests\test_ffi.py --tb=short`
  - Result: 139 passed.
- `python -m pytest -q helixc\tests\test_codegen.py -k "nn_ or stage35 or softmax or ce_loss or dense_classifier_sgd_step_f32 or adam_f32_step or grad_rev_rejects_aggregate_param or grad_rev_all_rejects_aggregate_param or builtin_bce_uses_stable_log_near_zero" --tb=short`
  - Result: 50 passed.

## Next Gate

The clean-gate counter remains `0/3`. The next audit must be a fresh Stage 35
clean gate on the fixed commit.
