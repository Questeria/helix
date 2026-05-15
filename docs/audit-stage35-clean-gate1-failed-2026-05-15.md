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
  - `dense_classifier_sgd_step_f32` no longer needs a full weight-gradient
    scratch matrix. It reuses the caller scratch for logits, probabilities, and
    output deltas, then applies each weight update directly.
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
- Dense classifier step reuses adequate scratch without growing the arena.
- Adam zero-gradient, zero-epsilon step keeps the weight unchanged.
- Builtin BCE uses stable log near zero.
- Builtin scalar Adam step avoids the same zero-denominator edge.
- Batch CE invalid-label sentinel is not averaged down across rows.
- Softmax-CE gradient rejects mixed valid/invalid batches before mutating any
  gradient rows.
- Public `grad` rejects aggregate parameters at the rewrite surface.
- `grad_rev` rejects aggregate parameters at the public rewrite surface.
- `grad_rev_all` rejects aggregate parameters at the public rewrite surface.
- Tile IR rejects unmapped scalar division and branch/control-flow ops instead
  of lowering them as opaque calls.
- PTX unsupported-op tests now expect the earlier tile-layer failure point.
- Public `--emit-ptx` reports Tile IR unsupported-op diagnostics as normal PTX
  errors, not as internal compiler bugs.

## Verification

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

## Next Gate

The clean-gate counter remains `0/3`. The next audit must be a fresh Stage 35
clean gate on the fixed commit.
