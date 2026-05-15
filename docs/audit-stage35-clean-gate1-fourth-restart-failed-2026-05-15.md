# Stage 35 Clean Gate 1 - Fourth Restart Failed

Date: 2026-05-15
Baseline: `da35363`
Result: failed. Clean-gate count remains `0/3`.

## Findings

Three fresh audit lanes found real issues:

1. Forward-mode `grad` still failed open on opaque calls. Reverse-mode had been
   hardened, but forward-mode still returned a zero derivative after an AD
   warning for unrecognized calls.
2. `grad_pass` accepted integer scalar parameters and generated f32-only
   signatures even for f64 gradient surfaces.
3. The Helix reverse-AD runtime tape could overflow its capacity and overwrite
   following arena cells.
4. `dense_classifier_sgd_step_f32` returned success for invalid model shapes,
   while invalid labels correctly returned the loud sentinel.
5. The standalone PTX CLI still did not match the main `--emit-ptx` command
   path. It skipped trace/panic/unwind/unsafe validation and skipped the
   fold/CSE path used by the main driver.
6. Public docs had Stage 35 contract drift: conflicting stage labels, overbroad
   autodiff/transform claims, stale reverse-mode status, incomplete FFI status,
   and a stale "next work" entry.

## Fix Sweep

- Forward AD now raises `NotImplementedError` on opaque calls unless a known
  chain rule exists.
- `grad_pass` now limits scalar gradient parameters to `f32`/`f64` and preserves
  f64 generated signatures instead of narrowing them to f32.
- Reverse-AD tape creation clamps negative capacity to zero, and `rev_push`
  returns `-1` without writing when the tape is full.
- `dense_classifier_sgd_step_f32` returns sentinel `35001` for invalid shapes.
- Standalone PTX CLI now runs trace, panic, unwind, unsafe, and autotune
  validation, runs `grad_pass`, and applies fold/CSE before kernel-only Tile IR
  lowering.
- Stage 35 public docs now align on AI/ML Capability Push, current scalar AD
  behavior, future transform targets, current FFI status, and active next work.

## Verification

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

## Gate Status

This was a failed clean-gate attempt because findings were real and required
code and docs changes. Restart Stage 35 clean gate 1 from the new commit.
Clean-gate count remains `0/3`.
