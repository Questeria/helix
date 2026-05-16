# Stage 35 Clean Gate 1 - Twenty-Ninth Restart Failed

Date: 2026-05-16
Base HEAD: `585ae84` (`Update Stage 35 handoff after restart 28`)
Result: NOT CLEAN. Stage 35 remains at `0/3` clean gates.

Restart 29 began from a green support baseline, then three audit lanes found
remaining blockers. The blockers were fixed in this restart, but because the
gate found issues, this restart does not count as a clean gate.

## Audit Findings

### Lane A - AD, Tensor, Runtime, NN

Status: findings present.

- Reverse-AD adjoint metadata could still be forged by allocating a fake slice
  at the immediate post-tape location and mutating `tape + 2` to point at it.
- `tf1d_dot_with_offset` rejected negative offsets but still allowed positive
  out-of-bounds slice lengths.
- `tf1d_argmax_in_range` and `tf1d_sum_in_range` trusted inflated logical
  lengths and could read beyond the real allocation.
- `tf1d_lerp` and modern NN activation writers could corrupt short output
  buffers.
- The first guard sweep was too strict for valid sub-slices such as `x + 1` and
  per-row softmax offsets, so it required a slice-aware capacity helper.

### Lane B - Backend, PTX, CLI

Status: findings present.

- The normal `helixc.check` CLI routed warning-as-error progress and diagnostic
  headers to stdout in some non-PTX modes.
- The direct x86 backend exited early on deprecated or validation errors without
  draining pending AD warnings.
- PTX kernel validation only ran tile lowering; it did not run PTX emission
  before marking embedded kernels validated.

### Lane C - Docs and Public Claims

Status: findings present.

- `HANDOFF_FOR_CHATGPT.md` was a Stage 30 historical handoff but still looked
  current in several places.
- `docs/STAGE35_PAUSE_HANDOFF_2026-05-15.md` pointed restart 29 at `e6c9ced`
  instead of the newer `585ae84` handoff commit.
- `helix_website/HELIX_REFERENCE.md` overclaimed the current PTX/GPU status.
- `README.md` used absolute bootstrap wording that conflicted with the current
  Python-hosted `helixc` production compiler status.

## Fix Sweep

- Added `t1d_slice_ok` so guarded 1D writers can accept valid interior slices
  while still rejecting short buffers.
- Hardened tensor range helpers and `tf1d_lerp` with real allocation-capacity
  checks.
- Switched NN vector guards to slice-aware checks and preserved valid sub-slice
  activation and row-softmax behavior.
- Added reverse-AD tape footer and adjoint guards derived from owner, cap,
  count, and actual adjoint start, closing the immediate-layout forgery.
- Routed warning-as-error CLI progress and diagnostics to stderr.
- Added direct x86 `_exit_after_ad_drain` and used it on early validation exits.
- Made PTX validation lower and emit PTX before setting the validated marker.
- Marked the old ChatGPT handoff as historical, updated the restart-29 pointer,
  and narrowed public PTX/bootstrap claims to shipped capability.

## Verification

- Per-file stdlib parser sweep for `tensor.hx`, `nn.hx`, and
  `autodiff_reverse.hx`: passed.
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
  - First attempt found two over-strict sub-slice guard regressions.
  - Final result after `t1d_slice_ok`: 142 passed, 728 deselected.
- `python -m pytest helixc\tests\test_codegen.py -q -k "revad or grad_rejects_allocator_let or side_effecting_final_assignment or negative_t1d_new or compile_module_to_elf_requires_pre_dce_kernel_validation or ptx_in_binary or kernel_ptx or wad"`
  - Result: 40 passed, 830 deselected.
- `python -m pytest helixc\tests --collect-only -q -p no:cacheprovider`
  - Result: 2,325 tests collected.
- `git diff --check`
  - Result: passed.

## Next Step

Restart 29 was committed and pushed as `b3f7796`. Restart 30 has since begun
and found additional issues, so use
`docs/audit-stage35-clean-gate1-thirtieth-restart-failed-2026-05-16.md` and
`docs/stage35-progress-2026-05-15.md` for current continuation state.
