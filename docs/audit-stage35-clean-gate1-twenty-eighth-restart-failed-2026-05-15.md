# Stage 35 Clean Gate 1 - Twenty-Eighth Restart Failed

Date: 2026-05-15
Base commit audited: `3830869`
Result: not clean
Clean-gate counter: remains `0/3`

## Summary

Restart 28 did not count as a clean Stage 35 gate. All three audit lanes found
remaining issues, but the widened prompts again returned multiple independent
findings in one pass instead of stopping at a single blocker.

## Findings Fixed

1. `helix_website/HELIX_REFERENCE.md`, `QUICKSTART.md`, and the pause handoff
   - Public and handoff wording still overclaimed the current bootstrap state
     and used stale restart/test-count language.
   - Fix: reframe self-hosting and reproducible bootstrap language as roadmap
     targets, keep the Python-hosted compiler status explicit, and update the
     status docs to restart 28 / 2,316 collected tests.

2. `helixc/check.py`
   - `-Wad=error` could still leak stdout artifacts for `--emit-ir`, default
     clean output, and `--check-only` clean output before the AD warning was
     promoted to an error.
   - Fix: drain and promote AD warnings before those stdout paths.
   - Regressions:
     `test_stage35_wad_error_emit_ir_does_not_print_artifact`,
     `test_stage35_wad_error_default_does_not_print_clean`,
     `test_stage35_wad_error_check_only_does_not_print_clean`.

3. `helixc/backend/x86_64.py`
   - Direct x86 output did not treat `-Wdeprecated=error` with CLI parity, and
     typecheck failures could bypass AD warning draining.
   - Fix: direct backend warning parsing now accepts `ad` and `deprecated`,
     drains AD warnings on type errors, and promotes deprecated warnings before
     writing direct x86 artifacts.
   - Regressions:
     `test_stage35_direct_x86_honors_deprecated_error_before_writing`,
     `test_stage35_direct_x86_drains_ad_warnings_on_type_error`.

4. `helixc/backend/ptx.py`, `helixc/ir/passes/dce.py`,
   `helixc/ir/passes/fdce.py`, and `helixc/tests/test_codegen.py`
   - Kernel tile validation could still be satisfied after DCE/FDCE erased an
     unsupported dead kernel operation.
   - Fix: DCE/FDCE mark kernel modules as blocked when validation has not run
     first, and PTX validation fails closed on that marker.
   - Regression:
     `test_stage35_compile_module_to_elf_requires_pre_dce_kernel_validation`.

5. `helixc/stdlib/tensor.hx` and `helixc/stdlib/nn.hx`
   - 1D tensor capacity checks still relied on current arena length, so
     negative indices, positive out-of-bounds writes after later allocations,
     short f32 gradient buffers, and direct f32 vector helpers could write
     outside their logical allocation.
   - Fix: 1D tensors now carry checked header/footer metadata while preserving
     the data-start handle API. Capacity checks accept valid 1D handles and
     valid 2D data handles, and common writer helpers fail closed on bad
     lengths.
   - Regressions:
     `test_t1d_setters_reject_negative_indices`,
     `test_t1d_setters_reject_positive_oob_even_after_later_allocations`,
     `test_dense_layer_f32_grad_x_rejects_short_output_buffer`,
     `test_mse_loss_f32_grad_rejects_short_output_buffer`,
     `test_tf1d_add_rejects_short_output_buffer`.

6. `helixc/stdlib/autodiff_reverse.hx`
   - Reverse-AD adjoint ownership metadata could still be forged consistently
     because all metadata fields were public arena slots.
   - Fix: adjoints must now be allocated immediately after their owner tape,
     and validation checks that layout invariant in addition to owner/cap/count
     metadata.
   - Regressions:
     `test_revad_backward_rejects_consistently_forged_adjoint_slice`,
     `test_revad_alloc_adjoints_rejects_interleaved_arena_allocation`.

## Verification

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 files.
- `python -m py_compile helixc\frontend\autodiff.py helixc\frontend\autodiff_reverse.py helixc\check.py helixc\backend\ptx.py helixc\backend\x86_64.py helixc\ir\passes\dce.py helixc\ir\passes\fdce.py`
  - Result: passed.
- Focused tensor/NN regression slice
  - Result: 56 passed, 809 deselected.
- Focused reverse-AD runtime slice
  - Result: 30 passed, 835 deselected.
- Focused CLI restart-28 slice
  - Result: 31 passed, 143 deselected.
- Focused PTX embedding/regression slice
  - Result: 5 passed, 860 deselected.
- Stage 35-adjacent codegen slice
  - Result: 145 passed, 720 deselected.
- `python -m pytest helixc/tests/test_transcendentals.py helixc/tests/test_autodiff.py helixc/tests/test_autodiff_reverse.py helixc/tests/test_cli.py helixc/tests/test_ptx.py helixc/tests/test_effect_check.py -q`
  - Result: 387 passed.
- `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider`
  - Result: 2,316 tests collected.

## Next Step

Start restart 29 from the committed restart-28 fix sweep and run another fresh
Stage 35 clean-gate audit. Do not advance the clean-gate counter unless all
audit lanes return clean.
