# Stage 35 Clean Gate 1 - Twenty-Seventh Restart Failed

Date: 2026-05-15
Base commit audited: `44c6b6a`
Result: not clean
Clean-gate counter: remains `0/3`

## Summary

Restart 27 did not count as a clean Stage 35 gate. The docs lane returned clean,
but the AD/tensor and backend/PTX lanes found multiple remaining blockers. This
restart was useful because the widened audit prompts found several issues in one
pass instead of stopping after the first confirmed failure.

## Findings Fixed

1. `helixc/frontend/autodiff.py`
   - Block final expressions were not checked with the same AD erasure guard as
     let/const/expression statements, so a final assignment could compile into a
     zero gradient.
   - Fix: reject side-effecting block final expressions before differentiation.
   - Regressions:
     `test_grad_rejects_side_effecting_final_assignment`,
     `test_grad_rev_rejects_side_effecting_final_assignment`.

2. `helixc/stdlib/autodiff_reverse.hx`
   - `rev_backward` could accept a foreign adjoint buffer if a mutable tape
     header was spoofed to point at it.
   - Fix: adjoint metadata now records the owner tape, and backward validation
     requires the owner to match.
   - Regression: `test_revad_backward_rejects_spoofed_foreign_adjoint_buffer`.

3. `helixc/stdlib/tensor.hx` and `helixc/stdlib/nn.hx`
   - Negative or empty 1D allocations could alias later allocations, and dense
     f32 gradient helpers could write through empty output buffers.
   - Fix: reserve an empty-allocation sentinel, add 1D capacity guards for
     setters, and check f32 dense-gradient input/output buffer capacities.
   - Regressions:
     `test_negative_t1d_new_does_not_alias_next_allocation`,
     `test_dense_layer_f32_grad_x_rejects_empty_output_buffer`.

4. `helixc/check.py` and `helixc/backend/ptx.py`
   - PTX validation lowered full host programs before AD rewriting, causing
     valid host `grad(...)` code to block `--emit-ptx`.
   - Fix: run `grad_pass` on the full validation program before lowering.
   - Regression: `test_stage35_emit_ptx_allows_valid_host_grad_call`.

5. `helixc/check.py` and `helixc/backend/x86_64.py`
   - `-Wad=error` could return an error after already emitting x86 artifacts.
   - Fix: drain/promote AD warnings before `-o`, `--emit-asm`, and direct x86
     artifact emission.
   - Regressions:
     `test_stage35_wad_error_output_binary_does_not_write_artifact`,
     `test_stage35_wad_error_emit_asm_does_not_print_artifact`,
     `test_stage35_direct_x86_honors_wad_error_before_writing`.

6. `helixc/backend/x86_64.py`
   - Direct backend API callers could embed PTX after DCE removed unsupported
     dead kernel operations.
   - Fix: `compile_module_to_elf` now requires a pre-DCE kernel tile validation
     marker before embedding PTX.
   - Regression:
     `test_stage35_compile_module_to_elf_requires_pre_dce_kernel_validation`.

## Verification

- `python -m py_compile helixc\frontend\autodiff.py helixc\backend\ptx.py helixc\backend\x86_64.py helixc\check.py`
  - Result: passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed stdlib files.
- Focused codegen regressions
  - Result: 6 passed, 852 deselected.
- Focused CLI regressions
  - Result: 4 passed, 165 deselected.
- `python -m pytest helixc/tests/test_transcendentals.py helixc/tests/test_autodiff.py helixc/tests/test_autodiff_reverse.py -q`
  - Result: 103 passed.
- Stage 35-adjacent codegen slice
  - Result: 139 passed, 719 deselected.
- `python -m pytest helixc/tests/test_cli.py -q`
  - Result: 169 passed.
- `python -m pytest helixc/tests/test_ptx.py -q`
  - Result: 76 passed.
- `python -m pytest helixc/tests/test_effect_check.py -q`
  - Result: 34 passed.
- `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider`
  - Result: 2,304 tests collected.

## Next Step

Start restart 28 from the committed restart-27 fix sweep and run another fresh
Stage 35 clean-gate audit. Do not advance the clean-gate counter unless all
audit lanes return clean.
