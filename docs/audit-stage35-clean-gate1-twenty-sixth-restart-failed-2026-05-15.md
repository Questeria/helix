# Stage 35 Clean Gate 1 - Twenty-Sixth Restart Failed

Date: 2026-05-15
Base commit audited: `45bf6ff`
Result: not clean
Clean-gate counter: remains `0/3`

## Summary

Restart 26 did not count as a clean Stage 35 gate. The support checks were green,
but the audit lanes found remaining correctness and documentation issues. The
fixes were implemented with direct regression coverage before the next restart.

## Findings Fixed

1. `helixc/stdlib/tensor.hx`
   - `tf2d_zeros` was still annotated `@pure` even though it allocates through
     `t2d_new`.
   - Fix: removed the misleading purity annotation.
   - Regression: `test_stage35_tensor_allocators_are_not_marked_pure`.

2. `helixc/stdlib/autodiff_reverse.hx`
   - `rev_backward` rejected tapes that grew after adjoint allocation but still
     accepted tapes whose logical count had been shrunk.
   - Fix: require the current tape count to exactly match the adjoint snapshot
     count.
   - Regression:
     `test_revad_backward_rejects_tape_shrunk_after_adjoints_allocated`.

3. `helixc/check.py`
   - `--emit-proof-obligations --strict` ran strict effect diagnostics on the
     raw program, so a dead helper with a `D<T>` signature could trigger an
     unresolved generic pipeline error.
   - Fix: prune unreachable differentiable-signature helpers before the strict
     AD/lowering diagnostic path.
   - Regression:
     `test_stage35_emit_proof_obligations_strict_ignores_dead_ad_helper`.

4. `docs/STAGE35_PAUSE_HANDOFF_2026-05-15.md`
   - The pause handoff still described restart 25 as beginning from `8f56b5b`
     even after restart 25 closed at `45bf6ff`.
   - Fix: mark restart 25 closed and restart 26 as the current continuation.

5. `helix_website/HELIX_REFERENCE.md`
   - One comparison row overclaimed current bootstrap status as self-hosting.
   - Fix: describe the live state as a 299-byte root with self-hosting still a
     target.

## Verification

- `python -m py_compile helixc\check.py`
  - Result: passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed stdlib files.
- `python -m pytest helixc/tests/test_codegen.py -k "revad_backward_rejects_tape_shrunk_after_adjoints_allocated or stage35_tensor_allocators_are_not_marked_pure or revad_backward_rejects_tape_grown_after_adjoints_allocated" -q`
  - Result: 3 passed, 849 deselected.
- `python -m pytest helixc/tests/test_cli.py -k "stage35_emit_proof_obligations_strict_ignores_dead_ad_helper or stage31_emit_proof_obligations_classifies_strict_effect_error or stage31_emit_proof_obligations_strict_effect_pass_failure_stays_json" -q`
  - Result: 3 passed, 162 deselected.
- `python -m pytest helixc/tests/test_transcendentals.py helixc/tests/test_autodiff.py helixc/tests/test_autodiff_reverse.py -q`
  - Result: 103 passed.
- `python -m pytest helixc/tests/test_codegen.py -k "t1d or t2d or ti2d or tf2d or tensor or stage35_2d or nn_ or dense_layer or softmax_rows_f32 or softmax_ce_grad_f32 or argmax_rows_f32 or accuracy_count_from_logits_f32 or ce_loss_batch_f32 or bce or gelu or revad or grad_rejects_allocator_let or stage35_tensor_allocators_are_not_marked_pure" -q`
  - Result: 133 passed, 719 deselected.
- `python -m pytest helixc/tests/test_cli.py -q`
  - Result: 165 passed.
- `python -m pytest helixc/tests/test_ptx.py -q`
  - Result: 76 passed.
- `python -m pytest helixc/tests/test_effect_check.py -q`
  - Result: 34 passed.
- `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider`
  - Result: 2,294 tests collected.

## Next Step

Start restart 27 from the committed restart-26 fix sweep and run another fresh
Stage 35 clean-gate audit. Do not advance the clean-gate counter unless all
audit lanes return clean.
