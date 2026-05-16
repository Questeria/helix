# Stage 35 Clean Gate 1 - Twenty-Fifth Restart Failed

Restart 25 began from commit `8f56b5b` after restart 24 was committed and
pushed. The support gate was green, but the fresh audit found remaining Stage
35 blockers. This restart therefore does not count toward the `0/3` clean-gate
requirement.

## Findings

Lane A - AD/runtime/stdlib:

- Reverse-AD tapes had no magic/footer validation, so forged arena buffers that
  looked like `count/cap` headers could be accepted and could be written through
  by `rev_alloc_adjoints`.
- Forward AD could erase allocation/effecting `let` bindings while flattening a
  differentiated block. Tensor allocators were also incorrectly marked `@pure`.

Lane B - CLI/backend/PTX:

- Embedded PTX binary paths could run host DCE before PTX tile validation,
  hiding unsupported operations inside kernels that `--emit-ptx` rejected.
- Non-strict PTX modes emitted kernels without reporting host effect warnings
  that host emit modes reported.

Lane C - docs/status:

- Restart 24 status was missing from the historical pause handoff.
- The progress ledger still said the next step was to commit restart 24.
- `HELIX_REFERENCE.md` contradicted itself by labeling current `helixc` as both
  Python-hosted and "in Helix itself".
- A repo-local Stage 35 API-contract audit surface was missing.

## Fixes

- Added reverse-AD tape magic/footer validation and required all tape-mutating
  APIs to reject forged or truncated tape buffers before writing.
- Removed misleading `@pure` annotations from tensor allocation helpers.
- Made AD helper inlining inspect function bodies instead of trusting explicit
  `@pure` blindly.
- Made AD let-flattening fail closed when it would erase allocation/effecting
  expressions, while still allowing pure containers such as `match`.
- Added shared PTX kernel-only and kernel-validation helpers.
- Validated kernel tile lowering before host DCE/FDCE in `helixc.check` and
  direct x86 binary emission.
- Reported full-program effect warnings in non-strict PTX modes while still
  emitting kernel PTX.
- Updated public/status docs and added `docs/stage35-api-contracts-2026-05-15.md`.

## Verification

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 stdlib files.
- `python -m py_compile helixc\frontend\autodiff.py helixc\check.py helixc\backend\ptx.py helixc\backend\x86_64.py`
  - Result: passed.
- Focused reverse-AD / AD / CLI / PTX regressions
  - Result: passed.
- `python -m pytest helixc/tests/test_transcendentals.py helixc/tests/test_autodiff.py helixc/tests/test_autodiff_reverse.py -q`
  - Result: 103 passed.
- `python -m pytest helixc/tests/test_codegen.py -k "t1d or t2d or ti2d or tf2d or tensor or stage35_2d or nn_ or dense_layer or softmax_rows_f32 or softmax_ce_grad_f32 or argmax_rows_f32 or accuracy_count_from_logits_f32 or ce_loss_batch_f32 or bce or gelu or revad or grad_rejects_allocator_let" -q`
  - Result: 131 passed.
- `python -m pytest helixc/tests/test_cli.py -q`
  - Result: 164 passed.
- `python -m pytest helixc/tests/test_ptx.py -q`
  - Result: 76 passed.
- `python -m pytest helixc/tests/test_effect_check.py -q`
  - Result: 34 passed.
- `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider`
  - Result: 2,291 tests collected.

## Gate Status

- Stage 35 clean gates remain `0/3`.
- Next step: start restart 26 as a fresh Stage 35 clean-gate audit from the
  newest committed fix sweep.
