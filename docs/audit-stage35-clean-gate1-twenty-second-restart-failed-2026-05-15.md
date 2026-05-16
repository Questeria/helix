# Stage 35 Clean Gate 1 - Twenty-Second Restart Failed

Date: 2026-05-15
Stage: 35 AI/ML Capability Push
Gate target: clean gate 1 of 3
Result: FAILED - findings found, clean gates remain `0/3`
Base commit: `c6dfb53 Fix Stage 35 twenty-first restart findings`

## Restart 22 Gate Evidence

Smoke/support checks before the audit:

- Parsed 16 stdlib files.
- `python -m pytest helixc/tests/test_transcendentals.py -k "gelu or bce" -q`
  - Result: 4 passed.
- `python -m pytest helixc/tests/test_codegen.py -k "stage35_2d or public_2d_helpers or negative_2d_shape or revad_backward_prevalidates_before_adj_mutation" -q`
  - Result: 8 passed.
- `python -m pytest helixc/tests/test_cli.py -k "stage35_emit_ptx" -q`
  - Result: 11 passed.
- `python -m pytest helixc/tests/test_ptx.py -k "stage35_direct_ptx_cli" -q`
  - Result: 20 passed.
- Broader support checks before the audit:
  - AD/transcendentals/reverse AD: 97 passed.
  - Selected Stage 35 codegen slice: 122 passed.
  - CLI suite: 156 passed.
  - PTX suite: 72 passed.
  - Collection: 2,264 tests collected.

## Findings

Lane A - AD, NN, and runtime:

- P2: Public numeric helpers such as `__log_stable` and f64 math helpers were
  available to user code but had no analytic forward/reverse AD rules, causing
  silent AD gaps or opaque-call behavior.
- P2: `rev_backward` accepted invalid tape/adjoint relationships and could
  read malformed or self-referential tape entries.
- P2: Matrix and NN helpers could return success after shape metadata failures,
  leaving stale caller output buffers.
- P2: `t2d_shape_ok` metadata was forgeable and did not prove the allocation
  extent still covered the claimed shape.

Lane B - PTX, tile, autotune, and CLI:

- P1: `--strict --emit-ptx` could skip full-program effect enforcement when an
  unreachable differentiable helper made the host lower fail.
- P2: Direct PTX CLI warning policy did not match `helixc.check`; `-Wad=warn`
  and deprecated-warning policy flags were rejected or ignored.
- P2: Embedded PTX binary paths were not isolated from dead host AD functions
  and could traceback instead of compiling clean reachable code.
- P3: Embedded PTX tests bypassed some production validation surfaces.

Lane C - docs and status:

- P2: Public quickstart/reference copy overclaimed self-hosting/bootstrap
  status.
- P2: The pause handoff was stale and still looked like a live restart-21 task
  list.
- P2: Website reference copy overstated bf16, Blackwell, and non-REG memory
  space support as shipped behavior.
- P3: Website clean-gate wording still described the historical Stage 30
  five-clean-gate policy as if it were current.
- P3: Website bootstrap API stubs could leak target byte counts as verified
  facts.

## Fix Sweep

- Added stronger 2D tensor metadata with a footer and extent checks, plus a
  shared `t2d_error()` sentinel for status-returning helpers.
- Updated public 2D/NN helpers to return `35001` on metadata mismatch or
  overflow instead of reporting success.
- Hardened `rev_backward` so the tape must match its adjoint buffer and binary
  operands must refer only to earlier tape entries.
- Added forward and reverse AD chain rules for `__log_stable` and f64 exp/log,
  sin/cos, sqrt, relu, sigmoid, and abs helpers.
- Pruned unreachable differentiable-signature functions before strict host
  lowering in the PTX and embedded-binary validation paths, so dead AD helpers
  no longer mask reachable host effect errors.
- Aligned direct PTX CLI warning handling with `helixc.check`, including
  `-Wad=warn`, `-Wad=error`, `-Wdeprecated`, and deprecated-warning emission.
- Added production-path tests for `helixc.check -o`, direct x86 output, direct
  PTX warning policy, reverse-AD tape rejection, forged 2D metadata, and the
  newly AD-known math helpers.
- Updated public docs/API contracts to distinguish live artifacts from roadmap
  targets and to report the restart-22 collection count.

## Verification After Fix Sweep

- `python -m py_compile helixc\frontend\autodiff.py helixc\frontend\autodiff_reverse.py helixc\check.py helixc\backend\ptx.py helixc\backend\x86_64.py`
  - Result: passed.
- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 stdlib files.
- `python -m pytest helixc/tests/test_transcendentals.py helixc/tests/test_autodiff.py helixc/tests/test_autodiff_reverse.py -q`
  - Result: 101 passed.
- `python -m pytest helixc/tests/test_codegen.py -k "t1d or t2d or ti2d or tf2d or tensor or stage35_2d or nn_ or dense_layer or softmax_rows_f32 or softmax_ce_grad_f32 or argmax_rows_f32 or accuracy_count_from_logits_f32 or ce_loss_batch_f32 or bce or gelu or revad" -q`
  - Result: 125 passed.
- `python -m pytest helixc/tests/test_cli.py -q`
  - Result: 159 passed.
- `python -m pytest helixc/tests/test_ptx.py -q`
  - Result: 75 passed.
- `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider`
  - Result: 2,277 tests collected.

## Gate Decision

Restart 22 does not count as a clean gate because all three lanes found issues.
After this fix sweep is committed, the next action is to start restart 23 from
the new commit and attempt clean gate 1 of 3 again.
