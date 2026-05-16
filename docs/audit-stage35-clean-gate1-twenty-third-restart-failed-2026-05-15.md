# Stage 35 Clean Gate 1 - Twenty-Third Restart Failed

Date: 2026-05-15
Stage: 35 AI/ML Capability Push
Gate target: clean gate 1 of 3
Result: FAILED - findings found, clean gates remain `0/3`
Base commit: `01f3d46 Fix Stage 35 twenty-second restart findings`

## Restart 23 Gate Evidence

Smoke/support checks before the audit:

- Parsed 16 stdlib files.
- Python syntax check passed for the recently changed compiler modules.
- AD math smoke: 8 passed.
- Stage 35 codegen smoke: 11 passed.
- `--emit-ptx` CLI smoke: 12 passed.
- Direct PTX CLI smoke: 23 passed.
- Broader support checks:
  - AD/transcendentals/reverse AD: 101 passed.
  - Selected Stage 35 codegen slice: 125 passed.
  - CLI suite: 159 passed.
  - PTX suite: 75 passed.
  - Autotune suite: 26 passed.
  - Tile IR suite: 8 passed.
  - Selected tensor/tile/diff typecheck slice: 12 passed.
  - Collection: 2,277 tests collected.

The first audit batch timed out after five minutes and was restarted with a
tighter scope. The replacement audit found real issues, so the gate does not
count as clean.

## Findings

Lane A - AD and runtime:

- High: `_is_inferably_pure()` only recognized the older f32 builtin set, so
  unannotated helper functions calling newly AD-known helpers such as
  `__log_stable` or `__sqrt_f64` were left opaque and could fail AD inlining.
- High: `rev_backward` accepted forged leaf records with non-leaf operands,
  allowing a corrupted operation to be downgraded to a leaf and silently drop
  gradients.

Lane B - CLI and host lowering:

- High: `helixc.check` pruned unreachable `D<T>`-signature helpers for `-o`
  output but not for other host-lowering paths such as `--emit-asm` or bare
  `--strict`, so dead AD helpers could still reach IR lowering and surface as
  internal compiler bugs.

Lane C - docs and status:

- P2: The website reference still made the target bootstrap chain sound live
  by saying the toolchain had zero external dependencies and each link compiles
  the next.
- P3: The restart-21 pause handoff still described restart 22 as the latest
  active work even after restart 22 had closed.

## Fix Sweep

- Added one shared `AD_KNOWN_PURE_CALLS` set in forward AD and used it for
  both inferred purity and builtin inlining skips.
- Added tests for inferred-pure helper functions using `__log_stable` and
  `__sqrt_f64`.
- Hardened reverse-AD tape prevalidation so leaf records must have both operand
  slots set to `-1`.
- Added a forged-leaf regression test proving `rev_backward` rejects the tape
  before mutating adjoints.
- Pruned unreachable differentiable-signature helpers for all host-lowering
  paths in `helixc.check`, not just `-o`.
- Added regression tests for `--emit-asm` and bare `--strict` with dead AD
  helpers.
- Updated bootstrap docs to say later links are roadmap targets until verified,
  and refreshed the pause handoff to point at restart 23.

## Verification After Fix Sweep

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 stdlib files.
- `python -m py_compile helixc\frontend\autodiff.py helixc\frontend\autodiff_reverse.py helixc\check.py helixc\backend\ptx.py helixc\backend\x86_64.py`
  - Result: passed.
- `python -m pytest helixc/tests/test_transcendentals.py helixc/tests/test_autodiff.py helixc/tests/test_autodiff_reverse.py -q`
  - Result: 103 passed.
- `python -m pytest helixc/tests/test_codegen.py -k "t1d or t2d or ti2d or tf2d or tensor or stage35_2d or nn_ or dense_layer or softmax_rows_f32 or softmax_ce_grad_f32 or argmax_rows_f32 or accuracy_count_from_logits_f32 or ce_loss_batch_f32 or bce or gelu or revad" -q`
  - Result: 126 passed.
- `python -m pytest helixc/tests/test_cli.py -q`
  - Result: 161 passed.
- `python -m pytest helixc/tests/test_ptx.py -q`
  - Result: 75 passed.
- `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider`
  - Result: 2,282 tests collected.

## Gate Decision

Restart 23 does not count as a clean gate because all three lanes found issues.
After this fix sweep is committed, the next action is to start restart 24 from
the new commit and attempt clean gate 1 of 3 again.
