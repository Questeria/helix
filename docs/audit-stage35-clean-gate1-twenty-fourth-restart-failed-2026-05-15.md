# Stage 35 Clean Gate 1 - Twenty-Fourth Restart Failed

Date: 2026-05-15
Stage: 35 AI/ML Capability Push
Gate target: clean gate 1 of 3
Result: FAILED - findings found, clean gates remain `0/3`
Base commit: `a3874b1 Fix Stage 35 twenty-third restart findings`

## Restart 24 Gate Evidence

Smoke/support checks before the audit:

- Parsed 16 stdlib files.
- Python syntax check passed for the recently changed compiler modules.
- AD helper smoke: 10 passed.
- Stage 35 codegen smoke: 12 passed.
- CLI smoke: 16 passed.
- Direct PTX smoke: 23 passed.
- Broader support checks:
  - AD/transcendentals/reverse AD: 103 passed.
  - Selected Stage 35 codegen slice: 126 passed.
  - Autotune and tile IR: 34 passed.
  - CLI suite: 161 passed.
  - PTX suite: 75 passed.
  - Collection: 2,282 tests collected.

## Findings

Lane A - AD and runtime:

- High: Reverse-AD adjoint APIs validated against tape capacity instead of the
  logical tape count. A tape with capacity greater than count could accept
  `rev_seed(adj, idx, seed)` and expose `rev_grad(adj, idx)` for nonexistent
  tape entries.

Lane B - CLI and backend:

- Clean. No blocker-grade issue was found in the requested CLI/backend surfaces.

Lane C - docs and status:

- P2: `docs/stage35-progress-2026-05-15.md` had Increment 42 inserted before
  Increment 28, so tail-reading the ledger made restart 22 look newest.
- P3: `docs/STAGE35_PAUSE_HANDOFF_2026-05-15.md` still presented restart 23 as
  the latest active work after restart 23 had closed.
- P2: Website comparison tables still described bootstrap/self-hosting in ways
  that could read as shipped rather than target behavior.

## Fix Sweep

- Reverse-AD adjoint metadata now records logical tape count at allocation.
- `rev_seed` and `rev_grad` now reject indices between logical count and
  capacity.
- `rev_backward` now rejects tapes that grew after adjoints were allocated.
- Added regressions for seed/grad count-vs-capacity, grown-after-allocation
  tapes, and the updated adjoint cap metadata layout.
- Reordered the Stage 35 progress ledger so Increments 42 and 43 are at the
  tail after Increment 41.
- Reworded bootstrap comparison docs to label full-from-hex/self-host behavior
  as targets.
- Reworded the pause handoff as historical continuity only.

## Verification After Fix Sweep

- Per-file stdlib parser sweep across `helixc/stdlib/*.hx`
  - Result: parsed 16 stdlib files.
- `python -m py_compile helixc\frontend\autodiff.py helixc\frontend\autodiff_reverse.py helixc\check.py helixc\backend\ptx.py helixc\backend\x86_64.py`
  - Result: passed.
- `python -m pytest helixc/tests/test_codegen.py -k "revad_seed_rejects_index_between_count_and_capacity or revad_grad_hides_index_between_count_and_capacity or revad_backward_rejects_tape_grown_after_adjoints_allocated or revad_seed_rejects_corrupt_adj_cap_metadata or revad_grad_rejects_corrupt_adj_cap_metadata or forged_leaf or foreign_adjoint_buffer or self_referential_operand or prevalidates_before_adj_mutation" -q`
  - Result: 9 passed.
- `python -m pytest helixc/tests/test_transcendentals.py helixc/tests/test_autodiff.py helixc/tests/test_autodiff_reverse.py -q`
  - Result: 103 passed.
- `python -m pytest helixc/tests/test_codegen.py -k "t1d or t2d or ti2d or tf2d or tensor or stage35_2d or nn_ or dense_layer or softmax_rows_f32 or softmax_ce_grad_f32 or argmax_rows_f32 or accuracy_count_from_logits_f32 or ce_loss_batch_f32 or bce or gelu or revad" -q`
  - Result: 129 passed.
- `python -m pytest helixc/tests/test_cli.py -q`
  - Result: 161 passed.
- `python -m pytest helixc/tests/test_ptx.py -q`
  - Result: 75 passed.
- `python -m pytest helixc/tests --collect-only -q -p no:cacheprovider`
  - Result: 2,285 tests collected.

## Gate Decision

Restart 24 does not count as a clean gate because Lane A and Lane C found
issues. After this fix sweep is committed, the next action is to start restart
25 from the new commit and attempt clean gate 1 of 3 again.
