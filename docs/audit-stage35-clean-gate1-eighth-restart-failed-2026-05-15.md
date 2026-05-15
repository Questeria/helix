# Stage 35 Clean Gate 1 - Eighth Restart Findings

Date: 2026-05-15

Result: FAIL. Clean-gate count remains 0/3.

This audit was run after commit `d02fd9b` (`Fix Stage 35 seventh restart
findings`). The fresh clean-gate attempt found remaining runtime, PTX strict,
and documentation issues, so the gate did not count as clean.

## Lane A - AD / NN / Runtime Correctness

- P1: `rev_backward` trusted a corrupted tape `count` above `cap`, allowing the
  backward pass to read/write past the allocated adjoint array.
- P2: 2D tensor helpers computed `rows * cols` before checking dimensions, so
  two negative dimensions became a positive element count.
- P2: `ti1d_min` and `ti1d_max` still used `n == 0`, causing negative lengths
  to read element zero.

## Lane B - PTX / Tile / Autotune CLI Parity

- P1: standalone direct PTX `--strict` skipped totality enforcement, unlike
  `helixc.check --emit-ptx --strict`, and could emit PTX for an unproven
  recursive function.

## Lane C - Documentation / Status Consistency

- P1: `docs/lang/agi-features.md` still contained broad uniqueness wording.
- P2: `docs/ROADMAP.md` still used moat/competitor phrasing.
- P3: `docs/stage35-progress-2026-05-15.md` still had confusing non-monotonic
  increment chronology in its older sections.

## Fix Plan

- Bound reverse-AD valid indices by both `count` and `cap`, and make
  `rev_backward` fail before looping if `count` exceeds tape or adjoint
  capacity.
- Add a safe 2D element-count helper and use it for shape-derived 2D tensor
  helper loops/allocations.
- Treat negative integer min/max lengths as empty.
- Mirror totality warnings/strict aborts in the direct PTX CLI.
- Reframe public docs and mechanically reorder the Stage 35 progress ledger.
