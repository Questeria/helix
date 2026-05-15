# Stage 35 Clean Gate 1 - Fifth Restart Findings

Date: 2026-05-15

Result: FAIL. Clean-gate count remains 0/3.

This audit was run after commit `9e72510` (`Fix Stage 35 fourth restart
findings`). Three read-only lanes found real issues, so the gate did not count
as clean.

## Lane A - AD / NN / Runtime Correctness

- P1: `grad_rev_all` preserved f64 signatures but still wrote gradients
  through `modify_f`, the f32 reflection path. A f64 gradient such as `6.0`
  could be stored as `0.0` when read through the reflection cell.
- P1: reverse-AD tape operations accepted invalid operand indices. A corrupted
  or invalid operand could later make `rev_backward` write before the adjoint
  array.
- P2: scalar `ce_loss` clamped only the lower probability bound, so
  probabilities above 1.0 could produce negative cross-entropy.
- P3: `softmax_layer` accepted negative lengths and could read one arena cell
  before returning success.

## Lane B - PTX / Tile / Autotune CLI Parity

- P1: standalone `python -m helixc.backend.ptx` still skipped the canonical
  surface-shaping pipeline used by `helixc.check --emit-ptx`, including module
  flattening, impl flattening, struct monomorphization, and function
  monomorphization.
- P2: duplicate `@autotune` keys silently overwrote earlier values.

## Lane C - Documentation / Status Consistency

- P1: `docs/HELIX_V1_FINAL_FEATURES.md` still described the active project as
  Stage 28.9 / Stage 30-bound instead of current Stage 35.
- P1: `docs/ROADMAP.md` still documented the old opaque-call zero-gradient
  behavior.
- P2: `docs/lang/spec.md` still overclaimed general PTX support and contained
  stale f64 lowering limitations.
- P3: `docs/ROADMAP.md` still used an old exact test-count heading.

## Fix Plan

- Add an explicit f64 reflection path for `grad_rev_all` cell writes.
- Harden reverse-AD tape index validation in operation creation and backward
  propagation.
- Clamp scalar CE probabilities to the same safe open interval used by BCE.
- Reject negative softmax lengths without touching output cells.
- Mirror the main PTX emit surface-shaping pipeline in the standalone PTX CLI.
- Reject duplicate autotune keys.
- Update public docs and progress notes to reflect the current Stage 35 state.
