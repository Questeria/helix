# Stage 35 Clean Gate 1 - Twelfth Restart Findings

Date: 2026-05-15

Result: FAIL. Clean-gate count remains 0/3.

This audit was run after commit `78c2e0e` (`Fix Stage 35 eleventh restart
findings`). The focused smoke checks were green, but documentation audit found
stale live-stage claims, and local runtime review found sibling invalid-shape
matrix helpers in the same bug family as recent tensor/NN fixes. The gate did
not count as clean.

## Lane A - AD / NN / Runtime Correctness

- P2: neighboring integer/f32 matrix helpers still lacked the non-positive shape
  no-op guards now used by matvec and dense-layer paths. `ti2d_matmul`,
  `tf2d_matmul`, `tf2d_row_sum`, and `tf2d_col_sum` could write outputs even
  when shape metadata was invalid.

## Lane B - PTX / Tile / Autotune CLI Parity

- No returned finding before timeout. The lane was closed and will be restarted
  fresh instead of waiting indefinitely.

## Lane C - Documentation / Status Consistency

- P2: `docs/HELIX_V1_FINAL_FEATURES.md` and
  `docs/HELIX_FINAL_PRODUCT_RESEARCH.md` still called Stage 34 the current
  stage despite their surrounding Stage 35 status.
- P3: `docs/lang/hbs.md` used live-sounding historical-verification wording and
  a historical 501-test count without saying it was not current Stage 35 gate
  evidence.
- P3: `docs/APPROACH_A_PLAN.md` and `docs/APPROACH_A_DETAILED_PLAN.md` still
  presented old bootstrap-stage snapshots as canonical live tracking.

## Fix Plan

- Add invalid-shape no-op guards to the sibling matrix helpers and targeted
  regression tests proving outputs remain unchanged.
- Mark stale stage docs as historical/superseded and point live tracking to
  `docs/ROADMAP.md` plus `docs/stage35-progress-2026-05-15.md`.
- Restart the PTX/tooling lane fresh after this fix instead of counting the
  timed-out audit as evidence.
