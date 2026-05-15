# Stage 35 Clean Gate 1 - Fifteenth Restart Findings

Date: 2026-05-15

Result: FAIL. Clean-gate count remains 0/3.

This audit was run after commit `795f8aa` (`Fix Stage 35 fourteenth restart
findings`). The focused smoke checks were green, but all clean-gate lanes found
actionable runtime, PTX/tooling, or documentation issues, so the gate did not
count as clean.

## Lane A - AD / NN / Runtime Correctness

- P2: `tf2d_diag` and `tf2d_trace` rejected rectangular shapes but did not
  reject positive square shapes whose `rows * cols` length overflows. That let
  them compute diagonal offsets directly even though `t2d_len` had already
  defined such shapes as invalid.

## Lane B - PTX / Tile / Autotune CLI Parity

- P2: `helixc.check --emit-ptx` routed some pre-PTX validation failures to
  stdout, including autotune and typecheck diagnostics, so failed artifact-mode
  invocations could still contaminate stdout.

## Lane C - Documentation / Status Consistency

- P3: `docs/research-log.md` still used unsupported competitor-exclusivity
  wording for a historical Phase 3 milestone.
- P3: `docs/research/WORK_QUEUE.md` still read like a live foreground queue and
  carried stale test-count/projection instructions without a snapshot banner.

## Fix Plan

- Make diagonal extraction and trace call `t2d_len(rows, cols)` before looping,
  treating a zero length for positive square shapes as overflow rejection.
- Route non-artifact diagnostics through stderr when `--emit-ptx` reserves
  stdout for PTX, and pin both autotune and typecheck failure cases.
- Mark the old research/work-queue docs as historical snapshots and remove
  unsupported current-exclusivity wording.
