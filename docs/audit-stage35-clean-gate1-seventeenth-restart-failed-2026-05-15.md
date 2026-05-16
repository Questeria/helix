# Stage 35 Clean Gate 1 - Seventeenth Restart Findings

Date: 2026-05-15

Result: FAIL. Clean-gate count remains 0/3.

This audit was run after commit `0636982` (`Fix Stage 35 sixteenth restart
findings`). The focused smoke checks were green, and the widened audit protocol
reported multiple related findings per lane instead of stopping at the first
issue. The gate did not count as clean.

## Lane A - AD / NN / Runtime Correctness

- P2: public 2D tensor helpers still bypassed `t2d_len` overflow rejection in
  row-major loops and offset math. Affected families included 2D matvec/matmul,
  transpose, row/column sum, and identity construction.
- P2: neural-network helpers repeated the same unchecked row-major shape math
  for dense, softmax, classifier, argmax, accuracy, and batch CE helpers.

## Lane B - PTX / Tile / Autotune CLI Parity

- P2: PTX artifact emission lowered the full program before filtering kernels,
  so an unrelated host AD function using `D<...>` could block an otherwise
  clean kernel artifact.
- P3: direct `helixc.backend.ptx` did not drain AD warnings on exit, unlike
  `helixc.check --emit-ptx`.

## Lane C - Documentation / Status Consistency

- P3: old research/work-queue notes still contained historical IO and landed
  ticket wording that read like current defects.
- P3: the current progress note's prior docs-scan result needed to acknowledge
  that the restart 17 lane found additional historical wording outside the
  first regex scan.

## Fix Plan

- Add `t2d_len` guards to public 2D tensor and NN helpers before any row-major
  loops or offset derivation.
- In PTX mode, filter to kernel AST before lower/optimization/tile lowering,
  while still typechecking and validating the full program.
- Give the direct PTX CLI an exit-time AD-warning drain routed to stderr.
- Reword the stale historical docs and update the Stage 35 progress note.
