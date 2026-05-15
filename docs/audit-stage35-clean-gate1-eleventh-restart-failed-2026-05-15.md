# Stage 35 Clean Gate 1 - Eleventh Restart Findings

Date: 2026-05-15

Result: FAIL. Clean-gate count remains 0/3.

This audit was run after commit `3ace998` (`Fix Stage 35 tenth restart
findings`). The focused smoke checks were green, but all clean-gate lanes still
found runtime, direct-PTX, or documentation consistency issues, so the gate did
not count as clean.

## Lane A - AD / NN / Runtime Correctness

- P1: `dense_layer_f32_grad_x` wrote zeroed gradients when `rows <= 0` but
  `cols > 0`, silently clobbering caller output buffers for invalid shape
  metadata.
- P1: f32 range/offset helpers accepted negative lower bounds or offsets, which
  allowed logical reads before the tensor start when an earlier arena slot
  existed.

## Lane B - PTX / Tile / Autotune CLI Parity

- P2: direct PTX still treated allowed-flags-only invocations such as
  `--strict` or `--stdlib` as stdin compilation requests, returning a later
  compile error instead of a bad-invocation exit code.

## Lane C - Documentation / Status Consistency

- P2: `docs/ROADMAP.md` still said tile types did not lower, contradicting the
  current Phase-0 PTX support for 1D HBM `tile<f32, ...>` /
  `tile<i32, ...>` kernels plus scalar ops.

## Fix Plan

- Guard `dense_layer_f32_grad_x` on non-positive rows or columns before writing
  `grad_x`.
- Make f32 range/offset helpers return sentinel or zero values for negative
  lower bounds, negative offsets, or non-positive lengths before arena access.
- Require a direct PTX source path after flag parsing unless an explicit stdin
  sentinel is added later.
- Reword roadmap tensor-codegen status to separate current Phase-0 PTX tile
  support from broader future GPU lowering work.
