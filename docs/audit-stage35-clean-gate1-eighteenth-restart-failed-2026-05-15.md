# Stage 35 Clean Gate 1 - Eighteenth Restart Findings

Date: 2026-05-15

Result: FAIL. Clean-gate count remains 0/3.

This audit was run after commit `017c873` (`Fix Stage 35 seventeenth restart
findings`). Focused smoke checks were green and the docs/status lane was clean,
but the runtime and PTX/CLI lanes found remaining bugs. The gate did not count
as clean.

## Smoke Verification Before Audit

- Reverse-AD focused smoke: 5 passed.
- Tensor/NN focused smoke: 5 passed.
- PTX/CLI focused smoke: 6 passed.
- Stdlib parser sweep: parsed 16 stdlib files.
- Full PTX tests: 70 passed.
- `emit_ptx` CLI slice: 17 passed.
- Full reverse-AD tests: 29 passed.
- Stage 35 regression slice: 37 passed.
- Tensor/NN family slice: 37 passed.
- Combined CLI/PTX suite: 221 passed.

## Lane A - AD / NN / Runtime Correctness

- P2: direct 2D tensor accessors still computed row-major offsets as
  `start + i * cols + j` without overflow or negative-index guards. Runtime
  probes showed `ti2d_set/get` and `tf2d_set/get` could alias slot 0 with
  overflowing inputs such as `cols = 65536, i = 65536, j = 0`.

## Lane B - PTX / Tile / Autotune CLI Parity

- P2: `helixc.check --emit-ptx -Wad=error` emitted a valid-looking PTX artifact
  on stdout and then exited with status 1 after the final AD-warning drain
  promoted the warning to an error. Failed artifact invocations must not leave
  usable artifact text on stdout.

## Lane C - Documentation / Status Consistency

- Clean in the replacement docs/status lane. It confirmed Stage 35 remains open,
  clean gates are still `0/3`, restart 17 docs were captured, and current
  Stage 35 test-count claims match the shallow selector counts.

## Fix Plan

- Drain AD warnings inside the `--emit-ptx` branch after PTX generation but
  before printing the artifact. In warning mode this keeps PTX on stdout and
  diagnostics on stderr; in `-Wad=error` mode it returns before printing PTX.
- Add a subprocess regression that requires stdout to be empty for
  `--emit-ptx -Wad=error`.
- Add a shared checked `t2d_offset` helper and route direct `ti2d_*` and
  `tf2d_*` accessors through it.
- Add behavioral regressions for overflowing and negative direct 2D accessor
  offsets.
