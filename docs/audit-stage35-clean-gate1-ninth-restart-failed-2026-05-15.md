# Stage 35 Clean Gate 1 - Ninth Restart Findings

Date: 2026-05-15

Result: FAIL. Clean-gate count remains 0/3.

This audit was run after commit `ce5a3fc` (`Fix Stage 35 eighth restart
findings`). The code smoke checks were green, but all clean-gate lanes still
found real parity, metadata, or documentation issues, so the gate did not count
as clean.

## Lane A - AD / NN / Runtime Correctness

- P1: corrupting adjoint capacity metadata upward reopened out-of-bounds
  `rev_seed` writes and `rev_grad` reads beyond the real adjoint array.

## Lane B - PTX / Tile / Autotune CLI Parity

- P2: standalone direct PTX bad-invocation paths returned exit code `1` for
  unknown flags, extra paths, and missing files, while `helixc.check` uses exit
  code `2` for those invocation errors.

## Lane C - Documentation / Status Consistency

- P1: AGI feature docs described reflection, `modify`, and auto-curriculum
  design targets in current-tense wording despite current stub/type-level
  behavior.
- P2: the spec still made reflection look "likely never" while other docs track
  verifier-gated reflection as a target/scaffold.
- P2: the spec's GPU/tile example was not clearly marked as design-target code.
- P3: the spec carried historical 2026-05-04 test status in a way that could be
  mistaken for the current Stage 35 gate claim.
- P3: the roadmap still used a broad AGI win comparison.

## Fix Plan

- Add redundant adjoint metadata guards and validate them before `rev_seed`,
  `rev_grad`, or `rev_backward` trusts an adjoint capacity.
- Return direct PTX exit code `2` for invocation errors.
- Reword AGI docs/spec/roadmap so current stub or type-level surfaces and future
  targets are explicitly separated.
