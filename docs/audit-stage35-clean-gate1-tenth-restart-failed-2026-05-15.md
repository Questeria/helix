# Stage 35 Clean Gate 1 - Tenth Restart Findings

Date: 2026-05-15

Result: FAIL. Clean-gate count remains 0/3.

This audit was run after commit `6c11daf` (`Fix Stage 35 ninth restart
findings`). The focused smoke checks were green, but all clean-gate lanes still
found runtime, direct-PTX, or documentation consistency issues, so the gate did
not count as clean.

## Lane A - AD / NN / Runtime Correctness

- P1: negative column counts still allowed `ti2d_matvec`, `tf2d_matvec`, and
  dense-layer helpers to write zeroed outputs when shape metadata should have
  made the operation a no-op.

## Lane B - PTX / Tile / Autotune CLI Parity

- P2: direct PTX leaked a traceback when strict stdlib loading hit a missing
  stdlib file.
- P3: direct PTX with no arguments treated the request as stdin input and failed
  later with a kernel error instead of returning a bad-invocation exit code.

## Lane C - Documentation / Status Consistency

- P1: `docs/lang/agi-features.md` implied `@effect(modify_self)` could already
  rewrite source, rather than describing the current capability boundary and the
  future source-rewrite target separately.
- P2: memory-tier docs described broad first-class `consolidate`, `recall`,
  and `retrieve` runtime behavior and compiler-enforced timestamp/dedup
  invariants, while Stage 35 currently has type-level wrappers plus selected
  builtin checks.
- P3: `docs/ROADMAP.md` listed four dogfood programs even though the current
  dogfood set has five tests/programs.

## Fix Plan

- Treat non-positive matrix rows or columns as empty/no-op shapes before matvec
  or dense-layer helpers write outputs.
- Make direct PTX no-argument invocation return exit code `2`, and convert
  strict stdlib `FileNotFoundError` into a clean diagnostic without a traceback.
- Reword current-vs-future documentation around `modify_self`, memory tiers,
  and the dogfood count.
