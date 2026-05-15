# Stage 35 Clean Gate 1 - Seventh Restart Findings

Date: 2026-05-15

Result: FAIL. Clean-gate count remains 0/3.

This audit was run after commit `c6cdf84` (`Fix Stage 35 sixth restart
findings`). The local verification was green, but all three fresh read-only
audit lanes found remaining issues, so the gate did not count as clean.

## Lane A - AD / NN / Runtime Correctness

- P2: `tf1d_is_empty` still treated negative lengths as non-empty while the
  integer mirror had already been hardened to `n <= 0`.

## Lane B - PTX / Tile / Autotune CLI Parity

- P1: standalone direct PTX `--strict` falsely failed clean default-stdlib
  kernels because effect diagnostics were run over bundled stdlib helpers
  instead of the scoped user/live function set used by `helixc.check`.
- P2: direct PTX rejected the compatibility `--stdlib` flag accepted by
  `helixc.check --emit-ptx`.
- P3: direct PTX missing-file diagnostics leaked a Python traceback.

## Lane C - Documentation / Status Consistency

- P2: `docs/lang/agi-features.md` still used broad public uniqueness language
  such as "no other language has" and absolute competitor claims.
- P2: `docs/ROADMAP.md` still used absolute moat language for future
  provenance-typed neuro-symbolic primitives.
- P3: `docs/lang/spec.md` still carried the old `2026-05-03` date despite
  current Stage 35 content.

## Fix Plan

- Treat negative f32 tensor lengths as empty in `tf1d_is_empty` and cover it in
  the negative-length regression.
- Scope direct PTX effect checking when bundled stdlib is included, accept
  `--stdlib` as a no-op compatibility flag, and catch input read failures.
- Reframe public docs around differentiator targets and update the living spec
  date.
