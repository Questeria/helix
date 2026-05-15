# Stage 35 Clean Gate 1 - Sixth Restart Findings

Date: 2026-05-15

Result: FAIL. Clean-gate count remains 0/3.

This audit was run after commit `03938c0` (`Fix Stage 35 fifth restart
findings`). The code-focused local verification was green, but all three
read-only audit lanes found remaining issues, so the gate did not count as
clean.

## Lane A - AD / NN / Runtime Correctness

- P1: `grad_rev_all` f64 writes still had a constant-gradient corruption path.
  When reverse AD simplified a derivative to an unsuffixed float literal, the
  literal lowered as f32 but was written through `modify_f64`.
- P1: `rev_seed` and `rev_grad` accepted invalid adjoint indices, allowing
  negative or oversized indices to read/write outside the adjoint array.
- P2: negative-length hardening covered `softmax_layer` and `tf1d_max`, but
  sibling tensor/NN helpers still treated `n < 0` as non-empty.

## Lane B - PTX / Tile / Autotune CLI Parity

- P1: standalone `python -m helixc.backend.ptx` ignored `--strict` and did not
  run IR effect checking, so strict builds could pass when
  `helixc.check --emit-ptx --strict` would reject them.
- P2: the direct PTX CLI parsed without the bundled stdlib by default, unlike
  `helixc.check --emit-ptx`.
- P3: the direct PTX CLI leaked raw parse/lex tracebacks where the main check
  path renders user-facing diagnostics.

## Lane C - Documentation / Status Consistency

- P1: `docs/lang/agi-features.md` presented the future `bf16` SMEM/REG tile
  matmul design as if it were current behavior.
- P2: `docs/stage35-progress-2026-05-15.md` listed increment 24 before
  increment 23, making the visible chronology stale.
- P3: the remaining-work table still described all tile codegen and scalar
  `grad` as future, despite current Phase-0 PTX and scalar AD surfaces.

## Fix Plan

- Suffix generated f64 gradient float literals before `modify_f64` emission.
- Store adjoint capacity metadata and guard `rev_seed`/`rev_grad`.
- Harden sibling negative-length helpers to return empty/sentinel values.
- Add strict flag handling, default stdlib parity, effect checks, and
  user-facing parse/lex diagnostics to the direct PTX CLI.
- Clarify AGI feature docs and restore chronological progress ordering.
