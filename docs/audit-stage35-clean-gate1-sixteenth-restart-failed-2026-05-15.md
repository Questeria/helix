# Stage 35 Clean Gate 1 - Sixteenth Restart Findings

Date: 2026-05-15

Result: FAIL. Clean-gate count remains 0/3.

This audit was run after commit `154ceb4` (`Fix Stage 35 fifteenth restart
findings`). The focused smoke checks were green, but all clean-gate lanes found
actionable reverse-AD, PTX/tooling, or documentation issues, so the gate did not
count as clean.

## Lane A - AD / NN / Runtime Correctness

- P1: reverse-mode AD's match-binder fail-closed check only detected direct
  `Name`/`Field` scrutinees. Compound scrutinees such as `x + 1.0` or
  `m.w + 1.0` still allowed pattern bindings that alias differentiable values,
  silently returning zero gradients.

## Lane B - PTX / Tile / Autotune CLI Parity

- P2: `helixc.check --emit-ptx` still allowed the final AD-warning summary to
  print to stdout after parsed invocations, contaminating artifact stdout on
  AD-warning paths.

## Lane C - Documentation / Status Consistency

- P3: `docs/ROADMAP.md` still used outdated exit-code-only IO wording, even
  though basic diagnostic stdout and narrow file builtins now exist.

## Fix Plan

- Make reverse-mode AD dependency detection recursively inspect compound
  scrutinees before allowing match pattern bindings.
- Route the final AD-warning drain summary to stderr for `--emit-ptx`, matching
  the artifact stdout policy used by the rest of `helixc.check`.
- Reword the roadmap to distinguish existing basic IO from richer
  capability-typed dataset/checkpoint workflows that remain Stage 35 work.
