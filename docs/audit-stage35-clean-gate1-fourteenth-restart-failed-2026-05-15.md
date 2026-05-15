# Stage 35 Clean Gate 1 - Fourteenth Restart Findings

Date: 2026-05-15

Result: FAIL. Clean-gate count remains 0/3.

This audit was run after commit `a264e49` (`Fix Stage 35 thirteenth restart
findings`). The focused smoke checks were green, but all clean-gate lanes found
actionable documentation, PTX/tooling, or runtime issues, so the gate did not
count as clean.

## Lane A - AD / NN / Runtime Correctness

- P2: reverse-mode AD silently dropped gradients through match pattern bindings
  that alias a differentiable scrutinee, such as `match x { y => y * y }`.
- P2: 2D tensor length computation could overflow positive shape metadata,
  causing zero-slot allocation and aliasing with the next arena allocation.

## Lane B - PTX / Tile / Autotune CLI Parity

- P2: `helixc.check --emit-ptx` printed progress text to stdout before the PTX
  module, so redirected output was not a pure PTX artifact.

## Lane C - Documentation / Status Consistency

- P2: `docs/lang/tutorial.md` still said the tutorial ended with every unique
  compile-time AGI feature and used a stale example count.
- P3: `docs/research/WAVE1_FINDINGS.md` contained unsupported competitor
  exclusivity wording.
- P3: pre-Stage-29 research docs carried historical stage/test-count context
  without an explicit superseded snapshot banner.

## Fix Plan

- Fail closed for reverse-mode AD match pattern bindings that alias a
  differentiable scrutinee until alias propagation is implemented.
- Make 2D tensor length overflow-aware and allocate a sentinel slot for
  positive overflowed constructors so the returned handle cannot alias the next
  allocation.
- Route `helixc.check --emit-ptx` progress/status output to stderr so stdout
  starts with the PTX module.
- Reword the stale documentation as historical or differentiator-target
  language.
