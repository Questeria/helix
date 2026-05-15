# Stage 35 Clean Gate 1 - Thirteenth Restart Findings

Date: 2026-05-15

Result: FAIL. Clean-gate count remains 0/3.

This audit was run after commit `0fc1ad1` (`Fix Stage 35 twelfth restart
findings`). The focused smoke checks were green, but all clean-gate lanes found
actionable PTX, runtime, or documentation issues, so the gate did not count as
clean.

## Lane A - AD / NN / Runtime Correctness

- P2: f32 range helpers rejected negative lower bounds but still had no vector
  length, so positive over-range `hi` values could read guard slots after the
  vector.
- P2: square matrix helpers accepted only a single side length, so rectangular
  matrix storage could be treated as square and read later arena slots.

## Lane B - PTX / Tile / Autotune CLI Parity

- P2: direct PTX source reading caught `OSError` but allowed
  `UnicodeDecodeError` to escape as a Python traceback, unlike `helixc.check`
  which reports a clean encoding diagnostic.

## Lane C - Documentation / Status Consistency

- P2: `docs/APPROACH_A_DETAILED_PLAN.md` still called itself the canonical
  detailed plan despite its historical/superseded banner.
- P2: `docs/HELIX_V1_FINAL_FEATURES.md` and
  `docs/HELIX_FINAL_PRODUCT_RESEARCH.md` still over-described the current
  reflection surface instead of consistently using reflective-cell / quote
  scaffold wording.
- P3: `docs/lang/hbs.md` still used live "NOW" wording in a historical
  bootstrap snapshot.
- P3: `docs/lang/tutorial.md` used broad competitor-exclusivity wording for the
  all-in-one example.

## Fix Plan

- Make f32 range helpers length-aware and reject `hi > n`.
- Make square matrix helpers take `rows, cols` and no-op / return zero unless
  the shape is non-empty and square.
- Read direct PTX sources as UTF-8 explicitly and convert decode failures into
  clean exit-code-2 diagnostics.
- Reword the stale docs to historical/scaffold/differentiator language.
