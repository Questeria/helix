# Stage 34 Clean Gate 1 Fixed-Width Arithmetic And Embedded Carry Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

Fresh auditors on commit `4be3e3c` found three issues.

1. Integer refinement predicate arithmetic used Python real division for `/`,
   while Helix integer division truncates toward zero. This allowed
   `self / 2 > 0` to prove true for `self = 1`.

2. Integer affine proof-carry extraction used real-number algebra over
   fixed-width machine integers. Division was directly unsound, and addition or
   multiplication can also become unsound when overflow would affect runtime
   machine semantics.

3. Plain `proof_artifact_validate.py` recomputed carried-proof metadata for an
   explicit `--source`, but skipped that recomputation when the artifact only
   supplied a valid embedded `path`. A source-backed artifact could therefore
   erase `proof_carries` and update the summary to match when validated without
   an explicit source argument.

A docs auditor also found stale wording in the Stage 34 progress file that
still described old float-looking affine examples as currently accepted.

## Fix

- Direct integer predicate arithmetic now uses Helix-style truncating division
  and modulo.
- Direct integer predicate arithmetic fails closed if an operation leaves the
  declared fixed-width integer range.
- Affine proof-carry extraction fails closed for all fixed-width numeric bases,
  leaving simple direct `self` versus constant bounds as the supported subset.
- Source-backed proof artifact validation now recomputes carried-proof metadata
  from either an explicit `--source` or a resolvable embedded artifact `path`.
- The progress doc now marks old affine examples as design intent, not current
  accepted behavior.

## Verification

- Focused integer-division, fixed-width-affine, embedded-source forgery, and
  docs-related regressions: `9 passed`.
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: passed.
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`: `473 passed`.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`: passed after the built-in retry recovered one no-codegen shard.
- `python -m pytest -q helixc/tests/test_strings_io.py::test_print_int_decimal_output`: passed after inspecting the recovered shard's transient failure.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
