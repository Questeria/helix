# Stage 34 Clean Gate 1 Failed Audit - Self-Independent Predicates

This audit attempt did not count as a clean gate. Two auditors reproduced a
real proof-honesty bug on commit `01aefd8`.

## Finding

Self-independent predicates such as `where true` could prove a refined value
even when the value was non-finite or otherwise not representable by the erased
target type.

Minimal shape:

```hx
type AlwaysF64 = f64 where true;
type AlwaysInt = i32 where true;
fn literal_bad() -> AlwaysF64 { 1e309_f64 }
fn cast_bad() -> AlwaysInt { 1e309_f64 as AlwaysInt }
fn main() -> i32 { 0 }
```

Observed before the fix:

- `--emit-proof-obligations --no-stdlib` exited `0`
- artifact `summary.typecheck_errors` was `0`
- artifact recorded proved obligations
- artifact could record a later `same-refinement` proof carry
- `proof_artifact_gate.py` accepted the artifact as clean

Root cause: when target representation conversion returned `None`, the checker
only emitted an error if `_check_self_independent_refinement` returned pending
`self`-dependent predicates. `where true` produced no pending predicates, so an
unrepresentable value could pass as refined.

## Fix Summary

- Known constant values that cannot be represented by the erased target type
  now fail closed even when every predicate is self-independent.
- Failed refined-return functions are tracked, and later direct calls or
  function references resolve their return type with refinements erased.
- Function body checking now reaches a fixed point over failed refined-return
  producers, so callers declared before a failed producer also fail closed.
- Added CLI, typecheck, and proof-gate regressions for the unrepresentable
  self-independent cases.
- Added the new regressions to the quick validation gate.

## Verification

- Exact focused regressions: `3 passed`
- Nearby proof-carry and proof-gate slice: `29 passed`
- `python scripts\stage31_validate.py --mode quick --skip-snapshot`: pass
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`:
  `447 passed`
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  pass after built-in retry recovered no-codegen shard 1

Clean-gate counter remains reset to `0`.
