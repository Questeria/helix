# Stage 34 Clean Gate 1 Primitive Producer And Coverage Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

Fresh auditors on commit `02007a5` found one proof-soundness issue and one
coverage/documentation mismatch.

The proof-soundness issue was a primitive-return producer false pass:

```hx
type AlwaysF64 = f64 where true;
fn raw_bad() -> f64 { 1e309_f64 }
fn f() -> AlwaysF64 { raw_bad() }
```

The primitive producer `raw_bad` could hide an unrepresentable scalar from the
later refined return proof in `f`, allowing the self-independent refinement to
look clean.

The coverage mismatch was that the previous progress note claimed
unrepresentable scalar detection walked blocks, tuples, arrays, structs, fields,
indexes, calls, and assignments, but tests only pinned a narrower subset.

A validator trust-boundary auditor did not find a proof artifact
gate/validator bypass on `02007a5`.

## Fix

- Primitive-return functions that produce unrepresentable typed constant
  scalars are now tracked as unsafe proof sources.
- Calls to those primitive producers are treated as unrepresentable evidence
  when later used to prove refined values.
- Primitive producer tracking does not emit an immediate type error for plain
  primitive returns, preserving older const-fold diagnostics when no refined
  proof depends on the value.
- Regression tests now pin primitive producers, explicit primitive returns,
  local call consumers, block final expressions, tuple elements, array
  elements, struct fields, field/index access, call arguments, assignment
  evidence, and assignment repair/clear behavior.
- The proof artifact gate now rejects the primitive-producer false-pass shape.
- The quick validation list now includes the new Stage 34 coverage tests.

## Verification

- Focused latest-reset regressions plus const-fold compatibility check:
  `4 passed`.
- Stage 34 focused typecheck/CLI/proof-gate slice: `53 passed`.
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`: `493 passed`.
- `python scripts\stage31_validate.py --mode quick`: passed.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  passed after built-in retry recovered no-codegen shard 1.
- `python -m pytest -q helixc/tests/test_strings_io.py::test_print_int_zero`:
  passed after inspecting the recovered shard's transient failure.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
