# Stage 34 Clean Gate 1 Index Assignment Finding

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates
Rotation: Same failed `c9f9606` clean-gate rotation as the reflection bound
comment finding; this was not a separate clean-gate restart.

## Finding

Fresh proof-soundness auditors on commit `c9f9606` found that indexed
assignments could hide unrepresentable scalar evidence:

```hx
type AlwaysF64 = f64 where true;
fn f(b: bool) -> AlwaysF64 {
    let mut xs = [0.0_f64];
    xs[0] = if b { 1e309_f64 } else { 0.0_f64 };
    xs[0]
}
```

Plain name assignments already updated local unrepresentable evidence, but
`xs[0] = ...` did not mark the aggregate `xs`. A later `xs[0]` read therefore
looked clean and could prove the self-independent refinement.

## Fix

When a simple static indexed assignment writes unrepresentable scalar evidence
into a named aggregate, the checker now marks that static element as carrying
the evidence. A later read of the same element sees the marker and refined
proof checking fails closed. A repair assignment, such as `xs[0] = 0.0_f64`,
clears only that static element for simple local aggregate cases.

## Verification

- Focused index-assignment and reflection checks:
  `python -m pytest -q helixc/tests/test_typecheck.py::test_stage34_unrepresentable_scalar_evidence_covers_index_assignment helixc/tests/test_proof_artifact_gate.py::test_gate_rejects_unrepresentable_index_assignment_false_pass helixc/tests/test_reflection.py::test_verifier_can_bound_state`: `3 passed`.
- Stage 34 focused typecheck/CLI/proof-gate slice plus reflection tests:
  `57 passed`.
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py helixc/tests/test_strings_io.py helixc/tests/test_reflection.py helixc/tests/test_select_codegen.py`: `530 passed`.
- `python scripts\stage31_validate.py --mode quick`: passed.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  passed across all 12 shards with no retries.

## Gate State

The clean-gate counter remained reset to `0/3` after this historical finding.
