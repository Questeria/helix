# Stage 34 Clean Gate 1 Index Repair And Chronology Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

Fresh clean-gate auditors on commit `b374615` found one proof-soundness issue
and two documentation discipline issues.

The proof-soundness issue was an overbroad indexed-repair rule. The checker
marked a named aggregate as unsafe after `xs[0] = bad`, but any clean indexed
write to the same aggregate cleared the whole marker. This made the following
program look clean even though `xs[0]` could still contain an unrepresentable
`f64` value:

```hx
type AlwaysF64 = f64 where true;
fn f(b: bool) -> AlwaysF64 {
    let mut xs = [0.0_f64, 0.0_f64];
    xs[0] = if b { 1e309_f64 } else { 0.0_f64 };
    xs[1] = 0.0_f64;
    xs[0]
}
```

The documentation issues were:

- The previous reflection-bound-comment and index-assignment findings were
  recorded as two sequential restarts even though both came from the same
  failed `c9f9606` clean-gate rotation.
- `helixc/tests/test_reflection.py` still had broad comment wording around the
  binary-classifier test.

## Fix

Simple static indexed evidence is now tracked per element. A bad write to
`xs[0]` marks the `xs[0]` evidence key. A clean write to `xs[1]` can clear only
the `xs[1]` evidence key, so a later `xs[0]` read remains fail-closed. A clean
write to the same static element, such as `xs[0] = 0.0_f64`, can still repair
that element.

The historical docs now describe the reflection-bound-comment and
index-assignment findings as one failed `c9f9606` rotation. The broad
binary-classifier wording was narrowed to the behavior under test.

## Verification

- Focused index-repair and reflection checks:
  `python -m pytest -q helixc/tests/test_typecheck.py::test_stage34_unrepresentable_scalar_evidence_covers_index_assignment helixc/tests/test_proof_artifact_gate.py::test_gate_rejects_unrepresentable_index_assignment_false_pass helixc/tests/test_proof_artifact_gate.py::test_gate_rejects_unrepresentable_index_assignment_wrong_index_repair helixc/tests/test_reflection.py::test_dogfood_05_binary_classifier`: `4 passed`.
- Stage 34 typecheck/proof-gate/reflection slice:
  `python -m pytest -q helixc/tests/test_typecheck.py -k "stage34" helixc/tests/test_proof_artifact_gate.py helixc/tests/test_reflection.py`: `38 passed, 296 deselected`.
- Broad metadata, proof, WSL-runtime helper, and reflection bundle:
  `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py helixc/tests/test_strings_io.py helixc/tests/test_reflection.py helixc/tests/test_select_codegen.py`: `531 passed`.
- `python scripts\stage31_validate.py --mode quick`: passed; snapshot check
  and compile returned `0`, and snapshot run returned `42`.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  passed across all 12 shards.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
