# Stage 34 Clean Gate 1 Generic Wrapper And Reflection Claims Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

Fresh auditors on commit `e27eb87` found one proof-soundness issue and one
documentation discipline issue.

The proof-soundness issue was a generic wrapper that returned a refined value:

```hx
type AlwaysF64 = f64 where true;
fn via[T](x: T) -> AlwaysF64 { x }
fn f() -> AlwaysF64 { via(1e309_f64) }
```

The previous generic call-argument fix checked deferred generic boundaries, but
used the formal parameter type `T` as the representability target. Because `T`
has no concrete numeric base, the check exited early and the proof artifact
gate accepted the wrapper as clean. Nested forms such as
`via(id(id(1e309_f64)))` had the same issue.

The documentation issue was that `helixc/tests/test_reflection.py` still had a
few broad demo comments around the self-improving-agent and verifier examples.

## Fix

- Unrepresentable scalar evidence now carries its erased numeric base through
  local evidence tracking.
- The Stage 34 representability check can recover that base from the expression
  itself when a formal argument type is still generic.
- Typecheck and proof-gate regressions now pin direct generic wrappers,
  local-let wrappers, and nested generic pass-through into a refined return.
- Reflection test comments were narrowed to describe the behavior under test,
  without broad marketing or AGI claims.
- The quick validation list includes the generic-wrapper regression.

## Verification

- Focused generic-wrapper and reflection regressions: `20 passed`.
- Stage 34 focused typecheck/CLI/proof-gate slice plus reflection tests:
  `56 passed`.
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py helixc/tests/test_strings_io.py helixc/tests/test_reflection.py helixc/tests/test_select_codegen.py`: `528 passed`.
- `python scripts\stage31_validate.py --mode quick`: passed.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  passed across all 12 shards with no retries.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
