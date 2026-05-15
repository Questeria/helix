# Stage 34 Clean Gate 1 Generic Call And Archive Repro Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

Fresh auditors on commit `8a497f4` found one proof-soundness issue and two
clean-gate reproducibility issues.

The proof-soundness issue was a generic pass-through hiding unrepresentable
scalar evidence before it reached a refined-return call:

```hx
type AlwaysF64 = f64 where true;
fn id[T](x: T) -> T { x }
fn accept(x: f64) -> AlwaysF64 { x }
fn f() -> AlwaysF64 { accept(id(1e309_f64)) }
```

The direct `accept(1e309_f64)` case failed correctly, but the generic
`id[T]` call left the argument type as `T`, so the Stage 34 call-boundary
representability check did not run. The proof artifact gate accepted the
result as clean.

The reproducibility issues were:

- `git archive` extracted shell scripts with CRLF line endings, so
  `bash scripts/run_all_tests.sh` failed before running tests.
- Some WSL runtime test helpers hardcoded `/mnt/c/Projects/Kovostov-Native`,
  so archive-copy tests could execute binaries from the live checkout instead
  of the extracted archive.

## Fix

- Stage 34 call-boundary representability checks now also run across deferred
  generic `TyVar` and `TySize` argument/parameter boundaries when the callee
  can return a refined value.
- Regression tests now pin generic pass-through forms in both typechecking and
  proof artifact gate coverage.
- The quick validation list includes the new generic call-argument regression.
- `.gitattributes` pins shell scripts to LF line endings for archive exports.
- WSL runtime helpers in strings I/O, reflection, and select-codegen tests now
  derive the WSL path from the current checkout path and use unique temporary
  executable names, avoiding live-checkout path trust and parallel shard file
  collisions.

## Verification

- Focused latest-reset and archive-helper regressions: `7 passed`.
- Stage 34 focused typecheck/CLI/proof-gate slice: `55 passed`.
- Direct archive-repro tests for shard guards and strings I/O: `5 passed`.
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_cli.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py helixc/tests/test_strings_io.py helixc/tests/test_reflection.py helixc/tests/test_select_codegen.py`: `526 passed`.
- `python scripts\stage31_validate.py --mode quick`: passed.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  passed across all 12 shards with no retries.
- Staged-tree archive check for shell scripts: `scripts/run_all_tests.sh`,
  `stage0/hex0/run_tests.sh`, and `stage0/hex0/build.sh` extracted with
  `CRLF=0`; `bash -n` accepted the shell scripts, and archive-copy
  `test_print_int_zero` passed.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
