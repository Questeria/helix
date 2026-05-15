# Stage 34 Clean Gate 1 Const Indirection And Replay Libs Restart

Date: 2026-05-15
Stage: 34
Gate: Clean gate 1
Result: Failed, fixed, and reset to 0/3 clean gates

## Findings

Fresh auditors on commit `e879f48` found two more Stage 34 issues.

1. Named constants could hide an unrepresentable typed scalar source. A top-level
   or local `const OVER: f64 = (3.4028235e38_f32 * 2.0_f32) as f64;` could be
   returned as `AlwaysF64 where true` and still produce a clean proof artifact,
   even though the inner `f32 * f32` operation overflowed before the `f64` cast.

2. The clean proof gate rejected output/debug flags but still accepted `-l`
   library inputs. The validator also trusted artifact `input.libs` enough to
   replay `-l <lib>` during source recomputation.

## Fix

- The typechecker now tracks top-level and local constants whose source scalar
  expression is known to contain an unrepresentable typed value. Later
  refinement checks treat references to those constants as unrepresentable
  proof sources, including self-independent refinements such as `where true`.
- Function final-expression refinement checking now happens while the block's
  local constant scope is still visible, so local const evidence cannot vanish
  before return proof validation.
- `proof_artifact_gate.py` now rejects `-l` inputs, and
  `proof_artifact_validate.py` requires `input.libs` to be empty for proof
  replay instead of reconstructing library arguments from artifacts.

## Verification

- Focused latest-reset regressions: `4 passed`.
- Wider targeted regressions: `7 passed`.
- `python scripts\stage31_validate.py --mode quick`: passed.
- `python -m pytest -q helixc/tests/test_typecheck.py helixc/tests/test_proof_artifact_validate.py helixc/tests/test_proof_artifact_gate.py`: `342 passed`.
- `python scripts\stage31_validate.py --mode full --skip-snapshot --shards 8`:
  passed after built-in retry recovered no-codegen shard 3.

## Gate State

The clean-gate counter remains reset to `0/3`. A fresh clean gate should start
from the commit containing this fix set.
