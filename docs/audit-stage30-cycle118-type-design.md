# Stage 30 Cycle 118 Type-Design Audit

Date: 2026-05-12

## Scope

Cycle 118 checked the remaining type-contract edges after Cycle 117: unary operator domains, assignment target validity, HBM tile assignment semantics, and PTX/lowering agreement for kernel tile indices.

## Findings

PASS, p_pass 0.93, confidence HIGH.

Resolved blockers:

- HBM tile compound assignment now fails during typecheck instead of passing typecheck and failing later in lowering.
- Unary operators now enforce their domains: `!` requires bool, `~` requires integer scalar, and unary `-` requires int or float scalar.
- Scalar variable assignment now requires a mutable binding.
- Invalid assignment targets such as `1 = 2` are rejected.
- Existing array-index assignment behavior remains covered so current Helix array tests continue to pass.

No blocking type-contract findings remained after refresh.

## Verification

- `python -m pytest helixc\tests\test_typecheck.py -k "c118 or c117" -q --tb=short` -> 11 passed
- `python -m pytest helixc\tests\test_ir.py -k "c118 or c117_direct_lower_rejects_unsupported_tensor_tile_indexing" -q --tb=short` -> 2 passed
- `python -m pytest helixc\tests\test_ptx.py -k "c118 or hbm_tile" -q --tb=short` -> 4 passed
- `python -m pytest helixc\tests\test_typecheck.py -q` -> 128 passed
- `python -m pytest helixc\tests\test_ir.py -q` -> 61 passed
- `python -m pytest helixc\tests\test_ptx.py -q` -> 25 passed
- `python -m pytest helixc\tests\test_cli.py -q` -> 48 passed
- `python -m pytest helixc\tests\test_const_fold.py -q` -> 54 passed
- `python -m pytest helixc\tests\test_provenance.py -q` -> 13 passed
- Bootstrap gate -> 18 passed, 686 deselected in 324.50s.

## Notes

This cycle keeps the current array-index assignment model intact while tightening scalar assignment. That avoids breaking existing array tests while closing the silent no-op scalar assignment path.
