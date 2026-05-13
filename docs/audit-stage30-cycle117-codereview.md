# Stage 30 Cycle 117 Code Review

Date: 2026-05-12

## Scope

Cycle 117 reviewed the patch for runtime correctness, regression coverage, and defensive behavior when callers use direct compiler internals instead of the normal checked CLI path.

## Findings

PASS, p_pass 0.93, confidence HIGH.

No concrete remaining Cycle 117 bugs were found after refresh.

Validated points:

- Indexed assignment now resolves the target element type, so bad stores and bad compound stores are rejected.
- Compound assignment, wrapped TyDiff/TyLogic arithmetic, and plain binary operators now share the same inner scalar domain rules.
- Direct lower_ast rejects mixed array literals after lowering element values if typecheck was bypassed.
- Direct lower_ast rejects unsupported tensor/tile indexing in expression and assignment target positions.
- Supported one-index HBM tile load/store still lowers successfully.
- Backend CLI now fails closed for struct monomorphization and panic validation errors.
- PTX emission now uses real Tile IR instead of an empty placeholder module.

## Verification

- `python -m py_compile helixc\frontend\typecheck.py helixc\ir\lower_ast.py helixc\backend\x86_64.py helixc\check.py helixc\tests\test_typecheck.py helixc\tests\test_ir.py helixc\tests\test_cli.py`
- `python -m pytest helixc\tests\test_typecheck.py -k "c117 or c116" -q --tb=short` -> 12 passed
- `python -m pytest helixc\tests\test_ir.py -k "c117" -q --tb=short` -> 2 passed
- `python -m pytest helixc\tests\test_cli.py -k "c117" -q --tb=short` -> 4 passed
- `python -m pytest helixc\tests\test_typecheck.py -q` -> 126 passed
- `python -m pytest helixc\tests\test_ir.py -q` -> 60 passed
- `python -m pytest helixc\tests\test_cli.py -q` -> 48 passed
- `python -m pytest helixc\tests\test_ptx.py -q` -> 23 passed
- `python -m pytest helixc\tests\test_const_fold.py -q` -> 54 passed
- `python -m pytest helixc\tests\test_provenance.py -q` -> 13 passed
- `python -m pytest helixc\tests\test_codegen.py -k "c111 or c112 or c115 or c116 or u64 or usize or i64_to_f64_then_back or f64_cast_target or shift" -q --tb=short` -> 36 passed, 668 deselected
- Bootstrap gate -> 18 passed, 686 deselected in 320.60s.
