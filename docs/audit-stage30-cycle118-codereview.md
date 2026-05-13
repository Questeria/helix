# Stage 30 Cycle 118 Code Review

Date: 2026-05-12

## Scope

Cycle 118 reviewed the implementation patch for direct PTX CLI behavior, defensive lower_ast behavior, assignment-target checking, and regression coverage.

## Findings

PASS, p_pass 0.93, confidence HIGH.

No concrete remaining Cycle 118 bugs were found after refresh.

Validated points:

- Direct PTX CLI now shares the typecheck fail-closed contract used by `check.py --emit-ptx`.
- Direct lowering rejects unsupported tensor/tile indexing in both expression and assignment forms, including expression callees.
- PTX HBM indexing no longer emits placeholder register fallbacks for missing index/store value operands.
- Unary and assignment-target typecheck rules are covered by focused regressions.
- Existing HBM tile load/store and array assignment tests continue to pass.

## Verification

- `python -m py_compile helixc\frontend\typecheck.py helixc\ir\lower_ast.py helixc\backend\ptx.py helixc\tests\test_typecheck.py helixc\tests\test_ir.py helixc\tests\test_ptx.py`
- `python -m pytest helixc\tests\test_typecheck.py -k "c118 or c117" -q --tb=short` -> 11 passed
- `python -m pytest helixc\tests\test_ir.py -k "c118 or c117_direct_lower_rejects_unsupported_tensor_tile_indexing" -q --tb=short` -> 2 passed
- `python -m pytest helixc\tests\test_ptx.py -k "c118 or hbm_tile" -q --tb=short` -> 4 passed
- `python -m pytest helixc\tests\test_typecheck.py -q` -> 128 passed
- `python -m pytest helixc\tests\test_ir.py -q` -> 61 passed
- `python -m pytest helixc\tests\test_ptx.py -q` -> 25 passed
- `python -m pytest helixc\tests\test_cli.py -q` -> 48 passed
- `python -m pytest helixc\tests\test_const_fold.py -q` -> 54 passed
- `python -m pytest helixc\tests\test_provenance.py -q` -> 13 passed
- `python -m pytest helixc\tests\test_codegen.py -k "c111 or c112 or c115 or c116 or c117 or c118 or u64 or usize or i64_to_f64_then_back or f64_cast_target or shift or array_assign" -q --tb=short` -> 37 passed, 667 deselected
- Bootstrap gate -> 18 passed, 686 deselected in 324.50s.
