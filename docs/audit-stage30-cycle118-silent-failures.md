# Stage 30 Cycle 118 Silent-Failures Audit

Date: 2026-05-12

## Scope

Cycle 118 audited fail-closed behavior around the direct PTX entry point, unsupported tensor/tile indexing, and backend fallbacks that could silently emit PTX with placeholder registers.

## Findings

PASS, p_pass 0.91, confidence HIGH.

Resolved blockers:

- Direct `python -m helixc.backend.ptx` now runs frontend typecheck before lowering and exits non-zero on type errors.
- Direct PTX CLI now catches lowering/backend exceptions and reports them without a Python traceback.
- Direct lower_ast now rejects unsupported tensor/tile indexing when the indexed callee is an expression such as `id(a)[0]`, not only simple names.
- PTX HBM tile load/store no longer silently substitutes `%r0` or `%f0` when an index or store-value register is missing.

No blocking silent-failure findings remained after refresh.

## Verification

- `python -m py_compile helixc\frontend\typecheck.py helixc\ir\lower_ast.py helixc\backend\ptx.py helixc\tests\test_typecheck.py helixc\tests\test_ir.py helixc\tests\test_ptx.py`
- `python -m pytest helixc\tests\test_typecheck.py -q` -> 128 passed
- `python -m pytest helixc\tests\test_ir.py -q` -> 61 passed
- `python -m pytest helixc\tests\test_ptx.py -q` -> 25 passed
- `python -m pytest helixc\tests\test_cli.py -q` -> 48 passed
- `python -m pytest helixc\tests\test_const_fold.py -q` -> 54 passed
- `python -m pytest helixc\tests\test_provenance.py -q` -> 13 passed
- `python -m pytest helixc\tests\test_codegen.py -k "c111 or c112 or c115 or c116 or c117 or c118 or u64 or usize or i64_to_f64_then_back or f64_cast_target or shift or array_assign" -q --tb=short` -> 37 passed, 667 deselected
- Bootstrap gate: `python -m pytest helixc\tests\test_codegen.py -k bootstrap -q --tb=line` -> 18 passed, 686 deselected in 324.50s.
