# Stage 30 Cycle 117 Silent-Failures Audit

Date: 2026-05-12

## Scope

Cycle 117 audited unsupported type paths that could still continue into lowering or backend emission after diagnostics. The focus was fail-closed behavior for direct backend CLI use, PTX emission, indexed assignment, array literals, and unsupported tensor/tile indexing.

## Findings

PASS, p_pass 0.91, confidence HIGH.

No blocking silent-failure findings remained after refresh.

Resolved blockers:

- Direct x86-64 backend CLI now aborts on struct monomorphization diagnostics before codegen.
- Direct x86-64 backend CLI now runs trace, panic, unsafe, unwind, and autotune validation before lowering/codegen.
- `--emit-ptx` now lowers the real TileModule and detects kernels through Tile IR function attributes.
- Mixed array literals are rejected in typecheck and defensively rejected in direct lowering if typecheck is bypassed.
- Unsupported tensor/tile indexing and indexed assignment now fail loudly instead of lowering to missing values or no-ops.

Residual risk:

- Lexer-level diagnostics for malformed source remain rough in a few direct CLI paths, but they fail closed and do not emit binaries.

## Verification

- `python -m py_compile helixc\frontend\typecheck.py helixc\ir\lower_ast.py helixc\backend\x86_64.py helixc\check.py helixc\tests\test_typecheck.py helixc\tests\test_ir.py helixc\tests\test_cli.py`
- `python -m pytest helixc\tests\test_typecheck.py -q` -> 126 passed
- `python -m pytest helixc\tests\test_ir.py -q` -> 60 passed
- `python -m pytest helixc\tests\test_cli.py -q` -> 48 passed
- `python -m pytest helixc\tests\test_ptx.py -q` -> 23 passed
- `python -m pytest helixc\tests\test_const_fold.py -q` -> 54 passed
- `python -m pytest helixc\tests\test_provenance.py -q` -> 13 passed
- `python -m pytest helixc\tests\test_codegen.py -k "c111 or c112 or c115 or c116 or u64 or usize or i64_to_f64_then_back or f64_cast_target or shift" -q --tb=short` -> 36 passed, 668 deselected
- Bootstrap gate: `python -m pytest helixc\tests\test_codegen.py -k bootstrap -q --tb=line` -> 18 passed, 686 deselected in 320.60s.
