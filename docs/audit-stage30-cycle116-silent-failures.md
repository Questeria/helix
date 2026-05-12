# Stage 30 Cycle 116 Silent-Failures Audit

Date: 2026-05-12

## Scope

Cycle 116 audited source-width parity and fail-closed compiler paths after Cycle 115 reset the Stage 30 clean streak. The silent-failures focus was whether the compiler could still accept or emit incorrect behavior after typecheck errors or narrow integer source-width mismatches.

## Findings

PASS, p_pass 0.86, confidence HIGH.

No blocking silent-failure findings remained after refresh.

Resolved blockers:

- Backend CLI now fails closed on typecheck errors before lowering/codegen.
- Non-64 ADD/SUB/MUL and BIT_AND/BIT_OR/BIT_XOR now reload operands through source-width-aware helpers.
- SHL/SHR source-width parity remains covered.
- Mixed float/int scalar operators and assignment mismatches are rejected at typecheck.
- Deep TyRef cast compatibility now hits trap 28803 instead of Python RecursionError.

Residual risk:

- Internal test helpers still bypass typecheck by design. Supported CLI paths now fail closed on type errors.

## Verification

- `python -m py_compile helixc\backend\x86_64.py helixc\frontend\typecheck.py helixc\tests\test_const_fold.py helixc\tests\test_codegen.py helixc\tests\test_typecheck.py`
- `python -m pytest helixc\tests\test_codegen.py -k "c116" -q --tb=short` -> 4 passed
- `python -m pytest helixc\tests\test_typecheck.py -k "d7_deep_ref_cast_bounded or c116" -q --tb=short` -> 4 passed
- `python -m pytest helixc\tests\test_typecheck.py -q` -> 117 passed
- `python -m pytest helixc\tests\test_const_fold.py -q` -> 54 passed
- `python -m pytest helixc\tests\test_ir.py -q` -> 58 passed
- `python -m pytest helixc\tests\test_provenance.py -q` -> 13 passed
- `python -m pytest helixc\tests\test_codegen.py -k "c111 or c112 or c115 or c116 or u64 or usize or i64_to_f64_then_back or f64_cast_target or shift" -q --tb=short` -> 36 passed
- Backend CLI good-source smoke wrote a binary.
- Backend CLI bad assignment source returned rc=1 and produced no binary.
- Bootstrap gate: `python -m pytest helixc\tests\test_codegen.py -k bootstrap -q --tb=line` -> 18 passed, 686 deselected in 343.34s.
