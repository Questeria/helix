# Stage 30 Cycle 116 Type-Design Audit

Date: 2026-05-12

## Scope

Cycle 116 checked whether TIR source and result types stay consistent across typecheck, constant folding, and x86-64 codegen for mixed scalar operations.

## Findings

PASS, p_pass 0.86, confidence HIGH.

No blocking type-contract findings remained after refresh.

Validated fixes:

- Mixed float scalar operations now require an explicit cast.
- Assignment and compound assignment now check value type against target type.
- Mixed integer operations remain allowed.
- Non-64 ADD/SUB/MUL and BIT_AND/BIT_OR/BIT_XOR now use source-width-aware operand reloads.
- Narrow SHL/SHR source-width behavior matches const-fold and runtime.
- Backend CLI now stops before codegen on type errors.

## Verification

- `python -m pytest helixc\tests\test_codegen.py -k "c116" -q --tb=short` -> 4 passed
- `python -m pytest helixc\tests\test_typecheck.py -k "d7_deep_ref_cast_bounded or c116" -q --tb=short` -> 4 passed
- `python -m pytest helixc\tests\test_const_fold.py -k "c116 or c115" -q` -> 11 passed
- `python -m pytest helixc\tests\test_typecheck.py -q` -> 117 passed
- `python -m pytest helixc\tests\test_const_fold.py -q` -> 54 passed
- `python -m pytest helixc\tests\test_ir.py -q` -> 58 passed
- `python -m pytest helixc\tests\test_codegen.py -k "c111 or c112 or c115 or c116 or u64 or usize or i64_to_f64_then_back or f64_cast_target or shift" -q --tb=short` -> 36 passed
- Bootstrap gate -> 18 passed, 686 deselected in 343.34s.

## Notes

The previous C116-TD-3 blocker was fixed by routing the affected 32-bit arithmetic and bitwise backend paths through `_load_cmp_operand_rax` and `_load_cmp_operand_rcx`, matching the constant folder's declared source-width model.
