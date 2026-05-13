# Stage 30 Cycle 117 Type-Design Audit

Date: 2026-05-12

## Scope

Cycle 117 checked whether source-level type contracts match the lowering paths Helix actually implements. The main focus was scalar operator domains, array element consistency, indexed assignment target types, and tensor/tile indexing contracts.

## Findings

PASS, p_pass 0.88, confidence HIGH.

No blocking type-contract findings remained after refresh.

Validated fixes:

- Non-scalar binary operands such as arrays and raw pointers are rejected instead of being treated as scalar-safe.
- Bool and char arithmetic/order operators are rejected while equality and boolean logic remain allowed.
- Float `%`, bitwise, and shift operators are rejected; same-width float arithmetic such as `/` remains allowed.
- Compound assignment now reuses operator-domain checks.
- Array literals must have element types compatible with the first element.
- Tensor indexing is rejected until matching lowering exists.
- Tile indexing is restricted to the supported `@kernel` HBM tile parameter form with exactly one index.
- TyDiff/TyLogic wrapped binary operators validate their inner scalar operator domain before returning wrapped results.

## Verification

- `python -m pytest helixc\tests\test_typecheck.py -k "c117 or diff_mixed_inner or diff_same_inner or c2_b_c6" -q --tb=short` -> 12 passed
- `python -m pytest helixc\tests\test_ir.py -k "c117" -q --tb=short` -> 2 passed
- `python -m pytest helixc\tests\test_typecheck.py -q` -> 126 passed
- `python -m pytest helixc\tests\test_ir.py -q` -> 60 passed
- `python -m pytest helixc\tests\test_cli.py -q` -> 48 passed
- `python -m pytest helixc\tests\test_ptx.py -q` -> 23 passed
- `python -m pytest helixc\tests\test_const_fold.py -q` -> 54 passed
- `python -m pytest helixc\tests\test_provenance.py -q` -> 13 passed
- `python -m pytest helixc\tests\test_codegen.py -k "c111 or c112 or c115 or c116 or u64 or usize or i64_to_f64_then_back or f64_cast_target or shift" -q --tb=short` -> 36 passed, 668 deselected
- Bootstrap gate -> 18 passed, 686 deselected in 320.60s.

## Notes

This cycle intentionally rejects tensor indexing for now instead of pretending it produces the element dtype. That keeps the frontend contract honest until a real Tensor IR indexing operation and backend path exist.
