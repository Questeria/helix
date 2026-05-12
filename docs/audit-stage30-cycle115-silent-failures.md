# Stage 30 Cycle 115 Silent-Failure Audit

Date: 2026-05-12
Verdict: PASS
Confidence: HIGH

## Scope

Re-audited optimizer/runtime parity and silent miscompile risk around integer source-width handling after the Cycle 114 failures. Focus areas:

- u64/usize high-value arithmetic, bitwise, compare, div/mod, and casts
- mixed signed/unsigned source widening
- u8/u16/i8/i16 wrapped values in compares, casts, and div/mod
- signed 32-bit div/mod guard behavior
- bootstrap stale-output handling

## Findings Resolved

- Mixed signed/unsigned constant folding now widens each operand by its declared source type before applying unsigned compare/div/mod interpretation.
- 32-bit and 64-bit compare paths now reload operands through source-width aware helpers, avoiding stale high bits from narrow slots.
- Integer casts now respect declared source width before storing to the destination width.
- Narrow integer-to-float casts now source-widen before using the REX.W float conversion path.
- Signed 32-bit div/mod now source-widens operands before `idiv`.
- Signed 32-bit div/mod zero-guard jump now skips exactly the emitted zero path, so normal `i32::MIN / 2` and `% 3` reach the real `idiv` path.

## Verification

- `python -m pytest helixc\tests\test_ir.py -q` -> 58 passed
- `python -m pytest helixc\tests\test_const_fold.py -q` -> 52 passed
- `python -m pytest helixc\tests\test_codegen.py -k "c111 or c112 or c115 or u64 or usize or i64_to_f64_then_back or f64_cast_target or shift" -q` -> 32 passed, 668 deselected
- `python -m pytest helixc\tests\test_codegen.py -k bootstrap -q --tb=line` -> 18 passed, 682 deselected in 535.44s

## Stage 30 Status

Cycle 115 is clean. Because Cycle 114 failed, the strict Stage 30 consecutive-clean streak is now 1/5.
