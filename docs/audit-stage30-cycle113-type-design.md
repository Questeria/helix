# Audit Stage 30 Cycle 113 - Type/design consistency

## Header

- **Date**: 2026-05-12
- **Worktree**: `C:\Projects\Kovostov-Native`
- **HEAD**: `74849c1`
- **Mode**: source/test read-only. The only file written by this audit is this document.
- **Audited uncommitted files**:
  - `helixc/backend/x86_64.py`
  - `helixc/ir/passes/const_fold.py`
  - `helixc/tests/test_codegen.py`
  - `helixc/tests/test_const_fold.py`
  - `helixc/tests/test_ir.py`
- **Scope**:
  1. unsigned `u64`/`usize -> f32/f64` conversion sequence;
  2. `CAST` arm ordering;
  3. `i64`/`isize -> f32`;
  4. signedness of `SHR`;
  5. const-fold/runtime parity, specifically the cycle112 unsigned const comparison blocker;
  6. `usize`/`isize` alias handling.
- **Blocker bar**: HIGH/CRITICAL confidence findings only.

## Verdict

**PASS** - no HIGH/CRITICAL type/design blockers found.

The cycle112 unsigned const comparison blocker is closed in the current uncommitted batch. The const folder now masks operands before ordered integer comparison folding when either operand is unsigned, matching the backend's unsigned `seta`/`setb`/`setae`/`setbe` dispatch. Focused optimized probes for the original high-bit `u64` case, plus `usize` and `u32` siblings, now return the runtime-expected result.

## Blockers

None.

## Cycle112 Blocker Closure

Prior blocker: folded unsigned high-bit values were stored as signed Python integers, then ordered comparisons such as `0x8000_0000_0000_0000_u64 > 0_u64` folded with signed Python ordering and diverged from runtime unsigned codegen.

Current fix in `helixc/ir/passes/const_fold.py`:

- `_UNSIGNED_INT_NAMES` includes `u8`, `u16`, `u32`, `u64`, and `usize`.
- `_int_bits_for_type` preserves `usize` as 64-bit through `_INT_BITS`.
- The comparison fold masks both operands to the maximum operand bit width before evaluating `<`, `<=`, `>`, or `>=` when either operand is unsigned.

Verification:

```text
cycle112_u64_opt_parity: got=42 expected=42 PASS
cycle112_usize_opt_parity: got=42 expected=42 PASS
u32_highbit_opt_parity: got=42 expected=42 PASS
```

This matches backend behavior, where integer comparison codegen chooses unsigned setters if either operand type is unsigned and uses the 64-bit path for `u64`/`usize`.

## Passing Checks

- Unsigned `u64`/`usize -> f32/f64` lowering uses the standard high-bit-safe sequence: fast signed conversion for non-negative `rax`; for high-bit values, `((x >> 1) | (x & 1))`, signed `cvtsi2sd/ss`, then `addsd/addss`.
- The unsigned float conversion arms run before the broad signed 64-bit float arms, so `u64`/`usize` cannot fall into signed-only `cvtsi2s*` handling.
- `i64`/`isize -> f32` now uses `REX.W cvtsi2ss xmm0, rax`; it is guarded by `_is_i64_type`, so it covers `isize` without stealing `u64`/`usize`.
- Runtime `SHR` dispatches by result signedness and width: unsigned `u*`/`usize` use logical `shr`; signed integer types use arithmetic `sar`; `u64`/`usize` stay on the 64-bit `rax/cl` path.
- Alias treatment is consistent across the changed surfaces: `usize` rides `_is_u64_type`, `_is_64bit_int_type`, and `_is_unsigned_int_type`; `isize` rides `_is_i64_type` and `_is_64bit_int_type`.

## Verification Performed

Focused pytest run:

```text
python -m pytest helixc/tests/test_ir.py::test_c111_cast_u64_to_f64_uses_unsigned_high_bit_path helixc/tests/test_ir.py::test_c111_cast_u64_to_f32_uses_unsigned_high_bit_path helixc/tests/test_ir.py::test_c112_cast_i64_to_f32_uses_64bit_signed_path helixc/tests/test_ir.py::test_c111_shr_u64_emits_logical_64bit_form helixc/tests/test_ir.py::test_c111_shr_usize_emits_logical_64bit_form helixc/tests/test_ir.py::test_c111_shr_u32_emits_logical_32bit_form helixc/tests/test_const_fold.py::test_c111_fold_u64_shr_is_logical helixc/tests/test_const_fold.py::test_c112_fold_u64_high_bit_compare_is_unsigned helixc/tests/test_codegen.py::test_c111_u64_to_f64_high_bit_runtime helixc/tests/test_codegen.py::test_c111_usize_to_f64_high_bit_runtime helixc/tests/test_codegen.py::test_c111_u64_to_f32_high_bit_runtime helixc/tests/test_codegen.py::test_c111_usize_to_f32_high_bit_runtime helixc/tests/test_codegen.py::test_c111_u64_shr_high_bit_runtime helixc/tests/test_codegen.py::test_c112_u64_shr_high_bit_compare_optimized_runtime
```

Result:

```text
14 passed
```

Additional manual probes:

```text
i64_to_f32_runtime: got=42 expected=42 PASS
isize_to_f32_runtime: got=42 expected=42 PASS
signed_i64_shr_arithmetic: got=42 expected=42 PASS
u64_low_odd_to_f32_fast_path: got=42 expected=42 PASS
u64_low_odd_to_f64_fast_path: got=42 expected=42 PASS
```

The low odd `u64` probes are important because a mistaken always-split/double lowering would convert `43` to `42`; the current fast path preserves `43`.

## Sub-threshold Observations

- **LOW, confidence 60**: `helixc/ir/tir.py` still comments that `SHR` is arithmetic right shift. The implemented design is now type-sensitive signed arithmetic vs unsigned logical, and the changed backend/const-fold paths follow that design. This is documentation drift, not a blocker for the current fix batch.
- **LOW, confidence 55**: There is no committed byte-pattern test specifically named for `isize -> f32`; the runtime probe above passes and `_is_i64_type` covers `isize`, so this is coverage asymmetry rather than a HIGH-confidence blocker.

## Final Verdict

**PASS.** The Stage 30 cycle112 unsigned const comparison blocker is closed, and no HIGH/CRITICAL type/design consistency blockers were found in the current uncommitted fix batch.
