# Audit Stage 30 cycle 113 - Code review

- Date: 2026-05-12
- HEAD: `74849c1`
- Scope: current uncommitted Stage 30 fix batch after cycle112 fixes:
  - `helixc/backend/x86_64.py`
  - `helixc/ir/passes/const_fold.py`
  - `helixc/tests/test_codegen.py`
  - `helixc/tests/test_const_fold.py`
  - `helixc/tests/test_ir.py`
- Constraint followed: source and tests were read-only; this audit document is the only file added by this audit.

## Summary verdict

**PASS** - no HIGH/CRITICAL-confidence blockers found.

The cycle112 blocker is closed: the batch now includes a direct `i64 -> f32` byte-pattern regression test, and that test fails when the new signed 64-bit `cvtsi2ss` arm is removed in a temp-copy mutation check. The unsigned `u64/usize -> f32/f64`, unsigned `SHR`, and unsigned const-compare fixes are implementation-correct under review and covered by tests that fail under direct plausible reverts.

## Verification performed

- Inspected the full uncommitted diff and the prior cycle112 codereview blocker.
- Reviewed implementation sites:
  - unsigned `u64/usize -> f32/f64` helper and cast ordering: `helixc/backend/x86_64.py:1112-1159`, `helixc/backend/x86_64.py:1380-1401`;
  - unsigned runtime `SHR`: `helixc/backend/x86_64.py:1688-1708`;
  - unsigned const-fold `SHR` and compare masking: `helixc/ir/passes/const_fold.py:521-538`, `helixc/ir/passes/const_fold.py:567-585`.
- Reviewed regression tests:
  - `helixc/tests/test_ir.py:740-802`, `helixc/tests/test_ir.py:865-907`;
  - `helixc/tests/test_codegen.py:973-1044`;
  - `helixc/tests/test_const_fold.py:135-156`.
- Ran targeted tests with pytest cache and pyc writes suppressed:

```text
python -m pytest -p no:cacheprovider \
  helixc/tests/test_ir.py::test_c111_cast_u64_to_f64_uses_unsigned_high_bit_path \
  helixc/tests/test_ir.py::test_c111_cast_u64_to_f32_uses_unsigned_high_bit_path \
  helixc/tests/test_ir.py::test_c112_cast_i64_to_f32_uses_64bit_signed_path \
  helixc/tests/test_ir.py::test_c111_shr_u64_emits_logical_64bit_form \
  helixc/tests/test_ir.py::test_c111_shr_usize_emits_logical_64bit_form \
  helixc/tests/test_ir.py::test_c111_shr_u32_emits_logical_32bit_form \
  helixc/tests/test_ir.py::test_c110_neg_u64_emits_64bit_form \
  helixc/tests/test_const_fold.py::test_c111_fold_u64_shr_is_logical \
  helixc/tests/test_const_fold.py::test_c112_fold_u64_high_bit_compare_is_unsigned \
  helixc/tests/test_codegen.py::test_c111_u64_to_f64_high_bit_runtime \
  helixc/tests/test_codegen.py::test_c111_usize_to_f64_high_bit_runtime \
  helixc/tests/test_codegen.py::test_c111_u64_to_f32_high_bit_runtime \
  helixc/tests/test_codegen.py::test_c111_usize_to_f32_high_bit_runtime \
  helixc/tests/test_codegen.py::test_c111_u64_shr_high_bit_runtime \
  helixc/tests/test_codegen.py::test_c112_u64_shr_high_bit_compare_optimized_runtime
```

Result: `15 passed in 10.29s`.

- Ran a direct runtime probe for the cycle112 blocker case:

```helix
fn main() -> i32 {
    let x: i64 = 4_294_967_296_i64;
    let f: f32 = x as f32;
    if f > 1_000_000_000.0_f32 { 42 } else { 7 }
}
```

Result: `42`.

- Ran temp-copy mutation checks for plausible direct reverts:

```text
revert_i64_f32_arm: exit=1
  FAILED helixc/tests/test_ir.py::test_c112_cast_i64_to_f32_uses_64bit_signed_path

revert_unsigned_cmp_mask: exit=1
  FAILED helixc/tests/test_codegen.py::test_c112_u64_shr_high_bit_compare_optimized_runtime

revert_unsigned_shr_to_sar: exit=1
  FAILED test_c111_shr_u64_emits_logical_64bit_form
  FAILED test_c111_shr_u32_emits_logical_32bit_form
  FAILED test_c111_u64_shr_high_bit_runtime

revert_u64_float_arms: exit=1
  FAILED test_c111_cast_u64_to_f64_uses_unsigned_high_bit_path
  FAILED test_c111_cast_u64_to_f32_uses_unsigned_high_bit_path
  FAILED test_c111_u64_to_f64_high_bit_runtime
  FAILED test_c111_u64_to_f32_high_bit_runtime
```

## Blockers

None.

## Notes below blocker bar

- The cycle112 `i64 -> f32` blocker is closed. `test_c112_cast_i64_to_f32_uses_64bit_signed_path` asserts `F3 48 0F 2A C0` and rejects the old non-REX `F3 0F 2A C0` form; removing the production arm makes this test fail.
- The unsigned const-compare fix is covered by the optimized runtime regression, but the narrower `test_c112_fold_u64_high_bit_compare_is_unsigned` is not a strong standalone discriminator. In the temp-copy removal of the unsigned compare mask, that unit test still passed because it only searched for a `CONST_INT` value `1`, which can be present for unrelated operands. This is not a blocker because the same revert fails the optimized runtime test in the new batch.
- The unsigned `u64/usize -> f32/f64` tests are discriminative against direct old signed/low-32-bit reverts. They still do not explicitly pin the low-value odd fast path, so a future rewrite that always used the high-bit split/double path could slip through despite miscompiling values such as `43_u64 as f32`. The current implementation has the correct `test rax, rax; jns fast_path` guard, so this is a coverage hardening note only.
- `test_c111_shr_u32_emits_logical_32bit_form` uses a two-byte positive opcode pattern that could also occur inside a 64-bit `shr rax, cl` sequence. The negative `sar` assertion and runtime `u64` test catch the direct prior regression; this is only a precision issue for that byte-pattern test.

## Final verdict

**PASS.** The current Stage 30 fix batch can advance from the code-review/test-quality gate.
