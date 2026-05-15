# Audit Stage 30 cycle 114 - Code review

- Date: 2026-05-12
- Worktree: `C:\Projects\Kovostov-Native`
- HEAD: `b117b9a65a42c741fb389590f528cff6cd8b04fc`
- Lane: code review
- Scope: Stage 30 cycle114 review of implementation edge cases, maintainability, regression test quality/discriminativity, and possible missed tests around `b117b9a`.
- Files reviewed:
  - `helixc/backend/x86_64.py`
  - `helixc/ir/passes/const_fold.py`
  - `helixc/tests/test_codegen.py`
  - `helixc/tests/test_const_fold.py`
  - `helixc/tests/test_ir.py`
- Constraint followed: source and test files were read-only. Existing untracked cycle111 audit docs were left untouched. This document is the only file added by this audit.

## Summary verdict

**PASS** - no HIGH/CRITICAL-confidence blockers found.

The `b117b9a` implementation appears correct for the Stage 30 fix surface: unsigned `u64`/`usize -> f32/f64` conversion uses the standard high-bit-safe split/sticky-bit/double lowering, signed `i64`/`isize -> f32` uses a REX.W `cvtsi2ss` path, unsigned runtime `SHR` now emits logical shifts by signedness/width, and unsigned constant comparison folding masks operands before ordered compares. The focused regression suite is discriminative against the direct prior failure modes.

## Verification performed

- Confirmed the worktree is at `b117b9a65a42c741fb389590f528cff6cd8b04fc`.
- Inspected `git diff b117b9a^ b117b9a --` for the changed source and regression tests.
- Reviewed implementation sites:
  - unsigned cast helper: `helixc/backend/x86_64.py:1112-1159`
  - CAST arm ordering: `helixc/backend/x86_64.py:1345-1487`
  - signedness-aware `SHR`: `helixc/backend/x86_64.py:1688-1709`
  - unsigned const-fold helpers and compare masking: `helixc/ir/passes/const_fold.py:113-123`, `helixc/ir/passes/const_fold.py:521-590`
- Ran focused regression tests with pytest cache and pyc writes suppressed:

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

Result: `15 passed in 15.27s`.

- Ran additional runtime probes for edges not committed as direct tests:

```text
u64_low_odd_to_f32: got=42 expected=42 PASS
u64_low_odd_to_f64: got=42 expected=42 PASS
i64_high_to_f32_runtime: got=42 expected=42 PASS
isize_high_to_f32_runtime: got=42 expected=42 PASS
signed_i64_shr_arithmetic: got=42 expected=42 PASS
u32_unsigned_shr_runtime: got=1 expected=1 PASS
```

## Blockers

None.

## Sub-threshold observations

- **LOW, confidence 60:** `helixc/ir/tir.py:153-155` still describes `SHR` as arithmetic-only and says unsigned integer types are unreachable. The backend and const-fold implementation now intentionally use signed arithmetic `SAR` for signed ints and logical `SHR` for unsigned ints. This is documentation drift, not a correctness blocker for `b117b9a`.
- **LOW, confidence 58:** The committed `u64`/`usize -> f32/f64` runtime tests exercise high-bit values, but not low odd values that must take the `jns` fast path. Manual probes for `43_u64 as f32/f64` passed. A future committed regression for a low odd value would make the fast-path guard more explicit.
- **LOW, confidence 55:** `test_c111_shr_u32_emits_logical_32bit_form` checks for `D3 E8`, which is also a suffix of the 64-bit `48 D3 E8` encoding. The implementation correctly routes `u32` through the 32-bit path, and the manual `u32` runtime probe passed; this is only a byte-pattern specificity weakness.
- **LOW, confidence 55:** There is no committed test named specifically for `isize -> f32`. The implementation is covered by `_is_i64_type`, and the runtime alias probe passed, so this is coverage asymmetry rather than a blocker.

## Positive observations

- The cycle112 code-review blocker is closed by `test_c112_cast_i64_to_f32_uses_64bit_signed_path`, which pins `F3 48 0F 2A C0` and rejects the old non-REX `cvtsi2ss xmm0, eax` path.
- The unsigned float conversion helper preserves exact low-path behavior and uses the conventional high-bit lowering: `test`, split/sticky, signed convert, then double.
- The unsigned `SHR` change is consistently applied in runtime codegen and const folding.
- The cycle112 unsigned high-bit const-compare issue is covered at both IR fold and optimized runtime levels.

## Final verdict

**PASS.** No HIGH/CRITICAL-confidence code-review blockers were found for Stage 30 cycle114 at `b117b9a`.
