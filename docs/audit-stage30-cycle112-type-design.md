# Audit Stage 30 Cycle 112 - Type/design consistency

## Header

- **Date**: 2026-05-12
- **Worktree**: `C:\Projects\Kovostov-Native`
- **Mode**: source/test read-only. The only file written by this audit is this document.
- **Audited uncommitted files**:
  - `helixc/backend/x86_64.py`
  - `helixc/ir/passes/const_fold.py`
  - `helixc/tests/test_codegen.py`
  - `helixc/tests/test_const_fold.py`
  - `helixc/tests/test_ir.py`
- **Scope**:
  1. unsigned `u64`/`usize` to `f32`/`f64` conversion sequence and CAST arm ordering;
  2. new signed `i64`/`isize` to `f32` arm;
  3. `SHR` runtime semantics by signedness;
  4. const-fold/runtime parity;
  5. alias handling for `usize`/`isize`.
- **Bar**: only HIGH/CRITICAL confidence findings are blockers. Sub-threshold observations are recorded separately.

## Verdict

**FAIL** - one HIGH-confidence blocker remains in const-fold/runtime parity for unsigned values.

The backend changes for unsigned 64-bit float conversion, `i64 -> f32`, runtime `SHR`, and `usize`/`isize` alias classification look type/design-consistent. The blocker is in the optimizer layer: after the new unsigned logical `SHR` fold produces the right bit pattern, constant comparison folding can still reinterpret high-bit unsigned constants as signed Python integers.

## Blockers

### C112-TD-F1 - unsigned const comparisons can still diverge after folded unsigned SHR

- **Severity**: HIGH
- **Confidence**: 92
- **File**: `helixc/ir/passes/const_fold.py`
- **Relevant lines**: unsigned `SHR` fold at 524-528; unconditional signed Python comparison at 557-567.

The Stage 30 batch correctly changes unsigned `SHR` folding to mask the left operand before shifting:

```python
if _res_ty.name in _UNSIGNED_INT_NAMES:
    v = (l & ((1 << _bits) - 1)) >> r
```

But the folded result is then passed through `_wrap_int_to_type`, which stores high-bit `u64`/`usize` results as negative Python integers preserving the two's-complement bit pattern. That representation is fine for backend emission, but the later constant-comparison folder compares those negative Python integers directly:

```python
tir.OpKind.CMP_GT: l > r
```

That is signed comparison even when the operands are typed `u64`/`usize`. Runtime codegen, by contrast, dispatches unsigned integer comparisons to `setb`/`setbe`/`seta`/`setae` when either operand is unsigned (`x86_64.py` 1864-1868). So optimized and unoptimized semantics diverge.

Minimal reproducer at the IR/optimizer level:

```helix
fn f() -> bool {
    ((1_u64 << 63_u64) >> 0_u64) > 0_u64
}
```

Observed fold probe:

```text
CONST_INT u64 value=-9223372036854775808
CONST_INT bool value=0
```

Expected runtime semantics: `0x8000000000000000_u64 > 0_u64` is true under unsigned comparison.

This is in scope because the new unsigned `SHR` fold is intended to restore const-fold/runtime parity, but the parity still fails when the folded unsigned result remains high-bit-set and participates in a constant comparison. The same representation hazard applies to `usize` because it is included in `_UNSIGNED_INT_NAMES`.

Suggested fix: teach the integer comparison fold to choose signed vs unsigned semantics from operand/result scalar types, mirroring backend `_is_unsigned_int_type`. For unsigned comparisons, compare `l & mask` and `r & mask` at the operand bit width before evaluating `<`, `<=`, `>`, `>=`. Equality can stay bit-pattern based, but using masked values uniformly is simpler and keeps `usize` at 64 bits.

## Passing checks

- Unsigned `u64`/`usize -> f64/f32` arm ordering is sound: the `_is_u64_type(from_ty)` arms at `x86_64.py` 1382-1387 run before the broad signed 64-bit float arms, preventing `u64`/`usize` from falling into signed `cvtsi2s*` paths.
- The high-bit unsigned float conversion sequence is the standard split/convert/double lowering: `test`, slow path `((x >> 1) | (x & 1))`, signed `cvtsi2sd/ss`, then `addsd/addss`. `and ecx, 1` zero-extends `rcx` before `or rax, rcx`, which is safe because only the sticky low bit is needed.
- The new `i64`/`isize -> f32` arm at `x86_64.py` 1395-1401 uses `cvtsi2ss xmm0, rax` with `REX.W` and is guarded by `_is_i64_type`, so it does not steal `u64`/`usize`.
- Runtime `SHR` now dispatches on result signedness: `u*`/`usize` use `shr`, signed types use `sar`, and 64-bit width is selected through `_is_64bit_int_type`.
- Alias handling is consistent in the changed backend predicates: `isize` rides `_is_i64_type`, `usize` rides `_is_u64_type` and `_is_unsigned_int_type`.

## Sub-threshold observations

- **MEDIUM, conf 70**: The new runtime tests for `u64/usize -> f32` only assert that the result is greater than `1_000_000_000.0_f32`. That catches the old low-32-bit fallback but does not strongly check high-end rounding or near-boundary behavior.
- **LOW, conf 60**: There is no direct runtime regression test for `isize -> f32`; the backend predicate makes the path look correct, but test coverage is asymmetric with the new `u64/usize` cases.
- **LOW, conf 55**: The const-fold test added for unsigned `SHR` covers the `>> 63` case that produces `1`. It does not cover `>> 0` or other high-bit-preserving results, which is why C112-TD-F1 remains exposed.
