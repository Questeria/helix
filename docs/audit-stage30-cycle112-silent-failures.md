# Stage 30 Cycle 112 Silent-Failures Audit

Verdict: FAIL

Audited uncommitted scope:

- `helixc/backend/x86_64.py`: `u64`/`usize` to `f32`/`f64` cast changes; signed vs unsigned `SHR` codegen.
- `helixc/ir/passes/const_fold.py`: unsigned `SHR` constant fold.
- `helixc/tests/test_ir.py`, `helixc/tests/test_codegen.py`, `helixc/tests/test_const_fold.py`: Stage 30 regression coverage.

## Blockers

### HIGH SF-C112-1: unsigned high-bit SHR can still fold into a signed ordered comparison

Confidence: HIGH 96

`const_fold.py` now gives unsigned `SHR` the correct logical shift math:

- `helixc/ir/passes/const_fold.py:524-526` masks the left operand to the result width before shifting when the result type is in `_UNSIGNED_INT_NAMES`.

The pass then immediately normalizes the folded integer through the existing signed storage convention:

- `helixc/ir/passes/const_fold.py:541` calls `_wrap_int_to_type(v, res.ty)`.
- For `u64`/`usize` values with bit 63 set, `_wrap_int_to_type` returns the signed Python spelling of the same bit pattern, for example `0x8000_0000_0000_0000` becomes `-9223372036854775808`.

That representation is bit-correct for backend emission, but it is not safe for later ordered constant comparisons in the same fold pass:

- `helixc/ir/passes/const_fold.py:557-566` compares `CONST_INT.attrs["value"]` with Python signed ordering for all integer types.
- The comparison fold does not inspect operand type signedness and does not mask unsigned operands before `<`, `<=`, `>`, or `>=`.

Reproducer:

```helix
fn main() -> i32 {
    let x: u64 = (1_u64 << 63_u64) >> 0_u64;
    if x > 0_u64 { 42 } else { 7 }
}
```

Observed in the current working tree:

```text
optimize=False -> 42
optimize=True  -> 7
```

Why this is a silent failure:

- Runtime codegen uses the unsigned path for `u64` comparison, so the unoptimized program correctly treats `0x8000_0000_0000_0000_u64 > 0_u64` as true.
- The optimized path folds the high-bit `u64` value to a negative Python int, then folds `x > 0_u64` using signed Python ordering, producing false.
- There is no diagnostic; default optimized compilation silently changes behavior.

Why the Stage 30 tests miss it:

- `test_c111_fold_u64_shr_is_logical` uses `>> 63`, whose result is `1`, so it never exercises a folded unsigned SHR result that keeps bit 63 set.
- The new runtime SHR/cast tests in `test_codegen.py` use `optimize=False`, so they intentionally bypass this optimizer interaction.

Suggested fix direction:

- Make ordered integer comparison folding unsigned-aware by checking operand/result scalar types and comparing masked values for `u8`/`u16`/`u32`/`u64`/`usize`.
- If type/signedness cannot be proven for both operands, do not fold ordered unsigned comparisons.
- Add an optimized end-to-end regression for the reproducer above.

## Sub-threshold observations

- The new `u64`/`usize -> f32/f64` x86 sequence in `x86_64.py` appears semantically sound: it branches on the sign bit, uses signed `cvtsi2s*` directly for low-half values, and uses the standard `((x >> 1) | (x & 1))` then double sequence for high-bit values.
- The SHR backend dispatch appears correctly split by unsignedness and width: unsigned `u32` uses `shr eax, cl`; unsigned `u64`/`usize` uses `shr rax, cl`; signed paths retain `sar`.
- The new byte-sequence tests are useful but not complete discriminators. They search the whole ELF for opcode subsequences and do not assert the `test rax, rax; jns fast_path` guard for low `u64` float casts. This is a coverage weakness, not a blocker, because the current emitted code contains the guard and runtime high-bit tests pass.
- The runtime cast tests use coarse threshold assertions. They catch the prior low-32/signed-negative failures, but they do not prove exact rounding near `2^63`, `2^64 - 1`, or low-path values like `1_u64`.

## Verification

Focused Stage 30 non-runtime IR/const-fold tests:

```text
python -m pytest helixc\tests\test_ir.py::test_c111_cast_u64_to_f64_uses_unsigned_high_bit_path helixc\tests\test_ir.py::test_c111_cast_u64_to_f32_uses_unsigned_high_bit_path helixc\tests\test_ir.py::test_c111_shr_u64_emits_logical_64bit_form helixc\tests\test_ir.py::test_c111_shr_usize_emits_logical_64bit_form helixc\tests\test_ir.py::test_c111_shr_u32_emits_logical_32bit_form helixc\tests\test_const_fold.py::test_c111_fold_u64_shr_is_logical
6 passed
```

Focused Stage 30 runtime codegen tests:

```text
python -m pytest helixc\tests\test_codegen.py::test_c111_u64_to_f64_high_bit_runtime helixc\tests\test_codegen.py::test_c111_usize_to_f64_high_bit_runtime helixc\tests\test_codegen.py::test_c111_u64_to_f32_high_bit_runtime helixc\tests\test_codegen.py::test_c111_usize_to_f32_high_bit_runtime helixc\tests\test_codegen.py::test_c111_u64_shr_high_bit_runtime
5 passed
```

Blocker repro:

```text
opt_false 42
opt_true 7
```
