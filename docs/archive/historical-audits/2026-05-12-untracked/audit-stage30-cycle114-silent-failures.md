# Stage 30 Cycle 114 Silent-Failures Audit

Verdict: FAIL

Date: 2026-05-12

Audited worktree: `C:\Projects\Kovostov-Native`

Audited commit: `b117b9a65a42c741fb389590f528cff6cd8b04fc` (`Stage 30 cycle 113 clean: fix unsigned casts and shifts`)

Constraint followed: source/test read-only. The only file added by this audit is `docs/audit-stage30-cycle114-silent-failures.md`.

## Scope

- Unsigned casts, especially `u64`/`usize -> f32/f64`.
- Signed casts to `f32`, especially `i64`/`isize -> f32`.
- Runtime `SHR` semantics for signed vs unsigned integer types.
- Constant-folded `SHR` and ordered comparisons.
- Regression tests added by the committed cycle113 fix batch.
- Silent fallback or silent miscompile windows introduced by `b117b9a`.

## Summary

One HIGH-confidence blocker remains in the committed cycle113 batch.

The direct cycle112 same-width `u64` high-bit comparison blocker is fixed, and the runtime codegen paths for unsigned 64-bit float casts, signed 64-bit-to-`f32` casts, and signed-vs-unsigned `SHR` dispatch look sound under focused review and probes.

However, the new unsigned constant-comparison fold masks both operands with the maximum operand width. That is incorrect for folded narrow unsigned values stored in the pass's signed spelling. A folded `u32` value like `0x80000000` is stored as `-2147483648`; comparing it against a `u64` causes the new code to apply a 64-bit mask to that negative Python int, turning it into `0xFFFFFFFF80000000` instead of zero-extending the original `u32` bit pattern to `0x0000000080000000`. Optimized and unoptimized programs silently diverge.

## Blockers

### HIGH SF-C114-1: mixed-width unsigned const comparisons sign-extend folded narrow unsigned values

Confidence: HIGH 95

Affected code:

- `helixc/ir/passes/const_fold.py:551` stores folded integer results with `_wrap_int_to_type(v, res.ty)`, so high-bit `u32` folded values are represented as negative Python ints.
- `helixc/ir/passes/const_fold.py:570-577` then checks whether either comparison operand is unsigned, computes `bits = max(lhs_bits, rhs_bits)`, and masks both operands with that same max-width mask before ordered comparison.

This loses the source operand width. A folded `u32` high-bit value must be zero-extended from 32 bits when compared against `u64`; the new max-width mask instead treats the negative Python spelling as already 64-bit.

Reproducer:

```helix
fn main() -> i32 {
    let x: u32 = (1_u32 << 31_u32) >> 0_u32;
    if x > 3_000_000_000_u64 { 42 } else { 7 }
}
```

Observed:

```text
optimize=False -> 7
optimize=True  -> 42
```

Expected: both paths return `7`, because `x` is `2_147_483_648_u32`, which is less than `3_000_000_000_u64`.

Second reproducer:

```helix
fn main() -> i32 {
    let x: u32 = 0_u32 - 1_u32;
    if x > 5_000_000_000_u64 { 42 } else { 7 }
}
```

Observed:

```text
optimize=False -> 7
optimize=True  -> 42
```

The typechecker reports zero errors for this program, so the silent failure is reachable through the normal non-strict compile path. The committed tests miss it because `test_c112_fold_u64_high_bit_compare_is_unsigned` covers same-width `u64` only; same-width `u64` is the case the new max-width mask handles correctly.

Suggested fix direction:

- When folding unsigned ordered comparisons, first decode each operand according to its own declared integer width.
- Then compare the decoded unsigned magnitudes, zero-extended to the comparison width as needed.
- Alternatively, refuse to fold mixed-width unsigned ordered comparisons until the IR has explicit cast/promotion nodes.
- Add optimized/unoptimized end-to-end regressions for `u32` high-bit folded values compared against `u64` thresholds.

## Audited Non-Blocker Surfaces

### Unsigned `u64`/`usize -> f32/f64`

PASS. The cast dispatch now routes `_is_u64_type(from_ty)` to `_emit_u64_to_float` before the signed 64-bit float arms (`helixc/backend/x86_64.py:1382-1387`). The helper emits the standard high-bit-safe sequence: branch on sign bit, use direct REX.W `cvtsi2s*` for low-half values, and for high-bit values convert `((x >> 1) | (x & 1))` then double (`helixc/backend/x86_64.py:1112-1159`). Focused tests and manual runtime probes passed for high-bit and low-path values.

### Signed `i64`/`isize -> f32`

PASS. The backend now emits REX.W `cvtsi2ss xmm0, rax` for `_is_i64_type(from_ty)`, which includes `isize` (`helixc/backend/x86_64.py:1051-1057`, `1395-1401`). The committed byte-pattern regression and manual runtime probes passed for positive and negative values beyond 32-bit range.

### Runtime `SHR`

PASS for the audited paths. Runtime `SHR` lowering now selects 64-bit vs 32-bit by `_is_64bit_int_type`, then selects logical `shr` for unsigned types and arithmetic `sar` for signed types (`helixc/backend/x86_64.py:1688-1708`). Manual probes passed for `u64`, `usize`, `u32`, and signed `i64` arithmetic shift behavior.

### Unsigned `SHR` Const Folding

PASS for same-width operands. The const folder masks unsigned `SHR` left operands to the result width before shifting (`helixc/ir/passes/const_fold.py:534-538`). Same-width `u64`, `usize`, and `u32` high-bit direct fold probes produced the expected constants. The blocker above is in the subsequent mixed-width comparison fold, not the unsigned `SHR` math itself.

### Tests

The committed cycle111/cycle112 focused regression slice passes, and full `test_const_fold.py` passes. Coverage gap: no test compares a folded high-bit `u32`/narrow unsigned value against a wider unsigned operand under optimization.

### Silent Fallbacks Introduced by `b117b9a`

No new source-level `return None` fallback was found in the committed source diff. The new x86 helper raises if its short-branch displacement assumptions are violated. The blocker is a silent wrong fold, not a newly introduced explicit fallback.

## Verification Performed

Focused committed regression slice:

```text
python -m pytest helixc\tests\test_ir.py -k "c111 or c112" helixc\tests\test_const_fold.py -k "c111 or c112" helixc\tests\test_codegen.py -k "c111 or c112"
```

Result:

```text
14 passed, 753 deselected
```

Additional focused slices:

```text
python -m pytest helixc\tests\test_const_fold.py
python -m pytest helixc\tests\test_ir.py -k "c111 or c112" helixc\tests\test_codegen.py -k "c111 or c112"
```

Results:

```text
43 passed
12 passed, 712 deselected
```

Manual pass probes:

```text
c112 unsigned SHR compare optimized: got=42 expected=42 opt=True PASS
c112 unsigned SHR compare unoptimized: got=42 expected=42 opt=False PASS
usize SHR compare optimized: got=42 expected=42 opt=True PASS
usize SHR compare unoptimized: got=42 expected=42 opt=False PASS
u32 logical SHR runtime: got=42 expected=42 opt=False PASS
i64 arithmetic SHR runtime: got=42 expected=42 opt=False PASS
u64 to f64 high bit exact pow2: got=42 expected=42 opt=False PASS
u64 to f32 high bit threshold: got=42 expected=42 opt=False PASS
u64 to f64 low path: got=42 expected=42 opt=False PASS
i64 to f32 high positive: got=42 expected=42 opt=False PASS
isize to f32 high positive: got=42 expected=42 opt=False PASS
i64 to f32 high negative: got=42 expected=42 opt=False PASS
```

Manual blocker probes:

```text
u32_shl_high_gt_5b_u64
 opt False 7
 opt True 42
u32_shl_high_gt_3b_u64
 opt False 7
 opt True 42
u32_minus_one_gt_5b_u64 opt False => 7
u32_minus_one_gt_5b_u64 opt True  => 42
typecheck errors for u32_minus_one_gt_5b_u64: 0
```

## Sub-Threshold Observations

- A `usize` high-bit comparison expressed through an explicit `u64 as usize` cast did not fully fold in the manual IR probe (`CMP_GT` remained), but optimized runtime still produced the correct result. Direct `usize` literal same-width forms do fold correctly. This is coverage/optimization granularity, not a blocker.
- The committed cast tests are mostly byte-sequence discriminators, not exact floating-rounding tests near every `u64` boundary. Manual high-bit and low-path probes passed, so this remains below blocker bar.
- The focused tests do not cover mixed-width unsigned comparison folding. This is the concrete gap behind SF-C114-1.
