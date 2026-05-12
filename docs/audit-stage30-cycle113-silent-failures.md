# Stage 30 Cycle 113 Silent-Failures Audit

Verdict: PASS

Date: 2026-05-12

Audited worktree: `C:\Projects\Kovostov-Native`

Scope:

- `u64`/`usize -> f32/f64` casts.
- `i64`/`isize -> f32` casts.
- Signed vs unsigned `SHR` runtime lowering.
- Unsigned `SHR` constant folding.
- Unsigned constant comparisons.
- Regression tests added for the Stage 30 fix batch.

Constraint followed: this audit is source/test read-only. The only file added by this audit is `docs/audit-stage30-cycle113-silent-failures.md`.

## Summary

No HIGH/CRITICAL confidence blockers found.

The cycle112 blockers are closed in the current uncommitted batch:

- Cycle112 silent-failures/type-design blocker: unsigned high-bit `SHR` folded into an unsigned ordered comparison now folds with unsigned semantics. The comparison folder masks operands when either operand type is unsigned before evaluating `<`, `<=`, `>`, or `>=` (`helixc/ir/passes/const_fold.py:570-577`). The former optimized/unoptimized divergence repro now returns `42` in both modes, and direct fold probes for both `u64` and `usize` produce `bool` constant `1`.
- Cycle112 code-review blocker: the newly added `i64 -> f32` production arm now has a direct regression test. `helixc/tests/test_ir.py:787-802` asserts `F3 48 0F 2A C0` and rejects the old non-REX `F3 0F 2A C0` low-32-bit path.

## Blockers

None.

## Verification Performed

Focused regression slice:

```text
python -m pytest helixc\tests\test_ir.py -k "c111 or c112" helixc\tests\test_const_fold.py -k "c111 or c112" helixc\tests\test_codegen.py -k "c111 or c112"
```

Result:

```text
14 passed, 753 deselected
```

Manual probes:

```text
PASS: cycle112 unsigned SHR+compare optimized: got=42 expected=42 optimize=True
PASS: cycle112 unsigned SHR+compare unoptimized: got=42 expected=42 optimize=False
PASS: i64 -> f32 runtime high value: got=42 expected=42 optimize=False
PASS: isize -> f32 runtime high value: got=42 expected=42 optimize=False
PASS: u64 logical SHR runtime: got=42 expected=42 optimize=False
PASS: i64 arithmetic SHR runtime: got=42 expected=42 optimize=False
PASS: u64 -> f64 high-bit runtime: got=42 expected=42 optimize=False
PASS: u64 -> f32 high-bit runtime: got=42 expected=42 optimize=False
FOLD: u64 high-bit SHR compare cmp_count=0 bool_consts=[1]
FOLD: usize high-bit SHR compare cmp_count=0 bool_consts=[1]
```

## Audited Findings

### Unsigned `u64`/`usize -> f32/f64`

PASS. The unsigned 64-bit cast arms run before the signed 64-bit float arms (`helixc/backend/x86_64.py:1382-1387`). The helper emits the standard high-bit-safe sequence: test sign bit, use direct signed `cvtsi2s*` for low-half values, and for high-bit values compute `((x >> 1) | (x & 1))`, convert with `cvtsi2sd`/`cvtsi2ss`, then double (`helixc/backend/x86_64.py:1121-1159`). Runtime probes and added codegen tests cover the old signed-negative and low-32-bit failure modes.

### Signed `i64`/`isize -> f32`

PASS. The backend now loads the full 64-bit source and emits `cvtsi2ss xmm0, rax` with `REX.W` for `_is_i64_type`, which includes both `i64` and `isize` (`helixc/backend/x86_64.py:1051-1057`, `1395-1401`). The cycle112 missing-test blocker is closed by `test_c112_cast_i64_to_f32_uses_64bit_signed_path`, and manual runtime probes passed for both `i64` and `isize` high values.

### Signed/Unsigned `SHR` Runtime

PASS. Runtime `SHR` lowering now selects width with `_is_64bit_int_type` and selects `shr` vs `sar` by unsignedness (`helixc/backend/x86_64.py:1688-1708`). The focused tests assert `u64`, `usize`, and `u32` use logical `shr`, while manual probes confirm `i64` still uses arithmetic semantics for negative values.

### Unsigned `SHR` Const Folding

PASS. The const folder masks unsigned `SHR` left operands to the result width before shifting (`helixc/ir/passes/const_fold.py:534-538`), preserving logical right-shift semantics for high-bit unsigned values. The existing `u64 >> 63` test catches the direct arithmetic-shift regression, and manual `u64`/`usize` high-bit-preserving probes also folded correctly.

### Unsigned Const Comparisons

PASS. The cycle112 blocker is closed. When either operand is unsigned, comparison folding now masks both operand values to the maximum operand bit width before evaluating ordered comparisons (`helixc/ir/passes/const_fold.py:570-584`). This mirrors the backend's unsigned setcc dispatch when either comparison operand is unsigned (`helixc/backend/x86_64.py:1864-1868`). The exact prior repro, `((1_u64 << 63_u64) >> 0_u64) > 0_u64`, now returns `42` with and without optimization.

## Sub-Threshold Observations

- The direct regression test for `i64 -> f32` is byte-pattern based, not runtime based. It is discriminative against the old low-32-bit opcode, so this is not a blocker.
- `isize -> f32` is verified by shared predicate review and a manual runtime probe, but there is no dedicated committed `isize` regression test. Below blocker bar because `_is_i64_type` explicitly includes `isize`, and the manual probe passed.
- The `u64`/`usize -> f32/f64` runtime tests use threshold checks rather than exact rounding assertions near `2^63`/`2^64`. The current implementation is the standard split/convert/double form, so this is coverage granularity rather than an observed defect.
