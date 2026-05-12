# Audit Stage 30 cycle 112 - Code review

- Date: 2026-05-12
- HEAD: `74849c1`
- Scope: current uncommitted Stage 30 fix batch in:
  - `helixc/backend/x86_64.py`
  - `helixc/ir/passes/const_fold.py`
  - `helixc/tests/test_codegen.py`
  - `helixc/tests/test_const_fold.py`
  - `helixc/tests/test_ir.py`
- Constraint followed: source and tests were read-only; this audit document is the only file added.

## Summary verdict

**FAIL** - 1 HIGH-confidence blocker.

The implementation changes for unsigned `u64/usize -> f32/f64`, unsigned logical `SHR`, and the corrected u64 `NEG` regression test are mechanically sound under review and the targeted tests pass. However, the same production hunk also adds a new `i64/isize -> f32` codegen arm with no matching regression test. A plausible revert of that arm leaves the current new tests green while restoring the old low-32-bit miscompile for large `i64` values.

## Verification performed

- Inspected the full uncommitted diff.
- Cross-checked prior cycle-111 codereview context: the previous vacuous `u64 NEG via SUB` test is now correctly changed to unary `-a` and asserts `48 F7 D8`.
- Ran targeted tests:

```text
python -m pytest helixc/tests/test_ir.py -k "c111 or c110_neg_u64" helixc/tests/test_const_fold.py -k c111 helixc/tests/test_codegen.py -k c111
```

Result: `11 passed, 753 deselected`.

- Ran manual runtime probes for low odd `u64 -> f32/f64`, optimized unsigned SHR const-fold parity, and the newly added `i64 -> f32` behavior. All probes returned expected values on the current worktree.

## Blockers

### F1 - HIGH confidence 90 - New `i64 -> f32` production arm has no discriminative regression test

**Production site:** `helixc/backend/x86_64.py:1395-1401`

The diff adds:

```python
# i64 -> f32: cvtsi2ss with REX.W.
if self._is_i64_type(from_ty) and to_is_float:
    self.asm.mov_rax_mem_rbp(src_slot)
    self.asm.b.emit(0xF3, 0x48, 0x0F, 0x2A, 0xC0)
    self.asm.movss_mem_rbp_xmm0(res_slot)
    return
```

This is a real correctness fix. Before this arm, `i64 as f32` fell through to the generic `i32 -> f32` path, which loaded only `eax` and emitted non-REX `cvtsi2ss xmm0, eax`. A value such as `4_294_967_296_i64` would convert as `0.0_f32` instead of approximately `4.29e9_f32`.

The current tests do not pin this arm:

- `test_c111_cast_u64_to_f32_uses_unsigned_high_bit_path` checks the unsigned `u64` split/convert/double sequence, not signed `i64 -> f32`.
- `test_c111_u64_to_f32_high_bit_runtime` and `test_c111_usize_to_f32_high_bit_runtime` cover unsigned paths only.
- Existing `test_i64_to_f64_then_back` covers `i64 -> f64`, not `i64 -> f32`.
- Grep found no `i64_to_f32` or large-`i64 as f32` regression test in `helixc/tests`.

Mental revert:

1. Remove only the new `i64 -> f32` arm at `x86_64.py:1395-1401`.
2. Keep all other cycle-112 changes.
3. The new `c111` u64/usize float and SHR tests still pass because they route through `_emit_u64_to_float` or SHR.
4. Large signed 64-bit values cast to `f32` silently regress to the low-32-bit path.

A discriminative test would be either:

- Byte-pattern: compile `fn i64_to_f32(x: i64) -> f32 { x as f32 }` and assert `F3 48 0F 2A C0`.
- Runtime: compile with `optimize=False`:

```helix
fn main() -> i32 {
    let x: i64 = 4_294_967_296_i64;
    let f: f32 = x as f32;
    if f > 1_000_000_000.0_f32 { 42 } else { 7 }
}
```

Current worktree returns `42`; the pre-arm path would return `7`.

## Sub-threshold observations

- **u64/usize float conversion tests focus on high-bit values.** The implementation has a correct `jns fast_path` branch for low values; manual probes for `43_u64 as f32/f64` passed. The committed tests do not explicitly pin the low odd-value fast path, so a future simplification that always used the high-bit split/double sequence could misconvert low odd values while the current high-value tests stayed green. This is below blocker bar because it is not a revert of the current fix and the implementation itself is correct.

- **`test_c111_shr_u32_emits_logical_32bit_form` uses a two-byte positive pattern.** `b"\xD3\xE8"` would also be contained inside `48 D3 E8` if a future bug accidentally emitted 64-bit `shr rax, cl` for `u32`. The negative `sar` assertion still catches the direct arithmetic-shift revert, so this is a minor discriminativity weakness rather than a blocker.

- **Unsigned SHR const-fold coverage is narrow but relevant.** The new const-fold test exercises `u64`; the implementation also covers `u8/u16/u32/usize`. The `u64` case is sufficient to catch the direct signed-Python-`>>` revert in the added logic.

## Positive observations

- The previous cycle-111 blocker around `test_c110_neg_u64_via_sub_emits_64bit_form` is fixed. The test now uses `fn neg_u64(a: u64) -> u64 { -a }` and asserts `48 F7 D8`, so a revert of the u64 NEG wide path would fail.
- `_emit_u64_to_float` uses the standard high-bit-set unsigned conversion shape: fast signed conversion for non-negative `rax`, and split/sticky-bit/double for high-bit-set values.
- Runtime tests for `u64/usize -> f32/f64` are discriminative against both old signed-64 conversion and old low-32-bit fallback.
- Runtime and byte-pattern tests for unsigned `SHR` are discriminative against the direct old `SAR` behavior for `u64`.

## Final verdict

**FAIL.** Fix batch should not advance until the newly added `i64/isize -> f32` production arm has a direct, discriminative regression test.
