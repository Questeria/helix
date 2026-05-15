# Audit Stage 30 Cycle 114 - Type/design consistency

## Header

- **Date**: 2026-05-12
- **Worktree**: `C:\Projects\Kovostov-Native`
- **HEAD**: `b117b9a65a42c741fb389590f528cff6cd8b04fc`
- **Mode**: source/test read-only. The only file written by this audit is `docs/audit-stage30-cycle114-type-design.md`.
- **Scope**:
  1. b117b9a type semantics;
  2. alias consistency for `usize`/`isize`;
  3. optimizer/runtime parity;
  4. `CAST` arm ordering;
  5. `SHR` signedness;
  6. fit of new abstractions to local backend/optimizer patterns.
- **Blocker bar**: HIGH/CRITICAL confidence findings only.

## Verdict

**FAIL** - two HIGH-confidence type/design blockers remain.

The b117b9a fixes close the previously audited unsigned `SHR`, unsigned const comparison, and `u64`/`usize -> f32/f64` conversion issues. However, the wider type surface still has reachable 64-bit unsigned division/modulo miscompiles and reachable float-to-64-bit-integer cast miscompiles.

## Blockers

### B1 - HIGH - `u64`/`usize` DIV/MOD still use the signed/32-bit backend path, diverging from optimized code

`helixc/backend/x86_64.py` routes 64-bit signed division/modulo through `_is_i64_type`, but that predicate intentionally excludes `u64`/`usize`:

- `DIV`: lines 1591-1600 use the 64-bit `cqo; idiv rcx` path only for `_is_i64_type`; all other integer results fall into `_emit_idiv_guarded`, the 32-bit signed path.
- `MOD`: lines 1606-1615 have the same `_is_i64_type` gate.
- There is no unsigned 64-bit `div rcx` helper in the backend; only signed `idiv_rcx` exists.

This is reachable source, not dead code. The optimizer can hide the bug for some constant cases, which creates an optimizer/runtime parity break:

```text
u64_div_high optimize=False: 0
u64_div_high optimize=True: 42
u64_mod_high optimize=False: 0
u64_mod_high optimize=True: 42
usize_div_high_unopt_parity optimize=False: 0
usize_div_high_unopt_parity optimize=True: 42
usize_mod_high_unopt_parity optimize=False: 0
usize_mod_high_unopt_parity optimize=True: 42
```

Probe shape:

```helix
fn main() -> i32 {
    let x: u64 = (1_u64 << 32_u64) + 84_u64;
    let y: u64 = x / 2_u64;
    if y > (1_u64 << 31_u64) { 42 } else { 0 }
}
```

Unsigned high-bit constants are also semantically wrong in the const folder because `DIV`/`MOD` folding uses signed Python integer arithmetic before `_wrap_int_to_type`:

```text
u64_div_highbit_by_2 optimize=False: 0
u64_div_highbit_by_2 optimize=True: 0
u64_mod_highbit_plus_42 optimize=False: 0
u64_mod_highbit_plus_42 optimize=True: 0
```

This is the same alias-consistency defect class the recent `_is_64bit_int_type` helpers were meant to remove, but `DIV`/`MOD` still sit on the old signed-only gate.

### B2 - HIGH - Float-to-`i64`/`u64`/`isize`/`usize` casts are allowed but lower through 32-bit `eax`

The typechecker allows numeric scalar casts in either direction (`helixc/frontend/typecheck.py` lines 2209-2211), including `f32/f64 -> i64/u64/isize/usize`.

The backend `CAST` arm computes `to_is_i64 = self._is_64bit_int_type(to_ty)` at line 1354, but the float-to-integer arms ignore it:

- `f64 -> integer`: lines 1448-1453 emit `cvttsd2si_eax_xmm0()` and store only `eax`.
- `f32 -> integer`: lines 1454-1458 emit `cvttss2si_eax_xmm0()` and store only `eax`.
- The assembler only exposes the 32-bit forms (`cvttss2si_eax_xmm0` at lines 652-654 and `cvttsd2si_eax_xmm0` at lines 722-724).

So any `f32/f64 -> 64-bit integer` cast above the signed 32-bit range is truncated/saturated through the wrong instruction width and writes only four bytes of an eight-byte destination slot. Optimized and unoptimized runs both miscompile because there is no CAST fold saving this path:

```text
f64_to_i64_over_32_compare optimize=False: 0
f64_to_i64_over_32_compare optimize=True: 0
f32_to_i64_over_32_compare optimize=False: 0
f32_to_i64_over_32_compare optimize=True: 0
f64_to_isize_over_32_compare optimize=False: 0
f64_to_isize_over_32_compare optimize=True: 0
f64_to_u64_over_32_compare optimize=False: 0
f64_to_u64_over_32_compare optimize=True: 0
f64_to_usize_over_32_compare optimize=False: 0
f64_to_usize_over_32_compare optimize=True: 0
f32_to_usize_over_32_compare optimize=False: 0
f32_to_usize_over_32_compare optimize=True: 0
```

Probe shape:

```helix
fn main() -> i32 {
    let x: u64 = 4294967338.0_f64 as u64;
    let y: u64 = 1_u64 << 32_u64;
    if x > y { 42 } else { 0 }
}
```

This is a `CAST` arm-ordering/design blocker: the broad `from_is_f64 and not to_is_float` / `from_is_float and not to_is_float` arms run before any 64-bit integer target-specific lowering exists.

## Passing Checks

- `u64`/`usize -> f32/f64` now routes through `_emit_u64_to_float` before the broad signed 64-bit float arms. The helper uses the standard high-bit-safe split/convert/double sequence and keeps `usize` aligned with `u64`.
- `i64`/`isize -> f32` uses the REX.W `cvtsi2ss xmm0, rax` form, and the guard is `_is_i64_type`, so `isize` is covered without catching `u64`/`usize`.
- Runtime `SHR` dispatches on signedness and width: unsigned `u*`/`usize` use logical `shr`; signed integer types use arithmetic `sar`.
- Const-fold `SHR` masks unsigned operands before shifting, while signed operands keep Python arithmetic right-shift semantics.
- Const-fold ordered integer comparisons mask operands when either side is unsigned, matching the backend's unsigned `seta`/`setb`/`setae`/`setbe` dispatch for the cases covered by b117b9a.
- Alias helpers are consistent on the changed surfaces: backend `_is_u64_type` includes `usize`, `_is_i64_type` includes `isize`, and const-fold `_INT_BITS` keeps both aliases at 64 bits.
- `_emit_u64_to_float` fits the local `FnCompiler` style: a private lowering helper adjacent to the other backend-specific emit helpers, using existing `Asm` byte-emission primitives.

## Verification Performed

Focused regression pytest:

```text
python -m pytest helixc/tests/test_ir.py::test_c111_cast_u64_to_f64_uses_unsigned_high_bit_path helixc/tests/test_ir.py::test_c111_cast_u64_to_f32_uses_unsigned_high_bit_path helixc/tests/test_ir.py::test_c112_cast_i64_to_f32_uses_64bit_signed_path helixc/tests/test_ir.py::test_c111_shr_u64_emits_logical_64bit_form helixc/tests/test_ir.py::test_c111_shr_usize_emits_logical_64bit_form helixc/tests/test_ir.py::test_c111_shr_u32_emits_logical_32bit_form helixc/tests/test_const_fold.py::test_c111_fold_u64_shr_is_logical helixc/tests/test_const_fold.py::test_c112_fold_u64_high_bit_compare_is_unsigned helixc/tests/test_codegen.py::test_c111_u64_to_f64_high_bit_runtime helixc/tests/test_codegen.py::test_c111_usize_to_f64_high_bit_runtime helixc/tests/test_codegen.py::test_c111_u64_to_f32_high_bit_runtime helixc/tests/test_codegen.py::test_c111_usize_to_f32_high_bit_runtime helixc/tests/test_codegen.py::test_c111_u64_shr_high_bit_runtime helixc/tests/test_codegen.py::test_c112_u64_shr_high_bit_compare_optimized_runtime -q
```

Result:

```text
14 passed
```

Additional manual probes were run through the normal parse -> flatten -> monomorphize -> grad -> lower -> optional optimize -> x86_64 ELF -> WSL execution path. The failing probe outputs are recorded in the blocker sections.

## Sub-threshold Observations

- **LOW, confidence 60**: `helixc/ir/tir.py` still documents `SHR` as arithmetic-only and says unsigned integer types do not exist. Runtime and const-fold behavior are now type-sensitive, so this is documentation drift rather than a blocker.
- **LOW, confidence 55**: unsigned type membership is duplicated between backend `_is_unsigned_int_type` and const-fold `_UNSIGNED_INT_NAMES`. The sets agree today, so this is a drift risk only.

## Final Verdict

**FAIL.** The b117b9a `SHR`, unsigned comparison, and unsigned-to-float fixes pass their focused checks, but the lane still has HIGH-confidence blockers in unsigned 64-bit `DIV`/`MOD` semantics and float-to-64-bit-integer casts.
