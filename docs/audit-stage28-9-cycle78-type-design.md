# Audit Stage 28.9 cycle 78 — Type design

**Scope.** HEAD `a792ee3` (cycle-77 fix-sweep: FFI_CALL int/float ABI split +
for-range increment dtype matches iterator type).

**Mode.** STRICT READ-ONLY. No edits performed. No source files modified by
this audit. Single Write to this doc only, as scoped.

**Charter (narrow).** Verify cycle-77 type-surface and rotate to fresh areas:

1. FFI_CALL fix predicate exclusivity / totality over the IR ScalarType space
   (`_is_float_type`, `_is_f64_type`, `_is_i64_type`, `_is_u64_type`).
2. for-range fix `"i32"` fallback safety when `start_v.ty` is not a
   `TIRScalar`.
3. Op result-type vs operand-type invariants for
   ADD/SUB/MUL/CMP_LT/CMP_GT/STORE_VAR (`helixc/ir/builder.py`,
   `helixc/ir/tir.py`).
4. Register-allocator ABI parity: does any other ABI path
   (LIBC_INTRINSIC, syscall, tail_call) have a similar int/float gap?

Prior C1–C77 + deferred-known not re-flagged.

**Criterion.** 0 findings at conf >= 75%.

## Result: PASS — 0 findings at conf >= 75%

## Item-by-item

### Item 1 — FFI_CALL predicate exclusivity & totality (PASS)

Read `_is_float_type` / `_is_f64_type` / `_is_i64_type` / `_is_u64_type` at
`helixc/backend/x86_64.py:999–1017`.

- `_is_float_type`: `TIRScalar` with name in `{f16, bf16, f32, f64}`
- `_is_f64_type`: `TIRScalar` with name == `f64`
- `_is_i64_type`: `TIRScalar` with name in `{i64, isize}`
- `_is_u64_type`: `TIRScalar` with name in `{u64, usize}`

**Exclusivity.** The four name-sets are disjoint. `_is_float_type ∩
_is_i64_type ∩ _is_u64_type = ∅`. `_is_f64_type ⊂ _is_float_type` is the only
overlap and it is intentional (f64 is the wide-float subcase). FFI_CALL's
dispatch chain at `x86_64.py:1758–1776` checks `_is_float_type` first, then
falls to the int-class branch — exclusivity is preserved at the routing
predicate.

**Totality over the reachable ScalarType space.** Stage-0 ScalarType names
that actually reach the backend (`bool`, `i8`, `i16`, `i32`, `i64`, `isize`,
`u8`, `u16`, `u32`, `u64`, `usize`, `f16`, `bf16`, `f32`, `f64`) all map into
exactly one branch of the FFI_CALL dispatch. The f16 / bf16 case looks like
a potential gap (f16 would route to the `movss`/`movsd_load_xmmN` 4-byte
path, reading 4 bytes from a 2-byte slot) — but `_check_float_supported`
(line 1019) runs during slot allocation at lines 927/931/935 against every
SSA value's type before any op is emitted, raising `NotImplementedError`
for f16/bf16. So any FFI_CALL referencing an f16/bf16 SSA value is
rejected before the FFI_CALL emit path is reached. The "narrow" classifier
is sound.

No defect at the 75% bar.

### Item 2 — for-range "i32" fallback safety (PASS)

Read `helixc/ir/lower_ast.py:1842–1856`.

```
inc_dtype = start_v.ty.name if isinstance(start_v.ty, tir.TIRScalar) else "i32"
one = self.builder.const_int(1, dtype=inc_dtype)
new_i = self.builder.emit(tir.OpKind.ADD, cur_i, one, result_ty=start_v.ty)
```

The fallback is exercised only when `start_v.ty` is not a `TIRScalar`. The
parser/typecheck path that feeds `expr.iter_expr.start` is the same
`_lower_expr` chain that produces every other IR value: for any non-scalar
result (a `TIRTensor`, `TIRTile`, etc.), the surrounding `ALLOC_VAR /
STORE_VAR / CMP_LT / ADD` chain emitted at lines 1798–1854 would already
fail in the backend before the increment matters (`STORE_VAR` to a tensor
type, `CMP_LT` of two tensors, etc. — none of these are wired). The
"i32" fallback only sets the dtype of the constant `1`; the ADD still
uses `result_ty=start_v.ty`. So a hypothetical non-scalar reach would
produce ADD(?, i32, result_ty=non-scalar) which the x86_64 backend cannot
emit — the failure is loud at codegen, not silent.

For `TIRScalar` with a non-int name (e.g. `f32`) the fallback is not taken
and `const_int(1, dtype="f32")` would silently emit a CONST_INT with a
float result type — a real type-confusion mode. Reachability requires a
float-typed range expression (`for i in 0.5_f32 .. 1.5_f32 { ... }`). I
searched `helixc/examples/`, `helixc/stdlib/`, `helixc/bootstrap/`, and
`stage0/` for any `for ... in ... .. ...` with a float range — none exist;
all observed ranges are integer literals or i32/i64 variables. The
typecheck `A.Range` arm at `typecheck.py:1645–1650` returns
`TyUnknown(hint="range")` and the `A.For` arm at lines 1627–1637 binds the
loop variable to `iter_ty` (`TyUnknown`), so float ranges typecheck but
emit denormal increments. This is a pre-existing structural issue
(float-range support is not in Phase-0 scope); the cycle-77 fix did not
introduce it and the for-range float-range path is unreachable from any
real Helix program. Not flagged.

No defect at the 75% bar.

### Item 3 — Op result-type vs operand-type invariants (PASS, with note)

Read `helixc/ir/tir.py:432–445` and the `emit()` constructor at lines
416–430. The `Builder.emit` API does not enforce any relationship between
the operand types and the `result_ty` argument — the caller is responsible
for picking a coherent `result_ty`. Cycle-77's F1 finding exposed exactly
this gap: `ADD(i64, i32, result_ty=i64)` was silently accepted. The fix
plugged the for-range instance at the lowering layer (the constant `1` now
carries the iterator's dtype). No structural invariant check was added at
the `emit()` / `builder` layer.

This is a pre-existing structural property of the TIR builder. The same
permissiveness allows other lowering bugs of the same class (e.g.
`SUB(i64, i32, result_ty=i64)`, `MUL(i64, i32, result_ty=i64)`,
`STORE_VAR(name, i32-val) where name is allocated as i64`). Scanning
`helixc/ir/lower_ast.py` for `tir.OpKind.{ADD,SUB,MUL}` call sites:

- Hash-arena ops (lines 1378–1465 region) pin `result_ty=tir.TIRScalar("i32")`
  with all operands explicitly constructed at i32 via `const_int(N)` (default
  dtype i32). Operand-result coherence holds.
- Array-index arithmetic (`LOAD_ELEM`/`STORE_ELEM` lowering) uses
  `const_int(N)` for i32 idx values; operand-result coherence holds.
- Algebraic operators lowered from `A.BinOp` flow operand types from
  `_lower_expr` and the typechecker pre-equalizes operand widths, so post-
  typecheck the operands match.
- The for-range increment was the documented exception, now fixed.

I did not find a second concrete `ADD(iN, iM, result_ty=iK)` shape at
lowering time at conf >= 75%. The structural "no invariant in `emit()`"
gap is pre-existing and informational; it is not a new defect.

No defect at the 75% bar.

### Item 4 — Register-allocator ABI parity (PASS)

Grepped `helixc/backend/x86_64.py` for `INT_REGS`, `xmm_idx`,
`LIBC_INTRINSIC`, `SYSCALL`, `TAIL_CALL`, `tail_call`. Only three sites
exercise the SysV class-split:

| Site | Lines | Status |
|------|-------|--------|
| fn prologue (param spill from arg-regs to stack) | 950–990 | int/float split present |
| OpKind.CALL arg pass + return | 1666–1719 | int/float split present (already correct pre-cycle-77) |
| OpKind.FFI_CALL arg pass + return | 1724–1787 | int/float arg split fixed in cycle-77 |

`OpKind` in `helixc/ir/tir.py:262` defines `FFI_CALL` and `CALL` as the only
call-class ops. No `LIBC_INTRINSIC`, `SYSCALL`, `TAIL_CALL`, or
`INDIRECT_CALL` kinds exist in the IR. The exit syscall path at line 2628
uses a fixed instruction sequence (mov eax,60; syscall) with no operand
shuffle, so it is not in the ABI-class-split family.

`xmm_idx` and `int_idx` counters in the FFI_CALL fix (lines 1754–1776) are
each initialized to 0 and only incremented inside their respective
branches — verified by direct read. The 8-xmm / 6-int register count
matches the SysV ABI. The newly added `_movss_load_xmmN` /
`_movsd_load_xmmN` helpers are the same primitives used by the regular
CALL arm at lines 1694–1696; cycle-77 simply mirrored the existing
infrastructure.

No defect at the 75% bar.

## Sibling-class checks examined but not flagged (informational)

- **FFI_CALL float return** is handled in
  `docs/audit-stage28-9-cycle78-silent-failures.md` (C78-1). Out of
  type-design scope and explicitly handled by the silent-failures sibling
  audit — not re-flagged here.
- **Regular CALL arm u64 routing** at `x86_64.py:1703` checks only
  `_is_i64_type(arg.ty)` (not `_is_u64_type`). The cycle-78
  silent-failures audit (line 59) explicitly reviewed this and concluded
  the asymmetry is correct given the matching `_is_i64_type`-only check
  in the callee prologue (line 986) — the caller/callee round-trip is
  consistent. Cycle-19 enumerated line 1703 in its routing-predicate
  table without flagging. Pre-existing-examined; not re-flagged per
  scope rules.
- **`||` operator lowering at `lower_ast.py:1135-1138`** emits
  `ADD(l, r, result_ty=bool)` — a result-type that doesn't match the
  ADD-by-result-type backend dispatch and could be a sibling of F1.
  Pre-existing; explicitly listed as deferred in
  `docs/audit-stage28-9-cycle78-silent-failures.md` line 66. Not
  re-flagged.
- **`tile_ir.py:220` "treat as opaque for v0.1"** TODO. Same deferred set.

## No code edits performed.

Read-only audit. No source files modified. Single Write to this doc only,
as scoped.
