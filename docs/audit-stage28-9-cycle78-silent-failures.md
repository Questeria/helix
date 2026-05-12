# Audit Stage 28.9 cycle 78 — Silent failures

**Scope.** Read-only HEAD `a792ee3`. Narrow conservative. Prior C1–C77 + deferred-known not re-flagged.

**Criterion.** 0 findings at conf >=75%.

## Result: 1 finding at >=75% — FAIL

---

## C78-1 (HIGH, conf 85): FFI_CALL float-return is read from eax, not xmm0 — partial cycle-77 fix

**Location.** `helixc/backend/x86_64.py` lines 1779–1786 (FFI_CALL result write-back).

**Class.** Same asymmetric-sibling class as C76-1. Cycle 77 fixed the FFI_CALL **arg** side (split int/float by class, send float args via xmm0..xmm7) but did NOT mirror the same split on the **return** side. The cycle-76 audit doc explicitly recommended this in the same finding (`docs/audit-stage28-9-cycle76-silent-failures.md` line 47: "And mirror the float-return path... using `self.asm.movss_mem_rbp_xmm0` / `_movsd_mem_rbp_xmmN`"). The cycle-77 fix-sweep commit message even acknowledged this with "Likely fine — float return goes to xmm0 which the existing return code reads" — but the existing return code at line 1786 does NOT read from xmm0; it reads from eax in the else-branch.

**Current return-handler shape (1779–1786):**
```
if op.results:
    res_slot = self._slot_of(op.results[0])
    # libc fns return int (eax) or pointer (rax). Stage 16.5
    # only wires int returns; other shapes deferred.
    if self._is_i64_type(op.results[0].ty) or self._is_u64_type(op.results[0].ty):
        self.asm.mov_mem_rbp_rax(res_slot)
    else:
        self.asm.mov_mem_rbp_eax(res_slot)
```

There is no `_is_f64_type` / `_is_float_type` branch. Contrast the regular CALL arm at lines 1709–1719 which DOES handle float returns via `movsd_mem_rbp_xmm0` / `movss_mem_rbp_xmm0`.

**Reachability probe.** Identical to C76-1:
```
extern "C" fn sinf(x: f32) -> f32;
fn main() -> i32 { let y: f32 = sinf(1.0_f32); 0 }
```
With cycle-77 in place, the f32 arg is now correctly placed in xmm0 (fixed). `sinf` writes its f32 result to xmm0 per SysV ABI. Then the codegen at line 1786 issues `mov_mem_rbp_eax` — it reads the integer return register (which `sinf` never touched), getting whatever int garbage was last in eax. The f32 slot `res_slot` gets that garbage bit pattern interpreted as an f32. The compile is clean, the typecheck is clean, the test passes if no one asserts on the value.

**Coverage of the existing gate.** Heavy gate post-fix passed 1508 tests (per commit message). This is consistent with C78-1 being silent — no existing test exercises a float-returning FFI extern. The cycle-76 audit doc explicitly noted this gap: "No diagnostic, no trap-id, no test currently exercises a float-arg extern so the gap is invisible." That observation applies symmetrically to float-return externs.

**Fix sketch (NOT applied — read-only).** Mirror the CALL arm's return-handling. Add before the i64/u64 check:
```
if self._is_f64_type(op.results[0].ty):
    self.asm.movsd_mem_rbp_xmm0(res_slot)
elif self._is_float_type(op.results[0].ty):
    self.asm.movss_mem_rbp_xmm0(res_slot)
elif self._is_i64_type(op.results[0].ty) or self._is_u64_type(op.results[0].ty):
    ...
```
Or, if Phase-0 wants to defer float returns explicitly, raise `NotImplementedError` for float result types so the miscompile becomes a loud compile error.

**Confidence rationale.** 85% (>=75 threshold). The defect is plainly visible in the code, the cycle-76 audit doc named it specifically as part of the same finding, the commit message's reassurance is factually incorrect (the "existing return code" reads eax not xmm0), and the SysV ABI return-class convention is unambiguous. The only reason confidence is not >=90% is that a future cycle could argue this is "Phase-0 deferred" — but the cycle-77 fix-sweep claimed to address C76-1 in full, not in part, and the deferred-known list does not include float FFI returns.

---

## Sibling-class checks that came back clean

- **Other call dispatches.** Grep for `CALL_INDIRECT|SHAPE_CALL|AGENT_CALL|VIRTUAL_CALL|METHOD_CALL` returned no matches. Only `OpKind.CALL` and `OpKind.FFI_CALL` exist; both arg paths now split by class.
- **`const_int(1)` / `const_int(0)` literal-construction in arithmetic ops.** Scanned all `const_int(N)` sites in `helixc/ir/lower_ast.py`. Apart from the for-range increment (F1, fixed), all other const_int(N) calls either: (a) feed array indexing (i32 by IR convention, matches LOAD_ELEM/STORE_ELEM i32 idx); (b) are placeholder fallbacks for missing-operand recovery (`or self.builder.const_int(0)`) whose downstream consumers also default to i32; (c) hard-coded i32 hash constants (lines 1381–1389) whose ADD/MUL ops explicitly pin `result_ty=i32`. No additional ADD(iN, i32, result_ty=iN) shapes found at lowering time.
- **CALL arm vs. FFI_CALL arm (arg side).** Mirror is faithful after cycle-77. FFI_CALL is even more permissive on the int side (accepts u64 in addition to i64 for pointer-shaped args), which is correct.
- **xmm_idx counter on mixed int+float args (FFI_CALL).** Counter is initialized to 0 and only incremented inside the float branch. Independent of int_idx. Verified by reading lines 1754–1776. Correct.
- **CSE / DCE / FDCE.** `_op_hash` includes `repr(op.results[0].ty)`, so const_int(1,dtype=i32) and const_int(1,dtype=i64) do not collide post-fix. DCE marks FFI_CALL as side-effecting (line 64). FDCE-call-graph correctly omits FFI_CALL (extern symbols are not in `module.functions`, so excluding them is correct, not a defect).
- **const_fold algebraic identities** (`x+0 → x`, `x*1 → x`, etc.). These forward only when an operand is a constant of the matching value; result-type / operand-type drift was the F1 root cause and is now plugged at the lowering layer. No const_fold-internal defect.

## Pre-existing items intentionally not flagged

- `||` lowering at `lower_ast.py:1135-1138` emits `ADD(l, r, result_ty=bool)` then `CMP_NE(sum, const_int(0))`; the backend ADD with result_ty=bool reads 32-bit slots even when operands are i64. This predates cycle 77 and is not in the cycle-77 fix-class. Not re-flagged per the deferred-known scope rules; out of cycle-78 narrow scope.
- `tile_ir.py:220` "treat as opaque for v0.1" TODO — tile lowering for GPU pipeline, separate from x86_64 path, marked TODO not silent failure.

## No code edits performed.

Read-only audit. No source files modified. Single Write to this doc only, as scoped.
