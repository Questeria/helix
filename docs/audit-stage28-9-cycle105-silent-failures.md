# Audit Stage 28.9 cycle 105 — Silent failures

**Scope**: HEAD `77e4b85` (`Stage 28.9 cycle-104 audits: 3/3 CLEAN, counter 1/5 -> 2/5`).

**Mode**: STRICT READ-ONLY. No source edits performed. This document is
the sole write of the audit. Source files only read/grepped.

**Rotation (per prompt — avoid recently-audited cmp/ADD/SUB/MUL):**
- `helixc/ir/passes/cse.py` — CSE for ADD/SUB/MUL post-cycle-102 width changes.
- `helixc/frontend/typecheck.py` — integer-promotion rules (32→64 widening).
- `helixc/backend/x86_64.py` — CONST_INT u64 emission path.

**Deferred-known list (NOT re-flagged):**
monomorphize `_mangle_ty` / hash_cons `_ast_equal` silent catchalls;
typecheck/struct_mono pre-flatten in `check.py`;
`autotune.collect_autotuned_fns` missing `iter_fn_decls`;
`struct_mono.mangle_struct` collision;
`A.StrLit` IR-lowering gap (cycle-101 F1 deferred);
DIV/MOD signed-vs-unsigned emit (cycle-101 deferred);
SHR signed-vs-unsigned emit (cycle-101 deferred).

---

## Cycle-102 fix verification (ADD/SUB/MUL u64)

Verified at HEAD `77e4b85`:

- `_is_64bit_int_type` predicate at `x86_64.py:1033` returns true for
  `{i64, isize, u64, usize}`. PASS.
- ADD/SUB/MUL emit at lines 1329 / 1359 / 1387 each dispatch on
  `_is_64bit_int_type(op.results[0].ty)`. PASS — cycle-101 F2
  remediated for these three opcodes.
- Regression test `test_c102_u64_add_emits_64bit_path` (`helixc/tests/
  test_ir.py:256`) asserts byte sequence `\x48\x01\xc8` (rex.W
  `add rax, rcx`) is present in the emitted ELF for `fn add_u64(a:
  u64, b: u64) -> u64 { a + b }`. The opcode is present.

However, the cycle-102 sweep was scoped to the ADD/SUB/MUL emit sites
only. The defect class — `_is_i64_type` name-equality missing the
pointer-width-alias siblings `u64`/`usize` — survives in **other**
emit sites where a u64/usize value transits the stack. See F1–F3.
The cycle-102 regression test inspects emitted bytes but does not
execute the result, so upstream operand truncation (CONST_INT u64,
param-spill u64) is not detected by it: the rex.W add executes
against slot operands whose high 32 bits are stale/uninitialized from
a 32-bit-only producer upstream.

---

## Rotation targets — clean

### cse.py (`helixc/ir/passes/cse.py`)

`_op_hash` at line 76 includes
`result_ty_key = repr(op.results[0].ty)` in the hash tuple, so an i32
ADD and an i64 ADD with operand-id-equal inputs hash to distinct
keys. Operand ids alias the producer's result Value, whose `ty`
carries the original width; two producers with the same numerical
value still receive distinct ids under IRBuilder's `new_value`
numbering, so two semantically-distinct widths never collide. Cycle
21 audit-T C20-T4 added BIT_*, SHL, SHR, MAXIMUM, MINIMUM, POW,
BITCAST — coverage matches `const_fold`'s foldable set. The defensive
list-copy at line 147 (cycle-18 C18-C1) prevents alias-mutation by a
downstream pass. Per-block scoping (no cross-block CSE without
dominance analysis) is sound by construction. The cycle-102 width
changes (ADD/SUB/MUL using rex.W for u64) do not perturb CSE
correctness — `_op_hash` already keyed on `repr(op.results[0].ty)`,
so the u64 fix is transparent to the dedupe key. No finding.

### typecheck.py integer-promotion / widening

`_widen_diff_inner` (line 257) fires only inside the `TyDiff` /
`TyLogic` binop branch (lines 1358-1502). The bare-arithmetic
fall-through at line 1503-1504 returns `l` (left operand's type) as
the integer binop result — explicitly labelled "simplified". For
mixed-width binops (`i32 + i64`, `u32 * u64`) this is a typing-
fidelity loss in a single direction: the inferred type for an
unannotated `let x = a_i32 + b_i64;` is i32, not i64. But:

- The downstream `_compatible` check at let-binding boundaries
  (line 1196) catches mismatches against declared types — so
  `let x: i64 = a_i32 + b_i64;` errors (declared i64 vs body i32).
- The IRBuilder `add()` at `tir.py:441` independently uses operand
  `a.ty` for the IR result type — IR codegen does not consume the
  typecheck inferred type.
- The `_INT_BOUNDS` table at line 1857 enforces literal-fits-width at
  the let-statement only when the value is a bare `A.IntLit`; mixed
  binops bypass it.

The silent-coercion path is therefore a name-display fidelity loss
in diagnostics (the printed `_fmt(value_ty)` says `i32` when the
backing IR is i64), but is not a backend miscompile on its own. The
trailing "(simplified)" comment is a documented acknowledgement.
Below the 75% silent-failure threshold (a documented simplification
that does not cause runtime miscompile alone). No finding.

### x86_64.py CONST_INT u64 emission

See F1 below.

---

## Findings

### F1 — CONST_INT u64/usize silently truncates high 32 bits (HIGH, conf 92)

**File**: `helixc/backend/x86_64.py`, `_emit_op` CONST_INT arm at
lines 1195-1204.

```python
if op.kind == tir.OpKind.CONST_INT:
    slot = self._slot_of(op.results[0])
    value = int(op.attrs["value"])
    if self._is_i64_type(op.results[0].ty):
        self.asm.mov_rax_imm64(value)
        self.asm.mov_mem_rbp_rax(slot)
    else:
        self.asm.mov_eax_imm32(value & 0xFFFFFFFF)
        self.asm.mov_mem_rbp_eax(slot)
    return
```

`_is_i64_type` (line 1019) returns true only for `{i64, isize}`. A
CONST_INT whose result type is `u64` or `usize` therefore takes the
else branch: 32-bit `mov eax, imm32` + 32-bit store into a slot that
is always 8 bytes wide (`_alloc_slot` at line 897 hard-codes 8 bytes
per value). The high 4 bytes of the slot are left as whatever the
previous use of that frame offset wrote — uninitialized on first use,
stale from a prior i32 store otherwise.

A `u64`-typed IntLit reaches CONST_INT via `helixc/ir/lower_ast.py:1038`:

```python
if isinstance(expr, A.IntLit):
    return self.builder.const_int(expr.value, expr.type_suffix or "i32")
```

`IRBuilder.const_int` at `helixc/ir/tir.py:432` stamps the result
type as `TIRScalar(dtype)`, so a literal `5_000_000_000_u64` (or any
small `u64` literal whose downstream consumer depends on a zero high
half) lowers to a CONST_INT with TIRScalar("u64") and emits the
truncated 32-bit store.

This is the same defect class as cycle-101 F2 / cycle-100 F1 /
cycle-19 C18-1: `_is_i64_type` name-equality missing the pointer-
width-alias siblings. Cycle-102 introduced `_is_64bit_int_type` and
swept ADD/SUB/MUL emit. CONST_INT was not swept.

**Impact**: `let x: u64 = (1u64 << 40);` followed by any 8-byte read
of `x` reads the corrupt slot. Even when the literal value fits in
32 bits, the high 4 bytes of the slot are stale — a downstream
rex.W ADD path (cycle-102 fixed) executes `mov rax, [rbp+slot]`
loading 8 bytes including the garbage high half, then a sign-agnostic
`add rax, rcx` propagates the garbage. Result returned in rax is
wrong; no diagnostic.

The cycle-102 byte-pattern test `assert b"\x48\x01\xc8" in elf`
passes (the rex.W opcode is present) but does not execute the
program, so this runtime miscompile is not caught by the heavy gate.
No existing helixc-Python test runs a `u64` CONST_INT through to
exit-code verification — the `u64`-suffix end-to-end tests at
`test_codegen.py:3123+` route through `compile_and_exec` (kovc
bootstrap), not `compile_and_run` (helixc Python).

**Recommended fix**: extend the CONST_INT branch predicate from
`_is_i64_type` to `_is_64bit_int_type` — a one-line change mirroring
cycle-102's sweep. Regression test: `test_const_int_u64_emits_movabs`
asserting `\x48\xb8` (movabs rax, imm64 = rex.W + `mov rax, imm64`)
or `\x48\x89` (mov [rbp+off], rax) appears for a `let x: u64 =
0x1_0000_0001_u64;` source.

### F2 — Param-spill u64/usize silently truncates to 32-bit register store (HIGH, conf 92)

**File**: `helixc/backend/x86_64.py`, function-prologue param-spill
loop at lines 969-990.

```python
if self._is_i64_type(p.ty):
    INT_SPILLS_64[int_idx](slot)
else:
    INT_SPILLS[int_idx](slot)
```

`INT_SPILLS_64` (line 961) lists `mov [rbp+off], rdi/rsi/rdx/rcx/r8/
r9` (8-byte). `INT_SPILLS` (line 953) lists `mov [rbp+off],
edi/esi/edx/ecx/r8d/r9d` (32-bit, which zero-extends the destination
register but writes only 4 bytes to memory). For a `u64`/`usize`
param, `_is_i64_type(p.ty)` is false → the 32-bit spill writes only
the low 4 bytes of the 8-byte slot. SysV ABI delivers the full 8
bytes of a `u64` arg in rdi/rsi/etc.; the high 4 bytes are intact in
the register at entry but discarded by the dword spill.

Same `_is_i64_type`-only membership miss as F1 / cycle-101 F2.

**Impact**: `fn add_u64(a: u64, b: u64) -> u64 { a + b }` (the exact
source of cycle-102's regression test at `test_ir.py:268`) executes:

1. Caller passes a, b in rdi, rsi with full 8 bytes valid.
2. Prologue param-spill emits 32-bit stores → slot[a] low 4 bytes
   correct, high 4 bytes = stack-prior garbage.
3. ADD emit (correctly cycle-102-fixed) loads slot[a] via
   `mov rax, [rbp+slot_a]` — 8-byte load includes garbage high half.
4. `add rax, rcx` propagates garbage.
5. Result returned in rax as garbage in the high 32 bits.

The cycle-102 byte-pattern test passes (the opcode `\x48\x01\xc8` is
present) but does not execute the result. A runtime-execution u64-
param test (e.g. caller pushes `0x1_0000_0001` in rdi, callee returns
`a + 1`, harness checks `rax == 0x1_0000_0002`) would catch this.

**Recommended fix**: gate the param-spill on `_is_64bit_int_type` —
identical one-line change to F1's fix.

### F3 — BITCAST / CAST u64-side predicates miss u64/usize (HIGH, conf 80)

**File**: `helixc/backend/x86_64.py`, BITCAST emit at lines 1228-1243
and CAST emit at lines 1244-1309 (predicates at lines 1253-1254).

BITCAST wide-classifier:
```python
wide = self._is_f64_type(res_ty) or self._is_i64_type(res_ty) \
       or self._is_f64_type(op.operands[0].ty) \
       or self._is_i64_type(op.operands[0].ty)
```

A `bitcast<u64>(f64)` or `bitcast<f64>(u64)` — exactly the use-case
the inline comment "f32 <-> i32: 4-byte mov; f64 <-> i64: 8-byte mov"
calls out — falls through to the narrow `mov_eax_mem_rbp` +
`mov_mem_rbp_eax` 4-byte copy when the integer side is typed `u64`
or `usize`. High 4 bytes dropped from source; high 4 bytes of dest
slot stale.

CAST predicates:
```python
from_is_i64 = self._is_i64_type(from_ty)
to_is_i64   = self._is_i64_type(to_ty)
```

The CAST dispatch table at lines 1255-1309 routes on `from_is_i64`
and `to_is_i64` for the i64↔i32 / i64↔f64 / i64↔i64 arms. A
`cast<u64, i32>` (intended truncation) executes the wrong arm — none
of the i64-specific arms fire — and falls to the bottom "same width"
arm at line 1303-1309 which emits a 4-byte copy with high half
unset. A `cast<u32, u64>` misses the `movsxd` i32→i64 sign-extension
arm (line 1261); even if it hit that arm, sign-extension is wrong
for a u32 source — the correct emit is zero-extension (a bare
`mov eax, [src]` against a 4-byte source destination-zero-extends the
register, which is the desired u32→u64 widening).

This site also bears the same hint typecheck.py:1786 actually
emits ("use a bitcast through a u64 intermediate" — for ptr-to-int
round trips), so the front-end actively guides users toward the
broken codegen path.

**Impact**: lower probability than F1/F2 because explicit
`bitcast`/`as` over u64 is less common than direct u64 arithmetic,
but any explicit `as` cast targeting u64 from a narrower type, or a
bitcast through a u64 intermediate (the exact pattern the typecheck
hint recommends), produces a silent miscompile.

**Recommended fix**: extend `wide`, `from_is_i64`, `to_is_i64` to
use `_is_64bit_int_type`. Add an explicit zero-extension arm for the
unsigned-narrow→unsigned-wide cases (`mov eax, [src]` + `mov [dst],
rax` — relies on x86-64's implicit zero-extension when writing a
32-bit destination register).

---

## Summary

**3 findings at confidence ≥ 75%**, all of the same root class:
`_is_i64_type` predicate name-equality missing the
`u64`/`usize` pointer-width-alias siblings, at emit sites that
cycle-102's ADD/SUB/MUL sweep did not touch (CONST_INT, param-spill,
BITCAST, CAST). Common fix is the one-line predicate swap to
`_is_64bit_int_type`, mirroring cycle-102.

The cycle-102 regression test (`test_c102_u64_add_emits_64bit_path`)
passes despite F1 / F2 because it inspects emitted byte patterns
(`\x48\x01\xc8`) rather than executing the result; the truncation
lives in the operand-producers (param-spill, CONST_INT), and the
correctly-emitted rex.W add silently propagates the corrupted high
half.

---

## Findings table

| Severity | Count |
|---|---|
| CRITICAL (conf ≥ 90) | 0 |
| HIGH (conf ≥ 75)     | 3 (F1 conf 92, F2 conf 92, F3 conf 80) |
| MEDIUM (60-74)       | 0 |
| LOW (<60)            | 0 |

---

## Verdict

**FAIL** — 3 findings at confidence ≥ 75% within cycle-105 scope.

Counter: cycle 105 FAIL → reset 2/5 → 0/5.

---

## Cross-references

- Cycle-102 fix-sweep commit `26dfa82`: closed ADD/SUB/MUL u64 only
  (`_is_64bit_int_type` introduced) — sibling emit sites (CONST_INT,
  param-spill, BITCAST, CAST) deferred (this audit's findings).
- Cycle-101 silent-failures F2 (`docs/audit-stage28-9-cycle101-
  silent-failures.md`): the root-cause defect class (u64/usize
  name-equality miss in backend predicates).
- Cycle-100 F1 (cmp dispatch unsigned-vs-signed setcc): the original
  surfacing of the alias-miss class in `cmp` emit; cycle-102
  generalised to ADD/SUB/MUL; cycle-105 surfaces the rest.
- Cycle-19 C18-1 (`isize`/`usize` backend classifier): the seminal
  alias-miss fix this class derives from.
- Source-of-truth files: `helixc/backend/x86_64.py` (CONST_INT 1195-
  1204; param-spill 969-990; BITCAST 1228-1243; CAST 1244-1309;
  predicates 1019-1042), `helixc/ir/lower_ast.py:1038` (u64 IntLit
  lowering), `helixc/ir/tir.py:432` (`const_int` result-type stamp).
