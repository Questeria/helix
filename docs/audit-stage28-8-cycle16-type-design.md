# Stage 28.8 Pre-29 Audit Gate — Cycle 16, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit**: 4c74627 (read-only)
**Cycle-15 baseline**: 1e4c3e6 (cycle-14 fix-sweep). Cycle-16 HEAD
4c74627 adds one commit — `Cycle 11 Audit A A1 (out-of-scope but
acted on): fix bootstrap test harness flake` — touching only
`helixc/tests/test_codegen.py` (+37 / -9 lines). Production code
under `helixc/frontend/`, `helixc/backend/`, `helixc/ir/` is
byte-identical to the cycle-15 baseline.

**Scope**: Counter-rotation lens. Per the user directive, this cycle
rotates the type-design lens to the **backend type-passing surface**
(`helixc/backend/x86_64.py`, `helixc/backend/ptx.py`) and the **TIR
op-result type invariants** (`helixc/ir/tir.py`). Cycles 1-15 of the
type-design audit category exclusively examined frontend
typecheck.py / monomorphize.py / struct_mono.py / autodiff.py /
check.py contract surfaces and (cycle 14-15) the `dce.py`
`SIDE_EFFECT_KINDS` set. None of the 15 prior type-design audits
read `x86_64.py:_emit_op` / `ptx.py:emit_op` / `tir.py:OpKind` from a
type-passing-soundness angle.

**Counter context** (per user directive 2026-05-10):

- Cycle 14: FULLY CLEAN (counter 0/5 → 1/5).
- Cycle 15: FULLY CLEAN (counter 1/5 → 2/5).
- Cycle 16 (this audit): if CLEAN under the strict criterion and
  conditional on the other two cycle-16 categories being CLEAN, the
  counter advances to 3/5.

---

## Cycle-16 production-code delta (since cycle-14 fix-sweep 1e4c3e6)

```
git diff 1e4c3e6..HEAD --stat -- helixc/
 helixc/tests/test_codegen.py | 46 +++++++++++++++++++++++++++++-----
 1 file changed, 37 insertions(+), 9 deletions(-)
```

The one commit between cycle-14 fix-sweep and HEAD (`4c74627`) is
test-only: a harness-side fix to `test_bootstrap_kovc_full_pipeline_arithmetic`
that asserts `/tmp/helix_bin_out.bin` exists before chmod+exec. No
production-code change. The full `helixc/frontend/`,
`helixc/backend/`, and `helixc/ir/` subtrees are byte-identical to
the cycle-15 baseline.

Cross-check against cycle-15 baseline 1e4c3e6:

```
git diff 1e4c3e6..HEAD -- helixc/frontend/   (empty)
git diff 1e4c3e6..HEAD -- helixc/backend/    (empty)
git diff 1e4c3e6..HEAD -- helixc/ir/         (empty)
git diff 1e4c3e6..HEAD -- helixc/check.py    (empty)
```

---

## Prior-cycle finding re-verification

| ID | Severity prev | Audit (prev) | Status now | Notes |
|---|---|---|---|---|
| — | n/a | type-design (cycle 15) | n/a (was CLEAN) | Cycle 15's type-design audit was CLEAN; no prior finding to re-verify. Per user directive, prior-cycle findings are NOT re-flagged. |

The five originally-scoped type-system contract surfaces remain
byte-identical to cycles 14 and 15:

| Surface | Location | Status |
|---|---|---|
| `_compatible` TyMemTier strict-separation | `helixc/frontend/typecheck.py:2248-2276` | unchanged |
| `_size_compatible` shape-position cascade | `helixc/frontend/typecheck.py:2232-2246` | unchanged |
| `_check_call_basic` symmetric filter | `helixc/frontend/typecheck.py:687-757` | unchanged |
| `Monomorphizer.run` iteration order | `helixc/frontend/monomorphize.py:433-492` | unchanged |
| `check.py` env-error helper + outer dispatch | `helixc/check.py` | unchanged |
| `SIDE_EFFECT_KINDS` set (cycle-14 addition) | `helixc/ir/passes/dce.py:32-81` | unchanged |

No new defect surfaced on those carried surfaces.

---

## Rotated lens: backend type-passing surface (NEW THIS CYCLE)

### Surface B1: `_is_float_type` / `_is_f64_type` / `_is_i64_type` / `_is_u64_type`

**Location**: `helixc/backend/x86_64.py:959-970`.

Type-narrowing predicates. Each is an `isinstance(ty, tir.TIRScalar)` +
name match. Used as the discrimination key in the entire `_emit_op`
dispatch tree (~500 sites of "if `_is_f64_type(ty)`: 64-bit path; elif
`_is_float_type(ty)`: f32 path; elif `_is_i64_type(ty)`: 64-bit int
path; else: i32 path").

**Invariant**: every type that the `_emit_op` dispatch handles is one
of {f32, f64, i32, i64, u64, bool}. Anything else falls through to
the default 32-bit-int path.

**Verification**:
- `_check_float_supported` (line 972-981) explicitly rejects f16/bf16
  (only narrow non-i32 IR types not yet supported).
- u64 is only produced by `STR_PTR` (lower_ast.py:1134) and `TyPtr`
  lowering (lower_ast.py:394). Verified there is **no path** that
  produces a `CONST_INT` with u64 result type — grep `CONST_INT.*u64`
  in `helixc/` returns no production matches. Surface 16.5 FFI is
  the only u64 producer and the only u64-consuming op path is
  `FFI_CALL` (line 1655-1698), which uses `_is_u64_type`.
- All ADD / SUB / MUL / DIV / MOD / BIT_AND/OR/XOR / SHL / SHR /
  BIT_NOT / NEG paths dispatch on `_is_f64_type` then
  `_is_float_type` then `_is_i64_type`, then default i32. The
  default-i32 fallback is correct for all bool / i8 / i16 / i32
  inputs (the upper bits of 32-bit-narrower types are zero-extended
  on Linux SysV ABI argument passing).

**Finding count from Surface B1**: 0 HIGH, 0 MEDIUM, 0 LOW.

---

### Surface B2: PTX backend type-passing (`helixc/backend/ptx.py`)

**Location**: `helixc/backend/ptx.py:32-352`.

The PTX emitter does its own type dispatch via:

- `_ptx_type_str` (line 157-168): maps `TIRScalar` name to PTX `.b32`
  / `.f32` / `.pred` / etc. Default for non-`TIRScalar`,
  non-`TIRUnit` is `.b64` (treats unknown types as pointer-shaped).
- `_dtype_size`, `_ptx_load_suffix`, `_ld_reg_prefix` (line 327-346):
  dtype-name string keys. Default `_dtype_size` is 4, default
  `_ptx_load_suffix` is `u32`. These defaults are only reached if a
  TILE_INDEX_LOAD/STORE op has an unknown `dtype` attr — the
  frontend always supplies one of the recognized names.
- `SCALAR_ADD` / `SCALAR_MUL` / `SCALAR_SUB` / `SCALAR_NEG` (line
  180-229): dispatch on the **register prefix** of the operand
  (`a_reg.startswith("%f")`). This is a side-channel: the PTX
  backend infers operand kind from the register name pool, not from
  TileValue.ty. Discussed below.

**Verification of register-prefix dispatch soundness**:

Each TileValue that lands in the reg_map enters through exactly one
op-emission site, and each of those sites picks a register prefix
based on a deterministic dtype source:
- `SCALAR_CONST_INT` → `%r` (integer)
- `SCALAR_CONST_FLOAT` → `%f` (float)
- `SCALAR_CMP` → `%p` (predicate)
- `THREAD_IDX` → `%r` (integer)
- `TILE_INDEX_LOAD_HBM` → `_ld_reg_prefix(dtype)` — `%f` for f-family
  dtypes, `%rd` for 64-bit int family, `%r` otherwise

Mixed-prefix dispatch is symmetric (`a_reg.startswith("%f") or
b_reg.startswith("%f")` — either operand). The dispatch covers:
- both float → float op
- both int → int op
- mixed → emits float op (implicit promotion)

This is consistent with how the surface language types tile-scalar
arithmetic (float is the wider type). No type-design defect.

**One subtle concern (NOT a finding)**: `_format_param` (line 152-155)
always emits `.param .b64` for kernel params regardless of TileValue
type. This is intentional Phase-0 convention — all kernel params are
addresses (pointers) so 64-bit width is correct. Confirmed by
inspection of `emit_kernel`'s use of `cvta.to.global.u64`.

**Finding count from Surface B2**: 0 HIGH, 0 MEDIUM, 0 LOW.

---

### Surface B3: SSA-value slot widths vs. ALLOC_ARRAY element-type tracking

**Location**: `helixc/backend/x86_64.py:838-855` (allocators) +
`helixc/backend/x86_64.py:2714-2748` (LOAD_ELEM / STORE_ELEM
codegen).

#### Description

`FnCompiler._alloc_slot` (line 857-861) reserves 8 bytes per SSA
value regardless of element type. `_alloc_array` (line 845-855)
reserves `length * 8` bytes per array element with a hard-coded
`elem_size = 8` parameter, documented as "8 bytes per element (i32
zero-padded) for simplicity in v0.1".

So the storage allocation is **always 8 bytes per element / per
SSA-value slot**. This is conservatively wide and safe in isolation.

**The defect lies in the load/store paths.** Both `LOAD_ELEM`
(line 2716-2733) and `STORE_ELEM` (line 2734-2748) emit:

```
LOAD_ELEM:    mov ecx, [rbp+idx_slot]    ; idx
              movsxd rcx, ecx
              mov eax, [rbp + rcx*8 + base]   ; ← 32-BIT LOAD ONLY
              mov [rbp + res_slot], eax       ; ← 32-BIT STORE ONLY

STORE_ELEM:   mov ecx, [rbp+idx_slot]
              movsxd rcx, ecx
              mov eax, [rbp + val_slot]       ; ← 32-BIT LOAD ONLY
              mov [rbp + rcx*8 + base], eax   ; ← 32-BIT STORE ONLY
```

Neither path branches on the array's element type — there is no
`_is_f64_type` / `_is_i64_type` check. The `esize` value unpacked
from `self.array_info[name]` (lines 2718, 2736) is silently
ignored.

#### Reachability

The TIR-level invariant for ALLOC_ARRAY / STORE_ELEM / LOAD_ELEM is
that the element type matches the array's declared dtype. This
invariant is preserved by the lowerer:

- `lower_ast.py:920` (ArrayLit lowering): `elem_ty = elem_vals[0].ty`
  — propagates the FIRST element's IR type as the array's element
  dtype.
- `lower_ast.py:1005-1006`: `FloatLit` with `type_suffix="f64"`
  produces `CONST_FLOAT` with `TIRScalar("f64")` result type.
- `lower_ast.py:927-928`: subsequent `STORE_ELEM` ops carry that
  value as their second operand.
- `lower_ast.py:1166-1168`: `LOAD_ELEM` for array indexing returns
  `result_ty=elem_ty`, again propagating f64.

The typechecker (`helixc/frontend/typecheck.py:1657-1660`,
`_check_expr` for `A.ArrayLit`) accepts ANY uniform element type —
including f64 / i64 / u64. There is no narrowing.

I verified end-to-end reachability by running the actual production
pipeline on a minimal source:

```
fn main() -> i32 {
    let xs = [1.0_f64, 2.5_f64];
    let y = xs[0];
    if y > 0.5 { 1 } else { 0 }
}
```

Result:
- `typecheck(prog)` returns `[]` (no diagnostics).
- `lower(prog)` produces 3× `CONST_FLOAT` with `result_ty=TIRScalar(name='f64')`,
  `ALLOC_ARRAY`, 2× `STORE_ELEM`, 1× `LOAD_ELEM` with
  `result_ty=TIRScalar(name='f64')`.
- `compile_module_to_elf(mod, entry_fn='main')` returns **a
  4830-byte ELF with no exception**.

The generated code performs the f64 store as a 4-byte write at
`[rbp + rcx*8 + base]` (the low 4 bytes only), leaving the upper 4
bytes of each f64 slot as whatever uninit stack contents happened
to be there. The subsequent `LOAD_ELEM` reads back the low 4 bytes
into a slot. When the IR-level user reads the result with f64
semantics (e.g. the `y > 0.5` comparison in the test source emits
`ucomisd xmm0, xmm1` against an 8-byte slot), the upper 4 bytes
of `xmm0` are stale stack garbage.

In short: **f64 array elements are silently truncated to 32 bits.**
The exact same defect applies symmetrically to i64 and u64 array
elements.

#### Why this is a type-design defect

The TIR-level type contract is sound (`Op.results[i].ty` carries the
correct element type at every point — verified by reading both
sides). The frontend typecheck contract is sound too (it accepts
the type, propagates it, and rejects mismatched widths upstream).
The defect is at the **IR→machine boundary**: the backend treats
the IR's element-type metadata as advisory rather than load-bearing,
and silently downcasts every memory op on arrays to 32-bit. There is
no compile-time check (no `_check_array_elem_size_supported(ty)`
assertion, no early raise for non-32-bit element types).

Compare this to `_check_float_supported` (line 972-981), which
correctly errors when an f16/bf16 SSA value is allocated. There is
**no equivalent guard** for array element types, even though the
defect surface is identical: a wide IR type silently lowered to
narrow machine code.

#### Severity

This is silent miscompilation. The compiler accepts an obviously
wrong program (the typechecker says "this is a valid f64 array")
and produces machine code that does the wrong thing without raising
any error or warning. Severity is **HIGH**.

#### Confidence

≥ 95%. I have:

1. Read the LOAD_ELEM / STORE_ELEM codegen paths and confirmed they
   emit only 32-bit operations regardless of element type.
2. Read the lowerer and confirmed it propagates f64 / i64 / u64
   element types into ALLOC_ARRAY / STORE_ELEM / LOAD_ELEM
   unconditionally.
3. Read the typechecker and confirmed it accepts any uniform array
   element type.
4. Run the production pipeline (parse → typecheck → lower →
   compile) on a minimal `[1.0_f64, 2.5_f64]` source and confirmed
   typecheck clean + IR carries f64 + codegen produces a
   no-exception ELF. The end-to-end miscompilation is reproducible.

The only mitigating factor: **no existing test exercises i64 / f64
/ u64 array literals**. Grep `helixc/tests/*.py` for
`array.*i64|array.*f64|let.*\[.*_i64|let.*\[.*_f64` returns
nothing. So the defect has gone undetected because the test suite
never gives it a chance to fail. This is *exactly* the silent-
failure pattern that the cycle-8 audit category names: a typed-
unsafe path that production code happens not to exercise.

The same defect exists for **struct literals with i64/f64 fields**
(struct literals lower to ArrayLit-shaped STORE_ELEM sequences at
`lower_ast.py:871-878`), **tuple literals** with i64/f64 elements
(line 896-903), and **match-bound struct/enum patterns** that use
LOAD_ELEM for field access. The blast radius is the entire
i64/f64/u64-bearing aggregate surface.

#### Recommendation

**Minimal fix** (single source file, ~15-20 lines):

1. Add a `_check_array_elem_size_supported(ty)` helper in
   `FnCompiler` modeled on `_check_float_supported`. Raise
   `NotImplementedError` for `_is_f64_type(ty) or _is_i64_type(ty)
   or _is_u64_type(ty)` element types — until the LOAD_ELEM /
   STORE_ELEM paths grow proper 8-byte movs.

2. Call it during ALLOC_ARRAY pre-pass (line 870-874) using the
   `dtype` attribute of the IR op. This forces explicit failure at
   codegen time on every reachable wide-element array.

3. Backstop: in `_emit_op`'s LOAD_ELEM / STORE_ELEM branches, raise
   `NotImplementedError` if `op.results[0].ty` (for LOAD) or
   `op.operands[1].ty` (for STORE) is wide. Cheap belt-and-suspenders
   in case the dtype attribute path is bypassed by some lowerer
   helper.

This is the cycle-3-style "narrow + loud" pattern: prefer a
NotImplementedError at codegen boundary over silent miscompile.
Then a full fix (proper 8-byte mov / movsd paths in LOAD_ELEM /
STORE_ELEM, scaling by `esize` instead of hard-coding 8-byte stride)
can land separately as a Stage-29 deliverable.

**Larger Stage-29-class fix** (deferred): thread the IR element
type into `_alloc_array`'s `elem_size` parameter and dispatch
LOAD_ELEM / STORE_ELEM on it like the existing ADD / SUB / etc.
paths. Estimated +40 lines, +6 regression tests.

**Finding count from Surface B3**: 1 HIGH, 0 MEDIUM, 0 LOW.

---

### Surface B4: TIR `Value.ty` field as a load-bearing invariant carrier

**Location**: `helixc/ir/tir.py:107-119`.

The `Value` dataclass carries `ty: TIRType` as a non-optional field
and uses identity-by-id `__eq__` / `__hash__`. The type is the
single load-bearing channel from the lowerer to the backends.

**Invariants verified by reading**:

- `Value` is a non-frozen dataclass — `v.ty = ...` is permitted by
  Python's runtime, but no code path in `helixc/` mutates `Value.ty`
  after construction (grep `\.ty = ` returns only dataclass field
  declarations and unrelated assignments). Effectively-immutable
  by convention.
- `Op.results: list[Value]` and `Op.operands: list[Value]` are both
  non-optional `list[Value]` (not Optional, not `list[Value | int]`).
  Strong static-shape invariant.
- The `IRBuilder.emit` method (line 418-430) is the single
  construction site for ops; it requires `result_ty: Optional[TIRType]`
  and constructs a single `Value(id=..., ty=result_ty)` per result.
  No alternative construction path; no untyped result Value is
  reachable.

**Invariant strength rating** (out of 10):
- Encapsulation: 8/10 — `Value` is mutable in principle (not frozen).
  The convention "don't mutate ty after construction" is unwritten.
  A `@dataclass(frozen=True)` would make this compile-time-enforceable
  but would also break the `__hash__` override. Not a cycle-16 finding.
- Expression: 9/10 — the type system clearly communicates that every
  SSA value carries a type. Type is mandatory at construction.
- Usefulness: 10/10 — backends rely on this field for every
  dispatch decision.
- Enforcement: 7/10 — load-bearing on convention; no readonly
  property; one bad mutation could derail an entire backend. Same
  as Surface B1 — not a regression, long-standing convention.

**Finding count from Surface B4**: 0 HIGH, 0 MEDIUM, 0 LOW. (See
forward note below for an optional `frozen=True` hardening.)

---

### Surface B5: `OpKind` enum as the type-of-operation discriminator

**Location**: `helixc/ir/tir.py:125-302`.

OpKind is a `str`-Enum with 80+ members. The enum is the type-key
for `_emit_op`'s `if op.kind == tir.OpKind.X` chain. Every existing
backend dispatch site uses `==` against `tir.OpKind` enum members
— no string comparison, no integer indexing.

**Invariants**:

- All enum members are declared at module-init; impossible to add
  members at runtime (Python enum semantics).
- All backend dispatch sites I read use `tir.OpKind.X` member
  references — typo'd member names would raise `AttributeError` at
  module-import time.
- The `SIDE_EFFECT_KINDS` set (covered in cycles 14-15) keys
  directly off enum members; same compile-time safety.

**Verification**: grep `tir\.OpKind\.` across `helixc/backend/x86_64.py`
returns 95 matches, all of which are member accesses against the
89-member `OpKind` enum. No string literal comparisons against
`op.kind.value` exist in production code. Independent grep
across `helixc/ir/` shows the same pattern.

**Finding count from Surface B5**: 0 HIGH, 0 MEDIUM, 0 LOW.

---

## Findings summary

| ID | Severity | Confidence | Location | Description |
|---|---|---|---|---|
| C16-1 | HIGH | ≥ 95% | `helixc/backend/x86_64.py:2714-2748` + `:845-855` | LOAD_ELEM / STORE_ELEM silently truncate i64/f64/u64 array elements to 32 bits. Typecheck + lowerer accept and propagate the wide type; backend ignores `op.results[i].ty` / `op.operands[i].ty` and emits 32-bit memory access. End-to-end miscompilation confirmed via production pipeline run on `[1.0_f64, 2.5_f64]`. |

**Total**: 1 HIGH, 0 MEDIUM, 0 LOW under the type-design audit
category.

---

## Cycle 16 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity at confidence ≥ 80.**

This cycle finds **1 HIGH finding (C16-1)**.

By the strict criterion, **cycle 16 does NOT count CLEAN** for the
type-design audit category.

**Counter status (5-clean-consecutive gate under the strict
criterion)**:
- Was 2/5 after cycle 15 full-clean.
- Cycle 16's type-design audit finds 1 HIGH (C16-1). Cycle 16 is
  not full-clean. **Counter resets to 0/5.**
- Stage 29 is now gated by five fresh consecutive clean cycles
  after the C16-1 fix-sweep lands.

The severity trend across cycles, against the strict-criterion bar:
- Cycle 1: HIGH-tier — not clean
- Cycle 2: HIGH + MEDIUM — not clean
- Cycle 3: HIGH + MEDIUM + LOW — not clean
- Cycle 4: MEDIUM — not clean
- Cycle 5: 3 MEDIUM + 3 LOW — not clean
- Cycle 6: 1 MEDIUM + 2 LOW — not clean
- Cycle 7-12: 0 + 0 + 0 — clean (counter 1/5 → 3/5)
- Cycle 13: 1 HIGH (C13-1) — not clean → reset to 0/5
- Cycle 14: 0 + 0 + 0 — clean → 1/5
- Cycle 15: 0 + 0 + 0 — clean → 2/5
- Cycle 16 (type-design): 1 HIGH (C16-1) — not clean → reset to 0/5

**Recommendation**: a cycle-16 fix-sweep is required before Stage 29
can proceed. Minimal narrow + loud fix (~15-20 lines, single
source file) recommended:

1. `_check_array_elem_size_supported(ty)` helper in `FnCompiler`,
   raises NotImplementedError for i64/f64/u64 element types.
2. Call from ALLOC_ARRAY pre-pass (line 870-874) on the IR
   `dtype` attr.
3. Backstop in LOAD_ELEM / STORE_ELEM branches on
   `op.results[0].ty` / `op.operands[1].ty`.

Regression tests:
- `test_c16_1_array_f64_rejected`: assert `[1.0_f64, 2.0_f64]`
  raises NotImplementedError at codegen with a clear "f64 array
  elements not yet supported" message.
- `test_c16_1_array_i64_rejected`: same for i64.
- `test_c16_1_array_i32_still_works`: assert the existing i32
  array path is unaffected.
- (Stage-29 deferred): correctness tests for full 8-byte LOAD_ELEM
  / STORE_ELEM lowering once that lands.

---

## Forward notes (not cycle-16 findings; carried from prior cycles
or newly recorded)

1. **Empty-string edge case for `_emit_env_error`**: no test asserts
   `_emit_env_error("")` produces `helixc: ` (and remains stable
   across refactors). No production callee passes empty. Not
   blocking. (Carried from cycle 10.)

2. **Nested-prefix edge case for `_emit_env_error`**: no test
   asserts `_emit_env_error("helixc: helixc: foo")` strips exactly
   one layer. Not blocking. (Carried from cycle 10.)

3. **Whitespace-handling edge case for `_emit_env_error`**: not
   blocking. (Carried from cycle 10.)

4. **Convention note for raise-message prefix**: not blocking.
   (Carried from cycle 10.)

5. **`SIDE_EFFECT_KINDS` static cross-check**: no static guarantee
   that every `OpKind` with side-effect semantics is in the set.
   Stage-29-class enum-attached-metadata hardening recorded in
   cycle 14. (Carried from cycle 14.)

6. **`Value.ty` not frozen** (NEW, cycle 16): `tir.Value` is a
   `@dataclass` (not `@dataclass(frozen=True)`). Conventionally
   immutable but not enforced. Hardening would require dropping
   the manual `__hash__` / `__eq__` overrides (since frozen
   dataclasses auto-generate eq=hash) and ensuring all id-based
   identity callers still get the right semantics. Not blocking;
   no in-tree mutation. Stage-29-class consideration.

7. **`Op.results: list[Value]`** (NEW, cycle 16): an `Op` may carry
   zero, one, or more results. The "single result or zero results"
   invariant — relied on by `IRBuilder.emit` (line 430:
   `return results[0] if results else None`) — is not enforced by
   the dataclass. Multi-result ops are theoretically representable;
   in practice no lowerer emits one. Not blocking; recorded for
   Stage-29 type-system rewrite. Could be expressed via
   `result: Optional[Value]` instead of `results: list[Value]` to
   make the invariant compile-time.

8. **Backend `_alloc_array` `elem_size` parameter** (NEW, cycle 16):
   `elem_size` is a parameter on `_alloc_array` (default 8) but
   every call site uses the default. The IR-level `ALLOC_ARRAY`
   op's `dtype` attr is read at line 871 but only the `name` and
   `length` are propagated; the dtype is silently dropped. The
   parameter exists but the wiring to flow the IR dtype through is
   absent. This is the structural mirror of C16-1: data is present
   in the IR, the backend has a parameter to receive it, but the
   connection is not made.

9. **PTX backend `_format_param` hard-codes `.param .b64`** (NEW,
   cycle 16): correct for Phase-0 kernel-pointer convention but
   silently incorrect if a kernel param is ever a non-pointer
   scalar. No production lowerer currently produces such; only
   recorded for documentation completeness.

10. **Cycle-17 baseline confirmation**: if the cycle-16 fix-sweep
    introduces a new test surface (e.g., `_check_array_elem_size_supported`
    helper + regression tests), cycle 17's type-design audit
    should give those a full read rather than relying on
    empty-diff shortcuts. Process note.

11. **Stage-29 readiness**: counter is reset to 0/5 by C16-1.
    Five clean cycles remain required after C16-1 fix-sweep lands.
