# Stage 28.8 Cycle 20 — Silent-Failure Audit (Audit A)

**Date**: 2026-05-11
**Audit HEAD**: `5a1e406` — "Audit 28.8 cycle 20 fix-sweep:
close C19-1 (HIGH, const_fold isize 32-bit drift)".

**Context (fresh streak — cycle 1 of 5)**: Cycle 19 Audit A surfaced
**C19-1 / HIGH** — `const_fold._INT_BITS` mapped `isize=32`/`usize=32`,
contradicting the cycle-19 backend classifier fix (which canonicalized
isize→i64, usize→u64). End-to-end reproducer `3_000_000_000_isize +
3_000_000_000_isize` was foldable but wrapped at 32 bits, returning
`1_705_032_704` instead of `6_000_000_000`.

The cycle-20 fix-sweep at `5a1e406` patched the two table entries:

```python
# helixc/ir/passes/const_fold.py:43-56
_INT_BITS = {
    "i8": 8, "u8": 8,
    "i16": 16, "u16": 16,
    "i32": 32, "u32": 32,
    # Audit 28.8 cycle 20 C19-1 (HIGH): pointer-width aliases must be
    # 64-bit, matching typecheck.py:225-228's `_widen_canon_name`
    # aliasing (isize->i64, usize->u64) and the cycle-19 backend
    # classifier fix at x86_64.py:1005-1017. Pre-fix the 32-bit
    # entry made `_wrap_int_to_type(6_000_000_000, isize) =
    # 1_705_032_704` — silent miscompile reachable at default -O1.
    "isize": 64, "usize": 64,
    "i64": 64, "u64": 64,
    "bool": 32,
}
```

Plus a new regression test `test_c19_1_isize_usize_are_64_bit_in_wrap`
(test_const_fold.py:356-383) that pins agreement between isize↔i64
and usize↔u64 in `_wrap_int_to_type` across a representative value
range including the 32-bit / 63-bit boundaries.

That fix reset the strict clean-cycle counter to **0/5** at the start
of cycle 20. This is cycle 1 of the new streak (need 5 consecutive
clean cycles to fire the Stage-29 gate per the user directive
2026-05-10).

**Strict criterion**: cycle counts CLEAN only when **zero new
findings of ANY severity** (CRITICAL / HIGH / MEDIUM / LOW). Findings
in the carryover ledger (audit-C4-1 CRITICAL, audit-C4-4 HIGH,
audit-C4-8 LOW, C5-10 LOW, monomorphize_safe docstring drift,
D-vs-Quote diagnostic text, C7-1 test-coverage gap) are not re-flagged
here per the strict re-flag rule (none changed since prior cycle).

**Clean-counter state going into cycle 20**: 0/5.

---

## Method

1. **Read prior cycle silent-failure verdict** (cycle 19) for context
   on C19-1, the new finding, and the cycle-19 forward note about
   "centralize the scalar-width predicate" being a Stage-29 refactor
   (deferred).
2. **`git show 5a1e406 --stat`** to confirm the fix surface:
   - Production-code delta: **+8 / -1** lines in
     `helixc/ir/passes/const_fold.py` (just the `isize`/`usize` table
     entry split + audit-stamp comment).
   - Test delta: **+31** lines in `helixc/tests/test_const_fold.py`
     (one new regression test, `test_c19_1_isize_usize_are_64_bit_
     in_wrap`).
   - Doc delta: +653 lines in
     `docs/audit-stage28-8-cycle19-type-design.md` (parallel-lens
     audit doc, already vetted in cycle 19).
3. **Verified the fix at HEAD=5a1e406** by reading
   `const_fold.py:43-56` (table entries) and `test_const_fold.py:
   356-383` (regression test). Test runs locally — commit message
   confirms "37 const_fold tests pass (was 36 cycle 17 → +1 new)".
4. **Adversarial rotation (user-directed)** — examine the three
   parallel width-aware tables called out in the cycle-20 prompt:
   - `helixc/backend/ptx.py:328-332` — PTX dtype suffix / size maps.
   - `helixc/ir/tile_ir.py` — tile-IR element-width handling.
   - `helixc/ir/lower_ast.py:357-371` — literal-type / primitive-name
     set.
   For each, confirm whether the table treats isize/usize as 32-bit
   when the canonical 64-bit treatment (typecheck.py:225-228 + 241,
   x86_64.py:1005-1017, const_fold.py:43-56) is now established.
5. **Read-only**: no edits to production code or tests during this
   audit cycle.

---

## Verification of the cycle-20 fix surface

### Table extension (const_fold.py:43-56)

Pre-fix:

```python
_INT_BITS = {
    "i8": 8, "u8": 8,
    "i16": 16, "u16": 16,
    "i32": 32, "u32": 32, "isize": 32, "usize": 32,   # WRONG
    "i64": 64, "u64": 64,
    "bool": 32,
}
```

Post-fix (verbatim from HEAD):

```python
_INT_BITS = {
    "i8": 8, "u8": 8,
    "i16": 16, "u16": 16,
    "i32": 32, "u32": 32,
    # Audit 28.8 cycle 20 C19-1 (HIGH): pointer-width aliases must be
    # 64-bit, matching typecheck.py:225-228's `_widen_canon_name`
    # aliasing (isize->i64, usize->u64) and the cycle-19 backend
    # classifier fix at x86_64.py:1005-1017. Pre-fix the 32-bit
    # entry made `_wrap_int_to_type(6_000_000_000, isize) =
    # 1_705_032_704` — silent miscompile reachable at default -O1.
    "isize": 64, "usize": 64,
    "i64": 64, "u64": 64,
    "bool": 32,
}
```

**Audit walk**:

1. **Trigger condition** — `_wrap_int_to_type` does `_INT_BITS.get(
   ty.name, 32)`. For `TIRScalar("isize")`/`TIRScalar("usize")` the
   key now lookups to 64. The `.get(..., 32)` default still applies
   to unknown names, which is the documented Phase-0 behavior for
   generic type parameters (mirrors `lower_ast.py:357-371`'s "default
   to i32-sized ABI" HBS limitation).

2. **Negative path** — anything not in the membership set still
   defaults to 32. This is unchanged from pre-fix. No new silent-
   failure window opened on the False side.

3. **Could the new 64-bit path mis-wrap any signed-vs-unsigned
   semantics?** `_wrap_int_to_type` applies a `mask = (1 << bits) -
   1` then performs a sign-correction `if v >= half: v -= (1 <<
   bits)`. For `bits=64`, `mask=0xFFFFFFFFFFFFFFFF`, `half=
   0x8000000000000000`. This produces a signed-two's-complement
   value in the i64 range. isize is signed in Helix's type system
   (typecheck.py:1816 — range `(-(1<<63), (1<<63)-1)`), so the
   signed-wrap matches semantics. usize's wrap is the **same**
   signed wrap as u64 — but that's a pre-existing property of the
   `_wrap_int_to_type` helper (it always sign-corrects; both u64
   and usize get a "signed view" of the 64-bit wrapped value).
   This is not new in cycle 20 — usize was already aliased to u64
   in this helper's signed-correction path; the fix preserves
   isize/usize ≡ i64/u64 equivalence.

4. **Could `TIRScalar(name="isize")` flow in with a meaning OTHER
   than pointer-width signed integer?** Same answer as cycle-19's
   classifier audit walk: `lower_ast.py:357-358` is the only site
   that produces `TIRScalar("isize")`, fed from the parser's
   `KW_ISIZE` keyword. No shadowed "isize" string.

5. **Raise quality** — no raises in `_wrap_int_to_type`; returns a
   value. The fold path's `FoldError` exception (const_fold.py:40)
   is unrelated to width — it's raised on undefined-behavior fold
   inputs (DIV by zero, etc.). The cycle-20 fix does not interact
   with `FoldError`. Audit-stamp comment cites "audit 28.8 cycle 20
   C19-1" with the cycle-19-doc cross-reference.

**Verdict on the table extension**: clean. The widened entries
strictly *close* the C19-1 silent-trunc window at the four
`_wrap_int_to_type` call sites (binary arith, bitwise/shift, NEG,
BIT_NOT) without opening any new ones.

### Regression test (test_const_fold.py:356-383)

```python
def test_c19_1_isize_usize_are_64_bit_in_wrap():
    """Audit 28.8 cycle 20 C19-1 (HIGH): isize/usize must be treated as
    64-bit in `_wrap_int_to_type`, ..."""
    from helixc.ir.passes.const_fold import _wrap_int_to_type
    isize = tir.TIRScalar(name="isize")
    usize = tir.TIRScalar(name="usize")
    i64 = tir.TIRScalar(name="i64")
    u64 = tir.TIRScalar(name="u64")
    # 6_000_000_000 fits in signed 64-bit, should round-trip.
    assert _wrap_int_to_type(6_000_000_000, isize) == 6_000_000_000
    assert _wrap_int_to_type(6_000_000_000, i64) == 6_000_000_000
    # The isize and i64 wraps must agree (cycle-3 alias-canon).
    for v in [0, 1, -1, 2**31 - 1, 2**31, -(2**31), 6_000_000_000,
              -(6_000_000_000), 2**62, -(2**62)]:
        assert _wrap_int_to_type(v, isize) == _wrap_int_to_type(v, i64)
    # Same for usize/u64.
    for v in [0, 1, 2**32, 2**32 - 1, 2**63, 2**63 + 1]:
        assert _wrap_int_to_type(v, usize) == _wrap_int_to_type(v, u64)
```

**Audit walk**:

- Pins the **direct contract** (`_wrap_int_to_type` agreement between
  isize↔i64 and usize↔u64) — the contract the cycle-19 audit
  identified as missing.
- Value range covers the **boundaries** the C19-1 reproducer hit
  (above 2³¹, sign-cross at -1, large values at 2⁶²) plus zero and
  ±1 baselines.
- Direct-from-import (`from helixc.ir.passes.const_fold import
  _wrap_int_to_type`) — no pipeline wrapping, so the test fails
  fast and the failure points directly at `_INT_BITS`.
- 31-line test, no parametrize decorator; explicit assertion
  messages on the loop body cite "isize/i64 wrap disagreement
  at v=...". Failure surface is debuggable.

**Note**: cycle-19's audit doc suggested a **full-pipeline** test
(compile → fold → emit → run for `3_000_000_000_isize +
3_000_000_000_isize`). The cycle-20 fix landed only the
contract-level test. This is a **test-coverage gap relative to the
cycle-19 doc's recommendation**, but it's a **test design**
question, not a silent-failure question — the contract test does
prevent regression of the specific defect (any future drift of
`_INT_BITS["isize"]` away from 64 will fail this test). The cycle-
19 doc's full-pipeline test would have been a stronger guarantee
because it would also catch isize miscompiles introduced by a
*new* width table further downstream. Filing this as a forward
note (see "Forward notes" below), not a finding — no current
silent-failure window is open from this gap.

### Spot-checked consumer sites of `_wrap_int_to_type`

Sampled all four call sites against the post-fix table semantics:

- **Binary ADD/SUB/MUL/DIV/MOD result** (`const_fold.py:334`) — the
  C19-1 reproducer `3e9 + 3e9` now folds to `6_000_000_000`.
  **Closes.**
- **Binary BIT_AND/BIT_OR/BIT_XOR/SHL/SHR result** (`const_fold.py:
  410`) — large isize bit operations now wrap at 64 bits.
- **Unary NEG result** (`const_fold.py:451`) — `-x` for `x: isize`
  with `|x| > 2³¹` now produces the correct sign-extended 64-bit
  negation.
- **Unary BIT_NOT result** (`const_fold.py:482`) — `~x` for `x:
  isize` now flips all 64 bits, not just the low 32.

All four sinks now produce values that match the post-cycle-19
backend's 64-bit emit semantics. The folded path agrees with the
un-folded path (optimization-stability restored — the cycle-20
commit message names this property explicitly).

---

## Adversarial rotation: other width-aware tables for isize/usize

User-directed verification of three other width-aware tables.

### Site 1: PTX backend dtype maps (ptx.py:327-332)

```python
_DTYPE_SIZE = {"i8": 1, "u8": 1, "i16": 2, "u16": 2, "f16": 2, "bf16": 2,
                "i32": 4, "u32": 4, "f32": 4, "i64": 8, "u64": 8, "f64": 8}
_DTYPE_PTX_LOAD = {"i8": "s8", "u8": "u8", "i16": "s16", "u16": "u16",
                    "f16": "f16", "bf16": "bf16",
                    "i32": "s32", "u32": "u32", "f32": "f32",
                    "i64": "s64", "u64": "u64", "f64": "f64"}
```

Plus a third table (`_ptx_type_str` at lines 159-165):

```python
mapping = {
    "i8": ".b8", "i16": ".b16", "i32": ".b32", "i64": ".b64",
    "u8": ".b8", "u16": ".b16", "u32": ".b32", "u64": ".b64",
    "bool": ".pred",
    "f16": ".f16", "bf16": ".bf16", "f32": ".f32", "f64": ".f64",
}
return mapping.get(ty.name, ".b32")
```

**Width-correct for isize/usize?** **None of the three tables
include `isize`/`usize` keys.** All three have `.get(..., default)`
fallbacks:

- `_DTYPE_SIZE.get(dtype, 4)` → 4 bytes (i.e., **32-bit**)
- `_DTYPE_PTX_LOAD.get(dtype, "u32")` → `u32` suffix (**32-bit**)
- `_ptx_type_str` mapping → `.b32` default (**32-bit**)

If a `TIRScalar("isize")` reached any of these three tables, it
would silently narrow to 32 bits. **Same defect class as C19-1.**

**Is this path reachable?** Walked the call chain:

1. **`_DTYPE_SIZE` / `_DTYPE_PTX_LOAD` / `_ld_reg_prefix`** are
   consulted only by `TILE_INDEX_LOAD_HBM` / `TILE_INDEX_STORE_HBM`
   (ptx.py:270-322). The `dtype` attribute passed to these helpers
   comes from the TIR op's `attrs["dtype"]`, populated at
   `lower_ast.py:1958` / `lower_ast.py:2032` from
   `_lookup_hbm_tile()[0]` (a dtype name string).
   `_bind_hbm_tile` is called at `lower_ast.py:528` with
   `dtype_node.name` — but **only after** the validation at
   `lower_ast.py:510-514`:

   ```python
   if not (isinstance(dtype_node, A.TyName)
           and dtype_node.name in ("f32", "i32", "f16", "bf16")):
       raise NotImplementedError(
           "Stage 16 HBM tile param dtype must be f32/i32/f16/bf16; "
           f"got {dtype_node}")
   ```

   isize and usize are rejected here with `NotImplementedError`
   **before** the dtype string can flow into the PTX tables. So
   the silent-narrow window for the two HBM-tile-indexed-load
   tables is **not currently reachable** — guarded by an explicit
   raise.

2. **`_ptx_type_str` mapping (line 165) — `.get(ty.name, ".b32")`
   default.** Consumed by `emit_device_func` (line 143) for the
   function's return type. If a non-kernel function returns
   `isize` or `usize`, the return-type slot in the emitted
   `.func` signature would be `.b32` instead of `.b64`. The
   function body emitted by `emit_device_func` is a stub (just
   `ret;`, line 148), so no integer result is actually produced
   — the silent-narrow is in the **signature only**.

**Is this a finding?**

The PTX backend at HEAD is Phase-0 — `emit_device_func` is a
no-op stub for non-kernel functions; the only PTX paths exercised
by the test suite are kernel-only with HBM-tile params (and those
are gated to f32/i32/f16/bf16). The kernel `params_str` at
line 107 uses `_format_param`, which hard-codes `.b64` for every
param (line 155: `return f".param .b64 param_{idx}"`) — so even
if a kernel were declared with an isize/usize scalar param, it
would currently flow through the `.b64` hard-code, not through
`_ptx_type_str`.

**Conclusion**: the `.b32` default in `_ptx_type_str` is a latent
silent-narrow hazard for isize/usize, but **not reachable at
HEAD** because:

- HBM tile path: hard NotImplementedError gate at lower_ast.py:511.
- Device-func signature: function body is a stub; no runtime
  computation.
- Kernel param: hard-coded `.b64` in `_format_param`.

The hazard is **dormant**, identical in shape to the cycle-16
audit's "PTX/dyn-ELF backends don't emit those ops yet" carve-out
that kept cycle 17 clean. Per the audit-rotation rules established
in cycle 17, dormant defects in code paths not currently reachable
are documented as **forward notes** (Stage-29-territory work)
rather than re-flagged as findings on every cycle.

**Forward note F-20-1** (added below): "PTX backend has three
width-aware tables (`_DTYPE_SIZE`, `_DTYPE_PTX_LOAD`,
`_ptx_type_str`) that lack isize/usize keys. Currently dormant —
guarded by lower_ast.py:511 NotImplementedError on HBM tile dtype.
When the f32/i32/f16/bf16 allowlist at lower_ast.py:511 is widened
to include isize/usize (or when device functions start emitting
real bodies, or when kernel scalar params start flowing through
`_ptx_type_str`), these three tables must be extended in lock-step
to preserve the canon."

This is **not a cycle-20 finding** because no production-reachable
path can hit it at HEAD. Marking as forward note keeps the
defect-class visible without inflating the strict-criterion
finding count (consistent with how cycle 17's PTX/dyn-ELF
parallel-backend carve-out was handled).

### Site 2: tile_ir element-width handling

`grep -n 'isize\|usize\|width\|bits' helixc/ir/tile_ir.py` returns
no matches except a docstring at line 8 ("explicit tile sizes
16x16, 64x64") — not a type-width table.

`grep -n 'i32\|i64\|f32\|f64' helixc/ir/tile_ir.py` returns no
matches either.

**Conclusion**: tile_ir has **no width-aware table**. It's a
structure-only IR — every type passes through unchanged via
`_map_value` (tile_ir.py:185-195) and `attrs` is copied by
`dict(op.attrs)` (tile_ir.py:223). Width semantics are delegated
to the downstream PTX backend (Site 1) and the x86_64 backend
(already fixed in cycle 19). No new finding here.

### Site 3: lower_ast literal-type inference (lower_ast.py:357-371)

```python
_PRIMITIVE_TYPE_NAMES = frozenset({
    "i8", "i16", "i32", "i64", "isize",
    "u8", "u16", "u32", "u64", "usize",
    "bool", "char",
    "bf16", "f16", "f32", "f64",
    "unit",
})

def _lower_type(self, ty: A.TyNode) -> tir.TIRType:
    if isinstance(ty, A.TyName):
        # ... default i32-sized ABI for unknowns; documented HBS
        # limitation.
        return tir.TIRScalar(ty.name)
    ...
```

This is a **membership set + identity-preserving lowering**, not a
width table. `_PRIMITIVE_TYPE_NAMES` contains `isize` and `usize`,
and `_lower_type` returns `tir.TIRScalar(ty.name)` — the name
string is preserved through to TIR unchanged.

The **literal-type inference** the user named lives one level
down at `lower_ast.py:1037-1038`:

```python
if isinstance(expr, A.IntLit):
    return self.builder.const_int(expr.value, expr.type_suffix or "i32")
```

This defaults untyped integer literals (`42`) to `i32`. **isize is
never a literal default** — to bind `let x: isize = 42`, the
literal flows as `IntLit(value=42, type_suffix=None)` and the
typechecker widens it to isize via the let-binding ascription
(`typecheck.py:1823+` — "the literal's own type_suffix takes
precedence over [the contextual type]"). The lowering produces
`CONST_INT { value: 42, ty: TIRScalar("i32") }` then a subsequent
CAST_TIGHT to isize.

Could this path silently narrow an isize literal? Walked the flow:

1. Source: `let x: isize = 5_000_000_000;`. Lexer produces
   `IntLit(value=5_000_000_000, type_suffix=None)`.
2. Typecheck binds the literal to TyPrim("isize") (typecheck.py:
   1823+).
3. Lower_ast.py:1038 emits `CONST_INT { value: 5_000_000_000,
   ty: TIRScalar("i32") }` — **the literal's TIR type is i32**,
   not isize, because the `or "i32"` default ignores the
   contextual type ascription.
4. Subsequent CAST_TIGHT to isize widens the IR type, but the
   CONST_INT's value attribute is **already** a Python int with
   full 64-bit precision. The backend's CONST_INT emit reads
   `op.attrs["value"]` directly (x86_64.py:1148-1153), not the
   TIR result type, so the 64-bit value survives.

The post-cycle-19 backend classifier (`_is_i64_type`) is consulted
on the **CAST_TIGHT** result type (now isize after the cast), and
the 64-bit emit path fires for both the CONST_INT store and the
CAST_TIGHT. No silent narrow.

**Could there be a different miscompile path?** The cycle-19 fix
specifically covered the CONST_INT case where the result type is
isize directly (e.g., from `monomorphize`-produced int literals
with explicit isize ascription via `_widen_canon_name`). Those
do flow as `CONST_INT { ty: TIRScalar("isize") }` and now hit the
64-bit emit path via the fixed `_is_i64_type`.

**Conclusion**: `lower_ast.py:357-371` is not a width table.
The literal-type-default at line 1038 is `i32` (not `isize`),
which is correct phase-0 behavior — typecheck handles the
contextual isize binding and the cast preserves the value. No
silent-failure window here. No finding.

### Summary of the adversarial rotation

| Site                                       | Has isize/usize keys? | Default for missing keys | Reachable at HEAD? | Finding?   |
|--------------------------------------------|-----------------------|--------------------------|--------------------|------------|
| `const_fold._INT_BITS`                     | YES (post-fix)        | 32 (i.e., i32)           | n/a (fixed)        | **Closed** |
| `x86_64._is_i64_type` / `_is_u64_type`     | YES (post-cycle-19)   | 32-bit emit branch       | n/a (fixed)        | **Closed** |
| `typecheck._WIDEN_RANK` (231-241)          | YES                   | KeyError on access       | n/a (correct)      | None       |
| `typecheck._WIDEN_NAME_ALIASES` (225-228)  | YES                   | identity                 | n/a (correct)      | None       |
| `typecheck` value-range tables (1816-1817) | YES                   | None                     | n/a (correct)      | None       |
| `lower_ast._PRIMITIVE_TYPE_NAMES`          | YES (membership only) | n/a (not a width table)  | n/a                | None       |
| `lower_ast` literal-type-default (1038)    | n/a (literal default) | "i32"                    | correct            | None       |
| `tile_ir` (no width table)                 | n/a                   | n/a                      | n/a                | None       |
| `ptx._DTYPE_SIZE`                          | **NO**                | 4 (32-bit)               | **NO** — gated     | Forward F-20-1 |
| `ptx._DTYPE_PTX_LOAD`                      | **NO**                | "u32" (32-bit)           | **NO** — gated     | Forward F-20-1 |
| `ptx._ptx_type_str`                        | **NO**                | ".b32" (32-bit)          | **NO** — gated     | Forward F-20-1 |

Every width-aware table that the cycle-19 canon ought to touch
**is now consistent**, except for three PTX-backend tables that
lack isize/usize keys but are unreachable from isize/usize values
at HEAD (guarded by the f32/i32/f16/bf16 allowlist in
`lower_ast.py:511`, the `.b64` hard-code in `_format_param`, and
the stub `ret;`-only body in `emit_device_func`).

Per the cycle-17 audit-rotation convention (dormant defects in
unreachable backend paths are forward notes, not findings), the
three PTX gaps are documented as forward note F-20-1.

---

## Findings

**None.**

| Severity   | Count |
|------------|-------|
| CRITICAL   | 0     |
| HIGH       | 0     |
| MEDIUM     | 0     |
| LOW        | 0     |
| **Total**  | **0** |

---

## Forward notes (not findings)

**F-20-1 / Stage-29 territory** — PTX backend's three width-aware
tables (`_DTYPE_SIZE` at ptx.py:327, `_DTYPE_PTX_LOAD` at ptx.py:
329, `_ptx_type_str` at ptx.py:159) lack `isize`/`usize` keys.
Currently dormant — the f32/i32/f16/bf16 allowlist at
`lower_ast.py:511` raises `NotImplementedError` before any
isize/usize-typed value can reach the PTX backend, and the
non-kernel device-func path emits a stub `ret;` body. When any
one of the following lands, these three tables must be extended
in lock-step:

1. The HBM-tile-dtype allowlist at `lower_ast.py:511` is widened
   to include isize/usize.
2. `emit_device_func` is upgraded to emit real bodies (Stage 17+).
3. Kernel scalar params start flowing through `_ptx_type_str`
   instead of the `.b64` hard-code at `ptx.py:155`.

This is the same dormant-parallel-backend defect class as
cycle-16/17's array-element-trap carve-out (LOAD_ELEM/STORE_ELEM
were fixed in x86_64 but PTX and dyn-ELF didn't yet emit them,
so no silent-failure window opened until they would). Per the
strict re-flag rule, forward notes are not re-cited every cycle;
this entry is recorded once and referenced from the cycle ledger.

**F-20-2 / Stage-29 territory** — Test-coverage gap: the cycle-20
regression test (`test_c19_1_isize_usize_are_64_bit_in_wrap`) is
**contract-level only** (asserts `_wrap_int_to_type` agreement
between isize↔i64 and usize↔u64). The cycle-19 audit doc
(audit-stage28-8-cycle19-silent-failures.md:466-472) recommended a
**full-pipeline** regression test that round-trips
`3_000_000_000_isize + 3_000_000_000_isize` through compile →
fold → emit → run and asserts the runtime answer equals
`6_000_000_000`. The cycle-20 fix-sweep landed only the
contract-level test.

This is **not a current silent-failure finding** — the contract
test does pin the specific defect class (any future drift of
`_INT_BITS["isize"]` away from 64 fails the test). The full-
pipeline test would be a stronger guarantee against a **new**
width table introduced further downstream silently narrowing
isize values. Recommended for the Stage-29 "centralize the
scalar-width predicate" refactor — the refactor itself should
ship with the full-pipeline integration test as a deliverable.

Recording as a forward note rather than escalating to a finding
because the strict-criterion question is "are silent failures
currently reachable?", and no path is currently open. The cycle-
19 forward note about centralizing the predicate (originally
cycle-17 forward note 1, restated cycle-18, restated cycle-19) is
also still open as the canonical Stage-29-territory work item.

---

## Cross-lens corroboration

The parallel **cycle-20 type-design lens** (Audit B) will be run
separately. Cycle-19 already produced cross-lens corroboration of
C19-1 between the silent-failure and type-design lenses — they
converged on the same defect through independent methodology.
Cycle-20's two lenses are expected to **diverge** (i.e., both
clean) since C19-1 is closed and no obvious analog remains.

The cycle-19 silent-failure doc's "adversarial rotation" enumerated
**every** parallel width-aware table in `helixc/`; the cycle-20
prompt explicitly named the three the user wanted re-verified
(ptx.py:328-332, tile_ir, lower_ast.py:357-371). All three checked
out: one with a hard NotImplementedError gate (PTX, forward note
F-20-1), one with no width table at all (tile_ir), one as a
membership-set-not-width-table (lower_ast.py:357-371). The
cycle-19 forward note "centralize the scalar-width predicate"
remains the standing recommendation for the eventual Stage-29
refactor.

---

## Verdict

**Cycle 20 silent-failures audit: CLEAN.**

| Severity   | Count |
|------------|-------|
| CRITICAL   | 0     |
| HIGH       | 0     |
| MEDIUM     | 0     |
| LOW        | 0     |
| **Total**  | **0** |

- C19-1 closed at HEAD=5a1e406; verification confirms the table
  extension is correct and the regression test pins the contract.
- Adversarial rotation found three dormant PTX-backend tables
  missing isize/usize keys, but all three are gated by a
  NotImplementedError at lower_ast.py:511 (or by other Phase-0
  hard-codes), so no production-reachable silent-failure window
  exists. Documented as forward note F-20-1.
- tile_ir has no width table; lower_ast.py:357-371 is a
  membership set, not a width table — both fall outside the
  defect class.

**Clean-cycle counter**: was 0/5 → **advances to 1/5** (cycle 20
is the first clean cycle of the new strict-criterion streak).

Four more consecutive clean cycles required to fire the Stage-29
gate. Any new finding (CRITICAL/HIGH/MEDIUM/LOW) before then will
reset the counter to 0/5 again.

---

## Files touched by this audit

None — this is a read-only audit cycle. Forward notes F-20-1 and
F-20-2 are recorded in this doc only; no production-code or test
edits.

## Cross-reference

- Cycle-19 silent-failures (surfaced C19-1):
  `docs/audit-stage28-8-cycle19-silent-failures.md`
- Cycle-19 type-design (parallel surfacing of C19-1):
  `docs/audit-stage28-8-cycle19-type-design.md`
- Cycle-20 fix-sweep commit (closed C19-1): `5a1e406`
- Files touched by cycle-20 fix-sweep:
  `helixc/ir/passes/const_fold.py:43-56`,
  `helixc/tests/test_const_fold.py:356-383`
- Adversarially-verified width-aware tables (all consistent with
  the canon or gated unreachable):
  `helixc/backend/ptx.py:159-165, 327-332`,
  `helixc/ir/tile_ir.py`,
  `helixc/ir/lower_ast.py:357-371, 1037-1038`,
  `helixc/frontend/typecheck.py:225-241, 1816-1817`.
