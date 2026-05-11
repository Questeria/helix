# Stage 28.8 Pre-29 Audit Gate — Cycle 18, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit**: 0243d5c (HEAD; production code byte-identical to c6136d4
and to cycle-17 baseline as cross-checked below).
**Cycle-17 baseline**: c6136d4 (production code at cycle-17 audit
HEAD).
**Cycle-17 status**: FULLY CLEAN under the strict criterion.
Counter advanced 0/5 → 1/5.

**Scope**: Audit category B (type-system / dispatch / soundness)
under the strict criterion. Cycle-17 forward notes called out two
candidates to promote at confidence ≥ 80:

1. (Cycle-17 forward note 1) `isize`/`usize` width inconsistency
   between `const_fold.py` (32-bit), `typecheck.py` (i64/u64
   aliases, 64-bit), and the C16-1 fix's array-elem trap (treats
   wide).
2. (Cycle-17 forward note 2) 4+ sites independently enumerate
   wide widths; no central scalar-width predicate.

Per the user directive, this audit also considered rotating to
the PTX `TileOpKind` type contracts.

**Counter context** (per user directive 2026-05-10):

- Cycle 16: HIGH finding (C16-1) — reset to 0/5.
- Cycle 17: FULLY CLEAN. Counter 0/5 → 1/5.
- Cycle 18 (this audit): if CLEAN under the strict criterion,
  conditional on other cycle-18 categories also being CLEAN, the
  counter advances 1/5 → 2/5. **A finding here resets the counter
  to 0/5.**

---

## Cycle-18 production-code delta (since cycle-17 baseline c6136d4)

```
git diff c6136d4..HEAD --stat -- helixc/
```

Returns empty. `0243d5c` (Phase-A staging-refinement commit between
c6136d4 and HEAD) is docs-only and does not change `helixc/`.

The entire `helixc/` tree is therefore byte-identical to the
cycle-17 audited state. **There is no cycle-18 fix-sweep to
audit.** The audit at this cycle is pure fresh-eyes re-read of the
type-design surface against the open cycle-17 forward notes plus
a rotated PTX lens.

Cross-checks:

```
git diff c6136d4..HEAD -- helixc/frontend/   (empty)
git diff c6136d4..HEAD -- helixc/check.py    (empty)
git diff c6136d4..HEAD -- helixc/ir/         (empty)
git diff c6136d4..HEAD -- helixc/backend/    (empty)
```

---

## Promotion analysis: cycle-17 forward note 1 (`isize`/`usize`)

The cycle-17 doc recorded forward note 1 as follows
(audit-stage28-8-cycle17-type-design.md:528-538):

> `helixc/ir/passes/const_fold.py:46` treats `isize`/`usize` as
> 32-bit (`"isize": 32, "usize": 32`), while
> `helixc/frontend/typecheck.py:226-227` aliases them to `i64` /
> `u64` (64-bit). The cycle-17 C16-1 fix's
> `_check_array_elem_size_supported` takes the typecheck.py
> position (treats them as wide). This is the safer side of the
> inconsistency — a false-positive trap is strictly preferable to
> a silent miscompile — but the underlying width-of-isize
> contradiction remains a Stage-29-class concern. Not blocking;
> not introduced by the cycle-17 fix.

The cycle-17 framing is correct **for the C16-1 helper** (which
treats isize/usize as wide and therefore traps conservatively —
that helper's side of the disagreement cannot produce a
miscompile). But the cycle-17 doc did not trace the OTHER side
of the contradiction: the x86_64 backend's general (non-array)
codegen path, which treats `TIRScalar("isize")` as a NON-i64 type
end-to-end. Tracing that side reveals a separate, currently
reachable, silent miscompile that is not gated by the C16-1
helper.

### Trace

#### Site 1: `_is_i64_type` / `_is_u64_type` are name-string-exact

`helixc/backend/x86_64.py:1005-1010`:

```python
def _is_i64_type(self, ty: tir.TIRType) -> bool:
    return isinstance(ty, tir.TIRScalar) and ty.name == "i64"

def _is_u64_type(self, ty: tir.TIRType) -> bool:
    # Stage 16.5: u64 is the IR type for raw pointers and FFI-arg widening.
    return isinstance(ty, tir.TIRScalar) and ty.name == "u64"
```

Neither predicate canonicalizes the pointer-width aliases. For
`TIRScalar("isize")`, `_is_i64_type` returns False; for
`TIRScalar("usize")`, `_is_u64_type` returns False.

`grep "_is_i64_type\|_is_u64_type" helixc/backend/x86_64.py` lists
~30 dispatch sites across `_emit_op`. Every one of those branches
sends `isize`/`usize` down the 32-bit narrow path because the
predicate misses the alias.

#### Site 2: CONST_INT emit silently truncates >32-bit literal

`helixc/backend/x86_64.py:1138-1147`:

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

For a literal lowered with result type `TIRScalar("isize")`, the
`_is_i64_type` check returns False, so the emitter takes the else
branch and **silently truncates the value to its low 32 bits via
`value & 0xFFFFFFFF`** — no diagnostic, no trap, the high 32 bits
disappear.

Concrete reproducer (typechecks today):

```helix
fn main() -> i32 {
    let x: isize = 5_000_000_000;  // 0x12A05F200
    if x > 4_000_000_000 { 1 } else { 0 }
}
```

Typecheck:
- `_INT_BOUNDS["isize"] = (-(1 << 63), (1 << 63) - 1)` per
  typecheck.py:1816 — 5_000_000_000 fits.
- Literal-fits check passes.
- Widening rank: `isize` is rank 40 = same as `i64` — no warning.

Lowering:
- AST `TyName("isize")` becomes `TIRScalar("isize")` per
  `lower_ast.py:357,371`.
- CONST_INT op carries `attrs["value"] = 5_000_000_000`, result
  type `TIRScalar("isize")`.

Backend (x86_64.py:1141):
- `_is_i64_type(TIRScalar("isize"))` returns False.
- Falls through to `mov_eax_imm32(5_000_000_000 & 0xFFFFFFFF) =
  mov_eax_imm32(705_032_704)`.
- The comparison `x > 4_000_000_000` is then computed against the
  truncated value 705_032_704, not the original. Runtime answer:
  0 instead of the program's intended 1.

No diagnostic at any pass boundary. Silent miscompile.

#### Site 3: FN-param spill silently truncates 64-bit isize argument

`helixc/backend/x86_64.py:971-989`:

```python
for p in self.fn.params:
    slot = self._slot_of(p)
    if self._is_float_type(p.ty):
        ...
    else:
        if int_idx >= len(INT_SPILLS):
            raise NotImplementedError(...)
        if self._is_i64_type(p.ty):
            INT_SPILLS_64[int_idx](slot)   # 64-bit mov [rbp-N], rdi
        else:
            INT_SPILLS[int_idx](slot)      # 32-bit mov [rbp-N], edi
        int_idx += 1
```

For a function declared `fn f(x: isize) -> isize { x + 1 }`:
- Param `p.ty = TIRScalar("isize")`.
- `_is_i64_type(p.ty)` returns False.
- Spill uses `INT_SPILLS[int_idx]` — 32-bit `mov [rbp-N], edi`.
- The top 32 bits of `rdi` (where the SysV ABI placed the 64-bit
  caller-side argument) are silently dropped.

A caller passing `5_000_000_000` produces the same truncation as
site 2.

#### Site 4: Operand-width dispatch in arithmetic

The 30+ `_is_i64_type(op.operands[0].ty)` and `_is_i64_type(op.
results[0].ty)` checks across `_emit_op` (ADD, SUB, MUL, CMP,
RET, CALL, etc.) all return False for `TIRScalar("isize")`,
sending each op down the 32-bit narrow path even though the
typechecker promised 64-bit semantics.

The CONST_INT + spill miscompile at sites 2-3 is enough on its
own; the operand-width sites compound it.

#### Site 5: const_fold's 32-bit wrap is internally consistent with the backend but contradicts typecheck

`helixc/ir/passes/const_fold.py:46`:

```python
_INT_BITS = {
    ...
    "i32": 32, "u32": 32, "isize": 32, "usize": 32,
    "i64": 64, "u64": 64,
    ...
}
```

`_wrap_int_to_type(v, TIRScalar("isize"))` masks `v` to 32 bits
and sign-extends from bit 31. This is consistent with the backend
sites 2-4 (which also treat isize as 32-bit). The cycle-17 doc
called this an inconsistency between const_fold and typecheck;
the deeper finding is that **const_fold is consistent with the
backend, and the backend's classifiers are the actual locus of
the contradiction with typecheck.** Fixing const_fold alone would
not close the silent-miscompile path through sites 2-3, and might
even widen the breach.

### Why this is reachable now (not Stage-29-class)

- `isize` / `usize` are primitive names per `lower_ast.py:357-358`
  — they flow into IR today; this is not a future feature.
- No typecheck error rejects `let x: isize = ...` or `fn f(x:
  isize)`. Tests use `isize` in autodiff D<isize> contexts already
  (`test_typecheck.py:1378-1385`).
- No frontend pass canonicalizes `isize`/`usize` into `i64`/`u64`
  before reaching the backend. (typecheck's `_widen_canon_name`
  at typecheck.py:231 is used only for widening-rank-tie
  resolution; it does not rewrite the underlying TIR scalar name.)
- The C16-1 helper (`_check_array_elem_size_supported`) gates
  only array element types via LOAD_ELEM / STORE_ELEM ops — it
  does NOT gate scalar `let x: isize = ...` (CONST_INT) or
  function-param spill paths. Sites 2-3 above are not in the
  C16-1 blast radius.
- The TIRScalar discriminator in the rest of the backend is the
  raw `.name` string. There is no abstraction layer that
  canonicalizes isize → i64 before reaching `_is_i64_type`.

The bug is reachable in straight-line user code today; it is a
silent miscompile (no diagnostic, valid-looking ELF emitted); the
result depends on whether the literal's high bits matter to the
program's behavior. Memory access in particular will be wrong
when isize is used as an offset (e.g., pointer arithmetic on a
large buffer).

### Confidence

- Trace is mechanical: read each of the five sites in
  `helixc/backend/x86_64.py` against an explicit input
  `TIRScalar("isize")`.
- Discriminator semantics verified by reading the source of
  `_is_i64_type` / `_is_u64_type` directly (single-line
  name-string equality).
- Reproducer is straight-line Helix source with no exotic
  features (`let x: isize = 5_000_000_000;` plus a comparison).
- Typecheck-permissiveness of the reproducer verified by reading
  `_INT_BOUNDS["isize"]` (typecheck.py:1816) and the widening
  table (typecheck.py:241).
- Lowering preservation of the isize name verified by reading
  `lower_ast.py:357-371`.

**Confidence ≥ 95%** that this is a real, reachable silent
miscompile. Promotes per the user directive (threshold ≥ 80%).

### Severity assessment

| Criterion | Assessment |
|---|---|
| Reachable today | Yes — straight-line user code |
| Diagnostic surface | None — silent |
| Output corruption | Wrong arithmetic, wrong control flow when literal or param exceeds 31 bits |
| Affects pointer arithmetic | Yes — usize is the index type for pointer math |
| Detectable by tests | Yes — a smoke test with `let x: isize = 5_000_000_000` and a runtime asserted comparison would catch it |
| Existing test coverage | None for codegen (only autodiff/widening tests use isize) |
| Type-system contract violated | "isize/usize are 64-bit aliases on 64-bit targets" (typecheck.py:212-216 comment) |
| Comparable to C13-1 / C16-1 | Yes — same defect class: backend silently narrows a wide type when the discriminator misses |

Severity: **HIGH**. Same class as C13-1 (DCE-drops-operand silent
miscompile) and C16-1 (wide-array-elem silent truncate). Output
of compiled code is wrong with no diagnostic, on input the
typechecker accepts.

---

## Promotion analysis: cycle-17 forward note 2 (centralize scalar-width predicate)

Forward note 2 calls for refactoring the 4+ wide-width
enumeration sites into a single `_scalar_width_bits(ty) -> int`
predicate.

If finding C18-1 (above) lands as a fix, the natural shape of
the fix is to make `_is_i64_type` / `_is_u64_type` canonicalize
`isize`/`usize` (or equivalently, to introduce a single
`_scalar_width_bits` predicate and drive both classifiers off
it). That refactor would close forward note 2 as a side-effect.
Forward note 2 should therefore be treated as a SHAPE
recommendation for the C18-1 fix, not a separate finding.

Forward note 2 alone (with all sites currently agreeing on the
narrow path for isize/usize) is a maintenance concern but does
not produce a miscompile. Confidence < 80% as an independent
finding. **Not promoted as a standalone cycle-18 finding.**
Carried as forward note (see below).

---

## Rotated lens: PTX backend `TileOpKind` type contracts

The user directive asked whether the PTX backend's TileOpKind
type contracts should be examined. Cycle 17 already audited the
PTX op-kind dispatch surface (cycle-17 doc, "Rotated lens"
section) and confirmed no LOAD_ELEM/STORE_ELEM analogue exists
on the tile IR. Re-reading at HEAD:

### TileOpKind enum closure

`helixc/ir/tile_ir.py:55-99` defines the full enum. Every variant
falls into one of four families:

| Family | Variants |
|---|---|
| Tile creation | TILE_ZEROS, TILE_CONST |
| Memory movement | TILE_LOAD_GLOBAL/SHARED, TILE_STORE_GLOBAL/SHARED, TMA_LOAD/STORE, BARRIER_WAIT |
| Compute | TILE_ADD/SUB/MUL/MATMUL/REDUCE, TILE_TRANSPOSE/RESHAPE |
| Scalar passthrough | SCALAR_CONST_INT/FLOAT, SCALAR_ADD/SUB/MUL/NEG/CMP/SELECT, CALL, RETURN, THREAD_IDX, TILE_INDEX_LOAD_HBM, TILE_INDEX_STORE_HBM |

The PTX backend dispatches on 11 of these in `helixc/backend/
ptx.py` (lines 173, 180, 195, 207, 219, 230, 241, 255, 258, 270,
302), matching the cycle-17 inventory.

### Type-contract surface

The PTX path's only width-dispatching site is `_ptx_load_suffix`
/ `_ld_reg_prefix` / `_ptx_store_suffix` (lines 327-346), driven
by `op.attrs["dtype"]` on `TILE_INDEX_LOAD_HBM` /
`TILE_INDEX_STORE_HBM`. The map at line 328 is:

```python
{"i32": 4, "u32": 4, "f32": 4, "i64": 8, "u64": 8, "f64": 8}
```

The map at line 332 is:

```python
{"i64": "s64", "u64": "u64", "f64": "f64", ...}
```

`isize` and `usize` are **not in either map**. If a TIL
TILE_INDEX_LOAD_HBM op is emitted with dtype attr "isize" /
"usize", the lookup falls through to whatever the default
behavior is.

Reading lines 327-346:

```python
def _ptx_load_suffix(self, dtype: str) -> str:
    sizes = {"i32": 4, "u32": 4, "f32": 4,
             "i64": 8, "u64": 8, "f64": 8}
    ...
```

Whether this site is reachable for isize/usize depends on the
tile-IR lowering path's behavior on isize-element tensors.
Reading `lower_ast.py:357-358` confirms isize/usize are valid
primitive scalar names that can be tensor element types in
principle. Reading the TIR → tile-IR lowerer in
`tir_to_tile_ir.py` (if present) to verify whether isize/usize
can land in `dtype` attr:

```
grep "dtype" helixc/ir/tile_ir.py
```

shows the `dtype` attribute is propagated through `TILE_INDEX_
LOAD_HBM` / `TILE_INDEX_STORE_HBM` from the lowering pass; the
attribute is the raw scalar name string. For an isize-element
tensor (today permitted by the type system as a tensor dtype per
typecheck.py:2091-2092), the dtype attribute would be `"isize"`,
and `_ptx_load_suffix` would KeyError at PTX emit.

This is a separate, narrower failure mode from sites 2-3 of
C18-1: it raises a KeyError rather than silently miscompiling.
**Loud, not silent.** Cycle-18 strict-criterion-promotable? Only
if the loud failure is wrong for a path the typechecker accepts.
The cycle-3 / cycle-16 "narrow + loud" pattern is the established
Phase-0 convention; a KeyError at the backend on a typecheck-
accepted input is the same shape as `_check_float_supported`
trapping on `f16`/`bf16`. Not a defect by Phase-0 conventions.

PTX rotation: **no findings**.

---

## Findings summary

| ID | Severity | Confidence | Location | Description |
|---|---|---|---|---|
| C18-1 | HIGH | ≥95% | `helixc/backend/x86_64.py:1005-1010` (`_is_i64_type` / `_is_u64_type`) | Backend classifiers miss `isize`/`usize` aliases (return False on `TIRScalar("isize")`). Cascades through 30+ dispatch sites in `_emit_op`. Reachable miscompile via CONST_INT emit (`mov_eax_imm32(value & 0xFFFFFFFF)`, x86_64.py:1145) silently truncating literals > 31 bits, and via FN-param spill (x86_64.py:982-989) silently dropping the high 32 bits of a 64-bit isize argument. Typecheck accepts the reproducer (`let x: isize = 5_000_000_000;`) because `_INT_BOUNDS["isize"]` is the i64 range (typecheck.py:1816). No diagnostic anywhere; silent wrong-answer output. |

**Total**: 1 HIGH, 0 MEDIUM, 0 LOW.

---

## Recommended fix (for cycle-19 fix-sweep)

Match the cycle-3-style "narrow + loud" pattern, two parallel
options ordered by Phase-0 conservatism:

### Option A (most conservative, recommended): trap at codegen on isize/usize, matching the C16-1 shape

Add a helper `_check_pointer_width_alias_unsupported(self, ty)`
analogous to `_check_array_elem_size_supported`, called from
the same SSA-value-allocation paths as `_check_float_supported`
(x86_64.py:927, 931, 935). Reject `TIRScalar("isize")` and
`TIRScalar("usize")` with a `NotImplementedError` containing
"isize/usize 64-bit aliases not yet supported by x86_64
backend; use i64/u64 explicitly until the alias canonicalization
path lands. Audit 28.8 cycle 18 C18-1." This is the same shape
as C16-1's helper and provides a loud trap on the existing
silent-miscompile path. Frontend code that needs pointer-width
behavior can use `i64`/`u64` explicitly.

### Option B (canonicalize at backend boundary)

Modify `_is_i64_type` and `_is_u64_type` to recognize the
pointer-width aliases:

```python
def _is_i64_type(self, ty: tir.TIRType) -> bool:
    return isinstance(ty, tir.TIRScalar) and ty.name in ("i64", "isize")

def _is_u64_type(self, ty: tir.TIRType) -> bool:
    return isinstance(ty, tir.TIRScalar) and ty.name in ("u64", "usize")
```

Plus matching adjustments to `_INT_BITS` in const_fold.py:46 to
treat isize/usize as 64-bit. This actually compiles isize/usize
correctly (instead of trapping) but is a larger blast radius —
every `_is_i64_type` call site (30+) is implicitly affected.
Should be paired with a regression test that round-trips
`let x: isize = 5_000_000_000` through the full pipeline and
asserts the runtime value matches.

### Recommended approach

**Option A first** (cycle-19), as the smallest, most
conservative landed fix that closes the silent-miscompile path
in a single localized change. **Option B as a Stage-29
deliverable** when the wider isize-as-i64 codegen path is fully
exercised by integration tests. This mirrors the cycle-16 / C16-1
"narrow + loud first; full lowering as Stage-29 deliverable"
pattern.

### Regression test sketch (for Option A)

```python
def test_c18_1_isize_traps_at_codegen():
    """Audit 28.8 cycle 18 C18-1: isize/usize must not silently
    compile through the 32-bit narrow path. typecheck accepts
    them as 64-bit aliases; the backend has no 64-bit dispatch
    path for them today, so a trap is correct."""
    src = """
    fn main() -> i32 {
        let x: isize = 5000000000;
        if x > 4000000000 { 1 } else { 0 }
    }
    """
    prog = parse_src(src)
    errs = type_check(prog)
    hard = [e for e in errs if not (hasattr(e, "is_warning") and e.is_warning)]
    assert not hard, f"reproducer should typecheck cleanly; got: {hard}"
    mod = lower(prog)
    try:
        compile_module_to_elf(mod)
        assert False, (
            "expected NotImplementedError on isize CONST_INT; "
            "backend silently truncated to 32 bits instead"
        )
    except NotImplementedError as e:
        assert "C18-1" in str(e) or "isize" in str(e), (
            f"expected C18-1 trap message, got: {e}"
        )
```

---

## Cycle 18 status

**Strict criterion (per user directive 2026-05-10): cycle clean
iff zero findings of ANY severity at confidence ≥ 80.**

This cycle finds **1 HIGH finding (C18-1)** at confidence ≥ 95%
under the type-design audit category.

By the strict criterion, **cycle 18's type-design audit is NOT
CLEAN**.

**Counter status (5-clean-consecutive gate under the strict
criterion)**:

- Was 1/5 after cycle 17 CLEAN.
- Cycle 18 type-design: NOT CLEAN. **Counter resets to 0/5.**
- Stage 29 is gated by five fresh consecutive clean cycles. Five
  more clean cycles required after the cycle-19 fix-sweep
  closes C18-1.

The severity trend across cycles, against the strict-criterion
bar:

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
- Cycle 16: 1 HIGH (C16-1) — not clean → reset to 0/5
- Cycle 17: 0 + 0 + 0 — clean → 1/5
- Cycle 18 (type-design): 1 HIGH (C18-1) — not clean → reset to 0/5

---

## Forward notes (not cycle-18 findings; recorded for visibility)

1. **Centralize scalar-width predicate** (CARRY from cycle-17
   forward note 2): the C18-1 fix-sweep (Option A or B) is the
   natural moment to factor a single
   `_scalar_width_bits(ty) -> int` predicate driving
   `_is_i64_type`, `_is_u64_type`, `_is_f64_type`,
   `_is_float_type`, `_check_array_elem_size_supported`, and
   const_fold's `_INT_BITS`. The cycle-19 fix-sweep should at
   minimum align the C18-1 fix's wide-set with the C16-1
   helper's wide-set; the full predicate factoring can land as
   a Stage-29-class refactor.

2. **PTX dtype-suffix map alias gap** (NEW, cycle 18):
   `_ptx_load_suffix` / `_ptx_store_suffix` /
   `_ld_reg_prefix` at `helixc/backend/ptx.py:327-346` lack
   `isize`/`usize` entries. Today this would manifest as a
   KeyError (loud, not silent) when an isize-element tensor
   reaches the tile-IR PTX emit path. Not currently reachable in
   integration tests but a structurally-related gap to C18-1.
   Stage-29-class once tensor-of-isize is exercised; not blocking.

3. **Operand-index addressing in TIR op handlers** (CARRY from
   cycle-17 forward note 3): `STORE_ELEM`'s value operand is
   addressed as `op.operands[1].ty` (positional). Named-operand
   accessors at TIR level would close this fragility.
   Stage-29-class.

4. **Dead `hard` local in C16-1 regression test** (CARRY from
   cycle-17 forward note 4): `test_codegen.py:457`'s
   `hard = [...]` is computed but never asserted on. Stylistic.

5. **Missing `i64`-array trap regression test** (CARRY from
   cycle-17 forward note 5): cycle-16 doc named three regression
   tests; the cycle-17 fix-sweep implemented one (`f64`). Add
   the `i64`-only variant alongside the cycle-19 C18-1 fix-sweep
   for defense-in-depth against future narrowing of the
   `wide_widths` set.

6. **Stage-29 deliverable: full 8-byte LOAD_ELEM / STORE_ELEM
   lowering** (CARRY from cycle-17 forward note 6): once
   landed, `_check_array_elem_size_supported` becomes either
   dead code or a narrower guard.

7. **Forward-carry from cycle 16, note 8** (`_alloc_array`
   `elem_size` parameter unwired) (CARRY from cycle-17 forward
   note 7): IR-level `ALLOC_ARRAY` op's `dtype` attribute read
   but not propagated. Phase-0 safe under C16-1 trap.
   Stage-29-class.

8. **`Value.ty` not frozen** (CARRY from cycle-17 forward note
   8): `tir.Value` is `@dataclass` not `@dataclass(frozen=True)`.
   Stage-29-class hardening.

9. **`Op.results: list[Value]` over-general** (CARRY from
   cycle-17 forward note 9): single-result Op convention is
   convention-only. Stage-29-class.

10. **`SIDE_EFFECT_KINDS` static cross-check** (CARRY from
    cycle-14 forward note 5): no static guarantee that every
    side-effecting `OpKind` is in the set. Stage-29-class.

11. **Cycle-19 baseline**: cycle 18's audit is read-only at HEAD
    0243d5c. The cycle-19 fix-sweep will be auditing one production-
    code file (`helixc/backend/x86_64.py`) plus possibly
    `const_fold.py` if Option B is chosen. Cycle-19's audit-B can
    re-read the C18-1 fix and its regression test as the only
    delta, and otherwise rely on the empty-diff shortcut against
    HEAD 0243d5c.

12. **Stage-29 readiness**: counter resets from 1/5 to 0/5. Five
    fresh consecutive clean cycles remain required after the
    cycle-19 fix-sweep closes C18-1.
