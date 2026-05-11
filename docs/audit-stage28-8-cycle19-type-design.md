# Stage 28.8 Pre-29 Audit Gate — Cycle 19, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit**: 0803902 (HEAD; cycle-19 fix-sweep landing C18-1).
**Cycle-18 baseline**: 0243d5c (production code byte-identical to
c6136d4 from cycle 17; the cycle-18 audit found 1 HIGH and was
NOT CLEAN).
**Cycle-18 status**: NOT CLEAN — reset counter to 0/5.
**Cycle-19 fix-sweep**: closes C18-1 (HIGH) by extending
`_is_i64_type` / `_is_u64_type` to recognize the pointer-width
aliases `isize` / `usize`.

**Scope**: Audit category B (type-system / dispatch / soundness)
under the strict criterion. Per the user directive, this audit:

1. Verifies the C18-1 fix matches typecheck.py's canonicalization.
2. Checks whether a third type-classifier site for similar widths
   got missed — in particular `const_fold.py:46` which the cycle-17
   forward note flagged with a 32-bit assumption for isize.
3. Rotates lens to grad_pass / autodiff width handling.

**Counter context** (per user directive 2026-05-10):

- Cycle 17: FULLY CLEAN. Counter 0/5 → 1/5.
- Cycle 18: 1 HIGH (C18-1). Reset to 0/5.
- Cycle 19 (this audit): if CLEAN under the strict criterion,
  counter advances 0/5 → 1/5. **A finding here keeps the counter
  at 0/5.**

---

## Cycle-19 production-code delta (since cycle-18 baseline 0243d5c)

```
git show --stat 0803902 -- helixc/
```

shows:

```
 helixc/backend/x86_64.py        | 11 ++++++++++-
 helixc/tests/test_codegen.py    | 27 ++++++++++++++++++++++++++
```

The fix is a single 11-line edit to `_is_i64_type` and `_is_u64_type`
at `helixc/backend/x86_64.py:1005-1017`:

```python
def _is_i64_type(self, ty: tir.TIRType) -> bool:
    # Audit 28.8 cycle 19 C18-1 (HIGH): `isize` is a pointer-width
    # alias of `i64` on 64-bit targets …
    return isinstance(ty, tir.TIRScalar) and ty.name in ("i64", "isize")

def _is_u64_type(self, ty: tir.TIRType) -> bool:
    # Stage 16.5: u64 is the IR type for raw pointers and FFI-arg widening.
    # Audit 28.8 cycle 19 C18-1: `usize` is a pointer-width alias of
    # `u64` on 64-bit targets. Same silent-trunc class as isize.
    return isinstance(ty, tir.TIRScalar) and ty.name in ("u64", "usize")
```

Plus a regression test `test_c18_1_isize_usize_recognized_as_64bit`
at `helixc/tests/test_codegen.py:477-501`.

---

## Verification of the cycle-19 fix against typecheck.py's canon

**Pass.** The fix correctly canonicalizes the pointer-width
aliases to match `typecheck.py`'s established treatment:

- `typecheck.py:225-228` defines `_WIDEN_NAME_ALIASES = {"isize":
  "i64", "usize": "u64"}` and the cycle-3 C3-2 fix uses this alias
  table to canonicalize before tie resolution at the widening-rank
  level.
- `typecheck.py:241` ranks `isize` at 40 = `i64`'s rank and `usize`
  at 41 = `u64`'s rank.
- `typecheck.py:1816-1817` defines `_INT_BOUNDS["isize"] = i64
  range` and `_INT_BOUNDS["usize"] = u64 range`.

The cycle-19 fix extends `_is_i64_type` / `_is_u64_type` to recognize
the aliases in the same direction (isize→i64-class, usize→u64-class).
This matches the established Phase-0 canon.

**Confirmed downstream effect** of the fix on the cycle-18 reproducer
(`let x: isize = 5_000_000_000_isize;`):

- CONST_INT emit at x86_64.py:1148 — `_is_i64_type` now returns True
  → takes the `mov_rax_imm64(value)` path. Correct 64-bit literal
  no longer silently truncates.
- Fn-param spill at x86_64.py:971-989 — `_is_i64_type(p.ty)` now
  returns True → uses `INT_SPILLS_64`. Correct 64-bit spill.
- 30+ operand-width dispatch sites (ADD, SUB, MUL, CMP, RET, CALL,
  etc.) all now route isize/usize through the 64-bit emit paths.

The cycle-18 cited reproducer (`let x: isize = 5_000_000_000_isize;`
with comparison `x > 4_000_000_000`) is closed.

---

## Finding C19-1 (HIGH): const_fold.py `_INT_BITS` now contradicts the backend post-fix; silent 32-bit truncation of folded isize/usize arithmetic

### Location

`helixc/ir/passes/const_fold.py:43-49`:

```python
_INT_BITS = {
    "i8": 8, "u8": 8,
    "i16": 16, "u16": 16,
    "i32": 32, "u32": 32, "isize": 32, "usize": 32,
    "i64": 64, "u64": 64,
    "bool": 32,  # bool comparisons reified to i32 in IR
}
```

`_wrap_int_to_type(v, ty)` at const_fold.py:52-67 masks `v` to
`_INT_BITS.get(ty.name, 32)` bits and sign-extends from `(bits - 1)`.

This function is called from every const-fold result path that
produces an integer literal: ADD/SUB/MUL/DIV/MOD (const_fold.py:327),
BIT_AND/BIT_OR/BIT_XOR/SHL/SHR (const_fold.py:403), NEG
(const_fold.py:444), BIT_NOT (const_fold.py:475).

### What changed at cycle 19

**Pre-cycle-19** (cycle-18 doc, Site 5, lines 206-227): the
backend's classifiers treated isize/usize as 32-bit (silently wrong),
and const_fold's `_INT_BITS` also treated them as 32-bit. The
cycle-18 audit explicitly framed this as:

> const_fold is consistent with the backend, and the backend's
> classifiers are the actual locus of the contradiction with
> typecheck. **Fixing const_fold alone would not close the
> silent-miscompile path** through sites 2-3, and might even widen
> the breach.

The cycle-18 audit's Option B recommendation (lines 434-452)
explicitly stated:

> Modify `_is_i64_type` and `_is_u64_type` to recognize the
> pointer-width aliases … **Plus matching adjustments to
> `_INT_BITS` in const_fold.py:46 to treat isize/usize as 64-bit.**
> This actually compiles isize/usize correctly (instead of trapping)
> but is a larger blast radius … Should be paired with a regression
> test that round-trips `let x: isize = 5_000_000_000` through the
> full pipeline and asserts the runtime value matches.

The cycle-19 fix took Option B (canonicalize at backend boundary)
but **skipped the matching adjustment to `_INT_BITS`**. The
cycle-19 commit message and regression test do not mention
const_fold at all. The cycle-18 forward note 1 (line 542-547,
"Centralize scalar-width predicate") explicitly called for
const_fold's `_INT_BITS` to be aligned with the wide-set in the
C18-1 fix; this was not done.

### The new contradiction

**Post-cycle-19**:

- Backend (`x86_64.py:1011, 1017`): `isize` and `usize` are 64-bit.
  `_is_i64_type(TIRScalar("isize"))` returns True; CONST_INT for
  isize emits a 64-bit literal load.
- const_fold (`const_fold.py:46`): `_INT_BITS["isize"] = 32`,
  `_INT_BITS["usize"] = 32`. `_wrap_int_to_type(v, TIRScalar("isize"))`
  masks `v` to 32 bits and sign-extends from bit 31.

The two passes now have inconsistent width contracts. When
const_fold folds an arithmetic op whose result is type
`TIRScalar("isize")`, it wraps the result to 32 bits — but the
backend then emits the wrapped value as a 64-bit literal.

### Concrete reproducer (verified end-to-end against HEAD)

```helix
fn main() -> isize {
    let a: isize = 3000000000_isize;
    let b: isize = 3000000000_isize;
    a + b
}
```

Pipeline trace at HEAD = 0803902 (verified by running the actual
parser → typecheck → lower → fold_module):

1. **Typecheck**: passes. Literal `3000000000_isize` routes through
   `_check_int_lit_fits` with `eff_name = "isize"` (suffix takes
   precedence over contextual ty), yielding `_INT_BOUNDS["isize"] =
   i64 range`. 3e9 fits.
2. **Lower to TIR**: two `CONST_INT(value=3_000_000_000)` ops of
   type `TIRScalar("isize")`, one `ADD` op of type
   `TIRScalar("isize")`.
3. **const_fold (default `-O1` per `check.py:581`)**: folds the
   ADD on two CONST_INTs:
   - `l = 3_000_000_000`, `r = 3_000_000_000`, `v = l + r =
     6_000_000_000`.
   - `_wrap_int_to_type(6_000_000_000, TIRScalar("isize"))`:
     - `bits = _INT_BITS["isize"] = 32`.
     - `mask = 0xFFFFFFFF`.
     - `v = 6_000_000_000 & 0xFFFFFFFF = 1_705_032_704`.
     - `v < half (1<<31)` → no sign correction.
     - **Returns `1_705_032_704`.**
   - Rewrites the ADD into `CONST_INT(value=1_705_032_704)` of
     type isize.
4. **Backend post-cycle-19** at x86_64.py:1148:
   - `_is_i64_type(TIRScalar("isize"))` returns True.
   - Emits `mov_rax_imm64(1_705_032_704)`.
   - Stores 8 bytes at the result slot.
5. **Runtime**: `main()` returns `1_705_032_704` instead of the
   program's intended `6_000_000_000`.

Empirical confirmation:

```
$ python -c "
from helixc.ir.passes.const_fold import _wrap_int_to_type
from helixc.ir import tir
print('wrap(6e9, isize) =', _wrap_int_to_type(6_000_000_000, tir.TIRScalar(name='isize')))
print('wrap(6e9, i64)   =', _wrap_int_to_type(6_000_000_000, tir.TIRScalar(name='i64')))
"
wrap(6e9, isize) = 1705032704
wrap(6e9, i64)   = 6000000000
```

And the full pipeline:

```
$ python -c "
from helixc.frontend.parser import parse
from helixc.frontend.typecheck import typecheck
from helixc.ir.lower_ast import lower
from helixc.ir.passes.const_fold import fold_module
src = '''
fn main() -> isize {
    let a: isize = 3000000000_isize;
    let b: isize = 3000000000_isize;
    a + b
}
'''
prog = parse(src)
errs = typecheck(prog)
assert not [e for e in errs if not (hasattr(e, 'is_warning') and e.is_warning)]
mod = lower(prog)
fold_module(mod)
for fn in mod.functions.values():
    for blk in fn.blocks:
        for op in blk.ops:
            print(op.kind.name, op.attrs, 'res_ty=', [str(r.ty) for r in op.results])
"
CONST_INT {'value': 3000000000} res_ty= ["TIRScalar(name='isize')"]
CONST_INT {'value': 3000000000} res_ty= ["TIRScalar(name='isize')"]
CONST_INT {'value': 1705032704} res_ty= ["TIRScalar(name='isize')"]
RETURN {} res_ty= []
```

The folded CONST_INT carries `value=1_705_032_704` and result type
`isize`, which the cycle-19 backend now emits as a 64-bit literal.
Silent wrong-answer output. **The same end-to-end miscompile that
C18-1 closed for the trivially-literal case (`let x: isize = 5e9`)
is reopened by const_fold for the folded-arithmetic case
(`let x: isize = 3e9_isize + 3e9_isize`).**

### Reachability

- Const-fold is on by default at -O1 per `check.py:568-581`. The
  default optimization level for `helixc compile` is -O1.
- The reproducer is straight-line Helix source. Suffix-form literals
  (`_isize`) are routed through `_check_int_lit_fits` directly per
  typecheck.py:1820-1845. No exotic syntax.
- Every const_fold result path that calls `_wrap_int_to_type` on a
  result of type isize/usize is affected: ADD, SUB, MUL, DIV, MOD,
  BIT_AND, BIT_OR, BIT_XOR, SHL, SHR, NEG, BIT_NOT. ~12 op kinds.
- No diagnostic at any pass boundary. The folded literal looks
  bit-identical to a user-written 32-bit-fitting literal.

### Reachable BEFORE the cycle-19 fix?

Yes, but with a different visible effect:

- Pre-fix: const_fold wraps to 32 bits AND backend emits 32 bits.
  The folded result is consistent — `let x: isize = 3e9_isize +
  3e9_isize` returns 1_705_032_704 whether or not -O1 is on (the
  runtime ADD also wraps in eax). Wrong, but optimization-stable.
- Post-fix: const_fold wraps to 32 bits, backend emits 64 bits.
  The folded result `1_705_032_704` mismatches the un-folded result
  `6_000_000_000`. **Wrong AND optimization-unstable** — the program's
  behavior depends on whether -O1 fires.

The cycle-19 fix correctly closed the un-folded path. It
**inadvertently introduced an -O1-dependent miscompile** for the
folded path, where pre-fix there was just a consistently-wrong
narrow result.

This satisfies the "reachable today, not Stage-29-class" bar.

### Severity assessment

| Criterion | Assessment |
|---|---|
| Reachable today | Yes — straight-line user code at default -O1 |
| Diagnostic surface | None — silent |
| Output corruption | Wrong arithmetic on folded isize/usize literals when result has bits 32+ set |
| Affects pointer arithmetic | Yes — usize is the index type for pointer math; folded `base + offset` with large offset wraps |
| Optimization-stability | Broken — folded and un-folded paths disagree |
| Detectable by tests | Yes — round-trip a folded isize sum through compile + run and assert the value |
| Existing test coverage | None — `test_const_fold.py` has zero isize/usize references |
| Type-system contract violated | "isize/usize are 64-bit aliases on 64-bit targets" (typecheck.py:212-216 comment), now violated by const_fold only |
| Comparable to C13-1 / C16-1 / C18-1 | Yes — same defect class: silent narrowing when one pass's width contract disagrees with another's |

Severity: **HIGH**. Reachable silent miscompile; produces
optimization-unstable wrong-answer output on input the typechecker
accepts.

### Confidence

- Trace is mechanical: read `_INT_BITS` (const_fold.py:46), read
  `_wrap_int_to_type` (const_fold.py:52-67), read every
  `_wrap_int_to_type` callsite in const_fold.py (lines 327, 403,
  444, 475).
- The "post-fix backend now emits 64-bit" half is verified by
  reading the cycle-19 fix at x86_64.py:1011, 1017 and the CONST_INT
  emit at x86_64.py:1148-1153.
- The reproducer is **empirically verified** end-to-end against HEAD
  by running the actual pipeline (output transcript above).
- typecheck-permissiveness of the reproducer is **empirically
  verified** (no hard errors from `typecheck(prog)`).
- The cycle-18 audit doc (lines 434-452) explicitly flagged this as
  the Option B-extra-step that needs to land alongside the
  classifier fix. The cycle-19 commit did not land it.

**Confidence ≥ 95%** that this is a real, reachable silent
miscompile introduced by the cycle-19 fix-sweep itself. Promotes
per the user directive (threshold ≥ 80%).

---

## Lens rotation: grad_pass / autodiff width handling

The user directive asked to consider rotating to autodiff/grad_pass
width handling. Reviewed `helixc/frontend/autodiff.py:60-79`:

```python
NUMERIC_FOR_AD: frozenset[str] = frozenset({
    "i8", "i16", "i32", "i64", "isize",
    "u8", "u16", "u32", "u64", "usize",
    "f16", "bf16", "f32", "f64",
    "fp8", "mxfp4", "nvfp4",
    "bool", "char",
})
```

Autodiff treats `isize` and `usize` as members of the broad
numeric-for-AD set (parallel to the typecheck `PRIMITIVES` and
`_NUMERIC_SCALAR_NAMES` sets at typecheck.py:337-338 and 2091-2092).
There is no width-specific dispatch in autodiff — it does not
distinguish 32-bit from 64-bit integers for the purposes of cast
arms. The cycle-2 B:C9 fix at autodiff.py:60-79 is correct.

The widening logic that DOES distinguish widths lives in
typecheck.py:`_widen_diff_inner` and uses `_WIDEN_RANK` /
`_widen_canon_name`. These are the cycle-3 C3-2 sites; they
canonicalize isize→i64, usize→u64 consistently with the new cycle-19
backend behavior.

No findings in autodiff / grad_pass width handling.

---

## Cross-check: any third type-classifier site missed?

Searched `helixc/` for the pattern of width-keyed scalar dictionaries
and name-string discriminators (grep results above):

- `helixc/backend/x86_64.py:1011, 1017` — cycle-19 fix, now correct.
- `helixc/backend/x86_64.py:1042` — `_check_array_elem_size_supported`
  wide_widths set already includes isize/usize per cycle-17. Correct.
- `helixc/ir/passes/const_fold.py:46` — **`_INT_BITS` carries 32-bit
  isize/usize assumption**, finding C19-1 above.
- `helixc/frontend/typecheck.py:241, 1816-1817` — both isize/usize
  are at 64-bit rank and i64-range bounds. Correct.
- `helixc/frontend/autodiff.py:74-75` — broad numeric set, no width
  dispatch. Correct.
- `helixc/frontend/monomorphize.py:704` — comment only, no width
  dispatch.
- `helixc/ir/lower_ast.py:357-358` — primitive name preservation,
  no width logic.
- `helixc/backend/ptx.py:328-332` — cycle-18 forward note 2: no
  isize/usize entries in the dtype-suffix map. KeyError on isize-
  tensor dtype. Loud, not silent. Same Phase-0 narrow+loud pattern
  as `_check_float_supported`. Not a regression of C19-1's class.
  Carried as forward note.

**No fourth site**. The const_fold site is the only outstanding
type-classifier-with-isize/usize width contradiction at HEAD.

---

## Findings summary

| ID | Severity | Confidence | Location | Description |
|---|---|---|---|---|
| C19-1 | HIGH | ≥95% | `helixc/ir/passes/const_fold.py:46` (`_INT_BITS["isize"]=32`, `_INT_BITS["usize"]=32`) | const_fold's `_wrap_int_to_type` masks isize/usize folded-arithmetic results to 32 bits (`6_000_000_000 → 1_705_032_704`) while the cycle-19 backend now emits the result as a 64-bit literal. Silent miscompile of `let x: isize = 3e9_isize + 3e9_isize` at default -O1. Reachable today, no diagnostic, optimization-unstable wrong-answer output. Cycle-18 doc Option B explicitly named this as needing to land with the classifier fix; cycle-19 fix-sweep skipped it. Same defect class as C13-1 / C16-1 / C18-1. Empirically verified against HEAD by running the parser → typecheck → lower → fold_module pipeline on the reproducer. |

**Total**: 1 HIGH, 0 MEDIUM, 0 LOW.

---

## Recommended fix (for cycle-20 fix-sweep)

Align `const_fold._INT_BITS` with the cycle-19 backend canon:

```python
_INT_BITS = {
    "i8": 8, "u8": 8,
    "i16": 16, "u16": 16,
    "i32": 32, "u32": 32,
    "i64": 64, "u64": 64, "isize": 64, "usize": 64,  # cycle-19 C19-1
    "bool": 32,
}
```

This is a single-line edit. It does not change behavior for any
non-isize/non-usize folded result. It restores consistency among:

- typecheck.py's i64-range bounds for isize/usize.
- x86_64.py's `_is_i64_type` / `_is_u64_type` 64-bit treatment
  (post-cycle-19).
- const_fold.py's fold-result wrap width.

### Regression test sketch (full-pipeline round-trip)

The cycle-19 regression test
(`test_c18_1_isize_usize_recognized_as_64bit`) pins only the
classifier contract — it does NOT exercise the const_fold path that
introduced C19-1. The cycle-20 fix should add a full-pipeline test:

```python
def test_c19_1_isize_const_fold_preserves_64bit():
    """Audit 28.8 cycle 19 C19-1 (HIGH): const_fold's `_wrap_int_to_type`
    must wrap isize/usize folded results to 64 bits, matching the
    cycle-19 backend `_is_i64_type` canon. Pre-fix const_fold wrapped
    to 32 bits, silently truncating `let x: isize = 3e9_isize +
    3e9_isize;` to 1_705_032_704."""
    from helixc.ir.passes.const_fold import _wrap_int_to_type
    from helixc.ir import tir
    assert _wrap_int_to_type(6_000_000_000, tir.TIRScalar(name="isize")) \
        == 6_000_000_000
    assert _wrap_int_to_type(6_000_000_000, tir.TIRScalar(name="usize")) \
        == 6_000_000_000
    # Also check the negative-range round-trip.
    assert _wrap_int_to_type(-(1 << 40), tir.TIRScalar(name="isize")) \
        == -(1 << 40)

    # Full-pipeline round-trip: a folded isize sum must keep its
    # value when compiled and run.
    src = """
    fn main() -> isize {
        let a: isize = 3000000000_isize;
        let b: isize = 3000000000_isize;
        a + b
    }
    """
    from helixc.frontend.parser import parse
    from helixc.frontend.typecheck import typecheck
    from helixc.ir.lower_ast import lower
    from helixc.ir.passes.const_fold import fold_module
    prog = parse(src)
    errs = typecheck(prog)
    hard = [e for e in errs if not (hasattr(e, "is_warning") and e.is_warning)]
    assert not hard
    mod = lower(prog)
    fold_module(mod)
    # Find the surviving CONST_INT and check its value.
    folded_vals = []
    for fn in mod.functions.values():
        for blk in fn.blocks:
            for op in blk.ops:
                if op.kind == tir.OpKind.CONST_INT:
                    folded_vals.append(int(op.attrs["value"]))
    assert 6_000_000_000 in folded_vals, (
        f"expected const-folded 6e9, got {folded_vals} "
        f"— see cycle-19 audit C19-1"
    )
```

### Centralize the predicate (carry from cycle-17/18 forward note)

Cycle-17 forward note 2 and cycle-18 forward note 1 both called
for a single `_scalar_width_bits(ty) -> int` predicate to drive
`_is_i64_type`, `_is_u64_type`, `_check_array_elem_size_supported`,
const_fold's `_INT_BITS`, and the PTX dtype maps. C19-1 is the
direct consequence of not having that centralized predicate — the
cycle-19 fix touched two of these sites and missed the third.

Doing the refactor as part of cycle-20:

```python
# helixc/ir/scalar_width.py (new file)
_PTR_WIDTH_BITS = 64  # 64-bit targets

_SCALAR_WIDTH_BITS: dict[str, int] = {
    "i8": 8, "u8": 8, "bool": 8,  # bool: see note re: i32 storage
    "i16": 16, "u16": 16,
    "i32": 32, "u32": 32, "f32": 32,
    "i64": 64, "u64": 64, "f64": 64,
    "f16": 16, "bf16": 16,
    "isize": _PTR_WIDTH_BITS, "usize": _PTR_WIDTH_BITS,
    "fp8": 8, "mxfp4": 4, "nvfp4": 4,
    "char": 32,
}

def scalar_width_bits(ty: "tir.TIRType") -> int | None:
    if not isinstance(ty, tir.TIRScalar):
        return None
    return _SCALAR_WIDTH_BITS.get(ty.name)
```

Then `_is_i64_type` becomes `scalar_width_bits(ty) == 64 and not
_is_float_type(ty)` and the signedness check is a separate small
predicate. This is Stage-29-class refactoring; for cycle-20 the
single-line `_INT_BITS` patch closes C19-1 and the refactor can land
when convenient.

---

## Cycle 19 status

**Strict criterion (per user directive 2026-05-10): cycle clean
iff zero findings of ANY severity at confidence ≥ 80.**

This cycle finds **1 HIGH finding (C19-1)** at confidence ≥ 95%
under the type-design audit category, **introduced by the cycle-19
fix-sweep itself**.

By the strict criterion, **cycle 19's type-design audit is NOT
CLEAN**.

**Counter status (5-clean-consecutive gate under the strict
criterion)**:

- Was 0/5 after cycle 18 NOT CLEAN.
- Cycle 19 type-design: NOT CLEAN. **Counter stays at 0/5.**
- Stage 29 is gated by five fresh consecutive clean cycles. Five
  more clean cycles required after the cycle-20 fix-sweep closes
  C19-1.

The severity trend across cycles, against the strict-criterion
bar:

- Cycles 1-6: HIGH/MEDIUM/LOW each cycle — not clean.
- Cycle 7-12: clean (counter 1/5 → 3/5).
- Cycle 13: 1 HIGH (C13-1) — not clean → reset to 0/5.
- Cycle 14: clean → 1/5.
- Cycle 15: clean → 2/5.
- Cycle 16: 1 HIGH (C16-1) — not clean → reset to 0/5.
- Cycle 17: clean → 1/5.
- Cycle 18: 1 HIGH (C18-1) — not clean → reset to 0/5.
- Cycle 19 (this audit): 1 HIGH (C19-1) — not clean → stays at 0/5.

**Pattern**: each of cycles 13, 16, 18, 19 has surfaced a HIGH
silent-miscompile of the same defect class — a backend or IR pass
silently narrows a wide type when one pass's width contract
disagrees with another's. C13-1 (DCE drops trace-exit operand),
C16-1 (LOAD_ELEM/STORE_ELEM silently truncates wide elem types),
C18-1 (backend classifier missed isize/usize aliases), C19-1
(const_fold's `_INT_BITS` not updated alongside the C18-1
classifier fix).

The fact that C19-1 was **introduced by the C18-1 fix-sweep itself**
— and that the cycle-18 audit doc had explicitly flagged the
const_fold half of the fix as required — is the strongest signal
yet that the type-design surface needs the centralized width
predicate the cycle-17/18 forward notes have been carrying.

---

## Forward notes (not cycle-19 findings; recorded for visibility)

1. **Centralize scalar-width predicate** (CARRY from cycle-17/18
   forward notes; reinforced by C19-1): the cycle-20 fix-sweep
   should at minimum align `const_fold._INT_BITS` with the cycle-19
   backend canon. The full predicate factoring into a single
   `_scalar_width_bits(ty) -> int` shared by typecheck, const_fold,
   x86_64 backend, PTX backend, and `_check_array_elem_size_supported`
   remains Stage-29-class. The C19-1 finding demonstrates that
   leaving the sites un-centralized produces a HIGH-severity
   regression every time one site is touched.

2. **PTX dtype-suffix map alias gap** (CARRY from cycle-18 forward
   note 2): `_ptx_load_suffix` / `_ptx_store_suffix` /
   `_ld_reg_prefix` at `helixc/backend/ptx.py:328-332` lack
   isize/usize entries. Today this would manifest as a KeyError
   (loud, not silent) when an isize-element tensor reaches the
   tile-IR PTX emit path. Stage-29-class once tensor-of-isize is
   exercised; not blocking. Adding isize/usize to that map at the
   same time as the cycle-20 const_fold patch would close it
   defensively.

3. **Cycle-19 regression test scope** (NEW): the cycle-19 fix's
   regression test `test_c18_1_isize_usize_recognized_as_64bit`
   pins only the classifier contract via direct calls. It does
   NOT exercise the full pipeline. Had it round-tripped a folded
   isize sum through `lower → fold_module → emit`, C19-1 would
   have been caught at fix-sweep time. Cycle-20 should land the
   full-pipeline round-trip sketched above.

4. **Operand-index addressing in TIR op handlers** (CARRY from
   cycle-17 forward note 3 / cycle-18 forward note 3): `STORE_ELEM`'s
   value operand is addressed as `op.operands[1].ty` (positional).
   Named-operand accessors at TIR level would close this fragility.
   Stage-29-class.

5. **Dead `hard` local in C16-1 regression test** (CARRY from
   cycle-17 forward note 4 / cycle-18 forward note 4):
   `test_codegen.py:457`'s `hard = [...]` is computed but never
   asserted on. Stylistic.

6. **Missing `i64`-array trap regression test** (CARRY from
   cycle-17 forward note 5 / cycle-18 forward note 5): cycle-16
   doc named three regression tests; cycle-17 implemented `f64`;
   cycle-19 did not add `i64`. Defense-in-depth against future
   narrowing of the `wide_widths` set.

7. **Stage-29 deliverable: full 8-byte LOAD_ELEM / STORE_ELEM
   lowering** (CARRY from cycle-17/18 forward note 6): once
   landed, `_check_array_elem_size_supported` becomes either
   dead code or a narrower guard.

8. **`_alloc_array` `elem_size` parameter unwired** (CARRY from
   cycle-17/18 forward note 7): IR-level `ALLOC_ARRAY` op's
   `dtype` attribute read but not propagated. Phase-0 safe under
   C16-1 trap. Stage-29-class.

9. **`Value.ty` not frozen** (CARRY from cycle-17/18 forward
   note 8): `tir.Value` is `@dataclass` not `@dataclass(frozen=
   True)`. Stage-29-class hardening.

10. **`Op.results: list[Value]` over-general** (CARRY from
    cycle-17/18 forward note 9): single-result Op convention is
    convention-only. Stage-29-class.

11. **`SIDE_EFFECT_KINDS` static cross-check** (CARRY from
    cycle-14 forward note 5): no static guarantee that every
    side-effecting `OpKind` is in the set. Stage-29-class.

12. **Cycle-20 baseline**: cycle 19's audit is read-only at HEAD
    0803902. The cycle-20 fix-sweep will be touching one production-
    code file (`helixc/ir/passes/const_fold.py`); cycle-20's audit-B
    can re-read the C19-1 fix and its regression test as the only
    delta against HEAD 0803902.

13. **Stage-29 readiness**: counter stays at 0/5. Five fresh
    consecutive clean cycles remain required after the cycle-20
    fix-sweep closes C19-1.
