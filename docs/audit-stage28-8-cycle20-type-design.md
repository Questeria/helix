# Stage 28.8 Pre-29 Audit Gate — Cycle 20, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit (HEAD)**: 5a1e406 (cycle-20 fix-sweep landing C19-1).
**Cycle-19 baseline**: 0803902 (cycle-19 audit found C19-1 HIGH).
**Cycle-19 status**: NOT CLEAN — counter at 0/5.
**Cycle-20 fix-sweep**: closes C19-1 (HIGH) by updating
`const_fold._INT_BITS["isize"] = 64`, `_INT_BITS["usize"] = 64`,
plus a new in-pass regression test
`test_c19_1_isize_usize_are_64_bit_in_wrap` (37 const_fold tests
pass, was 36).

**Scope**: Audit category B (type-system / dispatch / soundness)
under the strict criterion. Per the user directive, this audit:

1. Verifies the cycle-20 fix matches typecheck.py / backend
   classifiers / autodiff treatment of isize/usize.
2. Rotates lens to OTHER width-aware tables that might still
   drift in the wake of the cycle-19+20 alignment cascade — in
   particular the cycle-18/19-carried PTX dtype-suffix map at
   `helixc/backend/ptx.py:327-346`, `tile_ir.py` element-width
   handling, and `lower_ast.py:357-371` literal-type inference.

**Counter context** (per user directive 2026-05-10):

- Cycle 18 NOT CLEAN → reset to 0/5.
- Cycle 19 NOT CLEAN → stays at 0/5.
- Cycle 20 (this audit): if CLEAN under the strict criterion,
  counter advances 0/5 → 1/5. A finding here keeps the counter
  at 0/5.

---

## Cycle-20 production-code delta (since cycle-19 baseline 0803902)

```
git show --stat 5a1e406 -- helixc/
```

```
 helixc/ir/passes/const_fold.py   |   9 +-
 helixc/tests/test_const_fold.py  |  31 ++
```

The fix is a single edit to `_INT_BITS` at `const_fold.py:43-56`:

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
    "bool": 32,  # bool comparisons reified to i32 in IR
}
```

Plus a regression test `test_c19_1_isize_usize_are_64_bit_in_wrap`
at `test_const_fold.py:356-384` pinning `_wrap_int_to_type` agreement
between isize↔i64 and usize↔u64 across a range of values.

---

## Verification: cycle-20 fix is consistent across the type system

**Pass.** Post-cycle-20, the four width-aware tables for
isize/usize that interact in the lowering pipeline all agree:

| Site | Treatment of isize/usize | Source |
|---|---|---|
| `typecheck.py:225-228` `_WIDEN_NAME_ALIASES` | `isize→i64`, `usize→u64` | Pre-existing canon |
| `typecheck.py:241` `_WIDEN_RANK` | isize=40 (=i64), usize=41 (=u64) | Pre-existing canon |
| `typecheck.py:1816-1817` `_INT_BOUNDS` | isize=i64 range, usize=u64 range | Pre-existing canon |
| `autodiff.py:60-79` `NUMERIC_FOR_AD` | included in broad set, no width-dispatch | Cycle-2 fix |
| `x86_64.py:1005-1011` `_is_i64_type` | recognizes isize | Cycle-19 fix |
| `x86_64.py:1013-1017` `_is_u64_type` | recognizes usize | Cycle-19 fix |
| `x86_64.py:1042` `_check_array_elem_size_supported.wide_widths` | includes isize, usize | Cycle-17 fix |
| `const_fold.py:43-56` `_INT_BITS` | isize=64, usize=64 | **Cycle-20 fix** |

The cycle-20 reproducer from the cycle-19 doc:

```helix
fn main() -> isize {
    let a: isize = 3000000000_isize;
    let b: isize = 3000000000_isize;
    a + b
}
```

is closed end-to-end:

1. **Typecheck**: `_check_int_lit_fits` with `eff_name = "isize"`,
   `_INT_BOUNDS["isize"]` accepts 3e9. Pass.
2. **Lower to TIR**: two `CONST_INT(3_000_000_000)` of type
   `TIRScalar("isize")`, one `ADD`.
3. **const_fold at -O1** (default): folds ADD →
   `_wrap_int_to_type(6_000_000_000, TIRScalar("isize"))`. Post-fix,
   `_INT_BITS["isize"] = 64`, so `mask = (1<<64)-1`, `half = 1<<63`.
   `6_000_000_000 < half`, no sign correction. Returns
   `6_000_000_000`. Rewrites ADD into `CONST_INT(6_000_000_000)`
   of type isize.
4. **Backend at x86_64.py:1148**: `_is_i64_type(TIRScalar("isize"))`
   returns True. Emits `mov_rax_imm64(6_000_000_000)`. Stores 8
   bytes.
5. **Runtime**: returns 6_000_000_000. Correct.

The folded path and un-folded path now both produce
`6_000_000_000` (optimization-stable).

---

## Lens rotation: OTHER width-aware tables — finding C20-1

The user directive asked specifically about three further sites
that might still drift after the cycle-19+20 alignment cascade:

- **`helixc/backend/ptx.py:328-332`** dtype-suffix map (forward
  note from cycles 18 + 19).
- **`helixc/ir/tile_ir.py`** element-width handling.
- **`helixc/ir/lower_ast.py:357-371`** literal-type inference.

### lower_ast.py:357-371 — `_PRIMITIVE_TYPE_NAMES` (no finding)

`_PRIMITIVE_TYPE_NAMES` at lower_ast.py:356-362 is a name **set**,
not a width-keyed dict:

```python
_PRIMITIVE_TYPE_NAMES = frozenset({
    "i8", "i16", "i32", "i64", "isize",
    "u8", "u16", "u32", "u64", "usize",
    "bool", "char",
    "bf16", "f16", "f32", "f64",
    "unit",
})
```

It is used at `_lower_type` (line 364-371) only for membership;
the lowered form is `tir.TIRScalar(ty.name)` which preserves the
name string and defers width semantics to downstream. No width
classification on this path. Clean.

(There is a pre-existing AST→IR lowering subtlety at line 1037-1038
where an un-suffixed `IntLit` lowers as `const_int(value, "i32")`
regardless of the contextual `let x: isize` annotation. Typecheck
calls `_check_int_lit_fits` against the contextual ty so the
literal is accepted into isize range; the lowered IR value retains
type i32 because lower_ast does not thread context into bare
`IntLit` lowering. This is an existing Phase-0 narrow-by-default
behavior, not introduced by cycles 19/20, and is gated by user
having to write `let x: isize = 3000000000` without the `_isize`
suffix to hit it. Carried as forward note 7 — out of strict-criterion
scope for cycle-20 since not introduced by the cycle-19+20 cascade.)

### tile_ir.py — element-width handling (no finding)

`helixc/ir/tile_ir.py` carries the tile-IR `dtype` as
`tir.TIRScalar` (line 47: `TileType.dtype: tir.TIRScalar`) and
forwards it verbatim through tile-lowering. There are no
width-keyed dictionaries or name-string discriminators in
tile_ir.py — `grep -n "isize\|usize"` returns zero hits. Width
semantics live at the backends consuming tile-IR (PTX and x86)
not in tile-IR itself. Clean.

### Finding C20-1 (HIGH): `ptx.py` dtype tables silently default 32-bit for isize/usize and for several other dtypes

**This is the carry-forward note from cycle-18 forward note 2 and
cycle-19 forward note 2; the cycle-19 audit characterized it as
"loud KeyError" but on re-reading it is actually a SILENT default,
not loud, so it promotes from forward-note to finding under the
strict criterion.**

#### Location

`helixc/backend/ptx.py:157-168, 327-346`:

```python
def _ptx_type_str(self, ty: tir.TIRType) -> str:
    if isinstance(ty, tir.TIRScalar):
        mapping = {
            "i8": ".b8", "i16": ".b16", "i32": ".b32", "i64": ".b64",
            "u8": ".b8", "u16": ".b16", "u32": ".b32", "u64": ".b64",
            "bool": ".pred",
            "f16": ".f16", "bf16": ".bf16", "f32": ".f32", "f64": ".f64",
        }
        return mapping.get(ty.name, ".b32")       # ← silent .b32 fallback
    if isinstance(ty, tir.TIRUnit):
        return ""
    return ".b64"

# ...

_DTYPE_SIZE = {"i8": 1, "u8": 1, "i16": 2, "u16": 2, "f16": 2, "bf16": 2,
                "i32": 4, "u32": 4, "f32": 4, "i64": 8, "u64": 8, "f64": 8}
_DTYPE_PTX_LOAD = {"i8": "s8", "u8": "u8", "i16": "s16", "u16": "u16",
                    "f16": "f16", "bf16": "bf16",
                    "i32": "s32", "u32": "u32", "f32": "f32",
                    "i64": "s64", "u64": "u64", "f64": "f64"}

def _dtype_size(self, dtype: str) -> int:
    return self._DTYPE_SIZE.get(dtype, 4)         # ← silent 4-byte fallback

def _ptx_load_suffix(self, dtype: str) -> str:
    return self._DTYPE_PTX_LOAD.get(dtype, "u32") # ← silent u32 fallback

def _ld_reg_prefix(self, dtype: str) -> str:
    if dtype in ("f16", "bf16", "f32", "f64"):
        return "f"
    if dtype in ("i64", "u64"):
        return "rd"
    return "r"                                    # ← silent 32-bit r fallback
```

For the dtype string `"isize"` (or `"usize"`):

- `_ptx_type_str`: not in mapping → falls back to `.b32` (32-bit).
  Correct should be `.b64` matching typecheck/backend canon.
- `_dtype_size("isize")`: returns 4 (default). Correct: 8.
- `_ptx_load_suffix("isize")`: returns `"u32"` (default). Correct:
  `"s64"` (or `"u64"` for usize).
- `_ld_reg_prefix("isize")`: not in `("i64", "u64")` → returns
  `"r"` (32-bit register pool). Correct: `"rd"` (64-bit).

Four sites, all silent default-to-32-bit.

#### Why the cycle-18/19 "KeyError, loud" characterization was wrong

Cycle-18 forward note 2 and cycle-19 forward note 2 both stated:

> Today this would manifest as a KeyError (loud, not silent) when
> an isize-element tensor reaches the tile-IR PTX emit path.

Re-reading the actual code at ptx.py:335, 338, 165: every lookup
is a `.get(dtype, default)`, not a `[dtype]` indexed access.
`dict.get` returns the default on missing key. No KeyError is
raised. The PTX backend silently emits 32-bit-stride / `.b32` /
`u32`-load PTX for any unknown dtype — including isize/usize
(post-cycle-20 canon), and also including `unit`, generic-T
type-parameter leaks, and tile-IR types whose `dtype.name`
doesn't match the small known set.

This re-classification matters: under the cycle-13 / 16 / 18 / 19
defect-class framing ("silent narrowing when one pass's width
contract disagrees with another's"), a silent fallback is exactly
the same defect class. A loud KeyError would be a Phase-0 narrow+
loud-trap pattern (cf. `_check_float_supported` at x86_64.py:1019
and `_check_array_elem_size_supported` at x86_64.py:1030) — those
are accepted as Phase-0-safe trap-or-Stage-29 deliverables. The
silent fallback is NOT in that class; it's the same class as
C13-1 / C16-1 / C18-1 / C19-1.

#### Concrete reproducer

```helix
@kernel
fn copy_isize(dst: tensor<isize, [N], hbm>, src: tensor<isize, [N], hbm>) {
    let i: i32 = thread_idx_x();
    dst[i] = src[i];
}
```

Pipeline trace:

1. **Parser**: accepts `tensor<isize, ...>` per parser.py:703-729
   (no dtype-restriction; `_parse_tensor_type` calls
   `_parse_type` which accepts any name including isize/usize).
2. **Typecheck**: accepts. `_resolve_type` at typecheck.py:516-521
   resolves TyTensor by recursively resolving dtype; isize is a
   primitive name so it resolves to `TyPrim("isize")`. No "tensor
   dtype must be float-only" check exists.
3. **Lower to TIR**: `_lower_type` at lower_ast.py:374-380 carries
   `tir.TIRScalar("isize")` as the dtype of the `TIRTensorTy`.
4. **Lower to Tile IR**: `tile_ir.py` forwards the
   `TIRScalar("isize")` as `TileType.dtype`. No width dispatch.
5. **PTX emit at ptx.py:280, 305** (`TILE_INDEX_LOAD_HBM` /
   `TILE_INDEX_STORE_HBM`): reads `op.attrs["dtype"]` (the dtype
   name string "isize"), then emits:
   - `mul.wide.s32 {off}, {idx_reg}, 4` — stride 4 instead of 8.
     The address arithmetic walks the array at half-stride.
   - `ld.global.u32 {dst}, [{addr}]` — 32-bit load instead of 64.
     High half of each isize element is silently discarded.
   - `cvta.to.global.u64` is correct (this is the base-pointer
     conversion, not element-typed).
6. **Reg-pool routing at line 296**: `dst_prefix = self._ld_reg_prefix("isize")`
   → returns `"r"`, so the destination is a 32-bit `%r` register
   instead of a 64-bit `%rd`. Subsequent uses of this destination
   would chain the narrowing.

No diagnostic at any boundary. The cycle-19+20 backend-x86_64-side
fixes for isize/usize are now correct, but the PTX backend was
**not** updated alongside — so the same class of `let x: isize =
5_000_000_000` silent-truncation bug that C18-1 / C19-1 closed for
the x86_64 backend is **reopened on the PTX backend** for any
isize/usize-typed tile or HBM tensor element.

#### Reachability

- Requires user-written `@kernel` (or tile-fn) with isize/usize-
  typed tensor element OR a tile-IR `TILE_INDEX_LOAD_HBM` /
  `TILE_INDEX_STORE_HBM` carrying `attrs["dtype"] = "isize"` or
  `"usize"`.
- Requires `--emit-ptx` (or any path that calls `emit_ptx`).
  Default `helixc compile -o foo` does NOT invoke PTX emit per
  check.py:636-650 (PTX is gated behind the explicit flag).
- typecheck does NOT reject isize/usize tensor/tile dtypes.
- No examples in `helixc/examples/` use isize/usize tensor dtypes
  today (all `tensor<...>` uses are `f32` or `bf16`), so no
  in-tree program exhibits the bug.
- But the typecheck-permissive surface + silent emit means a
  user writing a pointer-table kernel (`tensor<isize, [N]>` for
  index arrays or pointer-arrays) hits this immediately.

#### Severity assessment

| Criterion | Assessment |
|---|---|
| Reachable today | Yes, gated by `--emit-ptx` + isize/usize tensor dtype |
| Diagnostic surface | None — silent .get fallback |
| Output corruption | Yes — wrong stride, wrong load width, wrong reg pool |
| Affects pointer arithmetic | Directly — usize is *the* index/pointer type, exactly what one would use for index-array kernels |
| Optimization-stability | N/A (PTX path doesn't run const_fold the same way) |
| Detectable by tests | Yes — PTX emit on a `tensor<isize>` kernel and grep for `ld.global.s64` / `st.global.s64` |
| Existing test coverage | None — no PTX tests use isize/usize tensor dtypes |
| Type-system contract violated | "isize/usize are 64-bit aliases on 64-bit targets" (typecheck.py canon, cycle-19+20 backend canon) — violated by PTX backend |
| Cycle-18/19 characterization | "loud KeyError" — empirically wrong; it's silent `.get(..., default)` |
| Same defect class as C13-1 / C16-1 / C18-1 / C19-1 | Yes — silent narrowing when one pass's width contract disagrees with another's |

The "gated by `--emit-ptx`" qualifier might suggest MEDIUM, but
under the strict criterion (zero findings of ANY severity at
confidence ≥ 80%) the gating doesn't matter — a silent
miscompile reachable from typecheck-accepted code through a
documented compiler flag is HIGH-class. **Severity: HIGH.**

#### Confidence

- Code trace is mechanical: read ptx.py:165 (`mapping.get(ty.name,
  ".b32")`), ptx.py:335 (`self._DTYPE_SIZE.get(dtype, 4)`),
  ptx.py:338 (`self._DTYPE_PTX_LOAD.get(dtype, "u32")`),
  ptx.py:340-346 (`_ld_reg_prefix`).
- The `.get(..., default)` semantics are Python stdlib —
  `dict.get(key, default)` returns `default` on missing key, does
  NOT raise.
- The PTX-emit path is reachable from typecheck-accepted user
  code per the typecheck.py:516-521 trace (no dtype restriction
  on `TyTensor` / `TyTile`).
- The "loud KeyError" cycle-18/19 forward-note characterization
  is empirically wrong by inspection.

**Confidence ≥ 90%** that this is a real, reachable silent
miscompile on the PTX path that the cycle-19+20 alignment cascade
left unaddressed. Promotes per the user directive (threshold ≥
80%).

### Adjacent narrowness in PTX emit (informational, not C20-1 itself)

The PTX backend has additional Phase-0 narrowings at
ptx.py:171-229 — `SCALAR_CONST_INT` unconditionally emits
`mov.b32`, `SCALAR_ADD`/`SCALAR_SUB`/`SCALAR_MUL`/`SCALAR_NEG`
unconditionally emit `.s32` (for non-float operands), and the
scalar register allocator only allocates `%r` (32-bit) for
non-float results. This is a broader Phase-0 PTX MVP limitation
documented at ptx.py:172 ("v0.1: only handle a tiny scalar subset
for sanity testing"). It is *not* C20-1; it's a pre-existing
Phase-0 scope choice. Flagged here so the cycle-20 fix-sweep
doesn't try to fix one site (the dtype-suffix map) and leave the
others.

A defensible fix is to either:
- Surface a loud `NotImplementedError` from `_dtype_size` /
  `_ptx_load_suffix` / `_ld_reg_prefix` when dtype is unknown —
  matches the cycle-3 / cycle-16 narrow+loud pattern. **This
  closes C20-1** at the cost of failing PTX emit for isize/usize
  tensors entirely until the wide-load PTX path lands. Stage-29-
  deferred, with a loud trap meanwhile.
- Or add isize/usize entries to all four tables (`_ptx_type_str`'s
  mapping, `_DTYPE_SIZE`, `_DTYPE_PTX_LOAD`, `_ld_reg_prefix`'s
  64-bit set) routing to 64-bit emit. This is more code but
  actually compiles isize/usize PTX. The autotuner / wide-PTX
  story would prefer this.

The cycle-19 doc's "Option B" framing for C18-1 applies here too:
the loud-trap fix is the minimum to close the strict-criterion
finding; the centralized scalar-width predicate from cycle-17/18
forward note 1 is the right Stage-29-class refactor.

---

## Cross-check: other width-classifier sites at HEAD

Searched `helixc/` for `_INT_BITS|_DTYPE_SIZE|_DTYPE_PTX|_is_i64|
_is_u64|_check_array_elem_size|wide_widths|_scalar_width|
elem_size_bytes`:

| Site | isize/usize handling | Status |
|---|---|---|
| `x86_64.py:1005-1011` `_is_i64_type` | includes "isize" | Cycle-19, correct |
| `x86_64.py:1013-1017` `_is_u64_type` | includes "usize" | Cycle-19, correct |
| `x86_64.py:1042` `wide_widths` | includes isize/usize | Cycle-17, correct |
| `x86_64.py:165, 327, 329, 338, 341-346` (PTX tables) | missing | **C20-1 (HIGH)** |
| `const_fold.py:43-56` `_INT_BITS` | isize=64, usize=64 | Cycle-20, correct |
| `typecheck.py:225-228, 241, 1816-1817` | canonical (i64 rank/range) | Pre-existing, correct |
| `autodiff.py:60-79` `NUMERIC_FOR_AD` | broad set, no width-dispatch | Cycle-2, correct |
| `lower_ast.py:356-362` `_PRIMITIVE_TYPE_NAMES` | name set, no width logic | Pre-existing, correct |
| `tile_ir.py` | forwards `TIRScalar`, no width logic | Pre-existing, correct |
| `tir.py:432, 436` `const_int` / `const_float` | takes dtype string, no width logic | Pre-existing, correct |

**Exactly one site remains misaligned**: the PTX backend's four
silent-default tables / branches. C20-1 above.

(Bootstrap `kovc.hx:5466-5473` has its own width-class
predicates for the Phase-1 self-hosted compiler — bootstrap-source
width handling is Phase-1 scope, out of the Stage 28.8 Phase-0
audit gate.)

---

## Findings summary

| ID | Severity | Confidence | Location | Description |
|---|---|---|---|---|
| C20-1 | HIGH | ≥90% | `helixc/backend/ptx.py:165, 327, 329, 338, 340-346` | PTX backend's `_ptx_type_str`, `_DTYPE_SIZE`, `_DTYPE_PTX_LOAD`, `_ld_reg_prefix` all silently default isize/usize (and any other unknown dtype) to 32-bit treatment (`.b32` / 4-byte stride / `u32` load suffix / `%r` register). The cycle-19+20 alignment cascade fixed the x86_64 backend and const_fold to treat isize/usize as 64-bit, but the PTX backend was not updated alongside. Same defect class as C13-1 / C16-1 / C18-1 / C19-1: one pass's width contract disagrees with another's. Reachable from typecheck-accepted `tensor<isize, ...>` / `tile<isize, ...>` user code via `--emit-ptx`. Cycle-18/19 forward note characterized this as "KeyError, loud" — re-reading shows it is `.get(..., default)` (silent fallback), so it promotes from forward-note to finding under the strict criterion. |

**Total**: 1 HIGH, 0 MEDIUM, 0 LOW.

---

## Recommended fix (for cycle-21 fix-sweep)

Two viable options, each closing C20-1:

### Option A: narrow+loud trap (Phase-0 safe)

Add an explicit unknown-dtype trap in all four PTX helpers,
matching the `_check_float_supported` / `_check_array_elem_size_
supported` pattern:

```python
def _dtype_size(self, dtype: str) -> int:
    if dtype not in self._DTYPE_SIZE:
        raise NotImplementedError(
            f"PTX backend does not yet support tile/tensor "
            f"element dtype '{dtype}' — would silently default to "
            f"4-byte stride. See audit-stage28-8 cycle 20 C20-1."
        )
    return self._DTYPE_SIZE[dtype]

# same for _ptx_load_suffix, _ld_reg_prefix, _ptx_type_str.
```

Pros: zero risk of silent miscompile. Matches cycle-3 / cycle-16
narrow+loud pattern. Single-cycle fix.
Cons: cuts off the PTX path for isize/usize tensors entirely
until the wide-PTX path lands.

### Option B: add isize/usize 64-bit entries (functional)

Extend each table:

```python
_DTYPE_SIZE = {..., "i64": 8, "u64": 8, "isize": 8, "usize": 8, "f64": 8}
_DTYPE_PTX_LOAD = {..., "i64": "s64", "u64": "u64",
                   "isize": "s64", "usize": "u64", "f64": "f64"}

# In _ptx_type_str's mapping: add "isize": ".b64", "usize": ".b64".
# In _ld_reg_prefix: extend the 64-bit set:
if dtype in ("i64", "u64", "isize", "usize"):
    return "rd"
```

Pros: matches the cycle-19+20 backend canon exactly. Functional —
isize/usize tensors compile correctly.
Cons: leaves the bigger "PTX scalar ops are always s32" Phase-0
narrowness (ptx.py:171-229) in place; would need
follow-up.

**Recommendation**: Option A for cycle-21 fix-sweep (closes C20-1
under the strict criterion with minimum blast radius). Option B
or the centralized scalar-width predicate is Stage-29-class.

### Regression test sketch

```python
def test_c20_1_ptx_isize_usize_not_silent_32bit():
    """Audit 28.8 cycle 20 C20-1 (HIGH): PTX backend must not
    silently default isize/usize to 32-bit treatment. The cycle-19+20
    backend/const_fold canon made isize/usize 64-bit; PTX must
    either match or raise NotImplementedError."""
    from helixc.backend.ptx import PtxEmitter
    e = PtxEmitter()
    # Option-A behavior: raise on unknown dtype.
    import pytest
    with pytest.raises(NotImplementedError):
        e._dtype_size("isize")
    # OR Option-B behavior: 8 bytes, "s64" suffix.
    # assert e._dtype_size("isize") == 8
    # assert e._ptx_load_suffix("isize") == "s64"
    # assert e._ptx_load_suffix("usize") == "u64"
    # assert e._ld_reg_prefix("isize") == "rd"
```

### Centralize the predicate (carry from cycle-17/18/19)

The pattern has now produced four HIGH findings in a row (C13-1,
C16-1, C18-1, C19-1, C20-1 if we count this cycle). Each time, a
single pass's width-keyed table is touched, another pass's width-
keyed table drifts. The cycle-17/18/19 forward notes have all
called for a single `_scalar_width_bits(ty) -> int` predicate to
drive *every* width-keyed dispatch. Cycle-21 should at minimum
land Option A for C20-1; the centralized predicate is Stage-29-
class.

Sketch (unchanged from cycle-19 doc):

```python
# helixc/ir/scalar_width.py (new file)
_PTR_WIDTH_BITS = 64  # 64-bit targets

_SCALAR_WIDTH_BITS: dict[str, int] = {
    "i8": 8, "u8": 8, "bool": 8,
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

Then `_is_i64_type`, const_fold's wrap, PTX's `_dtype_size`, all
become single-line wrappers and the canon lives in one file.

---

## Cycle 20 status

**Strict criterion (per user directive 2026-05-10): cycle clean
iff zero findings of ANY severity at confidence ≥ 80.**

This cycle finds **1 HIGH finding (C20-1)** at confidence ≥ 90%
under the type-design audit category. C20-1 is a re-classification
of the cycle-18/19-carried PTX forward note — the prior "KeyError,
loud" characterization was empirically wrong; the actual PTX
backend uses `.get(..., default)` and silently emits 32-bit PTX
for unknown dtypes including isize/usize.

By the strict criterion, **cycle 20's type-design audit is NOT
CLEAN**.

**Counter status (5-clean-consecutive gate under the strict
criterion)**:

- Was 0/5 after cycle 19 NOT CLEAN.
- Cycle 20 type-design: NOT CLEAN. **Counter stays at 0/5.**
- Stage 29 is gated by five fresh consecutive clean cycles. The
  cycle-21 fix-sweep should close C20-1 (Option A: narrow+loud
  trap is the minimum-blast-radius fix).

The severity trend, updated:

- Cycles 1-6: HIGH/MEDIUM/LOW each cycle — not clean.
- Cycles 7-12: clean (counter 1/5 → 3/5 across the run).
- Cycle 13: 1 HIGH (C13-1) — not clean → reset 0/5.
- Cycle 14: clean → 1/5.
- Cycle 15: clean → 2/5.
- Cycle 16: 1 HIGH (C16-1) — not clean → reset 0/5.
- Cycle 17: clean → 1/5.
- Cycle 18: 1 HIGH (C18-1) — not clean → reset 0/5.
- Cycle 19: 1 HIGH (C19-1, introduced by cycle-18 fix-sweep) —
  not clean → stays 0/5.
- Cycle 20 (this audit): 1 HIGH (C20-1, re-classified from
  cycle-18/19 forward note) — not clean → stays 0/5.

**Pattern**: each of cycles 13, 16, 18, 19, 20 has surfaced a
HIGH silent-miscompile of the same defect class — a pass silently
narrows a wide type when one pass's width contract disagrees with
another's. C13-1 (DCE drops trace-exit operand), C16-1 (LOAD_ELEM/
STORE_ELEM silently truncates wide elem types), C18-1 (backend
classifier missed isize/usize aliases), C19-1 (const_fold's
`_INT_BITS` not updated alongside C18-1), C20-1 (PTX backend's
four dtype tables silently default 32-bit for isize/usize, missed
by both the cycle-19 and cycle-20 fix-sweeps).

The recurring pattern across five cycles makes the case for the
centralized `_scalar_width_bits` predicate increasingly strong.
The cycle-19 audit said this was "the strongest signal yet" that
the centralized predicate is needed; cycle-20 doubles that signal.
Without the centralized predicate, each fix-sweep will continue to
leave one or more sites out of alignment, producing another
HIGH-class finding next cycle.

---

## Forward notes (not cycle-20 findings; recorded for visibility)

1. **Centralize scalar-width predicate** (CARRY from cycle-17/18/19
   forward notes; reinforced by C20-1): Stage-29-class. Each
   cycle's fix-sweep touches a subset of width-keyed sites and
   misses the rest. The centralized predicate would eliminate
   this class of regression.

2. **PTX scalar-op narrowness beyond C20-1** (NEW): ptx.py:171-229's
   `SCALAR_CONST_INT` / `SCALAR_ADD` / `SCALAR_SUB` / `SCALAR_MUL`
   / `SCALAR_NEG` are unconditionally `.s32` / `mov.b32`. Even
   with C20-1 fixed at the dtype-table layer, the scalar-op layer
   would still emit 32-bit ops on 64-bit operands. Phase-0 MVP
   scope. Stage-29-class once tile-IR scalar-op widths are
   demanded.

3. **PTX `.param` register convention** (NEW, ptx.py:152-155):
   `_format_param` treats all params as `.b64` (pointer-like).
   For non-pointer scalar params this is overshoot, not narrowing,
   and is currently safe. Documented for visibility.

4. **lower_ast.py un-suffixed IntLit context-insensitivity**
   (NEW, pre-existing): `lower_expr(IntLit)` at lower_ast.py:1037-
   1038 lowers as `const_int(value, "i32")` regardless of the
   `let x: isize = ...` declared type. Typecheck accepts the
   literal against the contextual width, but the lowered TIR
   value retains type i32, so the let-binding aliases an i32 IR
   value to a name the source-code labelled isize. Subsequent
   uses of that name in isize-typed positions would mismatch the
   IR type. Reachable but currently masked by the fact that
   typical isize-typed bindings use the `_isize` suffix; flagged
   for future investigation. Same defect class as C18-1 / C19-1
   / C20-1 — width-contract disagreement — but pre-existing, not
   introduced by cycle-19+20.

5. **Cycle-20 regression-test scope gap** (CARRY from cycle-19
   forward note 3): cycle-20 added `test_c19_1_isize_usize_are_64_
   bit_in_wrap` which tests `_wrap_int_to_type` directly; it
   does NOT round-trip a folded isize sum through the full
   `parse → typecheck → lower → fold_module → emit` pipeline.
   The cycle-19 audit doc recommended a full-pipeline test; this
   wasn't added in cycle-20. Defense-in-depth gap; not blocking.

6. **PTX dtype-suffix map alias gap empirically silent, not loud**
   (CORRECTION from cycle-18/19 forward note 2): see C20-1
   reasoning above. Cycle-18 / 19 said "KeyError, loud"; actual
   code is `.get(..., default)`, silent. Cycle-20 doc supersedes
   the cycle-18/19 characterization.

7. **Operand-index addressing in TIR op handlers** (CARRY from
   cycle-17 forward note 3): `STORE_ELEM`'s value operand is
   addressed as `op.operands[1].ty` (positional). Named-operand
   accessors would close the fragility. Stage-29-class.

8. **Dead `hard` local in C16-1 regression test** (CARRY from
   cycle-17 forward note 4): `test_codegen.py:457`'s `hard =
   [...]` is computed but never asserted on. Stylistic.

9. **Missing `i64`-array trap regression test** (CARRY from
   cycle-17/18/19 forward note 5): cycle-16 doc named three
   regression tests; cycle-17 implemented `f64`; cycles 19/20
   did not add `i64`. Defense-in-depth.

10. **Stage-29 deliverable: full 8-byte LOAD_ELEM / STORE_ELEM
    lowering** (CARRY from cycle-17/18/19 forward note 6): once
    landed, `_check_array_elem_size_supported` becomes either
    dead code or a narrower guard.

11. **`_alloc_array` `elem_size` parameter unwired** (CARRY):
    IR-level `ALLOC_ARRAY` op's `dtype` attribute read but not
    propagated. Phase-0 safe under C16-1 trap. Stage-29-class.

12. **`Value.ty` not frozen** (CARRY): `tir.Value` is `@dataclass`
    not `@dataclass(frozen=True)`. Stage-29-class hardening.

13. **`Op.results: list[Value]` over-general** (CARRY): single-
    result Op convention is convention-only. Stage-29-class.

14. **`SIDE_EFFECT_KINDS` static cross-check** (CARRY from
    cycle-14 forward note 5): no static guarantee that every
    side-effecting `OpKind` is in the set. Stage-29-class.

15. **Cycle-21 baseline**: cycle 20's audit is read-only at HEAD
    5a1e406. The cycle-21 fix-sweep will be touching one
    production-code file (`helixc/backend/ptx.py`); cycle-21's
    audit-B can re-read the C20-1 fix as the only delta against
    HEAD 5a1e406.

16. **Stage-29 readiness**: counter stays at 0/5. Five fresh
    consecutive clean cycles remain required after the cycle-21
    fix-sweep closes C20-1. The centralized scalar-width
    predicate, if it lands as part of the C20-1 fix or shortly
    after, would substantially de-risk the streak — each of
    cycles 13/16/18/19/20 is the same defect class and the
    refactor closes the class.
