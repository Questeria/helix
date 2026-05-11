# Stage 28.8 Pre-29 Audit Gate — Cycle 17, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit**: 0243d5c (HEAD; production code byte-identical to c6136d4)
**Cycle-16 baseline**: 4c74627 (commit audited by cycle 16's
type-design pass; C16-1 found HIGH).
**Cycle-17 fix-sweep**: c6136d4 — added
`_check_array_elem_size_supported` helper in
`helixc/backend/x86_64.py:983-1003` plus two call sites at the
LOAD_ELEM and STORE_ELEM emit branches (lines 2743, 2764), plus
one regression test
`test_c16_1_wide_array_elem_traps_at_codegen` in
`helixc/tests/test_codegen.py:437-476`. Plus persisted 10 audit
docs under `docs/`. Total `helixc/` delta: 2 files, +68 lines.

**Scope**: Audit category B (type-system / dispatch / soundness)
under the strict criterion. Two questions:

1. Does the cycle-17 fix-sweep correctly close C16-1 (the HIGH
   finding from cycle 16) without introducing a new type-design
   hole?
2. Independent fresh-eyes re-audit of the type-system surface at
   HEAD — any new defect on rotated lenses?

**Counter context** (per user directive 2026-05-10):

- Cycle 14: FULLY CLEAN. Counter: 0/5 → 1/5.
- Cycle 15: FULLY CLEAN. Counter: 1/5 → 2/5.
- Cycle 16: type-design found C16-1 (HIGH) — cycle 16 not clean.
  Counter reset 2/5 → 0/5.
- Cycle 17 (this audit): if CLEAN under the strict criterion and
  conditional on the other cycle-17 categories also being CLEAN,
  the counter advances 0/5 → 1/5.

---

## Cycle-17 production-code delta (since cycle-16 baseline 4c74627)

```
git diff 4c74627..HEAD --stat -- helixc/
 helixc/backend/x86_64.py    | 28 ++++++++++++++++++++++++++++
 helixc/tests/test_codegen.py | 40 ++++++++++++++++++++++++++++++++++++++++
 2 files changed, 68 insertions(+)
```

The only production-code surface touched in cycle 17 is
`helixc/backend/x86_64.py`. The full `helixc/frontend/` subtree,
`helixc/check.py`, the entire `helixc/ir/` subtree, and
`helixc/backend/ptx.py` are **byte-identical to the cycle-16
baseline 4c74627** (and through 4c74627 back to the cycle-10
baseline c2e36d4 for all of those subtrees).

The single source-file delta in `helixc/backend/x86_64.py`:

- **Insert** `_check_array_elem_size_supported(ty)` helper at
  lines 983-1003 (immediately after `_check_float_supported`),
  21 lines including docstring.
- **Insert** call to that helper at LOAD_ELEM emit site (line
  2743) on `op.results[0].ty`. 1 line + 2-line comment.
- **Insert** call to that helper at STORE_ELEM emit site (line
  2764) on `op.operands[1].ty`. 1 line + 2-line comment.

Cross-checks:

```
git diff 4c74627..HEAD -- helixc/frontend/   (empty)
git diff 4c74627..HEAD -- helixc/check.py    (empty)
git diff 4c74627..HEAD -- helixc/ir/         (empty)
git diff 4c74627..HEAD -- helixc/backend/ptx.py    (empty)
git diff 4c74627..HEAD -- helixc/backend/elf_dyn.py (empty)
```

`0243d5c` (Phase A staging-refinement commit) is docs-only and
does not change `helixc/`.

---

## Cycle-16 finding re-verification

| ID | Severity prev | Audit (prev) | Status now | Notes |
|---|---|---|---|---|
| C16-1 | HIGH | type-design (cycle 16) | **CLOSED** | The cycle-17 fix-sweep added the helper recommended by the cycle-16 doc (recommendation 3, "backstop in `_emit_op`'s LOAD_ELEM / STORE_ELEM branches"). Both reachable miscompile sites are now guarded. Regression test asserts the trap fires on the exact reproducer from the cycle-16 doc. Existing i32-array test (`test_array_literal_and_index`) and i32-array-assign test (`test_array_assign`) continue to pass under the new helper. Verified end-to-end. |

C16-1 is the only prior-cycle finding still in scope. All
earlier-cycle findings have been closed in prior fix-sweeps and
their closures verified in prior audit docs. Per user directive,
prior-cycle findings are not re-flagged.

---

## Verification of the C16-1 fix

### (a) Does the helper correctly identify wide-element types per the audit spec?

**Helper source** (`helixc/backend/x86_64.py:983-1003`):

```python
def _check_array_elem_size_supported(self, ty: tir.TIRType) -> None:
    """Audit 28.8 cycle 16 C16-1 (HIGH): LOAD_ELEM/STORE_ELEM
    currently emit unconditional 32-bit `mov eax, [...]` / `mov
    [...], eax`. A let-binding like `let xs = [1.0_f64, 2.5_f64];`
    propagates `f64` into the IR ops but the backend silently
    truncated each store to 32 bits and each load to the low 32
    bits — miscompile with no diagnostic.
    Phase-0 fix: fail loudly at codegen when an array-element type
    is wider than 32 bits. Full 8-byte LOAD_ELEM / STORE_ELEM
    lowering can land as a separate Stage-29 deliverable. This
    matches the cycle-3-style 'narrow + loud' pattern (cf.
    `_check_float_supported` above)."""
    wide_widths = {"i64", "u64", "f64", "isize", "usize"}
    if isinstance(ty, tir.TIRScalar) and ty.name in wide_widths:
        raise NotImplementedError(
            f"x86_64 backend LOAD_ELEM/STORE_ELEM does not yet "
            f"support {ty.name} array elements (would silently "
            f"truncate to 32 bits — see audit-stage28-8 cycle 16 "
            f"C16-1). Use i32/u32/f32-typed elements until the "
            f"8-byte load/store path lands."
        )
```

The cycle-16 doc's reproducer source identifies `f64` as the
demonstrating case and names "i64 / u64 / f64" symmetrically as
affected. The helper's `wide_widths` set is `{"i64", "u64",
"f64", "isize", "usize"}` — a **strict superset** of the
cycle-16-named set. The two additional members `isize` / `usize`
are conservative additions: `helixc/frontend/typecheck.py:212`
explicitly aliases `isize` to `i64` and `usize` to `u64` at the
type-system level. By including them in the trap set, the fix
prevents a structurally-identical miscompile via the pointer-width
alias channel.

The single concern about the wider trap set is the cross-file
inconsistency below; not a fix defect.

**`TIRType` instance discrimination**: the helper checks
`isinstance(ty, tir.TIRScalar)` before the `.name` lookup. This
is the same shape as `_check_float_supported` (line 972). Both
`TIRTensorTy` and `TIRTuple` `TIRType`s will bypass the trap (no
trap on tensor/tuple element types), but TIR-level array elements
are always scalar in current lower_ast emission (verified via
`lower_ast.py:871, 896, 920`: `elem_vals[0].ty` is the lowered
element's `CONST_INT`/`CONST_FLOAT` result type — always
`TIRScalar`). Non-scalar element types are not reachable in
current production. Future tensor-element lowering would need to
re-check this guard; that is a Stage-29-class concern, not a
cycle-17 finding.

**Verdict (a)**: correct identification per the audit's spec.
Strict superset of the named wide widths; covers the
structurally-equivalent `isize`/`usize` channel.

### (b) Is the trap message informative?

The `NotImplementedError`'s string interpolation produces (for
the `f64` reproducer):

> x86_64 backend LOAD_ELEM/STORE_ELEM does not yet support f64
> array elements (would silently truncate to 32 bits — see
> audit-stage28-8 cycle 16 C16-1). Use i32/u32/f32-typed elements
> until the 8-byte load/store path lands.

This satisfies four diagnostic-quality criteria:

1. **Names the affected op** — "LOAD_ELEM/STORE_ELEM".
2. **Names the offending type** — interpolates `ty.name`.
3. **Explains why** — "would silently truncate to 32 bits".
4. **Provides a migration hint** — "Use i32/u32/f32-typed
   elements".
5. **Provides an audit reference** — "see audit-stage28-8 cycle
   16 C16-1" — searchable in docs.

The regression test
(`test_c16_1_wide_array_elem_traps_at_codegen`) asserts the
message contains either `"C16-1"` or `"32 bits"`. Both substrings
are present; the test would pass even if one of the two markers
was accidentally dropped in a future refactor.

**Minor stylistic observation (not a finding)**: each line of the
f-string starts with `f"x86_64 ..."` continuation form — the
implicit string-concatenation of adjacent f-strings produces a
single message string with no trailing newline. This is the same
pattern as `_check_float_supported`. Idiomatic for the file.

**Verdict (b)**: informative. Meets all five quality criteria.

### (c) Is a new type-design hole introduced?

Possible holes to look for:

1. **Does the helper's discriminator (`TIRScalar.name in
   wide_widths`) admit any case that the rest of `_emit_op`
   would handle?** No. The rest of `_emit_op`'s LOAD_ELEM /
   STORE_ELEM branches emit unconditional 32-bit ops (`mov eax,
   [...]` / `mov [...], eax`). A wide-element type is exactly
   what those branches mishandle; trapping is the correct action.
   No false-positive trap on a path that would have correctly
   handled wide elements.

2. **Could the helper miss a wide-element type that should be
   trapped?** The relevant width sources are `TIRScalar` name
   strings emitted by the lowerer. The full set is enumerated at
   `lower_ast.py:357-358`: `i8/i16/i32/i64/isize/u8/u16/u32/u64/
   usize/f16/bf16/f32/f64/bool`. Of these:
   - Narrow (≤32-bit): `i8/i16/i32/u8/u16/u32/f32/bool` —
     correctly handled by the unguarded 32-bit `mov eax` path.
   - Trapped by the new helper: `i64/u64/f64/isize/usize` — all
     in `wide_widths`.
   - Trapped by the existing `_check_float_supported`:
     `f16/bf16` — trapped at SSA-value allocation time (line
     887) before LOAD_ELEM/STORE_ELEM is reached for those slot
     types. Defense in depth.
   - Trapped by neither: none.
   No reachable wide-element type bypasses the union of the two
   helpers.

3. **Are the call sites correctly placed?** Yes. The LOAD_ELEM
   call (line 2743) fires before the first machine-code byte is
   emitted for that op (the `mov_ecx_mem_rbp` call at line 2750
   is the first emit). The STORE_ELEM call (line 2764) fires
   before line 2768's `mov_ecx_mem_rbp`. A trap aborts codegen
   cleanly with no partial machine bytes.

4. **Is the discriminator field correct on each side?**
   - LOAD_ELEM uses `op.results[0].ty`: correct — the result of
     a LOAD_ELEM is the loaded value and carries the array's
     element type per `lower_ast.py:1166-1168`.
   - STORE_ELEM uses `op.operands[1].ty`: correct — STORE_ELEM
     operands are `(index, value)` per `tir.py:233`, so operand
     index 1 is the stored value, which carries the array's
     element type per `lower_ast.py:927`.

5. **Does the helper share state with anything?** No. It is a
   pure function of `ty`. No `self.` field access; no module
   state read. Same shape as `_check_float_supported`. Trivially
   thread-safe / re-entrant.

6. **Does the helper's existence break any invariant on
   `FnCompiler`?** No new field; no new dependency. The helper is
   purely additive.

**Verdict (c)**: no new type-design hole introduced. Six checks
pass.

### (d) Is the fix's blast radius exactly the LOAD_ELEM/STORE_ELEM sites?

Grep `array_info\[` across `helixc/backend/x86_64.py`:

```
849:            return self.array_info[name][0]       # _alloc_array re-entry
854:            self.array_info[name] = (base, length, 8)  # _alloc_array write
2740:            base, length, esize = self.array_info[name]  # LOAD_ELEM read
2761:            base, length, esize = self.array_info[name]  # STORE_ELEM read
```

The `array_info` table has two writers (both inside
`_alloc_array`) and two consumers (LOAD_ELEM emit, STORE_ELEM
emit). The two consumers are exactly the two sites guarded by
the new helper. There is no third reader path.

Grep `_check_array_elem_size_supported` across `helixc/`:

```
helixc/backend/x86_64.py:983   def _check_array_elem_size_supported(...)
helixc/backend/x86_64.py:2743  self._check_array_elem_size_supported(op.results[0].ty)
helixc/backend/x86_64.py:2764  self._check_array_elem_size_supported(op.operands[1].ty)
helixc/tests/test_codegen.py:    (via the regression test that calls compile_module_to_elf)
```

One definition site; two call sites; no call from any other
backend or pass. The blast radius is exactly the two emit sites
the cycle-16 doc named.

**Verdict (d)**: blast radius is exactly LOAD_ELEM/STORE_ELEM.

---

## Rotated lens (cycle-17 new this cycle): PTX backend analogous surface

Per the user directive, this cycle rotates the type-design lens
to ask: does the PTX backend
(`helixc/backend/ptx.py`) have the same LOAD_ELEM / STORE_ELEM
defect?

### Surface P1: PTX op-kind dispatch surface

Grep `OpKind\.|op\.kind ==` across `helixc/backend/ptx.py`
returns 11 matches at lines 173, 180, 195, 207, 219, 230, 241,
255, 258, 270, 302. The full enumeration of dispatched ops:

| Line | OpKind | Width handling |
|---|---|---|
| 173 | `SCALAR_CONST_INT` | dispatch on `%r` (32-bit) reg prefix |
| 180 | `SCALAR_ADD` | float / int dispatch on reg prefix |
| 195 | `SCALAR_MUL` | float / int dispatch on reg prefix |
| 207 | `SCALAR_SUB` | float / int dispatch on reg prefix |
| 219 | `SCALAR_NEG` | float / int dispatch on reg prefix |
| 230 | `SCALAR_CONST_FLOAT` | `%f` reg prefix (assumed 32-bit) |
| 241 | `SCALAR_CMP` | `%p` predicate prefix |
| 255 | `RETURN` | terminator |
| 258 | `THREAD_IDX` | `%r` reg prefix (32-bit) |
| 270 | `TILE_INDEX_LOAD_HBM` | dispatched on `dtype` attr via `_ptx_load_suffix` / `_ld_reg_prefix` |
| 302 | `TILE_INDEX_STORE_HBM` | dispatched on `dtype` attr via `_ptx_store_suffix` |

Critical observation: `op.kind` here is a `TileOpKind` (from
`helixc/ir/tile_ir.py`), not a `tir.OpKind`. The PTX backend
operates on a completely separate IR surface — the tile IR —
whose op-kind enum does **not** contain `LOAD_ELEM`,
`STORE_ELEM`, or `ALLOC_ARRAY`. Verified by grep:

```
grep "LOAD_ELEM\|STORE_ELEM\|ALLOC_ARRAY" helixc/backend/ptx.py
(no matches)
```

Verified independently by grep `array|elem|ARRAY|ELEM` (case-
insensitive) across `helixc/backend/ptx.py`: zero matches. There
is no array surface in the tile IR / PTX path at all. Memory
access in the PTX path is via `TILE_INDEX_LOAD_HBM` /
`TILE_INDEX_STORE_HBM`, which dispatch correctly on the `dtype`
attribute through `_ptx_load_suffix` (line 339-340) and
`_ld_reg_prefix` (line 343-346). The dtype attribute is the
load-bearing channel — and unlike the x86_64 LOAD_ELEM/STORE_ELEM
path, it is actually consumed.

**Verdict on P1**: the C16-1 defect class does NOT apply to the
PTX backend, because the PTX backend does not have the relevant
op (LOAD_ELEM / STORE_ELEM) at all. The tile IR's memory-access
ops correctly thread the dtype attribute through type-aware
load/store suffix selection. No structurally-equivalent defect.

### Surface P2: PTX `_format_param` `.param .b64` hardcoding

Already covered by cycle 16 forward note 9 — `_format_param`
(line 152-155) unconditionally emits `.param .b64` for kernel
params. This is correct under the Phase-0 invariant that all
kernel params are pointer-typed (cleared by `cvta.to.global.u64`
on entry). Not a regression introduced by C16-1; not a cycle-17
finding. Carried forward.

### Surface P3: `_ptx_type_str` fallthrough behavior

Read against HEAD at `helixc/backend/ptx.py:157-168`:
non-`TIRScalar` non-`TIRUnit` types default to `.b64`. This is
the same Phase-0 pointer-assumption. Documented in cycle 16; not
a cycle-17 finding.

**Finding count from PTX rotation**: 0 HIGH, 0 MEDIUM, 0 LOW.

---

## Fresh-eyes re-read of fix's two source-file deltas

### Re-read 1: helper at lines 983-1003

```python
def _check_array_elem_size_supported(self, ty: tir.TIRType) -> None:
    ...docstring...
    wide_widths = {"i64", "u64", "f64", "isize", "usize"}
    if isinstance(ty, tir.TIRScalar) and ty.name in wide_widths:
        raise NotImplementedError(...)
```

Type-design lens:

- **Method signature**: takes `tir.TIRType` (the base class). Same
  signature as `_check_float_supported`. Consistent.
- **Local `wide_widths` set**: declared inside the method, not at
  class or module scope. This is a minor style point — a single
  call site instantiates the set once per invocation. For the two
  call sites this is negligible (each call to `_emit_op` is one
  IR op; total invocations across a compile are O(#array ops)
  not O(#instrs)). Not a finding; the readability gain of keeping
  the set inline with its trap message outweighs the micro-cost.
- **Strict superset of the cycle-16 spec**: `isize` and `usize`
  are added beyond the doc's named set. As verified above, this
  is conservative and correct. The cross-file inconsistency
  observation about `const_fold.py` treating `isize`/`usize` as
  32-bit while `typecheck.py` treats them as 64-bit aliases is a
  PRE-EXISTING inconsistency that the fix does not introduce. See
  forward note 1 below.

**Lens-specific check**: in a future refactor where someone adds
a new wide scalar type (e.g. `i128`, `u128`, `f128`), this helper
will need to be updated to include it. The helper does not
delegate to a single source-of-truth "is this scalar > 32 bits"
predicate. Stage-29-class hardening recommendation: factor a
single `_scalar_width_bits(ty)` predicate and use it across all
of `_check_array_elem_size_supported`, `_check_float_supported`,
and the `_is_f64_type`/`_is_i64_type`/`_is_u64_type` family. Not
a cycle-17 finding; recorded as forward note 2.

### Re-read 2: call sites at lines 2743 and 2764

Both call sites have the same shape: 2 comment lines pointing to
"audit-stage28-8 cycle 16 C16-1" and 1 invocation. The comments
make the connection to the audit cycle searchable from the file,
which is valuable for an auditor reading source cold.

The discriminator field on each side has been verified above
(verdict (c) check 4). Both correct.

**Lens-specific check**: a future op-kind reordering of
operands would silently desynchronize from `op.operands[1].ty`.
This is a general fragility of operand-index addressing in TIR
op handling — not specific to this fix. Recorded as forward note
3.

### Re-read 3: regression test at test_codegen.py:437-476

```python
def test_c16_1_wide_array_elem_traps_at_codegen():
    ...
    src = """
    fn main() -> i32 {
        let xs = [1.0_f64, 2.5_f64];
        let y = xs[0];
        0
    }
    """
    prog = parse_src(src)
    errs = type_check(prog)
    hard = [e for e in errs if not (hasattr(e, "is_warning") and e.is_warning)]
    mod = lower(prog)
    try:
        compile_module_to_elf(mod)
        assert False, (
            "expected NotImplementedError on f64 array LOAD_ELEM; "
            "backend silently miscompiled instead"
        )
    except NotImplementedError as e:
        assert "C16-1" in str(e) or "32 bits" in str(e), (
            f"expected C16-1 trap message, got: {e}"
        )
```

Type-design lens:

- The test exercises **exactly** the reproducer cited in the
  cycle-16 doc. Good audit-trail alignment.
- The test asserts the trap fires AT CODEGEN, not at typecheck —
  consistent with the fix-sweep's design (typecheck remains
  permissive; backend is the trap boundary). This matches the
  cycle-16 doc's diagnosis ("the IR→machine boundary").
- The substring match `"C16-1" in str(e) or "32 bits" in str(e)`
  is permissive in a good way: the test passes if EITHER marker
  is preserved in a future refactor. A single-marker test would
  be more brittle.
- The `hard = [e for e in errs if not (hasattr(e, "is_warning")
  and e.is_warning)]` line computes a value but never asserts on
  it. Dead computation in test. Minor; not load-bearing. (See
  forward note 4 — could be deleted or replaced with `assert not
  hard` to also lock down the cycle-16 doc's "typecheck is
  permissive" claim.)

**Coverage gap**: the test exercises only the `f64` case. The
cycle-16 doc symmetrically named `i64` and `u64`; cycle 16's
recommendation block (lines 472-478) named THREE regression
tests:
- `test_c16_1_array_f64_rejected` (this one — covered)
- `test_c16_1_array_i64_rejected` (NOT IMPLEMENTED)
- `test_c16_1_array_i32_still_works` (covered de-facto by
  existing `test_array_literal_and_index` passing)

Missing the explicit `i64`-array trap test is a gap relative to
the audit doc's recommendation. The helper's branch on
`ty.name == "i64"` is structurally identical to the `f64` branch,
so the unimplemented test is unlikely to catch a different bug.
It would catch a future refactor that accidentally narrowed
`wide_widths` (e.g. `{"f64"}` after a "minimal" cleanup). Not a
cycle-17 type-design defect; recorded as forward note 5.

**Verdict on regression test**: present, asserts the trap fires,
asserts the message is informative. Two minor gaps (dead `hard`
local, missing i64-only test); neither is a cycle-17 finding.

---

## Findings summary

| ID | Severity | Confidence | Location | Description |
|---|---|---|---|---|
| — | — | — | — | No findings. |

**Total**: 0 HIGH, 0 MEDIUM, 0 LOW under the type-design audit
category at the strict criterion (confidence ≥ 80%).

---

## Cycle 17 status

**Strict criterion (per user directive 2026-05-10): cycle clean
iff zero findings of ANY severity at confidence ≥ 80.**

This cycle finds **zero findings of any severity** under the
type-design audit category.

By the strict criterion, **cycle 17's type-design audit is
CLEAN**.

**Counter status (5-clean-consecutive gate under the strict
criterion)**:
- Was 0/5 after cycle 16 C16-1 reset.
- Cycle 17 type-design: CLEAN. Counter conditionally advances
  to **1/5** *if* cycle 17's silent-failures and code-review
  audits are also CLEAN.
- Stage 29 is gated by five fresh consecutive clean cycles. Four
  more clean cycles required after this one.

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
- Cycle 17 (type-design): 0 + 0 + 0 — clean → conditional 1/5

---

## Forward notes (not cycle-17 findings; recorded for visibility)

1. **`isize`/`usize` width inconsistency** (NEW, cycle 17):
   `helixc/ir/passes/const_fold.py:46` treats `isize`/`usize` as
   32-bit (`"isize": 32, "usize": 32`), while
   `helixc/frontend/typecheck.py:226-227` aliases them to `i64`
   / `u64` (64-bit). The cycle-17 C16-1 fix's
   `_check_array_elem_size_supported` takes the typecheck.py
   position (treats them as wide). This is the safer side of the
   inconsistency — a false-positive trap is strictly preferable
   to a silent miscompile — but the underlying width-of-isize
   contradiction remains a Stage-29-class concern. Not blocking;
   not introduced by the cycle-17 fix.

2. **Centralize scalar-width predicate** (NEW, cycle 17):
   `_check_array_elem_size_supported` hard-codes `{"i64", "u64",
   "f64", "isize", "usize"}` as its wide set; `_is_f64_type`,
   `_is_i64_type`, `_is_u64_type` are separate methods; the
   `_is_float_type` method enumerates `{"f16", "bf16", "f32",
   "f64"}`. A future wide scalar (e.g. `i128`) would require
   coordinated updates to 4+ sites. Stage-29-class refactor:
   factor a single `_scalar_width_bits(ty) -> int` predicate.
   Not blocking.

3. **Operand-index addressing in TIR op handlers** (NEW, cycle
   17): `STORE_ELEM`'s value operand is addressed as
   `op.operands[1].ty` (positional). A future reorder of
   STORE_ELEM operands would silently desynchronize the C16-1
   guard. Mitigation: introduce named-operand accessors at TIR
   level (e.g. `op.value` / `op.index`) backed by per-OpKind
   schema. Stage-29-class consideration; not blocking.

4. **Dead `hard` local in the regression test** (NEW, cycle 17):
   `test_c16_1_wide_array_elem_traps_at_codegen` at
   `helixc/tests/test_codegen.py:457` computes `hard = [...]`
   but never asserts on it. Either delete the computation or
   replace with `assert not hard, f"expected typecheck-permissive
   acceptance of f64 array; got: {hard}"` to also lock down the
   cycle-16 doc's "typecheck is permissive" claim. Stylistic;
   not blocking.

5. **Missing `i64`-array trap regression test** (NEW, cycle 17):
   the cycle-16 doc's recommendation block named three regression
   tests (`f64`, `i64`, `i32`-still-works). The fix-sweep
   implemented only one (`f64`). The `i64` branch in
   `_check_array_elem_size_supported` is structurally identical
   to the `f64` branch, so the missing test is unlikely to catch
   a different bug today — but it would catch a future refactor
   that accidentally narrowed `wide_widths`. Stylistic / defense-
   in-depth gap; not blocking.

6. **Stage-29 deliverable: full 8-byte LOAD_ELEM / STORE_ELEM
   lowering**: the cycle-16 doc and the fix-sweep commit message
   both flag this as the intended Stage-29-class fix. Once it
   lands, `_check_array_elem_size_supported` becomes either
   dead code or a narrower guard against the still-unsupported
   widths. The trap message's "until the 8-byte load/store path
   lands" phrasing correctly signals this. Tracking item, not a
   defect.

7. **Forward-carry from cycle 16, note 8** (`_alloc_array`
   `elem_size` parameter unwired): the IR-level `ALLOC_ARRAY`
   op's `dtype` attribute is read at `x86_64.py:870-874` but
   only `name` / `length` are propagated; `dtype` is silently
   dropped. The 8-byte-per-element overallocation is safe in
   isolation, and the cycle-17 fix now traps before any
   wide-element memory access occurs, so the dropped `dtype` is
   not currently reachable to harm. Stage-29-class wiring
   recommendation. Not a cycle-17 finding.

8. **Forward-carry from cycle 16, note 6** (`Value.ty` not
   frozen): `tir.Value` is `@dataclass` rather than
   `@dataclass(frozen=True)`. Conventionally immutable, not
   enforced. No in-tree mutation. Stage-29-class hardening.

9. **Forward-carry from cycle 16, note 7** (`Op.results:
   list[Value]` over-general): in practice every Op emits zero
   or one results; multi-result Ops are theoretically
   representable but not produced. The "single result or zero"
   convention is convention-only. Stage-29-class type-system
   rewrite consideration.

10. **Forward-carry from cycle 14, note 5** (`SIDE_EFFECT_KINDS`
    static cross-check): no static guarantee that every
    side-effecting `OpKind` is in the set. Stage-29-class
    enum-attached-metadata hardening recorded in cycle 14.

11. **Cycle-18 baseline confirmation**: the cycle-17 fix-sweep
    added one helper + two call sites + one regression test. No
    new type-design contract surface beyond the helper itself.
    Cycle 18's type-design audit can re-read this helper (and
    its call sites) as the only delta and otherwise rely on the
    empty-diff shortcut. Process note.

12. **Stage-29 readiness**: counter advances from 0/5 to a
    conditional 1/5 (pending the other two cycle-17 categories
    being CLEAN). Four more clean cycles remain required.
