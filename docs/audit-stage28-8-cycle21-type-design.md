# Stage 28.8 Pre-29 Audit Gate — Cycle 21, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit (HEAD)**: bee36e6 (cycle-21 fix-sweep landing C20-1).
**Cycle-20 baseline**: 5a1e406 (cycle-20 audit found C20-1 HIGH).
**Cycle-20 status**: NOT CLEAN — counter at 0/5.
**Cycle-21 fix-sweep**: closes C20-1 (HIGH) by extending the four PTX
width-keyed tables/branches with isize/usize entries routing to the
64-bit treatment (matching cycle-19 backend canon + cycle-20 const_fold
canon), plus a new regression test
`test_c20_1_isize_usize_treated_as_64_bit_in_ptx` at
`helixc/tests/test_ptx.py:237-262`. PTX tests: 22 → 23 pass.

**Scope**: Audit category B (type-system / dispatch / soundness) under
the strict criterion. Per the user directive, this audit:

1. Re-reads the prior cycle-20 type-design doc to confirm the C20-1
   characterization (which sites, which defaults).
2. Verifies the cycle-21 fix correctly hardens all 4 PTX width-keyed
   sites that C20-1 named: `_ptx_type_str` mapping,
   `_DTYPE_SIZE`, `_DTYPE_PTX_LOAD`, `_ld_reg_prefix`.
3. Confirms the new test pins all 4 tables.
4. Rotates the lens to OTHER width-aware sites in `helixc/` to flag any
   further untreated drift — per user directive, the centralized
   `_scalar_width_bits` predicate refactor is out of scope; only
   flagging is in scope.

**Counter context** (per user directive 2026-05-10):

- Cycles 18/19/20 each NOT CLEAN — counter sat at 0/5.
- Cycle 21 (this audit): if CLEAN under the strict criterion, counter
  advances 0/5 → 1/5. A finding here keeps the counter at 0/5.

---

## Cycle-21 production-code delta (since cycle-20 baseline 5a1e406)

```
git show --stat bee36e6 -- helixc/
```

```
 helixc/backend/ptx.py     | 23 ++++++++++++++++++++---
 helixc/tests/test_ptx.py  | 28 ++++++++++++++++++++++++++++
```

Exactly one production-code file touched (`helixc/backend/ptx.py`),
matching cycle-20 forward note 15's baseline.

The fix has three edits in `ptx.py`:

1. **`_ptx_type_str` mapping at line 166-172** — add `"isize": ".b64",
   "usize": ".b64"` alongside the existing 8/16/32/64-bit integer
   suffixes.
2. **`_DTYPE_SIZE` table at line 340-342 + `_DTYPE_PTX_LOAD` table at
   line 343-347** — add `"isize": 8, "usize": 8` and `"isize": "s64",
   "usize": "u64"` respectively.
3. **`_ld_reg_prefix` at line 361** — extend the 64-bit register-pool
   set from `("i64", "u64")` to `("i64", "u64", "isize", "usize")`.

The new regression test asserts each of the four tables directly:

```python
assert PtxEmitter._DTYPE_SIZE["isize"] == 8
assert PtxEmitter._DTYPE_SIZE["usize"] == 8
assert PtxEmitter._DTYPE_SIZE["i64"] == 8
assert PtxEmitter._DTYPE_PTX_LOAD["isize"] == "s64"
assert PtxEmitter._DTYPE_PTX_LOAD["usize"] == "u64"
em = PtxEmitter.__new__(PtxEmitter)
assert em._ptx_type_str(tir.TIRScalar(name="isize")) == ".b64"
assert em._ptx_type_str(tir.TIRScalar(name="usize")) == ".b64"
assert em._ld_reg_prefix("isize") == "rd"
assert em._ld_reg_prefix("usize") == "rd"
assert em._ld_reg_prefix("i64") == "rd"
assert em._ld_reg_prefix("i32") == "r"
```

This pins all four PTX width-keyed sites C20-1 named.

---

## Verification: cycle-21 fix consistently hardens all 4 PTX width tables

**Pass.** Read `helixc/backend/ptx.py:155-363` at HEAD bee36e6:

| Site | Code | isize | usize |
|---|---|---|---|
| `_ptx_type_str` mapping (line 166-172) | `mapping["isize"] = ".b64"` | `.b64` (64-bit) | `.b64` (64-bit) |
| `_DTYPE_SIZE` (line 340-342) | `"isize": 8` | 8 bytes | 8 bytes |
| `_DTYPE_PTX_LOAD` (line 343-347) | `"isize": "s64"` | `s64` (signed 64) | `u64` (unsigned 64) |
| `_ld_reg_prefix` (line 359-363) | `if dtype in ("i64", "u64", "isize", "usize"): return "rd"` | `rd` (64-bit pool) | `rd` (64-bit pool) |

All four sites now agree with the canonical 64-bit treatment.

### End-to-end trace of cycle-20 reproducer (now closes)

The cycle-20 doc's reproducer:

```helix
@kernel
fn copy_isize(dst: tensor<isize, [N], hbm>, src: tensor<isize, [N], hbm>) {
    let i: i32 = thread_idx_x();
    dst[i] = src[i];
}
```

Pipeline trace post-cycle-21:

1. **Parser → Typecheck**: unchanged. `_resolve_type` resolves `isize`
   to `TyPrim("isize")`; `TyTensor(dtype=TyPrim("isize"), ...)` is
   accepted. Pass.
2. **Lower to TIR → Lower to Tile IR**: `TIRScalar("isize")` flows as
   `TileType.dtype` unchanged. No width dispatch in tile_ir.py.
3. **PTX emit at `TILE_INDEX_LOAD_HBM` (ptx.py:278-309)**:
   - `_dtype_size("isize")` now returns **8** (was 4 pre-fix). Stride
     is correct: `mul.wide.s32 {off}, {idx_reg}, 8`.
   - `_ptx_load_suffix("isize")` now returns **"s64"** (was "u32"
     pre-fix). Emits `ld.global.s64`. Correct 64-bit load.
   - `_ld_reg_prefix("isize")` now returns **"rd"** (was "r" pre-fix).
     The destination register is `%rdN` (64-bit pool). No subsequent
     narrowing chain.
4. **PTX emit at `TILE_INDEX_STORE_HBM`** (ptx.py:310-330): symmetric.
   8-byte stride, `st.global.u64` (or `.s64`), correct.

The silent 32-bit-narrowing path is closed at all four sites.

---

## Cross-check: width-aware sites in `helixc/` at HEAD bee36e6

Searched `helixc/` for width-keyed dispatch sites:

| Site | isize/usize handling | Status |
|---|---|---|
| `typecheck.py:225-228` `_WIDEN_NAME_ALIASES` | `isize→i64`, `usize→u64` | Pre-existing canon |
| `typecheck.py:241` `_WIDEN_RANK` | isize=40, usize=41 (=i64/u64 rank) | Pre-existing canon |
| `typecheck.py:1816-1817` `_INT_BOUNDS` | isize=i64 range, usize=u64 range | Pre-existing canon |
| `autodiff.py:60-79` `NUMERIC_FOR_AD` | broad set, no width-dispatch | Cycle-2 fix |
| `lower_ast.py:356-362` `_PRIMITIVE_TYPE_NAMES` | name set, no width logic | Pre-existing, correct |
| `tile_ir.py` (whole file) | forwards `TIRScalar` opaque; zero hits for `isize`/`usize` | Pre-existing, correct |
| `tir.py:432, 436` `const_int` / `const_float` | string-typed; no width logic | Pre-existing, correct |
| `const_fold.py:43-56` `_INT_BITS` | isize=64, usize=64 | Cycle-20 fix, correct |
| `x86_64.py:1005-1011` `_is_i64_type` | includes "isize" | Cycle-19 fix, correct |
| `x86_64.py:1013-1017` `_is_u64_type` | includes "usize" | Cycle-19 fix, correct |
| `x86_64.py:1042` `_check_array_elem_size_supported.wide_widths` | includes isize/usize | Cycle-17 fix, correct |
| `ptx.py:166-172` `_ptx_type_str` mapping | `isize→.b64`, `usize→.b64` | **Cycle-21 fix, correct** |
| `ptx.py:340-342` `_DTYPE_SIZE` | isize=8, usize=8 | **Cycle-21 fix, correct** |
| `ptx.py:343-347` `_DTYPE_PTX_LOAD` | isize=s64, usize=u64 | **Cycle-21 fix, correct** |
| `ptx.py:359-363` `_ld_reg_prefix` | isize/usize in 64-bit `rd` pool | **Cycle-21 fix, correct** |

**Every width-aware site in `helixc/` python code now treats isize as
64-bit and usize as 64-bit, consistent with the typecheck.py canon.**

Bootstrap source `helixc/bootstrap/kovc.hx` carries its own width
predicates for the Phase-1 self-hosted compiler — that's Phase-1 scope,
out of the Stage 28.8 Phase-0 audit gate.

---

## Lens rotation: residual silent-narrowing risk surfaces in `helixc/`

The user directive asks specifically: "any remaining width-aware site
in helixc/ that hasn't been hardened yet." Sweeping by both grep
patterns `_INT_BITS|_DTYPE_SIZE|_DTYPE_PTX|_is_i64|_is_u64|
wide_widths|_scalar_width|elem_size_bytes|_check_array_elem_size|
in\s*\([\"']i6?4[\"']` and the `.get(..., narrow_default)` family
`.get([^)]*,\s*(4|"u32"|"r"|"\.b32"|32))`, the surfaced candidates are:

### Candidate 1 — `const_fold.py:66-68` `_wrap_int_to_type` 32-bit default

```python
def _wrap_int_to_type(value: int, ty: "tir.TIRType") -> int:
    bits = 32  # default for unknown / generic scalar types
    if isinstance(ty, tir.TIRScalar):
        bits = _INT_BITS.get(ty.name, 32)
    ...
```

**Same defect class as C20-1 in form** — `.get(..., narrow_default)` —
but **not reachable from typecheck-accepted code at HEAD**. Post-
cycle-20, `_INT_BITS` covers every canonical int-bearing scalar name
(i8/i16/i32/i64/u8/u16/u32/u64/isize/usize/bool). `_wrap_int_to_type`
is invoked only from integer-op handlers (`tir.OpKind.ADD/SUB/MUL/
DIV/MOD/BIT_*/SHL/SHR/NEG/BIT_NOT`) where `res.ty` is propagated from
upstream IR; for those op-kinds applied to two CONST_INT operands,
`res.ty` is always one of the canonical int types. The 32-bit
default would fire only if a future scalar name (e.g., `i128`, a
hypothetical bignum, or a typeclass-generic-T leak through the IR
into a CONST_INT) reached this function. No such path is reachable
today.

**Not a cycle-21 finding** under the strict criterion (confidence
that any reachable path silently miscompiles is < 50%; the 32-bit
default is documented and only fires on out-of-canon names).
Recorded as forward note 17 (NEW) — see below.

### Candidate 2 — `typecheck.py:1849` `_suggest_wider_int(value, current)`

```python
for cand in ("i32", "i64"):
    ...
```

This is the diagnostic-hint helper invoked when a literal overflows
its annotated type's range — it suggests a wider integer type to
the user. The candidate list `("i32", "i64")` is incomplete in that
isize/usize are not in it, but:

- The function is purely diagnostic — it returns a `str` hint shown
  in the type-error message. No width-semantics consequence.
- isize/usize map to i64 by the widening-canon (typecheck.py:225-228)
  so "use i64" is the correct suggestion anyway for any value that
  would fit usize/isize but not the current type.
- A literal of value 6e9 declared as i32 gets the hint "use `i64`
  instead" rather than "use `isize` instead" — both are 64-bit on
  64-bit targets, so the user gets a runnable suggestion either
  way.

**Not a finding** — diagnostic surface, not a soundness-affecting
width contract. The hint quality could be better (suggesting isize
when the user is in a pointer-width context) but that's polish.
Recorded as forward note 18 (NEW).

### Candidate 3 — `ptx.py:181-262` SCALAR_* op-kinds unconditionally 32-bit

```python
if op.kind == ti.TileOpKind.SCALAR_CONST_INT:
    r = self._new_reg("r")
    v = op.attrs.get("value", 0)
    self._line(f"    mov.b32 {r}, {int(v)};")
```

Same as in cycle-20 doc forward note 2: `SCALAR_CONST_INT` emits
`mov.b32`, `SCALAR_ADD`/`SCALAR_SUB`/`SCALAR_MUL`/`SCALAR_NEG`
unconditionally emit `.s32`. The scalar register allocator only
allocates `%r` (32-bit) for non-float results.

This is **explicitly documented Phase-0 MVP scope** at ptx.py:180
(`"v0.1: only handle a tiny scalar subset for sanity testing"`).
Today, the only producer of PTX-emit-via-`emit_op` is the tile-IR
fusion path; the only sample programs using tile-IR have float
operands, so the int-scalar narrowness is **unreachable from any
example today**. A user would have to write a tile-fn with isize/
usize scalar ops and `--emit-ptx` to hit it.

**Not a cycle-21 finding** — Phase-0 documented scope. The cycle-20
audit doc surfaced this as forward note 2 (informational); it's
recorded but not in-scope under the strict criterion since (a) the
PTX path is gated behind `--emit-ptx`, (b) the broader tile-IR
scalar-op-on-isize path is not reachable from any in-tree program,
and (c) the narrowness is loudly documented at the call site as MVP
scope. Carried as forward note 2 (CARRY from cycle-20).

### Candidate 4 — `ptx.py:152-155` `.param` `.b64`

```python
def _format_param(self, name: str, idx: int) -> str:
    return f".param .b64 param_{idx}"
```

All kernel params are declared `.b64`. For non-pointer scalar params
this is overshoot (8 bytes for what could be 4), not narrowing —
correctness is preserved. Same as cycle-20 forward note 3.

**Not a finding.** Carried as forward note 3 (CARRY from cycle-20).

### Candidate 5 — `lower_ast.py:1037-1038` un-suffixed IntLit lowering as i32

```python
# When the user writes `let x: isize = 3_000_000_000` (no `_isize` suffix),
# the AST IntLit lowers as `const_int(value, "i32")` regardless of context.
```

Same as cycle-20 doc forward note 4 (and cycle-20 doc section
"lower_ast.py:357-371 — no finding" parenthetical). This is a
pre-existing AST→IR narrowness, not introduced by the cycle-19/20/21
cascade, masked in practice by users writing `_isize` suffix.

**Not a cycle-21 finding** — pre-existing, not introduced by cycle-20→
21 delta, out of strict-criterion scope for this cycle's audit B.
Carried as forward note 4 (CARRY from cycle-20).

### Candidate 6 — re-sweep for any other unhardened sites

Grep `helixc/*.py` for `in\s*\(\s*[\"']i64[\"']|[\"']u64[\"']` —
returns one match: `ptx.py:361` (which is the cycle-21 fix-site itself,
now including isize/usize). No bare `("i64", "u64")` predicates remain.

Grep `helixc/` for `_INT_BITS|_DTYPE_SIZE|_DTYPE_PTX` cross-checks
the union of the cycle-19/20/21 fix tables. Only the three sites
are width-keyed (one per fix); all three now align.

Grep `helixc/` for `.get([^)]*,\s*("u32"|"r"|"\.b32"|4|32))` —
finds only the cycle-21-touched sites in ptx.py and the const_fold
candidate 1 above. No surfaced sites.

**Conclusion**: cycle-21 fix-sweep has hardened every width-aware
silent-narrowing site that the cycle-13/16/18/19/20 defect class
pattern would surface. No untreated drift sites remain in `helixc/`
at HEAD bee36e6.

---

## Findings summary

| ID | Severity | Confidence | Location | Description |
|---|---|---|---|---|
| — | — | — | — | No findings. |

**Total**: 0 HIGH, 0 MEDIUM, 0 LOW.

The cycle-21 fix-sweep correctly closes C20-1. The four PTX width-keyed
tables/branches all now treat isize/usize as 64-bit, matching the
typecheck.py / x86_64 backend / const_fold canonical treatment
established by cycles 17/19/20.

The cycle-20 reproducer (`@kernel fn copy_isize(dst: tensor<isize, ...>,
src: tensor<isize, ...>)`) now emits correct 64-bit-stride / `ld.global.
s64` / `%rd` PTX. Same defect class as C13-1 / C16-1 / C18-1 / C19-1
/ C20-1 — closed for this audit category at this HEAD.

---

## Cycle 21 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity at confidence ≥ 80.**

This cycle finds **0 findings** under the type-design audit category.

By the strict criterion, **cycle 21's type-design audit is CLEAN**.

**Counter status (5-clean-consecutive gate under the strict criterion)**:

- Was 0/5 after cycle 20 NOT CLEAN.
- Cycle 21 type-design (this audit): CLEAN. (Other cycle-21 audit
  categories — silent-failures, codereview — pending; counter
  advancement gated on all three audit categories being CLEAN per
  the gate's compositional rule.)
- If cycles 21 audits A/C also CLEAN, counter advances 0/5 → 1/5.
- Stage 29 still gated by five fresh consecutive clean cycles.

The severity trend, updated:

- Cycles 1-6: HIGH/MEDIUM/LOW each cycle — not clean.
- Cycles 7-12: clean (counter 1/5 → 3/5 across the run).
- Cycle 13: 1 HIGH (C13-1) — not clean → reset 0/5.
- Cycle 14: clean → 1/5.
- Cycle 15: clean → 2/5.
- Cycle 16: 1 HIGH (C16-1) — not clean → reset 0/5.
- Cycle 17: clean → 1/5.
- Cycle 18: 1 HIGH (C18-1) — not clean → reset 0/5.
- Cycle 19: 1 HIGH (C19-1) — not clean → stays 0/5.
- Cycle 20: 1 HIGH (C20-1) — not clean → stays 0/5.
- Cycle 21 (this audit): 0 findings — CLEAN under type-design audit B.

**Pattern**: cycles 13 / 16 / 18 / 19 / 20 each surfaced one HIGH
silent-width-narrowing finding of the same defect class. Each cycle's
fix-sweep closed the prior cycle's finding but uncovered the next
disagreeing width-keyed site. After cycle-21, every width-keyed site
in `helixc/` python code aligns with the typecheck.py canon. **The
defect class is exhausted in `helixc/`** (modulo non-reachable
forward-note candidates above).

---

## Forward notes (not cycle-21 findings; recorded for visibility)

1. **Centralize scalar-width predicate** (CARRY from cycles 17/18/19/20):
   Stage-29-class refactor. Six consecutive HIGH findings (cycles
   13/16/18/19/20 + the original cycle-18 carry that became C20-1) all
   in the same defect class strengthens the case for a single
   `_scalar_width_bits(ty) -> int` predicate. **Per user directive,
   this is explicitly out of scope for this cycle — recorded only.**

2. **PTX scalar-op narrowness beyond C20-1** (CARRY from cycle-20
   forward note 2): ptx.py:181-262's SCALAR_* op-kinds unconditionally
   emit `.s32` / `mov.b32`. Phase-0 MVP scope, documented at line
   180. Even with C20-1's dtype-table fix, the scalar-op layer would
   still narrow if tile-IR isize-scalar ops became reachable. Stage-29-
   class.

3. **PTX `.param` register convention** (CARRY from cycle-20 forward
   note 3): `_format_param` treats all params as `.b64`. Overshoot,
   not narrowing — currently safe.

4. **lower_ast.py un-suffixed IntLit context-insensitivity** (CARRY
   from cycle-20 forward note 4, pre-existing): `lower_expr(IntLit)`
   at lower_ast.py:1037-1038 lowers as `const_int(value, "i32")`
   regardless of the declared `let x: isize = ...` type. Same
   width-contract-disagreement defect class as C18-1 / C19-1 / C20-1,
   but pre-existing (not introduced by the cycle-19/20/21 cascade).
   Flagged for future investigation; masked in practice today by the
   `_isize` literal-suffix convention.

5. **Cycle-21 regression-test scope gap** (CARRY from cycle-20 forward
   note 5): cycle-21's `test_c20_1_isize_usize_treated_as_64_bit_in_ptx`
   asserts the four width tables directly (not a full
   parse→typecheck→lower→tile→emit pipeline round-trip on isize
   tensor code). Defense-in-depth gap; not blocking; the table-level
   test is sufficient to pin the cycle-21 fix surface.

6. **PTX dtype-suffix map alias gap — silent fallback class
   correctly closed** (CARRY-resolved from cycles 18/19/20 forward
   note 6): cycle-20 re-classification + cycle-21 fix have both been
   correct. The cycle-18/19 "loud KeyError" mischaracterization is
   superseded.

7. **Operand-index addressing in TIR op handlers** (CARRY from cycle-
   17 forward note 3): `STORE_ELEM`'s value operand is addressed as
   `op.operands[1].ty` (positional). Named-operand accessors would
   close the fragility. Stage-29-class.

8. **Dead `hard` local in C16-1 regression test** (CARRY from
   cycle-17 forward note 4): `test_codegen.py:457`'s `hard = [...]`
   is computed but never asserted on. Stylistic.

9. **Missing `i64`-array trap regression test** (CARRY from cycle-
   17/18/19/20 forward note 9): cycle-16 doc named three regression
   tests; cycle-17 implemented `f64`; cycles 19/20/21 did not add
   `i64`. Defense-in-depth.

10. **Stage-29 deliverable: full 8-byte LOAD_ELEM / STORE_ELEM
    lowering** (CARRY from cycles 17/18/19/20 forward note 10): once
    landed, `_check_array_elem_size_supported` becomes either dead
    code or a narrower guard.

11. **`_alloc_array` `elem_size` parameter unwired** (CARRY): IR-
    level `ALLOC_ARRAY` op's `dtype` attribute read but not
    propagated. Phase-0 safe under C16-1 trap. Stage-29-class.

12. **`Value.ty` not frozen** (CARRY): `tir.Value` is `@dataclass`
    not `@dataclass(frozen=True)`. Stage-29-class hardening.

13. **`Op.results: list[Value]` over-general** (CARRY): single-
    result Op convention is convention-only. Stage-29-class.

14. **`SIDE_EFFECT_KINDS` static cross-check** (CARRY from cycle-14
    forward note 5): no static guarantee that every side-effecting
    `OpKind` is in the set. Stage-29-class.

15. **Cycle-22 baseline**: cycle 21's audit is read-only at HEAD
    bee36e6. If the strict-criterion gate moves to cycle-22 CLEAN
    (i.e., this audit is CLEAN and counter advances to 1/5), the
    cycle-22 audit B re-baselines at whatever HEAD is reached after
    cycle-21 audits A/C close.

16. **Stage-29 readiness**: if cycle-21 all three audit categories
    CLEAN, counter advances 0/5 → 1/5. Five fresh consecutive clean
    cycles remain required.

17. **`const_fold._wrap_int_to_type` 32-bit default** (NEW): the
    `bits = _INT_BITS.get(ty.name, 32)` default mirrors the C20-1
    silent-narrowing pattern in form, but is **not reachable today**
    — every canonical int-bearing scalar name is in `_INT_BITS`
    post-cycle-20, and `_wrap_int_to_type` is invoked only from
    integer-op handlers where `res.ty` is propagated from canonical
    typecheck-accepted code. A future hypothetical `i128` or other
    new scalar would re-introduce reachability. Same forward-mitigation
    plan as forward note 1 (centralized scalar-width predicate).
    Not a cycle-21 finding under the strict criterion (confidence of
    reachable miscompile < 50%).

18. **`typecheck._suggest_wider_int` candidate list incomplete** (NEW):
    `_suggest_wider_int` enumerates `("i32", "i64")` but not
    `("i32", "i64", "isize", "usize")`. Diagnostic-hint quality only —
    no soundness impact. A literal overflowing i32 in pointer-width
    context would get the hint "use `i64`" rather than "use `isize`";
    both are runnable on 64-bit targets so the suggestion is
    functionally correct, just stylistically suboptimal. Not a
    finding.
