# Audit Stage 28.9 cycle 107 — Type design

## Header

- **Date**: 2026-05-12
- **HEAD**: `6af8a46` ("Stage 28.9 cycle-106 fix-sweep: 4+
  cycle-105 findings (full _is_i64_type sweep + cross-precision
  cast + unit + break/continue)").
- **Counter at start**: 0/5 (cycle-106 fix-sweep landed 4
  cycle-105 findings + 1 cross-cut, resetting the audit-gate
  counter per standard discipline).
- **Mode**: STRICT READ-ONLY on `helixc/`. The only file written
  by this audit is this document. No Edit calls.
- **Scope**: cycle-107 type-design rotation surfaces — five
  verification points covering the cycle-106 fix-sweep delta and
  cross-cutting structural type-class consistency:
  1. **Unit-type normalization sweep** — walk every TyUnit()
     construction site and every TyName("()") construction site;
     confirm no remaining path constructs TyPrim("()") or compares
     TyUnit() == TyPrim("()") via cross-class __eq__.
  2. **Float type taxonomy** — TyPrim("f64") vs TyPrim("f32"):
     verify the new cvtsd2ss / cvtss2sd CAST arms dispatch
     correctly on both directions and the width gates remain
     consistent across the full CAST dispatch chain.
  3. **IR lowering type-flow** — `lower_ast.py`: verify
     Break/Continue raise NotImplementedError (loud-fail) is the
     right type-design choice vs producing a Trap node; verify
     For non-Range raise is consistent with the same pattern.
  4. **Bootstrap parser.hx named-mode generic branches**
     (3217-3504, 3505-3658): type-class drift after Stage 28.13
     struct-lit work.
  5. **Cross-cutting** — typecheck's `PRIMITIVES` table vs the
     rest of the type-class universe (lower_ast
     `_PRIMITIVE_TYPE_NAMES`, `_FLOAT_PRIM_NAMES`,
     `_NUMERIC_INT_PRIMS`, `_NUMERIC_FLOAT_PRIMS`).
- **Deferred-known and NOT re-flagged**:
  - monomorphize `_mangle_ty` / hash_cons `_ast_equal` silent
    catchalls
  - typecheck/struct_mono pre-flatten in `check.py`
  - `autotune.collect_autotuned_fns` missing `iter_fn_decls`
  - `struct_mono.mangle_struct` collision
  - DT_BIND_NOW unused constant
  - raw-200 enumeration in parser.hx (cycle-7 deferred)
  - `_is_i64_type` sibling emit-site enumeration
    (BIT_AND/BIT_OR/BIT_XOR/SHL/SHR/BIT_NOT/NEG/SELECT/BR/RETURN/
    COND_BR/FFI_CALL) — cycle-101 deferred
  - `A.StrLit` IR lowering gap (cycle-95 deferred); the broader
    "non-Let `A.StructLit` / `A.CharLit` / `A.TileLit` catch-all
    silent-drop in `_lower_expr`" is in the same defect class and
    falls under the cycle-105 silent-failures audit scope, NOT
    type-design.
  - Cycle 1-106 findings (the cumulative deferred set; see
    cycles 104 / 105 / 106 for the full enumeration).
- **Bar**: PASS = ZERO new HIGH/CRITICAL findings at confidence
  >= 80%. (Bar tightened from cycle-105's >= 75% per the parent
  agent's cycle-107 instructions.)
- **Source delta since cycle-105 audit base** (HEAD `77e4b85`):
  the cycle-106 fix-sweep commit `6af8a46` touched
  `helixc/backend/x86_64.py` (+53/-9),
  `helixc/frontend/typecheck.py` (+15/-1),
  `helixc/ir/lower_ast.py` (+32/-9), and added regression tests
  in `helixc/tests/test_ir.py` (+120). Three new audit docs were
  also added by cycle-106.

## Methodology

Five verification points across the cycle-106 fix-sweep delta
and cross-cutting consistency. Each point is a targeted re-walk:
read the relevant module(s), trace every dispatch path through
the surface language, check for residual representational
duplication after the cycle-105 fixes, and cross-check against
the deferred-known list to confirm any candidate defect is fresh
and reachable (not merely a defense-in-depth observation).

## V1 — Unit-type normalization sweep (PASS)

### `_resolve_type` path

`typecheck.py:514-520` handles `TyName("()")` before the
`PRIMITIVES` check (line 521). The early-return at line 519-520
maps to `TyUnit()`. `PRIMITIVES` (lines 336-351) no longer
contains the string `"()"`, so even if the early-return were
removed, the fallthrough at line 521 would NOT match `"()"` and
the name would resolve to `TyUnknown` (a loud diagnostic rather
than a silent duplicate). Defense in depth is intact.

### TyName("()") construction sites

The only producer of `TyName("()")` is `parser.py:756` (inside
`_parse_tuple_type` when the tuple-type parser sees an empty
`()`). No other site constructs `TyName("()")`. Grep:

```
TyName(.*"()")  →
  parser.py:756              return ast.TyName(span=..., name="()")
```

Every other reference is a docstring or comment.

### TyPrim("()") construction sites

Greppping `TyPrim\(["']\(\)|TyName\(["']\(\)` confirms NO Python
code constructs `TyPrim("()")` anywhere in `helixc/`. The only
matches are the comments in `typecheck.py:344-345` and a docstring
in `tests/test_ir.py:393-395` describing the pre-fix defect. No
residual path exists.

### TyUnit() / TyUnit cross-class comparison

`TyUnit` (typecheck.py:101-103) is a frozen dataclass with no
fields. `TyPrim` (typecheck.py:38-...) is a frozen dataclass with
a `name: str` field. Cross-class dataclass `__eq__` returns
`False` whenever class identity differs — so even if a stray
`TyPrim("()")` were to materialize, `_compatible(TyUnit(),
TyPrim("()"))` would fall through to the final `a == b` check
(typecheck.py:2425) and return `False`. The cycle-106 fix removes
the producer of the divergent representation; the consumer remains
strict (which is the correct invariant for unrelated type
classes).

### IR-layer residual drift (observation, NOT flagged)

`lower_ast._lower_type` (lines 364-371) walks the **AST**
`A.TyNode` directly, not the resolved `Type` from the frontend.
For `A.TyName("()")` it dispatches to the generic TyName arm at
line 371: `return tir.TIRScalar(ty.name)` — producing
`TIRScalar("()")`.

Meanwhile the implicit-unit return path at line 441 produces
`tir.TIRUnit()`:

```
ret = self._lower_type(fn.return_ty) if fn.return_ty else tir.TIRUnit()
```

So source-typed `fn foo() -> () {...}` yields
`return_ty = TIRScalar("()")` and implicit-unit
`fn foo() {...}` yields `return_ty = TIRUnit()`. The IR-layer
duplication that the cycle-106 typecheck fix removed at the
frontend level is RECREATED at IR-lowering time.

**Reachability through the surface language**: both forms
typecheck (the frontend fix makes them converge on `TyUnit()`).
The IR-level divergence then propagates to:

- `lower_ast.py:573, 578` — `isinstance(ir_fn.return_ty,
  tir.TIRUnit)` is the gate for "emit `ret(None)` vs
  `ret(body_val)`". For the explicit-`-> ()` form this gate is
  False (return_ty is TIRScalar, not TIRUnit), so the lowerer
  attempts to emit `ret(body_val)` IF body_val is not None.
- `lower_ast.py:1714` — callee CALL's `result_ty` inherits the
  callee's return_ty. So a caller's CALL op result is typed
  `TIRScalar("()")` when the callee is explicit-`-> ()`.

**Observable backend behaviour:**

x86_64 RETURN arm (`x86_64.py:1954-1972`) treats any operand
typed as non-f64 / non-f32 / non-i64 as i32 (line 1964-1965:
`mov eax, [rbp+slot]`) — a `TIRScalar("()")`-typed operand is
not in the f64/f32/i64 sets, so it routes through `mov eax,
mem`. Operand-free RETURN (line 1967) routes through `mov eax,
0`. For unit-typed body shapes (final-expr is `()` literal, or
the body has no final expr), `body_val` is `None` so the
explicit form falls to line 583 `ret(None)`, identical to
implicit form. For shapes where final-expr lowering produces a
non-None i32 (e.g. `if c { () } else { () }` falls back to
const_int(0) at lines 1746/1753/1755), explicit form emits
`ret(const_int_0_value)` and implicit form emits `ret(None)`.
But both compile to the same `mov eax, ...; ret` machine code
because the const-0 value's slot is initialized to 0 by CONST_INT.

PTX backend `_ptx_type_str` at `ptx.py:157-176` maps TIRScalar
through a width table that doesn't include `"()"`; the default
`.b32` fallback fires (line 173). TIRUnit takes the empty-string
branch (line 174-175). So PTX `.func` signature would differ —
explicit form emits `.func (.b32 %retval) name(...)`, implicit
emits `.func name(...)`. The PTX backend is a minimal stub (the
body is always `ret;` regardless), so this is signature drift
without body divergence.

**Severity assessment**: the x86_64 path is semantically
identical for the two forms because the const-0 normalization
pre-zero-initializes the slot at CONST_INT and the RETURN arm
treats unknown-name TIRScalar as i32. The PTX path differs in
function-signature text but the body is a stub. No present
miscompile is reachable through the Phase-0 surface. The drift
is a latent defense-in-depth gap for any future pass that
consumes `return_ty` in a type-class-sensitive way.

Confidence in NOT being a present miscompile: ~85%. Confidence
that this is a fresh **observation** worth recording: 100%.
Confidence that this meets the HIGH/CRITICAL conf-80% bar as a
finding: < 60% (the cycle-106 fix was the surface fix; the
IR-layer recovery is a deferred-class duplication that should be
addressed in a separate cycle when the consumer set widens).

**Not flagged.** Recorded here for the cycle-108 / next-rotation
scope: walk `_lower_type` for `TyName("()")` and emit
`TIRUnit()` instead of `TIRScalar("()")` to converge with the
frontend fix. Defense-in-depth refactor; not a present bug.

### V1 verdict: PASS

The cycle-106 frontend fix is complete and the only TyName("()")
producer (`parser.py:756`) is routed through the normalizing
early-return in `_resolve_type`. No residual `TyPrim("()")`
construction site exists. The latent IR-layer drift is recorded
as an observation, not a finding.

## V2 — Float type taxonomy: f64 <-> f32 cross casts (PASS)

The cycle-106 fix inserted two CAST arms at `x86_64.py:1338-1347`:

```
1338  if from_is_f64 and to_is_float and not to_is_f64:
1339      self.asm.movsd_xmm0_mem_rbp(src_slot)
1340      self.asm.cvtsd2ss_xmm0_xmm0()
1341      self.asm.movss_mem_rbp_xmm0(res_slot)
1342      return
1343  if from_is_float and not from_is_f64 and to_is_f64:
1344      self.asm.movss_xmm0_mem_rbp(src_slot)
1345      self.asm.cvtss2sd_xmm0_xmm0()
1346      self.asm.movsd_mem_rbp_xmm0(res_slot)
1347      return
```

Walking the dispatch for the four representative pairs:

**(f64, f32)**: from_is_f64=T, to_is_float=T, to_is_f64=F → arm
1338 matches. Emits `cvtsd2ss xmm0, xmm0`. **Correct.**

**(f32, f64)**: from_is_float=T, from_is_f64=F, to_is_f64=T →
arm 1343 matches. Emits `cvtss2sd xmm0, xmm0`. **Correct.**

**(f64, f64)**: arm 1338 fails (to_is_f64=T blocks the `not
to_is_f64` clause). Arm 1343 fails (from_is_f64=T blocks the
`not from_is_f64`). Falls to arm 1349 (`from_is_f64 and
to_is_f64`) → emits 8-byte memory copy. **Correct (identity
cast).**

**(f32, f32)**: arm 1338 fails (from_is_f64=F). Arm 1343 fails
(to_is_f64=F). Arm 1349 fails (from_is_f64=F). Falls to arm 1355
(`from_is_float == to_is_float`) → emits 4-byte memory copy.
**Correct (identity cast).**

### Width gate consistency across the CAST dispatch chain

The full chain at `x86_64.py:1284-1358`:

| Arm | Condition | Emits |
|-----|-----------|-------|
| 1285 | from_is_i64, not to_is_float, not to_is_i64 | i64→i32 truncate |
| 1290 | not from_is_float, not from_is_i64, to_is_i64 | i32→i64 sign-extend |
| 1297 | from_is_i64, to_is_f64 | i64→f64 cvtsi2sd REX.W |
| 1304 | from_is_i64, to_is_i64 | i64→i64 8-byte copy |
| 1309 | not from_is_float, to_is_f64 | i32→f64 cvtsi2sd |
| 1315 | not from_is_float, to_is_float | i32→f32 cvtsi2ss |
| 1321 | from_is_f64, not to_is_float | f64→i32 cvttsd2si |
| 1327 | from_is_float, not to_is_float | f32→i32 cvttss2si |
| **1338** | **from_is_f64, to_is_float, not to_is_f64** | **f64→f32 cvtsd2ss (new)** |
| **1343** | **from_is_float, not from_is_f64, to_is_f64** | **f32→f64 cvtss2sd (new)** |
| 1349 | from_is_f64, to_is_f64 | f64→f64 8-byte copy |
| 1355 | from_is_float == to_is_float | f32→f32 / i32→i32 4-byte copy |

Each arm's predicate is mutually exclusive given the arm-ordering
discipline. The new arms (1338, 1343) are inserted ABOVE the
f64→f64 (1349) and same-class (1355) memory-copy fallbacks, so
they shadow the prior silent-miscompile path. The width
gates (`from_is_f64`, `to_is_f64`, `from_is_float`,
`to_is_float`, `from_is_i64`, `to_is_i64`) all derive from the
`_is_f64_type` / `_is_float_type` / `_is_i64_type` predicates at
x86_64.py:1017-1043, which are name-based on `TIRScalar.name`.

### bf16 / f16 / fp8 / mxfp4 / nvfp4 / ternary reachability

`_is_float_type` (line 1029) includes the quantized-float names,
so they would route through `from_is_float = True` /
`to_is_float = True`. However, `_check_float_supported` runs
upfront at lines 937, 941, 945 over every value's type, and
rejects f16/bf16/fp8/mxfp4/nvfp4/ternary with
`NotImplementedError`. So no CAST involving these types reaches
the dispatch chain — slot-allocation aborts first. Loud failure;
not a silent miscompile.

### V2 verdict: PASS

The new arms are correct, the dispatch chain is mutually
exclusive, the width gates derive from consistent predicates,
and the unsupported-float forms are gated by upfront
`_check_float_supported` calls.

## V3 — Break / Continue / non-Range For loud-fail (PASS)

### Break / Continue arms

`lower_ast.py:1901-1918` adds:

```python
if isinstance(expr, A.Break):
    raise NotImplementedError(
        f"break not yet supported at "
        f"{expr.span.line}:{expr.span.col}")
if isinstance(expr, A.Continue):
    raise NotImplementedError(
        f"continue not yet supported at "
        f"{expr.span.line}:{expr.span.col}")
```

The pre-fix defect was silent fallthrough to the catch-all
`return None` at line 2235 — a `loop { ...; if c { break; } }`
silently emitted an infinite loop because `A.Break` was
swallowed into None and the surrounding While/Loop lowering had
no break-target stack to consume it.

**Trap-node alternative considered**: `tir.OpKind.TRAP`
(tir.py:284) exists as a runtime panic op. The loud-fail design
could in principle emit a TRAP and let the program abort at
runtime when break/continue executes. However:
- TRAP at runtime defers the failure mode to user-time, where
  the bootstrap binary contains a hidden landmine. The user gets
  a SIGABRT or trap instruction firing with no obvious connection
  to the `break` they wrote.
- TRAP would still mis-shape the surrounding loop's CFG (the
  break path would jump to TRAP instead of the loop's exit
  block) — semantically wrong, not just diagnostic-poor.
- `NotImplementedError` at compile time matches the F1 cycle-105
  fix discipline: the parser accepts the surface form, the
  typechecker passes it through, the lowerer rejects with a
  position-bearing diagnostic. The Phase-0 contract is
  "compile-fail beats run-mis".

The chosen design is correct. The same rationale applies to the
non-Range For arm at `lower_ast.py:1787-1797`:

```python
if not isinstance(expr.iter_expr, A.Range) or \
        expr.iter_expr.start is None or expr.iter_expr.end is None:
    raise NotImplementedError(
        f"for-loop with non-Range iter not yet supported "
        f"at {expr.span.line}:{expr.span.col} "
        f"(iter expr: {type(expr.iter_expr).__name__})")
```

Pre-fix the body was silently lowered once with the iter-var
unbound; now it raises. Consistent loud-fail pattern.

### Typecheck side

`typecheck.py:1761`:

```python
if isinstance(expr, (A.Break, A.Continue, A.Return)):
    return TyUnit()
```

The typechecker assigns unit type to break/continue/return as
expressions. This is sound — these forms diverge (never produce
a value), so any type works as long as the surrounding context
is unit. The cycle-106 fix at the IR layer makes break/continue
fail loudly at lower-time; the typecheck pass is intentionally
permissive (no scope-validation that break/continue are inside
a loop) because the IR layer now catches the unsupported case.
A future stage that adds CFG support for break/continue should
also add a scope check at typecheck time; for now the IR
loud-fail is the safety net.

### V3 verdict: PASS

The Break/Continue/non-Range-For loud-fail design is the right
type-design choice: compile-time NotImplementedError with
position info beats a TRAP-at-runtime that produces a
mis-shaped CFG. Consistent with the F1/F2 silent-failure fix
discipline.

## V4 — Bootstrap parser.hx named-mode generic branches (PASS)

The two branches:

- **Generic mode** (parser.hx:3217-3504): `Pt<i32> { x: 10, y: 32 }`
  — generic struct use with monomorphization, then named struct
  literal body.
- **Non-generic mode** (parser.hx:3505-3658): `Pt { x: 10, y: 32 }`
  — direct struct literal.

Both branches share the same Stage-28.13 named-struct-literal
algorithm (`peek_named_struct_lit` guard, `temp_base` sentinel
array, `struct_tab_field_lookup`, trap 50040/50041/50042). The
implementation is duplicated rather than abstracted (per the
INC-3b atomicity comment at line 3338-3341: "INCREMENT 3
atomicity requires the mono setup AND the body in one place;
sharing would require restructuring the dispatch").

### Structural diff between the two branches

| Aspect | Generic (3217-3504) | Non-generic (3505-3658) |
|--------|--------------------|--------------------------|
| Struct-idx source | `mono_s_idx` (synthesized via clone) | `s_idx` (direct lookup) |
| Arity source | `arity_m = __arena_get(entry_m + 2)` | `arity = __arena_get(entry + 2)` |
| field-lookup key | `mono_s_idx` | `s_idx` |
| Empty-lit handling | Trap 50040 if arity_m != 0 | Trap 50040 if arity != 0 (via post-loop missing check) |
| Build chain | Same TUPLE_CONS shape | Same TUPLE_CONS shape |
| Set last_struct_idx | Yes, after parse | Yes, after parse |

`struct_tab_field_lookup` (parser.hx:1028-1051) walks the
fields-region at stride 3 from `fields_ptr = __arena_get(entry +
3)`. It is keyed by `struct_idx` (positional index into the
struct_tab base). For the mono'd case, INC-3b.2 clones the
fields region with the same stride-3 layout (parser.hx:3310-3334
explicit), so the lookup function works identically on both
forms. No type-class drift.

### Empty-lit cycle-3-F4 fix consistency

The cycle-3 F4 fix at parser.hx:3370-3389 (generic) explicitly
mirrors the non-generic Audit A1-F7 fix: empty `Pt<i32>{}`
emits trap 50040 when arity_m > 0 rather than silently returning
a 0-arity tuple-lit. The non-generic branch at line 3649 has the
same check (`if n != arity { mk_node(99, 50040, 0, 0) }`). Both
arms guard against under-supply.

### Type-args arity check (line 3291-3293)

The generic branch enforces `ta_count == gp_count_pre`. If a user
writes `Pt<i32, i64>` on a Pt<T> with arity 1 generic params, trap
62032 fires before the struct-lit body is parsed. If they write
`Pt<>` (no type args), `ta_count=0` and trap 62032 fires for
arity mismatch with `gp_count_pre=1`. Consistent enforcement.

### `gp_marker_is` / `gp_marker_base` for field substitution

Line 3319-3329: when cloning fields, if `f_struct_idx` is a
gp-marker (encoded as `200 + gp_idx`), substitute with the
matched type-arg's struct_idx (or -1 for scalar like i32). The
gp_marker encoding stays in the 200..207 range (struct_tab cap
is 8). The cycle-7 deferred raw-200 sites are noted at lines
247-260 — non-migrated by design, not a fresh finding.

### V4 verdict: PASS

The two branches are structurally consistent under Stage 28.13.
The mono-clone preserves the fields-region layout exactly so
`struct_tab_field_lookup` works identically. No type-class drift
introduced by the recent struct work.

## V5 — Cross-cutting: PRIMITIVES vs the type-class universe (PASS, with observation)

### Reference tables

`typecheck.PRIMITIVES` (typecheck.py:336-351), post-cycle-106:

```
{i8, i16, i32, i64, isize, u8, u16, u32, u64, usize,
 bool, char, bf16, f16, f32, f64,
 fp8, mxfp4, nvfp4, ternary}
```

(20 entries; `()` removed.)

`typecheck._FLOAT_PRIM_NAMES` (line 395-398):

```
{f16, bf16, f32, f64, fp8, mxfp4, nvfp4, ternary}
```

(8 entries, all float-domain primitives in PRIMITIVES.)

`typecheck._INT_PRIM_NAMES` (line 399-401):

```
{i8, u8, i16, u16, i32, u32, i64, u64, isize, usize}
```

(10 entries, all int-domain primitives in PRIMITIVES.)

`typecheck._NUMERIC_INT_PRIMS` (line 2151-2154): same set as
`_INT_PRIM_NAMES`.

`typecheck._NUMERIC_FLOAT_PRIMS` (line 2155-2157):

```
{f16, bf16, f32, f64, fp8, mxfp4, nvfp4}
```

(7 entries — `ternary` is intentionally excluded from
`_NUMERIC_FLOAT_PRIMS` per the cast-matrix intent: `ternary` is
a non-numeric quantized form, NOT cast-compatible with f32/f64.)

`lower_ast._PRIMITIVE_TYPE_NAMES` (lower_ast.py:356-362):

```
{i8, i16, i32, i64, isize, u8, u16, u32, u64, usize,
 bool, char, bf16, f16, f32, f64, unit}
```

(17 entries; `fp8`, `mxfp4`, `nvfp4`, `ternary` MISSING; `unit`
present instead of `()`.)

### Consistency analysis

Cross-checking the typecheck tables against each other:
- `_FLOAT_PRIM_NAMES ⊆ PRIMITIVES` ✓
- `_INT_PRIM_NAMES ⊆ PRIMITIVES` ✓
- `_NUMERIC_INT_PRIMS = _INT_PRIM_NAMES` ✓
- `_NUMERIC_FLOAT_PRIMS ⊂ _FLOAT_PRIM_NAMES` (ternary excluded) ✓
  with the documented rationale that ternary is non-numeric.

Cross-checking lower_ast vs typecheck:
- `_PRIMITIVE_TYPE_NAMES` is missing `fp8`, `mxfp4`, `nvfp4`,
  `ternary` (vs `PRIMITIVES`).
- `_PRIMITIVE_TYPE_NAMES` has `"unit"` (the string), whereas
  `PRIMITIVES` had `"()"` pre-fix and now has neither.

### Is the lower_ast / typecheck drift a finding?

`_PRIMITIVE_TYPE_NAMES` is defined at lower_ast.py:356 but is
**never read** — the only mention is its definition. Prior audits
(cycle-19, cycle-20, cycle-21, cycle-22, cycle-103) all
confirmed it is "membership only, no width logic", "out-of-class
membership only", and "Pre-existing, correct" for the purposes
the type-design audits cared about. Cycle-104 explicitly noted
the surface is in the deferred-class set.

The cycle-106 fix didn't touch `_PRIMITIVE_TYPE_NAMES`. Its stale
contents (missing quantized floats, has `"unit"` not `"()"`) are
a defense-in-depth observation: if a future commit started
*using* this set for any width or class gating, it would diverge
from `typecheck.PRIMITIVES`. But because the set has no consumer,
no surface-language program reaches a bug through it.

**Suggested defense-in-depth (NOT prescribed; not flagged)**:
either delete `_PRIMITIVE_TYPE_NAMES` (since it has no callers)
or sync its contents to `typecheck.PRIMITIVES` (replacing
`"unit"` with `"()"` and adding the four quantized floats). The
latter is the conservative move so that any future caller starts
from a consistent baseline.

### Confidence

`_PRIMITIVE_TYPE_NAMES` as a present miscompile vector: 0%
(no consumers).

`_PRIMITIVE_TYPE_NAMES` as a defense-in-depth observation:
recorded above; below the conf-80% HIGH bar.

### V5 verdict: PASS

The typecheck tables are internally consistent and consistent
with each other under the documented intent (`ternary` excluded
from cast-numeric). The lower_ast `_PRIMITIVE_TYPE_NAMES` drift
is dead code and not a present miscompile. Defense-in-depth
sync recorded as observation.

## Findings table

| ID | Severity | Confidence | Topic | Disposition |
|----|----------|------------|-------|-------------|

(No findings.)

## Observations (not flagged; below the conf-80% HIGH bar)

| ID | Confidence | Topic |
|----|------------|-------|
| O107-1 | ~70 | `lower_ast._lower_type(TyName("()"))` produces `TIRScalar("()")` rather than `TIRUnit()`, recreating the cycle-106-fixed frontend duplication at IR-lowering time. Semantically equivalent x86_64 output today; PTX signature drift in the minimal-stub path. Suggest converging via an explicit `if isinstance(ty, A.TyName) and ty.name == "()": return tir.TIRUnit()` arm above the generic-name fallback. |
| O107-2 | ~55 | `lower_ast._PRIMITIVE_TYPE_NAMES` (line 356-362) is dead code — no callers in `helixc/`. Its contents drift from `typecheck.PRIMITIVES` (missing fp8/mxfp4/nvfp4/ternary; contains `"unit"` not `"()"`). Either delete or sync. |

## Verdict

**Verdict: CLEAN** — counter advances 0/5 → 1/5.

- V1 PASS (unit-type normalization fix is complete on the
  typecheck side; only `parser.py:756` constructs `TyName("()")`,
  routed through the normalizing early-return; no
  `TyPrim("()")` producer exists anywhere; IR-layer
  `TIRScalar("()")` drift recorded as observation O107-1, below
  HIGH bar).
- V2 PASS (f64<->f32 CAST arms are correctly inserted, mutually
  exclusive with surrounding arms, and reach only
  `_check_float_supported`-permitted operand types f32/f64).
- V3 PASS (Break/Continue/non-Range-For loud-fail
  `NotImplementedError` is the right type-design choice vs
  runtime TRAP; consistent with F1/F2 silent-failure fix
  discipline).
- V4 PASS (parser.hx Stage 28.13 named-mode generic and
  non-generic struct-lit branches are structurally consistent;
  `struct_tab_field_lookup` works identically on the cloned
  fields-region under mono).
- V5 PASS (typecheck `PRIMITIVES` / `_FLOAT_PRIM_NAMES` /
  `_INT_PRIM_NAMES` / `_NUMERIC_INT_PRIMS` / `_NUMERIC_FLOAT_PRIMS`
  tables are internally and mutually consistent under the
  documented `ternary`-excluded-from-numeric-cast invariant;
  `lower_ast._PRIMITIVE_TYPE_NAMES` drift is dead code and
  recorded as observation O107-2, below HIGH bar).

**Stage 28.9 audit-gate counter advances 0/5 → 1/5.** Cycle-107
clean pass on the cycle-106 fix-sweep delta and the cross-cutting
type-class consistency surfaces. The cycle-105 → cycle-106
fix-and-re-audit cycle is now closed; counter resumes its
ascending run.

## Cross-reference to cycles 101-106

- **Cycle 101**: PASS, 0 findings (cmp dispatch + parser
  tuple/array literal + fdce).
- **Cycle 102**: fix-sweep, 4 cycle-101 findings (ADD/SUB/MUL
  u64 + regression tests).
- **Cycle 103**: PASS, 0 findings on cycle-102 delta. Counter
  0 → 1.
- **Cycle 104**: PASS, 0 findings on parser.hx
  struct_gp_tab / gp_marker / named struct-lit / arith dispatch
  matrix. Counter 1 → 2.
- **Cycle 105**: FAIL, 1 finding F105-1 (f64<->f32 CAST silent
  miscompile). Counter 2 → 0.
- **Cycle 106**: fix-sweep, 4+ cycle-105 findings (full
  _is_i64_type sweep + cross-precision cast + unit + break/
  continue + non-Range for).
- **Cycle 107** (this doc): PASS, 0 findings on the cycle-106
  fix-sweep delta + cross-cutting type-class consistency.
  Counter 0 → 1.

Deferred items unchanged from cycle-105:
- monomorphize._mangle_ty hash discipline.
- hash_cons._ast_equal structural-recursion guard.
- struct_mono pre-flatten ordering.
- autotune iter_fn_decls scoping.
- A.StrLit IR lowering gap (and the broader non-Let
  StructLit/CharLit/TileLit catch-all silent-drop in
  `_lower_expr`).
- DIV / MOD / SHR signed-vs-unsigned dispatch (under cycle-101
  cycle-103 audited subset; broader sites enumeration deferred).
- struct_mono.mangle_struct collision risk.
- BITCAST u64/usize defense-in-depth (now fixed in cycle-106
  for the `wide` classifier; the cycle-105 observation is
  CLOSED).
- _is_i64_type sibling emit-site enumeration (BIT_AND / BIT_OR /
  BIT_XOR / SHL / SHR / BIT_NOT / NEG / SELECT / BR / RETURN /
  COND_BR / FFI_CALL) — cycle-101 deferred.
- raw-200 enumeration in parser.hx (cycle-7 deferred).
- DT_BIND_NOW unused constant.

No edits to source performed. This document is the only file
written by cycle-107.
