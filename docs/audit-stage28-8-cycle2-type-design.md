# Stage 28.8 Pre-29 Audit Gate — Cycle 2, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-10
**Commit**: 0171616 (read-only)
**Scope**: Re-audit type-system soundness focused on cycle-1's 14 fixes
(commits edf498e, 37efec7, ec0387e, 9a36cac, b43d15c, 37c655c, b091a63,
1cb9961) and their interactions. New TyQuote variant, `_widen_diff_inner`
+ `_WIDEN_RANK` table, `_check_cast_compat` matrix, `_SizeLitMarker`,
`DuplicateMethodError`, closure-trap 76003, all eight cycle-1 trap-id
reservations, and cross-stage interactions (Logic+AD, Quote+mono,
parametric-struct+unsafe).
**Method**: traced each new code path through resolve / substitute /
compatible / propagate / codegen. Wrote probe scripts exercising the
binop, cast, mono, AD-warning paths with edge-case inputs. Cross-checked
each trap-id has a real caller (not just a reservation). Ran the full
`helixc/tests/` suite as a baseline (no regressions).

**Result**: 11 new findings (3 HIGH, 5 MEDIUM, 3 LOW). Two of the
HIGH-severity issues are direct soundness regressions introduced by
cycle-1 fixes themselves: (a) `_WIDEN_RANK` is missing `fp8` / `mxfp4` /
`nvfp4` / `char` so `D<fp8> + D<i64>` widens to `i64` — a float silently
becomes an int with no AD warning fired in the int direction; (b) the
new closure capture gate trap 76003 only fires when the captured var is
**explicitly type-annotated** — the dominant Helix idiom `let pi = 3.14;`
(no annotation) leaves the var absent from `var_type_tab` so
`var_type_tab_lookup` returns -1, `cap_ty_tag > 0` is false, the trap
is silent, and bit truncation of the f64 to its low 32 bits is restored.
The third HIGH is the `TyQuote` variant being half-finished — there is
no syntactic path to spell `Quote<T>` as a type annotation (it resolves
to `TyUnknown(hint="generic Quote")`), so `fn unbox(q: Quote<i32>)` is
silently called with any value through the typecheck — TyQuote exists
only at expression-typing time.

Other findings are: WIDEN_RANK left-wins on same-rank pairs (`u32 vs
i32` -> `u32`, `i64 vs u64` -> `i64`, `bool vs u8` -> `bool` — sign /
signedness silently dropped); ref-to-ref cast matrix passes any inner
type pair (`&i64 as &f32` silently OK); the mixed-inner widening warning
fires ONLY when both sides are `TyDiff` — `D<f64> + i32` returns
`D<f64>` with no warning even though the same hazard exists; `check.py`
runs `monomorphize_structs` but NOT `flatten_impls`, so duplicate-method
trap 74002 is unreachable via the surface `helixc check` command; the
shape walker doesn't substitute `TyArray.size`; AD's numeric-Cast list
omits `bool`/`char`/`fp8`/`mxfp4`/`nvfp4`; the `_check_call_basic`
boundary check fires once per param so on a multi-param mismatch the
diagnostics duplicate.

Zero new HIGH/CRITICAL findings rise to "stop-the-line" status given
cycle 1's already-reset baseline — the three HIGHs above all need to be
addressed but they're regressions in the partial fixes, not new
soundness gaps. **Cycle 2 status**: 3 new HIGH means cycle 2 does **not**
count clean. See "Cycle 2 status" final paragraph.

---

## Summary table

| ID  | Severity | Component                           | Issue (short)                                                        |
|-----|----------|-------------------------------------|----------------------------------------------------------------------|
| C1  | HIGH     | typecheck `_WIDEN_RANK`             | Missing fp8/mxfp4/nvfp4/char — D<fp8> + D<i64> -> i64 (float to int) |
| C2  | HIGH     | bootstrap parser.hx:1810            | Closure trap 76003 silent on untyped captures (literal-typed lets)   |
| C3  | HIGH     | typecheck `TyQuote`                 | No resolve path for `Quote<T>` annotation — only expr-side variant   |
| C4  | MEDIUM   | typecheck `_WIDEN_RANK`             | Same-rank pairs left-wins: u32/i32, i64/u64, bool/u8 — silent flips  |
| C5  | MEDIUM   | typecheck `_check_cast_compat`      | Ref-to-ref allows any inner pair (&i64 as &f32 silent)               |
| C6  | MEDIUM   | typecheck binop                     | D<T> + bareT widening warning gated; only fires for D+D mixed inner  |
| C7  | MEDIUM   | check.py                            | `flatten_impls` not invoked — trap 74002 unreachable via check       |
| C8  | MEDIUM   | mono `substitute_ty`                | TyArray.size never substituted — `[T; N]` mono'd with N=8 keeps N    |
| C9  | LOW      | autodiff Cast arms                  | Numeric-cast list omits bool/char/fp8/mxfp4/nvfp4 — false AD warns   |
| C10 | LOW      | typecheck `_check_call_basic`       | Logic provenance check duplicates diagnostic on per-param mismatch   |
| C11 | LOW      | mono `_subst_shape_expr`            | Binary shape exprs substituted but not folded — `N*2` -> `64*2`      |

---

## Per-finding sections

### Finding C1: `_WIDEN_RANK` table missing fp8 / mxfp4 / nvfp4 / char

**File**: `helixc/frontend/typecheck.py:197-205` (`_WIDEN_RANK` dict)
and `helixc/frontend/typecheck.py:208-224` (`_widen_diff_inner`).
**Severity**: HIGH
**Category**: type soundness / silent precision loss

**Description**:
`_WIDEN_RANK` (cycle-1 B13 fix) covers exactly nine entries:
`bool` `i8` `u8` `i16` `u16` `i32` `u32` `i64` `u64` `isize` `usize`
`f16` `bf16` `f32` `f64`. The token set in `_NUMERIC_FLOAT_PRIMS` (line
1736-1738) lists `f16, bf16, f32, f64, fp8, mxfp4, nvfp4` — three more
float types Helix recognizes as numeric. None of those three are in
`_WIDEN_RANK`. `_widen_diff_inner` line 219 does `_WIDEN_RANK.get(a.name,
-1)`; the default -1 means any rank-having type beats them.

So `D<fp8> + D<i64>` calls `_widen_diff_inner(TyPrim("fp8"), TyPrim("i64"))`,
`ra=-1, rb=40`, the float silently becomes `i64`. Confirmed by probe:

```
D<fp8> + D<i64>:  widens to i64 (with warning, but warning says
                  "widened to i64", masking that this was a float
                  collapsing into an int).
```

`char` is also missing from `_WIDEN_RANK` but is accepted by
`_is_numeric_scalar` for cast purposes. `D<char> + D<i32>` widens to
`i32` (codepoint silently treated as integer). The warning fires but
calls a fundamentally wrong direction "widening".

**Reproducer**:
```python
from helixc.frontend.typecheck import _widen_diff_inner, TyPrim
_widen_diff_inner(TyPrim("fp8"), TyPrim("i64")).name   # "i64"
_widen_diff_inner(TyPrim("mxfp4"), TyPrim("f64")).name # "f64" (accidentally right)
_widen_diff_inner(TyPrim("nvfp4"), TyPrim("f16")).name # "f16" (loses fp4 quant)
_widen_diff_inner(TyPrim("char"), TyPrim("i32")).name  # "i32"
```

**Recommended fix**:
1. Add the missing entries with semantically-justified ranks. fp8 / mxfp4
   / nvfp4 are quantized floats; they should be at most rank `f16` (50)
   and *higher* than any integer rank.
2. Reject (don't widen) cross-domain pairs in `_widen_diff_inner` —
   `float vs int` should be a hard 24200 error not a widening, because
   AD over an int-valued tape is undefined.
3. Alternatively: emit a stronger warning that names the *kind*
   transition ("float→int via D-binop widening") so the user can spot
   the precision loss.

---

### Finding C2: Closure-capture trap 76003 silent on untyped captures

**File**: `helixc/bootstrap/parser.hx:1808-1812`
(`var_type_tab_lookup` -> `if cap_ty_tag > 0`).
**Severity**: HIGH
**Category**: silent failure regression on cycle-1 partial fix

**Description**:
Cycle-1 B4 fix probes the captured variable's type tag via
`var_type_tab_lookup(sb, ns, nl)`. The lookup (line 698-715) returns -1
when the name has no entry in `var_type_tab`. Entries are added by
`var_type_tab_add` at exactly one site, line 2255:

```
if after_name_tag == 14 {              // ':' present (typed let)
    ...
    let_ty_tag = ty_ident_to_tag(...);
}
if let_ty_tag >= 0 {
    var_type_tab_add(sb, name_start, name_len, let_ty_tag);
}
```

So only **explicitly-annotated** let bindings get tracked. The Helix
idiom `let pi = 3.14;` (untyped, even with a numeric suffix like
`3.14_f64`) leaves `pi` absent from `var_type_tab`. Then
`var_type_tab_lookup` returns -1, the guard `if cap_ty_tag > 0` is
false (because -1 is not > 0), the trap does not fire, and the closure
captures `pi`'s low 32 bits as i32 — exactly the silent corruption the
trap was supposed to block.

The pre-fix commit message acknowledges this gap as "The full fix
(stride-3 cl_capture_tab + per-capture type tag) is deferred to a later
cycle." But the deferral leaves the trap so narrowly applicable that it
fires only on the rare case of a user who annotates every let. In
practice, idiomatic untyped lets bypass the trap entirely.

**Reproducer**:
```helix
fn main() -> i32 {
    let pi = 3.14_f64;             // untyped — not in var_type_tab
    let c = |x: i32| x + (pi as i32);  // capture is silent — no 76003
    c(7)                              // result reads pi's low 32 bits
}
```

**Recommended fix**:
Either (a) commit the full stride-3 `cl_capture_tab` migration this
cycle, or (b) widen the gate so untyped-but-typeable lets are also
flagged — e.g. inspect the let's RHS literal kind (FloatLit, IntLit
with suffix, struct lit) to infer a tag at let-add time and populate
`var_type_tab` accordingly. Option (b) is the smaller change; (a) is
the correct long-term answer.

---

### Finding C3: `TyQuote` variant has no resolve path — `Quote<T>` annotations silently accept any value

**File**: `helixc/frontend/typecheck.py:183-191` (TyQuote def);
`helixc/frontend/typecheck.py:456-490` (`_resolve_type` TyGeneric arm);
`helixc/frontend/typecheck.py:1419-1436` (Quote/Splice handlers).
**Severity**: HIGH
**Category**: type soundness / half-finished variant

**Description**:
Cycle-1 B10 added a new `TyQuote(inner)` variant returned by the
expression-time Quote handler. `_fmt` line 1842 has an arm to render
`Quote<T>`. But three other necessary touchpoints are missing:

1. `_resolve_type` (line 456-490) has arms for `D<T>`, `Logic<T>`,
   `WorkingMem<T>`, `EpisodicMem<T>`, `SemanticMem<T>`, `ProceduralMem<T>`,
   and user generic structs. There is **no arm for `Quote<T>`**. So a
   parameter annotation `q: Quote<i32>` resolves to
   `TyUnknown(hint="generic Quote")`.
2. `_compatible` (line 1797-1806) has no arm for TyQuote. The
   fall-through `a == b` works for `TyQuote(TyPrim("i32")) ==
   TyQuote(TyPrim("i32"))` because both are frozen dataclasses, but
   the moment one side is `TyUnknown` (from a `Quote<T>` annotation)
   the call accepts any value.
3. `substitute_ty` in monomorphize.py has no Quote arm. `Quote<T>` in
   a generic fn signature wouldn't substitute.

Result: `fn unbox(q: Quote<i32>) -> i32 { splice(q) }` typechecks. Then
`unbox(99)` (passing a raw `i32`, not a Quote) also typechecks — the
parameter type is TyUnknown which accepts anything. Confirmed via probe:

```
fn unbox(q: Quote<i32>) -> i32 { splice(q) }
fn main() -> i32 {
    let bad: i32 = 99;
    unbox(bad)        // accepted — no diagnostic
}
```

The Splice inside `unbox` then runs on an i32 value, emitting trap
11001 only because the inner type is `TyUnknown` (which the Splice
handler treats as cascade-safe). Actually because of line 1426-1429,
TyUnknown bypasses the 11001 emission — so the user gets a clean
compile of nonsense code.

**Recommended fix**:
1. Add a `_resolve_type` arm: `if ty.base == "Quote" and len(ty.args) ==
   1: return TyQuote(inner=self._resolve_type(ty.args[0], scope))`.
2. Add a `_compatible` arm for TyQuote requiring inner-recursive match.
3. Decide on a Splice-of-TyUnknown policy: either still emit 11001 (so
   the cascade-safe path doesn't silently launder a non-Quote), or
   keep it cascade-safe but emit a sub-diagnostic referring to the
   unbound parameter.

---

### Finding C4: WIDEN_RANK same-rank pairs are left-wins — signedness silently dropped

**File**: `helixc/frontend/typecheck.py:197-224`.
**Severity**: MEDIUM
**Category**: type soundness / silent sign-domain transition

**Description**:
`_WIDEN_RANK` assigns identical ranks to signed/unsigned pairs of the
same width: `i8/u8 = 10`, `i16/u16 = 20`, `i32/u32 = 30`, `i64/u64 = 40`,
`isize/usize = 40`, `f16/bf16 = 50`, also `bool` shares rank 10 with
i8/u8. `_widen_diff_inner` line 221 picks left on tie: `return a if
ra >= rb else b`. So the result depends entirely on operand order:

```
_widen_diff_inner(u32, i32)  -> u32   (i32 silently treated as u32)
_widen_diff_inner(i64, u64)  -> i64   (u64 silently treated as i64)
_widen_diff_inner(bool, u8)  -> bool  (u8 silently treated as bool)
_widen_diff_inner(bool, i8)  -> bool  (i8 silently treated as bool!)
_widen_diff_inner(isize, usize) -> isize
```

The warning text says "widened to X" — but for these pairs there is no
true widening, just a sign-bit reinterpretation. A user writing
`D<u32> + D<i32>` who reads "widened to u32" might not realize the
right operand's sign is being silently dropped.

**Recommended fix**:
Either give signed and unsigned widths slightly different ranks (e.g.
signed `i32` = 30, unsigned `u32` = 31, so widening always picks
unsigned in a same-width tie — matching C's promotion rule), or make
same-rank pairs a hard error 24200 rather than silently picking left.

---

### Finding C5: `_check_cast_compat` ref-to-ref allows any inner pair

**File**: `helixc/frontend/typecheck.py:1771-1772`.
**Severity**: MEDIUM
**Category**: type soundness / Cast matrix gap

**Description**:
The Cast-compat matrix accepts `&T -> &U` for any T, U:

```python
if isinstance(src, TyRef) and isinstance(tgt, TyRef):
    return
```

The comment justifies this as "mut bit is borrow-check's concern" —
which is fine for the `&T` vs `&mut T` axis but not for the inner-type
axis. `&i64 as &f32` is just as much an invalid cast as `i64 as f32`
would be (it isn't — int-to-float is allowed) but `&Pt as &Line` is
clearly wrong and the matrix silently accepts it. Confirmed:

```python
src = TyRef(inner=TyPrim('i64'), is_mut=False)
tgt = TyRef(inner=TyPrim('f32'), is_mut=False)
tc._check_cast_compat(src, tgt, span)
# tc.errors == []
```

A user writing `let r: &Pt = (&line) as &Pt` would expect a diagnostic;
they get silent acceptance and then a SEGV at the first field access.

**Recommended fix**:
Recurse: `if isinstance(src, TyRef) and isinstance(tgt, TyRef): return
self._check_cast_compat(src.inner, tgt.inner, span)`. The recursion
naturally permits `&i32 as &i64` (the inner is in the numeric matrix)
while rejecting `&Pt as &Line`.

---

### Finding C6: Mixed-inner widening warning gated to `l_is_diff AND r_is_diff` — misses `D<T> + bareT`

**File**: `helixc/frontend/typecheck.py:1078-1085`.
**Severity**: MEDIUM
**Category**: type soundness / AD warning coverage

**Description**:
The widen-then-warn path fires only when both sides are TyDiff:

```python
if (l_is_diff and r_is_diff
        and l_inner != r_inner
        and not isinstance(l_inner, TyUnknown)
        and not isinstance(r_inner, TyUnknown)):
    inner = _widen_diff_inner(l_inner, r_inner)
    self._ad_warn_mixed_inner(...)
else:
    inner = l_inner if not isinstance(l_inner, TyUnknown) else r_inner
```

So `D<f64> + i32` (left is D, right is bare i32) returns `D<f64>` with
no warning — the i32 is silently promoted to f64 through the AD binop.
This is the same hazard as `D<f64> + D<i32>` but with a quieter
interface. Confirmed via probe:

```
D<f64> + i32 -> D<f64>     (no warning)
D<f64> + D<i32> -> D<f64>  (warning)
```

The cycle-1 docstring rationalized this as "mixed-inner promotion
between D and non-D is the design intent" — but the integer-into-D
direction has the same precision-loss / sign-flip hazards as D-vs-D.

**Recommended fix**:
Drop the `l_is_diff and r_is_diff` gate; fire the warning whenever
the inner-types differ AND at least one side is TyDiff. Or split into
two warning classes — D-D mixed (24200) vs D-bareT (e.g. 24201).

Similarly, `TyLogic + TyLogic` with different inner is also silent
right now (`Logic<i32> + Logic<i64>` -> `Logic<i32>` left-wins, no
warning). Whether to warn there depends on Logic semantics — but the
asymmetry with TyDiff is worth flagging.

---

### Finding C7: `check.py` runs `monomorphize_structs` but not `flatten_impls`

**File**: `helixc/check.py:284-296` (struct mono wired);
`helixc/backend/x86_64.py:3012` (flatten_impls wired here only).
**Severity**: MEDIUM
**Category**: pipeline gap / partial wiring

**Description**:
Cycle-1 wired `monomorphize_structs` into `check.py` after typecheck
(line 290-291). But `flatten_impls` (which raises `DuplicateMethodError`
trap 74002 per B11) is only invoked in `backend/x86_64.py:3012` — not
in `check.py`. The user-facing `helixc check foo.hx` command therefore
never sees the impl-flatten pass, and trap 74002 is unreachable via
that entry point.

Two structs each declaring a same-named method (e.g. `impl Pt { fn
area(...) }` + `impl Line { fn area(...) }`) typecheck clean via
`helixc check`. Only the backend codegen path raises. A user iterating
with `check` won't see the diagnostic until they try to actually emit
binary.

**Recommended fix**:
Invoke `flatten_impls(prog)` in `check.py` between typecheck and
`monomorphize_structs`. Wrap with try/except DuplicateMethodError so
the diagnostic surfaces as a structured error (not a raw Python
traceback). This also makes the check pipeline match the backend
pipeline in terms of which errors are seen.

---

### Finding C8: `substitute_ty` doesn't walk `TyArray.size`

**File**: `helixc/frontend/monomorphize.py:145-146`.
**Severity**: MEDIUM
**Category**: monomorphization gap (similar to cycle-1 B8)

**Description**:
The cycle-1 B8 fix added shape substitution for TyTile and TyTensor:

```python
if isinstance(t, A.TyTensor):
    new_shape = [_subst_shape_expr(s, subst) for s in t.shape]
    return A.TyTensor(... shape=new_shape ...)
```

But the TyArray arm (line 145-146) was left unchanged:

```python
if isinstance(t, A.TyArray):
    return A.TyArray(span=t.span, elem=substitute_ty(t.elem, subst),
                     size=t.size)              # ← not substituted!
```

So `fn copy<T, const N: usize>(buf: [T; N]) -> [T; N]` mono'd with
T=f64, N=8 produces a clone with `elem=TyName('f64')` but
`size=Name('N')` — exactly the pre-fix B8 bug for tensors/tiles, now
preserved for arrays. Confirmed via probe:

```python
arr = A.TyArray(elem=TyName('T'), size=A.Name('N'))
subst = {'T': TyName('f64'), 'N': _SizeLitMarker(8)}
substitute_ty(arr, subst).size  # still Name('N')
```

**Recommended fix**:
Apply `_subst_shape_expr` to `t.size`:
```python
if isinstance(t, A.TyArray):
    return A.TyArray(span=t.span,
                     elem=substitute_ty(t.elem, subst),
                     size=_subst_shape_expr(t.size, subst))
```

Same change needed in `struct_mono._ty_key`'s TyArray arm (line 330-331),
which currently encodes only the element type so `[T; 16]` and
`[T; 32]` produce identical keys — collapsing distinct mono'd structs.

---

### Finding C9: AD Cast arms omit bool / char / fp8 / mxfp4 / nvfp4

**File**: `helixc/frontend/autodiff.py:569-579`;
`helixc/frontend/autodiff_reverse.py:384-395`.
**Severity**: LOW
**Category**: AD warning false-positive

**Description**:
Cycle-1 B5 added Cast arms that propagate AD through numeric casts.
The numeric-type list is hardcoded:

```python
if isinstance(tgt, A.TyName) and tgt.name in (
    "i8", "i16", "i32", "i64", "u8", "u16", "u32", "u64",
    "isize", "usize", "f16", "bf16", "f32", "f64",
):
```

This omits `bool`, `char`, `fp8`, `mxfp4`, `nvfp4` — types that the
typecheck Cast matrix accepts as numeric. So `let q = x as bool` (valid
per the matrix) inside a `grad`-rewritten function emits a spurious
85001 warning even though the user wrote a clean type-checking program.

This is a low-severity, but the asymmetry between the type-system's
numeric set and AD's numeric set is the kind of thing that drifts.

**Recommended fix**:
Pull the numeric-type set into a shared frozenset (e.g.
`_NUMERIC_FOR_AD = _NUMERIC_INT_PRIMS | _NUMERIC_FLOAT_PRIMS |
_NUMERIC_BOOL_PRIMS`) and use it in both AD passes. Add a note on
which casts AD propagates trivially (1) vs which need special handling
(bool->int probably needs differentiability=0 since it's discontinuous).

---

### Finding C10: Logic provenance check duplicates diagnostic on per-param mismatch

**File**: `helixc/frontend/typecheck.py:581-600`.
**Severity**: LOW
**Category**: diagnostic UX

**Description**:
`_check_call_basic` calls `_check_logic_provenance_boundary` inside the
zip loop over params (line 599). Each iteration emits an
independent diagnostic. So `lift(a, b, c, d)` where four Logic-typed
params receive four non-Logic args produces four separate "trap 24100"
errors at the same call.span (because the helper uses `call.span` not
a per-arg span). The user sees four identical-looking diagnostics.

**Recommended fix**:
Either (a) thread the actual argument span (from `expr.args[i].span`)
through `_check_logic_provenance_boundary` so each diagnostic points
at its actual arg, or (b) aggregate the per-param failures and emit a
single batched diagnostic. (a) is more useful but requires plumbing the
arg spans through `_check_call_basic`'s call signature.

---

### Finding C11: Binary shape exprs substituted but not folded — `N*2` -> `64*2`

**File**: `helixc/frontend/monomorphize.py:63-107`.
**Severity**: LOW
**Category**: optimization gap

**Description**:
`_subst_shape_expr` walks Binary and Unary shape exprs and substitutes
Name leaves, but the result is left as a Binary node — `[N*2; 16]`
becomes `[IntLit(64)*IntLit(2); 16]`. If downstream lower-ast.py still
defaults non-IntLit shapes to 0 (the original B8 bug), this fails.

Confirmed via probe:
```python
shape = [Binary('*', Name('N'), IntLit(2))]
substituted shape[0] = Binary('*', IntLit(64), IntLit(2))   # not folded
```

**Recommended fix**:
Call `presburger` or `hash_cons.fold_const` on the substituted shape
expr. Or, more conservatively, add a `_fold_intlit_arith(expr)` helper
that folds Binary(IntLit, IntLit) -> IntLit one level deep.

---

## Cycle 1 fix re-verification

| Fix | Status     | Notes                                                                                                          |
|-----|------------|----------------------------------------------------------------------------------------------------------------|
| B1  | OK         | `monomorphize_structs` wired in check.py:290 and backend/x86_64.py:3017. typecheck `_resolve_type` extended.   |
| B2  | OK         | Trap 24100 fires both directions (Logic→raw, raw→Logic). Logic+Diff binop composes correctly per docstring.    |
| B3  | OK         | Trap 28603 fires. Unsafe-depth counter on TypeChecker properly inc/dec around UnsafeBlock.                     |
| B4  | DEGRADED   | See finding C2 — trap 76003 silent on untyped lets, which is the dominant case.                                |
| B5  | OK         | `_DIFF_WARNINGS` channel drains via `take_diff_warnings()`. Cast/UnsafeBlock/Quote/Splice/Modify arms in both. |
| B6  | OK         | substitute_ty TyPtr arm preserves is_mut. Tested in probe.                                                     |
| B7  | OK         | instantiate preserves is_extern + deep-copies where_clauses.                                                    |
| B8  | PARTIAL    | TyTensor/TyTile shape sub works for Name and Binary; TyArray.size NOT substituted (finding C8).                |
| B9  | OK         | flatten/unflatten cycle guards present. is_diff_leaf added. Trap 26003 fires from both pytree paths.           |
| B10 | DEGRADED   | TyQuote variant exists but resolve/compatible/substitute paths missing (finding C3).                           |
| B11 | OK*        | DuplicateMethodError fires from flatten_impls — but check.py doesn't invoke flatten_impls (finding C7).         |
| B12 | OK         | _walk_subst_expr UnsafeBlock arm recurses correctly into body.                                                  |
| B13 | DEGRADED   | _WIDEN_RANK missing fp8/mxfp4/nvfp4/char + same-rank left-wins (findings C1, C4, C6).                          |
| B14 | DEGRADED   | _check_cast_compat ref-to-ref allows any inner pair (finding C5).                                              |
| A13 | OK         | _ty_key arms for TyFn/TyTensor/TyTile distinguish properly. TyArray.size still collapses (related to C8).      |

Five cycle-1 fixes are degraded by gaps the audit found:
- **B4** trap doesn't fire on the dominant case (untyped lets).
- **B10** TyQuote is a half-finished variant.
- **B13** widening table is incomplete and silently flips signs.
- **B14** ref-to-ref is too permissive.
- **B11** trap exists but isn't reachable from the surface tool.

The other ten fixes hold up under re-audit.

---

## Cycle 2 status

**3 new HIGH findings (C1, C2, C3)** — cycle 2 is **NOT clean** for
the purpose of the 5-clean streak. The findings are regressions /
gaps in the cycle-1 partial fixes themselves rather than wholly-new
soundness holes, but they're real silent-failure paths that warrant
the same treatment as the original 14 — fix-then-re-audit. Recommended
sequence:

1. C2 first (closure trap silence is a runtime-corruption silent failure).
2. C1 and C3 together (both are typecheck-time gaps in a new variant /
   new table — small targeted patches).
3. C4-C8 as a bundle (medium-severity boundary refinements).
4. C9-C11 as a low-severity cleanup commit.

Once C1-C3 land + verify, cycle 3 can re-audit. The remaining HIGH-free
cycles needed to declare type-design soundness clean is one more clean
sweep after addressing the three regressions.
