# Stage 28.8 Pre-29 Audit Gate — Cycle 3, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-10
**Commit**: 40f58ec (read-only)
**Scope**: Re-audit type-system soundness focused on cycle-2's 11 fixes
(commits 3a29728, a086353, 0e53eb8, 7a74acc, 514165b, 134df9b, 5d23121,
4d530c4, 7682b14, 3126aec, a79e867, 87f390c, 40f58ec) and their
interactions. New TyQuote resolve path, asymmetric `_WIDEN_RANK` with
fp8/mxfp4/nvfp4/char, ref-to-ref recursive `_check_cast_compat`, TyArray
size substitution + Binary fold, flatten_impls wired into check.py,
parser-side closure-trap literal inference, grouped Logic-provenance
diagnostics, shared `NUMERIC_FOR_AD` set. Cross-stage interactions
(TyQuote+mono, fp8+autodiff, TyArray+resolve, ref-chain casts).

**Method**: traced each new code path through resolve / substitute /
compatible / propagate / codegen. Wrote probe scripts exercising the
binop, cast, mono, call-boundary paths with edge-case inputs. Verified
each cycle-2 fix holds up under cycle-3 probes. Cross-checked the cycle
3 focus questions (TyQuote-as-struct-field, Logic-wrapped fp8 with
provenance, fp8/mxfp4/nvfp4 in autodiff, TyArray.size=0/negative,
nested-ref cast recursion). Ran the full `helixc/tests/` suite as a
baseline (1396 passed, 1 skipped — matches cycle-2).

**Result**: **9 new findings (2 HIGH, 3 MEDIUM, 4 LOW)**. The two HIGH
findings are systemic call-boundary gaps that cycle-2's TyQuote /
TyDiff / TyLogic fixes exposed by adding new structural types without
extending the function-call argument-type check — `_check_call_basic`
only compares TyPrim names, so any non-TyPrim parameter (`D<i32>`,
`Logic<i32>`, `Quote<i32>`, user struct vs. unrelated struct) silently
accepts any value at every call site. The asymmetric closure-trap
inference in `parser.hx` works for literal-RHS lets but is silent for
the dominant call-RHS case (`let x = some_fn();`). MEDIUM findings
cover negative/zero-sized TyArray produced by substitution + fold,
Logic-vs-Logic mixed-inner silent widening, and the Unary fold gap in
`_subst_shape_expr`. LOW findings are diagnostic-quality issues
(TyStruct missing from `_fmt`, struct\_mono `_ty_key` arms missing for
wrapper types, recursive `_check_cast_compat` Python-recursion limit,
turbofish-inside-generic-body substitution gap).

Zero of the new findings are stop-the-line for the codebase as a
whole, but the call-boundary HIGH (B:D1) is a systemic soundness hole
that has been latent through Stages 24-28.8 — cycle 2 just made it
more obvious by adding new well-typed variants the boundary check
ignores. **Cycle 3 status**: 9 findings (2 HIGH) means cycle 3 does
**NOT** count clean under the strict criterion. See "Cycle 3 status"
final paragraph.

---

## Summary table

| ID  | Severity | Component                              | Issue (short)                                                          |
|-----|----------|----------------------------------------|------------------------------------------------------------------------|
| D1  | HIGH     | typecheck `_check_call_basic`          | Only checks TyPrim-vs-TyPrim; all non-prim arg-vs-param silently passes |
| D2  | HIGH     | bootstrap parser.hx:2261-2325          | Inferred capture tag only for literal-RHS; call-RHS still untracked (-1) |
| D3  | MEDIUM   | mono `_subst_shape_expr`+resolve_type  | Substituted TyArray size of 0 / negative accepted silently             |
| D4  | MEDIUM   | typecheck binop                        | `Logic<T1> + Logic<T2>` mixed-inner silent (no AD warn — D-D path only) |
| D5  | MEDIUM   | mono `_fold_intlit_arith`              | Unary-around-IntLit not folded; `-N` after subst stays Unary(IntLit)   |
| D6  | LOW      | struct\_mono `_ty_key`                 | Missing arms for TyMemTier/TyDiff/TyLogic etc fall to `("?", ...)` collide |
| D7  | LOW      | typecheck `_check_cast_compat`         | Recursive on nested refs hits Python RecursionError at ~500 nesting    |
| D8  | LOW      | typecheck `_fmt`                       | No arm for TyStruct — error messages show `TyStruct(name='Foo')`       |
| D9  | LOW      | mono `_rewrite_calls_in_expr`          | Turbofish `id::<T>(x)` inside generic fn not re-substituted on mono    |

---

## Per-finding sections

### Finding D1: `_check_call_basic` only checks TyPrim-vs-TyPrim — all non-prim arg/param mismatches silently accepted

**File**: `helixc/frontend/typecheck.py:622-671` (`_check_call_basic`).
**Severity**: HIGH
**Category**: type soundness / pre-existing systemic gap exposed by cycle-2 additions

**Description**:
The function-call argument-vs-parameter type check at line 645-655 is
gated on `isinstance(pty, TyPrim) and isinstance(aty, TyPrim)`. Any
parameter whose resolved type is **not** TyPrim — TyDiff, TyLogic,
TyQuote, TyStruct, TyArray, TyRef, TyTensor, TyTile, TyMemTier,
TySkill, TyFn, TyTuple — bypasses the check entirely. The
Logic-provenance check at line 656-660 catches the specific
TyLogic-vs-non-Logic and non-Logic-vs-TyLogic transition, but nothing
else.

Pre-cycle-2, this was a latent issue (no TyQuote resolve path existed,
so `Quote<T>` annotations resolved to TyUnknown and bypassed via the
cascade-safe rule). Cycle-2 B:C3 added the resolve arm so `Quote<i32>`
correctly resolves to `TyQuote(TyPrim('i32'))`, and cycle-2 B:C2-9
added the `_compatible` arm for TyQuote-vs-TyQuote. Both are working.
But the call-boundary still doesn't apply `_compatible` for non-prim
pairs — so `unbox(99)` where `unbox(q: Quote<i32>)` succeeds with zero
diagnostics (verified by probe).

**Reproducer** (all of these typecheck clean — they should NOT):
```helix
fn use_d(x: D<i32>) -> D<i32> { x }
fn main() -> D<i32> {
    let raw: i32 = 5;
    use_d(raw)               // i32 silently coerced to D<i32>
}

fn use_q(x: Quote<i32>) -> i32 { 0 }
fn main2() -> i32 {
    use_q(42)                // raw 42 (i32) passed where Quote<i32> expected
}

struct A { x: i32 }
struct B { y: i32 }
fn use_a(a: A) -> i32 { 0 }
fn main3() -> i32 {
    let b = B { y: 5 };
    use_a(b)                 // B silently accepted where A expected
}
```

Probe confirms: each of these produces zero diagnostics from
`TypeChecker.check()`. Only the StructLit boundary (`Box { item: 42 }`
where field is `Quote<i32>`) actually uses `_compatible` and rejects.

**Recommended fix**:
Within `_check_call_basic`, after the TyPrim arm, fall through to
`self._compatible(pty, aty)`. When False AND neither side is
TyUnknown, emit a diagnostic. Treat TyVar / TySize on the param side
as "defer to monomorphization" (don't fire). Keep the
Logic-provenance batch path as a specialization that fires *before*
the general check so the Logic-specific hint is preserved. A minimal
patch is approximately 8 lines:

```python
for (pname, pty), aty in zip(sig.params, arg_tys):
    if isinstance(pty, TyPrim) and isinstance(aty, TyPrim):
        # existing prim arm
        ...
    elif (not isinstance(pty, (TyVar, TySize, TyUnknown))
          and not isinstance(aty, TyUnknown)
          and not self._compatible(pty, aty)):
        # New: general non-prim mismatch
        # Skip Logic-provenance cases — they go through the
        # specialized path below for the better hint.
        if self._logic_provenance_violation_kind(pty, aty) is None:
            self.errors.append(TypeError_(
                f"call to {sig.name!r}: arg {pname!r} expects "
                f"{self._fmt(pty)}, got {self._fmt(aty)}",
                call.span,
            ))
    # provenance batch path unchanged
    ...
```

---

### Finding D2: Closure-capture trap 76003 still silent on call-RHS lets

**File**: `helixc/bootstrap/parser.hx:1807-1812` (cap probe);
`helixc/bootstrap/parser.hx:2261-2325` (let inference).
**Severity**: HIGH
**Category**: silent-failure regression on cycle-2 partial fix

**Description**:
Cycle-2 B:C2 added an inferred type-tag for untyped lets whose RHS is
a literal (FloatLit, IntLit with suffix, BoolLit). The dispatch tree
at lines 2290-2315 enumerates 11 literal AST tags. The lookup at the
capture site (line 1810) uses `if cap_ty_tag > 0` to fire trap 76003
when the captured var is provably non-i32.

But the comment at lines 2284-2289 explicitly acknowledges:

> Any other root tag (Binary / Call / Name / Block / If / ...) stays
> untracked (-1) and the capture probe will see an opaque value —
> which today silently passes the > 0 guard, BUT we want the
> closure-capture loop to treat untracked as "potentially non-i32
> unless RHS proves i32" (see capture-site update below).

There is no "capture-site update below". Line 1810's guard remains
`if cap_ty_tag > 0`, which silently passes -1 (untracked). The
docstring also at lines 1807-1808 says "any other value (including -1
= not-tracked) means we'd be silently truncating" — but the code
doesn't enforce what the docstring describes.

So the dominant idiom `let pi = get_pi(); let c = |x| x + (pi as i32);`
(untyped let with a Call RHS) is still silent. The narrowest variant
of B:C2's gap is closed (literal-RHS), but the wider variant remains
open.

**Reproducer**:
```helix
fn get_pi() -> f64 { 3.14_f64 }
fn main() -> i32 {
    let pi = get_pi();              // Call RHS — untracked (-1)
    let c = |x: i32| x + (pi as i32);  // capture silent, trap 76003 not fired
    c(7)
}
```

The capture-time tag lookup returns -1; `if cap_ty_tag > 0` is false;
trap silent; closure captures pi's low 32 bits as i32 — bit
truncation of the f64 bit pattern.

**Recommended fix**:
Per the parser.hx comment's promise, change line 1810 from `if
cap_ty_tag > 0` to `if cap_ty_tag != 0`. Then -1 (untracked) and any
positive value (proven non-i32) both fire the trap. Tag 0 (proven
i32) is the only safe pass. This is a one-character change in
parser.hx + a comment update.

Alternative: track every let-binding (including call-RHS) by emitting
tag 0 only for provably-i32 expressions (IntLit no-suffix, Binary of
two tag-0 operands, etc.). The full stride-3 cl\_capture\_tab solution
deferred from cycle 1 remains the long-term answer.

---

### Finding D3: TyArray size of 0 or negative accepted silently after substitution

**File**: `helixc/frontend/typecheck.py:544-558` (`_resolve_size_expr`);
`helixc/frontend/monomorphize.py:181-189` (TyArray subst arm).
**Severity**: MEDIUM
**Category**: type soundness / monomorphization gap

**Description**:
Cycle-2 B:C8 fixed `substitute_ty` to walk `TyArray.size` and
`_fold_intlit_arith` to fold `Binary(IntLit, IntLit)`. Both work
correctly for the positive-size case verified by probe.

But the fold can produce **negative** or **zero** IntLits. Examples:

- `[T; N-10]` with N=4 produces `IntLit(-6)`
- `[T; N-N]` with any N produces `IntLit(0)`
- `[T; N/M]` with `M=2, N=1` produces `IntLit(0)` (integer div)

Neither `_resolve_size_expr` nor `substitute_ty` flags these. The
resolved type becomes `TyArray(elem=..., size=TyPrim('size_-6'))` or
`TyArray(elem=..., size=TyPrim('size_0'))` — a silent
negative-encoded or zero size. Downstream lower-ast.py treats these
as bare numbers; a zero-size array allocates a 0-byte buffer; a
negative size sign-extends to a huge unsigned offset and the codegen
allocator either rejects (Phase-0 panic) or silently allocates
garbage memory.

Probe:
```python
arr = A.TyArray(elem=TyName('T'),
                size=A.Binary('-', Name('N'), IntLit(10)))
result = substitute_ty(arr, {'T': TyName('i32'), 'N': _SizeLitMarker(4)})
# result.size = IntLit(-6)

# Then _resolve_size_expr produces:
TyPrim('size_-6')   # silent
```

**Recommended fix**:
In `_resolve_size_expr` (and `_size_expr_to_lin`), after substitution
+ fold, check the IntLit value: if `< 0` emit a typecheck error
(trap 16004 / array size out of range); if `== 0` emit a warning
(zero-size arrays are legal in some languages but Phase-0 Helix
doesn't define semantics for them). Place the check in
`_resolve_size_expr` so it catches both source-level
`[i32; some_const]` if `some_const` ever becomes negative AND
mono-substituted shapes.

---

### Finding D4: `Logic<T1> + Logic<T2>` mixed-inner silent (asymmetric with D-D path)

**File**: `helixc/frontend/typecheck.py:1208-1265` (binop handler).
**Severity**: MEDIUM
**Category**: AD warning coverage / docstring-vs-implementation drift

**Description**:
The cycle-2 doc finding C6 explicitly noted this asymmetry but the
cycle-2 fix only widened the D-D path (added the asymmetric
`D<T> + bareT` warning). Pure `Logic<T1> + Logic<T2>` (neither side
TyDiff) remains silent.

The guarding condition at line 1242 is:

```python
if (l_is_diff or r_is_diff) and inner_mismatch:
    inner = _widen_diff_inner(...)
    ...
```

For `Logic<f64> + Logic<i32>`: `l_is_diff = False`, `r_is_diff =
False`, so the condition short-circuits. The else branch at line
1255-1257 picks `l_inner` if not TyUnknown, else `r_inner` —
silent left-wins. The result type is `Logic<f64>`, the i32 operand's
domain is dropped without any diagnostic.

Probe:
```
Logic<f64> + Logic<i32>:  result = Logic<f64>, ad warns = []
```

The cycle-2 doc said "whether to warn there depends on Logic
semantics — but the asymmetry with TyDiff is worth flagging". Cycle 3
still flags it because the asymmetry persists. The Logic docstring
(typecheck.py:128-162) explicitly says `Logic<T>` carries provenance
— silently dropping the right-side i32 provenance domain is exactly
the kind of silent loss that TyLogic was designed to prevent.

**Recommended fix**:
Extend the gate to fire when EITHER side is TyLogic OR TyDiff AND
inner-mismatch is non-trivial. The widening logic is identical
(`_widen_diff_inner`); the warning text would say "Logic-binop with
mixed inner types …" instead of "D-binop". Reuse the existing
`_ad_warn_mixed_inner` helper with a different `extra=` tag.

Alternative: split into a `_logic_warn_mixed_inner` companion that
emits trap 24201 (separate ID for logic-domain transition). The
distinction matters because logic widening is potentially valid in
fuzzy-relational semantics, whereas D widening over int<->float is
not.

---

### Finding D5: Unary fold gap — `-N` after subst stays Unary(IntLit)

**File**: `helixc/frontend/monomorphize.py:63-93` (`_fold_intlit_arith`);
`helixc/frontend/monomorphize.py:141-143` (Unary arm).
**Severity**: MEDIUM
**Category**: monomorphization gap (continuation of B:C11)

**Description**:
`_fold_intlit_arith` (cycle-2 B:C11) folds `Binary(IntLit, IntLit)` to
a single IntLit, but the symmetric one-level fold for
`Unary(-, IntLit)` is missing. The Unary case in `_subst_shape_expr`
at line 141-143 walks into the operand but does NOT post-process the
result the way the Binary case does. So `Unary(-, Name(N))` with N=5
becomes `Unary(-, IntLit(5))` — not `IntLit(-5)`.

Probe:
```python
arr = A.TyArray(elem=TyName('T'),
                size=A.Unary('-', Name('N')))
substitute_ty(arr, {'N': _SizeLitMarker(5)}).size
# -> Unary(-, IntLit(5))    not IntLit(-5)
```

Then `_resolve_size_expr` at line 555-558 falls through to:

```python
return TyUnknown(hint=f"size expr {type(expr).__name__}")
```

It has no Unary arm. So `[T; -N]` after substitution resolves to
TyArray with `size=TyUnknown(hint='size expr Unary')` — which the
shape solver then can't constrain, and the downstream lower-ast.py
defaults the length to 0 (the same silent-default bug that B8/B:C8
were supposed to close).

**Reproducer**: `[T; -N]` declared in a generic fn — but the parser
likely refuses negative literal sizes at source level. The practical
exposure path is mono-time fold producing a Unary(IntLit) after
substituting `N` into `-(N-3)` with N=10 → `Unary(-, IntLit(7))`,
which should be IntLit(-7) (and then trigger D3's check).

**Recommended fix**:
Add a `_fold_intlit_unary` helper symmetric to
`_fold_intlit_arith`:

```python
def _fold_intlit_unary(expr: A.Expr) -> A.Expr:
    if not isinstance(expr, A.Unary):
        return expr
    if not isinstance(expr.operand, A.IntLit):
        return expr
    if expr.op == "-":
        return A.IntLit(span=expr.span, value=-expr.operand.value,
                        type_suffix=None)
    if expr.op == "+":
        return expr.operand
    return expr
```

Call it from `_subst_shape_expr`'s Unary arm:

```python
if isinstance(expr, A.Unary):
    folded = A.Unary(span=expr.span, op=expr.op,
                     operand=_subst_shape_expr(expr.operand, subst))
    return _fold_intlit_unary(folded)
```

Then add an `A.Unary` arm to `_resolve_size_expr` so source-level
Unary sizes (rare but possible) resolve cleanly too.

---

### Finding D6: `_ty_key` missing arms for TyMemTier/TyDiff/TyLogic/TyQuote/TyFn-inside-generic

**File**: `helixc/frontend/struct_mono.py:326-373` (`_ty_key`).
**Severity**: LOW
**Category**: monomorphization key collision

**Description**:
`_ty_key` has arms for TyName, TyGeneric, TyTuple, TyArray, TyRef,
TyPtr, TyFn, TyTensor, TyTile (cycle-1 A13). It has **no** arms for
TyMemTier, TyDiff, TyLogic, TyQuote, TyUnit, TyUnknown, TyVar, TySize.
All eight fall through to `("?", type(t).__name__)`.

Most of these (TyDiff, TyLogic, TyQuote, TyMemTier) appear in user
code only as TyGeneric in the *AST* representation (i.e.
`A.TyGeneric(base='D', args=[...])` not `A.TyDiff`), so they hit the
TyGeneric arm and dedupe correctly. Verified by probe:

```python
_ty_key(A.TyGeneric('D', [A.TyName('i32')]))
# -> ('gen', 'D', (('name', 'i32'),))      OK
```

But if any path constructs a *resolved* `TyDiff(TyPrim('i32'))` and
hands it to `_ty_key` directly (e.g. a future refactor that unifies
the AST and resolved type tables), the fall-through would collapse
`Pt<D<i32>>` and `Pt<D<f64>>` to the same key. The bug isn't live
today because no caller passes resolved Type instances to `_ty_key`,
but the asymmetry between resolved and AST forms is the kind of thing
that drifts.

**Recommended fix**:
Add explicit arms for the resolved-Type cases as well, mirroring the
TyGeneric encoding:

```python
if isinstance(t, A.TyDiff) if hasattr(A, 'TyDiff') else False:
    return ("gen", "D", (_ty_key(t.inner),))
# (same pattern for TyLogic, TyQuote, TyMemTier, …)
```

Or alternatively, refactor so `_ty_key` always operates on the AST
form. Pick one canonical input and reject the other (assert + raise).

---

### Finding D7: Recursive `_check_cast_compat` blows Python recursion stack at deeply-nested refs

**File**: `helixc/frontend/typecheck.py:1948-1964`.
**Severity**: LOW
**Category**: robustness / runtime crash

**Description**:
Cycle-2 B:C5 fixed the ref-to-ref cast matrix to recurse into the
inner type pair. The recursion is unbounded in the input depth.
Python's default `sys.getrecursionlimit()` is ~1000, so a cast like
`&&&...&i32 as &&&...&i64` with ~500 levels of `&` on each side
raises `RecursionError`, which is not caught — the typechecker
crashes with an unhandled traceback.

Probe:
```python
src = TyPrim('i32')
tgt = TyPrim('i64')
for _ in range(1000):
    src = TyRef(inner=src, is_mut=False)
    tgt = TyRef(inner=tgt, is_mut=False)
tc._check_cast_compat(src, tgt, span)
# -> RecursionError: maximum recursion depth exceeded
```

Real source code never has 500-level nested refs (Phase-0 has no
syntactic way to write them), so this is a theoretical issue. But a
malicious or autogenerated source feed could trigger it.

**Recommended fix**:
Convert the recursive call to iterative: walk `src` and `tgt` in
lockstep peeling off TyRef wrappers until one side is not a TyRef,
then apply the rest of the cast matrix to the unwrapped pair.
Roughly:

```python
while isinstance(src, TyRef) and isinstance(tgt, TyRef):
    src = src.inner
    tgt = tgt.inner
# fall through to the matrix
```

Drop the asymmetric-emit logic — if one side runs out of refs while
the other doesn't, that's a clean error.

---

### Finding D8: `_fmt` has no arm for TyStruct — diagnostics show raw repr

**File**: `helixc/frontend/typecheck.py:2016-2053` (`_fmt`).
**Severity**: LOW
**Category**: diagnostic UX

**Description**:
`_fmt` has arms for every parametric type (TyDiff, TyLogic, TyMemTier,
TySkill, TyQuote, TyArray, TyRef, TyPtr, TyFn, TyTuple, TyTensor,
TyTile, TyPrim, TyVar, TySize, TyUnit, TyUnknown) but **none for
TyStruct**. The final `return repr(t)` produces `TyStruct(name='Foo')`
in user-facing error messages.

Visible in cycle-3 probe:
```
&Foo as &Bar
  -> invalid cast: source &TyStruct(name='Foo') cannot convert to &TyStruct(name='Bar') (trap 28604)
```

The user sees `&TyStruct(name='Foo')` instead of `&Foo`. Minor
papercut but the diagnostic loses readability — and the cycle-2 fix
B:C5 made this user-visible for the first time (pre-fix, ref-to-ref
silently passed and the diagnostic never fired).

**Recommended fix**:
Add an arm:

```python
if isinstance(t, TyStruct):
    return t.name
```

Place it near the TyPrim arm (line 2017) for consistency.

---

### Finding D9: Turbofish inside generic fn body not re-substituted at mono time

**File**: `helixc/frontend/monomorphize.py:402-425` (rewrite + instantiate).
**Severity**: LOW
**Category**: monomorphization gap

**Description**:
When a generic fn `caller[T]` calls another generic fn via turbofish
`id::<T>(x)`, the call-site rewriter at line 407-422 turns this into
`id__T(x)` (taking T literally as a type name) at the moment caller
is first walked, regardless of what T is later bound to. When the
monomorphizer later clones `caller` as `caller__i32`, the body still
contains `id__T(x)` instead of `id__i32(x)` — because the rewrite
runs before substitution, and substitution doesn't re-walk
`A.Name(generics=[…])` to update them.

Probe:
```
fn id[T](x: T) -> T { x }
fn caller[T](x: T) -> T { id::<T>(x) }
fn main() -> i32 { caller::<i32>(5) }

after monomorphize:
  id, caller, main, id__T, caller__i32
  caller__i32 body: Call(Name('id__T'), [Name('x')])    ← still T!
```

So the generated `caller__i32` calls a non-existent `id__T` symbol;
codegen will trap on the unresolved name (or worse, silently link to
a different mono of `id` if one happens to exist with `T` in its
mangled form). The cycle-2 tests don't exercise this path (they use
non-nested turbofish).

**Recommended fix**:
In `_walk_subst_expr`, when descending into `A.Call` with
`callee = A.Name(generics=[…])`, substitute the generics list too:

```python
if isinstance(e, A.Call):
    new_callee = e.callee
    if isinstance(new_callee, A.Name) and new_callee.generics:
        new_callee = A.Name(
            span=new_callee.span, name=new_callee.name,
            generics=[substitute_ty(g, subst) for g in new_callee.generics],
        )
    return A.Call(span=e.span,
                  callee=_walk_subst_expr(new_callee, subst),
                  args=[_walk_subst_expr(a, subst) for a in e.args])
```

Then the inner call's turbofish is re-substituted with the outer
mono's binding (`T` → `i32`), and the next mono iteration discovers
the call to `id::<i32>` and produces `id__i32`.

---

## Cycle 2 fix re-verification

| Fix | Status     | Notes                                                                                                       |
|-----|------------|-------------------------------------------------------------------------------------------------------------|
| C1  | OK         | `_WIDEN_RANK` now has fp8 (45), mxfp4/nvfp4 (43), char (5), bool (1), with asymmetric signed/unsigned ranks. fp8/mxfp4/nvfp4/char all beat ints by rank — float-into-int collapse closed.  |
| C2  | PARTIAL    | Literal-RHS inference works (probe: `let pi = 3.14_f64; let c = |x:i32| x + (pi as i32); c(7)` traps 76003). Call-RHS untracked — see finding D2.   |
| C3  | OK         | `_resolve_type` Quote arm + `_compatible` TyQuote arm fire. StructLit boundary uses `_compatible` correctly. **However**, `_check_call_basic` boundary doesn't — see D1.  |
| C4  | OK         | Same-rank ties trigger callback warn; with the new asymmetric ranks ties only occur on mxfp4/nvfp4 pairs (both rank 43) and the callback fires deterministically.   |
| C5  | OK*        | `&i32 as &i64` allowed via inner-recursive check, `&Foo as &Bar` rejected. Diagnostic shows raw `TyStruct(name='Foo')` (finding D8). Recursion-limit edge case at depth ~500 (finding D7). |
| C6  | OK         | `D<T> + bareT` now warns symmetrically. `Logic + Logic` mixed inner remains silent — see D4 (cycle-2 doc flagged this as worth noting, cycle 3 escalates to MEDIUM).  |
| C7  | OK         | `flatten_impls` invoked in check.py at line 376-381 between struct\_mono and totality. DuplicateMethodError surfaces as a clean diagnostic.   |
| C8  | OK*        | TyArray.size now substituted + folded for Binary; Unary fold missing (finding D5). Negative/zero result not validated (finding D3).   |
| C9  | OK         | `NUMERIC_FOR_AD` frozenset shared between autodiff.py and autodiff\_reverse.py; covers bool/char/fp8/mxfp4/nvfp4. False-warns closed.   |
| C10 | OK         | Logic provenance batched: 2+ violations emit a single grouped diagnostic with comma-separated param names. Verified via probe.   |
| C11 | OK         | Binary fold one-level deep works; deep nesting (100-level probe) folds correctly to a single IntLit. Negative results produced silently (rolls into D3).   |

Three cycle-2 fixes have residual gaps (`*`):
- **C2** still silent on the dominant call-RHS case (finding D2).
- **C3** the TyQuote variant is now reachable from annotations, but
  `_check_call_basic` ignores it like every other non-prim
  (finding D1).
- **C5** ref-to-ref recursive but stack-bounded (finding D7) and
  diagnostic shows raw repr (finding D8).
- **C8** missing Unary fold (finding D5); no validation on negative
  /zero sizes (finding D3).

The other seven fixes hold up under re-audit with no caveats.

---

## Cycle 3 focus question answers

**1. New TyQuote substitution — used as generic struct field type?**

Verified: `struct Box[T] { item: Quote<T> }` monomorphized against
T=i32 and T=f64 produces `Box__i32 { item: Quote<i32> }` and
`Box__f64 { item: Quote<f64> }` correctly. `_ty_key` dedupes them
correctly via the TyGeneric arm. Struct field-type compatibility at
StructLit assignment correctly uses `_compatible`, which has the
TyQuote arm. **Status**: working as designed. No new finding.

**2. Logic-wrapped fp8 / mxfp4 — provenance tracking?**

Verified: `Logic<fp8> + Logic<fp8>` returns `Logic<fp8>` cleanly,
`D<Logic<fp8>> + Logic<fp8>` correctly composes to `D<Logic<fp8>>`,
and `D<Logic<fp8>> + D<Logic<i32>>` correctly fires the
mixed-inner AD warning via the `_unwrap` helper that peels both D
and Logic wrappers. Pure `Logic<fp8> + Logic<i32>` stays silent —
see finding D4.

**3. fp8 / mxfp4 / nvfp4 in autodiff — propagate through `_widen_diff_inner`?**

Verified: ranks 45/43/43 sit above every integer (i32=30, i64=40) so
`D<fp8> + D<i64>` correctly widens to `fp8` (NOT i64). The cycle-2
B:C1 / C1 fix is sound and the AD warning fires correctly. Same-rank
ties (mxfp4 vs nvfp4) emit a tie-warning via the callback.
**Status**: working as designed. No new finding.

**4. TyArray.size resolving to 0 or negative — flagged?**

**No**. Substituted TyArray sizes that evaluate to 0 or negative
flow through `_resolve_size_expr` as `TyPrim('size_-6')` or
`TyPrim('size_0')` with no diagnostic. See finding D3.

**5. Recursive `_check_cast_compat` on nested refs (`&&T`)?**

Works correctly for any practically-occurring depth. `&&i32 as
&&i64` is accepted (inner numeric matrix); `&&Foo as &&Bar` is
rejected (inner struct mismatch). Cyclic refs aren't constructable
via frozen dataclasses. Deeply-nested refs (~500+ levels) hit
Python's recursion limit — see finding D7.

---

## Cycle 3 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity.**

This cycle finds **9 new findings (2 HIGH, 3 MEDIUM, 4 LOW)**. By the
strict criterion, **cycle 3 does NOT count clean**. The two HIGH
findings are both regressions / gaps in cycle-2 partial fixes (D2 is
the same shape as cycle-2's C2 — narrow fix didn't cover the dominant
case; D1 is a pre-existing systemic gap that cycle-2's TyQuote
addition made more obvious by introducing yet another non-TyPrim
parameter type that the call boundary doesn't check).

Recommended fix sequence for cycle 4:

1. **D1 first** (highest-impact systemic — every non-prim param fix
   improves with one ~8-line patch in `_check_call_basic`). Pair
   with **D2** (closure-trap one-line guard tightening) for the
   same commit — both are silent-failure regressions.
2. **D3 and D5** as a bundle (TyArray size validation + Unary fold —
   both touch the same monomorphize.py file, related to the cycle-2
   B:C8 / B:C11 fixes).
3. **D4** standalone — Logic-Logic mixed inner. Requires deciding
   the warning class (trap 24201 separate or just extend 24200's
   coverage).
4. **D6 / D7 / D8 / D9** as a low-severity cleanup commit.

Once D1-D5 land + verify, cycle 4 can re-audit. Per the strict
criterion, the remaining cycles needed to declare type-design soundness
clean is at least one full clean sweep after addressing all 9
findings — pending whether new gaps surface from the fixes themselves
(cycle 2's pattern of "fix exposes the next layer").
