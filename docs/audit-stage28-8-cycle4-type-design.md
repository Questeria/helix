# Stage 28.8 Pre-29 Audit Gate — Cycle 4, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit**: b3504a2 (read-only)
**Scope**: Re-audit type-system soundness focused on cycle-3's 15 fixes
(commits 025d55e, c31158c, ee7aa42, 74b72ec, 3358627, dccfc7e, 2b15928,
3b321e6, a878709, dda3b9d, b3504a2). New `ShapeFoldError` exception
class, asymmetric `_widen_canon_name` pointer-width alias dedup, D1
non-prim call-boundary check with widened `_compatible` (TyDiff /
TyLogic / TyTuple / TyArray / TyRef / TyPtr / TyFn structural arms),
iterative ref-peel with depth-8 guard in `_check_cast_compat`, D4
Logic-Logic mixed-inner warn, D5 `_fold_intlit_unary` symmetric one-
level Unary fold, D6 `_ty_key` strict guard (raise on non-AST.TyNode),
D9 turbofish re-substitution in `_walk_subst_expr`, C3-5 wide
`_inline_lets` recursion through Cast/Call/Field/Index/Match/etc., C3-3
try/finally + `except Exception` in `main()`. Cross-stage interactions
(ShapeFoldError catch contract, TyArray size comparison consistency,
Logic-wrap asymmetric coverage, tag-12 sentinel encoding).

**Method**: traced each new code path through resolve / substitute /
compatible / propagate / fold / driver-dispatch. Mentally executed
edge-case inputs against the new code (TyArray with TySize sizes at
call boundary, `Logic<f64> + f64`, `[T; N/0]` inside fn-mono path,
TileLit under `_inline_lets`, `let f = some_fn; f::<i32>(x)`). Verified
each cycle-3 fix holds under cycle-4 probes. Cross-checked cycle-3
strictness against the cycle-4 focus questions: D1's `_compatible`
inner-size symmetry, D4 Logic-wrap asymmetry, ShapeFoldError catch
contract scope, `_inline_lets` AST coverage completeness, tag-12
sentinel namespace overlap.

**Result**: **8 new findings (0 HIGH, 3 MEDIUM, 5 LOW)**. The three
MEDIUM findings expose the next-layer-down asymmetries that cycle-3's
fixes opened by widening the structural type-compatibility contract.
E1 is a regression introduced by D1's new `_compatible` TyArray arm
(uses raw `==` for size where every other composite arm uses recursive
`_compatible` — silent false-positive risk when a TyArray's size is
TyUnknown or one side has a fresh TySize symbol). E2 is the
Logic-wrap-asymmetric variant D4 didn't cover (`Logic<f64> + f64`
remains silent, parallel to the cycle-2 B:C6 → cycle-3 D4 escalation
pattern). E3 is the ShapeFoldError catch contract asymmetry — only
`monomorphize_structs` catches it; `monomorphize` (fn mono) at
x86_64.py:3021 doesn't, so a generic fn with `[T; N/0]` surfaces as
"compiler bug" instead of trap 28801. LOW findings cover diagnostic
text drift (E4), namespace overlap in the parser tag-12 sentinel (E5),
and three small gaps in the C3-5 `_inline_lets` widening (E6: dropped
turbofish in aliased Name callee, E7: missing TileLit arm leading to
catch-all over-fire, E8: Field-typed callees not walked).

Zero of the new findings are stop-the-line for the codebase as a
whole. None are HIGH. But E1 (TyArray size symmetry) and E2 (Logic-
wrap asymmetric) are real false-negative / false-positive contract
drift in the type system's structural-equality invariant, and E3
(ShapeFoldError catch) means a user-source error can crash the
compiler with a misleading "internal error" diagnostic via the
fn-mono path. **Cycle 4 status**: 8 findings (0 HIGH, 3 MEDIUM, 5
LOW) means cycle 4 does **NOT** count clean under the strict
criterion. See "Cycle 4 status" final paragraph.

---

## Summary table

| ID  | Severity | Component                                      | Issue (short)                                                                |
|-----|----------|------------------------------------------------|------------------------------------------------------------------------------|
| E1  | MEDIUM   | typecheck `_compatible` TyArray arm            | `a.size == b.size` raw eq instead of recursive `_compatible` — false-positive on TyUnknown / TySize size |
| E2  | MEDIUM   | typecheck binop Logic-domain warn              | `Logic<f64> + f64` (wrap-asymmetric) silent — D4 only closed Logic-Logic mixed; wrap-asymmetric still silent |
| E3  | MEDIUM   | monomorphize fn-mono / x86_64 driver           | `ShapeFoldError` not caught by `monomorphize()`; escapes to top-level C3-3 try/except as "compiler bug" |
| E4  | LOW      | typecheck `_resolve_size_expr` D3 diagnostic   | Negative branch says ">= 0", zero branch says "> 0" — inconsistent invariant phrasing |
| E5  | LOW      | bootstrap parser.hx D2 tag-12 sentinel         | Tag 12 collides with type-tag namespace (0-11 reserved for prims); future prim addition would silently break |
| E6  | LOW      | autodiff `_inline_lets` Call arm (C3-5)        | Aliased-Name callee substitution drops `callee.generics` turbofish list |
| E7  | LOW      | autodiff `_inline_lets` missing TileLit arm    | TileLit falls through to catch-all `_ad_warn`; over-fires AD warn on legal tile code |
| E8  | LOW      | autodiff `_inline_lets` Call.callee=Field      | `let me = self; me.method()` — Field-typed callee's `obj` not walked, let-binding silently dropped |

---

## Per-finding sections

### Finding E1: `_compatible` TyArray arm uses raw `==` for size, breaking cascade-safe / generic-substitution-deferred invariant

**File**: `helixc/frontend/typecheck.py:2181-2186` (`_compatible` TyArray arm).
**Severity**: MEDIUM
**Category**: structural-equality invariant violation introduced by D1

**Description**:
Cycle 3 D1 added a structural arm to `_compatible` so the new
non-TyPrim call-boundary check can recognize compatibility across
TyTuple / TyArray / TyRef / TyPtr / TyFn. Most arms use the
recursive `self._compatible(inner_a, inner_b)` for inner types,
inheriting the top-of-function cascade-safe rule (`isinstance(a,
TyUnknown) or isinstance(b, TyUnknown): return True`).

But the TyArray arm at line 2181-2184 is asymmetric:

```python
if isinstance(a, TyArray) and isinstance(b, TyArray):
    # Sizes are TyPrim('size_N'); compare nominally.
    return (self._compatible(a.elem, b.elem)
            and a.size == b.size)
```

`a.size` is `Type` (per the schema at line 76-78) — not necessarily
`TyPrim('size_N')`. `_resolve_size_expr` can return:
- `TyPrim('size_N')` for concrete IntLit sizes
- `TySize(name)` for size-kind generic params (from scope lookup)
- `TyUnknown(hint=...)` for unresolvable Binary / Unary / Name
  (e.g., post-D5 the Unary case is now folded, but Binary with a
  non-foldable Name leaf still falls through to TyUnknown)
- `TyVar(name)` if the size param happens to resolve to a TyVar

Two specific false-positive paths:

1. **Generic-array-call-boundary (false positive)**:
   ```helix
   fn f[N](a: [i32; N]) -> i32 { 0 }
   fn g[M](a: [i32; M]) -> i32 { f(a) }
   ```
   At the call to `f(a)` inside `g`, `pty = TyArray(i32, TySize('N'))`
   and `aty = TyArray(i32, TySize('M'))`. The D1 elif at typecheck.py
   line 730 filters out `pty: TyVar/TySize/TyUnknown` at the TOP level
   only — the inner TySize escapes. `_compatible` recurses into
   TyArray: elem matches via `_compatible(i32, i32)`; size compares
   via `TySize('N') == TySize('M')` → False (different `name`
   fields on the frozen dataclass). Result: False, the D1 check
   emits `"call to 'f': arg 'a' expects [i32; size:N], got [i32; size:M]"`
   — a false positive. The call IS valid (mono will instantiate N to
   whatever M is bound to at f's call site).

2. **TyUnknown-size cascade (false positive)**:
   ```helix
   fn f(a: [i32; 5]) -> i32 { 0 }
   ```
   Called with an arg of resolved type `TyArray(i32, TyUnknown(...))`
   (where the size couldn't be resolved due to e.g. a non-foldable
   Binary). Top-level `_compatible(a, b)` doesn't see TyUnknown (both
   are TyArray); recurses → `a.size == b.size` compares
   `TyPrim('size_5') == TyUnknown(...)` → False (different classes).
   Result: False. The cascade-safe rule ("TyUnknown anywhere skips
   the check") is violated for sub-component sizes.

Every other composite arm uses `_compatible` recursively for inner
types — TyTuple at 2174 recurses into elems, TyRef/TyPtr at 2187/2192
recurse into inner, TyFn at 2197 recurses into params + ret. The
TyArray arm is the lone outlier.

**Reproducer** (probe-level — typecheck on the source above would
mis-fire the D1 emit):
```python
from helixc.frontend.typecheck import (
    TypeChecker, TyArray, TyPrim, TySize, TyUnknown)
tc = TyperChecker(...)
a = TyArray(elem=TyPrim('i32'), size=TySize('N'))
b = TyArray(elem=TyPrim('i32'), size=TySize('M'))
tc._compatible(a, b)
# -> False (should defer to mono / accept as cascade-safe)

a2 = TyArray(elem=TyPrim('i32'), size=TyPrim('size_5'))
b2 = TyArray(elem=TyPrim('i32'), size=TyUnknown(hint='size expr Binary'))
tc._compatible(a2, b2)
# -> False (should be True via cascade-safe TyUnknown)
```

**Recommended fix**:
Change the TyArray arm to recurse on size too, so TyUnknown /
TyVar / TySize symmetric to elem:

```python
if isinstance(a, TyArray) and isinstance(b, TyArray):
    return (self._compatible(a.elem, b.elem)
            and self._compatible(a.size, b.size))
```

`_compatible` already cascades TyUnknown at the top (returns True),
and for `TyPrim` vs `TyPrim` at the bottom uses `a == b` (returns
True for matching `size_N` literals). The only behavior change is
that `TySize('N')` vs `TySize('M')` becomes "True" (cascade-safe
since neither is TyPrim and they fall through to `return a == b`
which is still False — but `_compatible` would still need a
TyVar/TySize defer arm). Pre-fix this never came up because the
call-boundary check didn't reach _compatible for non-prim pairs;
D1 made it reachable.

Alternative: at the D1 call-boundary site itself, recursively check
TyVar/TySize/TyUnknown anywhere in the inner structure and defer if
present. More invasive but isolates the cycle-3 change rather than
modifying _compatible's contract.

---

### Finding E2: `Logic<T> + bareT` wrap-asymmetric still silent — D4 fix covered only Logic-Logic mixed inner

**File**: `helixc/frontend/typecheck.py:1357-1374` (binop Logic branch);
`helixc/frontend/typecheck.py:128-162` (TyLogic docstring).
**Severity**: MEDIUM
**Category**: incomplete fix / contract drift, parallel to the
cycle-2 B:C6 → cycle-3 D4 escalation pattern

**Description**:
Cycle 3 D4 added a Logic-Logic mixed-inner warn for `Logic<f64> +
Logic<i32>`. The gate at line 1357 is:

```python
elif (l_is_logic or r_is_logic) and inner_mismatch:
    ...
```

For `Logic<f64> + f64` (one side Logic-wrapped, other bare): `l_inner
= _unwrap(Logic<f64>) = f64`, `r_inner = f64`, so `inner_mismatch =
(f64 != f64) = False`. Gate doesn't fire. Falls to the else at line
1375: `inner = l_inner`. The result type is rebuilt as `Logic<f64>`
(via line 1381's `if l_is_logic or r_is_logic: wrapped = TyLogic(...)`)
silently absorbing the bare f64 operand's missing provenance domain.

This is exactly the same shape as the cycle-2 D-D vs D-bare
asymmetric case (B:C6) — which cycle-2 closed with the gate change
from `l_is_diff AND r_is_diff` to `l_is_diff OR r_is_diff`. The D4
fix added the Logic arm with the same OR gate, but Logic-wrap
asymmetric isn't a "mixed inner" case at all — both inners are
identical (the bare operand IS the Logic's inner). The asymmetry is
in the *wrap*, not the inner.

The TyLogic docstring (typecheck.py:128-162) explicitly says Logic
carries provenance. Silently absorbing a bare-T operand defeats
that contract: the bare operand contributed no provenance domain,
yet the result claims Logic-provenance heritage.

Cycle 3's audit doc (D4 description) actually noted this shape:
"asymmetric with D-D path" — the cycle-3 fix closed Logic-Logic
mixed but did not parallel B:C6's wrap-asymmetric coverage. So this
is the cycle-2 → cycle-3 pattern (partial fix) repeating one cycle
later for Logic.

**Reproducer**:
```python
from helixc.frontend.typecheck import (
    TypeChecker, TyLogic, TyPrim)
# typecheck a binop where l is Logic<f64> and r is f64
# observe: result is Logic<f64>, AD warns list is empty (no
# diagnostic about the bare-f64 operand losing provenance)
```

Or via source:
```helix
fn f() -> Logic<f64> {
    let a: Logic<f64> = ... ;
    let b: f64 = 1.0;
    a + b   // silently returns Logic<f64> with no provenance warn
}
```

**Recommended fix**:
Extend the Logic gate to cover wrap-asymmetric (parallel to the
B:C6 fix for D):

```python
elif (l_is_logic or r_is_logic) and (
        inner_mismatch
        or not (l_is_logic and r_is_logic)  # wrap-asymmetric
):
    # Either inner mismatch OR one side bare. Both cases drop
    # provenance silently without this warn.
    ...
    if not (l_is_logic and r_is_logic):
        extra = " (one side Logic-wrapped, other bare)"
    ...
```

Or split into two arms — one for mixed inner, one for wrap-
asymmetric — with distinct trap text. Same shape as the cycle-2
B:C6 close.

---

### Finding E3: `ShapeFoldError` only caught by struct_mono; fn-mono path surfaces it as "compiler bug"

**File**: `helixc/frontend/monomorphize.py:63-72` (ShapeFoldError);
`helixc/frontend/monomorphize.py:677-679` (entry `monomorphize`);
`helixc/backend/x86_64.py:3021` (fn-mono call site, no catch);
`helixc/check.py:272-283` (top-level `except Exception`).
**Severity**: MEDIUM
**Category**: exception contract asymmetry / diagnostic-quality
regression on legitimate user error

**Description**:
Cycle 3 C3-6 introduced `ShapeFoldError(ValueError)` raised from
`_fold_intlit_arith` on `/0` and `%0` (trap 28801). The struct_mono
caller catches it at struct_mono.py:448 and surfaces it as a
diagnostic via `diags.append(str(e))`.

But `_fold_intlit_arith` is reachable from any `_subst_shape_expr`
invocation, which is called from `substitute_ty` (monomorphize.py:
239), which is called from fn-mono's `_instantiate` (line 648-651).
`_instantiate` is invoked from `Monomorphizer._rewrite_calls_in_expr`
at line 485, with no try/except wrapper. `Monomorphizer.run()` (line
403-429) doesn't catch either. The top-level entry
`monomorphize(prog)` at line 677-679 also doesn't catch. The x86_64
driver invokes it at line 3021 with no wrapper.

So if a generic fn has a shape expression with `/0` or `%0` after
substitution, ShapeFoldError propagates out through fn-mono and is
caught by the cycle-3 C3-3 top-level `except Exception` in check.py
at line 274, which prints:

```
helixc: internal error: ShapeFoldError: <message>
helixc: this is a compiler bug — please file an issue.
```

This is a **false bug report**: the user wrote source that triggered
a legitimate trap 28801 (shape fold by zero), and the compiler tells
them to file an issue instead of pointing at the array dim.

**Reproducer**:
```helix
fn f[N](a: [i32; N / (N - N)]) -> i32 { 0 }  // N - N = 0
fn main() -> i32 { f::<5>([0; 1]) }
```

At fn-mono time, `N / (N - N)` substitutes to `5 / (5 - 5)` =
`5 / 0`, `_fold_intlit_arith` raises ShapeFoldError. fn-mono
doesn't catch. Top-level `except Exception` catches and prints
"internal error" + "compiler bug".

**Recommended fix**:
Wrap `_instantiate` in `Monomorphizer.run()` (or at the
`_rewrite_calls_in_expr` call site) with a try/except for
ShapeFoldError, accumulate diagnostics on the Monomorphizer
instance, and surface via a return-value channel parallel to
`monomorphize_structs`'s `(prog, diags)` tuple. Then the top-level
driver consumes diags and prints them as trap 28801 with
file/line/col context instead of "internal error".

Minimal patch (defer the channel redesign, just catch + print):

```python
# In Monomorphizer.run() at the inner loop:
try:
    new_body = self._rewrite_calls_in_block(item.body, item)
except ShapeFoldError as e:
    print(f"error: {e}", file=sys.stderr)
    self.errors.append(str(e))  # accumulate
    continue
```

Plus update `monomorphize()` signature to return `(int, list[str])`
or just `int` with stderr printing. Either preserves the trap-id
attachment.

---

### Finding E4: D3 diagnostic text drifts — negative branch says ">= 0", zero branch says "> 0"

**File**: `helixc/frontend/typecheck.py:579-608` (`_resolve_size_expr`
D3 IntLit and Unary(-, IntLit) arms).
**Severity**: LOW
**Category**: diagnostic-quality / invariant-expression clarity

**Description**:
The D3 fix added two near-identical IntLit-validation arms (line
579-591 for IntLit, line 593-608 for Unary(-, IntLit)). Each emits
two different diagnostics depending on whether the value is negative
or zero:

- `< 0` → `"array size must be >= 0, got -6"` (line 582)
- `== 0` → `"array size must be > 0 in Phase-0, got 0"` (line 588)

The diagnostic claims the rule is `>= 0` for the negative case but
`> 0` for the zero case. These contradict each other from the user's
perspective: which rule actually applies?

The real rule (per the docstring at line 567-575) is "size must be
> 0 in Phase-0". The negative branch should say the same:

```
"array size must be > 0 in Phase-0, got -6 (trap 28802)"
```

This is a small invariant-expression issue: the type's contract
("size > 0") isn't communicated cleanly. A confused user could
read the negative diagnostic and try `[T; 0]` thinking that's
allowed (since `>= 0` would include zero), then hit the zero
diagnostic with a different rule.

**Reproducer**:
```python
from helixc.frontend.typecheck import TypeChecker
from helixc.frontend import ast_nodes as A
tc = TypeChecker(A.Program(span=A.Span(0,0), items=[]))
tc._resolve_size_expr(A.IntLit(span=A.Span(0,0), value=-6),
                      tc._root_scope())
# error: "array size must be >= 0, got -6 (trap 28802)"

tc._resolve_size_expr(A.IntLit(span=A.Span(0,0), value=0),
                      tc._root_scope())
# error: "array size must be > 0 in Phase-0, got 0 (trap 28802)"
```

Same rule, different phrasing.

**Recommended fix**:
Unify the diagnostic text. One concise message covering both:

```python
if expr.value <= 0:
    self.errors.append(TypeError_(
        f"array size must be > 0 in Phase-0, got {expr.value} "
        f"(trap 28802)",
        expr.span,
    ))
return TyPrim(f"size_{expr.value}")
```

Same change for the Unary(-, IntLit) arm. This collapses two arms
to one each, removes the negative/zero inconsistency, and makes
the type-invariant ("size > 0") obvious from the diagnostic.

---

### Finding E5: Parser tag-12 sentinel overlaps type-tag namespace

**File**: `helixc/bootstrap/parser.hx:2325-2333` (D2 tag-12 sentinel);
`helixc/bootstrap/parser.hx:2281-2292` (existing type tags 0-11).
**Severity**: LOW
**Category**: namespace / encoding overlap — design smell

**Description**:
Cycle 3 D2 added a sentinel value (tag 12) at the let-inference
site to mean "untracked Call RHS, assume non-i32 for closure-
capture trap purposes". The existing type tags 0-11 are reserved
for actual primitive types:

```
 0 = i32 (trivially safe)
 1 = f32
 2 = f64
 3 = i64
 4 = bf16
 6 = u32
 7 = u8
 8 = u16
 9 = u64
10 = i8
11 = i16
```

Tags 5 and 12+ are unassigned. The D2 fix picked 12 as the
"untracked-call" sentinel. The capture-site guard at parser.hx:1810
is `if cap_ty_tag > 0`, which currently treats 12 the same as 1-11
(non-i32 → trap 76003).

This works today. But the encoding mixes two concepts:
1. **Type-tag** (concrete prim type known at let-inference time)
2. **Provenance-tag** (concrete prim type NOT known, but inferred
   to be probably-non-i32)

If a future primitive type is added (say `char` = tag 12, or a new
quantized type), the encoding silently collides. Worse, the
documentation at line 2281-2292 enumerates 0, 1, 2, 3, 4, 6, 7, 8,
9, 10, 11 — leaving 5 and 12 conspicuously absent. A reader who
sees tag 5 vs tag 12 wouldn't know the conceptual difference.

This is a small design-smell — the sentinel space and the type-tag
space share an encoding without a clear separator.

**Reproducer**: conceptual — add a tag 12 prim and observe that
existing tag-12 lookups (literal-RHS untracked-call sentinel) now
get conflated with the new prim type.

**Recommended fix**:
Move sentinels into a clearly-out-of-band range, e.g., 100+:

```
 0-11: concrete prim types
100: untracked-call sentinel
101: untracked-other sentinel
```

Plus document the boundary at the file header. Then the capture-
site guard stays `> 0` (treats both type-tags and sentinels as
"non-i32"), but the encoding is non-overlapping and future-proof.

Alternative: split into two parallel tables. One for type-tags
(values 0-11 only), one for provenance (boolean is_safe). Cost:
more tables, more `var_type_tab_lookup` calls. Phase-0's stride-3
`cl_capture_tab` deferred from cycle 1 was always the long-term
answer; this is just a band-aid.

---

### Finding E6: `_inline_lets` Call arm drops `callee.generics` on aliased-Name substitution

**File**: `helixc/frontend/autodiff.py:559-570` (Call arm of C3-5
extension).
**Severity**: LOW
**Category**: latent invariant drop in autodiff lowering

**Description**:
Cycle 3 C3-5 added a Call arm to `_inline_lets` that substitutes
alias-of-name callees (when `expr.callee` is a Name and the env
substitution is also a Name/Path). The substitution at line 569
is:

```python
new_callee = cand  # cand is A.Name or A.Path from env
```

If `expr.callee` had a `generics` list (turbofish, e.g.
`f::<i32>(x)`), and the env binds `f` to `Name('g')`, then
`new_callee = Name('g')` — the turbofish list `[i32]` is dropped.

This means: `let f = g; f::<i32>(x)` after inlining becomes
`g(x)` instead of `g::<i32>(x)`. The mono pass downstream would
not pick up the turbofish, producing a non-turbofished call to
generic `g`.

In practice, `let f = some_fn` isn't a common pattern (Phase-0 has
no first-class fn aliasing through let), but the parser does
accept it. And the autodiff `_propagate` reach passes through
`_inline_lets` for the reverse-mode warn paths — any aliasing of
generic fns there would lose turbofish information.

**Reproducer**:
```python
# Construct AST manually:
e = A.Call(span=s, callee=A.Name(span=s, name='f', generics=[ty_i32]),
           args=[...])
env = {'f': A.Name(span=s, name='g', generics=[])}
out = _inline_lets(e, env)
# out.callee.generics is []  — lost the [i32]
```

**Recommended fix**:
When aliasing, preserve the original callee's `generics` list (the
alias is a Name; the original turbofish stays on it):

```python
if isinstance(expr.callee, A.Name) and expr.callee.name in env:
    cand = env[expr.callee.name]
    if isinstance(cand, (A.Name, A.Path)):
        # Preserve the original turbofish: alias-of-name
        # shouldn't strip explicit type-args at the call site.
        if isinstance(cand, A.Name):
            new_callee = A.Name(
                span=cand.span, name=cand.name,
                generics=list(expr.callee.generics) or cand.generics,
            )
        else:
            new_callee = cand
```

---

### Finding E7: `_inline_lets` missing TileLit arm — catch-all `_ad_warn` over-fires

**File**: `helixc/frontend/autodiff.py:680-687` (catch-all);
`helixc/frontend/ast_nodes.py:344-356` (TileLit).
**Severity**: LOW
**Category**: AST coverage gap in C3-5 widening

**Description**:
The cycle-3 C3-5 widening added arms for most Expr subtypes
(Cast / Call / Field / Index / ArrayLit / TupleLit / StructLit /
Range / Return / Break / Assign / UnsafeBlock / Match / Loop / For
/ While / Quote / Splice / Modify). The catch-all at line 681-686
emits an `_ad_warn` ("trap 85001 — let-bindings beyond this point
may not be substituted").

But `TileLit` (Expr subclass at ast_nodes.py:344, with `shape:
list["Expr"]` that can contain Name leaves) is missing. Any
TileLit appearing under a let-binding triggers the catch-all
warn, which is intended as a "loud failure" for unknown AST
extensions but fires for a legitimate Phase-0 AST node.

**Reproducer**:
```helix
fn f() -> tile<f32, [4, 4], REG> {
    let zero_tile = tile<f32, [4, 4], REG>::zeros();
    zero_tile
}
```

If this passes through `_inline_lets` (autodiff or reverse-mode
context), the TileLit `tile<f32, [4, 4], REG>::zeros()` hits the
catch-all and emits an AD warning that the user can't fix
(they're not doing anything wrong; the AST is just incomplete).

**Recommended fix**:
Add an explicit TileLit arm:

```python
if isinstance(expr, A.TileLit):
    return A.TileLit(
        span=expr.span,
        dtype=expr.dtype,
        shape=[_inline_lets(s, env) for s in expr.shape],
        memspace=expr.memspace,
        init=expr.init,
    )
```

Also consider adding Continue (no inner expr, just return as-is)
and Path (no inner expr, return as-is) arms so the catch-all
doesn't fire on legitimate constructs. Or shrink the catch-all
to a less-alarming "log-only" trace.

---

### Finding E8: `_inline_lets` Call arm doesn't walk Field-typed callees

**File**: `helixc/frontend/autodiff.py:559-570` (Call arm).
**Severity**: LOW
**Category**: substitution reach gap

**Description**:
The C3-5 Call arm only substitutes / walks when `expr.callee` is
a Name (line 566). If `expr.callee` is a Field (method call:
`obj.method()` parses as `Call(callee=Field(obj=Name('obj'),
name='method'), args=[...])`), the `obj` is NOT walked through
`_inline_lets`. A let-binding for `obj` is silently dropped at
this boundary.

```helix
fn f() {
    let me = self;
    me.method()       // me.method() → Call(Field(Name('me')))
                      // 'me' is NOT substituted with 'self'
}
```

The `_propagate` reverse-mode warn reach passes through
`_inline_lets`. Any reverse-mode AD that traverses a method call
on a let-bound receiver misses the binding.

**Reproducer**: similar shape to C3-5 the canonical reproducer —
`let r = (x as i64) % 2_i64; r.some_method()` would fail to
substitute `r` because the Call's callee is Field(Name('r'),
'some_method'), not Name('r').

Pre-cycle-3, the entire Call arm was a no-op, so this is no
worse than before, but the C3-5 fix targeted the call-args case
specifically. The Field-typed-callee case is the natural next
gap.

**Recommended fix**:
Walk `expr.callee` recursively through `_inline_lets` regardless of
its top-level type, then apply the alias-of-name special-case only
when the post-walked callee is a Name in env:

```python
if isinstance(expr, A.Call):
    new_args = [_inline_lets(a, env) for a in expr.args]
    new_callee = _inline_lets(expr.callee, env)
    # Then: alias-of-name special case for Name → Name/Path
    if isinstance(new_callee, A.Name) and new_callee.name in env:
        cand = env[new_callee.name]
        if isinstance(cand, (A.Name, A.Path)):
            new_callee = cand
    return A.Call(span=expr.span, callee=new_callee, args=new_args)
```

This covers Field-callee, Index-callee, and any other expression
that resolves to a Name through inlining.

---

## Cycle 3 fix re-verification

| Fix    | Status     | Notes                                                                                                       |
|--------|------------|-------------------------------------------------------------------------------------------------------------|
| C3-1   | OK         | grad_pass `_rewrite_in_expr` now handles `else_ = A.If` (chained `else if`). Mirror of `_resolve_in_expr`. No new issues. |
| C3-2   | OK         | `_WIDEN_NAME_ALIASES` + `_widen_canon_name` collapse isize↔i64, usize↔u64 so same-rank tie no longer double-fires. tie-callback / outer-emit dedup via `tie_fired[0]` flag works as designed. |
| C3-3   | OK*        | `_main_inner` wrapped in try/except/finally. **However**, the bare `except Exception` catches ShapeFoldError (a legitimate user-error trap 28801) and surfaces it as "compiler bug" via the fn-mono path — see finding E3. |
| C3-4   | OK         | `monomorphize_structs` dedups against `existing` set so second invocation is a no-op. Idempotency verified. ShapeFoldError caught with `except ShapeFoldError as e:` before the generic ValueError arm so trap-id stays attached. |
| C3-5   | OK*        | `_inline_lets` recurses through Cast / Call / Field / Index / Match / etc. Catch-all `_ad_warn` for unknown subtypes works. **However**, TileLit is missing (finding E7), Call-on-Field doesn't walk obj (E8), aliased-Name turbofish dropped (E6). |
| C3-6   | OK         | `ShapeFoldError(ValueError)` raised on `/0` and `%0` in `_fold_intlit_arith`. Trap 28801 reserved. struct_mono catches and surfaces as diagnostic. (Catch-contract gap covered in E3.) |
| D1     | OK*        | `_check_call_basic` falls through to `_compatible` for non-prim pairs. `_compatible` has structural arms for TyDiff / TyLogic / TyTuple / TyArray / TyRef / TyPtr / TyFn. **However**, TyArray arm uses raw `==` for size where every other composite uses recursive `_compatible` — false-positive risk on TySize / TyUnknown sizes (finding E1). |
| D2     | OK         | parser.hx tag-12 sentinel for Call-RHS lets makes trap 76003 fire. Function-param i32 captures still pass (intended, documented). Sentinel-encoding namespace overlap is finding E5 (LOW). |
| D3     | OK*        | `_resolve_size_expr` emits trap 28802 on IntLit `< 0` and `== 0`, plus Unary(-, IntLit) source-level case. **However**, diagnostic text drifts (negative says ">= 0", zero says "> 0") — finding E4. |
| D4     | OK*        | Logic-Logic mixed-inner emits AD warn with `[Logic-domain]` tag. **However**, Logic-wrap asymmetric (`Logic<f64> + f64`) still silent — finding E2. Parallel to the cycle-2 B:C6 → cycle-3 D4 pattern repeating one cycle later. |
| D5     | OK         | `_fold_intlit_unary` symmetric to `_fold_intlit_arith` folds `Unary(-, IntLit(N))` → `IntLit(-N)`. Composition through nested `Binary` / `Unary` verified. No new issues. |
| D6     | OK         | `_ty_key` raises `TypeError` on non-AST.TyNode inputs (loud failure rather than silent dedup). No new issues. |
| D7     | OK         | `_check_cast_compat` iteratively peels matching ref-pairs; depth-8 guard emits trap 28803. No more RecursionError risk. No new issues. |
| D8     | OK         | `_fmt` TyStruct arm returns `t.name` (e.g. `Foo` not `TyStruct(name='Foo')`). No new issues. |
| D9     | OK         | `_walk_subst_expr` Call arm substitutes `callee.generics` so turbofish inside generic body resolves correctly at mono time. No new issues. |

Five cycle-3 fixes have residual gaps (`*`):
- **C3-3** catches ShapeFoldError as "compiler bug" via fn-mono escape path (E3 covers).
- **C3-5** has three small gaps in the wide-recursion (E6, E7, E8 cover).
- **D1** has the TyArray size symmetry issue (E1).
- **D3** has the diagnostic-text drift (E4).
- **D4** has the wrap-asymmetric Logic case (E2).

The other ten fixes hold up under re-audit with no caveats.

---

## Cycle 4 focus question answers

**1. D1's widened `_compatible` for non-prim pairs — symmetric?**

**Almost**. TyTuple / TyRef / TyPtr / TyFn / TyDiff / TyLogic all
recurse via `self._compatible` for inner types. TyArray is the
outlier — it uses raw `==` for size while recursing on elem. This
breaks the cascade-safe invariant when one side has a TyUnknown or
TySize size (finding E1). The other arms are sound.

**2. ShapeFoldError surface paths — fully bounded?**

**No**. struct_mono catches it; fn-mono / `Monomorphizer.run()`
does not. A generic fn with `[T; N/0]` in its param/return type
hits the cycle-3 C3-3 top-level `except Exception` and surfaces as
"compiler bug" instead of trap 28801 (finding E3). The exception
class's catch contract is asymmetric across the two mono callers.

**3. D4 Logic-domain warning — symmetric with the D-bare path?**

**No**. The cycle-2 B:C6 fix closed the D-wrap-asymmetric case
(D<f64> + f64 now warns). The cycle-3 D4 fix closed only the
Logic-Logic mixed-inner case. Logic-wrap-asymmetric (Logic<f64> +
f64) remains silent (finding E2). Same shape as the
cycle-2 → cycle-3 pattern, repeating one cycle later for Logic.

**4. C3-5 `_inline_lets` widening — covers every Name-bearing Expr?**

**Almost**. The 17 added arms cover the common cases. Three gaps:
TileLit (E7, catch-all over-fire on legal tile syntax), Field-typed
callees in Call (E8, `obj.method()` won't substitute let-bound
`obj`), aliased-Name turbofish (E6, `let f = g; f::<i32>(x)` loses
the turbofish).

**5. Parser tag-12 sentinel — collision-safe encoding?**

**Borderline**. Today tag 12 is unused as a primitive-type tag, so
the sentinel encoding works. But the type-tag namespace 0-11 is
already documented at line 2281-2292; tag 5 and tag 12 are
"holes" without explicit semantics. A future addition of a prim
at tag 12 would silently break the closure-capture trap (finding
E5).

---

## Cycle 4 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity.**

This cycle finds **8 new findings (0 HIGH, 3 MEDIUM, 5 LOW)**. By
the strict criterion, **cycle 4 does NOT count clean**. Zero HIGH
findings is a real improvement over cycle 3 (which had 2 HIGH). The
three MEDIUMs are the next-layer-down asymmetries that cycle-3's
fixes opened by widening structural-equality / exception / autodiff-
substitution contracts. Each is a partial-fix-of-prior-fix in the
same shape as the cycle-1 → cycle-2 and cycle-2 → cycle-3 escalation
patterns.

Recommended fix sequence for cycle 5:

1. **E1 first** (TyArray symmetry — one-line change in
   `_compatible` to recurse on size; tightens the structural-
   equality invariant uniformly across all composite types).
2. **E2 with E1** (Logic-wrap asymmetric — small extension to the
   D4 gate, parallel to the B:C6 close for D).
3. **E3** as a separate commit (ShapeFoldError catch contract —
   small refactor to either wrap `Monomorphizer.run()` or surface
   diagnostics through a `(int, list[str])` return tuple).
4. **E4 / E5 / E6 / E7 / E8** as a low-severity cleanup commit.
   E4 is a one-line message merge; E5 is a parser.hx
   re-numbering; E6/E7/E8 are localized `_inline_lets` patches.

Once E1-E3 land + verify, cycle 5 can re-audit. Per the strict
criterion, the remaining cycles needed to declare type-design
soundness clean is at least one full clean sweep after addressing
all 8 findings — pending whether new gaps surface from the fixes
themselves (cycle 1 → 2 → 3 → 4's pattern of "fix exposes the
next layer" continuing).
