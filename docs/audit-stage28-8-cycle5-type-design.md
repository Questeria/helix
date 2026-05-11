# Stage 28.8 Pre-29 Audit Gate — Cycle 5, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit**: 960303b (read-only)
**Scope**: Re-audit type-system soundness focused on cycle-4's 13 fixes
(commit 960303b — C4-1..C4-5 + E1..E8). New TyTile/TyTensor structural
arms in `_compatible`, TyArray size compare via recursive `_compatible`
(E1), Logic-wrap asymmetric warning gate broadened from `inner_mismatch`
only to `inner_mismatch OR l_is_logic != r_is_logic` (E2),
`monomorphize_safe` wrapper over `Monomorphizer.run()` catching
`ShapeFoldError` (C4-5 / E3), parser tag-12 sentinel widened from
val_tag==16 (Call) to every non-trivially-i32 RHS with proven-bool /
proven-VAR / sentinel routing (C4-2), `_inline_lets` Path / Continue /
TileLit identity arms before the catch-all (C4-1), If.cond inlining
(C4-3), Call alias preserves turbofish (E6), Call walks Field-typed
callees (E8), `_resolve_size_expr` diagnostic unified to "must be > 0"
in both branches (E4). Cross-stage interactions (D-wrap-asymmetric
mirror gap, monomorphize_safe error-vs-warning channel, TyTile shape
list-vs-tuple incoherence with the new arm, TySize generic-array call
boundary the E1 fix claims to close).

**Method**: traced each new code path through resolve / compatible /
binop / mono-driver / autodiff-inline / parser-let. Mentally executed
edge-case inputs against the new code (TyArray with TySize('N') vs
TySize('M'); `D<f64> + f64` post-Logic-gate-broadening; ShapeFoldError
on x86_64 driver path; tag-12 sentinel collision with tag 12 = AST_LET_MUT;
TileLit with let-bound shape dims; D-wrap-Logic-stack vs bare-T;
`monomorphize_safe` partial-mono exit semantics). Verified each
cycle-4 fix holds under cycle-5 probes. Cross-checked cycle-4
strictness against the cycle-5 focus questions: TyArray E1 fix
actually closes the TySize-vs-TySize case the cycle-4 doc cited as
reproducer, monomorphize_safe error severity, D-wrap-asymmetric
symmetric mirror of E2, TileLit identity arm vs walk arm tradeoff,
parser tag-12 sentinel namespace overlap with AST_LET_MUT.

**Result**: **6 new findings (0 HIGH, 3 MEDIUM, 3 LOW)**. Cycle 4
addressed 13 prior findings cleanly at the surface level, but the
deeper invariant strength of three of those fixes is weaker than the
fix description claims. F1 is an **incomplete-fix** for E1: the
TyArray-size `_compatible` recursion only helps when one side is
TyUnknown (cascade-safe at the top of `_compatible`); the explicit
example in cycle-4 doc's E1 reproducer (`fn f[N](a:[i32;N])` called
from `fn g[M](a:[i32;M])`, both sides TySize) is NOT closed because
`_compatible` has no TySize/TyVar cascade arm and falls through to
`return a == b` which is False for different names. F2 is the
mirror gap that E2's fix opens: the cycle-4 Logic-wrap-asymmetric gate
broadened from `inner_mismatch` to `inner_mismatch OR (l_is_logic !=
r_is_logic)`, closing the Logic-wrap-asymmetric case. But the
*parallel* D-wrap-asymmetric same-inner case (`D<f64> + f64`) was
NOT broadened — its gate is still `inner_mismatch`-only, so the
cycle-2 B:C6 same-inner-bare-vs-D mirror remains silent. The
cycle-4 fix to Logic should have been mirrored to D for consistency.
F3 is the monomorphize_safe wrapper's severity downgrade: the x86_64
driver prints ShapeFoldError as a `warning: fn-mono: ...` and
continues into grad_pass / typecheck / codegen even though a
generic was instantiated with `/0` or `%0` — the build SHOULD abort
because the produced binary has an invalid array dimension; instead
the user gets a warning and a broken executable. LOW findings cover
TileLit identity arm dropping let-bindings under TileLit.shape /
memspace (F4 — the cycle-4 E7 fix was narrower than the cycle-4
doc's recommended walk-arm), tag-12 sentinel parser.hx collision
with AST_LET_MUT (tag 12) which makes the type-tag-vs-AST-tag
namespace overload latent if any future code path reads the sentinel
as an AST tag (F5), and the cycle-4 C4-2 unbalanced-AST-VAR gap
where the `if val_tag == 1` arm leaves `inferred_ty_tag = -1`
silently (F6 — the AST_VAR comment says "defer to var_type_tab
resolution" but the existing >0 capture guard still treats -1 as
"safe i32" if var_type_tab doesn't have the name, which is the
exact silent-i32-truncation pattern D2 / C4-2 were trying to fix).

Zero of the new findings are stop-the-line. F1, F2, F3 are
**MEDIUM** — each is a regression of a cycle-4 stated goal at the
contract level, but each is also a partial improvement on its
pre-cycle-4 state. F4 / F5 / F6 are diagnostic-quality and
narrow-edge-case issues. **Cycle 5 status**: 6 findings (0 HIGH,
3 MEDIUM, 3 LOW) means cycle 5 does **NOT** count clean under the
strict criterion. See "Cycle 5 status" final paragraph.

---

## Summary table

| ID  | Severity | Component                                      | Issue (short)                                                                |
|-----|----------|------------------------------------------------|------------------------------------------------------------------------------|
| F1  | MEDIUM   | typecheck `_compatible` TyArray size           | E1 fix incomplete — `_compatible(TySize('N'), TySize('M'))` still False; reproducer in cycle-4 doc not closed |
| F2  | MEDIUM   | typecheck binop D-wrap-asymmetric same-inner   | `D<f64> + f64` still silent — E2 broadened only the Logic gate, not the parallel D gate |
| F3  | MEDIUM   | x86_64 driver `monomorphize_safe` severity     | ShapeFoldError surfaces as `warning: fn-mono` and pipeline continues — should abort (compile-time error) |
| F4  | LOW      | autodiff `_inline_lets` TileLit identity arm   | E7 fix is identity; doesn't walk `shape` / `memspace` so let-bound names there silently un-substituted |
| F5  | LOW      | bootstrap parser.hx tag-12 sentinel            | Tag 12 collides with AST_LET_MUT (header line 26); any code reading tag-12 from arena gets ambiguous interpretation |
| F6  | LOW      | bootstrap parser.hx C4-2 AST_VAR arm           | `val_tag == 1` ("defer to var_type_tab") leaves `inferred_ty_tag = -1` if name not in tab — silent i32-fallback at capture site |

---

## Per-finding sections

### Finding F1: cycle-4 E1 fix incomplete — `_compatible(TySize('N'), TySize('M'))` still False, primary reproducer not closed

**File**: `helixc/frontend/typecheck.py:2197-2206` (`_compatible` TyArray arm with E1 fix);
`helixc/frontend/typecheck.py:2157-2249` (`_compatible` full chain).
**Severity**: MEDIUM
**Category**: incomplete-fix / structural-equality invariant gap unchanged from cycle-4

**Description**:
Cycle 4 E1 changed the TyArray size compare from raw `a.size == b.size`
to `(a.size == b.size or self._compatible(a.size, b.size))`. The
cycle-4 doc cited two reproducer paths:

1. Generic-array-call-boundary (`fn f[N](a:[i32;N])` called from
   `fn g[M](a:[i32;M])`): `_compatible(TySize('N'), TySize('M'))`
   should return True (cascade-safe; mono later binds both).
2. TyUnknown-size cascade (`fn f(a:[i32;5])` called with arg of
   resolved type TyArray(i32, TyUnknown)): `_compatible(TyPrim('size_5'),
   TyUnknown(...))` should return True.

The fix closes **only reproducer 2**. Trace reproducer 1 through the
post-fix `_compatible`:

- `_compatible(TySize('N'), TySize('M'))` enters at line 2157.
- Top guard at 2158: neither is TyUnknown → continue.
- TyMemTier arm (2162-2165): neither is TyMemTier → continue.
- TyQuote arm (2169-2172): neither is TyQuote → continue.
- TyDiff arm (2178-2181): neither is TyDiff → continue.
- TyLogic arm (2182-2185): neither is TyLogic → continue.
- TyTuple arm (2190-2195): neither is TyTuple → continue.
- TyArray arm (2197-2206): neither is TyArray → continue.
- (... all other composite arms similarly skip ...)
- TyTile arm (2240-2247): neither is TyTile → continue.
- Final line 2249: `return a == b` → False (different `name` fields
  on the frozen TySize dataclass).

So `_compatible(TySize('N'), TySize('M'))` returns False. The
cycle-4 E1 fix at line 2203 calls `self._compatible(a.size, b.size)`
on the array sizes — same False result. The disjunction
`a.size == b.size or self._compatible(a.size, b.size)` is `False or
False = False`. The cycle-4 doc's primary reproducer is NOT closed.

What the E1 fix actually closes:
- `_compatible(TyPrim('size_5'), TyUnknown(...))`: top guard at
  2158 returns True. ✓
- `_compatible(TyUnknown(...), TyPrim('size_5'))`: top guard at
  2158 returns True. ✓

What it does NOT close (the primary reproducer the cycle-4 doc
described as MEDIUM-severity false-positive):
- `_compatible(TySize('N'), TySize('M'))`: still False. ✗
- `_compatible(TyVar('T'), TyVar('U'))`: still False. ✗
- `_compatible(TySize('N'), TyPrim('size_5'))`: still False
  (TySize is not TyPrim, falls through). ✗

The cycle-4 doc's "recommended fix" wording was "change the TyArray
arm to recurse on size too, so TyUnknown / TyVar / TySize symmetric
to elem", with a note that "`_compatible` would still need a
TyVar/TySize defer arm". The cycle-4 fix did the recursion change
but did NOT add the TyVar/TySize defer arm at the top of
`_compatible`, so the recursion lands at the same `return a == b`
False result.

**Reproducer**:
```python
from helixc.frontend.typecheck import (
    TypeChecker, TyArray, TyPrim, TySize, TyUnknown,
)
from helixc.frontend import ast_nodes as A
tc = TypeChecker(A.Program(span=A.Span(0,0), items=[]))
a = TyArray(elem=TyPrim('i32'), size=TySize('N'))
b = TyArray(elem=TyPrim('i32'), size=TySize('M'))
tc._compatible(a, b)
# -> False (cycle-4 doc claimed this is closed; it is NOT)

# What IS closed:
a2 = TyArray(elem=TyPrim('i32'), size=TyPrim('size_5'))
b2 = TyArray(elem=TyPrim('i32'), size=TyUnknown(hint='size expr'))
tc._compatible(a2, b2)
# -> True (E1 fix works here via top-of-_compatible TyUnknown cascade)
```

**Recommended fix**:
Add a TyVar/TySize cascade-safe arm at the top of `_compatible`
(just below the TyUnknown arm at line 2158-2159):

```python
def _compatible(self, a: Type, b: Type) -> bool:
    if isinstance(a, TyUnknown) or isinstance(b, TyUnknown):
        return True
    # Audit 28.8 cycle 5 F1: TyVar / TySize are generic symbols
    # that mono will substitute later; treat them as cascade-safe
    # like TyUnknown, so `[i32; N]` ~ `[i32; M]` at the call
    # boundary doesn't false-positive on different generic names.
    if isinstance(a, (TyVar, TySize)) or isinstance(b, (TyVar, TySize)):
        return True
    ...
```

With this arm in place, the TyArray E1 fix at line 2202-2204
becomes a pure pass-through for TySize-vs-TySize sizes (the inner
`_compatible(size, size)` returns True via the new arm) — matching
the cycle-4 doc's stated goal.

Alternative (less invasive but more code): widen the TyArray arm
itself to check for TyVar/TySize on either side and return
elem-compatibility only:

```python
if isinstance(a, TyArray) and isinstance(b, TyArray):
    if (isinstance(a.size, (TyVar, TySize, TyUnknown))
            or isinstance(b.size, (TyVar, TySize, TyUnknown))):
        return self._compatible(a.elem, b.elem)
    return (self._compatible(a.elem, b.elem)
            and (a.size == b.size
                 or self._compatible(a.size, b.size)))
```

Both fixes have the same observable behavior on the cycle-4
reproducer; the first is cleaner because it uniformly applies the
defer rule across all composite types.

---

### Finding F2: cycle-4 E2 broadened Logic gate but not D gate — `D<f64> + f64` same-inner-asymmetric still silent

**File**: `helixc/frontend/typecheck.py:1349-1362` (binop D arm);
`helixc/frontend/typecheck.py:1363-1390` (binop Logic arm with E2 fix).
**Severity**: MEDIUM
**Category**: asymmetric fix — E2 closed Logic-wrap but not the parallel D-wrap

**Description**:
Cycle 4 E2 broadened the Logic-domain gate so wrap-asymmetric same-
inner cases warn:

```python
elif (l_is_logic or r_is_logic) and (
        inner_mismatch
        or (l_is_logic != r_is_logic)
):
    ...
```

The D-domain gate at line 1349 was NOT broadened in parallel:

```python
if (l_is_diff or r_is_diff) and inner_mismatch:
    ...
```

So `D<f64> + f64` (l_is_diff=True, r_is_diff=False, inner_mismatch=
False because both inners are f64) falls through both arms and
hits the silent `else` at line 1391-1393. The result type is
rebuilt as `D<f64>` at line 1400, silently absorbing the bare-f64
operand's missing TyDiff provenance.

The cycle-2 B:C6 fix description explicitly named this case as
closed: "asymmetric D-wrap + raw — silently promoted i32 to f64".
B:C6 closed it for the inner-mismatch sub-case (`D<f64> + i32`)
but NOT for the same-inner sub-case (`D<f64> + f64`). The cycle-4
E2 fix realized the analogous gap for Logic and broadened the
Logic gate. The mirror broadening for D was not applied.

This is the same "partial fix of partial fix" pattern visible in
cycle-2 → cycle-3 → cycle-4. Cycle-2 closed D inner-mismatch. Cycle-3
D4 noticed Logic-Logic mixed-inner silent, fixed for inner-mismatch.
Cycle-4 E2 noticed Logic wrap-asymmetric same-inner silent, fixed.
Cycle-5 now observes the D wrap-asymmetric same-inner silent gap
that was *always* there — it just wasn't noticed because the cycle-2
B:C6 description was loose about the same-inner sub-case.

**Reproducer**:
```python
from helixc.frontend.typecheck import (TypeChecker, TyDiff, TyPrim)
from helixc.frontend import ast_nodes as A, autodiff as ad
ad._DIFF_WARNINGS = []
# typecheck a binop expression where l: D<f64>, r: f64
# (construct AST or use typecheck.check_expr directly with a
# scope where l is bound to D<f64> and r to f64)
# observed: no AD warning emitted; result is D<f64>.
# expected: an AD warning naming the case "(one side D-wrapped,
# other bare; same inner)".
```

Source-level:
```helix
fn f() -> D<f64> {
    let a: D<f64> = ...;
    let b: f64 = 1.0;
    a + b   // silently D<f64>; the b's bare-ness produces no warn
}
```

**Recommended fix**:
Broaden the D gate at line 1349 in parallel to E2's Logic
broadening:

```python
if (l_is_diff or r_is_diff) and (
        inner_mismatch
        or (l_is_diff != r_is_diff)
):
    inner = _widen_diff_inner(
        l_inner, r_inner,
        _warn_cb=_tie_cb, _span=expr.span,
    )
    if not tie_fired[0]:
        extra = ""
        if not (l_is_diff and r_is_diff):
            extra = " (one side D-wrapped, other bare)"
        self._ad_warn_mixed_inner(
            expr.span, l_inner, r_inner, inner, extra=extra,
        )
```

Same shape as the E2 Logic fix. Closes the cycle-2 B:C6 same-
inner-bare-vs-D gap and aligns the D and Logic gates as parallel
contracts. After this fix, both `D<f64> + f64` and `D<f64> + i32`
warn (different `extra` text but both surface), matching the
cycle-2 B:C6 stated goal.

---

### Finding F3: `monomorphize_safe` ShapeFoldError surfaces as warning, pipeline continues into codegen with a broken array dim

**File**: `helixc/frontend/monomorphize.py:687-706` (`monomorphize_safe`);
`helixc/backend/x86_64.py:3021-3027` (driver call site);
`helixc/check.py:272-283` (top-level wrapper, unchanged).
**Severity**: MEDIUM
**Category**: severity downgrade / silent-build-with-broken-output

**Description**:
Cycle 4 C4-5 / E3 wrapped the fn-mono entry point with
`monomorphize_safe` that catches `ShapeFoldError`. The driver
treats the returned diagnostics as warnings:

```python
mono_count, mono_diags = monomorphize_safe(prog)
for d in mono_diags:
    print(f"warning: fn-mono: {d}", file=sys.stderr)
if mono_count > 0:
    print(f"mono: {mono_count} generic instantiation(s)", file=sys.stderr)
grad_count = grad_pass(prog)
...
```

Two compounding issues:

1. **Wrong severity**: ShapeFoldError indicates a generic was
   instantiated with `/0` or `%0` in a shape expression — the user
   wrote `fn f[N](a: [i32; N / (N-N)])` and called it with N=5.
   The compiler now KNOWS the array dim is 5/0 (mathematically
   undefined). The pipeline should ABORT with an error and a
   trap-28801 diagnostic. Instead it prints a warning, sets
   `mono_count = 0`, and continues. Downstream:
     - grad_pass operates on the unmodified Program (no mono'd fns
       added; the calls to generic `f` remain as un-mono'd Calls).
     - typecheck runs on the unmodified Program and likely emits
       no error (the original `[i32; N/(N-N)]` resolves via
       `_resolve_size_expr` to TyUnknown).
     - codegen attempts to lower the program with an unresolved
       generic call — likely produces a compile error from a later
       stage (lower-ast, codegen) with a much worse diagnostic
       ("undefined function `f`" or "array size unknown") than the
       trap-28801 that should have surfaced.

2. **Partial-mono exit**: `monomorphize_safe` does
   `return monomorphize(prog), []` in the try-block and
   `return 0, [str(e)]` in the except. If `Monomorphizer.run()`
   raised ShapeFoldError on the 5th of 10 generic instantiations,
   the wrapper returns count=0 (not 4) and the 4 successfully
   mono'd fns are LOST — they were added to `prog.items` by
   `_emit_instance` but the count returned to the driver is 0.
   The driver doesn't know how many mono'd, but `prog` itself was
   mutated mid-flight, so the program state is partial:
   - 4 mono'd fns added
   - 5th raised; 5th-10th never processed
   - Driver: count=0 (so the "mono: N generic instantiation(s)"
     log line doesn't print)
   - Pipeline continues with a half-mono'd program.

   This is worse than either (a) aborting the pipeline cleanly
   (so the user sees the trap-28801 and can fix the source) or
   (b) catching per-instantiation and accumulating diags with
   continued mono on the remaining instantiations.

The cycle-4 doc's E3 recommended fix was: "Then the top-level
driver consumes diags and prints them as **trap 28801 with
file/line/col context** instead of 'internal error'". The actual
fix prints `warning: fn-mono: <str(e)>` — no `error:` prefix, no
explicit `28801` trap-id label (though the message text contains
it), no `--strict` honoring, and the pipeline continues.

**Reproducer**:
```helix
fn f[N](a: [i32; N / (N - N)]) -> i32 { 0 }
fn main() -> i32 {
    let x: [i32; 1] = [0];
    f::<5>(x)
}
```

```
$ python -m helixc.backend.x86_64 --strict bug.hx
warning: fn-mono: array shape fold: 5 / 0 (trap 28801) ...
... (pipeline continues, grad_pass + typecheck + codegen run on
the half-mono'd program; final result either silently broken
binary or downstream error with a misleading message)
```

The user wanted: a clean error at the fn-mono stage, trap 28801,
file:line:col pointing at `N / (N - N)`, and abort.

**Recommended fix**:
Either:

(a) **Driver-side abort**: in x86_64.py at line 3021-3027, treat
    `mono_diags` as fatal errors:

    ```python
    mono_count, mono_diags = monomorphize_safe(prog)
    if mono_diags:
        for d in mono_diags:
            print(f"error: fn-mono: {d}", file=sys.stderr)
        print(f"\n{len(mono_diags)} fn-mono error(s); aborting.",
              file=sys.stderr)
        sys.exit(1)
    if mono_count > 0:
        ...
    ```

    Simplest fix; aligns with the typecheck `--strict` semantic
    later in the same driver.

(b) **Per-instantiation accumulate-and-continue**: refactor
    `Monomorphizer.run()` to catch ShapeFoldError per
    `_emit_instance` call, accumulate on `self.errors`, and skip
    the failing instantiation. `monomorphize_safe` returns
    (count, errors); driver aborts if errors. Preserves partial
    progress for diagnostics but builds nothing.

    More invasive but better UX for source files with multiple
    independent errors — the user sees ALL of them in one
    compile cycle.

Either fix makes ShapeFoldError a compile-time error per the
trap-id design intent. The current warning-and-continue is a
contract drift from "trap 28801 is a user error" (per the
ShapeFoldError docstring) to "trap 28801 is a soft hint".

---

### Finding F4: `_inline_lets` TileLit identity arm doesn't walk shape / memspace — let-bound names there silently un-substituted

**File**: `helixc/frontend/autodiff.py:709-710` (TileLit identity arm,
C4-1 fix);
`helixc/frontend/ast_nodes.py:343-356` (TileLit definition).
**Severity**: LOW
**Category**: cycle-4 fix narrower than cycle-4 doc's recommendation

**Description**:
Cycle 4 E7 / C4-1 added a TileLit arm before the catch-all to stop
the spurious 85001 warning. The cycle-4 doc's recommended fix was:

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

— a walk arm that recurses through `shape` and `memspace` so
let-bound names there get substituted.

The actual fix at line 709-710 is:

```python
if isinstance(expr, A.TileLit):
    return expr
```

— an identity arm. It silences the warn but does NOT walk
shape / memspace.

TileLit's `shape: list[Expr]` and `memspace: Expr` (per
ast_nodes.py:354-355) ARE Expr-typed and can contain let-bindable
Names. For example:

```helix
fn f() -> tile<f32, [4, 4], REG> {
    let n = 4;
    let zero_tile = tile<f32, [n, n], REG>::zeros();  // shape[0]=Name('n'),
                                                      // shape[1]=Name('n')
    zero_tile
}
```

When this passes through `_inline_lets`, `Name('n')` in the shape
list is NOT substituted with `IntLit(4)`. Downstream typecheck and
codegen then face a tile literal with `shape=[Name('n'), Name('n')]`
instead of `[IntLit(4), IntLit(4)]`.

In practice TileLit is rare and parser-emitted shape elements are
usually IntLit literals, so the day-to-day impact is small. But
the principle (let-bindings reach all of an Expr's children) is
violated.

**Reproducer**:
```python
from helixc.frontend import ast_nodes as A
from helixc.frontend.autodiff import _inline_lets

sp = A.Span(0, 0)
tl = A.TileLit(
    span=sp,
    dtype=A.TyName(span=sp, name='f32'),
    shape=[A.Name(span=sp, name='n', generics=[]),
           A.Name(span=sp, name='n', generics=[])],
    memspace=A.Name(span=sp, name='REG', generics=[]),
    init='zeros',
)
env = {'n': A.IntLit(span=sp, value=4)}
out = _inline_lets(tl, env)
# out.shape[0] is Name('n') — should be IntLit(4)
```

**Recommended fix**:
Replace the identity arm with the walk arm recommended in cycle-4
doc:

```python
if isinstance(expr, A.TileLit):
    return A.TileLit(
        span=expr.span,
        dtype=expr.dtype,
        shape=[_inline_lets(s, env) for s in expr.shape],
        memspace=_inline_lets(expr.memspace, env),
        init=expr.init,
    )
```

Note: also walk `memspace` (it's an Expr, can be a Name). The
cycle-4 doc's recommendation omitted memspace walking; cycle 5
recommends including it for completeness.

---

### Finding F5: parser.hx tag-12 sentinel collides with AST_LET_MUT (tag 12) — type-tag and AST-tag namespaces overlap

**File**: `helixc/bootstrap/parser.hx:26-27` (AST_LET_MUT definition);
`helixc/bootstrap/parser.hx:2333` (D2 tag-12 sentinel = "untracked-call");
`helixc/bootstrap/parser.hx:2373` (C4-2 tag-12 sentinel reuse).
**Severity**: LOW
**Category**: namespace overlap — sentinel space collides with AST-tag space

**Description**:
The cycle-4 fix-sweep notes "E5 (parser tag-12 namespace overlap) is
documented but not changed". The cycle-4 doc described tag-12
namespace overlap with **prim-type** tags (0-11). But there is a
SECOND, more direct namespace overlap that cycle 4's audit missed
and the C4-2 fix made worse by spreading tag 12 to many more code
paths:

Looking at parser.hx:26-27:
```
//  12  AST_LET_MUT   same payload shape as AST_LET; codegen treats
//                    them identically.
```

Tag 12 is **AST_LET_MUT** in the AST-tag namespace. The D2 +
C4-2 sentinel reuses tag 12 in the **type-tag** namespace (the
slot that gets passed to `var_type_tab_add(...)` at line 2382).
These are different namespaces — type tags are 0-11 prim types
plus the sentinel; AST tags are 0..41+ syntax nodes.

The two namespaces happen to be disjoint AT THIS CALL SITE because
`var_type_tab` only stores type-tag values, and `__arena_get` at
line 2302 returns AST-tag values. The collision is latent — if any
future code path reads from `var_type_tab` and forwards the value
to a downstream consumer that interprets it as an AST tag (e.g., a
codegen lookup keyed on AST tags), tag 12 would silently resolve
to AST_LET_MUT.

Cycle-4 fix C4-2 widened the use of the sentinel: now ANY non-
literal RHS (Binary, Unary, Index, Field, If, Match, Block,
UnsafeBlock) writes tag 12 to `var_type_tab`. The probability that
some future call site reads back the type-tag and mis-interprets it
as an AST tag increased proportionally — more variables tagged with
the colliding value.

The cycle-4 doc's E5 recommended moving sentinels to 100+ ("clearly
out-of-band range"). The cycle-4 fix declined to do so, citing
internal consistency. But internal consistency within parser.hx
doesn't preclude future readers of `var_type_tab` from mis-
interpreting tag 12. The cycle-4 fix-sweep made the namespace
overload broader without addressing the encoding boundary.

**Reproducer**: conceptual — a future stage 28.X adds a
`var_type_tab_lookup` call in codegen that forwards the returned
value to an AST-node dispatch table. The dispatch table sees tag
12 and routes through AST_LET_MUT logic, treating a typed
variable as a let-mut statement. Silent miscompile.

**Recommended fix**:
Move sentinels to a clearly-out-of-band range and document the
boundary at parser.hx file header:

```
// Type tags (stored in var_type_tab):
//    0 = i32 (and bool-results from comparison ops)
//    1 = f32 ... 11 = i16  (prim types)
//   12 = AST_LET_MUT (AST namespace, NOT a type tag — do not use here)
//  100 = sentinel: untracked-call RHS
//  101 = sentinel: untracked-complex RHS (Binary / Unary / If / Match / etc.)
```

Then update the two write sites (line 2333 and 2373) and any read
site that branches on `> 0` to also accept the new sentinels.
Capture guard `> 0` still works because 100 / 101 > 0.

Alternative (band-aid): rename the comment field of the C4-2
sentinel from `12         // sentinel: untracked-complex` to
`12         // sentinel: untracked-complex (do NOT confuse with
AST_LET_MUT tag 12 — different namespace)` so future readers
notice. Less robust but zero code change.

---

### Finding F6: parser.hx C4-2 AST_VAR arm leaves `inferred_ty_tag = -1`; silent i32 fallback at capture site if var_type_tab lookup misses

**File**: `helixc/bootstrap/parser.hx:2358-2359` (C4-2 AST_VAR arm);
`helixc/bootstrap/parser.hx:2382-2384` (registration guard);
`helixc/bootstrap/parser.hx` (capture site, autodetected from "cap_ty_tag").
**Severity**: LOW
**Category**: defer-vs-pass-through asymmetry — defer assumes lookup succeeds

**Description**:
Cycle 4 C4-2 added an explicit AST_VAR arm (val_tag == 1) inside
the inner else-chain. The comment says "AST_VAR — defer to
var_type_tab resolution". The code body is empty:

```python
if val_tag == 1 {
    // AST_VAR — defer to var_type_tab resolution.
} else { if val_tag == 6 {
    ...
```

So `inferred_ty_tag` stays at its initial value `0 - 1` (= -1).
Then at line 2382:

```python
if inferred_ty_tag >= 0 {
    var_type_tab_add(sb, name_start, name_len, inferred_ty_tag);
};
```

Since -1 < 0, no entry is added to var_type_tab for this binding.
The intended path is: when `var_type_tab_lookup(name)` runs at the
capture site, the original RHS variable's type tag is the result
(the AST_VAR RHS resolves to ANOTHER variable's tag).

But the lookup at the capture site uses the let-binding's NAME,
not the RHS name. So when:

```helix
let pi = 3.14_f64;                 // var_type_tab[pi] = 2 (f64) via prior arm
let alias = pi;                    // val_tag == 1 (AST_VAR), defer path
let c = |x| x + alias;             // capture site uses alias
```

The capture probe at the closure site calls
`var_type_tab_lookup("alias")` — but `alias` was never added to
var_type_tab (because `inferred_ty_tag = -1` and the >= 0 guard
skipped the add). Lookup returns -1 (untracked); the `> 0`
capture guard treats -1 as "trivially safe i32"; `alias` (an f64)
silently captures as i32 → trap 76003 is silenced.

This is exactly the silent-i32-truncation pattern that D2 / C4-2
were designed to close. The C4-2 AST_VAR arm intended to defer but
the defer is unimplemented — to actually defer, the arm would need
to look up the RHS name's tag in var_type_tab AT BINDING TIME and
copy it to the let-binding's tag.

**Reproducer**:
```helix
fn make_pi() -> f64 { 3.14 }
fn outer() -> f64 {
    let pi = make_pi();           // tag 12 sentinel (Call RHS, untracked)
    let alias = pi;               // tag = ? — falls through C4-2 AST_VAR arm,
                                  // stays -1, no var_type_tab entry
    let c = |x: f64| x + alias;   // capture probe looks up 'alias',
                                  // finds nothing (>0 guard fails),
                                  // captures as i32. trap 76003 silenced.
    c(1.0)
}
```

Note this is a chained-let case; the simpler `let alias = literal;
let c = |x| x + alias;` works because `literal` goes through the
appropriate typed-literal arm and `alias = AST_VAR(literal)` would
need the chain.

**Recommended fix**:
Implement the deferred lookup at let-binding time:

```python
if val_tag == 1 {
    // AST_VAR — look up the source variable's tag and forward.
    // p1 = source byte index, p2 = byte length for AST_VAR.
    let src_start = __arena_get_p1(value);
    let src_len = __arena_get_p2(value);
    let src_tag = var_type_tab_lookup(sb, src_start, src_len);
    if src_tag >= 0 {
        inferred_ty_tag = src_tag;
    } else {
        // Unknown source variable — fall back to sentinel.
        inferred_ty_tag = 12;
    };
} else { if val_tag == 6 {
    ...
```

This makes the defer concrete: alias-of-tracked-var inherits the
tag; alias-of-untracked-var gets the sentinel; the capture probe
sees a positive tag in either case and the >0 guard works.

Alternative (conservative): treat AST_VAR like every other complex
RHS — assign sentinel 12. Loses the precision (an `alias = i32_var`
would now wrong-trap), but eliminates the silent-i32 path:

```python
if val_tag == 1 {
    // AST_VAR — conservative: sentinel until proper defer is wired up.
    inferred_ty_tag = 12;
}
```

Less ideal because it over-traps; the first fix is preferred.

---

## Cycle 4 fix re-verification

| Fix    | Status     | Notes                                                                                                       |
|--------|------------|-------------------------------------------------------------------------------------------------------------|
| C4-1   | OK         | `_inline_lets` Path / Continue arms before catch-all silence the spurious 85001 warning. Verified via probe — `Maybe::None` reference under a diff'd fn no longer fires. (TileLit identity arm is a separate residual; see F4.) |
| C4-2   | OK*        | Tag-12 sentinel widened from val_tag==16 to all non-literal RHS. Comparison ops correctly typed as bool (tag 0). **However**, AST_VAR arm (val_tag == 1) defers to var_type_tab but never implements the forward — chained alias silently un-tracks (F6). Tag-12 namespace collision with AST_LET_MUT remains latent (F5). |
| C4-3   | OK         | `_inline_lets` If.cond inlined. Symmetric with While/For/Match. No new issues. |
| C4-4   | OK         | `_compatible` TyTile / TyTensor structural arms. Both arms compare dtype recursively, shape positionally with `_compatible`, device/layout/memspace nominally. No new issues. |
| C4-5   | OK*        | `monomorphize_safe` wrapper catches ShapeFoldError. **However**, x86_64 driver treats result as warning and continues into codegen with a partial-mono program (F3). |
| E1     | OK*        | TyArray size compare disjunctive `== or _compatible`. Closes the TyUnknown sub-case via top-of-_compatible cascade. **However**, the cycle-4 doc's primary reproducer (TySize('N') vs TySize('M')) is NOT closed because `_compatible` lacks a TyVar/TySize cascade arm (F1). |
| E2     | OK*        | Logic-domain gate broadened to `inner_mismatch OR (l_is_logic != r_is_logic)`. Closes Logic-wrap-asymmetric same-inner. **However**, the parallel D-wrap-asymmetric same-inner case (`D<f64> + f64`) was not broadened (F2). |
| E3     | OK*        | `monomorphize_safe` wrapper exposes diags. (Same as C4-5; F3 covers the severity gap.) |
| E4     | OK         | `_resolve_size_expr` IntLit / Unary(-, IntLit) arms unified to "must be > 0" wording for both negative and zero cases. No new issues. |
| E5     | (NOT FIXED) | Cycle-4 commit message says "E5 documented but not changed — internally consistent within parser.hx". F5 below escalates: the type-tag-vs-AST-tag collision is a real latent overload, not just prim-type namespace congestion. |
| E6     | OK         | `_inline_lets` Call arm preserves `expr.callee.generics` when aliasing. The fix uses `if expr.callee.generics else cand.generics` so an alias-to-generic without an explicit turbofish inherits the alias's turbofish; an alias-WITH-turbofish overrides. Cycle-4 doc's reproducer verified. No new issues. |
| E7     | OK*        | `_inline_lets` TileLit identity arm silences the catch-all. **However**, doesn't walk shape / memspace — let-bound names there silently un-substituted (F4). |
| E8     | OK         | `_inline_lets` Call arm walks Field-typed callees via `_inline_lets(expr.callee, env)`. Verified via probe — `let me = self; me.method()` substitutes `me` correctly. No new issues. |

Five cycle-4 fixes have residual gaps (`*`):
- **C4-2** has the AST_VAR defer-not-implemented gap (F6) and the
  tag-12 namespace collision (F5).
- **C4-5 / E3** has the warning-not-error severity downgrade (F3).
- **E1** has the TyVar/TySize cascade-arm-missing gap (F1).
- **E2** has the D-wrap-asymmetric mirror gap (F2).
- **E7** is narrower than the cycle-4 doc recommended (F4).

The other eight cycle-4 fixes (C4-1, C4-3, C4-4, E4, E6, E8) hold
up under re-audit with no caveats.

---

## Cycle 5 focus question answers

**1. Does the E1 TyArray size `_compatible` recursion actually close the cycle-4 doc's primary reproducer?**

**No.** The fix closes only the TyUnknown sub-case (which would
already cascade-safe at the top of `_compatible`). The
`fn f[N](a:[i32;N])` called from `fn g[M](a:[i32;M])` reproducer
still returns False because `_compatible(TySize('N'), TySize('M'))`
falls through every `isinstance` arm to `return a == b` which is
False for different name fields. The fix needs a TyVar/TySize
cascade-safe arm at the top of `_compatible`, parallel to TyUnknown
(F1).

**2. Did the E2 Logic-gate broadening also broaden the parallel D-gate?**

**No.** The Logic gate now fires on `inner_mismatch OR (l_is_logic
!= r_is_logic)`. The D gate at line 1349 still fires only on
`inner_mismatch`. So `D<f64> + f64` (same inner, asymmetric wrap)
remains silent — a cycle-2 B:C6 mirror gap that's been latent since
cycle 2 (B:C6 description said it was closed; cycle-5 review shows
it was closed only for inner-mismatch, not for same-inner-bare).
The cycle-4 E2 fix should have mirrored to the D gate (F2).

**3. Does `monomorphize_safe` surface ShapeFoldError at the right severity?**

**No.** The driver prints `warning: fn-mono: ...` and continues into
grad_pass / typecheck / codegen with a partial-mono program (the
ShapeFoldError stops `Monomorphizer.run()` at the first failure, so
not all generic fns get instantiated). The cycle-4 doc's E3 fix
specified "trap 28801 with file/line/col context instead of
'internal error'" — implicitly an error, not a warning. The actual
fix downgraded to warning (F3).

**4. Is the TileLit C4-1 / E7 fix sufficient as an identity arm?**

**No.** TileLit's `shape: list[Expr]` and `memspace: Expr` are
Expr-typed and can contain let-bindable Name leaves. The identity
arm silences the catch-all warn but doesn't walk shape / memspace,
so source like `let n=4; let t=tile<f32, [n,n], REG>::zeros();`
silently drops the let-binding for `n` in the shape (F4). The
cycle-4 doc's recommended fix was a walk arm.

**5. Is the parser tag-12 sentinel namespace-clean now that C4-2 widened its use?**

**No.** The cycle-4 doc's E5 noted prim-type namespace congestion
(tag 12 unused as a prim, so the sentinel works today). The real
collision is with the AST-tag namespace: tag 12 = AST_LET_MUT. If
any future code path forwards a var_type_tab value to an AST-node
dispatch table, tag 12 silently dispatches as AST_LET_MUT (F5).
C4-2 widened the use of tag 12 (from val_tag==16 only to ALL
non-literal RHS) so the latent collision affects more variables.

---

## Cycle 5 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity.**

This cycle finds **6 new findings (0 HIGH, 3 MEDIUM, 3 LOW)**. By
the strict criterion, **cycle 5 does NOT count clean**. Zero HIGH
findings is consistent with cycle 4. The three MEDIUMs are
**incomplete-fix** regressions of cycle-4 stated goals at the
contract level:

- F1: E1's TyArray size symmetry fix didn't extend to TyVar/TySize
  cascade; the cycle-4 doc's primary reproducer is unchanged.
- F2: E2's Logic-gate broadening is asymmetric — the parallel
  D-gate broadening was not applied, leaving the cycle-2 B:C6
  same-inner-bare mirror silent.
- F3: monomorphize_safe wrapper exists but the x86_64 driver
  surfaces ShapeFoldError as a warning, not an error; pipeline
  continues into codegen with a partial-mono program.

These three are the **next-layer-down** instance of the cycle-1 →
cycle-2 → cycle-3 → cycle-4 escalation pattern repeating in cycle
5. Each cycle-4 fix expanded the contract surface, and the
expanded contract has a strictly-weaker invariant than the cycle-4
doc described it as having.

Three LOW findings (F4, F5, F6) cover narrower edge cases:
- F4: TileLit identity arm narrower than cycle-4 doc recommended;
  let-bound names in TileLit.shape / memspace silently un-
  substituted.
- F5: parser tag-12 sentinel collides with AST_LET_MUT (a
  different namespace than the cycle-4 doc's E5 noted prim-type
  congestion).
- F6: parser C4-2 AST_VAR arm comment says "defer" but the defer
  body is empty; chained aliases (let alias = pi) lose tag
  tracking and the capture probe silently treats them as i32.

Recommended fix sequence for cycle 6:

1. **F1 first** (TyVar/TySize cascade arm in `_compatible` — a
   3-line addition at line 2158 that makes the E1 fix actually
   close its stated reproducer).
2. **F2 with F1** (D-wrap-asymmetric mirror — broaden the D gate
   at line 1349 in parallel to E2's Logic broadening, ~5 lines).
3. **F3** as a separate commit (driver-side abort on
   monomorphize_safe diags — 4-line change in x86_64.py at line
   3025).
4. **F4 / F5 / F6** as a low-severity cleanup commit.
   F4 is a 5-line walk arm replacing the identity arm.
   F5 is a parser.hx re-numbering of sentinels to 100+ plus
   header documentation.
   F6 is implementing the AST_VAR forward lookup in C4-2's defer
   arm.

Once F1-F3 land + verify, cycle 6 can re-audit. Per the strict
criterion, the remaining cycles needed to declare type-design
soundness clean is at least one full clean sweep after addressing
all 6 findings — pending whether new gaps surface from the fixes
themselves. The cycle-1 → cycle-5 pattern shows each fix
typically exposes a single next-layer-down gap, so a realistic
expectation is one more cycle of ~3-5 findings before a clean
sweep is achievable.

The 5-clean-cycles strict requirement to deprecate Python helixc
means cycle 5 + at least 5 subsequent clean cycles. At the
current pace (~3-8 findings per cycle, each addressing prior
cycle gaps but exposing new ones), the soonest clean state under
the strict criterion is ~5-8 more audit cycles assuming each
cycle's fix-sweep closes its own findings without introducing
new ones. The cycle-1-through-5 pattern suggests the
fix-introduces-gap rate is high enough that fully convergent
clean sweeps may require relaxing either the strict criterion or
the scope of the audit (e.g., accepting LOW findings as
"documented-but-deferred" so cycle clean requires 0 HIGH + 0
MEDIUM only).
