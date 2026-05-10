# Stage 28.8 Pre-29 Audit Gate — Cycle 1, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-10
**Commit**: fc96595 (read-only)
**Scope**: All helixc source. Focus on type-system soundness across stages 22-28.7
(new files never audited) AND interactions between stages.
**Method**: traced each new TyNode through resolution, binop propagation,
monomorphization (both fn and struct), unsafe-context threading, and the
AD passes. Verified that the compile pipeline (`backend/x86_64.py`,
`check.py`) actually invokes the new passes — several do not. Cross-checked
cycle-1 audits A (silent-failure) and C (codereview) findings to avoid
re-flagging.

**Result**: 14 findings (5 HIGH, 7 MEDIUM, 2 LOW). The single highest-severity
issue is that the Stage 28 parametric-struct monomorphization pass
(`struct_mono.monomorphize_structs`) is **defined but never invoked** by the
compile pipeline. Combined with the typechecker collapsing all `TyGeneric`
of user structs to `TyUnknown`, `Pt<i32>` and `Pt<f32>` are silently
unifiable at every use site and never produce distinct codegen — the type
system does not preserve parametric-struct identity at all.

Other major findings cluster around (a) Stage 24 TyLogic provenance
being silently dropped through every binop and call site (only TyDiff
propagates); (b) Stage 28.6 `unsafe` providing no actual capability
guard — integer→raw-pointer casts are accepted *outside* unsafe and
the unsafe-pass walker doesn't see into casts at all; (c) Stage 9
closure captures hardcoded to type-tag i32 in the bootstrap regardless
of the captured var's real type; (d) AD passes silently return 0
gradient for any unknown expression, including `Quote`/`Splice`/
`UnsafeBlock`; (e) `substitute_ty` missing `TyPtr` arm, plus
`_walk_subst_expr` missing `A.UnsafeBlock` recursion, so generics
through pointers and unsafe blocks don't substitute.

Cycle 1 is NOT clean. Stop-the-line on Finding B1 (parametric-struct
pipeline gap) and B2 (TyLogic dropped through binops).

---

## Finding B1: parametric struct monomorphization never runs

**File**: `helixc/frontend/struct_mono.py` (the entire module);
`helixc/check.py` (no call site); `helixc/backend/x86_64.py:2911-2945`
(only invokes `monomorphize` for fns, not `monomorphize_structs`).
**Severity**: HIGH (CRITICAL for stage-28's stated guarantees)
**Category**: type soundness / dispatch correctness

**Description**:
`struct_mono.monomorphize_structs(prog)` is defined and unit-tested in
`helixc/tests/test_struct_mono.py` but **never imported by the
production pipeline**:

1. `helixc/check.py` — no import of `struct_mono`.
2. `helixc/backend/x86_64.py.__main__` (lines 2906-2948) — imports
   `monomorphize` (fn mono) but NOT `monomorphize_structs`.

Combined with `typecheck.py:417-418`:
```python
if isinstance(ty, A.TyGeneric):
    ...
    # User type with generic args — v0.1 unknown
    return TyUnknown(hint=f"generic {ty.base}")
```

every `Pt<i32>`, `Pt<f32>`, `Tree<Edge>` etc. at a binding-type position
becomes `TyUnknown`. `_compatible(TyUnknown, _) == True` (line 1413-1414).
So:

- `let p1: Pt<i32> = Pt { x: 1, y: 2 };` — RHS is `TyStruct("Pt")`
  (line 1121), LHS is `TyUnknown`. Compatible. Bound as Pt.
- `let p2: Pt<f32> = p1;` — RHS is `TyStruct("Pt")` (the same), LHS
  `TyUnknown`. Compatible. Silent type-pun of an i32-shaped Pt as an
  f32-shaped Pt.
- `fn dist(p: Pt<i32>) -> i32`, called as `dist(make_pt_f32())` —
  silently accepted; runtime reads f32 bit pattern as i32.

The unit tests in `test_struct_mono.py` cover the algorithm in
isolation; they do NOT verify integration with the compile pipeline
(parser → typecheck → codegen).

**Reproducer**:
```helix
struct Pt[T] { x: T, y: T }
fn dist_i32(p: Pt<i32>) -> i32 { p.x + p.y }
fn dist_f32(p: Pt<f32>) -> f32 { p.x + p.y }
fn main() -> i32 {
    let pi: Pt<i32> = Pt { x: 1, y: 2 };
    let pf: Pt<f32> = pi;     // accepted; pi is f32-shaped now
    dist_f32(pf) as i32       // reads i32 bits as f32 → silent garbage
}
```

**Recommended fix**:
Two-step:
1. Wire `from .frontend.struct_mono import monomorphize_structs` into
   `backend/x86_64.py.__main__` and `check.py.main()`, invoked AFTER
   `flatten_impls` and BEFORE `monomorphize` (fn mono picks up the
   instantiated struct names).
2. Extend `typecheck._resolve_type` to look up `ty.base` in
   `self._struct_decls` and, if found, return a `TyStruct(mangled_name)`
   where `mangled_name = struct_mono.mangle_struct(ty.base, ty.args)`.
   Then `Pt<i32>` and `Pt<f32>` become `TyStruct("Pt__i32")` and
   `TyStruct("Pt__f32")` — distinct under `_compatible`.

Without (2), even after wiring (1), the typechecker still can't tell
the instantiations apart. Both fixes are needed.

**Trap-id reservation**:
28003 — `TyGeneric` on a known generic struct resolved to `TyUnknown`
(the lookup gap). 28004 — `Pt<i32>` and `Pt<f32>` unified at compile
time (post-fix-2 regression guard).

---

## Finding B2: TyLogic provenance silently dropped through binary ops

**File**: `helixc/frontend/typecheck.py:864-878`
**Severity**: HIGH
**Category**: type soundness (Stage 24)

**Description**:
The binop handler propagates `TyDiff` (lines 872-876) but has NO arm
for `TyLogic`. For `Logic<bool> + Logic<bool>`, neither operand is
`TyDiff`, so control falls to line 877-878: `return l` — the left
operand's type. This produces the right answer ONLY if the left is
the Logic-wrapped one.

Mixed cases:
- `Logic<T> + T` → left is `TyLogic`, returns `TyLogic` (correct
  by accident).
- `T + Logic<T>` → left is `T`, returns `T` — Logic wrapper silently
  stripped from result. Provenance lost.
- `Logic<T> + D<T>` → l_is_diff=False, r_is_diff=True, returns
  `TyDiff(inner=r.inner=T)` — **Logic wrapper on left silently
  stripped** in favor of D wrapper on right. The intended semantics
  per the docstring (line 132-148) is "Composing D<Logic<T>>
  represents a differentiable relational value" — so the result
  should be `TyDiff(TyLogic(T))`, not `TyDiff(T)`.
- `D<Logic<T>> + Logic<T>` → l_is_diff=True, inner=l.inner=Logic<T>,
  returns `TyDiff(TyLogic(T))` (correct by accident).

The provenance string field on `TyLogic` (line 154) is never read,
never propagated, never compared in `_compatible`. The "Trap 24001
emitted if a non-Logic value is passed where a Logic-typed parameter
is required" promise (line 145-147) is also unimplemented — no call
to emit 24001 anywhere in the codebase.

**Reproducer**:
```helix
fn classify(x: Logic<bool>, y: D<bool>) -> D<Logic<bool>> {
    x + y     // typecheck infers TyDiff(bool) — Logic wrapper dropped.
              // Result type advertised as D<Logic<bool>>; compiler
              // assigns D<bool>. No error surfaced.
}
```

**Recommended fix**:
Add the Logic-propagation arm before falling to the arithmetic
fallback:
```python
l_is_logic = isinstance(l, TyLogic)
r_is_logic = isinstance(r, TyLogic)
if l_is_logic or r_is_logic:
    inner = l.inner if l_is_logic else r.inner
    # If either side is also D<T>, the wrapping is D<Logic<T>>.
    if isinstance(inner, TyDiff) or l_is_diff or r_is_diff:
        return TyDiff(inner=TyLogic(inner=...))
    return TyLogic(inner=inner)
```

Then extend `_compatible` to require matching Logic-wrapping at
provenance-sensitive boundaries.

**Trap-id reservation**:
24001 — already reserved per docstring; wire it in at the call-site
boundary check.

---

## Finding B3: Cast (`expr as T`) accepts arbitrary integer→pointer with no unsafe gate

**File**: `helixc/frontend/typecheck.py:1124-1126`;
`helixc/frontend/unsafe_pass.py:100-109` (only flags Unary deref)
**Severity**: HIGH
**Category**: capability soundness (Stage 28.6)

**Description**:
The Cast handler is three lines:
```python
if isinstance(expr, A.Cast):
    self._check_expr(expr.value, scope)
    return self._resolve_type(expr.target_ty, scope)
```

No validation. Source type and target type can be anything. Specifically:
- `0_i64 as *mut i32` — silently accepted at top level (no `unsafe`
  required).
- `0xDEADBEEF_u64 as *const u8` — silently accepted.
- `(some_f64) as *mut Pt` — silently accepted.

The unsafe-pass walker (`unsafe_pass._is_raw_ptr_op`) only matches
`Unary('*', x)` — a syntactic deref. It does NOT match Cast nodes.
So the program `let p: *mut i32 = 0_i64 as *mut i32; ` outside any
unsafe block:
1. Cast typechecks (no error).
2. Unsafe-pass doesn't find a raw-ptr op (no Unary `*`).
3. Bind succeeds.
4. Later `*p` outside unsafe WOULD trap 28601, but at THIS point the
   pointer is already a non-pointer integer in disguise.

This violates the explicit audit-prompt question: "does the type
checker correctly NOT allow `unsafe` to coerce arbitrary i64 to
`*mut T`?" The answer is no — and even without `unsafe`, the cast is
allowed.

Inside `unsafe`, the cast is also allowed (no different behavior).
There's no extra-strict mode.

**Reproducer**:
```helix
fn main() -> i32 {
    let bad: *mut i32 = 0x1000_i64 as *mut i32;   // accepted, no unsafe
    // Now bad is a 'pointer' to address 0x1000 — typed.
    // unsafe { *bad } would deref a random heap address.
    0
}
```

**Recommended fix**:
At Cast typechecking:
1. Look up source type via `_check_expr(expr.value)`.
2. If target is a `TyPtr` and source is NOT a `TyPtr` or `TyRef`, require
   that we're inside an `UnsafeBlock` (thread context like
   `unsafe_pass._walk` does). If not, emit error 28603 (new trap):
   "raw-pointer cast outside unsafe block".
3. Also block float→ptr unconditionally (even inside unsafe is
   dubious).

Additionally, `unsafe_pass._is_raw_ptr_op` should match
`isinstance(node, A.Cast) and isinstance(target_ty, A.TyPtr)` so
out-of-unsafe ptr-casts surface diagnostically.

**Trap-id reservation**:
28603 — raw-pointer cast outside unsafe block.

---

## Finding B4: Stage 9 closure captures hardcoded to i32 — all non-i32 captures silently truncated/garbled

**File**: `helixc/bootstrap/parser.hx:1790-1820` (capture + param push)
**Severity**: HIGH
**Category**: type soundness (Stage 9)

**Description**:
The bootstrap closure-lowering builds `AST_PARAM` records for each
capture and each closure param, and **hardcodes** the type tag = 0
(i32) on lines 1795 and 1811:
```
let new_p = mk_node(18, ns, nl, 0);
__arena_push(0);                     // type tag = 0 (i32)
```

The comment at line 1705 acknowledges "all closure params default to
i32". But this is silent corruption for any non-i32 capture:

- `let s = "hello"; let c = |x| s.len() + x;`
  `s` is a string pointer (8 bytes). Capture-as-i32 reads only the low
  4 bytes into the closure env. The synthesized fn receives a 4-byte
  truncated pointer → SEGV on `s.len()`.
- `let pi = 3.14_f64; let c = |x| x + pi;`
  `pi` is f64 (8 bytes). Capture-as-i32 reads low 32 bits of the f64
  bit pattern → silent garbage arithmetic.
- `let m: Maybe = Maybe::Some(42); let c = |x| __enum_discriminant(m);`
  `m` is a pointer to the enum's heap rep. Truncated to 4 bytes → SEGV
  on discriminant read.

Also the `cl_capture_tab` entry stride is 2 (`name_s`, `name_l`)
per line 605-608 — there is no slot for the captured var's type tag.
Even if Stage 9 wanted to propagate the type, the table layout
discards it.

**Reproducer**:
```helix
fn main() -> i32 {
    let pi: f64 = 3.14_f64;
    let c = |x: i32| {
        // pi captured. Bootstrap synthesizes:
        //   fn __closure_0(pi: i32, x: i32) -> i32 { (pi as i32) + x }
        // The caller passes pi's low 32 bits as 'pi' arg.
        (pi as i32) + x
    };
    c(1)
}
```
The user expected pi=3.14 captured as f64; bootstrap captures pi as
i32 (low 32 bits of the IEEE-754 bit pattern).

**Recommended fix**:
Extend `cl_capture_tab` stride from 2 to 3 (add `ty_tag` slot).
Capture-site resolution (`mk_var_with_capture`) should also read
the captured var's type tag via `var_type_tab_lookup` (or
`bind_lookup_type` if accessible) and store it. Then the AST_PARAM
push at line 1795 uses the recorded tag, not 0.

For Phase-0 minimum: trap 76003 if a non-i32 capture is detected
(loud rather than silent).

**Trap-id reservation**:
76003 — closure capture has non-i32 type (Phase-0 limitation made
loud).

---

## Finding B5: AD passes silently return 0 gradient for Quote/Splice/UnsafeBlock/Cast-on-pointer

**File**: `helixc/frontend/autodiff.py:520-521`;
`helixc/frontend/autodiff_reverse.py` (no Quote handling either)
**Severity**: HIGH
**Category**: AD soundness (Cross-stage AD + reflection / unsafe)

**Description**:
Forward-mode `_diff` (line 465) dispatches on expression type. Any
node not matched (Quote, Splice, Modify, UnsafeBlock, Cast where the
target_ty is non-arithmetic) falls through to line 520-521:
```python
# Unsupported: emit zero (placeholder)
return A.FloatLit(span=expr.span, value=0.0)
```

A `grad(loss)` whose body contains:
- `splice(quoted_expr)` — derivative returns 0 silently.
- `unsafe { *raw_ptr }` — derivative returns 0 silently.
- `(x_f32 as f64)` — Cast is matched, but only via `_walk_subst_expr`;
  in `_diff` itself there's no Cast arm. Returns 0.

Symbolic gradient is therefore wrong without diagnostic. The user
sees `grad(f)(x) = 0` when the actual derivative is nonzero, and there
is no warning.

Same issue in `autodiff_reverse.py` for reverse-mode — no Quote /
Splice / Modify / UnsafeBlock arms.

**Reproducer**:
```helix
fn loss(x: f64) -> f64 {
    let y = unsafe { x * 2.0 };   // unsafe block (no real raw-ptr op)
    y * y
}
fn main() -> f64 {
    grad(loss)(3.0)   // expected 4*3 = 12; actual 0 (unsafe body
                       //  returned 0 derivative; outer * carries 0).
}
```
The unsafe block is a no-op semantically but breaks AD silently.

**Recommended fix**:
For each unmatched node type, emit a Helix-level diagnostic via
`prog._diff_warnings` (analogous to deprecated_pass): "node of type
{TypeName} is not differentiable; gradient assumed 0 at {span}".
Promotable to error via `-Wad=error`.

Alternatively (stricter): raise `ValueError` from `_diff` on
unhandled type, surfaced as a typecheck error before codegen.

**Trap-id reservation**:
85001 — AD assumed 0 derivative for unhandled node type. Warning
(default) or error (with `-Wad=error`).

---

## Finding B6: substitute_ty missing TyPtr arm — generic over raw pointers broken

**File**: `helixc/frontend/monomorphize.py:63-88`
**Severity**: MEDIUM (HIGH if Phase-0 ships any FFI generic)
**Category**: monomorphization soundness (Stage 28 + Stage 16.5)

**Description**:
`substitute_ty` walks TyName, TyTuple, TyArray, TyRef, TyFn, TyTensor,
TyTile, TyGeneric. The catchall `return t` (line 88) leaves TyPtr
unchanged. So `fn deref<T>(p: *mut T) -> T { ... }` mono'd with T=i32:
- new param is `*mut T` (NOT substituted) — TyName("T") stays inside.
- The cloned fn body's `*p` reads as TyPtr(TyName("T")). Downstream
  IR lowering decides element width from the inner type; TyName("T")
  isn't a known scalar → defaults silently to i32 (parser.hx:1027-1051
  documented behavior) regardless of the real T.

**Reproducer**:
```helix
fn deref<T>(p: *const T) -> T { unsafe { *p } }
fn main() -> f64 {
    let x: f64 = 3.14_f64;
    let p: *const f64 = &x;
    deref::<f64>(p)         // mono'd deref__f64 param is *const T (not f64)
                            // body reads as i32, returns low 32 bits of 3.14
}
```

**Recommended fix**:
Add a TyPtr arm to `substitute_ty`:
```python
if isinstance(t, A.TyPtr):
    return A.TyPtr(span=t.span, inner=substitute_ty(t.inner, subst),
                   is_mut=t.is_mut)
```
Same goes for `find_uninstantiated`'s `_ty_key` (struct_mono.py:123)
which also doesn't have a TyPtr arm — would silently key as `("?",
"TyPtr")` and collapse all `*T` to one entry.

**Trap-id reservation**: N/A (silent mono gap).

---

## Finding B7: monomorphize._instantiate shares where_clauses (and is_extern/attrs) with template

**File**: `helixc/frontend/monomorphize.py:440-450`
**Severity**: MEDIUM
**Category**: monomorphization correctness (Stage 28 + Stage 8.5)

**Description**:
The instantiated fn is built with `where_clauses=fn.where_clauses`
(line 446) — same Python list object as the template. Where clauses
may contain type-bearing expressions (`where T: Eq`, `where size:
N % 8 == 0`) that need substitution. Sharing the list means downstream
passes see the template's unsubstituted clauses for the mono'd clone.

Same issue with `attrs=list(fn.attrs)` — list copy is shallow, but
the attrs are strings so they survive. However if `is_extern` /
`extern_abi` carry meaning (line 458-459 of ast_nodes.py), neither is
propagated to the clone — the clone is implicitly non-extern.

Practically: a `extern "C" fn malloc<T>(n: usize) -> *mut T` (if such
generics on extern were allowed) mono'd to `malloc__i32` loses
extern-ness, gets emitted as a normal user-fn with empty body → ud2
trap at runtime.

**Recommended fix**:
1. Deep-copy `where_clauses` with `substitute_ty` applied per clause's
   constraint expression (walk Expr, replace TyName uses).
2. Propagate `is_extern` and `extern_abi`.
3. Optionally: refuse to mono generic extern decls (trap at parse time).

**Trap-id reservation**: N/A.

---

## Finding B8: TyTile shape/memspace never validated at call sites; size params don't substitute

**File**: `helixc/frontend/typecheck.py:527-565` (`_check_call_shapes`);
`helixc/frontend/monomorphize.py:82-84` (TyTile substitution)
**Severity**: MEDIUM
**Category**: shape soundness (Stage 16 + Stage 28)

**Description**:
`_check_call_shapes` is implemented only for `TyTensor`. The TyTile
branch is missing. So `fn k(t: Tile<f32, [16, 16], smem>)` called with
`Tile<f32, [32, 32], smem>` is silently accepted; the kernel may
overrun a 16x16 buffer. Same for memspace: `Tile<f32, [16,16], smem>`
vs `Tile<f32, [16,16], hbm>` — different physical address space, no
diagnostic.

Additionally, `substitute_ty` for `TyTile` only substitutes `t.dtype`
(line 83-84). `t.shape` and `t.memspace` are shared. So
`fn k<N: size>(t: Tile<f32, [N], HBM>)` mono'd as `k::<128>(...)`
produces a clone with `shape=[N]` (NOT `[128]`) — the unsubstituted
shape expression.

`ir/lower_ast.py:488-490` then sees the shape and tries
`shape[0].value if isinstance(shape[0], A.IntLit) else 0`. Since
`shape[0]` is `A.Name("N")`, not IntLit, **length silently defaults
to 0**. The PTX backend bound-checks against length 0 — every access
trips.

**Reproducer**:
```helix
@kernel
fn matvec<N: size>(a: Tile<f32, [N], HBM>, x: Tile<f32, [N], HBM>) -> f32 {
    a[0] * x[0]
}
fn main() -> f32 { matvec::<128>(load_a(), load_x()) }
// mono'd matvec__128 has shape=[N] still. length=0. a[0] traps OOB.
```

**Recommended fix**:
1. Extend `_check_call_shapes` to handle TyTile (mirror TyTensor
   branch with memspace equality check).
2. Substitute size-kind generic params in TyTile.shape / TyTensor.shape
   during `substitute_ty`. Use an AST walker on each shape expression,
   replacing `A.Name(generic)` with the concrete IntLit from subst
   when the generic kind is "size".

**Trap-id reservation**:
16003 — generic-size in tile/tensor shape not substituted at mono time.

---

## Finding B9: pytree flatten/unflatten asymmetry + cyclic-struct unbounded recursion

**File**: `helixc/frontend/pytree.py:99-122` (flatten),
`126-142` (pytree_depth, no cycle check), `162-174` (_unflatten).
**Severity**: MEDIUM
**Category**: type soundness / Phase-0 invariant (Stage 26)

**Description**:
1. **Flatten rejects non-leaf-non-struct fields** (line 117-122 raises
   ValueError 26002). **Unflatten silently sets non-diff fields to
   None** (line 173: `out[f.name] = None`). A round-trip `unflatten ∘
   flatten` is therefore undefined for any struct with non-pytree
   fields — flatten raises, unflatten silently degrades. Users testing
   `unflatten(flatten(x)) == x` (a natural test pattern) get inconsistent
   results depending on whether they catch the flatten exception.

2. **Cyclic struct refs cause unbounded Python recursion** in both
   `pytree_depth` (line 126-142) and `_unflatten` (line 162-174).
   Only `flatten_pytree` has the explicit `depth > MAX_DEPTH` guard
   (line 90-93). For `struct A { b: B } struct B { a: A }`,
   `pytree_depth(A)` recurses without bound; users see a Python
   `RecursionError` traceback instead of a clean compile-time
   diagnostic.

3. **`is_pytree_leaf` returns True for `f32 / f64 / bf16 / f16` whether
   or not the field is wrapped in `D<>`** (line 54-60). The `is_diff`
   flag tracks this, but downstream passes that key on `is_pytree_leaf
   == True` to decide AD-relevance see both wrapped and unwrapped
   floats as candidates. Mixed-type structs `{ w: D<f64>, b: f64 }`
   would both get gradients allocated, but only `w` propagates AD
   through binops. Silent inconsistency.

**Reproducer (asymmetry)**:
```python
from helixc.frontend import pytree
import helixc.frontend.ast_nodes as A
# Model with mixed pytree-leaf + bookkeeping field
model = A.StructDecl(...fields=[..f64, ..String])
pytree.flatten_pytree(model, {...})   # raises ValueError 26002
pytree.unflatten_pytree(model, {...}, {})   # returns {bookkeeping: None}
```

**Reproducer (cycle)**:
```helix
struct A { ptr_b: B, w: f64 }
struct B { ptr_a: A, b: f64 }
// pytree.pytree_depth(A_decl, {"A": A_decl, "B": B_decl})  → RecursionError
```

**Recommended fix**:
- Add visited-set guard to `pytree_depth` and `_unflatten`.
- Make `_unflatten` raise on non-leaf-non-struct fields (mirror
  flatten). Or alternately, relax flatten to also tolerate them (with
  a warning).
- Separate `is_pytree_leaf` and `is_diff_leaf` semantically; require
  passes that consume the result to declare which one they need.

**Trap-id reservation**:
26003 — cyclic struct ref reached MAX_DEPTH (currently silent
RecursionError).
26004 — pytree-leaf type but not D-wrapped (warning, optional).

---

## Finding B10: typecheck has no Quote/Splice/Modify/UnsafeBlock arms — they silently become TyUnknown

**File**: `helixc/frontend/typecheck.py:1127`
**Severity**: MEDIUM
**Category**: type soundness (Stage 11 + Stage 28.6)

**Description**:
The expression-type dispatcher ends with line 1127:
```python
return TyUnknown(hint=f"unhandled {type(expr).__name__}")
```

Five AST node types are silently typed as `TyUnknown`:
- `A.Quote` (Stage 11)
- `A.Splice` (Stage 11)
- `A.Modify` (Stage 11)
- `A.UnsafeBlock` (Stage 28.6)
- `A.For` / `A.While` / `A.Loop` — partially handled; While/Loop check
  body but return TyUnit. For's `iter_expr` and `body` are handled
  elsewhere.

`TyUnknown` is compatible with everything (line 1413-1414). So:
- `let x: i32 = unsafe { return_f64() };` — `unsafe` block returns
  TyUnknown, LHS is i32, compatible. RHS is actually f64. Silent
  truncation at codegen.
- `let y: i32 = splice(quote(some_expr));` — Quote/Splice both
  TyUnknown. Bound as i32 regardless of inner type.

Combined with Finding B5 (AD returns 0 for Quote/Splice), the typecheck
gap is masked — but the silent type-pun remains.

**Reproducer**:
```helix
fn maybe_pointer() -> *mut i32 { ... }
fn main() -> i32 {
    let x: i32 = unsafe { maybe_pointer() };   // TyUnknown → i32. Silent.
    x as i32  // x is a truncated 8-byte pointer in a 4-byte slot
}
```

**Recommended fix**:
- `UnsafeBlock`: return type of inner Block's final_expr (mirror
  Block's behavior).
- `Quote`: return `TyQuote(inner=_check_expr(expr.inner, scope))` —
  add a new Type variant for quoted ASTs.
- `Splice`: validate that `expr.inner` is `TyQuote`-typed and unwrap.
- `Modify`: typecheck target/transformation/verifier; return target's
  type.

**Trap-id reservation**:
11001 — Splice of non-Quote value.

---

## Finding B11: flatten_impls method dispatch picks first-registered type for same method name across structs

**File**: `helixc/frontend/flatten_impls.py:50-53`
**Severity**: MEDIUM
**Category**: dispatch soundness (Stage 8.5 + cross-struct)

**Description**:
```python
method_to_target.setdefault(m.name, item.target)
```

`setdefault` keeps the FIRST registered target. If `Pt` and `Line`
both have `impl ... { fn area(...) -> f32 }`, only `Pt__area` gets
registered in the method map. Subsequent `line_var.area()` calls
rewrite to `Pt__area(line_var)` — type confusion.

The pass-comment at line 13-17 acknowledges: "Disambiguation: if
multiple types have a method with the same name, resolution picks
the FIRST registered type (registration order). For v0.1 we just
emit and let the unresolved-symbol error trigger if the user-side
type doesn't match." But the rewrite IS emitted (line 69-76), and
the target IS registered (line 41-49), so the "unresolved-symbol
error" never triggers. fn_table_lookup of `Pt__area` succeeds; the
caller passes a `Line` value into `Pt__area`'s prologue. SEGV or
silent wrong data.

**Reproducer**:
```helix
struct Pt { x: f32, y: f32 }
struct Line { p1: Pt, p2: Pt }
impl Pt   { fn len(self) -> f32 { self.x } }
impl Line { fn len(self) -> f32 { self.p1.x + self.p2.x } }
fn main() -> f32 {
    let l = Line { p1: Pt { x: 1, y: 2 }, p2: Pt { x: 3, y: 4 } };
    l.len()         // rewrites to Pt__len(l). Pt__len reads .x = 1.
                    // Expected: Line__len returning 4.
}
```

**Recommended fix**:
Two paths:
(a) **Static dispatch by self-type**: at the rewrite site, look up
    `e.callee.obj`'s type via typecheck info; choose the matching
    `<TypeName>__method` entry. Requires running flatten_impls AFTER
    typecheck (currently runs BEFORE in x86_64.__main__).
(b) **Reject same-name methods on different structs** at parse time
    until (a) is implemented — trap 74002 on second registration.

Option (b) is the minimum-correct Phase-0 fix.

**Trap-id reservation**:
74002 — duplicate method name across structs (Phase-0 ambiguity-free
fallback).

---

## Finding B12: monomorphize._walk_subst_expr does NOT descend into A.UnsafeBlock — generics through unsafe are not substituted

**File**: `helixc/frontend/monomorphize.py:91-174`
**Severity**: MEDIUM
**Category**: monomorphization correctness (Stage 28.6 + Stage 28)

**Description**:
The expression walker has arms for Block, If, Match, For, While, Loop,
Binary, Unary, Call, Index, Field, TupleLit, ArrayLit, StructLit,
Assign, Return, Break, Range, Cast, Quote, Splice, Modify. **No arm
for A.UnsafeBlock.** Line 174's catchall `return e` leaves the unsafe
block unchanged — including the inner Block with its TyVar-bearing
let-bindings.

A generic fn whose body uses `unsafe` for any reason (raw-ptr ops,
intrinsics) won't get its type-vars substituted inside the unsafe
region.

**Reproducer**:
```helix
fn read<T>(p: *const T) -> T {
    unsafe {
        let x: T = *p;     // body uses T; not substituted in clone
        x
    }
}
fn main() -> i32 {
    let v: i32 = 42; let p = &v as *const i32;
    read::<i32>(p)
    // mono'd read__i32 body still has `let x: T = *p`. Lower-ast
    // resolves T as TyUnknown (Finding B10), reads as i32 by default.
    // Probably works for T=i32 by accident; breaks for T=f64.
}
```

**Recommended fix**:
Add UnsafeBlock arm to `_walk_subst_expr`:
```python
if isinstance(e, A.UnsafeBlock):
    return A.UnsafeBlock(span=e.span,
                         body=_walk_subst_expr(e.body, subst))
```

**Trap-id reservation**: N/A.

---

## Finding B13: TyDiff binop with mixed inner types silently coerces right to left

**File**: `helixc/frontend/typecheck.py:872-876`
**Severity**: LOW (HIGH for AD soundness once Logic etc. wired)
**Category**: type soundness (Stage 24)

**Description**:
```python
l_is_diff = isinstance(l, TyDiff)
r_is_diff = isinstance(r, TyDiff)
if l_is_diff or r_is_diff:
    inner = l.inner if l_is_diff else r.inner if r_is_diff else l
    return TyDiff(inner=inner)
```

When BOTH operands are `TyDiff` but with different inner types
(`D<f64> + D<i32>`), `inner = l.inner = f64`. Right side's i32 is
silently coerced. The docstring at line 870-871 acknowledges this:
"Mixing D<T1> with D<T2>: result is D<T1> (simplified; real compiler
would unify innerness)." But there's no warning surfaced — the
typechecker happily proceeds.

This may not be a practical bug yet (mixing dtype in AD is unusual),
but combined with B2 (Logic-wrapper drop) and the fact that AD passes
return 0 for unknown nodes (B5), the AD subsystem has multiple
silent-loss paths.

**Recommended fix**:
Either widen-then-warn (`D<f64> + D<i32> → D<f64>` with warning AD002)
or trap (`error: AD operands must have matching inner dtype`).

**Trap-id reservation**:
AD002 — mixed-dtype AD operands (warning by default).

---

## Finding B14: typecheck.Cast doesn't validate source-target compat — useful Cast errors swallowed

**File**: `helixc/frontend/typecheck.py:1124-1126`
**Severity**: LOW
**Category**: usability / type soundness (general)

**Description**:
Beyond Finding B3 (ptr-cast outside unsafe), Cast also accepts:
- `tuple_val as i32` — silent.
- `array_val as f64` — silent.
- `unit_val as Pt` — silent.
- `enum_val as f32` — silent.

None of these have well-defined coercion semantics. The codegen for
unhandled casts typically falls into default-i32 paths and produces
silent garbage.

The Stage 24B notes (docs/STAGE_24B_NOTES.md) acknowledge "Phase-0
cast accepts more than it should." This finding restates the gap
for type-design completeness.

**Recommended fix**:
Build an allowed-cast matrix: int↔int (with width check), int↔float,
float↔float, &T → *const T (inside unsafe only), *T → integer
(usize/u64), int → *T (unsafe only). All other casts: trap 28604
("invalid cast: source {S} cannot convert to {T}").

**Trap-id reservation**:
28604 — invalid scalar cast at typecheck.

---

## What was checked but found OK (no new finding)

- **TyDiff propagation through binop** (line 872-876): correctly fires
  for `D<T> +/- T`. The simplification (right's wrapper dropped) is
  acknowledged in B13 but works for the common case.
- **substitute_ty's TyGeneric arm** (line 85-87): recurses through
  generic args. `D<T>` and `Logic<T>` substitute correctly when used
  as TyGeneric, e.g. fn body `let x: D<T> = ...` mono'd correctly.
- **Stage 16 HBM tile param dtype check**
  (`ir/lower_ast.py:478-487`): strict dtype check (f32/i32/f16/bf16
  only). 1D shape required. These ARE validated. The length-defaults-
  to-0 path is the only soft spot (B8).
- **Stage 25 trace_pass**: shapes correct, no silent windows in
  typecheck or runtime simulation.
- **Stage 22 diagnostics**: cosmetic — pretty caret rendering. No
  type-system implications.
- **Stage 28 struct_mono.instantiate arity check** (line 145-149):
  correctly raises ValueError on arity mismatch.
- **fn monomorphize Monomorphizer.run** (line 203-229): fixed-point
  iteration converges; new clones added to prog.items idempotently.
- **monomorphize Cast.target_ty substitution** (line 95-98): correctly
  substituted.
- **Stage 28.7 deprecated_pass walker** (line 91-93): includes both
  `then`/`else_` AND `then_branch`/`else_branch`. Works.
- **unsafe_pass walker** (line 65-67): includes `then`/`else_`. Works
  for Unary-deref detection inside if-branches (but see B3 — Cast is
  uncovered).
- **TyMemTier cross-tier compatibility check** (line 1417-1420):
  correctly rejects mixed tiers. Explicit `consolidate` / `recall`
  required. Good design.
- **flatten_impls method-call rewriting for non-Field-callee**: leaves
  intact. Only the ambiguous same-name-cross-struct case (B11) is
  broken.

---

## Cross-stage interactions specifically verified

| Pair | Outcome |
|------|--------|
| Stage 8 (generics) + Stage 8.5 (traits) + closures | No Python closure support; bootstrap closures hardcode i32 (B4). Generic fn returning a closure: untested, likely broken at every layer. |
| Stage 12 (AD) + Stage 11 (reflection: Quote/Splice) | AD silently returns 0 for Quote/Splice (B5). Quote/Splice typecheck to TyUnknown (B10). Two silent-loss layers compose. |
| Stage 14.5 (@checkpoint) + Stage 15-16 (tile types) | No Python @checkpoint pass exists; checkpoint is bootstrap-only. Tile shape sub broken (B8). Combined: a @checkpoint kernel with generic-size tile params won't survive mono. |
| Stage 28 (param structs) + Stage 28.6 (unsafe) | Mono doesn't recurse into unsafe (B12). Generic over raw pointers broken (B6). |
| Stage 24 (TyLogic) + Stage 12 (AD) | D<Logic<T>> intended; binop drops Logic (B2). AD doesn't see Logic at all. The Tier-3 moat is unimplemented at the type level. |

---

## Summary

| #   | Severity | Finding |
|-----|----------|---------|
| B1  | HIGH     | Parametric struct monomorphization never invoked; `Pt<i32>` / `Pt<f32>` silently unifiable |
| B2  | HIGH     | TyLogic provenance dropped through every binop (only TyDiff propagates) |
| B3  | HIGH     | Cast accepts arbitrary integer→pointer with no unsafe gate; unsafe-pass doesn't see Cast |
| B4  | HIGH     | Stage 9 closure captures hardcoded to i32 in bootstrap regardless of captured var type |
| B5  | HIGH     | AD passes silently return 0 gradient for Quote/Splice/UnsafeBlock/unhandled nodes |
| B6  | MEDIUM   | `substitute_ty` missing TyPtr arm — generic raw-ptr fns don't substitute |
| B7  | MEDIUM   | Mono clone shares `where_clauses` with template; loses is_extern |
| B8  | MEDIUM   | TyTile shape/memspace never validated at call sites; size-param sub incomplete; length silently 0 |
| B9  | MEDIUM   | pytree flatten/unflatten asymmetric; pytree_depth recurses on cyclic structs |
| B10 | MEDIUM   | typecheck has no Quote/Splice/Modify/UnsafeBlock arms — TyUnknown unifies with everything |
| B11 | MEDIUM   | flatten_impls picks first-registered for same-name methods across structs — cross-struct type confusion |
| B12 | MEDIUM   | mono `_walk_subst_expr` skips A.UnsafeBlock — generics through unsafe unsubstituted |
| B13 | LOW      | TyDiff binop with mixed inner types silently coerces right to left |
| B14 | LOW      | Cast accepts tuple/array/unit/enum as scalar/ptr without restriction |

**5 HIGH, 7 MEDIUM, 2 LOW. Cycle 1 NOT clean.**

**Stop-the-line recommendation**: YES on Findings B1 and B2.

- **B1**: Without invoking `monomorphize_structs` and without
  resolving `TyGeneric` of user structs to mangled `TyStruct`, Stage 28
  is structurally absent. The unit tests exercise the algorithm but the
  pipeline never runs it. Any user-facing demo of parametric structs
  would silently type-pun `Pt<i32>` against `Pt<f32>`. This is the
  single biggest gap from the type-design audit perspective.
- **B2**: Stage 24 (TyLogic / provenance) is advertised as the Tier-3
  strategic moat for neuro-symbolic AGI. The binop handler — the
  central propagation point — has no arm for TyLogic. Every Logic value
  loses its wrapper at first arithmetic touch. The trap 24001 reservation
  exists but is never emitted.

Findings B3 (unsafe cast gate), B4 (closure-capture i32 hardcode), and
B5 (AD-returns-0) are the next tier. B3 affects safety (the entire point
of the `unsafe` block is undermined). B4 is bootstrap-side and produces
runtime SEGVs. B5 affects every grad() over any non-trivial expression.

Findings B6-B12 are MEDIUM and can be batched into a single Stage 28.8
follow-on. They follow a consistent pattern: passes added in stages
8-28.6 silently skip the AST nodes added in later stages.

Findings B13 and B14 are LOW but document type-design completeness gaps.

---

## Prior-finding re-verification

**From audit-stage5-6 (7 open: F2, F4, F9, F10, F11, F12, F13)**: not
re-flagged. Several remain relevant under type-design lens but were
already enumerated in the prior audit.

**From audit-stage7-8 (6 open: F4, F7, F9, F10, F11, F12)**:
- F4 (`clone_with_rewrite` only handles AST_CALL at root): still
  applies, see also B12 for the related Python `_walk_subst_expr`
  missing UnsafeBlock arm.
- F7 (PAT_LIT 32-bit cmp on wide scrut): unchanged, still open.
- F9 (`clone_with_rewrite` deeper recursion): still open.
- F10 (mono clone discards is_checkpoint): related to B7 (mono clone
  shares where_clauses). Both are "mono drops attribute" — same shape.
- F11 (self.method() in impl bodies): still open. Related to B11
  (method dispatch ambiguity).
- F12 (PAT_VARIANT sub-pat disp8 wrap): unchanged, still open.

**From audit-stage28-8-cycle1-codereview**: C1-H1 (panic_pass walker
field names), C1-M1 (deprecated_pass monkey-patch), C1-M2 (struct_mono
skips fn bodies), C1-M3 (test_ffi.py drive-letter), C1-L1 (panic_pass
on Stmts). C1-M2 is RELATED to B1 but distinct: C1-M2 is about
collect_concrete_uses missing fn-body let-tys; B1 is about the entire
pass not being called at all from the pipeline. Both fixes are needed
for the user-facing parametric-struct demo to compile correctly.

---

## Counts (audit-prompt format)

- **5 HIGH** new (B1, B2, B3, B4, B5)
- **7 MEDIUM** new (B6, B7, B8, B9, B10, B11, B12)
- **2 LOW** new (B13, B14)
- **0 CRITICAL** (B1 is HIGH-leaning-CRITICAL; classified HIGH because
  the missing-call could theoretically be patched in one line)

**Stop-the-line**: YES (on B1, B2).
**Cycle-clean determination**: NOT clean — 5 new HIGH findings.

---

## File-path index

- `C:\Projects\Kovostov-Native\helixc\frontend\struct_mono.py` — Finding B1
- `C:\Projects\Kovostov-Native\helixc\frontend\typecheck.py` — Findings B2, B3, B8, B10, B13, B14
- `C:\Projects\Kovostov-Native\helixc\frontend\unsafe_pass.py` — Finding B3
- `C:\Projects\Kovostov-Native\helixc\bootstrap\parser.hx` — Finding B4
- `C:\Projects\Kovostov-Native\helixc\frontend\autodiff.py` — Finding B5
- `C:\Projects\Kovostov-Native\helixc\frontend\autodiff_reverse.py` — Finding B5
- `C:\Projects\Kovostov-Native\helixc\frontend\monomorphize.py` — Findings B6, B7, B12
- `C:\Projects\Kovostov-Native\helixc\frontend\pytree.py` — Finding B9
- `C:\Projects\Kovostov-Native\helixc\frontend\flatten_impls.py` — Finding B11
- `C:\Projects\Kovostov-Native\helixc\check.py` — Finding B1 (no struct-mono import)
- `C:\Projects\Kovostov-Native\helixc\backend\x86_64.py` — Finding B1 (no struct-mono import)
- `C:\Projects\Kovostov-Native\helixc\ir\lower_ast.py` — Finding B8 (length-default-0 path)
