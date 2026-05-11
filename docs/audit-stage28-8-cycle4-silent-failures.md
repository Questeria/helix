# Stage 28.8 Cycle 4 — Silent-Failure Audit

**Date**: 2026-05-11
**Commit**: b3504a2 (read-only audit). Cycle-3 fix-sweep range
3779270..b3504a2 (11 commits: 025d55e, c31158c, ee7aa42, 74b72ec,
3358627, dccfc7e, 2b15928, 3b321e6, a878709, dda3b9d, b3504a2).
**Scope**: All Helix source — `helixc/bootstrap/*.hx`, `helixc/frontend/*.py`,
`helixc/ir/*.py`, `helixc/backend/*.py`, `helixc/stdlib/*.hx`. Specifically
re-audits the six cycle-3 silent-failure fixes (C3-1 through C3-6) and the
nine cycle-3 type-design fixes (D1, D2, D3, D4, D5, D6, D7, D8, D9), with
cross-stage interaction probes per the audit instructions.
**Trigger**: pre-Stage-29 audit gate — Cycle 4 of 5. Re-audits same scope
after Cycle 3 fixes were landed.
**Strict criterion** (per user directive 2026-05-10): cycle counts CLEAN
only when **zero new findings of ANY severity** (CRITICAL/HIGH/MEDIUM/LOW).

**Method**:
1. Read prior cycle silent-failure docs (cycle 1 — 13 findings, cycle 2 —
   6 findings, cycle 3 — 6 findings) to avoid re-flagging already-
   documented findings. Re-read cycle-4 sibling audits (codereview,
   type-design) to avoid re-flagging items they covered.
2. Walked `git log --oneline b3504a2 ^3779270` — 11 cycle-3 fix commits.
   Read `git show <sha>` for each.
3. For each cycle-3 fix's diff, traced data flow forward to check
   whether the fix opened a fresh silent window or left obvious gaps
   adjacent to the headline fix. Cycle-3's pattern was each fix closing
   the headline window but documenting deferred items; cycle-4 confirms
   the fixes work AND identifies adjacent windows the fixes did not
   close PLUS fix-introduced false-positives.
4. Spot-checked the new code for: dispatch holes (walker drift),
   state-leak after exception, error-channel reach, false-positive
   warnings (catch-all fallthrough on well-known leaves), END-TO-END
   pipeline probes (not just unit-level fix correctness).
5. Exercised the cycle-3 fixes via small Python repros (direct module
   invocations) and full CLI invocations of `python -m helixc.check`
   to confirm each documented behavior on the actual surface tool.

**Result**: **8 new findings (1 CRITICAL, 4 HIGH, 2 MEDIUM, 1 LOW)** —
Cycle 4 NOT clean. The dominant pattern is **fix-introduced false
positives** in cycle-3's D1 structural-`_compatible` widening (HIGH
false-positive classes — generic Tensor/Tile, generic Array via TySize,
TyVar arg to TyPrim param), plus a CRITICAL functional regression in
cycle-3's D2 (every call-RHS untyped let in a closure now SIGILLs, even
when the call returns i32). The D9 fix is paper-only — its unit
regression test passes but end-to-end mono still produces broken clones
referring to `id__U` (un-substituted type-var name) because the cycle-3
fix targeted `_walk_subst_expr` but the surrounding `Monomorphizer.run()`
iteration order still processes generic-fn bodies before clones, leaving
turbofish-already-resolved-to-mangled calls in place. C3-5's catch-all
`_ad_warn` over-fires on well-known leaf exprs (Path / Continue).

---

## CRITICAL FINDINGS

### Finding C4-1: D2 fix (3b321e6) tags ALL Call-RHS untyped lets as "non-i32" → closure capture of any i32-returning fn result SIGILLs at runtime

**Location**:
- helixc/bootstrap/parser.hx:2325-2334 (Call-RHS arm in
  `inferred_ty_tag` lookup) — D2 added `if val_tag == 16 { inferred_ty_tag = 12; }`
- helixc/bootstrap/parser.hx:1819-1820 (capture-site `> 0` guard
  reads tag 12 as "non-i32 → trap 76003")
- helixc/tests/test_codegen.py:3471-3477 (regression test codifies
  the broken behavior)
**Severity**: CRITICAL
**Category**: cycle-3-fix-introduced functional regression
**Stage**: 28.8 cycle-3 commit 3b321e6 (D2)

**Description**:
The D2 fix's commit message lists four capture patterns the fix is
supposed to preserve cleanly:
```
- let a = 10; let c = |x| x + a; c(5)   → tag 0, no trap (passes)
- let pi: i32 = 7; let c = |x| x + pi; ... → tag 0, no trap
- let pi = 3.14_f64; let c = |x| x + pi; ... → tag 2, traps
- let pi = get_pi(); let c = |x| x + pi; ... → tag 12, traps (D2)
```

The fourth bullet treats ALL Call-RHS lets as "non-i32" regardless
of the call's actual return type. But the AST tag 16 simply means
"AST_CALL" — it carries no return-type information. So
```
fn make_i32() -> i32 { 42 }
fn main() -> i32 {
    let a = make_i32();             // tag 12 registered
    let c = |x: i32| x + a;         // capture-site guard > 0 → trap 76003
    c(5)                            // SIGILL at runtime
}
```
**produces SIGILL (exit code 132) instead of returning 47.** The
fn `make_i32` returns i32; capturing the result in a closure is
semantically i32 and should be the SAFE case (tag 0).

The regression test in test_codegen.py lines 3471-3477 explicitly
asserts the broken behavior:
```python
assert compile_and_exec(
    "fn get_pi() -> i32 { 3 } "
    "fn main() -> i32 { let pi = get_pi() ; let c = |y| y + pi ; c(0) }"
) == 132, (
    "D2: call-RHS untyped capture now traps 76003 "
    "(SIGILL on closure invocation)"
)
```
This codifies the false positive. `get_pi() -> i32` returns i32;
the capture is semantically safe; SIGILL is wrong. The test's
function name `get_pi` is misleading — the body returns `3` (i32),
not a float. Trap 76003 was originally designed for f64-bit-truncation
in i32 closures; the D2 fix mis-extends it to all Call-RHS.

**Hidden errors**:
- Every legitimate `let x = call_returning_i32(); ... |...| ... x ...`
  pattern produces SIGILL at runtime. The trap 76003 was originally
  for bit-truncating an f64 down to i32 in a closure — a real bug.
  The D2 fix's broadening means trap 76003 now ALSO fires for
  non-bugs.
- Every realistic Helix program with `let x = some_helper(); let c = |...| ... + x;`
  ships a SIGILL even though it pre-D2-fix shipped a working binary.
- The fix's premise — "no typechecker to disambiguate" — was always
  true for the bootstrap parser. The pre-fix behavior (tag -1, pass
  cleanly) was wrong for f64 captures but RIGHT for i32 captures.
  Cycle 3 flipped the polarity: now wrong for i32 captures, right
  for f64 captures. **Net change: trades one class of silent
  miscompile for another, but the new class hits the dominant
  idiom (i32-returning helpers).**

**Recommendation**:
1. REVERT the D2 fix. The bootstrap parser cannot infer return
   types without a typechecker, and pre-fix tag -1 (untracked,
   pass cleanly) was the right behavior for the dominant pattern.
2. Alternatively, reject untyped Call-RHS captures with a
   parse-time error ("closure captures an untyped Call-RHS
   variable — annotate the let with an explicit type"), forcing
   the user to write `let a: i32 = get_pi();` (which then
   registers tag 0 and captures cleanly).
3. Replace the regression test: `let a = make_i32(); let c = |x| x + a; c(5)`
   should return 47 (clean), not 132 (SIGILL). The current test
   codifies the regression.

**Trap-id**: 76003 (existing — being mis-fired).

---

## HIGH FINDINGS

### Finding C4-2: D1's structural `_compatible` has no arm for TyTensor / TyTile → call-boundary false-positive on every generic tensor / tile call

**Location**:
- helixc/frontend/typecheck.py:2141-2205 (`_compatible`)
- helixc/frontend/typecheck.py:730-741 (D1 elif in `_check_call_basic`)
**Severity**: HIGH
**Category**: cycle-3-fix-introduced false positive (incomplete structural arms)
**Stage**: 28.8 cycle-3 commit 74b72ec (D1)

**Description**:
The cycle-3 D1 fix added structural arms to `_compatible` for
TyDiff, TyLogic, TyTuple, TyArray, TyRef, TyPtr, TyFn — but NOT
for TyTensor, TyTile (despite the commit message explicitly listing
"TyTile, TyTensor" as types-to-cover). The cycle-4 type-design audit
E1 catches the TyArray-size symmetry; this finding is the parallel
for TyTensor and TyTile.

Reachable false positive (verified via `python -c`):
```helix
@pure fn norm[N: size](x: tensor<f32, [N]>) -> f32 { 0.0_f32 }
@pure fn use_norm(m: tensor<f32, [3]>) -> f32 {
    norm(m)
}
```
```
type error: call to 'norm': arg 'x' expects tensor<f32, [size:N]>, got tensor<f32, [size_3]>
```

`pty = TyTensor(f32, [TySize('N')])`, `aty = TyTensor(f32, [TyPrim('size_3')])`.
D1 elif at line 730:
- pty is not (TyVar, TySize, TyUnknown) — true (pty is TyTensor)
- aty is not TyUnknown — true
- not BOTH TyPrim — true
- `_compatible` falls through ALL arms (no TyTensor arm) → reaches
  `return a == b` line 2205 → False (frozen-dataclass equality with
  different shape elements)
- D1 emits a spurious error

Same shape applies to TyTile. Verified via test:
```helix
@pure fn norm[N: size](x: [f32; N]) -> f32 { 0.0_f32 }
@pure fn use_norm(m: [f32; 4]) -> f32 { norm(m) }
```
emits `expects [f32; size:N], got [f32; size_4]` — TyArray case
already documented in cycle-4 type-design E1, but the TyTensor /
TyTile cases are NOT in E1.

**Hidden errors**:
- Every generic-shape tensor call (the dominant ML pattern,
  `fn norm[N](x: tensor<f32, [N]>) -> f32`) fails the D1 check
  even though the call IS legitimate (mono will bind N=3 at this
  site).
- Same for TyTile — every `fn tile_op[N](t: tile<f32, [N], REG>)` call
  emits a false-positive.
- Cascade: typecheck now emits an error where pre-D1 it silently
  accepted. Users that wrote correct generic-tensor code see new
  errors after cycle 3 lands — a functional regression at the
  surface.

**Recommendation**:
1. Add explicit structural arms for TyTensor / TyTile mirroring
   the TyArray pattern, with size-defer (cascade-safe TySize /
   TyUnknown) for shape components:
   ```python
   if isinstance(a, TyTensor) and isinstance(b, TyTensor):
       if not self._compatible(a.dtype, b.dtype):
           return False
       if len(a.shape) != len(b.shape):
           return False
       for sa, sb in zip(a.shape, b.shape):
           if isinstance(sa, (TyUnknown, TyVar, TySize)) \
                   or isinstance(sb, (TyUnknown, TyVar, TySize)):
               continue   # defer to mono / cascade-safe
           if not self._compatible(sa, sb):
               return False
       return a.device == b.device and a.layout == b.layout
   if isinstance(a, TyTensor) or isinstance(b, TyTensor):
       return False
   # ... same for TyTile
   ```
2. Either close the loophole in `_compatible` itself OR widen the
   D1 elif filter at line 730 to ALSO skip when either pty or aty
   contains a TySize / TyVar / TyUnknown anywhere in its inner
   shape — a deep cascade-safe check.
3. Regression test: `fn f[N](x: tensor<f32, [N]>); f(matrix_3)`
   typechecks clean. `fn f[N](x: tile<f32, [N], REG>); f(t_3)`
   typechecks clean.

**Trap-id**: n/a (typecheck error mis-fire, no trap-id).

---

### Finding C4-3: D1's `_check_call_basic` elif fires on TyVar arg passed to TyPrim param in a generic body → false-positive on every generic-forwarding call

**Location**:
- helixc/frontend/typecheck.py:730-741 (D1 elif gate)
**Severity**: HIGH
**Category**: cycle-3-fix-introduced false positive
**Stage**: 28.8 cycle-3 commit 74b72ec (D1)

**Description**:
The D1 elif gate at line 730 filters `pty` against `(TyVar, TySize,
TyUnknown)` but does NOT filter `aty` the same way. When a generic
fn's body forwards its generic parameter to a non-generic fn, `pty`
is a concrete prim (e.g., `i32`) and `aty` is `TyVar('T')`. The
elif fires:
- pty is not (TyVar, TySize, TyUnknown) — true (TyPrim)
- aty is not TyUnknown — true (TyVar)
- not both TyPrim — true (aty is TyVar)
- provenance None — true
- `_compatible(TyPrim('i32'), TyVar('T'))` — falls through all
  arms → `a == b` → False
- D1 emits a false-positive error

Reproducer (verified):
```helix
fn check_x(x: i32) -> i32 { x }
fn use_x[T](v: T) -> i32 { check_x(v) }
```
```
type error: call to 'check_x': arg 'x' expects i32, got T
```

The call is **the canonical generic adapter pattern**: a generic
fn forwarding its T-typed value to a concrete-typed helper. Mono
would eventually bind T to a concrete type at the call site of
`use_x`. The body-typecheck of `use_x[T]` should DEFER on TyVar's,
as the rest of the typechecker already does (the top-level filter
at line 730 explicitly excludes TyVar for pty for this reason —
but the symmetric exclusion was omitted for aty).

**Hidden errors**:
- Every "generic forwarder" pattern (a generic fn that calls a
  non-generic helper with the generic value) emits a false-positive.
- The error message is confusing: "expects i32, got T" — the user
  sees their generic param being rejected and likely files an
  invalid bug.
- Cycle-4 type-design E1 mentions this asymmetry for TyArray but
  the broader TyVar/TyPrim case is NOT covered.

**Recommendation**:
1. Extend the D1 elif filter to symmetric TyVar/TySize/TyUnknown
   exclusion:
   ```python
   elif (not isinstance(pty, (TyVar, TySize, TyUnknown))
         and not isinstance(aty, (TyVar, TySize, TyUnknown))
         and not (isinstance(pty, TyPrim) and isinstance(aty, TyPrim))
         and self._logic_provenance_violation_kind(pty, aty) is None
         and not self._compatible(pty, aty)):
   ```
2. Or fix the cascade in `_compatible` itself: add
   `if isinstance(a, TyVar) or isinstance(b, TyVar): return True`
   at the top, mirroring the existing TyUnknown short-circuit.
3. Regression test:
   `fn f(x: i32) -> i32; fn g[T](v: T) -> i32 { f(v) }`
   typechecks clean (TyVar defers).

**Trap-id**: n/a.

---

### Finding C4-4: D9 fix (dccfc7e) is paper-only — `Monomorphizer.run` iteration order means clones still reference unresolved `id__U` after mono completes

**Location**:
- helixc/frontend/monomorphize.py:417-429 (`Monomorphizer.run`
  iteration loop)
- helixc/frontend/monomorphize.py:311-330 (D9 fix in `_walk_subst_expr`)
- helixc/tests/test_struct_mono.py:779-805 (`test_d9_turbofish_inside_generic_body_substituted`
  — unit-only, doesn't probe end-to-end)
**Severity**: HIGH
**Category**: cycle-3 fix is paper-only — unit test passes,
end-to-end pipeline still broken
**Stage**: 28.8 cycle-3 commit dccfc7e (D9)

**Description**:
The D9 fix added a `_walk_subst_expr.Call` arm that substitutes
the callee's `generics` list, so `id::<T>(x)` with subst {T: i32}
becomes `id::<i32>(x)`. The D9 regression test
(`test_d9_turbofish_inside_generic_body_substituted`) hand-builds an
`A.Call(callee=Name('id', generics=[T]))`, runs `_walk_subst_expr`
with `subst={'T': i32}`, and asserts the result has
`generics=[i32]`. The unit-level fix is real.

But the **end-to-end mono pipeline still produces broken clones**.
`Monomorphizer.run` (line 417-429):
```python
while changed:
    changed = False
    for item in list(self.prog.items):
        if isinstance(item, A.FnDecl):
            new_body = self._rewrite_calls_in_block(item.body, item)
            if new_body is not item.body:
                item.body = new_body
                changed = True
```

`_rewrite_calls_in_block` MUTATES `caller`'s body in iteration 1:
discovers `id::<U>(v)` (where U is caller's own generic param),
mangles to `id__U`, clones `id_clone(body: U)`, and **rewrites
caller.body to `id__U(v)` (no more turbofish)**.

In iteration 2, main is walked. `caller::<i32>(7)` is found.
`_instantiate(caller, "caller__i32", {U: i32})` clones with subst.
`_walk_subst_expr(caller.body)` walks the (already-rewritten)
`id__U(v)` body. The D9 fix at line 321-327 checks
`new_callee.generics` — but the rewrite has already cleared this
to an empty list. The substitution at line 325 is a no-op. The
clone calls `id__U`, which is registered with body using U as a
generic param that was never bound to i32.

**Verified end-to-end via direct probe**:
```
$ python -c "...monomorphize(prog)..." on:
   fn id[T](x: T) -> T { x }
   fn caller[U](v: U) -> U { id::<U>(v) }
   fn main() -> i32 { caller::<i32>(7) }
$ output (call-trace dump):
   caller: Call id__U generics=[]
   main: Call caller__i32 generics=[]
   caller__i32: Call id__U generics=[]
```

The clone `caller__i32` calls `id__U`, which is registered with
body `(x: U) -> U` (U unresolved). Codegen would either crash or
silently produce broken output.

Reversing item order (main-first) doesn't help — it just shifts the
asymmetry from "id__U exists with unresolved U" to "id::<i32> in
body with no `id__i32` mono'd". Both orders produce a broken final
state.

Also: clones appended at line 428 AFTER the while loop exits are
NEVER walked for further mono — so any unresolved turbofish in
clones doesn't get followed up. Cycle-4 type-design audit's D9 row
("OK, no new issues") was based on the unit test alone; it did not
probe end-to-end.

**Hidden errors**:
- Any nested-generic call pattern (`caller[T]` calls `inner::<T>(...)`)
  produces broken mono'd code.
- Codegen sees calls to functions with unresolved generic params.
- The D9 regression test in test_struct_mono.py passes — it only
  unit-tests `_walk_subst_expr` directly with a hand-built Call
  that has intact turbofish. It does NOT run the end-to-end
  `monomorphize(prog)` pipeline.

**Recommendation**:
1. Fix the iteration order: delay rewriting generic-fn bodies
   until AFTER clones are produced. Track turbofish call sites
   in a separate list and process them in a post-iteration step.
2. Or change `_rewrite_calls_in_block` to ONLY rewrite calls
   inside non-generic fns (those with empty `generics`). Generic
   fns' bodies are only walked through `_instantiate` (cloned per
   subst).
3. Or include the clones in the next iteration so their bodies'
   remaining turbofish get followed:
   ```python
   self.prog.items = list(self.prog.items) + list(self.instantiated.values())
   self.instantiated = {}
   changed = True
   ```
   inside the while loop.
4. Add an end-to-end regression test:
   ```python
   def test_d9_end_to_end_nested_turbofish():
       src = '''fn id[T](x: T) -> T { x }
                fn caller[U](v: U) -> U { id::<U>(v) }
                fn main() -> i32 { caller::<i32>(7) }'''
       prog = parse(src)
       monomorphize(prog)
       # Walk prog: assert NO Call references id__U.
       # Assert caller__i32 calls id__i32, NOT id__U.
   ```
5. Or revert D9 fix as paper-only and revisit the mono pipeline
   design holistically.

**Trap-id**: n/a.

---

### Finding C4-5: `_inline_lets` catch-all `_ad_warn` false-positives on `A.Path` and `A.Continue` — every `Enum::Variant` reference in a differentiated fn body now spurious-warns

**Location**:
- helixc/frontend/autodiff.py:679-686 (catch-all fallthrough warn)
**Severity**: HIGH
**Category**: cycle-3-fix-introduced false-positive
**Stage**: 28.8 cycle-3 commit 3358627 (C3-5)

**Description**:
The C3-5 fix added comprehensive recursion for Cast / Call / Field /
Index / ArrayLit / TupleLit / StructLit / Range / Return / Break /
Assign / UnsafeBlock / Match / Loop / For / While / Quote / Splice /
Modify. To future-proof future AST extensions, the fix added a
catch-all fallthrough at line 679-686:

```python
_ad_warn(
    expr,
    f"_inline_lets fell through on Expr subtype "
    f"'{type(expr).__name__}' — let-bindings beyond this point "
    f"may not be substituted (trap 85001)",
)
return expr
```

But two existing well-known leaf Expr subtypes have NO explicit
dispatch arm and fall through to the catch-all:

1. **A.Path** — `Maybe::None`, `Enum::Variant`, any qualified-name
   reference. These are common in fn bodies that branch on enum
   discriminators. They contain no Name leaves (segments are strings),
   so let-bindings beyond them are not at risk — the warning is a
   false positive.
2. **A.Continue** — `continue;` statement-expr. No children at all.
   Same false-positive class.

The cycle-3 commit message said the catch-all is "loud-fall-through so
any future AST extension surfaces immediately". But it ALSO fires for
existing AST nodes that don't need recursion.

**Reproducer** (verified via `python -m helixc.check --emit-ir -Wad=error`):

```helix
enum Maybe { None, Some(i32) }
@pure fn loss(x: f64) -> f64 {
    let tag = Maybe::None;
    x * x
}
fn main() -> i32 {
    grad(loss)(3.14_f64);
    0
}
```

Output: `AD: assumed 0 derivative for Path (_inline_lets fell through ...) (trap 85001)`.
`-Wad=error` fails the compile.

Pre-fix (cycle-2 state): A.Path silently returned unchanged, no
warning. The user's program compiled clean. Post-fix: same program
emits the warn, and `-Wad=error` FAILS the compile.

**Hidden errors**:
- Every AD-able fn containing an enum-variant reference (`Maybe::None`,
  `Result::Err`, `Token::EOF`, etc.) silently fails `-Wad=error`.
- Same for any `continue;` in an AD'd loop body.
- A.Path and A.Continue are leaf-ish exprs that don't contain
  Name leaves — the warning's premise ("let-bindings beyond this
  point may not be substituted") is false for them.
- The C3-5 regression test exercises Cast only; the Path /
  Continue false-positive sneaks past the test.

**Recommendation**:
1. Add explicit no-op dispatch arms for A.Path and A.Continue:
   ```python
   if isinstance(expr, A.Path):
       return expr  # paths have no Name leaves to substitute
   if isinstance(expr, A.Continue):
       return expr  # continue is leaf-like
   ```
2. Audit ast_nodes.py for all Expr subtypes; classify each as
   {recurse-into / leaf-no-op / unknown-warn} explicitly. The
   catch-all should fire only for the third category.
3. Regression test: `_inline_lets(A.Path(...), {})` produces ZERO
   AD warnings.

**Trap-id**: 85001 (existing AD-assumed-zero, but fired spuriously).

---

## MEDIUM FINDINGS

### Finding C4-6: D2 closure-capture inference covers only Call RHS — Binary/Unary/Index/Field/If/Match/Block RHS still silently leave inferred_ty_tag = -1

**Location**:
- helixc/bootstrap/parser.hx:2300-2334 (let-RHS inferred-tag dispatch)
- helixc/bootstrap/parser.hx:1809 (closure-capture guard `cap_ty_tag > 0`)
**Severity**: MEDIUM
**Category**: cycle-3-fix-introduced incompleteness
**Stage**: 28.8 cycle-3 commit 3b321e6 (D2)

**Description**:
The D2 fix at parser.hx:2325-2333 added ONE arm for `val_tag == 16`
(AST_CALL) so `let pi = get_pi(); let c = |x| x + pi;` registers
inferred tag 12. But the existing inferred-tag chain (lines 2300-2334)
only covers 12 literal val_tags. Every other AST root tag falls through
with `inferred_ty_tag = -1` (the initial value). Common forms NOT
covered:

- **Binary** RHS (val_tag = 2): `let pi = a + b;` — bit pattern of
  pi depends on operand types, could be f64-bits captured as i32.
- **Unary** RHS (val_tag = 3): `let neg_pi = -pi;` (post-cycle-3
  D5 fold may produce IntLit, but Unary on non-IntLit still falls).
- **Index** RHS (val_tag = 18): `let v = arr[i];`.
- **Field** RHS (val_tag = 19): `let v = obj.field;`.
- **If** RHS (val_tag = 11): `let v = if cond { 1.0 } else { 2.0 };`.
- **Match** RHS (val_tag = 12): `let v = match x { ... };`.
- **Block** RHS (val_tag = 1): `let v = { ... };`.
- **UnsafeBlock** RHS (val_tag = 17): `let v = unsafe { ... };`.

This finding is the dual of C4-1 (CRITICAL). C4-1 covers the
over-trapping in Call RHS; this finding covers the silent
under-trapping in the other RHS classes. Both pathologies exist
in the current code because D2's polarity choice is wrong for the
dominant idiom.

**Reproducer** (parser.hx-level):

```helix
fn main() -> i32 {
    let pi: f64 = 1.0_f64 + 2.14_f64;  // Binary RHS — inferred_ty_tag = -1
    let c = |y| y + pi;                 // capture passes silently
    c(0);                               // pi's f64 bits captured as i32
    0
}
```

Expected: trap 76003 at closure-capture (pi is f64).
Actual (post-D2): silent capture, bit-truncation of f64 → i32.

**Hidden errors**:
- Every closure capturing a let-bound value computed from any
  non-literal-non-Call expression silently truncates to i32.
- The bootstrap kovc.hx itself contains many such patterns (arithmetic
  let-bindings inside helper fns). Currently safe only because those
  particular fns don't define closures, but the moment a closure
  captures such a let, silent miscompile.
- Future Helix programs that look "clean" by passing the existing
  trap-76003 test suite would still silently miscompile under common
  closure patterns.

**Recommendation**:
1. The right fix here is tied to C4-1's recommendation: REVERT D2's
   tag-12 polarity. Without a typechecker, the bootstrap parser
   cannot reliably narrow types from any of these RHS classes.
2. Alternatively, if cycle-5 retains D2's tag-12 sentinel, extend
   the dispatch to ALL non-trivially-i32 RHS classes (everything
   except `val_tag == 0` and explicit-i32-annotation paths). But
   this still produces C4-1's false positive for Call-RHS i32 fns.
3. Best long-term: introduce a real local-typecheck pass in the
   bootstrap parser, OR require an explicit annotation for
   closure-capture lets, with parse-time rejection of untyped
   captures. The current half-measure fix is unstable under any
   choice.

**Trap-id**: 76003 (existing — under-fires for these classes,
over-fires for Call RHS per C4-1).

---

### Finding C4-7: C3-3 `except Exception` mis-attributes user-environment errors as "compiler bug" (file I/O, encoding, import errors)

**Location**:
- helixc/check.py:264-292 (outer `main` wrapper)
**Severity**: MEDIUM
**Category**: cycle-3-fix-introduced over-broad catch
**Stage**: 28.8 cycle-3 commit ee7aa42 (C3-3)

**Description**:
The C3-3 fix wraps `_main_inner` in `try/except Exception`. This is
broader than necessary and produces wrong attribution for legitimate
user-environment errors:

1. **TOCTOU on file open** — line 316 `os.path.exists(path)` then
   line 320 `open(path, "r", encoding="utf-8")`. A `FileNotFoundError`
   between the two now prints "internal error: FileNotFoundError ...
   compiler bug — please file an issue."

2. **Encoding errors** — line 321 `f.read()` raises
   `UnicodeDecodeError` if the file isn't UTF-8. Now also "internal
   error ... compiler bug" instead of a clean diagnostic naming the
   file and encoding mismatch.

3. **ImportError** — A pipeline-phase module not installed (e.g.,
   `from .frontend.struct_mono import monomorphize_structs` line 376
   fails) gets attributed as "compiler bug" instead of "environment
   issue — module not found."

4. **Finally-raises** — If `_drain_ad_warnings(a_holder[0])` itself
   raises in the `finally` block, the new exception masks the
   original and propagates as a raw traceback (the C3-3 wrapper
   doesn't protect the finally).

The cycle-4 codereview audit at confidence-below-threshold described
this as "appropriate top-level CLI usage". By the strict-criterion
silent-failure lens, it's an over-broad catch that misattributes
user errors. Both perspectives are defensible — the silent-failure
lens flags it because the user-facing message is wrong, not because
the exception goes unhandled.

**Hidden errors**:
- Users with file-not-found or encoding issues get told their
  legitimate error is a compiler bug → noisy bug reports.
- The "compiler bug" message text is now too prominent — every
  hard error gets it, eroding signal value.
- Pre-cycle-3: only the explicit `compile_module_to_elf` exception
  at line 572 had this wrap. Post-cycle-3: the entire pipeline
  shares it.

**Recommendation**:
1. Catch specific known-compiler-bug exception classes and let
   user-env exceptions propagate to a distinct catch:
   ```python
   try:
       rc = _main_inner(argv, a_holder)
   except (FileNotFoundError, PermissionError, IsADirectoryError) as e:
       print(f"helixc: {e}", file=sys.stderr)
       rc = 2
   except UnicodeDecodeError as e:
       print(f"helixc: encoding error reading source: {e}", file=sys.stderr)
       rc = 2
   except (AttributeError, KeyError, IndexError, AssertionError,
           TypeError, RuntimeError, ValueError) as e:
       print(f"helixc: internal error: {type(e).__name__}: {e}",
             file=sys.stderr)
       print("helixc: this is a compiler bug — please file an issue.",
             file=sys.stderr)
       rc = 1
   ```
2. Move file-open + encoding logic OUT of `_main_inner` into the
   outer wrapper or into a separate try/except in `_main_inner`
   that returns rc=2 with a clean message.
3. Wrap the `finally` block's `_drain_ad_warnings` call in its own
   try/except so a drain failure doesn't mask the primary failure.

**Trap-id**: n/a.

---

## LOW FINDINGS

### Finding C4-8: `Monomorphizer.run()` does NOT catch `ShapeFoldError` (trap 28801); fn-mono path mis-attributes as compiler bug via C3-3 wrapper

**Location**:
- helixc/frontend/monomorphize.py:403-429 (`Monomorphizer.run`)
- helixc/frontend/struct_mono.py:445-456 (the only catch site)
- helixc/check.py:264-292 (where the unhandled exception lands)
**Severity**: LOW
**Category**: cycle-3-fix-introduced misattribution / unreachable via
current parser
**Stage**: 28.8 cycle-3 commits dccfc7e (C3-6) + 2b15928 (struct_mono
catch)

**Description**:
The C3-6 fix added `ShapeFoldError(ValueError)` (monomorphize.py:63)
and wired the catch in `struct_mono.monomorphize_structs` at line 448.
But the C3-6 commit did NOT wire the same catch in
`Monomorphizer.run()` (the fn-mono pass) at line 403-429.

When invoked via the x86_64 legacy CLI driver (which calls
`monomorphize(prog)` at line 3021), if a generic fn has a shape
expression with `/0` or `%0` after substitution, ShapeFoldError
propagates uncaught. Through `check.py` paths it gets caught by
the C3-3 outer wrapper and printed as:

```
helixc: internal error: ShapeFoldError: <message>
helixc: this is a compiler bug — please file an issue.
```

This is a **false bug report**: the user wrote source that triggered
a legitimate trap 28801 (shape fold by zero), and the compiler tells
them to file an issue instead of pointing at the array dim.

**Why LOW**: the current Phase-0 parser does NOT accept const-N
generic params on fns or arithmetic shape expressions on `tensor<...>`
/ `tile<...>` / `[T; ...]` in fn signatures fully. So the only
reliable path to trigger this is via direct Python API construction
of an AST that bypasses the parser. If future stages widen the
parser, this finding gets promoted to MEDIUM.

This finding overlaps with the cycle-4 type-design audit's E3 (same
issue, different perspective). Reported here for completeness of the
silent-failure perspective.

**Recommendation**:
1. Wrap the per-instance try in `Monomorphizer._rewrite_calls_in_expr`
   (where `_instantiate` is called):
   ```python
   if key not in self.instantiated:
       try:
           self.instantiated[key] = self._instantiate(fn, mangled, subst)
       except ShapeFoldError as e:
           self.diags.append(str(e))  # accumulate
           continue   # skip this instantiation
   ```
2. Add a `diags` field on Monomorphizer parallel to struct_mono's
   `(prog, diags)` return tuple.
3. Regression test exercises the fn-mono path with a generic fn
   `fn k[T, N](a: [T; N/0])`, asserts diag is recorded as trap
   28801 (not a raw raise).

**Trap-id**: 28801 (existing).

---

## Cycle 3 fix re-verification

Each of the 11 cycle-3 commits was inspected for paper-only fixes,
silent windows, and false positives introduced.

| Cycle-3 commit | Fix | Real? | C4 regression? |
|---|---|---|---|
| 025d55e | C3-1 grad_pass chained else-if | YES | none |
| c31154b | Cycle 3 audit-findings persistence | YES (docs) | none |
| ee7aa42 | C3-3 try/finally main | YES | **C4-7** (over-broad except Exception) |
| 74b72ec | C3-2 + D1 + D3 + D4 + D7 + D8 | YES | **C4-2** (D1 missing TyTensor/TyTile arms), **C4-3** (D1 asymmetric TyVar filter) |
| 3358627 | C3-5 _inline_lets wide recursion | YES | **C4-5** (catch-all over-fires on Path/Continue) |
| dccfc7e | C3-6 + D5 + D9 | YES (paper-only for D9) | **C4-4** (D9 is paper-only — end-to-end mono still broken) |
| 2b15928 | C3-4 + D6 + ShapeFoldError catch | YES | **C4-8** (Monomorphizer.run uncaught ShapeFoldError) |
| 3b321e6 | D2 closure trap on call-RHS lets | YES | **C4-1** (CRITICAL — tags ALL Call-RHS as non-i32; i32-fn results SIGILL), **C4-6** (other RHS classes still silent) |
| a878709 | trap IDs 28801/28802/28803 reserved | YES | none |
| dda3b9d | regression tests for 15 fixes | YES (but D9 test is unit-only, doesn't catch C4-4) | tested at unit-level only |
| b3504a2 | persist cycle-3 audit docs + edits batch | YES | (covered above) |

### Specific re-verifications from the audit instructions

- **74b72ec (D1 structural _compatible)**: introduces 2 new
  false-positive classes (C4-2 generic Tensor/Tile, C4-3 TyVar
  asymmetric filter — plus E1/E2 from cycle-4 type-design covering
  TyArray-size and Logic-wrap-asymmetric). The cycle-3 commit message
  claimed "structural arms for TyDiff / TyLogic / TyTuple / TyArray /
  TyRef / TyPtr / TyFn" — TyTensor / TyTile / TyStruct / TySkill /
  TyUnit / TyMemTier-inner were left to fall through to `a == b`.
  Material reach gap.
- **ee7aa42 (C3-3 try/finally)**: catches `Exception` broadly. C4-7
  documents three classes of user-env errors mis-attributed as
  "compiler bug." Per the strict criterion, the wrapper should
  catch only known internal-error classes.
- **3358627 (C3-5 _inline_lets)**: closes Cast / Call / Field /
  Index reach gap correctly. But the catch-all over-fires on Path
  and Continue (C4-5). C3-5's recursion is exponential-O(2^n) for
  pathological cases — same as pre-fix Block-arm behavior, so not
  a new finding.
- **dccfc7e (C3-6 + D5 + D9)**: C3-6 shape-fold trap on `/0` and
  `%0` works correctly. D5 Unary fold is depth-1 only (cycle-4
  type-design E1 covers TyArray-size symmetry; my D5 probe shows
  Cast-around-IntLit isn't folded but no current source-level
  pattern reaches this path). D9 is paper-only — see C4-4 for the
  end-to-end mono breakage.
- **2b15928 (struct_mono dedup)**: idempotency guard is name-based,
  not structure-based — cycle-3 didn't claim structure-based dedup,
  so this is acceptable behavior. ShapeFoldError catch is correctly
  wired ONLY for struct_mono path; fn-mono uncaught — see C4-8.
- **3b321e6 (parser.hx D2)**: CRITICAL. The fix tags ALL Call-RHS
  untyped lets as non-i32, making every legitimate i32-returning fn
  call → closure-capture pattern SIGILL at runtime. See C4-1. The
  fix should be REVERTED; the bootstrap parser cannot infer return
  types and the polarity choice should preserve the dominant pattern
  (i32-returning helpers) rather than the rare pattern (f64-returning
  via untyped let).

---

## Cross-stage interactions checked

- **_compatible structural recursion + cyclic TyRef chains**:
  Phase-0 has no syntax for recursive types; cyclic TyRef chains
  cannot be constructed. Recursive _compatible is safely bounded.
- **TyArray size-zero diagnostic + autotune variant gen**: autotune
  generates int parameter variants for `BLOCK_SIZE` etc. via
  Cartesian product, not array shapes. No interaction.
- **ShapeFoldError catch contract + double-mono path**: fn-mono path
  uncaught (C4-8). struct_mono path correctly catches.
- **D1 structural _compatible + tensor with TySize shapes**: false
  positive (C4-2).
- **C3-3 try/except + finally raises**: not protected (C4-7).
- **D1 + Unary `&` typecheck imprecision**: pre-existing
  imprecision exposed as new false-positive — but the imprecision
  itself is pre-cycle-3, so this is a side-effect of D1's widening
  rather than a fresh cycle-3 issue. Documented as deferred
  observation rather than new finding.
- **C3-5 _inline_lets exponential blow-up**: same complexity as
  pre-fix; not a new issue.

---

## What was checked but found OK (no new finding)

- C3-1 chained else-if: works correctly. `_rewrite_in_expr` A.If
  arm recurses into chained `else_ = A.If(...)`. Both Block and If
  else branches covered.
- C3-2 pointer-width aliases: `D<i64> + D<isize>` produces ZERO
  warnings. `_widen_canon_name` maps isize→i64 and usize→u64.
- C3-3 try/finally drain: drain runs on every exit path. State
  is clean after the call. (Issue is the broad-except — C4-7.)
- C3-4 idempotency: `monomorphize_structs(prog); monomorphize_structs(prog)`
  produces exactly one `Pt__i32`. (Note: name-based dedup hides
  re-mono after AST mutation — would be a finding if any pass
  mutated structs between mono calls; none do today.)
- C3-5 dispatch reach: Cast / Call / Field / Index work correctly.
  (Catch-all over-fires per C4-5; A.If.cond NOT inlined per
  prior-cycle audit's deferred observation — pre-existing.)
- C3-6 shape-fold trap: `/0` and `%0` raise ShapeFoldError. The
  struct_mono caller catches and surfaces. (fn-mono caller misses
  per C4-8.)
- D1 call-boundary: TyStruct mismatch errors correctly. Logic-
  provenance path produces specialized diagnostic. (TyTile/TyTensor
  gap per C4-2; TyVar asymmetric filter per C4-3.)
- D2 call-RHS closure trap: fires on Call (per CRITICAL C4-1
  false-positive). Other RHS classes silent per C4-6.
- D3 array size <= 0: trap 28802 fires on `[T; 0]`, `[T; -5]`,
  `tensor<f32, [0]>`. All paths correct.
- D4 Logic-Logic mixed-inner: `Logic<f64> + Logic<i32>` emits
  AD002. `tie_fired` flag suppresses double-emission. (Logic-wrap
  asymmetric — cycle-4 type-design E2 — not in this audit's scope.)
- D5 Unary fold: `Unary(-, IntLit(N))` folds correctly. (Depth-1
  fold limit observed; reachable only via direct AST construction
  today.)
- D6 _ty_key strict guard: raises TypeError on non-TyNode input.
  Defensive guard works.
- D7 ref-peel depth guard: 500-layer cast traps 28803. Diagnostic
  loses `&` prefix in error message — cycle-4 codereview below-
  threshold note; documented for cycle-5 attention.
- D8 _fmt TyStruct: prints `Foo` not `TyStruct(name='Foo')`.
- Trap-ids 28801/28802/28803 reserved correctly. 28801 has working
  catcher in struct_mono; 28802 emits directly from
  `_resolve_size_expr`; 28803 emits directly from
  `_check_cast_compat`. All wired.
- C3-3 + C2-1 interaction: leading `_drain_ad_init()` clears stale
  state, inner pipeline accumulates, drain in finally surfaces.
  Re-running `main()` twice in one process doesn't leak warnings.

---

## Deferred / out-of-scope observations (NOT new findings; cycle-5 candidates)

- **`_check_cast_compat` `&mut T as &T` silent acceptance**: the
  D7 fix peels matching ref-pairs iteratively. The `src.inner ==
  tgt.inner` shortcut at line 2073 returns immediately at any peel
  layer where inner types match, regardless of `is_mut` on the
  wrappers. Pre-existing (documented in cycle 3 deferred); D7
  did not close it.
- **D7 cast-matrix diagnostic loses `&` prefix**: cycle-4
  codereview at confidence 73 below threshold. Recorded for
  cycle-5.
- **`_inline_lets` A.If.cond not inlined**: pre-existing A.If arm
  preserves `expr.cond` literally. C3-5 added cond inlining for
  While/For/Loop but skipped the existing If arm. Documented in
  cycle-4 type-design context and prior-cycle deferred observations.
- **D5 Unary fold non-compositional through Cast**: `Unary(-, Cast(IntLit, i64))`
  doesn't fold. Reachable only via direct AST construction at this
  cycle's parser surface — not a hot path. Documented for cycle-5.
- **D9 fix paper-only at end-to-end**: see C4-4. The fix changes
  `_walk_subst_expr` but the surrounding iteration order means
  clones still reference unresolved generic-param names. Recommend
  cycle-5 either revert D9 or fix the iteration order holistically.
- **Tag-12 sentinel namespace overlap**: cycle-4 type-design E5
  captures.
- **C3-3 catches AssertionError as "compiler bug"**: pre-existing
  acceptable trade-off. AssertionErrors from genuine compiler-state
  asserts vs user-caused bad state aren't distinguishable.

---

## Summary

| #    | Severity | Location                                                    | Finding                                                                                                            |
|------|----------|-------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| C4-1 | CRITICAL | parser.hx:2325-2334 + 1819-1820                             | D2 tags ALL Call-RHS untyped lets as non-i32 — i32-returning fn result captures SIGILL (trap 76003)                  |
| C4-2 | HIGH     | typecheck.py:2141-2205 (`_compatible`)                      | D1 has no TyTensor/TyTile structural arms — generic tensor/tile calls emit false-positive errors                    |
| C4-3 | HIGH     | typecheck.py:730-741 (D1 elif)                              | D1 elif filters TyVar/TySize on pty side only — `fn g[T](v: T) { f(v) }` with `fn f(i: i32)` mis-fires              |
| C4-4 | HIGH     | monomorphize.py:417-429 (Monomorphizer.run iteration)        | D9 fix is paper-only — Monomorphizer iteration order means clones still call unresolved `id__U` after mono           |
| C4-5 | HIGH     | autodiff.py:679-686 (catch-all `_ad_warn` in `_inline_lets`) | A.Path / A.Continue trigger false-positive AD warnings; `-Wad=error` fails compile on enum-variant paths in AD'd fns |
| C4-6 | MEDIUM   | parser.hx:2300-2334 (let-RHS inferred-tag dispatch)         | D2 covers only Call RHS; Binary/Unary/Index/Field/If/Match/Block RHS still silently leave inferred_ty_tag = -1       |
| C4-7 | MEDIUM   | check.py:264-292 (outer `main` wrapper)                     | C3-3 `except Exception` mis-attributes file-not-found / encoding / import errors as "compiler bug"                  |
| C4-8 | LOW      | monomorphize.py:403-429 (`Monomorphizer.run`)               | C3-6 wired the ShapeFoldError catch only in struct_mono; fn-mono path uncaught → trap 28801 misattributed            |

**Total: 8 new findings (1 CRITICAL, 4 HIGH, 2 MEDIUM, 1 LOW).**

---

## Cycle 4 status

**Cycle 4 NOT clean.** Per the strict criterion (zero findings of
ANY severity), the 1 CRITICAL + 4 HIGH + 2 MEDIUM + 1 LOW new
findings BLOCK the cycle-4 clean determination.

### Stop-the-line determination: **YES, on C4-1, C4-4, and C4-2/C4-3**.

**C4-1 (CRITICAL)** is the highest-priority finding. The D2 fix
trades one class of silent miscompile (f64-bit-truncation in
closures, rare) for another (SIGILL on i32-fn-result captures,
common). The current regression test
`test_bootstrap_kovc_full_pipeline_arithmetic` codifies the broken
behavior — the test asserts `c(0)` returns 132 (SIGILL) when the
captured value comes from `get_pi() -> i32`. This is a functional
regression at the surface tool level.

**C4-4 (HIGH)** documents that the D9 fix is paper-only. The
regression test passes (it's a unit test against `_walk_subst_expr`
directly), but the end-to-end mono pipeline still produces clones
referring to `id__U` (unresolved generic param). The cycle-4
type-design audit's D9 row says "OK, no new issues" — that
verdict is based on the unit test, not end-to-end probe.
**This is a paper-only fix and should be either reverted or the
mono iteration order fixed.**

**C4-2 (HIGH)** affects every ML pattern using parametric tensors.
Generic-tensor calls produce false-positive errors. Same for tiles.

**C4-3 (HIGH)** is the most user-visible D1 regression for non-ML
code. Every generic-forwarding pattern (`fn g[T](v: T) -> i32 { f(v) }`
where `f` takes a concrete type) now emits a confusing "expects
i32, got T" error.

**C4-5 (HIGH)** is the most user-visible cycle-3 fix collateral
for AD code: every legitimate AD-able function that references an
enum variant (`Maybe::None`, `Result::Err`, etc.) or contains
`continue;` in a loop now emits a spurious AD warning.

**C4-6, C4-7 (MEDIUM)** are the cycle-3-fix-introduced silent
windows. C4-6 leaves other closure-capture RHS classes silently
truncating; C4-7 mis-attributes user errors.

**C4-8 (LOW)** is gated behind unimplemented parser features today.

### Cycle 4 → NEW FINDINGS COUNT for the strict-clean gate: 8 (1 CRITICAL + 4 HIGH + 2 MEDIUM + 1 LOW) — clean-counter remains at 0.

### Estimated remaining open findings going into cycle 5

- Cycle 1: 13 new (all fixed → 0 open).
- Cycle 2: 6 new (all fixed → 0 open).
- Cycle 3: 6 new (all fixed → 0 open).
- Cycle 4 silent-failure: 8 new (all open).
- Cycle 4 type-design (sibling audit): 8 new (all open).
- Cycle 4 codereview (sibling audit): 0 new.
- Prior audits (stage 5-6 + 7-8 + 9-16): 20 still-open at start
  of cycle 4 (unchanged from cycle 3).
- Cycle 4 net: 20 + 8 + 8 = **36 open findings** going into cycle 5.

Recommend prioritizing in this order for the cycle-5 fix batch:
1. **C4-1** (CRITICAL — REVERT D2 fix; codified test asserts
   broken behavior; every closure-of-i32-fn-result currently
   SIGILLs).
2. **C4-4** (HIGH — revert D9 fix or fix Monomorphizer iteration
   order; paper-only with passing unit test that doesn't probe
   end-to-end).
3. **C4-2** (HIGH — add TyTensor/TyTile arms to `_compatible`).
4. **C4-3** (HIGH — symmetric TyVar/TySize/TyUnknown filter on
   aty in D1 elif, or add TyVar short-circuit to `_compatible`).
5. **C4-5** (HIGH — add explicit Path/Continue no-op arms to
   `_inline_lets`).
6. **C4-6** (MEDIUM — couples with C4-1; revert D2 polarity).
7. **C4-7** (MEDIUM — narrow C3-3 `except Exception` to
   internal-error classes only).
8. **C4-8** (LOW — wrap `Monomorphizer.run` in ShapeFoldError
   catch, accumulate via `diags` field).
