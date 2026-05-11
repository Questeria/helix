# Stage 28.8 Pre-29 Audit Gate — Cycle 5, Audit A: Silent Failures

**Date**: 2026-05-11
**Commit**: 960303b (read-only audit). Cycle-4 fix-sweep range
b3504a2..960303b (2 commits: a59e233 audit-C C4-1..C4-5 + persist
cycle-4 docs; 960303b fix-sweep C4-1..C4-5 / E1..E8).
**Scope**: All Helix source — `helixc/bootstrap/*.hx`,
`helixc/frontend/*.py`, `helixc/ir/*.py`, `helixc/backend/*.py`,
`helixc/stdlib/*.hx`. Specifically re-audits both the eight cycle-4
silent-failure audit findings (C4-1..C4-8) AND the cycle-4 fix-sweep
commit (which renumbered its labels to C4-1..C4-5 + E1..E8). Also
audits the entire upstream stages 1-28.7 stack for silent-failure
patterns not previously caught (ast-hash dispatch, lower-ast
fallbacks, monomorphize.py turbofish-arity, monomorphize.py
substitution walker dispatch coverage).
**Trigger**: pre-Stage-29 audit gate — Cycle 5 of 5. Re-audits same
scope after Cycle 4 fixes were landed.
**Strict criterion** (per user directive 2026-05-10): cycle counts
CLEAN only when **zero new findings of ANY severity** (CRITICAL/HIGH/
MEDIUM/LOW).

**Method**:
1. Read prior cycle silent-failure docs (cycle 1 — 13 findings; cycle
   2 — 6 findings; cycle 3 — 6 findings; cycle 4 — 8 findings) plus
   cycle-5 sibling audits (type-design F1-F6, codereview C5-1/C5-2)
   to avoid re-flagging already-documented findings.
2. Cross-checked the cycle-4 *audit* C4-1..C4-8 numbering against
   the cycle-4 *fix-sweep* C4-1..C4-5 + E1..E8 numbering. Three
   cycle-4 audit findings (C4-3 HIGH, C4-4 HIGH, C4-7 MEDIUM) have
   no equivalent in the fix-sweep commit and remain unaddressed at
   HEAD.
3. Walked the cycle-4 fix-sweep diff line-by-line; traced each fix
   forward for fix-introduced silent windows.
4. Walked every `except Exception`/`return None`/`return False`/
   `pass`/`default to 0`/`silent` site in the in-scope source.
5. Audited every dispatch-by-type walker (`_inline_lets`,
   `_walk_subst_expr`, `_rewrite_calls_in_expr`, `_hash_into`,
   `_resolve_in_expr`, `_rewrite_in_expr`, `_compatible`) for
   completeness against the full Expr taxonomy in
   `helixc/frontend/ast_nodes.py`.
6. Cross-checked walker-fall-through false claims (comments that
   say "typecheck will flag X" against whether typecheck actually
   checks X).
7. Probed AST-hash → quote-handle-table flow for collision risks
   from the "Unknown" fallback arm.
8. Exercised the cycle-4 fixes via direct module invocations.

**Result**: **10 new findings (0 CRITICAL, 4 HIGH, 4 MEDIUM, 2 LOW)**
— Cycle 5 silent-failure NOT clean. The dominant pattern is
**cycle-4-fix-sweep-skipped findings** combined with **fix-
introduced regressions**: the cycle-4 fix-sweep silently dropped
three cycle-4 audit findings (C4-3, C4-4, C4-7) and took the wrong
direction on cycle-4 audit C4-1 (broadened instead of reverted the
D2 tag-12 polarity). On top of that, new silent-failure patterns
discovered in stages 1-28.7 outside the cycle-4 fix-sweep:
ast_hash.py's "Unknown" fallback collapses distinct
StructLit/TileLit/Path/Continue/UnsafeBlock/Return/Break/Modify
instances to a single hash; monomorphize.py's `_walk_subst_expr`
has no TileLit arm (parallel to cycle-5 type-design F4's gap in
`_inline_lets`); monomorphize.py's arity-mismatch arm at line 484
makes a false claim that "typecheck will flag" turbofish arity
errors (it doesn't); lower_ast.py contains silent `const_int(0)`
and silent-None-return fallbacks.

## Summary table

| ID | Severity | Component | Issue (short) |
|----|----------|-----------|---------------|
| C5-1 | HIGH | parser.hx:2334-2374 (D2 tag-12 broadening) | Cycle-4 fix-sweep BROADENED the cycle-4-audit C4-1 functional regression. `let a = 10 + 5; let c = |x| x + a; c(5)` now SIGILLs — every non-literal-non-AST_VAR-non-comparison RHS now traps 76003. The cycle-4 audit recommended REVERT; fix-sweep broadened instead. |
| C5-2 | HIGH | typecheck.py:2197-2249 (`_compatible` shape compare) | Cycle-4 commit-C4-4 + E1 added structural arms for TyTensor / TyTile / TyArray-size but the shape-element compare still cascades to `_compatible` which lacks a TyVar / TySize defer arm. Generic-shape call boundaries (TySize('N') vs TySize('M'), or TySize('N') vs TyPrim('size_3')) still false-positive. |
| C5-3 | HIGH | monomorphize.py:412-438 (`Monomorphizer.run` iteration) | Cycle-4 audit C4-4 (D9 paper-only) was NOT addressed at HEAD. The while-loop iterates `prog.items` not `instantiated`; clones added during iteration are appended AFTER the loop and never walked for nested turbofish. `caller__i32` ships referencing `id__U` with U unresolved. |
| C5-4 | HIGH | ast_hash.py:238-241 (`_hash_into` Unknown fallback) | StructLit / TileLit / Path / Continue / UnsafeBlock / Return / Break / Modify / Splice all fall through to `_emit("Unknown", class_name)`. Two distinct instances of the same unhandled class hash IDENTICALLY → quote-handle-table at lower_ast.py:2118 silently aliases distinct `quote(...)` expressions when their inner contains any of these subtypes. |
| C5-5 | MEDIUM | monomorphize.py:276-383 (`_walk_subst_expr`) | TileLit has no arm in the type-substitution walker; falls through to identity `return e`. Generic-shape TileLit like `tile<f32, [N], REG>::zeros()` mono'd with N=32 leaves `shape=[Name("N")]` un-substituted → downstream lower_ast.py defaults shape to 0. Parallel finding to cycle-5 type-design F4 (same gap in `_inline_lets`). |
| C5-6 | MEDIUM | autodiff.py:709-710 (`_inline_lets` TileLit identity arm) | The cycle-4 commit-C4-1 TileLit identity arm silently drops let-bindings in TileLit.shape and TileLit.memspace (Expr-typed children). False-leaf classification — Path / Continue are genuine leaves; TileLit isn't. Same area as cycle-5 type-design F4 but the silent-failure lens flags it as a positive drop (not just an incomplete walk). |
| C5-7 | MEDIUM | check.py:272-292 (outer `main` wrapper) | Cycle-4 audit C4-7 (C3-3 `except Exception` over-broad) was NOT fixed by cycle-4 sweep. File-not-found / encoding / import errors still mis-attributed as "compiler bug — please file an issue". |
| C5-8 | MEDIUM | monomorphize.py:484-486 ("typecheck will flag" comment) | Turbofish arity mismatch (`id::<i32, f32>(x)` where `fn id[T]`) silently passes through `_rewrite_calls_in_expr`. The comment claims "typecheck will flag" but typecheck.py has no turbofish-arity check at call sites. Reaches codegen as unmangled `id` call with non-empty turbofish. |
| C5-9 | LOW | monomorphize.py:691-706 (`monomorphize_safe` narrow catch + driver continues) | `monomorphize_safe` catches only ShapeFoldError; KeyError / AttributeError / TypeError still propagate to check.py:274 broad-except. Additionally x86_64.py:3025-3029 driver prints `warning:` and continues with a half-mutated prog (some `item.body` mutations applied; clones never appended) — silent miscompile window. |
| C5-10 | LOW | lower_ast.py:2113-2117 (Quote handle structural_hash fallback) + 2093-2101 (Cast None→0) + 2079-2092 (Field None-return) | Multiple silent-fallback patterns: `try: structural_hash(...) except Exception: key = _pretty(...)` silently downgrades keying when hashing raises; `if inner is None: inner = self.builder.const_int(0)` silently substitutes zero in Cast; Field-no-array-match path silently returns None. |

---

## CRITICAL FINDINGS

(none)

---

## HIGH FINDINGS

### Finding C5-1: cycle-4 fix-sweep BROADENS the cycle-4-audit C4-1 functional regression — `let a = 10 + 5; let c = |x| x + a; c(5)` (and every Binary / Unary / Index / Field / If / Match / Block / UnsafeBlock RHS of an i32-valued let) now SIGILLs at runtime

**File**:
- `helixc/bootstrap/parser.hx:2334-2374` (the cycle-4 commit C4-2
  catch-all `else { inferred_ty_tag = 12; }`)
- `helixc/bootstrap/parser.hx:1819` (capture-site guard `cap_ty_tag > 0`)
**Severity**: HIGH
**Category**: cycle-4-fix-introduced functional regression
(BROADENING of the still-open cycle-4-audit C4-1 CRITICAL)

**Description**:
The cycle-4 silent-failures audit doc C4-1 finding was a **CRITICAL**
regression introduced by cycle-3 D2: the parser tagged ALL Call-RHS
untyped lets as non-i32 (tag 12), so
`let pi = make_i32(); let c = |x| x + pi; c(5)` SIGILLed at runtime
even though `make_i32() -> i32`.

The cycle-4-audit's recommendation for C4-1 was clear:
> 1. REVERT the D2 fix. The bootstrap parser cannot infer return
>    types without a typechecker, and pre-fix tag -1 (untracked,
>    pass cleanly) was the right behavior for the dominant pattern.

The cycle-4 fix-sweep took the **opposite direction**. Instead of
reverting, it **broadened** the tag-12 sentinel from the single
`val_tag == 16` (Call) arm to a catch-all `else`-branch covering
every val_tag EXCEPT a small whitelist:

```hx
} else { if val_tag == 1 {
    // AST_VAR — defer to var_type_tab resolution.
} else { if val_tag == 6 { inferred_ty_tag = 0; ...   // AST_LT bool
} else { if val_tag == 19 { inferred_ty_tag = 0; ...  // AST_GT bool
} else { if val_tag == 20 { ... } else { ...          // AST_EQ-GE bool
} else {
    inferred_ty_tag = 12;     // sentinel: untracked-complex
};...};
```

So now **every** non-literal RHS except AST_VAR (which has its own
silent-fail per cycle-5 type-design F6) and comparison-bool gets
sentinel 12. The capture-site guard `cap_ty_tag > 0` triggers trap
76003 (SIGILL) for every i32-valued let RHS that isn't a literal:

```helix
fn main() -> i32 {
    let a = 10 + 5;             // Binary(IntLit, +, IntLit) — tag 12
    let c = |x: i32| x + a;     // capture site: tag 12 > 0 → trap 76003
    c(5)                        // SIGILL at runtime
}
```

The pre-fix behavior was: tag -1 (untracked) for i32-Binary lets,
silent pass through capture, correct codegen → 20.
Post-fix-sweep: tag 12 → trap 76003 → SIGILL.

**Hidden errors**:
- Every closure capturing a let-bound value computed from any
  arithmetic / index / field / branch expression silently SIGILLs.
- This is the **dominant** pattern in real Helix programs.
- The cycle-4 fix-sweep commit message claims this was the
  resolution for cycle-4-audit C4-6 ("Binary/Unary/Index/...
  RHS still silently leave inferred_ty_tag = -1"); but cycle-4
  C4-6's recommendation was to revert D2's polarity, not broaden.
  The fix-sweep went the wrong direction.

**Recommended fix**:
1. REVERT both the cycle-3 D2 and the cycle-4 commit-C4-2
   broadening. Restore the pre-D2 behavior of tag -1 (untracked,
   silent pass) for all non-trivially-i32-literal RHS.
2. The proper fix for the original closure-capture corruption
   requires a real local-typecheck pass in the bootstrap parser
   OR an explicit annotation requirement for closure-captured
   lets — both deferred to Stage 29+.
3. Regression test that codifies the correct behavior:
   `let a = 10 + 5; let c = |x: i32| x + a; c(5)` returns 20
   cleanly (exit 0, no SIGILL).

**Trap-id**: 76003 (existing — over-fires).

---

### Finding C5-2: cycle-4 commit-C4-4 + E1 `_compatible` structural arms are paper-only — shape-element compare cascades to `_compatible` which lacks a TyVar / TySize defer arm

**File**:
- `helixc/frontend/typecheck.py:2197-2249` (`_compatible` TyArray,
  TyTensor, TyTile structural arms post-fix-sweep)
- `helixc/frontend/typecheck.py:2157-2158` (top of `_compatible` —
  no TyVar / TySize cascade arm)
**Severity**: HIGH
**Category**: cycle-4-fix paper-only at the headline reproducer.
Same root cause closes cycle-4-audit C4-2 + C4-3 + cycle-5
type-design F1.

**Description**:
The cycle-4 fix-sweep added the long-deferred TyTile / TyTensor
structural arms to `_compatible` and added a disjunctive size-
compare to TyArray (`a.size == b.size or self._compatible(a.size,
b.size)`). The commit message claims this closes the cycle-4-audit
C4-2 finding ("generic tensor/tile call false-positive") and the
type-design E1 finding.

Trace the cycle-4-audit C4-2 reproducer through the post-fix
`_compatible`:
```helix
@pure fn norm[N: size](x: tensor<f32, [N]>) -> f32 { 0.0_f32 }
@pure fn use_norm(m: tensor<f32, [3]>) -> f32 { norm(m) }
```

Call boundary check at typecheck.py:736-742 fires:
- pty = TyTensor(dtype=f32, shape=[TySize('N')])
- aty = TyTensor(dtype=f32, shape=[TyPrim('size_3')])
- Neither TyUnknown nor TyVar (TyTensor) → elif fires
- `_compatible(pty, aty)`:
  - new arm at 2230-2237: TyTensor-TyTensor → compare shapes via
    `_compatible(TySize('N'), TyPrim('size_3'))`
  - `_compatible(TySize('N'), TyPrim('size_3'))`:
    - top guard at 2158: neither TyUnknown → continue
    - all composite arms skip → falls through to line 2249
      `return a == b` → False (frozen dataclass mismatch)
  - Returns False
- elif emits `arg 'x' expects tensor<f32, [size:N]>, got
  tensor<f32, [size_3]>`

The cycle-4 audit's headline reproducer **still false-positives**.
The fix-sweep added the outer structural arm but not the inner
TyVar / TySize defer.

The cycle-4-audit C4-3 (`fn g[T](v: T) -> i32 { f(v) }` with
`fn f(i: i32)`) is unaddressed for the same root cause: the elif
at line 736 filters TyVar / TySize / TyUnknown on pty only — when
aty is TyVar('T') the elif fires, `_compatible(TyPrim('i32'),
TyVar('T'))` falls through to `a == b` → False → false-positive.

**Hidden errors**:
- Every generic-array / generic-tensor / generic-tile call still
  false-positives.
- The cycle-4 fix-sweep commit message lists C4-4 + E1 as CLOSED;
  a future contributor following the commit log will assume these
  cases work.
- Cycle-5 type-design F1 covers the parallel cascade-arm gap; both
  findings have the same root cause and same single-fix solution.

**Recommended fix**:
1. Add a `TyVar` / `TySize` defer arm at the TOP of `_compatible`,
   mirroring the existing TyUnknown short-circuit:
   ```python
   if isinstance(a, TyUnknown) or isinstance(b, TyUnknown):
       return True
   if isinstance(a, (TyVar, TySize)) or isinstance(b, (TyVar, TySize)):
       return True   # defer to mono / cascade-safe
   ```
   This single change closes cycle-4-audit C4-2, C4-3, E1, AND
   cycle-5 type-design F1.
2. Regression test:
   `tc._compatible(TyArray(elem=f32, size=TySize('N')),
   TyArray(elem=f32, size=TyPrim('size_4')))` returns True.
3. Surface regression: `python -m helixc.check --strict` on the
   `norm` reproducer returns `typecheck: OK`.

**Trap-id**: n/a.

---

### Finding C5-3: cycle-4 audit C4-4 (D9 paper-only) NOT addressed in fix-sweep — `Monomorphizer.run` iteration order leaves clones referencing unresolved generic-param names

**File**: `helixc/frontend/monomorphize.py:412-438`
(`Monomorphizer.run`).
**Severity**: HIGH
**Category**: cycle-4 audit finding silently dropped by fix-sweep

**Description**:
The cycle-3 D9 fix added a `_walk_subst_expr.Call` arm that
substitutes the callee's `generics` list. The cycle-4 audit
(silent-failures C4-4) showed the fix is paper-only at the end-
to-end pipeline level: `Monomorphizer.run` iterates `prog.items`
inside the while-loop and only appends `self.instantiated` clones
AFTER the loop exits:

```python
def run(self) -> int:
    ...
    changed = True
    while changed:
        changed = False
        for item in list(self.prog.items):
            if isinstance(item, A.FnDecl):
                new_body = self._rewrite_calls_in_block(item.body, item)
                if new_body is not item.body:
                    item.body = new_body
                    changed = True
    # Append instantiated clones AFTER the loop — clones never iterate
    added = len(self.instantiated)
    self.prog.items = list(self.prog.items) + list(self.instantiated.values())
    return added
```

Trace `fn id[T](x: T); fn caller[U](v: U) { id::<U>(v) }; fn main() { caller::<i32>(7) }`:

1. **Iter 1**: caller's body has `id::<U>(v)`. `_rewrite_calls_in_expr`
   mangles to `id__U`, clones `id_clone(body: x: U → U)`, rewrites
   caller's body to `id__U(v)` (turbofish cleared). main's body has
   `caller::<i32>(7)` — mangled to `caller__i32`, clone with
   `subst={U: i32}` walks caller's (already-rewritten) body and
   sees `Call(Name('id__U', generics=[]))`. D9 fix checks
   `if new_callee.generics:` — empty list, so no substitution.
2. **End of while**: clones appended; `caller__i32` calls `id__U`;
   `id__U`'s body is `(x: U) -> U` with U unbound.

Codegen sees a call to a function with an unresolved generic-param
name. Either silent miscompile or downstream error referencing the
mangled name without a trap-id context.

The cycle-4 fix-sweep commit message explicitly states "D9 already
done in cycle 3" — it didn't re-examine the iteration order. The
cycle-4 audit doc's "Recommended fix sequence for cycle 5" listed
this as priority #2 (HIGH — revert D9 or fix iteration order). The
fix-sweep silently skipped it.

**Hidden errors**:
- Every nested-generic call pattern produces broken mono'd code.
- Codegen sees calls to functions with unresolved generic params.
- The D9 unit regression test only exercises `_walk_subst_expr`
  directly with a hand-built Call that has intact turbofish; it
  does NOT run the end-to-end pipeline.
- Cycle-5 type-design audit (sibling) didn't re-flag because its
  scope was the cycle-4 fix-sweep's *new* code paths, not the
  unchanged old C4-4 finding.

**Recommended fix**:
Per the cycle-4 audit's recommendation:
1. Include clones in subsequent iterations:
   ```python
   while changed:
       changed = False
       for item in (list(self.prog.items)
                    + list(self.instantiated.values())):
           ...
   ```
2. Or rewrite `_rewrite_calls_in_block` to skip generic-template
   fns entirely (only instantiated clones get rewritten; generic
   templates stay pristine).
3. End-to-end regression test:
   ```python
   def test_d9_end_to_end_nested_turbofish():
       src = '''fn id[T](x: T) -> T { x }
                fn caller[U](v: U) -> U { id::<U>(v) }
                fn main() -> i32 { caller::<i32>(7) }'''
       prog = parse(src)
       monomorphize(prog)
       # assert NO Call references id__U; caller__i32 calls id__i32.
   ```

**Trap-id**: n/a.

---

### Finding C5-4: `ast_hash.py` "Unknown" fallback collapses distinct StructLit / TileLit / Path / Continue / UnsafeBlock / Return / Break / Modify / Splice instances → silent quote-handle-table aliasing in lower_ast.py

**File**:
- `helixc/frontend/ast_hash.py:238-241` (`_hash_into` unenumerated
  fallback)
- `helixc/ir/lower_ast.py:2113-2127` (Quote handle table key
  derivation)
**Severity**: HIGH
**Category**: dispatch-walker incomplete coverage → silent
collision in hash-keyed table

**Description**:
`_hash_into` in `ast_hash.py` enumerates ~25 Expr/Block/FnDecl
subtypes. The final arm at line 241:

```python
# Fallback for anything we haven't enumerated: hash the type name.
# This is conservative: it prevents collisions but two different
# instances of the same class share a hash.
_emit(h, "Unknown", type(node).__name__)
```

The comment's claim that this "prevents collisions" is misleading:
two different instances of the same unhandled class share the
same hash. Subtypes that fall through to this arm:

- **A.StructLit** (no arm — line 196 has ArrayLit but not StructLit)
- **A.TileLit** (no arm)
- **A.Path** (no arm)
- **A.Continue** (no arm)
- **A.UnsafeBlock** (no arm)
- **A.Return** (no arm)
- **A.Break** (no arm)
- **A.Modify** (no arm)
- **A.Splice** (no arm — Quote has an arm but Splice doesn't)

`helixc/frontend/hash_cons.py:_SHAREABLE` restricts dedup to a safe
subset that excludes most of these, AND `_ast_equal` has a
defensive post-hash equality check (raises HashConsError trap
20001 on a false positive). So `hash_cons` is protected.

But the **Quote handle table** in `lower_ast.py:2113-2127` uses
`structural_hash` DIRECTLY as a dict key with NO post-hash equality
verification:

```python
try:
    key = structural_hash(expr.inner)
except Exception:
    key = _pretty(expr.inner)
if key not in self._quote_handle_table:
    idx = len(self._quote_handle_table)
    ...
    self._quote_handle_table[key] = idx
ast_handle = self._quote_handle_table[key]
```

When two distinct `quote(StructLit{x:1})` and
`quote(StructLit{x:99})` both hash to the same value (both fall
through to `Quote → Unknown StructLit`), the second reuses the
first's handle index. The cells backing `quote()` become silently
aliased — accessing one Quote's cell reads the other's data.

**Reproducer** (Python-level):
```python
from helixc.frontend.ast_hash import structural_hash
from helixc.frontend import ast_nodes as A
sp = A.Span(0, 0)
sl1 = A.StructLit(span=sp, name='Foo', fields=[('x', A.IntLit(span=sp, value=1))])
sl2 = A.StructLit(span=sp, name='Foo', fields=[('x', A.IntLit(span=sp, value=99))])
print(structural_hash(sl1) == structural_hash(sl2))  # True — COLLISION
```

Source-level (Phase-0 Quote support is minimal but exercised by
AGI-primitive stdlib code):
```helix
fn ai() -> i32 {
    let q1 = quote { Foo { x: 1 } };
    let q2 = quote { Foo { x: 99 } };
    use_quote(q1, q2);     // both quotes resolve to the same cell
    0
}
```

**Hidden errors**:
- Distinct `quote(...)` ASTs containing StructLit / TileLit / Path /
  Continue / UnsafeBlock / Return / Break / Modify / Splice
  silently alias.
- Surfaces only when user code accesses the cell contents —
  silent miscompile hard to attribute.
- `_emit("Unknown", class_name)` is documented as defensive, but its
  consumer (quote handle table) breaks the defensive contract.

**Recommended fix**:
1. Add explicit arms in `_hash_into` for every Expr/Stmt subtype:
   - StructLit: name + per-field (name, value-hash)
   - TileLit: dtype-repr + shape-hashes + memspace-hash + init
   - Path: segments tuple
   - Continue: marker only (truly leaf)
   - UnsafeBlock: body-hash
   - Return / Break: value-hash (or None marker)
   - Modify: target + transformation + verifier hashes
   - Splice: inner-hash
2. Replace `_emit("Unknown", ...)` fallback with
   `raise NotImplementedError(f"ast_hash: unhandled {type(node).__name__}")`
   so future AST extensions surface loudly.
3. In `lower_ast.py:2113-2127`, add a post-hash equality check
   parallel to `hash_cons._ast_equal`:
   ```python
   key = structural_hash(expr.inner)
   if key in self._quote_handle_table:
       prior_node = self._quote_inner_by_handle[self._quote_handle_table[key]]
       if not _ast_equal(prior_node, expr.inner):
           raise HashConsError("quote-handle hash collision (trap 20001)")
   ```
4. Regression test: distinct StructLit/TileLit quotes get distinct
   handles.

**Trap-id**: 20001 (existing hash-collision; reuse for the
quote-handle aliasing path).

---

## MEDIUM FINDINGS

### Finding C5-5: `monomorphize.py:_walk_subst_expr` has no TileLit arm — generic-shape TileLit silently un-substituted

**File**: `helixc/frontend/monomorphize.py:276-383`
(`_walk_subst_expr`).
**Severity**: MEDIUM
**Category**: dispatch-walker incomplete coverage in the type-
substitution walker; parallel to cycle-5 type-design F4 (same
gap in the autodiff `_inline_lets` walker)

**Description**:
`_walk_subst_expr` enumerates 24 Expr subtypes (Cast, Block, If,
Match, For, While, Loop, Binary, Unary, Call, Index, Field,
TupleLit, ArrayLit, StructLit, Assign, Return, Break, Range, Quote,
Splice, Modify, UnsafeBlock). It does NOT have an arm for
**A.TileLit**. Falls through to the final `return e` at line 383
(silent identity).

Cycle-5 type-design F4 documented the parallel gap in
`_inline_lets` (autodiff.py:709-710 — TileLit identity arm doesn't
walk shape / memspace). The same gap exists in the type-
substitution walker used during fn-mono: when a generic fn body
contains a TileLit with a size-generic in its shape, mono fails to
substitute the shape's Name leaves.

**Reproducer**:
```python
from helixc.frontend.monomorphize import _walk_subst_expr
from helixc.frontend import ast_nodes as A
sp = A.Span(0, 0)
tl = A.TileLit(
    span=sp, dtype=A.TyName(span=sp, name='f32'),
    shape=[A.Name(span=sp, name='N', generics=[])],
    memspace=A.Name(span=sp, name='REG', generics=[]),
    init='zeros',
)
out = _walk_subst_expr(tl, {'N': A.TyName(span=sp, name='size_32')})
# out IS tl (identity); shape[0] is Name('N') — should be IntLit(32)
```

Source-level:
```helix
fn zeros_tile[N: size]() -> tile<f32, [N], REG> {
    tile<f32, [N], REG>::zeros()
}
fn main() -> i32 {
    let t = zeros_tile::<32>();
    0
}
```

Post-mono, `zeros_tile__32`'s body contains `TileLit(shape=[Name("N")], ...)`
— N is the original generic-param name, never substituted to
IntLit(32). Downstream lower_ast.py defaults the unrecognized
shape to 0 and silent-allocates a zero-element tile.

**Hidden errors**:
- Every generic-shape TileLit mono produces a tile of wrong size.
- Lowering silently defaults the un-substituted Name to 0.
- Phase-0 TileLit is restricted to REG memspace and small shapes,
  so impact is narrow today, but the missing arm is an obvious
  soundness hole that future tile-IR work will trip over.

**Recommended fix**:
1. Add an explicit TileLit arm in `_walk_subst_expr` parallel to
   ArrayLit:
   ```python
   if isinstance(e, A.TileLit):
       return A.TileLit(
           span=e.span,
           dtype=substitute_ty(e.dtype, subst),
           shape=[_subst_shape_expr(s, subst) for s in e.shape],
           memspace=_walk_subst_expr(e.memspace, subst),
           init=e.init,
       )
   ```
2. Audit every walker in helixc/frontend/{monomorphize.py,
   autodiff.py, grad_pass.py, hash_cons.py, ast_hash.py} for the
   full Expr taxonomy; mark unhandled cases with `raise
   NotImplementedError` instead of silent identity.

**Trap-id**: n/a.

---

### Finding C5-6: `_inline_lets` cycle-4 commit-C4-1 TileLit identity arm silently drops let-bindings in tile shape / memspace

**File**:
- `helixc/frontend/autodiff.py:709-710` (TileLit identity arm)
- `helixc/frontend/ast_nodes.py:344-356` (A.TileLit definition)
**Severity**: MEDIUM
**Category**: cycle-4-fix-introduced silent drop (false-leaf
classification). Same area as cycle-5 type-design F4 but the
silent-failure lens flags the positive drop, not the incomplete
walk.

**Description**:
The cycle-4 fix-sweep added three "leaf-like" identity arms to
`_inline_lets`:

```python
if isinstance(expr, A.Path):
    return expr
if isinstance(expr, A.Continue):
    return expr
if isinstance(expr, A.TileLit):
    return expr
```

`A.Path` and `A.Continue` are genuinely leaf-like — no Expr
children. But `A.TileLit` is **not a leaf**:

```python
@dataclass
class TileLit(Expr):
    dtype: "TyNode"
    shape: list["Expr"]
    memspace: "Expr"
    init: str
```

Both `shape` and `memspace` are Expr-typed. The identity arm
silently drops let-bindings in either.

**Reproducer**:
```python
n = A.Name(span=span, name='N', generics=[])
tl = A.TileLit(span=span, dtype=ty_f32, shape=[n],
               memspace=A.Name(span=span, name='REG', generics=[]),
               init='zeros')
env = {'N': A.IntLit(span=span, value=8)}
r = _inline_lets(tl, env)
# Expected: r.shape[0] == IntLit(8)
# Actual: r.shape[0] == Name('N')  (identity returned, env ignored)
```

Pre-fix-sweep: A.TileLit fell through to the catch-all `_ad_warn` —
loud (over-fires per cycle-4-audit C4-5), but not silent.
Post-fix-sweep: silent identity. The catch-all is suppressed; any
let-binding inside the tile literal's shape/memspace is lost.

**Recommended fix**:
1. Replace the TileLit identity arm with a recursive walk arm:
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
2. Move the TileLit arm into the recursion block (alongside Cast,
   Call, Field, etc.), keeping only Path / Continue (genuinely
   leaf-like) in the post-recursion identity block before the
   catch-all.
3. Same arm should be added in `_walk_subst_expr` (C5-5 above).

**Trap-id**: n/a (silent let-binding drop).

---

### Finding C5-7: cycle-4 audit C4-7 (C3-3 `except Exception` over-broad in check.py main wrapper) was NOT fixed by cycle-4 sweep

**File**: `helixc/check.py:272-292` (outer `main` wrapper).
**Severity**: MEDIUM
**Category**: cycle-4 audit finding silently dropped by fix-sweep

**Description**:
Current code:
```python
try:
    rc = _main_inner(argv, a_holder)
except Exception as e:
    print(f"helixc: internal error: {type(e).__name__}: {e}", file=sys.stderr)
    print("helixc: this is a compiler bug — please file an issue.", file=sys.stderr)
    rc = 1
```

The cycle-4 audit identified four classes of user-env errors that
get mis-attributed:
1. TOCTOU on file open (FileNotFoundError between exists() and open()).
2. UnicodeDecodeError on non-UTF-8 source.
3. ImportError on missing pipeline module.
4. Drain-time errors in the `finally` block masking the original.

None are addressed at HEAD 960303b. The fix-sweep commit message
does not mention C4-7.

**Hidden errors**:
- Users see "compiler bug — please file an issue" for their own
  environmental errors → noisy bug reports.
- The "compiler bug" message dominates any hard exit path,
  eroding its signal value.

**Recommended fix**:
(per cycle-4 audit doc lines 644-668 — narrow the catch to
internal-error classes; let user-env errors propagate to a
distinct catch with a clean message).

**Trap-id**: n/a.

---

### Finding C5-8: `monomorphize.py:484-486` arity-mismatch arm claims "typecheck will flag" but typecheck has no turbofish-arity check; silent pass-through

**File**: `helixc/frontend/monomorphize.py:484-486` (arity mismatch
arm in `_rewrite_calls_in_expr`).
**Severity**: MEDIUM
**Category**: silent pass-through with false claim about downstream
coverage

**Description**:
```python
if isinstance(new_callee, A.Name) and new_callee.generics and new_callee.name in self.generic_fns:
    fn = self.generic_fns[new_callee.name]
    if len(new_callee.generics) != len(fn.generics):
        # Mismatch — leave alone, typecheck will flag
        return A.Call(span=e.span, callee=new_callee, args=new_args)
```

The comment "typecheck will flag" is false. `helixc/frontend/typecheck.py`
has no check for turbofish-arity mismatch at function call sites
— `grep "len.*generics", "arity.*generic", "turbofish"` in
typecheck.py returns only the struct-arity check at line 554-566,
NOT a function arity check at the call boundary.

A call like `id::<i32, f32>(x)` where `fn id[T](x: T)` (1 generic
param, 2 turbofish args) silently passes through
`_rewrite_calls_in_expr` and through typecheck, reaching codegen
with the original generic-fn name (no mangling happened, no clone
exists). Codegen sees a call to `id` with non-empty turbofish args
— likely silently lowers as a non-generic call.

**Reproducer**:
```helix
fn id[T](x: T) -> T { x }
fn main() -> i32 {
    id::<i32, f32>(42)   // 2 turbofish args vs 1 generic param
}
```

Expected: typecheck error "id expects 1 generic arg, got 2".
Actual at HEAD: no typecheck error. Mono silently skips. Codegen
attempts to lower `id::<i32, f32>(42)` — Phase-0 codegen likely
fails at link time, or produces a binary that mis-routes.

**Hidden errors**:
- Turbofish arity errors are NEVER caught at the surface tool.
- Users see opaque link-time / runtime errors instead of clean
  typecheck diagnostics.
- The comment misleads future maintainers into thinking the
  arity check exists elsewhere.

**Recommended fix**:
1. Either move the arity check into typecheck (where the comment
   claims it exists) with a new trap id 28804:
   ```python
   # typecheck.py, in _check_expr's Call arm:
   if isinstance(expr.callee, A.Name) and expr.callee.generics:
       fn_decl = self._lookup_fn_decl(expr.callee.name)
       if fn_decl is not None and len(expr.callee.generics) != len(fn_decl.generics):
           self.errors.append(TypeError_(
               f"call to {expr.callee.name!r}: expected "
               f"{len(fn_decl.generics)} generic arg(s), got "
               f"{len(expr.callee.generics)} (trap 28804)",
               expr.span,
           ))
   ```
2. Or surface the arity mismatch from mono via a `(prog, diags)`
   tuple parallel to struct_mono.
3. Regression test: arity-mismatch produces a clean typecheck
   error at the surface tool.

**Trap-id**: n/a today; recommend reserving 28804.

---

## LOW FINDINGS

### Finding C5-9: `monomorphize_safe` narrow catch + x86_64.py driver continues with half-mutated prog — silent miscompile window

**File**:
- `helixc/frontend/monomorphize.py:691-706` (`monomorphize_safe` wrapper)
- `helixc/frontend/monomorphize.py:412-438` (`Monomorphizer.run`
  mutations at line 432 before post-loop append at line 437)
- `helixc/backend/x86_64.py:3025-3029` (driver caller — warning,
  not error)
**Severity**: LOW
**Category**: cycle-4-fix-introduced silent-failure window + narrow
catch incompleteness

**Description**:
The cycle-4 fix-sweep added `monomorphize_safe` to catch
`ShapeFoldError`. Two compounding issues:

1. **Narrow catch**: only catches `ShapeFoldError`. Other
   exception classes (KeyError on missing generic param,
   AttributeError on malformed FnDecl, TypeError in
   `substitute_ty`, RecursionError on deep nesting) still propagate
   uncaught to check.py:274's broad-except → "compiler bug" mis-
   attribution. The wrapper's stated goal (clean trap-28801
   diagnostic) is only met for the ShapeFoldError sub-case.

2. **Driver continues**: x86_64.py at 3025-3029:
   ```python
   mono_count, mono_diags = monomorphize_safe(prog)
   for d in mono_diags:
       print(f"warning: fn-mono: {d}", file=sys.stderr)
   if mono_count > 0:
       print(f"mono: {mono_count} generic instantiation(s)", ...)
   grad_count = grad_pass(prog)
   ... typecheck(prog) ... codegen ...
   ```
   Prints `warning:` (not `error:`) and continues. When a
   `ShapeFoldError` raises mid-`Monomorphizer.run`:
   - Some prior iterations may have already `item.body = new_body`-
     mutated items processed before the failing item.
   - `self.instantiated` accumulated clones for keys that completed.
   - The raise occurs BEFORE the post-loop
     `prog.items = list(prog.items) + list(self.instantiated.values())`.
   - The wrapper catches and returns `(0, [str(e)])`.
   - The resulting `prog` references mangled names in mutated
     `item.body` fields but the clones for those mangled names are
     MISSING from `prog.items`.
   - Pipeline continues with broken intermediate prog state.

This finding overlaps with cycle-5 type-design F3 (severity
downgrade angle) — F3 covers the warning-vs-error driver behavior;
this C5-9 also covers the wrapper's narrow catch.

**Hidden errors**:
- Future-reachable silent miscompile when fn-mono shape arithmetic
  becomes parser-reachable.
- `mono_count = 0` is misreported on raise (the wrapper returns 0
  but partial mutations persisted).
- Non-ShapeFoldError exceptions still get mis-attributed.

**Recommended fix**:
1. Widen wrapper catch:
   ```python
   def monomorphize_safe(prog: A.Program) -> tuple[int, list[str]]:
       try:
           return monomorphize(prog), []
       except ShapeFoldError as e:
           return 0, [str(e)]
       except (KeyError, ValueError, TypeError) as e:
           return 0, [f"fn-mono internal: {type(e).__name__}: {e}"]
   ```
2. Driver aborts on non-empty diags:
   ```python
   if mono_diags:
       for d in mono_diags:
           print(f"error: fn-mono: {d}", file=sys.stderr)
       sys.exit(1)
   ```
3. Or rewrite `Monomorphizer.run` to catch per-instance (parallel
   to struct_mono's pattern).

**Trap-id**: 28801 (existing — should fire at error severity, not
warning).

---

### Finding C5-10: `lower_ast.py` contains multiple silent-fallback patterns: broad-except → _pretty for quote-hash; `inner=const_int(0)` for Cast None-inner; Field-no-array-match returns None

**File**:
- `helixc/ir/lower_ast.py:2113-2117` (Quote handle structural_hash
  fallback)
- `helixc/ir/lower_ast.py:2093-2101` (Cast lowering inner-None
  fallback to const_int(0))
- `helixc/ir/lower_ast.py:2079-2092` (Field-no-array-match silent
  None-return after side-effect `_lower_expr(expr.obj)`)
**Severity**: LOW
**Category**: silent-fallback / silent-None-return cluster in
lower_ast.py

**Description**:
Three related silent-fallback patterns:

**Pattern A** (line 2113-2117):
```python
try:
    key = structural_hash(expr.inner)
except Exception:
    # Fall back to pretty-string if hashing fails for any reason.
    key = _pretty(expr.inner)
```
Broad-except catches anything — including programmer bugs in
`structural_hash` (a future refactor breakage gets silently
downgraded to `_pretty` keying instead of raising). The two key
spaces (hex SHA-256 vs multi-line `_pretty` strings) share the
same dict; could theoretically alias.

**Pattern B** (line 2093-2101 in the Cast arm):
```python
if isinstance(expr, A.Cast):
    inner = self._lower_expr(expr.value)
    if inner is None:
        inner = self.builder.const_int(0)
    target = self._lower_type(expr.target_ty)
    return self.builder.emit(tir.OpKind.CAST, inner, ...)
```
When `_lower_expr(expr.value)` returns None (unrecognized expr
type, or a Field hitting the None-return path), the Cast silently
substitutes `0`. `(some_complex_expr as f64)` becomes `(0 as f64)`
→ `0.0` with no diagnostic.

**Pattern C** (line 2079-2092 in the Field arm):
```python
if (len(path_segs) == 1 and path_segs[0].isdigit()
        and struct_name is None):
    arr = self._lookup_array(base_name)
    if arr is not None:
        ...
        return self.builder.emit(tir.OpKind.LOAD_ELEM, ...)
# fallthrough:
self._lower_expr(expr.obj)
return None
```
When Field doesn't match any struct-path or tuple-index handling,
silently calls `_lower_expr(expr.obj)` for side effect and returns
None. The caller (often _lower_expr recursively in a Cast wrapper
context) gets None, triggers Pattern B, becomes 0 silently.

The three patterns compose: a Field on an unknown type wrapped in
a Cast silently becomes `(0 as target_ty)` with `expr.obj`'s side
effects fired and no diagnostic anywhere.

**Hidden errors**:
- Silent-zero substitution for unknown expr types in Cast.
- A structural_hash regression bug silently downgrades to _pretty
  keying.
- Field access on an unknown type silently becomes 0 via Cast
  composition.
- The diagnostic surfaces only at runtime as wrong-result values.

**Recommended fix**:
1. Pattern A: narrow except to specific recoverable exceptions OR
   raise loudly:
   ```python
   try:
       key = structural_hash(expr.inner)
   except Exception as exc:
       raise CompilerError(
           f"helixc: structural_hash failed on quote(...): {exc}"
       ) from exc
   ```
2. Pattern B: raise instead of falling back:
   ```python
   if inner is None:
       raise CompilerError(
           f"Cast: inner expr {type(expr.value).__name__} lowered "
           f"to None at {expr.span.line}:{expr.span.col}"
       )
   ```
3. Pattern C: return the obj's lowering result, OR raise, OR
   diagnose. The current side-effect-fire-then-None pattern is
   actively misleading.
4. Sweep lower_ast.py for the broader `if x is None: x =
   const_int(0)` pattern (~10 sites) — each is a candidate
   silent-zero substitution.

**Trap-id**: n/a today; recommend reserving 16005 for "lowering
returned None where a value was required" if these fixes get
adopted as raises.

---

## Cycle 4 fix-sweep re-verification

| fix-sweep label | What changed | Audit-doc cross-ref | C5 verdict |
|---|---|---|---|
| commit-C4-1 | `_inline_lets` Path/Continue/TileLit identity arms | audit-C4-5 + E7 | C5-6 (TileLit silently drops shape/memspace; not a leaf). Path/Continue OK. |
| commit-C4-2 | parser.hx tag-12 broadened to all complex RHS | (cycle-4-audit recommended REVERT D2, not broaden) | **C5-1** (BROADENS the still-open audit-C4-1 critical regression) |
| commit-C4-3 | `_inline_lets` If.cond inlining | (new deferred observation) | OK — correctly recurses |
| commit-C4-4 | `_compatible` TyTensor/TyTile structural arms | audit-C4-2 | **C5-2** (paper-only — shape compare cascades to `a == b`) |
| commit-C4-5 / E3 | `monomorphize_safe` wrapper | audit-C4-8 | **C5-9** (narrow catch + driver continues with half-mutated prog) |
| commit-E1 | TyArray size compare via `_compatible` | type-design E1 | **C5-2** (same root cause — `_compatible` lacks TyVar/TySize defer) |
| commit-E2 | Logic+bare-T wrap-asymmetric warn | type-design E2 | OK |
| commit-E4 | D3 diagnostic wording | type-design E4 | OK (cosmetic) |
| commit-E6 | `_inline_lets` Call arm preserve generics | type-design E6 | OK for Name cand |
| commit-E7 | TileLit identity arm (covered by commit-C4-1) | type-design E7 | **C5-6** (same finding) |
| commit-E8 | `_inline_lets` Call arm walks Field-typed callees | type-design E8 | OK — recurses correctly |

### Cycle-4 AUDIT findings status

The cycle-4 audit's eight findings vs the cycle-4 fix-sweep:

| Audit finding | Severity | Cycle-4-fix-sweep status | C5 status |
|---|---|---|---|
| audit-C4-1 | CRITICAL | NOT addressed (fix-sweep took wrong direction; should have reverted, broadened instead) | **still open** + BROADENED via C5-1 |
| audit-C4-2 | HIGH | Claimed closed (commit-C4-4); paper-only | **still open** as C5-2 |
| audit-C4-3 | HIGH | NOT addressed (no TyVar arm in `_compatible`) | **still open** — same root cause as C5-2 |
| audit-C4-4 | HIGH | NOT addressed (D9 paper-only; fix-sweep says "D9 already done in cycle 3") | **still open** as C5-3 |
| audit-C4-5 | HIGH | Closed (commit-C4-1) — Path/Continue work | **closed**; but TileLit arm introduced C5-6 |
| audit-C4-6 | MEDIUM | Claimed closed (commit-C4-2); fix took wrong direction | **still open** + BROADENED via C5-1 |
| audit-C4-7 | MEDIUM | NOT addressed (check.py still uses `except Exception`) | **still open** as C5-7 |
| audit-C4-8 | LOW | Claimed closed (commit-C4-5); caller doesn't abort | **still open** + introduced C5-9 silent-miscompile window |

---

## What was checked but found OK (no new finding)

- `autodiff.py:_inline_lets` Path / Continue identity arms silence
  the catch-all warn correctly (genuinely leaf-like).
- `autodiff.py:_simplify` Binary fold's narrowed `except
  (OverflowError, ZeroDivisionError, ValueError, TypeError)` is
  appropriate — fall-through to unsimplified expression is
  semantically equivalent.
- `pytree.py:validate_pytree` correctly catches ValueError and
  surfaces as diags. Clean.
- `lexer.py:401` and `parser.py:375` `except ValueError` paths are
  narrow (int parsing) and surface clean parse-time diagnostics.
- `typecheck.py:636` `except ValueError` in `_size_type_to_lin`
  for `int(t.name[6:])` parsing is narrow.
- `monomorphize.py:182` `except ValueError` in `_subst_shape_expr`
  is narrow.
- `monomorphize.py:_subst_shape_expr` Binary fold raises
  ShapeFoldError on `/0` and `%0` correctly; caught upstream.
- `struct_mono.py:445-456` catches ShapeFoldError and ValueError
  per-instance; appropriate batch-diagnostic pattern.
- `grad_pass.py:641` `except (AttributeError, TypeError)` for the
  FnDecl `_helix_grad_cache` attribute set is narrow (frozen-
  dataclass case); fallback "no cache, just recompute" acceptable.
- `grad_pass.py:_resolve_in_expr` dispatch coverage complete per
  cycle-2 C2-4.
- `hash_cons.py:_SHAREABLE` restricts dedup to safe subset;
  `_ast_equal` defensive post-hash check raises HashConsError on
  collision. (The Quote-handle-table path in lower_ast.py does
  NOT have this defense — see C5-4.)
- `panic_pass.py:validate_panic_args` and
  `unsafe_pass.py:check_unsafe_ops` return diags lists with clean
  propagation.
- `presburger.py` — no exception swallowing observed.
- `totality.py` — exceptions propagate; clean.
- `flatten_modules.py:36` — single `pass` is a module docstring
  marker, not a swallow.
- `flatten_impls.py:DuplicateMethodError` — structured trap, caught
  cleanly in check.py.
- `autotune.py:80` `except ValueError` for `int(v)` parsing
  produces a clean per-key diag; no silent swallow.

---

## Cross-stage interaction checks

- **C5-1 (parser tag-12 broaden) + cycle-5 type-design F6 (AST_VAR
  defer empty)**: The two interact in the let-RHS dispatch chain.
  C5-1's broadening over-tags every non-trivial RHS; F6's empty
  defer arm leaves AST_VAR un-tagged. The intersection: a
  `let a = 10 + 5; let b = a; let c = |x| x + b; c(5)` case has
  `a` get tag 12 (C5-1), `b` (AST_VAR alias) get tag -1 (F6),
  closure captures `b` and silently passes (-1 < guard threshold).
  Compound silent-corruption + silent-trap windows.
- **C5-3 (D9 paper-only) + C5-8 (turbofish arity silent)**: a
  turbofish arity mismatch in a nested-generic context produces
  both a silent-pass at the mono layer (C5-8) AND a broken clone
  reference (C5-3). The two combine to produce wrong codegen with
  no diagnostic.
- **C5-4 (ast_hash Unknown fallback) + C5-10 Pattern A (lower_ast
  except Exception fallback to _pretty)**: both paths affect the
  quote-handle-table key derivation. C5-4 ensures distinct
  StructLits collide on the hash; C5-10A silently downgrades to a
  different key space if structural_hash raises. Doubly fragile.
- **C5-5 (TileLit no arm in _walk_subst_expr) + C5-6 (TileLit
  identity in _inline_lets)**: same gap in two walkers. A generic-
  shape TileLit in an AD'd generic fn passes through both walkers;
  let-bindings in shape/memspace are dropped (C5-6) AND generic-
  param substitution is dropped (C5-5). Compound silent miscompile.
- **C5-9 (monomorphize_safe narrow catch) + C5-7 (check.py over-
  broad except)**: the wrapper's purpose was to bypass the over-
  broad except's mis-attribution. C5-9 shows the wrapper is
  incomplete; C5-7 shows the over-broad except still exists. Both
  layers of the defense are weakened.
- **C5-2 (D1 elif asymmetric) + cycle-5 type-design F1 (_compatible
  no TyVar/TySize cascade)**: F1's recommended fix (cascade arm at
  top of _compatible) transitively closes C5-2 because
  `_compatible(i32, TyVar('T'))` would return True, killing the
  elif's `not self._compatible(...)` clause. Single 3-line fix
  closes both.

---

## Cycle 5 silent-failure status

**Strict criterion (per user directive 2026-05-10): cycle clean
iff zero findings of ANY severity.**

This audit finds **10 new findings (0 CRITICAL, 4 HIGH, 4 MEDIUM,
2 LOW)**. By the strict criterion, **cycle 5 silent-failure does
NOT count clean**.

The dominant patterns:

1. **Cycle-4 fix-sweep silently dropped findings**: Three cycle-4
   audit findings (C4-3 HIGH, C4-4 HIGH, C4-7 MEDIUM) have no
   equivalent in the cycle-4 fix-sweep commit. The cycle-4 fix-
   sweep label re-numbering C4-1..C4-5 (in the commit) vs the
   cycle-4 audit's C4-1..C4-8 (in the audit doc) obscured the
   discrepancy. C5-3, C5-7 re-flag these. C5-2 re-flags the same
   root cause as audit-C4-3.
2. **Cycle-4 fix-sweep wrong-direction fix**: cycle-4 audit C4-1
   was a CRITICAL functional regression with an explicit "REVERT
   D2" recommendation. The fix-sweep broadened tag-12 instead of
   reverting it. C5-1 documents that the broadening regression is
   strictly worse than pre-D2 behavior.
3. **Cycle-4 fix-sweep paper-only fix**: cycle-4 commit-C4-4 + E1
   added structural arms to `_compatible` but left the inner
   TyVar/TySize cascade gap unfixed. C5-2 documents the paper-only
   nature. Same single-line fix closes C5-2 and cycle-5
   type-design F1.
4. **New silent-failure patterns**: C5-4 (ast_hash Unknown
   fallback), C5-5 (TileLit no arm in _walk_subst_expr), C5-8
   (turbofish arity silent), C5-9 (monomorphize_safe narrow catch),
   C5-10 (lower_ast silent fallbacks).

### Stop-the-line determination: YES, on C5-1, C5-2, C5-3, C5-4

**C5-1** is a HIGH regression introduced by cycle-4 fix-sweep that
makes every legitimate non-literal i32 closure-capture pattern
SIGILL at runtime. The dominant idiom for real Helix programs.
Must REVERT before Stage 29.

**C5-2** is a HIGH paper-only fix — cycle-4 claims closure but the
headline reproducer (cycle-4-audit C4-2 + cycle-5 type-design F1)
still false-positives. Generic-shape tensor/tile calls are blocked.

**C5-3** is a HIGH still-open cycle-4 finding (D9 paper-only). The
mono pipeline silently produces clones referencing unresolved
generic-param names. Every nested-generic call pattern is broken.

**C5-4** is a HIGH new finding affecting AGI-primitive Quote
soundness. Distinct quote(...) expressions silently alias when
their inner contains StructLit / TileLit / Path / etc.

The MEDIUM (C5-5, C5-6, C5-7, C5-8) and LOW (C5-9, C5-10) findings
are cycle-6 candidates.

### Cycle 5 silent-failure → NEW FINDINGS COUNT for the strict-clean gate

**10 (0 CRITICAL + 4 HIGH + 4 MEDIUM + 2 LOW). Clean-counter
remains at 0.**

### Recommended fix sequence for cycle 6

1. **C5-1**: REVERT cycle-3 D2 and cycle-4 commit-C4-2 broadening;
   restore pre-D2 tag -1 silent-pass for complex RHS.
2. **C5-2 + cycle-5 type-design F1 combined**: add `TyVar / TySize`
   cascade arm at top of `_compatible`. Single 3-line addition
   closes both.
3. **C5-3**: fix `Monomorphizer.run` iteration order — include
   clones in the while-loop iteration.
4. **C5-4**: add explicit arms for every Expr subtype in
   `ast_hash._hash_into`; replace "Unknown" fallback with
   `raise NotImplementedError`; add post-hash equality check in
   `lower_ast.py` Quote handle path.
5. **C5-5 + C5-6**: add TileLit recursive arms in both
   `_walk_subst_expr` and `_inline_lets`.
6. **C5-7**: narrow check.py `except Exception` to internal-error
   classes only.
7. **C5-8**: move turbofish arity check into typecheck.py OR
   surface from mono via diags pattern.
8. **C5-9**: widen `monomorphize_safe` catch; abort on non-empty
   diags from the driver.
9. **C5-10**: replace silent-fallback patterns in lower_ast.py
   with explicit raises or diagnostics.

### Estimated remaining open findings going into cycle 6

- Cycle 1: 13 new (all fixed → 0 open).
- Cycle 2: 6 new (all fixed → 0 open).
- Cycle 3: 6 new (all fixed → 0 open).
- Cycle 4 silent-failure: 8 new — 1 closed (C4-5), 7 still open
  (some re-flagged as cycle-5 C5-1/2/3/7).
- Cycle 4 type-design: 8 new — partial close per cycle-5
  type-design audit; ~5 still open.
- Cycle 4 codereview: 0 new.
- Cycle 5 type-design (sibling): 6 new — all open.
- Cycle 5 codereview (sibling): 2 new — both open.
- Cycle 5 silent-failure (this audit): 10 new — all open. (C5-1,
  C5-2, C5-3, C5-6, C5-7 overlap with cycle-4 still-open findings
  in root cause but count separately for the strict-zero
  criterion.)
- Prior audits (stage 5-6, 7-8, 9-16): ~20 still-open (unchanged).

Cycle 6 entry open-findings: ~50 open findings (cumulative across
all cycles).

After the recommended fix-sequence above lands, cycle 6 should
re-audit. The strict-zero criterion to deprecate Python helixc
(Stage 29) requires 5 consecutive clean cycles; at the current
rate the soonest clean state is ~5-8 more audit cycles assuming
each cycle's fix-sweep closes its own findings without introducing
new ones. Cycle-1-through-5 pattern shows the fix-introduces-gap
rate is high enough that a strict-zero-clean sweep may require
either (a) sustained reduction in fix-introduced regressions, or
(b) relaxation of the strict criterion to "zero HIGH or CRITICAL".

---

## Strict criterion note

Per user directive 2026-05-10, cycle 5 silent-failures audit is
**NOT CLEAN** because 10 new findings of various severities were
discovered. Zero findings of any severity is the required state
for a clean cycle.
