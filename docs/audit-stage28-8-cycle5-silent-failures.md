# Stage 28.8 Cycle 5 — Silent-Failure Audit

**Date**: 2026-05-11
**Commit**: 960303b (read-only audit). Cycle-4 fix-sweep range
b3504a2..960303b (2 commits: a59e233 audit-C C4-1..C4-5 + persist
cycle-4 docs; 960303b fix-sweep C4-1..C4-5 / E1..E8).
**Scope**: All Helix source — `helixc/bootstrap/*.hx`,
`helixc/frontend/*.py`, `helixc/ir/*.py`, `helixc/backend/*.py`,
`helixc/stdlib/*.hx`. Specifically re-audits the eight cycle-4
silent-failure fixes (C4-1..C4-5 in the fix-sweep commit numbering,
plus E1..E8 type-design fixes) for fresh silent windows introduced
by the fixes themselves.
**Trigger**: pre-Stage-29 audit gate — Cycle 5 of 5. Re-audits same
scope after Cycle 4 fixes were landed.
**Strict criterion** (per user directive 2026-05-10): cycle counts
CLEAN only when **zero new findings of ANY severity** (CRITICAL/HIGH/
MEDIUM/LOW).

**Method**:
1. Read prior cycle silent-failure docs (cycle 1 — 13 findings; cycle
   2 — 6 findings; cycle 3 — 6 findings; cycle 4 — 8 findings) to
   avoid re-flagging already-documented findings. Cross-checked the
   cycle-4 *audit* C4-1..C4-8 numbering (in cycle4 audit doc) against
   the cycle-4 *fix-sweep commit* C4-1..C4-5 + E1..E8 numbering
   (renumbered in commit 960303b) — they overlap but differ.
2. Walked `git log --oneline 960303b ^b3504a2` — 2 cycle-4 fix-sweep
   commits. Read `git show <sha>` for each.
3. For each cycle-4 fix's diff, traced data flow forward to check
   whether the fix opened a fresh silent window, left a fix
   incomplete (paper-only at the headline reproducer), or compounded
   a prior-cycle regression.
4. Spot-checked the new code for: dispatch holes (walker drift on
   the cycle-4 `_inline_lets` leaf classification), state-leak after
   exception (the `monomorphize_safe` wrapper), error-channel reach
   (warning-vs-abort), false-positive broadening (the parser tag-12
   sentinel extension).
5. Exercised the cycle-4 fixes via small Python repros (direct
   module invocations) and full CLI invocations of `python -m
   helixc.check` to confirm each documented behavior on the actual
   surface tool.

**Result**: **4 new findings (0 CRITICAL, 2 HIGH, 1 MEDIUM, 1 LOW)**
— Cycle 5 NOT clean. The dominant pattern is **paper-only fixes
in the cycle-4 sweep**: the typecheck `_compatible` TyArray/TyTensor/
TyTile arms were added (E1 + C4-4 in commit numbering) but the inner
shape compare still falls through `_compatible` to `a == b` on
`TySize('N')` vs `TyPrim('size_3')`, so the cycle-4-audit's
C4-2 + cycle-4-audit's E1 reproducers STILL fail post-cycle-4 fix.
Plus the cycle-4 commit took the wrong fix direction for the
cycle-4-audit C4-1 critical: instead of REVERTING D2's tag-12
polarity (the audit's stated recommendation), the cycle-4 commit
*broadened* tag-12 to cover Binary/Unary/Index/Field/If/Match/Block/
UnsafeBlock RHS, so EVERY `let a = 10 + 5; let c = |x| x + a; c(5)`
now SIGILLs at runtime (was: silent miscompile pre-D2 + cycle-3,
was: SIGILL only on Call-RHS post-cycle-3, NOW: SIGILL on every
non-literal-non-AST_VAR-non-comparison RHS). The
`_inline_lets` TileLit identity arm silently drops let-bindings
in tile shape/memspace exprs. The `monomorphize_safe` wrapper
catches ShapeFoldError but the only caller (x86_64.py) prints a
warning and continues with a half-mutated `prog` state, producing
a silent miscompile window when the fn-mono path raises.

---

## CRITICAL FINDINGS

(none)

---

## HIGH FINDINGS

### Finding C5-1: cycle-4 fix-sweep BROADENS the cycle-4-audit C4-1 functional regression — `let a = 10 + 5; let c = |x| x + a; c(5)` (and every Binary / Unary / Index / Field / If / Match / Block / UnsafeBlock RHS of an i32-valued let) now SIGILLs at runtime

**Location**:
- helixc/bootstrap/parser.hx:2334-2374 (the cycle-4 commit C4-2
  catch-all `else { inferred_ty_tag = 12; }`)
- helixc/bootstrap/parser.hx:1819 (capture-site guard `cap_ty_tag > 0`)
**Severity**: HIGH
**Category**: cycle-4-fix-introduced functional regression
(BROADENING of the still-open cycle-4-audit C4-1 critical)
**Stage**: 28.8 cycle-4 commit 960303b (C4-2 in fix-sweep numbering)

**Description**:
The cycle-4 silent-failures audit (doc audit-stage28-8-cycle4-
silent-failures.md) C4-1 finding was a **CRITICAL** regression
introduced by cycle-3 D2: the parser tagged ALL Call-RHS untyped
lets as non-i32 (tag 12), so `let pi = make_i32(); let c = |x| x +
pi; c(5)` SIGILLed at runtime even though `make_i32() -> i32`.

The cycle-4-audit's recommendation for C4-1 was clear:

> 1. REVERT the D2 fix. The bootstrap parser cannot infer return
>    types without a typechecker, and pre-fix tag -1 (untracked,
>    pass cleanly) was the right behavior for the dominant pattern.

The cycle-4 fix-sweep took the **opposite direction**. Instead of
reverting, it **broadened** the tag-12 sentinel from the single
`val_tag == 16` (Call) arm to a catch-all `else`-branch covering
every val_tag EXCEPT a small whitelist (AST_INT literal tags;
AST_VAR / val_tag == 1 deferred to existing var_type_tab path;
comparisons / val_tag in {6, 19, 20, 21, 22, 23} marked tag 0
because they reify to 0/1 i32 booleans).

The consequence: every Helix program with a closure capturing a
let bound to a complex RHS now SIGILLs at runtime.

Reproducers (parser.hx-level — would all SIGILL if not pre-existing
test corpus avoids the pattern):

```helix
// 1. Binary i32 RHS — should be safe (i32 + i32 = i32).
fn main() -> i32 {
    let a = 10 + 5;             // val_tag = 2 (AST_ADD); tag 12 set
    let c = |x| x + a;          // capture guard 12 > 0 → trap 76003
    c(5)                        // SIGILL at runtime
}

// 2. Unary i32 RHS.
fn main() -> i32 {
    let n = 7;
    let m = 0 - n;              // val_tag = 3 (AST_SUB); tag 12 set
    let c = |x| x + m;          // SIGILL
    c(3)
}

// 3. If-expression i32 RHS.
fn main() -> i32 {
    let a = if 1 > 0 { 10 } else { 20 };  // val_tag = 7 (AST_IF); tag 12 set
    let c = |x| x + a;          // SIGILL
    c(5)
}

// 4. Index i32 RHS.
fn main() -> i32 {
    let arr = [1, 2, 3];
    let v = arr[1];             // val_tag = AST_INDEX; tag 12 set
    let c = |x| x + v;          // SIGILL
    c(0)
}

// 5. Block i32 RHS.
fn main() -> i32 {
    let v = { let inner = 7; inner };  // val_tag = AST_BLOCK; tag 12 set
    let c = |x| x + v;          // SIGILL
    c(35)
}

// 6. Match i32 RHS.
fn main() -> i32 {
    let v = match 1 { 1 => 10, _ => 20 };  // val_tag = AST_MATCH; tag 12 set
    let c = |x| x + v;          // SIGILL
    c(0)
}
```

In each case the actual value is unambiguously i32. The
"bit-truncation-on-capture" trap 76003 was designed for f64 captures
(which truncate to i32 in the Phase-0 capture stride). The cycle-3
D2 mis-extended it to Call-RHS. The cycle-4 fix-sweep's "C4-2"
commit (different numbering: it's labelled "C4-2" in the fix-sweep
commit but it CLOSES the cycle-4-audit's C4-6 deferred-incompleteness
recommendation, NOT the C4-1 critical) now mis-extends it to **every
non-literal non-comparison RHS**. The recommendation in the cycle-4
audit was: "**REVERT** the polarity." The fix-sweep did the
opposite — preserved the polarity and broadened the surface.

Pre-fix-sweep (cycle 3 state): `let a = 10 + 5; let c = |x| x + a;
c(5)` returned 20 cleanly. tag -1 (untracked) silently passed the
capture guard.
Post-fix-sweep (cycle 4 state): same code SIGILLs (exit 132).

**Hidden errors**:
- Every closure capturing a let-bound integer arithmetic value
  produces SIGILL at runtime where pre-cycle-3 it ran cleanly.
- The bootstrap parser itself contains 11+ `let X = Y + Z;` /
  `let X = Y - Z;` patterns at file-level (verified via grep). None
  are inside closures TODAY, but any future refactor that moves
  such a pattern into a closure context will silently SIGILL.
- The fix-sweep's commit message claims to ADDRESS the cycle-4-audit
  findings, but C4-1 (CRITICAL) is **not** addressed — the
  regression test still asserts `c(0) == 132` (SIGILL is correct).
  Cycle 5 inherits BOTH the original C4-1 regression AND the
  broader Binary/Unary/Index/etc. extension.
- The trap-76003 over-fire was originally about Phase-0 closure
  capture stride limitations, not about RHS i32-ness. The polarity
  choice trades one class of miscompile (f64→i32 silent bit-trunc,
  rare; reachable via `let pi = 3.14_f64; ...`) for a
  vastly-more-common class (i32 arithmetic captured cleanly,
  common in any nontrivial program).

**Recommendation**:
1. REVERT the C4-2 fix-sweep broadening (parser.hx:2334-2374).
   Keep ONLY the explicit-typed-literal arms (val_tag == 27/31/34/
   35/36/37/38/39/40/41 for typed numeric literals) at proven
   non-i32 tags. Remove the `val_tag == 16` Call arm AND the new
   `else inferred_ty_tag = 12` catch-all. The pre-D2 behavior
   (tag -1 silent-pass for complex RHS) is the dominant-correct
   choice given no typechecker.
2. Document the f64-bit-truncation hole (the original trap-76003
   purpose) as a deferred limitation that requires a real
   typechecker to close. Reject only the cases where the RHS is
   PROVABLY non-i32 (explicit-typed literals, explicit type
   annotation `let x: f64 = ...`).
3. Update the regression test at test_codegen.py:3498-3504 to
   assert the **correct** behavior: `let pi = get_pi(); let c = |y|
   y + pi; c(0)` returns 3 (not 132). Same for the cycle-5
   reproducers above.
4. Pre-fix-sweep state preserved a working dominant idiom; the
   cycle-4 fix-sweep broke it; cycle-5 should restore it.

**Trap-id**: 76003 (existing — over-firing across an even wider
surface than the still-open cycle-4-audit C4-1).

---

### Finding C5-2: cycle-4 fix-sweep `_compatible` TyArray / TyTensor / TyTile structural arms are paper-only — `TySize('N')` vs `TyPrim('size_3')` in shape positions still cascades to `a == b` → False, so the cycle-4-audit C4-2 reproducer (every generic-tensor / generic-array / generic-tile call) STILL emits a false-positive

**Location**:
- helixc/frontend/typecheck.py:2197-2204 (E1 fix — TyArray size compare)
- helixc/frontend/typecheck.py:2230-2248 (C4-4 fix — TyTensor / TyTile arms)
- helixc/frontend/typecheck.py:2249 (`return a == b` catch-all)
**Severity**: HIGH
**Category**: cycle-4-fix-claimed-but-paper-only
**Stage**: 28.8 cycle-4 commit 960303b (E1 + C4-4 in fix-sweep numbering)

**Description**:
The cycle-4 fix-sweep commit 960303b claims to close the cycle-4-
audit C4-2 (HIGH — D1 has no TyTensor/TyTile structural arms) and
the cycle-4-type-design audit E1 (TyArray size compare). The
fix-sweep added structural arms:

```python
# E1 — TyArray size compare:
if isinstance(a, TyArray) and isinstance(b, TyArray):
    return (self._compatible(a.elem, b.elem)
            and (a.size == b.size
                 or self._compatible(a.size, b.size)))
```

```python
# C4-4 — TyTensor / TyTile arms:
if isinstance(a, TyTensor) and isinstance(b, TyTensor):
    if len(a.shape) != len(b.shape):
        return False
    return (self._compatible(a.dtype, b.dtype)
            and all(self._compatible(x, y)
                    for x, y in zip(a.shape, b.shape))
            and a.device == b.device
            and a.layout == b.layout)
```

Both arms recurse via `self._compatible` on the size / shape
components. But `_compatible` has **no arm for `TySize`** at all:
the cascade walks TyUnknown / TyMemTier / TyQuote / TyDiff / TyLogic
/ TyTuple / TyArray / TyRef / TyPtr / TyFn / TyTensor / TyTile —
none match TySize. So `_compatible(TySize('N'), TyPrim('size_3'))`
falls through to the catch-all `return a == b` at line 2249 →
False (different classes, frozen-dataclass equality).

The E1 fix has a softer fallback: `a.size == b.size OR
self._compatible(a.size, b.size)`. But both legs return False on
`TySize('N')` vs `TyPrim('size_3')` (the dominant
generic-call-against-concrete-shape case). The C4-4 TyTensor/TyTile
arm has NO `==` short-circuit — it's strict `_compatible`.

Verified end-to-end via `python -m helixc.check --strict` on:

```helix
@pure fn norm[N: size](x: tensor<f32, [N]>) -> f32 { 0.0_f32 }
@pure fn use_norm(m: tensor<f32, [3]>) -> f32 { norm(m) }
fn main() -> i32 { 0 }
```
```
typecheck: 1 ERRORS
  error: call to 'norm': arg 'x' expects tensor<f32, [size:N]>, got tensor<f32, [size_3]>
```

Same false-positive as the cycle-4-audit's C4-2 reproducer.
Verified the same for the TyArray case:
```helix
@pure fn norm[N: size](x: [f32; N]) -> f32 { 0.0_f32 }
@pure fn use_norm(m: [f32; 4]) -> f32 { norm(m) }
```
```
typecheck: 1 ERRORS
  error: call to 'norm': arg 'x' expects [f32; size:N], got [f32; size_4]
```

The cycle-4 fix-sweep's E1 + C4-4 fixes are **structurally added
but functionally inert** for the dominant generic-call-against-
concrete-shape case.

Verified via direct `_compatible` probe:
```
[f32;N] ~ [f32;4]: False
tensor<f32,[N]> ~ tensor<f32,[3]>: False
i32 ~ T (TyVar): False
```

**Hidden errors**:
- Every generic-array / generic-tensor / generic-tile call now
  emits a false-positive (same as pre-fix-sweep cycle-4-audit C4-2).
- The cycle-4 fix-sweep commit message lists C4-4 + E1 as
  CLOSED — they are not. A future contributor following the
  commit log will assume these cases work.
- Cycle-4-audit C4-3 (`fn g[T](v:T) -> i32 { f(v) }` with `fn
  f(i:i32)`) is unaddressed for the same root cause: `_compatible`
  has no TyVar arm. The cycle-4 fix-sweep did not even attempt
  this finding (it's not in the commit message). Cycle 5 inherits
  it.

**Recommendation**:
1. Add an explicit `TySize` defer arm at the TOP of `_compatible`,
   mirroring the existing TyUnknown short-circuit:
   ```python
   if isinstance(a, (TyVar, TySize)) or isinstance(b, (TyVar, TySize)):
       return True   # defer to mono / cascade-safe
   ```
   This single change closes BOTH the still-open C4-2 (audit) and
   C4-3 (audit) findings.
2. Alternatively, add per-shape-element defer logic inside each
   shape-bearing structural arm, mirroring the cycle-4-audit's
   recommendation #1 verbatim:
   ```python
   for sa, sb in zip(a.shape, b.shape):
       if isinstance(sa, (TyUnknown, TyVar, TySize)) \
               or isinstance(sb, (TyUnknown, TyVar, TySize)):
           continue   # defer to mono / cascade-safe
       if not self._compatible(sa, sb):
           return False
   ```
3. Add regression tests that exercise the actual surface:
   `python -m helixc.check --strict` on the reproducers above
   must return `typecheck: OK`. Unit tests that probe
   `tc._compatible(TyArray(elem=f32, size=TySize('N')),
   TyArray(elem=f32, size=TyPrim('size_4')))` must return True.

**Trap-id**: n/a (typecheck error mis-fire, no trap-id).

---

## MEDIUM FINDINGS

### Finding C5-3: `_inline_lets` cycle-4 C4-1 TileLit identity arm silently drops let-bindings in tile shape / memspace / dtype / init children — `tile<f32, [N], REG>::zeros()` with `N` let-bound silently leaves `N` un-substituted

**Location**:
- helixc/frontend/autodiff.py:709-710 (TileLit identity arm)
- helixc/frontend/ast_nodes.py:344-356 (`A.TileLit` definition —
  `shape: list["Expr"]`, `memspace: "Expr"`)
**Severity**: MEDIUM
**Category**: cycle-4-fix-introduced silent drop (false-leaf
classification)
**Stage**: 28.8 cycle-4 commit 960303b (C4-1 in fix-sweep numbering)

**Description**:
The cycle-4 fix-sweep added three "leaf-like" identity arms to
`_inline_lets` to close spurious AD warnings from the cycle-3 C3-5
catch-all (the cycle-4-audit C4-5 finding):

```python
if isinstance(expr, A.Path):
    return expr
if isinstance(expr, A.Continue):
    return expr
if isinstance(expr, A.TileLit):
    return expr
```

`A.Path` (qualified name like `Maybe::None`) and `A.Continue`
(statement-expr with no children) are genuinely leaf-like — they
contain no `A.Expr` children that could reference let-bound names.

But `A.TileLit` is **not a leaf**:
```python
@dataclass(frozen=True)
class TileLit(Expr):
    dtype: "TyNode"          # type
    shape: list["Expr"]      # the shape dims (e.g. [IntLit(4), IntLit(4)])
    memspace: "Expr"         # memspace marker (e.g. Name("REG"))
    init: str                # "zeros" or "ones"
```

Both `shape` and `memspace` are Expr-typed. The TileLit identity
arm silently drops let-bindings appearing in either.

Reproducer (Python-level via direct AST construction — surface
syntax for TileLit with a let-bound shape dim is reachable in
Stage 15+ code, see test_typecheck for shape var paths):

```python
n = A.Name(span=span, name='N', generics=[])
shape = [n]
mem = A.Name(span=span, name='REG', generics=[])
tl = A.TileLit(span=span, dtype=ty_f32, shape=shape, memspace=mem,
               init='zeros')
env = {'N': A.IntLit(span=span, value=8)}
r = _inline_lets(tl, env)
# Expected: r.shape[0] == IntLit(8)
# Actual: r.shape[0] == Name('N')  (identity returned, env ignored)
```

Verified — `r is tl` (same object), no warning, `N` un-substituted.

Pre-fix-sweep (cycle 3 state): `A.TileLit` fell through to the
catch-all `_ad_warn` warning. Loud (over-fired on legit code per
cycle-4-audit C4-5), but at least not silent.
Post-fix-sweep: `A.TileLit` silently passes through unchanged. The
catch-all is suppressed. Any let-binding inside the tile literal's
shape/memspace is lost.

The cycle-4 fix-sweep classified TileLit as "leaf-like" to suppress
the cycle-3-introduced false-positive warning, but TileLit has Expr
children. The correct fix is a recursive arm, not an identity arm.

**Hidden errors**:
- Future Helix code at Stage 15+ that names a tile shape via a
  let-bound const (`let N = 8; let t: tile<f32, [N], REG>::zeros();`
  — when surface syntax matures) will silently leave `N`
  un-substituted in the differentiated body. Downstream codegen
  may produce a tile of wrong shape, or fail with an obscure
  "unresolved Name" error.
- The fix-sweep commit message claims the arm covers "compile-time
  shape + init marker" — but Phase-0 `shape: list[Expr]` admits
  arbitrary expressions, not just IntLits.
- Symmetric with the Path/Continue case in PRESENTATION (identity
  arm before catch-all) but NOT in SEMANTICS — Path and Continue
  are genuinely leaf-like; TileLit isn't.

**Recommendation**:
1. Replace the TileLit identity arm with a recursive arm:
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
2. Move the TileLit arm into the recursion block above (alongside
   the Cast / Call / Field / Index / ArrayLit / ... arms), keeping
   only Path / Continue (genuinely leaf-like) in the post-recursion
   identity block before the catch-all.
3. Add regression test:
   `_inline_lets(TileLit(shape=[Name('N')], memspace=Name('REG')),
   {'N': IntLit(8)})` returns a TileLit with `shape=[IntLit(8)]`.

**Trap-id**: n/a (silent let-binding drop).

---

## LOW FINDINGS

### Finding C5-4: `monomorphize_safe` wrapper catches `ShapeFoldError` but the x86_64 driver prints a warning and CONTINUES the pipeline with a half-mutated `prog` (some `item.body` mutations applied, clones in `self.instantiated` never appended) — silent miscompile window

**Location**:
- helixc/frontend/monomorphize.py:691-706 (`monomorphize_safe` wrapper)
- helixc/frontend/monomorphize.py:412-438 (`Monomorphizer.run` —
  the `item.body = new_body` mutations at line 432 happen
  before the post-loop append at line 437)
- helixc/backend/x86_64.py:3025-3029 (the only caller — prints
  warning then continues to grad_pass, typecheck, codegen)
**Severity**: LOW
**Category**: cycle-4-fix-introduced silent-failure window
**Stage**: 28.8 cycle-4 commit 960303b (C4-5 / E3 in fix-sweep
numbering)

**Description**:
The cycle-4 fix-sweep added a `monomorphize_safe` wrapper around
the fn-mono `Monomorphizer.run()` to catch the still-open
cycle-4-audit C4-8 finding (uncaught `ShapeFoldError` in fn-mono
path mis-attributed by C3-3's outer `except Exception` as
"compiler bug"). The wrapper:

```python
def monomorphize_safe(prog: A.Program) -> tuple[int, list[str]]:
    try:
        return monomorphize(prog), []
    except ShapeFoldError as e:
        return 0, [str(e)]
```

The wrapper docstring states:
> On a ShapeFoldError, the caller should treat the diag as a
> typecheck error and abort the pipeline (callers that don't
> care can ignore diags; the count is 0 in that case).

But the only caller — x86_64.py:3025-3029 — does NOT abort:
```python
mono_count, mono_diags = monomorphize_safe(prog)
for d in mono_diags:
    print(f"warning: fn-mono: {d}", file=sys.stderr)
if mono_count > 0:
    print(f"mono: {mono_count} generic instantiation(s)", ...)
grad_count = grad_pass(prog)
... typecheck(prog) ...
... codegen ...
```

A `ShapeFoldError` mid-`Monomorphizer.run`:
1. Some prior iterations of the `while changed` loop at line 426-433
   may have already reassigned `item.body = new_body` for items
   processed before the failing item.
2. `self.instantiated` accumulated clones for keys that completed
   before the raise.
3. The raise propagates out of `_rewrite_calls_in_block` →
   `run()` BEFORE the post-loop `prog.items = list(prog.items) +
   list(self.instantiated.values())` at line 437.
4. `monomorphize_safe` catches and returns `(0, [str(e)])`.

The resulting `prog` state:
- Some `item.body` fields reference mangled names (e.g.,
  `id__i32`) due to step 1.
- The clones for those mangled names are MISSING from
  `prog.items` due to step 3.

The pipeline then runs `grad_pass`, `typecheck`, and `codegen` on
a prog that references undefined mangled functions. Possible
outcomes:
- typecheck emits confusing "unknown function" errors that don't
  point at the real cause (trap 28801).
- codegen emits a binary with unresolved references, OR a binary
  that successfully assembles but jumps to a nonexistent symbol.
- In `--strict` mode the typecheck errors abort. In default mode
  they emit as warnings and codegen proceeds.

Pre-fix-sweep (cycle 3 state): uncaught `ShapeFoldError` → check.py
outer `except Exception` catches → prints `helixc: internal error:
ShapeFoldError ...` + `compiler bug — please file an issue` →
rc=1 → pipeline exits cleanly with NO binary produced.
Post-fix-sweep: `warning: fn-mono: ...` printed → pipeline continues
→ broken intermediate `prog` state → possibly broken binary
produced (or confusing downstream typecheck errors that don't name
trap 28801).

**Why LOW**: same reachability gating as the still-open
cycle-4-audit C4-8. The Phase-0 parser does NOT currently accept
shape arithmetic on fn signatures (`fn k[T, N](a: [T; N/0])` etc.).
Reaching the fn-mono `ShapeFoldError` raise today requires direct
Python AST construction. If future stages widen the parser, this
promotes to MEDIUM (silent miscompile becomes user-reachable).

The wrapper's docstring contract is violated by the only caller.
The behavior is internally inconsistent: `monomorphize_structs`
returns `(prog, diags)` AND the struct_mono caller at x86_64.py:
3017-3020 prints diags as warnings then continues — but
struct_mono uses a per-instance try/except that ACCUMULATES diags
without leaving prog mutated. `monomorphize_safe` does NOT (the
try/except is at the top-level `run()` call, so partial mutations
persist).

**Hidden errors**:
- Future-reachable silent miscompile when fn-mono shape arithmetic
  with `/0` or `%0` becomes parser-reachable.
- `mono_count = 0` is misreported on raise — when partial
  iterations succeeded, the count understates the real instantiation
  attempts.
- Users see a warning that looks routine (`warning: fn-mono: ...`)
  in build output, easy to overlook.
- Downstream typecheck errors don't reference trap 28801, defeating
  the trap-id diagnostic system.

**Recommendation**:
1. Either:
   (a) Make `monomorphize_safe` ABORT the pipeline at its caller
       site:
   ```python
   mono_count, mono_diags = monomorphize_safe(prog)
   if mono_diags:
       for d in mono_diags:
           print(f"error: fn-mono: {d}", file=sys.stderr)
       sys.exit(1)
   ```
   This matches the docstring contract.
   (b) Or: rewrite `Monomorphizer.run` to catch ShapeFoldError
       per-instance (parallel to struct_mono's approach), so the
       wrapper's "(0, diags)" return reflects "0 SUCCESSFUL adds"
       and the prog state remains consistent (clones for
       successful instantiations appended; failing ones omitted).
2. If (a): also update check.py to call `monomorphize_safe` (it
   currently doesn't call fn-mono at all — separate concern flagged
   in cycle-4-audit C4-8 deferred observations).
3. Add a regression test that forces `ShapeFoldError` via direct
   Python AST construction, asserts the diag contains "trap
   28801", and asserts the pipeline exits with rc != 0 (not
   rc=0 with broken binary).

**Trap-id**: 28801 (existing — diag emitted at warning level
instead of error level).

---

## Cycle 4 fix-sweep re-verification

Each cycle-4 fix-sweep change was inspected for paper-only fixes,
silent windows, and false positives introduced. The cycle-4
fix-sweep commit (960303b) numbering uses C4-1..C4-5 + E1..E8
WHICH DIFFERS from the cycle-4 audit's C4-1..C4-8 numbering — the
table below maps both.

| fix-sweep label | What changed | Audit-doc cross-ref | C5 verdict |
|---|---|---|---|
| commit-C4-1 | `_inline_lets` Path/Continue/TileLit identity arms | audit-C4-5 | **C5-3** (TileLit silently drops shape/memspace; not a leaf) |
| commit-C4-2 | parser.hx tag-12 broadened to all complex RHS | audit-C4-6 (deferred) + still-open audit-C4-1 | **C5-1** (BROADENS the still-open audit-C4-1 critical regression) |
| commit-C4-3 | `_inline_lets` If.cond inlining | new (deferred observation) | OK — correctly recurses |
| commit-C4-4 | `_compatible` TyTensor/TyTile structural arms | audit-C4-2 | **C5-2** (paper-only — shape compare still cascades to `a == b`) |
| commit-C4-5 / E3 | `monomorphize_safe` wrapper | audit-C4-8 | **C5-4** (wrapper docstring contract violated by caller) |
| commit-E1 | TyArray size compare via `_compatible` | type-design E1 | **C5-2** (same root cause — `_compatible` lacks TySize defer) |
| commit-E2 | Logic+bare-T wrap-asymmetric warn | type-design E2 | OK — intended behavior change |
| commit-E4 | D3 diagnostic wording | type-design E4 | OK (cosmetic) |
| commit-E6 | `_inline_lets` Call arm preserve generics | type-design E6 | OK for Name cand; Path cand still loses generics (pre-existing AST shape limitation) |
| commit-E7 | TileLit identity arm | covered by commit-C4-1 | **C5-3** (same finding) |
| commit-E8 | `_inline_lets` Call arm walks Field-typed callees | type-design E8 | OK — recurses correctly |
| audit-C C4-1 | `TRAP_*` constants promotion | type-design / housekeeping | OK (defensive; `ShapeFoldError.trap_id` class attr is inert today but harmless) |
| audit-C C4-3 | `_inline_lets` catch-all no longer pre-embeds `(trap 85001)` | (housekeeping) | OK — `_ad_warn` appends; no double-print |

### Cycle-4 AUDIT findings status

The cycle-4 audit's eight findings (separate from the fix-sweep
commit's labels):

| Audit finding | Severity | Cycle-4-fix-sweep status | C5 status |
|---|---|---|---|
| audit-C4-1 | CRITICAL | NOT addressed (fix-sweep took wrong direction; should have reverted D2, broadened instead) | **still open** (and BROADENED via C5-1) |
| audit-C4-2 | HIGH | Claimed closed (commit-C4-4); paper-only | **still open** (re-flagged as C5-2 because fix is paper-only — the fix-sweep CLAIMS closure) |
| audit-C4-3 | HIGH | NOT addressed (no TyVar arm in `_compatible`) | **still open** (cycle-4 reflects same root cause as C5-2) |
| audit-C4-4 | HIGH | NOT addressed (D9 paper-only; fix-sweep says "D9 already done in cycle 3") | **still open** |
| audit-C4-5 | HIGH | Closed (commit-C4-1) — Path/Continue arms work | **closed**; but TileLit arm introduced C5-3 |
| audit-C4-6 | MEDIUM | Claimed closed (commit-C4-2); fix took wrong direction (broadening, not narrowing) | **still open** + BROADENED via C5-1 |
| audit-C4-7 | MEDIUM | NOT addressed (check.py still uses `except Exception`) | **still open** |
| audit-C4-8 | LOW | Claimed closed (commit-C4-5); but caller doesn't abort | **still open** + introduced C5-4 silent-miscompile window |

### Specific re-verifications from the audit instructions

- **commit-C4-1 (`_inline_lets` Path/Continue/TileLit)**: Path and
  Continue arms correct (verified — `_inline_lets(Path, env)`
  returns identity, no warning, no children lost since neither
  type has Expr children). TileLit arm is a false-leaf
  classification — TileLit has Expr children in `shape` and
  `memspace`. See C5-3.
- **commit-C4-2 (parser.hx tag-12 broaden)**: Broadens the existing
  cycle-3 D2 regression. The cycle-4 audit's recommendation for
  C4-1 was to REVERT D2; the fix-sweep instead broadened the
  surface. See C5-1.
- **commit-C4-3 (`_inline_lets` If.cond)**: Probed via direct
  call. `if g(x) > 0.0 { ... }` with `g` let-bound substitutes
  correctly post-fix. Result.cond shows the substituted callee.
  No fresh regression. Pre-existing minor concern: substituting
  a let-bound impure expr into cond duplicates side effects if
  cond evaluates more than once at runtime — but the if-cond
  evaluates exactly once, so this concern doesn't apply at the
  call site.
- **commit-C4-4 (`_compatible` TyTensor/TyTile)**: Structural arms
  added. Shape compare uses `_compatible` recursively → falls
  through to `a == b` on `TySize('N')` vs `TyPrim('size_3')`.
  Paper-only. See C5-2.
- **commit-C4-5 / E3 (`monomorphize_safe`)**: Wrapper added.
  Caller doesn't abort. Half-mutated prog state may pass downstream
  with confusing errors. See C5-4.
- **commit-E1 (TyArray size compare)**: Same root cause as C5-2 —
  `_compatible(TySize('N'), TyPrim('size_4'))` falls through. The
  `a.size == b.size OR _compatible(a.size, b.size)` short-circuit
  doesn't help.
- **commit-E2 (Logic+bare-T wrap-asymmetric)**: Gate broadened to
  `inner_mismatch OR l_is_logic != r_is_logic`. Intended behavior
  change. Probed a simple Logic-use case (`use_logic(x: Logic<bool>)
  -> Logic<bool> { x }`) — no spurious warning. Mixed surface
  syntax (Logic<bool> + bool) doesn't parse at Phase-0 surface, so
  the broadening's effect is gated by Logic-arithmetic surface
  syntax not yet present. No fresh regression at the surface.
- **commit-E4 (D3 wording)**: Cosmetic. Both branches now say
  "must be > 0". No silent failure.
- **commit-E6 (Call arm preserve generics)**: For `cand: A.Name`,
  generics preserved. For `cand: A.Path`, generics dropped — but
  A.Path has no `generics` field, so this is a pre-existing AST-
  shape limitation, not a fix-introduced regression.
- **commit-E8 (Call arm Field-typed callee)**: `_inline_lets`
  recurses into Field-typed callees. Probed — works correctly.
- **audit-C C4-1 (TRAP_* constants)**: Module-level constants
  added. `ShapeFoldError.trap_id` class attribute is technically
  inert (no consumer reads it). Defensive; harmless. No silent
  failure.
- **audit-C C4-3 (catch-all trap-id de-duplication)**: Removed
  pre-embedded `(trap 85001)` from `_ad_warn`'s `reason` arg.
  `_ad_warn` appends `(trap {TRAP_AD_ASSUMED_ZERO})` exactly once.
  Verified — no double-print.

---

## Cross-stage interactions checked

- **C5-1 broaden + bootstrap parser.hx**: Bootstrap parser.hx
  contains 11+ `let X = Y + Z;` patterns at file-level (verified
  via grep). None inside closures TODAY. Any future refactor that
  moves such a pattern into a closure context silently SIGILLs.
  Risk surface is real for future Helix code.
- **C5-2 paper-only + `_compatible` recursion depth**: TyArray /
  TyTensor / TyTile structural arms recurse via `_compatible` on
  inner types. Phase-0 has no cyclic types, so recursion is
  bounded. No infinite-loop risk.
- **C5-3 TileLit silent drop + AD-vs-grad_pass**: `_inline_lets`
  is called from `differentiate(expr, var)` (autodiff.py:169).
  TileLit silent drops would manifest as a differentiated
  expression that still references un-substituted Names. The
  derivative computation at `_diff` may then error on the
  un-substituted Name as "unknown variable" — but that error
  surfaces away from the root cause.
- **C5-4 wrapper + check.py**: `check.py` does NOT call fn-mono
  at all (only `monomorphize_structs`). So the `monomorphize_safe`
  wrapper only affects the x86_64 driver. Users running
  `helixc check foo.hx` against a program with a fn-mono
  `ShapeFoldError` trigger see NEITHER the trap-28801 diagnostic
  NOR the warning — they see no diagnostic at all, and the
  binary-emit driver later fails. A separate reachability gap
  but not a cycle-5 finding per se.
- **C5-1 + closure-call test corpus**: The closure test
  corpus in test_codegen.py exercises `let a = 10; let c = |x|
  x + a; c(5)` (literal RHS, val_tag 0, tag 0 — passes) and
  `let pi = get_pi(); let c = |y| y + pi; c(0)` (Call RHS,
  val_tag 16, tag 12 — SIGILL, codified). It does NOT cover any
  Binary / Unary / Index / Field / If / Match / Block / UnsafeBlock
  RHS — those would all SIGILL post-cycle-4 with no regression
  test catching the regression.

---

## What was checked but found OK (no new finding)

- commit-C4-3 (If.cond inline) works correctly. Substitutes
  let-bound callees / values into the condition.
- commit-E1 + commit-C4-4 added the structural arms (functionally
  inert per C5-2; the STRUCTURE is correct, just the recursion
  cascade is incomplete).
- commit-E2 Logic-asymmetric broadening: intended behavior change
  per the cycle-4 type-design audit's recommendation.
- commit-E4 D3 wording is cosmetic.
- commit-E6 Name-cand generics-preserve works correctly.
- commit-E8 Field-callee inline works correctly.
- TRAP_* module constants are added correctly; no consumer breakage.
- The cycle-3 D2 still-open audit-C4-1 critical is acknowledged as
  unfixed but NOT re-flagged as a new C5 finding (it's a cycle-4
  carryover). C5-1 separately flags the BROADENING introduced by
  cycle-4 fix-sweep (which compounds, not duplicates, the audit-C4-1).
- The cycle-4-audit C4-3 (typecheck.py D1 elif TyVar asymmetry) is
  acknowledged as unfixed but NOT re-flagged as new C5 — root cause
  is the same as C5-2 (missing TyVar/TySize arm in `_compatible`),
  and C5-2's fix recommendation closes both. The C5-2 finding is
  written to encompass the audit-C4-3 root cause.
- The cycle-4-audit C4-4 (D9 paper-only) is acknowledged as unfixed
  but NOT re-flagged as new C5 — already documented and unaddressed
  in cycle-4 fix-sweep ("D9 already done in cycle 3").
- The cycle-4-audit C4-7 (check.py `except Exception`) is unfixed
  but NOT re-flagged — already documented.
- The cycle-4-audit C4-8 (Monomorphizer.run uncaught ShapeFoldError)
  is "structurally closed" by commit-C4-5 / E3 wrapper but the
  closure has a fresh hole — C5-4 documents that hole, not the
  original C4-8.

---

## Deferred / out-of-scope observations (NOT new findings; cycle-6
candidates if any)

- **Bootstrap parser.hx C5-1 over-trap test corpus expansion**: a
  cycle-6 (or post-cycle-5 fix batch) should add regression tests
  exercising Binary / Unary / Index / Field / If / Match / Block /
  UnsafeBlock closure-capture patterns at both polarities
  (i32-typed RHS → clean; explicitly-non-i32-typed RHS → trap)
  once C5-1 is resolved.
- **`Monomorphizer.run` per-instance ShapeFoldError catch**: the
  C5-4 recommendation (b) of rewriting `run()` to catch per-instance
  would parallel struct_mono's approach. Larger refactor; deferred
  to a separate audit pass.
- **C3-5 catch-all classifications audit**: the cycle-3-introduced
  catch-all `_ad_warn` plus the cycle-4-introduced identity-arms
  (Path/Continue/TileLit) suggest the dispatch table for
  `_inline_lets` should be enumerated against `ast_nodes.Expr`
  subtypes explicitly (each one classified as recurse / leaf-no-op
  / unknown-warn). The cycle-4-audit C4-5 recommendation #2 already
  proposed this. Deferred.
- **Logic-binop E2 surface reach**: E2 fires only when both sides
  of a binop are Logic-or-bare types; Phase-0 surface for
  `Logic<bool> + bool` doesn't parse, so the broadening is gated by
  future surface work. Deferred verification of broadening on real
  surface code to a later cycle.

---

## Summary

| #    | Severity | Location                                                    | Finding                                                                                                            |
|------|----------|-------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------|
| C5-1 | HIGH     | parser.hx:2334-2374 + 1819                                  | cycle-4 commit-C4-2 BROADENS tag-12 sentinel to all complex RHS → every `let a = 10 + 5; let c = |x| x + a; c(5)` SIGILLs; compounds still-open audit-C4-1 |
| C5-2 | HIGH     | typecheck.py:2197-2204 + 2230-2248 + 2249                    | cycle-4 commit-E1 + commit-C4-4 structural arms are paper-only — `_compatible(TySize, TyPrim('size_N'))` cascades to `a == b` → False; audit-C4-2 + audit-E1 reproducers still fail |
| C5-3 | MEDIUM   | autodiff.py:709-710                                         | cycle-4 commit-C4-1 TileLit identity arm silently drops let-bindings in tile shape/memspace exprs (false-leaf classification) |
| C5-4 | LOW      | monomorphize.py:691-706 + x86_64.py:3025-3029                | cycle-4 commit-C4-5 `monomorphize_safe` catches ShapeFoldError but caller prints warning + continues with half-mutated prog → silent miscompile window |

**Total: 4 new findings (0 CRITICAL, 2 HIGH, 1 MEDIUM, 1 LOW).**

---

## Cycle 5 status

**Cycle 5 NOT clean.** Per the strict criterion (zero findings of
ANY severity), the 2 HIGH + 1 MEDIUM + 1 LOW new findings BLOCK the
cycle-5 clean determination.

### Stop-the-line determination: **YES, on C5-1 and C5-2**.

**C5-1 (HIGH)** is the highest-priority new finding. The cycle-4
fix-sweep took the wrong direction on the still-open
audit-C4-1 CRITICAL: instead of REVERTING D2's tag-12 polarity,
it BROADENED the surface to cover Binary/Unary/Index/Field/If/Match/
Block/UnsafeBlock RHS. **Every closure capturing a let-bound
arithmetic-result, index-result, if-result, or match-result of an
i32-valued expression now SIGILLs at runtime.** This is the most
common idiom in any non-trivial Helix program. The fix-sweep made
the regression worse, not better. The cycle-5 audit must escalate
this to a stop-the-line before Stage 29 can proceed.

**C5-2 (HIGH)** documents that the cycle-4 fix-sweep's E1 +
commit-C4-4 fixes are paper-only at the headline reproducer. The
structural arms were added but the inner shape compare falls
through `_compatible` to `a == b` on the dominant `TySize` vs
`TyPrim('size_N')` pair. The cycle-4 fix-sweep commit message
claims these CLOSE the audit-C4-2 + audit-E1 findings — they
don't. A future contributor walking the commit log will assume
these cases work. Stop-the-line because the audit-C4-2 was already
HIGH and the cycle-4 fix-sweep deceptively marked it closed.

**C5-3 (MEDIUM)** is the cycle-4-fix-sweep's false-leaf
classification of TileLit. Reachability is gated by Phase-0 surface
syntax for tile shapes; today the user-reachable surface only
admits IntLit shape components. Future stages widening tile-shape
expression syntax promote this to HIGH.

**C5-4 (LOW)** is the `monomorphize_safe` wrapper's docstring-vs-
caller mismatch. Reachability is gated by Phase-0 parser not
accepting fn-signature shape arithmetic; today only reachable via
direct Python AST construction. Future parser widening promotes
to MEDIUM.

### Cycle 5 → NEW FINDINGS COUNT for the strict-clean gate: 4 (0 CRITICAL + 2 HIGH + 1 MEDIUM + 1 LOW) — clean-counter remains at 0.

### Estimated remaining open findings going into cycle 6

- Cycle 1: 13 new (all fixed → 0 open).
- Cycle 2: 6 new (all fixed → 0 open).
- Cycle 3: 6 new (all fixed → 0 open).
- Cycle 4 silent-failure: 8 new — partial close.
  - audit-C4-1 CRITICAL: still open (compounded by C5-1).
  - audit-C4-2 HIGH: still open (cycle-4 fix paper-only per C5-2).
  - audit-C4-3 HIGH: still open (same root cause as C5-2).
  - audit-C4-4 HIGH: still open (D9 paper-only).
  - audit-C4-5 HIGH: closed by commit-C4-1.
  - audit-C4-6 MEDIUM: still open (compounded by C5-1).
  - audit-C4-7 MEDIUM: still open.
  - audit-C4-8 LOW: structurally addressed but introduced C5-4.
  - 6 of 8 still open.
- Cycle 4 type-design (sibling audit): 8 new — partial close
  (E1 paper-only per C5-2; E2/E4/E6/E8 closed; E3 introduced C5-4;
  E5 deferred; E7 closed via commit-C4-1).
- Cycle 4 codereview (sibling audit): 0 new (was already clean).
- Cycle 5 silent-failure: 4 new (all open).
- Prior audits (stage 5-6 + 7-8 + 9-16): ~20 still-open (unchanged
  going into cycle 5 — neither cycle-4 nor cycle-5 touched them).
- Cycle 5 net: 20 + 6 + (≥4 type-design partial) + 4 = **≥34 open
  findings** going into cycle 6.

Recommend prioritizing in this order for the cycle-6 fix batch:
1. **C5-1** (HIGH — REVERT both the cycle-3 D2 AND the cycle-4
   commit-C4-2 broadening; restore pre-D2 tag -1 silent-pass for
   complex RHS).
2. **C5-2** (HIGH — add `TySize` / `TyVar` defer arm at the top
   of `_compatible`; this single change closes audit-C4-2,
   audit-C4-3, audit-E1, AND C5-2).
3. **audit-C4-4** (HIGH — still-open D9 paper-only; revert D9 or
   fix `Monomorphizer.run` iteration order).
4. **C5-3** (MEDIUM — replace TileLit identity arm with recursive
   arm).
5. **C5-4** (LOW — abort pipeline at the x86_64 driver when
   `mono_diags` is non-empty; either match the wrapper's docstring
   or rewrite the wrapper to clean up partial state).
6. **audit-C4-6** (MEDIUM — superseded by C5-1; same fix closes both).
7. **audit-C4-7** (MEDIUM — narrow check.py `except Exception` to
   internal-error classes only).

After this batch lands, cycle 6 should re-audit. The "5 clean cycles
before Phase 0 deprecation" goal requires the strict criterion (zero
findings of any severity) to be met — cycle 5 is the 5th cycle and is
NOT clean, so the deprecation gate is not yet met.
