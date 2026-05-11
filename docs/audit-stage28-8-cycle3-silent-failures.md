# Stage 28.8 Cycle 3 — Silent-Failure Audit

**Date**: 2026-05-10
**Commit**: 40f58ec (read-only audit)
**Scope**: All Helix source — `helixc/bootstrap/*.hx`, `helixc/frontend/*.py`,
`helixc/ir/*.py`, `helixc/backend/*.py`, `helixc/stdlib/*.hx`.
**Trigger**: pre-Stage-29 audit gate — Cycle 3 of 5. Re-audits same scope
after Cycle 2 fixes were landed (12 commits across 5d23121 ... 40f58ec).
**Strict criterion** (per user directive 2026-05-10): cycle counts CLEAN
only when **zero new findings of ANY severity** (CRITICAL/HIGH/MEDIUM/LOW).
The earlier MEDIUM/LOW-pass-through relaxation is REVOKED.

**Method**:
1. Walked `git log --since=2026-05-10` — 12 cycle-2 fix commits
   (5d23121, 134df9b, 514165b, 7a74acc, 3a29728, a086353, 0e53eb8,
   a79e867, 3126aec, 7682b14, 40f58ec; plus cycle-1 carryover commits
   already verified in Cycle 2).
2. For each cycle-2 fix's diff, traced data flow forward to check
   whether the fix opened a fresh silent window or left obvious gaps.
3. Spot-checked the new code for: dispatch holes
   (panic_pass-style walker drift), state-leak after exception,
   error-channel reach (the C2-1 drain semantics), false-positive
   warnings (rank-table ties on `i64`/`isize`).
4. Ran targeted regression tests at each step (test_typecheck,
   test_autodiff, test_autodiff_reverse, test_cli) — all current
   suites pass at 1396+ tests.
5. Exercised the cycle-2 fixes via small Python repros (direct
   subprocess invocations of `python -m helixc.check`) to confirm
   each documented behavior on the actual surface tool.

**Result**: **6 new findings (0 CRITICAL, 3 HIGH, 3 MEDIUM, 0 LOW)** —
Cycle 3 NOT clean. The dominant pattern is **fix-introduced
incompleteness**: each of the C2-1 / C2-3 / C2-4 cycle-2 fixes
closed the headline silent window but left a structurally identical
silent window adjacent to the fix. The cycle-2 widening rank
overhaul (B:C1/B:C4) introduced a *fresh* false-positive class
where same-rank-different-name pairs (i64+isize, u64+usize) spurious-warn.

---

## CRITICAL FINDINGS

(none)

---

## HIGH FINDINGS

### Finding C3-1: `grad_pass._rewrite_in_expr` for `A.If` silently skips chained `else if` branch — `grad(loss)` in an `else if` body is never rewritten

**Location**:
- helixc/frontend/grad_pass.py:358-365 (`A.If` arm in `_rewrite_in_expr`)
**Severity**: HIGH
**Category**: cycle-2-fix-introduced silent window / incomplete walker
**Stage**: 28.8 cycle-2 commit 7a74acc (C2-4)

**Description**:
The C2-4 fix to `_rewrite_in_expr` added comprehensive recursive
coverage for ArrayLit, StructLit, TupleLit, UnsafeBlock, Loop,
Range, Return, Break, Quote, Splice, Modify, Field (lines 414-481).
But it **did not** touch the existing `A.If` arm at lines 358-365.
That arm reads:

```python
if isinstance(expr, A.If):
    new_cond, c_cond = _rewrite_in_expr(expr.cond, fn_by_name, new_fns)
    expr.cond = new_cond
    c_then = _rewrite_in_block(expr.then, fn_by_name, new_fns)
    c_else = 0
    if expr.else_ is not None and isinstance(expr.else_, A.Block):
        c_else = _rewrite_in_block(expr.else_, fn_by_name, new_fns)
    return (expr, c_cond + c_then + c_else)
```

`expr.else_` is `Optional[Block | If]` per `ast_nodes.py:220`. When
the user writes `if a { ... } else if b { grad(loss) }`, the parser
produces an `A.If` with `else_` = `A.If(...)` — and the arm above
silently treats this as `c_else = 0`, skipping the whole `else if`
recursion.

This is the **same family of bug** the C2-4 fix landed for —
walker dispatch that doesn't reach every reachable position. Worse,
the sibling walker `_resolve_in_expr` at line 218 DOES recurse into
the chained-If case correctly via `_resolve_in_expr(expr.else_, ...)`
— so the two walkers visibly disagree on chained-if dispatch.

The cycle-2 commit message says the fix "matches the same family of
fix applied to panic_pass / deprecated_pass / unsafe_pass in cycle
1" — but those passes use a unified `iter_expr` traversal that
handles chained-If symmetrically. `_rewrite_in_expr` does not.

**Reproducer** (verified):
```
@pure fn loss(x: f64) -> f64 { x * x }
fn main() -> i32 {
    if 1 > 0 { 0 }
    else if 2 > 0 { grad(loss)(3.14_f64) as i32; 1 }
    else { 2 }
}
```
Direct invocation of `grad_pass.grad_pass(prog)` returns `rewrite_count = 0`
and the `grad` Call is preserved unrewritten in main's body. Downstream
typecheck surfaces "unbound name 'grad'" — confusing, since this is
exactly the pattern C2-4 was supposed to fix.

**Hidden errors**:
- `grad(f)` / `grad_rev(f)` / `grad_rev_all(f)` in any chained
  `else if` branch silently never gets rewritten.
- User sees an "unbound name" error instead of the natural function
  call they wrote.
- Nested `else if` chains (3+ levels) compound — each level is
  skipped.

**Recommendation**:
1. Replace the `c_else = 0` branch with a unified recursion that
   handles both Block and If else-targets:
   ```python
   if expr.else_ is not None:
       if isinstance(expr.else_, A.Block):
           c_else = _rewrite_in_block(expr.else_, fn_by_name, new_fns)
       elif isinstance(expr.else_, A.If):
           new_else, c_else_inner = _rewrite_in_expr(
               expr.else_, fn_by_name, new_fns)
           expr.else_ = new_else
           c_else = c_else_inner
   ```
2. Add a regression test mirroring `_resolve_in_expr`'s coverage:
   `grad(loss)` in a chained `else if` body asserts the rewrite
   fires AND the original `grad` symbol is gone.
3. Consider unifying all three grad_pass walkers (`_expr_has_grad`,
   `_resolve_in_expr`, `_rewrite_in_expr`) behind a shared
   reflection-based child iterator — the same cycle-1 finding-6
   refactor proposal applies here.

**Trap-id**: n/a (analysis-pass; user-facing symptom is "unbound grad").

---

### Finding C3-2: `_widen_diff_inner` ranks `i64 == isize` (40 == 40) and `u64 == usize` (41 == 41) — `D<i64> + D<isize>` now spurious-warns TWICE (cycle-2 regression)

**Location**:
- helixc/frontend/typecheck.py:209-216 (`_WIDEN_RANK` table)
- helixc/frontend/typecheck.py:1230-1254 (tie callback emits AD002, then
  the outer `_ad_warn_mixed_inner` emits AD002 AGAIN at line 1252)
**Severity**: HIGH
**Category**: cycle-2-fix-introduced false-positive + double-emission
**Stage**: 28.8 cycle-2 commit 3a29728 (B:C1 + B:C4)

**Description**:
The C2-cycle B:C1+B:C4 rank reshuffle introduced **two** problems
for pointer-width pairs:

1. `i64` and `isize` both get rank 40; `u64` and `usize` both get
   rank 41. So `D<i64> + D<isize>` (or `D<u64> + D<usize>`) hits
   the same-rank-different-name branch and fires the `_warn_cb` tie
   callback at typecheck.py:243-246.

2. Worse, the wider call at line 1252 then ALSO calls
   `_ad_warn_mixed_inner` because `l_inner != r_inner`
   (different `TyPrim.name`). So a single `D<i64> + D<isize>`
   binop produces TWO warnings — one from the tie callback
   (with the same-rank-tie hint), one from the outer mismatch.

The cycle-2 commit message says ties "should be rare in practice
but the safety net stays." But `i64`/`isize` is the most common
pointer-width-arithmetic idiom — these aren't rare.

Pre-fix (cycle-1 state): `i64`/`isize` both ranked 40, same-name-
or-not, no callback path existed, so `D<i64> + D<isize>` silently
left-won with one B13 widening warn (correct user-facing behavior:
one warn for one mismatch). Post-fix: same path produces TWO warns,
with the second one accusing the user of "sign or quant domain
silently dropped" — but neither sign nor quantization changed.

**Reproducer** (verified via `python -m helixc.check --check-only`):
```
@pure fn use_d(a: D<i64>, b: D<isize>) -> D<i64> { a + b }
fn main() -> i32 { 0 }
```
Output:
```
   ad:        2 warning(s)
     helixc: 1:52: AD: D-binop with mixed inner types i64 vs isize — widened to i64 (trap 24200/AD002) (same-rank tie; sign or quant domain silently dropped without this warning)
     helixc: 1:52: AD: D-binop with mixed inner types i64 vs isize — widened to i64 (trap 24200/AD002)
```

**Hidden errors**:
- Every legitimate `D<i64> + D<isize>` binop (e.g., array-index
  derivative on 64-bit targets) emits two confusing warnings
  with false "sign/quant" attribution.
- Same for `D<u64> + D<usize>`.
- `-Wad=error` policy now FAILS the compile for benign pointer-
  width arithmetic in differentiable code.

**Recommendation**:
1. Recognize the i64/isize and u64/usize aliases — either give
   them identical names in `_WIDEN_RANK` lookups (treat
   `isize` → `i64`, `usize` → `u64` before rank lookup) OR add a
   pair-tolerance check in `_widen_diff_inner` that suppresses
   the tie callback when the pair is known-aliased.
2. De-duplicate the diagnostic: when the tie callback fires, the
   outer `_ad_warn_mixed_inner` should not also fire for the same
   span. Either gate the outer call on `_warn_cb is None` or
   thread a "did the callback fire" flag.
3. Regression test: `D<i64> + D<isize>` produces exactly ZERO
   warnings on a 64-bit target. `D<i32> + D<u32>` produces ZERO
   (rank 30 vs 31, no tie). `D<mxfp4> + D<nvfp4>` (both rank 43,
   different names) still produces ONE warn (the genuine quant-
   domain transition).

**Trap-id**: 24200 (existing).

---

### Finding C3-3: `check.main()` outer wrapper has no try/finally around `_main_inner` — any uncaught exception leaks `_DIFF_WARNINGS` and bypasses the C2-1 drain

**Location**:
- helixc/check.py:264-273 (outer `main` wrapper after C2-1)
**Severity**: HIGH
**Category**: cycle-2-fix-introduced silent window
**Stage**: 28.8 cycle-2 commit 5d23121 (C2-1)

**Description**:
The C2-1 fix split `main` into an outer wrapper (the drain
contract) plus `_main_inner` (the pipeline). The wrapper looks
like:

```python
def main(argv):
    _drain_ad_init()                       # clear stale state
    a_holder: list[CliArgs] = []
    rc = _main_inner(argv, a_holder)       # <-- unprotected
    if a_holder:
        drain_rc = _drain_ad_warnings(a_holder[0])
        if drain_rc != 0 and rc == 0:
            rc = drain_rc
    return rc
```

`_main_inner` is called WITHOUT a `try/finally`. Any uncaught
exception inside (typecheck crash, struct_mono bug, lower_ast
assertion, panic_pass throw, etc.) propagates straight out of
`main()` — and the drain at line 267 is NEVER reached.

This means:
1. The user sees a raw Python traceback (NOT a helixc error —
   cycle-1 Finding 9 partially fixed this for `--emit-asm` and
   `-o`, but did not cover the typecheck / struct_mono /
   flatten_impls / totality / deprecated / trace / panic /
   unsafe / autotune phases).
2. Any `_DIFF_WARNINGS` accumulated during typecheck before the
   crash are leaked into a follow-up `main()` invocation in the
   same process. The next call's `_drain_ad_init()` at line 263
   does clear it, so the leak is *bounded* — but any
   user-side reader of `_DIFF_WARNINGS` between calls sees
   stale state.

**Reproducer** (verified via direct Python-side crash injection):
```python
from helixc.frontend import autodiff, typecheck
from helixc import check
autodiff._DIFF_WARNINGS.append('stale warning from prior compile')
typecheck.typecheck = lambda prog: (_ for _ in ()).throw(RuntimeError("simulated typecheck crash"))
check.typecheck = typecheck.typecheck

rc = check.main(['some_file.hx'])    # <-- propagates RuntimeError
```
The exception escapes `main()` with `_DIFF_WARNINGS = []` (because
the leading `_drain_ad_init` cleared, then typecheck never ran to
re-populate). But in the realistic case where typecheck DOES write
warnings and a later pass crashes, the warnings are silently
discarded.

The C2-1 commit's stated invariant — "drain runs on every code
path that exits successfully from main()" — was implemented as
"every path through `_main_inner`'s `return` statements," NOT "every
exit from main()." Exception exits are a path the C2-1 design
missed.

**Hidden errors**:
- Compiler bugs (`AttributeError`, `KeyError`, `AssertionError`
  in any pipeline phase) leak Python tracebacks to the user.
- Same compiler bugs leak typecheck-stage `_DIFF_WARNINGS` (the
  B13 widening signal that the C2-1 fix was explicitly trying
  to surface) — exactly the symptom C2-1 was supposed to fix.
- Anyone composing `check.main(...)` from a longer-running
  harness sees ghost diagnostics on the next compile if they
  catch the exception themselves.

**Recommendation**:
1. Wrap the inner call in a `try/finally`:
   ```python
   rc = 1
   try:
       rc = _main_inner(argv, a_holder)
   except Exception as e:
       print(f"helixc: internal error: {type(e).__name__}: {e}",
             file=sys.stderr)
       print("helixc: this is a compiler bug — please file an issue.",
             file=sys.stderr)
       rc = 1
   finally:
       if a_holder:
           drain_rc = _drain_ad_warnings(a_holder[0])
           if drain_rc != 0 and rc == 0:
               rc = drain_rc
       else:
           _drain_ad_init()
   return rc
   ```
2. Cycle-1 Finding 9 fixed `--emit-asm` / `-o` paths — the same
   pattern should apply at the outer-wrapper level so ALL
   pipeline phases (typecheck, struct_mono, totality, deprecated,
   trace, panic, unsafe, autotune, grad_pass, lower, opts) inherit
   the same clean-error contract.
3. Regression test: monkey-patch a pipeline phase to raise,
   assert `main()` returns 1 with a clean stderr message (no
   Python traceback) and `_DIFF_WARNINGS` is empty after.

**Trap-id**: n/a (driver wiring).

---

## MEDIUM FINDINGS

### Finding C3-4: `monomorphize_structs` is invoked twice on the `-o` path (check.py + x86_64.__main__) — duplicate `Pt__i32` decls silently appended to `prog.items`

**Location**:
- helixc/check.py:358 (cycle-1 wiring of `monomorphize_structs`)
- helixc/backend/x86_64.py:3017 (existing CLI driver call)
- helixc/frontend/struct_mono.py:441 (append-without-dedup)
**Severity**: MEDIUM
**Category**: cycle-1-fix-introduced wiring (not regressed in cycle 2,
but cycle 2 added more pre-lowering passes that compound the cost)
**Stage**: 28.8 cycle-1 commit edf498e (A3/B1) + cycle-2 ambient

**Description**:
Cycle-1 commit edf498e wired `monomorphize_structs` into
`check.py` (line 358) so `--check-only` users see struct-mono
diagnostics. But `helixc/backend/x86_64.py:3017` already invokes
the same pass when its `__main__` runs (the legacy CLI driver). On
the `python -m helixc.check -o ... foo.hx` path, the user invokes
check.py (struct mono runs first) and then check.py imports
`compile_module_to_elf` from x86_64 — which currently does NOT
re-run struct mono (it takes a `tir.Module`). So check.py's path
is single-mono'd.

But anyone invoking `python -m helixc.backend.x86_64 foo.hx`
runs that legacy driver, which today does NOT call check.py's
pre-passes. Different harnesses → different pipelines.

The concrete silent window: if a future code path (or test
harness) chains both entry points, `prog.items` accumulates
duplicate `Pt__i32` StructDecls. Verified directly:

```python
prog, _ = monomorphize_structs(prog)   # appends Pt__i32
prog, _ = monomorphize_structs(prog)   # appends Pt__i32 AGAIN
# prog.items now has TWO Pt__i32 decls.
```

Current consumers (`lower_ast._struct_fields` dict, line 104)
silently dedup via dict assignment — so the duplicate is non-
corrupting today. But:
- Any future consumer that errors on duplicate-StructDecl
  silently breaks.
- The codegen's `compile_module_to_elf` doesn't see the
  duplicate (it takes a pre-lowered Module), but `lower_ast.lower()`
  iterates `prog.items` twice (once at line 102 for indexing,
  again at line 112 for dict-build) — the duplicate adds wasted
  work and creates a sticky surface for future bugs.

**Hidden errors**:
- Anyone composing check.py + x86_64.__main__ in the same process
  doubles the mono'd struct list. Silent today, silent-but-
  break tomorrow.
- Tests that count `prog.items` post-pipeline see N+1 or N+2
  entries depending on which driver ran.

**Recommendation**:
1. Add a guard: `monomorphize_structs` checks whether the mangled
   name already exists in `prog.items` before appending.
   ```python
   existing = {it.name for it in prog.items if isinstance(it, A.StructDecl)}
   for inst in mono_decls:
       if inst.name not in existing:
           prog.items.append(inst)
           existing.add(inst.name)
   ```
2. Alternative: mark `prog` with a `_struct_mono_done = True`
   sentinel; second call is a no-op.
3. Regression test:
   `monomorphize_structs(prog); monomorphize_structs(prog)` →
   exactly one `Pt__i32` in `prog.items`.

**Trap-id**: n/a.

---

### Finding C3-5: `_inline_lets` has no arm for `A.Cast` / `A.Call` / `A.Index` / `A.Field` etc. — let-bindings under those nodes silently unsubstituted, defeating cycle-2 C2-3's `%`-warn reach

**Location**:
- helixc/frontend/autodiff.py:473-548 (`_inline_lets` dispatch)
**Severity**: MEDIUM
**Category**: pre-existing reach gap exposed by cycle-2 fix
**Stage**: 28.8 cycle-2 commit 514165b (C2-3) [pre-existing gap]

**Description**:
The C2-3 fix added `_ad_warn` for unhandled Unary/Binary ops in
reverse-mode (e.g., `%`). The fix is correct in isolation — calling
`differentiate_reverse(Binary("%", x, IntLit(2)), ["x"])` emits the
warning as designed.

But the fix's effectiveness in real programs is REDUCED because
`_inline_lets` (which runs BEFORE `_propagate`) does not recurse
through several Expr subtypes:

- `A.Cast` — falls through to `return expr` at line 548 with
  inner Name NOT substituted.
- `A.Call`, `A.Field`, `A.Index`, `A.Match`, `A.Loop`, `A.For`,
  `A.While`, `A.UnsafeBlock`, `A.ArrayLit`, `A.TupleLit`,
  `A.StructLit`, `A.Range`, `A.Return`, `A.Break`, `A.Assign`,
  `A.Quote`, `A.Splice`, `A.Modify` — same.

The dominant Helix idiom for the C2-3 fix's reproducer is:
```
@pure fn loss(x: f64) -> f64 {
    let r = (x as i64) % 2_i64;
    r as f64
}
```
Verified: `differentiate_reverse(loss.body, ["x"])` returns
`{'x': FloatLit(0.0)}` with ZERO warnings — because `_inline_lets`
on the final_expr `Cast(Name("r"), f64)` doesn't recurse into the
Cast, so `Name("r")` is never substituted with `Binary("%",
Cast(Name("x")), IntLit(2))`. The Binary `%` is never reached by
`_propagate`; the warn never fires.

This is structurally identical to the cycle-1 walker-drift family
that panic_pass / deprecated_pass / unsafe_pass / grad_pass (in
cycle 2) were unified for. `_inline_lets` was not touched in either
cycle.

**Hidden errors**:
- The C2-3 fix's documented use case (the `% 2` derivative warning)
  silently does NOT fire when `%` appears under a Cast / Call /
  Field / Index / Loop / Match — i.e., almost every realistic
  position other than top-level Binary.
- Same for the C2-3 Unary warn — `!flag` or `~bits` under a Cast
  silently produces gradient 0 with no diagnostic.
- The cycle-2 deferred observation #18 (`_inline_lets` returning
  FloatLit(0.0) on empty Block) is fixed for the Block-no-final case,
  but the analogous "Cast we couldn't walk into" silently passes
  with no warning.

**Recommendation**:
1. Extend `_inline_lets` to recurse through every Expr subtype
   that can contain a Name leaf — at minimum: Cast, Call (callee +
   args), Field (obj), Index (callee + indices), Match (scrutinee +
   arm bodies), ArrayLit/TupleLit/StructLit (children), Range
   (start/end), Return/Break (value), Assign (target + value),
   UnsafeBlock (body), Loop/For/While (body + cond/iter), Quote/
   Splice (inner), Modify (target/transformation/verifier).
2. Add a final catch-all `else` branch that emits an `_ad_warn`:
   "_inline_lets fell through on Expr subtype 'X' — let-bindings
   beyond this point may not be substituted." Loud diagnostic on
   unknown nodes so future AST extensions surface immediately.
3. Regression test:
   `differentiate_reverse(loss.body, ["x"])` on the reproducer
   above asserts at least ONE warning fires for the unreachable
   `%` op.

**Trap-id**: 85001 (existing AD-assumed-zero).

---

### Finding C3-6: `monomorphize._fold_intlit_arith` silently returns the unfolded Binary on division-by-zero (`/` with rv=0) and modulo-by-zero (`%` with rv=0) — silent miscount default to 0 length

**Location**:
- helixc/frontend/monomorphize.py:78-90 (`_fold_intlit_arith`)
- helixc/ir/lower_ast.py:520-521 (`length = ... if isinstance(length_expr, A.IntLit) else 0`)
**Severity**: MEDIUM
**Category**: cycle-2-fix-introduced silent window
**Stage**: 28.8 cycle-2 commit 0e53eb8 (B:C11)

**Description**:
The B:C11 fix added `_fold_intlit_arith` to fold
`Binary(IntLit, op, IntLit)` shape expressions after substitution.
But the `/` and `%` operators are gated:

```python
if op == "/" and rv != 0:
    return A.IntLit(...)
if op == "%" and rv != 0:
    return A.IntLit(...)
return expr   # ← silent unfolded fallthrough on rv == 0
```

If a generic struct uses `[T; N / 0]` or `[T; N % 0]` after
substitution, the Binary stays unfolded. Downstream
`lower_ast.py:520-521` defaults `length = ... else 0` for any
non-IntLit shape, so:

```
struct Buf<const N: usize> { data: [i32; N / 0] }
```

silently compiles with `length = 0` and zero diagnostic. The
user wrote an obvious bug (divide-by-zero in a type-level
expression) and got a zero-length array instead of a clear
diagnostic.

Pre-fix (cycle-1 state): the unfolded Binary also fell through
silently — same end behavior. The cycle-2 B:C11 fix ADDED folding
for the well-defined cases but explicitly preserved the silent
path for div-by-zero. The fix's added value masks the silent
window further: most shape expressions now fold cleanly, so the
remaining silent path is rarer and less likely to be noticed.

Same applies to mod-by-zero. And tile/tensor shapes that hit
this path.

**Hidden errors**:
- `[T; N / 0]` silently compiles to `[T; 0]` — zero-length array.
- `[T; N % 0]` same.
- Codegen produces a buffer that can never be indexed; runtime
  bounds checks fail "out of range 0" with no upstream diagnostic.

**Recommendation**:
1. Extend `_fold_intlit_arith`'s `/` and `%` arms to raise a
   structured error on `rv == 0`:
   ```python
   if op == "/" and rv == 0:
       raise ValueError(
           f"{expr.span.line}:{expr.span.col}: division by zero "
           f"in shape expression (trap 28701)")
   ```
2. Wire the new trap-id 28701 in the trap-id table.
3. Catch the ValueError in `substitute_ty` callers and surface as
   a typecheck error (since shape evaluation runs at type-level).
4. Regression test: `[T; N / 0]` traps 28701 instead of silently
   producing length 0.

**Trap-id**: 28701 (new; needs reservation).

---

## LOW FINDINGS

(none new in this cycle)

---

## Cycle 2 fix re-verification

Each of the 12 cycle-2 commits was inspected for paper-only
fixes. All 12 are real fixes (the documented behavior is what
the code now does). The findings above are about *adjacent*
silent windows the cycle-2 fixes did not close, not about the
cycle-2 work being non-functional.

| Cycle-2 commit | Fix | Real? | C3 regression? |
|---|---|---|---|
| 5d23121 | C2-1 drain on every check.py exit | YES | **C3-3** (no try/finally around _main_inner) |
| 134df9b | C2-2 @trace early return → TRACE_EXIT | YES | none |
| 514165b | C2-3 reverse-mode warns on Unary/Binary | YES | **C3-5** (reach gap via _inline_lets) |
| 7a74acc | C2-4 grad_pass walkers cover Expr subtypes | YES | **C3-1** (chained else-if missed in _rewrite_in_expr) |
| 3a29728 | B:C1+B:C4+B:C6 widening + Quote + cast | YES | **C3-2** (i64/isize and u64/usize same-rank tie spurious-warns) |
| a086353 + 7682b14 | B:C2 closure trap on untyped lets + brace fix-up | YES | none — `let x = 1;` (i32 literal, tag 0) still passes cleanly |
| 0e53eb8 | C2-5 TyArray size + flatten_impls + shape fold | YES (partial) | **C3-4** (double-mono via two drivers); **C3-6** (div/mod-by-zero silent in fold) |
| a79e867 | B:C9 + B:C10 AD numeric set + Logic dedup | YES | none |
| 3126aec | deferred #17/#18/#19/#20 — depth + warns + narrow excepts | YES | none |
| 40f58ec | docs: trap-26001 also fires from _unflatten | YES (docs) | none |

Each cycle-2 fix's documented invariant is enforced by code.
The strict-clean criterion (zero new findings of ANY severity)
is the criterion that fails, not the cycle-2 work's correctness.

### Specific re-verifications from the audit instructions

- **5d23121 (drain on every exit)**: The drain runs on every
  `return` path of `_main_inner` AND clears stale state at outer
  entry. But the outer wrapper has NO `try/finally`, so exception-
  exits bypass the drain (Finding C3-3). The "every exit" claim
  is approximately true (it's "every *return-statement* exit").
- **134df9b (@trace early return)**: TRACE_EXIT emits before
  every `A.Return.value` IR-lowering site, mirroring the fall-
  through epilogue. State is push/restored via prev_/restore
  pair around `_lower_fn_body`. No `try/finally` — but there's
  no closure / inner-fn lowering yet that would benefit, so
  state-leak on exception is theoretical for now. Pass.
- **514165b (reverse-mode warns)**: `--` chains correctly chain
  via the `op == "-"` arm at each level — no false positive
  (verified directly). The cycle-2 fix is correct in isolation;
  see C3-5 for the reach gap.
- **7a74acc (grad_pass walkers)**: Coverage extended to most
  Expr subtypes; the chained-else-if hole (C3-1) is the
  remaining gap. `_resolve_in_expr` handles it correctly;
  `_rewrite_in_expr` does not.
- **3a29728 (typecheck cluster)**: `_widen_diff_inner` correctly
  picks the higher-rank type, but the same-rank-tie tolerance
  emits both the callback AND the outer warn (Finding C3-2). The
  Quote / Cast arms work; the `_check_cast_compat` recursion is
  correct for the `&Foo as &Bar` case but the (pre-existing)
  `&mut T as &T` mutability change still silently passes (not
  introduced in this cycle, but the cycle-2 fix sat in the same
  function and didn't close it).
- **a086353 + 7682b14 (closure trap on untyped lets)**: Verified
  that `let x = 1;` (AST_INT, tag 0) registers inferred_ty_tag=0
  (i32) and capture still passes cleanly. `let pi = 3.14_f64;`
  (AST_FLOATLIT_F64, tag 34) registers tag 2 (f64) so capture
  trap 76003 now fires loudly. Brace fix in commit 7682b14
  closes the 11-level nesting correctly.
- **0e53eb8 (TyArray size etc.)**: `_ty_key` distinguishes
  `[i32;4]` from `[i32;8]` (verified). `substitute_ty` routes
  TyArray.size through `_subst_shape_expr` and folds. Shape fold
  is correct for ops {+, -, \*, /, %} when `rv != 0`. See C3-6
  for the silent div/mod-by-zero path. The flatten_impls
  wiring into check.py is correct; the double-mono path
  (Finding C3-4) is a cycle-1 wiring concern compounded by C2's
  added pre-lowering passes.
- **3126aec (deferred observations)**: `_unflatten` depth check
  threads correctly. `_simplify` narrowed except is correct.
  `differentiate` narrowed except + emits warn on hash failure.
  `_inline_lets` empty-Block warn fires correctly. All verified.

---

## What was checked but found OK (no new finding)

- `_unflatten`'s depth threading is symmetric with
  `flatten_pytree` (depth+1 on recursion).
- The `_simplify` narrowed except catches `(OverflowError,
  ZeroDivisionError, ValueError, TypeError)` — covers the four
  arithmetic limits. Genuine bugs surface.
- `differentiate` narrowed except `(TypeError, ValueError,
  AttributeError)` + AD-warn on miss — perf regression no
  longer silent.
- `NUMERIC_FOR_AD` frozenset covers bool/char/fp8/mxfp4/nvfp4 —
  parity with typecheck's `_is_numeric_scalar`.
- Logic provenance grouped diagnostic correctly emits one
  message per call when 2+ params violate.
- `_resolve_in_expr` correctly handles chained `else if` via the
  recursive A.If arm at line 218-219 (contrast with
  `_rewrite_in_expr` — Finding C3-1).
- `flatten_impls` second invocation is a no-op (no ImplBlocks
  left after first call); does not double-rewrite method calls.
- @trace state restore (`prev_is_fn_traced` / `_current_fn_name`)
  uses prev/assign/restore pattern — exception-unsafe (no
  try/finally) but no lowering site between set and restore
  currently throws.
- `_rewrite_in_expr`'s A.Match recursion is symmetric with
  `_resolve_in_expr`.
- `--` chains do not false-positive in reverse-mode (verified).
- `let x = 1;` (i32 literal, untyped) still captures cleanly
  in parser.hx inferred-tag path.
- B:C2 11-level brace balance restored in commit 7682b14.
- 134df9b's TRACE_EXIT emission correctly handles `return;` (no
  value) via `const_int(0)` sentinel.
- Cycle-2 monomorphize is_checkpoint propagation (commit 7682b14)
  is correct (loss-fn slot 8 → clone slot 8).

---

## Deferred / out-of-scope observations (NOT new findings; documented for cycle-4 attention)

These are pre-existing silent gaps NOT introduced or changed by
cycle 2. They were observed during the audit and could become
cycle-4 candidates, but per the audit rules ("DO NOT re-flag
findings that prior audits already documented unless they CHANGED
in this cycle") they are not counted toward Cycle 3's tally.

- **`_check_cast_compat` `&mut T as &T` silent acceptance**: the
  C2-6 fix landed inner-type recursion but the inner-equal case
  (same `T`, different `is_mut`) still passes silently. Comment
  says "borrow-check concern" but no borrow-check exists yet.
- **`_DIFF_WARNINGS` is a module-level mutable list**: thread-
  unsafe; not relevant in current single-process Phase-0.
- **`_unflatten` `depth > MAX_DEPTH` check trap-id 26001**: text
  is symmetric with `flatten_pytree`, but the depth counter is
  threaded as a plain int parameter — a future caller passing
  a fresh `depth=0` could underflow the check. Minor.
- **`_inline_lets` for `A.If` wraps non-Block returns in a fresh
  Block** at line 543 — this is a structural transformation that
  silently rewrites the AST. Documented in the function comment;
  acceptable for Phase-0.

---

## Summary

| #    | Severity | Location                                                    | Finding                                                                                                   |
|------|----------|-------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| C3-1 | HIGH     | grad_pass.py:358-365 (A.If arm in `_rewrite_in_expr`)       | chained `else if` silently skipped — `grad()` nested there never gets rewritten                            |
| C3-2 | HIGH     | typecheck.py:209-216 + 1242-1253                            | `i64`/`isize` and `u64`/`usize` same-rank → spurious tie callback AND double-emission of AD002             |
| C3-3 | HIGH     | check.py:264-273 (outer `main` wrapper)                     | no `try/finally` around `_main_inner` → exceptions leak `_DIFF_WARNINGS` + bypass C2-1 drain               |
| C3-4 | MEDIUM   | check.py:358 + x86_64.py:3017 + struct_mono.py:441          | `monomorphize_structs` runs twice on some paths → duplicate `Pt__i32` decls accumulated in `prog.items`     |
| C3-5 | MEDIUM   | autodiff.py:473-548 (`_inline_lets` dispatch)               | no arm for Cast/Call/Index/Field/Match/etc. → cycle-2 C2-3 `%`-warn never fires in realistic programs       |
| C3-6 | MEDIUM   | monomorphize.py:78-90 + lower_ast.py:520-521                | `_fold_intlit_arith` silent on `/` and `%` by zero → shape silently becomes length 0                       |

**Total: 6 new findings (0 CRITICAL, 3 HIGH, 3 MEDIUM, 0 LOW).**

---

## Cycle 3 status

**Cycle 3 NOT clean.** Per the strict criterion (zero findings
of ANY severity), the 3 HIGH + 3 MEDIUM new findings BLOCK the
cycle-3 clean determination.

### Stop-the-line determination: **YES, on C3-1 and C3-3**.

C3-1 is a regression in the C2-4 fix's own scope: the cycle-2
commit explicitly addressed walker drift in grad_pass but left
the chained-if case unfixed in the very walker it was unifying.
The user-facing symptom (confusing "unbound name" error on
`grad(f)` in an `else if` body) is exactly the symptom C2-4 was
supposed to eliminate.

C3-3 is a regression in the C2-1 fix's own contract. The
commit's documented invariant ("drain runs on every exit") was
implemented as "drain runs on every return-statement exit" —
exception exits were missed. The fix introduced an outer wrapper
that's the natural place for the missing `try/finally`.

C3-2 is the most user-visible: every legitimate `D<i64> +
D<isize>` binop in differentiable code now emits two confusing
warnings with false "sign/quant" attribution. `-Wad=error` users
get FAILED compiles on benign pointer-width arithmetic.

C3-4, C3-5, C3-6 are MEDIUM. C3-4 is structurally dormant today
(consumers silently dedup); C3-5 reduces the reach of cycle-2's
C2-3 fix without breaking anything; C3-6 silently produces
zero-length arrays on div/mod-by-zero in shape exprs.

### Cycle 3 → NEW FINDINGS COUNT for the strict-clean gate: 6 (HIGH+MEDIUM) — clean-counter remains at 0.

### Estimated remaining open findings going into cycle 4

- Cycle 1: 13 new (all fixed → 0 open).
- Cycle 2: 6 new (all fixed → 0 open).
- Cycle 3: 6 new (all open).
- Prior audits (stage 5-6 + 7-8 + 9-16): 20 still-open at start
  of cycle 3 (unchanged from cycle 2 — none of the cycle-2 fixes
  touched the prior-audit scope).
- Cycle 3 net: 20 + 6 = **26 open findings** going into cycle 4 fixes.

Recommend prioritizing in this order for the cycle-4 fix batch:
1. C3-2 (false-positive spam in default-tooling output — most
   user-visible).
2. C3-1 (functional regression — chained else-if grad pattern).
3. C3-3 (silent traceback leak on compiler bugs).
4. C3-5 (reach gap — extend `_inline_lets`).
5. C3-6 (new trap 28701 for shape-time div-by-zero).
6. C3-4 (struct-mono idempotency guard).
