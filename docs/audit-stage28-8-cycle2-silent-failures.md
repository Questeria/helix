# Stage 28.8 Cycle 2 — Silent-Failure Audit

**Date**: 2026-05-10
**Commit**: 0171616 (read-only audit, isolated worktree)
**Scope**: All Helix source — `helixc/bootstrap/*.hx`, `helixc/frontend/*.py`,
`helixc/ir/*.py`, `helixc/backend/*.py`, `helixc/stdlib/*.hx`.
**Trigger**: pre-Stage-29 audit gate — Cycle 2 of 5. Re-audits same scope
after Cycle 1 fixes were landed (24 commits across Waves 1–3 +
the prior-audit bonus fix 505b4de). Heavy gate prior to this audit:
1359 tests passing, 1 skipped, 0 failed.
**Method**:
1. Walked the `git log --since=2026-05-09` (~24 commits) and read each
   fix's diff. For every code addition, traced data flow forward to see
   whether the fix opened a new silent window (regression).
2. Spot-checked `except Exception: pass`, `except TypeError: pass`,
   broad `except Exception:` clauses, `return TyUnknown(...)` /
   `return None` / `return 0` from validation-style helpers, and
   walker-style functions with fixed attr-name lists.
3. Verified each Cycle 1 fix that wired a validator into `check.py`
   is reachable (no dead validation re-introduced).
4. Cross-checked CLI driver `helixc/check.py` for new silent windows
   that the validator wirings might have introduced (e.g. accumulators
   that only get drained on a subset of code paths).
5. Re-read the 13 Cycle 1 findings against current source to confirm
   each is genuinely fixed (none "fixed" only on paper while the bug
   persists).

**Result**: **6 new findings (1 CRITICAL, 3 HIGH, 2 MEDIUM, 0 LOW)** —
Cycle 2 NOT clean. The dominant new pattern is **fix-introduced
silent windows**: the broad validation re-wiring that resolved Cycle 1
introduced its own gaps (e.g. AD warning drain only on lowering path,
TRACE_EXIT only on fall-through return, partial B13 coverage). A
secondary pattern is **incomplete reach** — Cycle 1's walker fixes
(panic_pass / deprecated_pass / unsafe_pass) standardized the attr
list, but the new B5 wiring of forward + reverse AD silently zeros
on Unary/Binary ops outside `+ - * / neg`, and `grad_pass._has_grad_call`
+ `_resolve_in_expr` walkers still miss many AST node kinds.

---

## CRITICAL FINDINGS

### Finding C2-1: AD warning channel (`_DIFF_WARNINGS`) only drained on lowering path — typecheck-emitted B13 widening warnings are silently lost when user runs without `--emit-*` or `-o`

**Location**:
- helixc/check.py:400-455 (drain inside `if any(f in a.flags ...) or a.output:` branch)
- helixc/check.py:396-398 (`--check-only` early return before drain)
- helixc/check.py:548-549 (`-- clean` return without drain)
- helixc/frontend/typecheck.py:1083 (`_ad_warn_mixed_inner` called during typecheck)
- helixc/frontend/typecheck.py:1791 (`_DIFF_WARNINGS.append`)
**Severity**: CRITICAL
**Category**: fix-introduced silent window / unreachable drain
**Stage**: 28.8 cycle-1 commit 1cb9961 (B13)

**Description**:
Cycle 1 commit 1cb9961 (B13) added a TyDiff mixed-inner widening warning
that emits during typecheck via `_ad_warn_mixed_inner` → appends to
`autodiff._DIFF_WARNINGS` (a module-level list). The drain logic at
check.py:446-455 calls `take_diff_warnings()` and prints them, optionally
escalating to error per `-Wad=error`.

The drain is INSIDE the conditional block at line 400:
```python
if any(f in a.flags for f in ("--emit-ir", "--emit-asm", "--emit-ptx")) \
        or a.output is not None:
    ...
    grad_pass(prog)
    mod = lower(prog)
    ...
    ad_warnings = take_diff_warnings()
    if ad_warnings: ...
```

For users who run `python -m helixc.check loss.hx` with no emit flags
and no `-o`, **the drain is never reached**. The B13 widening warnings
from typecheck silently accumulate in `_DIFF_WARNINGS` and are never
surfaced. Worse, `_DIFF_WARNINGS` is module-level state — those orphan
warnings persist into a subsequent compilation if the same process
runs typecheck again, producing bogus diagnostics on the wrong file.

The `--check-only` short-circuit at line 396-398 returns 0 before any
drain. Users specifically using `--check-only` to surface type errors
silently miss B13 warnings.

**Reproducer**:
```
# In loss.hx:
fn loss(x: D<f64>, y: D<i32>) -> D<f64> {
    x + y           // B13 should warn: mixed inner widened.
}
fn main() -> i32 { 0 }
```
`python -m helixc.check loss.hx` (no flags) → exit 0, no warning emitted.
`python -m helixc.check --emit-ir loss.hx` → warning correctly printed.

**Hidden errors**:
- The user gets silently-widened gradient types and no awareness.
- A second compile-step in the same process inherits dangling warnings
  from the first, attributing them to the wrong file.
- `--check-only` users (CI lint sweeps) silently bypass the whole
  B13 channel.

**Recommendation**:
1. Move the AD-warning drain outside the lowering branch. Either:
   a. Drain after every phase that may write to `_DIFF_WARNINGS`
      (right after typecheck, right after grad_pass).
   b. Drain at the top of `main()` exit — both clean and error paths.
2. Always call `take_diff_warnings()` (which clears the list) on
   compilation entry to prevent cross-compilation pollution.
3. Add a regression test: `subprocess.run(["python", "-m", "helixc.check",
   "loss.hx"])` where loss.hx has D<f64> + D<i32>; assert the warning
   appears on stderr.

**Trap-id**: 24200 (AD002), already reserved.

---

## HIGH FINDINGS

### Finding C2-2: `@trace` fn body's explicit `return X` (early returns) silently skip TRACE_EXIT — runtime trace pairs corrupt when enabled

**Location**:
- helixc/ir/lower_ast.py:544-555 (TRACE_EXIT only emitted before fall-through return)
- helixc/ir/lower_ast.py:1855-1858 (`A.Return` lowering emits `builder.ret(v)` with no TRACE_EXIT)
**Severity**: HIGH
**Category**: fix-introduced silent window / asymmetric event emission
**Stage**: 28.8 cycle-1 commit c418fb2 (A7)

**Description**:
Cycle 1 commit c418fb2 wired `@trace` codegen by emitting TRACE_ENTRY at
fn prologue (line 466) and TRACE_EXIT before the fall-through return
(line 554). The implementation:
```python
if is_fn_traced:
    ret_operand = body_val
    if isinstance(ir_fn.return_ty, tir.TIRUnit) or ret_operand is None:
        ret_operand = self.builder.const_int(0)
    self.builder.emit(tir.OpKind.TRACE_EXIT, ret_operand, ...)
# Emit return
if isinstance(ir_fn.return_ty, tir.TIRUnit):
    self.builder.ret(None)
elif body_val is not None:
    self.builder.ret(body_val)
```

But `A.Return` is lowered by a separate handler at line 1855-1858:
```python
if isinstance(expr, A.Return):
    v = self._lower_expr(expr.value) if expr.value is not None else None
    self.builder.ret(v)
    return None
```

No TRACE_EXIT emission. So `@trace fn f() -> i32 { if cond { return 1; }
2 }` emits TRACE_ENTRY at prologue, TRACE_EXIT before the fall-through
return-of-2, but NO TRACE_EXIT before the early `return 1`. When the
Stage-30 trace runtime exists, the buffer will see ENTRY-without-EXIT
on the early-return path, corrupting pair semantics.

Phase-0 backend currently emits TRACE_ENTRY/EXIT as nops (line 2487,
2501), so the user-visible impact today is zero. The silent window
will manifest the moment the runtime helper exists.

**Hidden errors**:
- ENTRY-without-EXIT in the trace stream confuses any tooling that
  pairs them (call-frame reconstruction, latency histograms).
- Tools assuming balanced pairs may double-count or miss exits.

**Recommendation**:
1. At the `A.Return` lowering site, check whether the enclosing fn
   carries `@trace` (via lowerer state) and emit TRACE_EXIT before
   `builder.ret(v)`.
2. Or, restructure so all return paths route through a single emit
   point — make `builder.ret` itself trace-aware when the current fn
   is traced.
3. Add a regression test in `test_trace.py`:
   `@trace fn f(x: i32) -> i32 { if x > 0 { return 1; } 2 }` —
   assert IR contains TWO TRACE_EXIT ops (one per return path).

**Trap-id**: n/a (codegen wiring; trap 25001 is the runtime overflow trap).

---

### Finding C2-3: `autodiff_reverse._propagate` silently skips Unary ops != `"-"` and Binary ops outside `{+, -, *, /}` — gradient drops with no warning

**Location**:
- helixc/frontend/autodiff_reverse.py:85-89 (`A.Unary` only matches `op == "-"`)
- helixc/frontend/autodiff_reverse.py:90-125 (`A.Binary` only matches `+ - * /`)
**Severity**: HIGH
**Category**: incomplete fix / fix-bypassing silent fallback
**Stage**: 28.8 cycle-1 commit b43d15c (B5)

**Description**:
Cycle 1 commit b43d15c (B5) added `_ad_warn` calls at unhandled-node
sites in both forward (`autodiff._diff`) and reverse (`autodiff_reverse
._propagate`) AD engines. The fix landed warnings for Quote/Splice/Modify,
Cast-to-non-numeric, opaque user calls, and the catch-all "unhandled
expression kind" branch at line 411.

But the Unary and Binary arms have NO warning for ops they don't handle:

Unary (line 85-89):
```python
if isinstance(node, A.Unary):
    if node.op == "-":
        neg = A.Unary(span=node.span, op="-", operand=adj)
        _propagate(node.operand, neg, acc)
    return            # ← silent return for ! ~ & * deref
```

Binary (line 90-125):
```python
if isinstance(node, A.Binary):
    l, r, op = node.left, node.right, node.op
    if op == "+":
        ...
    elif op == "-": ...
    elif op == "*": ...
    elif op == "/": ...
    # Other ops (comparisons, etc) have zero local derivative...
    return            # ← silent return for % == != < <= > >= && || & | ^
```

The line-124 comment says "Other ops have zero local derivative for our
cases" — but emits NO `_ad_warn`. So `grad_rev(fn(x) -> i32 { x % 2 })`
silently returns 0 with no diagnostic, even though the user almost
certainly meant something. Same for `grad_rev` of expressions containing
bitwise ops, comparisons in non-comparison-result positions, etc.

The forward-mode `_diff` in autodiff.py:520-523 has the same pattern:
```python
if isinstance(expr, A.Unary) and expr.op == "-":
    return A.Unary(span=span, op="-", operand=_diff(expr.operand, var))
if isinstance(expr, A.Binary):
    ...
```
But forward-mode does fall through to the final "Genuinely-unknown — warn
loudly" branch at line 591-593 if Binary doesn't match `+ - * /`. Wait —
actually forward-mode handles non-arithmetic ops correctly: line 545's
`if isinstance(expr, A.If):` then 553 Block, then 555 Call, then 569
Cast, etc. So for `A.Binary` with `op="%"`, the fall-through reaches
line 591-593 which DOES warn. So forward-mode is fine; only reverse-mode
silently zeros these.

The asymmetry is itself a silent window: forward and reverse gradients
of the same expression produce different diagnostic landscapes.

**Hidden errors**:
- `grad_rev` of any expression touching `%`, comparison operators in
  non-comparison-result positions, or bitwise ops silently produces
  zero gradients for that operand.
- Bare Unary like `!flag` (boolean NOT) or `~bits` (bitwise NOT) inside
  an AD'd expression silently contributes zero with no diagnostic.

**Reproducer**:
```
@pure fn loss(x: f64) -> f64 {
    // Reverse-mode silently zeros the % derivative.
    let r = x as i64 % 2_i64;
    r as f64
}
fn main() -> i32 { grad_rev(loss)(3.14_f64); 0 }
```
Expected: AD warning that `%` is non-differentiable.
Actual: no warning, gradient silently 0.

**Recommendation**:
1. In `autodiff_reverse._propagate`, add an `else` branch in each of the
   Unary and Binary arms that calls `_ad_warn(node, f"op {op!r} has no
   defined local derivative")`.
2. Symmetrize forward and reverse — both modes should warn on the same
   set of node kinds.
3. Regression test: `grad_rev` of an expression containing `%` asserts
   a warning is emitted.

**Trap-id**: 85001 (already reserved per autodiff.TRAP_AD_ASSUMED_ZERO).

---

### Finding C2-4: `grad_pass._rewrite_in_expr` and `_resolve_in_expr` walkers miss many AST node kinds — `grad(loss)` nested in Field/Match/Loop/UnsafeBlock/StructLit/etc. silently never rewritten

**Location**:
- helixc/frontend/grad_pass.py:208-320 (`_rewrite_in_expr`)
- helixc/frontend/grad_pass.py:149-182 (`_resolve_in_expr`)
- helixc/frontend/grad_pass.py:40-84 (`_expr_has_grad`)
**Severity**: HIGH
**Category**: fixed-attr walker / silent miss
**Stage**: pre-28.8 (not touched in cycle-1, but relevant for completeness)

**Description**:
The grad_pass walkers `_rewrite_in_expr` (line 208), `_resolve_in_expr`
(line 149), and `_expr_has_grad` (line 40) all use hand-rolled node-type
dispatch. The dispatch lists are incomplete relative to `ast_nodes.py`.

`_rewrite_in_expr` handles: Call, Binary, Unary, Block, If, Match, Cast,
Assign, Index, While, For. Misses:
- **A.Return** — no recursion → `return grad(f)` is NOT rewritten.
- **A.Break** — no recursion → `break grad(f)` not rewritten.
- **A.UnsafeBlock** — no recursion → `unsafe { grad(f) }` not rewritten.
- **A.Field** — no recursion → `grad(f).method()` not rewritten.
- **A.StructLit / A.TupleLit / A.ArrayLit** — no recursion → `Pt {
  x: grad(f) }` not rewritten.
- **A.Range** — no recursion → `0..grad(f)` not rewritten.
- **A.Loop** — no recursion → `loop { grad(f) }` not rewritten.
- **A.Quote / A.Splice / A.Modify** — no recursion (perhaps deliberate
  for reflection, but worth a warning).

`_resolve_in_expr` (alias resolution) misses the same set PLUS A.Match,
A.Loop, A.Field, A.Return, A.Break, A.Assign, A.Range, A.StructLit,
A.TupleLit, A.ArrayLit, A.UnsafeBlock, A.Quote/Splice/Modify.

`_expr_has_grad` (used as a short-circuit predicate at line 35) misses:
A.Field, A.Index, A.StructLit, A.TupleLit/ArrayLit, A.UnsafeBlock,
A.Quote/Splice/Modify, A.Range, A.Break.

Even though `_expr_has_grad` is "just" a short-circuit predicate
(returning False conservatively just means "don't skip; do the walk
anyway"), the silent miss in `_rewrite_in_expr` is real — the actual
rewrite step. A user-written `Pt { gradient: grad(loss) }` silently
never gets the rewrite; `grad` then surfaces as an unbound name during
typecheck (or as a call to a non-existent fn during lowering).

The walker pattern is the same family as panic_pass / deprecated_pass /
unsafe_pass — and the panic-pass fix in commit 5516935 unified the attr
list across those three. `grad_pass` was not touched by Cycle 1 fixes
and its walker drift is now structurally identical to the bug that
panic_pass had pre-fix.

**Hidden errors**:
- `grad(loss)` inside a struct literal: `Optim { lr: 0.01, grad: grad(
  loss) }` silently never rewritten.
- Same for `[grad(loss_a), grad(loss_b)]` array.
- Same for `return grad(loss)` in a multi-return fn body.
- Same for `unsafe { grad(loss) }` (acknowledged-unsafe AD).

**Reproducer**:
```
@pure fn loss(x: f64) -> f64 { x*x }
fn main() -> i32 {
    // Silently never rewritten — grad symbol surfaces as unbound.
    let arr = [grad(loss), grad(loss)];
    0
}
```
Expected: array of two function pointers to `loss__grad`.
Actual: compile error "unbound name 'grad'".

**Recommendation**:
1. Either share a single reflection-based walker across grad_pass,
   panic_pass, deprecated_pass, unsafe_pass (would benefit all four),
   OR extend grad_pass's fixed attr lists to cover the full set of
   AST nodes that can contain Expr subtrees.
2. Add regression tests for grad nested in: StructLit, ArrayLit, Range,
   Return, UnsafeBlock.

**Trap-id**: n/a (analysis-pass; user-facing symptom is "unbound grad").

---

## MEDIUM FINDINGS

### Finding C2-5: `struct_mono._ty_key` for `A.TyArray` excludes the size — `Pt<[i32; 4]>` and `Pt<[i32; 8]>` silently dedup to one mono'd struct

**Location**:
- helixc/frontend/struct_mono.py:330-331 (`("arr", _ty_key(t.elem))` — no size)
**Severity**: MEDIUM
**Category**: incomplete fix from cycle 1 A13
**Stage**: 28.8 cycle-1 commit 37c655c (A13)

**Description**:
Cycle 1 commit 37c655c (A13) fixed `_ty_key` collapses for TyFn / TyTensor
/ TyTile. The fix added proper arms encoding params+ret, dtype+shape+
device, dtype+shape+memspace. But `A.TyArray` was left untouched (still
existed pre-cycle-1 as well):
```python
if isinstance(t, A.TyArray):
    return ("arr", _ty_key(t.elem))
```
TyArray has both `elem` (TyNode) AND `size` (Expr per ast_nodes.py:47).
The size is excluded from the key. So `[i32; 4]` and `[i32; 8]` produce
the same key — silently dedup'd by the struct mono pass.

This is the same family of collapse the A13 fix was supposed to close:
`Pt<[i32; 4]>` and `Pt<[i32; 8]>` are semantically distinct (different
element counts → different struct layouts) but mono pass emits only one
of them, applying its size to both.

The audit-stage28-8-cycle1-silent-failures Finding 13 documented this
class of issue as MEDIUM. The cycle-1 fix listed TyFn/TyTensor/TyTile/
TyMemTier — TyMemTier isn't an AST type (it's typecheck-only) and the
TyArray fix was missed entirely.

**Hidden errors**:
- `Pt<[i32; 4]>` and `Pt<[i32; 8]>` collapse to one mono. Whichever
  field-list ends up in the mono'd struct silently determines the
  layout for BOTH; whichever is wrong gets garbage at codegen.

**Reproducer**:
```
struct Holder<T> { x: T }
fn use_arrays() -> i32 {
    let h_small: Holder<[i32; 4]> = Holder { x: [0, 0, 0, 0] };
    let h_large: Holder<[i32; 8]> = Holder { x: [0, 0, 0, 0, 0, 0, 0, 0] };
    // Mono emits ONE Holder__arr; h_small or h_large reads wrong layout.
    0
}
```

**Recommendation**:
1. Extend the TyArray arm to include the size key:
   ```python
   if isinstance(t, A.TyArray):
       return ("arr", _ty_key(t.elem), _shape_key(t.size))
   ```
2. Add a regression test:
   `test_ty_key_distinguishes_tyarray_size`.

**Trap-id**: n/a.

---

### Finding C2-6: `typecheck._check_cast_compat` unconditionally accepts `TyRef as TyRef` regardless of inner-type compatibility — `&Foo as &Bar` silently succeeds

**Location**:
- helixc/frontend/typecheck.py:1769-1772 (TyRef-to-TyRef early return)
- helixc/frontend/typecheck.py:1748-1782 (`_check_cast_compat`)
**Severity**: MEDIUM
**Category**: fix-introduced silent window from cycle 1 B14
**Stage**: 28.8 cycle-1 commit 1cb9961 (B14)

**Description**:
Cycle 1 commit 1cb9961 (B14) added the allowed-cast matrix
`_check_cast_compat` (typecheck.py:1748). The matrix correctly rejects
tuple→i32, struct→f64, unit→Pt at trap 28604. But the TyRef arm at
line 1769-1772 is:
```python
# Ref <-> Ref of compatible inner is OK at the type level (a
# change in is_mut is a separate concern handled by borrow-check).
if isinstance(src, TyRef) and isinstance(tgt, TyRef):
    return
```
The comment claims "compatible inner" but the check is unconditional —
**any TyRef can cast to any TyRef regardless of inner**. So `&Foo as
&Bar` (entirely unrelated struct refs) silently typechecks, and
`&[i32; 4] as &Foo` (array ref to struct ref) silently typechecks.

This is a regression introduced by B14: pre-fix, the Cast handler
accepted everything (no validation at all). Post-fix, most casts get
the matrix BUT the TyRef path skips inner-type compatibility. The
matrix purports to be a closed system but has this open window.

Additionally, the `_WIDEN_RANK` table in typecheck.py:197-205 includes
`f16/bf16`, `f32`, `f64`, but `_NUMERIC_FLOAT_PRIMS` (line 1736-1738)
also includes `fp8`, `mxfp4`, `nvfp4` — which are NOT in `_WIDEN_RANK`.
So `D<fp8> + D<mxfp4>` (both rank -1) silently picks left at line 221
of `_widen_diff_inner` with NO B13 warning. The matrix B14 accepts the
cast, B13 silently picks one side.

**Hidden errors**:
- `&Foo as &Bar` silently typechecks — codegen produces undefined behaviour
  reading the bar layout out of a foo-pointed buffer.
- `D<fp8> + D<mxfp4>` silently picks left's inner, no widening warn.
  User has no signal that the two fp8/mxfp4 didn't unify.

**Reproducer**:
```
struct Foo { x: i32 }
struct Bar { y: f64 }
fn bad(f: &Foo) -> &Bar {
    f as &Bar     // silently typechecks — should trap 28604.
}
```
Expected: cast error.
Actual: clean compile.

**Recommendation**:
1. Tighten the TyRef arm: only accept the cast if inner types are
   `_compatible`. Falling back to trap 28604 for unrelated refs.
2. Add `fp8` / `mxfp4` / `nvfp4` to `_WIDEN_RANK` (e.g. rank 45 between
   int and bf16, or rank 50 alongside f16/bf16).
3. Regression tests:
   `test_cast_unrelated_ref_to_ref_rejected` —
   `&Foo as &Bar` produces trap 28604.

**Trap-id**: 28604.

---

## What was checked but found OK (no new finding)

- **Pytree cycle guard (B9 (1))**: `flatten_pytree`, `pytree_depth`,
  `_unflatten` all thread `_visited` via `_visited | {decl.name}` —
  new sets per frame, so siblings don't pollute. Each top-level call
  initializes fresh `_visited`. Cycle guard correctly re-entrant.
- **Trap-id 24100 reassignment (A4)**: `kovc.hx:4220` still emits 24001
  for bf16-MOD; typecheck.py docstring + test_provenance.py + diagnostics
  examples all updated to 24100. No remaining collision.
- **Stdlib merge (A8)**: all named-item kinds now merged with proper
  conflict handling. Missing-file emits stderr warning; HELIXC_STDLIB_
  STRICT promotes to FileNotFoundError. ImplBlocks without names are
  always appended (no dedup) — Phase-0 acceptable.
- **Panic codegen TRAP op (A1)**: writes message to stderr, sys_exits
  with `trap_id & 0xFF`, ud2 belt-and-braces. Trap id truncation to
  byte means 28501 / 28245 / 28245+256k all yield 0x55 exit code — a
  documented limitation, not silent (the stderr message includes the
  full trap_id).
- **Unsafe-depth tracking (B3)**: `_in_unsafe_depth` counter pushed/popped
  via try/finally. Exception-safe; even if `_check_block` raises, the
  decrement happens.
- **`emit_warnings` no longer monkey-patches Program (C1-M1)**: returns
  list, caller stores locally in check.py. Multiple invocations are
  now well-defined.
- **`-O1/-O2/-O3` opt-level wiring (A10)**: fdce + fold at -O1; +cse +
  dce at -O2; -O3 alias. Help text matches.
- **Autotune dedup + diagnostics (A12)**: malformed attrs now produce
  diagnostics instead of silent continue. `validate_autotune_prog`
  wired into check.py. Per-key dedup applied before Cartesian product.
- **`closure capture non-i32` loud failure (B4)**: a captured var with
  `var_type_tab_lookup` returning `> 0` (tracked non-i32) emits
  AST_ERR(76003). Note: untracked vars (tag == -1) still silently pass
  as i32 — acknowledged as deferred in the commit message; the loud
  path covers the most common typed-let cases.
- **Substitute_ty TyPtr + size-param shape sub (B6/B8)**: TyPtr now
  preserved with is_mut + inner substituted. TyTile/TyTensor shape
  `Name(N)` → `IntLit(value)` via _SizeLitMarker sentinel.
- **Instantiate is_extern + where_clauses (B7)**: extern flag + abi
  propagated; where_clauses deep-copied with substitute_ty applied
  per constraint.
- **TyTile call-site shape + memspace check (B8)**: trap 16003 fires on
  rank, dim-size, or memspace mismatch. Pre-fix silently accepted.
- **Quote/Splice/Modify typecheck (B10)**: Quote returns TyQuote(inner);
  Splice unwraps or trap 11001; Modify returns i32.
- **flatten_impls duplicate method name (B11)**: raises
  DuplicateMethodError(trap 74002) — first-target-wins silent collapse
  closed.
- **UnsafeBlock subst walker (B12)**: `_walk_subst_expr` now recurses
  into UnsafeBlock.body; generic-T inside `unsafe { let x: T = ... }`
  correctly substituted.
- **Panic/deprecated/unsafe walker fixes (C1-H1, A5, A6)**: all three
  walkers now use the canonical `then`/`else_`/`iter_expr`/`obj`/...
  attr list; `except TypeError: pass` replaced with `raise`. No
  remaining drift across the three walkers.

---

## Status of cycle 1 findings

Re-verified each of the 13 cycle 1 findings against current source.
All 13 are genuinely fixed (no paper-fix-only) — though three of them
acquired the regression silent windows documented in this audit's
Findings C2-1 / C2-2 / C2-6.

| # | Cycle 1 Finding | Fix commit | Real fix? | Cycle 2 regression introduced? |
|---|---|---|---|---|
| 1 | panic("msg") non-functional | 62a461c | YES | none |
| 2 | unsafe gate non-functional | 1981594 | YES | none |
| 3 | struct mono body walk | edf498e | YES | none (TileLit gap pre-existing) |
| 4 | trap-id 24001 double-claim | 7287b74 | YES | none |
| 5 | deprecated walker gaps | 490bd1d | YES | none |
| 6 | panic walker if/else gap | 5516935 | YES | none |
| 7 | @trace non-functional | c418fb2 | YES (codegen wired) | **C2-2** (early return skips TRACE_EXIT) |
| 8 | stdlib drop non-fn items | 2189df1 | YES | none |
| 9 | --emit-asm / -o tracebacks | d026d4f | YES | none |
| 10 | -O2/-O3 silently == -O1 | 59c525d | YES | none |
| 11 | pytree unflatten zero-fill | b091a63 | YES | none (depth check still asymmetric — see "MEDIUM-NOT-PROMOTED") |
| 12 | autotune malformed attrs | 1c12e8f | YES | none |
| 13 | _ty_key TyFn/Tile/Tensor collapse | 37c655c | YES (partial: TyArray still collapses on size) | **C2-5** (TyArray size excluded) |

Plus bonus stage7-8 F4 fix (505b4de) for turbofish mr_tab overflow — verified, no regression.

---

## MEDIUM-NOT-PROMOTED observations (deferred, not blocking)

These would be MEDIUM in isolation but are deferred because (a) the
silent window is gated behind another unimplemented feature, or (b)
the impact is bounded and documented:

- **`_unflatten` asymmetric depth check**: `flatten_pytree` raises
  trap 26001 when depth > MAX_DEPTH=4; `_unflatten` has no depth check
  and would `RecursionError` instead of a clean trap. Test would need
  to construct a >4-deep struct without cycles. Acceptable for Phase-0
  given the `MAX_DEPTH=4` cap on `flatten` already gates the same shape.
- **`autodiff._inline_lets` Block with `final_expr=None`**: returns
  `FloatLit(0.0)` at line 477 with no diagnostic. The body's stmts
  were inlined into env but the final value defaults to 0. Edge case;
  fires only when a block in an AD context has stmts but no value.
- **`autodiff._simplify` broad `except Exception: pass`** at line 772-773:
  silently swallows arithmetic errors during constant-fold simplification.
  The unsimplified expression flows through unchanged; not a corruption
  but a silent fallback. Could narrow to (OverflowError, ZeroDivisionError,
  ValueError, TypeError).
- **`autodiff.differentiate` broad `except Exception: key = None`** at
  line 125-126: if `structural_hash` crashes, cache is silently bypassed
  with no diagnostic. Hash failures are rare but a future AST extension
  could trigger this silently — perf regression with no signal.
- **`_widen_diff_inner` rank-(-1) tie**: when both sides are TyPrim with
  names not in `_WIDEN_RANK` (e.g. both `fp8`, both `mxfp4`), both
  rank-(-1) and `a if ra >= rb else b` arbitrarily picks `a`. Either
  add fp8/mxfp4/nvfp4 to the rank table or fail loudly when rank is
  -1.
- **B13 widening only fires when BOTH sides are D-wrapped**: `D<f64> +
  f32` does NOT trigger widening warn (one side is bare). The
  asymmetric coercion (f32 → f64 because `_unwrap(l) = f64` wins) is
  silent. Pre-fix behavior preserved; B13 only catches the doubly-D
  case.
- **struct_mono `visit_expr` doesn't handle `A.TileLit`**: `tile<Pt<i32>,
  [4,4], REG>::zeros()` silently misses the Pt<i32> use. Minor —
  TileLit dtype is typically a primitive in practice.

---

## Summary

| #  | Severity  | Location | Finding |
|----|-----------|----------|---------|
| C2-1 | CRITICAL | check.py:400 | AD warning drain only on lowering path; typecheck-emitted B13 warnings lost on `--check-only` / no-emit |
| C2-2 | HIGH | lower_ast.py:1855 | `@trace` fn early `return` silently skips TRACE_EXIT op |
| C2-3 | HIGH | autodiff_reverse.py:85,90 | `_propagate` silently zeros Unary != "-" and Binary outside {+ - * /} with no warning |
| C2-4 | HIGH | grad_pass.py:208,149 | `_rewrite_in_expr` / `_resolve_in_expr` miss Field/Match/Loop/UnsafeBlock/StructLit/etc. — silent grad miss |
| C2-5 | MEDIUM | struct_mono.py:330 | `_ty_key` for TyArray omits size — `[i32;4]` and `[i32;8]` dedup to one mono |
| C2-6 | MEDIUM | typecheck.py:1769 | `_check_cast_compat` unconditionally accepts `&Foo as &Bar` regardless of inner types |

**Total: 6 new findings (1 CRITICAL, 3 HIGH, 2 MEDIUM, 0 LOW)**.

---

## Cycle 2 status

**Cycle 2 NOT clean.** With 1 CRITICAL + 3 HIGH new findings, this
cycle does not meet the "zero new HIGH/CRITICAL" criterion. The
clean-counter for the 5-clean gate remains at 0.

### Stop-the-line recommendation: **YES, on C2-1**.

C2-1 specifically is a fix-introduced regression where the new B13
widening channel is silently swallowed on the most common compile
path (`helixc/check.py loss.hx`, no flags). Until drained, the entire
B13 + B5 warning infrastructure is unreachable from the default user
path. This actively HIDES the cycle-1 fix's value from users and
masks real diagnostics in normal day-to-day use.

C2-2 (TRACE_EXIT skipped on early return) is deferred-impact (Phase-0
backend emits TRACE_ENTRY/EXIT as nops) but the wiring is now wrong
and would corrupt trace pairs the moment Stage 30 runtime exists.

C2-3 (reverse-mode AD silent zeros) and C2-4 (grad_pass walker gaps)
are correctness bugs of the same family as cycle 1's panic_pass walker
drift — same root cause, same family of fix. The walker unification
recommended in cycle 1's Finding 6 ("share a single reflection-based
walker") would close both at once.

C2-5 and C2-6 are MEDIUM partial-fix gaps from cycle 1's matrix work.
Mechanical fixes; add to the next fix batch.

### Cycle 2 ⇒ NEW FINDINGS COUNT for the gate's clean-counter logic: 4 (CRITICAL + HIGH only) — clean-counter remains at 0.

### Estimated remaining open findings going into cycle 3

Cycle 1: 13 new (all fixed → 0 open from cycle 1)
Cycle 2: 6 new (0 open)
Prior audits (stage 5-6 + 7-8 + 9-16): 21 still-open at start of cycle 1.
  - 1 closed by 505b4de (stage7-8 F4) during this gate
  - 20 still-open
Cycle 2 net: 20 + 6 = **26 open findings** going into cycle 3 fixes.

After cycle 2 fixes land (C2-1 priority): another audit cycle required.
Recommend prioritizing C2-1 fix (1 line move in check.py) + C2-3 +
C2-4 (walker unification refactor) before cycle 3 audit.
