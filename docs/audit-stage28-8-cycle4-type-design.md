# Stage 28.8 Pre-29 Audit Gate — Cycle 4, Audit B: Type-System / Dispatch / Soundness

**Date**: 2026-05-11
**Commit**: b3504a2 (read-only) — post cycle-3 fix sweep
**Scope**: Re-audit type-system soundness focused on the 15 cycle-3
fixes (commits 025d55e, c31158c, ee7aa42, 74b72ec, 3358627, dccfc7e,
2b15928, 3b321e6, a878709, dda3b9d, b3504a2). New widened `_compatible`
(TyDiff / TyLogic / TyTuple / TyArray / TyRef / TyPtr / TyFn structural
arms), D1 non-prim call-boundary check, D2 closure-trap call-RHS
sentinel tag 12, D3 trap 28802 for TyArray size <= 0, D4 Logic-domain
AD002, D5 `_fold_intlit_unary`, D6 `_ty_key` strict guard, D7 iterative
`_check_cast_compat` + depth-8 guard (trap 28803), D8 `_fmt(TyStruct)`,
D9 turbofish re-substitution, C3-2 `_WIDEN_NAME_ALIASES` for
isize/usize. Cross-stage interactions
(call-boundary-vs-substitute-vs-mono, RecursionError parity between
`_check_cast_compat` and `_compatible`, ShapeFoldError catch contract,
Logic vs D arm symmetry, trap 28801/28802/28803 reachability).

**Method**:
1. Traced each cycle-3 fix's diff against the cycle-2 issue it was
   meant to close. Re-read the cycle-3 audit B doc to identify the
   exact gap each fix targeted.
2. Walked `_compatible`, `_check_call_basic`, `_check_cast_compat`,
   `_resolve_size_expr`, `_fold_intlit_arith`, `_fold_intlit_unary`,
   `_walk_subst_expr`, `parser.hx::parse_let` (let-inference site +
   capture-site probe) at b3504a2.
3. Built Python probes against `helixc.frontend.typecheck` and
   `helixc.frontend.monomorphize` to exercise each cycle-3 fix at the
   boundary (verified cycle-3 fixes behave as documented; identified
   probes that *should* succeed but emit unexpected diagnostics).
4. Cross-checked the user's four cycle-4 focus questions: `_compatible`
   structural-recursion termination on all legitimate inputs; trap
   28801/28802/28803 reachability from legitimate user code;
   Logic-domain AD002 span sharing with D-domain variant; D2 call-RHS
   inference defer-vs-trap for forward-declared fns.
5. Cross-checked the user's cycle-3 fix re-verification questions (D1
   nested generic combinations / cyclic refs, D2 i32-returning Call
   RHS, D3 size positions, D4 wrapping identification, D5 cascaded
   negations, D6 TypeError surface, D7 legitimate-but-deep refs, D8
   parametric struct mangled names, D9 nested-turbofish, C3-2 alias
   pair completeness).
6. Ran the codegen test for the D2 regression
   (`test_bootstrap_kovc_full_pipeline_arithmetic`) and confirmed it
   asserts the over-trap.

**Result**: **10 new findings (1 CRITICAL, 0 HIGH, 4 MEDIUM, 5 LOW)**.
The CRITICAL is a functional regression: cycle-3 D2's parser tag-12
sentinel fires on ALL Call-RHS untyped lets, including `let n =
i32_returning_fn();`, causing trap 76003 (SIGILL) at any subsequent
closure capture of `n` — breaking the most common Phase-0 idiom.
The cycle-3 D2 regression test in `test_codegen.py:3471-3477`
codifies the broken behavior (assertion: `== 132`) which calcifies
the false-positive into the test suite. The four MEDIUM findings are
next-layer-down asymmetries cycle-3's structural-`_compatible`
widening (D1) introduced: TyArray size compared by raw `==` instead
of `_compatible` (false-positive for TySize / TyUnknown sizes);
Logic-wrap-asymmetric still silent (`Logic<f64> + f64`);
ShapeFoldError caught only by struct_mono, not fn-mono (escapes as
"compiler bug"); `_compatible` itself is recursive-unbounded
(parallel to the D7 RecursionError vulnerability that D7 closed for
`_check_cast_compat`). LOW findings cover diagnostic-text drift in
D3, namespace overlap in the parser tag-12 sentinel, parametric
struct mangled-name leak in `_fmt`, the "D-binop" text-prefix bleed
into Logic-domain warns, and absence of trap 28803 reachability from
any source-level construction (defensive but practically unreachable).

The CRITICAL is stop-the-line; the four MEDIUMs are real contract
drifts in the structural-equality invariant. **Cycle 4 status**: 10
findings (1 CRITICAL, 0 HIGH, 4 MEDIUM, 5 LOW) means cycle 4 does
**NOT** count clean under the strict criterion. See "Cycle 4 status"
final paragraph.

---

## Summary table

| ID  | Severity | Component                                      | Issue (short)                                                                          |
|-----|----------|------------------------------------------------|----------------------------------------------------------------------------------------|
| F1  | CRITICAL | bootstrap parser.hx D2 tag-12 sentinel         | Tag 12 fires on ALL Call-RHS lets; even `let n = i32_returning_fn()` traps on capture |
| F2  | MEDIUM   | typecheck `_compatible` TyArray arm            | `a.size == b.size` raw eq breaks cascade-safe / generic-defer for TySize/TyUnknown    |
| F3  | MEDIUM   | typecheck binop Logic-domain warn              | `Logic<f64> + f64` (wrap-asymmetric) silent — D4 closed only Logic-Logic mixed-inner  |
| F4  | MEDIUM   | monomorphize / x86_64 driver                   | ShapeFoldError not caught by `monomorphize()`; escapes to top-level as "compiler bug" |
| F5  | MEDIUM   | typecheck `_compatible` recursion              | Deeply-nested ref pairs (~500+) blow Python recursion stack — symmetric to pre-D7 hole|
| F6  | LOW      | typecheck `_resolve_size_expr` D3 text         | Negative branch says ">= 0", zero branch says "> 0" — inconsistent invariant phrasing |
| F7  | LOW      | bootstrap parser.hx D2 tag-12 sentinel         | Tag 12 overlaps the 0-11 prim-type-tag namespace; future prim addition silently breaks|
| F8  | LOW      | typecheck `_fmt` parametric TyStruct           | Mangled names (`Box__i32`) leak verbatim into user diagnostics instead of `Box<i32>`  |
| F9  | LOW      | typecheck `_ad_warn_mixed_inner` text          | Logic-domain warn text says "D-binop with mixed inner types ... [Logic-domain]"       |
| F10 | LOW      | typecheck `_check_cast_compat` trap 28803      | Trap unreachable from any source-level construction (parser caps refs at depth 1)     |

---

## Per-finding sections

### Finding F1: D2 parser tag-12 sentinel fires on ALL Call-RHS untyped lets, breaking i32-returning fn idiom

**File**: `helixc/bootstrap/parser.hx:2325-2334` (Call-RHS arm
registering inferred_ty_tag = 12); `helixc/bootstrap/parser.hx:1819-1820`
(capture-site `cap_ty_tag > 0` guard); `helixc/tests/test_codegen.py:3471-3477`
(regression test that codifies the broken behavior).
**Severity**: CRITICAL
**Category**: cycle-3-fix-introduced false-positive regression

**Description**:
Cycle-3 D2 (commit 3b321e6) extended cycle-2 B:C2 by adding a
`val_tag == 16` arm at the let-inference site that registers a
sentinel tag 12 for Call-RHS untyped lets. The capture-site guard
at parser.hx:1819-1820 is `if cap_ty_tag > 0`, which fires trap
76003 on tag 12 the same way it fires on tags 1-11 (proven-non-i32
prim types).

The cycle-3 commit message documents the intent:
> `let pi = get_pi(); let c = |x| x + pi; ...` → tag 12, traps (D2)

The bug: the parser has no access to function return-type
information. So Call-RHS lets are tagged 12 *regardless of the
called function's actual return type*. A call to an i32-returning
function like `let n = grid_n();` (where `fn grid_n() -> i32 { ... }`)
also gets tagged 12, and any subsequent capture of `n` in an i32
closure traps 76003 (SIGILL at runtime, exit code 132).

The user's question explicitly asked: "Does it correctly NOT fire
on `let x = i32_returning_fn();` when the fn return type is known
i32?" **Answer: No.** The fix is over-eager.

**Reproducer** (verified at b3504a2):
```helix
fn get_pi() -> i32 { 3 }
fn main() -> i32 { let pi = get_pi(); let c = |y| y + pi; c(0) }
```
Expected: returns 3 (no truncation — `pi` IS i32, capture is safe).
Actual: SIGILL (exit code 132), trap 76003 fires at closure build.

The cycle-3 commit added a regression test
(`test_codegen.py:3471-3477`) that codifies the over-trap:
```python
assert compile_and_exec(
    "fn get_pi() -> i32 { 3 } "
    "fn main() -> i32 { let pi = get_pi() ; let c = |y| y + pi ; c(0) }"
) == 132, "D2: call-RHS untyped capture now traps 76003 ..."
```
This calcifies the broken behavior into the test suite and gives the
audit a confidence trap: the test passes, so the fix is "correct" —
but the test is asserting a regression.

**Real-world exposure**: every Phase-0 `let x = foo();` followed by a
closure capture of `x` traps. This idiom is pervasive in
`helixc/examples/dashboard_agent.hx`, `agi_substrate_demo.hx`,
`dashboard_nn_agent.hx`, etc. The `let n = grid_n();` and `let goal
= goal_id();` patterns are everywhere.

There is also a separate test failure in
`test_bootstrap_kovc_full_pipeline_arithmetic` —
`compile_and_exec("__hash_i32(2)")` returns 132 (SIGILL) where the
test expects 187 (low byte of hash). This is plausibly the same D2
over-trap leaking through the bootstrap kovc.hx self-compilation
chain.

**Recommended fix**:
The parser cannot know return types. Two viable paths:

1. **Move the trap to typecheck**: The parser strips the tag-12
   sentinel; typecheck attaches a type-bit when emitting the
   closure-capture probe code, after typecheck has resolved the
   called fn's return type. This is the correct long-term shape.
2. **Require explicit annotation for Call-RHS**: revert the D2 fix
   entirely; the user must write `let pi: i32 = get_pi();` if the
   binding is captured. Loud failure (annotation absent + capture
   present) is better than the current silent over-trap.

The cycle-3 commit message says "the dominant idiom `let pi =
get_pi(); let c = |x| x + pi;` ... is still silent. The narrowest
variant of B:C2's gap is closed (literal-RHS), but the wider variant
remains open." — but closing the "wider variant" via a parser-side
sentinel without type info is fundamentally unsound. The cycle-3
fix should have been deferred to a typecheck-driven solution.

---

### Finding F2: `_compatible` TyArray arm uses raw `==` for size, breaking cascade-safe / generic-defer invariant introduced by D1

**File**: `helixc/frontend/typecheck.py:2181-2184` (`_compatible`
TyArray arm at b3504a2).
**Severity**: MEDIUM
**Category**: structural-equality invariant drift introduced by D1

**Description**:
Cycle-3 D1 added structural arms to `_compatible` for TyDiff /
TyLogic / TyTuple / TyArray / TyRef / TyPtr / TyFn so the new
non-TyPrim call-boundary check can recognize compatibility. All
arms except TyArray use `self._compatible(inner, inner)` for
sub-components, which inherits the top-of-function cascade-safe rule
(`isinstance(a, TyUnknown) or isinstance(b, TyUnknown): return True`).

The TyArray arm at b3504a2:
```python
if isinstance(a, TyArray) and isinstance(b, TyArray):
    # Sizes are TyPrim('size_N'); compare nominally.
    return (self._compatible(a.elem, b.elem)
            and a.size == b.size)
```

`a.size` is `Type` (per the schema at line 76-78), not necessarily
`TyPrim('size_N')`. `_resolve_size_expr` can return:
- `TyPrim('size_N')` for concrete IntLit sizes (after D3)
- `TySize(name)` for size-kind generic params
- `TyUnknown(hint=...)` for unresolvable Binary / Unary / Name
- `TyVar(name)` if the size param resolves to a type-kind var

Two false-positive paths:

1. **Generic-array-call-boundary**: `fn g[M](a:[i32;M]) { f(a) }` where
   `fn f[N](a:[i32;N])`. At the call to `f(a)`, `pty=TyArray(i32,
   TySize('N'))`, `aty=TyArray(i32, TySize('M'))`. D1's top-level
   TyVar/TySize filter only checks the OUTER type — TySize inside an
   array escapes. `_compatible` reaches the TyArray arm and compares
   `TySize('N') == TySize('M')` → False (different `name` on frozen
   dataclass). The D1 elif at line 736 then emits a false call-boundary
   mismatch diagnostic, when in fact mono will instantiate N=M.

2. **TyUnknown-size cascade**: `fn f(a:[i32;5])` called with arg of
   type `TyArray(i32, TyUnknown(...))`. Both sides are TyArray, so
   top-level cascade-safe at line 2158 doesn't fire. The arm compares
   `TyPrim('size_5') == TyUnknown(...)` → False. Cascade-safe rule
   ("TyUnknown anywhere skips the check") is violated for sub-component
   sizes — every other arm in `_compatible` propagates this rule via
   the recursive call.

**Reproducer** (probe-verified at b3504a2 by reading the file):
```python
a = TyArray(elem=TyPrim('i32'), size=TySize('N'))
b = TyArray(elem=TyPrim('i32'), size=TySize('M'))
tc._compatible(a, b)
# -> False  (should defer to mono via cascade-safe; never reached
#            pre-D1 because the call boundary didn't fall through
#            to `_compatible` for non-prim pairs)
```

**Recommended fix**:
Recurse on size to inherit cascade-safe + TySize-defer behavior:
```python
if isinstance(a, TyArray) and isinstance(b, TyArray):
    return (self._compatible(a.elem, b.elem)
            and self._compatible(a.size, b.size))
```
Then add an explicit TySize / TyVar arm to `_compatible` (or extend
the top cascade-safe rule) so `TySize('N') ~ TySize('M')` returns
True (defer to mono). `_compatible` already cascades TyUnknown at
the top. The change tightens TyArray's contract to match the rest
of the composite arms.

Symmetric concern: `_compatible` also has no TyTensor / TyTile arms.
Both fall through to `a == b` (raw nominal eq), which fails on TySize
shapes the same way TyArray's `a.size == b.size` does. The fix for
TyArray should be paired with structural TyTensor / TyTile arms (the
shape tuple recurses element-wise via `_compatible`, the
device/layout/memspace markers compare by raw eq).

---

### Finding F3: `Logic<T> + bareT` wrap-asymmetric still silent — D4 covered only Logic-Logic mixed-inner

**File**: `helixc/frontend/typecheck.py:1357-1374` (binop Logic
branch at b3504a2); `helixc/frontend/typecheck.py:128-162` (TyLogic
docstring asserting provenance tracking).
**Severity**: MEDIUM
**Category**: incomplete cycle-3 D4 coverage; parallel to the
cycle-2 B:C6 → cycle-3 D4 escalation pattern

**Description**:
Cycle-3 D4 added the gate:
```python
elif (l_is_logic or r_is_logic) and inner_mismatch:
    ...
    self._ad_warn_mixed_inner(..., extra=" [Logic-domain]" + extra)
```

For `Logic<f64> + f64` (one side Logic-wrapped, other bare): the
`_unwrap` helper at line 1280-1285 strips both TyDiff and TyLogic
wrappers symmetrically, so `l_inner = f64` and `r_inner = f64`.
`inner_mismatch` evaluates to `(f64 != f64) = False`. The gate
doesn't fire. The else branch at line 1375 picks `l_inner`. The
result type is rebuilt as `Logic<f64>` via line 1381 (`if l_is_logic
or r_is_logic: wrapped = TyLogic(...)`). The bare-f64 operand's
"no provenance" status is absorbed into Logic<f64>'s
provenance-tracked result with zero diagnostic.

This is the exact shape of the cycle-2 D-D vs D-bare asymmetric case
(B:C6) for TyDiff — cycle-2 closed it by changing the gate from
`l_is_diff AND r_is_diff` to `l_is_diff OR r_is_diff` AND adding the
asymmetric-extra suffix. The cycle-3 D4 fix added the Logic arm with
the OR gate at the wrap level but kept `inner_mismatch` as the gate
predicate. For Logic-wrap-asymmetric, both inners are identical (the
bare operand IS the Logic's inner), so `inner_mismatch` is always
False and the warning never fires.

The TyLogic docstring (typecheck.py:128-162) explicitly says Logic
carries provenance:
> Provenance lattice tracking which input atoms contributed to each
> derived value
> Trap 24100 emitted if a non-Logic value is passed where a
> Logic-typed parameter is required, or vice versa, in a
> provenance-sensitive context.

Silently absorbing a bare-T operand into a Logic<T> result defeats
the contract: the bare operand contributes no provenance domain, yet
the result claims Logic-provenance heritage.

**Reproducer** (probe at b3504a2):
Construct `Logic<f64>` and `f64` operands of a binop; observe that
`_DIFF_WARNINGS` stays empty after the binop call.

**Recommended fix**:
Extend the gate symmetric to cycle-2 B:C6:
```python
elif (l_is_logic or r_is_logic) and (
        inner_mismatch
        or (l_is_logic != r_is_logic)   # wrap-asymmetric
):
    ...
```
Or split into two arms — one for mixed inner, one for
wrap-asymmetric — with distinct trap text. Either matches B:C6's
close.

---

### Finding F4: ShapeFoldError only caught by struct_mono; fn-mono path surfaces it as "compiler bug"

**File**: `helixc/frontend/monomorphize.py:63-72` (ShapeFoldError);
`helixc/frontend/monomorphize.py:677-679` (entry `monomorphize`);
`helixc/backend/x86_64.py` (fn-mono call site, no catch);
`helixc/check.py:272-283` (top-level `except Exception` from C3-3).
**Severity**: MEDIUM
**Category**: exception contract asymmetry / diagnostic-quality
regression on a legitimate user error

**Description**:
Cycle-3 C3-6 introduced `ShapeFoldError(ValueError)` raised from
`_fold_intlit_arith` on `/0` and `%0` (trap 28801). `struct_mono`
catches it at struct_mono.py:448 and surfaces as a diagnostic via
`diags.append(str(e))`. `monomorphize_structs` returns `(prog, diags)`.

But `_fold_intlit_arith` is reachable from any `_subst_shape_expr`
invocation, which is called from `substitute_ty`, which is called
from fn-mono's `_instantiate`. `Monomorphizer.run()` (lines 403-429)
does not catch ShapeFoldError. The top-level `monomorphize(prog)`
does not catch. The x86_64 driver invokes `monomorphize` with no
wrapper.

A generic fn with `[T; N/0]` (after substitution) raises
ShapeFoldError from fn-mono, which propagates through `monomorphize`
into the cycle-3 C3-3 top-level `except Exception` in check.py,
which prints:
```
helixc: internal error: ShapeFoldError: <message>
helixc: this is a compiler bug — please file an issue.
```

This is a **false bug report**: the user wrote source triggering a
legitimate trap 28801 (shape fold by zero), and the compiler tells
them to file an issue instead of pointing at the array dim.

**Reproducer**:
```helix
fn f[N](a: [i32; N / (N - N)]) -> i32 { 0 }
fn main() -> i32 { f::<5>([0; 1]) }
```

At fn-mono time, `N / (N - N)` substitutes to `5 / 0`,
`_fold_intlit_arith` raises ShapeFoldError. fn-mono doesn't catch.
Top-level C3-3 except catches and prints "internal error".

**Recommended fix**:
Wrap `_instantiate` in `Monomorphizer.run()` (or at the
`_rewrite_calls_in_expr` call site) with try/except for
ShapeFoldError, accumulate diagnostics on the Monomorphizer
instance, and surface via return-value channel parallel to
`monomorphize_structs`'s `(prog, diags)` tuple. Then the top-level
driver consumes diags and prints trap 28801 with file/line/col
context.

---

### Finding F5: `_compatible` recursive on inner types — no depth guard; deeply-nested ref pairs blow Python recursion stack

**File**: `helixc/frontend/typecheck.py:2157-2206` (`_compatible`
recursion on TyRef.inner, TyPtr.inner, TyDiff.inner, TyLogic.inner,
TyMemTier.inner, TyQuote.inner, TyArray.elem, TyTuple.elems,
TyFn.params/ret).
**Severity**: MEDIUM
**Category**: cross-cutting consistency gap — D7 closed the symmetric
hole in `_check_cast_compat` but `_compatible` retained it

**Description**:
Cycle-3 D7 (in commit 74b72ec) closed the RecursionError
vulnerability in `_check_cast_compat` by converting the recursive
ref-pair walker to iterative + adding a depth-8 guard that emits
trap 28803. The motivation was: even though Phase-0 source syntax
caps refs at depth 1 (parser refuses `&&T`), programmatic
construction or future autogenerated AST input could blow the
Python recursion stack.

`_compatible` has the same shape: it recurses unboundedly through
TyRef.inner, TyPtr.inner, and the composite arms. Probe verified
at b3504a2:

```python
src = TyPrim('i32'); tgt = TyPrim('i32')
for _ in range(2000):
    src = TyRef(inner=src, is_mut=False)
    tgt = TyRef(inner=tgt, is_mut=False)
tc._compatible(src, tgt)
# -> RecursionError: maximum recursion depth exceeded
```

The D7 fix established the policy: even if practically unreachable
from source, recursion-bounded type-checker primitives are a real
defense-in-depth contract. `_compatible` lacks this guard. The
cycle-3 fix sweep applied the guard to one of two parallel
primitives — the other now stands out as the asymmetric outlier.

**Reachability from user code**: Low. Phase-0 parser caps explicit
nested refs at 1 layer (lexer treats `&&` as the LAND token).
Programmatic construction via `substitute_ty` of nested generic
types could theoretically reach depths >100, but not >250 in any
realistic Phase-0 source. Same low-reachability classification as
D7 itself.

**Recommended fix**:
Add a depth-bounded guard parallel to D7's. Either:
1. Convert `_compatible`'s ref/ptr arms to iterative ref-peeling
   (lockstep) before recursing into other arms.
2. Add a `_depth` parameter that bounds at 8, raising
   `TypeError_` with trap 28803 (same trap reuse) before
   approaching Python's recursion limit.

Option 2 is the simpler patch since `_compatible` has many composite
arms; iterative peel would only address TyRef/TyPtr. Bumping the
trap to a separate id (28804) would also be reasonable since the
two primitives have different call-site contracts.

---

### Finding F6: D3 diagnostic text drifts — negative branch says ">= 0", zero branch says "> 0"

**File**: `helixc/frontend/typecheck.py:579-608` (`_resolve_size_expr`
D3 arms at b3504a2).
**Severity**: LOW
**Category**: diagnostic-quality / invariant-expression clarity

**Description**:
The D3 fix added two near-identical IntLit-validation arms (lines
579-591 for IntLit, lines 593-608 for Unary(-, IntLit)). Each emits
two different diagnostics depending on whether the value is negative
or zero:

- `< 0` → `"array size must be >= 0, got -6 (trap 28802)"`
- `== 0` → `"array size must be > 0 in Phase-0, got 0 (trap 28802)"`

These contradict each other: the negative branch claims the rule is
`>= 0` (which would allow 0) and the zero branch claims `> 0` (which
forbids 0). The real rule (per the docstring at line 567-575) is
`> 0`. A user reading the negative diagnostic could reasonably try
`[T; 0]` thinking 0 is allowed, then hit the zero diagnostic with
the contradictory rule.

**Recommended fix**:
Unify the diagnostic text to a single rule. One concise message
covers both:
```python
if expr.value <= 0:
    self.errors.append(TypeError_(
        f"array size must be > 0 in Phase-0, got {expr.value} "
        f"(trap 28802)",
        expr.span,
    ))
```
Same change for the Unary(-, IntLit) arm. Collapses two arms to
one each, removes the negative/zero phrasing drift.

---

### Finding F7: Parser tag-12 sentinel overlaps the 0-11 prim-type-tag namespace

**File**: `helixc/bootstrap/parser.hx:2325-2333` (D2 tag-12
sentinel); `helixc/bootstrap/parser.hx:2281-2292` (existing type
tags 0-11).
**Severity**: LOW (separate from F1's functional regression)
**Category**: namespace / encoding overlap — design smell

**Description**:
Cycle-3 D2 added a sentinel value (tag 12) at the let-inference site
to mean "untracked Call-RHS, assume non-i32 for closure-capture trap
purposes". The existing type tags 0-11 are reserved for primitive
types:

```
0 = i32 (trivially safe)
1 = f32          7 = u8
2 = f64          8 = u16
3 = i64          9 = u64
4 = bf16        10 = i8
6 = u32         11 = i16
```

Tags 5 and 12+ are unassigned in source documentation. The D2 fix
picked 12 as the "untracked-call" sentinel. The capture-site guard
at parser.hx:1819-1820 is `if cap_ty_tag > 0`, currently treating
12 the same as 1-11 (non-i32 → trap 76003).

The encoding mixes two concepts:
1. **Type-tag** (concrete prim type known at let-inference time)
2. **Provenance-tag** (concrete prim type NOT known, but assumed
   non-i32 for trap purposes)

A future primitive type at tag 12 (or a new quantized type) would
silently collide. Worse, the comment block at line 2281-2292
enumerates 0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11 — leaving 5 and 12
conspicuously absent. A reader who sees tag 5 vs tag 12 wouldn't
know the conceptual difference.

This is a small design smell; the sentinel space and type-tag space
share an encoding without a clear separator.

**Recommended fix**:
Move sentinels to a clearly-out-of-band range, e.g., 100+:
```
0-11:  concrete prim types
100:   untracked-call sentinel
101:   untracked-other sentinel (future)
```
Plus document the boundary at the file header. The capture-site
guard stays `> 0`. Encoding is non-overlapping and future-proof.

Alternative: split into two parallel tables. The stride-3
`cl_capture_tab` deferred from cycle 1 is the long-term answer.

Note: this finding is independent of F1. F1 is the functional
regression (over-eager trap on all Call-RHS lets, regardless of
return type); F7 is the encoding-namespace overlap that would bite
a future prim-type addition even if F1 were fixed.

---

### Finding F8: `_fmt` parametric TyStruct leaks mangled name (`Box__i32`) verbatim

**File**: `helixc/frontend/typecheck.py:2212` (D8 TyStruct arm).
**Severity**: LOW
**Category**: diagnostic UX — partial-fix-of-prior-fix

**Description**:
Cycle-3 D8 added a `_fmt` arm for TyStruct:
```python
if isinstance(t, TyStruct): return t.name
```

This closes the user-visible `TyStruct(name='Foo')` leak for
non-parametric structs. But parametric struct types are stored under
their mangled name (`mangle_struct(base, args)` at line 557 returns
e.g. `Box__i32`), so `_fmt(TyStruct(name='Box__i32'))` returns
`Box__i32` verbatim. Users writing `Box<i32>` in source see
`Box__i32` in diagnostics — a subtler papercut than `TyStruct(...)`
but still confusing.

**Reproducer** (probe-verified at b3504a2):
```python
tc._fmt(TyStruct(name='Box__i32'))   # -> 'Box__i32'
tc._fmt(TyStruct(name='Pt__i32__f64'))   # -> 'Pt__i32__f64'
```

**Recommended fix**:
Demangle in `_fmt`: split on `__`, treat first segment as base, rest
as type args. Render as `base<arg1, arg2>`. The mangling shape is
declared at `struct_mono.mangle_struct`; the inverse is a small
helper. Place it adjacent to `_fmt`.

```python
if isinstance(t, TyStruct):
    parts = t.name.split('__')
    if len(parts) >= 2:
        return f"{parts[0]}<{', '.join(parts[1:])}>"
    return t.name
```

Won't handle nested-generic mangled names (`Box__Box__i32` →
`Box<Box<i32>>` not `Box<Box, i32>`) — but this is a known
limitation of the current mangling scheme.

---

### Finding F9: Logic-domain AD002 text says "D-binop" — the prefix bleeds through `_ad_warn_mixed_inner`

**File**: `helixc/frontend/typecheck.py:2135-2139`
(`_ad_warn_mixed_inner`).
**Severity**: LOW
**Category**: diagnostic UX — text drift from D4 fix

**Description**:
Cycle-3 D4 added a Logic-Logic-mixed-inner warn that funnels through
the existing `_ad_warn_mixed_inner` helper with a `[Logic-domain]`
suffix tag. The helper text is hardcoded:
```python
_ad._DIFF_WARNINGS.append(
    f"{span.line}:{span.col}: AD: D-binop with mixed inner "
    f"types {self._fmt(l)} vs {self._fmt(r)} — widened to "
    f"{self._fmt(chosen)} (trap 24200/AD002)" + extra
)
```

For `Logic<f64> + Logic<i32>`, the resulting message reads:
```
AD: D-binop with mixed inner types f64 vs i32 — widened to f64
(trap 24200/AD002) [Logic-domain]
```

The "D-binop" prefix is misleading — this is a Logic-binop. The
`[Logic-domain]` suffix qualifier helps but the user has to
back-track from "D-binop" → "[Logic-domain]" to figure out which
operation actually triggered. The user's question explicitly asked:
"Does the diagnostic correctly identify which Logic is wrapping
what?" The answer is partial — the suffix qualifies but the prefix
still says "D-binop".

**Recommended fix**:
Pass an explicit kind to `_ad_warn_mixed_inner` or parameterize the
prefix:
```python
def _ad_warn_mixed_inner(self, span, l, r, chosen,
                         kind: str = "D", extra: str = "") -> None:
    _ad._DIFF_WARNINGS.append(
        f"{span.line}:{span.col}: AD: {kind}-binop with mixed inner "
        f"types ..."
    )
```
Then the D-domain call passes `kind="D"` and the Logic-domain call
passes `kind="Logic"`. The suffix tag becomes redundant and can be
dropped.

---

### Finding F10: Trap 28803 (D7 ref-cast depth limit) is unreachable from any source-level construction

**File**: `helixc/frontend/typecheck.py:2078-2102` (depth-8 guard
trap 28803).
**Severity**: LOW
**Category**: trap reservation — defensive but unreachable

**Description**:
Cycle-3 D7 added an iterative ref-peel with a depth-8 guard that
emits trap 28803 if the cast matrix would recurse beyond 8 nested
TyRef wrappers on either side. The motivation was defense in depth
against RecursionError from malicious / autogenerated source.

Probe at b3504a2: Phase-0 lexer treats `&&` as the LAND token (the
parser does NOT support nested refs in source). A user cannot write
`&&i32 as &&i64` — the parser refuses at depth 2 already. And
`substitute_ty` produces only as many refs as the user wrote (the
substitution doesn't construct new TyRef wrappers around the
substituted type). So trap 28803 fires only on programmatically-
constructed input that doesn't pass through the parser at all.

This is not strictly a bug — the trap is a defense-in-depth
backstop. But the trap-id reservation in `docs/lang/trap-ids.md`
treats it as a regular user-facing diagnostic, alongside 28801 and
28802 (both genuinely reachable from user code). Reserving trap IDs
for unreachable paths is a minor invariant-expression issue: the
trap-id table is now non-uniform in reachability.

The user's question asked: "Does the limit trap 28803 fire on
legitimate-but-deep refs?" Answer: only on programmatic input;
syntactically unreachable. The defense-in-depth value is preserved
but the documentation should note "internal: triggered only by
programmatic AST construction" or similar.

**Recommended fix**:
Document the reachability boundary in `docs/lang/trap-ids.md` for
row 28803. Optionally: collapse 28801/28802/28803 into a single trap
ID family (28800 + sub-id) per the AST_tag * 1000 + sub_id
convention. Cycle-3's a878709 commit reserved the three IDs
separately; consolidation is a minor follow-on.

---

## Cycle 3 fix re-verification

| Fix    | Status     | Notes                                                                                                          |
|--------|------------|----------------------------------------------------------------------------------------------------------------|
| C3-1   | OK         | grad_pass `_rewrite_in_expr` recurses through chained else-if. Mirror of `_resolve_in_expr`. No regressions.   |
| C3-2   | OK         | `_WIDEN_NAME_ALIASES` collapses isize↔i64, usize↔u64. `_widen_canon_name` correctly drops the same-rank tie. No other alias pairs needed (`int`/`uint`/`c_int` not in Phase-0 vocabulary). |
| C3-3   | OK*        | `_main_inner` wrapped in try/except/finally. **However**, bare `except Exception` swallows ShapeFoldError from fn-mono path — see F4. |
| C3-4   | OK         | `monomorphize_structs` idempotent via existing-mangled-name set. Idempotency verified. ShapeFoldError caught with explicit `except ShapeFoldError as e:` before generic ValueError so trap-id stays attached. |
| C3-5   | OK         | `_inline_lets` recurses through Cast/Call/Field/Index/Match/etc. catch-all `_ad_warn`. (Out of scope for type-design audit — covered by audit A.) |
| C3-6   | OK         | `ShapeFoldError(ValueError)` raised on `/0` and `%0` in `_fold_intlit_arith`. Trap 28801 reserved. struct_mono catches and surfaces. (Catch-contract gap is F4.) |
| D1     | OK*        | `_check_call_basic` falls through to `_compatible` for non-prim pairs. `_compatible` has structural arms for TyDiff/TyLogic/TyTuple/TyArray/TyRef/TyPtr/TyFn. Frozen dataclasses prevent cyclic refs (terminates). **However**, TyArray arm uses raw `==` for size (F2); no TyTensor/TyTile arms (related but covered by audit A's C4-4); recursion unbounded (F5). |
| D2     | BROKEN     | parser.hx tag-12 sentinel **over-fires on all Call-RHS lets**, including i32-returning fn results — see F1 (CRITICAL). Tag-encoding namespace overlap is F7 (LOW). |
| D3     | OK*        | `_resolve_size_expr` emits trap 28802 on IntLit `< 0`, `== 0`, plus source-level `Unary(-, IntLit)`. All array-size positions caught (`[T; N]`, `[T; N-2]` after fold). **However**, diagnostic text drifts between branches — F6. |
| D4     | OK*        | Logic-Logic mixed-inner emits AD warn with `[Logic-domain]` tag. Span shared with D-domain variant (both use `expr.span` at lines 1354/1372). **However**, Logic-wrap-asymmetric still silent — F3; warn-text prefix still says "D-binop" — F9. |
| D5     | OK         | `_fold_intlit_unary` folds `Unary(-, IntLit(N))` → `IntLit(-N)` and `Unary(+, IntLit(N))` → `IntLit(N)`. Cascaded negation `--N` correctly folds to `N` (probe-verified). |
| D6     | OK         | `_ty_key` raises `TypeError` on non-AST.TyNode inputs. The TypeError surfaces with a clear `type-name + repr` message; useful diagnostic. |
| D7     | OK*        | `_check_cast_compat` iteratively peels matching ref-pairs; depth-8 guard emits trap 28803. No RecursionError on ref-cast path. **However**, trap unreachable from source-level construction — F10 (LOW); symmetric `_compatible` recursion vulnerability not closed — F5. |
| D8     | OK*        | `_fmt(TyStruct)` returns `t.name`. **However**, parametric struct mangled names (`Box__i32`) leak verbatim — F8. |
| D9     | OK         | `_walk_subst_expr` Call arm substitutes `callee.generics`. Nested-turbofish (`id::<U>(other::<T>(x))`) recurses correctly through the args list, so nested generic substitution works. Note: audit A may have an end-to-end mono finding (C4-4 in their doc) about iteration order, which is out of this audit's scope. |

Six cycle-3 fixes have residual gaps (`*` or **BROKEN**):
- **D2** is BROKEN (F1 — CRITICAL functional regression on legitimate code).
- **C3-3** swallows ShapeFoldError (F4).
- **D1** TyArray size raw eq (F2), `_compatible` unbounded recursion (F5).
- **D3** diagnostic text drift (F6).
- **D4** Logic-wrap-asymmetric (F3), text prefix bleed (F9).
- **D7** symmetric `_compatible` gap (F5), unreachable trap (F10).
- **D8** parametric mangled-name leak (F8).

Eight other cycle-3 fixes hold up under re-audit with no caveats.

---

## Cycle 4 focus question answers

**1. Does the new `_compatible` structural recursion terminate on all
legitimate inputs?**

**Almost**. Frozen dataclasses prevent cyclic type construction at
runtime, so any finite-depth Type pair terminates. But there is no
depth guard. Probe: a 1000-level nested TyRef pair raises
`RecursionError`. Practically unreachable from Phase-0 source (parser
caps at 1 ref layer), but the symmetric vulnerability of D7's
`_check_cast_compat` (also fixed in cycle 3) is left open here.
**Finding F5**.

**2. Are trap 28801/28802/28803 wirings actually unreachable from
legitimate user code?**

- **28801** (shape-fold div/mod by zero): reachable from legitimate
  generic code with `[T; N/0]` after substitution.
- **28802** (array size <= 0): reachable from user-source `[T; 0]`
  or `[T; -5]`.
- **28803** (cast-matrix depth > 8): unreachable from any source-
  level construction (parser caps refs at depth 1, the trap only
  fires on programmatic AST input). **Finding F10**.

**3. Does the `Logic-domain` AD002 variant share the right span with
the `D-domain` variant?**

**Yes** — both call `self._ad_warn_mixed_inner` with `expr.span`
(typecheck.py:1354 for D-domain, 1372 for Logic-domain), same
expression-level span. The text prefix is the issue (F9), not the
span. The user's question is answered cleanly on span — but the
diagnostic text says "D-binop" for both, requiring the user to read
the `[Logic-domain]` suffix to disambiguate.

**4. Does the new call-RHS closure-capture inference (D2) correctly
defer-vs-trap for forward-declared fns?**

**No**. The parser tag-12 sentinel fires on ALL Call-RHS untyped
lets, regardless of the called function's return type. Even
`let n = grid_n();` where `fn grid_n() -> i32` traps 76003 on
subsequent closure capture. The user's question asks specifically
about i32-returning fns — the answer is "the inference is
unconditional, not type-aware". **Finding F1 (CRITICAL)**.

---

## Cycle 3 fix re-verification — user question-by-question

**D1**: `_compatible` for nested generic combinations works through
the structural recursion (TyDiff(TyLogic(TyTuple(...))) etc.).
**Cyclic refs**: impossible at runtime (frozen dataclass). Termination
on legitimate input: yes (finite depth). **But** RecursionError on
~250+ depth (F5).

**D2**: Does it correctly NOT fire on `let x = i32_returning_fn();`
when the fn return type is known i32? **No** — fires unconditionally
(F1).

**D3**: All size positions (`[T; N]`, `[T; N-2]`, etc.) caught after
fold? Yes for IntLit-after-fold paths. The Binary fold from D5 also
feeds into `_resolve_size_expr` which catches `< 0` and `== 0`.

**D4**: Does the diagnostic correctly identify which Logic is
wrapping what? Span sharing OK (yes). Text disambiguation: the
`[Logic-domain]` suffix qualifies but the `D-binop` prefix bleeds
through (F9).

**D5**: Cascaded negations (`--N` → `N`)? Yes — probe-verified that
`Unary(-, Unary(-, Name('N')))` with N=5 folds to `IntLit(5)`.

**D6**: TypeError surface useful? Yes — `_ty_key` raises with
`f"_ty_key: expected A.TyNode, got {type(t).__name__}: {t!r}"` which
gives the user the type name and repr.

**D7**: Trap 28803 fires on legitimate-but-deep refs? Practically
unreachable (F10) — but defense-in-depth value preserved.

**D8**: Parametric struct mangled names? **No** — F8 (mangled name
leaks verbatim).

**D9**: Nested-turbofish substitutes correctly? Yes — `_walk_subst_expr`
recurses through Call.args, so `id::<U>(other::<T>(x))` re-substitutes
both generics lists at mono time.

**C3-2**: Other alias pairs needed? **No** — `int`/`uint`/`c_int`/
`intptr_t` are not in Phase-0 vocabulary. `isize`/`usize` are the
only platform-width aliases that need canonicalization.

---

## Cycle 4 status

**Strict criterion (per user directive 2026-05-10): cycle clean iff
zero findings of ANY severity.**

This cycle finds **10 new findings (1 CRITICAL, 0 HIGH, 4 MEDIUM, 5
LOW)**. By the strict criterion, **cycle 4 does NOT count clean**.

The CRITICAL is the standout: cycle-3's D2 closure-trap fix is
fundamentally unsound at the parser level — it cannot distinguish
i32-returning Call RHS from non-i32-returning Call RHS, because the
parser has no type information. The fix should have been deferred to
a typecheck-driven path. The fact that cycle-3 added a regression
test (`test_codegen.py:3471-3477`) that asserts the false-positive
behavior makes this insidious: the test passes, validating a
regression that breaks the most common Phase-0 idiom.

The four MEDIUMs (F2, F3, F4, F5) are the next-layer-down
asymmetries cycle-3's structural widening (D1) opened plus a
symmetric cross-cutting gap (F5) that mirrors the D7 hole D7 itself
closed. Same pattern as cycle-2 → cycle-3 (partial-fix-of-prior-fix
exposing next layer), repeating one cycle later.

Recommended fix sequence for cycle 5:

1. **F1 first** (CRITICAL — revert D2's parser-side tag-12 sentinel;
   move the closure-capture type-bit emission to typecheck where
   return-type info is available). Remove the regression test that
   codifies the broken behavior. Pair with F7 (sentinel namespace
   redesign) since both touch parser.hx.
2. **F2 + F5 bundle** (TyArray size symmetry + `_compatible` depth
   guard — both touch `_compatible`; the guard can reuse trap 28803
   or get a new id 28804).
3. **F3** standalone (Logic-wrap-asymmetric — small extension to
   the D4 gate parallel to B:C6).
4. **F4** standalone (ShapeFoldError catch contract — wrap fn-mono).
5. **F6 / F8 / F9 / F10** as a low-severity cleanup commit. F6 is a
   one-line message merge; F8 is a small demangle helper; F9 is a
   kind-parameter pass-through; F10 is a docs update.

Once F1-F5 land + verify, cycle 5 can re-audit. Per the strict
criterion, the remaining cycles needed to declare type-design
soundness clean is at least one full clean sweep after addressing
all 10 findings — pending whether new gaps surface from the fixes
themselves (the cycle 1 → 2 → 3 → 4 pattern of "fix exposes the next
layer" continuing).

The high cadence of partial-fix-of-prior-fix findings across four
cycles suggests the underlying type-design contracts (structural
equality, exception catch-scope, parser-vs-typecheck information
boundaries) need a comprehensive review rather than incremental
patches. A dedicated invariant-statement pass — writing down each
contract as an English sentence and probing each fix's
diff against the full invariant set — would likely surface the
recurring asymmetries in one pass instead of one-per-cycle.
