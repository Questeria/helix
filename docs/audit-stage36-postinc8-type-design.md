# Stage 36 Post-Increment-8 Audit — Type-Design Lane

**Date**: 2026-05-16
**Auditor**: pr-review-toolkit:type-design-analyzer
**HEAD audited**: `a451591` (Stage 36 Increment 8)
**Baseline**: `b8cafe7` (Stage 35 closure)
**Status**: **NOT CLEAN — 2 HIGH + 4 MEDIUM + 2 LOW**

## Findings

### A1 HIGH (conf 95) — `Logic<T>` is type-erased: `TyLogic` ignores its inner type, defeating `Logic<f32>` vs `Logic<i32>` distinction

**Files**:
- `helixc/frontend/typecheck.py:2880-2922` (fuzzy ops)
- `helixc/frontend/typecheck.py:2761-2812` (boolean ops)

The fuzzy ops (`fuzzy_and/or/not/xor/implies`) all type-check with
`isinstance(t, TyLogic)` but **never inspect `t.inner`**. So
`fuzzy_and(prove(1, 0), prove(2, 0))` (Logic<i32>!) passes the
typechecker, then lowers to `MUL` with `result_ty=TIRScalar("f32")`
(lower_ast.py:2016) — silent type punning.

Symmetrically, `and_logic` / `or_logic` / `not_logic` claim
"Logic<i32>" in their errors but accept any `Logic<T>`, then
unconditionally lower to BIT_AND/OR (lower_ast.py:1834) which is
wrong for `Logic<f32>` operands.

The "wrapper carries T semantics" invariant is **not enforced**.

**Fix family**: inspect `t.inner` against `TyPrim("i32")` (logic ops)
or `TyPrim("f32")` (fuzzy ops) and emit the trap-24100 error
otherwise. *Architectural*: changes which programs typecheck — needs
user approval.

### A2 HIGH (conf 88) — `register_derivation` two-arena-push pair is not atomic; any intervening ARENA_PUSH corrupts the side-table

**Files**:
- `helixc/ir/lower_ast.py:1957-1973`
- `helixc/backend/x86_64.py:2650-2683`

The handle invariant is "left at index N, right at index N+1". But
the arena is a single global cursor shared with `MatchDispatch`,
struct lowering (`lower_ast.py:1293, 2234, 2341`), and any other
ARENA_PUSH consumer. There is no critical section.

If an optimization pass reorders, or any other emitter pushes
between the two pushes (e.g. a future struct lowering of an inlined
arg), `parent_right_at(h)` returns garbage. The two ARENA_PUSH ops
are also independent in TIR (no data dependency between them), so
DCE/CSE/scheduler reordering is legal even today.

**Fix family**: introduce a fused `ARENA_PUSH_PAIR` opcode (atomic
at IR level), or write `[tag, left, right]` triples and return the
triple index. *Architectural*: new IR opcode — needs user approval.

### B1 MEDIUM (conf 80) — `prove(value: T, src: i32)` flattens `Logic<Logic<T>>` silently

**File**: `helixc/frontend/typecheck.py:2715-2719`

`if isinstance(inner, TyLogic): return inner` — calling
`prove(some_logic_value, src)` returns the input unchanged and
**discards `src`**. This violates the documented invariant that
`prove` attaches provenance. A programmer who wraps twice
(legitimate Phase-1 expectation: re-prove with new evidence)
silently loses the new tag.

**Fix family**: either reject (typecheck error) or wrap-and-keep
both tags.

### B2 MEDIUM (conf 78) — `derive(a, b)` evaluates `b` only for side effects but discards its value AND its provenance entirely

**File**: `helixc/ir/lower_ast.py:1846-1853`

The lowering is `_lower(b); return _lower(a)`. Comment says "Phase-0
single-tag provenance". But the typecheck contract gives it
2-parent semantics in its name. With the arena side-table from Inc 5
now available, `derive` should call `register_derivation` instead of
dropping `b`. Currently `derive(p, q)` and `p` are observationally
indistinguishable, making the combinator dead weight.

**Fix family**: wire `derive` through `register_derivation` so the
two-parent semantics implied by the name are real.

### B3 MEDIUM (conf 65) — Forward-mode AD chain rule for `prove` does not guard against differentiable source-tag arg

**File**: `helixc/frontend/autodiff.py:1112-1114`

`return _diff(call.args[0], var)` — correct for the value arg, but
if a user wrote `prove(x, x)` (legal i32 into i32 source), the
second arg's derivative is silently zero (correct mathematically
since src is non-diff, but no diagnostic). Reverse-mode at
`autodiff_reverse.py:563` propagates only to `node.args[0]` with no
check that `args[1]` isn't itself a differentiable subexpression.

**Fix family**: add a typecheck rule that source-tag arg must be a
literal or non-differentiable.

### B4 MEDIUM (conf 82) — `to_logic_bool` accepts any int scalar but lowers identity → `to_logic_bool(some_i64)` produces `Logic<i32>` that is actually i64 at IR

**Files**: `helixc/frontend/typecheck.py:2841-2849`,
`helixc/ir/lower_ast.py:1937-1941`

`self._is_int_scalar(arg_tys[0])` accepts i32/i64/u32/u64 (per
Stage 35 convention), but the result is unconditionally
`TyLogic(inner=TyPrim("i32"))`. Lowering returns the bare value with
no widening/narrowing. Subsequent BIT_AND on this against another
`Logic<i32>` performs 32-bit ops on i64 data → upper bits silently
dropped.

**Fix family**: reject non-i32 at typecheck, or insert an explicit
narrow + i32-saturation (Stage 35 convention).

### C1 LOW (conf 60) — `unwrap_logic` returns `arg_tys[0]` on type error

**File**: `helixc/frontend/typecheck.py:2728-2735`

Error recovery returns the input type. Most other builtins in this
file return a sentinel. Inconsistent — cascades misleading downstream
errors.

### C2 LOW (conf 55) — `derive`/`and_logic`/`or_logic` recovery returns `arg_tys[0]` even when it was non-Logic

**Files**: `helixc/frontend/typecheck.py:2746-2752`, `:2769`, `:2783`

After emitting trap-24100, fallback
`if isinstance(arg_tys[0], TyLogic): return arg_tys[0]` else
`return TyLogic(inner=TyPrim("i32"))` — for `derive` the fallback
returns whatever the first arg's type was, which could mask the
inner-type mismatch in chained calls.

## Verified clean

- All five fuzzy chain rules (forward + reverse) match the supplied
  formulas exactly. Symmetric between forward and reverse mode; all
  five ops are registered in both. The `attach/detach/prove/
  unwrap_logic` identity passthroughs are also symmetric.
- `grad_rev(loss, k)` index-out-of-range handled correctly at
  `grad_pass.py:542` and `:686`.
- `unwrap_logic` on a non-Logic value caught at typecheck (the
  wrong-type fallback is the C1 sub-issue).
