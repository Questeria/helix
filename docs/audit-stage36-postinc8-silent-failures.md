# Stage 36 Post-Increment-8 Audit — Silent-Failure Lane

**Date**: 2026-05-16
**Auditor**: pr-review-toolkit:silent-failure-hunter
**HEAD audited**: `a451591` (Stage 36 Increment 8)
**Baseline**: `b8cafe7` (Stage 35 closure)
**Scope**: `git diff b8cafe7..HEAD -- helixc/{frontend,ir,stdlib,examples}`
**Status**: **NOT CLEAN — 3 HIGH + 2 MEDIUM + 2 LOW**

## Findings

### A1 HIGH (conf 95) — `parent_left_at` / `parent_right_at` have NO bounds check

**File**: `helixc/ir/lower_ast.py:1974-1997`

`parent_left_at(idx)` lowers directly to `ARENA_GET(idx)` and
`parent_right_at(idx)` to `ARENA_GET(idx+1)` with **zero
`__arena_len()` comparison**. A user passing an arbitrary i32
(negative, zero before any `register_derivation`, or > arena_len)
silently returns whatever bit pattern the arena/heap holds at that
offset. The handle returned by `register_derivation` (line 1968) is
opaque — there is no way for the caller to validate it.

This is the **exact forged-handle pattern** swept clean in
restart 45-47 for the AGI typed handles (wm/ep/bfs/visited/pq/hashmap).

**Fix family**: bounds-check before `ARENA_GET` — emit `CMP_LT idx,
__arena_len()` and trap (e.g., trap 36500) on out-of-range, mirroring
the restart-45/47 forge-guard pattern.

### A2 HIGH (conf 85) — `register_derivation` handle has no missing-vs-zero discriminator

**File**: `helixc/ir/lower_ast.py:1949-1972`

If a user stores derivation handles in a side array and reads back
`0` for an unwritten slot, `parent_left_at(0)` returns whatever was
written at arena index 0 (perfectly valid if anything was pushed
there — e.g., by another arena user) and `parent_right_at(0)` returns
arena[1]. There is no sentinel/tombstone separating "no parent
recorded" from "parent recorded as source-id 0".

**Fix family**: arena-sentinel — either reserve index 0 as "null
derivation" and start `register_derivation` at index ≥ 1, OR return a
tagged handle (e.g., `idx | 0x8000_0000`) so `parent_*_at` can
fail-closed on the un-tagged sentinel.

### A3 HIGH (conf 90) — Fuzzy ops produce nonsense gradients on out-of-[0,1] inputs (no fail-closed)

**Files**:
- `helixc/frontend/typecheck.py:2876-2925` (no range check at type level)
- `helixc/frontend/autodiff.py:1046-1115` (chain rules apply blindly)
- `helixc/frontend/autodiff_reverse.py:556-655`
- `helixc/ir/lower_ast.py:2010-2080`

`fuzzy_and = a*b`, `fuzzy_or = a+b-a*b`, `fuzzy_xor = a+b-2ab`,
`fuzzy_implies = 1-a+ab`. These probabilistic forms are only sound
for `a, b ∈ [0, 1]`. With `a=2.0, b=-1.0` you get `fuzzy_or = 3.0`
with gradient `∂/∂a = 1-(-1) = 2` — silently. No NaN-fail-closed
(restart 50/51/62 pattern), no clamp, no diagnostic.

The SGD dogfood example (07) survives only because its initialization
stays in [0,1]; user code starting from a uniform-init weight near
0.5 ± 1.0 will drift out of range and the optimizer will diverge
silently with no surfaced signal.

**Fix family**: NaN-fail-closed + range clamp at lowering. Emit a
`CMP` against 0.0/1.0 and either (a) trap 36501 on out-of-range, or
(b) `clamp(a, 0.0, 1.0)` before the algebraic form. Document the
choice in `_BUILTIN_NAMES` comment.

### B1 MEDIUM (conf 70) — `_lower_expr` "return a or b" pattern masks one-side lowering failures

**File**: `helixc/ir/lower_ast.py` — every new 2-arg builtin (lines
1849, 1860, 1873, 1893, 1904, 1916, 1937, 1959, 2017, 2027, 2056,
2071)

The recurring idiom `if a is None or b is None: return a or b`
silently substitutes one operand for the binary result. If
`register_derivation(broken_l, ok_r)` has `a is None`, the function
returns `r`'s SSA value (an `i32`) as the "handle" — and downstream
`parent_left_at(handle)` then dereferences a value that was never an
arena index. The pattern appears 12× in the new code.

**Fix family**: convert the early-return to an explicit `return
None` (or, if downstream tolerates None, log a `_diag()` at the
lowerer that names the failed sub-lowering).

### B2 MEDIUM (conf 75) — `derive(a, b)` evaluates `b` only for side effects but drops the value AND drops true two-parent provenance

**File**: `helixc/ir/lower_ast.py:1837-1845`

Comment says "Evaluate b for side effects but return a's value" —
but Logic-typed expressions are advertised as pure. This drops the
second parent, contradicting the Inc-2 typecheck comment claiming
"two-parent provenance". There's no `register_derivation` call wired
into `derive` itself.

**Fix family**: either (a) make `derive` actually call
`register_derivation` internally and stash the handle (true
two-parent), or (b) add an explicit deprecation/warning at the
typecheck site saying "derive() is single-parent; use
register_derivation+parent_*_at for two-parent tracking".

### C1 LOW (conf 60) — `unwrap_logic` typecheck returns `arg_tys[0]` (a Logic) on error path

**File**: `helixc/frontend/typecheck.py:2728-2735`

When the arg isn't `TyLogic`, the error is appended but the function
returns `arg_tys[0]` unchanged. Downstream type inference sees the
non-Logic type and may report cascading confused errors that hide
the original.

**Fix family**: return `TyUnknown()` on the error path so cascading
is suppressed.

### C2 LOW (conf 80) — 12 builtins in `_BUILTIN_NAMES` but **absent** from `AD_KNOWN_PURE_CALLS`

**Files**: `helixc/frontend/typecheck.py:1838-1854` vs
`helixc/frontend/autodiff.py:58-67`

`derive`, `and_logic`, `or_logic`, `not_logic`, `xor_logic`,
`implies_logic`, `eq_logic`, `if_logic`, `to_logic_bool`,
`register_derivation`, `parent_left_at`, `parent_right_at` are in
`_BUILTIN_NAMES` but absent from `AD_KNOWN_PURE_CALLS`.

If a user writes `grad(loss)` where `loss` transitively calls
`and_logic` or `derive`, the reverse-mode AD pass will hit the
"opaque user call" path and fail closed — which is the right
behaviour, but it's a latent ergonomics gap. The Inc-6 comment
specifically claims fuzzy ops were registered "so grad() flows
through them automatically"; the same justification applies to
`derive`/`and_logic`/`or_logic`/`not_logic`/`if_logic`/`to_logic_bool`
which all lower to AD-pure i32 ops, and they were missed.

**Fix family**: add the boolean-algebra + Inc-5 names to
`AD_KNOWN_PURE_CALLS`, OR (if integer ops are intentionally
non-differentiable) add a comment in `_BUILTIN_NAMES` explaining
the asymmetry.

## Verified clean

- **No catch-all `except Exception`** introduced in the new code —
  checked all 700 added lines.
- **No `INT_MIN` arithmetic** in the new code (the only `idx+1` in
  `parent_right_at` would only overflow at INT32_MAX — possible but
  pre-existing arena bounds issue, not Stage-36-specific).
- **No magic-constant collisions** spotted; new error messages reuse
  the existing `[trap 24100]` Logic-boundary marker correctly.
- **AD forward + reverse chain rules for fuzzy_and/or/xor/implies/not**
  are mathematically correct (verified by hand).
- **`prove(v, src)` lowering as identity** is correctly placed and
  the source tag is correctly excluded from differentiation.
