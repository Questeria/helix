# Stage 36 Post-Increment-10 Audit — Silent-Failure Lane

**Date**: 2026-05-16
**Auditor**: pr-review-toolkit:silent-failure-hunter
**HEAD audited**: `821592f` (Stage 36 Inc 10 — knowledge-graph dogfood) + `14e1fa4` (Inc 9 catch-up ledger)
**Baseline**: `a451591` (Stage 36 Increment 8)
**Scope**: `git diff a451591..821592f -- helixc/{frontend,ir,backend,examples,tests}` + Inc 9 catch-up sweep verification
**Status**: **NOT CLEAN — 1 HIGH + 1 MEDIUM + 1 LOW**

## Findings

### H1 HIGH (conf 85) — `derive` / `register_derivation` in `AD_KNOWN_PURE_CALLS` silently erase arena pushes inside `grad()` bodies

**File**: `helixc/frontend/autodiff.py:76-79` (added by `0e548f0`, the Inc 9 C2 LOW fix)

Commit `0e548f0` added `derive`, `register_derivation`, `parent_left_at`, `parent_right_at` to `AD_KNOWN_PURE_CALLS`. The C2 comment justifies this as "they lower to AD-pure i32 ops". But **two commits later**, `707deff` (Inc 9 B2 fix) gave `derive` and `register_derivation` an observable side effect: both now emit `ARENA_PUSH_PAIR` (categorized as `"arena"` effect in `helixc/ir/passes/effect_check.py:96`). The categorization in `AD_KNOWN_PURE_CALLS` was never revisited.

Consequence: inside any function reached by `grad(...)` or `grad_rev(...)`, the inliner at `autodiff.py:617 _inline_lets` calls `_raise_if_ad_erases_effect(stmt.value, ...)` (line 650). That check delegates to `_is_ad_erasable_expr` (line 537), which returns `True` for any `A.Call` whose callee name is in `AD_KNOWN_PURE_CALLS`. So a `let _h = register_derivation(p1, p2);` whose name `_h` is never referenced downstream is silently inlined into `local_env` and then dropped from the differentiated expression — the `ARENA_PUSH_PAIR` is never emitted, the arena never grows, and subsequent `parent_*_at` lookups in non-AD code silently return -1 with no diagnostic. This is the exact silent-failure pattern the Inc 9 B2 fix was supposed to close.

**Recommended fix**: Remove `derive` and `register_derivation` from `AD_KNOWN_PURE_CALLS` (keep `parent_left_at`/`parent_right_at` — those are pure reads). Add a comment at the deletion site noting that the B2 ARENA_PUSH_PAIR side effect makes them ineligible. If grad-erasability is genuinely needed for these two, gate it on "callsite result is consumed downstream" rather than name-set membership.

### M1 MEDIUM (conf 70) — `dogfood_09_knowledge_graph.hx` witness is collapsible: `count * 21 * ev_ok = 42` accepts multiple broken code paths

**File**: `helixc/examples/dogfood_09_knowledge_graph.hx:68`; test at `helixc/tests/test_reflection.py:229-240`

Exit space is `{0, 21, 42}`; only `(count=2, ev_ok=1)` yields 42. But `count = unwrap_logic(gp_ac) + unwrap_logic(gp_ad)` collapses the two rule-firings into a sum. Concretely, if `and_logic` were silently lowered as `or_logic` (BIT_OR instead of BIT_AND), every all-TRUE input yields 1 for both, count=2, exit=42 — the test still passes. The same collapse hides a swap of `gp_ac` and `gp_ad` (since both expected truth values are 1) and a swap of `register_derivation` arg order in the (1,3) case if combined with a matching parent_*_at swap (lp_ad=3, rp_ad=1 wouldn't pass, but more subtle reorderings could). The test does not exercise the 0-based-vs-1-based handle distinction either: with pre-Inc-9 0-based handles, `h_ac=0, h_ad=2`, the arena lookups would still recover (1,2,1,3) and exit 42.

**Recommended fix**: change the witness to be per-rule and per-handle, e.g. `gp_ac * 10 + gp_ad * 10 + (lp_ac==1)*1 + (rp_ac==2)*2 + ... + h_ac*0 + h_ad*0` chosen so any single bit flip produces a distinguishable non-42 exit. Or, more directly, assert `h_ac == 1` (regression for the 1-based-handle invariant) inside the program with a separate trap arm.

### L1 LOW (conf 55) — `parent_left_at(idx) where idx is None` returns `idx` (i.e., `None`) — same arm as M1 caller path

**File**: `helixc/ir/lower_ast.py:2100-2101` and `2110-2112`

The post-B1 cleanup converted the binary builtin arms to `return None` on `is None`. The 1-arg `parent_left_at` / `parent_right_at` arms instead write `if idx is None: return idx`. Semantically identical (`idx is None ⇒ return None`), but stylistically inconsistent with the surrounding `return None` convention and slightly harder to grep for. Not a functional bug.

**Recommended fix**: convert to `return None` for grep symmetry with B1.

## Verified clean (substantive checks performed)

- **B1 sweep complete**: `grep "return a or b|return l or r|return t or e"` in `lower_ast.py` → 0 matches. All 11 sites flagged in the post-Inc-8 B1 finding are now `return None`.
- **A1 bounds check**: `parent_left_at` / `parent_right_at` both route through `_safe_arena_get` (lower_ast.py:2028) with `CMP_GE 0` + `CMP_LT arena_len` + `SELECT` to a -1 sentinel. No bare `ARENA_GET` with user-supplied index remains.
- **A2 1-based handles**: `register_derivation` returns `ARENA_PUSH_PAIR + 1`; `parent_*_at` subtracts 1 before lookup. Handle 0 routes to -1 sentinel via the bounds check (0 - 1 = -1 fails `>= 0`).
- **A2 atomicity**: `ARENA_PUSH_PAIR` is a single fused opcode (`helixc/ir/tir.py`); DCE keep-alive set updated (`helixc/ir/passes/dce.py`); effect_check classifies as `"arena"`.
- **A3 fuzzy clamp**: all four fuzzy ops (`fuzzy_and/or/xor/implies`) route inputs through `_clamp_unit_f32` (lower_ast.py:2134) before the algebraic form. Out-of-[0,1] inputs no longer produce nonsense.
- **B2 derive observable**: `derive` now emits `ARENA_PUSH_PAIR(a_v, b_v)` (lower_ast.py:1873) for the side effect, returns `a_v` as the user-visible value.
- **B3 prove src must be IntLit**: both `autodiff.py:1130-1140` and `autodiff_reverse.py:570-582` raise `NotImplementedError` with a pointer to `register_derivation` when `prove(x, non_lit)` is encountered in an AD context.
- **C1 unwrap_logic recovery**: returns `TyUnknown()` on type-error path; **C2 derive recovery** same. No cascading wrong-type errors.
- **B1-type-design prove flatten**: `prove(Logic<T>, src)` now rejected with diagnostic instead of silently dropping the new tag.
- **B4 to_logic_bool i32-strict**: rejects non-i32 inner types.
- **No new `except Exception` introduced** in any of the 9 catch-up commits or Inc 10.
- **No new magic-constant exit codes** in the dogfood; trap IDs unchanged.
- **Test `test_dogfood_09_knowledge_graph` does pin behavior** (assert == 42), but the witness is collapsible — see M1.

## Verification steps

1. Read prior audit (`docs/audit-stage36-postinc8-silent-failures.md`) and all 9 catch-up commit messages.
2. Diffed each commit's `lower_ast.py` / `autodiff.py` / `autodiff_reverse.py` / `typecheck.py` changes against the prior audit's HIGH/MEDIUM/LOW findings.
3. Confirmed `_is_ad_erasable_expr` → `_raise_if_ad_erases_effect` call path on `derive`/`register_derivation` membership in `AD_KNOWN_PURE_CALLS`.
4. Enumerated exit-code space for `dogfood_09_knowledge_graph.hx` and constructed two distinct broken-but-still-42 code paths (and_logic→or_logic; 0-based handles regression).
5. Inspected `effect_check.py` to confirm `ARENA_PUSH_PAIR` is classified `"arena"` (i.e., side-effectful) — confirming the H1 inconsistency.
