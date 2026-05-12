# Stage 28.9 Cycle 16 — Audit A (silent failures)

**Date:** 2026-05-11. **HEAD:** `f086023` (post C15-1/2/3: loud `_collect_binds` catchall, `_dup_expr` span param dropped, span-in-message catchall). **Lens:** silent failures. **Criterion:** ZERO findings at conf >=75 (strict). **Going in:** 0/5.

## Scope

Read-only delta `e847fa9..f086023` (`match_lower.py` +42/-8). No re-flag of C10–C15 cleared findings (PatOr binders, slot_load sharing, span loss, span-param no-op, BoolLit-True catchall, _collect_binds silent-[]).

## Finding C16-1 — `lower_matches` top-level walker silently skips non-`FnDecl` items containing Match nodes

**Severity:** HIGH. **Confidence:** 86. **Location:** `helixc/frontend/match_lower.py` lines 52-56.

**Issue.** The entry walker only descends into `A.FnDecl.body`:

```python
for item in prog.items:
    if isinstance(item, A.FnDecl):
        _rewrite_block(item.body)
```

Other `Item` subclasses that hold `Expr` children — `ConstDecl.value` (top-level `const X = match flag { ... };`), `ImplBlock.methods`, `ModBlock.items`, `AgentDecl.methods` — are silently skipped. flatten_modules/flatten_impls partially mitigate (lifting methods to top-level FnDecls), but: (a) flatten passes run only on the **codegen path** (`backend/x86_64.py:3097-3102`), not unconditionally before every `lower_matches` invocation; (b) `grad_pass` invokes `lower_matches` *inline* without guaranteeing flatten ran first; (c) `ConstDecl` is **never** flattened to FnDecl — its `value` Expr stays put. The author of `lower_ast.py` flagged exactly this in a comment at line 1905-1907 ("Match nodes inside ConstStmt or other items would slip through") but the source-side fix never landed. A `Match` surviving past lower_matches trips the AssertionError at `lower_ast.py:1908` — which is loud, but the `lower_matches` walker itself is silent about which item it failed to visit.

**Hidden errors.** `const X = match cfg { ... };` compiles successfully through parser+typecheck, then surfaces as an opaque "A.Match should not reach _lower_expr" assertion from deep in the IR pipeline. `impl Foo { fn m() { match ... } }` works only when the codegen-path flatten_impls precedes lower_matches; if any future call site (test, REPL, autodiff pre-pass) invokes lower_matches without flattening first, the inner Match silently survives. Same defect family as C4-1 (Assign target) and C7-1 (TileLit) — the **inner** expression walker `_rewrite_expr` was extended; the **outer** item walker was not.

**Recommendation.** Make the top-level walker exhaustive over Item subclasses that hold Expr children. Mirror C14-3 / C15-1: explicit `if`-arms for `FnDecl`/`ConstDecl`/`ImplBlock`/`ModBlock`/`AgentDecl` plus a trailing `raise NotImplementedError(f"lower_matches: unhandled Item {type(item).__name__}")` so a new Item subclass surfaces immediately. At minimum, descend into `ConstDecl.value` via `_rewrite_expr` and recurse into `ImplBlock.methods`/`AgentDecl.methods`/`ModBlock.items` regardless of whether flatten passes ran.

## Tally

| Severity | Count |
|---|---|
| HIGH | 1 |
| **Total** | **1** |

**Cycle 16: NOT CLEAN.** C15 closed the inner-dispatcher silent-accept and the span-param no-op. The outer item walker remains the symmetric gap one level up: a structural mirror of C4-1/C7-1/C22-C, acknowledged in a downstream comment but unfixed at source. Counter holds at 0/5.

## Files touched

None — read-only. Only this doc.
