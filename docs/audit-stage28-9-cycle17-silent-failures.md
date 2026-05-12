# Stage 28.9 Cycle 17 — Audit A (silent failures)

**Date:** 2026-05-11. **HEAD:** `8ed65a5` (post C16-1 outer item-walker + audit-C C1 FFI sentinel + C2 fold count). **Lens:** silent failures. **Criterion:** ZERO findings at conf >=75 (strict). **Going in:** 1/5 (cycle 16 not clean).

## Scope

Read-only delta `f086023..8ed65a5` (`match_lower.py` +53/-2, `const_fold.py` +14/-6, `effect_check.py` +15/0). No re-flag of C1–C16 cleared findings.

## Finding C17-1 — `_rewrite_item` recurses on `AgentMethod` which is NOT an `Item` subclass

**Severity:** HIGH. **Confidence:** 92. **Location:** `helixc/frontend/match_lower.py` lines 87-90.

**Issue.** The AgentDecl arm assumes `item.methods` are `FnDecl`s and recurses via `_rewrite_item(m)`:

```python
elif isinstance(item, A.AgentDecl):
    # AgentDecl.methods are FnDecls; recurse.
    for m in item.methods:
        _rewrite_item(m)
```

The comment, the C16-1 docstring (lines 64, 71), and the commit message all assert "AgentDecl.methods[].body" — but `ast_nodes.py:507-513` defines `AgentMethod` as a bare dataclass (NOT an `Item` subclass) with only `span`/`name`/`params`/`return_ty`. There is no `body` field. The parser at `parser.py:610-613` eats a `;` after the signature.

`_rewrite_item` dispatch on an `AgentMethod` instance falls through every explicit arm and hits the loud `NotImplementedError` catchall at line 102. Pre-C16-1 the outer walker silently skipped AgentDecls entirely (the original silent-failure C16-1 targeted). The fix over-corrected: it converted a silent skip into an unconditional crash. The C16-1 docstring's claim that AgentMethod carries Match-bearing exprs is structurally false — signatures cannot contain expressions at Phase-0 (params have no defaults, return_ty is a TyNode).

**Hidden errors.** Any user program with `agent Planner { fn propose(s: i32) -> i32; }` reaching `lower(prog)` now raises `NotImplementedError: _rewrite_item at L:C: unhandled Item subclass AgentMethod` from `match_lower.py:107`. The traceback points at `match_lower.py`, not at the user's agent declaration — debugging requires the reader to know AgentMethod's structure. No test exercises an AgentDecl through `lower_matches` (only `test_parser.py:430-466`), so the regression is latent. The same loud catchall that protects against future Item subclasses now misfires on a non-Item type the recursion shouldn't even reach.

**Recommendation.** AgentDecl is a leaf for match_lower (signatures hold no Exprs). Move `A.AgentDecl` into the leaf-decl tuple at line 91, alongside `StructDecl`/`EnumDecl`/`ModuleDecl`/`UseDecl`/`TypeAlias`, and remove the broken recursion. Update the C16-1 docstring (lines 64-71) and commit message — both claim AgentDecl.methods[].body. If future Phase-N AgentMethods grow bodies, add an explicit `for m in item.methods: _rewrite_block(m.body)` arm at that point (not before).

## Tally

| Severity | Count |
|---|---|
| HIGH | 1 |
| **Total** | **1** |

**Cycle 17: NOT CLEAN.** C16-1's exhaustive outer-walker fix introduced a structural mismatch: AgentMethod is treated as a recursable Item but isn't one. Counter resets — going in 1/5, ending 0/5.

## Files touched

None — read-only. Only this doc.
