# Audit Stage 28.9 cycle 5 — Type design

**Scope.** Type-design pass over cycle-5 commit `f7e7b02` (Stage 28.9
cycle-4 audit-C C4-1 fix: `match_lower._rewrite_expr` now descends into
`A.Assign.target` as well as `.value`). Single 7-line addition at
`helixc/frontend/match_lower.py:146-155`. HEAD at `f7e7b02`.

**Criterion.** Pass = ZERO findings at confidence >=75%.

## Result: PASS (0 findings >=75%)

### Type: `A.Assign` walker contract (match_lower.py:146-155)

`A.Assign` is declared in `ast_nodes.py:328-333` with three fields:
`target: Expr`, `op: str`, `value: Expr`. The cycle-5 fix descends into
both Expr-typed slots before returning the node. Order (`target` first,
then `value`) is irrelevant for the desugaring contract (both are
mutated in place); it merely matches source-textual order. The arm
mirrors the cycle 23 / C22-C defense-in-depth idiom and seals the
final Expr-with-Expr-children gap that prior audits identified as
reachable.

### Expr-subtype coverage (verified end-to-end)

Walked all 32 `Expr` subclasses in `ast_nodes.py`. Every subtype that
holds an `Expr` child now has a corresponding arm. `TileLit.shape`
and `TileLit.memspace` remain unrecursed but are guarded at IR-lower
(IntLit-only shape check at `lower_ast.py:619`; memspace consumed via
`_stringify_marker`) — same disposition as cycle 22 C22-C "Not
reachable" classification.

### Stability

No cycle-1/2/3/4 findings re-flagged. 26 match tests pass at HEAD.

Relevant files:
- `C:/Projects/Kovostov-Native/helixc/frontend/match_lower.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/ast_nodes.py`
