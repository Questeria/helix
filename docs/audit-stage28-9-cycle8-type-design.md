# Audit Stage 28.9 cycle 8 — Type design

**Scope.** Type-design pass at HEAD `fdbcfc5` (Stage 28.9 cycle-7 C7-1
fix: `match_lower._rewrite_expr` gains a `TileLit` arm that recurses
into `shape` and `memspace`, plus a `test_c7_1_*` regression test).
Read-only.

**Criterion.** Pass = ZERO findings at confidence >=75%. Prior-cycle
dispositions are not re-flagged.

## Result: PASS (0 findings >=75%)

### New surface at HEAD

- `helixc/frontend/match_lower.py:210-222` — new `A.TileLit` arm.
  Mutates `expr.shape` (list[Expr]) elementwise and `expr.memspace`
  (Expr), matching the established `TupleLit`/`ArrayLit` shape and
  the same defect-class fixes applied across UnsafeBlock / Range /
  Modify / Break / Quote / Splice.
- `helixc/tests/test_match.py:537-604` — direct AST construction,
  `lower_matches`, recursive walker asserting no `A.Match` survives.

### Invariant re-verification

- `lower_matches` postcondition ("no `A.Match` in AST") now pinned
  for every `Expr` subtype that holds Expr children, including
  `TileLit`. Walker arm count matches `Expr` subtype count modulo
  trivially-leaf nodes — no new gaps introduced.
- `TileLit.shape` / `TileLit.memspace` types are unchanged
  (`list[Expr]` / `Expr`); the cycle-5 "compile-time gate at
  `lower_ast._tile_shape_dims` makes the latent path unreachable
  in well-typed input" disposition stands. The cycle-7 arm is
  documented as defense-in-depth; type design is not weakened.

### Stability

No cycle-1 through cycle-7 findings re-flagged. The duplicated 8-line
`walk` closure between regression tests was already disposed by
cycles 5 / 6 / 7 and stays below the 75% bar. No new production type
surface beyond the single TileLit arm.

Relevant files:
- `C:/Projects/Kovostov-Native/helixc/frontend/match_lower.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/ast_nodes.py`
- `C:/Projects/Kovostov-Native/helixc/tests/test_match.py`
