# Audit Stage 28.9 cycle 14 — Type design

**Scope.** Read-only re-pass at HEAD `bb3b5ec` (post structural refactor:
`_pattern_test_expr` canonical impl + `_fresh_slot_load` / `_dup_expr`).
Prior dispositions not re-flagged.

**Criterion.** Pass = ZERO findings at confidence >=75%.

## Result: PASS (0 findings >=75%)

### HEAD delta since cycle 13

`bb3b5ec` deletes the `_sub_pat_or_test` helper and the legacy
`_pattern_test` body, introducing `_pattern_test_expr(pat, scrut_expr,
span)` as the canonical dispatch (`match_lower.py:323-393`).
`_pattern_test(pat, scrut: str, span)` becomes a thin wrapper. Helpers
`_fresh_slot_load` / `_dup_expr` enforce tree linearity. No AST/type
surface change: `Pattern` hierarchy, `PatOr.alts`, `PatTuple.elems`,
`PatVariant.sub_patterns` unchanged.

### Re-verified invariants

- Wrapper/canonical type signatures consistent: `_pattern_test` returns
  `A.Expr`, `_pattern_test_expr` returns `A.Expr`; both postconditions
  preserved on every arm (`BoolLit | Binary`).
- Tree linearity invariant now structurally enforced via
  `_fresh_slot_load` and `_dup_expr` at every reuse of `scrut_expr`
  (`:346, :352, :354, :362, :377, :382`) — closes C13-2 at the type
  boundary, not by convention.
- `_or_chain` / `_and` shape preserved; `BoolLit(True)` short-circuit
  in PatTuple / PatVariant arms is type-stable (still `A.Expr`).
- `lower_matches` postcondition (no `A.Match` survives) unaffected by
  the refactor; 29 match tests including C12-1 + C11-1 + C10-1 + C7-1
  + C4-1 + C22-C regression suite still pin it.

### Stability

No prior-cycle findings re-surface. Refactor is strictly subtractive on
the type surface (162 lines removed, 86 added; one helper deleted).

Files: `C:/Projects/Kovostov-Native/helixc/frontend/match_lower.py`,
`C:/Projects/Kovostov-Native/helixc/frontend/ast_nodes.py`.
