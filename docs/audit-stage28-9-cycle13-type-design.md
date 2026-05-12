# Audit Stage 28.9 cycle 13 — Type design

**Scope.** Read-only re-pass at HEAD `7108733` (C12-1 lands
`_sub_pat_or_test` + dispatch arms; C12-2 lands regression test).
Prior dispositions not re-flagged.

**Criterion.** Pass = ZERO findings at confidence >=75%.

## Result: PASS (0 findings >=75%)

### HEAD delta since cycle 12

`7108733` adds `match_lower._sub_pat_or_test` (`match_lower.py:291-348`)
and dispatches to it from the `PatOr` sub-arms inside `PatTuple.elems`
and `PatVariant.sub_patterns` (`:407-418`, `:467-474`). No AST/type
surface change: `PatOr.alts: list[Pattern]` and the parent
`sub_patterns`/`elems: list[Pattern]` unchanged.

### Re-verified invariants

- Helper return type `A.Expr | None` is honored at both call sites
  (None-check before append), so the `sub_tests` list type stays
  `list[A.Expr]`.
- `_pattern_test` postcondition (returns a boolean A.Expr) preserved
  on every arm; the new `||` chains close left-associatively, matching
  the existing `_or_chain` shape used at the top-level PatOr arm.
- Approximation (nested PatTuple/PatOr alt -> BoolLit(True)) is local
  to the helper, narrower than the pre-fix silent-true, and explicitly
  annotated as deeper-recurse follow-up — does not regress any
  invariant the cycle-12 audit re-verified.
- `lower_matches` postcondition (no `A.Match` survives) is pinned by
  the new `test_c12_1_nested_pat_or_in_tuple_sub_test_emitted`.

### Stability

No prior-cycle findings re-surface.

Files: `C:/Projects/Kovostov-Native/helixc/frontend/match_lower.py`,
`C:/Projects/Kovostov-Native/helixc/frontend/ast_nodes.py`,
`C:/Projects/Kovostov-Native/helixc/tests/test_match.py`.
