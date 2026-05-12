# Audit Stage 28.9 cycle 12 — Type design

**Scope.** Read-only re-pass at HEAD `4ad80fa` (cycle-11 C11-1 landed:
`A.PatOr` added to `_collect_binds` recurse-tuples in PatVariant and
PatTuple sub-pattern arms). Prior dispositions not re-flagged.

**Criterion.** Pass = ZERO findings at confidence >=75%.

## Result: PASS (0 findings >=75%)

### HEAD delta since cycle 11

`4ad80fa` extends two `isinstance(sub, (A.PatVariant, A.PatTuple))`
tuples to `(A.PatVariant, A.PatTuple, A.PatOr)` at
`match_lower.py:431` and `match_lower.py:458`. No AST/type-surface
changes: `A.PatOr.alts: list[Pattern]` unchanged
(`ast_nodes.py:264-266`); `PatVariant.sub_patterns` and `PatTuple.elems`
already type as `list[Pattern]` so `Pattern` was always the static
upper bound — the isinstance tuple is a runtime dispatch table, not a
type narrowing constraint.

### Re-verified invariants

- Binder-set / typecheck agreement preserved at any nesting depth.
  The cycle-11 fix routes nested PatOr through the same temp-bind
  recurse path used for PatVariant/PatTuple, so the top-level PatOr
  arm (`match_lower.py:472-490`) computes the binder intersection
  identically whether reached directly or via a sub-position temp.
- Recursion termination: `Pattern` is a finite tree; each recurse
  step at lines 444/471 descends to a strictly smaller sub-pattern,
  so `_collect_binds` still terminates.
- Fresh-name discipline: each nested sub gets its own `__sub` via
  `_fresh_name`, so the binder-source `scrut` strings stay distinct
  across nesting levels; no shadow risk in the emitted Lets.
- `lower_matches` postcondition (no `A.Match` survives) unchanged.

### Stability

No prior-cycle findings re-surface. The recurse-tuple dispatch is now
exhaustive for the three composite Pattern variants that can carry
binders; PatBind (leaf) and value-only patterns (PatLit, PatWildcard,
PatRange) remain correctly handled in their own arms.

Files: `C:/Projects/Kovostov-Native/helixc/frontend/match_lower.py`,
`C:/Projects/Kovostov-Native/helixc/frontend/ast_nodes.py`,
`C:/Projects/Kovostov-Native/helixc/frontend/typecheck.py`.
