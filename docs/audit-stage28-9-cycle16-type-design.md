# Audit Stage 28.9 cycle 16 — Type design

**Scope.** Read-only re-pass at HEAD `f086023` (post C15-1 `_collect_binds`
loud catchall + C15-2 `_dup_expr` legacy `span` param removal + C15-3
span-informed error messages). Prior dispositions not re-flagged.

**Criterion.** Pass = ZERO findings at confidence >=75%.

## Result: PASS (0 findings >=75%)

### HEAD delta since cycle 15

`f086023` adds explicit `PatLit | PatRange | PatWildcard` pass-through arm
+ `NotImplementedError` catchall to `_collect_binds`
(`match_lower.py:537-555`); drops the legacy unused `span` parameter from
`_dup_expr` (`:303`) and updates its sole caller `_fresh_slot_load`
(`:298`); enriches both `NotImplementedError` raises with `pat.span.line:col`
context (`:429, :549`). No AST/type surface change: `Pattern` hierarchy
and dispatcher return types unchanged.

### Re-verified invariants

- Dispatch totality now symmetric across BOTH lowering dispatchers:
  `_pattern_test_expr` (`:340-435`) and `_collect_binds` (`:452-556`)
  both terminate every `Pattern` subclass in an explicit arm or
  `NotImplementedError` — silent-accept class fully closed.
- `_dup_expr` signature minimal: single `expr: A.Expr` parameter; no
  vestigial unused arg. Span-of-truth is `expr.span` per C14-1.
- Internal-error messages carry source location, preserving
  `NotImplementedError` semantics (compiler bug, file issue) while
  enabling clean downstream diagnostic rendering.

### Stability

No prior-cycle findings re-surface. Refactor is strictly invariant-
strengthening on the type surface (42 insertions, 8 deletions).

Files: `C:/Projects/Kovostov-Native/helixc/frontend/match_lower.py`.
