# Audit Stage 28.9 cycle 15 — Type design

**Scope.** Read-only re-pass at HEAD `e847fa9` (post C14-1 span-preserve +
C14-2 right-operand dup + C14-3 loud catchall). Prior dispositions not
re-flagged.

**Criterion.** Pass = ZERO findings at confidence >=75%.

## Result: PASS (0 findings >=75%)

### HEAD delta since cycle 14

`e847fa9` modifies `_dup_expr` (preserve `expr.span`; add `A.Path` arm),
wraps right-operands of `==`/`>=`/`<=`/`<` Binaries in `_dup_expr`, and
replaces the trailing `BoolLit(True)` catchall in `_pattern_test_expr`
with `NotImplementedError`. No AST/type surface change: `Pattern`
hierarchy and `_dup_expr`/`_fresh_slot_load` signatures unchanged.

### Re-verified invariants

- `_dup_expr` postcondition strengthened: every explicit arm now returns
  a node whose `.span` equals the input's `.span`, matching the deepcopy
  fallback (`match_lower.py:319-335`). `span` parameter degenerates to
  unused legacy default — annotated in docstring.
- Tree linearity now total: BOTH left and right operands of every
  Binary built in `_pattern_test_expr` route through `_dup_expr`
  (`:362-364, :370-375, :402-404`). No shared `pat.value`/`pat.lo`/
  `pat.hi`/`pat.path` reachable from emitted Binaries.
- Dispatch totality enforced at type boundary: unhandled `Pattern`
  subclass now raises `NotImplementedError` (`:423-427`) — silent-true
  anti-pattern eliminated; new subclasses must declare semantics.
- `A.Path` explicit branch preserves `segments` as a fresh list copy
  (`:332`) — no aliasing of the segment sequence either.

### Stability

No prior-cycle findings re-surface. Refactor is strictly invariant-
strengthening on the type surface (50 insertions, 16 deletions).

Files: `C:/Projects/Kovostov-Native/helixc/frontend/match_lower.py`.
