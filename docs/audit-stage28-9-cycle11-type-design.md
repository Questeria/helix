# Audit Stage 28.9 cycle 11 — Type design

**Scope.** Read-only re-pass at HEAD `48a714e` (cycle-10 C10-1 landed:
`_collect_binds` PatOr arm). Prior dispositions not re-flagged.

**Criterion.** Pass = ZERO findings at confidence >=75%.

## Result: PASS (0 findings >=75%)

### HEAD delta since cycle 10

`48a714e` adds a PatOr arm to `match_lower._collect_binds` plus the
`test_c10_1_*` regression. No AST/type-surface changes
(`A.PatOr.alts: list[Pattern]` unchanged in `ast_nodes.py:264`).

### Re-verified invariants

- `_collect_binds` binder-set agreement with typecheck. Both sides now
  compute the same intersection: typecheck (`typecheck.py:1877-1896`)
  defines only names common to every alt; cycle-11 `_collect_binds`
  PatOr arm (`match_lower.py:458-476`) emits Lets only for that same
  intersection, using `pat.alts[0]`'s slot-load source. Sound: every
  alt sees identical `scrut` at this nesting depth, so any alt's
  load-site is well-typed for the shared name.
- `lower_matches` postcondition (no `A.Match` survives) unchanged —
  `_rewrite_expr` arms (TileLit, Assign.target, UnsafeBlock, Range,
  Modify, Break, Quote, Splice) all intact.
- `PatOr.alts` non-empty guarded by `if pat.alts:` before set ops —
  empty-alts edge handled without exception.

### Stability

No prior-cycle findings re-surface. `_collect_binds` recursion into
nested PatOr inside PatVariant/PatTuple sub-patterns remains absent,
but that is the cycle-10 disposition (depth-1 or-patterns dominate
typecheck's accepted surface) and stays below the bar.

Files: `C:/Projects/Kovostov-Native/helixc/frontend/match_lower.py`,
`C:/Projects/Kovostov-Native/helixc/frontend/typecheck.py`,
`C:/Projects/Kovostov-Native/helixc/frontend/ast_nodes.py`,
`C:/Projects/Kovostov-Native/helixc/tests/test_match.py`.
