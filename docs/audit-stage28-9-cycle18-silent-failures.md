# Audit Stage 28.9 cycle 18 — Silent failures

**Scope.** Read-only at HEAD `46e9952` (post C17-1 AgentDecl
leaf-tuple move + C18-C1 `seen[key] = list(op.results)` shallow
copy + C18-C2 cse_count threshold relax). Delta `40d2767..46e9952`:
`match_lower.py` +11/-6, `cse.py` +8/-1, `test_cse.py` +6/-2.
Prior C1–C17 dispositions not re-flagged.

**Criterion.** Pass = ZERO findings at conf >=75% (strict).

## Result: PASS (0 findings >=75%)

## Verified

- **C17-1 disposition holds.** `_rewrite_item` AgentDecl arm now in
  leaf tuple (`match_lower.py:87-88`). `agent X { fn f(); }`
  programs no longer crash the catchall.
- **C18-C1 shallow copy semantics.** `seen[key] = list(op.results)`
  isolates dict-stored list from later `op.results` mutation. Grep
  confirms no production site mutates `op.results` today, so the
  fix is defensive-in-depth, not closing an observable leak.
- **C18-C2 test resilience.** `cse_count >= 1` plus `muls == 1`
  retains correctness check while removing implementation-detail
  brittleness.
- **Item enumeration totality.** All 10 `Item` subclasses dispatched.

## Notes (below threshold)

The C18-C1 comment references "defensive list-copy pattern in
`const_fold._propagate_identities`" — no such `list(.results)`
pattern exists there (only `list(subst.items())` for safe dict
iteration). Misleading audit-trail comment (conf ~70).

## Files touched

None — read-only. Only this doc.
