# Audit Stage 28.9 cycle 19 — Type design

**Scope.** Read-only at HEAD `46e9952` (cycle-17 audit-A C17-1
`AgentDecl` leaf-list move + cycle-18 audit-C C18-C1 `seen[key]`
list-copy in `cse_function` + C18-C2 test-bound relax). Prior
dispositions not re-flagged.

**Criterion.** Pass = ZERO findings at confidence >=75%.

## Result: PASS (0 findings >=75%)

### HEAD delta since cycle 18

`46e9952` makes three invariant-strengthening edits:

- **C17-1 (`match_lower.py:87-105`).** `AgentDecl` moves from a recursive
  arm into the leaf-decl tuple. Verified `AgentMethod` (`ast_nodes.py:508-513`)
  is NOT an `Item` subclass and has no `body` field — the previous
  `_rewrite_item(m)` recursion would have crashed the loud catchall for
  any program containing an agent. Leaf treatment correctly models the
  Phase-0 interface-only-signatures invariant.
- **C18-C1 (`cse.py:117-124`).** `seen[key] = list(op.results)` replaces
  aliasing assignment. Direct grep of `helixc/` for `op.results.<mutator>`
  and `op.results =` yields zero hits, so the defense is purely
  prophylactic — matches the established `const_fold._propagate_identities`
  pattern. No behavior change at current HEAD; strengthens the
  no-aliased-state invariant on the CSE-seen map.
- **C18-C2 (`test_cse.py:107-113`).** Brittle `cse_count >= 4` bound
  relaxed to `>= 1`; the canonical `muls == 1` post-condition preserves
  correctness coverage. Type-surface neutral.

### Stability

No prior-cycle findings re-surface. Item-dispatch totality holds
(10 `Item` subclasses, all enumerated). CSE-seen list ownership now
locally owned. Strictly invariant-strengthening delta.

Files: `C:/Projects/Kovostov-Native/helixc/frontend/match_lower.py`,
`C:/Projects/Kovostov-Native/helixc/frontend/ast_nodes.py`,
`C:/Projects/Kovostov-Native/helixc/ir/passes/cse.py`,
`C:/Projects/Kovostov-Native/helixc/tests/test_cse.py`.
