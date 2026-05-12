# Audit Stage 28.9 cycle 17 — Type design

**Scope.** Read-only at HEAD `8ed65a5` (cycle-16 audit-A C16-1: outer
item-walker `_rewrite_item` with explicit Item-subclass arms + loud
`NotImplementedError` catchall). Prior dispositions not re-flagged.

**Criterion.** Pass = ZERO findings at confidence >=75%.

## Result: FAIL (1 finding @ conf 85)

### C17-T1 — `TypeAlias` regression in `_rewrite_item` catchall (conf 85)

`match_lower._rewrite_item` (`match_lower.py:58-107`) lists explicit
arms for `FnDecl, ConstDecl, ImplBlock, ModBlock, AgentDecl,
StructDecl, EnumDecl, ModuleDecl, UseDecl` and routes everything else
to the loud `NotImplementedError` catchall. But `A.TypeAlias`
(`ast_nodes.py:486`, an existing `Item` subclass constructed by
`parser.py:569` and `flatten_modules.py:120`) is missing from the
explicit-arm list.

**Effect.** Any program containing `type Foo = Bar;` now raises
`NotImplementedError("unhandled Item subclass TypeAlias …")` during
match-lowering — even when no `match` appears. The catchall's intent
is to flag *unknown future* subclasses, not silently break a known
present one. Cycle-16's "29 match tests still pass" check did not
cover TypeAlias coexistence.

**Disposition.** Add `TypeAlias` to the leaf-decl pass-through tuple
on line 91 (its `target` is `TyNode`, not `Expr`).

Files: `C:/Projects/Kovostov-Native/helixc/frontend/match_lower.py`,
`C:/Projects/Kovostov-Native/helixc/frontend/ast_nodes.py`.
