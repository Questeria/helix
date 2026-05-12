# Audit Stage 28.9 cycle 18 — Type design

**Scope.** Read-only re-pass at HEAD `40d2767` (post C17-T1
`_rewrite_item` leaf-decl pass-through tuple now includes `A.TypeAlias`).
Prior dispositions not re-flagged.

**Criterion.** Pass = ZERO findings at confidence >=75%.

## Result: PASS (0 findings >=75%)

### HEAD delta since cycle 17

`40d2767` adds `A.TypeAlias` to the leaf-decl pass-through tuple in
`_rewrite_item` (`match_lower.py:91-92`), strengthens the surrounding
comment to document its `TyNode`-only `target` field, and annotates the
fix's audit-trail (C17-T1 conf 85). 6 insertions, 1 deletion.

### Re-verified invariants

- **Item-dispatch totality.** Direct grep of
  `ast_nodes.py` for `class \w+\(Item\)` yields exactly 10 subclasses:
  `FnDecl, StructDecl, EnumDecl, TypeAlias, UseDecl, ConstDecl,
  AgentDecl, ModuleDecl, ModBlock, ImplBlock`. All 10 are now
  enumerated in `_rewrite_item`'s explicit arms; the
  `NotImplementedError` catchall guards only genuinely-new future
  subclasses, as designed.
- **Catchall semantic preserved.** The leaf tuple receives only
  subclasses whose fields are TyNode/metadata (no Expr children); the
  C17-T1 addition fits the pattern (`TypeAlias.target: TyNode`,
  `ast_nodes.py:486-490`). No silent-accept regression.
- **Aspirational comment.** The note "TraitDecl methods are
  interface-only" references a class that does not currently exist
  (`grep` confirms no `TraitDecl` definition); harmless documentation,
  not a live invariant claim.

### Stability

No prior-cycle findings re-surface. Strictly invariant-strengthening
delta on the Item-walker type surface.

Files: `C:/Projects/Kovostov-Native/helixc/frontend/match_lower.py`,
`C:/Projects/Kovostov-Native/helixc/frontend/ast_nodes.py`.
