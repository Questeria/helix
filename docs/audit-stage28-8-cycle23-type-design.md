# Stage 28.8 pre-29 audit gate — Cycle 23 (Audit B: type-design soundness)

**Date:** 2026-05-11
**HEAD:** `4bdc800` ("Cycle 22 Audit C C0: delete dead visit_stmt shim in
struct_mono")
**Lens:** type-design soundness (Audit B)
**Streak counter at start:** 2/5 (cycle 21 first clean, cycle 22 A+B
clean)
**Bar:** ZERO findings of ANY severity at confidence >= 75. Re-flagging
prior-cycle findings is forbidden; manufacturing findings is forbidden.

---

## Scope — delta-only

Cycle 22 type-design audited four targets exhaustively under the strict
criterion and returned CLEAN at HEAD `bee36e6`:

1. `helixc/frontend/ast_walker.py` field-introspection safety
2. `helixc/backend/x86_64.py:_op_suffix` collision potential
3. isize/usize cross-pass consistency (typecheck, lower_ast, const_fold,
   x86_64, PTX)
4. Deferred `grad_pass` rewriter type-soundness gap

The only commit between cycle-22 HEAD `bee36e6` and cycle-23 HEAD
`4bdc800` is `4bdc800` itself:

```
4bdc800 Cycle 22 Audit C C0: delete dead visit_stmt shim in struct_mono
 helixc/frontend/struct_mono.py | 8 ++++----
 1 file changed, 4 insertions(+), 4 deletions(-)
```

Cycle 23's scope reduces to: **does the C-fix introduce any type-design
soundness regression?**

---

## C-fix delta inspection

`git show 4bdc800 -- helixc/frontend/struct_mono.py`:

- Removed: a 4-line nested function `visit_stmt(s) -> None` that called
  `_body_visitor.visit(s)`. Defined inside `collect_concrete_uses`
  closure scope.
- Added: a 4-line comment noting the deletion + the documented escape
  hatch (call `_body_visitor.visit(stmt)` directly if needed).

Net change is **pure deletion of inert code**. No new types, no changed
function signatures, no changed dataclass fields, no new dispatch paths,
no new invariants introduced or removed.

### Caller-graph verification

`Grep visit_stmt` across the entire repo: one hit — the deletion-note
comment in `struct_mono.py:186`. Zero call sites (this is what makes the
removal safe).

`Grep visit_expr` in `struct_mono.py`: three hits — the function
definition (line 177), and exactly two callers (lines 199, 205) inside
the `for it in prog.items` loop:

```
199:                visit_expr(it.body)        # FnDecl body walk
205:            visit_expr(it.value)            # ConstDecl init walk
```

This matches the docstring's claim ("two remaining call sites") and
confirms the C-fix did not accidentally orphan `visit_expr` along with
`visit_stmt`. The class `_BodyVisitor` and its overrides
(`visit_Cast`, `visit_Name`, `visit_TileLit`, `visit_Let`,
`visit_ConstStmt`) are unaffected.

### Invariant-preservation check

`collect_concrete_uses` returns a `list[tuple[str, list[A.TyNode]]]` of
deduplicated (struct_name, type_args) pairs for monomorphization. Its
invariants:

- I1: every returned pair has `struct_name in generic_structs`.
- I2: every type arg in every returned pair is fully concrete (no
  unsubstituted type variables remain at the point this list is
  consumed by `instantiate`).
- I3: the dedup `seen` set covers every distinct
  `(struct_name, tuple(_ty_key(arg) for arg in args))` exactly once.
- I4: every reachable generic-struct use site (signature types,
  field types, body Let/Cast/Name/TileLit/ConstStmt) is visited.

The deleted `visit_stmt` shim had zero callers, so its removal cannot
affect I4 (no use-site previously routed through it). I1–I3 depend on
`visit_ty` (unchanged) and `_ty_key` (unchanged). All four invariants
are preserved.

### `_BodyVisitor` contract is intact

The `ASTVisitor`-based body walker still:

- Returns `None` from all five overrides (`visit_Cast`, `visit_Name`,
  `visit_TileLit`, `visit_Let`, `visit_ConstStmt`), letting
  `generic_visit` fire post-override to recurse into child Exprs.
- Calls `visit_ty(...)` explicitly for each TyNode-typed field
  (`Cast.target_ty`, `Name.generics`, `TileLit.dtype`, `Let.ty`,
  `ConstStmt.ty`) — exactly the type-field set that the central walker
  skip-list (`_TYPE_FIELD_NAMES`) excludes by contract.

This matches cycle 22's Target-1 conclusion verbatim and is unaffected
by the C-fix.

---

## Cross-target regression check

| Cycle-22 target | Touched by 4bdc800? | Status |
|---|---|---|
| 1. `ast_walker.py` field-introspection | No (file unchanged) | CLEAN preserved |
| 2. `_op_suffix` collision | No (file unchanged) | CLEAN preserved |
| 3. isize/usize cross-pass consistency | No (none of the 13 sites touched) | CLEAN preserved |
| 4. Deferred rewriter type-soundness | No (grad_pass unchanged) | CLEAN preserved |
| **C-fix delta in struct_mono.py** | Yes | inert deletion; no new surface |

No finding at any severity meets the 75-confidence bar.

---

## Streak verdict

Cycle 23, Audit B (type-design): **CLEAN** under the strict criterion.

Streak counter advance:
- Cycle 21: 1/5
- Cycle 22 (A clean + B clean): 2/5
- Cycle 23 (B clean — pending A this cycle): **3/5 if A also clean**,
  else holds at 2/5 pending A's verdict.
