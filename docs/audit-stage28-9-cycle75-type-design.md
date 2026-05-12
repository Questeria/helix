# Audit Stage 28.9 cycle 75 — Type design

Scope: HEAD `9a51cbf` (cycle-74 fix-sweep addressing cycle-73 type-design
CN-1 double-descent). NARROW conservative type-design audit of the
cycle-74 delta plus immediately adjacent recently-touched layers (cycles
66/68/71 ASTVisitor + iter_fn_decls + flatten_modules intra-mod-alias
discipline). Deferred-known broader-frontend items
(`monomorphize._mangle_ty` catchall, `hash_cons._ast_equal` SHA-256
fallback, `typecheck`/`struct_mono` pre-flatten asymmetry in `check.py`,
`autotune.collect_autotuned_fns`) are NOT re-flagged per cycle-67/70/71/72
deferral discipline.

Audit boundary (focus areas):

1. `helixc/frontend/totality.py` post-migration: type-surface stability
   of the ASTVisitor pattern; arm-completeness of `_SelfCallCollector`
   over Pattern + MatchArm-bearing subtrees.
2. `helixc/frontend/flatten_modules.py` intra-mod aliases (cycles 66/68):
   totality of the `local_lift_indices` list-of-indices walk over the
   item-branch decision table.
3. `helixc/frontend/ast_walker.iter_fn_decls` completeness over Item
   subclasses (FnDecl / ImplBlock / ModBlock / StructDecl / EnumDecl /
   TypeAlias / UseDecl / ConstDecl / AgentDecl / ModuleDecl).
4. Prior C1–C74 findings + deferred registry: no regression observed.

## Verdict

**PASS** — 0 findings at confidence >= 75%.

## Probe-by-probe summary

### Focus 1 — `_SelfCallCollector.visit_Call` post-cycle-74

The fix removes the explicit `self.generic_visit(node)` inside
`visit_Call` so the `ASTVisitor.visit` base auto-descends exactly once
(ast_walker.py:191-196). Hand-trace of the regression input
`fn rec(n) { rec(rec(n - 1)) }` confirms exactly 2 records:

- `visit(fn.body)` → Block has no override → `generic_visit` yields
  `stmts=[]` + `final_expr` (outer Call).
- Outer Call → `visit_Call` records (callee `Name("rec")` matches);
  returns None → auto-descent into `callee` (Name, no-op) + `args` (inner
  Call).
- Inner Call → `visit_Call` records; auto-descent walks `n - 1` (Binary;
  no calls). Total = 2.

Sister `panic_pass._PanicCollector.visit_Call` (line 65-66) uses the
identical post-cycle-74 idiom (record then return None and let the base
auto-descend), so the visitor surface is internally consistent across
the cycle-58 walker-discipline cohort. The new
`test_c73_cn1_totality_no_double_descent` regression in
`tests/test_deprecated.py` (lines 608-638) asserts `len(calls) == 2`
which would have failed at 4+ pre-fix.

Pattern-bearing subtrees (Match → MatchArm → Pattern subclasses
PatLit/PatRange) carry `Expr` children. Tracing `generic_visit` through
the introspection in `_iter_child_nodes` (ast_walker.py:119-153) shows
MatchArm yields `guard`/`body` Exprs and Pattern subclasses yield their
Expr fields; `_SelfCallCollector` would notice a Call(rec, …) appearing
in any of those positions. Whether such call-bearing patterns are
semantically reachable is a separate, pre-existing question (PatLit
values are bound by typecheck to literals); it does not surface a NEW
type-design defect in the cycle-74 delta.

### Focus 2 — `flatten_modules._flatten_one` intra-mod alias walk

The cycle-68 fix replaced the cycle-66 `range(direct_lifts_start,
len(new_items))` slice with a per-direct-lift `local_lift_indices: list`
appended at each non-recursive branch entry (line 139). The decision
table is exhaustive over the Item subclasses that appear in
`ModBlock.items` post-parse:

| sub-item kind     | local_lift_indices? | rewritten in loop?                | rewritten post-loop? |
|-------------------|--------------------|-----------------------------------|----------------------|
| ModBlock          | NO (own recursion) | n/a                               | n/a                  |
| FnDecl            | YES                | no                                | YES (line 225-226)   |
| StructDecl        | YES                | no                                | no-op (skip)         |
| EnumDecl          | YES                | no                                | no-op (skip)         |
| ConstDecl         | YES                | no                                | YES (line 227-228)   |
| TypeAlias         | YES                | no                                | no-op (skip)         |
| UseDecl           | YES                | no                                | no-op (skip)         |
| ImplBlock         | YES                | YES (line 200-210, methods)       | no-op (skip)         |
| else (AgentDecl…) | YES                | no                                | no-op (skip)         |

Each direct-lift branch records its slot index BEFORE the append, so
no nested-mod lift can be picked up by the post-loop walk. The
`intra_mod_aliases` keyset is built upfront from the same five Item
kinds whose lifted-name slot is `base + "__" + sub.name`, which matches
the rewrite target. ImplBlock method bodies are rewritten in-loop with
the same alias map, and the ImplBlock `target` field is remapped (line
200) so a sibling `impl Foo { … }` and `struct Foo { … }` in the same
mod stay coupled post-flatten. AgentDecl / ModuleDecl correctly fall
through to the inert `else: append(sub)` branch.

The `range`-vs-list-of-indices distinction is materially type-design
relevant: the pre-cycle-68 code's loop-index range implicitly carried a
nested-frame scope it did not actually filter for. The cycle-68 fix
makes the scope explicit (a real `list[int]` populated only at this
frame). The audit confirms the list is monotonic, populated only at
direct lifts, and consumed only by FnDecl/ConstDecl branches that hold
Expr children sensitive to alias rewriting.

### Focus 3 — `iter_fn_decls` over Item subclasses

The walker matches on three concrete Items: FnDecl (yield), ImplBlock
(yield each method that is FnDecl), ModBlock (recurse). The full Item
subclass set is:

- FnDecl ............... yielded.
- StructDecl / EnumDecl / TypeAlias / UseDecl / ConstDecl ........
  intentionally skipped — none of them carry FnDecls.
- AgentDecl ............ skipped; carries `AgentMethod` (signatures
  only, not FnDecl). Correct per cycle-58 deprecated-pass discipline.
- ModuleDecl ........... skipped; `path` only, no items.
- ModBlock ............. recurse.
- ImplBlock ............ yield methods.

Nested ImplBlock-inside-ModBlock is reached via the ModBlock recursion
which redispatches on the inner items including ImplBlock. Method
listed inside `ImplBlock.methods` that is somehow not a FnDecl is
defensively skipped by the inner `isinstance(m, A.FnDecl)` filter — a
nice belt-and-braces against future trait-default-method shapes.

### Focus 4 — Prior C1–C74 + deferred registry sanity

Cycle 73 CN-1 (this cycle's predecessor) is closed by the
`visit_Call` change. Cycle 73 CN-2 (commit-msg vs. code drift for
`_mangle_ty`) is closed by acknowledging in the cycle-74 commit
message that the loud-fail did ship in `a22cba0`. Cycle-71 CN-3
(StructLit.name remap inside flatten_modules) remains in place at
`flatten_modules.py:302-304`. Cycle-66/68 CN-1/CN-2 fixes remain
intact. No regression of prior accepted fixes was observed in the
audited surface.

## Areas audited with no >=75%-conf findings

- `_arg_strictly_decreases` decision logic (totality.py:127-156): the
  conservative "return False if we can't verify" stance correctly
  flags `rec(rec(n-1))` as non-decreasing for the outer call (outer
  arg is a Call, not a Binary), matching the documented intent.
- `_iter_child_nodes` skip-list (`_TYPE_FIELD_NAMES`,
  `_NON_NODE_FIELD_NAMES`): no field-name drift detected between the
  AST node definitions and the walker's exclusion sets.
- ImplBlock branch in `_flatten_one` (lines 184-210): single-rewrite
  surface (in-loop only); no double-rewrite path observable.
- `_PanicCollector` (panic_pass.py:54-66): sister visitor pattern
  remains consistent with the cycle-74 totality fix.

## No-edit attestation

This audit performed only Read, Grep, Glob, and Bash. No source file
was modified. The only Write was this report at
`docs/audit-stage28-9-cycle75-type-design.md`.
