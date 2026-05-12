# Audit Stage 28.9 cycle 83 — Silent failures

Scope: HEAD `42f4e11`

## Areas audited (rotated, fresh)

1. `helixc/frontend/grad_pass.py` — gradient-pass walker, grad/grad_rev/grad_rev_all rewrite + let-alias resolution
2. Reflection-pass code paths — `helixc/ir/lower_ast.py` Quote/Splice/Modify lowering + `helixc/ir/passes/effect_check.py` MODIFY/SPLICE/QUOTE/REFLECT_HASH effect mapping
3. `helixc/ir/passes/fdce.py` — function-level DCE invariants (CALL + MODIFY.verifier_fn + QUOTE.ast_pretty reachability)

## Deferred-known (NOT re-flagged, per scope)

- `monomorphize._mangle_ty` silent catchall
- `hash_cons._ast_equal` silent catchall
- `typecheck`/`struct_mono` pre-flatten in `check.py`
- `autotune.collect_autotuned_fns` missing `iter_fn_decls`

## Method

- Read each target file end-to-end
- Cross-referenced AST/Stmt/Expr subtype coverage against `ast_nodes.py`
- Checked all `except`/`pass`/fallback branches for missing diagnostics
- Verified call-graph roots in fdce against indirect-call mechanisms (`OpKind` enum has no `CALL_INDIRECT`; all calls are direct CALL or MODIFY.verifier_fn or QUOTE.ast_pretty — all three rooted)
- Verified `_rewrite_in_block` and `_resolve_let_aliases` against the closed set of `Stmt` subclasses (Let / ExprStmt / ConstStmt)
- Verified `_rewrite_in_expr` and `_resolve_in_expr` against every `Expr` subtype that contains nested `Expr` (Call/Binary/Unary/Cast/Block/UnsafeBlock/If/Match/Loop/Index/Field/While/For/Return/Break/Assign/Range/TupleLit/ArrayLit/StructLit/Quote/Splice/Modify) — all 22 subtypes covered in both walkers since cycle-2 C2-4

## Sub-threshold observations (kept here for cycle-84+ rotation, NOT counted as findings)

- `_resolve_let_aliases` (grad_pass.py:143) handles `Let` + `ExprStmt` but not `ConstStmt`. `_rewrite_in_block` is symmetric (handles all three). Real-world impact gated by language semantics (a Block-scoped `const` initializer cannot legitimately reference a runtime `let` alias), so the asymmetry is mostly inert. Confidence ~55%, below 75% threshold.
- `lower_ast._pretty` returns `<Field>` for `A.Field` nodes (line 2291 generic fallback), causing fdce's QUOTE.ast_pretty regex scan to miss field-selector names. This is actually correct (field access in a quote is a selector, not a free function reference; the receiver name still appears via the `A.Name`/`obj` recursion arm) — not a silent failure.
- `lower_ast._lower_expr` for `A.Quote` (line 2155) has a broad `except Exception:` swallowing `structural_hash` errors and falling back to `_pretty` keying. This is the documented fallback pattern that matches the deferred-known catchall category; not re-flagged.
- `grad_pass._generate_grad_fn` cache write (line 618-620) uses `except (AttributeError, TypeError): pass` to degrade to no-cache on frozen-dataclass FnDecl. Narrow exception types; degradation is correctness-preserving (just slower). Not a silent failure.
- `lower_ast._verifier_abi_matches` (line 2230) silently falls back on wrong-arity / non-i32-params, but explicitly raises on right-shape-wrong-return-type. Documented and intentional; not a silent failure.

## Findings at confidence >= 75%

None.

## Verdict

PASS — 0 findings at confidence >= 75%.
