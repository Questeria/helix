# Audit Stage 28.9 cycle 56 — Silent failures

**Scope.** Read-only HEAD `5d58d3d`. Prior C1–C54 not re-flagged.
**Criterion.** 0 findings at conf >=75%.

## Result: 0 findings at >=75% — PASS

The cycle-55 delta extends `_fn_table_sig` from
`{name}:{body_hash}` to `{name}/{arity}/{attrs}/{body_hash}` and
broadens both autodiff cache-layer except-clauses to include
`NotImplementedError`. The delta is correct and surface-tight:

- Catch tuple `(TypeError, ValueError, AttributeError,
  NotImplementedError)` is symmetric across `_fn_table_sig` (line
  146) and `differentiate` (line 184). NIE is the documented
  loud-fail exception raised by `structural_hash._hash_into`
  fallthrough at `ast_hash.py:496`; both call sites now catch it
  and degrade to `<unhashable:{id}>` sentinel / `key=None` bypass.
  The bypass paths each emit an `_ad_warn` so the user sees the
  perf regression instead of silently losing the cache.
- The arity/attrs additions plug the cache-collision class that
  C54-AD1 and C54-AD2 documented: de-Bruijn body hash conflates
  `fn g(x,y) = x` with `fn g(x) = x`, and bodies identical modulo
  `@pure` would otherwise reuse a wrong derivative.
- Regression tests `test_c54_ad1/ad2/ad3` exercise all three
  fixes; full `test_autodiff.py` passes (40/40).

Inspected edge cases that did NOT produce findings:

- Empty `fn.attrs` (`""` join, `f"{name}/{arity}//{body_hash}"`):
  distinct from any non-empty attrs string. No collision.
- Attrs with embedded `:` or `,` (e.g. `autotune:K=1,2`,
  `deprecated:msg`): the `,`-joined attrs section is delimited
  from arity by `/`, which Helix identifiers cannot contain.
  Adversarial attr names like `"foo/1//body"` would require a
  parser-accepted ident with `/`, which the lexer rejects.
- FFI/extern fns in fn_table: `body` is an empty placeholder
  Block. `structural_hash` produces a stable hash for empty
  Block; `_inline_user_calls` doesn't gate on `is_extern` so cache
  conflation across extern↔non-extern with same body is benign
  (both treated as opaque-or-inline by the same logic).
- `RecursionError` / `MemoryError` from deep ASTs: not in catch
  tuple, propagates loud — consistent with the project's
  loud-fail discipline.
- AST nodes that hash via `structural_hash` but bypass
  `_fn_table_sig` (the `expr` argument to `differentiate`): the
  outer try at line 182-197 catches the same NIE class and emits
  a warning; cache bypasses, no silent zero gradient.
- `id(fn.body)` sentinel for unhashable bodies: not stable across
  runs but the test asserts only the `<unhashable:` prefix; intra-
  run cache hits for the same unhashable body still work (id is
  stable within a process for a live object).

Secondary scan across `helixc/frontend/`, `helixc/ir/passes/`,
`helixc/backend/` found no NEW silent-failure pattern at
confidence >=75. The five remaining `except Exception:` sites
(`diagnostics.py:76` isatty fallback, `const_fold.py:323,405,437,
514`) are each prefixed by a `FoldError` re-raise per cycle-21
C20-R1 and bounded to arithmetic-only logic, so the trap-17001/
17002 contract still surfaces.

## Notes (<75)

- `autodiff_reverse.py:149-152`: `_propagate` on an `A.Block`
  with stmts but `final_expr=None` silently returns without
  `_ad_warn`. Forward-mode warns via `_inline_lets`
  (`autodiff.py:553-559`) before propagate runs, so the path is
  currently unreachable in practice (the inliner replaces such
  blocks with `FloatLit(0.0)`+warning). Latent asymmetry; would
  matter only if a caller invokes `_propagate` on a non-inlined
  Block. Confidence ~55.
- `ast_hash.py:569`: `_hash_pattern` catch-all `_emit(h,
  "PatUnknown", type(pat).__name__)` is soft (no recurse, no
  raise) while `_hash_into:496` is loud-fail. Currently
  unreachable — every `A.Pat*` subclass has an explicit arm — but
  a future Pattern subclass added without an arm would silently
  collide with siblings of the same class. Asymmetric to the
  cycle-34 / cycle-35 discipline. Confidence ~70.
- `_fn_table_sig` does not include `fn.return_ty`,
  `fn.is_extern`, `fn.is_pub`, `fn.generics`, `fn.where_clauses`.
  None of those are read by `_inline_user_calls` or
  `_is_inferably_pure`, so they cannot perturb the emitted
  derivative AST today. Latent risk if a future inliner gates on
  `is_extern` (e.g., refusing to inline FFI shims) — at that
  point the cache key would need extension. Confidence ~50.
- `_fn_table_sig` `attrs_part` uses `,` as inner separator and
  `/` as outer. Adversarial attr value containing `/` followed by
  another attr could in principle craft a string-collision across
  distinct attr lists, but the parser only emits attrs of the
  forms `@ident`, `effect:ident`, `autotune:K=v1,v2`,
  `deprecated:msg`, `since:vstr`. None can contain `/` because
  the lexer rejects `/` in identifiers and the string-attr
  capture (`_parse_string_attr_arg`) is unverified here for
  escape handling but the joined output is per-fn-name-prefixed,
  so cross-fn collisions are bounded. Confidence ~40.
