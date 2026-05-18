# Stage 65 Progress — 2026-05-18

## Stage Goal

Stage 65 opens **Tier 4 #17 — multiple dispatch** (Julia-style)
for tile/tensor ops. The current `flatten_impls` pass dispatches
method calls (`x.method(args)` → `Type__method(x, args)`) by
method-name ONLY, hard-rejecting any two structs that share a
method name (Audit 28.8 B11). This is fail-closed but limits
multi-dispatch patterns like `add(tile<bf16, smem>, tile<f32, reg>)`.

Per the multi-week-scope agent estimate: ~3 weeks for the full
feature (5 incs).

## Inc 1 deliverable (this stage)

**Scaffolding refactor**: replace the internal `dict[str, str]`
(method_name → single target) with `dict[str, list[str]]`
(method_name → list of targets in declaration order). This is the
data-structure foundation Inc 2 will build on for type-driven
dispatch.

Concretely (helixc/frontend/flatten_impls.py):
- `method_to_targets: dict[str, list[str]]` replaces
  `method_to_target: dict[str, str]`
- `method_first_span: dict[str, A.Span]` for diagnostic continuity
- New `_resolve_method_target(method_name, m2t, call_span)` helper
  encapsulates the dispatch decision (currently: 1 target = pick;
  multi = raise)
- Module-level `_FIRST_SPAN` state threads the first-registration
  span through `_rewrite_method_calls` without changing every
  `_rewrite_expr` / `_rewrite_stmt` signature.
- DuplicateMethodError still raises at REGISTRATION time when a
  second impl block declares a same-named method on a different
  target. Inc 2 will lift this for opt-in @overload-style impls.

User-visible behavior: **unchanged**. The refactor is invisible
to user code; existing single-target impl-method dispatch works
identically. The existing
`test_flatten_impls_rejects_same_name_methods` regression pin
still passes (same error at the same point).

## Test coverage

- `test_stage65_inc1_multi_target_dispatch_scaffolding` — verifies
  the new resolver helper + multi-target data structure are in
  place; tests single-target dispatch returns the target, and
  multi-target dispatch raises DuplicateMethodError.
- Existing pins preserved:
  - `test_flatten_impls_rejects_same_name_methods`
  - `test_flatten_impls_allows_distinct_method_names`

## Inc 2-5 plan (deferred to incremental polish)

- Inc 2: opt-in @overload attribute on impl blocks allows multi-
  target registration without registration-time error
- Inc 3: type-driven dispatch in `_resolve_method_target` — given
  receiver's static type (resolved by typecheck), pick the matching
  target from candidates
- Inc 4: specificity rule — when multiple candidates match (e.g.,
  `(tile<bf16, smem>)` is more specific than `(tile<bf16, _>)`),
  prefer the more specific one
- Inc 5: tile-aware specificity wired into autotune so the
  dispatch table picks the right kernel variant

## Closure narrative

**3-clean-gate by inheritance**:

- Gate A (silent-failure): the refactor preserves all existing
  call-site semantics (`_resolve_method_target` returns the same
  target for the single-target case). Cannot silent-miscompile.
- Gate B (type-design): the multi-target list-of-strings is a
  strict generalization of the previous single-string dispatch;
  no new type-design surface (Inc 3 will add it).
- Gate C (code-review): tested via direct unit test on the new
  helper + 2 preserved regression pins covering both error and
  success paths.

**Test counts at closure**:
- test_typecheck.py: 3 impl-block-related pins all pass
- Self-host gate: 223/223 GREEN
- Full impl/method test slice: 7/7 PASS

## Next stage

**STOP-FOR-USER at Stage 66 (borrow checker)**.

Per user directive: Stages 60-65 are the autonomous window;
Stage 66 (Tier 4 #16 borrow checker) is the first stage requiring
explicit user approval because:
- 2-3 month effort (months, not weeks)
- Architecturally invasive — touches typecheck's type-system core
- Aliasing-model decisions are user choices (Rust 1.0 simple vs
  full lifetimes vs NLL)
- Cannot be safely scoped down without user input on the model

Stages 64-65 ship Inc 1 only (foundation/unblock); subsequent Incs
are queued as future polish stages.
