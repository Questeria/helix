# Audit Stage 28.9 cycle 59 — Type design

**Scope.** Read-only at HEAD `722baf8` (cycle-58 fix-sweep: totality
walker recursion; flatten_impls/flatten_modules UnsafeBlock+TileLit
arms with NotImplementedError catchalls; pytree TyGeneric
resolution via `_resolve_struct_name`; match_lower loud-fail
catchall; deprecated_pass `_walk_items_for_fns`). Adversarial 3rd
pass, rotating to areas not covered by cycle 56/57:

- `helixc/frontend/struct_mono.py` — TyGeneric mono correctness,
  `mangle_struct` injectivity.
- `helixc/ir/passes/effect_check.py` — effect propagation rules.
- `helixc/frontend/presburger.py` — index-analysis soundness (no
  `helixc/ir/passes/presburger.py` exists; the only presburger
  module is the frontend one).
- `helixc/backend/x86_64.py` — register / stack-slot allocator and
  lifetime invariants (the backend is a one-slot-per-SSA-value
  always-spill allocator; no formal lifetime analysis exists).
- Cycle-58 fixes' type surface:
  - `_is_struct_ref(TyGeneric(...))` totality over all TyGeneric
    shapes.
  - `_walk_items_for_fns` (deprecated_pass) double-visit hazard if
    ModBlock contains ImplBlock.

Prior C1–C58 dispositions not re-flagged.

**Criterion.** Pass = ZERO findings at confidence >=75%.

## Result: FAIL (1 finding >=75%)

### F59-1 — match_lower `_rewrite_expr` now crashes on bare `return;` (HIGH, conf 95)

Cycle-58 audit-R C57-4 (`match_lower.py:281-306`) replaced the
prior trailing `return expr` implicit-passthrough with a loud
`NotImplementedError` catchall and added an explicit leaf-pass
list `(IntLit, FloatLit, StrLit, CharLit, BoolLit, Name, Path,
Continue)`. The leaf list intentionally excludes `Return` because
the existing Return arm at line 227 handles it:

```python
if isinstance(expr, A.Return) and expr.value is not None:
    expr.value = _rewrite_expr(expr.value)
    return expr
```

The `and expr.value is not None` guard means a `Return` node with
`value=None` (i.e. a bare `return;` statement, which the parser
explicitly supports — `parser.py:1137-1142` constructs
`ast.Return(span=..., value=None)` when the next token is `;` or
`}`) FALLS THROUGH this arm. It then misses every subsequent arm
and hits the new `NotImplementedError` raise at line 301.

Reproduction (run from `C:/Projects/Kovostov-Native`):

```python
from helixc.frontend.parser import parse
from helixc.frontend.match_lower import lower_matches
prog = parse("fn foo() { return; }")
lower_matches(prog)
# NotImplementedError: match_lower._rewrite_expr: unhandled
# expression kind Return at Span(line=1, col=12). ...
```

Same source compiled cleanly at HEAD `2f3dcbc` (cycle-56) because
the pre-cycle-58 trailing `return expr` passthrough silently
accepted `Return(value=None)`.

Why no test caught it:
- `helixc/tests/test_trace.py:test_c2_2_early_return_void`
  intentionally exercises early-return-in-traced-fn but with
  `return 0;` (value-bearing) — see test source line 297.
- A grep of `helixc/bootstrap/*.hx` finds zero bare `return;`
  occurrences, so the bootstrap-pipeline regression tests don't
  trip it either.
- The match_lower unit test directory has no test for bare-return
  through `lower_matches`.

Same defect class as cycle-23 C22-C (UnsafeBlock/Range/Modify gap
in the same walker) — except this regression was INTRODUCED by
the cycle-58 hardening rather than discovered as latent.

Fix sketch (do not apply; finding-only): add an explicit
Return arm OR widen the gated Return arm to handle the None case:

```python
if isinstance(expr, A.Return):
    if expr.value is not None:
        expr.value = _rewrite_expr(expr.value)
    return expr
```

Mirror of the cycle-23 Break handling at lines 261-264, which
correctly puts `return expr` outside the inner None-guard.

Files: `C:/Projects/Kovostov-Native/helixc/frontend/match_lower.py`.

## Sub-75% observations (not flagged)

- **`presburger._reduce_via_eqs` substitution-direction comment
  vs code (conf ~50).** Line 281-283 contains a dead-disjunctive
  `if False else (...)` ternary. The `if False` branch is
  literally unreachable; the else branch correctly negates `rest`
  when `c == 1` (since `c*v + rest = 0` ⇒ `v = -rest/1 = -rest`)
  and passes `rest` through when `c == -1` (since `v = -rest/-1 =
  rest`). The arithmetic is correct; the dead branch is dev-time
  scratch. Cleanup is cosmetic, not a soundness issue.

- **pytree `_is_struct_ref(TyGeneric(...))` mangle-key drift (conf
  ~60).** The cycle-57 C57-3 fix uses `mangle_struct(ty.base,
  list(ty.args))` to derive the dict-lookup key. That mangle
  scheme is the same one `struct_mono.instantiate` uses for the
  decl-name when it appends the mono'd struct to `prog.items`. So
  the lookup succeeds iff struct_mono has already run on the
  program. The two callers in this tree (`reverse_diff_pass` and
  `autodiff_pass`) both run after `monomorphize_structs` in
  `check.py:422` / `x86_64.py:3100+`, so the ordering invariant
  holds at HEAD. The hazard is latent for any future caller
  invoking pytree on a pre-mono program: `_is_struct_ref` would
  return False for `Pt<i32>` (mangle "Pt__i32" not present),
  silently routing to trap 26002 with the misleading
  "non-differentiable type" message that C57-3 was supposed to
  fix. Below 75 because: (a) current callers ordered correctly,
  (b) no surface-language path reaches pytree pre-mono.

- **`_walk_items_for_fns` (deprecated_pass) ModBlock+ImplBlock
  double-visit (conf ~25).** The audit prompt explicitly asks
  whether `ModBlock { ImplBlock { ... } }` triggers double-visit.
  It does NOT: `_walk_items_for_fns` is called once on
  `prog.items`; each Item is dispatched exactly once by its
  outer-type isinstance ladder. Nested ModBlock recurses; nested
  ImplBlock iterates its methods. No item is visited twice
  regardless of nesting depth. The same structural claim holds
  for `totality.collect_items`. (The cycle-57 doc-comment mentions
  `scan_items` as the deprecated_pass helper, but the actual
  function is named `_walk_items_for_fns` — naming inconsistency,
  not a soundness issue.)

- **`x86_64._is_u64_type` coverage gap (conf ~62, restated from
  cycle 57 §Notes).** Re-verified at HEAD `722baf8`; predicate is
  still consulted only at FFI call-sites (`x86_64.py:1752, 1763`),
  not at CONST_INT/ADD/SUB/MUL/BITCAST/CMP/RETURN/BR. Below 75
  per the same triage as cycle 57 (Phase-1 deferral).

- **Expr-subclass coverage of cycle-58 walker catchalls
  (re-verified, no finding).** Both flatten_impls `_rewrite_expr`
  and flatten_modules `_rewrite_expr` explicitly handle 24 Expr
  subclasses with rewrites plus 8 leaves via the catchall
  isinstance tuple = 32 total, matching the 32 `Expr` subclasses
  declared in `ast_nodes.py`. The flatten_modules leaf catchall
  omits `Name` (it has a dedicated earlier arm at line 233) —
  intentional, not a gap. match_lower's leaf list and explicit
  arms also sum to 32, except for the F59-1 bare-Return gap above.

### Stability

The cycle-58 fix-sweep was strictly invariant-strengthening in
spirit (loud-fail catchalls + container-aware item walks). The
one regression is at a single arm-guard expression that the
sweep authors did not flag in their own walker-totality review;
cycle 57's type-design audit (§Expr-walker totality, lines 44-53)
explicitly claimed the C57-4 Continue arm "closes the last
silent fall-through" — but the Return arm's value-gate predates
cycle 58 and was not re-examined when the loud catchall landed.

Files inspected:
- `C:/Projects/Kovostov-Native/helixc/frontend/match_lower.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/flatten_impls.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/flatten_modules.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/totality.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/deprecated_pass.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/pytree.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/struct_mono.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/monomorphize.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/ast_nodes.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/parser.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/presburger.py`
- `C:/Projects/Kovostov-Native/helixc/ir/passes/effect_check.py`
- `C:/Projects/Kovostov-Native/helixc/backend/x86_64.py`
- `C:/Projects/Kovostov-Native/helixc/check.py` (cross-check)
