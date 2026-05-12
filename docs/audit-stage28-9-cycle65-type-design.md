# Audit Stage 28.9 cycle 65 — Type design

**Scope.** Read-only at HEAD `e7bd9c6` (cycle-64 fix-sweep:
pipeline-contract — `flatten_modules` added to `helixc/check.py`
between `flatten_impls` and the analysis passes; `iter_fn_decls`
docstring rewritten; `find_deprecated_decls` docstring updated).
Adversarial pass rotated to areas not covered by cycles 56–64:

- `helixc/frontend/struct_mono.py` — `monomorphize_structs`
  correctness, `mangle_struct` injectivity for `TyGeneric` with
  multiple type args.
- `helixc/frontend/typecheck.py` — where-clause discharge,
  trait/method dispatch (flattened to free fns by `flatten_impls`,
  so no in-typechecker method-resolution path).
- `helixc/ir/lower_ast.py`, `helixc/ir/tir.py` (no `builder.py`
  exists in this tree — audit-prompt reference is stale) — Op
  result-type vs operand-type invariants.
- Cycle 60–64 fixes' type surface: `iter_fn_decls` totality over
  `Item` subclasses; `ModBlock`-containing-`ImplBlock`-containing-
  `ModBlock` double-yield hazard.

Phase-0 has no lifetime AST (verified — no `Lifetime` node in
`ast_nodes.py`), so the audit-prompt's "lifetimes" sub-question is
vacuous.

Prior C1–C64 dispositions not re-flagged.

**Criterion.** Pass = ZERO findings at confidence >=75%.

## Result: FAIL (1 finding >=75%)

### F65-1 — `helixc/check.py` flattens impls before modules; mod-nested `impl` blocks survive both passes as stale top-level `ImplBlock` (HIGH, conf 85)

At HEAD `e7bd9c6`, `helixc/check.py` runs `flatten_impls(prog)`
(lines 438–449) BEFORE `flatten_modules(prog)` (lines 466–474).
The cycle-64 fix added `flatten_modules` AFTER the existing
`flatten_impls` call, not before it. The codegen driver
`helixc/backend/x86_64.py` runs them in the opposite order
(lines 3104, 3107: `flatten_modules` → `flatten_impls`), so the
two production drivers DIVERGE on prefix-pass ordering.

The bug: `flatten_impls` iterates only `prog.items` top-level
(`flatten_impls.py:72`, `for item in prog.items: if
isinstance(item, A.ImplBlock):` — no `ModBlock` recursion). So an
`impl Foo { ... }` block nested inside a `mod m { ... }` is
invisible to `flatten_impls` at this point. Then `flatten_modules`
runs; in `_flatten_one` (`flatten_modules.py:86–138`) the
explicit-dispatch ladder enumerates `ModBlock` / `FnDecl` /
`StructDecl` / `EnumDecl` / `ConstDecl` / `TypeAlias` / `UseDecl`
but NOT `ImplBlock`, so `ImplBlock` falls through to
`else: new_items.append(sub)` (line 136–137) — appended verbatim
at top level with `target` field unmangled and no method-lifting.

End state after both passes:
- a stale `ImplBlock` is now top-level (was inside `m`);
- its methods were never lifted to `Foo__method` top-level FnDecls;
- method-call sites like `x.bar()` were never rewritten to
  `Foo__bar(x)` because `_rewrite_method_calls` ran during
  `flatten_impls` (before the `ImplBlock` was visible).

Reproduction (from `C:/Projects/Kovostov-Native`):

```python
from helixc.frontend.parser import parse
from helixc.frontend.flatten_impls import flatten_impls
from helixc.frontend.flatten_modules import flatten_modules
src = """
struct Pt { x: i32, y: i32 }
mod m { impl Pt { fn area(self: Pt) -> i32 { 42 } } }
fn main() -> i32 { let p = Pt { x: 1, y: 2 }; p.area() }
"""
prog = parse(src)
flatten_impls(prog)
flatten_modules(prog)
for it in prog.items:
    print(type(it).__name__, getattr(it, "name", "?"),
          getattr(it, "target", None))
# StructDecl Pt None
# ImplBlock ? Pt          <-- stale; methods not lifted
# FnDecl main None
```

Compare against `backend/x86_64.py` ordering on the same input:
`flatten_modules` runs first and the `ImplBlock` (a non-handled
Item kind in `_flatten_one`'s ladder) is STILL appended verbatim
to top level, but THEN `flatten_impls` runs, finds it at top
level, lifts `Pt::area` to `Pt__area`, and rewrites `p.area()`
→ `Pt__area(p)`. Correct.

Why no test caught it (yet):
- `helixc/check.py` runs `typecheck` BEFORE either flatten pass
  (line 403), so method-call-bearing programs (even
  top-level-impl, no mod) emit "struct 'Pt' has no field 'area'"
  and `check` returns 1. The check tool was empirically already
  rejecting ALL method-call programs — so the mod-nested-impl
  regression is masked by the broader pre-flatten-typecheck
  issue. (That pre-flatten-typecheck issue is itself latent — but
  not new in cycle 64, so out-of-scope for this finding.)
- No `helixc/tests/test_*.py` covers
  `mod m { impl T { ... } }` end-to-end via the surface tool.

Why it surfaces NOW: the cycle-64 commit message claims it
landed `flatten_modules` to share "the same prefix-pass invariant"
across the two drivers — but actually inserted it AFTER
`flatten_impls` in check.py while the backend has the opposite
order. The "shared invariant" claim is false.

Fix sketch (do not apply; finding-only): swap the two
`try: flatten_*(prog)` blocks in `check.py:438–474` so
`flatten_modules` runs first, matching `backend/x86_64.py:3104,
3107`. Mirror the same swap in any direct-API caller that mixes
the two passes (none identified in this audit).

Files: `C:/Projects/Kovostov-Native/helixc/check.py`,
`C:/Projects/Kovostov-Native/helixc/frontend/flatten_impls.py`,
`C:/Projects/Kovostov-Native/helixc/frontend/flatten_modules.py`,
`C:/Projects/Kovostov-Native/helixc/backend/x86_64.py`.

## Sub-75% observations (not flagged)

- **`mangle_struct` non-injective on underscore-bearing names
  (conf ~65).** `mangle_struct("Pt", [TyName("i32_f64")])` and
  `mangle_struct("Pt", [TyName("i32"), TyName("f64")])` both
  produce `"Pt__i32_f64"` because args are joined with `_` and
  identifier-level underscores are syntactically legal. However:
  (a) `struct_mono._ty_key` correctly distinguishes the two
  (tuple-keyed), so both make it into `rewrite_map` as separate
  entries; (b) the silent loss happens via the `existing` set
  check at `struct_mono.py:376`, which would discard the second
  instantiation. (c) Reachability requires either a user-source
  struct or `TyName` with an underscore matching a multi-arg
  decomposition — no current `.hx` source in the tree triggers it.
  Latent hazard; below 75 because no reachable exploit on current
  inputs and mono'd structs themselves have `generics=[]` so
  can't recurse the pattern.

- **`iter_fn_decls` totality over Item subclasses (conf ~25, not
  a finding).** The 10 `Item` subclasses (FnDecl, StructDecl,
  EnumDecl, TypeAlias, UseDecl, ConstDecl, AgentDecl, ModuleDecl,
  ModBlock, ImplBlock) are handled correctly: FnDecl yielded,
  ImplBlock yields its `methods` (which are typed `list[FnDecl]`,
  not `list[Item]`, so no recursive Item subclass inside),
  ModBlock recurses `_walk(it.items)`. Non-fn Item kinds (Const,
  Use, etc.) are correctly skipped — the helper's contract is
  "yield FnDecls", not "yield all Items". The
  ModBlock→ImplBlock→ModBlock double-yield scenario the audit
  prompt asked about is structurally impossible: `ImplBlock` only
  holds `FnDecl` methods (per `ast_nodes.py:555`), not Items, so
  no nested `ModBlock` inside an `ImplBlock`.

- **`_mangle_ty` TyGeneric same `_`-separator non-injectivity
  (conf ~55).** `monomorphize._mangle_ty` for `TyGeneric` returns
  `t.base + "_" + "_".join(_mangle_ty(a) for a in t.args)` (line
  149). Symmetric to the struct_mono concern above; reachability
  is the same narrow case (underscore in a base name or arg's
  type name). The fn-mono path goes through a separate dedup key
  (`mangle(name, ty_args)` at `monomorphize.py:53–62`), and an
  upstream collision in the mangled output is similarly silent.
  Below 75 for the same reasons.

- **`_mangle_ty` catchall returns `"X"` (conf ~30, not a
  finding).** Line 150 returns the literal string `"X"` for any
  TyNode not in the 9 enumerated arms. The current enumerated
  set is exhaustive over Phase-0 TyNode subclasses (TyName,
  TyTuple, TyArray, TyRef, TyPtr, TyFn, TyTensor, TyTile,
  TyGeneric — verified against `ast_nodes.py:28-93`), so the
  catchall is reachable only if a future TyNode subclass is
  added without extending the switch. No new TyNode subclasses
  in this cycle's HEAD diff. The loud-fail pattern (`raise
  NotImplementedError`) from cycle-58's walker-drift sweep was
  NOT applied here; this is a latent drift hazard but not a
  current defect.

### Stability

The cycle-64 fix-sweep landed the pipeline-contract claim —
`flatten_impls` AND `flatten_modules` both run in `check.py`
before the analysis passes — but did so in the wrong order. The
docstring update to `iter_fn_decls` (now correctly states
"post-flatten" for both drivers) holds for the backend driver
but not for check.py until the order is swapped. The finding is
the symmetric inverse of cycle-63 CN-A: cycle-63 closed the gap
of `flatten_modules` being absent; cycle-65 should close the
gap of it being present-but-after.

Files inspected:
- `C:/Projects/Kovostov-Native/helixc/check.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/flatten_impls.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/flatten_modules.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/struct_mono.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/monomorphize.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/typecheck.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/ast_walker.py`
- `C:/Projects/Kovostov-Native/helixc/frontend/ast_nodes.py`
- `C:/Projects/Kovostov-Native/helixc/ir/lower_ast.py`
- `C:/Projects/Kovostov-Native/helixc/ir/tir.py`
- `C:/Projects/Kovostov-Native/helixc/backend/x86_64.py` (cross-check)
