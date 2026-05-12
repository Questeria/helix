# Audit Stage 28.9 cycle 65 — Silent failures

**Scope.** Read-only HEAD `e7bd9c6`. Adversarial 5th pass rotated to:
cycle-63/64 fix (`flatten_modules` wiring into `check.py`),
`ast_walker.iter_fn_decls` edge cases, `helixc/ir/passes` (cse / const_fold /
dce / fdce / effect_check), `helixc/backend/x86_64.py` + `helixc/backend/ptx.py`
op-dispatch fallthroughs. Prior C1–C64 not re-flagged.

**Criterion.** 0 findings at conf >=75%.

## Result: 1 finding at >=75% — FAIL

## Finding C65-1 — `helixc/check.py` runs `flatten_impls` BEFORE `flatten_modules`; `mod`-nested `impl` blocks survive both passes as stale top-level `ImplBlock` with unlifted methods and un-rewritten call sites

**Severity:** HIGH. **Confidence:** 88.
**Location:** `helixc/check.py` lines 438–470 (flatten ordering);
`helixc/frontend/flatten_impls.py` lines 57–102 (top-level-only iteration);
`helixc/frontend/flatten_modules.py` lines 86–138 (`_flatten_one`
explicit-dispatch ladder lacks an `ImplBlock` arm).
Co-flagged independently as F65-1 in the cycle-65 type-design audit
(conf 85) via the structural-correctness lens; mirrored here as a
silent-failure because the user-visible symptom is asymmetric driver
behavior with no diagnostic.

**Issue.** The cycle-64 commit (e7bd9c6, "Stage 28.9 cycle-64 fix-sweep:
cycle-63 findings (pipeline contract)") added `flatten_modules(prog)`
to `check.py` AFTER the pre-existing `flatten_impls(prog)` call at
line 442. Order at HEAD:

1. line 442: `flatten_impls(prog)` — iterates only `prog.items`
   top-level (`flatten_impls.py:72`), filters for `isinstance(item,
   A.ImplBlock)`. Has NO recursion into `ModBlock.items`, so an `impl X
   { ... }` nested inside `mod m { ... }` is invisible to this pass.
2. line 466: `flatten_modules(prog)` — lifts mod items to top level.
   The `_flatten_one` dispatch ladder
   (`flatten_modules.py:86–138`) explicitly handles `ModBlock` /
   `FnDecl` / `StructDecl` / `EnumDecl` / `ConstDecl` / `TypeAlias` /
   `UseDecl` / `UseDecl`, but has NO `ImplBlock` arm. So `ImplBlock`
   falls through to `else: new_items.append(sub)` (line 136–137) and
   is appended verbatim at top level with `target` unmangled and
   methods unlifted.

End state after both passes in `check.py`: an un-flattened `ImplBlock`
sits at the top level, its methods were never lifted to `Type__method`
FnDecls, and method-call sites like `p.area()`
(`Call(callee=Field(p, "area"), ...)`) were never rewritten to
`Type__area(p)` because `_rewrite_method_calls`
(`flatten_impls.py:105–108`) had already finished iterating
`prog.items` at step 1 before the `ImplBlock` was visible.

The codegen driver `helixc/backend/x86_64.py` lines 3104, 3107 runs
the OPPOSITE order: `flatten_modules` FIRST (lifting the mod-nested
ImplBlock to top level as an unmangled item via the same else-branch),
THEN `flatten_impls` (which now sees it at top level, lifts methods,
rewrites callsites). Backend output is correct; check.py output is not.

**Hidden errors.** The cycle-64 inline comment at `check.py:455–464`
(C-numbered "Stage 28.9 cycle 63 CN-A fix") and the `iter_fn_decls`
docstring at `ast_walker.py:225–249` both assert that the two
production drivers "now share the same prefix-pass order" and
"converge on the canonical post-flatten AST shape." Both claims
are factually wrong at HEAD `e7bd9c6` — the orders differ. The
comment is misleading documentation that masks the regression
introduced by inserting `flatten_modules` at the wrong position.

Same defect-class as the cycle-57 / C57-2 walker-drift cluster
(missing UnsafeBlock / TileLit arms in `_rewrite_expr`) and the
cycle-61 O60-F type-design finding (lifted FnDecl dropped `is_extern`):
an explicit-dispatch ladder missing one of the Item subclasses.
`_flatten_one`'s missing `ImplBlock` arm is the same shape.

**Impact.** A user-authored program of the form:

```
struct Pt { x: i32, y: i32 }
mod m { impl Pt { fn area(self: Pt) -> i32 { self.x * self.y } } }
fn main() -> i32 { let p = Pt { x: 3, y: 4 }; p.area() }
```

run through `helixc check` either:
(a) silently miscompiles (the un-rewritten `Field(p, "area")` callee
    survives into lower_ast where it is then either failed loudly or
    produces unreachable lowered code, depending on the downstream
    lowering arm reached), OR
(b) typecheck rejects it first with a misleading "struct 'Pt' has no
    field 'area'" diagnostic — also silent in the sense that the real
    cause (wrong flatten order) is hidden from the user.

The same input fed through `python -m helixc.backend.x86_64` compiles
correctly because the backend's flatten order is the reverse. Two
production drivers diverging on a documented post-flatten invariant
is precisely the regression class cycle-58 / cycle-60 hardened
`iter_fn_decls` and `_walk_items_for_fns` against — and cycle-64
re-introduced a sibling instance one layer up by ordering the two
flatten passes wrong.

**Recommendation.** Swap the two `try: flatten_*(prog)` blocks in
`check.py:438–470` so `flatten_modules` runs first (lines 466–470
moved before lines 438–446), matching `backend/x86_64.py:3104,3107`.
Additionally: add an explicit `ImplBlock` arm to `_flatten_one` in
`flatten_modules.py` that mangles `target` (e.g. `m__Pt`) and recurses
into `methods` — defense-in-depth so a future caller that bypasses
the canonical flatten_modules→flatten_impls order also gets correct
state. Update the cycle-63 CN-A comment at `check.py:455–464` and
the `iter_fn_decls` docstring to truthfully describe the order.
Regression test: `tests/test_flatten_pipeline.py` checking the
above reproducer compiles cleanly via `helixc check` (or asserts
that the post-flatten AST contains a top-level `FnDecl` named
`Pt__area`, NOT a top-level `ImplBlock`).

## Notes (<75)

- `helixc/frontend/flatten_modules.py:136–137`: the `else:
  new_items.append(sub)` fallthrough in `_flatten_one` does not
  bump the `n += 1` count, so the returned `flattened` value
  under-reports items lifted out of mods when an unhandled Item
  subclass (ImplBlock today, future Item subclasses tomorrow) is
  silently passed through. Conf ~70.
- `helixc/backend/x86_64.py:2833–2834`: `_emit_op` falls through
  to "Unsupported op — emit nothing (placeholder); v0.2 will lower
  tensor ops to runtime calls" for unhandled `tir.OpKind`. Documented
  as Phase-0 stub; current `lower_ast` never produces tensor /
  reduce / matmul ops, so unreachable today. Would prefer `raise
  NotImplementedError(f"x86_64: unhandled {op.kind.name} at
  {op.span}")` so a future lower_ast extension surfaces loudly.
  Conf ~60.
- `helixc/backend/ptx.py:331–332`: `// TODO: {op.kind.value}` silent
  comment for unhandled tile ops. Same observation as cycle-57
  notes — partial loudness via ptxas reject, but emit-time silent.
  Re-flagging would duplicate cycle-57; conf ~65.
- `helixc/ir/passes/fdce.py:30–31`: `if entry_fn not in
  module.functions: return 0` silent no-op without diagnostic.
  Documented as intentional but the caller cannot distinguish "no
  dead fns" from "skipped because entry_fn missing." Repeated
  observation from cycle-57. Conf ~55.
- `helixc/ir/passes/cse.py:101`: `cse_function` rewrite map is
  block-scoped and the doc explicitly limits CSE to "per-block ...
  for v0.1 per-block is sound". Cross-block CSE would need
  dominance analysis; absence is documented. Conf ~25.

## Edits made

NONE. This audit was conducted in strict read-only mode per the
cycle-65 silent-failures instructions. The only file written is
this audit document at the path the prompt specified. No source
files were modified, no Edits/Writes against source were
attempted, and the working-tree state (an uncommitted modification
of `helixc/check.py` reordering the flatten passes — presumably a
cycle-66 fix prepared by a parallel agent) was not touched. The
audit analyzed HEAD `e7bd9c6` only.
