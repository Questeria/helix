# Audit Stage 28.9 cycle 98 — Type design

Scope: HEAD `1ff41ff`. Narrow rotation per cycle-98 brief:

- Verify cycle-97 type-surface fixes (`_is_float_type` exhaustiveness over
  lexer-accepted float suffixes; `_check_float_supported` arm completeness
  over quantized + halfprec types; `A.Loop` arm preserves block-CFG-graph
  invariant — no orphans).
- Fresh rotation:
  - `helixc/frontend/totality.py` — arm completeness over Pattern subclasses.
  - `helixc/ir/tir.py` — `Op` / `Block` / `FnIR` dataclass invariants.
  - `helixc/frontend/struct_mono.py` — `mangle_struct` injectivity.

Read-only audit (Read / Grep / Glob / Bash). No code edits performed.

Prior-cycle findings C1..C97 and known deferred items are intentionally
NOT re-flagged. Parallel Stage 28.10 / 28.11 commits are out of scope.

---

## Verdict: PASS — 0 findings at confidence ≥ 75 %

---

## Cycle-97 fix re-verification (all confirmed)

### V1 — `_is_float_type` extension is exhaustive over lexer suffixes

`helixc/frontend/lexer.py:338-341` enumerates the suffix set the lexer
recognizes at a numeric-literal trailing `_<name>`:

```
{"i8","i16","i32","i64","isize",
 "u8","u16","u32","u64","usize",
 "bf16","f16","f32","f64",
 "fp8","mxfp4","nvfp4","ternary"}
```

The float-domain partition (excluding the integer ones) is exactly
`{bf16, f16, f32, f64, fp8, mxfp4, nvfp4, ternary}` — 8 names.

`helixc/backend/x86_64.py:999-1014` (`_is_float_type`) post-cycle-97
returns `True` on `TIRScalar.name` ∈ this same 8-name set. Exhaustive
match with the lexer's float-suffix universe. No silent integer-ABI
fallthrough remains for any literal a programmer can write.

(The TIR layer also constructs `TIRScalar("fp8_e4m3")`-style variants
per the docstring at `tir.py:41`, but those are not produced by the
Phase-0 lexer / typechecker path — only the 8 canonical names reach
the backend in current code. If a future parser admits an `fp8_e4m3`
suffix, `_is_float_type` will need re-extension; that is a future-stage
concern, not a current-cycle finding.)

### V2 — `_check_float_supported` arm enumerates unsupported set

`helixc/backend/x86_64.py:1036-1052` (`_check_float_supported`) raises
`NotImplementedError` for `TIRScalar.name` ∈
`{f16, bf16, fp8, mxfp4, nvfp4, ternary}`. The supported set
(= float-domain \ rejected) is `{f32, f64}`, matching the diagnostic
text exactly. With V1 above, every float value now reaches this guard
before any backend-side codegen, so the quantized / halfprec types
surface a loud diagnostic instead of miscompiling under the integer
ABI.

### V3 — `A.Loop` fix preserves CFG block-graph invariants

`helixc/ir/lower_ast.py:1909-1910` now calls
`self.builder.append_block()` for both header and body. The contract
of `append_block` (`tir.py:400-406`) is: (a) mint a fresh `Block`,
(b) append it to `self.current_fn.blocks`. The contract of the
former `new_block()` (`tir.py:379-382`) was only (a), which orphaned
the header/body off the function.

Post-fix, all four CFG-building sites in `_lower_expr` are uniform:
`If/Else` (~1733), `For` (~1813), `While` (~1873), `Loop` (~1909).
Every block produced reaches `fn.blocks`; every consumer that walks
`fn.blocks` (backend slot pre-allocation, label emission, BR target
lookup) sees the same set of blocks that produced the IR. Regression
test `test_ir.py::test_c96_loop_blocks_appended_to_fn_blocks` pins
both the `len(fn.blocks) >= 3` invariant and the BR-target ↔
`fn.blocks` membership check.

---

## Fresh rotation findings

### `helixc/frontend/totality.py` — Pattern subclass arm completeness

No finding. The module's discrimination over pattern shapes is
deliberately conservative-fail by design:

- Pattern enumeration over recursive-call shapes lives in
  `_is_strictly_smaller` (lines 147-156), which recognizes two
  syntactic strict-decrease forms (`p - k`, `p / k≥2`). The docstring
  at lines 16-19 explicitly states: *"Conservative: returns True
  (= 'totality unprovable') for any pattern we don't yet recognize."*
  Unrecognized arg shapes → `False` → no parameter strictly decreases
  → fn flagged. This is sound (false-positive rather than
  false-negative on partiality detection).
- Walker discipline uses `ASTVisitor.visit_Call` with no manual
  `generic_visit` call (per cycle-73 fix: ASTVisitor auto-descends
  unless the override returns `False`, see `ast_walker.py:192-196`).
  Verified the auto-descent contract in current `ast_walker.py`.
- Fn enumeration uses `iter_fn_decls` (cycle-71 fix), which recurses
  through `ImplBlock.methods` and `ModBlock.items` — drift-proof for
  future container subclasses.

Arm-completeness over `Pattern` subclasses is N/A here because totality
operates on `Expr` / `Call` shapes, not match-`Pattern` subclasses.
(Pattern-arm coverage for the Match-desugar pipeline is the rotation
target of a separate module — `match_lower.py` — not in cycle-98 scope.)

### `helixc/ir/tir.py` — `Op` / `Block` / `FnIR` dataclass invariants

No finding. Invariants verified:

- **`FnIR.blocks` ↔ `FnIR.entry`** (line 350-352). The `entry` property
  unconditionally returns `blocks[0]`. The invariant *"blocks[0] is the
  entry block"* is established by `IRBuilder.begin_function` (line 388-
  390): the entry block is constructed via `new_block()` and placed at
  index 0 of the fresh `blocks` list. The only mutation primitive
  thereafter is `append_block` (line 400-406), which appends to the
  end, preserving `blocks[0]`. No public API in `tir.py` permits
  prepending, deleting, or reordering blocks. Invariant holds.
- **`Module.next_block_id` / `next_value_id` monotonicity.** Both are
  incremented only inside `new_block` / `new_value`. Single-threaded
  builder use is the documented contract (`current_fn` / `current_block`
  fields are scalars, not stacks); under that contract IDs are
  globally unique within a `Module`. No collision pathway visible.
- **`Op.results` cardinality.** `IRBuilder.emit` (line 418-430)
  produces a `results` list of length 0 or 1 (driven by whether
  `result_ty` is provided). No multi-result construction pathway in the
  builder. Consumers (e.g. fmt_op, backend) handle the 0/1 cases
  symmetrically.
- **`Op.attrs` typing.** Declared `dict[str, object]`. Per-OpKind
  attr schemas are encoded informally as comments on the `OpKind`
  enum members. Schema enforcement remains deferred (consistent with
  prior cycle disposition — not re-flagged).
- **`Value` identity.** `__hash__` / `__eq__` use `id` only; type is
  not part of equality. Standard SSA practice; matches Cranelift CLIF
  / Swift SIL convention (cf. module-header docstring line 10).

### `helixc/frontend/struct_mono.py` — `mangle_struct` injectivity

No fresh finding. The underscore-separator non-injectivity in
`mangle_struct` / `_mangle_ty(TyGeneric)` (two-arg `Pt<foo_bar, baz>`
collides with `Pt<foo, bar_baz>` because the inter-arg `_` separator
is indistinguishable from a `_` inside a `TyName.name`) is the same
issue documented as deferred-known across cycles 65, 76, 87, 88, 94,
and 96 (most recently in `docs/audit-stage28-9-cycle96-silent-
failures.md:122-126`: *"deferred-known items (`monomorphize._mangle_ty`
silent catchall, `hash_cons._ast_equal`, `typecheck/struct_mono`
pre-flatten in `check.py`, `autotune.collect_autotuned_fns
iter_fn_decls`, `struct_mono.mangle_struct` collision) were not
re-examined."*).

Per scope rules, deferred-known is NOT re-flagged.

Other injectivity-adjacent properties verified clean:

- `_ty_key` (lines 244-304) has proper arms for every `TyNode` AST
  subclass produced by the parser; the fall-through (line 304) is
  guarded by an `isinstance(t, A.TyNode)` check at line 298 that
  raises `TypeError` for non-AST types (cycle-3 D6 fix). The
  fall-through `("?", type(t).__name__)` is only reachable for a
  future `TyNode` subclass that hasn't been wired — the same defect
  class as `_mangle_ty`'s `NotImplementedError` (cycle 71 CN-2),
  except this site is dedup-side and silent. That asymmetry is the
  documented mangle-struct deferred bug above, not a new one.
- `instantiate` (lines 307-332) checks `len(ty_args) == len(decl.generics)`
  before substitution. Arity mismatch raises `ValueError` (line 313-
  316). No silent failure.
- `monomorphize_structs` (lines 335-388) dedupes mangled names against
  pre-existing struct decls in `prog.items` (cycle-3 C3-4 fix), so
  re-invocation on the same program is idempotent.

---

## Re-checks of recently-audited paths (not re-flagged)

Per scope rules, deferred-known items (`monomorphize._mangle_ty`
silent catchall — *now loud-fail per cycle-71 CN-2 but the cycle-65
analysis around dependent fall-through stays deferred*;
`hash_cons._ast_equal`; `typecheck / struct_mono` pre-flatten in
`check.py`; `autotune.collect_autotuned_fns iter_fn_decls`;
`struct_mono.mangle_struct` `_`-separator collision) were not
re-examined.

Cycle-97 fixes (`_is_float_type` + `_check_float_supported` exhaustiveness;
`A.Loop` `append_block` migration) were re-verified above — V1, V2, V3 —
all confirmed.

---

## Summary line

PASS — 0 findings at conf ≥ 75 %.
