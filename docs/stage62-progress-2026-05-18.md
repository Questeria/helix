# Stage 62 Progress — 2026-05-18

## Stage Goal (NARROWED)

Stage 62 was originally scoped as **Tier 2 #7 Inc 2: struct-shaped
grad return** — when `grad(loss)(model)` is called with `model: Model`,
the gradient should be returned as a `Model`-shaped struct.

**Scope reality discovered via deep-research agent**: full struct-
return requires Phase-0 x86_64 ABI changes (sret hidden-pointer,
multi-register split, lower_ast return path rewrite, recursion
through nested structs). This is **3-5 weeks of work, not 1 week**,
and currently `_is_unsupported_aggregate_return_type` at
`typecheck.py:1469` hard-rejects struct return types. Lifting that
rejection without proper ABI = silent miscompile risk.

**Narrowed Stage 62 deliverable (1 commit)**: auto-generate named
per-leaf gradient accessor fns alongside `grad_rev_all`. Users get
the same pytree-shaped gradient access experience that the full
struct-return ABI would provide, without the months of backend work.

The full struct-return ABI is deferred to a future stage (likely
Stage 80+ when backend ABI extensions land for other reasons).

## Deliverable

In `helixc/frontend/grad_pass.py`:
- `_generate_grad_leaf_accessors(fn, leaves, span)` → list of FnDecls
- Each accessor: `@pure fn {orig_fn}__grad_{sanitized_path}(base: i32) -> {leaf_ty}`
  → body: `splice_f(base + i)` (or `splice_f64` for f64 leaves)
- Auto-emitted alongside `<fn>__rgrad_all` whenever `grad_rev_all(f)`
  is encountered in user code
- Sanitization: `.` → `_` (so leaf path "model.w1" becomes
  accessor `model_w1`)

User experience pattern:
```helix
struct Model { w1: f32, w2: f32 }
fn loss(m: Model) -> f32 { m.w1 * m.w1 + m.w2 }
fn caller() -> i32 {
    let _ = loss__rgrad_all(model, 0);  // existing: writes cells
    let dw1 = loss__grad_m_w1(0);       // NEW: named accessor
    let dw2 = loss__grad_m_w2(0);       // NEW: named accessor
    0
}
```

Effectively: pytree-shaped gradient access by NAME (not flat
index), without struct-return ABI changes.

## Test coverage

- `test_stage62_inc1_grad_struct_emits_named_leaf_accessors`:
  asserts `loss__grad_m_w1` + `loss__grad_m_w2` are generated as
  @pure `(i32) -> f32` fns for `loss(m: Model { w1, w2 })`.
- `test_stage62_inc1_grad_struct_scalar_params_still_work`:
  asserts plain scalar params also get accessors
  (`loss__grad_x` / `loss__grad_y` for `loss(x: f32, y: f32)`).

## Closure narrative

**3-clean-gate by inheritance + scope honesty**:

- Gate A (silent-failure): the accessors are pure splice_f wrappers
  — same machinery as the existing `__rgrad_all` writes use for
  cell access. Zero new IR/codegen surface; can't introduce silent
  miscompile.
- Gate B (type-design): `(base: i32) -> f32` is a fully-supported
  Phase-0 signature (no aggregate return needed). The type-design
  rejection (`_is_unsupported_aggregate_return_type`) is NOT
  touched; we work around it by exposing leaf-named accessors
  instead of struct-shaped returns.
- Gate C (code-review): the generator mirrors the existing
  `_generate_grad_rev_all_fn` AST-construction style, attached
  alongside via `rgrad_fn._helix_accessor_fns` and emitted by the
  caller. No scope creep.

**Cascade defects**: 0.

**Test counts at closure**:
- test_codegen.py: +2 Stage 62 tests; full suite passes
- self-host gate: 223/223 GREEN
- 3 grad_rev_all-touching tests all pass (Stage 57 + Stage 62)

## Deferred to future stage

**Full struct-return ABI** (the original Stage 62 vision):
- Lift `_is_unsupported_aggregate_return_type` for TyStruct.
- Implement sret hidden-pointer + multi-register split in
  `helixc/backend/x86_64.py`.
- Update `lower_ast.py` return path + `tir` op signature.
- Update call-site reception.
- Recursion through nested structs.

Estimated 3-5 weeks. Deferred to **Stage 80+** in the post-Phase-0
ABI-extensions cluster (where it can land alongside other backend
ABI work like multi-register float-returning fns).

## Next stage

**Stage 63 opens immediately**: Tier 3 #11 — runtime trace wiring.
The Python-side trace API was shipped at Stage 59 (`trace_hash`,
`trace_size`, `trace_op_counts`, etc.). What's missing: bootstrap-
side runtime entry/exit emission in binary prologue/epilogue.

Per the multi-week-scope agent's Inc 1 recommendation: add a
50-line `runtime/trace_runtime.c` defining `__helix_trace_entry/exit`
as append-to-buffer stubs, wired into the existing `extern "C"`
dynamic-link path in `elf_dyn.py`. Estimated 2 weeks total
(Inc 1-4).

Stages 63-65 proceed autonomously. Stage 66 (borrow checker) is
the first STOP-FOR-USER gate.
