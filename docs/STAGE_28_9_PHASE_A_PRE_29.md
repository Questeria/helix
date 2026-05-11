# Stage 28.9–28.13: Phase A — Pre-29 Port Work

**Created**: 2026-05-10 (after Stage 28.8 cycle 1 research)
**Trigger**: User directive "fix the frontend passes also" after research doc surfaced 8 Python-only frontend passes with no bootstrap counterpart.
**Source**: `docs/helix-pre-self-host-research.md`
**Predecessor**: Stage 28.8 (pre-29 audit gate — must reach 5 clean cycles first)
**Successor**: Stage 29 (drop Python reference — GATED on user approval)

## Rationale

`docs/helix-pre-self-host-research.md` identified the actual blocker for Stage 29: **8 Python-only frontend passes that have no Helix-side counterpart**. Dropping the Python reference at Stage 29 without porting them would mean every test case that exercises those features silently uses "whatever the bootstrap does today" — a provably proper subset of Python's behavior.

Phase A is the discipline path: port what we have, plus the ergonomics the bootstrap itself will benefit from. Phase B (Tier-3 moat) is deferred to v0.2.

## Phase A scope (Stages 28.9–28.13)

### Stage 28.9: Port match_lower into kovc.hx

**Python file**: `helixc/frontend/match_lower.py`
**What it does**: Desugars `Match` expressions into a chain of `If`/`Let`/comparison ops at lower-AST time. Handles literal patterns, variant patterns, tuple patterns, or-patterns, range patterns, guards.
**Current bootstrap status**: kovc.hx handles literal + simple variant patterns; PatTuple, PatOr, PatRange tested but trap on the complex paths; guards are Phase-1.
**Port plan**:
1. Add a `match_lower_pass` fn in kovc.hx that walks the AST after parse and before codegen.
2. Lowering rules per pattern kind:
   - `PatLit(lit)`: emit `if scrut == lit { body } else { next_arm }`
   - `PatBind(name)`: emit `let name = scrut; body` (always matches)
   - `PatVariant(name, subpats)`: emit `if __enum_discriminant(scrut) == variant_idx { let payload = __enum_payload(scrut); recurse on subpats } else { next_arm }`
   - `PatTuple(subpats)`: emit nested if/let chain on tuple-field accesses.
   - `PatOr(a, b)`: emit `if matches!(scrut, a) { body } else if matches!(scrut, b) { body } else { next_arm }`
   - `PatRange(lo, hi, inclusive)`: emit `if scrut >= lo && scrut <= hi { body } else { next_arm }` (or `<` for exclusive)
3. Guard: each arm's body is conditionally wrapped: `if guard_expr { body } else { next_arm }`.
4. Add tests: each pattern kind compiles to the same final bytes as Python's `match_lower` does on a small corpus.

**Risk**: kovc.hx already has partial match handling; the port has to NOT regress the simple paths.
**Effort**: ~1 stage, 6-10 commits.

### Stage 28.10: Port struct_mono into kovc.hx

**Python file**: `helixc/frontend/struct_mono.py`
**What it does**: For each `struct Foo<T> { ... }` referenced as `Foo<i32>`, `Foo<f64>`, etc., emit a monomorphized clone with the type substituted. Mangles to `Foo__i32` / `Foo__f64`. Walks both fn signatures AND fn bodies (Wave 1 of cycle 1 just fixed the body-walking gap in Python).
**Current bootstrap status**: kovc.hx has `mono_table` for FN monomorphization (existing); has NO struct monomorphization.
**Port plan**:
1. Add `struct_mono_table` analogous to `mono_table` — keyed on `(struct_name, type_args_tuple)`.
2. Pre-codegen walk:
   - Collect concrete uses: walk all `TyGeneric(base, args)` references in fn signatures, fn bodies (including Let-stmt annotations, struct-lit base, cast targets).
   - For each `(base, args)` where `base` is a user-defined generic struct, register a mono entry.
3. Emit mono clones: for each entry, deep-clone the struct decl, substitute type params throughout field types, register under mangled name.
4. Use-site rewrite: replace `TyGeneric("Foo", [i32])` with `TyName("Foo__i32")` in all AST positions.
5. Tests mirror Python's `test_struct_mono.py` cases.

**Risk**: Cross-stage with the typechecker — bootstrap's typecheck needs to lookup mangled names where Python's typecheck has the resolution logic.
**Effort**: ~1.5 stages, 8-12 commits.

### Stage 28.11: Port validation passes (panic + unsafe + deprecated + trace)

**Python files**: `helixc/frontend/panic_pass.py`, `unsafe_pass.py`, `deprecated_pass.py`, `trace_pass.py`
**What they do**: Static-analysis passes that diagnose: panic-arg validation, raw-ptr ops outside unsafe blocks, deprecated-symbol use sites, trace-attr placement.
**Current bootstrap status**: kovc.hx has no diagnostic-pass infrastructure. All diagnostics today are `emit_trap_with_id` at codegen, which is too late to catch validation issues cleanly.
**Port plan**:
1. Add a `diag_arena` in kovc.hx — a side-table mapping `(span, severity, code, msg_ptr)` for collected diagnostics.
2. Add a `validate_pass` orchestrator that runs after typecheck and before lowering. Each sub-pass (panic, unsafe, deprecated, trace) gets a slot.
3. Per sub-pass:
   - **panic_pass**: walk the AST; for each `Call(callee=Name("panic"), args)`, validate `args.len()==1 && args[0] is StrLit`. Emit diag if not.
   - **unsafe_pass**: thread an `in_unsafe` boolean through the AST walk. For raw-ptr ops (Unary `*`, Cast to TyPtr, etc.), if `!in_unsafe`, emit diag with trap-id 28601.
   - **deprecated_pass**: walk the AST; for each call site whose callee resolves to a `@deprecated` fn, emit warning with the deprecation message.
   - **trace_pass**: validate `@trace` is only on fns (not extern, not methods on traits).
4. Render diags via the bootstrap's existing `emit_diagnostic` helper.
5. Tests: each pass detects its target conditions; non-empty diag list fails the build (return 1 from driver_main).

**Risk**: The AST-walker scaffolding doesn't exist in kovc.hx — building it once benefits all four passes. But getting the visitor right (matches Python's walker, hits all sub-expressions including indices/guards/iter_expr per audit-cycle-1 walker-fix findings) is the load-bearing part.
**Effort**: ~1.5 stages, 10-15 commits.

### Stage 28.12: Port pytree (flatten/unflatten for AD over user structs)

**Python file**: `helixc/frontend/pytree.py`
**What it does**: Decomposes a user struct into its leaf gradients along the AD pipeline. `flatten_pytree(model_struct, grads_dict)` produces a flat list of (path, value) tuples; `unflatten_pytree` reverses.
**Current bootstrap status**: kovc.hx has no pytree pass. `grad(loss)(model)` for `model: SomeStruct` is unimplemented in bootstrap.
**Port plan**:
1. Add `pytree_flatten` in kovc.hx that walks a struct decl + field-tag table, emits a list of `(field_path, field_offset, leaf_ty_tag)` tuples for each leaf (f32/f64/bf16/f16, optionally D<>-wrapped).
2. Add `pytree_unflatten` for the reverse.
3. Cycle-detection guard: max depth = 8 (mirror Python's flatten_pytree cap).
4. Wire `grad(loss)(model_struct)` to use pytree decomposition.
5. Tests: round-trip flatten→unflatten preserves struct identity; cyclic structs trap rather than recurse.

**Risk**: Phase-0 limitation: bootstrap's grad pass is forward-mode only and doesn't handle struct args at all. Some scaffolding required upstream.
**Effort**: ~1 stage, 5-8 commits.

### Stage 28.13: Ergonomics cluster

**Items**:
1. **`?` operator** for Option/Result early-return.
2. **`let-else`** for refutable binding with else-block.
3. **Named struct-lit fields** `Foo { x: 1, y: 2 }` (currently positional only in bootstrap).
4. **`f"..."` string interpolation** with `{expr}` splicing.
5. **Bootstrap-side render_caret** for pretty error messages — currently Python-only.

**Per-item port plan**: each is a parser extension + lowering rule. Roughly 1 commit per item. Total: ~5 commits.

**Risk**: Low individually. High collectively because each forces a re-baseline of the byte-identical gate at Stage 29 — so the cost of skipping is steep (every retrofit becomes more expensive).
**Effort**: ~1 stage, 5-8 commits.

## Phase A summary

| Stage | What | Effort | Risk |
|-------|------|--------|------|
| 28.9 | Port match_lower | 1 stage | MEDIUM (regression risk on partial existing impl) |
| 28.10 | Port struct_mono | 1.5 stages | MEDIUM (cross-cuts typecheck) |
| 28.11 | Port validation passes (4 of them) | 1.5 stages | LOW (additive) |
| 28.12 | Port pytree | 1 stage | LOW (additive) |
| 28.13 | Ergonomics cluster | 1 stage | LOW (per item) |

**Total**: 5 stages, ~30-50 commits. Each stage follows the same per-stage protocol: spec + implement + test + multi-agent audit + iterate to zero new findings.

## Phase A audit discipline

Each Phase A stage follows the Stage 28.8 audit pattern:
1. After implementation, spawn 3 parallel audit subagents.
2. Findings tracked in `docs/audit-stage28-9-*.md`, `audit-stage28-10-*.md`, etc.
3. Counter advances on zero new HIGH/CRITICAL findings.
4. Once Phase A is complete + 5 consecutive clean cycles, Stage 29 gate opens (user approval still required).

## Phase B (DEFERRED to v0.2)

Per research recommendation, Tier-3 strategic moat work (D<Logic<T>> fuzzy AND/OR/NOT codegen + TyMemTier cost annotations) is deferred until an external neuro-symbolic benchmark exists to validate semantics. Tracked as Stages 28.14–28.20 in the master plan but NOT scheduled.

## Open questions

1. Does the bootstrap have enough infrastructure today to support a generic AST walker, or does Stage 28.11 require building that first? **Investigation owed pre-Stage-28.11**.
2. Does Stage 28.12 (pytree) need Stage 28.10 (struct_mono) to land first, since pytree operates over user structs? **Likely yes — order matters**.
3. Are there any Python-only passes I missed beyond the 8 enumerated? **Re-scan `helixc/frontend/*.py` before each stage starts**.

## Status (2026-05-10 20:18)

- Stage 28.8 cycle 1 audit fixes: Wave 1 done (8 commits, 1290 tests pass); Wave 2 running in background agent ade7cc5b1e8d48c1b; Wave 3 + cycle restart pending.
- Phase A: planned (this doc); will fire AFTER Stage 28.8 reaches 5 clean cycles.
- Stage 29: GATED on user approval, not on autonomous trigger.
