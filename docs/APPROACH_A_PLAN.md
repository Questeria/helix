# Approach A: Full Bootstrap Port — Helix Compiled in Helix

**Goal:** `kovc.hx` (Helix bootstrap) compiles ALL programs that `helixc-Python` compiles, byte-identical. After completion, `helixc-Python` is dropped as a dependency. The final product builds from raw binary (hex0 → ... → kovc) without Python.

**Constraints:**
- No demos this run — language completion only.
- Per-stage audit-fix cycle (mirrors Phase 1.10 pattern: 5 multi-agent audits, all clean).
- Final gate: 5 consecutive clean audits across the whole codebase.
- Autonomous; TG updates per stage and per audit batch.

---

## Stages (ordered by dependency)

### Foundation: numeric types + collections

**Stage 1: i64 in bootstrap**
- Lex `_i64` suffix; lex large literals.
- Parse AST_INTLIT_64 (new tag).
- Codegen 8-byte `movabs rax, imm64` for i64 literals.
- 3-way bind type tag now extends to {i32=0, f32=1, f64=2, i64=3}.
- Helpers: `emit_add_rax_rcx_64`, comparisons (cmp via REX.W), etc.
- Tests: i64 round-trip, arithmetic, comparison, mixed-type ud2 trap.

**Stage 2: u8/u16/u32/u64 + i8/i16**
- Parser: type idents `u8` `u16` `u32` `u64` `i8` `i16` → bind tag.
- Codegen: zero-extension load (movzx), sign-extension load (movsx).
- Smaller sizes use the same 8-byte stack slot but emit narrower stores.

**Stage 3: Strings (full)**
- Representation: `(ptr: i64, len: i64)` — fat pointer in two stack slots.
- String literals: emit bytes into `.rodata`, return `(ptr_offset, len)` as a 16-byte struct.
- Operations: `str_len`, `str_eq`, `str_concat`, `str_slice`, `str_byte_at`.
- Lex literal: full escape handling (`\n`, `\t`, `\\`, `\"`, `\xHH`).

**Stage 4: Tuples + arrays**
- Tuple `(a, b, c)`: tag-based AST node, codegen as N-slot stack region.
- Tuple field `.0`, `.1`, etc.
- Static arrays `[T; N]`: contiguous stack region.
- Dynamic arrays via stdlib `vec.hx` (already exists in helixc-Python).

### Aggregates

**Stage 5: Structs (basic)**
- `struct Foo { a: i32, b: f64 }` → declare layout, allocate slot range.
- Field access `foo.a` → read at offset.
- By-value pass: callee gets struct flattened into N args.
- Lit construction `Foo { a: 1, b: 2.0_f64 }`.

**Stage 6: Enums (with payloads)**
- `enum Maybe { None, Some(i32) }` → discriminant + payload.
- Variant construction: `Maybe::Some(42)` → tag + payload write.

**Stage 7: Pattern matching (full Tier A)**
- `match x { ... }` with arm-by-arm dispatch.
- PatBind, PatLit, PatRange, or-patterns, guards.
- Exhaustiveness check at compile time.
- AD-through-match (defer if too complex; just need codegen first).

**Stage 8: Generics + monomorphization**
- `fn id<T>(x: T) -> T { x }` → emit per-instantiation.
- Mangling: `id<i32>` vs `id<f64>` get distinct fn names.
- Type substitution at call site.

### Mid-level features

**Stage 9: Closures**
- Capture environment as struct, lower to fn-with-env-arg.

**Stage 10: Modules + use**
- Already partially in parser; ensure full namespacing through codegen.

**Stage 11: Reflection runtime (kovc.hx side)**
- Mirror helixc-Python's `Quote`/`Splice`/`modify` semantics.
- Verifier-gated cells.

### ML primitives

**Stage 12: AD framework — forward mode in bootstrap**
- Tape-based forward AD; mirror `autodiff.hx`.

**Stage 13: AD across user-defined fn calls**
- Inline at AD time OR chain-rule through call ops.

**Stage 14: AD framework — reverse mode**
- Per-parameter buckets; multi-output return.

**Stage 15: Tile + tensor types + lowering**
- `tile T<HBM>`, `tile T<SMEM>`, etc.
- Memory-space attribute on stores; codegen lowers to actual SIMD.
- Matmul, elementwise lowering.

**Stage 16: PTX backend (port from Python)**
- Port `helixc/backend/ptx.py` to Helix bootstrap.
- GPU codegen for tile types.

### Pipeline passes

**Stage 17: const-fold pass (port from Python)**
- IR-level pattern-match folds (x*1, x+0, etc.).

**Stage 18: CSE / DCE / FDCE**
- Standard optimization passes.

**Stage 19: Effect/capability check pass**
- `@effect(io.read_file)` enforcement.

**Stage 20: Hash-cons (AST + IR)**
- Memoize differentiate(); structural sharing.

**Stage 21: Total-by-default check**
- Structural-recursion termination check; `@partial` annotation parse.

### Polish & differentiators

**Stage 22: Pretty error display**
- Source-with-caret format; did-you-mean suggestions.

**Stage 23: CLI flags**
- `--emit-ir`, `--dump-ast-hashes`, `--check-only`, etc.

**Stage 24: Provenance-typed neuro-symbolic** (Tier 3 strategic moat)
- `D<Logic<T>>` — differentiable relational data.

**Stage 25: Trace-based introspection**
- Runtime trace capture; verifier check on trace equivalence.

**Stage 26: JAX-style pytrees**
- `grad(loss)(model)` over nested structs.

**Stage 27: Triton-style autotune**
- `@autotune` parameter sweeps.

**Stage 28: Mojo-style parametric structs**
- Shape/dtype parametrization.

### Final gate

**Stage 29: Drop helixc-Python**
- Verify kovc.hx compiles every test case helixc-Python compiles, byte-identical.
- Mark helixc-Python as deprecated reference.

**Stage 30: 5 consecutive clean audits**
- Run multi-agent audit cycles until 5 in a row find zero new findings.

---

## Per-stage protocol

For each stage:
1. Spec the feature (what tags / encodings / type rules).
2. Implement in lexer.hx, parser.hx, kovc.hx.
3. Add tests covering the feature.
4. Run full test suite; verify pass.
5. Commit (one or more commits, batched per the lesson learned).
6. Run multi-agent audit cycle (3 agents in parallel: code-reviewer, silent-failure-hunter, type-design-analyzer).
7. Fix all findings.
8. Re-run full test suite.
9. Repeat audit until cycle finds zero new issues.
10. TG update: stage X complete, audits clean, moving to stage X+1.

## Final 5-clean-audits gate

After Stage 29:
1. Run multi-agent audit (3 agents).
2. If zero findings: count = count + 1.
3. If findings: fix them, count = 0, repeat.
4. Continue until count = 5.
5. TG: HELIX FULLY FINALIZED.

---

## Plan v2 amendments (research-agent integrations)

Research agent (2026-05-07) flagged production-grade gaps. Integrated:

**Inserted earlier in the sequence:**
- **Stage 1.5: bf16 / f16 scalar dtypes** — before tile codegen so numeric primitives don't need rewriting. SSE conversion intrinsics (`vcvtps2ph`/`vcvtph2ps`).
- **Stage 8.5: Traits + typeclasses (minimal Rust-style)** — before generics monomorphization wraps up; without traits, `kovc.hx` self-host writes the same boilerplate dozens of times.
- **Stage 14.5: `@checkpoint` / rematerialization for reverse-mode AD** — JAX-style memory-vs-compute knob; deep models OOM without it.
- **Stage 16.5: FFI / `extern "C"` + `repr(C)`** — before tile codegen, so we can link cuBLAS/cuDNN instead of reimplementing matmul.

**Added near end:**
- **Stage 28.5: panic / abort policy** — pick `abort` default, reserve `@unwind`. Documented + plumbed through codegen.
- **Stage 28.6: `unsafe` block for raw-ptr ops** — capability boundary for FFI + arena pointer arithmetic.
- **Stage 28.7: `@deprecated` + `@since` version gating** — stdlib evolution path post-self-host.

**Tooling appendix (after the language ships):**
- LSP server (textDocument/publishDiagnostics first, then hover/completion)
- Property-based testing harness (`@property fn`)
- Coverage-guided fuzzing
- `///` doc-comment generator
- Source maps (DWARF for `gdb` integration)

**Out of scope (research-agent disagreed with):**
- Full borrow checker (uniqueness types are enough for AGI/ML)
- Lean-4 proof-carrying terms (defer until external adoption)
- JIT / REPL (AOT + autotune covers this)
- Garbage collection (region/arena + affine buffers suffice)
- Cargo-style package manager (path-based modules through v1.0)
- Row-polymorphic effects (closed-set @effect is enough)

## Status

- **Started:** 2026-05-07
- **Current stage:** Stage 1 (i64 in bootstrap)
- **Total stages:** 30 + 7 amendments + tooling appendix
- **Estimated commits:** 200-400
- **Estimated audit cycles:** 50-100
- **Estimated wall time:** 6-12 months across many loop iterations

This document is the canonical plan. Each loop iteration reads it and resumes from the current stage.
