# Helix Trap ID Registry

**Last updated**: 2026-05-14 (Stage 33 bootstrap metadata — `28701` aux now points to deprecated metadata; autotune aux payloads are specific)
**Convention**: Each runtime trap has a numeric ID. The ID is encoded into `eax` immediately before a `ud2` instruction (SIGILL on x86_64), or surfaced as a structured `HelixCompileError` at compile time. Tools and tests cross-reference traps by ID.

## Two ID namespaces

### 1. Bootstrap codegen traps: `AST_TAG * 1000 + sub_id`

Used by the Helix-side bootstrap compiler (`helixc/bootstrap/kovc.hx`, `helixc/bootstrap/parser.hx`). The ID encodes which AST node tag emitted the trap, so debugging traces map back to source. AST tag table is in `helixc/bootstrap/parser.hx` (tag constants); a partial listing:

| AST tag | Node kind | Trap base |
|--------:|-----------|----------:|
| 1 | INT literal | 1xxx |
| 2 | BINOP | 2xxx |
| 3 | UNARY | 3xxx |
| 4 | VAR / NAME | 4xxx |
| 5 | BIND / LET | 5xxx |
| 7 | IF | 7xxx |
| 8 | LET | 8xxx |
| 10 | WHILE | 10xxx |
| 16 | CALL | 16xxx |
| 18 | PARAM | 18xxx |
| 24 | MOD (`%`) | 24xxx (collides with Stage-24 provenance namespace — see note below) |
| 71 | turbofish / type-param | 71xxx |
| 74 | impl block / method | 74xxx |
| 76 | closure | 76xxx |
| 91 | tile / tensor | 91xxx |
| 99 | error sentinel | 99xxx |

Bootstrap trap call sites use the form `emit_trap_with_id(N)`.

### 2. Stage-level reservations: `S * 1000 + sub_id`

Used by the Python frontend (`helixc/frontend/*.py`) and audit-introduced trap IDs. These do NOT follow the AST-tag convention. Each Python pass declares a module-level `TRAP_*` constant; auditors can collision-detect against this table.

## Known trap IDs

| ID | Constant name | Defined in | Stage | Meaning |
|----:|---------------|------------|-------|---------|
| 10030 | (bootstrap) | `helixc/bootstrap/kovc.hx` | stage5-6 F11 | `bind_alloc_offset` overflow past 1024-byte prologue allocation |
| 11001 | (typecheck) | `helixc/frontend/typecheck.py:1432` | 11 | `splice()` on a non-Quote value |
| 16003 | (monomorphize) | `helixc/frontend/typecheck.py:645` | 16 | tile/tensor call-site shape or memspace mismatch |
| 24001 | (bootstrap MOD) | `helixc/bootstrap/kovc.hx:~4220` | 24 (collision) | bf16 operand in `%` — **collides with the AST-tag 24 namespace**; resolved by reassigning Stage-24 provenance violation to 24100 |
| 24100 | (typecheck) | `helixc/frontend/typecheck.py:594, 746, 755` | 24 | non-Logic value passed where `Logic<T>` parameter required; provenance silently dropped |
| 24200 / AD002 | (typecheck) | `helixc/frontend/typecheck.py:1068, 1786` | 24 | TyDiff binop with mixed inner types; auto-widened with warning |
| 25001 | `TRAP_TRACE_OVERFLOW` | `helixc/frontend/trace_pass.py:37` | 25 | `@trace` buffer overflow at runtime |
| 25002 | `TRAP_TRACE_EQUIV_SHAPE_MISMATCH` | `helixc/frontend/trace_pass.py:38` | 25 | `trace_equiv` predicate shape mismatch |
| 25003 | (bootstrap trace_pass) | `helixc/bootstrap/kovc.hx` | 28.9 | `@trace` attribute recognised but codegen instrumentation pending (severity=1 warning only) |
| 26001 | `TRAP_PYTREE_DEPTH` | `helixc/frontend/pytree.py:49` | 26 | pytree flatten/unflatten depth cap exceeded (Phase-0 cap = 4); _unflatten guard added cycle 2 (deferred observation #17) |
| 26002 | `TRAP_PYTREE_NON_DIFF_LEAF` | `helixc/frontend/pytree.py:50` | 26 | pytree flatten saw a non-leaf-non-struct field type |
| 26003 | `TRAP_PYTREE_CYCLE` | `helixc/frontend/pytree.py:51` | 26 | cyclic struct reference detected during pytree walk |
| 27001 | `TRAP_AUTOTUNE_OVERSIZED` | `helixc/frontend/autotune.py:42` | 27 | `@autotune` variant Cartesian product exceeds 16; bootstrap aux is the saturated product |
| 27002 | (bootstrap autotune_pass) | `helixc/bootstrap/kovc.hx` | 33 | `@autotune` attribute is present without required `@kernel` |
| 27003 | (bootstrap autotune_pass) | `helixc/bootstrap/kovc.hx` | 33 | malformed, empty, or missing `@autotune(...)` parameter list; bootstrap aux kind: 1 missing parens, 2 malformed shape/token, 3 empty params/value-list |
| 28001 | `TRAP_PARAM_STRUCT_UNINSTANTIATED` | `helixc/frontend/struct_mono.py:38` | 28 | parametric struct used without `<T>` instantiation |
| 28002 | `TRAP_PARAM_STRUCT_CONSTEVAL` | `helixc/frontend/struct_mono.py:39` | 28 | parametric struct const-eval failed |
| 28501 | `TRAP_PANIC_INVOKED` | `helixc/frontend/panic_pass.py:35` | 28.5 | `panic("msg")` reached at runtime |
| 28502 | `TRAP_UNWIND_NOT_SUPPORTED` | `helixc/frontend/panic_pass.py:36` | 28.5 | `@unwind` attribute reserved but unimplemented |
| 28601 | `TRAP_UNSAFE_OP_OUTSIDE` | `helixc/frontend/unsafe_pass.py:33` | 28.6 | raw-pointer op (deref, arith) outside `unsafe {}` block |
| 28602 | `TRAP_EXTERN_CALL_OUTSIDE_UNSAFE` | `helixc/frontend/unsafe_pass.py:34` | 28.6 | `extern "C"` call outside `unsafe {}` block |
| 28603 | (typecheck) | `helixc/frontend/typecheck.py:1361, 1378, 1388` | 28.6 | raw-pointer Cast outside unsafe context |
| 28604 | (typecheck) | `helixc/frontend/typecheck.py:1394, 1408, 1777` | 28.6 | invalid scalar cast (not in allowed-cast matrix) |
| 28701 | (bootstrap deprecated_pass) | `helixc/bootstrap/kovc.hx` | 28.9 / 33 | call site of `@deprecated` fn (severity=1 warning by default; matches Python -Wdeprecated=warn policy). Stage 33: diag aux points to a deprecated metadata entry containing callee name and optional message. |
| 28702 | (bootstrap deprecated_pass) | `helixc/bootstrap/kovc.hx` | 28.9 cycle 1 | dep_tab cap reached (17th+ `@deprecated` fn in one program). Severity-1 warning emitted once per dropped name; Phase-0 cap is 16. Prevents silent loss of call-site detection. |
| 28999 | (bootstrap diag_arena) | `helixc/bootstrap/kovc.hx` | 28.9 | diag_arena overflow (>64 collected validation-pass diagnostics in a single program) |
| 28801 | `TRAP_SHAPE_FOLD_ZERO_DIV` | `helixc/frontend/monomorphize.py` (raised via `ShapeFoldError`) | 28.8 cycle 3 | division-by-zero or modulo-by-zero in a shape expression (e.g. `[T; N / 0]`). Hard error — silent fallthrough to length 0 is no longer allowed. |
| 28802 | `TRAP_ARRAY_SIZE_NEGATIVE_OR_ZERO` | `helixc/frontend/typecheck.py` | 28.8 cycle 3 | array size resolves to a negative or zero IntLit (source `[T; -5]` or mono-substituted `[T; N-N]`). Phase-0 requires size > 0. |
| 28803 | `TRAP_CAST_MATRIX_RECURSION_DEPTH` | `helixc/frontend/typecheck.py` | 28.8 cycle 3 | ref-nesting in cast exceeds 8 levels (`&&&...&i32 as &&&...&i64`). Defense in depth against Python recursion limit. |
| 60030 | (bootstrap pattern subpat) | `helixc/bootstrap/kovc.hx:emit_variant_subpats / emit_tuple_subpats` | 5-6 F9 | sub-pattern index > 15 in PAT_VARIANT or PAT_TUPLE — the disp8 form of the `mov rax, [rax+disp8]` load wraps signed at offset >= 128. Trap fires before the wrapping load. |
| 71001 | (bootstrap turbofish) | `helixc/bootstrap/parser.hx:~2626` | 7-8 F4 | `mr_tab_add` overflow at 33rd unique generic instantiation |
| 74002 | `TRAP_DUPLICATE_METHOD_NAME` | `helixc/frontend/flatten_impls.py:32` | 28 | duplicate method name across structs (Phase-0 ambiguity-free fallback) |
| 76001 | (bootstrap closure) | `helixc/bootstrap/parser.hx` | 9 | nested-closure error sentinel |
| 76002 | (bootstrap closure) | `helixc/bootstrap/parser.hx` | 9 | closure capture-table overflow (5th+ free var) |
| 76003 | (bootstrap closure) | `helixc/bootstrap/parser.hx` | 9 | closure capture of non-i32 local OR a local whose type can't be confirmed as i32. Phase-0 loud failure. Triggers in (a) typed-non-i32 case (`let pi: f64 = 3.14`), (b) untyped-uninferrable literal case (`let pi = 3.14_f64` whose type wasn't tracked into var_type_tab), and (c) untyped Call-RHS lets (`let pi = get_pi();`) — the parser registers tag 12 "untracked-call sentinel" that the capture probe treats as non-i32 per cycle-3 D2 (commit 3b321e6). |
| 85001 | `TRAP_AD_ASSUMED_ZERO` | `helixc/frontend/autodiff.py:57` | 12-14 | AD pass assumed 0 derivative for unhandled node type |
| 91001 | (bootstrap tile) | `helixc/ir/lower_ast.py` | 15 | tile shape cap (Phase-0: HBM 1D only) |
| 99001 | (bootstrap AST_ERR) | `helixc/bootstrap/kovc.hx` | n/a | generic error sentinel (catch-all in codegen fallback) |

## Reserved-but-unused

| ID | Meaning | Note |
|----:|---------|------|
| 28701 (would be Stage 28.7) | `@deprecated` runtime violation | Currently surfaces as compile-time warning only; no runtime path |

## How to add a new trap ID

1. Pick a namespace:
   - Bootstrap codegen: use `AST_TAG * 1000 + sub_id`. Look up the AST tag in `helixc/bootstrap/parser.hx`. Pick the next unused sub_id.
   - Python frontend: use `S * 1000 + sub_id` where S is the stage number, OR pick a stage-scoped subrange. Avoid AST-tag prefixes 1-99 unless your trap really belongs to that AST node kind.
2. Add a `TRAP_*` constant at module-level. Comment with the human-readable meaning + the namespace rationale.
3. Add a row to this table.
4. Add a regression test asserting the trap fires for the documented condition.
5. Run the cross-file collision check (search for the new number in `helixc/`): `grep -r '<number>' helixc/`.

## Collision history

- **24001** double-claim (resolved 2026-05-10 cycle 1 A4): bootstrap kovc.hx emits 24001 for AST_MOD bf16 (existing); typecheck.py reserved 24001 for provenance — reassigned to 24100. Going forward, the rule is: Python-side reservations should NOT overlap the bootstrap's `AST_TAG * 1000 + sub_id` ranges 1xxx..99xxx.

## Audit-time invariants

- Every `TRAP_*` constant must have at least one caller that actually emits it. Audit C1 cycle 1 found `@trace` reserved 25001 but never invoked it (now fixed in commit c418fb2).
- Every emitted ID must appear in this table. Cycle 2 audit C C2-L2 created this table because new IDs added in cycle 1 were only documented in source.
