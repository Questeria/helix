# K-Bootstrap Feature-Parity Matrix

**Status:** K0 chunk 1 (first authoritative survey) · **Date:** 2026-05-25
· **Parent plan:** [`HELIX_K_BOOTSTRAP_MASTER_PLAN.md`](HELIX_K_BOOTSTRAP_MASTER_PLAN.md).

This document is the gap list between **Python `helixc/`** (the
canonical compiler today) and **Helix-in-Helix `helixc/bootstrap/`**
(`kovc.hx` + `parser.hx` + `lexer.hx` + `evaluator.hx`, 15,618
lines). The K-track ports every PARITY → KOVC-MISSING row until
no row is missing; then the Python package can be deleted.

**Methodology:** two read-only survey agents walked the codebase
independently. The Python side enumerated `frontend/ast_nodes.py`
(60 dataclasses) + frontend passes (22) + IR passes (6). The
Helix side enumerated parser.hx AST tag comments (44 tags), kovc.hx
codegen dispatch, and the bootstrap built-in surface. The matrix
below is their merge — first draft; refinements land as the K-track
iterates.

**Legend:**

| Symbol | Meaning |
|--------|---------|
| ✅ | Fully supported (parses + lowers + matches Python behavior). |
| ⚠️ | Partial — parses but codegen traps OR has known gaps (e.g. stack args > 6, mixed-type binops). |
| ❌ | Missing — Python supports it, `kovc.hx` does not (codegen `ud2`-traps on dispatch). |
| ─ | Not applicable to this side. |
| ? | Survey uncertain; refine in a follow-up chunk. |

---

## 1. Types

| Feature | Python `helixc` | `kovc.hx` | Status |
|---------|-----------------|-----------|--------|
| `TyName` (i32, f32, named types) | ✅ | ✅ | PARITY |
| `TyTuple` (`(T1, T2, ...)`) | ✅ | ⚠️ (parsed; codegen ud2) | KOVC-MISSING |
| `TyArray` (`[T; N]`) | ✅ | ⚠️ (parsed; codegen ud2) | KOVC-MISSING |
| `TyRef` (`&T`, `&mut T`) | ✅ | ❌ | KOVC-MISSING |
| `TyPtr` (`*const T`, `*mut T`) | ✅ | ❌ | KOVC-MISSING |
| `TyFn` (`fn(T1) -> R`) | ✅ | ❌ (no fn-type values) | KOVC-MISSING |
| `TyTensor` | ✅ | ❌ | KOVC-MISSING |
| `TyTile` | ✅ | ❌ | KOVC-MISSING |
| `TyGeneric` (`Foo<A, B>`) | ✅ | ⚠️ (parsed via gp_tab; no monomorphization) | KOVC-MISSING |

## 2. Scalar literals + numerics

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `IntLit` (i32 default) | ✅ | ✅ (AST_INT, tag 0) | PARITY |
| `i64` literal suffix | ✅ | ✅ (AST_INTLIT_I64, tag 35) | PARITY |
| `i8 / i16 / u8 / u16 / u32 / u64` | ✅ | ✅ (tags 36-41) | PARITY |
| `FloatLit` (f32) | ✅ | ✅ (AST_FLOATLIT) | PARITY |
| `f64` literal | ✅ | ✅ (AST_FLOATLIT_F64, tag 34) | PARITY |
| `bf16` literal | ✅ | ✅ (AST_FLOATLIT_BF16, tag 42) | PARITY |
| `f16` literal | ✅ | ? | UNKNOWN |
| `BoolLit` (`true`/`false`) | ✅ | ? (no AST tag found in survey) | UNKNOWN |
| `CharLit` (`'a'`) | ✅ | ❌ | KOVC-MISSING |
| `StrLit` | ✅ | ⚠️ (AST_STR_LIT, tag 25; codegen emits 0 — only useful as file-IO arg) | KOVC-MISSING |

## 3. Operators

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `+ - * /` (i32 same-type) | ✅ | ✅ | PARITY |
| `%` (modulo) | ✅ | ✅ (AST_MOD, tag 24) | PARITY |
| Mixed-type binops (e.g. i64+i32) | ✅ (implicit conversion) | ⚠️ (codegen traps ud2) | KOVC-MISSING |
| Float arithmetic (f32) | ✅ | ✅ (via `__fadd/sub/mul/div/neg` SSE builtins) | PARITY |
| Float arithmetic (f64) | ✅ | ✅ | PARITY |
| Mixed f32/f64 arithmetic | ✅ | ⚠️ (traps) | KOVC-MISSING |
| Unary `-` (`AST_NEG`) | ✅ | ✅ | PARITY |
| Unary `!` (`AST_NOT`) | ✅ | ✅ | PARITY |
| Unary `~` / `AST_BNOT` | ✅ | ✅ | PARITY |
| Bitwise `& | ^` | ✅ | ✅ (BAND/BOR/BXOR) | PARITY |
| Shifts `<< >>` | ✅ | ✅ (AST_SHL/AST_SHR) | PARITY |
| Comparisons `< > <= >= == !=` | ✅ | ✅ | PARITY |
| Logical `&&` `||` | ✅ | ? (short-circuit semantics need check) | UNKNOWN |
| Address-of `&`, deref `*` | ✅ | ❌ | KOVC-MISSING |

## 4. Control flow

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `if` / `else` (`AST_IF`) | ✅ | ✅ | PARITY |
| `while` (`AST_WHILE`) | ✅ | ✅ | PARITY |
| `for` | ✅ | ❌ | KOVC-MISSING |
| `loop` (infinite) | ✅ | ❌ | KOVC-MISSING |
| `break` (with optional value) | ✅ | ❌ | KOVC-MISSING |
| `continue` | ✅ | ❌ | KOVC-MISSING |
| `return` (explicit) | ✅ | ❌ (AST_RET parsed; no codegen) | KOVC-MISSING |
| `match` + patterns | ✅ | ⚠️ (parsed; codegen ud2) | KOVC-MISSING |
| `Range` (`a..b`, `a..=b`) | ✅ | ❌ | KOVC-MISSING |

## 5. Patterns (for `match`)

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `PatLit` (literal) | ✅ | ❌ | KOVC-MISSING |
| `PatBind` (`x`, `mut x`) | ✅ | ❌ | KOVC-MISSING |
| `PatWildcard` (`_`) | ✅ | ❌ | KOVC-MISSING |
| `PatTuple` (`(a, b, c)`) | ✅ | ❌ | KOVC-MISSING |
| `PatOr` (`a | b | c`) | ✅ | ❌ | KOVC-MISSING |
| `PatRange` (`0..10`, `0..=10`) | ✅ | ❌ | KOVC-MISSING |
| `PatVariant` (`Enum::Variant(p)`) | ✅ | ❌ | KOVC-MISSING |
| `PatStruct` (`Point { x, y }`) | ✅ | ❌ | KOVC-MISSING |

## 6. Aggregates

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `TupleLit` (`(a, b, c)`) | ✅ | ⚠️ (AST_TUPLE_LIT, tag 50 parsed; codegen ud2) | KOVC-MISSING |
| Tuple field access (`.0`, `.1`) | ✅ | ⚠️ (AST_TUPLE_FIELD, tag 52 parsed; codegen ud2) | KOVC-MISSING |
| `ArrayLit` (`[1, 2, 3]`) | ✅ | ❌ | KOVC-MISSING |
| `Index` (`a[i, j]`) | ✅ | ⚠️ (AST_INDEX, tag 53 parsed; codegen ud2) | KOVC-MISSING |
| `StructLit` (`Point { x: 1, y: 2 }`) | ✅ | ⚠️ (parser has struct_table; codegen ud2) | KOVC-MISSING |
| Struct field access | ✅ | ⚠️ (Stage 5 Iter C only loads 64-bit; no real layout walk) | KOVC-MISSING |
| `TileLit` (`tile<f32, [N,M], mem>::zeros()`) | ✅ | ❌ | KOVC-MISSING |

## 7. Statements

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `let x = v` (`AST_LET`) | ✅ | ✅ | PARITY |
| `let mut x = v` (`AST_LET_MUT`) | ✅ | ✅ | PARITY |
| `x = v` (`AST_ASSIGN`) | ✅ | ✅ | PARITY |
| `x += v` etc. | ✅ | ? | UNKNOWN |
| `ExprStmt` (`expr;`) | ✅ | ✅ (via AST_SEQ chains) | PARITY |
| `const X: T = expr;` | ✅ | ❌ | KOVC-MISSING |
| `Cast` (`expr as T`) | ✅ | ❌ | KOVC-MISSING |

## 8. Declarations / items

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `fn` (basic) | ✅ | ✅ (`AST_FN_DECL`, 0-6 args) | PARITY |
| `fn` with stack-passed args (> 6) | ✅ (`SYSV_STACK_ARG_*` infra) | ⚠️ (traps 16002) | KOVC-MISSING |
| Generic `fn<T>` | ✅ (`monomorphize.py`) | ⚠️ (parser tracks gp_tab; no monomorph) | KOVC-MISSING |
| `where` clauses | ✅ | ❌ | KOVC-MISSING |
| `struct Foo { ... }` | ✅ | ⚠️ (parser has struct_table; codegen missing) | KOVC-MISSING |
| Parametric struct `struct<T>` | ✅ (`struct_mono.py`) | ❌ | KOVC-MISSING |
| `enum Foo { A, B(i32) }` | ✅ | ⚠️ (parser has enum_table; codegen ud2) | KOVC-MISSING |
| `type Alias = T;` | ✅ | ❌ | KOVC-MISSING |
| `const X: T = expr;` (top-level) | ✅ | ❌ | KOVC-MISSING |
| `use foo::bar::baz;` | ✅ | ❌ | KOVC-MISSING |
| `mod foo { ... }` / module decl | ✅ (`flatten_modules.py`) | ❌ | KOVC-MISSING |
| `impl Type { methods }` | ✅ (`flatten_impls.py`) | ❌ | KOVC-MISSING |
| `agent Foo { ... }` (AGI primitive) | ✅ | ❌ | KOVC-MISSING |

## 9. AGI / metaprogramming primitives

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `quote { ... }` (`AST_QUOTE`) | ✅ | ❌ | KOVC-MISSING |
| `splice(ast_value)` (`AST_SPLICE`) | ✅ | ❌ | KOVC-MISSING |
| `modify(target, tx, verifier)` (`AST_MODIFY`) | ✅ | ❌ | KOVC-MISSING |
| `reflect_hash(ast)` | ✅ | ❌ | KOVC-MISSING |
| `@trace` attribute | ✅ (`trace_pass.py`) | ❌ | KOVC-MISSING |
| `@checkpoint` (rematerialization) | ✅ | ❌ | KOVC-MISSING |
| `@autotune(KEY: [v1, v2, ...])` | ✅ (`autotune.py` + `autotune_expand.py`) | ❌ | KOVC-MISSING |
| `@deprecated` / `@since` | ✅ (`deprecated_pass.py`) | ❌ | KOVC-MISSING |
| `@partial` (non-totality) | ✅ (`totality.py`) | ❌ | KOVC-MISSING |
| `@pure` / `@effect(...)` capability typing | ✅ (`effect_check.py`) | ❌ | KOVC-MISSING |
| `unsafe { ... }` blocks | ✅ (`unsafe_pass.py`) | ❌ | KOVC-MISSING |
| `panic("msg")` builtin | ✅ (`panic_pass.py`) | ❌ | KOVC-MISSING |

## 10. AD framework

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `grad(f)` forward-mode | ✅ (`autodiff.py` + `grad_pass.py`) | ⚠️ (rudimentary differentiation present in parser.hx lines 5316+ for AD chain rules over AST_ADD/SUB/MUL/DIV/NEG; no `grad` builtin) | KOVC-MISSING |
| `grad_rev(f)` reverse-mode | ✅ (`autodiff_reverse.py`) | ⚠️ (per-param adj-bucket logic visible in parser.hx 5707+; not exposed as language feature) | KOVC-MISSING |
| `grad_rev_all` | ✅ | ❌ | KOVC-MISSING |
| 11 chain-rule builtins (sin, cos, exp, ...) | ✅ | ❌ | KOVC-MISSING |
| Kink-warn (non-smooth funcs) | ✅ | ❌ | KOVC-MISSING |

## 11. Type-system wrappers (v1.0 Tier-S/A)

`Diff<T>`, `Logic<T>`, `Modal<T>`, `Causal<T>`, `Conf<T>`, `Taint<T>`, `DP<T>`,
`Quant<T>`, `Domain<T>`, `Robust<T>`, `Energy<T>`, `Enclave<T>`,
`Counterfactual<T>`, `Deadline<T>`, `Attribution<T>` (15 composable
wrappers) — all ✅ in Python, all ❌ in `kovc.hx`. **Status: KOVC-MISSING
(all 15).**

## 12. Tile / tensor / GPU

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| Tile types `tile<T, [d], mem>` | ✅ | ❌ | KOVC-MISSING |
| `TILE_ZEROS` / `TILE_ADD/SUB/MUL` | ✅ | ❌ | KOVC-MISSING |
| `TILE_MATMUL` (wmma m16n16k16) | ✅ | ❌ | KOVC-MISSING |
| PTX backend | ✅ | ❌ | KOVC-MISSING |
| ROCm / Metal / WebGPU backends | ✅ | ❌ | KOVC-MISSING |
| MLIR migration path (Phase E) | ✅ | ❌ | KOVC-MISSING |

## 13. Built-in functions (runtime)

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `__arena_push / get / set / len` | ✅ | ✅ | PARITY |
| `__arena_push_pair / triple` (atomic) | ✅ | ❌ | KOVC-MISSING |
| `read_file_to_arena` / `write_file_to_arena` | ✅ | ✅ | PARITY |
| `print_int(i32)` | ✅ | ❌ | KOVC-MISSING |
| `__trace_event` (trace ring buffer) | ✅ | ❌ | KOVC-MISSING |
| `__helix_splice` / `__helix_modify` (reflection) | ✅ | ❌ | KOVC-MISSING |
| `__helix_reflect_hash` | ✅ | ❌ | KOVC-MISSING |
| FFI / `extern "C"` (linked syscalls) | ✅ | ⚠️ (`open`/`read`/`write`/`close` syscall stubs in kovc.hx ELF emitter) | KOVC-MISSING |

## 14. Frontend passes (Python only)

22 Python frontend passes. Each one is KOVC-MISSING in the
bootstrap — they need to land as `.hx` modules before the cutover.

| Pass | Purpose | Status |
|------|---------|--------|
| `ast_hash` | Structural hashing + alpha-equivalence | KOVC-MISSING |
| `ast_walker` | Shared traversal dispatcher | KOVC-MISSING |
| `autodiff` | Forward-mode AD | KOVC-MISSING |
| `autodiff_reverse` | Reverse-mode AD | KOVC-MISSING |
| `autotune` | `@autotune` validation | KOVC-MISSING |
| `autotune_expand` | `@autotune` cartesian expansion | KOVC-MISSING |
| `deprecated_pass` | `@deprecated` warnings | KOVC-MISSING |
| `flatten_impls` | Method-call dispatch | KOVC-MISSING |
| `flatten_modules` | Module flattening | KOVC-MISSING |
| `grad_pass` | `grad(f)` rewriting | KOVC-MISSING |
| `hash_cons` | AST hash-consing | KOVC-MISSING |
| `match_lower` | `Match` → `If`/`Let` desugar | KOVC-MISSING |
| `monomorphize` | Generic fn instantiation | KOVC-MISSING |
| `panic_pass` | `panic("msg")` lowering | KOVC-MISSING |
| `struct_mono` | Parametric struct instantiation | KOVC-MISSING |
| `totality` | Structural-recursion check | KOVC-MISSING |
| `trace_pass` | `@trace` instrumentation | KOVC-MISSING |
| `unsafe_pass` | `unsafe` block validation | KOVC-MISSING |
| `presburger` | Linear-arithmetic refinement solver | KOVC-MISSING |
| `pytree` | Pytree expansion | KOVC-MISSING |
| `diagnostics` | Caret-rendering error display | KOVC-MISSING |
| `typecheck` (full) | Type inference + refinement + effects | ⚠️ KOVC-MISSING (parser.hx has minimal type tags; no inference) |

## 15. IR passes (Python only)

| Pass | Purpose | Status |
|------|---------|--------|
| `const_fold` | Constant folding | KOVC-MISSING |
| `cse` | Common-subexpression elimination | KOVC-MISSING |
| `dce` | Dead-code elimination | KOVC-MISSING |
| `effect_check` | Effect-discipline verification | KOVC-MISSING |
| `fdce` | Function-level DCE | KOVC-MISSING |
| `tile_opt` | Tile-IR optimization | KOVC-MISSING |

## 16. Backends

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| x86-64 ELF (Linux) direct from AST | ✅ (`x86_64.py`) | ✅ (kovc.hx is exactly this) | PARITY (subset only) |
| LLVM IR text emitter | ✅ (`llvm_ir.py`) | ❌ | KOVC-MISSING (and possibly not needed — kovc.hx is direct-to-ELF) |
| LLVM toolchain wrapper | ✅ (`llvm_toolchain.py`) | ❌ | KOVC-MISSING (same reasoning) |
| MLIR substrate (Phase E) | ✅ | ❌ | KOVC-MISSING |
| Backend Protocol (Stage 220) | ✅ | ❌ | KOVC-MISSING |
| Parity gate (Stage 207 / 215) | ✅ (`llvm_parity.py`) | ❌ | KOVC-MISSING (this becomes Track P) |

---

## 17. Coverage tally

Rough count from the matrix above (UNKNOWN rows excluded from
both buckets):

| Bucket | Count |
|--------|-------|
| **PARITY** (kovc.hx matches Python) | ~28 rows |
| **KOVC-MISSING** (Python has it, kovc.hx does not) | ~110 rows |
| **PYTHON-MISSING** (kovc.hx has it but Python doesn't) | 0 |
| **UNKNOWN** (survey uncertain — refine next chunk) | ~5 |

The bulk of Helix's surface — types beyond scalars, control flow
beyond if/while, all patterns, all aggregates, all metaprogramming,
all AD, all type-system wrappers, all tile/GPU, all frontend passes,
all IR passes, all backends except the kovc.hx direct-x86 path — is
**KOVC-MISSING**. Honest reading: the bootstrap chain is a working
proof-of-concept for self-host; it covers ~20% of the actual Helix
language surface.

This is a multi-month porting effort. The K-track per the master
plan attacks these rows in dependency order, audited per-chunk,
gated by the parity harness (Track P) before any Python deletion.

## 18. Priority order for K1 (first ports)

Suggested order based on dependency (foundations first):

1. **Stack args > 6** (`fn` calls with > 6 params) — unblocks every
   downstream port that has multi-arg helpers. Small, isolated.
2. **`return` statement codegen** — already parsed; adding the
   codegen arm is one chunk.
3. **`for` loop + `Range`** — `while` is already supported; `for`
   desugars to `while` over a `Range`. Two coupled chunks.
4. **`break` / `continue` / `loop`** — completes the control-flow
   primitives.
5. **String literals (functional)** — needed by `panic("msg")` and
   any user-facing error or print path. Currently kovc.hx parses
   strings but the codegen emits `0`.
6. **`print_int` builtin** — the smallest non-trivial runtime
   builtin, needed for any tested program to produce observable
   output without using file IO.
7. **Tuples (`TupleLit` + `.field` access)** — already parsed (tag
   50, 52); needs codegen. Unlocks multi-return.
8. **`Cast` (`as` operator)** — Many Python features assume cast
   exists.
9. **`const` declarations** — small but commonly used.
10. **Structs (basic, non-generic)** — large surface; lays the
    foundation for the type-system-wrapper work later.

Subsequent K1 chunks pick up the remaining KOVC-MISSING rows in
roughly the order they unblock other work.

## 19. UNKNOWN-row refinement (deferred chunk)

A follow-up K0 chunk should resolve the UNKNOWN rows:

- `f16` literal in `kovc.hx`?
- `BoolLit` (`true` / `false`) AST tag in parser.hx?
- `&&` / `||` short-circuit semantics in kovc.hx?
- Compound assignment (`+=` etc.) in kovc.hx?
- Any AST tags 24, 27, 43-49, 51, 54-98 reserved or in use?

Resolving these tightens the gap-list count by ~5 rows and removes
the question marks.

---

## Appendix A — methodology

This matrix was produced by two parallel survey agents reading
the codebase read-only (no edits):

- **Python-side agent** read `helixc/frontend/ast_nodes.py`,
  `parser.py`, `typecheck.py`, `docs/lang/spec.md`, and listed
  every `frontend/` and `ir/passes/` module.
- **Helix-side agent** read all of `helixc/bootstrap/{lexer,
  parser,kovc,evaluator}.hx`, enumerated `// AST_*` tag comments
  from `parser.hx`, walked the codegen dispatch in `kovc.hx`,
  and listed built-in `__*` functions.

Both agent reports were merged into this matrix. Any row marked
UNKNOWN is where the two surveys did not converge; subsequent
chunks resolve them.

## Appendix B — what this matrix is NOT

- **Not a Python source-line count.** It enumerates language
  *features*, not implementation lines. A single row (e.g.
  "structs") corresponds to thousands of Python lines.
- **Not a runtime/library catalog.** Stdlib (`safety.hx`,
  `vec.hx`, etc.) is separate; this matrix is the *compiler*
  feature surface.
- **Not a test-suite parity check.** Track P (the parity harness)
  does that — runs every test through both compilers and asserts
  identical output. This matrix is the static feature gap.
