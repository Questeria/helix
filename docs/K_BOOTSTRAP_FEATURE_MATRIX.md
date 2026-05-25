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
| `f16` literal | ✅ | ❌ (type tag 5 reserved in kovc.hx line 1177; no AST tag emitted) | KOVC-MISSING |
| `BoolLit` (`true`/`false`) | ✅ | ❌ (no `true`/`false` keyword in lexer.hx; the bootstrap has NO bool literal — `i32` 1/0 is used instead) | KOVC-MISSING |
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
| Logical `&&` `||` | ✅ | ❌ (no `&&` or `||` token in lexer.hx — only `&` `|` for bitwise) | KOVC-MISSING |
| Address-of `&`, deref `*` | ✅ | ❌ | KOVC-MISSING |

## 4. Control flow

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `if` / `else` (`AST_IF`) | ✅ | ✅ | PARITY |
| `while` (`AST_WHILE`) | ✅ | ✅ | PARITY |
| `for` | ✅ | ✅ (K1.G, 2026-05-25, commits 889b8b1 + 52599d7: parse_for desugars `for var in start..end { body }` to AST_LET_MUT + AST_WHILE + AST_SEQ + AST_ASSIGN + AST_ADD + AST_LT using only existing tags) | PARITY |
| `loop` (infinite) | ✅ | ✅ (K1.H1, 2026-05-25, commits 41497a3 + this commit: parse_loop desugars `loop { body }` to AST_WHILE(AST_INT(1), body), no new tag; break/continue still pending as K1.H2/H3) | PARITY |
| `break` (with optional value) | ✅ | ❌ | KOVC-MISSING |
| `continue` | ✅ | ❌ | KOVC-MISSING |
| `return` (explicit) | ✅ | ✅ (K1.C, 2026-05-25, commits 816ce51 + b02017f: AST_RET tag 43 + parse_return + parse_primary arm) | PARITY |
| `match` + patterns | ✅ | ✅ (Stage 5+ match-arm codegen at kovc.hx -- int-literal arms, wildcard `_`, enum-variant tags + payload destructure, bare tuple destructure all verified end-to-end. K1.F-discovery batch 2 2026-05-25: 4 regression tests pin behaviour via bootstrap-self-host) | PARITY |
| `Range` (`a..b`, `a..=b`) | ✅ | ❌ | KOVC-MISSING |

## 5. Patterns (for `match`)

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `PatLit` (literal) | ✅ | ❌ | KOVC-MISSING |
| `PatBind` (`x`, `mut x`) | ✅ | ✅ (K1.F-discovery batch 2 2026-05-25: verified inside `(a, b)` tuple destructure -- the bare-binding sub-pattern binds local vars that are then usable in the arm body) | PARITY |
| `PatWildcard` (`_`) | ✅ | ✅ (K1.F-discovery batch 2 2026-05-25: verified via `test_bootstrap_kovc_match_wildcard_fallback_self_host` -- wildcard arm fires when no literal matches) | PARITY |
| `PatTuple` (`(a, b, c)`) | ✅ | ✅ (K1.F-discovery batch 2 2026-05-25: verified via `test_bootstrap_kovc_pat_tuple_destructure_self_host` -- `match (3,4) { (a,b) => a+b }` returns 7) | PARITY |
| `PatOr` (`a | b | c`) | ✅ | ❌ | KOVC-MISSING |
| `PatRange` (`0..10`, `0..=10`) | ✅ | ❌ | KOVC-MISSING |
| `PatVariant` (`Enum::Variant(p)`) | ✅ | ❌ | KOVC-MISSING |
| `PatStruct` (`Point { x, y }`) | ✅ | ❌ | KOVC-MISSING |

## 6. Aggregates

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `TupleLit` (`(a, b, c)`) | ✅ | ✅ (Stage 4 iter A landed long ago at kovc.hx:5072 -- AST_TUPLE_LIT allocates N rbp-relative slots via bind_alloc_offset, stores each child, returns slot0 address. K1.F discovery 2026-05-25: codegen present + works end-to-end through bootstrap-self-host; the previous "codegen ud2" matrix entry was stale audit data) | PARITY |
| Tuple field access (`.0`, `.1`) | ✅ | ✅ (Stage 4 iter B at kovc.hx:5024 -- AST_TUPLE_FIELD reads `[rax + p2*8]`, width dispatch on p3 for scalar vs struct fields. K1.F discovery 2026-05-25; verified via test_bootstrap_kovc_tuple_literal_and_field_access_self_host) | PARITY |
| `ArrayLit` (`[1, 2, 3]`) | ✅ | ❌ | KOVC-MISSING |
| `Index` (`a[i, j]`) | ✅ | ⚠️ (AST_INDEX, tag 53 parsed; codegen ud2) | KOVC-MISSING |
| `StructLit` (`Point { x: 1, y: 2 }`) | ✅ | ✅ (Stage 5 Iter D landed long ago: struct lits fold to AST_TUPLE_LIT at parse time, share the tuple-lit codegen at kovc.hx:5072 with rbp-relative slots avoiding nested-aliasing. K1.F-discovery batch 2 2026-05-25: pinned via `test_bootstrap_kovc_struct_literal_and_field_self_host` -- `Pt { x: 5, y: 9 }; p.x + p.y` returns 14) | PARITY |
| Struct field access | ✅ | ⚠️ (Stage 5 Iter C only loads 64-bit; no real layout walk) | KOVC-MISSING |
| `TileLit` (`tile<f32, [N,M], mem>::zeros()`) | ✅ | ❌ | KOVC-MISSING |

## 7. Statements

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `let x = v` (`AST_LET`) | ✅ | ✅ | PARITY |
| `let mut x = v` (`AST_LET_MUT`) | ✅ | ✅ | PARITY |
| `x = v` (`AST_ASSIGN`) | ✅ | ✅ | PARITY |
| `x += v` etc. (compound assign) | ✅ | ❌ (no `+=`/`-=`/`*=`/`/=` etc. tokens in lexer.hx) | KOVC-MISSING |
| `ExprStmt` (`expr;`) | ✅ | ✅ (via AST_SEQ chains) | PARITY |
| `const X: T = expr;` | ✅ | ❌ | KOVC-MISSING |
| `Cast` (`expr as T`) | ✅ | ❌ | KOVC-MISSING |

## 8. Declarations / items

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `fn` (basic) | ✅ | ✅ (`AST_FN_DECL`, 0-6 args) | PARITY |
| `fn` with stack-passed args (> 6) | ✅ (`SYSV_STACK_ARG_*` infra) | ✅ (K1.B, 2026-05-25, commit cb63d78: SysV caller-cleanup pattern; 9 new mov-rsp-disp32 helpers; float args via int regs is documented divergence from x86_64.py) | PARITY |
| Generic `fn<T>` | ✅ (`monomorphize.py`) | ⚠️ (parser tracks gp_tab; no monomorph) | KOVC-MISSING |
| `where` clauses | ✅ | ❌ | KOVC-MISSING |
| `struct Foo { ... }` | ✅ | ⚠️ (parser has struct_table; codegen missing) | KOVC-MISSING |
| Parametric struct `struct<T>` | ✅ (`struct_mono.py`) | ❌ | KOVC-MISSING |
| `enum Foo { A, B(i32) }` | ✅ | ✅ (Stage 6 enum codegen landed long ago: unit variants encoded as tag-only, payload variants destructured via match. K1.F-discovery batch 2 2026-05-25: pinned via `test_bootstrap_kovc_enum_unit_variant_match_self_host` (Color::Green) + `test_bootstrap_kovc_enum_payload_variant_match_self_host` (N::Val(42))) | PARITY |
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
| `print_int(i32)` | ✅ | ✅ (K1.D, 2026-05-25, commits c02ff71 stub + 550329e impl: byte-literal dispatch + 90-byte inline ASCII conversion + write syscall) | PARITY |
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

Rough count from the matrix above, post K0 + K1.B + K1.C
shipping (live count tracked in `scripts/helix_status.py`'s
`K_BOOTSTRAP_PARITY_DONE`):

| Bucket | Count | Notes |
|--------|-------|-------|
| **PARITY** (kovc.hx matches Python) | ~30 rows | +2 since K0: K1.B (stack args > 6) + K1.C (return) |
| **KOVC-MISSING** (Python has it, kovc.hx does not) | ~113 rows | -2 since K0 |
| **PYTHON-MISSING** (kovc.hx has it but Python doesn't) | 0 | |
| **UNKNOWN** (survey uncertain) | 0 | resolved K0 chunk 2 |

The K-bootstrap percentage in the Telegram status update is
computed live from these counts: `30 / 143 ≈ 21%`.

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

## 19. UNKNOWN-row refinement — RESOLVED (K0 chunk 2)

K0 chunk 2 resolved all 5 UNKNOWN rows; all five resolved to
**KOVC-MISSING** (the bootstrap simply does not have the feature
at the lexer level — adding it requires lexer + parser + codegen
work):

- **`f16` literal in `kovc.hx`?** Resolved: type tag 5 is reserved
  in `kovc.hx` line 1177 ("`5 = f16 (Stage 1.5, reserved)`") but no
  AST tag is emitted by `parser.hx` for `f16` literals — they would
  lex as identifiers. KOVC-MISSING.
- **`BoolLit` (`true`/`false`) AST tag in parser.hx?** Resolved:
  `lexer.hx` has no `true`/`false` keyword. The bootstrap uses raw
  `i32` `1`/`0` instead of a boolean type. KOVC-MISSING (entirely
  absent, not just codegen-stubbed).
- **`&&` / `||` short-circuit semantics?** Resolved: `lexer.hx` has
  only `TK_AMP` (`&`, tag 27) and `TK_PIPE` (`|`, tag 28) — single-
  character bitwise tokens. No two-character `&&` / `||`. KOVC-MISSING.
- **Compound assignment `+=` etc.?** Resolved: `lexer.hx` has only
  `TK_EQ` (`=`, tag 15). No compound-assignment tokens. KOVC-MISSING.
- **Reserved AST tag holes 43-49, 54-75, 77-98?** Resolved: a full
  enumeration of `mk_node(N, ...)` calls in `parser.hx` shows the
  ACTIVE tag set is {0-42, 50, 51, 52, 53, 76, 99}. The remaining
  numbers (43-49, 54-75, 77-98) are unused — placeholders for
  future AST shapes. Not a gap in the matrix.

The matrix now has **0 UNKNOWN rows**. Gap list is final-form
(subject to refinement as new Python features land or as new ports
expose surface I missed).

## 20. Active AST-tag enumeration (post-survey reference)

For future K-track chunks, this is the authoritative list of AST
tags that `parser.hx` actually emits (via `mk_node`). Tags not
listed are unused holes in the numbering.

| Tag | Name | Comment |
|-----|------|---------|
| 0 | AST_INT | i32 literal (p1 = value) |
| 1 | AST_VAR | identifier (p1 = name_start, p2 = name_len) |
| 2 | AST_ADD | binary `+` |
| 3 | AST_SUB | binary `-` |
| 4 | AST_MUL | binary `*` |
| 5 | AST_DIV | binary `/` |
| 6 | AST_LT | `<` |
| 7 | AST_IF | `if cond then else` |
| 8 | AST_LET | immutable binding |
| 9 | AST_NEG | unary `-` |
| 10 | AST_WHILE | `while cond body` |
| 11 | AST_ASSIGN | `x = v` |
| 12 | AST_LET_MUT | mutable binding |
| 13 | AST_SEQ | semicolon sequence (left-to-right) |
| 14 | AST_FN_DECL | function declaration |
| 15 | AST_FN_LIST | linked list of fn decls |
| 16 | AST_CALL | function call (p1, p2 = name; p3 = args_head) |
| 17 | AST_ARG | argument cons-cell |
| 18 | AST_PARAM | function parameter (p4 = type_tag) |
| 19 | AST_GT | `>` |
| 20 | AST_EQ | `==` |
| 21 | AST_NE | `!=` |
| 22 | AST_LE | `<=` |
| 23 | AST_GE | `>=` |
| 24 | AST_MOD | binary `%` |
| 25 | AST_STR_LIT | string literal (codegen stub — emits 0) |
| 26 | AST_BNOT | unary `~` (bitwise not) |
| 27 | AST_FLOATLIT | f32 literal (default float type) |
| 28 | AST_BAND | binary `&` (bitwise AND) |
| 29 | AST_BOR | binary `|` (bitwise OR) |
| 30 | AST_BXOR | binary `^` (bitwise XOR) |
| 31 | AST_NOT | unary `!` (logical not, currently same as bitwise on i32) |
| 32 | AST_SHL | `<<` |
| 33 | AST_SHR | `>>` arithmetic right shift |
| 34 | AST_FLOATLIT_F64 | f64 literal |
| 35 | AST_INTLIT_I64 | i64 literal |
| 36 | AST_INTLIT_U32 | u32 literal |
| 37 | AST_INTLIT_U8 | u8 literal |
| 38 | AST_INTLIT_U64 | u64 literal |
| 39 | AST_INTLIT_I8 | i8 literal |
| 40 | AST_INTLIT_I16 | i16 literal |
| 41 | AST_INTLIT_U16 | u16 literal |
| 42 | AST_FLOATLIT_BF16 | bf16 literal (Stage 1.5) |
| 50 | AST_TUPLE_LIT | tuple literal (p1 = arity, p2 = head_idx) |
| 51 | AST_TUPLE_CONS | tuple element cons-cell |
| 52 | AST_TUPLE_FIELD | `.field` access (p2 = field_idx) |
| 53 | AST_INDEX | `a[i]` |
| 76 | AST_STDLIB_FN | stdlib fn reference (parse_program splices these) |
| 99 | AST_ERR | parse/lex error (p1 = trap_id) |

**Reserved (unused) tag holes**: 43-49, 54-75, 77-98. The K-track
can claim these for new AST shapes (bool literal, char literal,
`&&` / `||`, `for`, `loop`, `break`, `continue`, `return`, `match`,
`PatLit` etc., struct literal, enum constructor, cast, const,
`unsafe`, AGI `quote`/`splice`/`modify`, attribute parsing). 50+
slots remain — enough for the entire K1 port work.

**Token-tag holes in `lexer.hx`**: 19, 20, 21, 22, 24, 26. These
are similarly available for new tokens (`true`, `false`, `&&`, `||`,
compound-assign tokens, `for`, `loop`, `match`, etc.).

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

## Appendix A2 — bootstrap-fragility lessons (post K1.A-D)

After shipping K1.A, K1.B, K1.C and attempting K1.D, two
defect patterns have surfaced that any future K-track chunk
should respect:

### Pattern 1: host-parser recursion budget (K1.B audit-fix)

Inserting nested control flow (a `while` loop, a multi-arm
if-cascade) DIRECTLY inside an existing AST_CALL or similar
deeply-nested codegen arm trips Python helixc's parser
recursion budget. Symptom: the bootstrap compiles via Python
but the produced K1 binary miscompiles its own source when
generating K2 (so K2 is broken even though K1 looks OK).

**Mitigation:** extract the new logic into top-level helper
functions. The arm at the deeply-nested site becomes a single
flat call to the helper. K1.B's `emit_stack_args_reverse_copy`
and `emit_load_six_int_args` are the established pattern.

### Pattern 2: install_builtin_names arena-budget invariant

**Investigation 2026-05-25: root cause IDENTIFIED.**

Bisection found that:
- Bare `while i < 152` → `while i < 160` bump alone: PASSES
  self-host. The init-loop count is NOT the culprit.
- Bump + adding 9 `__arena_push` calls for "print_int" bytes
  (with or without the corresponding `__arena_set` slot
  pointer): FAILS self-host.

Conclusion: **adding additional `__arena_push` calls inside
`install_builtin_names` breaks the self-host fixpoint.** Each
extra push advances the arena cursor by 1 i32 slot. The
post-install arena cursor must land at a SPECIFIC position —
likely a downstream consumer (emit_elf_for_ast_to_path or the
str_state setup at slots 7-8) expects the cursor at exactly
where the existing 32 names leave it.

**The fragility surface:** `install_builtin_names` is a
variadic-byte-count factory disguised as a fixed-layout
reserve. Bumping the init-loop count (152 → 160) extends the
ZERO-init region, but adding name-byte pushes extends the
CURSOR position, which has implicit invariants downstream that
nobody documented.

**Mitigations for future K-track chunks needing new builtins:**

- **Option A (preferred): direct byte-literal comparison
  in `try_emit_builtin_call`** that avoids `install_builtin_names`
  entirely. Write a small `is_print_int_name(s, l)` helper that
  checks 9 bytes inline via `__arena_get(s+i)` comparisons.
  Cost: more verbose dispatch. Risk: parser depth for ~9
  inline comparisons (mitigated by a flat `while` loop with
  hardcoded expected bytes per index).
- **Option B**: find the downstream consumer that expects the
  cursor at a specific position; bump it in lockstep with the
  install-bytes addition. Requires reading
  emit_elf_for_ast_to_path carefully.
- **Option C**: defer K1.D entirely. Implement print_int as
  a USER-LAND helper in the language's stdlib (not a backend
  builtin). After K2 (parity harness) lands and the
  reference-oracle comparison surfaces the position-sensitive
  invariant, the right fix becomes obvious.

Preferred next attempt: option A. The byte-literal approach
doesn't touch install_builtin_names at all.

### Pattern 3 (suspected, untested): the parse_primary IDENT
sub-cascade

K1.C's first attempt added a new arm in parse_primary's IDENT
keyword cascade but adjusted the closing brace count at the
WRONG location (the outer cascade closer at parser.hx:3848,
not the IDENT sub-cascade closer at parser.hx:3722). Counted
+2 instead of the correct +1.

**Mitigation:** carefully count brace pairs locally at the
insertion site, not at the file-end closer pile. Each `} else
{ if X { ... } else { ... }` pattern is +2 opens, +1 close at
the site, requiring +1 close downstream (not +2). The K1.C
deadcode + wireup split lets the wireup be a tiny isolated
edit with verifiable arithmetic.

These patterns will recur on K1.E-J chunks. The discipline:
write the chunk small, run the bootstrap-kovc test slice
(`pytest -k bootstrap_kovc`, ~5-min wall) before commit, REVERT
clean if it breaks, document the lesson.

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
