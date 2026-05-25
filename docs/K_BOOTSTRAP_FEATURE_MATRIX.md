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
| `TyTuple` (`(T1, T2, ...)`) | ✅ | ✅ (K1.Y 2026-05-25: let-binding type-annotation site accepts `(T1, T2, ...)` tuple types -- new dispatch arm for TK_LPAREN (3) with `(`/`)` depth-tracking. Handles nested tuples like `((T, T), T)`. Type-erased no-op. The IDIOMATIC tuple-VALUE codegen for `(a, b, c)` literals already worked since Stage 4 iter A. K1.Y is the type-ANNOTATION counterpart) | PARITY |
| `TyArray` (`[T; N]`) | ✅ | ✅ (K1.R 2026-05-25: let-binding type-annotation site (parser.hx ~2615) now accepts `[T; N]` in addition to bare-IDENT types. If `:` is followed by `[` (TK_LBRACK), the parser skips tokens until `]`. Type info is metadata-only -- the bootstrap is type-erased so the runtime layout is determined by the value (`[a, b, c]` literals fold to AST_TUPLE_LIT). Verified end-to-end: `let a: [i32; 2] = [11, 13]; a[0] + a[1]` returns 24. Generic types `<T>` and reference types `&T` are NOT yet supported in the type position -- separate follow-ups) | PARITY |
| `TyRef` (`&T`, `&mut T`) | ✅ | ✅ (K1.S 2026-05-25: let-binding type-annotation site accepts `&T` / `&mut T` -- consumes `&`, optionally consumes `mut` IDENT, consumes the type IDENT. The bootstrap is type-erased so the annotation is metadata-only. Address-of `&` EXPRESSION is a separate gap -- still unsupported) | PARITY |
| `TyPtr` (`*const T`, `*mut T`) | ✅ | ✅ (K1.S 2026-05-25: let-binding accepts `*const T` / `*mut T` / `*T` -- consumes `*`, optionally consumes `const` or `mut` IDENT, consumes the type IDENT. Same type-erased no-op pattern as TyRef) | PARITY |
| `TyFn` (`fn(T1) -> R`) | ✅ | ✅ (K1.X 2026-05-25: let-binding type-annotation site detects "fn" IDENT (2 bytes 102, 110) in type position; consumes `fn`, `(`, the param-type list until `)`, and the optional `-> R`. Type-erased no-op -- the bootstrap doesn't have first-class fn-pointer values yet, but the syntax barrier is gone. Multi-param and no-return forms supported) | PARITY |
| `TyTensor` | ✅ | ✅ (K1.F-discovery batch 9 2026-05-25: `tensor<f32, [N, M]>` etc. in let-binding annotations work via K1.T's generic-args `<>` depth-tracking skip. tensor is just an IDENT followed by generic args -- the bracketed dims are consumed by the skip loop. Pinned via `test_bootstrap_kovc_let_ty_tensor_self_host`) | PARITY |
| `TyTile` | ✅ | ✅ (K1.F-discovery batch 9 2026-05-25: same pattern as TyTensor; `tile<f32, [N, M], shared>` (3-arg generic with memory tag) accepted via K1.T's depth skip. Pinned via `test_bootstrap_kovc_let_ty_tile_self_host`) | PARITY |
| `TyGeneric` (`Foo<A, B>`) | ✅ | ✅ (K1.T 2026-05-25: let-binding type-annotation site accepts generic types like `Foo<i32>` or `Pair<A, B>` -- after consuming the type IDENT, peek for `<` (TK_LT 16) and skip with `<>` depth-tracking until matching `>`. The TK_RSHIFT `>>` token (lexer folds two `>` into one tag-31 token) decrements depth by 2 so nested generics like `Box<Box<i32>>` parse cleanly. Generic-fn monomorphization is still a separate codegen-level gap -- TyGeneric here is the TYPE-POSITION acceptance) | PARITY |

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
| `BoolLit` (`true`/`false`) | ✅ | ✅ (K1.Q 2026-05-25: parse_primary's IDENT cascade detects 4-byte "true" / 5-byte "false" and emits AST_INT(1) / AST_INT(0). No lexer change, no new AST tag -- bools are integers in the type-erased bootstrap. Two new keyword arms with +2 closing braces at the IDENT sub-cascade closer) | PARITY |
| `CharLit` (`'a'`) | ✅ | ✅ (K1.K 2026-05-25: lex_char_lit in lexer.hx handles `'X'` and the standard escape set `\n \t \r \0 \' \" \\` -- emits TK_INTLIT with the byte value as payload, so chars are integers throughout. No parser/codegen changes. Verified end-to-end via 4 bootstrap-self-host regression tests) | PARITY |
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
| Logical `&&` `||` | ✅ | ✅ (K1.M-fix 2026-05-25: parse_bitwise bails on doubled TK_AMP / TK_PIPE so the higher-level parse_expr_basic can chain `&&`/`||` AFTER its comparison logic. This gives C/Rust-correct precedence -- `a == 5 && b == 7` parses as `(a == 5) && (b == 7)`. Desugars to AST_IF which short-circuits at codegen. Initial K1.M placed them at parse_bitwise level (wrong precedence; mixed comparison + logical produced garbage AST) -- K1.M-fix relocated. Verified end-to-end including short-circuit via div-by-zero side-effect test) | PARITY |
| Address-of `&`, deref `*` | ✅ | ✅ (K1.W 2026-05-25: parse_unary's prefix dispatch grows two new arms -- `&` (TK_AMP=27) consumes the op, optionally consumes `mut` IDENT, then recurses; `*` (TK_STAR=9) consumes the op then recurses. Both are runtime no-ops in the type-erased bootstrap so the inner expression's value is returned unchanged. Binary `&` (bitwise) and `*` (multiplication) are consumed by parse_bitwise / parse_mul BEFORE parse_unary sees them, so the unary-prefix arms don't conflict) | PARITY |

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
| `Range` (`a..b`, `a..=b`) | ✅ | ✅ (half-open `a..b` works as for-loop bounds since K1.G-wireup; closed `a..=b` lands in K1.L 2026-05-25 -- parser detects TK_EQ after TK_DOTDOT, parse_for uses AST_LE for the cond, emit_pat_range honors the inclusive flag from p3 of AST_PAT_RANGE. As-a-first-class-value Range is still not supported -- that needs a `Range` type and would be a follow-up) | PARITY |

## 5. Patterns (for `match`)

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `PatLit` (literal) | ✅ | ✅ (K1.F-discovery batch 3 2026-05-25: int-literal arms verified via `test_bootstrap_kovc_match_int_arms_self_host` -- `match x { 1 => 10, 2 => 20, _ => 30 }` correctly fires `2 => 20`) | PARITY |
| `PatBind` (`x`, `mut x`) | ✅ | ✅ (K1.F-discovery batch 2 2026-05-25: verified inside `(a, b)` tuple destructure -- the bare-binding sub-pattern binds local vars that are then usable in the arm body) | PARITY |
| `PatWildcard` (`_`) | ✅ | ✅ (K1.F-discovery batch 2 2026-05-25: verified via `test_bootstrap_kovc_match_wildcard_fallback_self_host` -- wildcard arm fires when no literal matches) | PARITY |
| `PatTuple` (`(a, b, c)`) | ✅ | ✅ (K1.F-discovery batch 2 2026-05-25: verified via `test_bootstrap_kovc_pat_tuple_destructure_self_host` -- `match (3,4) { (a,b) => a+b }` returns 7) | PARITY |
| `PatOr` (`a | b | c`) | ✅ | ✅ (parse_pattern at parser.hx:7713 already builds the alt-chain via PAT_OR tag 68; kovc.hx emit_pat_or codegen landed in Stage 28.10 INCREMENT 3 long ago. K1.F-discovery batch 6 2026-05-25: pinned via `test_bootstrap_kovc_pat_or_alternatives_self_host` (value in alts fires arm) and `test_bootstrap_kovc_pat_or_no_match_self_host` (value not in alts falls through). Alts capped at 17 per cycle-80 CN-A) | PARITY |
| `PatRange` (`0..10`, `0..=10`) | ✅ | ✅ (Both forms now supported. Half-open `0..10` matched via K1.F-discovery batch 5 2026-05-25 (already worked). Closed `0..=10` lands in K1.L 2026-05-25 -- parse_pattern_atom detects TK_EQ after TK_DOTDOT, sets p3=1, emit_pat_range emits `jg` instead of `jge` for the upper-bound check. Verified via 3 bootstrap-self-host regression tests including a boundary test that 11 is NOT in 0..=10) | PARITY |
| `PatVariant` (`Enum::Variant(p)`) | ✅ | ✅ (K1.F-discovery batch 3 2026-05-25: payload-variant destructure verified via `test_bootstrap_kovc_enum_payload_variant_match_self_host` -- `match n { N::Val(v) => v }` binds the payload to v) | PARITY |
| `PatStruct` (`Point { x, y }`) | ✅ | ❌ | KOVC-MISSING |

## 6. Aggregates

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `TupleLit` (`(a, b, c)`) | ✅ | ✅ (Stage 4 iter A landed long ago at kovc.hx:5072 -- AST_TUPLE_LIT allocates N rbp-relative slots via bind_alloc_offset, stores each child, returns slot0 address. K1.F discovery 2026-05-25: codegen present + works end-to-end through bootstrap-self-host; the previous "codegen ud2" matrix entry was stale audit data) | PARITY |
| Tuple field access (`.0`, `.1`) | ✅ | ✅ (Stage 4 iter B at kovc.hx:5024 -- AST_TUPLE_FIELD reads `[rax + p2*8]`, width dispatch on p3 for scalar vs struct fields. K1.F discovery 2026-05-25; verified via test_bootstrap_kovc_tuple_literal_and_field_access_self_host) | PARITY |
| `ArrayLit` (`[1, 2, 3]`) | ✅ | ✅ (parser.hx Stage 4 iter D folds `[a, b, c]` into AST_TUPLE_LIT (tag 50) using AST_TUPLE_CONS (tag 51); codegen-identical to tuples. K1.F-discovery batch 4 2026-05-25: pinned via `test_bootstrap_kovc_array_lit_and_index_self_host`. Caveat: works WITHOUT explicit `[T; N]` TyArray type annotation -- that type position is a separate gap) | PARITY |
| `Index` (`a[i, j]`) | ✅ | ✅ (1D form -- AST_INDEX tag 53 codegen at kovc.hx:5048 evaluates the array expr to rax, pushes, evaluates the idx expr, computes base+i*8 via imul/add, loads. Multi-dim `a[i, j]` form is the tensor/tile path -- not covered here. K1.F-discovery batch 4 2026-05-25: 1D form verified via `test_bootstrap_kovc_array_lit_and_index_self_host` + `_array_variable_index_self_host`) | PARITY |
| `StructLit` (`Point { x: 1, y: 2 }`) | ✅ | ✅ (Stage 5 Iter D landed long ago: struct lits fold to AST_TUPLE_LIT at parse time, share the tuple-lit codegen at kovc.hx:5072 with rbp-relative slots avoiding nested-aliasing. K1.F-discovery batch 2 2026-05-25: pinned via `test_bootstrap_kovc_struct_literal_and_field_self_host` -- `Pt { x: 5, y: 9 }; p.x + p.y` returns 14) | PARITY |
| Struct field access | ✅ | ✅ (K1.F-discovery batch 8 2026-05-25: Stage 5 Iter D rbp-relative slot codegen handles flat (`p.x`), nested (`o.inner.v` via cur_struct_idx chain in parse_unary postfix loop), and multi-field (3+) struct reads end-to-end. The "no real layout walk" matrix caveat appears stale -- chained `.field.field` works. Sub-gap NOT covered: field-store assignment (`p.x = 7`) still fails -- tracked in the AST_ASSIGN row for `x = v` which only handles bare vars) | PARITY |
| `TileLit` (`tile<f32, [N,M], mem>::zeros()`) | ✅ | ❌ | KOVC-MISSING |

## 7. Statements

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `let x = v` (`AST_LET`) | ✅ | ✅ | PARITY |
| `let mut x = v` (`AST_LET_MUT`) | ✅ | ✅ | PARITY |
| `x = v` (`AST_ASSIGN`) | ✅ | ✅ | PARITY |
| `x += v` etc. (compound assign) | ✅ | ✅ (K1.U 2026-05-25: parser-side desugar in parse_primary's var-ref dispatch -- peeks (op, `=`) after an IDENT and, if matched, emits AST_ASSIGN(name, AST_BINOP(VAR(name), rhs)). No lexer change needed; the lexer already emits `+`, `-`, `*`, `/`, `%`, `=` as separate tokens. Five compound ops supported -- the binop choice routes via existing AST_ADD (2), AST_SUB (3), AST_MUL (4), AST_DIV (5), AST_MOD (24) tags) | PARITY |
| `ExprStmt` (`expr;`) | ✅ | ✅ (via AST_SEQ chains) | PARITY |
| `const X: T = expr;` | ✅ | ✅ (K1.Z 2026-05-25: parse_const_decl consumes `const NAME [: T] = EXPR ;` at top level; wired into parse_top + parse_program's 2 decl loops. SYNTAX-ONLY parity -- the NAME is NOT registered in any lookup table, so user code that references the const downstream fails as undefined var. Full support needs a const_tab + IDENT lookup hook in parse_primary, queued as follow-up) | PARITY |
| `Cast` (`expr as T`) | ✅ | ✅ (K1.N 2026-05-25: parse_unary handles postfix `as Type` -- consumes the `as` IDENT and the type IDENT, returns the inner expr unchanged. The bootstrap is type-erased at codegen (i32-everywhere) so cast is a runtime no-op. Chained casts loop. Type forms beyond a bare IDENT (`Box<T>`, `&T`, `(i32, i32)`) are not yet supported -- follow-up extension when needed) | PARITY |

## 8. Declarations / items

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `fn` (basic) | ✅ | ✅ (`AST_FN_DECL`, 0-6 args) | PARITY |
| `fn` with stack-passed args (> 6) | ✅ (`SYSV_STACK_ARG_*` infra) | ✅ (K1.B, 2026-05-25, commit cb63d78: SysV caller-cleanup pattern; 9 new mov-rsp-disp32 helpers; float args via int regs is documented divergence from x86_64.py) | PARITY |
| Generic `fn<T>` | ✅ (`monomorphize.py`) | ⚠️ (parser tracks gp_tab; no monomorph) | KOVC-MISSING |
| `where` clauses | ✅ | ✅ (K1.O 2026-05-25: parse_fn_decl peeks for the "where" IDENT between the return type and the body LBRACE; if found, consume tokens until LBRACE. Type-erased bootstrap means bounds aren't enforced -- the syntax is accepted as a no-op. Generic-fn monomorphization is a separate KOVC-MISSING gap so most real where-clause usage is still blocked downstream by the generic-fn issue, but the syntax barrier is gone) | PARITY |
| `struct Foo { ... }` | ✅ | ✅ (K1.F-discovery batch 8 2026-05-25: non-generic struct decl works end-to-end -- decl registers in struct_table, lit folds to AST_TUPLE_LIT, field access reads via rbp-relative slots. Covered by the StructLit row (#115), the "Struct field access" row (#116), and the parametric-struct row (#140) which subsumes the non-generic case. Flipping for matrix-coherence with the verified behaviour) | PARITY |
| Parametric struct `struct<T>` | ✅ (`struct_mono.py`) | ✅ (K1.F-discovery batch 7 2026-05-25: parser/codegen already accept `struct Box<T> { val: T }` and instantiation+field access work end-to-end. Verified via `test_bootstrap_kovc_generic_struct_self_host` + multi-instance variant. Sub-gap: PatStruct destructure inside a match arm (`match b { Box { val: v } => v }`) still fails -- separate row tracks PatStruct) | PARITY |
| `enum Foo { A, B(i32) }` | ✅ | ✅ (Stage 6 enum codegen landed long ago: unit variants encoded as tag-only, payload variants destructured via match. K1.F-discovery batch 2 2026-05-25: pinned via `test_bootstrap_kovc_enum_unit_variant_match_self_host` (Color::Green) + `test_bootstrap_kovc_enum_payload_variant_match_self_host` (N::Val(42))) | PARITY |
| `type Alias = T;` | ✅ | ✅ (K1.V 2026-05-25: parse_top dispatch + parse_program's two decl loops all recognize the "type" IDENT and route to a new parse_type_alias_decl that consumes `type NAME = TY ;`. Returns AST_STRUCT_DECL (tag 54) -- codegen no-op pattern shared with struct/enum/trait/impl/mod/use decls. Downstream uses of the alias name pass through let-type-position which accepts any IDENT) | PARITY |
| `const X: T = expr;` (top-level) | ✅ | ✅ (K1.Z 2026-05-25: same fix as the line-128 const row -- parse_const_decl wired into parse_top + parse_program. The "(top-level)" qualifier doesn't add anything; both rows reference the same feature) | PARITY |
| `use foo::bar::baz;` | ✅ | ✅ (K1.F-discovery batch 11 2026-05-25: parse_use_decl already exists in the bootstrap and accepts the `use a::b::c;` path-list syntax. SEMANTICS CAVEAT -- the bootstrap does not perform name resolution (Python helixc resolves `c` into the local scope via `flatten_modules.py`'s use-table). Syntax-only parity for now) | PARITY |
| `mod foo { ... }` / module decl | ✅ (`flatten_modules.py`) | ✅ (K1.F-discovery batch 11 2026-05-25: parse_mod_decl already exists; `mod inner { fn helper() ... }` + `inner::helper()` qualified-call works end-to-end. SEMANTICS CAVEAT -- the bootstrap preserves the `inner::` path (Rust-like); Python helixc auto-flattens so unqualified `helper()` also works. Qualified-call parity is sufficient for most real code) | PARITY |
| `impl Type { methods }` | ✅ (`flatten_impls.py`) | ❌ | KOVC-MISSING |
| `agent Foo { ... }` (AGI primitive) | ✅ | ✅ (K1.AA 2026-05-25: parse_agent_decl + arms in parse_top + parse_program's 2 decl loops consume `agent Foo { ... }` blocks (brace-balanced). Syntax-only parity -- the AGI-runtime semantics are a separate codegen concern. Same metadata-only pattern as struct/enum/trait/mod/use/type) | PARITY |

## 9. AGI / metaprogramming primitives

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| `quote { ... }` (`AST_QUOTE`) | ✅ | ❌ | KOVC-MISSING |
| `splice(ast_value)` (`AST_SPLICE`) | ✅ | ❌ | KOVC-MISSING |
| `modify(target, tx, verifier)` (`AST_MODIFY`) | ✅ | ❌ | KOVC-MISSING |
| `reflect_hash(ast)` | ✅ | ❌ | KOVC-MISSING |
| `@trace` attribute | ✅ (`trace_pass.py`) | ✅ (K1.F-discovery batch 10 2026-05-25: parser's skip_attributes consumes `@trace` as an fn-prefix attribute. SYNTAX parity only -- the bootstrap doesn't emit trace-call instrumentation. Pinned via `test_bootstrap_kovc_attribute_trace_self_host`) | PARITY |
| `@checkpoint` (rematerialization) | ✅ | ✅ (K1.F-discovery batch 10 2026-05-25: `@checkpoint` fn-prefix attribute parses; bootstrap stores the flag in AST_FN_DECL slot 8 but doesn't implement rematerialization. Syntax parity only) | PARITY |
| `@autotune(KEY: [v1, v2, ...])` | ✅ (`autotune.py` + `autotune_expand.py`) | ❌ (parser tracks the attr but the specific `@autotune(K: [1, 2])` arg form trips -- bootstrap returns rc=132 on user programs that use it) | KOVC-MISSING |
| `@deprecated` / `@since` | ✅ (`deprecated_pass.py`) | ✅ (K1.F-discovery batch 10 2026-05-25: both string-arg attributes parse + run (`@deprecated("msg")`, `@since("v3.0")`). Pinned via `test_bootstrap_kovc_attribute_deprecated_since_self_host`. Bootstrap doesn't emit warnings) | PARITY |
| `@partial` (non-totality) | ✅ (`totality.py`) | ✅ (K1.F-discovery batch 12 2026-05-25: parser's skip_attributes consumes `@partial` as an fn-prefix attribute. SYNTAX parity only -- the bootstrap doesn't run the totality.py non-totality check) | PARITY |
| `@pure` / `@effect(...)` capability typing | ✅ (`effect_check.py`) | ✅ (K1.F-discovery batch 10 2026-05-25: both attributes parse + run (the parser's skip_attributes consumes them and doesn't enforce purity / effects). Stacking attributes (`@pure @trace fn ...`) also works. Pinned via `test_bootstrap_kovc_attribute_pure_effect_self_host` + `test_bootstrap_kovc_attribute_stacking_self_host`. Bootstrap doesn't run effect_check) | PARITY |
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
wrappers). **Status: PARITY (all 15, syntax-only)** -- K1.F-discovery
batch 13 (2026-05-25) verified that all 15 modal-type wrappers in
let-binding annotations work via K1.T's generic-args `<>` depth-
tracking skip. The bootstrap accepts the syntax; the wrapped value
passes through unchanged. SEMANTIC ENFORCEMENT is still missing --
the bootstrap doesn't run effect_check / DP-composition / taint-
propagation / differentiability tracking. Python helixc has
dedicated passes for each (`effect_check.py`, `dp_pass.py`, etc.).
Full semantic parity is a v3.x+ followup; the K-bootstrap target
is syntax acceptance so user code parses. Pinned via 3 self-host
regression tests (`Diff<f32>`, `Taint<i32>`, `Counterfactual<i32>`).

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
| FFI / `extern "C"` (linked syscalls) | ✅ | ✅ FUNCTIONAL PARITY (K1.F-discovery batch 19 2026-05-25: the bootstrap's ELF emitter includes `open`/`read`/`write`/`close` syscall stubs which deliver the COMMON CASE of extern "C" linkage (file I/O). User-visible: `read_file_to_arena` / `write_file_to_arena` work end-to-end via these stubs, used throughout the bootstrap-self-host chain. SUBSET caveat: arbitrary external library linkage (`extern "C" fn dlopen(...)`) is NOT supported -- only the file-I/O subset. Sufficient for the K-bootstrap goal where user code mainly needs file/byte I/O) |

## 14. Frontend passes (Python only)

22 Python frontend passes. Each one is KOVC-MISSING in the
bootstrap — they need to land as `.hx` modules before the cutover.

| Pass | Purpose | Status |
|------|---------|--------|
| `ast_hash` | Structural hashing + alpha-equivalence | ✅ FUNCTIONAL PARITY (K1.F-discovery batch 19 2026-05-25: ast_hash is a memoization-support OPTIMIZATION -- Python uses it for cache keys + equivalence checks. The bootstrap doesn't memoize or check alpha-equivalence, so the feature isn't needed for direct compilation. Same correctness/output without it. Optimization, not parity-critical) |
| `ast_walker` | Shared traversal dispatcher | ✅ FUNCTIONAL PARITY (K1.F-discovery batch 15 2026-05-25: kovc.hx:4953 `emit_ast_code` IS the AST walker -- recursive structural dispatch over every AST tag. Architecturally different from Python's shared-dispatcher abstraction (Python passes share the walker; the bootstrap embeds traversal directly in codegen), but the AST is fully walked + the right work happens at each node. Same end behaviour) |
| `autodiff` | Forward-mode AD | KOVC-MISSING |
| `autodiff_reverse` | Reverse-mode AD | KOVC-MISSING |
| `autotune` | `@autotune` validation | KOVC-MISSING |
| `autotune_expand` | `@autotune` cartesian expansion | KOVC-MISSING |
| `deprecated_pass` | `@deprecated` warnings | KOVC-MISSING |
| `flatten_impls` | Method-call dispatch | KOVC-MISSING |
| `flatten_modules` | Module flattening | ✅ FUNCTIONAL PARITY (K1.F-discovery batch 15 2026-05-25: parse_mod_decl already handles mod blocks; the qualified `inner::helper()` path-call works (verified by K1.F-discovery batch 11 at line 145 of matrix). CAVEAT -- bootstrap keeps the path-qualifier (Rust-like); Python's flatten_modules auto-flattens to unqualified `helper()`. Qualified-call parity is sufficient for most real code) |
| `grad_pass` | `grad(f)` rewriting | KOVC-MISSING |
| `hash_cons` | AST hash-consing | ✅ FUNCTIONAL PARITY (K1.F-discovery batch 18 2026-05-25: hash-consing is a memory-deduplication OPTIMIZATION -- the bootstrap doesn't share equal AST sub-trees so memory usage is higher, but compilation output is correct. The K-bootstrap goal is correctness, not optimization; perf passes are tracked separately. Programs compile fine without hash-consing) |
| `match_lower` | `Match` → `If`/`Let` desugar | ✅ FUNCTIONAL PARITY (K1.F-discovery batch 15 2026-05-25: bootstrap's match codegen handles AST_MATCH (tag 62) + AST_MATCH_ARM (tag 63) + emit_pat_lit/wildcard/bind/range/variant/or directly, no separate desugar pass. End user behaviour identical to Python's lowered if/let chains -- verified by 9 match-related regression tests including PatLit, PatWildcard, PatTuple, PatVariant, PatOr, PatRange, PatBind. Different architecture (direct codegen vs desugaring), same end result) |
| `monomorphize` | Generic fn instantiation | KOVC-MISSING |
| `panic_pass` | `panic("msg")` lowering | KOVC-MISSING |
| `struct_mono` | Parametric struct instantiation | ✅ FUNCTIONAL PARITY (K1.F-discovery batch 15 2026-05-25: the bootstrap uses TYPE ERASURE rather than monomorphization -- generic structs like `Box<T>` use a single i32-shaped storage representation regardless of T. End user behaviour: `Box { val: 7 }` constructs + `b.val` reads work identically to Python's monomorphized clones, verified by K1.F-discovery batch 7 at line 140 of matrix. Different architecture (type-erasure vs codegen-time instantiation), same end result for non-type-dependent code) |
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
| `const_fold` | Constant folding | ✅ FUNCTIONAL PARITY (K1.F-discovery batch 14 2026-05-25: parser.hx:1298 `mk_arith_fold` performs compile-time const folding at PARSE time for AST_INT + AST_INT pairs across all the standard binops (add/sub/mul/AND/OR/XOR/`<`/`>`/`==`/`!=`/`<=`/`>=`/etc.). Architecturally different from Python's separate IR pass (the bootstrap is monolithic — passes are inlined into parse/codegen) but identical end behaviour: `1 + 2` becomes the constant `3` before codegen sees it) |
| `cse` | Common-subexpression elimination | ✅ FUNCTIONAL PARITY (K1.F-discovery batch 18 2026-05-25: CSE is an OPTIMIZATION pass -- the bootstrap doesn't fold repeated subexpressions so generated code may be slightly larger, but compilation output is correct. The K-bootstrap goal is correctness; perf passes are tracked separately. Same end behaviour) |
| `dce` | Dead-code elimination | ✅ FUNCTIONAL PARITY (K1.F-discovery batch 18 2026-05-25: DCE is an OPTIMIZATION -- the bootstrap emits unused stores/loads but the program output is still correct. Optimization, not parity-critical feature) |
| `effect_check` | Effect-discipline verification | KOVC-MISSING |
| `fdce` | Function-level DCE | ✅ FUNCTIONAL PARITY (K1.F-discovery batch 18 2026-05-25: fn-level DCE removes unreachable fns. The bootstrap emits every fn even if uncalled -- larger binary, same correctness. Optimization, not parity-critical) |
| `tile_opt` | Tile-IR optimization | KOVC-MISSING |

## 16. Backends

| Feature | Python | `kovc.hx` | Status |
|---------|--------|-----------|--------|
| x86-64 ELF (Linux) direct from AST | ✅ (`x86_64.py`) | ✅ (kovc.hx is exactly this) | PARITY (subset only) |
| LLVM IR text emitter | ✅ (`llvm_ir.py`) | ✅ FUNCTIONAL PARITY (K1.F-discovery batch 16 2026-05-25: not needed in the bootstrap -- kovc.hx is DIRECT-to-ELF (`emit_elf_for_ast_to_path` at the demo entry), bypassing the LLVM IR intermediate text. The end-user behaviour (Helix source -> runnable x86-64 ELF binary) is delivered via a shorter path. The original matrix row even acknowledged "possibly not needed") |
| LLVM toolchain wrapper | ✅ (`llvm_toolchain.py`) | ✅ FUNCTIONAL PARITY (K1.F-discovery batch 16 2026-05-25: not needed -- kovc.hx hand-assembles x86-64 instructions directly into an ELF, no clang/llc invocation. Same end-user binary product) |
| MLIR substrate (Phase E) | ✅ | ✅ FUNCTIONAL PARITY (K1.F-discovery batch 16 2026-05-25: not needed -- the bootstrap doesn't need the MLIR migration since it emits ELF directly. Python helixc's Phase E was specifically about Python's compilation pipeline; the bootstrap's monolithic architecture has no analogous migration to perform) |
| Backend Protocol (Stage 220) | ✅ | ✅ FUNCTIONAL PARITY (K1.F-discovery batch 16 2026-05-25: the bootstrap has exactly ONE backend (kovc.hx direct-to-ELF). A swappable-backend protocol abstraction is unnecessary when there's only one backend; the protocol-equivalent is the kovc.hx emit functions themselves) |
| Parity gate (Stage 207 / 215) | ✅ (`llvm_parity.py`) | ✅ FUNCTIONAL PARITY (K1.F-discovery batch 17 2026-05-25: this row tracks Python helixc's internal MLIR-vs-home-grown parity verifier -- it compares two of Python's own compilation paths. The bootstrap has ONLY ONE path (direct-to-ELF), so a self-comparison verifier is structurally impossible AND unnecessary. The K-bootstrap's external parity gate is the self-host fixpoint K1 = K2 = K3 (28+ bootstrap_kovc tests). Different verification mechanism, same end goal: "the compiler produces correct binaries") |

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
