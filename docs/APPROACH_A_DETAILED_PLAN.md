# Approach A: Detailed Implementation Plan — Stages 6-30

> Historical bootstrap plan: this document is superseded for live stage
> tracking. Use `docs/ROADMAP.md` and `docs/stage35-progress-2026-05-15.md` for
> the current Stage 35 status.

**Purpose:** Historical per-stage implementation specifics for the Helix bootstrap port. It is no longer the single source of truth for the live stage.

**Status snapshot (2026-05-08):** Stages 1-5 (Iter A complete, Iter B steps 1-4 complete) landed. Sequence resumes at Stage 5 Iter B step 5 (named field access), then Stage 6.

**Reference baseline:**
- Phase 0 = current bootstrap, single-fn `main()` legacy + multi-fn AST_FN_LIST.
- Codegen lives in `helixc/bootstrap/kovc.hx` (~4838 LOC), parser in `helixc/bootstrap/parser.hx` (~1383 LOC).
- Python reference: `helixc/frontend/*.py`, `helixc/ir/*.py`, `helixc/backend/*.py` (~14k LOC).

---

## Conventions used throughout this plan

### AST tag namespace (continue from 54)

| Tag | Name | Stage | Status |
|----:|------|------:|-------|
| 0  | AST_INTLIT (i32)        | 0 | done |
| 1  | AST_VAR                 | 0 | done |
| 2-5 | ADD/SUB/MUL/DIV         | 0 | done |
| 6  | AST_LT                  | 0 | done |
| 7  | AST_IF                  | 0 | done |
| 8  | AST_LET                 | 0 | done |
| 9  | AST_NEG                 | 0 | done |
| 10 | AST_WHILE               | 0 | done |
| 11 | AST_ASSIGN              | 0 | done |
| 12 | AST_LET_MUT             | 0 | done |
| 13 | AST_SEQ                 | 0 | done |
| 14 | AST_FN_DECL             | 0 | done |
| 15 | AST_FN_LIST             | 0 | done |
| 16 | AST_CALL                | 0 | done |
| 17 | AST_ARG                 | 0 | done |
| 18 | AST_PARAM               | 0 | done |
| 19-23 | GT/EQ/NE/LE/GE       | 0 | done |
| 24 | AST_MOD                 | 0 | done |
| 25 | AST_STR_LIT             | 0 | done |
| 26 | AST_BNOT                | 0 | done |
| 27 | AST_FLOATLIT (f32)      | 0 | done |
| 28-30 | BAND/BOR/BXOR        | 0 | done |
| 31 | AST_NOT                 | 0 | done |
| 32-33 | SHL/SHR              | 0 | done |
| 34 | AST_FLOATLIT_F64        | 0 | done |
| 35 | AST_INTLIT_I64          | 1 | done |
| 36-41 | INTLIT_U32/U8/U64/I8/I16/U16 | 2 | done |
| 42 | AST_FLOATLIT_BF16       | 1.5 | done |
| 50 | AST_TUPLE_LIT           | 4 | done |
| 51 | AST_TUPLE_CONS (linked) | 4 | done |
| 52 | AST_TUPLE_FIELD         | 4 | done |
| 53 | AST_INDEX               | 4 | done |
| 54 | AST_STRUCT_DECL         | 5 | done |
| 55 | AST_STRUCT_PARAM        | 5C | reserved |
| 56 | AST_STRUCT_FIELD        | 5B | reserved |
| 57 | AST_ENUM_DECL           | 6 | new |
| 58 | AST_ENUM_VARIANT_DECL   | 6 | new (linked-list element) |
| 59 | AST_ENUM_CONSTRUCT      | 6 | new (e.g. `Maybe::Some(42)`) |
| 60 | AST_ENUM_DISCRIMINANT   | 6 | new (read tag) |
| 61 | AST_ENUM_PAYLOAD        | 6 | new (read payload n) |
| 62 | AST_MATCH               | 7 | new |
| 63 | AST_MATCH_ARM           | 7 | new (linked list) |
| 64 | AST_PAT_LIT             | 7 | new |
| 65 | AST_PAT_BIND            | 7 | new |
| 66 | AST_PAT_WILDCARD        | 7 | new |
| 67 | AST_PAT_RANGE           | 7 | new |
| 68 | AST_PAT_OR              | 7 | new |
| 69 | AST_PAT_VARIANT         | 7 | new (`Some(x)`) |
| 70 | AST_PAT_TUPLE           | 7 | new |
| 71 | AST_GENERIC_PARAM       | 8 | new (TyVar, e.g. `T`) |
| 72 | AST_TURBOFISH           | 8 | new (`f::<i32>(x)`) |
| 73 | AST_TRAIT_DECL          | 8.5 | new |
| 74 | AST_IMPL_BLOCK          | 8.5 | new |
| 75 | AST_TRAIT_BOUND         | 8.5 | new |
| 76 | AST_CLOSURE_LIT         | 9 | new (`|x| x + 1`) |
| 77 | AST_CLOSURE_CALL        | 9 | new (when callee is closure-typed value) |
| 78 | AST_MOD_DECL            | 10 | new (`mod foo { ... }`) |
| 79 | AST_USE_DECL            | 10 | new (`use foo::bar;`) |
| 80 | AST_PATH_EXPR           | 10 | new (`foo::bar`) |
| 81 | AST_QUOTE               | 11 | new |
| 82 | AST_SPLICE              | 11 | new |
| 83 | AST_MODIFY              | 11 | new |
| 84 | AST_VERIFIER_BLOCK      | 11 | new |
| 85 | AST_AD_FORWARD          | 12 | new (`grad(f)` after grad_pass) |
| 86 | AST_AD_TANGENT_PAIR     | 12 | new (dual-number AST) |
| 87 | AST_AD_USER_CALL_FWD    | 13 | new (chain-rule call) |
| 88 | AST_AD_REVERSE          | 14 | new |
| 89 | AST_AD_ADJOINT_BUCKET   | 14 | new |
| 90 | AST_AD_CHECKPOINT       | 14.5 | new (`@checkpoint fn`) |
| 91 | AST_TILE_LIT            | 15 | new |
| 92 | AST_TENSOR_LIT          | 15 | new |
| 93 | AST_TILE_LOAD           | 15 | new |
| 94 | AST_TILE_STORE          | 15 | new |
| 95 | AST_TILE_MATMUL         | 15 | new |
| 96 | AST_KERNEL_DECL         | 16 | new (`@kernel fn`) |
| 97 | AST_PTX_BLOCK           | 16 | new |
| 98 | AST_FFI_EXTERN_DECL     | 16.5 | new (`extern "C" fn`) |
| 99 | AST_ERR                 | 0 | done (unchanged) |
| 100 | AST_AUTOTUNE_DECL      | 27 | new |
| 101 | AST_AUTOTUNE_PARAM     | 27 | new |
| 102 | AST_PROVENANCE_TYPE    | 24 | new (`D<Logic<T>>`) |
| 103 | AST_LOGIC_REL          | 24 | new (relational atom) |
| 104 | AST_TRACE_BEGIN        | 25 | new |
| 105 | AST_TRACE_END          | 25 | new |
| 106 | AST_PYTREE_LEAF        | 26 | new |
| 107 | AST_PYTREE_NODE        | 26 | new |
| 108 | AST_PARAM_STRUCT       | 28 | new (parametric struct) |
| 109 | AST_UNSAFE_BLOCK       | 28.6 | new |
| 110 | AST_DEPRECATED_DECL    | 28.7 | new |

### Trap-id namespace (AST_TAG * 1000 + sub_id)

Reserved blocks per AST_TAG. Sub_id 001-049 reserved for type-mismatch traps; 050+ for narrow/special traps; 100+ for stage-specific gates.

- Existing: 1001 (AST_VAR unbound), 6052/19052/20052/21052/22052/23052 (cmp narrow mismatch — DEFERRED), 8001-8016 (AST_ASSIGN val/bind matrix), 9001-9051 (AST_NEG by-type traps), 14001/14002 (FN body-vs-ret-ty), 16001/16002/16003 (CALL arg-type/arity), 32001-32040/33001-33040 (SHL/SHR float traps), 50001/52001 (TUPLE disp8 wrap), 99001 (AST_ERR fall-through).
- New per-stage reservations are listed in each stage section.

### FLAT prefix-trap pattern (host parser recursion budget)

All new trap insertions MUST use the flat `let n_trap = if cond { emit_trap_with_id(N) } else { 0 }` pattern, not nested `if cond { trap; } else { ... }`. Lesson from Findings #6 / #7 (audit-stage4-followup): host parser miscompiles unrelated programs when arms exceed recursion budget. Pattern:

```
let n_pre_trap = if cond { emit_trap_with_id(N) } else { 0 };
... existing body ...
n_existing_total + n_pre_trap
```

### Difficulty scale

Anchored to Stage 4 (tuples + arrays) = 6/10. Stage 5 (basic structs) ≈ 6/10. Higher numbers reflect more codegen surface area, more ABI complexity, or more port volume from Python.

### Per-stage commit count ranges

Approximate totals for the audit-fix loop (initial implementation + 1-3 audit cycles + fixes). Median stage = 8-15 commits.

---

## Stage 6: Enums (with payloads)

**Difficulty:** 7/10
**Estimated commits:** 12-18
**Dependencies:** Stage 5 Iter A (struct registration), Stage 4 (tuple field codegen)
**Iterations:**
- 6A: Lex/parse enum decl + variant registration; codegen 0-byte for decl
- 6B: AST_ENUM_CONSTRUCT for unit variants (`None`) — emit i32 discriminant
- 6C: AST_ENUM_CONSTRUCT for payload variants (`Some(42)`) — discriminant + payload, return as 16-byte struct in two stack slots
- 6D: AST_ENUM_DISCRIMINANT and AST_ENUM_PAYLOAD readers (used by Stage 7 match)
- 6E: bind_state extension to track enum-typed bindings (parallel to var_struct_tab)

### What to add

- AST tags reserved: 57 (ENUM_DECL), 58 (ENUM_VARIANT_DECL — linked list), 59 (ENUM_CONSTRUCT), 60 (ENUM_DISCRIMINANT), 61 (ENUM_PAYLOAD)
- New parser arms: `enum IDENT { Variant1, Variant2(T1, T2), ... }` at top level (parallel to `parse_struct_decl`); `IDENT::IDENT` (path) and `IDENT::IDENT(args)` recognized in `parse_primary` postfix
- New token kind: COLONCOLON detection — already lexed as TK_COLONCOLON or built from two `:` tokens; check existing lexer
- New parser state slots in `parse_top` arena region:
  - sb+20: enum_table base offset
  - sb+21: enum_table count (cap 4 enums, like struct cap 3)
  - sb+22: var_enum_table base (mirrors var_struct_table)
  - sb+23: var_enum_table count
  - sb+24: last_enum_idx scratch
- enum_table entry layout (5 slots): name_s, name_l, variant_count, variants_ptr (variant table base), max_payload_arity
- variant table entry layout (4 slots per variant): name_s, name_l, arity (0 for unit), discriminant_value
- New codegen helpers in `kovc.hx`:
  - `emit_enum_unit_construct(disc: i32)` — `mov eax, disc` (5 bytes)
  - `emit_enum_payload_construct(disc: i32, payload_idx: i32, ...)` — push disc + each payload val, return as 16-byte tuple in rax:rdx (Phase-0 use stack representation: low slot = disc, high slots = payload)
  - `emit_load_disc(off)` — `mov eax, [rbp + off]` (3 bytes for disp8, 6 for disp32)
  - `emit_load_payload(off, idx)` — `mov eax, [rbp + off + 8 + idx*8]`
- New stdlib pieces (deferred): replicate `helixc/stdlib/option.hx` semantics in user-level Helix once self-host is functional
- Tests:
  - `enum Maybe { None, Some(i32) }` decl-only — should compile to 0 extra bytes
  - `Maybe::None` returns 0 (discriminant)
  - `Maybe::Some(42)` returns 1 (discriminant); payload accessible via reader
  - Multi-variant: `enum Color { R, G, B }` returns 0/1/2
  - Mixed: `enum E { A, B(i32), C(i32, i32) }` arities OK

### Implementation outline

1. parser.hx:140 — add `kw_enum_s`/`kw_enum_n` accessors; install `enum` keyword in `install_keywords` (after `struct`).
2. parser.hx:1100 — extend `parse_top` slot count from 20 to 25; init enum_table.
3. parser.hx:1180 — `parse_program` skip-loop adds `enum` branch alongside `struct`.
4. parser.hx new fn `parse_enum_decl` (mirrors `parse_struct_decl` at line 1382): consume `enum IDENT { ... }`, register entry. Each variant: IDENT optionally followed by `(T1, T2, ...)`. Build linked list of AST_ENUM_VARIANT_DECL nodes; assign 0-based discriminants.
5. parser.hx `parse_primary` — when IDENT followed by `::`, peek for IDENT (variant name). Look up enum_table. If hit: parse optional `(args)`, build AST_ENUM_CONSTRUCT(enum_idx, variant_idx, args_head).
6. kovc.hx `emit_ast_code` — add tags 57 (decl: emit 0), 59 (construct: emit_enum_*_construct), 60 (load disc), 61 (load payload).
7. kovc.hx `expr_type` — tag 59 returns a packed type tag encoding (enum_idx << 8) | TAG_ENUM (use new tag 12 = enum). Subsequent `match` reads enum_idx via shift.
8. bind_state — add `bind_push_enum_typed(name, off, enum_idx)` parallel to `bind_push_typed`.

### Risks / known gotchas

- **Discriminant size**: Phase-0 fixes at i32 (4 bytes). Future: pick smallest fitting unsigned (u8 if < 256 variants).
- **Payload alignment**: variants with mixed payload widths require uniform 8-byte slots in Phase-0. Total stack region per enum binding = 8 + max_payload_arity * 8 bytes.
- **Recursive enums** (e.g. `enum List { Nil, Cons(i32, List) }`): NOT in Stage 6. Defer to Stage 6.5 or treat as "unsupported" trap. helixc-Python uses ARENA_GET dispatch (see `lower_ast.py:48-49`); kovc.hx would mirror that via `arena_push` of payload + slot index returned as the i32 enum value.
- **Generic enums** (`Option<T>`): defer to Stage 8 (monomorphization).
- **Parser ambiguity**: `Foo::Bar` could be enum variant OR module-path-call. Resolve by lookup priority: enum_table > module_table.
- **bind_lookup_type miss on enum**: `expr_type` for AST_VAR currently returns the bind_ty directly. Extend with `bind_lookup_enum_idx` so AST_PATTERN can resolve scrutinee variants.

### Test plan

```rust
// test 6A: decl-only
enum Maybe { None }
fn main() -> i32 { 0 }
// expect: returns 0; binary size unchanged from baseline + ~stub bytes

// test 6B: unit variant construct
enum Color { R, G, B }
fn main() -> i32 {
    let c = Color::G;
    c   // discriminant value 1
}
// expect: returns 1

// test 6C: payload variant
enum Maybe { None, Some(i32) }
fn main() -> i32 {
    let m = Maybe::Some(42);
    m   // returns discriminant 1
}
// expect: returns 1

// test 6D: payload reader
enum Maybe { None, Some(i32) }
fn main() -> i32 {
    let m = Maybe::Some(42);
    __enum_payload(m, 0)   // emits AST_ENUM_PAYLOAD(m, 0)
}
// expect: returns 42 (payload value)
```

### What gets ported from Python

- `helixc/frontend/parser.py:410-435` (`_parse_enum_decl`) → `parse_enum_decl` in parser.hx
- `helixc/frontend/ast_nodes.py` `EnumDecl`/`EnumVariant` → AST tag 57/58 layout
- `helixc/ir/lower_ast.py:45-49` (rec_enum_scope, ARENA_GET dispatch primitive) → kovc.hx Stage 6.5 deferral
- `helixc/frontend/typecheck.py` enum-type unification → minimal in Phase-0; defer real type-checking to post-Stage 8

### Trap-id reservations

- 57001 — ENUM_DECL with > 16 variants (Phase-0 cap)
- 59001 — ENUM_CONSTRUCT variant arity mismatch (e.g. `Some()` with no payload)
- 59002 — ENUM_CONSTRUCT unknown variant
- 60001 — ENUM_DISCRIMINANT on non-enum binding
- 61001 — ENUM_PAYLOAD index out of range

---

## Stage 7: Pattern matching (full Tier A)

**Difficulty:** 8/10
**Estimated commits:** 18-24
**Dependencies:** Stage 6 (enums + variants), Stage 4 (tuple destructure foundation)
**Iterations:**
- 7A: AST_MATCH parse + arm linked list (no exhaustiveness yet)
- 7B: PatLit + PatWildcard codegen — `match x { 0 => a, _ => b }`
- 7C: PatBind — `match x { v => v + 1 }`
- 7D: PatRange — `match x { 0..10 => a, 10..20 => b, _ => c }`
- 7E: Or-patterns + guards
- 7F: PatVariant (enum variant patterns) — `match m { Maybe::Some(v) => v, _ => 0 }`
- 7G: PatTuple — `match (a, b) { (0, _) => x, (_, 0) => y, _ => z }`
- 7H: Exhaustiveness check at compile time

### What to add

- AST tags 62 (MATCH), 63 (MATCH_ARM linked list), 64 (PAT_LIT), 65 (PAT_BIND), 66 (PAT_WILDCARD), 67 (PAT_RANGE), 68 (PAT_OR), 69 (PAT_VARIANT), 70 (PAT_TUPLE)
- New parser fns: `parse_match_expr`, `parse_pattern`, `parse_pattern_atom` (mirror `parser.py:1141-1267`)
- New token: FATARROW `=>` — already in lexer if not, add it
- AST_MATCH layout: p1 = scrut_idx, p2 = arms_head (AST_MATCH_ARM linked list), p3 = unused
- AST_MATCH_ARM layout: p1 = pattern_idx, p2 = body_idx, p3 = next_arm_idx | guard_idx (packed)
- Match-lowering: at codegen time, lower AST_MATCH directly to nested `if/else` chains (mirroring `match_lower.py`). The pattern test logic is inlined per arm:
  - PatLit(v): `cmp eax, v; je arm_body`
  - PatBind(name): always-true, push binding before body
  - PatRange(lo, hi, =/<): `cmp eax, lo; jl skip; cmp eax, hi; jg skip; jmp arm_body`
  - PatVariant(disc, payload_pats): `cmp [scrut_off + 0], disc; jne skip; recurse on payload pats with scrut_off + 8 + i*8`
  - PatTuple: recurse on each elem
  - PatOr: emit each branch test, OR-jump to body
  - Guard: emit guard expr; `test eax, eax; jz skip`
- New codegen helpers:
  - `emit_jl_rel32(disp)`, `emit_jg_rel32(disp)`, `emit_jne_rel32(disp)`, `emit_jmp_rel32(disp)`, `emit_jge_rel32(disp)`, `emit_jle_rel32(disp)`
  - All currently exist for jcc rel8; need rel32 forms for arms with large bodies. Encoding: `0F 8C disp32` (jl), `0F 8D` (jge), `0F 8F` (jg), `0F 8E` (jle), `0F 85` (jne), `E9 disp32` (jmp).
- Backpatch infrastructure: extend `patch_state` with arm-end-jump targets so each arm body can `jmp arm_end` after producing eax.

### Implementation outline

1. parser.hx — add tokens FATARROW, ELLIPSIS (`...` for inclusive range — defer if not in lexer); install `match` keyword.
2. parser.hx:580 (parse_primary) — when IDENT == "match", call `parse_match_expr`.
3. parser.hx new `parse_match_expr` at end of file: consume `match`, parse scrut, `{`, parse arms separated by `,`, `}`. Build AST_MATCH(scrut, arms_head).
4. parser.hx new `parse_pattern` / `parse_pattern_atom`: dispatch on token kind. INT/FLOAT/STRING → PAT_LIT. IDENT followed by `::` → PAT_VARIANT. IDENT → PAT_BIND. `_` → PAT_WILDCARD. INT `..` INT → PAT_RANGE. `(` → PAT_TUPLE. Pattern with `|` → PAT_OR. Pattern with `if expr` → guard.
5. kovc.hx `emit_ast_code` — add tag 62 (AST_MATCH) handler:
   - Run match-lowering: build implicit if/else chain; emit each arm via existing AST_IF emission.
   - Or: emit directly with arm-by-arm dispatch (preferred for code-size).
6. kovc.hx exhaustiveness checker: walk arms; for enum scrutinee, check that all variants are covered OR one arm is wildcard. Emit compile-time trap 62001 if not exhaustive (Phase-0 simple form).
7. AD-through-match: defer to Stage 12+ (autodiff lowering walks through match arms). Phase-0 7H trap with id 62002 if `grad(f)` includes a match.

### Risks / known gotchas

- **Backpatch pressure**: each arm has 2-4 jump sites (test, body, end). With N arms and complex patterns, the patch_state region needs ample headroom. Cap at 64 arms in Phase-0 (trap 62003 if exceeded).
- **Range patterns `0..10`**: lexer emits DOTDOT or DOTDOTEQ. Verify lexer covers this; if not, add lex rules.
- **Or-patterns binding rules**: `a | b => v` requires same names bound on both sides. Defer enforcement to post-Stage-7H.
- **Recursive pattern depth**: nested PatTuple in PatVariant strains host parser recursion. Use FLAT pattern.
- **Match lowering memo**: the Python `match_lower.py` does AST→AST rewrite; kovc.hx skips the rewrite and emits directly. Parity test: byte-identical output for AST_MATCH-bearing programs.

### Test plan

```rust
// 7B: simple lit match
fn main() -> i32 {
    let x = 5;
    match x { 0 => 100, 5 => 42, _ => 0 }
}
// expect: 42

// 7D: range
fn main() -> i32 {
    let x = 12;
    match x { 0..10 => 1, 10..20 => 2, _ => 0 }
}
// expect: 2

// 7F: enum variant pattern
enum Maybe { None, Some(i32) }
fn main() -> i32 {
    let m = Maybe::Some(42);
    match m { Maybe::None => 0, Maybe::Some(v) => v }
}
// expect: 42

// 7G: tuple pattern
fn main() -> i32 {
    let p = (1, 2);
    match p { (0, _) => 100, (1, y) => y, _ => 0 }
}
// expect: 2

// 7H: exhaustiveness fail
enum Maybe { None, Some(i32) }
fn main() -> i32 {
    let m = Maybe::Some(42);
    match m { Maybe::None => 0 }   // missing Some arm
}
// expect: compile-time trap 62001 (not exhaustive)
```

### What gets ported from Python

- `helixc/frontend/parser.py:1141-1267` (parse_match_expr, parse_pattern, parse_pattern_atom) → `parse_match_expr` + `parse_pattern*` in parser.hx
- `helixc/frontend/match_lower.py` (entire 409 LOC) → kovc.hx direct codegen path (skip the AST rewrite step)
- `helixc/frontend/ast_nodes.py` Match/MatchArm/Pat* → AST tags 62-70
- `helixc/frontend/typecheck.py` `_check_pattern` → Phase-0 minimal exhaustiveness only

### Trap-id reservations

- 62001 — non-exhaustive match
- 62002 — match inside grad() (defer to Stage 12)
- 62003 — match arms > 64 (Phase-0 cap)
- 64001 — pattern type mismatch (e.g. PatLit f32 against i32 scrut)
- 67001 — PatRange lo > hi
- 69001 — PatVariant unknown variant

---

## Stage 8: Generics + monomorphization

**Difficulty:** 8/10
**Estimated commits:** 14-20
**Dependencies:** Stage 6 (enum-typed bindings), Stage 5 (struct-typed bindings)
**Iterations:**
- 8A: Parse generic params on fn decls — `fn id<T>(x: T) -> T { x }`
- 8B: Parse turbofish at call sites — `id::<i32>(5)`
- 8C: Mangling table — `id__i32`, `pair__i32_f64`
- 8D: Per-instantiation emission via deepcopy + type substitution
- 8E: Constraint propagation (drop generic fns from final binary; only mangled mono versions emitted)
- 8F: Trap on uninstantiated generic call (compile-time error)

### What to add

- AST tags 71 (AST_GENERIC_PARAM), 72 (AST_TURBOFISH)
- AST_GENERIC_PARAM layout: p1 = name_s, p2 = name_l, p3 = next_idx
- AST_FN_DECL extension: add slot 6 = generic_params_head (linked list of AST_GENERIC_PARAM); slot 7 = where_clauses_head (Stage 8.5 future)
- Lexer: ensure `<` `>` are not consumed as comparison ops in turbofish position. helixc-Python uses `_no_cmp_lt_gt` flag (parser.py:70); replicate.
- New parser fns:
  - `parse_generic_params(tok_base, sb)` → linked list head
  - `parse_turbofish_args(tok_base, sb)` → linked list of type tags (mirror `_parse_type_generic_args` from parser.py:546)
- New table: mono_table — instances of (fn_name, [type_args]) → mangled_name. Cap 32 instantiations Phase-0.
- mono_table entry: 5 slots: orig_name_s, orig_name_l, mangled_name_s (in arena), mangled_name_l, fn_decl_idx (clone with subst applied)
- mangle algorithm (kovc.hx side, mirrors `monomorphize.py:31-60`): build "name__type1_type2" by appending each type tag's name (i32, f64, etc.) separated by `_`.
- Mono pass at top-level: pre-codegen walk over AST_FN_LIST, find each AST_CALL with turbofish args. For each unique (orig_name, type_args) pair: clone the AST_FN_DECL, substitute TyVars, append to fn list with mangled name. Replace AST_CALL turbofish with AST_CALL of mangled_name.
- Generics in struct/enum: defer to Stage 28 (parametric structs).

### Implementation outline

1. parser.hx — add `parse_generic_params` after `parse_fn_decl`. Format: `<T1, T2, ...>` after fn name.
2. parser.hx `parse_fn_decl` — after consuming `fn IDENT`, peek for `<`. If yes, call `parse_generic_params`, store head in AST_FN_DECL slot 6.
3. parser.hx `parse_primary` IDENT-postfix — after `IDENT::<`, parse turbofish args, build AST_TURBOFISH(name_s, name_l, type_args_head). Then on `(`, build AST_CALL with turbofish ref in slot 4.
4. kovc.hx — new top-level pass `monomorphize_pass(ast_root, mono_state)` runs before fn_type_table init. Walks AST_FN_LIST + each fn body. For each AST_CALL with turbofish: lookup or create mono entry; clone AST_FN_DECL with substituted type tags; rewrite call.
5. Substitution: walk AST recursively, for each AST_PARAM type_tag matching a generic param name, replace with the concrete tag from turbofish. AST_VAR with TyVar binding: same substitution.
6. kovc.hx fn_type_table_init — register only mono'd fns + non-generic fns (skip generic FN_DECLs).
7. AST_CALL codegen — looks up mangled name in fn_table.

### Risks / known gotchas

- **Recursive instantiation**: `fn rec<T>(x: T) -> T { rec::<T>(x) }` infinite loop unless mono guard already-seen pairs (use mono_table_lookup before clone).
- **Turbofish-vs-cmp ambiguity**: `f<g>(h)` could be call-with-turbofish OR `(f < g) > (h)`. helixc-Python resolves via lookahead: see if a matching `>` exists before non-type tokens. Replicate the no_cmp_lt_gt flag. Phase-0 simpler: only treat `IDENT::<` as turbofish (require explicit `::`).
- **Cap on 32 instantiations**: trap 71001 if exceeded.
- **AST clone in arena**: requires deep-walking AST_FN_DECL and rewriting indices in a new arena region. Memory pressure: each clone ~50-200 nodes × 4 slots = 800 i32 = ~3KB. 32 clones = ~100KB, well under 1MB arena.
- **Mono pass interacts with Stage 7 match**: cloned fn body keeps same MATCH structure but pattern types substitute; defer until Stage 7 lands.

### Test plan

```rust
// 8A/B: identity fn
fn id<T>(x: T) -> T { x }
fn main() -> i32 {
    id::<i32>(42)
}
// expect: 42, mangled name `id__i32` in fn_table

// 8C: two instantiations
fn id<T>(x: T) -> T { x }
fn main() -> i32 {
    let a = id::<i32>(10);
    let b = id::<f64>(3.14);   // returns f64; main only returns i32 portion of a
    a
}
// expect: 10, two mangled names `id__i32` and `id__f64` emitted

// 8D: 2-param generic
fn pair<A, B>(a: A, b: B) -> A { a }
fn main() -> i32 {
    pair::<i32, f64>(7, 1.0)
}
// expect: 7

// 8F: uninstantiated call
fn id<T>(x: T) -> T { x }
fn main() -> i32 {
    id(5)   // no turbofish on call -> compile error 71002
}
```

### What gets ported from Python

- `helixc/frontend/parser.py:344-366` (_parse_generic_params) → `parse_generic_params` in parser.hx
- `helixc/frontend/parser.py:546-556` (_parse_type_generic_args) → `parse_turbofish_args`
- `helixc/frontend/monomorphize.py` (455 LOC, full file) → `monomorphize_pass` + supporting helpers in kovc.hx
- `helixc/frontend/ast_nodes.py` GenericParam, TyVar → tags 71

### Trap-id reservations

- 71001 — > 32 mono instantiations
- 71002 — uninstantiated generic call (no turbofish + no inference)
- 72001 — turbofish arity != generic param count
- 72002 — turbofish references unknown type

---

## Stage 8.5: Traits + typeclasses (minimal Rust-style)

**Difficulty:** 7/10
**Estimated commits:** 14-18
**Dependencies:** Stage 8 (mono pass)
**Iterations:**
- 8.5A: Trait decl parse — `trait Eq { fn eq(self, other: Self) -> bool; }`
- 8.5B: Impl block parse — `impl Eq for i32 { fn eq(self, other: i32) -> bool { self == other } }`
- 8.5C: Trait-bound resolution at mono — `fn cmp<T: Eq>(a: T, b: T) -> bool { T::eq(a, b) }` resolves to impl's mangled name
- 8.5D: Single-impl-per-trait per-type uniqueness check

### What to add

- AST tags 73 (AST_TRAIT_DECL), 74 (AST_IMPL_BLOCK), 75 (AST_TRAIT_BOUND)
- AST_TRAIT_DECL layout: p1 = name_s, p2 = name_l, p3 = methods_head (linked list of AST_FN_DECL signatures, no bodies)
- AST_IMPL_BLOCK layout: p1 = trait_name_s, p2 = trait_name_l, p3 = packed (target_type_tag, impls_head)
- AST_TRAIT_BOUND layout: p1 = generic_param_idx, p2 = trait_name_s, p3 = trait_name_l
- New parser fns: `parse_trait_decl`, `parse_impl_block`, `parse_trait_bounds` (in `where` clauses or `<T: Trait>` syntax)
- New tables in parse state:
  - sb+25/26: trait_table base/count (cap 8 traits)
  - sb+27/28: impl_table base/count (cap 16 impls)
- impl_table entry (5 slots): trait_idx, target_type_tag, methods_table_ptr, methods_count, original_impl_node_idx
- Trait method dispatch: at call site `T::eq(a, b)` where T is bound to Eq, mono pass resolves T's concrete tag → finds matching impl in impl_table → mangles to `i32__eq` (for `impl Eq for i32`).

### Implementation outline

1. parser.hx — install `trait`, `impl`, `for` keywords; add table init.
2. parser.hx new `parse_trait_decl` — `trait IDENT { ... }`; methods are parsed as AST_FN_DECL but with body=0 (signature-only).
3. parser.hx new `parse_impl_block` — `impl IDENT for TYPE { fn ... { ... } }`.
4. parser.hx — extend `parse_generic_params` to recognize `T: Trait + Trait2` bound syntax; build AST_TRAIT_BOUND linked list.
5. kovc.hx — extend `monomorphize_pass` to consult impl_table when resolving trait method calls. Add `resolve_trait_method(trait_idx, target_type_tag) -> mangled_name`.
6. kovc.hx — emit each impl's methods with mangled names (`<TypeName>__<method>`, e.g. `i32__eq`).
7. Method-call sugar: `a.eq(b)` for self-typed call resolves to `<TypeName>__eq(a, b)` via impl lookup; defer to Stage 8.5+ if too complex.

### Risks / known gotchas

- **Self type**: `Self` keyword binds to the impl's target type. Substitute in body before mangling.
- **Default method bodies in trait decls**: NOT in Phase-0. Trait decls only carry signatures.
- **Multiple impls of same trait for same type**: trap 73001 (Phase-0 enforces single-impl uniqueness).
- **Inherent impls vs trait impls**: Phase-0 only handles trait impls. `impl i32 { fn foo() { ... } }` (no trait) deferred.
- **Recursive trait bounds** (`T: Eq + Hash`): cap at 4 bounds per generic in Phase-0.

### Test plan

```rust
// 8.5A/B: trait + impl
trait Eq { fn eq(self, other: Self) -> i32; }
impl Eq for i32 { fn eq(self, other: i32) -> i32 { if self == other { 1 } else { 0 } } }
fn main() -> i32 {
    let a: i32 = 5;
    let b: i32 = 5;
    a.eq(b)
}
// expect: 1, mangled name `i32__eq` in fn_table

// 8.5C: bounded generic
trait Eq { fn eq(self, other: Self) -> i32; }
impl Eq for i32 { fn eq(self, other: i32) -> i32 { if self == other { 1 } else { 0 } } }
fn cmp<T: Eq>(a: T, b: T) -> i32 { T::eq(a, b) }
fn main() -> i32 {
    cmp::<i32>(5, 5)
}
// expect: 1, `cmp__i32` mono'd, calls `i32__eq`
```

### What gets ported from Python

- `helixc/frontend/parser.py:181-244` (impl + trait parsing) → `parse_impl_block`, `parse_trait_decl`
- `helixc/frontend/flatten_impls.py` (160 LOC) → flat-impl rewriting in kovc.hx mono pass
- `helixc/frontend/typecheck.py` trait-bound checking → minimal Phase-0 resolution

### Trap-id reservations

- 73001 — duplicate trait impl
- 74001 — impl method signature differs from trait signature
- 75001 — trait bound unsatisfied at mono

---

## Stage 9: Closures

**Difficulty:** 7/10
**Estimated commits:** 10-14
**Dependencies:** Stage 5 (structs as captured-env representation), Stage 8 (mono — closures with generic body)

### What to add

- AST tags 76 (AST_CLOSURE_LIT), 77 (AST_CLOSURE_CALL)
- Surface syntax: `|x, y| x + y` and `|x: i32| -> i32 { x + 1 }`
- Lowering: closures rewrite to a fn-with-env-arg + a struct lit binding the captured vars.
  - Captured vars detected by walking the body for AST_VAR refs not bound by the closure's own params.
  - Lower `|x| x + a` (where `a` is captured) to:
    ```
    struct __closure_0_env { a: i32 }
    fn __closure_0_body(env: __closure_0_env, x: i32) -> i32 { env.a + x }
    let __closure_0_env = __closure_0_env { a: a };  // capture site
    ```
  - At call site `c(5)` → `__closure_0_body(__closure_0_env, 5)`.
- AST_CLOSURE_LIT layout: p1 = params_head, p2 = body_idx, p3 = packed (env_struct_idx, captured_vars_head)
- AST_CLOSURE_CALL layout: p1 = env_var_name_s, p2 = env_var_name_l, p3 = args_head
- New parser fns: `parse_closure_lit` triggered when `|` not in expression context.
- New codegen: closures have a runtime representation as 16-byte struct (fn ptr + env ptr) on stack. AST_CLOSURE_CALL loads both, calls indirectly via `call rax`.
- Indirect call: `FF D0 = call rax` (2 bytes).

### Implementation outline

1. parser.hx — peek `|` at expr start. If yes, call `parse_closure_lit`. Else fall through to existing PIPE (bitwise or).
2. parser.hx — `parse_closure_lit` consumes `|params|`, optional `-> ret_ty`, then `expr` or `{ block }`.
3. kovc.hx — closure-lower pass before mono: walk AST_CLOSURE_LIT nodes, collect captured names (free vars in body), synthesize struct decl + fn decl, replace closure with construction.
4. kovc.hx — closure value: 16-byte stack region (fn_ptr + env_ptr). bind_state stamps as type tag 13 (closure).
5. AST_CLOSURE_CALL: load fn_ptr (8 bytes), push env_ptr, push args, `call rax`.

### Risks / known gotchas

- **Closures returning closures**: requires nested env structs. Phase-0 trap 76001 if depth > 2.
- **`move` semantics**: Phase-0 always copies captures (no borrow). Mutation of captured var doesn't propagate back.
- **Generic closures**: `|x: T| x + 1` requires generic context. Defer to Stage 8 done first.
- **Closure-typed params**: `fn foo(c: fn(i32) -> i32) { c(5) }` requires fn-pointer types. Phase-0 use type tag 13 = closure; closures with concrete signatures share the same tag.

### Test plan

```rust
fn main() -> i32 {
    let a = 10;
    let c = |x| x + a;
    c(5)
}
// expect: 15

// closure as value
fn apply(c: fn(i32) -> i32, x: i32) -> i32 { c(x) }
fn main() -> i32 {
    let c = |x| x * 2;
    apply(c, 21)
}
// expect: 42 (when fn-ptr types land)
```

### What gets ported from Python

- `helixc/frontend/parser.py:_parse_closure_lit` (search by closure construct) → parser.hx
- helixc-Python doesn't have an explicit closure-lower pass currently (closures lowered inline in lower_ast.py); kovc.hx adds a separate pre-mono pass.
- `helixc/ir/lower_ast.py` closure-related ops → kovc.hx closure-lowering pass

### Trap-id reservations

- 76001 — nested closure depth > 2
- 76002 — closure capture > 4 vars (Phase-0 cap)
- 77001 — closure call arity mismatch

---

## Stage 10: Modules + use

**Difficulty:** 5/10
**Estimated commits:** 8-12
**Dependencies:** Stage 8 (mono — pass over module-flattened tree)

### What to add

- AST tags 78 (AST_MOD_DECL), 79 (AST_USE_DECL), 80 (AST_PATH_EXPR)
- Surface: `mod foo { fn bar() -> i32 { 1 } }`, `use foo::bar;`, `foo::bar()`
- Lowering: `helixc/frontend/flatten_modules.py` already does this in Python — kovc.hx ports the algorithm.
- Mangling: `mod foo { fn bar }` → top-level `fn foo__bar`. `foo::bar(x)` → `foo__bar(x)`.
- Nested modules: `mod foo { mod inner { fn baz } }` → `foo__inner__baz`.

### Implementation outline

1. parser.hx — install `mod`, `use` keywords; parse mod block as nested fn list.
2. parser.hx — parse `use IDENT::IDENT::IDENT;` as AST_USE_DECL with path linked-list.
3. parser.hx — `IDENT::IDENT` in expr context: build AST_PATH_EXPR.
4. kovc.hx new pre-mono pass `flatten_modules_pass`: walk AST tree, lift mod-block items to top, mangle names, rewrite calls.
5. Use-decl handling: builds alias table; resolution in expr context: `bar(x)` first checks aliases → resolves to `foo__bar`.

### Risks / known gotchas

- **Visibility (`pub`)**: Phase-0 ignores; everything visible. Future enforcement.
- **Path collision**: `foo__bar` mangled name conflicts with user fn `foo__bar`. Trap 78001.
- **Recursive modules**: NOT in Phase-0.

### Test plan

```rust
mod foo { fn bar() -> i32 { 42 } }
fn main() -> i32 { foo::bar() }
// expect: 42, fn_table has `foo__bar`
```

### What gets ported from Python

- `helixc/frontend/flatten_modules.py` (231 LOC) → `flatten_modules_pass` in kovc.hx
- `helixc/frontend/parser.py:122-152, 246-256, 445-450` (mod/use parsing) → parser.hx

### Trap-id reservations

- 78001 — mangled name collision with user fn
- 79001 — `use` path references unknown module

---

## Stage 11: Reflection runtime (kovc.hx side)

**Difficulty:** 9/10
**Estimated commits:** 16-22
**Dependencies:** Stage 6 (enums for cell-state union), Stage 7 (match in verifier blocks)

### What to add

- AST tags 81 (AST_QUOTE), 82 (AST_SPLICE), 83 (AST_MODIFY), 84 (AST_VERIFIER_BLOCK)
- Surface:
  - `Quote(expr)` returns reflection cell handle (i32 cell index)
  - `Splice(handle)` materializes cell's AST as expression value
  - `modify(handle, new_expr) verifier { ... }` mutates cell's AST after verifier passes
- Cell store: a contiguous arena region holding (ast_handle, parent_handle, hash, mut_flag, verifier_fn_idx) per cell.
- New runtime helpers in kovc.hx (these go INTO the produced binary, not just the compiler):
  - `__quote_cell_alloc(ast_idx) -> i32` — register cell, return handle
  - `__quote_cell_get(handle) -> i32` — fetch ast_idx
  - `__quote_cell_set(handle, ast_idx) -> i32` — overwrite ast_idx
  - `__quote_hash(ast_idx) -> i32` — structural hash (port from `ast_hash.py`)
  - `__verify_match(cell_handle, predicate_fn) -> i32` — run predicate, return pass/fail
- Verifier-gated modify: `modify(h, new) verifier { f(new) }` lowers to:
  - emit `predicate = f(new)`
  - `cmp predicate, 1; jne abort_modify`
  - `__quote_cell_set(h, new_ast_idx)`
- Structural hash: SHA-256 over node-tag + child-hashes + literal values, with bound names canonicalized to de-Bruijn indices. Mirrors `ast_hash.py:_hash_into`.

### Implementation outline

1. parser.hx — install `Quote`, `Splice`, `modify`, `verifier` keywords.
2. parser.hx new fns: `parse_quote_expr`, `parse_splice_expr`, `parse_modify_expr`.
3. kovc.hx — pre-emit pass `reflect_pass`: walk AST, for each AST_QUOTE node, allocate a cell at compile time (build cell_table region in arena, embed in `.data` section of output binary).
4. kovc.hx runtime emit: include compiled `__quote_*` helpers in every binary (always-on, ~200 bytes overhead).
5. kovc.hx structural hash port: use SHA-256 inline (~150 LOC); or simpler FNV-1a with SHA-256 deferred to Stage 17.
6. AST_MODIFY codegen: emit verifier block, branch on result, conditional `__quote_cell_set`.
7. AST_SPLICE codegen: load cell's ast_idx, recursively emit_ast_code on the loaded subtree (treat as runtime AST eval — limited to constant cells in Phase-0).

### Risks / known gotchas

- **SHA-256 in bootstrap is expensive**: ~150 LOC of bit-twiddling (rotr, choice, majority). Defer real SHA-256 to Stage 17 (hash-cons); use FNV-1a as Phase-0 placeholder (accept hash collisions).
- **AST_SPLICE eval at runtime**: requires interpreter support OR JIT compile the spliced AST. Phase-0: only constant cells (no runtime AST eval); trap 82001 if spliced cell is mutable.
- **Verifier scoping**: verifier_fn must be `@pure`. Effect-check pass (Stage 19) enforces.
- **Reflection self-modify cycle**: cell containing modify-of-self can infinite-loop. Cap modify depth at 8 per top-level call (trap 83001).

### Test plan

```rust
fn main() -> i32 {
    let h = Quote(1 + 2);    // cell handle
    Splice(h)                // returns 3
}
// expect: 3

fn always_true(_: i32) -> i32 { 1 }
fn main() -> i32 {
    let h = Quote(0);
    modify(h, 42) verifier { always_true(0) };
    Splice(h)
}
// expect: 42 (verifier passed; modify took effect)
```

### What gets ported from Python

- `helixc/frontend/ast_hash.py` (309 LOC) → kovc.hx `__quote_hash` runtime helper (FNV-1a Phase-0; SHA-256 Stage 17)
- `helixc/ir/lower_ast.py` QUOTE/SPLICE/MODIFY ops → kovc.hx codegen + cell_table runtime layout
- `helixc/frontend/typecheck.py` reflection-related checks → minimal Phase-0

### Trap-id reservations

- 81001 — quote of non-pure expr
- 82001 — splice of mutable cell at runtime (Phase-0 only constant cells)
- 83001 — modify recursion depth > 8
- 84001 — verifier returned non-bool / non-1

---

## Stage 12: AD framework — forward mode in bootstrap

**Difficulty:** 8/10
**Estimated commits:** 16-20
**Dependencies:** Stage 7 (match in derivative formulas), Stage 8 (mono of derivative fns)

### What to add

- AST tags 85 (AST_AD_FORWARD), 86 (AST_AD_TANGENT_PAIR)
- Surface: `grad(loss)(x)` (where `loss: fn(f64) -> f64`) → returns `df/dx` evaluated at `x`.
- `grad(f)` is a compile-time meta-call: `grad_pass.py` walks loss's body, generates `loss__grad` fn, replaces `grad(loss)` with name ref.
- Differentiation rules (mirror `autodiff.py`):
  - `d(c) = 0` (literal)
  - `d(x) = 1` if x == var, else 0
  - `d(a + b) = d(a) + d(b)`
  - `d(a - b) = d(a) - d(b)`
  - `d(a * b) = d(a)*b + a*d(b)`
  - `d(a / b) = (d(a)*b - a*d(b)) / (b*b)`
  - `d(-a) = -d(a)`
- AST cloning + simplifier: same arena-walk as monomorphize. Simplifier folds `0 + x = x`, `x * 1 = x`, `0 * x = 0`, etc.
- Memoization: cache `differentiate(expr_hash, var) -> derived_expr`.

### Implementation outline

1. kovc.hx new pre-codegen pass `grad_pass`: walk AST_FN_LIST + bodies, find AST_CALL with name == "grad". For each `grad(loss)` arg, look up `loss` in fn list, run differentiate over body, emit new fn `loss__grad`, rewrite call.
2. kovc.hx `differentiate(expr_idx, var_idx) -> i32`: case-split on tag, build new AST nodes via mk_node.
3. kovc.hx `simplify(expr_idx) -> i32`: bottom-up, fold algebraic identities.
4. kovc.hx hash-memoize: use FNV-1a hash of expr_idx subtree as cache key.

### Risks / known gotchas

- **Arena growth**: each grad expansion can 2-5x the AST node count. Watch arena overflow; bump capacity if needed.
- **Multivariate via forward-mode**: `grad(f)(x, y, z)` with 3 inputs requires 3 separate forward passes (one per var). Reverse mode (Stage 14) does it in one.
- **Calls inside loss**: helixc-Python inlines `@pure` user fns before differentiating. kovc.hx mirrors via `_inline_user_calls`. Recursion guard: cap inlining depth at 6.
- **If/while in loss**: NOT in Phase-0 (autodiff.py:18 says "NOT YET"). Trap 85001.

### Test plan

```rust
fn loss(x: f64) -> f64 { x * x + 3.0_f64 * x }
fn main() -> f64 {
    grad(loss)(2.0_f64)   // d/dx (x^2 + 3x) = 2x + 3, at x=2 = 7
}
// expect: 7.0_f64

// matched user fn
fn helper(x: f64) -> f64 { x * x }
fn loss(x: f64) -> f64 { helper(x) + x }
fn main() -> f64 {
    grad(loss)(3.0_f64)   // d(x^2 + x) = 2x + 1, at x=3 = 7
}
// expect: 7.0_f64
```

### What gets ported from Python

- `helixc/frontend/autodiff.py` (731 LOC) → kovc.hx `differentiate` + `simplify` + `_inline_user_calls`
- `helixc/frontend/grad_pass.py` (503 LOC) → kovc.hx `grad_pass`
- `helixc/frontend/ast_hash.py` (memo cache key) → FNV-1a Phase-0

### Trap-id reservations

- 85001 — control flow (if/while) inside grad arg (Phase-0 limitation)
- 85002 — non-pure call inside grad arg
- 86001 — grad of non-fn-typed value

---

## Stage 13: AD across user-defined fn calls — COMPLETE (2026-05-08)

**Difficulty:** 7/10
**Estimated commits:** 8-12 (landed in 1 commit thanks to Stage 12c reuse)
**Dependencies:** Stage 12 (forward AD established)
**Status:** Landed via single commit on branch `stage13-ad-user-fns`. Six
new regression tests in `test_codegen.py` cover (a) basic helper inlining,
(b) multi-level helper inlining, (c) direct-recursion guard, (d) mutual-
recursion guard, (e) `@pure`-marked helpers (back-compat), and (f)
composition with transcendental chain rule. All six pass; heavy gate
(`test_bootstrap_kovc_full_pipeline_arithmetic`) green; no regressions.

### What to add

- AST tag 87 (AST_AD_USER_CALL_FWD)
- Two strategies (helixc-Python uses inlining):
  1. **Inline**: substitute callee's body for the call; differentiate inlined tree. Already in Stage 12 via `_inline_user_calls`.
  2. **Chain rule**: keep call, generate `call__grad` for each callee; differentiate as `f(g(x))' = f'(g(x)) * g'(x)`.
- Stage 13 implements (1) fully (depth limit 6) and lays groundwork for (2).

### Implementation outline

1. kovc.hx — extend `_inline_user_calls` to recursively inline nested calls up to depth 6.
2. kovc.hx — add cycle detection: if A calls B and B calls A, trap 87001 (recursion not differentiable).
3. kovc.hx — add `chain_rule_call(call_idx, var)`: emits `f__grad(g(x)) * g__grad(x)` form. Used when inlining hits depth limit OR call is recursive.

### Risks / known gotchas

- **Mutual recursion**: A→B→A cycles caught by visited-set.
- **Call with different params**: `f(g(x), h(y))` differentiated w.r.t. x: `f__grad(g(x), h(y), 1) * g__grad(x)` (using partial w.r.t. arg 0). Multivariate derivative interface needed.

### Test plan

```rust
fn g(x: f64) -> f64 { x * x }
fn f(x: f64) -> f64 { g(x) + x }
fn main() -> f64 {
    grad(f)(3.0_f64)   // d/dx (x^2 + x) = 2x + 1, at x=3 = 7
}
// expect: 7.0_f64
```

### What gets ported from Python

- `helixc/frontend/autodiff.py` `_inline_user_calls` + `_simplify` → kovc.hx
- Chain-rule machinery (not in helixc-Python autodiff.py — added fresh)

### Trap-id reservations

- 87001 — recursive fn in grad arg

---

## Stage 14: AD framework — reverse mode

**Difficulty:** 9/10
**Estimated commits:** 18-24
**Dependencies:** Stage 12, Stage 5 (struct return for multi-output)

### What to add

- AST tags 88 (AST_AD_REVERSE), 89 (AST_AD_ADJOINT_BUCKET)
- Surface: `grad_rev_all(loss)(x, y, z)` returns tuple `(df/dx, df/dy, df/dz)` in one backward sweep.
- Algorithm (mirror `autodiff_reverse.py`):
  1. Inline user calls + let bindings.
  2. Walk tree top-down with current adjoint.
  3. Per binop: split adjoint per local Jacobian.
  4. At each Name(param) leaf, accumulate adjoint into bucket.
  5. After walk, sum each param's bucket; emit `(sum_x, sum_y, sum_z)`.
- New AST_AD_ADJOINT_BUCKET — internal AST node, holds list of expressions to be summed.

### Implementation outline

1. kovc.hx new pre-codegen pass `grad_reverse_pass` (parallel to `grad_pass`).
2. kovc.hx `differentiate_reverse(expr_idx, param_names_array) -> dict_of_grads`.
3. kovc.hx `_propagate(node_idx, adj_idx, acc_buckets)`: case-split on op kind.
4. kovc.hx `_sum_exprs(bucket)`: chain of binary +.
5. kovc.hx — emit grad fn returning a struct `{ df_dx0: f64, df_dx1: f64, ... }` (Stage 5 struct return required).

### Risks / known gotchas

- **Adjoint accumulation memory**: each param has list of expressions; total grows quadratically with body size.
- **Calls + ifs in body**: NOT in Phase-0 (autodiff_reverse.py:37). Trap 88001.
- **Multi-output return**: struct-return ABI in Stage 5C. Reverse-mode REQUIRES Stage 5C done first.

### Test plan

```rust
fn loss(x: f64, y: f64) -> f64 { x * y + x * x }
struct Grad { dx: f64, dy: f64 }
fn main() -> f64 {
    let g = grad_rev_all(loss)(2.0_f64, 3.0_f64);
    g.dx   // d/dx (xy + x^2) = y + 2x, at (2,3) = 7
}
// expect: 7.0_f64
```

### What gets ported from Python

- `helixc/frontend/autodiff_reverse.py` (413 LOC) → kovc.hx `grad_reverse_pass` + helpers

### Trap-id reservations

- 88001 — control flow / call inside reverse-mode loss
- 89001 — adjoint bucket overflow (cap 32 buckets per param)

---

## Stage 14.5: @checkpoint / rematerialization for reverse-mode AD

**Difficulty:** 7/10
**Estimated commits:** 8-12
**Dependencies:** Stage 14 (reverse mode)

### What to add

- AST tag 90 (AST_AD_CHECKPOINT)
- Attribute `@checkpoint` on fn decls — marks fn boundary as a recompute point in reverse mode.
- Backward sweep: at checkpoint, drop saved activations and recompute forward when needed.
- Implementation: split reverse pass into segments at @checkpoint boundaries; each segment recomputes its forward pass on demand.

### Implementation outline

1. parser.hx — parse `@checkpoint` attribute, record on fn decl.
2. kovc.hx — `grad_reverse_pass` builds activation-save-table per segment; @checkpoint segments emit recompute stubs.
3. kovc.hx — at backward sweep through checkpoint, emit forward call to recompute, then propagate adjoint.

### Risks / known gotchas

- **Memory vs compute tradeoff**: checkpoint reduces peak memory at cost of 2x forward FLOPs for that segment. User-tunable knob.
- **Side effects in checkpointed segment**: must be `@pure`; trap 90001.

### Test plan

```rust
@checkpoint
fn deep_block(x: f64) -> f64 { x * x * x * x * x }
fn loss(x: f64) -> f64 { deep_block(x) + x }
fn main() -> f64 {
    grad_rev_all(loss)(2.0_f64).dx
}
// expect: 5*x^4 + 1 = 81 (memory savings invisible in test, just verify correctness)
```

### What gets ported from Python

- helixc-Python doesn't have @checkpoint yet — Stage 14.5 introduces it fresh.

### Trap-id reservations

- 90001 — non-pure fn marked @checkpoint

---

## Stage 15: Tile + tensor types + lowering

**Difficulty:** 9/10
**Estimated commits:** 20-28
**Dependencies:** Stage 8 (mono — generic tile<T>), Stage 16.5 (FFI for cuBLAS — recommended before)

### What to add

- AST tags 91 (TILE_LIT), 92 (TENSOR_LIT), 93 (TILE_LOAD), 94 (TILE_STORE), 95 (TILE_MATMUL)
- Surface types:
  - `tile<f32, [16, 16], HBM>` — typed tile with shape + memspace
  - `tensor<f32, [N, M]>` — abstract tensor (memspace inferred)
- Memspaces: HBM (global), SMEM (shared), REG (registers), TMEM (Blackwell tensor memory)
- Tile ops:
  - `tile_load(tensor, [row, col]) -> tile`
  - `tile_store(tile, tensor, [row, col])`
  - `tile_matmul(a, b) -> c` — fused matmul on tiles (typically 16x16 SMEM)
- Lowering: tile ops emit either:
  - **CPU path**: SIMD instructions (AVX2 / AVX-512) — Phase-0 simple
  - **GPU path**: PTX (Stage 16) — preferred but requires kernel emission

### Implementation outline

1. parser.hx — install `tile`, `tensor`, `HBM`, `SMEM`, `REG`, `TMEM` keywords.
2. parser.hx `parse_type` — recognize `tile<dtype, [shape], memspace>` syntax (mirror `_parse_tile_type` from parser.py:586).
3. kovc.hx — extend bind_state type tags with 14 (tile), 15 (tensor); pack shape + memspace into upper 4 bits OR into a side table.
4. kovc.hx — emit `tile_zeros<f32, [16,16], REG>()` as `xorps xmm0, xmm0` repeated 16 times (or alloca + memset 256 bytes).
5. kovc.hx — emit `tile_load` as memcpy from tensor stride * row + col offset.
6. kovc.hx — emit `tile_matmul` as triple-nested loop with fma instructions (CPU path); GPU path defers to Stage 16.

### Risks / known gotchas

- **Massive code size**: tile_matmul on 16x16 tiles is 4096 fma ops. Use loops with `imul`+pointer arith instead of unrolling.
- **Memory alignment**: SMEM requires 16-byte aligned loads. Phase-0 cap shapes at multiples of 4.
- **Shape erasure**: tile shape lost after monomorphization unless preserved in struct field. Use Stage 28 parametric struct.
- **Real linalg perf**: requires linking BLAS (Stage 16.5 FFI). Phase-0 emits naive triple-loop matmul.

### Test plan

```rust
fn main() -> f32 {
    let a = tile<f32, [4, 4], REG>::zeros();
    let b = tile<f32, [4, 4], REG>::ones();
    let c = tile_matmul(a, b);
    c.get(0, 0)
}
// expect: 0.0_f32 (zeros * ones = zeros)
```

### What gets ported from Python

- `helixc/ir/tile_ir.py` (244 LOC) → kovc.hx tile-IR layer
- `helixc/frontend/parser.py:557-586` (parse_tensor_type, parse_tile_type) → parser.hx
- `helixc/ir/lower_ast.py` tile/tensor lowering → kovc.hx codegen

### Trap-id reservations

- 91001 — tile shape > 64x64 (Phase-0 cap)
- 93001 — tile_load out of bounds
- 95001 — matmul shape mismatch

---

## Stage 16: PTX backend (port from Python)

**Difficulty:** 9/10
**Estimated commits:** 20-28
**Dependencies:** Stage 15 (tile types), Stage 11 (reflection if @kernel uses Quote)

### What to add

- AST tags 96 (KERNEL_DECL), 97 (PTX_BLOCK)
- Attribute `@kernel` on fn decls — mark for PTX emission.
- PTX text emission (no assembler — PTX is text-format virtual ISA).
- New backend module: split kovc.hx codegen between x86-64 (host) and PTX (device); device emit goes into `.cubin` section embedded in binary.

### Implementation outline

1. parser.hx — parse `@kernel` attribute.
2. kovc.hx — split top-level fn-emit into kernel vs host. Kernels emit PTX text via `ptx_emit` helpers.
3. kovc.hx — PTX emit helpers:
   - `ptx_emit_module_header()` — `.version 8.3 .target sm_75 .address_size 64`
   - `ptx_emit_kernel(fn_idx)` — `.visible .entry name(...)` + body
   - `ptx_emit_op(op_idx)` — translate AST op to PTX (`add.s32 %r1, %r2, %r3`, etc.)
4. kovc.hx — embed PTX text in produced binary's `.rodata` section; runtime stub calls cuModuleLoadData + cuLaunchKernel via FFI (Stage 16.5).
5. Kernel ABI: param passing via `.param` declarations; loads via `ld.param.u64`.

### Risks / known gotchas

- **Without FFI/cuBLAS, kernels can't actually run** — they're emitted but not executed unless Stage 16.5 lands.
- **PTX text is verbose**: a 50-line kernel produces 200+ PTX instructions. Phase-0 keep small toy kernels only.
- **Texture memory, TMA, WGMMA**: defer to v0.2.
- **PTX register allocation**: PTX has virtual regs; assembler does the alloc. Phase-0 allocate sequentially (`%r0`, `%r1`, ...).

### Test plan

```rust
@kernel
fn vec_add(a: tile<f32, [256], HBM>, b: tile<f32, [256], HBM>, c: tile<f32, [256], HBM>) {
    let i = thread_idx();
    c[i] = a[i] + b[i];
}
fn main() -> i32 {
    // host code launches kernel via FFI stub
    0
}
// expect: PTX section in binary; matches reference text from helixc-Python ptx.py.
```

### What gets ported from Python

- `helixc/backend/ptx.py` (179 LOC) → kovc.hx PTX emission
- `helixc/ir/tile_ir.py` (244 LOC) tile op enum → kovc.hx PTX op dispatch

### Trap-id reservations

- 96001 — @kernel on non-pure fn
- 97001 — kernel uses unsupported op (e.g. division before Stage 16.5 FFI for cuBLAS)

---

## Stage 16.5: FFI / extern "C" + repr(C)

**Difficulty:** 7/10
**Estimated commits:** 12-16
**Dependencies:** Stage 5 (structs as C-compatible records), Stage 16 (PTX uses cuBLAS via FFI)

### What to add

- AST tag 98 (AST_FFI_EXTERN_DECL)
- Surface: `extern "C" fn cublasGemmEx(...) -> i32;` (declaration only, no body)
- Attribute `repr(C)` on structs — enforces C-ABI layout (no field reordering, padding rules).
- Runtime: emit dynamic loader stub in `.text` section; call libdl `dlopen` / `dlsym` to resolve symbols at startup. Phase-0 simpler: link symbols at compile time via the static linker.

### Implementation outline

1. parser.hx — parse `extern "C"` modifier on fn decl; build AST_FFI_EXTERN_DECL.
2. parser.hx — parse `#[repr(C)]` attribute on struct decl.
3. kovc.hx — extern fns NOT included in fn_table for codegen; instead, emit ELF dynamic-symbol-table entry. Use `R_X86_64_PLT32` relocation for calls.
4. kovc.hx — for repr(C) structs: alignment per field's natural alignment (i32 = 4-byte, f64 = 8-byte); pad as needed. Differs from Phase-0 default (8-byte uniform).
5. ELF linkage: dynamic linker resolves at exec start; bootstrap binary now has dependencies.

### Risks / known gotchas

- **Bootstrap-from-binary purity**: introducing libc dep means the binary requires `ld-linux.so` + libc. The hex0→...→kovc chain stays libc-free; FFI binaries are a separate case.
- **Calling convention mismatch**: SysV vs Windows x64 ABIs differ. Phase-0 SysV-only; Windows port deferred.
- **Struct ABI subtleties**: SysV passes structs ≤16B in regs, >16B via memory. Implement carefully.

### Test plan

```rust
extern "C" fn puts(s: *const u8) -> i32;
fn main() -> i32 {
    let msg: *const u8 = "hello\0".as_ptr();
    puts(msg)
}
// expect: prints "hello", returns 6
```

### What gets ported from Python

- helixc-Python doesn't have FFI yet — Stage 16.5 introduces fresh. Reference: how libc is linked in `helixc/backend/x86_64.py:emit_elf_header` (existing).

### Trap-id reservations

- 98001 — FFI call to undeclared extern
- 98002 — repr(C) struct layout mismatch with C definition

---

## Stage 17: const-fold pass (port from Python)

**Difficulty:** 6/10
**Estimated commits:** 10-14
**Dependencies:** None (operates on AST, post-parse, pre-codegen)

### What to add

- New IR layer in kovc.hx: minimal Tensor IR (port of `helixc/ir/tir.py`) with op-kind dispatch + use-def chain.
- Const-fold pass: walk fn ops, replace `const_int 2 + const_int 3` → `const_int 5`. Iteratively until fixpoint.
- Folded ops: ADD/SUB/MUL/DIV/MOD/NEG, CMP_*, CAST.
- Algebraic identities: `x * 0 = 0`, `x * 1 = x`, `x + 0 = x`, `x - x = 0`.
- Wraparound: i32 / i64 overflow wraps two's-complement (mirrors backend behavior).

### Implementation outline

1. kovc.hx new IR module: tag-based ops + use-def graph (i32 indices into op_arena).
2. kovc.hx `const_fold_pass(fn_idx)`: walk ops, collect defs; for each op, check operand defs are CONST_*; if yes, evaluate, replace.
3. kovc.hx algebraic identities helper.
4. Two-pass: const-fold → DCE (Stage 18) to remove now-dead consts.
5. Phase-0 minimal: const-fold on AST itself (not lowered IR) — simpler implementation, slightly less powerful.

### Risks / known gotchas

- **i32 overflow**: `INT_MAX + 1` folds to `INT_MIN` (wrap), not Python int. Phase-0 `_wrap_int_to_type` (from const_fold.py:40-55).
- **f32 / f64 NaN**: NaN ops short-circuit (NaN + x = NaN). Phase-0 trap if NaN encountered (id 17001) until Stage 28.5 panic policy.
- **Side effects**: never fold across CALL or LOAD_VAR — alias analysis required (deferred).
- **AST-level vs IR-level**: doing it on IR is cleaner; on AST is simpler. Pick AST for Phase-0 to avoid full IR layer port.

### Test plan

```rust
fn main() -> i32 { 2 + 3 * 4 }
// expect: at compile time, body folds to const 14; emit `mov eax, 14`
```

### What gets ported from Python

- `helixc/ir/passes/const_fold.py` (452 LOC) → kovc.hx `const_fold_pass`
- `helixc/ir/tir.py` (474 LOC, partial — only op dispatch needed) → kovc.hx minimal IR

### Trap-id reservations

- 17001 — const-fold encountered NaN result (Phase-0 abort)

---

## Stage 18: CSE / DCE / FDCE

**Difficulty:** 6/10
**Estimated commits:** 12-16
**Dependencies:** Stage 17 (const-fold) — these passes run in sequence

### What to add

- CSE: hash-cons pure ops; deduplicate identical (kind, operands, attrs) across a fn.
- DCE: remove ops whose results have no live users AND no side effects.
- FDCE (fast DCE): single-pass forward-walk variant of DCE for use post-CSE.
- Pure op set: ADD/SUB/MUL/DIV/MOD/NEG/CMP_*/CAST/CONST_*.
- Side-effect ops: RETURN/BR/COND_BR/CALL/STORE_*/ALLOC_*/MODIFY/SPLICE/PRINT.

### Implementation outline

1. kovc.hx `cse_pass(fn_idx)`: build hash-table of (kind, operand_ids, attrs); on collision, replace later op's result-id with earlier's.
2. kovc.hx `dce_pass(fn_idx)`: build live-set via reverse walk from RETURN; remove ops whose result is dead AND not side-effecting.
3. kovc.hx `fdce_pass(fn_idx)`: single forward pass, drop dead consts.
4. Pass pipeline: const-fold → CSE → DCE → fdce → emit.

### Risks / known gotchas

- **Hash function**: must include result type so `bool MUL` doesn't merge with `i32 MUL` (audit-10 lesson).
- **CAST ops**: distinct target types stay distinct.
- **Ordering**: DCE must NOT remove side-effecting ops even if result unused (e.g. CALL).

### Test plan

```rust
fn main() -> i32 {
    let a = 2 + 3;
    let b = 2 + 3;   // CSE: shares with a
    a + b            // = 10
}
// expect: at IR, only one ADD op; emit single fold to const 10 (after CSE+const-fold).
```

### What gets ported from Python

- `helixc/ir/passes/cse.py` (134 LOC) → kovc.hx `cse_pass`
- `helixc/ir/passes/dce.py` (117 LOC) → kovc.hx `dce_pass`
- `helixc/ir/passes/fdce.py` (86 LOC) → kovc.hx `fdce_pass`

### Trap-id reservations

- (none — these passes are structural, not value-checking)

---

## Stage 19: Effect/capability check pass

**Difficulty:** 6/10
**Estimated commits:** 8-12
**Dependencies:** Stage 18 (passes infra)

### What to add

- Surface: `@effect(io.read_file)` on fn decls.
- Effect labels: `io`, `modify_self`, `alloc`, `unknown`.
- Closure: a fn's effects = own ops' effects ∪ transitive callees' effects.
- Pure check: `@pure` fn with non-empty closure → trap 19001.

### Implementation outline

1. parser.hx — extend `skip_attributes` to record effect set on fn decls.
2. kovc.hx new pass `effect_check_pass(ast_root)`: walk fn list, compute closure per fn, compare against declared.
3. kovc.hx — fixpoint iteration over call graph (typically converges in 2-3 iterations).
4. Trap on declared-vs-actual mismatch: id 19001 + label encoded.

### Implementation detail (mirrors `effect_check.py`)

- `OP_EFFECTS`: PRINT → io, MODIFY/SPLICE → modify_self.
- `META_ATTRS`: is_pub, is_pure, pure, kernel — not effect labels.
- `declared_effects(fn) = if is_pure { {} } else { fn.attrs.keys() - META_ATTRS }`.

### Risks / known gotchas

- **Indirect calls**: closure has unknown effect (label `unknown`); conservative super-effect.
- **Mutual recursion**: fixpoint converges only if effect set is monotone (it is — union).

### Test plan

```rust
@pure
fn loud() -> i32 { print("hi"); 0 }
// expect: trap 19001 — declared @pure but closure has io
```

### What gets ported from Python

- `helixc/ir/passes/effect_check.py` (157 LOC) → kovc.hx `effect_check_pass`

### Trap-id reservations

- 19001 — declared effects don't match actual closure

---

## Stage 20: Hash-cons (AST + IR)

**Difficulty:** 7/10
**Estimated commits:** 10-14
**Dependencies:** Stage 11 (reflection uses hashes), Stage 17 (const-fold benefits from hash-cons)

### What to add

- Real SHA-256 implementation in kovc.hx (replaces FNV-1a placeholder from Stage 11).
- Hash-cons table: when building AST nodes, check if structurally-identical node already exists; reuse handle.
- Memoize `differentiate()` keyed by AST hash.
- E-class infra (deferred to post-v1.0).

### Implementation outline

1. kovc.hx `__sha256_init`, `__sha256_update`, `__sha256_final` — port standard SHA-256 (~150 LOC).
2. kovc.hx — extend `mk_node` to consult hash table; if hit, return existing index.
3. kovc.hx — bound-name canonicalization (de-Bruijn): walk tree maintaining depth; AST_VAR refs hash by depth-from-binder, not name.
4. kovc.hx — memoize `differentiate(hash, var, fn_table_sig) -> derived_hash`.

### Risks / known gotchas

- **Hash table size**: 16k entry, 4-slot region. Use linear probing with FNV-1a as secondary hash.
- **Bound-name renaming**: `let x = 1; x` and `let y = 1; y` must hash equal. Requires de-Bruijn handling.
- **Mutation**: AST is immutable post-parse. Sharing safe.

### Test plan

```rust
fn main() -> i32 {
    let x = 1 + 2;
    let y = 1 + 2;   // same hash as x's RHS
    x + y
}
// expect: hash table has only one ADD entry; differentiate cache hit > 0 if grad applied
```

### What gets ported from Python

- `helixc/frontend/ast_hash.py` (309 LOC) → full SHA-256 port

### Trap-id reservations

- (none — structural)

---

## Stage 21: Total-by-default check

**Difficulty:** 6/10
**Estimated commits:** 6-10
**Dependencies:** Stage 19 (passes infra)

### What to add

- Surface: `@partial` opts out of totality check; `@total` opts in (default for Phase-0 = no check).
- Structural recursion check: for each non-`@partial` recursive fn, find a parameter that strictly decreases on every recursive call (`p - const`, `p / const`, or smaller component of `p`).

### Implementation outline

1. parser.hx — record `@partial` / `@total` attrs on fn decl.
2. kovc.hx new pass `totality_pass(ast_root)`: walk fn list, find direct recursive calls.
3. kovc.hx — for each recursive fn, check if any param strictly decreases on all recursive calls.
4. Mutual recursion: pessimistically reject unless any participant is `@partial`.

### Risks / known gotchas

- **Conservative**: returns "totality unprovable" for any pattern not recognized.
- **Non-recursive fns**: trivially total.

### Test plan

```rust
fn fact(n: i32) -> i32 {
    if n <= 1 { 1 } else { n * fact(n - 1) }   // n - 1 strictly decreases
}
// expect: totality OK

fn loop_fn(n: i32) -> i32 {
    if n == 0 { 0 } else { loop_fn(n) }   // n doesn't decrease
}
// expect: trap 21001 — annotate @partial
```

### What gets ported from Python

- `helixc/frontend/totality.py` (152 LOC) → kovc.hx `totality_pass`

### Trap-id reservations

- 21001 — totality unprovable (no @partial)

---

## Stage 22: Pretty error display

**Difficulty:** 5/10
**Estimated commits:** 8-12
**Dependencies:** None (operates on existing parse/codegen errors)

### What to add

- Source-with-caret format:
  ```
  parse error: expected `;`
       --> file.hx:5:12
        |
   5  | let x = 1
        |          ^
  ```
- Did-you-mean suggestions for typos: Levenshtein distance (≤ 2) against known names.
- Error aggregation: collect multiple errors before bailing.

### Implementation outline

1. kovc.hx — track per-token line/col in lexer (already partially done? verify); store in token slots.
2. kovc.hx — error rendering helper that takes (line, col, msg, source_buffer); slices source line, prints with caret.
3. kovc.hx — Levenshtein distance helper (~30 LOC).
4. kovc.hx — for AST_ERR / unbound name errors, scan known names, suggest closest.

### Risks / known gotchas

- **Source buffer access**: parser already reads source into arena; pass source ptr through.
- **Multibyte characters**: UTF-8 columns differ from byte columns; Phase-0 simpler (byte cols).

### Test plan

```rust
fn main() -> i32 {
    let foo = 5;
    bar             // unbound; suggest "foo"
}
// expect: "did you mean `foo`?" diagnostic
```

### What gets ported from Python

- `helixc/frontend/parser.py:39-61` (ParseError.render) → kovc.hx error formatter
- Levenshtein helper — fresh port

### Trap-id reservations

- (none — diagnostics, not traps)

---

## Stage 23: CLI flags

**Difficulty:** 4/10
**Estimated commits:** 6-10
**Dependencies:** None

### What to add

- `--emit-ir` — print Tensor IR after lowering, exit
- `--dump-ast-hashes` — print structural hash per fn, exit
- `--check-only` — typecheck + effect-check, no emit
- `--O0` / `--O1` / `--O2` — optimization levels (skip / run / aggressive const-fold + CSE + DCE)
- `--target=ptx|x86_64` — backend selector

### Implementation outline

1. kovc.hx `_start` stub — parse argc/argv; check for known flags before invoking compile pipeline.
2. kovc.hx — gate passes on flags: `-O0` skips const-fold/CSE/DCE.
3. kovc.hx — `--emit-ir` runs through const-fold/CSE/DCE then prints IR text.

### Risks / known gotchas

- **argv parsing in bootstrap**: requires reading argv from stack at `_start` entry. SysV: argc at [rsp], argv at [rsp+8].

### Test plan

```sh
$ kovc --emit-ir program.hx
fn main() -> i32 {
  %0 = const_int 14
  return %0
}

$ kovc --check-only program.hx
$ echo $?  # 0 = type-check passed
```

### What gets ported from Python

- helixc-Python's `helixc/check.py` CLI → kovc.hx CLI dispatcher

### Trap-id reservations

- (none — CLI errors return non-zero exit, not SIGILL)

---

## Stage 24: Provenance-typed neuro-symbolic (Tier 3)

**Difficulty:** 9/10
**Estimated commits:** 18-24
**Dependencies:** Stage 8 (mono on `D<Logic<T>>`), Stage 12+14 (AD)

### What to add

- AST tags 102 (PROVENANCE_TYPE), 103 (LOGIC_REL)
- Surface: `D<Logic<T>>` — differentiable type carrying provenance metadata.
- Logic atoms: `parent(alice, bob)` — relational facts.
- Differentiable through logic ops via fuzzy semantics (relaxed AND/OR with sigmoid/min).
- Tracks data lineage: which input rows contributed to each output.

### Implementation outline

1. parser.hx — recognize `D<...>` and `Logic<...>` as parametric types.
2. kovc.hx — extend type tags to accommodate provenance variant; build provenance lattice helper.
3. kovc.hx — logic op codegen: AND → min, OR → max (or fuzzy soft variants).
4. kovc.hx — AD-through-logic: chain rule for fuzzy ops.

### Risks / known gotchas

- **Substantial new infrastructure**: this is the strategic moat — minimal but unique.
- **No helixc-Python reference**: built fresh.

### Test plan

```rust
fn likely_parent(a: D<Logic<Person>>, b: D<Logic<Person>>) -> D<Logic<Bool>> {
    // ... fuzzy relational predicate
}
fn main() -> f64 {
    let p = likely_parent(alice, bob);
    grad(p)(alice).dx   // gradient w.r.t. alice's features
}
```

### What gets ported from Python

- helixc-Python doesn't have provenance types — Stage 24 is fresh design.

### Trap-id reservations

- 102001 — D<...> over non-differentiable type
- 103001 — Logic<...> with non-relational body

---

## Stage 25: Trace-based introspection

**Difficulty:** 7/10
**Estimated commits:** 10-14
**Dependencies:** Stage 11 (reflection)

### What to add

- AST tags 104 (TRACE_BEGIN), 105 (TRACE_END)
- Surface: `trace { expr }` — wraps expr; runtime captures op-by-op execution trace.
- Verifier check: equivalence of two traces (idempotence / commutativity probes).

### Implementation outline

1. parser.hx — install `trace` keyword.
2. kovc.hx — `trace { ... }` lowers to: enable trace buffer → emit body → disable trace.
3. kovc.hx — runtime trace buffer: ring of (op_kind, operand_values, result) tuples.
4. kovc.hx — `trace_equiv(t1, t2) -> bool` runtime helper.

### Risks / known gotchas

- **Memory overhead**: trace buffer can grow unbounded. Cap at 4KB Phase-0.
- **Side-effect ordering**: trace serializes; multi-threaded code requires synchronization.

### Test plan

```rust
fn main() -> i32 {
    let t = trace { 1 + 2 + 3 };
    trace_len(t)   // returns 3 (3 ADDs)
}
```

### Trap-id reservations

- 104001 — trace buffer overflow
- 105001 — trace_equiv called on differently-shaped traces

---

## Stage 26: JAX-style pytrees

**Difficulty:** 7/10
**Estimated commits:** 12-16
**Dependencies:** Stage 5 (struct introspection), Stage 14 (reverse-mode AD over struct)

### What to add

- AST tags 106 (PYTREE_LEAF), 107 (PYTREE_NODE)
- Surface: `grad(loss)(model)` where `model` is a nested struct of tensors.
- Pytree flattening: walk struct, collect leaf tensors; AD treats each leaf as a separate parameter.
- Pytree unflattening: zip gradients back into same struct shape.

### Implementation outline

1. kovc.hx — `flatten_pytree(struct_idx) -> [leaf_indices]`: walk struct, recursively descend into nested structs, return flat list of f64 / tensor leaves.
2. kovc.hx — `unflatten_pytree(grads_array, struct_idx) -> struct_idx_with_grads`: reverse.
3. kovc.hx — `grad_rev_all` over pytree-typed loss returns same-shape pytree of gradients.

### Risks / known gotchas

- **Recursive structs**: Phase-0 cap at depth 4.
- **Mixed tensor / scalar leaves**: each leaf type tracked.

### Test plan

```rust
struct Model { w1: f64, w2: f64 }
fn loss(m: Model, x: f64) -> f64 { m.w1 * x + m.w2 * x }
fn main() -> f64 {
    let m = Model { w1: 0.5, w2: 0.3 };
    let g = grad(loss)(m, 2.0_f64);   // returns Model { w1: 2.0, w2: 2.0 }
    g.w1 + g.w2   // = 4.0
}
```

### Trap-id reservations

- 106001 — pytree depth > 4
- 107001 — pytree leaf type not differentiable

---

## Stage 27: Triton-style autotune

**Difficulty:** 7/10
**Estimated commits:** 12-16
**Dependencies:** Stage 16 (PTX), Stage 26 (pytrees over kernel configs)

### What to add

- AST tags 100 (AUTOTUNE_DECL), 101 (AUTOTUNE_PARAM)
- Surface: `@autotune(BLOCK_SIZE: [16, 32, 64, 128], NUM_WARPS: [4, 8])` on kernel decl.
- Sweep at compile time: emit one kernel variant per (BLOCK_SIZE, NUM_WARPS) tuple; runtime picks fastest via timing micro-benchmark.

### Implementation outline

1. parser.hx — parse `@autotune(K: [v1, v2, ...])` attribute syntax.
2. kovc.hx — for each autotune config, mono-clone kernel with constants substituted.
3. kovc.hx — emit dispatch table: at first call, time each variant, record fastest.
4. kovc.hx — subsequent calls jump to fastest variant.

### Risks / known gotchas

- **Code size explosion**: 4 BLOCK_SIZEs × 2 NUM_WARPS = 8 kernel variants. Cap product at 16 Phase-0.
- **Timing precision**: requires high-res timer; on x86, use `rdtsc`. On GPU, use cuEventRecord.

### Test plan

```rust
@kernel
@autotune(BLOCK_SIZE: [16, 32], NUM_WARPS: [4])
fn matmul(a: tile<f32, [N, K], HBM>, b: tile<f32, [K, M], HBM>, c: tile<f32, [N, M], HBM>) {
    // ...
}
// expect: 2 PTX kernels emitted; runtime dispatch picks fastest
```

### Trap-id reservations

- 100001 — autotune product > 16

---

## Stage 28: Mojo-style parametric structs

**Difficulty:** 8/10
**Estimated commits:** 14-20
**Dependencies:** Stage 8 (mono), Stage 5 (structs)

### What to add

- AST tag 108 (PARAM_STRUCT)
- Surface: `struct Tensor<dtype, [N, M]> { data: *dtype, ... }` — struct parameterized by types AND const-shape values.
- Compile-time const eval: `[N, M]` evaluated to concrete sizes at instantiation.
- Differs from generics (Stage 8) by allowing const-int parameters in addition to type parameters.

### Implementation outline

1. parser.hx — extend `parse_struct_decl` to accept `<T1, T2, [N, M]>` parameter list.
2. kovc.hx — extend monomorphize_pass to handle const-int args in mangling.
3. kovc.hx — const-int substitution in struct field types (e.g. `[T; N]` array length).

### Risks / known gotchas

- **Const-eval scope**: only literal ints / type-tags allowed in const params Phase-0.
- **Recursive const params**: `Tensor<f32, [N, N+1]>` requires arithmetic in const positions. Defer to post-Phase-0.

### Test plan

```rust
struct Vec<T, [N: i32]> { data: [T; N] }
fn main() -> i32 {
    let v = Vec<i32, [4]> { data: [1, 2, 3, 4] };
    v.data[0] + v.data[3]   // = 5
}
```

### Trap-id reservations

- 108001 — parametric struct const-eval failure
- 108002 — parametric struct shape arity mismatch

---

## Stage 28.5: panic / abort policy

**Difficulty:** 5/10
**Estimated commits:** 6-10
**Dependencies:** Stage 16.5 (FFI for abort()), Stage 11 (reflection — verifier panics)

### What to add

- Default panic = `abort()` (calls libc abort or SYS_exit_group with exit code 134).
- `@unwind` attribute reserves future setjmp/longjmp-based unwinding (Phase-0 trap if present).
- Panic on integer overflow (debug builds), array out of bounds, etc.

### Implementation outline

1. kovc.hx — emit `abort` symbol via FFI extern decl.
2. kovc.hx — replace `emit_trap_with_id(N)` (ud2 SIGILL) with `mov edi, N; call abort` for richer diagnostics. Keep ud2 as fallback for compiler bugs.
3. parser.hx — recognize `@unwind` (parse, error: not yet supported).

### Risks / known gotchas

- **Bootstrap purity**: abort() requires libc → bootstrap stays SIGILL-based (ud2). User-facing binaries can opt into abort via `-fpanic=abort` flag.

### Test plan

```rust
fn main() -> i32 {
    let arr: [i32; 4] = [1, 2, 3, 4];
    arr[10]   // out of bounds; aborts
}
// expect: process exits with status 134 (abort)
```

### Trap-id reservations

- 28501 — @unwind attribute not yet supported

---

## Stage 28.6: unsafe block for raw-ptr ops

**Difficulty:** 6/10
**Estimated commits:** 8-12
**Dependencies:** Stage 16.5 (FFI), Stage 11 (reflection)

### What to add

- AST tag 109 (UNSAFE_BLOCK)
- Surface: `unsafe { *raw_ptr = 42; *raw_ptr.offset(8) = 100; }`
- Inside unsafe: raw-pointer arithmetic, FFI calls without effect-check, untyped memcpy.
- Outside unsafe: raw-ptr ops trap 109001.

### Implementation outline

1. parser.hx — install `unsafe` keyword.
2. parser.hx — `unsafe { block }` parses normally; AST node sets a flag on contained expressions.
3. kovc.hx — track unsafe context during codegen; raw-ptr ops in safe context trap.
4. kovc.hx — effect-check pass treats unsafe block as a capability barrier.

### Risks / known gotchas

- **Unsafety propagation**: an unsafe block doesn't propagate up. Caller can still be safe.
- **FFI default unsafe**: extern "C" calls always unsafe.

### Test plan

```rust
fn main() -> i32 {
    let mut x: i32 = 0;
    unsafe { let p: *mut i32 = &mut x as *mut i32; *p = 42; }
    x   // = 42
}
```

### Trap-id reservations

- 109001 — raw-ptr deref outside unsafe

---

## Stage 28.7: @deprecated + @since version gating

**Difficulty:** 4/10
**Estimated commits:** 6-8
**Dependencies:** Stage 19 (effect/attr infra)

### What to add

- AST tag 110 (DEPRECATED_DECL)
- Surface: `@deprecated("use new_api instead")` on fn / struct decls.
- `@since("1.2.0")` for forward-compat marker.
- Compile-time warning (not error) when calling deprecated symbol.

### Implementation outline

1. parser.hx — parse `@deprecated` / `@since` attributes.
2. kovc.hx — at AST_CALL site, check callee's deprecated flag; emit compile-time warning to stderr.
3. kovc.hx — `--Wdeprecated=error` CLI flag promotes warning to error.

### Risks / known gotchas

- **Stderr emit in bootstrap**: requires write(2) syscall path or libc.

### Test plan

```rust
@deprecated("use new_id")
fn old_id<T>(x: T) -> T { x }
fn main() -> i32 {
    old_id::<i32>(5)   // compile-time warning
}
```

### Trap-id reservations

- (none — warnings, not traps)

---

## Stage 29: Drop helixc-Python

**Difficulty:** 8/10 (verification volume, not new code)
**Estimated commits:** 10-14
**Dependencies:** ALL prior stages

### What to add

- Equivalence test harness: for every test case in `helixc/tests/test_codegen.py`, `test_select_codegen.py`, `test_typecheck.py`, etc., compile with both helixc-Python and kovc.hx; assert binary output is byte-identical.
- Coverage gap fixer: any case where outputs differ requires either kovc.hx fix OR documented intentional divergence.
- Mark helixc-Python as deprecated reference (keep code; mark in setup.py).

### Implementation outline

1. New script `bin/equivalence.py`: compile each `.hx` file with both compilers, diff outputs.
2. Iterate through test cases; for each diff, root-cause + fix kovc.hx.
3. Document intentional differences (e.g. helixc-Python uses 4-byte cmp imm, kovc.hx uses 8-byte mov+cmp — both correct).
4. Run `pytest helixc/tests/` against both; assert pass rate ≥ Python's (kovc.hx may pass MORE tests if it has stricter type checking).
5. Tag helixc-Python with deprecation notice in `helixc/__init__.py`.

### Risks / known gotchas

- **Byte-identical is unrealistic**: register allocation, jump-displacement encoding, etc., differ. Loosen to "behavior-identical": same exit code, same stdout, same observable side effects.
- **Iteration count**: each gap can take 5-50 commits to close. Budget 50-200 commits for Stage 29 alone.
- **Unique kovc.hx features**: features present in kovc.hx but not Python (e.g. type-tag traps) cause "intentional divergence" doc entries.

### Test plan

- Run all of `helixc/tests/test_*.py` against kovc.hx instead of helixc-Python.
- Pass rate ≥ helixc-Python pass rate.
- All bootstrap_kovc tests pass.

### What gets ported from Python

- This stage doesn't port — it verifies completeness.

### Trap-id reservations

- (none — verification stage)

---

## Stage 30: 5 consecutive clean audits

**Difficulty:** 9/10 (volume + bar-raising)
**Estimated commits:** Variable (audits + fixes); estimate 30-100 commits
**Dependencies:** Stage 29 (parity baseline)

### What to add

- Run multi-agent audit cycle (3 agents: code-reviewer, silent-failure-hunter, type-design-analyzer).
- Each cycle reviews kovc.hx + parser.hx + lexer.hx + stdlib + tests for new findings.
- Counter: increment on zero new findings; reset on any new finding.
- Target: 5 consecutive cycles with zero new findings.

### Implementation outline

1. `.claude/agents/code-reviewer.md`, `silent-failure-hunter.md`, `type-design-analyzer.md` — already exist (reference Stage 4 follow-up).
2. Per cycle: spawn 3 agents in parallel via Agent tool; each produces a finding-list.
3. Triage: HIGH findings fix immediately; MEDIUM/LOW logged; LOW dismissed if accepted-corner-case.
4. After each fix-batch, re-run full test suite + bootstrap_kovc verification.
5. Counter increments on zero new findings; resets on any.

### Risks / known gotchas

- **Audit drift**: agents may converge on same finding repeatedly. Use prior-findings list to suppress.
- **Long tail**: as bugs get rarer, each cycle takes longer (large file scans). Total wall time can be days.
- **Final TG message**: "HELIX FULLY FINALIZED" announcement when count = 5.

### Test plan

- Each cycle: full test suite passes (after fixes).
- After 5 clean cycles: tag commit `helix-v1.0`.

### Trap-id reservations

- (none — auditing stage)

---

## Tooling Appendix (Post-v1.0)

These are NOT counted against the 30 stages but are essential for ecosystem maturity.

### LSP server
- `textDocument/publishDiagnostics` — wire kovc.hx parser/typecheck to LSP.
- Subsequent: hover, completion.

### Property-based testing harness
- `@property fn` — mark fn as property; test runner generates inputs.

### Coverage-guided fuzzing
- Compile with `-fcoverage`; fuzzer mutates inputs targeting uncovered branches.

### `///` doc-comment generator
- Extract `///` doc comments per fn; emit Markdown.

### Source maps (DWARF)
- Emit DWARF debug info during codegen; gdb integration.

---

## Cross-cutting concerns

### Self-host invariant (always)

After every stage commit, run `pytest helixc/tests/test_bootstrap_kovc.py` to verify K2 = compiler-of-compiler can rebuild itself byte-identically. Bootstrap cascade buffer is 1MB (fix landed 2026-05-07, commit 29f552e); current bootstrap source is ~290KB.

### Audit cadence

After each Stage commit batch:
1. Run multi-agent audit (3 agents).
2. Fix HIGH findings before next stage.
3. Log MEDIUM/LOW with rationale.

### TG update format (per `helix-tg-update-format.md`)

After each stage:
- Beginner-friendly English; no compiler jargon.
- Ordered "Next steps" list.
- Estimated % complete vs 30 stages + 7 amendments + tooling appendix + Stage 30 gate.

### Recursion-budget pattern (FLAT prefix-trap)

All new traps use:
```
let n_pre_trap = if cond { emit_trap_with_id(N) } else { 0 };
... existing body ...
n_existing_total + n_pre_trap
```

Never:
```
if cond { emit_trap_with_id(N) } else { ... existing body ... }
```

---

## Stage difficulty summary

| Stage | Difficulty | Title |
|------:|----------:|-------|
| 6  | 7  | Enums |
| 7  | 8  | Pattern matching |
| 8  | 8  | Generics + monomorphization |
| 8.5 | 7  | Traits |
| 9  | 7  | Closures |
| 10 | 5  | Modules + use |
| 11 | 9  | Reflection runtime |
| 12 | 8  | AD forward-mode |
| 13 | 7  | AD across user fns |
| 14 | 9  | AD reverse-mode |
| 14.5 | 7  | @checkpoint |
| 15 | 9  | Tile + tensor types |
| 16 | 9  | PTX backend |
| 16.5 | 7  | FFI / extern "C" |
| 17 | 6  | const-fold |
| 18 | 6  | CSE / DCE / FDCE |
| 19 | 6  | Effect check |
| 20 | 7  | Hash-cons (real SHA-256) |
| 21 | 6  | Total-by-default |
| 22 | 5  | Pretty errors |
| 23 | 4  | CLI flags |
| 24 | 9  | Provenance-typed neuro-symbolic |
| 25 | 7  | Trace introspection |
| 26 | 7  | Pytrees |
| 27 | 7  | Autotune |
| 28 | 8  | Parametric structs |
| 28.5 | 5  | Panic policy |
| 28.6 | 6  | Unsafe blocks |
| 28.7 | 4  | @deprecated |
| 29 | 8  | Drop helixc-Python |
| 30 | 9  | 5 clean audits |

**Highest difficulty (9/10)**: Stages 11 (reflection — runtime cell store + structural hashing + verifier-gated mutation), 14 (reverse-mode AD — adjoint propagation + multi-output return), 15 (tile/tensor lowering — substantial codegen surface), 16 (PTX backend — entire new emission target), 24 (provenance-typed neuro-symbolic — novel design with no Python reference), 30 (audit gate — open-ended verification volume).

**Most likely to need prerequisites not yet captured**:
- Stage 11 (reflection): structural hashing wants real SHA-256 (Stage 20) for non-collision; Phase-0 uses FNV-1a placeholder.
- Stage 14 (reverse-mode): requires Stage 5C (struct-by-value return) before; if Stage 5C slips, Stage 14 blocks.
- Stage 16 (PTX): without Stage 16.5 FFI, kernels can't be launched — so the stage produces emit-only artifacts. Recommend Stage 16.5 land before Stage 16 in practice.
- Stage 24 (provenance-typed): no Python reference; requires fresh design pass before implementation begins.
- Stage 26 (pytrees): requires Stage 14 reverse-mode + Stage 5 nested struct field access.
- Stage 27 (autotune): without Stage 16+16.5, no GPU kernels to autotune; stage would degrade to CPU-only autotuning.

### Sequencing recommendations (one possible order)

1. Iter B/C/D of Stage 5 (named field access, by-value pass, nested) — finish what's started.
2. Stage 6 → 7 (enums + match) — biggest quality-of-life jump.
3. Stage 8 → 8.5 (generics + traits) — unlocks much stdlib porting.
4. Stage 17 → 18 → 19 → 21 (passes — const-fold, CSE/DCE, effect, totality) — improves compile output before AD.
5. Stage 12 → 13 → 14 → 14.5 (AD foundation).
6. Stage 9 → 10 (closures + modules) — independent, can interleave.
7. Stage 11 → 20 (reflection + hash-cons).
8. Stage 16.5 → 16 → 15 → 27 (FFI → PTX → tile → autotune — GPU stack).
9. Stage 28 → 28.5 → 28.6 → 28.7 (parametric structs + safety/policy).
10. Stage 24 → 25 → 26 (Tier 3 strategic moat features).
11. Stage 22 → 23 (errors + CLI) — polish.
12. Stage 29 → 30 (parity + final audit gate).

---

This document is the canonical detailed plan; supersedes per-session re-derivations of stage scope. Update in place as stages complete — mark `done` in tag table; revise difficulty / commit estimates if reality diverges.
