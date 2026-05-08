# Stage 5: Structs — Design Notes

**Status**: queued. Stage 4 (tuples + arrays) complete. Stage 5 starts after parallelization-experiment subagents return.

## Plan summary

Stage 5 adds named-field aggregate types: `struct Foo { a: i32, b: i32 }`.

Per APPROACH_A_PLAN.md:
- struct decl: declare layout, allocate slot range
- Field access `foo.a`: read at offset
- By-value pass: callee gets struct flattened into N args
- Lit construction `Foo { a: 1, b: 2 }`

## Multi-iteration plan

### Iteration A: minimal struct decl + struct lit (positional field access)

**Scope**: parser-only changes. Reuse Stage 4 codegen.

1. **Lexer**: no change. `struct` lexed as IDENT (keyword recognition happens in parser via byte_eq).

2. **Parser top-level**: extend `parse_program` to accept `struct IDENT { ... }` in addition to `fn` decls.
   - Need to install `struct` keyword in `install_keywords` (next slot after kw_fn).
   - Add `kw_struct_s` / `kw_struct_n` accessors.

3. **Struct table**: add a struct_state region similar to fn_type_state. Each entry:
   - name byte ptr (i32)
   - name length (i32)
   - field count (i32)
   - field offset table base (i32) — points to a sequence of (name_ptr, name_len, type_tag) tuples in arena.

4. **AST_STRUCT_DECL (new tag, e.g., 54)**: parsed but treated as metadata. Codegen: emit nothing (the binary doesn't need the struct decl bytes).

5. **Struct lit parsing**: when parse_primary sees IDENT followed by `{`, look up struct_table:
   - Hit → parse `Foo { 1, 2 }` (positional values, no field names yet) → build AST_TUPLE_LIT (tag 50, REUSE existing codegen).
   - Miss → fall through to existing IDENT handling.

6. **Field access**: keep `.NUM` (positional) for now. `.NAME` deferred to Iter B.

7. **Test**: `struct Pt { x: i32, y: i32 } fn main() -> i32 { let p = Pt { 10, 32 }; p.0 + p.1 }` returns 42.

**Lines of code estimate**: 80-120 LOC across parser.hx + small kovc.hx for AST tag bookkeeping.

**Risks**:
- Parsing `IDENT { ... }` is ambiguous with `IDENT { stmt; expr }` block — must check struct_table for the IDENT.
- struct_state placement in bind_state: pick a free slot offset (current bn_state uses up to slot 83; structs could go at 84+).

### Iteration B: track struct types in bind_state + named field access

1. **bind_state extension**: when binding a struct-typed value, store the struct's id (index into struct_table) alongside the type tag.
2. **expr_type for AST_VAR**: returns a "struct-typed" tag with embedded struct id.
3. **Postfix `.NAME` parsing**: when `.` is followed by IDENT (not INT), look up the binding's struct type, find NAME's offset in the struct's field table, build AST_STRUCT_FIELD (tag 56) with the resolved offset.
4. **Codegen for AST_STRUCT_FIELD**: identical to AST_TUPLE_FIELD — `mov eax, [rax + offset]` (3 bytes).
5. **Test**: `struct Pt { x: i32, y: i32 } fn main() -> i32 { let p = Pt { 10, 32 }; p.x + p.y }` returns 42.

### Iteration C: by-value struct pass to fn calls

1. struct-typed fn params: callee receives struct flattened into N args (Rust-style).
2. SysV ABI: structs ≤ 16 bytes pass in registers (rdi, rsi); larger pass via memory.
3. Codegen: caller pushes/copies fields into arg slots; callee reassembles via stack region.
4. **Test**: pass a Pt to a fn that returns `p.x + p.y`.

### Iteration D: nested structs

1. Struct fields can be other struct types.
2. Layout: nested struct's bytes inlined into parent's slot range.
3. **Test**: `struct Line { from: Pt, to: Pt }` and access `l.from.x`.

## Open questions

- **Recursive structs**: not in Phase 0. Defer to a later phase.
- **Generic structs**: Stage 8 (generics) handles this.
- **Default field values**: not in Phase 0.
- **Anonymous structs**: not in Phase 0.

## Codegen reuse summary

| AST | Tag | Iter | Codegen |
|---|---|---|---|
| AST_STRUCT_DECL | 54 | A | none (metadata only) |
| AST_STRUCT_LIT | 50 (alias) | A | reuse AST_TUPLE_LIT |
| AST_STRUCT_FIELD | 56 | B | mirror AST_TUPLE_FIELD |
| AST_STRUCT_PARAM | 55 | C | new — flatten/reassemble |

## Python helixc CSE bug awareness

The Iter C codegen for "flatten struct into args" might use `let mut idx * const` patterns. If so, apply the same `off += stride` workaround as in tuple LIT (see `runtime/memory/semantic/helixc-python-cse-loop-variant-bug.md`).

## Parallelism notes

Stage 5 work touches `helixc/bootstrap/parser.hx` and `helixc/bootstrap/kovc.hx`. While subagents are running on:
- Python helixc optimization (helixc/frontend/*.py, helixc/ir/*.py)
- Audit doc (docs/audit-stage4-followup.md)

these don't conflict with Stage 5. Main thread can land Stage 5 Iter A in parallel.
