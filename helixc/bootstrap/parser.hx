// Stage-2 parser for the Helix bootstrap compiler.
//
// Consumes the token stream emitted by stage-1 lexer and builds an
// AST in the arena. Each AST node is a 4-slot record:
//
//   [tag, p1, p2, p3]
//
// AST tags (Phase 0 — minimal subset that already powers our
// metacircular evaluator demo):
//
//   0  AST_INT       p1 = literal value
//   1  AST_VAR       p1 = source byte index, p2 = byte length
//   2  AST_ADD       p1 = lhs node idx, p2 = rhs
//   3  AST_SUB       ditto
//   4  AST_MUL       ditto
//   5  AST_DIV       ditto
//   6  AST_LT        ditto (returns reified 0/1)
//   7  AST_IF        p1 = cond, p2 = then, p3 = else
//   8  AST_LET       p1 = name byte index, p2 = name length,
//                    p3 = packed (value_idx * 65536 + body_idx)
//   9  AST_NEG       p1 = inner
//  10  AST_WHILE     p1 = cond, p2 = body. Always returns 0.
//  11  AST_ASSIGN    p1 = name byte start, p2 = name length,
//                    p3 = value_idx. Stores eax to the binding's
//                    stack slot; result IS the assigned value.
//  12  AST_LET_MUT   same payload shape as AST_LET; codegen treats
//                    them identically. Distinct tag preserved for
//                    future static analysis (e.g. mutability check).
//  13  AST_SEQ       p1 = first_idx, p2 = second_idx. Evaluate
//                    first (discard), then second (return its value).
//                    Built by `;` chaining inside parse_expr.
//  14  AST_FN_DECL   p1 = name byte start, p2 = name byte length,
//                    p3 = body_idx. Phase-0: no params, return-type
//                    annotation parsed but ignored, body is a single
//                    expression. Codegen treats the body as the
//                    main expression.
//  15  AST_FN_LIST   p1 = current fn_decl_idx, p2 = next list node
//                    idx (or 0 at end). Linked list of top-level fn
//                    declarations. Built by parse_top when source
//                    has multiple `fn ... { ... }` items.
//  16  AST_CALL      p1 = name byte start, p2 = name byte length,
//                    p3 = args_head_idx (linked list of AST_ARG
//                    nodes), or 0 if no args. Detected by
//                    parse_primary when IDENT is followed by `(`.
//  17  AST_ARG       p1 = expr_idx (the arg's value expression),
//                    p2 = next_arg_idx (or 0). Linked-list element
//                    used by AST_CALL.
//  18  AST_PARAM     p1 = name_start, p2 = name_len, p3 = next_param_idx,
//                    p4 = type_tag (Phase 1.10 step 5c follow-on:
//                    0 = i32 default, 1 = f32 if annotation was `: f32`).
//                    The codegen reads p4 to call bind_push_typed so
//                    f32 params propagate through is_f32_expr to SSE.
//                    Linked list of fn decl params. Stored at the
//                    head index referenced by AST_FN_DECL.p3 (packed
//                    with body_idx the same way AST_LET does).
//  19  AST_GT        p1 = lhs, p2 = rhs.  result = (lhs > rhs ? 1 : 0)
//  20  AST_EQ        p1 = lhs, p2 = rhs.  result = (lhs == rhs ? 1 : 0)
//  21  AST_NE        p1 = lhs, p2 = rhs.  result = (lhs != rhs ? 1 : 0)
//  22  AST_LE        p1 = lhs, p2 = rhs.  result = (lhs <= rhs ? 1 : 0)
//  23  AST_GE        p1 = lhs, p2 = rhs.  result = (lhs >= rhs ? 1 : 0)
//  25  AST_STR_LIT   p1 = body byte_start, p2 = body byte_len.
//  26  AST_BNOT      p1 = inner. Bitwise NOT (`not eax`). Mirrors helixc-Python
//                    OpKind.BIT_NOT (commit 4e6b4fa).
//                    Phase-0: as a value, lowers to mov eax, 0.
//                    Recognized as the first arg of read_file_to_arena
//                    or write_file_to_arena, where the body bytes get
//                    embedded in the produced binary's .data section.
//  28  AST_BAND      p1 = lhs, p2 = rhs. Binary bitwise AND. Codegen
//                    emits `and eax, ecx` (0x21 0xC8). Mirrors
//                    helixc-Python OpKind.BIT_AND (commit f676fca).
//  29  AST_BOR       p1 = lhs, p2 = rhs. `or eax, ecx` (0x09 0xC8).
//  30  AST_BXOR      p1 = lhs, p2 = rhs. `xor eax, ecx` (0x31 0xC8).
//  32  AST_SHL       p1 = lhs, p2 = rhs. `shl eax, cl` (0xD3 0xE0).
//  33  AST_SHR       p1 = lhs, p2 = rhs. `sar eax, cl` (0xD3 0xF8) —
//                    arithmetic shift right, preserves sign for signed i32.
//                    Mirrors helixc-Python OpKind.SHL/SHR (commit 1410f91).
//  31  AST_NOT       p1 = inner. Logical NOT. Codegen emits
//                    `test eax, eax; mov eax, 0; sete al` so the
//                    result is 1 when inner == 0, else 0. Mirrors
//                    helixc-Python: `!x` lowers to CMP_EQ(inner, 0).
//  99  AST_ERR       p1 = unexpected token tag
//
// Grammar (recursive descent, classic precedence climbing):
//   expr     := add ("<" add)?
//   add      := mul (("+" | "-") mul)*
//   mul      := unary (("*" | "/") unary)*
//   unary    := "-" unary | primary
//   primary  := INT | IDENT | "(" expr ")" | if-expr | let-expr
//   if-expr  := "if" expr "{" expr "}" "else" "{" expr "}"
//   let-expr := "let" IDENT "=" expr ";" expr
//
// SysV ABI on x86-64 limits the codegen to 6 int params, so we
// stash all parser state in a contiguous arena region and pass
// only (tok_base, state_base) to every parser function:
//
//   state_base+0   cursor (current token index)
//   state_base+1   kw_let_start
//   state_base+2   kw_let_len
//   state_base+3   kw_if_start
//   state_base+4   kw_if_len
//   state_base+5   kw_else_start
//   state_base+6   kw_else_len
//   state_base+7   kw_while_start
//   state_base+8   kw_while_len
//   state_base+9   kw_mut_start
//   state_base+10  kw_mut_len
//   state_base+11  kw_fn_start
//   state_base+12  kw_fn_len
//
// License: Apache 2.0.

// --------------------------------------------------------------
// Token-stream helpers. Tokens are 4 slots each; index k -> slot
// tok_base + k*4.
// --------------------------------------------------------------
@pure fn tok_tag(tok_base: i32, k: i32) -> i32 { __arena_get(tok_base + k * 4) }
@pure fn tok_p1(tok_base: i32, k: i32) -> i32  { __arena_get(tok_base + k * 4 + 1) }
@pure fn tok_p2(tok_base: i32, k: i32) -> i32  { __arena_get(tok_base + k * 4 + 2) }
@pure fn tok_p3(tok_base: i32, k: i32) -> i32  { __arena_get(tok_base + k * 4 + 3) }

// State accessors.
fn cur_get(sb: i32) -> i32 { __arena_get(sb) }
fn cur_set(sb: i32, v: i32) -> i32 { __arena_set(sb, v); 0 }
fn cur_advance(sb: i32) -> i32 { let c = cur_get(sb); cur_set(sb, c + 1); 0 }
fn kw_let_s(sb: i32) -> i32  { __arena_get(sb + 1) }
fn kw_let_n(sb: i32) -> i32  { __arena_get(sb + 2) }
fn kw_if_s(sb: i32) -> i32   { __arena_get(sb + 3) }
fn kw_if_n(sb: i32) -> i32   { __arena_get(sb + 4) }
fn kw_else_s(sb: i32) -> i32 { __arena_get(sb + 5) }
fn kw_else_n(sb: i32) -> i32 { __arena_get(sb + 6) }
fn kw_while_s(sb: i32) -> i32 { __arena_get(sb + 7) }
fn kw_while_n(sb: i32) -> i32 { __arena_get(sb + 8) }
fn kw_mut_s(sb: i32) -> i32 { __arena_get(sb + 9) }
fn kw_mut_n(sb: i32) -> i32 { __arena_get(sb + 10) }
fn kw_fn_s(sb: i32) -> i32 { __arena_get(sb + 11) }
fn kw_fn_n(sb: i32) -> i32 { __arena_get(sb + 12) }
// Stage 5: struct keyword installed at sb+13/sb+14.
fn kw_struct_s(sb: i32) -> i32 { __arena_get(sb + 13) }
fn kw_struct_n(sb: i32) -> i32 { __arena_get(sb + 14) }
// Stage 5: struct_table state — sb+15 = arena base offset of the
// 12-slot region (3 entries x 4 fields), sb+16 = registered count.
fn struct_tab_base(sb: i32) -> i32 { __arena_get(sb + 15) }
fn struct_tab_count(sb: i32) -> i32 { __arena_get(sb + 16) }
// Stage 5 Iter B: var-to-struct binding table — sb+17 = base offset,
// sb+18 = count. Each entry is 3 slots (var_name_s, var_name_l,
// struct_idx). Cap 4 vars in Iter B; expand later. Used so that when
// parse_primary's postfix branch sees `varname.IDENT`, it can resolve
// IDENT to a numeric field offset via struct_tab_field_lookup.
fn var_struct_tab_base(sb: i32) -> i32 { __arena_get(sb + 17) }
fn var_struct_tab_count(sb: i32) -> i32 { __arena_get(sb + 18) }
// sb+19 = "last_struct_idx" scratch slot. parse_struct_lit writes this
// when it produces a struct lit; the surrounding let parser reads then
// clears it (-1 = none) to associate the bound name with a struct id.
fn last_struct_idx(sb: i32) -> i32 { __arena_get(sb + 19) }
fn set_last_struct_idx(sb: i32, v: i32) -> i32 { __arena_set(sb + 19, v); 0 }
// Stage 6: enum_table state — sb+20 = arena base offset of the enum
// region, sb+21 = registered count. Each entry is 5 slots
// (name_s, name_l, variant_count, variants_ptr, max_payload_arity).
// Cap 4 enums for now; expand later.
fn enum_tab_base(sb: i32) -> i32 { __arena_get(sb + 20) }
fn enum_tab_count(sb: i32) -> i32 { __arena_get(sb + 21) }
// Stage 6: var-to-enum binding table — sb+22 = base offset, sb+23 =
// count. Each entry is 3 slots (var_name_s, var_name_l, enum_idx).
// Cap 4 vars in 6A; expand later. Used so `let m = Maybe::Some(...)`
// can later resolve `m`'s enum_idx for typed dispatch.
fn var_enum_tab_base(sb: i32) -> i32 { __arena_get(sb + 22) }
fn var_enum_tab_count(sb: i32) -> i32 { __arena_get(sb + 23) }
// Stage 6: scratch slot — sb+24 = "last_enum_idx" written by
// parse_primary's enum-construct branch when it produces a value;
// surrounding let-parser reads then clears (-1 = none).
fn last_enum_idx(sb: i32) -> i32 { __arena_get(sb + 24) }
fn set_last_enum_idx(sb: i32, v: i32) -> i32 { __arena_set(sb + 24, v); 0 }
// Stage 6: enum keyword installed at sb+25/sb+26.
fn kw_enum_s(sb: i32) -> i32 { __arena_get(sb + 25) }
fn kw_enum_n(sb: i32) -> i32 { __arena_get(sb + 26) }
// Stage 7: match keyword installed at sb+27/sb+28.
fn kw_match_s(sb: i32) -> i32 { __arena_get(sb + 27) }
fn kw_match_n(sb: i32) -> i32 { __arena_get(sb + 28) }
// Stage 8: generic-params scratch table for the CURRENT fn being parsed.
// sb+29 = base (offset of 8-slot region: 4 entries x 2 fields name_s,name_l).
// sb+30 = count (0..4). Reset to 0 by parse_fn_decl when entering, set
// while parsing `<T1, T2, ...>`, used during AST_PARAM type resolution
// to mark generic-typed params with type_tag = 200 + idx.
fn gp_tab_base(sb: i32) -> i32 { __arena_get(sb + 29) }
fn gp_tab_count(sb: i32) -> i32 { __arena_get(sb + 30) }
fn gp_tab_reset(sb: i32) -> i32 { __arena_set(sb + 30, 0); 0 }
fn gp_tab_add(sb: i32, name_s: i32, name_l: i32) -> i32 {
    let count = gp_tab_count(sb);
    if count >= 4 {
        0 - 1
    } else {
        let base = gp_tab_base(sb);
        let entry = base + count * 2;
        __arena_set(entry, name_s);
        __arena_set(entry + 1, name_l);
        __arena_set(sb + 30, count + 1);
        count
    }
}
// Lookup by IDENT bytes; return 0..3 (the param idx) on hit, -1 on miss.
fn gp_tab_lookup(sb: i32, name_s: i32, name_l: i32) -> i32 {
    let base = gp_tab_base(sb);
    let count = gp_tab_count(sb);
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < count {
        let entry = base + i * 2;
        let ns = __arena_get(entry);
        let nl = __arena_get(entry + 1);
        if byte_eq(name_s, name_l, ns, nl) == 1 {
            found = i;
            i = count;
        } else {
            i = i + 1;
        };
    }
    found
}
// Stage 8: mono-instantiation request table. Entries pushed by turbofish-
// at-call-site code in parse_primary; consumed at end of parse_program
// to synthesize cloned AST_FN_DECL nodes with concrete-type substitution.
// sb+31 = base (offset of 192-slot region: 32 entries x 6 fields).
// sb+32 = count (0..32).
// Entry layout (6 slots):
//   slot 0: orig_name_s
//   slot 1: orig_name_l
//   slot 2: mangled_name_s (in arena)
//   slot 3: mangled_name_l
//   slot 4: type_args_packed (4 bits per arg, up to 6 args)
//   slot 5: type_args_count (0..6)
fn mr_tab_base(sb: i32) -> i32 { __arena_get(sb + 31) }
fn mr_tab_count(sb: i32) -> i32 { __arena_get(sb + 32) }
fn mr_tab_add(sb: i32, orig_s: i32, orig_l: i32, mang_s: i32, mang_l: i32, packed: i32, ta_count: i32) -> i32 {
    let count = mr_tab_count(sb);
    if count >= 32 {
        0 - 1
    } else {
        let base = mr_tab_base(sb);
        let entry = base + count * 6;
        __arena_set(entry, orig_s);
        __arena_set(entry + 1, orig_l);
        __arena_set(entry + 2, mang_s);
        __arena_set(entry + 3, mang_l);
        __arena_set(entry + 4, packed);
        __arena_set(entry + 5, ta_count);
        __arena_set(sb + 32, count + 1);
        count
    }
}
// Lookup an instantiation by (orig_name, type_args_packed, ta_count).
// Returns the entry idx on hit (so caller can re-use mangled name), -1 on miss.
fn mr_tab_lookup(sb: i32, orig_s: i32, orig_l: i32, packed: i32, ta_count: i32) -> i32 {
    let base = mr_tab_base(sb);
    let count = mr_tab_count(sb);
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < count {
        let entry = base + i * 6;
        let ns = __arena_get(entry);
        let nl = __arena_get(entry + 1);
        let p = __arena_get(entry + 4);
        let c = __arena_get(entry + 5);
        if byte_eq(orig_s, orig_l, ns, nl) == 1 {
            if p == packed {
                if c == ta_count {
                    found = i;
                    i = count;
                } else { i = i + 1; }
            } else { i = i + 1; }
        } else {
            i = i + 1;
        };
    }
    found
}
fn var_struct_tab_add(sb: i32, name_s: i32, name_l: i32, struct_idx: i32) -> i32 {
    let count = var_struct_tab_count(sb);
    if count >= 4 {
        0 - 1
    } else {
        let base = var_struct_tab_base(sb);
        let entry = base + count * 3;
        __arena_set(entry, name_s);
        __arena_set(entry + 1, name_l);
        __arena_set(entry + 2, struct_idx);
        __arena_set(sb + 18, count + 1);
        count
    }
}
fn var_struct_tab_lookup(sb: i32, name_s: i32, name_l: i32) -> i32 {
    let base = var_struct_tab_base(sb);
    let count = var_struct_tab_count(sb);
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < count {
        let entry = base + i * 3;
        let ns = __arena_get(entry);
        let nl = __arena_get(entry + 1);
        if byte_eq(name_s, name_l, ns, nl) == 1 {
            found = __arena_get(entry + 2);
            i = count;
        } else {
            i = i + 1;
        };
    }
    found
}
// Append an entry. Returns the new index (0..2) on success, -1 on
// overflow. Iter A cap is 3 structs; expand later if needed.
// Iter B: 4-slot stride (name_s, name_l, arity, fields_ptr).
// fields_ptr is the arena offset of a (name_s, name_l) pair sequence
// (2*arity slots). 0 means no fields region (e.g. empty struct).
fn struct_tab_add(sb: i32, name_s: i32, name_l: i32, arity: i32, fields_ptr: i32) -> i32 {
    let count = struct_tab_count(sb);
    if count >= 3 {
        0 - 1
    } else {
        let base = struct_tab_base(sb);
        let entry = base + count * 4;
        __arena_set(entry, name_s);
        __arena_set(entry + 1, name_l);
        __arena_set(entry + 2, arity);
        __arena_set(entry + 3, fields_ptr);
        __arena_set(sb + 16, count + 1);
        count
    }
}
// Look up a struct by name. Returns the recorded arity (>= 0) on hit,
// or -1 on miss. Used by parse_primary to detect `IDENT { ... }` as
// a struct literal vs a regular IDENT/block.
fn struct_tab_lookup(sb: i32, name_s: i32, name_l: i32) -> i32 {
    let base = struct_tab_base(sb);
    let count = struct_tab_count(sb);
    let mut i: i32 = 0;
    let mut found_arity: i32 = 0 - 1;
    while i < count {
        let entry = base + i * 4;
        let ns = __arena_get(entry);
        let nl = __arena_get(entry + 1);
        if byte_eq(name_s, name_l, ns, nl) == 1 {
            found_arity = __arena_get(entry + 2);
            i = count;
        } else {
            i = i + 1;
        };
    }
    found_arity
}
// Iter B: same as struct_tab_lookup but returns the entry INDEX
// (0..count-1) instead of arity. -1 on miss. Needed so callers can
// then drill into fields_ptr / arity at entry+2,+3.
fn struct_tab_lookup_idx(sb: i32, name_s: i32, name_l: i32) -> i32 {
    let base = struct_tab_base(sb);
    let count = struct_tab_count(sb);
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < count {
        let entry = base + i * 4;
        let ns = __arena_get(entry);
        let nl = __arena_get(entry + 1);
        if byte_eq(name_s, name_l, ns, nl) == 1 {
            found = i;
            i = count;
        } else {
            i = i + 1;
        };
    }
    found
}
// Iter B: given a struct's table index and a field name, return the
// 0-based field index, or -1 on miss / no fields region.
// Iter D: stride extended from 2 to 3 (name_s, name_l, field_struct_idx).
// field_struct_idx is the struct_idx of the field's type if it is a
// registered struct, or -1 if the field is a scalar (i32/f32/etc.).
fn struct_tab_field_lookup(sb: i32, struct_idx: i32, field_s: i32, field_l: i32) -> i32 {
    let base = struct_tab_base(sb);
    let entry = base + struct_idx * 4;
    let arity = __arena_get(entry + 2);
    let fields_ptr = __arena_get(entry + 3);
    if fields_ptr == 0 {
        0 - 1
    } else {
        let mut i: i32 = 0;
        let mut found: i32 = 0 - 1;
        while i < arity {
            let pair = fields_ptr + i * 3;
            let ns = __arena_get(pair);
            let nl = __arena_get(pair + 1);
            if byte_eq(field_s, field_l, ns, nl) == 1 {
                found = i;
                i = arity;
            } else {
                i = i + 1;
            };
        }
        found
    }
}

// Stage 6: enum_table append. Returns the new index (0..3) on success,
// -1 on overflow. Cap 4 enums. Stride 5 (name_s, name_l, variant_count,
// variants_ptr, max_payload_arity).
fn enum_tab_add(sb: i32, name_s: i32, name_l: i32, variant_count: i32, variants_ptr: i32, max_arity: i32) -> i32 {
    let count = enum_tab_count(sb);
    if count >= 4 {
        0 - 1
    } else {
        let base = enum_tab_base(sb);
        let entry = base + count * 5;
        __arena_set(entry, name_s);
        __arena_set(entry + 1, name_l);
        __arena_set(entry + 2, variant_count);
        __arena_set(entry + 3, variants_ptr);
        __arena_set(entry + 4, max_arity);
        __arena_set(sb + 21, count + 1);
        count
    }
}

// Stage 6: look up an enum by name. Returns the entry index on hit, -1
// on miss. Used by parse_primary to detect `IDENT::` as an enum-variant
// path. Both struct_tab and enum_tab share IDENT namespace; struct_tab
// is checked first (via existing IDENT { ... } path) but `::` is unique
// to enums in Phase 0 so no ambiguity.
fn enum_tab_lookup_idx(sb: i32, name_s: i32, name_l: i32) -> i32 {
    let base = enum_tab_base(sb);
    let count = enum_tab_count(sb);
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < count {
        let entry = base + i * 5;
        let ns = __arena_get(entry);
        let nl = __arena_get(entry + 1);
        if byte_eq(name_s, name_l, ns, nl) == 1 {
            found = i;
            i = count;
        } else {
            i = i + 1;
        };
    }
    found
}

// Stage 6: variant table entries are 4 slots/variant
// (name_s, name_l, arity, discriminant). Look up a variant by name on
// a given enum_idx. Returns the variant's discriminant on hit, -1 on
// miss. Reads variant_count from enum entry's slot+2.
fn enum_tab_variant_lookup_disc(sb: i32, enum_idx: i32, vname_s: i32, vname_l: i32) -> i32 {
    let base = enum_tab_base(sb);
    let entry = base + enum_idx * 5;
    let vcount = __arena_get(entry + 2);
    let vptr = __arena_get(entry + 3);
    if vptr == 0 {
        0 - 1
    } else {
        let mut i: i32 = 0;
        let mut found: i32 = 0 - 1;
        while i < vcount {
            let ent = vptr + i * 4;
            let ns = __arena_get(ent);
            let nl = __arena_get(ent + 1);
            if byte_eq(vname_s, vname_l, ns, nl) == 1 {
                found = __arena_get(ent + 3);
                i = vcount;
            } else {
                i = i + 1;
            };
        }
        found
    }
}

// Stage 6: same lookup but returns the variant's arity (0 = unit, >=1
// = payload variant). -1 on miss.
fn enum_tab_variant_lookup_arity(sb: i32, enum_idx: i32, vname_s: i32, vname_l: i32) -> i32 {
    let base = enum_tab_base(sb);
    let entry = base + enum_idx * 5;
    let vcount = __arena_get(entry + 2);
    let vptr = __arena_get(entry + 3);
    if vptr == 0 {
        0 - 1
    } else {
        let mut i: i32 = 0;
        let mut found: i32 = 0 - 1;
        while i < vcount {
            let ent = vptr + i * 4;
            let ns = __arena_get(ent);
            let nl = __arena_get(ent + 1);
            if byte_eq(vname_s, vname_l, ns, nl) == 1 {
                found = __arena_get(ent + 2);
                i = vcount;
            } else {
                i = i + 1;
            };
        }
        found
    }
}

// Stage 6: register a var->enum_idx binding. Returns 0 on success,
// -1 on overflow.
fn var_enum_tab_add(sb: i32, name_s: i32, name_l: i32, enum_idx: i32) -> i32 {
    let count = var_enum_tab_count(sb);
    if count >= 4 {
        0 - 1
    } else {
        let base = var_enum_tab_base(sb);
        let entry = base + count * 3;
        __arena_set(entry, name_s);
        __arena_set(entry + 1, name_l);
        __arena_set(entry + 2, enum_idx);
        __arena_set(sb + 23, count + 1);
        count
    }
}

// Stage 6: look up a var name in var_enum_tab. Returns enum_idx on hit,
// -1 on miss.
fn var_enum_tab_lookup(sb: i32, name_s: i32, name_l: i32) -> i32 {
    let base = var_enum_tab_base(sb);
    let count = var_enum_tab_count(sb);
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < count {
        let entry = base + i * 3;
        let ns = __arena_get(entry);
        let nl = __arena_get(entry + 1);
        if byte_eq(name_s, name_l, ns, nl) == 1 {
            found = __arena_get(entry + 2);
            i = count;
        } else {
            i = i + 1;
        };
    }
    found
}

// Iter D: given a struct's table index and a field index, return the
// field's struct_idx if its declared type is a registered struct, or
// -1 if scalar / out of range / no fields region.
fn struct_tab_field_struct_idx(sb: i32, struct_idx: i32, field_idx: i32) -> i32 {
    let base = struct_tab_base(sb);
    let entry = base + struct_idx * 4;
    let arity = __arena_get(entry + 2);
    let fields_ptr = __arena_get(entry + 3);
    if fields_ptr == 0 {
        0 - 1
    } else { if field_idx < 0 {
        0 - 1
    } else { if field_idx >= arity {
        0 - 1
    } else {
        let pair = fields_ptr + field_idx * 3;
        __arena_get(pair + 2)
    }}}
}

// --------------------------------------------------------------
// AST builder.
// --------------------------------------------------------------
fn mk_node(tag: i32, p1: i32, p2: i32, p3: i32) -> i32 {
    let i = __arena_push(tag);
    __arena_push(p1);
    __arena_push(p2);
    __arena_push(p3);
    i
}

// --------------------------------------------------------------
// Compare two byte-spans in the arena for equality.
// --------------------------------------------------------------
@pure
fn byte_eq(src_a: i32, len_a: i32, src_b: i32, len_b: i32) -> i32 {
    if len_a != len_b { 0 }
    else {
        let mut i: i32 = 0;
        let mut ok: i32 = 1;
        while i < len_a {
            if ok == 1 {
                let ba = __arena_get(src_a + i);
                let bb = __arena_get(src_b + i);
                if ba != bb { ok = 0; };
            };
            i = i + 1;
        }
        ok
    }
}

// --------------------------------------------------------------
// Forward-style state-passing parser. Each function takes only
// tok_base + state_base; arena slots store the rest.
// --------------------------------------------------------------

// `parse_expr` is the public entry that chains expressions with the
// sequencing operator `;`. Each segment between `;`s is parsed by
// `parse_expr_basic`. Right-associative: `a ; b ; c` becomes
// AST_SEQ(a, AST_SEQ(b, c)). Evaluation order: a, b, c (left-to-right);
// final value is c.
//
// `parse_expr_basic` is the place to call when the caller does NOT
// want sequencing — e.g., the value position of a let-binding or
// assignment, where `;` is the let-terminator, not a sequencer.
fn parse_expr(tok_base: i32, sb: i32) -> i32 {
    let first = parse_expr_basic(tok_base, sb);
    let k = cur_get(sb);
    let kt = tok_tag(tok_base, k);
    // Audit-15: implicit `;` after a statement-like expression
    // whose result is a `}` block. Specifically: AST_WHILE (10),
    // AST_IF (7), AST_LET (8), AST_LET_MUT (12) — these chain into
    // the next expression even without an explicit semicolon. This
    // matches surface-Helix semantics; without it, the bootstrap
    // source's many `while ... { ... } <expr>` patterns split into
    // two unrelated expressions and the latter falls off the parser.
    let first_tag = __arena_get(first);
    let first_is_block = if first_tag == 10 { 1 }
        else { if first_tag == 7 { 1 }
        else { if first_tag == 8 { 1 }
        else { if first_tag == 12 { 1 } else { 0 }}}};
    if kt == 12 {     // 12 = TK_SEMI
        cur_advance(sb);
        // Don't chain `;` if the next token signals end-of-block
        // (the `;` was just a terminator after a statement-like
        // expression). End-of-block tokens: `}` (6), EOF (0), `)` (4).
        let nk = cur_get(sb);
        let nt = tok_tag(tok_base, nk);
        if nt == 0 {
            first
        } else { if nt == 6 {
            first
        } else { if nt == 4 {
            first
        } else {
            let rest = parse_expr(tok_base, sb);
            mk_node(13, first, rest, 0)
        }}}
    } else { if first_is_block == 1 {
        // No explicit `;` but `first` is a statement-block.
        // Implicitly chain with the next expression unless we're
        // at end-of-block (`}`/EOF/`)`).
        if kt == 0 {
            first
        } else { if kt == 6 {
            first
        } else { if kt == 4 {
            first
        } else {
            let rest = parse_expr(tok_base, sb);
            mk_node(13, first, rest, 0)
        }}}
    } else {
        first
    }}
}

fn parse_expr_basic(tok_base: i32, sb: i32) -> i32 {
    let lhs = parse_bitwise(tok_base, sb);
    let k = cur_get(sb);
    let t = tok_tag(tok_base, k);
    let t2 = tok_tag(tok_base, k + 1);
    // Token tags: 15='=', 16='<', 17='>', 18='!'.
    // Compound comparisons require the next char to be `=`.
    if t == 16 {
        if t2 == 15 {
            // `<=`
            cur_advance(sb); cur_advance(sb);
            let rhs = parse_bitwise(tok_base, sb);
            mk_node(22, lhs, rhs, 0)
        } else {
            // `<`
            cur_advance(sb);
            let rhs = parse_bitwise(tok_base, sb);
            mk_node(6, lhs, rhs, 0)
        }
    } else { if t == 17 {
        if t2 == 15 {
            // `>=`
            cur_advance(sb); cur_advance(sb);
            let rhs = parse_bitwise(tok_base, sb);
            mk_node(23, lhs, rhs, 0)
        } else {
            // `>`
            cur_advance(sb);
            let rhs = parse_bitwise(tok_base, sb);
            mk_node(19, lhs, rhs, 0)
        }
    } else { if t == 15 {
        if t2 == 15 {
            // `==`
            cur_advance(sb); cur_advance(sb);
            let rhs = parse_bitwise(tok_base, sb);
            mk_node(20, lhs, rhs, 0)
        } else { lhs }
    } else { if t == 18 {
        if t2 == 15 {
            // `!=`
            cur_advance(sb); cur_advance(sb);
            let rhs = parse_bitwise(tok_base, sb);
            mk_node(21, lhs, rhs, 0)
        } else { lhs }
    } else { lhs }}}}
}

// Phase 1.10 step 5+: binary bitwise AND/OR/XOR at one precedence level
// between additive and comparison. Not strictly C-correct (C separates
// & ^ | into three levels) but enough for AGI substrate work where most
// callers use parens. Left-associative.
fn parse_bitwise(tok_base: i32, sb: i32) -> i32 {
    let mut lhs = parse_add(tok_base, sb);
    let mut keep: i32 = 1;
    while keep == 1 {
        let k = cur_get(sb);
        let t = tok_tag(tok_base, k);
        if t == 27 {       // TK_AMP -> AST_BAND
            cur_advance(sb);
            let rhs = parse_add(tok_base, sb);
            lhs = mk_node(28, lhs, rhs, 0);
        } else { if t == 28 {       // TK_PIPE -> AST_BOR
            cur_advance(sb);
            let rhs = parse_add(tok_base, sb);
            lhs = mk_node(29, lhs, rhs, 0);
        } else { if t == 29 {       // TK_CARET -> AST_BXOR
            cur_advance(sb);
            let rhs = parse_add(tok_base, sb);
            lhs = mk_node(30, lhs, rhs, 0);
        } else { if t == 30 {       // TK_LSHIFT -> AST_SHL
            cur_advance(sb);
            let rhs = parse_add(tok_base, sb);
            lhs = mk_node(32, lhs, rhs, 0);
        } else { if t == 31 {       // TK_RSHIFT -> AST_SHR
            cur_advance(sb);
            let rhs = parse_add(tok_base, sb);
            lhs = mk_node(33, lhs, rhs, 0);
        } else {
            keep = 0;
        }}}}};
    }
    lhs
}

fn parse_add(tok_base: i32, sb: i32) -> i32 {
    let mut lhs = parse_mul(tok_base, sb);
    let mut keep: i32 = 1;
    while keep == 1 {
        let k = cur_get(sb);
        let t = tok_tag(tok_base, k);
        if t == 7 {
            cur_advance(sb);
            let rhs = parse_mul(tok_base, sb);
            lhs = mk_node(2, lhs, rhs, 0);
        } else { if t == 8 {
            cur_advance(sb);
            let rhs = parse_mul(tok_base, sb);
            lhs = mk_node(3, lhs, rhs, 0);
        } else {
            keep = 0;
        }};
    }
    lhs
}

fn parse_mul(tok_base: i32, sb: i32) -> i32 {
    let mut lhs = parse_unary(tok_base, sb);
    let mut keep: i32 = 1;
    while keep == 1 {
        let k = cur_get(sb);
        let t = tok_tag(tok_base, k);
        if t == 9 {
            cur_advance(sb);
            let rhs = parse_unary(tok_base, sb);
            lhs = mk_node(4, lhs, rhs, 0);
        } else { if t == 10 {
            cur_advance(sb);
            let rhs = parse_unary(tok_base, sb);
            lhs = mk_node(5, lhs, rhs, 0);
        } else { if t == 11 {
            // Modulo (`%`). AST_MOD = tag 24 (chosen to avoid the
            // existing 19-23 comparison range; codegen handler in
            // kovc.hx maps it to idiv + remainder-in-edx).
            cur_advance(sb);
            let rhs = parse_unary(tok_base, sb);
            lhs = mk_node(24, lhs, rhs, 0);
        } else {
            keep = 0;
        }}};
    }
    lhs
}

fn parse_unary(tok_base: i32, sb: i32) -> i32 {
    let k = cur_get(sb);
    let tg = tok_tag(tok_base, k);
    if tg == 8 {     // unary minus
        cur_advance(sb);
        let inner = parse_unary(tok_base, sb);
        mk_node(9, inner, 0, 0)
    } else { if tg == 23 {     // '~' bitwise NOT
        cur_advance(sb);
        let inner = parse_unary(tok_base, sb);
        mk_node(26, inner, 0, 0)
    } else { if tg == 18 {     // '!' logical NOT — AST_NOT (tag 31).
        cur_advance(sb);
        let inner = parse_unary(tok_base, sb);
        mk_node(31, inner, 0, 0)
    } else {
        // Stage 4 iter B + E: postfix tuple field access AND array index.
        //   .NUM         → AST_TUPLE_FIELD (tag 52, static idx).
        //   [idx_expr]   → AST_INDEX (tag 53, dynamic idx).
        // Stage 5 Iter D: chained `.IDENT.IDENT` for nested structs.
        //   Track cur_struct_idx through the chain: starts at the LHS
        //   var's struct_idx; after each `.IDENT` whose field is a
        //   struct, update to that field's struct_idx (and emit
        //   AST_TUPLE_FIELD with p3 == 1 — codegen reads the slot as
        //   an 8-byte child pointer instead of a 4-byte i32); else
        //   emit p3 == 0 (4-byte read) and reset cur_struct_idx to -1,
        //   which makes any further `.IDENT` bail.
        let mut prim = parse_primary(tok_base, sb);
        let mut cur_struct_idx: i32 = 0 - 1;
        let mut keep_p: i32 = 1;
        while keep_p == 1 {
            let pk = cur_get(sb);
            let pt = tok_tag(tok_base, pk);
            if pt == 22 {                              // TK_DOT
                let nt = tok_tag(tok_base, pk + 1);
                if nt == 1 {                           // TK_INT
                    cur_advance(sb);
                    let idx_val = tok_p1(tok_base, pk + 1);
                    cur_advance(sb);
                    prim = mk_node(52, prim, idx_val, 0);
                    cur_struct_idx = 0 - 1;
                } else { if nt == 2 {
                    // Stage 5 Iter B: `.IDENT` named field access.
                    // Iter D: cur_struct_idx may already be set from a
                    // prior `.IDENT` step in the chain. If still -1
                    // (first iteration), look up the LHS var.
                    let mut lhs_struct_idx: i32 = cur_struct_idx;
                    if lhs_struct_idx < 0 {
                        let prim_tag = __arena_get(prim);
                        if prim_tag == 1 {
                            let var_s = __arena_get(prim + 1);
                            let var_l = __arena_get(prim + 2);
                            lhs_struct_idx = var_struct_tab_lookup(sb, var_s, var_l);
                        };
                    };
                    if lhs_struct_idx >= 0 {
                        cur_advance(sb);                       // consume '.'
                        let fk = cur_get(sb);
                        let field_s = tok_p2(tok_base, fk);
                        let field_l = tok_p3(tok_base, fk);
                        cur_advance(sb);                       // consume IDENT
                        let f_idx = struct_tab_field_lookup(sb, lhs_struct_idx, field_s, field_l);
                        if f_idx >= 0 {
                            // Iter D: is this field struct-typed?
                            let f_struct_idx = struct_tab_field_struct_idx(sb, lhs_struct_idx, f_idx);
                            if f_struct_idx >= 0 {
                                // Nested struct field: emit AST_TUPLE_FIELD
                                // with p3 == 1 to mark an 8-byte (REX.W)
                                // read of the child pointer, and propagate
                                // struct_idx forward for the next chained
                                // access.
                                prim = mk_node(52, prim, f_idx, 1);
                                cur_struct_idx = f_struct_idx;
                            } else {
                                prim = mk_node(52, prim, f_idx, 0);
                                cur_struct_idx = 0 - 1;
                            };
                        } else { keep_p = 0; };
                    } else { keep_p = 0; };
                } else { keep_p = 0; }};
            } else { if pt == 20 {                     // TK_LBRACK
                cur_advance(sb);                       // skip '['
                let idx_expr = parse_expr(tok_base, sb);
                cur_advance(sb);                       // skip ']'
                prim = mk_node(53, prim, idx_expr, 0);
                cur_struct_idx = 0 - 1;
            } else { keep_p = 0; }; };
        }
        prim
    }}}
}

fn parse_primary(tok_base: i32, sb: i32) -> i32 {
    let k = cur_get(sb);
    let t = tok_tag(tok_base, k);
    if t == 1 {
        let v = tok_p1(tok_base, k);
        cur_advance(sb);
        mk_node(0, v, 0, 0)
    } else { if t == 26 {
        // Float literal (TK_FLOATLIT). Phase 1.10b: parser emits
        // AST_FLOATLIT (tag 27) carrying byte_start + byte_len of the
        // literal text. Codegen converts to IEEE 754 bits at compile
        // time. Until codegen lands, emit-with-AST_ERR fallback so
        // bootstrap-compiled programs that touch floats fail loudly
        // instead of silently miscompiling.
        let body_s = tok_p2(tok_base, k);
        let body_l = tok_p3(tok_base, k);
        cur_advance(sb);
        mk_node(27, body_s, body_l, 0)
    } else { if t == 41 {
        // Stage 1.5: TK_FLOATLIT_BF16 (tag 41) -> AST_FLOATLIT_BF16
        // (tag 42). Codegen reuses the f32 float-bits parser then masks
        // off the low 16 mantissa bits to produce the bf16 truncation.
        let body_s = tok_p2(tok_base, k);
        let body_l = tok_p3(tok_base, k);
        cur_advance(sb);
        mk_node(42, body_s, body_l, 0)
    } else { if t == 32 {
        // Step 7b: TK_FLOATLIT_F64 (tag 32) -> AST_FLOATLIT_F64 (tag 34).
        // Distinct from AST_FLOATLIT (tag 27, f32) so codegen can branch
        // on element width. Step 7b only threads the tag through with
        // identical semantics to f32; step 7c will switch to true 8-byte
        // codegen (movabs rax, imm64 + movq xmm0, rax). p1=byte_start,
        // p2=byte_len pointing at the literal text in the source buffer.
        let body_s = tok_p2(tok_base, k);
        let body_l = tok_p3(tok_base, k);
        cur_advance(sb);
        mk_node(34, body_s, body_l, 0)
    } else { if t == 33 {
        // Approach A Stage 1: TK_INTLIT_I64 (tag 33) -> AST_INTLIT_I64
        // (tag 35). Distinct AST tag so codegen emits 8-byte
        // `movabs rax, imm64` (loads full 64-bit pattern, sign-extended
        // for negative values that fit in i32) instead of 4-byte
        // `mov eax, imm32`. The 64-bit width matters when the i64
        // value flows into a let-binding, fn param, or arithmetic op
        // typed as i64 — the high half must survive.
        let v = tok_p1(tok_base, k);
        cur_advance(sb);
        mk_node(35, v, 0, 0)
    } else { if t == 34 {
        // Approach A Stage 2.1: TK_INTLIT_U32 (tag 34) -> AST_INTLIT_U32
        // (tag 36). Codegen emits identical bits to AST_INTLIT (i32) —
        // x86 `mov eax, imm32` works for both signed and unsigned,
        // overflow wraps mod 2^32 either way. The DISTINCT AST tag
        // matters for type-tracking: expr_type returns 6 (u32) so
        // u32 values don't accidentally match i32 in 4-way dispatch
        // sites that care about signedness (DIV/MOD/comparison —
        // Stage 2.2 wires the unsigned variants).
        let v = tok_p1(tok_base, k);
        cur_advance(sb);
        mk_node(36, v, 0, 0)
    } else { if t == 35 {
        // Approach A Stage 2.3: TK_INTLIT_U8 (tag 35) -> AST_INTLIT_U8
        // (tag 37). Same codegen as AST_INTLIT (mov eax, imm32) — the
        // value is small enough that low byte holds it; high bytes
        // are zero. expr_type returns 7 (u8) so DIV/MOD/comparisons
        // dispatch to unsigned variants. Narrow load/store via movzx
        // is deferred to Stage 2.3b (8-byte stack slots remain since
        // alignment matters more than packing for Phase 0).
        let v = tok_p1(tok_base, k);
        cur_advance(sb);
        mk_node(37, v, 0, 0)
    } else { if t == 36 {
        // Approach A Stage 2.4: TK_INTLIT_U64 (tag 36) -> AST_INTLIT_U64
        // (tag 38). Codegen emits 8-byte `movabs rax, imm64` (same as
        // i64 literal). x86 64-bit ops work for both signed and
        // unsigned operands; only DIV/MOD and comparisons differ —
        // u64 dispatches to `48 31 D2; 48 F7 F1` (xor rdx,rdx; div rcx)
        // for unsigned division, setb/seta/setbe/setae for unsigned
        // comparisons. expr_type returns 9 (u64) for type tracking.
        let v = tok_p1(tok_base, k);
        cur_advance(sb);
        mk_node(38, v, 0, 0)
    } else { if t == 37 {
        // Approach A Stage 2.5b: TK_INTLIT_I8 (tag 37) -> AST_INTLIT_I8
        // (tag 39). Same minimal scaffold as u8 / u32 — codegen emits
        // `mov eax, imm32` and lets the i32-shaped storage hold the
        // value; signed range [-128, 127] fits in i32 with no sign
        // surprise since x86 mov eax,imm32 takes 32 bits as-is. expr_type
        // returns 10 (i8 type tag per the namespace doc). Narrow load/
        // store via movsx (sign-extend) is deferred to a follow-on
        // stage; arena slots remain 4 bytes wide.
        let v = tok_p1(tok_base, k);
        cur_advance(sb);
        mk_node(39, v, 0, 0)
    } else { if t == 38 {
        // Approach A Stage 2.5c: TK_INTLIT_I16 (tag 38) -> AST_INTLIT_I16
        // (tag 40). Same minimal scaffold as i8 — `mov eax, imm32` keeps
        // the value in i32-shaped storage. i16 range [-32768, 32767]
        // fits in i32 with no surprises. expr_type returns 11 (i16 type
        // tag). Narrow movsx load and masked store deferred.
        let v = tok_p1(tok_base, k);
        cur_advance(sb);
        mk_node(40, v, 0, 0)
    } else { if t == 39 {
        // Approach A Stage 2.5c: TK_INTLIT_U16 (tag 39) -> AST_INTLIT_U16
        // (tag 41). Mirror of i16 with type tag 8 (u16). Fits in i32 with
        // high bytes zero. Stage 2.2 / 2.4 unsigned dispatch already
        // works correctly for u32 / u64; u16 falls through to i32 path
        // for arithmetic since x86 add/sub/mul are signedness-agnostic.
        let v = tok_p1(tok_base, k);
        cur_advance(sb);
        mk_node(41, v, 0, 0)
    } else { if t == 25 {
        // String literal (TK_STRLIT). Token slots:
        //   payload   = body byte_start (in the source buffer)
        //   src_len   = body byte length (excluding quotes)
        // We forward both to AST_STR_LIT so codegen can emit the
        // exact bytes into a .data blob. As a value, AST_STR_LIT
        // currently lowers to `mov eax, 0` — strings are only
        // meaningful as the FIRST argument of a file builtin in
        // Phase 0.
        let body_s = tok_p2(tok_base, k);
        let body_l = tok_p3(tok_base, k);
        cur_advance(sb);
        mk_node(25, body_s, body_l, 0)
    } else { if t == 2 {
        let id_start = tok_p2(tok_base, k);
        let id_len = tok_p3(tok_base, k);
        if byte_eq(id_start, id_len, kw_let_s(sb), kw_let_n(sb)) == 1 {
            cur_advance(sb);
            // Optional `mut` keyword.
            let nk0 = cur_get(sb);
            let nk0_tag = tok_tag(tok_base, nk0);
            let mut is_mut: i32 = 0;
            if nk0_tag == 2 {
                let nk0_s = tok_p2(tok_base, nk0);
                let nk0_l = tok_p3(tok_base, nk0);
                if byte_eq(nk0_s, nk0_l, kw_mut_s(sb), kw_mut_n(sb)) == 1 {
                    is_mut = 1;
                    cur_advance(sb);
                };
            };
            let nk = cur_get(sb);
            let name_start = tok_p2(tok_base, nk);
            let name_len = tok_p3(tok_base, nk);
            cur_advance(sb);     // name
            // Optional `: T` type annotation. Phase-0 only has `i32`
            // so we silently skip both the colon and the following
            // ident. Without this, `let mut i: i32 = 0` would mis-
            // align the cursor and break self-host of the bootstrap
            // parser.
            let after_name_tag = tok_tag(tok_base, cur_get(sb));
            if after_name_tag == 14 {
                cur_advance(sb);    // consume ':'
                cur_advance(sb);    // consume type IDENT
            };
            cur_advance(sb);     // '='
            // value uses parse_expr_basic so the `;` after the
            // value belongs to the let-terminator, not a sequencer.
            let value = parse_expr_basic(tok_base, sb);
            // Iter B: if the value was a struct lit, last_struct_idx
            // is now set; register the binding name -> struct_idx so
            // postfix `.IDENT` on this var resolves to a field offset.
            let s_idx_b = last_struct_idx(sb);
            if s_idx_b >= 0 {
                var_struct_tab_add(sb, name_start, name_len, s_idx_b);
                set_last_struct_idx(sb, 0 - 1);
            };
            cur_advance(sb);     // ';'
            let body = parse_expr(tok_base, sb);
            // Audit-14: AST_LET / AST_LET_MUT used to pack
            // `value_idx * 65536 + body_idx` into p3, but arena
            // indices for large sources easily exceed 16 bits
            // (kovc.hx self-host has AST nodes at slot 150K+).
            // Extend the node to 5 slots: p3 = body_idx, p4 =
            // value_idx, both 32-bit.
            let tag = if is_mut == 1 { 12 } else { 8 };
            let node = mk_node(tag, name_start, name_len, body);
            __arena_push(value);
            node
        } else { if byte_eq(id_start, id_len, kw_if_s(sb), kw_if_n(sb)) == 1 {
            cur_advance(sb);
            let cond = parse_expr_basic(tok_base, sb);
            cur_advance(sb);     // '{'
            let then_e = parse_expr(tok_base, sb);
            cur_advance(sb);     // '}'
            // Optional `else` arm. If next token is `else` (ident),
            // parse `else { ... }`. Otherwise the if-expr's value
            // when cond is false is 0 (the AST_INT(0) emitted from
            // the synthetic else branch). Audit-15: bootstrap parser
            // used to require else; without this guard, byte_eq's
            // `if ba != bb { ok = 0; };` (no else) shifted the cursor
            // and corrupted everything downstream during self-host.
            let after_then_tok = cur_get(sb);
            let after_then_tag = tok_tag(tok_base, after_then_tok);
            let mut else_e: i32 = 0;
            if after_then_tag == 2 {
                let ats_s = tok_p2(tok_base, after_then_tok);
                let ats_l = tok_p3(tok_base, after_then_tok);
                if byte_eq(ats_s, ats_l, kw_else_s(sb), kw_else_n(sb)) == 1 {
                    cur_advance(sb);     // 'else'
                    // `else if` chaining: peek the next token. If it
                    // is the keyword `if`, parse a nested if-expr as
                    // the else branch directly — the recursive call
                    // owns its own `{ ... }` boundaries, so we must
                    // NOT eat a `{`/`}` pair here. Mirrors the
                    // helixc-Python desugaring of `else if` to
                    // `else { if ... }` without the surplus block.
                    let elif_tok = cur_get(sb);
                    let elif_tag = tok_tag(tok_base, elif_tok);
                    let mut is_elif: i32 = 0;
                    if elif_tag == 2 {
                        let elif_s = tok_p2(tok_base, elif_tok);
                        let elif_l = tok_p3(tok_base, elif_tok);
                        if byte_eq(elif_s, elif_l, kw_if_s(sb), kw_if_n(sb)) == 1 {
                            is_elif = 1;
                        };
                    };
                    if is_elif == 1 {
                        else_e = parse_expr_basic(tok_base, sb);
                    } else {
                        cur_advance(sb);     // '{'
                        else_e = parse_expr(tok_base, sb);
                        cur_advance(sb);     // '}'
                    };
                } else {
                    else_e = mk_node(0, 0, 0, 0);   // AST_INT(0)
                };
            } else {
                else_e = mk_node(0, 0, 0, 0);       // AST_INT(0)
            };
            mk_node(7, cond, then_e, else_e)
        } else { if byte_eq(id_start, id_len, kw_while_s(sb), kw_while_n(sb)) == 1 {
            // while expr { body } — Phase-0 returns 0.
            cur_advance(sb);
            let cond = parse_expr_basic(tok_base, sb);
            cur_advance(sb);     // '{'
            let body = parse_expr(tok_base, sb);
            cur_advance(sb);     // '}'
            mk_node(10, cond, body, 0)
        } else { if byte_eq(id_start, id_len, kw_match_s(sb), kw_match_n(sb)) == 1 {
            // Stage 7: match scrut { pat => body, pat => body, ... }
            // Build AST_MATCH (tag 62) with p1 = scrut_idx, p2 = arms_head_idx.
            // Each arm is AST_MATCH_ARM (tag 63) p1=pattern, p2=body, p3=next.
            parse_match_expr(tok_base, sb)
        } else {
            // Plain identifier. Could be a var ref, an assignment
            // (`name = expr`), or a fn call (`name()`). Peek the
            // NEXT token to decide.
            // Stage 6: PRE-CHECK — IDENT followed by `::` IDENT and the
            // first IDENT matches a registered enum. We look up the
            // enum BEFORE consuming the leading IDENT so that the
            // 4-way dispatch below doesn't need a 5th nested if (host
            // parser recursion budget — Finding #7). The peek looks at
            // tok_at(k+1) and tok_at(k+2): both must be TK_COLON (14)
            // and tok_at(k+3) must be TK_IDENT (2). For 6B (unit
            // variant) the next-after-variant must NOT be `(` (= 3);
            // 6C handles the `(` case (payload variant).
            // FLAT prefix-trap pattern (Finding #7): single-binding
            // ladder of let-rebinds, NO nested if-else statements.
            let e_idx_pre = enum_tab_lookup_idx(sb, id_start, id_len);
            let t1_pre = tok_tag(tok_base, k + 1);
            let t2_pre = tok_tag(tok_base, k + 2);
            let t3_pre = tok_tag(tok_base, k + 3);
            let t4_pre = tok_tag(tok_base, k + 4);
            let is_enum_path = if e_idx_pre >= 0 {
                if t1_pre == 14 { if t2_pre == 14 { if t3_pre == 2 { 1 } else { 0 } } else { 0 } } else { 0 }
            } else { 0 };
            // Distinguish unit (6B) vs payload (6C): payload variant
            // has `(` at k+4. 6C handled below as a separate prefix.
            let is_enum_unit = if is_enum_path == 1 { if t4_pre == 3 { 0 } else { 1 } } else { 0 };
            let is_enum_payload = if is_enum_path == 1 { if t4_pre == 3 { 1 } else { 0 } } else { 0 };
            if is_enum_unit == 1 {
                // Consume IDENT, `:`, `:`, variant-IDENT.
                cur_advance(sb);                       // outer IDENT (enum name)
                cur_advance(sb);                       // first ':'
                cur_advance(sb);                       // second ':'
                let vk = cur_get(sb);
                let v_name_s = tok_p2(tok_base, vk);
                let v_name_l = tok_p3(tok_base, vk);
                cur_advance(sb);                       // variant IDENT
                let disc = enum_tab_variant_lookup_disc(sb, e_idx_pre, v_name_s, v_name_l);
                let arity = enum_tab_variant_lookup_arity(sb, e_idx_pre, v_name_s, v_name_l);
                let safe_disc = if disc < 0 { 0 } else { disc };
                // Stage 7F fix: if the enum has ANY payload variants
                // (max_payload_arity > 0), unit variants must use the
                // pointer-shaped rep too — otherwise Stage 7's PAT_VARIANT
                // codegen segfaults trying to deref the disc-as-pointer.
                // For all-unit enums (e.g. Color { R, G, B }), keep the
                // AST_INT fold for backward compat with Stage 6B tests.
                let enum_entry = enum_tab_base(sb) + e_idx_pre * 5;
                let max_arity = __arena_get(enum_entry + 4);
                if max_arity > 0 {
                    // Build 1-slot AST_TUPLE_LIT with disc only.
                    let disc_node = mk_node(0, safe_disc, 0, 0);
                    let head_idx = mk_node(51, disc_node, 0, 0);
                    set_last_enum_idx(sb, e_idx_pre);
                    mk_node(50, 1, head_idx, 0)
                } else {
                    // All-unit enum — fold to plain AST_INT (Stage 6B).
                    mk_node(0, safe_disc, 0, 0)
                }
            } else { if is_enum_payload == 1 {
                // Stage 6C: payload variant `Maybe::Some(42)`. Build
                // an AST_TUPLE_LIT (tag 50) with arity = 1 + payload
                // arity, head = TUPLE_CONS chain whose first element
                // is the discriminant (AST_INT) and rest are the
                // parenthesized payload args. Codegen reuses tuple-lit
                // entirely: rax holds a pointer to a stack region with
                // [disc, arg0, arg1, ...]. Reading the discriminant is
                // .0 (= AST_TUPLE_FIELD with idx 0), reading payload
                // arg i is .(i+1).
                cur_advance(sb);                       // outer IDENT (enum name)
                cur_advance(sb);                       // first ':'
                cur_advance(sb);                       // second ':'
                let vk = cur_get(sb);
                let v_name_s = tok_p2(tok_base, vk);
                let v_name_l = tok_p3(tok_base, vk);
                cur_advance(sb);                       // variant IDENT
                cur_advance(sb);                       // '('
                let disc = enum_tab_variant_lookup_disc(sb, e_idx_pre, v_name_s, v_name_l);
                let arity = enum_tab_variant_lookup_arity(sb, e_idx_pre, v_name_s, v_name_l);
                let safe_disc = if disc < 0 { 0 } else { disc };
                let safe_arity = if arity < 0 { 0 } else { arity };
                // Build the discriminant TUPLE_CONS head.
                let disc_node = mk_node(0, safe_disc, 0, 0);
                let mut head_idx: i32 = mk_node(51, disc_node, 0, 0);
                let mut tail_idx: i32 = head_idx;
                let mut n_args: i32 = 1;     // counts disc
                // Walk comma-separated payload args until ')'.
                let mut keep: i32 = 1;
                while keep == 1 {
                    let at = tok_tag(tok_base, cur_get(sb));
                    if at == 4 {
                        keep = 0;            // ')'
                    } else { if at == 13 {
                        cur_advance(sb);     // ','
                    } else { if at == 0 {    // EOF safety
                        keep = 0;
                    } else {
                        let arg_expr = parse_expr_basic(tok_base, sb);
                        let new_node = mk_node(51, arg_expr, 0, 0);
                        let prev_tail = tail_idx;
                        __arena_set(prev_tail + 2, new_node);
                        tail_idx = new_node;
                        n_args = n_args + 1;
                    }}};
                }
                cur_advance(sb);                       // consume ')'
                // Mark the surrounding let-parser: this binding is
                // enum-typed. Reuses last_enum_idx scratch slot.
                set_last_enum_idx(sb, e_idx_pre);
                // n_args >= 1 (always includes the discriminant).
                mk_node(50, n_args, head_idx, 0)
            } else {
            cur_advance(sb);
            let next = cur_get(sb);
            let nt = tok_tag(tok_base, next);
            if nt == 15 {
                // Could be `=` (assign) or `==` (equality). Peek one
                // more ahead: if it's also `=`, this is `name == ...`,
                // and we should NOT consume the `=`s here — leave
                // them for parse_expr_basic to handle as a comparison.
                let nt2 = tok_tag(tok_base, cur_get(sb) + 1);
                if nt2 == 15 {
                    mk_node(1, id_start, id_len, 0)
                } else {
                    cur_advance(sb);
                    let value = parse_expr_basic(tok_base, sb);
                    mk_node(11, id_start, id_len, value)
                }
            } else { if nt == 3 {
                // CALL: name(arg1, arg2, ...). Args become AST_ARG
                // linked list; head index goes in CALL.p3 (or 0 if
                // no args).
                cur_advance(sb);     // consume '('
                let mut args_head: i32 = 0;
                let mut prev_arg: i32 = 0;
                let mut k_keep: i32 = 1;
                while k_keep == 1 {
                    let at = tok_tag(tok_base, cur_get(sb));
                    if at == 4 {
                        k_keep = 0;
                    } else { if at == 13 {
                        cur_advance(sb);
                    } else {
                        let arg_expr = parse_expr_basic(tok_base, sb);
                        let new_arg = mk_node(17, arg_expr, 0, 0);
                        if args_head == 0 {
                            args_head = new_arg;
                            prev_arg = new_arg;
                        } else {
                            __arena_set(prev_arg + 2, new_arg);
                            prev_arg = new_arg;
                        };
                    }};
                }
                cur_advance(sb);     // consume ')'
                // Stage 6D: detect __enum_payload(value_expr, idx_intlit)
                // and rewrite to AST_TUPLE_FIELD(value_expr, idx + 1).
                // The value lives at slot idx+1 in the tuple-lit-shaped
                // enum representation (slot 0 = discriminant). Folds
                // into existing AST_TUPLE_FIELD codegen — no new arm.
                // Match name bytes: "__enum_payload" = 14 chars.
                // FLAT prefix-trap pattern (Finding #7): NO nested
                // if-else statements — single ladder.
                let is_ep = if id_len == 14 {
                    let b0 = __arena_get(id_start);
                    let b1 = __arena_get(id_start + 1);
                    let b2 = __arena_get(id_start + 2);
                    let b3 = __arena_get(id_start + 3);
                    let b4 = __arena_get(id_start + 4);
                    let b5 = __arena_get(id_start + 5);
                    let b6 = __arena_get(id_start + 6);
                    let b7 = __arena_get(id_start + 7);
                    let b8 = __arena_get(id_start + 8);
                    let b9 = __arena_get(id_start + 9);
                    let b10 = __arena_get(id_start + 10);
                    let b11 = __arena_get(id_start + 11);
                    let b12 = __arena_get(id_start + 12);
                    let b13 = __arena_get(id_start + 13);
                    let m1 = if b0 == 95 { if b1 == 95 { 1 } else { 0 } } else { 0 };
                    let m2 = if b2 == 101 { if b3 == 110 { if b4 == 117 { if b5 == 109 { 1 } else { 0 } } else { 0 } } else { 0 } } else { 0 };
                    let m3 = if b6 == 95 { if b7 == 112 { 1 } else { 0 } } else { 0 };
                    let m4 = if b8 == 97 { if b9 == 121 { if b10 == 108 { if b11 == 111 { if b12 == 97 { if b13 == 100 { 1 } else { 0 } } else { 0 } } else { 0 } } else { 0 } } else { 0 } } else { 0 };
                    if m1 == 1 { if m2 == 1 { if m3 == 1 { if m4 == 1 { 1 } else { 0 } } else { 0 } } else { 0 } } else { 0 }
                } else { 0 };
                if is_ep == 1 {
                    let a0 = __arena_get(args_head + 1);     // value expr
                    let next_arg = __arena_get(args_head + 2);
                    let a1 = __arena_get(next_arg + 1);      // idx expr
                    // Stage 6D: idx must be an INTLIT (compile-time
                    // constant) — Phase 0 does not support dynamic
                    // payload indices. Trap (AST_ERR) if not.
                    let a1_tag = __arena_get(a1);
                    let idx_val = if a1_tag == 0 {
                        __arena_get(a1 + 1)    // AST_INT.p1 = value
                    } else { 0 };
                    // Emit AST_TUPLE_FIELD(value, idx+1, 0). The +1
                    // skips the discriminant slot at offset 0.
                    mk_node(52, a0, idx_val + 1, 0)
                } else {
                    mk_node(16, id_start, id_len, args_head)
                }
            } else { if nt == 5 {
                // Stage 5 Iter A: IDENT followed by '{' might be a struct
                // literal `Pt { 10, 32 }`. Look up the IDENT in
                // struct_table; on hit (arity >= 0), parse positional
                // values into an AST_TUPLE_LIT chain, reusing tuple
                // codegen entirely. On miss, fall through to var-ref.
                // Iter B: use struct_tab_lookup_idx so we can also
                // record the struct_idx in the last_struct_idx scratch
                // slot for the surrounding let-parser to pick up.
                let s_idx = struct_tab_lookup_idx(sb, id_start, id_len);
                let arity = if s_idx >= 0 {
                    let entry = struct_tab_base(sb) + s_idx * 4;
                    __arena_get(entry + 2)
                } else { 0 - 1 };
                if arity >= 0 {
                    cur_advance(sb);     // consume '{'
                    // Empty struct `Foo {}` — arity 0.
                    let pk_first = cur_get(sb);
                    let pt_first = tok_tag(tok_base, pk_first);
                    if pt_first == 6 {
                        cur_advance(sb);   // consume '}'
                        // Set last_struct_idx AFTER children (here:
                        // arity 0, no children) so nested struct lits
                        // can't overwrite the outer's idx — Iter D fix
                        // for `let l = Outer { Inner {...} }`.
                        set_last_struct_idx(sb, s_idx);
                        mk_node(50, 0, 0, 0)
                    } else {
                        let first = parse_expr(tok_base, sb);
                        let mut head_idx: i32 = mk_node(51, first, 0, 0);
                        let mut tail_idx: i32 = head_idx;
                        let mut n: i32 = 1;
                        let mut keep: i32 = 1;
                        while keep == 1 {
                            let ck = cur_get(sb);
                            let ct = tok_tag(tok_base, ck);
                            if ct == 13 {
                                cur_advance(sb);    // ','
                                let pk2 = cur_get(sb);
                                let pt2 = tok_tag(tok_base, pk2);
                                if pt2 == 6 { keep = 0; }     // trailing ','
                                else {
                                    let child = parse_expr(tok_base, sb);
                                    let new_node = mk_node(51, child, 0, 0);
                                    let prev_tail = tail_idx;
                                    __arena_set(prev_tail + 2, new_node);
                                    tail_idx = new_node;
                                    n = n + 1;
                                };
                            } else { keep = 0; };
                        }
                        cur_advance(sb);    // consume '}'
                        // Iter D fix: set last_struct_idx AFTER parsing
                        // children. Inner struct lits set it to their
                        // own idx during their parse_primary; setting
                        // here last writes the OUTER's idx, which is
                        // what surrounding let-parsing needs.
                        set_last_struct_idx(sb, s_idx);
                        mk_node(50, n, head_idx, 0)
                    }
                } else {
                    // Not a registered struct — treat as var ref.
                    mk_node(1, id_start, id_len, 0)
                }
            } else {
                // Var ref
                mk_node(1, id_start, id_len, 0)
            }}}
            }}
        }}}}
    } else { if t == 3 {
        // Stage 4 iteration A: tuple literal vs parenthesized expr.
        // After the inner expr, peek for TK_COMMA (13). If found, this
        // is a tuple literal — build a TUPLE_CONS chain. Otherwise it's
        // a normal parenthesized expr.
        cur_advance(sb);
        let inner = parse_expr(tok_base, sb);
        let nk = cur_get(sb);
        let nt = tok_tag(tok_base, nk);
        if nt == 13 {
            // Tuple literal: walk comma-separated children, build
            // TUPLE_CONS chain (head -> [child0, next] -> [child1, next] -> ...).
            // mk_node tag 51 = AST_TUPLE_CONS, p1 = child_idx, p2 = next_idx.
            cur_advance(sb);   // skip first ','
            let mut head_idx: i32 = mk_node(51, inner, 0, 0);
            let mut tail_idx: i32 = head_idx;
            let mut arity: i32 = 1;
            let mut keep: i32 = 1;
            while keep == 1 {
                // Allow trailing comma: peek after comma for ')'.
                let pk = cur_get(sb);
                let pt = tok_tag(tok_base, pk);
                if pt == 4 { keep = 0; }
                else {
                    let child = parse_expr(tok_base, sb);
                    let new_node = mk_node(51, child, 0, 0);
                    // Patch previous tail's p2 to point to new_node.
                    let prev_tail = tail_idx;
                    __arena_set(prev_tail + 2, new_node);
                    tail_idx = new_node;
                    arity = arity + 1;
                    let ck = cur_get(sb);
                    let ct = tok_tag(tok_base, ck);
                    if ct == 13 { cur_advance(sb); }     // skip ',' continue
                    else { keep = 0; }
                };
            }
            cur_advance(sb);   // skip ')'
            // mk_node tag 50 = AST_TUPLE_LIT, p1 = arity, p2 = head_idx.
            mk_node(50, arity, head_idx, 0)
        } else {
            cur_advance(sb);     // ')'
            inner
        }
    } else { if t == 20 {
        // Stage 4 iteration D: static array literal [a, b, c].
        // Same shape as tuples (tag 50/51) but uses TK_LBRACK / TK_RBRACK
        // delimiters. Reuses AST_TUPLE_CONS (tag 51) for the chain and
        // AST_TUPLE_LIT (tag 50) for the head — codegen-identical
        // (homogeneous arrays vs heterogeneous tuples differ only in
        // static type-checking, which Phase 0 doesn't enforce strictly).
        cur_advance(sb);     // skip '['
        let pk = cur_get(sb);
        let pt = tok_tag(tok_base, pk);
        if pt == 21 {
            // Empty array `[]` — arity 0. Just allocate a 0-byte region.
            cur_advance(sb);    // skip ']'
            mk_node(50, 0, 0, 0)
        } else {
            let first = parse_expr(tok_base, sb);
            let mut head_idx: i32 = mk_node(51, first, 0, 0);
            let mut tail_idx: i32 = head_idx;
            let mut arity: i32 = 1;
            let mut keep: i32 = 1;
            while keep == 1 {
                let ck = cur_get(sb);
                let ct = tok_tag(tok_base, ck);
                if ct == 13 {
                    cur_advance(sb);    // skip ','
                    // Allow trailing comma.
                    let pk2 = cur_get(sb);
                    let pt2 = tok_tag(tok_base, pk2);
                    if pt2 == 21 { keep = 0; }
                    else {
                        let child = parse_expr(tok_base, sb);
                        let new_node = mk_node(51, child, 0, 0);
                        let prev_tail = tail_idx;
                        __arena_set(prev_tail + 2, new_node);
                        tail_idx = new_node;
                        arity = arity + 1;
                    };
                } else { keep = 0; };
            }
            cur_advance(sb);    // skip ']'
            mk_node(50, arity, head_idx, 0)
        }
    } else {
        // Audit-7 fix: don't advance past TK_EOF (tag 0). Without
        // this guard, a malformed input like `1 + (` walks the
        // cursor past the EOF sentinel into uninitialized arena
        // slots, and the parse_add/parse_mul while-loops then read
        // arbitrary values as if they were tokens — non-deterministic
        // junk AST. We return AST_ERR but hold the cursor at EOF
        // so callers immediately re-encounter EOF and unwind cleanly.
        //
        // Audit-16 extension: also don't advance past `}` (tag 6) or
        // `)` (tag 4). Empty blocks like `else {}` (used in kovc.hx's
        // pidx-register switch fallthrough) were broken — parse_expr
        // descended into parse_primary on the `}` of the empty body,
        // the catch-all consumed it, and the if-handler's followup
        // cur_advance then ate the OUTER `}`. Cursor desynced for the
        // rest of the file. Same idea for `)` in calls like `f()`.
        if t != 0 {
            if t != 6 {
                if t != 4 {
                    cur_advance(sb);
                };
            };
        };
        mk_node(99, t, 0, 0)
    }}}}}}}}}}}}}}}
}

// Stage 5 Iter B: struct_table region — 12 slots = 3 entries x 4 fields
// (name_s, name_l, arity, fields_ptr). fields_ptr is 0 in Iter A; Iter B
// fills it with an arena offset to a per-struct field-names region built
// during parse_struct_decl, used to resolve `p.IDENT` -> field index.
fn struct_tab_init(sb: i32) -> i32 {
    let st_base = __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_set(sb + 15, st_base);
    __arena_set(sb + 16, 0);
    0
}

// Stage 5 Iter B: var_struct_table region — 12 slots = 4 entries x 3
// fields (var_name_s, var_name_l, struct_idx). Cap 4 vars; expand later.
fn var_struct_tab_init(sb: i32) -> i32 {
    let vs_base = __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_set(sb + 17, vs_base);
    __arena_set(sb + 18, 0);
    0
}

// Stage 6: enum_table region — 20 slots = 4 entries x 5 fields
// (name_s, name_l, variant_count, variants_ptr, max_payload_arity).
fn enum_tab_init(sb: i32) -> i32 {
    let et_base = __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_set(sb + 20, et_base);
    __arena_set(sb + 21, 0);
    0
}

// Stage 6: var_enum_table region — 12 slots = 4 entries x 3 fields
// (var_name_s, var_name_l, enum_idx). Cap 4 vars.
fn var_enum_tab_init(sb: i32) -> i32 {
    let ve_base = __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_set(sb + 22, ve_base);
    __arena_set(sb + 23, 0);
    0
}

// --------------------------------------------------------------
// install_keywords: stash "let", "if", "else" bytes in the arena
// and write their (start, len) into state_base+1..state_base+6.
// --------------------------------------------------------------
fn install_keywords(sb: i32) -> i32 {
    let let_s = __arena_push(108); __arena_push(101); __arena_push(116);
    __arena_set(sb + 1, let_s);
    __arena_set(sb + 2, 3);
    let if_s = __arena_push(105); __arena_push(102);
    __arena_set(sb + 3, if_s);
    __arena_set(sb + 4, 2);
    let else_s = __arena_push(101); __arena_push(108); __arena_push(115); __arena_push(101);
    __arena_set(sb + 5, else_s);
    __arena_set(sb + 6, 4);
    // "while" = 119 104 105 108 101
    let while_s = __arena_push(119); __arena_push(104); __arena_push(105);
    __arena_push(108); __arena_push(101);
    __arena_set(sb + 7, while_s);
    __arena_set(sb + 8, 5);
    // "mut" = 109 117 116
    let mut_s = __arena_push(109); __arena_push(117); __arena_push(116);
    __arena_set(sb + 9, mut_s);
    __arena_set(sb + 10, 3);
    // "fn" = 102 110
    let fn_s = __arena_push(102); __arena_push(110);
    __arena_set(sb + 11, fn_s);
    __arena_set(sb + 12, 2);
    // Stage 5: "struct" = 115 116 114 117 99 116
    let struct_s = __arena_push(115); __arena_push(116); __arena_push(114);
    __arena_push(117); __arena_push(99); __arena_push(116);
    __arena_set(sb + 13, struct_s);
    __arena_set(sb + 14, 6);
    struct_tab_init(sb);
    // Stage 6: "enum" = 101 110 117 109
    let enum_s = __arena_push(101); __arena_push(110); __arena_push(117); __arena_push(109);
    __arena_set(sb + 25, enum_s);
    __arena_set(sb + 26, 4);
    enum_tab_init(sb);
    // Stage 7: "match" = 109 97 116 99 104
    let match_s = __arena_push(109); __arena_push(97); __arena_push(116);
    __arena_push(99); __arena_push(104);
    __arena_set(sb + 27, match_s);
    __arena_set(sb + 28, 5);
    0
}

// --------------------------------------------------------------
// Top-level parse: return the arena index of the root AST node.
// Reserves 7 state slots, then dispatches into parse_expr.
// --------------------------------------------------------------
fn parse_top(tok_base: i32) -> i32 {
    // 33 state slots: cursor + 7 keyword (start, len) pairs +
    // struct_table (base + count) + var_struct_table (base + count) +
    // last_struct_idx scratch + enum_table (base + count) +
    // var_enum_table (base + count) + last_enum_idx scratch +
    // enum keyword (start + len) + match keyword (start + len) +
    // generic_params (base + count) + mono_request (base + count).
    // Stage 6 added slots 20..26; Stage 7 added 27..28; Stage 8 added 29..32.
    let cur_slot = __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0);
    // Stage 6: slots 20..26 = enum_table base/count, var_enum_table
    // base/count, last_enum_idx scratch, enum kw start/len.
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0);
    // Stage 7: slots 27..28 = match keyword (start + len).
    __arena_push(0); __arena_push(0);
    // Stage 8: slots 29..32 = generic_params (base + count) + mono_request
    // (base + count). gp scratch reset per fn; mono_request accumulates.
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    install_keywords(cur_slot);
    var_struct_tab_init(cur_slot);
    var_enum_tab_init(cur_slot);
    __arena_set(cur_slot + 19, 0 - 1);
    __arena_set(cur_slot + 24, 0 - 1);
    // Stage 8: gp_tab region (8 slots, 4 entries x 2 fields).
    let gp_base = __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_set(cur_slot + 29, gp_base);
    __arena_set(cur_slot + 30, 0);
    // Stage 8: mr_tab region (192 slots, 32 entries x 6 fields).
    let mr_base = __arena_push(0);
    let mut mri: i32 = 1;
    while mri < 192 {
        __arena_push(0);
        mri = mri + 1;
    }
    __arena_set(cur_slot + 31, mr_base);
    __arena_set(cur_slot + 32, 0);
    // Peek the first token. If it's `fn`, parse a function decl.
    // Otherwise treat the whole input as a single expression
    // (legacy mode) for backward compat with all existing tests.
    // Skip leading attributes (`@pure`, `@effect`, etc.) — Phase 0
    // doesn't enforce them, just parses past so kovc.hx and other
    // attribute-decorated source compiles.
    skip_attributes(tok_base, cur_slot);
    let k = cur_get(cur_slot);
    if tok_tag(tok_base, k) == 2 {
        let id_s = tok_p2(tok_base, k);
        let id_l = tok_p3(tok_base, k);
        let is_fn = byte_eq(id_s, id_l, kw_fn_s(cur_slot), kw_fn_n(cur_slot));
        let is_struct = byte_eq(id_s, id_l, kw_struct_s(cur_slot), kw_struct_n(cur_slot));
        let is_enum = byte_eq(id_s, id_l, kw_enum_s(cur_slot), kw_enum_n(cur_slot));
        if is_fn == 1 {
            parse_program(tok_base, cur_slot)
        } else { if is_struct == 1 {
            parse_program(tok_base, cur_slot)
        } else { if is_enum == 1 {
            parse_program(tok_base, cur_slot)
        } else {
            parse_expr(tok_base, cur_slot)
        }}}
    } else {
        parse_expr(tok_base, cur_slot)
    }
}

// Consume zero or more `@<IDENT>` (or `@<IDENT>(<args>)`) attribute
// markers. Currently we just skip them; future Phase-1 work could
// store them on the surrounding fn decl.
fn skip_attributes(tok_base: i32, sb: i32) -> i32 {
    let mut keep: i32 = 1;
    while keep == 1 {
        if tok_tag(tok_base, cur_get(sb)) == 24 {
            cur_advance(sb);     // consume '@'
            // Optional IDENT after the '@'.
            if tok_tag(tok_base, cur_get(sb)) == 2 {
                cur_advance(sb);
            };
            // Optional `(args)` — skip everything until matching ')'.
            if tok_tag(tok_base, cur_get(sb)) == 3 {
                cur_advance(sb);     // '('
                let mut depth: i32 = 1;
                while depth > 0 {
                    let tt = tok_tag(tok_base, cur_get(sb));
                    if tt == 3 { depth = depth + 1; };
                    if tt == 4 { depth = depth - 1; };
                    if tt == 0 { depth = 0; };       // EOF safety
                    cur_advance(sb);
                };
            };
        } else {
            keep = 0;
        };
    }
    0
}

// Parse a sequence of one or more `fn` declarations at the top
// level, returning a linked list head. If only one fn is present,
// the list has a single node. The codegen looks up "main" by name
// and emits its body; other fns are placed in the binary but only
// callable once AST_CALL lands.
fn parse_program(tok_base: i32, sb: i32) -> i32 {
    // Stage 5 Iter A + Stage 6: skip leading `struct ... { ... }` and
    // `enum ... { ... }` decls. Each registers in struct_table or
    // enum_table; the returned AST_STRUCT_DECL nodes (tag 54) are
    // discarded because codegen treats them as 0-byte no-ops.
    let mut keep_decl: i32 = 1;
    while keep_decl == 1 {
        let kk = cur_get(sb);
        let tt = tok_tag(tok_base, kk);
        if tt == 2 {
            let s = tok_p2(tok_base, kk);
            let l = tok_p3(tok_base, kk);
            if byte_eq(s, l, kw_struct_s(sb), kw_struct_n(sb)) == 1 {
                parse_struct_decl(tok_base, sb);
            } else { if byte_eq(s, l, kw_enum_s(sb), kw_enum_n(sb)) == 1 {
                parse_enum_decl(tok_base, sb);
            } else {
                keep_decl = 0;
            }};
        } else {
            keep_decl = 0;
        };
    }
    let first_fn = parse_fn_decl(tok_base, sb);
    let head = mk_node(15, first_fn, 0, 0);
    let mut prev_list = head;
    let mut keep: i32 = 1;
    while keep == 1 {
        // Skip any attributes before the next fn decl.
        skip_attributes(tok_base, sb);
        let k2 = cur_get(sb);
        let t2 = tok_tag(tok_base, k2);
        if t2 == 0 {
            keep = 0;
        } else { if t2 == 2 {
            let s = tok_p2(tok_base, k2);
            let l = tok_p3(tok_base, k2);
            if byte_eq(s, l, kw_fn_s(sb), kw_fn_n(sb)) == 1 {
                let next_fn = parse_fn_decl(tok_base, sb);
                let new_node = mk_node(15, next_fn, 0, 0);
                __arena_set(prev_list + 2, new_node);
                prev_list = new_node;
            } else {
                keep = 0;
            };
        } else {
            keep = 0;
        }};
    }
    head
}

// Parse `fn name(arg1: T, arg2: T, ...) -> i32 { body }`. Each arg
// becomes an AST_PARAM node in a linked list; the head index is
// stored in the fn_decl's p3 packed with body_idx (head*65536+body).
// 0 head_idx means no params. Phase 0: types are parsed but ignored.
fn parse_fn_decl(tok_base: i32, sb: i32) -> i32 {
    cur_advance(sb);     // consume 'fn'
    let nk = cur_get(sb);
    let name_start = tok_p2(tok_base, nk);
    let name_len = tok_p3(tok_base, nk);
    cur_advance(sb);     // name
    cur_advance(sb);     // '('
    // Param list: zero or more `name: T` separated by `,`.
    let mut params_head: i32 = 0;
    let mut prev_param: i32 = 0;
    let mut keep: i32 = 1;
    while keep == 1 {
        let pt = tok_tag(tok_base, cur_get(sb));
        if pt == 4 {
            keep = 0;            // ')'
        } else { if pt == 13 {
            cur_advance(sb);     // ','
        } else {
            let pname_tok = cur_get(sb);
            let pname_s = tok_p2(tok_base, pname_tok);
            let pname_l = tok_p3(tok_base, pname_tok);
            cur_advance(sb);     // param name
            cur_advance(sb);     // ':'
            // Capture the type IDENT bytes to determine if it's "f32"
            // (or "f64", treated the same in bootstrap codegen). Step 5c
            // follow-on: this lets fn(a: f32, b: f32) -> f32 { a + b }
            // bind a and b with type=f32 so is_f32_expr resolves through
            // them and AST_ADD dispatches to SSE.
            let ty_tok = cur_get(sb);
            let ty_s = tok_p2(tok_base, ty_tok);
            let ty_l = tok_p3(tok_base, ty_tok);
            cur_advance(sb);     // type IDENT
            // Audit fix (Stage 1 cycle): all 3 bytes must match exactly.
            // Strict: 'f32' (102 51 50) → 1; 'f64' (102 54 52) → 2;
            // 'i64' (105 54 52) → 3; 'i32' (105 51 50) → 0; else 0.
            // Strict third-byte check prevents nonsense like 'i65'/'f33'
            // from silently mis-tagging.
            let p_ty = if ty_l == 3 {
                let b0 = __arena_get(ty_s);
                let b1 = __arena_get(ty_s + 1);
                let b2 = __arena_get(ty_s + 2);
                if b0 == 102 {
                    if b1 == 54 { if b2 == 52 { 2 } else { 0 } }
                    else { if b1 == 51 { if b2 == 50 { 1 } else { 0 } } else { 0 } }
                } else { if b0 == 105 {
                    if b1 == 54 { if b2 == 52 { 3 } else { 0 } }                // i64
                    else { if b1 == 49 { if b2 == 54 { 11 } else { 0 } } else { 0 } }  // i16 (Stage 2.5c)
                } else { if b0 == 117 {                  // 'u' — Stage 2.1 + 2.4 + 2.5c
                    if b1 == 51 { if b2 == 50 { 6 } else { 0 } }                // u32
                    else { if b1 == 54 { if b2 == 52 { 9 } else { 0 } }                // u64
                    else { if b1 == 49 { if b2 == 54 { 8 } else { 0 } } else { 0 } } }  // u16
                } else { 0 } } }
            } else { if ty_l == 2 {
                // Stage 2.3: 2-byte type idents — `u8` -> 7.
                // Stage 2.5b: `i8` (105 56) -> 10.
                let b0 = __arena_get(ty_s);
                let b1 = __arena_get(ty_s + 1);
                if b0 == 117 { if b1 == 56 { 7 } else { 0 } }
                else { if b0 == 105 { if b1 == 56 { 10 } else { 0 } } else { 0 } }
            } else { if ty_l == 4 {
                // Stage 1.5: 4-byte type idents — `bf16` (98 102 49 54) -> 4.
                // bf16 is the brain-float-16 dtype: truncated f32 (drop low
                // 16 bits of mantissa). Codegen treats bf16 bindings as
                // i32-shaped storage with low 16 bits zeroed; literal
                // truncation deferred to a follow-on (or to user code via
                // bit-masked __bits_of_f32).
                let b0 = __arena_get(ty_s);
                let b1 = __arena_get(ty_s + 1);
                let b2 = __arena_get(ty_s + 2);
                let b3 = __arena_get(ty_s + 3);
                if b0 == 98 {
                    if b1 == 102 { if b2 == 49 { if b3 == 54 { 4 } else { 0 } } else { 0 } }
                    else { 0 }
                } else { 0 }
            } else { 0 } } };
            // Stage 5 Iter C: detect struct-typed param. If p_ty is 0
            // (unknown primitive) AND the type IDENT matches a registered
            // struct in struct_table, encode p_ty as 100 + struct_idx so
            // codegen can recognize struct params for by-value pass.
            // Also register (param_name -> struct_idx) in var_struct_tab
            // so the body's `p.IDENT` resolves to a field offset.
            // FLAT prefix-trap pattern (Finding #7): use a single-binding
            // ladder of let-rebinds, NOT nested if-else statements, to
            // avoid host-parser recursion overflow.
            let s_idx_p = struct_tab_lookup_idx(sb, ty_s, ty_l);
            let p_ty_struct = if s_idx_p >= 0 { 100 + s_idx_p } else { 0 };
            let p_ty_final = if p_ty == 0 { p_ty_struct } else { p_ty };
            let n_register = if p_ty_final >= 100 {
                var_struct_tab_add(sb, pname_s, pname_l, p_ty_final - 100)
            } else { 0 };
            let _drop_n = n_register;
            let new_param = mk_node(18, pname_s, pname_l, 0);
            __arena_push(p_ty_final);   // p4: type tag (100+ = struct)
            if params_head == 0 {
                params_head = new_param;
                prev_param = new_param;
            } else {
                __arena_set(prev_param + 3, new_param);
                prev_param = new_param;
            };
        }};
    }
    cur_advance(sb);     // ')'
    cur_advance(sb);     // '-' (part of '->')
    cur_advance(sb);     // '>' (the second char of '->')
    // Capture the return-type IDENT bytes the same way AST_PARAM does.
    // 'f' first byte (length 3) -> f32 / f64 -> ret_ty = 1.
    let rt_tok = cur_get(sb);
    let rt_s = tok_p2(tok_base, rt_tok);
    let rt_l = tok_p3(tok_base, rt_tok);
    cur_advance(sb);     // return-type IDENT
    // Audit fix (Stage 1 cycle): strict 3-byte type-ident check.
    let ret_ty = if rt_l == 3 {
        let b0 = __arena_get(rt_s);
        let b1 = __arena_get(rt_s + 1);
        let b2 = __arena_get(rt_s + 2);
        if b0 == 102 {
            if b1 == 54 { if b2 == 52 { 2 } else { 0 } }
            else { if b1 == 51 { if b2 == 50 { 1 } else { 0 } } else { 0 } }
        } else { if b0 == 105 {
            if b1 == 54 { if b2 == 52 { 3 } else { 0 } }                // i64
            else { if b1 == 49 { if b2 == 54 { 11 } else { 0 } } else { 0 } }  // i16 (Stage 2.5c)
        } else { if b0 == 117 {                  // 'u' — Stage 2.1 + 2.4 + 2.5c
            if b1 == 51 { if b2 == 50 { 6 } else { 0 } }                // u32
            else { if b1 == 54 { if b2 == 52 { 9 } else { 0 } }                // u64
            else { if b1 == 49 { if b2 == 54 { 8 } else { 0 } } else { 0 } } }  // u16
        } else { 0 } } }
    } else { if rt_l == 2 {
        // Stage 2.3: 2-byte type idents — `u8` -> 7.
        // Stage 2.5b: `i8` (105 56) -> 10.
        let b0 = __arena_get(rt_s);
        let b1 = __arena_get(rt_s + 1);
        if b0 == 117 { if b1 == 56 { 7 } else { 0 } }
        else { if b0 == 105 { if b1 == 56 { 10 } else { 0 } } else { 0 } }
    } else { if rt_l == 4 {
        // Stage 1.5: 4-byte type idents — `bf16` -> 4.
        let b0 = __arena_get(rt_s);
        let b1 = __arena_get(rt_s + 1);
        let b2 = __arena_get(rt_s + 2);
        let b3 = __arena_get(rt_s + 3);
        if b0 == 98 {
            if b1 == 102 { if b2 == 49 { if b3 == 54 { 4 } else { 0 } } else { 0 } }
            else { 0 }
        } else { 0 }
    } else { 0 } } };
    cur_advance(sb);     // '{'
    let body = parse_expr(tok_base, sb);
    cur_advance(sb);     // '}'
    // Audit-14: same overflow issue as AST_LET — packed encoding
    // breaks for arena indices > 65535. Extend to 5 slots: p3 =
    // body_idx, p4 = params_head. Step 5c follow-on: 6th slot p5 =
    // ret_ty (0 = i32, 1 = f32, 2 = f64). Codegen reads p5 to populate
    // the fn_type_table; is_f32_expr / is_f64_expr's AST_CALL fallback
    // resolves user-defined fn types via the table.
    let node = mk_node(14, name_start, name_len, body);
    __arena_push(params_head);
    __arena_push(ret_ty);
    node
}

// Stage 5 Iter A: parse `struct IDENT { f1: T1, f2: T2, ... }`.
// Caller has already verified the cursor sits on the `struct` IDENT.
// Iter D: each field-region entry is 3 slots (name_s, name_l,
// field_struct_idx). field_struct_idx is the struct_idx of the field's
// type IDENT if it is a registered struct, or -1 if scalar (i32/f32/etc.).
// Registers the (name, arity, fields_ptr) into struct_table so
// parse_primary can detect `IDENT { ... }` as a struct lit later.
// Returns a tag-54 AST_STRUCT_DECL node which codegen treats as a no-op.
// Stage 6: parse `enum Name { Variant1, Variant2(T1, T2), ... }`. Each
// variant gets a 0-based discriminant. Variants table layout: 4 slots
// per entry (name_s, name_l, arity, discriminant). Codegen uses tag 54
// (AST_STRUCT_DECL) — emits 0 bytes — so no new emit_ast_code arm.
// The folding works because both struct decl and enum decl are pure
// metadata at codegen time (the construction sites use existing tags
// AST_INT for unit variants and AST_TUPLE_LIT for payload variants).
fn parse_enum_decl(tok_base: i32, sb: i32) -> i32 {
    cur_advance(sb);                         // consume 'enum'
    let nk = cur_get(sb);
    let name_s = tok_p2(tok_base, nk);
    let name_l = tok_p3(tok_base, nk);
    cur_advance(sb);                         // consume name IDENT
    cur_advance(sb);                         // consume '{' (LBRACE = 5)
    let mut variant_count: i32 = 0;
    let mut variants_ptr: i32 = 0;
    let mut max_arity: i32 = 0;
    let mut keep: i32 = 1;
    while keep == 1 {
        let tt = tok_tag(tok_base, cur_get(sb));
        if tt == 6 {                         // RBRACE
            keep = 0;
        } else { if tt == 0 {                // EOF safety
            keep = 0;
        } else {
            // Variant name IDENT.
            let vk = cur_get(sb);
            let v_name_s = tok_p2(tok_base, vk);
            let v_name_l = tok_p3(tok_base, vk);
            cur_advance(sb);                 // consume variant-name IDENT
            // Optional `(T1, T2, ...)` payload-types list. Phase-0:
            // the type IDENTs are parsed and discarded; only the arity
            // is recorded (and folded into max_arity).
            let mut arity: i32 = 0;
            let after_name_t = tok_tag(tok_base, cur_get(sb));
            if after_name_t == 3 {           // '('
                cur_advance(sb);             // consume '('
                let mut keep_args: i32 = 1;
                while keep_args == 1 {
                    let at = tok_tag(tok_base, cur_get(sb));
                    if at == 4 {             // ')'
                        keep_args = 0;
                    } else { if at == 13 {   // ','
                        cur_advance(sb);
                    } else { if at == 0 {    // EOF safety
                        keep_args = 0;
                    } else {
                        // Type IDENT — just consume.
                        cur_advance(sb);
                        arity = arity + 1;
                    }}};
                }
                cur_advance(sb);             // consume ')'
            };
            // Push variant entry: (name_s, name_l, arity, discriminant).
            // Capture variants_ptr from the FIRST push so subsequent
            // variants append after it.
            let pushed = __arena_push(v_name_s);
            if variant_count == 0 {
                variants_ptr = pushed;
            };
            __arena_push(v_name_l);
            __arena_push(arity);
            __arena_push(variant_count);     // discriminant = 0-based index
            if arity > max_arity { max_arity = arity; };
            variant_count = variant_count + 1;
            // Optional COMMA between variants.
            if tok_tag(tok_base, cur_get(sb)) == 13 {
                cur_advance(sb);
            };
        }};
    }
    cur_advance(sb);                         // consume '}' (RBRACE = 6)
    enum_tab_add(sb, name_s, name_l, variant_count, variants_ptr, max_arity);
    // Reuse AST_STRUCT_DECL tag (54) — codegen treats both as 0-byte
    // metadata. Avoids adding a new emit_ast_code arm (Iter D Finding
    // #7 — host-parser recursion budget is tight at 45 arms).
    mk_node(54, 0, 0, 0)
}

fn parse_struct_decl(tok_base: i32, sb: i32) -> i32 {
    cur_advance(sb);                         // consume 'struct' IDENT
    let nk = cur_get(sb);
    let name_s = tok_p2(tok_base, nk);
    let name_l = tok_p3(tok_base, nk);
    cur_advance(sb);                         // consume name IDENT
    cur_advance(sb);                         // consume '{' (LBRACE = 5)
    let mut field_count: i32 = 0;
    let mut fields_ptr: i32 = 0;             // 0 if no fields
    let mut keep: i32 = 1;
    while keep == 1 {
        let tt = tok_tag(tok_base, cur_get(sb));
        if tt == 6 {                         // RBRACE
            keep = 0;
        } else { if tt == 0 {                // EOF safety
            keep = 0;
        } else {
            // Capture field-name token bytes BEFORE advancing.
            let fk = cur_get(sb);
            let f_name_s = tok_p2(tok_base, fk);
            let f_name_l = tok_p3(tok_base, fk);
            cur_advance(sb);                 // field-name IDENT
            cur_advance(sb);                 // ':' (COLON = 14)
            // Iter D: capture type IDENT bytes BEFORE advancing so we
            // can resolve nested struct types via struct_tab_lookup_idx.
            let tk = cur_get(sb);
            let t_s = tok_p2(tok_base, tk);
            let t_l = tok_p3(tok_base, tk);
            cur_advance(sb);                 // consume type IDENT
            let f_struct_idx = struct_tab_lookup_idx(sb, t_s, t_l);
            // Push (name_s, name_l, field_struct_idx) triple into
            // fields region. Capture fields_ptr from the FIRST push so
            // subsequent fields append after it (arena grows linearly).
            let pushed = __arena_push(f_name_s);
            if field_count == 0 {
                fields_ptr = pushed;
            };
            __arena_push(f_name_l);
            __arena_push(f_struct_idx);
            field_count = field_count + 1;
            if tok_tag(tok_base, cur_get(sb)) == 13 {  // optional COMMA
                cur_advance(sb);
            };
        }};
    }
    cur_advance(sb);                         // consume '}' (RBRACE = 6)
    struct_tab_add(sb, name_s, name_l, field_count, fields_ptr);
    mk_node(54, 0, 0, 0)
}

// Stage 7: parse a single pattern. Dispatches on the current token:
//   INT          -> PAT_LIT (tag 64) p1 = value
//                   if followed by `..` and another INT: PAT_RANGE (tag 67)
//                   p1 = lo, p2 = hi (exclusive). For Phase-0 only INT lo/hi
//                   are supported; PatBind for ranges deferred.
//   IDENT == "_" -> PAT_WILDCARD (tag 66)
//   IDENT::IDENT -> PAT_VARIANT (tag 69) p1 = disc, p2 = sub_pats_head
//                   p3 = enum_idx. Sub-pats reuse AST_TUPLE_CONS (tag 51).
//   IDENT        -> PAT_BIND (tag 65) p1 = name_start, p2 = name_len
//   LPAREN       -> PAT_TUPLE (tag 70) p1 = arity, p2 = sub_pats_head
//
// FLAT prefix-trap pattern: single ladder of let-rebinds, no nested
// if-else statements. Returns the AST node index.
fn parse_pattern(tok_base: i32, sb: i32) -> i32 {
    let k = cur_get(sb);
    let t = tok_tag(tok_base, k);
    if t == 1 {
        // INT literal pattern. Check for `..` to detect range.
        let v = tok_p1(tok_base, k);
        cur_advance(sb);                     // consume INT
        let nk = cur_get(sb);
        let nt = tok_tag(tok_base, nk);
        if nt == 43 {                        // TK_DOTDOT
            cur_advance(sb);                 // consume `..`
            let hk = cur_get(sb);
            let hi = tok_p1(tok_base, hk);
            cur_advance(sb);                 // consume hi INT
            mk_node(67, v, hi, 0)
        } else {
            mk_node(64, v, 0, 0)
        }
    } else { if t == 2 {
        // IDENT — could be `_` (wildcard), `EnumName::Variant(...)` (variant
        // pattern), or a plain bind name.
        let id_s = tok_p2(tok_base, k);
        let id_l = tok_p3(tok_base, k);
        // Wildcard: single `_` (1 char, byte 95).
        let is_wild = if id_l == 1 {
            let b0 = __arena_get(id_s);
            if b0 == 95 { 1 } else { 0 }
        } else { 0 };
        // Pre-check for `::` enum-variant path. Same FLAT pattern as
        // parse_primary's enum dispatch: peek tok+1, tok+2, tok+3.
        let e_idx_pre = enum_tab_lookup_idx(sb, id_s, id_l);
        let t1_pre = tok_tag(tok_base, k + 1);
        let t2_pre = tok_tag(tok_base, k + 2);
        let t3_pre = tok_tag(tok_base, k + 3);
        let is_enum_path = if e_idx_pre >= 0 {
            if t1_pre == 14 { if t2_pre == 14 { if t3_pre == 2 { 1 } else { 0 } } else { 0 } } else { 0 }
        } else { 0 };
        if is_wild == 1 {
            cur_advance(sb);                 // consume '_'
            mk_node(66, 0, 0, 0)
        } else { if is_enum_path == 1 {
            // PAT_VARIANT: consume IDENT, '::', '::', variant-IDENT.
            cur_advance(sb);                 // outer IDENT
            cur_advance(sb);                 // first ':'
            cur_advance(sb);                 // second ':'
            let vk = cur_get(sb);
            let v_name_s = tok_p2(tok_base, vk);
            let v_name_l = tok_p3(tok_base, vk);
            cur_advance(sb);                 // variant IDENT
            let disc = enum_tab_variant_lookup_disc(sb, e_idx_pre, v_name_s, v_name_l);
            let safe_disc = if disc < 0 { 0 } else { disc };
            // Optional `(sub_pat1, sub_pat2, ...)` — payload sub-patterns.
            let after_t = tok_tag(tok_base, cur_get(sb));
            let mut sub_head: i32 = 0;
            if after_t == 3 {                // '('
                cur_advance(sb);             // consume '('
                let first_pat = parse_pattern(tok_base, sb);
                sub_head = mk_node(51, first_pat, 0, 0);
                let mut tail_idx: i32 = sub_head;
                let mut keep: i32 = 1;
                while keep == 1 {
                    let at = tok_tag(tok_base, cur_get(sb));
                    if at == 4 {             // ')'
                        keep = 0;
                    } else { if at == 13 {   // ','
                        cur_advance(sb);
                        let next_pat = parse_pattern(tok_base, sb);
                        let new_node = mk_node(51, next_pat, 0, 0);
                        __arena_set(tail_idx + 2, new_node);
                        tail_idx = new_node;
                    } else { if at == 0 {    // EOF safety
                        keep = 0;
                    } else {
                        // Defensive: shouldn't happen given the grammar.
                        keep = 0;
                    }}};
                }
                cur_advance(sb);             // consume ')'
            };
            mk_node(69, safe_disc, sub_head, e_idx_pre)
        } else {
            // Plain identifier binding pattern.
            cur_advance(sb);                 // consume IDENT
            mk_node(65, id_s, id_l, 0)
        }}
    } else { if t == 3 {
        // LPAREN — tuple pattern (sub_pat1, sub_pat2, ...).
        cur_advance(sb);                     // consume '('
        let first_pat = parse_pattern(tok_base, sb);
        let mut sub_head: i32 = mk_node(51, first_pat, 0, 0);
        let mut tail_idx: i32 = sub_head;
        let mut arity: i32 = 1;
        let mut keep: i32 = 1;
        while keep == 1 {
            let at = tok_tag(tok_base, cur_get(sb));
            if at == 4 {                     // ')'
                keep = 0;
            } else { if at == 13 {           // ','
                cur_advance(sb);
                let nt2 = tok_tag(tok_base, cur_get(sb));
                if nt2 == 4 {                // trailing ',' before ')'
                    keep = 0;
                } else {
                    let next_pat = parse_pattern(tok_base, sb);
                    let new_node = mk_node(51, next_pat, 0, 0);
                    __arena_set(tail_idx + 2, new_node);
                    tail_idx = new_node;
                    arity = arity + 1;
                };
            } else { if at == 0 {            // EOF safety
                keep = 0;
            } else {
                keep = 0;
            }}};
        }
        cur_advance(sb);                     // consume ')'
        mk_node(70, arity, sub_head, 0)
    } else {
        // Unknown pattern token — produce wildcard as fallback.
        cur_advance(sb);
        mk_node(66, 0, 0, 0)
    }}}
}

// Stage 7: parse `match scrut { pat => body, pat => body, ... }`.
// Returns AST_MATCH (tag 62) node idx.
//   p1 = scrut_idx
//   p2 = arms_head_idx (linked list of AST_MATCH_ARM nodes)
//   p3 = unused
// Each arm AST_MATCH_ARM (tag 63):
//   p1 = pattern_idx
//   p2 = body_idx
//   p3 = next_arm_idx (0 at end)
// `match` keyword has already been peeked but NOT consumed by caller.
fn parse_match_expr(tok_base: i32, sb: i32) -> i32 {
    cur_advance(sb);                         // consume 'match' IDENT
    let scrut_idx = parse_expr_basic(tok_base, sb);
    cur_advance(sb);                         // consume '{'
    // Parse first arm.
    let first_pat = parse_pattern(tok_base, sb);
    cur_advance(sb);                         // consume '=>' (TK_FATARROW = 42)
    let first_body = parse_expr_basic(tok_base, sb);
    let arms_head = mk_node(63, first_pat, first_body, 0);
    let mut tail_idx: i32 = arms_head;
    let mut keep: i32 = 1;
    while keep == 1 {
        let at = tok_tag(tok_base, cur_get(sb));
        if at == 6 {                         // '}'
            keep = 0;
        } else { if at == 13 {               // ','
            cur_advance(sb);                 // consume ','
            // Allow trailing comma before '}'.
            let nt = tok_tag(tok_base, cur_get(sb));
            if nt == 6 {
                keep = 0;
            } else {
                let next_pat = parse_pattern(tok_base, sb);
                cur_advance(sb);             // consume '=>'
                let next_body = parse_expr_basic(tok_base, sb);
                let new_arm = mk_node(63, next_pat, next_body, 0);
                __arena_set(tail_idx + 3, new_arm);
                tail_idx = new_arm;
            };
        } else { if at == 0 {                // EOF safety
            keep = 0;
        } else {
            keep = 0;
        }}};
    }
    cur_advance(sb);                         // consume '}'
    mk_node(62, scrut_idx, arms_head, 0)
}
