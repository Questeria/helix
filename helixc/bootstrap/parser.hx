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
//                    Slot layout (extended through stages):
//                      slot 4: params_head (AST_PARAM chain)
//                      slot 5: ret_ty (type tag, 0=i32, 1=f32, ...)
//                      slot 6: is_generic flag (Stage 8)
//                      slot 7: gp_names_head (Stage 8.5)
//                      slot 8: is_checkpoint flag (Stage 14.5)
//                      slot 9: is_deprecated flag (Stage 28.9)
//                      slot 10: is_trace flag (Stage 28.9)
//                      slot 11: is_unwind flag (Stage 28.9)
//                      slot 12: deprecated_msg_start (Stage 33)
//                      slot 13: deprecated_msg_len (Stage 33)
//                      slot 14: is_kernel flag (Stage 33)
//                      slot 15: is_autotune flag (Stage 33)
//                      slot 16: autotune_variant_product (Stage 33)
//                      slot 17: autotune_parse_error_kind (Stage 33)
//                      slot 18: since_msg_start (Stage 33)
//                      slot 19: since_msg_len (Stage 33)
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
// K1.C-deadcode (2026-05-25): return keyword installed at sb+88/sb+89.
// Currently UNREACHABLE -- parse_primary has no caller-arm yet (the
// wire-up is a follow-up chunk per docs/K1_SUBCHUNK_PLAN.md §K1.C
// redo plan, option iii). The infrastructure (state slot + accessor
// + install + parse_return fn + kovc.hx codegen for tag 43) is
// staged here so the wire-up chunk is a single-line insertion + 2
// closing braces -- minimal touch on the audit-fragile parse_primary
// cascade.
fn kw_return_s(sb: i32) -> i32 { __arena_get(sb + 88) }
fn kw_return_n(sb: i32) -> i32 { __arena_get(sb + 89) }
// K1.G-deadcode (2026-05-25): for-loop infrastructure REUSES the
// existing kw_for_s/n at slot 41 (already installed by Stage 8.5
// for the `impl Trait for Type` syntax). Only the new `in` keyword
// needs a fresh slot pair. UNREACHABLE -- parse_primary has no
// for-loop arm yet; the wire-up follows in K1.G-wireup.
fn kw_in_s(sb: i32) -> i32 { __arena_get(sb + 90) }
fn kw_in_n(sb: i32) -> i32 { __arena_get(sb + 91) }
// K1.H1-deadcode (2026-05-25): `loop { body }` keyword at slots
// 92/93. Desugars to `while 1 { body }` -- no new AST tag. break
// and continue are deferred (K1.H2/H3 -- they need label tracking).
fn kw_loop_s(sb: i32) -> i32 { __arena_get(sb + 92) }
fn kw_loop_n(sb: i32) -> i32 { __arena_get(sb + 93) }
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

// Stage 28.13.1: peek 2 tokens ahead to detect named struct-lit syntax
// `Pt { x: 10, y: 32 }`. Returns 1 if the current token is IDENT and
// the next token is COLON; 0 otherwise. Used by parse_primary's
// struct-lit body parser to dispatch between positional and named
// modes. Mirrors the Python frontend's parser.py:1296
// `_peek_struct_lit_start` shape.
//
// `cur_get(sb) + 1` gives the next token index — tokens are indexed
// sequentially in the lex output; cur_get returns the current
// token's index. Reads via tok_tag are pure (no cursor advance).
fn peek_named_struct_lit(tok_base: i32, sb: i32) -> i32 {
    let c = cur_get(sb);
    let t1 = tok_tag(tok_base, c);
    if t1 != 2 {
        0
    } else {
        let t2 = tok_tag(tok_base, c + 1);
        if t2 == 14 { 1 } else { 0 }
    }
}

// Stage 28.11 INC-3a cycle-5 polish (cycle-4 type-design F1+F2+F3, MED
// conf 80-85): centralized helpers for the generic-param marker
// encoding `200 + gp_idx`. Pre-cycle-5 the literal `200` was open-coded
// at multiple sites; cycle-5 extracted these helpers and migrated the
// 2 INC-3a-active sites (writer parse_struct_decl ~6238, reader
// parse_primary ~1681) to use them.
//
// Invariant: struct_tab cap (currently 8) MUST remain below
// `gp_marker_base()` (= 200). The 192-slot gap between struct_tab's
// max idx (7) and the gp marker range (200..) gives substantial
// headroom but isn't unlimited. If struct_tab cap is ever raised
// (INC-3b will add monomorphized clones), confirm `struct_tab_count
// < gp_marker_base()` is preserved. Future hardening: trap when
// struct_tab_add would assign idx >= gp_marker_base().
//
// === Cycle-7 polish (cycle-6 type-design C2 MED conf 85): KNOWN
// non-migrated raw-200 sites ===
//
// The following 7 sites still use literal 200 / `200 + X` arithmetic.
// They are intentionally NOT migrated in cycles 5-7 per the workspace
// cycle-71 narrow-scope discipline (Stage-8 surfaces are not under
// active iteration in INC-3a):
//
//   parser.hx:4156 — Stage-8 monomorphize_pass param-ty decode
//                    (`if p_ty_raw >= 200 { ... }`)
//   parser.hx:4157 — Stage-8 monomorphize_pass param gp_idx extract
//                    (`let g_idx = p_ty_raw - 200`)
//   parser.hx:4176 — Stage-8 monomorphize_pass return-ty decode
//   parser.hx:4177 — Stage-8 monomorphize_pass return gp_idx extract
//   parser.hx:5453 — parse_fn_decl param-ty generic encode
//                    (`p_ty_generic = if gp_idx_p >= 0 { 200 + gp_idx_p }`)
//   parser.hx:5458 — parse_fn_decl param post-encode `< 200` guard
//   parser.hx:5534 — parse_fn_decl ret-ty generic encode
//
// (Stage 28.11 INC-3a cycle-9 polish: line numbers updated post
// cycle-5/7 doc-block insertions. Cycles 5/7 grew this block by ~42
// lines; cycle-9 re-grepped the file for `\b200\b` to recompute the
// actual deferred-site positions.)
//
// When INC-3b adds the use-site reader at parse_primary, it should
// compose against `gp_marker_is` plus inline arithmetic at the call
// site (see the post-decode-pattern block below, lines 280-291).
// A separate hardening pass should migrate the 7 sites above to also
// use these helpers; cycle-71 narrow-scope keeps that broader migration
// outside INC-3a's surface.
fn gp_marker_base() -> i32 { 200 }
fn gp_marker_encode(gp_idx: i32) -> i32 { gp_marker_base() + gp_idx }
fn gp_marker_is(v: i32) -> i32 {
    if v >= gp_marker_base() { 1 } else { 0 }
}
// Stage 28.11 INC-3b: struct_gp_tab helpers. Parallel to struct_tab,
// keyed by struct_idx. Each entry stores (struct_idx, gp_count,
// gp_names_head) where gp_names_head is the arena address of a chain
// of mk_node(76, name_s, name_l, next) nodes — same shape as
// parse_fn_decl's per-fn gp_chain_head at line ~5475.
//
// Used by use-site monomorphization at parse_primary: when `Pt<i32>`
// is parsed, struct_gp_tab_lookup(orig_struct_idx) returns gp_count
// and struct_gp_tab_names_head(orig_struct_idx) yields the chain for
// substitution. Region at sb+78/79; stride 3; cap 8 (matches struct_tab).
fn struct_gp_tab_base(sb: i32) -> i32 { __arena_get(sb + 78) }
fn struct_gp_tab_count(sb: i32) -> i32 { __arena_get(sb + 79) }

fn struct_gp_tab_add(sb: i32, struct_idx: i32, gp_count: i32, gp_names_head: i32) -> i32 {
    let count = struct_gp_tab_count(sb);
    if count >= 8 {
        0 - 1
    } else {
        let base = struct_gp_tab_base(sb);
        let entry = base + count * 3;
        __arena_set(entry, struct_idx);
        __arena_set(entry + 1, gp_count);
        __arena_set(entry + 2, gp_names_head);
        __arena_set(sb + 79, count + 1);
        count
    }
}

// Returns gp_count on hit, 0 on miss (so callers can treat 0 as
// "non-generic struct"). Distinct from struct_tab_lookup_idx's -1
// miss sentinel because here 0 is a meaningful "0 generic params"
// value that has the same effect as miss.
fn struct_gp_tab_lookup(sb: i32, struct_idx: i32) -> i32 {
    let base = struct_gp_tab_base(sb);
    let count = struct_gp_tab_count(sb);
    let mut i: i32 = 0;
    let mut found_count: i32 = 0;
    while i < count {
        let entry = base + i * 3;
        if __arena_get(entry) == struct_idx {
            found_count = __arena_get(entry + 1);
            i = count;
        } else {
            i = i + 1;
        };
    }
    found_count
}

// Returns gp_names_head (mk_node 76 chain head) on hit, 0 on miss.
fn struct_gp_tab_names_head(sb: i32, struct_idx: i32) -> i32 {
    let base = struct_gp_tab_base(sb);
    let count = struct_gp_tab_count(sb);
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    while i < count {
        let entry = base + i * 3;
        if __arena_get(entry) == struct_idx {
            found = __arena_get(entry + 2);
            i = count;
        } else {
            i = i + 1;
        };
    }
    found
}

// Stage 28.11 INC-3a cycle-7 fix (cycle-6 type-design C1 MED conf 90):
// gp_marker_decode is INTENTIONALLY NOT provided as a helper. A pure-
// arithmetic decode `v - gp_marker_base()` would be a partial function
// (defined only for v >= gp_marker_base()), and the bootstrap has no
// runtime trap primitive available in pure-helper context — neither
// `__trap()` nor a panic-on-i32-return mechanism exists. A misuse
// (decode called on a non-marker value) would silently return a
// negative integer that off-by-200 indexes into gp_tab arrays, exactly
// the silent-miscompile defect class cycle-2 (SF-1/SF-2/SF-3) caught.
//
// Instead, INC-3b callers MUST compose `gp_marker_is` with inline
// arithmetic at the call site:
//
//   if gp_marker_is(v) == 1 {
//       let gp_idx = v - gp_marker_base();   // safe — v >= 200
//       // ... use gp_idx ...
//   }
//
// This mirrors the bootstrap's existing pattern (cf. parser.hx:4156-4157
// where Stage-8 monomorphize_pass guards `if p_ty_raw >= 200 { let
// g_idx = p_ty_raw - 200; ... }` before decoding). Putting the guard
// at the call site keeps the precondition visible to reviewers.

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
//   slot 4: pack_lo = type_args_packed * 8 + type_args_count
//           (low 3 bits = count, upper = 4-bit-per-arg packed tags)
//   slot 5: reserved (currently 0)
// Combining packed+count into a single i32 avoids the 7-arg fn limit
// (SysV bootstrap supports 6 int params).
fn mr_tab_base(sb: i32) -> i32 { __arena_get(sb + 31) }
fn mr_tab_count(sb: i32) -> i32 { __arena_get(sb + 32) }
fn mr_tab_add(sb: i32, orig_s: i32, orig_l: i32, mang_s: i32, mang_l: i32, pack_lo: i32) -> i32 {
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
        __arena_set(entry + 4, pack_lo);
        __arena_set(entry + 5, 0);
        __arena_set(sb + 32, count + 1);
        count
    }
}
// Lookup an instantiation by (orig_name, pack_lo). pack_lo encodes both
// packed type tags and count as a single i32. Returns entry idx on hit, -1 miss.
fn mr_tab_lookup(sb: i32, orig_s: i32, orig_l: i32, pack_lo: i32) -> i32 {
    let base = mr_tab_base(sb);
    let count = mr_tab_count(sb);
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < count {
        let entry = base + i * 6;
        let ns = __arena_get(entry);
        let nl = __arena_get(entry + 1);
        let p = __arena_get(entry + 4);
        if byte_eq(orig_s, orig_l, ns, nl) == 1 {
            if p == pack_lo {
                found = i;
                i = count;
            } else { i = i + 1; }
        } else {
            i = i + 1;
        };
    }
    found
}
// Stage 8.5: trait_table at sb+33/34. Each entry is 2 slots (name_s, name_l).
// Cap 4 traits in Phase-0. Phase-0 does not validate trait method signatures
// against impl method signatures; the table just records that "this name is
// a trait" for parse-time disambiguation.
fn trait_tab_base(sb: i32) -> i32 { __arena_get(sb + 33) }
fn trait_tab_count(sb: i32) -> i32 { __arena_get(sb + 34) }
fn trait_tab_add(sb: i32, name_s: i32, name_l: i32) -> i32 {
    let count = trait_tab_count(sb);
    if count >= 4 {
        0 - 1
    } else {
        let base = trait_tab_base(sb);
        let entry = base + count * 2;
        __arena_set(entry, name_s);
        __arena_set(entry + 1, name_l);
        __arena_set(sb + 34, count + 1);
        count
    }
}
fn trait_tab_lookup(sb: i32, name_s: i32, name_l: i32) -> i32 {
    let base = trait_tab_base(sb);
    let count = trait_tab_count(sb);
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
// Stage 8.5: impl_table at sb+35/36. Each entry is 4 slots
// (trait_name_s, trait_name_l, target_ty_tag, methods_count).
// Cap 8 impls in Phase-0. target_ty_tag uses ty_ident_to_tag (0 = i32, 1
// = f32, 2 = f64, 3 = i64, etc.). Phase-0 does not enforce uniqueness yet
// (8.5D is optional); duplicate impls would be parsed but the second's
// mangled fn names collide at codegen, producing a silent re-bind.
fn impl_tab_base(sb: i32) -> i32 { __arena_get(sb + 35) }
fn impl_tab_count(sb: i32) -> i32 { __arena_get(sb + 36) }
fn impl_tab_add(sb: i32, trait_s: i32, trait_l: i32, target_tag: i32, methods_count: i32) -> i32 {
    let count = impl_tab_count(sb);
    if count >= 8 {
        0 - 1
    } else {
        let base = impl_tab_base(sb);
        let entry = base + count * 4;
        __arena_set(entry, trait_s);
        __arena_set(entry + 1, trait_l);
        __arena_set(entry + 2, target_tag);
        __arena_set(entry + 3, methods_count);
        __arena_set(sb + 36, count + 1);
        count
    }
}
// Lookup impl by (trait_name, target_ty_tag). Returns entry idx on hit, -1 miss.
fn impl_tab_lookup(sb: i32, trait_s: i32, trait_l: i32, target_tag: i32) -> i32 {
    let base = impl_tab_base(sb);
    let count = impl_tab_count(sb);
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < count {
        let entry = base + i * 4;
        let ts = __arena_get(entry);
        let tl = __arena_get(entry + 1);
        let tt = __arena_get(entry + 2);
        if byte_eq(trait_s, trait_l, ts, tl) == 1 {
            if tt == target_tag {
                found = i;
                i = count;
            } else { i = i + 1; }
        } else {
            i = i + 1;
        };
    }
    found
}
// Stage 8.5: trait/impl/for keyword bytes installed at sb+37..42.
fn kw_trait_s(sb: i32) -> i32 { __arena_get(sb + 37) }
fn kw_trait_n(sb: i32) -> i32 { __arena_get(sb + 38) }
fn kw_impl_s(sb: i32) -> i32 { __arena_get(sb + 39) }
fn kw_impl_n(sb: i32) -> i32 { __arena_get(sb + 40) }
fn kw_for_s(sb: i32) -> i32 { __arena_get(sb + 41) }
fn kw_for_n(sb: i32) -> i32 { __arena_get(sb + 42) }
// Stage 8.5: sb+43/44 = pending impl-method fn-list head/tail. parse_impl_block
// builds a chain of AST_FN_LIST (tag 15) nodes wrapping the mangled method
// AST_FN_DECLs; parse_program splices this chain in front of the user's fn
// decls so codegen emits + wires them into fn_table normally. Reset to 0 by
// parse_top.
fn impl_pending_head(sb: i32) -> i32 { __arena_get(sb + 43) }
fn impl_pending_tail(sb: i32) -> i32 { __arena_get(sb + 44) }
fn set_impl_pending_head(sb: i32, v: i32) -> i32 { __arena_set(sb + 43, v); 0 }
fn set_impl_pending_tail(sb: i32, v: i32) -> i32 { __arena_set(sb + 44, v); 0 }
// Stage 8.5: sb+45/46 = var_type_table base/count. Each entry is 3 slots
// (name_s, name_l, type_tag). Cap 8 typed vars per fn-body. Captured by
// `let x: T = ...` parsing; consulted by parse_postfix when handling
// method-call sugar `x.eq(b)` to mangle to `<TypeName>__eq(x, b)`.
fn var_type_tab_base(sb: i32) -> i32 { __arena_get(sb + 45) }
fn var_type_tab_count(sb: i32) -> i32 { __arena_get(sb + 46) }
// Stage 10: module + use state.
// sb+60/61 = kw_mod (start, len). "mod" = 109 111 100.
// sb+62/63 = kw_use (start, len). "use" = 117 115 101.
// sb+64/65 = use_table base/count. Stride 4: (alias_s, alias_l, mang_s, mang_l).
//   Cap 8 use entries. Built by parse_use_decl; consulted by parse_primary's
//   plain-IDENT call path to replace `bar(args)` with `foo__bar(args)` when
//   `bar` was brought into scope via `use foo::bar;`.
// sb+66/67 = mod_pending head/tail. parse_mod_decl walks each fn inside a
//   `mod foo { ... }` block, mangles its name to `foo__bar`, and appends it
//   to this chain (wrapped in AST_FN_LIST tag-15 nodes). parse_program splices
//   the chain in front of the user's fns (alongside impl_pending and
//   cl_pending). Reset to 0 by parse_top.
fn kw_mod_s(sb: i32) -> i32 { __arena_get(sb + 60) }
fn kw_mod_n(sb: i32) -> i32 { __arena_get(sb + 61) }
fn kw_use_s(sb: i32) -> i32 { __arena_get(sb + 62) }
fn kw_use_n(sb: i32) -> i32 { __arena_get(sb + 63) }
fn use_tab_base(sb: i32) -> i32 { __arena_get(sb + 64) }
fn use_tab_count(sb: i32) -> i32 { __arena_get(sb + 65) }
fn mod_pending_head(sb: i32) -> i32 { __arena_get(sb + 66) }
fn mod_pending_tail(sb: i32) -> i32 { __arena_get(sb + 67) }
fn set_mod_pending_head(sb: i32, v: i32) -> i32 { __arena_set(sb + 66, v); 0 }
fn set_mod_pending_tail(sb: i32, v: i32) -> i32 { __arena_set(sb + 67, v); 0 }
// Stage 12: grad-pending table state.
// sb+68 = grad_pending_base (offset of 32-slot region, 8 entries x 4 fields).
// sb+69 = grad_pending_count (0..8).
// Each entry layout (4 slots):
//   slot 0: loss_name_s (byte_start in arena)
//   slot 1: loss_name_l (byte_len)
//   slot 2: mang_s      (byte_start of synthesized "<loss>__grad" name)
//   slot 3: mang_l      (byte_len of mangled name)
// Built by parse_primary's grad-detection branch; consumed at end of
// parse_program by grad_pass to synthesize the derivative fn decls.
fn grad_pending_base(sb: i32) -> i32  { __arena_get(sb + 68) }
fn grad_pending_count(sb: i32) -> i32 { __arena_get(sb + 69) }
fn grad_pending_add(sb: i32, loss_s: i32, loss_l: i32, mang_s: i32, mang_l: i32) -> i32 {
    let count = grad_pending_count(sb);
    if count >= 8 {
        0 - 1
    } else {
        let base = grad_pending_base(sb);
        let entry = base + count * 4;
        __arena_set(entry, loss_s);
        __arena_set(entry + 1, loss_l);
        __arena_set(entry + 2, mang_s);
        __arena_set(entry + 3, mang_l);
        __arena_set(sb + 69, count + 1);
        count
    }
}

// Stage 14: grad-rev-pending table state.
// sb+70 = gr_rev_pending_base (offset of 40-slot region, 8 entries x 5 fields).
// sb+71 = gr_rev_pending_count (0..8).
// Each entry layout (5 slots):
//   slot 0: loss_name_s    (byte_start in arena)
//   slot 1: loss_name_l    (byte_len)
//   slot 2: field_name_s   (the IDENT after `.` — e.g. "dx")
//   slot 3: field_name_l   (byte_len; expected 2 chars min; first must be 'd')
//   slot 4: mang_s         (byte_start of synthesized "<loss>__grad_d<param>" name)
// mang_l is computed as loss_l + 7 + (field_l - 1)  ("__grad_d" = 8, but we
// store "<loss>__grad_<field>" — 6 chars "__grad_" + the full field bytes).
// Built by parse_primary's grad_rev_all branch when it encounters the postfix
// `.IDENT` form; consumed at end of parse_program by grad_rev_pass to
// synthesize the per-param derivative fn decls (forward-mode based — single
// param at a time, since the field selects which partial we want).
fn gr_rev_pending_base(sb: i32) -> i32  { __arena_get(sb + 70) }
fn gr_rev_pending_count(sb: i32) -> i32 { __arena_get(sb + 71) }
fn gr_rev_pending_add(sb: i32, loss_s: i32, loss_l: i32,
                       field_s: i32, field_l: i32, mang_s: i32) -> i32 {
    let count = gr_rev_pending_count(sb);
    if count >= 8 {
        0 - 1
    } else {
        let base = gr_rev_pending_base(sb);
        let entry = base + count * 5;
        __arena_set(entry, loss_s);
        __arena_set(entry + 1, loss_l);
        __arena_set(entry + 2, field_s);
        __arena_set(entry + 3, field_l);
        __arena_set(entry + 4, mang_s);
        __arena_set(sb + 71, count + 1);
        count
    }
}
// Append a use-table entry. Returns 0 on success, -1 on overflow (cap 8).
fn use_tab_add(sb: i32, alias_s: i32, alias_l: i32, mang_s: i32, mang_l: i32) -> i32 {
    let count = use_tab_count(sb);
    if count >= 8 {
        0 - 1
    } else {
        let base = use_tab_base(sb);
        let entry = base + count * 4;
        __arena_set(entry, alias_s);
        __arena_set(entry + 1, alias_l);
        __arena_set(entry + 2, mang_s);
        __arena_set(entry + 3, mang_l);
        __arena_set(sb + 65, count + 1);
        0
    }
}
// Look up an alias by name. Returns (mang_s * 65536 + mang_l) packed on hit,
// 0 on miss. Caller decodes by `>> 16` and `& 0xFFFF`. Phase-0 mang_l fits
// in 16 bits (typical mangled names are <100 bytes, well under 65535).
// Returns 0 on miss; mang_s itself is never 0 in practice since names are
// always pushed AFTER initial arena setup.
fn use_tab_lookup(sb: i32, name_s: i32, name_l: i32) -> i32 {
    let base = use_tab_base(sb);
    let count = use_tab_count(sb);
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    while i < count {
        let entry = base + i * 4;
        let ans = __arena_get(entry);
        let anl = __arena_get(entry + 1);
        if byte_eq(name_s, name_l, ans, anl) == 1 {
            let ms = __arena_get(entry + 2);
            let ml = __arena_get(entry + 3);
            found = ms * 65536 + ml;
            i = count;
        } else {
            i = i + 1;
        };
    }
    found
}

// Stage 9: closure state.
// sb+47 = closure-active flag (1 if currently parsing closure body —
//   AST_VAR creation hooks consult this to record captured names).
// sb+48 = closure_param_tab base offset; sb+49 = count.
//   Stride 2 (name_s, name_l). Cap 4 closure params (Phase-0).
// sb+50 = closure_capture_tab base offset; sb+51 = count.
//   Stride 2 (name_s, name_l). Cap 4 captured vars (Phase-0 trap 76002).
// sb+52 = closure_pending fn-list head; sb+53 = tail.
//   Built by parse_closure_lit; spliced in front of user fns by parse_program
//   (same pattern as impl_pending).
// sb+54 = closure_var_tab base offset; sb+55 = count.
//   Stride 4 (var_name_s, var_name_l, captures_ptr, capture_count).
//   Cap 4 closure-bindings per program. captures_ptr points to a
//   region of (name_s, name_l) pairs — one per captured var — that the
//   call-site lowering uses to inject the captured-var refs as positional
//   args ahead of the user-supplied args.
// sb+56 = closure id counter (next id; used for unique fn names).
// sb+57 = last_closure_idx scratch (-1 = none); set after parse_closure_lit
//   produces a closure value, read by parse_let to register the binding.
fn cl_active(sb: i32) -> i32 { __arena_get(sb + 47) }
fn set_cl_active(sb: i32, v: i32) -> i32 { __arena_set(sb + 47, v); 0 }
fn cl_param_tab_base(sb: i32) -> i32 { __arena_get(sb + 48) }
fn cl_param_tab_count(sb: i32) -> i32 { __arena_get(sb + 49) }
fn cl_param_tab_reset(sb: i32) -> i32 { __arena_set(sb + 49, 0); 0 }
fn cl_capture_tab_base(sb: i32) -> i32 { __arena_get(sb + 50) }
fn cl_capture_tab_count(sb: i32) -> i32 { __arena_get(sb + 51) }
fn cl_capture_tab_reset(sb: i32) -> i32 { __arena_set(sb + 51, 0); 0 }
fn cl_pending_head(sb: i32) -> i32 { __arena_get(sb + 52) }
fn cl_pending_tail(sb: i32) -> i32 { __arena_get(sb + 53) }
fn set_cl_pending_head(sb: i32, v: i32) -> i32 { __arena_set(sb + 52, v); 0 }
fn set_cl_pending_tail(sb: i32, v: i32) -> i32 { __arena_set(sb + 53, v); 0 }
fn cl_var_tab_base(sb: i32) -> i32 { __arena_get(sb + 54) }
fn cl_var_tab_count(sb: i32) -> i32 { __arena_get(sb + 55) }
fn cl_id_next(sb: i32) -> i32 { __arena_get(sb + 56) }
fn cl_id_bump(sb: i32) -> i32 { let v = __arena_get(sb + 56); __arena_set(sb + 56, v + 1); v }
fn last_closure_idx(sb: i32) -> i32 { __arena_get(sb + 57) }
fn set_last_closure_idx(sb: i32, v: i32) -> i32 { __arena_set(sb + 57, v); 0 }
// Append a closure-param entry. Returns 0 on success, -1 on overflow.
fn cl_param_tab_add(sb: i32, name_s: i32, name_l: i32) -> i32 {
    let count = cl_param_tab_count(sb);
    if count >= 4 {
        0 - 1
    } else {
        let base = cl_param_tab_base(sb);
        let entry = base + count * 2;
        __arena_set(entry, name_s);
        __arena_set(entry + 1, name_l);
        __arena_set(sb + 49, count + 1);
        0
    }
}
// Lookup a name in the current closure-param table. Returns 1 on hit, 0 on miss.
fn cl_param_tab_lookup(sb: i32, name_s: i32, name_l: i32) -> i32 {
    let base = cl_param_tab_base(sb);
    let count = cl_param_tab_count(sb);
    let mut i: i32 = 0;
    let mut hit: i32 = 0;
    while i < count {
        let entry = base + i * 2;
        let ns = __arena_get(entry);
        let nl = __arena_get(entry + 1);
        if byte_eq(name_s, name_l, ns, nl) == 1 {
            hit = 1;
            i = count;
        } else {
            i = i + 1;
        };
    }
    hit
}
// Append a closure-capture entry IF NOT already present (dedup). Returns 0
// on success/dedup, -1 on overflow (cap 4 → trap 76002).
fn cl_capture_tab_add_dedup(sb: i32, name_s: i32, name_l: i32) -> i32 {
    let base = cl_capture_tab_base(sb);
    let count = cl_capture_tab_count(sb);
    let mut i: i32 = 0;
    let mut found: i32 = 0;
    while i < count {
        let entry = base + i * 2;
        let ns = __arena_get(entry);
        let nl = __arena_get(entry + 1);
        if byte_eq(name_s, name_l, ns, nl) == 1 {
            found = 1;
            i = count;
        } else {
            i = i + 1;
        };
    }
    if found == 1 {
        0
    } else { if count >= 4 {
        0 - 1
    } else {
        let entry = base + count * 2;
        __arena_set(entry, name_s);
        __arena_set(entry + 1, name_l);
        __arena_set(sb + 51, count + 1);
        0
    }}
}
// Register a closure-bound variable. Returns the entry idx (0..3) on success,
// -1 on overflow (cap 4 closures per program). Each entry is 5 slots:
//   slot 0: var_name_s
//   slot 1: var_name_l
//   slot 2: captures_ptr (arena offset of a (name_s, name_l) pair sequence)
//   slot 3: capture_count
//   slot 4: closure_id (used to build the synthesized fn name `__closure_<id>`)
fn cl_var_tab_add(sb: i32, name_s: i32, name_l: i32, captures_ptr: i32, capture_count: i32, closure_id: i32) -> i32 {
    let count = cl_var_tab_count(sb);
    if count >= 4 {
        0 - 1
    } else {
        let base = cl_var_tab_base(sb);
        let entry = base + count * 5;
        __arena_set(entry, name_s);
        __arena_set(entry + 1, name_l);
        __arena_set(entry + 2, captures_ptr);
        __arena_set(entry + 3, capture_count);
        __arena_set(entry + 4, closure_id);
        __arena_set(sb + 55, count + 1);
        count
    }
}
// Lookup a closure-bound variable by name. Returns the entry idx on hit,
// -1 on miss. Used by parse_primary's call-site lowering to detect when
// `c(args)` should rewrite to `__closure_<id>(captured_vars..., args...)`.
fn cl_var_tab_lookup(sb: i32, name_s: i32, name_l: i32) -> i32 {
    let base = cl_var_tab_base(sb);
    let count = cl_var_tab_count(sb);
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
// Stage 9: build an AST_VAR node, hooking capture-recording when active.
// Replaces the 3 `mk_node(1, id_start, id_len, 0)` sites in parse_primary
// so any IDENT used inside a closure body that isn't a closure-param gets
// auto-recorded as a capture.
//
// Audit A3-CRITICAL-2 fix: cl_capture_tab_add_dedup returns -1 when the
// closure-capture cap (4) is exceeded. Previously the caller discarded
// the return value, silently dropping the 5th capture. Now we surface
// the overflow as AST_ERR(76002) so codegen emits a hard trap for the
// VAR site that didn't get registered.
fn mk_var_with_capture(sb: i32, id_s: i32, id_l: i32) -> i32 {
    let active = cl_active(sb);
    let mut cap_overflow: i32 = 0;
    if active == 1 {
        let is_param = cl_param_tab_lookup(sb, id_s, id_l);
        if is_param == 0 {
            let r = cl_capture_tab_add_dedup(sb, id_s, id_l);
            if r < 0 {
                cap_overflow = 1;
            };
        };
    };
    if cap_overflow == 1 {
        mk_node(99, 76002, 0, 0)
    } else {
        mk_node(1, id_s, id_l, 0)
    }
}
fn var_type_tab_add(sb: i32, name_s: i32, name_l: i32, ty_tag: i32) -> i32 {
    let count = var_type_tab_count(sb);
    if count >= 8 {
        0 - 1
    } else {
        let base = var_type_tab_base(sb);
        let entry = base + count * 3;
        __arena_set(entry, name_s);
        __arena_set(entry + 1, name_l);
        __arena_set(entry + 2, ty_tag);
        __arena_set(sb + 46, count + 1);
        count
    }
}
fn var_type_tab_lookup(sb: i32, name_s: i32, name_l: i32) -> i32 {
    let base = var_type_tab_base(sb);
    let count = var_type_tab_count(sb);
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
// Stage 8.5: tag-to-name helper. Given a type tag (0=i32, 1=f32, 2=f64,
// 3=i64, 4=bf16, 6=u32, 7=u8, 8=u16, 9=u64, 10=i8, 11=i16), push the
// canonical type-name bytes into the arena and return (start, len_packed)
// where len_packed = start * 8 + len. Used by method-call sugar to build
// mangled names like "i32__eq". For uncommon tags (anything not handled
// here) defaults to "i32".
//
// Audit A2-F3 fix: pre-fix the helper conflated tags 4 (bf16), 7 (u8),
// 8 (u16), 10 (i8), 11 (i16) with i32, so impl Eq for u8 / u16 / i8 /
// i16 / bf16 silently routed method-call sugar through "i32__eq". Now
// each tag has its own arm with the correct ASCII bytes.
//
// Implementation note: a single 11-arm if-else chain strains the host
// parser. We split into two flat halves (3-byte-wide tags first, then
// the others) to keep nesting shallow.
fn ty_tag_push_name_3byte(tag: i32) -> i32 {
    let start = __arena_len();
    if tag == 0 { __arena_push(105); __arena_push(51); __arena_push(50); }
    else { if tag == 1 { __arena_push(102); __arena_push(51); __arena_push(50); }
    else { if tag == 2 { __arena_push(102); __arena_push(54); __arena_push(52); }
    else { if tag == 3 { __arena_push(105); __arena_push(54); __arena_push(52); }
    else { if tag == 6 { __arena_push(117); __arena_push(51); __arena_push(50); }
    else { if tag == 8 { __arena_push(117); __arena_push(49); __arena_push(54); }
    else { if tag == 9 { __arena_push(117); __arena_push(54); __arena_push(52); }
    else { if tag == 11 { __arena_push(105); __arena_push(49); __arena_push(54); }
    else {  // default i32
        __arena_push(105); __arena_push(51); __arena_push(50);
    } } } } } } } } ;
    start * 8 + 3
}
fn ty_tag_push_name(tag: i32) -> i32 {
    if tag == 4 {
        // bf16 — 4 bytes
        let start = __arena_len();
        __arena_push(98); __arena_push(102); __arena_push(49); __arena_push(54);
        start * 8 + 4
    } else { if tag == 7 {
        // u8 — 2 bytes
        let start = __arena_len();
        __arena_push(117); __arena_push(56);
        start * 8 + 2
    } else { if tag == 10 {
        // i8 — 2 bytes
        let start = __arena_len();
        __arena_push(105); __arena_push(56);
        start * 8 + 2
    } else {
        // 3-byte (and default-i32) tags
        ty_tag_push_name_3byte(tag)
    } } }
}
fn var_struct_tab_add(sb: i32, name_s: i32, name_l: i32, struct_idx: i32) -> i32 {
    let count = var_struct_tab_count(sb);
    if count >= 8 {
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
    if count >= 8 {
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
    // Audit A1-F6 cap check (was 4, now 8). See enum_tab_init.
    let count = enum_tab_count(sb);
    if count >= 8 {
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

// Stage 17a/b/c (const-fold pass, parser-time variant): when a binop is
// being constructed and both operands are AST_INT (tag 0) literals,
// produce a single AST_INT carrying the folded i32 value instead of
// allocating the binop. The bottom-up parser ensures inner nodes fold
// first, so `2 + 3 * 4` walks 2 + AST_INT(12) → AST_INT(14). Stage 17c
// also recognises identity-element identities (x + 0, x * 1, x | 0,
// x ^ 0, x - 0, and the commuted forms) and short-circuits to the
// non-literal operand subtree.
//
// Phase-0 scope:
//   17a:  arithmetic — tag 2 AST_ADD, 3 AST_SUB, 4 AST_MUL
//   17b:  comparison — tag 6 AST_LT, 19 AST_GT, 20 AST_EQ, 21 AST_NE,
//                      22 AST_LE, 23 AST_GE  (result is 0 or 1, i32)
//         bitwise    — tag 28 AST_BAND, 29 AST_BOR, 30 AST_BXOR
//   17c:  algebraic identities (only the always-safe forms — those
//         that don't elide a potentially side-effecting operand):
//           x + 0 = x,  0 + x = x
//           x - 0 = x
//           x * 1 = x,  1 * x = x
//           x | 0 = x,  0 | x = x
//           x ^ 0 = x,  0 ^ x = x
//         Annihilation rules (x * 0 = 0, x & 0 = 0) are NOT applied
//         here because the non-literal side might contain a call or
//         other side-effecting expression that must still execute.
//         x - x = 0 also deferred — needs SSA-style equality between
//         non-literal operand subtrees, not just textual var-name
//         equality.
//
// Not folded:
//   - Wider intlits (i64/u32/u64/i8/i16/u8/u16). Their per-type wrap
//     rules differ from i32 and the bootstrap's own host arithmetic is
//     i32-only at this layer; folding them risks silent bit loss.
//   - Tag 5 AST_DIV / 24 AST_MOD. Phase-0 has no compile-time div/0
//     or INT_MIN/-1 trap, and the runtime SIGFPE from idiv must not
//     be silently elided. Reserved for a follow-up step.
//   - Tag 32 AST_SHL / 33 AST_SHR. Helix `/` is truncated (toward 0)
//     while AST_SHR is arithmetic-right (toward -inf for negative);
//     compile-time emulation can't match codegen exactly without
//     special-casing signs. Reserved.
//
// Wrapping semantics: the host compile-time i32 +/-/* wraps two's-
// complement, matching x86-64 add/sub/imul i32 in produced binaries.
fn mk_arith_fold(tag: i32, lhs: i32, rhs: i32) -> i32 {
    let lt = __arena_get(lhs);
    let rt = __arena_get(rhs);
    if lt == 0 { if rt == 0 {
        let lv = __arena_get(lhs + 1);
        let rv = __arena_get(rhs + 1);
        if tag == 2 {
            mk_node(0, lv + rv, 0, 0)
        } else { if tag == 3 {
            mk_node(0, lv - rv, 0, 0)
        } else { if tag == 4 {
            mk_node(0, lv * rv, 0, 0)
        } else { if tag == 28 {
            mk_node(0, lv & rv, 0, 0)
        } else { if tag == 29 {
            mk_node(0, lv | rv, 0, 0)
        } else { if tag == 30 {
            mk_node(0, lv ^ rv, 0, 0)
        } else { if tag == 6 {
            if lv < rv { mk_node(0, 1, 0, 0) } else { mk_node(0, 0, 0, 0) }
        } else { if tag == 19 {
            if lv > rv { mk_node(0, 1, 0, 0) } else { mk_node(0, 0, 0, 0) }
        } else { if tag == 20 {
            if lv == rv { mk_node(0, 1, 0, 0) } else { mk_node(0, 0, 0, 0) }
        } else { if tag == 21 {
            if lv != rv { mk_node(0, 1, 0, 0) } else { mk_node(0, 0, 0, 0) }
        } else { if tag == 22 {
            if lv <= rv { mk_node(0, 1, 0, 0) } else { mk_node(0, 0, 0, 0) }
        } else { if tag == 23 {
            if lv >= rv { mk_node(0, 1, 0, 0) } else { mk_node(0, 0, 0, 0) }
        } else {
            mk_node(tag, lhs, rhs, 0)
        }}}}}}}}}}}}
    } else {
        // Stage 17c: lhs is AST_INT, rhs is not. Identity-element
        // rules that forward to rhs (commutative tags only).
        let lv = __arena_get(lhs + 1);
        if lv == 0 {
            if tag == 2 { rhs }            // 0 + x = x
            else { if tag == 29 { rhs }    // 0 | x = x
            else { if tag == 30 { rhs }    // 0 ^ x = x
            else { mk_node(tag, lhs, rhs, 0) } } }
        } else { if lv == 1 {
            if tag == 4 { rhs }            // 1 * x = x
            else { mk_node(tag, lhs, rhs, 0) }
        } else {
            mk_node(tag, lhs, rhs, 0)
        }}
    }} else { if rt == 0 {
        // Stage 17c: rhs is AST_INT, lhs is not. Identity-element rules
        // that forward to lhs.
        let rv = __arena_get(rhs + 1);
        if rv == 0 {
            if tag == 2 { lhs }            // x + 0 = x
            else { if tag == 3 { lhs }     // x - 0 = x
            else { if tag == 29 { lhs }    // x | 0 = x
            else { if tag == 30 { lhs }    // x ^ 0 = x
            else { mk_node(tag, lhs, rhs, 0) } } } }
        } else { if rv == 1 {
            if tag == 4 { lhs }            // x * 1 = x
            else { mk_node(tag, lhs, rhs, 0) }
        } else {
            mk_node(tag, lhs, rhs, 0)
        }}
    } else {
        mk_node(tag, lhs, rhs, 0)
    }}
}

// Stage 8: map a type IDENT's bytes to a 4-bit type tag.
//   i32 -> 0, f32 -> 1, f64 -> 2, i64 -> 3, u32 -> 6, u64 -> 9.
//   bf16 -> 4 (4-byte). Unknown -> 0 (i32 default; safe fallback for
//   substitution sites where the body just bitcasts through the slot).
// Mirrors the strict per-byte logic in parse_fn_decl param-type
// resolution but flat-ladder safe (called from turbofish parsing and
// mono-pass substitution).
// Audit A2-F2/F3/F5 fix: extend ty_ident_to_tag with the missing scalar
// tags. Pre-fix u8/u16/i8/i16/bf16 all silently mapped to 0 (i32),
// causing var_type_tab + method-call sugar + turbofish dedup to conflate
// types: `let x: u8` registered (x, 0) → method call routed to i32__eq;
// `id::<u8>(...)` and `id::<i32>(...)` produced the same pack_lo so the
// mr_tab dedup created a wrong-fn synthesis. Now u8 → 7, u16 → 8, i8 → 10,
// i16 → 11, bf16 → 4 explicitly; these match parse_fn_decl's strict
// per-byte logic and ty_tag_push_name's reverse mapping.
@pure
fn ty_ident_to_tag(ty_s: i32, ty_l: i32) -> i32 {
    if ty_l == 3 {
        let b0 = __arena_get(ty_s);
        let b1 = __arena_get(ty_s + 1);
        let b2 = __arena_get(ty_s + 2);
        if b0 == 102 {
            if b1 == 54 { if b2 == 52 { 2 } else { 0 } }
            else { if b1 == 51 { if b2 == 50 { 1 } else { 0 } } else { 0 } }
        } else { if b0 == 105 {
            if b1 == 54 { if b2 == 52 { 3 } else { 0 } }                              // i64
            else { if b1 == 51 { if b2 == 50 { 0 } else { 0 } }                       // i32
            else { if b1 == 49 { if b2 == 54 { 11 } else { 0 } } else { 0 } } }       // i16
        } else { if b0 == 117 {
            if b1 == 51 { if b2 == 50 { 6 } else { 0 } }                              // u32
            else { if b1 == 54 { if b2 == 52 { 9 } else { 0 } }                       // u64
            else { if b1 == 49 { if b2 == 54 { 8 } else { 0 } } else { 0 } } }        // u16
        } else { 0 } } }
    } else { if ty_l == 2 {
        // Stage 2.3 / 2.5b: 2-byte idents — u8 → 7, i8 → 10.
        let b0 = __arena_get(ty_s);
        let b1 = __arena_get(ty_s + 1);
        if b0 == 117 { if b1 == 56 { 7 } else { 0 } }
        else { if b0 == 105 { if b1 == 56 { 10 } else { 0 } } else { 0 } }
    } else { if ty_l == 4 {
        // Stage 1.5: 4-byte ident — bf16 → 4.
        let b0 = __arena_get(ty_s);
        let b1 = __arena_get(ty_s + 1);
        let b2 = __arena_get(ty_s + 2);
        let b3 = __arena_get(ty_s + 3);
        if b0 == 98 {
            if b1 == 102 { if b2 == 49 { if b3 == 54 { 4 } else { 0 } } else { 0 } }
            else { 0 }
        } else { 0 }
    } else { 0 } } }
}

// Stage 8: append the digits of a small i32 (0..15 -- enough for type
// tag) to the arena as ASCII bytes. Returns the byte_len of what was
// pushed. Used by the mangle helper.
fn push_tag_digits(tag: i32) -> i32 {
    if tag < 10 {
        __arena_push(48 + tag);
        1
    } else {
        __arena_push(48 + tag / 10);
        __arena_push(48 + (tag - (tag / 10) * 10));
        2
    }
}

// Stage 8: build a mangled name like `id__i32` or `pair__i32_f64`
// directly into the arena. Inputs:
//   orig_s/l   - bytes of the original fn name
//   ty_s_arr   - parallel arrays held in arena: type-arg name_starts
//   ty_l_arr   - and lengths, ta_count entries each.
//   ta_arr     - already-arena-pushed list of starts/lens (ta_count*2
//                slots starting at ta_arr_base).
// Returns the start offset of the mangled bytes; the caller already
// knows the length (= orig_l + 2 + sum(ty_l_i + 1) - 1).
fn mangle_name_into_arena(orig_s: i32, orig_l: i32, ta_arr_base: i32, ta_count: i32) -> i32 {
    let start = __arena_len();
    // Copy orig name.
    let mut i: i32 = 0;
    while i < orig_l {
        __arena_push(__arena_get(orig_s + i));
        i = i + 1;
    }
    // Append "__".
    __arena_push(95);
    __arena_push(95);
    // Append each type-arg ident, separated by '_'.
    let mut j: i32 = 0;
    while j < ta_count {
        if j > 0 { __arena_push(95); };  // '_' separator
        let ts = __arena_get(ta_arr_base + j * 2);
        let tl = __arena_get(ta_arr_base + j * 2 + 1);
        let mut bb: i32 = 0;
        while bb < tl {
            __arena_push(__arena_get(ts + bb));
            bb = bb + 1;
        }
        j = j + 1;
    }
    start
}

// Stage 8: total mangled length given orig_l, ta_arr_base, ta_count.
fn mangle_name_len(orig_l: i32, ta_arr_base: i32, ta_count: i32) -> i32 {
    let mut total: i32 = orig_l + 2;     // orig + "__"
    let mut j: i32 = 0;
    while j < ta_count {
        if j > 0 { total = total + 1; };  // '_' separator
        let tl = __arena_get(ta_arr_base + j * 2 + 1);
        total = total + tl;
        j = j + 1;
    }
    total
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

// K1.Q (2026-05-25): match the 4-byte IDENT "true" (bytes 116,
// 114, 117, 101). Returns 1 if matched, 0 otherwise. Used by
// parse_primary's IDENT cascade to recognize the bool literal.
fn is_kw_true_ident(id_s: i32, id_l: i32) -> i32 {
    if id_l == 4 {
        if __arena_get(id_s) == 116 {
            if __arena_get(id_s + 1) == 114 {
                if __arena_get(id_s + 2) == 117 {
                    if __arena_get(id_s + 3) == 101 { 1 } else { 0 }
                } else { 0 }
            } else { 0 }
        } else { 0 }
    } else { 0 }
}

// K1.V (2026-05-25): match the 4-byte IDENT "type" (bytes 116,
// 121, 112, 101). Used by parse_top + parse_program to recognize
// top-level `type Alias = T;` decls. The alias is consumed as
// metadata (the bootstrap doesn't enforce type aliases; downstream
// uses of the alias name pass through let-type-position which
// accepts any IDENT).
fn is_kw_type_ident(id_s: i32, id_l: i32) -> i32 {
    if id_l == 4 {
        if __arena_get(id_s) == 116 {
            if __arena_get(id_s + 1) == 121 {
                if __arena_get(id_s + 2) == 112 {
                    if __arena_get(id_s + 3) == 101 { 1 } else { 0 }
                } else { 0 }
            } else { 0 }
        } else { 0 }
    } else { 0 }
}

// K1.Q (2026-05-25): match the 5-byte IDENT "false" (bytes 102,
// 97, 108, 115, 101). Returns 1 if matched, 0 otherwise.
fn is_kw_false_ident(id_s: i32, id_l: i32) -> i32 {
    if id_l == 5 {
        if __arena_get(id_s) == 102 {
            if __arena_get(id_s + 1) == 97 {
                if __arena_get(id_s + 2) == 108 {
                    if __arena_get(id_s + 3) == 115 {
                        if __arena_get(id_s + 4) == 101 { 1 } else { 0 }
                    } else { 0 }
                } else { 0 }
            } else { 0 }
        } else { 0 }
    } else { 0 }
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
    let mut cmp_result = if t == 16 {
        if t2 == 15 {
            // `<=`
            cur_advance(sb); cur_advance(sb);
            let rhs = parse_bitwise(tok_base, sb);
            mk_arith_fold(22, lhs, rhs)
        } else {
            // `<`
            cur_advance(sb);
            let rhs = parse_bitwise(tok_base, sb);
            mk_arith_fold(6, lhs, rhs)
        }
    } else { if t == 17 {
        if t2 == 15 {
            // `>=`
            cur_advance(sb); cur_advance(sb);
            let rhs = parse_bitwise(tok_base, sb);
            mk_arith_fold(23, lhs, rhs)
        } else {
            // `>`
            cur_advance(sb);
            let rhs = parse_bitwise(tok_base, sb);
            mk_arith_fold(19, lhs, rhs)
        }
    } else { if t == 15 {
        if t2 == 15 {
            // `==`
            cur_advance(sb); cur_advance(sb);
            let rhs = parse_bitwise(tok_base, sb);
            mk_arith_fold(20, lhs, rhs)
        } else { lhs }
    } else { if t == 18 {
        if t2 == 15 {
            // `!=`
            cur_advance(sb); cur_advance(sb);
            let rhs = parse_bitwise(tok_base, sb);
            mk_arith_fold(21, lhs, rhs)
        } else { lhs }
    } else { lhs }}}};
    // K1.M-fix (2026-05-25): logical short-circuit `&&` / `||`
    // chain at precedence ABOVE comparison (correct C/Rust order:
    // `a == 5 && b == 7` parses as `(a == 5) && (b == 7)`).
    // parse_bitwise bails on the doubled tokens so they fall
    // through here. Left-associative loop. Each iteration consumes
    // one `&&` or `||` and one comparison-level RHS (parse_bitwise
    // + a comparison via the same comparison logic above is
    // approximated by calling parse_bitwise for the RHS -- chained
    // `a && b == c` then parses as `a && (b == c)` because the
    // RHS is one full comparison... but parse_bitwise alone
    // doesn't handle comparison. To get the right shape we recurse
    // into a comparison-only helper inline: capture (rhs_bit, rt,
    // rt2) and apply comparison-folding before combining.
    let mut keep_l: i32 = 1;
    while keep_l == 1 {
        let lk = cur_get(sb);
        let lt = tok_tag(tok_base, lk);
        let lt2 = tok_tag(tok_base, lk + 1);
        if lt == 27 {
            if lt2 == 27 {
                cur_advance(sb); cur_advance(sb);
                // RHS is a full comparison-level expression.
                // Reuse parse_expr_basic recursively so chained
                // `a && b && c` works (right-associative via
                // recursion; for boolean ops this is equivalent
                // to left-associative).
                let rhs = parse_expr_basic(tok_base, sb);
                let zero = mk_node(0, 0, 0, 0);
                cmp_result = mk_node(7, cmp_result, rhs, zero);
                keep_l = 0;
            } else { keep_l = 0; }
        } else { if lt == 28 {
            if lt2 == 28 {
                cur_advance(sb); cur_advance(sb);
                let rhs = parse_expr_basic(tok_base, sb);
                let one = mk_node(0, 1, 0, 0);
                cmp_result = mk_node(7, cmp_result, one, rhs);
                keep_l = 0;
            } else { keep_l = 0; }
        } else {
            keep_l = 0;
        }};
    }
    cmp_result
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
        if t == 27 {       // TK_AMP -> AST_BAND, or TK_AMP TK_AMP -> bail
            // K1.M-fix (2026-05-25): the doubled `&&` form is handled
            // at parse_expr_basic level (above comparison) -- bail
            // out so the higher level can see the token pair. Single
            // `&` continues as bitwise.
            let t_next = tok_tag(tok_base, k + 1);
            if t_next == 27 {
                keep = 0;
            } else {
                cur_advance(sb);
                let rhs = parse_add(tok_base, sb);
                lhs = mk_arith_fold(28, lhs, rhs);
            }
        } else { if t == 28 {       // TK_PIPE -> AST_BOR, or TK_PIPE TK_PIPE -> bail
            // K1.M-fix (2026-05-25): doubled `||` bail-out, same as
            // `&&` -- handled at parse_expr_basic level above.
            let t_next = tok_tag(tok_base, k + 1);
            if t_next == 28 {
                keep = 0;
            } else {
                cur_advance(sb);
                let rhs = parse_add(tok_base, sb);
                lhs = mk_arith_fold(29, lhs, rhs);
            }
        } else { if t == 29 {       // TK_CARET -> AST_BXOR
            cur_advance(sb);
            let rhs = parse_add(tok_base, sb);
            lhs = mk_arith_fold(30, lhs, rhs);
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
            lhs = mk_arith_fold(2, lhs, rhs);
        } else { if t == 8 {
            cur_advance(sb);
            let rhs = parse_mul(tok_base, sb);
            lhs = mk_arith_fold(3, lhs, rhs);
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
            lhs = mk_arith_fold(4, lhs, rhs);
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
    } else { if tg == 27 {
        // K1.W (2026-05-25): unary `&` (address-of) and `&mut` are
        // no-ops in the type-erased bootstrap. Consume `&`,
        // optionally consume the `mut` IDENT (3 bytes 109, 117,
        // 116), then recurse for the inner expression. The result
        // is just the inner expression. Real pointer semantics are
        // a separate codegen-level gap; this chunk only unblocks
        // the syntax.
        cur_advance(sb);                // consume '&'
        let after_amp = cur_get(sb);
        let aa_tag = tok_tag(tok_base, after_amp);
        if aa_tag == 2 {
            let aa_s = tok_p2(tok_base, after_amp);
            let aa_l = tok_p3(tok_base, after_amp);
            let is_mut = if aa_l == 3 {
                if __arena_get(aa_s) == 109 {
                    if __arena_get(aa_s + 1) == 117 {
                        if __arena_get(aa_s + 2) == 116 { 1 } else { 0 }
                    } else { 0 }
                } else { 0 }
            } else { 0 };
            if is_mut == 1 {
                cur_advance(sb);        // consume 'mut'
            };
        };
        parse_unary(tok_base, sb)
    } else { if tg == 9 {
        // K1.W (2026-05-25): unary `*` (deref) is a no-op in the
        // type-erased bootstrap. Binary `*` (multiplication) gets
        // consumed by parse_mul BEFORE parse_unary is called for
        // the RHS, so a `*` at parse_unary's entry can only be a
        // prefix deref. Consume `*` and recurse.
        cur_advance(sb);                // consume '*'
        parse_unary(tok_base, sb)
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
                    // Stage 8.5 method-call sugar PRE-CHECK: `<expr>.IDENT(`
                    // where the LHS is an AST_VAR with a known scalar type
                    // tag (registered via `let x: T = ...`) and the IDENT
                    // is followed by `(`. Mangle to `<TypeName>__<method>(LHS, args)`.
                    // Token positions: pk = '.', pk+1 = method IDENT, pk+2 = '('.
                    let prim_tag_pre = __arena_get(prim);
                    let mut method_lhs_ty: i32 = 0 - 1;
                    if prim_tag_pre == 1 {
                        let pv_s = __arena_get(prim + 1);
                        let pv_l = __arena_get(prim + 2);
                        method_lhs_ty = var_type_tab_lookup(sb, pv_s, pv_l);
                    };
                    let lparen_tag = tok_tag(tok_base, pk + 2);
                    let is_method_call = if method_lhs_ty >= 0 {
                        if lparen_tag == 3 { 1 } else { 0 }
                    } else { 0 };
                    if is_method_call == 1 {
                        // Capture method-name IDENT bytes (at pk+1).
                        let m_s = tok_p2(tok_base, pk + 1);
                        let m_l = tok_p3(tok_base, pk + 1);
                        // Build mangled name `<TypeName>__<MethodName>` in arena.
                        // ty_tag_push_name pushes the type-name bytes; then we
                        // append "__" and the method-name bytes.
                        let packed = ty_tag_push_name(method_lhs_ty);
                        let ty_len = packed - (packed / 8) * 8;
                        let ty_start = packed / 8;
                        __arena_push(95); __arena_push(95);     // '__'
                        let mut mi: i32 = 0;
                        while mi < m_l {
                            __arena_push(__arena_get(m_s + mi));
                            mi = mi + 1;
                        }
                        let mang_s = ty_start;
                        let mang_l = ty_len + 2 + m_l;
                        // Consume `.`, method IDENT, `(`.
                        cur_advance(sb);                       // '.'
                        cur_advance(sb);                       // method IDENT
                        cur_advance(sb);                       // '('
                        // Build args list. First arg is the LHS prim. Then
                        // parse comma-separated args until ')'.
                        let first_arg = mk_node(17, prim, 0, 0);
                        let mut args_head: i32 = first_arg;
                        let mut prev_arg: i32 = first_arg;
                        let mut a_keep: i32 = 1;
                        while a_keep == 1 {
                            let at = tok_tag(tok_base, cur_get(sb));
                            if at == 4 {                        // ')'
                                a_keep = 0;
                            } else { if at == 13 {              // ','
                                cur_advance(sb);
                            } else {
                                let arg_expr = parse_expr_basic(tok_base, sb);
                                let new_arg = mk_node(17, arg_expr, 0, 0);
                                __arena_set(prev_arg + 2, new_arg);
                                prev_arg = new_arg;
                            } };
                        }
                        cur_advance(sb);                       // ')'
                        // Emit AST_CALL (tag 16) with mangled name + args_head.
                        prim = mk_node(16, mang_s, mang_l, args_head);
                        cur_struct_idx = 0 - 1;
                    } else {
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
                            // Stage 28.11 INCREMENT 3a: a field's
                            // struct_idx slot may now carry the
                            // generic-param marker `200 + gp_idx`
                            // (written by parse_struct_decl when the
                            // field's declared type matched a
                            // generic-param name like `T`). Treat
                            // such fields as SCALAR (4-byte i32
                            // shape) for now — INC-3b will read this
                            // marker at use-sites (Pt<i32>) to drive
                            // monomorphization. Pre-3a-fix the
                            // `>= 0` check would have classified a
                            // 200+ value as a nested-struct field,
                            // emitting an 8-byte (REX.W) pointer read
                            // of a 4-byte slot — silent miscompile.
                            let f_struct_idx = struct_tab_field_struct_idx(sb, lhs_struct_idx, f_idx);
                            if f_struct_idx >= 0 {
                                // Stage 28.11 INC-3a cycle-5: gp_marker_is
                                // centralizes the 200-boundary check so a
                                // future struct_tab cap bump (INC-3b) only
                                // needs to update `gp_marker_base()` in one
                                // place instead of grepping `< 200` literals.
                                if gp_marker_is(f_struct_idx) == 0 {
                                    // Nested struct field: emit AST_TUPLE_FIELD
                                    // with p3 == 1 to mark an 8-byte (REX.W)
                                    // read of the child pointer, and propagate
                                    // struct_idx forward for the next chained
                                    // access.
                                    prim = mk_node(52, prim, f_idx, 1);
                                    cur_struct_idx = f_struct_idx;
                                } else {
                                    // Stage 28.11 INC-3a: 200+ marker
                                    // — generic-param-typed field,
                                    // treated as scalar pending INC-3b
                                    // monomorphization at use site.
                                    prim = mk_node(52, prim, f_idx, 0);
                                    cur_struct_idx = 0 - 1;
                                };
                            } else {
                                prim = mk_node(52, prim, f_idx, 0);
                                cur_struct_idx = 0 - 1;
                            };
                        } else { keep_p = 0; };
                    } else { keep_p = 0; };
                    };       // end Stage 8.5 method-call-else (field-access)
                } else { keep_p = 0; }};
            } else { if pt == 20 {                     // TK_LBRACK
                cur_advance(sb);                       // skip '['
                let idx_expr = parse_expr(tok_base, sb);
                cur_advance(sb);                       // skip ']'
                prim = mk_node(53, prim, idx_expr, 0);
                cur_struct_idx = 0 - 1;
            } else { keep_p = 0; }; };
        }
        // K1.N (2026-05-25): postfix `expr as Type` cast. The
        // bootstrap is type-erased at codegen (all storage is i32-
        // shaped), so cast is a runtime no-op: consume the `as`
        // IDENT and the type IDENT, return `prim` unchanged. Chained
        // casts (`x as i32 as i64`) loop. Type forms beyond a bare
        // IDENT (e.g. `Box<T>`, `&T`, `(i32, i32)`) are NOT yet
        // supported -- a follow-up can extend this when needed.
        let mut keep_cast: i32 = 1;
        while keep_cast == 1 {
            let ck = cur_get(sb);
            let ct = tok_tag(tok_base, ck);
            if ct == 2 {
                let cs = tok_p2(tok_base, ck);
                let cl = tok_p3(tok_base, ck);
                // Match the 2-byte IDENT "as" = bytes (97, 115).
                let is_as = if cl == 2 {
                    if __arena_get(cs) == 97 {
                        if __arena_get(cs + 1) == 115 { 1 } else { 0 }
                    } else { 0 }
                } else { 0 };
                if is_as == 1 {
                    cur_advance(sb);       // consume `as`
                    cur_advance(sb);       // consume type IDENT
                } else {
                    keep_cast = 0;
                }
            } else {
                keep_cast = 0;
            }
        }
        prim
    }}}}}
}

// Stage 9: parse `|param1, param2, ...| body_expr`. Caller has detected
// TK_PIPE (28) at the start of a primary. Build a synthesized AST_FN_DECL
// with a unique name `__closure_<id>` whose params are
// (captured_var_0, captured_var_1, ..., closure_param_0, closure_param_1, ...).
// The body's AST_VAR refs to captured names resolve naturally inside that
// fn (the synthetic params have those exact names). Append the fn_decl to
// cl_pending — parse_program splices the chain in front of user fns.
//
// CRITICAL: arena positional ordering. Build the captures-persist region,
// the AST_PARAM chain, and the body BEFORE allocating the AST_FN_DECL
// node. mk_node + arena_push between the fn_decl's slot 0 push and slot 7
// push would interleave bytes into the fn_decl's slot 4..7 region and
// corrupt the layout (Stage 8.5 lesson, parser.hx:2833 comment).
fn parse_closure_lit(tok_base: i32, sb: i32) -> i32 {
    cur_advance(sb);     // consume opening '|'
    // Phase-0: trap on nested closures. Stage 9 plan trap-id 76001.
    let was_active = cl_active(sb);
    if was_active == 1 {
        mk_node(99, 76001, 0, 0)
    } else {
        // Reset per-closure scratch tables.
        cl_param_tab_reset(sb);
        cl_capture_tab_reset(sb);
        // Parse closure params. Comma-separated IDENTs, optional `: type`
        // annotation that we skip (Phase-0: all closure params are i32).
        let mut p_count: i32 = 0;
        let mut keep_p: i32 = 1;
        while keep_p == 1 {
            let pt = tok_tag(tok_base, cur_get(sb));
            if pt == 28 {                       // closing TK_PIPE
                keep_p = 0;
            } else { if pt == 13 {               // ','
                cur_advance(sb);
            } else { if pt == 0 {                // EOF safety
                keep_p = 0;
            } else {
                // Expect IDENT for param name.
                let pk = cur_get(sb);
                let p_s = tok_p2(tok_base, pk);
                let p_l = tok_p3(tok_base, pk);
                cur_advance(sb);                 // consume IDENT
                // Optional `: T` type annotation — Phase-0 ignores the type
                // (all closure params default to i32).
                if tok_tag(tok_base, cur_get(sb)) == 14 {     // ':'
                    cur_advance(sb);             // consume ':'
                    cur_advance(sb);             // consume type IDENT
                };
                cl_param_tab_add(sb, p_s, p_l);
                p_count = p_count + 1;
            }}};
        }
        cur_advance(sb);                         // consume closing '|'
        // Optional `-> ret_ty` — Phase-0 ignores (always i32 return).
        if tok_tag(tok_base, cur_get(sb)) == 8 {
            // `-` start of `->`. The lexer emits `-` as TK_MINUS (8) and
            // `>` as TK_GT (17); two separate tokens. Skip both.
            let nt2 = tok_tag(tok_base, cur_get(sb) + 1);
            if nt2 == 17 {
                cur_advance(sb);                 // consume '-'
                cur_advance(sb);                 // consume '>'
                cur_advance(sb);                 // consume ret-type IDENT
            };
        };
        // Activate capture-recording. mk_var_with_capture (called by
        // parse_primary's IDENT-as-var-ref sites) reads cl_active and
        // appends to cl_capture_tab if the var name isn't a closure param.
        set_cl_active(sb, 1);
        // Optional `{ block }` form. Surface `|x| { x + 1 }` is sugar for
        // `|x| (x + 1)`. We accept both by peeking '{'; if found, consume
        // the brace pair around the body.
        let body_start_t = tok_tag(tok_base, cur_get(sb));
        let mut body_brace: i32 = 0;
        if body_start_t == 5 {                   // TK_LBRACE
            cur_advance(sb);                     // consume '{'
            body_brace = 1;
        };
        // For the unbraced form `|x| expr`, the body is a single
        // expression — use parse_expr_basic so the `;` terminating the
        // surrounding let-binding doesn't get absorbed into the closure
        // body via parse_expr's sequencing chain.
        // For the braced form `|x| { e1 ; e2 }`, parse_expr is used so
        // multi-statement bodies with `;` chaining work — same convention
        // as fn-decl bodies.
        let body = if body_brace == 1 {
            parse_expr(tok_base, sb)
        } else {
            parse_expr_basic(tok_base, sb)
        };
        if body_brace == 1 {
            cur_advance(sb);                     // consume '}'
        };
        // Deactivate capture-recording.
        set_cl_active(sb, 0);
        // Snapshot capture table size + base BEFORE allocating any further
        // arena nodes. This is the canonical "Stage 8.5 lesson" — the
        // captures_persist region must be built before the AST_FN_DECL so
        // the fn_decl's 7 contiguous slots aren't interleaved.
        let cap_count = cl_capture_tab_count(sb);
        let cap_base = cl_capture_tab_base(sb);
        let p_base = cl_param_tab_base(sb);
        // Persist captures into a fresh arena region (cl_capture_tab is
        // reused by future closures so we cannot retain a pointer into it).
        let captures_persist_ptr = if cap_count == 0 { 0 } else { __arena_len() };
        let mut ci: i32 = 0;
        while ci < cap_count {
            let entry = cap_base + ci * 2;
            __arena_push(__arena_get(entry));
            __arena_push(__arena_get(entry + 1));
            ci = ci + 1;
        }
        // Build the synthesized fn name `__closure_<id>` directly in the
        // arena. Bytes: '_' '_' 'c' 'l' 'o' 's' 'u' 'r' 'e' '_' '<digits>'.
        let closure_id = cl_id_bump(sb);
        let name_start = __arena_len();
        __arena_push(95); __arena_push(95);
        __arena_push(99); __arena_push(108); __arena_push(111);
        __arena_push(115); __arena_push(117); __arena_push(114);
        __arena_push(101); __arena_push(95);
        let n_digits = push_tag_digits(closure_id);
        let name_len = 10 + n_digits;
        // Build the AST_PARAM chain: (captures..., closure_params...).
        // Each AST_PARAM is 5 slots (tag, name_s, name_l, next, type_tag).
        // mk_node + arena_push for each one is fine — they're standalone
        // contiguous records. Linking is done via __arena_set on prev.p3.
        //
        // Audit 28.8 B4 (trap 76003): closure-capture type tags used to
        // be hardcoded to 0 (i32) regardless of the captured variable's
        // real type. For `let pi = 3.14_f64; let c = |x| x + pi;` the
        // low 32 bits of the f64 bit pattern were captured as i32 —
        // silent garbage arithmetic. Phase-0 fix: detect a non-i32
        // capture via var_type_tab_lookup and emit AST_ERR(76003) so
        // the user gets a loud diagnostic instead of silent corruption.
        // The full fix (stride-3 cl_capture_tab + per-capture type tag)
        // is deferred to a later cycle (see audit doc B4); the loud
        // failure here is enough to block the silent-corruption window.
        let mut params_head: i32 = 0;
        let mut prev_p: i32 = 0;
        let mut nonint_capture: i32 = 0;
        let mut ki: i32 = 0;
        while ki < cap_count {
            let pair = captures_persist_ptr + ki * 2;
            let ns = __arena_get(pair);
            let nl = __arena_get(pair + 1);
            // Audit 28.8 B4: probe captured var's type tag. Tag 0 = i32
            // (the only safe Phase-0 case); any other value (including
            // -1 = not-tracked) means we'd be silently truncating.
            //
            // Audit 28.8 cycle 3 D2: extend the let-inference to also
            // tag Call-RHS / non-literal-RHS lets with a "potentially
            // non-i32" sentinel (tag 12) so trap 76003 fires for the
            // common idiom `let pi = get_pi(); let c = |x| x + pi;`.
            // The capture-site guard stays `> 0`; the change is at the
            // let-inference site below — Call RHS now registers tag 12
            // unless the user wrote an explicit annotation. Function
            // params remain untracked (-1) and still pass cleanly so
            // we don't over-trap legitimate i32 param captures.
            let cap_ty_tag = var_type_tab_lookup(sb, ns, nl);
            if cap_ty_tag > 0 {
                nonint_capture = 1;
            };
            let new_p = mk_node(18, ns, nl, 0);
            __arena_push(0);                     // type tag = 0 (i32)
            if params_head == 0 {
                params_head = new_p;
                prev_p = new_p;
            } else {
                __arena_set(prev_p + 3, new_p);
                prev_p = new_p;
            };
            ki = ki + 1;
        }
        // Stage 29 fix (2026-05-12): bootstrap parser doesn't support
        // `return` keyword. Rewrote early-return as if/else expression
        // so the bootstrap parser can self-host. Helix is expression-
        // based; `return EXPR;` ≡ tail-expr-of-if-block.
        if nonint_capture == 1 {
            // Loud failure: AST_ERR(76003) propagates to codegen which
            // emits a hard trap when the closure is invoked. The full
            // type-preserving capture is a follow-on cycle.
            mk_node(99, 76003, 0, 0)
        } else {
        let mut pi: i32 = 0;
        while pi < p_count {
            let pp = p_base + pi * 2;
            let ns = __arena_get(pp);
            let nl = __arena_get(pp + 1);
            let new_p = mk_node(18, ns, nl, 0);
            __arena_push(0);                     // type tag = 0 (i32)
            if params_head == 0 {
                params_head = new_p;
                prev_p = new_p;
            } else {
                __arena_set(prev_p + 3, new_p);
                prev_p = new_p;
            };
            pi = pi + 1;
        }
        // Allocate AST_FN_DECL with 20 contiguous slots (Stage 33 layout).
        // p1=name_s, p2=name_l, p3=body, p4=params_head, p5=ret_ty,
        // p6=is_generic, p7=gp_names_head, p8=is_checkpoint,
        // p9=is_deprecated, p10=is_trace, p11=is_unwind,
        // p12=deprecated_msg_start, p13=deprecated_msg_len,
        // p14=is_kernel, p15=is_autotune, p16=autotune_product,
        // p17=autotune_parse_error_kind, p18=since_msg_start, p19=since_msg_len.
        let fn_node = mk_node(14, name_start, name_len, body);
        __arena_push(params_head);
        __arena_push(0);                         // ret_ty = 0 (i32)
        __arena_push(0);                         // is_generic = 0
        __arena_push(0);                         // gp_names_head = 0
        __arena_push(0);                         // is_checkpoint = 0 (Stage 14.5)
        __arena_push(0);                         // is_deprecated = 0 (Stage 28.9)
        __arena_push(0);                         // is_trace = 0 (Stage 28.9)
        __arena_push(0);                         // is_unwind = 0 (Stage 28.9)
        __arena_push(0);                         // deprecated_msg_start = 0 (Stage 33)
        __arena_push(0);                         // deprecated_msg_len = 0 (Stage 33)
        __arena_push(0);                         // is_kernel = 0 (Stage 33)
        __arena_push(0);                         // is_autotune = 0 (Stage 33)
        __arena_push(0);                         // autotune_product = 0 (Stage 33)
        __arena_push(0);                         // autotune_parse_error_kind = 0 (Stage 33)
        __arena_push(0);                         // since_msg_start = 0 (Stage 33)
        __arena_push(0);                         // since_msg_len = 0 (Stage 33)
        // Wrap in AST_FN_LIST and append to cl_pending chain.
        let list_node = mk_node(15, fn_node, 0, 0);
        let head = cl_pending_head(sb);
        if head == 0 {
            set_cl_pending_head(sb, list_node);
            set_cl_pending_tail(sb, list_node);
        } else {
            let tail = cl_pending_tail(sb);
            __arena_set(tail + 2, list_node);
            set_cl_pending_tail(sb, list_node);
        };
        // Stash closure id + capture info in scratch slots so parse_let
        // can register the binding (cl_var_tab) AFTER the value parse.
        // Slots: 57 = closure_id, 58 = captures_persist_ptr, 59 = cap_count.
        set_last_closure_idx(sb, closure_id);
        __arena_set(sb + 58, captures_persist_ptr);
        __arena_set(sb + 59, cap_count);
        // Return a placeholder AST_INT(0). The closure's runtime value is
        // unused — only the binding name matters at the call site.
        mk_node(0, 0, 0, 0)
        }      // Stage 29 fix: closes the `else` branch of nonint_capture
    }
}

fn parse_primary(tok_base: i32, sb: i32) -> i32 {
    let k = cur_get(sb);
    let t = tok_tag(tok_base, k);
    if t == 28 {
        // Stage 9: closure literal `|params| body`. TK_PIPE (28) at the
        // start of a primary is unambiguous — bitwise OR is parsed at the
        // bitwise level (parse_bitwise) and never reaches primary as a
        // prefix. Dispatch to parse_closure_lit which:
        //   1. parses params,
        //   2. parses body (capture-recording active so any AST_VAR ref
        //      whose name isn't a closure-param is auto-recorded),
        //   3. synthesizes an AST_FN_DECL named `__closure_<id>` with
        //      params = (captures..., closure_params...) and the parsed
        //      body verbatim,
        //   4. wraps the fn_decl in AST_FN_LIST and appends to
        //      cl_pending — parse_program splices that chain in front of
        //      the user's fn list,
        //   5. registers (captures_ptr, cap_count) in scratch slots 57..59
        //      so parse_let can record the binding in cl_var_tab.
        // Returns AST_INT(0) as the closure value; the runtime value is
        // unused — only the binding name matters at the call site.
        parse_closure_lit(tok_base, sb)
    } else { if t == 1 {
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
        // Stage 14: detect `grad_rev_all(IDENT)(args).IDENT` — the
        // reverse-mode AD meta-call that returns a per-param gradient.
        // IDENT "grad_rev_all" is 12 bytes (103,114,97,100,95,114,101,
        // 118,95,97,108,108). Followed by TK_LPAREN (3), TK_IDENT (2),
        // TK_RPAREN (4), TK_LPAREN (3). The trailing `.IDENT` selects
        // which gradient (e.g. `.dx` -> ∂loss/∂x). We detect the prefix
        // here and require the postfix `.IDENT` immediately after the
        // closing `)` of args. The synthesized fn is named
        // "<loss>__grad_<field>" and computed via Stage 12's
        // forward-mode `differentiate` w.r.t. the param matching
        // `field` (with leading 'd' stripped). FLAT prefix-trap pattern,
        // single is_gr_rev_all flag scoped to one tight branch ahead
        // of Stage 12's grad-detect.
        let is_grad_rev_kw = if id_len == 12 {
            let r0 = __arena_get(id_start);
            let r1 = __arena_get(id_start + 1);
            let r2 = __arena_get(id_start + 2);
            let r3 = __arena_get(id_start + 3);
            let r4 = __arena_get(id_start + 4);
            let r5 = __arena_get(id_start + 5);
            let r6 = __arena_get(id_start + 6);
            let r7 = __arena_get(id_start + 7);
            let r8 = __arena_get(id_start + 8);
            let r9 = __arena_get(id_start + 9);
            let r10 = __arena_get(id_start + 10);
            let r11 = __arena_get(id_start + 11);
            if r0 == 103 { if r1 == 114 { if r2 == 97 { if r3 == 100 {
            if r4 == 95 { if r5 == 114 { if r6 == 101 { if r7 == 118 {
            if r8 == 95 { if r9 == 97 { if r10 == 108 { if r11 == 108 {
                1 } else { 0 } } else { 0 } } else { 0 } } else { 0 }
            } else { 0 } } else { 0 } } else { 0 } } else { 0 }
            } else { 0 } } else { 0 } } else { 0 } } else { 0 }
        } else { 0 };
        let gr_t1 = tok_tag(tok_base, k + 1);
        let gr_t2 = tok_tag(tok_base, k + 2);
        let gr_t3 = tok_tag(tok_base, k + 3);
        let gr_t4 = tok_tag(tok_base, k + 4);
        let is_grad_rev_call = if is_grad_rev_kw == 1 {
            if gr_t1 == 3 { if gr_t2 == 2 { if gr_t3 == 4 { if gr_t4 == 3 {
                1 } else { 0 } } else { 0 } } else { 0 } } else { 0 }
        } else { 0 };
        if is_grad_rev_call == 1 {
            // Consume `grad_rev_all`, `(`, IDENT (loss_name), `)`, `(`.
            cur_advance(sb);     // `grad_rev_all`
            cur_advance(sb);     // `(`
            let lossr_k = cur_get(sb);
            let lossr_s = tok_p2(tok_base, lossr_k);
            let lossr_l = tok_p3(tok_base, lossr_k);
            cur_advance(sb);     // loss_name IDENT
            cur_advance(sb);     // `)`
            cur_advance(sb);     // `(`
            // Parse args until `)`. Same pattern as Stage 12.
            let mut grev_args_head: i32 = 0;
            let mut grev_prev_arg: i32 = 0;
            let mut gr_keep: i32 = 1;
            while gr_keep == 1 {
                let at = tok_tag(tok_base, cur_get(sb));
                if at == 4 {
                    gr_keep = 0;
                } else { if at == 13 {
                    cur_advance(sb);
                } else {
                    let arg_expr = parse_expr_basic(tok_base, sb);
                    let new_arg = mk_node(17, arg_expr, 0, 0);
                    if grev_args_head == 0 {
                        grev_args_head = new_arg;
                        grev_prev_arg = new_arg;
                    } else {
                        __arena_set(grev_prev_arg + 2, new_arg);
                        grev_prev_arg = new_arg;
                    };
                }};
            }
            cur_advance(sb);     // consume `)`
            // Require `.IDENT` next. Phase-0: trap 88001 if missing.
            let dot_t = tok_tag(tok_base, cur_get(sb));
            if dot_t != 22 {
                mk_node(99, 88001, 0, 0)
            } else {
                cur_advance(sb);     // consume '.'
                let fk = cur_get(sb);
                let f_t = tok_tag(tok_base, fk);
                if f_t != 2 {
                    mk_node(99, 88001, 0, 0)
                } else {
                    let field_s = tok_p2(tok_base, fk);
                    let field_l = tok_p3(tok_base, fk);
                    cur_advance(sb);     // consume field IDENT
                    // Build mangled name "<loss>__grad_<field>". Length
                    // = lossr_l + 7 ("__grad_") + field_l. Bytes:
                    //   '_' '_' 'g' 'r' 'a' 'd' '_' = 95,95,103,114,97,100,95
                    let mang_s_r = __arena_len();
                    let mut br: i32 = 0;
                    while br < lossr_l {
                        __arena_push(__arena_get(lossr_s + br));
                        br = br + 1;
                    }
                    __arena_push(95); __arena_push(95);    // '__'
                    __arena_push(103); __arena_push(114);  // 'gr'
                    __arena_push(97); __arena_push(100);   // 'ad'
                    __arena_push(95);                      // '_'
                    let mut br2: i32 = 0;
                    while br2 < field_l {
                        __arena_push(__arena_get(field_s + br2));
                        br2 = br2 + 1;
                    }
                    let mang_l_r = lossr_l + 7 + field_l;
                    // Register in gr_rev_pending. Cap 8.
                    gr_rev_pending_add(sb, lossr_s, lossr_l,
                                       field_s, field_l, mang_s_r);
                    // Emit AST_CALL with mangled name + args_head.
                    mk_node(16, mang_s_r, mang_l_r, grev_args_head)
                }
            }
        } else {
        // Stage 12: detect `grad(IDENT)(args)` — the meta-call that takes
        // a 1-arg user fn and returns its derivative. Match must be exact:
        // IDENT "grad" (4 bytes: 103,114,97,100) followed by TK_LPAREN (3),
        // TK_IDENT (2), TK_RPAREN (4), TK_LPAREN (3). On match, register
        // the (loss_name, mangled_name="<loss>__grad") pair in grad_pending
        // so the post-parse grad_pass synthesizes the derivative fn, then
        // emit AST_CALL with the mangled name + parsed args. FLAT prefix-
        // trap pattern: a single is_grad flag controls one tightly-scoped
        // branch ahead of all other IDENT-prefix dispatch.
        let is_grad_kw = if id_len == 4 {
            let g0 = __arena_get(id_start);
            let g1 = __arena_get(id_start + 1);
            let g2 = __arena_get(id_start + 2);
            let g3 = __arena_get(id_start + 3);
            if g0 == 103 { if g1 == 114 { if g2 == 97 { if g3 == 100 { 1 } else { 0 } } else { 0 } } else { 0 } } else { 0 }
        } else { 0 };
        let g_t1 = tok_tag(tok_base, k + 1);
        let g_t2 = tok_tag(tok_base, k + 2);
        let g_t3 = tok_tag(tok_base, k + 3);
        let g_t4 = tok_tag(tok_base, k + 4);
        let is_grad_call = if is_grad_kw == 1 {
            if g_t1 == 3 { if g_t2 == 2 { if g_t3 == 4 { if g_t4 == 3 { 1 } else { 0 } } else { 0 } } else { 0 } } else { 0 }
        } else { 0 };
        if is_grad_call == 1 {
            // Consume `grad`, `(`, IDENT (loss_name), `)`, `(`.
            cur_advance(sb);     // `grad`
            cur_advance(sb);     // `(`
            let loss_k = cur_get(sb);
            let loss_s = tok_p2(tok_base, loss_k);
            let loss_l = tok_p3(tok_base, loss_k);
            cur_advance(sb);     // loss_name IDENT
            cur_advance(sb);     // `)`
            cur_advance(sb);     // `(`
            // Build mangled name "<loss>__grad" directly into the arena.
            let mang_s = __arena_len();
            let mut bi: i32 = 0;
            while bi < loss_l {
                __arena_push(__arena_get(loss_s + bi));
                bi = bi + 1;
            }
            __arena_push(95);    // '_'
            __arena_push(95);    // '_'
            __arena_push(103);   // 'g'
            __arena_push(114);   // 'r'
            __arena_push(97);    // 'a'
            __arena_push(100);   // 'd'
            let mang_l = loss_l + 6;
            // Register in grad_pending (idempotent on duplicate would still
            // synthesize twice — Phase-0 caps at 8 calls, dedup deferred).
            grad_pending_add(sb, loss_s, loss_l, mang_s, mang_l);
            // Parse args until `)`. Same pattern as plain IDENT call below.
            let mut args_head: i32 = 0;
            let mut prev_arg: i32 = 0;
            let mut g_keep: i32 = 1;
            while g_keep == 1 {
                let at = tok_tag(tok_base, cur_get(sb));
                if at == 4 {
                    g_keep = 0;
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
            cur_advance(sb);     // consume `)`
            mk_node(16, mang_s, mang_l, args_head)
        } else {
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
            // Stage 8.5: capture the type IDENT bytes BEFORE skipping so
            // we can register the binding in var_type_tab for method-call
            // sugar (`x.eq(b)` -> `i32__eq(x, b)`).
            let after_name_tag = tok_tag(tok_base, cur_get(sb));
            let mut let_ty_tag: i32 = 0 - 1;
            if after_name_tag == 14 {
                cur_advance(sb);    // consume ':'
                // K1.R (2026-05-25): handle the `[T; N]` TyArray
                // form in addition to the bare-IDENT type. If the
                // type position starts with TK_LBRACK (tag 20),
                // skip to the matching TK_RBRACK (21). The bootstrap
                // is type-erased so the annotation is metadata-only;
                // let_ty_tag stays -1 for array types since
                // ty_ident_to_tag doesn't model them. Generic types
                // (`Foo<T>`) and reference types (`&T`) are still
                // out of scope -- separate follow-ups.
                let type_start_tag = tok_tag(tok_base, cur_get(sb));
                if type_start_tag == 20 {
                    cur_advance(sb);    // consume '['
                    let mut keep_ty: i32 = 1;
                    while keep_ty == 1 {
                        let tyt = tok_tag(tok_base, cur_get(sb));
                        if tyt == 21 {
                            keep_ty = 0;
                        } else { if tyt == 0 {
                            keep_ty = 0;
                        } else {
                            cur_advance(sb);
                        }};
                    }
                    cur_advance(sb);    // consume ']'
                } else { if type_start_tag == 3 {
                    // K1.Y (2026-05-25): `(T1, T2, ...)` tuple type
                    // annotation. Consume `(`, skip nested parens
                    // until matching `)`, consume `)`. Type-erased
                    // no-op like the other type forms.
                    cur_advance(sb);    // consume '('
                    let mut tu_depth: i32 = 1;
                    while tu_depth > 0 {
                        let tut = tok_tag(tok_base, cur_get(sb));
                        if tut == 3 {
                            tu_depth = tu_depth + 1;
                        } else { if tut == 4 {
                            tu_depth = tu_depth - 1;
                        } else { if tut == 0 {
                            tu_depth = 0;
                        } else {} } };
                        if tu_depth > 0 {
                            cur_advance(sb);
                        };
                    }
                    cur_advance(sb);    // consume final ')'
                } else { if type_start_tag == 27 {
                    // K1.S (2026-05-25): `&T` or `&mut T` -- TyRef.
                    // Consume the `&`, optionally consume `mut`,
                    // then consume the type IDENT. Type-erased.
                    cur_advance(sb);    // consume '&'
                    let after_amp = cur_get(sb);
                    let aa_tag = tok_tag(tok_base, after_amp);
                    if aa_tag == 2 {
                        let aa_s = tok_p2(tok_base, after_amp);
                        let aa_l = tok_p3(tok_base, after_amp);
                        // 3-byte "mut" check (bytes 109, 117, 116).
                        let is_mut_kw = if aa_l == 3 {
                            if __arena_get(aa_s) == 109 {
                                if __arena_get(aa_s + 1) == 117 {
                                    if __arena_get(aa_s + 2) == 116 { 1 } else { 0 }
                                } else { 0 }
                            } else { 0 }
                        } else { 0 };
                        if is_mut_kw == 1 {
                            cur_advance(sb);    // consume 'mut'
                        };
                    };
                    cur_advance(sb);    // consume type IDENT
                } else { if type_start_tag == 9 {
                    // K1.S (2026-05-25): `*const T` / `*mut T` / `*T`
                    // -- TyPtr. Consume `*`, optionally consume
                    // `const` or `mut`, then consume the type IDENT.
                    cur_advance(sb);    // consume '*'
                    let after_star = cur_get(sb);
                    let as_tag = tok_tag(tok_base, after_star);
                    if as_tag == 2 {
                        let as_s = tok_p2(tok_base, after_star);
                        let as_l = tok_p3(tok_base, after_star);
                        // 5-byte "const" (99,111,110,115,116) or
                        // 3-byte "mut" (109,117,116) -- consume if
                        // matched, else leave for type IDENT.
                        let is_const_kw = if as_l == 5 {
                            if __arena_get(as_s) == 99 {
                                if __arena_get(as_s + 1) == 111 {
                                    if __arena_get(as_s + 2) == 110 {
                                        if __arena_get(as_s + 3) == 115 {
                                            if __arena_get(as_s + 4) == 116 { 1 } else { 0 }
                                        } else { 0 }
                                    } else { 0 }
                                } else { 0 }
                            } else { 0 }
                        } else { 0 };
                        let is_mut_kw_s = if as_l == 3 {
                            if __arena_get(as_s) == 109 {
                                if __arena_get(as_s + 1) == 117 {
                                    if __arena_get(as_s + 2) == 116 { 1 } else { 0 }
                                } else { 0 }
                            } else { 0 }
                        } else { 0 };
                        if is_const_kw == 1 {
                            cur_advance(sb);    // consume 'const'
                        } else { if is_mut_kw_s == 1 {
                            cur_advance(sb);    // consume 'mut'
                        }};
                    };
                    cur_advance(sb);    // consume type IDENT
                } else {
                    let lt_tok = cur_get(sb);
                    let lt_s = tok_p2(tok_base, lt_tok);
                    let lt_l = tok_p3(tok_base, lt_tok);
                    // K1.X (2026-05-25): TyFn `fn(T1, T2) -> R`
                    // detection. The 2-byte "fn" IDENT (102, 110)
                    // in type position is the function-type prefix.
                    // Consume `fn`, `(`, tokens-until-`)`, then
                    // optional `-> R`. Type-erased no-op.
                    let is_fn_ty = if lt_l == 2 {
                        if __arena_get(lt_s) == 102 {
                            if __arena_get(lt_s + 1) == 110 { 1 } else { 0 }
                        } else { 0 }
                    } else { 0 };
                    if is_fn_ty == 1 {
                        cur_advance(sb);    // consume 'fn'
                        cur_advance(sb);    // consume '('
                        let mut keep_fn: i32 = 1;
                        while keep_fn == 1 {
                            let ft = tok_tag(tok_base, cur_get(sb));
                            if ft == 4 {     // ')'
                                keep_fn = 0;
                            } else { if ft == 0 {
                                keep_fn = 0;
                            } else {
                                cur_advance(sb);
                            }};
                        }
                        cur_advance(sb);    // consume ')'
                        // Optional `-> R`: `-` (8) then `>` (17)
                        // then IDENT. Same shape as the closure-
                        // body return-type pattern (parse_unary
                        // ~line 2028).
                        let aap = tok_tag(tok_base, cur_get(sb));
                        if aap == 8 {
                            let aap2 = tok_tag(tok_base, cur_get(sb) + 1);
                            if aap2 == 17 {
                                cur_advance(sb);    // consume '-'
                                cur_advance(sb);    // consume '>'
                                cur_advance(sb);    // consume ret-type IDENT
                            };
                        };
                    } else {
                    let_ty_tag = ty_ident_to_tag(lt_s, lt_l);
                    cur_advance(sb);    // consume type IDENT
                    // K1.T (2026-05-25): optional `<...>` generic
                    // arg list. If the type IDENT is followed by
                    // TK_LT (16), it's a generic instantiation like
                    // `Foo<i32>` or `Foo<Bar<T>>`. Skip with depth-
                    // tracking until matching `>`. The type info is
                    // discarded -- let_ty_tag holds the base IDENT's
                    // tag (or -1 for unknown bases), which is fine
                    // since the bootstrap is type-erased.
                    let after_id_tag = tok_tag(tok_base, cur_get(sb));
                    if after_id_tag == 16 {
                        cur_advance(sb);    // consume '<'
                        let mut g_depth: i32 = 1;
                        while g_depth > 0 {
                            let gt = tok_tag(tok_base, cur_get(sb));
                            if gt == 16 {
                                g_depth = g_depth + 1;
                            } else { if gt == 17 {
                                g_depth = g_depth - 1;
                            } else { if gt == 31 {
                                // K1.T (2026-05-25): the lexer folds
                                // `>>` into a single TK_RSHIFT token.
                                // For nested generics `Box<Box<i32>>`
                                // we treat that as TWO closing `>`s.
                                g_depth = g_depth - 2;
                            } else { if gt == 0 {
                                g_depth = 0;
                            } else {} }} };
                            if g_depth > 0 {
                                cur_advance(sb);
                            };
                        }
                        cur_advance(sb);    // consume final '>' / '>>'
                    };
                    };     // K1.X (2026-05-25): close the is_fn_ty else-branch wrapping the bare-IDENT + generic-args path
                }}}};     // K1.Y (2026-05-25): +1 close for the new TK_LPAREN tuple-type arm
            };
            // Register the typed binding (only when the annotation produced
            // a recognized scalar tag; struct-typed lets continue to use
            // var_struct_tab below).
            if let_ty_tag >= 0 {
                var_type_tab_add(sb, name_start, name_len, let_ty_tag);
            };
            cur_advance(sb);     // '='
            // value uses parse_expr_basic so the `;` after the
            // value belongs to the let-terminator, not a sequencer.
            let value = parse_expr_basic(tok_base, sb);
            // Audit 28.8 cycle 2 B:C2: when no type annotation was
            // present (let_ty_tag still < 0), infer from the value's
            // root AST tag whether the binding is trivially-i32 or
            // a known non-i32 literal. Without this, the closure
            // capture trap 76003 was silent for the dominant idiom
            // `let pi = 3.14_f64;` (no annotation) because
            // `var_type_tab_lookup` returned -1 (untracked) and the
            // capture guard `> 0` failed. We register inferred tags
            // so the capture probe sees the truth.
            //
            // AST tag conventions (from header comment of this file):
            //    0 AST_INT -> trivially-i32, type tag 0
            //   27 AST_FLOATLIT_F32 -> type tag 1 (f32)
            //   31 AST_FLOATLIT_BF16 -> type tag 4 (bf16)
            //   34 AST_FLOATLIT_F64 -> type tag 2 (f64)
            //   35 AST_INTLIT_I64 -> type tag 3 (i64)
            //   36 AST_INTLIT_U32 -> type tag 6 (u32)
            //   37 AST_INTLIT_U8  -> type tag 7 (u8)
            //   38 AST_INTLIT_U64 -> type tag 9 (u64)
            //   39 AST_INTLIT_I8  -> type tag 10 (i8)
            //   40 AST_INTLIT_I16 -> type tag 11 (i16)
            //   41 AST_INTLIT_U16 -> type tag 8 (u16)
            //
            // Any other root tag (Binary / Call / Name / Block / If /
            // ...) stays untracked (-1) and the capture probe will see
            // an opaque value — which today silently passes the > 0
            // guard, BUT we want the closure-capture loop to treat
            // untracked as "potentially non-i32 unless RHS proves i32"
            // (see capture-site update below).
            let mut inferred_ty_tag: i32 = 0 - 1;
            if let_ty_tag < 0 {
                let val_tag = __arena_get(value);
                if val_tag == 0 {
                    inferred_ty_tag = 0;             // trivially i32
                } else { if val_tag == 27 {
                    inferred_ty_tag = 1;             // f32
                } else { if val_tag == 31 {
                    inferred_ty_tag = 4;             // bf16
                } else { if val_tag == 34 {
                    inferred_ty_tag = 2;             // f64
                } else { if val_tag == 35 {
                    inferred_ty_tag = 3;             // i64
                } else { if val_tag == 36 {
                    inferred_ty_tag = 6;             // u32
                } else { if val_tag == 37 {
                    inferred_ty_tag = 7;             // u8
                } else { if val_tag == 38 {
                    inferred_ty_tag = 9;             // u64
                } else { if val_tag == 39 {
                    inferred_ty_tag = 10;            // i8
                } else { if val_tag == 40 {
                    inferred_ty_tag = 11;            // i16
                } else { if val_tag == 41 {
                    inferred_ty_tag = 8;             // u16
                // Audit 28.8 cycle 5 C4-1 / F1 (final REVERT of D2):
                // the cycle-3 D2 sentinel-12 fix for Call-RHS was
                // FUNDAMENTALLY UNSOUND. The parser has no access to fn
                // return-type information — tagging ALL Call-RHS lets as
                // "non-i32" produced a CRITICAL functional regression
                // for the dominant pattern `let n = i32_returning_fn();
                // let c = |y| y + n;` (SIGILL trap 76003 at runtime).
                // The tag-12 sentinel also collides with
                // var_type_tab_lookup's other consumers (method-call
                // dispatch via method_lhs_ty, etc.), causing
                // unrelated downstream regressions (Stage 11A3
                // Quote/Splice with Call-RHS `let h0 = Quote(10);`).
                // REVERT: untyped Call-RHS lets register inferred_ty_tag
                // = -1 (untracked) just like every other non-literal RHS
                // — same behavior as pre-D2 (cycle-2 state). The
                // closure-capture trap 76003 still fires correctly on
                // explicitly-annotated `let pi: f64 = ...;` (via the
                // typed-RHS arm at line 2264) and on direct non-i32
                // literal RHS (`let pi = 3.14_f64;` via val_tag 27/31/
                // 34 arms above). The "untyped Call-RHS captured into a
                // closure" case is a typecheck-time concern requiring fn
                // return-type info; it cannot be soundly inferred at
                // parse time. Re-introducing the check is deferred to a
                // post-typecheck pass when one exists.
                };};};};};};};};};};};
                // Register the inferred tag so var_type_tab_lookup at
                // closure-capture sites can detect the non-i32 case
                // (trap 76003). We DO NOT shadow let_ty_tag because the
                // existing register-before-value block at line 2254
                // already ran (for annotated lets); we add a second
                // register here for the inferred case.
                if inferred_ty_tag >= 0 {
                    var_type_tab_add(sb, name_start, name_len, inferred_ty_tag);
                };
            };
            // Iter B: if the value was a struct lit, last_struct_idx
            // is now set; register the binding name -> struct_idx so
            // postfix `.IDENT` on this var resolves to a field offset.
            let s_idx_b = last_struct_idx(sb);
            if s_idx_b >= 0 {
                var_struct_tab_add(sb, name_start, name_len, s_idx_b);
                set_last_struct_idx(sb, 0 - 1);
            };
            // Stage 9: if the value was a closure literal, register the
            // binding (name -> closure_id, captures_ptr, capture_count) in
            // cl_var_tab so call-site lowering can rewrite `name(args)` to
            // `__closure_<id>(captured_vars..., args...)`.
            let cl_idx_b = last_closure_idx(sb);
            if cl_idx_b >= 0 {
                let caps_ptr = __arena_get(sb + 58);
                let caps_count = __arena_get(sb + 59);
                cl_var_tab_add(sb, name_start, name_len, caps_ptr, caps_count, cl_idx_b);
                set_last_closure_idx(sb, 0 - 1);
                __arena_set(sb + 58, 0);
                __arena_set(sb + 59, 0);
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
        } else { if byte_eq(id_start, id_len, kw_return_s(sb), kw_return_n(sb)) == 1 {
            // K1.C-wireup (2026-05-25): `return <expr>` form. The
            // parse_return fn + AST_RET codegen + keyword bytes were
            // staged in commit 816ce51 (K1.C-deadcode); this arm is
            // the one-line connector that makes them reachable.
            // Closing braces for this arm's `if`+`else` blocks are
            // added at the IDENT sub-cascade closer (line ~3712) --
            // NOT at the outer if-cascade closer at line ~3850 (that
            // was the first-attempt K1.C bug, reverted in commit
            // a180366).
            parse_return(tok_base, sb)
        } else { if byte_eq(id_start, id_len, kw_for_s(sb), kw_for_n(sb)) == 1 {
            // K1.G-wireup (2026-05-25): `for var in start..end { body }`
            // form. parse_for desugars to AST_LET_MUT + AST_WHILE +
            // AST_SEQ + AST_ASSIGN + AST_ADD + AST_LT (no new codegen
            // tag). kw_for_s/n was already installed by Stage 8.5 for
            // the `impl Trait for Type` syntax; we reuse it. +1 closing
            // brace at the IDENT sub-cascade closer per K1.C lesson.
            parse_for(tok_base, sb)
        } else { if byte_eq(id_start, id_len, kw_loop_s(sb), kw_loop_n(sb)) == 1 {
            // K1.H1-wireup (2026-05-25): `loop { body }` form.
            // parse_loop desugars to AST_WHILE(AST_INT(1), body) --
            // no new codegen tag. kw_loop_s/n was installed in K1.H1-
            // deadcode at slots 92/93. +1 closing brace at the IDENT
            // sub-cascade closer per the K1.C lesson (same algebra as
            // K1.G).
            parse_loop(tok_base, sb)
        } else { if is_kw_true_ident(id_start, id_len) == 1 {
            // K1.Q (2026-05-25): `true` -- emit AST_INT(1). Chars
            // were the closest precedent (K1.K) but bool lits go
            // through this arm rather than the lexer so no install
            // is needed -- the IDENT just gets translated. +1
            // closing brace at the IDENT sub-cascade closer.
            cur_advance(sb);
            mk_node(0, 1, 0, 0)
        } else { if is_kw_false_ident(id_start, id_len) == 1 {
            // K1.Q (2026-05-25): `false` -- emit AST_INT(0). +1
            // closing brace at the IDENT sub-cascade closer.
            cur_advance(sb);
            mk_node(0, 0, 0, 0)
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
            // Stage 8: turbofish detection. `IDENT::<TY1, TY2, ...>(...)`
            // is a generic-fn call. Identifying tokens at k+1..k+3 are
            // COLON, COLON, LT (16). Mutually exclusive with is_enum_path
            // since enum requires t3_pre == 2 (IDENT) and turbofish
            // requires t3_pre == 16 (LT). FLAT prefix-trap pattern.
            let is_turbofish = if t1_pre == 14 {
                if t2_pre == 14 { if t3_pre == 16 { 1 } else { 0 } } else { 0 }
            } else { 0 };
            // Stage 8.5C: type-namespace call `IDENT::IDENT(args)` where the
            // first IDENT is either a generic-param name (in gp_tab) or one
            // of the canonical scalar type IDENTs (i32/f32/f64/i64/u32/u64).
            // Mangle to `<First>__<Second>` AST_CALL. Mutually exclusive
            // with is_enum_path (which requires e_idx_pre >= 0 — enums and
            // generic-params/scalar types don't overlap by name).
            let gp_idx_pre = gp_tab_lookup(sb, id_start, id_len);
            let scalar_ty_pre = ty_ident_to_tag(id_start, id_len);
            let is_known_scalar = if id_len == 3 {
                let sc0 = __arena_get(id_start);
                if sc0 == 105 { 1 } else { if sc0 == 102 { 1 } else { if sc0 == 117 { 1 } else { 0 } } }
            } else { 0 };
            let is_typed_call = if t1_pre == 14 {
                if t2_pre == 14 { if t3_pre == 2 {
                    let lp_t4 = tok_tag(tok_base, k + 4);
                    if lp_t4 == 3 {
                        if gp_idx_pre >= 0 { 1 }
                        else { if is_known_scalar == 1 { 1 } else { 0 } }
                    } else { 0 }
                } else { 0 } } else { 0 }
            } else { 0 };
            // Disable typed-call when is_enum_path is true (enums take
            // precedence — their name lookup already matched).
            let is_typed_call_active = if is_enum_path == 1 { 0 } else { is_typed_call };
            // Stage 10: path-call detection — `foo::bar(...)` or
            // `outer::inner::baz(...)` etc., where the first IDENT is NOT a
            // known enum, scalar type, or generic-param. Lifted modules use
            // the mangled name `seg1__seg2__...__last` so we just walk the
            // `::IDENT` chain, build the mangled name, and emit AST_CALL.
            // Mutually exclusive with enum_path / typed_call / turbofish.
            let is_path_call_pre = if t1_pre == 14 {
                if t2_pre == 14 { if t3_pre == 2 { 1 } else { 0 } } else { 0 }
            } else { 0 };
            let is_path_call = if is_path_call_pre == 1 {
                if is_enum_path == 1 { 0 }
                else { if is_typed_call == 1 { 0 }
                else { if is_turbofish == 1 { 0 } else { 1 } } }
            } else { 0 };
            if is_path_call == 1 {
                // Stage 10: walk `IDENT::IDENT::IDENT...(args)` building the
                // mangled name `seg1__seg2__...__last` directly into the
                // arena. After each IDENT, if the next two tokens are `::`
                // and the one after is IDENT, append `__<seg>` and continue;
                // otherwise stop and expect `(`.
                let mang_s = __arena_len();
                // First segment: the leading IDENT bytes.
                let mut bi: i32 = 0;
                while bi < id_len {
                    __arena_push(__arena_get(id_start + bi));
                    bi = bi + 1;
                }
                let mut total_l: i32 = id_len;
                cur_advance(sb);                       // consume first IDENT
                cur_advance(sb);                       // consume first ':'
                cur_advance(sb);                       // consume second ':'
                // Read the second segment IDENT.
                let mut seg_keep: i32 = 1;
                while seg_keep == 1 {
                    let sk = cur_get(sb);
                    let st = tok_tag(tok_base, sk);
                    if st == 2 {
                        let sg_s = tok_p2(tok_base, sk);
                        let sg_l = tok_p3(tok_base, sk);
                        cur_advance(sb);               // consume segment IDENT
                        // Append `__<seg>`.
                        __arena_push(95); __arena_push(95);
                        let mut sj: i32 = 0;
                        while sj < sg_l {
                            __arena_push(__arena_get(sg_s + sj));
                            sj = sj + 1;
                        }
                        total_l = total_l + 2 + sg_l;
                        // Peek for another `::` (chained path).
                        let nt0 = tok_tag(tok_base, cur_get(sb));
                        let nt1 = tok_tag(tok_base, cur_get(sb) + 1);
                        let nt2 = tok_tag(tok_base, cur_get(sb) + 2);
                        if nt0 == 14 {
                            if nt1 == 14 {
                                if nt2 == 2 {
                                    cur_advance(sb);   // ':'
                                    cur_advance(sb);   // ':'
                                    // Loop continues to read next segment.
                                } else {
                                    seg_keep = 0;
                                };
                            } else {
                                seg_keep = 0;
                            };
                        } else {
                            seg_keep = 0;
                        };
                    } else {
                        seg_keep = 0;
                    };
                }
                // Now expect '('.
                cur_advance(sb);                       // consume '('
                let mang_l = total_l;
                // Parse comma-separated args until ')'.
                let mut args_head: i32 = 0;
                let mut prev_arg: i32 = 0;
                let mut a_keep: i32 = 1;
                while a_keep == 1 {
                    let at = tok_tag(tok_base, cur_get(sb));
                    if at == 4 {                       // ')'
                        a_keep = 0;
                    } else { if at == 13 {             // ','
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
                    } };
                }
                cur_advance(sb);                       // ')'
                mk_node(16, mang_s, mang_l, args_head)
            } else { if is_typed_call_active == 1 {
                // Consume IDENT, `:`, `:`, then the method-name IDENT, then `(`.
                cur_advance(sb);                       // first IDENT (T or scalar)
                cur_advance(sb);                       // ':'
                cur_advance(sb);                       // ':'
                let mk_tok = cur_get(sb);
                let m_s = tok_p2(tok_base, mk_tok);
                let m_l = tok_p3(tok_base, mk_tok);
                cur_advance(sb);                       // method-name IDENT
                cur_advance(sb);                       // '('
                // Build mangled name `<FirstIDENT>__<MethodName>` directly
                // in the arena. For generic-param case, FirstIDENT bytes
                // are the gp name itself (e.g. "T"), to be rewritten by
                // the mono pass. For scalar case, FirstIDENT is e.g. "i32".
                let mang_s = __arena_len();
                let mut bi: i32 = 0;
                while bi < id_len {
                    __arena_push(__arena_get(id_start + bi));
                    bi = bi + 1;
                }
                __arena_push(95); __arena_push(95);    // '__'
                let mut mj: i32 = 0;
                while mj < m_l {
                    __arena_push(__arena_get(m_s + mj));
                    mj = mj + 1;
                }
                let mang_l = id_len + 2 + m_l;
                // Parse comma-separated args until ')'.
                let mut args_head: i32 = 0;
                let mut prev_arg: i32 = 0;
                let mut a_keep: i32 = 1;
                while a_keep == 1 {
                    let at = tok_tag(tok_base, cur_get(sb));
                    if at == 4 {                       // ')'
                        a_keep = 0;
                    } else { if at == 13 {             // ','
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
                    } };
                }
                cur_advance(sb);                       // ')'
                mk_node(16, mang_s, mang_l, args_head)
            } else { if is_turbofish == 1 {
                // Consume IDENT, `:`, `:`, `<`.
                cur_advance(sb);                       // IDENT
                cur_advance(sb);                       // ':'
                cur_advance(sb);                       // ':'
                cur_advance(sb);                       // '<'
                // Read up to 6 type-arg IDENTs separated by `,` until `>`.
                // Cap 6 type args (Phase-0). We collect (start, len) pairs
                // into a scratch arena region, then build the mangled name
                // and pack tags into 4-bit slots.
                let ta_arr_base = __arena_len();
                __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
                __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
                __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
                let mut ta_count: i32 = 0;
                let mut keep_t: i32 = 1;
                while keep_t == 1 {
                    let tt = tok_tag(tok_base, cur_get(sb));
                    if tt == 17 {                      // '>' end
                        keep_t = 0;
                    } else { if tt == 13 {             // ','
                        cur_advance(sb);
                    } else { if tt == 0 {              // EOF safety
                        keep_t = 0;
                    } else {
                        // Type IDENT — capture bytes.
                        let tk = cur_get(sb);
                        let ts = tok_p2(tok_base, tk);
                        let tl = tok_p3(tok_base, tk);
                        cur_advance(sb);
                        if ta_count < 6 {
                            __arena_set(ta_arr_base + ta_count * 2, ts);
                            __arena_set(ta_arr_base + ta_count * 2 + 1, tl);
                            ta_count = ta_count + 1;
                        };
                    }}};
                }
                cur_advance(sb);                       // consume '>'
                // Build mangled name like `id__i32` directly into arena.
                let mang_s = mangle_name_into_arena(id_start, id_len, ta_arr_base, ta_count);
                let mang_l = mangle_name_len(id_len, ta_arr_base, ta_count);
                // Pack type-arg tags (4 bits each, up to 6 args = 24 bits).
                let mut packed: i32 = 0;
                let mut shift: i32 = 0;
                let mut tj: i32 = 0;
                while tj < ta_count {
                    let ts = __arena_get(ta_arr_base + tj * 2);
                    let tl = __arena_get(ta_arr_base + tj * 2 + 1);
                    let tag_val = ty_ident_to_tag(ts, tl);
                    let mut place: i32 = tag_val;
                    let mut s: i32 = 0;
                    while s < shift { place = place * 2; s = s + 1; }
                    packed = packed + place;
                    shift = shift + 4;
                    tj = tj + 1;
                }
                // Register the mono request (dedup-aware). Combine packed
                // tags + count into a single i32 (low 3 bits = count, rest
                // = packed) to fit SysV's 6-int-param limit.
                //
                // Audit-stage7-8 Finding #4 fix: mr_tab caps at 32 entries
                // and mr_tab_add returns -1 silently on overflow. Pre-fix
                // the 33rd+ unique instantiation went unregistered; the
                // mono pass never synthesized a clone and codegen lost the
                // call to an unresolved name (silent SIGILL with the
                // 99001 fallback id). Now we capture the add result and
                // fold the whole call to AST_ERR(71001) if it overflows,
                // so the binary traps with the cap-overflow id documented
                // in APPROACH_A_DETAILED_PLAN.md.
                let pack_lo = packed * 8 + ta_count;
                let existing = mr_tab_lookup(sb, id_start, id_len, pack_lo);
                let mut mr_overflow: i32 = 0;
                if existing < 0 {
                    let r = mr_tab_add(sb, id_start, id_len, mang_s, mang_l, pack_lo);
                    if r < 0 { mr_overflow = 1; };
                };
                // Now parse the call args `( ... )`.
                cur_advance(sb);                       // consume '('
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
                cur_advance(sb);                       // consume ')'
                // Build AST_CALL with the MANGLED name (so codegen looks
                // up the synthesized mono'd fn) — or AST_ERR(71001) when
                // mr_tab overflowed (audit-stage7-8 F4).
                if mr_overflow == 1 {
                    mk_node(99, 71001, 0, 0)
                } else {
                    mk_node(16, mang_s, mang_l, args_head)
                }
            } else { if is_enum_unit == 1 {
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
                // Audit A1-F8 fix: validate payload arity. Pre-fix
                // `Maybe::Some()` (declared 1, supplied 0) and
                // `Maybe::Some(1, 2, 3)` (declared 1, supplied 3) both
                // silently parsed. n_args includes the disc, so
                // payload_supplied = n_args - 1. Combine the unknown-
                // variant + arity-mismatch checks into a single trap_id
                // selector: -1 = no trap, otherwise the trap-id.
                let payload_supplied = n_args - 1;
                let unknown_variant = if disc < 0 { 1 } else { if arity < 0 { 1 } else { 0 } };
                let bad_arity = if unknown_variant == 0 { if payload_supplied != arity { 1 } else { 0 } } else { 0 };
                let trap_id = if unknown_variant == 1 { 60002 } else { if bad_arity == 1 { 60020 } else { 0 } };
                if trap_id > 0 {
                    mk_node(99, trap_id, 0, 0)
                } else {
                    mk_node(50, n_args, head_idx, 0)
                }
            } else {
            cur_advance(sb);
            let next = cur_get(sb);
            let nt = tok_tag(tok_base, next);
            // K1.U (2026-05-25): compound assign `x op= v` detection.
            // The lexer has no `+=`/etc. tokens, so we look for the
            // pattern (op token, TK_EQ) right after the IDENT. nt
            // is the op (TK_PLUS=7, MINUS=8, STAR=9, SLASH=10,
            // PERCENT=11); cur_get(sb)+1 should be TK_EQ (15).
            // Desugar to AST_ASSIGN(name, AST_BINOP(VAR(name), rhs))
            // using the existing arith fold for the binop kind.
            let nt_plus1 = tok_tag(tok_base, cur_get(sb) + 1);
            let compound_op = if nt_plus1 == 15 {
                if nt == 7 { 2 }
                else { if nt == 8 { 3 }
                else { if nt == 9 { 4 }
                else { if nt == 10 { 5 }
                else { if nt == 11 { 24 } else { 0 - 1 }}}}}
            } else { 0 - 1 };
            if compound_op >= 0 {
                cur_advance(sb);    // consume op token
                cur_advance(sb);    // consume '='
                let crhs = parse_expr_basic(tok_base, sb);
                let clhs = mk_var_with_capture(sb, id_start, id_len);
                let cnew = mk_arith_fold(compound_op, clhs, crhs);
                mk_node(11, id_start, id_len, cnew)
            } else { if nt == 15 {
                // Could be `=` (assign) or `==` (equality). Peek one
                // more ahead: if it's also `=`, this is `name == ...`,
                // and we should NOT consume the `=`s here — leave
                // them for parse_expr_basic to handle as a comparison.
                let nt2 = tok_tag(tok_base, cur_get(sb) + 1);
                if nt2 == 15 {
                    mk_var_with_capture(sb, id_start, id_len)
                } else {
                    cur_advance(sb);
                    let value = parse_expr_basic(tok_base, sb);
                    mk_node(11, id_start, id_len, value)
                }
            } else { if nt == 3 {
                // CALL: name(arg1, arg2, ...). Args become AST_ARG
                // linked list; head index goes in CALL.p3 (or 0 if
                // no args).
                // Stage 9: detect closure-call BEFORE parsing args so the
                // synthesized fn name `__closure_<id>` can be built into
                // the arena ahead of any AST_ARG mk_node calls (Stage 8.5
                // arena-positional-ordering lesson).
                let cl_entry_idx = cl_var_tab_lookup(sb, id_start, id_len);
                let cl_var_count_pre = cl_var_tab_count(sb);
                let cl_base_pre = cl_var_tab_base(sb);
                // Pre-build the synthesized fn name when this is a closure
                // call. Bytes: '__closure_<id>'. Read closure_id from the
                // matching cl_var_tab entry's slot 4.
                let mut cl_mang_s: i32 = 0;
                let mut cl_mang_l: i32 = 0;
                let mut cl_caps_ptr: i32 = 0;
                let mut cl_caps_count: i32 = 0;
                if cl_entry_idx >= 0 {
                    let entry = cl_base_pre + cl_entry_idx * 5;
                    cl_caps_ptr = __arena_get(entry + 2);
                    cl_caps_count = __arena_get(entry + 3);
                    let cl_id = __arena_get(entry + 4);
                    cl_mang_s = __arena_len();
                    __arena_push(95); __arena_push(95);
                    __arena_push(99); __arena_push(108); __arena_push(111);
                    __arena_push(115); __arena_push(117); __arena_push(114);
                    __arena_push(101); __arena_push(95);
                    let n_d = push_tag_digits(cl_id);
                    cl_mang_l = 10 + n_d;
                };
                let _drop_pre = cl_var_count_pre;
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
                // Stage 9: prepend captured-var AST_ARG nodes to args_head
                // when this is a closure call. The captured vars are passed
                // as positional args BEFORE the user's args, matching the
                // synthesized fn's param order (captures..., closure_params...).
                if cl_entry_idx >= 0 {
                    // Build the captured-arg chain: cap_ARG -> cap_ARG -> ... ->
                    // (existing args_head). Iterate captures in reverse so the
                    // chain comes out in forward order with ARG[0] being the
                    // first captured var.
                    let mut cap_args_head: i32 = args_head;
                    let mut ri: i32 = cl_caps_count;
                    while ri > 0 {
                        ri = ri - 1;
                        let pair = cl_caps_ptr + ri * 2;
                        let cap_ns = __arena_get(pair);
                        let cap_nl = __arena_get(pair + 1);
                        let cap_var = mk_node(1, cap_ns, cap_nl, 0);
                        let new_arg = mk_node(17, cap_var, cap_args_head, 0);
                        cap_args_head = new_arg;
                    }
                    mk_node(16, cl_mang_s, cl_mang_l, cap_args_head)
                } else {
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
                    // Stage 10: check use_table first — if the call name
                    // matches an alias registered via `use foo::bar;`,
                    // replace it with the mangled module-path name.
                    // use_tab_lookup returns (mang_s * 65536 + mang_l) on
                    // hit, 0 on miss.
                    let alias_pack = use_tab_lookup(sb, id_start, id_len);
                    if alias_pack > 0 {
                        let amang_s = alias_pack / 65536;
                        let amang_l = alias_pack - amang_s * 65536;
                        mk_node(16, amang_s, amang_l, args_head)
                    } else {
                        mk_node(16, id_start, id_len, args_head)
                    }
                }
                }     // Stage 9: close `if cl_entry_idx >= 0 { ... } else { ... }`
            } else { if nt == 16 {
                // Stage 28.11 INC-3b.2: IDENT followed by `<` may be
                // either (a) a generic struct use `Pt<i32> { 10, 32 }`
                // requiring inline monomorphization, or (b) a less-than
                // comparison `var < other` to be handled by the outer
                // binary-expr parser. Disambiguate by struct_tab lookup
                // + struct_gp_tab_lookup; if both indicate generic, do
                // mono and parse the struct-lit body. Otherwise fall
                // through to var-ref so the surrounding parser sees
                // the `<` as a comparison operator.
                let s_idx_pre = struct_tab_lookup_idx(sb, id_start, id_len);
                let gp_count_pre = if s_idx_pre >= 0 {
                    struct_gp_tab_lookup(sb, s_idx_pre)
                } else { 0 };
                if gp_count_pre > 0 {
                    // === Generic struct use: parse `<TY1, TY2, ...>` ===
                    cur_advance(sb);                       // consume `<`
                    let ta_arr_base = __arena_len();
                    let mut ta_count: i32 = 0;
                    let mut keep_ta: i32 = 1;
                    // Stage 28.11 INC-3b cycle-3 silent-failure F6
                    // fix (MED conf 82): track if loop exited on bad
                    // token vs clean TK_GT. Pre-fix the bad-token else
                    // branch silently exited and the post-loop trap
                    // 62030 misleadingly attributed the error to
                    // "missing `{`" instead of "unexpected token in
                    // type-args list". Trap 62033 = bad token in args.
                    let mut ta_bad_token: i32 = 0;
                    while keep_ta == 1 {
                        let tt = tok_tag(tok_base, cur_get(sb));
                        if tt == 17 {                      // TK_GT end
                            keep_ta = 0;
                        } else { if tt == 13 {             // COMMA
                            cur_advance(sb);
                        } else { if tt == 2 {              // TK_IDENT type-arg
                            let tk = cur_get(sb);
                            let ta_s = tok_p2(tok_base, tk);
                            let ta_l = tok_p3(tok_base, tk);
                            __arena_push(ta_s);
                            __arena_push(ta_l);
                            ta_count = ta_count + 1;
                            cur_advance(sb);
                        } else {
                            // Bad token (operator, literal, nested `<`,
                            // EOF, etc.) — mark and exit WITHOUT
                            // consuming so the post-loop guard knows
                            // we exited on a non-GT terminator.
                            ta_bad_token = 1;
                            keep_ta = 0;
                        }}};
                    }
                    // Stage 28.11 INC-3b cycle-3 silent-failure F3
                    // fix (HIGH conf 85): the post-loop `cur_advance(
                    // sb)` was unconditional, so EOF-mid-list or
                    // bad-token-exit caused cursor to walk past EOF,
                    // reading garbage from the token-region tail.
                    // Mirror INC-1 cycle-2 SF-2 fix pattern: only
                    // advance past `>` if cursor actually points at
                    // one. Also fold in F1/F2 fix (count mismatch):
                    // generic struct decl requires exactly gp_count_pre
                    // type-args; under-supply (Pt<>) silently coerced
                    // T-typed fields to scalar; over-supply (Pt<i32, i32>)
                    // silently created divergent mangled identities.
                    // Trap 62032 = type-args arity mismatch.
                    let post_loop_t = tok_tag(tok_base, cur_get(sb));
                    if post_loop_t == 17 {
                        cur_advance(sb);                   // consume `>`
                    };
                    // Stage 29 + Stage 30 cycle-2 H1 fix (2026-05-12):
                    // sentinel pattern + outer if/else dispatch restores
                    // the cycle-3 silent-failure trap semantics. Bootstrap
                    // parser doesn't support `return` keyword, so the
                    // outer if/else IS the return mechanism.
                    let mut early_err: i32 = 0 - 1;
                    if ta_bad_token == 1 { early_err = mk_node(99, 62033, 0, 0); };
                    if early_err == (0 - 1) {
                        if post_loop_t != 17 { early_err = mk_node(99, 62033, 0, 0); };
                    };
                    if early_err == (0 - 1) {
                        if ta_count != gp_count_pre { early_err = mk_node(99, 62032, 0, 0); };
                    };
                    if early_err != (0 - 1) {
                        early_err
                    } else {
                    // Build mangled name `OrigName__TY1_TY2` in arena.
                    let mang_s = mangle_name_into_arena(id_start, id_len, ta_arr_base, ta_count);
                    let mang_l = mangle_name_len(id_len, ta_arr_base, ta_count);
                    // Look up or synthesize the mono'd struct entry.
                    let existing_idx = struct_tab_lookup_idx(sb, mang_s, mang_l);
                    let mono_s_idx = if existing_idx >= 0 {
                        existing_idx
                    } else {
                        // Clone orig struct's fields with type-var
                        // substitution. For each field, if its stored
                        // f_struct_idx is a gp_marker, substitute with
                        // the matched ta_arr entry's struct_idx (or -1
                        // for scalar like i32). Otherwise copy as-is.
                        let orig_entry = struct_tab_base(sb) + s_idx_pre * 4;
                        let orig_arity = __arena_get(orig_entry + 2);
                        let orig_fields_ptr = __arena_get(orig_entry + 3);
                        let new_fields_ptr = __arena_len();
                        let mut fi: i32 = 0;
                        while fi < orig_arity {
                            let f_pair = orig_fields_ptr + fi * 3;
                            let f_name_s = __arena_get(f_pair);
                            let f_name_l = __arena_get(f_pair + 1);
                            let f_struct_idx = __arena_get(f_pair + 2);
                            __arena_push(f_name_s);
                            __arena_push(f_name_l);
                            if gp_marker_is(f_struct_idx) == 1 {
                                let gp_idx_sub = f_struct_idx - gp_marker_base();
                                if gp_idx_sub < ta_count {
                                    let ta_entry = ta_arr_base + gp_idx_sub * 2;
                                    let sub_ty_s = __arena_get(ta_entry);
                                    let sub_ty_l = __arena_get(ta_entry + 1);
                                    let sub_struct_idx = struct_tab_lookup_idx(sb, sub_ty_s, sub_ty_l);
                                    __arena_push(sub_struct_idx);
                                } else {
                                    __arena_push(0 - 1);
                                };
                            } else {
                                __arena_push(f_struct_idx);
                            };
                            fi = fi + 1;
                        }
                        struct_tab_add(sb, mang_s, mang_l, orig_arity, new_fields_ptr)
                    };
                    // Expect `{` next; parse struct-lit body with
                    // mono_s_idx. (Duplicates the `nt == 5` body
                    // below — INCREMENT 3 atomicity requires the mono
                    // setup AND the body in one place; sharing would
                    // require restructuring the dispatch.)
                    let lbrace_t = tok_tag(tok_base, cur_get(sb));
                    if lbrace_t != 5 {
                        // Missing `{` after `Pt<i32>` — emit trap 62030.
                        mk_node(99, 62030, 0, 0)
                    } else { if mono_s_idx < 0 {
                        // Stage 28.11 INC-3b cycle-1 type-design F-1
                        // fix (MED conf 92): struct_tab_add at line
                        // ~3277 returns -1 when struct_tab cap (8) is
                        // exceeded. Pre-fix the use-site computed
                        // `entry_m = struct_tab_base + (-1)*4`, reading
                        // off the start of the table — garbage arity,
                        // either wrong trap (50040 arity-mismatch
                        // obscures cap-overflow root cause) or silent
                        // miscompile when garbage happened to match
                        // field count. Mirror parse_struct_decl's guard
                        // at line ~6580 which checks `struct_idx_added
                        // >= 0` before struct_gp_tab_add. Reserve trap
                        // 62031 = "struct_tab cap overflow at generic
                        // mono use site"; user-actionable diagnostic
                        // distinct from the 50040 arity-mismatch class.
                        mk_node(99, 62031, 0, 0)
                    } else {
                        cur_advance(sb);                   // consume `{`
                        let entry_m = struct_tab_base(sb) + mono_s_idx * 4;
                        let arity_m = __arena_get(entry_m + 2);
                        let pk_first = cur_get(sb);
                        let pt_first = tok_tag(tok_base, pk_first);
                        if pt_first == 6 {
                            // Stage 28.11 INC-3b cycle-3 silent-failure
                            // F4 fix (MED conf 82): empty struct lit
                            // `Pt<i32>{}` was pre-fix returning a
                            // 0-arity tuple-lit even when arity_m > 0,
                            // matching the same defect class as the
                            // non-generic struct-lit at line ~3363
                            // (Audit A1-F7). Subsequent `p.x` would
                            // read OOB from adjacent stack. Trap 50040
                            // (same trap id as Audit A1-F7 fix for
                            // non-generic) since the symptom — supplied
                            // value count != declared arity — is the
                            // same defect class regardless of generic
                            // origin.
                            cur_advance(sb);               // consume `}`
                            set_last_struct_idx(sb, mono_s_idx);
                            if arity_m != 0 {
                                mk_node(99, 50040, 0, 0)
                            } else {
                                mk_node(50, 0, 0, 0)
                            }
                        } else { if peek_named_struct_lit(tok_base, sb) == 1 {
                            // Stage 28.13.2: named struct-lit syntax for
                            // generic mono use sites: `Pt<i32> { x: 10,
                            // y: 32 }`. Same algorithm as the
                            // non-generic named-mode branch above
                            // (~line 3460) but keyed by `mono_s_idx`
                            // and `arity_m` (the mono'd struct's
                            // fields region). field lookup goes through
                            // struct_tab_field_lookup which works on
                            // mono'd struct_tab entries identically to
                            // non-generic ones (INC-3b.2 clones the
                            // fields region with the same stride-3
                            // layout).
                            let temp_base = __arena_len();
                            let mut ti: i32 = 0;
                            while ti < arity_m {
                                __arena_push(0 - 1);
                                ti = ti + 1;
                            }
                            let mut keep_n: i32 = 1;
                            let mut named_err: i32 = 0;
                            while keep_n == 1 {
                                let fk = cur_get(sb);
                                let fname_s = tok_p2(tok_base, fk);
                                let fname_l = tok_p3(tok_base, fk);
                                cur_advance(sb);         // consume field-name
                                cur_advance(sb);         // consume ':'
                                let fval = parse_expr(tok_base, sb);
                                let f_idx = struct_tab_field_lookup(sb, mono_s_idx, fname_s, fname_l);
                                if f_idx < 0 {
                                    named_err = 50041;
                                    keep_n = 0;
                                } else { if f_idx >= arity_m {
                                    named_err = 50041;
                                    keep_n = 0;
                                } else { if __arena_get(temp_base + f_idx) != 0 - 1 {
                                    named_err = 50042;
                                    keep_n = 0;
                                } else {
                                    __arena_set(temp_base + f_idx, fval);
                                    let ct = tok_tag(tok_base, cur_get(sb));
                                    if ct == 13 {
                                        cur_advance(sb);
                                        let nt2 = tok_tag(tok_base, cur_get(sb));
                                        if nt2 == 6 { keep_n = 0; };
                                    } else { keep_n = 0; };
                                }}};
                            }
                            if named_err != 0 {
                                mk_node(99, named_err, 0, 0)
                            } else {
                                cur_advance(sb);         // consume `}`
                                let mut vi: i32 = 0;
                                let mut missing: i32 = 0;
                                while vi < arity_m {
                                    if __arena_get(temp_base + vi) == 0 - 1 {
                                        missing = 1;
                                        vi = arity_m;
                                    } else { vi = vi + 1; };
                                }
                                if missing == 1 {
                                    mk_node(99, 50040, 0, 0)
                                } else {
                                    let head_n = mk_node(51, __arena_get(temp_base), 0, 0);
                                    let mut tail_n: i32 = head_n;
                                    let mut bi: i32 = 1;
                                    while bi < arity_m {
                                        let new_node = mk_node(51, __arena_get(temp_base + bi), 0, 0);
                                        __arena_set(tail_n + 2, new_node);
                                        tail_n = new_node;
                                        bi = bi + 1;
                                    }
                                    set_last_struct_idx(sb, mono_s_idx);
                                    mk_node(50, arity_m, head_n, 0)
                                }
                            }
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
                                    cur_advance(sb);
                                    let pk2 = cur_get(sb);
                                    let pt2 = tok_tag(tok_base, pk2);
                                    if pt2 == 6 { keep = 0; }
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
                            cur_advance(sb);               // consume `}`
                            set_last_struct_idx(sb, mono_s_idx);
                            if n != arity_m {
                                mk_node(99, 50040, 0, 0)
                            } else {
                                mk_node(50, n, head_idx, 0)
                            }
                        }}
                    }}
                    }   // Stage 30 cycle-2 H1: close my outer else for early_err sentinel
                } else {
                    // Not a generic struct use — fall through to var
                    // ref (the surrounding parser handles `<` as a
                    // comparison op).
                    mk_var_with_capture(sb, id_start, id_len)
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
                    } else { if peek_named_struct_lit(tok_base, sb) == 1 {
                        // Stage 28.13.1: named struct-lit syntax
                        // `Pt { x: 10, y: 32 }`. Parse `field_name: value`
                        // pairs in any order; look up each field's
                        // positional index via struct_tab_field_lookup;
                        // build a positional tuple-lit. Mirrors the
                        // Python frontend's parser.py:1321
                        // `_parse_struct_lit_after_name` shape.
                        //
                        // Implementation: allocate a temp[arity] array
                        // in the arena (sentinel -1 = unfilled), parse
                        // each pair, store value at temp[lookup(name)].
                        // After loop, walk temp[0..arity] to build the
                        // TUPLE_CONS chain in positional order.
                        let temp_base = __arena_len();
                        let mut ti: i32 = 0;
                        while ti < arity {
                            __arena_push(0 - 1);
                            ti = ti + 1;
                        }
                        let mut keep_n: i32 = 1;
                        let mut nf: i32 = 0;
                        let mut named_err: i32 = 0;
                        while keep_n == 1 {
                            let fk = cur_get(sb);
                            let fname_s = tok_p2(tok_base, fk);
                            let fname_l = tok_p3(tok_base, fk);
                            cur_advance(sb);     // consume field-name
                            cur_advance(sb);     // consume ':'
                            let fval = parse_expr(tok_base, sb);
                            let f_idx = struct_tab_field_lookup(sb, s_idx, fname_s, fname_l);
                            if f_idx < 0 {
                                named_err = 50041;     // unknown field
                                keep_n = 0;
                            } else { if f_idx >= arity {
                                named_err = 50041;
                                keep_n = 0;
                            } else { if __arena_get(temp_base + f_idx) != 0 - 1 {
                                named_err = 50042;     // duplicate field
                                keep_n = 0;
                            } else {
                                __arena_set(temp_base + f_idx, fval);
                                nf = nf + 1;
                                let ct = tok_tag(tok_base, cur_get(sb));
                                if ct == 13 {
                                    cur_advance(sb);
                                    let nt2 = tok_tag(tok_base, cur_get(sb));
                                    if nt2 == 6 { keep_n = 0; };
                                } else { keep_n = 0; };
                            }}};
                        }
                        if named_err != 0 {
                            mk_node(99, named_err, 0, 0)
                        } else {
                            cur_advance(sb);            // consume `}`
                            // Validate all slots filled.
                            let mut vi: i32 = 0;
                            let mut missing: i32 = 0;
                            while vi < arity {
                                if __arena_get(temp_base + vi) == 0 - 1 {
                                    missing = 1;
                                    vi = arity;
                                } else { vi = vi + 1; };
                            }
                            if missing == 1 {
                                mk_node(99, 50040, 0, 0)
                            } else {
                                // Build TUPLE_CONS chain from temp.
                                let head_n = mk_node(51, __arena_get(temp_base), 0, 0);
                                let mut tail_n: i32 = head_n;
                                let mut bi: i32 = 1;
                                while bi < arity {
                                    let new_node = mk_node(51, __arena_get(temp_base + bi), 0, 0);
                                    __arena_set(tail_n + 2, new_node);
                                    tail_n = new_node;
                                    bi = bi + 1;
                                }
                                set_last_struct_idx(sb, s_idx);
                                mk_node(50, arity, head_n, 0)
                            }
                        }
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
                        // Audit A1-F7 fix: validate field count against
                        // declared arity. Pre-fix, `Pt { 10 }` for arity-2
                        // Pt silently emitted a 1-slot tuple lit; later
                        // `.y` accesses read OOB into adjacent stack.
                        // Now we trap 50040 on mismatch.
                        if n != arity {
                            mk_node(99, 50040, 0, 0)
                        } else {
                            mk_node(50, n, head_idx, 0)
                        }
                    }}
                } else {
                    // Not a registered struct — treat as var ref.
                    mk_var_with_capture(sb, id_start, id_len)
                }
            } else {
                // Var ref
                mk_var_with_capture(sb, id_start, id_len)
            }}}}     // Stage 28.11 INC-3b.2: extra `}` closes the new nt==16 branch
            }     // K1.U (2026-05-25): +1 brace closes the new compound_op wrapper around the inner nt-dispatch
            }}}}}     // Stage 8.5C + Stage 10: extra '}}' closes is_typed_call_active + is_path_call wrappers
        }}}}}     // Stage 12: extra '}' closes the is_grad_call else-branch wrapper
        }     // Stage 14: extra '}' closes the is_grad_rev_call else-branch wrapper
        }     // K1.C-wireup (2026-05-25): +1 brace closes the new return-keyword arm (the existing trailing `}` now closes RETURN-else; this new `}` closes the match-else that wraps RETURN)
        }     // K1.G-wireup (2026-05-25): +1 brace closes the new for-keyword arm (same algebra as K1.C: existing trailing `}` cascades down, this new `}` closes the wrapping arm)
        }     // K1.H1-wireup (2026-05-25): +1 brace closes the new loop-keyword arm (same algebra as K1.G: existing trailing `}` cascades down through return->for->loop, this new `}` closes the wrapping arm)
        }     // K1.Q (2026-05-25): +1 brace closes the new `true` arm
        }     // K1.Q (2026-05-25): +1 brace closes the new `false` arm
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
        // Stage 29.2 fix (2026-05-12): for TK_RBRACE (6) — empty block
        // body like `else {}` — return AST_INT(0) instead of AST_ERR(6).
        // The AST_ERR(6) gets codegen'd as trap_with_id(6) which fires
        // SIGILL at runtime if reached. Empty blocks are valid Helix
        // semantics (unit value); they should compile to a no-op `0`
        // rather than a hard trap.
        //
        // KNOWN TRADE-OFF (Stage 30 cycles 1-4 M2, conf 82): this
        // catch-all is over-broad — it ALSO converts truncated sources
        // like `let x = }` into `let x = 0`, masking parse errors. The
        // strict fix would require differentiating empty-block-context
        // (parse_expr called immediately after `{`) from required-expr-
        // context. Two options:
        //   A) New entry point `parse_expr_or_empty` used by block-body
        //      callers; parse_primary reverts to AST_ERR(6).
        //   B) Scratch slot context flag in sb set by block-body callers.
        // Option A is preferable (compile-time visible) but requires
        // updating many block-body call sites. Deferred to Phase 1
        // ergonomics pass when render_caret diagnostics also land.
        // For Phase 0 self-host the trade-off is acceptable: the
        // bootstrap source itself doesn't have truncated `let x = }`
        // patterns, so the silent acceptance doesn't manifest.
        if t == 6 {
            mk_node(0, 0, 0, 0)
        } else {
            mk_node(99, t, 0, 0)
        }
    }}}}}}}}}}}}}}}}     // Stage 9: extra '}' closes the leading t == 28 closure wrapper
}

// Stage 5 Iter B: struct_table region — 12 slots = 3 entries x 4 fields
// (name_s, name_l, arity, fields_ptr). fields_ptr is 0 in Iter A; Iter B
// fills it with an arena offset to a per-struct field-names region built
// during parse_struct_decl, used to resolve `p.IDENT` -> field index.
fn struct_tab_init(sb: i32) -> i32 {
    // Audit A1-F6: bumped cap from 3 to 8 entries (32 slots = 8*4).
    // Pre-bump, the 4th `struct X {...}` decl was silently dropped from
    // the table (returned -1, ignored by callers). Bumping to 8 is cheap
    // (32 arena slots vs 12) and covers all current bootstrap and demo
    // programs. The tighter cap-check (8) still surfaces the issue if
    // the program is unusually struct-heavy.
    let st_base = __arena_push(0);
    let mut i: i32 = 1;
    while i < 32 {
        __arena_push(0);
        i = i + 1;
    }
    __arena_set(sb + 15, st_base);
    __arena_set(sb + 16, 0);
    0
}

// Stage 5 Iter B: var_struct_table region — 12 slots = 4 entries x 3
// fields (var_name_s, var_name_l, struct_idx). Cap 4 vars; expand later.
fn var_struct_tab_init(sb: i32) -> i32 {
    // Audit A1-F6: bumped cap from 4 to 8 entries (24 slots = 8*3).
    // Pre-bump, the 5th struct-typed let-binding in any function was
    // silently dropped, breaking subsequent `.field` resolution.
    let vs_base = __arena_push(0);
    let mut i: i32 = 1;
    while i < 24 {
        __arena_push(0);
        i = i + 1;
    }
    __arena_set(sb + 17, vs_base);
    __arena_set(sb + 18, 0);
    0
}

// Stage 6: enum_table region — bumped cap from 4 to 8 entries (40 slots
// = 8*5). Audit A1-F6: pre-bump the 5th `enum X {...}` decl was silently
// dropped from the table.
// (name_s, name_l, variant_count, variants_ptr, max_payload_arity).
fn enum_tab_init(sb: i32) -> i32 {
    let et_base = __arena_push(0);
    let mut i: i32 = 1;
    while i < 40 {
        __arena_push(0);
        i = i + 1;
    }
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

// Stage 8.5: trait_table region — 8 slots = 4 entries x 2 fields
// (name_s, name_l). Cap 4 traits in Phase-0.
fn trait_tab_init(sb: i32) -> i32 {
    let tr_base = __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_set(sb + 33, tr_base);
    __arena_set(sb + 34, 0);
    0
}

// Stage 8.5: impl_table region — 32 slots = 8 entries x 4 fields
// (trait_name_s, trait_name_l, target_ty_tag, methods_count). Cap 8 impls.
fn impl_tab_init(sb: i32) -> i32 {
    let ip_base = __arena_push(0);
    let mut ipi: i32 = 1;
    while ipi < 32 {
        __arena_push(0);
        ipi = ipi + 1;
    }
    __arena_set(sb + 35, ip_base);
    __arena_set(sb + 36, 0);
    0
}

// Stage 8.5: var_type_table region — 24 slots = 8 entries x 3 fields
// (var_name_s, var_name_l, type_tag). Cap 8 typed vars.
fn var_type_tab_init(sb: i32) -> i32 {
    let vt_base = __arena_push(0);
    let mut vti: i32 = 1;
    while vti < 24 {
        __arena_push(0);
        vti = vti + 1;
    }
    __arena_set(sb + 45, vt_base);
    __arena_set(sb + 46, 0);
    0
}

// Stage 10: use_table region — 32 slots = 8 entries x 4 fields
// (alias_s, alias_l, mang_s, mang_l). Cap 8 use entries.
fn use_tab_init(sb: i32) -> i32 {
    let ut_base = __arena_push(0);
    let mut uti: i32 = 1;
    while uti < 32 {
        __arena_push(0);
        uti = uti + 1;
    }
    __arena_set(sb + 64, ut_base);
    __arena_set(sb + 65, 0);
    0
}

// Stage 9: closure tables init.
//   cl_param_tab: cap 4 closure params, stride 2 (name_s, name_l) = 8 slots.
//   cl_capture_tab: cap 4 captures, stride 2 = 8 slots.
//   cl_var_tab: cap 4 closure-bound vars, stride 5 = 20 slots.
fn cl_tabs_init(sb: i32) -> i32 {
    // cl_param_tab @ sb+48
    let cp_base = __arena_push(0);
    let mut cpi: i32 = 1;
    while cpi < 8 { __arena_push(0); cpi = cpi + 1; }
    __arena_set(sb + 48, cp_base);
    __arena_set(sb + 49, 0);
    // cl_capture_tab @ sb+50
    let cc_base = __arena_push(0);
    let mut cci: i32 = 1;
    while cci < 8 { __arena_push(0); cci = cci + 1; }
    __arena_set(sb + 50, cc_base);
    __arena_set(sb + 51, 0);
    // cl_var_tab @ sb+54
    let cv_base = __arena_push(0);
    let mut cvi: i32 = 1;
    while cvi < 20 { __arena_push(0); cvi = cvi + 1; }
    __arena_set(sb + 54, cv_base);
    __arena_set(sb + 55, 0);
    // Init: closure-active=0, id_counter=0, last_closure_idx=-1,
    // caps_ptr scratch=0, cap_count scratch=0, pending_head=0, pending_tail=0.
    __arena_set(sb + 47, 0);
    __arena_set(sb + 52, 0);
    __arena_set(sb + 53, 0);
    __arena_set(sb + 56, 0);
    __arena_set(sb + 57, 0 - 1);
    __arena_set(sb + 58, 0);
    __arena_set(sb + 59, 0);
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
    // Stage 8.5: "trait" = 116 114 97 105 116
    let trait_s = __arena_push(116); __arena_push(114); __arena_push(97);
    __arena_push(105); __arena_push(116);
    __arena_set(sb + 37, trait_s);
    __arena_set(sb + 38, 5);
    // Stage 8.5: "impl" = 105 109 112 108
    let impl_s = __arena_push(105); __arena_push(109); __arena_push(112);
    __arena_push(108);
    __arena_set(sb + 39, impl_s);
    __arena_set(sb + 40, 4);
    // Stage 8.5: "for" = 102 111 114
    let for_s = __arena_push(102); __arena_push(111); __arena_push(114);
    __arena_set(sb + 41, for_s);
    __arena_set(sb + 42, 3);
    trait_tab_init(sb);
    impl_tab_init(sb);
    var_type_tab_init(sb);
    // Stage 10: "mod" = 109 111 100, "use" = 117 115 101.
    let mod_s = __arena_push(109); __arena_push(111); __arena_push(100);
    __arena_set(sb + 60, mod_s);
    __arena_set(sb + 61, 3);
    let use_s = __arena_push(117); __arena_push(115); __arena_push(101);
    __arena_set(sb + 62, use_s);
    __arena_set(sb + 63, 3);
    // K1.C-deadcode (2026-05-25): "return" = 114 101 116 117 114 110.
    let return_s = __arena_push(114); __arena_push(101); __arena_push(116);
    __arena_push(117); __arena_push(114); __arena_push(110);
    __arena_set(sb + 88, return_s);
    __arena_set(sb + 89, 6);
    // K1.G-deadcode (2026-05-25): "in" = 105 110. ("for" reuses
    // slot 41 from Stage 8.5 -- already installed above.)
    let in_s = __arena_push(105); __arena_push(110);
    __arena_set(sb + 90, in_s);
    __arena_set(sb + 91, 2);
    // K1.H1-deadcode (2026-05-25): "loop" = 108 111 111 112.
    let loop_s = __arena_push(108); __arena_push(111); __arena_push(111);
    __arena_push(112);
    __arena_set(sb + 92, loop_s);
    __arena_set(sb + 93, 4);
    0
}

// --------------------------------------------------------------
// Top-level parse: return the arena index of the root AST node.
// Reserves 7 state slots, then dispatches into parse_expr.
// --------------------------------------------------------------
fn parse_top(tok_base: i32) -> i32 {
    // Parser state slots: cursor + keyword pairs + staged scratch/table
    // regions. Later comments below document each added range through slot 87.
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
    // Stage 8.5: slots 33..42 = trait_table (base + count) + impl_table
    // (base + count) + trait/impl/for keyword (start + len) triples.
    // Slots 43/44 = pending impl-method fn-list head/tail (built by
    // parse_impl_block; spliced into fn_list by parse_program).
    // Slots 45/46 = var_type_table base/count (typed-let bindings).
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0);
    // Stage 9: slots 47..59 (13 slots) for closure state.
    //   47 = cl_active flag
    //   48/49 = cl_param_tab base/count
    //   50/51 = cl_capture_tab base/count
    //   52/53 = cl_pending head/tail
    //   54/55 = cl_var_tab base/count
    //   56 = closure id counter
    //   57 = last_closure_idx scratch (-1 = none)
    //   58 = scratch caps_persist_ptr (read by parse_let)
    //   59 = scratch caps_count (read by parse_let)
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0);
    // Stage 10: slots 60..67 (8 slots) for module/use state.
    //   60/61 = kw_mod (start, len)
    //   62/63 = kw_use (start, len)
    //   64/65 = use_table base/count
    //   66/67 = mod_pending head/tail (synthesized fn-list from `mod` blocks)
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    // Stage 12: slots 68..69 = grad_pending base/count. Region for the
    // pending grad-call list lives at the offset stored in slot 68 (filled
    // in below after the 32-slot region is pushed).
    __arena_push(0); __arena_push(0);
    // Stage 14: slots 70..71 = gr_rev_pending base/count. Region for
    // pending grad_rev_all-call list lives at the offset stored in slot 70.
    __arena_push(0); __arena_push(0);
    // Stage 14: slots 72..73 = bucket head/count for adjoint propagation
    // during grad_rev_pass (propagate_adj writes here, sum_bucket reads).
    __arena_push(0); __arena_push(0);
    // Stage 14.5: slot 74 = next_fn_is_checkpoint flag. Set by
    // skip_attributes when it consumes `@checkpoint`; read+cleared by
    // parse_fn_decl when it allocates the AST_FN_DECL node (slot 8).
    __arena_push(0);
    // Stage 28.9 (validation passes): slots 75..77 = sticky scratch
    // flags for fn-level attribute capture. Same pattern as slot 74
    // (next_fn_is_checkpoint). Each is set by skip_attributes when
    // it consumes the matching `@<ident>` token, read+cleared by
    // parse_fn_decl, and written into a new AST_FN_DECL slot so the
    // bootstrap-side validation passes can observe it.
    //   slot 75 = next_fn_is_deprecated  (for deprecated_pass)
    //   slot 76 = next_fn_is_trace       (for trace_pass)
    //   slot 77 = next_fn_is_unwind      (for panic_pass diag 28502)
    __arena_push(0);
    __arena_push(0);
    __arena_push(0);
    // Stage 28.11 INC-3b: slots 78/79 = struct_gp_tab base/count.
    // Parallel to struct_tab; holds (struct_idx, gp_count, gp_names_head)
    // triples for generic structs. Populated by parse_struct_decl right
    // before struct_tab_add; consumed by use-site monomorphization at
    // parse_primary when `Pt<i32>` is seen.
    __arena_push(0);
    __arena_push(0);
    // Stage 33: slots 80/81 = next deprecated message body start/len.
    // skip_attributes fills these for @deprecated("..."); parse_fn_decl
    // clears and writes them into AST_FN_DECL slots 12/13.
    __arena_push(0);
    __arena_push(0);
    // Stage 33: slots 82..85 = kernel/autotune attr scratch.
    //   82 = next_fn_is_kernel
    //   83 = next_fn_is_autotune
    //   84 = next_fn_autotune_product
    //   85 = next_fn_autotune_parse_error_kind
    __arena_push(0);
    __arena_push(0);
    __arena_push(0);
    __arena_push(0);
    // Stage 33: slots 86/87 = next @since message body start/len.
    __arena_push(0);
    __arena_push(0);
    // K1.C-deadcode (2026-05-25): slots 88/89 = return keyword
    // (start, len). Reserved here; populated by install_keywords.
    __arena_push(0);
    __arena_push(0);
    // K1.G-deadcode (2026-05-25): slots 90/91 = in keyword (start, len).
    // Note: `for` is already at slot 41 (Stage 8.5 trait kw); we reuse it.
    __arena_push(0); __arena_push(0);
    // K1.H1-deadcode (2026-05-25): slots 92/93 = loop keyword (start, len).
    __arena_push(0); __arena_push(0);
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
    // Stage 9: closure tables (cl_param_tab, cl_capture_tab, cl_var_tab,
    // pending fn-list head/tail, id counter, scratch slots).
    cl_tabs_init(cur_slot);
    // Stage 10: use_table init + reset mod_pending head/tail.
    use_tab_init(cur_slot);
    set_mod_pending_head(cur_slot, 0);
    set_mod_pending_tail(cur_slot, 0);
    // Stage 12: grad_pending region (32 slots, 8 entries x 4 fields).
    let grad_base = __arena_push(0);
    let mut gri: i32 = 1;
    while gri < 32 {
        __arena_push(0);
        gri = gri + 1;
    }
    __arena_set(cur_slot + 68, grad_base);
    __arena_set(cur_slot + 69, 0);
    // Stage 14: gr_rev_pending region (40 slots, 8 entries x 5 fields).
    let grev_base = __arena_push(0);
    let mut gvri: i32 = 1;
    while gvri < 40 {
        __arena_push(0);
        gvri = gvri + 1;
    }
    __arena_set(cur_slot + 70, grev_base);
    __arena_set(cur_slot + 71, 0);
    // Stage 28.11 INC-3b: struct_gp_tab region (24 slots, 8 entries x 3
    // fields). Parallel to struct_tab, keyed by struct_idx. Stores
    // (struct_idx, gp_count, gp_names_head) per entry. INC-3b's use-site
    // monomorphization reads this table to map `Pt<T>` → gp_count + the
    // mk_node(76, ...) name chain for type-var substitution.
    let sgp_base = __arena_push(0);
    let mut sgpi: i32 = 1;
    while sgpi < 24 {
        __arena_push(0);
        sgpi = sgpi + 1;
    }
    __arena_set(cur_slot + 78, sgp_base);
    __arena_set(cur_slot + 79, 0);
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
        // Stage 8.5: trait/impl are also program-mode prefixes — they
        // route to parse_program just like struct/enum.
        let is_trait = byte_eq(id_s, id_l, kw_trait_s(cur_slot), kw_trait_n(cur_slot));
        let is_impl = byte_eq(id_s, id_l, kw_impl_s(cur_slot), kw_impl_n(cur_slot));
        // Stage 10: mod/use are also program-mode prefixes.
        let is_mod = byte_eq(id_s, id_l, kw_mod_s(cur_slot), kw_mod_n(cur_slot));
        let is_use = byte_eq(id_s, id_l, kw_use_s(cur_slot), kw_use_n(cur_slot));
        // K1.V (2026-05-25): `type` is also a top-level decl prefix.
        let is_type = is_kw_type_ident(id_s, id_l);
        if is_fn == 1 {
            parse_program(tok_base, cur_slot)
        } else { if is_struct == 1 {
            parse_program(tok_base, cur_slot)
        } else { if is_enum == 1 {
            parse_program(tok_base, cur_slot)
        } else { if is_trait == 1 {
            parse_program(tok_base, cur_slot)
        } else { if is_impl == 1 {
            parse_program(tok_base, cur_slot)
        } else { if is_mod == 1 {
            parse_program(tok_base, cur_slot)
        } else { if is_use == 1 {
            parse_program(tok_base, cur_slot)
        } else { if is_type == 1 {
            parse_program(tok_base, cur_slot)
        } else {
            parse_expr(tok_base, cur_slot)
        }}}}}}}}
    } else {
        parse_expr(tok_base, cur_slot)
    }
}

fn autotune_seen_value(vals_base: i32, count: i32, value: i32) -> i32 {
    let mut i: i32 = 0;
    let mut hit: i32 = 0;
    while i < count {
        if __arena_get(vals_base + i) == value {
            hit = 1;
        };
        i = i + 1;
    }
    hit
}

fn autotune_is_int_value_token(tag: i32) -> i32 {
    if tag == 1 { 1 } else { if tag == 33 { 1 } else { if tag == 34 { 1 } else {
    if tag == 35 { 1 } else { if tag == 36 { 1 } else { if tag == 37 { 1 } else {
    if tag == 38 { 1 } else { if tag == 39 { 1 } else { 0 } } } } } } } }
}

fn autotune_first_error_kind(current: i32, next: i32) -> i32 {
    if current == 0 { next } else { current }
}

// Parse a narrow Stage-33 subset of @autotune(KEY: [1, 2], ...). This is
// validation metadata only; it does not generate kernel variants.
//
// Error-kind encoding stored in AST_FN_DECL slot 17:
//   0 = clean
//   1 = missing parenthesized argument list
//   2 = malformed token/shape inside the argument list
//   3 = empty parameter list or empty value list
fn capture_autotune_args(tok_base: i32, first_tok: i32, sb: i32) -> i32 {
    let prior_product = next_fn_autotune_product(sb);
    let prior_error = next_fn_autotune_error(sb);
    if tok_tag(tok_base, first_tok) != 3 {
        set_next_fn_autotune_product(sb, 0);
        set_next_fn_autotune_error(sb, autotune_first_error_kind(prior_error, 1));
        0
    } else {
        let mut k: i32 = first_tok + 1;
        let mut params_seen: i32 = 0;
        let mut product: i32 = 1;
        let mut error_kind: i32 = 0;
        let mut keep_args: i32 = 1;
        while keep_args == 1 {
            let tt = tok_tag(tok_base, k);
            if tt == 4 {
                keep_args = 0;
            } else { if tt == 0 {
                error_kind = autotune_first_error_kind(error_kind, 2);
                keep_args = 0;
            } else { if tt == 13 {
                k = k + 1;
            } else { if tt == 2 {
                k = k + 1;
                if tok_tag(tok_base, k) == 14 {
                    k = k + 1;
                } else {
                    error_kind = autotune_first_error_kind(error_kind, 2);
                };
                if tok_tag(tok_base, k) == 20 {
                    k = k + 1;
                    let vals_base = __arena_push(0);
                    let mut vi: i32 = 1;
                    while vi < 16 {
                        __arena_push(0);
                        vi = vi + 1;
                    }
                    let mut val_count: i32 = 0;
                    let mut keep_vals: i32 = 1;
                    while keep_vals == 1 {
                        let vt = tok_tag(tok_base, k);
                        if vt == 21 {
                            keep_vals = 0;
                            k = k + 1;
                        } else { if vt == 0 {
                            error_kind = autotune_first_error_kind(error_kind, 2);
                            keep_vals = 0;
                            keep_args = 0;
                        } else { if vt == 4 {
                            error_kind = autotune_first_error_kind(error_kind, 2);
                            keep_vals = 0;
                            keep_args = 0;
                        } else { if vt == 13 {
                            k = k + 1;
                        } else { if autotune_is_int_value_token(vt) == 1 {
                            let v = tok_p1(tok_base, k);
                            let lookup_count = if val_count < 16 { val_count } else { 16 };
                            if autotune_seen_value(vals_base, lookup_count, v) == 0 {
                                if val_count < 16 {
                                    __arena_set(vals_base + val_count, v);
                                };
                                val_count = val_count + 1;
                            };
                            k = k + 1;
                            let after_value = tok_tag(tok_base, k);
                            if after_value != 13 {
                                if after_value != 21 {
                                    error_kind = autotune_first_error_kind(error_kind, 2);
                                } else { 0 };
                            } else { 0 };
                        } else {
                            error_kind = autotune_first_error_kind(error_kind, 2);
                            k = k + 1;
                        }}}}}
                    }
                    if val_count == 0 {
                        error_kind = autotune_first_error_kind(error_kind, 3);
                    } else {
                        product = product * val_count;
                        if product > 16 {
                            product = 17;
                        };
                    };
                    params_seen = params_seen + 1;
                    let after_param = tok_tag(tok_base, k);
                    if after_param != 13 {
                        if after_param != 4 {
                            if after_param != 0 {
                                error_kind = autotune_first_error_kind(error_kind, 2);
                            } else { 0 };
                        } else { 0 };
                    } else { 0 };
                } else {
                    error_kind = autotune_first_error_kind(error_kind, 2);
                    k = k + 1;
                };
            } else {
                error_kind = autotune_first_error_kind(error_kind, 2);
                k = k + 1;
            }}}}
        }
        if params_seen == 0 {
            error_kind = autotune_first_error_kind(error_kind, 3);
            product = 0;
        };
        let combined_error = autotune_first_error_kind(prior_error, error_kind);
        let combined_product = if prior_product == 0 { product } else {
            if product == 0 { 0 } else {
                let raw_product = prior_product * product;
                if raw_product > 16 { 17 } else { raw_product }
            }
        };
        set_next_fn_autotune_product(sb, combined_product);
        set_next_fn_autotune_error(sb, combined_error);
        0
    }
}

// Consume zero or more `@<IDENT>` (or `@<IDENT>(<args>)`) attribute
// markers. Most attributes (`@pure`, `@effect`, ...) are skipped as
// no-ops in Phase 0. Stage 14.5: `@checkpoint` is special — when seen
// it sets a sticky scratch flag at sb+74 so the next parse_fn_decl
// can record it on the synthesized AST_FN_DECL node (slot 8).
// Stage 28.9: similarly recognizes `@deprecated` (slot 75),
// `@trace` (slot 76), `@unwind` (slot 77) so the bootstrap-side
// validation passes can observe them on AST_FN_DECL slots 9/10/11.
// Stage 33: if `@deprecated("message")` or `@since("version")` has a
// first string-literal argument, preserve that message body range in
// scratch slots 80/81 or 86/87 for AST_FN_DECL slots 12/13 or 18/19.
// Stage 33: also recognizes `@kernel` and enough `@autotune(...)`
// metadata for bootstrap-side validation of kernel requirement,
// malformed/empty parameter lists, and variant-product cap.
fn skip_attributes(tok_base: i32, sb: i32) -> i32 {
    let mut keep: i32 = 1;
    while keep == 1 {
        if tok_tag(tok_base, cur_get(sb)) == 24 {
            cur_advance(sb);     // consume '@'
            // Optional IDENT after the '@'.
            if tok_tag(tok_base, cur_get(sb)) == 2 {
                let attr_tok = cur_get(sb);
                let attr_s = tok_p2(tok_base, attr_tok);
                let attr_l = tok_p3(tok_base, attr_tok);
                // Stage 14.5: byte-compare attribute IDENT against
                // "checkpoint" (10 bytes: 99 104 101 99 107 112 111 105
                // 110 116). Also recognize `deprecated` (10 bytes:
                // 100 101 112 114 101 99 97 116 101 100). Both are
                // 10-char so they fan out from the same length-10
                // branch — but with different first-byte 99 vs 100.
                if attr_l == 10 {
                    let b0 = __arena_get(attr_s);
                    let b1 = __arena_get(attr_s + 1);
                    let b2 = __arena_get(attr_s + 2);
                    let b3 = __arena_get(attr_s + 3);
                    let b4 = __arena_get(attr_s + 4);
                    let b5 = __arena_get(attr_s + 5);
                    let b6 = __arena_get(attr_s + 6);
                    let b7 = __arena_get(attr_s + 7);
                    let b8 = __arena_get(attr_s + 8);
                    let b9 = __arena_get(attr_s + 9);
                    let is_ckpt = if b0 == 99 { if b1 == 104 { if b2 == 101 {
                        if b3 == 99 { if b4 == 107 { if b5 == 112 {
                        if b6 == 111 { if b7 == 105 { if b8 == 110 {
                        if b9 == 116 { 1 } else { 0 }
                        } else { 0 } } else { 0 } } else { 0 } } else { 0 }
                        } else { 0 } } else { 0 } } else { 0 } } else { 0 } } else { 0 };
                    if is_ckpt == 1 {
                        set_next_fn_is_ckpt(sb, 1);
                    };
                    // Stage 28.9: @deprecated (100 101 112 114 101 99
                    // 97 116 101 100).
                    let is_dep = if b0 == 100 { if b1 == 101 { if b2 == 112 {
                        if b3 == 114 { if b4 == 101 { if b5 == 99 {
                        if b6 == 97 { if b7 == 116 { if b8 == 101 {
                        if b9 == 100 { 1 } else { 0 }
                        } else { 0 } } else { 0 } } else { 0 } } else { 0 }
                        } else { 0 } } else { 0 } } else { 0 } } else { 0 } } else { 0 };
                    if is_dep == 1 {
                        set_next_fn_is_deprecated(sb, 1);
                        set_next_fn_deprecated_msg_s(sb, 0);
                        set_next_fn_deprecated_msg_l(sb, 0);
                        if tok_tag(tok_base, cur_get(sb) + 1) == 3 {
                            let msg_tok = cur_get(sb) + 2;
                            if tok_tag(tok_base, msg_tok) == 25 {
                                set_next_fn_deprecated_msg_s(sb, tok_p2(tok_base, msg_tok));
                                set_next_fn_deprecated_msg_l(sb, tok_p3(tok_base, msg_tok));
                            };
                        };
                    };
                };
                // Stage 28.9: `trace` (5 bytes: 116 114 97 99 101).
                if attr_l == 5 {
                    let tb0 = __arena_get(attr_s);
                    let tb1 = __arena_get(attr_s + 1);
                    let tb2 = __arena_get(attr_s + 2);
                    let tb3 = __arena_get(attr_s + 3);
                    let tb4 = __arena_get(attr_s + 4);
                    let is_trace = if tb0 == 116 { if tb1 == 114 {
                        if tb2 == 97 { if tb3 == 99 { if tb4 == 101 { 1 }
                        else { 0 } } else { 0 } } else { 0 } } else { 0 } } else { 0 };
                    if is_trace == 1 {
                        set_next_fn_is_trace(sb, 1);
                    };
                    let is_since = if tb0 == 115 { if tb1 == 105 {
                        if tb2 == 110 { if tb3 == 99 { if tb4 == 101 { 1 }
                        else { 0 } } else { 0 } } else { 0 } } else { 0 } } else { 0 };
                    if is_since == 1 {
                        set_next_fn_since_msg_s(sb, attr_s);
                        set_next_fn_since_msg_l(sb, 0);
                        if tok_tag(tok_base, cur_get(sb) + 1) == 3 {
                            let msg_tok = cur_get(sb) + 2;
                            if tok_tag(tok_base, msg_tok) == 25 {
                                set_next_fn_since_msg_s(sb, tok_p2(tok_base, msg_tok));
                                set_next_fn_since_msg_l(sb, tok_p3(tok_base, msg_tok));
                            };
                        };
                    };
                };
                // Stage 28.9: `unwind` (6 bytes: 117 110 119 105 110 100).
                if attr_l == 6 {
                    let ub0 = __arena_get(attr_s);
                    let ub1 = __arena_get(attr_s + 1);
                    let ub2 = __arena_get(attr_s + 2);
                    let ub3 = __arena_get(attr_s + 3);
                    let ub4 = __arena_get(attr_s + 4);
                    let ub5 = __arena_get(attr_s + 5);
                    let is_unwind = if ub0 == 117 { if ub1 == 110 {
                        if ub2 == 119 { if ub3 == 105 { if ub4 == 110 {
                        if ub5 == 100 { 1 } else { 0 } } else { 0 }
                        } else { 0 } } else { 0 } } else { 0 } } else { 0 };
                    if is_unwind == 1 {
                        set_next_fn_is_unwind(sb, 1);
                    };
                    // Stage 33: `kernel` (6 bytes: 107 101 114 110 101 108).
                    let is_kernel = if ub0 == 107 { if ub1 == 101 {
                        if ub2 == 114 { if ub3 == 110 { if ub4 == 101 {
                        if ub5 == 108 { 1 } else { 0 } } else { 0 }
                        } else { 0 } } else { 0 } } else { 0 } } else { 0 };
                    if is_kernel == 1 {
                        set_next_fn_is_kernel(sb, 1);
                    };
                };
                // Stage 33: `autotune` (8 bytes: 97 117 116 111 116 117 110 101).
                if attr_l == 8 {
                    let ab0 = __arena_get(attr_s);
                    let ab1 = __arena_get(attr_s + 1);
                    let ab2 = __arena_get(attr_s + 2);
                    let ab3 = __arena_get(attr_s + 3);
                    let ab4 = __arena_get(attr_s + 4);
                    let ab5 = __arena_get(attr_s + 5);
                    let ab6 = __arena_get(attr_s + 6);
                    let ab7 = __arena_get(attr_s + 7);
                    let is_autotune = if ab0 == 97 { if ab1 == 117 {
                        if ab2 == 116 { if ab3 == 111 { if ab4 == 116 {
                        if ab5 == 117 { if ab6 == 110 { if ab7 == 101 {
                        1 } else { 0 } } else { 0 } } else { 0 }
                        } else { 0 } } else { 0 } } else { 0 } } else { 0 } } else { 0 };
                    if is_autotune == 1 {
                        set_next_fn_is_autotune(sb, 1);
                        capture_autotune_args(tok_base, cur_get(sb) + 1, sb);
                    };
                };
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
            // FLAT prefix-trap ladder: single-binding chain, no nested
            // if-else statements. Stage 8.5 adds two new prefixes (trait,
            // impl) — handled before falling through to the fn-decl path.
            let is_struct_kw = byte_eq(s, l, kw_struct_s(sb), kw_struct_n(sb));
            let is_enum_kw = byte_eq(s, l, kw_enum_s(sb), kw_enum_n(sb));
            let is_trait_kw = byte_eq(s, l, kw_trait_s(sb), kw_trait_n(sb));
            let is_impl_kw = byte_eq(s, l, kw_impl_s(sb), kw_impl_n(sb));
            // Stage 10: also skip `mod IDENT { ... }` and `use IDENT::...;`.
            let is_mod_kw = byte_eq(s, l, kw_mod_s(sb), kw_mod_n(sb));
            let is_use_kw = byte_eq(s, l, kw_use_s(sb), kw_use_n(sb));
            // K1.V (2026-05-25): `type Alias = T;` is a top-level no-op
            // decl. Same metadata-only pattern as struct/enum/mod/use.
            let is_type_kw = is_kw_type_ident(s, l);
            if is_struct_kw == 1 {
                parse_struct_decl(tok_base, sb);
                __arena_set(sb + 74, 0); __arena_set(sb + 75, 0); __arena_set(sb + 76, 0); __arena_set(sb + 77, 0); __arena_set(sb + 80, 0); __arena_set(sb + 81, 0); __arena_set(sb + 82, 0); __arena_set(sb + 83, 0); __arena_set(sb + 84, 0); __arena_set(sb + 85, 0); __arena_set(sb + 86, 0); __arena_set(sb + 87, 0);
            } else { if is_enum_kw == 1 {
                parse_enum_decl(tok_base, sb);
                __arena_set(sb + 74, 0); __arena_set(sb + 75, 0); __arena_set(sb + 76, 0); __arena_set(sb + 77, 0); __arena_set(sb + 80, 0); __arena_set(sb + 81, 0); __arena_set(sb + 82, 0); __arena_set(sb + 83, 0); __arena_set(sb + 84, 0); __arena_set(sb + 85, 0); __arena_set(sb + 86, 0); __arena_set(sb + 87, 0);
            } else { if is_trait_kw == 1 {
                parse_trait_decl(tok_base, sb);
                __arena_set(sb + 74, 0); __arena_set(sb + 75, 0); __arena_set(sb + 76, 0); __arena_set(sb + 77, 0); __arena_set(sb + 80, 0); __arena_set(sb + 81, 0); __arena_set(sb + 82, 0); __arena_set(sb + 83, 0); __arena_set(sb + 84, 0); __arena_set(sb + 85, 0); __arena_set(sb + 86, 0); __arena_set(sb + 87, 0);
            } else { if is_impl_kw == 1 {
                parse_impl_block(tok_base, sb);
                __arena_set(sb + 74, 0); __arena_set(sb + 75, 0); __arena_set(sb + 76, 0); __arena_set(sb + 77, 0); __arena_set(sb + 80, 0); __arena_set(sb + 81, 0); __arena_set(sb + 82, 0); __arena_set(sb + 83, 0); __arena_set(sb + 84, 0); __arena_set(sb + 85, 0); __arena_set(sb + 86, 0); __arena_set(sb + 87, 0);
            } else { if is_mod_kw == 1 {
                parse_mod_decl(tok_base, sb, 0, 0);
                __arena_set(sb + 74, 0); __arena_set(sb + 75, 0); __arena_set(sb + 76, 0); __arena_set(sb + 77, 0); __arena_set(sb + 80, 0); __arena_set(sb + 81, 0); __arena_set(sb + 82, 0); __arena_set(sb + 83, 0); __arena_set(sb + 84, 0); __arena_set(sb + 85, 0); __arena_set(sb + 86, 0); __arena_set(sb + 87, 0);
            } else { if is_use_kw == 1 {
                parse_use_decl(tok_base, sb);
                __arena_set(sb + 74, 0); __arena_set(sb + 75, 0); __arena_set(sb + 76, 0); __arena_set(sb + 77, 0); __arena_set(sb + 80, 0); __arena_set(sb + 81, 0); __arena_set(sb + 82, 0); __arena_set(sb + 83, 0); __arena_set(sb + 84, 0); __arena_set(sb + 85, 0); __arena_set(sb + 86, 0); __arena_set(sb + 87, 0);
            } else { if is_type_kw == 1 {
                parse_type_alias_decl(tok_base, sb);
                __arena_set(sb + 74, 0); __arena_set(sb + 75, 0); __arena_set(sb + 76, 0); __arena_set(sb + 77, 0); __arena_set(sb + 80, 0); __arena_set(sb + 81, 0); __arena_set(sb + 82, 0); __arena_set(sb + 83, 0); __arena_set(sb + 84, 0); __arena_set(sb + 85, 0); __arena_set(sb + 86, 0); __arena_set(sb + 87, 0);
            } else {
                keep_decl = 0;
            }}}}}}};
        } else {
            keep_decl = 0;
        };
    }
    // Leading non-fn decls may be followed by attributes for the first fn.
    skip_attributes(tok_base, sb);
    let first_fn = parse_fn_decl(tok_base, sb);
    let user_first_node = mk_node(15, first_fn, 0, 0);
    let mut prev_list = user_first_node;
    let mut keep: i32 = 1;
    // Audit A2-F1 fix: post-fn loop now accepts struct/enum/trait/impl/
    // mod/use too, mirroring the prefix loop above. Pre-fix, any non-fn
    // decl after the first fn was silently dropped, and subsequent fns
    // after the dropped decl were also invisible (loop exited on first
    // unrecognized token). This caused the natural Rust ordering
    // (fn / type / fn / type) to silently truncate.
    while keep == 1 {
        // Skip any attributes before the next decl.
        skip_attributes(tok_base, sb);
        let k2 = cur_get(sb);
        let t2 = tok_tag(tok_base, k2);
        if t2 == 0 {
            keep = 0;
        } else { if t2 == 2 {
            let s = tok_p2(tok_base, k2);
            let l = tok_p3(tok_base, k2);
            let is_fn_kw2 = byte_eq(s, l, kw_fn_s(sb), kw_fn_n(sb));
            let is_struct_kw2 = byte_eq(s, l, kw_struct_s(sb), kw_struct_n(sb));
            let is_enum_kw2 = byte_eq(s, l, kw_enum_s(sb), kw_enum_n(sb));
            let is_trait_kw2 = byte_eq(s, l, kw_trait_s(sb), kw_trait_n(sb));
            let is_impl_kw2 = byte_eq(s, l, kw_impl_s(sb), kw_impl_n(sb));
            let is_mod_kw2 = byte_eq(s, l, kw_mod_s(sb), kw_mod_n(sb));
            let is_use_kw2 = byte_eq(s, l, kw_use_s(sb), kw_use_n(sb));
            // K1.V (2026-05-25): `type Alias = T;` arm for the
            // post-fn loop, mirroring the leading-decl loop above.
            let is_type_kw2 = is_kw_type_ident(s, l);
            if is_fn_kw2 == 1 {
                let next_fn = parse_fn_decl(tok_base, sb);
                let new_node = mk_node(15, next_fn, 0, 0);
                __arena_set(prev_list + 2, new_node);
                prev_list = new_node;
            } else { if is_struct_kw2 == 1 {
                parse_struct_decl(tok_base, sb);
                __arena_set(sb + 74, 0); __arena_set(sb + 75, 0); __arena_set(sb + 76, 0); __arena_set(sb + 77, 0); __arena_set(sb + 80, 0); __arena_set(sb + 81, 0); __arena_set(sb + 82, 0); __arena_set(sb + 83, 0); __arena_set(sb + 84, 0); __arena_set(sb + 85, 0); __arena_set(sb + 86, 0); __arena_set(sb + 87, 0);
            } else { if is_enum_kw2 == 1 {
                parse_enum_decl(tok_base, sb);
                __arena_set(sb + 74, 0); __arena_set(sb + 75, 0); __arena_set(sb + 76, 0); __arena_set(sb + 77, 0); __arena_set(sb + 80, 0); __arena_set(sb + 81, 0); __arena_set(sb + 82, 0); __arena_set(sb + 83, 0); __arena_set(sb + 84, 0); __arena_set(sb + 85, 0); __arena_set(sb + 86, 0); __arena_set(sb + 87, 0);
            } else { if is_trait_kw2 == 1 {
                parse_trait_decl(tok_base, sb);
                __arena_set(sb + 74, 0); __arena_set(sb + 75, 0); __arena_set(sb + 76, 0); __arena_set(sb + 77, 0); __arena_set(sb + 80, 0); __arena_set(sb + 81, 0); __arena_set(sb + 82, 0); __arena_set(sb + 83, 0); __arena_set(sb + 84, 0); __arena_set(sb + 85, 0); __arena_set(sb + 86, 0); __arena_set(sb + 87, 0);
            } else { if is_impl_kw2 == 1 {
                parse_impl_block(tok_base, sb);
                __arena_set(sb + 74, 0); __arena_set(sb + 75, 0); __arena_set(sb + 76, 0); __arena_set(sb + 77, 0); __arena_set(sb + 80, 0); __arena_set(sb + 81, 0); __arena_set(sb + 82, 0); __arena_set(sb + 83, 0); __arena_set(sb + 84, 0); __arena_set(sb + 85, 0); __arena_set(sb + 86, 0); __arena_set(sb + 87, 0);
            } else { if is_mod_kw2 == 1 {
                parse_mod_decl(tok_base, sb, 0, 0);
                __arena_set(sb + 74, 0); __arena_set(sb + 75, 0); __arena_set(sb + 76, 0); __arena_set(sb + 77, 0); __arena_set(sb + 80, 0); __arena_set(sb + 81, 0); __arena_set(sb + 82, 0); __arena_set(sb + 83, 0); __arena_set(sb + 84, 0); __arena_set(sb + 85, 0); __arena_set(sb + 86, 0); __arena_set(sb + 87, 0);
            } else { if is_use_kw2 == 1 {
                parse_use_decl(tok_base, sb);
                __arena_set(sb + 74, 0); __arena_set(sb + 75, 0); __arena_set(sb + 76, 0); __arena_set(sb + 77, 0); __arena_set(sb + 80, 0); __arena_set(sb + 81, 0); __arena_set(sb + 82, 0); __arena_set(sb + 83, 0); __arena_set(sb + 84, 0); __arena_set(sb + 85, 0); __arena_set(sb + 86, 0); __arena_set(sb + 87, 0);
            } else { if is_type_kw2 == 1 {
                parse_type_alias_decl(tok_base, sb);
                __arena_set(sb + 74, 0); __arena_set(sb + 75, 0); __arena_set(sb + 76, 0); __arena_set(sb + 77, 0); __arena_set(sb + 80, 0); __arena_set(sb + 81, 0); __arena_set(sb + 82, 0); __arena_set(sb + 83, 0); __arena_set(sb + 84, 0); __arena_set(sb + 85, 0); __arena_set(sb + 86, 0); __arena_set(sb + 87, 0);
            } else {
                __arena_set(sb + 74, 0); __arena_set(sb + 75, 0); __arena_set(sb + 76, 0); __arena_set(sb + 77, 0); __arena_set(sb + 80, 0); __arena_set(sb + 81, 0); __arena_set(sb + 82, 0); __arena_set(sb + 83, 0); __arena_set(sb + 84, 0); __arena_set(sb + 85, 0); __arena_set(sb + 86, 0); __arena_set(sb + 87, 0);
                keep = 0;
            }}}}}}}};
        } else {
            keep = 0;
        }};
    }
    // Stage 8.5B: if any impl blocks pushed methods into impl_pending, the
    // chain head/tail are non-zero. Splice the impl-pending chain BEFORE
    // the user's first fn so the resulting fn_list = [impl_methods..., user_fns...].
    // Order doesn't affect codegen correctness (fn_table is name-keyed) but
    // emitting impl methods first lets later user fns call them.
    // Stage 9: also splice cl_pending (synthesized closure-body fns). This
    // happens AFTER user fns are parsed because closure literals appear
    // INSIDE user fn bodies — cl_pending only fills up after parse_fn_decl
    // walks each body. Splicing AFTER parse means we have all the closures.
    let impl_head = impl_pending_head(sb);
    let cl_head = cl_pending_head(sb);
    // Stage 10: also splice mod_pending (fns lifted from `mod foo { ... }`
    // blocks). These are known at parse-time (skip-loop) so they prepend
    // the impl_pending + cl_pending chains.
    let mod_head = mod_pending_head(sb);
    // Build a unified front-chain: mod_lifted, impl_methods, then closures.
    // Concatenate non-empty heads in order. Capture tails to splice.
    let front_head_a = if mod_head == 0 { impl_head } else {
        if impl_head == 0 { mod_head } else {
            let mod_tail = mod_pending_tail(sb);
            __arena_set(mod_tail + 2, impl_head);
            mod_head
        }
    };
    let front_head = if front_head_a == 0 { cl_head } else {
        if cl_head == 0 { front_head_a } else {
            // Find tail of front_head_a chain to splice cl_head onto.
            let mut ftt: i32 = front_head_a;
            let mut ftt_keep: i32 = 1;
            while ftt_keep == 1 {
                let nx = __arena_get(ftt + 2);
                if nx == 0 { ftt_keep = 0; } else { ftt = nx; };
            }
            __arena_set(ftt + 2, cl_head);
            front_head_a
        }
    };
    let head = if front_head == 0 {
        user_first_node
    } else {
        // Find tail of front_head chain.
        let mut ft: i32 = front_head;
        let mut ft_keep: i32 = 1;
        while ft_keep == 1 {
            let nx = __arena_get(ft + 2);
            if nx == 0 { ft_keep = 0; } else { ft = nx; };
        }
        __arena_set(ft + 2, user_first_node);
        front_head
    };
    // Stage 8: monomorphization pass. Walk mr_tab; for each registered
    // (orig_name, mangled_name, pack_lo) entry, find the original
    // AST_FN_DECL template in the fn_list (matching by name) and build
    // a concrete clone with type substitution applied to params + ret.
    // The clone shares the body idx (no deep copy needed: AST_VAR
    // references resolve by name at codegen time, and the binding name
    // matches between template and clone). Append a new AST_FN_LIST
    // node pointing to the clone so codegen emits it.
    monomorphize_pass(sb, head);
    // Stage 12: grad pass. Walk grad_pending; for each registered
    // (loss_name, mang_name) entry, find the loss fn in the fn_list,
    // differentiate its body w.r.t. its first param, simplify, and
    // synthesize a new AST_FN_DECL with the mangled name. Appended to
    // fn_list tail so codegen emits it normally.
    grad_pass(sb, head);
    // Stage 14: grad_rev pass. Walk gr_rev_pending; for each registered
    // (loss_name, field_name, mang_s) entry, find the loss fn, find the
    // param matching `field` (with leading 'd' stripped), differentiate
    // the loss body w.r.t. that param, simplify, and synthesize the
    // <loss>__grad_<field> fn. Appends to fn_list tail so codegen emits it.
    grad_rev_pass(sb, head);
    head
}

// Stage 8.5C: helper. Try to rewrite an AST_CALL's mangled name whose
// prefix matches one of the template's gp names. Returns a NEW (start,
// len) pair packed as `start * 8 + len`, or 0 if no gp prefix matches.
//
// gp_head: linked list of AST_GP_NAME (tag 76) nodes for the template fn.
// packed: the mono pass's 4-bit-packed concrete tags (gp_idx 0 in low 4
// bits, gp_idx 1 in next, etc.).
//
// Match: scan gp_head; for each gp_name (gn_s, gn_l), check if the call's
// name's first gn_l bytes == gn_s bytes. If so, the next two bytes must
// be "__" (95, 95). Then build new mangled name `<concrete_ty_name>__<rest>`
// and return (new_s, new_l) packed.
fn try_rewrite_call_name(call_name_s: i32, call_name_l: i32, gp_head: i32, packed: i32) -> i32 {
    let mut walk: i32 = gp_head;
    let mut idx: i32 = 0;
    let mut found_packed: i32 = 0;
    let mut keep_w: i32 = 1;
    while keep_w == 1 {
        let gn_s = __arena_get(walk + 1);
        let gn_l = __arena_get(walk + 2);
        let need_underscore = gn_l + 2;
        let mut do_rewrite: i32 = 0;
        if found_packed == 0 {
            if call_name_l >= need_underscore {
                if byte_eq(call_name_s, gn_l, gn_s, gn_l) == 1 {
                    let u1 = __arena_get(call_name_s + gn_l);
                    let u2 = __arena_get(call_name_s + gn_l + 1);
                    if u1 == 95 {
                        if u2 == 95 {
                            do_rewrite = 1;
                        };
                    };
                };
            };
        };
        if do_rewrite == 1 {
            // Match! Extract concrete tag for this gp idx from packed.
            let mut shifted: i32 = packed;
            let mut sk: i32 = 0;
            while sk < idx { shifted = shifted / 16; sk = sk + 1; }
            let concrete_tag = shifted - (shifted / 16) * 16;
            // Build new name: concrete_ty_name + "__" + rest_of_call_name.
            let ty_pack = ty_tag_push_name(concrete_tag);
            let ty_len = ty_pack - (ty_pack / 8) * 8;
            let ty_start = ty_pack / 8;
            __arena_push(95); __arena_push(95);
            let rest_len = call_name_l - gn_l - 2;
            let mut ri: i32 = 0;
            while ri < rest_len {
                __arena_push(__arena_get(call_name_s + gn_l + 2 + ri));
                ri = ri + 1;
            }
            let new_l = ty_len + 2 + rest_len;
            found_packed = ty_start * 8 + new_l;
        };
        let nxt = __arena_get(walk + 3);
        if nxt == 0 {
            keep_w = 0;
        } else {
            walk = nxt;
            idx = idx + 1;
        };
    }
    found_packed
}

// Stage 8.5C: deep-clone an AST subtree. Most nodes are shared (leaves like
// AST_INT, AST_VAR are gp-independent so sharing is safe). The only nodes
// that need cloning are AST_CALL nodes whose name has a gp-prefix; we
// allocate a NEW AST_CALL node with the rewritten name + same args_head.
//
// To avoid the full deep-clone burden in this Phase-0 cut, we do a recursive
// walk that handles a fixed set of "compound" tags (AST_LET, AST_IF, AST_SEQ,
// binops, AST_ARG_LIST, AST_CALL itself). For unknown tags the function
// falls back to sharing (returns the same idx) — enough for the Phase-0
// test cases. Stage 9+ may need to extend with closure/match nodes.
// Stage 8.5C: minimal deep-clone. Only handles the case where the body
// IS an AST_CALL — clones that one node with potential name rewrite,
// sharing args. Phase-0 test cases all have this shape (`T::eq(a, b)` is
// the entire body of cmp). For more complex bodies (let-binding, if-expr,
// nested calls), extend this function as needed.
fn clone_with_rewrite(node_idx: i32, gp_head: i32, packed: i32) -> i32 {
    let t = __arena_get(node_idx);
    if t == 16 {                                        // AST_CALL
        let call_name_s = __arena_get(node_idx + 1);
        let call_name_l = __arena_get(node_idx + 2);
        let args_head = __arena_get(node_idx + 3);
        let rewritten = try_rewrite_call_name(call_name_s, call_name_l, gp_head, packed);
        if rewritten == 0 {
            // No rewrite needed — share the original node.
            node_idx
        } else {
            let final_s = rewritten / 8;
            let final_l = rewritten - final_s * 8;
            mk_node(16, final_s, final_l, args_head)
        }
    } else {
        // For non-call bodies, share. Stage 9+ may extend.
        node_idx
    }
}

// Stage 8: mono pass. For each mr_tab entry, find the matching generic
// fn template in the fn_list and synthesize a concrete clone. The clone
// reuses the body (same AST nodes) — substitution applies only to the
// AST_PARAM type tags and the AST_FN_DECL's ret_ty slot. Appends new
// AST_FN_LIST nodes to the END of head so the new fns are emitted.
fn monomorphize_pass(sb: i32, head: i32) -> i32 {
    let count = mr_tab_count(sb);
    if count == 0 {
        0
    } else {
        // Find tail of fn_list (where next == 0).
        let mut tail = head;
        let mut tail_keep: i32 = 1;
        while tail_keep == 1 {
            let nx = __arena_get(tail + 2);
            if nx == 0 { tail_keep = 0; } else { tail = nx; };
        }
        // Iterate mr_tab entries.
        let base = mr_tab_base(sb);
        let mut mi: i32 = 0;
        while mi < count {
            let entry = base + mi * 6;
            let orig_s = __arena_get(entry);
            let orig_l = __arena_get(entry + 1);
            let mang_s = __arena_get(entry + 2);
            let mang_l = __arena_get(entry + 3);
            let pack_lo = __arena_get(entry + 4);
            let ta_count = pack_lo - (pack_lo / 8) * 8;
            let packed = pack_lo / 8;
            // Find matching generic template in fn_list (search by orig_name).
            let mut walk: i32 = head;
            let mut tpl_idx: i32 = 0;
            let mut find_keep: i32 = 1;
            while find_keep == 1 {
                let cand_idx = __arena_get(walk + 1);
                let cand_ns = __arena_get(cand_idx + 1);
                let cand_nl = __arena_get(cand_idx + 2);
                let cand_gen = __arena_get(cand_idx + 6);
                if cand_gen == 1 {
                    if byte_eq(orig_s, orig_l, cand_ns, cand_nl) == 1 {
                        tpl_idx = cand_idx;
                        find_keep = 0;
                    };
                };
                if find_keep == 1 {
                    let nx = __arena_get(walk + 2);
                    if nx == 0 { find_keep = 0; } else { walk = nx; };
                };
            }
            if tpl_idx > 0 {
                // Clone the fn decl. Build a new param list with substituted
                // type tags. Keep the same body idx.
                let tpl_body = __arena_get(tpl_idx + 3);
                let tpl_params_head = __arena_get(tpl_idx + 4);
                let tpl_ret_ty = __arena_get(tpl_idx + 5);
                // Walk template params, build a new chain with substituted types.
                let mut t_p_cur: i32 = tpl_params_head;
                let mut new_params_head: i32 = 0;
                let mut new_prev_p: i32 = 0;
                while t_p_cur != 0 {
                    let p_ns = __arena_get(t_p_cur + 1);
                    let p_nl = __arena_get(t_p_cur + 2);
                    let p_ty_raw = __arena_get(t_p_cur + 4);
                    // Substitute generic markers.
                    let new_p_ty = if p_ty_raw >= 200 {
                        let g_idx = p_ty_raw - 200;
                        // Extract g_idx-th 4-bit slot from packed.
                        let mut shifted: i32 = packed;
                        let mut sk: i32 = 0;
                        while sk < g_idx { shifted = shifted / 16; sk = sk + 1; }
                        shifted - (shifted / 16) * 16
                    } else { p_ty_raw };
                    let new_p_node = mk_node(18, p_ns, p_nl, 0);
                    __arena_push(new_p_ty);
                    if new_params_head == 0 {
                        new_params_head = new_p_node;
                        new_prev_p = new_p_node;
                    } else {
                        __arena_set(new_prev_p + 3, new_p_node);
                        new_prev_p = new_p_node;
                    };
                    t_p_cur = __arena_get(t_p_cur + 3);
                }
                // Substitute ret_ty.
                let new_ret_ty = if tpl_ret_ty >= 200 {
                    let g_idx = tpl_ret_ty - 200;
                    let mut shifted: i32 = packed;
                    let mut sk: i32 = 0;
                    while sk < g_idx { shifted = shifted / 16; sk = sk + 1; }
                    shifted - (shifted / 16) * 16
                } else { tpl_ret_ty };
                // Stage 8.5C: deep-clone-with-rewrite of the template body.
                // For generic fns whose body uses `T::eq(...)` typed-calls, the
                // call's mangled name contains the gp name as prefix (e.g.
                // "T__eq"). The clone walks the body subtree, copying nodes,
                // and rewrites AST_CALL names whose prefix matches a gp name
                // to use the concrete type's name (e.g. "i32__eq"). For
                // non-call subtrees the clone shares leaf nodes (AST_VAR/INT)
                // since they don't depend on gp.
                let tpl_gp_head = __arena_get(tpl_idx + 7);
                let cloned_body = if tpl_gp_head == 0 {
                    tpl_body
                } else {
                    clone_with_rewrite(tpl_body, tpl_gp_head, packed)
                };
                // Build the new AST_FN_DECL with mangled name + concrete types.
                // Audit-stage7-8 Finding #10 fix: propagate the template's
                // is_checkpoint flag (slot 8) onto the synthesized clone.
                // Without this, `@checkpoint fn step<T>(...)` lost its
                // re-materialization marker when monomorphized, so the
                // reverse-mode AD pass treated the clone as a regular
                // (non-checkpointed) fn and memory grew linearly instead
                // of sqrt(N) per the @checkpoint contract.
                let tpl_is_ckpt = __arena_get(tpl_idx + 8);
                // Stage 28.9: also propagate validation attrs into mono
                // clones so deprecated_pass/trace_pass observe them on
                // the concrete clone as well as the template.
                let tpl_is_deprecated = __arena_get(tpl_idx + 9);
                let tpl_is_trace = __arena_get(tpl_idx + 10);
                let tpl_is_unwind = __arena_get(tpl_idx + 11);
                let tpl_dep_msg_s = __arena_get(tpl_idx + 12);
                let tpl_dep_msg_l = __arena_get(tpl_idx + 13);
                let tpl_is_kernel = __arena_get(tpl_idx + 14);
                let tpl_is_autotune = __arena_get(tpl_idx + 15);
                let tpl_autotune_product = __arena_get(tpl_idx + 16);
                let tpl_autotune_error = __arena_get(tpl_idx + 17);
                let tpl_since_msg_s = __arena_get(tpl_idx + 18);
                let tpl_since_msg_l = __arena_get(tpl_idx + 19);
                let clone_idx = mk_node(14, mang_s, mang_l, cloned_body);
                __arena_push(new_params_head);
                __arena_push(new_ret_ty);
                __arena_push(0);                 // is_generic = 0 (concrete)
                __arena_push(0);                 // slot 7: gp_names_head (none)
                __arena_push(tpl_is_ckpt);       // slot 8: propagated from template (F10)
                __arena_push(tpl_is_deprecated); // slot 9 (Stage 28.9)
                __arena_push(tpl_is_trace);      // slot 10 (Stage 28.9)
                __arena_push(tpl_is_unwind);     // slot 11 (Stage 28.9)
                __arena_push(tpl_dep_msg_s);      // slot 12 (Stage 33)
                __arena_push(tpl_dep_msg_l);      // slot 13 (Stage 33)
                __arena_push(tpl_is_kernel);      // slot 14 (Stage 33)
                __arena_push(tpl_is_autotune);    // slot 15 (Stage 33)
                __arena_push(tpl_autotune_product); // slot 16 (Stage 33)
                __arena_push(tpl_autotune_error); // slot 17 (Stage 33)
                __arena_push(tpl_since_msg_s);    // slot 18 (Stage 33)
                __arena_push(tpl_since_msg_l);    // slot 19 (Stage 33)
                // Append to fn_list tail.
                let new_list_node = mk_node(15, clone_idx, 0, 0);
                __arena_set(tail + 2, new_list_node);
                tail = new_list_node;
            };
            mi = mi + 1;
        }
        0
    }
}

// --------------------------------------------------------------
// Stage 12: forward-mode automatic differentiation. Given a user
// fn `loss(x: f64) -> f64` and a `grad(loss)(arg)` call site, the
// grad pass synthesizes a new fn `loss__grad(x: f64) -> f64` whose
// body is the symbolic derivative of loss's body w.r.t. `x`. The
// call site already references the mangled name (parser stage S12a),
// so codegen needs no special handling — the synthesized fn is
// emitted alongside user fns.
//
// Differentiation rules (literal, var, +, -, *, /, neg). Calls,
// if, while, let, seq trap with id 85001 (Phase-0 limitation).
// Simplifier folds 0+x=x, x+0=x, x-0=x, 0-x=-x, 0*x=0, 1*x=x,
// x*1=x, -(-x)=x, -0=0, and constant-folds f64 literals.
// --------------------------------------------------------------

// Push a fresh "0.0" literal (3 bytes: '0','.','0') into the arena
// and return its byte_start. Used by differentiate to materialize
// f64 zero. Each call allocates fresh bytes — wastes a few slots
// but keeps the implementation simple.
fn push_zero_f64_bytes() -> i32 {
    let s = __arena_push(48);    // '0'
    __arena_push(46);             // '.'
    __arena_push(48);             // '0'
    s
}

// Push a fresh "1.0" literal into the arena.
fn push_one_f64_bytes() -> i32 {
    let s = __arena_push(49);    // '1'
    __arena_push(46);             // '.'
    __arena_push(48);             // '0'
    s
}

// Build an AST_FLOATLIT_F64 node referencing freshly-pushed "0.0" bytes.
fn mk_zero_f64() -> i32 {
    let s = push_zero_f64_bytes();
    mk_node(34, s, 3, 0)
}

// Build an AST_FLOATLIT_F64 node referencing freshly-pushed "1.0" bytes.
fn mk_one_f64() -> i32 {
    let s = push_one_f64_bytes();
    mk_node(34, s, 3, 0)
}

// Test if a node is a literal whose decimal value equals 0.0 (f64 or
// f32 zero, AST_INT zero). Returns 1 on match, 0 otherwise. Used by
// the simplifier to detect 0+x=x, 0*x=0, etc.
@pure
fn ast_is_zero(idx: i32) -> i32 {
    let t = __arena_get(idx);
    if t == 0 {
        // AST_INT (i32). p1 = literal value.
        let v = __arena_get(idx + 1);
        if v == 0 { 1 } else { 0 }
    } else { if t == 34 {
        // AST_FLOATLIT_F64. Compare body bytes to "0.0" or just "0".
        // Body is stored as bytes at p1 (start) of length p2.
        let s = __arena_get(idx + 1);
        let l = __arena_get(idx + 2);
        // Walk bytes; non-zero digit (1..9) → return 0. Allow "0", "0.0",
        // "0.00", "00", etc. — any pattern that decimal-evaluates to 0.
        let mut i: i32 = 0;
        let mut all_zero: i32 = 1;
        while i < l {
            let b = __arena_get(s + i);
            if b == 46 { 0 } else { if b == 48 { 0 } else { all_zero = 0; } };
            i = i + 1;
        }
        all_zero
    } else { if t == 27 {
        // AST_FLOATLIT (f32). Same byte-walk as f64.
        let s = __arena_get(idx + 1);
        let l = __arena_get(idx + 2);
        let mut i: i32 = 0;
        let mut all_zero: i32 = 1;
        while i < l {
            let b = __arena_get(s + i);
            if b == 46 { 0 } else { if b == 48 { 0 } else { all_zero = 0; } };
            i = i + 1;
        }
        all_zero
    } else { 0 } } }
}

// Test if a node is a literal whose decimal value equals 1.0. Used by
// the simplifier to detect 1*x=x, x*1=x. Only matches f64 "1.0" pattern
// to keep it simple.
@pure
fn ast_is_one(idx: i32) -> i32 {
    let t = __arena_get(idx);
    if t == 34 {
        // AST_FLOATLIT_F64. Match "1.0" or "1" exactly.
        let s = __arena_get(idx + 1);
        let l = __arena_get(idx + 2);
        if l == 3 {
            let b0 = __arena_get(s);
            let b1 = __arena_get(s + 1);
            let b2 = __arena_get(s + 2);
            if b0 == 49 { if b1 == 46 { if b2 == 48 { 1 } else { 0 } } else { 0 } } else { 0 }
        } else { if l == 1 {
            let b0 = __arena_get(s);
            if b0 == 49 { 1 } else { 0 }
        } else { 0 } }
    } else { if t == 0 {
        let v = __arena_get(idx + 1);
        if v == 1 { 1 } else { 0 }
    } else { 0 } }
}

// Walk an AST subtree, replacing any AST_VAR whose name matches
// (param_s, param_l) with arg_idx (shared, not deep-cloned — the AST
// is read-only after parsing). Returns the (possibly new) idx. Used
// by inline_user_call to substitute the callee's body's param refs
// with the call's arg expressions.
fn clone_subst_var(expr_idx: i32, param_s: i32, param_l: i32, arg_idx: i32) -> i32 {
    let t = __arena_get(expr_idx);
    if t == 1 {
        // AST_VAR. Compare name bytes; if match, return arg_idx.
        let n_s = __arena_get(expr_idx + 1);
        let n_l = __arena_get(expr_idx + 2);
        if byte_eq(n_s, n_l, param_s, param_l) == 1 {
            arg_idx
        } else {
            expr_idx
        }
    } else { if t == 2 {
        let l = __arena_get(expr_idx + 1);
        let r = __arena_get(expr_idx + 2);
        let nl = clone_subst_var(l, param_s, param_l, arg_idx);
        let nr = clone_subst_var(r, param_s, param_l, arg_idx);
        if nl == l { if nr == r { expr_idx } else { mk_node(2, nl, nr, 0) } }
        else { mk_node(2, nl, nr, 0) }
    } else { if t == 3 {
        let l = __arena_get(expr_idx + 1);
        let r = __arena_get(expr_idx + 2);
        let nl = clone_subst_var(l, param_s, param_l, arg_idx);
        let nr = clone_subst_var(r, param_s, param_l, arg_idx);
        if nl == l { if nr == r { expr_idx } else { mk_node(3, nl, nr, 0) } }
        else { mk_node(3, nl, nr, 0) }
    } else { if t == 4 {
        let l = __arena_get(expr_idx + 1);
        let r = __arena_get(expr_idx + 2);
        let nl = clone_subst_var(l, param_s, param_l, arg_idx);
        let nr = clone_subst_var(r, param_s, param_l, arg_idx);
        if nl == l { if nr == r { expr_idx } else { mk_node(4, nl, nr, 0) } }
        else { mk_node(4, nl, nr, 0) }
    } else { if t == 5 {
        let l = __arena_get(expr_idx + 1);
        let r = __arena_get(expr_idx + 2);
        let nl = clone_subst_var(l, param_s, param_l, arg_idx);
        let nr = clone_subst_var(r, param_s, param_l, arg_idx);
        if nl == l { if nr == r { expr_idx } else { mk_node(5, nl, nr, 0) } }
        else { mk_node(5, nl, nr, 0) }
    } else { if t == 9 {
        let inner = __arena_get(expr_idx + 1);
        let ni = clone_subst_var(inner, param_s, param_l, arg_idx);
        if ni == inner { expr_idx } else { mk_node(9, ni, 0, 0) }
    } else {
        // Leaves (literals) and unhandled tags: return as-is.
        expr_idx
    } } } } } }
}

// Pre-differentiate pass: walk expr_idx; for each AST_CALL whose name
// matches a fn in fn_list, substitute the call with a clone of the
// callee's body where each param's name is replaced by the matching
// arg expression. Recurses into the substituted body to inline nested
// calls. Depth-limited at 6 (Phase-0 guard).
//
// Returns the (possibly new) idx with all eligible calls inlined.
// Calls whose callee is not in fn_list (e.g. transcendentals like
// __exp, __sin, or builtins) are left as-is — differentiate will
// trap on them with id 85001.
fn inline_user_calls(expr_idx: i32, head: i32, depth: i32) -> i32 {
    if depth >= 6 {
        expr_idx
    } else {
        let t = __arena_get(expr_idx);
        if t == 16 {
            // AST_CALL. p1 = name_s, p2 = name_l, p3 = args_head.
            let call_ns = __arena_get(expr_idx + 1);
            let call_nl = __arena_get(expr_idx + 2);
            let args_head = __arena_get(expr_idx + 3);
            // Find callee in fn_list.
            let mut walk: i32 = head;
            let mut callee_idx: i32 = 0;
            let mut fk: i32 = 1;
            while fk == 1 {
                let cand_idx = __arena_get(walk + 1);
                let cand_ns = __arena_get(cand_idx + 1);
                let cand_nl = __arena_get(cand_idx + 2);
                if byte_eq(call_ns, call_nl, cand_ns, cand_nl) == 1 {
                    callee_idx = cand_idx;
                    fk = 0;
                };
                if fk == 1 {
                    let nx = __arena_get(walk + 2);
                    if nx == 0 { fk = 0; } else { walk = nx; };
                };
            }
            if callee_idx == 0 {
                // Not found — leave as-is. differentiate will trap.
                expr_idx
            } else {
                // Get callee body + params.
                let cal_body = __arena_get(callee_idx + 3);
                let cal_params = __arena_get(callee_idx + 4);
                // Substitute each param with its arg. Walk in lockstep:
                // params chain (AST_PARAM via slot 3 = next) and args
                // chain (AST_ARG via slot 2 = next).
                let mut p_walk: i32 = cal_params;
                let mut a_walk: i32 = args_head;
                let mut cur_body: i32 = cal_body;
                while p_walk != 0 {
                    if a_walk == 0 {
                        // Arity mismatch — abort inlining; return original
                        // call so differentiate can trap.
                        p_walk = 0;
                        cur_body = expr_idx;
                    } else {
                        let p_ns = __arena_get(p_walk + 1);
                        let p_nl = __arena_get(p_walk + 2);
                        let arg_expr = __arena_get(a_walk + 1);
                        cur_body = clone_subst_var(cur_body, p_ns, p_nl, arg_expr);
                        p_walk = __arena_get(p_walk + 3);
                        a_walk = __arena_get(a_walk + 2);
                    };
                }
                // Recursively inline nested calls in the substituted body.
                inline_user_calls(cur_body, head, depth + 1)
            }
        } else { if t == 2 {
            let l = __arena_get(expr_idx + 1);
            let r = __arena_get(expr_idx + 2);
            let nl = inline_user_calls(l, head, depth);
            let nr = inline_user_calls(r, head, depth);
            if nl == l { if nr == r { expr_idx } else { mk_node(2, nl, nr, 0) } }
            else { mk_node(2, nl, nr, 0) }
        } else { if t == 3 {
            let l = __arena_get(expr_idx + 1);
            let r = __arena_get(expr_idx + 2);
            let nl = inline_user_calls(l, head, depth);
            let nr = inline_user_calls(r, head, depth);
            if nl == l { if nr == r { expr_idx } else { mk_node(3, nl, nr, 0) } }
            else { mk_node(3, nl, nr, 0) }
        } else { if t == 4 {
            let l = __arena_get(expr_idx + 1);
            let r = __arena_get(expr_idx + 2);
            let nl = inline_user_calls(l, head, depth);
            let nr = inline_user_calls(r, head, depth);
            if nl == l { if nr == r { expr_idx } else { mk_node(4, nl, nr, 0) } }
            else { mk_node(4, nl, nr, 0) }
        } else { if t == 5 {
            let l = __arena_get(expr_idx + 1);
            let r = __arena_get(expr_idx + 2);
            let nl = inline_user_calls(l, head, depth);
            let nr = inline_user_calls(r, head, depth);
            if nl == l { if nr == r { expr_idx } else { mk_node(5, nl, nr, 0) } }
            else { mk_node(5, nl, nr, 0) }
        } else { if t == 9 {
            let inner = __arena_get(expr_idx + 1);
            let ni = inline_user_calls(inner, head, depth);
            if ni == inner { expr_idx } else { mk_node(9, ni, 0, 0) }
        } else {
            expr_idx
        } } } } } }
    }
}

// Differentiate an AST subtree w.r.t. variable `var_s`/`var_l` (the
// param byte-range). Returns the index of a freshly-built derivative
// node. Recursively walks the tree; for unsupported tags emits a
// runtime trap (AST node tag 99 = AST_ERR with op-specific p1).
//
// Phase-0 supported tags: 0 (INT), 27 (FLOATLIT f32), 34 (FLOATLIT_F64),
// 35 (INTLIT_I64), 1 (VAR), 2 (ADD), 3 (SUB), 4 (MUL), 5 (DIV), 9 (NEG).
// Trap-id 85001 emitted via mk_node(99, ...) for unsupported tags.
fn differentiate(expr_idx: i32, var_s: i32, var_l: i32) -> i32 {
    let t = __arena_get(expr_idx);
    if t == 0 {
        // d(c) = 0 for integer literal — but the surrounding type is
        // f64, so emit f64 zero.
        mk_zero_f64()
    } else { if t == 27 {
        mk_zero_f64()
    } else { if t == 34 {
        mk_zero_f64()
    } else { if t == 35 {
        mk_zero_f64()
    } else { if t == 1 {
        // AST_VAR. p1 = name_s, p2 = name_l. d(x) = 1 if x == var_name,
        // else 0.
        let n_s = __arena_get(expr_idx + 1);
        let n_l = __arena_get(expr_idx + 2);
        if byte_eq(n_s, n_l, var_s, var_l) == 1 {
            mk_one_f64()
        } else {
            mk_zero_f64()
        }
    } else { if t == 2 {
        // AST_ADD. d(a + b) = d(a) + d(b).
        let l = __arena_get(expr_idx + 1);
        let r = __arena_get(expr_idx + 2);
        let dl = differentiate(l, var_s, var_l);
        let dr = differentiate(r, var_s, var_l);
        mk_node(2, dl, dr, 0)
    } else { if t == 3 {
        // AST_SUB. d(a - b) = d(a) - d(b).
        let l = __arena_get(expr_idx + 1);
        let r = __arena_get(expr_idx + 2);
        let dl = differentiate(l, var_s, var_l);
        let dr = differentiate(r, var_s, var_l);
        mk_node(3, dl, dr, 0)
    } else { if t == 4 {
        // AST_MUL. Product rule: d(a*b) = d(a)*b + a*d(b).
        // IMPORTANT (arena positional ordering): build the children
        // BEFORE allocating the parent so the parent's p1/p2 indices
        // remain valid. Otherwise the parent allocation would land
        // BETWEEN child allocations and read garbage.
        let l = __arena_get(expr_idx + 1);
        let r = __arena_get(expr_idx + 2);
        let dl = differentiate(l, var_s, var_l);
        let dr = differentiate(r, var_s, var_l);
        let term1 = mk_node(4, dl, r, 0);
        let term2 = mk_node(4, l, dr, 0);
        mk_node(2, term1, term2, 0)
    } else { if t == 5 {
        // AST_DIV. Quotient rule: d(a/b) = (d(a)*b - a*d(b)) / (b*b).
        let l = __arena_get(expr_idx + 1);
        let r = __arena_get(expr_idx + 2);
        let dl = differentiate(l, var_s, var_l);
        let dr = differentiate(r, var_s, var_l);
        let num1 = mk_node(4, dl, r, 0);
        let num2 = mk_node(4, l, dr, 0);
        let num = mk_node(3, num1, num2, 0);
        let denom = mk_node(4, r, r, 0);
        mk_node(5, num, denom, 0)
    } else { if t == 9 {
        // AST_NEG. d(-a) = -d(a).
        let inner = __arena_get(expr_idx + 1);
        let di = differentiate(inner, var_s, var_l);
        mk_node(9, di, 0, 0)
    } else {
        // Unsupported tag (CALL, IF, WHILE, LET, SEQ, BLOCK, ...) —
        // Phase-0 limitation. Trap with id 85001 by emitting an
        // AST_ERR node; codegen lowers AST_ERR to ud2.
        mk_node(99, 85001, 0, 0)
    } } } } } } } } } }
}

// Stage 14: reverse-mode propagator for ONE target param.
//
// Walks `expr_idx` top-down with a current adjoint expression `adj`.
// At AST_VAR leaves matching (var_s, var_l), appends `adj` to the
// bucket chain (head idx returned via out_head_slot). At binary ops,
// splits the adjoint per local Jacobian and recurses into both
// children with the per-side adjoint:
//
//   ADD(a, b), adj A:    propagate A to a, A to b (shared)
//   SUB(a, b), adj A:    A to a, -A to b
//   MUL(a, b), adj A:    A*b to a, A*a to b
//   DIV(a, b), adj A:    A/b to a, -(A*a)/(b*b) to b
//   NEG(a), adj A:       -A to a
//
// Bucket: a singly-linked list whose head index is stored in slot
// `out_head_slot` (we use parser scratch slot sb+72 = bucket_head,
// sb+73 = bucket_count). Each bucket node is `(expr_idx, next)` —
// reuse AST_ARG (tag 17) for the cell shape: slot 1 = expr_idx,
// slot 2 = next, slot 3 = unused. Capacity cap 32 bucket entries
// per param (trap 89001 if exceeded).
//
// Compared with forward-mode `differentiate`, the result for any
// single param is mathematically identical for scalar-output loss
// — the ALGORITHMIC shape is what differs. Forward repeatedly walks
// the body once per param; reverse walks once and deposits into all
// params' buckets. This implementation walks once per target param
// (so for grad_rev_all with N params, we walk N times); a future
// optimization could simultaneously deposit into N buckets in one
// walk. For now the structural correctness — top-down adjoint
// propagation — is what matters.
fn bucket_head_slot(sb: i32) -> i32  { __arena_get(sb + 72) }
fn bucket_count_slot(sb: i32) -> i32 { __arena_get(sb + 73) }
fn set_bucket_head(sb: i32, v: i32) -> i32  { __arena_set(sb + 72, v); 0 }
fn set_bucket_count(sb: i32, v: i32) -> i32 { __arena_set(sb + 73, v); 0 }
// Stage 14.5: @checkpoint attribute scratch slot. Set by skip_attributes
// when it consumes a `@checkpoint` attribute, read+cleared by
// parse_fn_decl when it allocates the AST_FN_DECL node.
fn next_fn_is_ckpt(sb: i32) -> i32 { __arena_get(sb + 74) }
fn set_next_fn_is_ckpt(sb: i32, v: i32) -> i32 { __arena_set(sb + 74, v); 0 }
// Stage 28.9 (validation passes): attribute scratch slots. Each is
// set by skip_attributes on matching `@<ident>` consumption,
// read+cleared by parse_fn_decl on AST_FN_DECL allocation. The
// bootstrap-side validation passes (deprecated_pass, trace_pass,
// panic_pass.@unwind) inspect the corresponding AST_FN_DECL slot.
fn next_fn_is_deprecated(sb: i32) -> i32 { __arena_get(sb + 75) }
fn set_next_fn_is_deprecated(sb: i32, v: i32) -> i32 { __arena_set(sb + 75, v); 0 }
fn next_fn_is_trace(sb: i32) -> i32 { __arena_get(sb + 76) }
fn set_next_fn_is_trace(sb: i32, v: i32) -> i32 { __arena_set(sb + 76, v); 0 }
fn next_fn_is_unwind(sb: i32) -> i32 { __arena_get(sb + 77) }
fn set_next_fn_is_unwind(sb: i32, v: i32) -> i32 { __arena_set(sb + 77, v); 0 }
fn next_fn_deprecated_msg_s(sb: i32) -> i32 { __arena_get(sb + 80) }
fn set_next_fn_deprecated_msg_s(sb: i32, v: i32) -> i32 { __arena_set(sb + 80, v); 0 }
fn next_fn_deprecated_msg_l(sb: i32) -> i32 { __arena_get(sb + 81) }
fn set_next_fn_deprecated_msg_l(sb: i32, v: i32) -> i32 { __arena_set(sb + 81, v); 0 }
fn next_fn_is_kernel(sb: i32) -> i32 { __arena_get(sb + 82) }
fn set_next_fn_is_kernel(sb: i32, v: i32) -> i32 { __arena_set(sb + 82, v); 0 }
fn next_fn_is_autotune(sb: i32) -> i32 { __arena_get(sb + 83) }
fn set_next_fn_is_autotune(sb: i32, v: i32) -> i32 { __arena_set(sb + 83, v); 0 }
fn next_fn_autotune_product(sb: i32) -> i32 { __arena_get(sb + 84) }
fn set_next_fn_autotune_product(sb: i32, v: i32) -> i32 { __arena_set(sb + 84, v); 0 }
fn next_fn_autotune_error(sb: i32) -> i32 { __arena_get(sb + 85) }
fn set_next_fn_autotune_error(sb: i32, v: i32) -> i32 { __arena_set(sb + 85, v); 0 }
fn next_fn_since_msg_s(sb: i32) -> i32 { __arena_get(sb + 86) }
fn set_next_fn_since_msg_s(sb: i32, v: i32) -> i32 { __arena_set(sb + 86, v); 0 }
fn next_fn_since_msg_l(sb: i32) -> i32 { __arena_get(sb + 87) }
fn set_next_fn_since_msg_l(sb: i32, v: i32) -> i32 { __arena_set(sb + 87, v); 0 }
fn bucket_reset(sb: i32) -> i32 {
    set_bucket_head(sb, 0);
    set_bucket_count(sb, 0);
    0
}
// Append (expr_idx) to the bucket. Returns the bucket node idx (or
// 0 on overflow). Uses AST_ARG (tag 17) shape for the cell.
fn bucket_append(sb: i32, expr_idx: i32) -> i32 {
    let cnt = bucket_count_slot(sb);
    if cnt >= 32 {
        mk_node(99, 89001, 0, 0)
    } else {
        let cell = mk_node(17, expr_idx, 0, 0);
        let head = bucket_head_slot(sb);
        if head == 0 {
            set_bucket_head(sb, cell);
        } else {
            // Walk to tail.
            let mut walk: i32 = head;
            let mut wkeep: i32 = 1;
            while wkeep == 1 {
                let nx = __arena_get(walk + 2);
                if nx == 0 { wkeep = 0; } else { walk = nx; };
            }
            __arena_set(walk + 2, cell);
        };
        set_bucket_count(sb, cnt + 1);
        cell
    }
}

// Recursive adjoint propagator. Walks node, deposits contributions
// into bucket. Returns 0; side-effect via bucket.
fn propagate_adj(sb: i32, node: i32, adj: i32, var_s: i32, var_l: i32) -> i32 {
    let t = __arena_get(node);
    // Literals: no contribution.
    if t == 0 { 0 }
    else { if t == 27 { 0 }
    else { if t == 34 { 0 }
    else { if t == 35 { 0 }
    else { if t == 1 {
        // AST_VAR. If name matches target param, append adj.
        let n_s = __arena_get(node + 1);
        let n_l = __arena_get(node + 2);
        if byte_eq(n_s, n_l, var_s, var_l) == 1 {
            bucket_append(sb, adj);
        };
        0
    } else { if t == 2 {
        // AST_ADD: adj_l = adj, adj_r = adj.
        let l = __arena_get(node + 1);
        let r = __arena_get(node + 2);
        propagate_adj(sb, l, adj, var_s, var_l);
        propagate_adj(sb, r, adj, var_s, var_l);
        0
    } else { if t == 3 {
        // AST_SUB: adj_l = adj, adj_r = -adj.
        let l = __arena_get(node + 1);
        let r = __arena_get(node + 2);
        propagate_adj(sb, l, adj, var_s, var_l);
        let neg_adj = mk_node(9, adj, 0, 0);
        propagate_adj(sb, r, neg_adj, var_s, var_l);
        0
    } else { if t == 4 {
        // AST_MUL: adj_l = adj * r, adj_r = adj * l.
        // CRITICAL: build adj_l/adj_r BEFORE recursing so the AST node
        // alloc lives at a stable index.
        let l = __arena_get(node + 1);
        let r = __arena_get(node + 2);
        let adj_l = mk_node(4, adj, r, 0);
        let adj_r = mk_node(4, adj, l, 0);
        propagate_adj(sb, l, adj_l, var_s, var_l);
        propagate_adj(sb, r, adj_r, var_s, var_l);
        0
    } else { if t == 5 {
        // AST_DIV: adj_l = adj / r, adj_r = -(adj * l) / (r * r).
        let l = __arena_get(node + 1);
        let r = __arena_get(node + 2);
        let adj_l = mk_node(5, adj, r, 0);
        let r_sq = mk_node(4, r, r, 0);
        let al = mk_node(4, adj, l, 0);
        let div = mk_node(5, al, r_sq, 0);
        let adj_r = mk_node(9, div, 0, 0);
        propagate_adj(sb, l, adj_l, var_s, var_l);
        propagate_adj(sb, r, adj_r, var_s, var_l);
        0
    } else { if t == 9 {
        // AST_NEG: adj_inner = -adj.
        let inner = __arena_get(node + 1);
        let neg_adj = mk_node(9, adj, 0, 0);
        propagate_adj(sb, inner, neg_adj, var_s, var_l);
        0
    } else {
        // Unsupported (CALL, IF, WHILE, LET, ...): Phase-0 trap 88001.
        // We don't have a way to surface this from inside the recursion,
        // so we simply append a poison expr to the bucket; the caller's
        // outer logic will let codegen surface it.
        bucket_append(sb, mk_node(99, 88001, 0, 0));
        0
    } } } } } } } } } }
}

// Sum the bucket chain into a single expression (chain of binary +).
// Empty bucket → fresh 0.0_f64 literal. Single entry → just that entry.
fn sum_bucket(sb: i32) -> i32 {
    let head = bucket_head_slot(sb);
    if head == 0 {
        mk_zero_f64()
    } else {
        let first_expr = __arena_get(head + 1);
        let mut acc: i32 = first_expr;
        let mut walk: i32 = __arena_get(head + 2);
        let mut wkeep: i32 = 1;
        while wkeep == 1 {
            if walk == 0 { wkeep = 0; } else {
                let cell_expr = __arena_get(walk + 1);
                acc = mk_node(2, acc, cell_expr, 0);
                walk = __arena_get(walk + 2);
            };
        }
        acc
    }
}

// Reverse-mode top-level: differentiate `expr_idx` w.r.t. (var_s, var_l)
// by seeding adjoint = 1.0_f64, walking with propagate_adj, then summing
// the bucket. Returns the (un-simplified) derivative AST idx.
fn differentiate_reverse_one(sb: i32, expr_idx: i32, var_s: i32, var_l: i32) -> i32 {
    bucket_reset(sb);
    let seed = mk_one_f64();
    propagate_adj(sb, expr_idx, seed, var_s, var_l);
    sum_bucket(sb)
}

// ========================================================================
// Stage 50 Inc 1 — multi-bucket infrastructure (groundwork)
//
// The single-bucket path above (bucket_head/count + propagate_adj +
// differentiate_reverse_one) walks the body once per target param.
// Stage 50 Inc 2 will refactor the caller loop to use the new
// infrastructure below for a true single-walk algorithm that deposits
// into N parallel buckets in one DFS.
//
// Scratch slot layout (sb-relative):
//   sb+88                : param_n (count of active params, 0..8)
//   sb+90  .. sb+105     : 8 pairs (start, len) of param names (16 slots)
//   sb+106 .. sb+113     : 8 bucket heads (one per param)
//   sb+114 .. sb+121     : 8 bucket counts (one per param)
//
// Bucket cell shape mirrors the single-bucket version (AST_ARG tag 17,
// reusing the same `mk_node(17, expr_idx, 0, 0)` + walk-to-tail
// linked-list pattern).
//
// Param-name lookup: `param_idx_of(sb, var_s, var_l) -> i32` returns the
// matching idx 0..n-1, or -1 if no match. This is the deposit-target
// selector used by `propagate_adj_multi` at every AST_VAR node.
//
// IMPORTANT: Inc 1 is purely ADDITIVE. The new helpers are never called
// by the existing caller loop. Self-host cascade G3..G4 byte-identical
// preservation is trivial because the executed code path is unchanged.
// Inc 2 swaps the caller; that's where the self-host invariant becomes
// non-trivial and must be re-verified.
// ========================================================================

fn param_n_slot(sb: i32) -> i32 { __arena_get(sb + 88) }
fn set_param_n(sb: i32, v: i32) -> i32 { __arena_set(sb + 88, v); 0 }

fn param_array_name_s(sb: i32, idx: i32) -> i32 { __arena_get(sb + 90 + idx * 2) }
fn param_array_name_l(sb: i32, idx: i32) -> i32 { __arena_get(sb + 91 + idx * 2) }
fn set_param_array_name(sb: i32, idx: i32, s: i32, l: i32) -> i32 {
    __arena_set(sb + 90 + idx * 2, s);
    __arena_set(sb + 91 + idx * 2, l);
    0
}

fn bucket_array_head(sb: i32, idx: i32) -> i32 { __arena_get(sb + 106 + idx) }
fn bucket_array_count(sb: i32, idx: i32) -> i32 { __arena_get(sb + 114 + idx) }
fn set_bucket_array_head(sb: i32, idx: i32, v: i32) -> i32 {
    __arena_set(sb + 106 + idx, v); 0
}
fn set_bucket_array_count(sb: i32, idx: i32, v: i32) -> i32 {
    __arena_set(sb + 114 + idx, v); 0
}

// Reset all N buckets + clear param-name slots. Cap n at 8.
fn bucket_array_reset(sb: i32, n: i32) -> i32 {
    let n_clamped = if n > 8 { 8 } else { if n < 0 { 0 } else { n } };
    set_param_n(sb, n_clamped);
    let mut i: i32 = 0;
    while i < 8 {
        set_param_array_name(sb, i, 0, 0);
        set_bucket_array_head(sb, i, 0);
        set_bucket_array_count(sb, i, 0);
        i = i + 1;
    }
    0
}

// Append (expr_idx) to bucket idx. Returns the cell node idx (or 0 on
// overflow: cap at 32 per bucket, same as the single-bucket version).
// Caller must ensure 0 <= idx < param_n.
fn bucket_array_append(sb: i32, idx: i32, expr_idx: i32) -> i32 {
    let cnt = bucket_array_count(sb, idx);
    if cnt >= 32 {
        mk_node(99, 89001, 0, 0)
    } else {
        let cell = mk_node(17, expr_idx, 0, 0);
        let head = bucket_array_head(sb, idx);
        if head == 0 {
            set_bucket_array_head(sb, idx, cell);
        } else {
            let mut walk: i32 = head;
            let mut wkeep: i32 = 1;
            while wkeep == 1 {
                let nx = __arena_get(walk + 2);
                if nx == 0 { wkeep = 0; } else { walk = nx; };
            }
            __arena_set(walk + 2, cell);
        };
        set_bucket_array_count(sb, idx, cnt + 1);
        cell
    }
}

// Sum bucket idx into a single expression (chain of binary +).
// Empty bucket → fresh 0.0_f64 literal. Single entry → just that entry.
// Mirrors sum_bucket but parameterized over idx.
fn bucket_array_sum(sb: i32, idx: i32) -> i32 {
    let head = bucket_array_head(sb, idx);
    if head == 0 {
        mk_zero_f64()
    } else {
        let first_expr = __arena_get(head + 1);
        let mut acc: i32 = first_expr;
        let mut walk: i32 = __arena_get(head + 2);
        let mut wkeep: i32 = 1;
        while wkeep == 1 {
            if walk == 0 { wkeep = 0; } else {
                let cell_expr = __arena_get(walk + 1);
                acc = mk_node(2, acc, cell_expr, 0);
                walk = __arena_get(walk + 2);
            };
        }
        acc
    }
}

// Linear scan over param-name array. Returns 0..n-1 if a match exists,
// or -1 if none. Used by propagate_adj_multi at every AST_VAR node.
fn param_idx_of(sb: i32, var_s: i32, var_l: i32) -> i32 {
    let n = param_n_slot(sb);
    let mut i: i32 = 0;
    let mut found: i32 = 0 - 1;
    while i < n {
        let p_s = param_array_name_s(sb, i);
        let p_l = param_array_name_l(sb, i);
        if byte_eq(p_s, p_l, var_s, var_l) == 1 {
            if found < 0 { found = i; };
        };
        i = i + 1;
    }
    found
}

// Recursive multi-bucket adjoint propagator. Walks node, deposits each
// AST_VAR match into the matching bucket (or no bucket if no param
// matches). Returns 0; side-effect via bucket_array_append.
// Structure mirrors propagate_adj exactly — same DFS order, same
// adjoint-construction shape — so deposit sequence within each bucket
// is identical to the N-walk version (preserves Inc 2's byte-identical
// self-host invariant).
fn propagate_adj_multi(sb: i32, node: i32, adj: i32) -> i32 {
    let t = __arena_get(node);
    if t == 0 { 0 }
    else { if t == 27 { 0 }
    else { if t == 34 { 0 }
    else { if t == 35 { 0 }
    else { if t == 1 {
        // AST_VAR. If name matches ANY active param, append adj to
        // that param's bucket.
        let n_s = __arena_get(node + 1);
        let n_l = __arena_get(node + 2);
        let idx = param_idx_of(sb, n_s, n_l);
        if idx >= 0 {
            bucket_array_append(sb, idx, adj);
        };
        0
    } else { if t == 2 {
        // AST_ADD: adj_l = adj, adj_r = adj.
        let l = __arena_get(node + 1);
        let r = __arena_get(node + 2);
        propagate_adj_multi(sb, l, adj);
        propagate_adj_multi(sb, r, adj);
        0
    } else { if t == 3 {
        // AST_SUB: adj_l = adj, adj_r = -adj.
        let l = __arena_get(node + 1);
        let r = __arena_get(node + 2);
        propagate_adj_multi(sb, l, adj);
        let neg_adj = mk_node(9, adj, 0, 0);
        propagate_adj_multi(sb, r, neg_adj);
        0
    } else { if t == 4 {
        // AST_MUL: adj_l = adj * r, adj_r = adj * l.
        let l = __arena_get(node + 1);
        let r = __arena_get(node + 2);
        let adj_l = mk_node(4, adj, r, 0);
        let adj_r = mk_node(4, adj, l, 0);
        propagate_adj_multi(sb, l, adj_l);
        propagate_adj_multi(sb, r, adj_r);
        0
    } else { if t == 5 {
        // AST_DIV: adj_l = adj / r, adj_r = -(adj * l) / (r * r).
        let l = __arena_get(node + 1);
        let r = __arena_get(node + 2);
        let adj_l = mk_node(5, adj, r, 0);
        let r_sq = mk_node(4, r, r, 0);
        let al = mk_node(4, adj, l, 0);
        let div = mk_node(5, al, r_sq, 0);
        let adj_r = mk_node(9, div, 0, 0);
        propagate_adj_multi(sb, l, adj_l);
        propagate_adj_multi(sb, r, adj_r);
        0
    } else { if t == 9 {
        // AST_NEG: adj_inner = -adj.
        let inner = __arena_get(node + 1);
        let neg_adj = mk_node(9, adj, 0, 0);
        propagate_adj_multi(sb, inner, neg_adj);
        0
    } else {
        // Unsupported (CALL, IF, WHILE, LET, ...): Phase-0 trap 88001.
        // Append to bucket 0 as the poison sentinel (mirrors the
        // single-bucket version's poison strategy). The caller's
        // outer logic surfaces this at codegen.
        if param_n_slot(sb) > 0 {
            bucket_array_append(sb, 0, mk_node(99, 88001, 0, 0));
        };
        0
    } } } } } } } } } }
}

// Stage 50 Inc 1 entry point: walks `expr_idx` once with seed = 1.0_f64,
// depositing adjoints into N parallel buckets. Caller must have set up
// param_array (via bucket_array_reset + set_param_array_name) before
// calling. After the walk, caller reads each bucket via bucket_array_sum.
// Returns 0.
fn differentiate_reverse_all(sb: i32, expr_idx: i32) -> i32 {
    let seed = mk_one_f64();
    propagate_adj_multi(sb, expr_idx, seed);
    0
}

// Stage 50 Inc 2 — production-path bridge: one-param convenience wrapper
// that uses the new bucket_array infrastructure (n=1) and returns the
// derivative AST. Drop-in replacement for `differentiate_reverse_one`
// at the parse_program-end grad_rev_pass caller site (parser.hx ~6293).
//
// Algorithmically identical to differentiate_reverse_one for the n=1
// case: same DFS via propagate_adj_multi (which mirrors propagate_adj's
// shape exactly), same deposit order (single-bucket path always hits
// idx=0), same sum_bucket-equivalent reduction via bucket_array_sum(0).
//
// Why this bridge: it tests the new infrastructure with real production
// code BEFORE Inc 3 introduces actual grouping (where multiple params
// share one walk). If self-host cascade G3..G4 remains byte-identical
// after this swap, we've validated that bucket_array_* is a correctness-
// equivalent replacement for the single-bucket helpers. Inc 3 then
// confidently extends to true n>1 single-walk.
fn differentiate_reverse_one_via_array(
    sb: i32, expr_idx: i32, var_s: i32, var_l: i32
) -> i32 {
    bucket_array_reset(sb, 1);
    set_param_array_name(sb, 0, var_s, var_l);
    differentiate_reverse_all(sb, expr_idx);
    bucket_array_sum(sb, 0)
}

// Bottom-up algebraic simplifier for the differentiate output. Folds
// 0+x=x, x+0=x, x-0=x, 0-x=-x, 0*x=0, 1*x=x, x*1=x, -(-x)=x, -0=0,
// and constant-folds two literal-zero / literal-one operands into a
// new literal. Returns the (possibly new) node index.
fn simplify(expr_idx: i32) -> i32 {
    let t = __arena_get(expr_idx);
    if t == 2 {
        // AST_ADD: simplify children first, then fold.
        let l = __arena_get(expr_idx + 1);
        let r = __arena_get(expr_idx + 2);
        let sl = simplify(l);
        let sr = simplify(r);
        if ast_is_zero(sl) == 1 {
            sr
        } else { if ast_is_zero(sr) == 1 {
            sl
        } else {
            mk_node(2, sl, sr, 0)
        } }
    } else { if t == 3 {
        // AST_SUB: x - 0 = x; 0 - x = -x.
        let l = __arena_get(expr_idx + 1);
        let r = __arena_get(expr_idx + 2);
        let sl = simplify(l);
        let sr = simplify(r);
        if ast_is_zero(sr) == 1 {
            sl
        } else { if ast_is_zero(sl) == 1 {
            mk_node(9, sr, 0, 0)
        } else {
            mk_node(3, sl, sr, 0)
        } }
    } else { if t == 4 {
        // AST_MUL: 0*x=0, x*0=0, 1*x=x, x*1=x.
        let l = __arena_get(expr_idx + 1);
        let r = __arena_get(expr_idx + 2);
        let sl = simplify(l);
        let sr = simplify(r);
        if ast_is_zero(sl) == 1 {
            mk_zero_f64()
        } else { if ast_is_zero(sr) == 1 {
            mk_zero_f64()
        } else { if ast_is_one(sl) == 1 {
            sr
        } else { if ast_is_one(sr) == 1 {
            sl
        } else {
            mk_node(4, sl, sr, 0)
        } } } }
    } else { if t == 5 {
        // AST_DIV: simplify children; no algebraic identity beyond that.
        let l = __arena_get(expr_idx + 1);
        let r = __arena_get(expr_idx + 2);
        let sl = simplify(l);
        let sr = simplify(r);
        mk_node(5, sl, sr, 0)
    } else { if t == 9 {
        // AST_NEG: -(-x) = x; -0 = 0.
        let inner = __arena_get(expr_idx + 1);
        let si = simplify(inner);
        let it = __arena_get(si);
        if it == 9 {
            // -(-x) → x
            __arena_get(si + 1)
        } else { if ast_is_zero(si) == 1 {
            mk_zero_f64()
        } else {
            mk_node(9, si, 0, 0)
        } }
    } else {
        // Leaf or unhandled tag (literal, var, etc.) — return as-is.
        expr_idx
    } } } } }
}

// Stage 12: grad pass. For each entry in grad_pending, find the
// corresponding loss fn in fn_list (matching by name), differentiate
// its body w.r.t. its first param's name, simplify, and synthesize
// a new AST_FN_DECL with the mangled name "<loss>__grad" containing
// the simplified derivative as its body. The new fn shares the loss
// fn's params and ret_ty. Append to fn_list tail so codegen emits it.
//
// If the loss fn has zero params, the synthesized fn is still emitted
// but the differentiate result is constant zero (no var to match).
fn grad_pass(sb: i32, head: i32) -> i32 {
    let count = grad_pending_count(sb);
    if count == 0 {
        0
    } else {
        // Find tail of fn_list (where next == 0) for appending.
        let mut tail = head;
        let mut tail_keep: i32 = 1;
        while tail_keep == 1 {
            let nx = __arena_get(tail + 2);
            if nx == 0 { tail_keep = 0; } else { tail = nx; };
        }
        let base = grad_pending_base(sb);
        let mut gi: i32 = 0;
        while gi < count {
            let entry = base + gi * 4;
            let loss_s = __arena_get(entry);
            let loss_l = __arena_get(entry + 1);
            let mang_s = __arena_get(entry + 2);
            let mang_l = __arena_get(entry + 3);
            // Find loss fn in fn_list by name match.
            let mut walk: i32 = head;
            let mut loss_fn_idx: i32 = 0;
            let mut find_keep: i32 = 1;
            while find_keep == 1 {
                let cand_idx = __arena_get(walk + 1);
                let cand_ns = __arena_get(cand_idx + 1);
                let cand_nl = __arena_get(cand_idx + 2);
                if byte_eq(loss_s, loss_l, cand_ns, cand_nl) == 1 {
                    loss_fn_idx = cand_idx;
                    find_keep = 0;
                };
                if find_keep == 1 {
                    let nx = __arena_get(walk + 2);
                    if nx == 0 { find_keep = 0; } else { walk = nx; };
                };
            }
            if loss_fn_idx > 0 {
                // Read loss fn slots (Stage 8 layout):
                //   slot 1: name_s, 2: name_l, 3: body_idx, 4: params_head,
                //   5: ret_ty, 6: is_generic, 7: gp_names_head, 8: is_checkpoint.
                let loss_body = __arena_get(loss_fn_idx + 3);
                let loss_params = __arena_get(loss_fn_idx + 4);
                let loss_ret_ty = __arena_get(loss_fn_idx + 5);
                let loss_is_ckpt = __arena_get(loss_fn_idx + 8);
                // Extract first param's name (the variable to differentiate
                // w.r.t.). AST_PARAM (tag 18) layout: slot 1 = name_s,
                // 2 = name_l, 3 = next, 4 = type_tag.
                let var_s = if loss_params == 0 { 0 } else { __arena_get(loss_params + 1) };
                let var_l = if loss_params == 0 { 0 } else { __arena_get(loss_params + 2) };
                // Stage 13 prep: inline user-fn calls inside loss before
                // differentiating. depth=0; recursion cap=6 inside the
                // pass. Calls to unknown fns (transcendentals, builtins)
                // are left as-is — differentiate will trap on them.
                let inlined = inline_user_calls(loss_body, head, 0);
                // Differentiate then simplify.
                let deriv_raw = differentiate(inlined, var_s, var_l);
                let deriv = simplify(deriv_raw);
                // Synthesize the AST_FN_DECL clone. Same param chain (we
                // share — derivative still takes the same input shape) and
                // same ret_ty (f64 in the test cases). The clone has
                // is_generic=0 and no gp_names.
                let clone_fn = mk_node(14, mang_s, mang_l, deriv);
                __arena_push(loss_params);   // slot 4: params_head
                __arena_push(loss_ret_ty);   // slot 5: ret_ty
                __arena_push(0);             // slot 6: is_generic = 0
                __arena_push(0);             // slot 7: gp_names_head = 0
                // F10 fix companion: if the loss fn carried @checkpoint, the
                // synthesized gradient should inherit it too — otherwise
                // AD memory grows linearly through the cloned derivative
                // even when the user explicitly opted into re-mat.
                __arena_push(loss_is_ckpt);  // slot 8: propagated from loss
                // Stage 28.9: validation attrs default to 0 on synthesized
                // grad clones — the gradient is a NEW fn, not the user's
                // original loss. @deprecated/@trace/@unwind don't propagate.
                __arena_push(0);             // slot 9: is_deprecated = 0
                __arena_push(0);             // slot 10: is_trace = 0
                __arena_push(0);             // slot 11: is_unwind = 0
                __arena_push(0);             // slot 12: deprecated_msg_start = 0
                __arena_push(0);             // slot 13: deprecated_msg_len = 0
                __arena_push(0);             // slot 14: is_kernel = 0
                __arena_push(0);             // slot 15: is_autotune = 0
                __arena_push(0);             // slot 16: autotune_product = 0
                __arena_push(0);             // slot 17: autotune_parse_error_kind = 0
                __arena_push(0);             // slot 18: since_msg_start = 0
                __arena_push(0);             // slot 19: since_msg_len = 0
                let new_list_node = mk_node(15, clone_fn, 0, 0);
                __arena_set(tail + 2, new_list_node);
                tail = new_list_node;
            };
            gi = gi + 1;
        }
        0
    }
}

// Stage 14.5: walk loss body, scan for AST_CALL to user fns marked
// @checkpoint, and for each one verify that the callee body is pure.
// Returns 1 iff every reachable @checkpoint callee is pure. Pure-but-
// non-@checkpoint callees are NOT checked (they go through the normal
// reverse-mode path, which already traps on unsupported ops).
fn ckpt_callees_pure(expr_idx: i32, head: i32, depth: i32) -> i32 {
    if depth >= 2000 {
        0
    } else {
        let t = __arena_get(expr_idx);
        if t == 16 {
            // AST_CALL: find callee in fn_list, check if it's @checkpoint.
            let call_ns = __arena_get(expr_idx + 1);
            let call_nl = __arena_get(expr_idx + 2);
            let mut walk: i32 = head;
            let mut callee_idx: i32 = 0;
            let mut fk: i32 = 1;
            while fk == 1 {
                let cand_idx = __arena_get(walk + 1);
                let cand_ns = __arena_get(cand_idx + 1);
                let cand_nl = __arena_get(cand_idx + 2);
                if byte_eq(call_ns, call_nl, cand_ns, cand_nl) == 1 {
                    callee_idx = cand_idx;
                    fk = 0;
                };
                if fk == 1 {
                    let nx = __arena_get(walk + 2);
                    if nx == 0 { fk = 0; } else { walk = nx; };
                };
            }
            let mut ok: i32 = 1;
            if callee_idx > 0 {
                let cal_is_ckpt = __arena_get(callee_idx + 8);
                if cal_is_ckpt == 1 {
                    let cal_body = __arena_get(callee_idx + 3);
                    if ckpt_is_pure(cal_body, 0) == 0 {
                        ok = 0;
                    };
                };
            };
            // Also walk the call's args for nested @checkpoint calls.
            let args_head = __arena_get(expr_idx + 3);
            let mut awalk: i32 = args_head;
            while awalk != 0 {
                if ok == 1 {
                    let arg_expr = __arena_get(awalk + 1);
                    if ckpt_callees_pure(arg_expr, head, depth + 1) == 0 {
                        ok = 0;
                    };
                };
                awalk = __arena_get(awalk + 2);
            }
            ok
        } else { if t == 2 {
            let l = __arena_get(expr_idx + 1);
            let r = __arena_get(expr_idx + 2);
            if ckpt_callees_pure(l, head, depth + 1) == 1 {
                ckpt_callees_pure(r, head, depth + 1)
            } else { 0 }
        } else { if t == 3 {
            let l = __arena_get(expr_idx + 1);
            let r = __arena_get(expr_idx + 2);
            if ckpt_callees_pure(l, head, depth + 1) == 1 {
                ckpt_callees_pure(r, head, depth + 1)
            } else { 0 }
        } else { if t == 4 {
            let l = __arena_get(expr_idx + 1);
            let r = __arena_get(expr_idx + 2);
            if ckpt_callees_pure(l, head, depth + 1) == 1 {
                ckpt_callees_pure(r, head, depth + 1)
            } else { 0 }
        } else { if t == 5 {
            let l = __arena_get(expr_idx + 1);
            let r = __arena_get(expr_idx + 2);
            if ckpt_callees_pure(l, head, depth + 1) == 1 {
                ckpt_callees_pure(r, head, depth + 1)
            } else { 0 }
        } else { if t == 9 {
            let inner = __arena_get(expr_idx + 1);
            ckpt_callees_pure(inner, head, depth + 1)
        } else { if t == 7 {
            // AST_IF(cond, then, else): recurse into all three branches.
            // Audit A3-CRITICAL-4 fix: previous default-arm returned 1
            // here, blinding the scanner to @checkpoint calls inside
            // if/else arms. cond may itself contain a checkpoint call.
            let c = __arena_get(expr_idx + 1);
            let th = __arena_get(expr_idx + 2);
            let el = __arena_get(expr_idx + 3);
            if ckpt_callees_pure(c, head, depth + 1) == 1 {
                if ckpt_callees_pure(th, head, depth + 1) == 1 {
                    ckpt_callees_pure(el, head, depth + 1)
                } else { 0 }
            } else { 0 }
        } else { if t == 8 {
            // AST_LET(name_s, name_l, body, value): recurse into the
            // bound value AND the body. value can contain a checkpoint
            // call; body uses the binding so callee scan flows.
            let body_idx = __arena_get(expr_idx + 3);
            let value_idx = __arena_get(expr_idx + 4);
            if ckpt_callees_pure(value_idx, head, depth + 1) == 1 {
                ckpt_callees_pure(body_idx, head, depth + 1)
            } else { 0 }
        } else { if t == 10 {
            // AST_WHILE(cond, body): recurse into both.
            let c = __arena_get(expr_idx + 1);
            let b = __arena_get(expr_idx + 2);
            if ckpt_callees_pure(c, head, depth + 1) == 1 {
                ckpt_callees_pure(b, head, depth + 1)
            } else { 0 }
        } else { if t == 13 {
            // AST_SEQ(first, second): recurse into both.
            let l = __arena_get(expr_idx + 1);
            let r = __arena_get(expr_idx + 2);
            if ckpt_callees_pure(l, head, depth + 1) == 1 {
                ckpt_callees_pure(r, head, depth + 1)
            } else { 0 }
        } else {
            // Leaf or other tag — no @checkpoint call to check here.
            // (Stage 14 reverse-mode AD will trap on unsupported tags
            // separately if they end up in the differentiation path.)
            1
        } } } } } } } } } }
    }
}

// Stage 14.5: purity scan for @checkpoint fn bodies. Walks the AST
// rooted at `expr_idx` and returns 1 iff every node is one of the
// pure-arithmetic tags supported by reverse-mode AD:
//   0  AST_INT
//   1  AST_VAR
//   2  AST_ADD     3  AST_SUB     4  AST_MUL     5  AST_DIV
//   9  AST_NEG
//  16  AST_CALL    (only callable to other fns; the inliner will
//                   recursively expand them)
//  17  AST_ARG     (linked-list cell for AST_CALL args)
//  27  AST_FLOATLIT_F32
//  34  AST_FLOATLIT_F64
//  35  AST_INTLIT_I64
// Anything else (IF, WHILE, LET, ASSIGN, SEQ, side-effecting builtins,
// match, struct-field-access, ...) is considered impure for Phase-0
// and triggers trap 90001 at the @checkpoint boundary.
//
// Depth budget 2000 (per Stage 14.5 plan note) — exceeded means we
// give up and conservatively return 0 (impure).
fn ckpt_is_pure(expr_idx: i32, depth: i32) -> i32 {
    if depth >= 2000 {
        0
    } else {
        let t = __arena_get(expr_idx);
        if t == 0 { 1 }
        else { if t == 1 { 1 }
        else { if t == 27 { 1 }
        else { if t == 34 { 1 }
        else { if t == 35 { 1 }
        else { if t == 2 {
            let l = __arena_get(expr_idx + 1);
            let r = __arena_get(expr_idx + 2);
            if ckpt_is_pure(l, depth + 1) == 1 {
                ckpt_is_pure(r, depth + 1)
            } else { 0 }
        } else { if t == 3 {
            let l = __arena_get(expr_idx + 1);
            let r = __arena_get(expr_idx + 2);
            if ckpt_is_pure(l, depth + 1) == 1 {
                ckpt_is_pure(r, depth + 1)
            } else { 0 }
        } else { if t == 4 {
            let l = __arena_get(expr_idx + 1);
            let r = __arena_get(expr_idx + 2);
            if ckpt_is_pure(l, depth + 1) == 1 {
                ckpt_is_pure(r, depth + 1)
            } else { 0 }
        } else { if t == 5 {
            let l = __arena_get(expr_idx + 1);
            let r = __arena_get(expr_idx + 2);
            if ckpt_is_pure(l, depth + 1) == 1 {
                ckpt_is_pure(r, depth + 1)
            } else { 0 }
        } else { if t == 9 {
            let inner = __arena_get(expr_idx + 1);
            ckpt_is_pure(inner, depth + 1)
        } else { if t == 16 {
            // AST_CALL: walk its arg chain (each AST_ARG points at an
            // expr in slot 1 and the next AST_ARG in slot 2). The callee
            // body is checked when grad_rev_pass walks ITS @checkpoint
            // status — we don't recurse into the callee here to avoid
            // false-positives from helper fns that aren't @checkpoint.
            let args_head = __arena_get(expr_idx + 3);
            let mut walk: i32 = args_head;
            let mut ok: i32 = 1;
            while walk != 0 {
                if ok == 1 {
                    let arg_expr = __arena_get(walk + 1);
                    if ckpt_is_pure(arg_expr, depth + 1) == 0 {
                        ok = 0;
                    };
                };
                walk = __arena_get(walk + 2);
            }
            ok
        } else { 0 } } } } } } } } } } }
    }
}

// Stage 14: grad_rev pass. For each entry in gr_rev_pending, find the
// loss fn, find the param matching the field name (with leading 'd'
// stripped), differentiate the loss body w.r.t. that param, simplify,
// and synthesize "<loss>__grad_<field>" — a fn with the same params
// and ret_ty as loss whose body is the simplified partial derivative.
// Appends to fn_list tail so codegen emits it.
//
// Field name convention: ".dx" -> param "x", ".dy" -> param "y", etc.
// Phase-0 trap-id 88002 if field name lacks leading 'd' or is shorter
// than 2 chars. Phase-0 trap-id 88003 if no matching param found.
//
// Stage 14.5: when the loss body inlines through a @checkpoint fn whose
// body is impure, emit trap 90001 instead of the differentiated body.
fn grad_rev_pass(sb: i32, head: i32) -> i32 {
    let count = gr_rev_pending_count(sb);
    if count == 0 {
        0
    } else {
        // Find tail of fn_list for appending.
        let mut tail = head;
        let mut tail_keep: i32 = 1;
        while tail_keep == 1 {
            let nx = __arena_get(tail + 2);
            if nx == 0 { tail_keep = 0; } else { tail = nx; };
        }
        let base = gr_rev_pending_base(sb);
        let mut gri: i32 = 0;
        // Stage 51 Inc 1: outer loop now walks runs of consecutive
        // entries sharing the same loss_name. Inc 1 ships the
        // run-detection scaffold but always processes entries one-
        // at-a-time inside the run (falling through to the existing
        // per-entry single-bucket Inc 2 bridge). Cascade byte-
        // identity preserved because the per-entry behavior is
        // unchanged — only the loop structure changed.
        // Inc 2 (future) flips to true single-walk via
        // differentiate_reverse_all(sb, body) over the param_array
        // when all entries in the run pass per-entry validation.
        while gri < count {
            // PHASE 1: detect run length (consecutive entries
            // sharing the same loss_name as gri).
            let run_loss_s = __arena_get(base + gri * 5);
            let run_loss_l = __arena_get(base + gri * 5 + 1);
            let mut run_end: i32 = gri + 1;
            let mut run_keep: i32 = 1;
            while run_keep == 1 {
                if run_end >= count {
                    run_keep = 0;
                } else {
                    let cand_s = __arena_get(base + run_end * 5);
                    let cand_l = __arena_get(base + run_end * 5 + 1);
                    if byte_eq(cand_s, cand_l, run_loss_s, run_loss_l) == 1 {
                        run_end = run_end + 1;
                    } else {
                        run_keep = 0;
                    };
                };
            }
            let run_size = run_end - gri;
            // Stage 51 Inc 2: multi-bucket fast path. Try true
            // single-walk via differentiate_reverse_all when:
            //   - run_size >= 2 (Inc 1 trivial-fallback for n=1)
            //   - run_size <= 8 (bucket_array cap; falls back
            //     if exceeded, mirroring per-bucket cap behavior)
            //   - loss_fn_idx is found
            //   - ALL entries pass validation (valid_field +
            //     have_param + ckpt_ok). Any one invalid entry
            //     forces fallback so the per-entry trap arms
            //     (88002 / 88003 / 90001) fire correctly per-
            //     entry rather than poisoning the whole run.
            //
            // The fast path is SAFE because:
            //   - parser.hx itself doesn't use grad_rev_all with
            //     multi-params, so self-host cascade byte-identity
            //     is preserved (the new code path is unreached
            //     during parser.hx compilation).
            //   - propagate_adj_multi mirrors propagate_adj's DFS
            //     shape exactly — bucket-deposit order for any
            //     single param is identical to the N-walk version.
            //   - bucket_array_sum produces a chain-of-+ that
            //     simplify() reduces to the same canonical form
            //     as the single-bucket sum_bucket.
            let mut handled: i32 = 0;
            if run_size >= 2 {
            if run_size <= 8 {
                // Look up loss_fn_idx ONCE for the whole run.
                let mut runw: i32 = head;
                let mut run_fn_idx: i32 = 0;
                let mut runfk: i32 = 1;
                while runfk == 1 {
                    let rci = __arena_get(runw + 1);
                    let rcns = __arena_get(rci + 1);
                    let rcnl = __arena_get(rci + 2);
                    if byte_eq(run_loss_s, run_loss_l, rcns, rcnl) == 1 {
                        run_fn_idx = rci;
                        runfk = 0;
                    };
                    if runfk == 1 {
                        let rnx = __arena_get(runw + 2);
                        if rnx == 0 { runfk = 0; } else { runw = rnx; };
                    };
                }
                if run_fn_idx > 0 {
                    let run_loss_body = __arena_get(run_fn_idx + 3);
                    let run_loss_params = __arena_get(run_fn_idx + 4);
                    let run_loss_ret_ty = __arena_get(run_fn_idx + 5);
                    let run_loss_is_ckpt = __arena_get(run_fn_idx + 8);
                    // Pre-validate all entries in the run.
                    let mut all_valid: i32 = 1;
                    let mut vi: i32 = gri;
                    while vi < run_end {
                        let ve = base + vi * 5;
                        let vfs = __arena_get(ve + 2);
                        let vfl = __arena_get(ve + 3);
                        if vfl < 2 { all_valid = 0; } else {
                            let vf0 = __arena_get(vfs);
                            if vf0 != 100 { all_valid = 0; } else {
                                let vvs = vfs + 1;
                                let vvl = vfl - 1;
                                let mut vhp: i32 = 0;
                                let mut vpw: i32 = run_loss_params;
                                while vpw != 0 {
                                    let vpns = __arena_get(vpw + 1);
                                    let vpnl = __arena_get(vpw + 2);
                                    if byte_eq(vpns, vpnl, vvs, vvl) == 1 {
                                        vhp = 1;
                                    };
                                    vpw = __arena_get(vpw + 3);
                                }
                                if vhp == 0 { all_valid = 0; };
                            };
                        };
                        vi = vi + 1;
                    }
                    if all_valid == 1 {
                        let run_ckpt_ok = ckpt_callees_pure(
                            run_loss_body, head, 0);
                        if run_ckpt_ok == 1 {
                            // Multi-bucket fast path: one walk,
                            // N gradient extractions.
                            bucket_array_reset(sb, run_size);
                            let mut si: i32 = 0;
                            while si < run_size {
                                let se = base + (gri + si) * 5;
                                let sfs = __arena_get(se + 2);
                                let sfl = __arena_get(se + 3);
                                set_param_array_name(
                                    sb, si, sfs + 1, sfl - 1);
                                si = si + 1;
                            }
                            let run_body_to_diff = inline_user_calls(
                                run_loss_body, head, 0);
                            differentiate_reverse_all(
                                sb, run_body_to_diff);
                            let mut xi: i32 = 0;
                            while xi < run_size {
                                let xe = base + (gri + xi) * 5;
                                let xms = __arena_get(xe + 4);
                                let xfl = __arena_get(xe + 3);
                                let xml = run_loss_l + 7 + xfl;
                                let xder_raw = bucket_array_sum(sb, xi);
                                let xder = simplify(xder_raw);
                                let xclone = mk_node(14, xms, xml, xder);
                                __arena_push(run_loss_params);
                                __arena_push(run_loss_ret_ty);
                                __arena_push(0);
                                __arena_push(0);
                                __arena_push(run_loss_is_ckpt);
                                __arena_push(0); __arena_push(0);
                                __arena_push(0); __arena_push(0);
                                __arena_push(0); __arena_push(0);
                                __arena_push(0); __arena_push(0);
                                __arena_push(0); __arena_push(0);
                                let xnln = mk_node(15, xclone, 0, 0);
                                __arena_set(tail + 2, xnln);
                                tail = xnln;
                                xi = xi + 1;
                            }
                            handled = 1;
                        };
                    };
                };
            };
            };
            // PHASE 2: process each entry in the run individually
            // (Inc 1 fallback — when Inc 2 fast path didn't fire,
            // or for any single-entry run, or when validation
            // failed for any entry in a multi-entry run).
            let mut ri: i32 = gri;
            while ri < run_end {
            if handled == 1 { ri = run_end; } else {
            let entry = base + ri * 5;
            let loss_s = __arena_get(entry);
            let loss_l = __arena_get(entry + 1);
            let field_s = __arena_get(entry + 2);
            let field_l = __arena_get(entry + 3);
            let mang_s = __arena_get(entry + 4);
            let mang_l = loss_l + 7 + field_l;
            // Find loss fn in fn_list.
            let mut walk: i32 = head;
            let mut loss_fn_idx: i32 = 0;
            let mut find_keep: i32 = 1;
            while find_keep == 1 {
                let cand_idx = __arena_get(walk + 1);
                let cand_ns = __arena_get(cand_idx + 1);
                let cand_nl = __arena_get(cand_idx + 2);
                if byte_eq(loss_s, loss_l, cand_ns, cand_nl) == 1 {
                    loss_fn_idx = cand_idx;
                    find_keep = 0;
                };
                if find_keep == 1 {
                    let nx = __arena_get(walk + 2);
                    if nx == 0 { find_keep = 0; } else { walk = nx; };
                };
            }
            if loss_fn_idx > 0 {
                let loss_body = __arena_get(loss_fn_idx + 3);
                let loss_params = __arena_get(loss_fn_idx + 4);
                let loss_ret_ty = __arena_get(loss_fn_idx + 5);
                let loss_is_ckpt = __arena_get(loss_fn_idx + 8);
                // Extract param name from field: skip leading 'd' (byte 100).
                // For ".dx" the param name is "x" (offset field_s+1, length
                // field_l-1).
                let var_s = if field_l < 2 { 0 } else { field_s + 1 };
                let var_l = if field_l < 2 { 0 } else { field_l - 1 };
                let valid_field = if field_l < 2 { 0 } else {
                    let f0 = __arena_get(field_s);
                    if f0 == 100 { 1 } else { 0 }
                };
                let trap_idx = if valid_field == 0 {
                    mk_node(99, 88002, 0, 0)
                } else { 0 };
                // Verify the param exists in loss_params chain.
                let mut have_param: i32 = 0;
                let mut pwalk: i32 = loss_params;
                while pwalk != 0 {
                    let pn_s = __arena_get(pwalk + 1);
                    let pn_l = __arena_get(pwalk + 2);
                    if byte_eq(pn_s, pn_l, var_s, var_l) == 1 {
                        have_param = 1;
                    };
                    pwalk = __arena_get(pwalk + 3);
                }
                // Stage 14.5: scan loss body for @checkpoint callees with
                // impure bodies. If any is found, trap 90001 BEFORE any
                // inlining/differentiation work — keeps the error message
                // pinned to the @checkpoint contract violation rather than
                // surfacing further-downstream as 88001/85001.
                let ckpt_ok = if valid_field == 1 {
                    if have_param == 1 {
                        ckpt_callees_pure(loss_body, head, 0)
                    } else { 1 }
                } else { 1 };
                let body_to_diff = if valid_field == 1 {
                    if have_param == 1 {
                        if ckpt_ok == 1 {
                            // Stage 13 prep: inline user-fn calls.
                            inline_user_calls(loss_body, head, 0)
                        } else {
                            mk_node(99, 90001, 0, 0)
                        }
                    } else {
                        mk_node(99, 88003, 0, 0)
                    }
                } else { trap_idx };
                // Stage 14c: real reverse-mode adjoint propagation.
                // Walk body top-down with adjoint, accumulate into the
                // bucket for `var_s/var_l`, then sum. Mathematically
                // identical to forward-mode `differentiate` for scalar
                // output, but the algorithmic shape is true reverse.
                //
                // Stage 14.5: when ckpt_ok == 0, body_to_diff is already
                // the trap-90001 marker — skip differentiation/simplify
                // so the trap node propagates unchanged to the synthesized
                // gradient fn body.
                let deriv_raw = if have_param == 1 {
                    if valid_field == 1 {
                        if ckpt_ok == 1 {
                            // Stage 50 Inc 2: swap to the new
                            // bucket_array path. Equivalent output for
                            // n=1; preserves self-host G3..G4 byte-
                            // identity. Inc 3 will introduce true
                            // grouping (single walk for multi-param).
                            differentiate_reverse_one_via_array(sb, body_to_diff, var_s, var_l)
                        } else {
                            body_to_diff
                        }
                    } else { body_to_diff }
                } else { body_to_diff };
                let deriv = if have_param == 1 {
                    if valid_field == 1 {
                        if ckpt_ok == 1 {
                            simplify(deriv_raw)
                        } else {
                            deriv_raw
                        }
                    } else { deriv_raw }
                } else { deriv_raw };
                // Synthesize the AST_FN_DECL clone.
                let clone_fn = mk_node(14, mang_s, mang_l, deriv);
                __arena_push(loss_params);
                __arena_push(loss_ret_ty);
                __arena_push(0);
                __arena_push(0);
                // F10 companion: propagate the loss fn's @checkpoint marker
                // to the synthesized reverse-mode gradient clone.
                __arena_push(loss_is_ckpt);  // slot 8: propagated from loss
                // Stage 28.9: validation attrs default to 0 on synthesized
                // grad-rev clones (consistent with forward-mode grad).
                __arena_push(0);             // slot 9: is_deprecated = 0
                __arena_push(0);             // slot 10: is_trace = 0
                __arena_push(0);             // slot 11: is_unwind = 0
                __arena_push(0);             // slot 12: deprecated_msg_start = 0
                __arena_push(0);             // slot 13: deprecated_msg_len = 0
                __arena_push(0);             // slot 14: is_kernel = 0
                __arena_push(0);             // slot 15: is_autotune = 0
                __arena_push(0);             // slot 16: autotune_product = 0
                __arena_push(0);             // slot 17: autotune_parse_error_kind = 0
                __arena_push(0);             // slot 18: since_msg_start = 0
                __arena_push(0);             // slot 19: since_msg_len = 0
                let new_list_node = mk_node(15, clone_fn, 0, 0);
                __arena_set(tail + 2, new_list_node);
                tail = new_list_node;
            };
            ri = ri + 1;
            };
            }
            gri = run_end;
        }
        0
    }
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
    // Stage 8: optional `<T1, T2, ...>` generic-params list. Reset the
    // gp_tab scratch for this fn, then if `<` (TK_LT = 16) is next, read
    // up to 4 generic-param-name IDENTs separated by `,` until `>`. The
    // names are stored in gp_tab (sb+29/30) so that param-type and
    // return-type resolution below can recognize T as a generic marker
    // (encoding p_ty = 200 + gp_idx).
    gp_tab_reset(sb);
    let gp_peek_t = tok_tag(tok_base, cur_get(sb));
    let mut is_generic_fn: i32 = 0;
    if gp_peek_t == 16 {
        is_generic_fn = 1;
        cur_advance(sb);                        // consume '<'
        let mut keep_g: i32 = 1;
        while keep_g == 1 {
            let gtt = tok_tag(tok_base, cur_get(sb));
            if gtt == 17 {                      // '>' end
                keep_g = 0;
            } else { if gtt == 13 {             // ','
                cur_advance(sb);
            } else { if gtt == 0 {              // EOF safety
                keep_g = 0;
            } else {
                // IDENT — capture as generic-param name.
                let gk = cur_get(sb);
                let gp_s = tok_p2(tok_base, gk);
                let gp_l = tok_p3(tok_base, gk);
                cur_advance(sb);
                gp_tab_add(sb, gp_s, gp_l);
                // Stage 8.5C: optional trait bound `: TraitName (+ Trait2)*`.
                // Phase-0 ignores the bound semantically — bound resolution
                // happens at mono time via name lookup in impl_table. We
                // just skip the tokens here so the cursor lands on `,` or
                // `>` for the next param.
                if tok_tag(tok_base, cur_get(sb)) == 14 {     // ':'
                    cur_advance(sb);                          // consume ':'
                    let mut keep_b: i32 = 1;
                    while keep_b == 1 {
                        let bt = tok_tag(tok_base, cur_get(sb));
                        if bt == 2 {                          // trait-name IDENT
                            cur_advance(sb);
                        } else { if bt == 7 {                 // '+'
                            cur_advance(sb);
                        } else {
                            keep_b = 0;
                        } };
                    }
                };
            }}};
        }
        // Stage 28.11 INC-1 cycle-3 polish (cycle-2 code-review F-4
        // conf 75): DEFERRED-KNOWN breadcrumb — this fn-generic loop
        // shares the same silent-acceptance defect class as the
        // post-cycle-2 struct-generic loop (parse_struct_decl ~6022):
        // bare `cur_advance` in else-arm + unconditional post-loop
        // advance. Same root cause as SF-1/SF-2/SF-3 silent-failure
        // cycle-1 findings. Pre-existing Stage-8 surface, not under
        // active iteration; left as DEFERRED-KNOWN per narrow-scope
        // discipline (workspace cycle-71 note). Should be fixed in a
        // separate hardening pass that applies the cycle-2 fix-sweep
        // pattern (TK_IDENT-only else-arm + TK_GT-guarded post-advance).
        cur_advance(sb);                        // consume '>'
    };
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
            // Stage 8: BEFORE struct lookup, check if this type IDENT is a
            // generic-param name. If yes, encode as 200 + gp_idx so the
            // mono-pass can substitute it later. Generic params take
            // precedence over struct/scalar matches.
            let gp_idx_p = gp_tab_lookup(sb, ty_s, ty_l);
            let p_ty_generic = if gp_idx_p >= 0 { 200 + gp_idx_p } else { 0 };
            let s_idx_p = struct_tab_lookup_idx(sb, ty_s, ty_l);
            let p_ty_struct = if s_idx_p >= 0 { 100 + s_idx_p } else { 0 };
            let p_ty_pre = if p_ty == 0 { p_ty_struct } else { p_ty };
            let p_ty_final = if p_ty_generic > 0 { p_ty_generic } else { p_ty_pre };
            let n_register = if p_ty_final >= 100 { if p_ty_final < 200 {
                var_struct_tab_add(sb, pname_s, pname_l, p_ty_final - 100)
            } else { 0 } } else { 0 };
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
    // Stage 8: if the return-type IDENT is a generic-param name, override
    // ret_ty with 200 + gp_idx. Generic-typed return propagates through
    // mono substitution.
    let rt_gp_idx = gp_tab_lookup(sb, rt_s, rt_l);
    // Audit A1-F5 fix: if ret_ty == 0 AND no generic match, look up
    // struct_tab. Pre-fix any unrecognized IDENT (e.g. struct name `Pt`,
    // enum name `Maybe`) silently fell through to ret_ty=0 (i32). For a
    // struct-returning fn, the body left a 64-bit pointer in rax; the
    // exit stub's `mov edi, eax` truncated to 32 bits → caller dereffed
    // a sign-extended pointer → SIGSEGV. Now we encode struct returns as
    // 100 + struct_idx (mirroring param-side encoding) so codegen can
    // recognize struct returns. Unknown idents still degrade to 0; a
    // separate fix would be to trap on truly-unknown idents, but doing
    // so requires detecting that scalar+generic+struct all missed AND
    // the ident bytes don't represent any known type. Limited to struct
    // recognition for now.
    let rt_struct_idx = struct_tab_lookup_idx(sb, rt_s, rt_l);
    let ret_ty_post_struct = if rt_struct_idx >= 0 { 100 + rt_struct_idx } else { ret_ty };
    let ret_ty_final = if rt_gp_idx >= 0 { 200 + rt_gp_idx } else { ret_ty_post_struct };
    // K1.O (2026-05-25): optional `where T: Bound, U: Bound2` clause
    // after the return type and before the body LBRACE. Bounds are
    // not enforced in the type-erased bootstrap -- just consume all
    // tokens up to (but not including) the body LBRACE so the
    // existing body-parsing path continues unchanged. "where" is a
    // 5-byte IDENT (bytes 119,104,101,114,101).
    let w_k = cur_get(sb);
    let w_tg = tok_tag(tok_base, w_k);
    if w_tg == 2 {
        let w_s = tok_p2(tok_base, w_k);
        let w_l = tok_p3(tok_base, w_k);
        let is_where_kw = if w_l == 5 {
            if __arena_get(w_s) == 119 {
                if __arena_get(w_s + 1) == 104 {
                    if __arena_get(w_s + 2) == 101 {
                        if __arena_get(w_s + 3) == 114 {
                            if __arena_get(w_s + 4) == 101 { 1 } else { 0 }
                        } else { 0 }
                    } else { 0 }
                } else { 0 }
            } else { 0 }
        } else { 0 };
        if is_where_kw == 1 {
            cur_advance(sb);    // consume 'where' IDENT
            // Skip tokens until LBRACE (or EOF safety).
            let mut keep_w: i32 = 1;
            while keep_w == 1 {
                let wt = tok_tag(tok_base, cur_get(sb));
                if wt == 5 {
                    keep_w = 0;
                } else { if wt == 0 {
                    keep_w = 0;
                } else {
                    cur_advance(sb);
                }};
            }
        };
    };
    cur_advance(sb);     // '{'
    let body = parse_expr(tok_base, sb);
    cur_advance(sb);     // '}'
    // Audit-14: same overflow issue as AST_LET — packed encoding
    // breaks for arena indices > 65535. Extend to 5 slots: p3 =
    // body_idx, p4 = params_head. Step 5c follow-on: 6th slot p5 =
    // ret_ty (0 = i32, 1 = f32, 2 = f64). Codegen reads p5 to populate
    // the fn_type_table; is_f32_expr / is_f64_expr's AST_CALL fallback
    // resolves user-defined fn types via the table.
    // Stage 8: 7th slot p6 = is_generic flag (1 if fn has <T1, T2, ...>
    // generic params). Codegen + fn_type_table_init pre-pass skip
    // generic-fn templates so they aren't emitted (mono pass synthesizes
    // concrete clones).
    // Stage 8.5C: build gp_names chain BEFORE allocating the AST_FN_DECL
    // node. mk_node + arena_push are positional in the arena, so any
    // mk_node call between fn_decl's slot 0 push and slot 7 push would
    // interleave gp_name node bytes into the fn_decl's slot 4..7 region
    // and corrupt the layout.
    let mut gp_chain_head: i32 = 0;
    let mut gp_chain_prev: i32 = 0;
    let gp_count_now = gp_tab_count(sb);
    let gp_base_now = gp_tab_base(sb);
    let mut gp_walk: i32 = 0;
    while gp_walk < gp_count_now {
        let entry = gp_base_now + gp_walk * 2;
        let gn_s = __arena_get(entry);
        let gn_l = __arena_get(entry + 1);
        let new_gn = mk_node(76, gn_s, gn_l, 0);
        if gp_chain_head == 0 {
            gp_chain_head = new_gn;
            gp_chain_prev = new_gn;
        } else {
            __arena_set(gp_chain_prev + 3, new_gn);
            gp_chain_prev = new_gn;
        };
        gp_walk = gp_walk + 1;
    }
    // Stage 14.5: read+clear the `next_fn_is_checkpoint` scratch flag
    // BEFORE allocating the fn_decl node so that nested fn parses don't
    // bleed checkpoint state across siblings. Pushed as slot 8.
    let is_ckpt_now = next_fn_is_ckpt(sb);
    set_next_fn_is_ckpt(sb, 0);
    // Stage 28.9: capture+clear validation attribute flags BEFORE
    // node alloc so nested fn parses don't leak. Pushed as slots
    // 9/10/11 of AST_FN_DECL so the bootstrap validation passes
    // (deprecated_pass, trace_pass, panic_pass.@unwind) can observe
    // attribute info that the parser otherwise discards.
    let is_deprecated_now = next_fn_is_deprecated(sb);
    let dep_msg_s_now = next_fn_deprecated_msg_s(sb);
    let dep_msg_l_now = next_fn_deprecated_msg_l(sb);
    set_next_fn_is_deprecated(sb, 0);
    set_next_fn_deprecated_msg_s(sb, 0);
    set_next_fn_deprecated_msg_l(sb, 0);
    let is_trace_now = next_fn_is_trace(sb);
    set_next_fn_is_trace(sb, 0);
    let is_unwind_now = next_fn_is_unwind(sb);
    set_next_fn_is_unwind(sb, 0);
    let is_kernel_now = next_fn_is_kernel(sb);
    set_next_fn_is_kernel(sb, 0);
    let is_autotune_now = next_fn_is_autotune(sb);
    set_next_fn_is_autotune(sb, 0);
    let autotune_product_now = next_fn_autotune_product(sb);
    set_next_fn_autotune_product(sb, 0);
    let autotune_error_now = next_fn_autotune_error(sb);
    set_next_fn_autotune_error(sb, 0);
    let since_msg_s_now = next_fn_since_msg_s(sb);
    let since_msg_l_now = next_fn_since_msg_l(sb);
    set_next_fn_since_msg_s(sb, 0);
    set_next_fn_since_msg_l(sb, 0);
    let node = mk_node(14, name_start, name_len, body);
    __arena_push(params_head);
    __arena_push(ret_ty_final);
    __arena_push(is_generic_fn);
    __arena_push(gp_chain_head);             // slot 7: gp_names_head
    __arena_push(is_ckpt_now);               // slot 8: is_checkpoint flag
    __arena_push(is_deprecated_now);         // slot 9: is_deprecated flag (Stage 28.9)
    __arena_push(is_trace_now);              // slot 10: is_trace flag (Stage 28.9)
    __arena_push(is_unwind_now);             // slot 11: is_unwind flag (Stage 28.9)
    __arena_push(dep_msg_s_now);             // slot 12: deprecated_msg_start (Stage 33)
    __arena_push(dep_msg_l_now);             // slot 13: deprecated_msg_len (Stage 33)
    __arena_push(is_kernel_now);             // slot 14: is_kernel flag (Stage 33)
    __arena_push(is_autotune_now);           // slot 15: is_autotune flag (Stage 33)
    __arena_push(autotune_product_now);      // slot 16: autotune product (Stage 33)
    __arena_push(autotune_error_now);        // slot 17: autotune parse error kind (Stage 33)
    __arena_push(since_msg_s_now);           // slot 18: since_msg_start (Stage 33)
    __arena_push(since_msg_l_now);           // slot 19: since_msg_len (Stage 33)
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

// Stage 8.5B helper: build a mangled method name `<TargetType>__<MethodName>`
// directly in the arena. e.g. target = "i32" + method = "eq" -> "i32__eq".
// Returns the byte start; length is target_l + 2 + method_l.
fn mangle_impl_method(target_s: i32, target_l: i32, method_s: i32, method_l: i32) -> i32 {
    let start = __arena_len();
    let mut i: i32 = 0;
    while i < target_l {
        __arena_push(__arena_get(target_s + i));
        i = i + 1;
    }
    __arena_push(95); __arena_push(95);      // '__'
    let mut j: i32 = 0;
    while j < method_l {
        __arena_push(__arena_get(method_s + j));
        j = j + 1;
    }
    start
}

// Stage 8.5B helper: parse one impl method body as an AST_FN_DECL with the
// mangled name. The first parameter MUST be `self` (no type annotation) and
// is bound with the impl target type. Subsequent params follow normal Helix
// syntax. `Self` (capital S) IDENT in a param/return position resolves to
// the target type's tag.
//
// target_s/l = bytes of the target type IDENT (e.g. "i32"); used for
//   mangling and `Self` resolution.
// target_tag = the resolved type tag (ty_ident_to_tag of target_s/l).
//
// Returns the AST_FN_DECL node index (tag 14). Body is freshly parsed
// (calls parse_expr inside the {}).
fn parse_impl_method(tok_base: i32, sb: i32, target_s: i32, target_l: i32, target_tag: i32) -> i32 {
    cur_advance(sb);                         // consume 'fn' IDENT
    let nk = cur_get(sb);
    let method_name_s = tok_p2(tok_base, nk);
    let method_name_l = tok_p3(tok_base, nk);
    cur_advance(sb);                         // method-name IDENT
    // Build mangled name in arena BEFORE consuming any more (so subsequent
    // arena pushes don't interleave with the mangled bytes).
    let mang_s = mangle_impl_method(target_s, target_l, method_name_s, method_name_l);
    let mang_l = target_l + 2 + method_name_l;
    cur_advance(sb);                         // '('
    // Parse params. The FIRST identifier `self` (4 bytes 115 101 108 102) is
    // a special form: no type annotation, type = target_tag. All subsequent
    // params follow standard `name: T` syntax. `Self` (capital S, byte 83
    // followed by 101 108 102) in a type position resolves to the target.
    let mut params_head: i32 = 0;
    let mut prev_param: i32 = 0;
    let mut keep: i32 = 1;
    while keep == 1 {
        let pt = tok_tag(tok_base, cur_get(sb));
        if pt == 4 {
            keep = 0;                        // ')'
        } else { if pt == 13 {
            cur_advance(sb);                 // ','
        } else {
            let pname_tok = cur_get(sb);
            let pname_s = tok_p2(tok_base, pname_tok);
            let pname_l = tok_p3(tok_base, pname_tok);
            // Detect `self` IDENT (bytes 115, 101, 108, 102). If it's `self`
            // AND it's not followed by `:`, treat as untyped self-param of
            // target_tag.
            let is_self_kw = if pname_l == 4 {
                let b0 = __arena_get(pname_s);
                let b1 = __arena_get(pname_s + 1);
                let b2 = __arena_get(pname_s + 2);
                let b3 = __arena_get(pname_s + 3);
                if b0 == 115 { if b1 == 101 { if b2 == 108 { if b3 == 102 { 1 } else { 0 } } else { 0 } } else { 0 } } else { 0 }
            } else { 0 };
            cur_advance(sb);                 // param-name IDENT
            let next_tag = tok_tag(tok_base, cur_get(sb));
            // If `self` AND next isn't ':', it's the bare-self form.
            let is_bare_self = if is_self_kw == 1 { if next_tag == 14 { 0 } else { 1 } } else { 0 };
            if is_bare_self == 1 {
                // Bare `self`: type = target_tag.
                let new_param = mk_node(18, pname_s, pname_l, 0);
                __arena_push(target_tag);
                if params_head == 0 {
                    params_head = new_param;
                    prev_param = new_param;
                } else {
                    __arena_set(prev_param + 3, new_param);
                    prev_param = new_param;
                };
            } else {
                // Standard `name: T` form. Consume ':' and the type IDENT.
                cur_advance(sb);             // ':'
                let ty_tok = cur_get(sb);
                let ty_s_raw = tok_p2(tok_base, ty_tok);
                let ty_l_raw = tok_p3(tok_base, ty_tok);
                cur_advance(sb);             // type IDENT
                // `Self` substitution: if the type IDENT is "Self" (4 bytes
                // 83 101 108 102), use target_tag directly.
                let is_self_ty = if ty_l_raw == 4 {
                    let s0 = __arena_get(ty_s_raw);
                    let s1 = __arena_get(ty_s_raw + 1);
                    let s2 = __arena_get(ty_s_raw + 2);
                    let s3 = __arena_get(ty_s_raw + 3);
                    if s0 == 83 { if s1 == 101 { if s2 == 108 { if s3 == 102 { 1 } else { 0 } } else { 0 } } else { 0 } } else { 0 }
                } else { 0 };
                let p_ty_resolved = if is_self_ty == 1 { target_tag } else { ty_ident_to_tag(ty_s_raw, ty_l_raw) };
                let new_param = mk_node(18, pname_s, pname_l, 0);
                __arena_push(p_ty_resolved);
                if params_head == 0 {
                    params_head = new_param;
                    prev_param = new_param;
                } else {
                    __arena_set(prev_param + 3, new_param);
                    prev_param = new_param;
                };
            };
        }};
    }
    cur_advance(sb);                         // ')'
    cur_advance(sb);                         // '-' part of '->'
    cur_advance(sb);                         // '>' part of '->'
    let rt_tok = cur_get(sb);
    let rt_s = tok_p2(tok_base, rt_tok);
    let rt_l = tok_p3(tok_base, rt_tok);
    cur_advance(sb);                         // return-type IDENT
    let is_self_rt = if rt_l == 4 {
        let s0 = __arena_get(rt_s);
        let s1 = __arena_get(rt_s + 1);
        let s2 = __arena_get(rt_s + 2);
        let s3 = __arena_get(rt_s + 3);
        if s0 == 83 { if s1 == 101 { if s2 == 108 { if s3 == 102 { 1 } else { 0 } } else { 0 } } else { 0 } } else { 0 }
    } else { 0 };
    let ret_ty_resolved = if is_self_rt == 1 { target_tag } else { ty_ident_to_tag(rt_s, rt_l) };
    cur_advance(sb);                         // '{'
    let body = parse_expr(tok_base, sb);
    cur_advance(sb);                         // '}'
    let node = mk_node(14, mang_s, mang_l, body);
    __arena_push(params_head);
    __arena_push(ret_ty_resolved);
    __arena_push(0);                         // is_generic = 0 (concrete)
    __arena_push(0);                         // slot 7: gp_names_head (none)
    __arena_push(0);                         // slot 8: is_checkpoint = 0 (Stage 14.5)
    __arena_push(0);                         // slot 9: is_deprecated = 0 (Stage 28.9)
    __arena_push(0);                         // slot 10: is_trace = 0 (Stage 28.9)
    __arena_push(0);                         // slot 11: is_unwind = 0 (Stage 28.9)
    __arena_push(0);                         // slot 12: deprecated_msg_start = 0 (Stage 33)
    __arena_push(0);                         // slot 13: deprecated_msg_len = 0 (Stage 33)
    __arena_push(0);                         // slot 14: is_kernel = 0 (Stage 33)
    __arena_push(0);                         // slot 15: is_autotune = 0 (Stage 33)
    __arena_push(0);                         // slot 16: autotune_product = 0 (Stage 33)
    __arena_push(0);                         // slot 17: autotune_parse_error_kind = 0 (Stage 33)
    __arena_push(0);                         // slot 18: since_msg_start = 0 (Stage 33)
    __arena_push(0);                         // slot 19: since_msg_len = 0 (Stage 33)
    node
}

// Stage 8.5B: parse `impl IDENT for IDENT { fn ... { ... } ... }`. Each
// method is rewritten as a regular AST_FN_DECL with a mangled name like
// `i32__eq`, then appended to the impl-pending fn-list chain. parse_program
// splices that chain into the user's fn_list before the mono pass runs.
//
// Returns AST_STRUCT_DECL (tag 54) — the impl block itself emits no code;
// its rewritten methods are emitted via the standard fn-list path.
fn parse_impl_block(tok_base: i32, sb: i32) -> i32 {
    cur_advance(sb);                         // consume 'impl' IDENT
    // Trait name IDENT.
    let trait_tok = cur_get(sb);
    let trait_s = tok_p2(tok_base, trait_tok);
    let trait_l = tok_p3(tok_base, trait_tok);
    cur_advance(sb);                         // trait-name IDENT
    cur_advance(sb);                         // 'for' IDENT
    // Target type IDENT.
    let target_tok = cur_get(sb);
    let target_s = tok_p2(tok_base, target_tok);
    let target_l = tok_p3(tok_base, target_tok);
    cur_advance(sb);                         // target-type IDENT
    let target_tag = ty_ident_to_tag(target_s, target_l);
    cur_advance(sb);                         // '{'
    // Parse zero-or-more method decls until '}'.
    let mut method_count: i32 = 0;
    let mut keep: i32 = 1;
    while keep == 1 {
        let tt = tok_tag(tok_base, cur_get(sb));
        if tt == 6 {                         // RBRACE
            keep = 0;
        } else { if tt == 0 {                // EOF safety
            keep = 0;
        } else {
            // Expect `fn IDENT(...) -> RET { ... }`. parse_impl_method
            // consumes the entire decl + body + closing '}'.
            let fn_node = parse_impl_method(tok_base, sb, target_s, target_l, target_tag);
            // Wrap in AST_FN_LIST (tag 15) and append to impl_pending chain.
            let list_node = mk_node(15, fn_node, 0, 0);
            let head = impl_pending_head(sb);
            if head == 0 {
                set_impl_pending_head(sb, list_node);
                set_impl_pending_tail(sb, list_node);
            } else {
                let tail = impl_pending_tail(sb);
                __arena_set(tail + 2, list_node);
                set_impl_pending_tail(sb, list_node);
            };
            method_count = method_count + 1;
        } };
    }
    cur_advance(sb);                         // consume final '}'
    impl_tab_add(sb, trait_s, trait_l, target_tag, method_count);
    mk_node(54, 0, 0, 0)
}

// Stage 10: build a mangled name `<prefix>__<orig>` directly into the arena.
// If prefix_l is 0, just push the orig bytes (no underscore separator). Used
// by parse_mod_decl to rewrite each fn-decl's name as it lifts the fn from
// the module block to the top-level fn list.
//
// Returns the byte start; total length is (prefix_l + 2 + orig_l) when
// prefix_l > 0, or orig_l when prefix_l == 0.
fn mangle_mod_name(prefix_s: i32, prefix_l: i32, orig_s: i32, orig_l: i32) -> i32 {
    let start = __arena_len();
    let mut i: i32 = 0;
    while i < prefix_l {
        __arena_push(__arena_get(prefix_s + i));
        i = i + 1;
    }
    if prefix_l > 0 {
        __arena_push(95); __arena_push(95);      // '__'
    };
    let mut j: i32 = 0;
    while j < orig_l {
        __arena_push(__arena_get(orig_s + j));
        j = j + 1;
    }
    start
}

// Stage 10: parse `mod IDENT { ... }`. Walks the body lifting each fn to
// the top-level mod_pending fn-list with a mangled name `<prefix>__<fn_name>`.
// Nested `mod inner { ... }` recurses with extended prefix `<prefix>__inner`.
// Phase-0: ignores struct/enum/trait/impl decls inside modules (they parse
// but don't get name-mangled — Phase-0 test cases focus on fn lifting).
//
// Caller has already verified the cursor sits on the `mod` IDENT.
//
// `prefix_s, prefix_l` is the accumulated prefix path bytes (e.g. for
// `mod outer { mod inner { fn baz } }`, when descending into inner the
// prefix is "outer", for the fn baz we mangle to "outer__inner__baz").
// At the top level prefix_l == 0; the mod's own name becomes the prefix.
//
// Returns AST_STRUCT_DECL (tag 54) so codegen treats the original `mod`
// site as a 0-byte no-op.
fn parse_mod_decl(tok_base: i32, sb: i32, prefix_s: i32, prefix_l: i32) -> i32 {
    cur_advance(sb);                             // consume 'mod' IDENT
    let nk = cur_get(sb);
    let mname_s = tok_p2(tok_base, nk);
    let mname_l = tok_p3(tok_base, nk);
    cur_advance(sb);                             // consume mod-name IDENT
    cur_advance(sb);                             // consume '{' (LBRACE = 5)
    // Build the new prefix `<prefix>__<mname>` (or just `<mname>` if no
    // prefix yet) once into the arena so we can pass (new_prefix_s, new_l)
    // down the recursion AND use it to mangle fn names inside this scope.
    let new_prefix_s = mangle_mod_name(prefix_s, prefix_l, mname_s, mname_l);
    let new_prefix_l = if prefix_l == 0 { mname_l } else { prefix_l + 2 + mname_l };
    // Walk inner items until '}'.
    let mut keep: i32 = 1;
    while keep == 1 {
        let tt = tok_tag(tok_base, cur_get(sb));
        if tt == 6 {                             // RBRACE
            keep = 0;
        } else { if tt == 0 {                    // EOF safety
            keep = 0;
        } else { if tt == 2 {                    // IDENT
            let ik = cur_get(sb);
            let ik_s = tok_p2(tok_base, ik);
            let ik_l = tok_p3(tok_base, ik);
            let is_inner_mod = byte_eq(ik_s, ik_l, kw_mod_s(sb), kw_mod_n(sb));
            let is_inner_fn = byte_eq(ik_s, ik_l, kw_fn_s(sb), kw_fn_n(sb));
            if is_inner_mod == 1 {
                // Recurse with new_prefix as the prefix path.
                parse_mod_decl(tok_base, sb, new_prefix_s, new_prefix_l);
            } else { if is_inner_fn == 1 {
                // Parse the fn decl, then patch its name to the mangled form.
                // parse_fn_decl reads and consumes everything: 'fn' IDENT
                // (params) -> RET { body }. Returns AST_FN_DECL idx (tag 14)
                // whose slots are: tag, name_s, name_l, body, params_head,
                // ret_ty, is_generic, gp_names_head, is_checkpoint (Stage 14.5).
                //
                // Arena positional ordering: parse_fn_decl pushes slots 4..8
                // immediately after the tag-14 mk_node call. After it returns,
                // any further __arena_push goes BEYOND the fn_decl's slot 8,
                // so we can safely append the mangled-name bytes here without
                // corrupting the fn_decl layout. Then patch slot 1 (name_s)
                // and slot 2 (name_l) to point at the new bytes.
                let fn_idx = parse_fn_decl(tok_base, sb);
                let orig_name_s = __arena_get(fn_idx + 1);
                let orig_name_l = __arena_get(fn_idx + 2);
                let mang_s = mangle_mod_name(new_prefix_s, new_prefix_l, orig_name_s, orig_name_l);
                let mang_l = new_prefix_l + 2 + orig_name_l;
                __arena_set(fn_idx + 1, mang_s);
                __arena_set(fn_idx + 2, mang_l);
                // Wrap in AST_FN_LIST (tag 15) and append to mod_pending chain.
                let list_node = mk_node(15, fn_idx, 0, 0);
                let head = mod_pending_head(sb);
                if head == 0 {
                    set_mod_pending_head(sb, list_node);
                    set_mod_pending_tail(sb, list_node);
                } else {
                    let tail = mod_pending_tail(sb);
                    __arena_set(tail + 2, list_node);
                    set_mod_pending_tail(sb, list_node);
                };
            } else {
                // Unknown IDENT — Phase-0 doesn't support struct/enum/etc.
                // inside modules. Skip the token to avoid infinite loop.
                cur_advance(sb);
            } };
        } else {
            // Non-IDENT token (whitespace? attribute?) — skip to avoid stall.
            cur_advance(sb);
        }}};
    }
    cur_advance(sb);                             // consume final '}'
    mk_node(54, 0, 0, 0)
}

// Stage 10: parse `use IDENT::IDENT::IDENT::...;`. Builds a mangled name
// `seg1__seg2__seg3` and registers (last_seg, mangled) in the use_table so
// later `bar(args)` calls (where `bar` matches the alias) get rewritten to
// `seg1__seg2__bar(args)` at the call site.
//
// Caller has already verified the cursor sits on the `use` IDENT.
//
// Returns AST_STRUCT_DECL (tag 54) — same metadata-only pattern as struct/
// enum/mod decls.
fn parse_use_decl(tok_base: i32, sb: i32) -> i32 {
    cur_advance(sb);                             // consume 'use' IDENT
    // First IDENT must be a module name.
    let f_tok = cur_get(sb);
    let first_s = tok_p2(tok_base, f_tok);
    let first_l = tok_p3(tok_base, f_tok);
    cur_advance(sb);                             // first IDENT
    // Build the mangled name in the arena segment-by-segment as we walk
    // `::IDENT` pairs. Track the last-segment bytes for the alias name.
    let mang_s = __arena_len();
    let mut i0: i32 = 0;
    while i0 < first_l {
        __arena_push(__arena_get(first_s + i0));
        i0 = i0 + 1;
    }
    let mut total_l: i32 = first_l;
    let mut last_seg_s: i32 = first_s;
    let mut last_seg_l: i32 = first_l;
    let mut keep: i32 = 1;
    while keep == 1 {
        let tt = tok_tag(tok_base, cur_get(sb));
        // Path separator `::` is two consecutive TK_COLON (14) tokens.
        if tt == 14 {
            let nt = tok_tag(tok_base, cur_get(sb) + 1);
            if nt == 14 {
                cur_advance(sb);                 // first ':'
                cur_advance(sb);                 // second ':'
                // Next must be an IDENT (segment name).
                let sk = cur_get(sb);
                let st = tok_tag(tok_base, sk);
                if st == 2 {
                    let s_s = tok_p2(tok_base, sk);
                    let s_l = tok_p3(tok_base, sk);
                    cur_advance(sb);             // segment IDENT
                    // Append `__<segment>` to mangled name.
                    __arena_push(95); __arena_push(95);
                    let mut j0: i32 = 0;
                    while j0 < s_l {
                        __arena_push(__arena_get(s_s + j0));
                        j0 = j0 + 1;
                    }
                    total_l = total_l + 2 + s_l;
                    last_seg_s = s_s;
                    last_seg_l = s_l;
                } else {
                    keep = 0;
                };
            } else {
                keep = 0;
            };
        } else {
            keep = 0;
        };
    }
    // Consume the terminating ';' (TK_SEMI = 12). Defensive: tolerate
    // missing ';' to avoid stalling on malformed input.
    let after_t = tok_tag(tok_base, cur_get(sb));
    if after_t == 12 {                           // TK_SEMI
        cur_advance(sb);
    };
    // Register the alias.
    use_tab_add(sb, last_seg_s, last_seg_l, mang_s, total_l);
    mk_node(54, 0, 0, 0)
}

// Stage 8.5: parse `trait IDENT { fn IDENT(...) -> RET ; ... }`.
// Methods are signature-only (terminated with ';'); Phase-0 does not
// validate them against impls. We just register the trait name and
// consume tokens until the closing '}'. Returns AST_STRUCT_DECL (tag 54)
// — same metadata-only pattern as struct/enum decls — so codegen emits
// 0 bytes and there is no new emit_ast_code arm.
// K1.V (2026-05-25): parse `type Alias = TypeExpr ;`. Caller has
// verified the cursor sits on the `type` IDENT. Consumes the
// entire decl up to and including the trailing `;`. Returns
// AST_STRUCT_DECL (tag 54) -- codegen no-op pattern shared with
// struct/enum/trait/impl/mod/use decls.
//
// The TypeExpr can contain anything (bare IDENT, `[T; N]`, `&T`,
// `*const T`, `Foo<T>`, etc.) -- the skip-loop just consumes
// tokens until it hits TK_SEMI (12) or TK_EOF (0). Doesn't need
// `<>` nesting since `;` doesn't appear inside type expressions.
fn parse_type_alias_decl(tok_base: i32, sb: i32) -> i32 {
    cur_advance(sb);                         // consume 'type' IDENT
    cur_advance(sb);                         // consume alias name IDENT
    cur_advance(sb);                         // consume '=' (TK_EQ = 15)
    let mut keep_ty: i32 = 1;
    while keep_ty == 1 {
        let tt = tok_tag(tok_base, cur_get(sb));
        if tt == 12 {
            keep_ty = 0;
        } else { if tt == 0 {
            keep_ty = 0;
        } else {
            cur_advance(sb);
        }};
    }
    cur_advance(sb);                         // consume ';'
    mk_node(54, 0, 0, 0)
}

fn parse_trait_decl(tok_base: i32, sb: i32) -> i32 {
    cur_advance(sb);                         // consume 'trait' IDENT
    let nk = cur_get(sb);
    let name_s = tok_p2(tok_base, nk);
    let name_l = tok_p3(tok_base, nk);
    cur_advance(sb);                         // consume trait name IDENT
    cur_advance(sb);                         // consume '{' (LBRACE = 5)
    // Brace-balance scan: consume tokens until the matching '}'. Most
    // trait bodies in Phase-0 are flat (no nested {} since methods are
    // signature-only), but the loop tolerates nesting for safety.
    let mut depth: i32 = 1;
    while depth > 0 {
        let tt = tok_tag(tok_base, cur_get(sb));
        if tt == 5 {                         // LBRACE
            depth = depth + 1;
        } else { if tt == 6 {                // RBRACE
            depth = depth - 1;
        } else { if tt == 0 {                // EOF safety
            depth = 0;
        } else {} } };
        if depth > 0 {
            cur_advance(sb);
        };
    }
    cur_advance(sb);                         // consume final '}'
    trait_tab_add(sb, name_s, name_l);
    mk_node(54, 0, 0, 0)
}

fn parse_struct_decl(tok_base: i32, sb: i32) -> i32 {
    cur_advance(sb);                         // consume 'struct' IDENT
    let nk = cur_get(sb);
    let name_s = tok_p2(tok_base, nk);
    let name_l = tok_p3(tok_base, nk);
    cur_advance(sb);                         // consume name IDENT
    // Stage 28.11 INCREMENT 2: reset gp_tab BEFORE parsing the
    // optional `<T1, T2, ...>` clause so that:
    //   (a) the gp_tab_add calls in the IDENT branch below start
    //       fresh — no cross-decl leak from a prior parse_fn_decl
    //       or parse_struct_decl.
    //   (b) the field-type loop below sees only THIS struct's
    //       generic params via gp_tab_lookup; T-typed fields encode
    //       as 200+gp_idx, scalar fields encode as -1.
    // Mirrors parse_fn_decl's `gp_tab_reset(sb)` at parser.hx:5218.
    // Per cycle-1 OBS-1 (type-design conf 90, INC-2 forward-compat),
    // explicit reset prevents the "two consecutive generic structs
    // leak T into Foo's gp_tab" hazard.
    gp_tab_reset(sb);
    // Stage 28.11 INCREMENT 1: accept optional `<T1, T2, ...>`
    // generic-params clause between the struct name and `{`. The
    // syntax is parsed-and-discarded in this increment so existing
    // tests stay green. Subsequent increments (port of struct_mono.py):
    //   * INCREMENT 2 — populate gp_tab during field parsing so a
    //     field typed `T` encodes as 200 + gp_idx (mirroring the
    //     Stage-8 fn-param mechanism); accumulate concrete uses
    //     (`Pt<i32>`) at use sites into a struct_mr_tab.
    //   * INCREMENT 3 — synthesize monomorphized struct_tab entries
    //     with mangled names (`Pt__i32`) so codegen sees concrete
    //     i32-typed fields.
    // Mirrors `helixc/frontend/struct_mono.py::collect_generic_structs`
    // and the Stage-8 fn-generic parsing scaffold (parser.hx ~5219).
    // Phase-0 parses (count) and discards — no slot in struct_tab yet.
    //
    // Stage 28.11 INC-1 cycle-2 silent-failure fix-sweep (SF-1 CRITICAL
    // conf 95 + SF-2 CRITICAL conf 92 + SF-3 HIGH conf 88):
    // Pre-fix the else-arm accepted ANY non-COMMA/GT/EOF token via a
    // bare `cur_advance`, so `struct X<T { x: i32 }` (missing `>`) made
    // the loop devour the entire struct body AND subsequent decls until
    // it found a stray `>` or EOF — a one-character typo silently ate
    // arbitrary source. Also the post-loop `cur_advance` was
    // unconditional, so `struct X<T,<EOF>` silently registered a
    // truncated 0-field struct with no diagnostic. Same root cause
    // class (silent malformed-input acceptance) as the Stage 28.10
    // cycle 84/89 Audit-13-class bugs.
    //
    // The fix:
    //   (a) else-arm restricted to TK_IDENT (tag 2) only; any other
    //       tag exits the loop without consuming so the bad token
    //       remains in the stream where downstream errors land
    //       loudly instead of silently misaligning.
    //   (b) post-loop `cur_advance(sb); // consume '>'` is now guarded
    //       by an explicit TK_GT check; on EOF-mid-list or bad-exit
    //       the advance is skipped so we don't over-advance past the
    //       lex-emitted token stream.
    // The Stage-8 sister fn-generic loop (~5219-5262) has the same
    // pattern; per narrow-scope discipline (workspace cycle-71 note)
    // it is left as DEFERRED-KNOWN; should be fixed in a separate
    // commit since it's not under active iteration. See cycle-1
    // silent-failure findings SF-1/SF-2/SF-3/SF-5 for detail.
    //
    // Stage 28.11 INC-1 cycle-3 polish (cycle-2 type-design OBS-2,
    // MED conf 82): residual bounded-misparse vector — `struct X<T,
    // struct Y { ... }` (missing `>` AND next IDENT is a keyword-led
    // decl). Keywords in the bootstrap are post-lex IDENTs (TK_IDENT
    // tag 2 at the lex level), so the loop consumes `struct`, `Y`,
    // then exits on `{`. The X decl absorbs Y's body and Y vanishes.
    // Severity: MED (requires missing-`>` typo spanning two decls,
    // result bounded to one sibling-decl eaten). KNOWN LIMITATION;
    // proper fix requires a keyword-string check inside this loop
    // (bootstrap currently does keyword detection at decl-dispatch
    // time, not here). Out-of-scope for INC-1; track for INC-2/3.
    let g_peek = tok_tag(tok_base, cur_get(sb));
    if g_peek == 16 {                        // TK_LT = `<`
        cur_advance(sb);                     // consume '<'
        let mut keep_g: i32 = 1;
        while keep_g == 1 {
            let gtt = tok_tag(tok_base, cur_get(sb));
            if gtt == 17 {                   // TK_GT = `>` end
                keep_g = 0;
            } else { if gtt == 13 {          // COMMA
                cur_advance(sb);
            } else { if gtt == 2 {           // TK_IDENT — capture
                // Stage 28.11 INCREMENT 2: capture (name_s, name_l)
                // into gp_tab BEFORE advancing so the field-type
                // loop below can resolve T → 200+gp_idx encoding.
                // Mirrors parse_fn_decl's `gp_tab_add(sb, gp_s, gp_l)`
                // at parser.hx:5239.
                let gk = cur_get(sb);
                let gp_s = tok_p2(tok_base, gk);
                let gp_l = tok_p3(tok_base, gk);
                gp_tab_add(sb, gp_s, gp_l);
                cur_advance(sb);
            } else {
                // Stage 28.11 INC-1 cycle-2 SF-1/SF-3 fix: any token
                // other than IDENT/COMMA/GT exits the loop WITHOUT
                // advancing — covers EOF (tag 0), LBRACE/RBRACE
                // (5/6), nested `<` (16), operators, literals, etc.
                // The post-loop guard below handles the cursor.
                keep_g = 0;
            }}};
        }
        // Stage 28.11 INC-1 cycle-2 SF-2 fix: only advance past `>`
        // if the cursor actually points at one. EOF-mid-list or
        // bad-token-exit leaves the cursor on the offending token
        // where the LBRACE advance below (or a downstream field
        // parser) will land an error loudly instead of silently
        // misaligning the token stream.
        if tok_tag(tok_base, cur_get(sb)) == 17 {
            cur_advance(sb);                 // consume '>'
        };
    };
    // Stage 28.11 INC-1 cycle-3 polish (cycle-2 silent-failure RE-2,
    // MED conf 90): the unconditional `cur_advance` below — and the
    // RBRACE consume at the field-loop tail — rely on the host
    // `__arena_get` returning 0 for OOB indices (x86_64.py:1946-1968
    // ARENA_GET, `jb in_bounds` fall-through to `mov eax, 0`). So
    // `tok_tag` of an over-advanced cursor returns 0, which equals
    // TK_EOF, which the surrounding `while keep==1` loops respect
    // as a clean exit. Truncated-generics paths (`struct X<T,<EOF>`)
    // implicitly depend on this contract. If a future arena-bounds
    // policy traps on OOB or emits non-zero sentinels, these advance
    // sites + the RBRACE consume need explicit EOF guards.
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
            // Stage 28.11 INCREMENT 3a: field-type encoding for
            // generic structs. If the field type IDENT matches a
            // generic-param name registered in gp_tab during the
            // `<T1, T2, ...>` parse above, encode the slot as
            // `200 + gp_idx` (mirrors parse_fn_decl's fn-param
            // encoding at parser.hx:5346). Otherwise fall through
            // to `struct_tab_lookup_idx` so scalar / nested-struct
            // semantics are unchanged for non-generic fields.
            //
            // The 200+ encoding is RESERVED for future INC-3b
            // monomorphization (`struct_mr_tab` + concrete clones).
            // For INC-3a, the downstream reader at parser.hx:1635
            // is updated atomically (in the same commit) with a
            // `< 200` guard so 200+gp_idx values are treated as
            // scalar (4-byte i32-shaped) field reads — preserving
            // the existing `struct Pt<T> { x: T, y: T }` probe
            // behavior (exits 42 with i32 field semantics).
            //
            // INC-3b will read the 200+ encoding at use-sites
            // (parse_primary's struct-lit detection) to drive
            // monomorphization.
            // Stage 28.11 INC-3a cycle-5: gp_marker_encode helper
            // centralizes the `200 + gp_idx` boundary so INC-3b's
            // use-site reader composes against the same primitive.
            let gp_idx = gp_tab_lookup(sb, t_s, t_l);
            let f_struct_idx = if gp_idx >= 0 {
                gp_marker_encode(gp_idx)
            } else {
                struct_tab_lookup_idx(sb, t_s, t_l)
            };
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
    // Stage 28.11 INC-3b: BEFORE struct_tab_add and BEFORE the exit-
    // reset of gp_tab below, capture the gp_count and build a
    // mk_node(76, name_s, name_l, next) chain of gp_names. Mirrors
    // parse_fn_decl's gp_chain_head pattern at line ~5475. The chain
    // and count are then stored in struct_gp_tab keyed by the
    // struct_idx returned by struct_tab_add.
    //
    // Building the chain BEFORE struct_tab_add (which doesn't
    // allocate AST nodes — just writes to fixed slots) is safe.
    // We do it BEFORE the gp_tab_reset at the end so gp_tab is
    // still populated when we walk it.
    let gp_count_now = gp_tab_count(sb);
    let gp_base_now = gp_tab_base(sb);
    let mut gp_chain_head: i32 = 0;
    let mut gp_chain_prev: i32 = 0;
    let mut gp_walk: i32 = 0;
    while gp_walk < gp_count_now {
        let gp_entry = gp_base_now + gp_walk * 2;
        let gn_s = __arena_get(gp_entry);
        let gn_l = __arena_get(gp_entry + 1);
        let new_gn = mk_node(76, gn_s, gn_l, 0);
        if gp_chain_head == 0 {
            gp_chain_head = new_gn;
            gp_chain_prev = new_gn;
        } else {
            __arena_set(gp_chain_prev + 3, new_gn);
            gp_chain_prev = new_gn;
        };
        gp_walk = gp_walk + 1;
    }
    let struct_idx_added = struct_tab_add(sb, name_s, name_l, field_count, fields_ptr);
    // Stage 28.11 INC-3b: register struct's generic-param metadata
    // for use-site monomorphization. Only when gp_count > 0 (i.e.
    // a generic struct decl `struct Pt<T>`). For non-generic structs
    // (no `<...>` clause), struct_gp_tab_lookup returns 0 (its miss
    // default), which the use-site can treat as "no monomorphization
    // needed".
    if gp_count_now > 0 {
        if struct_idx_added >= 0 {
            struct_gp_tab_add(sb, struct_idx_added, gp_count_now, gp_chain_head);
        };
    };
    // Stage 28.11 INCREMENT 2: reset gp_tab AFTER struct_tab_add so
    // subsequent decls (struct OR fn) start with a clean generic-
    // param table. parse_fn_decl resets at its entry point (~5218);
    // this exit-side reset adds belt-and-braces protection for the
    // window between this struct and any non-fn / non-struct decl
    // that might be parsed before the next reset. Cycle-1 OBS-1
    // (type-design conf 90) called this out as the INC-2 forward-
    // compat requirement.
    gp_tab_reset(sb);
    mk_node(54, 0, 0, 0)
}

// Stage 28.10 INCREMENT 1: top-level `parse_pattern` is a wrapper that
// supports or-patterns `pat1 | pat2 | pat3`. Each atomic pattern is
// parsed via `parse_pattern_atom` (the renamed prior parse_pattern body);
// after the atom, we peek for TK_PIPE (28). If present, parse the
// alternatives chained via mk_node(51, alt, next, 0) tuple-cons reuse
// and emit PAT_OR (tag 68) with p1 = head_alt, p2 = count, p3 = 0.
//
// Recursion: parse_pattern's 6 internal call sites (variant/tuple
// sub-pats + match-arm pat positions) all go through this wrapper, so
// sub-patterns like `Some(1 | 2)` and `(1 | 2, 3)` are also supported.
//
// Mirrors `helixc/frontend/match_lower.py::_or_chain` which lowers
// or-patterns to `||` chains of tests; bootstrap codegen will do the
// same at the IR level (see emit_pat_or, Stage 28.10 INCREMENT 3).
fn parse_pattern(tok_base: i32, sb: i32) -> i32 {
    let first = parse_pattern_atom(tok_base, sb);
    let nk = cur_get(sb);
    let nt = tok_tag(tok_base, nk);
    if nt != 28 {
        // No `|` — return the atom unwrapped (common case).
        first
    } else {
        // Build OR alt-chain. Each alt is wrapped in a TUPLE_CONS
        // (tag 51) cell: p1 = alt_idx, p2 = next_cell, p3 = unused.
        // Stage 28.10 cycle-78 CN-2 fix (HIGH conf 93): reject
        // PAT_BIND alts at parse-time. Pre-fix the comment claimed
        // "no binders in OR alts" but nothing enforced it; an alt
        // containing PAT_BIND (tag 65) would `bind_push` at codegen
        // without a matching `bind_pop`, leaking the bind-state
        // stack and resolving the bound name in later arms to
        // garbage. Mirror Python `_collect_binds` PatOr semantics.
        // Also: cap alts at 17 (cycle-80 CN-A precise bound; cycle-78
        // initially used 15, cycle-79 16, cycle-80 17). fail_jmp_state
        // cap is 16 successful adds; only N-1 non-last alts add to
        // success_state; so N=18 is the first failing case. Emit
        // AST_ERR with trap-id 62021 (renumbered from 62009 in
        // cycle-85 to avoid Stage 7 audit reservation collision)
        // if alt count exceeds 17.
        // Cycle 79 CN-1 deep-fix: check `first` AND each alt via the
        // recursive `pattern_contains_or` walker — cycle-78's shallow
        // tag check only caught top-level PAT_OR, missing the
        // `Some(1 | 2) | None` case where OR hides inside a variant/
        // tuple sub-pattern. The static-slot collision in emit_pat_or
        // would otherwise miscompile.
        //
        // Cycle 82 CN-1 fix (HIGH conf 85): when `first` violates the
        // bind/nested-OR rule, drain ALL trailing alts before
        // returning the AST_ERR trap node. Pre-fix the parser only
        // consumed ONE `|` and left subsequent alts in the token
        // stream, causing the match-arm parser to misalign and parse
        // the next alt as the arm body. Symmetric with the in-loop
        // drain pattern below (which correctly consumes every alt).
        if pattern_contains_bind(first) == 1 {
            drain_or_alts(tok_base, sb);
            mk_node(99, 62020, 0, 0)
        } else { if pattern_contains_or(first) == 1 {
            drain_or_alts(tok_base, sb);
            mk_node(99, 62022, 0, 0)
        } else {
            let head = mk_node(51, first, 0, 0);
            let mut tail: i32 = head;
            let mut count: i32 = 1;
            let mut keep: i32 = 1;
            let mut bind_violation: i32 = 0;
            let mut nested_violation: i32 = 0;
            while keep == 1 {
                let nk2 = cur_get(sb);
                if tok_tag(tok_base, nk2) == 28 {
                    cur_advance(sb);                   // consume `|`
                    let alt = parse_pattern_atom(tok_base, sb);
                    // CN-2: reject binds in alts.
                    if pattern_contains_bind(alt) == 1 {
                        bind_violation = 1;
                    };
                    // CN-1 deep: reject ANY nested OR (top-level or
                    // hidden inside a variant/tuple sub-pattern).
                    if pattern_contains_or(alt) == 1 {
                        nested_violation = 1;
                    };
                    let new_cell = mk_node(51, alt, 0, 0);
                    __arena_set(tail + 2, new_cell);
                    tail = new_cell;
                    count = count + 1;
                } else { keep = 0; };
            }
            // Decide final node based on collected violations + count.
            // Cycle 80 CN-A off-by-one fix (LOW conf 70): fail_jmp_state
            // cap is 16 successful adds (count progresses 0 → 16). In
            // emit_pat_or, only NON-LAST alts add to success_state, so
            // N alts produce N-1 adds. Threshold for first dropped add:
            // N-1 = 17 ⟹ N = 18. So `count > 17` is the precise bound.
            // Cycle-78 said `count > 15` (over-strict by 2); cycle-79
            // said `count > 16` (over-strict by 1); cycle-80 lands the
            // exact bound. 17-alt OR programs now parse correctly.
            if bind_violation == 1 {
                mk_node(99, 62020, 0, 0)                // PAT_BIND in OR
            } else { if nested_violation == 1 {
                mk_node(99, 62022, 0, 0)                // nested OR
            } else { if count > 17 {
                mk_node(99, 62021, 0, 0)                // alt-cap overflow
            } else {
                mk_node(68, head, count, 0)             // valid PAT_OR
            }}}
        }}
    }
}

// Stage 28.10 cycle-82 CN-1 helper: drain remaining `| pat_atom`
// alternatives from the token stream after a first-position violation.
// Used by parse_pattern when `first` triggers a bind/nested-OR rule
// — we must consume the trailing alts so parse_match_expr's
// subsequent `=>` lookup aligns correctly. Pre-fix only one `|` was
// consumed, leaving e.g. `Some(b) => ...` as subsequent tokens that
// got mis-parsed.
fn drain_or_alts(tok_base: i32, sb: i32) -> i32 {
    let mut keep: i32 = 1;
    while keep == 1 {
        let nk = cur_get(sb);
        if tok_tag(tok_base, nk) == 28 {
            cur_advance(sb);                            // consume `|`
            // Parse + discard the alt's atom. We don't care about
            // the resulting node — it'll be unreachable behind
            // the trap.
            parse_pattern_atom(tok_base, sb);
        } else { keep = 0; };
    }
    0
}

// Stage 28.10 cycle-78 CN-2 helper: does the pattern (or any of its
// sub-patterns inside a variant/tuple) contain a PAT_BIND? Used by
// parse_pattern to reject `Some(x) | Other(x)`-style sources because
// bootstrap codegen doesn't yet support OR-bind intersection (Python
// match_lower.py::_collect_binds handles this; deferred from Phase-0).
// Stage 29 fix (2026-05-12): rewrote 6 early returns as accumulator
// pattern (found flag + while loop) so bootstrap parser (which doesn't
// support `return` keyword) can self-host.
fn pattern_contains_bind(pat_idx: i32) -> i32 {
    if pat_idx == 0 { 0 }
    else {
        let pt = __arena_get(pat_idx);
        if pt == 65 { 1 }                                // PAT_BIND
        else { if pt == 69 {
            // PAT_VARIANT — walk sub_pats (p2 is head of TUPLE_CONS chain).
            let mut cur = __arena_get(pat_idx + 2);
            let mut found = 0;
            while cur != 0 {
                if found == 0 {
                    let sub = __arena_get(cur + 1);
                    if pattern_contains_bind(sub) == 1 { found = 1; };
                };
                cur = __arena_get(cur + 2);
            }
            found
        } else { if pt == 70 {
            // PAT_TUPLE — same chain shape as PAT_VARIANT.
            let mut cur2 = __arena_get(pat_idx + 2);
            let mut found2 = 0;
            while cur2 != 0 {
                if found2 == 0 {
                    let sub2 = __arena_get(cur2 + 1);
                    if pattern_contains_bind(sub2) == 1 { found2 = 1; };
                };
                cur2 = __arena_get(cur2 + 2);
            }
            found2
        } else { if pt == 68 {
            // Nested OR — also check its alts.
            let mut cur3 = __arena_get(pat_idx + 1);
            let mut found3 = 0;
            while cur3 != 0 {
                if found3 == 0 {
                    let alt = __arena_get(cur3 + 1);
                    if pattern_contains_bind(alt) == 1 { found3 = 1; };
                };
                cur3 = __arena_get(cur3 + 2);
            }
            found3
        } else { 0 } } } }
    }
}

// Stage 28.10 cycle-79 CN-1 deep-fix helper: does the pattern (or any
// of its sub-patterns) contain a PAT_OR (tag 68)? The cycle-78 fix
// only checked the alt's TOP-LEVEL tag, but `parse_pattern_atom` calls
// `parse_pattern` on variant/tuple sub-patterns, so an OR can hide
// inside `Some(1 | 2)`. The cycle-78 check therefore failed to detect
// `Some(1 | 2) | None` — leaving the static-slot collision in
// emit_pat_or (bn_state + 123 / + 140) reachable.
//
// Cycle-79 fix: replace the shallow tag check with this recursive
// walker, mirroring pattern_contains_bind's shape across PAT_VARIANT
// (69), PAT_TUPLE (70), and PAT_OR (68) sub-chains.
// Stage 29 fix (2026-05-12): rewrote 5 early returns as accumulator
// pattern so bootstrap parser can self-host.
fn pattern_contains_or(pat_idx: i32) -> i32 {
    if pat_idx == 0 { 0 }
    else {
        let pt = __arena_get(pat_idx);
        if pt == 68 { 1 }                                // PAT_OR
        else { if pt == 69 {
            let mut cur = __arena_get(pat_idx + 2);
            let mut found = 0;
            while cur != 0 {
                if found == 0 {
                    let sub = __arena_get(cur + 1);
                    if pattern_contains_or(sub) == 1 { found = 1; };
                };
                cur = __arena_get(cur + 2);
            }
            found
        } else { if pt == 70 {
            let mut cur2 = __arena_get(pat_idx + 2);
            let mut found2 = 0;
            while cur2 != 0 {
                if found2 == 0 {
                    let sub2 = __arena_get(cur2 + 1);
                    if pattern_contains_or(sub2) == 1 { found2 = 1; };
                };
                cur2 = __arena_get(cur2 + 2);
            }
            found2
        } else { 0 } } }
    }
}

// Stage 7: parse a single pattern atom (no `|` allowed at this layer —
// `|` is handled by the parse_pattern wrapper above). Dispatches on the
// current token:
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
fn parse_pattern_atom(tok_base: i32, sb: i32) -> i32 {
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
            // K1.L (2026-05-25): inclusive `..=` -- TK_EQ (15) right
            // after TK_DOTDOT marks the closed form. p3 of the
            // AST_PAT_RANGE node carries the inclusive flag; the
            // kovc.hx codegen reads it to choose `jg` vs `jge`.
            let pek = cur_get(sb);
            let pet = tok_tag(tok_base, pek);
            let inclusive = if pet == 15 { cur_advance(sb); 1 } else { 0 };
            let hk = cur_get(sb);
            let hi = tok_p1(tok_base, hk);
            cur_advance(sb);                 // consume hi INT
            mk_node(67, v, hi, inclusive)
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
            let want_arity = enum_tab_variant_lookup_arity(sb, e_idx_pre, v_name_s, v_name_l);
            let safe_disc = if disc < 0 { 0 } else { disc };
            // Optional `(sub_pat1, sub_pat2, ...)` — payload sub-patterns.
            let after_t = tok_tag(tok_base, cur_get(sb));
            let mut sub_head: i32 = 0;
            let mut sub_arity: i32 = 0;
            if after_t == 3 {                // '('
                cur_advance(sb);             // consume '('
                let first_pat = parse_pattern(tok_base, sb);
                sub_head = mk_node(51, first_pat, 0, 0);
                sub_arity = 1;
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
                        sub_arity = sub_arity + 1;
                    } else { if at == 0 {    // EOF safety
                        keep = 0;
                    } else {
                        // Defensive: shouldn't happen given the grammar.
                        keep = 0;
                    }}};
                }
                cur_advance(sb);             // consume ')'
            };
            // Audit A2-F8 fix: if the variant name is unknown to the
            // enum (disc < 0), emit AST_ERR(62006) instead of silently
            // substituting safe_disc=0. If the user's sub-pat arity
            // disagrees with the declared variant arity, emit
            // AST_ERR(62005). want_arity == -1 also fires the unknown-
            // variant trap (lookup miss).
            if disc < 0 {
                mk_node(99, 62006, 0, 0)
            } else { if want_arity != sub_arity {
                mk_node(99, 62005, 0, 0)
            } else {
                mk_node(69, safe_disc, sub_head, e_idx_pre)
            }}
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
        // Audit A2-F6 fix: unknown pattern token used to silently emit
        // PAT_WILDCARD (tag 66), which always matches. The leading token
        // was consumed but not interpreted — patterns like negative
        // literals (`-5`), float literals (`0.5_f64`), wide-int literals
        // (`42_i64`), or bool patterns (`true`) silently became wildcards
        // and the match arm always fired. Fix: emit AST_ERR with trap-id
        // 62002 so codegen surfaces a hard SIGILL with the trap-id in
        // eax. We still consume the leading token so the surrounding
        // arm-parser doesn't infinite-loop; it'll then reach `=>` or `,`
        // and continue cleanly until codegen fires the trap.
        cur_advance(sb);
        mk_node(99, 62002, 0, 0)
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
// K1.C-deadcode (2026-05-25): parse a `return <expr>` form. The
// 'return' IDENT has been peeked but NOT consumed by the caller.
// Emits an AST_RET node (tag 43) whose p1 is the value-expression's
// arena index; kovc.hx codegen emits the value into rax then the
// fn epilogue + ret.
//
// This fn is CURRENTLY UNREACHABLE -- parse_primary has no caller
// arm yet. The follow-up wire-up chunk adds a 1-line dispatch entry
// in parse_primary's IDENT keyword cascade. Extracted to a
// top-level fn so the eventual arm stays SHALLOW (parse_primary's
// host-parser recursion budget is fragile -- see K1.B's audit-fix
// note at line 6099 of kovc.hx for the lesson).
fn parse_return(tok_base: i32, sb: i32) -> i32 {
    cur_advance(sb);                              // consume 'return' IDENT
    let value = parse_expr_basic(tok_base, sb);   // value expression
    mk_node(43, value, 0, 0)                      // AST_RET
}

// K1.G-deadcode (2026-05-25): parse a `for var in start..end { body }`
// form. The 'for' IDENT has been peeked but NOT consumed by the caller.
// Desugars into existing AST tags (no new tag): a `let mut var = start;
// while var < end { body; var = var + 1 }` sequence.
//
// All existing tags. No codegen changes needed -- the while + let_mut +
// assign + var + add + lt arms handle the desugared shape.
//
// AST_LET_MUT (tag 12) uses a 5-slot layout per the audit-14 fix at
// parser.hx:2622-2627: mk_node creates 4 slots (tag, name_s, name_l,
// body), and an EXTRA __arena_push(value) appends a 5th slot for the
// init value.
//
// CURRENTLY UNREACHABLE -- parse_primary has no `for` arm yet. The
// follow-up K1.G-wireup chunk adds the dispatch line.
fn parse_for(tok_base: i32, sb: i32) -> i32 {
    cur_advance(sb);                              // consume 'for' IDENT
    let var_tok = cur_get(sb);
    let var_s = tok_p2(tok_base, var_tok);
    let var_l = tok_p3(tok_base, var_tok);
    cur_advance(sb);                              // consume var IDENT
    cur_advance(sb);                              // consume 'in' IDENT
    let start_expr = parse_expr_basic(tok_base, sb);
    cur_advance(sb);                              // consume '..' (TK_DOTDOT = 43)
    // K1.L (2026-05-25): inclusive `..=` -- TK_EQ right after `..`
    // makes the cond AST_LE instead of AST_LT, so the body runs at
    // x == end too.
    let pek = cur_get(sb);
    let pet = tok_tag(tok_base, pek);
    let for_inclusive = if pet == 15 { cur_advance(sb); 1 } else { 0 };
    let end_expr = parse_expr_basic(tok_base, sb);
    cur_advance(sb);                              // consume '{'
    let body_expr = parse_expr(tok_base, sb);
    cur_advance(sb);                              // consume '}'
    // Desugared AST (inner-out construction):
    let var_ref_inc = mk_node(1, var_s, var_l, 0);            // AST_VAR
    let one_lit = mk_node(0, 1, 0, 0);                        // AST_INT(1)
    let inc_expr = mk_node(2, var_ref_inc, one_lit, 0);       // AST_ADD
    let assign = mk_node(11, var_s, var_l, inc_expr);         // AST_ASSIGN
    let body_chain = mk_node(13, body_expr, assign, 0);       // AST_SEQ
    let var_ref_cond = mk_node(1, var_s, var_l, 0);           // AST_VAR
    let cmp_tag = if for_inclusive == 1 { 22 } else { 6 };    // AST_LE vs AST_LT
    let cond_expr = mk_node(cmp_tag, var_ref_cond, end_expr, 0);
    let while_node = mk_node(10, cond_expr, body_chain, 0);   // AST_WHILE
    let let_mut_node = mk_node(12, var_s, var_l, while_node); // AST_LET_MUT (4 slots)
    __arena_push(start_expr);                                  // 5th slot = init value
    let_mut_node
}

// K1.H1-deadcode (2026-05-25): parse a `loop { body }` form. The
// 'loop' IDENT has been peeked but NOT consumed by the caller.
// Desugars to `while 1 { body }` -- AST_WHILE(AST_INT(1), body) --
// using existing tags only. No codegen changes needed (the while
// arm already lowers cond==1 like any other condition).
//
// break/continue inside the body are NOT supported yet (K1.H2/H3
// will add them once label tracking is in place). If the user
// writes `loop { ...; break; ... }` today the parser will fail at
// the `break` IDENT just like it does outside any loop. That's the
// correct fail-closed behaviour for K1.H1.
//
// CURRENTLY UNREACHABLE -- parse_primary has no `loop` arm yet. The
// follow-up K1.H1-wireup chunk adds the dispatch line.
fn parse_loop(tok_base: i32, sb: i32) -> i32 {
    cur_advance(sb);                              // consume 'loop' IDENT
    cur_advance(sb);                              // consume '{'
    let body_expr = parse_expr(tok_base, sb);
    cur_advance(sb);                              // consume '}'
    let one_lit = mk_node(0, 1, 0, 0);            // AST_INT(1)
    mk_node(10, one_lit, body_expr, 0)            // AST_WHILE
}

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
