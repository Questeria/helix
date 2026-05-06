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
//  18  AST_PARAM     p1 = name_start, p2 = name_len, p3 = next_param_idx.
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
        // Mirrors helixc-Python: `!x` lowers to `(x == 0) ? 1 : 0`.
        cur_advance(sb);
        let inner = parse_unary(tok_base, sb);
        mk_node(31, inner, 0, 0)
    } else {
        parse_primary(tok_base, sb)
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
                    cur_advance(sb);     // '{'
                    else_e = parse_expr(tok_base, sb);
                    cur_advance(sb);     // '}'
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
        } else {
            // Plain identifier. Could be a var ref, an assignment
            // (`name = expr`), or a fn call (`name()`). Peek the
            // NEXT token to decide.
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
                mk_node(16, id_start, id_len, args_head)
            } else {
                // Var ref
                mk_node(1, id_start, id_len, 0)
            }}
        }}}
    } else { if t == 3 {
        cur_advance(sb);
        let inner = parse_expr(tok_base, sb);
        cur_advance(sb);     // ')'
        inner
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
    }}}}}
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
    0
}

// --------------------------------------------------------------
// Top-level parse: return the arena index of the root AST node.
// Reserves 7 state slots, then dispatches into parse_expr.
// --------------------------------------------------------------
fn parse_top(tok_base: i32) -> i32 {
    // 13 state slots: cursor + 6 keyword (start, len) pairs.
    let cur_slot = __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    install_keywords(cur_slot);
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
        if byte_eq(id_s, id_l, kw_fn_s(cur_slot), kw_fn_n(cur_slot)) == 1 {
            parse_program(tok_base, cur_slot)
        } else {
            parse_expr(tok_base, cur_slot)
        }
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
            cur_advance(sb);     // type IDENT (ignored)
            let new_param = mk_node(18, pname_s, pname_l, 0);
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
    cur_advance(sb);     // return-type IDENT (ignored)
    cur_advance(sb);     // '{'
    let body = parse_expr(tok_base, sb);
    cur_advance(sb);     // '}'
    // Audit-14: same overflow issue as AST_LET — packed encoding
    // breaks for arena indices > 65535. Extend to 5 slots: p3 =
    // body_idx, p4 = params_head.
    let node = mk_node(14, name_start, name_len, body);
    __arena_push(params_head);
    node
}
