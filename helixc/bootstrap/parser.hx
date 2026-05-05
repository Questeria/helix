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

fn parse_expr(tok_base: i32, sb: i32) -> i32 {
    let lhs = parse_add(tok_base, sb);
    let k = cur_get(sb);
    if tok_tag(tok_base, k) == 16 {     // 16 = TK_LT
        cur_advance(sb);
        let rhs = parse_add(tok_base, sb);
        mk_node(6, lhs, rhs, 0)
    } else { lhs }
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
        } else {
            keep = 0;
        }};
    }
    lhs
}

fn parse_unary(tok_base: i32, sb: i32) -> i32 {
    let k = cur_get(sb);
    if tok_tag(tok_base, k) == 8 {     // unary minus
        cur_advance(sb);
        let inner = parse_unary(tok_base, sb);
        mk_node(9, inner, 0, 0)
    } else {
        parse_primary(tok_base, sb)
    }
}

fn parse_primary(tok_base: i32, sb: i32) -> i32 {
    let k = cur_get(sb);
    let t = tok_tag(tok_base, k);
    if t == 1 {
        let v = tok_p1(tok_base, k);
        cur_advance(sb);
        mk_node(0, v, 0, 0)
    } else { if t == 2 {
        let id_start = tok_p2(tok_base, k);
        let id_len = tok_p3(tok_base, k);
        if byte_eq(id_start, id_len, kw_let_s(sb), kw_let_n(sb)) == 1 {
            cur_advance(sb);
            let nk = cur_get(sb);
            let name_start = tok_p2(tok_base, nk);
            let name_len = tok_p3(tok_base, nk);
            cur_advance(sb);     // name
            cur_advance(sb);     // '='
            let value = parse_expr(tok_base, sb);
            cur_advance(sb);     // ';'
            let body = parse_expr(tok_base, sb);
            let packed = value * 65536 + body;
            mk_node(8, name_start, name_len, packed)
        } else { if byte_eq(id_start, id_len, kw_if_s(sb), kw_if_n(sb)) == 1 {
            cur_advance(sb);
            let cond = parse_expr(tok_base, sb);
            cur_advance(sb);     // '{'
            let then_e = parse_expr(tok_base, sb);
            cur_advance(sb);     // '}'
            cur_advance(sb);     // 'else'
            cur_advance(sb);     // '{'
            let else_e = parse_expr(tok_base, sb);
            cur_advance(sb);     // '}'
            mk_node(7, cond, then_e, else_e)
        } else { if byte_eq(id_start, id_len, kw_while_s(sb), kw_while_n(sb)) == 1 {
            // while expr { body } — Phase-0 returns 0 (no useful
            // result; body must produce side effects via assign,
            // which lands in a future commit).
            cur_advance(sb);
            let cond = parse_expr(tok_base, sb);
            cur_advance(sb);     // '{'
            let body = parse_expr(tok_base, sb);
            cur_advance(sb);     // '}'
            mk_node(10, cond, body, 0)
        } else {
            cur_advance(sb);
            mk_node(1, id_start, id_len, 0)
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
        if t != 0 {
            cur_advance(sb);
        };
        mk_node(99, t, 0, 0)
    }}}
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
    0
}

// --------------------------------------------------------------
// Top-level parse: return the arena index of the root AST node.
// Reserves 7 state slots, then dispatches into parse_expr.
// --------------------------------------------------------------
fn parse_top(tok_base: i32) -> i32 {
    // 9 state slots: cursor + 4 keyword (start, len) pairs.
    let cur_slot = __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    install_keywords(cur_slot);
    parse_expr(tok_base, cur_slot)
}
