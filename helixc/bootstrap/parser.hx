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
    if tok_tag(tok_base, k) == 12 {     // 12 = TK_SEMI
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
    } else { first }
}

fn parse_expr_basic(tok_base: i32, sb: i32) -> i32 {
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
            cur_advance(sb);     // '='
            // value uses parse_expr_basic so the `;` after the
            // value belongs to the let-terminator, not a sequencer.
            let value = parse_expr_basic(tok_base, sb);
            cur_advance(sb);     // ';'
            let body = parse_expr(tok_base, sb);
            let packed = value * 65536 + body;
            let tag = if is_mut == 1 { 12 } else { 8 };
            mk_node(tag, name_start, name_len, packed)
        } else { if byte_eq(id_start, id_len, kw_if_s(sb), kw_if_n(sb)) == 1 {
            cur_advance(sb);
            let cond = parse_expr_basic(tok_base, sb);
            cur_advance(sb);     // '{'
            let then_e = parse_expr(tok_base, sb);
            cur_advance(sb);     // '}'
            cur_advance(sb);     // 'else'
            cur_advance(sb);     // '{'
            let else_e = parse_expr(tok_base, sb);
            cur_advance(sb);     // '}'
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
            // Plain identifier. Could be a var ref OR an assignment
            // (`name = expr`). Peek the NEXT token: if it's TK_EQ
            // (15), this is an assign; otherwise a var ref.
            cur_advance(sb);
            let next = cur_get(sb);
            if tok_tag(tok_base, next) == 15 {
                cur_advance(sb);     // consume '='
                // assign value uses basic expression: the `;` after
                // the value is a sequencer, not part of the value.
                let value = parse_expr_basic(tok_base, sb);
                mk_node(11, id_start, id_len, value)
            } else {
                mk_node(1, id_start, id_len, 0)
            }
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

// Parse a sequence of one or more `fn` declarations at the top
// level, returning a linked list head. If only one fn is present,
// the list has a single node. The codegen looks up "main" by name
// and emits its body; other fns are placed in the binary but only
// callable once AST_CALL lands.
fn parse_program(tok_base: i32, sb: i32) -> i32 {
    let first_fn = parse_fn_decl(tok_base, sb);
    // Build list backwards: collect fn idxs then chain. But we
    // can't materialize an array easily here, so do it forward:
    // build the head node first, then attach further nodes by
    // patching the previous node's `next` field. We track the
    // previous-list-node's slot to patch it in place.
    let head = mk_node(15, first_fn, 0, 0);
    let mut prev_list = head;
    let mut keep: i32 = 1;
    while keep == 1 {
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
                // Patch prev_list's p2 (slot prev_list+2) to point at new_node.
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

// Parse `fn name() -> i32 { body }`. Phase 0: no params, single
// expression body, return type parsed but ignored. The result is
// AST_FN_DECL with body_idx in p3.
fn parse_fn_decl(tok_base: i32, sb: i32) -> i32 {
    cur_advance(sb);     // consume 'fn'
    let nk = cur_get(sb);
    let name_start = tok_p2(tok_base, nk);
    let name_len = tok_p3(tok_base, nk);
    cur_advance(sb);     // name
    cur_advance(sb);     // '('
    cur_advance(sb);     // ')'
    cur_advance(sb);     // '-' (part of '->')
    cur_advance(sb);     // '>' (the second char of '->')
    cur_advance(sb);     // return-type IDENT (ignored)
    cur_advance(sb);     // '{'
    let body = parse_expr(tok_base, sb);
    cur_advance(sb);     // '}'
    mk_node(14, name_start, name_len, body)
}
