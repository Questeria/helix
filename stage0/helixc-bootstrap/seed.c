/* SPDX-License-Identifier: Apache-2.0
 * helixc-bootstrap seed -- the trusted Helix-subset bootstrap compiler.
 *
 * The first ORIGINAL rung of the Kovostov-Native ladder (everything below it is
 * hand-authored hex0 or vendored stage0/M2-Planet sources). A small C program in
 * the M2-Planet C subset, compiled by our stage0 ladder with NO external
 * toolchain. Its job: compile the tiny Helix subset that helixc
 * (helixc/bootstrap/{kovc,parser,lexer}.hx) is written in, minting the first
 * helixc WITHOUT Python -- replacing Python as the K1 minter.
 *
 * Apache-2.0, statically separable from the GPL-3.0 vendored M2-Planet/M2libc
 * (we only BUILD with those; none of their source is copied here).
 *
 * Subset spec: docs/K_TASK0_HELIX_SUBSET_FINDINGS.md (i32-only; one arena;
 * while + if-as-expression + recursion; six intrinsics).
 *
 * INCREMENTS:
 *   0 DONE -- project + build pipeline + the global-arena core.
 *   1 THIS -- LEXER: tokenize the Helix subset into parallel token arrays.
 *   next   -- parser -> AST, then x86-64 ELF codegen, then compile kovc.hx.
 *
 * M2-subset notes (lesson 33): no global ARRAY definitions (use calloc'd global
 * pointers); sizeof + calloc OK; declare locals at top of each function.
 */

/* ===================== arena (increment 0) ===================== */
/* One flat int buffer, bump-allocated, never freed. */
int* ARENA;
int ARENA_LEN;

int arena_init() {
    ARENA = calloc(4096, sizeof(int));
    ARENA_LEN = 0;
    return 0;
}
int arena_push(int v) {
    int idx;
    idx = ARENA_LEN;
    ARENA[idx] = v;
    ARENA_LEN = ARENA_LEN + 1;
    return idx;
}
int arena_get(int i) { return ARENA[i]; }
int arena_set(int i, int v) { ARENA[i] = v; return 0; }
int arena_len() { return ARENA_LEN; }

/* ===================== character helpers ===================== */
int is_digit(int c) {
    if (c >= '0') { if (c <= '9') { return 1; } }
    return 0;
}
int is_hex(int c) {
    if (is_digit(c)) { return 1; }
    if (c >= 'a') { if (c <= 'f') { return 1; } }
    if (c >= 'A') { if (c <= 'F') { return 1; } }
    return 0;
}
int hexval(int c) {
    if (c <= '9') { return c - '0'; }
    if (c <= 'F') { return c - 'A' + 10; }
    return c - 'a' + 10;
}
int is_alpha(int c) {
    if (c == '_') { return 1; }
    if (c >= 'a') { if (c <= 'z') { return 1; } }
    if (c >= 'A') { if (c <= 'Z') { return 1; } }
    return 0;
}
int is_alnum(int c) {
    if (is_alpha(c)) { return 1; }
    if (is_digit(c)) { return 1; }
    return 0;
}
int is_space(int c) {
    if (c == ' ') { return 1; }
    if (c == 9) { return 1; }   /* tab */
    if (c == 10) { return 1; }  /* LF  */
    if (c == 13) { return 1; }  /* CR  */
    return 0;
}
int cstr_len(char* s) {
    int n;
    n = 0;
    while (s[n] != 0) { n = n + 1; }
    return n;
}

/* ===================== token tags ===================== */
/* The seed is self-contained, so it uses its own clean tag scheme (it need not
 * match lexer.hx's numbers). Keywords are recognized in the lexer. */
int TK_EOF;    int TK_IDENT; int TK_INT;   int TK_STR;
int TK_FN;     int TK_LET;   int TK_MUT;   int TK_IF;   int TK_ELSE;
int TK_WHILE;  int TK_RETURN;
int TK_LPAREN; int TK_RPAREN; int TK_LBRACE; int TK_RBRACE;
int TK_COMMA;  int TK_SEMI;  int TK_COLON; int TK_ARROW;
int TK_EQ;     int TK_PLUS;  int TK_MINUS; int TK_STAR; int TK_SLASH; int TK_PERCENT;
int TK_EQEQ;   int TK_NE;    int TK_LT;    int TK_LE;   int TK_GT;    int TK_GE;
int TK_AMP;    int TK_PIPE;  int TK_CARET; int TK_ANDAND; int TK_OROR;
int TK_BANG;

int tags_init() {
    TK_EOF = 0;    TK_IDENT = 1;  TK_INT = 2;    TK_STR = 3;
    TK_FN = 10;    TK_LET = 11;   TK_MUT = 12;   TK_IF = 13;   TK_ELSE = 14;
    TK_WHILE = 15; TK_RETURN = 16;
    TK_LPAREN = 20; TK_RPAREN = 21; TK_LBRACE = 22; TK_RBRACE = 23;
    TK_COMMA = 24; TK_SEMI = 25;  TK_COLON = 26; TK_ARROW = 27;
    TK_EQ = 28;    TK_PLUS = 29;  TK_MINUS = 30; TK_STAR = 31; TK_SLASH = 32; TK_PERCENT = 33;
    TK_EQEQ = 34;  TK_NE = 35;    TK_LT = 36;    TK_LE = 37;   TK_GT = 38;   TK_GE = 39;
    TK_AMP = 40;   TK_PIPE = 41;  TK_CARET = 42; TK_ANDAND = 43; TK_OROR = 44;
    TK_BANG = 45;
    return 0;
}

/* ===================== lexer state ===================== */
char* SRC;     /* source text (NUL-terminated)            */
int SRC_LEN;   /* byte length of SRC                       */
int* TOK;      /* stride-4 records: tag, val, start, len   */
int TOK_N;     /* number of tokens                          */

int tok_push(int tag, int val, int start, int len) {
    int i;
    i = TOK_N * 4;
    TOK[i] = tag;
    TOK[i + 1] = val;
    TOK[i + 2] = start;
    TOK[i + 3] = len;
    TOK_N = TOK_N + 1;
    return 0;
}
int tok_tag(int i)   { return TOK[i * 4]; }
int tok_val(int i)   { return TOK[i * 4 + 1]; }
int tok_start(int i) { return TOK[i * 4 + 2]; }
int tok_len(int i)   { return TOK[i * 4 + 3]; }

/* does the source span [start,start+len) equal C-string kw exactly? */
int span_eq(int start, int len, char* kw) {
    int i;
    i = 0;
    while (i < len) {
        if (SRC[start + i] != kw[i]) { return 0; }
        i = i + 1;
    }
    if (kw[len] != 0) { return 0; }   /* kw must end exactly at len */
    return 1;
}
int keyword_tag(int start, int len) {
    if (span_eq(start, len, "fn"))     { return TK_FN; }
    if (span_eq(start, len, "let"))    { return TK_LET; }
    if (span_eq(start, len, "mut"))    { return TK_MUT; }
    if (span_eq(start, len, "if"))     { return TK_IF; }
    if (span_eq(start, len, "else"))   { return TK_ELSE; }
    if (span_eq(start, len, "while"))  { return TK_WHILE; }
    if (span_eq(start, len, "return")) { return TK_RETURN; }
    return TK_IDENT;
}

/* tokenize SRC into TOK; returns 0 on success */
int lex() {
    int p; int c; int n; int start; int val; int depth;
    p = 0;
    TOK_N = 0;
    while (p < SRC_LEN) {
        c = SRC[p];
        if (p + 1 < SRC_LEN) { n = SRC[p + 1]; } else { n = 0; }

        if (is_space(c)) {
            p = p + 1;
        } else if (c == '/') { if (n == '/') {
            /* line comment */
            while (p < SRC_LEN) { if (SRC[p] == 10) { p = p; break; } p = p + 1; }
        } else { if (n == '*') {
            /* nested block comment */
            depth = 1;
            p = p + 2;
            while (p < SRC_LEN) {
                if (depth == 0) { break; }
                if (SRC[p] == '/') { if (p + 1 < SRC_LEN) { if (SRC[p + 1] == '*') { depth = depth + 1; p = p + 2; } else { p = p + 1; } } else { p = p + 1; } }
                else { if (SRC[p] == '*') { if (p + 1 < SRC_LEN) { if (SRC[p + 1] == '/') { depth = depth - 1; p = p + 2; } else { p = p + 1; } } else { p = p + 1; } }
                else { p = p + 1; } }
            }
        } else {
            tok_push(TK_SLASH, 0, p, 1); p = p + 1;
        } } }
        else if (c == '@') {
            /* skip an @attr ident (subset attrs carry no codegen meaning) */
            p = p + 1;
            while (p < SRC_LEN) { if (is_alnum(SRC[p])) { p = p + 1; } else { break; } }
        }
        else if (is_digit(c)) {
            start = p; val = 0;
            if (c == '0') { if (n == 'x') {
                p = p + 2;
                while (p < SRC_LEN) { if (is_hex(SRC[p])) { val = val * 16 + hexval(SRC[p]); p = p + 1; } else { break; } }
                tok_push(TK_INT, val, start, p - start);
            } else {
                while (p < SRC_LEN) { if (is_digit(SRC[p])) { val = val * 10 + (SRC[p] - '0'); p = p + 1; } else { break; } }
                tok_push(TK_INT, val, start, p - start);
            } } else {
                while (p < SRC_LEN) { if (is_digit(SRC[p])) { val = val * 10 + (SRC[p] - '0'); p = p + 1; } else { break; } }
                tok_push(TK_INT, val, start, p - start);
            }
        }
        else if (is_alpha(c)) {
            start = p;
            while (p < SRC_LEN) { if (is_alnum(SRC[p])) { p = p + 1; } else { break; } }
            tok_push(keyword_tag(start, p - start), 0, start, p - start);
        }
        else if (c == '"') {
            start = p + 1; p = p + 1;
            while (p < SRC_LEN) { if (SRC[p] == '"') { break; } p = p + 1; }
            tok_push(TK_STR, 0, start, p - start);
            p = p + 1;   /* skip closing quote */
        }
        /* two-character operators */
        else if (c == '=') { if (n == '=') { tok_push(TK_EQEQ, 0, p, 2); p = p + 2; } else { tok_push(TK_EQ, 0, p, 1); p = p + 1; } }
        else if (c == '!') { if (n == '=') { tok_push(TK_NE, 0, p, 2); p = p + 2; } else { tok_push(TK_BANG, 0, p, 1); p = p + 1; } }
        else if (c == '<') { if (n == '=') { tok_push(TK_LE, 0, p, 2); p = p + 2; } else { tok_push(TK_LT, 0, p, 1); p = p + 1; } }
        else if (c == '>') { if (n == '=') { tok_push(TK_GE, 0, p, 2); p = p + 2; } else { tok_push(TK_GT, 0, p, 1); p = p + 1; } }
        else if (c == '-') { if (n == '>') { tok_push(TK_ARROW, 0, p, 2); p = p + 2; } else { tok_push(TK_MINUS, 0, p, 1); p = p + 1; } }
        else if (c == '&') { if (n == '&') { tok_push(TK_ANDAND, 0, p, 2); p = p + 2; } else { tok_push(TK_AMP, 0, p, 1); p = p + 1; } }
        else if (c == '|') { if (n == '|') { tok_push(TK_OROR, 0, p, 2); p = p + 2; } else { tok_push(TK_PIPE, 0, p, 1); p = p + 1; } }
        /* single-character punctuation/operators */
        else if (c == '(') { tok_push(TK_LPAREN, 0, p, 1); p = p + 1; }
        else if (c == ')') { tok_push(TK_RPAREN, 0, p, 1); p = p + 1; }
        else if (c == '{') { tok_push(TK_LBRACE, 0, p, 1); p = p + 1; }
        else if (c == '}') { tok_push(TK_RBRACE, 0, p, 1); p = p + 1; }
        else if (c == ',') { tok_push(TK_COMMA, 0, p, 1); p = p + 1; }
        else if (c == ';') { tok_push(TK_SEMI, 0, p, 1); p = p + 1; }
        else if (c == ':') { tok_push(TK_COLON, 0, p, 1); p = p + 1; }
        else if (c == '+') { tok_push(TK_PLUS, 0, p, 1); p = p + 1; }
        else if (c == '*') { tok_push(TK_STAR, 0, p, 1); p = p + 1; }
        else if (c == '%') { tok_push(TK_PERCENT, 0, p, 1); p = p + 1; }
        else if (c == '^') { tok_push(TK_CARET, 0, p, 1); p = p + 1; }
        else {
            /* unknown byte: skip it (the parser will catch real errors) */
            p = p + 1;
        }
    }
    tok_push(TK_EOF, 0, p, 0);
    return 0;
}

/* ===================== parser (increment 2a: expressions) ===================== *
 * AST nodes live in a calloc'd int pool, stride 5: {kind, a, b, c, next}.
 * `next` chains siblings (call args now; statement lists next increment). Only
 * EXPRESSION node kinds this increment; fn/statements/blocks come next. The
 * precedence ladder mirrors C (and the Helix subset): || < && < | < ^ < & <
 * ==/!= < rel < +/- < * / %, with unary minus via the `0 - x` idiom.
 */
int ND_INT; int ND_VAR; int ND_BIN; int ND_CALL; int ND_IFE;
int ND_LET; int ND_ASSIGN; int ND_WHILE; int ND_IF; int ND_BLOCK; int ND_FN; int ND_PARAM;
int* ND;
int ND_N;
int PERR;   /* parser error flag (set by expect_tag on a mismatch) */

int nodes_init() {
    ND = calloc(8192 * 5, sizeof(int));
    ND_N = 0;
    PERR = 0;
    ND_INT = 1; ND_VAR = 2; ND_BIN = 3; ND_CALL = 4; ND_IFE = 5;
    ND_LET = 6; ND_ASSIGN = 7; ND_WHILE = 8; ND_IF = 9; ND_BLOCK = 10; ND_FN = 11; ND_PARAM = 12;
    return 0;
}
int nodes_reset() { ND_N = 0; return 0; }
int node_new(int kind, int a, int b, int c) {
    int i;
    i = ND_N * 5;
    ND[i] = kind; ND[i + 1] = a; ND[i + 2] = b; ND[i + 3] = c; ND[i + 4] = 0 - 1;
    ND_N = ND_N + 1;
    return ND_N - 1;
}
int nd_kind(int i) { return ND[i * 5]; }
int nd_a(int i)    { return ND[i * 5 + 1]; }
int nd_b(int i)    { return ND[i * 5 + 2]; }
int nd_c(int i)    { return ND[i * 5 + 3]; }
int nd_next(int i) { return ND[i * 5 + 4]; }
int nd_set_next(int i, int v) { ND[i * 5 + 4] = v; return 0; }

/* parser cursor over TOK */
int CUR;
int p_tag() { return tok_tag(CUR); }
int p_adv() { CUR = CUR + 1; return 0; }

/* forward declarations (the grammar is mutually recursive) */
int parse_expr();
int parse_or();   int parse_and();  int parse_bor(); int parse_bxor();
int parse_band(); int parse_eq();   int parse_rel(); int parse_add();
int parse_mul();  int parse_unary(); int parse_primary();
int parse_block(); int parse_if();  int parse_let();  int parse_while();
int parse_fn();    int parse_program();

/* expect a token tag: advance on match, else raise the parser error flag */
int expect_tag(int tag) {
    if (p_tag() == tag) { p_adv(); return 1; }
    PERR = 1;
    return 0;
}

int parse_primary() {
    int t; int v; int node; int name; int firstarg; int last; int arg;
    t = p_tag();
    if (t == TK_IF) { return parse_if(); }   /* if as a value-producing expression */
    if (t == TK_INT) {
        v = tok_val(CUR);
        p_adv();
        return node_new(ND_INT, v, 0, 0);
    }
    if (t == TK_LPAREN) {
        p_adv();
        node = parse_expr();
        if (p_tag() == TK_RPAREN) { p_adv(); }
        return node;
    }
    if (t == TK_IDENT) {
        name = CUR;          /* token index of the name (span looked up later) */
        p_adv();
        if (p_tag() == TK_LPAREN) {
            p_adv();
            firstarg = 0 - 1;
            last = 0 - 1;
            while (p_tag() != TK_RPAREN) {
                if (p_tag() == TK_EOF) { break; }
                arg = parse_expr();
                if (firstarg == 0 - 1) { firstarg = arg; last = arg; }
                else { nd_set_next(last, arg); last = arg; }
                if (p_tag() == TK_COMMA) { p_adv(); }
            }
            if (p_tag() == TK_RPAREN) { p_adv(); }
            return node_new(ND_CALL, name, firstarg, 0);
        }
        return node_new(ND_VAR, name, 0, 0);
    }
    /* unrecognized token: consume it and emit a 0 literal so the parser is total */
    p_adv();
    return node_new(ND_INT, 0, 0, 0);
}

int parse_unary() {
    int operand;
    if (p_tag() == TK_MINUS) {
        p_adv();
        operand = parse_unary();
        return node_new(ND_BIN, TK_MINUS, node_new(ND_INT, 0, 0, 0), operand);
    }
    return parse_primary();
}
int parse_mul() {
    int left; int op; int right;
    left = parse_unary();
    op = p_tag();
    while (op == TK_STAR || op == TK_SLASH || op == TK_PERCENT) {
        p_adv(); right = parse_unary();
        left = node_new(ND_BIN, op, left, right);
        op = p_tag();
    }
    return left;
}
int parse_add() {
    int left; int op; int right;
    left = parse_mul();
    op = p_tag();
    while (op == TK_PLUS || op == TK_MINUS) {
        p_adv(); right = parse_mul();
        left = node_new(ND_BIN, op, left, right);
        op = p_tag();
    }
    return left;
}
int parse_rel() {
    int left; int op; int right;
    left = parse_add();
    op = p_tag();
    while (op == TK_LT || op == TK_LE || op == TK_GT || op == TK_GE) {
        p_adv(); right = parse_add();
        left = node_new(ND_BIN, op, left, right);
        op = p_tag();
    }
    return left;
}
int parse_eq() {
    int left; int op; int right;
    left = parse_rel();
    op = p_tag();
    while (op == TK_EQEQ || op == TK_NE) {
        p_adv(); right = parse_rel();
        left = node_new(ND_BIN, op, left, right);
        op = p_tag();
    }
    return left;
}
int parse_band() {
    int left; int op; int right;
    left = parse_eq();
    op = p_tag();
    while (op == TK_AMP) {
        p_adv(); right = parse_eq();
        left = node_new(ND_BIN, op, left, right);
        op = p_tag();
    }
    return left;
}
int parse_bxor() {
    int left; int op; int right;
    left = parse_band();
    op = p_tag();
    while (op == TK_CARET) {
        p_adv(); right = parse_band();
        left = node_new(ND_BIN, op, left, right);
        op = p_tag();
    }
    return left;
}
int parse_bor() {
    int left; int op; int right;
    left = parse_bxor();
    op = p_tag();
    while (op == TK_PIPE) {
        p_adv(); right = parse_bxor();
        left = node_new(ND_BIN, op, left, right);
        op = p_tag();
    }
    return left;
}
int parse_and() {
    int left; int op; int right;
    left = parse_bor();
    op = p_tag();
    while (op == TK_ANDAND) {
        p_adv(); right = parse_bor();
        left = node_new(ND_BIN, op, left, right);
        op = p_tag();
    }
    return left;
}
int parse_or() {
    int left; int op; int right;
    left = parse_and();
    op = p_tag();
    while (op == TK_OROR) {
        p_adv(); right = parse_and();
        left = node_new(ND_BIN, op, left, right);
        op = p_tag();
    }
    return left;
}
int parse_expr() { return parse_or(); }

/* ----- statements, blocks, functions (increment 2b) ----- */

/* `let` [`mut`] NAME [`:` TYPE] `=` expr `;`  ->  ND_LET{a=name-tok,b=init,c=mut?} */
int parse_let() {
    int mut; int name; int init;
    expect_tag(TK_LET);
    mut = 0;
    if (p_tag() == TK_MUT) { mut = 1; p_adv(); }
    name = CUR;
    expect_tag(TK_IDENT);
    if (p_tag() == TK_COLON) { p_adv(); if (p_tag() == TK_IDENT) { p_adv(); } }
    expect_tag(TK_EQ);
    init = parse_expr();
    expect_tag(TK_SEMI);
    return node_new(ND_LET, name, init, mut);
}

/* `while` expr block  ->  ND_WHILE{a=cond,b=block} */
int parse_while() {
    int cond; int body;
    expect_tag(TK_WHILE);
    cond = parse_expr();
    body = parse_block();
    return node_new(ND_WHILE, cond, body, 0);
}

/* `if` expr block (`else` (block | if))?  ->  ND_IF{a=cond,b=then,c=else-or-(-1)}
 * Usable as both a statement and a value-producing expression. */
int parse_if() {
    int cond; int then_b; int else_b;
    expect_tag(TK_IF);
    cond = parse_expr();
    then_b = parse_block();
    else_b = 0 - 1;
    if (p_tag() == TK_ELSE) {
        p_adv();
        if (p_tag() == TK_IF) { else_b = parse_if(); }
        else { else_b = parse_block(); }
    }
    return node_new(ND_IF, cond, then_b, else_b);
}

/* `{` stmt* tail-expr? `}`  ->  ND_BLOCK{a=first-stmt-or-(-1),c=tail-expr-or-(-1)}
 * stmts chain via nd_next; the final bare expr (no `;`) is the block's value. */
int parse_block() {
    int first; int last; int tail; int t; int s; int e; int rhs; int done;
    expect_tag(TK_LBRACE);
    first = 0 - 1; last = 0 - 1; tail = 0 - 1; done = 0;
    while (done == 0) {
        t = p_tag();
        if (t == TK_RBRACE || t == TK_EOF) { done = 1; }
        else if (t == TK_LET) {
            s = parse_let();
            if (first == 0 - 1) { first = s; last = s; } else { nd_set_next(last, s); last = s; }
        }
        else if (t == TK_WHILE) {
            s = parse_while();
            if (first == 0 - 1) { first = s; last = s; } else { nd_set_next(last, s); last = s; }
        }
        else if (t == TK_IF) {
            s = parse_if();
            if (p_tag() == TK_RBRACE) { tail = s; done = 1; }
            else { if (first == 0 - 1) { first = s; last = s; } else { nd_set_next(last, s); last = s; } }
        }
        else {
            e = parse_expr();
            if (p_tag() == TK_EQ) {
                p_adv(); rhs = parse_expr(); expect_tag(TK_SEMI);
                s = node_new(ND_ASSIGN, nd_a(e), rhs, 0);
                if (first == 0 - 1) { first = s; last = s; } else { nd_set_next(last, s); last = s; }
            }
            else if (p_tag() == TK_SEMI) {
                p_adv();
                if (first == 0 - 1) { first = e; last = e; } else { nd_set_next(last, e); last = e; }
            }
            else { tail = e; done = 1; }
        }
    }
    expect_tag(TK_RBRACE);
    return node_new(ND_BLOCK, first, 0, tail);
}

/* `fn` NAME `(` params `)` (`->` TYPE)? block  ->  ND_FN{a=name-tok,b=first-param,c=body} */
int parse_fn() {
    int name; int firstp; int lastp; int ptok; int pnode; int body;
    expect_tag(TK_FN);
    name = CUR;
    expect_tag(TK_IDENT);
    expect_tag(TK_LPAREN);
    firstp = 0 - 1; lastp = 0 - 1;
    while (p_tag() != TK_RPAREN && p_tag() != TK_EOF) {
        ptok = CUR;
        expect_tag(TK_IDENT);
        if (p_tag() == TK_COLON) { p_adv(); if (p_tag() == TK_IDENT) { p_adv(); } }
        pnode = node_new(ND_PARAM, ptok, 0, 0);
        if (firstp == 0 - 1) { firstp = pnode; lastp = pnode; }
        else { nd_set_next(lastp, pnode); lastp = pnode; }
        if (p_tag() == TK_COMMA) { p_adv(); }
    }
    expect_tag(TK_RPAREN);
    if (p_tag() == TK_ARROW) { p_adv(); if (p_tag() == TK_IDENT) { p_adv(); } }
    body = parse_block();
    return node_new(ND_FN, name, firstp, body);
}

/* program = fn*  ->  first fn node (rest chained via nd_next), or -1 */
int parse_program() {
    int first; int last; int fn;
    first = 0 - 1; last = 0 - 1;
    while (p_tag() == TK_FN) {
        fn = parse_fn();
        if (first == 0 - 1) { first = fn; last = fn; }
        else { nd_set_next(last, fn); last = fn; }
    }
    return first;
}

/* ===================== codegen: x86-64 self-contained ELF (increment 3a) ===================== *
 * Emit the same self-contained static-ELF shape kovc uses: ELF64 header + one
 * PT_LOAD (R|W|X) at 0x400000, code at file offset 0x1000 (entry 0x401000),
 * `_start` calls main then sys_exit with its return. INC 3a handles only a tail
 * integer literal; later increments grow expression/statement/call codegen. */
char* IMG;   /* whole output image: headers + pad + code            */
int IMGN;    /* write cursor (code begins at file offset 0x1000)     */
int PROG;    /* parsed program (first fn node)                       */

int img_init() {
    IMG = calloc(8 * 1048576, 1);   /* 8 MiB output image */
    IMGN = 4096;                     /* code begins at file offset 0x1000 */
    return 0;
}
int emit_byte(int b) { IMG[IMGN] = b; IMGN = IMGN + 1; return 0; }
int emit_u32le(int v) {
    emit_byte(v & 255); emit_byte((v >> 8) & 255);
    emit_byte((v >> 16) & 255); emit_byte((v >> 24) & 255);
    return 0;
}
/* poke fixed-width little-endian values at an absolute image position */
int put_u16(int pos, int v) { IMG[pos] = v & 255; IMG[pos + 1] = (v >> 8) & 255; return 0; }
int put_u32(int pos, int v) { put_u16(pos, v); put_u16(pos + 2, v >> 16); return 0; }
int put_u64(int pos, int v) { put_u32(pos, v); put_u32(pos + 4, 0); return 0; }   /* v < 2^32 */

/* fill the ELF + program headers now that the total size (IMGN) is known */
int build_headers() {
    int total;
    total = IMGN;
    IMG[0] = 127; IMG[1] = 69; IMG[2] = 76; IMG[3] = 70;   /* 0x7F 'E' 'L' 'F' */
    IMG[4] = 2;   /* ELFCLASS64 */
    IMG[5] = 1;   /* little-endian */
    IMG[6] = 1;   /* EI_VERSION */
    put_u16(16, 2);        /* e_type = ET_EXEC      */
    put_u16(18, 62);       /* e_machine = x86-64    */
    put_u32(20, 1);        /* e_version             */
    put_u64(24, 4198400);  /* e_entry = 0x401000    */
    put_u64(32, 64);       /* e_phoff               */
    put_u64(40, 0);        /* e_shoff               */
    put_u32(48, 0);        /* e_flags               */
    put_u16(52, 64);       /* e_ehsize              */
    put_u16(54, 56);       /* e_phentsize           */
    put_u16(56, 1);        /* e_phnum               */
    put_u16(58, 0);        /* e_shentsize           */
    put_u16(60, 0);        /* e_shnum               */
    put_u16(62, 0);        /* e_shstrndx            */
    put_u32(64, 1);        /* p_type = PT_LOAD      */
    put_u32(68, 7);        /* p_flags = R|W|X       */
    put_u64(72, 0);        /* p_offset              */
    put_u64(80, 4194304);  /* p_vaddr = 0x400000    */
    put_u64(88, 4194304);  /* p_paddr = 0x400000    */
    put_u64(96, total);    /* p_filesz              */
    put_u64(104, total);   /* p_memsz               */
    put_u64(112, 4096);    /* p_align = 0x1000      */
    return 0;
}

/* ----- per-function locals: name -> stack slot (increment 3c) ----- *
 * Each local (param or let) gets a slot; slot i lives at [rbp - 8*(i+1)].
 * Names are matched by their source span (idents are not interned). */
int* LOCAL_TOK;   /* LOCAL_TOK[slot] = the token index of that local's name */
int LOCAL_N;

int spans_eq(int sa, int la, int sb, int lb) {
    int i;
    if (la != lb) { return 0; }
    i = 0;
    while (i < la) { if (SRC[sa + i] != SRC[sb + i]) { return 0; } i = i + 1; }
    return 1;
}
int locals_init() { LOCAL_TOK = calloc(8192, sizeof(int)); LOCAL_N = 0; return 0; }
int locals_reset() { LOCAL_N = 0; return 0; }
int local_find(int name_tok) {     /* -> slot, or -1 */
    int i; int s; int l;
    s = tok_start(name_tok); l = tok_len(name_tok);
    i = 0;
    while (i < LOCAL_N) {
        if (spans_eq(s, l, tok_start(LOCAL_TOK[i]), tok_len(LOCAL_TOK[i]))) { return i; }
        i = i + 1;
    }
    return 0 - 1;
}
int local_add(int name_tok) {      /* assign (or reuse) a slot */
    int slot;
    slot = local_find(name_tok);
    if (slot != 0 - 1) { return slot; }
    LOCAL_TOK[LOCAL_N] = name_tok;
    LOCAL_N = LOCAL_N + 1;
    return LOCAL_N - 1;
}
/* pre-pass: collect params + top-level lets so the frame size is known up front
 * (nested-block lets are collected when while/if codegen lands in inc 3d). */
int collect_locals(int fn) {
    int p; int body; int stmt;
    locals_reset();
    p = nd_b(fn);
    while (p != 0 - 1) { local_add(nd_a(p)); p = nd_next(p); }
    body = nd_c(fn);
    stmt = nd_a(body);
    while (stmt != 0 - 1) {
        if (nd_kind(stmt) == ND_LET) { local_add(nd_a(stmt)); }
        stmt = nd_next(stmt);
    }
    return 0;
}
/* mov eax, [rbp - 8*(slot+1)]  /  mov [rbp - 8*(slot+1)], eax  (disp32 form) */
int emit_load_local(int slot) {
    emit_byte(0x8B); emit_byte(0x85); emit_u32le(0 - 8 * (slot + 1)); return 0;
}
int emit_store_local(int slot) {
    emit_byte(0x89); emit_byte(0x85); emit_u32le(0 - 8 * (slot + 1)); return 0;
}

/* forward declarations (cg_expr <-> cg_bin <-> cg_stmt are mutually recursive) */
int cg_expr(int node);
int cg_bin(int op, int left, int right);
int cg_stmt(int node);

/* codegen a binary op: eval left -> rax (saved), right -> rcx, then the op.
 * Result in eax. left in rax, right in rcx after the setup below. */
int cg_bin(int op, int left, int right) {
    cg_expr(left);
    emit_byte(0x50);                                    /* push rax            */
    cg_expr(right);
    emit_byte(0x48); emit_byte(0x89); emit_byte(0xC1);  /* mov rcx, rax (right)*/
    emit_byte(0x58);                                    /* pop rax (left)      */
    if (op == TK_PLUS)    { emit_byte(0x01); emit_byte(0xC8); return 0; }                   /* add eax,ecx  */
    if (op == TK_MINUS)   { emit_byte(0x29); emit_byte(0xC8); return 0; }                   /* sub eax,ecx  */
    if (op == TK_STAR)    { emit_byte(0x0F); emit_byte(0xAF); emit_byte(0xC1); return 0; }  /* imul eax,ecx */
    if (op == TK_SLASH)   { emit_byte(0x99); emit_byte(0xF7); emit_byte(0xF9); return 0; }  /* cdq; idiv ecx -> eax */
    if (op == TK_PERCENT) { emit_byte(0x99); emit_byte(0xF7); emit_byte(0xF9);
                            emit_byte(0x89); emit_byte(0xD0); return 0; }                   /* cdq; idiv ecx; mov eax,edx */
    if (op == TK_AMP)     { emit_byte(0x21); emit_byte(0xC8); return 0; }                   /* and eax,ecx  */
    if (op == TK_PIPE)    { emit_byte(0x09); emit_byte(0xC8); return 0; }                   /* or  eax,ecx  */
    if (op == TK_CARET)   { emit_byte(0x31); emit_byte(0xC8); return 0; }                   /* xor eax,ecx  */
    if (op == TK_EQEQ || op == TK_NE || op == TK_LT || op == TK_LE || op == TK_GT || op == TK_GE) {
        emit_byte(0x39); emit_byte(0xC8);               /* cmp eax, ecx        */
        emit_byte(0x0F);
        if (op == TK_EQEQ)    { emit_byte(0x94); }      /* sete  */
        else if (op == TK_NE) { emit_byte(0x95); }      /* setne */
        else if (op == TK_LT) { emit_byte(0x9C); }      /* setl  */
        else if (op == TK_LE) { emit_byte(0x9E); }      /* setle */
        else if (op == TK_GT) { emit_byte(0x9F); }      /* setg  */
        else                  { emit_byte(0x9D); }      /* setge */
        emit_byte(0xC0);                                /* setcc al            */
        emit_byte(0x0F); emit_byte(0xB6); emit_byte(0xC0); /* movzx eax, al    */
        return 0;
    }
    if (op == TK_ANDAND) {   /* non-short-circuit: (left!=0) * (right!=0) */
        emit_byte(0x85); emit_byte(0xC0);                      /* test eax,eax  */
        emit_byte(0x0F); emit_byte(0x95); emit_byte(0xC0);     /* setne al      */
        emit_byte(0x0F); emit_byte(0xB6); emit_byte(0xC0);     /* movzx eax,al  */
        emit_byte(0x85); emit_byte(0xC9);                      /* test ecx,ecx  */
        emit_byte(0x0F); emit_byte(0x95); emit_byte(0xC1);     /* setne cl      */
        emit_byte(0x0F); emit_byte(0xB6); emit_byte(0xC9);     /* movzx ecx,cl  */
        emit_byte(0x0F); emit_byte(0xAF); emit_byte(0xC1);     /* imul eax,ecx  */
        return 0;
    }
    if (op == TK_OROR) {     /* non-short-circuit: (left|right) != 0 */
        emit_byte(0x09); emit_byte(0xC8);                      /* or eax,ecx    */
        emit_byte(0x85); emit_byte(0xC0);                      /* test eax,eax  */
        emit_byte(0x0F); emit_byte(0x95); emit_byte(0xC0);     /* setne al      */
        emit_byte(0x0F); emit_byte(0xB6); emit_byte(0xC0);     /* movzx eax,al  */
        return 0;
    }
    return 0;
}

/* codegen an expression -> result in eax (INC 3b: int literals + binary ops) */
int cg_expr(int node) {
    int k;
    k = nd_kind(node);
    if (k == ND_INT) {
        emit_byte(0xB8);              /* mov eax, imm32 */
        emit_u32le(nd_a(node));
        return 0;
    }
    if (k == ND_BIN) {
        return cg_bin(nd_a(node), nd_b(node), nd_c(node));
    }
    if (k == ND_VAR) {
        emit_load_local(local_find(nd_a(node)));   /* mov eax, [rbp - slot] */
        return 0;
    }
    emit_byte(0xB8); emit_u32le(0);   /* ND_CALL/ND_IF -> later increments */
    return 0;
}

/* codegen a statement (INC 3c: let / assign / expr-statement) */
int cg_stmt(int node) {
    int k;
    k = nd_kind(node);
    if (k == ND_LET) {
        cg_expr(nd_b(node));                       /* init -> eax */
        emit_store_local(local_find(nd_a(node)));
        return 0;
    }
    if (k == ND_ASSIGN) {
        cg_expr(nd_b(node));                       /* rhs -> eax  */
        emit_store_local(local_find(nd_a(node)));
        return 0;
    }
    cg_expr(node);                                 /* expr-statement: value discarded */
    return 0;
}

/* codegen a function: reserve a frame for locals, run the statements, then
 * the tail expression -> eax, then epilogue + ret. */
int cg_fn(int fn) {
    int body; int stmt; int tail; int frame;
    collect_locals(fn);
    frame = LOCAL_N * 8;
    frame = ((frame + 15) / 16) * 16;                  /* 16-byte align   */
    emit_byte(0x55);                                   /* push rbp        */
    emit_byte(0x48); emit_byte(0x89); emit_byte(0xE5); /* mov rbp, rsp    */
    if (frame > 0) {
        emit_byte(0x48); emit_byte(0x81); emit_byte(0xEC); emit_u32le(frame); /* sub rsp, frame */
    }
    body = nd_c(fn);
    stmt = nd_a(body);
    while (stmt != 0 - 1) { cg_stmt(stmt); stmt = nd_next(stmt); }
    tail = nd_c(body);
    if (tail != 0 - 1) { cg_expr(tail); }
    else { emit_byte(0xB8); emit_u32le(0); }           /* default eax = 0 */
    emit_byte(0x48); emit_byte(0x89); emit_byte(0xEC); /* mov rsp, rbp    */
    emit_byte(0x5D);                                   /* pop rbp         */
    emit_byte(0xC3);                                   /* ret             */
    return 0;
}

/* emit _start (call main; sys_exit eax) then the program's functions */
int codegen() {
    int call_rel_pos; int main_off; int rel;
    emit_byte(0xE8);                  /* call rel32 -> main          */
    call_rel_pos = IMGN;
    emit_u32le(0);                    /* rel32 placeholder           */
    emit_byte(0x89); emit_byte(0xC7); /* mov edi, eax (exit status)  */
    emit_byte(0xB8); emit_u32le(60);  /* mov eax, 60 (sys_exit)      */
    emit_byte(0x0F); emit_byte(0x05); /* syscall                     */
    main_off = IMGN;
    cg_fn(PROG);                      /* INC 3a: the single fn IS main */
    rel = main_off - (call_rel_pos + 4);
    put_u32(call_rel_pos, rel);
    return 0;
}

/* read a whole file into SRC / SRC_LEN */
int read_file(char* path) {
    FILE* f; int c;
    f = fopen(path, "r");
    SRC = calloc(4 * 1048576, 1);
    SRC_LEN = 0;
    c = fgetc(f);
    while (c != 0 - 1) { SRC[SRC_LEN] = c; SRC_LEN = SRC_LEN + 1; c = fgetc(f); }
    fclose(f);
    return 0;
}

/* write the built image to disk */
int write_image(char* path) {
    FILE* f;
    build_headers();
    f = fopen(path, "w");
    fwrite(IMG, 1, IMGN, f);
    fclose(f);
    return 0;
}

/* ===================== self-tests ===================== */
int lex_str(char* s) {
    SRC = s;
    SRC_LEN = cstr_len(s);
    lex();
    return 0;
}

/* inc 1: lex `fn main() -> i32 { let x = 41; x + 1 }`, assert the token stream */
int check_lexer() {
    lex_str("fn main() -> i32 { let x = 41; x + 1 }");
    if (TOK_N != 17)             { return 1; }
    if (tok_tag(0) != TK_FN)     { return 2; }
    if (tok_tag(4) != TK_ARROW)  { return 3; }
    if (tok_tag(6) != TK_LBRACE) { return 4; }
    if (tok_tag(7) != TK_LET)    { return 5; }
    if (tok_tag(10) != TK_INT)   { return 6; }
    if (tok_val(10) != 41)       { return 7; }
    if (tok_tag(14) != TK_INT)   { return 8; }
    if (tok_val(14) != 1)        { return 9; }
    if (tok_tag(16) != TK_EOF)   { return 10; }
    return 0;
}

/* inc 2a: parse expressions; assert precedence, parens, and a call */
int check_parser() {
    int root; int l; int r; int rr;
    /* precedence: 2 + 3 * 4  =>  +( 2, *(3,4) ) */
    lex_str("2 + 3 * 4"); nodes_reset(); CUR = 0;
    root = parse_expr();
    if (nd_kind(root) != ND_BIN)  { return 1; }
    if (nd_a(root) != TK_PLUS)    { return 2; }
    l = nd_b(root); r = nd_c(root);
    if (nd_kind(l) != ND_INT)     { return 3; }
    if (nd_a(l) != 2)             { return 4; }
    if (nd_kind(r) != ND_BIN)     { return 5; }
    if (nd_a(r) != TK_STAR)       { return 6; }
    rr = nd_b(r);
    if (nd_kind(rr) != ND_INT)    { return 7; }
    if (nd_a(rr) != 3)            { return 8; }
    if (nd_a(nd_c(r)) != 4)       { return 9; }
    /* parens override: (2 + 3) * 4  =>  *( +(2,3), 4 ) */
    lex_str("(2 + 3) * 4"); nodes_reset(); CUR = 0;
    root = parse_expr();
    if (nd_kind(root) != ND_BIN)       { return 10; }
    if (nd_a(root) != TK_STAR)         { return 11; }
    if (nd_kind(nd_b(root)) != ND_BIN) { return 12; }
    if (nd_a(nd_b(root)) != TK_PLUS)   { return 13; }
    if (nd_a(nd_c(root)) != 4)         { return 14; }
    /* call: f(7, x)  =>  ND_CALL with two chained args (7 then x) */
    lex_str("f(7, x)"); nodes_reset(); CUR = 0;
    root = parse_expr();
    if (nd_kind(root) != ND_CALL)               { return 15; }
    if (nd_kind(nd_b(root)) != ND_INT)          { return 16; }
    if (nd_a(nd_b(root)) != 7)                  { return 17; }
    if (nd_kind(nd_next(nd_b(root))) != ND_VAR) { return 18; }
    return 0;
}

/* inc 2b: parse full functions; assert the statement/fn/block AST shapes */
int check_parser_stmts() {
    int prog; int fn; int body; int s1; int s2;
    /* a function with let-mut, an assignment, and a tail expression */
    lex_str("fn main() -> i32 { let mut x = 41; x = x + 1; x }");
    nodes_reset(); CUR = 0; PERR = 0;
    prog = parse_program();
    if (PERR != 0)                     { return 1; }
    if (prog == 0 - 1)                 { return 2; }
    fn = prog;
    if (nd_kind(fn) != ND_FN)          { return 3; }
    body = nd_c(fn);
    if (nd_kind(body) != ND_BLOCK)     { return 4; }
    s1 = nd_a(body);                   /* let mut x = 41 */
    if (nd_kind(s1) != ND_LET)         { return 5; }
    if (nd_c(s1) != 1)                 { return 6; }   /* mut flag */
    if (nd_kind(nd_b(s1)) != ND_INT)   { return 7; }
    if (nd_a(nd_b(s1)) != 41)          { return 8; }   /* init value 41 */
    s2 = nd_next(s1);                  /* x = x + 1 */
    if (nd_kind(s2) != ND_ASSIGN)      { return 9; }
    if (nd_kind(nd_b(s2)) != ND_BIN)   { return 10; }
    if (nd_next(s2) != 0 - 1)          { return 11; }  /* no more chained stmts */
    if (nd_kind(nd_c(body)) != ND_VAR) { return 12; }  /* tail expr = x */

    /* a function with a while loop and an if-expression tail */
    lex_str("fn f() -> i32 { let mut i = 0; while i < 10 { i = i + 1; } if i > 5 { 1 } else { 0 } }");
    nodes_reset(); CUR = 0; PERR = 0;
    prog = parse_program();
    if (PERR != 0)                     { return 13; }
    fn = prog; body = nd_c(fn);
    s1 = nd_a(body);                   /* let mut i = 0 */
    if (nd_kind(s1) != ND_LET)         { return 14; }
    s2 = nd_next(s1);                  /* while i < 10 { ... } */
    if (nd_kind(s2) != ND_WHILE)       { return 15; }
    if (nd_kind(nd_a(s2)) != ND_BIN)   { return 16; }  /* cond i < 10 */
    if (nd_kind(nd_b(s2)) != ND_BLOCK) { return 17; }  /* while body block */
    if (nd_kind(nd_c(body)) != ND_IF)  { return 18; }  /* tail = if-expr */
    return 0;
}

int main(int argc, char** argv) {
    int rc;
    arena_init();
    tags_init();
    TOK = calloc(65536, sizeof(int));
    nodes_init();

    if (argc < 3) {
        /* no input/output args -> run the lexer + parser self-tests */
        rc = check_lexer();        if (rc != 0) { return rc; }
        rc = check_parser();       if (rc != 0) { return 20 + rc; }
        rc = check_parser_stmts(); if (rc != 0) { return 50 + rc; }
        return 42;
    }

    /* compile mode: argv[1] = input .hx, argv[2] = output ELF path */
    img_init();
    locals_init();
    read_file(argv[1]);
    lex();
    CUR = 0; PERR = 0;
    PROG = parse_program();
    if (PERR != 0)     { return 90; }   /* parse error */
    if (PROG == 0 - 1) { return 91; }   /* no function found */
    codegen();
    write_image(argv[2]);
    return 0;
}
