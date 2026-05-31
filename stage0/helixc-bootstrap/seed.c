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

/* ===================== increment-1 self-test ===================== *
 * Lex `fn main() -> i32 { let x = 41; x + 1 }` and assert the token stream.
 * Returns 42 on success, or a small diagnostic code at the first failed check.
 */
int main() {
    arena_init();
    tags_init();
    SRC = "fn main() -> i32 { let x = 41; x + 1 }";
    SRC_LEN = cstr_len(SRC);
    TOK = calloc(4096, sizeof(int));
    lex();

    if (TOK_N != 17)            { return 1; }   /* 16 tokens + EOF */
    if (tok_tag(0) != TK_FN)    { return 2; }
    if (tok_tag(1) != TK_IDENT) { return 3; }
    if (tok_tag(2) != TK_LPAREN){ return 4; }
    if (tok_tag(3) != TK_RPAREN){ return 5; }
    if (tok_tag(4) != TK_ARROW) { return 6; }
    if (tok_tag(5) != TK_IDENT) { return 7; }   /* i32 (type word = ident) */
    if (tok_tag(6) != TK_LBRACE){ return 8; }
    if (tok_tag(7) != TK_LET)   { return 9; }
    if (tok_tag(8) != TK_IDENT) { return 10; }  /* x */
    if (tok_tag(9) != TK_EQ)    { return 11; }
    if (tok_tag(10) != TK_INT)  { return 12; }
    if (tok_val(10) != 41)      { return 13; }
    if (tok_tag(11) != TK_SEMI) { return 14; }
    if (tok_tag(12) != TK_IDENT){ return 15; }  /* x */
    if (tok_tag(13) != TK_PLUS) { return 16; }
    if (tok_tag(14) != TK_INT)  { return 17; }
    if (tok_val(14) != 1)       { return 18; }
    if (tok_tag(15) != TK_RBRACE){ return 19; }
    if (tok_tag(16) != TK_EOF)  { return 20; }
    return 42;
}
