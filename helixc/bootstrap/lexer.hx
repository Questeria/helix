// Stage-0 lexer for the Helix bootstrap compiler.
//
// Reads a source file via read_file_to_arena, walks the resulting
// byte sequence, and emits a stream of tokens into a separate arena
// region. Each token occupies 4 i32 slots:
//
//   [tag, payload, src_start, src_len]
//
// tag values (small enum, expanded over time):
//   0  TK_EOF
//   1  TK_INT       payload = the integer value
//   2  TK_IDENT     payload = byte index of the first character; src_len = byte length
//   3  TK_LPAREN    "("
//   4  TK_RPAREN    ")"
//   5  TK_LBRACE    "{"
//   6  TK_RBRACE    "}"
//   7  TK_PLUS      "+"
//   8  TK_MINUS     "-"
//   9  TK_STAR      "*"
//  10  TK_SLASH     "/"
//  11  TK_PERCENT   "%"
//  12  TK_SEMI      ";"
//  13  TK_COMMA     ","
//  14  TK_COLON     ":"
//  15  TK_EQ        "="
//  16  TK_LT        "<"
//  17  TK_GT        ">"
//  18  TK_BANG      "!"
//
// Whitespace (space, tab, newline, CR) and `//` line comments are
// skipped silently. Keywords like `let`, `fn`, `if` are emitted as
// TK_IDENT here; a post-lex keyword pass (or the parser) does the
// final classification.
//
// The lexer assumes the source bytes live in slots [src_start ..
// src_start + src_len) of the arena. The token stream begins at
// the arena cursor when lex() is called and ends at whatever cursor
// is when lex() returns; the caller can compare cursor before/after
// to learn the token count divided by 4.
//
// Runnable today as a Phase-0 program. Once the parser exists, this
// file becomes the input to it. License: Apache 2.0.

// --------------------------------------------------------------
// Byte classification helpers — each takes the byte value (i32)
// returned by __arena_get and returns 1/0.
// --------------------------------------------------------------
@pure
fn is_whitespace(b: i32) -> i32 {
    if b == 32 { 1 }
    else { if b == 9 { 1 }
    else { if b == 10 { 1 }
    else { if b == 13 { 1 }
    else { 0 }}}}
}

@pure
fn is_digit(b: i32) -> i32 {
    if b >= 48 { if b <= 57 { 1 } else { 0 } } else { 0 }
}

@pure
fn is_alpha(b: i32) -> i32 {
    if b >= 65 { if b <= 90 { 1 }            // 'A'..'Z'
    else { if b >= 97 { if b <= 122 { 1 }    // 'a'..'z'
    else { 0 }} else { 0 }}}
    else { if b == 95 { 1 } else { 0 }}      // '_'
}

@pure
fn is_alnum(b: i32) -> i32 {
    if is_digit(b) == 1 { 1 }
    else { if is_alpha(b) == 1 { 1 } else { 0 }}
}

// --------------------------------------------------------------
// Push a 4-slot token record into the arena. Returns the slot index
// of the tag (the start of this token's record).
// --------------------------------------------------------------
fn push_token(tag: i32, payload: i32, src_start: i32, src_len: i32) -> i32 {
    let i = __arena_push(tag);
    __arena_push(payload);
    __arena_push(src_start);
    __arena_push(src_len);
    i
}

// --------------------------------------------------------------
// Skip a `//` line comment starting at byte index `pos`. Assumes
// caller has already verified bytes [pos..pos+2) == "//". Advances
// past the trailing newline (or EOF). Returns the new byte index.
// --------------------------------------------------------------
fn skip_line_comment(src_start: i32, src_len: i32, pos: i32) -> i32 {
    let mut p: i32 = pos + 2;
    let end = src_start + src_len;
    let mut keep: i32 = 1;
    while keep == 1 {
        if p >= end {
            keep = 0;
        } else {
            let b = __arena_get(p);
            if b == 10 {     // '\n'
                p = p + 1;
                keep = 0;
            } else {
                p = p + 1;
            };
        };
    }
    p
}

// --------------------------------------------------------------
// Lex an integer literal starting at byte index `pos`. Reads digits
// while is_digit() holds, accumulates the value, emits a single
// TK_INT token. Returns the byte index after the last digit.
// --------------------------------------------------------------
fn lex_int(src_start: i32, src_len: i32, pos: i32) -> i32 {
    let mut p: i32 = pos;
    let end = src_start + src_len;
    let mut value: i32 = 0;
    let mut keep: i32 = 1;
    while keep == 1 {
        if p >= end {
            keep = 0;
        } else {
            let b = __arena_get(p);
            if is_digit(b) == 1 {
                value = value * 10 + (b - 48);
                p = p + 1;
            } else {
                keep = 0;
            };
        };
    }
    let length = p - pos;
    push_token(1, value, pos, length);
    p
}

// --------------------------------------------------------------
// Lex an identifier starting at byte index `pos`. Reads while
// is_alnum holds, emits a TK_IDENT token whose payload is the
// start byte index. Returns byte index after the identifier.
// --------------------------------------------------------------
fn lex_ident(src_start: i32, src_len: i32, pos: i32) -> i32 {
    let mut p: i32 = pos;
    let end = src_start + src_len;
    let mut keep: i32 = 1;
    while keep == 1 {
        if p >= end {
            keep = 0;
        } else {
            let b = __arena_get(p);
            if is_alnum(b) == 1 {
                p = p + 1;
            } else {
                keep = 0;
            };
        };
    }
    let length = p - pos;
    push_token(2, pos, pos, length);
    p
}

// --------------------------------------------------------------
// Lex a single-character punctuation token. Returns the kind tag
// (3..18 per the table at the top of this file), or 0 if `b` is
// not a known punctuation byte.
// --------------------------------------------------------------
@pure
fn punct_kind(b: i32) -> i32 {
    if b == 40 { 3 }       // '('
    else { if b == 41 { 4 }       // ')'
    else { if b == 123 { 5 }      // '{'
    else { if b == 125 { 6 }      // '}'
    else { if b == 43 { 7 }       // '+'
    else { if b == 45 { 8 }       // '-'
    else { if b == 42 { 9 }       // '*'
    else { if b == 47 { 10 }      // '/'
    else { if b == 37 { 11 }      // '%'
    else { if b == 59 { 12 }      // ';'
    else { if b == 44 { 13 }      // ','
    else { if b == 58 { 14 }      // ':'
    else { if b == 61 { 15 }      // '='
    else { if b == 60 { 16 }      // '<'
    else { if b == 62 { 17 }      // '>'
    else { if b == 33 { 18 }      // '!'
    else { 0 }}}}}}}}}}}}}}}}
}

// --------------------------------------------------------------
// Main loop: walk the source bytes, dispatching to per-kind
// helpers. Returns the number of tokens emitted (counting EOF).
// --------------------------------------------------------------
fn lex(src_start: i32, src_len: i32) -> i32 {
    let token_base = __arena_len();
    let mut pos: i32 = src_start;
    let end = src_start + src_len;
    while pos < end {
        let b = __arena_get(pos);
        if is_whitespace(b) == 1 {
            pos = pos + 1;
        } else { if b == 47 {
            // Possible '//' line comment, else slash punctuation.
            if pos + 1 < end {
                let nxt = __arena_get(pos + 1);
                if nxt == 47 {
                    pos = skip_line_comment(src_start, src_len, pos);
                } else {
                    push_token(10, 0, pos, 1);
                    pos = pos + 1;
                };
            } else {
                push_token(10, 0, pos, 1);
                pos = pos + 1;
            };
        } else { if is_digit(b) == 1 {
            pos = lex_int(src_start, src_len, pos);
        } else { if is_alpha(b) == 1 {
            pos = lex_ident(src_start, src_len, pos);
        } else {
            let pk = punct_kind(b);
            if pk == 0 {
                // Unknown byte — emit it with tag 19 so the caller
                // can detect and report a lex error without crashing.
                push_token(19, b, pos, 1);
                pos = pos + 1;
            } else {
                push_token(pk, 0, pos, 1);
                pos = pos + 1;
            };
        }}}};
    }
    push_token(0, 0, pos, 0);   // TK_EOF sentinel
    let after = __arena_len();
    (after - token_base) / 4
}

// --------------------------------------------------------------
// Demo: load a file via read_file_to_arena and lex it. Returns
// the token count. Run with /tmp/helix_lex_input.hx prepared.
// --------------------------------------------------------------
fn main() -> i32 {
    let src_start = __arena_len();
    let src_len = read_file_to_arena("/tmp/helix_lex_input.hx");
    if src_len <= 0 { 0 - 1 }
    else { lex(src_start, src_len) }
}
