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
//  23  TK_TILDE     "~"
//  25  TK_STRLIT    payload = body byte_start, src_len = body length
//  27  TK_AMP       "&"   (binary bitwise AND; bootstrap has no refs)
//  28  TK_PIPE      "|"   (binary bitwise OR)
//  29  TK_CARET     "^"   (binary bitwise XOR)
//                  (BOTH exclude the surrounding quotes). Phase-0 has
//                  no escape sequences — `\` inside a string is taken
//                  literally. Multi-line strings allowed.
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
    // '_' (byte 95) lives BETWEEN 'A' (65) and 'z' (122), so check
    // it before the case-letter ranges. The previous structure put
    // the underscore check in the `else` of `b >= 65`, which is
    // unreachable for byte 95 — silently broke any identifier
    // containing an underscore (e.g. `sum_to` lexed as `sum`,
    // TK_ERR, `to`).
    if b == 95 { 1 }
    else { if b >= 65 {
        if b <= 90 { 1 }
        else { if b >= 97 {
            if b <= 122 { 1 } else { 0 }
        } else { 0 }}
    } else { 0 }}
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
// Lex an integer literal starting at byte index `pos`. Recognises a
// `0x` / `0X` prefix for hex literals (digits 0-9, a-f, A-F); falls
// back to decimal otherwise. Emits a single TK_INT token whose
// payload is the parsed value. Returns the byte index after the
// last consumed digit.
//
// The Python reference lexer (helixc/frontend/lexer.py) accepts
// `0x401000` and friends; the bootstrap MUST do the same, otherwise
// any hex literal in `kovc.hx` (e.g. `emit_u64_le_split(0x401000,
// 0)` in `emit_elf_header`) silently splits into a TK_INT(0)
// followed by an `x401000` IDENT, which the parser folds into an
// extra call argument that codegen resolves to `[rbp+0x0]` (the
// saved rbp slot). Result: the self-compiled K2 emits an ELF whose
// e_entry is a stack pointer, and any further use of K2 segfaults.
// --------------------------------------------------------------
fn lex_int(src_start: i32, src_len: i32, pos: i32) -> i32 {
    let mut p: i32 = pos;
    let end = src_start + src_len;
    let mut value: i32 = 0;
    // Detect a `0x` / `0X` prefix. We need at least two more bytes
    // and the first must be `0`; otherwise fall through to decimal.
    let mut is_hex: i32 = 0;
    if p + 1 < end {
        let c0 = __arena_get(p);
        let c1 = __arena_get(p + 1);
        if c0 == 48 {
            // 'x' = 120, 'X' = 88
            if c1 == 120 { is_hex = 1; }
            else { if c1 == 88 { is_hex = 1; } };
        };
    };
    if is_hex == 1 {
        p = p + 2;     // consume `0x`
        let mut keep_h: i32 = 1;
        while keep_h == 1 {
            if p >= end {
                keep_h = 0;
            } else {
                let b = __arena_get(p);
                // 0..9 → b-48; a..f → b-87; A..F → b-55
                if is_digit(b) == 1 {
                    value = value * 16 + (b - 48);
                    p = p + 1;
                } else { if b >= 97 {
                    if b <= 102 {
                        value = value * 16 + (b - 87);
                        p = p + 1;
                    } else { keep_h = 0; }
                } else { if b >= 65 {
                    if b <= 70 {
                        value = value * 16 + (b - 55);
                        p = p + 1;
                    } else { keep_h = 0; }
                } else { keep_h = 0; }};
                };
            };
        }
    } else {
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
    }
    // Phase 1.10 float-literal lookahead: if we hit `.` AND the next
    // byte is a digit, this is a float (e.g. `1.5`). Switch to float
    // lexing — keep consuming digits and emit TK_FLOATLIT (tag 26).
    // The token's payload carries byte_start + byte_len of the LITERAL
    // TEXT (similar to TK_STRLIT); the parser/codegen must convert
    // the text to IEEE 754 bits at parse time. (Direct bit-conversion
    // here would need 8-byte arithmetic which Phase-0 Helix lacks.)
    let mut is_float: i32 = 0;
    if is_hex == 0 {
        if p < end {
            let dot = __arena_get(p);
            if dot == 46 {
                if p + 1 < end {
                    let nxt = __arena_get(p + 1);
                    if is_digit(nxt) == 1 {
                        is_float = 1;
                        p = p + 1;
                        let mut keep_f: i32 = 1;
                        while keep_f == 1 {
                            if p >= end {
                                keep_f = 0;
                            } else {
                                let b = __arena_get(p);
                                if is_digit(b) == 1 { p = p + 1; }
                                else { keep_f = 0; };
                            };
                        }
                    };
                };
            };
        };
    }
    // Phase 1.10 step 5a: optional `_f32` / `_f64` / `_i32` / `_i64` type
    // suffix on numeric literals. The lexer consumes them so they don't
    // appear as a separate IDENT token. Step 7a: also TRACK whether the
    // suffix was `_f64` so a distinct TK_FLOATLIT_F64 token (tag 32) can
    // be emitted — the parser then knows the literal needs 8-byte f64
    // codegen (step 7b+) instead of 4-byte f32.
    let mut is_f64_suffix: i32 = 0;
    let mut is_i64_suffix: i32 = 0;
    let mut is_u32_suffix: i32 = 0;
    let mut is_u8_suffix: i32 = 0;
    let mut is_u64_suffix: i32 = 0;
    let mut is_i8_suffix: i32 = 0;
    let mut is_i16_suffix: i32 = 0;
    let mut is_u16_suffix: i32 = 0;
    let mut is_bf16_suffix: i32 = 0;
    if p + 3 < end {
        let b0 = __arena_get(p);
        if b0 == 95 {   // '_'
            let b1 = __arena_get(p + 1);
            let b2 = __arena_get(p + 2);
            let b3 = __arena_get(p + 3);
            // f32 / f64
            if b1 == 102 {
                if b2 == 51 {
                    if b3 == 50 { p = p + 4; };   // _f32
                };
                if b2 == 54 {
                    if b3 == 52 {                   // _f64
                        p = p + 4;
                        is_f64_suffix = 1;
                    };
                };
            };
            // i32 / i64 — Stage 1 (Approach A): track _i64 suffix so a
            // distinct TK_INTLIT_I64 token (tag 33) can be emitted. Parser
            // produces an AST_INTLIT_I64 node; codegen emits 8-byte movabs
            // instead of 4-byte mov eax, imm32. Values that fit in i32
            // are still fine; large values defer to a later stage.
            if b1 == 105 {
                if b2 == 51 {
                    if b3 == 50 { p = p + 4; };   // _i32
                };
                if b2 == 54 {
                    if b3 == 52 {                   // _i64
                        p = p + 4;
                        is_i64_suffix = 1;
                    };
                };
                // Stage 2.5: _i8 (3 bytes only).
                if b2 == 56 {
                    p = p + 3;
                    is_i8_suffix = 1;
                };
                // Stage 2.5c: _i16 (4 bytes: '_' 'i' '1' '6').
                if b2 == 49 {
                    if b3 == 54 {                   // _i16
                        p = p + 4;
                        is_i16_suffix = 1;
                    };
                };
            };
            // Stage 2.1 (Approach A): _u32 suffix produces TK_INTLIT_U32
            // (tag 34). Codegen treats u32 literals identically to i32
            // (`mov eax, imm32`) — overflow wraps mod 2^32 for both, and
            // the SAME x86 add/sub/mul instructions work for signed and
            // unsigned operands. Only DIV/MOD and comparison ops differ
            // (idiv vs div, setl vs setb), which Stage 2.2 will dispatch.
            // Stage 2.3: _u8 suffix produces TK_INTLIT_U8 (tag 35).
            // Same codegen as i32 for the literal (mov eax, imm32 with
            // value masked to 0..255 by the parser); type tag 7 in
            // expr_type tracks u8-ness for unsigned dispatch in
            // DIV/MOD/comparisons.
            if b1 == 117 {                          // 'u'
                if b2 == 51 {
                    if b3 == 50 {                   // _u32
                        p = p + 4;
                        is_u32_suffix = 1;
                    };
                };
                if b2 == 56 {                       // _u8 (only 3 bytes)
                    p = p + 3;
                    is_u8_suffix = 1;
                };
                if b2 == 54 {
                    if b3 == 52 {                   // _u64
                        p = p + 4;
                        is_u64_suffix = 1;
                    };
                };
                // Stage 2.5c: _u16 (4 bytes: '_' 'u' '1' '6').
                if b2 == 49 {
                    if b3 == 54 {                   // _u16
                        p = p + 4;
                        is_u16_suffix = 1;
                    };
                };
            };
        };
    }
    // Stage 1.5: _bf16 (5 bytes: '_' 'b' 'f' '1' '6'). Brain Float 16
    // is f32 with the low 16 mantissa bits truncated. Codegen emits the
    // f32 bit pattern AND-masked with 0xFFFF0000.
    if p + 4 < end {
        let b0 = __arena_get(p);
        if b0 == 95 {                              // '_'
            let b1 = __arena_get(p + 1);
            if b1 == 98 {                          // 'b'
                let b2 = __arena_get(p + 2);
                if b2 == 102 {                     // 'f'
                    let b3 = __arena_get(p + 3);
                    if b3 == 49 {                  // '1'
                        let b4 = __arena_get(p + 4);
                        if b4 == 54 {              // '6'
                            p = p + 5;
                            is_bf16_suffix = 1;
                        };
                    };
                };
            };
        };
    }
    if is_float == 1 {
        let flen = p - pos;
        // Step 7a: distinguish f32 (tag 26) from f64 (tag 32) at lex time.
        // Stage 1.5: _bf16 suffix → token tag 41 (TK_FLOATLIT_BF16).
        // Parser routes to AST_FLOATLIT_BF16 (tag 42); codegen masks
        // low 16 mantissa bits.
        let tk = if is_bf16_suffix == 1 { 41 }
                 else { if is_f64_suffix == 1 { 32 } else { 26 } };
        push_token(tk, pos, pos, flen);
    } else { if is_bf16_suffix == 1 {
        // bf16 suffix on a non-float number (e.g., `42_bf16`). Treat
        // as a bf16 literal — emit the same token. Parser/codegen will
        // parse it through the floatlit machinery (which handles
        // missing decimal point gracefully — no fractional digits).
        let flen = p - pos;
        push_token(41, pos, pos, flen);
    } else {
        let length = p - pos;
        // Stage 1: TK_INTLIT_I64 (tag 33) for _i64-suffixed literals;
        // Stage 2.1: TK_INTLIT_U32 (tag 34) for _u32-suffixed literals;
        // Stage 2.3: TK_INTLIT_U8  (tag 35) for _u8-suffixed literals;
        // Stage 2.4: TK_INTLIT_U64 (tag 36) for _u64-suffixed literals;
        // TK_INT (tag 1) for plain or _i32-suffixed.
        // Stage 2.4b audit fix: loud-failure guard for u64 literal
        // overflow. The lex_int decimal accumulator is i32; values
        // >= 2^32 silently truncate. Cap: if the digit run is > 10
        // chars AND `_u64` suffix, emit token tag 40 (no parser arm
        // → falls through to AST_ERR → ud2 at codegen). 10 digits
        // accept up to 9_999_999_999 (slightly over 2^32-1 =
        // 4294967295); values in [2^32, 9_999_999_999] still wrap
        // silently — proper fix is the queued lex-side accumulator
        // widening. 11+ digits definitely overflow u64? No, u64 max
        // is 18_446_744_073_709_551_615 (20 digits). So 11+ digits
        // are NOT inherently invalid — they're just past what i32
        // can hold. The cap intentionally errs toward LESS conservative
        // here so legal valuese.g. 2147483648_u64 (= 2^31, 10 digits)
        // still work via the hi32=0 partial fix from commit 09c8858.
        let u64_oversize = if is_u64_suffix == 1 {
            if (length - 4) > 10 { 1 } else { 0 }
        } else { 0 };
        let tk = if u64_oversize == 1 { 40 }
                 else { if is_i64_suffix == 1 { 33 }
                 else { if is_u32_suffix == 1 { 34 }
                 else { if is_u8_suffix == 1 { 35 }
                 else { if is_u64_suffix == 1 { 36 }
                 else { if is_i8_suffix == 1 { 37 }
                 else { if is_i16_suffix == 1 { 38 }
                 else { if is_u16_suffix == 1 { 39 } else { 1 } } } } } } } };
        push_token(tk, value, pos, length);
    } };
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
// Lex a string literal. Caller has verified bytes[pos] == 34 ('"').
// Walks forward to the next unescaped '"' (Phase-0: no escape
// processing yet — '\' is treated as a literal byte). Emits a
// TK_STRLIT whose payload = body byte_start, src_len = body length
// (both EXCLUDE the quotes). Returns byte index AFTER the closing
// quote, or AFTER end-of-source if string was unterminated (we
// emit anyway so the parser sees something coherent).
// --------------------------------------------------------------
fn lex_string(src_start: i32, src_len: i32, pos: i32) -> i32 {
    let body_start = pos + 1;
    let mut p: i32 = body_start;
    let end = src_start + src_len;
    let mut keep: i32 = 1;
    while keep == 1 {
        if p >= end {
            keep = 0;
        } else {
            let b = __arena_get(p);
            if b == 34 {
                keep = 0;
            } else {
                p = p + 1;
            };
        };
    }
    let body_len = p - body_start;
    // Audit fix #13b (cycle 1, polish): when the closing quote is
    // missing (p >= end before seeing '"'), emit TK_ERR (tag 19)
    // instead of TK_STRLIT (tag 25). The parser then treats the
    // unterminated string as a parse error rather than silently
    // consuming the rest of the file as a string body.
    let tk = if p < end { 25 } else { 19 };
    push_token(tk, body_start, body_start, body_len);
    if p < end { p + 1 } else { p }
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
    else { if b == 126 { 23 }     // '~' (bitwise NOT — see parse_unary)
    else { if b == 64 { 24 }      // '@' (used by @pure / @effect attrs;
                                  // parser skips them as no-ops)
    else { if b == 38 { 27 }      // '&' (TK_AMP — binary bitwise AND)
    else { if b == 124 { 28 }     // '|' (TK_PIPE — binary bitwise OR)
    else { if b == 94 { 29 }      // '^' (TK_CARET — binary bitwise XOR)
    else { 0 }}}}}}}}}}}}}}}}}}}}}
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
        } else { if b == 34 {
            // '"' — string literal.
            pos = lex_string(src_start, src_len, pos);
        } else { if b == 60 {
            // '<' — could be `<<` (TK_LSHIFT=30) or single `<` (TK_LT=16).
            // `<=` is still emitted as two separate TK_LT TK_EQ tokens
            // for the parser to combine (matches the existing `==`/`>=`
            // approach so we don't churn parse_compare).
            if pos + 1 < end {
                let nxt = __arena_get(pos + 1);
                if nxt == 60 {
                    push_token(30, 0, pos, 2);
                    pos = pos + 2;
                } else {
                    push_token(16, 0, pos, 1);
                    pos = pos + 1;
                };
            } else {
                push_token(16, 0, pos, 1);
                pos = pos + 1;
            };
        } else { if b == 62 {
            // '>' — could be `>>` (TK_RSHIFT=31) or single `>` (TK_GT=17).
            if pos + 1 < end {
                let nxt = __arena_get(pos + 1);
                if nxt == 62 {
                    push_token(31, 0, pos, 2);
                    pos = pos + 2;
                } else {
                    push_token(17, 0, pos, 1);
                    pos = pos + 1;
                };
            } else {
                push_token(17, 0, pos, 1);
                pos = pos + 1;
            };
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
        }}}}}}};
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
