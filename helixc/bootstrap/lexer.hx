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

// K1.AP (2026-05-25): Skip a `/* ... */` block comment starting
// at byte index `pos`. Assumes caller has already verified
// bytes [pos..pos+2) == "/*". Supports nested block comments
// (each `/*` increments depth; `*/` decrements; returns when
// depth reaches 0). Hits EOF safety if the closer is missing
// in malformed input -- returns end-of-source which the caller
// observes as the lex loop terminating.
fn skip_block_comment(src_start: i32, src_len: i32, pos: i32) -> i32 {
    let mut p: i32 = pos + 2;
    let end = src_start + src_len;
    let mut depth: i32 = 1;
    while depth > 0 {
        if p >= end {
            depth = 0;       // EOF safety -- bail out
        } else { if p + 1 >= end {
            // Only 1 byte left; can't form `/*` or `*/`. Skip it.
            p = p + 1;
        } else {
            let b0 = __arena_get(p);
            let b1 = __arena_get(p + 1);
            if b0 == 47 {           // '/'
                if b1 == 42 {       // '/*' nested opener
                    depth = depth + 1;
                    p = p + 2;
                } else {
                    p = p + 1;
                };
            } else { if b0 == 42 {  // '*'
                if b1 == 47 {       // '*/' closer
                    depth = depth - 1;
                    p = p + 2;
                } else {
                    p = p + 1;
                };
            } else {
                p = p + 1;
            }};
        }};
    }
    p
}

// Stage 1.5 audit fix helpers: u64 lex overflow detection for 10-digit
// values. Returns the i'th byte of "4294967295" (2^32-1, the max valid
// 10-digit u64 literal via the two's-complement bit-trick).
fn ref_byte_4294967295(i: i32) -> i32 {
    if i == 0 { 52 }       // '4'
    else { if i == 1 { 50 }  // '2'
    else { if i == 2 { 57 }  // '9'
    else { if i == 3 { 52 }  // '4'
    else { if i == 4 { 57 }  // '9'
    else { if i == 5 { 54 }  // '6'
    else { if i == 6 { 55 }  // '7'
    else { if i == 7 { 50 }  // '2'
    else { if i == 8 { 57 }  // '9'
    else { if i == 9 { 53 }  // '5'
    else { 0 } } } } } } } } } }
}

// Returns 1 if the 10-digit literal at p1..p1+10 represents a u64
// value > 4294967295 (= 2^32-1). Used by the u64 oversize guard to
// catch the [4294967296, 9999999999] range that the partial fix in
// 471b27f missed (those values are 10-digit but multi-wrap the i32
// accumulator). Caller must verify literal is exactly 10 digits.
fn check_u64_10digit_overflow(p1: i32) -> i32 {
    let mut i: i32 = 0;
    let mut keep: i32 = 1;
    let mut result: i32 = 0;
    while keep == 1 {
        if i >= 10 { keep = 0; }
        else {
            let lit_b = __arena_get(p1 + i);
            let ref_b = ref_byte_4294967295(i);
            if lit_b > ref_b { result = 1; keep = 0; }
            else { if lit_b < ref_b { keep = 0; }
            else { i = i + 1; } };
        };
    }
    result
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
    // K1.AQ (2026-05-25): also detect `0b` (binary) and `0o` (octal).
    // Underscores `_` are accepted as no-op separators in any base
    // (matches Rust's 1_000_000 / 0b1010_1010 conventions).
    let mut is_hex: i32 = 0;
    let mut is_bin: i32 = 0;
    let mut is_oct: i32 = 0;
    if p + 1 < end {
        let c0 = __arena_get(p);
        let c1 = __arena_get(p + 1);
        if c0 == 48 {
            // 'x'=120, 'X'=88, 'b'=98, 'B'=66, 'o'=111, 'O'=79
            if c1 == 120 { is_hex = 1; }
            else { if c1 == 88 { is_hex = 1; }
            else { if c1 == 98 { is_bin = 1; }
            else { if c1 == 66 { is_bin = 1; }
            else { if c1 == 111 { is_oct = 1; }
            else { if c1 == 79 { is_oct = 1; } } } } } };
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
                } else { if b == 95 {           // K1.AQ: '_' separator
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
                }};
            };
        }
    } else { if is_bin == 1 {
        // K1.AQ: 0b binary literal. Digits 0 or 1; underscores skipped.
        p = p + 2;     // consume `0b`
        let mut keep_b: i32 = 1;
        while keep_b == 1 {
            if p >= end {
                keep_b = 0;
            } else {
                let b = __arena_get(p);
                if b == 48 {                    // '0'
                    value = value * 2;
                    p = p + 1;
                } else { if b == 49 {           // '1'
                    value = value * 2 + 1;
                    p = p + 1;
                } else { if b == 95 {           // '_' separator
                    p = p + 1;
                } else { keep_b = 0; }}};
            };
        }
    } else { if is_oct == 1 {
        // K1.AQ: 0o octal literal. Digits 0..7; underscores skipped.
        p = p + 2;     // consume `0o`
        let mut keep_o: i32 = 1;
        while keep_o == 1 {
            if p >= end {
                keep_o = 0;
            } else {
                let b = __arena_get(p);
                if b >= 48 {                    // '0'..
                    if b <= 55 {                // ..'7'
                        value = value * 8 + (b - 48);
                        p = p + 1;
                    } else { if b == 95 {       // '_' separator
                        p = p + 1;
                    } else { keep_o = 0; }};
                } else { keep_o = 0; };
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
                } else { if b == 95 {           // '_' separator
                    // K1.E1d-fix (2026-05-26): only treat '_' as a
                    // digit-separator when the NEXT byte is also a
                    // decimal digit. If it's anything else (e.g. the
                    // start of an `_i64` / `_u32` / `_f32` / `_bf16`
                    // type suffix), STOP the digit loop here so the
                    // suffix-detection cascade below sees `_` as its
                    // first byte. K1.AQ originally introduced the
                    // `_`-as-separator semantic but unconditionally
                    // consumed it -- which silently swallowed the
                    // leading `_` of every typed-suffix literal,
                    // so the lexer tagged `42_i64` as plain TK_INTLIT
                    // (tag 1) instead of TK_INTLIT_I64 (tag 33).
                    // Downstream that meant the parser produced
                    // AST_INT (tag 0) instead of AST_INTLIT_I64
                    // (tag 35), and codegen emitted 5-byte
                    // `mov eax, imm32` instead of 10-byte
                    // `movabs rax, imm64` -- which the width-class
                    // trap then correctly flagged as a width
                    // mismatch, raising SIGILL for any `fn main() ->
                    // i64 { ... }`-shape program. The 1-byte
                    // lookahead here restores the separator
                    // semantic for `1_000_000` while keeping
                    // `42_i64`'s suffix visible.
                    if p + 1 < end {
                        let nxt = __arena_get(p + 1);
                        if is_digit(nxt) == 1 {
                            p = p + 1;
                        } else {
                            keep = 0;
                        };
                    } else {
                        keep = 0;
                    };
                } else {
                    keep = 0;
                }};
            };
        }
    }}}
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
    let mut is_f16_suffix: i32 = 0;
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
    // K1.BH (2026-05-26): _f16 (4 bytes: '_' 'f' '1' '6'). IEEE 754
    // half-precision. K1.F15 (2026-05-27) splits the lex path from
    // bf16 so codegen can emit the true f16 bit pattern (1+5+10)
    // instead of the bf16-shaped truncation (1+8+7). Sets a separate
    // is_f16_suffix flag and emits token tag 44 (TK_FLOATLIT_F16)
    // -- distinct from tag 41 (TK_FLOATLIT_BF16). Parser routes
    // tag 44 to AST_FLOATLIT_F16 (tag 80); codegen at t==80 uses the
    // new f32_to_f16_bits helper (see kovc.hx).
    if p + 3 < end {
        let g0 = __arena_get(p);
        if g0 == 95 {                              // '_'
            let g1 = __arena_get(p + 1);
            if g1 == 102 {                         // 'f'
                let g2 = __arena_get(p + 2);
                if g2 == 49 {                      // '1'
                    let g3 = __arena_get(p + 3);
                    if g3 == 54 {                  // '6'
                        p = p + 4;
                        is_f16_suffix = 1;
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
        // K1.F15 (2026-05-27): _f16 suffix → token tag 44 (TK_FLOATLIT_
        // F16). Parser routes to AST_FLOATLIT_F16 (tag 80); codegen
        // calls f32_to_f16_bits to encode IEEE-754 half-precision.
        let tk = if is_f16_suffix == 1 { 44 }
                 else { if is_bf16_suffix == 1 { 41 }
                 else { if is_f64_suffix == 1 { 32 } else { 26 } } };
        push_token(tk, pos, pos, flen);
    } else { if is_f16_suffix == 1 {
        // K1.F15 (2026-05-27): _f16 suffix on a non-float number (e.g.,
        // `42_f16`). Emit TK_FLOATLIT_F16; floatlit machinery handles
        // the integer-shape gracefully.
        let flen = p - pos;
        push_token(44, pos, pos, flen);
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
        // Stage 1.5 audit fix (post-bf16 sweep): u64 lex overflow guard.
        // The partial fix in 471b27f caught > 10 digits via tk=40. But
        // 10-digit values in the range [4294967296, 9999999999] also
        // wrap silently in the i32 accumulator (multi-wrap, bit-trick
        // doesn't preserve high half). Catch via digit-by-digit lex
        // compare against "4294967295" (= 2^32-1, max valid 10-digit
        // value via the two's-complement bit-trick + hi32=0 fix from
        // 09c8858). If 10-digit literal lex-greater than "4294967295",
        // set u64_oversize.
        let digit_count = length - 4;
        let u64_oversize = if is_u64_suffix == 1 {
            if digit_count > 10 { 1 }
            else { if digit_count == 10 {
                check_u64_10digit_overflow(pos)
            } else { 0 } }
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
    } } };  // K1.F15: +1 brace for new is_f16_suffix outer arm
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
// K1.K (2026-05-25): lex a char literal. Caller has verified
// bytes[pos] == 39 (`'`). Two shapes:
//   `'X'` -- bytes[pos+1] is the literal byte, bytes[pos+2] == 39.
//            Emit TK_INTLIT (tag 1) with payload = byte value, span
//            3 bytes. Returns pos+3.
//   `'\X'` -- bytes[pos+1] == 92 ('\'), bytes[pos+2] is the escape
//            code. Resolves the standard set: n=10, t=9, r=13, 0=0,
//            '=39, "=34, \=92. Emit TK_INTLIT with payload = the
//            resolved byte, span 4 bytes. Returns pos+4.
//
// Unterminated / unknown shapes emit TK_ERR (tag 19) and advance
// minimally so the parser unwinds cleanly (same convention as
// lex_string's audit-13b fix).
//
// Lexing as TK_INTLIT (rather than introducing a TK_CHARLIT tag)
// is deliberate: char literals are int values in Helix (per the
// Python helixc semantics) so the parser + codegen go through
// the int-literal path unchanged. Matches the K1.F-discovery
// pattern of reusing existing tags rather than minting new ones.
// --------------------------------------------------------------
fn lex_char_lit(src_start: i32, src_len: i32, pos: i32) -> i32 {
    let end = src_start + src_len;
    // Need at least pos+2 in bounds for `'X'` (3 bytes total).
    if pos + 2 >= end {
        push_token(19, 0, pos, 1);
        pos + 1
    } else {
        let b1 = __arena_get(pos + 1);
        if b1 == 92 {
            // Escape form: `'\X'` -- need pos+3 in bounds.
            if pos + 3 >= end {
                push_token(19, 0, pos, 1);
                pos + 1
            } else {
                let esc = __arena_get(pos + 2);
                let close = __arena_get(pos + 3);
                if close != 39 {
                    push_token(19, 0, pos, 1);
                    pos + 1
                } else {
                    // Resolve standard escape codes. Unknown
                    // escapes emit TK_ERR rather than passing the
                    // raw escape byte through (matches Python's
                    // strict behaviour).
                    let val =
                        if esc == 110 { 10 }           // \n
                        else { if esc == 116 { 9 }     // \t
                        else { if esc == 114 { 13 }    // \r
                        else { if esc == 48 { 0 }      // \0
                        else { if esc == 39 { 39 }     // \'
                        else { if esc == 34 { 34 }     // \"
                        else { if esc == 92 { 92 }     // \\
                        else { 0 - 1 } } } } } } };    // sentinel for unknown
                    if val == (0 - 1) {
                        push_token(19, 0, pos, 1);
                        pos + 1
                    } else {
                        push_token(1, val, pos, 4);
                        pos + 4
                    }
                }
            }
        } else {
            // Simple form: `'X'`. The closing apostrophe must be at
            // pos+2 (one byte body).
            let close = __arena_get(pos + 2);
            if close != 39 {
                // K1.CQ (2026-05-26): not a valid char-lit. If b1 is
                // alpha or `_` (a valid IDENT start), treat as a Rust
                // lifetime annotation (`'a`, `'static`, `'_`, etc.)
                // and silently SKIP the leading `'` -- the next lex
                // iteration will pick up `a` / `static` / `_` as a
                // normal IDENT. The parser's existing generic-param
                // skip (K1.T) and where-clause skip (K1.O / K1.CD)
                // treat IDENTs inside `<...>` and `where ...` as
                // type-erased no-ops, so lifetime IDENTs flow through
                // without further handling.
                //
                // The bootstrap is type-erased and doesn't enforce
                // lifetime constraints; this is purely syntactic
                // acceptance so common Rust source (`fn id<'a>(...)`,
                // `impl<'a> ...`, `&'static str`) parses cleanly.
                //
                // For non-alpha b1 (e.g., `'5'` would be a real char-
                // lit caught above; `' ` whitespace or `''` empty are
                // never valid Rust and stay as TK_ERR).
                if is_alpha(b1) == 1 {
                    pos + 1
                } else {
                    push_token(19, 0, pos, 1);
                    pos + 1
                }
            } else {
                push_token(1, b1, pos, 3);
                pos + 3
            }
        }
    }
}

// --------------------------------------------------------------
// Lex a string literal. Caller has verified bytes[pos] == 34 ('"').
// Walks forward to the next UNESCAPED '"', decoding backslash escapes
// IN PLACE as it goes. Emits a TK_STRLIT whose payload = body
// byte_start, src_len = the DECODED body length (both EXCLUDE the
// quotes). Returns byte index AFTER the closing quote, or AFTER
// end-of-source if the string was unterminated (we emit anyway so
// the parser sees something coherent).
//
// Escape decoding mirrors the Python reference lexer
// (helixc/frontend/lexer.py _decode_escape) and the bootstrap's own
// lex_char_lit: the standard single-byte set
//   n=10, t=9, r=13, 0=0, '=39, "=34, \=92
// is resolved; an ESCAPED '"' (`\"`) therefore does NOT terminate
// the string. Any OTHER escape byte is "unknown" — Python raises a
// LexError, so we emit TK_ERR (tag 19), matching its strict
// behaviour (and lex_char_lit's precedent). A trailing backslash
// with no following byte before end-of-source is likewise an error.
//
// In-place decode-compact: a read pointer (rp) and write pointer
// (wp) both start at body_start. The decoded body is never longer
// than the raw body, so wp <= rp at all times — overwriting the
// source bytes in [body_start, wp) is safe (those raw bytes are
// never re-read for this token; downstream reads use src_start +
// src_len = the decoded region). Note (v0.5 fix): __strlen reads
// this src_len (kovc.hx:5436) so the decoded length now matches
// Python's len(s.encode("utf-8")).
// --------------------------------------------------------------
fn lex_string(src_start: i32, src_len: i32, pos: i32) -> i32 {
    let body_start = pos + 1;
    let end = src_start + src_len;
    let mut rp: i32 = body_start;
    let mut wp: i32 = body_start;
    let mut keep: i32 = 1;
    let mut closed: i32 = 0;
    while keep == 1 {
        if rp >= end {
            keep = 0;
        } else {
            let b = __arena_get(rp);
            if b == 34 {
                // Unescaped closing quote — end of string.
                closed = 1;
                keep = 0;
            } else {
                if b == 92 {
                    // Backslash escape. Need rp+1 < end for the
                    // escape byte; a trailing '\' is an error.
                    if rp + 1 >= end {
                        // Trailing '\' with no escape byte: error
                        // (Python raises). Leave closed=0 -> TK_ERR.
                        keep = 0;
                    } else {
                        let esc = __arena_get(rp + 1);
                        let dec =
                            if esc == 110 { 10 }           // \n
                            else { if esc == 116 { 9 }     // \t
                            else { if esc == 114 { 13 }    // \r
                            else { if esc == 48 { 0 }      // \0
                            else { if esc == 39 { 39 }     // \'
                            else { if esc == 34 { 34 }     // \"
                            else { if esc == 92 { 92 }     // \\
                            else { 0 - 1 } } } } } } };     // unknown sentinel
                        if dec == (0 - 1) {
                            // Unknown escape: Python raises LexError.
                            // Leave closed=0 -> TK_ERR (tag 19).
                            keep = 0;
                        } else {
                            __arena_set(wp, dec);
                            wp = wp + 1;
                            rp = rp + 2;
                        };
                    };
                } else {
                    // Ordinary byte — copy through.
                    __arena_set(wp, b);
                    wp = wp + 1;
                    rp = rp + 1;
                };
            };
        };
    }
    let body_len = wp - body_start;
    // Audit fix #13b (cycle 1, polish): when the closing quote is
    // missing (or an escape was malformed), emit TK_ERR (tag 19)
    // instead of TK_STRLIT (tag 25). The parser then treats the
    // unterminated/invalid string as a parse error rather than
    // silently consuming a bogus body.
    let tk = if closed == 1 { 25 } else { 19 };
    push_token(tk, body_start, body_start, body_len);
    // Advance past the closing quote when one was found; otherwise
    // stop at rp (end-of-source or the malformed-escape position).
    if closed == 1 { rp + 1 } else { rp }
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
    else { if b == 46 { 22 }      // '.' (TK_DOT — Stage 4 tuple field access)
    else { if b == 91 { 20 }      // '[' (TK_LBRACK — Stage 4 array literal)
    else { if b == 93 { 21 }      // ']' (TK_RBRACK — Stage 4 array literal)
    else { 0 }}}}}}}}}}}}}}}}}}}}}}}}
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
        } else { if b == 35 {
            // K1.BK (2026-05-26): '#' (byte 35) opens a Rust outer
            // attribute `#[...]` or inner attribute `#![...]`. The
            // bootstrap has no attribute-driven codegen for the Rust
            // attribute family (#[derive(Debug)], #[inline], etc.) --
            // those are syntactic decoration only, semantically a
            // no-op in Phase-0. Skip the entire bracketed block (and
            // the optional `!`) at lex time as if it were a comment;
            // no token is emitted so the parser sees a clean decl.
            // EOF-safe: if `]` never arrives, we stop at end-of-input.
            let mut p_a = pos + 1;
            if p_a < end {
                if __arena_get(p_a) == 33 {                  // '!' (inner attr)
                    p_a = p_a + 1;
                };
            };
            if p_a < end {
                if __arena_get(p_a) == 91 {                  // '['
                    p_a = p_a + 1;
                    let mut depth_a: i32 = 1;
                    while depth_a > 0 {
                        if p_a >= end {
                            depth_a = 0;
                        } else {
                            let ab = __arena_get(p_a);
                            if ab == 91 { depth_a = depth_a + 1; };
                            if ab == 93 { depth_a = depth_a - 1; };
                            p_a = p_a + 1;
                        };
                    };
                    pos = p_a;
                } else {
                    // '#' not followed by '[' is unexpected; emit it as
                    // an unknown-byte token so the parser fails loud
                    // rather than silently mis-parsing. Tag 0 = EOF/
                    // unknown sentinel which downstream parsers treat
                    // as end-of-stream -- same fail-loud contract as
                    // any other unrecognized byte.
                    push_token(0, 0, pos, 1);
                    pos = pos + 1;
                };
            } else {
                push_token(0, 0, pos, 1);
                pos = pos + 1;
            };
        } else { if b == 47 {
            // Possible '//' line comment, '/*' block comment (K1.AP),
            // else slash punctuation.
            if pos + 1 < end {
                let nxt = __arena_get(pos + 1);
                if nxt == 47 {
                    pos = skip_line_comment(src_start, src_len, pos);
                } else { if nxt == 42 {
                    // K1.AP (2026-05-25): '/*' block comment opener.
                    pos = skip_block_comment(src_start, src_len, pos);
                } else {
                    push_token(10, 0, pos, 1);
                    pos = pos + 1;
                }};
            } else {
                push_token(10, 0, pos, 1);
                pos = pos + 1;
            };
        } else { if is_digit(b) == 1 {
            pos = lex_int(src_start, src_len, pos);
        } else { if is_alpha(b) == 1 {
            // K1.CK (2026-05-26): prefixed string literals
            // `b"..."` (byte string), `r"..."` (raw string), and
            // `c"..."` (C string, Rust 2021+). The prefix is a
            // single alphabetic byte that is IMMEDIATELY followed
            // by `"` (no space between). Without this special-
            // casing, `b` / `r` / `c` lex as 1-byte IDENTs and the
            // following `"..."` lexes as a separate STRLIT, which
            // makes the parser trip when it sees IDENT-then-STRLIT
            // in expression position. The bootstrap does not model
            // byte / raw / cstring distinctions; they all decay to
            // TK_STRLIT (tag 25) with the body bytes verbatim.
            //
            // Detection: b == 98 ('b') OR b == 114 ('r') OR
            // b == 99 ('c'); AND the next byte exists; AND the
            // next byte is 34 ('"'). On match, skip the prefix
            // byte by passing pos+1 to lex_string.
            let next_byte_ck = if pos + 1 < end { __arena_get(pos + 1) } else { 0 };
            let is_str_pfx_kw = if next_byte_ck == 34 {
                if b == 98 { 1 }
                else { if b == 114 { 1 }
                else { if b == 99 { 1 } else { 0 } } }
            } else { 0 };
            if is_str_pfx_kw == 1 {
                pos = lex_string(src_start, src_len, pos + 1);
            } else {
                pos = lex_ident(src_start, src_len, pos);
            };
        } else { if b == 34 {
            // '"' — string literal.
            pos = lex_string(src_start, src_len, pos);
        } else { if b == 39 {
            // K1.K (2026-05-25): `'` — char literal. Lexes to
            // TK_INTLIT with the byte value as payload (no new tag
            // needed; chars are int values in Helix). Handles the
            // standard escape set (\n \t \r \0 \' \" \\).
            pos = lex_char_lit(src_start, src_len, pos);
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
        } else { if b == 61 {
            // Stage 7: '=' — could be `=>` (TK_FATARROW=42) or single
            // `=` (TK_EQ=15). `==` is still emitted as two TK_EQ tokens
            // for the parser to combine into AST_EQ.
            if pos + 1 < end {
                let nxt = __arena_get(pos + 1);
                if nxt == 62 {
                    push_token(42, 0, pos, 2);
                    pos = pos + 2;
                } else {
                    push_token(15, 0, pos, 1);
                    pos = pos + 1;
                };
            } else {
                push_token(15, 0, pos, 1);
                pos = pos + 1;
            };
        } else { if b == 46 {
            // Stage 7: '.' — could be `..` (TK_DOTDOT=43) or single
            // `.` (TK_DOT=22, used by tuple field access since Stage 4).
            if pos + 1 < end {
                let nxt = __arena_get(pos + 1);
                if nxt == 46 {
                    push_token(43, 0, pos, 2);
                    pos = pos + 2;
                } else {
                    push_token(22, 0, pos, 1);
                    pos = pos + 1;
                };
            } else {
                push_token(22, 0, pos, 1);
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
        }}}}}}}}}}};     // K1.BK (2026-05-26): +1 close for the new b==35 (`#`) attribute-skip arm
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
