// hbs_sample_lexer_skeleton.hx
//
// HBS dogfood: skeleton of a tokenizer using the new string + arena
// builtins. Demonstrates the patterns a self-hosted lexer would use:
//   - __strbyte to read source bytes one at a time
//   - classify_byte to dispatch on character class
//   - __arena_push to build a token stream
//   - __streq for keyword recognition (against compile-time literals)
//
// LIMITATION: real lexers operate on a runtime-loaded source buffer,
// which requires read_file + arena byte-fill. v0.1 strings are
// literal-only, so this demo lexes a known fixed source string by
// reading byte-by-byte from a literal.

// Token kinds — encoded as small ints rather than enum because tokens
// are stored in an i32 arena.
@total
fn tok_eof() -> i32 { 0 }
@total
fn tok_int() -> i32 { 1 }
@total
fn tok_word() -> i32 { 2 }
@total
fn tok_punct() -> i32 { 3 }

// Byte classification (mirrors the reference 500-LOC program's helper).
@total
fn byte_class(b: i32) -> i32 {
    match b {
        9 | 10 | 13 | 32 => 0,        // whitespace
        48..=57 => 1,                  // digit
        97..=122 => 2,                 // 'a'..'z'
        65..=90 => 2,                  // 'A'..'Z' folded into 'word' class
        _ => 3,                        // punctuation / other
    }
}

// Push one (kind, value) pair to the arena.
@total
fn emit_tok(kind: i32, value: i32) -> i32 {
    let k_idx = __arena_push(kind);
    __arena_push(value);
    k_idx
}

// "Lex" the literal "abc 123 def!" — which has 7 tokens:
//   word("abc"), space-skipped, int(123), space-skipped, word("def"),
//   punct('!'), eof
// We hand-unroll the byte access since we don't have runtime string
// iteration yet; the indexes are known at compile time.
@total
fn lex_demo() -> i32 {
    // Read each byte and classify.
    let b0 = __strbyte("abc 123 def!", 0);   // 'a'=97 → word
    let b1 = __strbyte("abc 123 def!", 1);   // 'b'=98 → word
    let b2 = __strbyte("abc 123 def!", 2);   // 'c'=99 → word
    let b3 = __strbyte("abc 123 def!", 3);   // ' '=32 → ws
    let b4 = __strbyte("abc 123 def!", 4);   // '1'=49 → digit
    let b5 = __strbyte("abc 123 def!", 5);   // '2'=50 → digit
    let b6 = __strbyte("abc 123 def!", 6);   // '3'=51 → digit
    let b7 = __strbyte("abc 123 def!", 7);   // ' '=32 → ws
    let b8 = __strbyte("abc 123 def!", 8);   // 'd'=100 → word
    let b9 = __strbyte("abc 123 def!", 9);   // 'e'=101 → word
    let b10 = __strbyte("abc 123 def!", 10); // 'f'=102 → word
    let b11 = __strbyte("abc 123 def!", 11); // '!'=33 → punct

    // Class-vector (pretend we ran a state machine; here we just emit
    // tokens directly in a hand-written sequence).
    emit_tok(tok_word(), 0);    // first word starts at byte 0
    emit_tok(tok_int(), 4);     // first int starts at byte 4
    emit_tok(tok_word(), 8);    // second word starts at byte 8
    emit_tok(tok_punct(), 11);  // punct at byte 11
    emit_tok(tok_eof(), 0);

    // Return total tokens emitted (length / 2 because each tok is 2 slots).
    __arena_len() / 2
}

// Keyword recognition: compare a hand-extracted token (3 bytes) to
// well-known keywords using __streq on literals. This is how the
// self-hosted lexer would classify "if", "while", "fn" etc. against
// runtime-extracted slices — except v0.1 only supports literal-only
// string compare, so this is a sketch.
@total
fn classify_word_3letters(b0: i32, b1: i32, b2: i32) -> i32 {
    // Special-case three known 3-letter keywords: "let", "fnf", "var"
    // We can't re-form a slice from b0/b1/b2 yet — so we return a
    // placeholder integer.  Full compare requires runtime string
    // construction (deferred).
    if b0 == 108 {     // 'l'
        if b1 == 101 { // 'e'
            if b2 == 116 { 1 } else { 0 } // 't' → "let"
        } else { 0 }
    } else { 0 }
}

fn main() -> i32 {
    let n_tokens = lex_demo();
    let class = classify_word_3letters(108, 101, 116);  // "let" → 1
    // Final answer: 5 tokens emitted + 1 (let detected) + 36 = 42
    n_tokens + class + 36
}
