# Front end: lexer, parser, typecheck

*What this chapter covers:* how `kovc`'s front end turns Helix source text into an AST — the
arena-based lexer ([`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx)), the
recursive-descent parser ([`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx)),
the compile-time `file:line:col: parse error` diagnostic
([`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx)), and how the front end "tracks
types" (type *tags*, not a separate type-checking pass). This is a **compiler-internals** chapter:
unlike the Helix programs quoted elsewhere in the book, every code block here is a **Fragment** of
`kovc`'s own source — quoted verbatim with a line-region citation, not a runnable user program.

If you have not yet read it, [Part I — The trust story at a glance](../part1-orientation/04-trust-at-a-glance.md)
explains why this code is itself a trust artifact: `kovc` is the compiler *written in Helix* that
reproduces itself byte-for-byte (the self-host fixpoint, pinned `0992dddd…`). The front end you are
about to read is part of that fixpoint — it compiles itself. That fact drives several design
choices below, so the AI callouts flag where the front end feeds the self-host fixpoint.

> **For AI agents:** the three front-end source files are
> [`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx),
> [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx), and the driver glue in
> [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx). When you reason about token or
> AST tags, key off the numeric tables quoted in this chapter (re-derived from those files), not
> from English paraphrase. If a tag number here ever disagrees with the source, the source wins.

---

## The shape of the pipeline

`kovc` is a single-pass-ish recursive-descent compiler with no separate IR module between the
parser and codegen: the lexer fills an arena with tokens, the parser walks those tokens and builds
an AST in the same arena, and the back end (covered in
[the x86-64 ELF back end](03-x86-backend.md)) walks the AST to emit machine code. Everything lives
in one growable arena, addressed by integer slot indices — there are no heap pointers, because the
language the bootstrap compiles is a deliberately small subset that has no allocator of its own.

Three properties of that arena design shape the whole front end:

1. **Tokens and AST nodes are fixed-width 4-slot records.** A token is `[tag, payload, src_start,
   src_len]`; an AST node is `[tag, p1, p2, p3]` (some node kinds use extra trailing slots). A
   "node index" is just the arena slot of the record's tag.
2. **Source bytes live in the same arena.** The lexer reads the file into an arena region and then
   walks those bytes in place; tokens carry *absolute* arena indices back into that byte region.
   This is why the parse-error diagnostic can recover a `line:col` later — the byte offset is
   preserved end to end.
3. **State is passed as a base index, not a struct.** The SysV ABI on x86-64 limits `kovc`'s
   codegen to six integer parameters, so the parser stashes *all* of its mutable state in a
   contiguous arena region and threads only `(tok_base, state_base)` through every function. The
   header of [`parser.hx`](../../../helixc/bootstrap/parser.hx) documents this directly:

**Fragment** — [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx) lines 109–127
(the state-region rationale and its first slots):

```helix
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
```

That region grows well past slot 12 — the parser now uses slots through ~129 for struct/enum/trait
tables, closure state, generic-parameter scratch, and autodiff buckets (all set up in `parse_top`,
below). The principle never changes: one base index in, every accessor is a tiny
`__arena_get(sb + N)` getter.

---

## The lexer

### Token records and the tag table

The lexer's contract is stated at the very top of the file: read a file into an arena, walk the
bytes, and emit a stream of 4-slot token records.

**Fragment** — [`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx) lines 1–9 (the
token-record layout):

```helix
// Stage-0 lexer for the Helix bootstrap compiler.
//
// Reads a source file via read_file_to_arena, walks the resulting
// byte sequence, and emits a stream of tokens into a separate arena
// region. Each token occupies 4 i32 slots:
//
//   [tag, payload, src_start, src_len]
```

The token *kinds* are a small integer enum that has grown one feature at a time. A representative
slice of that table, quoted verbatim:

**Fragment** — [`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx) lines 11–33 (token
kinds; the enum continues with multi-byte operators emitted later in `lex`):

```helix
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
```

Two design notes from this table matter for the rest of the pipeline:

- **Keywords are not lexed as keywords.** `let`, `fn`, `if`, `while`, `struct`, … all lex as
  `TK_IDENT` (tag 2). Classification into keywords happens later, in the parser, by comparing the
  identifier's bytes against the installed keyword strings. The header says so explicitly: "Keywords
  like `let`, `fn`, `if` are emitted as `TK_IDENT` here; a post-lex keyword pass (or the parser)
  does the final classification" ([`lexer.hx`](../../../helixc/bootstrap/lexer.hx) lines 39–41).
- **Char literals reuse `TK_INT`.** A `'X'` literal lexes to `TK_INTLIT` (tag 1) with the byte value
  as payload — "chars are int values in Helix … so the parser + codegen go through the int-literal
  path unchanged" ([`lexer.hx`](../../../helixc/bootstrap/lexer.hx) lines 616–621). Reusing existing
  tags rather than minting new ones is a recurring theme; it keeps the parser and codegen surface
  small, which keeps the self-host fixpoint stable.

> **For AI agents:** do not assume there is a keyword token. There is not. `fn`/`let`/`if`/… are
> `TK_IDENT` (tag 2); the parser decides keyword-ness by byte comparison against the strings set up
> in `install_keywords` (see [`parser.hx`](../../../helixc/bootstrap/parser.hx) line 9174). The
> false-positive guard you would otherwise need ("is this identifier actually the keyword?") is
> done in `parse_top` / `parse_primary`, not in the lexer.

### Byte classification and whitespace

The lexer works on raw bytes (the result of `__arena_get`), not characters. Whitespace handling is
exactly what the header promises — space, tab, newline, CR, plus `//` line comments and `/* … */`
block comments, all skipped silently:

**Fragment** — [`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx) lines 56–62
(`is_whitespace`):

```helix
@pure
fn is_whitespace(b: i32) -> i32 {
    if b == 32 { 1 }
    else { if b == 9 { 1 }
    else { if b == 10 { 1 }
    else { if b == 13 { 1 }
    else { 0 }}}}
}
```

`is_alpha` carries a comment that is worth reading as a cautionary tale, because it documents a real
bug the self-host process surfaced. The underscore byte (95) sorts *between* `A` (65) and `z` (122),
so a naive range check silently broke every identifier containing `_`:

**Fragment** — [`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx) lines 70–85
(`is_alpha`, with the underscore-ordering note):

```helix
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
```

This is characteristic of the whole front end: a bug like "underscored identifiers truncate" does
not show up on small tests — it shows up when the compiler tries to compile *itself*, where
identifiers like `sum_to`, `tok_base`, and `src_start` are everywhere. The self-host fixpoint is the
test.

### The arena scan: `push_token` and the main loop

Every token is appended with one helper, whose return value is the slot index of the new record —
that index *is* the token's address:

**Fragment** — [`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx) lines 97–103
(`push_token`):

```helix
fn push_token(tag: i32, payload: i32, src_start: i32, src_len: i32) -> i32 {
    let i = __arena_push(tag);
    __arena_push(payload);
    __arena_push(src_start);
    __arena_push(src_len);
    i
}
```

The main loop, `lex`, records the arena length at entry (`token_base`), walks `[src_start,
src_start+src_len)`, dispatches on the leading byte, and returns the token count. Its opening and
its dispatch on whitespace, `#` attributes, and `/`-comments:

**Fragment** — [`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx) lines 837–845 and
890–909 (the loop entry and two dispatch arms; the full arm cascade runs to line 1024):

```helix
fn lex(src_start: i32, src_len: i32) -> i32 {
    let token_base = __arena_len();
    let mut pos: i32 = src_start;
    let end = src_start + src_len;
    while pos < end {
        let b = __arena_get(pos);
        if is_whitespace(b) == 1 {
            pos = pos + 1;
        } else { if b == 35 {
```

```helix
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
```

After the loop, `lex` pushes the EOF sentinel and returns the token count, computed by dividing the
arena delta by the 4-slot record width:

**Fragment** — [`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx) lines 1026–1028
(EOF sentinel + token-count return):

```helix
    push_token(0, 0, pos, 0);   // TK_EOF sentinel
    let after = __arena_len();
    (after - token_base) / 4
```

Multi-character operators are disambiguated with one byte of lookahead inside `lex`: `<` becomes
`<<` (tag 30) or stays `<` (tag 16); `>` becomes `>>` (tag 31) or `>`; `=` becomes `=>` (tag 42) or
`=`; `.` becomes `..` (tag 43) or `.` (tag 22). Notably, the *two-character comparisons* `<=`, `>=`,
`==`, `!=` are **not** combined in the lexer — they are emitted as two single tokens and recombined
by the parser. The lexer comment is explicit about why ("`<=` is still emitted as two separate
TK_LT TK_EQ tokens for the parser to combine … so we don't churn parse_compare",
[`lexer.hx`](../../../helixc/bootstrap/lexer.hx) lines 949–951). The single-char punctuation map
itself is a flat `punct_kind` table ([`lexer.hx`](../../../helixc/bootstrap/lexer.hx) lines
803–831).

### Number literals carry the source text

`lex_int` is the largest lexer helper, and the reason is the literal-typing system. It recognises
`0x`/`0b`/`0o` prefixes and `_` digit separators, then an optional width/sign suffix (`_i8 … _u64`,
`_f32 _f64 _bf16 _f16`), and emits a *different token tag per suffix* — e.g. tag 33 for `_i64`, tag
36 for `_u64`, tag 26 for an `f32` float, tag 32 for `_f64`. Crucially, for wide literals the token
also carries a *reference to the literal's source text* (`src_start`/`src_len`), because the lexer's
accumulator is only an `i32`:

**Fragment** — [`helixc/bootstrap/lexer.hx`](../../../helixc/bootstrap/lexer.hx) lines 564–571 (the
suffix→tag selection at the end of `lex_int`):

```helix
        let tk = if is_i64_suffix == 1 { 33 }
                 else { if is_u32_suffix == 1 { 34 }
                 else { if is_u8_suffix == 1 { 35 }
                 else { if is_u64_suffix == 1 { 36 }
                 else { if is_i8_suffix == 1 { 37 }
                 else { if is_i16_suffix == 1 { 38 }
                 else { if is_u16_suffix == 1 { 39 } else { 1 } } } } } } };
        push_token(tk, value, pos, length);
```

Two things here feed the self-host fixpoint directly. First, the hex-prefix handling exists because
`kovc.hx` itself writes hex literals like `0x401000` in `emit_elf_header`; the comment warns that
losing hex support would split that into `TK_INT(0)` followed by an `x401000` identifier and produce
a `K2` whose emitted ELF entry-point is a stack pointer ([`lexer.hx`](../../../helixc/bootstrap/lexer.hx)
lines 185–192). Second, the one-byte lookahead that decides whether `_` is a digit separator or the
start of a type suffix exists because `42_i64` must stay tag-33, not collapse to plain `TK_INT` —
the comment traces the full failure chain from a mis-tagged literal down to a `SIGILL`
([`lexer.hx`](../../../helixc/bootstrap/lexer.hx) lines 297–319). The lexer's correctness is not
abstract; each of these notes is a fixpoint regression that was caught and fixed.

> **For AI agents:** a numeric literal's *width* is carried by its **token tag**, not just its
> payload. `42` is tag 1; `42_i64` is tag 33; `5_000_000_000_u64` is tag 36 and its `i32` payload is
> vestigial — codegen re-decodes the value from the literal's source text via a 16-bit-limb path
> (see [Types: widths, structs, and enums](../part3-language/02-types.md) and the spec's §1 on wide
> literals, [`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md)). Do not
> assume the token payload holds the full value for wide literals.

### Error and skip tokens

Unknown bytes do not crash the lexer — they emit `TK_ERR` (tag 19) so the parser can report a clean
diagnostic. An unterminated string, a malformed escape, or a stray byte all surface as tag 19
([`lexer.hx`](../../../helixc/bootstrap/lexer.hx) lines 1015–1019 for the unknown-byte arm;
789–795 for the unterminated-string arm). Rust attribute blocks `#[...]` / `#![...]` are skipped
entirely at lex time as decoration, because the bootstrap has no attribute-driven codegen for the
Rust attribute family ([`lexer.hx`](../../../helixc/bootstrap/lexer.hx) lines 845–889). Keeping
those out of the token stream means the parser never has to special-case them.

---

## The parser

### AST nodes and the tag legend

The parser consumes the token stream and builds an AST in the arena. Each AST node is a 4-slot
record `[tag, p1, p2, p3]`, and the canonical legend lives at the top of
[`parser.hx`](../../../helixc/bootstrap/parser.hx). A representative slice:

**Fragment** — [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx) lines 4–32 (AST
node layout + the first tags; the legend continues through tag 99):

```helix
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
```

Several encodings are worth calling out from the legend because they recur in the back end:

- **Packed payloads.** `AST_LET` packs `value_idx * 65536 + body_idx` into a single slot (`p3`).
  This is a space optimization that works only because node indices stay small; the same packing
  trick is reused by `AST_FN_DECL`.
- **`AST_FN_DECL` (tag 14) is wide.** Its base `[tag, name_s, name_l, body_idx]` is followed by
  extra trailing slots — `slot 4: params_head`, `slot 5: ret_ty`, `slot 6: is_generic`, … up through
  attribute flags at slots 8–19 ([`parser.hx`](../../../helixc/bootstrap/parser.hx) lines 37–53). A
  top-level program is a linked list of these, wrapped in `AST_FN_LIST` (tag 15) nodes.
- **`AST_ERR` (tag 99) is the parse-error node.** `p1` holds the unexpected token's tag; as we will
  see, `p2` is repurposed to hold the *source byte offset* so the driver can print a `line:col`.

The grammar the parser implements is classic recursive-descent precedence climbing. The header
states the core of it:

**Fragment** — [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx) lines 100–107
(the documented grammar core; the real parser adds bitwise/shift levels and many statement forms):

```helix
// Grammar (recursive descent, classic precedence climbing):
//   expr     := add ("<" add)?
//   add      := mul (("+" | "-") mul)*
//   mul      := unary (("*" | "/") unary)*
//   unary    := "-" unary | primary
//   primary  := INT | IDENT | "(" expr ")" | if-expr | let-expr
//   if-expr  := "if" expr "{" expr "}" "else" "{" expr "}"
//   let-expr := "let" IDENT "=" expr ";" expr
```

### The token-stream and state accessors

The parser never indexes tokens by hand. Four `@pure` getters read a token's fields by index `k`,
and a family of one-line getters read parser-state slots. The token getters:

**Fragment** — [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx) lines 133–141
(token accessors + the cursor helpers):

```helix
@pure fn tok_tag(tok_base: i32, k: i32) -> i32 { __arena_get(tok_base + k * 4) }
@pure fn tok_p1(tok_base: i32, k: i32) -> i32  { __arena_get(tok_base + k * 4 + 1) }
@pure fn tok_p2(tok_base: i32, k: i32) -> i32  { __arena_get(tok_base + k * 4 + 2) }
@pure fn tok_p3(tok_base: i32, k: i32) -> i32  { __arena_get(tok_base + k * 4 + 3) }

fn cur_get(sb: i32) -> i32 { __arena_get(sb) }
fn cur_set(sb: i32, v: i32) -> i32 { __arena_set(sb, v); 0 }
fn cur_advance(sb: i32) -> i32 { let c = cur_get(sb); cur_set(sb, c + 1); 0 }
```

`tok_p2(tok_base, k)` returns a token's `src_start` — its absolute byte offset in the source. Hold
on to that; it is the single value that makes the parse-error diagnostic possible.

### `parse_top`: set up state, then dispatch

`parse_top(tok_base)` is the parser's entry point. It does two jobs: build the giant state region
(pushing ~130 slots and several inline tables), then peek the first token to decide whether the
input is a *program* (one or more declarations) or a bare *expression* (legacy single-expression
mode, kept for the early tests). It begins by reserving the cursor and keyword slots:

**Fragment** — [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx) lines 9290–9298
(the start of `parse_top` — reserving the state region):

```helix
fn parse_top(tok_base: i32) -> i32 {
    // Parser state slots: cursor + keyword pairs + staged scratch/table
    // regions. Later comments below document each added range through slot 87.
    let cur_slot = __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0); __arena_push(0);
    __arena_push(0); __arena_push(0); __arena_push(0);
```

After all the tables are allocated and `install_keywords` has populated the keyword strings,
`parse_top` skips leading attributes and `pub`/`extern` modifiers, then classifies the first
identifier by byte comparison. The dispatch is a flat keyword cascade — and every keyword arm routes
to the same `parse_program`, with a bare expression as the fallthrough:

**Fragment** — [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx) lines 9555–9560
and 9579–9608 (keyword detection by byte-equality, then the dispatch):

```helix
    let k = cur_get(cur_slot);
    if tok_tag(tok_base, k) == 2 {
        let id_s = tok_p2(tok_base, k);
        let id_l = tok_p3(tok_base, k);
        let is_fn = byte_eq(id_s, id_l, kw_fn_s(cur_slot), kw_fn_n(cur_slot));
        let is_struct = byte_eq(id_s, id_l, kw_struct_s(cur_slot), kw_struct_n(cur_slot));
```

```helix
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
        } else { if is_const == 1 {
            parse_program(tok_base, cur_slot)
        } else { if is_static == 1 {
            parse_program(tok_base, cur_slot)
        } else { if is_union == 1 {
            parse_program(tok_base, cur_slot)
        } else { if is_agent == 1 {
            parse_program(tok_base, cur_slot)
        } else {
            parse_expr(tok_base, cur_slot)
        }}}}}}}}}}}}
    } else {
        parse_expr(tok_base, cur_slot)
    }
```

`parse_program` (at [`parser.hx`](../../../helixc/bootstrap/parser.hx) line 9928) loops over the
top-level declarations, calling the appropriate `parse_fn_decl` / `parse_struct_decl` /
`parse_enum_decl` / … per item and chaining the results into the `AST_FN_LIST`. The function
inventory is large — `parse_primary` alone spans roughly lines 3301–9013 — because it absorbs the
entire surface syntax of `if`/`while`/`match`/`let`/calls/closures/struct-literals/indexing into one
recursive-descent core.

### Expression parsing and precedence

Binary-operator precedence is encoded as a *chain of functions*, tightest-binding at the bottom.
`parse_expr` handles `;`-sequencing and the implicit-statement chaining for block-shaped
expressions; `parse_expr_basic` handles comparisons and the short-circuit `&&`/`||` desugar; below
it sit the bitwise/shift levels and then arithmetic. The precedence split is documented at the
bitwise layer:

**Fragment** — [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx) lines 2437–2442
(the bitwise/shift precedence hierarchy, and why `parse_bitwise` stays the top entry):

```helix
// hierarchy (tightest -> loosest): `<<`/`>>` > `&` > `^` > `|`, all between
// additive and comparison. parse_bitwise is KEPT as the top (`|`) entry so
// every existing caller (parse_expr_basic lhs + comparison RHS) is unaffected
// and automatically gets the full corrected chain. The `&&`/`||` bail-outs
// (handled above comparison in parse_expr_basic) are preserved at their
// respective single-char levels.
```

Two parser behaviours from this region are easy to get wrong if you only read the grammar header:

- **`<= >= == !=` are reassembled here, not in the lexer.** `parse_expr_basic` peeks the next two
  token tags and, when it sees `<` followed by `=`, consumes both and folds an `AST_LE` (tag 22);
  similarly `>=`→23, `==`→20, `!=`→21 ([`parser.hx`](../../../helixc/bootstrap/parser.hx) lines
  2315–2385). This is the matching half of the lexer's deliberate choice to *not* merge those pairs.
- **`&&`/`||` are desugared to nested `if`, not real operators.** `parse_expr_basic` rewrites
  `a && b` into `AST_IF(a, b, 0)` and `a || b` into `AST_IF(a, 1, b)`
  ([`parser.hx`](../../../helixc/bootstrap/parser.hx) lines 2399–2428). The spec records this as the
  language having no `&&`/`||` tokens at all
  ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) §1). The `||` desugar
  is exercised by the gate's `L4_short_circuit` corpus row.

These desugars are why the grammar header lists only `<` at the comparison level: the full operator
set is built up by the function chain and by these fold/rewrite steps, not by additional grammar
productions.

---

## Parse errors: the `file:line:col` diagnostic

A trust-grade compiler must reject malformed input *at compile time*, loudly, and point at the
offending location — not emit a binary that traps at runtime. `kovc` does this with a small,
deliberately surgical mechanism that threads a byte offset from the parser to the driver.

### How an error node is born

When `parse_primary` hits a token it cannot start an expression from, it falls to a catch-all. That
catch-all is careful not to advance past `EOF`/`}`/`)` (so callers unwind cleanly), and — the load-
bearing line — it stores the offending token's *source byte offset* into `AST_ERR`'s `p2` slot:

**Fragment** — [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx) lines 8998–9011
(the `parse_primary` catch-all that mints `AST_ERR` with a byte offset in `p2`):

```helix
        if t == 6 {
            mk_node(0, 0, 0, 0)
        } else {
            // H-3 (file:line:col diagnostics): thread the offending
            // token's SOURCE BYTE OFFSET (token-record slot 2 =
            // src_start, read via tok_p2) into AST_ERR's p2. Codegen
            // (kovc.hx tag-99 arm) reads only p1 for the runtime trap-id,
            // so p2 is free for diagnostic use. The driver's
            // find_first_err_offset walk reads p2 to map byte->line:col
            // and print `path:line:col: parse error: ...` at COMPILE time.
            // For a CLEAN program this site is never reached, so the AST
            // (and the self-host fixpoint) is byte-identical.
            mk_node(99, t, tok_p2(tok_base, k), 0)
        }
```

The comment captures the key invariant for trust: **on a clean program this site is never reached**,
so the AST is byte-identical and the self-host fixpoint is unperturbed. The diagnostic machinery is
purely a malformed-input path.

### How the driver finds and reports it

Between parse and codegen, the driver in [`kovc.hx`](../../../helixc/bootstrap/kovc.hx) walks the
AST for the *first* `AST_ERR` node and reads its byte offset. If none is found, it returns `-1` and
the normal emit path runs:

**Fragment** — [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 15537–15546
(the start of `walk_for_first_err`; the full walk descends every expression-bearing slot the parser
can produce):

```helix
fn walk_for_first_err(idx: i32) -> i32 {
    if idx == 0 { 0 - 1 } else {
        let t = __arena_get(idx);
        let p1 = __arena_get(idx + 1);
        let p2 = __arena_get(idx + 2);
        let p3 = __arena_get(idx + 3);
        if t == 99 {
            // AST_ERR: p2 holds the source byte offset (H-3). Found.
            p2
        } else { if t == 16 {
```

The byte offset is mapped to a 1-based line by counting newlines, and to a 1-based column by
measuring from the byte after the last newline. The two mappers are tiny and exact:

**Fragment** — [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 15508–15530
(`src_off_to_line` and `src_off_to_col`):

```helix
fn src_off_to_line(src_base: i32, off: i32) -> i32 {
    let mut i: i32 = src_base;
    let mut line: i32 = 1;
    while i < off {
        if __arena_get(i) == 10 { line = line + 1; };
        i = i + 1;
    }
    line
}

// Map a source byte offset to a 1-based COLUMN = (off - index just
// after the last newline before off) + 1. If no newline precedes
// off, the line starts at src_base, so col = off - src_base + 1.
fn src_off_to_col(src_base: i32, off: i32) -> i32 {
    // line_start = index of the first byte of off's line.
    let mut line_start: i32 = src_base;
    let mut i: i32 = src_base;
    while i < off {
        if __arena_get(i) == 10 { line_start = i + 1; };
        i = i + 1;
    }
    (off - line_start) + 1
}
```

Finally, `report_parse_diag` prints the `:line:col: parse error: …` suffix (the driver prints the
`path` prefix itself, since `read_file_to_arena` paths are compile-time literals):

**Fragment** — [`helixc/bootstrap/kovc.hx`](../../../helixc/bootstrap/kovc.hx) lines 15689–15700
(`report_parse_diag`):

```helix
fn report_parse_diag(src_base: i32, off: i32) -> i32 {
    let line = src_off_to_line(src_base, off);
    let col = src_off_to_col(src_base, off);
    print_str(":");
    print_int(line);
    print_str(":");
    print_int(col);
    // print_str_ln appends a REAL newline byte (separate 1-byte
    // sys_write); a plain "\n" in a string literal would emit a
    // literal backslash-n (the lexer has NO string escapes).
    print_str_ln(": parse error: unexpected token");
    0
}
```

The full emitted line is therefore `<path>:<line>:<col>: parse error: unexpected token`, followed by
a non-zero compiler exit and *no* output ELF.

### This path is gate-tested with hand-computed line:col

This is not a best-effort diagnostic — it is a hard gate requirement. Step `[4b]` of the gate
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)) is a *negative* corpus: it feeds the
freshly self-hosted `K2` malformed fixtures and asserts the exact diagnostic string, the non-zero
exit, and the absence of an output ELF:

**Fragment** — [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) lines 633–639 (the
`chk_err` assertion harness):

```bash
  local out rc want; out=$(timeout 20 /tmp/K2.bin 2>&1); rc=$?
  want="/tmp/k2_in.hx:${el}:${ec}: parse error: unexpected token"
  if [ "$rc" = "0" ]; then echo "  EFAIL $b (compiler exited 0 on a parse error)"; efail=$((efail+1)); return; fi
  if [ -s /tmp/k2_out.bin ]; then echo "  EFAIL $b (wrote an output ELF despite the error)"; efail=$((efail+1)); return; fi
  if [ "$out" = "$want" ]; then echo "  EPASS $b -> '$out' (exit $rc)"; epass=$((epass+1));
  else echo "  EFAIL $b: got '$out' want '$want' (exit $rc)"; efail=$((efail+1)); fi
```

The four checked fixtures pin specific positions of a stray `@` token — `chk_err
"$GENC/err_at_l1.hx" 1 20`, `… err_let_rhs.hx 1 28`, `… err_multiline_l3.hx 3 13`, `…
err_after_op_l2.hx 2 9` ([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) lines 645–648) —
and the gate fails if fewer than four pass or any fails
([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) lines 658–660). This is the `CHECK_ERR: 4
passed, 0 failed` line you see in the gate output recorded in
[Part I — Trust at a glance](../part1-orientation/04-trust-at-a-glance.md).

> **For AI agents:** the diagnostic string is an exact contract: `<path>:<line>:<col>: parse error:
> unexpected token`, on a non-zero exit, with no ELF written. If you script around `kovc`'s error
> handling, match that literal (the gate does, in
> [`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh) step `[4b]`), and remember the `path`
> portion is the path you passed to the compiler. A clean program produces *no* diagnostic at all —
> absence of output is success on this axis.

---

## "Typecheck": how the front end tracks types

If you come from a compiler with a distinct semantic-analysis phase, calibrate expectations here:
`kovc`'s front end does **not** run a separate type-checking pass that rejects programs on type
mismatches. The bootstrap is, in the spec's words, **type-erased** — it parses the full annotated
syntax but does not enforce it as a checker
([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md) honesty legend,
`[erased]`). What the front end *does* do is **track a small integer "type tag" per value** so that
codegen can dispatch correctly — i32 vs f32 vs i64 vs u64 arithmetic land on different x86
instructions, and that choice is driven by these tags.

The mapping from a type-name identifier to a tag is `ty_ident_to_tag`, a flat byte-comparison table:

**Fragment** — [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx) lines 1658–1675
(`ty_ident_to_tag`, the 3-byte type names; 2- and 4-byte names follow):

```helix
fn ty_ident_to_tag(ty_s: i32, ty_l: i32) -> i32 {
    if ty_l == 3 {
        let b0 = __arena_get(ty_s);
        let b1 = __arena_get(ty_s + 1);
        let b2 = __arena_get(ty_s + 2);
        if b0 == 102 {
            if b1 == 54 { if b2 == 52 { 2 } else { 0 } }                              // f64
            else { if b1 == 51 { if b2 == 50 { 1 } else { 0 } }                       // f32
            else { if b1 == 49 { if b2 == 54 { 5 } else { 0 } } else { 0 } } }        // v1.3 f16 GAP FIX: f16 (f-1-6) -> tag 5 (reaches emit_f16_binop F16C path)
        } else { if b0 == 105 {
            if b1 == 54 { if b2 == 52 { 3 } else { 0 } }                              // i64
            else { if b1 == 51 { if b2 == 50 { 0 } else { 0 } }                       // i32
            else { if b1 == 49 { if b2 == 54 { 11 } else { 0 } } else { 0 } } }       // i16
        } else { if b0 == 117 {
            if b1 == 51 { if b2 == 50 { 6 } else { 0 } }                              // u32
            else { if b1 == 54 { if b2 == 52 { 9 } else { 0 } }                       // u64
            else { if b1 == 49 { if b2 == 54 { 8 } else { 0 } } else { 0 } } }        // u16
        } else { 0 } } }
```

These tags flow into the AST in two places:

**Function parameters carry their tag in `AST_PARAM` slot 4.** When `parse_fn_decl` reads a param's
`: T` annotation, it resolves the type IDENT and stores the tag so the back end can bind the param
with the right width — the comment is explicit that this is what routes `f32` params to SSE:

**Fragment** — [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx) lines 13007–13017
(`parse_fn_decl` capturing a parameter's type-IDENT bytes):

```helix
            // Capture the type IDENT bytes to determine if it's "f32"
            // (or "f64", treated the same in bootstrap codegen). Step 5c
            // follow-on: this lets fn(a: f32, b: f32) -> f32 { a + b }
            // bind a and b with type=f32 so is_f32_expr resolves through
            // them and AST_ADD dispatches to SSE.
            // K1.DI: also gate on tuple_consumed_di -- if a tuple was
            // consumed by the lookahead path above, no IDENT remains.
            let skip_ident_read = if slice_consumed_cv == 1 { 1 } else { if tuple_consumed_di == 1 { 1 } else { if fnty_consumed_a2c == 1 { 1 } else { 0 } } };
            let ty_tok = cur_get(sb);
            let ty_s = if skip_ident_read == 0 { tok_p2(tok_base, ty_tok) } else { 0 };
            let ty_l = if skip_ident_read == 0 { tok_p3(tok_base, ty_tok) } else { 0 };
```

**Typed `let` bindings are recorded in a small per-function table.** `var_type_tab_add` /
`var_type_tab_lookup` keep `(name_s, name_l, ty_tag)` triples so a later reference to that variable
recovers its width. The table is intentionally tiny — capacity 8 — because it scopes a single
function body:

**Fragment** — [`helixc/bootstrap/parser.hx`](../../../helixc/bootstrap/parser.hx) lines 1154–1167
(`var_type_tab_add` — the typed-binding table writer):

```helix
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
```

The same identifier-table pattern recurs across the parser's state: `struct_tab` resolves struct
names to field layouts, `enum_tab` resolves enum variants to discriminants and arities, `gp_tab`
holds the current function's generic parameters for monomorphization, and `mr_tab` records mangled
monomorphization names. They are all flat arena tables with an `_add` / `_lookup` pair, keyed by
byte-equality on names. Together they are the closest thing the bootstrap has to a symbol table —
but they exist to *drive codegen dispatch and name resolution*, not to *reject ill-typed programs*.

> **For AI agents:** do not describe `kovc` as having a type checker that rejects type errors. It is
> type-*erased* in the spec's sense ([`docs/HELIX_V1_LANGUAGE_SPEC.md`](../../../docs/HELIX_V1_LANGUAGE_SPEC.md)
> `[erased]` legend). The front end tracks integer *type tags* (`ty_ident_to_tag`,
> `var_type_tab`, `AST_PARAM` slot 4) so the back end picks the right instruction width; mismatches
> are not diagnosed at parse time. The diagnostics that *are* enforced are the parse-error path above
> plus the negative codegen checks the back end emits (see
> [the x86-64 ELF back end](03-x86-backend.md)).

---

## How the front end feeds the self-host fixpoint

Pulling the threads together: the front end you have just read is one of the inputs to its own
output. `seed` compiles `lexer.hx` + `parser.hx` + `kovc.hx` into `K1`; `K1` compiles the same
sources into `K2`; and `K2 == K3 == K4` must be byte-identical to the pinned `0992dddd…`
([Part I — Trust at a glance](../part1-orientation/04-trust-at-a-glance.md);
[`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md)). That round trip is exactly why
the front end is shaped the way it is:

- Every lexer fix quoted above (`is_alpha` and underscores, hex literals, the `_`-vs-suffix
  lookahead) is a fixpoint regression — a case where mis-lexing `kovc`'s *own* source produced a
  wrong `K2`. The self-host fixpoint is the front end's most demanding test, because the front end's
  input includes itself.
- The parse-error diagnostic was added with a hard invariant that clean programs reach no `AST_ERR`
  node, so the AST — and therefore the emitted ELF — stays byte-identical. A diagnostic feature that
  perturbed clean-program output would have broken the fixpoint; this one cannot, by construction.
- The type-tag tracking is enough to compile a compiler that compiles a transformer's GPU kernels
  (the capstone), but it is deliberately *not* a full checker — the bootstrap keeps its surface
  small so the self-host round trip stays tractable and auditable. The honest residual posture for
  that capstone and the GPU path lives in
  [`docs/TRUST_CHAIN_CLOSED.md`](../../../docs/TRUST_CHAIN_CLOSED.md) (complete to PTX, not GPU
  machine code; GPU throughput a fraction of cuBLAS); none of it is claimed by the front end itself.

> **For AI agents:** when you modify any front-end source, the binding check is the gate
> ([`scripts/gate_kovc.sh`](../../../scripts/gate_kovc.sh)): it must still print `GATE_PASS`, with
> the self-host fixpoint line `FIXPOINT OK (K2==K3==K4 byte-identical AND == pinned known-good)`,
> the `CORPUS: 109 passed, 0 failed`, and `CHECK_ERR: 4 passed, 0 failed`. A lexer or parser change
> that alters the AST for any clean program will change the emitted `kovc` and break the fixpoint —
> treat a fixpoint mismatch after a front-end edit as a real regression, not as needing a re-pin.

---

**Next:** with source turned into an AST, [IR & lowering passes](02-ir-and-passes.md)
covers what `kovc` does to the AST *before* codegen — the monomorphization, autodiff, and folding
passes that run between `parse_top` and the ELF back end.
