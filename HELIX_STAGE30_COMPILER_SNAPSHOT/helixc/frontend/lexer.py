"""
helixc/frontend/lexer.py — Helix language lexer (Python prototype).

Build-time only. Will be replaced by self-hosted Helix implementation in Phase 4.

Token kinds match the Helix spec at docs/lang/spec.md.

License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator


# ============================================================================
# Token kinds
# ============================================================================
class T(Enum):
    # Literals
    INT = auto()           # 42, 0xFF, 0b1010, 0o755 (with optional _typeSuffix)
    FLOAT = auto()         # 3.14, 1e-5 (with optional _typeSuffix)
    STRING = auto()        # "hello\n"
    CHAR = auto()          # 'a'
    IDENT = auto()         # foo_bar

    # Keywords (kept as separate kinds for parser clarity)
    KW_FN = auto(); KW_LET = auto(); KW_MUT = auto(); KW_CONST = auto()
    KW_TYPE = auto(); KW_STRUCT = auto(); KW_ENUM = auto()
    KW_TRAIT = auto(); KW_IMPL = auto()
    KW_IF = auto(); KW_ELSE = auto(); KW_MATCH = auto()
    KW_FOR = auto(); KW_WHILE = auto(); KW_LOOP = auto()
    KW_BREAK = auto(); KW_CONTINUE = auto(); KW_RETURN = auto()
    KW_TRUE = auto(); KW_FALSE = auto()
    KW_BOOL = auto(); KW_CHAR_TY = auto()
    KW_I8 = auto(); KW_I16 = auto(); KW_I32 = auto(); KW_I64 = auto(); KW_ISIZE = auto()
    KW_U8 = auto(); KW_U16 = auto(); KW_U32 = auto(); KW_U64 = auto(); KW_USIZE = auto()
    KW_BF16 = auto(); KW_F16 = auto(); KW_F32 = auto(); KW_F64 = auto()
    KW_FP8 = auto(); KW_MXFP4 = auto(); KW_NVFP4 = auto(); KW_TERNARY = auto()
    KW_TILE = auto(); KW_TENSOR = auto()
    KW_WHERE = auto(); KW_AS = auto(); KW_IN = auto(); KW_OF = auto()
    KW_PUB = auto(); KW_PRIV = auto(); KW_MOD = auto(); KW_USE = auto(); KW_MODULE = auto()
    KW_ASYNC = auto(); KW_AWAIT = auto()
    KW_DEVICE = auto(); KW_CPU = auto(); KW_GPU = auto()
    KW_HBM = auto(); KW_SMEM = auto(); KW_REG = auto(); KW_TMEM = auto()
    KW_KERNEL = auto(); KW_GRAD = auto(); KW_JVP = auto(); KW_VJP = auto(); KW_VMAP = auto()
    KW_SIZE = auto()
    # AGI-specific keywords
    KW_QUOTE = auto(); KW_SPLICE = auto()
    KW_MODIFY = auto(); KW_VERIFIER = auto()
    KW_AGENT = auto(); KW_SOCIETY = auto()
    KW_PURE = auto(); KW_EFFECT = auto()
    # Stage 16.5: FFI keyword
    KW_EXTERN = auto()
    # Stage 28.6: unsafe block keyword
    KW_UNSAFE = auto()

    # Operators / punctuation
    PLUS = auto(); MINUS = auto(); STAR = auto(); SLASH = auto(); PERCENT = auto()
    EQ = auto(); EQEQ = auto(); NEQ = auto()
    LT = auto(); GT = auto(); LEQ = auto(); GEQ = auto()
    LAND = auto(); LOR = auto(); BANG = auto()
    AMP = auto(); PIPE = auto(); CARET = auto(); TILDE = auto()
    SHL = auto(); SHR = auto()
    PLUSEQ = auto(); MINUSEQ = auto(); STAREQ = auto(); SLASHEQ = auto(); PERCENTEQ = auto()
    ARROW = auto(); FATARROW = auto()
    COLON = auto(); COLONCOLON = auto(); SEMI = auto(); COMMA = auto()
    DOT = auto(); DOTDOT = auto(); DOTDOTEQ = auto()
    LPAREN = auto(); RPAREN = auto()
    LBRACK = auto(); RBRACK = auto()
    LBRACE = auto(); RBRACE = auto()
    AT = auto(); QUESTION = auto()

    # Special
    NEWLINE = auto()       # not significant for v0.1, kept for diagnostics
    EOF = auto()


KEYWORDS = {
    "fn": T.KW_FN, "let": T.KW_LET, "mut": T.KW_MUT, "const": T.KW_CONST,
    "type": T.KW_TYPE, "struct": T.KW_STRUCT, "enum": T.KW_ENUM,
    "trait": T.KW_TRAIT, "impl": T.KW_IMPL,
    "if": T.KW_IF, "else": T.KW_ELSE, "match": T.KW_MATCH,
    "for": T.KW_FOR, "while": T.KW_WHILE, "loop": T.KW_LOOP,
    "break": T.KW_BREAK, "continue": T.KW_CONTINUE, "return": T.KW_RETURN,
    "true": T.KW_TRUE, "false": T.KW_FALSE,
    "bool": T.KW_BOOL, "char": T.KW_CHAR_TY,
    "i8": T.KW_I8, "i16": T.KW_I16, "i32": T.KW_I32, "i64": T.KW_I64, "isize": T.KW_ISIZE,
    "u8": T.KW_U8, "u16": T.KW_U16, "u32": T.KW_U32, "u64": T.KW_U64, "usize": T.KW_USIZE,
    "bf16": T.KW_BF16, "f16": T.KW_F16, "f32": T.KW_F32, "f64": T.KW_F64,
    "fp8": T.KW_FP8, "mxfp4": T.KW_MXFP4, "nvfp4": T.KW_NVFP4, "ternary": T.KW_TERNARY,
    "tile": T.KW_TILE, "tensor": T.KW_TENSOR,
    "where": T.KW_WHERE, "as": T.KW_AS, "in": T.KW_IN, "of": T.KW_OF,
    "pub": T.KW_PUB, "priv": T.KW_PRIV, "mod": T.KW_MOD, "use": T.KW_USE,
    "module": T.KW_MODULE,
    "async": T.KW_ASYNC, "await": T.KW_AWAIT,
    "device": T.KW_DEVICE, "cpu": T.KW_CPU, "gpu": T.KW_GPU,
    "hbm": T.KW_HBM, "smem": T.KW_SMEM, "reg": T.KW_REG, "tmem": T.KW_TMEM,
    "kernel": T.KW_KERNEL, "grad": T.KW_GRAD, "jvp": T.KW_JVP,
    "vjp": T.KW_VJP, "vmap": T.KW_VMAP,
    "size": T.KW_SIZE,
    "quote": T.KW_QUOTE, "splice": T.KW_SPLICE,
    "modify": T.KW_MODIFY, "verifier": T.KW_VERIFIER,
    "agent": T.KW_AGENT, "society": T.KW_SOCIETY,
    "pure": T.KW_PURE, "effect": T.KW_EFFECT,
    # Stage 16.5: FFI
    "extern": T.KW_EXTERN,
    # Stage 28.6: unsafe block
    "unsafe": T.KW_UNSAFE,
}


@dataclass(frozen=True)
class Token:
    kind: T
    value: str       # raw source slice (the lexeme)
    line: int        # 1-based
    col: int         # 1-based, byte column
    # For literals, optionally a typed payload:
    int_value: int | None = None
    float_value: float | None = None
    type_suffix: str | None = None   # "i32", "u8", "bf16" etc on numeric literals
    string_value: str | None = None  # decoded string content
    char_value: str | None = None

    def __repr__(self) -> str:
        extra = ""
        if self.int_value is not None:
            extra = f" int={self.int_value}"
        elif self.float_value is not None:
            extra = f" float={self.float_value}"
        elif self.string_value is not None:
            extra = f" str={self.string_value!r}"
        elif self.char_value is not None:
            extra = f" char={self.char_value!r}"
        suffix = f" suffix={self.type_suffix}" if self.type_suffix else ""
        return f"<{self.kind.name} {self.value!r} @{self.line}:{self.col}{extra}{suffix}>"


# ============================================================================
# Lexer
# ============================================================================
class LexError(Exception):
    def __init__(self, msg: str, line: int, col: int):
        super().__init__(f"{line}:{col}: {msg}")
        self.line = line
        self.col = col


class Lexer:
    def __init__(self, source: str, filename: str = "<input>"):
        self.src = source
        self.filename = filename
        self.pos = 0
        self.line = 1
        self.col = 1
        # Speedup #6: cache len(self.src). The source string is immutable
        # after construction, so its length never changes. _peek and _at_eof
        # were each calling len(self.src) on every invocation (1.5M+ calls
        # per bootstrap build → ~1.3 sec spent in the len() built-in).
        self._n = len(source)

    # ---- char helpers ----
    def _peek(self, ahead: int = 0) -> str:
        p = self.pos + ahead
        return self.src[p] if p < self._n else ""

    def _advance(self) -> str:
        c = self.src[self.pos]
        self.pos += 1
        if c == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return c

    def _starts_with(self, s: str) -> bool:
        return self.src.startswith(s, self.pos)

    def _consume(self, s: str) -> bool:
        if self._starts_with(s):
            for _ in s:
                self._advance()
            return True
        return False

    def _at_eof(self) -> bool:
        return self.pos >= self._n

    # ---- main loop ----
    def tokens(self) -> list[Token]:
        out: list[Token] = []
        while not self._at_eof():
            tok = self._next()
            if tok is not None:
                out.append(tok)
        out.append(Token(T.EOF, "", self.line, self.col))
        return out

    def _next(self) -> Token | None:
        # Skip whitespace and comments
        while not self._at_eof():
            c = self._peek()
            if c in (" ", "\t", "\r", "\n"):
                self._advance()
            elif c == "/" and self._peek(1) == "/":
                # line comment
                while not self._at_eof() and self._peek() != "\n":
                    self._advance()
            elif c == "/" and self._peek(1) == "*":
                # block comment, nestable
                self._advance(); self._advance()
                depth = 1
                while not self._at_eof() and depth > 0:
                    if self._consume("/*"):
                        depth += 1
                    elif self._consume("*/"):
                        depth -= 1
                    else:
                        self._advance()
                if depth != 0:
                    raise LexError("unterminated block comment", self.line, self.col)
            else:
                break

        if self._at_eof():
            return None

        line, col = self.line, self.col
        c = self._peek()

        # Identifiers and keywords
        if c.isalpha() or c == "_":
            return self._lex_ident(line, col)

        # Number literals (integer or float)
        if c.isdigit():
            return self._lex_number(line, col)

        # String / char literals
        if c == '"':
            return self._lex_string(line, col)
        if c == "'":
            return self._lex_char(line, col)

        # Operators / punctuation
        return self._lex_op(line, col)

    # ---- ident / keyword ----
    def _lex_ident(self, line: int, col: int) -> Token:
        # Speedup #7: cache _peek() result once per iteration. Original
        # called _peek() TWICE per loop iter (once for isalnum, once for
        # underscore check). For 28k+ idents × ~10 chars × 2 peeks =
        # ~560k extra _peek calls eliminated.
        start = self.pos
        while not self._at_eof():
            c = self._peek()
            if not (c.isalnum() or c == "_"):
                break
            self._advance()
        lexeme = self.src[start:self.pos]
        kind = KEYWORDS.get(lexeme, T.IDENT)
        return Token(kind, lexeme, line, col)

    # ---- number ----
    def _lex_number(self, line: int, col: int) -> Token:
        start = self.pos
        is_float = False
        # Detect base prefix
        if self._peek() == "0" and self._peek(1) in ("x", "X"):
            self._advance(); self._advance()
            digits_start = self.pos
            while not self._at_eof() and (self._peek().lower() in "0123456789abcdef" or self._peek() == "_"):
                self._advance()
            digits = self.src[digits_start:self.pos].replace("_", "")
            if not digits:
                raise LexError("hex literal needs at least one digit", line, col)
            value = int(digits, 16)
        elif self._peek() == "0" and self._peek(1) in ("b", "B"):
            self._advance(); self._advance()
            digits_start = self.pos
            while not self._at_eof() and (self._peek() in "01" or self._peek() == "_"):
                self._advance()
            digits = self.src[digits_start:self.pos].replace("_", "")
            if not digits:
                raise LexError("binary literal needs at least one digit", line, col)
            value = int(digits, 2)
        elif self._peek() == "0" and self._peek(1) in ("o", "O"):
            self._advance(); self._advance()
            digits_start = self.pos
            while not self._at_eof() and (self._peek() in "01234567" or self._peek() == "_"):
                self._advance()
            digits = self.src[digits_start:self.pos].replace("_", "")
            if not digits:
                raise LexError("octal literal needs at least one digit", line, col)
            value = int(digits, 8)
        else:
            # Decimal int or float — allow underscores BETWEEN digits only
            # (so '1_000_000' works but '42_i32' lets the '_' kick off suffix parsing)
            def _consume_digit_run() -> None:
                while not self._at_eof():
                    if self._peek().isdigit():
                        self._advance()
                    elif self._peek() == "_" and self._peek(1).isdigit():
                        self._advance()  # underscore-between-digits
                    else:
                        break
            _consume_digit_run()
            if not self._at_eof() and self._peek() == "." and self._peek(1).isdigit():
                is_float = True
                self._advance()  # consume '.'
                _consume_digit_run()
            # exponent
            if not self._at_eof() and self._peek() in ("e", "E"):
                is_float = True
                self._advance()
                if not self._at_eof() and self._peek() in ("+", "-"):
                    self._advance()
                if self._at_eof() or not self._peek().isdigit():
                    raise LexError("exponent needs digits", line, col)
                _consume_digit_run()
            digits = self.src[start:self.pos].replace("_", "")
            value = float(digits) if is_float else int(digits)

        # Optional type suffix: _i32, _u8, _bf16, etc
        suffix = None
        if self._peek() == "_":
            save_pos, save_line, save_col = self.pos, self.line, self.col
            self._advance()
            if not self._at_eof() and (self._peek().isalpha()):
                suf_start = self.pos
                while not self._at_eof() and (self._peek().isalnum()):
                    self._advance()
                candidate = self.src[suf_start:self.pos]
                if candidate in {"i8","i16","i32","i64","isize",
                                 "u8","u16","u32","u64","usize",
                                 "bf16","f16","f32","f64",
                                 "fp8","mxfp4","nvfp4","ternary"}:
                    suffix = candidate
                else:
                    # not a recognized suffix; rewind
                    self.pos, self.line, self.col = save_pos, save_line, save_col
            else:
                self.pos, self.line, self.col = save_pos, save_line, save_col

        lexeme = self.src[start:self.pos]
        if is_float:
            return Token(T.FLOAT, lexeme, line, col,
                         float_value=value, type_suffix=suffix)
        return Token(T.INT, lexeme, line, col,
                     int_value=value, type_suffix=suffix)

    # ---- string ----
    def _lex_string(self, line: int, col: int) -> Token:
        self._advance()  # consume opening "
        start = self.pos
        decoded: list[str] = []
        while not self._at_eof() and self._peek() != '"':
            if self._peek() == "\\":
                self._advance()
                if self._at_eof():
                    raise LexError("unterminated escape", line, col)
                esc = self._advance()
                decoded.append(self._decode_escape(esc, line, col))
            else:
                decoded.append(self._advance())
        if self._at_eof():
            raise LexError("unterminated string literal", line, col)
        self._advance()  # consume closing "
        lexeme = self.src[start - 1 : self.pos]
        return Token(T.STRING, lexeme, line, col, string_value="".join(decoded))

    def _decode_escape(self, esc: str, line: int, col: int) -> str:
        simple = {"n": "\n", "t": "\t", "r": "\r", "\\": "\\",
                  '"': '"', "'": "'", "0": "\0"}
        if esc in simple:
            return simple[esc]
        if esc == "x":
            h1 = self._advance() if not self._at_eof() else ""
            h2 = self._advance() if not self._at_eof() else ""
            hex_chars = "0123456789abcdefABCDEF"
            if (len(h1 + h2) != 2 or h1 not in hex_chars
                    or h2 not in hex_chars):
                raise LexError(r"\x escape needs 2 hex digits", line, col)
            return chr(int(h1 + h2, 16))
        if esc == "u":
            if self._peek() != "{":
                raise LexError(r"\u escape needs {hex}", line, col)
            self._advance()
            digs = ""
            while not self._at_eof() and self._peek() != "}":
                digs += self._advance()
            if self._peek() != "}":
                raise LexError(r"\u escape unterminated", line, col)
            self._advance()
            try:
                return chr(int(digs, 16))
            except ValueError:
                raise LexError(r"\u escape: invalid hex", line, col) from None
        raise LexError(f"unknown escape \\{esc}", line, col)

    # ---- char ----
    def _lex_char(self, line: int, col: int) -> Token:
        self._advance()  # consume opening '
        if self._at_eof():
            raise LexError("unterminated char literal", line, col)
        if self._peek() == "\\":
            self._advance()
            esc = self._advance()
            ch = self._decode_escape(esc, line, col)
        else:
            ch = self._advance()
        if self._at_eof() or self._peek() != "'":
            raise LexError("unterminated char literal", line, col)
        self._advance()  # consume closing '
        return Token(T.CHAR, f"'{ch}'", line, col, char_value=ch)

    # ---- operators ----
    def _lex_op(self, line: int, col: int) -> Token:
        # Multi-character operators first (longest match). Speedup #5:
        # _MULTI_CHAR_OPS and _SINGLE_CHAR_OPS hoisted to module level
        # below — was previously rebuilt on every call (36k+ calls per
        # bootstrap compile), wasting ~0.5 sec on tuple/dict alloc.
        for s, kind in _MULTI_CHAR_OPS:
            if self._consume(s):
                return Token(kind, s, line, col)

        # Single-character
        c = self._advance()
        kind = _SINGLE_CHAR_OPS.get(c)
        if kind is not None:
            return Token(kind, c, line, col)
        raise LexError(f"unexpected character {c!r}", line, col)


# Speedup #5: module-level operator tables (hot in _lex_op).
# Tuple of (str, T) ordered by longest-match-first. Before this hoist,
# both tables were rebuilt on every _lex_op call (36k+ allocs per
# bootstrap build) — measurable in cProfile.
_MULTI_CHAR_OPS = (
    ("==", T.EQEQ), ("!=", T.NEQ), ("<=", T.LEQ), (">=", T.GEQ),
    ("&&", T.LAND), ("||", T.LOR),
    ("<<", T.SHL), (">>", T.SHR),
    ("+=", T.PLUSEQ), ("-=", T.MINUSEQ), ("*=", T.STAREQ),
    ("/=", T.SLASHEQ), ("%=", T.PERCENTEQ),
    ("->", T.ARROW), ("=>", T.FATARROW),
    ("::", T.COLONCOLON), ("..=", T.DOTDOTEQ), ("..", T.DOTDOT),
)
_SINGLE_CHAR_OPS = {
    "+": T.PLUS, "-": T.MINUS, "*": T.STAR, "/": T.SLASH, "%": T.PERCENT,
    "=": T.EQ, "<": T.LT, ">": T.GT,
    "!": T.BANG, "&": T.AMP, "|": T.PIPE, "^": T.CARET, "~": T.TILDE,
    ":": T.COLON, ";": T.SEMI, ",": T.COMMA, ".": T.DOT,
    "(": T.LPAREN, ")": T.RPAREN,
    "[": T.LBRACK, "]": T.RBRACK,
    "{": T.LBRACE, "}": T.RBRACE,
    "@": T.AT, "?": T.QUESTION,
}


def lex(source: str, filename: str = "<input>") -> list[Token]:
    """Convenience: lex a full source string into a token list."""
    return Lexer(source, filename).tokens()


# ============================================================================
# CLI for quick testing
# ============================================================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        with open(sys.argv[1], "r", encoding="utf-8") as f:
            src = f.read()
    else:
        src = sys.stdin.read()
    for tok in lex(src):
        print(tok)
