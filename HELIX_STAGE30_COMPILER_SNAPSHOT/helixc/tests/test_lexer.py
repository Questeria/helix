"""Tests for helixc.frontend.lexer."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.lexer import lex, T, LexError


# ============================================================================
# Helpers
# ============================================================================
def kinds(src: str) -> list[T]:
    return [t.kind for t in lex(src)[:-1]]  # drop trailing EOF


def value(src: str) -> list[str]:
    return [t.value for t in lex(src)[:-1]]


# ============================================================================
# Tests — basic tokens
# ============================================================================
def test_empty():
    toks = lex("")
    assert len(toks) == 1
    assert toks[0].kind == T.EOF


def test_whitespace_only():
    toks = lex("   \t\n  \r\n  ")
    assert len(toks) == 1
    assert toks[0].kind == T.EOF


def test_line_comment():
    assert kinds("// just a comment") == []
    assert kinds("// comment\n42") == [T.INT]


def test_block_comment():
    assert kinds("/* hello */") == []
    assert kinds("/* nested /* inner */ done */ x") == [T.IDENT]


def test_unterminated_block_comment():
    try:
        lex("/* never ends")
        assert False, "expected LexError"
    except LexError:
        pass


# ---- identifiers and keywords ----
def test_identifiers():
    toks = lex("foo _bar baz123 _")
    assert all(t.kind == T.IDENT for t in toks[:-1])


def test_keywords_basic():
    assert kinds("fn let mut const") == [T.KW_FN, T.KW_LET, T.KW_MUT, T.KW_CONST]


def test_keywords_types():
    assert kinds("i32 u64 f32 bf16 ternary") == [
        T.KW_I32, T.KW_U64, T.KW_F32, T.KW_BF16, T.KW_TERNARY,
    ]


def test_keywords_tile_tensor():
    assert kinds("tile tensor smem hbm reg tmem") == [
        T.KW_TILE, T.KW_TENSOR, T.KW_SMEM, T.KW_HBM, T.KW_REG, T.KW_TMEM,
    ]


def test_module_use():
    assert kinds("module use pub priv") == [T.KW_MODULE, T.KW_USE, T.KW_PUB, T.KW_PRIV]


# ---- integer literals ----
def test_int_decimal():
    t = lex("42")[0]
    assert t.kind == T.INT and t.int_value == 42


def test_int_hex():
    t = lex("0xFF")[0]
    assert t.int_value == 255


def test_int_binary():
    t = lex("0b1010")[0]
    assert t.int_value == 10


def test_int_octal():
    t = lex("0o755")[0]
    assert t.int_value == 0o755


def test_int_underscores():
    t = lex("1_000_000")[0]
    assert t.int_value == 1_000_000


def test_int_with_suffix():
    t = lex("42_i32")[0]
    assert t.kind == T.INT and t.int_value == 42 and t.type_suffix == "i32"


def test_int_with_unrecognized_suffix_rewinds():
    # 42_foo should lex as INT(42), then IDENT(_foo) — wait, _foo is also an ident
    # Actually our rewind: 42 lexes as int with no suffix, then "_foo" starts a new ident
    toks = lex("42_unknownsuffix")
    # The lexer should rewind on unrecognized suffix and the underscore starts a new ident
    assert toks[0].kind == T.INT
    assert toks[0].int_value == 42
    assert toks[0].type_suffix is None
    assert toks[1].kind == T.IDENT
    assert toks[1].value == "_unknownsuffix"


# ---- float literals ----
def test_float_basic():
    t = lex("3.14")[0]
    assert t.kind == T.FLOAT and abs(t.float_value - 3.14) < 1e-9


def test_float_exp():
    t = lex("1e-5")[0]
    assert t.kind == T.FLOAT and abs(t.float_value - 1e-5) < 1e-15


def test_float_typed():
    t = lex("3.14_f32")[0]
    assert t.kind == T.FLOAT and t.type_suffix == "f32"


def test_float_bf16_suffix():
    t = lex("1.5_bf16")[0]
    assert t.kind == T.FLOAT and t.type_suffix == "bf16"


# ---- strings ----
def test_string_basic():
    t = lex('"hello"')[0]
    assert t.kind == T.STRING and t.string_value == "hello"


def test_string_escapes():
    t = lex(r'"a\nb\tc"')[0]
    assert t.string_value == "a\nb\tc"


def test_string_hex_escape():
    t = lex(r'"\x48\x69"')[0]
    assert t.string_value == "Hi"


def test_string_unicode_escape():
    t = lex(r'"\u{1F600}"')[0]
    assert t.string_value == "\U0001F600"


def test_unterminated_string():
    try:
        lex('"never closes')
        assert False
    except LexError:
        pass


# ---- chars ----
def test_char_simple():
    t = lex("'a'")[0]
    assert t.kind == T.CHAR and t.char_value == "a"


def test_char_escape():
    t = lex(r"'\n'")[0]
    assert t.char_value == "\n"


# ---- operators ----
def test_arith_ops():
    assert kinds("+ - * / %") == [T.PLUS, T.MINUS, T.STAR, T.SLASH, T.PERCENT]


def test_compare_ops():
    assert kinds("== != < > <= >=") == [
        T.EQEQ, T.NEQ, T.LT, T.GT, T.LEQ, T.GEQ,
    ]


def test_logical_ops():
    assert kinds("&& || !") == [T.LAND, T.LOR, T.BANG]


def test_bitwise_ops():
    assert kinds("& | ^ ~ << >>") == [
        T.AMP, T.PIPE, T.CARET, T.TILDE, T.SHL, T.SHR,
    ]


def test_assign_ops():
    assert kinds("= += -= *= /= %=") == [
        T.EQ, T.PLUSEQ, T.MINUSEQ, T.STAREQ, T.SLASHEQ, T.PERCENTEQ,
    ]


def test_punctuation():
    assert kinds("-> => :: . .. ;") == [
        T.ARROW, T.FATARROW, T.COLONCOLON, T.DOT, T.DOTDOT, T.SEMI,
    ]


def test_brackets():
    assert kinds("( ) [ ] { }") == [
        T.LPAREN, T.RPAREN, T.LBRACK, T.RBRACK, T.LBRACE, T.RBRACE,
    ]


def test_at_question():
    assert kinds("@ ?") == [T.AT, T.QUESTION]


# ---- realistic snippets ----
def test_function_signature():
    src = "fn matmul[N: size, M: size](a: tensor<f32, [N, M]>) -> i32 { 0 }"
    toks = lex(src)
    # Spot-check a few key tokens
    assert toks[0].kind == T.KW_FN
    assert toks[1].kind == T.IDENT and toks[1].value == "matmul"
    assert toks[2].kind == T.LBRACK
    # Find the size keyword
    sizes = [t for t in toks if t.kind == T.KW_SIZE]
    assert len(sizes) == 2


def test_tile_type():
    src = "tile<bf16, [16, 16], smem>"
    toks = lex(src)
    assert toks[0].kind == T.KW_TILE
    assert toks[2].kind == T.KW_BF16
    # smem
    assert any(t.kind == T.KW_SMEM for t in toks)


def test_attribute():
    src = "@kernel fn foo() {}"
    toks = lex(src)
    assert toks[0].kind == T.AT
    assert toks[1].kind == T.KW_KERNEL


def test_grad_call():
    src = "let g = grad(loss);"
    toks = lex(src)
    assert any(t.kind == T.KW_GRAD for t in toks)


def test_full_example_lexes():
    src = """
    module examples::matmul

    @pure
    fn matmul[N: size, M: size, P: size](
        a: tensor<bf16, [N, M], gpu(0)>,
        b: tensor<bf16, [M, P], gpu(0)>,
    ) -> tensor<f32, [N, P], gpu(0)>
    where N % 16 == 0,
    {
        let mut c = tensor::zeros::<f32, [N, P]>(gpu(0));
        c
    }
    """
    toks = lex(src)
    assert toks[-1].kind == T.EOF
    # Some sanity: there should be multiple KW_TENSOR, KW_BF16, KW_F32, KW_GPU
    assert sum(1 for t in toks if t.kind == T.KW_TENSOR) >= 4
    assert sum(1 for t in toks if t.kind == T.KW_GPU) >= 3


# ---- error reporting ----
def test_unknown_char_error():
    try:
        lex("foo `bar")  # backtick is not valid
        assert False
    except LexError as e:
        assert e.line == 1
        assert e.col >= 5


# ============================================================================
# Test runner
# ============================================================================
def main():
    import inspect
    tests = [(name, fn) for name, fn in globals().items()
             if name.startswith("test_") and callable(fn)]
    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
