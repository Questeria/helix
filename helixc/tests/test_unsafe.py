"""Stage 28.6: tests for unsafe block + raw-ptr-op gating."""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse, ParseError
from helixc.frontend import ast_nodes as A
from helixc.frontend.unsafe_pass import (
    find_unsafe_blocks,
    find_raw_ptr_ops,
    check_unsafe_ops,
    TRAP_UNSAFE_OP_OUTSIDE,
    TRAP_EXTERN_CALL_OUTSIDE_UNSAFE,
)


def test_parse_unsafe_block_empty():
    src = """
fn main() -> i32 {
    unsafe { }
    0
}
"""
    prog = parse(src)
    blocks = find_unsafe_blocks(prog)
    assert len(blocks) == 1


def test_parse_unsafe_block_with_body():
    src = """
fn main() -> i32 {
    unsafe { let x = 1; x }
}
"""
    prog = parse(src)
    blocks = find_unsafe_blocks(prog)
    assert len(blocks) == 1
    inner = blocks[0].body
    assert isinstance(inner, A.Block)


def test_no_unsafe_blocks():
    src = "fn main() -> i32 { 0 }"
    prog = parse(src)
    assert find_unsafe_blocks(prog) == []


def test_unsafe_keyword_recognized():
    """unsafe shouldn't tokenize as an identifier."""
    from helixc.frontend.lexer import lex, T
    toks = lex("unsafe", "x.hx")
    assert toks[0].kind == T.KW_UNSAFE


def test_raw_ptr_op_inside_unsafe_ok():
    """`*p` inside unsafe should NOT trigger 28601."""
    src = """
fn main() -> i32 {
    let p: *const i32 = 0 as *const i32;
    unsafe { *p }
}
"""
    prog = parse(src)
    diags = check_unsafe_ops(prog)
    assert diags == []


def test_raw_ptr_op_outside_unsafe_diag():
    """`*p` outside any unsafe block should trigger 28601 diag."""
    src = """
fn main() -> i32 {
    let p: *const i32 = 0 as *const i32;
    *p
}
"""
    prog = parse(src)
    diags = check_unsafe_ops(prog)
    assert diags
    assert "28601" in diags[0]


def test_multiple_unsafe_blocks():
    src = """
fn main() -> i32 {
    unsafe { 1 };
    unsafe { 2 };
    0
}
"""
    prog = parse(src)
    blocks = find_unsafe_blocks(prog)
    assert len(blocks) == 2


def test_find_raw_ptr_ops_records_context():
    src = """
fn main() -> i32 {
    let p: *const i32 = 0 as *const i32;
    let inside = unsafe { *p };
    let outside = *p;
    0
}
"""
    prog = parse(src)
    ops = find_raw_ptr_ops(prog)
    assert len(ops) == 2
    # First should be inside unsafe, second not.
    contexts = sorted({c for (_, _, c) in ops})
    assert contexts == [False, True]


def test_trap_ids():
    assert TRAP_UNSAFE_OP_OUTSIDE == 28601
    assert TRAP_EXTERN_CALL_OUTSIDE_UNSAFE == 28602


def test_nested_unsafe_blocks():
    """nested unsafe { unsafe { } } — both register."""
    src = """
fn main() -> i32 {
    unsafe { unsafe { 1 } }
}
"""
    prog = parse(src)
    blocks = find_unsafe_blocks(prog)
    assert len(blocks) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
