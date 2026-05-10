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


# ----------------------------------------------------------------------
# Audit 28.8 A2 — UnsafeBlock now lowers (treated as a transparent
# Block) and check_unsafe_ops is wired into helixc/check.py so a raw
# deref outside unsafe surfaces trap 28601.
# ----------------------------------------------------------------------
def test_unsafe_block_lowers_as_block():
    """A2: `unsafe { ... }` lowers identically to a plain Block. The
    IR for `unsafe { 42 }` should match the IR for `{ 42 }`."""
    from helixc.ir.lower_ast import lower
    src_unsafe = "fn main() -> i32 { unsafe { 42 } }"
    src_plain = "fn main() -> i32 { { 42 } }"
    mod_u = lower(parse(src_unsafe))
    mod_p = lower(parse(src_plain))
    # Both should compile to an IR with the same op-kind sequence (no
    # extra ops from the UnsafeBlock wrapper).
    u_kinds = [op.kind for blk in mod_u.functions["main"].blocks for op in blk.ops]
    p_kinds = [op.kind for blk in mod_p.functions["main"].blocks for op in blk.ops]
    assert u_kinds == p_kinds, (
        f"UnsafeBlock should lower identically to Block; "
        f"got unsafe={u_kinds}, plain={p_kinds}"
    )


def test_unsafe_block_with_deref_lowers():
    """A2: `unsafe { *p }` lowers without errors (pre-fix it would
    fall into the generic dispatch and produce an unexpected op-kind)."""
    from helixc.ir.lower_ast import lower
    src = """
fn main() -> i32 {
    let p: *const i32 = 0 as *const i32;
    unsafe { *p }
}
"""
    prog = parse(src)
    # Lowering should succeed without raising — the IR may not be
    # *runnable* (0 as *const i32 is a null ptr), but the codegen path
    # must accept the syntactic form.
    mod = lower(prog)
    assert "main" in mod.functions


def test_cli_check_unsafe_ops_wired(capsys, tmp_path):
    """A2: helixc/check.py invokes check_unsafe_ops directly (we drive
    the pass via the same import wiring the CLI uses, since a full
    --check-only run may surface typecheck errors first for the same
    program shape)."""
    from helixc.check import main
    # Use `;` to discard the *p value so typecheck is happy.
    src = """
fn helper() -> i32 {
    let p: *const i32 = 0 as *const i32;
    *p;
    0
}
fn main() -> i32 { helper() }
"""
    p = str(tmp_path / "in.hx")
    with open(p, "w") as f:
        f.write(src)
    rc = main([p, "--check-only"])
    out = capsys.readouterr().out
    assert rc == 1, f"expected build to fail; rc={rc}; out={out!r}"
    assert "unsafe:" in out
    assert "28601" in out


def test_cli_unsafe_block_passes(capsys, tmp_path):
    """A2: `unsafe { *p; 0 }` (inside unsafe) passes check-only cleanly."""
    from helixc.check import main
    src = """
fn helper() -> i32 {
    let p: *const i32 = 0 as *const i32;
    unsafe { *p; 0 }
}
fn main() -> i32 { helper() }
"""
    p = str(tmp_path / "in.hx")
    with open(p, "w") as f:
        f.write(src)
    rc = main([p, "--check-only"])
    out = capsys.readouterr().out
    assert rc == 0, f"expected clean exit; rc={rc}; out={out!r}"
    assert "unsafe:" not in out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
