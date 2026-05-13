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
    """`*p` inside unsafe should NOT trigger 28601.

    Audit 28.8 B3: the setup `let p: *const i32 = 0 as *const i32;`
    is itself a raw-pointer cast now, so the let-binding is wrapped
    in an unsafe block to keep the test focused on the deref check."""
    src = """
fn main() -> i32 {
    unsafe {
        let p: *const i32 = 0 as *const i32;
        *p
    }
}
"""
    prog = parse(src)
    diags = check_unsafe_ops(prog)
    assert diags == []


def test_raw_ptr_op_outside_unsafe_diag():
    """`*p` outside any unsafe block should trigger 28601 diag.

    Audit 28.8 B3: the cast is also now a raw-pointer op, but
    check_unsafe_ops returns 28601 for ALL ops outside unsafe — the
    cast contributes its own diagnostic now too. We just assert that
    at least one 28601 fires (the deref or the cast)."""
    src = """
fn main() -> i32 {
    let p: *const i32 = 0 as *const i32;
    *p
}
"""
    prog = parse(src)
    diags = check_unsafe_ops(prog)
    assert diags
    assert any("28601" in d for d in diags)


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
    """Audit 28.8 B3: Cast targeting *T counts as a raw-pointer op
    too, so we now also see the cast in the ops list. The original
    intent — "context tracking works: inside-unsafe vs outside" — is
    preserved; we just have one extra op in the count."""
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
    # 3 raw-ptr ops total: 1 cast outside + 1 deref inside + 1 deref outside.
    assert len(ops) == 3
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


def test_unsafe_block_with_deref_lowering_fails_closed():
    """`unsafe { *p }` typechecks, but lowering must not pretend that
    dereference is a no-op until real pointer-load IR exists."""
    from helixc.ir.lower_ast import lower
    src = """
fn main() -> i32 {
    let p: *const i32 = unsafe { 0 as *const i32 };
    unsafe { *p }
}
"""
    prog = parse(src)
    with pytest.raises(NotImplementedError, match="dereference lowering"):
        lower(prog)


def test_address_of_lowering_fails_closed():
    """`&x` has a check-only reference type, but lowering is blocked
    until references have explicit storage/address semantics."""
    from helixc.ir.lower_ast import lower
    src = """
fn main() -> i32 {
    let x: i32 = 7;
    &x
}
"""
    prog = parse(src)
    with pytest.raises(NotImplementedError, match="address-of lowering"):
        lower(prog)


def test_cli_check_unsafe_ops_wired(capsys, tmp_path):
    """A2: helixc/check.py invokes check_unsafe_ops directly.

    Audit 28.8 B3: the typecheck now also blocks `0 as *const i32`
    outside unsafe with trap 28603 — so we wrap the cast in an
    unsafe block to keep this test focused on trap 28601 for raw
    deref outside unsafe. Stage 31 unary typing can now surface the
    same trap before the separate unsafe_pass banner is printed."""
    from helixc.check import main
    # Wrap the cast in unsafe so typecheck passes; the bare `*p;` is
    # what we want check_unsafe_ops to flag with 28601.
    src = """
fn helper() -> i32 {
    let p: *const i32 = unsafe { 0 as *const i32 };
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
    assert "28601" in out


def test_cli_unsafe_block_deref_fails_until_pointer_load_ir(capsys, tmp_path):
    """A2: `unsafe { *p; 0 }` clears the unsafe-boundary gate, but
    check-only still fails until pointer-load IR exists.

    Audit 28.8 B3: also wrap the cast so typecheck doesn't fire 28603."""
    from helixc.check import main
    src = """
fn helper() -> i32 {
    let p: *const i32 = unsafe { 0 as *const i32 };
    unsafe { *p; 0 }
}
fn main() -> i32 { helper() }
"""
    p = str(tmp_path / "in.hx")
    with open(p, "w") as f:
        f.write(src)
    rc = main([p, "--check-only"])
    out = capsys.readouterr().out
    assert rc == 1, f"expected unsupported deref failure; rc={rc}; out={out!r}"
    assert "unsafe:" not in out
    assert "raw-pointer dereference is type-known but not lowerable yet" in out


# ----------------------------------------------------------------------
# Audit 28.8 B3 — Cast (int as *T / float as *T) gate
# ----------------------------------------------------------------------
def test_b3_cast_int_to_ptr_outside_unsafe_blocks():
    """Audit 28.8 B3: `x as *mut T` outside `unsafe { ... }` is a
    forged pointer; typecheck must emit trap 28603."""
    from helixc.frontend.typecheck import typecheck
    src = """
fn main() -> i32 {
    let p: *mut i32 = 0 as *mut i32;
    0
}
"""
    prog = parse(src)
    errs = typecheck(prog)
    assert any("28603" in str(e) for e in errs), \
        f"expected trap 28603 for int→ptr cast outside unsafe, " \
        f"got: {[str(e) for e in errs]}"


def test_b3_cast_int_to_ptr_inside_unsafe_ok():
    """Audit 28.8 B3: same cast wrapped in unsafe is accepted."""
    from helixc.frontend.typecheck import typecheck
    src = """
fn main() -> i32 {
    let p: *mut i32 = unsafe { 0 as *mut i32 };
    0
}
"""
    prog = parse(src)
    errs = typecheck(prog)
    assert not any("28603" in str(e) for e in errs), \
        f"unexpected 28603 inside unsafe: {[str(e) for e in errs]}"


def test_b3_cast_float_to_ptr_blocked_even_in_unsafe():
    """Audit 28.8 B3: float→ptr is rejected unconditionally — there's
    no well-defined coercion. Even inside unsafe."""
    from helixc.frontend.typecheck import typecheck
    src = """
fn main() -> i32 {
    let pi: f64 = 3.14_f64;
    let p: *mut i32 = unsafe { pi as *mut i32 };
    0
}
"""
    prog = parse(src)
    errs = typecheck(prog)
    assert any("28603" in str(e) for e in errs), \
        f"expected 28603 for float→ptr even inside unsafe, " \
        f"got: {[str(e) for e in errs]}"


def test_b3_ptr_to_ptr_cast_ok_outside_unsafe():
    """Audit 28.8 B3: pointer-to-pointer cast (`*const T` → `*mut U`)
    is NOT a forged-pointer scenario — source already has pointer
    semantics. Accepted outside unsafe."""
    from helixc.frontend.typecheck import typecheck
    src = """
fn main() -> i32 {
    let p: *const i32 = unsafe { 0 as *const i32 };
    let q: *mut i32 = p as *mut i32;
    0
}
"""
    prog = parse(src)
    errs = typecheck(prog)
    assert not any("28603" in str(e) for e in errs), \
        f"unexpected 28603 for ptr→ptr cast: {[str(e) for e in errs]}"


def test_b3_unsafe_pass_walker_sees_cast():
    """Audit 28.8 B3: unsafe_pass._is_raw_ptr_op now also matches Cast
    nodes targeting TyPtr — previously only Unary deref was matched,
    so out-of-unsafe ptr-casts escaped check_unsafe_ops entirely."""
    from helixc.frontend.unsafe_pass import find_raw_ptr_ops
    src = """
fn main() -> i32 {
    let p: *const i32 = 0 as *const i32;
    0
}
"""
    prog = parse(src)
    ops = find_raw_ptr_ops(prog)
    # The Cast is the only raw-ptr op in this program (no deref).
    assert any(span.line == 3 for (_, span, _) in ops), \
        f"expected Cast to register as raw-ptr op, got: {ops}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
