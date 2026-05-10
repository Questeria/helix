"""Stage 25: tests for trace-based introspection (@trace + TraceBuffer)."""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend import ast_nodes as A
from helixc.frontend.trace_pass import (
    TraceBuffer,
    TraceEvent,
    trace_equiv,
    is_traced,
    traced_fn_names,
    validate_trace_attrs,
    TRAP_TRACE_OVERFLOW,
    TRAP_TRACE_EQUIV_SHAPE_MISMATCH,
    DEFAULT_TRACE_CAP,
)


def test_parse_trace_attribute():
    """A fn marked `@trace` carries 'trace' in its attrs list."""
    src = """
@trace
fn f(x: i32) -> i32 { x + 1 }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl) and it.name == "f")
    assert "trace" in fn.attrs
    assert is_traced(fn)


def test_trace_attribute_combines():
    """@trace coexists with @pure / @inline."""
    src = """
@trace
@pure
fn f(x: i32) -> i32 { x + 1 }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl) and it.name == "f")
    assert "trace" in fn.attrs
    assert "pure" in fn.attrs


def test_no_trace_attribute():
    """A plain fn is not traced."""
    src = """
fn g(y: i32) -> i32 { y * 2 }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl) and it.name == "g")
    assert not is_traced(fn)


def test_traced_fn_names():
    src = """
@trace fn a(x: i32) -> i32 { x }
fn b(x: i32) -> i32 { x }
@trace fn c(x: i32) -> i32 { x }
"""
    prog = parse(src)
    names = traced_fn_names(prog)
    assert names == ["a", "c"]


def test_trace_buffer_push_and_len():
    tb = TraceBuffer()
    assert len(tb) == 0
    tb.push(TraceEvent(op_kind="entry", fn_name="f", operands=(1, 2)))
    tb.push(TraceEvent(op_kind="exit", fn_name="f", operands=(3,)))
    assert len(tb) == 2


def test_trace_buffer_overflow_raises():
    """Past cap, push raises OverflowError (= trap 25001)."""
    tb = TraceBuffer(cap=2)
    tb.push(TraceEvent("entry", "f", ()))
    tb.push(TraceEvent("exit", "f", ()))
    with pytest.raises(OverflowError):
        tb.push(TraceEvent("op", "g", ()))


def test_trace_buffer_clear():
    tb = TraceBuffer()
    tb.push(TraceEvent("entry", "f", ()))
    assert len(tb) == 1
    tb.clear()
    assert len(tb) == 0


def test_trace_equiv_identical():
    a = TraceBuffer()
    b = TraceBuffer()
    for ev in (
        TraceEvent("entry", "f", (1, 2)),
        TraceEvent("op", "add", (1, 2), result=3),
        TraceEvent("exit", "f", (3,)),
    ):
        a.push(ev)
        b.push(ev)
    assert trace_equiv(a, b)


def test_trace_equiv_different_length():
    a = TraceBuffer()
    b = TraceBuffer()
    a.push(TraceEvent("entry", "f", ()))
    assert not trace_equiv(a, b)


def test_trace_equiv_different_ops():
    a = TraceBuffer()
    b = TraceBuffer()
    a.push(TraceEvent("entry", "f", (1,)))
    b.push(TraceEvent("entry", "f", (2,)))  # different operands
    assert not trace_equiv(a, b)


def test_trap_25001_reserved():
    """Document the trap-id reservations."""
    assert TRAP_TRACE_OVERFLOW == 25001
    assert TRAP_TRACE_EQUIV_SHAPE_MISMATCH == 25002


def test_default_cap_is_4kb():
    """Phase-0 default trace cap is 4096 entries."""
    assert DEFAULT_TRACE_CAP == 4096


def test_validate_trace_on_extern_rejected():
    """@trace on an extern \"C\" fn must produce a diagnostic."""
    src = '''
@trace
extern "C" fn malloc(n: u64) -> u64;
'''
    prog = parse(src)
    diags = validate_trace_attrs(prog)
    assert diags
    assert "malloc" in diags[0]
    assert "extern" in diags[0]


def test_validate_trace_clean():
    src = """
@trace fn f(x: i32) -> i32 { x }
"""
    prog = parse(src)
    diags = validate_trace_attrs(prog)
    assert diags == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
