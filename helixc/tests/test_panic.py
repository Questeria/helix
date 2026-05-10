"""Stage 28.5: tests for panic / abort policy + @unwind reservation."""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend import ast_nodes as A
from helixc.frontend.panic_pass import (
    collect_panics,
    validate_panic_args,
    find_unwind_attrs,
    validate_unwind,
    TRAP_PANIC_INVOKED,
    TRAP_UNWIND_NOT_SUPPORTED,
)


def test_collect_panics_basic():
    src = '''
fn die() -> i32 {
    panic("oh no");
    0
}
'''
    prog = parse(src)
    out = collect_panics(prog)
    assert len(out) == 1
    fn_name, span, msg = out[0]
    assert fn_name == "die"
    assert msg == "oh no"


def test_collect_panics_multiple():
    src = '''
fn die_a() -> i32 { panic("a"); 0 }
fn die_b() -> i32 { panic("b"); 0 }
fn ok() -> i32 { 42 }
'''
    prog = parse(src)
    out = collect_panics(prog)
    assert len(out) == 2
    names = {x[0] for x in out}
    msgs = {x[2] for x in out}
    assert names == {"die_a", "die_b"}
    assert msgs == {"a", "b"}


def test_collect_panics_none():
    src = "fn safe() -> i32 { 42 }"
    prog = parse(src)
    assert collect_panics(prog) == []


def test_validate_panic_args_clean():
    src = 'fn die() -> i32 { panic("ok"); 0 }'
    prog = parse(src)
    assert validate_panic_args(prog) == []


def test_validate_panic_args_zero_args():
    """panic() with no args should diag."""
    src = 'fn die() -> i32 { panic(); 0 }'
    prog = parse(src)
    diags = validate_panic_args(prog)
    assert diags
    assert "expected 1 arg" in diags[0]


def test_validate_panic_args_too_many():
    src = 'fn die() -> i32 { panic("a", "b"); 0 }'
    prog = parse(src)
    diags = validate_panic_args(prog)
    assert diags
    assert "got 2" in diags[0]


def test_validate_panic_args_non_string():
    """panic(42) — non-string arg."""
    src = 'fn die() -> i32 { panic(42); 0 }'
    prog = parse(src)
    diags = validate_panic_args(prog)
    assert diags
    assert "string literal" in diags[0]


def test_panic_not_unbound_name():
    """`panic` shouldn't fire 'unbound name' since it's in _BUILTIN_NAMES."""
    from helixc.frontend.typecheck import typecheck
    src = 'fn die() -> i32 { panic("x"); 0 }'
    prog = parse(src)
    errs = typecheck(prog)
    msgs = [str(e) for e in errs]
    assert not any("unbound name 'panic'" in m for m in msgs)


def test_find_unwind_attrs_clean():
    src = 'fn safe() -> i32 { 42 }'
    prog = parse(src)
    assert find_unwind_attrs(prog) == []


def test_find_unwind_attrs_present():
    src = '''
@unwind
fn risky() -> i32 { 42 }
'''
    prog = parse(src)
    out = find_unwind_attrs(prog)
    assert len(out) == 1
    assert out[0].name == "risky"


def test_validate_unwind_diag():
    src = '''
@unwind
fn risky() -> i32 { 42 }
'''
    prog = parse(src)
    diags = validate_unwind(prog)
    assert diags
    assert "trap 28502" in diags[0]


def test_validate_unwind_clean():
    src = 'fn safe() -> i32 { 42 }'
    prog = parse(src)
    assert validate_unwind(prog) == []


def test_trap_ids():
    assert TRAP_PANIC_INVOKED == 28501
    assert TRAP_UNWIND_NOT_SUPPORTED == 28502


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
