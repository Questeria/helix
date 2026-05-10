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


# C1-H1 / A6 regression tests for if-arm / for-iter walker coverage are
# defined further down in this module (after the validation-arg group).


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


# ----------------------------------------------------------------------
# Audit-28.8 C1-H1 / A6 regressions: walker must descend into if-branches
# and for-loop iterables. Pre-fix the walker used "then_branch" /
# "else_branch" attr names that don't exist on A.If — every panic inside
# an if-arm was silently invisible. Same gap for For.iter_expr.
# ----------------------------------------------------------------------
def test_collect_panics_inside_if_then():
    """C1-H1: panic() inside if.then must be reported."""
    src = '''
fn die_if(flag: i32) -> i32 {
    if flag > 0 {
        panic("zero is bad");
        0
    } else {
        1
    }
}
'''
    prog = parse(src)
    out = collect_panics(prog)
    assert len(out) == 1
    fn_name, _span, msg = out[0]
    assert fn_name == "die_if"
    assert msg == "zero is bad"


def test_collect_panics_inside_if_else():
    """C1-H1: panic() inside if.else_ must be reported."""
    src = '''
fn die_else(flag: i32) -> i32 {
    if flag > 0 {
        0
    } else {
        panic("else path");
        1
    }
}
'''
    prog = parse(src)
    out = collect_panics(prog)
    assert len(out) == 1
    fn_name, _span, msg = out[0]
    assert fn_name == "die_else"
    assert msg == "else path"


def test_validate_panic_args_inside_if_branch():
    """C1-H1: validate_panic_args must surface non-string args
    inside if-arms (pre-fix it returned []; post-fix it emits a diag)."""
    src = '''
fn check_if(flag: i32) -> i32 {
    if flag > 0 {
        panic(42);
        0
    } else {
        1
    }
}
'''
    prog = parse(src)
    diags = validate_panic_args(prog)
    assert diags, "expected at least one diagnostic for panic(42) in if.then"
    assert "string literal" in diags[0]


def test_collect_panics_inside_for_body():
    """A6: panic() inside a For.body must be reported via the body recursion
    (For has `body` which the walker already handled — but the regression
    here is that the prior walker had no `iter_expr` attribute so panics
    in range bounds were missed; the body case verifies the loop still
    works end-to-end after the walker rewrite)."""
    src = '''
fn die_for(n: i32) -> i32 {
    for i in 0..n {
        panic("loop");
    }
    0
}
'''
    prog = parse(src)
    out = collect_panics(prog)
    assert len(out) == 1
    fn_name, _span, msg = out[0]
    assert fn_name == "die_for"
    assert msg == "loop"


def test_walker_does_not_callback_on_stmts():
    """C1-L1: walker invokes callback only on `A.Expr` subclasses, not
    on `A.Stmt` subclasses (Let, ExprStmt, ConstStmt). Note that `A.Block`
    is itself an `A.Expr` subclass in this AST schema, so the walker DOES
    fire on Blocks — what the fix prevents is firing on Stmt-tagged nodes."""
    src = '''
fn body() -> i32 {
    let x: i32 = 0;
    x
}
'''
    prog = parse(src)
    seen: list = []

    def cb(e):
        seen.append(e)

    from helixc.frontend.panic_pass import _walk_exprs
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    _walk_exprs(fn.body, cb)
    # None of the seen nodes should be an A.Stmt subclass.
    for n in seen:
        assert isinstance(n, A.Expr), (
            f"_walk_exprs callback fired on {type(n).__name__!r}, which "
            f"is not an A.Expr subclass; only Exprs are expected"
        )
    # At least one Expr should have fired (the Block, the `0` IntLit, the
    # `x` Name, etc.).
    assert seen, "callback should have fired on at least one Expr"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
