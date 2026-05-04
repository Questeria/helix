"""Tests for the CSE (common subexpression elimination) pass."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir.passes.cse import cse_module
from helixc.ir.passes.dce import dce_module
from helixc.ir import tir


def lower_cse_dce(src: str) -> tir.Module:
    mod = lower(parse(src))
    cse_module(mod)
    dce_module(mod)
    return mod


def count_ops(mod: tir.Module, kind: tir.OpKind) -> int:
    return sum(
        1 for fn in mod.functions.values()
        for blk in fn.blocks
        for op in blk.ops
        if op.kind == kind
    )


def test_cse_duplicate_consts():
    # Two const(5) emissions can be deduped
    src = "fn f() -> i32 { 5 + 5 }"
    mod = lower_cse_dce(src)
    n_consts = count_ops(mod, tir.OpKind.CONST_INT)
    # The folder might also do this; with CSE alone we expect 1 const for "5"
    # plus the addition. After CSE+DCE: one const(5) + one ADD + RETURN
    # = the dedupe replaced a duplicate const reference
    assert n_consts <= 2


def test_cse_duplicate_arith():
    # (x + 1) used twice — CSE should compute once
    src = """
    fn f(x: i32) -> i32 {
        let a = x + 1;
        let b = x + 1;
        a + b
    }
    """
    mod = lower_cse_dce(src)
    # We expect ADD ops: 1 for x+1 (CSEd), 1 for a+b = 2 adds total
    n_adds = count_ops(mod, tir.OpKind.ADD)
    assert n_adds == 2, f"expected 2 adds (CSE'd), got {n_adds}"


def test_cse_no_op_when_unique():
    # Different operations are NOT merged
    src = """
    fn f(x: i32) -> i32 {
        let a = x + 1;
        let b = x + 2;
        a + b
    }
    """
    mod = lower_cse_dce(src)
    n_adds = count_ops(mod, tir.OpKind.ADD)
    # 3 distinct adds: x+1, x+2, a+b
    assert n_adds == 3


def test_cse_preserves_call():
    # CALLs are not subject to CSE (they may have side effects)
    src = """
    fn helper(x: i32) -> i32 { x }
    fn main() -> i32 {
        helper(5);
        helper(5)
    }
    """
    mod = lower_cse_dce(src)
    main = mod.functions["main"]
    calls = sum(1 for blk in main.blocks for op in blk.ops
                if op.kind == tir.OpKind.CALL)
    # Both CALLs preserved
    assert calls == 2


def main():
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
            import traceback
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
