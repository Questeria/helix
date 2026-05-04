"""Tests for the constant-folding IR optimization pass."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir.passes.const_fold import fold_module
from helixc.ir import tir


def lower_and_fold(src: str) -> tir.Module:
    mod = lower(parse(src))
    fold_module(mod)
    return mod


def count_ops(mod: tir.Module, kind: tir.OpKind) -> int:
    return sum(
        1 for fn in mod.functions.values()
        for blk in fn.blocks
        for op in blk.ops
        if op.kind == kind
    )


def test_fold_int_addition():
    mod = lower_and_fold("fn f() -> i32 { 2 + 3 }")
    # ADD should be gone; only one CONST_INT remains as the fn's value
    adds = count_ops(mod, tir.OpKind.ADD)
    assert adds == 0
    consts = count_ops(mod, tir.OpKind.CONST_INT)
    # Original 2 consts collapsed; 1 const remains plus possibly intermediate
    # constants from old defs that are now dead. We just assert the ADD is gone.


def test_fold_chain():
    # 1 + 2 + 3 should fold to 6
    mod = lower_and_fold("fn f() -> i32 { 1 + 2 + 3 }")
    adds = count_ops(mod, tir.OpKind.ADD)
    assert adds == 0


def test_fold_mul():
    mod = lower_and_fold("fn f() -> i32 { 6 * 7 }")
    muls = count_ops(mod, tir.OpKind.MUL)
    assert muls == 0


def test_fold_sub():
    mod = lower_and_fold("fn f() -> i32 { 100 - 58 }")
    subs = count_ops(mod, tir.OpKind.SUB)
    assert subs == 0


def test_fold_neg():
    mod = lower_and_fold("fn f() -> i32 { -42 }")
    negs = count_ops(mod, tir.OpKind.NEG)
    assert negs == 0


def test_fold_doesnt_touch_runtime_values():
    # Values from function args can't be folded
    mod = lower_and_fold("fn f(x: i32) -> i32 { x + 5 }")
    adds = count_ops(mod, tir.OpKind.ADD)
    assert adds == 1  # cannot fold (x is runtime)


def test_fold_float_arith():
    mod = lower_and_fold("fn f() -> f32 { 2.0 + 3.0 }")
    adds = count_ops(mod, tir.OpKind.ADD)
    assert adds == 0


def test_fold_comparison_true():
    mod = lower_and_fold("fn f() -> i32 { let b = 5 < 10; b + 41 }")
    cmps = count_ops(mod, tir.OpKind.CMP_LT)
    assert cmps == 0


def test_fold_does_not_break_running_program():
    # End-to-end: program should still run correctly after folding.
    # We don't run it here (that needs WSL), just verify the fold doesn't
    # produce garbage. The full codegen tests run the binaries and would
    # catch any breakage.
    mod = lower_and_fold("fn main() -> i32 { 17 + 25 }")
    # At minimum, the module still has a main function with at least one op
    assert "main" in mod.functions
    assert any(op.kind == tir.OpKind.RETURN
               for blk in mod.functions["main"].blocks
               for op in blk.ops)


def test_fold_div_negative_dividend():
    # C semantics: -7 / 2 = -3 (truncation toward zero), NOT -4 (Python //)
    mod = lower_and_fold("fn main() -> i32 { -7 / 2 }")
    consts = [op for fn in mod.functions.values() for blk in fn.blocks
              for op in blk.ops if op.kind == tir.OpKind.CONST_INT]
    # Find the const that was the result of the division: it should be -3
    values = [op.attrs["value"] for op in consts]
    assert -3 in values, f"expected fold to produce -3 (C semantics), got {values}"


def test_fold_div_negative_divisor():
    # C semantics: 7 / -2 = -3
    mod = lower_and_fold("fn main() -> i32 { 7 / (-2) }")
    consts = [op for fn in mod.functions.values() for blk in fn.blocks
              for op in blk.ops if op.kind == tir.OpKind.CONST_INT]
    values = [op.attrs["value"] for op in consts]
    assert -3 in values, f"expected fold to produce -3 (C semantics), got {values}"


def test_fold_mod_negative_dividend():
    # C semantics: -7 % 2 = -1 (sign of dividend), NOT 1 (Python %)
    mod = lower_and_fold("fn main() -> i32 { -7 % 2 }")
    consts = [op for fn in mod.functions.values() for blk in fn.blocks
              for op in blk.ops if op.kind == tir.OpKind.CONST_INT]
    values = [op.attrs["value"] for op in consts]
    assert -1 in values, f"expected fold to produce -1 (C semantics), got {values}"


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
