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


def test_x_times_zero_int_folds():
    # x * 0 = 0 for integers
    mod = lower_and_fold("fn main() -> i32 { let x = 17; x * 0 }")
    muls = count_ops(mod, tir.OpKind.MUL)
    assert muls == 0, f"expected MUL folded away, got {muls}"


def test_zero_times_x_int_folds():
    mod = lower_and_fold("fn main() -> i32 { let x = 17; 0 * x }")
    muls = count_ops(mod, tir.OpKind.MUL)
    assert muls == 0


def test_x_minus_x_int_folds():
    mod = lower_and_fold("fn main() -> i32 { let x = 17; x - x }")
    subs = count_ops(mod, tir.OpKind.SUB)
    assert subs == 0


def test_x_minus_x_float_NOT_folded_for_nan_safety():
    # NaN - NaN = NaN, not 0.0. The algebraic identity fold is restricted
    # to integers so this stays preserved at runtime. Using a function
    # parameter (not a literal) so const-fold can't fold via numerical
    # evaluation of the operands.
    mod = lower_and_fold("fn main(x: f32) -> f32 { x - x }")
    subs = count_ops(mod, tir.OpKind.SUB)
    assert subs == 1, f"float x-x must NOT fold (NaN-NaN=NaN); got {subs} SUBs"


def test_x_mod_one_folds_to_zero():
    mod = lower_and_fold("fn main() -> i32 { let x = 17; x % 1 }")
    mods = count_ops(mod, tir.OpKind.MOD)
    assert mods == 0


def test_self_int_compare_folds():
    # x == x (int) should fold to 1
    mod = lower_and_fold("""
    fn main() -> i32 {
        let x = 5;
        if x == x { 1 } else { 0 }
    }
    """)
    cmps = count_ops(mod, tir.OpKind.CMP_EQ)
    assert cmps == 0, f"expected CMP_EQ folded, got {cmps}"


def test_x_times_one_folds():
    """x * 1 should be forwarded to x — the MUL op disappears."""
    mod = lower_and_fold("fn main() -> i32 { let x = 7; x * 1 }")
    muls = count_ops(mod, tir.OpKind.MUL)
    assert muls == 0, f"expected x*1 to fold, MUL count = {muls}"


def test_one_times_x_folds():
    mod = lower_and_fold("fn main() -> i32 { let x = 7; 1 * x }")
    muls = count_ops(mod, tir.OpKind.MUL)
    assert muls == 0, f"expected 1*x to fold, MUL count = {muls}"


def test_x_plus_zero_folds():
    mod = lower_and_fold("fn main() -> i32 { let x = 7; x + 0 }")
    adds = count_ops(mod, tir.OpKind.ADD)
    assert adds == 0, f"expected x+0 to fold, ADD count = {adds}"


def test_zero_plus_x_folds():
    mod = lower_and_fold("fn main() -> i32 { let x = 7; 0 + x }")
    adds = count_ops(mod, tir.OpKind.ADD)
    assert adds == 0, f"expected 0+x to fold, ADD count = {adds}"


def test_x_div_one_folds():
    mod = lower_and_fold("fn main() -> i32 { let x = 7; x / 1 }")
    divs = count_ops(mod, tir.OpKind.DIV)
    assert divs == 0, f"expected x/1 to fold, DIV count = {divs}"


def test_x_minus_zero_folds():
    mod = lower_and_fold("fn main() -> i32 { let x = 7; x - 0 }")
    subs = count_ops(mod, tir.OpKind.SUB)
    assert subs == 0, f"expected x-0 to fold, SUB count = {subs}"


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
