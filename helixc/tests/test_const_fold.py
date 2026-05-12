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


def test_fold_bitwise_and():
    # 250 & 42 = 42 (folded at compile time)
    mod = lower_and_fold("fn f() -> i32 { 250 & 42 }")
    assert count_ops(mod, tir.OpKind.BIT_AND) == 0
    consts = [op for fn in mod.functions.values() for blk in fn.blocks
              for op in blk.ops if op.kind == tir.OpKind.CONST_INT]
    values = [op.attrs["value"] for op in consts]
    assert 42 in values


def test_fold_bitwise_or():
    # 32 | 10 = 42
    mod = lower_and_fold("fn f() -> i32 { 32 | 10 }")
    assert count_ops(mod, tir.OpKind.BIT_OR) == 0


def test_fold_bitwise_xor():
    # 52 ^ 30 = 42
    mod = lower_and_fold("fn f() -> i32 { 52 ^ 30 }")
    assert count_ops(mod, tir.OpKind.BIT_XOR) == 0


def test_fold_shl():
    # 21 << 1 = 42
    mod = lower_and_fold("fn f() -> i32 { 21 << 1 }")
    assert count_ops(mod, tir.OpKind.SHL) == 0


def test_fold_shr_arithmetic():
    # 84 >> 1 = 42 ; (-1) >> 25 = -1 (sign-preserving)
    mod = lower_and_fold("fn f() -> i32 { 84 >> 1 }")
    assert count_ops(mod, tir.OpKind.SHR) == 0
    mod2 = lower_and_fold("fn f() -> i32 { (-1) >> 25 }")
    consts2 = [op.attrs["value"] for fn in mod2.functions.values()
               for blk in fn.blocks for op in blk.ops
               if op.kind == tir.OpKind.CONST_INT]
    # Python's `>>` on signed -1 stays -1 — matches x86 SAR semantics.
    assert -1 in consts2


def test_fold_bit_not():
    # ~5 = -6
    mod = lower_and_fold("fn f() -> i32 { ~5 }")
    assert count_ops(mod, tir.OpKind.BIT_NOT) == 0
    consts = [op.attrs["value"] for fn in mod.functions.values()
              for blk in fn.blocks for op in blk.ops
              if op.kind == tir.OpKind.CONST_INT]
    assert -6 in consts


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


def test_x_minus_zero_folds():
    mod = lower_and_fold("fn main() -> i32 { let x = 7; x - 0 }")
    subs = count_ops(mod, tir.OpKind.SUB)
    assert subs == 0, f"expected x-0 to fold, SUB count = {subs}"


def test_x_div_one_folds():
    """x / 1 should drop the DIV op entirely via SSA forwarding."""
    mod = lower_and_fold("fn main() -> i32 { let x = 7; x / 1 }")
    divs = count_ops(mod, tir.OpKind.DIV)
    assert divs == 0, f"expected x/1 to fold, DIV count = {divs}"


def test_x_mod_one_folds_to_zero():
    """x % 1 = 0 — the MOD should be replaced with const_int(0)."""
    mod = lower_and_fold("fn main() -> i32 { let x = 7; x % 1 }")
    mods = count_ops(mod, tir.OpKind.MOD)
    assert mods == 0, f"expected x%1 to fold, MOD count = {mods}"


def test_identity_forwarding_runs_correctly_across_blocks():
    """End-to-end check that identity forwarding preserves correctness
    when the identity op's result is used in a different block."""
    from helixc.tests.test_codegen import compile_and_run
    src = """
    fn main() -> i32 {
        let x = 21;
        let y = x * 1;
        if y > 0 { y * 2 } else { 0 }
    }
    """
    code = compile_and_run(src)
    assert code == 42, f"expected 42 (x*1=21, y*2=42), got {code}"


# --- Stage 17 regression tests ---

def test_stage17_2_plus_3_times_4_folds_to_14():
    """Stage 17 goal-test: `2 + 3 * 4` folds to const 14 at compile time.
    Verifies operator precedence is preserved by the fold and that the
    intermediate `3 * 4 = 12` is collapsed before the outer ADD."""
    mod = lower_and_fold("fn main() -> i32 { 2 + 3 * 4 }")
    # No ADD or MUL ops should remain — everything is folded.
    assert count_ops(mod, tir.OpKind.ADD) == 0, "ADD must fold"
    assert count_ops(mod, tir.OpKind.MUL) == 0, "MUL must fold"
    # The final const carries 14.
    consts = [op.attrs["value"]
              for fn in mod.functions.values() for blk in fn.blocks
              for op in blk.ops if op.kind == tir.OpKind.CONST_INT]
    assert 14 in consts, f"expected final const 14, got {consts}"


def test_stage17_nan_fold_traps_17001():
    """Stage 17 trap-id 17001: a fold that would bake a NaN literal into
    a CONST_FLOAT is rejected at compile time. We trigger this via
    (1e200 * 1e200) - (1e200 * 1e200) — each multiplication overflows
    to +inf in IEEE-754 f64, and inf - inf = NaN."""
    from helixc.ir.passes.const_fold import FoldError
    src = """
    fn main() -> f32 {
        let a = 1.0e200 * 1.0e200;
        let b = 1.0e200 * 1.0e200;
        a - b
    }
    """
    mod = lower(parse(src))
    try:
        fold_module(mod)
    except FoldError as e:
        assert FoldError.trap_id == 17001
        assert "17001" in str(e)
        return
    raise AssertionError("expected FoldError trap 17001")


def test_stage17_nan_fold_via_neg_traps_17001():
    """NEG of NaN is still NaN; the unary fold path must also trap 17001."""
    from helixc.ir.passes.const_fold import FoldError
    src = """
    fn main() -> f32 {
        let inf1 = 1.0e200 * 1.0e200;
        let inf2 = 1.0e200 * 1.0e200;
        let nan = inf1 - inf2;
        -nan
    }
    """
    mod = lower(parse(src))
    try:
        fold_module(mod)
    except FoldError as e:
        assert "17001" in str(e)
        return
    raise AssertionError("expected FoldError trap 17001 via NEG")


def test_stage19_shift_out_of_range_traps_17002():
    """Stage 28.9 cycle 21 audit-R C20-R1 regression test (HIGH).
    Cycle 19 added `ShiftFoldError` for out-of-range const shifts as
    'loud failure / trap 17002', but the raise sits inside a
    `try/except Exception: return None` block — so the trap was
    silently swallowed. The cycle 21 fix adds `except FoldError: raise`
    above the catch-all. This test exercises that the trap actually
    propagates."""
    from helixc.ir.passes.const_fold import ShiftFoldError, FoldError
    # Build the SHL directly in IR — `1 << 64` written in surface
    # Helix may get rejected upstream by the typecheck or otherwise
    # not survive into the fold pass intact.
    mod = tir.Module()
    i32 = tir.TIRScalar("i32")
    blk = tir.Block(id=0)
    v_l = tir.Value(id=0, ty=i32)
    v_r = tir.Value(id=1, ty=i32)
    v_s = tir.Value(id=2, ty=i32)
    blk.ops = [
        tir.Op(kind=tir.OpKind.CONST_INT, operands=[], results=[v_l],
               attrs={"value": 1}),
        tir.Op(kind=tir.OpKind.CONST_INT, operands=[], results=[v_r],
               attrs={"value": 64}),  # out of range [0, 63]
        tir.Op(kind=tir.OpKind.SHL, operands=[v_l, v_r], results=[v_s]),
        tir.Op(kind=tir.OpKind.RETURN, operands=[v_s], results=[]),
    ]
    fn = tir.FnIR(name="main", params=[], return_ty=i32, blocks=[blk])
    mod.functions["main"] = fn
    mod.next_value_id = 3
    mod.next_block_id = 1
    try:
        fold_module(mod)
    except ShiftFoldError as e:
        assert ShiftFoldError.trap_id == 17002
        assert "17002" in str(e)
        # Confirm it's also a FoldError subclass (the documented contract).
        assert isinstance(e, FoldError)
        return
    raise AssertionError(
        "expected ShiftFoldError trap 17002 for SHL with shift>=64 "
        "(cycle 21 C20-R1: was being silently swallowed by "
        "`except Exception` in _try_fold_op's bitwise/shift block)"
    )


def test_stage19_shr_out_of_range_traps_17002():
    """Same regression as test_stage19_shift_out_of_range_traps_17002
    but exercising SHR (line 435) instead of SHL (line 428)."""
    from helixc.ir.passes.const_fold import ShiftFoldError
    mod = tir.Module()
    i32 = tir.TIRScalar("i32")
    blk = tir.Block(id=0)
    v_l = tir.Value(id=0, ty=i32)
    v_r = tir.Value(id=1, ty=i32)
    v_s = tir.Value(id=2, ty=i32)
    blk.ops = [
        tir.Op(kind=tir.OpKind.CONST_INT, operands=[], results=[v_l],
               attrs={"value": 256}),
        tir.Op(kind=tir.OpKind.CONST_INT, operands=[], results=[v_r],
               attrs={"value": -1}),  # negative shift, out of range
        tir.Op(kind=tir.OpKind.SHR, operands=[v_l, v_r], results=[v_s]),
        tir.Op(kind=tir.OpKind.RETURN, operands=[v_s], results=[]),
    ]
    fn = tir.FnIR(name="main", params=[], return_ty=i32, blocks=[blk])
    mod.functions["main"] = fn
    mod.next_value_id = 3
    mod.next_block_id = 1
    try:
        fold_module(mod)
    except ShiftFoldError as e:
        assert "17002" in str(e)
        return
    raise AssertionError("expected ShiftFoldError trap 17002 for SHR shift<0")


def test_stage17_i32_overflow_wraps_two_complement():
    """Stage 17 spec: i32 fold must wrap on overflow (two's-complement).
    INT_MAX + 1 = -2147483648 in i32."""
    mod = lower_and_fold("fn main() -> i32 { 2147483647 + 1 }")
    consts = [op.attrs["value"]
              for fn in mod.functions.values() for blk in fn.blocks
              for op in blk.ops if op.kind == tir.OpKind.CONST_INT]
    assert -2147483648 in consts, \
        f"expected i32 overflow wrap to -2147483648, got {consts}"


def test_stage17_emits_mov_eax_14():
    """Stage 17 spec end-to-end: 2+3*4 is folded so the backend emits
    `mov eax, 14` (B8 0E 00 00 00) as the entry-stub exit code."""
    from helixc.backend.x86_64 import compile_module_to_elf
    mod = lower_and_fold("fn main() -> i32 { 2 + 3 * 4 }")
    elf = compile_module_to_elf(mod)
    # B8 imm32 = mov eax, imm32. 14 = 0x0E little-endian.
    target = bytes([0xB8, 0x0E, 0x00, 0x00, 0x00])
    assert target in elf, "expected `mov eax, 14` in emitted ELF"


def test_c85_1_shift_bound_uses_result_type_bitwidth():
    """Stage 28.9 cycle 86 audit-R C85-1 regression (HIGH conf 90):
    pre-fix the SHL/SHR const-fold bound was hard-coded `[0, 63]`
    regardless of result type. `1_i32 << 32_i32` const-folded to 0
    but x86 SHL masks cl to log2(bitwidth) — 5 bits for i32 — so the
    runtime computes `1_i32 << 0` = 1. Fold/runtime divergence is a
    silent miscompile across `-O0` vs default `-O1`. Now the bound
    is the result-type bitwidth: i32 → [0, 32), i64 → [0, 64)."""
    from helixc.ir.passes.const_fold import ShiftFoldError
    # Build SHL with i32 operands and shift=32 (in [32, 64) gap).
    mod = tir.Module()
    i32 = tir.TIRScalar("i32")
    blk = tir.Block(id=0)
    v_l = tir.Value(id=0, ty=i32)
    v_r = tir.Value(id=1, ty=i32)
    v_s = tir.Value(id=2, ty=i32)
    blk.ops = [
        tir.Op(kind=tir.OpKind.CONST_INT, operands=[], results=[v_l],
               attrs={"value": 1}),
        tir.Op(kind=tir.OpKind.CONST_INT, operands=[], results=[v_r],
               attrs={"value": 32}),  # in [32, 64): pre-fix silently folded
        tir.Op(kind=tir.OpKind.SHL, operands=[v_l, v_r], results=[v_s]),
        tir.Op(kind=tir.OpKind.RETURN, operands=[v_s], results=[]),
    ]
    fn = tir.FnIR(name="main", params=[], return_ty=i32, blocks=[blk])
    mod.functions["main"] = fn
    mod.next_value_id = 3
    mod.next_block_id = 1
    try:
        fold_module(mod)
    except ShiftFoldError as e:
        assert "32" in str(e)
        return
    raise AssertionError(
        "expected ShiftFoldError for i32 SHL shift=32 — pre-fix this "
        "was silently folded to 0 while x86 hardware computes 1"
    )


def test_c85_1_shift_i64_still_allows_up_to_63():
    """C85-1 regression: i64 shift bound remains [0, 64) — large
    shifts that pre-fix worked must still work."""
    mod = lower_and_fold("fn main() -> i64 { 1_i64 << 63_i64 }")
    consts = [op.attrs["value"]
              for fn in mod.functions.values() for blk in fn.blocks
              for op in blk.ops if op.kind == tir.OpKind.CONST_INT]
    # Expect the shifted constant to be present (with two's-complement wrap).
    # 1 << 63 = 0x8000000000000000 which wraps to -2^63 in signed i64.
    assert -(2**63) in consts or (2**63) in consts, (
        f"expected i64 1<<63 to fold; got consts={consts}"
    )


def test_c19_1_isize_usize_are_64_bit_in_wrap():
    """Audit 28.8 cycle 20 C19-1 (HIGH): isize/usize must be treated as
    64-bit in `_wrap_int_to_type`, matching the cycle-19 backend
    classifier fix (which canonicalizes isize→i64, usize→u64).

    Pre-fix `_INT_BITS["isize"] = 32` caused `_wrap_int_to_type(6e9,
    isize)` to mask to 32 bits → 1_705_032_704. Post-fix preserves
    full 64-bit precision so the folded path matches the un-folded
    path (optimization-stable)."""
    from helixc.ir.passes.const_fold import _wrap_int_to_type
    isize = tir.TIRScalar(name="isize")
    usize = tir.TIRScalar(name="usize")
    i64 = tir.TIRScalar(name="i64")
    u64 = tir.TIRScalar(name="u64")
    # 6_000_000_000 fits in signed 64-bit, should round-trip.
    assert _wrap_int_to_type(6_000_000_000, isize) == 6_000_000_000
    assert _wrap_int_to_type(6_000_000_000, i64) == 6_000_000_000
    # The isize and i64 wraps must agree (cycle-3 alias-canon).
    for v in [0, 1, -1, 2**31 - 1, 2**31, -(2**31), 6_000_000_000,
              -(6_000_000_000), 2**62, -(2**62)]:
        assert _wrap_int_to_type(v, isize) == _wrap_int_to_type(v, i64), (
            f"isize/i64 wrap disagreement at v={v}: "
            f"{_wrap_int_to_type(v, isize)} vs {_wrap_int_to_type(v, i64)}"
        )
    # Same for usize/u64.
    for v in [0, 1, 2**32, 2**32 - 1, 2**63, 2**63 + 1]:
        assert _wrap_int_to_type(v, usize) == _wrap_int_to_type(v, u64), (
            f"usize/u64 wrap disagreement at v={v}"
        )


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
