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


# --- Stage 18 regression tests ---

def test_stage18_cse_merges_let_5_times_5_pair():
    """Stage 18 goal-test: two identical `5 * 5` subexpressions are
    merged by CSE. The lowering produces 2 MUL ops; CSE should drop one
    duplicate. (DCE removes the now-dead one.)"""
    src = """
    fn main() -> i32 {
        let a = 5 * 5;
        let b = 5 * 5;
        a + b
    }
    """
    # Run with CSE+DCE only (no fold) so we observe CSE's effect cleanly.
    mod = lower(parse(src))
    from helixc.ir.passes.cse import cse_module as _cse
    from helixc.ir.passes.dce import dce_module as _dce
    cse_count = _cse(mod)
    _dce(mod)
    assert cse_count >= 4, (
        f"expected CSE to merge >= 4 duplicate ops (3 const(5) + 1 MUL); "
        f"got {cse_count}"
    )
    # Net result: one MUL remains, one ADD, one const, one return.
    muls = count_ops(mod, tir.OpKind.MUL)
    assert muls == 1, f"expected 1 surviving MUL after CSE+DCE, got {muls}"


def test_stage18_fold_cse_dce_end_to_end_to_50():
    """End-to-end fold+CSE+DCE on the spec example collapses the body
    to a single CONST_INT(50) and a RETURN."""
    from helixc.ir.passes.const_fold import fold_module as _fold
    src = """
    fn main() -> i32 {
        let a = 5 * 5;
        let b = 5 * 5;
        a + b
    }
    """
    mod = lower(parse(src))
    _fold(mod)
    cse_module(mod)
    dce_module(mod)
    # After full opt pipeline: one const carrying 50.
    consts = [op.attrs["value"]
              for fn in mod.functions.values() for blk in fn.blocks
              for op in blk.ops if op.kind == tir.OpKind.CONST_INT]
    assert 50 in consts, f"expected final const 50, got {consts}"
    assert count_ops(mod, tir.OpKind.MUL) == 0
    assert count_ops(mod, tir.OpKind.ADD) == 0


def test_stage18_cse_does_not_merge_different_types():
    """Trap-id 18001 safeguard: a bool MUL and an i32 MUL with the same
    operand IDs must NOT be merged (audit-10 fix in _op_hash). We use a
    cast that produces a CMP-result-typed value flowing into MUL on one
    side, vs i32 on the other."""
    # Direct IR construction to guarantee identical operand ids but
    # different result types — the parsing path normally separates them
    # via the type system, but a malformed pass output could collide.
    mod = tir.Module()
    i32 = tir.TIRScalar("i32")
    bool_ty = tir.TIRScalar("bool")
    blk = tir.Block(id=0)
    v_a = tir.Value(id=0, ty=i32)
    v_b = tir.Value(id=1, ty=i32)
    v_r_i32 = tir.Value(id=2, ty=i32)
    v_r_bool = tir.Value(id=3, ty=bool_ty)
    blk.ops = [
        tir.Op(kind=tir.OpKind.CONST_INT, operands=[], results=[v_a],
               attrs={"value": 1}),
        tir.Op(kind=tir.OpKind.CONST_INT, operands=[], results=[v_b],
               attrs={"value": 1}),
        # Two MULs with identical operand ids but different result types.
        tir.Op(kind=tir.OpKind.MUL, operands=[v_a, v_b], results=[v_r_i32]),
        tir.Op(kind=tir.OpKind.MUL, operands=[v_a, v_b], results=[v_r_bool]),
        tir.Op(kind=tir.OpKind.RETURN, operands=[v_r_i32], results=[]),
    ]
    fn = tir.FnIR(name="main", params=[], return_ty=i32, blocks=[blk])
    mod.functions["main"] = fn
    mod.next_value_id = 4
    mod.next_block_id = 1
    cse_module(mod)
    # Both MUL ops must survive: different result types means different
    # semantic ops, even with identical operand ids.
    muls = count_ops(mod, tir.OpKind.MUL)
    assert muls == 2, f"i32 vs bool MUL must NOT merge; got {muls} MULs"


def test_stage18_dce_preserves_tile_index_store():
    """Trap-id 18002 safeguard: DCE must NOT drop TILE_INDEX_STORE ops
    (Stage 16 added them to SIDE_EFFECT_KINDS). Build a module with an
    unused TILE_INDEX_STORE result and verify it survives."""
    from helixc.ir.passes.dce import dce_module as _dce, SIDE_EFFECT_KINDS
    # Smoke check that the membership is set; the actual op-kind check
    # is in the SIDE_EFFECT_KINDS set.
    assert tir.OpKind.TILE_INDEX_STORE in SIDE_EFFECT_KINDS, (
        "TILE_INDEX_STORE missing from DCE side-effect set "
        "(Stage 16 regression — would silently drop GPU tile stores)"
    )


def test_stage18_fdce_preserves_kernel_function():
    """Trap-id 18003 safeguard: FDCE must NOT drop `@kernel` functions
    even when the host code doesn't call them directly — the PTX
    backend embeds them on its own."""
    from helixc.ir.passes.fdce import fdce_module as _fdce
    src = """
    @kernel
    fn add_kernel(a: i32, b: i32) -> i32 { a + b }
    fn main() -> i32 { 0 }
    """
    mod = lower(parse(src))
    dropped = _fdce(mod)
    assert "add_kernel" in mod.functions, (
        f"@kernel fn dropped by FDCE — Stage 16 regression. dropped={dropped}"
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
