"""Tests for helixc.ir.lower_ast (Tensor IR lowering)."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir import tir


def lower_src(src: str) -> tir.Module:
    return lower(parse(src))


def test_empty_function():
    mod = lower_src("fn nothing() {}")
    assert "nothing" in mod.functions
    fn = mod.functions["nothing"]
    assert len(fn.params) == 0
    assert isinstance(fn.return_ty, tir.TIRUnit)
    # Should end with a return op
    assert any(op.kind == tir.OpKind.RETURN for op in fn.entry.ops)


def test_arith_function():
    mod = lower_src("fn add(a: i32, b: i32) -> i32 { a + b }")
    fn = mod.functions["add"]
    assert len(fn.params) == 2
    assert isinstance(fn.return_ty, tir.TIRScalar)
    assert fn.return_ty.name == "i32"
    # Should have an ADD op
    assert any(op.kind == tir.OpKind.ADD for op in fn.entry.ops)


def test_constant_int():
    mod = lower_src("fn k() -> i32 { 42 }")
    fn = mod.functions["k"]
    consts = [op for op in fn.entry.ops if op.kind == tir.OpKind.CONST_INT]
    assert len(consts) == 1
    assert consts[0].attrs["value"] == 42


def test_constant_float():
    mod = lower_src("fn k() -> f32 { 3.14 }")
    fn = mod.functions["k"]
    consts = [op for op in fn.entry.ops if op.kind == tir.OpKind.CONST_FLOAT]
    assert len(consts) == 1
    assert abs(consts[0].attrs["value"] - 3.14) < 1e-6


def test_let_binding():
    mod = lower_src("fn f() -> i32 { let x = 7; x + x }")
    fn = mod.functions["f"]
    # Let binds x to a const(7); then x + x should reuse v_x twice in the ADD
    add_ops = [op for op in fn.entry.ops if op.kind == tir.OpKind.ADD]
    assert len(add_ops) == 1
    assert add_ops[0].operands[0] == add_ops[0].operands[1]


def test_call():
    src = """
    fn double(x: i32) -> i32 { x + x }
    fn main() -> i32 { double(5) }
    """
    mod = lower_src(src)
    main = mod.functions["main"]
    calls = [op for op in main.entry.ops if op.kind == tir.OpKind.CALL]
    assert len(calls) == 1
    assert calls[0].attrs["target"] == "double"


def test_nested_calls():
    src = """
    fn double(x: i32) -> i32 { x + x }
    fn add(a: i32, b: i32) -> i32 { a + b }
    fn main() -> i32 { add(double(3), 4) }
    """
    mod = lower_src(src)
    main = mod.functions["main"]
    calls = [op for op in main.entry.ops if op.kind == tir.OpKind.CALL]
    assert len(calls) == 2
    targets = [c.attrs["target"] for c in calls]
    assert "double" in targets
    assert "add" in targets


def test_tensor_type_lowered():
    src = """
    fn matmul[N: size, M: size, P: size](
        a: tensor<f32, [N, M]>,
        b: tensor<f32, [M, P]>,
    ) -> tensor<f32, [N, P]> {
        a
    }
    """
    mod = lower_src(src)
    fn = mod.functions["matmul"]
    # Param 0 should be a TensorTy with f32 dtype and 2 dim vars
    p0_ty = fn.params[0].ty
    assert isinstance(p0_ty, tir.TIRTensorTy)
    assert p0_ty.dtype.name == "f32"
    assert len(p0_ty.shape) == 2
    assert isinstance(p0_ty.shape[0], tir.DimVar) and p0_ty.shape[0].name == "N"


def test_tile_type_lowered():
    src = "fn k(x: tile<bf16, [16, 16], smem>) {}"
    mod = lower_src(src)
    fn = mod.functions["k"]
    p0_ty = fn.params[0].ty
    assert isinstance(p0_ty, tir.TIRTileTy)
    assert p0_ty.dtype.name == "bf16"
    assert p0_ty.memspace == "smem"


def test_kernel_attribute():
    src = "@kernel fn k() {}"
    mod = lower_src(src)
    fn = mod.functions["k"]
    assert fn.attrs.get("kernel") is True


def test_if_lowered_to_cfg():
    src = "fn f(b: bool) -> i32 { if b { 1 } else { 2 } }"
    mod = lower_src(src)
    fn = mod.functions["f"]
    # CFG-based lowering creates extra blocks (then/else/merge) and
    # emits cond_br + br ops
    cond_brs = [op for blk in fn.blocks for op in blk.ops
                if op.kind == tir.OpKind.COND_BR]
    brs = [op for blk in fn.blocks for op in blk.ops
           if op.kind == tir.OpKind.BR]
    assert len(cond_brs) == 1
    assert len(brs) >= 2  # one from each arm to merge
    # Merge block should have a single param for the if-result
    assert len(fn.blocks) >= 4  # entry + then + else + merge


def test_unary_neg():
    src = "fn f() -> i32 { -42 }"
    mod = lower_src(src)
    fn = mod.functions["f"]
    negs = [op for op in fn.entry.ops if op.kind == tir.OpKind.NEG]
    assert len(negs) == 1


def test_unique_value_ids():
    src = "fn f() -> i32 { 1 + 2 + 3 }"
    mod = lower_src(src)
    fn = mod.functions["f"]
    all_ids = []
    for op in fn.entry.ops:
        for r in op.results:
            all_ids.append(r.id)
    assert len(all_ids) == len(set(all_ids)), "all SSA value ids must be unique"


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
