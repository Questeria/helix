"""Tests for helixc.backend.ptx (PTX emission)."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir.tile_ir import lower_to_tile
from helixc.backend.ptx import emit_ptx


def emit(src: str) -> str:
    return emit_ptx(lower_to_tile(lower(parse(src))))


def test_module_header():
    out = emit("@kernel fn k() {}")
    assert ".version" in out
    assert ".target sm_75" in out
    assert ".address_size 64" in out


def test_kernel_directive():
    out = emit("@kernel fn my_kernel() {}")
    assert ".visible .entry my_kernel" in out
    assert "{" in out and "}" in out


def test_kernel_has_register_declarations():
    out = emit("@kernel fn k() {}")
    assert ".reg .pred" in out
    assert ".reg .b32" in out
    assert ".reg .f32" in out


def test_kernel_ret():
    out = emit("@kernel fn k() {}")
    # Every kernel must end with ret;
    assert "ret;" in out


def test_scalar_const_int():
    src = "@kernel fn k() { let x = 42; }"
    out = emit(src)
    assert "mov.b32" in out
    assert "42" in out


def test_scalar_add():
    src = "@kernel fn k() { let x = 1; let y = 2; let z = x + y; }"
    out = emit(src)
    assert "add.s32" in out


def test_scalar_mul():
    src = "@kernel fn k() { let z = 3 * 4; }"
    out = emit(src)
    assert "mul.lo.s32" in out


def test_non_kernel_emits_func():
    src = """
    fn helper() -> i32 { 42 }
    @kernel fn k() {}
    """
    out = emit(src)
    assert ".func" in out
    assert ".visible .entry k" in out


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
