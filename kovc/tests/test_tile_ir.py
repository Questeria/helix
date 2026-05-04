"""Tests for kovc.ir.tile_ir (Tile IR + Tensor IR lowering)."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kovc.frontend.parser import parse
from kovc.ir.lower_ast import lower
from kovc.ir.tile_ir import lower_to_tile, TileOpKind, MemSpace


def lower_chain(src: str):
    prog = parse(src)
    tir_mod = lower(prog)
    tile_mod = lower_to_tile(tir_mod)
    return tile_mod


def test_empty_function():
    mod = lower_chain("fn nothing() {}")
    assert "nothing" in mod.functions


def test_arith_passes_through():
    mod = lower_chain("fn add(a: i32, b: i32) -> i32 { a + b }")
    fn = mod.functions["add"]
    kinds = [op.kind for op in fn.entry.ops]
    assert TileOpKind.SCALAR_ADD in kinds
    assert TileOpKind.RETURN in kinds


def test_cmp_carries_attr():
    mod = lower_chain("fn f() -> bool { 1 < 2 }")
    fn = mod.functions["f"]
    cmp_ops = [op for op in fn.entry.ops if op.kind == TileOpKind.SCALAR_CMP]
    assert len(cmp_ops) == 1
    assert cmp_ops[0].attrs.get("cmp") == "cmp.lt"


def test_if_lowered_to_cfg_in_tile_ir():
    mod = lower_chain("fn f(b: bool) -> i32 { if b { 1 } else { 2 } }")
    fn = mod.functions["f"]
    # The Tile IR maps Tensor IR's COND_BR/BR opaquely (via CALL fallback in
    # v0.1). What matters for now is that the function has multiple blocks.
    assert len(fn.blocks) >= 4


def test_call_lowered():
    src = """
    fn double(x: i32) -> i32 { x + x }
    fn main() -> i32 { double(7) }
    """
    mod = lower_chain(src)
    main = mod.functions["main"]
    calls = [op for op in main.entry.ops if op.kind == TileOpKind.CALL]
    assert len(calls) >= 1


def test_value_id_consistency():
    mod = lower_chain("fn f() -> i32 { 1 + 2 + 3 }")
    fn = mod.functions["f"]
    ids = []
    for op in fn.entry.ops:
        for r in op.results:
            ids.append(r.id)
    assert len(ids) == len(set(ids)), "tile values must have unique ids"


def test_function_attrs_carried():
    mod = lower_chain("@kernel fn k() {}")
    fn = mod.functions["k"]
    assert fn.attrs.get("kernel") is True


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
