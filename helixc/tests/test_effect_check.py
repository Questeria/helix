"""Tests for IR-level effect verification."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.ir.lower_ast import lower
from helixc.ir.passes.effect_check import (
    check_module, verify_module, EffectError, compute_closure,
    own_op_effects, declared_effects, is_pure_decl,
)
from helixc.ir import tir


def lower_only(src: str) -> tir.Module:
    return lower(parse(src))


def test_pure_function_with_no_effects_passes():
    src = """
    @pure fn add(a: i32, b: i32) -> i32 { a + b }
    fn main() -> i32 { add(20, 22) }
    """
    mod = lower_only(src)
    errs = check_module(mod)
    assert errs == [], f"unexpected: {errs}"


def test_pure_function_calling_pure_function_passes():
    src = """
    @pure fn double(x: i32) -> i32 { x + x }
    @pure fn quadruple(x: i32) -> i32 { double(double(x)) }
    fn main() -> i32 { quadruple(10) }
    """
    mod = lower_only(src)
    assert check_module(mod) == []


def test_pure_function_using_print_fails():
    # print_int is a PRINT op which has the "io" effect. A @pure fn can't have it.
    src = """
    @pure fn shout(x: i32) -> i32 {
        print_int(x);
        x
    }
    fn main() -> i32 { shout(5) }
    """
    mod = lower_only(src)
    errs = check_module(mod)
    assert any("@pure" in e and "shout" in e for e in errs), f"got {errs}"


def test_pure_function_calling_impure_transitively_fails():
    # @pure A → B → io. Should be caught: A's closure includes io.
    src = """
    fn impure_helper() -> i32 {
        print_int(7);
        7
    }
    @pure fn caller() -> i32 {
        impure_helper()
    }
    fn main() -> i32 { caller() }
    """
    mod = lower_only(src)
    errs = check_module(mod)
    assert any("@pure" in e and "caller" in e for e in errs), f"got {errs}"


def test_unknown_callee_treated_as_unknown_effect():
    # Build a module whose CALL targets an undeclared function name — the
    # checker should flag the caller as having effect "unknown" (which will
    # only be allowed if explicitly declared).
    mod = tir.Module()
    i32 = tir.TIRScalar("i32")
    blk = tir.Block(id=0)
    v = tir.Value(id=0, ty=i32)
    blk.ops = [
        tir.Op(kind=tir.OpKind.CALL, operands=[], results=[v],
               attrs={"target": "extern_unknown"}),
        tir.Op(kind=tir.OpKind.RETURN, operands=[v], results=[]),
    ]
    fn = tir.FnIR(name="caller", params=[], return_ty=i32, blocks=[blk],
                  attrs={"is_pure": True})
    mod.functions["caller"] = fn
    mod.next_value_id = 1
    mod.next_block_id = 1

    errs = check_module(mod)
    assert any("caller" in e and "unknown" in e for e in errs), f"got {errs}"


def test_verify_module_raises_on_violation():
    src = """
    @pure fn bad() -> i32 {
        print_int(1);
        0
    }
    fn main() -> i32 { bad() }
    """
    mod = lower_only(src)
    try:
        verify_module(mod)
    except EffectError:
        return
    raise AssertionError("expected EffectError")


def test_verify_module_passes_on_clean_module():
    src = """
    @pure fn add(a: i32, b: i32) -> i32 { a + b }
    fn main() -> i32 { add(1, 2) }
    """
    mod = lower_only(src)
    verify_module(mod)  # must not raise


def test_recursive_pure_function_has_empty_closure():
    src = """
    @pure fn fact(n: i32) -> i32 {
        if n <= 1 { 1 } else { n * fact(n - 1) }
    }
    fn main() -> i32 { fact(5) }
    """
    mod = lower_only(src)
    closure = compute_closure(mod)
    assert closure["fact"] == frozenset(), f"got {closure['fact']}"


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
