"""Tests for module-level (function) dead-code elimination."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend.grad_pass import grad_pass
from helixc.ir.lower_ast import lower
from helixc.ir.passes.fdce import fdce_module


def lower_and_fdce(src: str) -> tuple[set[str], int]:
    prog = parse(src)
    grad_pass(prog)
    mod = lower(prog)
    n_dropped = fdce_module(mod)
    remaining = set(mod.functions.keys())
    return remaining, n_dropped


def test_drop_function_only_used_by_grad_pass():
    # `loss` is only referenced through grad(loss); after the rewrite, only
    # loss__grad is called from main. The original `loss` is now dead.
    src = """
    fn loss(x: f32) -> f32 { x * x }
    fn main() -> i32 {
        grad(loss)(21.0) as i32
    }
    """
    remaining, dropped = lower_and_fdce(src)
    assert "main" in remaining
    assert "loss__grad" in remaining
    # The original `loss` should be dropped — main never calls it.
    assert "loss" not in remaining, f"loss should be dead, got {remaining}"
    assert dropped >= 1


def test_keep_directly_called_function():
    src = """
    fn add(a: i32, b: i32) -> i32 { a + b }
    fn main() -> i32 { add(20, 22) }
    """
    remaining, _ = lower_and_fdce(src)
    assert "add" in remaining


def test_keep_transitively_called():
    src = """
    fn double(x: i32) -> i32 { x + x }
    fn quad(x: i32) -> i32 { double(double(x)) }
    fn main() -> i32 { quad(10) }
    """
    remaining, _ = lower_and_fdce(src)
    assert "double" in remaining
    assert "quad" in remaining
    assert "main" in remaining


def test_drop_chain_of_dead_functions():
    # If `helper2` is unused, and `helper1` only calls `helper2`, both die.
    src = """
    fn helper2() -> i32 { 7 }
    fn helper1() -> i32 { helper2() + 1 }
    fn main() -> i32 { 42 }
    """
    remaining, dropped = lower_and_fdce(src)
    assert "main" in remaining
    assert "helper1" not in remaining
    assert "helper2" not in remaining
    assert dropped == 2


def test_no_main_means_no_drops():
    # Without a recognized entry point we don't risk emptying the module.
    src = """
    fn other() -> i32 { 7 }
    """
    prog = parse(src)
    mod = lower(prog)
    # main is missing — fdce should be a no-op
    n = fdce_module(mod, entry_fn="main")
    assert n == 0
    assert "other" in mod.functions


def test_pub_function_kept_even_if_unreachable():
    # A `pub fn` is an exported entry point and should not be dropped even
    # when nothing inside the module calls it. This caught a silent
    # miscompile where lower_ast wasn't propagating fn.is_pub to FnIR.attrs.
    src = """
    pub fn exported_helper() -> i32 { 7 }
    fn main() -> i32 { 42 }
    """
    remaining, _ = lower_and_fdce(src)
    assert "exported_helper" in remaining, \
        f"pub fn should not be dropped, got {remaining}"
    assert "main" in remaining


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
