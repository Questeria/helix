"""Tests for helixc.frontend.typecheck (v0.1 scaffold)."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend.typecheck import typecheck


def check(src: str) -> list[str]:
    prog = parse(src)
    errs = typecheck(prog)
    return [str(e) for e in errs]


# ============================================================================
# Should typecheck (no errors)
# ============================================================================
def test_simple_add():
    assert check("fn add(a: i32, b: i32) -> i32 { a + b }") == []


def test_let_typed_match():
    assert check("fn f() { let x: i32 = 42; }") == []


def test_let_inferred():
    assert check("fn f() { let x = 42; }") == []


def test_if_branches_match():
    src = "fn f(b: bool) -> i32 { if b { 1 } else { 2 } }"
    assert check(src) == []


def test_function_call():
    src = """
    fn double(x: i32) -> i32 { x + x }
    fn f() -> i32 { double(7) }
    """
    assert check(src) == []


def test_generic_signature():
    src = """
    fn id[T](x: T) -> T { x }
    """
    # T propagates; should typecheck (param T -> return T)
    assert check(src) == []


def test_tensor_signature():
    src = """
    fn matmul[N: size, M: size, P: size](
        a: tensor<f32, [N, M]>,
        b: tensor<f32, [M, P]>,
    ) -> tensor<f32, [N, P]>
    {
        let c = tensor::zeros();
        c
    }
    """
    # tensor::zeros returns Unknown which is compatible-with-anything in v0.1
    assert check(src) == []


def test_struct_def_no_check():
    src = "struct Point { x: f32, y: f32 }"
    # struct definitions don't trigger body checking
    assert check(src) == []


# ============================================================================
# Should detect errors
# ============================================================================
def test_let_type_mismatch():
    errs = check("fn f() { let x: bool = 42; }")
    assert any("declared bool but value is i32" in e for e in errs), errs


def test_return_type_mismatch():
    errs = check("fn f() -> bool { 42 }")
    assert any("does not match return type" in e for e in errs), errs


def test_if_branches_differ():
    errs = check("fn f(b: bool) -> bool { if b { 1 } else { true } }")
    # First arm i32, second bool. Whole if is i32; outer return type bool -> mismatch
    assert any("does not match return type" in e or "if/else branches differ" in e for e in errs), errs


def test_duplicate_function():
    errs = check("fn foo() {} fn foo() {}")
    assert any("duplicate function" in e for e in errs), errs


# ============================================================================
# Compile-time shape checking via Presburger solver (Phase 3-iv)
# ============================================================================
def test_shape_check_concrete_match():
    # Both args have concrete shape [4, 4]; should typecheck cleanly.
    src = """
    fn matmul(a: tensor<f32, [4, 4]>, b: tensor<f32, [4, 4]>) -> tensor<f32, [4, 4]> {
        a
    }
    fn caller(x: tensor<f32, [4, 4]>, y: tensor<f32, [4, 4]>) {
        matmul(x, y);
    }
    """
    assert check(src) == []


def test_shape_check_concrete_mismatch():
    # Caller passes [5, 4] where formal expects [4, 4] — should reject.
    src = """
    fn matmul(a: tensor<f32, [4, 4]>, b: tensor<f32, [4, 4]>) -> tensor<f32, [4, 4]> {
        a
    }
    fn caller(x: tensor<f32, [4, 4]>, z: tensor<f32, [5, 4]>) {
        matmul(x, z);
    }
    """
    errs = check(src)
    assert any("shape constraint violated" in e or "rank" in e for e in errs), errs


def test_shape_check_rank_mismatch():
    src = """
    fn takes2d(a: tensor<f32, [4, 4]>) {}
    fn caller(x: tensor<f32, [4]>) {
        takes2d(x);
    }
    """
    errs = check(src)
    assert any("rank" in e for e in errs), errs


def test_shape_check_size_polymorphic_match():
    src = """
    fn matmul[N: size, M: size, P: size](
        a: tensor<f32, [N, M]>,
        b: tensor<f32, [M, P]>,
    ) -> tensor<f32, [N, P]> { a }
    fn caller(x: tensor<f32, [3, 4]>, y: tensor<f32, [4, 5]>) {
        matmul(x, y);
    }
    """
    errs = check(src)
    assert not any("shape constraint violated" in e for e in errs), errs


# ============================================================================
# Effect/capability type system (Phase 3-ii)
# ============================================================================
def test_pure_calls_pure_ok():
    src = """
    @pure fn helper() -> i32 { 7 }
    @pure fn caller() -> i32 { helper() }
    """
    assert check(src) == []


def test_pure_calls_effectful_rejected():
    # @pure cannot call a function with side effects
    src = """
    @effect fn writes_disk() {}
    @pure fn caller() { writes_disk() }
    """
    errs = check(src)
    assert any("cannot call" in e or "non-pure" in e for e in errs), errs


def test_effectful_can_call_effectful():
    # caller declares the same effects, so the call is permitted
    src = """
    @io fn writes_disk() {}
    @io fn caller() { writes_disk() }
    """
    assert check(src) == []


def test_caller_missing_capability_rejected():
    # io-capable function called from a function without that capability
    src = """
    @io fn read_file() -> i32 { 0 }
    fn naive_caller() { read_file(); }
    """
    errs = check(src)
    assert any("requires effect" in e or "does not declare" in e for e in errs), errs


def test_pure_and_effect_conflict():
    # Function declared both @pure and @effect — conflict
    src = """
    @pure
    @io
    fn confused() {}
    """
    errs = check(src)
    assert any("cannot be both" in e for e in errs), errs


def test_pure_calls_unmarked_function_rejected():
    # Unmarked function is treated as non-pure (default has no @pure)
    src = """
    fn does_anything() -> i32 { 7 }
    @pure fn caller() -> i32 { does_anything() }
    """
    errs = check(src)
    assert any("cannot call" in e or "non-pure" in e for e in errs), errs


# ============================================================================
# Differentiable types D<T> (Phase 3-iii)
# ============================================================================
def test_diff_type_round_trip():
    # Function takes D<f32>, returns D<f32>; should typecheck
    src = """
    fn loss(x: D<f32>) -> D<f32> { x }
    """
    assert check(src) == []


def test_diff_propagates_through_arith():
    # D<f32> * D<f32> should be D<f32>
    src = """
    fn loss(x: D<f32>, y: D<f32>) -> D<f32> {
        x * y
    }
    """
    assert check(src) == []


def test_diff_propagates_with_scalar():
    # D<f32> * f32 should be D<f32> (the f32 is a constant)
    src = """
    fn scale(x: D<f32>) -> D<f32> {
        x * x
    }
    """
    assert check(src) == []


def test_diff_return_type_mismatch():
    # Returning a non-D from a D-typed return is an error
    src = """
    fn bad(x: D<f32>) -> f32 {
        x
    }
    """
    errs = check(src)
    # x is D<f32>, return type declared f32 — type mismatch
    assert any("does not match return type" in e for e in errs), errs


def test_diff_in_tensor():
    # D<tensor<...>> — differentiable tensor
    src = """
    fn loss(x: D<f32>, y: D<f32>) -> D<f32> {
        x * y + x
    }
    """
    assert check(src) == []


# ============================================================================
# Memory-tier types (Phase 3-v)
# ============================================================================
def test_working_mem_type():
    src = """
    fn f(x: WorkingMem<i32>) -> WorkingMem<i32> { x }
    """
    assert check(src) == []


def test_episodic_mem_type():
    src = """
    fn store(e: EpisodicMem<i32>) -> EpisodicMem<i32> { e }
    """
    assert check(src) == []


def test_cannot_pass_episodic_as_semantic():
    # EpisodicMem and SemanticMem are different tiers — must explicitly
    # consolidate. Direct passing should be rejected.
    src = """
    fn takes_semantic(x: SemanticMem<i32>) -> SemanticMem<i32> { x }
    fn caller(e: EpisodicMem<i32>) -> SemanticMem<i32> {
        let s: SemanticMem<i32> = e;
        s
    }
    """
    errs = check(src)
    assert any("declared SemanticMem" in e or "declared" in e for e in errs), errs


def test_same_tier_compatible():
    src = """
    fn takes_working(x: WorkingMem<i32>) -> WorkingMem<i32> { x }
    fn caller(w: WorkingMem<i32>) -> WorkingMem<i32> {
        let y: WorkingMem<i32> = w;
        y
    }
    """
    assert check(src) == []


def test_let_episodic_to_working_rejected():
    src = """
    fn f(e: EpisodicMem<i32>) -> WorkingMem<i32> {
        let w: WorkingMem<i32> = e;
        w
    }
    """
    errs = check(src)
    assert any("declared" in e for e in errs), errs


# ============================================================================
# Test runner
# ============================================================================
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
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
