"""Tests for the Helix Presburger constraint solver."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.presburger import (
    LinExpr, Eq, Le, Divides, Solver, lit, var,
)


# ============================================================================
# LinExpr basics
# ============================================================================
def test_lit_const():
    e = lit(5)
    assert e.is_const() and e.const == 5


def test_var_basic():
    e = var("N")
    assert not e.is_const()
    assert e.coef_of("N") == 1
    assert e.const == 0


def test_add_combines_coefs():
    e = var("N") + var("N")
    assert e.coef_of("N") == 2


def test_add_separate_vars():
    e = var("N") + var("M")
    assert e.coef_of("N") == 1
    assert e.coef_of("M") == 1


def test_sub_zeros_out():
    e = var("N") - var("N")
    assert e.is_zero()


def test_mul_const():
    e = var("N") * 3
    assert e.coef_of("N") == 3


def test_neg():
    e = -(var("N"))
    assert e.coef_of("N") == -1


def test_lin_expr_const_plus_var():
    e = var("N") + lit(5)
    assert e.coef_of("N") == 1
    assert e.const == 5


# ============================================================================
# Eq reasoning
# ============================================================================
def test_eq_implies_self():
    s = Solver()
    s.add_eq_pair(var("N"), var("M"))
    assert s.implies(Eq(var("N") - var("M"))) == True


def test_eq_implies_zero_const():
    s = Solver()
    assert s.implies(Eq(lit(0))) == True


def test_eq_refutes_nonzero_const():
    s = Solver()
    assert s.implies(Eq(lit(1))) == False


def test_eq_chain():
    # M = N, K = M  =>  K = N
    s = Solver()
    s.add_eq_pair(var("M"), var("N"))
    s.add_eq_pair(var("K"), var("M"))
    assert s.implies(Eq(var("K") - var("N"))) == True


def test_eq_with_arith():
    # N = M  =>  N + 1 = M + 1
    s = Solver()
    s.add_eq_pair(var("N"), var("M"))
    assert s.implies(Eq((var("N") + lit(1)) - (var("M") + lit(1)))) == True


# ============================================================================
# Divides reasoning
# ============================================================================
def test_divides_self():
    s = Solver()
    s.add_divides(var("N"), 16)
    assert s.implies(Divides(var("N"), 16)) == True


def test_divides_constant_yes():
    s = Solver()
    assert s.implies(Divides(lit(32), 16)) == True


def test_divides_constant_no():
    s = Solver()
    assert s.implies(Divides(lit(33), 16)) == False


def test_divides_via_coefficient():
    # 16*N is always divisible by 16
    s = Solver()
    assert s.implies(Divides(var("N") * 16, 16)) == True


def test_divides_unknown():
    # We know nothing about N; can't conclude N % 16 == 0
    s = Solver()
    assert s.implies(Divides(var("N"), 16)) is None


def test_divides_partial_known():
    # N % 16 == 0 and we ask about N % 32 — unknown (could be 16 or 32)
    s = Solver()
    s.add_divides(var("N"), 16)
    assert s.implies(Divides(var("N"), 32)) is None


# ============================================================================
# Realistic shape-checking scenarios (the actual use case)
# ============================================================================
def test_matmul_inner_dim_match():
    # tensor<f32, [N, M]> * tensor<f32, [M, P]> requires the M's match.
    # If a function's caller substitutes M_caller = K_caller,
    # we need to verify M_a (caller's) == M_b (caller's)
    # Simplest: at the call site, both inner dims are the SAME variable
    s = Solver()
    # Constraint: a's inner dim and b's outer dim are both K (same var)
    # Verifying matmul's M parameter is consistent
    a_inner = var("K")
    b_outer = var("K")
    assert s.implies(Eq(a_inner - b_outer)) == True


def test_matmul_inner_dim_mismatch():
    # Two different size variables, M and K; without an equality constraint
    # we can't prove they're equal. The compiler should reject the call.
    s = Solver()
    a_inner = var("M")
    b_outer = var("K")
    assert s.implies(Eq(a_inner - b_outer)) is None  # cannot prove


def test_matmul_with_explicit_mismatch():
    # M and K explicitly different (M = K + 1)
    s = Solver()
    s.add_eq_pair(var("M"), var("K") + lit(1))
    assert s.implies(Eq(var("M") - var("K"))) == False


def test_block_matmul_constraints():
    # fn block_matmul[N: size, M: size, B: size] where N % B == 0, M % B == 0
    # Caller passes B=16, N=64, M=32. We need to check N%B==0 and M%B==0.
    s = Solver()
    s.add_eq_pair(var("B"), lit(16))
    s.add_eq_pair(var("N"), lit(64))
    s.add_eq_pair(var("M"), lit(32))
    # 64 % 16 == 0 and 32 % 16 == 0 should be implied
    # Substitute N -> 64 first, then check
    assert s.implies(Divides(var("N"), 16)) == True
    assert s.implies(Divides(var("M"), 16)) == True


def test_block_matmul_wrong_block_size():
    # N = 65 is NOT divisible by 16
    s = Solver()
    s.add_eq_pair(var("N"), lit(65))
    assert s.implies(Divides(var("N"), 16)) == False


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
            import traceback
            print(f"ERROR {name}: {type(e).__name__}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
