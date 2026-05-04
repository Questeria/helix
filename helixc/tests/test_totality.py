"""Tests for the structural-recursion totality stub."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend.totality import check_totality


def test_factorial_accepted():
    """factorial(n) calls factorial(n-1) — strict decrease, total."""
    src = """
    fn factorial(n: i32) -> i32 {
        if n <= 1 { 1 } else { n * factorial(n - 1) }
    }
    """
    fails = check_totality(parse(src))
    assert fails == [], f"expected factorial accepted, got {fails}"


def test_collatz_rejected_without_partial():
    """Collatz: even n -> n/2, odd n -> 3*n+1. The 3n+1 step doesn't
    strictly decrease — must be @partial or rejected."""
    src = """
    fn collatz(n: i32) -> i32 {
        if n <= 1 { 0 }
        else { if (n % 2) == 0 { collatz(n / 2) } else { collatz(3*n + 1) } }
    }
    """
    fails = check_totality(parse(src))
    names = [name for name, _ in fails]
    assert "collatz" in names, \
        f"expected collatz rejected, got fails={fails}"


def test_collatz_with_partial_accepted():
    """Same Collatz with `@partial` annotation — totality check skips it."""
    src = """
    @partial
    fn collatz(n: i32) -> i32 {
        if n <= 1 { 0 }
        else { if (n % 2) == 0 { collatz(n / 2) } else { collatz(3*n + 1) } }
    }
    """
    fails = check_totality(parse(src))
    assert fails == [], f"expected @partial collatz accepted, got {fails}"


def test_non_recursive_accepted():
    """A non-recursive function is trivially total."""
    src = "fn add(a: i32, b: i32) -> i32 { a + b }"
    fails = check_totality(parse(src))
    assert fails == [], f"expected non-recursive accepted, got {fails}"


def test_division_by_two_accepted():
    """`f(n) -> f(n / 2)` is strictly decreasing for n > 1."""
    src = """
    fn binary_search_depth(n: i32) -> i32 {
        if n <= 1 { 0 } else { 1 + binary_search_depth(n / 2) }
    }
    """
    fails = check_totality(parse(src))
    assert fails == [], f"expected n/2 recursion accepted, got {fails}"


def test_recursion_inside_branch_must_decrease_in_all_paths():
    """A self-call through one branch with no decrease should reject,
    even if another branch is fine."""
    src = """
    fn maybe_dec(n: i32, b: bool) -> i32 {
        if b { maybe_dec(n - 1, b) } else { maybe_dec(n, b) }
    }
    """
    fails = check_totality(parse(src))
    names = [name for name, _ in fails]
    assert "maybe_dec" in names, \
        f"expected maybe_dec rejected (false-branch doesn't decrease), got {fails}"


def test_partial_attribute_disables_check():
    """@partial silences the check even on absurd recursion."""
    src = """
    @partial
    fn whatever(n: i32) -> i32 {
        whatever(n + 1)
    }
    """
    fails = check_totality(parse(src))
    assert fails == [], f"expected @partial to disable check, got {fails}"


def test_constant_arg_recursion_rejected():
    """Calling self with the same arg unchanged — non-terminating."""
    src = """
    fn forever(n: i32) -> i32 {
        forever(n)
    }
    """
    fails = check_totality(parse(src))
    names = [name for name, _ in fails]
    assert "forever" in names, f"expected forever rejected, got {fails}"


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
