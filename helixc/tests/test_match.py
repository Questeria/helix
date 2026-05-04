"""Tests for `match` typecheck — Tier A WORK_QUEUE items."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend.typecheck import TypeChecker


def _check(src: str) -> list:
    prog = parse(src)
    tc = TypeChecker(prog)
    tc.check()
    return tc.errors


def test_match_binds_pattern_var():
    """A `PatBind` arm binds the binder into arm-body scope."""
    src = """
    fn f(x: i32) -> i32 {
        match x {
            y => y + 1,
            _ => 0,
        }
    }
    """
    errs = _check(src)
    assert errs == [], f"unexpected typecheck errors: {errs}"


def test_match_binder_visible_in_guard():
    """Binders introduced by the pattern are visible in the arm guard."""
    src = """
    fn f(x: i32) -> i32 {
        match x {
            y if y > 0 => y,
            _ => 0,
        }
    }
    """
    errs = _check(src)
    assert errs == [], f"unexpected typecheck errors: {errs}"


def test_match_wildcard_only_typechecks():
    src = """
    fn f(x: i32) -> i32 {
        match x {
            _ => 42,
        }
    }
    """
    errs = _check(src)
    assert errs == [], f"unexpected typecheck errors: {errs}"


def test_match_unbound_in_outer_scope():
    """Binders from a match arm must NOT leak to the outer scope."""
    src = """
    fn f(x: i32) -> i32 {
        let _r = match x {
            y => y,
            _ => 0,
        };
        y
    }
    """
    errs = _check(src)
    # The `y` reference outside the match arm should be unbound.
    assert any("y" in repr(e) or "unbound" in repr(e).lower() for e in errs), \
        f"expected unbound-name error for outer `y`, got: {errs}"


def test_match_guard_must_be_bool():
    """A non-bool guard expression is a typecheck error."""
    src = """
    fn f(x: i32) -> i32 {
        match x {
            y if y => y,
            _ => 0,
        }
    }
    """
    errs = _check(src)
    assert any("guard" in repr(e).lower() and "bool" in repr(e).lower() for e in errs), \
        f"expected 'guard must be bool' error, got: {errs}"


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
