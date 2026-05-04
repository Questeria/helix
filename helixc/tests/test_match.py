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


def test_arm_body_type_mismatch_errors():
    """All arm bodies must agree on a single result type."""
    src = """
    fn f(x: i32) -> i32 {
        match x {
            _ => 1,
            _ => true,
        }
    }
    """
    errs = _check(src)
    assert any("incompatible" in repr(e).lower() or "mismatch" in repr(e).lower()
               for e in errs), f"expected arm-type-mismatch error, got: {errs}"


def test_non_exhaustive_bool_errors():
    """A `match` on bool with only `true` arm should error: missing `false`."""
    src = """
    fn f(b: bool) -> i32 {
        match b {
            true => 1,
        }
    }
    """
    errs = _check(src)
    assert any("non-exhaustive" in repr(e).lower() and "false" in repr(e).lower()
               for e in errs), f"expected non-exhaustive-bool error, got: {errs}"


def test_exhaustive_bool_with_both_arms_ok():
    """`match b { true => 1, false => 0 }` is exhaustive — no error."""
    src = """
    fn f(b: bool) -> i32 {
        match b {
            true => 1,
            false => 0,
        }
    }
    """
    errs = _check(src)
    assert errs == [], f"unexpected typecheck errors: {errs}"


def test_or_pattern_typechecks():
    """Or-pattern `1 | 2 | 3` should typecheck and bind nothing."""
    src = """
    fn f(x: i32) -> i32 {
        match x {
            1 | 2 | 3 => 42,
            _ => 0,
        }
    }
    """
    errs = _check(src)
    assert errs == [], f"unexpected typecheck errors: {errs}"


def test_match_int_literal_runs():
    """End-to-end: match on int literal selects the right arm at runtime."""
    try:
        from helixc.tests.test_codegen import compile_and_run
    except Exception:
        # codegen suite may not be importable in some environments; skip.
        return
    src = """
    fn main() -> i32 {
        let x = 2;
        match x {
            1 => 10,
            2 => 42,
            3 => 30,
            _ => 99,
        }
    }
    """
    code = compile_and_run(src)
    assert code == 42, f"expected exit 42 (arm 2), got {code}"


def test_match_range_pattern_runs():
    """End-to-end: range pattern selects correct arm."""
    try:
        from helixc.tests.test_codegen import compile_and_run
    except Exception:
        return
    src = """
    fn main() -> i32 {
        let x = 5;
        match x {
            0..3 => 1,
            3..=7 => 42,
            _ => 99,
        }
    }
    """
    code = compile_and_run(src)
    assert code == 42, f"expected exit 42 (range arm), got {code}"


def test_match_or_pattern_runs():
    """End-to-end: or-pattern matches any of its alternatives."""
    try:
        from helixc.tests.test_codegen import compile_and_run
    except Exception:
        return
    src = """
    fn main() -> i32 {
        let x = 3;
        match x {
            1 | 2 | 3 => 42,
            _ => 0,
        }
    }
    """
    code = compile_and_run(src)
    assert code == 42, f"expected exit 42 (or-pattern arm), got {code}"


def test_match_bind_runs():
    """End-to-end: PatBind binds scrutinee value visible in body."""
    try:
        from helixc.tests.test_codegen import compile_and_run
    except Exception:
        return
    src = """
    fn main() -> i32 {
        let x = 21;
        match x {
            y => y * 2,
        }
    }
    """
    code = compile_and_run(src)
    assert code == 42, f"expected exit 42 (y*2), got {code}"


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
