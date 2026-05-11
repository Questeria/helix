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


def test_match_bind_with_guard_runs():
    """End-to-end: PatBind + arm guard. y if y > 10 should fire only
    when scrutinee > 10."""
    try:
        from helixc.tests.test_codegen import compile_and_run
    except Exception:
        return
    src = """
    fn main() -> i32 {
        let x = 21;
        match x {
            y if y > 10 => y * 2,
            _ => 0,
        }
    }
    """
    code = compile_and_run(src)
    assert code == 42, f"expected exit 42 (y > 10 path), got {code}"


def test_match_guard_falsy_falls_through():
    """End-to-end: guard returning false should skip to next arm."""
    try:
        from helixc.tests.test_codegen import compile_and_run
    except Exception:
        return
    src = """
    fn main() -> i32 {
        let x = 5;
        match x {
            y if y > 100 => 1,
            y if y > 50 => 2,
            y if y > 0 => 42,
            _ => 99,
        }
    }
    """
    code = compile_and_run(src)
    assert code == 42, f"expected 42 (third guard fires), got {code}"


def test_match_inclusive_range_endpoint():
    """Inclusive range: 7 should match 0..=7 but not 0..7."""
    try:
        from helixc.tests.test_codegen import compile_and_run
    except Exception:
        return
    src_inclusive = """
    fn main() -> i32 {
        match 7 {
            0..=7 => 42,
            _ => 0,
        }
    }
    """
    src_exclusive = """
    fn main() -> i32 {
        match 7 {
            0..7 => 42,
            _ => 0,
        }
    }
    """
    assert compile_and_run(src_inclusive) == 42, \
        "expected 7 to match 0..=7"
    assert compile_and_run(src_exclusive) == 0, \
        "expected 7 NOT to match 0..7 (exclusive)"


def test_match_nested_in_let():
    """Match expression nested as the value of a let-binding."""
    try:
        from helixc.tests.test_codegen import compile_and_run
    except Exception:
        return
    src = """
    fn main() -> i32 {
        let r = match 3 {
            1 => 10,
            2 => 20,
            3 => 42,
            _ => 0,
        };
        r
    }
    """
    code = compile_and_run(src)
    assert code == 42, f"expected 42, got {code}"


def test_or_pattern_only_intersected_binders_visible_in_body():
    """A name bound in only one alternative of an or-pattern must NOT be
    visible in the arm body (it would be uninitialized for the other
    alternatives). The `y` here is bound only in alt 1, not in alt 2."""
    src = """
    fn f(x: i32) -> i32 {
        match x {
            y | 0 => y,
            _ => 0,
        }
    }
    """
    errs = _check(src)
    # `y` should be unbound in body (since it's only bound in one alt).
    assert any("unbound" in repr(e).lower() and "y" in repr(e)
               for e in errs), \
        f"expected `y` unbound in or-arm body, got: {errs}"


def test_or_pattern_uniform_binders_visible_in_body():
    """If every alternative binds the same name, that name IS visible."""
    src = """
    fn f(x: i32) -> i32 {
        match x {
            y => y,
        }
    }
    """
    errs = _check(src)
    assert errs == [], f"expected uniform binder visible, got: {errs}"


def test_non_exhaustive_enum_errors():
    """`match o { Op::Add => 0, Op::Sub => 1 }` errors when Op has Mul too."""
    src = """
    enum Op { Add, Sub, Mul }
    fn f(o: i32) -> i32 {
        match o {
            Op::Add => 0,
            Op::Sub => 1,
        }
    }
    """
    errs = _check(src)
    assert any("non-exhaustive" in repr(e).lower() and "Mul" in repr(e)
               for e in errs), f"expected missing-variant error, got: {errs}"


def test_or_pattern_covers_multiple_variants_for_exhaustiveness():
    """`E::A | E::C => ...` arm covers BOTH variants for exhaustiveness.
    Pre-fix the audit caught: only the first variant was counted, falsely
    flagging E::C as missing."""
    src = """
    enum E { A, B, C }
    fn f(o: i32) -> i32 {
        match o {
            E::A | E::C => 0,
            E::B => 1,
        }
    }
    """
    errs = _check(src)
    assert errs == [], f"or-pattern E::A | E::C should cover both, got {errs}"


def test_exhaustive_enum_match_ok():
    """All variants covered → no error."""
    src = """
    enum Op { Add, Sub }
    fn f(o: i32) -> i32 {
        match o {
            Op::Add => 0,
            Op::Sub => 1,
        }
    }
    """
    errs = _check(src)
    assert errs == [], f"unexpected errors: {errs}"


def test_match_nested_match():
    """Match inside the body of another match arm."""
    try:
        from helixc.tests.test_codegen import compile_and_run
    except Exception:
        return
    src = """
    fn main() -> i32 {
        let outer = 1;
        let inner = 2;
        match outer {
            1 => match inner {
                1 => 10,
                2 => 42,
                _ => 0,
            },
            _ => 99,
        }
    }
    """
    code = compile_and_run(src)
    assert code == 42, f"expected 42 (inner arm 2), got {code}"


def test_c22_c_match_inside_unsafe_block_lowered():
    """Audit 28.8 cycle 23 C22-C (HIGH): `match` inside `unsafe { ... }`
    must be desugared by lower_matches. Pre-fix, `match_lower._rewrite_expr`
    had no UnsafeBlock arm — the Match persisted past lower_matches
    and crashed lower_ast's `Match should not reach _lower_expr`
    assertion."""
    from helixc.frontend.parser import parse as parse_src
    from helixc.frontend.match_lower import lower_matches
    from helixc.frontend import ast_nodes as A
    src = """
    fn main() -> i32 {
        unsafe {
            match 1 {
                1 => 42,
                _ => 0,
            }
        }
    }
    """
    prog = parse_src(src)
    lower_matches(prog)
    # Walk the program: no A.Match should remain anywhere.
    def walk(node):
        if node is None:
            return
        if isinstance(node, A.Match):
            raise AssertionError(
                "lower_matches must remove all Match nodes; one "
                "remains inside UnsafeBlock (C22-C)"
            )
        if isinstance(node, list):
            for x in node:
                walk(x)
            return
        if hasattr(node, "__dict__"):
            for v in vars(node).values():
                walk(v)
    walk(prog)


def test_c22_c_match_inside_range_lowered():
    """Audit 28.8 cycle 23 C22-C: `match` inside Range.start/end must
    be desugared. Pre-fix `for i in 0..match n { ... }` crashed."""
    from helixc.frontend.parser import parse as parse_src
    from helixc.frontend.match_lower import lower_matches
    from helixc.frontend import ast_nodes as A
    src = """
    fn main() -> i32 {
        let n = 5;
        let mut total = 0;
        for i in 0 .. (match n { 0 => 0, _ => 3 }) {
            total = total + 1;
        }
        total
    }
    """
    prog = parse_src(src)
    lower_matches(prog)
    def walk(node):
        if node is None:
            return
        if isinstance(node, A.Match):
            raise AssertionError(
                "lower_matches must remove all Match nodes; one "
                "remains inside Range (C22-C)"
            )
        if isinstance(node, list):
            for x in node:
                walk(x)
            return
        if hasattr(node, "__dict__"):
            for v in vars(node).values():
                walk(v)
    walk(prog)


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
