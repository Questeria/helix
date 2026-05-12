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


# --- Stage 21 regression tests ---

def test_stage21_factorial_passes_spec_example():
    """Stage 21 spec example: factorial(n) calls factorial(n-1) — strict
    decrease, total. No @partial annotation needed."""
    src = """
    fn factorial(n: i32) -> i32 {
        if n <= 1 { 1 } else { n * factorial(n - 1) }
    }
    """
    fails = check_totality(parse(src))
    assert fails == [], f"factorial must be accepted, got {fails}"


def test_stage21_collatz_with_partial_passes_spec_example():
    """Stage 21 spec example: Collatz with @partial annotation passes;
    without @partial it traps 21001."""
    src_with_partial = """
    @partial
    fn collatz(n: i32) -> i32 {
        if n == 1 { 1 }
        else { if n % 2 == 0 { collatz(n / 2) } else { collatz(n * 3 + 1) } }
    }
    """
    fails = check_totality(parse(src_with_partial))
    assert fails == [], f"@partial collatz must pass, got {fails}"


def test_stage21_trap_21001_collatz_without_partial():
    """Stage 21 trap-id 21001: a non-`@partial` recursive function that
    does not strictly decrease on every recursive call is rejected.
    Verifies the trap surfaces in the totality failure report."""
    src = """
    fn collatz_no_partial(n: i32) -> i32 {
        if n == 1 { 1 }
        else { if n % 2 == 0 { collatz_no_partial(n / 2) }
               else { collatz_no_partial(n * 3 + 1) } }
    }
    """
    fails = check_totality(parse(src))
    names = [name for name, _ in fails]
    assert "collatz_no_partial" in names, (
        f"expected trap 21001 for collatz_no_partial, got {fails}"
    )


def test_stage21_totality_runs_in_x86_64_driver_pipeline():
    """The x86_64 driver's __main__ now runs check_totality before
    lowering. Smoke check: imports + call-site present."""
    from helixc.backend import x86_64
    import inspect
    src = inspect.getsource(x86_64)
    assert "check_totality" in src, (
        "x86_64.py driver no longer runs check_totality — Stage 21 wiring lost"
    )
    assert "trap 21001" in src, (
        "x86_64.py driver no longer surfaces trap 21001 — wiring incomplete"
    )


def test_stage21_strict_mode_aborts_on_totality_failure():
    """In --strict mode, a totality failure makes the driver exit
    nonzero. We test this via subprocess to confirm the wiring is
    end-to-end (not just the warning-printing path)."""
    import os, subprocess, tempfile
    proj_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    src_dir = os.path.join(proj_root, "helixc", "tests", "_tmp")
    os.makedirs(src_dir, exist_ok=True)
    src_path = os.path.join(src_dir, "_stage21_strict.hx")
    out_path = os.path.join(src_dir, "_stage21_strict.bin")
    with open(src_path, "w", encoding="utf-8") as f:
        f.write("""
fn forever(n: i32) -> i32 {
    forever(n)
}
fn main() -> i32 { 0 }
""")
    cmd = [sys.executable, "-m", "helixc.backend.x86_64",
           src_path, out_path, "--strict"]
    r = subprocess.run(cmd, capture_output=True, cwd=proj_root, text=True,
                       timeout=60)
    # --strict + totality failure => nonzero exit.
    assert r.returncode != 0, (
        f"expected nonzero exit on --strict totality failure; "
        f"got rc={r.returncode}, stderr={r.stderr!r}"
    )
    assert "21001" in r.stderr or "totality" in r.stderr, (
        f"expected trap 21001 / 'totality' in stderr, got: {r.stderr}"
    )


def test_c57_1_recursion_inside_mod_block_detected():
    """Stage 28.9 cycle 58 audit-R C57-1 regression (HIGH conf 88):
    a non-@partial recursive fn inside `mod m { ... }` must be flagged
    by `check_totality`. Pre-fix the outer walker only inspected
    `prog.items` filtered for `A.FnDecl`, so a ModBlock-nested fn
    silently reported `totality: OK` from `helixc check` even though
    the backend driver's flatten_modules-first pipeline did catch it."""
    src = """
    mod inner {
        fn forever(n: i32) -> i32 {
            forever(n)
        }
    }
    """
    fails = check_totality(parse(src))
    names = [name for name, _ in fails]
    assert "forever" in names, (
        f"expected mod-nested forever flagged, got fails={fails}"
    )


def test_c57_1_recursion_inside_impl_method_detected():
    """C57-1 regression (HIGH): a non-@partial recursive associated
    fn inside `impl X { ... }` must be flagged by `check_totality`.
    Same item-walker gap as the ModBlock case. Uses an associated fn
    (no `self` receiver) to keep the test minimal and parser-stable."""
    src = """
    struct S { x: i32 }
    impl S {
        fn forever(n: i32) -> i32 {
            forever(n)
        }
    }
    """
    fails = check_totality(parse(src))
    names = [name for name, _ in fails]
    assert "forever" in names, (
        f"expected impl-method forever flagged, got fails={fails}"
    )


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
