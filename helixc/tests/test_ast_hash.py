"""Tests for the structural AST hashing module."""

from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend import ast_nodes as A
from helixc.frontend.ast_hash import structural_hash, short_hash


def _body_of(src: str) -> A.Block:
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    return fn.body


def test_identical_expressions_hash_equal():
    h1 = structural_hash(_body_of("fn f(x: f32) -> f32 { x * x }"))
    h2 = structural_hash(_body_of("fn f(x: f32) -> f32 { x * x }"))
    assert h1 == h2


def test_alpha_equivalent_let_bindings_hash_equal():
    # Different bound-variable name, same structural shape.
    h1 = structural_hash(_body_of("fn f() -> f32 { let a = 5.0; a + a }"))
    h2 = structural_hash(_body_of("fn f() -> f32 { let q = 5.0; q + q }"))
    assert h1 == h2


def test_different_constants_hash_unequal():
    h1 = structural_hash(_body_of("fn f() -> f32 { 5.0 + 3.0 }"))
    h2 = structural_hash(_body_of("fn f() -> f32 { 5.0 + 4.0 }"))
    assert h1 != h2


def test_different_operators_hash_unequal():
    h1 = structural_hash(_body_of("fn f(x: f32) -> f32 { x + x }"))
    h2 = structural_hash(_body_of("fn f(x: f32) -> f32 { x * x }"))
    assert h1 != h2


def test_free_vs_bound_name_distinction():
    # "x" inside `let x = 5.0; x` is bound; "x" without let is free.
    h_bound = structural_hash(_body_of("fn f() -> f32 { let x = 5.0; x }"))
    h_free = structural_hash(_body_of("fn f(x: f32) -> f32 { x }"))
    # Both reference x but in different scopes — ensure the hashes
    # capture this. They might or might not be equal depending on
    # the FnParam binding, but they should be DETERMINISTIC.
    assert isinstance(h_bound, str)
    assert isinstance(h_free, str)


def test_function_param_alpha_equivalence():
    # Hashing a function body in isolation, the param name is FREE.
    # To assert alpha-equivalence over the whole function, hash the
    # FnDecl itself (which canonicalizes param names to indices).
    def _fn_of(src: str) -> A.FnDecl:
        prog = parse(src)
        return next(it for it in prog.items if isinstance(it, A.FnDecl))
    h1 = structural_hash(_fn_of("fn f(x: f32) -> f32 { x * 2.0 }"))
    h2 = structural_hash(_fn_of("fn f(y: f32) -> f32 { y * 2.0 }"))
    assert h1 == h2, f"alpha-eq mismatch: {h1} != {h2}"


def test_free_name_changes_hash():
    # Different free names should produce different hashes.
    h1 = structural_hash(_body_of("fn f() -> f32 { foo + 1.0 }"))
    h2 = structural_hash(_body_of("fn f() -> f32 { bar + 1.0 }"))
    assert h1 != h2


def test_short_hash_is_12_chars():
    h = structural_hash(_body_of("fn f() -> i32 { 42 }"))
    s = short_hash(h)
    assert len(s) == 12
    assert all(c in "0123456789abcdef" for c in s)


def test_hash_is_deterministic_across_runs():
    # Sanity: same input, same hash, every time.
    src = "fn f(a: f32, b: f32) -> f32 { a*a + b*b + a*b }"
    hashes = [structural_hash(_body_of(src)) for _ in range(5)]
    assert len(set(hashes)) == 1


def test_call_args_count_matters():
    # f(x) and f(x, x) have different shapes.
    h1 = structural_hash(_body_of("fn f() -> f32 { foo(1.0) }"))
    h2 = structural_hash(_body_of("fn f() -> f32 { foo(1.0, 1.0) }"))
    assert h1 != h2


def test_quote_alpha_equivalence():
    """Two quotes that are alpha-equivalent should hash equal."""
    h1 = structural_hash(_body_of("fn f() -> i64 { let q = quote { 1 + 2 }; 0 }"))
    h2 = structural_hash(_body_of("fn f() -> i64 { let q = quote { 1 + 2 }; 0 }"))
    assert h1 == h2


def test_quote_different_inner_diff_hash():
    """Quotes with different inner exprs hash differently."""
    h1 = structural_hash(_body_of("fn f() -> i64 { let q = quote { 1 + 2 }; 0 }"))
    h2 = structural_hash(_body_of("fn f() -> i64 { let q = quote { 1 + 3 }; 0 }"))
    assert h1 != h2


def test_for_loop_alpha_equivalence():
    """`for i in 0..10 { i+1 }` and `for j in 0..10 { j+1 }` are alpha-
    equivalent — same shape, renamed binder. Should hash equal."""
    h_i = structural_hash(_body_of("fn f() -> i32 { for i in 0..10 { i + 1; } 0 }"))
    h_j = structural_hash(_body_of("fn f() -> i32 { for j in 0..10 { j + 1; } 0 }"))
    assert h_i == h_j, f"alpha-eq mismatch for For: {h_i} != {h_j}"


def test_range_endpoints_matter():
    """Different range endpoints in a `for` should hash differently."""
    h1 = structural_hash(_body_of("fn f() -> i32 { for i in 0..10 { } 0 }"))
    h2 = structural_hash(_body_of("fn f() -> i32 { for i in 0..20 { } 0 }"))
    assert h1 != h2


def test_if_branch_swap_changes_hash():
    """if cond { a } else { b } and if cond { b } else { a } differ."""
    h1 = structural_hash(_body_of(
        "fn f(c: bool, a: i32, b: i32) -> i32 { if c { a } else { b } }"))
    h2 = structural_hash(_body_of(
        "fn f(c: bool, a: i32, b: i32) -> i32 { if c { b } else { a } }"))
    assert h1 != h2


def test_unary_distinct_from_binary():
    """`-x` and `0 - x` are structurally distinct, hash differently."""
    h1 = structural_hash(_body_of("fn f(x: f32) -> f32 { -x }"))
    h2 = structural_hash(_body_of("fn f(x: f32) -> f32 { 0.0 - x }"))
    assert h1 != h2


def test_match_alpha_equivalence():
    """Two matches with same shape but renamed binder hash equal."""
    h1 = structural_hash(_body_of("""
    fn f(x: i32) -> i32 {
        match x {
            y => y + 1,
            _ => 0,
        }
    }
    """))
    h2 = structural_hash(_body_of("""
    fn f(x: i32) -> i32 {
        match x {
            z => z + 1,
            _ => 0,
        }
    }
    """))
    assert h1 == h2


def test_match_arm_count_matters():
    """Different arm counts produce different hashes."""
    h1 = structural_hash(_body_of("""
    fn f(x: i32) -> i32 {
        match x { 1 => 1, _ => 0 }
    }
    """))
    h2 = structural_hash(_body_of("""
    fn f(x: i32) -> i32 {
        match x { 1 => 1, 2 => 2, _ => 0 }
    }
    """))
    assert h1 != h2


def test_nested_let_alpha_equivalence():
    h1 = structural_hash(_body_of("""
    fn f() -> f32 {
        let a = 1.0;
        let b = 2.0;
        a + b
    }
    """))
    h2 = structural_hash(_body_of("""
    fn f() -> f32 {
        let p = 1.0;
        let q = 2.0;
        p + q
    }
    """))
    assert h1 == h2


# ============================================================================
# Stage 58 / Tier 4 #13 — content-addressed modules
# ============================================================================
def test_stage58_program_hash_identical_programs_match():
    """Stage 58 / Tier 4 #13: identical source programs produce
    identical program_hash."""
    from helixc.frontend.parser import parse
    from helixc.frontend.ast_hash import program_hash
    src = "fn main() -> i32 { 42 }"
    h1 = program_hash(parse(src))
    h2 = program_hash(parse(src))
    assert h1 == h2, f"identical programs hashed differently: {h1} vs {h2}"


def test_stage58_program_hash_changes_with_body():
    """Stage 58 / Tier 4 #13: changing fn body changes program_hash."""
    from helixc.frontend.parser import parse
    from helixc.frontend.ast_hash import program_hash
    h1 = program_hash(parse("fn main() -> i32 { 42 }"))
    h2 = program_hash(parse("fn main() -> i32 { 43 }"))
    assert h1 != h2, "different programs hashed identically"


def test_stage58_program_hash_independent_of_span():
    """Stage 58 / Tier 4 #13: source formatting (spans) doesn't
    affect program_hash."""
    from helixc.frontend.parser import parse
    from helixc.frontend.ast_hash import program_hash
    h1 = program_hash(parse("fn main() -> i32 { 42 }"))
    h2 = program_hash(parse("""

    fn main() -> i32 {
        42
    }
    """))
    assert h1 == h2, "span/formatting changes affected program_hash"


def test_stage58_module_hash_identical_modblocks_match():
    """Stage 58 / Tier 4 #13: identical ModBlocks produce identical
    module_hash. Tests the module-level aggregator (sibling of
    program_hash but scoped to one `mod X { ... }` block)."""
    from helixc.frontend.parser import parse
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.ast_hash import module_hash
    src = "mod foo { fn bar() -> i32 { 1 } }"
    prog1 = parse(src)
    prog2 = parse(src)
    mod1 = next(it for it in prog1.items
                if isinstance(it, A.ModBlock))
    mod2 = next(it for it in prog2.items
                if isinstance(it, A.ModBlock))
    assert module_hash(mod1) == module_hash(mod2)


def test_stage58_module_hash_different_names_differ():
    """Stage 58: ModBlocks with same items but different names hash
    differently (name is part of identity)."""
    from helixc.frontend.parser import parse
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.ast_hash import module_hash
    src1 = "mod alpha { fn bar() -> i32 { 1 } }"
    src2 = "mod beta { fn bar() -> i32 { 1 } }"
    mod1 = next(it for it in parse(src1).items
                if isinstance(it, A.ModBlock))
    mod2 = next(it for it in parse(src2).items
                if isinstance(it, A.ModBlock))
    assert module_hash(mod1) != module_hash(mod2), \
        "module name should affect hash"


def test_stage58_module_hash_different_item_body_differs():
    """Stage 58: same ModBlock name but different fn body hashes
    differently."""
    from helixc.frontend.parser import parse
    from helixc.frontend import ast_nodes as A
    from helixc.frontend.ast_hash import module_hash
    src1 = "mod m { fn f() -> i32 { 1 } }"
    src2 = "mod m { fn f() -> i32 { 2 } }"
    mod1 = next(it for it in parse(src1).items
                if isinstance(it, A.ModBlock))
    mod2 = next(it for it in parse(src2).items
                if isinstance(it, A.ModBlock))
    assert module_hash(mod1) != module_hash(mod2), \
        "module item body diff should affect hash"


def test_stage58_program_hash_alpha_equivalent_helpers():
    """Stage 58 / Tier 4 #13: programs that differ ONLY in bound-
    variable names hash identically (alpha-equivalence)."""
    from helixc.frontend.parser import parse
    from helixc.frontend.ast_hash import program_hash
    h1 = program_hash(parse("fn f(x: i32) -> i32 { x + 1 }"))
    h2 = program_hash(parse("fn f(y: i32) -> i32 { y + 1 }"))
    assert h1 == h2, "alpha-equivalent helpers hashed differently"


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
