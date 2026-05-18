"""Stage 27: tests for @autotune attribute parsing + variant generation."""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse, ParseError
from helixc.frontend import ast_nodes as A
from helixc.frontend.autotune import (
    parse_autotune_attrs,
    parse_autotune_attrs_dict,
    autotune_variants,
    variant_count,
    has_autotune,
    has_kernel,
    mangled_variant_name,
    validate_autotune,
    validate_autotune_prog,
    collect_autotuned_fns,
    TRAP_AUTOTUNE_OVERSIZED,
    MAX_VARIANT_PRODUCT,
)


def test_parse_single_key_autotune():
    src = """
@autotune(BLOCK_SIZE: [16, 32])
fn matmul(a: i32) -> i32 { a }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    assert "autotune" in fn.attrs
    # Audit 28.8 A12: parse_autotune_attrs now returns (dict, diags).
    params, diags = parse_autotune_attrs(fn)
    assert params == {"BLOCK_SIZE": [16, 32]}
    assert diags == []


def test_stage56_autotune_expand_single_key():
    """Stage 56 Inc 1: expand_autotune_kernels emits N FnDecl
    variants from one @autotune @kernel fn. Single key with 2
    values → 2 variants."""
    from helixc.frontend.autotune_expand import expand_autotune_kernels
    src = """
@autotune(BLOCK_SIZE: [16, 32])
@kernel
fn vec_add(a: i32) -> i32 { BLOCK_SIZE + a }
"""
    prog = parse(src)
    expanded = expand_autotune_kernels(prog)
    fns = [it for it in expanded.items if isinstance(it, A.FnDecl)]
    # Should have 2 variants of vec_add (original replaced).
    vec_fns = [f for f in fns if f.name.startswith("vec_add")]
    assert len(vec_fns) == 2, \
        f"expected 2 variants, got {len(vec_fns)}: {[f.name for f in vec_fns]}"
    names = sorted(f.name for f in vec_fns)
    assert names == [
        "vec_add__autotune_BLOCK_SIZE_16",
        "vec_add__autotune_BLOCK_SIZE_32",
    ], f"unexpected names: {names}"
    # Each variant must still be @kernel.
    for f in vec_fns:
        assert "kernel" in f.attrs
        assert "autotune" not in f.attrs


def test_stage56_autotune_expand_two_keys_product():
    """Stage 56 Inc 1: 2 keys × 2 values each → 4 variants."""
    from helixc.frontend.autotune_expand import expand_autotune_kernels
    src = """
@autotune(BLOCK_SIZE: [16, 32], NUM_WARPS: [4, 8])
@kernel
fn mm(a: i32) -> i32 { BLOCK_SIZE * NUM_WARPS + a }
"""
    prog = parse(src)
    expanded = expand_autotune_kernels(prog)
    fns = [it for it in expanded.items if isinstance(it, A.FnDecl)]
    mm_fns = [f for f in fns if f.name.startswith("mm")]
    assert len(mm_fns) == 4, \
        f"expected 4 variants (2x2), got {len(mm_fns)}"


def test_stage56_autotune_expand_constant_substitution():
    """Stage 56 Inc 1: Name(KEY) refs in the body are replaced
    by IntLit(VAL) for each variant's config."""
    from helixc.frontend.autotune_expand import expand_autotune_kernels
    src = """
@autotune(BLOCK_SIZE: [16, 32])
@kernel
fn k(a: i32) -> i32 { BLOCK_SIZE + a }
"""
    prog = parse(src)
    expanded = expand_autotune_kernels(prog)
    fns = [it for it in expanded.items if isinstance(it, A.FnDecl)]
    v16 = next(f for f in fns if "16" in f.name)
    # Body should have BLOCK_SIZE replaced with IntLit(16).
    # Final expr is Binary(BLOCK_SIZE + a) → Binary(IntLit(16) + Name(a))
    body = v16.body.final_expr
    assert isinstance(body, A.Binary)
    assert isinstance(body.left, A.IntLit), \
        f"expected IntLit, got {type(body.left).__name__}"
    assert body.left.value == 16


def test_stage56_autotune_expand_non_autotune_passthrough():
    """Stage 56 Inc 1: non-@autotune fns pass through unchanged."""
    from helixc.frontend.autotune_expand import expand_autotune_kernels
    src = """
fn plain(a: i32) -> i32 { a + 1 }
"""
    prog = parse(src)
    expanded = expand_autotune_kernels(prog)
    fns = [it for it in expanded.items if isinstance(it, A.FnDecl)]
    assert len(fns) == 1
    assert fns[0].name == "plain"


def test_parse_two_key_autotune():
    src = """
@autotune(BLOCK_SIZE: [16, 32, 64], NUM_WARPS: [4, 8])
fn matmul(a: i32) -> i32 { a }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    params, diags = parse_autotune_attrs(fn)
    assert params == {"BLOCK_SIZE": [16, 32, 64], "NUM_WARPS": [4, 8]}
    assert diags == []


def test_parse_with_kernel_attribute():
    src = """
@kernel
@autotune(BLOCK_SIZE: [16])
fn matmul(a: i32) -> i32 { a }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    assert has_kernel(fn)
    assert has_autotune(fn)


def test_variant_count():
    assert variant_count({"A": [1, 2]}) == 2
    assert variant_count({"A": [1, 2], "B": [3, 4, 5]}) == 6
    assert variant_count({}) == 0


def test_variants_single_param():
    vs = autotune_variants({"BLOCK_SIZE": [16, 32]})
    assert vs == [{"BLOCK_SIZE": 16}, {"BLOCK_SIZE": 32}]


def test_variants_cross_product():
    vs = autotune_variants({"A": [1, 2], "B": [3, 4]})
    assert len(vs) == 4
    assert {"A": 1, "B": 3} in vs
    assert {"A": 1, "B": 4} in vs
    assert {"A": 2, "B": 3} in vs
    assert {"A": 2, "B": 4} in vs


def test_mangled_variant_name():
    name = mangled_variant_name("matmul", {"BLOCK_SIZE": 32, "NUM_WARPS": 4})
    # Keys sorted alphabetically: BLOCK_SIZE then NUM_WARPS
    assert name == "matmul__autotune_BLOCK_SIZE_32_NUM_WARPS_4"


def test_validate_requires_kernel():
    src = """
@autotune(B: [1])
fn x(a: i32) -> i32 { a }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    diags = validate_autotune(fn)
    assert any("requires @kernel" in d for d in diags)


def test_validate_clean_with_kernel():
    src = """
@kernel @autotune(B: [16, 32])
fn x(a: i32) -> i32 { a }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    diags = validate_autotune(fn)
    assert diags == []


def test_validate_oversized_product():
    """Cap is 16 variants. 5*5 = 25 should diag."""
    src = """
@kernel
@autotune(A: [1, 2, 3, 4, 5], B: [10, 20, 30, 40, 50])
fn x(a: i32) -> i32 { a }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    diags = validate_autotune(fn)
    assert any("trap 27001" in d for d in diags)


def test_validate_at_cap():
    """16 variants exactly is OK."""
    src = """
@kernel
@autotune(A: [1, 2, 3, 4], B: [10, 20, 30, 40])
fn x(a: i32) -> i32 { a }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    diags = validate_autotune(fn)
    # Product = 16 = cap, should pass
    assert not any("27001" in d for d in diags)


def test_collect_autotuned_fns():
    src = """
@kernel @autotune(A: [1]) fn k1(x: i32) -> i32 { x }
fn plain(x: i32) -> i32 { x }
@kernel @autotune(B: [2, 4]) fn k2(x: i32) -> i32 { x }
"""
    prog = parse(src)
    fns = collect_autotuned_fns(prog)
    assert [f.name for f in fns] == ["k1", "k2"]


def test_constants():
    assert TRAP_AUTOTUNE_OVERSIZED == 27001
    assert MAX_VARIANT_PRODUCT == 16


def test_empty_value_list_error():
    """@autotune(K: []) should produce a diagnostic."""
    src = """
@kernel @autotune(B: [])
fn x(a: i32) -> i32 { a }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    diags = validate_autotune(fn)
    assert any("empty" in d for d in diags)


# ----------------------------------------------------------------------
# Audit 28.8 A12 — malformed-attr diagnostics + dedup
# ----------------------------------------------------------------------
def test_parse_autotune_diagnostic_on_non_int_value():
    """Audit 28.8 A12: pre-fix `except ValueError: continue` silently
    dropped the entire key on a non-int value. The new behavior
    surfaces a diagnostic so the typo is fixable."""
    # We can't easily get the parser to emit a non-int attr from
    # source; construct the FnDecl by hand with a bogus attr.
    span = A.Span(0, 0)
    fn = A.FnDecl(
        span=span, name="k", generics=[], params=[],
        return_ty=A.TyName(span=span, name="i32"),
        where_clauses=[],
        body=A.Block(span=span, stmts=[],
                     final_expr=A.IntLit(span=span, value=0,
                                         type_suffix=None)),
        attrs=["kernel", "autotune", "autotune:BS=16,fast,32"],
        is_pub=False,
    )
    params, diags = parse_autotune_attrs(fn)
    # BS value list excludes the bad value but keeps the good ones.
    assert params == {"BS": [16, 32]}
    # And the bad value produced a diagnostic.
    assert any("fast" in d and "not an integer" in d for d in diags)


def test_parse_autotune_diagnostic_on_no_equals():
    """A missing `=` (e.g. `autotune:BS`) is malformed."""
    span = A.Span(0, 0)
    fn = A.FnDecl(
        span=span, name="k", generics=[], params=[],
        return_ty=A.TyName(span=span, name="i32"),
        where_clauses=[],
        body=A.Block(span=span, stmts=[],
                     final_expr=A.IntLit(span=span, value=0,
                                         type_suffix=None)),
        attrs=["kernel", "autotune", "autotune:NO_EQUALS"],
        is_pub=False,
    )
    params, diags = parse_autotune_attrs(fn)
    assert params == {}
    assert any("no `=`" in d for d in diags)


def test_parse_autotune_diagnostic_on_duplicate_key():
    span = A.Span(0, 0)
    fn = A.FnDecl(
        span=span, name="k", generics=[], params=[],
        return_ty=A.TyName(span=span, name="i32"),
        where_clauses=[],
        body=A.Block(span=span, stmts=[],
                     final_expr=A.IntLit(span=span, value=0,
                                         type_suffix=None)),
        attrs=[
            "kernel", "autotune",
            "autotune:BS=16,32",
            "autotune:BS=64",
        ],
        is_pub=False,
    )
    params, diags = parse_autotune_attrs(fn)
    assert params == {"BS": [16, 32]}
    assert any("duplicate parameter" in d for d in diags)


def test_autotune_variants_dedups():
    """Audit 28.8 A12: `@autotune(X: [1, 1, 2])` should generate
    2 variants, not 3, so duplicate-named variants don't collide."""
    vs = autotune_variants({"BS": [1, 1, 2]})
    assert vs == [{"BS": 1}, {"BS": 2}]


def test_stage35_autotune_variants_sorted_key_order():
    """Stage 35: variant generation order is stable regardless of attr order."""
    vs = autotune_variants({"Z": [9], "A": [1, 2]})
    names = [mangled_variant_name("k", cfg) for cfg in vs]
    assert names == [
        "k__autotune_A_1_Z_9",
        "k__autotune_A_2_Z_9",
    ]


def test_autotune_variant_count_dedups():
    """variant_count must agree with the deduped autotune_variants
    output (otherwise the cap check at 16 lies)."""
    assert variant_count({"BS": [1, 1, 2]}) == 2
    assert variant_count({"A": [1, 1], "B": [2, 2, 3]}) == 2


def test_validate_autotune_surfaces_parse_diags():
    """Audit 28.8 A12: validate_autotune must include the diagnostics
    from parse_autotune_attrs so the user sees the real cause."""
    span = A.Span(0, 0)
    fn = A.FnDecl(
        span=span, name="k", generics=[], params=[],
        return_ty=A.TyName(span=span, name="i32"),
        where_clauses=[],
        body=A.Block(span=span, stmts=[],
                     final_expr=A.IntLit(span=span, value=0,
                                         type_suffix=None)),
        attrs=["kernel", "autotune", "autotune:BS=16,fast,32"],
        is_pub=False,
    )
    diags = validate_autotune(fn)
    assert any("not an integer" in d for d in diags)


def test_validate_autotune_prog_runs_over_all_fns():
    """validate_autotune_prog is the program-level entry point so
    check.py can run a single call across all autotuned fns."""
    src = """
@kernel @autotune(B: [16, 32]) fn k1(a: i32) -> i32 { a }
@kernel @autotune(B: [4, 8])  fn k2(a: i32) -> i32 { a }
fn plain(a: i32) -> i32 { a }
"""
    prog = parse(src)
    diags = validate_autotune_prog(prog)
    # Both k1 and k2 are clean — no diags.
    assert diags == []


def test_validate_autotune_prog_collects_diags():
    """validate_autotune_prog returns the flat union across fns."""
    src = """
@autotune(B: [16]) fn no_kernel(a: i32) -> i32 { a }
@kernel @autotune(A: [1, 2, 3, 4, 5], C: [6, 7, 8, 9, 10]) fn too_many(a: i32) -> i32 { a }
"""
    prog = parse(src)
    diags = validate_autotune_prog(prog)
    # no_kernel produces a "requires @kernel" diagnostic
    assert any("requires @kernel" in d for d in diags)
    # too_many produces a cap-violation diagnostic (5*5 = 25 > 16)
    assert any("trap 27001" in d for d in diags)


def test_parse_autotune_attrs_dict_legacy_wrapper():
    """The dict-only alias preserves the old API for callers that
    don't care about diagnostics."""
    src = """
@kernel @autotune(B: [16, 32])
fn k(a: i32) -> i32 { a }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    params = parse_autotune_attrs_dict(fn)
    assert params == {"B": [16, 32]}


def test_c94_f2_autotune_int_digit_separators_preserved():
    """Stage 28.9 cycle 95 audit-R F2 regression (HIGH conf 90):
    `_parse_autotune_int` pre-fix used `t.value.split('_')[0]` to
    strip the type-suffix. But `_` is ALSO the digit-separator
    character, so `1_000_000` got split to `['1', '000', '000']`
    and only `'1'` survived — silently truncating
    `@autotune(BLOCK: [1_000_000])` to `[1]`. Now uses the lexer's
    pre-computed `t.int_value` which honours separators correctly."""
    src = """
@autotune(BLOCK: [1_000_000, 2_000, 16])
fn k(a: i32) -> i32 { a }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    params, _diags = parse_autotune_attrs(fn)
    assert params == {"BLOCK": [1_000_000, 2_000, 16]}, (
        f"expected digit-separator-preserving parse to give "
        f"[1000000, 2000, 16]; got {params}"
    )


def test_c94_f2_autotune_int_suffix_still_stripped():
    """C94-F2 regression: ensure typed literals still work — the
    int_value path strips the suffix correctly."""
    src = """
@autotune(BLOCK: [16_i32, 32_i32])
fn k(a: i32) -> i32 { a }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    params, _diags = parse_autotune_attrs(fn)
    assert params == {"BLOCK": [16, 32]}, (
        f"expected typed-literal parse to give [16, 32]; got {params}"
    )


def test_stage59_autotune_variant_names_for_returns_all_mangled():
    """Stage 59 follow-on / Tier 2 #8 polish: autotune_variant_names_for
    returns the full list of mangled variant names a fn would emit."""
    from helixc.frontend.autotune_expand import autotune_variant_names_for
    src = """
@autotune(BLOCK: [16, 32], WARPS: [2, 4])
@kernel
fn k(a: i32) -> i32 { a + BLOCK + WARPS }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    names = autotune_variant_names_for(fn)
    # Cartesian product → 4 variants.
    assert len(names) == 4
    # All variant names contain the original "k" prefix.
    assert all("k" in n for n in names)
    # Names are unique.
    assert len(set(names)) == 4


def test_stage59_autotune_variant_names_for_non_autotune_passthrough():
    """Stage 59 follow-on: a fn without @autotune @kernel returns
    [fn.name] — the singleton (pass-through case)."""
    from helixc.frontend.autotune_expand import autotune_variant_names_for
    src = """
fn plain(x: i32) -> i32 { x + 1 }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    assert autotune_variant_names_for(fn) == ["plain"]


def test_stage59_autotune_variant_count_for_matches_cartesian():
    """Stage 59 follow-on: autotune_variant_count_for is the
    Cartesian-product cardinality."""
    from helixc.frontend.autotune_expand import autotune_variant_count_for
    src = """
@autotune(A: [1, 2, 3], B: [10, 20])
@kernel
fn k(x: i32) -> i32 { x + A + B }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    # 3 * 2 = 6 variants.
    assert autotune_variant_count_for(fn) == 6


def test_stage59_autotune_expansion_summary_omits_non_autotune():
    """Stage 59 follow-on: autotune_expansion_summary returns a
    {fn_name: variant_count} dict for @autotune @kernel fns only,
    omitting non-autotune fns."""
    from helixc.frontend.autotune_expand import autotune_expansion_summary
    src = """
fn plain(x: i32) -> i32 { x }
@autotune(B: [8, 16])
@kernel
fn k1(x: i32) -> i32 { x + B }
@autotune(N: [4, 8, 16])
@kernel
fn k2(x: i32) -> i32 { x + N }
"""
    prog = parse(src)
    summary = autotune_expansion_summary(prog)
    assert summary == {"k1": 2, "k2": 3}
    assert "plain" not in summary


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
