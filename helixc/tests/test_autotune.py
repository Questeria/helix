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
    autotune_variants,
    variant_count,
    has_autotune,
    has_kernel,
    mangled_variant_name,
    validate_autotune,
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
    params = parse_autotune_attrs(fn)
    assert params == {"BLOCK_SIZE": [16, 32]}


def test_parse_two_key_autotune():
    src = """
@autotune(BLOCK_SIZE: [16, 32, 64], NUM_WARPS: [4, 8])
fn matmul(a: i32) -> i32 { a }
"""
    prog = parse(src)
    fn = next(it for it in prog.items if isinstance(it, A.FnDecl))
    params = parse_autotune_attrs(fn)
    assert params == {"BLOCK_SIZE": [16, 32, 64], "NUM_WARPS": [4, 8]}


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
