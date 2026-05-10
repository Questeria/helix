"""Stage 26: tests for JAX-style pytrees (struct flatten/unflatten)."""

from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from helixc.frontend.parser import parse
from helixc.frontend import ast_nodes as A
from helixc.frontend.pytree import (
    PytreeLeaf,
    is_pytree_leaf,
    flatten_pytree,
    unflatten_pytree,
    pytree_depth,
    validate_pytree,
    TRAP_PYTREE_DEPTH,
    TRAP_PYTREE_NON_DIFF_LEAF,
    MAX_DEPTH,
    DIFF_LEAF_PRIMS,
)


def _decls(src: str) -> dict:
    """Parse src and return a name->StructDecl dict."""
    prog = parse(src)
    return {it.name: it for it in prog.items if isinstance(it, A.StructDecl)}


def test_is_pytree_leaf_floats():
    for prim in ("f64", "f32", "bf16", "f16"):
        ty = A.TyName(span=A.Span(0, 0), name=prim)
        assert is_pytree_leaf(ty)


def test_is_pytree_leaf_int_not_leaf():
    ty = A.TyName(span=A.Span(0, 0), name="i32")
    assert not is_pytree_leaf(ty)


def test_is_pytree_leaf_d_wrapped():
    inner = A.TyName(span=A.Span(0, 0), name="f64")
    ty = A.TyGeneric(span=A.Span(0, 0), base="D", args=[inner])
    assert is_pytree_leaf(ty)


def test_flatten_flat_model():
    """A flat struct has one leaf per field."""
    src = """
struct Model { w1: f64, w2: f64 }
"""
    decls = _decls(src)
    leaves = flatten_pytree(decls["Model"], decls)
    assert len(leaves) == 2
    assert leaves[0].path == "w1"
    assert leaves[0].ty_name == "f64"
    assert leaves[1].path == "w2"
    assert all(not l.is_diff for l in leaves)


def test_flatten_d_wrapped_leaves_marks_diff():
    src = """
struct Model { w: D<f64> }
"""
    decls = _decls(src)
    leaves = flatten_pytree(decls["Model"], decls)
    assert len(leaves) == 1
    assert leaves[0].is_diff
    assert leaves[0].ty_name == "f64"


def test_flatten_nested_struct():
    """Nested struct: paths are dot-joined."""
    src = """
struct Layer { w: f64, b: f64 }
struct Net { l1: Layer, l2: Layer }
"""
    decls = _decls(src)
    leaves = flatten_pytree(decls["Net"], decls)
    paths = [l.path for l in leaves]
    assert paths == ["l1.w", "l1.b", "l2.w", "l2.b"]


def test_flatten_depth_3():
    """Depth-3 nesting still works."""
    src = """
struct Inner { x: f64 }
struct Mid { i: Inner }
struct Outer { m: Mid }
"""
    decls = _decls(src)
    leaves = flatten_pytree(decls["Outer"], decls)
    assert len(leaves) == 1
    assert leaves[0].path == "m.i.x"


def test_pytree_depth():
    """depth = max nesting; flat struct -> 0; nested -> deeper."""
    src = """
struct Flat { x: f64 }
struct Nest1 { f: Flat }
struct Nest2 { n: Nest1 }
"""
    decls = _decls(src)
    assert pytree_depth(decls["Flat"], decls) == 0
    assert pytree_depth(decls["Nest1"], decls) == 1
    assert pytree_depth(decls["Nest2"], decls) == 2


def test_flatten_rejects_non_diff_leaf():
    """A struct with an integer field is not a valid pytree (trap 26002)."""
    src = """
struct Bad { x: f64, n: i32 }
"""
    decls = _decls(src)
    with pytest.raises(ValueError, match="26002"):
        flatten_pytree(decls["Bad"], decls)


def test_flatten_rejects_too_deep():
    """Depth > MAX_DEPTH = 4 should raise trap 26001."""
    # Build a 6-deep nested struct programmatically — easier than the
    # parser since we'd need 6 distinct names.
    src = """
struct A { x: f64 }
struct B { a: A }
struct C { b: B }
struct D { c: C }
struct E { d: D }
struct F { e: E }
"""
    decls = _decls(src)
    # F nests through E->D->C->B->A: depth 5, > MAX_DEPTH 4
    with pytest.raises(ValueError, match="26001"):
        flatten_pytree(decls["F"], decls)


def test_unflatten_round_trip():
    """flatten then unflatten produces same-shape dict."""
    src = """
struct Layer { w: f64, b: f64 }
struct Net { l1: Layer, l2: Layer }
"""
    decls = _decls(src)
    leaves = flatten_pytree(decls["Net"], decls)
    # Simulate gradients
    grads = {l.path: float(i + 1) for i, l in enumerate(leaves)}
    out = unflatten_pytree(decls["Net"], decls, grads)
    assert out["l1"]["w"] == 1.0
    assert out["l1"]["b"] == 2.0
    assert out["l2"]["w"] == 3.0
    assert out["l2"]["b"] == 4.0


def test_unflatten_missing_path_defaults_zero():
    """Paths not in the grads dict default to 0.0."""
    src = """
struct Model { w1: f64, w2: f64 }
"""
    decls = _decls(src)
    grads = {"w1": 7.5}  # w2 missing
    out = unflatten_pytree(decls["Model"], decls, grads)
    assert out["w1"] == 7.5
    assert out["w2"] == 0.0


def test_validate_clean():
    src = "struct M { w: f64 }"
    decls = _decls(src)
    assert validate_pytree(decls["M"], decls) == []


def test_validate_dirty():
    src = "struct B { w: f64, n: i32 }"
    decls = _decls(src)
    diags = validate_pytree(decls["B"], decls)
    assert diags
    assert "26002" in diags[0]


def test_constants():
    assert TRAP_PYTREE_DEPTH == 26001
    assert TRAP_PYTREE_NON_DIFF_LEAF == 26002
    assert MAX_DEPTH == 4
    assert DIFF_LEAF_PRIMS == frozenset({"f64", "f32", "bf16", "f16"})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
