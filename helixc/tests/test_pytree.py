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
    is_diff_leaf,
    flatten_pytree,
    unflatten_pytree,
    pytree_depth,
    validate_pytree,
    TRAP_PYTREE_DEPTH,
    TRAP_PYTREE_NON_DIFF_LEAF,
    TRAP_PYTREE_CYCLE,
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


def test_unflatten_missing_path_raises():
    """Audit 28.8 A11: by default, a leaf path missing from
    `grads_by_path` raises ValueError. Pre-fix, missing paths silently
    defaulted to 0.0 — so a buggy AD pass with off-by-one paths would
    produce zero gradients with no signal."""
    src = """
struct Model { w1: f64, w2: f64 }
"""
    decls = _decls(src)
    grads = {"w1": 7.5}  # w2 missing
    with pytest.raises(ValueError, match="missing from gradients"):
        unflatten_pytree(decls["Model"], decls, grads)


def test_unflatten_missing_path_default_opt_in():
    """The pre-fix behaviour (missing path -> 0.0) is still reachable
    via `default=0.0`. This preserves the explicit-permissive API."""
    src = """
struct Model { w1: f64, w2: f64 }
"""
    decls = _decls(src)
    grads = {"w1": 7.5}
    out = unflatten_pytree(decls["Model"], decls, grads, default=0.0)
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
    assert TRAP_PYTREE_CYCLE == 26003
    assert MAX_DEPTH == 4
    assert DIFF_LEAF_PRIMS == frozenset({"f64", "f32", "bf16", "f16"})


# ----------------------------------------------------------------------
# Audit 28.8 B9 — pytree flatten/unflatten symmetry + cycle guard
# ----------------------------------------------------------------------
def test_is_diff_leaf_strict():
    """Audit 28.8 B9 (3): `is_diff_leaf` returns True ONLY for
    D-wrapped scalars. Bare `f64` returns False (use is_pytree_leaf if
    you want both)."""
    f64 = A.TyName(span=A.Span(0, 0), name="f64")
    d_f64 = A.TyGeneric(span=A.Span(0, 0), base="D", args=[f64])
    assert is_pytree_leaf(f64)  # bare float is a pytree leaf
    assert not is_diff_leaf(f64)  # but not a diff leaf
    assert is_pytree_leaf(d_f64)
    assert is_diff_leaf(d_f64)


def test_is_diff_leaf_rejects_int():
    """is_diff_leaf rejects integers / non-float types even if
    D-wrapped (D<i32> isn't a valid leaf type)."""
    i32 = A.TyName(span=A.Span(0, 0), name="i32")
    d_i32 = A.TyGeneric(span=A.Span(0, 0), base="D", args=[i32])
    assert not is_diff_leaf(i32)
    assert not is_diff_leaf(d_i32)


def test_pytree_depth_handles_cycle():
    """Audit 28.8 B9 (1): cyclic struct refs must NOT blow the Python
    recursion stack. pytree_depth returns a bound rather than
    raising — see flatten_pytree for the trap 26003 raise."""
    # Construct cycle manually since the parser won't tolerate forward
    # refs in struct fields the same way.
    span = A.Span(0, 0)
    a_decl = A.StructDecl(
        span=span, name="A", generics=[],
        fields=[A.FnParam(span=span, name="b",
                          ty=A.TyName(span=span, name="B"), is_mut=False),
                A.FnParam(span=span, name="w",
                          ty=A.TyName(span=span, name="f64"), is_mut=False)],
        is_pub=False,
    )
    b_decl = A.StructDecl(
        span=span, name="B", generics=[],
        fields=[A.FnParam(span=span, name="a",
                          ty=A.TyName(span=span, name="A"), is_mut=False),
                A.FnParam(span=span, name="b",
                          ty=A.TyName(span=span, name="f64"), is_mut=False)],
        is_pub=False,
    )
    decls = {"A": a_decl, "B": b_decl}
    # Must complete without RecursionError.
    d = pytree_depth(a_decl, decls)
    assert isinstance(d, int)
    assert d >= 0


def test_flatten_pytree_raises_on_cycle():
    """Audit 28.8 B9 (1): flatten_pytree raises trap 26003 when a
    cycle is detected (rather than letting Python's RecursionError
    surface)."""
    span = A.Span(0, 0)
    a_decl = A.StructDecl(
        span=span, name="Cyc", generics=[],
        fields=[A.FnParam(span=span, name="self_ref",
                          ty=A.TyName(span=span, name="Cyc"), is_mut=False)],
        is_pub=False,
    )
    decls = {"Cyc": a_decl}
    with pytest.raises(ValueError, match="26003"):
        flatten_pytree(a_decl, decls)


def test_unflatten_pytree_raises_on_non_diff_field():
    """Audit 28.8 B9 (2): unflatten_pytree no longer silently emits
    `None` for non-diff-non-struct fields. It raises (mirroring
    flatten's behavior) so `unflatten(flatten(x))` is well-defined."""
    span = A.Span(0, 0)
    bad_decl = A.StructDecl(
        span=span, name="Bad", generics=[],
        fields=[A.FnParam(span=span, name="w",
                          ty=A.TyName(span=span, name="f64"), is_mut=False),
                # Non-pytree field: i32 (an integer is not differentiable)
                A.FnParam(span=span, name="bookkeeping",
                          ty=A.TyName(span=span, name="i32"),
                          is_mut=False)],
        is_pub=False,
    )
    decls = {"Bad": bad_decl}
    with pytest.raises(ValueError, match="26002"):
        unflatten_pytree(bad_decl, decls, {"w": 1.0})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
