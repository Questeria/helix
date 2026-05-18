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
    flatten_pytree_param,
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


def test_stage35_flatten_pytree_param_prefixes_model_name():
    src = """
struct Layer { w: f32 }
struct Model { layer: Layer, bias: f32 }
"""
    decls = _decls(src)
    param = A.FnParam(
        span=A.Span(0, 0),
        name="model",
        ty=A.TyName(span=A.Span(0, 0), name="Model"),
    )
    leaves = flatten_pytree_param(param, decls)
    assert [l.path for l in leaves] == ["model.layer.w", "model.bias"]
    assert [l.ty_name for l in leaves] == ["f32", "f32"]


def test_stage35_flatten_pytree_param_accepts_scalar_param():
    param = A.FnParam(
        span=A.Span(0, 0),
        name="x",
        ty=A.TyName(span=A.Span(0, 0), name="f32"),
    )
    leaves = flatten_pytree_param(param, {})
    assert leaves == [PytreeLeaf(path="x", ty_name="f32", is_diff=False)]


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


# ----------------------------------------------------------------------
# Audit 28.8 cycle 2 (deferred observation #17) — _unflatten depth guard
# ----------------------------------------------------------------------
def test_unflatten_pytree_too_deep_traps_26001():
    """Deferred observation #17: pre-fix `_unflatten` had a cycle
    guard (`_visited`) but no MAX_DEPTH check. A straight-line nested
    chain that exceeded MAX_DEPTH would RecursionError instead of a
    clean trap. Now mirrors `flatten_pytree`'s 26001 path."""
    span = A.Span(0, 0)
    # Build a chain: D1 -> D2 -> D3 -> D4 -> D5 -> D6 (depth 5,
    # MAX_DEPTH=4). Each non-leaf holds an inner struct reference.
    decls: dict[str, A.StructDecl] = {}
    for i in range(6, 0, -1):
        if i == 6:
            fields = [A.FnParam(span=span, name="x",
                                ty=A.TyName(span=span, name="f64"),
                                is_mut=False)]
        else:
            fields = [A.FnParam(span=span, name="inner",
                                ty=A.TyName(span=span, name=f"D{i+1}"),
                                is_mut=False)]
        decls[f"D{i}"] = A.StructDecl(
            span=span, name=f"D{i}", generics=[],
            fields=fields, is_pub=False,
        )
    with pytest.raises(ValueError, match="26001"):
        unflatten_pytree(decls["D1"], decls, {}, default=0.0)


def test_tree_map_scales_all_leaves():
    """Stage 59 follow-on / Tier 2 #7 polish: `tree_map(decl, struct_decls,
    leaves_by_path, leaf_fn)` applies leaf_fn to each value and
    rebuilds the nested struct shape."""
    from helixc.frontend.pytree import tree_map
    span = A.Span(0, 0)
    # struct Point { x: f64, y: f64 }
    point = A.StructDecl(
        span=span, name="Point", generics=[], is_pub=False,
        fields=[
            A.FnParam(span=span, name="x",
                       ty=A.TyName(span=span, name="f64"),
                       is_mut=False),
            A.FnParam(span=span, name="y",
                       ty=A.TyName(span=span, name="f64"),
                       is_mut=False),
        ],
    )
    leaves = {"x": 10.0, "y": 20.0}
    scaled = tree_map(point, {"Point": point}, leaves, lambda v: v * 0.1)
    assert scaled == {"x": 1.0, "y": 2.0}


def test_tree_map_preserves_nested_shape():
    """Stage 59 follow-on: tree_map round-trips through nested struct."""
    from helixc.frontend.pytree import tree_map
    span = A.Span(0, 0)
    # struct Inner { v: f64 }
    inner = A.StructDecl(
        span=span, name="Inner", generics=[], is_pub=False,
        fields=[A.FnParam(span=span, name="v",
                          ty=A.TyName(span=span, name="f64"),
                          is_mut=False)],
    )
    # struct Outer { inner: Inner, b: f64 }
    outer = A.StructDecl(
        span=span, name="Outer", generics=[], is_pub=False,
        fields=[
            A.FnParam(span=span, name="inner",
                       ty=A.TyName(span=span, name="Inner"),
                       is_mut=False),
            A.FnParam(span=span, name="b",
                       ty=A.TyName(span=span, name="f64"),
                       is_mut=False),
        ],
    )
    decls = {"Inner": inner, "Outer": outer}
    leaves = {"inner.v": 5.0, "b": 3.0}
    out = tree_map(outer, decls, leaves, lambda v: v + 1.0)
    # Expected: {"inner": {"v": 6.0}, "b": 4.0}
    assert out["b"] == 4.0
    assert out["inner"]["v"] == 6.0


def test_tree_leaves_canonical_order():
    """Stage 59 follow-on: tree_leaves returns leaf values in
    sorted-by-path order (deterministic)."""
    from helixc.frontend.pytree import tree_leaves
    leaves = {"c": 3.0, "a": 1.0, "b": 2.0}
    assert tree_leaves(leaves) == [1.0, 2.0, 3.0]


def test_tree_paths_canonical_order():
    """Stage 59 follow-on: tree_paths returns paths in same sorted
    order as tree_leaves. zip(paths, leaves) round-trips the dict."""
    from helixc.frontend.pytree import tree_leaves, tree_paths
    leaves = {"a.x": 1.0, "a.y": 2.0, "b": 3.0}
    paths = tree_paths(leaves)
    values = tree_leaves(leaves)
    assert paths == ["a.x", "a.y", "b"]
    assert dict(zip(paths, values)) == leaves


def test_tree_leaves_empty():
    """Stage 59 follow-on: tree_leaves on empty dict returns []."""
    from helixc.frontend.pytree import tree_leaves
    assert tree_leaves({}) == []


def test_tree_pytree_sgd_step_integration():
    """Stage 59 follow-on / Tier 2 #7: integration test composing all
    4 pytree functional primitives in a realistic SGD step.

    Workflow:
      1. flatten_pytree(model_decl) → declares the leaf structure
      2. tree_zip(params, grads, p - lr*g) → SGD update
      3. tree_reduce(new_params, lambda a, p: a + abs(p), 0.0) →
         L1 parameter norm for convergence check
      4. tree_equal(old_params, new_params, eq_fn=approx) →
         did we converge?

    Demonstrates that the Python pytree functional API can drive
    a complete optimizer step end-to-end, mirroring JAX/Optax."""
    from helixc.frontend.pytree import (
        tree_map, tree_reduce, tree_zip, tree_equal,
        flatten_pytree,
    )
    span = A.Span(0, 0)
    # struct Linear { w: f64, b: f64 } — 2-param linear layer.
    layer = A.StructDecl(
        span=span, name="Linear", generics=[], is_pub=False,
        fields=[
            A.FnParam(span=span, name="w",
                       ty=A.TyName(span=span, name="f64"),
                       is_mut=False),
            A.FnParam(span=span, name="b",
                       ty=A.TyName(span=span, name="f64"),
                       is_mut=False),
        ],
    )
    decls = {"Linear": layer}
    # Verify the leaf paths
    leaves = flatten_pytree(layer, decls)
    leaf_paths = sorted(leaf.path for leaf in leaves)
    assert leaf_paths == ["b", "w"]

    # Initial params and computed gradients
    params = {"w": 1.0, "b": 2.0}
    grads = {"w": 0.5, "b": 0.25}
    lr = 0.01

    # SGD step via tree_zip
    new_params = tree_zip(layer, decls, params, grads,
                           lambda p, g: p - lr * g)
    # new_params has the nested struct shape
    assert new_params == {"w": 0.995, "b": 1.9975}

    # L1 norm of new params via tree_reduce
    l1_norm = tree_reduce(new_params, lambda a, p: a + abs(p), 0.0)
    assert abs(l1_norm - 2.9925) < 1e-9

    # Convergence check: did params change?
    converged = tree_equal(params, new_params,
                            eq_fn=lambda a, b: abs(a - b) < 1e-9)
    assert not converged  # 1 step shouldn't converge

    # After zero-grad step, params should be unchanged
    zero_grads = {"w": 0.0, "b": 0.0}
    same_params = tree_zip(layer, decls, params, zero_grads,
                             lambda p, g: p - lr * g)
    converged_after_zero = tree_equal(
        params, same_params,
        eq_fn=lambda a, b: abs(a - b) < 1e-9)
    assert converged_after_zero  # zero grad → no change

    # tree_map demo: scale gradients (e.g., clip-by-scale)
    scaled_grads = tree_map(layer, decls, grads, lambda g: g * 0.5)
    assert scaled_grads == {"w": 0.25, "b": 0.125}


def test_tree_equal_same_leaves():
    """Stage 59 follow-on: tree_equal returns True for identical leaves."""
    from helixc.frontend.pytree import tree_equal
    a = {"x": 1.0, "y": 2.0}
    b = {"x": 1.0, "y": 2.0}
    assert tree_equal(a, b)


def test_tree_equal_different_leaves():
    """Stage 59 follow-on: tree_equal returns False on value mismatch."""
    from helixc.frontend.pytree import tree_equal
    a = {"x": 1.0, "y": 2.0}
    b = {"x": 1.0, "y": 2.5}
    assert not tree_equal(a, b)


def test_tree_equal_different_shape():
    """Stage 59 follow-on: tree_equal returns False on key mismatch."""
    from helixc.frontend.pytree import tree_equal
    a = {"x": 1.0, "y": 2.0}
    b = {"x": 1.0}
    assert not tree_equal(a, b)


def test_tree_equal_approx_eq_fn():
    """Stage 59 follow-on: tree_equal accepts a custom eq_fn for
    approximate-equality (floating-point tolerance)."""
    from helixc.frontend.pytree import tree_equal
    a = {"x": 1.0, "y": 2.0}
    b = {"x": 1.0 + 1e-12, "y": 2.0 - 1e-12}
    # Strict == fails (the values differ slightly).
    assert not tree_equal(a, b)
    # Approximate equality succeeds.
    assert tree_equal(a, b, eq_fn=lambda x, y: abs(x - y) < 1e-9)


def test_tree_zip_gradient_update():
    """Stage 59 follow-on: tree_zip combines params + grads via
    SGD-style update `p - 0.01 * g`. Canonical use case."""
    from helixc.frontend.pytree import tree_zip
    span = A.Span(0, 0)
    model = A.StructDecl(
        span=span, name="Model", generics=[], is_pub=False,
        fields=[
            A.FnParam(span=span, name="w",
                       ty=A.TyName(span=span, name="f64"),
                       is_mut=False),
            A.FnParam(span=span, name="b",
                       ty=A.TyName(span=span, name="f64"),
                       is_mut=False),
        ],
    )
    params = {"w": 1.0, "b": 2.0}
    grads = {"w": 0.5, "b": 0.25}
    updated = tree_zip(model, {"Model": model}, params, grads,
                       lambda p, g: p - 0.01 * g)
    assert updated == {"w": 0.995, "b": 1.9975}


def test_tree_zip_missing_key_raises():
    """Stage 59 follow-on: tree_zip raises on missing path by default."""
    from helixc.frontend.pytree import tree_zip
    span = A.Span(0, 0)
    pt = A.StructDecl(
        span=span, name="P", generics=[], is_pub=False,
        fields=[A.FnParam(span=span, name="x",
                          ty=A.TyName(span=span, name="f64"),
                          is_mut=False)],
    )
    a = {"x": 1.0}
    b = {}  # missing 'x'
    with pytest.raises(ValueError, match="missing in b"):
        tree_zip(pt, {"P": pt}, a, b, lambda p, q: p + q)


def test_tree_reduce_sum():
    """Stage 59 follow-on: tree_reduce sums all leaf values."""
    from helixc.frontend.pytree import tree_reduce
    leaves = {"x": 1.0, "y": 2.0, "z": 3.0}
    total = tree_reduce(leaves, lambda acc, v: acc + v, 0.0)
    assert total == 6.0


def test_tree_reduce_count():
    """Stage 59 follow-on: tree_reduce counts leaves."""
    from helixc.frontend.pytree import tree_reduce
    leaves = {"a.b.c": 1, "a.b.d": 2, "e": 3, "f.g": 4}
    n = tree_reduce(leaves, lambda acc, _: acc + 1, 0)
    assert n == 4


def test_tree_reduce_deterministic_order():
    """Stage 59 follow-on: tree_reduce iterates leaves in sorted-by-
    path order for determinism (matters when reduce_fn is non-
    commutative)."""
    from helixc.frontend.pytree import tree_reduce
    leaves = {"c": 3, "a": 1, "b": 2}
    # Concatenate values as a string in iteration order.
    seq = tree_reduce(leaves, lambda acc, v: acc + str(v), "")
    # Sorted keys: a, b, c → "1" + "2" + "3" = "123"
    assert seq == "123"


def test_tree_paths_matching_returns_sorted_paths():
    """Stage 59 follow-on / Tier 2 #7 polish: tree_paths_matching
    returns sorted list of paths whose values match predicate."""
    from helixc.frontend.pytree import tree_paths_matching
    leaves = {"w3": 0.0, "w1": 1.0, "w2": -2.0, "b": 0.0}
    # Zero leaves, sorted alphabetically.
    assert tree_paths_matching(leaves, lambda v: v == 0.0) == ["b", "w3"]
    # All leaves.
    assert tree_paths_matching(leaves, lambda v: True) == ["b", "w1", "w2", "w3"]
    # No matches.
    assert tree_paths_matching(leaves, lambda v: False) == []


def test_tree_to_canonical_json_deterministic():
    """Stage 59 follow-on: tree_to_canonical_json produces the same
    string for the same input regardless of dict insertion order."""
    from helixc.frontend.pytree import tree_to_canonical_json
    a = {"w1": 1.0, "w2": 2.0, "b": 0.5}
    b = {"b": 0.5, "w2": 2.0, "w1": 1.0}
    assert tree_to_canonical_json(a) == tree_to_canonical_json(b)


def test_tree_to_canonical_json_valid_json():
    """Stage 59 follow-on: output parses back to an equivalent dict."""
    import json
    from helixc.frontend.pytree import tree_to_canonical_json
    leaves = {"w": 1.5, "b": 0.0}
    s = tree_to_canonical_json(leaves)
    parsed = json.loads(s)
    assert parsed == {"w": 1.5, "b": 0.0}


def test_tree_to_csv_basic_format():
    """Stage 59 follow-on / Tier 2 #7 polish: tree_to_csv exports
    a pytree as a two-column CSV with 'path,value' header and one
    row per leaf sorted by path."""
    from helixc.frontend.pytree import tree_to_csv
    leaves = {"w2": -2.0, "w1": 1.0, "b": 0.5}
    csv = tree_to_csv(leaves)
    lines = csv.split("\n")
    assert lines[0] == "path,value"
    # Sorted: b, w1, w2
    assert lines[1] == "b,0.5"
    assert lines[2] == "w1,1.0"
    assert lines[3] == "w2,-2.0"


def test_tree_to_csv_handles_commas_in_values():
    """Stage 59 follow-on: values with commas (e.g., tuple reprs)
    are CSV-quoted with embedded-quote escaping."""
    from helixc.frontend.pytree import tree_to_csv
    leaves = {"w": (1.0, 2.0)}  # repr → "(1.0, 2.0)" has comma
    csv = tree_to_csv(leaves)
    # The line should be quoted because the value contains commas.
    rows = csv.split("\n")[1:]
    assert len(rows) == 1
    assert rows[0].startswith("w,")
    # The value portion is the quoted repr.
    val_portion = rows[0].split(",", 1)[1]
    assert val_portion.startswith('"')


def test_tree_to_csv_empty():
    """Stage 59 follow-on: empty pytree → header only."""
    from helixc.frontend.pytree import tree_to_csv
    assert tree_to_csv({}) == "path,value"


def test_tree_csv_round_trips_basic():
    """Stage 59 follow-on / Tier 2 #7 polish: round-trip pin —
    tree_from_csv(tree_to_csv(d)) == d for basic float/int dicts."""
    from helixc.frontend.pytree import tree_to_csv, tree_from_csv
    original = {"w1": 1.5, "w2": -2.25, "b": 0.0}
    csv = tree_to_csv(original)
    restored = tree_from_csv(csv)
    assert restored == original


def test_tree_csv_round_trips_with_tuple_values():
    """Stage 59 follow-on: round-trip preserves tuple values via
    repr+literal_eval."""
    from helixc.frontend.pytree import tree_to_csv, tree_from_csv
    original = {"shape": (3, 4, 5), "x": 1.0}
    csv = tree_to_csv(original)
    restored = tree_from_csv(csv)
    assert restored == original
    assert isinstance(restored["shape"], tuple)


def test_tree_from_csv_empty():
    """Stage 59 follow-on: empty string + header-only CSV both
    return empty dict."""
    from helixc.frontend.pytree import tree_from_csv
    assert tree_from_csv("") == {}
    assert tree_from_csv("path,value") == {}


def test_tree_canonical_json_round_trips():
    """Stage 59 follow-on / Tier 2 #7 polish: round-trip pin.
    tree_from_canonical_json(tree_to_canonical_json(d)) == d for
    dicts with JSON-native values."""
    from helixc.frontend.pytree import (
        tree_to_canonical_json, tree_from_canonical_json,
    )
    original = {"w1": 1.5, "w2": -2.25, "b": 0.0, "label": "ok"}
    serialized = tree_to_canonical_json(original)
    restored = tree_from_canonical_json(serialized)
    assert restored == original


def test_tree_from_canonical_json_handles_empty():
    """Stage 59 follow-on: empty pytree round-trips through empty
    JSON object."""
    from helixc.frontend.pytree import (
        tree_to_canonical_json, tree_from_canonical_json,
    )
    s = tree_to_canonical_json({})
    assert tree_from_canonical_json(s) == {}


def test_tree_to_canonical_json_sorted_keys():
    """Stage 59 follow-on: output keys appear in sorted order in the
    raw JSON string (not just in the parsed dict)."""
    from helixc.frontend.pytree import tree_to_canonical_json
    leaves = {"z": 1, "a": 2, "m": 3}
    s = tree_to_canonical_json(leaves)
    # 'a' must appear before 'm' before 'z' in the raw string.
    assert s.index('"a"') < s.index('"m"') < s.index('"z"')


def test_tree_select_paths_extracts_subset():
    """Stage 59 follow-on / Tier 2 #7 polish: tree_select_paths returns
    a new dict with only the entries whose path is in the path list."""
    from helixc.frontend.pytree import tree_select_paths
    leaves = {"w1": 1.0, "w2": 2.0, "b": 0.5}
    sub = tree_select_paths(leaves, ["w1", "b"])
    assert sub == {"w1": 1.0, "b": 0.5}


def test_tree_select_paths_silently_skips_missing():
    """Stage 59 follow-on: paths not in source dict are silently
    skipped (use tree_size to detect)."""
    from helixc.frontend.pytree import tree_select_paths, tree_size
    leaves = {"w": 1.0}
    sub = tree_select_paths(leaves, ["w", "missing"])
    assert sub == {"w": 1.0}
    assert tree_size(sub) == 1


def test_tree_select_paths_empty_list():
    """Stage 59 follow-on: empty path list returns empty dict."""
    from helixc.frontend.pytree import tree_select_paths
    assert tree_select_paths({"w": 1.0}, []) == {}


def test_tree_count_matches_predicate():
    """Stage 59 follow-on / Tier 2 #7 polish: tree_count returns the
    number of leaves where predicate(value) is True."""
    from helixc.frontend.pytree import tree_count
    leaves = {"w1": 1.0, "w2": -2.0, "w3": 0.0, "b": 3.5}
    # Count positive leaves
    assert tree_count(leaves, lambda v: v > 0) == 2
    # Count zero leaves
    assert tree_count(leaves, lambda v: v == 0.0) == 1
    # Count all (always true)
    assert tree_count(leaves, lambda v: True) == 4
    # Count none (always false)
    assert tree_count(leaves, lambda v: False) == 0
    # Empty dict
    assert tree_count({}, lambda v: True) == 0


def test_tree_filter_returns_matching_subset():
    """Stage 59 follow-on: tree_filter returns a new dict with only
    paths where predicate(value) is True; original dict unmodified."""
    from helixc.frontend.pytree import tree_filter
    leaves = {"w1": 1.0, "w2": -2.0, "w3": 0.0, "b": 3.5}
    positive = tree_filter(leaves, lambda v: v > 0)
    assert positive == {"w1": 1.0, "b": 3.5}
    # Original untouched
    assert leaves == {"w1": 1.0, "w2": -2.0, "w3": 0.0, "b": 3.5}
    # Empty result when no match
    assert tree_filter(leaves, lambda v: False) == {}


def test_tree_filter_composes_with_tree_size():
    """Stage 59 follow-on: tree_size(tree_filter(d, p)) ==
    tree_count(d, p) — the composition identity."""
    from helixc.frontend.pytree import tree_filter, tree_size, tree_count
    leaves = {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
    p = lambda v: v % 2 == 1  # odd
    assert tree_size(tree_filter(leaves, p)) == tree_count(leaves, p)
    assert tree_size(tree_filter(leaves, p)) == 3  # 1, 3, 5


def test_tree_size_counts_leaves():
    """Stage 59 follow-on / Tier 2 #7 polish: tree_size returns the
    number of leaves (len of the dict, but spelled clearly)."""
    from helixc.frontend.pytree import tree_size
    assert tree_size({}) == 0
    assert tree_size({"w": 1.0}) == 1
    assert tree_size({"w1": 1.0, "w2": 2.0, "b": 0.0}) == 3


def test_tree_diff_empty_when_equal():
    """Stage 59 follow-on: tree_diff returns [] iff trees are equal."""
    from helixc.frontend.pytree import tree_diff, tree_equal
    a = {"w": 1.0, "b": 0.0}
    b = {"w": 1.0, "b": 0.0}
    assert tree_diff(a, b) == []
    assert tree_equal(a, b)


def test_tree_diff_reports_differing_paths():
    """Stage 59 follow-on: tree_diff returns sorted paths where leaves
    diverge (in value) or where a path is in only one tree."""
    from helixc.frontend.pytree import tree_diff
    a = {"w": 1.0, "b": 0.0, "only_a": 9.0}
    b = {"w": 2.0, "b": 0.0, "only_b": 9.0}
    # 'w' differs in value, 'only_a' missing in b, 'only_b' missing in a
    assert tree_diff(a, b) == ["only_a", "only_b", "w"]


def test_tree_diff_with_approx_eq_fn():
    """Stage 59 follow-on: custom eq_fn enables float-tolerance diff —
    leaves within tolerance are considered equal."""
    from helixc.frontend.pytree import tree_diff
    a = {"w": 1.0, "b": 0.5}
    b = {"w": 1.0 + 1e-12, "b": 0.6}
    approx = lambda x, y: abs(x - y) < 1e-9
    # 'w' equal under approx, 'b' diverges (|0.5-0.6|=0.1 > 1e-9)
    assert tree_diff(a, b, eq_fn=approx) == ["b"]


def test_tree_hash_deterministic():
    """Stage 59 follow-on / Tier 2 #7 polish: tree_hash returns the
    same SHA-256 for the same leaves dict, across calls."""
    from helixc.frontend.pytree import tree_hash
    leaves = {"w1": 1.5, "w2": -2.25, "b": 0.0}
    h1 = tree_hash(leaves)
    h2 = tree_hash(leaves)
    assert h1 == h2
    # SHA-256 hex is 64 chars
    assert len(h1) == 64
    assert all(c in "0123456789abcdef" for c in h1)


def test_tree_hash_sort_order_independent():
    """Stage 59 follow-on: tree_hash sorts by path so dict insertion
    order doesn't change the hash. Two dicts with identical key→value
    mappings hash identically regardless of how they were built."""
    from helixc.frontend.pytree import tree_hash
    a = {"w1": 1.0, "w2": 2.0, "b": 0.5}
    b = {"b": 0.5, "w2": 2.0, "w1": 1.0}  # different insertion order
    assert tree_hash(a) == tree_hash(b)


def test_tree_hash_different_values_differ():
    """Stage 59 follow-on: changing any leaf value produces a
    different hash — the basic invariant of content addressing."""
    from helixc.frontend.pytree import tree_hash
    base = {"w": 1.0, "b": 0.0}
    diff_value = {"w": 1.0 + 1e-9, "b": 0.0}
    diff_key = {"w": 1.0, "c": 0.0}
    empty = {}
    assert tree_hash(base) != tree_hash(diff_value)
    assert tree_hash(base) != tree_hash(diff_key)
    assert tree_hash(base) != tree_hash(empty)
    # Empty dict still produces valid SHA-256
    assert len(tree_hash(empty)) == 64


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
