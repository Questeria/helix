"""
helixc/frontend/pytree.py — Stage 26: JAX-style pytrees.

A "pytree" is a nested struct (or struct-of-structs-of-...) where the
leaves are differentiable scalar / tensor values. JAX's killer feature
is that `grad(loss)(model)` works when `model` is an arbitrary nested
struct — pytree machinery walks the struct, treats each scalar leaf as
its own AD input, and zips the gradients back into the same struct
shape.

This module provides the Python-side spec:
  * flatten_pytree(decl, struct_decls, path="") -> list of (path, leaf_ty)
    walks a StructDecl recursively, yielding one entry per scalar leaf.
  * flatten_pytree_param(param, struct_decls) -> list of PytreeLeaf
    prefixes a function parameter name onto every leaf path, e.g.
    model.layer.w. This is the static naming bridge used by AD surfaces.
  * unflatten_pytree(decl, gradients_by_path, default=...) -> dict
    given a path -> gradient mapping, reassembles a same-shape dict.
    By default raises ValueError on missing paths so AD-pass bugs that
    fail to populate a leaf are loud (Audit 28.8 A11). Pass
    `default=0.0` (or any sentinel) to opt into the pre-fix zero-fill
    behavior.
  * is_pytree_leaf(ty) — True for f64 / f32 / bf16 / TyDiff(scalar).
  * is_diff_leaf(ty) — True ONLY when the field is `D<f*>`-wrapped.
    Audit 28.8 B9 (3): passes that only want true differentiable leaves
    should use is_diff_leaf; passes that want any pytree-shaped leaf
    should use is_pytree_leaf. Pre-fix, both lived under the single
    is_pytree_leaf predicate so non-D-wrapped floats counted as
    differentiable, silently allocating gradients that wouldn't
    propagate.
  * pytree_depth(decl, struct_decls) — recursion depth (cycle-safe).
  * Trap 26001 — pytree depth > 4 (Phase-0 cap).
  * Trap 26002 — pytree leaf type not differentiable.
  * Trap 26003 — cyclic struct reference reached MAX_DEPTH
    (Audit 28.8 B9 (1)).

The runtime emission (grad_rev_all over a struct value) is bootstrap-
side; this module gives the typechecker + AD pass a static view.

License: Apache 2.0
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import ast_nodes as A


# Trap-id reservations
TRAP_PYTREE_DEPTH = 26001
TRAP_PYTREE_NON_DIFF_LEAF = 26002
TRAP_PYTREE_CYCLE = 26003


MAX_DEPTH = 4
DIFF_LEAF_PRIMS = frozenset({"f64", "f32", "bf16", "f16"})


# Sentinel used by unflatten_pytree when the caller has not specified
# a default. We can't use `None` (a legitimate "no gradient" value for
# non-diff fields), and we can't use a bare ValueError instance (mypy
# would complain). A unique object identity does the job.
_RAISE_ON_MISSING = object()


@dataclass(frozen=True)
class PytreeLeaf:
    """One leaf in a pytree walk. `path` is dot-joined field names
    like 'layer1.w' for grandfather struct `Model` -> field `layer1`
    (struct) -> field `w` (f64)."""
    path: str
    ty_name: str            # "f64", "f32", etc.
    is_diff: bool = False   # True iff parameter type was D<...>


def is_pytree_leaf(ty: A.TyNode) -> bool:
    """A leaf is a primitive float type (optionally wrapped in D<>).

    Note: passes that ONLY want gradient-bearing leaves should use
    `is_diff_leaf` instead — `is_pytree_leaf` returns True for both
    `f64` and `D<f64>`, which collapses two semantically-distinct
    cases (Audit 28.8 B9 (3))."""
    if isinstance(ty, A.TyName):
        return ty.name in DIFF_LEAF_PRIMS
    if isinstance(ty, A.TyGeneric) and ty.base == "D" and len(ty.args) == 1:
        return is_pytree_leaf(ty.args[0])
    return False


def is_diff_leaf(ty: A.TyNode) -> bool:
    """A diff leaf is a `D<scalar-float>`-wrapped value. Distinct from
    `is_pytree_leaf` (which also accepts bare scalar floats). Use this
    when only D-wrapped fields should participate in AD.

    Audit 28.8 B9 (3) split: pre-fix, downstream passes used
    `is_pytree_leaf` and ended up allocating gradients for bare floats
    that never propagated through binops — silent inconsistency."""
    return isinstance(ty, A.TyGeneric) and ty.base == "D" \
        and len(ty.args) == 1 and is_pytree_leaf(ty.args[0])


def _is_diff_wrapper(ty: A.TyNode) -> bool:
    return isinstance(ty, A.TyGeneric) and ty.base == "D" and len(ty.args) == 1


def _ty_to_prim_name(ty: A.TyNode) -> str:
    if isinstance(ty, A.TyName):
        return ty.name
    if _is_diff_wrapper(ty):
        return _ty_to_prim_name(ty.args[0])
    return repr(ty)


def _is_struct_ref(ty: A.TyNode, struct_decls: dict) -> bool:
    """True if `ty` resolves to a known StructDecl.

    Cycle-57 C57-3 fix: accept both `TyName(Pt)` and `TyGeneric(Pt,
    [i32])` shapes. Pre-fix only TyName was recognized, so any nested
    struct field whose type was a TyGeneric (e.g. `inner: Pt<i32>`
    before struct_mono rewrites it) tripped trap 26002 with a
    misleading "non-differentiable type" message. The real cause was
    walker drift — _is_struct_ref dropping a case that mirror passes
    (struct_mono.collect_concrete_uses) handle correctly. We resolve
    the TyGeneric via the same mangling scheme struct_mono uses, so
    pytree sees the monomorphized StructDecl regardless of pass
    ordering.
    """
    if isinstance(ty, A.TyName) and ty.name in struct_decls:
        return True
    if isinstance(ty, A.TyGeneric):
        # Local import to avoid a hard cycle: struct_mono imports from
        # monomorphize, monomorphize does not import pytree, but pytree
        # importing struct_mono at module load would create a future
        # circular-import hazard if struct_mono ever needs pytree
        # (it currently doesn't, but the local import is cheap insurance).
        from .struct_mono import mangle_struct
        return mangle_struct(ty.base, list(ty.args)) in struct_decls
    return False


def _resolve_struct_name(ty: A.TyNode) -> str:
    """Given a type known to satisfy _is_struct_ref, return the
    struct_decls key. Mirror of _is_struct_ref's resolution logic."""
    if isinstance(ty, A.TyName):
        return ty.name
    if isinstance(ty, A.TyGeneric):
        from .struct_mono import mangle_struct
        return mangle_struct(ty.base, list(ty.args))
    raise ValueError(f"pytree: _resolve_struct_name on non-struct {ty!r}")


def flatten_pytree(decl, struct_decls: dict,
                   path: str = "", depth: int = 0,
                   _visited: Optional[set] = None) -> list[PytreeLeaf]:
    """Walk a StructDecl recursively, yielding one PytreeLeaf per
    scalar field. Raises ValueError on:
      * depth > MAX_DEPTH (Phase-0 cap = 4) — trap 26001
      * leaf type not in DIFF_LEAF_PRIMS (trap 26002)
      * cyclic struct reference — trap 26003 (Audit 28.8 B9 (1))

    `decl` may be either a StructDecl or a TyName that resolves to one.
    `struct_decls` is name->StructDecl mapping.
    `_visited` is internal cycle-guard state (set of struct names
    currently on the recursion stack); callers pass nothing."""
    if _visited is None:
        _visited = set()
    if depth > MAX_DEPTH:
        raise ValueError(
            f"pytree depth > {MAX_DEPTH} at path {path!r} (trap 26001)"
        )
    if isinstance(decl, A.TyName):
        sd = struct_decls.get(decl.name)
        if sd is None:
            raise ValueError(f"pytree: unknown struct {decl.name!r}")
        decl = sd
    if not isinstance(decl, A.StructDecl):
        raise ValueError(f"pytree: expected StructDecl, got {type(decl).__name__}")

    if decl.name in _visited:
        raise ValueError(
            f"pytree: cyclic struct reference at path {path!r} "
            f"(struct {decl.name!r} reachable from itself) (trap 26003)"
        )
    _visited = _visited | {decl.name}

    leaves: list[PytreeLeaf] = []
    for f in decl.fields:
        fpath = f"{path}.{f.name}" if path else f.name
        if is_pytree_leaf(f.ty):
            leaves.append(PytreeLeaf(
                path=fpath,
                ty_name=_ty_to_prim_name(f.ty),
                is_diff=_is_diff_wrapper(f.ty),
            ))
        elif _is_struct_ref(f.ty, struct_decls):
            inner_decl = struct_decls[_resolve_struct_name(f.ty)]
            leaves.extend(
                flatten_pytree(inner_decl, struct_decls,
                               path=fpath, depth=depth + 1,
                               _visited=_visited)
            )
        else:
            # Non-leaf, non-struct: Phase-0 rejection (trap 26002).
            raise ValueError(
                f"pytree leaf at {fpath!r}: non-differentiable type "
                f"{_ty_to_prim_name(f.ty)} (trap 26002)"
            )
    return leaves


def flatten_pytree_param(param: A.FnParam,
                         struct_decls: dict) -> list[PytreeLeaf]:
    """Flatten one function parameter into AD-ready leaf paths.

    Scalar float parameters become a single leaf named after the parameter.
    Struct parameters become `param.field.subfield` leaves.
    """
    ty = param.ty
    if is_pytree_leaf(ty):
        return [PytreeLeaf(
            path=param.name,
            ty_name=_ty_to_prim_name(ty),
            is_diff=_is_diff_wrapper(ty),
        )]
    if _is_struct_ref(ty, struct_decls):
        decl = struct_decls[_resolve_struct_name(ty)]
        return [
            PytreeLeaf(
                path=f"{param.name}.{leaf.path}",
                ty_name=leaf.ty_name,
                is_diff=leaf.is_diff,
            )
            for leaf in flatten_pytree(decl, struct_decls)
        ]
    raise ValueError(
        f"pytree param {param.name!r}: non-differentiable type "
        f"{_ty_to_prim_name(ty)} (trap 26002)"
    )


def pytree_depth(decl, struct_decls: dict, depth: int = 0,
                 _visited: Optional[set] = None) -> int:
    """Maximum depth of nested struct fields inside `decl`. A flat
    struct (only scalar fields) has depth 0.

    Audit 28.8 B9 (1): cycle-guard via _visited so cyclic struct refs
    (e.g. `struct A { b: B }; struct B { a: A }`) don't blow Python's
    recursion stack. Returns depth as-of-current-frame on cycle (rather
    than raising) to keep the function usable as a query helper — only
    `flatten_pytree` raises 26003."""
    if _visited is None:
        _visited = set()
    if isinstance(decl, A.TyName):
        decl = struct_decls.get(decl.name)
        if decl is None:
            return depth
    if not isinstance(decl, A.StructDecl):
        return depth
    if decl.name in _visited:
        # Cycle detected — return the current depth as the bound
        # rather than recursing further. flatten_pytree raises 26003;
        # depth-query callers usually just want a bound.
        return depth
    _visited = _visited | {decl.name}
    max_d = depth
    for f in decl.fields:
        if _is_struct_ref(f.ty, struct_decls):
            inner = struct_decls[_resolve_struct_name(f.ty)]
            d = pytree_depth(inner, struct_decls, depth + 1, _visited)
            if d > max_d:
                max_d = d
    return max_d


def unflatten_pytree(decl, struct_decls: dict,
                     grads_by_path: dict,
                     default=_RAISE_ON_MISSING) -> dict:
    """Inverse of flatten_pytree: given path -> gradient mapping,
    build a nested dict with the same shape as `decl`.

    By default raises ValueError when a leaf path is missing from
    `grads_by_path` (Audit 28.8 A11). Pre-fix, missing paths silently
    defaulted to 0.0 — so a buggy AD pass with a path-name typo would
    silently produce zero gradients with no diagnostic. Pass
    `default=0.0` (or any value) to opt into permissive zero-fill
    behavior, matching the old API.

    Returns a dict for clarity in tests / introspection; the runtime
    would build an actual struct-typed value in registers."""
    if isinstance(decl, A.TyName):
        decl = struct_decls.get(decl.name)
        if decl is None:
            raise ValueError("unknown struct")
    if not isinstance(decl, A.StructDecl):
        raise ValueError("expected StructDecl")
    return _unflatten(decl, struct_decls, grads_by_path, prefix="",
                      default=default, _visited=set(), depth=0)


def tree_reduce(leaves_by_path: dict, reduce_fn, init):
    """Stage 59 follow-on / Tier 2 #7 polish — JAX-style tree_reduce.

    Reduce all leaf values to a single value via `reduce_fn(acc, leaf)`
    starting from `init`. Iterates leaves in sorted-by-path order for
    determinism (matters when reduce_fn is non-commutative).

    Use cases:
    - Compute gradient L1 norm: `tree_reduce(grads, lambda a, g: a + abs(g), 0.0)`
    - Count parameters: `tree_reduce(params, lambda a, _: a + 1, 0)`
    - Find max gradient: `tree_reduce(grads, max, float("-inf"))`
    - Check all positive: `tree_reduce(vals, lambda a, v: a and v > 0, True)`

    Pure functional — does not need the struct decl since values are
    already flat (keyed by path).
    """
    acc = init
    for path in sorted(leaves_by_path.keys()):
        acc = reduce_fn(acc, leaves_by_path[path])
    return acc


def tree_map(decl, struct_decls: dict, leaves_by_path: dict,
              leaf_fn, default=_RAISE_ON_MISSING) -> dict:
    """Stage 59 follow-on / Tier 2 #7 polish — JAX-style tree_map.

    Apply `leaf_fn(value)` to each leaf of the pytree shaped like
    `decl` whose values come from `leaves_by_path`. Returns a nested
    dict mirroring the struct's hierarchy with each leaf transformed.

    Use case: scale all gradients by learning rate, clip by norm,
    add weight decay, etc., without manually walking the tree.

    Example:
        # Multiply every weight in a Model gradient by 0.01.
        scaled = tree_map(Model_decl, struct_decls,
                          gradients_by_path, lambda g: g * 0.01)

    Composes flatten_pytree + unflatten_pytree machinery without
    needing the source struct value — the caller already has the
    leaves keyed by path (typical case: differentiate_reverse output).
    """
    mapped = {path: leaf_fn(v) for path, v in leaves_by_path.items()}
    return unflatten_pytree(decl, struct_decls, mapped, default=default)


def _unflatten(decl: A.StructDecl, struct_decls: dict,
               grads: dict, prefix: str, default,
               _visited: set, depth: int = 0) -> dict:
    # Audit 28.8 cycle 2 (deferred observation #17): mirror flatten's
    # depth-bound guard. Pre-fix `_unflatten` only had the cycle check
    # via `_visited`. A >MAX_DEPTH-deep struct WITHOUT a cycle (e.g. a
    # straight-line A -> B -> C -> D -> E -> F nesting) would
    # RecursionError instead of cleanly trapping. With this guard the
    # behavior is symmetric with `flatten_pytree`'s 26001 path.
    if depth > MAX_DEPTH:
        raise ValueError(
            f"pytree depth > {MAX_DEPTH} at prefix {prefix!r} "
            f"(struct {decl.name!r}) (trap 26001)"
        )
    if decl.name in _visited:
        raise ValueError(
            f"pytree: cyclic struct reference at prefix {prefix!r} "
            f"(struct {decl.name!r} reachable from itself) (trap 26003)"
        )
    _visited = _visited | {decl.name}
    out = {}
    for f in decl.fields:
        path = f"{prefix}.{f.name}" if prefix else f.name
        if is_pytree_leaf(f.ty):
            if path in grads:
                out[f.name] = grads[path]
            elif default is _RAISE_ON_MISSING:
                raise ValueError(
                    f"unflatten_pytree: leaf {path!r} missing from "
                    f"gradients (pass default=0.0 to opt into "
                    f"zero-fill)"
                )
            else:
                out[f.name] = default
        elif _is_struct_ref(f.ty, struct_decls):
            inner = struct_decls[_resolve_struct_name(f.ty)]
            out[f.name] = _unflatten(inner, struct_decls, grads, path,
                                     default, _visited, depth + 1)
        else:
            # Non-leaf, non-struct: mirror flatten's behavior — raise
            # rather than silently emitting None (Audit 28.8 B9 (2)).
            # The pre-fix path produced an asymmetric `unflatten(flatten(
            # x)) != x` contract: flatten raised on non-diff fields but
            # unflatten quietly defaulted them to None.
            raise ValueError(
                f"unflatten_pytree leaf at {path!r}: "
                f"non-differentiable type "
                f"{_ty_to_prim_name(f.ty)} (trap 26002)"
            )
    return out


def validate_pytree(decl, struct_decls: dict) -> list[str]:
    """Run flatten as a validation check; return diagnostic strings on
    failure, empty list on clean."""
    diags: list[str] = []
    try:
        flatten_pytree(decl, struct_decls)
    except ValueError as e:
        diags.append(str(e))
    return diags
