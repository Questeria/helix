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
  * unflatten_pytree(decl, gradients_by_path) -> dict
    given a path -> gradient mapping, reassembles a same-shape dict.
  * is_pytree_leaf(ty) — True for f64 / f32 / bf16 / TyDiff(scalar).
  * pytree_depth(decl, struct_decls) — recursion depth.
  * Trap 26001 — pytree depth > 4 (Phase-0 cap).
  * Trap 26002 — pytree leaf type not differentiable.

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


MAX_DEPTH = 4
DIFF_LEAF_PRIMS = frozenset({"f64", "f32", "bf16", "f16"})


@dataclass(frozen=True)
class PytreeLeaf:
    """One leaf in a pytree walk. `path` is dot-joined field names
    like 'layer1.w' for grandfather struct `Model` -> field `layer1`
    (struct) -> field `w` (f64)."""
    path: str
    ty_name: str            # "f64", "f32", etc.
    is_diff: bool = False   # True iff parameter type was D<...>


def is_pytree_leaf(ty: A.TyNode) -> bool:
    """A leaf is a primitive float type (optionally wrapped in D<>)."""
    if isinstance(ty, A.TyName):
        return ty.name in DIFF_LEAF_PRIMS
    if isinstance(ty, A.TyGeneric) and ty.base == "D" and len(ty.args) == 1:
        return is_pytree_leaf(ty.args[0])
    return False


def _is_diff_wrapper(ty: A.TyNode) -> bool:
    return isinstance(ty, A.TyGeneric) and ty.base == "D" and len(ty.args) == 1


def _ty_to_prim_name(ty: A.TyNode) -> str:
    if isinstance(ty, A.TyName):
        return ty.name
    if _is_diff_wrapper(ty):
        return _ty_to_prim_name(ty.args[0])
    return repr(ty)


def _is_struct_ref(ty: A.TyNode, struct_decls: dict) -> bool:
    if isinstance(ty, A.TyName) and ty.name in struct_decls:
        return True
    return False


def flatten_pytree(decl, struct_decls: dict,
                   path: str = "", depth: int = 0) -> list[PytreeLeaf]:
    """Walk a StructDecl recursively, yielding one PytreeLeaf per
    scalar field. Raises ValueError on:
      * depth > MAX_DEPTH (Phase-0 cap = 4) — trap 26001
      * leaf type not in DIFF_LEAF_PRIMS (trap 26002)

    `decl` may be either a StructDecl or a TyName that resolves to one.
    `struct_decls` is name->StructDecl mapping."""
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
            inner_decl = struct_decls[f.ty.name]
            leaves.extend(
                flatten_pytree(inner_decl, struct_decls,
                               path=fpath, depth=depth + 1)
            )
        else:
            # Non-leaf, non-struct: Phase-0 rejection (trap 26002).
            raise ValueError(
                f"pytree leaf at {fpath!r}: non-differentiable type "
                f"{_ty_to_prim_name(f.ty)} (trap 26002)"
            )
    return leaves


def pytree_depth(decl, struct_decls: dict, depth: int = 0) -> int:
    """Maximum depth of nested struct fields inside `decl`. A flat
    struct (only scalar fields) has depth 0."""
    if isinstance(decl, A.TyName):
        decl = struct_decls.get(decl.name)
        if decl is None:
            return depth
    if not isinstance(decl, A.StructDecl):
        return depth
    max_d = depth
    for f in decl.fields:
        if _is_struct_ref(f.ty, struct_decls):
            inner = struct_decls[f.ty.name]
            d = pytree_depth(inner, struct_decls, depth + 1)
            if d > max_d:
                max_d = d
    return max_d


def unflatten_pytree(decl, struct_decls: dict,
                     grads_by_path: dict) -> dict:
    """Inverse of flatten_pytree: given path -> gradient mapping,
    build a nested dict with the same shape as `decl`. Missing paths
    default to 0.0.

    Returns a dict for clarity in tests / introspection; the runtime
    would build an actual struct-typed value in registers."""
    if isinstance(decl, A.TyName):
        decl = struct_decls.get(decl.name)
        if decl is None:
            raise ValueError("unknown struct")
    if not isinstance(decl, A.StructDecl):
        raise ValueError("expected StructDecl")
    return _unflatten(decl, struct_decls, grads_by_path, prefix="")


def _unflatten(decl: A.StructDecl, struct_decls: dict,
               grads: dict, prefix: str) -> dict:
    out = {}
    for f in decl.fields:
        path = f"{prefix}.{f.name}" if prefix else f.name
        if is_pytree_leaf(f.ty):
            out[f.name] = grads.get(path, 0.0)
        elif _is_struct_ref(f.ty, struct_decls):
            inner = struct_decls[f.ty.name]
            out[f.name] = _unflatten(inner, struct_decls, grads, path)
        else:
            out[f.name] = None  # non-diff field, no gradient
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
