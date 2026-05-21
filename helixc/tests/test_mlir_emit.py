"""Tests for helixc.ir.mlir.emit — v3.0 Phase E, Stage 212 chunk A:
the Helix-IR -> MLIR type bridge.

`emit.py` opens Stage 212 — the parallel MLIR translation path. Chunk A
is `render_mlir_type`: every Helix IR type (`tir.TIRType`) rendered as
MLIR type syntax — scalars to `i32` / `f32` / `i1`, tensors to
`tensor<...>`, tiles to `vector<...>`, tuples to `tuple<...>`, unit to
`none`. The translator FAILS CLOSED (`MLIRTranslationError`) on any
type it cannot faithfully render rather than emit a guessed type.

These tests pin: the scalar-dtype mapping; the fail-closed behaviour on
`char` / the quantized dtypes / unknown types / non-static tile dims;
shape-dimension rendering (static and dynamic); tensor / tile / tuple /
unit rendering; the two module-load coverage guards and their
non-vacuity; and — the mock-path rule — that the module never
`import mlir`.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from helixc.ir import tir
from helixc.ir.mlir import emit
from helixc.ir.mlir.emit import (
    MLIRTranslationError, render_dim, render_mlir_type,
)

_S = tir.TIRScalar


# --------------------------------------------------------------------------
# MLIRTranslationError
# --------------------------------------------------------------------------
def test_mlir_translation_error_is_exception():
    """`MLIRTranslationError` is a plain `Exception` subclass — the
    fail-closed signal of the translator."""
    assert issubclass(MLIRTranslationError, Exception)


# --------------------------------------------------------------------------
# scalar types
# --------------------------------------------------------------------------
def test_render_scalar_types():
    """Helix scalar dtypes render to their MLIR spelling — integers are
    signless (`u32` and `i32` both `i32`), `isize`/`usize` are 64-bit,
    floats map 1:1, `bool` is `i1`."""
    cases = {
        "bool": "i1", "i8": "i8", "u8": "i8", "i16": "i16",
        "u16": "i16", "i32": "i32", "u32": "i32", "i64": "i64",
        "u64": "i64", "isize": "i64", "usize": "i64",
        "f16": "f16", "bf16": "bf16", "f32": "f32", "f64": "f64",
    }
    for dtype, mlir in cases.items():
        assert render_mlir_type(_S(dtype)) == mlir, dtype


def test_render_scalar_char_fails_closed():
    """`char` has no MLIR type — its bit width is not pinned (the LLVM
    path defers it the same way) — so it fails closed."""
    with pytest.raises(MLIRTranslationError, match="char"):
        render_mlir_type(_S("char"))


def test_render_scalar_quantized_fails_closed():
    """The front-end-only quantized dtypes have no backend codegen and
    no MLIR type — each fails closed, never a guessed width."""
    for dtype in ("fp8", "mxfp4", "nvfp4", "ternary"):
        with pytest.raises(MLIRTranslationError, match="no MLIR type"):
            render_mlir_type(_S(dtype))


def test_render_scalar_unknown_fails_closed():
    """An unrecognised dtype name fails closed."""
    with pytest.raises(MLIRTranslationError):
        render_mlir_type(_S("not_a_dtype"))


# --------------------------------------------------------------------------
# shape dimensions
# --------------------------------------------------------------------------
def test_render_dim():
    """A constant dimension renders as its integer; a dynamic,
    symbolic, or computed dimension renders as MLIR's dynamic `?`."""
    assert render_dim(tir.DimConst(4)) == "4"
    assert render_dim(tir.DimConst(0)) == "0"
    assert render_dim(tir.DimDyn()) == "?"
    assert render_dim(tir.DimVar("N")) == "?"
    assert render_dim(tir.DimExpr("+", (tir.DimConst(1),
                                       tir.DimVar("N")))) == "?"


def test_render_dim_negative_fails_closed():
    """A negative constant dimension is not a valid MLIR extent —
    fails closed."""
    with pytest.raises(MLIRTranslationError, match="negative"):
        render_dim(tir.DimConst(-1))


# --------------------------------------------------------------------------
# tensor / tile / tuple / unit types
# --------------------------------------------------------------------------
def test_render_tensor_type():
    """`TIRTensorTy` renders to `tensor<dims x dtype>` — a 0-d tensor
    has no dims, a non-constant size becomes a dynamic `?`."""
    assert render_mlir_type(
        tir.TIRTensorTy(_S("f32"),
                        (tir.DimConst(2), tir.DimConst(3)))
    ) == "tensor<2x3xf32>"
    assert render_mlir_type(
        tir.TIRTensorTy(_S("f32"), ())) == "tensor<f32>"
    assert render_mlir_type(
        tir.TIRTensorTy(_S("i32"), (tir.DimDyn(), tir.DimConst(4)))
    ) == "tensor<?x4xi32>"


def test_render_tensor_non_default_layout_fails_closed():
    """A non-default tensor layout (COL_MAJOR / BLOCKED) is
    correctness-relevant and has no slot in MLIR's builtin `tensor`
    type — the translator fails closed rather than silently erase it
    into a default row-major spelling."""
    for layout in (tir.Layout.COL_MAJOR, tir.Layout.BLOCKED):
        ty = tir.TIRTensorTy(_S("f32"), (tir.DimConst(2),),
                             layout=layout)
        with pytest.raises(MLIRTranslationError,
                           match="non-default layout"):
            render_mlir_type(ty)


def test_render_tensor_non_default_device_fails_closed():
    """A non-default tensor device likewise fails closed — it has no
    builtin-`tensor`-type slot and must not be silently erased."""
    ty = tir.TIRTensorTy(_S("f32"), (tir.DimConst(2),), device="cuda")
    with pytest.raises(MLIRTranslationError, match="non-default device"):
        render_mlir_type(ty)


def test_render_tile_type():
    """`TIRTileTy` renders to `vector<dims x dtype>` — MLIR's tile /
    SIMD type. `memspace` is not part of the spelling."""
    assert render_mlir_type(
        tir.TIRTileTy(_S("f32"),
                      (tir.DimConst(8), tir.DimConst(8)), "reg")
    ) == "vector<8x8xf32>"
    assert render_mlir_type(
        tir.TIRTileTy(_S("i32"), (tir.DimConst(16),), "smem")
    ) == "vector<16xi32>"


def test_render_tile_non_static_fails_closed():
    """An MLIR `vector` requires static dimensions — a tile with a
    dynamic / symbolic extent fails closed."""
    with pytest.raises(MLIRTranslationError, match="non-constant"):
        render_mlir_type(
            tir.TIRTileTy(_S("f32"), (tir.DimDyn(),), "reg"))


def test_render_tile_zero_d_fails_closed():
    """A 0-dimensional tile has no MLIR `vector` rendering — fails
    closed (an MLIR vector has at least one dimension)."""
    with pytest.raises(MLIRTranslationError, match="0-dimensional"):
        render_mlir_type(tir.TIRTileTy(_S("f32"), (), "reg"))


def test_render_tuple_type():
    """`TIRTuple` renders to `tuple<...>`, recursively over its
    elements."""
    assert render_mlir_type(
        tir.TIRTuple((_S("i32"), _S("f32")))) == "tuple<i32, f32>"
    # nested — `bool` is the Helix dtype name; it renders to MLIR `i1`
    assert render_mlir_type(
        tir.TIRTuple((_S("bool"),
                      tir.TIRTensorTy(_S("f32"), (tir.DimConst(2),))))
    ) == "tuple<i1, tensor<2xf32>>"


def test_render_empty_tuple_fails_closed():
    """An empty tuple has no MLIR `tuple` spelling — fails closed (a
    no-value result is `TIRUnit`, not an empty tuple)."""
    with pytest.raises(MLIRTranslationError, match="empty TIRTuple"):
        render_mlir_type(tir.TIRTuple(()))


def test_render_unit_type():
    """`TIRUnit` renders to MLIR's builtin unit type `none`."""
    assert render_mlir_type(tir.TIRUnit()) == "none"


def test_render_unknown_tir_type_fails_closed():
    """An object that is not a known `tir.TIRType` subclass fails
    closed — the translator never guesses a rendering."""
    with pytest.raises(MLIRTranslationError, match="unknown type"):
        render_mlir_type(object())          # type: ignore[arg-type]


# --------------------------------------------------------------------------
# module-load coverage guards
# --------------------------------------------------------------------------
def test_tir_type_coverage_guard():
    """`_check_tir_type_coverage` is callable and passes — `_TYPE_
    RENDERERS` covers exactly the concrete `tir.TIRType` subclasses."""
    emit._check_tir_type_coverage()  # must not raise
    assert {t.__name__ for t in emit._TYPE_RENDERERS} == {
        t.__name__ for t in tir.TIRType.__subclasses__()}


def test_dim_coverage_guard():
    """`_check_dim_coverage` is callable and passes — `_DIM_RENDERERS`
    covers exactly the concrete `tir.Dim` subclasses."""
    emit._check_dim_coverage()  # must not raise
    assert {t.__name__ for t in emit._DIM_RENDERERS} == {
        t.__name__ for t in tir.Dim.__subclasses__()}


def test_coverage_guards_are_not_vacuous(monkeypatch):
    """Each coverage guard genuinely catches an uncovered type — drop a
    renderer and confirm the guard raises, naming the now-unhandled
    class. A guard that always passed would not protect against drift."""
    short_types = dict(emit._TYPE_RENDERERS)
    del short_types[tir.TIRUnit]
    monkeypatch.setattr(emit, "_TYPE_RENDERERS", short_types)
    with pytest.raises(AssertionError, match=r"unhandled.*TIRUnit"):
        emit._check_tir_type_coverage()

    short_dims = dict(emit._DIM_RENDERERS)
    del short_dims[tir.DimExpr]
    monkeypatch.setattr(emit, "_DIM_RENDERERS", short_dims)
    with pytest.raises(AssertionError, match=r"unhandled.*DimExpr"):
        emit._check_dim_coverage()


# --------------------------------------------------------------------------
# the mock-path rule — emit is pure text, never `import mlir`
# --------------------------------------------------------------------------
def test_emit_module_is_pure_text_no_mlir_import():
    """THE MOCK-PATH RULE (Stage 210 decision, section 3): `emit` is a
    pure-text translator — it NEVER `import mlir`, at module top level
    or anywhere. Parse the module's AST and confirm not one
    `import mlir` / `from mlir ...` statement."""
    tree = ast.parse(Path(emit.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                assert not a.name.startswith("mlir"), a.name
        elif isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("mlir"), node.module
